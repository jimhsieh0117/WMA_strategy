"""Live paper trading 主入口。

啟動：
    .venv/bin/python -m scripts.live_sim [--config configs/live.yaml]
                                          [--port 8060]
                                          [--warmup-bars 96]
                                          [--poll-interval 60]

執行流程：
    1. 讀 config (configs/live.yaml)
    2. 建立 long / short Account / Strategy
    3. 啟動 FastAPI server（含 WebSocket）→ 瀏覽器訪問 http://localhost:8060
    4. 在 lifespan 內：
       a. LiveCandleFeed.warmup()：拉 N 根歷史 K 線 → prepare_indicators →
          標記 is_warmed_up=True（暖機完成才開始模擬交易）
       b. 啟動 feed.run() background task：每 60s poll Binance，新 closed K 觸發
          LiveSimulator.on_new_closed_bar
    5. Trades / equity / config snapshot 寫到 results/live/{YYYYMMDD_HHMMSS}/

不下實單。完全用 BrokerSimulator 模擬撮合，符合 CLAUDE.md §二.5 規範。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import (  # noqa: E402
    _build_strategy, build_broker, build_engine_config, build_strategy_params,
)
from src.broker.account import Account  # noqa: E402
from src.live.feed import LiveCandleFeed  # noqa: E402
from src.live.server import build_app  # noqa: E402
from src.live.simulator import LiveSimulator  # noqa: E402
from src.live.state import LiveState  # noqa: E402
from src.utils.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)


# ccxt symbol 格式（BTCUSDT → BTC/USDT）
def _to_ccxt_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    raise ValueError(f"cannot derive ccxt symbol from '{symbol}'")


async def _amain(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # 輸出資料夾
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / "live" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out_dir.resolve())

    # 寫 config snapshot
    snapshot = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "symbol": cfg.symbol,
        "timeframe": cfg.timeframe,
        "warmup_bars": args.warmup_bars,
        "poll_interval_sec": args.poll_interval,
        "config_path": str(Path(args.config).resolve()),
    }
    (out_dir / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # 兩個 strategy params / engine_config / broker
    long_params = build_strategy_params(cfg, "long")
    short_params = build_strategy_params(cfg, "short")
    engine_cfg = build_engine_config(cfg)
    broker = build_broker(cfg)

    long_account = Account(cfg.initial_capital, name="long_live")
    short_account = Account(cfg.initial_capital, name="short_live")
    long_strategy = _build_strategy("long", long_params)
    short_strategy = _build_strategy("short", short_params)

    state = LiveState(
        cfg=cfg,
        long_account=long_account,
        short_account=short_account,
        long_strategy=long_strategy,
        short_strategy=short_strategy,
        df_long=pd.DataFrame(),                                   # warmup 後填入
        df_short=pd.DataFrame(),
    )

    simulator = LiveSimulator(
        state=state, engine_config=engine_cfg, broker=broker, output_dir=out_dir,
    )

    app, server = build_app(state)

    # 把 server.broadcast 註冊為 simulator event handler
    simulator.register_async_event_handler(server.broadcast)

    feed = LiveCandleFeed(
        symbol=_to_ccxt_symbol(cfg.symbol),
        timeframe=cfg.timeframe,
        warmup_bars=args.warmup_bars,
        poll_interval_sec=args.poll_interval,
        on_new_closed_bar=simulator.on_new_closed_bar,
        on_warmup_loaded=simulator.on_warmup_loaded,
    )

    # FastAPI lifespan：暖機 + 啟動 feed background task
    feed_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(app):
        nonlocal feed_task
        logger.info("=" * 70)
        logger.info("  WMA Live Paper Trading")
        logger.info("  Symbol: %s  Timeframe: %s", cfg.symbol, cfg.timeframe)
        logger.info("  Long WMA: %d/%d  Short WMA: %d/%d",
                    cfg.wma_long_fast, cfg.wma_long_slow,
                    cfg.wma_short_fast, cfg.wma_short_slow)
        logger.info("  Warmup: %d bars  Poll: %ds", args.warmup_bars, args.poll_interval)
        logger.info("  Output: %s", out_dir.resolve())
        logger.info("  http://localhost:%d", args.port)
        logger.info("=" * 70)
        try:
            await feed.warmup()
        except Exception:
            logger.exception("warmup failed")
            raise
        feed_task = asyncio.create_task(feed.run(), name="live-feed")
        try:
            yield
        finally:
            logger.info("shutting down live feed ...")
            feed.stop()
            if feed_task is not None:
                feed_task.cancel()
                try:
                    await feed_task
                except (asyncio.CancelledError, Exception):
                    pass

    app.router.lifespan_context = lifespan

    # uvicorn 起 server（在當前 event loop 跑）
    server_cfg = uvicorn.Config(
        app, host=args.host, port=args.port, log_level="info",
    )
    uv_server = uvicorn.Server(server_cfg)
    await uv_server.serve()


def main() -> None:
    parser = argparse.ArgumentParser(description="WMA live paper trading")
    parser.add_argument("--config", default="configs/live.yaml")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--warmup-bars", type=int, default=96,
        help="啟動時抓 N 根歷史 K 線暖機（15m × 96 = 24h）",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Binance K 線 poll 間隔（秒）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        logger.info("interrupted by user")


if __name__ == "__main__":
    main()
