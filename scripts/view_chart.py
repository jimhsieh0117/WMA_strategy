"""啟動互動式 K 線回測檢視器。

用法：
    .venv/bin/python scripts/view_chart.py [--sample is|oos] [--port 8050]
                                            [--panels bollinger wma volume wavetrend]
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn  # noqa: E402

from scripts._runner import run_single_strategy  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.strategy.base import prepare_indicators  # noqa: E402
from src.strategy.types import StrategyParams, TrailingStopParams  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.viewer.indicators import REGISTRY, default_panels  # noqa: E402
from src.viewer.server import build_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="互動式 K 線回測檢視器（FastAPI + LWC）"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--panels", nargs="*", default=None,
        help=(
            "指標**初始勾選**清單；server 永遠載入全部指標，"
            f"使用者可在網頁 UI 切換。"
            f"可選: {sorted(REGISTRY)}"
        ),
    )
    parser.add_argument("--no-open", action="store_true",
                        help="不要自動開瀏覽器")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    initial_enabled = args.panels or default_panels()
    for name in initial_enabled:
        if name not in REGISTRY:
            raise SystemExit(
                f"unknown panel '{name}'. available: {sorted(REGISTRY)}"
            )

    # ---- 跑兩支策略 ----
    print("[1/4] 跑回測（多）...")
    long_result = run_single_strategy(cfg, direction="long", sample=args.sample)
    print("[2/4] 跑回測（空）...")
    short_result = run_single_strategy(cfg, direction="short", sample=args.sample)
    combined_result = build_merged_result(
        f"combined_{cfg.timeframe}_{args.sample}",
        [long_result, short_result],
    )

    # ---- 重新載資料 + 算指標（含 panels 要求的） ----
    print("[3/4] 載資料 + 計算指標...")
    period = cfg.in_sample if args.sample == "is" else cfg.out_of_sample
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m

    trailing = TrailingStopParams(
        swing_lookback=cfg.trailing.swing_lookback,
        stage1_slippage_buffer=cfg.trailing.stage1_slippage_buffer,
        stage2_normal_trigger_r=cfg.trailing.stage2_normal_trigger_r,
        stage2_abnormal_trigger_r=cfg.trailing.stage2_abnormal_trigger_r,
        stage2_buffer_r=cfg.trailing.stage2_buffer_r,
        stage3_normal_trigger_r=cfg.trailing.stage3_normal_trigger_r,
        stage3_abnormal_trigger_r=cfg.trailing.stage3_abnormal_trigger_r,
        bollinger_period=cfg.trailing.bollinger_period,
        bollinger_num_std=cfg.trailing.bollinger_num_std,
    )
    # 用 long 的 WMA 算 base 指標欄位（含 BB / chop / structure）
    long_params = StrategyParams(
        wma_fast=cfg.wma_long_fast,
        wma_slow=cfg.wma_long_slow,
        trailing=trailing,
    )
    augmented = prepare_indicators(df, long_params)
    augmented = augmented.rename(columns={
        "wma_fast": "wma_long_fast",
        "wma_slow": "wma_long_slow",
    })
    # 再用 short 的 WMA 算一組，只保留 wma_fast/wma_slow 欄並 rename
    short_params = StrategyParams(
        wma_fast=cfg.wma_short_fast,
        wma_slow=cfg.wma_short_slow,
        trailing=trailing,
    )
    short_aug = prepare_indicators(df, short_params)
    augmented["wma_short_fast"] = short_aug["wma_fast"]
    augmented["wma_short_slow"] = short_aug["wma_slow"]

    # 永遠把 REGISTRY 內所有指標都算好；前端再依 enabled 切換顯隱
    all_indicators = list(REGISTRY.values())
    for reg in all_indicators:
        augmented = reg.compute(augmented)

    # ---- 啟動 server ----
    print("[4/4] 啟動 FastAPI...")
    app = build_app(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        sample=args.sample,
        df_main=augmented,
        indicators=all_indicators,
        initial_enabled=initial_enabled,
        long_result=long_result,
        short_result=short_result,
        combined_result=combined_result,
    )

    url = f"http://{args.host}:{args.port}/"
    print()
    print("=" * 60)
    print(f"  📊 圖表服務啟動於：{url}")
    print(f"     初始勾選：{initial_enabled}")
    print(f"     全部指標：{sorted(REGISTRY)}（可於網頁切換）")
    print(f"     按 Ctrl+C 停止伺服器")
    print("=" * 60)
    print()

    if not args.no_open:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("\n伺服器已停止")


if __name__ == "__main__":
    main()
