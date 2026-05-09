"""Live simulation runner for WMA trend strategy using ccxt OHLCV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import ccxt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.types import BacktestResult, EngineConfig  # noqa: E402
from src.broker.account import Account  # noqa: E402
from src.broker.simulator import BrokerSimulator  # noqa: E402
from src.broker.types import Bar, BrokerConfig, Trade  # noqa: E402
from src.metrics.calculator import MetricsReport, compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.reporting.exporter import (  # noqa: E402
    export_config_snapshot,
    export_metrics_json,
    export_summary_text,
    export_trades_csv,
)
from src.reporting.plotter import plot_drawdown, plot_equity_curves  # noqa: E402
from src.strategy.base import BaseTrendStrategy, prepare_indicators  # noqa: E402
from src.strategy.long_strategy import LongTrendStrategy  # noqa: E402
from src.strategy.short_strategy import ShortTrendStrategy  # noqa: E402
from src.strategy.trailing import TrailingStopController  # noqa: E402
from src.strategy.types import EntrySignal, StrategyParams, TrailingStopParams  # noqa: E402
from src.utils.config import FullConfig, load_config  # noqa: E402
from src.utils.exceptions import ConfigError, DataIntegrityError, OrderExecutionError  # noqa: E402
from src.utils.types import Direction  # noqa: E402
from src.utils.validation import validate_ohlc  # noqa: E402

logger = logging.getLogger(__name__)


# 過濾未收盤 K 線時的安全 buffer（毫秒），抵消本地鐘相對 binance 的偏差
SAFE_BUFFER_MS = 1500

# 連續 fetch 失敗多少次後 raise（避免 binance 維護期間無限重試而沒人發現）
MAX_CONSECUTIVE_FAILURES = 30

# 指數 backoff 上限（秒）
MAX_BACKOFF_SECONDS = 300

# state.json schema 版本
STATE_SCHEMA_VERSION = 1


EventType = Literal["ENTRY", "EXIT", "STOP_UPDATE"]


@dataclass(frozen=True)
class PositionEvent:
    timestamp: pd.Timestamp
    position_id: int
    direction: Direction
    event: EventType
    price: float
    quantity: float
    reason: str
    stop_price: float | None = None


@dataclass(frozen=True)
class LiveBarResult:
    closed_trades: list[Trade]
    events: list[PositionEvent]


@dataclass
class LiveEngine:
    name: str
    strategy: BaseTrendStrategy
    account: Account
    broker: BrokerSimulator
    config: EngineConfig
    pending_signal: EntrySignal | None = None
    trailings: dict[int, TrailingStopController] = field(default_factory=dict)
    signals_emitted: int = 0
    signals_filled: int = 0
    signals_unfilled: int = 0
    signals_skipped_pending: int = 0
    bars_processed: int = 0

    def process_bar(self, df: pd.DataFrame, bar_index: int) -> LiveBarResult:
        """處理單根已收盤 K 線，回傳平倉交易與進出場事件。"""
        ts = df.index[bar_index]
        row = df.iloc[bar_index]
        bar = Bar.from_row(ts, row)
        closed_trades: list[Trade] = []
        events: list[PositionEvent] = []

        # Step 1: 撮合 pending 限價單（在本根開盤）
        if self.pending_signal is not None:
            limit_price = _compute_limit_price(bar.open, self.pending_signal.direction, self.broker.config.slippage_pct)
            equity_now = self.account.equity(bar.open)
            if equity_now > 0 and limit_price > 0:
                quantity, target_risk, sizing_ok, sizing_reason = _compute_quantity(
                    config=self.config,
                    direction=self.pending_signal.direction,
                    limit_price=limit_price,
                    initial_stop=self.pending_signal.initial_stop,
                    equity_now=equity_now,
                    taker_fee_rate=self.broker.config.taker_fee_rate,
                    existing_notional=self.account.total_notional_at_entry,
                )
                stop_ok = _stop_on_correct_side(
                    self.pending_signal.direction,
                    self.pending_signal.initial_stop,
                    limit_price,
                )
                if sizing_ok and quantity > 0 and stop_ok:
                    order = _build_limit_order(
                        direction=self.pending_signal.direction,
                        limit_price=limit_price,
                        quantity=quantity,
                        initial_stop=self.pending_signal.initial_stop,
                        target_risk_usdt=target_risk,
                    )
                    try:
                        result = self.broker.try_fill_limit(
                            order,
                            bar,
                            self.account,
                            allow_multi=self.config.allow_pyramiding,
                        )
                    except OrderExecutionError:
                        logger.exception("[%s] fill failed at %s", self.name, ts)
                        raise
                    if result.filled:
                        self.signals_filled += 1
                        pid = result.position_id
                        if pid is None:
                            raise DataIntegrityError("filled order missing position_id")
                        if result.fill_price is None:
                            raise DataIntegrityError("filled order missing fill_price")
                        new_pos = self.account.position_by_id(pid)
                        controller = TrailingStopController(
                            position=new_pos,
                            params=self.strategy.params.trailing,
                            broker_config=self.broker.config,
                        )
                        self.trailings[pid] = controller
                        events.append(PositionEvent(
                            timestamp=bar.timestamp,
                            position_id=pid,
                            direction=new_pos.direction,
                            event="ENTRY",
                            price=float(result.fill_price),
                            quantity=new_pos.quantity,
                            reason=self.pending_signal.reason,
                            stop_price=new_pos.stop_price,
                        ))
                        logger.debug(
                            "[%s] FILLED %s @ %.4f qty=%.6f stop=%.4f pid=%d",
                            self.name,
                            self.pending_signal.direction,
                            result.fill_price,
                            quantity,
                            self.pending_signal.initial_stop,
                            pid,
                        )
                    else:
                        self.signals_unfilled += 1
                        logger.debug("[%s] UNFILLED %s: %s", self.name, self.pending_signal.direction, result.reason)
                else:
                    self.signals_unfilled += 1
                    if not sizing_ok:
                        logger.debug(
                            "[%s] SKIP_FILL %s: sizing rejected (%s)",
                            self.name,
                            self.pending_signal.direction,
                            sizing_reason,
                        )
                    else:
                        logger.debug(
                            "[%s] SKIP_FILL %s: stop %.4f wrong side of limit %.4f",
                            self.name,
                            self.pending_signal.direction,
                            self.pending_signal.initial_stop,
                            limit_price,
                        )
            else:
                self.signals_unfilled += 1
            self.pending_signal = None

        # Step 2: 盤中止損檢查
        if self.account.has_position():
            closed_trades = self.broker.check_stops(self.account, bar)
            for trade in closed_trades:
                self.trailings.pop(trade.position_id, None)
                events.append(PositionEvent(
                    timestamp=trade.exit_timestamp,
                    position_id=trade.position_id,
                    direction=trade.direction,
                    event="EXIT",
                    price=trade.exit_price,
                    quantity=trade.quantity,
                    reason=trade.exit_reason,
                    stop_price=_last_stop_price(trade),
                ))
                logger.debug(
                    "[%s] STOP %s @ %.4f net_pnl=%.4f reason=%s pid=%d",
                    self.name,
                    trade.direction,
                    trade.exit_price,
                    trade.net_pnl,
                    trade.exit_reason,
                    trade.position_id,
                )

        # Step 3a: 收盤 ratchet 拖曳止損
        for pid, pos in self.account.positions.items():
            controller = self.trailings.get(pid)
            if controller is None:
                continue
            new_stop = controller.update(
                bar=bar,
                df=df,
                bar_index=bar_index,
                current_stop=pos.stop_price,
            )
            if new_stop is not None:
                old_stop = pos.stop_price
                self.account.update_stop_by_id(pid, new_stop, timestamp=ts)
                events.append(PositionEvent(
                    timestamp=ts,
                    position_id=pid,
                    direction=pos.direction,
                    event="STOP_UPDATE",
                    price=float(new_stop),
                    quantity=pos.quantity,
                    reason=f"stage={controller.stage}",
                    stop_price=float(new_stop),
                ))
                logger.debug(
                    "[%s] RATCHET %s pid=%d stop %.4f -> %.4f (stage=%d)",
                    self.name,
                    pos.direction,
                    pid,
                    old_stop,
                    new_stop,
                    controller.stage,
                )

        # Step 3b: 收盤偵測進場訊號
        can_detect = self.config.allow_pyramiding or not self.account.has_position()
        if can_detect:
            new_signal = self.strategy.detect_entry(df, bar_index)
            if new_signal is not None:
                self.signals_emitted += 1
                if self.pending_signal is not None and self.config.skip_signal_when_pending:
                    self.signals_skipped_pending += 1
                    logger.debug("[%s] SKIP_SIGNAL %s: pending order exists", self.name, new_signal.direction)
                else:
                    self.pending_signal = new_signal

        # Step 4: 記錄權益
        self.account.snapshot_equity(bar.close, ts)
        self.bars_processed += 1
        return LiveBarResult(closed_trades=closed_trades, events=events)

    def force_close(self, last_bar: Bar) -> None:
        if not self.account.has_position():
            return
        for pid, pos in list(self.account.positions.items()):
            notional = pos.quantity * last_bar.close
            fee = notional * self.broker.config.taker_fee_rate
            self.account.close_position_by_id(
                position_id=pid,
                exit_price=last_bar.close,
                exit_timestamp=last_bar.timestamp,
                fee=fee,
                reason="FORCE_CLOSE_END",
            )
            self.trailings.pop(pid, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live simulation for WMA trend strategy (Binance USDT-M OHLCV)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--warmup-extra", type=int, default=20)
    parser.add_argument(
        "--resume",
        default=None,
        help="run_tag 或 run_dir 路徑；指定後自動還原 state.json + klines.csv 繼續跑",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    base_dir = Path("live_sim")
    data_dir = base_dir / "data"
    results_root = base_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)

    exchange = _build_exchange()
    symbol = _resolve_symbol(cfg.symbol, exchange)
    timeframe = _normalize_timeframe(cfg.timeframe)
    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)

    params = _build_params(cfg)
    warmup_bars = params.warmup_bars
    warmup_limit = warmup_bars + max(args.warmup_extra, 0)

    broker = BrokerSimulator(BrokerConfig(
        taker_fee_rate=cfg.taker_fee_rate,
        maker_fee_rate=cfg.maker_fee_rate,
        slippage_pct=cfg.slippage_pct,
    ))
    engine_cfg = EngineConfig(
        sizing_mode=cfg.sizing_mode,  # type: ignore[arg-type]
        position_size_pct=cfg.position_size_pct,
        risk_per_trade_usdt=cfg.risk_per_trade_usdt,
        allow_pyramiding=cfg.allow_pyramiding,
        leverage_cap=cfg.leverage_cap,
        force_close_at_end=cfg.force_close_at_end,
    )

    if args.resume:
        run_tag, run_dir, kline_path, state_path = _resolve_resume_paths(
            args.resume, results_root, data_dir
        )
        logger.info("Live sim 還原：run_tag=%s", run_tag)

        raw_df = _load_klines_csv(kline_path)
        validate_ohlc(raw_df, require_volume=True)
        _assert_continuous(raw_df, timeframe_ms, where="resumed-csv")

        # 補拉斷線期間遺漏的 K（resume 後 last_ts 到 now 之間可能有缺）
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if int(state.get("version", 0)) != STATE_SCHEMA_VERSION:
            raise DataIntegrityError(
                f"state.json schema 不相容：need {STATE_SCHEMA_VERSION}, "
                f"got {state.get('version')}"
            )
        last_ts = pd.Timestamp(state["last_ts"])
        now_ms = exchange.milliseconds()
        catchup = _fetch_new_bars(
            exchange, symbol, timeframe, last_ts, timeframe_ms, now_ms
        )
        if not catchup.empty:
            logger.info("補拉 %d 根斷線期間的 K 線", len(catchup))
            raw_df = _merge_new_bars(raw_df, catchup)
            _assert_continuous(raw_df, timeframe_ms, where="resumed-merged")
            _append_klines_csv(kline_path, catchup, write_header=False)

        df = prepare_indicators(raw_df, params)

        long_engine = _restore_engine(
            name="long",
            state=state["long"],
            strategy=LongTrendStrategy(params),
            broker=broker,
            config=engine_cfg,
            df=df,
        )
        short_engine = _restore_engine(
            name="short",
            state=state["short"],
            strategy=ShortTrendStrategy(params),
            broker=broker,
            config=engine_cfg,
            df=df,
        )

        trades_stream_path = run_dir / "trades_live.csv"
        equity_stream_path = run_dir / "equity_curve_live.csv"
        position_events_path = run_dir / "position_events.csv"

        # 處理斷線期間補拉到的 K（不重新 prime 最後一根，那根已在 state 裡處理過）
        if not catchup.empty:
            _process_batch(
                df, list(catchup.index),
                [long_engine, short_engine],
                trades_stream_path, equity_stream_path, position_events_path,
            )
        last_ts = df.index[-1]
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_tag = f"{cfg.symbol}_{cfg.timeframe}_live_{run_id}"
        run_dir = results_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        kline_path = data_dir / f"{run_tag}_klines.csv"
        trades_stream_path = run_dir / "trades_live.csv"
        equity_stream_path = run_dir / "equity_curve_live.csv"
        position_events_path = run_dir / "position_events.csv"
        state_path = run_dir / "state.json"

        logger.info("Live sim 啟動：symbol=%s timeframe=%s warmup=%d",
                    symbol, timeframe, warmup_limit)

        raw_df = _fetch_recent_bars(
            exchange, symbol, timeframe, warmup_limit, timeframe_ms
        )
        if len(raw_df) < warmup_bars:
            raise DataIntegrityError(
                f"warmup bars insufficient: need >= {warmup_bars}, got {len(raw_df)}"
            )
        validate_ohlc(raw_df, require_volume=True)
        _append_klines_csv(kline_path, raw_df, write_header=True)

        df = prepare_indicators(raw_df, params)

        long_engine = LiveEngine(
            name="long",
            strategy=LongTrendStrategy(params),
            account=Account(cfg.initial_capital, name=f"long_{cfg.timeframe}_live"),
            broker=broker,
            config=engine_cfg,
        )
        short_engine = LiveEngine(
            name="short",
            strategy=ShortTrendStrategy(params),
            account=Account(cfg.initial_capital, name=f"short_{cfg.timeframe}_live"),
            broker=broker,
            config=engine_cfg,
        )

        # Prime with the latest closed bar to seed pending signals and equity history
        _process_batch(
            df, [df.index[-1]],
            [long_engine, short_engine],
            trades_stream_path, equity_stream_path, position_events_path,
        )
        last_ts = df.index[-1]
        _persist_state(
            state_path,
            run_tag=run_tag, last_ts=last_ts,
            long_engine=long_engine, short_engine=short_engine,
        )
    consecutive_failures = 0

    try:
        while True:
            now_ms = exchange.milliseconds()
            try:
                new_df = _fetch_new_bars(exchange, symbol, timeframe, last_ts, timeframe_ms, now_ms)
            except (
                ccxt.NetworkError,
                ccxt.ExchangeNotAvailable,
                ccxt.RequestTimeout,
            ) as exc:
                consecutive_failures += 1
                if consecutive_failures > MAX_CONSECUTIVE_FAILURES:
                    raise DataIntegrityError(
                        f"連續 {consecutive_failures} 次 fetch 失敗，放棄"
                    ) from exc
                delay = min(
                    args.poll_seconds * (2 ** min(consecutive_failures, 8)),
                    MAX_BACKOFF_SECONDS,
                )
                logger.warning(
                    "network 錯誤 (%d/%d)：%s — sleep %ds 後重試",
                    consecutive_failures, MAX_CONSECUTIVE_FAILURES, exc, delay,
                )
                time.sleep(delay)
                continue
            except ccxt.RateLimitExceeded as exc:
                consecutive_failures += 1
                logger.warning("rate limit hit：%s — sleep 60s", exc)
                time.sleep(60)
                continue
            except (ccxt.AuthenticationError, ccxt.PermissionDenied):
                # 設定錯誤類，fail-fast
                logger.exception("auth / permission 錯誤，停止 live sim")
                raise
            except DataIntegrityError:
                # OHLCV 連續性 / 分頁失敗 — 不假裝沒事
                logger.exception("OHLCV 完整性檢查失敗，停止 live sim")
                raise

            consecutive_failures = 0
            if new_df.empty:
                time.sleep(args.poll_seconds)
                continue

            raw_df = _merge_new_bars(raw_df, new_df)
            _assert_continuous(raw_df, timeframe_ms, where="merged")
            df = prepare_indicators(raw_df, params)

            _append_klines_csv(kline_path, new_df, write_header=False)
            _process_batch(
                df,
                list(new_df.index),
                [long_engine, short_engine],
                trades_stream_path,
                equity_stream_path,
                position_events_path,
            )

            last_ts = df.index[-1]
            _persist_state(
                state_path,
                run_tag=run_tag,
                last_ts=last_ts,
                long_engine=long_engine,
                short_engine=short_engine,
            )
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        logger.info("Live sim 已由使用者中止")

    last_bar = Bar.from_row(df.index[-1], df.iloc[-1])
    if cfg.force_close_at_end:
        long_engine.force_close(last_bar)
        short_engine.force_close(last_bar)

    long_result = _build_result(long_engine, last_bar.close)
    short_result = _build_result(short_engine, last_bar.close)
    combined = build_merged_result("combined_live", [long_result, short_result])

    _write_result(long_result, cfg, run_dir / "long")
    _write_result(short_result, cfg, run_dir / "short")
    _write_result(combined, cfg, run_dir / "combined")

    logger.info("Live sim 輸出已保存：%s", run_dir)


# --------------------------------------------------------------------------- #
# Data fetch helpers
# --------------------------------------------------------------------------- #

def _build_exchange() -> ccxt.Exchange:
    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    exchange.load_markets()
    if not exchange.has.get("fetchOHLCV", False):
        raise ConfigError("exchange does not support fetchOHLCV")
    return exchange


def _normalize_timeframe(timeframe: str) -> str:
    if timeframe.endswith("H"):
        return timeframe[:-1] + "h"
    return timeframe


def _resolve_symbol(symbol: str, exchange: ccxt.Exchange) -> str:
    if symbol in exchange.markets:
        return symbol
    if "/" in symbol:
        if symbol in exchange.markets:
            return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        for candidate in (f"{base}/USDT:USDT", f"{base}/USDT"):
            if candidate in exchange.markets:
                return candidate
    raise ConfigError(f"symbol '{symbol}' not found in exchange markets")


def _fetch_recent_bars(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    limit: int,
    timeframe_ms: int,
) -> pd.DataFrame:
    # +1：binance 通常會回包含當前未收盤 K 的最新 limit 根，過濾後容易少 1 根
    rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 1)
    df = _ohlcv_to_df(rows)
    now_ms = exchange.milliseconds()
    df = _filter_closed_bars(df, timeframe_ms, now_ms)
    if df.empty:
        raise DataIntegrityError("no closed bars returned for warmup")
    _assert_continuous(df, timeframe_ms, where="warmup")
    return df


def _fetch_new_bars(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    last_ts: pd.Timestamp,
    timeframe_ms: int,
    now_ms: int,
    *,
    page_limit: int = 500,
) -> pd.DataFrame:
    """拉 ``last_ts`` 之後的所有已收盤 K，必要時自動分批 paginate。"""
    last_ms = _ts_to_unix_ms(last_ts)
    # 從 last_ts−2 根起拉，避免錯過剛剛邊界 K（重複的會在合併時丟掉）
    cursor_ms = last_ms - timeframe_ms * 2

    chunks: list[pd.DataFrame] = []
    # 防呆：理論上 (now_ms - last_ms)/timeframe_ms 不會超過 page_limit*N，N 取一個夠大值
    max_pages = 50
    for _ in range(max_pages):
        rows = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=cursor_ms, limit=page_limit
        )
        chunk = _ohlcv_to_df(rows)
        if chunk.empty:
            break
        chunks.append(chunk)
        last_chunk_ms = _ts_to_unix_ms(chunk.index[-1])
        # 抓到的最後一根已逼近 now → 結束分頁
        if last_chunk_ms + timeframe_ms >= now_ms:
            break
        # 否則從最後一根的下一根繼續
        next_cursor = last_chunk_ms + timeframe_ms
        if next_cursor <= cursor_ms:
            # 沒有前進，避免死迴圈
            break
        cursor_ms = next_cursor
    else:
        raise DataIntegrityError(
            f"_fetch_new_bars: paginate exceeded {max_pages} pages "
            f"(last_ts={last_ts}, now_ms={now_ms})"
        )

    if not chunks:
        return _ohlcv_to_df([])

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = _filter_closed_bars(df, timeframe_ms, now_ms)
    if df.empty:
        return df
    df = df[df.index > last_ts]
    if df.empty:
        return df
    validate_ohlc(df, require_volume=True)
    _assert_continuous(df, timeframe_ms, where="incremental")
    return df


def _assert_continuous(
    df: pd.DataFrame, timeframe_ms: int, *, where: str
) -> None:
    """檢查 OHLCV 時間序列連續、無缺口、無重複。失敗即 raise（CLAUDE.md §5）。"""
    if len(df) < 2:
        return
    expected = pd.Timedelta(milliseconds=timeframe_ms)
    diffs = df.index[1:] - df.index[:-1]
    if not (diffs == expected).all():
        bad = [
            (df.index[i], df.index[i + 1], diffs[i])
            for i in range(len(diffs))
            if diffs[i] != expected
        ]
        raise DataIntegrityError(
            f"OHLCV not continuous ({where}): expected step={expected}, "
            f"violations (first 5)={bad[:5]}"
        )


def _ohlcv_to_df(rows: list[list[float]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")
    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True).tz_convert(None)
    data = {
        "open": [r[1] for r in rows],
        "high": [r[2] for r in rows],
        "low": [r[3] for r in rows],
        "close": [r[4] for r in rows],
        "volume": [r[5] for r in rows],
    }
    df = pd.DataFrame(data=data, index=ts)
    df.index.name = "timestamp"
    return df


def _index_to_unix_ms(idx: pd.Index) -> "pd.Index":
    """DatetimeIndex → unix milliseconds（int64）。

    pandas 2.x 的 ``DatetimeIndex`` 預設 storage 為 ``datetime64[us]``，直接
    ``astype('int64')`` 取到的會是 microseconds 而非 nanoseconds。先 ``as_unit('ns')``
    強制統一到 ns，再 ``// 1_000_000`` 才是正確的毫秒。
    """
    return idx.as_unit("ns").asi8 // 1_000_000  # type: ignore[attr-defined]


def _ts_to_unix_ms(ts: pd.Timestamp) -> int:
    """單一 Timestamp → unix milliseconds（int）。同樣經 ``as_unit('ns')`` 統一單位。"""
    return int(ts.as_unit("ns").value // 1_000_000)


def _filter_closed_bars(df: pd.DataFrame, timeframe_ms: int, now_ms: int) -> pd.DataFrame:
    """只保留「肯定已收盤」的 K：``open_ts + timeframe + SAFE_BUFFER ≤ now_ms``。

    ``SAFE_BUFFER_MS`` 用來抵消本地鐘相對 binance server 的偏差（NTP 漂移），
    避免把 server 端尚未收盤的 K 誤當成已收盤而吃進策略。
    """
    if df.empty:
        return df
    ts_ms = _index_to_unix_ms(df.index)
    mask = ts_ms + timeframe_ms + SAFE_BUFFER_MS <= now_ms
    return df.loc[mask]


def _merge_new_bars(raw_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([raw_df, new_df]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


# --------------------------------------------------------------------------- #
# Strategy / sizing helpers
# --------------------------------------------------------------------------- #

def _build_params(cfg: FullConfig) -> StrategyParams:
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
        stage3_mode=cfg.trailing.stage3_mode,  # type: ignore[arg-type]
        r_ladder_normal_first_trigger=cfg.trailing.r_ladder_normal_first_trigger,
        r_ladder_normal_step=cfg.trailing.r_ladder_normal_step,
        r_ladder_abnormal_first_trigger=cfg.trailing.r_ladder_abnormal_first_trigger,
        r_ladder_abnormal_step=cfg.trailing.r_ladder_abnormal_step,
        r_ladder_trigger_offset=cfg.trailing.r_ladder_trigger_offset,
        r_ladder_abnormal_trigger_offset=cfg.trailing.r_ladder_abnormal_trigger_offset,
    )
    return StrategyParams(
        wma_fast=cfg.wma_fast,
        wma_slow=cfg.wma_slow,
        entry_source=cfg.entry_source,  # type: ignore[arg-type]
        trailing=trailing,
    )


def _compute_limit_price(bar_open: float, direction: Direction, slippage_pct: float) -> float:
    if direction is Direction.LONG:
        return bar_open * (1.0 + slippage_pct)
    return bar_open * (1.0 - slippage_pct)


def _stop_on_correct_side(direction: Direction, stop_price: float, limit_price: float) -> bool:
    if direction is Direction.LONG:
        return stop_price < limit_price
    return stop_price > limit_price


def _compute_quantity(
    *,
    config: EngineConfig,
    direction: Direction,
    limit_price: float,
    initial_stop: float,
    equity_now: float,
    taker_fee_rate: float,
    existing_notional: float = 0.0,
) -> tuple[float, float | None, bool, str]:
    if config.sizing_mode == "pct":
        if config.allow_pyramiding:
            cap_total = equity_now * config.leverage_cap
            available = cap_total - existing_notional
            if available <= 0:
                return 0.0, None, False, "leverage_cap_exhausted"
            target_notional = min(available, equity_now * config.position_size_pct)
        else:
            target_notional = equity_now * config.position_size_pct
        quantity = target_notional / limit_price
        return quantity, None, True, "ok"

    denom = abs(limit_price - initial_stop) + (limit_price + initial_stop) * taker_fee_rate
    if denom <= 0:
        return 0.0, None, False, "denom_non_positive"
    quantity = config.risk_per_trade_usdt / denom
    notional = quantity * limit_price

    if config.allow_pyramiding:
        cap_total = equity_now * config.leverage_cap
        if existing_notional + notional > cap_total:
            return 0.0, None, False, "leverage_cap_exceeded"
        return quantity, config.risk_per_trade_usdt, True, "ok"

    if notional > equity_now:
        return 0.0, None, False, "single_position_overleveraged"
    return quantity, config.risk_per_trade_usdt, True, "ok"


def _build_limit_order(
    *,
    direction: Direction,
    limit_price: float,
    quantity: float,
    initial_stop: float,
    target_risk_usdt: float | None,
):
    from src.broker.types import LimitOrder
    return LimitOrder(
        direction=direction,
        limit_price=limit_price,
        quantity=quantity,
        initial_stop=initial_stop,
        target_risk_usdt=target_risk_usdt,
    )


# --------------------------------------------------------------------------- #
# Recording helpers
# --------------------------------------------------------------------------- #

def _append_klines_csv(path: Path, df: pd.DataFrame, *, write_header: bool) -> None:
    if df.empty:
        return
    df.to_csv(path, mode="a", header=write_header, index_label="timestamp")


def _append_trades_csv(path: Path, trades: Iterable[Trade]) -> None:
    rows = [_trade_to_row(t) for t in trades]
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_position_events_csv(path: Path, events: Iterable[PositionEvent]) -> None:
    rows = [_event_to_row(e) for e in events]
    if not rows:
        return
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _append_equity_csv(path: Path, account: Account) -> None:
    if not account.equity_history:
        return
    ts, equity = account.equity_history[-1]
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "account", "equity"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": ts.isoformat(),
            "account": account.name,
            "equity": equity,
        })


# --------------------------------------------------------------------------- #
# State persistence — crash 後可用 --resume 還原
# --------------------------------------------------------------------------- #

def _resolve_resume_paths(
    resume_arg: str, results_root: Path, data_dir: Path
) -> tuple[str, Path, Path, Path]:
    """``--resume`` 參數可以是 run_tag 或 run_dir 路徑；回 (run_tag, run_dir, kline_path, state_path)。"""
    candidate = Path(resume_arg)
    if candidate.is_dir():
        run_dir = candidate.resolve()
    else:
        run_dir = (results_root / resume_arg).resolve()
    if not run_dir.is_dir():
        raise ConfigError(f"--resume: run dir 不存在：{run_dir}")
    run_tag = run_dir.name
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise ConfigError(f"--resume: 找不到 state.json：{state_path}")
    kline_path = data_dir / f"{run_tag}_klines.csv"
    if not kline_path.is_file():
        raise ConfigError(f"--resume: 找不到 klines.csv：{kline_path}")
    return run_tag, run_dir, kline_path, state_path


def _load_klines_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "timestamp"
    return df


def _serialize_pending_signal(sig: EntrySignal | None) -> dict | None:
    if sig is None:
        return None
    return {
        "direction": sig.direction.value,
        "timestamp": sig.timestamp.isoformat(),
        "initial_stop": float(sig.initial_stop),
        "reason": sig.reason,
    }


def _deserialize_pending_signal(
    data: dict | None, df: pd.DataFrame
) -> EntrySignal | None:
    if data is None:
        return None
    ts = pd.Timestamp(data["timestamp"])
    if ts not in df.index:
        # 該 K 已不在 df（不該發生：raw_df 從 csv 還原應包含），保險起見 raise
        raise DataIntegrityError(
            f"resume: pending_signal timestamp {ts} not found in raw_df index"
        )
    bar_index = int(df.index.get_loc(ts))
    return EntrySignal(
        direction=Direction(data["direction"]),
        bar_index=bar_index,
        timestamp=ts,
        initial_stop=float(data["initial_stop"]),
        reason=str(data["reason"]),
    )


def _serialize_engine(engine: "LiveEngine") -> dict:
    return {
        "name": engine.name,
        "account": engine.account.snapshot_state(),
        "pending_signal": _serialize_pending_signal(engine.pending_signal),
        "trailings": {
            str(pid): ctrl.snapshot_runtime()
            for pid, ctrl in engine.trailings.items()
        },
        "signals_emitted": engine.signals_emitted,
        "signals_filled": engine.signals_filled,
        "signals_unfilled": engine.signals_unfilled,
        "signals_skipped_pending": engine.signals_skipped_pending,
        "bars_processed": engine.bars_processed,
    }


def _persist_state(
    path: Path,
    *,
    run_tag: str,
    last_ts: pd.Timestamp,
    long_engine: "LiveEngine",
    short_engine: "LiveEngine",
) -> None:
    """以 atomic write（write tmp + fsync + rename）落地 state.json，避免 crash 留半行。"""
    payload = {
        "version": STATE_SCHEMA_VERSION,
        "run_tag": run_tag,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "last_ts": last_ts.isoformat(),
        "long": _serialize_engine(long_engine),
        "short": _serialize_engine(short_engine),
    }
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".state-", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        # 殘留 tmp 不致命，但別讓它累積
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _restore_engine(
    *,
    name: str,
    state: dict,
    strategy: BaseTrendStrategy,
    broker: BrokerSimulator,
    config: EngineConfig,
    df: pd.DataFrame,
) -> "LiveEngine":
    account = Account.restore_state(state["account"])
    engine = LiveEngine(
        name=name,
        strategy=strategy,
        account=account,
        broker=broker,
        config=config,
    )
    engine.pending_signal = _deserialize_pending_signal(
        state.get("pending_signal"), df
    )
    engine.signals_emitted = int(state["signals_emitted"])
    engine.signals_filled = int(state["signals_filled"])
    engine.signals_unfilled = int(state["signals_unfilled"])
    engine.signals_skipped_pending = int(state["signals_skipped_pending"])
    engine.bars_processed = int(state["bars_processed"])

    # Rebuild trailing controllers：每筆持倉重建一個 controller，再還原 runtime stage/peak
    trailing_states: dict[str, dict] = state.get("trailings", {})
    for pid, pos in account.positions.items():
        ctrl = TrailingStopController(
            position=pos,
            params=strategy.params.trailing,
            broker_config=broker.config,
        )
        runtime = trailing_states.get(str(pid))
        if runtime is None:
            logger.warning(
                "[%s] resume: trailing runtime for pid=%d 缺失，以 stage=1 重建",
                name, pid,
            )
        else:
            ctrl.restore_runtime(runtime)
        engine.trailings[pid] = ctrl
    return engine


def _last_stop_price(trade: Trade) -> float | None:
    if not trade.stop_history:
        return None
    return float(trade.stop_history[-1][1])


def _trade_to_row(trade: Trade) -> dict[str, float | str | None]:
    initial_stop = trade.stop_history[0][1] if trade.stop_history else None
    stop_distance = abs(trade.entry_price - initial_stop) if initial_stop is not None else None
    entry_notional = trade.entry_price * trade.quantity
    risk_usdt_no_fee = stop_distance * trade.quantity if stop_distance is not None else None
    position_value_per_1u = (
        (trade.entry_price / stop_distance)
        if stop_distance is not None and stop_distance > 0
        else None
    )
    return {
        "direction": trade.direction.value,
        "entry_ts": trade.entry_timestamp.isoformat(),
        "exit_ts": trade.exit_timestamp.isoformat(),
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "quantity": trade.quantity,
        "gross_pnl": trade.gross_pnl,
        "net_pnl": trade.net_pnl,
        "return_pct": trade.return_pct * 100.0,
        "exit_reason": trade.exit_reason,
        "entry_fee": trade.entry_fee,
        "exit_fee": trade.exit_fee,
        "holding_minutes": trade.holding_duration.total_seconds() / 60.0,
        "position_id": trade.position_id,
        "entry_notional": entry_notional,
        "initial_stop": initial_stop,
        "stop_distance": stop_distance,
        "risk_usdt_no_fee": risk_usdt_no_fee,
        "position_value_per_1u": position_value_per_1u,
    }


def _event_to_row(event: PositionEvent) -> dict[str, float | str | None]:
    return {
        "timestamp": event.timestamp.isoformat(),
        "position_id": event.position_id,
        "direction": event.direction.value,
        "event": event.event,
        "price": event.price,
        "quantity": event.quantity,
        "reason": event.reason,
        "stop_price": event.stop_price,
    }


def _process_batch(
    df: pd.DataFrame,
    timestamps: list[pd.Timestamp],
    engines: list[LiveEngine],
    trades_stream_path: Path,
    equity_stream_path: Path,
    position_events_path: Path,
) -> None:
    for ts in timestamps:
        bar_index = df.index.get_loc(ts)
        closed_trades: list[Trade] = []
        events: list[PositionEvent] = []
        for engine in engines:
            result = engine.process_bar(df, bar_index)
            closed_trades.extend(result.closed_trades)
            events.extend(result.events)
        _append_trades_csv(trades_stream_path, closed_trades)
        _append_position_events_csv(position_events_path, events)
        for engine in engines:
            _append_equity_csv(equity_stream_path, engine.account)


def _build_result(engine: LiveEngine, last_close: float) -> BacktestResult:
    if engine.account.equity_history:
        equity_curve = pd.Series(
            data=[v for _, v in engine.account.equity_history],
            index=pd.DatetimeIndex(
                [t for t, _ in engine.account.equity_history],
                name="timestamp",
            ),
            name="equity",
            dtype="float64",
        )
    else:
        equity_curve = pd.Series(dtype="float64", name="equity")
    final_equity = engine.account.equity(last_close)

    return BacktestResult(
        account_name=engine.account.name,
        initial_capital=engine.account.initial_capital,
        final_equity=final_equity,
        trades=engine.account.trade_log,
        equity_curve=equity_curve,
        bars_processed=engine.bars_processed,
        signals_emitted=engine.signals_emitted,
        signals_filled=engine.signals_filled,
        signals_unfilled=engine.signals_unfilled,
        signals_skipped_pending=engine.signals_skipped_pending,
        config_snapshot={
            "mode": "live_sim",
            "sizing_mode": engine.config.sizing_mode,
            "position_size_pct": engine.config.position_size_pct,
            "risk_per_trade_usdt": engine.config.risk_per_trade_usdt,
            "allow_pyramiding": engine.config.allow_pyramiding,
            "leverage_cap": engine.config.leverage_cap,
            "force_close_at_end": engine.config.force_close_at_end,
            "skip_signal_when_pending": engine.config.skip_signal_when_pending,
            "taker_fee_rate": engine.broker.config.taker_fee_rate,
            "slippage_pct": engine.broker.config.slippage_pct,
            "strategy_params": _serialize_params(engine.strategy.params),
        },
    )


def _serialize_params(params: StrategyParams) -> dict:
    from dataclasses import asdict
    return asdict(params)


def _write_result(result: BacktestResult, cfg: FullConfig, out_dir: Path) -> MetricsReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(result, timeframe=cfg.timeframe)

    summary_lines = _format_summary_lines(metrics, result)
    export_summary_text(summary_lines, out_dir / "summary.txt")
    export_metrics_json(metrics, out_dir / "metrics.json")
    export_trades_csv(result.trades, out_dir / "trades.csv")
    export_config_snapshot(result.config_snapshot, out_dir / "config_snapshot.json")

    plot_equity_curves({result.account_name: result.equity_curve}, out_dir / "equity.png",
                      title=f"Equity Curve — {result.account_name}")
    plot_drawdown(result.equity_curve, out_dir / "drawdown.png",
                  title=f"Drawdown — {result.account_name}")
    return metrics


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #

def _format_summary_lines(m: MetricsReport, result: BacktestResult) -> list[str]:
    buf: list[str] = []
    line = "=" * 60
    buf.append(line)
    buf.append(f"  績效摘要 / Performance Summary  [{result.account_name}]")
    buf.append(line)
    buf.append(f"  區間：{m.start} ~ {m.end}  ({m.duration_days:.1f} days)")
    buf.append("-" * 60)
    buf.append(f"  Initial Capital:     {m.initial_capital:>10.2f} USDT")
    buf.append(f"  Final Equity:        {m.final_equity:>10.2f} USDT")
    buf.append(f"  Total Return:        {m.total_return_pct:>+10.2f} %")
    buf.append(f"  Annualized Return:   {m.annualized_return_pct:>+10.2f} %")
    buf.append("-" * 60)
    buf.append(f"  Sharpe Ratio:        {m.sharpe_ratio:>10.2f}")
    buf.append(f"  Sortino Ratio:       {m.sortino_ratio:>10.2f}")
    buf.append(f"  Max Drawdown:        {m.max_drawdown_pct:>10.2f} %")
    buf.append(f"  Calmar Ratio:        {m.calmar_ratio:>10.2f}")
    buf.append("-" * 60)
    buf.append(f"  Total Trades:        {m.total_trades:>10d}")
    buf.append(f"  Avg Trades / Day:    {m.avg_trades_per_day:>10.2f}")
    buf.append(f"  Win Rate:            {m.win_rate_pct:>10.2f} %")
    buf.append(f"  Profit Factor:       {m.profit_factor:>10.2f}")
    buf.append(f"  Expectancy / trade:  {m.expectancy:>10.4f} USDT")
    buf.append(f"  Avg Win / Loss:      {m.avg_win:>10.4f} / {m.avg_loss:.4f}")
    buf.append(f"  Max Consec W / L:    {m.max_consecutive_wins:>10d} / {m.max_consecutive_losses}")
    buf.append(f"  Avg Hold (bars):     {m.avg_holding_bars:>10.2f}")
    buf.append("-" * 60)
    buf.append(f"  Stop Loss (intraday):{m.stop_loss_count:>10d}")
    buf.append(f"  Stop Loss (gap):     {m.stop_loss_gap_count:>10d}")
    buf.append("-" * 60)
    buf.append(
        f"  訊號統計：emitted={result.signals_emitted}, "
        f"filled={result.signals_filled}, "
        f"unfilled={result.signals_unfilled}, "
        f"skipped_pending={result.signals_skipped_pending}"
    )
    buf.append(line)
    return buf


if __name__ == "__main__":
    main()
