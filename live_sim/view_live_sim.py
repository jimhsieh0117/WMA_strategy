"""Live simulation viewer for the latest live_sim results.

用法：
    python -m live_sim.view_live_sim [--run-dir live_sim/results/...]
                                   [--port 8060] [--host 127.0.0.1]

預設會自動挑選 live_sim/results 底下最新的 run 目錄。
畫面沿用回測版 FastAPI + Lightweight Charts，但資料改讀 live_sim 的最新輸出。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.types import BacktestResult  # noqa: E402
from src.strategy.base import prepare_indicators  # noqa: E402
from src.strategy.types import StrategyParams, TrailingStopParams  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.types import Direction  # noqa: E402
from src.viewer.indicators import REGISTRY, default_panels_for  # noqa: E402
from src.viewer.server import build_app  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveRunBundle:
    run_dir: Path
    run_tag: str
    kline_path: Path
    combined_dir: Path
    long_dir: Path
    short_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live sim viewer（FastAPI + Lightweight Charts）"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    bundle = _resolve_live_bundle(args.run_dir)
    snapshot = _load_snapshot(bundle.combined_dir / "config_snapshot.json")
    params = _build_params(cfg, snapshot)

    initial_enabled = default_panels_for(params.entry_source)
    for name in initial_enabled:
        if name not in REGISTRY:
            raise SystemExit(f"unknown panel '{name}'. available: {sorted(REGISTRY)}")

    df = _load_kline_frame(bundle.kline_path)
    augmented = prepare_indicators(df, params)
    for reg in REGISTRY.values():
        augmented = reg.compute(augmented)

    long_result = _load_result(bundle.long_dir, bundle.run_tag)
    short_result = _load_result(bundle.short_dir, bundle.run_tag)
    combined_result = _load_result(bundle.combined_dir, bundle.run_tag)

    app = build_app(
        symbol=snapshot.get("symbol", cfg.symbol),
        timeframe=snapshot.get("timeframe", cfg.timeframe),
        sample="live",
        df_main=augmented,
        indicators=list(REGISTRY.values()),
        initial_enabled=initial_enabled,
        long_result=long_result,
        short_result=short_result,
        combined_result=combined_result,
    )

    url = f"http://{args.host}:{args.port}/"
    print()
    print("=" * 60)
    print(f"  📈 Live viewer 啟動於：{url}")
    print(f"     run_dir：{bundle.run_dir}")
    print(f"     初始勾選：{initial_enabled}")
    print(f"     全部指標：{sorted(REGISTRY)}")
    print("     按 Ctrl+C 停止伺服器")
    print("=" * 60)
    print()

    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        print("\n伺服器已停止")


# --------------------------------------------------------------------------- #
# Bundle / IO
# --------------------------------------------------------------------------- #

def _resolve_live_bundle(run_dir_arg: str | None) -> LiveRunBundle:
    results_root = Path(__file__).resolve().parent / "results"
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"run dir not found: {run_dir}")
    else:
        run_dir = _latest_run_dir(results_root)

    run_tag = run_dir.name
    kline_path = Path(__file__).resolve().parent / "data" / f"{run_tag}_klines.csv"
    if not kline_path.is_file():
        raise SystemExit(f"kline csv not found: {kline_path}")

    return LiveRunBundle(
        run_dir=run_dir,
        run_tag=run_tag,
        kline_path=kline_path,
        combined_dir=run_dir / "combined",
        long_dir=run_dir / "long",
        short_dir=run_dir / "short",
    )


def _latest_run_dir(results_root: Path) -> Path:
    if not results_root.is_dir():
        raise SystemExit(f"results root not found: {results_root}")
    candidates = [p for p in results_root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"no live run directory found under {results_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_snapshot(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_kline_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        raise SystemExit(f"kline csv is empty: {path}")
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index)
    return df[["open", "high", "low", "close", "volume"]]


# --------------------------------------------------------------------------- #
# Strategy params
# --------------------------------------------------------------------------- #

def _build_params(cfg, snapshot: dict) -> StrategyParams:
    strategy_params = snapshot.get("strategy_params", {}) if snapshot else {}
    trailing_raw = strategy_params.get("trailing", {}) if isinstance(strategy_params, dict) else {}
    trailing = TrailingStopParams(
        swing_lookback=int(trailing_raw.get("swing_lookback", cfg.trailing.swing_lookback)),
        stage1_slippage_buffer=float(trailing_raw.get("stage1_slippage_buffer", cfg.trailing.stage1_slippage_buffer)),
        stage2_normal_trigger_r=float(trailing_raw.get("stage2_normal_trigger_r", cfg.trailing.stage2_normal_trigger_r)),
        stage2_abnormal_trigger_r=float(trailing_raw.get("stage2_abnormal_trigger_r", cfg.trailing.stage2_abnormal_trigger_r)),
        stage2_buffer_r=float(trailing_raw.get("stage2_buffer_r", cfg.trailing.stage2_buffer_r)),
        stage3_normal_trigger_r=float(trailing_raw.get("stage3_normal_trigger_r", cfg.trailing.stage3_normal_trigger_r)),
        stage3_abnormal_trigger_r=float(trailing_raw.get("stage3_abnormal_trigger_r", cfg.trailing.stage3_abnormal_trigger_r)),
        bollinger_period=int(trailing_raw.get("bollinger_period", cfg.trailing.bollinger_period)),
        bollinger_num_std=float(trailing_raw.get("bollinger_num_std", cfg.trailing.bollinger_num_std)),
        stage3_mode=str(trailing_raw.get("stage3_mode", cfg.trailing.stage3_mode)),
        r_ladder_normal_first_trigger=float(trailing_raw.get("r_ladder_normal_first_trigger", cfg.trailing.r_ladder_normal_first_trigger)),
        r_ladder_normal_step=float(trailing_raw.get("r_ladder_normal_step", cfg.trailing.r_ladder_normal_step)),
        r_ladder_abnormal_first_trigger=float(trailing_raw.get("r_ladder_abnormal_first_trigger", cfg.trailing.r_ladder_abnormal_first_trigger)),
        r_ladder_abnormal_step=float(trailing_raw.get("r_ladder_abnormal_step", cfg.trailing.r_ladder_abnormal_step)),
        r_ladder_trigger_offset=float(trailing_raw.get("r_ladder_trigger_offset", cfg.trailing.r_ladder_trigger_offset)),
        r_ladder_abnormal_trigger_offset=float(trailing_raw.get("r_ladder_abnormal_trigger_offset", cfg.trailing.r_ladder_abnormal_trigger_offset)),
    )
    return StrategyParams(
        wma_fast=int(strategy_params.get("wma_fast", cfg.wma_fast)),
        wma_slow=int(strategy_params.get("wma_slow", cfg.wma_slow)),
        entry_source=str(strategy_params.get("entry_source", cfg.entry_source)),
        trailing=trailing,
    )


# --------------------------------------------------------------------------- #
# Result loading
# --------------------------------------------------------------------------- #

def _load_result(result_dir: Path, run_tag: str) -> BacktestResult:
    trades = _load_trades(result_dir / "trades.csv")
    equity_curve = _load_equity_curve(run_tag, result_dir.name)
    metrics_path = result_dir / "metrics.json"
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    initial_capital = float(metrics_payload.get("initial_capital", 0.0))
    final_equity = float(metrics_payload.get("final_equity", equity_curve.iloc[-1] if len(equity_curve) else 0.0))

    return BacktestResult(
        account_name=result_dir.name,
        initial_capital=initial_capital,
        final_equity=final_equity,
        trades=trades,
        equity_curve=equity_curve,
        bars_processed=len(equity_curve),
        signals_emitted=0,
        signals_filled=0,
        signals_unfilled=0,
        signals_skipped_pending=0,
        config_snapshot=_load_json(result_dir / "config_snapshot.json"),
    )


def _load_trades(path: Path) -> list:
    if not path.is_file():
        return []
    trades_df = pd.read_csv(path)
    if trades_df.empty:
        return []

    events = _load_position_events(path.parent.parent / "position_events.csv")
    trades = []
    for _, row in trades_df.iterrows():
        direction = Direction(str(row["direction"]))
        position_id = int(row.get("position_id", 0))
        key = (direction.value, position_id)
        stop_history = events.get(key, [])
        if not stop_history and not pd.isna(row.get("initial_stop", pd.NA)):
            entry_ts = pd.Timestamp(row["entry_ts"])
            stop_history = [(entry_ts, float(row["initial_stop"]))]
            exit_ts = pd.Timestamp(row["exit_ts"])
            stop_history.append((exit_ts, float(row["initial_stop"])))

        trades.append(
            _trade_from_row(row, direction=direction, stop_history=tuple(stop_history))
        )
    return trades


def _trade_from_row(row: pd.Series, *, direction: Direction, stop_history: tuple[tuple[pd.Timestamp, float], ...]):
    from src.broker.types import Trade

    return Trade(
        direction=direction,
        quantity=float(row["quantity"]),
        entry_price=float(row["entry_price"]),
        entry_timestamp=pd.Timestamp(row["entry_ts"]),
        exit_price=float(row["exit_price"]),
        exit_timestamp=pd.Timestamp(row["exit_ts"]),
        entry_fee=float(row["entry_fee"]),
        exit_fee=float(row["exit_fee"]),
        gross_pnl=float(row["gross_pnl"]),
        net_pnl=float(row["net_pnl"]),
        return_pct=float(row["return_pct"]) / 100.0,
        exit_reason=str(row["exit_reason"]),
        stop_history=stop_history,
        position_id=int(row.get("position_id", 0)),
    )


def _load_position_events(path: Path) -> dict[tuple[str, int], list[tuple[pd.Timestamp, float]]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return {}
    df = df.sort_values(["direction", "position_id", "timestamp"])
    grouped: dict[tuple[str, int], list[tuple[pd.Timestamp, float]]] = {}
    for _, row in df.iterrows():
        key = (str(row["direction"]), int(row["position_id"]))
        event = str(row["event"])
        stop_price = row.get("stop_price")
        if pd.isna(stop_price):
            continue
        if event not in {"ENTRY", "STOP_UPDATE", "EXIT"}:
            continue
        grouped.setdefault(key, [])
        ts = pd.Timestamp(row["timestamp"])
        value = float(stop_price)
        if not grouped[key] or grouped[key][-1][1] != value:
            grouped[key].append((ts, value))
    return grouped


def _load_equity_curve(run_tag: str, bucket: str) -> pd.Series:
    equity_path = Path(__file__).resolve().parent / "results" / run_tag / "equity_curve_live.csv"
    if not equity_path.is_file():
        return pd.Series(dtype="float64", name="equity")

    df = pd.read_csv(equity_path, parse_dates=["timestamp"])
    if df.empty:
        return pd.Series(dtype="float64", name="equity")

    pivot = df.pivot_table(index="timestamp", columns="account", values="equity", aggfunc="last").sort_index()
    if pivot.empty:
        return pd.Series(dtype="float64", name="equity")

    long_cols = [c for c in pivot.columns if str(c).startswith("long_")]
    short_cols = [c for c in pivot.columns if str(c).startswith("short_")]
    long_series = pivot[long_cols[0]].ffill().fillna(0.0) if long_cols else pd.Series(index=pivot.index, data=0.0)
    short_series = pivot[short_cols[0]].ffill().fillna(0.0) if short_cols else pd.Series(index=pivot.index, data=0.0)

    if bucket == "long":
        series = long_series
    elif bucket == "short":
        series = short_series
    else:
        series = (long_series + short_series)

    series = series.rename("equity")
    series.index.name = "timestamp"
    return series


# --------------------------------------------------------------------------- #
# Generic JSON helper
# --------------------------------------------------------------------------- #

def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
