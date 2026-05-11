"""每筆訂單的「實際出場 R」（vs peak_progress_r）。

realized_r = net_pnl / (|entry_price − initial_stop| × quantity)
  - initial_stop 取 trade.stop_history[0][1]
  - 包含手續費（用 net_pnl，不是 gross），符合「實際拿到手」的口徑
  - 與 peak_progress_r 並列，看 trailing stop 吐回多少

報告：
  - 每個 stage（1/2/3）的 realized R 與 peak R 統計、平均吐回幅度
  - baseline vs chop-filtered，IS + OOS

用法：
    python scripts/analyze_realized_R.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


def _apply_combo(fv, bbw_thr, atr_thr, adx_thr) -> np.ndarray:
    return (
        (fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
        (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
        (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"])
    )


def _realized_r(t) -> float:
    """net_pnl / (|entry − initial_stop| × quantity)。"""
    if not t.stop_history:
        return float("nan")
    initial_stop = float(t.stop_history[0][1])
    risk_per_unit = abs(t.entry_price - initial_stop)
    total_risk = risk_per_unit * t.quantity
    if total_risk <= 0:
        return float("nan")
    return t.net_pnl / total_risk


def _build_df(trades, keep_mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "stage":      [t.final_stage for t in trades],
        "peak_R":     [t.peak_progress_r for t in trades],
        "realized_R": [_realized_r(t) for t in trades],
        "pnl":        [t.net_pnl for t in trades],
        "kept":       keep_mask.astype(bool),
    })


def _summary_by_stage(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for s in (1, 2, 3):
        sub = df[df.stage == s]
        if len(sub) == 0:
            continue
        rR = sub.realized_R.values
        pR = sub.peak_R.values
        rR = rR[np.isfinite(rR)]
        pR = pR[np.isfinite(pR)]
        rows.append({
            "stage": s,
            "n":            len(sub),
            "realized_mean": float(rR.mean()) if rR.size else np.nan,
            "realized_p50":  float(np.median(rR)) if rR.size else np.nan,
            "realized_max":  float(rR.max()) if rR.size else np.nan,
            "realized_min":  float(rR.min()) if rR.size else np.nan,
            "peak_mean":     float(pR.mean()) if pR.size else np.nan,
            "peak_p50":      float(np.median(pR)) if pR.size else np.nan,
            "giveback_R":    float(pR.mean() - rR.mean()) if rR.size and pR.size else np.nan,
            "capture_%":     float(rR.mean() / pR.mean() * 100) if (rR.size and pR.size and pR.mean() > 0) else np.nan,
            "expectancy_R":  float(rR.mean()) if rR.size else np.nan,
        })
    df_out = pd.DataFrame(rows)
    df_out.insert(0, "sample", label)
    return df_out


def _overall(df: pd.DataFrame, label: str) -> dict:
    rR = df.realized_R.values
    rR = rR[np.isfinite(rR)]
    return {
        "sample": label,
        "n":              len(df),
        "exp_R_per_trade": float(rR.mean()) if rR.size else np.nan,
        "exp_R_median":   float(np.median(rR)) if rR.size else np.nan,
        "win_rate%":      float((rR > 0).mean() * 100) if rR.size else 0.0,
        "total_R":        float(rR.sum()) if rR.size else 0.0,
    }


def _run_sample(cfg, start, end, args, tf_delta, label) -> dict:
    print(f"\n>>> [{label}]  {start} ~ {end}", flush=True)
    cfg_p = dataclasses.replace(cfg, in_sample=PeriodSpec(
        start=pd.Timestamp(start), end=pd.Timestamp(end)))
    L = run_single_strategy(cfg_p, direction="long", sample="is")
    S = run_single_strategy(cfg_p, direction="short", sample="is")
    trades = L.trades + S.trades

    df1m = load_ohlcv(cfg_p.source_parquet,
                      start=pd.Timestamp(start), end=pd.Timestamp(end))
    cache = _compute_filter_cache(resample(df1m, args.timeframe))
    fv = {col: _filter_lookup(trades, cache[col], tf_delta)
          for col in ["adx", "atr_rank", "bbw_rank"]}
    keep = _apply_combo(fv, args.bbw, args.atr, args.adx)

    df_base = _build_df(trades, keep)
    df_filt = df_base[df_base.kept].copy()

    by_stage_base = _summary_by_stage(df_base, f"{label} baseline")
    by_stage_filt = _summary_by_stage(df_filt, f"{label} filtered")
    overall_base = _overall(df_base, f"{label} baseline")
    overall_filt = _overall(df_filt, f"{label} filtered")

    return {
        "df_base": df_base, "df_filt": df_filt,
        "by_stage": pd.concat([by_stage_base, by_stage_filt], ignore_index=True),
        "overall":  pd.DataFrame([overall_base, overall_filt]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--wma-fast", type=int, default=4)
    parser.add_argument("--wma-slow", type=int, default=6)
    parser.add_argument("--bbw", type=float, default=40.0)
    parser.add_argument("--atr", type=float, default=40.0)
    parser.add_argument("--adx", type=float, default=20.0)
    parser.add_argument("--is-start", default="2023-01-01")
    parser.add_argument("--is-end",   default="2024-12-31")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--oos-end",   default="2026-03-14")
    parser.add_argument("--out-dir", default="results/realized_R")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_m = dataclasses.replace(
        cfg, show_progress=False, log_level="WARNING",
        timeframe=args.timeframe,
        wma_fast=args.wma_fast, wma_slow=args.wma_slow,
    )
    tf_delta = pd.Timedelta(args.timeframe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== filter: BBW>={args.bbw} & ATR>={args.atr} & ADX>={args.adx} "
          f"| WMA({args.wma_fast},{args.wma_slow}) {args.timeframe} ===")

    is_out  = _run_sample(cfg_m, args.is_start,  args.is_end,  args, tf_delta, "IS")
    oos_out = _run_sample(cfg_m, args.oos_start, args.oos_end, args, tf_delta, "OOS")

    print("\n" + "=" * 110)
    print("=== Stage-level realized R 統計（含 peak vs realized 比對） ===")
    print("=" * 110)
    by_stage = pd.concat([is_out["by_stage"], oos_out["by_stage"]], ignore_index=True)
    print(by_stage.to_string(index=False, float_format="%.3f"))

    print("\n" + "=" * 70)
    print("=== 整體 expectancy（R 單位） ===")
    print("=" * 70)
    overall = pd.concat([is_out["overall"], oos_out["overall"]], ignore_index=True)
    print(overall.to_string(index=False, float_format="%.3f"))

    by_stage.to_csv(out_dir / "by_stage.csv", index=False, float_format="%.4f")
    overall.to_csv(out_dir / "overall.csv", index=False, float_format="%.4f")
    print(f"\n💾 → {out_dir}")


if __name__ == "__main__":
    main()
