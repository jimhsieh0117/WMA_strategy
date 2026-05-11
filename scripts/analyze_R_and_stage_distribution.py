"""W(4,6) baseline vs chop-filtered：peak_progress_R 分佈 + Stage 1/2/3 數量。

對 IS + OOS 各跑一次 baseline，套 chop filter (BBW>=40 & ATR>=40 & ADX>=20)。
產出：
  - Stage 1/2/3 數量（baseline vs filtered，含百分比）
  - peak_progress_r 分位數（p10/p25/p50/p75/p90/p95）
  - R bucket（<0.5 / 0.5-1 / 1-2 / 2-3 / >=3）× stage 的次數表
  - 兩張對照直方圖（baseline / filtered）

用法：
    python scripts/analyze_R_and_stage_distribution.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


R_BUCKETS = [(-np.inf, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, np.inf)]
R_BUCKET_LABELS = ["<0.5", "0.5~1", "1~2", "2~3", ">=3"]


def _apply_combo(fv, bbw_thr, atr_thr, adx_thr) -> np.ndarray:
    return (
        (fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
        (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
        (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"])
    )


def _build_trade_df(trades) -> pd.DataFrame:
    return pd.DataFrame({
        "stage":  [t.final_stage for t in trades],
        "R":      [t.peak_progress_r for t in trades],
        "pnl":    [t.net_pnl for t in trades],
    })


def _stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    rows = []
    for s in (1, 2, 3):
        sub = df[df.stage == s]
        rows.append({
            "stage": s,
            "n": len(sub),
            "share%": len(sub) / n * 100 if n else 0.0,
            "R_mean":  float(sub.R.mean())   if len(sub) else np.nan,
            "R_p50":   float(sub.R.median()) if len(sub) else np.nan,
            "R_max":   float(sub.R.max())    if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


def _r_quantiles(df: pd.DataFrame) -> pd.Series:
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    return pd.Series(df.R.quantile(qs).values,
                     index=[f"p{int(q*100)}" for q in qs])


def _r_bucket_x_stage(df: pd.DataFrame) -> pd.DataFrame:
    out = {label: [] for label in R_BUCKET_LABELS}
    for s in (1, 2, 3):
        sub = df[df.stage == s]
        for label, (lo, hi) in zip(R_BUCKET_LABELS, R_BUCKETS):
            cnt = ((sub.R >= lo) & (sub.R < hi)).sum()
            out[label].append(int(cnt))
    bucket_df = pd.DataFrame(out, index=[f"stage{s}" for s in (1, 2, 3)])
    bucket_df["total"] = bucket_df.sum(axis=1)
    return bucket_df


def _run_sample(cfg, start, end, args, tf_delta, label: str) -> dict:
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

    df_base = _build_trade_df(trades)
    df_filt = _build_trade_df([t for t, k in zip(trades, keep) if k])

    print(f"  baseline n={len(df_base)}  filtered n={len(df_filt)}")
    return {"baseline": df_base, "filtered": df_filt}


def _print_section(title: str, base: pd.DataFrame, filt: pd.DataFrame) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}")

    s_b = _stage_summary(base)
    s_f = _stage_summary(filt)
    print("\nStage 數量 / 占比 / R 統計：")
    merged = s_b.merge(s_f, on="stage", suffixes=("_base", "_filt"))
    print(merged.to_string(index=False, float_format="%.2f"))

    print("\nR (peak_progress_r) 分位數：")
    q = pd.DataFrame({"baseline": _r_quantiles(base),
                      "filtered": _r_quantiles(filt)})
    print(q.to_string(float_format="%.3f"))

    print("\nR bucket × stage（baseline）：")
    print(_r_bucket_x_stage(base).to_string())
    print("\nR bucket × stage（filtered）：")
    print(_r_bucket_x_stage(filt).to_string())


def _plot_hist(sample_out: dict, sample_label: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, kind in zip(axes, ("baseline", "filtered")):
        df = sample_out[kind]
        # 切到 0..6R 顯示主體；極端尾巴在 quantile 表已含
        r = df.R.clip(lower=0, upper=6).values
        # 用 stage 著色
        colors = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}
        for s in (1, 2, 3):
            sub = df[df.stage == s].R.clip(lower=0, upper=6).values
            ax.hist(sub, bins=40, range=(0, 6), alpha=0.6,
                    label=f"stage {s} (n={len(sub)})", color=colors[s], stacked=False)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=0.8)
        ax.axvline(2.0, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(f"{sample_label} — {kind}  (n={len(df)})")
        ax.set_xlabel("peak_progress_r")
        ax.set_ylabel("trade count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
    parser.add_argument("--out-dir", default="results/r_stage_distribution")
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

    _print_section("IS (2023-2024)",  is_out["baseline"],  is_out["filtered"])
    _print_section("OOS (2025+)",     oos_out["baseline"], oos_out["filtered"])

    _plot_hist(is_out,  "IS",  out_dir / "R_hist_IS.png")
    _plot_hist(oos_out, "OOS", out_dir / "R_hist_OOS.png")
    print(f"\n💾 plots → {out_dir}")


if __name__ == "__main__":
    main()
