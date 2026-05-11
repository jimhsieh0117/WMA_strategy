"""逐小時 (UTC 0..23) 訂單分布：baseline vs chop-filtered，IS + OOS。

對每個 hour bucket：
  - 訂單數、stage 1/2/3 計數、stage 3 占比
  - 平均 / 總 PnL
  - 過濾移除率（filter 對該小時的削減）

用法：
    python scripts/analyze_hour_distribution.py
    python scripts/analyze_hour_distribution.py --wma-fast 2 --wma-slow 4
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


def _apply_combo(fv, bbw_thr, atr_thr, adx_thr) -> np.ndarray:
    return (
        (fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
        (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
        (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"])
    )


def _trade_rows(trades, keep_mask: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "hour":  [pd.Timestamp(t.entry_timestamp).hour for t in trades],
        "stage": [t.final_stage for t in trades],
        "pnl":   [t.net_pnl for t in trades],
        "kept":  keep_mask.astype(bool),
    })


def _hour_summary(df: pd.DataFrame, subset_mask: pd.Series | None = None) -> pd.DataFrame:
    if subset_mask is not None:
        df = df[subset_mask].copy()
    rows = []
    for h in range(24):
        sub = df[df.hour == h]
        n = len(sub)
        if n == 0:
            rows.append({"hour": h, "n": 0, "s1": 0, "s2": 0, "s3": 0,
                         "s3%": 0.0, "win%": 0.0, "pnl": 0.0, "avg_pnl": 0.0})
            continue
        rows.append({
            "hour": h,
            "n":    n,
            "s1":   int((sub.stage == 1).sum()),
            "s2":   int((sub.stage == 2).sum()),
            "s3":   int((sub.stage == 3).sum()),
            "s3%":  float((sub.stage == 3).mean() * 100),
            "win%": float((sub.pnl > 0).mean() * 100),
            "pnl":  float(sub.pnl.sum()),
            "avg_pnl": float(sub.pnl.mean()),
        })
    return pd.DataFrame(rows)


def _run_sample(cfg, start, end, args, tf_delta, label) -> pd.DataFrame:
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

    df = _trade_rows(trades, keep)
    base = _hour_summary(df).rename(columns=lambda c: f"base_{c}" if c != "hour" else c)
    filt = _hour_summary(df, df.kept).rename(
        columns=lambda c: f"filt_{c}" if c != "hour" else c)
    out = base.merge(filt, on="hour")
    out["remove%"] = np.where(out.base_n > 0,
                              (out.base_n - out.filt_n) / out.base_n * 100, 0.0)
    out["sample"] = label
    return out


def _print_table(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*100}\n=== {label} 每小時 (UTC) 訂單分布 ==={'='*100}")
    show = df[["hour", "base_n", "base_s1", "base_s2", "base_s3", "base_s3%",
               "base_win%", "base_pnl", "base_avg_pnl",
               "filt_n", "filt_s3", "filt_s3%", "filt_win%",
               "filt_pnl", "filt_avg_pnl", "remove%"]]
    print(show.to_string(index=False, float_format="%.2f"))

    print(f"\n[{label}] 每小時 baseline 訂單數 top 5：")
    print(df.sort_values("base_n", ascending=False).head(5)[
        ["hour", "base_n", "base_s3%", "base_avg_pnl"]
    ].to_string(index=False, float_format="%.2f"))
    print(f"[{label}] 每小時 filtered 平均 PnL top 5（n>=20）：")
    cand = df[df.filt_n >= 20].sort_values("filt_avg_pnl", ascending=False).head(5)
    print(cand[["hour", "filt_n", "filt_s3%", "filt_avg_pnl", "filt_pnl"]
               ].to_string(index=False, float_format="%.2f"))
    print(f"[{label}] 每小時 filtered 平均 PnL bot 5（n>=20）：")
    cand = df[df.filt_n >= 20].sort_values("filt_avg_pnl").head(5)
    print(cand[["hour", "filt_n", "filt_s3%", "filt_avg_pnl", "filt_pnl"]
               ].to_string(index=False, float_format="%.2f"))


def _plot(df_is: pd.DataFrame, df_oos: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for col, (df_s, name) in enumerate([(df_is, "IS"), (df_oos, "OOS")]):
        # Top: count baseline vs filtered
        ax = axes[0, col]
        w = 0.4
        ax.bar(df_s.hour - w/2, df_s.base_n, width=w, label="baseline",
               color="gray", alpha=0.7)
        ax.bar(df_s.hour + w/2, df_s.filt_n, width=w, label="filtered",
               color="steelblue")
        ax.set_title(f"{name}：每小時訂單數")
        ax.set_ylabel("trade count")
        ax.set_xticks(range(0, 24))
        ax.legend()
        ax.grid(alpha=0.3)
        # Bottom: avg PnL baseline vs filtered
        ax = axes[1, col]
        ax.bar(df_s.hour - w/2, df_s.base_avg_pnl, width=w, label="baseline",
               color="gray", alpha=0.7)
        ax.bar(df_s.hour + w/2, df_s.filt_avg_pnl, width=w, label="filtered",
               color="steelblue")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(f"{name}：每小時平均 PnL")
        ax.set_xlabel("hour (UTC)")
        ax.set_ylabel("avg PnL (USDT/trade)")
        ax.set_xticks(range(0, 24))
        ax.legend()
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
    parser.add_argument("--out-dir", default="results/hour_distribution")
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

    df_is  = _run_sample(cfg_m, args.is_start,  args.is_end,  args, tf_delta, "IS")
    df_oos = _run_sample(cfg_m, args.oos_start, args.oos_end, args, tf_delta, "OOS")

    _print_table(df_is, "IS (2023-2024)")
    _print_table(df_oos, "OOS (2025+)")

    combined = pd.concat([df_is, df_oos], ignore_index=True)
    combined.to_csv(out_dir / "hour_distribution.csv", index=False,
                    float_format="%.4f")
    _plot(df_is, df_oos, out_dir / "hour_distribution.png")
    print(f"\n💾 → {out_dir}")


if __name__ == "__main__":
    main()
