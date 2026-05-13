"""Stage 3 訂單進場時的 ADX / BBW / ATR 共同區間分析。

目的：找出 stage 3 訂單在 entry timestamp 的盤整類指標自然分布，
評估 sweep_chop_filters_combo.py 找出的門檻（BBW≥40 & ATR≥40 & ADX≥20）
是否與 stage 3 trade 的本質分布一致，或還有可挖空間。

執行：
    python -m scripts.analyze_stage3_chop_range
    python -m scripts.analyze_stage3_chop_range --end 2024-12-31

預設：from data start to 2024-12-31（純 IS，不碰 OOS）。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup,
)
from scripts.analyze_stage3_features_v2 import (  # noqa: E402
    _atr, _bb_width,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FEATURES = ["adx", "bbw_rank", "atr_rank", "bbw", "atr"]
PCTS = [5, 10, 25, 50, 75, 90, 95]


def _percentiles(arr: np.ndarray) -> dict[str, float]:
    clean = arr[np.isfinite(arr)]
    if clean.size == 0:
        return {f"p{p}": float("nan") for p in PCTS} | {
            "n": 0, "mean": float("nan"), "std": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    out: dict[str, float] = {"n": int(clean.size)}
    out["mean"] = float(clean.mean())
    out["std"] = float(clean.std())
    out["min"] = float(clean.min())
    out["max"] = float(clean.max())
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(clean, p))
    return out


def _enrichment_curve(
    s3_vals: np.ndarray, s1_vals: np.ndarray, bins: np.ndarray,
) -> pd.DataFrame:
    """對每個 bucket 計算 stage 3 占比與富集倍率 vs baseline。"""
    s3 = s3_vals[np.isfinite(s3_vals)]
    s1 = s1_vals[np.isfinite(s1_vals)]
    total = s3.size + s1.size
    if total == 0:
        return pd.DataFrame()
    baseline_s3_pct = s3.size / total * 100

    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_s3 = ((s3 >= lo) & (s3 < hi)).sum()
        in_s1 = ((s1 >= lo) & (s1 < hi)).sum()
        sub_total = in_s3 + in_s1
        if sub_total == 0:
            continue
        s3_pct = in_s3 / sub_total * 100
        rows.append({
            "bucket_lo": float(lo),
            "bucket_hi": float(hi),
            "n_s3": int(in_s3),
            "n_s1": int(in_s1),
            "s3_pct_in_bucket": float(s3_pct),
            "enrichment_vs_baseline": float(s3_pct / baseline_s3_pct) if baseline_s3_pct > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def _bins_for(feature: str) -> np.ndarray:
    if feature == "adx":
        return np.arange(0, 80, 5, dtype=float)
    if feature in ("bbw_rank", "atr_rank"):
        return np.arange(0, 105, 10, dtype=float)
    if feature == "bbw":
        # BBW 數值差異大，動態 bin 在主程式設
        return np.linspace(0, 1, 21)
    if feature == "atr":
        return np.linspace(0, 1, 21)
    raise ValueError(feature)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 entry-time chop indicator distribution."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--start", default=None,
        help="起始日期 (None = 從資料起始)。default: 從資料起始",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="結束日期（含）。default: 2024-12-31（純 IS，不碰 OOS）",
    )
    parser.add_argument(
        "--out-dir", default="results/stage3_chop_range",
        help="輸出資料夾",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    start_ts = pd.Timestamp(args.start) if args.start else pd.Timestamp("2020-01-01")
    end_ts = pd.Timestamp(args.end)

    print(f"Period: {start_ts.date()} ~ {end_ts.date()}  timeframe: {base_cfg.timeframe}")

    period = PeriodSpec(start=start_ts, end=end_ts)
    cfg = dataclasses.replace(
        base_cfg, in_sample=period, show_progress=False, log_level="WARNING",
    )

    print("Running long backtest ...")
    L = run_single_strategy(cfg, direction="long", sample="is")
    print(f"  long trades: {len(L.trades)}")
    print("Running short backtest ...")
    S = run_single_strategy(cfg, direction="short", sample="is")
    print(f"  short trades: {len(S.trades)}")

    trades = L.trades + S.trades
    print(f"Total trades: {len(trades)}")

    print("Loading bars + computing filter cache ...")
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m
    cache = _compute_filter_cache(df)
    # 額外算 BBW / ATR 的「絕對值」做分布觀察
    cache["bbw"] = _bb_width(df["close"], period=20, k=2.0)
    cache["atr"] = _atr(df, period=14)

    tf_delta = df.index[1] - df.index[0]

    # 對每筆 trade 取 entry-time 指標值
    n = len(trades)
    feat_vals: dict[str, np.ndarray] = {
        f: _filter_lookup(trades, cache[f], tf_delta) for f in FEATURES
    }
    stages = np.array([int(t.final_stage) for t in trades])

    # ----- 描述統計 -----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_rows = []
    for f in FEATURES:
        for stage in (1, 2, 3):
            mask = stages == stage
            stats = _percentiles(feat_vals[f][mask])
            stats_rows.append({"feature": f, "stage": stage, **stats})
        # 整體（all stages）做 baseline 對照
        stats = _percentiles(feat_vals[f])
        stats_rows.append({"feature": f, "stage": "all", **stats})
    stats_df = pd.DataFrame(stats_rows)
    stats_path = out_dir / "stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\nStats CSV: {stats_path}")

    # ----- 富集度（stage 1 vs stage 3）-----
    enr_dfs = []
    for f in FEATURES:
        if f == "bbw":
            v = feat_vals[f]
            clean = v[np.isfinite(v)]
            if clean.size > 0:
                bins = np.linspace(clean.min(), np.percentile(clean, 99), 20)
            else:
                bins = _bins_for(f)
        elif f == "atr":
            v = feat_vals[f]
            clean = v[np.isfinite(v)]
            if clean.size > 0:
                bins = np.linspace(clean.min(), np.percentile(clean, 99), 20)
            else:
                bins = _bins_for(f)
        else:
            bins = _bins_for(f)
        enr = _enrichment_curve(
            feat_vals[f][stages == 3], feat_vals[f][stages == 1], bins,
        )
        enr.insert(0, "feature", f)
        enr_dfs.append(enr)
    enr_df = pd.concat(enr_dfs, ignore_index=True)
    enr_path = out_dir / "enrichment.csv"
    enr_df.to_csv(enr_path, index=False)
    print(f"Enrichment CSV: {enr_path}")

    # ----- Console 摘要 -----
    n_s1 = int((stages == 1).sum())
    n_s2 = int((stages == 2).sum())
    n_s3 = int((stages == 3).sum())
    print(f"\nStage counts: s1={n_s1}  s2={n_s2}  s3={n_s3}  total={len(trades)}")
    print(f"Baseline stage 3 ratio: {n_s3 / max(1, len(trades)) * 100:.2f}%")

    print("\n=== Stage 3 entry-time 分布（共同區間 = P25 ~ P75）===")
    header = f"{'feature':<12} {'P10':>7} {'P25':>7} {'P50':>7} {'P75':>7} {'P90':>7} | {'S1 P50':>7} | shift"
    print(header)
    print("-" * len(header))
    s1_row_lookup = {(r["feature"], r["stage"]): r for r in stats_rows}
    for f in FEATURES:
        s3 = s1_row_lookup[(f, 3)]
        s1 = s1_row_lookup[(f, 1)]
        shift = s3["p50"] - s1["p50"] if np.isfinite(s3["p50"]) and np.isfinite(s1["p50"]) else float("nan")
        print(
            f"{f:<12} {s3['p10']:7.2f} {s3['p25']:7.2f} {s3['p50']:7.2f} "
            f"{s3['p75']:7.2f} {s3['p90']:7.2f} | {s1['p50']:7.2f} | {shift:+.2f}"
        )

    print("\n=== 建議門檻候選（保留 ≥90% stage 3 trade = stage3 的 P10）===")
    for f in ("adx", "bbw_rank", "atr_rank"):
        s3 = s1_row_lookup[(f, 3)]
        # 在該門檻下會砍掉多少 stage 1 vs stage 3
        thr = s3["p10"]
        s1_vals = feat_vals[f][stages == 1]
        s3_vals = feat_vals[f][stages == 3]
        s1_clean = s1_vals[np.isfinite(s1_vals)]
        s3_clean = s3_vals[np.isfinite(s3_vals)]
        cut_s1 = (s1_clean < thr).sum()
        cut_s3 = (s3_clean < thr).sum()
        ratio = cut_s1 / cut_s3 if cut_s3 > 0 else float("inf")
        print(
            f"  {f} >= {thr:.2f}  → 砍 stage 3: {cut_s3}/{s3_clean.size} ({cut_s3/s3_clean.size*100:.1f}%)"
            f"  | 砍 stage 1: {cut_s1}/{s1_clean.size} ({cut_s1/s1_clean.size*100:.1f}%)"
            f"  | s1/s3 cut ratio: {ratio:.2f}x"
        )

    # ----- 直方圖 -----
    print(f"\nWriting histograms to {out_dir} ...")
    for f in FEATURES:
        s1_vals = feat_vals[f][stages == 1]
        s3_vals = feat_vals[f][stages == 3]
        s1_clean = s1_vals[np.isfinite(s1_vals)]
        s3_clean = s3_vals[np.isfinite(s3_vals)]
        if s1_clean.size == 0 and s3_clean.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if f in ("bbw", "atr"):
            all_clean = np.concatenate([s1_clean, s3_clean])
            bins = np.linspace(all_clean.min(), np.percentile(all_clean, 99), 30)
        else:
            bins = _bins_for(f) if f != "adx" else np.arange(0, 70, 3, dtype=float)
        ax.hist(s1_clean, bins=bins, alpha=0.5, label=f"stage 1 (n={s1_clean.size})",
                color="tab:red", density=True)
        ax.hist(s3_clean, bins=bins, alpha=0.5, label=f"stage 3 (n={s3_clean.size})",
                color="tab:green", density=True)
        if s3_clean.size > 0:
            ax.axvline(np.percentile(s3_clean, 25), color="tab:green",
                       linestyle="--", linewidth=1, label="s3 P25/P75")
            ax.axvline(np.percentile(s3_clean, 75), color="tab:green",
                       linestyle="--", linewidth=1)
            ax.axvline(np.percentile(s3_clean, 10), color="tab:green",
                       linestyle=":", linewidth=1, alpha=0.7,
                       label="s3 P10/P90")
            ax.axvline(np.percentile(s3_clean, 90), color="tab:green",
                       linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"{f}: stage 1 vs stage 3 進場分布 ({start_ts.date()}~{end_ts.date()})")
        ax.set_xlabel(f)
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{f}.png", dpi=110)
        plt.close(fig)
    print(f"Done.  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
