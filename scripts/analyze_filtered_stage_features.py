"""在 chop-filtered 子集裡，重做 stage 1 vs stage 3 特徵分析。

問題：之前 univariate / multivariate 分析在全 trade set 上找不到能區分
stage 1 / stage 3 的訊號（max |d| < 0.15）。在已過濾掉「明顯爛環境」的
子集裡，是否還剩下能挑 stage 3 的訊號？

做法：
  1. 用 analyze_stage3_multivariate._build_features 生 24 特徵（2023-2024 IS）
  2. 套 chop filter (BBW>=40 & ATR>=40 & ADX>=20) 算 filtered mask
  3. 比較 stage 1 vs stage 3：
     - 全集合 (baseline) Cohen's d
     - 過濾後子集 (filtered) Cohen's d
     - 兩者差距：filter 後是否「顯化」了原本被噪音掩蓋的訊號？
  4. 額外印 stage 3 共同特徵（mean / median）作為「stage 3 長什麼樣」

用法：
    python scripts/analyze_filtered_stage_features.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_stage3_multivariate import (  # noqa: E402
    FEATURES, _build_features,
)
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    sa = a.var(ddof=1)
    sb = b.var(ddof=1)
    pooled = np.sqrt(((a.size - 1) * sa + (b.size - 1) * sb) / (a.size + b.size - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def _compare(df: pd.DataFrame, label: str) -> pd.DataFrame:
    s3 = df[df.final_stage == 3]
    s1 = df[df.final_stage == 1]
    rows = []
    for f in FEATURES:
        if f not in df.columns:
            continue
        a = s3[f].to_numpy()
        b = s1[f].to_numpy()
        d = _cohens_d(a, b)
        rows.append({
            "feature": f,
            f"{label}_d": d,
            f"{label}_s3_mean": float(np.nanmean(a)) if a.size else np.nan,
            f"{label}_s1_mean": float(np.nanmean(b)) if b.size else np.nan,
        })
    return pd.DataFrame(rows).set_index("feature")


def _stage3_common(df: pd.DataFrame, label: str) -> pd.DataFrame:
    s3 = df[df.final_stage == 3]
    rows = []
    for f in FEATURES:
        if f not in df.columns:
            continue
        v = s3[f].to_numpy()
        v = v[np.isfinite(v)]
        if v.size < 5:
            continue
        rows.append({
            "feature":     f,
            f"{label}_mean":    float(v.mean()),
            f"{label}_p25":     float(np.percentile(v, 25)),
            f"{label}_p50":     float(np.percentile(v, 50)),
            f"{label}_p75":     float(np.percentile(v, 75)),
        })
    return pd.DataFrame(rows).set_index("feature")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--wma-fast", type=int, default=4)
    parser.add_argument("--wma-slow", type=int, default=6)
    parser.add_argument("--bbw", type=float, default=40.0)
    parser.add_argument("--atr", type=float, default=40.0)
    parser.add_argument("--adx", type=float, default=20.0)
    parser.add_argument("--out-dir", default="results/filtered_stage_features")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    base_cfg = load_config(args.config)

    print(f">>> building features for {args.timeframe} W({args.wma_fast},{args.wma_slow}) "
          f"on 2023-01-01 ~ 2024-12-31 ...", flush=True)
    df_feat = _build_features(args.timeframe, args.wma_fast, args.wma_slow, base_cfg)
    print(f"  total trades with features: {len(df_feat)}")

    # 取 IS 對應的 timeframe-resampled bars，計算 filter cache 並對每筆 row 標記 keep
    print(">>> computing chop filter values ...")
    period = PeriodSpec(start=pd.Timestamp("2023-01-01"),
                        end=pd.Timestamp("2024-12-31"))
    df1m = load_ohlcv(base_cfg.source_parquet, start=period.start, end=period.end)
    df_tf = resample(df1m, args.timeframe)
    cache = _compute_filter_cache(df_tf)
    tf_delta = pd.Timedelta(args.timeframe)

    # 因為 df_feat 沒有 trade obj，我們直接用 entry_ts → signal_ts 查表
    def _lookup(series: pd.Series, ts_arr: np.ndarray) -> np.ndarray:
        out = np.full(ts_arr.size, np.nan)
        idx = series.index
        for i, ts in enumerate(ts_arr):
            signal_ts = pd.Timestamp(ts) - tf_delta
            pos = idx.searchsorted(signal_ts, side="right") - 1
            if pos >= 0:
                v = series.iloc[pos]
                if np.isfinite(v):
                    out[i] = float(v)
        return out

    ts_arr = df_feat["entry_ts"].to_numpy()
    adx_v = _lookup(cache["adx"],       ts_arr)
    atr_v = _lookup(cache["atr_rank"],  ts_arr)
    bbw_v = _lookup(cache["bbw_rank"],  ts_arr)
    keep = (
        (adx_v >= args.adx) & np.isfinite(adx_v) &
        (atr_v >= args.atr) & np.isfinite(atr_v) &
        (bbw_v >= args.bbw) & np.isfinite(bbw_v)
    )
    df_feat["keep"] = keep
    df_kept = df_feat[keep].copy()

    n_base = len(df_feat)
    n_kept = len(df_kept)
    print(f"  baseline n={n_base}  filtered n={n_kept} ({n_kept/n_base*100:.1f}%)")
    print(f"  baseline stage: s1={(df_feat.final_stage==1).sum()}  "
          f"s2={(df_feat.final_stage==2).sum()}  "
          f"s3={(df_feat.final_stage==3).sum()}")
    print(f"  filtered stage: s1={(df_kept.final_stage==1).sum()}  "
          f"s2={(df_kept.final_stage==2).sum()}  "
          f"s3={(df_kept.final_stage==3).sum()}")

    # =========== 1) stage 3 vs stage 1 Cohen's d 對照 ===========
    cmp_base = _compare(df_feat, "base")
    cmp_filt = _compare(df_kept, "filt")
    cmp = cmp_base.join(cmp_filt, how="left")
    cmp["delta_|d|"] = cmp["filt_d"].abs() - cmp["base_d"].abs()
    cmp = cmp.sort_values("filt_d", key=lambda s: s.abs(), ascending=False)

    print("\n" + "=" * 90)
    print("=== Cohen's d 對照（stage 3 − stage 1，|d|>0.2 視為有效訊號） ===")
    print("=" * 90)
    print(cmp[["base_d", "filt_d", "delta_|d|",
               "base_s3_mean", "base_s1_mean",
               "filt_s3_mean", "filt_s1_mean"]].to_string(float_format="%.3f"))

    significant = cmp[cmp["filt_d"].abs() >= 0.2]
    print(f"\n→ filtered 子集裡 |d|>=0.2 的特徵數: {len(significant)} / {len(cmp)}")
    if not significant.empty:
        print(significant[["base_d", "filt_d", "delta_|d|"]].to_string(float_format="%.3f"))
    else:
        print("  （沒有任何特徵達到 |d|>=0.2 門檻 — filter 後仍無法區分 s1 / s3）")

    # =========== 2) stage 3 共同特徵：mean / 四分位 ===========
    s3_base = _stage3_common(df_feat, "base")
    s3_filt = _stage3_common(df_kept, "filt")
    s3 = s3_base.join(s3_filt, how="left")
    print("\n" + "=" * 90)
    print("=== Stage 3 共同特徵：mean / p25 / p50 / p75（baseline vs filtered） ===")
    print("=" * 90)
    print(s3.to_string(float_format="%.3f"))

    # 寫檔
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(out_dir / "cohens_d_compare.csv", float_format="%.4f")
    s3.to_csv(out_dir / "stage3_common.csv", float_format="%.4f")
    df_kept.to_csv(out_dir / "filtered_trades_with_features.csv",
                   index=False, float_format="%.6f")
    print(f"\n💾 → {out_dir}")


if __name__ == "__main__":
    main()
