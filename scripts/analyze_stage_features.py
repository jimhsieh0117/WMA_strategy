"""新增特徵：volume_ratio / r_over_atr / wma_spread_pct / hour_of_day
比對 stage1 vs stage3 的分佈差異（Cohen's d + 分位數）。

問題：在「進場前可得資訊」下，是否有特徵能事先區分 stage1（會立刻死）
與 stage3（能跟到趨勢）？

用法：
    python scripts/analyze_stage_features.py [--config configs/default.yaml]
                                              [--sample is|oos]
                                              [--window 14]   # ATR / volume average
                                              [--out results/stage_features.csv]

輸出：
    1. stdout 比較表（mean / median / Cohen's d）
    2. CSV：每筆 trade × 4 個特徵（給後續視覺化用）
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
from src.utils.config import PeriodSpec, load_config  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)


def _atr_series(df: pd.DataFrame, period: int) -> pd.Series:
    """Welles Wilder ATR（簡化版：rolling mean of TR）。"""
    h = df["high"]
    l = df["low"]
    c_prev = df["close"].shift(1)
    tr = pd.concat([
        h - l,
        (h - c_prev).abs(),
        (l - c_prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _build_features(
    trades: list,
    df: pd.DataFrame,
    timeframe_delta: pd.Timedelta,
    window: int,
) -> pd.DataFrame:
    """每筆 trade 在訊號 K（entry_ts − 1 timeframe）的特徵。"""
    atr = _atr_series(df, window)
    vol_avg = df["volume"].rolling(window=window, min_periods=window).mean()

    rows = []
    skipped = 0
    for t in trades:
        entry_ts = pd.Timestamp(t.entry_timestamp)
        signal_ts = entry_ts - timeframe_delta
        if signal_ts not in df.index:
            pos = df.index.searchsorted(signal_ts, side="right") - 1
            if pos < 0:
                skipped += 1
                continue
            signal_ts = df.index[pos]
        try:
            close = float(df["close"].loc[signal_ts])
            volume = float(df["volume"].loc[signal_ts])
            wma_f = float(df["wma_fast"].loc[signal_ts])
            wma_s = float(df["wma_slow"].loc[signal_ts])
            atr_v = float(atr.loc[signal_ts])
            vol_a = float(vol_avg.loc[signal_ts])
        except (KeyError, ValueError):
            skipped += 1
            continue
        if not (np.isfinite(close) and np.isfinite(atr_v) and np.isfinite(vol_a)
                and atr_v > 0 and vol_a > 0 and close > 0):
            skipped += 1
            continue

        r_price = abs(t.entry_price - t.stop_history[0][1])
        rows.append({
            "direction": t.direction.value,
            "entry_ts": entry_ts,
            "final_stage": int(t.final_stage),
            "peak_progress_r": float(t.peak_progress_r),
            "net_pnl": float(t.net_pnl),
            # ---- features ----
            "volume_ratio":   volume / vol_a,
            "r_over_atr":     r_price / atr_v,
            "wma_spread_pct": abs(wma_f - wma_s) / close,
            "hour_of_day":    signal_ts.hour,
            # 額外保留：給更深度分析
            "r_pct":          r_price / t.entry_price,
        })
    if skipped:
        print(f"  [info] skipped {skipped} trades (no signal-bar history)", file=sys.stderr)
    return pd.DataFrame(rows)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean(a) − mean(b)) / pooled_sd。"""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _compare_stage(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """對每個特徵：stage1 vs stage3 的 mean / median / Cohen's d。"""
    s1 = df[df.final_stage == 1]
    s3 = df[df.final_stage == 3]
    rows = []
    for f in features:
        a = s1[f].dropna().to_numpy()
        b = s3[f].dropna().to_numpy()
        rows.append({
            "feature": f,
            "s1_n": len(a),
            "s3_n": len(b),
            "s1_mean": float(np.mean(a)) if len(a) else float("nan"),
            "s3_mean": float(np.mean(b)) if len(b) else float("nan"),
            "s1_med":  float(np.median(a)) if len(a) else float("nan"),
            "s3_med":  float(np.median(b)) if len(b) else float("nan"),
            "cohens_d_(s3−s1)": _cohens_d(b, a),
        })
    return pd.DataFrame(rows)


def _hour_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """hour_of_day × final_stage 交叉表。"""
    g = df.groupby(["hour_of_day", "final_stage"]).size().unstack(fill_value=0)
    if 1 not in g.columns:
        g[1] = 0
    if 2 not in g.columns:
        g[2] = 0
    if 3 not in g.columns:
        g[3] = 0
    g = g[[1, 2, 3]].copy()
    g["total"] = g.sum(axis=1)
    g["s1%"] = g[1] / g["total"] * 100
    g["s2%"] = g[2] / g["total"] * 100
    g["s3%"] = g[3] / g["total"] * 100
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--window", type=int, default=14, help="ATR / volume avg 期間")
    parser.add_argument("--out", default="results/stage_features.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    cfg_m = dataclasses.replace(cfg, **overrides)

    print(f"running long ...  ({cfg_m.symbol} {cfg_m.timeframe} "
          f"{cfg_m.in_sample.start.date()} ~ {cfg_m.in_sample.end.date()})")
    L = run_single_strategy(cfg_m, direction="long", sample=args.sample)
    print("running short ...")
    S = run_single_strategy(cfg_m, direction="short", sample=args.sample)

    # 取得已含指標的 df：重跑 prepare_indicators 一次
    from src.data.loader import load_ohlcv
    from src.data.resampler import resample
    from src.strategy.base import prepare_indicators
    from src.strategy.types import (
        RCapParams, SignalFilterParams, StrategyParams, TrailingStopParams,
    )
    period = cfg_m.in_sample if args.sample == "is" else cfg_m.out_of_sample
    raw1m = load_ohlcv(cfg_m.source_parquet, start=period.start, end=period.end)
    df_tf = resample(raw1m, cfg_m.timeframe) if cfg_m.timeframe != "1m" else raw1m
    params = StrategyParams(
        wma_fast=cfg_m.wma_fast, wma_slow=cfg_m.wma_slow,
        entry_source=cfg_m.entry_source,  # type: ignore[arg-type]
        trailing=TrailingStopParams(
            swing_lookback=cfg_m.trailing.swing_lookback,
            stage1_slippage_buffer=cfg_m.trailing.stage1_slippage_buffer,
            stage2_normal_trigger_r=cfg_m.trailing.stage2_normal_trigger_r,
            stage2_abnormal_trigger_r=cfg_m.trailing.stage2_abnormal_trigger_r,
            stage2_buffer_r=cfg_m.trailing.stage2_buffer_r,
            stage2_pct_trigger=cfg_m.trailing.stage2_pct_trigger,
            stage3_normal_trigger_r=cfg_m.trailing.stage3_normal_trigger_r,
            stage3_abnormal_trigger_r=cfg_m.trailing.stage3_abnormal_trigger_r,
            bollinger_period=cfg_m.trailing.bollinger_period,
            bollinger_num_std=cfg_m.trailing.bollinger_num_std,
            stage3_mode=cfg_m.trailing.stage3_mode,  # type: ignore[arg-type]
            r_ladder_normal_first_trigger=cfg_m.trailing.r_ladder_normal_first_trigger,
            r_ladder_normal_step=cfg_m.trailing.r_ladder_normal_step,
            r_ladder_abnormal_first_trigger=cfg_m.trailing.r_ladder_abnormal_first_trigger,
            r_ladder_abnormal_step=cfg_m.trailing.r_ladder_abnormal_step,
            r_ladder_trigger_offset=cfg_m.trailing.r_ladder_trigger_offset,
            r_ladder_abnormal_trigger_offset=cfg_m.trailing.r_ladder_abnormal_trigger_offset,
        ),
        signal_filter=SignalFilterParams(
            mode=cfg_m.signal_filter.mode,  # type: ignore[arg-type]
            window=cfg_m.signal_filter.window,
            threshold=cfg_m.signal_filter.threshold,
            source=cfg_m.signal_filter.source,
        ),
        r_cap=RCapParams(mode=cfg_m.r_cap.mode, window=cfg_m.r_cap.window),  # type: ignore[arg-type]
    )
    df_aug = prepare_indicators(df_tf, params)
    timeframe_delta = pd.tseries.frequencies.to_offset(
        cfg_m.timeframe.replace("H", "h")  # pandas 2 用 "h"
    ).delta if False else (df_aug.index[1] - df_aug.index[0])

    feat = _build_features(L.trades + S.trades, df_aug, timeframe_delta, args.window)
    print(f"\ntotal trades with features: {len(feat)}")
    print(f"  stage1: {(feat.final_stage==1).sum()},  stage2: {(feat.final_stage==2).sum()},  stage3: {(feat.final_stage==3).sum()}")

    features = ["volume_ratio", "r_over_atr", "wma_spread_pct", "hour_of_day", "r_pct"]
    cmp = _compare_stage(feat, features)
    print("\n=== stage1 vs stage3 比較（Cohen's d 指 s3 − s1，正值代表 s3 該特徵較大）===")
    print(cmp.to_string(index=False, float_format="%.5f"))

    print("\n=== hour_of_day × stage 分佈 ===")
    hb = _hour_breakdown(feat)
    print(hb[["total", "s1%", "s2%", "s3%"]].to_string(float_format="%.2f"))
    print(f"\n  全樣本 stage 分佈：s1={hb[1].sum()/hb['total'].sum()*100:.2f}%  "
          f"s2={hb[2].sum()/hb['total'].sum()*100:.2f}%  "
          f"s3={hb[3].sum()/hb['total'].sum()*100:.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 features → {out_path}")


if __name__ == "__main__":
    main()
