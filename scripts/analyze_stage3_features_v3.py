"""V3：WMA 偏移類 + Wave Trend + MACD divergence。

V1 / V2 共 17 個特徵都未過 |d|=0.2。V3 換維度測：

  WMA 偏移（B 類：曲率 / 加速度）
    - wma_slow_curvature   : (Δ²wma_slow) / close × direction
    - wma_fast_curvature   : (Δ²wma_fast) / close × direction
    - wma_spread_accel     : Δ(wma_fast − wma_slow) / close × direction

  WMA 偏移（C 類：價格距 WMA）
    - close_dist_wma_slow  : (close − wma_slow) / wma_slow × direction
    - close_dist_wma_fast  : (close − wma_fast) / wma_fast × direction
    - bar_straddles_fast   : 訊號 K 的 [low, high] 是否包含 wma_fast (0/1)

  Wave Trend (LazyBear, n1=10, n2=21)
    - wt1                  : WT1 × direction（順向 = 正）
    - wt_zone              : 區間 −1/0/+1（≤−60 OS / 中性 / ≥+60 OB）× direction
                              對 LONG：在 OS=+1 視為 favorable；在 OB=−1
                              對 SHORT 鏡像

  MACD 背離（4 根窗）
    - macd_div_signed      : sign(macd_chg_4bar) − sign(price_chg_4bar) × direction
                              長：價跌但 MACD 升 → +2 = bullish div = 進場 favorable

對每個特徵：stage1 vs stage3 mean / Cohen's d，4 組合並排。

用法：
    python scripts/analyze_stage3_features_v3.py [--out results/stage3_features_v3.csv]
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
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)

COMBOS = [
    ("5m",  2, 4),
    ("5m",  4, 6),
    ("15m", 2, 4),
    ("15m", 4, 6),
]

FEATURES = [
    "wma_slow_curvature", "wma_fast_curvature", "wma_spread_accel",
    "close_dist_wma_slow", "close_dist_wma_fast", "bar_straddles_fast",
    "wt1", "wt_zone",
    "macd_div_signed",
]


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wave_trend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.Series:
    """LazyBear Wave Trend WT1。"""
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = _ema(ap, n1)
    d = _ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = _ema(ci, n2)
    return wt1


def _macd(c: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return _ema(c, fast) - _ema(c, slow)


# --------------------------------------------------------------------------- #
# Pre-entry 特徵
# --------------------------------------------------------------------------- #

def _pre_features(t, df: pd.DataFrame, tf_delta: pd.Timedelta,
                  cache: dict, direction: int) -> dict | None:
    entry_ts = pd.Timestamp(t.entry_timestamp)
    signal_ts = entry_ts - tf_delta
    if signal_ts not in df.index:
        pos = df.index.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            return None
        signal_ts = df.index[pos]
    pos = df.index.get_loc(signal_ts)
    if pos < 4:
        return None
    try:
        c = float(df["close"].loc[signal_ts])
        h = float(df["high"].loc[signal_ts])
        l = float(df["low"].loc[signal_ts])
        wf = float(df["wma_fast"].loc[signal_ts])
        ws = float(df["wma_slow"].loc[signal_ts])
        wf_1 = float(df["wma_fast"].iloc[pos - 1])
        wf_2 = float(df["wma_fast"].iloc[pos - 2])
        ws_1 = float(df["wma_slow"].iloc[pos - 1])
        ws_2 = float(df["wma_slow"].iloc[pos - 2])
        wt = float(cache["wt1"].loc[signal_ts])
        macd_now = float(cache["macd"].loc[signal_ts])
        macd_4ago = float(cache["macd"].iloc[pos - 4])
        c_4ago = float(df["close"].iloc[pos - 4])
    except (KeyError, ValueError):
        return None

    if not all(np.isfinite(x) and x != 0 for x in (c, ws, wf)):
        return None
    if not all(np.isfinite(x) for x in (wt, macd_now, macd_4ago, c_4ago)):
        return None

    # 二階差分
    wma_slow_d2 = (ws - 2 * ws_1 + ws_2)
    wma_fast_d2 = (wf - 2 * wf_1 + wf_2)
    spread_d1 = (wf - ws) - (wf_1 - ws_1)

    # WT zone：±60 邊界
    if wt >= 60:
        zone = -1  # overbought：對多單不利
    elif wt <= -60:
        zone = +1  # oversold：對多單有利
    else:
        zone = 0

    # MACD divergence：sign 差
    s_macd = np.sign(macd_now - macd_4ago)
    s_price = np.sign(c - c_4ago)

    return {
        "wma_slow_curvature":  (wma_slow_d2 / c) * direction,
        "wma_fast_curvature":  (wma_fast_d2 / c) * direction,
        "wma_spread_accel":    (spread_d1 / c) * direction,
        "close_dist_wma_slow": ((c - ws) / ws) * direction,
        "close_dist_wma_fast": ((c - wf) / wf) * direction,
        "bar_straddles_fast":  int(l <= wf <= h),
        "wt1":                 wt * direction,
        "wt_zone":             zone * direction,
        "macd_div_signed":     float(s_macd - s_price) * direction,
    }


# --------------------------------------------------------------------------- #
# 一個組合
# --------------------------------------------------------------------------- #

def _build_one(timeframe: str, wma_fast: int, wma_slow: int,
               base_cfg, period: PeriodSpec) -> pd.DataFrame:
    cfg = dataclasses.replace(
        base_cfg,
        timeframe=timeframe, wma_fast=wma_fast, wma_slow=wma_slow,
        in_sample=period, show_progress=False, log_level="WARNING",
        entry_hour_blacklist=(),
    )
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")

    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, timeframe) if timeframe != "1m" else df1m

    from src.strategy.base import prepare_indicators
    from src.strategy.types import (
        RCapParams, SignalFilterParams, StrategyParams, TrailingStopParams,
    )
    params = StrategyParams(
        wma_fast=wma_fast, wma_slow=wma_slow,
        entry_source=cfg.entry_source,  # type: ignore[arg-type]
        trailing=TrailingStopParams(
            swing_lookback=cfg.trailing.swing_lookback,
            stage1_slippage_buffer=cfg.trailing.stage1_slippage_buffer,
            stage2_normal_trigger_r=cfg.trailing.stage2_normal_trigger_r,
            stage2_abnormal_trigger_r=cfg.trailing.stage2_abnormal_trigger_r,
            stage2_buffer_r=cfg.trailing.stage2_buffer_r,
            stage2_pct_trigger=cfg.trailing.stage2_pct_trigger,
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
        ),
        signal_filter=SignalFilterParams(
            mode=cfg.signal_filter.mode,  # type: ignore[arg-type]
            window=cfg.signal_filter.window,
            threshold=cfg.signal_filter.threshold,
            source=cfg.signal_filter.source,
        ),
        r_cap=RCapParams(mode=cfg.r_cap.mode, window=cfg.r_cap.window),  # type: ignore[arg-type]
    )
    df_aug = prepare_indicators(df, params)

    cache = {
        "wt1":  _wave_trend(df_aug),
        "macd": _macd(df_aug["close"]),
    }
    tf_delta = df_aug.index[1] - df_aug.index[0]

    rows = []
    skipped = 0
    for t in L.trades + S.trades:
        direction = 1 if t.direction.value == "long" else -1
        feat = _pre_features(t, df_aug, tf_delta, cache, direction)
        if feat is None:
            skipped += 1
            continue
        rows.append({
            "final_stage": int(t.final_stage),
            "net_pnl":     float(t.net_pnl),
            "direction":   t.direction.value,
            **feat,
        })
    if skipped:
        print(f"  [info] {timeframe} W({wma_fast},{wma_slow}): skipped {skipped} trades")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 比較
# --------------------------------------------------------------------------- #

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _stage_compare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    s1 = df[df.final_stage == 1]
    s3 = df[df.final_stage == 3]
    rows = []
    for f in features:
        a = s1[f].dropna().to_numpy()
        b = s3[f].dropna().to_numpy()
        rows.append({
            "feature": f,
            "s1_mean": float(np.mean(a)) if len(a) else float("nan"),
            "s3_mean": float(np.mean(b)) if len(b) else float("nan"),
            "d_(s3−s1)": _cohens_d(b, a),
        })
    return pd.DataFrame(rows).set_index("feature")


def _side_by_side(per_combo: dict[str, pd.DataFrame], features: list[str],
                  metric: str) -> pd.DataFrame:
    out = pd.DataFrame(index=features)
    for name, tbl in per_combo.items():
        out[name] = tbl[metric].reindex(features)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/stage3_features_v3.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    base_cfg = load_config(args.config)

    per_combo: dict[str, pd.DataFrame] = {}
    feature_dfs: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for tf, wf, ws in COMBOS:
        name = f"{tf}_W{wf}-{ws}"
        print(f"\n>>> running {name} ...")
        feat = _build_one(tf, wf, ws, base_cfg, DEFAULT_PERIOD)
        if feat.empty:
            continue
        n = len(feat)
        n1 = (feat.final_stage == 1).sum()
        n3 = (feat.final_stage == 3).sum()
        print(f"  trades={n}  s1={n1}  s3={n3}  s3%={n3/n*100:.2f}%")
        summary_rows.append({"combo": name, "n": n, "s1": n1, "s3": n3, "s3%": n3/n*100})
        per_combo[name] = _stage_compare(feat, FEATURES)
        feature_dfs[name] = feat.assign(combo=name)

    if not per_combo:
        return

    summary = pd.DataFrame(summary_rows).set_index("combo")
    print("\n=== Summary ===")
    print(summary.to_string(float_format="%.2f"))

    d_tbl = _side_by_side(per_combo, FEATURES, "d_(s3−s1)")
    print("\n=== Cohen's d (s3 − s1) ─ 跨 4 組合 ===")
    print(d_tbl.to_string(float_format="%+.3f"))

    s3_tbl = _side_by_side(per_combo, FEATURES, "s3_mean")
    s1_tbl = _side_by_side(per_combo, FEATURES, "s1_mean")
    print("\n=== s3 mean ===")
    print(s3_tbl.to_string(float_format="%.5f"))
    print("\n=== s1 mean ===")
    print(s1_tbl.to_string(float_format="%.5f"))

    print("\n=== 穩健 edge（4 組合 d 同號且全 |d| > 0.2）===")
    consistent = d_tbl[(d_tbl > 0.2).all(axis=1) | (d_tbl < -0.2).all(axis=1)]
    print(consistent.to_string(float_format="%+.3f") if not consistent.empty else "  （無）")

    print("\n=== 中度 edge（4 組合 d 同號且全 |d| > 0.1）===")
    medium = d_tbl[(d_tbl > 0.1).all(axis=1) | (d_tbl < -0.1).all(axis=1)]
    print(medium.to_string(float_format="%+.3f") if not medium.empty else "  （無）")

    out = pd.concat(feature_dfs.values(), ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 raw features → {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()
