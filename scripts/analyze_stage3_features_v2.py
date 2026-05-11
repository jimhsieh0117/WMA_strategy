"""V2：擴充 pre-entry 技術指標，找 stage3 vs stage1 的穩健 edge。

V1（analyze_stage3_features.py）只放了 6 個基礎特徵，跨組合最大 |d| ≈ 0.15，
無法區分。V2 加入 11 個更具 trend/regime/momentum/volatility 訊息量的指標：

  趨勢
    - adx_14            : ADX(14) 強度
    - dmi_diff          : (DI+ − DI−) × direction
    - wma_fast_slope    : (wma_fast[t] − wma_fast[t−3]) / close × direction
    - dist_sma50_pct    : (close − SMA50) / SMA50 × direction

  動能
    - rsi_14            : RSI(14)，多單高於 50 視為順勢；空鏡像（這裡保留原值）
    - macd_hist         : MACD(12,26,9) histogram × direction

  波動 regime
    - bb_width_pct      : Bollinger(20,2) 寬度 / close
    - atr_pct_rank_200  : ATR(14) 在過去 200 根 ATR 的百分位 (0..1)

  訊號結構
    - bars_since_cross  : 距上次 wma_fast / wma_slow 交叉的根數
    - bar_streak         : signal K 之前連續同色 HA bar 數（與本筆方向同色才計）

  跨時框架
    - htf_h1_aligned    : H1 WMA(4,6) 方向是否與本筆同向（0/1）

對每個特徵：stage1 vs stage3 mean / Cohen's d，4 組合並排。

用法：
    python scripts/analyze_stage3_features_v2.py [--out results/stage3_features_v2.csv]
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
from src.indicators.wma import wma  # noqa: E402
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
    "adx_14", "dmi_diff", "wma_fast_slope", "dist_sma50_pct",
    "rsi_14", "macd_hist",
    "bb_width_pct", "atr_pct_rank_200",
    "bars_since_cross", "bar_streak",
    "htf_h1_aligned",
]


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing (alpha = 1/period)。"""
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _adx_dmi(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _wilder_smooth(dx, period)
    return adx, plus_di, minus_di


def _rsi(c: pd.Series, period: int = 14) -> pd.Series:
    d = c.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    rs = _wilder_smooth(gain, period) / _wilder_smooth(loss, period).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(c: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    ema_f = c.ewm(span=fast, adjust=False).mean()
    ema_s = c.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd - signal


def _bb_width(c: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    m = c.rolling(period, min_periods=period).mean()
    s = c.rolling(period, min_periods=period).std(ddof=0)
    return (m + k * s) - (m - k * s)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    return _wilder_smooth(tr, period)


def _atr_pct_rank(atr: pd.Series, window: int = 200) -> pd.Series:
    """ATR 在過去 window 根 ATR 中的百分位 (0..1)。"""
    return atr.rolling(window, min_periods=window).rank(pct=True)


def _bars_since_cross(wf: pd.Series, ws: pd.Series) -> pd.Series:
    """距上一次 wma_fast/wma_slow 交叉的根數（含當下訊號 K = 0）。"""
    sign = np.sign(wf - ws).fillna(0)
    flip = (sign != sign.shift(1)).astype(int)
    flip.iloc[0] = 0
    # 累計：每次 flip 重置
    grp = flip.cumsum()
    return grp.groupby(grp).cumcount()


def _bar_streak(df: pd.DataFrame) -> pd.Series:
    """每根 K 線之前（含自己）連續同色 bar 數。
    +N = N 根連續綠（close > open）；−N = N 根連續紅。
    """
    color = np.sign(df["close"] - df["open"]).fillna(0).astype(int)
    out = np.zeros(len(color), dtype=int)
    streak = 0
    last = 0
    for i, v in enumerate(color.values):
        if v == 0 or v != last:
            streak = v
        else:
            streak = streak + (1 if v > 0 else -1)
        out[i] = streak
        last = v
    return pd.Series(out, index=color.index)


# --------------------------------------------------------------------------- #
# HTF 對齊（resample 1m → 1H 後算 WMA(4,6) 方向）
# --------------------------------------------------------------------------- #

def _h1_wma_direction(df1m: pd.DataFrame) -> pd.Series:
    """回傳 H1 上每個 bar close 後的 WMA(4,6) 方向：+1 / −1 / 0。
    Index 為 H1 bar timestamp（bar 起始時間）。
    """
    h1 = resample(df1m, "1H")
    fast = wma(h1["close"], 4)
    slow = wma(h1["close"], 6)
    sign = np.sign(fast - slow).fillna(0).astype(int)
    return sign


def _htf_aligned(signal_ts: pd.Timestamp, h1_dir: pd.Series, direction: int) -> int:
    """signal_ts 當下，找最近一根**已收盤**的 H1 bar 的方向。
    H1 bar at T 在 T+1H 才收盤；signal_ts 必須 > T+1H 才能用該 bar。
    """
    # 最近一根 H1 bar 起始時間 ≤ signal_ts − 1H（保證已收盤）
    cutoff = signal_ts - pd.Timedelta("1h")
    pos = h1_dir.index.searchsorted(cutoff, side="right") - 1
    if pos < 0:
        return 0
    s = int(h1_dir.iloc[pos])
    if s == 0:
        return 0
    return 1 if s == direction else 0


# --------------------------------------------------------------------------- #
# Pre-entry 特徵（在 signal K = entry K − 1 timeframe）
# --------------------------------------------------------------------------- #

def _pre_features(t, df: pd.DataFrame, tf_delta: pd.Timedelta,
                  cache: dict, h1_dir: pd.Series, direction: int) -> dict | None:
    entry_ts = pd.Timestamp(t.entry_timestamp)
    signal_ts = entry_ts - tf_delta
    if signal_ts not in df.index:
        pos = df.index.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            return None
        signal_ts = df.index[pos]
    try:
        c = float(df["close"].loc[signal_ts])
        if not (np.isfinite(c) and c > 0):
            return None

        adx = float(cache["adx"].loc[signal_ts])
        dmi_diff_raw = float(cache["plus_di"].loc[signal_ts] - cache["minus_di"].loc[signal_ts])
        wf_now = float(df["wma_fast"].loc[signal_ts])
        wf_pos = df.index.get_loc(signal_ts)
        if wf_pos < 3:
            return None
        wf_3ago = float(df["wma_fast"].iloc[wf_pos - 3])
        sma50 = float(cache["sma50"].loc[signal_ts])
        rsi_v = float(cache["rsi"].loc[signal_ts])
        macd_v = float(cache["macd_hist"].loc[signal_ts])
        bbw = float(cache["bb_width"].loc[signal_ts])
        atr_rank = float(cache["atr_rank"].loc[signal_ts])
        bars_cross = float(cache["bars_cross"].loc[signal_ts])
        bar_streak_v = float(cache["bar_streak"].loc[signal_ts])
    except (KeyError, ValueError):
        return None

    if not all(np.isfinite(x) for x in (adx, sma50, rsi_v, macd_v, bbw, atr_rank)):
        return None

    htf = _htf_aligned(signal_ts, h1_dir, direction)

    return {
        "adx_14":           adx,
        "dmi_diff":         dmi_diff_raw * direction,
        "wma_fast_slope":   (wf_now - wf_3ago) / c * direction,
        "dist_sma50_pct":   (c - sma50) / sma50 * direction,
        "rsi_14":           rsi_v if direction == 1 else 100.0 - rsi_v,  # 鏡像：空單高 = 順勢
        "macd_hist":        macd_v * direction,
        "bb_width_pct":     bbw / c,
        "atr_pct_rank_200": atr_rank,
        "bars_since_cross": bars_cross,
        "bar_streak":        bar_streak_v * direction,
        "htf_h1_aligned":   htf,
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

    # 預先計算所有指標（一次過）
    adx, plus_di, minus_di = _adx_dmi(df_aug, 14)
    cache = {
        "adx":         adx,
        "plus_di":     plus_di,
        "minus_di":    minus_di,
        "sma50":       df_aug["close"].rolling(50, min_periods=50).mean(),
        "rsi":         _rsi(df_aug["close"], 14),
        "macd_hist":   _macd_hist(df_aug["close"]),
        "bb_width":    _bb_width(df_aug["close"]),
        "atr_rank":    _atr_pct_rank(_atr(df_aug, 14), 200),
        "bars_cross":  _bars_since_cross(df_aug["wma_fast"], df_aug["wma_slow"]),
        "bar_streak":   _bar_streak(df_aug),
    }
    h1_dir = _h1_wma_direction(df1m)
    tf_delta = df_aug.index[1] - df_aug.index[0]

    rows = []
    skipped = 0
    for t in L.trades + S.trades:
        direction = 1 if t.direction.value == "long" else -1
        feat = _pre_features(t, df_aug, tf_delta, cache, h1_dir, direction)
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
    parser.add_argument("--out", default="results/stage3_features_v2.csv")
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
            print(f"  [warn] no trades for {name}")
            continue
        n = len(feat)
        n1 = (feat.final_stage == 1).sum()
        n3 = (feat.final_stage == 3).sum()
        print(f"  trades={n}  s1={n1}  s3={n3}  s3%={n3/n*100:.2f}%")
        summary_rows.append({"combo": name, "n": n, "s1": n1, "s3": n3, "s3%": n3/n*100})
        per_combo[name] = _stage_compare(feat, FEATURES)
        feature_dfs[name] = feat.assign(combo=name)

    if not per_combo:
        print("no combos")
        return

    summary = pd.DataFrame(summary_rows).set_index("combo")
    print("\n=== Summary ===")
    print(summary.to_string(float_format="%.2f"))

    d_tbl = _side_by_side(per_combo, FEATURES, "d_(s3−s1)")
    print("\n=== Cohen's d (s3 − s1)  ─ 跨 4 組合 ===")
    print(d_tbl.to_string(float_format="%+.3f"))

    s3_tbl = _side_by_side(per_combo, FEATURES, "s3_mean")
    s1_tbl = _side_by_side(per_combo, FEATURES, "s1_mean")
    print("\n=== s3 mean ===")
    print(s3_tbl.to_string(float_format="%.4f"))
    print("\n=== s1 mean ===")
    print(s1_tbl.to_string(float_format="%.4f"))

    print("\n=== 穩健 edge（4 組合 d 同號且全 |d| > 0.2）===")
    consistent = d_tbl[(d_tbl > 0.2).all(axis=1) | (d_tbl < -0.2).all(axis=1)]
    if consistent.empty:
        print("  （無）")
    else:
        print(consistent.to_string(float_format="%+.3f"))

    print("\n=== 中度 edge（4 組合 d 同號且全 |d| > 0.1）===")
    medium = d_tbl[(d_tbl > 0.1).all(axis=1) | (d_tbl < -0.1).all(axis=1)]
    if medium.empty:
        print("  （無）")
    else:
        print(medium.to_string(float_format="%+.3f"))

    out = pd.concat(feature_dfs.values(), ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 raw features → {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()
