"""4 組合（5m / 15m × WMA(2,4) / WMA(4,6)）stage 3 共通特徵分析。

兩類特徵：
  pre-entry（訊號 K 可得資訊）
    - r_pct          : R_price / entry_price
    - r_over_atr     : R_price / ATR(window)
    - wma_spread_pct : |wma_fast − wma_slow| / close
    - volume_ratio   : volume / volume_avg(window)
    - body_pct       : signal K 方向化 body 佔 range（多: (close−open)/range；空鏡像）
    - hour_of_day    : signal K 的 UTC hour

  post-entry（進場後 N 根原始 K，R 為單位）
    - first_close_r  : 進場後第 1 根 close 相對 entry / R
    - first_peak_r   : 進場後第 1 根 max favorable / R
    - first_mae_r    : 進場後第 1 根 max adverse / R
    - mfe_5bar_r     : 進場後 5 根 cumulative max favorable / R
    - mae_5bar_r     : 進場後 5 根 cumulative max adverse / R
    - bars_to_0.5R / 1R / 2R : 累計 MFE 達標所用 bar 數（20 bar 內，未達 = NaN）
    - hold_bars      : (exit_ts − entry_ts) / timeframe
    - consec_pos_close_3 : 前 3 根 K 收盤同向次數（多: close > entry；空鏡像）

對每個特徵：stage1 vs stage3 mean / median / Cohen's d，4 組合並排。
跨組合一致 edge → 穩健；只在某組合出現 → 過擬合。

用法：
    python scripts/analyze_stage3_features.py [--out results/stage3_features.csv]
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

PRE_FEATURES = [
    "r_pct", "r_over_atr", "wma_spread_pct", "volume_ratio",
    "body_pct", "hour_of_day",
]
POST_FEATURES = [
    "first_close_r", "first_peak_r", "first_mae_r",
    "mfe_5bar_r", "mae_5bar_r",
    "bars_to_0.5R", "bars_to_1R", "bars_to_2R",
    "hold_bars", "consec_pos_close_3",
]


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# --------------------------------------------------------------------------- #
# Pre-entry 特徵（在 signal K = entry K − 1 timeframe）
# --------------------------------------------------------------------------- #

def _pre_features(t, df: pd.DataFrame, tf_delta: pd.Timedelta, atr: pd.Series,
                  vol_avg: pd.Series, direction: int) -> dict | None:
    entry_ts = pd.Timestamp(t.entry_timestamp)
    signal_ts = entry_ts - tf_delta
    if signal_ts not in df.index:
        pos = df.index.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            return None
        signal_ts = df.index[pos]
    try:
        o = float(df["open"].loc[signal_ts])
        h = float(df["high"].loc[signal_ts])
        l = float(df["low"].loc[signal_ts])
        c = float(df["close"].loc[signal_ts])
        v = float(df["volume"].loc[signal_ts])
        wf = float(df["wma_fast"].loc[signal_ts])
        ws = float(df["wma_slow"].loc[signal_ts])
        a = float(atr.loc[signal_ts])
        va = float(vol_avg.loc[signal_ts])
    except (KeyError, ValueError):
        return None
    if not all(np.isfinite(x) and x > 0 for x in (c, a, va)):
        return None
    rng = h - l
    if rng <= 0:
        return None

    r_price = abs(t.entry_price - t.stop_history[0][1])
    body = (c - o) * direction  # 多: close-open；空: open-close
    return {
        "r_pct":          r_price / t.entry_price,
        "r_over_atr":     r_price / a,
        "wma_spread_pct": abs(wf - ws) / c,
        "volume_ratio":   v / va,
        "body_pct":       body / rng,
        "hour_of_day":    int(signal_ts.hour),
    }


# --------------------------------------------------------------------------- #
# Post-entry 特徵（從進場 K 起向後 max_bars 根原始 K）
# --------------------------------------------------------------------------- #

def _post_features(t, df: pd.DataFrame, tf_delta: pd.Timedelta,
                   max_bars: int = 20) -> dict | None:
    entry_ts = pd.Timestamp(t.entry_timestamp)
    exit_ts = pd.Timestamp(t.exit_timestamp)

    pos = df.index.searchsorted(entry_ts)
    if pos >= len(df) or df.index[pos] != entry_ts:
        # 實際上 entry_ts 應該嚴格是 bar 的 timestamp；對不上就略過
        return None
    end = min(pos + max_bars, len(df))
    bars = df.iloc[pos:end]
    if len(bars) == 0:
        return None

    direction = 1 if t.direction.value == "long" else -1
    entry = float(t.entry_price)
    R = abs(entry - float(t.stop_history[0][1]))
    if R <= 0:
        return None

    if direction == 1:
        bar_fav = (bars["high"] - entry) / R           # 多
        bar_adv = (entry - bars["low"]) / R
        bar_close = (bars["close"] - entry) / R
    else:
        bar_fav = (entry - bars["low"]) / R            # 空鏡像
        bar_adv = (bars["high"] - entry) / R
        bar_close = (entry - bars["close"]) / R

    cum_mfe = bar_fav.cummax()
    cum_mae = bar_adv.cummax()

    def bars_to(thr: float) -> float:
        hit = cum_mfe[cum_mfe >= thr]
        if len(hit) == 0:
            return float("nan")
        return float(cum_mfe.index.get_loc(hit.index[0]))

    pos_closes = (bar_close > 0).astype(int).tolist()
    consec = 0
    for x in pos_closes[:3]:
        if x == 1:
            consec += 1
        else:
            break

    hold_bars = (exit_ts - entry_ts) / tf_delta

    five = min(4, len(cum_mfe) - 1)
    return {
        "first_close_r": float(bar_close.iloc[0]),
        "first_peak_r":  float(bar_fav.iloc[0]),
        "first_mae_r":   float(bar_adv.iloc[0]),
        "mfe_5bar_r":    float(cum_mfe.iloc[five]),
        "mae_5bar_r":    float(cum_mae.iloc[five]),
        "bars_to_0.5R":  bars_to(0.5),
        "bars_to_1R":    bars_to(1.0),
        "bars_to_2R":    bars_to(2.0),
        "hold_bars":     float(hold_bars),
        "consec_pos_close_3": int(consec),
    }


# --------------------------------------------------------------------------- #
# 一個組合的回測 + 特徵建構
# --------------------------------------------------------------------------- #

def _build_one(timeframe: str, wma_fast: int, wma_slow: int,
               base_cfg, period: PeriodSpec, window: int) -> pd.DataFrame:
    cfg = dataclasses.replace(
        base_cfg,
        timeframe=timeframe,
        wma_fast=wma_fast,
        wma_slow=wma_slow,
        in_sample=period,
        show_progress=False,
        log_level="WARNING",
        # 特徵分析：取消 hour blacklist 以保留全樣本
        entry_hour_blacklist=(),
    )
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")

    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, timeframe) if timeframe != "1m" else df1m

    # signal-K 用 indicators 增強版（要 wma_fast/wma_slow 欄）；用 prepare_indicators
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
    tf_delta = df_aug.index[1] - df_aug.index[0]
    atr = _atr(df_aug, window)
    vol_avg = df_aug["volume"].rolling(window=window, min_periods=window).mean()

    rows = []
    skipped = 0
    for t in L.trades + S.trades:
        direction = 1 if t.direction.value == "long" else -1
        pre = _pre_features(t, df_aug, tf_delta, atr, vol_avg, direction)
        post = _post_features(t, df_aug, tf_delta, max_bars=20)
        if pre is None or post is None:
            skipped += 1
            continue
        rows.append({
            "final_stage": int(t.final_stage),
            "net_pnl": float(t.net_pnl),
            "direction": t.direction.value,
            **pre,
            **post,
        })
    if skipped:
        print(f"  [info] {timeframe} WMA({wma_fast},{wma_slow}): skipped {skipped} trades")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 比較 / 輸出
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
            "s1_med":  float(np.median(a)) if len(a) else float("nan"),
            "s3_med":  float(np.median(b)) if len(b) else float("nan"),
            "d_(s3−s1)": _cohens_d(b, a),
        })
    return pd.DataFrame(rows).set_index("feature")


def _side_by_side(per_combo: dict[str, pd.DataFrame], features: list[str],
                  metric: str = "d_(s3−s1)") -> pd.DataFrame:
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
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--out", default="results/stage3_features.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    base_cfg = load_config(args.config)

    per_combo: dict[str, pd.DataFrame] = {}
    feature_dfs: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for tf, wf, ws in COMBOS:
        name = f"{tf}_W{wf}-{ws}"
        print(f"\n>>> running {name} ...")
        feat = _build_one(tf, wf, ws, base_cfg, DEFAULT_PERIOD, args.window)
        if feat.empty:
            print(f"  [warn] no trades for {name}")
            continue
        n = len(feat)
        n1, n2, n3 = (feat.final_stage == 1).sum(), (feat.final_stage == 2).sum(), (feat.final_stage == 3).sum()
        pnl = feat["net_pnl"].sum()
        print(f"  trades={n}  s1={n1}  s2={n2}  s3={n3}  s3%={n3/n*100:.2f}%  PnL={pnl:+.2f}")
        summary_rows.append({"combo": name, "n": n, "s1": n1, "s2": n2, "s3": n3,
                             "s3%": n3/n*100, "pnl": pnl})

        per_combo[name] = _stage_compare(feat, PRE_FEATURES + POST_FEATURES)
        feature_dfs[name] = feat.assign(combo=name)

    if not per_combo:
        print("no combos produced features")
        return

    # 1) 整體 summary
    summary = pd.DataFrame(summary_rows).set_index("combo")
    print("\n=== Summary（4 組合）===")
    print(summary.to_string(float_format="%.2f"))

    # 2) Pre-entry: Cohen's d 並排
    pre_d = _side_by_side(per_combo, PRE_FEATURES, "d_(s3−s1)")
    print("\n=== Pre-entry  Cohen's d (s3 − s1)  ─ 跨組合 ===")
    print(pre_d.to_string(float_format="%+.3f"))

    # 3) Post-entry: Cohen's d 並排
    post_d = _side_by_side(per_combo, POST_FEATURES, "d_(s3−s1)")
    print("\n=== Post-entry  Cohen's d (s3 − s1)  ─ 跨組合 ===")
    print(post_d.to_string(float_format="%+.3f"))

    # 4) Pre-entry: s3 mean 並排（看 stage 3 的「絕對水準」）
    pre_s3 = _side_by_side(per_combo, PRE_FEATURES, "s3_mean")
    print("\n=== Pre-entry  s3 mean  ─ 跨組合 ===")
    print(pre_s3.to_string(float_format="%.4f"))

    # 5) Post-entry: s3 mean 並排
    post_s3 = _side_by_side(per_combo, POST_FEATURES, "s3_mean")
    print("\n=== Post-entry  s3 mean  ─ 跨組合 ===")
    print(post_s3.to_string(float_format="%.4f"))

    # 6) 跨組合一致 edge：4 個 d 同號且 |d| > 0.2
    print("\n=== 穩健 edge（4 組合 Cohen's d 同號且全 |d| > 0.2）===")
    all_d = pd.concat([pre_d, post_d])
    consistent = all_d[(all_d > 0.2).all(axis=1) | (all_d < -0.2).all(axis=1)]
    if consistent.empty:
        print("  （無）")
    else:
        print(consistent.to_string(float_format="%+.3f"))

    # 7) 落地 raw 特徵
    out = pd.concat(feature_dfs.values(), ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 raw features → {out_path}  (rows={len(out)})")


if __name__ == "__main__":
    main()
