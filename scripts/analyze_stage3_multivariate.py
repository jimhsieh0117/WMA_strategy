"""多變量 stage3 vs stage1 分類器。

Target  : final_stage == 3 (positive) vs final_stage == 1 (negative)；stage 2 排除
Features: 合併 v1 + v2 + v3 的 24 個 pre-entry 特徵
Model   : Logistic Regression（標準化）+ Gradient Boosting
Split   : train = 2023 全年 / test = 2024 全年（皆在 IS 範圍）

對 4 組合（5m/15m × WMA(2,4)/(4,6)）逐一 train/test，輸出：
  - train / test AUC
  - top feature importance（LR coef、GBM feature_importance）
  - test set 三段 quantile（top/mid/bot 30%）的實際 stage 3 命中率

判斷：
  test AUC > 0.55 跨 4 組合 → 多變量還有可挖
  test AUC ≈ 0.50          → 蓋章「pre-entry 無 edge」

用法：
    python scripts/analyze_stage3_multivariate.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.indicators.wma import wma as _wma_fn  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# 訓練 / 測試切點：兩段都在原 IS 範圍內，避免動到 2025+ OOS
TRAIN_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2023-12-31"),
)
TEST_PERIOD = PeriodSpec(
    start=pd.Timestamp("2024-01-01"),
    end=pd.Timestamp("2024-12-31"),
)
# 為了節省時間：載入一次涵蓋兩段的 1m，然後用 entry_ts 切
FULL_PERIOD = PeriodSpec(
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
    # v1 pre
    "r_pct", "r_over_atr", "wma_spread_pct", "volume_ratio", "body_pct", "hour_of_day",
    # v2
    "adx_14", "dmi_diff", "wma_fast_slope", "dist_sma50_pct",
    "rsi_14", "macd_hist", "bb_width_pct", "atr_pct_rank_200",
    "bar_streak", "htf_h1_aligned",
    # v3
    "wma_slow_curvature", "wma_fast_curvature", "wma_spread_accel",
    "close_dist_wma_slow", "close_dist_wma_fast",
    "wt1", "wt_zone", "macd_div_signed",
]


# --------------------------------------------------------------------------- #
# 指標
# --------------------------------------------------------------------------- #

def _wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    return _wilder(tr, period)


def _adx_dmi(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    up = h.diff(); dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr
    minus_di = 100 * _wilder(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, period), plus_di, minus_di


def _rsi(c: pd.Series, period: int = 14) -> pd.Series:
    d = c.diff()
    rs = _wilder(d.clip(lower=0), period) / _wilder((-d).clip(lower=0), period).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _macd_hist(c: pd.Series) -> pd.Series:
    macd = _ema(c, 12) - _ema(c, 26)
    return macd - _ema(macd, 9)


def _macd_line(c: pd.Series) -> pd.Series:
    return _ema(c, 12) - _ema(c, 26)


def _bb_width(c: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    s = c.rolling(period, min_periods=period).std(ddof=0)
    return 4 * k * s / 2  # = 2*k*s = 上下界距離；簡化為標準差倍


def _atr_pct_rank(atr: pd.Series, window: int = 200) -> pd.Series:
    return atr.rolling(window, min_periods=window).rank(pct=True)


def _bar_streak(df: pd.DataFrame) -> pd.Series:
    color = np.sign(df["close"] - df["open"]).fillna(0).astype(int)
    out = np.zeros(len(color), dtype=int); streak = 0; last = 0
    for i, v in enumerate(color.values):
        if v == 0 or v != last:
            streak = v
        else:
            streak += 1 if v > 0 else -1
        out[i] = streak; last = v
    return pd.Series(out, index=color.index)


def _wave_trend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.Series:
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = _ema(ap, n1)
    d = _ema((ap - esa).abs(), n1)
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    return _ema(ci, n2)


def _h1_wma_dir(df1m: pd.DataFrame) -> pd.Series:
    h1 = resample(df1m, "1H")
    sign = np.sign(_wma_fn(h1["close"], 4) - _wma_fn(h1["close"], 6)).fillna(0).astype(int)
    return sign


# --------------------------------------------------------------------------- #
# 特徵建構
# --------------------------------------------------------------------------- #

def _pre_features(t, df: pd.DataFrame, tf_delta: pd.Timedelta,
                  cache: dict, h1_dir: pd.Series, direction: int) -> dict | None:
    entry_ts = pd.Timestamp(t.entry_timestamp)
    signal_ts = entry_ts - tf_delta
    if signal_ts not in df.index:
        pos_s = df.index.searchsorted(signal_ts, side="right") - 1
        if pos_s < 0:
            return None
        signal_ts = df.index[pos_s]
    pos = df.index.get_loc(signal_ts)
    if pos < 4:
        return None
    try:
        o = float(df["open"].loc[signal_ts])
        h = float(df["high"].loc[signal_ts])
        l = float(df["low"].loc[signal_ts])
        c = float(df["close"].loc[signal_ts])
        v = float(df["volume"].loc[signal_ts])
        wf = float(df["wma_fast"].loc[signal_ts])
        ws = float(df["wma_slow"].loc[signal_ts])
        wf_1 = float(df["wma_fast"].iloc[pos - 1])
        wf_2 = float(df["wma_fast"].iloc[pos - 2])
        ws_1 = float(df["wma_slow"].iloc[pos - 1])
        ws_2 = float(df["wma_slow"].iloc[pos - 2])
        wf_3 = float(df["wma_fast"].iloc[pos - 3])
        a = float(cache["atr"].loc[signal_ts])
        va = float(cache["vol_avg"].loc[signal_ts])
        adx = float(cache["adx"].loc[signal_ts])
        plus_di = float(cache["plus_di"].loc[signal_ts])
        minus_di = float(cache["minus_di"].loc[signal_ts])
        sma50 = float(cache["sma50"].loc[signal_ts])
        rsi_v = float(cache["rsi"].loc[signal_ts])
        macd_h = float(cache["macd_hist"].loc[signal_ts])
        macd_now = float(cache["macd_line"].loc[signal_ts])
        macd_4ago = float(cache["macd_line"].iloc[pos - 4])
        c_4ago = float(df["close"].iloc[pos - 4])
        bbw = float(cache["bb_width"].loc[signal_ts])
        atr_rank = float(cache["atr_rank"].loc[signal_ts])
        bar_streak_v = float(cache["bar_streak"].loc[signal_ts])
        wt = float(cache["wt1"].loc[signal_ts])
    except (KeyError, ValueError):
        return None

    if not all(np.isfinite(x) and x > 0 for x in (c, a, va, ws, wf)):
        return None
    rng = h - l
    if rng <= 0:
        return None
    if not all(np.isfinite(x) for x in (adx, sma50, rsi_v, macd_h, bbw, atr_rank, wt,
                                         macd_now, macd_4ago, c_4ago)):
        return None

    r_price = abs(t.entry_price - t.stop_history[0][1])

    cutoff = signal_ts - pd.Timedelta("1h")
    pos_h = h1_dir.index.searchsorted(cutoff, side="right") - 1
    if pos_h < 0:
        htf = 0
    else:
        s = int(h1_dir.iloc[pos_h])
        htf = 1 if (s != 0 and s == direction) else 0

    if wt >= 60:
        zone = -1
    elif wt <= -60:
        zone = +1
    else:
        zone = 0

    return {
        # v1 pre
        "r_pct":          r_price / t.entry_price,
        "r_over_atr":     r_price / a,
        "wma_spread_pct": abs(wf - ws) / c,
        "volume_ratio":   v / va,
        "body_pct":       (c - o) * direction / rng,
        "hour_of_day":    int(signal_ts.hour),
        # v2
        "adx_14":           adx,
        "dmi_diff":         (plus_di - minus_di) * direction,
        "wma_fast_slope":   (wf - wf_3) / c * direction,
        "dist_sma50_pct":   (c - sma50) / sma50 * direction,
        "rsi_14":           rsi_v if direction == 1 else 100.0 - rsi_v,
        "macd_hist":        macd_h * direction,
        "bb_width_pct":     bbw / c,
        "atr_pct_rank_200": atr_rank,
        "bar_streak":        bar_streak_v * direction,
        "htf_h1_aligned":   htf,
        # v3
        "wma_slow_curvature": (ws - 2 * ws_1 + ws_2) / c * direction,
        "wma_fast_curvature": (wf - 2 * wf_1 + wf_2) / c * direction,
        "wma_spread_accel":   ((wf - ws) - (wf_1 - ws_1)) / c * direction,
        "close_dist_wma_slow": (c - ws) / ws * direction,
        "close_dist_wma_fast": (c - wf) / wf * direction,
        "wt1":            wt * direction,
        "wt_zone":        zone * direction,
        "macd_div_signed": float(np.sign(macd_now - macd_4ago) - np.sign(c - c_4ago)) * direction,
    }


# --------------------------------------------------------------------------- #
# 一個組合：build features (含 entry_ts) + 訓練評估
# --------------------------------------------------------------------------- #

def _build_features(timeframe: str, wma_fast: int, wma_slow: int,
                    base_cfg) -> pd.DataFrame:
    cfg = dataclasses.replace(
        base_cfg,
        timeframe=timeframe, wma_fast=wma_fast, wma_slow=wma_slow,
        in_sample=FULL_PERIOD,
        show_progress=False, log_level="WARNING",
        entry_hour_blacklist=(),
    )
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")

    df1m = load_ohlcv(cfg.source_parquet, start=FULL_PERIOD.start, end=FULL_PERIOD.end)
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
            window=cfg.signal_filter.window, threshold=cfg.signal_filter.threshold,
            source=cfg.signal_filter.source,
        ),
        r_cap=RCapParams(mode=cfg.r_cap.mode, window=cfg.r_cap.window),  # type: ignore[arg-type]
    )
    df_aug = prepare_indicators(df, params)

    adx, plus_di, minus_di = _adx_dmi(df_aug, 14)
    cache = {
        "atr":         _atr(df_aug, 14),
        "vol_avg":     df_aug["volume"].rolling(14, min_periods=14).mean(),
        "adx":         adx, "plus_di": plus_di, "minus_di": minus_di,
        "sma50":       df_aug["close"].rolling(50, min_periods=50).mean(),
        "rsi":         _rsi(df_aug["close"], 14),
        "macd_hist":   _macd_hist(df_aug["close"]),
        "macd_line":   _macd_line(df_aug["close"]),
        "bb_width":    _bb_width(df_aug["close"]),
        "atr_rank":    _atr_pct_rank(_atr(df_aug, 14), 200),
        "bar_streak":   _bar_streak(compute_ha(df_aug)),
        "wt1":         _wave_trend(df_aug),
    }
    h1_dir = _h1_wma_dir(df1m)
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
            "entry_ts":    pd.Timestamp(t.entry_timestamp),
            "final_stage": int(t.final_stage),
            "peak_progress_r": float(t.peak_progress_r),
            "net_pnl":     float(t.net_pnl),
            "direction":   t.direction.value,
            **feat,
        })
    if skipped:
        print(f"  [info] skipped {skipped}")
    return pd.DataFrame(rows)


def _train_eval(df_feat: pd.DataFrame) -> dict:
    """train=2023, test=2024；只用 stage 1 與 stage 3。"""
    df_feat = df_feat[df_feat.final_stage.isin([1, 3])].copy()
    df_feat["y"] = (df_feat.final_stage == 3).astype(int)

    train = df_feat[(df_feat.entry_ts >= TRAIN_PERIOD.start) &
                    (df_feat.entry_ts <= TRAIN_PERIOD.end)]
    test = df_feat[(df_feat.entry_ts >= TEST_PERIOD.start) &
                   (df_feat.entry_ts <= TEST_PERIOD.end)]

    if train.empty or test.empty:
        return {"error": "empty split"}

    Xtr, ytr = train[FEATURES].to_numpy(), train["y"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test["y"].to_numpy()

    # 1) Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(Xtr, ytr)
    lr_train = roc_auc_score(ytr, lr.predict_proba(Xtr)[:, 1])
    lr_test = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    coefs = lr.named_steps["lr"].coef_[0]
    lr_imp = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)

    # 2) Gradient Boosting
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
        subsample=0.8,
    )
    gbm.fit(Xtr, ytr)
    gbm_train = roc_auc_score(ytr, gbm.predict_proba(Xtr)[:, 1])
    gbm_test = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
    gbm_imp = sorted(zip(FEATURES, gbm.feature_importances_),
                     key=lambda x: x[1], reverse=True)

    # 3) test set 三段 quantile：top/mid/bot 30% 的 stage3 命中率（GBM 機率排序）
    proba_te = gbm.predict_proba(Xte)[:, 1]
    test_df = test.assign(p=proba_te).sort_values("p")
    n = len(test_df)
    q1 = test_df.iloc[:n // 3]
    q2 = test_df.iloc[n // 3: 2 * n // 3]
    q3 = test_df.iloc[2 * n // 3:]
    base_rate = test["y"].mean()
    quartile_stats = {
        "base_rate": float(base_rate),
        "bot_30%": float(q1["y"].mean()),
        "mid_30%": float(q2["y"].mean()),
        "top_30%": float(q3["y"].mean()),
    }

    return {
        "n_train": len(train), "n_test": len(test),
        "train_pos_rate": float(ytr.mean()),
        "test_pos_rate": float(yte.mean()),
        "lr_train_auc": lr_train, "lr_test_auc": lr_test,
        "gbm_train_auc": gbm_train, "gbm_test_auc": gbm_test,
        "lr_top5": lr_imp[:5], "gbm_top5": gbm_imp[:5],
        "quartile_stats": quartile_stats,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/stage3_multivariate.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    base_cfg = load_config(args.config)

    all_results = []
    feature_dfs = []

    for tf, wf, ws in COMBOS:
        name = f"{tf}_W{wf}-{ws}"
        print(f"\n>>> {name}: building features ...")
        df_feat = _build_features(tf, wf, ws, base_cfg)
        if df_feat.empty:
            continue
        feature_dfs.append(df_feat.assign(combo=name))
        print(f"  total trades: {len(df_feat)}  "
              f"s1={int((df_feat.final_stage==1).sum())} "
              f"s3={int((df_feat.final_stage==3).sum())}")

        res = _train_eval(df_feat)
        res["combo"] = name
        all_results.append(res)

        print(f"  n_train={res['n_train']} (pos {res['train_pos_rate']*100:.1f}%) "
              f"n_test={res['n_test']} (pos {res['test_pos_rate']*100:.1f}%)")
        print(f"  LR  AUC  train={res['lr_train_auc']:.4f}  test={res['lr_test_auc']:.4f}")
        print(f"  GBM AUC  train={res['gbm_train_auc']:.4f}  test={res['gbm_test_auc']:.4f}")
        print("  GBM top-5 importance:")
        for f, v in res["gbm_top5"]:
            print(f"    {f:25s} {v:.4f}")
        print("  LR  top-5 |coef|:")
        for f, v in res["lr_top5"]:
            print(f"    {f:25s} {v:+.4f}")
        q = res["quartile_stats"]
        print(f"  Test quantile（GBM 機率排序）s3 命中率：")
        print(f"    base_rate = {q['base_rate']*100:.2f}%")
        print(f"    bot 30%   = {q['bot_30%']*100:.2f}%")
        print(f"    mid 30%   = {q['mid_30%']*100:.2f}%")
        print(f"    top 30%   = {q['top_30%']*100:.2f}%   "
              f"(lift = {q['top_30%']/q['base_rate']:.2f}x)")

    print("\n\n" + "=" * 70)
    print("=== 跨 4 組合 AUC 摘要 ===")
    print("=" * 70)
    summary = pd.DataFrame([
        {
            "combo": r["combo"],
            "lr_train": r["lr_train_auc"], "lr_test": r["lr_test_auc"],
            "gbm_train": r["gbm_train_auc"], "gbm_test": r["gbm_test_auc"],
            "top30_lift": r["quartile_stats"]["top_30%"] / r["quartile_stats"]["base_rate"],
        } for r in all_results
    ]).set_index("combo")
    print(summary.to_string(float_format="%.4f"))

    print("\n判定：")
    mean_test = summary[["lr_test", "gbm_test"]].mean(axis=None)
    print(f"  4 組合 test AUC 平均（LR + GBM）= {mean_test:.4f}")
    if mean_test >= 0.55:
        print("  → 多變量還有可挖空間（>0.55）")
    elif mean_test >= 0.52:
        print("  → 邊際訊號（0.52~0.55），收益不明")
    else:
        print("  → 蓋章 pre-entry 無 edge（≤0.52）")

    if feature_dfs:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(feature_dfs, ignore_index=True).to_csv(
            out_path, index=False, float_format="%.6f")
        print(f"\n💾 features → {out_path}")


if __name__ == "__main__":
    main()
