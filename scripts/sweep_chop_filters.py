"""盤整過濾器 sweep：四種候選 × 多個閾值 × MC bootstrap CI。

對 baseline 回測（預設 15m W4-6 both）的每筆交易，在 signal_ts（=entry_ts − tf）
量測四種「盤整偵測」指標：
  1. ADX(14)         <X = 盤整 → skip
  2. Choppiness(14)  >X = 盤整 → skip
  3. ATR pct rank200 <X = 低波動/盤整 → skip
  4. BB width rank200 <X = squeeze → skip

對每組 (filter, threshold)：
  - 過濾後 trade count / total PnL / win rate / PF
  - MC bootstrap PF 95% CI、P(PF > 1)

用法：
    python scripts/sweep_chop_filters.py
    python scripts/sweep_chop_filters.py --timeframe 5m --wma-fast 4 --wma-slow 6
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
from scripts.analyze_stage3_features_v2 import (  # noqa: E402
    _adx_dmi, _atr, _atr_pct_rank, _bb_width, _wilder_smooth,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.validation.monte_carlo import MCConfig, bootstrap  # noqa: E402


def _choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Choppiness Index：>61.8 視為盤整、<38.2 視為強趨勢。

    CI = 100 * log10( sum(TR, n) / (max(H, n) - min(L, n)) ) / log10(n)
    """
    h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    sum_tr = tr.rolling(period, min_periods=period).sum()
    hh = h.rolling(period, min_periods=period).max()
    ll = l.rolling(period, min_periods=period).min()
    rng = (hh - ll).replace(0, np.nan)
    ci = 100.0 * np.log10(sum_tr / rng) / np.log10(period)
    return ci


def _rank_pct(s: pd.Series, window: int = 200) -> pd.Series:
    return s.rolling(window, min_periods=window).rank(pct=True)


def _compute_filter_cache(df: pd.DataFrame) -> pd.DataFrame:
    """對 timeframe df 計算四個 filter 指標。"""
    adx, _pdi, _mdi = _adx_dmi(df, period=14)
    ci = _choppiness(df, period=14)
    atr = _atr(df, period=14)
    atr_rank = _atr_pct_rank(atr, window=200) * 100  # 轉百分比
    bbw = _bb_width(df["close"], period=20, k=2.0)
    bbw_rank = _rank_pct(bbw, window=200) * 100
    return pd.DataFrame({
        "adx": adx,
        "ci":  ci,
        "atr_rank": atr_rank,
        "bbw_rank": bbw_rank,
    })


def _trade_metrics(pnls: np.ndarray) -> dict:
    n = pnls.size
    if n == 0:
        return {"n": 0, "win%": 0.0, "pnl": 0.0, "pf": float("nan")}
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    pf = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)
    return {
        "n": n,
        "win%": float((pnls > 0).mean() * 100),
        "pnl": float(pnls.sum()),
        "pf": float(pf),
    }


def _filter_lookup(
    trades: list, filter_series: pd.Series, tf_delta: pd.Timedelta,
) -> np.ndarray:
    """對每筆 trade 取 signal_ts 的 filter 值；找不到 → NaN。"""
    vals = np.full(len(trades), np.nan)
    idx = filter_series.index
    for i, t in enumerate(trades):
        signal_ts = pd.Timestamp(t.entry_timestamp) - tf_delta
        pos = idx.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            continue
        v = filter_series.iloc[pos]
        if np.isfinite(v):
            vals[i] = float(v)
    return vals


# --------------------------------------------------------------------------- #
# 過濾規則：keep_mask = True 表示**保留**該交易
# --------------------------------------------------------------------------- #

FILTERS = [
    # (name, series_col, op, thresholds, description)
    # op="lt_keep" → 過濾掉 series < threshold（即 series >= threshold 保留）
    # op="gt_keep" → 過濾掉 series > threshold（即 series <= threshold 保留）
    ("ADX<X→skip",        "adx",      "ge_keep", [15, 20, 25],  "ADX 低於 X 視為無趨勢"),
    ("CI>X→skip",         "ci",       "le_keep", [55, 60, 65],  "Choppiness 高於 X 視為盤整"),
    ("ATR_rank<X→skip",   "atr_rank", "ge_keep", [20, 30, 40],  "ATR 百分位低於 X 視為低波動"),
    ("BBW_rank<X→skip",   "bbw_rank", "ge_keep", [20, 30, 40],  "BB width 百分位低於 X 視為 squeeze"),
]


def _apply_filter(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op == "ge_keep":
        keep = values >= threshold
    elif op == "le_keep":
        keep = values <= threshold
    else:
        raise ValueError(f"unknown op {op}")
    # NaN（暖機期不夠）一律 skip：cache 沒值就保守過濾
    keep &= np.isfinite(values)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--wma-fast", type=int, default=4)
    parser.add_argument("--wma-slow", type=int, default=6)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--mc-n", type=int, default=5000)
    parser.add_argument("--out", default="results/chop_filter_sweep.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_m = dataclasses.replace(
        cfg,
        show_progress=False, log_level="WARNING",
        timeframe=args.timeframe,
        wma_fast=args.wma_fast, wma_slow=args.wma_slow,
        in_sample=PeriodSpec(start=pd.Timestamp(args.start),
                             end=pd.Timestamp(args.end)),
    )

    tf_delta = pd.Timedelta(args.timeframe)

    print(f">>> backtest {cfg_m.symbol} {args.timeframe} W{args.wma_fast}-{args.wma_slow}  "
          f"{args.start} ~ {args.end}", flush=True)
    L = run_single_strategy(cfg_m, direction="long", sample="is")
    S = run_single_strategy(cfg_m, direction="short", sample="is")
    trades = L.trades + S.trades
    print(f"  long={len(L.trades)}  short={len(S.trades)}  total={len(trades)}")

    if not trades:
        print("no trades")
        return

    # 重新載入 1m 並 resample 到 timeframe（與 engine 一致）以算 filter
    print(">>> computing filter indicators ...")
    df1m = load_ohlcv(cfg_m.source_parquet, start=cfg_m.in_sample.start,
                      end=cfg_m.in_sample.end)
    df_tf = resample(df1m, args.timeframe)
    cache = _compute_filter_cache(df_tf)

    # 預先抽出每個 filter 的 per-trade 值
    pnls_all = np.array([t.net_pnl for t in trades])
    filter_vals = {col: _filter_lookup(trades, cache[col], tf_delta)
                   for col in ["adx", "ci", "atr_rank", "bbw_rank"]}

    base = _trade_metrics(pnls_all)
    print(f"\n[baseline] n={base['n']}  win={base['win%']:.2f}%  "
          f"pnl={base['pnl']:+.2f}  PF={base['pf']:.3f}")

    # MC bootstrap PF CI for baseline
    mc_cfg = MCConfig(n_simulations=args.mc_n, initial_capital=1000.0,
                      ruin_threshold_pct=50.0, seed=42)
    res = bootstrap(trades, mc_cfg)
    pf_arr = res.profit_factor
    pf_arr = pf_arr[np.isfinite(pf_arr)]
    base_ci = (np.percentile(pf_arr, 5), np.percentile(pf_arr, 50),
               np.percentile(pf_arr, 95), float((pf_arr > 1.0).mean() * 100))
    print(f"           MC PF CI: p05={base_ci[0]:.3f}  p50={base_ci[1]:.3f}  "
          f"p95={base_ci[2]:.3f}  P(PF>1)={base_ci[3]:.2f}%")

    rows = [{
        "filter": "baseline", "threshold": np.nan, "op": "",
        "n_kept": base["n"], "kept%": 100.0,
        "win%": base["win%"], "pnl": base["pnl"], "pf": base["pf"],
        "mc_pf_p05": base_ci[0], "mc_pf_p50": base_ci[1], "mc_pf_p95": base_ci[2],
        "p_pf_gt1": base_ci[3],
    }]

    for name, col, op, thresholds, desc in FILTERS:
        print(f"\n=== {name}  ({desc}) ===")
        vals = filter_vals[col]
        for thr in thresholds:
            keep = _apply_filter(vals, op, thr)
            kept_pnls = pnls_all[keep]
            kept_trades = [t for t, k in zip(trades, keep) if k]
            m = _trade_metrics(kept_pnls)
            if m["n"] < 50:
                print(f"  thr={thr:>3}  n={m['n']} 太少，跳過 MC")
                continue

            mc_res = bootstrap(kept_trades, mc_cfg)
            pf_arr = mc_res.profit_factor
            pf_arr = pf_arr[np.isfinite(pf_arr)]
            p05, p50, p95 = np.percentile(pf_arr, [5, 50, 95])
            p_gt1 = float((pf_arr > 1.0).mean() * 100)

            print(f"  thr={thr:>3}  n={m['n']:4d} ({m['n']/base['n']*100:5.1f}%)  "
                  f"win={m['win%']:5.2f}%  pnl={m['pnl']:+8.2f}  PF={m['pf']:.3f}  "
                  f"MC[{p05:.3f}, {p50:.3f}, {p95:.3f}]  P(PF>1)={p_gt1:5.2f}%")

            rows.append({
                "filter": name, "threshold": thr, "op": op,
                "n_kept": m["n"], "kept%": m["n"] / base["n"] * 100,
                "win%": m["win%"], "pnl": m["pnl"], "pf": m["pf"],
                "mc_pf_p05": p05, "mc_pf_p50": p50, "mc_pf_p95": p95,
                "p_pf_gt1": p_gt1,
            })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("=== Summary ===")
    print("=" * 100)
    print(df.to_string(index=False, float_format="%.3f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 → {out_path}")


if __name__ == "__main__":
    main()
