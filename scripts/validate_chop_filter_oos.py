"""盤整過濾器 OOS 驗證 + Stage 削減比例分析。

用 IS（2023-01-01 ~ 2024-12-31）上選出的最佳組合
  BBW_rank >= 40 & ATR_rank >= 40 & ADX >= 20
在 OOS（2025-01-01 ~ 資料末）重新驗證：
  - 過濾前/後的 PnL、PF、win rate
  - MC bootstrap PF CI、P(PF > 1)
  - **Stage 1/2/3 被刪減的數量與比例**（看哪個 stage 被砍最多）

用法：
    python scripts/validate_chop_filter_oos.py
    python scripts/validate_chop_filter_oos.py --bbw 40 --atr 40 --adx 20
    python scripts/validate_chop_filter_oos.py --oos-end 2026-03-14
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
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup, _trade_metrics,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.validation.monte_carlo import MCConfig, bootstrap  # noqa: E402


def _build_trades(cfg, start, end):
    cfg_p = dataclasses.replace(cfg, in_sample=PeriodSpec(
        start=pd.Timestamp(start), end=pd.Timestamp(end)))
    L = run_single_strategy(cfg_p, direction="long", sample="is")
    S = run_single_strategy(cfg_p, direction="short", sample="is")
    return L.trades + S.trades


def _apply_combo(trades, fv, bbw_thr, atr_thr, adx_thr) -> np.ndarray:
    return (
        (fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
        (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
        (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"])
    )


def _stage_table(trades, keep_mask) -> pd.DataFrame:
    stages = np.array([t.final_stage for t in trades])
    pnls = np.array([t.net_pnl for t in trades])
    rows = []
    for s in (1, 2, 3):
        total_mask = (stages == s)
        kept_mask = total_mask & keep_mask
        removed_mask = total_mask & (~keep_mask)
        n_total = int(total_mask.sum())
        n_kept = int(kept_mask.sum())
        n_rem  = int(removed_mask.sum())
        rows.append({
            "stage": s,
            "n_all": n_total,
            "n_kept": n_kept,
            "n_removed": n_rem,
            "remove_rate%": (n_rem / n_total * 100) if n_total else 0.0,
            "pnl_all":     float(pnls[total_mask].sum()),
            "pnl_kept":    float(pnls[kept_mask].sum()),
            "pnl_removed": float(pnls[removed_mask].sum()),
        })
    return pd.DataFrame(rows)


def _bootstrap_pf_ci(trades, mc_cfg) -> tuple[float, float, float, float]:
    res = bootstrap(trades, mc_cfg)
    pf = res.profit_factor
    pf = pf[np.isfinite(pf)]
    return (float(np.percentile(pf, 5)),
            float(np.percentile(pf, 50)),
            float(np.percentile(pf, 95)),
            float((pf > 1.0).mean() * 100))


def _report_sample(label, trades, fv, args, mc_cfg, tf_delta):
    pnls_all = np.array([t.net_pnl for t in trades])
    keep = _apply_combo(trades, fv, args.bbw, args.atr, args.adx)
    kt = [t for t, k in zip(trades, keep) if k]

    base = _trade_metrics(pnls_all)
    sub = _trade_metrics(pnls_all[keep])

    print("\n" + "=" * 70)
    print(f"=== {label}  filter: BBW>={args.bbw} & ATR>={args.atr} & ADX>={args.adx} ===")
    print("=" * 70)
    print(f"All  n={base['n']:4d}  win={base['win%']:5.2f}%  "
          f"pnl={base['pnl']:+8.2f}  PF={base['pf']:.3f}")

    if base["n"] > 0:
        bp = _bootstrap_pf_ci(trades, mc_cfg)
        print(f"     MC[{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]  P(PF>1)={bp[3]:5.2f}%")

    print(f"Kept n={sub['n']:4d} ({sub['n']/base['n']*100:5.1f}%)  "
          f"win={sub['win%']:5.2f}%  pnl={sub['pnl']:+8.2f}  PF={sub['pf']:.3f}")

    if sub["n"] >= 50:
        sp = _bootstrap_pf_ci(kt, mc_cfg)
        print(f"     MC[{sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}]  P(PF>1)={sp[3]:5.2f}%")

    stage_df = _stage_table(trades, keep)
    total_n = stage_df["n_all"].sum()
    total_rem = stage_df["n_removed"].sum()
    print(f"\nStage breakdown （總刪減 {total_rem}/{total_n} = "
          f"{total_rem/total_n*100:.1f}%）：")
    print(stage_df.to_string(index=False, float_format="%.2f"))

    return stage_df


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
    parser.add_argument("--mc-n", type=int, default=5000)
    parser.add_argument("--out", default="results/chop_oos_validation.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_m = dataclasses.replace(
        cfg, show_progress=False, log_level="WARNING",
        timeframe=args.timeframe,
        wma_fast=args.wma_fast, wma_slow=args.wma_slow,
    )
    tf_delta = pd.Timedelta(args.timeframe)
    mc_cfg = MCConfig(n_simulations=args.mc_n, initial_capital=1000.0,
                      ruin_threshold_pct=50.0, seed=42)

    # IS
    print(f">>> [IS]  backtest {args.is_start} ~ {args.is_end}", flush=True)
    is_trades = _build_trades(cfg_m, args.is_start, args.is_end)
    df1m_is = load_ohlcv(cfg_m.source_parquet,
                         start=pd.Timestamp(args.is_start),
                         end=pd.Timestamp(args.is_end))
    cache_is = _compute_filter_cache(resample(df1m_is, args.timeframe))
    fv_is = {col: _filter_lookup(is_trades, cache_is[col], tf_delta)
             for col in ["adx", "atr_rank", "bbw_rank"]}
    is_stage = _report_sample("IS (2023-2024)", is_trades, fv_is,
                              args, mc_cfg, tf_delta)

    # OOS
    print(f"\n>>> [OOS] backtest {args.oos_start} ~ {args.oos_end}", flush=True)
    oos_trades = _build_trades(cfg_m, args.oos_start, args.oos_end)
    df1m_oos = load_ohlcv(cfg_m.source_parquet,
                          start=pd.Timestamp(args.oos_start),
                          end=pd.Timestamp(args.oos_end))
    cache_oos = _compute_filter_cache(resample(df1m_oos, args.timeframe))
    fv_oos = {col: _filter_lookup(oos_trades, cache_oos[col], tf_delta)
              for col in ["adx", "atr_rank", "bbw_rank"]}
    oos_stage = _report_sample("OOS (2025+)", oos_trades, fv_oos,
                               args, mc_cfg, tf_delta)

    # 對照
    combined = pd.concat(
        [is_stage.assign(sample="IS"), oos_stage.assign(sample="OOS")],
        ignore_index=True,
    )
    combined = combined[["sample", "stage", "n_all", "n_kept", "n_removed",
                         "remove_rate%", "pnl_all", "pnl_kept", "pnl_removed"]]
    print("\n" + "=" * 70)
    print("=== IS vs OOS Stage Breakdown ===")
    print("=" * 70)
    print(combined.to_string(index=False, float_format="%.2f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 → {out_path}")


if __name__ == "__main__":
    main()
