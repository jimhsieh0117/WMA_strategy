"""WMA 較長參數 sweep。

對比目前 baseline (2,4) / (4,6) 與「稍微長一點」的候選 (5,8) / (6,10) / (8,13)，
每組合在 5m 與 15m 上分別跑 long + short（IS 期間），彙整：
  - trade-level：n, stage 1/2/3 分布, total PnL
  - account-level（per-direction）：Profit Factor, Sharpe, Max DD

用法：
    python scripts/sweep_wma_long.py [--config configs/default.yaml]
                                     [--out results/wma_sweep_long.csv]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)

WMA_COMBOS = [
    (2, 4),    # baseline-A（現行偏快）
    (4, 6),    # baseline-B（現行偏慢）
    (5, 8),    # 候選 1：Fib 微調
    (6, 10),   # 候選 2：~1.5× baseline-B
    (8, 13),   # 候選 3：Fib swing
]

TIMEFRAMES = ["5m", "15m"]


def _trade_agg(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "s1%": 0.0, "s2%": 0.0, "s3%": 0.0, "pnl": 0.0}
    s1 = sum(1 for t in trades if t.final_stage == 1)
    s2 = sum(1 for t in trades if t.final_stage == 2)
    s3 = sum(1 for t in trades if t.final_stage == 3)
    return {
        "n": n,
        "s1%": s1 / n * 100, "s2%": s2 / n * 100, "s3%": s3 / n * 100,
        "pnl": sum(t.net_pnl for t in trades),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/wma_sweep_long.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_base = dataclasses.replace(
        cfg,
        show_progress=False,
        log_level="WARNING",
        in_sample=DEFAULT_PERIOD,
    )

    rows = []
    for tf in TIMEFRAMES:
        for fast, slow in WMA_COMBOS:
            tag = f"{tf}_W{fast}-{slow}"
            print(f"\n>>> {tag}", flush=True)
            cfg_m = dataclasses.replace(
                cfg_base, timeframe=tf, wma_fast=fast, wma_slow=slow,
            )

            L = run_single_strategy(cfg_m, direction="long", sample="is")
            S = run_single_strategy(cfg_m, direction="short", sample="is")

            mL = compute_metrics(L, timeframe=tf)
            mS = compute_metrics(S, timeframe=tf)
            agg = _trade_agg(L.trades + S.trades)

            row = {
                "tf": tf, "fast": fast, "slow": slow,
                **agg,
                "L_pf": mL.profit_factor, "L_sharpe": mL.sharpe_ratio,
                "L_dd": mL.max_drawdown_pct, "L_ret": mL.total_return_pct,
                "S_pf": mS.profit_factor, "S_sharpe": mS.sharpe_ratio,
                "S_dd": mS.max_drawdown_pct, "S_ret": mS.total_return_pct,
            }
            rows.append(row)
            print(f"  n={row['n']:4d}  s3%={row['s3%']:5.2f}  pnl={row['pnl']:+8.2f}  "
                  f"L_PF={row['L_pf']:.2f} S_PF={row['S_pf']:.2f}  "
                  f"L_DD={row['L_dd']:.1f}% S_DD={row['S_dd']:.1f}%")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print(f"=== WMA sweep (ETHUSDT IS {DEFAULT_PERIOD.start.date()} ~ "
          f"{DEFAULT_PERIOD.end.date()}) ===")
    print("=" * 78)
    show = df[[
        "tf", "fast", "slow", "n", "s1%", "s2%", "s3%", "pnl",
        "L_pf", "S_pf", "L_sharpe", "S_sharpe", "L_dd", "S_dd",
    ]]
    print(show.to_string(index=False, float_format="%.2f"))

    print("\n=== 依 total_pnl 排序 ===")
    print(df.sort_values("pnl", ascending=False)[
        ["tf", "fast", "slow", "n", "s3%", "pnl", "L_pf", "S_pf"]
    ].to_string(index=False, float_format="%.2f"))

    print("\n=== 依 s3% 排序 ===")
    print(df.sort_values("s3%", ascending=False)[
        ["tf", "fast", "slow", "n", "s3%", "pnl", "L_pf", "S_pf"]
    ].to_string(index=False, float_format="%.2f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
