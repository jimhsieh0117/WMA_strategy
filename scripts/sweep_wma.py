"""WMA fast/slow 參數 sweep。

掃 fast ∈ {2,3,4}, slow ∈ {4,5,6}（限 fast < slow），對每組合在 2 年 IS 15m 跑回測，
彙整 stage 1/2/3 分布、各 stage 平均 PnL、總 PnL。

用法：
    python scripts/sweep_wma.py [--config configs/default.yaml] [--timeframe 15m]
                                 [--out results/wma_sweep.csv]
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
from src.utils.config import PeriodSpec, load_config  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)


def _aggregate(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "n": 0, "s1": 0, "s2": 0, "s3": 0,
            "s1%": 0.0, "s2%": 0.0, "s3%": 0.0,
            "pnl": 0.0, "s1_pnl": 0.0, "s2_pnl": 0.0, "s3_pnl": 0.0,
        }
    s1 = [t for t in trades if t.final_stage == 1]
    s2 = [t for t in trades if t.final_stage == 2]
    s3 = [t for t in trades if t.final_stage == 3]
    return {
        "n": n,
        "s1": len(s1), "s2": len(s2), "s3": len(s3),
        "s1%": len(s1) / n * 100, "s2%": len(s2) / n * 100, "s3%": len(s3) / n * 100,
        "pnl": sum(t.net_pnl for t in trades),
        "s1_pnl": sum(t.net_pnl for t in s1),
        "s2_pnl": sum(t.net_pnl for t in s2),
        "s3_pnl": sum(t.net_pnl for t in s3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--out", default="results/wma_sweep.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {
        "show_progress": False,
        "log_level": "WARNING",
        "in_sample": DEFAULT_PERIOD,
    }
    if args.timeframe is not None:
        overrides["timeframe"] = args.timeframe
    cfg_base = dataclasses.replace(cfg, **overrides)

    combos = [(f, s) for f in (2, 3, 4) for s in (4, 5, 6) if f < s]
    rows = []
    for fast, slow in combos:
        cfg_m = dataclasses.replace(cfg_base, wma_fast=fast, wma_slow=slow)
        print(f"[{fast},{slow}] running long ...", flush=True)
        L = run_single_strategy(cfg_m, direction="long", sample="is")
        print(f"[{fast},{slow}] running short ...", flush=True)
        S = run_single_strategy(cfg_m, direction="short", sample="is")
        agg = _aggregate(L.trades + S.trades)
        rows.append({"fast": fast, "slow": slow, **agg})

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("s3%", ascending=False).reset_index(drop=True)

    print(f"\n=== WMA sweep ({cfg_base.symbol} {cfg_base.timeframe} "
          f"{cfg_base.in_sample.start.date()} ~ {cfg_base.in_sample.end.date()}) ===")
    pretty = df_sorted[[
        "fast", "slow", "n", "s1%", "s2%", "s3%",
        "pnl", "s1_pnl", "s2_pnl", "s3_pnl",
    ]]
    print(pretty.to_string(index=False, float_format="%.2f"))

    print("\n依 total_pnl 排序：")
    print(df.sort_values("pnl", ascending=False)[[
        "fast", "slow", "n", "s3%", "pnl"
    ]].to_string(index=False, float_format="%.2f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
