"""比較 entry_hour_blacklist 不同組合（baseline / [0] / [12] / [0,12]）。

對 2-yr IS 15m WMA(4,6) 跑全部組合，輸出 PnL / s1/s2/s3 / 訊號被拒數。
"""

from __future__ import annotations

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


def _summarise(label: str, L_res, S_res) -> dict:
    trades = L_res.trades + S_res.trades
    n = len(trades)
    s1 = sum(1 for t in trades if t.final_stage == 1)
    s2 = sum(1 for t in trades if t.final_stage == 2)
    s3 = sum(1 for t in trades if t.final_stage == 3)
    pnl = sum(t.net_pnl for t in trades)
    emitted = L_res.signals_emitted + S_res.signals_emitted
    filled = L_res.signals_filled + S_res.signals_filled
    return {
        "label": label, "emitted": emitted, "filled": filled, "n": n,
        "s1": s1, "s2": s2, "s3": s3,
        "s1%": s1/n*100 if n else 0,
        "s3%": s3/n*100 if n else 0,
        "pnl": pnl,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg_base = load_config("configs/default.yaml")
    overrides = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    cfg_base = dataclasses.replace(cfg_base, **overrides)

    period = cfg_base.in_sample if args.sample == "is" else cfg_base.out_of_sample

    cases = [
        ("baseline (no blacklist)", ()),
        ("blacklist [0]", (0,)),
        ("blacklist [12]", (12,)),
        ("blacklist [0, 12]", (0, 12)),
    ]
    rows = []
    for label, bl in cases:
        cfg = dataclasses.replace(cfg_base, entry_hour_blacklist=bl)
        print(f"[{label}] running ...", flush=True)
        L = run_single_strategy(cfg, direction="long", sample=args.sample)
        S = run_single_strategy(cfg, direction="short", sample=args.sample)
        rows.append(_summarise(label, L, S))

    df = pd.DataFrame(rows)
    print(f"\n=== entry_hour_blacklist sweep (15m WMA 4/6, "
          f"{args.sample.upper()} {period.start.date()} ~ "
          f"{period.end.date() if period.end else 'end-of-data'}) ===")
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
