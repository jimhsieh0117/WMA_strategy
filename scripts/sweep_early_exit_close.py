"""Sweep first_close_r 門檻 × early_exit_enabled，跟 baseline 對比。

對 2-year IS 15m WMA(4,6) 跑：
- baseline（early_exit_enabled=False）
- close-mode 門檻：-1.0R, -0.5R, -0.2R, 0.0R

報告每組的總 PnL、s1/s2/s3 數量、EARLY_CANCEL 數。
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


def _summarise(label: str, trades_long: list, trades_short: list) -> dict:
    trades = trades_long + trades_short
    n = len(trades)
    n1 = sum(1 for t in trades if t.final_stage == 1)
    n2 = sum(1 for t in trades if t.final_stage == 2)
    n3 = sum(1 for t in trades if t.final_stage == 3)
    cancels = sum(1 for t in trades if t.exit_reason == "EARLY_CANCEL")
    pnl = sum(t.net_pnl for t in trades)
    pnl_s3 = sum(t.net_pnl for t in trades if t.final_stage == 3)
    pnl_s1 = sum(t.net_pnl for t in trades if t.final_stage == 1)
    return {
        "label": label, "n": n, "s1": n1, "s2": n2, "s3": n3,
        "s1%": n1/n*100 if n else 0, "s3%": n3/n*100 if n else 0,
        "cancels": cancels, "pnl": pnl,
        "pnl_s1": pnl_s1, "pnl_s3": pnl_s3,
    }


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config("configs/default.yaml")
    cfg = dataclasses.replace(
        cfg,
        show_progress=False, log_level="WARNING",
        in_sample=DEFAULT_PERIOD,
    )

    rows = []

    # baseline: early_exit disabled
    trailing_off = dataclasses.replace(cfg.trailing, early_exit_enabled=False)
    cfg_off = dataclasses.replace(cfg, trailing=trailing_off)
    print("[baseline] running ...", flush=True)
    L = run_single_strategy(cfg_off, direction="long", sample="is")
    S = run_single_strategy(cfg_off, direction="short", sample="is")
    rows.append(_summarise("baseline (off)", L.trades, S.trades))

    # close-mode sweep
    for thr in [-1.0, -0.5, -0.2, 0.0]:
        trailing = dataclasses.replace(
            cfg.trailing,
            early_exit_enabled=True,
            early_exit_observation_bars=1,
            early_exit_metric="close",
            early_exit_min_close_r=thr,
        )
        cfg_thr = dataclasses.replace(cfg, trailing=trailing)
        print(f"[close ≥ {thr}] running ...", flush=True)
        L = run_single_strategy(cfg_thr, direction="long", sample="is")
        S = run_single_strategy(cfg_thr, direction="short", sample="is")
        rows.append(_summarise(f"close ≥ {thr}", L.trades, S.trades))

    df = pd.DataFrame(rows)
    print(f"\n=== first_close_r early_exit sweep (15m, WMA 4/6, 2-yr IS) ===")
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
