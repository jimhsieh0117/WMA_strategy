"""Early-exit 三種 metric 比較 sweep（peak / peak_pct / close）。

baseline + peak {0.0, 0.05, 0.10, 0.20, 0.30} R
        + peak_pct {0.05%, 0.10%, 0.20%, 0.30%, 0.50%}
        + close {-1.0, -0.5, -0.2, 0.0} R

對 2-yr IS 15m WMA(4,6) 跑全部組合。
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
    real_s1 = n1 - cancels  # 排除 EARLY_CANCEL 的「真止損」s1
    pnl = sum(t.net_pnl for t in trades)
    return {
        "label": label, "n": n,
        "real_s1": real_s1, "cancels": cancels,
        "s2": n2, "s3": n3,
        "s1%": n1/n*100 if n else 0,
        "s3%": n3/n*100 if n else 0,
        "pnl": pnl,
    }


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config("configs/default.yaml")
    cfg = dataclasses.replace(
        cfg, show_progress=False, log_level="WARNING",
        in_sample=DEFAULT_PERIOD,
    )

    rows = []

    # baseline
    trailing_off = dataclasses.replace(cfg.trailing, early_exit_enabled=False)
    cfg_off = dataclasses.replace(cfg, trailing=trailing_off)
    print("[baseline] running ...", flush=True)
    L = run_single_strategy(cfg_off, direction="long", sample="is")
    S = run_single_strategy(cfg_off, direction="short", sample="is")
    rows.append(_summarise("baseline", L.trades, S.trades))

    # peak (R-based)
    for thr in [0.0, 0.05, 0.10, 0.20, 0.30]:
        trailing = dataclasses.replace(
            cfg.trailing,
            early_exit_enabled=True,
            early_exit_observation_bars=1,
            early_exit_metric="peak",
            early_exit_min_peak_r=thr,
        )
        cfg_thr = dataclasses.replace(cfg, trailing=trailing)
        label = f"peak ≥ {thr:.2f}R"
        print(f"[{label}] running ...", flush=True)
        L = run_single_strategy(cfg_thr, direction="long", sample="is")
        S = run_single_strategy(cfg_thr, direction="short", sample="is")
        rows.append(_summarise(label, L.trades, S.trades))

    # peak_pct (% of entry)
    for thr in [0.0005, 0.0010, 0.0020, 0.0030, 0.0050]:
        trailing = dataclasses.replace(
            cfg.trailing,
            early_exit_enabled=True,
            early_exit_observation_bars=1,
            early_exit_metric="peak_pct",
            early_exit_min_peak_pct=thr,
        )
        cfg_thr = dataclasses.replace(cfg, trailing=trailing)
        label = f"peak_pct ≥ {thr*100:.2f}%"
        print(f"[{label}] running ...", flush=True)
        L = run_single_strategy(cfg_thr, direction="long", sample="is")
        S = run_single_strategy(cfg_thr, direction="short", sample="is")
        rows.append(_summarise(label, L.trades, S.trades))

    df = pd.DataFrame(rows)
    print("\n=== early_exit metric sweep (15m WMA 4/6, 2-yr IS) ===")
    print(df.to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
