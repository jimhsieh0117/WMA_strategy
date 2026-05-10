"""按 day-of-week × hour-of-day 拆 stage 分布。

用 entry_ts 的 dayofweek (0=Mon..6=Sun) 與 hour (0..23, UTC)。

輸出：
1. day-of-week 彙總（含 weekend vs weekday 對比）
2. hour-of-day 彙總（s3% / s1% / 平均 PnL）
3. day × hour 交叉表（s3%）
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
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _extract(trades: list) -> pd.DataFrame:
    rows = []
    for t in trades:
        ts = pd.Timestamp(t.entry_timestamp)
        rows.append({
            "dow": int(ts.dayofweek),
            "hour": int(ts.hour),
            "stage": int(t.final_stage),
            "pnl": float(t.net_pnl),
            "is_weekend": int(ts.dayofweek) >= 5,
        })
    return pd.DataFrame(rows)


def _stage_table(g: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    s = g.size().rename("n")
    s1 = g.apply(lambda d: (d["stage"] == 1).sum(), include_groups=False).rename("s1")
    s2 = g.apply(lambda d: (d["stage"] == 2).sum(), include_groups=False).rename("s2")
    s3 = g.apply(lambda d: (d["stage"] == 3).sum(), include_groups=False).rename("s3")
    pnl = g["pnl"].sum().rename("pnl")
    out = pd.concat([s, s1, s2, s3, pnl], axis=1)
    out["s1%"] = out["s1"] / out["n"] * 100
    out["s3%"] = out["s3"] / out["n"] * 100
    out["avg_pnl"] = out["pnl"] / out["n"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/day_hour.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg = dataclasses.replace(
        cfg, show_progress=False, log_level="WARNING",
        in_sample=DEFAULT_PERIOD,
    )

    print(f"running long / short ... ({cfg.symbol} {cfg.timeframe} "
          f"{cfg.in_sample.start.date()} ~ {cfg.in_sample.end.date()})")
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")

    df = _extract(L.trades + S.trades)
    n = len(df)
    n3 = (df["stage"] == 3).sum()
    print(f"\ntotal trades: {n}, baseline s3%: {n3/n*100:.2f}%, "
          f"total PnL: {df['pnl'].sum():+.2f}")

    # 1) Day-of-week 表
    dow_tbl = _stage_table(df.groupby("dow"))
    dow_tbl.index = [DOW_LABELS[i] for i in dow_tbl.index]
    print("\n=== Day-of-week × stage ===")
    print(dow_tbl[["n", "s1%", "s3%", "pnl", "avg_pnl"]].to_string(float_format="%.2f"))

    # 2) Weekend vs weekday
    we = _stage_table(df.groupby("is_weekend"))
    we.index = ["Weekday (Mon-Fri)", "Weekend (Sat-Sun)"]
    print("\n=== Weekend vs Weekday ===")
    print(we[["n", "s1%", "s3%", "pnl", "avg_pnl"]].to_string(float_format="%.2f"))

    # 3) Hour-of-day 表
    hour_tbl = _stage_table(df.groupby("hour"))
    print("\n=== Hour-of-day (UTC) × stage ===")
    print(hour_tbl[["n", "s1%", "s3%", "pnl", "avg_pnl"]].to_string(float_format="%.2f"))

    # 4) Cross table (s3% by dow × hour)
    cross_n = df.groupby(["dow", "hour"]).size().unstack(fill_value=0)
    cross_n.index = [DOW_LABELS[i] for i in cross_n.index]
    cross_s3 = (df[df["stage"] == 3].groupby(["dow", "hour"]).size()
                .unstack(fill_value=0).reindex(columns=cross_n.columns, fill_value=0))
    cross_s3.index = [DOW_LABELS[i] for i in cross_s3.index]
    cross_pct = (cross_s3 / cross_n.replace(0, pd.NA) * 100).round(1)
    print("\n=== s3% (dow × hour, UTC) ===")
    print(cross_pct.to_string(na_rep="—"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
