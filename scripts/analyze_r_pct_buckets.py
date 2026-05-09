"""把 trades 的 R% 分佈用 0.02% 級距分桶輸出成 CSV。

用法：
    python scripts/analyze_r_pct_buckets.py [--config configs/default.yaml]
                                            [--sample is|oos] [--step 0.0002]
                                            [--out results/r_pct_buckets.csv]

預設行為：跑 2 年 IS（2023-01-01 ~ 2024-12-31），其他設定吃 yaml。
輸出欄：bucket_lo / bucket_hi / count / cum_count / cum_share% / win% / s3% /
       avg_pnl / sum_pnl / s1 / s2 / s3
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
from src.utils.config import PeriodSpec, load_config  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--step", type=float, default=0.0002,
                        help="R%% bucket step（預設 0.0002 = 0.02%%）")
    parser.add_argument("--max", type=float, default=0.025,
                        help="bucket 上限（超過丟到 >=N 尾桶；預設 0.025 = 2.5%%）")
    parser.add_argument("--out", default="results/r_pct_buckets.csv")
    parser.add_argument("--no-r-min", action="store_true",
                        help="覆寫 r_min_pct=0 以包含全部進場訊號（預設啟用）")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides: dict = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    if args.no_r_min or True:  # 一律覆寫，保證看得到全部 trades
        overrides["r_min_pct"] = 0.0
    cfg_m = dataclasses.replace(cfg, **overrides)

    print(f"running long ...  ({cfg_m.symbol} {cfg_m.timeframe} "
          f"{cfg_m.in_sample.start.date()} ~ {cfg_m.in_sample.end.date()})")
    L = run_single_strategy(cfg_m, direction="long", sample=args.sample)
    print("running short ...")
    S = run_single_strategy(cfg_m, direction="short", sample=args.sample)

    rows = []
    for t in L.trades + S.trades:
        r_price = abs(t.entry_price - t.stop_history[0][1])
        rows.append({
            "final_stage": t.final_stage,
            "r_pct": r_price / t.entry_price,
            "net_pnl": t.net_pnl,
        })
    df = pd.DataFrame(rows)
    print(f"total trades: {len(df)} (LONG {len(L.trades)} / SHORT {len(S.trades)})")

    edges = np.arange(0.0, args.max + args.step, args.step)
    labels = list(range(len(edges) - 1))
    df["b_idx"] = pd.cut(df["r_pct"], bins=edges, labels=labels, right=False,
                         include_lowest=True)

    g = df.groupby("b_idx", observed=True)
    out = pd.DataFrame({
        "bucket_lo%": [edges[int(b)] * 100 for b in g.size().index],
        "bucket_hi%": [edges[int(b) + 1] * 100 for b in g.size().index],
        "count": g.size().values,
        "avg_pnl": g["net_pnl"].mean().values,
        "sum_pnl": g["net_pnl"].sum().values,
        "win%": (g["net_pnl"].apply(lambda x: (x > 0).mean()) * 100).values,
        "s1": g["final_stage"].apply(lambda x: (x == 1).sum()).values,
        "s2": g["final_stage"].apply(lambda x: (x == 2).sum()).values,
        "s3": g["final_stage"].apply(lambda x: (x == 3).sum()).values,
    })
    out["s3%"] = out["s3"] / out["count"] * 100
    out["cum_count"] = out["count"].cumsum()
    out["cum_share%"] = out["cum_count"] / len(df) * 100

    # 尾桶（>= max）
    tail = df[df["r_pct"] >= args.max]
    if len(tail):
        tail_row = pd.DataFrame([{
            "bucket_lo%": args.max * 100,
            "bucket_hi%": float("inf"),
            "count": len(tail),
            "avg_pnl": tail["net_pnl"].mean(),
            "sum_pnl": tail["net_pnl"].sum(),
            "win%": (tail["net_pnl"] > 0).mean() * 100,
            "s1": (tail["final_stage"] == 1).sum(),
            "s2": (tail["final_stage"] == 2).sum(),
            "s3": (tail["final_stage"] == 3).sum(),
            "s3%": (tail["final_stage"] == 3).mean() * 100,
            "cum_count": len(df),
            "cum_share%": 100.0,
        }])
        out = pd.concat([out, tail_row], ignore_index=True)

    out = out[[
        "bucket_lo%", "bucket_hi%", "count", "cum_count", "cum_share%",
        "win%", "s3%", "avg_pnl", "sum_pnl", "s1", "s2", "s3",
    ]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 written → {out_path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
