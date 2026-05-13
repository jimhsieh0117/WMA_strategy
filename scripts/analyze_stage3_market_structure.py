"""Stage 3 訂單進場時的市場結構（BoS / CHoCH）分布分析。

目的
====

對 2 年 IS 期間所有交易，依進場 bar 當時的 ``ms_trend``（bull / bear / 無）
× ``final_stage``（1/2/3）建立交叉表，回答兩個問題：

1. **Stage 3 的成功單，是否集中在「同向結構」？**
   （long stage 3 多在 bull structure；short stage 3 多在 bear）
2. **Stage 1（虧損）單在「反向結構」是否過度集中？**
   （long 卻在 bear structure → 反趨勢進場 → 應該被未來的 gate 過濾）

執行
====

    python -m scripts.analyze_stage3_market_structure
    python -m scripts.analyze_stage3_market_structure --start 2023-01-01 --end 2024-12-31

預設純 IS：2023-01-01 ~ 2024-12-31。
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.indicators.market_structure import compute_market_structure  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


def _lookup_at_signal(
    trades: list, ms_series: pd.Series, tf_delta: pd.Timedelta,
) -> np.ndarray:
    """取每筆 trade signal bar（=entry-1tf）的 ms 值。"""
    vals = np.array([""] * len(trades), dtype=object)
    idx = ms_series.index
    for i, t in enumerate(trades):
        signal_ts = pd.Timestamp(t.entry_timestamp) - tf_delta
        pos = idx.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            continue
        v = ms_series.iloc[pos]
        if isinstance(v, str):
            vals[i] = v
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 entry-time market structure distribution."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31",
                        help="結束日（含）；default 2024-12-31（純 IS）")
    parser.add_argument("--pivot-left", type=int, default=10)
    parser.add_argument("--pivot-right", type=int, default=10)
    parser.add_argument("--out-dir", default="results/stage3_market_structure")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    period = PeriodSpec(start=start_ts, end=end_ts)
    cfg = dataclasses.replace(
        base_cfg, in_sample=period, show_progress=False, log_level="WARNING",
    )

    print(f"Period: {start_ts.date()} ~ {end_ts.date()}  tf: {cfg.timeframe}  "
          f"pivot: {args.pivot_left}/{args.pivot_right}")

    print("Running long backtest ...")
    L = run_single_strategy(cfg, direction="long", sample="is")
    print(f"  long trades: {len(L.trades)}")
    print("Running short backtest ...")
    S = run_single_strategy(cfg, direction="short", sample="is")
    print(f"  short trades: {len(S.trades)}")

    print("Loading bars + computing market structure ...")
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m
    ms = compute_market_structure(
        df, pivot_left=args.pivot_left, pivot_right=args.pivot_right,
    )
    tf_delta = df.index[1] - df.index[0]

    # 對每筆 trade 取進場 signal bar 的 ms_trend
    long_trends = _lookup_at_signal(L.trades, ms["ms_trend"], tf_delta)
    short_trends = _lookup_at_signal(S.trades, ms["ms_trend"], tf_delta)
    long_stages = np.array([int(t.final_stage) for t in L.trades])
    short_stages = np.array([int(t.final_stage) for t in S.trades])
    long_pnls = np.array([float(t.net_pnl) for t in L.trades])
    short_pnls = np.array([float(t.net_pnl) for t in S.trades])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 交叉表：direction × stage × ms_trend ----
    rows = []
    for direction, trends, stages, pnls in (
        ("long", long_trends, long_stages, long_pnls),
        ("short", short_trends, short_stages, short_pnls),
    ):
        for stage in (1, 2, 3):
            mask_s = stages == stage
            n_total = int(mask_s.sum())
            for tr in ("bull", "bear", ""):
                mask = mask_s & (trends == tr)
                n = int(mask.sum())
                pct = n / n_total * 100 if n_total > 0 else 0.0
                pnl_sum = float(pnls[mask].sum()) if n else 0.0
                pnl_mean = float(pnls[mask].mean()) if n else 0.0
                wins = int((pnls[mask] > 0).sum()) if n else 0
                win_rate = (wins / n * 100) if n else 0.0
                rows.append({
                    "direction": direction,
                    "stage": stage,
                    "ms_trend": tr or "none",
                    "n": n,
                    "pct_of_stage": round(pct, 1),
                    "pnl_sum": round(pnl_sum, 2),
                    "pnl_mean": round(pnl_mean, 3),
                    "win_rate%": round(win_rate, 1),
                })
    cross = pd.DataFrame(rows)
    csv_path = out_dir / "cross_tab.csv"
    cross.to_csv(csv_path, index=False)
    print(f"\nCross-tab CSV: {csv_path}")

    # ---- Console 摘要 ----
    print("\n=== Cross-tab: direction × stage × ms_trend ===")
    for direction in ("long", "short"):
        sub = cross[cross["direction"] == direction]
        print(f"\n--- {direction.upper()} ---")
        pivot = sub.pivot_table(
            index="stage", columns="ms_trend", values="n", fill_value=0,
        )
        print(pivot.to_string())

    # ---- 同向 / 反向結構摘要 ----
    print("\n=== Aligned-vs-counter structure summary ===")
    print(f"{'direction':<8} {'stage':<6} {'aligned':>9} {'counter':>9} "
          f"{'none':>7} {'aligned%':>10}")
    print("-" * 60)
    for direction, trends, stages in (
        ("long", long_trends, long_stages),
        ("short", short_trends, short_stages),
    ):
        aligned_tr = "bull" if direction == "long" else "bear"
        counter_tr = "bear" if direction == "long" else "bull"
        for stage in (1, 2, 3):
            mask_s = stages == stage
            n_total = int(mask_s.sum())
            if n_total == 0:
                continue
            n_aligned = int((mask_s & (trends == aligned_tr)).sum())
            n_counter = int((mask_s & (trends == counter_tr)).sum())
            n_none = int((mask_s & (trends == "")).sum())
            pct_aligned = n_aligned / n_total * 100
            print(f"{direction:<8} {stage:<6} {n_aligned:>9} {n_counter:>9} "
                  f"{n_none:>7} {pct_aligned:>9.1f}%")

    print(f"\n所有 long trades: {len(L.trades)}  short trades: {len(S.trades)}")


if __name__ == "__main__":
    main()
