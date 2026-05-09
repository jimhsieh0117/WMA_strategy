"""B：分析「進場後第 1 根 K 的浮盈」與最終 stage 的關聯。

問題：給定 entry K 之後的下一根 K，它的有利浮盈（多單 high、空單 low）達到 X R 後，
後續真正走到 stage 3 的條件機率有多高？這是「動態 cancel」機制的數據基礎。

用法：
    python scripts/analyze_first_bar_response.py [--config configs/default.yaml]
                                                  [--sample is|oos]
                                                  [--out results/first_bar_response.csv]

輸出：
    1. stdout：first-bar peak (R 單位) 分桶 × final_stage 交叉表 + 條件機率
    2. CSV：每筆 trade × first_bar_peak_r / first_bar_peak_pct / final_stage / ...

不改任何核心邏輯，純後分析。entry K = entry_ts；第 1 根 K = entry_ts + 1 × timeframe。
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
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.utils.types import Direction  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)


def _first_bar_peak(
    trades: list,
    df: pd.DataFrame,
    timeframe_delta: pd.Timedelta,
) -> pd.DataFrame:
    """每筆 trade 在 entry_ts + 1×TF 那根的浮盈高點（R 與 % 兩種單位）。"""
    rows = []
    skipped = 0
    for t in trades:
        entry_ts = pd.Timestamp(t.entry_timestamp)
        first_bar_ts = entry_ts + timeframe_delta
        if first_bar_ts not in df.index:
            pos = df.index.searchsorted(first_bar_ts, side="right") - 1
            if pos < 0:
                skipped += 1
                continue
            first_bar_ts = df.index[pos]
            if abs((first_bar_ts - (entry_ts + timeframe_delta)).total_seconds()) > timeframe_delta.total_seconds():
                skipped += 1
                continue
        try:
            high = float(df["high"].loc[first_bar_ts])
            low = float(df["low"].loc[first_bar_ts])
            close = float(df["close"].loc[first_bar_ts])
        except (KeyError, ValueError):
            skipped += 1
            continue

        r_price = abs(t.entry_price - t.stop_history[0][1])
        if r_price <= 0:
            skipped += 1
            continue
        if t.direction is Direction.LONG:
            peak_price = high
            close_pnl = close - t.entry_price
        else:
            peak_price = low
            close_pnl = t.entry_price - close
        first_peak = (peak_price - t.entry_price) if t.direction is Direction.LONG else (t.entry_price - peak_price)
        first_peak_r = first_peak / r_price
        first_peak_pct = first_peak / t.entry_price
        first_close_r = close_pnl / r_price

        rows.append({
            "direction": t.direction.value,
            "entry_ts": entry_ts,
            "first_bar_ts": first_bar_ts,
            "final_stage": int(t.final_stage),
            "peak_progress_r": float(t.peak_progress_r),
            "net_pnl": float(t.net_pnl),
            "r_pct": r_price / t.entry_price,
            "first_peak_r":   first_peak_r,
            "first_peak_pct": first_peak_pct,
            "first_close_r":  first_close_r,
        })
    if skipped:
        print(f"  [info] skipped {skipped} trades (no first-bar)", file=sys.stderr)
    return pd.DataFrame(rows)


def _bucket_by(df: pd.DataFrame, col: str, edges: list[float]) -> pd.DataFrame:
    labels = list(range(len(edges) - 1))
    df = df.copy()
    df["b"] = pd.cut(df[col], bins=edges, labels=labels, right=False, include_lowest=True)
    g = df.groupby("b", observed=True)
    out = pd.DataFrame({
        "lo": [edges[int(b)] for b in g.size().index],
        "hi": [edges[int(b)+1] for b in g.size().index],
        "count": g.size().values,
        "s1": g["final_stage"].apply(lambda x: (x==1).sum()).values,
        "s2": g["final_stage"].apply(lambda x: (x==2).sum()).values,
        "s3": g["final_stage"].apply(lambda x: (x==3).sum()).values,
    })
    out["s1%"] = out["s1"] / out["count"] * 100
    out["s2%"] = out["s2"] / out["count"] * 100
    out["s3%"] = out["s3"] / out["count"] * 100
    out["share%"] = out["count"] / out["count"].sum() * 100
    return out


def _conditional_s3_given_threshold(df: pd.DataFrame, thresholds: list[float], col: str) -> pd.DataFrame:
    """給定 first_peak_r >= X，後續是 stage3 的條件機率。"""
    rows = []
    n_total = len(df)
    n_s3_total = (df.final_stage == 3).sum()
    for thr in thresholds:
        sub = df[df[col] >= thr]
        n = len(sub)
        n_s3 = (sub.final_stage == 3).sum()
        n_s1 = (sub.final_stage == 1).sum()
        rows.append({
            f"{col}>=": thr,
            "n_kept": n,
            "kept%": n / n_total * 100,
            "n_s3_kept": int(n_s3),
            "P(s3|kept)%": n_s3 / n * 100 if n > 0 else float("nan"),
            "s3_recall%": n_s3 / n_s3_total * 100 if n_s3_total > 0 else float("nan"),
            "n_s1_kept": int(n_s1),
            "P(s1|kept)%": n_s1 / n * 100 if n > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--out", default="results/first_bar_response.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    cfg_m = dataclasses.replace(cfg, **overrides)

    print(f"running long ...  ({cfg_m.symbol} {cfg_m.timeframe} "
          f"{cfg_m.in_sample.start.date()} ~ {cfg_m.in_sample.end.date()})")
    L = run_single_strategy(cfg_m, direction="long", sample=args.sample)
    print("running short ...")
    S = run_single_strategy(cfg_m, direction="short", sample=args.sample)

    period = cfg_m.in_sample if args.sample == "is" else cfg_m.out_of_sample
    raw1m = load_ohlcv(cfg_m.source_parquet, start=period.start, end=period.end)
    df_tf = resample(raw1m, cfg_m.timeframe) if cfg_m.timeframe != "1m" else raw1m
    timeframe_delta = df_tf.index[1] - df_tf.index[0]

    df = _first_bar_peak(L.trades + S.trades, df_tf, timeframe_delta)
    n = len(df)
    print(f"\ntotal trades analysed: {n}")
    n1, n2, n3 = (df.final_stage == 1).sum(), (df.final_stage == 2).sum(), (df.final_stage == 3).sum()
    print(f"  s1: {n1} ({n1/n*100:.1f}%),  s2: {n2} ({n2/n*100:.1f}%),  s3: {n3} ({n3/n*100:.1f}%)")

    print("\n=== first_bar peak (R 單位) 分桶 × final_stage ===")
    edges_r = [-1.5, -1.0, -0.5, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    out_r = _bucket_by(df, "first_peak_r", edges_r)
    pretty = pd.DataFrame({
        "range": [f"{r['lo']:+.2f}~{r['hi']:+.2f}" for _, r in out_r.iterrows()],
        "count": out_r["count"], "share%": out_r["share%"],
        "s1%": out_r["s1%"], "s2%": out_r["s2%"], "s3%": out_r["s3%"],
    })
    print(pretty.to_string(index=False, float_format="%.2f"))

    print("\n=== 條件機率：first_peak_r ≥ X 時，最終 stage 分佈 ===")
    cond = _conditional_s3_given_threshold(df, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2], "first_peak_r")
    print(cond.to_string(index=False, float_format="%.2f"))

    print("\n=== first_bar 收盤（first_close_r）分桶 — 看「收盤站穩浮盈」是否更可靠 ===")
    edges_c = [-1.5, -1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 5.0]
    out_c = _bucket_by(df, "first_close_r", edges_c)
    pretty_c = pd.DataFrame({
        "range": [f"{r['lo']:+.2f}~{r['hi']:+.2f}" for _, r in out_c.iterrows()],
        "count": out_c["count"], "share%": out_c["share%"],
        "s1%": out_c["s1%"], "s2%": out_c["s2%"], "s3%": out_c["s3%"],
    })
    print(pretty_c.to_string(index=False, float_format="%.2f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
