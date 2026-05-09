"""比較 baseline vs body_sum / body_sq_sum 在 2 年 IS 上的訊號過濾分佈。

問題：濾網實際把哪一階段的訊號濾掉最多？
做法：
  1. 跑 baseline（mode=off）取得所有 trade 的 final_stage。
  2. 跑 body_sum / body_sq_sum 取得保留的 trade 集合。
  3. 以 (entry_ts, direction) 為 key 比對 baseline ↔ filtered，找出「被濾掉」的 trade。
  4. 對「被濾掉」的 trade 做 final_stage 分桶（1 / 2 / 3）+ 勝率/期望值統計，
     再對「保留」的 trade 做相同統計，回答：濾掉的多半是不是 stage1？

期間：2023-01-01 ~ 2024-12-31（2 年 IS），timeframe 跟隨當前 yaml。
不寫檔到 results/，純 stdout 報表。
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.utils.config import PeriodSpec, SignalFilterConfig, load_config  # noqa: E402

CFG_PATH = "configs/default.yaml"
TWO_YEAR_IS = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)
MODES = ("off", "body_sum", "body_sq_sum")


def _trades_to_df(result) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append({
            "entry_ts": pd.Timestamp(t.entry_timestamp),
            "direction": t.direction.value,
            "exit_reason": t.exit_reason,
            "final_stage": t.final_stage,
            "peak_progress_r": t.peak_progress_r,
            "net_pnl": t.net_pnl,
        })
    return pd.DataFrame(rows)


def _run_one(cfg, mode: str) -> pd.DataFrame:
    sf = SignalFilterConfig(
        mode=mode,
        window=cfg.signal_filter.window,
        threshold=cfg.signal_filter.threshold,
        source=cfg.signal_filter.source,
    )
    cfg_m = dataclasses.replace(cfg, signal_filter=sf, in_sample=TWO_YEAR_IS,
                                show_progress=False, log_level="WARNING")
    long_res = run_single_strategy(cfg_m, direction="long", sample="is")
    short_res = run_single_strategy(cfg_m, direction="short", sample="is")
    df_l = _trades_to_df(long_res)
    df_s = _trades_to_df(short_res)
    return pd.concat([df_l, df_s], ignore_index=True)


def _stage_breakdown(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """以 final_stage 分桶，輸出 count / win_rate / avg_pnl。"""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["is_win"] = df["net_pnl"] > 0
    g = df.groupby("final_stage")
    out = pd.DataFrame({
        "count": g.size(),
        "win_rate%": g["is_win"].mean() * 100,
        "avg_pnl": g["net_pnl"].mean(),
        "sum_pnl": g["net_pnl"].sum(),
    })
    out["share%"] = out["count"] / out["count"].sum() * 100
    out = out[["count", "share%", "win_rate%", "avg_pnl", "sum_pnl"]]
    return out


def main() -> None:
    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(CFG_PATH)

    print(f"\n=== 2-year IS comparison ({TWO_YEAR_IS.start.date()} ~ {TWO_YEAR_IS.end.date()}) "
          f"on {cfg.symbol} {cfg.timeframe} ===\n")

    results: dict[str, pd.DataFrame] = {}
    for mode in MODES:
        print(f"running mode={mode} ...")
        results[mode] = _run_one(cfg, mode)
        print(f"  total trades = {len(results[mode])}")

    base = results["off"]
    base["key"] = list(zip(base["entry_ts"], base["direction"]))
    base_keys = set(base["key"])

    print("\n" + "=" * 78)
    print("Baseline (mode=off) stage breakdown")
    print("=" * 78)
    print(_stage_breakdown(base, "baseline").to_string(float_format="%.3f"))

    for mode in ("body_sum", "body_sq_sum"):
        kept = results[mode]
        kept["key"] = list(zip(kept["entry_ts"], kept["direction"]))
        kept_keys = set(kept["key"])

        # 被濾掉 = 在 baseline，但不在 kept（用 baseline 的 final_stage 來分類）
        filtered_mask = ~base["key"].isin(kept_keys)
        kept_mask = base["key"].isin(kept_keys)
        df_filtered = base.loc[filtered_mask].copy()
        df_kept = base.loc[kept_mask].copy()

        print("\n" + "=" * 78)
        print(f"Mode = {mode}   (baseline trades: {len(base)})")
        print("=" * 78)
        print(f"  filtered out  = {len(df_filtered):>4d}  ({len(df_filtered)/len(base)*100:.1f}%)")
        print(f"  kept (matched)= {len(df_kept):>4d}  ({len(df_kept)/len(base)*100:.1f}%)")
        unmatched = len(kept) - len(df_kept)
        if unmatched:
            print(f"  ⚠ kept-but-not-in-baseline = {unmatched}（同時間直接的訊號因濾網改變導致 R 不同）")

        print("\n  --- 被濾掉的 trade 的 baseline final_stage 分佈 ---")
        print(_stage_breakdown(df_filtered, "filtered").to_string(float_format="%.3f"))
        print("\n  --- 保留下來的 trade 的 baseline final_stage 分佈 ---")
        print(_stage_breakdown(df_kept, "kept").to_string(float_format="%.3f"))

        # 過濾比例（每階段被濾掉的比例）
        per_stage = pd.DataFrame({
            "baseline": base.groupby("final_stage").size(),
            "filtered": df_filtered.groupby("final_stage").size(),
        }).fillna(0).astype(int)
        per_stage["filter_rate%"] = per_stage["filtered"] / per_stage["baseline"] * 100
        print("\n  --- 每階段被濾掉的比例（filter_rate = filtered / baseline）---")
        print(per_stage.to_string(float_format="%.2f"))


if __name__ == "__main__":
    main()
