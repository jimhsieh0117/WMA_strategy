"""Short 端 filter 通過率審計。

模擬 short_strategy._try_entry 的條件鏈，逐根 K 統計每個 filter 攔截多少 WMA 死叉訊號。
（retry=1：只看 cross 那根 K，不重試 — 跟目前 config short_max_attempts=1 一致。）

階段順序：
    cross               WMA 死叉
    structure_obc       open[bar-2] > close[bar]
    signal_filter       body_sum / body_sq_sum 比例
    chop_filter         BBW_rank / ATR_rank / ADX AND
    structure_filter    market structure aligned
    stage1_safety       swing_high*(1+buf) > close 且 R/entry ≥ r_min_pct

用法：
    .venv/bin/python -m scripts.audit_short_filters [--sample is|oos]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.strategy.base import (  # noqa: E402
    passes_chop_filter, passes_signal_filter,
    passes_structure_filter, prepare_indicators,
)
from src.strategy.types import (  # noqa: E402
    ChopFilterParams, EntryRetryParams, RCapParams, SignalFilterParams,
    StrategyParams, StructureFilterParams, TrailingStopParams,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.types import Direction  # noqa: E402


def build_short_params(cfg) -> StrategyParams:
    trailing = TrailingStopParams(
        swing_lookback=cfg.trailing.swing_lookback,
        stage1_slippage_buffer=cfg.trailing.stage1_slippage_buffer,
        stage2_normal_trigger_r=cfg.trailing.stage2_normal_trigger_r,
        stage2_abnormal_trigger_r=cfg.trailing.stage2_abnormal_trigger_r,
        stage2_buffer_r=cfg.trailing.stage2_buffer_r,
        stage2_pct_trigger=cfg.trailing.stage2_pct_trigger,
        stage3_normal_trigger_r=cfg.trailing.stage3_normal_trigger_r,
        stage3_abnormal_trigger_r=cfg.trailing.stage3_abnormal_trigger_r,
        bollinger_period=cfg.trailing.bollinger_period,
        bollinger_num_std=cfg.trailing.bollinger_num_std,
        stage3_mode=cfg.trailing.stage3_mode,  # type: ignore[arg-type]
        r_ladder_normal_first_trigger=cfg.trailing.r_ladder_normal_first_trigger,
        r_ladder_normal_step=cfg.trailing.r_ladder_normal_step,
        r_ladder_abnormal_first_trigger=cfg.trailing.r_ladder_abnormal_first_trigger,
        r_ladder_abnormal_step=cfg.trailing.r_ladder_abnormal_step,
        r_ladder_trigger_offset=cfg.trailing.r_ladder_trigger_offset,
        r_ladder_abnormal_trigger_offset=cfg.trailing.r_ladder_abnormal_trigger_offset,
        early_exit_enabled=cfg.trailing.early_exit_enabled,
        early_exit_observation_bars=cfg.trailing.early_exit_observation_bars,
        early_exit_metric=cfg.trailing.early_exit_metric,  # type: ignore[arg-type]
        early_exit_min_peak_r=cfg.trailing.early_exit_min_peak_r,
        early_exit_min_peak_pct=cfg.trailing.early_exit_min_peak_pct,
        early_exit_min_close_r=cfg.trailing.early_exit_min_close_r,
        stage1_time_cut_enabled=cfg.trailing.stage1_time_cut_enabled,
        stage1_time_cut_bars=cfg.trailing.stage1_time_cut_bars,
        stage1_time_cut_peak_r_max=cfg.trailing.stage1_time_cut_peak_r_max,
    )
    return StrategyParams(
        wma_fast=cfg.wma_short_fast,
        wma_slow=cfg.wma_short_slow,
        trailing=trailing,
        signal_filter=SignalFilterParams(
            mode=cfg.signal_filter.mode,  # type: ignore[arg-type]
            window=cfg.signal_filter.window,
            threshold=cfg.signal_filter.threshold,
        ),
        r_cap=RCapParams(
            mode=cfg.r_cap.mode,  # type: ignore[arg-type]
            window=cfg.r_cap.window,
        ),
        chop_filter=ChopFilterParams(
            enabled=cfg.chop_filter.enabled,
            bbw_rank_min=cfg.chop_filter.bbw_rank_min,
            atr_rank_min=cfg.chop_filter.atr_rank_min,
            adx_min=cfg.chop_filter.adx_min,
            bb_period=cfg.chop_filter.bb_period,
            bb_num_std=cfg.chop_filter.bb_num_std,
            atr_period=cfg.chop_filter.atr_period,
            adx_period=cfg.chop_filter.adx_period,
            rank_window=cfg.chop_filter.rank_window,
        ),
        structure_filter=StructureFilterParams(
            enabled=cfg.structure_filter.enabled,
            mode=cfg.structure_filter.mode,
            pivot_left=cfg.structure_filter.pivot_left,
            pivot_right=cfg.structure_filter.pivot_right,
        ),
        entry_retry=EntryRetryParams(
            long_max_attempts=cfg.entry_retry.long_max_attempts,
            short_max_attempts=cfg.entry_retry.short_max_attempts,
        ),
    )


def detect_short_cross(df: pd.DataFrame, i: int) -> bool:
    if i < 1:
        return False
    wf_t = df["wma_fast"].iat[i]
    wf_p = df["wma_fast"].iat[i - 1]
    ws_t = df["wma_slow"].iat[i]
    ws_p = df["wma_slow"].iat[i - 1]
    if (math.isnan(wf_t) or math.isnan(wf_p)
            or math.isnan(ws_t) or math.isnan(ws_p)):
        return False
    return (wf_t < ws_t) and (wf_p >= ws_p)


def audit(cfg, sample: str) -> dict:
    period = cfg.in_sample if sample == "is" else cfg.out_of_sample
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m
    params = build_short_params(cfg)
    aug = prepare_indicators(df, params)

    n_bars = len(aug)
    counts = {
        "cross": 0,
        "structure_obc": 0,
        "signal_filter": 0,
        "chop_filter": 0,
        "structure_filter": 0,
        "stage1_safety": 0,
    }
    reject_at = {  # 紀錄是被哪一步擋下
        "structure_obc": 0,
        "signal_filter": 0,
        "chop_filter": 0,
        "structure_filter": 0,
        "stage1_safety": 0,
    }

    for i in range(n_bars):
        if not detect_short_cross(aug, i):
            continue
        counts["cross"] += 1

        # 結構 open[i-2] > close[i]
        if i < 2:
            reject_at["structure_obc"] += 1
            continue
        c_t = aug["close"].iat[i]
        o_t2 = aug["open"].iat[i - 2]
        if math.isnan(c_t) or math.isnan(o_t2) or not (o_t2 > c_t):
            reject_at["structure_obc"] += 1
            continue
        counts["structure_obc"] += 1

        # signal_filter
        if not passes_signal_filter(aug, i, Direction.SHORT, params):
            reject_at["signal_filter"] += 1
            continue
        counts["signal_filter"] += 1

        # chop_filter
        if not passes_chop_filter(aug, i, params):
            reject_at["chop_filter"] += 1
            continue
        counts["chop_filter"] += 1

        # structure_filter
        if not passes_structure_filter(aug, i, Direction.SHORT, params):
            reject_at["structure_filter"] += 1
            continue
        counts["structure_filter"] += 1

        # stage 1 safety + r_min_pct
        n = params.trailing.swing_lookback
        if i + 1 < n:
            reject_at["stage1_safety"] += 1
            continue
        swing_high = float(aug["high"].iloc[i - n + 1 : i + 1].max())
        initial_stop = swing_high * (1.0 + params.trailing.stage1_slippage_buffer)
        if initial_stop <= c_t:
            reject_at["stage1_safety"] += 1
            continue
        r = (initial_stop - c_t) / c_t
        if r < cfg.r_min_pct:
            reject_at["stage1_safety"] += 1
            continue
        counts["stage1_safety"] += 1

    return {"n_bars": n_bars, "counts": counts, "reject_at": reject_at}


def fmt_pct(n: int, base: int) -> str:
    if base == 0:
        return "  —  "
    return f"{n / base * 100:5.1f}%"


def print_report(sample: str, result: dict) -> None:
    c = result["counts"]
    r = result["reject_at"]
    cross = c["cross"]
    print()
    print("=" * 72)
    print(f"Short filter audit — sample={sample}  bars={result['n_bars']}")
    print("=" * 72)
    print(f"{'stage':<22} {'pass':>8} {'pass%':>7} {'cum%':>7} {'rejected':>10}")
    print("-" * 72)
    stages = [
        ("cross",              c["cross"],            cross,                  0),
        ("structure (o>c)",    c["structure_obc"],    c["cross"],             r["structure_obc"]),
        ("signal_filter",      c["signal_filter"],    c["structure_obc"],     r["signal_filter"]),
        ("chop_filter",        c["chop_filter"],      c["signal_filter"],     r["chop_filter"]),
        ("structure_filter",   c["structure_filter"], c["chop_filter"],       r["structure_filter"]),
        ("stage1_safety",      c["stage1_safety"],    c["structure_filter"],  r["stage1_safety"]),
    ]
    for name, passed, base, rejected in stages:
        cum = fmt_pct(passed, cross) if cross else "  —  "
        step = fmt_pct(passed, base) if base else "  —  "
        print(f"{name:<22} {passed:>8} {step:>7} {cum:>7} {rejected:>10}")
    print("-" * 72)
    print(f"final entries (= cross × cum pass): {c['stage1_safety']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--sample", choices=["is", "oos", "both"], default="both")
    args = p.parse_args()
    cfg = load_config(args.config)
    print(f"Symbol: {cfg.symbol}  tf: {cfg.timeframe}")
    print(f"Short WMA: fast={cfg.wma_short_fast} / slow={cfg.wma_short_slow}")
    print(f"Filters: signal={cfg.signal_filter.mode} thr={cfg.signal_filter.threshold} | "
          f"chop bbw≥{cfg.chop_filter.bbw_rank_min} atr≥{cfg.chop_filter.atr_rank_min} "
          f"adx≥{cfg.chop_filter.adx_min} | "
          f"struct={'on' if cfg.structure_filter.enabled else 'off'} ({cfg.structure_filter.mode})")
    samples = ["is", "oos"] if args.sample == "both" else [args.sample]
    for s in samples:
        result = audit(cfg, s)
        print_report(s, result)


if __name__ == "__main__":
    main()
