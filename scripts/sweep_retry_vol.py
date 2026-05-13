"""Long retry × volatility 2D sweep（IS only，無 look-ahead）。

固定：signal_filter=body_sum/win6/th0.73、structure_filter aligned、ADX≥20、
short_max_attempts=1（短側 retry 已知爆 MDD）。

掃描：
- long_max_attempts ∈ {1, 3}
- vol (BBW_rank_min = ATR_rank_min) ∈ {5, 15, 25, 35, 45, 55}

目的：驗證假設「retry 救回遲熟訊號、高 vol 擋雜訊」能不能找到甜蜜點。

執行：
    python -m scripts.sweep_retry_vol
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
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import (  # noqa: E402
    ChopFilterConfig, EntryRetryConfig, load_config,
)


LONG_RETRY_VALUES = [1, 3]
VOL_VALUES = [5, 15, 25, 35, 45, 55]


def stage_summary(trades: list) -> dict:
    n_total = len(trades)
    if n_total == 0:
        return {
            "n": 0,
            **{f"s{s}_n": 0 for s in (1, 2, 3)},
            "total_pnl": 0.0,
            "s3_s1": float("nan"),
        }
    pnls = np.array([float(t.net_pnl) for t in trades])
    stages = np.array([int(t.final_stage) for t in trades])
    out = {"n": n_total}
    for s in (1, 2, 3):
        out[f"s{s}_n"] = int((stages == s).sum())
    out["total_pnl"] = round(float(pnls.sum()), 2)
    out["s3_s1"] = (
        out["s3_n"] / out["s1_n"] if out["s1_n"] > 0 else float("inf")
    )
    return out


def run_combo(base_cfg, *, long_max: int, vol: int) -> dict:
    base_chop = base_cfg.chop_filter
    cfg = dataclasses.replace(
        base_cfg,
        chop_filter=ChopFilterConfig(
            enabled=True,
            bbw_rank_min=float(vol),
            atr_rank_min=float(vol),
            adx_min=base_chop.adx_min,
            bb_period=base_chop.bb_period,
            bb_num_std=base_chop.bb_num_std,
            atr_period=base_chop.atr_period,
            adx_period=base_chop.adx_period,
            rank_window=base_chop.rank_window,
        ),
        entry_retry=EntryRetryConfig(
            long_max_attempts=long_max, short_max_attempts=1,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")
    C = build_merged_result("combined", [L, S])
    long_s = stage_summary(L.trades)
    short_s = stage_summary(S.trades)
    comb_s = stage_summary(L.trades + S.trades)
    mc = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "long": long_s, "short": short_s, "combined": comb_s,
        "ret": mc.total_return_pct, "pf": mc.profit_factor,
        "wr": mc.win_rate_pct, "mdd": mc.max_drawdown_pct,
        "sharpe": mc.sharpe_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long retry × volatility 2D sweep。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out-csv", default="results/retry_vol_sweep/stages.csv",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"IS: {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"Fixed: signal_filter={base_cfg.signal_filter.mode}/win{base_cfg.signal_filter.window}/"
          f"th{base_cfg.signal_filter.threshold}, "
          f"structure={base_cfg.structure_filter.enabled}/{base_cfg.structure_filter.mode}, "
          f"ADX≥{base_cfg.chop_filter.adx_min}, short_max_attempts=1")
    print(f"Sweep: long_max ∈ {LONG_RETRY_VALUES} × vol ∈ {VOL_VALUES} ({len(LONG_RETRY_VALUES)*len(VOL_VALUES)} combos)")
    print()

    results = {}
    all_rows = []
    for lm in LONG_RETRY_VALUES:
        for vol in VOL_VALUES:
            print(f">>> long_max={lm}, vol={vol}")
            r = run_combo(base_cfg, long_max=lm, vol=vol)
            results[(lm, vol)] = r
            for direction in ("long", "short", "combined"):
                row = {"long_max": lm, "vol": vol, "direction": direction,
                       **r[direction]}
                all_rows.append(row)

    # ---------- Combined 主表 ----------
    print()
    print("=" * 130)
    print("Combined metrics by (long_max_attempts × vol)")
    print("=" * 130)
    print(f"{'long_max':>9}  {'vol':>4}  {'n':>5}  "
          f"{'ret%':>8}  {'PF':>5}  {'WR%':>5}  {'MDD%':>5}  {'Sharpe':>6}  "
          f"{'L_n':>4}  {'L_pnl':>8}  {'L_s3/s1':>8}  "
          f"{'S_n':>4}  {'S_pnl':>8}")
    for lm in LONG_RETRY_VALUES:
        for vol in VOL_VALUES:
            r = results[(lm, vol)]
            c = r["combined"]
            l = r["long"]
            s = r["short"]
            ls = l["s3_s1"] if np.isfinite(l["s3_s1"]) else 0.0
            print(f"{lm:>9}  {vol:>4}  {c['n']:>5}  "
                  f"{r['ret']:>+8.2f}  {r['pf']:>5.2f}  {r['wr']:>4.1f}%  "
                  f"{r['mdd']:>4.1f}%  {r['sharpe']:>+6.2f}  "
                  f"{l['n']:>4}  {l['total_pnl']:>+8.2f}  {ls:>8.2f}  "
                  f"{s['n']:>4}  {s['total_pnl']:>+8.2f}")

    # ---------- Delta vs baseline (long_max=1, vol=5) ----------
    print()
    print("=" * 130)
    print("Delta combined PnL vs baseline (long_max=1, vol=5)")
    print("=" * 130)
    base = results[(1, 5)]["combined"]
    base_pnl = base["total_pnl"]
    base_n = base["n"]
    print(f"{'long_max':>9}  {'vol':>4}  {'Δn':>6}  {'ΔPnL':>10}  "
          f"{'L_Δn':>6}  {'L_ΔPnL':>10}")
    for lm in LONG_RETRY_VALUES:
        for vol in VOL_VALUES:
            r = results[(lm, vol)]
            c = r["combined"]
            l = r["long"]
            dn = c["n"] - base_n
            dpnl = c["total_pnl"] - base_pnl
            l_dn = l["n"] - results[(1, 5)]["long"]["n"]
            l_dpnl = l["total_pnl"] - results[(1, 5)]["long"]["total_pnl"]
            print(f"{lm:>9}  {vol:>4}  {dn:>+6}  {dpnl:>+10.2f}  "
                  f"{l_dn:>+6}  {l_dpnl:>+10.2f}")

    # ---------- 排名（combined PnL + 風險調整）----------
    print()
    print("=" * 130)
    print("Top combos by combined PnL (filter: MDD <= 15%, n >= 80)")
    print("=" * 130)
    scored = []
    for (lm, vol), r in results.items():
        if r["mdd"] > 15 or r["combined"]["n"] < 80:
            continue
        scored.append({
            "long_max": lm, "vol": vol,
            "n": r["combined"]["n"],
            "ret": r["ret"], "pf": r["pf"], "mdd": r["mdd"],
            "sharpe": r["sharpe"],
            "pnl": r["combined"]["total_pnl"],
        })
    scored.sort(key=lambda x: x["pnl"], reverse=True)
    print(f"{'rank':>4}  {'long_max':>9}  {'vol':>4}  {'n':>5}  "
          f"{'PnL':>8}  {'ret%':>8}  {'PF':>5}  {'MDD%':>5}  {'Sharpe':>7}")
    for i, row in enumerate(scored, start=1):
        print(f"{i:>4}  {row['long_max']:>9}  {row['vol']:>4}  {row['n']:>5}  "
              f"{row['pnl']:>+8.2f}  {row['ret']:>+8.2f}  {row['pf']:>5.2f}  "
              f"{row['mdd']:>5.1f}  {row['sharpe']:>+7.2f}")

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
