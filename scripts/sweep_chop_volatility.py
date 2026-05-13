"""Chop filter 波動門檻放寬 sweep（IS only，無 look-ahead）。

固定 ADX=20、signal_filter 用當前 yaml（window=6, threshold=0.73），
同步放寬 BBW_rank_min 與 ATR_rank_min：
    {40, 35, 30, 25, 20, 15, 10, 5, 0}

對每組記錄：
- n_total / s1 / s2 / s3 分布
- 多空各自 PnL、勝率
- combined PF / MDD / Sharpe / total return

目標：找到「放寬波動後保留品質、增加獲利」的甜蜜點，
而非單純放更多訊號（會降低 s3/s1 比率）。

執行：
    python -m scripts.sweep_chop_volatility
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
from src.utils.config import ChopFilterConfig, load_config  # noqa: E402


VOL_THRESHOLDS = [40, 35, 30, 25, 20, 15, 10, 5, 0]


def stage_summary(trades: list, direction: str) -> dict:
    n_total = len(trades)
    out: dict = {"direction": direction, "n_total": n_total}
    if n_total == 0:
        for s in (1, 2, 3):
            out[f"s{s}_n"] = 0
            out[f"s{s}_pct"] = 0.0
            out[f"s{s}_pnl"] = 0.0
            out[f"s{s}_wr"] = 0.0
        out["total_pnl"] = 0.0
        return out
    pnls = np.array([float(t.net_pnl) for t in trades])
    stages = np.array([int(t.final_stage) for t in trades])
    for s in (1, 2, 3):
        mask = stages == s
        n = int(mask.sum())
        out[f"s{s}_n"] = n
        out[f"s{s}_pct"] = round(n / n_total * 100, 1)
        out[f"s{s}_pnl"] = round(float(pnls[mask].sum()), 2)
        out[f"s{s}_wr"] = (
            round(float((pnls[mask] > 0).mean()) * 100, 1) if n else 0.0
        )
    out["total_pnl"] = round(float(pnls.sum()), 2)
    return out


def run_combo(base_cfg, *, vol_min: int) -> dict:
    """跑一個 chop 設定，回傳 long / short / combined 結果與 metrics。"""
    base_chop = base_cfg.chop_filter
    cfg = dataclasses.replace(
        base_cfg,
        chop_filter=ChopFilterConfig(
            enabled=True,
            bbw_rank_min=float(vol_min),
            atr_rank_min=float(vol_min),
            adx_min=base_chop.adx_min,     # 固定不動
            bb_period=base_chop.bb_period,
            bb_num_std=base_chop.bb_num_std,
            atr_period=base_chop.atr_period,
            adx_period=base_chop.adx_period,
            rank_window=base_chop.rank_window,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")
    C = build_merged_result("combined", [L, S])

    long_s = stage_summary(L.trades, "long")
    short_s = stage_summary(S.trades, "short")
    comb_s = stage_summary(L.trades + S.trades, "combined")
    mc = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "long": long_s,
        "short": short_s,
        "combined": comb_s,
        "pf": mc.profit_factor,
        "wr": mc.win_rate_pct,
        "mdd": mc.max_drawdown_pct,
        "sharpe": mc.sharpe_ratio,
        "ret": mc.total_return_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chop filter 波動門檻放寬 sweep（IS only）。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out-csv", default="results/chop_vol_sweep/stages.csv",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"IS: {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"Fixed: ADX≥{base_cfg.chop_filter.adx_min}, "
          f"signal_filter={base_cfg.signal_filter.mode}/win{base_cfg.signal_filter.window}/"
          f"th{base_cfg.signal_filter.threshold}, "
          f"structure_filter={base_cfg.structure_filter.enabled}/"
          f"{base_cfg.structure_filter.mode}")
    print(f"Sweeping bbw_rank_min == atr_rank_min ∈ {VOL_THRESHOLDS}")
    print()

    results = {}
    all_rows = []
    for v in VOL_THRESHOLDS:
        print(f">>> vol_rank_min = {v}")
        r = run_combo(base_cfg, vol_min=v)
        results[v] = r
        for direction in ("long", "short", "combined"):
            row = {"vol_rank_min": v, "adx_min": base_cfg.chop_filter.adx_min,
                   **r[direction]}
            all_rows.append(row)

    # ---------- 主表：combined + metrics ----------
    print()
    print("=" * 130)
    print("Combined metrics (long + short)")
    print("=" * 130)
    print(f"{'vol≥':>5}  {'n':>5}  {'s1n':>5} {'s1%':>5} {'s2n':>5} {'s2%':>5} "
          f"{'s3n':>5} {'s3%':>5}  {'s3/s1':>6}  "
          f"{'ret%':>8}  {'PF':>5}  {'WR%':>5}  {'MDD%':>5}  {'Sharpe':>6}  {'pnl':>8}")
    for v in VOL_THRESHOLDS:
        r = results[v]
        c = r["combined"]
        s3_s1 = c["s3_n"] / c["s1_n"] if c["s1_n"] > 0 else float("inf")
        print(f"{v:>5}  {c['n_total']:>5}  "
              f"{c['s1_n']:>5} {c['s1_pct']:>4.1f}%  "
              f"{c['s2_n']:>5} {c['s2_pct']:>4.1f}%  "
              f"{c['s3_n']:>5} {c['s3_pct']:>4.1f}%  "
              f"{s3_s1:>6.2f}  "
              f"{r['ret']:>+8.2f}  {r['pf']:>5.2f}  {r['wr']:>4.1f}%  "
              f"{r['mdd']:>4.1f}%  {r['sharpe']:>+6.2f}  {c['total_pnl']:>+8.2f}")

    # ---------- 分 direction ----------
    print()
    print("=" * 130)
    print("By direction × stage")
    print("=" * 130)
    for direction in ("long", "short"):
        print(f"\n  [{direction}]")
        print(f"  {'vol≥':>5}  {'n':>4}  "
              f"{'s1n':>4} {'s1pnl':>8} {'s1wr':>5}  "
              f"{'s2n':>4} {'s2pnl':>8} {'s2wr':>5}  "
              f"{'s3n':>4} {'s3pnl':>8} {'s3wr':>5}  "
              f"{'total':>9}")
        for v in VOL_THRESHOLDS:
            d = results[v][direction]
            print(f"  {v:>5}  {d['n_total']:>4}  "
                  f"{d['s1_n']:>4} {d['s1_pnl']:>+8.2f} {d['s1_wr']:>4.1f}%  "
                  f"{d['s2_n']:>4} {d['s2_pnl']:>+8.2f} {d['s2_wr']:>4.1f}%  "
                  f"{d['s3_n']:>4} {d['s3_pnl']:>+8.2f} {d['s3_wr']:>4.1f}%  "
                  f"{d['total_pnl']:>+9.2f}")

    # ---------- 增量分析（vs baseline 40）----------
    print()
    print("=" * 130)
    print("Delta vs baseline (vol≥40)")
    print("=" * 130)
    base = results[40]
    base_n = base["combined"]["n_total"]
    base_s3 = base["combined"]["s3_n"]
    base_pnl = base["combined"]["total_pnl"]
    print(f"{'vol≥':>5}  {'Δn':>6}  {'Δs1':>5}  {'Δs2':>5}  {'Δs3':>5}  "
          f"{'Δpnl':>8}  {'new s3/(new total)':>22}")
    for v in VOL_THRESHOLDS:
        c = results[v]["combined"]
        dn = c["n_total"] - base_n
        ds1 = c["s1_n"] - base["combined"]["s1_n"]
        ds2 = c["s2_n"] - base["combined"]["s2_n"]
        ds3 = c["s3_n"] - base_s3
        dpnl = c["total_pnl"] - base_pnl
        # 邊際比率：新增的訂單裡有多少 % 是 stage 3？
        new_s3_ratio = (ds3 / dn * 100) if dn != 0 else 0.0
        new_s1_ratio = (ds1 / dn * 100) if dn != 0 else 0.0
        print(f"{v:>5}  {dn:>+6}  {ds1:>+5}  {ds2:>+5}  {ds3:>+5}  "
              f"{dpnl:>+8.2f}  "
              f"new s3={new_s3_ratio:>5.1f}% / new s1={new_s1_ratio:>5.1f}%")

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
