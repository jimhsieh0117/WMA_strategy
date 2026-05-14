"""梯度放寬濾網對照（IS only，避免 OOS look-ahead）。

目的：當前 baseline (A) 在 IS 表現亮眼但 OOS 退化嚴重，懷疑層層挑優過擬。
逐步放寬濾網，觀察訊號密度 / PF / WR / MDD / Sharpe 的變化，找品質還在
但訊號明顯變多的點，降低 IS 對單一參數組合的依賴。

固定：
- entry_retry: 長短各 1（保留昨天加的機制）
- structure_filter pivot 10/10
- signal_filter mode=body_sum, window=6（門檻 0.65~0.73 微調）

掃描 5 組梯度配置：
    A current : vol=45 ADX=20 th=0.73 structure=aligned
    B 微鬆    : vol=35 ADX=20 th=0.70 structure=aligned
    C 中鬆    : vol=25 ADX=15 th=0.70 structure=aligned
    D 大鬆    : vol=15 ADX=15 th=0.65 structure=exclude_counter
    E 最寬    : vol= 5 ADX=15 th=0.65 structure=off

執行：
    python -m scripts.sweep_relax_filters
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
    ChopFilterConfig,
    SignalFilterConfig,
    StructureFilterConfig,
    load_config,
)


# (label, vol, adx, signal_threshold, structure_mode_or_off)
CONFIGS = [
    ("A current", 45, 20, 0.73, "aligned"),
    ("B 微鬆",    35, 20, 0.70, "aligned"),
    ("C 中鬆",    25, 15, 0.70, "aligned"),
    ("D 大鬆",    15, 15, 0.65, "exclude_counter"),
    ("E 最寬",     5, 15, 0.65, "off"),
]


def stage_summary(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "s1": 0, "s2": 0, "s3": 0, "pnl": 0.0}
    pnls = np.array([float(t.net_pnl) for t in trades])
    stages = np.array([int(t.final_stage) for t in trades])
    return {
        "n": n,
        "s1": int((stages == 1).sum()),
        "s2": int((stages == 2).sum()),
        "s3": int((stages == 3).sum()),
        "pnl": round(float(pnls.sum()), 2),
    }


def build_cfg(base_cfg, vol: int, adx: int, th: float, structure_mode: str):
    base_chop = base_cfg.chop_filter
    base_sig = base_cfg.signal_filter
    base_struct = base_cfg.structure_filter
    return dataclasses.replace(
        base_cfg,
        chop_filter=ChopFilterConfig(
            enabled=True,
            bbw_rank_min=float(vol),
            atr_rank_min=float(vol),
            adx_min=float(adx),
            bb_period=base_chop.bb_period,
            bb_num_std=base_chop.bb_num_std,
            atr_period=base_chop.atr_period,
            adx_period=base_chop.adx_period,
            rank_window=base_chop.rank_window,
        ),
        signal_filter=SignalFilterConfig(
            mode=base_sig.mode,
            window=base_sig.window,
            threshold=float(th),
        ),
        structure_filter=StructureFilterConfig(
            enabled=(structure_mode != "off"),
            mode=structure_mode if structure_mode != "off" else "aligned",
            pivot_left=base_struct.pivot_left,
            pivot_right=base_struct.pivot_right,
        ),
        show_progress=False,
        log_level="WARNING",
    )


def run_one(base_cfg, label: str, vol: int, adx: int, th: float, sm: str) -> dict:
    cfg = build_cfg(base_cfg, vol, adx, th, sm)
    L = run_single_strategy(cfg, direction="long", sample="is")
    S = run_single_strategy(cfg, direction="short", sample="is")
    C = build_merged_result("combined", [L, S])
    mL = compute_metrics(L, timeframe=cfg.timeframe)
    mS = compute_metrics(S, timeframe=cfg.timeframe)
    mC = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "label": label,
        "vol": vol, "adx": adx, "th": th, "struct": sm,
        "long": {**stage_summary(L.trades),
                 "pf": mL.profit_factor, "wr": mL.win_rate_pct,
                 "ret": mL.total_return_pct, "mdd": mL.max_drawdown_pct,
                 "sharpe": mL.sharpe_ratio},
        "short": {**stage_summary(S.trades),
                  "pf": mS.profit_factor, "wr": mS.win_rate_pct,
                  "ret": mS.total_return_pct, "mdd": mS.max_drawdown_pct,
                  "sharpe": mS.sharpe_ratio},
        "comb": {"n": len(L.trades) + len(S.trades),
                 "pf": mC.profit_factor, "wr": mC.win_rate_pct,
                 "ret": mC.total_return_pct, "mdd": mC.max_drawdown_pct,
                 "sharpe": mC.sharpe_ratio},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-csv", default="results/relax_sweep/summary.csv")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}  "
          f"IS: {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"Fixed: signal_filter={base_cfg.signal_filter.mode}/win{base_cfg.signal_filter.window}, "
          f"entry_retry=(L={base_cfg.entry_retry.long_max_attempts}, "
          f"S={base_cfg.entry_retry.short_max_attempts})")
    print(f"Sweep: {len(CONFIGS)} 組\n")

    rows = []
    for label, vol, adx, th, sm in CONFIGS:
        print(f">>> {label}: vol={vol} ADX={adx} th={th} struct={sm}")
        r = run_one(base_cfg, label, vol, adx, th, sm)
        rows.append(r)

    # ---------- Combined main table ----------
    print()
    print("=" * 120)
    print("Combined IS metrics (long + short)")
    print("=" * 120)
    print(f"{'label':<10}  {'vol':>3}  {'adx':>3}  {'th':>5}  {'struct':<16}  "
          f"{'n':>4}  {'ret%':>7}  {'PF':>5}  {'WR%':>5}  {'MDD%':>5}  {'Sharpe':>7}")
    for r in rows:
        c = r["comb"]
        print(f"{r['label']:<10}  {r['vol']:>3}  {r['adx']:>3}  {r['th']:>5}  "
              f"{r['struct']:<16}  {c['n']:>4}  {c['ret']:>+7.2f}  "
              f"{c['pf']:>5.2f}  {c['wr']:>4.1f}%  {c['mdd']:>4.1f}%  "
              f"{c['sharpe']:>+7.2f}")

    # ---------- Per-direction breakdown ----------
    print()
    print("=" * 120)
    print("Per-direction breakdown")
    print("=" * 120)
    print(f"{'label':<10}  {'dir':<5}  {'n':>4}  "
          f"{'s1/s2/s3':>10}  {'ret%':>7}  {'PF':>5}  {'WR%':>5}  "
          f"{'MDD%':>5}  {'Sharpe':>7}")
    for r in rows:
        for d in ("long", "short"):
            x = r[d]
            stages = f"{x['s1']}/{x['s2']}/{x['s3']}"
            print(f"{r['label']:<10}  {d:<5}  {x['n']:>4}  "
                  f"{stages:>10}  {x['ret']:>+7.2f}  {x['pf']:>5.2f}  "
                  f"{x['wr']:>4.1f}%  {x['mdd']:>4.1f}%  {x['sharpe']:>+7.2f}")

    # ---------- Signal density delta ----------
    print()
    print("=" * 120)
    print("Δ vs A current (combined)")
    print("=" * 120)
    base = rows[0]["comb"]
    print(f"{'label':<10}  {'Δn':>5}  {'Δret%':>7}  {'ΔPF':>6}  "
          f"{'ΔMDD%':>7}  {'ΔSharpe':>8}")
    for r in rows:
        c = r["comb"]
        print(f"{r['label']:<10}  {c['n'] - base['n']:>+5}  "
              f"{c['ret'] - base['ret']:>+7.2f}  "
              f"{c['pf'] - base['pf']:>+6.2f}  "
              f"{c['mdd'] - base['mdd']:>+7.2f}  "
              f"{c['sharpe'] - base['sharpe']:>+8.2f}")

    # ---------- CSV ----------
    flat = []
    for r in rows:
        for d in ("long", "short", "comb"):
            x = r[d]
            flat.append({
                "label": r["label"], "vol": r["vol"], "adx": r["adx"],
                "th": r["th"], "struct": r["struct"], "dir": d,
                "n": x["n"],
                "s1": x.get("s1", None), "s2": x.get("s2", None),
                "s3": x.get("s3", None),
                "ret": x["ret"], "pf": x["pf"], "wr": x["wr"],
                "mdd": x["mdd"], "sharpe": x["sharpe"],
            })
    df = pd.DataFrame(flat)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV: {out.resolve()}")


if __name__ == "__main__":
    main()
