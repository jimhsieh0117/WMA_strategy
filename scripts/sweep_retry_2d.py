"""Long × Short retry 2D sweep（IS + OOS 同時對照）。

目的：在當前 baseline 濾網組合下，掃 long_max × short_max ∈ {1, 2, 3}，
看「給 WMA 交叉後 N 根 K 重試機會」是否能找到 IS / OOS 都站得住的穩定 alpha。

固定：vol=45 ADX=20 th=0.73 structure=aligned（沿用當前 default.yaml）

執行：
    python -m scripts.sweep_retry_2d
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
from src.utils.config import EntryRetryConfig, load_config  # noqa: E402


LONG_VALUES = [1, 2, 3]
SHORT_VALUES = [1, 2, 3]


def stage_pnl(trades: list) -> dict:
    if not trades:
        return {"pnl": 0.0}
    pnls = np.array([float(t.net_pnl) for t in trades])
    return {"pnl": round(float(pnls.sum()), 2)}


def run_one(base_cfg, *, long_max: int, short_max: int, sample: str) -> dict:
    cfg = dataclasses.replace(
        base_cfg,
        entry_retry=EntryRetryConfig(
            long_max_attempts=long_max, short_max_attempts=short_max,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    C = build_merged_result(f"combined_{sample}", [L, S])
    mL = compute_metrics(L, timeframe=cfg.timeframe)
    mS = compute_metrics(S, timeframe=cfg.timeframe)
    mC = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "long": {"n": len(L.trades), **stage_pnl(L.trades),
                 "pf": mL.profit_factor, "ret": mL.total_return_pct,
                 "wr": mL.win_rate_pct, "mdd": mL.max_drawdown_pct,
                 "sharpe": mL.sharpe_ratio},
        "short": {"n": len(S.trades), **stage_pnl(S.trades),
                  "pf": mS.profit_factor, "ret": mS.total_return_pct,
                  "wr": mS.win_rate_pct, "mdd": mS.max_drawdown_pct,
                  "sharpe": mS.sharpe_ratio},
        "comb": {"n": len(L.trades) + len(S.trades),
                 "pf": mC.profit_factor, "ret": mC.total_return_pct,
                 "wr": mC.win_rate_pct, "mdd": mC.max_drawdown_pct,
                 "sharpe": mC.sharpe_ratio},
    }


def print_table(title: str, results: dict, sample: str) -> None:
    print()
    print("=" * 110)
    print(f"{title}  [{sample.upper()}]")
    print("=" * 110)
    print(f"{'L_max':>5}  {'S_max':>5}  {'n':>4}  {'ret%':>8}  {'PF':>5}  "
          f"{'WR%':>5}  {'MDD%':>5}  {'Sharpe':>7}  "
          f"{'L_n':>4}  {'L_PF':>5}  {'L_ret%':>7}  "
          f"{'S_n':>4}  {'S_PF':>5}  {'S_ret%':>7}")
    for lm in LONG_VALUES:
        for sm in SHORT_VALUES:
            r = results[(lm, sm)][sample]
            c, l, s = r["comb"], r["long"], r["short"]
            print(f"{lm:>5}  {sm:>5}  {c['n']:>4}  {c['ret']:>+8.2f}  "
                  f"{c['pf']:>5.2f}  {c['wr']:>4.1f}%  {c['mdd']:>4.1f}%  "
                  f"{c['sharpe']:>+7.2f}  "
                  f"{l['n']:>4}  {l['pf']:>5.2f}  {l['ret']:>+7.2f}  "
                  f"{s['n']:>4}  {s['pf']:>5.2f}  {s['ret']:>+7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-csv", default="results/retry_2d/summary.csv")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"IS:  {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"OOS: {base_cfg.out_of_sample.start.date()} ~ {base_cfg.out_of_sample.end}")
    print(f"Fixed: chop vol={base_cfg.chop_filter.bbw_rank_min}/ADX={base_cfg.chop_filter.adx_min}, "
          f"signal_filter={base_cfg.signal_filter.mode}/win{base_cfg.signal_filter.window}/"
          f"th{base_cfg.signal_filter.threshold}, "
          f"structure={base_cfg.structure_filter.enabled}/{base_cfg.structure_filter.mode}")
    print(f"Sweep: long_max × short_max ∈ {{1, 2, 3}} = 9 combos, IS + OOS\n")

    results: dict = {}
    for lm in LONG_VALUES:
        for sm in SHORT_VALUES:
            print(f">>> long_max={lm}, short_max={sm}")
            results[(lm, sm)] = {
                "is": run_one(base_cfg, long_max=lm, short_max=sm, sample="is"),
                "oos": run_one(base_cfg, long_max=lm, short_max=sm, sample="oos"),
            }

    print_table("Combined retry sweep", results, "is")
    print_table("Combined retry sweep", results, "oos")

    # ---------- IS↔OOS robustness：兩邊都要正、PF 不要崩太多 ----------
    print()
    print("=" * 110)
    print("Robustness 排行（IS+OOS 都 ret>0 且 PF>=1，按 OOS ret 排）")
    print("=" * 110)
    scored = []
    for (lm, sm), r in results.items():
        ci, co = r["is"]["comb"], r["oos"]["comb"]
        if ci["ret"] <= 0 or co["ret"] <= 0 or ci["pf"] < 1 or co["pf"] < 1:
            continue
        scored.append({
            "lm": lm, "sm": sm,
            "is_n": ci["n"], "is_ret": ci["ret"], "is_pf": ci["pf"],
            "is_sharpe": ci["sharpe"], "is_mdd": ci["mdd"],
            "oos_n": co["n"], "oos_ret": co["ret"], "oos_pf": co["pf"],
            "oos_sharpe": co["sharpe"], "oos_mdd": co["mdd"],
        })
    scored.sort(key=lambda x: x["oos_ret"], reverse=True)
    if not scored:
        print("(無組合通過 robustness filter)")
    else:
        print(f"{'L_max':>5}  {'S_max':>5}  "
              f"{'IS_n':>4}  {'IS_ret%':>7}  {'IS_PF':>5}  {'IS_MDD':>6}  "
              f"{'OOS_n':>5}  {'OOS_ret%':>8}  {'OOS_PF':>6}  {'OOS_MDD':>7}")
        for r in scored:
            print(f"{r['lm']:>5}  {r['sm']:>5}  "
                  f"{r['is_n']:>4}  {r['is_ret']:>+7.2f}  {r['is_pf']:>5.2f}  "
                  f"{r['is_mdd']:>5.1f}%  "
                  f"{r['oos_n']:>5}  {r['oos_ret']:>+8.2f}  {r['oos_pf']:>6.2f}  "
                  f"{r['oos_mdd']:>6.1f}%")

    # ---------- CSV ----------
    flat = []
    for (lm, sm), r in results.items():
        for sample in ("is", "oos"):
            for d in ("long", "short", "comb"):
                x = r[sample][d]
                flat.append({
                    "long_max": lm, "short_max": sm, "sample": sample, "dir": d,
                    "n": x["n"], "ret": x["ret"], "pf": x["pf"],
                    "wr": x["wr"], "mdd": x["mdd"], "sharpe": x["sharpe"],
                })
    df = pd.DataFrame(flat)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV: {out.resolve()}")


if __name__ == "__main__":
    main()
