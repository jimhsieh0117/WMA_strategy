"""r_ladder step / offset sweep（IS + OOS）。

問題：當前 step=0.5 / offset=0.2 對 stage 3 階梯太密；trade 從 peak 回吐
0.2R 就出場，可能在波動大的趨勢中段被洗掉。

固定：first_trigger=2.5
掃描合法組合（offset < step）：
    (step=0.5, offset=0.2)  baseline
    (step=1.0, offset=0.2)
    (step=1.0, offset=0.5)
    (step=1.5, offset=0.2)
    (step=1.5, offset=0.5)
    (step=1.5, offset=0.8)

執行：
    python -m scripts.sweep_r_ladder
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
from src.utils.config import TrailingConfig, load_config  # noqa: E402


COMBOS = [
    (0.5, 0.2),  # baseline
    (1.0, 0.2),
    (1.0, 0.5),
    (1.5, 0.2),
    (1.5, 0.5),
    (1.5, 0.8),
]


def s3_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "pnl": 0.0, "avg_pnl": 0.0,
                "avg_peak_r": 0.0, "avg_hold": 0.0}
    s3 = [t for t in trades if int(t.final_stage) == 3]
    if not s3:
        return {"n": 0, "pnl": 0.0, "avg_pnl": 0.0,
                "avg_peak_r": 0.0, "avg_hold": 0.0}
    pnls = np.array([float(t.net_pnl) for t in s3])
    peaks = np.array([float(t.peak_progress_r) for t in s3])
    holds = np.array([
        (t.exit_timestamp - t.entry_timestamp).total_seconds() / 900
        for t in s3
    ])
    return {
        "n": len(s3),
        "pnl": round(float(pnls.sum()), 2),
        "avg_pnl": round(float(pnls.mean()), 2),
        "avg_peak_r": round(float(peaks.mean()), 3),
        "avg_hold": round(float(holds.mean()), 1),
    }


def run_combo(base_cfg, step: float, offset: float, sample: str) -> dict:
    base_tr = base_cfg.trailing
    cfg = dataclasses.replace(
        base_cfg,
        trailing=TrailingConfig(
            swing_lookback=base_tr.swing_lookback,
            stage1_slippage_buffer=base_tr.stage1_slippage_buffer,
            stage2_normal_trigger_r=base_tr.stage2_normal_trigger_r,
            stage2_abnormal_trigger_r=base_tr.stage2_abnormal_trigger_r,
            stage2_buffer_r=base_tr.stage2_buffer_r,
            stage2_pct_trigger=base_tr.stage2_pct_trigger,
            stage3_normal_trigger_r=base_tr.stage3_normal_trigger_r,
            stage3_abnormal_trigger_r=base_tr.stage3_abnormal_trigger_r,
            bollinger_period=base_tr.bollinger_period,
            bollinger_num_std=base_tr.bollinger_num_std,
            stage3_mode=base_tr.stage3_mode,
            r_ladder_normal_first_trigger=base_tr.r_ladder_normal_first_trigger,
            r_ladder_normal_step=step,
            r_ladder_abnormal_first_trigger=base_tr.r_ladder_abnormal_first_trigger,
            r_ladder_abnormal_step=base_tr.r_ladder_abnormal_step,
            r_ladder_trigger_offset=offset,
            r_ladder_abnormal_trigger_offset=base_tr.r_ladder_abnormal_trigger_offset,
            early_exit_enabled=base_tr.early_exit_enabled,
            early_exit_observation_bars=base_tr.early_exit_observation_bars,
            early_exit_metric=base_tr.early_exit_metric,
            early_exit_min_peak_r=base_tr.early_exit_min_peak_r,
            early_exit_min_peak_pct=base_tr.early_exit_min_peak_pct,
            early_exit_min_close_r=base_tr.early_exit_min_close_r,
            stage1_time_cut_enabled=base_tr.stage1_time_cut_enabled,
            stage1_time_cut_bars=base_tr.stage1_time_cut_bars,
            stage1_time_cut_peak_r_max=base_tr.stage1_time_cut_peak_r_max,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    C = build_merged_result(f"comb_{sample}", [L, S])
    mC = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "step": step, "offset": offset, "sample": sample,
        "n": len(L.trades) + len(S.trades),
        "ret": mC.total_return_pct, "pf": mC.profit_factor,
        "wr": mC.win_rate_pct, "mdd": mC.max_drawdown_pct,
        "sharpe": mC.sharpe_ratio,
        "long_s3": s3_stats(L.trades),
        "short_s3": s3_stats(S.trades),
    }


def print_table(title: str, rows: list[dict]) -> None:
    print()
    print("=" * 120)
    print(title)
    print("=" * 120)
    print(f"{'step':>4}  {'offset':>6}  "
          f"{'n':>4}  {'ret%':>7}  {'PF':>5}  {'WR%':>5}  {'MDD%':>5}  {'Sharpe':>7}  "
          f"{'L_s3_n':>6}  {'L_s3_pnl':>9}  {'L_s3_avg':>9}  {'L_peakR':>8}  "
          f"{'S_s3_n':>6}  {'S_s3_pnl':>9}")
    for r in rows:
        l = r["long_s3"]
        s = r["short_s3"]
        baseline_mark = " ★" if (r["step"] == 0.5 and r["offset"] == 0.2) else "  "
        print(f"{r['step']:>4.1f}{baseline_mark}{r['offset']:>4.1f}  "
              f"{r['n']:>4}  {r['ret']:>+7.2f}  {r['pf']:>5.2f}  "
              f"{r['wr']:>4.1f}%  {r['mdd']:>4.1f}%  {r['sharpe']:>+7.2f}  "
              f"{l['n']:>6}  {l['pnl']:>+9.2f}  {l['avg_pnl']:>+9.2f}  "
              f"{l['avg_peak_r']:>7.2f}  "
              f"{s['n']:>6}  {s['pnl']:>+9.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-csv", default="results/r_ladder_sweep/summary.csv")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"Fixed: first_trigger={base_cfg.trailing.r_ladder_normal_first_trigger}, "
          f"stage1_time_cut={base_cfg.trailing.stage1_time_cut_enabled}")
    print(f"Sweep: {len(COMBOS)} combos x IS+OOS\n")

    is_rows, oos_rows = [], []
    for step, offset in COMBOS:
        print(f">>> step={step}, offset={offset}, sample=is")
        is_rows.append(run_combo(base_cfg, step, offset, "is"))
        print(f">>> step={step}, offset={offset}, sample=oos")
        oos_rows.append(run_combo(base_cfg, step, offset, "oos"))

    print_table("Combined IS (★ = current baseline)", is_rows)
    print_table("Combined OOS (★ = current baseline)", oos_rows)

    # IS+OOS robustness 排行
    print()
    print("=" * 100)
    print("Robustness 排行（IS ret>0 且 OOS PF >= IS PF 的 80%，按 OOS PF 排）")
    print("=" * 100)
    paired = []
    for ir, oor in zip(is_rows, oos_rows):
        if ir["ret"] <= 0:
            continue
        paired.append({
            "step": ir["step"], "offset": ir["offset"],
            "is_ret": ir["ret"], "is_pf": ir["pf"], "is_mdd": ir["mdd"],
            "oos_ret": oor["ret"], "oos_pf": oor["pf"], "oos_mdd": oor["mdd"],
            "drop_ratio": oor["pf"] / ir["pf"] if ir["pf"] > 0 else 0,
        })
    paired.sort(key=lambda x: x["oos_pf"], reverse=True)
    print(f"{'step':>5}  {'offset':>6}  "
          f"{'IS_ret%':>8}  {'IS_PF':>6}  {'IS_MDD%':>7}  "
          f"{'OOS_ret%':>9}  {'OOS_PF':>7}  {'OOS_MDD%':>8}  "
          f"{'PF_drop':>8}")
    for r in paired:
        print(f"{r['step']:>5.1f}  {r['offset']:>6.1f}  "
              f"{r['is_ret']:>+8.2f}  {r['is_pf']:>6.2f}  {r['is_mdd']:>6.1f}%  "
              f"{r['oos_ret']:>+9.2f}  {r['oos_pf']:>7.2f}  {r['oos_mdd']:>7.1f}%  "
              f"{r['drop_ratio']:>7.2f}x")

    flat = []
    for r in is_rows + oos_rows:
        flat.append({
            "step": r["step"], "offset": r["offset"], "sample": r["sample"],
            "n": r["n"], "ret": r["ret"], "pf": r["pf"],
            "wr": r["wr"], "mdd": r["mdd"], "sharpe": r["sharpe"],
            "long_s3_n": r["long_s3"]["n"],
            "long_s3_pnl": r["long_s3"]["pnl"],
            "long_s3_avg_peak_r": r["long_s3"]["avg_peak_r"],
            "short_s3_n": r["short_s3"]["n"],
            "short_s3_pnl": r["short_s3"]["pnl"],
        })
    df = pd.DataFrame(flat)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nCSV: {out.resolve()}")


if __name__ == "__main__":
    main()
