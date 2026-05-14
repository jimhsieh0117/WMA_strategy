"""Dump retry∈{1,2,3} × IS+OOS 詳細 trade 資料，供 subagent 深度分析。

每組 (retry, sample) 寫入 results/retry_stage_compare/r{N}_{sample}/：
- long_trades.csv
- short_trades.csv
- summary.json  （含各 stage 的 n / pf / wr / avg_R / avg_hold_bars / total_pnl）

執行：
    python -m scripts.dump_retry_stage_compare
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import EntryRetryConfig, load_config  # noqa: E402


RETRY_VALUES = [1, 2, 3]
SAMPLES = ["is", "oos"]
OUT_ROOT = Path("results/retry_stage_compare")


def trade_to_row(t, timeframe_bars: int = 15) -> dict:
    return {
        "direction": "long" if t.direction.name == "LONG" else "short",
        "entry_timestamp": t.entry_timestamp,
        "exit_timestamp": t.exit_timestamp,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "gross_pnl": t.gross_pnl,
        "net_pnl": t.net_pnl,
        "return_pct": t.return_pct,
        "exit_reason": t.exit_reason,
        "final_stage": int(t.final_stage),
        "peak_progress_r": float(t.peak_progress_r),
        "hold_bars": int((t.exit_timestamp - t.entry_timestamp).total_seconds() // (timeframe_bars * 60)),
        "hour_utc": int(t.entry_timestamp.hour),
        "dow": int(t.entry_timestamp.dayofweek),
    }


def stage_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "stages": {}}
    rows = [trade_to_row(t) for t in trades]
    df = pd.DataFrame(rows)
    out: dict = {"n": len(rows), "stages": {}}
    for s in (1, 2, 3):
        sub = df[df.final_stage == s]
        if len(sub) == 0:
            out["stages"][f"s{s}"] = {"n": 0}
            continue
        wins = sub[sub.net_pnl > 0]
        gross_win = float(wins.net_pnl.sum())
        gross_loss = float(sub[sub.net_pnl <= 0].net_pnl.abs().sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        out["stages"][f"s{s}"] = {
            "n": int(len(sub)),
            "wins": int(len(wins)),
            "wr_pct": round(100 * len(wins) / len(sub), 2),
            "pf": round(pf, 3) if np.isfinite(pf) else None,
            "total_pnl": round(float(sub.net_pnl.sum()), 2),
            "avg_pnl": round(float(sub.net_pnl.mean()), 2),
            "avg_peak_r": round(float(sub.peak_progress_r.mean()), 3),
            "avg_hold_bars": round(float(sub.hold_bars.mean()), 1),
            "med_hold_bars": int(sub.hold_bars.median()),
        }
    return out


def run_one(base_cfg, retry: int, sample: str) -> dict:
    cfg = dataclasses.replace(
        base_cfg,
        entry_retry=EntryRetryConfig(long_max_attempts=retry, short_max_attempts=retry),
        show_progress=False, log_level="WARNING",
    )
    out_dir = OUT_ROOT / f"r{retry}_{sample}"
    out_dir.mkdir(parents=True, exist_ok=True)

    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    C = build_merged_result(f"comb_r{retry}_{sample}", [L, S])

    # 存 trades CSV
    if L.trades:
        pd.DataFrame([trade_to_row(t) for t in L.trades]).to_csv(
            out_dir / "long_trades.csv", index=False)
    if S.trades:
        pd.DataFrame([trade_to_row(t) for t in S.trades]).to_csv(
            out_dir / "short_trades.csv", index=False)

    mL = compute_metrics(L, timeframe=cfg.timeframe)
    mS = compute_metrics(S, timeframe=cfg.timeframe)
    mC = compute_metrics(C, timeframe=cfg.timeframe)

    summary = {
        "retry": retry, "sample": sample,
        "config_snapshot": {
            "chop_vol": cfg.chop_filter.bbw_rank_min,
            "adx_min": cfg.chop_filter.adx_min,
            "signal_th": cfg.signal_filter.threshold,
            "structure_mode": cfg.structure_filter.mode,
            "structure_enabled": cfg.structure_filter.enabled,
        },
        "long": {
            **stage_stats(L.trades),
            "ret_pct": mL.total_return_pct, "pf": mL.profit_factor,
            "wr_pct": mL.win_rate_pct, "mdd_pct": mL.max_drawdown_pct,
            "sharpe": mL.sharpe_ratio,
        },
        "short": {
            **stage_stats(S.trades),
            "ret_pct": mS.total_return_pct, "pf": mS.profit_factor,
            "wr_pct": mS.win_rate_pct, "mdd_pct": mS.max_drawdown_pct,
            "sharpe": mS.sharpe_ratio,
        },
        "combined": {
            "n": len(L.trades) + len(S.trades),
            "ret_pct": mC.total_return_pct, "pf": mC.profit_factor,
            "wr_pct": mC.win_rate_pct, "mdd_pct": mC.max_drawdown_pct,
            "sharpe": mC.sharpe_ratio,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def main() -> None:
    base_cfg = load_config("configs/default.yaml")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"Fixed: chop vol={base_cfg.chop_filter.bbw_rank_min}/ADX={base_cfg.chop_filter.adx_min}, "
          f"signal th={base_cfg.signal_filter.threshold}, "
          f"structure={base_cfg.structure_filter.mode}")
    print(f"Sweep: retry ∈ {RETRY_VALUES}, sample ∈ {SAMPLES}\n")

    all_summaries = []
    for retry in RETRY_VALUES:
        for sample in SAMPLES:
            print(f">>> retry={retry}, sample={sample}")
            s = run_one(base_cfg, retry, sample)
            all_summaries.append(s)

    # 印整體對照表
    print()
    print("=" * 100)
    print("Combined retry x sample summary")
    print("=" * 100)
    print(f"{'retry':>5}  {'sample':>6}  {'n':>4}  {'ret%':>7}  {'PF':>5}  "
          f"{'WR%':>5}  {'MDD%':>5}  {'Sharpe':>7}  "
          f"{'L_s3_n':>6}  {'L_s3_pnl':>9}  {'S_s3_n':>6}  {'S_s3_pnl':>9}")
    for s in all_summaries:
        c = s["combined"]
        l_s3 = s["long"]["stages"].get("s3", {})
        s_s3 = s["short"]["stages"].get("s3", {})
        print(f"{s['retry']:>5}  {s['sample']:>6}  {c['n']:>4}  "
              f"{c['ret_pct']:>+7.2f}  {c['pf']:>5.2f}  "
              f"{c['wr_pct']:>4.1f}%  {c['mdd_pct']:>4.1f}%  "
              f"{c['sharpe']:>+7.2f}  "
              f"{l_s3.get('n', 0):>6}  {l_s3.get('total_pnl', 0):>+9.2f}  "
              f"{s_s3.get('n', 0):>6}  {s_s3.get('total_pnl', 0):>+9.2f}")

    # 全部 summary 寫到 root
    (OUT_ROOT / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False, default=str))
    print(f"\nOutput root: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
