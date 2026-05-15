"""Stage 3 mode 比較：r_ladder vs bollinger（IS）。

固定其他全部設定，只切換 ``trailing.stage3_mode``。對每個 mode：
- 跑 long + short combined backtest
- 算 trade-level 統計（peak_R / final_R / giveback / win-loss ratio）
- 列表格對照

用法：
    .venv/bin/python -m scripts.compare_stage3_modes [--sample is|oos]
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import TrailingConfig, load_config  # noqa: E402
from src.utils.types import Direction  # noqa: E402


def run_mode(base_cfg, mode: str, sample: str) -> dict:
    """跑一個 stage3_mode 的 long + short combined backtest。"""
    base_tr = base_cfg.trailing
    new_tr = dataclasses.replace(base_tr, stage3_mode=mode)
    cfg = dataclasses.replace(
        base_cfg, trailing=new_tr,
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    C = build_merged_result(f"comb_{sample}_{mode}", [L, S])
    mC = compute_metrics(C, timeframe=cfg.timeframe)

    # trade-level give-back metrics
    trades = list(L.trades) + list(S.trades)
    peaks, finals, givebacks, holds_win = [], [], [], []
    wins = losses = 0
    win_pnl_sum = 0.0
    loss_pnl_sum = 0.0
    for t in trades:
        if not t.stop_history:
            continue
        initial_stop = float(t.stop_history[0][1])
        R_price = abs(float(t.entry_price) - initial_stop)
        if R_price <= 0:
            continue
        risk_amount = float(t.quantity) * R_price
        final_R = float(t.net_pnl) / risk_amount if risk_amount > 0 else 0.0
        peak_R = float(t.peak_progress_r)
        giveback = peak_R - final_R
        peaks.append(peak_R)
        finals.append(final_R)
        givebacks.append(giveback)
        if t.net_pnl > 0:
            wins += 1
            win_pnl_sum += float(t.net_pnl)
            holds_win.append(
                (t.exit_timestamp - t.entry_timestamp).total_seconds() / 900
            )
        else:
            losses += 1
            loss_pnl_sum += float(t.net_pnl)

    peaks_arr = np.array(peaks) if peaks else np.array([0.0])
    finals_arr = np.array(finals) if finals else np.array([0.0])
    givebacks_arr = np.array(givebacks) if givebacks else np.array([0.0])

    return {
        "mode": mode,
        "n_trades": len(trades),
        "long_final_equity": L.equity_curve.iloc[-1],
        "short_final_equity": S.equity_curve.iloc[-1],
        "combined_return_pct": mC.total_return_pct,
        "combined_pf": mC.profit_factor,
        "combined_wr": mC.win_rate_pct,
        "combined_mdd": mC.max_drawdown_pct,
        "combined_sharpe": mC.sharpe_ratio,
        "combined_expectancy": mC.expectancy,
        "combined_avg_win": mC.avg_win,
        "combined_avg_loss": mC.avg_loss,
        "combined_winloss_ratio": mC.avg_win / abs(mC.avg_loss) if mC.avg_loss != 0 else 0.0,
        "combined_avg_hold": mC.avg_holding_bars,
        "peak_R_mean": float(peaks_arr.mean()),
        "peak_R_med": float(np.median(peaks_arr)),
        "final_R_mean": float(finals_arr.mean()),
        "final_R_med": float(np.median(finals_arr)),
        "giveback_R_mean": float(givebacks_arr.mean()),
        "giveback_R_med": float(np.median(givebacks_arr)),
        "n_wins": wins,
        "n_losses": losses,
    }


def print_table(rows: list[dict]) -> None:
    if not rows:
        return
    keys_order = [
        ("n_trades",                  "Trades",          "{:>5d}"),
        ("combined_return_pct",       "Return %",        "{:>+8.2f}"),
        ("combined_pf",               "PF",              "{:>6.2f}"),
        ("combined_expectancy",       "期望值/筆",        "{:>+8.4f}"),
        ("combined_wr",               "WR %",            "{:>6.2f}"),
        ("combined_avg_win",          "Avg Win",         "{:>8.4f}"),
        ("combined_avg_loss",         "Avg Loss",        "{:>8.4f}"),
        ("combined_winloss_ratio",    "W/L ratio",       "{:>8.2f}"),
        ("combined_mdd",              "MDD %",           "{:>6.2f}"),
        ("combined_sharpe",           "Sharpe",          "{:>+7.2f}"),
        ("combined_avg_hold",         "Avg Hold",        "{:>8.2f}"),
        ("long_final_equity",         "Long final",      "{:>10.2f}"),
        ("short_final_equity",        "Short final",     "{:>10.2f}"),
        ("peak_R_mean",               "peak_R mean",     "{:>+8.2f}"),
        ("peak_R_med",                "peak_R med",      "{:>+8.2f}"),
        ("final_R_mean",              "final_R mean",    "{:>+8.2f}"),
        ("final_R_med",               "final_R med",     "{:>+8.2f}"),
        ("giveback_R_mean",           "giveback mean",   "{:>+8.2f}"),
        ("giveback_R_med",            "giveback med",    "{:>+8.2f}"),
    ]
    print()
    print("=" * 64)
    print(f"  {'metric':<18} | {'r_ladder':>12} | {'bollinger':>12} | {'diff':>12}")
    print("=" * 64)
    a, b = rows[0], rows[1]  # assume [r_ladder, bollinger] order
    for key, label, fmt in keys_order:
        av, bv = a[key], b[key]
        diff = bv - av
        try:
            row = f"  {label:<18} | {fmt.format(av):>12} | {fmt.format(bv):>12} | {fmt.format(diff):>12}"
        except (ValueError, TypeError):
            row = f"  {label:<18} | {av!r:>12} | {bv!r:>12} | {diff!r:>12}"
        print(row)
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Symbol: {cfg.symbol}  tf: {cfg.timeframe}  sample: {args.sample}")
    print(f"WMA long={cfg.wma_long_fast}/{cfg.wma_long_slow} "
          f"short={cfg.wma_short_fast}/{cfg.wma_short_slow}")
    print(f"stage2_enabled={cfg.trailing.stage2_enabled}  "
          f"retry long/short={cfg.entry_retry.long_max_attempts}/"
          f"{cfg.entry_retry.short_max_attempts}")
    print(f"r_ladder: first={cfg.trailing.r_ladder_normal_first_trigger}, "
          f"step={cfg.trailing.r_ladder_normal_step}, "
          f"offset={cfg.trailing.r_ladder_trigger_offset}")
    print(f"bollinger: period={cfg.trailing.bollinger_period}, "
          f"num_std={cfg.trailing.bollinger_num_std}")

    rows = []
    for mode in ("r_ladder", "bollinger"):
        print(f"\n>>> running stage3_mode={mode} ...")
        rows.append(run_mode(cfg, mode, args.sample))

    print_table(rows)


if __name__ == "__main__":
    main()
