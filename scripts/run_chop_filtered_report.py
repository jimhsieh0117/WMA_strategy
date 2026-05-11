"""套用盤整 filter 後的完整績效報告（IS + OOS）。

對 long / short 各帳戶分別跑 baseline 回測，再用 chop filter
(BBW>=40 & ATR>=40 & ADX>=20) 過濾交易，重建 equity curve 並算
完整 metrics（PF / Sharpe / MDD / Calmar / Win rate ...）。

⚠️ 注意：本腳本為 post-hoc 過濾，position sizing 沿用原始 trades 的
   net_pnl（=以原始 equity 軌跡為基準計算）。真實 in-loop 過濾的
   絕對 USDT 會略有差異（每筆 size 重新計算），但 PF / win rate /
   trade 級別指標皆正確。要看真實曲線需在 engine 整合 filter，留待
   後續實作。

用法：
    python scripts/run_chop_filtered_report.py
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup,
)
from src.backtest.types import BacktestResult  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


def _apply_combo(fv, bbw_thr, atr_thr, adx_thr) -> np.ndarray:
    return (
        (fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
        (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
        (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"])
    )


def _build_filtered_result(
    orig: BacktestResult, trades_kept, period_start, period_end,
) -> BacktestResult:
    """從 kept trades 重建 BacktestResult（trade-indexed equity curve）。"""
    initial = orig.initial_capital
    if not trades_kept:
        eq = pd.Series([initial, initial],
                       index=[pd.Timestamp(period_start), pd.Timestamp(period_end)])
        final = initial
    else:
        sorted_trades = sorted(trades_kept, key=lambda t: t.exit_timestamp)
        ts = [pd.Timestamp(t.exit_timestamp) for t in sorted_trades]
        pnls = np.cumsum([t.net_pnl for t in sorted_trades])
        eq_vals = initial + pnls
        eq = pd.Series(eq_vals, index=ts)
        # 同 timestamp 多筆 → 取最後一筆累積值（去重）
        eq = eq[~eq.index.duplicated(keep="last")]
        # 前置起點 + 後置結束點，讓 duration 反映完整區間
        if pd.Timestamp(period_start) < eq.index[0]:
            eq = pd.concat([pd.Series([initial],
                                      index=[pd.Timestamp(period_start)]), eq])
        if eq.index[-1] < pd.Timestamp(period_end):
            eq = pd.concat([eq, pd.Series([eq.iloc[-1]],
                                          index=[pd.Timestamp(period_end)])])
        final = float(eq.iloc[-1])
    return BacktestResult(
        account_name=orig.account_name + "_filtered",
        initial_capital=initial,
        final_equity=final,
        trades=list(trades_kept),
        equity_curve=eq,
        bars_processed=orig.bars_processed,
        signals_emitted=orig.signals_emitted,
        signals_filled=len(trades_kept),
        signals_unfilled=orig.signals_unfilled,
        signals_skipped_pending=orig.signals_skipped_pending,
        config_snapshot=orig.config_snapshot,
    )


def _format_row(label, m) -> str:
    return (f"  {label:24s}  ret={m.total_return_pct:+7.2f}%  "
            f"PF={m.profit_factor:5.2f}  Sharpe={m.sharpe_ratio:+6.2f}  "
            f"MDD={m.max_drawdown_pct:5.2f}%  Calmar={m.calmar_ratio:+6.2f}  "
            f"win={m.win_rate_pct:5.2f}%  n={m.total_trades:4d}")


def _run_one_sample(
    cfg, start, end, args, tf_delta, label,
) -> dict:
    print(f"\n>>> [{label}]  {start} ~ {end}", flush=True)
    cfg_p = dataclasses.replace(cfg, in_sample=PeriodSpec(
        start=pd.Timestamp(start), end=pd.Timestamp(end)))

    bt_L = run_single_strategy(cfg_p, direction="long", sample="is")
    bt_S = run_single_strategy(cfg_p, direction="short", sample="is")

    df1m = load_ohlcv(cfg_p.source_parquet,
                      start=pd.Timestamp(start), end=pd.Timestamp(end))
    cache = _compute_filter_cache(resample(df1m, args.timeframe))

    out = {}
    for direction, bt in [("long", bt_L), ("short", bt_S)]:
        fv = {col: _filter_lookup(bt.trades, cache[col], tf_delta)
              for col in ["adx", "atr_rank", "bbw_rank"]}
        keep = _apply_combo(fv, args.bbw, args.atr, args.adx)
        kept = [t for t, k in zip(bt.trades, keep) if k]

        m_base = compute_metrics(bt, timeframe=args.timeframe)
        bt_f = _build_filtered_result(bt, kept, start, end)
        m_filt = compute_metrics(bt_f, timeframe=args.timeframe)

        out[direction] = {
            "baseline": (bt, m_base),
            "filtered": (bt_f, m_filt),
        }
        print(_format_row(f"{direction:5s} baseline", m_base))
        print(_format_row(f"{direction:5s} filtered", m_filt))

    # 合併：把 long + short 兩條 equity 對齊相加（baseline / filtered 各一條）
    out["combined"] = {}
    for kind in ("baseline", "filtered"):
        eq_l = out["long"][kind][0].equity_curve
        eq_s = out["short"][kind][0].equity_curve
        eq_l = eq_l[~eq_l.index.duplicated(keep="last")]
        eq_s = eq_s[~eq_s.index.duplicated(keep="last")]
        all_ts = eq_l.index.union(eq_s.index)
        eq_l_f = eq_l.reindex(all_ts).ffill().fillna(out["long"][kind][0].initial_capital)
        eq_s_f = eq_s.reindex(all_ts).ffill().fillna(out["short"][kind][0].initial_capital)
        eq_c = eq_l_f + eq_s_f
        combined_initial = (out["long"][kind][0].initial_capital +
                            out["short"][kind][0].initial_capital)
        combined_trades = out["long"][kind][0].trades + out["short"][kind][0].trades
        bt_c = BacktestResult(
            account_name=f"combined_{kind}",
            initial_capital=combined_initial,
            final_equity=float(eq_c.iloc[-1]),
            trades=combined_trades,
            equity_curve=eq_c,
            bars_processed=0,
            signals_emitted=0, signals_filled=len(combined_trades),
            signals_unfilled=0, signals_skipped_pending=0,
        )
        m_c = compute_metrics(bt_c, timeframe=args.timeframe)
        out["combined"][kind] = (bt_c, m_c)
        print(_format_row(f"combined {kind:8s}", m_c))

    return out


def _plot_equity(is_out, oos_out, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sample_out, title in [
        (axes[0], is_out,  "IS (2023-2024)"),
        (axes[1], oos_out, "OOS (2025+)"),
    ]:
        eq_b = sample_out["combined"]["baseline"][0].equity_curve
        eq_f = sample_out["combined"]["filtered"][0].equity_curve
        ax.plot(eq_b.index, eq_b.values, label="baseline", color="gray", alpha=0.7)
        ax.plot(eq_f.index, eq_f.values, label="chop-filtered", color="steelblue")
        ax.axhline(eq_b.iloc[0], color="black", linestyle=":", linewidth=0.8)
        ax.set_title(f"{title}：combined equity (long+short)")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--wma-fast", type=int, default=4)
    parser.add_argument("--wma-slow", type=int, default=6)
    parser.add_argument("--bbw", type=float, default=40.0)
    parser.add_argument("--atr", type=float, default=40.0)
    parser.add_argument("--adx", type=float, default=20.0)
    parser.add_argument("--is-start", default="2023-01-01")
    parser.add_argument("--is-end",   default="2024-12-31")
    parser.add_argument("--oos-start", default="2025-01-01")
    parser.add_argument("--oos-end",   default="2026-03-14")
    parser.add_argument("--out-dir", default="results/chop_filter_report")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_m = dataclasses.replace(
        cfg, show_progress=False, log_level="WARNING",
        timeframe=args.timeframe,
        wma_fast=args.wma_fast, wma_slow=args.wma_slow,
    )
    tf_delta = pd.Timedelta(args.timeframe)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== filter: BBW>={args.bbw} & ATR>={args.atr} & ADX>={args.adx} ===")
    is_out = _run_one_sample(cfg_m, args.is_start, args.is_end, args,
                             tf_delta, "IS")
    oos_out = _run_one_sample(cfg_m, args.oos_start, args.oos_end, args,
                              tf_delta, "OOS")

    _plot_equity(is_out, oos_out, out_dir / "equity_compare.png")
    print(f"\n💾 equity plot → {out_dir / 'equity_compare.png'}")


if __name__ == "__main__":
    main()
