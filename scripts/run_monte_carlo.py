"""Monte Carlo 回測驗證 entry script。

跑一次完整回測（long + short 合併、或指定單向），把 trade list 餵給
`src.validation.monte_carlo` 做兩種模擬：
  - reshuffle：交易順序重排
  - bootstrap：交易層級有放回抽樣（可加 --block-size 做 block bootstrap）

輸出：
  - stdout：summary 表（baseline / mean / p05~p95）+ ruin / profit 機率
  - results/mc/<combo>/{reshuffle,bootstrap}_summary.csv
  - results/mc/<combo>/{reshuffle,bootstrap}_hist.png（4 個 metric 直方圖）
  - results/mc/<combo>/{reshuffle,bootstrap}_metrics.csv（每次模擬的原始指標）

用法：
    python scripts/run_monte_carlo.py                          # 5m W4-6 long+short
    python scripts/run_monte_carlo.py --timeframe 15m
    python scripts/run_monte_carlo.py --direction long --n 5000
    python scripts/run_monte_carlo.py --block-size 5           # block bootstrap
"""

from __future__ import annotations

import argparse
import dataclasses
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
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.validation.monte_carlo import (  # noqa: E402
    MCConfig, MCResult, bootstrap, reshuffle,
)


def _plot_histograms(result: MCResult, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("Final Equity",     result.final_equity,     result.baseline["final_equity"]),
        ("Max Drawdown (%)", result.max_drawdown_pct, result.baseline["max_drawdown_pct"]),
        ("Profit Factor",    result.profit_factor,    result.baseline["profit_factor"]),
        ("Sharpe / trade",   result.sharpe_trade,     result.baseline["sharpe_trade"]),
    ]
    for ax, (label, arr, base) in zip(axes.flat, panels):
        clean = arr[np.isfinite(arr)]
        if clean.size == 0:
            ax.set_title(f"{label}（無有效資料）")
            continue
        # PF 可能極端 → clip 顯示
        if label == "Profit Factor":
            clean = np.clip(clean, 0, np.percentile(clean, 99))
        # 常數陣列（如 reshuffle 下的 final_equity / PF / Sharpe）無法畫直方圖
        # 用相對 tolerance：permutation 對總和的影響來自浮點順序，差距 ~1e-10 級
        scale = max(abs(clean.mean()), 1.0)
        if (clean.max() - clean.min()) / scale < 1e-8:
            ax.axvline(float(clean[0]), color="red", linewidth=1.5)
            ax.set_title(f"{label} ≡ {clean[0]:.4f}（順序不變量）")
            ax.grid(alpha=0.3)
            continue
        ax.hist(clean, bins=60, color="steelblue", alpha=0.75)
        p05, p50, p95 = np.percentile(clean, [5, 50, 95])
        ax.axvline(base, color="red", linewidth=1.5, label=f"baseline={base:.3f}")
        ax.axvline(p50, color="black", linestyle="--", linewidth=1, label=f"p50={p50:.3f}")
        ax.axvline(p05, color="gray", linestyle=":", linewidth=1, label=f"p05={p05:.3f}")
        ax.axvline(p95, color="gray", linestyle=":", linewidth=1, label=f"p95={p95:.3f}")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _print_block(name: str, res: MCResult) -> None:
    print("\n" + "=" * 70)
    print(f"=== {name}  (n_sims={res.config.n_simulations}, n_trades={res.n_trades}) ===")
    print("=" * 70)
    print(res.summary().to_string(float_format="%.4f"))
    print(f"\n  P(ruin >= {res.config.ruin_threshold_pct:.0f}% DD)  = {res.prob_ruin*100:6.2f} %")
    print(f"  P(profitable: equity > initial) = {res.prob_profit*100:6.2f} %")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default=None,
                        help="覆寫 config 的 timeframe，例如 5m / 15m")
    parser.add_argument("--wma-fast", type=int, default=None)
    parser.add_argument("--wma-slow", type=int, default=None)
    parser.add_argument("--direction", choices=["long", "short", "both"], default="both")
    parser.add_argument("--n", type=int, default=10_000, help="模擬次數")
    parser.add_argument("--block-size", type=int, default=1,
                        help="bootstrap block size（>1 啟用 block bootstrap）")
    parser.add_argument("--ruin-pct", type=float, default=50.0,
                        help="判定破產的累計 DD 門檻（百分比）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--out-root", default="results/mc")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {
        "show_progress": False,
        "log_level": "WARNING",
        "in_sample": PeriodSpec(start=pd.Timestamp(args.start),
                                end=pd.Timestamp(args.end)),
    }
    if args.timeframe is not None:
        overrides["timeframe"] = args.timeframe
    if args.wma_fast is not None:
        overrides["wma_fast"] = args.wma_fast
    if args.wma_slow is not None:
        overrides["wma_slow"] = args.wma_slow
    cfg_m = dataclasses.replace(cfg, **overrides)

    # 跑回測，蒐集 trades
    print(f">>> backtest {cfg_m.symbol} {cfg_m.timeframe} W{cfg_m.wma_fast}-{cfg_m.wma_slow}  "
          f"{args.start} ~ {args.end}  direction={args.direction}", flush=True)
    trades = []
    if args.direction in ("long", "both"):
        L = run_single_strategy(cfg_m, direction="long", sample="is")
        trades.extend(L.trades)
        print(f"  long  trades={len(L.trades)}")
    if args.direction in ("short", "both"):
        S = run_single_strategy(cfg_m, direction="short", sample="is")
        trades.extend(S.trades)
        print(f"  short trades={len(S.trades)}")

    if not trades:
        print("✗ no trades produced — abort")
        return

    # 兩個獨立帳戶合併權益：把 initial_capital 設為 long+short 帳戶總和
    initial_capital = (cfg_m.initial_capital * (2 if args.direction == "both" else 1))
    mc_cfg = MCConfig(
        n_simulations=args.n,
        initial_capital=initial_capital,
        ruin_threshold_pct=args.ruin_pct,
        block_size=args.block_size,
        seed=args.seed,
    )

    combo_tag = f"{cfg_m.timeframe}_W{cfg_m.wma_fast}-{cfg_m.wma_slow}_{args.direction}"
    out_dir = Path(args.out_root) / combo_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Monte Carlo (n_sims={args.n}, initial={initial_capital:.0f}, "
          f"ruin>={args.ruin_pct:.0f}%DD, block_size={args.block_size})")

    res_reshuffle = reshuffle(trades, mc_cfg)
    _print_block("Method 1：Reshuffle（順序重排）", res_reshuffle)
    res_reshuffle.summary().to_csv(out_dir / "reshuffle_summary.csv",
                                   float_format="%.6f")
    _save_raw(res_reshuffle, out_dir / "reshuffle_metrics.csv")
    _plot_histograms(res_reshuffle, out_dir / "reshuffle_hist.png",
                     f"MC Reshuffle — {combo_tag}")

    res_boot = bootstrap(trades, mc_cfg)
    _print_block(
        f"Method 2：Bootstrap"
        f"{'（block='+str(args.block_size)+'）' if args.block_size > 1 else ''}",
        res_boot,
    )
    res_boot.summary().to_csv(out_dir / "bootstrap_summary.csv",
                              float_format="%.6f")
    _save_raw(res_boot, out_dir / "bootstrap_metrics.csv")
    _plot_histograms(res_boot, out_dir / "bootstrap_hist.png",
                     f"MC Bootstrap — {combo_tag}")

    print(f"\n💾 outputs → {out_dir}")


def _save_raw(res: MCResult, path: Path) -> None:
    pd.DataFrame({
        "final_equity":     res.final_equity,
        "total_return_pct": res.total_return_pct,
        "max_drawdown_pct": res.max_drawdown_pct,
        "profit_factor":    res.profit_factor,
        "sharpe_trade":     res.sharpe_trade,
        "ruined":           res.ruined,
        "profitable":       res.profitable,
    }).to_csv(path, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
