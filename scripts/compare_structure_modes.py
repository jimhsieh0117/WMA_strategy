"""3 模式 structure_filter 回測比較。

跑同一期間（預設 2 年 IS），分別套：
- ``off``              : structure_filter.enabled = False
- ``exclude_counter``  : enabled=True, mode=exclude_counter
- ``aligned``          : enabled=True, mode=aligned

輸出 long / short / combined 三組 metrics 對照表。

執行：
    python -m scripts.compare_structure_modes
    python -m scripts.compare_structure_modes --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import PeriodSpec, StructureFilterConfig, load_config  # noqa: E402


MODES = [
    ("off", False, "aligned"),                     # disabled
    ("exclude_counter", True, "exclude_counter"),  # 只擋反向
    ("aligned", True, "aligned"),                  # 嚴格順勢
]


def run_one(base_cfg, label: str, enabled: bool, mode: str) -> dict:
    """跑一種 structure_filter 設定，回傳 long/short/combined metrics dict。"""
    cfg = dataclasses.replace(
        base_cfg,
        structure_filter=StructureFilterConfig(
            enabled=enabled, mode=mode,
            pivot_left=base_cfg.structure_filter.pivot_left,
            pivot_right=base_cfg.structure_filter.pivot_right,
        ),
        show_progress=False, log_level="WARNING",
    )
    long_r = run_single_strategy(cfg, direction="long", sample="is")
    short_r = run_single_strategy(cfg, direction="short", sample="is")
    combined = build_merged_result(
        f"combined_{label}", [long_r, short_r],
    )

    return {
        "label": label,
        "long": compute_metrics(long_r, timeframe=cfg.timeframe),
        "short": compute_metrics(short_r, timeframe=cfg.timeframe),
        "combined": compute_metrics(combined, timeframe=cfg.timeframe),
        "long_trades": len(long_r.trades),
        "short_trades": len(short_r.trades),
    }


def format_row(name: str, m, n_trades: int | None = None) -> str:
    n = n_trades if n_trades is not None else m.total_trades
    return (
        f"  {name:<10}"
        f"  n={n:>5}"
        f"  ret={m.total_return_pct:>+7.2f}%"
        f"  PF={m.profit_factor:>5.2f}"
        f"  WR={m.win_rate_pct:>5.1f}%"
        f"  MDD={m.max_drawdown_pct:>5.1f}%"
        f"  Sharpe={m.sharpe_ratio:>5.2f}"
        f"  exp={m.expectancy:>+6.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3 模式 structure_filter 回測比較。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument(
        "--out-csv", default="results/structure_filter_compare/metrics.csv",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    period = PeriodSpec(start=pd.Timestamp(args.start), end=pd.Timestamp(args.end))
    base_cfg = dataclasses.replace(base_cfg, in_sample=period)

    print(f"Period: {args.start} ~ {args.end}  symbol: {base_cfg.symbol}  "
          f"tf: {base_cfg.timeframe}")
    print(f"signal_filter mode: {base_cfg.signal_filter.mode}  "
          f"chop_filter enabled: {base_cfg.chop_filter.enabled}")
    print(f"Pivot: left={base_cfg.structure_filter.pivot_left}, "
          f"right={base_cfg.structure_filter.pivot_right}")
    print()

    results = []
    for label, enabled, mode in MODES:
        print(f">>> Running structure_filter={label} ...")
        r = run_one(base_cfg, label, enabled, mode)
        print(f"    long trades: {r['long_trades']}   "
              f"short trades: {r['short_trades']}")
        results.append(r)

    # ----- 對照表 -----
    print()
    print("=" * 96)
    print("3-mode comparison（IS）")
    print("=" * 96)
    for r in results:
        print(f"\n[{r['label']}]")
        print(format_row("long", r["long"], r["long_trades"]))
        print(format_row("short", r["short"], r["short_trades"]))
        print(format_row("combined", r["combined"]))

    # ----- diff 視角（vs off baseline）-----
    print()
    print("-" * 96)
    print("Combined vs off baseline:")
    print(f"{'mode':<18} {'Δret(pp)':>10} {'ΔPF':>7} {'ΔWR(pp)':>10}"
          f" {'ΔMDD(pp)':>10} {'Δtrades':>9}")
    base = results[0]
    base_c = base["combined"]
    base_n = base["long_trades"] + base["short_trades"]
    for r in results:
        c = r["combined"]
        n = r["long_trades"] + r["short_trades"]
        print(
            f"{r['label']:<18}"
            f" {c.total_return_pct - base_c.total_return_pct:>+10.2f}"
            f" {c.profit_factor - base_c.profit_factor:>+7.2f}"
            f" {c.win_rate_pct - base_c.win_rate_pct:>+10.2f}"
            f" {c.max_drawdown_pct - base_c.max_drawdown_pct:>+10.2f}"
            f" {n - base_n:>+9}"
        )

    # ----- CSV 輸出 -----
    rows = []
    for r in results:
        for which in ("long", "short", "combined"):
            m = r[which]
            n_trades = (
                r["long_trades"] if which == "long" else
                r["short_trades"] if which == "short" else
                r["long_trades"] + r["short_trades"]
            )
            rows.append({
                "mode": r["label"], "side": which,
                "n_trades": n_trades,
                "total_return_pct": round(m.total_return_pct, 3),
                "profit_factor": round(m.profit_factor, 3),
                "win_rate_pct": round(m.win_rate_pct, 2),
                "max_drawdown_pct": round(m.max_drawdown_pct, 2),
                "sharpe_ratio": round(m.sharpe_ratio, 3),
                "expectancy": round(m.expectancy, 4),
                "final_equity": round(m.final_equity, 2),
            })
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nCSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
