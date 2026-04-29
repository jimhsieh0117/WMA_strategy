"""同時跑多 + 空兩支策略，分別 report，並合併成總體權益曲線 report。

用法：
    python scripts/run_combined.py [--config configs/default.yaml] [--sample is|oos]

合併語意（ARCHITECTURE.md §9.7 選項 B）：
- 多空各持一個 500 USDT 帳戶，**獨立**累積 PnL
- 合併權益曲線 = 兩條 equity 直接相加（時間軸對齊；無中途再平衡）
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import report_result, run_single_strategy  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WMA Long + Short combined backtest with merged equity"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    suffix = f"{cfg.timeframe}_{args.sample}"

    # 1) 跑兩邊
    long_res = run_single_strategy(cfg, direction="long", sample=args.sample)
    short_res = run_single_strategy(cfg, direction="short", sample=args.sample)

    # 2) 分別 report
    report_result(long_res, cfg, label=f"long_{suffix}")
    report_result(short_res, cfg, label=f"short_{suffix}")

    # 3) 合併 + report；equity.png 中順便疊上 long / short 兩條子曲線供對照
    combined = build_merged_result(f"combined_{suffix}", [long_res, short_res])
    combined_metrics = report_result(
        combined, cfg, label=f"combined_{suffix}",
        extra_curves={
            "long":  long_res.equity_curve,
            "short": short_res.equity_curve,
        },
    )

    print()
    print("=" * 60)
    print("  ✅ Combined run finished")
    print("=" * 60)
    print(f"  long  final equity:    {long_res.final_equity:>10.2f} USDT")
    print(f"  short final equity:    {short_res.final_equity:>10.2f} USDT")
    print(f"  combined final equity: {combined.final_equity:>10.2f} USDT")
    print(f"  combined return:       {combined_metrics.total_return_pct:>+9.2f} %")
    print(f"  combined Sharpe:       {combined_metrics.sharpe_ratio:>10.2f}")


if __name__ == "__main__":
    main()
