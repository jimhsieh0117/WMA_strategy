"""執行多頭策略回測。

用法：
    python scripts/run_long.py [--config configs/default.yaml] [--sample is|oos]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import report_result, run_single_strategy  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="WMA Long Trend Strategy backtest")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    label = f"long_{cfg.timeframe}_{args.sample}"
    result = run_single_strategy(cfg, direction="long", sample=args.sample)
    report_result(result, cfg, label=label)


if __name__ == "__main__":
    main()
