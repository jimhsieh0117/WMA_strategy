"""執行空頭策略回測。

用法：
    python scripts/run_short.py [--config configs/default.yaml] [--sample is|oos]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="WMA Short Trend Strategy backtest")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument(
        "--sample", choices=["is", "oos"], default="is",
        help="in-sample (default) or out-of-sample period",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    metrics, trades_df = run_single_strategy(cfg, direction="short", sample=args.sample)

    out = cfg.output_dir / f"{cfg.symbol}_short_{cfg.timeframe}_{args.sample}"
    out.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out / "trades.csv", index=False)
    print(f"\n💾 trades.csv saved → {out}")


if __name__ == "__main__":
    main()
