"""輸出 metrics / trades / summary 至檔案。

JSON 序列化處理 Timestamp / Timedelta / dataclass 自動轉換。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.broker.types import Trade
from src.metrics.calculator import MetricsReport

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #

def _json_default(obj: Any) -> Any:
    """擴充 ``json`` 的型別支援：Timestamp / Timedelta / dataclass / Path。"""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --------------------------------------------------------------------------- #
# Public exporters
# --------------------------------------------------------------------------- #

def export_metrics_json(metrics: MetricsReport, output_path: str | Path) -> Path:
    """將 MetricsReport 寫成 JSON。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = asdict(metrics)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default, ensure_ascii=False)

    logger.info("metrics.json saved: %s", out)
    return out


def export_trades_csv(trades: list[Trade], output_path: str | Path) -> Path:
    """將 trade 列表寫成 CSV（空 list 也會寫出僅含 header 的檔案）。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not trades:
        # 空交易仍寫 header，方便外部腳本一致處理
        df = pd.DataFrame(
            columns=[
                "direction", "entry_ts", "exit_ts", "entry_price", "exit_price",
                "quantity", "gross_pnl", "net_pnl", "return_pct", "exit_reason",
                "entry_fee", "exit_fee", "holding_minutes",
            ]
        )
    else:
        df = pd.DataFrame(
            [
                {
                    "direction": t.direction.value,
                    "entry_ts": t.entry_timestamp,
                    "exit_ts": t.exit_timestamp,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct * 100.0,
                    "exit_reason": t.exit_reason,
                    "entry_fee": t.entry_fee,
                    "exit_fee": t.exit_fee,
                    "holding_minutes": t.holding_duration.total_seconds() / 60.0,
                }
                for t in trades
            ]
        )

    df.to_csv(out, index=False)
    logger.info("trades.csv saved: %s (%d rows)", out, len(df))
    return out


def export_summary_text(
    summary_lines: list[str], output_path: str | Path
) -> Path:
    """將純文字摘要寫到檔案（caller 提供已經格式化的行）。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    logger.info("summary.txt saved: %s", out)
    return out


def export_config_snapshot(
    snapshot: dict[str, Any], output_path: str | Path
) -> Path:
    """將回測時的 config_snapshot 寫成 JSON 以便重現。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=_json_default, ensure_ascii=False)
    logger.info("config_snapshot.json saved: %s", out)
    return out
