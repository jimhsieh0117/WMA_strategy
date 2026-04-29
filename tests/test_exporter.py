"""Reporting exporter 單元測試。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.broker.types import Trade
from src.metrics.calculator import MetricsReport
from src.reporting.exporter import (
    export_config_snapshot,
    export_metrics_json,
    export_summary_text,
    export_trades_csv,
)
from src.utils.types import Direction


def _sample_metrics() -> MetricsReport:
    return MetricsReport(
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-12-31"),
        duration_days=365.0,
        initial_capital=500.0,
        final_equity=600.0,
        total_return_pct=20.0,
        annualized_return_pct=20.0,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown_pct=10.0,
        calmar_ratio=2.0,
        total_trades=42,
        win_rate_pct=55.0,
        profit_factor=1.8,
        expectancy=2.4,
        avg_win=10.0,
        avg_loss=-5.0,
        avg_holding_bars=8.5,
        max_consecutive_wins=5,
        max_consecutive_losses=3,
        avg_trades_per_day=0.115,
        stop_loss_count=10,
        stop_loss_gap_count=2,
    )


def _sample_trade(net_pnl: float) -> Trade:
    return Trade(
        direction=Direction.LONG, quantity=1.0,
        entry_price=100.0, entry_timestamp=pd.Timestamp("2024-01-01"),
        exit_price=100.0 + net_pnl,
        exit_timestamp=pd.Timestamp("2024-01-01 00:30"),
        entry_fee=0.05, exit_fee=0.05,
        gross_pnl=net_pnl + 0.1, net_pnl=net_pnl,
        return_pct=net_pnl / 100.0, exit_reason="MANUAL",
    )


# --------------------------------------------------------------------------- #

class TestExportMetricsJson:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "metrics.json"
        export_metrics_json(_sample_metrics(), out)
        assert out.is_file()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["initial_capital"] == 500.0
        assert loaded["sharpe_ratio"] == 1.5
        # Timestamp 應已序列化為 ISO 字串
        assert loaded["start"] == "2024-01-01T00:00:00"

    def test_handles_path_in_snapshot(self, tmp_path: Path) -> None:
        # 透過 config_snapshot 走額外的 _json_default 分支
        out = tmp_path / "snap.json"
        export_config_snapshot({"path": Path("/tmp/x"), "n": 3}, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["path"] == "/tmp/x"


class TestExportTradesCsv:
    def test_writes_with_trades(self, tmp_path: Path) -> None:
        out = tmp_path / "trades.csv"
        export_trades_csv([_sample_trade(10), _sample_trade(-5)], out)
        df = pd.read_csv(out)
        assert len(df) == 2
        for col in [
            "direction", "entry_ts", "exit_ts", "net_pnl", "return_pct",
            "exit_reason", "holding_minutes",
        ]:
            assert col in df.columns

    def test_writes_empty_with_header(self, tmp_path: Path) -> None:
        out = tmp_path / "trades.csv"
        export_trades_csv([], out)
        df = pd.read_csv(out)
        assert len(df) == 0
        assert "direction" in df.columns


class TestExportSummary:
    def test_round_trip(self, tmp_path: Path) -> None:
        out = tmp_path / "summary.txt"
        export_summary_text(["line 1", "line 2", "line 3"], out)
        text = out.read_text(encoding="utf-8")
        assert text.splitlines() == ["line 1", "line 2", "line 3"]
