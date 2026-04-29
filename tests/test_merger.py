"""Equity curve merger 單元測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.types import BacktestResult
from src.broker.types import Trade
from src.metrics.merger import build_merged_result, merge_equity_curves
from src.utils.types import Direction


def _make_curve(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name="equity")


def _make_result(initial: float, equity: list[float], trades: list[Trade] | None = None,
                 *, name: str = "x") -> BacktestResult:
    eq = _make_curve(equity)
    return BacktestResult(
        account_name=name,
        initial_capital=initial,
        final_equity=equity[-1],
        trades=trades or [],
        equity_curve=eq,
        bars_processed=len(equity),
        signals_emitted=10,
        signals_filled=8,
        signals_unfilled=2,
        signals_skipped_pending=0,
        config_snapshot={"src": name},
    )


def _trade(net_pnl: float, entry_minute: int) -> Trade:
    entry = pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=entry_minute)
    return Trade(
        direction=Direction.LONG, quantity=1.0,
        entry_price=100.0, entry_timestamp=entry,
        exit_price=100.0 + net_pnl, exit_timestamp=entry + pd.Timedelta(minutes=10),
        entry_fee=0.05, exit_fee=0.05,
        gross_pnl=net_pnl + 0.1, net_pnl=net_pnl,
        return_pct=net_pnl / 100.0, exit_reason="MANUAL",
    )


# --------------------------------------------------------------------------- #

class TestMergeEquityCurves:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            merge_equity_curves([])

    def test_single_returns_copy(self) -> None:
        c = _make_curve([100, 105, 110])
        out = merge_equity_curves([c])
        # 內容相同
        pd.testing.assert_series_equal(out, c, check_names=False)
        # 是 copy（修改 out 不影響 c）
        out.iloc[0] = 999.0
        assert c.iloc[0] == 100

    def test_same_index_sum(self) -> None:
        a = _make_curve([100, 105, 110])
        b = _make_curve([200, 195, 210])
        out = merge_equity_curves([a, b])
        assert list(out.values) == [300, 300, 320]
        assert (out.index == a.index).all()

    def test_three_curves_same_index(self) -> None:
        a = _make_curve([100, 110])
        b = _make_curve([200, 190])
        c = _make_curve([50, 55])
        out = merge_equity_curves([a, b, c])
        assert list(out.values) == [350, 355]

    def test_different_index_ffill(self) -> None:
        # 兩個 series 時間軸錯開
        a = pd.Series([100, 105, 110],
                      index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05",
                                            "2024-01-01 00:10"]))
        b = pd.Series([200, 195],
                      index=pd.to_datetime(["2024-01-01 00:05", "2024-01-01 00:10"]))
        out = merge_equity_curves([a, b])
        # union index 共 3 個時間點；t=0 b 用首值 200
        assert len(out) == 3
        assert out.iloc[0] == 100 + 200  # b leading NaN -> 200 (first value)
        assert out.iloc[1] == 105 + 200
        assert out.iloc[2] == 110 + 195


class TestBuildMergedResult:
    def test_aggregation_basic(self) -> None:
        long_r = _make_result(500, [500, 510, 520], name="long")
        short_r = _make_result(500, [500, 495, 505], name="short")
        merged = build_merged_result("combined", [long_r, short_r])

        assert merged.account_name == "combined"
        assert merged.initial_capital == 1000
        assert merged.final_equity == 1025  # 520 + 505
        assert merged.signals_emitted == 20  # 10 + 10
        assert merged.signals_filled == 16
        assert merged.signals_unfilled == 4
        assert merged.bars_processed == 3
        assert "components" in merged.config_snapshot

    def test_trades_concatenated_and_sorted(self) -> None:
        long_r = _make_result(500, [500, 500], trades=[_trade(10, 0), _trade(5, 30)],
                              name="long")
        short_r = _make_result(500, [500, 500], trades=[_trade(-3, 15)], name="short")
        merged = build_merged_result("combined", [long_r, short_r])

        ts = [t.entry_timestamp for t in merged.trades]
        # 應按 entry_timestamp 升序：0min, 15min, 30min
        assert ts == sorted(ts)
        assert len(merged.trades) == 3

    def test_empty_components_raises(self) -> None:
        with pytest.raises(ValueError):
            build_merged_result("x", [])
