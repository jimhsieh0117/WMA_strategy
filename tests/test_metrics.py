"""Metrics 計算測試。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.types import BacktestResult
from src.broker.types import Trade
from src.metrics.calculator import bars_per_year, compute_metrics
from src.utils.exceptions import ConfigError
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _build_result(
    initial: float,
    equity_values: list[float],
    trades: list[Trade] | None = None,
    *,
    freq: str = "5min",
) -> BacktestResult:
    idx = pd.date_range("2024-01-01", periods=len(equity_values), freq=freq)
    eq = pd.Series(equity_values, index=idx, name="equity")
    return BacktestResult(
        account_name="test",
        initial_capital=initial,
        final_equity=equity_values[-1],
        trades=trades or [],
        equity_curve=eq,
        bars_processed=len(equity_values),
        signals_emitted=0,
        signals_filled=0,
        signals_unfilled=0,
        signals_skipped_pending=0,
    )


def _trade(net_pnl: float, *, hold_minutes: int = 30) -> Trade:
    """構造一筆交易，只在意 net_pnl / 方向 / 時長。"""
    entry_ts = pd.Timestamp("2024-01-01 00:00")
    exit_ts = entry_ts + pd.Timedelta(minutes=hold_minutes)
    return Trade(
        direction=Direction.LONG, quantity=1.0,
        entry_price=100.0, entry_timestamp=entry_ts,
        exit_price=100.0 + net_pnl, exit_timestamp=exit_ts,
        entry_fee=0.05, exit_fee=0.05,
        gross_pnl=net_pnl + 0.1, net_pnl=net_pnl,
        return_pct=net_pnl / 100.0, exit_reason="MANUAL",
    )


# --------------------------------------------------------------------------- #
# bars_per_year
# --------------------------------------------------------------------------- #

class TestBarsPerYear:
    def test_known_values(self) -> None:
        # 5m: 365 * 24 * 60 / 5 = 105120
        assert bars_per_year("5m") == 105120
        assert bars_per_year("1m") == 525600
        assert bars_per_year("1H") == 8760
        assert bars_per_year("4H") == 2190

    def test_unknown_raises(self) -> None:
        with pytest.raises(ConfigError):
            bars_per_year("2H")


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #

class TestMetricsBasics:
    def test_total_return(self) -> None:
        res = _build_result(initial=100.0, equity_values=[100.0, 105.0, 110.0])
        m = compute_metrics(res, timeframe="5m")
        assert m.total_return_pct == pytest.approx(10.0)
        assert m.final_equity == 110.0

    def test_no_trades_zero_winrate(self) -> None:
        res = _build_result(100.0, [100.0] * 10)
        m = compute_metrics(res, timeframe="5m")
        assert m.total_trades == 0
        assert m.win_rate_pct == 0.0
        assert m.profit_factor == 0.0
        assert m.expectancy == 0.0

    def test_max_drawdown(self) -> None:
        # 100 → 120 → 90 → 110；MDD 從 120 → 90 = -25%
        res = _build_result(100.0, [100.0, 120.0, 90.0, 110.0])
        m = compute_metrics(res, timeframe="5m")
        assert m.max_drawdown_pct == pytest.approx(25.0, rel=1e-6)

    def test_zero_drawdown(self) -> None:
        res = _build_result(100.0, [100.0, 105.0, 110.0])
        m = compute_metrics(res, timeframe="5m")
        assert m.max_drawdown_pct == pytest.approx(0.0, abs=1e-9)


class TestMetricsTradeStats:
    def test_win_rate_and_profit_factor(self) -> None:
        # 3 winners: +10, +20, +30 = 60；2 losers: -5, -15 = -20
        trades = [_trade(10), _trade(20), _trade(30), _trade(-5), _trade(-15)]
        res = _build_result(100.0, [100.0, 110.0, 130.0, 160.0, 155.0, 140.0], trades)
        m = compute_metrics(res, timeframe="5m")
        assert m.total_trades == 5
        assert m.win_rate_pct == pytest.approx(60.0)
        assert m.profit_factor == pytest.approx(60.0 / 20.0)
        assert m.avg_win == pytest.approx(20.0)
        assert m.avg_loss == pytest.approx(-10.0)

    def test_consecutive_streaks(self) -> None:
        # W W W L L W L L L W
        signs = [10, 5, 8, -3, -2, 7, -1, -4, -2, 6]
        trades = [_trade(s) for s in signs]
        res = _build_result(100.0, [100.0] * 11, trades)
        m = compute_metrics(res, timeframe="5m")
        assert m.max_consecutive_wins == 3
        assert m.max_consecutive_losses == 3

    def test_avg_holding_bars_5m(self) -> None:
        # 持倉 30 分鐘 = 6 根 5m
        trades = [_trade(10, hold_minutes=30), _trade(-5, hold_minutes=60)]
        res = _build_result(100.0, [100.0] * 5, trades)
        m = compute_metrics(res, timeframe="5m")
        assert m.avg_holding_bars == pytest.approx((6 + 12) / 2)

    def test_stop_loss_counts(self) -> None:
        t1 = Trade(
            direction=Direction.LONG, quantity=1.0,
            entry_price=100.0, entry_timestamp=pd.Timestamp("2024-01-01"),
            exit_price=95.0, exit_timestamp=pd.Timestamp("2024-01-01 00:30"),
            entry_fee=0, exit_fee=0,
            gross_pnl=-5, net_pnl=-5, return_pct=-0.05,
            exit_reason="STOP_LOSS",
        )
        t2 = Trade(
            direction=Direction.LONG, quantity=1.0,
            entry_price=100.0, entry_timestamp=pd.Timestamp("2024-01-01"),
            exit_price=92.0, exit_timestamp=pd.Timestamp("2024-01-01 00:30"),
            entry_fee=0, exit_fee=0,
            gross_pnl=-8, net_pnl=-8, return_pct=-0.08,
            exit_reason="STOP_LOSS_GAP",
        )
        res = _build_result(100.0, [100.0, 95.0, 92.0], [t1, t2])
        m = compute_metrics(res, timeframe="5m")
        assert m.stop_loss_count == 1
        assert m.stop_loss_gap_count == 1


class TestMetricsAnnualized:
    def test_annualized_return_one_year(self) -> None:
        # 一年，總收益 100% → 年化 100%
        idx = pd.date_range("2024-01-01", "2025-01-01", periods=2)
        eq = pd.Series([100.0, 200.0], index=idx)
        res = BacktestResult(
            account_name="x", initial_capital=100.0, final_equity=200.0,
            trades=[], equity_curve=eq, bars_processed=2,
            signals_emitted=0, signals_filled=0, signals_unfilled=0,
            signals_skipped_pending=0,
        )
        m = compute_metrics(res, timeframe="5m")
        assert m.annualized_return_pct == pytest.approx(100.0, rel=1e-2)
