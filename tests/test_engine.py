"""Backtest engine 主迴圈測試。

策略行為以 Mock 取代，專注驗證 engine 的撮合 / ratchet / 持倉管理 / equity 紀錄邏輯。
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _compute_limit_price, run_backtest
from src.backtest.types import BacktestResult, EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import BrokerConfig
from src.strategy.base import REQUIRED_INDICATOR_COLUMNS, BaseTrendStrategy
from src.strategy.types import EntrySignal, StrategyParams
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_test_df(n: int, *, base_price: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    """構造合法的 augmented DataFrame：OHLCV + 全部指標欄。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = np.array([base_price + i * drift for i in range(n)], dtype=np.float64)
    opens = closes - 0.5
    highs = closes + 1.0
    lows = closes - 1.5
    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [10.0] * n,
            "ha_open": closes,
            "ha_high": highs,
            "ha_low": lows,
            "ha_close": closes,
            "ha_wma_fast": closes,
            "ha_wma_slow": closes,
            "atr": [1.0] * n,
        },
        index=idx,
    )
    # OHLC 一致性
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


class MockStrategy(BaseTrendStrategy):
    """測試專用：在指定 bar 發訊號，依設定提供拖曳止損候選值。"""

    def __init__(
        self,
        direction: Direction,
        *,
        entry_at: int | None = None,
        initial_stop: float = 95.0,
        trailing_provider=None,
    ) -> None:
        super().__init__(StrategyParams())
        self.direction = direction  # type: ignore[misc]
        self.entry_at = entry_at
        self.initial_stop = initial_stop
        self.trailing_provider = trailing_provider

    def detect_entry(self, df, bar_index):
        if self.entry_at is None or bar_index != self.entry_at:
            return None
        return EntrySignal(
            direction=self.direction,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=self.initial_stop,
            reason="mock",
        )

    def compute_trailing_stop_candidate(self, df, bar_index):
        if self.trailing_provider is None:
            return math.nan
        return self.trailing_provider(bar_index)


def _account_and_broker():
    return (
        Account(initial_capital=500.0, name="test"),
        BrokerSimulator(BrokerConfig(taker_fee_rate=0.0005, slippage_pct=0.0003)),
    )


# --------------------------------------------------------------------------- #
# Helpers (low-level)
# --------------------------------------------------------------------------- #

class TestLimitPriceHelper:
    def test_long_adds_slippage(self) -> None:
        assert _compute_limit_price(100.0, Direction.LONG, 0.001) == pytest.approx(100.1)

    def test_short_subtracts_slippage(self) -> None:
        assert _compute_limit_price(100.0, Direction.SHORT, 0.001) == pytest.approx(99.9)

    def test_zero_slippage(self) -> None:
        assert _compute_limit_price(100.0, Direction.LONG, 0.0) == 100.0


# --------------------------------------------------------------------------- #
# Engine integration
# --------------------------------------------------------------------------- #

class TestEngineNoSignal:
    def test_flat_equity_when_no_signal(self) -> None:
        df = make_test_df(50)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=None)
        result = run_backtest(df, strat, acct, broker, EngineConfig())

        assert result.bars_processed == 50
        assert len(result.trades) == 0
        assert result.signals_emitted == 0
        # equity 全為 initial（無交易）
        assert (result.equity_curve == acct.initial_capital).all()
        assert result.final_equity == pytest.approx(acct.initial_capital)


class TestEngineSingleTrade:
    def test_long_entry_then_stop_loss(self) -> None:
        # 在 bar 10 發訊號，stop 設低；構造 bar 12 的 low 跌穿 stop
        df = make_test_df(20)
        df.iloc[12, df.columns.get_loc("low")] = 90.0  # 跌破 stop=95
        df.iloc[12, df.columns.get_loc("high")] = max(df.iloc[12]["close"], 96.0)
        # 重新確保 OHLC 一致
        df.loc[df.index[12], "open"] = max(df.iloc[12]["low"], min(df.iloc[12]["high"], df.iloc[12]["open"]))

        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=95.0)
        result = run_backtest(df, strat, acct, broker, EngineConfig())

        assert result.signals_emitted == 1
        assert result.signals_filled == 1
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.direction is Direction.LONG
        assert trade.exit_reason in ("STOP_LOSS", "STOP_LOSS_GAP")
        assert result.final_equity < acct.initial_capital  # 賠錢

    def test_long_position_size_60pct_of_equity(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        result = run_backtest(df, strat, acct, broker, EngineConfig(position_size_pct=0.6))

        assert len(acct.trade_log) == 0  # 位於 bar 11 才成交，沒止損
        assert acct.has_position()
        pos = acct.position
        # bar 11 open ≈ 100 + 11*0 - 0.5 = 99.5；limit = 99.5 * 1.0003
        bar11_open = float(df.iloc[11]["open"])
        expected_limit = bar11_open * 1.0003
        # quantity ≈ (500 * 0.6) / expected_limit
        expected_qty = (500.0 * 0.6) / expected_limit
        assert pos.quantity == pytest.approx(expected_qty, rel=1e-9)


class TestEngineTrailingStop:
    def test_long_trailing_ratchet_up_only(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()

        # 從 bar 12 起，提供逐根上升的止損候選
        def trailer(i: int) -> float:
            if i < 12:
                return math.nan
            return 90.0 + (i - 12) * 0.5  # 90, 90.5, 91.0, ...

        strat = MockStrategy(
            Direction.LONG, entry_at=10, initial_stop=85.0, trailing_provider=trailer
        )
        result = run_backtest(df, strat, acct, broker, EngineConfig())

        # 最後 stop 應已被 ratchet 過
        assert acct.has_position()
        assert acct.position.stop_price > 85.0  # 已從初始 85 上移

    def test_long_trailing_does_not_ratchet_down(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()

        # 候選一直比初始低 → 不該更新
        def trailer(i: int) -> float:
            if i < 12:
                return math.nan
            return 80.0  # 小於初始 stop 85

        strat = MockStrategy(
            Direction.LONG, entry_at=10, initial_stop=85.0, trailing_provider=trailer
        )
        result = run_backtest(df, strat, acct, broker, EngineConfig())

        assert acct.has_position()
        assert acct.position.stop_price == 85.0  # 維持不變


class TestEngineForceClose:
    def test_force_close_at_end(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        result = run_backtest(
            df, strat, acct, broker, EngineConfig(force_close_at_end=True)
        )
        assert not acct.has_position()
        assert len(acct.trade_log) == 1
        assert acct.trade_log[0].exit_reason == "FORCE_CLOSE_END"

    def test_no_force_close_keeps_position(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        result = run_backtest(
            df, strat, acct, broker, EngineConfig(force_close_at_end=False)
        )
        assert acct.has_position()
        assert len(acct.trade_log) == 0


class TestEngineEquityCurve:
    def test_curve_length_matches_bars(self) -> None:
        df = make_test_df(50)
        acct, broker = _account_and_broker()
        result = run_backtest(df, MockStrategy(Direction.LONG), acct, broker)
        assert len(result.equity_curve) == len(df)
        assert result.equity_curve.index[0] == df.index[0]
        assert result.equity_curve.index[-1] == df.index[-1]


class TestEngineErrors:
    def test_empty_df_raises(self) -> None:
        df = make_test_df(0)
        acct, broker = _account_and_broker()
        with pytest.raises(ValueError):
            run_backtest(df, MockStrategy(Direction.LONG), acct, broker)

    def test_missing_indicator_column_raises(self) -> None:
        df = make_test_df(20).drop(columns=["atr"])
        acct, broker = _account_and_broker()
        from src.utils.exceptions import DataIntegrityError
        with pytest.raises(DataIntegrityError):
            run_backtest(df, MockStrategy(Direction.LONG), acct, broker)
