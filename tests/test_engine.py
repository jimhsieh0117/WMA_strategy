"""Backtest engine 主迴圈測試（M5+：含 TrailingStopController 整合）。

策略行為以 Mock 取代，專注驗證 engine 的撮合 / 持倉管理 / equity 紀錄 / 三階段 stop 整合。
TrailingStopController 內部邏輯另在 test_trailing.py 直接測試，這裡只驗整合面。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import _compute_limit_price, run_backtest
from src.backtest.types import EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import BrokerConfig
from src.strategy.base import BaseTrendStrategy
from src.strategy.types import EntrySignal, StrategyParams, TrailingStopParams
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_test_df(
    n: int,
    *,
    base_price: float = 100.0,
    drift: float = 0.0,
    bb_lower_const: float = 90.0,
    bb_upper_const: float = 110.0,
) -> pd.DataFrame:
    """構造合法的 augmented DataFrame：OHLCV + 全部指標欄。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = np.array([base_price + i * drift for i in range(n)], dtype=np.float64)
    opens = closes - 0.5
    highs = closes + 1.0
    lows = closes - 1.5
    df = pd.DataFrame(
        {
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": [10.0] * n,
            "ha_open": closes, "ha_high": highs, "ha_low": lows, "ha_close": closes,
            "ha_wma_fast": closes, "ha_wma_slow": closes,
            "wma_fast": closes, "wma_slow": closes,
            "bb_middle": closes,
            "bb_upper": [bb_upper_const] * n,
            "bb_lower": [bb_lower_const] * n,
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


class MockStrategy(BaseTrendStrategy):
    """測試用：在指定 bar 發訊號，固定 initial_stop。

    Stage 1/2/3 邏輯由 engine + TrailingStopController 根據 Bollinger 處理；
    Mock 不再需要 compute_trailing_stop_candidate。
    """

    def __init__(
        self,
        direction: Direction,
        *,
        entry_at: int | None = None,
        initial_stop: float = 95.0,
    ) -> None:
        # 用較小的 BB period 避免 mock df 暖機不足
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=5, swing_lookback=2)
        )
        super().__init__(params)
        self.direction = direction  # type: ignore[misc]
        self.entry_at = entry_at
        self.initial_stop = initial_stop

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


def _account_and_broker():
    return (
        Account(initial_capital=500.0, name="test"),
        BrokerSimulator(BrokerConfig(taker_fee_rate=0.0005, slippage_pct=0.0003)),
    )


# --------------------------------------------------------------------------- #
# Limit price helper
# --------------------------------------------------------------------------- #

class TestLimitPriceHelper:
    def test_long_adds_slippage(self) -> None:
        assert _compute_limit_price(100.0, Direction.LONG, 0.001) == pytest.approx(100.1)

    def test_short_subtracts_slippage(self) -> None:
        assert _compute_limit_price(100.0, Direction.SHORT, 0.001) == pytest.approx(99.9)


# --------------------------------------------------------------------------- #
# 沒訊號 / 邊界
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
        assert (result.equity_curve == acct.initial_capital).all()


class TestEngineRiskSizing:
    """risk 模式：撞 stop 時的虧損應接近 risk_per_trade_usdt。"""

    def test_risk_mode_quantity_formula(self) -> None:
        df = make_test_df(20)
        # 構造：bar 12 必撞 stop（low=80）
        df.iloc[12, df.columns.get_loc("low")] = 80.0
        df.iloc[12, df.columns.get_loc("open")] = 99.0
        df.iloc[12, df.columns.get_loc("close")] = 95.0
        df.iloc[12, df.columns.get_loc("high")] = 99.5

        acct, broker = _account_and_broker()
        # initial_stop = 95；entry 約 bar11 open ≈ 99.5；R 約 5
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=95.0)
        cfg = EngineConfig(sizing_mode="risk", risk_per_trade_usdt=1.0)
        result = run_backtest(df, strat, acct, broker, cfg)

        assert len(result.trades) == 1
        trade = result.trades[0]
        # 虧損應 ≈ 1 USDT（含手續費）；允許 ±0.05 USDT 容錯
        assert -1.05 < trade.net_pnl < -0.95, (
            f"expected ~-1 USDT, got {trade.net_pnl:.4f}"
        )

    def test_risk_mode_rejects_when_overleveraged(self) -> None:
        df = make_test_df(20)
        # 把 stop 設得極接近 entry → R 太小 → notional 會爆
        # entry 約 99.5、stop = 99.49 → R ≈ 0.01；risk=1 → qty ≈ 100 → notional ≈ 9950 > equity
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=99.49)
        cfg = EngineConfig(sizing_mode="risk", risk_per_trade_usdt=1.0)
        result = run_backtest(df, strat, acct, broker, cfg)

        # 訊號發出但拒絕成交
        assert result.signals_emitted == 1
        assert result.signals_filled == 0
        assert result.signals_unfilled >= 1
        assert len(result.trades) == 0


class TestEngineErrors:
    def test_empty_df_raises(self) -> None:
        df = make_test_df(0)
        acct, broker = _account_and_broker()
        with pytest.raises(ValueError):
            run_backtest(df, MockStrategy(Direction.LONG), acct, broker)

    def test_missing_indicator_column_raises(self) -> None:
        df = make_test_df(20).drop(columns=["bb_lower"])
        acct, broker = _account_and_broker()
        from src.utils.exceptions import DataIntegrityError
        with pytest.raises(DataIntegrityError):
            run_backtest(df, MockStrategy(Direction.LONG), acct, broker)


# --------------------------------------------------------------------------- #
# 單筆交易 + 止損
# --------------------------------------------------------------------------- #

class TestEngineSingleTrade:
    def test_long_entry_then_stop_loss(self) -> None:
        df = make_test_df(20)
        # 構造 bar 12 的 low 跌穿 stop=95
        df.iloc[12, df.columns.get_loc("low")] = 90.0
        df.iloc[12, df.columns.get_loc("open")] = 99.0
        df.iloc[12, df.columns.get_loc("close")] = 95.5
        df.iloc[12, df.columns.get_loc("high")] = 99.5

        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=95.0)
        result = run_backtest(df, strat, acct, broker, EngineConfig())

        assert result.signals_emitted == 1
        assert result.signals_filled == 1
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.direction is Direction.LONG
        assert trade.exit_reason in ("STOP_LOSS", "STOP_LOSS_GAP")
        assert result.final_equity < acct.initial_capital

    def test_long_position_size_60pct_of_equity(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        result = run_backtest(df, strat, acct, broker, EngineConfig(position_size_pct=0.6))

        assert acct.has_position()
        pos = acct.position
        bar11_open = float(df.iloc[11]["open"])
        expected_limit = bar11_open * 1.0003
        expected_qty = (500.0 * 0.6) / expected_limit
        assert pos.quantity == pytest.approx(expected_qty, rel=1e-9)


# --------------------------------------------------------------------------- #
# Trailing 整合（透過 Bollinger 與價格走勢）
# --------------------------------------------------------------------------- #

class TestEngineTrailingIntegration:
    def test_long_ratchets_via_bollinger_when_price_advances(self) -> None:
        # 構造價格大漲到 stage 3，且 bb_lower 給足夠高 → stop 應 ratchet 上去
        df = make_test_df(20, bb_lower_const=88.0)
        # bar 12 起價格大漲到 entry + 3R 以上
        for i in range(12, 20):
            df.iloc[i, df.columns.get_loc("high")] = 200.0
            df.iloc[i, df.columns.get_loc("open")] = 150.0
            df.iloc[i, df.columns.get_loc("close")] = 180.0
            # low 必須高於 stage2 ratchet 後的 stop（≈ entry+0.2R），否則會被掃
            df.iloc[i, df.columns.get_loc("low")] = 130.0

        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        # entry ≈ bar 11 open ≈ 99.5；R ≈ 19.5
        run_backtest(df, strat, acct, broker, EngineConfig())

        # stop 應遠超 entry（進獲利區）
        assert acct.has_position()
        # stop 應 >= stage 2 fixed = entry × (1 + 2×taker) + 0.2R
        # （滑點已在 entry_price，不再加 slippage_pct）
        entry = acct.position.entry_price
        R = abs(entry - 80.0)
        expected_min = entry * 1.0010 + 0.2 * R
        assert acct.position.stop_price >= expected_min

    def test_long_no_ratchet_when_price_stagnant(self) -> None:
        df = make_test_df(20)  # 價格維持 ~100，Bollinger lower=90
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=85.0)
        run_backtest(df, strat, acct, broker, EngineConfig())

        # 價格沒漲 → stage 不推進 → stop 維持初始 85
        if acct.has_position():
            assert acct.position.stop_price == 85.0


# --------------------------------------------------------------------------- #
# Force close
# --------------------------------------------------------------------------- #

class TestEngineForceClose:
    def test_force_close_at_end(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        run_backtest(df, strat, acct, broker, EngineConfig(force_close_at_end=True))
        assert not acct.has_position()
        assert any(t.exit_reason == "FORCE_CLOSE_END" for t in acct.trade_log)

    def test_no_force_close_keeps_position(self) -> None:
        df = make_test_df(20)
        acct, broker = _account_and_broker()
        strat = MockStrategy(Direction.LONG, entry_at=10, initial_stop=80.0)
        run_backtest(df, strat, acct, broker, EngineConfig(force_close_at_end=False))
        assert acct.has_position()


# --------------------------------------------------------------------------- #
# Equity curve
# --------------------------------------------------------------------------- #

class TestEngineEquityCurve:
    def test_curve_length_matches_bars(self) -> None:
        df = make_test_df(50)
        acct, broker = _account_and_broker()
        result = run_backtest(df, MockStrategy(Direction.LONG), acct, broker)
        assert len(result.equity_curve) == len(df)
        assert result.equity_curve.index[0] == df.index[0]
        assert result.equity_curve.index[-1] == df.index[-1]
