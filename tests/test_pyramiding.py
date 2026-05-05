"""多倉並行（allow_pyramiding=True）測試。

驗證：
1. Account 多倉 dict API（open_position_multi / close_by_id / update_stop_by_id / equity 累加）
2. Engine 在 allow_pyramiding=True 下能同時持有多筆獨立止損的倉位
3. 每根 K 線最多新增 1 筆（pending 機制）
4. leverage_cap 槽位用盡時拒絕新倉
5. 個別倉位被止損平倉，其他倉位不受影響、controller 正確移除
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from src.backtest.types import EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import BrokerConfig
from src.strategy.base import BaseTrendStrategy
from src.strategy.types import EntrySignal, StrategyParams, TrailingStopParams
from src.utils.exceptions import AccountInvariantError
from src.utils.types import Direction

TS1 = pd.Timestamp("2024-01-01 00:00")
TS2 = pd.Timestamp("2024-01-01 00:05")
TS3 = pd.Timestamp("2024-01-01 00:10")


# --------------------------------------------------------------------------- #
# Account 多倉 API
# --------------------------------------------------------------------------- #

class TestAccountMulti:
    def test_open_two_positions_assigns_unique_ids(self) -> None:
        a = Account(500.0)
        pid1 = a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        pid2 = a.open_position_multi(Direction.LONG, 2.0, 101.0, TS2, 96.0, fee=0.10)
        assert pid1 != pid2
        assert a.position_count == 2
        assert set(a.positions.keys()) == {pid1, pid2}

    def test_legacy_position_property_raises_when_multi(self) -> None:
        a = Account(500.0)
        a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        a.open_position_multi(Direction.LONG, 1.0, 101.0, TS2, 96.0, fee=0.05)
        with pytest.raises(AccountInvariantError):
            _ = a.position

    def test_legacy_close_position_raises_when_multi(self) -> None:
        a = Account(500.0)
        a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        a.open_position_multi(Direction.LONG, 1.0, 101.0, TS2, 96.0, fee=0.05)
        with pytest.raises(AccountInvariantError):
            a.close_position(110.0, TS3, fee=0.1, reason="MANUAL")

    def test_equity_sums_unrealized_pnl_across_positions(self) -> None:
        a = Account(500.0)
        a.open_position_multi(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.0)
        a.open_position_multi(Direction.LONG, 1.0, 102.0, TS2, 97.0, fee=0.0)
        # mark @ 105：第一筆 +10、第二筆 +3
        assert a.equity(105.0) == pytest.approx(500.0 + 10.0 + 3.0)

    def test_close_by_id_only_affects_that_position(self) -> None:
        a = Account(500.0)
        pid1 = a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        pid2 = a.open_position_multi(Direction.LONG, 2.0, 102.0, TS2, 97.0, fee=0.10)
        trade = a.close_position_by_id(pid1, 110.0, TS3, fee=0.05, reason="MANUAL")
        assert trade.position_id == pid1
        assert a.position_count == 1
        assert pid2 in a.positions
        assert pid1 not in a.positions

    def test_update_stop_by_id_only_affects_that_position(self) -> None:
        a = Account(500.0)
        pid1 = a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        pid2 = a.open_position_multi(Direction.LONG, 1.0, 100.0, TS2, 95.0, fee=0.05)
        a.update_stop_by_id(pid1, 96.0, timestamp=TS3)
        assert a.positions[pid1].stop_price == 96.0
        assert a.positions[pid2].stop_price == 95.0
        # stop_history 含 entry + 一次 ratchet
        assert len(a.positions[pid1].stop_history) == 2

    def test_total_notional_sum(self) -> None:
        a = Account(500.0)
        a.open_position_multi(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        a.open_position_multi(Direction.LONG, 2.0, 110.0, TS2, 105.0, fee=0.10)
        assert a.total_notional_at_entry == pytest.approx(100.0 + 220.0)


# --------------------------------------------------------------------------- #
# Engine 整合：每根 K 一筆訊號，連續多 K 開倉
# --------------------------------------------------------------------------- #

def _make_df(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    closes = np.full(n, 100.0)
    df = pd.DataFrame(
        {
            "open": closes - 0.5, "high": closes + 1.0, "low": closes - 1.5,
            "close": closes, "volume": np.full(n, 10.0),
            "ha_open": closes, "ha_high": closes + 1.0, "ha_low": closes - 1.5,
            "ha_close": closes,
            "ha_wma_fast": closes, "ha_wma_slow": closes,
            "wma_fast": closes, "wma_slow": closes,
            "bb_middle": closes,
            "bb_upper": np.full(n, 110.0), "bb_lower": np.full(n, 90.0),
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


class _MultiSignalStrategy(BaseTrendStrategy):
    """每根指定 bar 都發一個訊號，固定 stop。"""

    def __init__(self, direction: Direction, *, entry_bars: list[int],
                 initial_stop: float = 95.0) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=5, swing_lookback=2)
        )
        super().__init__(params)
        self.direction = direction  # type: ignore[misc]
        self.entry_bars = set(entry_bars)
        self.initial_stop = initial_stop

    def detect_entry(self, df, bar_index):
        if bar_index not in self.entry_bars:
            return None
        return EntrySignal(
            direction=self.direction,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=self.initial_stop,
            reason="mock",
        )


def _broker() -> BrokerSimulator:
    return BrokerSimulator(BrokerConfig(taker_fee_rate=0.0005, slippage_pct=0.0003))


class TestEnginePyramiding:
    def test_two_concurrent_positions_filled(self) -> None:
        df = _make_df(20)
        acct = Account(500.0, name="t")
        # bar 5 與 bar 7 各發一個訊號 → bar 6、bar 8 各成交一筆
        strat = _MultiSignalStrategy(
            Direction.LONG, entry_bars=[5, 7], initial_stop=95.0
        )
        cfg = EngineConfig(
            sizing_mode="risk", risk_per_trade_usdt=1.0,
            allow_pyramiding=True, leverage_cap=1.0,
        )
        run_backtest(df, strat, acct, _broker(), cfg)
        # 在 bar 8 結束時兩筆都還在；之後沒有訊號也沒撞 stop
        # 回測尾沒設 force_close → 兩筆持倉留到結尾
        assert acct.position_count == 2
        # 每筆 stop_history 至少有一筆（entry 時寫入）
        for pos in acct.positions.values():
            assert len(pos.stop_history) >= 1

    def test_disabled_pyramiding_keeps_single_position(self) -> None:
        """allow_pyramiding=False 時行為與舊版一致：第二個訊號被過濾掉。"""
        df = _make_df(20)
        acct = Account(500.0, name="t")
        strat = _MultiSignalStrategy(
            Direction.LONG, entry_bars=[5, 7], initial_stop=95.0
        )
        cfg = EngineConfig(
            sizing_mode="risk", risk_per_trade_usdt=1.0,
            allow_pyramiding=False,
        )
        run_backtest(df, strat, acct, _broker(), cfg)
        assert acct.position_count <= 1

    def test_one_position_stopped_others_survive(self) -> None:
        """讓第一筆撞 stop，第二筆繼續存活。"""
        df = _make_df(25)
        # bar 12 製造大跌：把 low 拉到 90，但 open/close 維持高位 → 只有第一筆（stop=95）撞，
        # 第二筆 stop=92 不會撞
        df.iloc[12, df.columns.get_loc("low")] = 90.0
        df.iloc[12, df.columns.get_loc("open")] = 99.0
        df.iloc[12, df.columns.get_loc("close")] = 96.0
        df.iloc[12, df.columns.get_loc("high")] = 99.5

        acct = Account(500.0, name="t")
        # 兩個訊號用不同 stop：
        class TwoStopsStrategy(BaseTrendStrategy):
            def __init__(self) -> None:
                super().__init__(StrategyParams(
                    trailing=TrailingStopParams(bollinger_period=5, swing_lookback=2)
                ))
                self.direction = Direction.LONG  # type: ignore[misc]

            def detect_entry(self, df, bar_index):
                if bar_index == 5:
                    return EntrySignal(Direction.LONG, bar_index,
                                       df.index[bar_index], 95.0, "s1")
                if bar_index == 7:
                    return EntrySignal(Direction.LONG, bar_index,
                                       df.index[bar_index], 88.0, "s2")
                return None

        cfg = EngineConfig(
            sizing_mode="risk", risk_per_trade_usdt=1.0,
            allow_pyramiding=True, leverage_cap=1.0,
        )
        result = run_backtest(df, TwoStopsStrategy(), acct, _broker(), cfg)
        # 第一筆已被止損平倉，第二筆仍在帳上
        assert len(result.trades) == 1
        assert result.trades[0].exit_reason in ("STOP_LOSS", "STOP_LOSS_GAP")
        assert acct.position_count == 1

    def test_leverage_cap_rejects_excess_positions(self) -> None:
        """leverage_cap 用盡 → 拒絕後續新倉（risk 模式）。"""
        df = _make_df(30)

        class ManySignals(BaseTrendStrategy):
            def __init__(self) -> None:
                super().__init__(StrategyParams(
                    trailing=TrailingStopParams(bollinger_period=5, swing_lookback=2)
                ))
                self.direction = Direction.LONG  # type: ignore[misc]

            def detect_entry(self, df, bar_index):
                # bar 5..15 連發訊號（stop=98 在 bar.low=98.5 之下，避免進場即觸發）
                if 5 <= bar_index <= 15:
                    return EntrySignal(Direction.LONG, bar_index,
                                       df.index[bar_index], 98.0, "many")
                return None

        acct = Account(500.0, name="t")
        # entry≈99.53、stop=98 → R≈1.53、denom≈1.63、qty≈0.61、notional≈61
        # cap = 500 × 1.0 = 500 → 理論上最多約 8 筆並行
        cfg = EngineConfig(
            sizing_mode="risk", risk_per_trade_usdt=1.0,
            allow_pyramiding=True, leverage_cap=1.0,
        )
        result = run_backtest(df, ManySignals(), acct, _broker(), cfg)
        assert result.signals_emitted >= 10
        # 受 leverage_cap 限制：應該至少擋下 1 筆
        assert result.signals_filled < result.signals_emitted
        assert result.signals_unfilled >= 1
        # 累計 notional 不超過 cap
        assert acct.total_notional_at_entry <= 500.0 + 1e-6
