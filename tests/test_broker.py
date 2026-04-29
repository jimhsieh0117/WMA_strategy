"""Broker 模組單元測試（types / Account / BrokerSimulator）。

涵蓋場景：
1. Position / Trade dataclass 計算
2. Account 開倉、平倉、cash 流動、不變式
3. Simulator 限價成交（多空 / 成交 / 不成交 / 邊界價格）
4. 止損觸發（盤中 / 跳空 / 同根進場）
5. 同根進場 + 同根止損的雙手續費結算
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import (
    Bar,
    BrokerConfig,
    FillResult,
    LimitOrder,
    Position,
    Trade,
)
from src.utils.exceptions import (
    AccountInvariantError,
    ConfigError,
    OrderExecutionError,
)
from src.utils.types import Direction

TS1 = pd.Timestamp("2024-01-01 00:00:00")
TS2 = pd.Timestamp("2024-01-01 00:05:00")
TS3 = pd.Timestamp("2024-01-01 00:10:00")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _bar(ts: pd.Timestamp, o: float, h: float, l: float, c: float) -> Bar:
    return Bar(timestamp=ts, open=o, high=h, low=l, close=c)


# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

class TestBar:
    def test_from_row(self) -> None:
        row = pd.Series({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
        bar = Bar.from_row(TS1, row)
        assert bar.timestamp == TS1
        assert bar.open == 1.0
        assert bar.high == 2.0
        assert bar.low == 0.5
        assert bar.close == 1.5


class TestBrokerConfig:
    def test_defaults(self) -> None:
        c = BrokerConfig()
        assert c.taker_fee_rate == 0.0005
        assert c.maker_fee_rate == 0.0002
        assert c.slippage_pct == 0.0003

    def test_invalid_negative(self) -> None:
        with pytest.raises(ConfigError):
            BrokerConfig(taker_fee_rate=-0.001)

    def test_invalid_too_large(self) -> None:
        with pytest.raises(ConfigError):
            BrokerConfig(slippage_pct=0.5)


class TestLimitOrder:
    def test_invalid_quantity_zero(self) -> None:
        with pytest.raises(ConfigError):
            LimitOrder(Direction.LONG, limit_price=100.0, quantity=0, initial_stop=99.0)

    def test_invalid_negative_price(self) -> None:
        with pytest.raises(ConfigError):
            LimitOrder(Direction.LONG, limit_price=-1.0, quantity=1.0, initial_stop=99.0)


class TestPosition:
    def test_unrealized_long(self) -> None:
        p = Position(Direction.LONG, 2.0, 100.0, TS1, 95.0, 0.1)
        assert p.unrealized_pnl(105.0) == pytest.approx(10.0)
        assert p.unrealized_pnl(95.0) == pytest.approx(-10.0)

    def test_unrealized_short(self) -> None:
        p = Position(Direction.SHORT, 2.0, 100.0, TS1, 105.0, 0.1)
        assert p.unrealized_pnl(95.0) == pytest.approx(10.0)
        assert p.unrealized_pnl(105.0) == pytest.approx(-10.0)

    def test_notional(self) -> None:
        p = Position(Direction.LONG, 3.0, 100.0, TS1, 95.0, 0.1)
        assert p.notional_at_entry == pytest.approx(300.0)


# --------------------------------------------------------------------------- #
# Account
# --------------------------------------------------------------------------- #

class TestAccount:
    def test_initial_state(self) -> None:
        a = Account(500.0, name="test")
        assert a.cash == 500.0
        assert a.position is None
        assert not a.has_position()
        assert a.equity(0) == 500.0  # mark_price 不影響無倉位 equity
        assert a.trade_log == []

    def test_invalid_initial_capital(self) -> None:
        with pytest.raises(AccountInvariantError):
            Account(0)
        with pytest.raises(AccountInvariantError):
            Account(-100)

    def test_open_long_deducts_fee(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 3.0, 100.0, TS1, 95.0, fee=0.15)
        assert a.cash == pytest.approx(499.85)
        assert a.has_position()
        assert a.position.direction is Direction.LONG
        assert a.position.entry_price == 100.0
        assert a.position.stop_price == 95.0

    def test_double_open_raises(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.1)
        with pytest.raises(AccountInvariantError, match="already has"):
            a.open_position(Direction.SHORT, 1.0, 100.0, TS2, 105.0, fee=0.1)

    def test_open_invalid_long_stop(self) -> None:
        a = Account(500.0)
        with pytest.raises(AccountInvariantError, match="long stop"):
            a.open_position(Direction.LONG, 1.0, 100.0, TS1, 105.0, fee=0.1)

    def test_open_invalid_short_stop(self) -> None:
        a = Account(500.0)
        with pytest.raises(AccountInvariantError, match="short stop"):
            a.open_position(Direction.SHORT, 1.0, 100.0, TS1, 95.0, fee=0.1)

    def test_open_negative_quantity_raises(self) -> None:
        a = Account(500.0)
        with pytest.raises(AccountInvariantError, match="quantity"):
            a.open_position(Direction.LONG, 0.0, 100.0, TS1, 95.0, fee=0.1)
        with pytest.raises(AccountInvariantError):
            a.open_position(Direction.LONG, -1.0, 100.0, TS1, 95.0, fee=0.1)

    def test_open_fee_exceeds_cash_raises(self) -> None:
        a = Account(0.05)
        with pytest.raises(AccountInvariantError, match="negative"):
            a.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.5)

    def test_close_realizes_pnl_long(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.1)
        # 平倉於 110 → gross = 2*(110-100) = 20，exit_fee=0.11
        trade = a.close_position(110.0, TS2, fee=0.11, reason="MANUAL")
        assert trade.gross_pnl == pytest.approx(20.0)
        assert trade.net_pnl == pytest.approx(20.0 - 0.1 - 0.11)
        assert trade.exit_reason == "MANUAL"
        # cash = 500 - 0.1 (entry_fee) + 20 (gross) - 0.11 (exit_fee) = 519.79
        assert a.cash == pytest.approx(519.79)
        assert a.position is None
        assert len(a.trade_log) == 1

    def test_close_realizes_pnl_short(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.SHORT, 2.0, 100.0, TS1, 105.0, fee=0.1)
        # 平倉於 95 → gross = -1 * 2 * (95 - 100) = 10
        trade = a.close_position(95.0, TS2, fee=0.095, reason="MANUAL")
        assert trade.gross_pnl == pytest.approx(10.0)
        assert a.cash == pytest.approx(500.0 - 0.1 + 10.0 - 0.095)

    def test_close_no_position_raises(self) -> None:
        a = Account(500.0)
        with pytest.raises(AccountInvariantError, match="no position"):
            a.close_position(100.0, TS2, fee=0.1, reason="MANUAL")

    def test_update_stop_accepts_above_entry_for_long(self) -> None:
        """拖曳止損可超越 entry 進入利潤區（價格大漲後 ratchet 結果）。"""
        a = Account(500.0)
        a.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.1)
        a.update_stop(97.0)
        assert a.position.stop_price == 97.0
        # 多單在價格上漲後合法地把 stop 移到 entry 之上
        a.update_stop(105.0)
        assert a.position.stop_price == 105.0

    def test_update_stop_rejects_non_positive(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.1)
        with pytest.raises(AccountInvariantError, match=">"):
            a.update_stop(0.0)
        with pytest.raises(AccountInvariantError, match=">"):
            a.update_stop(-5.0)

    def test_update_stop_no_position_raises(self) -> None:
        a = Account(500.0)
        with pytest.raises(AccountInvariantError, match="no position"):
            a.update_stop(95.0)

    def test_equity_with_position(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.1)
        # equity = cash + 2 * (mark - 100) = 499.9 + 20 = 519.9 at mark=110
        assert a.equity(110.0) == pytest.approx(519.9)
        assert a.equity(100.0) == pytest.approx(499.9)

    def test_snapshot_equity(self) -> None:
        a = Account(500.0)
        a.snapshot_equity(100.0, TS1)
        a.snapshot_equity(101.0, TS2)
        hist = a.equity_history
        assert len(hist) == 2
        assert hist[0] == (TS1, 500.0)
        assert hist[1] == (TS2, 500.0)

    def test_trade_log_returns_copy(self) -> None:
        a = Account(500.0)
        a.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.1)
        a.close_position(110.0, TS2, fee=0.11, reason="MANUAL")
        log = a.trade_log
        log.clear()
        # 內部不應被外部清空
        assert len(a.trade_log) == 1


# --------------------------------------------------------------------------- #
# BrokerSimulator: Limit fill
# --------------------------------------------------------------------------- #

class TestBrokerLimitFill:
    def setup_method(self) -> None:
        self.broker = BrokerSimulator(BrokerConfig())
        self.account = Account(500.0)

    def test_long_limit_fills_when_low_below_limit(self) -> None:
        bar = _bar(TS1, o=100.0, h=101.0, l=99.5, c=100.5)
        order = LimitOrder(Direction.LONG, limit_price=100.03, quantity=3.0, initial_stop=98.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert res.filled
        assert res.fill_price == 100.03
        # fee = 3 * 100.03 * 0.0005 = 0.150045
        assert res.fee == pytest.approx(0.150045)
        assert self.account.has_position()

    def test_short_limit_fills_when_high_above_limit(self) -> None:
        bar = _bar(TS1, o=100.0, h=101.5, l=99.5, c=100.5)
        order = LimitOrder(Direction.SHORT, limit_price=100.5, quantity=2.0, initial_stop=102.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert res.filled
        assert res.fill_price == 100.5

    def test_long_no_fill_when_low_above_limit(self) -> None:
        # limit 太低 → bar.low > limit → 不成交
        bar = _bar(TS1, o=200.0, h=201.0, l=199.0, c=200.5)
        order = LimitOrder(Direction.LONG, limit_price=100.0, quantity=1.0, initial_stop=95.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert not res.filled
        assert "below bar.low" in res.reason
        assert not self.account.has_position()

    def test_short_no_fill_when_high_below_limit(self) -> None:
        bar = _bar(TS1, o=50.0, h=51.0, l=49.0, c=50.5)
        order = LimitOrder(Direction.SHORT, limit_price=100.0, quantity=1.0, initial_stop=105.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert not res.filled
        assert "above bar.high" in res.reason

    def test_long_limit_above_bar_high_caps_at_high(self) -> None:
        """Marketable limit（限價 > bar.high）以 bar.high 成交，不應 raise。"""
        bar = _bar(TS1, o=1000.0, h=1000.5, l=999.5, c=1000.2)
        order = LimitOrder(Direction.LONG, limit_price=1001.0, quantity=1.0, initial_stop=995.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert res.filled
        assert res.fill_price == 1000.5  # capped at bar.high

    def test_short_limit_below_bar_low_caps_at_low(self) -> None:
        bar = _bar(TS1, o=1000.0, h=1000.5, l=999.5, c=1000.2)
        order = LimitOrder(Direction.SHORT, limit_price=999.0, quantity=1.0, initial_stop=1005.0)
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert res.filled
        assert res.fill_price == 999.5  # capped at bar.low

    def test_long_stop_above_fill_after_cap_aborts(self) -> None:
        """訊號 bar 算的 stop 跨 bar 後高過實際 fill_price → 放棄進場（不 raise）。"""
        # bar.high = 1000.05；limit = 1001.0 被 cap 到 1000.05；
        # 但 initial_stop = 1000.10 > fill 1000.05 → 不安全
        bar = _bar(TS1, o=1000.0, h=1000.05, l=999.5, c=1000.0)
        order = LimitOrder(
            Direction.LONG, limit_price=1001.0, quantity=1.0, initial_stop=1000.10,
        )
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert not res.filled
        assert "not below fill" in res.reason
        assert not self.account.has_position()

    def test_short_stop_below_fill_after_cap_aborts(self) -> None:
        bar = _bar(TS1, o=1000.0, h=1000.5, l=999.95, c=1000.0)
        order = LimitOrder(
            Direction.SHORT, limit_price=999.0, quantity=1.0, initial_stop=999.90,
        )
        res = self.broker.try_fill_limit(order, bar, self.account)
        assert not res.filled
        assert "not above fill" in res.reason

    def test_fill_with_existing_position_raises(self) -> None:
        # 已有持倉再嘗試成交 → raise
        self.account.open_position(Direction.LONG, 1.0, 100.0, TS1, 95.0, fee=0.05)
        bar = _bar(TS2, o=100.0, h=101.0, l=99.0, c=100.5)
        order = LimitOrder(Direction.LONG, limit_price=100.5, quantity=1.0, initial_stop=95.0)
        with pytest.raises(OrderExecutionError, match="already has"):
            self.broker.try_fill_limit(order, bar, self.account)


# --------------------------------------------------------------------------- #
# BrokerSimulator: Stop trigger
# --------------------------------------------------------------------------- #

class TestBrokerStop:
    def setup_method(self) -> None:
        self.broker = BrokerSimulator(BrokerConfig())
        self.account = Account(500.0)

    # ---------- LONG ----------

    def test_long_stop_triggered_intraday(self) -> None:
        # 先開倉於 TS1
        self.account.open_position(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.1)
        # 後續 bar：open=98 (>95), low=94.5 (<=95) → 觸發於 95
        bar = _bar(TS2, o=98.0, h=99.0, l=94.5, c=96.0)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is not None
        assert trade.exit_price == 95.0
        assert trade.exit_reason == "STOP_LOSS"
        assert not self.account.has_position()

    def test_long_stop_gap_down(self) -> None:
        # 開倉於 TS1，TS2 跳空：open=92 (<95) → 平倉於 92
        self.account.open_position(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.1)
        bar = _bar(TS2, o=92.0, h=93.0, l=91.0, c=92.5)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is not None
        assert trade.exit_price == 92.0
        assert trade.exit_reason == "STOP_LOSS_GAP"
        # gross = 2*(92-100) = -16，比於 stop 平倉的 -10 更糟
        assert trade.gross_pnl == pytest.approx(-16.0)

    def test_long_stop_not_triggered(self) -> None:
        self.account.open_position(Direction.LONG, 2.0, 100.0, TS1, 95.0, fee=0.1)
        bar = _bar(TS2, o=99.0, h=101.0, l=98.0, c=100.0)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is None
        assert self.account.has_position()

    # ---------- SHORT ----------

    def test_short_stop_triggered_intraday(self) -> None:
        self.account.open_position(Direction.SHORT, 2.0, 100.0, TS1, 105.0, fee=0.1)
        # open=102 (<105), high=106 (>=105) → 觸發於 105
        bar = _bar(TS2, o=102.0, h=106.0, l=101.0, c=104.0)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is not None
        assert trade.exit_price == 105.0
        assert trade.exit_reason == "STOP_LOSS"

    def test_short_stop_gap_up(self) -> None:
        self.account.open_position(Direction.SHORT, 2.0, 100.0, TS1, 105.0, fee=0.1)
        # 跳空：open=108 (>=105) → 平倉於 108
        bar = _bar(TS2, o=108.0, h=109.0, l=107.0, c=108.5)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is not None
        assert trade.exit_price == 108.0
        assert trade.exit_reason == "STOP_LOSS_GAP"
        # gross = -1 * 2 * (108 - 100) = -16
        assert trade.gross_pnl == pytest.approx(-16.0)

    def test_short_stop_not_triggered(self) -> None:
        self.account.open_position(Direction.SHORT, 2.0, 100.0, TS1, 105.0, fee=0.1)
        bar = _bar(TS2, o=101.0, h=104.0, l=98.0, c=99.0)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is None

    # ---------- 同根進場 + 同根止損 ----------

    def test_same_bar_entry_and_stop_no_gap_rule(self) -> None:
        """同根進場後 bar.low 觸及止損 → 平倉於 stop_price，**不**套用跳空規則。"""
        # 模擬 engine：同一根 bar 內先成交，再 check_stop
        bar = _bar(TS1, o=100.0, h=100.5, l=94.0, c=95.5)  # low 跌穿止損
        # bar.open <= initial_stop=95.0 → 若誤套跳空規則會錯誤觸發於 bar.open=100... 但這裡 open>stop
        order = LimitOrder(Direction.LONG, limit_price=100.05, quantity=2.0, initial_stop=95.0)
        broker = self.broker
        broker.try_fill_limit(order, bar, self.account)
        # 此時 position.entry_timestamp == bar.timestamp → 跳空規則不適用
        trade = broker.check_stop(self.account, bar)
        assert trade is not None
        assert trade.exit_price == 95.0  # 不是 bar.open
        assert trade.exit_reason == "STOP_LOSS"

    def test_same_bar_entry_then_no_stop(self) -> None:
        # bar.low 沒跌穿 → 持倉繼續
        bar = _bar(TS1, o=100.0, h=101.0, l=96.0, c=100.5)
        order = LimitOrder(Direction.LONG, limit_price=100.05, quantity=2.0, initial_stop=95.0)
        self.broker.try_fill_limit(order, bar, self.account)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is None
        assert self.account.has_position()

    def test_same_bar_round_trip_charges_two_fees(self) -> None:
        bar = _bar(TS1, o=100.0, h=100.5, l=94.0, c=95.5)
        order = LimitOrder(Direction.LONG, limit_price=100.05, quantity=2.0, initial_stop=95.0)
        self.broker.try_fill_limit(order, bar, self.account)
        trade = self.broker.check_stop(self.account, bar)
        assert trade is not None
        # entry_fee = 2 * 100.05 * 0.0005 = 0.10005
        # exit_fee  = 2 * 95.0   * 0.0005 = 0.095
        assert trade.entry_fee == pytest.approx(0.10005)
        assert trade.exit_fee == pytest.approx(0.095)
        # net = gross - entry_fee - exit_fee = 2*(95-100.05) - 0.19505
        expected_net = 2 * (95.0 - 100.05) - 0.10005 - 0.095
        assert trade.net_pnl == pytest.approx(expected_net)

    # ---------- 邊界 ----------

    def test_check_stop_no_position_returns_none(self) -> None:
        bar = _bar(TS1, o=100.0, h=101.0, l=99.0, c=100.0)
        assert self.broker.check_stop(self.account, bar) is None
