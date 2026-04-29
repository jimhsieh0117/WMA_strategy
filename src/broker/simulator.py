"""撮合與止損模擬。

對 engine 暴露兩個操作：
1. ``try_fill_limit(order, bar, account)``：嘗試在當根 K 線撮合限價單
2. ``check_stop(account, bar)``：檢查當根 K 線是否觸發止損，若是則平倉

設計原則：
- 純粹的撮合 / 止損規則模擬，不知道策略邏輯
- 所有成交價落點皆驗證在 ``[bar.low, bar.high]`` 內，違反 raise ``OrderExecutionError``
- 同根進場 + 同根止損的處理方式：``Account.open_position`` + ``check_stop`` 各扣一次手續費

未來實盤 broker（``BinanceLiveBroker`` 等）只要實作相同方法簽名即可替換 ``BrokerSimulator``，
engine 不需要修改（CLAUDE.md §10 預留接口）。
"""

from __future__ import annotations

from src.broker.account import Account
from src.broker.types import Bar, BrokerConfig, FillResult, LimitOrder, Trade
from src.utils.exceptions import OrderExecutionError
from src.utils.types import Direction


class BrokerSimulator:
    """限價成交與止損的撮合引擎。"""

    def __init__(self, config: BrokerConfig) -> None:
        self.config = config

    # ----------------------------------------------------------------------- #
    # 限價單撮合
    # ----------------------------------------------------------------------- #

    def try_fill_limit(
        self,
        order: LimitOrder,
        bar: Bar,
        account: Account,
    ) -> FillResult:
        """嘗試在 ``bar`` 撮合 ``order``。

        撮合規則：
        - LONG: ``bar.low <= limit_price`` → 成交於 ``limit_price``
        - SHORT: ``bar.high >= limit_price`` → 成交於 ``limit_price``
        - 否則作廢，不延期到下根 K 線

        Returns:
            FillResult。``filled=True`` 時 account 已開倉、cash 已扣 fee。

        Raises:
            OrderExecutionError: account 已有持倉、成交價超出 bar 範圍。
        """
        if account.has_position():
            raise OrderExecutionError(
                f"cannot fill limit: account already has "
                f"{account.position.direction} position"
            )

        if order.direction is Direction.LONG:
            if bar.low > order.limit_price:
                return FillResult(
                    filled=False,
                    reason=(
                        f"long limit {order.limit_price:.6f} below bar.low {bar.low:.6f}"
                    ),
                )
            fill_price = order.limit_price
        else:  # SHORT
            if bar.high < order.limit_price:
                return FillResult(
                    filled=False,
                    reason=(
                        f"short limit {order.limit_price:.6f} above bar.high {bar.high:.6f}"
                    ),
                )
            fill_price = order.limit_price

        # 不變式：成交價必須在 K 線範圍內
        if not (bar.low <= fill_price <= bar.high):
            raise OrderExecutionError(
                f"fill_price {fill_price} outside bar range [{bar.low}, {bar.high}]"
            )

        notional = order.quantity * fill_price
        fee = notional * self.config.taker_fee_rate

        account.open_position(
            direction=order.direction,
            quantity=order.quantity,
            entry_price=fill_price,
            entry_timestamp=bar.timestamp,
            stop_price=order.initial_stop,
            fee=fee,
        )

        return FillResult(
            filled=True,
            fill_price=fill_price,
            fee=fee,
            reason=f"filled at {fill_price:.6f}",
        )

    # ----------------------------------------------------------------------- #
    # 止損觸發檢查
    # ----------------------------------------------------------------------- #

    def check_stop(self, account: Account, bar: Bar) -> Trade | None:
        """檢查 ``bar`` 是否觸發目前持倉的止損。

        多單：
        - 帶倉跨 bar 且 ``bar.open <= stop`` → 跳空 → 平倉於 ``bar.open``
        - 否則若 ``bar.low <= stop`` → 平倉於 ``stop``

        空單為鏡像。

        同根進場：``position.entry_timestamp == bar.timestamp`` → 不適用跳空規則
        （我們是在當根開盤後進場的，不存在 bar 之前的價格資訊）。

        Returns:
            Trade（已關倉），若未觸發則 None。
        """
        if not account.has_position():
            return None

        pos = account.position
        assert pos is not None  # mypy hint
        is_carry_over = pos.entry_timestamp != bar.timestamp

        fill_price: float | None = None
        reason: str = ""

        if pos.direction is Direction.LONG:
            if is_carry_over and bar.open <= pos.stop_price:
                fill_price = bar.open
                reason = "STOP_LOSS_GAP"
            elif bar.low <= pos.stop_price:
                fill_price = pos.stop_price
                reason = "STOP_LOSS"
        else:  # SHORT
            if is_carry_over and bar.open >= pos.stop_price:
                fill_price = bar.open
                reason = "STOP_LOSS_GAP"
            elif bar.high >= pos.stop_price:
                fill_price = pos.stop_price
                reason = "STOP_LOSS"

        if fill_price is None:
            return None

        if not (bar.low <= fill_price <= bar.high):
            raise OrderExecutionError(
                f"stop fill_price {fill_price} outside bar [{bar.low}, {bar.high}]"
            )

        notional = pos.quantity * fill_price
        fee = notional * self.config.taker_fee_rate
        return account.close_position(
            exit_price=fill_price,
            exit_timestamp=bar.timestamp,
            fee=fee,
            reason=reason,
        )
