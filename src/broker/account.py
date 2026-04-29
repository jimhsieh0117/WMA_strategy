"""帳戶狀態管理（cash / position / trade log / equity history）。

設計原則：
- 1x 永續合約模型：開倉**不**扣 notional，只扣手續費；equity = cash + 未實現 PnL
- 任何狀態變動皆做不變式檢查（CLAUDE.md §5），違反即 raise ``AccountInvariantError``
- 對外暴露唯讀 property（trade_log / equity_history 回傳副本，避免外部誤改）
"""

from __future__ import annotations

import pandas as pd

from src.broker.types import Position, Trade
from src.utils.exceptions import AccountInvariantError
from src.utils.types import Direction


class Account:
    """單帳戶模型。多空策略各自持一個 ``Account`` 實例。"""

    def __init__(self, initial_capital: float, name: str = "default") -> None:
        if initial_capital <= 0:
            raise AccountInvariantError(
                f"initial_capital must be > 0, got {initial_capital}"
            )
        self._name = name
        self._initial_capital = float(initial_capital)
        self._cash = float(initial_capital)
        self._position: Position | None = None
        self._trade_log: list[Trade] = []
        self._equity_history: list[tuple[pd.Timestamp, float]] = []

    # ---------- properties ----------

    @property
    def name(self) -> str:
        return self._name

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def position(self) -> Position | None:
        return self._position

    @property
    def trade_log(self) -> list[Trade]:
        return list(self._trade_log)

    @property
    def equity_history(self) -> list[tuple[pd.Timestamp, float]]:
        return list(self._equity_history)

    def has_position(self) -> bool:
        return self._position is not None

    def equity(self, mark_price: float) -> float:
        """以 ``mark_price`` mark-to-market 計算當下 equity。"""
        if self._position is None:
            return self._cash
        return self._cash + self._position.unrealized_pnl(mark_price)

    # ---------- state mutations ----------

    def open_position(
        self,
        direction: Direction,
        quantity: float,
        entry_price: float,
        entry_timestamp: pd.Timestamp,
        stop_price: float,
        fee: float,
    ) -> None:
        """開倉並從 cash 扣除手續費。

        Raises:
            AccountInvariantError: 已有持倉、quantity / entry_price / fee 非法、
                                   stop 與 direction 方向不一致、扣除手續費後 cash < 0。
        """
        if self._position is not None:
            raise AccountInvariantError(
                f"[{self._name}] cannot open: already has {self._position.direction} position"
            )
        if quantity <= 0:
            raise AccountInvariantError(f"quantity must be > 0, got {quantity}")
        if entry_price <= 0:
            raise AccountInvariantError(f"entry_price must be > 0, got {entry_price}")
        if fee < 0:
            raise AccountInvariantError(f"fee must be >= 0, got {fee}")

        if direction is Direction.LONG and stop_price >= entry_price:
            raise AccountInvariantError(
                f"long stop_price ({stop_price}) must be < entry_price ({entry_price})"
            )
        if direction is Direction.SHORT and stop_price <= entry_price:
            raise AccountInvariantError(
                f"short stop_price ({stop_price}) must be > entry_price ({entry_price})"
            )

        new_cash = self._cash - fee
        if new_cash < 0:
            raise AccountInvariantError(
                f"cash would go negative after fee: {self._cash} - {fee} = {new_cash}"
            )

        self._cash = new_cash
        self._position = Position(
            direction=direction,
            quantity=float(quantity),
            entry_price=float(entry_price),
            entry_timestamp=entry_timestamp,
            stop_price=float(stop_price),
            entry_fee=float(fee),
        )

    def close_position(
        self,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        fee: float,
        reason: str,
    ) -> Trade:
        """平倉並結算 PnL，產生不可變的 ``Trade`` 紀錄。

        Cash 變化：``cash += gross_pnl - exit_fee``。
        （entry_fee 已在 ``open_position`` 時扣除，此處不重複處理。）
        """
        if self._position is None:
            raise AccountInvariantError(f"[{self._name}] cannot close: no position")
        if exit_price <= 0:
            raise AccountInvariantError(f"exit_price must be > 0, got {exit_price}")
        if fee < 0:
            raise AccountInvariantError(f"fee must be >= 0, got {fee}")

        pos = self._position
        gross_pnl = pos.unrealized_pnl(exit_price)
        net_pnl = gross_pnl - pos.entry_fee - fee
        notional = pos.notional_at_entry
        return_pct = net_pnl / notional if notional > 0 else 0.0

        self._cash += gross_pnl - fee

        trade = Trade(
            direction=pos.direction,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            entry_timestamp=pos.entry_timestamp,
            exit_price=float(exit_price),
            exit_timestamp=exit_timestamp,
            entry_fee=pos.entry_fee,
            exit_fee=float(fee),
            gross_pnl=float(gross_pnl),
            net_pnl=float(net_pnl),
            return_pct=float(return_pct),
            exit_reason=reason,
        )
        self._trade_log.append(trade)
        self._position = None
        return trade

    def update_stop(self, new_stop: float) -> None:
        """直接覆寫止損價。

        ratchet 規則（多單只能上移、空單只能下移）由 engine 判斷後才呼叫此方法。
        本方法只檢查 ``new_stop > 0``，不檢查與 entry_price 相對位置——
        因為拖曳止損可能移入利潤區（multi-bar 持有時 stop 可超越 entry），
        該情境合法。
        """
        if self._position is None:
            raise AccountInvariantError(
                f"[{self._name}] cannot update stop: no position"
            )
        if new_stop <= 0:
            raise AccountInvariantError(f"new_stop must be > 0, got {new_stop}")
        self._position.stop_price = float(new_stop)

    def snapshot_equity(self, mark_price: float, timestamp: pd.Timestamp) -> None:
        """記錄當下 equity 至歷史，供日後績效計算使用。"""
        self._equity_history.append((timestamp, self.equity(mark_price)))
