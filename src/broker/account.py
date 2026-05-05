"""帳戶狀態管理（cash / positions / trade log / equity history）。

設計原則：
- 1x 永續合約模型：開倉**不**扣 notional，只扣手續費；equity = cash + 未實現 PnL 總和
- 內部以 ``dict[position_id, Position]`` 儲存，可持有 0..N 筆並行倉位
- 任何狀態變動皆做不變式檢查（CLAUDE.md §5），違反即 raise ``AccountInvariantError``

API 風格：
- 「單倉」舊 API（``open_position`` / ``close_position`` / ``update_stop`` / ``position``）
  在 ≤ 1 筆持倉時行為與舊版完全一致；遇到 > 1 筆時 raise，避免歧義
- 「多倉」新 API（``open_position_multi`` / ``close_position_by_id`` /
  ``update_stop_by_id`` / ``positions``）由 engine 在 ``allow_pyramiding=True``
  時使用
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
        self._positions: dict[int, Position] = {}
        self._next_position_id: int = 1
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
        """單倉舊 API：0 筆 → None；1 筆 → 該 Position；>1 筆 → raise（呼叫方應改用 ``positions``）。"""
        n = len(self._positions)
        if n == 0:
            return None
        if n == 1:
            return next(iter(self._positions.values()))
        raise AccountInvariantError(
            f"[{self._name}] account holds {n} positions; use `positions` instead of `position`"
        )

    @property
    def positions(self) -> dict[int, Position]:
        """多倉 API：回傳 dict 副本（避免外部誤改）。"""
        return dict(self._positions)

    @property
    def position_count(self) -> int:
        return len(self._positions)

    @property
    def total_notional_at_entry(self) -> float:
        """所有未平倉的 entry 期 notional 加總，做槓桿檢查用。"""
        return sum(p.notional_at_entry for p in self._positions.values())

    @property
    def trade_log(self) -> list[Trade]:
        return list(self._trade_log)

    @property
    def equity_history(self) -> list[tuple[pd.Timestamp, float]]:
        return list(self._equity_history)

    def has_position(self) -> bool:
        return len(self._positions) > 0

    def equity(self, mark_price: float) -> float:
        """以 ``mark_price`` mark-to-market 計算當下 equity（cash + 全部未實現 PnL）。"""
        unrealized = sum(p.unrealized_pnl(mark_price) for p in self._positions.values())
        return self._cash + unrealized

    def position_by_id(self, position_id: int) -> Position:
        if position_id not in self._positions:
            raise AccountInvariantError(
                f"[{self._name}] no open position with id={position_id}"
            )
        return self._positions[position_id]

    # ---------- state mutations ----------

    def open_position(
        self,
        direction: Direction,
        quantity: float,
        entry_price: float,
        entry_timestamp: pd.Timestamp,
        stop_price: float,
        fee: float,
    ) -> int:
        """單倉舊 API：已有任何持倉時 raise，否則開倉。

        Returns:
            新 position_id。
        """
        if self._positions:
            existing = next(iter(self._positions.values()))
            raise AccountInvariantError(
                f"[{self._name}] cannot open: already has {existing.direction} position"
            )
        return self._open(
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            stop_price=stop_price,
            fee=fee,
        )

    def open_position_multi(
        self,
        direction: Direction,
        quantity: float,
        entry_price: float,
        entry_timestamp: pd.Timestamp,
        stop_price: float,
        fee: float,
    ) -> int:
        """多倉新 API：不檢查既有持倉，直接新增一筆，回傳新 position_id。"""
        return self._open(
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=entry_timestamp,
            stop_price=stop_price,
            fee=fee,
        )

    def _open(
        self,
        *,
        direction: Direction,
        quantity: float,
        entry_price: float,
        entry_timestamp: pd.Timestamp,
        stop_price: float,
        fee: float,
    ) -> int:
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

        pid = self._next_position_id
        self._next_position_id += 1
        self._cash = new_cash
        self._positions[pid] = Position(
            direction=direction,
            quantity=float(quantity),
            entry_price=float(entry_price),
            entry_timestamp=entry_timestamp,
            stop_price=float(stop_price),
            entry_fee=float(fee),
            stop_history=[(entry_timestamp, float(stop_price))],
            position_id=pid,
        )
        return pid

    def close_position(
        self,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        fee: float,
        reason: str,
    ) -> Trade:
        """單倉舊 API：恰好 1 筆持倉時平倉；0 / >1 時 raise。"""
        n = len(self._positions)
        if n == 0:
            raise AccountInvariantError(f"[{self._name}] cannot close: no position")
        if n > 1:
            raise AccountInvariantError(
                f"[{self._name}] cannot close: account holds {n} positions; "
                "use close_position_by_id"
            )
        pid = next(iter(self._positions.keys()))
        return self.close_position_by_id(pid, exit_price, exit_timestamp, fee, reason)

    def close_position_by_id(
        self,
        position_id: int,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        fee: float,
        reason: str,
    ) -> Trade:
        """多倉新 API：依 ``position_id`` 平倉並產生 ``Trade`` 紀錄。

        Cash 變化：``cash += gross_pnl - exit_fee``。
        （entry_fee 已在開倉時扣除，此處不重複處理。）
        """
        if exit_price <= 0:
            raise AccountInvariantError(f"exit_price must be > 0, got {exit_price}")
        if fee < 0:
            raise AccountInvariantError(f"fee must be >= 0, got {fee}")
        if position_id not in self._positions:
            raise AccountInvariantError(
                f"[{self._name}] cannot close: no open position with id={position_id}"
            )

        pos = self._positions[position_id]
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
            stop_history=tuple(pos.stop_history),
            position_id=pos.position_id,
        )
        self._trade_log.append(trade)
        del self._positions[position_id]
        return trade

    def update_stop(
        self,
        new_stop: float,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """單倉舊 API：恰好 1 筆持倉時更新 stop；0 / >1 時 raise。"""
        n = len(self._positions)
        if n == 0:
            raise AccountInvariantError(
                f"[{self._name}] cannot update stop: no position"
            )
        if n > 1:
            raise AccountInvariantError(
                f"[{self._name}] cannot update stop: account holds {n} positions; "
                "use update_stop_by_id"
            )
        pid = next(iter(self._positions.keys()))
        self.update_stop_by_id(pid, new_stop, timestamp)

    def update_stop_by_id(
        self,
        position_id: int,
        new_stop: float,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        """多倉新 API：覆寫指定持倉的 stop_price。

        ratchet 規則由 engine 決定後才呼叫；本方法只驗 ``new_stop > 0``。
        若提供 ``timestamp``，同步寫入該倉位的 ``stop_history``（圖表用）。
        """
        if new_stop <= 0:
            raise AccountInvariantError(f"new_stop must be > 0, got {new_stop}")
        if position_id not in self._positions:
            raise AccountInvariantError(
                f"[{self._name}] cannot update stop: no position with id={position_id}"
            )
        pos = self._positions[position_id]
        pos.stop_price = float(new_stop)
        if timestamp is not None:
            pos.stop_history.append((timestamp, float(new_stop)))

    def snapshot_equity(self, mark_price: float, timestamp: pd.Timestamp) -> None:
        """記錄當下 equity 至歷史，供日後績效計算使用。"""
        self._equity_history.append((timestamp, self.equity(mark_price)))
