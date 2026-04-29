"""Broker 模組的資料型別。

設計原則：
- ``Bar``、``LimitOrder``、``FillResult``、``Trade``、``BrokerConfig`` 一律 frozen
- ``Position`` 為 mutable（``stop_price`` 會被 ratchet 更新）
- 所有 PnL / equity 計算的方向處理用 ``Direction.sign`` 統一管理
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.exceptions import ConfigError
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Bar — engine 在每根 K 線傳給 broker 的最小資料容器
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Bar:
    """單根 K 線的不可變快照。"""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @classmethod
    def from_row(cls, timestamp: pd.Timestamp, row: pd.Series) -> "Bar":
        return cls(
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
        )


# --------------------------------------------------------------------------- #
# BrokerConfig — 手續費與滑點設定
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class BrokerConfig:
    """Binance 永續合約 USDT-M 的成本設定。

    依 CLAUDE.md §11 預設 VIP 0：
    - taker 0.05%、maker 0.02%
    - 限價含滑點 0.03% → 即時吃單 → 採用 taker 費率
    """

    taker_fee_rate: float = 0.0005
    maker_fee_rate: float = 0.0002
    slippage_pct: float = 0.0003

    def __post_init__(self) -> None:
        for name, val in (
            ("taker_fee_rate", self.taker_fee_rate),
            ("maker_fee_rate", self.maker_fee_rate),
            ("slippage_pct", self.slippage_pct),
        ):
            if val < 0 or val > 0.1:
                raise ConfigError(
                    f"{name} must be in [0, 0.1], got {val}"
                )


# --------------------------------------------------------------------------- #
# LimitOrder / FillResult
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LimitOrder:
    """由 engine 在「下根 K 線開盤」前構造的限價單。

    Attributes:
        direction: 方向。
        limit_price: 限價（Long 為上限、Short 為下限）。
        quantity: 標的數量（如 ETH 顆數），必須 > 0。
        initial_stop: 成交後立即套用的止損價。
    """

    direction: Direction
    limit_price: float
    quantity: float
    initial_stop: float

    def __post_init__(self) -> None:
        if self.limit_price <= 0:
            raise ConfigError(f"limit_price must be > 0, got {self.limit_price}")
        if self.quantity <= 0:
            raise ConfigError(f"quantity must be > 0, got {self.quantity}")
        if self.initial_stop <= 0:
            raise ConfigError(f"initial_stop must be > 0, got {self.initial_stop}")
        # 方向 vs 止損相對位置在 Account.open_position 再驗一次（關注運行時）


@dataclass(frozen=True)
class FillResult:
    """``BrokerSimulator.try_fill_limit`` 的回報。"""

    filled: bool
    fill_price: float | None = None
    fee: float = 0.0
    reason: str = ""


# --------------------------------------------------------------------------- #
# Position（mutable）
# --------------------------------------------------------------------------- #

@dataclass
class Position:
    """當前持倉。``stop_price`` 為 mutable：engine 在每根 K 線收盤後可呼叫
    ``Account.update_stop`` 進行 ratchet 更新。
    """

    direction: Direction
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_price: float
    entry_fee: float

    @property
    def notional_at_entry(self) -> float:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """以 ``mark_price`` 計算未實現損益（含方向）。"""
        return self.direction.sign * self.quantity * (mark_price - self.entry_price)


# --------------------------------------------------------------------------- #
# Trade（frozen，平倉後永久紀錄）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Trade:
    """完整的一筆來回交易（已平倉）。"""

    direction: Direction
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    exit_price: float
    exit_timestamp: pd.Timestamp
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    exit_reason: str

    @property
    def holding_duration(self) -> pd.Timedelta:
        return self.exit_timestamp - self.entry_timestamp
