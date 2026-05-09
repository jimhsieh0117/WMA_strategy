"""Broker 模組的資料型別。

設計原則：
- ``Bar``、``LimitOrder``、``FillResult``、``Trade``、``BrokerConfig`` 一律 frozen
- ``Position`` 為 mutable（``stop_price`` 會被 ratchet 更新）
- 所有 PnL / equity 計算的方向處理用 ``Direction.sign`` 統一管理
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        quantity: 標的數量（如 ETH 顆數），engine 在 limit_price 上預估的值；
            cap 後 broker 會依 ``target_risk_usdt`` 或維持 notional 重算。
        initial_stop: 成交後立即套用的止損價。
        target_risk_usdt: 若非 None，broker 在 fill_price 確定後依風險公式
            重算 quantity（`risk / [|fill - stop| + (fill + stop) × taker]`），
            確保「撞 stop 時固定虧損」。預設 None 沿用舊版 notional 維持邏輯。
    """

    direction: Direction
    limit_price: float
    quantity: float
    initial_stop: float
    target_risk_usdt: float | None = None

    def __post_init__(self) -> None:
        if self.limit_price <= 0:
            raise ConfigError(f"limit_price must be > 0, got {self.limit_price}")
        if self.quantity <= 0:
            raise ConfigError(f"quantity must be > 0, got {self.quantity}")
        if self.initial_stop <= 0:
            raise ConfigError(f"initial_stop must be > 0, got {self.initial_stop}")
        if self.target_risk_usdt is not None and self.target_risk_usdt <= 0:
            raise ConfigError(
                f"target_risk_usdt must be > 0, got {self.target_risk_usdt}"
            )
        # 方向 vs 止損相對位置在 Account.open_position 再驗一次（關注運行時）


@dataclass(frozen=True)
class FillResult:
    """``BrokerSimulator.try_fill_limit`` 的回報。"""

    filled: bool
    fill_price: float | None = None
    fee: float = 0.0
    reason: str = ""
    position_id: int | None = None


# --------------------------------------------------------------------------- #
# Position（mutable）
# --------------------------------------------------------------------------- #

@dataclass
class Position:
    """當前持倉。``stop_price`` 為 mutable：engine 在每根 K 線收盤後可呼叫
    ``Account.update_stop`` 進行 ratchet 更新。

    ``stop_history`` 記錄止損的每次變動（含初始值），用於圖表呈現
    Stage 1→2→3 跳躍與 Stage 3 ratchet 的軌跡。
    """

    direction: Direction
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    stop_price: float
    entry_fee: float
    stop_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    position_id: int = 0

    @property
    def notional_at_entry(self) -> float:
        return self.quantity * self.entry_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """以 ``mark_price`` 計算未實現損益（含方向）。"""
        return self.direction.sign * self.quantity * (mark_price - self.entry_price)

    # ---- 序列化（給 live_sim resume 用，不影響既有邏輯）----

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "stop_price": self.stop_price,
            "entry_fee": self.entry_fee,
            "stop_history": [(ts.isoformat(), float(v)) for ts, v in self.stop_history],
            "position_id": self.position_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(
            direction=Direction(data["direction"]),
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            entry_timestamp=pd.Timestamp(data["entry_timestamp"]),
            stop_price=float(data["stop_price"]),
            entry_fee=float(data["entry_fee"]),
            stop_history=[(pd.Timestamp(ts), float(v)) for ts, v in data["stop_history"]],
            position_id=int(data["position_id"]),
        )


# --------------------------------------------------------------------------- #
# Trade（frozen，平倉後永久紀錄）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Trade:
    """完整的一筆來回交易（已平倉）。

    ``stop_history`` 為這筆交易期間止損每次變動的時間序列，含初始值；
    用於視覺化呈現三階段 stop 軌跡。
    """

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
    stop_history: tuple[tuple[pd.Timestamp, float], ...] = ()
    position_id: int = 0
    # 平倉時 trailing controller 所處的最終 stage（1/2/3）。
    # 預設 1 對應「從未進到 stage 2（initial swing stop 直接被打到）」。
    final_stage: int = 1
    # 平倉時的歷史最大有利進度（以 R 為單位）。
    peak_progress_r: float = 0.0

    @property
    def holding_duration(self) -> pd.Timedelta:
        return self.exit_timestamp - self.entry_timestamp

    # ---- 序列化（給 live_sim resume 用，不影響既有邏輯）----

    def to_dict(self) -> dict:
        return {
            "direction": self.direction.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "exit_price": self.exit_price,
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "entry_fee": self.entry_fee,
            "exit_fee": self.exit_fee,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "stop_history": [(ts.isoformat(), float(v)) for ts, v in self.stop_history],
            "position_id": self.position_id,
            "final_stage": self.final_stage,
            "peak_progress_r": self.peak_progress_r,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trade":
        return cls(
            direction=Direction(data["direction"]),
            quantity=float(data["quantity"]),
            entry_price=float(data["entry_price"]),
            entry_timestamp=pd.Timestamp(data["entry_timestamp"]),
            exit_price=float(data["exit_price"]),
            exit_timestamp=pd.Timestamp(data["exit_timestamp"]),
            entry_fee=float(data["entry_fee"]),
            exit_fee=float(data["exit_fee"]),
            gross_pnl=float(data["gross_pnl"]),
            net_pnl=float(data["net_pnl"]),
            return_pct=float(data["return_pct"]),
            exit_reason=str(data["exit_reason"]),
            stop_history=tuple(
                (pd.Timestamp(ts), float(v)) for ts, v in data["stop_history"]
            ),
            position_id=int(data["position_id"]),
            final_stage=int(data.get("final_stage", 1)),
            peak_progress_r=float(data.get("peak_progress_r", 0.0)),
        )
