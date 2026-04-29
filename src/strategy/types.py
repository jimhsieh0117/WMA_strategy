"""Strategy 模組的資料型別。

設計原則：
- ``StrategyParams``：不可變設定（``frozen=True``），便於 cache、安全傳遞
- ``EntrySignal``：策略的純輸出，不含 broker / account 概念
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.exceptions import ConfigError
from src.utils.types import Direction


@dataclass(frozen=True)
class StrategyParams:
    """策略可調參數，對應 configs/default.yaml 的 ``strategy`` 區塊。

    建立時呼叫 ``validate()`` 確保值合法。
    """

    wma_fast: int = 2
    wma_slow: int = 4
    atr_period: int = 14
    atr_multiplier: float = 2.0
    atr_lookback: int = 14

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.wma_fast < 1 or self.wma_slow < 1:
            raise ConfigError(
                f"WMA periods must be >= 1, got fast={self.wma_fast}, slow={self.wma_slow}"
            )
        if self.wma_fast >= self.wma_slow:
            raise ConfigError(
                f"wma_fast ({self.wma_fast}) must be < wma_slow ({self.wma_slow})"
            )
        if self.atr_period < 1:
            raise ConfigError(f"atr_period must be >= 1, got {self.atr_period}")
        if self.atr_lookback < 1:
            raise ConfigError(f"atr_lookback must be >= 1, got {self.atr_lookback}")
        if self.atr_multiplier <= 0:
            raise ConfigError(
                f"atr_multiplier must be > 0, got {self.atr_multiplier}"
            )

    @property
    def warmup_bars(self) -> int:
        """暖機所需最少根數，超過此值後策略才會產生有效訊號。"""
        # WMA 暖機 wma_slow - 1，ATR 暖機 atr_period - 1，
        # 進場條件還需要回看 bar[t-3] → 整體取最大 + 3
        return max(self.wma_slow, self.atr_period, self.atr_lookback) + 3


@dataclass(frozen=True)
class EntrySignal:
    """策略在 bar[bar_index] 收盤後判斷應進場時產生。

    Attributes:
        direction: 多 / 空。
        bar_index: 訊號產生的 K 線 index（df.iloc[bar_index] 即訊號 bar）。
        timestamp: 訊號 bar 的時間戳，即 ``df.index[bar_index]``。
        initial_stop: 進場時的初始止損價（用 bar[bar_index] 收盤後的資訊算出）。
        reason: debug / log 訊息。
    """

    direction: Direction
    bar_index: int
    timestamp: pd.Timestamp
    initial_stop: float
    reason: str
