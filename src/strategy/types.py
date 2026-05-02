"""Strategy 模組的資料型別。

設計原則：
- ``StrategyParams``：不可變設定（``frozen=True``），便於 cache、安全傳遞
- ``TrailingStopParams``：三階段止損的子設定（嵌在 StrategyParams 內）
- ``EntrySignal``：策略的純輸出，不含 broker / account 概念
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.utils.exceptions import ConfigError
from src.utils.types import Direction


# 進場訊號可使用的 K 線來源
EntrySource = Literal["ha", "raw"]
VALID_ENTRY_SOURCES: tuple[str, ...] = ("ha", "raw")


# --------------------------------------------------------------------------- #
# 三階段拖曳止損設定
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrailingStopParams:
    """三階段止損的所有參數。對應 ARCHITECTURE.md §11。

    Stage 1（剛進場、初始保護）：
        多單 stop = min(low over [t-N+1..t]) × (1 − slippage_buffer)
        空單鏡像

    Stage 2（鎖利保本）：
        觸發：normal R 時 1.2R / abnormal R 時 2.4R
        新 stop = entry × (1 ± (taker×2 + slippage)) ± buffer_r × R

    Stage 3（趨勢跟蹤）：
        觸發：normal 2.4R / abnormal 4.8R
        多單 stop 跟 Bollinger lower band（WMA, 20, 2σ）
        空單跟 upper band
        在 Stage 3 中，仍與 Stage 2 fixed 取較有利者作為 floor
    """

    # ---- Stage 1 ----
    swing_lookback: int = 4              # 進場 K 線「前 N 根」的 N
    stage1_slippage_buffer: float = 0.0003  # 0.03% buffer，遠離極值方向

    # ---- Stage 2 ----
    stage2_normal_trigger_r: float = 1.2
    stage2_abnormal_trigger_r: float = 2.4
    stage2_buffer_r: float = 0.2         # 保本 stop 額外加 0.2R buffer

    # ---- Stage 3 ----
    stage3_normal_trigger_r: float = 2.4
    stage3_abnormal_trigger_r: float = 4.8
    bollinger_period: int = 20
    bollinger_num_std: float = 2.0

    def __post_init__(self) -> None:
        for name, val, low_ok in [
            ("swing_lookback", self.swing_lookback, 1),
            ("bollinger_period", self.bollinger_period, 2),
        ]:
            if not isinstance(val, int) or isinstance(val, bool) or val < low_ok:
                raise ConfigError(f"{name} must be int >= {low_ok}, got {val}")

        for name, val in [
            ("stage1_slippage_buffer", self.stage1_slippage_buffer),
            ("stage2_normal_trigger_r", self.stage2_normal_trigger_r),
            ("stage2_abnormal_trigger_r", self.stage2_abnormal_trigger_r),
            ("stage2_buffer_r", self.stage2_buffer_r),
            ("stage3_normal_trigger_r", self.stage3_normal_trigger_r),
            ("stage3_abnormal_trigger_r", self.stage3_abnormal_trigger_r),
            ("bollinger_num_std", self.bollinger_num_std),
        ]:
            if val < 0:
                raise ConfigError(f"{name} must be >= 0, got {val}")

        if self.stage3_normal_trigger_r < self.stage2_normal_trigger_r:
            raise ConfigError(
                f"stage3_normal_trigger_r ({self.stage3_normal_trigger_r}) "
                f"must be >= stage2_normal_trigger_r ({self.stage2_normal_trigger_r})"
            )
        if self.stage3_abnormal_trigger_r < self.stage2_abnormal_trigger_r:
            raise ConfigError(
                f"stage3_abnormal_trigger_r ({self.stage3_abnormal_trigger_r}) "
                f"must be >= stage2_abnormal_trigger_r ({self.stage2_abnormal_trigger_r})"
            )


# --------------------------------------------------------------------------- #
# 策略總設定
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyParams:
    """策略可調參數，對應 configs/default.yaml 的 ``strategy`` 區塊。"""

    # ---- 進場條件 ----
    wma_fast: int = 2
    wma_slow: int = 4
    entry_source: EntrySource = "ha"
    """進場訊號使用的 K 線來源：
        "ha"  = Heikin-Ashi 平均 K 線（預設、原本設計）
        "raw" = 原始 K 線（OHLC + close）
    止損計算（三階段）始終使用原始 K 線，不受此參數影響。
    """

    # ---- 拖曳止損子設定 ----
    trailing: TrailingStopParams = field(default_factory=TrailingStopParams)

    def __post_init__(self) -> None:
        if self.wma_fast < 1 or self.wma_slow < 1:
            raise ConfigError(
                f"WMA periods must be >= 1, got fast={self.wma_fast}, slow={self.wma_slow}"
            )
        if self.wma_fast >= self.wma_slow:
            raise ConfigError(
                f"wma_fast ({self.wma_fast}) must be < wma_slow ({self.wma_slow})"
            )
        if self.entry_source not in VALID_ENTRY_SOURCES:
            raise ConfigError(
                f"entry_source must be one of {VALID_ENTRY_SOURCES}, "
                f"got {self.entry_source!r}"
            )

    @property
    def warmup_bars(self) -> int:
        """暖機所需最少根數，超過此值後策略才會產生有效訊號。

        包含：WMA(slow) 暖機、Bollinger 暖機、swing lookback、進場條件回看 3 根。
        """
        return (
            max(
                self.wma_slow,
                self.trailing.bollinger_period,
                self.trailing.swing_lookback,
            )
            + 3
        )


# --------------------------------------------------------------------------- #
# 訊號
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EntrySignal:
    """策略在 bar[bar_index] 收盤後判斷應進場時產生。

    Attributes:
        direction: 多 / 空。
        bar_index: 訊號產生的 K 線 index。
        timestamp: ``df.index[bar_index]``。
        initial_stop: 進場時的初始止損價，由策略以「前 N 根 swing low/high」算出。
        reason: debug / log。
    """

    direction: Direction
    bar_index: int
    timestamp: pd.Timestamp
    initial_stop: float
    reason: str
