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

# Signal filter modes
#   off          = 不啟用
#   body_sum     = 線性實體比例（陽K長度合 / 全部長度合）
#   body_sq_sum  = 平方加權（陽K長度² 合 / 全部長度² 合）— 大實體影響加倍
SignalFilterMode = Literal["off", "body_sum", "body_sq_sum"]
VALID_SIGNAL_FILTER_MODES: tuple[str, ...] = ("off", "body_sum", "body_sq_sum")
VALID_SIGNAL_FILTER_SOURCES: tuple[str, ...] = ("raw", "ha")


# --------------------------------------------------------------------------- #
# 訊號濾網（進場前 N 根 K 線實體比例閘門）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SignalFilterParams:
    """進場前 N 根 K 線實體比例濾網。

    對 LONG 訊號：要求 ratio ≥ threshold
    對 SHORT 訊號：要求 ratio ≤ (1 − threshold)（對稱鏡像）

    其中 ratio = bull_metric / (bull_metric + bear_metric)：
    - body_sum: metric = body length（線性）
    - body_sq_sum: metric = body length²（平方加權，超大 K 影響更大）
    """

    mode: SignalFilterMode = "off"
    window: int = 6
    threshold: float = 0.60
    source: str = "raw"  # "raw" | "ha"

    def __post_init__(self) -> None:
        if self.mode not in VALID_SIGNAL_FILTER_MODES:
            raise ConfigError(
                f"signal_filter.mode must be one of {VALID_SIGNAL_FILTER_MODES}, "
                f"got {self.mode!r}"
            )
        if self.source not in VALID_SIGNAL_FILTER_SOURCES:
            raise ConfigError(
                f"signal_filter.source must be one of {VALID_SIGNAL_FILTER_SOURCES}, "
                f"got {self.source!r}"
            )
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ConfigError(f"signal_filter.window must be int >= 1, got {self.window}")
        if not (0.0 < self.threshold < 1.0):
            raise ConfigError(
                f"signal_filter.threshold must be in (0, 1), got {self.threshold}"
            )


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

    # Stage 3 候選計算模式：
    #   "bollinger" = 追 Bollinger lower/upper（原版）
    #   "r_ladder"  = R 倍數階梯：peak 跨 (N+offset_trigger)R 後鎖到 (N+offset_stop)R
    stage3_mode: Literal["bollinger", "r_ladder"] = "bollinger"

    # r_ladder 參數（normal R）
    r_ladder_normal_first_trigger: float = 2.8   # 第一檔啟動倍數
    r_ladder_normal_step: float = 1.0            # 檔距
    # r_ladder 參數（abnormal R，倍數加倍）
    r_ladder_abnormal_first_trigger: float = 5.6
    r_ladder_abnormal_step: float = 2.0
    # 啟動倍數與鎖倉倍數的差（trigger − stop）。
    # normal=0.3 → 2.8 觸發鎖到 2.5；abnormal=0.6 → 5.6 觸發鎖到 5.0
    r_ladder_trigger_offset: float = 0.3
    r_ladder_abnormal_trigger_offset: float = 0.6

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

        if self.stage3_mode not in ("bollinger", "r_ladder"):
            raise ConfigError(
                f"stage3_mode must be 'bollinger' or 'r_ladder', got {self.stage3_mode!r}"
            )

        for name, val in [
            ("r_ladder_normal_first_trigger", self.r_ladder_normal_first_trigger),
            ("r_ladder_abnormal_first_trigger", self.r_ladder_abnormal_first_trigger),
            ("r_ladder_trigger_offset", self.r_ladder_trigger_offset),
            ("r_ladder_abnormal_trigger_offset", self.r_ladder_abnormal_trigger_offset),
        ]:
            if val <= 0:
                raise ConfigError(f"{name} must be > 0, got {val}")
        for name, val in [
            ("r_ladder_normal_step", self.r_ladder_normal_step),
            ("r_ladder_abnormal_step", self.r_ladder_abnormal_step),
        ]:
            if val <= 0:
                raise ConfigError(f"{name} must be > 0, got {val}")
        # offset 必須小於 step，否則相鄰檔的 stop 會超過下一檔的 trigger
        if self.r_ladder_trigger_offset >= self.r_ladder_normal_step:
            raise ConfigError(
                f"r_ladder_trigger_offset ({self.r_ladder_trigger_offset}) "
                f"must be < r_ladder_normal_step ({self.r_ladder_normal_step})"
            )
        if self.r_ladder_abnormal_trigger_offset >= self.r_ladder_abnormal_step:
            raise ConfigError(
                f"r_ladder_abnormal_trigger_offset ({self.r_ladder_abnormal_trigger_offset}) "
                f"must be < r_ladder_abnormal_step ({self.r_ladder_abnormal_step})"
            )


# --------------------------------------------------------------------------- #
# R-cap：用近期 trades 的平均 R 抑制單筆過大 R 的影響
# --------------------------------------------------------------------------- #

RCapMode = Literal["off", "rolling_avg"]
VALID_R_CAP_MODES: tuple[str, ...] = ("off", "rolling_avg")


@dataclass(frozen=True)
class RCapParams:
    """R-cap：以近期 trades 平均 R 作為「止盈進度單位」的上限（trigger-only）。

    機制：進場時計算過去 ``window`` 根 K 線內的歷史 trades + 未平倉持倉的初始 R 平均。
    若當筆 R（|entry − initial_stop|）大於該平均，則 controller 內部用 avg_R 作為
    progress_r 的分母 → stage 2 / stage 3 / r_ladder 的 **trigger 提前**達成。
    但 stop 放置（stage2 buffer、r_ladder offset）仍用實際 R，保留趨勢段呼吸空間。
    Stage 1 stop 位置永不變（仍由 swing 決定，1U 風險預算照舊）。

    窗口內 0 筆歷史 → 不 cap（fallback 用實際 R）。
    """

    mode: RCapMode = "off"
    window: int = 100

    def __post_init__(self) -> None:
        if self.mode not in VALID_R_CAP_MODES:
            raise ConfigError(
                f"r_cap.mode must be one of {VALID_R_CAP_MODES}, got {self.mode!r}"
            )
        if not isinstance(self.window, int) or isinstance(self.window, bool) or self.window < 1:
            raise ConfigError(f"r_cap.window must be int >= 1, got {self.window}")


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

    # ---- 進場訊號濾網（可選）----
    signal_filter: SignalFilterParams = field(default_factory=SignalFilterParams)

    # ---- R-cap（可選）----
    r_cap: RCapParams = field(default_factory=RCapParams)

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
