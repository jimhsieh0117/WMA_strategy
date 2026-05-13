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


# Signal filter modes
#   off          = 不啟用
#   body_sum     = 線性實體比例（陽K長度合 / 全部長度合）
#   body_sq_sum  = 平方加權（陽K長度² 合 / 全部長度² 合）— 大實體影響加倍
SignalFilterMode = Literal["off", "body_sum", "body_sq_sum"]
VALID_SIGNAL_FILTER_MODES: tuple[str, ...] = ("off", "body_sum", "body_sq_sum")

# Early-exit 度量模式：
#   peak     = 觀測 bar 的「極值」/ R         (多 = (high−entry)/R；空鏡像)
#   peak_pct = 觀測 bar 的「極值」/ entry      (多 = (high−entry)/entry；空鏡像)
#   close    = 觀測 bar 的「收盤」相對 entry / R (多 = (close−entry)/R；可為負)
EarlyExitMetric = Literal["peak", "peak_pct", "close"]
VALID_EARLY_EXIT_METRICS: tuple[str, ...] = ("peak", "peak_pct", "close")


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

    def __post_init__(self) -> None:
        if self.mode not in VALID_SIGNAL_FILTER_MODES:
            raise ConfigError(
                f"signal_filter.mode must be one of {VALID_SIGNAL_FILTER_MODES}, "
                f"got {self.mode!r}"
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
    stage2_buffer_r: float = 0.2
    # 額外的 %-based 觸發（OR 條件）：peak_pct = (peak − entry) / entry。
    # 0 = 關閉（與舊版一致）；>0 → stage 1→2 額外觸發於 peak_pct ≥ 此值。
    # stop 放置仍走 R-based（entry + 0.2R + 雙向 taker）。Stage 3 transition 不受影響。
    stage2_pct_trigger: float = 0.0         # 保本 stop 額外加 0.2R buffer

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

    # ---- Early-exit（進場後 N 根 K 主動 cancel）----
    # 機制：觀測期最後一根 K 收盤時若 stage 仍為 1 且該根 K 的指定度量低於門檻，
    # 該 bar.close 主動平倉（exit_reason="EARLY_CANCEL"，final_stage=1）。
    # observation_bars=1 表示「進場後 1 根 K」。
    #
    # metric:
    #   peak     = 該 bar 最大有利浮盈 / R         (多 = (high−entry)/R；空鏡像)
    #   peak_pct = 該 bar 最大有利浮盈 / entry     (多 = (high−entry)/entry；空鏡像)
    #   close    = 該 bar 收盤相對 entry / R       (多 = (close−entry)/R；可為負)
    # 切換 metric 時請設好對應的 threshold；其他 threshold 不會被使用。
    early_exit_enabled: bool = False
    early_exit_observation_bars: int = 1
    early_exit_metric: EarlyExitMetric = "peak"
    early_exit_min_peak_r: float = 0.0
    early_exit_min_peak_pct: float = 0.0
    early_exit_min_close_r: float = 0.0

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
            ("stage2_pct_trigger", self.stage2_pct_trigger),
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

        # early_exit 驗證
        if not isinstance(self.early_exit_enabled, bool):
            raise ConfigError(
                f"early_exit_enabled must be bool, got {self.early_exit_enabled!r}"
            )
        if (not isinstance(self.early_exit_observation_bars, int)
                or isinstance(self.early_exit_observation_bars, bool)
                or self.early_exit_observation_bars < 0):
            raise ConfigError(
                f"early_exit_observation_bars must be int >= 0, "
                f"got {self.early_exit_observation_bars}"
            )
        if self.early_exit_metric not in VALID_EARLY_EXIT_METRICS:
            raise ConfigError(
                f"early_exit_metric must be one of {VALID_EARLY_EXIT_METRICS}, "
                f"got {self.early_exit_metric!r}"
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
# Chop Filter：盤整 / 低波動濾網（AND 邏輯）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChopFilterParams:
    """盤整濾網。三條件 AND：BBW_rank ≥ bbw_rank_min AND ATR_rank ≥ atr_rank_min
    AND ADX ≥ adx_min。低於門檻 → 拒絕進場。

    與 trailing 的 Bollinger band **不共用**：chop_filter 是 entry gate、trailing 是
    exit 機制，職責分離；雖然 default 參數一致（period=20、num_std=2），但兩邊
    獨立 config，未來可個別實驗。

    暖機：max(rank_window, bb_period, atr_period, 2×adx_period) 根 K 線。
    rank_window=200 主導，前 ~200 根 chop_filter 必拒絕（rank=NaN）。

    研究依據（IS 2 年 stage 3 entry 分布）：stage 3 在這三個指標的分布與 stage 1
    幾乎重疊，本身不是強 alpha。baking 進策略的主因是「避開低波動 + 未來資金
    費率成本」，靠提高每筆趨勢強度間接護盤。
    """

    enabled: bool = False         # dataclass 預設關閉；yaml 顯式設 true 開啟
    bbw_rank_min: float = 40.0    # BBW 百分位（0..100）下界
    atr_rank_min: float = 40.0    # ATR 百分位（0..100）下界
    adx_min: float = 20.0         # ADX 絕對值下界
    # 指標獨立週期（與 trailing 完全分開）
    bb_period: int = 20
    bb_num_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    rank_window: int = 200        # ATR / BBW rolling percent rank 視窗

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError(f"chop_filter.enabled must be bool, got {self.enabled!r}")
        for name, val in [
            ("bbw_rank_min", self.bbw_rank_min),
            ("atr_rank_min", self.atr_rank_min),
        ]:
            if not 0 <= val <= 100:
                raise ConfigError(f"chop_filter.{name} must be in [0, 100], got {val}")
        if self.adx_min < 0:
            raise ConfigError(f"chop_filter.adx_min must be >= 0, got {self.adx_min}")
        if self.bb_num_std <= 0:
            raise ConfigError(f"chop_filter.bb_num_std must be > 0, got {self.bb_num_std}")
        for name, val, lo in [
            ("bb_period", self.bb_period, 2),
            ("atr_period", self.atr_period, 1),
            ("adx_period", self.adx_period, 1),
            ("rank_window", self.rank_window, 2),
        ]:
            if not isinstance(val, int) or isinstance(val, bool) or val < lo:
                raise ConfigError(f"chop_filter.{name} must be int >= {lo}, got {val}")


# --------------------------------------------------------------------------- #
# 策略總設定
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrategyParams:
    """策略可調參數，對應 configs/default.yaml 的 ``strategy`` 區塊。"""

    # ---- 進場條件 ----
    wma_fast: int = 2
    wma_slow: int = 4

    # ---- 拖曳止損子設定 ----
    trailing: TrailingStopParams = field(default_factory=TrailingStopParams)

    # ---- 進場訊號濾網（可選）----
    signal_filter: SignalFilterParams = field(default_factory=SignalFilterParams)

    # ---- R-cap（可選）----
    r_cap: RCapParams = field(default_factory=RCapParams)

    # ---- 盤整濾網（chop filter，AND 邏輯）----
    chop_filter: "ChopFilterParams" = field(default_factory=lambda: ChopFilterParams())

    def __post_init__(self) -> None:
        if self.wma_fast < 1 or self.wma_slow < 1:
            raise ConfigError(
                f"WMA periods must be >= 1, got fast={self.wma_fast}, slow={self.wma_slow}"
            )
        if self.wma_fast >= self.wma_slow:
            raise ConfigError(
                f"wma_fast ({self.wma_fast}) must be < wma_slow ({self.wma_slow})"
            )

    @property
    def warmup_bars(self) -> int:
        """暖機所需最少根數，超過此值後策略才會產生有效訊號。

        包含：WMA(slow) / Bollinger / swing / chop_filter 的最大暖機 + 進場回看 3 根。
        chop_filter 啟用時暖機由 rank_window 主導（預設 200）。
        """
        chop_warmup = (
            max(
                self.chop_filter.rank_window,
                self.chop_filter.bb_period,
                self.chop_filter.atr_period,
                2 * self.chop_filter.adx_period,
            )
            if self.chop_filter.enabled
            else 0
        )
        return (
            max(
                self.wma_slow,
                self.trailing.bollinger_period,
                self.trailing.swing_lookback,
                chop_warmup,
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
