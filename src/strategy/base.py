"""BaseTrendStrategy ABC 與指標準備工具。

關鍵設計：
- 策略**僅負責進場訊號**，不再持有止損更新邏輯（M5+ 後改由 ``TrailingStopController``）
- 所有指標欄位由 ``prepare_indicators`` 預先算好；策略不重複算
- ``detect_entry`` 為純函式：(df, bar_index) → EntrySignal | None

對應 ARCHITECTURE.md §3.3 + §11。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.indicators.bollinger import bollinger_bands
from src.indicators.heikin_ashi import compute_ha
from src.indicators.wma import wma
from src.strategy.types import EntrySignal, StrategyParams
from src.utils.exceptions import DataIntegrityError
from src.utils.types import Direction
from src.utils.validation import validate_ohlc

# 指標準備後 df 必含的欄位（策略 + 拖曳止損都會用到）
# - HA 系列：entry_source="ha" 用
# - 原始 WMA：entry_source="raw" 用
# - Bollinger：Stage 3 拖曳止損用（不論 entry_source）
REQUIRED_INDICATOR_COLUMNS: tuple[str, ...] = (
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",
    "ha_wma_fast",
    "ha_wma_slow",
    "wma_fast",
    "wma_slow",
    "bb_middle",
    "bb_upper",
    "bb_lower",
)


def prepare_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """為原始 OHLCV 加上策略 + 拖曳止損所需的全部指標欄。

    一律算齊兩種 entry_source 用的 WMA，避免 engine 間切換時要重算指標。
    成本很低（一個額外 WMA call），可換取 config-driven 切換的彈性。

    依序計算：
    1. Heikin-Ashi（ha_open / ha_high / ha_low / ha_close）  ← entry_source="ha" 用
    2. HA_WMA_fast / HA_WMA_slow（基於 ha_close）            ← entry_source="ha" 用
    3. WMA_fast / WMA_slow（基於原始 close）                  ← entry_source="raw" 用
    4. Bollinger Band（基於原始 close，WMA 中軌、2σ）        ← Stage 3 拖曳用（永遠）
    """
    validate_ohlc(df, require_volume=True)

    out = compute_ha(df)
    # HA 路線
    out["ha_wma_fast"] = wma(out["ha_close"], params.wma_fast)
    out["ha_wma_slow"] = wma(out["ha_close"], params.wma_slow)
    # 原始 K 線路線
    out["wma_fast"] = wma(df["close"], params.wma_fast)
    out["wma_slow"] = wma(df["close"], params.wma_slow)

    bb_mid, bb_up, bb_lo = bollinger_bands(
        df["close"],
        period=params.trailing.bollinger_period,
        num_std=params.trailing.bollinger_num_std,
        ma_type="wma",
    )
    out["bb_middle"] = bb_mid
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_lo
    return out


def assert_indicators_ready(df: pd.DataFrame) -> None:
    """確認 df 已含全部指標欄。策略執行前的防呆。"""
    missing = set(REQUIRED_INDICATOR_COLUMNS) - set(df.columns)
    if missing:
        raise DataIntegrityError(
            f"strategy input missing indicator columns: {sorted(missing)}; "
            "did you forget to call prepare_indicators()?"
        )


class BaseTrendStrategy(ABC):
    """多空趨勢策略的共同基底。

    子類別僅需實作：
    - ``direction``: 類別變數（LONG / SHORT）
    - ``detect_entry``: 在 bar 收盤後判斷是否該進場，含 Stage 1 初始止損計算

    狀態完全外部化：策略物件只持有 params，不知道目前是否有持倉。
    Stage 2 / 3 拖曳止損改由 ``TrailingStopController`` 處理（per-position 實例）。
    """

    direction: Direction  # 子類別覆寫

    def __init__(self, params: StrategyParams) -> None:
        self.params = params

    @abstractmethod
    def detect_entry(
        self, df: pd.DataFrame, bar_index: int
    ) -> EntrySignal | None:
        """在 bar[bar_index] 收盤後判斷進場條件是否成立。

        實作必須遵守 look-ahead 防護：只能讀 ``df.iloc[: bar_index + 1]``。
        回傳的 EntrySignal 含 Stage 1 初始止損（swing-based）。
        """
