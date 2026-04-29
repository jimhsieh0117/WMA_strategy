"""BaseTrendStrategy ABC 與指標準備工具。

關鍵設計：
- 策略**不持有持倉狀態**，是純函式：(df, bar_index) → Signal | None
- ratchet（止損只能往有利方向移動）由 engine 實作，策略只回報「候選值」
- 所有指標欄位由 ``prepare_indicators`` 預先算好；策略不重複算

對應 ARCHITECTURE.md §3.3。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from src.indicators.atr import atr
from src.indicators.heikin_ashi import compute_ha
from src.indicators.wma import wma
from src.strategy.types import EntrySignal, StrategyParams
from src.utils.exceptions import DataIntegrityError
from src.utils.types import Direction
from src.utils.validation import validate_ohlc

# 指標準備後 df 必含的欄位
REQUIRED_INDICATOR_COLUMNS: tuple[str, ...] = (
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",
    "ha_wma_fast",
    "ha_wma_slow",
    "atr",
)


def prepare_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """為原始 OHLCV 加上策略所需的全部指標欄。

    依序計算：
    1. Heikin-Ashi（ha_open / ha_high / ha_low / ha_close）
    2. HA_WMA_fast / HA_WMA_slow（基於 ha_close）
    3. ATR（基於原始 K 線）

    Args:
        df: 原始 OHLCV，DatetimeIndex + 小寫欄位。
        params: 策略參數。

    Returns:
        新 DataFrame，含原始欄 + 全部指標欄。
    """
    validate_ohlc(df, require_volume=True)

    out = compute_ha(df)
    out["ha_wma_fast"] = wma(out["ha_close"], params.wma_fast)
    out["ha_wma_slow"] = wma(out["ha_close"], params.wma_slow)
    out["atr"] = atr(df, params.atr_period)
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

    子類別僅需實作三件事：
    1. ``direction``: 類別變數，標明 LONG 或 SHORT
    2. ``detect_entry``: 在 bar[bar_index] 收盤後判斷是否該進場
    3. ``compute_trailing_stop_candidate``: 該 bar 收盤後的候選止損價

    狀態完全外部化：策略物件只持有 params，不知道目前是否有持倉。
    這讓策略可被 WFA / Monte Carlo 多次重複呼叫而不必重置。
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
        """

    @abstractmethod
    def compute_trailing_stop_candidate(
        self, df: pd.DataFrame, bar_index: int
    ) -> float:
        """在 bar[bar_index] 收盤後計算「**候選**」止損價。

        - LONG: ``highest_high(N) - ATR × multiplier``
        - SHORT: ``lowest_low(N) + ATR × multiplier``

        是否實際更新由 engine 決定（ratchet 規則：只往有利方向移動）。
        若資料不足回傳 ``float('nan')``。
        """
