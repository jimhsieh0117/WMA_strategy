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

from src.indicators.adx import adx_dmi
from src.indicators.atr import atr as compute_atr
from src.indicators.bollinger import bollinger_bands
from src.indicators.market_structure import compute_market_structure
from src.indicators.rank import percent_rank
from src.indicators.wma import wma
from src.strategy.types import EntrySignal, StrategyParams
from src.utils.exceptions import DataIntegrityError
from src.utils.types import Direction
from src.utils.validation import validate_ohlc

# 指標準備後 df 必含的欄位（策略 + 拖曳止損都會用到）
REQUIRED_INDICATOR_COLUMNS: tuple[str, ...] = (
    "wma_fast",
    "wma_slow",
    "bb_middle",
    "bb_upper",
    "bb_lower",
)

# chop_filter 啟用時額外必要的欄位
CHOP_FILTER_COLUMNS: tuple[str, ...] = (
    "chop_adx",
    "chop_bbw_rank",
    "chop_atr_rank",
)

# structure_filter 啟用時額外必要的欄位（只看 ms_trend，不需要事件欄位）
STRUCTURE_FILTER_COLUMNS: tuple[str, ...] = (
    "ms_trend",
)


def prepare_indicators(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """為原始 OHLCV 加上策略 + 拖曳止損所需的全部指標欄。

    依序計算：
    1. WMA_fast / WMA_slow（基於原始 close）            ← 進場訊號用
    2. Bollinger Band（基於原始 close，WMA 中軌、2σ）  ← Stage 3 拖曳用
    """
    validate_ohlc(df, require_volume=True)

    out = df.copy()
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

    # chop_filter 指標：與 trailing BB **完全獨立**（職責不同 → entry gate vs exit）
    if params.chop_filter.enabled:
        cf = params.chop_filter
        # chop 用自己的 BB 算 BBW（避免污染 trailing 的 stop 計算）
        chop_mid, chop_up, chop_lo = bollinger_bands(
            df["close"], period=cf.bb_period, num_std=cf.bb_num_std, ma_type="wma",
        )
        chop_bbw = (chop_up - chop_lo) / chop_mid.replace(0, pd.NA)
        out["chop_bbw_rank"] = percent_rank(chop_bbw.astype(float), cf.rank_window) * 100.0

        chop_atr = compute_atr(df, period=cf.atr_period)
        out["chop_atr_rank"] = percent_rank(chop_atr, cf.rank_window) * 100.0

        adx, _pdi, _mdi = adx_dmi(df, period=cf.adx_period)
        out["chop_adx"] = adx

    # 結構順勢濾網：算 market structure 並只保留 ms_trend（節省欄位污染）
    if params.structure_filter.enabled:
        sf = params.structure_filter
        ms_full = compute_market_structure(
            df, pivot_left=sf.pivot_left, pivot_right=sf.pivot_right,
        )
        out["ms_trend"] = ms_full["ms_trend"]
    return out


def passes_signal_filter(
    df: pd.DataFrame,
    bar_index: int,
    direction: Direction,
    params: StrategyParams,
) -> bool:
    """進場前 N 根 K 實體比例濾網。

    多單：要求 bull_metric / total_metric ≥ threshold
    空單：要求 bull_metric / total_metric ≤ (1 − threshold)

    metric 由 ``params.signal_filter.mode`` 決定：
        body_sum    → |body|
        body_sq_sum → body²
    Window 包含訊號 K 自身（bar_index − window + 1 .. bar_index）。一律用原始 K。
    """
    sf = params.signal_filter
    if sf.mode == "off":
        return True
    n = sf.window
    start = bar_index - n + 1
    if start < 0:
        return False  # 暖機不足，視同未通過

    opens = df["open"].iloc[start:bar_index + 1].to_numpy()
    closes = df["close"].iloc[start:bar_index + 1].to_numpy()

    bodies = closes - opens
    if sf.mode == "body_sum":
        bull_metric = float(bodies[bodies > 0].sum())
        bear_metric = float(-bodies[bodies < 0].sum())
    else:  # "body_sq_sum"
        bull_metric = float((bodies[bodies > 0] ** 2).sum())
        bear_metric = float((bodies[bodies < 0] ** 2).sum())
    total = bull_metric + bear_metric
    if total <= 0:
        return False  # 全部 doji，視同無方向訊號

    bull_ratio = bull_metric / total
    if direction is Direction.LONG:
        return bull_ratio >= sf.threshold
    return bull_ratio <= (1.0 - sf.threshold)


def passes_chop_filter(
    df: pd.DataFrame, bar_index: int, params: StrategyParams,
) -> bool:
    """盤整濾網：BBW_rank、ATR_rank、ADX 皆達門檻才放行。

    暖機未滿（任一欄為 NaN）→ 不放行（保守處理）。
    """
    cf = params.chop_filter
    if not cf.enabled:
        return True
    try:
        bbw_r = df["chop_bbw_rank"].iat[bar_index]
        atr_r = df["chop_atr_rank"].iat[bar_index]
        adx_v = df["chop_adx"].iat[bar_index]
    except KeyError:
        # 防呆：chop_filter 開啟但 prepare_indicators 沒生欄位
        raise DataIntegrityError(
            "chop_filter enabled but indicator columns missing; "
            "did you forget to call prepare_indicators() with chop_filter enabled?"
        )
    if pd.isna(bbw_r) or pd.isna(atr_r) or pd.isna(adx_v):
        return False
    return (
        bbw_r >= cf.bbw_rank_min
        and atr_r >= cf.atr_rank_min
        and adx_v >= cf.adx_min
    )


def passes_structure_filter(
    df: pd.DataFrame, bar_index: int, direction: Direction, params: StrategyParams,
) -> bool:
    """結構順勢濾網：依 ms_trend 判定是否允許進場。

    - ``mode="aligned"``：嚴格——long 要求 trend=='bull'，short 要求 trend=='bear'；
      none（暖機/未確認）一律擋。
    - ``mode="exclude_counter"``：只擋反向；aligned + none 都放行。

    ``enabled=False`` 直接放行（不檢查欄位是否存在）。
    """
    sf = params.structure_filter
    if not sf.enabled:
        return True
    try:
        trend = df["ms_trend"].iat[bar_index]
    except KeyError:
        raise DataIntegrityError(
            "structure_filter enabled but 'ms_trend' column missing; "
            "did you forget to call prepare_indicators() with structure_filter enabled?"
        )
    # 規範化：NaN / pd.NA → 空字串
    if not isinstance(trend, str):
        trend = ""

    aligned_trend = "bull" if direction is Direction.LONG else "bear"
    counter_trend = "bear" if direction is Direction.LONG else "bull"

    if sf.mode == "aligned":
        return trend == aligned_trend
    # exclude_counter
    return trend != counter_trend


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
