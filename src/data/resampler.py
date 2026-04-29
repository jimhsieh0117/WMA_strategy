"""K 線週期重採樣（1m → 1m/3m/5m/15m/30m/1H/4H）。

聚合規則（標準 OHLCV）：
    open    → first
    high    → max
    low     → min
    close   → last
    volume  → sum

對齊方式採用 pandas 預設 ``closed='left', label='left'``：
標籤為時段起點（例：5m K 標 12:00 → 涵蓋 [12:00, 12:05) 的 1m 資料）。
與 Binance / TradingView 的 K 線顯示慣例一致。

不完整的最後一段（資料未涵蓋完整週期）→ 直接丟棄，避免下游用到不完整 bar。
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from src.utils.exceptions import ConfigError, DataIntegrityError
from src.utils.validation import validate_ohlc

# pandas resample 規則對照（key = 使用者輸入；value = pandas freq alias）
SUPPORTED_TIMEFRAMES: Final[dict[str, str]] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1H": "1h",
    "4H": "4h",
}

# 對應每根 K 線的 timedelta（用於丟棄未涵蓋完整週期的尾段）
TIMEFRAME_DURATION: Final[dict[str, pd.Timedelta]] = {
    tf: pd.Timedelta(alias) for tf, alias in SUPPORTED_TIMEFRAMES.items()
}


def resample(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """將 1m OHLCV 聚合為指定週期。

    Args:
        df_1m: 來源 1m K 線（DatetimeIndex + 小寫 OHLCV 欄位）。
        timeframe: 目標週期，必須屬於 ``SUPPORTED_TIMEFRAMES``。

    Returns:
        重採樣後的 DataFrame，欄位與輸入一致；最後一段若不完整則丟棄。

    Raises:
        ConfigError: ``timeframe`` 不在支援清單。
        DataIntegrityError: 輸入格式不合法、輸入間距非 1m、聚合後為空等。
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ConfigError(
            f"unsupported timeframe '{timeframe}', "
            f"must be one of {list(SUPPORTED_TIMEFRAMES)}"
        )

    validate_ohlc(df_1m, require_volume=True)

    if len(df_1m) == 0:
        raise DataIntegrityError("input dataframe is empty")

    # 防呆：確認來源真的是 1m（取 index diff 的眾數）
    if len(df_1m) >= 2:
        diffs = df_1m.index.to_series().diff().dropna()
        if not diffs.empty:
            mode = diffs.mode().iloc[0]
            if mode != pd.Timedelta(minutes=1):
                raise DataIntegrityError(
                    f"resample expects 1m source, got median interval {mode}"
                )

    # 1m → 1m 直接回傳副本（避免無謂運算）
    if timeframe == "1m":
        return df_1m.copy()

    rule = SUPPORTED_TIMEFRAMES[timeframe]
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # 保留 OHLCV 之外的欄位以備不時之需，但不參與下游運算
    extra = [c for c in df_1m.columns if c not in agg]
    for col in extra:
        agg[col] = "last"

    resampled = df_1m.resample(rule, closed="left", label="left").agg(agg)
    # resample 會在資料缺口插入 NaN row → 一律丟棄（永遠不 forward-fill 價格）
    resampled = resampled.dropna(subset=list(agg.keys()))

    # 丟棄最後一段不完整週期：若最後一根 K 標籤對應的時段尾端
    # 超出原始資料的最後一根 1m 標籤，代表資料不夠 → 捨棄
    if not resampled.empty:
        last_label = resampled.index[-1]
        period = TIMEFRAME_DURATION[timeframe]
        last_bar_covers_until = last_label + period
        # 原始 1m 最後一根的下一分鐘是它「涵蓋到」的時間
        source_covers_until = df_1m.index[-1] + pd.Timedelta(minutes=1)
        if last_bar_covers_until > source_covers_until:
            resampled = resampled.iloc[:-1]

    if resampled.empty:
        raise DataIntegrityError(
            f"resample to {timeframe} produced empty dataframe"
        )

    return resampled
