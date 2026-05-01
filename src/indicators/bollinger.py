"""Bollinger Band（中軌可選 SMA / WMA，標準差以中軌為基準）。

公式：
    middle[t] = MA(close, period)            # 預設 WMA（依專案需求）
    sigma[t]  = sqrt( mean((close_i - middle[t])^2)  for i in [t-N+1..t] )
    upper[t]  = middle[t] + num_std × sigma[t]
    lower[t]  = middle[t] − num_std × sigma[t]

向量化：利用恆等式
    E[(x − m)^2] = E[x^2] − 2·m·E[x] + m^2
其中 E[·] 為視窗 N 的簡單平均，m = middle (WMA)。
此寫法把 σ 計算降到純 pandas rolling，無 Python 迴圈。

Look-ahead 防護：rolling 視窗只看 ``[t-N+1..t]``，min_periods=period 確保暖機期 NaN，
不會 backward-fill。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.indicators.wma import wma
from src.utils.validation import validate_period


def bollinger_bands(
    close: pd.Series,
    period: int,
    num_std: float = 2.0,
    *,
    ma_type: Literal["wma", "sma"] = "wma",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """計算 Bollinger Band 三條線。

    Args:
        close: 收盤價序列。
        period: 視窗長度（典型 20）。
        num_std: 上下軌距中軌的標準差倍數（典型 2）。
        ma_type: 中軌計算方式。"wma"（預設）= 線性權重；"sma" = 簡單平均。
            σ 一律以中軌為基準，不論 ma_type 為何。

    Returns:
        (middle, upper, lower)，皆為與輸入 index 對齊的 pd.Series。
        前 ``period - 1`` 根為 NaN（暖機期）。

    Raises:
        DataIntegrityError: ``period`` 不為正整數、``num_std`` 非正。
    """
    validate_period(period, name="Bollinger period")
    if num_std <= 0:
        from src.utils.exceptions import DataIntegrityError
        raise DataIntegrityError(f"num_std must be > 0, got {num_std}")

    # 中軌
    if ma_type == "wma":
        middle = wma(close, period)
    elif ma_type == "sma":
        middle = close.rolling(window=period, min_periods=period).mean()
    else:
        from src.utils.exceptions import DataIntegrityError
        raise DataIntegrityError(f"unknown ma_type '{ma_type}'")

    # σ²：用 E[x²] − 2·m·E[x] + m²，向量化計算
    mean_x = close.rolling(window=period, min_periods=period).mean()
    mean_x2 = (close**2).rolling(window=period, min_periods=period).mean()
    variance = mean_x2 - 2 * middle * mean_x + middle**2
    # 浮點誤差可能讓 variance 微負（< 1e-10），clip 確保 sqrt 安全
    variance = variance.clip(lower=0)
    sigma = np.sqrt(variance)

    upper = middle + num_std * sigma
    lower = middle - num_std * sigma

    return (
        middle.rename("bb_middle"),
        upper.rename("bb_upper"),
        lower.rename("bb_lower"),
    )
