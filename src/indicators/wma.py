"""Weighted Moving Average（線性權重）。

公式：
    WMA(n)[t] = (1*x[t-n+1] + 2*x[t-n+2] + ... + n*x[t]) / (1+2+...+n)

Look-ahead 防護：window 僅取 ``[t-n+1 .. t]``，min_periods=period 確保暖機期為 NaN，
不會用任何 backward-fill。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation import validate_period


def wma(series: pd.Series, period: int) -> pd.Series:
    """計算線性權重移動平均。

    使用 ``np.convolve`` 在 C 層級執行（比 ``rolling.apply`` lambda 快 ~ 數十倍），
    在大序列（百萬根 K）下仍能即時運算。

    Args:
        series: 輸入時間序列（通常為 close）。
        period: WMA 週期，正整數。

    Returns:
        與輸入 index 對齊的 WMA 序列；前 ``period - 1`` 根為 NaN（暖機期）。

    Raises:
        DataIntegrityError: ``period`` 不是正整數。
    """
    validate_period(period, name="WMA period")

    n = len(series)
    arr = series.to_numpy(dtype=np.float64)
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights_sum = weights.sum()

    result = np.full(n, np.nan, dtype=np.float64)
    if n >= period:
        # np.convolve 會將 kernel 反轉，故傳入 weights[::-1] 讓最新 bar 對應最大權重
        rolling_dot = np.convolve(arr, weights[::-1], mode="valid")
        result[period - 1 :] = rolling_dot / weights_sum

    return pd.Series(result, index=series.index, name=series.name)
