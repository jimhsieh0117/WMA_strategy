"""Heikin-Ashi 平均K線。

公式：
    HA_Close[t] = (Open[t] + High[t] + Low[t] + Close[t]) / 4
    HA_Open[0]  = (Open[0] + Close[0]) / 2                       # seed
    HA_Open[t]  = (HA_Open[t-1] + HA_Close[t-1]) / 2     for t>=1
    HA_High[t]  = max(High[t], HA_Open[t], HA_Close[t])
    HA_Low[t]   = min(Low[t], HA_Open[t], HA_Close[t])

Look-ahead 防護：所有 HA[t] 僅依賴 ``bar[0..t]``，無未來資料。
HA_Open 為遞迴定義，因此使用 numpy for-loop（O(n)），不可用任何 backward-fill。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation import validate_ohlc

HA_COLUMNS: tuple[str, ...] = ("ha_open", "ha_high", "ha_low", "ha_close")


def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    """計算 Heikin-Ashi 並回傳新增 ``ha_open / ha_high / ha_low / ha_close`` 欄的副本。

    Args:
        df: 含 ``open / high / low / close`` 的 OHLC DataFrame，index 為 DatetimeIndex。

    Returns:
        原 DataFrame 加上 4 個 HA 欄位的新副本（不就地修改）。

    Raises:
        DataIntegrityError: 輸入格式不合法。
    """
    validate_ohlc(df)

    n = len(df)
    result = df.copy()
    if n == 0:
        for col in HA_COLUMNS:
            result[col] = pd.Series(dtype="float64")
        return result

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    # HA_Close 純元素級計算
    ha_close = (o + h + low + c) / 4.0

    # HA_Open 遞迴：無法向量化，用 O(n) 迴圈
    ha_open = np.empty(n, dtype=np.float64)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for t in range(1, n):
        ha_open[t] = (ha_open[t - 1] + ha_close[t - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

    result["ha_open"] = ha_open
    result["ha_high"] = ha_high
    result["ha_low"] = ha_low
    result["ha_close"] = ha_close
    return result
