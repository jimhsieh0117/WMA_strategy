"""Average True Range（簡單平均版本）。

公式：
    TR[0] = High[0] - Low[0]                                     # 首根無 prev_close
    TR[t] = max(High[t]-Low[t],
                |High[t]-Close[t-1]|,
                |Low[t] -Close[t-1]|)                            for t>=1
    ATR(n)[t] = mean(TR[t-n+1..t])

採用簡單移動平均（SMA）而非 Wilder RMA，與多數圖表平台行為一致。
策略文件對 ATR 的平滑方式未指定 → 採用 SMA 實作；若日後要改 Wilder，介面不需要動。

Look-ahead 防護：TR[t] 只看 bar[t] 與 bar[t-1]；ATR 滾動視窗只看過去與當下。
ATR 來源為**原始 K 線**（非 Heikin-Ashi），與策略文件一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.validation import validate_ohlc, validate_period


def true_range(df: pd.DataFrame) -> pd.Series:
    """計算每根 K 線的 True Range。

    Args:
        df: 含 high/low/close 的 OHLC DataFrame。

    Returns:
        與輸入 index 對齊的 TR 序列。
    """
    validate_ohlc(df)
    n = len(df)
    tr = np.empty(n, dtype=np.float64)
    if n == 0:
        return pd.Series(tr, index=df.index, name="tr", dtype="float64")

    h = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)

    tr[0] = h[0] - low[0]
    if n > 1:
        prev_c = c[:-1]
        hl = h[1:] - low[1:]
        hc = np.abs(h[1:] - prev_c)
        lc = np.abs(low[1:] - prev_c)
        tr[1:] = np.maximum(hl, np.maximum(hc, lc))

    return pd.Series(tr, index=df.index, name="tr")


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """計算 ATR（True Range 的簡單移動平均）。

    Args:
        df: 原始 K 線 OHLC DataFrame。
        period: ATR 週期，正整數。

    Returns:
        與輸入 index 對齊的 ATR 序列；前 ``period - 1`` 根為 NaN（暖機期）。

    Raises:
        DataIntegrityError: 輸入格式或週期不合法。
    """
    validate_period(period, name="ATR period")
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=period).mean().rename("atr")
