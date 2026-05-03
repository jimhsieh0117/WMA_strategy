"""WaveTrend Oscillator（LazyBear 版本）。

公式：
    ap   = (high + low + close) / 3                     # typical price
    esa  = EMA(ap, n1)                                   # 平滑均線
    d    = EMA(|ap − esa|, n1)                          # 平均偏離
    ci   = (ap − esa) / (0.015 × d)                     # 標準化通道指標
    wt1  = EMA(ci, n2)                                   # 主線
    wt2  = SMA(wt1, 4)                                   # 訊號線

預設 n1=10, n2=21（LazyBear 原始）。

判讀規則（供策略參考）：
- wt1 > +60：超買區
- wt1 < −60：超賣區
- wt1 上穿 wt2：多頭訊號
- wt1 下穿 wt2：空頭訊號
- wt1 上穿 0：強多
- wt1 下穿 0：強空

Look-ahead 防護：
- pandas ``ewm(adjust=False)`` 為遞迴計算（每根只依賴前一根與當下值），
  本質無 look-ahead；min_periods 控制暖機 NaN 數量
- ``rolling(4).mean()`` 視窗為 [t-3..t]，無未來資料
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.exceptions import DataIntegrityError
from src.utils.validation import validate_ohlc, validate_period


def wavetrend(
    df: pd.DataFrame,
    n1: int = 10,
    n2: int = 21,
    *,
    signal_period: int = 4,
) -> pd.DataFrame:
    """計算 WaveTrend，回傳新增 ``wt1`` / ``wt2`` 欄位的副本。

    Args:
        df: OHLC DataFrame（須含 high/low/close）。
        n1: ESA / D 的 EMA 週期，典型 10。
        n2: CI 平滑成 wt1 的 EMA 週期，典型 21。
        signal_period: wt2 = SMA(wt1, signal_period)，典型 4。

    Returns:
        新 DataFrame，原欄位 + ``wt1`` / ``wt2``。

    Raises:
        DataIntegrityError: OHLC 不合規或週期非法。
    """
    validate_ohlc(df)
    validate_period(n1, name="WaveTrend n1")
    validate_period(n2, name="WaveTrend n2")
    validate_period(signal_period, name="WaveTrend signal_period")

    out = df.copy()
    n = len(df)
    if n == 0:
        out["wt1"] = pd.Series(dtype="float64")
        out["wt2"] = pd.Series(dtype="float64")
        return out

    ap = (df["high"] + df["low"] + df["close"]) / 3.0

    # esa = EMA(ap, n1)
    esa = ap.ewm(span=n1, adjust=False, min_periods=n1).mean()

    # d = EMA(|ap - esa|, n1)
    d = (ap - esa).abs().ewm(span=n1, adjust=False, min_periods=n1).mean()

    # ci = (ap - esa) / (0.015 × d)；d=0 時除以零 → 改 NaN
    denom = 0.015 * d
    ci = (ap - esa) / denom.where(denom != 0, np.nan)
    ci = ci.replace([np.inf, -np.inf], np.nan)

    # wt1 = EMA(ci, n2)
    wt1 = ci.ewm(span=n2, adjust=False, min_periods=n2).mean()

    # wt2 = SMA(wt1, signal_period)
    wt2 = wt1.rolling(window=signal_period, min_periods=signal_period).mean()

    out["wt1"] = wt1
    out["wt2"] = wt2
    return out
