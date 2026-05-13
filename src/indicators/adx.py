"""Average Directional Index（ADX）+ Directional Movement Index（DMI）。

Wilder smoothing（RMA）版本，與多數圖表平台 / TradingView ADX(14) 一致。

公式：
    +DM[t] = max(H[t]-H[t-1], 0) if (H[t]-H[t-1]) > (L[t-1]-L[t]) else 0
    -DM[t] = max(L[t-1]-L[t], 0) if (L[t-1]-L[t]) > (H[t]-H[t-1]) else 0
    TR[t]  = max(H-L, |H-prev_C|, |L-prev_C|)
    Wilder smoothing (period N):  S[t] = S[t-1] - S[t-1]/N + X[t]
    +DI = 100 × W(+DM) / W(TR)
    -DI = 100 × W(-DM) / W(TR)
    DX  = 100 × |+DI - -DI| / (+DI + -DI)
    ADX = W(DX)  ← 第二次 Wilder smoothing

Look-ahead 防護：所有計算僅用 ≤ t 的資料；rolling/cumulative 操作只往回看。
ADX 計算用**原始 K 線**，與策略文件一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.atr import true_range
from src.utils.validation import validate_ohlc, validate_period


def _wilder_smooth(s: pd.Series, period: int) -> pd.Series:
    """Wilder RMA：等同於 alpha = 1/period 的 EMA，但用 `min_periods=period` 暖機。

    與「先 sum N 根再每步 (S − S/N + X)」的傳統 Wilder 公式數值上等價，
    pandas EWM 實作向後相容性穩定。
    """
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx_dmi(
    df: pd.DataFrame, period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """計算 ADX、+DI、-DI（皆 0..100 區間）。

    Args:
        df: 含 high/low/close 的 OHLC DataFrame。
        period: Wilder 平滑週期，預設 14。

    Returns:
        (adx, plus_di, minus_di)；前 ~2×period 根為 NaN（兩次 Wilder smoothing 暖機）。
    """
    validate_ohlc(df)
    validate_period(period, name="ADX period")

    h = df["high"]
    low = df["low"]

    up_move = h.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    tr = true_range(df)
    atr_w = _wilder_smooth(tr, period)
    plus_di = 100.0 * _wilder_smooth(plus_dm, period) / atr_w
    minus_di = 100.0 * _wilder_smooth(minus_dm, period) / atr_w

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder_smooth(dx, period)
    return adx.rename("adx"), plus_di.rename("plus_di"), minus_di.rename("minus_di")
