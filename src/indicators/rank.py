"""Rolling percent rank：對序列 s 在過去 window 根的相對位置（0..1）。

用於 chop_filter 將 ATR / BBW 的絕對值轉成「百分位」，避免不同 timeframe / 不同
時段絕對值差異造成門檻不可比。

Look-ahead 防護：rolling 視窗只用 ≤ t 的資料。
"""

from __future__ import annotations

import pandas as pd


def percent_rank(s: pd.Series, window: int) -> pd.Series:
    """rolling 百分位 (0..1)。前 window-1 根為 NaN（暖機期）。

    Args:
        s: 任意數值序列。
        window: rolling 視窗根數，必須 ≥ 2。

    Returns:
        與輸入 index 對齊的百分位序列（0..1）。
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError(f"percent_rank window must be int >= 2, got {window}")
    return s.rolling(window=window, min_periods=window).rank(pct=True).rename("pct_rank")
