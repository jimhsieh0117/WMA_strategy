"""Market Structure（LuxAlgo 風格 pivot-based）— BoS / CHoCH 偵測。

核心概念
========

- **Pivot High (PH)**：bar ``p`` 滿足 ``high[p] = max(high[p-left .. p+right])``，
  即左右各 ``N`` 根 K 線都不超過該根高點。**確認時間在 bar p+right**（需要看到
  右側 N 根才能確定）。
- **Pivot Low (PL)**：對稱定義於低點。
- **狀態**：``trend ∈ {bull, bear, none}``、``last_PH``、``last_PL``。
- **事件**（在「突破當下」那根 bar 標記）：
    - **BoS_up**：trend=bull 時，``close[t] > last_PH`` → 上升結構延續，更新 last_PH。
    - **BoS_down**：trend=bear 時，``close[t] < last_PL`` → 下降結構延續，更新 last_PL。
    - **CHoCH_up**：trend=bear 時，``close[t] > last_PH`` → 結構翻轉為多。
    - **CHoCH_down**：trend=bull 時，``close[t] < last_PL`` → 結構翻轉為空。

Look-ahead 安全性
==================

- ``ms_swing_high[p]`` 雖填在 bar p 的位置，但**只在 bar p+right 之後才被填入**；
  在 bar t<p+right 視角下，該欄仍是 NaN。
- 事件偵測使用 ``last_PH/last_PL``，這兩個變數只在 pivot 確認後（``p+right``）才
  更新。bar t 的事件判定只依賴 ``high/low/close[≤t]``。
- ``ms_trend`` 由事件 forward-fill 產生，事件在 bar t 確定的話 trend[t]
  也是 bar t 收盤後可得的因果值。

回傳欄位
========

- ``ms_swing_high``：float，NaN 除了已確認的 PH bar（值=high[p]）
- ``ms_swing_low``：float，NaN 除了已確認的 PL bar（值=low[p]）
- ``ms_event``：str，bar t 上發生的事件名稱（``bos_up`` / ``bos_down`` /
  ``choch_up`` / ``choch_down``）或空字串
- ``ms_event_price``：float，事件當下用於繪圖的價格（突破 close）
- ``ms_trend``：str，bar t 收盤後的當前結構方向（``bull`` / ``bear`` / 空字串）
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.exceptions import DataIntegrityError
from src.utils.validation import validate_ohlc, validate_period


def compute_market_structure(
    df: pd.DataFrame,
    *,
    pivot_left: int = 10,
    pivot_right: int = 10,
) -> pd.DataFrame:
    """計算市場結構欄位（pivot + BoS/CHoCH 事件 + 趨勢狀態）。

    Args:
        df: OHLC DataFrame（需含 high/low/close，DatetimeIndex）。
        pivot_left: pivot 左側回看根數。
        pivot_right: pivot 右側往前看根數（決定確認延遲）。

    Returns:
        新 DataFrame，原欄位 + ``ms_swing_high`` / ``ms_swing_low`` /
        ``ms_event`` / ``ms_event_price`` / ``ms_trend``。

    Raises:
        DataIntegrityError: OHLC 不合規或週期非法。
    """
    validate_ohlc(df)
    validate_period(pivot_left, name="pivot_left")
    validate_period(pivot_right, name="pivot_right")

    out = df.copy()
    n = len(df)

    swing_high = np.full(n, np.nan, dtype="float64")
    swing_low = np.full(n, np.nan, dtype="float64")
    event = np.array([""] * n, dtype=object)
    event_price = np.full(n, np.nan, dtype="float64")
    trend = np.array([""] * n, dtype=object)

    if n == 0:
        out["ms_swing_high"] = swing_high
        out["ms_swing_low"] = swing_low
        out["ms_event"] = event
        out["ms_event_price"] = event_price
        out["ms_trend"] = trend
        return out

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    last_ph: float | None = None
    last_pl: float | None = None
    current_trend: str = ""  # "bull" / "bear" / ""

    # 主迴圈：每根 bar t 都做兩件事
    # 1. 先做事件判定（用 close[t] 與 last_PH/last_PL，後者只反映 ≤t-pivot_right 的資訊）
    # 2. 收盤後，檢查 bar t-pivot_right 是否為已確認的 pivot；若是則更新 last_PH/last_PL
    for t in range(n):
        # ---- (1) 事件判定（突破當下立即標記）----
        bos_or_choch: str = ""
        if last_ph is not None and close[t] > last_ph:
            if current_trend == "bear":
                bos_or_choch = "choch_up"
                current_trend = "bull"
            else:
                bos_or_choch = "bos_up"
                current_trend = "bull"
            # 突破後 last_PH 不再有效（已被穿越），等下一個 pivot 確認才重設
            last_ph = None
        elif last_pl is not None and close[t] < last_pl:
            if current_trend == "bull":
                bos_or_choch = "choch_down"
                current_trend = "bear"
            else:
                bos_or_choch = "bos_down"
                current_trend = "bear"
            last_pl = None

        if bos_or_choch:
            event[t] = bos_or_choch
            event_price[t] = close[t]

        trend[t] = current_trend

        # ---- (2) 確認 pivot：bar p = t - pivot_right ----
        p = t - pivot_right
        if p < pivot_left:
            continue  # 左側不足，無法確認

        window_start = p - pivot_left
        window_end = p + pivot_right + 1  # 不含
        win_high = high[window_start:window_end]
        win_low = low[window_start:window_end]
        # pivot high：bar p 嚴格高於左右 N 根，**且不與其他 bar 並列**
        # 用 strict 條件避免水平段同時形成兩個 PH/PL
        if high[p] == win_high.max() and (win_high < high[p]).sum() == len(win_high) - 1:
            swing_high[p] = high[p]
            # last_ph 永遠採用「最近一個確認的 PH」（LuxAlgo 標準）；
            # 即使新 PH 較低也覆蓋，因為結構參考點是「最近」而非「最高」。
            # 這樣 lower-high 才能形成 bear 結構訊號。
            last_ph = high[p]
        if low[p] == win_low.min() and (win_low > low[p]).sum() == len(win_low) - 1:
            swing_low[p] = low[p]
            # 同理：採最近 PL；higher-low 才能在 bull trend 中作為 CHoCH_down 參考。
            last_pl = low[p]

    out["ms_swing_high"] = swing_high
    out["ms_swing_low"] = swing_low
    out["ms_event"] = event
    out["ms_event_price"] = event_price
    out["ms_trend"] = trend
    return out
