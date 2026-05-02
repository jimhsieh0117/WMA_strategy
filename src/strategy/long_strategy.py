"""多頭趨勢策略 — 對應 多頭趨勢策略_v2.md + ARCHITECTURE.md §10 三階段止損。

進場條件（Bar[t] 收盤後判斷，K 線來源由 ``entry_source`` 切換）：

  條件 1（黃金交叉）：
    WMA_fast[t]   >  WMA_slow[t]
    WMA_fast[t-1] <= WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    Close[t-2] < Close[t]
    Close[t-3] < Close[t]

兩條件同時成立 → 產生 ENTRY 訊號。

``entry_source`` 切換規則：
- ``"ha"``：用 ha_wma_fast / ha_wma_slow / ha_close（HA 平滑後的訊號）
- ``"raw"``：用 wma_fast / wma_slow / close（原始 K 線訊號）

初始止損（Stage 1）始終由原始 K 線的 swing low 算，不受 entry_source 影響：
    initial_stop = min(low over [t - swing_lookback + 1 .. t]) × (1 − slippage_buffer)

Look-ahead 防護：只讀 ``df.iloc[: bar_index + 1]``。
"""

from __future__ import annotations

import math

import pandas as pd

from src.strategy.base import BaseTrendStrategy, assert_indicators_ready
from src.strategy.types import EntrySignal
from src.utils.types import Direction


class LongTrendStrategy(BaseTrendStrategy):
    direction: Direction = Direction.LONG

    def detect_entry(
        self, df: pd.DataFrame, bar_index: int
    ) -> EntrySignal | None:
        assert_indicators_ready(df)
        if bar_index < self.params.warmup_bars:
            return None
        if bar_index < 3:
            return None

        # 依 entry_source 切換指標欄位
        if self.params.entry_source == "ha":
            wma_fast = df["ha_wma_fast"]
            wma_slow = df["ha_wma_slow"]
            close = df["ha_close"]
            source_tag = "HA"
        else:  # "raw"
            wma_fast = df["wma_fast"]
            wma_slow = df["wma_slow"]
            close = df["close"]
            source_tag = "RAW"

        wma_f_t = wma_fast.iat[bar_index]
        wma_f_prev = wma_fast.iat[bar_index - 1]
        wma_s_t = wma_slow.iat[bar_index]
        wma_s_prev = wma_slow.iat[bar_index - 1]
        c_t = close.iat[bar_index]
        c_t2 = close.iat[bar_index - 2]
        c_t3 = close.iat[bar_index - 3]

        # 暖機 NaN 防護
        for v in (wma_f_t, wma_f_prev, wma_s_t, wma_s_prev, c_t, c_t2, c_t3):
            if math.isnan(v):
                return None

        # 條件 1：當根金叉 + 前根尚未交叉
        if not ((wma_f_t > wma_s_t) and (wma_f_prev <= wma_s_prev)):
            return None

        # 條件 2：交叉前 -2 / -3 根 close 低於當根
        if not ((c_t2 < c_t) and (c_t3 < c_t)):
            return None

        # Stage 1 初始止損：前 N 根原始 K 線最低低點，再往下 buffer
        # （止損永遠用原始 low，與 entry_source 無關）
        n = self.params.trailing.swing_lookback
        if bar_index + 1 < n:
            return None
        swing_window = df["low"].iloc[bar_index - n + 1 : bar_index + 1]
        swing_low = float(swing_window.min())
        initial_stop = swing_low * (1.0 - self.params.trailing.stage1_slippage_buffer)

        # 安全檢查：止損不能高於當下原始 close（會立即被打掉）
        ref_price = float(df["close"].iat[bar_index])
        if initial_stop >= ref_price:
            return None

        return EntrySignal(
            direction=Direction.LONG,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=initial_stop,
            reason=(
                f"[{source_tag}] golden_cross & structure: "
                f"wma_f={wma_f_t:.4f} > wma_s={wma_s_t:.4f}, "
                f"c[-2]={c_t2:.4f}, c[-3]={c_t3:.4f} < c[0]={c_t:.4f}, "
                f"swing_low={swing_low:.4f}"
            ),
        )
