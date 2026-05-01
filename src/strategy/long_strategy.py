"""多頭趨勢策略 — 對應 多頭趨勢策略_v2.md + ARCHITECTURE.md §11 三階段止損。

進場條件（Bar[t] 收盤後判斷）：

  條件 1（黃金交叉）：
    HA_WMA_fast[t]   >  HA_WMA_slow[t]
    HA_WMA_fast[t-1] <= HA_WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    HA_Close[t-2] < HA_Close[t]
    HA_Close[t-3] < HA_Close[t]

兩條件同時成立 → 產生 ENTRY 訊號。

初始止損（Stage 1，由本檔負責）：
    initial_stop = min(low over [t - swing_lookback + 1 .. t]) × (1 − slippage_buffer)
    // 用前 N 根原始 K 線的最低點，再往下 buffer 一個滑點

Stage 2 / Stage 3 止損由 ``TrailingStopController`` 處理，本檔不涉。

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

        wma_fast = df["ha_wma_fast"]
        wma_slow = df["ha_wma_slow"]
        ha_close = df["ha_close"]

        wma_f_t = wma_fast.iat[bar_index]
        wma_f_prev = wma_fast.iat[bar_index - 1]
        wma_s_t = wma_slow.iat[bar_index]
        wma_s_prev = wma_slow.iat[bar_index - 1]
        hc_t = ha_close.iat[bar_index]
        hc_t2 = ha_close.iat[bar_index - 2]
        hc_t3 = ha_close.iat[bar_index - 3]

        # 暖機 NaN 防護
        for v in (wma_f_t, wma_f_prev, wma_s_t, wma_s_prev, hc_t, hc_t2, hc_t3):
            if math.isnan(v):
                return None

        # 條件 1：當根金叉 + 前根尚未交叉
        if not ((wma_f_t > wma_s_t) and (wma_f_prev <= wma_s_prev)):
            return None

        # 條件 2：交叉前 -2 / -3 根 HA_Close 低於當根
        if not ((hc_t2 < hc_t) and (hc_t3 < hc_t)):
            return None

        # Stage 1 初始止損：前 N 根原始 K 線最低低點，再往下 buffer
        n = self.params.trailing.swing_lookback
        if bar_index + 1 < n:
            return None
        swing_window = df["low"].iloc[bar_index - n + 1 : bar_index + 1]
        swing_low = float(swing_window.min())
        initial_stop = swing_low * (1.0 - self.params.trailing.stage1_slippage_buffer)

        # 安全檢查：止損不能高於當下收盤（會立即被打掉）；
        # 訊號 bar 收盤是 entry 的最佳近似（實際 entry 在 bar t+1 開盤）
        ref_price = float(df["close"].iat[bar_index])
        if initial_stop >= ref_price:
            return None

        return EntrySignal(
            direction=Direction.LONG,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=initial_stop,
            reason=(
                f"golden_cross & structure: "
                f"wma_f={wma_f_t:.4f} > wma_s={wma_s_t:.4f}, "
                f"hc[-2]={hc_t2:.4f}, hc[-3]={hc_t3:.4f} < hc[0]={hc_t:.4f}, "
                f"swing_low={swing_low:.4f}"
            ),
        )
