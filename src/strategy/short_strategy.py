"""空頭趨勢策略 — 對應 空頭趨勢策略_v2.md + ARCHITECTURE.md §11 三階段止損。

進場條件（Bar[t] 收盤後判斷）：

  條件 1（死亡交叉）：
    HA_WMA_fast[t]   <  HA_WMA_slow[t]
    HA_WMA_fast[t-1] >= HA_WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    HA_Close[t-2] > HA_Close[t]
    HA_Close[t-3] > HA_Close[t]

兩條件同時成立 → 產生 ENTRY 訊號。

初始止損（Stage 1）：
    initial_stop = max(high over [t - swing_lookback + 1 .. t]) × (1 + slippage_buffer)

Stage 2 / Stage 3 止損由 ``TrailingStopController`` 處理。

Look-ahead 防護：只讀 ``df.iloc[: bar_index + 1]``。
"""

from __future__ import annotations

import math

import pandas as pd

from src.strategy.base import BaseTrendStrategy, assert_indicators_ready
from src.strategy.types import EntrySignal
from src.utils.types import Direction


class ShortTrendStrategy(BaseTrendStrategy):
    direction: Direction = Direction.SHORT

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

        for v in (wma_f_t, wma_f_prev, wma_s_t, wma_s_prev, hc_t, hc_t2, hc_t3):
            if math.isnan(v):
                return None

        # 條件 1：當根死叉
        if not ((wma_f_t < wma_s_t) and (wma_f_prev >= wma_s_prev)):
            return None

        # 條件 2：交叉前 -2 / -3 根 HA_Close 高於當根
        if not ((hc_t2 > hc_t) and (hc_t3 > hc_t)):
            return None

        # Stage 1 初始止損：前 N 根原始 K 線最高點，再往上 buffer
        n = self.params.trailing.swing_lookback
        if bar_index + 1 < n:
            return None
        swing_window = df["high"].iloc[bar_index - n + 1 : bar_index + 1]
        swing_high = float(swing_window.max())
        initial_stop = swing_high * (1.0 + self.params.trailing.stage1_slippage_buffer)

        ref_price = float(df["close"].iat[bar_index])
        if initial_stop <= ref_price:
            return None

        return EntrySignal(
            direction=Direction.SHORT,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=initial_stop,
            reason=(
                f"death_cross & structure: "
                f"wma_f={wma_f_t:.4f} < wma_s={wma_s_t:.4f}, "
                f"hc[-2]={hc_t2:.4f}, hc[-3]={hc_t3:.4f} > hc[0]={hc_t:.4f}, "
                f"swing_high={swing_high:.4f}"
            ),
        )
