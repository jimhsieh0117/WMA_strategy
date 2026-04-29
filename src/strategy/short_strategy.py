"""空頭趨勢策略 — 對應 空頭趨勢策略_v2.md 規格。

進場條件（Bar[0] 收盤後判斷，bar_index 對應 Bar[0]）：

  條件 1（死亡交叉）：
    HA_WMA_fast[t]   <  HA_WMA_slow[t]
    HA_WMA_fast[t-1] >= HA_WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    HA_Close[t-2] > HA_Close[t]
    HA_Close[t-3] > HA_Close[t]

兩條件同時成立 → 產生 ENTRY 訊號，初始止損 = lowest(N) + ATR × multiplier。

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

        # 條件 1：當根死叉 + 前根尚未交叉
        cond_cross = (wma_f_t < wma_s_t) and (wma_f_prev >= wma_s_prev)
        if not cond_cross:
            return None

        # 條件 2：交叉前 -2 / -3 根 HA_Close 高於當根
        cond_structure = (hc_t2 > hc_t) and (hc_t3 > hc_t)
        if not cond_structure:
            return None

        initial_stop = self.compute_trailing_stop_candidate(df, bar_index)
        if math.isnan(initial_stop):
            return None

        return EntrySignal(
            direction=Direction.SHORT,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=float(initial_stop),
            reason=(
                f"death_cross & structure: "
                f"wma_f={wma_f_t:.4f} < wma_s={wma_s_t:.4f}, "
                f"hc[-2]={hc_t2:.4f}, hc[-3]={hc_t3:.4f} > hc[0]={hc_t:.4f}"
            ),
        )

    def compute_trailing_stop_candidate(
        self, df: pd.DataFrame, bar_index: int
    ) -> float:
        assert_indicators_ready(df)
        n = self.params.atr_lookback
        if bar_index + 1 < n:
            return math.nan

        atr_val = df["atr"].iat[bar_index]
        if math.isnan(atr_val):
            return math.nan

        # 用原始 K 線 low
        window = df["low"].iloc[bar_index - n + 1 : bar_index + 1]
        lowest = float(window.min())
        return lowest + atr_val * self.params.atr_multiplier
