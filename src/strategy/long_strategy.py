"""多頭趨勢策略 — 對應 多頭趨勢策略_v2.md 規格。

進場條件（Bar[0] 收盤後判斷，bar_index 對應 Bar[0]）：

  條件 1（黃金交叉）：
    HA_WMA_fast[t]   >  HA_WMA_slow[t]
    HA_WMA_fast[t-1] <= HA_WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    HA_Close[t-2] < HA_Close[t]
    HA_Close[t-3] < HA_Close[t]

兩條件同時成立 → 產生 ENTRY 訊號，初始止損 = highest(N) - ATR × multiplier。

Look-ahead 防護：只讀 ``df.iloc[: bar_index + 1]``，禁止使用 ``bar_index + 1`` 之後的資料。
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
        # 進場需要 bar[t-3..t] 共 4 根，外加暖機期
        if bar_index < self.params.warmup_bars:
            return None
        if bar_index < 3:
            return None

        # 切片：只看 [t-3 .. t]，杜絕 look-ahead
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

        # 暖機 NaN 保護：任何指標尚未就緒 → 不出訊號
        for v in (wma_f_t, wma_f_prev, wma_s_t, wma_s_prev, hc_t, hc_t2, hc_t3):
            if math.isnan(v):
                return None

        # 條件 1：當根金叉 + 前根尚未交叉
        cond_cross = (wma_f_t > wma_s_t) and (wma_f_prev <= wma_s_prev)
        if not cond_cross:
            return None

        # 條件 2：交叉前 -2 / -3 根 HA_Close 低於當根
        cond_structure = (hc_t2 < hc_t) and (hc_t3 < hc_t)
        if not cond_structure:
            return None

        # 計算初始止損
        initial_stop = self.compute_trailing_stop_candidate(df, bar_index)
        if math.isnan(initial_stop):
            return None

        return EntrySignal(
            direction=Direction.LONG,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=float(initial_stop),
            reason=(
                f"golden_cross & structure: "
                f"wma_f={wma_f_t:.4f} > wma_s={wma_s_t:.4f}, "
                f"hc[-2]={hc_t2:.4f}, hc[-3]={hc_t3:.4f} < hc[0]={hc_t:.4f}"
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

        # 用原始 K 線 high；只看 [t-N+1 .. t]，無 look-ahead
        window = df["high"].iloc[bar_index - n + 1 : bar_index + 1]
        highest = float(window.max())
        return highest - atr_val * self.params.atr_multiplier
