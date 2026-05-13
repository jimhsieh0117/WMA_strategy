"""空頭趨勢策略 — 對應 空頭趨勢策略_v2.md + ARCHITECTURE.md §10 三階段止損。

進場條件（Bar[t] 收盤後判斷，一律用原始 K 線）：

  條件 1（死亡交叉）：
    WMA_fast[t]   <  WMA_slow[t]
    WMA_fast[t-1] >= WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    Close[t-2] > Close[t]
    Close[t-3] > Close[t]

兩條件同時成立 → 產生 ENTRY 訊號。

初始止損（Stage 1）用原始 K 線 swing high：
    initial_stop = max(high over [t - swing_lookback + 1 .. t]) × (1 + slippage_buffer)

Look-ahead 防護：只讀 ``df.iloc[: bar_index + 1]``。
"""

from __future__ import annotations

import math

import pandas as pd

from src.strategy.base import (
    BaseTrendStrategy, assert_indicators_ready,
    passes_chop_filter, passes_signal_filter, passes_structure_filter,
)
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

        wma_fast = df["wma_fast"]
        wma_slow = df["wma_slow"]
        close = df["close"]
        open_ = df["open"]

        wma_f_t = wma_fast.iat[bar_index]
        wma_f_prev = wma_fast.iat[bar_index - 1]
        wma_s_t = wma_slow.iat[bar_index]
        wma_s_prev = wma_slow.iat[bar_index - 1]
        c_t = close.iat[bar_index]
        # 趨勢結構確認改用 open（實驗）：t-2 / t-3 的「開盤價」與當根 close 比較
        o_t2 = open_.iat[bar_index - 2]
        o_t3 = open_.iat[bar_index - 3]

        for v in (wma_f_t, wma_f_prev, wma_s_t, wma_s_prev, c_t, o_t2, o_t3):
            if math.isnan(v):
                return None

        # 條件 1：當根死叉
        if not ((wma_f_t < wma_s_t) and (wma_f_prev >= wma_s_prev)):
            return None

        # 條件 2：交叉前 -2 根 open 高於當根 close（暫時只看 t-2）
        if not (o_t2 > c_t):
            return None
        # if not (o_t3 > c_t):
        #     return None

        # 條件 3（可選）：進場前 N 根 K 實體比例濾網
        if not passes_signal_filter(df, bar_index, Direction.SHORT, self.params):
            return None

        # 條件 4（可選）：盤整濾網（BBW_rank / ATR_rank / ADX）
        if not passes_chop_filter(df, bar_index, self.params):
            return None

        # 條件 5（可選）：結構順勢濾網（market structure ms_trend）
        # 只在進場擋；持倉/止損照原邏輯走，結構翻轉不主動平倉。
        if not passes_structure_filter(
            df, bar_index, Direction.SHORT, self.params,
        ):
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
                f"o[-2]={o_t2:.4f}, o[-3]={o_t3:.4f} > c[0]={c_t:.4f}, "
                f"swing_high={swing_high:.4f}"
            ),
        )
