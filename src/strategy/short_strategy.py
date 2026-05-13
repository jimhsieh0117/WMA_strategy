"""空頭趨勢策略 — 對應 空頭趨勢策略_v2.md + ARCHITECTURE.md §10 三階段止損。

進場條件（Bar[t] 收盤後判斷，一律用原始 K 線）：

  條件 1（死亡交叉，僅在交叉根判定一次）：
    WMA_fast[t]   <  WMA_slow[t]
    WMA_fast[t-1] >= WMA_slow[t-1]

  條件 2（趨勢結構確認）：
    open[bar-2] > close[bar]    # bar 在 retry 時跟著推進

  其他濾網（signal_filter / chop_filter / structure_filter）+ Stage 1 安全。

Entry retry（``params.entry_retry.max_attempts``）：
  交叉發生後給連續 N 根 K 嘗試其他條件。任一根全部過 → 進場、消費 pending。
  WMA 交叉條件不重檢；視為機會訊號鎖定。

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

        max_attempts = self.params.entry_retry.short_max_attempts

        if self._detect_cross(df, bar_index):
            self._pending_cross_bar = bar_index
            self._pending_attempts_used = 0

        if self._pending_cross_bar is None:
            return None

        if self._pending_attempts_used >= max_attempts:
            self._pending_cross_bar = None
            self._pending_attempts_used = 0
            return None

        self._pending_attempts_used += 1
        signal = self._try_entry(df, bar_index)
        if signal is not None:
            self._pending_cross_bar = None
            self._pending_attempts_used = 0
            return signal
        return None

    def _detect_cross(self, df: pd.DataFrame, bar_index: int) -> bool:
        """是否在 bar_index 形成死叉（WMA_fast 從上穿越 WMA_slow）。"""
        if bar_index < 1:
            return False
        wma_f_t = df["wma_fast"].iat[bar_index]
        wma_f_prev = df["wma_fast"].iat[bar_index - 1]
        wma_s_t = df["wma_slow"].iat[bar_index]
        wma_s_prev = df["wma_slow"].iat[bar_index - 1]
        if (math.isnan(wma_f_t) or math.isnan(wma_f_prev)
                or math.isnan(wma_s_t) or math.isnan(wma_s_prev)):
            return False
        return (wma_f_t < wma_s_t) and (wma_f_prev >= wma_s_prev)

    def _try_entry(
        self, df: pd.DataFrame, bar_index: int,
    ) -> EntrySignal | None:
        """檢查除 WMA 交叉外的所有進場條件；通過則組 EntrySignal。"""
        if bar_index < 2:
            return None
        c_t = df["close"].iat[bar_index]
        o_t2 = df["open"].iat[bar_index - 2]
        if math.isnan(c_t) or math.isnan(o_t2):
            return None

        # 結構：open[bar-2] > close[bar]
        if not (o_t2 > c_t):
            return None

        if not passes_signal_filter(df, bar_index, Direction.SHORT, self.params):
            return None

        if not passes_chop_filter(df, bar_index, self.params):
            return None

        if not passes_structure_filter(
            df, bar_index, Direction.SHORT, self.params,
        ):
            return None

        n = self.params.trailing.swing_lookback
        if bar_index + 1 < n:
            return None
        swing_window = df["high"].iloc[bar_index - n + 1 : bar_index + 1]
        swing_high = float(swing_window.max())
        initial_stop = swing_high * (1.0 + self.params.trailing.stage1_slippage_buffer)

        ref_price = float(c_t)
        if initial_stop <= ref_price:
            return None

        wma_f_t = df["wma_fast"].iat[bar_index]
        wma_s_t = df["wma_slow"].iat[bar_index]
        bars_after_cross = (
            bar_index - self._pending_cross_bar
            if self._pending_cross_bar is not None else 0
        )
        return EntrySignal(
            direction=Direction.SHORT,
            bar_index=bar_index,
            timestamp=df.index[bar_index],
            initial_stop=initial_stop,
            reason=(
                f"short entry (attempt {self._pending_attempts_used}, "
                f"+{bars_after_cross}bar from cross): "
                f"wma_f={wma_f_t:.4f} < wma_s={wma_s_t:.4f}, "
                f"o[-2]={o_t2:.4f} > c[0]={c_t:.4f}, "
                f"swing_high={swing_high:.4f}"
            ),
        )
