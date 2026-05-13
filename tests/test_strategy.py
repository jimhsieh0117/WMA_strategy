"""Strategy 訊號層單元測試（M5+：swing-based 初始止損 + 三階段拖曳由 controller 處理）。

策略本身只負責進場訊號 + Stage 1 止損；Stage 2/3 拖曳邏輯歸 ``TrailingStopController``，
另在 test_trailing.py 測試。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import (
    REQUIRED_INDICATOR_COLUMNS,
    assert_indicators_ready,
    prepare_indicators,
)
from src.strategy.long_strategy import LongTrendStrategy
from src.strategy.short_strategy import ShortTrendStrategy
from src.strategy.types import ChopFilterParams, EntrySignal, StrategyParams, TrailingStopParams
from src.utils.exceptions import ConfigError, DataIntegrityError
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_augmented(
    n: int,
    *,
    wma_fast: list[float] | None = None,
    wma_slow: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
    bb_lower: list[float] | None = None,
    bb_upper: list[float] | None = None,
) -> pd.DataFrame:
    """構造已含全部指標欄的 DataFrame，用於直接測試策略邏輯。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    if close is None:
        close = [100.0] * n
    if wma_fast is None:
        wma_fast = close[:]
    if wma_slow is None:
        wma_slow = close[:]
    if high is None:
        high = [c + 1.0 for c in close]
    if low is None:
        low = [c - 1.0 for c in close]
    if bb_lower is None:
        bb_lower = [c - 5.0 for c in close]
    if bb_upper is None:
        bb_upper = [c + 5.0 for c in close]

    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": [1.0] * n,
            "wma_fast": wma_fast,
            "wma_slow": wma_slow,
            "bb_middle": close,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# StrategyParams / TrailingStopParams
# --------------------------------------------------------------------------- #

class TestStrategyParams:
    def test_defaults(self) -> None:
        p = StrategyParams()
        assert p.wma_fast == 2
        assert p.wma_slow == 4
        # nested trailing
        assert p.trailing.swing_lookback == 4
        assert p.trailing.bollinger_period == 20
        assert p.trailing.stage2_normal_trigger_r == 1.2

    def test_immutable(self) -> None:
        p = StrategyParams()
        with pytest.raises(Exception):
            p.wma_fast = 5  # type: ignore[misc]

    def test_invalid_wma_order(self) -> None:
        with pytest.raises(ConfigError):
            StrategyParams(wma_fast=4, wma_slow=4)

    def test_warmup_includes_bollinger_period(self) -> None:
        # 20 (BB period) > 4 (wma_slow) → warmup = 20 + 3
        p = StrategyParams()
        assert p.warmup_bars == 20 + 3


class TestTrailingStopParams:
    def test_invalid_negative(self) -> None:
        with pytest.raises(ConfigError):
            TrailingStopParams(stage2_buffer_r=-0.1)

    def test_stage3_must_be_at_least_stage2(self) -> None:
        with pytest.raises(ConfigError, match="stage3"):
            TrailingStopParams(stage2_normal_trigger_r=2.0, stage3_normal_trigger_r=1.0)

    def test_invalid_bollinger_period(self) -> None:
        with pytest.raises(ConfigError):
            TrailingStopParams(bollinger_period=1)


# --------------------------------------------------------------------------- #
# Indicators ready guard
# --------------------------------------------------------------------------- #

class TestIndicatorsReady:
    def test_passes_with_all_columns(self) -> None:
        df = make_augmented(30)
        assert_indicators_ready(df)

    def test_raises_on_missing_bollinger(self) -> None:
        df = make_augmented(30).drop(columns=["bb_lower"])
        with pytest.raises(DataIntegrityError, match="bb_lower"):
            assert_indicators_ready(df)


# --------------------------------------------------------------------------- #
# Long strategy: entry + Stage 1 swing stop
# --------------------------------------------------------------------------- #

class TestLongEntry:
    def _build_setup(self, t: int = 25, n: int = 30) -> pd.DataFrame:
        close = [100.0] * n
        close[t - 3] = 98.0
        close[t - 2] = 99.0
        close[t - 1] = 100.5
        close[t] = 101.0

        wma_fast = [100.0] * n
        wma_slow = [100.0] * n
        wma_fast[t - 1] = 99.5
        wma_slow[t - 1] = 100.0
        wma_fast[t] = 100.8
        wma_slow[t] = 100.5

        # 給 low 一個變動序列才能驗證 swing low 計算
        # 我們用 swing_lookback=4，要看 [t-3..t]
        low = [99.0] * n
        low[t - 3] = 97.5
        low[t - 2] = 98.2
        low[t - 1] = 98.5
        low[t] = 99.5
        # 其他 bar 保持 99.0，避免污染前 N 根之外的視窗

        return make_augmented(
            n, close=close, wma_fast=wma_fast,
            wma_slow=wma_slow, low=low,
        )

    def test_emits_signal_when_all_conditions_met(self) -> None:
        # 把 warmup 拉小才能在 t=25 驗證
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        # warmup = max(4, 10, 4) + 3 = 13，t=25 OK
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30)

        sig = strat.detect_entry(df, 25)
        assert isinstance(sig, EntrySignal)
        assert sig.direction is Direction.LONG
        # Stage 1 stop = min(low[22..25]) × (1 - 0.0003) = 97.5 × 0.9997
        assert sig.initial_stop == pytest.approx(97.5 * (1 - 0.0003))

    def test_no_signal_at_warmup(self) -> None:
        strat = LongTrendStrategy(StrategyParams())
        df = self._build_setup(t=25, n=30)
        # 預設 warmup = 23，bar 5 應為 None
        assert strat.detect_entry(df, 5) is None

    def test_rejects_when_no_crossover(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30).copy()
        df.loc[df.index[24], "wma_fast"] = 100.5
        df.loc[df.index[24], "wma_slow"] = 100.0  # fast > slow already
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_structure_fails_t2(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30).copy()
        df.loc[df.index[23], "close"] = 102.0
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_initial_stop_above_close(self) -> None:
        # 病態場景：swing_low 高於收盤價 → 不安全進場
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30).copy()
        # swing window low 全部高於 close
        for i in range(22, 26):
            df.iloc[i, df.columns.get_loc("low")] = 200.0
        assert strat.detect_entry(df, 25) is None

    def test_swing_lookback_param_respected(self) -> None:
        # swing_lookback=2 → 只看 [t-1, t]
        params = StrategyParams(
            trailing=TrailingStopParams(
                bollinger_period=10, swing_lookback=2,
            )
        )
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30)
        sig = strat.detect_entry(df, 25)
        assert sig is not None
        # min(low[24..25]) = min(98.5, 99.5) = 98.5
        assert sig.initial_stop == pytest.approx(98.5 * (1 - 0.0003))


# --------------------------------------------------------------------------- #
# Short strategy: 鏡像 + 高點 swing
# --------------------------------------------------------------------------- #

class TestShortEntry:
    def _build_setup(self, t: int = 25, n: int = 30) -> pd.DataFrame:
        close = [100.0] * n
        close[t - 3] = 102.0
        close[t - 2] = 101.0
        close[t - 1] = 100.5
        close[t] = 99.0

        wma_fast = [100.0] * n
        wma_slow = [100.0] * n
        wma_fast[t - 1] = 100.5
        wma_slow[t - 1] = 100.0
        wma_fast[t] = 99.5
        wma_slow[t] = 100.0

        high = [101.0] * n
        high[t - 3] = 102.5
        high[t - 2] = 101.8
        high[t - 1] = 101.5
        high[t] = 100.5

        return make_augmented(
            n, close=close, wma_fast=wma_fast,
            wma_slow=wma_slow, high=high,
        )

    def test_emits_signal_when_all_conditions_met(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = ShortTrendStrategy(params)
        df = self._build_setup(t=25, n=30)

        sig = strat.detect_entry(df, 25)
        assert isinstance(sig, EntrySignal)
        assert sig.direction is Direction.SHORT
        # Stage 1 stop = max(high[22..25]) × (1 + 0.0003) = 102.5 × 1.0003
        assert sig.initial_stop == pytest.approx(102.5 * (1 + 0.0003))


# --------------------------------------------------------------------------- #
# Look-ahead invariant
# --------------------------------------------------------------------------- #

class TestNoLookahead:
    def test_long_detect_entry_unchanged_by_future(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = LongTrendStrategy(params)
        clean = TestLongEntry()._build_setup(t=25, n=30)
        # 污染未來指標欄
        poisoned = clean.copy()
        for col in REQUIRED_INDICATOR_COLUMNS:
            poisoned.iloc[26:, poisoned.columns.get_loc(col)] = np.nan

        a = strat.detect_entry(clean, 25)
        b = strat.detect_entry(poisoned, 25)
        assert a is not None and b is not None
        assert a.initial_stop == pytest.approx(b.initial_stop)


# --------------------------------------------------------------------------- #
# Integration: prepare_indicators
# --------------------------------------------------------------------------- #

class TestIntegration:
    def test_prepare_indicators_adds_required_columns(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        rng = np.random.default_rng(0)
        close = 100 + np.cumsum(rng.standard_normal(100))
        df = pd.DataFrame(
            {
                "open": close + rng.standard_normal(100) * 0.1,
                "high": close + np.abs(rng.standard_normal(100)),
                "low": close - np.abs(rng.standard_normal(100)),
                "close": close,
                "volume": rng.uniform(1, 10, 100),
            },
            index=idx,
        )
        df["open"] = np.clip(df["open"], df["low"], df["high"])
        df["close"] = np.clip(df["close"], df["low"], df["high"])

        out = prepare_indicators(df, StrategyParams())
        for col in REQUIRED_INDICATOR_COLUMNS:
            assert col in out.columns

    def test_strategy_runs_on_prepared_df(self) -> None:
        idx = pd.date_range("2024-01-01", periods=100, freq="5min")
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.standard_normal(100))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + np.abs(rng.standard_normal(100)),
                "low": close - np.abs(rng.standard_normal(100)),
                "close": close,
                "volume": rng.uniform(1, 10, 100),
            },
            index=idx,
        )
        df["open"] = np.clip(df["open"], df["low"], df["high"])

        augmented = prepare_indicators(df, StrategyParams())
        strat = LongTrendStrategy(StrategyParams())
        for i in range(len(augmented)):
            strat.detect_entry(augmented, i)


class TestChopFilter:
    """chop_filter gate：BBW_rank / ATR_rank / ADX 三條件 AND。"""

    def _augment_with_chop(
        self, n: int, *, bbw_rank: float, atr_rank: float, adx: float,
    ) -> pd.DataFrame:
        df = make_augmented(n)
        df["chop_bbw_rank"] = bbw_rank
        df["chop_atr_rank"] = atr_rank
        df["chop_adx"] = adx
        return df

    def _entry_setup(self, n: int = 30, t: int = 25) -> dict:
        close = [100.0] * n
        close[t - 3] = 98.0
        close[t - 2] = 99.0
        close[t - 1] = 100.5
        close[t] = 101.0
        wma_fast = [100.0] * n
        wma_slow = [100.0] * n
        wma_fast[t - 1] = 99.5
        wma_slow[t - 1] = 100.0
        wma_fast[t] = 100.8
        wma_slow[t] = 100.5
        low = [99.0] * n
        low[t - 3] = 97.5
        low[t - 2] = 98.2
        low[t - 1] = 98.5
        low[t] = 99.5
        return {"close": close, "wma_fast": wma_fast, "wma_slow": wma_slow, "low": low}

    def _build_params(self, **chop_kw) -> StrategyParams:
        # 把 chop_filter 暖機相關週期全部壓小到 < t=25，讓測試聚焦在 gate 邏輯
        kw = dict(
            enabled=True, rank_window=10,
            bb_period=5, atr_period=5, adx_period=5,
            **chop_kw,
        )
        return StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
            chop_filter=ChopFilterParams(**kw),
        )

    def test_passes_when_all_above_thresholds(self) -> None:
        params = self._build_params(
            bbw_rank_min=40.0, atr_rank_min=40.0, adx_min=20.0,
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        df["chop_bbw_rank"] = 50.0
        df["chop_atr_rank"] = 50.0
        df["chop_adx"] = 25.0
        sig = LongTrendStrategy(params).detect_entry(df, 25)
        assert sig is not None

    def test_blocks_when_bbw_below(self) -> None:
        params = self._build_params(
            bbw_rank_min=40.0, atr_rank_min=40.0, adx_min=20.0,
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        df["chop_bbw_rank"] = 30.0   # < 40
        df["chop_atr_rank"] = 50.0
        df["chop_adx"] = 25.0
        assert LongTrendStrategy(params).detect_entry(df, 25) is None

    def test_blocks_when_atr_below(self) -> None:
        params = self._build_params(
            bbw_rank_min=40.0, atr_rank_min=40.0, adx_min=20.0,
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        df["chop_bbw_rank"] = 50.0
        df["chop_atr_rank"] = 30.0   # < 40
        df["chop_adx"] = 25.0
        assert LongTrendStrategy(params).detect_entry(df, 25) is None

    def test_blocks_when_adx_below(self) -> None:
        params = self._build_params(
            bbw_rank_min=40.0, atr_rank_min=40.0, adx_min=20.0,
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        df["chop_bbw_rank"] = 50.0
        df["chop_atr_rank"] = 50.0
        df["chop_adx"] = 15.0   # < 20
        assert LongTrendStrategy(params).detect_entry(df, 25) is None

    def test_blocks_on_nan_warmup(self) -> None:
        params = self._build_params(
            bbw_rank_min=40.0, atr_rank_min=40.0, adx_min=20.0,
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        df["chop_bbw_rank"] = float("nan")
        df["chop_atr_rank"] = 50.0
        df["chop_adx"] = 25.0
        assert LongTrendStrategy(params).detect_entry(df, 25) is None

    def test_disabled_skips_gate(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
            chop_filter=ChopFilterParams(enabled=False),
        )
        setup = self._entry_setup()
        df = make_augmented(30, **setup)
        # 沒有 chop_* 欄位也應通過（gate 被略過）
        sig = LongTrendStrategy(params).detect_entry(df, 25)
        assert sig is not None

    def test_prepare_indicators_populates_chop_cols(self) -> None:
        params = StrategyParams(
            chop_filter=ChopFilterParams(enabled=True, rank_window=20)
        )
        idx = pd.date_range("2024-01-01", periods=300, freq="5min")
        rng = np.random.default_rng(1)
        close = 100 + np.cumsum(rng.standard_normal(300))
        df = pd.DataFrame({
            "open": close,
            "high": close + np.abs(rng.standard_normal(300)),
            "low": close - np.abs(rng.standard_normal(300)),
            "close": close,
            "volume": rng.uniform(1, 10, 300),
        }, index=idx)
        df["open"] = np.clip(df["open"], df["low"], df["high"])
        out = prepare_indicators(df, params)
        for col in ("chop_adx", "chop_bbw_rank", "chop_atr_rank"):
            assert col in out.columns
        # 暖機後值應落在有意義範圍
        assert (out["chop_bbw_rank"].dropna() <= 100).all()
        assert (out["chop_atr_rank"].dropna() <= 100).all()
        assert (out["chop_adx"].dropna() >= 0).all()
