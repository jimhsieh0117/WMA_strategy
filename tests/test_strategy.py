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
from src.strategy.types import EntrySignal, StrategyParams, TrailingStopParams
from src.utils.exceptions import ConfigError, DataIntegrityError
from src.utils.types import Direction


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_augmented(
    n: int,
    *,
    ha_close: list[float] | None = None,
    ha_wma_fast: list[float] | None = None,
    ha_wma_slow: list[float] | None = None,
    raw_wma_fast: list[float] | None = None,
    raw_wma_slow: list[float] | None = None,
    high: list[float] | None = None,
    low: list[float] | None = None,
    close: list[float] | None = None,
    bb_lower: list[float] | None = None,
    bb_upper: list[float] | None = None,
) -> pd.DataFrame:
    """構造已含全部指標欄的 DataFrame，用於直接測試策略邏輯。"""
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    if ha_close is None:
        ha_close = [100.0] * n
    if ha_wma_fast is None:
        ha_wma_fast = ha_close[:]
    if ha_wma_slow is None:
        ha_wma_slow = ha_close[:]
    if close is None:
        close = ha_close[:]
    # 若沒指定 raw WMA → 預設等於 close（與 HA 路線一樣的構造邏輯）
    if raw_wma_fast is None:
        raw_wma_fast = close[:]
    if raw_wma_slow is None:
        raw_wma_slow = close[:]
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
            "ha_open": ha_close,
            "ha_high": [c + 0.5 for c in ha_close],
            "ha_low": [c - 0.5 for c in ha_close],
            "ha_close": ha_close,
            "ha_wma_fast": ha_wma_fast,
            "ha_wma_slow": ha_wma_slow,
            "wma_fast": raw_wma_fast,
            "wma_slow": raw_wma_slow,
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
        assert p.entry_source == "ha"
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

    def test_entry_source_raw_accepted(self) -> None:
        p = StrategyParams(entry_source="raw")
        assert p.entry_source == "raw"

    def test_invalid_entry_source(self) -> None:
        with pytest.raises(ConfigError, match="entry_source"):
            StrategyParams(entry_source="bogus")  # type: ignore[arg-type]


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
        ha_close = [100.0] * n
        ha_close[t - 3] = 98.0
        ha_close[t - 2] = 99.0
        ha_close[t - 1] = 100.5
        ha_close[t] = 101.0

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
            n, ha_close=ha_close, ha_wma_fast=wma_fast,
            ha_wma_slow=wma_slow, low=low,
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
        df.loc[df.index[24], "ha_wma_fast"] = 100.5
        df.loc[df.index[24], "ha_wma_slow"] = 100.0  # fast > slow already
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_structure_fails_t2(self) -> None:
        params = StrategyParams(
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4)
        )
        strat = LongTrendStrategy(params)
        df = self._build_setup(t=25, n=30).copy()
        df.loc[df.index[23], "ha_close"] = 102.0
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
        ha_close = [100.0] * n
        ha_close[t - 3] = 102.0
        ha_close[t - 2] = 101.0
        ha_close[t - 1] = 100.5
        ha_close[t] = 99.0

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
            n, ha_close=ha_close, ha_wma_fast=wma_fast,
            ha_wma_slow=wma_slow, high=high,
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
# entry_source 切換：HA vs raw K 行為差異
# --------------------------------------------------------------------------- #

class TestEntrySourceSwitching:
    """構造一個情境：HA 路線符合進場條件、raw 路線不符合（或反之），
    驗證同一個 df 在不同 entry_source 下會產生不同訊號。"""

    def _build(self, t: int = 25, n: int = 30) -> pd.DataFrame:
        # HA 路線符合金叉 + 結構上升
        ha_close = [100.0] * n
        ha_close[t - 3] = 98.0
        ha_close[t - 2] = 99.0
        ha_close[t - 1] = 100.5
        ha_close[t] = 101.0

        ha_wma_fast = [100.0] * n
        ha_wma_slow = [100.0] * n
        ha_wma_fast[t - 1] = 99.5
        ha_wma_slow[t - 1] = 100.0
        ha_wma_fast[t] = 100.8
        ha_wma_slow[t] = 100.5

        # raw 路線**沒有**金叉：fast 一直 <= slow
        raw_wma_fast = [99.0] * n
        raw_wma_slow = [100.0] * n

        # close 也不符合條件 2（不是上升結構）
        close = [100.0] * n  # 全平
        # low 給足夠低讓 stop 計算合理
        low = [98.0] * n
        return make_augmented(
            n,
            ha_close=ha_close,
            ha_wma_fast=ha_wma_fast,
            ha_wma_slow=ha_wma_slow,
            raw_wma_fast=raw_wma_fast,
            raw_wma_slow=raw_wma_slow,
            close=close,
            low=low,
        )

    def test_ha_source_fires_raw_does_not(self) -> None:
        params_ha = StrategyParams(
            entry_source="ha",
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
        )
        params_raw = StrategyParams(
            entry_source="raw",
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
        )
        df = self._build(t=25, n=30)

        sig_ha = LongTrendStrategy(params_ha).detect_entry(df, 25)
        sig_raw = LongTrendStrategy(params_raw).detect_entry(df, 25)

        # HA 路線符合 → 出訊號；raw 路線不符合 → None
        assert sig_ha is not None
        assert "HA" in sig_ha.reason
        assert sig_raw is None

    def test_raw_source_fires_ha_does_not(self) -> None:
        # 反向構造：raw 路線符合金叉、HA 路線不符合
        n, t = 30, 25
        # raw 金叉：bar t-1 fast<=slow，bar t fast>slow
        raw_wma_fast = [100.0] * n
        raw_wma_slow = [100.0] * n
        raw_wma_fast[t - 1] = 99.5
        raw_wma_slow[t - 1] = 100.0
        raw_wma_fast[t] = 100.8
        raw_wma_slow[t] = 100.5
        # close 結構上升
        close = [100.0] * n
        close[t - 3] = 98.0
        close[t - 2] = 99.0
        close[t - 1] = 100.5
        close[t] = 101.0

        # HA 路線不交叉
        ha_wma_fast = [99.0] * n
        ha_wma_slow = [100.0] * n
        ha_close = [100.0] * n  # HA close 也平

        df = make_augmented(
            n, ha_close=ha_close,
            ha_wma_fast=ha_wma_fast, ha_wma_slow=ha_wma_slow,
            raw_wma_fast=raw_wma_fast, raw_wma_slow=raw_wma_slow,
            close=close, low=[98.0] * n,
        )

        params_ha = StrategyParams(
            entry_source="ha",
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
        )
        params_raw = StrategyParams(
            entry_source="raw",
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
        )

        assert LongTrendStrategy(params_ha).detect_entry(df, 25) is None
        sig_raw = LongTrendStrategy(params_raw).detect_entry(df, 25)
        assert sig_raw is not None
        assert "RAW" in sig_raw.reason

    def test_short_raw_source_works(self) -> None:
        # 鏡像測試：raw 死叉
        n, t = 30, 25
        raw_wma_fast = [100.0] * n
        raw_wma_slow = [100.0] * n
        raw_wma_fast[t - 1] = 100.5
        raw_wma_slow[t - 1] = 100.0
        raw_wma_fast[t] = 99.5
        raw_wma_slow[t] = 100.0
        close = [100.0] * n
        close[t - 3] = 102.0
        close[t - 2] = 101.0
        close[t - 1] = 100.5
        close[t] = 99.0

        # HA 路線不交叉
        ha_wma_fast = [101.0] * n
        ha_wma_slow = [100.0] * n
        ha_close = [100.0] * n

        df = make_augmented(
            n, ha_close=ha_close,
            ha_wma_fast=ha_wma_fast, ha_wma_slow=ha_wma_slow,
            raw_wma_fast=raw_wma_fast, raw_wma_slow=raw_wma_slow,
            close=close, high=[102.0] * n,
        )

        params_raw = StrategyParams(
            entry_source="raw",
            trailing=TrailingStopParams(bollinger_period=10, swing_lookback=4),
        )

        sig = ShortTrendStrategy(params_raw).detect_entry(df, 25)
        assert sig is not None
        assert sig.direction is Direction.SHORT
        assert "RAW" in sig.reason


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
