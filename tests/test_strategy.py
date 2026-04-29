"""Strategy 訊號層單元測試。

策略不依賴 broker，可用構造的 augmented DataFrame 直接測試 detect_entry / trailing stop。
重點：
1. 進場條件正/反例（C1 / C2 各別觸發失敗）
2. 初始止損 / 拖曳止損候選值正確性
3. Look-ahead invariant（future-bar poisoning）
4. 暖機期 / NaN 防護
5. 與 prepare_indicators 整合（raw OHLCV → 訊號）
"""

from __future__ import annotations

import math

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
from src.strategy.types import EntrySignal, StrategyParams
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
    high: list[float] | None = None,
    low: list[float] | None = None,
    atr_val: float = 1.0,
) -> pd.DataFrame:
    """構造已含指標欄的 DataFrame，用於直接測試策略邏輯。

    未指定的欄位用合理預設填滿；OHLC 保持一致性以通過 validate_ohlc。
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    if ha_close is None:
        ha_close = [100.0] * n
    if ha_wma_fast is None:
        ha_wma_fast = ha_close[:]
    if ha_wma_slow is None:
        ha_wma_slow = ha_close[:]
    if high is None:
        high = [c + 1.0 for c in ha_close]
    if low is None:
        low = [c - 1.0 for c in ha_close]

    return pd.DataFrame(
        {
            "open": ha_close,
            "high": high,
            "low": low,
            "close": ha_close,
            "volume": [1.0] * n,
            "ha_open": ha_close,
            "ha_high": [c + 0.5 for c in ha_close],
            "ha_low": [c - 0.5 for c in ha_close],
            "ha_close": ha_close,
            "ha_wma_fast": ha_wma_fast,
            "ha_wma_slow": ha_wma_slow,
            "atr": [atr_val] * n,
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# StrategyParams
# --------------------------------------------------------------------------- #

class TestStrategyParams:
    def test_defaults(self) -> None:
        p = StrategyParams()
        assert p.wma_fast == 2
        assert p.wma_slow == 4
        assert p.atr_period == 14
        assert p.atr_multiplier == 2.0
        assert p.atr_lookback == 14

    def test_immutable(self) -> None:
        p = StrategyParams()
        with pytest.raises(Exception):
            p.wma_fast = 5  # type: ignore[misc]

    def test_invalid_wma_order(self) -> None:
        with pytest.raises(ConfigError, match="wma_fast"):
            StrategyParams(wma_fast=4, wma_slow=4)
        with pytest.raises(ConfigError, match="wma_fast"):
            StrategyParams(wma_fast=5, wma_slow=4)

    def test_invalid_zero_period(self) -> None:
        with pytest.raises(ConfigError):
            StrategyParams(wma_fast=0)
        with pytest.raises(ConfigError):
            StrategyParams(atr_period=0)
        with pytest.raises(ConfigError):
            StrategyParams(atr_lookback=0)

    def test_invalid_multiplier(self) -> None:
        with pytest.raises(ConfigError):
            StrategyParams(atr_multiplier=0)
        with pytest.raises(ConfigError):
            StrategyParams(atr_multiplier=-1.0)

    def test_warmup_bars(self) -> None:
        p = StrategyParams(wma_fast=2, wma_slow=4, atr_period=14, atr_lookback=14)
        # max(4, 14, 14) + 3 = 17
        assert p.warmup_bars == 17


# --------------------------------------------------------------------------- #
# Indicators ready guard
# --------------------------------------------------------------------------- #

class TestIndicatorsReady:
    def test_passes_with_all_columns(self) -> None:
        df = make_augmented(20)
        assert_indicators_ready(df)  # no raise

    def test_raises_on_missing_column(self) -> None:
        df = make_augmented(20).drop(columns=["atr"])
        with pytest.raises(DataIntegrityError, match="atr"):
            assert_indicators_ready(df)


# --------------------------------------------------------------------------- #
# Long strategy
# --------------------------------------------------------------------------- #

class TestLongEntry:
    """構造合法進場場景並驗證 long 訊號發出。"""

    def _build_long_setup(self, t: int = 25, n: int = 30) -> pd.DataFrame:
        """建立一個在 bar t 觸發 long 訊號的 augmented df。

        - HA_Close 在 [t-3, t-2, t-1, t] 為 [98, 99, 100, 101] → 結構成立
        - WMA fast/slow 在 t-1 為 fast<=slow，在 t 為 fast>slow → 金叉
        """
        ha_close = [100.0] * n
        ha_close[t - 3] = 98.0
        ha_close[t - 2] = 99.0
        ha_close[t - 1] = 100.5
        ha_close[t] = 101.0

        wma_fast = [100.0] * n
        wma_slow = [100.0] * n
        wma_fast[t - 1] = 99.5
        wma_slow[t - 1] = 100.0  # fast <= slow
        wma_fast[t] = 100.8
        wma_slow[t] = 100.5  # fast > slow

        # 給 high 一個變動序列才能驗證 highest 計算
        high = [102.0 + (i * 0.1) for i in range(n)]
        return make_augmented(
            n, ha_close=ha_close, ha_wma_fast=wma_fast,
            ha_wma_slow=wma_slow, high=high, atr_val=1.5,
        )

    def test_emits_signal_when_all_conditions_met(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)

        sig = strat.detect_entry(df, 25)
        assert isinstance(sig, EntrySignal)
        assert sig.direction is Direction.LONG
        assert sig.bar_index == 25
        assert sig.timestamp == df.index[25]
        assert sig.initial_stop > 0

    def test_no_signal_at_warmup(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)
        # warmup_bars = 17，bar_index 5 應為 None
        assert strat.detect_entry(df, 5) is None

    def test_rejects_when_no_crossover(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)
        # 改 t-1：fast 已經在上方 → 不算金叉
        df = df.copy()
        df.loc[df.index[24], "ha_wma_fast"] = 100.5
        df.loc[df.index[24], "ha_wma_slow"] = 100.0  # fast > slow already
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_structure_fails_t2(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)
        # 把 hc[t-2] 改高於 hc[t]
        df = df.copy()
        df.loc[df.index[23], "ha_close"] = 102.0  # > hc[t]=101
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_structure_fails_t3(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)
        df = df.copy()
        df.loc[df.index[22], "ha_close"] = 102.0
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_indicator_nan(self) -> None:
        params = StrategyParams()
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)
        df = df.copy()
        df.loc[df.index[25], "ha_wma_fast"] = float("nan")
        assert strat.detect_entry(df, 25) is None

    def test_initial_stop_value(self) -> None:
        """initial_stop = max(high[t-N+1..t]) - atr * multiplier"""
        params = StrategyParams(atr_lookback=5, atr_multiplier=2.0)
        strat = LongTrendStrategy(params)
        df = self._build_long_setup(t=25, n=30)

        sig = strat.detect_entry(df, 25)
        assert sig is not None
        # high[21..25] 中最高的；high = [102.0 + i*0.1 for i in range(30)]
        expected_high = max(102.0 + i * 0.1 for i in range(21, 26))
        expected_stop = expected_high - 1.5 * 2.0  # atr * multiplier
        assert sig.initial_stop == pytest.approx(expected_stop)


class TestLongTrailingStop:
    def test_candidate_value(self) -> None:
        params = StrategyParams(atr_lookback=5, atr_multiplier=2.0)
        strat = LongTrendStrategy(params)
        n = 20
        high = [100.0 + i for i in range(n)]
        df = make_augmented(n, high=high, atr_val=1.0)
        # bar_index = 10 → high[6..10] = [106..110] → max = 110
        candidate = strat.compute_trailing_stop_candidate(df, 10)
        assert candidate == pytest.approx(110.0 - 2.0)

    def test_returns_nan_during_warmup(self) -> None:
        params = StrategyParams(atr_lookback=10)
        strat = LongTrendStrategy(params)
        df = make_augmented(20, atr_val=1.0)
        # bar_index = 5 < lookback=10 → NaN
        assert math.isnan(strat.compute_trailing_stop_candidate(df, 5))

    def test_returns_nan_when_atr_nan(self) -> None:
        params = StrategyParams(atr_lookback=5)
        strat = LongTrendStrategy(params)
        df = make_augmented(20, atr_val=1.0).copy()
        df.loc[df.index[10], "atr"] = float("nan")
        assert math.isnan(strat.compute_trailing_stop_candidate(df, 10))


# --------------------------------------------------------------------------- #
# Short strategy (mirror)
# --------------------------------------------------------------------------- #

class TestShortEntry:
    def _build_short_setup(self, t: int = 25, n: int = 30) -> pd.DataFrame:
        # HA_Close 在 [t-3, t-2, t-1, t] 為 [102, 101, 100.5, 99] → 結構成立（下跌）
        ha_close = [100.0] * n
        ha_close[t - 3] = 102.0
        ha_close[t - 2] = 101.0
        ha_close[t - 1] = 100.5
        ha_close[t] = 99.0

        wma_fast = [100.0] * n
        wma_slow = [100.0] * n
        wma_fast[t - 1] = 100.5
        wma_slow[t - 1] = 100.0  # fast >= slow（前根）
        wma_fast[t] = 99.5
        wma_slow[t] = 100.0  # fast < slow（死叉）

        low = [98.0 - (i * 0.1) for i in range(n)]
        return make_augmented(
            n, ha_close=ha_close, ha_wma_fast=wma_fast,
            ha_wma_slow=wma_slow, low=low, atr_val=1.5,
        )

    def test_emits_signal_when_all_conditions_met(self) -> None:
        strat = ShortTrendStrategy(StrategyParams())
        df = self._build_short_setup(t=25, n=30)
        sig = strat.detect_entry(df, 25)
        assert isinstance(sig, EntrySignal)
        assert sig.direction is Direction.SHORT
        assert sig.bar_index == 25
        assert sig.initial_stop > 0

    def test_rejects_when_no_crossover(self) -> None:
        strat = ShortTrendStrategy(StrategyParams())
        df = self._build_short_setup(t=25, n=30).copy()
        df.loc[df.index[24], "ha_wma_fast"] = 99.0
        df.loc[df.index[24], "ha_wma_slow"] = 100.0  # fast already < slow
        assert strat.detect_entry(df, 25) is None

    def test_rejects_when_structure_fails(self) -> None:
        strat = ShortTrendStrategy(StrategyParams())
        df = self._build_short_setup(t=25, n=30).copy()
        df.loc[df.index[23], "ha_close"] = 98.0  # < hc[t]=99
        assert strat.detect_entry(df, 25) is None

    def test_initial_stop_value(self) -> None:
        params = StrategyParams(atr_lookback=5, atr_multiplier=2.0)
        strat = ShortTrendStrategy(params)
        df = self._build_short_setup(t=25, n=30)
        sig = strat.detect_entry(df, 25)
        assert sig is not None
        # low[21..25]，low = [98.0 - i*0.1] → 最低 = 98 - 25*0.1 = 95.5
        expected_low = min(98.0 - i * 0.1 for i in range(21, 26))
        expected_stop = expected_low + 1.5 * 2.0
        assert sig.initial_stop == pytest.approx(expected_stop)


class TestShortTrailingStop:
    def test_candidate_value(self) -> None:
        params = StrategyParams(atr_lookback=5, atr_multiplier=2.0)
        strat = ShortTrendStrategy(params)
        n = 20
        low = [100.0 - i for i in range(n)]
        df = make_augmented(n, low=low, atr_val=1.0)
        # bar_index = 10 → low[6..10] = [94..90] → min = 90
        candidate = strat.compute_trailing_stop_candidate(df, 10)
        assert candidate == pytest.approx(90.0 + 2.0)


# --------------------------------------------------------------------------- #
# Look-ahead invariant (critical guard)
# --------------------------------------------------------------------------- #

class TestNoLookahead:
    """以「污染未來資料」驗證策略只讀過去 + 當下。"""

    def _poison_future(self, df: pd.DataFrame, after_index: int) -> pd.DataFrame:
        out = df.copy()
        for col in REQUIRED_INDICATOR_COLUMNS:
            out.iloc[after_index + 1 :, out.columns.get_loc(col)] = np.nan
        return out

    def test_long_detect_entry_unchanged_by_future(self) -> None:
        strat = LongTrendStrategy(StrategyParams())
        clean = TestLongEntry()._build_long_setup(t=25, n=30)
        poisoned = self._poison_future(clean, 25)
        assert strat.detect_entry(clean, 25) is not None
        assert strat.detect_entry(poisoned, 25) is not None
        # initial_stop 應一致
        a = strat.detect_entry(clean, 25)
        b = strat.detect_entry(poisoned, 25)
        assert a is not None and b is not None
        assert a.initial_stop == pytest.approx(b.initial_stop)

    def test_long_trailing_unchanged_by_future(self) -> None:
        strat = LongTrendStrategy(StrategyParams(atr_lookback=5))
        n = 30
        high = [100.0 + i for i in range(n)]
        clean = make_augmented(n, high=high, atr_val=1.0)
        poisoned = self._poison_future(clean, 15)
        a = strat.compute_trailing_stop_candidate(clean, 15)
        b = strat.compute_trailing_stop_candidate(poisoned, 15)
        assert a == pytest.approx(b)

    def test_short_detect_entry_unchanged_by_future(self) -> None:
        strat = ShortTrendStrategy(StrategyParams())
        clean = TestShortEntry()._build_short_setup(t=25, n=30)
        poisoned = self._poison_future(clean, 25)
        a = strat.detect_entry(clean, 25)
        b = strat.detect_entry(poisoned, 25)
        assert a is not None and b is not None
        assert a.initial_stop == pytest.approx(b.initial_stop)


# --------------------------------------------------------------------------- #
# Integration: prepare_indicators + strategy
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
        # 確保 OHLC 一致
        df["open"] = np.clip(df["open"], df["low"], df["high"])
        df["close"] = np.clip(df["close"], df["low"], df["high"])

        out = prepare_indicators(df, StrategyParams())
        for col in REQUIRED_INDICATOR_COLUMNS:
            assert col in out.columns

    def test_strategy_runs_on_prepared_df(self) -> None:
        """端對端：raw → prepare → detect_entry 不爆。"""
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
        # 跑完不應 raise
        for i in range(len(augmented)):
            strat.detect_entry(augmented, i)
            strat.compute_trailing_stop_candidate(augmented, i)
