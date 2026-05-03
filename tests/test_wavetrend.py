"""WaveTrend 單元測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.wavetrend import wavetrend
from src.utils.exceptions import DataIntegrityError


def _make_ohlc(n: int, base: float = 100.0, noise_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(noise_seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = base + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    open_ = close + rng.standard_normal(n) * 0.1
    open_ = np.clip(open_, low, high)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": [1.0] * n},
        index=idx,
    )


class TestWaveTrendBasics:
    def test_returns_wt1_wt2(self) -> None:
        df = _make_ohlc(50)
        out = wavetrend(df, n1=10, n2=21)
        assert "wt1" in out.columns
        assert "wt2" in out.columns

    def test_warmup_nan_count(self) -> None:
        # n1=10, n2=21 → wt1 暖機約 n1+n2-1 ≈ 29
        # wt2 = SMA(wt1, 4) → 多 3 根暖機
        df = _make_ohlc(60)
        out = wavetrend(df, n1=10, n2=21, signal_period=4)
        # 前若干根應為 NaN（無法精確算因 ewm 用 min_periods）
        assert out["wt1"].isna().any()
        # 末段應有有效值
        assert out["wt1"].iloc[-1] == pytest.approx(out["wt1"].iloc[-1])  # not NaN

    def test_invalid_periods(self) -> None:
        df = _make_ohlc(50)
        with pytest.raises(DataIntegrityError):
            wavetrend(df, n1=0)
        with pytest.raises(DataIntegrityError):
            wavetrend(df, n1=10, n2=-1)
        with pytest.raises(DataIntegrityError):
            wavetrend(df, signal_period=0)

    def test_empty_df(self) -> None:
        idx = pd.DatetimeIndex([])
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=idx,
        )
        out = wavetrend(df)
        assert "wt1" in out.columns
        assert len(out) == 0


class TestWaveTrendValueRange:
    """WaveTrend 值通常落在 [-200, +200]，做粗略 sanity 驗證。"""

    def test_typical_range(self) -> None:
        df = _make_ohlc(500, noise_seed=42)
        out = wavetrend(df)
        valid = out["wt1"].dropna()
        assert valid.abs().max() < 500  # very loose upper bound


class TestWaveTrendEdgeCases:
    def test_constant_price_handled(self) -> None:
        """常數價格 → d=0 → ci 為 NaN（不 raise）。"""
        idx = pd.date_range("2024-01-01", periods=50, freq="5min")
        df = pd.DataFrame(
            {"open": [100.0] * 50, "high": [100.0] * 50,
             "low": [100.0] * 50, "close": [100.0] * 50,
             "volume": [1.0] * 50},
            index=idx,
        )
        out = wavetrend(df)
        # wt1 整段為 NaN（因 d=0）
        assert out["wt1"].isna().all()


class TestWaveTrendNoLookahead:
    def test_truncation_invariant(self) -> None:
        """截斷重算應與原序列前段一致 → 無 look-ahead。"""
        df = _make_ohlc(200, noise_seed=7)
        full = wavetrend(df)
        for n in (50, 100, 199):
            partial = wavetrend(df.iloc[:n])
            np.testing.assert_array_equal(
                full["wt1"].iloc[:n].fillna(-999).to_numpy(),
                partial["wt1"].fillna(-999).to_numpy(),
                err_msg=f"wt1 look-ahead at n={n}",
            )
            np.testing.assert_array_equal(
                full["wt2"].iloc[:n].fillna(-999).to_numpy(),
                partial["wt2"].fillna(-999).to_numpy(),
                err_msg=f"wt2 look-ahead at n={n}",
            )
