"""Bollinger Band 單元測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.bollinger import bollinger_bands
from src.utils.exceptions import DataIntegrityError


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="5min")
    return pd.Series(values, index=idx, name="close")


class TestBollingerBasics:
    def test_warmup_nan(self) -> None:
        s = _series(list(range(30)))
        mid, up, lo = bollinger_bands(s, period=20)
        assert mid.iloc[:19].isna().all()
        assert mid.iloc[19:].notna().all()
        assert up.iloc[:19].isna().all()
        assert lo.iloc[:19].isna().all()

    def test_short_series_all_nan(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        mid, up, lo = bollinger_bands(s, period=20)
        assert mid.isna().all()

    def test_invalid_period(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        with pytest.raises(DataIntegrityError):
            bollinger_bands(s, period=0)
        with pytest.raises(DataIntegrityError):
            bollinger_bands(s, period=-5)

    def test_invalid_num_std(self) -> None:
        s = _series([1.0, 2.0, 3.0])
        with pytest.raises(DataIntegrityError):
            bollinger_bands(s, period=2, num_std=0)


class TestBollingerSMA:
    def test_against_manual(self) -> None:
        # period=3, simple MA, 計算 t=2 的值
        # close = [10, 20, 30] → mean=20, σ² = mean((10-20)^2 + 0 + (30-20)^2) = 200/3
        s = _series([10.0, 20.0, 30.0])
        mid, up, lo = bollinger_bands(s, period=3, num_std=2.0, ma_type="sma")
        assert mid.iloc[2] == pytest.approx(20.0)
        expected_sigma = np.sqrt(200.0 / 3)
        assert (up.iloc[2] - mid.iloc[2]) == pytest.approx(2 * expected_sigma)
        assert (mid.iloc[2] - lo.iloc[2]) == pytest.approx(2 * expected_sigma)


class TestBollingerWMA:
    def test_against_manual(self) -> None:
        # period=3, WMA, close = [10, 20, 30]
        # WMA[2] = (1*10 + 2*20 + 3*30) / 6 = 140/6
        # σ² = mean((10-140/6)^2 + (20-140/6)^2 + (30-140/6)^2) / 3
        s = _series([10.0, 20.0, 30.0])
        mid, up, lo = bollinger_bands(s, period=3, num_std=2.0, ma_type="wma")
        m = 140.0 / 6
        assert mid.iloc[2] == pytest.approx(m)

        expected_var = ((10 - m) ** 2 + (20 - m) ** 2 + (30 - m) ** 2) / 3
        expected_sigma = np.sqrt(expected_var)
        assert (up.iloc[2] - m) == pytest.approx(2 * expected_sigma)
        assert (m - lo.iloc[2]) == pytest.approx(2 * expected_sigma)


class TestBollingerNoLookahead:
    def test_truncation_invariant(self) -> None:
        rng = np.random.default_rng(0)
        s = _series(list(100 + np.cumsum(rng.standard_normal(200))))
        full_mid, full_up, full_lo = bollinger_bands(s, period=20)

        for n in (50, 100, 199):
            p_mid, p_up, p_lo = bollinger_bands(s.iloc[:n], period=20)
            np.testing.assert_array_equal(
                full_mid.iloc[:n].fillna(-1).to_numpy(),
                p_mid.fillna(-1).to_numpy(),
                err_msg=f"middle look-ahead at n={n}",
            )
            np.testing.assert_array_equal(
                full_up.iloc[:n].fillna(-1).to_numpy(),
                p_up.fillna(-1).to_numpy(),
            )
            np.testing.assert_array_equal(
                full_lo.iloc[:n].fillna(-1).to_numpy(),
                p_lo.fillna(-1).to_numpy(),
            )

    def test_constant_series_zero_band_width(self) -> None:
        # 常數序列 → σ=0 → upper=lower=middle
        s = _series([100.0] * 25)
        mid, up, lo = bollinger_bands(s, period=20)
        assert mid.iloc[24] == pytest.approx(100.0)
        assert up.iloc[24] == pytest.approx(100.0)
        assert lo.iloc[24] == pytest.approx(100.0)
