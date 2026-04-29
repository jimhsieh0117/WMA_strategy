"""indicators (HA / WMA / ATR) 單元測試。

重點：
1. 數值正確性（與手算對照）
2. Look-ahead invariant：截斷重算 == 原序列前段
3. 邊界條件：空輸入、單根、暖機期不足、非法週期
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.atr import atr, true_range
from src.indicators.heikin_ashi import HA_COLUMNS, compute_ha
from src.indicators.wma import wma
from src.utils.exceptions import DataIntegrityError


class TestHeikinAshi:
    def test_basic_values(self, simple_ohlcv: pd.DataFrame) -> None:
        ha = compute_ha(simple_ohlcv)
        # HA_Close = (O+H+L+C)/4
        assert ha["ha_close"].iloc[0] == pytest.approx((100 + 102 + 99 + 101) / 4)
        assert ha["ha_close"].iloc[1] == pytest.approx((101 + 103 + 100 + 102) / 4)

    def test_ha_open_seed(self, simple_ohlcv: pd.DataFrame) -> None:
        ha = compute_ha(simple_ohlcv)
        # 首根：HA_Open = (Open + Close) / 2
        assert ha["ha_open"].iloc[0] == pytest.approx((100 + 101) / 2)

    def test_ha_open_recursion(self, simple_ohlcv: pd.DataFrame) -> None:
        ha = compute_ha(simple_ohlcv)
        # HA_Open[t] = (HA_Open[t-1] + HA_Close[t-1]) / 2
        for t in range(1, len(ha)):
            expected = (ha["ha_open"].iloc[t - 1] + ha["ha_close"].iloc[t - 1]) / 2
            assert ha["ha_open"].iloc[t] == pytest.approx(expected)

    def test_ha_high_low_envelope(self, random_ohlcv: pd.DataFrame) -> None:
        ha = compute_ha(random_ohlcv)
        # HA_High >= max(High, HA_Open, HA_Close)
        assert (ha["ha_high"] >= ha["high"]).all()
        assert (ha["ha_high"] >= ha["ha_open"]).all()
        assert (ha["ha_high"] >= ha["ha_close"]).all()
        # HA_Low <= min(Low, HA_Open, HA_Close)
        assert (ha["ha_low"] <= ha["low"]).all()
        assert (ha["ha_low"] <= ha["ha_open"]).all()
        assert (ha["ha_low"] <= ha["ha_close"]).all()

    def test_no_lookahead(self, random_ohlcv: pd.DataFrame) -> None:
        """截斷重算應與原序列前段完全一致 → 證明無 look-ahead。"""
        full = compute_ha(random_ohlcv)
        for n in (10, 50, 100, 199):
            partial = compute_ha(random_ohlcv.iloc[:n])
            for col in HA_COLUMNS:
                np.testing.assert_array_equal(
                    full[col].iloc[:n].to_numpy(),
                    partial[col].to_numpy(),
                    err_msg=f"look-ahead detected in {col} at n={n}",
                )

    def test_empty_input(self) -> None:
        idx = pd.DatetimeIndex([])
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=idx,
        )
        ha = compute_ha(df)
        for col in HA_COLUMNS:
            assert col in ha.columns
            assert len(ha[col]) == 0

    def test_missing_column_raises(self, simple_ohlcv: pd.DataFrame) -> None:
        bad = simple_ohlcv.drop(columns=["close"])
        with pytest.raises(DataIntegrityError, match="missing"):
            compute_ha(bad)

    def test_non_datetime_index_raises(self, simple_ohlcv: pd.DataFrame) -> None:
        bad = simple_ohlcv.reset_index(drop=True)
        with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
            compute_ha(bad)

    def test_inconsistent_ohlc_raises(self, simple_ohlcv: pd.DataFrame) -> None:
        bad = simple_ohlcv.copy()
        bad.loc[bad.index[0], "high"] = 50  # high < low
        with pytest.raises(DataIntegrityError, match="OHLC consistency"):
            compute_ha(bad)


class TestWMA:
    def test_basic_values(self) -> None:
        s = pd.Series(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            index=pd.date_range("2024-01-01", periods=5, freq="5min"),
        )
        # WMA(3)[2] = (1*10 + 2*20 + 3*30) / (1+2+3) = 140/6
        out = wma(s, 3)
        assert pd.isna(out.iloc[0])
        assert pd.isna(out.iloc[1])
        assert out.iloc[2] == pytest.approx(140 / 6)
        # WMA(3)[3] = (1*20 + 2*30 + 3*40) / 6 = 200/6
        assert out.iloc[3] == pytest.approx(200 / 6)
        # WMA(3)[4] = (1*30 + 2*40 + 3*50) / 6 = 260/6
        assert out.iloc[4] == pytest.approx(260 / 6)

    def test_period_one_is_identity(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="5min"))
        np.testing.assert_array_equal(wma(s, 1).to_numpy(), s.to_numpy())

    def test_short_series_all_nan(self) -> None:
        s = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2, freq="5min"))
        out = wma(s, 5)
        assert out.isna().all()
        assert len(out) == 2

    def test_warmup_nan_count(self) -> None:
        s = pd.Series(
            np.arange(10, dtype=np.float64),
            index=pd.date_range("2024-01-01", periods=10, freq="5min"),
        )
        out = wma(s, 4)
        # 前 period - 1 = 3 根為 NaN
        assert out.iloc[:3].isna().all()
        assert out.iloc[3:].notna().all()

    def test_invalid_period_raises(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="5min"))
        with pytest.raises(DataIntegrityError):
            wma(s, 0)
        with pytest.raises(DataIntegrityError):
            wma(s, -1)
        with pytest.raises(DataIntegrityError):
            wma(s, 2.5)  # type: ignore[arg-type]
        with pytest.raises(DataIntegrityError):
            wma(s, True)  # bool 不算 int

    def test_no_lookahead(self, random_ohlcv: pd.DataFrame) -> None:
        s = random_ohlcv["close"]
        full = wma(s, 4)
        for n in (10, 50, 100, 199):
            partial = wma(s.iloc[:n], 4)
            np.testing.assert_array_equal(
                full.iloc[:n].fillna(-1).to_numpy(),
                partial.fillna(-1).to_numpy(),
                err_msg=f"look-ahead detected in WMA at n={n}",
            )


class TestATR:
    def test_tr_first_bar(self, simple_ohlcv: pd.DataFrame) -> None:
        tr = true_range(simple_ohlcv)
        # 首根無 prev_close → TR = H - L
        assert tr.iloc[0] == pytest.approx(102 - 99)

    def test_tr_subsequent(self, simple_ohlcv: pd.DataFrame) -> None:
        tr = true_range(simple_ohlcv)
        # TR[1] = max(H-L, |H-prev_C|, |L-prev_C|)
        # bar[1]: H=103, L=100, prev_C=101 → max(3, 2, 1) = 3
        assert tr.iloc[1] == pytest.approx(3)

    def test_atr_value(self, simple_ohlcv: pd.DataFrame) -> None:
        # 構造的 K 線每根 H-L=3、相鄰收盤差小 → TR 全為 3 → ATR=3
        result = atr(simple_ohlcv, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(3.0)
        assert result.iloc[-1] == pytest.approx(3.0)

    def test_atr_warmup(self, simple_ohlcv: pd.DataFrame) -> None:
        result = atr(simple_ohlcv, 5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()

    def test_invalid_period_raises(self, simple_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(DataIntegrityError):
            atr(simple_ohlcv, 0)
        with pytest.raises(DataIntegrityError):
            atr(simple_ohlcv, -3)

    def test_no_lookahead(self, random_ohlcv: pd.DataFrame) -> None:
        full = atr(random_ohlcv, 14)
        for n in (20, 50, 100, 199):
            partial = atr(random_ohlcv.iloc[:n], 14)
            np.testing.assert_array_equal(
                full.iloc[:n].fillna(-1).to_numpy(),
                partial.fillna(-1).to_numpy(),
                err_msg=f"look-ahead detected in ATR at n={n}",
            )

    def test_single_bar(self) -> None:
        idx = pd.date_range("2024-01-01", periods=1, freq="5min")
        df = pd.DataFrame(
            {"open": [100], "high": [102], "low": [99], "close": [101], "volume": [1]},
            index=idx,
        )
        tr = true_range(df)
        assert tr.iloc[0] == pytest.approx(3.0)
        result = atr(df, 1)
        assert result.iloc[0] == pytest.approx(3.0)
