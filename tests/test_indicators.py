"""indicators (WMA / ATR) 單元測試。

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
from src.indicators.wma import wma
from src.utils.exceptions import DataIntegrityError


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
