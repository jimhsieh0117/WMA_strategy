"""resampler 單元測試。

重點：
1. 聚合規則正確（open=first, high=max, low=min, close=last, volume=sum）
2. 不完整尾段被丟棄
3. 非法輸入 raise
4. 1m → 1m 為識別運算（identity）
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.resampler import SUPPORTED_TIMEFRAMES, resample
from src.utils.exceptions import ConfigError, DataIntegrityError


class TestResampler:
    def test_5m_aggregation(self, minute_ohlcv: pd.DataFrame) -> None:
        # 30 根 1m → 6 根 5m
        out = resample(minute_ohlcv, "5m")
        assert len(out) == 6

        # 第一根 5m: 1m bar[0..4]
        first = out.iloc[0]
        src = minute_ohlcv.iloc[:5]
        assert first["open"] == src["open"].iloc[0]
        assert first["high"] == src["high"].max()
        assert first["low"] == src["low"].min()
        assert first["close"] == src["close"].iloc[-1]
        assert first["volume"] == src["volume"].sum()

    def test_label_is_left(self, minute_ohlcv: pd.DataFrame) -> None:
        out = resample(minute_ohlcv, "5m")
        # 第一根 5m 標籤應為 1m bar[0] 的時間（時段起點）
        assert out.index[0] == minute_ohlcv.index[0]

    def test_drops_incomplete_tail(self) -> None:
        # 7 根 1m → 1 根完整 5m + 2 根不足的尾段，應只剩 1 根
        idx = pd.date_range("2024-01-01", periods=7, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100] * 7, "high": [101] * 7, "low": [99] * 7,
                "close": [100] * 7, "volume": [1.0] * 7,
            },
            index=idx,
        )
        out = resample(df, "5m")
        assert len(out) == 1
        assert out.index[0] == idx[0]

    def test_identity_1m(self, minute_ohlcv: pd.DataFrame) -> None:
        out = resample(minute_ohlcv, "1m")
        pd.testing.assert_frame_equal(out, minute_ohlcv)

    def test_invalid_timeframe_raises(self, minute_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(ConfigError, match="unsupported"):
            resample(minute_ohlcv, "7m")

    def test_non_1m_source_raises(self) -> None:
        # 構造 5m 間距的資料丟給 resampler → 應拒絕
        idx = pd.date_range("2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "open": [100] * 10, "high": [101] * 10, "low": [99] * 10,
                "close": [100] * 10, "volume": [1.0] * 10,
            },
            index=idx,
        )
        with pytest.raises(DataIntegrityError, match="1m source"):
            resample(df, "15m")

    def test_all_timeframes_work(self, minute_ohlcv: pd.DataFrame) -> None:
        # 30 根 1m 對所有 timeframe 都應產生非空結果（除了 4H 因為資料不夠 1 根）
        for tf in SUPPORTED_TIMEFRAMES:
            if tf in ("1H", "4H"):
                continue  # 30 分鐘資料不夠
            out = resample(minute_ohlcv, tf)
            assert len(out) > 0, f"{tf} produced empty"

    def test_4h_insufficient_data_raises(self, minute_ohlcv: pd.DataFrame) -> None:
        # 30 分鐘資料無法形成完整 4H bar
        with pytest.raises(DataIntegrityError, match="empty"):
            resample(minute_ohlcv, "4H")
