"""loader 單元測試。

涉及對外 IO，故同時跑 mock 測試（pytest tmp_path）與 real-data smoke test。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import OHLCV_COLUMNS, load_ohlcv
from src.utils.exceptions import DataIntegrityError

PPO_ETH_PARQUET = Path(
    "/Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/data/processed/ETHUSDT_1m.parquet"
)


class TestLoaderMock:
    def _write_parquet(self, tmp_path: Path, df: pd.DataFrame) -> Path:
        p = tmp_path / "test.parquet"
        df.to_parquet(p)
        return p

    def test_loads_basic(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5,
                "close": [100.5] * 5, "volume": [1.0] * 5,
            },
            index=idx,
        )
        p = self._write_parquet(tmp_path, df)
        out = load_ohlcv(p)
        assert len(out) == 5
        for col in OHLCV_COLUMNS:
            assert col in out.columns

    def test_filters_date_range(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10, "high": [101.0] * 10, "low": [99.0] * 10,
                "close": [100.5] * 10, "volume": [1.0] * 10,
            },
            index=idx,
        )
        p = self._write_parquet(tmp_path, df)
        out = load_ohlcv(p, start="2024-01-01 00:02:00", end="2024-01-01 00:05:00")
        assert len(out) == 4
        assert out.index[0] == pd.Timestamp("2024-01-01 00:02:00")
        assert out.index[-1] == pd.Timestamp("2024-01-01 00:05:00")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ohlcv("/no/such/file.parquet")

    def test_empty_range_raises(self, tmp_path: Path) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5, "high": [101.0] * 5, "low": [99.0] * 5,
                "close": [100.5] * 5, "volume": [1.0] * 5,
            },
            index=idx,
        )
        p = self._write_parquet(tmp_path, df)
        with pytest.raises(DataIntegrityError, match="no rows"):
            load_ohlcv(p, start="2030-01-01", end="2030-12-31")


@pytest.mark.skipif(
    not PPO_ETH_PARQUET.is_file(),
    reason="PPO_TradingModel ETHUSDT parquet not available on this machine",
)
class TestLoaderRealData:
    """smoke test 跑真實資料；若兄弟專案不存在自動跳過。"""

    def test_loads_real_eth_subset(self) -> None:
        df = load_ohlcv(PPO_ETH_PARQUET, start="2024-01-01", end="2024-01-02")
        assert len(df) > 0
        for col in OHLCV_COLUMNS:
            assert col in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
