"""共用 fixtures。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_ohlcv() -> pd.DataFrame:
    """10 根遞增後遞減的 5m K 線，便於手算驗證指標。"""
    idx = pd.date_range("2024-01-01", periods=10, freq="5min")
    return pd.DataFrame(
        {
            "open":   [100, 101, 102, 103, 104, 103, 102, 101, 100,  99],
            "high":   [102, 103, 104, 105, 106, 105, 104, 103, 102, 101],
            "low":    [ 99, 100, 101, 102, 103, 102, 101, 100,  99,  98],
            "close":  [101, 102, 103, 104, 105, 104, 103, 102, 101, 100],
            "volume": [10] * 10,
        },
        index=idx,
    )


@pytest.fixture
def random_ohlcv() -> pd.DataFrame:
    """較長的隨機 OHLCV，用於 look-ahead invariant 測試。"""
    rng = np.random.default_rng(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 1000 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    open_ = close + rng.standard_normal(n) * 0.3
    open_ = np.clip(open_, low, high)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1, 100, n),
        },
        index=idx,
    )


@pytest.fixture
def minute_ohlcv() -> pd.DataFrame:
    """30 分鐘的 1m K 線，用於 resample 測試。"""
    idx = pd.date_range("2024-01-01", periods=30, freq="1min")
    return pd.DataFrame(
        {
            "open":   [100 + i for i in range(30)],
            "high":   [101 + i for i in range(30)],
            "low":    [ 99 + i for i in range(30)],
            "close":  [100.5 + i for i in range(30)],
            "volume": [1.0] * 30,
        },
        index=idx,
    )
