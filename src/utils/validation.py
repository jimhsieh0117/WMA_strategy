"""共用的資料驗證工具。

依 CLAUDE.md §5 嚴格 fail-fast：任何不符規範的輸入皆 raise，不靜默修補。
"""

from __future__ import annotations

import pandas as pd

from .exceptions import DataIntegrityError

REQUIRED_OHLC: tuple[str, ...] = ("open", "high", "low", "close")


def validate_ohlc(df: pd.DataFrame, *, require_volume: bool = False) -> None:
    """驗證 DataFrame 為合法的 OHLC(V) 格式。

    檢查項目：
    1. 物件型別為 ``pd.DataFrame``
    2. 包含 ``open / high / low / close``（require_volume=True 時加上 ``volume``）
    3. index 為 ``DatetimeIndex`` 且嚴格單調遞增
    4. OHLC 邏輯一致（high >= max(open,close,low)、low <= min(open,close,high)）
    5. 必要欄位無 NaN

    違反任一項即 raise ``DataIntegrityError``。
    """
    if not isinstance(df, pd.DataFrame):
        raise DataIntegrityError(
            f"expected pandas DataFrame, got {type(df).__name__}"
        )

    required: set[str] = set(REQUIRED_OHLC)
    if require_volume:
        required.add("volume")
    missing = required - set(df.columns)
    if missing:
        raise DataIntegrityError(f"missing columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataIntegrityError(
            f"index must be DatetimeIndex, got {type(df.index).__name__}"
        )

    if len(df) > 1 and not df.index.is_monotonic_increasing:
        raise DataIntegrityError("index must be monotonic increasing")

    if df.index.has_duplicates:
        raise DataIntegrityError("index contains duplicate timestamps")

    cols_to_check = list(required)
    if df[cols_to_check].isna().any().any():
        na_counts = df[cols_to_check].isna().sum()
        raise DataIntegrityError(
            f"NaN found in required columns: {na_counts[na_counts > 0].to_dict()}"
        )

    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]
    invalid = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )
    if invalid.any():
        n = int(invalid.sum())
        raise DataIntegrityError(f"{n} bars violate OHLC consistency")


def validate_period(period: int, *, name: str = "period") -> None:
    """驗證指標週期為正整數（排除 bool）。"""
    if isinstance(period, bool) or not isinstance(period, int):
        raise DataIntegrityError(
            f"{name} must be int, got {type(period).__name__}"
        )
    if period < 1:
        raise DataIntegrityError(f"{name} must be >= 1, got {period}")
