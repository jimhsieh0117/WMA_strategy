"""讀取歷史 K 線 parquet。

設計原則：
- 來源 parquet 唯讀（CLAUDE.md §2.1 兄弟專案禁改），永遠不寫回
- 只負責 IO + 日期過濾 + 欄位標準化，不做任何指標計算
- 嚴格驗證輸出，不符規格立即 raise
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.exceptions import DataIntegrityError
from src.utils.validation import validate_ohlc

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def load_ohlcv(
    parquet_path: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    columns: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """從 parquet 載入 OHLCV K 線並過濾日期區間。

    Args:
        parquet_path: parquet 檔案路徑（通常指向 PPO_TradingModel/data/processed/*.parquet）。
        start: 起始時間（含）；可為 ISO 字串或 Timestamp；None 表示從頭。
        end: 結束時間（含）；可為 ISO 字串或 Timestamp；None 表示到尾。
        columns: 只讀取指定欄位（含 timestamp/index），加速大檔案載入。
            預設 None = 只讀 OHLCV + timestamp，避免 PPO parquet 多餘特徵欄拖慢 IO。

    Returns:
        DataFrame：DatetimeIndex + 小寫欄位 ``[open, high, low, close, volume]``。

    Raises:
        FileNotFoundError: 檔案不存在。
        DataIntegrityError: 缺欄位、index 異常、OHLC 邏輯錯誤、過濾後為空等。
    """
    path = Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"parquet not found: {path}")

    if columns is None:
        # 只讀必要欄位，極大幅減少 IO（PPO parquet 含 30+ 特徵欄）
        # timestamp 可能是 index 也可能是欄位 → 嘗試兩者
        wanted = ["timestamp", *OHLCV_COLUMNS]
        try:
            df = pd.read_parquet(path, columns=wanted)
        except Exception:
            # timestamp 為 index 而非欄位 → 退回讀全表（少見情況）
            df = pd.read_parquet(path)
    else:
        df = pd.read_parquet(path, columns=list(columns))

    # index 標準化：可能存於 timestamp 欄位或 index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    # 欄位名統一小寫（PPO 專案資料已是小寫，但做防呆）
    df.columns = [str(col).lower() for col in df.columns]

    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise DataIntegrityError(
            f"parquet missing required columns: {sorted(missing)}"
        )

    df = df.sort_index()

    if start is not None:
        df = df.loc[pd.Timestamp(start) :]
    if end is not None:
        df = df.loc[: pd.Timestamp(end)]

    if len(df) == 0:
        raise DataIntegrityError(
            f"no rows in range [{start}, {end}] from {path.name}"
        )

    # 強制驗證輸出符合下游約定
    validate_ohlc(df, require_volume=True)

    return df
