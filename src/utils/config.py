"""YAML 設定載入與驗證。

對應 configs/default.yaml schema。預設不寫死路徑——caller 傳入 path 字串。
所有不合法欄位皆 raise ``ConfigError``，CLAUDE.md §5 fail-fast。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.exceptions import ConfigError


@dataclass(frozen=True)
class PeriodSpec:
    start: pd.Timestamp
    end: pd.Timestamp | None  # None = 用到資料最後

    @classmethod
    def parse(cls, raw: dict[str, Any], section: str) -> "PeriodSpec":
        if "start" not in raw:
            raise ConfigError(f"period.{section}.start missing")
        start = pd.Timestamp(raw["start"])
        end_raw = raw.get("end")
        end = pd.Timestamp(end_raw) if end_raw is not None else None
        if end is not None and end <= start:
            raise ConfigError(
                f"period.{section}: end ({end}) must be > start ({start})"
            )
        return cls(start=start, end=end)


@dataclass(frozen=True)
class FullConfig:
    """完整設定的扁平容器。"""

    # data
    source_parquet: Path
    symbol: str
    timeframe: str

    # period
    in_sample: PeriodSpec
    out_of_sample: PeriodSpec

    # account
    initial_capital: float
    position_size_pct: float

    # fees
    taker_fee_rate: float
    maker_fee_rate: float
    slippage_pct: float

    # strategy
    wma_fast: int
    wma_slow: int
    atr_period: int
    atr_multiplier: float
    atr_lookback: int

    # backtest
    output_dir: Path
    show_progress: bool
    force_close_at_end: bool
    log_level: str

    raw: dict[str, Any] = field(default_factory=dict)


_VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1H", "4H"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}


def load_config(path: str | Path) -> FullConfig:
    """讀取並驗證 YAML 設定檔。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be dict, got {type(raw).__name__}")

    try:
        data = raw["data"]
        period = raw["period"]
        account = raw["account"]
        fees = raw["fees"]
        strategy = raw["strategy"]
        backtest = raw["backtest"]
    except KeyError as e:
        raise ConfigError(f"missing top-level section: {e}") from e

    # ---- data ----
    source = Path(data["source_parquet"]).expanduser()
    timeframe = data["timeframe"]
    if timeframe not in _VALID_TIMEFRAMES:
        raise ConfigError(
            f"data.timeframe '{timeframe}' invalid; must be one of {sorted(_VALID_TIMEFRAMES)}"
        )
    symbol = str(data["symbol"])

    # ---- period ----
    in_sample = PeriodSpec.parse(period["in_sample"], "in_sample")
    oos = PeriodSpec.parse(period["out_of_sample"], "out_of_sample")

    # ---- account ----
    initial_capital = float(account["initial_capital"])
    position_size_pct = float(account["position_size_pct"])

    # ---- fees ----
    taker = float(fees["taker_fee_rate"])
    maker = float(fees["maker_fee_rate"])
    slip = float(fees["slippage_pct"])

    # ---- strategy ----
    wma_fast = int(strategy["wma_fast"])
    wma_slow = int(strategy["wma_slow"])
    atr_period = int(strategy["atr_period"])
    atr_multiplier = float(strategy["atr_multiplier"])
    atr_lookback = int(strategy["atr_lookback"])

    # ---- backtest ----
    output_dir = Path(backtest.get("output_dir", "results")).expanduser()
    show_progress = bool(backtest.get("show_progress", True))
    force_close_at_end = bool(backtest.get("force_close_at_end", False))
    log_level = str(backtest.get("log_level", "INFO")).upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"backtest.log_level '{log_level}' invalid; must be one of {sorted(_VALID_LOG_LEVELS)}"
        )

    return FullConfig(
        source_parquet=source,
        symbol=symbol,
        timeframe=timeframe,
        in_sample=in_sample,
        out_of_sample=oos,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        taker_fee_rate=taker,
        maker_fee_rate=maker,
        slippage_pct=slip,
        wma_fast=wma_fast,
        wma_slow=wma_slow,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        atr_lookback=atr_lookback,
        output_dir=output_dir,
        show_progress=show_progress,
        force_close_at_end=force_close_at_end,
        log_level=log_level,
        raw=raw,
    )
