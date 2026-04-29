"""Backtest engine 的資料型別。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.broker.types import Trade
from src.utils.exceptions import ConfigError


@dataclass(frozen=True)
class EngineConfig:
    """回測引擎的執行參數。

    Attributes:
        position_size_pct: 每筆倉位佔當前 equity 的比例（0 < v <= 1）。
        skip_signal_when_pending: 既有 pending 限價單時，新訊號是否丟棄。
            預設 True（保守）：避免單一 K 線連續訊號造成搶倉。
        force_close_at_end: 回測結束時若仍有持倉，是否以最後一根 close 平倉。
            False = 把未平倉部位的浮動 PnL 算進 equity，但不留 Trade 紀錄。
    """

    position_size_pct: float = 0.6
    skip_signal_when_pending: bool = True
    force_close_at_end: bool = False

    def __post_init__(self) -> None:
        if not (0 < self.position_size_pct <= 1):
            raise ConfigError(
                f"position_size_pct must be in (0, 1], got {self.position_size_pct}"
            )


@dataclass(frozen=True)
class BacktestResult:
    """回測完成的結果，可被 metrics / reporting 模組消費。

    equity_curve 為 ``pd.Series``（DatetimeIndex），方便算 Sharpe / MDD。
    """

    account_name: str
    initial_capital: float
    final_equity: float
    trades: list[Trade]
    equity_curve: pd.Series
    bars_processed: int
    signals_emitted: int
    signals_filled: int
    signals_unfilled: int
    signals_skipped_pending: int
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self.final_equity - self.initial_capital) / self.initial_capital * 100.0
