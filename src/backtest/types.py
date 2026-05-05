"""Backtest engine 的資料型別。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from src.broker.types import Trade
from src.utils.exceptions import ConfigError


SizingMode = Literal["pct", "risk"]


@dataclass(frozen=True)
class EngineConfig:
    """回測引擎的執行參數。

    Attributes:
        sizing_mode: ``"pct"`` = 每筆倉位佔當前 equity 的固定比例；
            ``"risk"`` = 每筆倉位以「撞到 stop 時固定虧損 ``risk_per_trade_usdt``」反推。
        position_size_pct: ``sizing_mode="pct"`` 時生效（0 < v <= 1）。
        risk_per_trade_usdt: ``sizing_mode="risk"`` 時生效。每筆預期最大虧損（含
            雙向 taker fee）。公式：
            ``qty = risk / [|entry − stop| + (entry + stop) × taker]``
            （滑點已隱含在 entry_price，不重複計）。
            若算出 ``qty × entry > equity``，視為過槓桿，**拒絕該筆進場**。
        skip_signal_when_pending: 既有 pending 限價單時，新訊號是否丟棄。
        force_close_at_end: 回測結束時若仍有持倉，是否以最後一根 close 平倉。
        allow_pyramiding: 是否允許在已持倉狀態下開新倉（多筆並行）。
            - False（預設）：與舊版完全一致，每段時間只會有一筆持倉
            - True：每根 K 線最多新增 1 筆（仍受 pending 機制節流），
                每筆獨立 stop / 獨立 trailing controller / 獨立 sizing
        leverage_cap: ``allow_pyramiding=True`` 時生效，限制
            ``Σ open_notional + new_notional ≤ equity_now × leverage_cap``。
            預設 1.0（不允許超過自有資金；即不開實質槓桿）。
    """

    sizing_mode: SizingMode = "pct"
    position_size_pct: float = 0.6
    risk_per_trade_usdt: float = 1.0
    skip_signal_when_pending: bool = True
    force_close_at_end: bool = False
    allow_pyramiding: bool = False
    leverage_cap: float = 1.0

    def __post_init__(self) -> None:
        if self.sizing_mode not in ("pct", "risk"):
            raise ConfigError(
                f"sizing_mode must be 'pct' or 'risk', got {self.sizing_mode!r}"
            )
        if not (0 < self.position_size_pct <= 1):
            raise ConfigError(
                f"position_size_pct must be in (0, 1], got {self.position_size_pct}"
            )
        if self.risk_per_trade_usdt <= 0:
            raise ConfigError(
                f"risk_per_trade_usdt must be > 0, got {self.risk_per_trade_usdt}"
            )
        if self.leverage_cap <= 0:
            raise ConfigError(
                f"leverage_cap must be > 0, got {self.leverage_cap}"
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
