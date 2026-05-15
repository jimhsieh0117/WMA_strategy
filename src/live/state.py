"""LiveState：跨組件共享的可變狀態。

所有寫入由單一 LiveSimulator coroutine 進行；FastAPI handler / WebSocket
broadcaster 只讀。Python asyncio 為 single-thread cooperative，只要寫入
方在無 await 點時更新即可保證一致性，不需要 explicit lock。
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.broker.account import Account
from src.broker.types import Bar
from src.strategy.base import BaseTrendStrategy
from src.strategy.trailing import TrailingStopController
from src.strategy.types import EntrySignal
from src.utils.config import FullConfig

logger = logging.getLogger(__name__)


@dataclass
class LiveEvent:
    """SSE / WebSocket 推播到前端的事件。"""

    ts_unix: int                 # 事件時間（K 線收盤時間，UTC unix sec）
    kind: str                    # "BAR" | "TRADE_OPEN" | "TRADE_CLOSE" | "RATCHET" |
                                 # "PAUSED" | "RESUMED" | "WARMUP" | "INFO" | "ERROR"
    payload: dict[str, Any]      # 不同 kind 不同欄位


@dataclass
class LiveState:
    """單例。LiveSimulator + LiveCandleFeed + FastAPI server 共用。"""

    cfg: FullConfig

    # 兩個獨立帳戶 + 各自的 df（WMA per-direction 算 indicator 結果不同）
    long_account: Account
    short_account: Account
    long_strategy: BaseTrendStrategy
    short_strategy: BaseTrendStrategy
    df_long: pd.DataFrame                                # 含 long params indicator
    df_short: pd.DataFrame                               # 含 short params indicator

    # 每筆持倉一個 controller，key = position_id
    trailings_long: dict[int, TrailingStopController] = field(default_factory=dict)
    trailings_short: dict[int, TrailingStopController] = field(default_factory=dict)

    # 待撮合（下一根 K 開盤撮合的限價單）
    pending_long: EntrySignal | None = None
    pending_short: EntrySignal | None = None

    # 控制
    paused: bool = False                                 # 暫停新進場（trailing/stop 照跑）
    is_warmed_up: bool = False                           # 暖機完成才開始模擬交易

    # 統計
    started_at: pd.Timestamp | None = None
    last_processed_ts: pd.Timestamp | None = None
    bars_processed: int = 0                              # 含暖機載入的根數

    # 事件 buffer（給 server snapshot endpoint，最近 N 個）
    event_buffer: deque[LiveEvent] = field(
        default_factory=lambda: deque(maxlen=500),
    )

    def has_open_position(self, direction: str) -> bool:
        if direction == "long":
            return self.long_account.has_position()
        return self.short_account.has_position()

    def total_equity(self, mark_price: float) -> float:
        return (
            self.long_account.equity(mark_price)
            + self.short_account.equity(mark_price)
        )

    def append_event(self, event: LiveEvent) -> None:
        self.event_buffer.append(event)
        logger.info(
            "EVENT %s ts=%s payload=%s",
            event.kind, event.ts_unix, event.payload,
        )
