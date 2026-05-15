"""FastAPI app for live paper trading viewer。

提供：
- ``GET /``                index.html（lightweight-charts 前端）
- ``GET /api/snapshot``    當前 state（暖機 df + 已完成 trades + paused 狀態）
- ``GET /api/status``      簡短狀態回報
- ``POST /api/pause``      暫停新進場
- ``POST /api/resume``     恢復新進場
- ``WebSocket /ws``        即時 event push（BAR / TRADE_OPEN / TRADE_CLOSE / RATCHET / EQUITY ...）
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from src.broker.types import Trade
from src.live.state import LiveEvent, LiveState

logger = logging.getLogger(__name__)


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _to_unix(ts: pd.Timestamp) -> int:
    return int(ts.timestamp())


def _ohlc_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "time": _to_unix(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for idx, row in df.iterrows()
    ]


def _trade_record(direction: str, t: Trade) -> dict:
    return {
        "direction": direction,
        "pid": int(t.position_id),
        "entry_time": _to_unix(t.entry_timestamp),
        "exit_time": _to_unix(t.exit_timestamp),
        "entry_price": float(t.entry_price),
        "exit_price": float(t.exit_price),
        "quantity": float(t.quantity),
        "net_pnl": float(t.net_pnl),
        "exit_reason": t.exit_reason,
        "final_stage": int(t.final_stage),
        "peak_progress_r": float(t.peak_progress_r),
    }


class LiveServer:
    """WebSocket broadcaster + REST 控制。"""

    def __init__(self, state: LiveState) -> None:
        self.state = state
        self._clients: set[WebSocket] = set()
        self._broadcast_lock = asyncio.Lock()

    async def broadcast(self, event: LiveEvent) -> None:
        if not self._clients:
            return
        payload = {
            "ts_unix": event.ts_unix,
            "kind": event.kind,
            "payload": event.payload,
        }
        text = json.dumps(payload, default=str)
        async with self._broadcast_lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(text)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def attach(self, app: FastAPI) -> None:
        state = self.state

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(_TEMPLATE_DIR / "index.html")

        @app.get("/api/snapshot")
        async def snapshot() -> JSONResponse:
            # df_long 包含 OHLCV 與 indicator；前端只需要 OHLCV
            df = state.df_long
            ohlc = _ohlc_records(df) if df is not None and not df.empty else []
            long_trades = [
                _trade_record("long", t) for t in state.long_account.trade_log
            ]
            short_trades = [
                _trade_record("short", t) for t in state.short_account.trade_log
            ]
            last_close = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0.0
            return JSONResponse({
                "meta": {
                    "symbol": state.cfg.symbol,
                    "timeframe": state.cfg.timeframe,
                    "started_at": str(state.started_at) if state.started_at else None,
                    "last_processed_ts": (
                        str(state.last_processed_ts)
                        if state.last_processed_ts else None
                    ),
                    "bars_processed": state.bars_processed,
                    "is_warmed_up": state.is_warmed_up,
                    "paused": state.paused,
                },
                "ohlc": ohlc,
                "trades": {"long": long_trades, "short": short_trades},
                "equity": {
                    "long": state.long_account.equity(last_close),
                    "short": state.short_account.equity(last_close),
                    "total": state.total_equity(last_close),
                },
                "recent_events": [
                    {"ts_unix": e.ts_unix, "kind": e.kind, "payload": e.payload}
                    for e in list(state.event_buffer)[-50:]
                ],
            })

        @app.get("/api/status")
        async def status() -> JSONResponse:
            return JSONResponse({
                "paused": state.paused,
                "is_warmed_up": state.is_warmed_up,
                "bars_processed": state.bars_processed,
                "last_processed_ts": (
                    str(state.last_processed_ts) if state.last_processed_ts else None
                ),
                "open_positions": {
                    "long": state.long_account.position_count,
                    "short": state.short_account.position_count,
                },
            })

        @app.post("/api/pause")
        async def pause() -> JSONResponse:
            state.paused = True
            await self.broadcast(LiveEvent(
                ts_unix=_to_unix(pd.Timestamp.now(tz="UTC").tz_convert(None)),
                kind="PAUSED",
                payload={},
            ))
            return JSONResponse({"paused": True})

        @app.post("/api/resume")
        async def resume() -> JSONResponse:
            state.paused = False
            await self.broadcast(LiveEvent(
                ts_unix=_to_unix(pd.Timestamp.now(tz="UTC").tz_convert(None)),
                kind="RESUMED",
                payload={},
            ))
            return JSONResponse({"paused": False})

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            self._clients.add(ws)
            logger.info("ws connect; total=%d", len(self._clients))
            try:
                while True:
                    # 等對方關閉；我們不接收客端訊息（client→server 用 REST）
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(ws)
                logger.info("ws disconnect; total=%d", len(self._clients))


def build_app(state: LiveState) -> tuple[FastAPI, LiveServer]:
    app = FastAPI(title=f"WMA Live Paper Trading — {state.cfg.symbol} {state.cfg.timeframe}")
    server = LiveServer(state)
    server.attach(app)
    return app, server
