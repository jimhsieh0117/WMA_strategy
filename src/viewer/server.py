"""FastAPI server — 把 backtest 結果序列化為 LWC 前端可用的 JSON。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from src.backtest.types import BacktestResult
from src.broker.types import Trade
from src.metrics.calculator import MetricsReport, compute_metrics
from src.viewer.panels import PanelSpec, SeriesSpec


_TEMPLATE_DIR = Path(__file__).parent / "templates"


# --------------------------------------------------------------------------- #
# 序列化 helpers — pandas 物件 → LWC 接受的 JSON
# --------------------------------------------------------------------------- #

def _to_unix_seconds(ts: pd.Timestamp) -> int:
    """LWC time scale 接受 unix 秒（int）。"""
    return int(ts.timestamp())


def _ohlc_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "time": _to_unix_seconds(idx),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for idx, row in df.iterrows()
    ]


def _line_records(s: pd.Series) -> list[dict]:
    """把 Series 轉成 LWC line / area 可吃的 [{time, value}, ...]，跳過 NaN。"""
    valid = s.dropna()
    return [
        {"time": _to_unix_seconds(idx), "value": float(v)}
        for idx, v in valid.items()
    ]


def _volume_records(df: pd.DataFrame) -> list[dict]:
    """Volume histogram：每根用陽紅 / 陰綠著色。"""
    out = []
    for idx, row in df.iterrows():
        is_up = row["close"] >= row["open"]
        color = "rgba(38, 166, 154, 0.5)" if is_up else "rgba(239, 83, 80, 0.5)"
        out.append({
            "time": _to_unix_seconds(idx),
            "value": float(row["volume"]),
            "color": color,
        })
    return out


def _trade_records(trades: list[Trade]) -> list[dict]:
    return [
        {
            "direction": t.direction.value,
            "entry_time": _to_unix_seconds(t.entry_timestamp),
            "exit_time": _to_unix_seconds(t.exit_timestamp),
            "entry_price": float(t.entry_price),
            "exit_price": float(t.exit_price),
            "quantity": float(t.quantity),
            "net_pnl": float(t.net_pnl),
            "return_pct": float(t.return_pct * 100.0),
            "exit_reason": t.exit_reason,
        }
        for t in trades
    ]


# --------------------------------------------------------------------------- #
# Trade-derived overlays（持倉連線 + 止損軌道）
# --------------------------------------------------------------------------- #

def _holding_segments(
    trades: list[Trade], *, win: bool
) -> list[list[dict]]:
    """每筆交易一個 segment：``[entry_point, exit_point]``。

    回傳的是 list-of-segments，frontend 為每個 segment 建立獨立 LineSeries，
    從根本上避免「跨 trade 連線」的 LWC v4 行為（whitespace + WithSteps 無法
    可靠斷開等待期間的線）。
    """
    segments: list[list[dict]] = []
    for t in sorted(trades, key=lambda x: x.entry_timestamp):
        if (t.net_pnl > 0) != win:
            continue
        et = _to_unix_seconds(t.entry_timestamp)
        xt = _to_unix_seconds(t.exit_timestamp)
        if xt <= et:
            continue
        segments.append([
            {"time": et, "value": float(t.entry_price)},
            {"time": xt, "value": float(t.exit_price)},
        ])
    return segments


def _stop_track_segments(trades: list[Trade]) -> list[list[dict]]:
    """每筆交易一個 segment 的階梯軌跡。

    用「**手動編碼階梯點**」取代 LWC 的 ``LineType.WithSteps``：
    每次 stop 變化前 1 秒插入前一個 stop 值的延伸點，再用 Solid line
    連起來，視覺上等於直角階梯，但完全不依賴 LWC 不可靠的階梯渲染。

    範例：history=[(t=100,95),(t=150,97)], exit=200 →
        [(100,95), (149,95), (150,97), (199,97), (200,97)]
        ↑      ↑99秒橫線↑斜1秒↑99秒橫線↑斜1秒↑
    """
    segments: list[list[dict]] = []
    for t in sorted(trades, key=lambda x: x.entry_timestamp):
        if not t.stop_history:
            continue
        history = list(t.stop_history)
        last_stop = history[-1][1]
        # 加上 (exit_ts, last_stop) 讓階梯延伸到出場
        history.append((t.exit_timestamp, last_stop))

        seg: list[dict] = []
        prev_t = -1
        prev_v: float | None = None
        for ts, stop in history:
            ut = _to_unix_seconds(ts)
            if ut <= prev_t:
                continue  # 略過時序倒退或重複
            # 插入「前 1 秒 + 前一值」做手動階梯
            if prev_v is not None and ut > prev_t + 1:
                seg.append({"time": ut - 1, "value": float(prev_v)})
            seg.append({"time": ut, "value": float(stop)})
            prev_t = ut
            prev_v = float(stop)

        if seg:
            segments.append(seg)
    return segments


def _metrics_summary(m: MetricsReport) -> dict:
    """為 header 顯示縮減版 metrics（避免 JSON 巨大）。"""
    return {
        "initial_capital": m.initial_capital,
        "final_equity": m.final_equity,
        "total_return_pct": m.total_return_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "win_rate_pct": m.win_rate_pct,
        "profit_factor": m.profit_factor,
        "total_trades": m.total_trades,
        "sharpe_ratio": m.sharpe_ratio,
    }


def _series_payload(s: SeriesSpec, df: pd.DataFrame) -> dict:
    """組成單一 series 的完整 payload。"""
    if s.type == "histogram" and s.column == "volume":
        data = _volume_records(df)
    else:
        data = _line_records(df[s.column])
    return {
        "column": s.column,
        "title": s.title,
        "type": s.type,
        "color": s.color,
        "line_width": s.line_width,
        "line_style": s.line_style,
        "data": data,
    }


def _panel_payload(panel: PanelSpec, df: pd.DataFrame) -> dict:
    return {
        "id": panel.id,
        "title": panel.title,
        "height_ratio": panel.height_ratio,
        "series": [_series_payload(s, df) for s in panel.series],
        "horizontal_lines": [asdict(line) for line in panel.horizontal_lines],
    }


# --------------------------------------------------------------------------- #
# 主 build_app
# --------------------------------------------------------------------------- #

def build_app(
    *,
    symbol: str,
    timeframe: str,
    sample: str,
    df_main: pd.DataFrame,
    main_overlays: list[SeriesSpec],
    panels: list[PanelSpec],
    long_result: BacktestResult,
    short_result: BacktestResult,
    combined_result: BacktestResult,
) -> FastAPI:
    """建立 FastAPI app：serve index.html + /api/data。"""
    app = FastAPI(title=f"WMA Backtest Viewer — {symbol} {timeframe}")

    # 一次性算好 payload，避免每次 request 都序列化
    long_metrics = compute_metrics(long_result, timeframe=timeframe)
    short_metrics = compute_metrics(short_result, timeframe=timeframe)
    combined_metrics = compute_metrics(combined_result, timeframe=timeframe)

    payload = {
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "sample": sample,
            "long_metrics": _metrics_summary(long_metrics),
            "short_metrics": _metrics_summary(short_metrics),
            "combined_metrics": _metrics_summary(combined_metrics),
        },
        "ohlc": _ohlc_to_records(df_main),
        "main_overlays": [_series_payload(s, df_main) for s in main_overlays],
        "trades": {
            "long": _trade_records(long_result.trades),
            "short": _trade_records(short_result.trades),
        },
        "trade_lines": {
            # 每個 key 是 list-of-segments：每筆交易一個 segment（一個 LineSeries）
            "holding_wins": _holding_segments(
                long_result.trades + short_result.trades, win=True,
            ),
            "holding_losses": _holding_segments(
                long_result.trades + short_result.trades, win=False,
            ),
            "long_stops": _stop_track_segments(long_result.trades),
            "short_stops": _stop_track_segments(short_result.trades),
        },
        "equity": {
            "long": _line_records(long_result.equity_curve),
            "short": _line_records(short_result.equity_curve),
            "combined": _line_records(combined_result.equity_curve),
        },
        "panels": [_panel_payload(p, df_main) for p in panels],
    }

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(_TEMPLATE_DIR / "index.html")

    @app.get("/api/data")
    async def get_data() -> JSONResponse:
        return JSONResponse(payload)

    return app
