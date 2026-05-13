"""指標註冊表（registry）— 圖表面板系統的入口。

每個指標註冊三件事：
1. ``compute(df) -> df``：把所需欄位算進 df（若 prepare_indicators 已算就回傳原 df）
2. ``overlay_series``：要疊加在主圖（candlestick）上的 series 列表
3. ``panel``：要在主圖下方新增的獨立面板（None = 不需要）

加新指標的 3 步驟：
1. 在 ``src/indicators/`` 寫一個純函式（純 indicator computation）
2. 在此檔的 ``REGISTRY`` 加一個 ``IndicatorRegistration`` 條目
3. 跑 ``view_chart.py --panels <name>`` 即可看到（前端不必改）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.indicators.market_structure import compute_market_structure
from src.indicators.wavetrend import wavetrend
from src.viewer.panels import HorizontalLine, PanelSpec, SeriesSpec


@dataclass(frozen=True)
class IndicatorRegistration:
    """單一指標的完整註冊項。"""

    name: str
    compute: Callable[[pd.DataFrame], pd.DataFrame]
    label: str = ""              # UI 顯示名（chip / 下拉），空字串 → 用 name
    overlay_series: tuple[SeriesSpec, ...] = ()
    panel: PanelSpec | None = None
    # 主圖 marker source：給定 df 後回傳 LWC marker dict list
    # （shape/position/color/text/time 等鍵），與 trade markers 合併渲染。
    markers_compute: Callable[[pd.DataFrame], list[dict]] | None = None

    @property
    def display_label(self) -> str:
        return self.label or self.name.replace("_", " ").title()


def _identity(df: pd.DataFrame) -> pd.DataFrame:
    """指標已由 prepare_indicators 預先算好，直接回傳。"""
    return df


def _compute_wavetrend(df: pd.DataFrame) -> pd.DataFrame:
    return wavetrend(df, n1=10, n2=21)


# market structure 預設 pivot 參數（與 yaml/strategy 用同一組值才有一致性）
_MS_PIVOT_LEFT = 10
_MS_PIVOT_RIGHT = 10


def _compute_market_structure(df: pd.DataFrame) -> pd.DataFrame:
    return compute_market_structure(
        df, pivot_left=_MS_PIVOT_LEFT, pivot_right=_MS_PIVOT_RIGHT,
    )


def _ms_markers(df: pd.DataFrame) -> list[dict]:
    """把 ms_swing_high/low + ms_event 序列轉成 LWC markers。

    Pivot 用 circle 小點（綠=PL、紅=PH），事件用 arrow + 文字。
    """
    if "ms_event" not in df.columns:
        return []

    markers: list[dict] = []
    # swing high → circle 紅
    sh = df["ms_swing_high"].dropna()
    for ts, _v in sh.items():
        markers.append({
            "time": int(ts.timestamp()),
            "position": "aboveBar",
            "color": "#dc2626",
            "shape": "circle",
            "size": 1,  # LWC v4: 1=small, 2=default
            "text": "",
        })
    # swing low → circle 綠
    sl = df["ms_swing_low"].dropna()
    for ts, _v in sl.items():
        markers.append({
            "time": int(ts.timestamp()),
            "position": "belowBar",
            "color": "#16a34a",
            "shape": "circle",
            "size": 1,
            "text": "",
        })
    # BoS / CHoCH → 文字標
    evt_col = df["ms_event"]
    nonempty = evt_col[evt_col.astype(str).str.len() > 0]
    label_map = {
        "bos_up": ("aboveBar", "arrowUp", "#3b82f6", "BoS"),
        "bos_down": ("belowBar", "arrowDown", "#3b82f6", "BoS"),
        "choch_up": ("aboveBar", "arrowUp", "#f59e0b", "CHoCH"),
        "choch_down": ("belowBar", "arrowDown", "#f59e0b", "CHoCH"),
    }
    for ts, evt in nonempty.items():
        spec = label_map.get(str(evt))
        if spec is None:
            continue
        position, shape, color, text = spec
        markers.append({
            "time": int(ts.timestamp()),
            "position": position,
            "color": color,
            "shape": shape,
            "text": text,
        })

    markers.sort(key=lambda m: m["time"])
    return markers


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, IndicatorRegistration] = {
    # ---- 主圖 overlay：策略已用的指標 ----
    "bollinger": IndicatorRegistration(
        name="bollinger",
        label="Bollinger Bands",
        compute=_identity,
        overlay_series=(
            SeriesSpec("bb_upper", "BB Upper",
                       color="#facc15", line_style="dotted"),
            SeriesSpec("bb_middle", "BB Middle",
                       color="#facc15"),
            SeriesSpec("bb_lower", "BB Lower",
                       color="#facc15", line_style="dotted"),
        ),
    ),
    # 進場訊號用：原始 close 上的 WMA
    "wma": IndicatorRegistration(
        name="wma",
        label="WMA",
        compute=_identity,
        overlay_series=(
            SeriesSpec("wma_fast", "WMA Fast", color="#22d3ee"),
            SeriesSpec("wma_slow", "WMA Slow", color="#a855f7"),
        ),
    ),

    # ---- 獨立副面板 ----
    "volume": IndicatorRegistration(
        name="volume",
        label="Volume",
        compute=_identity,
        panel=PanelSpec(
            id="volume",
            title="Volume",
            height_ratio=0.15,
            series=(
                SeriesSpec("volume", "Volume",
                           type="histogram", color="#6b7280"),
            ),
        ),
    ),
    "market_structure": IndicatorRegistration(
        name="market_structure",
        label="Market Structure",
        compute=_compute_market_structure,
        markers_compute=_ms_markers,
    ),
    "wavetrend": IndicatorRegistration(
        name="wavetrend",
        label="WaveTrend",
        compute=_compute_wavetrend,
        panel=PanelSpec(
            id="wavetrend",
            title="WaveTrend (10, 21)",
            height_ratio=0.25,
            series=(
                SeriesSpec("wt1", "WT1", color="#3b82f6"),
                SeriesSpec("wt2", "WT2", color="#f59e0b"),
            ),
            horizontal_lines=(
                HorizontalLine(60.0, "#dc2626", "OB +60"),
                HorizontalLine(-60.0, "#16a34a", "OS −60"),
                HorizontalLine(0.0, "#6b7280", ""),
            ),
        ),
    ),
}


def default_panels() -> list[str]:
    """預設面板組合。"""
    return ["bollinger", "wma", "volume", "wavetrend", "market_structure"]
