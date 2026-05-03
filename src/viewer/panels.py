"""圖表面板與 series 規格。

提供面板系統，讓「在主圖加 overlay」與「在底下新增獨立指標面板」都用同一份介面。
未來新增指標只需註冊 ``IndicatorRegistration``（見 ``indicators.py``），
前端會自動建立對應的圖表 / overlay。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SeriesType = Literal["line", "histogram", "area"]
LineStyle = Literal["solid", "dashed", "dotted"]


@dataclass(frozen=True)
class SeriesSpec:
    """一條 series（可放主圖 overlay 或副面板）。"""

    column: str                  # 對應 DataFrame 欄位名
    title: str                   # 圖例顯示名（也用於滑鼠十字線）
    type: SeriesType = "line"
    color: str = "#3b82f6"
    line_width: int = 1
    line_style: LineStyle = "solid"


@dataclass(frozen=True)
class HorizontalLine:
    """副面板上的水平參考線（如 WaveTrend 的 ±60、0）。"""

    value: float
    color: str = "#6b7280"
    label: str = ""


@dataclass(frozen=True)
class PanelSpec:
    """一個獨立的副面板，自帶 y 軸。"""

    id: str
    title: str
    height_ratio: float = 0.25
    series: tuple[SeriesSpec, ...] = ()
    horizontal_lines: tuple[HorizontalLine, ...] = ()
