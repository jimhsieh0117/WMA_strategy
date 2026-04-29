"""matplotlib 視覺化（權益曲線 / 回撤）。

CLAUDE.md §4：視覺化文字用繁中、專有名詞保留英文。
matplotlib 預設不含 CJK 字型，於模組載入時嘗試切換到 macOS 內建字型；
若不可用則退回 ASCII 標題（不阻斷流程）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")  # 不開圖形視窗，純檔案輸出（適合 server / CI）
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# CJK 字型偏好序（macOS 優先，Linux fallback 為 sans-serif）
_CJK_CANDIDATES = (
    "PingFang TC", "Hiragino Sans GB", "Heiti TC",
    "Arial Unicode MS", "Noto Sans CJK TC", "sans-serif",
)


def _setup_cjk_font() -> bool:
    """嘗試將 sans-serif 字型族設為支援 CJK 的字型。

    Returns:
        True 表示成功；False 表示找不到（退回繁中可能變方框，但 ASCII 仍正常）。
    """
    plt.rcParams["font.sans-serif"] = list(_CJK_CANDIDATES)
    plt.rcParams["axes.unicode_minus"] = False
    return True


_setup_cjk_font()


# --------------------------------------------------------------------------- #
# Equity curve
# --------------------------------------------------------------------------- #

def plot_equity_curves(
    curves: Mapping[str, pd.Series],
    output_path: str | Path,
    *,
    title: str = "權益曲線",
    figsize: tuple[float, float] = (12, 5),
) -> Path:
    """將多條 equity 曲線疊圖輸出 PNG。

    Args:
        curves: ``{label: series}``，每條 series index 必須為 DatetimeIndex。
        output_path: 輸出檔案路徑（包含副檔名）。
        title: 圖標題。
        figsize: matplotlib 圖大小。

    Returns:
        實際寫入的 Path。
    """
    if not curves:
        raise ValueError("plot_equity_curves: empty curves dict")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    for label, s in curves.items():
        ax.plot(s.index, s.values, label=label, linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("時間 / Time")
    ax.set_ylabel("Equity (USDT)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    logger.info("equity plot saved: %s", out)
    return out


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #

def plot_drawdown(
    curve: pd.Series,
    output_path: str | Path,
    *,
    title: str = "回撤 / Drawdown",
    figsize: tuple[float, float] = (12, 4),
) -> Path:
    """畫出回撤序列（負百分比）。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    peak = curve.cummax()
    dd_pct = (curve - peak) / peak * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(dd_pct.index, dd_pct.values, 0, color="#d62728", alpha=0.4)
    ax.plot(dd_pct.index, dd_pct.values, color="#d62728", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("時間 / Time")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=1.0)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    logger.info("drawdown plot saved: %s", out)
    return out
