"""績效指標計算（單一回測結果）。

對應 ARCHITECTURE.md §3.6 的指標清單。所有計算為純函式，輸入 ``BacktestResult``，
輸出不可變的 ``MetricsReport``。

Sharpe / Sortino 採用 bar-level 報酬序列，乘以 ``sqrt(periods_per_year)`` 年化。
crypto 永續 24/7 → ``periods_per_year`` 由 timeframe 推導（``bars_per_year``）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.types import BacktestResult
from src.broker.types import Trade
from src.utils.exceptions import ConfigError

# 加密貨幣 24/7 → 一年 365 天 × 1440 分鐘
_MINUTES_PER_YEAR = 365 * 24 * 60

# 各 timeframe 對應每根 K 線的分鐘數
_TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "4H": 240,
}


def bars_per_year(timeframe: str) -> int:
    """回傳指定 K 線週期的年化倍數，用於 Sharpe / Sortino 年化。"""
    if timeframe not in _TIMEFRAME_MINUTES:
        raise ConfigError(
            f"unknown timeframe '{timeframe}', expected one of {list(_TIMEFRAME_MINUTES)}"
        )
    return _MINUTES_PER_YEAR // _TIMEFRAME_MINUTES[timeframe]


@dataclass(frozen=True)
class MetricsReport:
    # ---- Period ----
    start: pd.Timestamp
    end: pd.Timestamp
    duration_days: float

    # ---- Returns ----
    initial_capital: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float

    # ---- Risk-adjusted ----
    sharpe_ratio: float
    sortino_ratio: float

    # ---- Drawdown ----
    max_drawdown_pct: float
    calmar_ratio: float

    # ---- Trade stats ----
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    avg_holding_bars: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_trades_per_day: float

    # ---- Stop loss specifics ----
    stop_loss_count: int
    stop_loss_gap_count: int


def compute_metrics(
    result: BacktestResult,
    *,
    timeframe: str,
) -> MetricsReport:
    """從 BacktestResult 計算完整績效指標。

    Args:
        result: 回測結果。
        timeframe: K 線週期（"5m" 等），用於 Sharpe / Sortino 年化。
    """
    eq = result.equity_curve
    if eq.empty:
        raise ValueError("equity_curve is empty; cannot compute metrics")

    start = eq.index[0]
    end = eq.index[-1]
    duration = end - start
    duration_days = duration.total_seconds() / 86400.0
    duration_years = duration_days / 365.0 if duration_days > 0 else 0.0

    # ---- Returns ----
    initial = result.initial_capital
    final = result.final_equity
    total_return = (final - initial) / initial if initial > 0 else 0.0

    # 年化要求至少 1 天，否則 (1+r)^(1/duration_years) 會在短區間溢位且無實質意義
    if duration_years >= 1.0 / 365.0 and (1.0 + total_return) > 0:
        try:
            ann_return = (1.0 + total_return) ** (1.0 / duration_years) - 1.0
        except OverflowError:
            ann_return = float("inf") if total_return > 0 else 0.0
    else:
        ann_return = 0.0

    # ---- Sharpe / Sortino（bar-level 報酬序列）----
    bar_returns = eq.pct_change().dropna()
    bpy = bars_per_year(timeframe)

    if len(bar_returns) > 1:
        std = bar_returns.std()
        if std > 0:
            sharpe = (bar_returns.mean() / std) * np.sqrt(bpy)
        else:
            sharpe = 0.0

        downside = bar_returns[bar_returns < 0]
        if len(downside) > 1 and downside.std() > 0:
            sortino = (bar_returns.mean() / downside.std()) * np.sqrt(bpy)
        else:
            sortino = 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # ---- Drawdown ----
    peak = eq.cummax()
    drawdown = (eq - peak) / peak
    mdd = float(drawdown.min()) if len(drawdown) > 0 else 0.0
    mdd_pct = abs(mdd) * 100.0
    calmar = (ann_return * 100.0) / mdd_pct if mdd_pct > 0 else 0.0

    # ---- Trade stats ----
    trades = result.trades
    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]

    win_rate = (len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win = float(np.mean([t.net_pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.net_pnl for t in losses])) if losses else 0.0
    sum_wins = sum(t.net_pnl for t in wins)
    sum_losses = sum(t.net_pnl for t in losses)
    profit_factor = (sum_wins / abs(sum_losses)) if sum_losses != 0 else 0.0
    expectancy = (
        (win_rate / 100.0) * avg_win + (1.0 - win_rate / 100.0) * avg_loss
        if total_trades > 0 else 0.0
    )

    avg_holding = (
        float(np.mean([_holding_bars(t, timeframe) for t in trades]))
        if trades else 0.0
    )

    max_w_streak = _max_consecutive(trades, win=True)
    max_l_streak = _max_consecutive(trades, win=False)

    trades_per_day = total_trades / duration_days if duration_days > 0 else 0.0

    stop_count = sum(1 for t in trades if t.exit_reason == "STOP_LOSS")
    stop_gap = sum(1 for t in trades if t.exit_reason == "STOP_LOSS_GAP")

    return MetricsReport(
        start=start,
        end=end,
        duration_days=duration_days,
        initial_capital=initial,
        final_equity=final,
        total_return_pct=total_return * 100.0,
        annualized_return_pct=ann_return * 100.0,
        sharpe_ratio=float(sharpe),
        sortino_ratio=float(sortino),
        max_drawdown_pct=mdd_pct,
        calmar_ratio=float(calmar),
        total_trades=total_trades,
        win_rate_pct=win_rate,
        profit_factor=float(profit_factor),
        expectancy=float(expectancy),
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_holding_bars=avg_holding,
        max_consecutive_wins=max_w_streak,
        max_consecutive_losses=max_l_streak,
        avg_trades_per_day=trades_per_day,
        stop_loss_count=stop_count,
        stop_loss_gap_count=stop_gap,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _holding_bars(trade: Trade, timeframe: str) -> float:
    """以 timeframe 為單位估算持倉根數（取整較不必要，回傳浮點）。"""
    minutes = trade.holding_duration.total_seconds() / 60.0
    return minutes / _TIMEFRAME_MINUTES[timeframe]


def _max_consecutive(trades: list[Trade], *, win: bool) -> int:
    best = 0
    cur = 0
    for t in trades:
        is_win = t.net_pnl > 0
        if is_win == win:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best
