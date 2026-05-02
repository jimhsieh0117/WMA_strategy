"""共用執行流程：跑回測 + 產出報告。

API 拆成兩段，方便 run_combined.py 跑兩次回測後再合併：
- ``run_single_strategy(cfg, direction, sample) -> BacktestResult``：純回測，不印不存
- ``report_result(result, cfg, label) -> MetricsReport``：印 + 算 metrics + 落地全部輸出
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.types import BacktestResult, EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import BrokerConfig
from src.data.loader import load_ohlcv
from src.data.resampler import resample
from src.metrics.calculator import MetricsReport, compute_metrics
from src.reporting.exporter import (
    export_config_snapshot,
    export_metrics_json,
    export_summary_text,
    export_trades_csv,
)
from src.reporting.plotter import plot_drawdown, plot_equity_curves
from src.strategy.base import BaseTrendStrategy, prepare_indicators
from src.strategy.long_strategy import LongTrendStrategy
from src.strategy.short_strategy import ShortTrendStrategy
from src.strategy.types import StrategyParams, TrailingStopParams
from src.utils.config import FullConfig, PeriodSpec

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 內部工具
# --------------------------------------------------------------------------- #

def _build_strategy(
    direction: Literal["long", "short"], params: StrategyParams
) -> BaseTrendStrategy:
    if direction == "long":
        return LongTrendStrategy(params)
    if direction == "short":
        return ShortTrendStrategy(params)
    raise ValueError(f"unknown direction '{direction}'")


def _resolve_period(cfg: FullConfig, sample: Literal["is", "oos"]) -> PeriodSpec:
    return cfg.in_sample if sample == "is" else cfg.out_of_sample


# --------------------------------------------------------------------------- #
# 1) 單支策略回測
# --------------------------------------------------------------------------- #

def run_single_strategy(
    cfg: FullConfig,
    direction: Literal["long", "short"],
    sample: Literal["is", "oos"] = "is",
) -> BacktestResult:
    """執行單一方向策略回測，回傳 ``BacktestResult``（不印、不存檔）。"""
    period = _resolve_period(cfg, sample)
    label = f"{direction}_{cfg.timeframe}_{sample}"

    logger.info("=" * 60)
    logger.info("[%s] WMA Backtest: %s %s", label, cfg.symbol, cfg.timeframe)
    logger.info("Period: %s ~ %s", period.start, period.end)
    logger.info("=" * 60)

    logger.info("[1/4] loading 1m data ...")
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    logger.info("    loaded %s 1m bars", f"{len(df1m):,}")

    if cfg.timeframe != "1m":
        logger.info("[2/4] resampling 1m -> %s ...", cfg.timeframe)
        df = resample(df1m, cfg.timeframe)
    else:
        df = df1m
    logger.info("    %s %s bars after resample", f"{len(df):,}", cfg.timeframe)

    logger.info("[3/4] preparing indicators ...")
    trailing = TrailingStopParams(
        swing_lookback=cfg.trailing.swing_lookback,
        stage1_slippage_buffer=cfg.trailing.stage1_slippage_buffer,
        stage2_normal_trigger_r=cfg.trailing.stage2_normal_trigger_r,
        stage2_abnormal_trigger_r=cfg.trailing.stage2_abnormal_trigger_r,
        stage2_buffer_r=cfg.trailing.stage2_buffer_r,
        stage3_normal_trigger_r=cfg.trailing.stage3_normal_trigger_r,
        stage3_abnormal_trigger_r=cfg.trailing.stage3_abnormal_trigger_r,
        bollinger_period=cfg.trailing.bollinger_period,
        bollinger_num_std=cfg.trailing.bollinger_num_std,
    )
    params = StrategyParams(
        wma_fast=cfg.wma_fast,
        wma_slow=cfg.wma_slow,
        entry_source=cfg.entry_source,  # type: ignore[arg-type]
        trailing=trailing,
    )
    augmented = prepare_indicators(df, params)

    broker = BrokerSimulator(BrokerConfig(
        taker_fee_rate=cfg.taker_fee_rate,
        maker_fee_rate=cfg.maker_fee_rate,
        slippage_pct=cfg.slippage_pct,
    ))
    account = Account(cfg.initial_capital, name=label)
    strategy = _build_strategy(direction, params)
    engine_cfg = EngineConfig(
        position_size_pct=cfg.position_size_pct,
        force_close_at_end=cfg.force_close_at_end,
    )

    logger.info("[4/4] running backtest ...")
    return run_backtest(
        augmented, strategy=strategy, account=account, broker=broker,
        config=engine_cfg, show_progress=cfg.show_progress,
    )


# --------------------------------------------------------------------------- #
# 2) Report：印摘要 + 落地全部輸出
# --------------------------------------------------------------------------- #

def report_result(
    result: BacktestResult,
    cfg: FullConfig,
    *,
    label: str,
    extra_curves: dict[str, pd.Series] | None = None,
) -> MetricsReport:
    """為一個回測結果計算 metrics、印摘要，並輸出到 ``cfg.output_dir/{symbol}_{label}/``。

    Args:
        result: BacktestResult。
        cfg: 完整設定。
        label: 子目錄與圖示用標籤（如 "long_15m_is"、"combined_15m_is"）。
        extra_curves: 額外要疊到 equity 圖的曲線（key 為圖示 label）。

    Returns:
        MetricsReport。
    """
    metrics = compute_metrics(result, timeframe=cfg.timeframe)

    # 印摘要到 stdout 並保留同樣文字寫進 summary.txt
    summary_lines = _format_summary_lines(metrics, result)
    print()
    for line in summary_lines:
        print(line)

    out_dir = cfg.output_dir / f"{cfg.symbol}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 文字 / json
    export_summary_text(summary_lines, out_dir / "summary.txt")
    export_metrics_json(metrics, out_dir / "metrics.json")
    export_trades_csv(result.trades, out_dir / "trades.csv")
    export_config_snapshot(result.config_snapshot, out_dir / "config_snapshot.json")

    # 圖
    curves: dict[str, pd.Series] = {label: result.equity_curve}
    if extra_curves:
        curves.update(extra_curves)
    plot_equity_curves(curves, out_dir / "equity.png", title=f"Equity Curve — {label}")
    plot_drawdown(result.equity_curve, out_dir / "drawdown.png",
                  title=f"Drawdown — {label}")

    print(f"\n💾 outputs saved → {out_dir}")
    return metrics


# --------------------------------------------------------------------------- #
# Summary 文字
# --------------------------------------------------------------------------- #

def _format_summary_lines(m: MetricsReport, result: BacktestResult) -> list[str]:
    buf = io.StringIO()
    line = "=" * 60
    print(line, file=buf)
    print(f"  績效摘要 / Performance Summary  [{result.account_name}]", file=buf)
    print(line, file=buf)
    print(f"  區間：{m.start} ~ {m.end}  ({m.duration_days:.1f} days)", file=buf)
    print("-" * 60, file=buf)
    print(f"  Initial Capital:     {m.initial_capital:>10.2f} USDT", file=buf)
    print(f"  Final Equity:        {m.final_equity:>10.2f} USDT", file=buf)
    print(f"  Total Return:        {m.total_return_pct:>+10.2f} %", file=buf)
    print(f"  Annualized Return:   {m.annualized_return_pct:>+10.2f} %", file=buf)
    print("-" * 60, file=buf)
    print(f"  Sharpe Ratio:        {m.sharpe_ratio:>10.2f}", file=buf)
    print(f"  Sortino Ratio:       {m.sortino_ratio:>10.2f}", file=buf)
    print(f"  Max Drawdown:        {m.max_drawdown_pct:>10.2f} %", file=buf)
    print(f"  Calmar Ratio:        {m.calmar_ratio:>10.2f}", file=buf)
    print("-" * 60, file=buf)
    print(f"  Total Trades:        {m.total_trades:>10d}", file=buf)
    print(f"  Avg Trades / Day:    {m.avg_trades_per_day:>10.2f}", file=buf)
    print(f"  Win Rate:            {m.win_rate_pct:>10.2f} %", file=buf)
    print(f"  Profit Factor:       {m.profit_factor:>10.2f}", file=buf)
    print(f"  Expectancy / trade:  {m.expectancy:>10.4f} USDT", file=buf)
    print(f"  Avg Win / Loss:      {m.avg_win:>10.4f} / {m.avg_loss:.4f}", file=buf)
    print(f"  Max Consec W / L:    {m.max_consecutive_wins:>10d} / {m.max_consecutive_losses}",
          file=buf)
    print(f"  Avg Hold (bars):     {m.avg_holding_bars:>10.2f}", file=buf)
    print("-" * 60, file=buf)
    print(f"  Stop Loss (intraday):{m.stop_loss_count:>10d}", file=buf)
    print(f"  Stop Loss (gap):     {m.stop_loss_gap_count:>10d}", file=buf)
    print("-" * 60, file=buf)
    print(
        f"  訊號統計：emitted={result.signals_emitted}, "
        f"filled={result.signals_filled}, "
        f"unfilled={result.signals_unfilled}, "
        f"skipped_pending={result.signals_skipped_pending}",
        file=buf,
    )
    print(line, file=buf)
    return buf.getvalue().rstrip("\n").split("\n")
