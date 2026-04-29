"""共用的單支策略回測 runner（被 run_long.py / run_short.py 呼叫）。

抽出共用邏輯：載入資料 → resample → prepare 指標 → 建 strategy/account/broker
→ 跑 engine → 計算 metrics → 輸出摘要。
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Literal

import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.types import EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import BrokerConfig
from src.data.loader import load_ohlcv
from src.data.resampler import resample
from src.metrics.calculator import MetricsReport, compute_metrics
from src.strategy.base import BaseTrendStrategy, prepare_indicators
from src.strategy.long_strategy import LongTrendStrategy
from src.strategy.short_strategy import ShortTrendStrategy
from src.strategy.types import StrategyParams
from src.utils.config import FullConfig, PeriodSpec

logger = logging.getLogger(__name__)


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


def run_single_strategy(
    cfg: FullConfig,
    direction: Literal["long", "short"],
    sample: Literal["is", "oos"] = "is",
) -> tuple[MetricsReport, pd.DataFrame]:
    """執行單一方向策略回測。

    Returns:
        (metrics, trades_df) 供 caller 自行落地或進一步分析。
    """
    period = _resolve_period(cfg, sample)
    label = f"{direction}_{cfg.timeframe}_{sample}"

    logger.info("=" * 60)
    logger.info("[%s] WMA Backtest: %s %s", label, cfg.symbol, cfg.timeframe)
    logger.info("Period: %s ~ %s", period.start, period.end)
    logger.info("=" * 60)

    # 1. 載入 + resample
    logger.info("[1/5] loading 1m data ...")
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    logger.info("    loaded %s 1m bars", f"{len(df1m):,}")

    if cfg.timeframe != "1m":
        logger.info("[2/5] resampling 1m -> %s ...", cfg.timeframe)
        df = resample(df1m, cfg.timeframe)
    else:
        df = df1m
    logger.info("    %s %s bars after resample", f"{len(df):,}", cfg.timeframe)

    # 2. 指標
    logger.info("[3/5] preparing indicators ...")
    params = StrategyParams(
        wma_fast=cfg.wma_fast,
        wma_slow=cfg.wma_slow,
        atr_period=cfg.atr_period,
        atr_multiplier=cfg.atr_multiplier,
        atr_lookback=cfg.atr_lookback,
    )
    augmented = prepare_indicators(df, params)

    # 3. broker / account / strategy
    broker_cfg = BrokerConfig(
        taker_fee_rate=cfg.taker_fee_rate,
        maker_fee_rate=cfg.maker_fee_rate,
        slippage_pct=cfg.slippage_pct,
    )
    broker = BrokerSimulator(broker_cfg)
    account = Account(cfg.initial_capital, name=label)
    strategy = _build_strategy(direction, params)

    engine_cfg = EngineConfig(
        position_size_pct=cfg.position_size_pct,
        force_close_at_end=cfg.force_close_at_end,
    )

    # 4. engine
    logger.info("[4/5] running backtest ...")
    result = run_backtest(
        augmented,
        strategy=strategy,
        account=account,
        broker=broker,
        config=engine_cfg,
        show_progress=cfg.show_progress,
    )

    # 5. metrics
    logger.info("[5/5] computing metrics ...")
    metrics = compute_metrics(result, timeframe=cfg.timeframe)

    # trades dataframe（供落地）
    trades_df = pd.DataFrame(
        [
            {
                "direction": t.direction.value,
                "entry_ts": t.entry_timestamp,
                "exit_ts": t.exit_timestamp,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "return_pct": t.return_pct * 100.0,
                "exit_reason": t.exit_reason,
                "holding_minutes": t.holding_duration.total_seconds() / 60.0,
            }
            for t in result.trades
        ]
    )

    print_summary(metrics, result.signals_emitted, result.signals_filled,
                  result.signals_unfilled, result.signals_skipped_pending)

    return metrics, trades_df


def print_summary(
    m: MetricsReport,
    signals_emitted: int,
    signals_filled: int,
    signals_unfilled: int,
    signals_skipped: int,
) -> None:
    """印出單一策略的績效摘要。"""
    line = "=" * 60
    print()
    print(line)
    print("  績效摘要 / Performance Summary")
    print(line)
    print(f"  區間：{m.start} ~ {m.end}  ({m.duration_days:.1f} days)")
    print("-" * 60)
    print(f"  Initial Capital:     {m.initial_capital:>10.2f} USDT")
    print(f"  Final Equity:        {m.final_equity:>10.2f} USDT")
    print(f"  Total Return:        {m.total_return_pct:>+10.2f} %")
    print(f"  Annualized Return:   {m.annualized_return_pct:>+10.2f} %")
    print("-" * 60)
    print(f"  Sharpe Ratio:        {m.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:       {m.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:        {m.max_drawdown_pct:>10.2f} %")
    print(f"  Calmar Ratio:        {m.calmar_ratio:>10.2f}")
    print("-" * 60)
    print(f"  Total Trades:        {m.total_trades:>10d}")
    print(f"  Avg Trades / Day:    {m.avg_trades_per_day:>10.2f}")
    print(f"  Win Rate:            {m.win_rate_pct:>10.2f} %")
    print(f"  Profit Factor:       {m.profit_factor:>10.2f}")
    print(f"  Expectancy / trade:  {m.expectancy:>10.4f} USDT")
    print(f"  Avg Win / Loss:      {m.avg_win:>10.4f} / {m.avg_loss:.4f}")
    print(f"  Max Consec W / L:    {m.max_consecutive_wins:>10d} / {m.max_consecutive_losses}")
    print(f"  Avg Hold (bars):     {m.avg_holding_bars:>10.2f}")
    print("-" * 60)
    print(f"  Stop Loss (intraday):{m.stop_loss_count:>10d}")
    print(f"  Stop Loss (gap):     {m.stop_loss_gap_count:>10d}")
    print("-" * 60)
    print(f"  訊號統計：emitted={signals_emitted}, filled={signals_filled}, "
          f"unfilled={signals_unfilled}, skipped_pending={signals_skipped}")
    print(line)
