"""Event-driven backtest 主迴圈。

把 strategy + broker + account 串起來。設計嚴格按照 ARCHITECTURE.md §3.5 + §7：

對每根 K 線 i 的處理順序：

    1. 撮合 pending limit 單（在 bar[i].open 撮合）
       - limit_price 在此時才計算（= bar.open ± slippage_pct）
       - quantity = (account.equity * position_size_pct) / limit_price
       - 成功 → 開倉；失敗 → 訊號作廢

    2. 盤中止損檢查（broker.check_stop 用 bar.high/low）
       - 帶倉跨 bar：若 bar.open 已穿越 stop → 跳空平倉於 bar.open
       - 同根進場：不適用跳空規則

    3. 收盤後策略動作：
       3a. 持倉中 → ratchet 拖曳止損
           （多單只能上移、空單只能下移）
       3b. 無倉位且無 pending → detect_entry 是否該進場

    4. 紀錄 equity（mark = bar.close）

設計原則（CLAUDE.md §10 預留接口）：
- engine 不知道是「多策略」還是「空策略」，只接受 BaseTrendStrategy
- 將來實盤 broker 滿足相同 protocol 即可替換
"""

from __future__ import annotations

import logging
import math

import pandas as pd
from tqdm import tqdm

from src.backtest.types import BacktestResult, EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import Bar, LimitOrder
from src.strategy.base import BaseTrendStrategy, assert_indicators_ready
from src.utils.exceptions import OrderExecutionError
from src.utils.types import Direction
from src.utils.validation import validate_ohlc

logger = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,
    strategy: BaseTrendStrategy,
    account: Account,
    broker: BrokerSimulator,
    config: EngineConfig | None = None,
    *,
    show_progress: bool = False,
) -> BacktestResult:
    """執行回測。

    Args:
        df: 已含全部指標欄的 DataFrame（``prepare_indicators`` 的輸出）。
        strategy: 已綁定 params 的策略實例。
        account: 帳戶實例（caller 提供，方便外部觀察 trade_log）。
        broker: 撮合器（持有 fee / slippage 設定）。
        config: 引擎設定。預設 ``EngineConfig()`` 即倉位 60%。
        show_progress: 是否顯示 tqdm 進度條。

    Returns:
        BacktestResult，包含交易、equity 曲線、訊號統計。
    """
    config = config or EngineConfig()
    validate_ohlc(df, require_volume=True)
    assert_indicators_ready(df)

    n = len(df)
    if n == 0:
        raise ValueError("empty dataframe")

    pending_signal = None
    signals_emitted = 0
    signals_filled = 0
    signals_unfilled = 0
    signals_skipped_pending = 0

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc=f"[{account.name}] backtest")

    slippage = broker.config.slippage_pct

    for i in iterator:
        ts = df.index[i]
        row = df.iloc[i]
        bar = Bar.from_row(ts, row)

        # === Step 1: 撮合 pending limit 單於本根 K 線開盤 ===
        if pending_signal is not None:
            limit_price = _compute_limit_price(
                bar.open, pending_signal.direction, slippage
            )
            equity_now = account.equity(bar.open)
            if equity_now > 0 and limit_price > 0:
                quantity = (equity_now * config.position_size_pct) / limit_price

                # 防呆：跨 bar 後 stop 與 limit 相對位置可能變化，導致違反 stop_direction
                stop_ok = _stop_on_correct_side(
                    pending_signal.direction, pending_signal.initial_stop, limit_price
                )
                if quantity > 0 and stop_ok:
                    order = LimitOrder(
                        direction=pending_signal.direction,
                        limit_price=limit_price,
                        quantity=quantity,
                        initial_stop=pending_signal.initial_stop,
                    )
                    try:
                        result = broker.try_fill_limit(order, bar, account)
                    except OrderExecutionError:
                        logger.exception(
                            "fill failed at %s for %s", ts, pending_signal.direction
                        )
                        raise
                    if result.filled:
                        signals_filled += 1
                        logger.debug(
                            "FILLED %s @ %.4f qty=%.6f stop=%.4f",
                            pending_signal.direction,
                            result.fill_price,
                            quantity,
                            pending_signal.initial_stop,
                        )
                    else:
                        signals_unfilled += 1
                        logger.debug("UNFILLED %s: %s", pending_signal.direction, result.reason)
                else:
                    signals_unfilled += 1
                    logger.debug(
                        "SKIP_FILL %s: stop %.4f wrong side of limit %.4f",
                        pending_signal.direction,
                        pending_signal.initial_stop,
                        limit_price,
                    )
            else:
                signals_unfilled += 1
            pending_signal = None

        # === Step 2: 盤中止損檢查 ===
        if account.has_position():
            trade = broker.check_stop(account, bar)
            if trade is not None:
                logger.debug(
                    "STOP %s @ %.4f net_pnl=%.4f reason=%s",
                    trade.direction, trade.exit_price, trade.net_pnl, trade.exit_reason,
                )

        # === Step 3a: 收盤 ratchet 拖曳止損 ===
        if account.has_position():
            candidate = strategy.compute_trailing_stop_candidate(df, i)
            if not math.isnan(candidate):
                pos = account.position
                assert pos is not None
                if pos.direction is Direction.LONG and candidate > pos.stop_price:
                    account.update_stop(candidate)
                    logger.debug("RATCHET LONG stop %.4f -> %.4f", pos.stop_price, candidate)
                elif pos.direction is Direction.SHORT and candidate < pos.stop_price:
                    account.update_stop(candidate)
                    logger.debug("RATCHET SHORT stop %.4f -> %.4f", pos.stop_price, candidate)

        # === Step 3b: 收盤偵測進場訊號 ===
        if not account.has_position():
            new_signal = strategy.detect_entry(df, i)
            if new_signal is not None:
                signals_emitted += 1
                if pending_signal is not None and config.skip_signal_when_pending:
                    signals_skipped_pending += 1
                    logger.debug("SKIP_SIGNAL %s: pending order exists", new_signal.direction)
                else:
                    pending_signal = new_signal

        # === Step 4: 紀錄 equity ===
        account.snapshot_equity(bar.close, ts)

    # 收尾：最後一根後若仍有 pending_signal → 算未成交
    if pending_signal is not None:
        signals_unfilled += 1

    # 收尾：若仍有持倉，依 config 決定是否強平
    if account.has_position() and config.force_close_at_end:
        last_bar = Bar.from_row(df.index[-1], df.iloc[-1])
        # 用 close 強平，手續費用 taker
        pos = account.position
        assert pos is not None
        notional = pos.quantity * last_bar.close
        fee = notional * broker.config.taker_fee_rate
        account.close_position(
            exit_price=last_bar.close,
            exit_timestamp=last_bar.timestamp,
            fee=fee,
            reason="FORCE_CLOSE_END",
        )

    final_equity = account.equity(float(df.iloc[-1]["close"]))
    equity_curve = pd.Series(
        data=[v for _, v in account.equity_history],
        index=pd.DatetimeIndex([t for t, _ in account.equity_history], name="timestamp"),
        name="equity",
        dtype="float64",
    )

    return BacktestResult(
        account_name=account.name,
        initial_capital=account.initial_capital,
        final_equity=final_equity,
        trades=account.trade_log,
        equity_curve=equity_curve,
        bars_processed=n,
        signals_emitted=signals_emitted,
        signals_filled=signals_filled,
        signals_unfilled=signals_unfilled,
        signals_skipped_pending=signals_skipped_pending,
        config_snapshot={
            "position_size_pct": config.position_size_pct,
            "force_close_at_end": config.force_close_at_end,
            "skip_signal_when_pending": config.skip_signal_when_pending,
            "taker_fee_rate": broker.config.taker_fee_rate,
            "slippage_pct": broker.config.slippage_pct,
            "strategy_params": vars(strategy.params),
        },
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _compute_limit_price(
    bar_open: float, direction: Direction, slippage_pct: float
) -> float:
    """計算限價：多單 = open*(1+slip)，空單 = open*(1-slip)。"""
    if direction is Direction.LONG:
        return bar_open * (1.0 + slippage_pct)
    return bar_open * (1.0 - slippage_pct)


def _stop_on_correct_side(
    direction: Direction, stop_price: float, limit_price: float
) -> bool:
    """限價成交前快速檢查 stop 與 limit 的相對位置合不合法。"""
    if direction is Direction.LONG:
        return stop_price < limit_price
    return stop_price > limit_price
