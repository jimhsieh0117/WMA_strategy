"""Event-driven backtest 主迴圈。

把 strategy + broker + account + trailing controller 串起來。對應
ARCHITECTURE.md §3.5 + §7 + §11。

對每根 K 線 i 的處理順序：

    1. 撮合 pending limit 單（在 bar[i].open 撮合）
       - limit_price = bar.open ± slippage_pct
       - quantity 由 sizing_mode 決定
       - allow_pyramiding=True 時走多倉路徑（open_position_multi）
       - 成功 → 開倉 + 立即 instantiate TrailingStopController（存到 trailings[pid]）
       - 失敗 → 訊號作廢

    2. 盤中止損檢查：
       - 單倉模式：broker.check_stop
       - 多倉模式：broker.check_stops（遍歷所有 position）
       - 觸發 → 平倉 + 移除對應 controller

    3. 收盤後策略動作：
       3a. 對每筆持倉：controller.update(...) 取得新 stop（三階段狀態機 + ratchet）
       3b. allow_pyramiding=False 且已有持倉 → 不偵測新訊號
           其餘情況 → strategy.detect_entry → pending_signal（每根 K 最多 1 個 pending）

    4. 紀錄 equity（mark = bar.close）
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
from src.strategy.trailing import TrailingStopController
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
        config: 引擎設定。預設 ``EngineConfig()`` 即倉位 60%、單倉模式。
        show_progress: 是否顯示 tqdm 進度條。
    """
    config = config or EngineConfig()
    validate_ohlc(df, require_volume=True)
    assert_indicators_ready(df)

    n = len(df)
    if n == 0:
        raise ValueError("empty dataframe")

    pending_signal = None
    # 每筆持倉一個 controller，key = position_id
    trailings: dict[int, TrailingStopController] = {}

    signals_emitted = 0
    signals_filled = 0
    signals_unfilled = 0
    signals_skipped_pending = 0

    iterator = range(n)
    if show_progress:
        iterator = tqdm(iterator, total=n, desc=f"[{account.name}] backtest")

    slippage = broker.config.slippage_pct
    trailing_params = strategy.params.trailing
    r_cap_params = strategy.params.r_cap

    for i in iterator:
        ts = df.index[i]
        row = df.iloc[i]
        bar = Bar.from_row(ts, row)

        # 進場時段黑名單：fill bar 的 hour 命中 → 拒絕 pending 並清除（fall-through 到 Step 2）
        if (pending_signal is not None
                and config.entry_hour_blacklist
                and ts.hour in config.entry_hour_blacklist):
            signals_unfilled += 1
            logger.debug(
                "SKIP_FILL %s: hour %d in entry_hour_blacklist",
                pending_signal.direction, ts.hour,
            )
            pending_signal = None

        # === Step 1: 撮合 pending limit 單於本根 K 線開盤 ===
        if pending_signal is not None:
            limit_price = _compute_limit_price(
                bar.open, pending_signal.direction, slippage
            )
            equity_now = account.equity(bar.open)
            if equity_now > 0 and limit_price > 0:
                quantity, target_risk, sizing_ok, sizing_reason = _compute_quantity(
                    config=config,
                    direction=pending_signal.direction,
                    limit_price=limit_price,
                    initial_stop=pending_signal.initial_stop,
                    equity_now=equity_now,
                    taker_fee_rate=broker.config.taker_fee_rate,
                    existing_notional=account.total_notional_at_entry,
                )

                stop_ok = _stop_on_correct_side(
                    pending_signal.direction, pending_signal.initial_stop, limit_price
                )
                if sizing_ok and quantity > 0 and stop_ok:
                    order = LimitOrder(
                        direction=pending_signal.direction,
                        limit_price=limit_price,
                        quantity=quantity,
                        initial_stop=pending_signal.initial_stop,
                        target_risk_usdt=target_risk,
                    )
                    try:
                        result = broker.try_fill_limit(
                            order, bar, account,
                            allow_multi=config.allow_pyramiding,
                        )
                    except OrderExecutionError:
                        logger.exception(
                            "fill failed at %s for %s", ts, pending_signal.direction
                        )
                        raise
                    if result.filled:
                        signals_filled += 1
                        pid = result.position_id
                        assert pid is not None
                        new_pos = account.position_by_id(pid)
                        # r_cap：以「過去 window 根 K」內的歷史 trades + 其他未平倉持倉
                        # 的初始 R 平均，當作 effective_R 的上限。
                        effective_r_override = _compute_effective_r_override(
                            account=account, df=df, bar_index=i,
                            r_cap=r_cap_params, exclude_pid=pid,
                            actual_r=abs(new_pos.entry_price - new_pos.stop_price),
                        )
                        controller = TrailingStopController(
                            position=new_pos,
                            params=trailing_params,
                            broker_config=broker.config,
                            effective_r_override=effective_r_override,
                        )
                        trailings[pid] = controller
                        logger.debug(
                            "FILLED %s @ %.4f qty=%.6f stop=%.4f R=%.6f effR=%.6f "
                            "abnormal_R=%s pid=%d",
                            pending_signal.direction,
                            result.fill_price, quantity,
                            pending_signal.initial_stop,
                            controller.R, controller.effective_R,
                            controller.is_abnormal_r, pid,
                        )
                    else:
                        signals_unfilled += 1
                        logger.debug("UNFILLED %s: %s", pending_signal.direction, result.reason)
                else:
                    signals_unfilled += 1
                    if not sizing_ok:
                        logger.debug(
                            "SKIP_FILL %s: sizing rejected (%s, mode=%s, "
                            "limit=%.4f stop=%.4f equity=%.4f)",
                            pending_signal.direction, sizing_reason, config.sizing_mode,
                            limit_price, pending_signal.initial_stop, equity_now,
                        )
                    else:
                        logger.debug(
                            "SKIP_FILL %s: stop %.4f wrong side of limit %.4f",
                            pending_signal.direction,
                            pending_signal.initial_stop,
                            limit_price,
                        )
            else:
                signals_unfilled += 1
            pending_signal = None

        # === Step 2: 盤中止損檢查（多倉一次處理所有持倉）===
        if account.has_position():
            stop_meta = {
                pid: (ctrl.stage, ctrl.peak_progress_r)
                for pid, ctrl in trailings.items()
            }
            closed_trades = broker.check_stops(account, bar, metadata_by_pid=stop_meta)
            for trade in closed_trades:
                logger.debug(
                    "STOP %s @ %.4f net_pnl=%.4f reason=%s pid=%d",
                    trade.direction, trade.exit_price, trade.net_pnl,
                    trade.exit_reason, trade.position_id,
                )
                trailings.pop(trade.position_id, None)

        # === Step 3a: 收盤 ratchet 拖曳止損（每筆獨立）===
        # 兩階段：先全部 update（更新 stage / peak / bars_observed），再依
        # should_early_exit 平倉。避免在 iteration 中改變 account.positions。
        early_exit_pids: list[int] = []
        for pid, pos in account.positions.items():
            controller = trailings.get(pid)
            if controller is None:
                continue
            new_stop = controller.update(
                bar=bar, df=df, bar_index=i,
                current_stop=pos.stop_price,
            )
            if new_stop is not None:
                old_stop = pos.stop_price
                account.update_stop_by_id(pid, new_stop, timestamp=ts)
                logger.debug(
                    "RATCHET %s pid=%d stop %.4f -> %.4f (stage=%d)",
                    pos.direction, pid, old_stop, new_stop, controller.stage,
                )
            if controller.should_early_exit():
                early_exit_pids.append(pid)

        # === Step 3a.5: Early-exit cancel（觀測期內未達浮盈門檻）===
        for pid in early_exit_pids:
            pos = account.positions.get(pid)
            if pos is None:
                continue  # 同一 bar 同時被 stage 1 stop 觸發 → 已平倉
            controller = trailings.pop(pid, None)
            exit_price = float(bar.close)
            notional = pos.quantity * exit_price
            fee = notional * broker.config.taker_fee_rate
            account.close_position_by_id(
                position_id=pid,
                exit_price=exit_price,
                exit_timestamp=ts,
                fee=fee,
                reason="EARLY_CANCEL",
                final_stage=1,
                peak_progress_r=controller.peak_progress_r if controller else 0.0,
            )
            logger.debug(
                "EARLY_CANCEL %s pid=%d @ %.4f (peak_r=%.3f < %.3f)",
                pos.direction, pid, exit_price,
                controller.peak_progress_r if controller else 0.0,
                trailing_params.early_exit_min_peak_r,
            )

        # === Step 3b: 收盤偵測進場訊號 ===
        # 進場閘門：
        # - allow_pyramiding=False：維持舊版「無持倉才偵測」
        # - allow_pyramiding=True：每根 K 都偵測（pending 機制保證 1 K = 1 新倉）
        can_detect = config.allow_pyramiding or not account.has_position()
        if can_detect:
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

    # 收尾
    if pending_signal is not None:
        signals_unfilled += 1

    if account.has_position() and config.force_close_at_end:
        last_bar = Bar.from_row(df.index[-1], df.iloc[-1])
        for pid, pos in list(account.positions.items()):
            notional = pos.quantity * last_bar.close
            fee = notional * broker.config.taker_fee_rate
            ctrl = trailings.get(pid)
            account.close_position_by_id(
                position_id=pid,
                exit_price=last_bar.close,
                exit_timestamp=last_bar.timestamp,
                fee=fee,
                reason="FORCE_CLOSE_END",
                final_stage=ctrl.stage if ctrl else 1,
                peak_progress_r=ctrl.peak_progress_r if ctrl else 0.0,
            )
            trailings.pop(pid, None)

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
            "sizing_mode": config.sizing_mode,
            "position_size_pct": config.position_size_pct,
            "risk_per_trade_usdt": config.risk_per_trade_usdt,
            "risk_per_trade_pct": config.risk_per_trade_pct,
            "allow_pyramiding": config.allow_pyramiding,
            "leverage_cap": config.leverage_cap,
            "force_close_at_end": config.force_close_at_end,
            "skip_signal_when_pending": config.skip_signal_when_pending,
            "taker_fee_rate": broker.config.taker_fee_rate,
            "slippage_pct": broker.config.slippage_pct,
            "strategy_params": _serialize_params(strategy.params),
        },
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _compute_limit_price(
    bar_open: float, direction: Direction, slippage_pct: float
) -> float:
    """計算限價：多單 = open × (1+slip)、空單 = open × (1−slip)。"""
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


def _compute_quantity(
    *,
    config: EngineConfig,
    direction: Direction,
    limit_price: float,
    initial_stop: float,
    equity_now: float,
    taker_fee_rate: float,
    existing_notional: float = 0.0,
) -> tuple[float, float | None, bool, str]:
    """依 ``sizing_mode`` 計算下單 quantity，並做槓桿上限檢查。

    多倉模式下：
        Σ open_notional + new_notional ≤ equity_now × leverage_cap
    違反則拒絕該筆進場（fail-fast，不嘗試降低倉位硬擠進去）。

    Returns:
        (quantity, target_risk_usdt | None, sizing_ok, reason)
    """
    if config.sizing_mode == "pct":
        # pct 模式：以「剩餘可用 equity」乘上比例計算 notional，避免在多倉模式下
        # 反覆以同樣 60% 開倉導致 Σnotional 失控
        if config.allow_pyramiding:
            cap_total = equity_now * config.leverage_cap
            available = cap_total - existing_notional
            if available <= 0:
                return 0.0, None, False, "leverage_cap_exhausted"
            target_notional = min(available, equity_now * config.position_size_pct)
        else:
            target_notional = equity_now * config.position_size_pct
        quantity = target_notional / limit_price
        return quantity, None, True, "ok"

    # risk 模式：先做 R 噪音門檻（R_price / limit_price < r_min_pct → 拒絕）
    r_price = abs(limit_price - initial_stop)
    if config.r_min_pct > 0 and limit_price > 0:
        r_pct = r_price / limit_price
        if r_pct < config.r_min_pct:
            return 0.0, None, False, (
                f"r_too_small (r_pct={r_pct:.5f} < r_min_pct={config.r_min_pct:.5f})"
            )

    # 動態 risk：pct > 0 啟用「equity × pct」，否則沿用固定 USDT
    if config.risk_per_trade_pct > 0:
        if equity_now <= 0:
            return 0.0, None, False, f"equity_non_positive ({equity_now:.4f})"
        effective_risk_usdt = equity_now * config.risk_per_trade_pct
    else:
        effective_risk_usdt = config.risk_per_trade_usdt

    # risk 模式：qty = R / [|limit-stop| + (limit+stop)×taker]
    denom = r_price + (limit_price + initial_stop) * taker_fee_rate
    if denom <= 0:
        return 0.0, None, False, "denom_non_positive"
    quantity = effective_risk_usdt / denom
    notional = quantity * limit_price

    if config.allow_pyramiding:
        cap_total = equity_now * config.leverage_cap
        if existing_notional + notional > cap_total:
            return 0.0, None, False, (
                f"leverage_cap_exceeded (existing={existing_notional:.2f}, "
                f"new={notional:.2f}, cap={cap_total:.2f})"
            )
        return quantity, effective_risk_usdt, True, "ok"

    # 單倉模式維持舊行為：notional > equity 視為過槓桿
    if notional > equity_now:
        return 0.0, None, False, "single_position_overleveraged"
    return quantity, effective_risk_usdt, True, "ok"


def _compute_effective_r_override(
    *,
    account: Account,
    df: pd.DataFrame,
    bar_index: int,
    r_cap,
    exclude_pid: int,
    actual_r: float,
) -> float | None:
    """r_cap 啟用時計算 effective_R override，否則回傳 None（controller 走實際 R）。

    窗口：``[df.index[max(0, bar_index - window)] .. df.index[bar_index]]``，
    把歷史 trades 與其他未平倉持倉的「初始 R」（``stop_history[0]``）一起取平均。
    僅當 ``avg_R < actual_r`` 才實際 cap；否則回 None（不變）。
    窗口內無樣本 → 回 None（fallback 用實際 R）。
    """
    if r_cap.mode == "off":
        return None

    window = int(r_cap.window)
    start_idx = max(0, bar_index - window)
    window_start_ts = df.index[start_idx]

    rs: list[float] = []
    for trade in account.trade_log:
        if trade.entry_timestamp < window_start_ts:
            continue
        if not trade.stop_history:
            continue
        r = abs(float(trade.entry_price) - float(trade.stop_history[0][1]))
        if r > 0:
            rs.append(r)
    for pid, pos in account.positions.items():
        if pid == exclude_pid:
            continue
        if pos.entry_timestamp < window_start_ts:
            continue
        if not pos.stop_history:
            continue
        r = abs(float(pos.entry_price) - float(pos.stop_history[0][1]))
        if r > 0:
            rs.append(r)

    if not rs:
        return None
    avg_r = sum(rs) / len(rs)
    if avg_r >= actual_r:
        return None
    return avg_r


def _serialize_params(params) -> dict:
    """把 StrategyParams（含 nested TrailingStopParams）攤平成 dict。"""
    from dataclasses import asdict
    return asdict(params)
