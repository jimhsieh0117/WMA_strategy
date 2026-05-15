"""LiveSimulator：對每根新 closed K 跑 engine step（per-direction）。

工程考量：
- engine.run_backtest 是「從頭跑整個 df」設計，不能 resume；live 環境需要增量
  step。為避免 engine.py 大改、保持回測 264 個測試綠燈，這裡複製 engine.py
  for-loop 內部邏輯成 ``_process_bar_for_direction``。
- **未來 refactor**：把 engine.py for-loop body extract 為共用 helper，
  此處改 import。當前選擇「複製 + 文件標記」優先讓 live 功能跑起來。
- 兩個 strategy 共用同一根原始 K 線（bar），但各自有自己的 df（WMA per-direction
  的 indicator column 不同），及各自的 account / trailings / pending_signal。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.backtest.engine import (
    _compute_effective_r_override,
    _compute_limit_price,
    _compute_quantity,
    _stop_on_correct_side,
)
from src.backtest.types import EngineConfig
from src.broker.account import Account
from src.broker.simulator import BrokerSimulator
from src.broker.types import Bar, LimitOrder, Trade
from src.live.state import LiveEvent, LiveState
from src.strategy.base import BaseTrendStrategy, prepare_indicators
from src.strategy.trailing import TrailingStopController
from src.strategy.types import EntrySignal
from src.utils.exceptions import OrderExecutionError
from src.utils.types import Direction

logger = logging.getLogger(__name__)


class LiveSimulator:
    """每根新 closed K 收到後跑兩個 strategy 各自的 engine step。"""

    def __init__(
        self,
        *,
        state: LiveState,
        engine_config: EngineConfig,
        broker: BrokerSimulator,
        output_dir: Path,
    ) -> None:
        self.state = state
        self.engine_cfg = engine_config
        self.broker = broker
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trades_csv = output_dir / "trades.csv"
        self.equity_csv = output_dir / "equity.csv"
        self._trades_header_written = self.trades_csv.exists()
        self._equity_header_written = self.equity_csv.exists()

        # broadcast 用 callbacks（server.py 註冊）：async (LiveEvent) -> None
        self._on_event_async = []                                          # type: ignore[var-annotated]

    def register_async_event_handler(self, handler) -> None:
        self._on_event_async.append(handler)

    # ------------------------------------------------------------------ #
    # Feed callbacks
    # ------------------------------------------------------------------ #

    async def on_warmup_loaded(self, raw_df: pd.DataFrame) -> None:
        """LiveCandleFeed.warmup() 完成後呼叫一次。

        - 對 raw_df 用 long / short 各自 params 跑 ``prepare_indicators``
        - 存入 state.df_long / state.df_short
        - 標記 is_warmed_up=True；之後 on_new_closed_bar 才會真的處理
        - 暖機期間不執行任何 strategy.detect_entry / trailing；
          只是建立 indicator baseline
        """
        long_df = prepare_indicators(raw_df, self.state.long_strategy.params)
        short_df = prepare_indicators(raw_df, self.state.short_strategy.params)
        self.state.df_long = long_df
        self.state.df_short = short_df
        self.state.started_at = pd.Timestamp.now(tz="UTC").tz_convert(None)
        self.state.last_processed_ts = raw_df.index[-1]
        self.state.bars_processed = len(raw_df)
        self.state.is_warmed_up = True
        await self._emit(LiveEvent(
            ts_unix=int(raw_df.index[-1].timestamp()),
            kind="WARMUP",
            payload={
                "bars_loaded": len(raw_df),
                "start": str(raw_df.index[0]),
                "end": str(raw_df.index[-1]),
            },
        ))
        logger.info(
            "warmup complete: %d bars (%s ~ %s); ready to process live K bars",
            len(raw_df), raw_df.index[0], raw_df.index[-1],
        )

    async def on_new_closed_bar(
        self,
        ts: pd.Timestamp, o: float, h: float, l: float, c: float, v: float,
    ) -> None:
        """每根新 closed K 收到時觸發。Append → recompute indicators → step both strategies。"""
        if not self.state.is_warmed_up:
            logger.warning("Received bar %s before warmup complete; ignoring", ts)
            return

        # === 1. Append raw bar 到 df_long / df_short ===
        new_row = pd.DataFrame(
            {"open": [o], "high": [h], "low": [l], "close": [c], "volume": [v]},
            index=[ts],
        )
        # 把 OHLCV 新 row 加到既有 df，然後對「合併後」重算 indicators（簡單可靠）
        raw_long = self.state.df_long[["open", "high", "low", "close", "volume"]]
        raw_short = self.state.df_short[["open", "high", "low", "close", "volume"]]
        raw_long_new = pd.concat([raw_long, new_row], axis=0)
        raw_short_new = pd.concat([raw_short, new_row], axis=0)
        self.state.df_long = prepare_indicators(
            raw_long_new, self.state.long_strategy.params,
        )
        self.state.df_short = prepare_indicators(
            raw_short_new, self.state.short_strategy.params,
        )

        # bar 物件（用 raw OHLCV）
        bar = Bar(timestamp=ts, open=o, high=h, low=l, close=c)
        bar_index_long = len(self.state.df_long) - 1
        bar_index_short = len(self.state.df_short) - 1

        await self._emit(LiveEvent(
            ts_unix=int(ts.timestamp()),
            kind="BAR",
            payload={
                "open": o, "high": h, "low": l, "close": c, "volume": v,
                "time": str(ts),
            },
        ))

        # === 2. Long / Short 各跑 engine step ===
        await self._process_bar_for_direction(
            direction="long",
            bar=bar,
            bar_index=bar_index_long,
            df=self.state.df_long,
            account=self.state.long_account,
            strategy=self.state.long_strategy,
            trailings=self.state.trailings_long,
            pending_attr="pending_long",
        )
        await self._process_bar_for_direction(
            direction="short",
            bar=bar,
            bar_index=bar_index_short,
            df=self.state.df_short,
            account=self.state.short_account,
            strategy=self.state.short_strategy,
            trailings=self.state.trailings_short,
            pending_attr="pending_short",
        )

        self.state.last_processed_ts = ts
        self.state.bars_processed += 1

        # === 3. 寫 equity log + broadcast ===
        total_eq = self.state.total_equity(c)
        long_eq = self.state.long_account.equity(c)
        short_eq = self.state.short_account.equity(c)
        self._append_equity_row(ts, long_eq, short_eq, total_eq)
        await self._emit(LiveEvent(
            ts_unix=int(ts.timestamp()),
            kind="EQUITY",
            payload={
                "long": long_eq, "short": short_eq, "total": total_eq,
            },
        ))

    # ------------------------------------------------------------------ #
    # Per-direction step（複製 engine.py for-loop body 邏輯）
    # ------------------------------------------------------------------ #

    async def _process_bar_for_direction(
        self,
        *,
        direction: str,
        bar: Bar,
        bar_index: int,
        df: pd.DataFrame,
        account: Account,
        strategy: BaseTrendStrategy,
        trailings: dict[int, TrailingStopController],
        pending_attr: str,
    ) -> None:
        """單一 direction 一根 K 的處理流程。

        順序鏡像 engine.run_backtest for-loop body：
          1. hour blacklist 拒絕 pending
          2. 撮合 pending（bar.open）
          3. check stops（盤中價穿越 stop）
          4. trailing update + forced exits（time_cut / early_exit）
          5. detect entry（paused=True 時跳過）
          6. snapshot equity
        """
        ts = bar.timestamp
        slippage = self.broker.config.slippage_pct
        trailing_params = strategy.params.trailing
        r_cap_params = strategy.params.r_cap

        pending_signal: EntrySignal | None = getattr(self.state, pending_attr)

        # --- 1. hour blacklist ---
        if (pending_signal is not None
                and self.engine_cfg.entry_hour_blacklist
                and ts.hour in self.engine_cfg.entry_hour_blacklist):
            logger.debug(
                "SKIP_FILL %s: hour %d in blacklist", direction, ts.hour,
            )
            pending_signal = None

        # --- 2. 撮合 pending limit 單 ---
        if pending_signal is not None:
            await self._try_fill_pending(
                direction=direction, pending_signal=pending_signal,
                bar=bar, bar_index=bar_index, df=df,
                account=account, trailings=trailings,
                trailing_params=trailing_params, r_cap_params=r_cap_params,
                slippage=slippage,
            )
            pending_signal = None
            setattr(self.state, pending_attr, None)

        # --- 3. 盤中 stop hit ---
        if account.has_position():
            stop_meta = {
                pid: (ctrl.stage, ctrl.peak_progress_r)
                for pid, ctrl in trailings.items()
            }
            closed_trades = self.broker.check_stops(
                account, bar, metadata_by_pid=stop_meta,
            )
            for trade in closed_trades:
                trailings.pop(trade.position_id, None)
                await self._emit_trade_close(direction, trade)

        # --- 4. trailing update + forced exits ---
        forced_exits: list[tuple[int, str]] = []
        for pid, pos in account.positions.items():
            controller = trailings.get(pid)
            if controller is None:
                continue
            new_stop = controller.update(
                bar=bar, df=df, bar_index=bar_index,
                current_stop=pos.stop_price,
            )
            if new_stop is not None:
                old_stop = pos.stop_price
                account.update_stop_by_id(pid, new_stop, timestamp=ts)
                await self._emit(LiveEvent(
                    ts_unix=int(ts.timestamp()),
                    kind="RATCHET",
                    payload={
                        "direction": direction, "pid": pid,
                        "old_stop": old_stop, "new_stop": new_stop,
                        "stage": controller.stage,
                        "peak_r": controller.peak_progress_r,
                    },
                ))

            if controller.should_early_exit():
                forced_exits.append((pid, "EARLY_CANCEL"))
            elif controller.should_time_cut():
                forced_exits.append((pid, "TIME_CUT"))

        # --- 4.5 Forced exit ---
        for pid, reason in forced_exits:
            pos = account.positions.get(pid)
            if pos is None:
                continue
            controller = trailings.pop(pid, None)
            exit_price = float(bar.close)
            notional = pos.quantity * exit_price
            fee = notional * self.broker.config.taker_fee_rate
            trade = account.close_position_by_id(
                position_id=pid,
                exit_price=exit_price,
                exit_timestamp=ts,
                fee=fee,
                reason=reason,
                final_stage=1,
                peak_progress_r=controller.peak_progress_r if controller else 0.0,
            )
            if trade is not None:
                await self._emit_trade_close(direction, trade)

        # --- 5. detect entry（paused 時跳過）---
        if not self.state.paused:
            can_detect = self.engine_cfg.allow_pyramiding or not account.has_position()
            if can_detect:
                new_signal = strategy.detect_entry(df, bar_index)
                if new_signal is not None:
                    setattr(self.state, pending_attr, new_signal)
                    await self._emit(LiveEvent(
                        ts_unix=int(ts.timestamp()),
                        kind="SIGNAL_PENDING",
                        payload={
                            "direction": direction,
                            "initial_stop": new_signal.initial_stop,
                            "reason": new_signal.reason,
                        },
                    ))

        # --- 6. snapshot equity ---
        account.snapshot_equity(bar.close, ts)

    # ------------------------------------------------------------------ #
    # Fill pending（複製 engine.py Step 1 邏輯）
    # ------------------------------------------------------------------ #

    async def _try_fill_pending(
        self,
        *,
        direction: str,
        pending_signal: EntrySignal,
        bar: Bar,
        bar_index: int,
        df: pd.DataFrame,
        account: Account,
        trailings: dict[int, TrailingStopController],
        trailing_params,
        r_cap_params,
        slippage: float,
    ) -> None:
        limit_price = _compute_limit_price(
            bar.open, pending_signal.direction, slippage,
        )
        equity_now = account.equity(bar.open)
        if equity_now <= 0 or limit_price <= 0:
            return
        quantity, target_risk, sizing_ok, sizing_reason = _compute_quantity(
            config=self.engine_cfg,
            direction=pending_signal.direction,
            limit_price=limit_price,
            initial_stop=pending_signal.initial_stop,
            equity_now=equity_now,
            taker_fee_rate=self.broker.config.taker_fee_rate,
            existing_notional=account.total_notional_at_entry,
        )
        stop_ok = _stop_on_correct_side(
            pending_signal.direction, pending_signal.initial_stop, limit_price,
        )
        if not (sizing_ok and quantity > 0 and stop_ok):
            logger.debug(
                "UNFILLED %s: sizing=%s qty=%.6f stop_ok=%s",
                direction, sizing_reason, quantity, stop_ok,
            )
            return

        order = LimitOrder(
            direction=pending_signal.direction,
            limit_price=limit_price,
            quantity=quantity,
            initial_stop=pending_signal.initial_stop,
            target_risk_usdt=target_risk,
        )
        try:
            result = self.broker.try_fill_limit(
                order, bar, account,
                allow_multi=self.engine_cfg.allow_pyramiding,
            )
        except OrderExecutionError:
            logger.exception("fill failed at %s for %s", bar.timestamp, direction)
            raise
        if not result.filled:
            return

        pid = result.position_id
        assert pid is not None
        new_pos = account.position_by_id(pid)
        effective_r_override = _compute_effective_r_override(
            account=account, df=df, bar_index=bar_index,
            r_cap=r_cap_params, exclude_pid=pid,
            actual_r=abs(new_pos.entry_price - new_pos.stop_price),
        )
        controller = TrailingStopController(
            position=new_pos,
            params=trailing_params,
            broker_config=self.broker.config,
            effective_r_override=effective_r_override,
            r_min_pct=self.engine_cfg.r_min_pct,
        )
        trailings[pid] = controller

        await self._emit(LiveEvent(
            ts_unix=int(bar.timestamp.timestamp()),
            kind="TRADE_OPEN",
            payload={
                "direction": direction, "pid": pid,
                "entry_price": result.fill_price, "quantity": quantity,
                "stop_price": pending_signal.initial_stop,
                "R": controller.R,
                "target_risk_usdt": target_risk,
            },
        ))

    # ------------------------------------------------------------------ #
    # Event emit + log
    # ------------------------------------------------------------------ #

    async def _emit_trade_close(self, direction: str, trade: Trade) -> None:
        await self._emit(LiveEvent(
            ts_unix=int(trade.exit_timestamp.timestamp()),
            kind="TRADE_CLOSE",
            payload={
                "direction": direction, "pid": int(trade.position_id),
                "entry_time": int(trade.entry_timestamp.timestamp()),
                "exit_time": int(trade.exit_timestamp.timestamp()),
                "entry_price": float(trade.entry_price),
                "exit_price": float(trade.exit_price),
                "quantity": float(trade.quantity),
                "net_pnl": float(trade.net_pnl),
                "exit_reason": trade.exit_reason,
                "final_stage": int(trade.final_stage),
                "peak_progress_r": float(trade.peak_progress_r),
            },
        ))
        self._append_trade_row(direction, trade)

    async def _emit(self, event: LiveEvent) -> None:
        self.state.append_event(event)
        for handler in self._on_event_async:
            try:
                await handler(event)
            except Exception:
                logger.exception("event handler failed for %s", event.kind)

    def _append_trade_row(self, direction: str, trade: Trade) -> None:
        row = {
            "direction": direction,
            "pid": trade.position_id,
            "entry_time": trade.entry_timestamp.isoformat(),
            "exit_time": trade.exit_timestamp.isoformat(),
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "net_pnl": trade.net_pnl,
            "return_pct": trade.return_pct,
            "exit_reason": trade.exit_reason,
            "final_stage": trade.final_stage,
            "peak_progress_r": trade.peak_progress_r,
        }
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            self.trades_csv, mode="a", index=False,
            header=not self._trades_header_written,
        )
        self._trades_header_written = True

    def _append_equity_row(
        self, ts: pd.Timestamp, long_eq: float, short_eq: float, total: float,
    ) -> None:
        df_row = pd.DataFrame([{
            "time": ts.isoformat(),
            "long_equity": long_eq,
            "short_equity": short_eq,
            "total_equity": total,
        }])
        df_row.to_csv(
            self.equity_csv, mode="a", index=False,
            header=not self._equity_header_written,
        )
        self._equity_header_written = True
