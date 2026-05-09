"""三階段拖曳止損狀態機。

對應 ARCHITECTURE.md §11。Engine 在每筆持倉成交後 instantiate 一個
``TrailingStopController`` 實例，每根 K 線收盤時呼叫 ``update()`` 取得新止損。
持倉平倉後 controller 廢棄。

階段定義：
    Stage 1：剛進場。stop = 進場 K 線「前 N 根」反向極值 ± slippage_buffer
    Stage 2：價格達 stage2_trigger_r。stop 拉到「保本 + 雙向手續費 + 滑點 + buffer_r×R」
    Stage 3：價格達 stage3_trigger_r。stop 跟著 Bollinger lower（多）/ upper（空）band。
            Stage 2 fixed 仍作為 floor / ceiling，與 Bollinger 取較有利者。

異常 R 處理（``R < taker×2 + slippage``）：trigger 全部 ×2（1.2→2.4、2.4→4.8）

ratchet 規則：stop 永遠只能往有利方向移動；不利方向回拉時維持原值。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from src.broker.types import Bar, BrokerConfig, Position
from src.strategy.types import TrailingStopParams
from src.utils.types import Direction

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage 變化通知（給 engine 用於 logging / 統計）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StageTransition:
    """Stage 從 ``from_stage`` 進到 ``to_stage`` 的紀錄。"""

    from_stage: int
    to_stage: int
    bar_timestamp: pd.Timestamp
    progress_r: float


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #

class TrailingStopController:
    """三階段止損狀態機，生命週期 = 一筆持倉。

    與 Position / Account 解耦：constructor 拍下進場時的 entry_price / initial_stop，
    之後不依賴 Position 的可變狀態（除了 update 時讀 current_stop 做 ratchet 比較）。
    """

    def __init__(
        self,
        position: Position,
        params: TrailingStopParams,
        broker_config: BrokerConfig,
    ) -> None:
        if position.entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {position.entry_price}")
        if position.stop_price <= 0:
            raise ValueError(f"stop_price must be > 0, got {position.stop_price}")

        self.params = params
        self.broker_config = broker_config
        self.direction = position.direction
        self.entry_price = float(position.entry_price)
        self.initial_stop = float(position.stop_price)

        # R = 風險距離（進場價到初始止損的絕對差）
        self.R = abs(self.entry_price - self.initial_stop)
        if self.R <= 0:
            raise ValueError(
                f"R = |entry - initial_stop| must be > 0, got {self.R}"
            )

        # 異常 R 偵測：R < 雙向手續費的價格距離。
        # 滑點已內含在 entry_price 中（engine 用 open × (1 ± slippage_pct) 為 limit），
        # 因此「成本」只計入需在 stop 平倉時補回的雙向 taker fee；slippage 不重複計。
        cost_pct = 2 * broker_config.taker_fee_rate
        cost_distance = self.entry_price * cost_pct
        self.is_abnormal_r = self.R < cost_distance

        # 設定觸發點
        if self.is_abnormal_r:
            self.stage2_trigger = params.stage2_abnormal_trigger_r
            self.stage3_trigger = params.stage3_abnormal_trigger_r
        else:
            self.stage2_trigger = params.stage2_normal_trigger_r
            self.stage3_trigger = params.stage3_normal_trigger_r

        self.stage = 1
        # 追蹤歷史最大有利進度（單位 R），給 r_ladder 模式判斷觸發過的最高檔
        self.peak_progress_r = 0.0
        self._transitions: list[StageTransition] = []

        logger.debug(
            "TrailingStopController init: dir=%s entry=%.6f initial_stop=%.6f "
            "R=%.6f abnormal_r=%s stage2_trigger=%.2f stage3_trigger=%.2f",
            self.direction, self.entry_price, self.initial_stop,
            self.R, self.is_abnormal_r, self.stage2_trigger, self.stage3_trigger,
        )

    # ----------------------------------------------------------------------- #
    # 主介面：每根 K 線收盤後 engine 呼叫
    # ----------------------------------------------------------------------- #

    def update(
        self,
        bar: Bar,
        df: pd.DataFrame,
        bar_index: int,
        current_stop: float,
    ) -> float | None:
        """檢查階段是否推進、計算新止損候選值並回傳（不更新即回 None）。

        Args:
            bar: 當根 K 線。
            df: 完整 DataFrame（含 Bollinger 欄位）。
            bar_index: bar 在 df 中的位置。
            current_stop: 帳戶當前止損（caller 傳入做 ratchet 比較）。

        Returns:
            新的止損價（要 ratchet）或 None（保持不動）。
        """
        # 1. 用 bar 的有利方向極值衡量本根 K 達到的進度（多→high、空→low）
        progress_r = self._compute_progress_r(bar)
        if progress_r > self.peak_progress_r:
            self.peak_progress_r = progress_r

        # 2. 階段推進（單向）
        if self.stage == 1 and progress_r >= self.stage2_trigger:
            self._transition(2, bar.timestamp, progress_r)
        if self.stage == 2 and progress_r >= self.stage3_trigger:
            self._transition(3, bar.timestamp, progress_r)

        # 3. 取得本階段的止損候選
        candidate = self._compute_candidate(df, bar_index)
        if candidate is None or math.isnan(candidate):
            return None

        # 4. ratchet：只往有利方向動
        if self._is_more_favorable(candidate, current_stop):
            return candidate
        return None

    @property
    def transitions(self) -> list[StageTransition]:
        """回傳目前累積的 stage 變化紀錄（防外部修改）。"""
        return list(self._transitions)

    # ----------------------------------------------------------------------- #
    # 序列化（給 live_sim crash-resume 用；持倉資訊由外部 Position 還原後重建）
    # ----------------------------------------------------------------------- #

    def snapshot_runtime(self) -> dict:
        """匯出 runtime 狀態（stage / peak_progress_r）。

        靜態欄位（entry_price / R / triggers）會由 constructor 從 Position 與 params
        重新推導，故無須序列化。
        """
        return {
            "stage": int(self.stage),
            "peak_progress_r": float(self.peak_progress_r),
        }

    def restore_runtime(self, snapshot: dict) -> None:
        """以 ``snapshot_runtime`` 的輸出還原 stage 與 peak（不補回 transitions log）。"""
        stage = int(snapshot["stage"])
        if stage not in (1, 2, 3):
            raise ValueError(f"invalid stage in snapshot: {stage}")
        peak = float(snapshot["peak_progress_r"])
        if peak < 0:
            raise ValueError(f"peak_progress_r must be >= 0, got {peak}")
        self.stage = stage
        self.peak_progress_r = peak

    # ----------------------------------------------------------------------- #
    # 內部：階段邏輯
    # ----------------------------------------------------------------------- #

    def _compute_progress_r(self, bar: Bar) -> float:
        """目前最大有利移動 = 多單 bar.high − entry / R；空單反之。"""
        if self.direction is Direction.LONG:
            return (bar.high - self.entry_price) / self.R
        return (self.entry_price - bar.low) / self.R

    def _transition(
        self, to_stage: int, ts: pd.Timestamp, progress_r: float
    ) -> None:
        from_stage = self.stage
        self.stage = to_stage
        self._transitions.append(
            StageTransition(
                from_stage=from_stage,
                to_stage=to_stage,
                bar_timestamp=ts,
                progress_r=progress_r,
            )
        )
        logger.debug(
            "stage %d -> %d at %s progress=%.2fR",
            from_stage, to_stage, ts, progress_r,
        )

    def _compute_candidate(
        self, df: pd.DataFrame, bar_index: int
    ) -> float | None:
        """以當前階段計算止損候選。

        Stage 1: 不更新（回 None）→ 保持初始 stop
        Stage 2: 保本 + 成本 + buffer×R 的固定值
        Stage 3: 取 Stage 2 fixed 與 Bollinger band 之中較有利者
        """
        if self.stage == 1:
            return None

        stage2_value = self._stage2_breakeven_stop()

        if self.stage == 2:
            return stage2_value

        # Stage 3: 依 mode 決定追蹤候選，再與 stage2_value 取較有利者作為 floor / ceiling
        if self.params.stage3_mode == "r_ladder":
            track_value = self._r_ladder_stop()
        else:
            track_value = self._bollinger_stop(df, bar_index)

        if track_value is None or math.isnan(track_value):
            return stage2_value

        if self.direction is Direction.LONG:
            return max(stage2_value, track_value)
        return min(stage2_value, track_value)

    def _stage2_breakeven_stop(self) -> float:
        """保本 + 雙向手續費 + buffer_r×R。

        滑點已隱含於 entry_price（fill = open × (1 ± slippage_pct)）；
        若再加 slippage_pct 將重複計算。回測模型未對 stop 模擬負滑，
        因此只需補回 entry/exit 兩次 taker fee 即可達真正保本。
        """
        cost_pct = 2 * self.broker_config.taker_fee_rate
        buffer = self.params.stage2_buffer_r * self.R
        if self.direction is Direction.LONG:
            return self.entry_price * (1 + cost_pct) + buffer
        return self.entry_price * (1 - cost_pct) - buffer

    def _r_ladder_stop(self) -> float | None:
        """R 倍數階梯：peak 跨 (first + k·step)R 後，stop 鎖到 (first + k·step − offset)R。

        normal: first=2.8, step=1.0, offset=0.3 →
            peak ≥ 2.8R → stop = 2.5R；peak ≥ 3.8R → stop = 3.5R；…
        abnormal（R < 雙向手續費）: first=5.6, step=2.0 →
            peak ≥ 5.6R → stop = 5.0R；peak ≥ 7.6R → stop = 7.0R；…
        """
        if self.is_abnormal_r:
            first = self.params.r_ladder_abnormal_first_trigger
            step = self.params.r_ladder_abnormal_step
            offset = self.params.r_ladder_abnormal_trigger_offset
        else:
            first = self.params.r_ladder_normal_first_trigger
            step = self.params.r_ladder_normal_step
            offset = self.params.r_ladder_trigger_offset

        if self.peak_progress_r < first:
            return None

        # 已觸發的最高檔 k：peak ≥ first + k·step
        k = int(math.floor((self.peak_progress_r - first) / step))
        stop_r = first + k * step - offset

        if self.direction is Direction.LONG:
            return self.entry_price + stop_r * self.R
        return self.entry_price - stop_r * self.R

    def _bollinger_stop(
        self, df: pd.DataFrame, bar_index: int
    ) -> float | None:
        col = "bb_lower" if self.direction is Direction.LONG else "bb_upper"
        if col not in df.columns:
            return None
        return float(df[col].iat[bar_index])

    def _is_more_favorable(self, candidate: float, current: float) -> bool:
        if self.direction is Direction.LONG:
            return candidate > current
        return candidate < current
