"""TrailingStopController（三階段止損狀態機）單元測試。

涵蓋：
1. 構造正確性（R 計算、abnormal_R 偵測、trigger 設定）
2. Stage 1：完全不更新（return None）
3. Stage 1 → 2 觸發 + Stage 2 stop 計算
4. Stage 2 → 3 觸發 + Bollinger 止損
5. Ratchet：止損只能往有利方向動
6. abnormal_R 模式：trigger 全部 ×2
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.broker.types import Bar, BrokerConfig, Position
from src.strategy.trailing import TrailingStopController
from src.strategy.types import TrailingStopParams
from src.utils.types import Direction


TS = pd.Timestamp("2024-01-01 00:00:00")


def _bar(o: float, h: float, l: float, c: float, ts: pd.Timestamp = TS) -> Bar:
    return Bar(timestamp=ts, open=o, high=h, low=l, close=c)


def _make_df(n: int, *, bb_lower_const: float = 90.0, bb_upper_const: float = 110.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "open": [100.0] * n, "high": [101.0] * n,
            "low": [99.0] * n, "close": [100.0] * n,
            "bb_middle": [100.0] * n,
            "bb_upper": [bb_upper_const] * n,
            "bb_lower": [bb_lower_const] * n,
        },
        index=idx,
    )


def _long_position(entry: float = 100.0, stop: float = 95.0) -> Position:
    return Position(
        direction=Direction.LONG, quantity=1.0,
        entry_price=entry, entry_timestamp=TS,
        stop_price=stop, entry_fee=0.0,
    )


def _short_position(entry: float = 100.0, stop: float = 105.0) -> Position:
    return Position(
        direction=Direction.SHORT, quantity=1.0,
        entry_price=entry, entry_timestamp=TS,
        stop_price=stop, entry_fee=0.0,
    )


def _broker_cfg(taker: float = 0.0005, slip: float = 0.0003) -> BrokerConfig:
    return BrokerConfig(taker_fee_rate=taker, slippage_pct=slip)


# --------------------------------------------------------------------------- #
# 構造
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_long_R_computed(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.R == pytest.approx(5.0)
        assert ctrl.direction is Direction.LONG
        assert ctrl.stage == 1

    def test_short_R_computed(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.R == pytest.approx(5.0)

    def test_abnormal_R_detected(self) -> None:
        # R = 0.05 < cost = 100 × (0.001 + 0.0003) = 0.13 → abnormal
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=99.95),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.is_abnormal_r is True
        assert ctrl.stage2_trigger == 2.4
        assert ctrl.stage3_trigger == 4.8

    def test_normal_R_uses_default_triggers(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.is_abnormal_r is False
        assert ctrl.stage2_trigger == 1.2
        assert ctrl.stage3_trigger == 2.4

    def test_zero_R_raises(self) -> None:
        with pytest.raises(ValueError, match="R"):
            TrailingStopController(
                position=Position(
                    direction=Direction.LONG, quantity=1.0,
                    entry_price=100.0, entry_timestamp=TS,
                    stop_price=100.0, entry_fee=0.0,
                ),
                params=TrailingStopParams(),
                broker_config=_broker_cfg(),
            )


# --------------------------------------------------------------------------- #
# Stage 1
# --------------------------------------------------------------------------- #

class TestStage1NoUpdate:
    def test_return_none_when_no_progress(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        # bar.high = 100.5（progress = (100.5-100)/5 = 0.1R）→ 不到 1.2R
        bar = _bar(o=100.0, h=100.5, l=99.5, c=100.2)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert result is None
        assert ctrl.stage == 1


# --------------------------------------------------------------------------- #
# Stage 1 → 2
# --------------------------------------------------------------------------- #

class TestStage2Transition:
    def test_long_triggers_at_1_2R(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(stage2_buffer_r=0.2),
            broker_config=_broker_cfg(taker=0.0005, slip=0.0003),
        )
        df = _make_df(5)
        # bar.high 達到 entry + 1.2R = 106
        bar = _bar(o=100.0, h=106.5, l=99.5, c=105.0)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 2
        # stage 2 stop = entry × (1 + 2×taker) + 0.2 × R
        #             = 100 × 1.0010 + 1.0 = 100.10 + 1.0 = 101.10
        # （滑點已隱含於 entry_price，不再重複計）
        assert result == pytest.approx(101.10)

    def test_short_triggers_at_1_2R(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(stage2_buffer_r=0.2),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        # bar.low 達到 entry − 1.2R = 94
        bar = _bar(o=100.0, h=100.5, l=93.5, c=95.0)
        result = ctrl.update(bar, df, 0, current_stop=105.0)
        assert ctrl.stage == 2
        # stage 2 stop = 100 × (1 − 0.0010) − 1.0 = 99.90 − 1.0 = 98.90
        assert result == pytest.approx(98.90)

    def test_no_transition_below_threshold(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        # progress = 1.0R < 1.2R
        bar = _bar(o=100.0, h=105.0, l=99.5, c=104.5)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 1
        assert result is None


# --------------------------------------------------------------------------- #
# Stage 2 → 3 + Bollinger
# --------------------------------------------------------------------------- #

class TestStage3Bollinger:
    def test_long_uses_bollinger_lower_when_higher_than_stage2(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        # 第一次 update：直接打到 2.4R（同時跨 Stage 1→2→3）
        df = _make_df(5, bb_lower_const=103.0)
        bar = _bar(o=100.0, h=113.0, l=99.5, c=112.0)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 3
        # stage 2 fixed = 101.10；Bollinger lower = 103
        # 多單取較高（較有利）= 103
        assert result == pytest.approx(103.0)

    def test_long_uses_stage2_floor_when_bb_lower_below(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        # Bollinger lower = 90，低於 stage 2 fixed 101.10 → stop 仍取 stage2
        df = _make_df(5, bb_lower_const=90.0)
        bar = _bar(o=100.0, h=113.0, l=99.5, c=112.0)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 3
        assert result == pytest.approx(101.10)

    def test_short_uses_bollinger_upper(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        # Bollinger upper = 97（低於 stage 2 fixed 98.90）→ 空單取較低 = 97
        df = _make_df(5, bb_upper_const=97.0)
        bar = _bar(o=100.0, h=100.5, l=87.0, c=88.0)  # progress = 2.6R
        result = ctrl.update(bar, df, 0, current_stop=105.0)
        assert ctrl.stage == 3
        assert result == pytest.approx(97.0)


# --------------------------------------------------------------------------- #
# Ratchet 規則
# --------------------------------------------------------------------------- #

class TestRatchet:
    def test_long_does_not_decrease_stop(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5, bb_lower_const=80.0)
        # 進 stage 3，但 Bollinger lower 低於目前 stop 102 → 不應更新
        bar = _bar(o=100.0, h=113.0, l=99.5, c=112.0)
        result = ctrl.update(bar, df, 0, current_stop=102.0)
        # stage2_value = 101.10、bb_lower = 80 → max = 101.10
        # 但 current_stop=102 已比 101.10 高 → ratchet 拒絕
        assert result is None

    def test_short_does_not_increase_stop(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5, bb_upper_const=120.0)
        bar = _bar(o=100.0, h=100.5, l=87.0, c=88.0)
        # current_stop=98 已比 stage2_value 98.90 低（更有利）→ ratchet 拒絕
        result = ctrl.update(bar, df, 0, current_stop=98.0)
        assert result is None


# --------------------------------------------------------------------------- #
# Abnormal R 模式
# --------------------------------------------------------------------------- #

class TestAbnormalR:
    def test_doubled_triggers_used(self) -> None:
        # 構造 abnormal R: entry=100, stop=99.95 → R=0.05
        # cost = 100 × (0.001 + 0.0003) = 0.13；R<cost → abnormal
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=99.95),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.is_abnormal_r
        df = _make_df(5)
        # 1.5R = 0.075 → 不夠（abnormal 要 2.4R = 0.12）
        bar = _bar(o=100.0, h=100.075, l=99.95, c=100.05)
        result = ctrl.update(bar, df, 0, current_stop=99.95)
        assert ctrl.stage == 1
        assert result is None

        # 2.5R = 0.125 → 觸發 stage 2
        bar2 = _bar(o=100.0, h=100.125, l=99.95, c=100.10)
        result2 = ctrl.update(bar2, df, 0, current_stop=99.95)
        assert ctrl.stage == 2
        assert result2 is not None


# --------------------------------------------------------------------------- #
# Stage transitions 紀錄
# --------------------------------------------------------------------------- #

class TestTransitionLog:
    def test_records_each_stage_change(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5, bb_lower_const=103.0)
        bar = _bar(o=100.0, h=113.0, l=99.5, c=112.0)
        ctrl.update(bar, df, 0, current_stop=95.0)
        trans = ctrl.transitions
        # 一根 bar 內可能跨兩級 → 應有 2 筆 transition
        assert len(trans) == 2
        assert trans[0].from_stage == 1 and trans[0].to_stage == 2
        assert trans[1].from_stage == 2 and trans[1].to_stage == 3


# --------------------------------------------------------------------------- #
# Stage 3 r_ladder 模式
# --------------------------------------------------------------------------- #

class TestRLadder:
    """r_ladder：peak 跨 (first + k·step)R 後鎖到 (first + k·step − offset)R。"""

    def _ladder_params(self) -> TrailingStopParams:
        return TrailingStopParams(stage3_mode="r_ladder")

    def test_long_first_rung_28r_locks_25r(self) -> None:
        # entry=100, stop=95 → R=5；2.8R=14 ⇒ high=114；stop 候選 = 100+2.5*5=112.5
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        bar = _bar(o=100.0, h=114.0, l=99.5, c=113.0)
        new_stop = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 3
        assert new_stop == pytest.approx(112.5)

    def test_long_higher_rung_38r_locks_35r(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        # peak=3.8R=19 → high=119；stop 候選 = 100 + 3.5*5 = 117.5
        bar = _bar(o=100.0, h=119.0, l=99.5, c=118.0)
        new_stop = ctrl.update(bar, df, 0, current_stop=95.0)
        assert new_stop == pytest.approx(117.5)

    def test_long_below_first_trigger_no_ladder(self) -> None:
        # peak=2.5R 還不到 2.8R → 沒有 ladder 候選；stop 應為 stage2 保本值
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        # 先強行進到 stage 3：給一根 high=114 觸發
        ctrl.update(_bar(100.0, 114.0, 99.5, 113.0), df, 0, current_stop=95.0)
        # 接著一根 high 退到 2.5R=112.5（雖低於 first_trigger，但 peak 已是 2.8R+ 不變）
        # 為驗證「peak 從未到 2.8R 的情況」，新建一個 controller
        ctrl2 = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        # 先把它推到 stage 3（trigger 預設 2.4R=12 → high=112.5 即可）
        ctrl2.update(_bar(100.0, 112.5, 99.5, 112.0), df, 0, current_stop=95.0)
        assert ctrl2.stage == 3
        # peak=2.5R 未達 first_trigger=2.8R → ladder 回 None，候選退回 stage2 保本
        new_stop = ctrl2.update(_bar(100.0, 112.5, 99.5, 112.0), df, 1,
                                current_stop=95.0)
        # stage2 保本 = 100*(1+2*0.0005) + 0.2*5 = 101.1
        assert new_stop == pytest.approx(101.1)

    def test_short_first_rung_28r_locks_25r(self) -> None:
        # entry=100, stop=105 → R=5；空單 2.8R 下移至 86，stop 候選 = 100 − 2.5*5 = 87.5
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        bar = _bar(o=100.0, h=100.5, l=86.0, c=87.0)
        new_stop = ctrl.update(bar, df, 0, current_stop=105.0)
        assert ctrl.stage == 3
        assert new_stop == pytest.approx(87.5)

    def test_ladder_ratchet_not_pulled_back(self) -> None:
        # 先到 3.8R 鎖 117.5；之後 peak 退回 3.0R，stop 不應回退
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5)
        ctrl.update(_bar(100.0, 119.0, 99.5, 118.0), df, 0, current_stop=95.0)
        # 下一根 high 只到 115（3.0R），peak_progress_r 仍是 3.8R
        new_stop = ctrl.update(_bar(115.0, 115.0, 113.0, 114.0), df, 1,
                               current_stop=117.5)
        # 候選仍為 117.5，與 current 相等→ ratchet 判定非「更有利」回 None
        assert new_stop is None

    def test_abnormal_r_uses_doubled_first_trigger(self) -> None:
        # 構造 abnormal R：R < 2*taker*entry。taker=0.05，entry=100 → 雙向成本=10
        # 設 R=8（abnormal），entry=100 stop=92。第一檔 5.6R=44.8 → high=144.8
        # stop = 100 + 5.0*8 = 140.0
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=92.0),
            params=self._ladder_params(),
            broker_config=_broker_cfg(taker=0.05),  # 雙向 10% > R=8 → abnormal
        )
        assert ctrl.is_abnormal_r is True
        df = _make_df(5)
        # 先把 stage 推到 3：abnormal stage3 trigger=4.8R → 4.8*8=38.4 → high≥138.4
        # 同時 5.6R=44.8 ≤ high，所以 ladder 也會觸發
        bar = _bar(o=100.0, h=144.8, l=99.5, c=140.0)
        new_stop = ctrl.update(bar, df, 0, current_stop=92.0)
        assert ctrl.stage == 3
        assert new_stop == pytest.approx(140.0)


# --------------------------------------------------------------------------- #
# effective_R override（r_cap 機制）
# --------------------------------------------------------------------------- #

class TestEffectiveROverride:
    """以小於實際 R 的 effective_R 重做 stage2/3/ladder 計算，stage1 stop 不受影響。"""

    def test_default_equals_actual(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        assert ctrl.effective_R == pytest.approx(ctrl.R) == pytest.approx(5.0)

    def test_reject_non_positive_or_larger_than_R(self) -> None:
        with pytest.raises(ValueError):
            TrailingStopController(
                position=_long_position(entry=100.0, stop=95.0),
                params=TrailingStopParams(),
                broker_config=_broker_cfg(),
                effective_r_override=0.0,
            )
        with pytest.raises(ValueError):
            TrailingStopController(
                position=_long_position(entry=100.0, stop=95.0),
                params=TrailingStopParams(),
                broker_config=_broker_cfg(),
                effective_r_override=10.0,  # > R=5
            )

    def test_long_stage2_triggers_earlier_with_capped_R(self) -> None:
        # 實際 R=10（entry=100、stop=90）。effective_R=2 表示 stage2 trigger 1.2R = 2.4
        # 即只要 high ≥ 102.4 就會跳 stage2（vs 預設要到 high≥112）。
        # 但 stop 放置仍用 actual_R：buffer = 0.2 * 10 = 2.0
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
            effective_r_override=2.0,
        )
        df = _make_df(3)
        bar = _bar(o=100.0, h=102.5, l=99.5, c=102.4)
        new_stop = ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 2
        # stage2 stop = 100*(1 + 2*0.0005) + 0.2 * actual_R(=10) = 100.1 + 2.0 = 102.1
        assert new_stop == pytest.approx(102.1)

    def test_short_stage2_uses_actual_R_for_buffer(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=110.0),  # 實際 R=10
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
            effective_r_override=2.0,
        )
        df = _make_df(3)
        # 空單 progress = (entry - low) / effective_R = (100 - 97.6) / 2 = 1.2 → stage2
        bar = _bar(o=100.0, h=100.5, l=97.6, c=98.0)
        new_stop = ctrl.update(bar, df, 0, current_stop=110.0)
        assert ctrl.stage == 2
        # stage2 stop（空）= 100*(1 - 2*0.0005) - 0.2 * actual_R(=10) = 99.9 - 2.0 = 97.9
        assert new_stop == pytest.approx(97.9)

    def test_r_ladder_trigger_uses_effective_R_but_stop_uses_actual_R(self) -> None:
        # 實際 R=10、effective_R=2。第一檔 trigger = 2.8 × effective_R = 5.6
        # → high ≥ 105.6 觸發。但 stop 放置 = entry + 2.5 × actual_R = 125.0
        params = TrailingStopParams(stage3_mode="r_ladder")
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),
            params=params,
            broker_config=_broker_cfg(),
            effective_r_override=2.0,
        )
        df = _make_df(3)
        bar = _bar(o=100.0, h=106.0, l=99.5, c=105.5)
        new_stop = ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 3
        # max(stage2_value=102.1, ladder=125.0) = 125.0
        assert new_stop == pytest.approx(125.0)

    def test_initial_stop_position_unchanged(self) -> None:
        # Stage 1 完全不更新 stop（return None），不論 effective_R 為何
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
            effective_r_override=2.0,
        )
        assert ctrl.initial_stop == pytest.approx(90.0)
        df = _make_df(3)
        # progress_r = 0.5 / 2 = 0.25，未達 stage2_trigger=1.2 → 仍 stage1
        bar = _bar(o=100.0, h=100.5, l=99.5, c=100.5)
        new_stop = ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 1
        assert new_stop is None


# --------------------------------------------------------------------------- #
# stage2_pct_trigger（OR 觸發）
# --------------------------------------------------------------------------- #

class TestStage2PctTrigger:
    """%-based OR 觸發：peak_pct ≥ stage2_pct_trigger 時即使 progress_r < 1.2 也進 stage 2。"""

    def test_default_disabled_legacy_behavior(self) -> None:
        # 預設 stage2_pct_trigger=0 → 與舊行為一致（必須 progress_r >= 1.2 才進 stage 2）
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),  # R=10
            params=TrailingStopParams(),  # stage2_pct_trigger=0
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        # peak_pct = 0.5%，progress_r = 0.5 < 1.2 → 不該進 stage 2
        bar = _bar(o=100.0, h=100.5, l=99.5, c=100.5)
        ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 1

    def test_pct_trigger_fires_when_pct_threshold_hit_first(self) -> None:
        # R=10 (entry=100, stop=90)；progress_r=0.5 還沒到 1.2，但 peak_pct=0.5% ≥ 0.3%
        params = TrailingStopParams(stage2_pct_trigger=0.003)
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),
            params=params,
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        bar = _bar(o=100.0, h=100.5, l=99.5, c=100.5)  # high=100.5, peak_pct=0.5%
        new_stop = ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 2
        # stage2 stop 仍走 R-based: entry*(1+2*0.0005) + 0.2*R = 100.1 + 2.0 = 102.1
        assert new_stop == pytest.approx(102.1)

    def test_short_pct_trigger(self) -> None:
        params = TrailingStopParams(stage2_pct_trigger=0.003)
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=110.0),  # R=10
            params=params,
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        # low=99.5, peak_pct = (100-99.5)/100 = 0.5% ≥ 0.3% → 進 stage 2
        bar = _bar(o=100.0, h=100.5, l=99.5, c=99.6)
        new_stop = ctrl.update(bar, df, 0, current_stop=110.0)
        assert ctrl.stage == 2
        # stage2 stop（空）= 100*(1 - 2*0.0005) - 0.2*10 = 99.9 - 2.0 = 97.9
        assert new_stop == pytest.approx(97.9)

    def test_pct_trigger_below_threshold_no_fire(self) -> None:
        # peak_pct = 0.2% < 0.3%，progress_r = 0.2 < 1.2 → 兩條件都不滿足
        params = TrailingStopParams(stage2_pct_trigger=0.003)
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=90.0),
            params=params,
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        bar = _bar(o=100.0, h=100.2, l=99.5, c=100.1)
        ctrl.update(bar, df, 0, current_stop=90.0)
        assert ctrl.stage == 1

    def test_r_path_still_works_when_pct_path_disabled(self) -> None:
        # stage2_pct_trigger=0；progress_r >= 1.2 仍照觸發
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),  # R=5
            params=TrailingStopParams(stage2_pct_trigger=0.0),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        # high=106 → progress_r = 6/5 = 1.2 → 觸發
        bar = _bar(o=100.0, h=106.0, l=99.5, c=105.0)
        ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 2


# --------------------------------------------------------------------------- #
# Early-exit cancel
# --------------------------------------------------------------------------- #

class TestEarlyExitCancel:
    """進場後 N 根 K 觀測，浮盈不足 → should_early_exit=True。"""

    def test_default_disabled(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),  # early_exit_enabled=False
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        ctrl.update(_bar(100.0, 100.5, 99.5, 100.0), df, 0, current_stop=95.0)
        ctrl.update(_bar(100.0, 100.0, 99.5, 99.7), df, 1, current_stop=95.0)
        assert ctrl.should_early_exit() is False

    def test_long_no_excursion_triggers_cancel(self) -> None:
        # entry=100, stop=95, observation=1 → bar[i+1] high<entry 觸發 cancel
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=1,
                early_exit_min_peak_r=0.0,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        # bar[i]：本根 K 略高（上一根 high=100.3 > entry，progress_r > 0），但這不是觀測點
        ctrl.update(_bar(100.0, 100.3, 99.5, 100.0), df, 0, current_stop=95.0)
        assert ctrl.bars_observed == 1
        assert ctrl.should_early_exit() is False  # 還沒到觀測點
        # bar[i+1]：high < entry → 該根 K 浮盈 = 0，按 progress_r=(99.9-100)/5=-0.02 < 0 → cancel
        ctrl.update(_bar(99.95, 99.9, 99.5, 99.6), df, 1, current_stop=95.0)
        assert ctrl.bars_observed == 2
        assert ctrl.should_early_exit() is True

    def test_long_positive_excursion_no_cancel(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=1,
                early_exit_min_peak_r=0.0,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        ctrl.update(_bar(100.0, 100.3, 99.5, 100.0), df, 0, current_stop=95.0)
        # bar[i+1]: high=101 > entry → per-bar progress_r = (101-100)/5 = 0.2 > 0 → no cancel
        ctrl.update(_bar(100.0, 101.0, 99.5, 100.5), df, 1, current_stop=95.0)
        assert ctrl.should_early_exit() is False

    def test_short_no_excursion_triggers_cancel(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=1,
                early_exit_min_peak_r=0.0,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        ctrl.update(_bar(100.0, 100.5, 99.7, 100.0), df, 0, current_stop=105.0)
        # bar[i+1]: low=100.05 > entry=100 → 空單浮盈 = (100-100.05)/5 = -0.01 < 0 → cancel
        ctrl.update(_bar(100.0, 100.5, 100.05, 100.4), df, 1, current_stop=105.0)
        assert ctrl.should_early_exit() is True

    def test_after_stage2_promoted_no_cancel(self) -> None:
        # 觀測期內就晉級 stage 2 → 不該 cancel（交給 trailing 處理）
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=1,
                early_exit_min_peak_r=0.0,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        ctrl.update(_bar(100.0, 100.3, 99.5, 100.0), df, 0, current_stop=95.0)
        # bar[i+1]: high=106.5 → progress_r = 1.3 → stage 2 觸發
        ctrl.update(_bar(100.0, 106.5, 99.5, 105.0), df, 1, current_stop=95.0)
        assert ctrl.stage == 2
        assert ctrl.should_early_exit() is False

    def test_min_peak_r_threshold(self) -> None:
        # threshold=0.1：bar[i+1] progress_r=0.05 < 0.1 → cancel
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),  # R=5
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=1,
                early_exit_min_peak_r=0.1,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(3)
        ctrl.update(_bar(100.0, 100.3, 99.5, 100.0), df, 0, current_stop=95.0)
        # bar[i+1] high=100.25 → progress_r = 0.25/5 = 0.05 < 0.1 → cancel
        ctrl.update(_bar(100.0, 100.25, 99.5, 100.1), df, 1, current_stop=95.0)
        assert ctrl.should_early_exit() is True

    def test_check_only_at_observation_point(self) -> None:
        # observation_bars=2 → 在 bars_observed==3 時才檢查
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(
                early_exit_enabled=True,
                early_exit_observation_bars=2,
                early_exit_min_peak_r=0.0,
            ),
            broker_config=_broker_cfg(),
        )
        df = _make_df(4)
        ctrl.update(_bar(100.0, 99.95, 99.5, 99.7), df, 0, current_stop=95.0)
        assert ctrl.should_early_exit() is False  # bars_observed=1
        ctrl.update(_bar(99.7, 99.6, 99.4, 99.5), df, 1, current_stop=95.0)
        assert ctrl.should_early_exit() is False  # bars_observed=2，但觀測點是 3
        ctrl.update(_bar(99.5, 99.4, 99.2, 99.3), df, 2, current_stop=95.0)
        assert ctrl.should_early_exit() is True   # bars_observed=3
