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
        # stage 2 stop = entry × (1 + 2×0.0005 + 0.0003) + 0.2 × R
        #             = 100 × 1.0013 + 1.0 = 100.13 + 1.0 = 101.13
        assert result == pytest.approx(101.13)

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
        # stage 2 stop = 100 × (1 − 0.0013) − 1.0 = 99.87 − 1.0 = 98.87
        assert result == pytest.approx(98.87)

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
        # stage 2 fixed = 101.13；Bollinger lower = 103
        # 多單取較高（較有利）= 103
        assert result == pytest.approx(103.0)

    def test_long_uses_stage2_floor_when_bb_lower_below(self) -> None:
        ctrl = TrailingStopController(
            position=_long_position(entry=100.0, stop=95.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        # Bollinger lower = 90，低於 stage 2 fixed 101.13 → stop 仍取 stage2
        df = _make_df(5, bb_lower_const=90.0)
        bar = _bar(o=100.0, h=113.0, l=99.5, c=112.0)
        result = ctrl.update(bar, df, 0, current_stop=95.0)
        assert ctrl.stage == 3
        assert result == pytest.approx(101.13)

    def test_short_uses_bollinger_upper(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        # Bollinger upper = 97（低於 stage 2 fixed 98.87）→ 空單取較低 = 97
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
        # stage2_value = 101.13、bb_lower = 80 → max = 101.13
        # 但 current_stop=102 已比 101.13 高 → ratchet 拒絕
        assert result is None

    def test_short_does_not_increase_stop(self) -> None:
        ctrl = TrailingStopController(
            position=_short_position(entry=100.0, stop=105.0),
            params=TrailingStopParams(),
            broker_config=_broker_cfg(),
        )
        df = _make_df(5, bb_upper_const=120.0)
        bar = _bar(o=100.0, h=100.5, l=87.0, c=88.0)
        # current_stop=98 已比 stage2_value 98.87 低（更有利）→ ratchet 拒絕
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
