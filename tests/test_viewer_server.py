"""viewer.server 的序列化邏輯（持倉連線 / 止損軌道）測試。

新 schema：每筆交易一個 segment（list of points），不再用 single-series +
whitespace 的方式。Frontend 為每個 segment 建立獨立 LineSeries。
"""

from __future__ import annotations

import pandas as pd

from src.broker.types import Trade
from src.utils.types import Direction
from src.viewer.server import _holding_segments, _stop_track_segments


def _trade(
    *,
    entry_t: int,
    exit_t: int,
    entry_price: float = 100.0,
    exit_price: float = 100.0,
    net_pnl: float = 0.0,
    direction: Direction = Direction.LONG,
    stop_history: tuple[tuple[pd.Timestamp, float], ...] = (),
) -> Trade:
    return Trade(
        direction=direction,
        quantity=1.0,
        entry_price=entry_price,
        entry_timestamp=pd.Timestamp(entry_t, unit="s"),
        exit_price=exit_price,
        exit_timestamp=pd.Timestamp(exit_t, unit="s"),
        entry_fee=0.0,
        exit_fee=0.0,
        gross_pnl=net_pnl,
        net_pnl=net_pnl,
        return_pct=net_pnl / entry_price,
        exit_reason="MANUAL",
        stop_history=stop_history,
    )


# --------------------------------------------------------------------------- #
# 持倉連線：每筆交易一個 segment
# --------------------------------------------------------------------------- #

class TestHoldingSegments:
    def test_each_trade_is_isolated_segment(self) -> None:
        t1 = _trade(entry_t=100, exit_t=200, entry_price=10, exit_price=20, net_pnl=10)
        t2 = _trade(entry_t=1000, exit_t=1100, entry_price=11, exit_price=21, net_pnl=10)
        out = _holding_segments([t1, t2], win=True)

        # 兩個獨立 segments，每個 segment 兩個點
        assert len(out) == 2
        assert out[0] == [
            {"time": 100, "value": 10.0},
            {"time": 200, "value": 20.0},
        ]
        assert out[1] == [
            {"time": 1000, "value": 11.0},
            {"time": 1100, "value": 21.0},
        ]

    def test_filter_by_win(self) -> None:
        win = _trade(entry_t=100, exit_t=200, net_pnl=5.0)
        loss = _trade(entry_t=300, exit_t=400, net_pnl=-3.0)
        wins_out = _holding_segments([win, loss], win=True)
        losses_out = _holding_segments([win, loss], win=False)
        assert len(wins_out) == 1
        assert len(losses_out) == 1

    def test_skip_degenerate_zero_duration(self) -> None:
        # exit == entry → skip
        t = _trade(entry_t=100, exit_t=100, net_pnl=1.0)
        assert _holding_segments([t], win=True) == []

    def test_empty_input(self) -> None:
        assert _holding_segments([], win=True) == []


# --------------------------------------------------------------------------- #
# 止損軌道：手動編碼階梯點
# --------------------------------------------------------------------------- #

class TestStopTrackSegments:
    def test_manual_step_encoding(self) -> None:
        sh = (
            (pd.Timestamp(100, unit="s"), 95.0),
            (pd.Timestamp(150, unit="s"), 97.0),
        )
        t = _trade(entry_t=100, exit_t=200, stop_history=sh)
        out = _stop_track_segments([t])

        assert len(out) == 1  # 一個 segment
        seg = out[0]
        # 預期：(100,95), (149,95) 延伸, (150,97), (199,97) 延伸, (200,97) 出場延伸
        assert seg == [
            {"time": 100, "value": 95.0},
            {"time": 149, "value": 95.0},
            {"time": 150, "value": 97.0},
            {"time": 199, "value": 97.0},
            {"time": 200, "value": 97.0},
        ]

    def test_each_trade_is_isolated_segment(self) -> None:
        sh1 = ((pd.Timestamp(100, unit="s"), 95.0),)
        sh2 = ((pd.Timestamp(1000, unit="s"), 105.0),)
        t1 = _trade(entry_t=100, exit_t=200, stop_history=sh1)
        t2 = _trade(entry_t=1000, exit_t=1100, stop_history=sh2)
        out = _stop_track_segments([t1, t2])

        # 兩個獨立 segments，第二個 segment 不會與第一個共用任何點
        assert len(out) == 2
        # 第一個 segment 結束於 (200, 95)
        assert out[0][-1] == {"time": 200, "value": 95.0}
        # 第二個 segment 從 (1000, 105) 開始
        assert out[1][0] == {"time": 1000, "value": 105.0}

    def test_no_history_skipped(self) -> None:
        t = _trade(entry_t=100, exit_t=200, stop_history=())
        assert _stop_track_segments([t]) == []

    def test_extends_to_exit(self) -> None:
        sh = ((pd.Timestamp(100, unit="s"), 95.0),)  # 只有初始 stop
        t = _trade(entry_t=100, exit_t=200, stop_history=sh)
        out = _stop_track_segments([t])
        seg = out[0]
        # 應有 (100,95) + 延伸點 (199,95) + 出場 (200,95)
        assert seg[0] == {"time": 100, "value": 95.0}
        assert seg[-1] == {"time": 200, "value": 95.0}
