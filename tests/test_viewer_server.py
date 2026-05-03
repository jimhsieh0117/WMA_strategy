"""viewer.server 的序列化邏輯（持倉連線 / 止損軌道）測試。"""

from __future__ import annotations

import pandas as pd

from src.broker.types import Trade
from src.utils.types import Direction
from src.viewer.server import _holding_line_data, _stop_track_data


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
# 持倉連線（whitespace 應放在 last_exit + 1）
# --------------------------------------------------------------------------- #

class TestHoldingLineData:
    def test_two_trades_have_whitespace_right_after_exit(self) -> None:
        # trade1: 100~200; 等待; trade2: 1000~1100
        t1 = _trade(entry_t=100, exit_t=200, entry_price=10, exit_price=20, net_pnl=10)
        t2 = _trade(entry_t=1000, exit_t=1100, entry_price=11, exit_price=21, net_pnl=10)
        out = _holding_line_data([t1, t2], win=True)

        # 預期：[entry1, exit1, ws@201, entry2, exit2, ws@1101 (final)]
        assert out[0] == {"time": 100, "value": 10.0}
        assert out[1] == {"time": 200, "value": 20.0}
        assert out[2] == {"time": 201}                # whitespace 在 exit + 1
        assert out[3] == {"time": 1000, "value": 11.0}
        assert out[4] == {"time": 1100, "value": 21.0}
        # 結尾 whitespace 防止延伸到右邊
        assert out[5] == {"time": 1101}

    def test_filter_by_win(self) -> None:
        win = _trade(entry_t=100, exit_t=200, net_pnl=5.0)
        loss = _trade(entry_t=300, exit_t=400, net_pnl=-3.0)
        wins_out = _holding_line_data([win, loss], win=True)
        losses_out = _holding_line_data([win, loss], win=False)
        # win 那邊只有一筆 trade（兩個資料點 + 結尾 ws）
        assert sum(1 for p in wins_out if "value" in p) == 2
        assert sum(1 for p in losses_out if "value" in p) == 2

    def test_empty_input(self) -> None:
        assert _holding_line_data([], win=True) == []


# --------------------------------------------------------------------------- #
# 止損軌道
# --------------------------------------------------------------------------- #

class TestStopTrackData:
    def test_whitespace_after_exit_not_before_next_entry(self) -> None:
        sh = (
            (pd.Timestamp(100, unit="s"), 95.0),
            (pd.Timestamp(150, unit="s"), 97.0),
        )
        t1 = _trade(entry_t=100, exit_t=200, stop_history=sh)
        t2 = _trade(
            entry_t=1000, exit_t=1100,
            stop_history=((pd.Timestamp(1000, unit="s"), 105.0),),
        )
        out = _stop_track_data([t1, t2])

        # 找到第一個 whitespace 的位置 — 應緊貼 trade1 exit (=200)，而非 trade2 entry-1
        ws_times = [p["time"] for p in out if "value" not in p]
        assert ws_times[0] == 201, f"expected whitespace at 201, got {ws_times[0]}"
        # 結尾還會有一個 ws
        assert ws_times[-1] == 1101

    def test_stop_track_extends_to_exit_with_last_value(self) -> None:
        sh = (
            (pd.Timestamp(100, unit="s"), 95.0),
            (pd.Timestamp(150, unit="s"), 97.0),
        )
        t1 = _trade(entry_t=100, exit_t=200, stop_history=sh)
        out = _stop_track_data([t1])
        valid = [p for p in out if "value" in p]
        # 階梯點 + (exit_t, last_stop) 延伸點
        assert valid[0] == {"time": 100, "value": 95.0}
        assert valid[1] == {"time": 150, "value": 97.0}
        assert valid[2] == {"time": 200, "value": 97.0}  # 延伸到 exit

    def test_no_stop_history_skipped(self) -> None:
        t = _trade(entry_t=100, exit_t=200, stop_history=())
        out = _stop_track_data([t])
        assert out == []
