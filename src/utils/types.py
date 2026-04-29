"""跨模組共用的型別定義。"""

from __future__ import annotations

from enum import StrEnum


class Direction(StrEnum):
    """交易方向。strategy / broker / engine 共用。"""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """LONG → +1，SHORT → -1。便於數值運算。"""
        return 1 if self is Direction.LONG else -1
