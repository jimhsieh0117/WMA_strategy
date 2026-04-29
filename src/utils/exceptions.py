"""WMA Strategy 專用例外層級。

依 CLAUDE.md §5 嚴格 fail-fast 原則：所有不變式違反皆以自訂例外 raise，
禁用 assert（避免 ``python -O`` 時被優化掉）。
"""

from __future__ import annotations


class WMAStrategyError(Exception):
    """所有專案內例外的共同基底，方便上層一次捕捉。"""


class LookAheadError(WMAStrategyError):
    """指標、策略或撮合在計算 ``value[t]`` 時使用了 ``bar[>t]`` 的資料。

    依 CLAUDE.md §2.3 為絕對禁止行為。
    """


class DataIntegrityError(WMAStrategyError):
    """輸入資料品質不合規：欄位缺失、index 非 DatetimeIndex、OHLC 邏輯錯誤、
    時間戳非單調遞增、含 NaN 等。
    """


class ConfigError(WMAStrategyError):
    """設定檔欄位缺失、型別錯誤、數值超出合理範圍。"""


class AccountInvariantError(WMAStrategyError):
    """帳戶狀態違反不變式：cash 為負、equity 與 cash + position_value 不一致、
    持倉數量 / 方向異常等。
    """


class OrderExecutionError(WMAStrategyError):
    """訂單撮合違反不變式：成交價超出 K 線 [low, high] 範圍、
    限價單方向與當前持倉衝突等。
    """
