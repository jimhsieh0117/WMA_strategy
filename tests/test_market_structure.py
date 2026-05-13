"""Market Structure 指標單元測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.market_structure import compute_market_structure
from src.utils.exceptions import DataIntegrityError


def _make_ohlc(highs: list[float], lows: list[float],
               closes: list[float] | None = None) -> pd.DataFrame:
    """從 high/low 序列建測試 K 線；close 預設取 (high+low)/2。"""
    n = len(highs)
    assert len(lows) == n
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    # open 隨意夾在 low/high 之間
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": [1.0] * n},
        index=idx,
    )


class TestPivotDetection:
    def test_simple_peak_detected(self) -> None:
        # 中間 bar 4 是明顯高點：左右各 2 根都低於它
        highs = [10, 11, 12, 13, 20, 13, 12, 11, 10]
        lows = [h - 2 for h in highs]
        df = _make_ohlc(highs, lows)
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        # bar 4 應該被標為 swing_high
        assert out["ms_swing_high"].iloc[4] == 20.0
        # 其他位置應該都是 NaN（中間幾根做 pivot 偵測時不滿足條件）
        non_nan = out["ms_swing_high"].dropna()
        assert len(non_nan) == 1

    def test_simple_trough_detected(self) -> None:
        lows = [20, 18, 16, 14, 5, 14, 16, 18, 20]
        highs = [l + 2 for l in lows]
        df = _make_ohlc(highs, lows)
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        assert out["ms_swing_low"].iloc[4] == 5.0

    def test_tied_high_not_pivot(self) -> None:
        # 兩個並列高點：strict 條件下兩者都不算 pivot
        highs = [10, 11, 12, 15, 13, 15, 12, 11, 10]
        lows = [h - 2 for h in highs]
        df = _make_ohlc(highs, lows)
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        # bar 3 與 bar 5 並列 15，都不該被標
        assert pd.isna(out["ms_swing_high"].iloc[3])
        assert pd.isna(out["ms_swing_high"].iloc[5])


class TestBoSCHoCH:
    def test_bos_up_after_pivot_confirmed(self) -> None:
        # 先做出一個 PH=20，之後 close 突破
        # 結構：上升→pivot high→回拉→突破
        highs = [10, 11, 12, 20, 13, 12, 25, 24, 23]
        lows = [h - 2 for h in highs]
        # close 設成 high-0.5，確保 close[6]=24.5 > 20（突破）
        closes = [h - 0.5 for h in highs]
        df = _make_ohlc(highs, lows, closes)
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        # bar 3 (=PH=20) 在 bar 5 才確認；bar 6 突破
        assert out["ms_swing_high"].iloc[3] == 20.0
        # bar 6 應該標 bos_up（首次突破，trend 從 "" → bull，仍視為 bos_up
        # 因為沒有反向 trend 要翻轉，無論是首次或延續皆為 bos_up）
        assert out["ms_event"].iloc[6] == "bos_up"
        assert out["ms_trend"].iloc[6] == "bull"

    def test_choch_down_reverses_bull(self) -> None:
        # 先建立 bull trend（突破 PH），再向下突破 PL → CHoCH_down
        # bars: 0..2 build up, 3 = PH=20, 4..5 confirmation buffer,
        # 6 = breakout above 20 → bull,
        # 7 = local high then 8 = PL=5, 9..10 confirmation, 11 = breakdown
        highs = [10, 11, 12, 20, 13, 14, 25, 24, 8, 15, 15, 10, 9, 8]
        lows = [5, 6, 7, 18, 11, 12, 22, 22, 5, 12, 12, 2, 2, 2]
        # close: bar 6 = 24 突破 PH=20 → bull
        # bar 11 = 3 突破 PL=5 (PL 在 bar 8，於 bar 10 確認) → choch_down
        closes = [7, 8, 9, 19, 12, 13, 24, 23, 6, 13, 13, 3, 5, 5]
        df = _make_ohlc(highs, lows, closes)
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        assert out["ms_event"].iloc[6] == "bos_up"
        assert out["ms_trend"].iloc[6] == "bull"
        # bar 8 是 PL，bar 10 確認
        assert out["ms_swing_low"].iloc[8] == 5.0
        # bar 11 close=3 < PL=5 → choch_down
        assert out["ms_event"].iloc[11] == "choch_down"
        assert out["ms_trend"].iloc[11] == "bear"


class TestLookAheadSafety:
    def test_prefix_matches_full_when_safe(self) -> None:
        """驗證對序列前綴計算 == 對完整序列計算（截至前綴尾端）。

        要求：bar t 的 ms_event / ms_trend 不應依賴 bars > t。
        ms_swing_high/low 例外：pivot p 的右側確認需要 bars p+1..p+right，
        所以前綴只看到 bars ≤ t 時，bars > t-right 的 pivot 還沒確認。
        """
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="15min")
        close = 100 + np.cumsum(rng.standard_normal(n) * 1.0)
        high = close + np.abs(rng.standard_normal(n)) * 2
        low = close - np.abs(rng.standard_normal(n)) * 2
        df_full = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": [1.0] * n},
            index=idx,
        )

        left, right = 5, 5
        full = compute_market_structure(df_full, pivot_left=left, pivot_right=right)

        cutoff = 150
        prefix = compute_market_structure(
            df_full.iloc[:cutoff], pivot_left=left, pivot_right=right,
        )

        # 對 bar t ≤ cutoff - right - 1，事件與趨勢應完全一致
        # （pivot 已在 prefix 內確認，後續 bar 也不會改寫過去事件）
        safe_end = cutoff - right - 1
        for col in ("ms_event", "ms_trend"):
            assert (
                prefix[col].iloc[:safe_end].reset_index(drop=True)
                .equals(full[col].iloc[:safe_end].reset_index(drop=True))
            ), f"{col} 在安全範圍內 prefix 與 full 不一致"


class TestEdgeCases:
    def test_empty_df_returns_empty(self) -> None:
        df = _make_ohlc([], [])
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        for col in ("ms_swing_high", "ms_swing_low", "ms_event",
                    "ms_event_price", "ms_trend"):
            assert col in out.columns
            assert len(out[col]) == 0

    def test_short_df_no_pivots(self) -> None:
        # n=3, left=right=2 → 沒有任何 bar 能滿足 pivot
        df = _make_ohlc([10, 11, 12], [8, 9, 10])
        out = compute_market_structure(df, pivot_left=2, pivot_right=2)
        assert out["ms_swing_high"].isna().all()
        assert out["ms_swing_low"].isna().all()
        assert (out["ms_event"] == "").all()

    def test_invalid_pivot_param(self) -> None:
        df = _make_ohlc([10] * 10, [8] * 10)
        with pytest.raises(DataIntegrityError):
            compute_market_structure(df, pivot_left=0, pivot_right=2)
        with pytest.raises(DataIntegrityError):
            compute_market_structure(df, pivot_left=2, pivot_right=-1)
