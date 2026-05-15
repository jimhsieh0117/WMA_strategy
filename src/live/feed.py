"""LiveCandleFeed：每 60 秒從 Binance USDT-M perpetual 抓最新 K 線。

只把「已收盤」的 K 線交給 callback；當前正在 forming 的最後一根 K 不會處理。
透過 ccxt.async_support 異步抓取，rate-limit 已內建。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import ccxt.async_support as ccxt_async
import pandas as pd

logger = logging.getLogger(__name__)

# ccxt timeframe → 毫秒
_TF_TO_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1H": 3_600_000,
    "4H": 14_400_000,
}


class LiveCandleFeed:
    """Binance USDT-M perpetual 即時 K 線 poll 迴圈。

    - 不需要 API key（公開端點）
    - poll_interval_sec 預設 60s（Binance fapi 公開限額 1200 req/min，遠超我們用量）
    - on_new_closed_bar callback 由 LiveSimulator 提供，每根新 closed K 觸發一次
    - on_warmup_bars callback 啟動時觸發一次，回傳 N 根歷史 K（含當下 forming 之前的）
    """

    def __init__(
        self,
        *,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        warmup_bars: int = 96,                    # 24h × 4 (15m K) = 一天
        poll_interval_sec: int = 60,
        on_new_closed_bar: Callable[[pd.Timestamp, float, float, float, float, float], Awaitable[None]] | None = None,
        on_warmup_loaded: Callable[[pd.DataFrame], Awaitable[None]] | None = None,
    ) -> None:
        if timeframe not in _TF_TO_MS:
            raise ValueError(f"timeframe '{timeframe}' not supported by LiveCandleFeed")
        self.symbol = symbol
        self.timeframe = timeframe
        self.warmup_bars = warmup_bars
        self.poll_interval_sec = poll_interval_sec
        self.on_new_closed_bar = on_new_closed_bar
        self.on_warmup_loaded = on_warmup_loaded
        self._exchange = ccxt_async.binanceusdm({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self._tf_ms = _TF_TO_MS[timeframe]
        self._last_closed_ts_ms: int | None = None
        self._stop_requested = False

    async def warmup(self) -> pd.DataFrame:
        """啟動時抓 ``warmup_bars`` 根歷史 K 線（已收盤的；不含當下 forming）。

        回傳一個 DataFrame with index=timestamp, columns=[open, high, low, close, volume]。
        同時觸發 on_warmup_loaded callback。
        """
        # 多抓 2 根 buffer，最後一根可能 forming
        fetch_limit = self.warmup_bars + 2
        logger.info(
            "Fetching %d %s %s warmup bars from Binance USDT-M ...",
            fetch_limit, self.symbol, self.timeframe,
        )
        ohlcv = await self._exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=fetch_limit,
        )
        now_ms = self._exchange.milliseconds()

        rows = []
        for ts_ms, o, h, l, c, v in ohlcv:
            if ts_ms + self._tf_ms > now_ms:
                # forming bar，跳過
                continue
            rows.append((ts_ms, o, h, l, c, v))

        # 取最後 warmup_bars 根
        rows = rows[-self.warmup_bars:]
        if len(rows) == 0:
            raise RuntimeError("warmup fetch returned no closed bars")

        df = pd.DataFrame(
            [(pd.Timestamp(ts, unit="ms", tz="UTC").tz_convert(None), o, h, l, c, v)
             for ts, o, h, l, c, v in rows],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).set_index("timestamp")

        self._last_closed_ts_ms = rows[-1][0]
        logger.info(
            "Warmup loaded: %d bars, %s ~ %s",
            len(df), df.index[0], df.index[-1],
        )
        if self.on_warmup_loaded is not None:
            await self.on_warmup_loaded(df)
        return df

    async def run(self) -> None:
        """Main poll loop。每 ``poll_interval_sec`` 拉一次最新 K 線，
        對「新收盤」的 K 觸發 on_new_closed_bar。
        """
        if self._last_closed_ts_ms is None:
            raise RuntimeError("call warmup() before run()")
        logger.info(
            "LiveCandleFeed started: poll every %ds for new %s %s closed bars",
            self.poll_interval_sec, self.symbol, self.timeframe,
        )
        try:
            while not self._stop_requested:
                try:
                    await self._poll_once()
                except Exception as e:
                    logger.warning("poll error: %s; will retry next cycle", e)
                # 對齊到下一個 poll_interval（用 sleep）
                try:
                    await asyncio.sleep(self.poll_interval_sec)
                except asyncio.CancelledError:
                    break
        finally:
            await self._exchange.close()
            logger.info("LiveCandleFeed stopped")

    async def _poll_once(self) -> None:
        """抓最近 5 根 K，把新收盤的（時間 > last_closed_ts）逐一交給 callback。"""
        ohlcv = await self._exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=5,
        )
        now_ms = self._exchange.milliseconds()
        for ts_ms, o, h, l, c, v in ohlcv:
            # 跳過 forming bar
            if ts_ms + self._tf_ms > now_ms:
                continue
            # 跳過已 process
            if self._last_closed_ts_ms is not None and ts_ms <= self._last_closed_ts_ms:
                continue
            bar_ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_convert(None)
            logger.info(
                "New closed bar: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.2f",
                bar_ts, o, h, l, c, v,
            )
            if self.on_new_closed_bar is not None:
                await self.on_new_closed_bar(bar_ts, o, h, l, c, v)
            self._last_closed_ts_ms = ts_ms

    def stop(self) -> None:
        self._stop_requested = True
