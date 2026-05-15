"""即時 paper trading 模組。

不下實單；完全復用回測引擎的 strategy / trailing / broker，只把資料源從
parquet 換成 Binance USDT-M perpetual 即時 K 線（ccxt poll）。

主要組件：
- ``state.LiveState``      shared mutable state（df / trades / equity / paused）
- ``feed.LiveCandleFeed``  Binance 15m K 線 poll loop
- ``simulator.LiveSimulator`` 對每根新 closed K 跑 engine 一步
- ``server.build_app``     FastAPI + WebSocket push + 暫停 REST
"""
