# WMA Trend Strategy — 架構設計文件

> **狀態**：設計階段 v0.1（待審視）
> **作者協作**：Jim Hsieh × Claude
> **日期**：2026-04-29

---

## 一、設計目標與原則

### 1.1 業務目標
- 實作並回測兩支對稱的 HA + WMA 交叉趨勢策略（多/空各一）
- 兩支策略**分屬不同帳戶**，獨立回測後再**合併權益曲線**評估綜效
- 樣本內（IS）2023-01-01 ~ 2024-12-31 用於開發/調參，**樣本外（OOS）2025-01-01 之後**作驗證

### 1.2 工程原則
1. **低耦合 / 高內聚**：每個模組只負責一件事，模組間用明確的資料結構溝通，不互相讀對方內部狀態
2. **純函式優先**：indicators / metrics 全部寫成 pure function（同樣輸入 → 同樣輸出，無 side effect）
3. **設定資料分離**：所有可調參數進 YAML，程式碼裡不寫魔術數字
4. **介面穩定，實作可換**：策略邏輯 ↔ 撮合引擎 ↔ 資料來源 三者透過 protocol 解耦，將來換實盤交易所只改 broker 實作
5. **多空對稱**：抽出 `BaseTrendStrategy`，多空只是方向參數的不同
6. **唯讀外部資料**：直接讀 PPO_TradingModel 的 parquet，不複製、不修改

---

## 二、整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                        Entry Scripts                        │
│   run_long.py  │  run_short.py  │  run_combined.py          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   backtest.engine   │   ← 事件驅動主迴圈
                └─────────┬───────────┘
                          │
       ┌──────────────────┼──────────────────────┐
       ▼                  ▼                      ▼
┌──────────────┐  ┌────────────────┐    ┌────────────────┐
│  data        │  │  strategy      │    │  broker        │
│  ──────      │  │  ──────        │    │  ──────        │
│  loader      │  │  base          │    │  simulator     │
│  resampler   │  │  long          │    │  account       │
└──────┬───────┘  │  short         │    │  order         │
       │          └────────┬───────┘    └────────┬───────┘
       │                   │                     │
       │                   ▼                     │
       │          ┌────────────────┐             │
       └─────────▶│  indicators    │             │
                  │  ha / wma /atr │             │
                  └────────────────┘             │
                                                 ▼
                                       ┌────────────────┐
                                       │  metrics       │
                                       │  reporting     │
                                       └────────────────┘
```

**依賴方向（單向）**：`entry → engine → {data, strategy, broker} → indicators / metrics`
策略**不直接呼叫** broker；broker **不知道**策略邏輯；兩者透過 engine 中介。

---

## 三、模組規格

### 3.1 `src/indicators/`（純函式）

| 檔案 | 函式 | 輸入 | 輸出 |
|------|------|------|------|
| `heikin_ashi.py` | `compute_ha(df)` | OHLCV DataFrame | DataFrame，加上 `ha_open / ha_high / ha_low / ha_close` 欄 |
| `wma.py` | `wma(series, period)` | pd.Series, int | pd.Series |
| `atr.py` | `atr(df, period)` | OHLC DataFrame, int | pd.Series |

**設計重點**：
- 全部無狀態、無副作用
- HA 必須遞迴計算（前一根 HA_Open 影響後一根），所以一次性向量化計算整個序列
- ATR 用原始 K 線（**不是** HA），與策略文件一致

#### ⚠️ Look-ahead bias 防護（強制要求）

所有指標的 `value[t]` 只能由 `bar[0..t]` 的資料推導，**絕對不可使用 `bar[t+1..]`**。

| 指標 | 允許輸入 | 禁止 |
|------|----------|------|
| `HA_Close[t]` | `bar[t].(O,H,L,C)` | 任何 `bar[>t]` |
| `HA_Open[t]` | `HA_Open[t-1]`, `HA_Close[t-1]` | 任何 `bar[>=t]` |
| `WMA[t]` | `series[t-period+1 .. t]` | `series[>t]` |
| `ATR[t]` | `bar[t-period+1 .. t]` 的 TR | `bar[>t]` |

**實作守則**：
- 寫指標時若出現 `df.shift(-1)`、`iloc[i+1:]`、`rolling(...).shift(-x)`、`pd.Series.bfill()` → **必須是 bug**
- 用 `min_periods=period`，不用 `min_periods=1`，避免暖機階段有不完整數值誤導策略
- 單元測試必須包含「截斷後重算等於原序列前段」的驗證：
  ```python
  full = compute_ha(df)
  partial = compute_ha(df.iloc[:n])
  assert (full.iloc[:n] == partial).all().all()  # 同一根 t 的值不應因為未來有沒有資料而改變
  ```
- 違反 → `raise LookAheadError(...)`

### 3.2 `src/data/`

| 檔案 | 類別/函式 | 職責 |
|------|----------|------|
| `loader.py` | `load_ohlcv(parquet_path, start, end)` | 從 parquet 載 1m K 線，過濾日期區間 |
| `resampler.py` | `resample(df_1m, timeframe)` | 1m → 1m/3m/5m/15m/30m/1H/4H |

**Resample 規則**（pandas `resample`）：
- `open`: first
- `high`: max
- `low`: min
- `close`: last
- `volume`: sum
- 對齊到完整週期（不完整週期捨棄）

**支援週期**：`["1m", "3m", "5m", "15m", "30m", "1H", "4H"]`

**輸出格式約定**：所有下游模組都假設 DataFrame 有 `DatetimeIndex` + `[open, high, low, close, volume]` 欄位（小寫）

### 3.3 `src/strategy/`

#### `base.py` — `BaseTrendStrategy`
共用邏輯：
- 接收已含指標的 DataFrame（HA、HA_WMA2、HA_WMA4、ATR）
- 在每根 K 線收盤後產出**訊號事件**（不直接執行訂單）
- 持倉中時，每根 K 線收盤更新止損

```python
@dataclass
class Signal:
    type: Literal["ENTRY", "UPDATE_STOP", "EXIT"]
    direction: Literal["LONG", "SHORT"]
    bar_index: int            # 訊號在哪根 K 線收盤後產生
    limit_price: float | None  # ENTRY 用：下一根 K 開盤撮合
    stop_price: float | None
    reason: str               # debug / log
```

#### `long_strategy.py`、`short_strategy.py`
僅實作方向特定的判斷邏輯：
- Long: 黃金交叉 + HA_Close[-2,-3] < HA_Close[0]；止損 = `highest(N) - ATR×k`，只能上移
- Short: 死亡交叉 + HA_Close[-2,-3] > HA_Close[0]；止損 = `lowest(N) + ATR×k`，只能下移

**重點**：策略**只產出 Signal**，不知道帳戶餘額、不算手續費、不下實際訂單。

### 3.4 `src/broker/`

#### `account.py` — `Account`
- 持有：`equity`, `cash`, `position`（含進場價、數量、方向、當前止損）
- 提供：`update_equity(mark_price)`、`open_position(...)`、`close_position(...)`
- 紀錄：trade log（每筆完整交易）、equity history（每根 K 線的權益）

#### `order.py`
```python
@dataclass
class LimitOrder:
    direction: Literal["LONG", "SHORT"]
    limit_price: float
    quantity: float
    stop_price: float
    expire_after_bars: int = 1   # 1 根內未成交就作廢

@dataclass
class FillResult:
    filled: bool
    fill_price: float | None
    fee: float
```

#### `simulator.py` — `BrokerSimulator`
撮合與成本模擬：
- **限價單撮合**：在訊號發出後的下一根 K 線
  - Long limit: 若 `bar.low <= limit_price` → 成交於 `limit_price`
  - Short limit: 若 `bar.high >= limit_price` → 成交於 `limit_price`
  - 否則作廢
- **滑點**：限價價格已含滑點（由策略產出時加上）；額外可在 simulator 再加成交滑點（保留參數，預設 0）
- **手續費**：開 + 平 各扣一次
- **止損觸發**：每根 K 線盤中價檢查
  - Long: `bar.low <= stop_price` → 以 `stop_price` 平倉（保守假設無滑點負滑）
  - Short: `bar.high >= stop_price` → 同上
  - 同根 K 線同時觸發進場與止損的處理：先入場再判斷止損（順序明確）

**Binance 永續合約 USDT-M 手續費假設（VIP 0 一般會員）**：
- Maker: 0.02%（限價單未成交即掛單）
- Taker: 0.05%（限價單即時撮合或市價）
- 我們的「開盤限價 + 滑點」會立刻吃單 → 採用 **Taker 0.05%**
- 平倉用市價 → Taker 0.05%
- ⚠️ **待 Jim 確認**：是否要假設用 BNB 折抵（taker 0.045%）

### 3.5 `src/backtest/engine.py`

事件驅動主迴圈，把 strategy 與 broker 串起來：

```python
def run_backtest(
    df: pd.DataFrame,            # 含指標
    strategy: BaseTrendStrategy,
    broker: BrokerSimulator,
    account: Account,
) -> BacktestResult:
    for i in range(warmup, len(df)):
        bar = df.iloc[i]

        # 1) 撮合上根 K 收盤後產出的限價單（在這根 K 線開盤撮合）
        if pending_order:
            broker.try_fill_limit(pending_order, bar)
            pending_order = None

        # 2) 盤中止損檢查（用 high/low）
        if account.has_position():
            broker.check_stop(account, bar)

        # 3) 收盤後跑策略，產出訊號
        signals = strategy.on_bar_close(df, i, account.position)

        # 4) 處理訊號：UPDATE_STOP 立即生效；ENTRY 變成 pending_order
        for sig in signals:
            if sig.type == "UPDATE_STOP":
                account.update_stop(sig.stop_price)
            elif sig.type == "ENTRY":
                pending_order = build_limit_order(sig, account)

        # 5) 紀錄該根 K 線的權益
        account.snapshot_equity(bar.close, bar.timestamp)

    return BacktestResult(account.trades, account.equity_curve)
```

**為何不直接用 `backtesting.py`？**
- 我們需要**兩個獨立帳戶**，`backtesting.py` 是單帳戶模型
- 「Bar[1] 開盤限價」+「止損只能單向移動」這類細節，自寫迴圈最直白
- Python 3.14 對 `backtesting.py` wheel 相容性未驗證
- **代價**：要自己實作 metrics（已規劃在 `src/metrics/`）

### 3.6 `src/metrics/`

#### `calculator.py`
從 `equity_curve + trades` 計算：
- `total_return_pct`
- `annualized_return_pct`（依時間區間長度）
- `sharpe_ratio`（年化，用每根 K 的 return）
- `sortino_ratio`
- `max_drawdown_pct`
- `calmar_ratio`
- `win_rate`、`profit_factor`、`expectancy`
- `avg_holding_bars`、`max_consecutive_wins/losses`
- `total_trades`、`avg_trades_per_day`

#### `merger.py`
- `merge_equity_curves(curve_long, curve_short, init_capital_each)` → 合併曲線
- 處理時間軸對齊（兩邊 K 線時間戳一致就直接相加）
- 重新計算合併後的 Sharpe / MDD / 等指標

### 3.7 `src/reporting/`
- `plotter.py`：權益曲線圖、回撤圖、月度收益熱力圖
- `exporter.py`：metrics.json、trades.csv

---

## 四、設定檔規格 `configs/default.yaml`

```yaml
data:
  source_parquet: "/Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/data/processed/ETHUSDT_1m.parquet"
  symbol: "ETHUSDT"
  timeframe: "5m"            # 可選: 1m, 3m, 5m, 15m, 30m, 1H, 4H

period:
  in_sample:
    start: "2023-01-01"
    end:   "2024-12-31"
  out_of_sample:
    start: "2025-01-01"
    end:   null              # null = 用到資料最後

account:
  initial_capital: 500.0      # USDT，多空各 500
  position_size_pct: 0.60     # 每筆倉位佔當前權益 60%

fees:
  maker_rate: 0.0002          # 0.02%
  taker_rate: 0.0005          # 0.05%
  slippage_pct: 0.0003        # 0.03%（用於限價單偏移）

strategy:
  # 進場條件
  wma_fast: 2
  wma_slow: 4
  # 三階段拖曳止損（詳見 §10）
  trailing:
    swing_lookback:           4
    stage1_slippage_buffer:   0.0003
    stage2_normal_trigger_r:  1.2
    stage2_abnormal_trigger_r: 2.4
    stage2_buffer_r:          0.2
    stage3_normal_trigger_r:  2.4
    stage3_abnormal_trigger_r: 4.8
    bollinger_period:         20
    bollinger_num_std:        2.0

backtest:
  warmup_bars: 50             # 暖機根數（讓指標收斂）
  output_dir: "results"
```

**說明**：每個策略執行時可載 `default.yaml` + 命令列覆蓋（例如 `--timeframe 15m`）。

---

## 五、目錄結構

```
WMA_strategy/
├── .venv/                          # Python 3.14
├── ARCHITECTURE.md                 # 本文件
├── 多頭趨勢策略_v2.md
├── 空頭趨勢策略_v2.md
├── requirements.txt
│
├── configs/
│   ├── default.yaml
│   └── example_15m.yaml            # 覆蓋範例
│
├── src/
│   ├── __init__.py
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── heikin_ashi.py
│   │   ├── wma.py
│   │   └── atr.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── resampler.py
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseTrendStrategy + Signal
│   │   ├── long_strategy.py
│   │   └── short_strategy.py
│   │
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── account.py
│   │   ├── order.py
│   │   └── simulator.py
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── engine.py
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── calculator.py
│   │   └── merger.py
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── plotter.py
│   │   └── exporter.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py               # YAML 載入 + 驗證
│
├── scripts/                        # 執行入口（薄殼）
│   ├── run_long.py                 # 只跑多頭策略
│   ├── run_short.py                # 只跑空頭策略
│   └── run_combined.py             # 兩個都跑 + 合併
│
├── tests/                          # 單元測試
│   ├── test_indicators.py
│   ├── test_resampler.py
│   ├── test_broker.py
│   └── test_strategy_signals.py    # 用構造的小型 K 線序列驗證進場/出場條件
│
└── results/                        # 回測輸出（gitignore）
    ├── eth_long_5m_is/
    │   ├── metrics.json
    │   ├── trades.csv
    │   ├── equity_curve.png
    │   └── drawdown.png
    ├── eth_short_5m_is/
    └── eth_combined_5m_is/
```

---

## 六、模組間介面協議（避免耦合的關鍵）

| 介面 | 上游 | 下游 | 資料形態 |
|------|------|------|----------|
| OHLCV | `data.loader` | 全部 | `pd.DataFrame[DatetimeIndex, [open,high,low,close,volume]]` |
| Indicator-augmented DF | `indicators.*` | `strategy.*` | DataFrame 加上 `ha_*`, `ha_wma_fast`, `ha_wma_slow`, `atr` |
| Signal | `strategy.*` | `backtest.engine` | `Signal` dataclass |
| LimitOrder | `engine` | `broker.simulator` | `LimitOrder` dataclass |
| FillResult | `broker.simulator` | `engine` | `FillResult` dataclass |
| BacktestResult | `engine` | `metrics`, `reporting` | `{trades: list[Trade], equity_curve: pd.Series}` |

**關鍵原則**：dataclass 僅含資料、無業務邏輯；模組間不傳遞「物件參考」讓對方修改（除了明確設計成 in-place 的 Account）。

---

## 七、關鍵時序問題（避免未來實作時踩坑）

### 7.1 訊號 → 撮合的時序
```
t=0  Bar[0] 收盤 → 跑策略 → 產出 ENTRY signal（含 limit_price = Bar[1] 開盤前未知）
                  ⚠️ 限價價格如何計算？
```

**問題**：策略文件說 `limit = Bar[1].open + slippage`，但 Bar[0] 收盤時還不知道 Bar[1].open。

**解決方案**（兩種，待 Jim 拍板）：
- **方案 A（嚴格）**：訊號只標記「下根 K 線開盤掛單」，真正的 limit_price 在 engine 主迴圈進入 t=1 開盤時才計算 = `bar[1].open + slippage`。這對應實盤行為（券商 API 的 OPEN order）。
- **方案 B（簡化）**：用 Bar[0] 收盤價當作 Bar[1] 開盤的近似值。誤差很小（5m K 開盤通常等於前根收盤），但與文件不嚴格一致。

→ **建議用方案 A**，最貼近實盤。

### 7.2 同根 K 線同時觸發進場與止損
- 進場是當根開盤限價成交
- 同根的 high/low 可能跨過止損
- **約定**：先成交進場，再用該根 high/low 檢查止損 → 同根可能立即止損，視為一次「真實」交易（含完整手續費），符合實盤可能發生的情境

### 7.3 止損更新時機
- 文件規定「每根 K 線收盤後」重算
- engine 在「收盤後跑策略」步驟收 `UPDATE_STOP`，下根 K 線生效
- 不在盤中即時更新止損（與文件一致）

---

## 八、開發順序（建議里程碑）

| 階段 | 內容 | 驗收 |
|------|------|------|
| **M1** | 環境 + indicators + data | `python -m pytest tests/test_indicators.py tests/test_resampler.py` 全綠 |
| **M2** | strategy.long + strategy.short（純訊號，先不接 broker） | 用構造的小型 K 線序列，斷言訊號出現在預期位置 |
| **M3** | broker（account / simulator） | 單元測試覆蓋成交、止損、手續費計算 |
| **M4** | backtest.engine + metrics 整合 | `scripts/run_long.py` 能完整跑出 IS 結果 |
| **M5** | merger + reporting + run_combined | 兩支策略合併權益曲線可視化 |
| **M6** | OOS 驗證 + 多週期掃描（1m~4H） | 比較表 + 結論 |
| **M7** | Walk-Forward Analysis（§10.1） | 待 M1~M6 穩定後展開 |
| **M8** | Monte Carlo 驗證（§10.2） | 待 M7 完成後展開 |

---

## 九、待確認事項（Jim 請拍板）

1. **手續費**：Taker 0.05%（不用 BNB 折抵）vs 0.045%（用 BNB 折抵）
2. **限價價格計算時機**：方案 A（嚴格、用 Bar[1] 真實開盤）vs 方案 B（簡化、用 Bar[0] 收盤）→ 我推薦 A
3. **止損成交價假設**：止損觸發時用 `stop_price` 成交（無負滑），還是要再扣一個滑點 buffer？實盤 stop-market 單常會有負滑，但起步階段建議先用 0 簡化。
4. **同根 K 線進場 + 止損**：照 7.2 描述處理 OK 嗎？
5. **是否要單元測試 / pytest**：建議要（用構造的 K 線序列驗證策略訊號），但會增加開發量
6. **OOS 區間結尾**：用到資料最後（2026-03-14）？還是明確指定 2025-12-31？
7. **多空合併權益曲線的合併方式**：
   - (A) 等權加總：`combined = long_equity + short_equity`
   - (B) 兩個帳戶獨立累積，最後總和（與 A 同義，假設無重新分配資金）
   - (C) 動態再平衡：每月把總資金按 50/50 重分配回兩個帳戶
   → 我推薦 (B)，最貼近「兩個獨立帳戶」的實盤情境

---

## 十、三階段拖曳止損設計（M5+ 已實作）

> 取代舊版單一 Chandelier Exit。對應 `src/strategy/trailing.py`。

### 11.1 設計目標

舊版止損（Chandelier `highest(N) − ATR×k`）在 5m / 15m 上太緊：
1. 進場後價格小幅震盪即被掃出
2. 鎖利門檻不存在 → 已獲利的部位常吐回到負值
3. 沒有區分「初始保護」與「鎖利後跟蹤」

新版分三階段，每階段適用不同價格區間：
- Stage 1：剛進場、方向未明，用 swing low/high 給寬鬆 buffer 避免誤觸
- Stage 2：價格朝對的方向走到 1.2R，鎖保本 + 成本，確保不會獲利後吐成負值
- Stage 3：價格走更遠（2.4R+），用 Bollinger Band 跟蹤趨勢

### 11.2 三階段定義

| 階段 | 觸發 | Stop 計算 |
|------|------|-----------|
| **Stage 1** | 持倉開始 | 多：`min(low over [t-N+1..t]) × (1 − slip_buffer)`<br>空：`max(high) × (1 + slip_buffer)`<br>N = `swing_lookback`（預設 4）|
| **Stage 2** | progress ≥ stage2_trigger_r（normal 1.2R / abnormal 2.4R）| 多：`entry × (1 + 2×taker + slippage) + buffer_r × R`<br>空：鏡像<br>buffer_r = 0.2 |
| **Stage 3** | progress ≥ stage3_trigger_r（normal 2.4R / abnormal 4.8R）| 多 stop = `max(stage2_stop, BB_lower)`<br>空 stop = `min(stage2_stop, BB_upper)`<br>BB = `WMA(20) ± 2σ` on 原始 close |

`progress = (bar.high − entry) / R`（多）/`(entry − bar.low) / R`（空）

`R = |entry_price − initial_stop|`（即 Stage 1 的 stop 與 entry 的距離）

### 11.3 異常 R 處理（R < taker×2 + slippage）

若風險距離小到光交易成本就吃掉，直接用 1.2R 拉保本會導致：
- 1.2R 的價格漲幅 < 成本距離 → 保本 stop 高於當前 mark → 立即被掃

解法：將兩個 trigger 全部 ×2（1.2 → 2.4，2.4 → 4.8）。等價於要求更明顯的 momentum 才啟動鎖利。

### 11.4 ratchet 規則

每根 K 線收盤後 controller 計算候選 stop。**只有「對倉位更有利」才更新**：
- 多單：candidate > current_stop 才 ratchet
- 空單：candidate < current_stop 才 ratchet
- 不利方向（Bollinger 回勾、價格震盪）→ 維持原值

### 11.5 介面

```python
# 在 fill 後 instantiate（一筆持倉一個 controller）
controller = TrailingStopController(
    position=account.position,
    params=strategy.params.trailing,   # TrailingStopParams
    broker_config=broker.config,        # 用 fee/slippage 計算成本
)

# 每根 bar 收盤呼叫
new_stop = controller.update(bar, df, bar_index, current_stop=position.stop_price)
if new_stop is not None:
    account.update_stop(new_stop)

# 持倉結束（止損觸發或 force_close）→ controller 廢棄
```

### 11.6 與其他模組的關係

```
strategy.detect_entry(bar t close)
    ↓ EntrySignal(initial_stop = swing-based)
engine: bar t+1 open fill
    ↓ instantiate TrailingStopController
controller.update(bar) per bar close
    ↓ stage transitions: 1 → 2 → 3
    ↓ ratchet candidate
account.update_stop(new_stop)
broker.check_stop(account, bar)  # 用最新 stop 撮合
```

---

## 十一、後續驗證模組（占位，待核心完成後實作）

> 待 M1~M5（核心策略 + 回測引擎 + 合併權益）完成、單一回測流程穩定後再展開。
> 此處先預留模組位置與輸入/輸出介面，避免日後加入時要回頭重構。

### 10.1 Walk-Forward Analysis（WFA）

**目的**：驗證策略在「滾動樣本內調參 → 滾動樣本外驗證」下是否仍有 edge，避免單一 IS/OOS 切分的過擬合。

**規劃位置**：`src/validation/wfa.py`

**初步介面草案**（細節待後續討論）：
```python
def run_wfa(
    df: pd.DataFrame,
    strategy_cls: type[BaseTrendStrategy],
    param_grid: dict[str, list],
    is_window_days: int,
    oos_window_days: int,
    step_days: int,
    metric: str = "sharpe_ratio",
) -> WFAReport:
    """每段 IS optimize → OOS validate，滾動切分整段時間。"""
```

**關鍵指標**（待 spec 化）：
- OOS 通過率（盈利 fold 佔比）
- 整體 OOS Sharpe / Profit Factor
- IS vs OOS 績效衰減度
- 參數穩定性（最佳參數隨時間漂移程度）

### 10.2 Monte Carlo 回測驗證

**目的**：將「單一歷史路徑的回測結果」轉成統計分布，評估策略績效是否具備統計顯著性，量化 MDD 與最終報酬的尾部風險。

**規劃位置**：`src/validation/monte_carlo.py`

**初步介面草案**：
```python
def run_monte_carlo(
    trades: list[Trade],
    n_simulations: int = 10_000,
    method: Literal["shuffle_returns", "bootstrap_trades", "block_bootstrap"] = "shuffle_returns",
) -> MCReport:
    """對交易順序/收益序列做隨機重排，產生績效分布。"""
```

**關鍵輸出**：
- 最終報酬分布（mean / median / 5%-95% CI）
- MDD 分布（看實盤遇到極端 path 的破產風險）
- Sharpe 分布
- 「原始回測結果在分布中的百分位」（看歷史是運氣好還是策略強）

### 10.3 與核心模組的關係

```
backtest.engine → BacktestResult
                     │
                     ├─→ metrics（單次績效）
                     ├─→ reporting（圖表/json）
                     ├─→ validation.wfa（多次回測編排）
                     └─→ validation.monte_carlo（單次結果統計擴增）
```

兩個驗證模組都**呼叫現有 engine**，不應重複實作回測邏輯。

---

## 十二、Python 3.14 套件相容性備註

- `pandas`、`numpy`、`pyarrow`：應該 OK
- `matplotlib`：應該 OK
- `pyyaml`：OK
- ⚠️ 若未來想加 `backtesting.py` 對照驗證，可能需要降到 Python 3.12

`requirements.txt` 起手式（待 M1 驗證）：
```
pandas>=2.2
numpy>=1.26
pyarrow>=15.0
pyyaml>=6.0
matplotlib>=3.8
pytest>=8.0
```

---

**請審視，特別是第六/七/九節。確認後我會依 M1 → M6 順序實作。**
