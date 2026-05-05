# HANDOFF — WMA Strategy 專案接手文件

> 給下個 Claude session 的接手筆記。完整準則見 `CLAUDE.md`，架構規格見 `ARCHITECTURE.md`，策略邏輯見 `多頭趨勢策略_v2.md` / `空頭趨勢策略_v2.md`。
> **首次接手請先讀完 `CLAUDE.md` § 一～十二**（特別是 §一 code review 心態、§二 絕對禁止、§五 fail-fast）。
> 最後更新：2026-05-04

---

## 〇、TL;DR — 一分鐘看完

- 加密貨幣（ETHUSDT 為主）多/空趨勢回測專案，**已完成 M1～M5+，含互動式 viewer**
- 184 tests 全綠，工作樹乾淨（`git status` clean）在 `main`
- 最後一個動作：commit `645f4b2` — 把 viewer 的「持倉連線」與「止損軌道」改成 **per-trade LineSeries** + 手動階梯編碼，徹底解決 LWC v4.2 跨 trade 連線的視覺 bug
- **下一步等使用者**：使用者尚未驗證 commit `645f4b2` 是否在視覺上真的修好。不要主動再動 viewer，等回饋

---

## 一、專案位置

```
/Users/jim_hsieh/Documents/GitHub/WMA_strategy/
├── CLAUDE.md                    # 行為準則（最重要）
├── ARCHITECTURE.md              # 架構設計
├── 多頭趨勢策略_v2.md          # 策略文件（多）
├── 空頭趨勢策略_v2.md          # 策略文件（空）
├── HANDOFF.md                   # ← 你現在在讀的這份
├── configs/default.yaml         # 主設定檔
├── pyproject.toml / requirements.txt
├── src/                         # 實作
│   ├── data/        loader, resampler
│   ├── indicators/  ha, wma, atr, bollinger, wavetrend
│   ├── strategy/    base, long_strategy, short_strategy, trailing, types
│   ├── broker/      account, simulator, types
│   ├── backtest/    engine, types
│   ├── metrics/     calculator, merger
│   ├── reporting/   plotter, exporter   (註：列表上沒看到，請以 ls src/ 為準)
│   ├── viewer/      server, panels, indicators, templates/index.html
│   └── utils/       config, exceptions, types, validation
├── scripts/                     # entry points
│   ├── _runner.py               # 共用 runner
│   ├── run_long.py / run_short.py / run_combined.py
│   └── view_chart.py            # FastAPI + LWC viewer
├── tests/                       # 184 tests
└── results/                     # 回測產物（.gitignore）
```

外部資料來源（**唯讀**，不准修改 PPO_TradingModel）：
```
/Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/data/processed/ETHUSDT_1m.parquet
```

---

## 二、環境

- Python 3.14
- `.venv/` 在專案根目錄；用 `.venv/bin/python` / `.venv/bin/pytest` 直接呼叫，不需 source
- 套件管理：單一 `requirements.txt`，加套件流程見 `CLAUDE.md` §八

---

## 三、常用指令

### 回測
```bash
.venv/bin/python scripts/run_combined.py --sample is    # 多+空合併（最常用）
.venv/bin/python scripts/run_combined.py --sample oos
.venv/bin/python scripts/run_long.py  --sample is
.venv/bin/python scripts/run_short.py --sample is
```

選項：`--config configs/default.yaml`（預設）、`--sample is|oos`

### 圖表檢視器
```bash
.venv/bin/python scripts/view_chart.py --sample is
```
選項：`--sample is|oos`、`--port 8050`、`--host 127.0.0.1`、`--panels bollinger ha_wma volume wavetrend`、`--no-open`

啟動後自動開瀏覽器到 `http://127.0.0.1:8050/`。

### 測試
```bash
.venv/bin/pytest                              # 全部
.venv/bin/pytest tests/test_engine.py         # 單檔
.venv/bin/pytest -k "risk_mode"               # 關鍵字
```

---

## 四、目前進度（Milestones）

| M | 內容 | 狀態 |
|---|------|------|
| M1 | data + indicators（含 look-ahead guard） | ✅ |
| M2 | strategy 訊號層（多/空對稱，無 broker 耦合） | ✅ |
| M3 | broker 層（Account / Simulator / types） | ✅ |
| M4 | engine + metrics + entry scripts | ✅ |
| M5 | equity merger + reporting（plotter/exporter）+ run_combined | ✅ |
| M5+ | 三階段 trailing stop（Stage 1 swing → 2 breakeven → 3 Bollinger） | ✅ |
| M5+ | `entry_source` 切換 HA / raw K | ✅ |
| M5+ | 互動式 viewer（FastAPI + Lightweight Charts） | ✅ |
| M5+ | risk-based 倉位（每筆撞 stop 固定虧 1U） | ✅ |
| M5+ | viewer 持倉連線 + 止損階梯軌道（per-trade segments） | ✅（commit 645f4b2，**待使用者視覺驗證**） |
| 後續 | 超參數優化（Optuna）、WFA、Monte Carlo | 預留接口未實作 |

### 近期 commits（新→舊）
```
645f4b2 fix: per-trade LineSeries with manual step encoding for stop track
7a893a4 feat: viewer line-break fix + risk-based sizing mode
21ab488 feat: viewer holding lines + per-trade stop tracks
b96b27a feat: interactive backtest viewer (FastAPI + LWC) with panel architecture
1b12e00 fix: code review issues #1 and #3 from short-strategy review
65c34a8 feat: entry_source switch for HA vs raw K-line entry signals
db97943 feat: M5+ three-stage trailing stop with Bollinger band
0fa93fd feat: M5 equity merger + reporting (plotter/exporter) + run_combined
e97cbe4 feat: M4 backtest engine + metrics + entry scripts
5eb49ba feat: M3 broker layer (Account, simulator, types)
2cfeb6d test: M2 strategy unit tests (28 cases)
84d7bb8 feat: M2 strategy signal layer (long/short, no broker coupling)
89122bb feat: M1 indicators and data layer with look-ahead guards
20eb4c3 chore: initial project skeleton
```

---

## 五、最近三件事的關鍵設計（必讀）

### 5.1 三階段 Trailing Stop（`src/strategy/trailing.py`）

對應 `ARCHITECTURE.md §11`。`TrailingStopController` 在每筆持倉成交後 instantiate，每根 K 線收盤呼叫 `update()`：

| Stage | 觸發 | stop 公式 |
|------|------|---------|
| 1 | 進場後 | 進場 K 線「前 N 根」反向極值 ± `stage1_slippage_buffer` |
| 2 | 達 `stage2_trigger_r`（normal 1.2R / abnormal 2.4R） | 保本 + 雙向 taker fee + `stage2_buffer_r × R`；無滑點（已內含於 entry_price） |
| 3 | 達 `stage3_trigger_r`（normal 2.4R / abnormal 4.8R） | Bollinger lower（多）/ upper（空），floor 為 stage 2 fixed |

**異常 R 判定**：`R < taker×2`（**沒有再加 slippage**，因為 entry_price 已含滑點）。觸發時所有 trigger ×2。

**Ratchet 規則**：stop 永遠只能往有利方向移動，逆向回拉維持原值。

### 5.2 Risk-based 倉位（`src/backtest/engine.py:_compute_quantity`）

設定切換：`configs/default.yaml` 的 `account.sizing_mode = pct | risk`。

- **pct 模式**：`qty = equity × position_size_pct / limit_price`（舊邏輯）
- **risk 模式**：每筆撞 stop 固定虧損 `risk_per_trade_usdt`：

  ```
  qty = risk / [|limit − stop| + (limit + stop) × taker]
  ```

  滑點已隱含在 `entry_price`（engine: `limit = open × (1 ± slip)`），**不重複加進 cost_pct**。

  **過槓桿處理**：若 `qty × limit > equity` → 直接拒絕（回 `(0, None, False)`）。

  **Broker 重算**：`LimitOrder.target_risk_usdt` 非 None 時，broker 在 `fill_price` 確定後再重算一次 qty，確保「fill 滑動」後虧損仍精確。

  **驗證**：1 年 IS 長單實測 1013 筆，`avg_loss = -0.9957 USDT`（目標 -1.00），公式正確。

> ⚠️ **注意**：使用者在 Q2.2 表示「我選 b（cap notional at equity X%）」但實作為 (a) reject。`test_risk_mode_rejects_when_overleveraged` 鎖死了 (a) 行為。若使用者後續提出修正，需改為 cap notional 並更新該測試。

### 5.3 Viewer：per-trade LineSeries + 手動階梯編碼（`src/viewer/server.py`）

**問題**：LWC v4.2 的 `LineType.WithSteps` 不可靠 — whitespace + WithSteps 無法在 trade 之間斷開階梯，導致「等待期間的線會延續」。

**解法**（commit 645f4b2）：
- `_holding_segments` / `_stop_track_segments` 改回傳 `list[list[point]]`（每筆 trade 一個 segment）
- Frontend 為每個 segment 建立**獨立** `LineSeries` → 物理上不可能跨 trade
- 止損軌道用「**手動編碼階梯點**」：每次 stop 變化前 1 秒插入前一個 stop 值的延伸點，再用 Solid line 連起來，視覺上等於直角階梯。範例：
  ```
  history=[(t=100, 95), (t=150, 97)], exit=200
  → [(100,95), (149,95), (150,97), (199,97), (200,97)]
  ```
- 持倉線寬度 `lineWidth: 2 → 1`（使用者要求變細）

API payload schema：
```
trade_lines.holding_wins  : list[list[{time, value}]]   # 每筆 trade 一個 segment
trade_lines.holding_losses: list[list[{time, value}]]
trade_lines.long_stops    : list[list[{time, value}]]
trade_lines.short_stops   : list[list[{time, value}]]
```

**待驗證**：使用者尚未確認視覺上是否真的修好。

---

## 六、模組關鍵介面（避免你重新看程式碼推導）

### 6.1 `EngineConfig`（`src/backtest/types.py`）

```python
@dataclass(frozen=True)
class EngineConfig:
    sizing_mode: SizingMode = "pct"          # "pct" | "risk"
    position_size_pct: float = 0.6            # pct 模式生效
    risk_per_trade_usdt: float = 1.0          # risk 模式生效
    skip_signal_when_pending: bool = True
    force_close_at_end: bool = False
```

### 6.2 `Position` / `Trade`（`src/broker/types.py`）

- `Position`：mutable，`stop_history: list[tuple[Timestamp, float]]`（含初始值）
- `Trade`：frozen，`stop_history: tuple[tuple[Timestamp, float], ...]`（平倉時從 Position copy）

### 6.3 `LimitOrder`（`src/broker/types.py`）

`target_risk_usdt: float | None = None` — 非 None 時 broker 用風險公式重算 qty。

### 6.4 `Account`（`src/broker/account.py`）

- `open_position(...)` 會 seed `stop_history = [(entry_timestamp, stop_price)]`
- `update_stop(new_stop, *, timestamp=None)` — 帶 timestamp 時 append 到 `stop_history`
- `close_position(...)` 把 `tuple(pos.stop_history)` 凍進 Trade

### 6.5 Engine 主迴圈（`src/backtest/engine.py:run_backtest`）

每根 K 線順序：
1. 撮合 pending limit 單於 `bar.open`（`_compute_quantity` 依 sizing_mode 分支）
2. 盤中止損檢查（`broker.check_stop`）
3a. 收盤 ratchet：`trailing.update()` → `account.update_stop(new_stop, timestamp=ts)`
3b. 收盤偵測新訊號（`strategy.detect_entry`）
4. `account.snapshot_equity(bar.close, ts)`

---

## 七、絕對禁止（CLAUDE.md §二 摘要 — 還是要去讀原文）

1. **不准動兄弟專案**：`~/Documents/GitHub/` 下其他專案唯讀（PPO_TradingModel、Alpha_team、Pine_Strategies）
2. **不准執行被 settings 阻擋的指令**（pip install / rm / git push / WebFetch）
3. **Look-ahead bias**：絕對禁止 `df.shift(-1)` / `iloc[i+1:]` / `bfill()`
4. **未經告知不跑 > 30 秒命令**
5. **不寫真實 API 下單邏輯**（broker protocol 可預留）

---

## 八、工作流（CLAUDE.md §三 摘要）

- **大型實作**：先講計畫 → 等使用者點頭 → 動工
- **小修**：直接做，簡述變更
- **不主動寫文件**：README / CHANGELOG / 額外 design doc 必須使用者明確要求
- **發現策略文件 / 架構文件 / 既有程式碼有 bug 或不一致 → 停下來通報，不擅自改**
- **一個 prompt = 一次 commit**，message 英文 Conventional Commits 風格，**不加 Co-Authored-By 署名**
- 純 `.md` 變更不需要 commit

---

## 九、設定檔精要（`configs/default.yaml`）

```yaml
data:
  source_parquet: /Users/jim_hsieh/Documents/GitHub/PPO_TradingModel/data/processed/ETHUSDT_1m.parquet
  symbol: ETHUSDT
  timeframe: 15m

period:
  in_sample:     {start: 2023-01-01, end: 2023-02-01}
  out_of_sample: {start: 2025-01-01, end: null}

account:
  initial_capital: 500.0
  sizing_mode: pct                # pct | risk
  position_size_pct: 0.60         # pct 時生效
  risk_per_trade_usdt: 1.0        # risk 時生效

fees:
  taker_fee_rate: 0.0005
  maker_fee_rate: 0.0002
  slippage_pct:   0.0003

strategy:
  wma_fast: 2
  wma_slow: 4
  entry_source: raw               # ha | raw
  trailing:
    swing_lookback: 4
    stage1_slippage_buffer: 0.0003
    stage2_normal_trigger_r:   1.2
    stage2_abnormal_trigger_r: 2.4
    stage2_buffer_r: 0.2
    stage3_normal_trigger_r:   2.4
    stage3_abnormal_trigger_r: 4.8
    bollinger_period: 20
    bollinger_num_std: 2.0
```

⚠️ 預設 IS 期間只有 1 個月（2023-01 ~ 2023-02），這是調試用的小視窗，不是最終評估期間。完整 IS 應為 2023-01-01 ~ 2024-12-31（見 `ARCHITECTURE.md §1.1`）。

---

## 十、測試覆蓋

- 184 tests，全綠
- 必測（CLAUDE.md §六）：indicators / broker / strategy / resampler — 都有 ✅
- 重點測試檔案：
  - `tests/test_engine.py` — 含 `TestEngineRiskSizing`（risk 模式公式正確 + 過槓桿拒絕）
  - `tests/test_trailing.py` — 三階段狀態機
  - `tests/test_viewer_server.py` — per-trade segment 隔離 + 手動階梯編碼
  - `tests/test_metrics.py` — Expectancy 公式（勝率×平均盈利 − 虧損率×平均虧損）

---

## 十一、待辦 / 已知議題

1. **使用者尚未驗證 commit `645f4b2`** 的視覺修復。等回饋；不要主動再動 viewer。
2. **risk 模式過槓桿處理**：使用者選 (b) cap notional，實作為 (a) reject。若使用者提出來，需改實作 + 更新 `test_risk_mode_rejects_when_overleveraged`。
3. **預留未實作**（CLAUDE.md §十）：實盤 broker、Optuna、WFA、Monte Carlo。介面已設計成可替換，動工前先問使用者要不要做。
4. **預設 IS 期間太短**（1 個月）：這是調試值，使用者可能會在某個 milestone 改回 2023 全年或 2 年。

---

## 十二、首次接手 checklist（從 CLAUDE.md §十二 抄來，每次 session 開始自查）

1. ✅ 讀 `CLAUDE.md`（行為準則）、`ARCHITECTURE.md`（架構）、本 `HANDOFF.md`（最新狀態）
2. ✅ `git status` 確認沒有未提交變更被遺忘
3. ✅ 確認當前任務是否需要先講計畫
4. ✅ 確認是否會動到 `src/`（決定要不要 commit）
5. ✅ 不確定就問使用者，不要 silent 改

---

*與使用者一律繁體中文；技術名詞、變數、commit message 使用英文。*
