# CLAUDE.md — WMA Trend Strategy 專案行為準則

> 給 Claude 的工作守則。每次任務開始前讀過一遍。
> 與使用者互動一律繁體中文；技術名詞、變數、commit message 使用英文。
> 完整架構規劃見 `ARCHITECTURE.md`，策略邏輯見 `多頭趨勢策略_v2.md` / `空頭趨勢策略_v2.md`。

---

## 〇、專案性質

- 加密貨幣多/空趨勢策略**回測**專案；標的可由 `data.symbol` 切換（ETHUSDT / BTCUSDT / ...）
- WMA 交叉訊號 + 三階段移動止損（Stage 1 swing → Stage 2 保本 → Stage 3 r_ladder 跟蹤）
- 多空兩支策略**獨立帳戶回測**，最後合併權益曲線
- Python 3.14 + 自製 event-driven 回測引擎（不用 backtesting.py）
- 進場濾網：`signal_filter`（六根 K 實體比例）+ `chop_filter`（BBW/ATR/ADX 三條件 AND）
- 倉位 sizing：每筆風險 = `equity_now × risk_per_trade_pct`（預設 1%）
- 一律使用**原始 K 線**（Heikin-Ashi 已於 cffa883 完整棄用）
- 預留實盤 broker / 超參數優化 / WFA / Monte Carlo 接口（先不實作，但保留可替換性）

---

## 一、Code Review 心態（最重要）

本專案的程式碼**會被外部 LLM 獨立審查**（Codex、Gemini、另一個 Claude Code session）。
寫每一行 code 時，假設「會被一個比你更嚴格的 reviewer 看過」：

- 邏輯不偷懶（不能用「應該不會發生」搪塞）
- 邊界條件要明確處理：空 DataFrame、單根 K 線、暖機期不足、限價未成交、同根 K 同時觸發進場與止損、止損與進場同價
- 不變式要主動檢查並 raise，不靜默吞 bug
- 命名、註解、錯誤訊息要扎實，能讓另一個 reviewer 不問也看得懂

**這條凌駕其他所有條款**。如果你在偷懶，請停下來重寫。

---

## 二、絕對禁止

1. **動作邊界（檔案層級）**：
   - **本專案 (`WMA_strategy/`)**：可自由新增、修改檔案，以及執行測試（`pytest`、跑回測等）。
   - **其他兄弟專案** (`~/Documents/GitHub/` 下的 PPO_TradingModel、Alpha_team、Pine_Strategies 等)：**僅限讀取**。不得新增、修改、刪除任何檔案。**例外**：使用者在 prompt 中明確點名允許修改某個檔案。
   - **刪除檔案 / 刪除類指令**（本專案與其他專案皆然）：一律**不執行**。需要刪除時，把指令貼給使用者，由使用者自行確認後執行。涵蓋：`rm`、`rm -rf`、`git rm`、`git clean -f`、`git branch -D`、`shutil.rmtree`、`Path.unlink` 等。
2. **執行被 settings 阻擋的指令**：不繞過 `pip install` / `rm` / `git push` / `WebFetch` 等限制。需要時請使用者執行或先請求允許。
3. **Look-ahead bias**：指標 / 策略 / broker 任何位置都不得使用未來 K 線資料。詳見 `ARCHITECTURE.md §3.1`。
   - WMA / ATR / ADX / Bollinger / rank 等所有指標的 `value[t]` 只能由 `bar[0..t]` 推導
   - 出現 `df.shift(-1)` / `iloc[i+1:]` / `bfill()` → 必為 bug
4. **未經告知就跑長命令**：> 30 秒的命令必須先預告（預估時間、寫入路徑、影響範圍），等使用者點頭才執行。
5. **實盤交易程式碼**：不寫真實 API 下單邏輯。Broker 接口可以**預留**（讓未來 `BinanceLiveBroker` 能無痛接入），但不實作。

---

## 三、工作流程

### 3.1 動工前

| 任務型態 | 流程 |
|---------|------|
| 大型實作（新模組、跨檔案重構、引入新依賴） | 先講解計畫、列出影響範圍 → 等使用者點頭 → 動工 |
| 小修（修字、補註解、3 處內的局部修改、命名統一） | 直接做、簡述變更 |
| 產生新文件（README、CHANGELOG、額外 design doc） | 必須使用者**明確要求**才寫，不主動產出 |

### 3.2 進行中

- 多步驟實作用 `TaskCreate` 追蹤進度，完成一項就 `TaskUpdate`
- **發現策略文件 / 架構文件 / 既有程式碼有 bug 或不一致 → 停下來通報，不擅自改**
- Long-running 命令前用以下格式預告：
  ```
  即將執行：xxx
  預估耗時：~ N 秒/分鐘
  寫入路徑：results/xxx/
  影響範圍：不會動到 src/ 或其他資料夾
  ```
- 迴圈 > 100 次的批次工作（資料載入、resample、多週期掃描、WFA）加 `tqdm` 進度條
- 短迴圈（< 100 次）不要加，避免雜訊

### 3.3 結束時

- 若該 prompt 動到 `src/` 或其他**程式檔**（純 `.md` 文件變更不需要），用 `git commit` 一次
- **一個 prompt = 一次 commit**（可含多檔變更）
- Commit message 格式：英文、Conventional Commits 風格
  - `feat: add chop filter as entry gate`
  - `fix: correct ATR rolling window min_periods`
  - `test: add WMA boundary cases`
  - `refactor: extract Signal dataclass`
  - `chore: update gitignore`
- **不加 Co-Authored-By 署名**
- 若是重大架構改動 → 同時更新 `CHANGELOG.md`（一般小改不維護）

---

## 四、程式風格

| 項目 | 規範 |
|------|------|
| **Type hints** | 所有函式、方法、dataclass 必須完整標注，含回傳型別 |
| **Docstring** | 不強制；複雜函式可寫，格式自由 |
| **業務邏輯註解** | 進場條件、止損計算、訂單撮合、look-ahead 防護等**關鍵邏輯**寫繁中註解；其他不寫 |
| **命名語言** | 變數、函式、類別、檔案 → 英文；inline comment → 繁中 |
| **視覺化文字** | matplotlib 標題/標籤、log 訊息、print → 繁中；專有名詞（ATR、Sharpe、Drawdown、Profit Factor）保留英文 |
| **Line length** | 不限制 |
| **Commit message** | 英文 |

---

## 五、錯誤處理（嚴格 fail-fast）

- 缺資料、缺欄位、設定錯、不變式違反 → 立即 `raise`，**不要 try-except 後 return None / 給 default**
- 不變式檢查用**自訂 Exception**，不用 `assert`（`python -O` 會把 assert 優化掉，金融場景不安全）

建議的自訂例外（在 `src/utils/exceptions.py` 集中定義）：
```python
class WMAStrategyError(Exception): ...
class LookAheadError(WMAStrategyError): ...
class AccountInvariantError(WMAStrategyError): ...
class OrderExecutionError(WMAStrategyError): ...
class DataIntegrityError(WMAStrategyError): ...
class ConfigError(WMAStrategyError): ...
```

**必須做不變式檢查的位置**：
- `Account` 開倉/平倉後：cash >= 0、position 數量正確、equity = cash + position_value
- 限價單成交：成交價在 bar 的 [low, high] 範圍內
- 指標計算後：序列長度等於原序列、無 NaN 在不該有的位置
- 設定載入後：必填欄位存在、型別正確、數值在合理範圍

---

## 六、測試

- 框架：`pytest`
- **必測**：`indicators`（WMA / ATR / ADX / Bollinger / rank + look-ahead 截斷測試）、`broker`（成交/止損/手續費邊界）、`strategy` 訊號生成 + 各 filter gate、`resampler`
- **可選**：`reporting`、`metrics`、entry scripts（用真實回測結果肉眼驗證即可）
- 邊寫邊補測試，不強制 TDD；critical path 在 commit 前必須綠
- **不需要**驗證與 PPO_TradingModel 的撮合一致性

---

## 七、Logging

- 用 Python `logging` 模組，不用 print（除了 entry script 的最終摘要表）
- 預設 `INFO`，加 `--verbose` flag 開 `DEBUG`
- 回測初期傾向冗長：每筆交易進場/出場/止損更新都印
- 後期視噪音再回頭調靜
- Logger 命名用模組名（`logging.getLogger(__name__)`）

---

## 八、套件管理

- 單一 `requirements.txt`，不分 dev/prod
- 加套件流程：
  1. 列出新套件 + 用途 + 預估版本
  2. 等使用者確認
  3. 直接執行 `pip install ...`（使用者會在權限對話框確認）
- Python 3.14 相容性疑慮的套件先試裝，失敗再退路（自寫 / 換套件 / 降版本）

---

## 九、Git

- `.gitignore` 必含：
  ```
  .venv/
  results/
  __pycache__/
  *.pyc
  .pytest_cache/
  .DS_Store
  *.egg-info/
  ```
- **不 commit**：回測結果（`results/`）、中間產物、`.venv/`
- `CHANGELOG.md`：重大架構改動時手動更新，否則不維護
- `README.md`：專案接近完成才寫，不主動產出

---

## 十、預留接口（不實作但要保留可替換性）

寫核心模組時要意識到下列未來會被**替換 / 接入**，避免硬編碼：

| 未來組件 | 影響的設計 |
|---------|-----------|
| **實盤 Broker**（如 `BinanceLiveBroker`） | `BrokerSimulator` 與未來實盤 broker 應符合**同一個 protocol**，engine 不需要改 |
| **超參數優化器**（Optuna / grid search） | 策略參數一律從 config 注入，**禁止寫死**；建議策略類別接受 `params: dict` 參數 |
| **WFA**（`src/validation/wfa.py`） | engine 必須能被多次呼叫、給定 sub-DataFrame，不依賴全域狀態 |
| **Monte Carlo**（`src/validation/monte_carlo.py`） | `BacktestResult` 必須完整保留 `trades` 列表（含進出場時間、PnL、direction），讓 MC 能重排 |

**判斷標準**：實作完一個模組後，問自己「未來要把它替換成 X，要改幾個地方？」如果 > 1 處（呼叫端外加實作端），介面設計可能太緊。

---

## 十一、領域知識（避免低級錯誤）

- 指標一律**原始 K 線**（不再使用 Heikin-Ashi；HA 已於 cffa883 完整棄用）
- WMA / ATR / ADX / BBW：只用 ≤ 當前 K 線的資料（look-ahead 嚴禁）
- 止損**只能往有利方向移動**，逆向不更新
- 限價單未成交視為廢單，不延期到下下根 K 線
- 同根 K 線同時觸發進場與止損：先進場再判止損（單筆完整交易，含完整手續費）
- Binance 永續 USDT-M VIP 0：taker 0.05% / maker 0.02%；本專案以 **taker 0.05%** 為主（限價含滑點 → 即時吃單）
- 滑點：限價單偏移 0.03%（多單往上、空單往下，確保成交）
- 帳戶設定：多空各 1000 USDT 獨立帳戶；每筆風險 = `equity_now × 1%`（隨權益動態，由 `risk_per_trade_pct` 控制）
- 內部 `leverage_cap = 8.0`（對應 Binance 端逐倉 20x），允許 2–3 筆典型倉位並存
- `r_min_pct = 0.0022`：R/entry < 0.22% 直接拒絕進場（涵蓋 abnormal R 區域，trailing 端有 invariant 保護）

---

## 十二、初次接手 Checklist（每次 session 開始時自查）

1. ✅ 讀過 `ARCHITECTURE.md` 確認當前進度與下一個 milestone
2. ✅ `git status` 確認沒有未提交的變更被遺忘
3. ✅ 確認 `.venv` 還在，必要時 `source .venv/bin/activate`
4. ✅ 確認當前任務是否需要先講計畫
5. ✅ 確認是否會動到 `src/`（決定要不要 commit）

---

*最後更新：2026-05-13*
