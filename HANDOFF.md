# HANDOFF — WMA Strategy 專案當前狀態快照

> 給下個 Claude session 的接手筆記。完整準則見 `CLAUDE.md`，架構規格見 `ARCHITECTURE.md`，當前實作快照見 `ARCHITECTURE.md §13`。
> 首次接手請先讀 `CLAUDE.md` 全文（§ 一 code review 心態、§ 二 絕對禁止、§ 五 fail-fast 最關鍵）。
> 最後更新：2026-05-13

---

## 〇、TL;DR — 一分鐘看完

- 加密貨幣多/空趨勢回測專案；標的可由 `data.symbol` 切換（已驗證 ETHUSDT / BTCUSDT）
- **Heikin-Ashi 已於 cffa883 完整棄用**，所有指標一律原始 K 線
- 228 tests 全綠，工作樹乾淨在 `main`，最後 commit `7d06568`
- 最近幾個重點 commit：
  - `7d06568` source_dir + symbol 推導 parquet 路徑（修 ETH/BTC mislabel bug）
  - `c3fb134` chop_filter（BBW/ATR/ADX）整合進 strategy 為進場 gate
  - `a14d4a2` abnormal R invariant check
  - `9a207b3` `risk_per_trade_pct` 動態 sizing（1% equity）
  - `cffa883` HA 完整棄用

---

## 一、專案結構（精簡版）

```
WMA_strategy/
├── CLAUDE.md                    # 行為準則（最重要）
├── ARCHITECTURE.md              # 架構設計 + §13 當前狀態快照
├── HANDOFF.md                   # ← 本檔
├── 多頭趨勢策略_v2.md / v2.2.md
├── 空頭趨勢策略_v2.md / v2.2.md
├── configs/default.yaml
├── docs/
│   ├── RESEARCH_REPORT.md       # 給開發者的詳細研究記錄
│   ├── REPORT_summary.md        # 給非開發者的精簡版
│   ├── EDGE_EXPLORATION.md      # 下一階段 edge 探索 roadmap
│   └── advanced_quant_research_roadmap.md
├── src/
│   ├── indicators/              # wma / atr / bollinger / adx / rank（無 heikin_ashi）
│   ├── data/                    # loader, resampler
│   ├── strategy/                # base, long, short, trailing, types
│   ├── broker/                  # account, simulator, types
│   ├── backtest/                # engine, types
│   ├── metrics/                 # calculator, merger
│   ├── reporting/               # plotter, summary
│   ├── validation/              # monte_carlo（WFA 未實作）
│   ├── viewer/                  # 互動式 Dash chart
│   └── utils/                   # config, exceptions, types, validation
├── scripts/                     # run_*, analyze_*, sweep_*, view_chart
└── tests/                       # 228 tests
```

---

## 二、當前策略狀態

### 2.1 進場流程

```
WMA fast 穿 slow + close 前 2/3 根確認
  → passes_signal_filter()      六根 K 實體比例 ≥ 0.60
  → passes_chop_filter()        BBW_rank≥40 AND ATR_rank≥40 AND ADX≥20
  → Stage 1 swing-based stop
  → entry_hour_blacklist 在 fill bar 階段檢查（UTC 0, 12 拒絕）
  → r_min_pct / leverage_cap 在 sizing 階段檢查
```

### 2.2 倉位 sizing

每筆風險 = `equity_now × risk_per_trade_pct (1%)`，多/空各自獨立 equity。
`r_min_pct = 0.0022` → R/entry < 0.22% 拒絕進場。
`leverage_cap = 8.0` → Σ open_notional ≤ 8 × equity。

### 2.3 出場（三階段 trailing）

- **Stage 1**：swing-based 初始 stop（`swing_lookback=4`、`slippage_buffer=0.03%`）
- **Stage 2**：1.2R 觸發保本（+0.2R buffer）
- **Stage 3**：2.0R 觸發進入；`r_ladder` 模式，first=2.5R / step=0.5R / offset=0.2R

---

## 三、最新性能基準（IS 2023-01-01 ~ 2024-12-31，ETHUSDT 15m）

| | Long | Short | Combined |
|---|---:|---:|---:|
| Final | 974.6 | 609.7 | 1584.3 |
| Return | -2.5% | -39.0% | -20.8% |
| PF | 0.99 | 0.79 | 0.90 |
| Win% | 46.1% | 45.1% | 45.6% |
| Sharpe | — | -1.12 | -0.74 |
| MDD | — | 45.4% | 30.1% |
| Trades | 425 | 452 | 877 |

**觀察**：Long 接近 breakeven，Short 嚴重虧損（2023–2024 ETH 偏多趨勢）。chop_filter 整體把 PF 從 ~0.77 拉到 0.90，但**還沒過 1.0**。

主要瓶頸：stage 3 capture 62%（peak 3.75R vs realized 2.34R），1.4R giveback 是 PF 卡關主因。

---

## 四、下一階段優先級

詳見 `docs/EDGE_EXPLORATION.md`。簡述：

**Tier 1（最高 ROI）**
1. **Exit engineering**：trailing offline simulation，找最佳 k×ATR / chandelier / 動態 ladder offset
2. **Volatility-adaptive trailing**：依 ATR_pct / ADX 動態調 k
3. **Session × trailing**：UTC 黃金時段（15）放寬 trailing，弱勢時段收緊

**Tier 2**：vol regime state machine、動態 sizing、BTC→ETH lead-lag
**不做**：更多 indicator、複雜 ML、Order Block / FVG / orderbook

---

## 五、Session checklist

1. ✅ `git status` 確認乾淨
2. ✅ `source .venv/bin/activate`
3. ✅ `python -m pytest -x -q` 確認 228 tests 全綠
4. ✅ 讀 `ARCHITECTURE.md §13` 確認當前實作狀態
5. ✅ 確認當前任務是否需要先講計畫（大型實作要先講）
