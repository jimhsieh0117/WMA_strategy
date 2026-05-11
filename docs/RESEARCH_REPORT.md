# WMA Trend Strategy — 研究報告

> 最後更新：2026-05-11
> 對象：ETHUSDT 永續 USDT-M、15m / 5m timeframe
> 範圍：自進場端特徵探索 → 盤整過濾器 → Monte Carlo 驗證 → 結構性分析的完整歷程

---

## 0. TL;DR

策略目前狀態：**距離可上架仍有顯著差距**。最佳組合
（15m, WMA(4,6), chop filter BBW≥40 & ATR≥40 & ADX≥20）達到：

- IS：PF 0.96，return -0.84%，MDD 2.62%，expectancy -0.03R/筆
- OOS：PF 0.90，return -1.53%，MDD 2.93%，expectancy -0.07R/筆

MC bootstrap 95% CI 仍有 60–80% 機率收在 PF<1。**結構性問題在 stage 1 比例
過高與 trailing capture 不足，而非進場時機可被預測**。

---

## 1. 策略骨架

- 訊號：原始 K close + WMA cross trigger（多頭：fast 向上穿 slow；空頭反向）
- 進場：訊號 K 收盤後下一根 limit 進場（滑點 0.03%，未成交視為廢單）
- 止損：3 階段 trailing stop
  - Stage 1：initial swing stop
  - Stage 2：normal/abnormal trigger（R 或 BB squeeze 觸發），含 buffer R
  - Stage 3：r_ladder trail（每 +n R 提高 stop）
- 帳戶：多/空獨立 1000 USDT，每筆 60% equity，taker 0.05%

---

## 2. 研究問題

| 問題 | 動機 |
|------|------|
| Q1 進場 K 線上有沒有訊號能預測「這筆 trade 會跑到 stage 3」？ | 若能，可加 filter 篩出高品質訊號 |
| Q2 有沒有時段／市場狀態（震盪 / 趨勢）能避開虧損訊號？ | 砍掉爛環境 → 提升期望值 |
| Q3 trailing 設計是否充分擷取 stage 3 潛在收益？ | 看 peak R vs realized R 落差 |
| Q4 在統計上是否有 edge？目前虧損是運氣還是結構？ | MC 信賴區間判讀 |
| Q5 距離可上架還差多少？ | 量化下一步優化目標 |

---

## 3. 方法總覽

| 類別 | 腳本 | 問題 | 結論 |
|------|------|------|------|
| 進場端 univariate | `analyze_stage3_features.py` | v1：6 pre-entry + 10 post-entry | pre-entry max \|d\|=0.15，全部不達 0.2 門檻 |
| 進場端 univariate | `analyze_stage3_features_v2.py` | v2：ADX / RSI / MACD / BB / HTF 等 11 指標 | 全部 \|d\|<0.14；HTF aligned 還反向 |
| 進場端 univariate | `analyze_stage3_features_v3.py` | v3：WMA curvature / Wave Trend / MACD div 9 個 | 全部 \|d\|<0.07 |
| 進場端 multivariate | `analyze_stage3_multivariate.py` | 24 特徵 LR + GBM，target = stage3 | test AUC 0.50–0.52，GBM train 0.87–0.99 嚴重過擬合 |
| 進場端 target 重框 | `analyze_peak2r_multivariate.py` | target = peak_progress_r > 2R | test AUC 0.504，仍無 edge |
| WMA 參數 | `sweep_wma_long.py` | (5,8) / (6,10) / (8,13) 對比 | (4,6) 為甜蜜點 |
| Monte Carlo | `src/validation/monte_carlo.py` + `run_monte_carlo.py` | reshuffle + bootstrap | 確認 baseline 沒有運氣翻盤機會 |
| 盤整過濾 | `sweep_chop_filters.py` | ADX / CI / ATR-rank / BBW-rank 單條件 | BBW≥40 單一最佳，PF 0.768→0.90 |
| 盤整過濾 | `sweep_chop_filters_combo.py` | BBW×ATR×ADX 三條件 | 最佳 P(PF>1)=33.8% |
| 盤整 OOS | `validate_chop_filter_oos.py` | IS 最佳組合在 OOS 跑 | OOS PF 0.90，P(PF>1)=16% |
| 全績效 | `run_chop_filtered_report.py` | 重建 equity 看 Sharpe/DD/Calmar | MDD 縮 4–7 倍 |
| R 分布 | `analyze_R_and_stage_distribution.py` | peak R 分布、stage 計數、bucket | filter 為「均勻收縮」式改善 |
| 進場端 ×2 | `analyze_filtered_stage_features.py` | filter 後再做 stage1 vs stage3 比對 | r_over_atr 勉強 \|d\|=0.204 |
| 時段 | `analyze_hour_distribution.py` | UTC 逐小時 expectancy | UTC 15 為跨樣本黃金小時、UTC 13/14 為虧損黑洞 |
| 實際 R | `analyze_realized_R.py` | net_pnl / initial_risk | stage 3 capture 只有 62%（peak 3.75 → realized 2.34） |

---

## 4. 進場特徵探索

### 4.1 Univariate（Cohen's d）— 三輪共 36 個特徵

對 4 個 combo（5m/15m × WMA(2,4)/(4,6)）逐特徵比較 stage 3 vs stage 1
分布，閾值 |d|≥0.2 視為穩健 edge：

| 輪次 | 特徵類別 | 數量 | 最大 \|d\| |
|------|---------|------|-----------|
| v1 | r_pct / r_over_atr / volume_ratio / hour_of_day / 5 post-entry | 16 | 0.15（pre）|
| v2 | ADX / RSI / MACD hist / BB width / ATR rank / HTF aligned | 11 | 0.14 |
| v3 | WMA curvature / Wave Trend / MACD divergence / close_dist_wma | 9 | 0.07 |

**結論**：在 cross-trigger 訊號定義下，stage 3 與 stage 1 在進場時的特徵分布
幾乎無法分辨。

### 4.2 Multivariate ML

- 模型：`LogisticRegression`（StandardScaler + balanced）、`GradientBoostingClassifier`
  （n_estimators=200, depth=3, lr=0.05, subsample=0.8）
- 切分：train=2023、test=2024（2025+ 保留做真正 OOS）
- 結果：

| Target | LR test AUC | GBM test AUC | GBM train AUC | top-30% lift |
|--------|-------------|--------------|---------------|--------------|
| stage3 vs stage1 | 0.50–0.52 | 0.50–0.52 | 0.87–0.99（過擬合）| ~1.0× |
| peak > 2R | 0.50 | 0.50 | 0.87+ | 1.0× |

24 個特徵組合依然無法預測 stage 結果。**進場端 edge 路線實證上已枯竭**。

---

## 5. WMA 參數搜尋

`sweep_wma_long.py` 比較 (2,4) / (4,6) / (5,8) / (6,10) / (8,13) × 5m/15m
（IS 2 年）：

| TF | WMA | n | s3% | PnL | L_PF | S_PF | L_DD |
|----|-----|---|-----|-----|------|------|------|
| **15m** | **(4,6)** | 2596 | 19.84 | **-328** | **0.84** | **0.69** | **13.7%** |
| 15m | (5,8) | 3022 | 19.03 | -453 | 0.79 | 0.68 | 19.8% |
| 15m | (8,13) | 3634 | 18.88 | -524 | 0.81 | 0.67 | 21.8% |
| 15m | (6,10) | 4162 | 18.86 | -657 | 0.78 | 0.65 | 27.1% |
| 15m | (2,4) | 4224 | 17.99 | -895 | 0.67 | 0.58 | 41.9% |
| 5m | 全部 | 6.5k+ | ~17 | <-1400 | <0.7 | <0.6 | >70% |

**(15m, WMA(4,6)) 為局部甜蜜點**。拉長參數反而劣化（震盪市產生更多假突破
cross/cross-back）。

---

## 6. Monte Carlo 驗證框架

模組：`src/validation/monte_carlo.py`
入口：`scripts/run_monte_carlo.py`

### 6.1 Reshuffle（順序重排）

permute trade 順序，保留全部交易。final_equity / PF / Sharpe 為 permutation
不變量（聚合統計），只有 Max DD 受順序影響。

> baseline (15m, W(4,6), combined)：
> - Max DD 95% CI = [16.4%, 17.7%]
> - 你目前的 17.9% DD 比 95% 模擬還糟（運氣偏壞，但差距小）

### 6.2 Bootstrap（有放回抽樣）

從 trade list 有放回抽 N 筆，重抽 5000–10000 次，估算 PF / return / Sharpe
信賴區間。支援 `block_size` 做 block bootstrap 保留時間自相關。

> baseline (15m, W(4,6), combined)：
> - PF 95% CI = [0.699, 0.838]，**完全壓在 1.0 以下**
> - P(獲利收尾) = 0%
> - **不是運氣問題，是結構性無 edge**

![](../results/mc/15m_W4-6_both/bootstrap_hist.png)

---

## 7. 盤整過濾器系列

### 7.1 四種候選

| Filter | 邏輯 | 閾值候選 |
|--------|------|----------|
| ADX(14) | 趨勢強度，<X 視為無趨勢 | {15, 20, 25} |
| Choppiness Index(14) | ATR-based 0–100，>X 視為盤整 | {55, 60, 65} |
| ATR percentile rank(200) | 近 200 根 ATR 的百分位 | {20, 30, 40} |
| BB width rank(200) | BB 寬度近 200 根的百分位 | {20, 30, 40} |

### 7.2 單條件 sweep（`sweep_chop_filters.py`）

對 baseline trades 在 signal_ts 量測，post-hoc 過濾，MC bootstrap PF CI：

| Filter | 最佳閾值 | n_kept | PF | MC PF p95 | P(PF>1) |
|--------|---------|--------|-----|-----------|---------|
| baseline | — | 2596 | 0.768 | 0.838 | 0% |
| **BBW_rank** | <40 skip | 1318 (50.8%) | **0.901** | **1.018** | **7.88%** |
| ATR_rank | <40 skip | 1478 (56.9%) | 0.860 | 0.968 | 1.80% |
| ADX | <20 skip | 1647 (63.4%) | 0.810 | 0.904 | 0.04% |
| CI | 任一 | — | 0.755~0.768 | <0.85 | 0% |

**BBW squeeze 過濾是最強單條件**。CI 反直覺地沒用甚至略傷。

### 7.3 雙/三條件疊加（`sweep_chop_filters_combo.py`）

| 條件 | n_kept | PF | MC 95% CI | **P(PF>1)** |
|------|--------|-----|-----------|-------------|
| BBW≥40 (單) | 1318 | 0.901 | [0.79, 1.02] | 7.88% |
| BBW≥40 & ATR≥40 | 1088 | 0.929 | [0.81, 1.06] | 18.54% |
| **BBW≥40 & ATR≥40 & ADX≥20** | **874 (33.7%)** | **0.963** | **[0.82, 1.12]** | **33.82%** |
| BBW≥40 & ATR≥40 & ADX≥25 | 676 (26%) | 0.896 | [0.75, 1.06] | 14.92%（樣本太少反劣化） |

### 7.4 OOS 驗證（`validate_chop_filter_oos.py`）

最佳組合在 2025-01-01 ~ 2026-03-14 上：

| Sample | n | PF | MC P(PF>1) |
|--------|---|-----|------------|
| IS filtered | 874 (33.7%) | 0.963 | 33.82% |
| **OOS filtered** | **544 (33.9%)** | **0.895** | **16.36%** |

P(PF>1) 從 33.8% → 16.4%，但 OOS PF 0.895 仍明顯優於 OOS baseline 0.724。
**真實 edge 約貢獻 +0.17 PF，IS 過擬合貢獻 +0.07 PF**。

### 7.5 全績效報告（`run_chop_filtered_report.py`）

| Sample | account | Return | PF | MDD | Win% | n |
|--------|---------|--------|-----|-----|------|---|
| IS | combined baseline | -16.42% | 0.77 | 17.48% | 44.30% | 2596 |
| **IS** | **combined filtered** | **-0.84%** | **0.96** | **2.62%** | 46.91% | 874 |
| OOS | combined baseline | -12.29% | 0.72 | 12.54% | 43.90% | 1606 |
| **OOS** | **combined filtered** | **-1.53%** | **0.90** | **2.93%** | 46.51% | 544 |

MDD 從 17.5% → 2.6%（IS）/ 12.5% → 2.9%（OOS），**縮 4–7 倍**。
Return 從顯著虧損 → 接近打平。但 PF 仍 < 1。

![](../results/chop_filter_report/equity_compare.png)

---

## 8. 結構性分析

### 8.1 R 分布與 stage 削減（`analyze_R_and_stage_distribution.py`）

| Stage | IS base | IS filt | OOS base | OOS filt |
|-------|---------|---------|----------|----------|
| 1 占比 | 54.20% | 51.72% | 55.29% | 53.49% |
| 2 占比 | 25.96% | 26.09% | 26.59% | 27.57% |
| 3 占比 | 19.84% | **22.20%** | 18.12% | 18.93% |

filter 後 stage 3 占比僅輕微提升（IS +2.4pp、OOS +0.8pp）。三 stage 移除率
接近（62–68%），filter 屬「均勻收縮」式改善而非「精挑 stage 3」。

R 中位數從 base 1.000 → filt 1.121（+12%），分布輕度右移。

![](../results/r_stage_distribution/R_hist_IS.png)

### 8.2 時段（UTC 小時）分布（`analyze_hour_distribution.py`）

跨 IS / OOS 一致發現：

| Hour (UTC) | 對照時間 | 跨樣本表現 |
|-----------|---------|------------|
| **15** | 美股開盤（NY 11:00）| **雙樣本最強**：IS +0.37 / OOS +0.39 USDT/筆 |
| 18 | 美股下午 | IS +0.05 / OOS +0.36（OOS 強）|
| 23 | 美股收盤後 | 雙樣本正期望值 |
| **13** | 歐洲下午 / 美股盤前 | IS -0.24 / OOS -0.07（雙負）|
| **14** | 同上 | IS -0.12 / OOS -0.42（雙負）|
| **5** | 亞洲淡靜 | IS -0.15 / OOS -0.45（雙負）|

filter 主動避開亞洲深夜與歐洲早盤（移除率 78–85%），對美股時段相對寬鬆
（45–55%）。

![](../results/hour_distribution/hour_distribution.png)

### 8.3 Realized R（`analyze_realized_R.py`）

`realized_R = net_pnl / (|entry − initial_stop| × quantity)`

| Sample | Stage | n | realized_mean | peak_mean | 吐回 R | capture % |
|--------|-------|---|---------------|-----------|--------|-----------|
| IS base | 1 | 1407 | -1.19 | 0.41 | — | — |
| IS base | 2 | 674 | +0.17 | 1.62 | -1.45 | 10.5% |
| IS base | 3 | 515 | +2.21 | 3.56 | -1.36 | 62.0% |
| **IS filt** | 3 | 194 | **+2.34** | 3.75 | -1.40 | **62.6%** |
| OOS filt | 3 | 103 | +2.50 | 3.70 | -1.20 | 67.6% |

**Stage 3 trailing 只 capture 62% 的潛在收益**。peak 3.75R 平均吐回 1.4R
（37%）才出場。改善 trailing 設計是最高 ROI 的優化方向之一。

整體 expectancy（IS filtered）：

```
P(s1) × E(s1) + P(s2) × E(s2) + P(s3) × E(s3)
= 0.517 × (-1.15) + 0.261 × (+0.17) + 0.222 × (+2.34)
= -0.595 + 0.044 + 0.520
= -0.032 R/trade
```

→ Stage 3 賺 0.52R / 筆，幾乎完全被 stage 1 虧 0.60R / 筆抵消。
Stage 2 觸發後 trailing 吐到 0.17R，幾乎不貢獻。

### 8.4 Filter 後重做特徵分析（`analyze_filtered_stage_features.py`）

在 chop-filtered 子集（n=961）內，重做 stage 1 vs stage 3 univariate：

- 達 \|d\|≥0.2 門檻的特徵數：**1 / 24**（baseline：0 / 24）
- 唯一壓線：`r_over_atr` filt d=0.204（s3 mean 2.09 vs s1 mean 1.91）
- 反直覺：filter 後 ADX 在 stage 3 反而**低於** stage 1
- `atr_pct_rank` / `r_pct` 因 filter 條件本身就限縮其值域，被「自我消化」失去鑑別力

→ **即使在更乾淨的環境，stage 1 vs stage 3 依然資訊上無法區分**。

---

## 9. 上架門檻分析

### 9.1 業界共識門檻

| 指標 | 最低門檻 | 穩健門檻 | 當前 (W4-6 filtered) |
|------|---------|---------|---------------------|
| PF (IS + OOS) | > 1.20 | > 1.40 | IS 0.96 / OOS 0.90（未達） |
| MC bootstrap PF p05 | > 1.00 | > 1.10 | IS 0.82 / OOS 0.73（未達） |
| MC P(PF>1) | > 80% | > 95% | IS 34% / OOS 16%（未達） |
| Expectancy / trade (扣費後) | > +0.2R | > +0.3R | IS -0.03R（未達） |
| Annualized Sharpe | > 1.0 | > 1.5 | 無法可靠估算 |
| Calmar | > 0.5 | > 1.0 | IS -0.16 / OOS -0.44（未達） |
| OOS / IS metric ratio | > 0.70 | > 0.85 | PF 0.94（通過） |
| Max DD | < 25% | < 15% | IS 2.6% / OOS 2.9%（通過） |

### 9.2 缺口

要從目前 -0.03R 達到最低門檻 +0.2R，需 expectancy 提升 **+0.23R**。
換算成 PF 約 0.96 → 1.20（+25%）。

可能的提升路徑（已知改動潛力）：

| 改動 | 預估貢獻 | 備註 |
|------|---------|------|
| Trailing 收緊 stage 3 capture 62%→80% | +0.10R | realized 2.34 → 2.85 |
| 降低 stage 1 手續費侵蝕 -1.15→-1.05R（限價優化） | +0.05R | 滑點/費率敏感 |
| 縮短 entry_hour_blacklist（加 13/14/5） | +0.02R | 砍掉雙樣本確認的虧損時段 |
| Stage 2 trigger 動態化（讓更多 s2 進到 s3）| +0.05R | 待驗證 |
| **合計（樂觀）** | **+0.22R** | **勉強壓上架線** |

---

## 10. 結論與下一步

### 10.1 三個確定結論

1. **進場端 edge 路線已實證枯竭**。Univariate 三輪 36 特徵 + multivariate ML
   兩個 target × 4 個 combo，全部接近隨機（test AUC ≈ 0.50）。
2. **盤整 filter 有真實 edge 但不足**。BBW × ATR × ADX 三條件把 PF 從 0.77 →
   0.96，MDD 從 17.5% → 2.6%，OOS 驗證一致。但仍未跨過 PF 1.0。
3. **結構性瓶頸在 stage 3 trailing capture (62%) 與 stage 1 比例 (52%)**，
   不在進場時機可預測性。

### 10.2 建議下一步（按 ROI 排序）

1. **改善 stage 3 trailing capture**（最高 ROI）
   - 目前 r_ladder 設計太鬆，peak 吐回 1.4R 才止盈
   - 候選：parabolic SAR、ATR chandelier、更密的 R-ladder step
2. **Limit-only 進場降低 stage 1 滑點**
   - 目前 stage 1 實際 -1.15R vs 理論 -1.00R，多出 0.15R 全為費用
3. **疊加 hour blacklist（UTC 13, 14, 5）**
   - 雙樣本確認的虧損時段，成本低
4. **Dynamic R-sizing**
   - 高波動 / 強趨勢期放大倉位，低波動期縮小

### 10.3 已經試過但失敗的路徑（避免重複）

- 任何進場端 univariate filter（包括 r_pct / ADX / RSI / MACD / BB / WT / HTF aligned）
- 把 WMA 拉長到 (5,8) / (6,10) / (8,13)
- target 改框成 peak > 2R 仍想用 ML 預測

---

## 附錄 A：腳本索引

```
scripts/
├── _runner.py                              # 共用回測 runner
├── run_combined.py / run_long.py / run_short.py  # 標準回測 entry
├── run_monte_carlo.py                      # Monte Carlo entry
├── run_chop_filtered_report.py             # Chop filter 全績效報告
├── sweep_wma.py / sweep_wma_long.py        # WMA 參數 sweep
├── sweep_chop_filters.py                   # 單條件盤整 filter sweep
├── sweep_chop_filters_combo.py             # 雙/三條件疊加
├── sweep_early_exit_modes.py / sweep_early_exit_close.py  # 提前出場 sweep
├── sweep_hour_blacklist.py                 # 小時黑名單 sweep
├── validate_chop_filter_oos.py             # IS→OOS 驗證
├── analyze_stage3_features.py              # v1 univariate (16 features)
├── analyze_stage3_features_v2.py           # v2 univariate (11 indicators)
├── analyze_stage3_features_v3.py           # v3 univariate (WMA/WT/MACD div)
├── analyze_stage3_multivariate.py          # LR + GBM 24 features
├── analyze_peak2r_multivariate.py          # target = peak > 2R
├── analyze_filtered_stage_features.py      # filter 後再做特徵分析
├── analyze_R_and_stage_distribution.py     # R 分布 + stage 計數
├── analyze_realized_R.py                   # realized R vs peak R
├── analyze_hour_distribution.py            # 小時 expectancy
├── analyze_day_hour.py                     # day × hour 交叉
├── analyze_exit_stages.py                  # 各 stage 出場原因
├── analyze_first_bar_response.py           # 進場後第一根 K
├── analyze_prior_wick.py                   # 進場前 wick 形態
├── analyze_r_pct_buckets.py                # r_pct 分桶
├── analyze_rejection_shape.py              # 拒絕形態
├── analyze_signal_features.py              # 訊號當下特徵
├── compare_filter_stage_breakdown.py       # filter 對 stage 影響
└── view_chart.py                           # TradingView lightweight charts
```

## 附錄 B：模組架構

```
src/
├── backtest/        # event-driven engine
├── broker/          # account + simulator
├── data/            # loader + resampler
├── indicators/      # HA + WMA + ATR
├── metrics/         # 績效指標
├── reporting/       # 圖表 + 摘要輸出
├── strategy/        # long / short trend strategy
├── utils/           # config + exceptions
├── validation/      # Monte Carlo（新增）
└── viewer/          # FastAPI 視覺化
```

## 附錄 C：關鍵 PNG 索引

- `results/mc/15m_W4-6_both/bootstrap_hist.png` — MC bootstrap CI
- `results/mc/15m_W4-6_both/reshuffle_hist.png` — MC reshuffle CI
- `results/chop_filter_report/equity_compare.png` — IS/OOS equity baseline vs filtered
- `results/chop_filter_report_w24/equity_compare.png` — 同上 W(2,4)
- `results/r_stage_distribution/R_hist_IS.png` — R 分布（stage 著色）IS
- `results/r_stage_distribution/R_hist_OOS.png` — 同 OOS
- `results/hour_distribution/hour_distribution.png` — 小時 trade count + avg PnL
