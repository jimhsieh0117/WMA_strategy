"""多變量分類器 v2：target = peak_progress_r > 2 （任何曾達 2R 的訂單）。

與 analyze_stage3_multivariate.py 共用 24 個 pre-entry 特徵與 train/test 切點，
但把 target 從「stage 3」改成「曾達 2R」：
  - 不被 trailing stop 設計干擾（stage label 是 trailing 結果，不是純市場行為）
  - 包含被 stage 2 提前止盈但實際走過 2R 的訂單
  - 預期 target 樣本數略少且訊號可能更乾淨

用法：
    python scripts/analyze_peak2r_multivariate.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analyze_stage3_multivariate import (  # noqa: E402
    COMBOS, FEATURES, TEST_PERIOD, TRAIN_PERIOD, _build_features,
)
from src.utils.config import load_config  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

PEAK_THRESHOLD = 2.0


def _train_eval(df_feat: pd.DataFrame) -> dict:
    df_feat = df_feat.copy()
    df_feat["y"] = (df_feat.peak_progress_r > PEAK_THRESHOLD).astype(int)

    train = df_feat[(df_feat.entry_ts >= TRAIN_PERIOD.start) &
                    (df_feat.entry_ts <= TRAIN_PERIOD.end)]
    test = df_feat[(df_feat.entry_ts >= TEST_PERIOD.start) &
                   (df_feat.entry_ts <= TEST_PERIOD.end)]

    if train.empty or test.empty:
        return {"error": "empty split"}

    Xtr, ytr = train[FEATURES].to_numpy(), train["y"].to_numpy()
    Xte, yte = test[FEATURES].to_numpy(), test["y"].to_numpy()

    if ytr.sum() == 0 or yte.sum() == 0:
        return {"error": "no positives in split"}

    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(Xtr, ytr)
    lr_train = roc_auc_score(ytr, lr.predict_proba(Xtr)[:, 1])
    lr_test = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
    coefs = lr.named_steps["lr"].coef_[0]
    lr_imp = sorted(zip(FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True)

    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, subsample=0.8,
    )
    gbm.fit(Xtr, ytr)
    gbm_train = roc_auc_score(ytr, gbm.predict_proba(Xtr)[:, 1])
    gbm_test = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
    gbm_imp = sorted(zip(FEATURES, gbm.feature_importances_),
                     key=lambda x: x[1], reverse=True)

    proba_te = gbm.predict_proba(Xte)[:, 1]
    test_df = test.assign(p=proba_te).sort_values("p")
    n = len(test_df)
    q1 = test_df.iloc[:n // 3]
    q2 = test_df.iloc[n // 3: 2 * n // 3]
    q3 = test_df.iloc[2 * n // 3:]
    base = test["y"].mean()
    quartile = {
        "base_rate": float(base),
        "bot_30%":   float(q1["y"].mean()),
        "mid_30%":   float(q2["y"].mean()),
        "top_30%":   float(q3["y"].mean()),
        # 同時看 PnL：top 30% 訂單真的拿來交易會怎樣
        "all_pnl":   float(test["net_pnl"].sum()),
        "top30_pnl": float(q3["net_pnl"].sum()),
    }

    return {
        "n_train": len(train), "n_test": len(test),
        "train_pos_rate": float(ytr.mean()),
        "test_pos_rate": float(yte.mean()),
        "lr_train_auc": lr_train, "lr_test_auc": lr_test,
        "gbm_train_auc": gbm_train, "gbm_test_auc": gbm_test,
        "lr_top5": lr_imp[:5], "gbm_top5": gbm_imp[:5],
        "quartile_stats": quartile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="results/peak2r_multivariate.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    base_cfg = load_config(args.config)

    all_results = []
    feature_dfs = []

    for tf, wf, ws in COMBOS:
        name = f"{tf}_W{wf}-{ws}"
        print(f"\n>>> {name}: building features ...")
        df_feat = _build_features(tf, wf, ws, base_cfg)
        if df_feat.empty:
            continue
        n = len(df_feat)
        n_pos = (df_feat.peak_progress_r > PEAK_THRESHOLD).sum()
        print(f"  trades={n}  peak>2R: {n_pos} ({n_pos/n*100:.2f}%)")
        feature_dfs.append(df_feat.assign(combo=name))

        res = _train_eval(df_feat)
        res["combo"] = name
        all_results.append(res)

        if "error" in res:
            print(f"  [skip] {res['error']}")
            continue

        print(f"  n_train={res['n_train']} (pos {res['train_pos_rate']*100:.1f}%) "
              f"n_test={res['n_test']} (pos {res['test_pos_rate']*100:.1f}%)")
        print(f"  LR  AUC  train={res['lr_train_auc']:.4f}  test={res['lr_test_auc']:.4f}")
        print(f"  GBM AUC  train={res['gbm_train_auc']:.4f}  test={res['gbm_test_auc']:.4f}")
        print("  GBM top-5 importance:")
        for f, v in res["gbm_top5"]:
            print(f"    {f:25s} {v:.4f}")
        print("  LR  top-5 |coef|:")
        for f, v in res["lr_top5"]:
            print(f"    {f:25s} {v:+.4f}")
        q = res["quartile_stats"]
        print(f"  Test quantile（GBM 機率排序）peak>2R 命中率：")
        print(f"    base_rate = {q['base_rate']*100:.2f}%")
        print(f"    bot 30%   = {q['bot_30%']*100:.2f}%")
        print(f"    mid 30%   = {q['mid_30%']*100:.2f}%")
        print(f"    top 30%   = {q['top_30%']*100:.2f}%   "
              f"(lift = {q['top_30%']/q['base_rate']:.2f}x)")
        print(f"  Test PnL：全集合 {q['all_pnl']:+.2f}  "
              f"top 30% 子集 {q['top30_pnl']:+.2f}")

    print("\n\n" + "=" * 70)
    print("=== 跨 4 組合 AUC 摘要（target = peak > 2R）===")
    print("=" * 70)
    summary = pd.DataFrame([
        {
            "combo": r["combo"],
            "lr_train": r.get("lr_train_auc", float("nan")),
            "lr_test":  r.get("lr_test_auc", float("nan")),
            "gbm_train": r.get("gbm_train_auc", float("nan")),
            "gbm_test":  r.get("gbm_test_auc", float("nan")),
            "top30_lift": (r["quartile_stats"]["top_30%"] / r["quartile_stats"]["base_rate"]
                           if "quartile_stats" in r else float("nan")),
            "top30_pnl": r.get("quartile_stats", {}).get("top30_pnl", float("nan")),
            "all_pnl":   r.get("quartile_stats", {}).get("all_pnl",   float("nan")),
        } for r in all_results
    ]).set_index("combo")
    print(summary.to_string(float_format="%.4f"))

    print("\n判定：")
    valid = summary[["lr_test", "gbm_test"]].dropna()
    if not valid.empty:
        mean_test = valid.mean(axis=None)
        print(f"  4 組合 test AUC 平均（LR + GBM）= {mean_test:.4f}")
        if mean_test >= 0.55:
            print("  → 多變量在 peak>2R target 上有 edge")
        elif mean_test >= 0.52:
            print("  → 邊際訊號")
        else:
            print("  → target 改框依然無 edge")

    if feature_dfs:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(feature_dfs, ignore_index=True).to_csv(
            out_path, index=False, float_format="%.6f")
        print(f"\n💾 features → {out_path}")


if __name__ == "__main__":
    main()
