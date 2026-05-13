"""Signal filter threshold sweep + stage 分布深度分析。

對 window=6 固定，掃 threshold ∈ {0.60..0.90}（step 0.05），對每組記錄：
- 每個 stage（1/2/3）的訂單數、貢獻 PnL、勝率
- IS + OOS 各自一份

目的：找出「threshold 提高時，是把 stage 1 砍掉多 vs 把 stage 3 也一起砍掉」。
理想是 threshold↑ 砍 s1 比例 > 砍 s3 比例，這樣 s3/s1 比率上升 → 訊號品質提升。

執行：
    python -m scripts.sweep_signal_filter_stages
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.utils.config import SignalFilterConfig, load_config  # noqa: E402


THRESHOLDS = [round(0.65 + i * 0.01, 2) for i in range(16)]  # 0.65..0.80 step 0.01
WINDOW = 4


def stage_summary(trades: list, direction: str) -> dict:
    """對一支策略的 trade list，按 final_stage 切分回統計。"""
    n_total = len(trades)
    out: dict = {"direction": direction, "n_total": n_total}
    if n_total == 0:
        for s in (1, 2, 3):
            out[f"s{s}_n"] = 0
            out[f"s{s}_pct"] = 0.0
            out[f"s{s}_pnl"] = 0.0
            out[f"s{s}_wr"] = 0.0
        out["total_pnl"] = 0.0
        return out

    pnls = np.array([float(t.net_pnl) for t in trades])
    stages = np.array([int(t.final_stage) for t in trades])
    for s in (1, 2, 3):
        mask = stages == s
        n = int(mask.sum())
        out[f"s{s}_n"] = n
        out[f"s{s}_pct"] = round(n / n_total * 100, 1) if n_total else 0.0
        out[f"s{s}_pnl"] = round(float(pnls[mask].sum()), 2)
        out[f"s{s}_wr"] = (
            round(float((pnls[mask] > 0).mean()) * 100, 1) if n else 0.0
        )
    out["total_pnl"] = round(float(pnls.sum()), 2)
    return out


def run_combo(base_cfg, *, threshold: float, sample: str) -> dict:
    cfg = dataclasses.replace(
        base_cfg,
        signal_filter=SignalFilterConfig(
            mode="body_sum", window=WINDOW, threshold=threshold,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    long_s = stage_summary(L.trades, "long")
    short_s = stage_summary(S.trades, "short")
    # combined
    all_trades = L.trades + S.trades
    comb_s = stage_summary(all_trades, "combined")
    return {"long": long_s, "short": short_s, "combined": comb_s}


def fmt_combined(threshold: float, m: dict) -> str:
    c = m["combined"]
    return (
        f"th={threshold:.2f}  n={c['n_total']:>4}  pnl={c['total_pnl']:>+8.2f}  "
        f"|  s1: {c['s1_n']:>3} ({c['s1_pct']:>4.1f}%) pnl={c['s1_pnl']:>+7.2f} wr={c['s1_wr']:>4.1f}%  "
        f"|  s2: {c['s2_n']:>3} ({c['s2_pct']:>4.1f}%) pnl={c['s2_pnl']:>+7.2f} wr={c['s2_wr']:>4.1f}%  "
        f"|  s3: {c['s3_n']:>3} ({c['s3_pct']:>4.1f}%) pnl={c['s3_pnl']:>+7.2f} wr={c['s3_wr']:>4.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal filter threshold × stage 分布分析。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out-csv", default="results/signal_filter_stage_sweep/stages.csv",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}  window: {WINDOW}")
    print(f"IS:  {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"Thresholds: {THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f} step 0.01  "
          f"(IS only, no look-ahead)")
    print()

    all_rows = []
    is_results = {}
    for th in THRESHOLDS:
        print(f">>> threshold={th:.2f}")
        is_results[th] = run_combo(base_cfg, threshold=th, sample="is")
        for direction in ("long", "short", "combined"):
            row = {"threshold": th, "sample": "IS",
                   **is_results[th][direction]}
            all_rows.append(row)

    # ---------- Console 視覺化 ----------
    print()
    print("=" * 140)
    print("Stage distribution (combined long+short)  —  IS only")
    print("=" * 140)
    for th in THRESHOLDS:
        print("  " + fmt_combined(th, is_results[th]))

    # ---------- 跨 threshold 對比 ----------
    print()
    print("=" * 110)
    print("IS  —  關鍵指標跨 threshold 對比（combined）")
    print("=" * 110)
    print(f"{'th':>5}  {'n':>5}  {'s1n':>5} {'s1%':>6} {'s2n':>5} {'s2%':>6} "
          f"{'s3n':>5} {'s3%':>6}  {'s3/s1':>6}  {'pnl':>9}")
    for th in THRESHOLDS:
        c = is_results[th]["combined"]
        s3_s1 = (c["s3_n"] / c["s1_n"]) if c["s1_n"] > 0 else float("inf")
        print(f"{th:>5.2f}  {c['n_total']:>5}  "
              f"{c['s1_n']:>5} {c['s1_pct']:>5.1f}%  "
              f"{c['s2_n']:>5} {c['s2_pct']:>5.1f}%  "
              f"{c['s3_n']:>5} {c['s3_pct']:>5.1f}%  "
              f"{s3_s1:>6.2f}  {c['total_pnl']:>+9.2f}")

    # ---------- 分 long / short ----------
    print()
    print("=" * 110)
    print("IS  —  分 direction × stage")
    print("=" * 110)
    for direction in ("long", "short"):
        print(f"\n  [{direction}]")
        print(f"  {'th':>5}  {'n':>4}  "
              f"{'s1n':>4} {'s1pnl':>8} {'s1wr':>5}  "
              f"{'s2n':>4} {'s2pnl':>8} {'s2wr':>5}  "
              f"{'s3n':>4} {'s3pnl':>8} {'s3wr':>5}  "
              f"{'total':>9}")
        for th in THRESHOLDS:
            d = is_results[th][direction]
            print(f"  {th:>5.2f}  {d['n_total']:>4}  "
                  f"{d['s1_n']:>4} {d['s1_pnl']:>+8.2f} {d['s1_wr']:>4.1f}%  "
                  f"{d['s2_n']:>4} {d['s2_pnl']:>+8.2f} {d['s2_wr']:>4.1f}%  "
                  f"{d['s3_n']:>4} {d['s3_pnl']:>+8.2f} {d['s3_wr']:>4.1f}%  "
                  f"{d['total_pnl']:>+9.2f}")

    df = pd.DataFrame(all_rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
