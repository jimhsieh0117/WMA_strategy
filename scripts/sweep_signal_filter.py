"""Signal filter window × threshold 參數掃描。

跑同一期間，對 (window, threshold) 二維網格各跑一次回測，輸出對照表：
- window     : {4, 6}（前 N 根 K，含訊號 bar）
- threshold  : {0.50, 0.55, 0.60, 0.65, 0.70}

對每組記錄 IS + OOS 的 combined metrics + n_trades。
最後依 OOS 的 PF 與 ret 排名挑出穩定組。

執行：
    python -m scripts.sweep_signal_filter
    python -m scripts.sweep_signal_filter --mode body_sq_sum
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import SignalFilterConfig, load_config  # noqa: E402


WINDOWS = [4, 6]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]


def run_combo(base_cfg, *, mode: str, window: int, threshold: float, sample: str) -> dict:
    cfg = dataclasses.replace(
        base_cfg,
        signal_filter=SignalFilterConfig(
            mode=mode, window=window, threshold=threshold,
        ),
        show_progress=False, log_level="WARNING",
    )
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    C = build_merged_result("combined", [L, S])
    ml = compute_metrics(L, timeframe=cfg.timeframe)
    ms = compute_metrics(S, timeframe=cfg.timeframe)
    mc = compute_metrics(C, timeframe=cfg.timeframe)
    return {
        "n_long": len(L.trades),
        "n_short": len(S.trades),
        "n": len(L.trades) + len(S.trades),
        "ret": mc.total_return_pct,
        "pf": mc.profit_factor,
        "wr": mc.win_rate_pct,
        "mdd": mc.max_drawdown_pct,
        "sharpe": mc.sharpe_ratio,
        "long_ret": ml.total_return_pct,
        "long_pf": ml.profit_factor,
        "short_ret": ms.total_return_pct,
        "short_pf": ms.profit_factor,
    }


def fmt_row(label: str, m: dict) -> str:
    return (
        f"  {label:<20}"
        f" n={m['n']:>4}"
        f" ret={m['ret']:>+7.2f}%"
        f" PF={m['pf']:>5.2f}"
        f" WR={m['wr']:>5.1f}%"
        f" MDD={m['mdd']:>5.1f}%"
        f" Sharpe={m['sharpe']:>5.2f}"
        f" | L:{m['long_ret']:>+6.2f}%/PF{m['long_pf']:>4.2f}"
        f" S:{m['short_ret']:>+6.2f}%/PF{m['short_pf']:>4.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal filter window × threshold 參數掃描。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", default="body_sum",
                        choices=["body_sum", "body_sq_sum"])
    parser.add_argument(
        "--out-csv", default="results/signal_filter_sweep/metrics.csv",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}  "
          f"mode: {args.mode}")
    print(f"IS:  {base_cfg.in_sample.start.date()} ~ {base_cfg.in_sample.end.date()}")
    print(f"OOS: {base_cfg.out_of_sample.start.date()} ~ "
          f"{base_cfg.out_of_sample.end}")
    print(f"Chop filter: {base_cfg.chop_filter.enabled}, "
          f"Structure filter: {base_cfg.structure_filter.enabled}/"
          f"{base_cfg.structure_filter.mode}")
    print()

    rows = []
    for w in WINDOWS:
        for th in THRESHOLDS:
            print(f">>> window={w}  threshold={th:.2f}")
            is_m = run_combo(base_cfg, mode=args.mode, window=w, threshold=th, sample="is")
            oos_m = run_combo(base_cfg, mode=args.mode, window=w, threshold=th, sample="oos")
            for label, sample_m in (("IS", is_m), ("OOS", oos_m)):
                rows.append({
                    "window": w, "threshold": th, "sample": label,
                    **sample_m,
                })

    df = pd.DataFrame(rows)

    # ----- Console 表 -----
    print()
    print("=" * 130)
    print(f"Signal filter sweep — mode={args.mode}")
    print("=" * 130)
    for w in WINDOWS:
        for th in THRESHOLDS:
            label = f"w={w}, th={th:.2f}"
            print(f"\n[{label}]")
            for sample in ("IS", "OOS"):
                m = df[(df["window"] == w) & (df["threshold"] == th)
                       & (df["sample"] == sample)].iloc[0].to_dict()
                print(fmt_row(sample, m))

    # ----- 排名 -----
    print()
    print("=" * 130)
    print("OOS ranking by composite (PF × sign(ret) × |ret|)")
    print("=" * 130)
    oos = df[df["sample"] == "OOS"].copy()
    # 簡單複合分數：PF * ret/100（同號相乘獎勵正、懲罰負）
    oos["score"] = oos["pf"] * (oos["ret"] / 100.0)
    oos_sorted = oos.sort_values("score", ascending=False)
    print(f"{'rank':>4} {'window':>7} {'thresh':>7} {'n':>5} "
          f"{'ret%':>8} {'PF':>5} {'MDD%':>6} {'Sharpe':>7} {'score':>7}")
    for i, row in enumerate(oos_sorted.itertuples(), start=1):
        print(f"{i:>4} {row.window:>7} {row.threshold:>7.2f} {row.n:>5} "
              f"{row.ret:>+8.2f} {row.pf:>5.2f} {row.mdd:>6.1f} "
              f"{row.sharpe:>+7.2f} {row.score:>+7.3f}")

    # ----- CSV -----
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
