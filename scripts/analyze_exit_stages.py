"""分析回測 / live_sim 的 trades.csv：出場 stage 分布、趨勢捕獲率、peak_R 直方圖。

用法：
    python -m scripts.analyze_exit_stages results/ETHUSDT_long_5m_is/trades.csv
    python -m scripts.analyze_exit_stages results/ETHUSDT_long_5m_is results/ETHUSDT_short_5m_is

可同時傳多個 trades.csv 或 run dir（會自動找其下 trades.csv），逐一分析後再合併。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

REQUIRED_COLS = {"final_stage", "peak_progress_r", "exit_reason", "net_pnl",
                 "direction", "stop_distance", "quantity"}


def _resolve_trades_csv(target: Path) -> Path:
    if target.is_file():
        return target
    if target.is_dir():
        candidate = target / "trades.csv"
        if candidate.is_file():
            return candidate
        candidate = target / "trades_live.csv"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"No trades.csv / trades_live.csv under {target}")
    raise FileNotFoundError(f"Not a file or directory: {target}")


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing columns {sorted(missing)}. "
            f"Re-run backtest after the schema upgrade (final_stage / peak_progress_r)."
        )
    return df


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _classify_exit(row: pd.Series) -> str:
    """把 (final_stage, exit_reason) 映射成單一分類字串。"""
    reason = str(row["exit_reason"])
    if reason == "FORCE_CLOSE_END":
        return "force_close"
    stage = int(row["final_stage"])
    is_gap = reason == "STOP_LOSS_GAP"
    base = {1: "stage1_stop", 2: "stage2_stop", 3: "stage3_stop"}.get(stage, "unknown")
    return base + ("_gap" if is_gap else "")


def _r_multiple(row: pd.Series) -> float | None:
    """net_pnl 換算成 R 倍數（除以 |entry − stop| × quantity）。stop_distance × quantity 為扣費前風險。"""
    risk = row.get("stop_distance")
    qty = row.get("quantity")
    if pd.isna(risk) or pd.isna(qty) or risk in (0, None) or qty in (0, None):
        return None
    return float(row["net_pnl"]) / (float(risk) * float(qty))


def summarize(df: pd.DataFrame, *, label: str) -> dict:
    df = df.copy()
    df["exit_class"] = df.apply(_classify_exit, axis=1)
    df["r_multiple"] = df.apply(_r_multiple, axis=1)

    total = len(df)
    excluded = (df["exit_class"] == "force_close").sum()
    counted = total - excluded

    counts = df["exit_class"].value_counts().to_dict()

    def _pct(n: int) -> float:
        return 100.0 * n / total if total else 0.0

    s1 = sum(v for k, v in counts.items() if k.startswith("stage1_stop"))
    s2 = sum(v for k, v in counts.items() if k.startswith("stage2_stop"))
    s3 = sum(v for k, v in counts.items() if k.startswith("stage3_stop"))
    fc = counts.get("force_close", 0)

    trend_capture = (s3 / counted * 100.0) if counted else 0.0
    pnl_winrate = (df["net_pnl"] > 0).sum() / counted * 100.0 if counted else 0.0

    # 各分類的 PnL 統計
    cat_stats: dict[str, dict] = {}
    for cls, group in df.groupby("exit_class"):
        cat_stats[str(cls)] = {
            "count": int(len(group)),
            "avg_net_pnl": float(group["net_pnl"].mean()),
            "avg_r_multiple": float(group["r_multiple"].dropna().mean()) if group["r_multiple"].notna().any() else 0.0,
            "median_r_multiple": float(group["r_multiple"].dropna().median()) if group["r_multiple"].notna().any() else 0.0,
        }

    # peak_progress_r 直方
    bins = [0.0, 1.2, 2.4, 4.0, 6.0, 9.0, 1e9]
    bin_labels = ["<1.2R", "1.2–2.4R", "2.4–4R", "4–6R", "6–9R", "≥9R"]
    df["peak_bucket"] = pd.cut(df["peak_progress_r"], bins=bins,
                                labels=bin_labels, right=False, include_lowest=True)
    peak_dist = df["peak_bucket"].value_counts().reindex(bin_labels, fill_value=0).to_dict()

    return {
        "label": label,
        "total": int(total),
        "excluded_force_close": int(fc),
        "counted": int(counted),
        "stage1_stop_count": int(s1),
        "stage2_stop_count": int(s2),
        "stage3_stop_count": int(s3),
        "stage1_pct": _pct(s1),
        "stage2_pct": _pct(s2),
        "stage3_pct": _pct(s3),
        "force_close_pct": _pct(fc),
        "trend_capture_rate": trend_capture,
        "pnl_win_rate": pnl_winrate,
        "by_class": cat_stats,
        "peak_distribution": {k: int(v) for k, v in peak_dist.items()},
    }


# --------------------------------------------------------------------------- #
# Pretty print
# --------------------------------------------------------------------------- #

def render_summary(s: dict) -> str:
    lines: list[str] = []
    bar = "=" * 64
    lines.append(bar)
    lines.append(f"  Exit Stage Analysis  [{s['label']}]")
    lines.append(bar)
    lines.append(f"  Total trades:          {s['total']:>6d}")
    lines.append(f"  Force closed (excluded):{s['excluded_force_close']:>5d}  "
                 f"({s['force_close_pct']:.2f}%)")
    lines.append(f"  Counted (denom):       {s['counted']:>6d}")
    lines.append("-" * 64)
    lines.append(f"  Stage 1 stop:          {s['stage1_stop_count']:>6d}  "
                 f"({s['stage1_pct']:.2f}%)")
    lines.append(f"  Stage 2 stop:          {s['stage2_stop_count']:>6d}  "
                 f"({s['stage2_pct']:.2f}%)")
    lines.append(f"  Stage 3 stop:          {s['stage3_stop_count']:>6d}  "
                 f"({s['stage3_pct']:.2f}%)")
    lines.append("-" * 64)
    lines.append(f"  Trend-capture rate:    {s['trend_capture_rate']:>6.2f} %  "
                 f"(stage3 / counted)")
    lines.append(f"  Traditional win rate:  {s['pnl_win_rate']:>6.2f} %  "
                 f"(net_pnl > 0)")
    lines.append("-" * 64)
    lines.append("  By exit class:")
    lines.append(f"    {'class':<22} {'count':>6} {'avg PnL':>10} {'avg R':>8} {'med R':>8}")
    for cls in sorted(s["by_class"].keys()):
        st = s["by_class"][cls]
        lines.append(f"    {cls:<22} {st['count']:>6d} {st['avg_net_pnl']:>10.4f} "
                     f"{st['avg_r_multiple']:>+8.3f} {st['median_r_multiple']:>+8.3f}")
    lines.append("-" * 64)
    lines.append("  Peak progress (R) distribution:")
    for bucket, cnt in s["peak_distribution"].items():
        bar_len = int(cnt / max(s["total"], 1) * 40)
        lines.append(f"    {bucket:<10} {cnt:>5d}  {'█' * bar_len}")
    lines.append(bar)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Exit stage analyzer")
    parser.add_argument("paths", nargs="+",
                        help="trades.csv 路徑或 run dir（自動找 trades.csv）")
    parser.add_argument("--combined", action="store_true",
                        help="若多個來源，最後再做一份合併分析")
    args = parser.parse_args()

    all_dfs: list[tuple[str, pd.DataFrame]] = []
    for raw in args.paths:
        path = _resolve_trades_csv(Path(raw))
        df = load_trades(path)
        label = path.parent.name + "/" + path.name
        all_dfs.append((label, df))

    for label, df in all_dfs:
        summary = summarize(df, label=label)
        print(render_summary(summary))
        print()

    if args.combined and len(all_dfs) > 1:
        merged = pd.concat([df for _, df in all_dfs], ignore_index=True)
        summary = summarize(merged, label=f"COMBINED ({len(all_dfs)} sources)")
        print(render_summary(summary))


if __name__ == "__main__":
    main()
