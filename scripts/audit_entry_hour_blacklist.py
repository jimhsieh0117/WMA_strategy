"""Audit entry_hour_blacklist=[0, 12] 是否真的有效（IS + OOS）。

問題：當前 [0, 12] 是 IS sweep 後驗結果，疑似 overfit。
做法：
  1. 跑「無黑名單」版本 IS + OOS（讓 hour 0 與 hour 12 的訊號全部進場）
  2. 按 hour bucket 統計 PnL / WR / count
  3. 對照 hour 0 跟 hour 12 在 IS 與 OOS 是否仍劣勢
  4. 整體 metrics 對照：有黑名單 vs 無黑名單

執行：
    python -m scripts.audit_entry_hour_blacklist
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.metrics.calculator import compute_metrics  # noqa: E402
from src.metrics.merger import build_merged_result  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def trade_rows(trades: list, direction: str) -> list[dict]:
    out = []
    for t in trades:
        out.append({
            "direction": direction,
            "hour": int(t.entry_timestamp.hour),
            "net_pnl": float(t.net_pnl),
            "final_stage": int(t.final_stage),
            "is_win": bool(t.net_pnl > 0),
        })
    return out


def run_pair(cfg, sample: str) -> tuple[pd.DataFrame, dict]:
    """跑 long + short，回傳 (trades_df, combined_metrics)。"""
    L = run_single_strategy(cfg, direction="long", sample=sample)
    S = run_single_strategy(cfg, direction="short", sample=sample)
    rows = trade_rows(L.trades, "long") + trade_rows(S.trades, "short")
    df = pd.DataFrame(rows)
    C = build_merged_result(f"comb_{sample}", [L, S])
    mC = compute_metrics(C, timeframe=cfg.timeframe)
    return df, {
        "n": len(rows),
        "ret_pct": mC.total_return_pct, "pf": mC.profit_factor,
        "wr_pct": mC.win_rate_pct, "mdd_pct": mC.max_drawdown_pct,
        "sharpe": mC.sharpe_ratio,
    }


def hour_stats(df: pd.DataFrame) -> pd.DataFrame:
    """每個 hour 的 n / wins / WR% / total_pnl / avg_pnl / PF。"""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for h in range(24):
        sub = df[df.hour == h]
        if len(sub) == 0:
            rows.append({"hour": h, "n": 0, "wr": np.nan, "total": 0.0,
                         "avg": np.nan, "pf": np.nan})
            continue
        wins = sub[sub.net_pnl > 0]
        gw = float(wins.net_pnl.sum())
        gl = float(sub[sub.net_pnl <= 0].net_pnl.abs().sum())
        pf = gw / gl if gl > 0 else float("inf")
        rows.append({
            "hour": h, "n": len(sub),
            "wr": 100 * len(wins) / len(sub),
            "total": float(sub.net_pnl.sum()),
            "avg": float(sub.net_pnl.mean()),
            "pf": pf,
        })
    return pd.DataFrame(rows)


def print_hour_table(title: str, df: pd.DataFrame, mark: list[int]) -> None:
    print()
    print("=" * 90)
    print(title + f"   （★ = 當前黑名單 hour {mark}）")
    print("=" * 90)
    print(f"{'hour':>4}  {'mark':>4}  {'n':>4}  {'WR%':>6}  {'total':>9}  "
          f"{'avg':>7}  {'PF':>6}")
    for _, r in df.iterrows():
        h = int(r.hour)
        star = "★" if h in mark else " "
        wr = f"{r.wr:>5.1f}%" if not np.isnan(r.wr) else "    —"
        avg = f"{r.avg:>+6.2f}" if not np.isnan(r.avg) else "     —"
        pf = "    —" if np.isnan(r.pf) else (f"{r.pf:>5.2f}" if np.isfinite(r.pf) else "  inf")
        print(f"{h:>4}  {star:>4}  {int(r.n):>4}  {wr}  "
              f"{r.total:>+9.2f}  {avg}  {pf}")


def main() -> None:
    base_cfg = load_config("configs/default.yaml")
    current_bl = list(base_cfg.entry_hour_blacklist)
    print(f"Symbol: {base_cfg.symbol}  tf: {base_cfg.timeframe}")
    print(f"當前 entry_hour_blacklist: {current_bl}")
    print(f"Audit: 跑「無黑名單」版本 IS + OOS, 看 hour 0/12 是否真的劣勢\n")

    # 跑無黑名單版本
    cfg_no_bl = dataclasses.replace(
        base_cfg, entry_hour_blacklist=(),
        show_progress=False, log_level="WARNING",
    )
    print(">>> 無黑名單 IS")
    is_df, is_metrics = run_pair(cfg_no_bl, "is")
    print(">>> 無黑名單 OOS")
    oos_df, oos_metrics = run_pair(cfg_no_bl, "oos")

    # 跑有黑名單版本（baseline）
    print(">>> 當前黑名單 IS")
    _, is_bl_metrics = run_pair(base_cfg, "is")
    print(">>> 當前黑名單 OOS")
    _, oos_bl_metrics = run_pair(base_cfg, "oos")

    # 整體對照
    print()
    print("=" * 90)
    print("Combined metrics 對照（黑名單 on vs off）")
    print("=" * 90)
    print(f"{'config':<18}  {'sample':>6}  {'n':>4}  {'ret%':>7}  {'PF':>5}  "
          f"{'WR%':>5}  {'MDD%':>5}  {'Sharpe':>7}")
    rows = [
        ("黑名單 [0, 12]", "IS", is_bl_metrics),
        ("無黑名單",        "IS", is_metrics),
        ("黑名單 [0, 12]", "OOS", oos_bl_metrics),
        ("無黑名單",        "OOS", oos_metrics),
    ]
    for label, sample, m in rows:
        print(f"{label:<18}  {sample:>6}  {m['n']:>4}  "
              f"{m['ret_pct']:>+7.2f}  {m['pf']:>5.2f}  "
              f"{m['wr_pct']:>4.1f}%  {m['mdd_pct']:>4.1f}%  "
              f"{m['sharpe']:>+7.2f}")

    # 各 hour 表現
    is_hs = hour_stats(is_df)
    oos_hs = hour_stats(oos_df)
    print_hour_table("IS hour bucket（無黑名單）", is_hs, current_bl)
    print_hour_table("OOS hour bucket（無黑名單）", oos_hs, current_bl)

    # 重點對照 hour 0 跟 12
    print()
    print("=" * 90)
    print("Audit 結論：[0, 12] 在 IS 與 OOS 是否仍是「該擋的爛時段」？")
    print("=" * 90)
    print(f"{'hour':>4}  {'IS_n':>5}  {'IS_avg':>8}  {'IS_PF':>6}  "
          f"{'OOS_n':>6}  {'OOS_avg':>9}  {'OOS_PF':>7}  {'verdict':>15}")
    for h in current_bl:
        is_row = is_hs[is_hs.hour == h].iloc[0]
        oos_row = oos_hs[oos_hs.hour == h].iloc[0]
        is_avg = is_row.avg if is_row.n > 0 else None
        oos_avg = oos_row.avg if oos_row.n > 0 else None
        is_pf = is_row.pf if is_row.n > 0 else None
        oos_pf = oos_row.pf if oos_row.n > 0 else None

        # 判決邏輯：OOS avg < 0 或 OOS PF < 1 → 仍劣（保留）
        # OOS avg > 0 且 OOS PF >= 1 → 黑名單在 OOS 無效（建議砍）
        # OOS n < 5 → 樣本不足
        if oos_avg is None or oos_row.n < 5:
            verdict = "OOS 樣本不足"
        elif oos_avg < 0 and (oos_pf is None or oos_pf < 1):
            verdict = "仍劣勢 → 保留"
        else:
            verdict = "OOS 不劣 → 該砍"

        is_avg_s = f"{is_avg:>+7.2f}" if is_avg is not None else "      —"
        oos_avg_s = f"{oos_avg:>+8.2f}" if oos_avg is not None else "       —"
        is_pf_s = f"{is_pf:>5.2f}" if is_pf is not None and np.isfinite(is_pf) else "    —"
        oos_pf_s = f"{oos_pf:>6.2f}" if oos_pf is not None and np.isfinite(oos_pf) else "     —"

        print(f"{h:>4}  {int(is_row.n):>5}  {is_avg_s}  {is_pf_s}  "
              f"{int(oos_row.n):>6}  {oos_avg_s}  {oos_pf_s}  {verdict:>15}")

    out_dir = Path("results/audit_hour_blacklist")
    out_dir.mkdir(parents=True, exist_ok=True)
    is_df.to_csv(out_dir / "is_trades_no_blacklist.csv", index=False)
    oos_df.to_csv(out_dir / "oos_trades_no_blacklist.csv", index=False)
    is_hs.to_csv(out_dir / "is_hour_stats.csv", index=False)
    oos_hs.to_csv(out_dir / "oos_hour_stats.csv", index=False)
    print(f"\nOutput: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
