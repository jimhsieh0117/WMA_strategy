"""分析「進場前 N 根 raw K 線特徵」與「最終出場 stage / R 倍數」的關聯。

主要回答：4-K 陰陽比例（與其他 candle-based 特徵）是否能事先區分出
高機率走進 stage 3 的訊號 → 用來設計訊號濾網。

用法：
    python -m scripts.analyze_signal_features \
        results/ETHUSDT_long_5m_is/trades.csv \
        results/ETHUSDT_short_5m_is/trades.csv \
        --config configs/default.yaml \
        --window 4

每根 K 用 raw OHLC（不是 HA）。entry_ts 對應 fill 那根，因此前 4 根 = 訊號本身（t）+ 之前 3 根（t-3..t）。

特徵：
    bull_count_ratio  : 陽 K 數 / N（多單看大、空單看小才有意義）
    body_weighted_ratio: Σ 陽實體 / Σ |實體|
    net_momentum_pct  : (close[t] − close[t−N+1]) / close[t−N+1]
    body_fullness     : avg(|close−open|) / avg(high−low)，衡量 K 線實體飽滿度

每個特徵分桶後做 cross-tab：
    bucket → trade_count / stage1_% / stage2_% / stage3_% / avg_R / trend_capture_rate
多單與空單分開報（多單期待陽多、空單期待陰多 — 對齊後可看 trend_alignment_ratio）。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# 載入
# --------------------------------------------------------------------------- #

REQUIRED = {"direction", "entry_ts", "exit_reason", "final_stage",
            "peak_progress_r", "net_pnl", "stop_distance", "quantity"}


def _resolve_trades_csv(target: Path) -> Path:
    if target.is_file():
        return target
    if target.is_dir():
        for name in ("trades.csv", "trades_live.csv"):
            cand = target / name
            if cand.is_file():
                return cand
    raise FileNotFoundError(f"not a trades.csv or run dir: {target}")


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_ts"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: 缺欄位 {sorted(missing)}；請用 final_stage 上線後的回測重跑"
        )
    return df


# --------------------------------------------------------------------------- #
# 特徵計算
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureSet:
    bull_count_ratio: float
    body_weighted_ratio: float
    net_momentum_pct: float
    body_fullness: float


def _compute_features(window: pd.DataFrame) -> FeatureSet:
    """``window`` 為連續 N 根 raw K（DataFrame，欄位含 open/high/low/close）。"""
    o = window["open"].to_numpy(dtype="float64")
    h = window["high"].to_numpy(dtype="float64")
    l = window["low"].to_numpy(dtype="float64")
    c = window["close"].to_numpy(dtype="float64")

    bodies = c - o
    abs_bodies = np.abs(bodies)
    ranges = h - l

    bull_count = int(np.sum(bodies > 0))
    n = len(window)
    bull_ratio = bull_count / n if n > 0 else 0.0

    bull_body_sum = float(np.sum(bodies[bodies > 0]))
    abs_body_sum = float(np.sum(abs_bodies))
    body_w = (bull_body_sum / abs_body_sum) if abs_body_sum > 0 else 0.5

    momentum = (c[-1] - c[0]) / c[0] if c[0] > 0 else 0.0
    fullness = (
        float(np.mean(abs_bodies)) / float(np.mean(ranges))
        if np.mean(ranges) > 0 else 0.0
    )

    return FeatureSet(
        bull_count_ratio=bull_ratio,
        body_weighted_ratio=body_w,
        net_momentum_pct=momentum,
        body_fullness=fullness,
    )


def attach_features(
    trades: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    window: int,
    timeframe_delta: pd.Timedelta,
) -> pd.DataFrame:
    """為每筆 trade 加上「進場前 N 根 raw K」特徵。

    entry_ts 是 fill 那根（t+1）；訊號 K 是 t = entry_ts − 1*timeframe；
    分析窗口 = [t-N+1, t]，總 N 根。
    """
    out_rows: list[dict] = []
    skipped_no_history = 0
    skipped_force_close = 0

    raw_idx = raw_df.index
    # 以 timestamp 找 bar 位置；entry_ts 應對齊某根 bar 的開盤時間
    for _, row in trades.iterrows():
        if str(row["exit_reason"]) == "FORCE_CLOSE_END":
            skipped_force_close += 1
            continue
        entry_ts = pd.Timestamp(row["entry_ts"])
        signal_ts = entry_ts - timeframe_delta
        # signal bar 在 raw_df 的位置
        if signal_ts not in raw_idx:
            # 對齊問題：找第一個 <= signal_ts 的 bar
            pos = raw_idx.searchsorted(signal_ts, side="right") - 1
            if pos < 0:
                skipped_no_history += 1
                continue
            actual_signal_ts = raw_idx[pos]
            # 容忍小於 1 個 timeframe 的偏差；超過則略過
            if abs(actual_signal_ts - signal_ts) > timeframe_delta:
                skipped_no_history += 1
                continue
            signal_pos = pos
        else:
            signal_pos = int(raw_idx.get_loc(signal_ts))

        start = signal_pos - window + 1
        if start < 0:
            skipped_no_history += 1
            continue
        win_df = raw_df.iloc[start:signal_pos + 1]
        if len(win_df) != window:
            skipped_no_history += 1
            continue

        feats = _compute_features(win_df)
        risk = row.get("stop_distance")
        qty = row.get("quantity")
        r_mult = (
            float(row["net_pnl"]) / (float(risk) * float(qty))
            if pd.notna(risk) and pd.notna(qty) and risk > 0 and qty > 0
            else float("nan")
        )

        out_rows.append({
            "direction": str(row["direction"]),
            "entry_ts": entry_ts,
            "exit_reason": row["exit_reason"],
            "final_stage": int(row["final_stage"]),
            "peak_progress_r": float(row["peak_progress_r"]),
            "net_pnl": float(row["net_pnl"]),
            "r_multiple": r_mult,
            "bull_count_ratio": feats.bull_count_ratio,
            "body_weighted_ratio": feats.body_weighted_ratio,
            "net_momentum_pct": feats.net_momentum_pct,
            "body_fullness": feats.body_fullness,
        })

    if skipped_no_history or skipped_force_close:
        print(
            f"  [info] skipped: no_history={skipped_no_history}, "
            f"force_close={skipped_force_close}",
            file=sys.stderr,
        )
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Cross-tab
# --------------------------------------------------------------------------- #

def _trend_alignment(row: pd.Series) -> float:
    """多單看陽 K 多寡（bull_count_ratio）；空單看陰 K 多寡（1 − ratio）。"""
    base = row["bull_count_ratio"]
    return 1.0 - base if str(row["direction"]).upper() == "SHORT" else base


def _trend_alignment_body(row: pd.Series) -> float:
    base = row["body_weighted_ratio"]
    return 1.0 - base if str(row["direction"]).upper() == "SHORT" else base


def _signed_momentum(row: pd.Series) -> float:
    """多單正向 momentum 為正、空單反之。"""
    sign = -1.0 if str(row["direction"]).upper() == "SHORT" else 1.0
    return sign * row["net_momentum_pct"]


def _bucket_summary(df: pd.DataFrame, *, feature: str, buckets, labels) -> pd.DataFrame:
    bins = pd.cut(df[feature], bins=buckets, labels=labels,
                  include_lowest=True, right=False)
    aug = df.assign(
        bucket=bins,
        is_s1=(df["final_stage"] == 1),
        is_s2=(df["final_stage"] == 2),
        is_s3=(df["final_stage"] == 3),
        is_win=(df["net_pnl"] > 0),
    )
    g = aug.groupby("bucket", observed=True)
    out = pd.DataFrame({
        "n":         g.size().astype(int),
        "stage1_%":  g["is_s1"].mean() * 100.0,
        "stage2_%":  g["is_s2"].mean() * 100.0,
        "stage3_%":  g["is_s3"].mean() * 100.0,
        "avg_R":     g["r_multiple"].mean(),
        "med_peakR": g["peak_progress_r"].median(),
        "winR_>0":   g["is_win"].mean() * 100.0,
    })
    return out.reindex(labels)


def render_table(df: pd.DataFrame, *, title: str) -> str:
    if df.empty:
        return f"  [{title}] (empty)\n"
    lines = [f"  {title}"]
    lines.append("  " + "-" * 88)
    header = (f"  {'bucket':<14} {'n':>5} {'stage1%':>8} {'stage2%':>8} "
              f"{'stage3%':>8} {'avg_R':>8} {'med_peakR':>10} {'winR>0%':>8}")
    lines.append(header)
    for bucket, row in df.iterrows():
        n = int(row["n"]) if not pd.isna(row["n"]) else 0
        if n == 0:
            lines.append(f"  {str(bucket):<14} {0:>5}  (empty)")
            continue
        lines.append(
            f"  {str(bucket):<14} {n:>5d} "
            f"{row['stage1_%']:>7.2f}% {row['stage2_%']:>7.2f}% "
            f"{row['stage3_%']:>7.2f}% {row['avg_R']:>+8.3f} "
            f"{row['med_peakR']:>+10.3f} {row['winR_>0']:>7.2f}%"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Signal feature analyzer")
    parser.add_argument("paths", nargs="+", help="trades.csv 路徑 / run dir")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window", type=int, default=4,
                        help="進場前回看 K 線根數（預設 4）")
    parser.add_argument("--save", default=None,
                        help="若指定，輸出帶特徵的 trade 表至此 CSV")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timeframe_delta = pd.Timedelta(
        cfg.timeframe.replace("H", "h").replace("M", "m")
    )

    print(f"[info] loading raw 1m parquet: {cfg.source_parquet}")
    df1m = load_ohlcv(cfg.source_parquet)
    print(f"[info] resampling 1m → {cfg.timeframe} ({len(df1m):,} bars)")
    raw_df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m

    all_features: list[pd.DataFrame] = []
    for raw in args.paths:
        path = _resolve_trades_csv(Path(raw))
        trades = load_trades(path)
        print(f"\n[info] processing {path} (trades={len(trades)})")
        feats = attach_features(
            trades, raw_df,
            window=args.window, timeframe_delta=timeframe_delta,
        )
        feats["source"] = path.parent.name
        all_features.append(feats)

    df_all = pd.concat(all_features, ignore_index=True)
    if df_all.empty:
        print("no analyzable trades", file=sys.stderr)
        sys.exit(1)

    df_all["trend_alignment"] = df_all.apply(_trend_alignment, axis=1)
    df_all["trend_alignment_body"] = df_all.apply(_trend_alignment_body, axis=1)
    df_all["signed_momentum"] = df_all.apply(_signed_momentum, axis=1)

    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_path, index=False)
        print(f"[info] saved features → {out_path}")

    # 全體 baseline
    n_total = len(df_all)
    s3_overall = (df_all["final_stage"] == 3).mean() * 100
    avg_R_overall = df_all["r_multiple"].mean()
    print()
    bar = "=" * 92
    print(bar)
    print(f"  Signal Feature Analysis  (window={args.window} raw bars before fill)")
    print(bar)
    print(f"  Total trades analyzed: {n_total}  "
          f"|  baseline stage3_pct={s3_overall:.2f}%  avg_R={avg_R_overall:+.3f}")
    print()

    # 1) bull_count_ratio（多空合一，看 trend_alignment）
    # 5 buckets 需要 6 條 edge；用 0/4=0, 1/4=0.25, 2/4=0.5, 3/4=0.75, 4/4=1.0 中間切
    align_buckets = [-0.01, 0.125, 0.375, 0.625, 0.875, 1.01]
    align_labels = ["0/4", "1/4", "2/4", "3/4", "4/4"]
    print("=" * 92)
    print("  [1] trend_alignment（多單=陽K占比；空單=陰K占比；4/4 = 完全順勢）")
    print(render_table(
        _bucket_summary(df_all, feature="trend_alignment",
                        buckets=align_buckets, labels=align_labels),
        title="combined long+short, by trend_alignment_count",
    ))
    for d in ("LONG", "SHORT"):
        sub = df_all[df_all["direction"].str.upper() == d]
        print(render_table(
            _bucket_summary(sub, feature="trend_alignment",
                            buckets=align_buckets, labels=align_labels),
            title=f"{d}-only",
        ))

    # 2) body_weighted_ratio
    body_buckets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    body_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    print("=" * 92)
    print("  [2] trend_alignment_body（實體大小加權；空單已對齊方向）")
    print(render_table(
        _bucket_summary(df_all, feature="trend_alignment_body",
                        buckets=body_buckets, labels=body_labels),
        title="combined long+short, by body-weighted alignment",
    ))

    # 3) signed_momentum（close[t] − close[t−N+1]，方向對齊後）
    mom_buckets = [-0.05, -0.01, -0.003, 0.003, 0.01, 0.05]
    mom_labels = ["<-1%", "-1~-0.3%", "-0.3~+0.3%", "+0.3~+1%", ">+1%"]
    # 用 quantile 改成自適應比較公平
    qs = df_all["signed_momentum"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    qs = sorted(set(qs))
    if len(qs) >= 6:
        qlabels = [f"q{i + 1}" for i in range(len(qs) - 1)]
        print("=" * 92)
        print("  [3] signed_momentum（quantile bucket，方向對齊後 (close_t - close_{t-3}) / close_{t-3}）")
        print(render_table(
            _bucket_summary(df_all, feature="signed_momentum",
                            buckets=qs, labels=qlabels),
            title="combined long+short, by signed momentum quintile",
        ))
        # 顯示桶邊界供解讀
        edges_str = ", ".join(f"{x:+.4f}" for x in qs)
        print(f"      quantile edges: [{edges_str}]")
        print()

    # 4) body_fullness
    full_buckets = [0.0, 0.3, 0.5, 0.7, 1.01]
    full_labels = ["<0.3", "0.3-0.5", "0.5-0.7", "≥0.7"]
    print("=" * 92)
    print("  [4] body_fullness（avg|實體| / avg(high-low)；高 = K 線實體飽滿、影線少）")
    print(render_table(
        _bucket_summary(df_all, feature="body_fullness",
                        buckets=full_buckets, labels=full_labels),
        title="combined long+short, by body_fullness",
    ))

    # 5) trend_alignment × body_fullness（雙特徵互動）
    print("=" * 92)
    print("  [5] 雙特徵：trend_alignment × body_fullness（看是否互補）")
    df_all["align_b"] = pd.cut(df_all["trend_alignment"],
                                bins=align_buckets, labels=align_labels,
                                include_lowest=True, right=False)
    df_all["full_b"] = pd.cut(df_all["body_fullness"],
                               bins=full_buckets, labels=full_labels,
                               include_lowest=True, right=False)
    pivot = df_all.pivot_table(
        index="align_b", columns="full_b",
        values="final_stage",
        aggfunc=lambda x: 100.0 * (x == 3).sum() / max(len(x), 1),
        observed=True,
    )
    n_pivot = df_all.pivot_table(
        index="align_b", columns="full_b",
        values="final_stage", aggfunc="count", observed=True,
    )
    print(f"  stage3 % by (align × fullness)：（括號 = 樣本數）")
    print(f"  {'':<8}", end="")
    for col in pivot.columns:
        print(f"{str(col):>14}", end="")
    print()
    for idx in pivot.index:
        print(f"  {str(idx):<8}", end="")
        for col in pivot.columns:
            v = pivot.loc[idx, col] if (idx, col) in pivot.stack().index else None
            n = int(n_pivot.loc[idx, col]) if (idx, col) in n_pivot.stack().index else 0
            if v is None or pd.isna(v):
                print(f"{'-':>14}", end="")
            else:
                print(f"  {v:>5.1f}% (n={n:>3})", end="")
        print()

    print()
    print("=" * 92)
    print("  解讀指引：")
    print("  - 看 stage3% 是否隨 bucket 單調上升 → 該特徵能分群")
    print("  - 比較最高桶 vs baseline 的 stage3% 差距 → 越大越能濾")
    print("  - 也比較最高桶的 avg_R 與整體 avg_R")
    print("  - 雙特徵 cross-tab 看是否有「對角強訊號」可組合濾網")
    print("=" * 92)


if __name__ == "__main__":
    main()
