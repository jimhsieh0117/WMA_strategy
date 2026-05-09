"""壞訊號分析：對比 stage1（被 swing stop 直接打死）與 stage3（吃到趨勢）的訊號特徵。

目的：找出哪些 entry-prior 特徵能事先區分「無趨勢延續力」的訊號 → 設計濾網。

每個特徵會回答：
  1. stage1 vs stage3 的分布有沒有顯著差距（用 Cohen's d-like 效應量）
  2. 若用「該特徵 ≥ 閾值」做單一濾網，能不能在保留 stage3 同時砍掉 stage1
  3. 從多個閾值掃出「最大 lift」與「最高 stage3 留存率」兩組推薦

用法：
    python -m scripts.analyze_bad_signals \\
        results/ETHUSDT_long_5m_is/trades.csv \\
        results/ETHUSDT_short_5m_is/trades.csv \\
        --config configs/default.yaml --window 4
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


REQUIRED = {"direction", "entry_ts", "exit_reason", "final_stage",
            "peak_progress_r", "net_pnl", "stop_distance", "quantity"}


# --------------------------------------------------------------------------- #
# 載入
# --------------------------------------------------------------------------- #

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
        raise ValueError(f"{path}: 缺欄位 {sorted(missing)}")
    return df


# --------------------------------------------------------------------------- #
# 特徵計算（擴充版）
# --------------------------------------------------------------------------- #

def _compute_features(window: pd.DataFrame, *, is_short: bool) -> dict:
    """``window`` = N 根 raw K（最後一根是訊號 K t）。

    所有方向相關特徵已對齊：long 不變、short 翻號（讓「順勢」=正向）。
    """
    o = window["open"].to_numpy("float64")
    h = window["high"].to_numpy("float64")
    l = window["low"].to_numpy("float64")
    c = window["close"].to_numpy("float64")
    n = len(window)

    bodies = c - o
    abs_bodies = np.abs(bodies)
    ranges = h - l
    upper_wicks = h - np.maximum(o, c)
    lower_wicks = np.minimum(o, c) - l

    # 訊號 K 自身（最後一根）
    sig_o, sig_h, sig_l, sig_c = o[-1], h[-1], l[-1], c[-1]
    sig_body = sig_c - sig_o
    sig_range = sig_h - sig_l
    sig_abs_body = abs(sig_body)

    # 視窗高低
    win_high = float(h.max())
    win_low = float(l.min())
    win_span = win_high - win_low

    sign = -1.0 if is_short else 1.0

    feats = {}

    # ---- 已有的 4 個 ----
    bull_ratio = float(np.sum(bodies > 0) / n)
    feats["align_count"] = bull_ratio if not is_short else (1.0 - bull_ratio)

    bull_body_sum = float(bodies[bodies > 0].sum())
    abs_body_sum = float(abs_bodies.sum())
    body_w = (bull_body_sum / abs_body_sum) if abs_body_sum > 0 else 0.5
    feats["align_body"] = body_w if not is_short else (1.0 - body_w)

    feats["momentum_pct"] = sign * ((c[-1] - c[0]) / c[0]) if c[0] > 0 else 0.0
    feats["body_fullness"] = (abs_bodies.mean() / ranges.mean()) if ranges.mean() > 0 else 0.0

    # ---- 訊號 K 自身的特徵 ----
    feats["sig_body_pct"] = sign * (sig_body / sig_o) if sig_o > 0 else 0.0
    feats["sig_body_size_pct"] = sig_abs_body / sig_o if sig_o > 0 else 0.0
    feats["sig_range_pct"] = sig_range / sig_l if sig_l > 0 else 0.0
    feats["sig_body_to_range"] = (sig_abs_body / sig_range) if sig_range > 0 else 0.0
    # 訊號 K 收盤位置：long 期待靠 high，short 期待靠 low
    if sig_range > 0:
        raw_pos = (sig_c - sig_l) / sig_range  # 0=收最低、1=收最高
        feats["sig_close_in_bar"] = raw_pos if not is_short else (1.0 - raw_pos)
    else:
        feats["sig_close_in_bar"] = 0.5

    # ---- 訊號 K 在視窗內的位置 ----
    if win_span > 0:
        raw_pos_w = (sig_c - win_low) / win_span
        feats["sig_close_in_window"] = raw_pos_w if not is_short else (1.0 - raw_pos_w)
    else:
        feats["sig_close_in_window"] = 0.5

    # 突破：訊號 K 是否創 N-bar 新高（長）/ 新低（短）
    prior_high = float(h[:-1].max()) if n > 1 else sig_h
    prior_low = float(l[:-1].min()) if n > 1 else sig_l
    if not is_short:
        feats["sig_breakout"] = (sig_c - prior_high) / prior_high if prior_high > 0 else 0.0
    else:
        feats["sig_breakout"] = (prior_low - sig_c) / prior_low if prior_low > 0 else 0.0

    # ---- 視窗整體波動 / 影線結構 ----
    feats["avg_range_pct"] = float(ranges.mean() / c.mean()) if c.mean() > 0 else 0.0
    total_range_sum = float(ranges.sum())
    if total_range_sum > 0:
        # long：上影線比例小較好（收高未被打回）；short 反之 → 對齊後「逆向影線比例」
        upper_ratio = float(upper_wicks.sum() / total_range_sum)
        lower_ratio = float(lower_wicks.sum() / total_range_sum)
        feats["adverse_wick_ratio"] = upper_ratio if not is_short else lower_ratio
    else:
        feats["adverse_wick_ratio"] = 0.0

    return feats


def attach_features(
    trades: pd.DataFrame,
    raw_df: pd.DataFrame,
    *,
    window: int,
    timeframe_delta: pd.Timedelta,
) -> pd.DataFrame:
    out_rows: list[dict] = []
    skipped = {"no_history": 0, "force_close": 0}
    raw_idx = raw_df.index

    for _, row in trades.iterrows():
        if str(row["exit_reason"]) == "FORCE_CLOSE_END":
            skipped["force_close"] += 1
            continue
        entry_ts = pd.Timestamp(row["entry_ts"])
        signal_ts = entry_ts - timeframe_delta
        if signal_ts not in raw_idx:
            pos = raw_idx.searchsorted(signal_ts, side="right") - 1
            if pos < 0:
                skipped["no_history"] += 1
                continue
            actual = raw_idx[pos]
            if abs(actual - signal_ts) > timeframe_delta:
                skipped["no_history"] += 1
                continue
            signal_pos = pos
        else:
            signal_pos = int(raw_idx.get_loc(signal_ts))
        start = signal_pos - window + 1
        if start < 0:
            skipped["no_history"] += 1
            continue
        win_df = raw_df.iloc[start:signal_pos + 1]
        if len(win_df) != window:
            skipped["no_history"] += 1
            continue

        is_short = str(row["direction"]).upper() == "SHORT"
        feats = _compute_features(win_df, is_short=is_short)

        risk = row.get("stop_distance")
        qty = row.get("quantity")
        r_mult = (
            float(row["net_pnl"]) / (float(risk) * float(qty))
            if pd.notna(risk) and pd.notna(qty) and risk > 0 and qty > 0
            else float("nan")
        )

        out_rows.append({
            "direction": str(row["direction"]).upper(),
            "entry_ts": entry_ts,
            "final_stage": int(row["final_stage"]),
            "peak_progress_r": float(row["peak_progress_r"]),
            "net_pnl": float(row["net_pnl"]),
            "r_multiple": r_mult,
            **feats,
        })

    if any(skipped.values()):
        print(f"  [info] skipped: {skipped}", file=sys.stderr)
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# 統計：stage1 vs stage3 對比
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureCompare:
    feature: str
    n1: int
    n3: int
    mean_s1: float
    mean_s3: float
    median_s1: float
    median_s3: float
    std_s1: float
    std_s3: float
    cohens_d: float    # 標準化效應量
    diff_pct: float    # (mean_s3 - mean_s1) / |mean_overall|


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d；用 pooled std。|d| < 0.2 微小、0.2-0.5 小、0.5-0.8 中、>0.8 大。"""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s1, s3 = float(a.std(ddof=1)), float(b.std(ddof=1))
    n1, n3 = len(a), len(b)
    pooled = ((n1 - 1) * s1 ** 2 + (n3 - 1) * s3 ** 2) / max(n1 + n3 - 2, 1)
    pooled_std = pooled ** 0.5
    if pooled_std == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled_std)


def compare_stages(df: pd.DataFrame, features: list[str]) -> list[FeatureCompare]:
    s1 = df[df["final_stage"] == 1]
    s3 = df[df["final_stage"] == 3]
    out: list[FeatureCompare] = []
    for feat in features:
        a = s1[feat].dropna().to_numpy("float64")
        b = s3[feat].dropna().to_numpy("float64")
        if len(a) == 0 or len(b) == 0:
            continue
        all_mean = float(df[feat].mean())
        denom = abs(all_mean) if abs(all_mean) > 1e-9 else 1.0
        out.append(FeatureCompare(
            feature=feat,
            n1=len(a), n3=len(b),
            mean_s1=float(a.mean()), mean_s3=float(b.mean()),
            median_s1=float(np.median(a)), median_s3=float(np.median(b)),
            std_s1=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            std_s3=float(b.std(ddof=1)) if len(b) > 1 else 0.0,
            cohens_d=_cohens_d(a, b),
            diff_pct=100.0 * (b.mean() - a.mean()) / denom,
        ))
    out.sort(key=lambda x: abs(x.cohens_d), reverse=True)
    return out


# --------------------------------------------------------------------------- #
# 單特徵閾值掃描
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ThresholdResult:
    feature: str
    direction: str            # "ge" (≥ threshold) 或 "le" (≤ threshold)
    threshold: float
    n_kept: int
    n_total: int
    kept_stage3_pct: float
    baseline_stage3_pct: float
    lift: float               # kept_stage3_pct / baseline
    stage1_kept: int
    stage1_total: int
    stage3_kept: int
    stage3_total: int
    avg_R_kept: float


def scan_thresholds(
    df: pd.DataFrame, *, feature: str, direction: str = "ge", n_steps: int = 25,
) -> list[ThresholdResult]:
    """從 5%-95% 分位掃 n_steps 個閾值，每個閾值算「過濾後」的指標。"""
    series = df[feature].dropna()
    if len(series) < 50:
        return []
    qs = np.linspace(0.05, 0.95, n_steps)
    thresholds = series.quantile(qs).unique()
    baseline_s3 = (df["final_stage"] == 3).mean() * 100
    s1_total = int((df["final_stage"] == 1).sum())
    s3_total = int((df["final_stage"] == 3).sum())

    results: list[ThresholdResult] = []
    for thr in thresholds:
        if direction == "ge":
            kept = df[df[feature] >= thr]
        else:
            kept = df[df[feature] <= thr]
        if len(kept) < 30:
            continue
        s3_kept = int((kept["final_stage"] == 3).sum())
        s1_kept = int((kept["final_stage"] == 1).sum())
        s3_pct = (s3_kept / len(kept)) * 100
        results.append(ThresholdResult(
            feature=feature, direction=direction,
            threshold=float(thr),
            n_kept=len(kept), n_total=len(df),
            kept_stage3_pct=s3_pct, baseline_stage3_pct=baseline_s3,
            lift=s3_pct / baseline_s3 if baseline_s3 > 0 else 0.0,
            stage1_kept=s1_kept, stage1_total=s1_total,
            stage3_kept=s3_kept, stage3_total=s3_total,
            avg_R_kept=float(kept["r_multiple"].mean()),
        ))
    return results


def best_filters(
    df: pd.DataFrame, features: list[str], *, min_retention_pct: float = 50.0,
) -> list[ThresholdResult]:
    """每個特徵在『保留率 ≥ min_retention_pct%』約束下，找 lift 最大的閾值。

    同時試 ≥ 與 ≤ 兩種方向，取較好的。
    """
    out: list[ThresholdResult] = []
    n_total = len(df)
    min_kept = int(n_total * min_retention_pct / 100)
    for feat in features:
        candidates: list[ThresholdResult] = []
        for direction in ("ge", "le"):
            scan = scan_thresholds(df, feature=feat, direction=direction)
            for r in scan:
                if r.n_kept >= min_kept:
                    candidates.append(r)
        if candidates:
            best = max(candidates, key=lambda r: r.lift)
            out.append(best)
    out.sort(key=lambda r: r.lift, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# 印報表
# --------------------------------------------------------------------------- #

def render_compare(results: list[FeatureCompare]) -> str:
    lines = ["  特徵分布對比（按 |Cohen's d| 由大到小排序）"]
    lines.append("  " + "-" * 102)
    lines.append(f"  {'feature':<22} {'n1/n3':<9} {'mean_s1':>10} "
                 f"{'mean_s3':>10} {'med_s1':>10} {'med_s3':>10} "
                 f"{'cohen_d':>9} {'diff_%':>8}")
    lines.append("  " + "-" * 102)
    for r in results:
        lines.append(
            f"  {r.feature:<22} {r.n1}/{r.n3:<5} "
            f"{r.mean_s1:>+10.4f} {r.mean_s3:>+10.4f} "
            f"{r.median_s1:>+10.4f} {r.median_s3:>+10.4f} "
            f"{r.cohens_d:>+9.3f} {r.diff_pct:>+8.1f}"
        )
    return "\n".join(lines) + "\n"


def render_filters(results: list[ThresholdResult], *, retention_label: str) -> str:
    lines = [f"  最佳單特徵濾網（保留率 ≥ {retention_label}），按 lift 排序"]
    lines.append("  " + "-" * 110)
    lines.append(f"  {'feature':<22} {'op':<3} {'thr':>10} "
                 f"{'kept':>6} {'kept%':>7} {'s3_pct':>8} {'lift':>6} "
                 f"{'s1_drop':>9} {'s3_drop':>9} {'avg_R':>8}")
    lines.append("  " + "-" * 110)
    for r in results:
        kept_pct = 100.0 * r.n_kept / r.n_total
        s1_drop_pct = 100.0 * (r.stage1_total - r.stage1_kept) / max(r.stage1_total, 1)
        s3_drop_pct = 100.0 * (r.stage3_total - r.stage3_kept) / max(r.stage3_total, 1)
        op = "≥" if r.direction == "ge" else "≤"
        lines.append(
            f"  {r.feature:<22} {op:<3} {r.threshold:>+10.5f} "
            f"{r.n_kept:>6d} {kept_pct:>6.1f}% "
            f"{r.kept_stage3_pct:>7.2f}% {r.lift:>5.2f}x "
            f"{s1_drop_pct:>8.1f}% {s3_drop_pct:>8.1f}% "
            f"{r.avg_R_kept:>+8.3f}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

ALL_FEATURES = [
    "align_count", "align_body", "momentum_pct", "body_fullness",
    "sig_body_pct", "sig_body_size_pct", "sig_range_pct",
    "sig_body_to_range", "sig_close_in_bar", "sig_close_in_window",
    "sig_breakout", "avg_range_pct", "adverse_wick_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="壞訊號特徵分析（stage1 vs stage3）")
    parser.add_argument("paths", nargs="+", help="trades.csv / run dir")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--save", default=None)
    parser.add_argument("--per-direction", action="store_true",
                        help="也分別印 LONG-only / SHORT-only 對比")
    args = parser.parse_args()

    cfg = load_config(args.config)
    timeframe_delta = pd.Timedelta(
        cfg.timeframe.replace("H", "h").replace("M", "m")
    )

    print(f"[info] loading raw 1m parquet: {cfg.source_parquet}")
    df1m = load_ohlcv(cfg.source_parquet)
    print(f"[info] resampling 1m → {cfg.timeframe} ({len(df1m):,} bars)")
    raw_df = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m

    all_feats: list[pd.DataFrame] = []
    for raw in args.paths:
        path = _resolve_trades_csv(Path(raw))
        trades = load_trades(path)
        print(f"[info] processing {path} (trades={len(trades)})")
        feats = attach_features(
            trades, raw_df, window=args.window, timeframe_delta=timeframe_delta,
        )
        all_feats.append(feats)

    df = pd.concat(all_feats, ignore_index=True)
    if df.empty:
        print("no analyzable trades", file=sys.stderr)
        sys.exit(1)

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.save, index=False)
        print(f"[info] saved features → {args.save}")

    s1_n = int((df["final_stage"] == 1).sum())
    s3_n = int((df["final_stage"] == 3).sum())
    s2_n = int((df["final_stage"] == 2).sum())
    baseline = 100.0 * s3_n / len(df)

    bar = "=" * 110
    print()
    print(bar)
    print(f"  Bad-Signal Analysis  (window={args.window}, total={len(df)}, "
          f"stage1={s1_n}, stage2={s2_n}, stage3={s3_n}, baseline_s3={baseline:.2f}%)")
    print(bar)

    print()
    print(render_compare(compare_stages(df, ALL_FEATURES)))

    print(bar)
    print()
    print(render_filters(best_filters(df, ALL_FEATURES, min_retention_pct=70),
                         retention_label="70%"))
    print(render_filters(best_filters(df, ALL_FEATURES, min_retention_pct=50),
                         retention_label="50%"))
    print(render_filters(best_filters(df, ALL_FEATURES, min_retention_pct=30),
                         retention_label="30%"))

    if args.per_direction:
        for d in ("LONG", "SHORT"):
            sub = df[df["direction"] == d]
            sub_baseline = 100.0 * (sub["final_stage"] == 3).mean()
            print(bar)
            print(f"  [{d}-only]  n={len(sub)}, baseline_s3={sub_baseline:.2f}%")
            print(bar)
            print(render_compare(compare_stages(sub, ALL_FEATURES)))
            print(render_filters(
                best_filters(sub, ALL_FEATURES, min_retention_pct=50),
                retention_label="50%"))


if __name__ == "__main__":
    main()
