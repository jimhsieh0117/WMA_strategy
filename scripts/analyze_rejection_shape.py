"""K bar 拒絕形態分析（針對 t-1 / t / t+1 各別檢查）。

形態定義（以 LONG 為例）：
    range = high − low
    body_pct       = |close − open| / range          ← 實體
    upper_wick_pct = (high − max(open, close)) / range  ← 上影線（close 遠離 high）
    lower_wick_pct = (min(open, close) − low) / range   ← 下影線（close 遠離 low）

LONG 拒絕形態（趨勢上攻被打回）：
    upper_wick_pct ≥ wick_thr  AND  body_pct ≤ body_thr
SHORT 拒絕形態（鏡像）：
    lower_wick_pct ≥ wick_thr  AND  body_pct ≤ body_thr

針對三根 K 分別觀察：
    t-1：訊號 K 前一根
    t  ：訊號 K（= entry bar，金叉成立的那根）
    t+1：進場後第一根

並提供「t+1 任一條件命中」的整合視角。

用法：
    python scripts/analyze_rejection_shape.py [--config configs/default.yaml]
                                               [--sample is|oos]
                                               [--body-thr 0.40]
                                               [--out results/rejection_shape.csv]
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.utils.types import Direction  # noqa: E402

DEFAULT_PERIOD = PeriodSpec(
    start=pd.Timestamp("2023-01-01"),
    end=pd.Timestamp("2024-12-31"),
)

WICK_THRS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _bar_shape(df: pd.DataFrame) -> pd.DataFrame:
    rng = (df["high"] - df["low"]).astype(float)
    body_top = df[["open", "close"]].max(axis=1).astype(float)
    body_bot = df[["open", "close"]].min(axis=1).astype(float)
    body = (body_top - body_bot).clip(lower=0.0)
    upper = (df["high"].astype(float) - body_top).clip(lower=0.0)
    lower = (body_bot - df["low"].astype(float)).clip(lower=0.0)
    safe = rng.where(rng > 0, np.nan)
    return pd.DataFrame({
        "body_pct":  (body / safe).fillna(0.0),
        "upper_pct": (upper / safe).fillna(0.0),
        "lower_pct": (lower / safe).fillna(0.0),
    }, index=df.index)


def _per_trade_shape(
    trades: list,
    df: pd.DataFrame,
    shape: pd.DataFrame,
) -> pd.DataFrame:
    """每筆 trade 撈 t-1 / t / t+1 三根 K 的 body/upper/lower 比例。"""
    rows = []
    skipped = 0
    for tr in trades:
        entry_ts = pd.Timestamp(tr.entry_timestamp)
        if entry_ts in df.index:
            t_pos = df.index.get_loc(entry_ts)
        else:
            t_pos = df.index.searchsorted(entry_ts, side="left") - 1
        if t_pos < 1 or t_pos + 1 >= len(df.index):
            skipped += 1
            continue

        rec = {
            "direction": tr.direction.value,
            "entry_ts": entry_ts,
            "final_stage": int(tr.final_stage),
            "net_pnl": float(tr.net_pnl),
        }
        for offset, name in ((-1, "tm1"), (0, "t"), (1, "tp1")):
            r = shape.iloc[t_pos + offset]
            if tr.direction is Direction.LONG:
                rej_wick = float(r["upper_pct"])  # 上方拒絕
                sup_wick = float(r["lower_pct"])
            else:
                rej_wick = float(r["lower_pct"])  # 下方拒絕
                sup_wick = float(r["upper_pct"])
            rec[f"{name}_body"] = float(r["body_pct"])
            rec[f"{name}_rej_wick"] = rej_wick
            rec[f"{name}_sup_wick"] = sup_wick
        rows.append(rec)
    if skipped:
        print(f"  [info] skipped {skipped} trades (邊界 K 線不足)", file=sys.stderr)
    return pd.DataFrame(rows)


def _summarise_bar(
    df: pd.DataFrame,
    bar_label: str,
    body_thr: float,
    wick_thrs: list[float],
) -> pd.DataFrame:
    """對指定 K（tm1/t/tp1）做拒絕形態 sweep。"""
    body_col = f"{bar_label}_body"
    wick_col = f"{bar_label}_rej_wick"
    n_total = len(df)
    n_s1_total = int((df.final_stage == 1).sum())
    n_s3_total = int((df.final_stage == 3).sum())
    rows = []
    for thr in wick_thrs:
        mask = (df[wick_col] >= thr) & (df[body_col] <= body_thr)
        flagged = df[mask]
        kept = df[~mask]
        nf, nk = len(flagged), len(kept)
        rows.append({
            "wick≥": thr,
            "flag_n": nf,
            "flag%": nf / n_total * 100 if n_total else 0.0,
            "flag_s1%": (flagged.final_stage == 1).sum() / nf * 100 if nf else float("nan"),
            "flag_s2%": (flagged.final_stage == 2).sum() / nf * 100 if nf else float("nan"),
            "flag_s3%": (flagged.final_stage == 3).sum() / nf * 100 if nf else float("nan"),
            "flag_pnl": float(flagged["net_pnl"].sum()),
            "kept_pnl": float(kept["net_pnl"].sum()),
            "delta_pnl": -float(flagged["net_pnl"].sum()),  # 拒絕後省下的 PnL（+ 表示對總體有利）
            "s1_cut%": (flagged.final_stage == 1).sum() / n_s1_total * 100 if n_s1_total else float("nan"),
            "s3_cut%": (flagged.final_stage == 3).sum() / n_s3_total * 100 if n_s3_total else float("nan"),
        })
    out = pd.DataFrame(rows)
    title = f"{bar_label} 拒絕形態（body ≤ {body_thr:.2f}, rej_wick ≥ thr）"
    print(f"\n=== {title} ===")
    print(out.to_string(index=False, float_format="%.2f"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--timeframe", default=None, help="覆蓋 config 的 timeframe，如 5m / 1H")
    parser.add_argument("--body-thr", type=float, default=0.40)
    parser.add_argument("--out", default="results/rejection_shape.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    if args.timeframe is not None:
        overrides["timeframe"] = args.timeframe
    cfg_m = dataclasses.replace(cfg, **overrides)

    print(f"running long ...  ({cfg_m.symbol} {cfg_m.timeframe} "
          f"{cfg_m.in_sample.start.date()} ~ {cfg_m.in_sample.end.date()}, "
          f"body_thr={args.body_thr})")
    L = run_single_strategy(cfg_m, direction="long", sample=args.sample)
    print("running short ...")
    S = run_single_strategy(cfg_m, direction="short", sample=args.sample)

    period = cfg_m.in_sample if args.sample == "is" else cfg_m.out_of_sample
    raw1m = load_ohlcv(cfg_m.source_parquet, start=period.start, end=period.end)
    df_tf = resample(raw1m, cfg_m.timeframe) if cfg_m.timeframe != "1m" else raw1m
    shape = _bar_shape(df_tf)

    df = _per_trade_shape(L.trades + S.trades, df_tf, shape)
    n = len(df)
    n1 = int((df.final_stage == 1).sum())
    n2 = int((df.final_stage == 2).sum())
    n3 = int((df.final_stage == 3).sum())
    print(f"\ntotal trades: {n}  (s1={n1}/{n1/n*100:.1f}%, "
          f"s2={n2}/{n2/n*100:.1f}%, s3={n3}/{n3/n*100:.1f}%)  "
          f"total_pnl={df['net_pnl'].sum():+.2f}")
    print("""
欄位提示：
  flag_n / flag%   = 命中拒絕形態（rej_wick ≥ thr & body ≤ body_thr）的 trade 數 / 佔比
  flag_sX%         = 命中組的 stage X 比例（高 s1% = 預測 stop-out 越強）
  flag_pnl         = 命中組總 PnL（負值越多 = 拒絕越值得）
  delta_pnl        = 若用此 filter 拒絕命中組，省下的 PnL（+ 越大越好）
  s1_cut% / s3_cut%= 切掉幾 % 的 stage1 / stage3（理想：s1_cut 大、s3_cut 小）
""")

    for label in ("tm1", "t", "tp1"):
        _summarise_bar(df, label, args.body_thr, WICK_THRS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
