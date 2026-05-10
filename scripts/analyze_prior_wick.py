"""進場前 N 根 K 的影線（wick）分析。

問題：進場前 4 根 K 出現「反向大影線」的訂單，stage 1/2/3 分布如何？
- LONG 訂單 → 反向 = 上影線（賣壓拒絕）；同向 = 下影線（買壓支撐）
- SHORT 訂單 → 反向 = 下影線（買壓支撐）；同向 = 上影線（賣壓拒絕）

影線比例定義：
    upper_wick = high − max(open, close)
    lower_wick = min(open, close) − low
    range = high − low（range == 0 視為無影線）
    ratio = wick / range

掃 threshold = 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
標籤條件：「進場前 N 根 K 中**至少一根**該方向影線 ratio ≥ threshold」

用法：
    python scripts/analyze_prior_wick.py [--config configs/default.yaml]
                                          [--sample is|oos]
                                          [--lookback 4]
                                          [--out results/prior_wick.csv]
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

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _wick_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """逐根 K 計算 upper / lower wick 佔整根 K range 的比例。range==0 視為 0。"""
    rng = (df["high"] - df["low"]).astype(float)
    body_top = df[["open", "close"]].max(axis=1).astype(float)
    body_bot = df[["open", "close"]].min(axis=1).astype(float)
    upper = (df["high"].astype(float) - body_top).clip(lower=0.0)
    lower = (body_bot - df["low"].astype(float)).clip(lower=0.0)
    safe_rng = rng.where(rng > 0, np.nan)
    out = pd.DataFrame({
        "upper_ratio": (upper / safe_rng).fillna(0.0),
        "lower_ratio": (lower / safe_rng).fillna(0.0),
    }, index=df.index)
    return out


def _per_trade_max_wick(
    trades: list,
    df: pd.DataFrame,
    wick: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """每筆 trade：抓進場前 N 根 K 的 max(反向影線比例) 與 max(同向影線比例)。"""
    rows = []
    skipped = 0
    for t in trades:
        entry_ts = pd.Timestamp(t.entry_timestamp)
        # 找進場 K 在 df 中的位置；prior_n bars 為 [pos − N, pos)
        pos = df.index.searchsorted(entry_ts, side="left")
        if pos == 0:
            skipped += 1
            continue
        # 對齊：若 entry_ts 不在 df.index，pos 指向第一個 > entry_ts；前一根才是 entry K
        if pos < len(df.index) and df.index[pos] != entry_ts:
            pass  # entry 不在 K 線上 → pos 即為 entry K「後」位置；prior_n = [pos − N, pos)
        start = max(0, pos - lookback)
        if start >= pos:
            skipped += 1
            continue
        window = wick.iloc[start:pos]
        if window.empty:
            skipped += 1
            continue
        max_upper = float(window["upper_ratio"].max())
        max_lower = float(window["lower_ratio"].max())
        if t.direction is Direction.LONG:
            opp_max, same_max = max_upper, max_lower
        else:
            opp_max, same_max = max_lower, max_upper
        rows.append({
            "direction": t.direction.value,
            "entry_ts": entry_ts,
            "final_stage": int(t.final_stage),
            "net_pnl": float(t.net_pnl),
            "opp_wick_max": opp_max,
            "same_wick_max": same_max,
        })
    if skipped:
        print(f"  [info] skipped {skipped} trades (insufficient lookback)", file=sys.stderr)
    return pd.DataFrame(rows)


def _summarise_by_threshold(
    df: pd.DataFrame, col: str, thresholds: list[float], label: str,
) -> pd.DataFrame:
    """對每個 threshold：分「至少一根 ≥ thr」與「全部 < thr」兩組，列出 s1/s2/s3 分布。"""
    n_total = len(df)
    n_s1_total = int((df.final_stage == 1).sum())
    n_s3_total = int((df.final_stage == 3).sum())
    rows = []
    for thr in thresholds:
        mask = df[col] >= thr
        flagged = df[mask]
        kept = df[~mask]
        nf = len(flagged)
        nk = len(kept)
        s1f = int((flagged.final_stage == 1).sum())
        s2f = int((flagged.final_stage == 2).sum())
        s3f = int((flagged.final_stage == 3).sum())
        s1k = int((kept.final_stage == 1).sum())
        s2k = int((kept.final_stage == 2).sum())
        s3k = int((kept.final_stage == 3).sum())
        rows.append({
            "thr": thr,
            "flag_n": nf,
            "flag%": nf / n_total * 100 if n_total else 0.0,
            "flag_s1%": s1f / nf * 100 if nf else float("nan"),
            "flag_s2%": s2f / nf * 100 if nf else float("nan"),
            "flag_s3%": s3f / nf * 100 if nf else float("nan"),
            "flag_pnl": float(flagged["net_pnl"].sum()),
            "kept_n": nk,
            "kept_s1%": s1k / nk * 100 if nk else float("nan"),
            "kept_s2%": s2k / nk * 100 if nk else float("nan"),
            "kept_s3%": s3k / nk * 100 if nk else float("nan"),
            "kept_pnl": float(kept["net_pnl"].sum()),
            "s1_cut%": s1f / n_s1_total * 100 if n_s1_total else float("nan"),
            "s3_cut%": s3f / n_s3_total * 100 if n_s3_total else float("nan"),
        })
    out = pd.DataFrame(rows)
    print(f"\n=== {label} ===")
    print(out.to_string(index=False, float_format="%.2f"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sample", choices=["is", "oos"], default="is")
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--out", default="results/prior_wick.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    overrides = {"show_progress": False, "log_level": "WARNING"}
    if args.sample == "is":
        overrides["in_sample"] = DEFAULT_PERIOD
    cfg_m = dataclasses.replace(cfg, **overrides)

    print(f"running long ...  ({cfg_m.symbol} {cfg_m.timeframe} "
          f"{cfg_m.in_sample.start.date()} ~ {cfg_m.in_sample.end.date()}, "
          f"lookback={args.lookback} bars)")
    L = run_single_strategy(cfg_m, direction="long", sample=args.sample)
    print("running short ...")
    S = run_single_strategy(cfg_m, direction="short", sample=args.sample)

    period = cfg_m.in_sample if args.sample == "is" else cfg_m.out_of_sample
    raw1m = load_ohlcv(cfg_m.source_parquet, start=period.start, end=period.end)
    df_tf = resample(raw1m, cfg_m.timeframe) if cfg_m.timeframe != "1m" else raw1m
    wick = _wick_ratios(df_tf)

    df = _per_trade_max_wick(L.trades + S.trades, df_tf, wick, args.lookback)
    n = len(df)
    n1 = int((df.final_stage == 1).sum())
    n2 = int((df.final_stage == 2).sum())
    n3 = int((df.final_stage == 3).sum())
    pnl_total = float(df["net_pnl"].sum())
    print(f"\ntotal trades analysed: {n}  (s1={n1}/{n1/n*100:.1f}%, "
          f"s2={n2}/{n2/n*100:.1f}%, s3={n3}/{n3/n*100:.1f}%)  total_pnl={pnl_total:+.2f}")

    print("""
欄位說明：
  flag_n / flag%   = 進場前 N 根 K 至少一根「該方向」影線 ≥ thr 的 trade 數 / 佔比
  flag_sX%         = 被 flag 的 trade 中 stage X 比例（高 s1% = 預測 stop-out 強）
  flag_pnl         = 被 flag 的 trade 總 PnL（負值越多 = 越值得拒絕）
  kept_*           = 沒被 flag 的對照組
  s1_cut% / s3_cut%= 若用此 thr 當 filter，會切掉幾 % 的 stage1 / stage3
                     （理想：s1_cut 高 & s3_cut 低）
""")

    _summarise_by_threshold(df, "opp_wick_max", THRESHOLDS,
                            f"反向影線（LONG=上影 / SHORT=下影），lookback={args.lookback}")
    _summarise_by_threshold(df, "same_wick_max", THRESHOLDS,
                            f"同向影線（LONG=下影 / SHORT=上影），lookback={args.lookback}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n💾 written → {out_path}")


if __name__ == "__main__":
    main()
