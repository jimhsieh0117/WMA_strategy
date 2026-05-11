"""盤整過濾器**雙條件**疊加 sweep。

延續 sweep_chop_filters.py 的單條件結果，這支腳本對 BBW_rank × ATR_rank
（必要時也納入 ADX）做 AND 疊加，看雙條件是否能把 MC P(PF>1) 推上去。

過濾規則：keep 當且僅當所有指標都通過閾值。

用法：
    python scripts/sweep_chop_filters_combo.py
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
from scripts.sweep_chop_filters import (  # noqa: E402
    _compute_filter_cache, _filter_lookup, _trade_metrics,
)
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402
from src.validation.monte_carlo import MCConfig, bootstrap  # noqa: E402


# 候選組合：(BBW_rank threshold, ATR_rank threshold)，皆為「>= thr 才保留」
PAIRS_BBW_ATR = [
    (20, 20), (20, 30), (20, 40),
    (30, 20), (30, 30), (30, 40),
    (40, 20), (40, 30), (40, 40),
]

# 三條件：BBW + ATR + ADX (>= adx_thr 才保留)
TRIPLES = [
    (30, 30, 20),
    (40, 30, 20),
    (40, 40, 20),
    (40, 40, 25),
]


def _bootstrap_pf_ci(trades, mc_cfg) -> tuple[float, float, float, float]:
    res = bootstrap(trades, mc_cfg)
    pf = res.profit_factor
    pf = pf[np.isfinite(pf)]
    return (float(np.percentile(pf, 5)),
            float(np.percentile(pf, 50)),
            float(np.percentile(pf, 95)),
            float((pf > 1.0).mean() * 100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--wma-fast", type=int, default=4)
    parser.add_argument("--wma-slow", type=int, default=6)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--mc-n", type=int, default=5000)
    parser.add_argument("--out", default="results/chop_filter_combo.csv")
    args = parser.parse_args()

    logging.basicConfig(level="WARNING", format="%(message)s")
    cfg = load_config(args.config)
    cfg_m = dataclasses.replace(
        cfg,
        show_progress=False, log_level="WARNING",
        timeframe=args.timeframe,
        wma_fast=args.wma_fast, wma_slow=args.wma_slow,
        in_sample=PeriodSpec(start=pd.Timestamp(args.start),
                             end=pd.Timestamp(args.end)),
    )
    tf_delta = pd.Timedelta(args.timeframe)

    print(f">>> backtest {cfg_m.symbol} {args.timeframe} W{args.wma_fast}-{args.wma_slow}  "
          f"{args.start} ~ {args.end}", flush=True)
    L = run_single_strategy(cfg_m, direction="long", sample="is")
    S = run_single_strategy(cfg_m, direction="short", sample="is")
    trades = L.trades + S.trades
    print(f"  total trades: {len(trades)}")

    df1m = load_ohlcv(cfg_m.source_parquet, start=cfg_m.in_sample.start,
                      end=cfg_m.in_sample.end)
    df_tf = resample(df1m, args.timeframe)
    cache = _compute_filter_cache(df_tf)

    pnls_all = np.array([t.net_pnl for t in trades])
    fv = {col: _filter_lookup(trades, cache[col], tf_delta)
          for col in ["adx", "atr_rank", "bbw_rank"]}

    mc_cfg = MCConfig(n_simulations=args.mc_n, initial_capital=1000.0,
                      ruin_threshold_pct=50.0, seed=42)

    base = _trade_metrics(pnls_all)
    bp = _bootstrap_pf_ci(trades, mc_cfg)
    print(f"\n[baseline] n={base['n']}  win={base['win%']:.2f}%  "
          f"pnl={base['pnl']:+.2f}  PF={base['pf']:.3f}  "
          f"MC[{bp[0]:.3f}, {bp[1]:.3f}, {bp[2]:.3f}]  P(PF>1)={bp[3]:.2f}%")

    rows = [{
        "type": "baseline", "bbw": np.nan, "atr": np.nan, "adx": np.nan,
        "n_kept": base["n"], "kept%": 100.0,
        "win%": base["win%"], "pnl": base["pnl"], "pf": base["pf"],
        "mc_p05": bp[0], "mc_p50": bp[1], "mc_p95": bp[2], "p_pf_gt1": bp[3],
    }]

    print("\n=== BBW_rank × ATR_rank 雙條件 (>= thr 才保留) ===")
    for bbw_thr, atr_thr in PAIRS_BBW_ATR:
        keep = ((fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
                (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]))
        kt = [t for t, k in zip(trades, keep) if k]
        m = _trade_metrics(pnls_all[keep])
        if m["n"] < 50:
            print(f"  BBW>={bbw_thr} & ATR>={atr_thr}  n={m['n']} too few")
            continue
        p = _bootstrap_pf_ci(kt, mc_cfg)
        print(f"  BBW>={bbw_thr} & ATR>={atr_thr}  n={m['n']:4d} "
              f"({m['n']/base['n']*100:5.1f}%)  win={m['win%']:5.2f}%  "
              f"pnl={m['pnl']:+7.2f}  PF={m['pf']:.3f}  "
              f"MC[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  P(PF>1)={p[3]:5.2f}%")
        rows.append({
            "type": "BBW+ATR", "bbw": bbw_thr, "atr": atr_thr, "adx": np.nan,
            "n_kept": m["n"], "kept%": m["n"] / base["n"] * 100,
            "win%": m["win%"], "pnl": m["pnl"], "pf": m["pf"],
            "mc_p05": p[0], "mc_p50": p[1], "mc_p95": p[2], "p_pf_gt1": p[3],
        })

    print("\n=== BBW_rank × ATR_rank × ADX 三條件 ===")
    for bbw_thr, atr_thr, adx_thr in TRIPLES:
        keep = ((fv["bbw_rank"] >= bbw_thr) & np.isfinite(fv["bbw_rank"]) &
                (fv["atr_rank"] >= atr_thr) & np.isfinite(fv["atr_rank"]) &
                (fv["adx"] >= adx_thr) & np.isfinite(fv["adx"]))
        kt = [t for t, k in zip(trades, keep) if k]
        m = _trade_metrics(pnls_all[keep])
        if m["n"] < 50:
            print(f"  BBW>={bbw_thr} & ATR>={atr_thr} & ADX>={adx_thr}  n={m['n']} too few")
            continue
        p = _bootstrap_pf_ci(kt, mc_cfg)
        print(f"  BBW>={bbw_thr} & ATR>={atr_thr} & ADX>={adx_thr}  n={m['n']:4d} "
              f"({m['n']/base['n']*100:5.1f}%)  win={m['win%']:5.2f}%  "
              f"pnl={m['pnl']:+7.2f}  PF={m['pf']:.3f}  "
              f"MC[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]  P(PF>1)={p[3]:5.2f}%")
        rows.append({
            "type": "BBW+ATR+ADX", "bbw": bbw_thr, "atr": atr_thr, "adx": adx_thr,
            "n_kept": m["n"], "kept%": m["n"] / base["n"] * 100,
            "win%": m["win%"], "pnl": m["pnl"], "pf": m["pf"],
            "mc_p05": p[0], "mc_p50": p[1], "mc_p95": p[2], "p_pf_gt1": p[3],
        })

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values("p_pf_gt1", ascending=False)
    print("\n" + "=" * 100)
    print("=== 依 P(PF>1) 排序 (top 8) ===")
    print("=" * 100)
    print(df_sorted.head(8).to_string(index=False, float_format="%.3f"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n💾 → {out_path}")


if __name__ == "__main__":
    main()
