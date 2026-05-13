"""Stage × Market Structure 深度分析 HTML 報告（plotly）。

對 IS 期間每筆交易，記錄進場 signal bar 當下的：
- ``ms_trend``  ：當下結構方向（bull / bear / none）
- ``bars_since_event``：距離最近一次 BoS/CHoCH 事件的 K 線數（越小=剛突破）
- ``last_event_type``：最近一次事件（bos_up / bos_down / choch_up / choch_down / none）

輸出 HTML，內含 6 個互動圖表：
1. Stage × Structure trade count heatmap（per direction）
2. Aligned-structure 比例（per stage × direction）
3. PnL distribution box plot（per stage × structure）
4. Win rate heatmap
5. Bars-since-last-event histogram（per stage）
6. Aligned vs Counter 總 PnL 貢獻 pie

執行：
    python -m scripts.report_stage_market_structure
    python -m scripts.report_stage_market_structure --start 2023-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._runner import run_single_strategy  # noqa: E402
from src.data.loader import load_ohlcv  # noqa: E402
from src.data.resampler import resample  # noqa: E402
from src.indicators.market_structure import compute_market_structure  # noqa: E402
from src.utils.config import PeriodSpec, load_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Trade × Market Structure 關聯
# --------------------------------------------------------------------------- #

def build_trade_frame(
    trades: list, direction: str,
    ms: pd.DataFrame, tf_delta: pd.Timedelta,
) -> pd.DataFrame:
    """把 trade list 與市場結構欄位合併成扁平 DataFrame。"""
    if not trades:
        return pd.DataFrame()

    idx = ms.index
    ms_trend = ms["ms_trend"]
    ms_event = ms["ms_event"]

    # 預計算「最近一次事件的位置」 → 用 ffill 後對齊
    event_mask = ms_event.astype(str).str.len() > 0
    last_event_pos_series = pd.Series(
        np.where(event_mask, np.arange(len(ms)), np.nan),
        index=ms.index,
    ).ffill()
    last_event_type_series = ms_event.where(event_mask).ffill().fillna("")

    rows = []
    for t in trades:
        signal_ts = pd.Timestamp(t.entry_timestamp) - tf_delta
        pos = idx.searchsorted(signal_ts, side="right") - 1
        if pos < 0:
            continue
        trend = ms_trend.iloc[pos]
        if not isinstance(trend, str):
            trend = ""
        last_evt_pos = last_event_pos_series.iloc[pos]
        bars_since = (
            int(pos - last_evt_pos) if np.isfinite(last_evt_pos)
            else -1  # 無事件
        )
        last_evt = last_event_type_series.iloc[pos]

        aligned = (
            (direction == "long" and trend == "bull")
            or (direction == "short" and trend == "bear")
        )
        counter = (
            (direction == "long" and trend == "bear")
            or (direction == "short" and trend == "bull")
        )
        structure_label = (
            "aligned" if aligned else ("counter" if counter else "none")
        )

        rows.append({
            "direction": direction,
            "position_id": int(t.position_id),
            "entry_time": pd.Timestamp(t.entry_timestamp),
            "exit_time": pd.Timestamp(t.exit_timestamp),
            "final_stage": int(t.final_stage),
            "net_pnl": float(t.net_pnl),
            "return_pct": float(t.return_pct * 100),
            "exit_reason": t.exit_reason,
            "ms_trend": trend or "none",
            "structure": structure_label,
            "bars_since_event": bars_since,
            "last_event": last_evt or "none",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 視覺化
# --------------------------------------------------------------------------- #

COLOR_ALIGNED = "#16a34a"
COLOR_COUNTER = "#dc2626"
COLOR_NONE = "#9ca3af"
STRUCTURE_ORDER = ["aligned", "counter", "none"]
STAGE_ORDER = [1, 2, 3]


def fig_count_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("LONG", "SHORT"),
        horizontal_spacing=0.18,
    )
    for col, direction in enumerate(("long", "short"), start=1):
        sub = df[df["direction"] == direction]
        mat = (
            sub.groupby(["final_stage", "structure"]).size().unstack(
                fill_value=0,
            ).reindex(index=STAGE_ORDER, columns=STRUCTURE_ORDER, fill_value=0)
        )
        z = mat.values
        text = [[str(v) for v in row] for row in z]
        fig.add_trace(
            go.Heatmap(
                z=z, x=mat.columns.tolist(),
                y=[f"Stage {s}" for s in mat.index],
                text=text, texttemplate="%{text}",
                textfont={"size": 14},
                colorscale="Blues",
                showscale=(col == 2),
                hovertemplate=(
                    "stage=%{y}<br>structure=%{x}<br>count=%{z}<extra></extra>"
                ),
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title="① Trade Count: Stage × Structure",
        height=380, template="plotly_dark", margin=dict(t=70, b=40),
    )
    return fig


def fig_aligned_ratio(df: pd.DataFrame) -> go.Figure:
    rows = []
    for direction in ("long", "short"):
        for stage in STAGE_ORDER:
            sub = df[(df["direction"] == direction) & (df["final_stage"] == stage)]
            n = len(sub)
            if n == 0:
                continue
            aligned = (sub["structure"] == "aligned").sum()
            counter = (sub["structure"] == "counter").sum()
            none = (sub["structure"] == "none").sum()
            rows.append({
                "direction": direction, "stage": f"Stage {stage}",
                "aligned%": aligned / n * 100,
                "counter%": counter / n * 100,
                "none%": none / n * 100,
                "n": n,
            })
    plot_df = pd.DataFrame(rows)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("LONG", "SHORT"),
        horizontal_spacing=0.12,
    )
    for col, direction in enumerate(("long", "short"), start=1):
        sub = plot_df[plot_df["direction"] == direction]
        for struct, color in (
            ("aligned%", COLOR_ALIGNED),
            ("counter%", COLOR_COUNTER),
            ("none%", COLOR_NONE),
        ):
            fig.add_trace(
                go.Bar(
                    x=sub["stage"], y=sub[struct],
                    name=struct.replace("%", ""),
                    marker_color=color,
                    showlegend=(col == 1),
                    text=[f"{v:.1f}%" for v in sub[struct]],
                    textposition="inside",
                    hovertemplate=(
                        f"{struct}: %{{y:.1f}}%<br>n=%{{customdata}}<extra></extra>"
                    ),
                    customdata=sub["n"],
                ),
                row=1, col=col,
            )
    fig.update_layout(
        title=(
            "② Aligned vs Counter Structure Ratio by Stage<br>"
            "<sub>越往 stage 3，aligned% 應越高（趨勢進入正確結構）</sub>"
        ),
        barmode="stack",
        height=420, template="plotly_dark", margin=dict(t=90, b=40),
        yaxis=dict(range=[0, 100], title="ratio (%)"),
        yaxis2=dict(range=[0, 100]),
    )
    return fig


def fig_pnl_box(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("LONG", "SHORT"),
        horizontal_spacing=0.10,
    )
    for col, direction in enumerate(("long", "short"), start=1):
        sub = df[df["direction"] == direction]
        for struct, color in (
            ("aligned", COLOR_ALIGNED),
            ("counter", COLOR_COUNTER),
            ("none", COLOR_NONE),
        ):
            for stage in STAGE_ORDER:
                slc = sub[
                    (sub["structure"] == struct) & (sub["final_stage"] == stage)
                ]
                if len(slc) == 0:
                    continue
                fig.add_trace(
                    go.Box(
                        y=slc["net_pnl"], name=f"S{stage}-{struct}",
                        marker_color=color, boxmean=True,
                        x=[f"Stage {stage}"] * len(slc),
                        offsetgroup=struct,
                        legendgroup=struct,
                        showlegend=(col == 1 and stage == 1),
                    ),
                    row=1, col=col,
                )
    fig.update_layout(
        title=(
            "③ Net PnL Distribution: Stage × Structure<br>"
            "<sub>aligned 結構的 stage 3 應該有最厚的右尾</sub>"
        ),
        boxmode="group",
        height=480, template="plotly_dark", margin=dict(t=90, b=40),
        yaxis_title="net PnL (USDT)",
    )
    return fig


def fig_winrate_heatmap(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("LONG", "SHORT"),
        horizontal_spacing=0.18,
    )
    for col, direction in enumerate(("long", "short"), start=1):
        sub = df[df["direction"] == direction]
        rows = []
        for stage in STAGE_ORDER:
            row = []
            for struct in STRUCTURE_ORDER:
                slc = sub[
                    (sub["structure"] == struct) & (sub["final_stage"] == stage)
                ]
                if len(slc) == 0:
                    row.append(np.nan)
                else:
                    row.append((slc["net_pnl"] > 0).mean() * 100)
            rows.append(row)
        z = np.array(rows)
        text = [
            [f"{v:.0f}%" if np.isfinite(v) else "" for v in row]
            for row in z
        ]
        fig.add_trace(
            go.Heatmap(
                z=z, x=STRUCTURE_ORDER,
                y=[f"Stage {s}" for s in STAGE_ORDER],
                text=text, texttemplate="%{text}",
                textfont={"size": 14},
                colorscale="RdYlGn", zmin=0, zmax=100,
                showscale=(col == 2),
                hovertemplate=(
                    "stage=%{y}<br>structure=%{x}<br>win%=%{z:.1f}<extra></extra>"
                ),
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title="④ Win Rate (%): Stage × Structure",
        height=380, template="plotly_dark", margin=dict(t=70, b=40),
    )
    return fig


def fig_bars_since_event(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Stage {s}" for s in STAGE_ORDER],
    )
    for col, stage in enumerate(STAGE_ORDER, start=1):
        sub = df[df["final_stage"] == stage]
        # 截斷在 100 根之內以利顯示
        vals = sub["bars_since_event"].clip(upper=100)
        fig.add_trace(
            go.Histogram(
                x=vals, nbinsx=20,
                marker_color="#3b82f6",
                hovertemplate="bars=%{x}<br>count=%{y}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title=(
            "⑤ Bars Since Last BoS/CHoCH at Entry<br>"
            "<sub>左偏=剛突破進場；右偏=老 trend 進場（clip @ 100 bars）</sub>"
        ),
        height=380, template="plotly_dark", margin=dict(t=90, b=40),
        xaxis_title="bars since event",
        yaxis_title="trade count",
    )
    return fig


def fig_pnl_contribution(df: pd.DataFrame) -> go.Figure:
    """Aligned / Counter / None 各對總 PnL 貢獻多少 USDT。"""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("LONG total PnL", "SHORT total PnL"),
    )
    for col, direction in enumerate(("long", "short"), start=1):
        sub = df[df["direction"] == direction]
        sums = sub.groupby("structure")["net_pnl"].sum().reindex(
            STRUCTURE_ORDER, fill_value=0,
        )
        # 對於負總和的部分，pie 不能直接顯示負值——用絕對值繪 + 文字標示符號
        labels = []
        values = []
        colors = []
        for struct in STRUCTURE_ORDER:
            v = sums[struct]
            labels.append(f"{struct} ({v:+.1f})")
            values.append(abs(v))
            colors.append({
                "aligned": COLOR_ALIGNED,
                "counter": COLOR_COUNTER,
                "none": COLOR_NONE,
            }[struct])
        fig.add_trace(
            go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hole=0.45,
                sort=False,
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title=(
            "⑥ Total PnL Contribution by Structure<br>"
            "<sub>標籤含實際 PnL (USDT)，餅圖大小用 |PnL|；理想是 aligned 主導</sub>"
        ),
        height=420, template="plotly_dark", margin=dict(t=90, b=40),
    )
    return fig


# --------------------------------------------------------------------------- #
# 摘要表 HTML
# --------------------------------------------------------------------------- #

def summary_table_html(df: pd.DataFrame) -> str:
    rows = []
    for direction in ("long", "short"):
        for stage in STAGE_ORDER:
            sub = df[(df["direction"] == direction) & (df["final_stage"] == stage)]
            n_total = len(sub)
            if n_total == 0:
                continue
            for struct in STRUCTURE_ORDER:
                slc = sub[sub["structure"] == struct]
                n = len(slc)
                if n == 0:
                    continue
                rows.append({
                    "direction": direction,
                    "stage": stage,
                    "structure": struct,
                    "n": n,
                    "pct_of_stage": round(n / n_total * 100, 1),
                    "pnl_sum": round(slc["net_pnl"].sum(), 2),
                    "pnl_mean": round(slc["net_pnl"].mean(), 3),
                    "win_rate%": round((slc["net_pnl"] > 0).mean() * 100, 1),
                    "bars_since_event_median": int(slc["bars_since_event"].median()),
                })
    summary = pd.DataFrame(rows)
    return summary.to_html(
        index=False, classes="summary-table",
        float_format=lambda x: f"{x:.2f}",
    )


# --------------------------------------------------------------------------- #
# 主程式
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage × Market Structure HTML 報告。"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--pivot-left", type=int, default=10)
    parser.add_argument("--pivot-right", type=int, default=10)
    parser.add_argument(
        "--out", default=None,
        help="輸出 HTML 路徑；不指定則依是否 --no-filters 自動命名",
    )
    parser.add_argument(
        "--no-filters", action="store_true",
        help="關閉 signal_filter（mode=off）與 chop_filter（enabled=false）"
             "→ 看「原始 WMA 訊號」的結構分布（過濾前對照組）",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    period = PeriodSpec(start=start_ts, end=end_ts)
    cfg = dataclasses.replace(
        base_cfg, in_sample=period, show_progress=False, log_level="WARNING",
    )

    # 視 --no-filters 覆寫兩支 filter 設定
    if args.no_filters:
        cfg = dataclasses.replace(
            cfg,
            signal_filter=dataclasses.replace(cfg.signal_filter, mode="off"),
            chop_filter=dataclasses.replace(cfg.chop_filter, enabled=False),
        )
        filter_tag = "PRE-FILTER (raw WMA signal)"
        default_out = "results/stage_market_structure/report_no_filters.html"
    else:
        filter_tag = "POST-FILTER (signal_filter + chop_filter applied)"
        default_out = "results/stage_market_structure/report.html"

    out_path_str = args.out or default_out

    print(f"Period: {start_ts.date()} ~ {end_ts.date()}  tf: {cfg.timeframe}")
    print(f"Pivot: {args.pivot_left}/{args.pivot_right}  symbol: {cfg.symbol}")
    print(f"Filters: {filter_tag}")

    print("Running long backtest ...")
    L = run_single_strategy(cfg, direction="long", sample="is")
    print(f"  long trades: {len(L.trades)}")
    print("Running short backtest ...")
    S = run_single_strategy(cfg, direction="short", sample="is")
    print(f"  short trades: {len(S.trades)}")

    print("Computing market structure ...")
    df1m = load_ohlcv(cfg.source_parquet, start=period.start, end=period.end)
    df_tf = resample(df1m, cfg.timeframe) if cfg.timeframe != "1m" else df1m
    ms = compute_market_structure(
        df_tf, pivot_left=args.pivot_left, pivot_right=args.pivot_right,
    )
    tf_delta = df_tf.index[1] - df_tf.index[0]

    df_long = build_trade_frame(L.trades, "long", ms, tf_delta)
    df_short = build_trade_frame(S.trades, "short", ms, tf_delta)
    df_all = pd.concat([df_long, df_short], ignore_index=True)
    print(f"Joined trades: {len(df_all)}")

    # ----- 建圖 -----
    figs = [
        fig_count_heatmap(df_all),
        fig_aligned_ratio(df_all),
        fig_pnl_box(df_all),
        fig_winrate_heatmap(df_all),
        fig_bars_since_event(df_all),
        fig_pnl_contribution(df_all),
    ]

    summary_html = summary_table_html(df_all)

    # ----- 組 HTML -----
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body_parts = [
        "<!DOCTYPE html><html lang='zh-Hant'><head>",
        "<meta charset='UTF-8'>",
        "<title>Stage × Market Structure Report</title>",
        "<script src='https://cdn.plot.ly/plotly-3.0.0.min.js'></script>",
        """<style>
        body { background:#0a0a0a; color:#e5e5e5; font-family:-apple-system,
               BlinkMacSystemFont,'PingFang TC',sans-serif;
               margin:0; padding:24px; max-width:1400px; margin:auto; }
        h1 { color:#fafafa; font-size:22px; border-bottom:1px solid #333;
             padding-bottom:10px; }
        h2 { color:#fafafa; font-size:16px; margin-top:32px; }
        .meta { background:#1a1a1a; padding:12px 16px; border-radius:6px;
                font-size:13px; color:#aaa; margin:16px 0; }
        .meta b { color:#fafafa; }
        .summary-table { border-collapse:collapse; width:100%;
                         font-size:12px; margin-top:14px; }
        .summary-table th, .summary-table td {
            border:1px solid #2a2a2a; padding:6px 10px; text-align:right; }
        .summary-table th { background:#1f1f1f; color:#fafafa; }
        .summary-table td:first-child, .summary-table td:nth-child(2),
        .summary-table td:nth-child(3) { text-align:left; }
        .summary-table tr:hover { background:#161616; }
        .fig-wrap { background:#0a0a0a; margin:16px 0; }
        </style></head><body>""",
        f"<h1>Stage × Market Structure 深度分析報告</h1>",
        "<div class='meta'>",
        f"<b>Filters</b>: <span style='color:#fbbf24'>{filter_tag}</span><br>",
        f"<b>Symbol</b>: {cfg.symbol} &nbsp;|&nbsp; ",
        f"<b>Timeframe</b>: {cfg.timeframe} &nbsp;|&nbsp; ",
        f"<b>Period</b>: {start_ts.date()} ~ {end_ts.date()} (IS) &nbsp;|&nbsp; ",
        f"<b>Pivot</b>: left={args.pivot_left}, right={args.pivot_right} &nbsp;|&nbsp; ",
        f"<b>Long trades</b>: {len(df_long)} &nbsp;|&nbsp; ",
        f"<b>Short trades</b>: {len(df_short)}",
        "</div>",
        "<h2>視覺化</h2>",
    ]

    for fig in figs:
        body_parts.append("<div class='fig-wrap'>")
        body_parts.append(
            fig.to_html(
                include_plotlyjs=False, full_html=False,
                default_height=str(fig.layout.height or 400) + "px",
            ),
        )
        body_parts.append("</div>")

    body_parts.append("<h2>明細摘要表</h2>")
    body_parts.append(summary_html)

    body_parts.append("</body></html>")

    out_path.write_text("\n".join(body_parts), encoding="utf-8")
    print(f"\nHTML 報告：{out_path.resolve()}")
    print(f"用瀏覽器開啟：file://{out_path.resolve()}")


if __name__ == "__main__":
    main()
