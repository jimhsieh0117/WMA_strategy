"""合併多個獨立帳戶的權益曲線與交易紀錄。

用於「多空策略各跑一個帳戶，最後合併資金曲線」的情境（ARCHITECTURE.md §9.7 選項 B）。

合併語意：
- 兩個帳戶完全獨立累積 PnL，不在中途重新分配資金
- 合併權益曲線 = 各帳戶 equity 之和（時間軸對齊後）
- 合併 trade log = 各帳戶 trades 串接，按 entry_timestamp 排序
- 合併 metrics 由 ``metrics.calculator.compute_metrics`` 直接消費合併後的 BacktestResult
"""

from __future__ import annotations

import pandas as pd

from src.backtest.types import BacktestResult


def merge_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """合併多條 equity 曲線為單條（element-wise sum，缺值 ffill）。

    Args:
        curves: 至少一條 ``pd.Series``（DatetimeIndex）。

    Returns:
        合併後的 Series，index 為所有輸入 index 的 outer-union。

    Raises:
        ValueError: ``curves`` 為空。
    """
    if not curves:
        raise ValueError("merge_equity_curves received empty list")

    if len(curves) == 1:
        return curves[0].copy()

    # 大宗情形：所有 index 相等 → 直接相加（最快）
    first_idx = curves[0].index
    if all(c.index.equals(first_idx) for c in curves[1:]):
        merged = curves[0].copy()
        for c in curves[1:]:
            merged = merged + c
        merged.name = "merged_equity"
        return merged

    # 不同 index → outer join + ffill；leading NaN 用各曲線首值補
    df = pd.concat(
        {f"_c{i}": c for i, c in enumerate(curves)},
        axis=1,
    ).sort_index()
    df = df.ffill()
    for i, c in enumerate(curves):
        col = f"_c{i}"
        df[col] = df[col].fillna(c.iloc[0])

    return df.sum(axis=1).rename("merged_equity")


def build_merged_result(
    name: str,
    component_results: list[BacktestResult],
) -> BacktestResult:
    """將多個 ``BacktestResult`` 合併為一個（用於整體績效計算）。

    - 權益曲線：``merge_equity_curves``
    - 交易紀錄：串接後按 entry_timestamp 排序
    - 訊號統計：對應欄位相加
    - initial_capital：相加（兩個獨立帳戶總資金）

    config_snapshot 收進 ``components`` 子欄供日後追溯。
    """
    if not component_results:
        raise ValueError("build_merged_result received empty list")

    merged_curve = merge_equity_curves(
        [r.equity_curve for r in component_results]
    )

    all_trades = [t for r in component_results for t in r.trades]
    all_trades.sort(key=lambda t: t.entry_timestamp)

    total_initial = sum(r.initial_capital for r in component_results)
    final_equity = (
        float(merged_curve.iloc[-1]) if len(merged_curve) > 0 else total_initial
    )

    return BacktestResult(
        account_name=name,
        initial_capital=total_initial,
        final_equity=final_equity,
        trades=all_trades,
        equity_curve=merged_curve,
        bars_processed=max(r.bars_processed for r in component_results),
        signals_emitted=sum(r.signals_emitted for r in component_results),
        signals_filled=sum(r.signals_filled for r in component_results),
        signals_unfilled=sum(r.signals_unfilled for r in component_results),
        signals_skipped_pending=sum(
            r.signals_skipped_pending for r in component_results
        ),
        config_snapshot={
            "components": [
                {"account": r.account_name, **r.config_snapshot}
                for r in component_results
            ],
        },
    )
