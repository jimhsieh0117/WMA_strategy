"""Monte Carlo 回測驗證。

提供兩種模擬方法：
  1. ``reshuffle``：交易順序重排（permutation）。保留全部交易、只變順序。
     回答「同樣這批交易換個順序，DD 會多糟？」。
  2. ``bootstrap``：交易層級有放回抽樣（可帶 block size 做 block bootstrap）。
     回答「在母體分布類似的情境下，績效信賴區間範圍？」。

設計：
  - 輸入只需要 ``Trade`` 列表 + ``MCConfig``，與 backtest engine 解耦
  - 全程 numpy 向量化，10000 sims × 5000 trades 大約 1~2 秒
  - 不變式檢查：trade 數 >= 1、initial_capital > 0、n_simulations > 0
  - return-path / bar-level bootstrap 留待未來實作（API 已預留 namespace）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.broker.types import Trade
from src.utils.exceptions import DataIntegrityError

logger = logging.getLogger(__name__)

Method = Literal["reshuffle", "bootstrap"]


# --------------------------------------------------------------------------- #
# Config / Result
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MCConfig:
    n_simulations: int = 10_000
    initial_capital: float = 500.0
    ruin_threshold_pct: float = 50.0  # equity 跌破 initial*(1-pct/100) 視為破產
    block_size: int = 1               # bootstrap 用；1 = 一般 bootstrap，>1 = block bootstrap
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.n_simulations <= 0:
            raise DataIntegrityError("n_simulations must be > 0")
        if self.initial_capital <= 0:
            raise DataIntegrityError("initial_capital must be > 0")
        if not (0 < self.ruin_threshold_pct < 100):
            raise DataIntegrityError("ruin_threshold_pct must be in (0, 100)")
        if self.block_size < 1:
            raise DataIntegrityError("block_size must be >= 1")


@dataclass
class MCResult:
    method: Method
    config: MCConfig
    n_trades: int
    # 每次模擬一筆 metric → (n_simulations, ) 各欄
    final_equity: np.ndarray
    total_return_pct: np.ndarray
    max_drawdown_pct: np.ndarray
    profit_factor: np.ndarray
    sharpe_trade: np.ndarray
    ruined: np.ndarray            # bool
    profitable: np.ndarray        # bool
    # 原始（未模擬）的基準
    baseline: dict = field(default_factory=dict)

    # ------- 統計 ------- #
    def summary(self) -> pd.DataFrame:
        cols = {
            "final_equity":     self.final_equity,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "profit_factor":    self.profit_factor,
            "sharpe_trade":     self.sharpe_trade,
        }
        rows = {}
        for name, arr in cols.items():
            arr_f = arr[np.isfinite(arr)]
            if arr_f.size == 0:
                rows[name] = {q: float("nan") for q in
                              ["mean", "p05", "p25", "p50", "p75", "p95"]}
            else:
                rows[name] = {
                    "mean": float(np.mean(arr_f)),
                    "p05":  float(np.percentile(arr_f, 5)),
                    "p25":  float(np.percentile(arr_f, 25)),
                    "p50":  float(np.percentile(arr_f, 50)),
                    "p75":  float(np.percentile(arr_f, 75)),
                    "p95":  float(np.percentile(arr_f, 95)),
                }
        df = pd.DataFrame(rows).T
        df["baseline"] = [self.baseline.get(k, float("nan")) for k in df.index]
        return df[["baseline", "mean", "p05", "p25", "p50", "p75", "p95"]]

    @property
    def prob_ruin(self) -> float:
        return float(self.ruined.mean())

    @property
    def prob_profit(self) -> float:
        return float(self.profitable.mean())


# --------------------------------------------------------------------------- #
# 核心數值
# --------------------------------------------------------------------------- #

def _equity_paths(pnl_matrix: np.ndarray, initial_capital: float) -> np.ndarray:
    """從 (n_sims, n_trades) PnL 矩陣產出 (n_sims, n_trades+1) equity 路徑。"""
    n_sims, n_trades = pnl_matrix.shape
    eq = np.empty((n_sims, n_trades + 1), dtype=np.float64)
    eq[:, 0] = initial_capital
    np.cumsum(pnl_matrix, axis=1, out=eq[:, 1:])
    eq[:, 1:] += initial_capital
    return eq


def _max_drawdown_pct(equity_paths: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity_paths, axis=1)
    dd = (running_max - equity_paths) / running_max
    return dd.max(axis=1) * 100.0


def _profit_factor(pnl_matrix: np.ndarray) -> np.ndarray:
    wins = np.where(pnl_matrix > 0, pnl_matrix, 0.0).sum(axis=1)
    losses = np.where(pnl_matrix < 0, -pnl_matrix, 0.0).sum(axis=1)
    # 全勝 → inf；全敗 → 0
    pf = np.full(pnl_matrix.shape[0], np.nan)
    mask = losses > 0
    pf[mask] = wins[mask] / losses[mask]
    pf[(~mask) & (wins > 0)] = np.inf
    pf[(~mask) & (wins == 0)] = 0.0
    return pf


def _sharpe_per_trade(pnl_matrix: np.ndarray) -> np.ndarray:
    """每筆交易 Sharpe（未年化）：mean(pnl) / std(pnl)。"""
    mu = pnl_matrix.mean(axis=1)
    sd = pnl_matrix.std(axis=1, ddof=1)
    sharpe = np.where(sd > 0, mu / sd, np.nan)
    return sharpe


def _compute_metrics(
    pnl_matrix: np.ndarray, cfg: MCConfig,
) -> tuple[np.ndarray, ...]:
    eq = _equity_paths(pnl_matrix, cfg.initial_capital)
    final_eq = eq[:, -1]
    total_ret = (final_eq / cfg.initial_capital - 1.0) * 100.0
    max_dd = _max_drawdown_pct(eq)
    pf = _profit_factor(pnl_matrix)
    sharpe = _sharpe_per_trade(pnl_matrix)
    ruin_line = cfg.initial_capital * (1.0 - cfg.ruin_threshold_pct / 100.0)
    ruined = (eq.min(axis=1) <= ruin_line)
    profitable = (final_eq > cfg.initial_capital)
    return final_eq, total_ret, max_dd, pf, sharpe, ruined, profitable


def _baseline(pnl: np.ndarray, cfg: MCConfig) -> dict:
    eq = _equity_paths(pnl.reshape(1, -1), cfg.initial_capital)[0]
    final = float(eq[-1])
    return {
        "final_equity":     final,
        "total_return_pct": (final / cfg.initial_capital - 1.0) * 100.0,
        "max_drawdown_pct": float(_max_drawdown_pct(eq.reshape(1, -1))[0]),
        "profit_factor":    float(_profit_factor(pnl.reshape(1, -1))[0]),
        "sharpe_trade":     float(_sharpe_per_trade(pnl.reshape(1, -1))[0]),
    }


# --------------------------------------------------------------------------- #
# 兩種模擬方法
# --------------------------------------------------------------------------- #

def _extract_pnl(trades: list[Trade]) -> np.ndarray:
    if not trades:
        raise DataIntegrityError("trades list is empty — cannot run Monte Carlo")
    return np.asarray([t.net_pnl for t in trades], dtype=np.float64)


def reshuffle(trades: list[Trade], cfg: MCConfig) -> MCResult:
    """交易順序重排：保留 trade set，只 permute 順序。"""
    pnl = _extract_pnl(trades)
    n = pnl.size
    rng = np.random.default_rng(cfg.seed)

    # 矩陣化：對每行做 independent permutation
    pnl_matrix = np.tile(pnl, (cfg.n_simulations, 1))
    # rng.permuted 對指定 axis 各行獨立 shuffle
    pnl_matrix = rng.permuted(pnl_matrix, axis=1)

    final_eq, total_ret, max_dd, pf, sharpe, ruined, profitable = \
        _compute_metrics(pnl_matrix, cfg)

    return MCResult(
        method="reshuffle", config=cfg, n_trades=n,
        final_equity=final_eq, total_return_pct=total_ret,
        max_drawdown_pct=max_dd, profit_factor=pf, sharpe_trade=sharpe,
        ruined=ruined, profitable=profitable,
        baseline=_baseline(pnl, cfg),
    )


def bootstrap(trades: list[Trade], cfg: MCConfig) -> MCResult:
    """有放回抽樣：可帶 block_size 做 block bootstrap。

    block_size = 1 → 標準 bootstrap（每筆交易獨立抽）。
    block_size > 1 → block bootstrap：一次抽連續 k 筆，保留短期自相關。
    """
    pnl = _extract_pnl(trades)
    n = pnl.size
    k = cfg.block_size
    rng = np.random.default_rng(cfg.seed)

    if k == 1:
        idx = rng.integers(0, n, size=(cfg.n_simulations, n))
        pnl_matrix = pnl[idx]
    else:
        # 抽 ceil(n/k) 個 block 起點，每個 block 連續取 k 筆，再 trim 到 n
        n_blocks = int(np.ceil(n / k))
        starts = rng.integers(0, n - k + 1, size=(cfg.n_simulations, n_blocks))
        # 對每個 start 攤平成 [s, s+1, ..., s+k-1]
        offsets = np.arange(k)
        idx_full = (starts[:, :, None] + offsets[None, None, :]).reshape(
            cfg.n_simulations, n_blocks * k)
        pnl_matrix = pnl[idx_full[:, :n]]

    final_eq, total_ret, max_dd, pf, sharpe, ruined, profitable = \
        _compute_metrics(pnl_matrix, cfg)

    return MCResult(
        method="bootstrap", config=cfg, n_trades=n,
        final_equity=final_eq, total_return_pct=total_ret,
        max_drawdown_pct=max_dd, profit_factor=pf, sharpe_trade=sharpe,
        ruined=ruined, profitable=profitable,
        baseline=_baseline(pnl, cfg),
    )
