"""YAML 設定載入與驗證。

對應 configs/default.yaml schema（含 §11 三階段止損 trailing 子區塊）。
所有不合法欄位皆 raise ``ConfigError``，CLAUDE.md §5 fail-fast。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.exceptions import ConfigError


@dataclass(frozen=True)
class PeriodSpec:
    start: pd.Timestamp
    end: pd.Timestamp | None  # None = 用到資料最後

    @classmethod
    def parse(cls, raw: dict[str, Any], section: str) -> "PeriodSpec":
        if "start" not in raw:
            raise ConfigError(f"period.{section}.start missing")
        start = pd.Timestamp(raw["start"])
        end_raw = raw.get("end")
        end = pd.Timestamp(end_raw) if end_raw is not None else None
        if end is not None and end <= start:
            raise ConfigError(
                f"period.{section}: end ({end}) must be > start ({start})"
            )
        return cls(start=start, end=end)


@dataclass(frozen=True)
class TrailingConfig:
    """三階段止損設定（對應 strategy.trailing 區塊）。"""

    swing_lookback: int = 4
    stage1_slippage_buffer: float = 0.0003

    stage2_normal_trigger_r: float = 1.2
    stage2_abnormal_trigger_r: float = 2.4
    stage2_buffer_r: float = 0.2
    stage2_pct_trigger: float = 0.0   # 0 = 關閉；>0 = OR 條件 (peak_pct ≥ 此值)

    stage3_normal_trigger_r: float = 2.4
    stage3_abnormal_trigger_r: float = 4.8
    bollinger_period: int = 20
    bollinger_num_std: float = 2.0

    stage3_mode: str = "bollinger"   # "bollinger" | "r_ladder"
    r_ladder_normal_first_trigger: float = 2.8
    r_ladder_normal_step: float = 1.0
    r_ladder_abnormal_first_trigger: float = 5.6
    r_ladder_abnormal_step: float = 2.0
    r_ladder_trigger_offset: float = 0.3
    r_ladder_abnormal_trigger_offset: float = 0.6

    # early_exit
    early_exit_enabled: bool = False
    early_exit_observation_bars: int = 1
    early_exit_metric: str = "peak"   # "peak" | "peak_pct" | "close"
    early_exit_min_peak_r: float = 0.0
    early_exit_min_peak_pct: float = 0.0
    early_exit_min_close_r: float = 0.0


@dataclass(frozen=True)
class RCapConfig:
    """R-cap 設定（對應 strategy.r_cap 區塊）。"""
    mode: str = "off"          # "off" | "rolling_avg"
    window: int = 100


@dataclass(frozen=True)
class SignalFilterConfig:
    """進場訊號濾網設定（對應 strategy.signal_filter 區塊）。"""
    mode: str = "off"          # "off" | "body_sum" | "body_sq_sum"
    window: int = 6
    threshold: float = 0.60


@dataclass(frozen=True)
class ChopFilterConfig:
    """盤整濾網設定（對應 strategy.chop_filter 區塊）。"""
    enabled: bool = True
    bbw_rank_min: float = 40.0
    atr_rank_min: float = 40.0
    adx_min: float = 20.0
    bb_period: int = 20
    bb_num_std: float = 2.0
    atr_period: int = 14
    adx_period: int = 14
    rank_window: int = 200


@dataclass(frozen=True)
class FullConfig:
    """完整設定的扁平容器。"""

    # data
    source_parquet: Path
    symbol: str
    timeframe: str

    # period
    in_sample: PeriodSpec
    out_of_sample: PeriodSpec

    # account
    initial_capital: float
    position_size_pct: float
    sizing_mode: str  # "pct" | "risk"
    risk_per_trade_usdt: float
    risk_per_trade_pct: float
    allow_pyramiding: bool
    leverage_cap: float
    r_min_pct: float
    entry_hour_blacklist: tuple[int, ...]

    # fees
    taker_fee_rate: float
    maker_fee_rate: float
    slippage_pct: float

    # strategy: entry
    wma_fast: int
    wma_slow: int

    # strategy: trailing stop
    trailing: TrailingConfig

    # strategy: signal filter
    signal_filter: SignalFilterConfig

    # strategy: r_cap
    r_cap: RCapConfig

    # strategy: chop filter
    chop_filter: ChopFilterConfig

    # backtest
    output_dir: Path
    show_progress: bool
    force_close_at_end: bool
    log_level: str

    raw: dict[str, Any] = field(default_factory=dict)


_VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1H", "4H"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}
_VALID_SIZING_MODES = {"pct", "risk"}
_VALID_SIGNAL_FILTER_MODES = {"off", "body_sum", "body_sq_sum"}
_VALID_R_CAP_MODES = {"off", "rolling_avg"}


def load_config(path: str | Path) -> FullConfig:
    """讀取並驗證 YAML 設定檔。"""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be dict, got {type(raw).__name__}")

    try:
        data = raw["data"]
        period = raw["period"]
        account = raw["account"]
        fees = raw["fees"]
        strategy = raw["strategy"]
        backtest = raw["backtest"]
    except KeyError as e:
        raise ConfigError(f"missing top-level section: {e}") from e

    # ---- data ----
    timeframe = data["timeframe"]
    if timeframe not in _VALID_TIMEFRAMES:
        raise ConfigError(
            f"data.timeframe '{timeframe}' invalid; must be one of {sorted(_VALID_TIMEFRAMES)}"
        )
    symbol = str(data["symbol"])
    # 資料夾 + symbol 推導 parquet 路徑：避免 symbol 與檔案路徑不同步的 bug。
    # 慣例檔名為 ``{SYMBOL}_1m.parquet``，與 PPO_TradingModel data/processed/ 一致。
    if "source_dir" not in data:
        raise ConfigError(
            "data.source_dir missing (改為以資料夾 + symbol 推導 parquet 路徑；"
            "刪除舊的 data.source_parquet 並新增 data.source_dir)"
        )
    source_dir = Path(data["source_dir"]).expanduser()
    if not source_dir.is_dir():
        raise ConfigError(f"data.source_dir not a directory: {source_dir}")
    source = source_dir / f"{symbol}_1m.parquet"
    if not source.is_file():
        raise ConfigError(
            f"resolved parquet not found: {source} "
            f"(symbol='{symbol}', source_dir='{source_dir}'); "
            "請確認檔案存在或更新 data.symbol"
        )

    # ---- period ----
    in_sample = PeriodSpec.parse(period["in_sample"], "in_sample")
    oos = PeriodSpec.parse(period["out_of_sample"], "out_of_sample")

    # ---- account ----
    initial_capital = float(account["initial_capital"])
    position_size_pct = float(account["position_size_pct"])
    sizing_mode = str(account.get("sizing_mode", "pct")).lower()
    if sizing_mode not in _VALID_SIZING_MODES:
        raise ConfigError(
            f"account.sizing_mode '{sizing_mode}' invalid; "
            f"must be one of {sorted(_VALID_SIZING_MODES)}"
        )
    risk_per_trade_usdt = float(account.get("risk_per_trade_usdt", 1.0))
    if risk_per_trade_usdt <= 0:
        raise ConfigError(
            f"account.risk_per_trade_usdt must be > 0, got {risk_per_trade_usdt}"
        )
    # 動態 risk：> 0 啟用 (equity × pct)，覆蓋 fixed USDT；= 0 關閉
    risk_per_trade_pct = float(account.get("risk_per_trade_pct", 0.0))
    if risk_per_trade_pct < 0 or risk_per_trade_pct > 0.10:
        raise ConfigError(
            "account.risk_per_trade_pct must be in [0, 0.10] "
            f"(>0.10 視為誤打，例如 0.1 應為 0.01)，got {risk_per_trade_pct}"
        )
    allow_pyramiding = bool(account.get("allow_pyramiding", False))
    leverage_cap = float(account.get("leverage_cap", 1.0))
    if leverage_cap <= 0:
        raise ConfigError(
            f"account.leverage_cap must be > 0, got {leverage_cap}"
        )
    r_min_pct = float(account.get("r_min_pct", 0.0))
    if r_min_pct < 0:
        raise ConfigError(
            f"account.r_min_pct must be >= 0, got {r_min_pct}"
        )

    raw_blacklist = account.get("entry_hour_blacklist", []) or []
    if not isinstance(raw_blacklist, list):
        raise ConfigError(
            f"account.entry_hour_blacklist must be a list, got {type(raw_blacklist).__name__}"
        )
    entry_hour_blacklist: tuple[int, ...] = tuple(int(h) for h in raw_blacklist)
    for h in entry_hour_blacklist:
        if not 0 <= h <= 23:
            raise ConfigError(
                f"account.entry_hour_blacklist 元素必須為 0..23，got {h}"
            )

    # ---- fees ----
    taker = float(fees["taker_fee_rate"])
    maker = float(fees["maker_fee_rate"])
    slip = float(fees["slippage_pct"])

    # ---- strategy: entry ----
    wma_fast = int(strategy["wma_fast"])
    wma_slow = int(strategy["wma_slow"])

    # ---- strategy: trailing ----
    trailing_raw = strategy.get("trailing", {})
    trailing = TrailingConfig(
        swing_lookback=int(trailing_raw.get("swing_lookback", 4)),
        stage1_slippage_buffer=float(trailing_raw.get("stage1_slippage_buffer", 0.0003)),
        stage2_normal_trigger_r=float(trailing_raw.get("stage2_normal_trigger_r", 1.2)),
        stage2_abnormal_trigger_r=float(trailing_raw.get("stage2_abnormal_trigger_r", 2.4)),
        stage2_buffer_r=float(trailing_raw.get("stage2_buffer_r", 0.2)),
        stage2_pct_trigger=float(trailing_raw.get("stage2_pct_trigger", 0.0)),
        stage3_normal_trigger_r=float(trailing_raw.get("stage3_normal_trigger_r", 2.4)),
        stage3_abnormal_trigger_r=float(trailing_raw.get("stage3_abnormal_trigger_r", 4.8)),
        bollinger_period=int(trailing_raw.get("bollinger_period", 20)),
        bollinger_num_std=float(trailing_raw.get("bollinger_num_std", 2.0)),
        stage3_mode=str(trailing_raw.get("stage3_mode", "bollinger")).lower(),
        r_ladder_normal_first_trigger=float(
            trailing_raw.get("r_ladder_normal_first_trigger", 2.8)
        ),
        r_ladder_normal_step=float(trailing_raw.get("r_ladder_normal_step", 1.0)),
        r_ladder_abnormal_first_trigger=float(
            trailing_raw.get("r_ladder_abnormal_first_trigger", 5.6)
        ),
        r_ladder_abnormal_step=float(trailing_raw.get("r_ladder_abnormal_step", 2.0)),
        r_ladder_trigger_offset=float(trailing_raw.get("r_ladder_trigger_offset", 0.3)),
        r_ladder_abnormal_trigger_offset=float(
            trailing_raw.get("r_ladder_abnormal_trigger_offset", 0.6)
        ),
        early_exit_enabled=bool(trailing_raw.get("early_exit_enabled", False)),
        early_exit_observation_bars=int(
            trailing_raw.get("early_exit_observation_bars", 1)
        ),
        early_exit_metric=str(trailing_raw.get("early_exit_metric", "peak")),
        early_exit_min_peak_r=float(
            trailing_raw.get("early_exit_min_peak_r", 0.0)
        ),
        early_exit_min_peak_pct=float(
            trailing_raw.get("early_exit_min_peak_pct", 0.0)
        ),
        early_exit_min_close_r=float(
            trailing_raw.get("early_exit_min_close_r", 0.0)
        ),
    )

    # ---- strategy: signal filter ----
    sf_raw = strategy.get("signal_filter", {}) or {}
    # YAML 裡 'off' / 'on' 會被解析成 boolean，這裡容錯回 string
    raw_mode = sf_raw.get("mode", "off")
    if raw_mode is False:
        raw_mode = "off"
    elif raw_mode is True:
        raw_mode = "on"  # 後面驗證會 raise，提示使用者
    sf_mode = str(raw_mode).lower()
    if sf_mode not in _VALID_SIGNAL_FILTER_MODES:
        raise ConfigError(
            f"strategy.signal_filter.mode '{sf_mode}' invalid; "
            f"must be one of {sorted(_VALID_SIGNAL_FILTER_MODES)}"
        )
    sf_window = int(sf_raw.get("window", 6))
    if sf_window < 1:
        raise ConfigError(f"strategy.signal_filter.window must be >= 1, got {sf_window}")
    sf_threshold = float(sf_raw.get("threshold", 0.60))
    if not (0.0 < sf_threshold < 1.0):
        raise ConfigError(
            f"strategy.signal_filter.threshold must be in (0, 1), got {sf_threshold}"
        )
    signal_filter = SignalFilterConfig(
        mode=sf_mode, window=sf_window, threshold=sf_threshold,
    )

    # ---- strategy: r_cap ----
    rc_raw = strategy.get("r_cap", {}) or {}
    raw_rc_mode = rc_raw.get("mode", "off")
    if raw_rc_mode is False:
        raw_rc_mode = "off"
    elif raw_rc_mode is True:
        raw_rc_mode = "on"  # 後面驗證會 raise
    rc_mode = str(raw_rc_mode).lower()
    if rc_mode not in _VALID_R_CAP_MODES:
        raise ConfigError(
            f"strategy.r_cap.mode '{rc_mode}' invalid; "
            f"must be one of {sorted(_VALID_R_CAP_MODES)}"
        )
    rc_window = int(rc_raw.get("window", 100))
    if rc_window < 1:
        raise ConfigError(f"strategy.r_cap.window must be >= 1, got {rc_window}")
    r_cap = RCapConfig(mode=rc_mode, window=rc_window)

    # ---- strategy: chop_filter ----
    cf_raw = strategy.get("chop_filter", {}) or {}
    cf_enabled = bool(cf_raw.get("enabled", True))
    cf_bbw = float(cf_raw.get("bbw_rank_min", 40.0))
    cf_atr = float(cf_raw.get("atr_rank_min", 40.0))
    cf_adx = float(cf_raw.get("adx_min", 20.0))
    if not 0 <= cf_bbw <= 100:
        raise ConfigError(f"strategy.chop_filter.bbw_rank_min must be in [0, 100], got {cf_bbw}")
    if not 0 <= cf_atr <= 100:
        raise ConfigError(f"strategy.chop_filter.atr_rank_min must be in [0, 100], got {cf_atr}")
    if cf_adx < 0:
        raise ConfigError(f"strategy.chop_filter.adx_min must be >= 0, got {cf_adx}")
    cf_bb_period = int(cf_raw.get("bb_period", 20))
    cf_bb_std = float(cf_raw.get("bb_num_std", 2.0))
    cf_atr_p = int(cf_raw.get("atr_period", 14))
    cf_adx_p = int(cf_raw.get("adx_period", 14))
    cf_rank_w = int(cf_raw.get("rank_window", 200))
    if cf_bb_std <= 0:
        raise ConfigError(f"strategy.chop_filter.bb_num_std must be > 0, got {cf_bb_std}")
    for name, val, lo in [
        ("bb_period", cf_bb_period, 2),
        ("atr_period", cf_atr_p, 1),
        ("adx_period", cf_adx_p, 1),
        ("rank_window", cf_rank_w, 2),
    ]:
        if val < lo:
            raise ConfigError(f"strategy.chop_filter.{name} must be >= {lo}, got {val}")
    chop_filter = ChopFilterConfig(
        enabled=cf_enabled,
        bbw_rank_min=cf_bbw, atr_rank_min=cf_atr, adx_min=cf_adx,
        bb_period=cf_bb_period, bb_num_std=cf_bb_std,
        atr_period=cf_atr_p, adx_period=cf_adx_p, rank_window=cf_rank_w,
    )

    # ---- backtest ----
    output_dir = Path(backtest.get("output_dir", "results")).expanduser()
    show_progress = bool(backtest.get("show_progress", True))
    force_close_at_end = bool(backtest.get("force_close_at_end", False))
    log_level = str(backtest.get("log_level", "INFO")).upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"backtest.log_level '{log_level}' invalid; must be one of {sorted(_VALID_LOG_LEVELS)}"
        )

    return FullConfig(
        source_parquet=source,
        symbol=symbol,
        timeframe=timeframe,
        in_sample=in_sample,
        out_of_sample=oos,
        initial_capital=initial_capital,
        position_size_pct=position_size_pct,
        sizing_mode=sizing_mode,
        risk_per_trade_usdt=risk_per_trade_usdt,
        risk_per_trade_pct=risk_per_trade_pct,
        allow_pyramiding=allow_pyramiding,
        leverage_cap=leverage_cap,
        r_min_pct=r_min_pct,
        entry_hour_blacklist=entry_hour_blacklist,
        taker_fee_rate=taker,
        maker_fee_rate=maker,
        slippage_pct=slip,
        wma_fast=wma_fast,
        wma_slow=wma_slow,
        trailing=trailing,
        signal_filter=signal_filter,
        r_cap=r_cap,
        chop_filter=chop_filter,
        output_dir=output_dir,
        show_progress=show_progress,
        force_close_at_end=force_close_at_end,
        log_level=log_level,
        raw=raw,
    )
