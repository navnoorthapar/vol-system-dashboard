"""
backtest_engine.py — C10: Full Backtest & Reporting Engine

Period:   2018-01-01 → 2025-03-24
Capital:  $1,000,000 (initial)
Signals:  S1 (IVR Spread), S2 (VIX TS Curve), S3 (Dispersion Proxy), Combined

Cost model (non-negotiable):
  Bid-ask:    0.15 vol pts per leg  → 0.15 × 0.01 × vega × n_contracts × 100
  Commission: $0.65 per contract per leg (2 legs per straddle)
  Margin:     20% of K × multiplier × n_contracts for short options
  Slippage:   0.05% of |Δdelta| × S × multiplier × n_contracts on delta-hedge rebalancing

P&L methodology:
  S1/S2 — 30-day ATM straddle, delta-hedged daily (C7 framework):
    daily_pnl_t = direction × (V_t − V_{t−1} − Δ_{t−1} × ΔS) × n_contracts × 100
  S3  — VIX/VVIX ratio z-score proxy (no actual options traded):
    daily_pnl_t = position × kelly × nav × Δz_ratio × S3_Z_SCALE

Assumptions (documented for transparency):
  1. 30-day ATM straddles for S1/S2; new entry re-strikes ATM at current spot
  2. Vol surface: VIX TS interpolated in total-variance space (ATM only, no smile)
  3. Fixed r=0.045, q=0.013 throughout the backtest period
  4. Heston params fixed from 2026-03-24 calibration (no time-varying recalibration)
  5. Regime gate: no NEW entries in Regime 2; existing trades close on natural exit
  6. Margin for short options held as reserve (no interest charged on margin)
  7. No correlation netting: each signal sizes independently off full NAV
  8. Walk-forward: C8 XGBoost retrained on expanding window; signal thresholds fixed

Honest failures (from C4, C7 — see report):
  • PDV over-forecast in 2020: σ_PDV=91.6% vs σ_ATM=55.4% on COVID crash
  • Heston rho=−0.99 boundary convergence: steep 2026 skew under-modelled
  • C6: 3 unstable vomma nodes; delta-only hedge insufficient on those days
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import norm

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.backtest.delta_hedger import (
    _bs_straddle_value,
    _bs_straddle_greeks,
    _interp_atm_iv,
)
from joint_vol_calibration.signals.signal_engine import SignalEngine
from joint_vol_calibration.signals.regime_classifier import (
    RegimeClassifier,
    build_features,
    build_regime_labels,
)

logger = logging.getLogger(__name__)

# ── Backtest constants ─────────────────────────────────────────────────────────

BT_START:          str   = "2018-01-01"
BT_END:            str   = "2025-03-24"
INITIAL_CAPITAL:   float = 1_000_000.0
RISK_FREE_RATE:    float = 0.05      # annualised rf for Sharpe (5%)
R_BACKTEST:        float = 0.045     # discount rate in option pricing
Q_BACKTEST:        float = 0.013     # SPX dividend yield
TRADING_DAYS:      int   = 252

# Cost model — non-negotiable per spec
BID_ASK_VOL_PTS:   float = 0.15     # vol points per leg (0.15 × 0.01 = 0.0015 decimal)
COMMISSION_PER_LEG: float = 0.65    # $ per contract per leg
MARGIN_RATIO:      float = 0.20     # short-option initial margin as fraction of notional
SLIPPAGE_PCT:      float = 0.0005   # 0.05% of |Δdelta| × S on delta-hedge rebalancing
CONTRACT_MULT:     int   = 100      # SPX options: 100 shares per contract

# S1/S2 straddle tenor
STRADDLE_TENOR_DAYS: int = 30       # 30-day ATM straddle at entry

# S3 proxy parameters
S3_Z_SCALE:        float = 0.02    # 2% of kelly-weighted nav per unit z improvement
S3_PROXY_COST:     float = 0.001   # 0.1% of notional at entry/exit (execution proxy)

# Walk-forward
WF_N_WINDOWS:      int   = 12
WF_WINDOW_MONTHS:  int   = 6

# Output paths
BACKTEST_DIR = DATA_DIR / "backtest"
RESULTS_PATH = BACKTEST_DIR / "full_results.parquet"
REPORTS_DIR  = BACKTEST_DIR / "reports"
REPORT_PATH  = REPORTS_DIR / "full_report.html"

# Crisis periods (start, end) for reporting
CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "COVID_2020":    ("2020-02-19", "2020-04-30"),
    "FedHikes_2022": ("2022-01-01", "2022-12-31"),
    "Tariffs_2025":  ("2025-01-01", "2025-03-24"),
}


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """One completed straddle trade."""
    signal:        str
    direction:     int            # +1 = long, −1 = short
    entry_date:    pd.Timestamp
    exit_date:     Optional[pd.Timestamp]
    K:             float          # ATM strike at entry
    S_entry:       float
    sigma_entry:   float
    n_contracts:   int
    kelly_frac:    float
    entry_cost:    float
    exit_cost:     float = 0.0
    gross_pnl:     float = 0.0
    slippage_cost: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.entry_cost - self.exit_cost - self.slippage_cost

    @property
    def duration_days(self) -> int:
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days


# ── Module-level BS helpers ────────────────────────────────────────────────────

def _straddle_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BS straddle vega = 2 × call vega ($ per decimal-vol-unit per share)."""
    T = max(T, 1e-8)
    sqrtT = np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(max(F / K, 1e-10)) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    return 2.0 * S * np.exp(-q * T) * norm.pdf(d1) * sqrtT


def _one_way_cost(vega: float, n_contracts: int) -> float:
    """
    One-way transaction cost for opening or closing a straddle position.

    Bid-ask: 0.15 vol pts × 0.01 (decimal) × vega_straddle × n_contracts × 100
    Commission: $0.65 × 2 legs × n_contracts
    """
    ba   = BID_ASK_VOL_PTS * 0.01 * vega * CONTRACT_MULT * n_contracts
    comm = COMMISSION_PER_LEG * 2 * n_contracts
    return ba + comm


def _build_s3_zscore(vix_wide: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    Compute the VIX/VVIX ratio z-score used by Signal 3.
    Matches signal_engine.py generate_signal3() logic exactly (zero look-ahead).
    """
    idx = vix_wide.set_index("date") if "date" in vix_wide.columns else vix_wide
    idx.index = pd.to_datetime(idx.index)
    ratio = (idx["^VIX"] / idx["^VVIX"].clip(lower=1e-6)).clip(lower=1e-6)
    mu    = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    std   = ratio.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    return ((ratio - mu) / std).rename("z_ratio")


# ── Core P&L simulators ────────────────────────────────────────────────────────

def _simulate_straddle_pnl(
    position:     pd.Series,      # index=date, values ∈ {−1, 0, +1} (or fractional; sign = direction)
    kelly:        pd.Series,      # index=date, values ∈ [0, 0.20]
    nav_series:   pd.Series,      # index=date, capital available for sizing
    spx_close:    pd.Series,      # index=date, SPX close price
    vix_wide_idx: pd.DataFrame,   # date-indexed wide VIX DataFrame
    r:            float = R_BACKTEST,
    q:            float = Q_BACKTEST,
    tenor_days:   int   = STRADDLE_TENOR_DAYS,
    signal_label: str   = "s",
) -> Tuple[pd.Series, List[TradeRecord]]:
    """
    Simulate delta-hedged 30-day ATM straddle P&L for a position series.

    P&L formula (C7 framework, zero look-ahead):
      daily_pnl_t = direction × (V_t − V_{t−1} − Δ_{t−1} × ΔS_t) × n_contracts × 100

    Where Δ_{t−1} = BS straddle delta computed at close of day t−1.
    Entry/exit costs applied on transition days.
    Hedge rebalancing slippage on every active day.
    """
    T_entry = tenor_days / 365.0
    dates   = position.index
    pnl     = pd.Series(0.0, index=dates, dtype=float)
    trades: List[TradeRecord] = []

    # Trade state
    in_trade  = False
    direction = 0
    K = prev_V = prev_delta = prev_S = 0.0
    n_contracts = 0
    days_biz    = 0
    entry_date  = None
    kelly_entry = 0.0
    S_entry     = 0.0
    sigma_entry = 0.0
    entry_cost_ = 0.0
    trade_gross = 0.0
    trade_slip  = 0.0

    for i, date in enumerate(dates):
        raw_pos = float(position.iloc[i])
        pos     = int(np.sign(raw_pos))   # collapse fractional → {−1, 0, +1}
        k       = float(kelly.iloc[i])
        nav     = float(nav_series.iloc[i])

        if date not in spx_close.index or date not in vix_wide_idx.index:
            continue

        S       = float(spx_close.loc[date])
        vix_row = vix_wide_idx.loc[date]
        T_rem   = max(T_entry - days_biz / TRADING_DAYS, 1.0 / 365.0) if in_trade else T_entry
        sigma   = float(_interp_atm_iv(vix_row, T_rem))
        if not (sigma > 0) or np.isnan(sigma):
            sigma = 0.20

        if not in_trade and pos != 0:
            # ── ENTRY ─────────────────────────────────────────────────────────
            direction   = pos
            K           = S
            S_entry     = S
            sigma_entry = sigma
            kelly_entry = k
            V0          = max(float(_bs_straddle_value(S, K, T_entry, r, q, sigma)), 1e-4)
            vega0       = _straddle_vega(S, K, T_entry, r, q, sigma)
            n_contracts = max(1, int(k * nav / (V0 * CONTRACT_MULT)))

            entry_cost_  = _one_way_cost(vega0, n_contracts)
            pnl.iloc[i] -= entry_cost_

            prev_V, _, _, _ = (V0, None, None, None)
            prev_delta      = float(_bs_straddle_greeks(S, K, T_entry, r, q, sigma)[0])
            prev_S          = S
            prev_V          = V0
            in_trade        = True
            entry_date      = date
            days_biz        = 0
            trade_gross     = 0.0
            trade_slip      = 0.0

        elif in_trade:
            # ── HOLD ──────────────────────────────────────────────────────────
            days_biz += 1
            V_t         = float(_bs_straddle_value(S, K, T_rem, r, q, sigma))
            delta_t     = float(_bs_straddle_greeks(S, K, T_rem, r, q, sigma)[0])

            # Delta-hedged P&L
            raw_pnl_t  = direction * (V_t - prev_V - prev_delta * (S - prev_S)) * n_contracts * CONTRACT_MULT
            slip_t     = SLIPPAGE_PCT * abs(delta_t - prev_delta) * S * n_contracts * CONTRACT_MULT

            pnl.iloc[i] += raw_pnl_t - slip_t
            trade_gross += raw_pnl_t
            trade_slip  += slip_t

            prev_V     = V_t
            prev_delta = delta_t
            prev_S     = S

            if pos == 0:
                # ── EXIT ──────────────────────────────────────────────────────
                exit_cost_   = _one_way_cost(_straddle_vega(S, K, T_rem, r, q, sigma), n_contracts)
                pnl.iloc[i] -= exit_cost_

                trades.append(TradeRecord(
                    signal=signal_label, direction=direction,
                    entry_date=entry_date, exit_date=date,
                    K=K, S_entry=S_entry, sigma_entry=sigma_entry,
                    n_contracts=n_contracts, kelly_frac=kelly_entry,
                    entry_cost=entry_cost_, exit_cost=exit_cost_,
                    gross_pnl=trade_gross, slippage_cost=trade_slip,
                ))
                in_trade    = False
                direction   = 0
                K           = 0.0
                n_contracts = 0

    return pnl, trades


def _simulate_s3_pnl(
    position:   pd.Series,    # index=date, values ∈ {0, +1} (long-only)
    kelly:      pd.Series,
    nav_series: pd.Series,
    z_ratio:    pd.Series,    # precomputed z-score (date-indexed)
) -> Tuple[pd.Series, List[TradeRecord]]:
    """
    Proxy P&L for Signal 3 (dispersion trade).

    Assumption (documented): no actual options traded.
    P&L ≈ position × kelly × nav × Δz_ratio × S3_Z_SCALE (= 2% per z-unit).
    Entry/exit cost: 0.1% of kelly-weighted notional (proxy for execution spread).

    A full mean-reversion (z: −1.0 → −0.3, ~15 days) yields ≈ 1.4% × kelly × nav gross.
    """
    dates  = position.index
    pnl    = pd.Series(0.0, index=dates, dtype=float)
    trades: List[TradeRecord] = []

    in_trade    = False
    entry_date  = None
    entry_cost_ = 0.0
    notional_0  = 0.0
    trade_gross = 0.0

    for i, date in enumerate(dates):
        pos = int(np.sign(float(position.iloc[i])))
        k   = float(kelly.iloc[i])
        nav = float(nav_series.iloc[i])
        z   = float(z_ratio.loc[date]) if date in z_ratio.index else 0.0

        if not in_trade and pos != 0:
            notional_0   = k * nav
            entry_cost_  = S3_PROXY_COST * notional_0
            pnl.iloc[i] -= entry_cost_
            in_trade     = True
            entry_date   = date
            trade_gross  = 0.0

        elif in_trade:
            # Δz = change in z-score today
            if i > 0 and dates[i - 1] in z_ratio.index:
                dz = z - float(z_ratio.loc[dates[i - 1]])
            else:
                dz = 0.0

            notional    = k * nav
            raw_pnl_t   = notional * dz * S3_Z_SCALE   # long-only
            pnl.iloc[i] += raw_pnl_t
            trade_gross += raw_pnl_t

            if pos == 0:
                notional_exit = k * nav
                exit_cost_    = S3_PROXY_COST * notional_exit
                pnl.iloc[i] -= exit_cost_

                trades.append(TradeRecord(
                    signal="s3", direction=1,
                    entry_date=entry_date, exit_date=date,
                    K=0.0, S_entry=notional_0, sigma_entry=0.0,
                    n_contracts=0, kelly_frac=k,
                    entry_cost=entry_cost_, exit_cost=exit_cost_,
                    gross_pnl=trade_gross, slippage_cost=0.0,
                ))
                in_trade = False

    return pnl, trades


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(
    equity_df: pd.DataFrame,
    rf:        float = RISK_FREE_RATE,
    trade_log: Optional[List[TradeRecord]] = None,
) -> Dict:
    """
    Compute full set of performance metrics from an equity curve.

    Parameters
    ----------
    equity_df : DataFrame with column 'nav' (portfolio value, date-indexed)
    rf        : annualised risk-free rate (default 5%)
    trade_log : optional list of TradeRecord objects for trade-level stats

    Returns
    -------
    dict with: ann_return, sharpe, sortino, max_drawdown, dd_duration_days,
               calmar, win_rate, avg_pnl_per_trade, worst_day, best_day,
               n_trades, n_days, cumulative_return
    """
    nav  = equity_df["nav"].dropna()
    rets = nav.pct_change().dropna()
    n    = len(rets)
    if n == 0:
        return {}

    # Guard: completely flat NAV (zero-trade window) → all ratios undefined
    if rets.abs().max() < 1e-10:
        tl = trade_log or []
        return {
            "ann_return": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "dd_duration_days": 0, "calmar": 0.0,
            "win_rate": None, "avg_pnl_per_trade": None,
            "worst_day": 0.0, "best_day": 0.0,
            "n_trades": len(tl), "n_days": n, "cumulative_return": 0.0,
        }

    rf_daily = rf / TRADING_DAYS
    ann_ret  = float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / n) - 1.0)
    excess   = rets - rf_daily
    sharpe   = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)) if excess.std() > 0 else 0.0

    downside = rets[rets < rf_daily] - rf_daily
    sortino  = (float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))
                if len(downside) > 1 and downside.std() > 0 else 0.0)

    roll_max = nav.cummax()
    dd       = (nav - roll_max) / roll_max
    max_dd   = float(dd.min())

    # Max drawdown duration (consecutive days in drawdown)
    dd_dur = cur = 0
    for v in (dd < -1e-6):
        if v:
            cur   += 1
            dd_dur = max(dd_dur, cur)
        else:
            cur = 0

    calmar   = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0
    cum_ret  = float(nav.iloc[-1] / nav.iloc[0] - 1.0)

    # Trade-level stats
    tl = trade_log or []
    net_pnls  = [t.net_pnl for t in tl] if tl else []
    win_rate  = float(sum(1 for p in net_pnls if p > 0) / len(net_pnls)) if net_pnls else None
    avg_pnl   = float(np.mean(net_pnls)) if net_pnls else None

    return {
        "ann_return":        ann_ret,
        "sharpe":            sharpe,
        "sortino":           sortino,
        "max_drawdown":      max_dd,
        "dd_duration_days":  dd_dur,
        "calmar":            calmar,
        "win_rate":          win_rate,
        "avg_pnl_per_trade": avg_pnl,
        "worst_day":         float(rets.min()),
        "best_day":          float(rets.max()),
        "n_trades":          len(net_pnls),
        "n_days":            n,
        "cumulative_return": cum_ret,
    }


def compute_crisis_performance(
    equity_df: pd.DataFrame,
    periods:   Dict[str, Tuple[str, str]] = CRISIS_PERIODS,
) -> Dict[str, Dict]:
    """Compute performance metrics for each named crisis period."""
    results = {}
    for name, (start, end) in periods.items():
        mask = (equity_df.index >= pd.Timestamp(start)) & (equity_df.index <= pd.Timestamp(end))
        sub  = equity_df.loc[mask]
        if len(sub) < 2:
            results[name] = {"n_days": 0, "note": "No data in period"}
            continue
        nav  = sub["nav"]
        rets = nav.pct_change().dropna()
        dd   = (nav - nav.cummax()) / nav.cummax()
        results[name] = {
            "n_days":       int(len(rets)),
            "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
            "max_drawdown": float(dd.min()),
            "sharpe":       float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS))
                            if len(rets) > 1 and rets.std() > 0 else 0.0,
            "worst_day":    float(rets.min()) if len(rets) else 0.0,
        }
    return results


# ── BacktestEngine ─────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Full backtest engine for C9 trading signals (2018-01-01 → 2025-03-24).

    Usage
    -----
      engine    = BacktestEngine()
      equity_df = engine.run(spx_df, vix_wide)
      metrics   = engine.compute_metrics()
      crisis    = engine.crisis_performance()
      wf_df     = engine.walk_forward_validation(spx_df, vix_wide)
      engine.save()
    """

    ASSUMPTIONS: List[str] = [
        "30-day ATM straddles for S1/S2; re-struck at each new entry",
        "Delta-hedged daily (C7 framework): pnl = direction×(ΔV−Δ_{t−1}×ΔS)×n×100",
        "S3 P&L: VIX/VVIX z-score proxy; 2% of kelly-weighted NAV per z-unit (no real options)",
        "Vol surface: VIX TS interpolated in total-variance space (ATM only, no smile)",
        "Fixed r=0.045, q=0.013 throughout",
        "Heston params fixed at 2026-03-24 calibration (no time-varying recalibration)",
        "Regime gate: new entries blocked in Regime 2; existing trades close naturally",
        "No correlation netting: each signal sizes independently off full NAV ÷ 4",
        "Walk-forward: C8 XGBoost retrained on expanding window; signal thresholds fixed",
    ]

    HONEST_FAILURES: List[str] = [
        "PDV over-forecast vol in 2020 (σ_PDV=91.6% vs σ_ATM=55.4% on COVID crash); "
        "hedge efficiency Run B = 0.303 vs Run A = 0.016",
        "Heston rho=−0.99 boundary convergence (2026 calibration); steep skew under-modelled",
        "C6: 3 unstable vomma nodes; delta-only hedge insufficient on those days",
        "S3 is a proxy: no real single-stock options data in DB; P&L scaling approximate",
    ]

    def __init__(
        self,
        start_date:      str   = BT_START,
        end_date:        str   = BT_END,
        initial_capital: float = INITIAL_CAPITAL,
        r:               float = R_BACKTEST,
        q:               float = Q_BACKTEST,
        seed:            int   = RANDOM_SEED,
    ):
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_capital = initial_capital
        self.r               = r
        self.q               = q
        self.seed            = seed
        np.random.seed(seed)

        self.equity_df:   Optional[pd.DataFrame] = None
        self.signals_df:  Optional[pd.DataFrame] = None
        self.all_trades:  List[TradeRecord]       = []
        self.metrics_:    Optional[Dict]          = None
        self._wf_results: Optional[pd.DataFrame] = None

    # ── Data prep helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_df_with_date_col(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a 'date' column (not only index) for signal_engine."""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date", df.index.name or "index": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def _to_date_indexed(df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame indexed by date (Timestamp)."""
        df = df.copy()
        if "date" in df.columns:
            df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        return df

    @staticmethod
    def _load_clf() -> Optional[RegimeClassifier]:
        try:
            return RegimeClassifier.load()
        except Exception as exc:
            logger.warning("C8 classifier not loaded (%s); using rule-based regimes", exc)
            return None

    @staticmethod
    def _ensure_log_return(spx_df: pd.DataFrame) -> pd.DataFrame:
        df = spx_df.copy()
        if "log_return" not in df.columns:
            df = df.sort_values("date").copy()
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(
        self,
        spx_df:   Optional[pd.DataFrame] = None,
        vix_wide: Optional[pd.DataFrame] = None,
        clf:      Optional[RegimeClassifier] = None,
    ) -> pd.DataFrame:
        """
        Run the full backtest for [start_date, end_date].

        Parameters
        ----------
        spx_df   : SPX OHLCV DataFrame with 'date', 'close', 'log_return' columns
        vix_wide : Wide VIX TS with columns '^VIX', '^VIX3M', '^VVIX', etc.
        clf      : Optional fitted C8 RegimeClassifier; loads from disk if None

        Returns
        -------
        equity_df : date-indexed DataFrame with columns:
          nav, daily_pnl, nav_s1, nav_s2, nav_s3, nav_combined,
          pnl_s1, pnl_s2, pnl_s3, pnl_combined, regime,
          s1_position, s2_position, s3_position, combined_pos
        """
        # ── Load data if not provided ──────────────────────────────────────────
        if spx_df is None or vix_wide is None:
            from joint_vol_calibration.data.database import (
                get_spx_ohlcv,
                get_vix_term_structure_wide,
            )
            spx_df   = get_spx_ohlcv()
            vix_wide = get_vix_term_structure_wide()

        # Normalise to have 'date' column
        spx_col  = self._to_df_with_date_col(self._to_date_indexed(spx_df))
        vix_col  = self._to_df_with_date_col(self._to_date_indexed(vix_wide))
        spx_col  = self._ensure_log_return(spx_col)

        # Date-indexed versions for fast lookup
        spx_idx  = self._to_date_indexed(spx_col)
        vix_idx  = self._to_date_indexed(vix_col)

        # ── Generate C9 signals ────────────────────────────────────────────────
        if clf is None:
            clf = self._load_clf()

        engine  = SignalEngine(clf=clf)
        sig_df  = engine.generate(
            spx_col, vix_col,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self.signals_df = sig_df

        # ── Build S3 z-score (matches C9 exactly) ─────────────────────────────
        z_ratio = _build_s3_zscore(vix_col).reindex(sig_df.index)

        # ── Fixed-nav sizing (each signal sizes off initial capital) ───────────
        # No compounding — conservative; avoids look-ahead in sizing
        nav_const = pd.Series(self.initial_capital, index=sig_df.index, dtype=float)

        spx_close = spx_idx["close"]

        # ── Simulate per-signal P&L ────────────────────────────────────────────
        pnl_s1, t1 = _simulate_straddle_pnl(
            sig_df["s1_position"], sig_df["s1_kelly"],
            nav_const, spx_close, vix_idx,
            r=self.r, q=self.q, signal_label="s1",
        )
        pnl_s2, t2 = _simulate_straddle_pnl(
            sig_df["s2_position"], sig_df["s2_kelly"],
            nav_const, spx_close, vix_idx,
            r=self.r, q=self.q, signal_label="s2",
        )
        pnl_s3, t3 = _simulate_s3_pnl(
            sig_df["s3_position"], sig_df["s3_kelly"],
            nav_const, z_ratio,
        )
        pnl_cb, t4 = _simulate_straddle_pnl(
            sig_df["combined_pos"], sig_df["combined_kelly"],
            nav_const, spx_close, vix_idx,
            r=self.r, q=self.q, signal_label="combined",
        )

        self.all_trades = t1 + t2 + t3 + t4

        # ── Build equity curves ────────────────────────────────────────────────
        # Portfolio = equal-weight average of 4 strategies (avoids 4× leverage)
        cap = self.initial_capital
        pnl_total = (pnl_s1 + pnl_s2 + pnl_s3 + pnl_cb) / 4.0

        eq = pd.DataFrame({
            "nav":          cap + pnl_total.cumsum(),
            "daily_pnl":    pnl_total,
            "nav_s1":       cap + pnl_s1.cumsum(),
            "nav_s2":       cap + pnl_s2.cumsum(),
            "nav_s3":       cap + pnl_s3.cumsum(),
            "nav_combined": cap + pnl_cb.cumsum(),
            "pnl_s1":       pnl_s1,
            "pnl_s2":       pnl_s2,
            "pnl_s3":       pnl_s3,
            "pnl_combined": pnl_cb,
        }, index=sig_df.index)

        # Attach signal columns
        for col in ["s1_position", "s2_position", "s3_position", "combined_pos", "regime"]:
            if col in sig_df.columns:
                eq[col] = sig_df[col]

        self.equity_df = eq
        logger.info(
            "Backtest complete: %d days | NAV $%.0f → $%.0f | trades: %d",
            len(eq), cap, float(eq["nav"].iloc[-1]), len(self.all_trades),
        )
        return eq

    # ── Metrics interface ──────────────────────────────────────────────────────

    def compute_metrics(self, equity_df: Optional[pd.DataFrame] = None) -> Dict:
        """Compute full performance metrics; also adds per-signal breakdown."""
        df = equity_df if equity_df is not None else self.equity_df
        if df is None:
            raise RuntimeError("No equity DataFrame. Call run() first.")

        m = compute_metrics(df, trade_log=self.all_trades)

        # Per-signal Sharpe and annualised return
        for sig, nav_col in [("s1","nav_s1"),("s2","nav_s2"),("s3","nav_s3"),("combined","nav_combined")]:
            if nav_col in df.columns:
                sub = pd.DataFrame({"nav": df[nav_col]})
                sm  = compute_metrics(sub)
                m[f"{sig}_sharpe"]     = sm.get("sharpe", np.nan)
                m[f"{sig}_ann_return"] = sm.get("ann_return", np.nan)

        # Per-signal trade stats
        from collections import defaultdict
        sig_trades: Dict[str, List] = defaultdict(list)
        for t in self.all_trades:
            sig_trades[t.signal].append(t.net_pnl)
        for sig, pnls in sig_trades.items():
            if pnls:
                m[f"{sig}_win_rate"]  = float(sum(1 for p in pnls if p > 0) / len(pnls))
                m[f"{sig}_avg_pnl"]  = float(np.mean(pnls))
                m[f"{sig}_n_trades"] = len(pnls)
                m[f"{sig}_total_pnl"] = float(sum(pnls))

        self.metrics_ = m
        return m

    def crisis_performance(self, equity_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """Compute performance metrics for each crisis period."""
        df = equity_df if equity_df is not None else self.equity_df
        if df is None:
            raise RuntimeError("No equity DataFrame. Call run() first.")
        return compute_crisis_performance(df)

    # ── Walk-forward validation ────────────────────────────────────────────────

    def walk_forward_validation(
        self,
        spx_df:        pd.DataFrame,
        vix_wide:      pd.DataFrame,
        n_windows:     int = WF_N_WINDOWS,
        window_months: int = WF_WINDOW_MONTHS,
    ) -> pd.DataFrame:
        """
        Expanding-window walk-forward validation.

        For each of 12 six-month OOS windows (2019 → 2024):
          1. Re-train C8 classifier on all data up to test_start − 1 day
          2. Run backtest on 6-month test window using retrained classifier
          3. Compute Sharpe and flag windows with Sharpe < 0

        Returns
        -------
        DataFrame with columns: window, train_end, test_start, test_end,
          sharpe, ann_return, max_drawdown, n_trades, below_zero_sharpe
        """
        spx_col = self._to_df_with_date_col(self._to_date_indexed(spx_df))
        vix_col = self._to_df_with_date_col(self._to_date_indexed(vix_wide))
        spx_col = self._ensure_log_return(spx_col)

        records = []
        test_starts = pd.date_range(start="2019-01-01", periods=n_windows, freq="6MS")

        for i, test_start in enumerate(test_starts):
            test_end  = test_start + pd.DateOffset(months=window_months) - pd.Timedelta(days=1)
            train_end = test_start - pd.Timedelta(days=1)

            test_end = min(test_end, pd.Timestamp(self.end_date))
            if test_start > pd.Timestamp(self.end_date):
                break

            logger.info(
                "WF window %d/%d: train→%s  test %s→%s",
                i + 1, n_windows, train_end.date(), test_start.date(), test_end.date(),
            )

            # Re-train C8 on expanding window
            try:
                clf = self._retrain_classifier(spx_col, vix_col, train_end)
            except Exception as exc:
                logger.warning("Classifier retrain failed (window %d): %s", i + 1, exc)
                clf = None

            # Run sub-backtest
            try:
                sub = BacktestEngine(
                    start_date      = test_start.strftime("%Y-%m-%d"),
                    end_date        = test_end.strftime("%Y-%m-%d"),
                    initial_capital = self.initial_capital,
                    r=self.r, q=self.q, seed=self.seed,
                )
                eq_sub = sub.run(spx_col, vix_col, clf=clf)
                m_sub  = sub.compute_metrics(eq_sub)
            except Exception as exc:
                logger.warning("WF window %d backtest failed: %s", i + 1, exc)
                m_sub = {"sharpe": np.nan, "ann_return": np.nan, "max_drawdown": np.nan, "n_trades": 0}

            sh = m_sub.get("sharpe", np.nan)
            records.append({
                "window":            i + 1,
                "train_end":         str(train_end.date()),
                "test_start":        str(test_start.date()),
                "test_end":          str(test_end.date()),
                "sharpe":            sh,
                "ann_return":        m_sub.get("ann_return", np.nan),
                "max_drawdown":      m_sub.get("max_drawdown", np.nan),
                "n_trades":          m_sub.get("n_trades", 0),
                "below_zero_sharpe": bool(pd.notna(sh) and sh < 0),
            })

        wf_df = pd.DataFrame(records)
        self._wf_results = wf_df
        return wf_df

    @staticmethod
    def _retrain_classifier(
        spx_df:    pd.DataFrame,
        vix_wide:  pd.DataFrame,
        train_end: pd.Timestamp,
    ) -> RegimeClassifier:
        """Re-train C8 XGBoost on data up to train_end (expanding window)."""
        feats  = build_features(spx_df, vix_wide)
        labels = build_regime_labels(spx_df, vix_wide)

        mask    = feats.index <= train_end
        X_train = feats[mask].dropna()
        y_train = labels.reindex(X_train.index).dropna()
        X_train = X_train.reindex(y_train.index)

        if len(X_train) < 50:
            raise ValueError(f"Insufficient training data: {len(X_train)} rows")

        clf = RegimeClassifier()
        clf.fit(X_train, y_train)
        return clf

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        """Save equity curve to parquet and engine state (trades, metrics) to pickle."""
        path = Path(path or RESULTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.equity_df is not None:
            self.equity_df.to_parquet(path)

        state = {
            "start_date":      self.start_date,
            "end_date":        self.end_date,
            "initial_capital": self.initial_capital,
            "r":               self.r,
            "q":               self.q,
            "seed":            self.seed,
            "metrics":         self.metrics_,
            "all_trades":      self.all_trades,
        }
        pkl_path = path.with_suffix(".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(state, f)

        logger.info("BacktestEngine saved → %s + %s", path, pkl_path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "BacktestEngine":
        """Load BacktestEngine from saved parquet + pickle."""
        path     = Path(path or RESULTS_PATH)
        pkl_path = path.with_suffix(".pkl")

        eq_df = pd.read_parquet(path)
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)

        engine            = cls(
            start_date      = state["start_date"],
            end_date        = state["end_date"],
            initial_capital = state["initial_capital"],
            r               = state.get("r", R_BACKTEST),
            q               = state.get("q", Q_BACKTEST),
            seed            = state.get("seed", RANDOM_SEED),
        )
        engine.equity_df  = eq_df
        engine.metrics_   = state.get("metrics")
        engine.all_trades = state.get("all_trades", [])
        return engine

    def __repr__(self) -> str:
        n   = len(self.equity_df) if self.equity_df is not None else 0
        nav = (f"${float(self.equity_df['nav'].iloc[-1]):,.0f}"
               if self.equity_df is not None and n > 0 else "not run")
        return (
            f"BacktestEngine(start={self.start_date}, end={self.end_date}, "
            f"capital=${self.initial_capital:,.0f}, days={n}, final_nav={nav})"
        )
