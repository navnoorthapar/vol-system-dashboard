"""
backtest_engine.py — C10: Full Backtest & Reporting Engine

Period:   2018-01-01 → 2025-03-24
Capital:  $1,000,000 (initial)
Signals:  portfolio  S1C (contrarian PDV), S3 (dispersion, index leg), S4 (VRP)
          references S1 (PDV–IV spread), S2 (VIX TS), Combined

Cost model (non-negotiable):
  Bid-ask:    0.15 vol pts per leg  → 0.15 × 0.01 × vega × n_contracts × 100
  Commission: $0.65 per contract per leg (2 legs per straddle)
  Margin:     20% of K × multiplier × n_contracts for short options
  Slippage:   0.05% of |hedge shares traded| × S on every delta-hedge trade,
              INCLUDING the opening hedge and the closing unwind

P&L methodology (every signal, S3 included, uses the same straddle engine):
  30-day ATM straddle, delta-hedged daily, self-financing:
    daily_pnl_t = direction × [ V_t − V_{t−1} − Δ_{t−1}·ΔS
                                + (r_{t−1}·(Δ_{t−1}·S_{t−1} − V_{t−1})
                                   − q·Δ_{t−1}·S_{t−1}) / 252 ] × n × 100
  The second line is the carry of the book: the cash account (−V + Δ·S)
  accrues r and the stock hedge pays the dividend yield. Without it the
  hedged P&L is not ½Γ(ΔS² − σ²S²Δt) even under Black-Scholes.
  A straddle that reaches expiry (21 sessions for a 30-day tenor) settles
  at intrinsic |S − K|; if the signal is still on it is rolled into a
  fresh ATM straddle the same close.

  S3 (long dispersion) is priced as the INDEX LEG of the dispersion trade:
  long dispersion ⇒ SHORT SPX straddle. There is no single-stock option
  data, so the single-stock leg is not modelled; S3's P&L is a proxy for
  half the trade, not a dispersion P&L.

Implied vol: ATM vol is not VIX. VIX is a 30-day variance-swap rate and
contains the skew/convexity premium, so it sits above ATM implied vol.
Straddles are priced and marked at
    σ_ATM(T) = ATM_TO_VIX_RATIO × VIX-TS(T)
(VIX term structure interpolated in total variance). ATM_TO_VIX_RATIO is
estimated from every real SPX chain available (see
estimate_atm_to_vix_ratio and run_backtest --estimate-beta).

Performance convention: the strategy runs on $1M of unfunded collateral
assumed to earn T-bills, so its P&L is already an EXCESS return over cash.
Sharpe/Sortino use rf = 0 on that P&L everywhere (headline, per signal,
crisis windows, sub-periods). Subtracting ^IRX again would charge the cash
hurdle twice. The T-bill path is reported separately as the cash
benchmark (nav_tbill, metrics['cash_cum_return']).

Assumptions (documented for transparency):
  1. 30-day ATM straddles; new entries and expiry rolls re-strike ATM
  2. Vol surface: β × VIX TS (ATM only, no smile). NO historical option
     prices exist in the DB — all option P&L is MARK-TO-MODEL (BS), which
     ignores skew dynamics and pin risk. The Heston/joint-calibration
     stack is NOT in this P&L path; it serves C6 (Greeks) and C7.
  3. Per-date r from ^IRX 3M T-bill (previous available print on gaps;
     fallback 0.045 only if the DB has no usable series), q = 0.013
  4. Regime labels: lagged deterministic rule (rv_20d vs VIX; VVIX gate →
     R2) built by regime_classifier.build_regime_labels with its default
     gate; the ML classifier is not in the trading loop
  5. Regime gate: no NEW entries in Regime 2; S1C/S4 also force-close open
     positions when the regime transitions INTO R2
  6. Sizing off each sleeve's own running NAV (through the prior close);
     short straddles capped at NAV/(0.20 × K × 100) contracts
  7. Ex-ante sizing overlay, shifted 1 day, applied in a second pass:
     correlation netting m_i = 1/√(1 + Σ_j max(ρ_ij, 0)) on trailing 60d
     P&L correlation, times a trailing-252d net-Sharpe scale in [0.5, 1]
  8. pdv_iv_spread = σ̂_PDV(t) − VIX(t): σ̂ is the walk-forward PDVLinear
     forecast of the next-21-session RMS vol made at the close of t (refit
     daily on labels observed by t, purged); lagged one session

The C22Corrections switches below exist only so c22_bridge.py can replay
the C22 correction round one fix at a time from the published C21 numbers.
Every default is the corrected behaviour.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.backtest.delta_hedger import (
    _bs_straddle_value,
    _bs_straddle_greeks,
    _interp_atm_iv,
    fit_pdv_before,
)
from joint_vol_calibration.signals.signal_engine import SignalEngine
from joint_vol_calibration.signals.regime_classifier import (
    RegimeClassifier,
    build_regime_labels,
)

logger = logging.getLogger(__name__)

# ── Backtest constants ─────────────────────────────────────────────────────────

BT_START:          str   = "2018-01-01"
BT_END:            str   = "2025-03-24"
INITIAL_CAPITAL:   float = 1_000_000.0
# Sharpe/Sortino hurdle on the strategy P&L. The P&L is an excess return
# (collateral assumed to earn T-bills), so the hurdle is zero. Before C22 a
# per-date ^IRX hurdle was subtracted from this unfunded P&L — charging
# cash twice (headline Sharpe −1.80 vs −1.14 on the same NAV).
SHARPE_RF:         float = 0.0
R_BACKTEST:        float = 0.045     # pricing rate used only if no ^IRX series is usable
Q_BACKTEST:        float = 0.013     # SPX dividend yield
TRADING_DAYS:      int   = 252

# Cost model — non-negotiable per spec
BID_ASK_VOL_PTS:   float = 0.15     # vol points per leg (0.15 × 0.01 = 0.0015 decimal)
COMMISSION_PER_LEG: float = 0.65    # $ per contract per leg
MARGIN_RATIO:      float = 0.20     # short-option initial margin as fraction of notional
SLIPPAGE_PCT:      float = 0.0005   # 0.05% of |hedge shares traded| × S
CONTRACT_MULT:     int   = 100      # SPX options: 100 shares per contract

# Straddle tenor
STRADDLE_TENOR_DAYS: int = 30       # 30-calendar-day ATM straddle at entry

# ATM implied vol per unit of VIX-term-structure vol: σ_ATM = β · VIX-TS.
# Estimated (C22) as the mean ratio of the 30-day ATM-forward implied vol
# (Black-76 on parity-implied forwards, total-variance interpolated to 30
# calendar days) to the same-time ^VIX, over every real SPX chain with a
# time-matched VIX print (run_backtest --estimate-beta --live):
#   DB 2026-03-24 0.824 · 2026-03-27 0.841 · 2026-05-29 0.813 ·
#   2026-06-18 0.814 · live Yahoo GTH chain 2026-10-12 04:38 ET 0.796
#   → mean 0.818, median 0.814, sd 0.017, n = 5. (A sixth, 2026-06-05
#   chain in the old spx_options table was excluded — its parity-implied
#   spot was 2.2% above that session's close, so its quotes were not from
#   the close whose VIX they would be compared with — and the table was
#   dropped from the DB in C22.) All five chains are from 2026; no
#   history of ATM IV exists in the DB, so β is assumed constant over
#   2018-2025. Pre-C22 the engine used β = 1 (VIX as ATM vol).
# The estimation set is frozen (BETA_ESTIMATION_SESSIONS): the daily bot now
# stores a chain every session, and a default that moved with each one would
# re-define the canonical backtest daily. Later chains are reported by
# --estimate-beta as out-of-sample checks: the first, 2026-10-16 (VVIX 110,
# an R2 day), gives 0.702, which would pull the all-chain mean to 0.80. β is
# therefore not constant across regimes; the β grid includes 0.80.
ATM_TO_VIX_RATIO:  float = 0.82
BETA_ESTIMATION_SESSIONS: Tuple[str, ...] = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18")

# Horizon (sessions) of the walk-forward PDV forecast behind pdv_iv_spread.
# 21 (canonical, C22): the forecast of the RMS vol over the next 21 sessions
# — variance-consistent and horizon-matched with the 30-day VIX — from
# regime_classifier.walk_forward_pdv_vol (PDVLinear refit DAILY on labels
# fully observed at the forecast origin; purged, embargo = 21).
# 1 (pre-C22, kept for c22_bridge): PDVLinear on |r_{t+1}|·√252, refit once
# a year on pre-year returns — an E|r| forecast (≈0.66× RMS vol on SPX), so
# "VIX > σ̂ + 2 vp" held on ~96% of 2018-2025 sessions.
PDV_SPREAD_HORIZON: int = 21

# S3 legacy z-score proxy parameters (pre-C22 P&L model, kept for the bridge)
S3_Z_SCALE:        float = 0.02    # 2% of kelly-weighted nav per unit z improvement
S3_PROXY_COST:     float = 0.001   # 0.1% of notional at entry/exit (execution proxy)

# Ex-ante sizing overlay
CORR_WINDOW:       int   = 60      # trailing days for pairwise P&L correlation
NET_SHARPE_WINDOW: int   = 252     # trailing days for the net-Sharpe scale

# Sub-period stability windows
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

PORTFOLIO_SIGNALS: Tuple[str, ...] = ("s1c", "s3", "s4")
REFERENCE_SIGNALS: Tuple[str, ...] = ("s1", "s2", "combined")

# Sentinel: "use regime_classifier.build_regime_labels' own default gate"
_DEFAULT_GATE = object()

# Straddle direction of the S3 index leg per unit of s3_position (+1 = the
# signal is on). −1: long dispersion ⇒ SHORT SPX straddle (the C22 reading:
# a dispersion trade is short index vol). The signal's own documentation may
# read +1 as "the VIX/VVIX ratio reverts up", i.e. long index vol; pass
# s3_leg_direction=+1 to price that reading instead.
S3_LEG_DIRECTION: int = -1


# ── C22 correction switches ────────────────────────────────────────────────────

@dataclass(frozen=True)
class C22Corrections:
    """
    Switches for the C22 correction round (audit finding id in brackets).

    Every default is the corrected behaviour. ``C22Corrections.legacy()``
    turns them all off and, together with ``atm_to_vix_ratio=1.0`` and the
    fixed VVIX>100 gate, reproduces the published C21 headline
    (NAV $738,794 / P&L −$261,206 / Sharpe −1.80 / MaxDD −31.5%).
    They exist so c22_bridge.py can apply the corrections one at a time;
    nothing else should switch them off.
    """
    sharpe_rf_zero:    bool = True   # Sharpe on excess P&L, rf=0 [backtest-sharpe-double-counts-rf]
    rate_ffill:        bool = True   # previous ^IRX print on gaps, not 4.5% [backtest-rate-missing-dates-fallback]
    s3_exit_day:       bool = True   # S3 proxy holds entry size through exit day [backtest-s3-exit-day-dropped]
    hedge_costs:       bool = True   # opening hedge + closing unwind, no phantom exit rebalance [backtest-hedge-entry-unwind-cost-missing]
    financing:         bool = True   # r on cash (−V+ΔS), q on stock leg [backtest-no-premium-hedge-financing]
    expiry_roll:       bool = True   # settle at expiry, roll if signal on [backtest-no-expiry-roll]
    open_trade_log:    bool = True   # trade still open at the end is logged [backtest-open-trade-and-sizing-bookkeeping]
    sleeve_nav_sizing: bool = True   # size/margin off the sleeve's running NAV [backtest-open-trade-and-sizing-bookkeeping]
    corr_netting:      bool = True   # correlation netting that can reduce size [backtest-corr-netting-noop]
    s3_index_leg:      bool = True   # S3 priced as short SPX straddle, not z-units [signals-s3-mechanical-zscore-pnl]

    @classmethod
    def legacy(cls) -> "C22Corrections":
        """All corrections off — the published C21 engine."""
        return cls(**{f.name: False for f in fields(cls)})

    def with_(self, **changes) -> "C22Corrections":
        """Copy with some switches changed."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d.update(changes)
        return C22Corrections(**d)


# ── Trade record ───────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """
    One straddle trade (or one S3 proxy trade).

    ``direction`` is the STRADDLE direction (+1 long, −1 short). For S3 the
    signal is long dispersion, which the index leg trades by default as a
    SHORT SPX straddle (direction −1, see S3_LEG_DIRECTION); the legacy
    z-proxy records +1.
    ``is_open`` marks the trade still open on the last backtest date; its
    exit_date is that date, its P&L is mark-to-model and no closing cost
    has been charged.
    """
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
    slippage_cost: float = 0.0    # all delta-hedge trading costs (open, rebalances, unwind)
    financing:     float = 0.0    # carry of the cash account and the stock hedge
    exit_reason:   str   = ""     # "signal" | "expiry" | "open"
    is_open:       bool  = False

    @property
    def net_pnl(self) -> float:
        return (self.gross_pnl + self.financing
                - self.entry_cost - self.exit_cost - self.slippage_cost)

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


def _hedge_cost(delta_shares: float, S: float, n_contracts: int) -> float:
    """Slippage on a delta-hedge trade of |delta_shares| per straddle."""
    return SLIPPAGE_PCT * abs(delta_shares) * S * n_contracts * CONTRACT_MULT


def _atm_sigma(vix_row: pd.Series, T: float, atm_to_vix_ratio: float) -> float:
    """σ_ATM(T) = β · VIX-TS(T); 20% if the VIX row is unusable."""
    sigma = atm_to_vix_ratio * float(_interp_atm_iv(vix_row, T))
    if not (sigma > 0) or np.isnan(sigma):
        sigma = 0.20
    return sigma


def _build_s3_zscore(vix_wide: pd.DataFrame, lookback: int = 252) -> pd.Series:
    """
    VIX/VVIX ratio z-score on the full unshifted history (legacy S3 proxy P&L).

    The S3 TRADING signal is computed in signal_engine.generate_signal3 on
    1-day-shifted features; this series only drives the legacy proxy's Δz.
    """
    idx = vix_wide.set_index("date") if "date" in vix_wide.columns else vix_wide
    idx.index = pd.to_datetime(idx.index)
    ratio = (idx["^VIX"] / idx["^VVIX"].clip(lower=1e-6)).clip(lower=1e-6)
    mu    = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    std   = ratio.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    return ((ratio - mu) / std).rename("z_ratio")


def _resolve_rates(
    r: "float | pd.Series",
    dates: pd.Index,
    ffill: bool,
) -> np.ndarray:
    """
    Per-date pricing rate.

    ffill=True  : the previous available print on dates the series misses
                  (e.g. sessions the ^IRX download skipped); negative or zero
                  yields are kept as published. Dates before the first print
                  use R_BACKTEST.
    ffill=False : legacy — any missing date priced at R_BACKTEST (4.5%).
    """
    if not isinstance(r, pd.Series):
        return np.full(len(dates), float(r))
    s = r.copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    if ffill:
        full = s.reindex(s.index.union(pd.DatetimeIndex(dates))).ffill()
        return full.reindex(dates).fillna(R_BACKTEST).values
    return np.array([float(s.get(d, R_BACKTEST)) for d in dates])


def rate_series_spans_window(dates: "pd.Series | pd.Index", start: str, end: str) -> bool:
    """
    True if a rate series with these print dates covers [start, end]: its
    first print inside the window is within 10 days of ``start``, its last
    print within 10 days of ``end``, and it has at least 80% of the
    window's business days.

    An emptiness check is not enough: the shipped DB once held exactly 7
    T-bill rows inside the window (a unit-test fixture that leaked into the
    tracked database); forward-filled, those 7 synthetic days became the
    rate for five years (C21).
    """
    d = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates))).sort_values()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    inside = d[d >= start_ts]
    if len(inside) == 0:
        return False
    span_days = max((end_ts - start_ts).days, 1)
    expected = span_days / 7 * 5 * 0.80
    return bool(
        inside[0] <= start_ts + pd.Timedelta(days=10)
        and d[-1] >= end_ts - pd.Timedelta(days=10)
        and len(inside) >= expected
    )


# ── Core P&L simulators ────────────────────────────────────────────────────────

@dataclass
class _OpenStraddle:
    direction:   int
    K:           float
    S_entry:     float
    sigma_entry: float
    n:           int
    kelly:       float
    entry_date:  pd.Timestamp
    entry_cost:  float
    V:           float
    delta:       float
    S:           float
    r:           float
    days_biz:    int = 0
    gross:       float = 0.0
    slip:        float = 0.0
    fin:         float = 0.0


def _simulate_straddle_pnl(
    position:     pd.Series,      # index=date, values ∈ {−1, 0, +1} (or fractional; sign = direction)
    kelly:        pd.Series,      # index=date, values ∈ [0, 0.20]
    nav_series:   pd.Series,      # index=date, sizing capital (see corrections.sleeve_nav_sizing)
    spx_close:    pd.Series,      # index=date, SPX close price
    vix_wide_idx: pd.DataFrame,   # date-indexed wide VIX DataFrame
    r:            "float | pd.Series" = R_BACKTEST,   # scalar or date-indexed Series
    q:            float = Q_BACKTEST,
    tenor_days:   int   = STRADDLE_TENOR_DAYS,
    signal_label: str   = "s",
    atm_to_vix_ratio: float = ATM_TO_VIX_RATIO,
    corrections:  Optional[C22Corrections] = None,
) -> Tuple[pd.Series, List[TradeRecord]]:
    """
    Simulate delta-hedged 30-day ATM straddle P&L for a position series.

    Timing: the position on day t is decided from t−1 information and
    traded at the close of t. On an entry day the straddle is bought/sold
    and hedged at the close; each later day books the move from the prior
    close; on an exit day (position 0) the straddle and hedge are closed at
    the close.

    Daily P&L (C7 framework, Greeks from the prior close):
      direction × (V_t − V_{t−1} − Δ_{t−1}·ΔS_t) × n × 100        mark-to-model
      + direction × (r_{t−1}(Δ_{t−1}S_{t−1} − V_{t−1})
                     − qΔ_{t−1}S_{t−1}) / 252 × n × 100            financing
      − 5 bp × |shares traded| × S                                 hedge costs
    with σ = ATM_TO_VIX_RATIO × VIX-TS(T_rem). Entry/exit pay the option
    bid-ask and commission; the hedge pays slippage when opened, on every
    rebalance and when unwound. A straddle reaching expiry settles at
    |S − K| (no option exit cost) and, if the signal is still on, is
    rolled into a fresh ATM straddle the same close.

    ``nav_series`` is the sizing capital. With corrections.sleeve_nav_sizing
    the sizing/margin NAV on day t is nav_series[t] plus this sleeve's own
    cumulative P&L through the close of t−1, i.e. the sleeve's running NAV.

    The sum of ``net_pnl`` over the returned trades equals ``pnl.sum()``
    (the trade still open at the end is included as ``is_open``).
    """
    c = corrections or C22Corrections()
    T_entry   = tenor_days / 365.0
    n_expiry  = max(1, int(np.ceil(T_entry * TRADING_DAYS - 1e-9)))   # sessions to expiry
    dates     = position.index
    pnl       = pd.Series(0.0, index=dates, dtype=float)
    trades: List[TradeRecord] = []
    rates     = _resolve_rates(r, dates, ffill=c.rate_ffill)

    st: Optional[_OpenStraddle] = None
    cum_pnl   = 0.0
    last_date = None

    def _record(s: _OpenStraddle, exit_date, exit_cost: float, reason: str, is_open=False):
        trades.append(TradeRecord(
            signal=signal_label, direction=s.direction,
            entry_date=s.entry_date, exit_date=exit_date,
            K=s.K, S_entry=s.S_entry, sigma_entry=s.sigma_entry,
            n_contracts=s.n, kelly_frac=s.kelly,
            entry_cost=s.entry_cost, exit_cost=exit_cost,
            gross_pnl=s.gross, slippage_cost=s.slip, financing=s.fin,
            exit_reason=reason, is_open=is_open,
        ))

    def _open(direction, date, S, vix_row, r_t, k, nav) -> Tuple[Optional[_OpenStraddle], float]:
        """Open a straddle at today's close; returns (state, cash cost)."""
        if c.sleeve_nav_sizing and nav <= 0:
            return None, 0.0          # sleeve wiped out: nothing to size off
        sigma = _atm_sigma(vix_row, T_entry, atm_to_vix_ratio)
        K  = S
        V0 = max(float(_bs_straddle_value(S, K, T_entry, r_t, q, sigma)), 1e-4)
        n  = max(1, int(k * nav / (V0 * CONTRACT_MULT)))
        # Margin cap on short straddles: initial margin = 20% of strike
        # notional per contract must fit within NAV. Long straddles are
        # premium-capped by construction (k·nav sizing above).
        if direction < 0:
            n_margin_cap = int(nav / (MARGIN_RATIO * K * CONTRACT_MULT))
            if n_margin_cap < 1:
                return None, 0.0      # NAV cannot support even one short contract
            n = min(n, n_margin_cap)
        entry_cost = _one_way_cost(_straddle_vega(S, K, T_entry, r_t, q, sigma), n)
        delta0 = float(_bs_straddle_greeks(S, K, T_entry, r_t, q, sigma)[0])
        hedge0 = _hedge_cost(delta0, S, n) if c.hedge_costs else 0.0
        s = _OpenStraddle(
            direction=direction, K=K, S_entry=S, sigma_entry=sigma, n=n, kelly=k,
            entry_date=date, entry_cost=entry_cost, V=V0, delta=delta0, S=S, r=r_t,
            slip=hedge0,
        )
        return s, entry_cost + hedge0

    for i, date in enumerate(dates):
        raw_pos = float(position.iloc[i])
        pos     = int(np.sign(raw_pos))   # collapse fractional → {−1, 0, +1}
        k       = float(kelly.iloc[i])
        nav     = float(nav_series.iloc[i]) + (cum_pnl if c.sleeve_nav_sizing else 0.0)

        # Defensive: a direct sign flip while holding (+1→−1 with no flat day)
        # is treated as a close today; the opposite leg opens on the next
        # flat→entry day. The state machines force a flat day between
        # opposite positions, so this never triggers today.
        if st is not None and pos != 0 and pos != st.direction:
            pos = 0

        if date not in spx_close.index or date not in vix_wide_idx.index:
            continue

        r_t     = float(rates[i])
        S       = float(spx_close.loc[date])
        vix_row = vix_wide_idx.loc[date]
        day     = 0.0
        last_date = date

        if st is not None:
            # ── HOLD / EXIT / EXPIRY ──────────────────────────────────────────
            # days_biz still holds the count as of the previous bar; one more
            # session has elapsed, hence days_biz + 1 (C21: without it the
            # first holding day carried no time decay at all).
            days_biz = st.days_biz
            T_raw    = T_entry - (days_biz + 1) / TRADING_DAYS
            expired  = c.expiry_roll and (days_biz + 1) >= n_expiry
            T_rem    = max(T_raw, 0.0) if c.expiry_roll else max(T_raw, 1.0 / 365.0)
            st.days_biz = days_biz + 1
            mult = st.n * CONTRACT_MULT

            if expired:
                sigma   = st.sigma_entry
                V_t     = abs(S - st.K)
                delta_t = 0.0
            else:
                sigma   = _atm_sigma(vix_row, T_rem, atm_to_vix_ratio)
                V_t     = float(_bs_straddle_value(S, st.K, T_rem, r_t, q, sigma))
                delta_t = float(_bs_straddle_greeks(S, st.K, T_rem, r_t, q, sigma)[0])

            raw = st.direction * (V_t - st.V - st.delta * (S - st.S)) * mult
            fin = 0.0
            if c.financing:
                fin = (st.direction
                       * (st.r * (st.delta * st.S - st.V) - q * st.delta * st.S)
                       / TRADING_DAYS * mult)

            closing = expired or pos == 0
            if closing and c.hedge_costs:
                slip = _hedge_cost(st.delta, S, st.n)          # unwind the hedge
            else:
                slip = _hedge_cost(delta_t - st.delta, S, st.n)  # rebalance (legacy: also on exit)

            day      += raw + fin - slip
            st.gross += raw
            st.fin   += fin
            st.slip  += slip
            st.V, st.delta, st.S, st.r = V_t, delta_t, S, r_t

            if closing:
                exit_cost = 0.0 if expired else _one_way_cost(
                    _straddle_vega(S, st.K, T_rem, r_t, q, sigma), st.n)
                day -= exit_cost
                _record(st, date, exit_cost, "expiry" if expired else "signal")
                st = None
                if expired and pos != 0:
                    # ── ROLL: signal still on → fresh ATM straddle today ─────
                    st, cost = _open(pos, date, S, vix_row, r_t, k, nav)
                    day -= cost

        elif pos != 0:
            # ── ENTRY ─────────────────────────────────────────────────────────
            st, cost = _open(pos, date, S, vix_row, r_t, k, nav)
            day -= cost

        pnl.iloc[i] += day
        cum_pnl += day

    if st is not None and c.open_trade_log:
        _record(st, last_date, 0.0, "open", is_open=True)

    return pnl, trades


def _simulate_s3_pnl(
    position:   pd.Series,    # index=date, values ∈ {0, +1} (long-only)
    kelly:      pd.Series,
    nav_series: pd.Series,
    z_ratio:    pd.Series,    # precomputed z-score (date-indexed)
    corrections: Optional[C22Corrections] = None,
) -> Tuple[pd.Series, List[TradeRecord]]:
    """
    LEGACY z-score proxy P&L for Signal 3 (pre-C22; kept for c22_bridge).

    P&L = position × kelly × nav × Δz_ratio × S3_Z_SCALE (2% per z-unit);
    entry/exit cost 0.1% of notional. No option is priced.

    DO NOT read this as evidence of edge: z is a standardised statistic
    that drifts back toward 0 through its own rolling normalisation, so
    "enter at z<−1, exit at z>−0.3" books positive P&L on pure random walks
    (~98% of seeded random-walk paths; see s3_timing_null_test). The engine
    now prices S3 as the short-SPX-straddle index leg instead.

    With corrections.s3_exit_day the notional is fixed at entry (kelly at
    entry × NAV at entry) and held through the exit day, which books that
    day's Δz and pays the exit cost on the entry notional. Before C22 the
    exit day read that day's kelly, which is 0 once the signal is flat, so
    the last day's P&L and the exit cost were silently dropped.
    """
    c = corrections or C22Corrections()
    dates  = position.index
    pnl    = pd.Series(0.0, index=dates, dtype=float)
    trades: List[TradeRecord] = []

    in_trade    = False
    entry_date  = None
    entry_cost_ = 0.0
    notional_0  = 0.0
    k_in        = 0.0
    trade_gross = 0.0
    cum_pnl     = 0.0

    for i, date in enumerate(dates):
        pos = int(np.sign(float(position.iloc[i])))
        k   = float(kelly.iloc[i])
        nav = float(nav_series.iloc[i]) + (cum_pnl if c.sleeve_nav_sizing else 0.0)
        z   = float(z_ratio.loc[date]) if date in z_ratio.index else 0.0
        day = 0.0

        if not in_trade and pos != 0:
            k_in         = k
            notional_0   = k * max(nav, 0.0)
            entry_cost_  = S3_PROXY_COST * notional_0
            day         -= entry_cost_
            in_trade     = True
            entry_date   = date
            trade_gross  = 0.0

        elif in_trade:
            # Δz = change in z-score today
            if i > 0 and dates[i - 1] in z_ratio.index:
                dz = z - float(z_ratio.loc[dates[i - 1]])
            else:
                dz = 0.0
            if not np.isfinite(dz):
                dz = 0.0   # missing z-ratio (NaN gap) → no P&L that day, not NaN NAV

            notional    = notional_0 if c.s3_exit_day else k * nav
            raw_pnl_t   = notional * dz * S3_Z_SCALE   # long-only
            day        += raw_pnl_t
            trade_gross += raw_pnl_t

            if pos == 0:
                notional_exit = notional_0 if c.s3_exit_day else k * nav
                exit_cost_    = S3_PROXY_COST * notional_exit
                day          -= exit_cost_

                trades.append(TradeRecord(
                    signal="s3", direction=1,
                    entry_date=entry_date, exit_date=date,
                    K=0.0, S_entry=notional_0, sigma_entry=0.0,
                    n_contracts=0, kelly_frac=k_in if c.s3_exit_day else k,
                    entry_cost=entry_cost_, exit_cost=exit_cost_,
                    gross_pnl=trade_gross, slippage_cost=0.0,
                    exit_reason="signal",
                ))
                in_trade = False

        pnl.iloc[i] += day
        cum_pnl += day

    if in_trade and c.open_trade_log:
        trades.append(TradeRecord(
            signal="s3", direction=1,
            entry_date=entry_date, exit_date=dates[-1],
            K=0.0, S_entry=notional_0, sigma_entry=0.0,
            n_contracts=0, kelly_frac=k_in,
            entry_cost=entry_cost_, exit_cost=0.0,
            gross_pnl=trade_gross, slippage_cost=0.0,
            exit_reason="open", is_open=True,
        ))

    return pnl, trades


# ── Ex-ante sizing overlay (pass 2) ────────────────────────────────────────────

def _corr_netting_multipliers(
    pnl_mat: pd.DataFrame,
    window: int = CORR_WINDOW,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Correlation-netting size multiplier per sleeve (NOT shifted; caller shifts).

        m_i,t = 1 / sqrt(1 + Σ_{j≠i} max(ρ_ij,t, 0))

    with ρ the trailing ``window``-day correlation of daily P&L. For N
    sleeves with equal vol and common correlation ρ ≥ 0 this keeps the
    equal-weight portfolio variance at the uncorrelated level
    (m² · (N + N(N−1)ρ) = N). Uncorrelated or anti-correlated sleeves keep
    full size (no upsizing); perfectly correlated ones get 1/√N.
    Undefined correlations (a sleeve flat over the window) count as 0.
    """
    cols = list(pnl_mat.columns)
    out = pd.DataFrame(1.0, index=pnl_mat.index, columns=cols)
    excess = {c: pd.Series(0.0, index=pnl_mat.index) for c in cols}
    for a_i, a in enumerate(cols):
        for b in cols[a_i + 1:]:
            rho = (pnl_mat[a].rolling(window, min_periods=min_periods)
                   .corr(pnl_mat[b]).clip(-1.0, 1.0))
            # rolling().corr can return ±inf/NaN when a window is flat
            rho = rho.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
            excess[a] = excess[a] + rho
            excess[b] = excess[b] + rho
    for c in cols:
        out[c] = 1.0 / np.sqrt(1.0 + excess[c])
    return out


def _legacy_corr_kelly_mult(a: pd.Series, b: pd.Series, window: int = CORR_WINDOW) -> pd.Series:
    """
    Pre-C22 pairwise multiplier sqrt(2/(1+ρ)). It is ≥ 1 for every ρ in
    [−1, 1], so after the caller's clip(0.5, 1.0) it is identically 1.0 —
    the advertised netting never changed a position. Kept for the bridge.
    """
    rho = a.rolling(window, min_periods=20).corr(b).clip(-1.0, 1.0)
    return np.sqrt(2.0 / (1.0 + rho.fillna(0.0)))


def _net_kelly_scale(pnl: pd.Series, window: int = NET_SHARPE_WINDOW) -> pd.Series:
    """
    Trailing net-Sharpe size scale (NOT shifted; caller shifts).
    Sharpe ≥ 0 → 1.0 | Sharpe = −1 → 0.75 | Sharpe ≤ −2 → 0.5. Never upsizes.
    """
    mu  = pnl.rolling(window, min_periods=90).mean()
    sig = pnl.rolling(window, min_periods=90).std().clip(lower=1e-8)
    sh  = (mu / sig * np.sqrt(TRADING_DAYS)).fillna(0.0)
    return (1.0 + sh.clip(-2.0, 0.0) / 4.0).clip(0.5, 1.0)


# ── Performance metrics ────────────────────────────────────────────────────────

def compute_metrics(
    equity_df: pd.DataFrame,
    rf:        float = SHARPE_RF,
    trade_log: Optional[List[TradeRecord]] = None,
    rf_series: Optional[pd.Series] = None,
    base_nav:  Optional[float] = None,
) -> Dict:
    """
    Compute full set of performance metrics from an equity curve.

    Parameters
    ----------
    equity_df : DataFrame with column 'nav' (portfolio value, date-indexed)
    rf        : annualised hurdle subtracted from daily returns for
                Sharpe/Sortino. Default 0: the backtest NAV is unfunded
                collateral + strategy P&L, i.e. already an excess return
                over cash (collateral assumed to earn T-bills), so any
                positive hurdle would charge cash twice.
    trade_log : optional list of TradeRecord objects for trade-level stats
    rf_series : optional date-indexed annualised rates. Supply it only for a
                FUNDED NAV (one that already accrues interest on its cash);
                daily excess returns then use the per-date rate.
    base_nav  : the NAV at the close BEFORE the curve's first row: the
                initial capital for a full run, the last NAV before the
                window for a slice of a longer curve. Every row is then a
                session with a return measured from the row before it (the
                first from base_nav), so the first session's P&L (e.g. the
                entry costs of a position opened on it) counts in the
                returns, Sharpe, drawdown and cumulative return, and
                n_days = len(nav). BacktestEngine.compute_metrics always
                passes it.
                Default None (backward compatible): the first row IS the
                base — a starting value, not a session — so its own P&L is
                not measured and n_days = len(nav) − 1. Right only for a
                curve whose first row is the capital (or the NAV before the
                window), e.g. a slice with the prior session's NAV prepended.

    Returns
    -------
    dict with: ann_return, sharpe, sortino, max_drawdown, dd_duration_days,
               calmar, win_rate, avg_pnl_per_trade, worst_day, best_day,
               n_trades, n_days, cumulative_return
    """
    nav  = equity_df["nav"].dropna()
    if base_nav is None:
        rets  = nav.pct_change().dropna()
        start = float(nav.iloc[0]) if len(nav) else np.nan
    else:
        start = float(base_nav)
        if not (np.isfinite(start) and start > 0):
            raise ValueError(f"base_nav must be a positive NAV, got {base_nav!r}")
        prev = nav.shift(1)
        if len(prev):
            prev.iloc[0] = start
        rets = nav / prev - 1.0
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

    if (rf_series is not None and len(rf_series) > 0
            and not rate_series_spans_window(rf_series.index, rets.index[0], rets.index[-1])):
        # Forward-filling a few prints across the window is how seven fixture
        # rows became a five-year hurdle (C21): refuse it, use the scalar.
        logger.warning("compute_metrics: rf_series does not span %s..%s; using rf=%.4f",
                       rets.index[0].date(), rets.index[-1].date(), rf)
        rf_series = None
    if rf_series is not None and len(rf_series) > 0:
        rf_daily = (
            rf_series.reindex(rets.index).ffill().fillna(rf) / TRADING_DAYS
        )
    else:
        rf_daily = rf / TRADING_DAYS
    growth   = float(nav.iloc[-1] / start)
    # A NAV at or below zero has lost everything; a fractional power of a
    # non-positive ratio is NaN, so report the total loss instead.
    ann_ret  = growth ** (TRADING_DAYS / n) - 1.0 if growth > 0 else -1.0
    excess   = rets - rf_daily
    ex_mu, ex_sd = float(excess.mean()), float(excess.std())
    if ex_sd > 1e-12 * max(abs(ex_mu), 1e-12):
        sharpe = ex_mu / ex_sd * np.sqrt(TRADING_DAYS)
    else:
        # Zero-variance excess return: the ratio is ±∞ (or 0 for a zero mean),
        # not 0 — a riskless positive return must not read as "no edge".
        sharpe = float(np.sign(ex_mu) * np.inf) if ex_mu != 0 else 0.0

    # Sortino denominator = downside deviation about the target (rf), i.e.
    # sqrt(mean(min(0, excess)^2)) over ALL days — not the sample std of the
    # negative subset about its own mean (which understates downside risk).
    downside_dev = float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))
    if downside_dev > 0:
        sortino = float(excess.mean() / downside_dev * np.sqrt(TRADING_DAYS))
    else:
        # No day below the target: same convention as the Sharpe above.
        sortino = float(np.sign(ex_mu) * np.inf) if ex_mu != 0 else 0.0

    roll_max = nav.cummax()
    if base_nav is not None:
        roll_max = roll_max.clip(lower=start)     # the peak includes the base
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
    cum_ret  = float(nav.iloc[-1] / start - 1.0)

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
    base_nav:  Optional[float] = None,
) -> Dict[str, Dict]:
    """
    Performance metrics for each named crisis period.

    Each window is measured from the NAV at the close before its first
    session: the curve's last NAV before the window or, for a window that
    starts on the curve's first row, ``base_nav`` (the NAV before that row,
    e.g. the initial capital). The window's first session therefore counts
    in its return, drawdown and Sharpe. Only a window that starts on the
    first row with no base_nav falls back to its first NAV as the base.
    (Before this was fixed the base was always the window's first NAV, which
    dropped the first session: on the canonical run COVID_2020 read +1.76%
    for +1.69% and FedHikes_2022 +5.12% for +4.71%.)

    Sharpe uses rf = 0 on the strategy's excess P&L, the same convention as
    compute_metrics' default.
    """
    results = {}
    nav_all = equity_df["nav"]
    for name, (start, end) in periods.items():
        s, e  = pd.Timestamp(start), pd.Timestamp(end)
        sub   = nav_all.loc[(nav_all.index >= s) & (nav_all.index <= e)].dropna()
        prior = nav_all.loc[nav_all.index < s].dropna()
        base  = float(prior.iloc[-1]) if len(prior) else base_nav
        nav   = (sub.reset_index(drop=True) if base is None
                 else pd.Series(np.r_[float(base), sub.to_numpy(dtype=float)]))
        if len(sub) == 0 or len(nav) < 2:
            results[name] = {"n_days": 0, "note": "No data in period"}
            continue
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


# ── ATM-vol / VIX ratio estimation ─────────────────────────────────────────────

def _black76_price(F: float, K: float, T: float, DF: float, sigma: float, is_call: bool) -> float:
    """Black-76 price of a European option on forward F, discounted by DF."""
    sd = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if is_call:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _black76_implied_vol(price: float, F: float, K: float, T: float, DF: float,
                         is_call: bool) -> float:
    """Black-76 implied vol by Brent; NaN outside the no-arbitrage bounds."""
    intrinsic = DF * max((F - K) if is_call else (K - F), 0.0)
    upper = DF * (F if is_call else K)
    if not np.isfinite(price) or price <= intrinsic + 1e-10 or price >= upper:
        return np.nan
    try:
        return float(brentq(lambda s: _black76_price(F, K, T, DF, s, is_call) - price,
                            1e-4, 5.0, xtol=1e-10))
    except ValueError:
        return np.nan


def atm_forward_term_structure(
    chain:     pd.DataFrame,
    as_of:     pd.Timestamp,
    r:         float,
    min_days:  float = 7.0,
    max_days:  float = 75.0,
    n_forward: int   = 3,
) -> pd.DataFrame:
    """
    ATM-forward implied vol per expiry from raw SPX quotes.

    chain : columns expiry, strike, right ('C'/'P'), bid, ask
    as_of : quote time (naive, US/Eastern); expiries are taken at 16:00 ET
    r     : continuously-compounded rate for discounting (^IRX)

    Per expiry (Cboe VIX-white-paper construction):
      • DF = exp(−rT); forward from put-call parity, F = K + (C − P)/DF,
        averaged over the ``n_forward`` strikes with the smallest |C − P|
        (mid quotes, both legs bid > 0)
      • Black-76 implied vol of call and put at the strikes bracketing F,
        averaged per strike, then interpolated linearly in log-moneyness
        to K = F.
    Returns columns expiry, T (years, 365-day), F, atm_iv.
    """
    as_of = pd.Timestamp(as_of)
    out = []
    for expiry, g in chain.groupby("expiry"):
        T = ((pd.Timestamp(expiry) + pd.Timedelta(hours=16) - as_of).total_seconds()
             / (365.0 * 86400.0))
        if not (min_days / 365.0 <= T <= max_days / 365.0):
            continue
        g = g[(g["bid"] > 0) & (g["ask"] >= g["bid"])]
        mid = (g["bid"] + g["ask"]) / 2.0
        calls = mid[g["right"] == "C"].groupby(g.loc[g["right"] == "C", "strike"]).mean()
        puts  = mid[g["right"] == "P"].groupby(g.loc[g["right"] == "P", "strike"]).mean()
        ks = calls.index.intersection(puts.index).sort_values()
        if len(ks) < max(n_forward, 2):
            continue
        DF = float(np.exp(-r * T))
        diff = (calls[ks] - puts[ks])
        nearest = diff.abs().sort_values().index[:n_forward]
        F = float(np.mean([k + diff[k] / DF for k in nearest]))
        below, above = ks[ks <= F], ks[ks > F]
        if len(below) == 0 or len(above) == 0:
            continue
        k1, k2 = float(below.max()), float(above.min())
        ivs = []
        for k in (k1, k2):
            ic = _black76_implied_vol(float(calls[k]), F, k, T, DF, True)
            ip = _black76_implied_vol(float(puts[k]),  F, k, T, DF, False)
            ivs.append(np.nanmean([ic, ip]) if np.isfinite([ic, ip]).any() else np.nan)
        if not np.all(np.isfinite(ivs)):
            continue
        x1, x2 = np.log(k1 / F), np.log(k2 / F)
        atm = ivs[0] + (ivs[1] - ivs[0]) * (0.0 - x1) / (x2 - x1)
        out.append({"expiry": str(pd.Timestamp(expiry).date()), "T": T, "F": F, "atm_iv": float(atm)})
    return pd.DataFrame(out, columns=["expiry", "T", "F", "atm_iv"])


def atm_iv_at_tenor(term: pd.DataFrame, target_days: float = 30.0) -> Tuple[float, str, str]:
    """
    ATM vol at ``target_days`` calendar days: linear interpolation in total
    variance σ²T between the nearest expiries below and above (the VIX
    construction). Returns (σ, expiry_below, expiry_above).
    """
    T30 = target_days / 365.0
    lo, hi = term[term["T"] <= T30], term[term["T"] > T30]
    if lo.empty or hi.empty:
        raise ValueError(f"no expiries bracket {target_days:g} days")
    a = lo.loc[lo["T"].idxmax()]
    b = hi.loc[hi["T"].idxmin()]
    w1, w2 = a["atm_iv"] ** 2 * a["T"], b["atm_iv"] ** 2 * b["T"]
    w = w1 + (w2 - w1) * (T30 - a["T"]) / (b["T"] - a["T"])
    return float(np.sqrt(w / T30)), str(a["expiry"]), str(b["expiry"])


def estimate_atm_to_vix_ratio(
    chain: pd.DataFrame,
    as_of: pd.Timestamp,
    vix:   float,
    r:     float,
    q:     float = Q_BACKTEST,
) -> Dict:
    """
    β = σ_ATM,30d / VIX for one SPX chain and a time-matched ^VIX print.

    ``vix`` is in index points (e.g. 15.39). Also returns the spot implied
    by the 30-day forward, S = F·exp(−(r−q)T), so callers can check that the
    quotes really belong to the session whose VIX they are compared with.
    """
    term = atm_forward_term_structure(chain, as_of, r)
    atm30, e_lo, e_hi = atm_iv_at_tenor(term, 30.0)
    near = term.iloc[(term["T"] - 30.0 / 365.0).abs().argsort()[:1]].iloc[0]
    s_impl = float(near["F"] * np.exp(-(r - q) * near["T"]))
    return {
        "atm30": atm30, "vix": vix / 100.0, "ratio": atm30 / (vix / 100.0),
        "expiry_below": e_lo, "expiry_above": e_hi, "implied_spot": s_impl,
        "n_expiries": int(len(term)),
    }


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
      sub_df    = engine.sub_period_stability(spx_df, vix_wide)
      null      = s3_timing_null_test(engine)
      engine.save()
    """

    ASSUMPTIONS: List[str] = [
        "30-day ATM straddles for every signal; re-struck ATM at each new entry "
        "and at expiry (21 sessions), when an open position is rolled",
        "Delta-hedged daily and self-financing: pnl = direction×[ΔV − Δ_{t−1}ΔS "
        "+ (r(ΔS−V) − qΔS)/252]×n×100; the hedge pays 5 bp on the opening "
        "trade, every rebalance and the closing unwind",
        "ATM vol = β × VIX term structure (β = ATM_TO_VIX_RATIO, estimated from "
        "real 2026 SPX chains); VIX itself is a variance-swap rate that "
        "includes the skew premium, so it is never used as ATM vol directly",
        "MARK-TO-MODEL P&L: no historical option prices in DB — straddles priced "
        "via Black-Scholes at β×VIX-TS (no smile, no pin risk); the "
        "Heston/joint-calibration stack is NOT in the backtest P&L path",
        "S3 = index leg of the dispersion trade: long dispersion ⇒ short SPX "
        "straddle on the same engine; the single-stock leg is not modelled "
        "(no single-stock option data)",
        "Per-date r from ^IRX 3M T-bill (previous print on gaps), q=0.013. "
        "The P&L is an excess return on $1M unfunded collateral assumed to "
        "earn T-bills: Sharpe/Sortino use rf=0 on it",
        "Sizing off each sleeve's running NAV; short straddles capped at "
        "NAV/(0.20×K×100) contracts",
        "Regime labels: lagged deterministic rule (rv_20d vs VIX; VVIX gate→R2); "
        "the ML classifier is not in the trading loop (it did not beat the "
        "persistence baseline)",
        "Regime gate: new entries blocked in R2; S1C/S4 force-close open "
        "positions on transition INTO R2",
        "Ex-ante sizing overlay, shifted 1 day, second simulation pass: "
        "correlation netting 1/√(1+Σmax(ρ,0)) on trailing 60d P&L corr × "
        "trailing-252d net-Sharpe scale in [0.5, 1]",
        ("Walk-forward PDVLinear forecast of the next-21-session RMS vol "
         "(refit daily on labels observed at the forecast origin, purged) "
         "behind the S1/S1C pdv_iv_spread" if PDV_SPREAD_HORIZON > 1 else
         "Walk-forward PDVLinear retrained per year on pre-year returns"),
    ]

    # Each entry is "Title: explanation" (the report renders the title).
    HONEST_FAILURES: List[str] = [
        "β is one constant from 2026 chains: ATM vol is mapped from VIX with a "
        "single ratio estimated on five 2026 SPX chains; there is no 2018-2025 "
        "ATM-IV history and the ratio moves with skew. At β = 0.82 the model's "
        "30-day ATM vol averages 0.3 vol pt BELOW the realised vol of the next "
        "21 sessions over 2018-2025 (median 1.2 pt above); at β = 1 it averages "
        "3.3 pt above (median 4.4). If "
        "ATM implied exceeded realised on average in this window, as most SPX "
        "studies find, the steep-skew 2026 chains understate β for it. Short-vol "
        "P&L is very sensitive to β (run_backtest --beta-sensitivity)",
        "Mark-to-model only: no smile dynamics, no pin risk, and the 0.15 vol-pt "
        "bid-ask does not widen in crises",
        "S3 is an index-leg proxy: the single-stock leg of the dispersion trade "
        "is not modelled, so S3's P&L is a short-SPX-straddle P&L timed by the "
        "VIX/VVIX rule. Its pre-C22 z-unit P&L was positive by construction "
        "(positive on most random walks); judge its timing only against the "
        "null test (s3_timing_null_test)",
        "S1C data snooping: the contrarian direction was chosen AFTER observing "
        "S1 lose on 2018-2025 — its backtest P&L is in-sample by construction and "
        "needs confirmation on post-2026-06 data before being trusted",
        "S1C pseudo-OOS is confounded: 2013-2017 (run_backtest --pseudo-oos) was "
        "the best systematic short-vol era in modern history (pre-Volmageddon), "
        "so a result there may be short-vol beta, not contrarian alpha; entry "
        "thresholds are shared with the discovery window",
        *([
            "PDV spread units: the S1/S1C pdv_iv_spread compares a 1-day E|r| PDV "
            "forecast with 30-day VIX (units and horizon mismatch); the spread is "
            "biased negative and the S1/S1C entry filter is close to always-on in R1",
        ] if PDV_SPREAD_HORIZON == 1 else []),
        "Regime classifier: it did not beat the trivial persistence baseline for "
        "next-day regime prediction and was removed from the trading loop",
        "Sub-periods are not out-of-sample: the 6-month windows re-run the same "
        "rules; nothing is re-fitted per window, so they test stability, not skill",
        "C7 Run B is not a hedge: the PDV-scaled vega line is an attribution "
        "variant of the SAME hedge; it explains less P&L variance than Run A and "
        "says nothing about hedging with PDV",
        "No statistical significance: few trades per sleeve on one 7-year path; "
        "this is a research artifact and audit trail, not a deployable edge",
    ]

    def __init__(
        self,
        start_date:      str   = BT_START,
        end_date:        str   = BT_END,
        initial_capital: float = INITIAL_CAPITAL,
        r:               float = R_BACKTEST,
        q:               float = Q_BACKTEST,
        seed:            int   = RANDOM_SEED,
        atm_to_vix_ratio: float = ATM_TO_VIX_RATIO,
        corrections:     Optional[C22Corrections] = None,
        vvix_threshold   = _DEFAULT_GATE,
        pdv_horizon:     Optional[int] = None,
        s3_leg_direction: int = S3_LEG_DIRECTION,
        pdv_vol_series:  Optional[pd.Series] = None,
    ):
        """
        atm_to_vix_ratio : β in σ_ATM = β·VIX-TS (1.0 = pre-C22 "VIX is ATM vol")
        corrections      : C22Corrections switches (default: all corrected)
        vvix_threshold   : R2 gate passed to build_regime_labels. Left at the
                           default, the engine uses build_regime_labels' own
                           default gate, so the backtest follows the single
                           gate regime_classifier defines; a float is a fixed
                           VVIX level, None the adaptive 80th percentile.
        pdv_horizon      : build_xy horizon of the walk-forward PDV behind
                           pdv_iv_spread (default PDV_SPREAD_HORIZON)
        s3_leg_direction : straddle direction of the S3 index leg when the
                           signal is on (see S3_LEG_DIRECTION)
        pdv_vol_series   : optional precomputed walk-forward σ̂ for
                           pdv_horizon > 1 (regime_classifier.walk_forward_pdv_vol,
                           value at t dated ≤ t). run_backtest builds it from
                           SPX history since PDV_HISTORY_START so the first
                           forecasts are not trained on 2015-2017 alone;
                           without it σ̂ is built from the spx_df given to run().
                           Ignored for pdv_horizon == 1.
        """
        self.start_date      = start_date
        self.end_date        = end_date
        self.initial_capital = initial_capital
        self.r               = r
        self.q               = q
        self.seed            = seed
        self.atm_to_vix_ratio = float(atm_to_vix_ratio)
        self.corrections     = corrections or C22Corrections()
        self.vvix_threshold  = vvix_threshold
        self.pdv_horizon     = int(pdv_horizon if pdv_horizon is not None else PDV_SPREAD_HORIZON)
        if s3_leg_direction not in (-1, 1):
            raise ValueError("s3_leg_direction must be -1 (short straddle) or +1 (long)")
        self.s3_leg_direction = int(s3_leg_direction)
        self.pdv_vol_series  = pdv_vol_series
        self.pdv_vol_:   Optional[pd.Series] = None
        np.random.seed(seed)

        self.equity_df:   Optional[pd.DataFrame] = None
        self.signals_df:  Optional[pd.DataFrame] = None
        self.all_trades:  List[TradeRecord]       = []
        self.metrics_:    Optional[Dict]          = None
        self._wf_results: Optional[pd.DataFrame] = None
        self._rate_series: Optional[pd.Series]   = None
        self._market: Dict = {}
        self._pass1_pnl: Dict[str, pd.Series] = {}

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
    def _ensure_log_return(spx_df: pd.DataFrame) -> pd.DataFrame:
        df = spx_df.copy()
        if "log_return" not in df.columns:
            df = df.sort_values("date").copy()
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def _load_rate_series(self) -> Optional[pd.Series]:
        """
        Per-date ^IRX series from the DB, or None if it does not span the
        window (rate_series_spans_window: a stub must not be forward-filled
        into a five-year hurdle). Read from 40 days before the start so the
        first backtest dates have a previous print to carry forward.
        """
        from joint_vol_calibration.data.database import get_tbill_rates_series

        fetch_start = (pd.Timestamp(self.start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        df = get_tbill_rates_series(as_of_date=self.end_date, start_date=fetch_start)
        if df.empty:
            logger.warning("No T-bill rates in DB — using fixed r=%.4f", self.r)
            return None
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if not rate_series_spans_window(df["date"], self.start_date, self.end_date):
            logger.warning(
                "T-bill series does not span the backtest window "
                "(%d rows, %s → %s, window %s → %s) — "
                "falling back to fixed r=%.4f rather than forward-filling a stub",
                len(df), df["date"].iloc[0].date(), df["date"].iloc[-1].date(),
                self.start_date, self.end_date, self.r,
            )
            return None
        return df.set_index("date")["rate"].astype(float)

    # ── Walk-forward PDV model ────────────────────────────────────────────────

    def _build_walkforward_pdv_vol(self, spx_df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Walk-forward PDV vol forecasts σ̂(t) with zero look-ahead.

        horizon > 1 (canonical 21): regime_classifier.walk_forward_pdv_vol —
          on each session t a PDVLinear (sigma1, sigma2, lev) is refit on the
          forward-RMS-vol labels whose window ends by t (purged, embargo =
          horizon) and predicts from t's features, so σ̂(t) uses returns
          dated ≤ t only and targets sqrt(252/h·Σ r²_{t+1..t+h}).
        horizon 1 (pre-C22 replay): one PDVLinear per backtest year, fit on
          returns strictly before January 1 (delta_hedger.fit_pdv_before) on
          the |r_{t+1}| target.

        SignalEngine lags features by one session, so the spread traded on
        session t is built from information dated t−1.

        Returns
        -------
        pd.Series of annualised vol forecasts (decimal), or None if none
        could be made (too little history) — the rv_20d proxy is then used.
        """
        if self.pdv_horizon > 1:
            from joint_vol_calibration.signals.regime_classifier import walk_forward_pdv_vol
            if self.pdv_vol_series is not None:
                out = pd.Series(self.pdv_vol_series, dtype=float).dropna()
                out.index = pd.to_datetime(out.index)
            else:
                out = walk_forward_pdv_vol(spx_df, horizon=self.pdv_horizon)
            out = out[out.index <= pd.Timestamp(self.end_date)]
            if out.empty:
                logger.warning("WF PDV (h=%d): no forecasts — rv proxy fallback", self.pdv_horizon)
                return None
            logger.info("WF PDV (h=%d, daily purged refit): %d forecasts %s → %s",
                        self.pdv_horizon, len(out), out.index[0].date(), out.index[-1].date())
            return out
        return self._build_yearly_pdv_vol(spx_df)

    def _build_yearly_pdv_vol(self, spx_df: pd.DataFrame) -> Optional[pd.Series]:
        """Pre-C22 per-year PDVLinear on the 1-day |r| target (bridge replay)."""
        from joint_vol_calibration.models.pdv import PDVLinear, extract_pdv_features

        df = spx_df.copy()
        if "date" in df.columns:
            df = df.set_index(pd.to_datetime(df["date"]))
        df = df.sort_index()
        log_ret = df["log_return"].dropna()
        log_ret = log_ret[~log_ret.index.duplicated(keep="first")]

        # Full-history features: EMAs at year boundaries stay warm (the EMA
        # recursion only ever uses past returns, so this is look-ahead-free).
        feats_full = extract_pdv_features(log_ret)

        backtest_start = pd.Timestamp(self.start_date)
        backtest_end   = pd.Timestamp(self.end_date)

        preds: List[pd.Series] = []
        for year in range(backtest_start.year, backtest_end.year + 1):
            cutoff   = pd.Timestamp(f"{year}-01-01")
            year_end = pd.Timestamp(f"{year}-12-31")

            model, train_end = fit_pdv_before(log_ret, cutoff, horizon=self.pdv_horizon)
            if model is None:
                logger.warning(
                    "WF PDV: insufficient training data for %d — rv proxy fallback for that year",
                    year,
                )
                continue

            mask   = (feats_full.index >= cutoff) & (feats_full.index <= year_end)
            X_year = feats_full.loc[mask].dropna(subset=PDVLinear._XCOLS)
            if len(X_year) == 0:
                continue
            preds.append(model.predict(X_year))
            logger.info(
                "WF PDV year %d (h=%d): trained through %s → predicted %d days  (%s)",
                year, self.pdv_horizon, train_end.date(), len(X_year), model,
            )

        if not preds:
            logger.warning("WF PDV: no forecasts generated — rv proxy fallback")
            return None

        out = pd.concat(preds).sort_index()
        return out[~out.index.duplicated(keep="first")]

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
        clf      : ignored by the trading path (regimes are the lagged rule
                   labels, passed to SignalEngine as precomputed_regimes,
                   which take precedence). Kept for API compatibility; it is
                   no longer loaded from disk.

        Returns
        -------
        equity_df : date-indexed DataFrame with columns:
          nav, daily_pnl, nav_s1c, nav_s3, nav_s4, nav_s1, nav_s2,
          nav_combined, nav_tbill, pnl_s1c, pnl_s3, pnl_s4, pnl_s1, pnl_s2,
          pnl_combined, regime, s1_position, s1c_position, s2_position,
          s3_position, s4_position, combined_pos
        """
        c = self.corrections
        beta = self.atm_to_vix_ratio

        # ── Load data if not provided ──────────────────────────────────────────
        if spx_df is None or vix_wide is None:
            from joint_vol_calibration.data.database import (
                get_spx_ohlcv,
                get_vix_term_structure_wide,
            )
            spx_df   = get_spx_ohlcv(as_of_date=self.end_date)
            vix_wide = get_vix_term_structure_wide(as_of_date=self.end_date)

        # ── Per-date T-bill rate (pricing, financing, cash benchmark) ─────────
        _rate_series = self._load_rate_series()
        self._rate_series = _rate_series

        # NOTE (C17): there is no monthly Heston recalibration. Backtest
        # pricing is Black-Scholes on β × the VIX-term-structure vol; the
        # Heston/joint calibration stack serves C6 (Greeks) and C7.

        # Normalise to have 'date' column
        spx_col  = self._to_df_with_date_col(self._to_date_indexed(spx_df))
        vix_col  = self._to_df_with_date_col(self._to_date_indexed(vix_wide))
        spx_col  = self._ensure_log_return(spx_col)

        # Date-indexed versions for fast lookup
        spx_idx  = self._to_date_indexed(spx_col)
        vix_idx  = self._to_date_indexed(vix_col)

        # ── Regime labels: lagged deterministic rule (C17) ─────────────────────
        # The label is DEFINED by same-day observables (rv_20d vs VIX; VVIX
        # gate → R2), so yesterday's label is known at the close of t−1;
        # SignalEngine shifts it by 1 day (the persistence predictor). The
        # gate is build_regime_labels' default unless one was passed in.
        logger.info("Building rule-based regime labels (persistence predictor) ...")
        gate_kw = ({} if self.vvix_threshold is _DEFAULT_GATE
                   else {"vvix_threshold": self.vvix_threshold})
        rule_regimes = build_regime_labels(spx_col, vix_col, **gate_kw)

        # ── Walk-forward PDV vol forecasts (zero look-ahead) ───────────────────
        logger.info("Building walk-forward PDV forecasts ...")
        wf_pdv_vol = self._build_walkforward_pdv_vol(spx_col)
        self.pdv_vol_ = wf_pdv_vol

        # ── Generate C9 signals ────────────────────────────────────────────────
        engine  = SignalEngine(clf=clf)
        sig_df  = engine.generate(
            spx_col, vix_col,
            start_date=self.start_date,
            end_date=self.end_date,
            precomputed_regimes=rule_regimes,
            precomputed_pdv_vol=wf_pdv_vol,
        )
        self.signals_df = sig_df

        # Legacy S3 proxy z-score (only used when s3_index_leg is off)
        z_ratio = _build_s3_zscore(vix_col).reindex(sig_df.index)

        # Sizing capital. With sleeve_nav_sizing the simulators add each
        # sleeve's own running P&L (through t−1) to this base.
        nav_const = pd.Series(self.initial_capital, index=sig_df.index, dtype=float)
        spx_close = spx_idx["close"]
        _r = _rate_series if _rate_series is not None else self.r
        # Everything the S3 null test needs to re-run SignalEngine exactly as
        # here (full history, same regimes and PDV series) on synthetic ratios.
        self._market = {"spx_close": spx_close, "vix_idx": vix_idx, "r": _r,
                        "nav_base": nav_const, "spx_col": spx_col, "vix_col": vix_col,
                        "regimes": rule_regimes, "pdv_vol": wf_pdv_vol}

        def _straddle(pos, kel, label):
            return _simulate_straddle_pnl(
                pos, kel, nav_const, spx_close, vix_idx,
                r=_r, q=self.q, signal_label=label,
                atm_to_vix_ratio=beta, corrections=c,
            )

        def _s3(kel):
            if c.s3_index_leg:
                # Index leg: long dispersion = SHORT the SPX straddle (default)
                return _straddle(self.s3_leg_direction * sig_df["s3_position"], kel, "s3")
            return _simulate_s3_pnl(sig_df["s3_position"], kel, nav_const, z_ratio,
                                    corrections=c)

        # ── Pass 1: every signal at its base size ──────────────────────────────
        pnl_s1,  t1 = _straddle(sig_df["s1_position"],  sig_df["s1_kelly"],  "s1")
        pnl_s2,  t2 = _straddle(sig_df["s2_position"],  sig_df["s2_kelly"],  "s2")
        pnl_s3,  t3 = _s3(sig_df["s3_kelly"])
        pnl_cb,  t4 = _straddle(sig_df["combined_pos"], sig_df["combined_kelly"], "combined")
        pnl_s4,  t5 = _straddle(sig_df["s4_position"],  sig_df["s4_kelly"],  "s4")
        pnl_s1c, t6 = _straddle(sig_df["s1c_position"], sig_df["s1c_kelly"], "s1c")
        self._pass1_pnl = {"s1": pnl_s1, "s2": pnl_s2, "s3": pnl_s3, "combined": pnl_cb,
                           "s4": pnl_s4, "s1c": pnl_s1c}

        # ── Pass 2: ex-ante sizing overlay for the portfolio sleeves ──────────
        # PORTFOLIO: S1C (contrarian PDV) + S3 (dispersion, index leg) + S4 (VRP).
        # S1, S2 and Combined are research reference curves (no weight).
        # From pass-1 P&L we compute per-day multipliers using ONLY trailing
        # windows, then SHIFT BY 1 DAY so the multiplier applied on day t is
        # computable at the close of t−1. Pass 2 re-simulates the trades with
        # kelly × multiplier, so contract counts, entry/exit costs and
        # slippage reflect the reduced size.
        pnl_mat = pd.DataFrame({"s1c": pnl_s1c, "s3": pnl_s3, "s4": pnl_s4}).fillna(0.0)
        if c.corr_netting:
            m_corr = _corr_netting_multipliers(pnl_mat)
        else:
            m12 = _legacy_corr_kelly_mult(pnl_mat["s1c"], pnl_mat["s3"])
            m13 = _legacy_corr_kelly_mult(pnl_mat["s1c"], pnl_mat["s4"])
            m23 = _legacy_corr_kelly_mult(pnl_mat["s3"],  pnl_mat["s4"])
            m_corr = pd.DataFrame({"s1c": (m12 * m13) ** 0.5, "s3": (m12 * m23) ** 0.5,
                                   "s4": (m13 * m23) ** 0.5})
        k_mult = {
            s: (m_corr[s].clip(0.5, 1.0) * _net_kelly_scale(pnl_mat[s])).shift(1).fillna(1.0)
            for s in PORTFOLIO_SIGNALS
        }
        self._k_mult = k_mult

        pnl_s1c_adj, t6b = _straddle(sig_df["s1c_position"], sig_df["s1c_kelly"] * k_mult["s1c"], "s1c")
        pnl_s3_adj,  t3b = _s3(sig_df["s3_kelly"] * k_mult["s3"])
        pnl_s4_adj,  t5b = _straddle(sig_df["s4_position"], sig_df["s4_kelly"] * k_mult["s4"], "s4")

        # Trade log: pass-2 records for portfolio signals (actual sizing);
        # pass-1 records for the S1/S2/Combined research references.
        self.all_trades = t1 + t2 + t4 + t3b + t5b + t6b

        # ── Build equity curves ────────────────────────────────────────────────
        cap = self.initial_capital
        pnl_total = (pnl_s1c_adj + pnl_s3_adj + pnl_s4_adj) / 3.0

        # Cash benchmark: $cap compounding at the prior day's ^IRX
        rates = _resolve_rates(_r, sig_df.index, ffill=True)
        growth = np.concatenate([[1.0], 1.0 + rates[:-1] / TRADING_DAYS])
        nav_tbill = cap * np.cumprod(growth)

        eq = pd.DataFrame({
            "nav":          cap + pnl_total.cumsum(),
            "daily_pnl":    pnl_total,
            # Main portfolio signals
            "nav_s1c":      cap + pnl_s1c_adj.cumsum(),
            "nav_s3":       cap + pnl_s3_adj.cumsum(),
            "nav_s4":       cap + pnl_s4_adj.cumsum(),
            # Research reference curves (not in portfolio)
            "nav_s1":       cap + pnl_s1.cumsum(),
            "nav_s2":       cap + pnl_s2.cumsum(),
            "nav_combined": cap + pnl_cb.cumsum(),
            # Cash benchmark (what the collateral alone would have earned)
            "nav_tbill":    nav_tbill,
            # Raw P&L columns
            "pnl_s1c":      pnl_s1c_adj,
            "pnl_s3":       pnl_s3_adj,
            "pnl_s4":       pnl_s4_adj,
            "pnl_s1":       pnl_s1,
            "pnl_s2":       pnl_s2,
            "pnl_combined": pnl_cb,
        }, index=sig_df.index)

        # Attach signal columns
        for col in ["s1_position", "s1c_position", "s2_position", "s3_position",
                    "s4_position", "combined_pos", "regime"]:
            if col in sig_df.columns:
                eq[col] = sig_df[col]

        self.equity_df = eq
        logger.info(
            "Backtest complete: %d days | NAV $%.0f → $%.0f | trades: %d | β=%.2f",
            len(eq), cap, float(eq["nav"].iloc[-1]), len(self.all_trades), beta,
        )
        return eq

    # ── Metrics interface ──────────────────────────────────────────────────────

    def portfolio_trades(self) -> List[TradeRecord]:
        """Trades of the three portfolio sleeves (S1C, S3, S4) only."""
        return [t for t in self.all_trades if t.signal in PORTFOLIO_SIGNALS]

    def compute_metrics(self, equity_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Full performance metrics plus per-signal breakdown.

        Sharpe/Sortino: rf = 0 on the excess P&L (see SHARPE_RF). Portfolio
        trade stats (win_rate, n_trades, avg_pnl_per_trade) cover the three
        portfolio sleeves only; the research references have their own
        per-signal keys. ``cash_cum_return`` is what the $1M collateral
        would have earned in T-bills over the same dates.

        Every curve is measured from the NAV before its first session
        (``_nav_before``): the initial capital for this run's own curve
        (each sleeve and each sub-period window starts flat on it), the
        last NAV before the slice for a later slice of it. The first
        session's P&L — the entry costs of a position opened on it — is
        thus in every return, Sharpe and drawdown, and n_days counts every
        session of the frame. (Until this was fixed the base was the first
        row's NAV, after that session's P&L: on the 2019-07-01 sub-period
        window the −$678.52 of first-day costs were dropped and the
        cumulative return read −3.746% for −3.811%.)
        """
        df = equity_df if equity_df is not None else self.equity_df
        if df is None:
            raise RuntimeError("No equity DataFrame. Call run() first.")

        c = self.corrections
        _rf_series = None if c.sharpe_rf_zero else getattr(self, "_rate_series", None)
        _rf = SHARPE_RF if c.sharpe_rf_zero else 0.05
        m = compute_metrics(df, rf=_rf, trade_log=self.portfolio_trades(), rf_series=_rf_series,
                            base_nav=self._nav_before(df, "nav"))
        m["sharpe_convention"] = ("excess P&L, rf=0 (collateral earns T-bills)"
                                  if c.sharpe_rf_zero else "NAV minus per-date ^IRX (pre-C22)")

        # Per-signal Sharpe and annualised return
        for sig, nav_col in [("s1","nav_s1"),("s2","nav_s2"),("s3","nav_s3"),
                              ("s4","nav_s4"),("s1c","nav_s1c"),("combined","nav_combined")]:
            if nav_col in df.columns:
                sub = pd.DataFrame({"nav": df[nav_col]})
                sm  = compute_metrics(sub, rf=_rf, rf_series=_rf_series,
                                      base_nav=self._nav_before(df, nav_col))
                m[f"{sig}_sharpe"]     = sm.get("sharpe", np.nan)
                m[f"{sig}_ann_return"] = sm.get("ann_return", np.nan)
                m[f"{sig}_max_drawdown"] = sm.get("max_drawdown", np.nan)

        if "nav_tbill" in df.columns and len(df) > 0:
            m["cash_cum_return"] = float(df["nav_tbill"].iloc[-1]
                                         / self._nav_before(df, "nav_tbill") - 1.0)

        # Per-signal trade stats
        from collections import defaultdict
        sig_trades: Dict[str, List] = defaultdict(list)
        for t in self.all_trades:
            sig_trades[t.signal].append(t)
        for sig, tl in sig_trades.items():
            pnls = [t.net_pnl for t in tl]
            if pnls:
                m[f"{sig}_win_rate"]  = float(sum(1 for p in pnls if p > 0) / len(pnls))
                m[f"{sig}_avg_pnl"]  = float(np.mean(pnls))
                m[f"{sig}_n_trades"] = len(pnls)
                m[f"{sig}_total_pnl"] = float(sum(pnls))
                m[f"{sig}_n_open"]   = int(sum(1 for t in tl if t.is_open))

        m["atm_to_vix_ratio"] = self.atm_to_vix_ratio
        self.metrics_ = m
        return m

    def _nav_before(self, df: pd.DataFrame, col: str) -> float:
        """
        NAV of ``col`` at the close before df's first session — the base its
        returns are measured from. When df is a later slice of this run's
        equity curve, that is the column's last value in the curve before
        the slice; otherwise df starts on the run's first session and the
        base is the initial capital, where every NAV column of run()
        (portfolio, sleeves, T-bill cash) starts.
        """
        first = df[col].first_valid_index() if len(df) else None
        full = self.equity_df
        if first is not None and full is not None and col in full.columns:
            prior = full.loc[full.index < first, col].dropna()
            if len(prior):
                return float(prior.iloc[-1])
        return float(self.initial_capital)

    def crisis_performance(self, equity_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
        """Compute performance metrics for each crisis period (rf = 0), each
        measured from the NAV before its first session."""
        df = equity_df if equity_df is not None else self.equity_df
        if df is None:
            raise RuntimeError("No equity DataFrame. Call run() first.")
        return compute_crisis_performance(df, base_nav=self._nav_before(df, "nav"))

    # ── Sub-period stability ───────────────────────────────────────────────────

    def sub_period_stability(
        self,
        spx_df:        pd.DataFrame,
        vix_wide:      pd.DataFrame,
        n_windows:     int = WF_N_WINDOWS,
        window_months: int = WF_WINDOW_MONTHS,
    ) -> pd.DataFrame:
        """
        Re-run the full pipeline on consecutive 6-month windows (2019 →).

        Each window starts flat with fresh capital and the same rules and
        parameters (β, corrections, gate, PDV horizon), and reads the same
        causal PDV forecast as the main run (the PDV-21 σ̂ series, each value
        dated ≤ its session; with pdv_horizon = 1 the per-year refit on
        pre-year data). Nothing is fitted per window — so this measures how
        stable the strategy is across sub-periods. It is NOT an
        out-of-sample test: the signal rules, thresholds and S1C's direction
        were chosen on the whole 2018-2025 sample.

        (Before C22 this method also re-trained the C8 classifier per
        window, ~6 min each, and passed it to run(), which ignores it —
        regimes are the precomputed rule labels.)

        Each window's metrics are measured from its fresh capital (the
        window's engine computes them from its own initial_capital), so a
        position opened on the window's first session pays its entry costs
        inside the window's return, Sharpe and drawdown.

        Returns
        -------
        DataFrame with columns: window, train_end, test_start, test_end,
          sharpe (rf=0), ann_return, max_drawdown, n_trades (portfolio
          sleeves), below_zero_sharpe. ``train_end`` is the day before the
          window: the last date any PDV fit for the window can see.
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

            logger.info("Sub-period %d/%d: %s → %s", i + 1, n_windows,
                        test_start.date(), test_end.date())
            try:
                sub = BacktestEngine(
                    start_date      = test_start.strftime("%Y-%m-%d"),
                    end_date        = test_end.strftime("%Y-%m-%d"),
                    initial_capital = self.initial_capital,
                    r=self.r, q=self.q, seed=self.seed,
                    atm_to_vix_ratio=self.atm_to_vix_ratio,
                    corrections=self.corrections,
                    vvix_threshold=self.vvix_threshold,
                    pdv_horizon=self.pdv_horizon,
                    s3_leg_direction=self.s3_leg_direction,
                    # σ̂(t) is causal, so the full-run series is valid for
                    # every window (and saves a refit per window)
                    pdv_vol_series=(self.pdv_vol_series if self.pdv_vol_series is not None
                                    else self.pdv_vol_),
                )
                eq_sub = sub.run(spx_col, vix_col)
                m_sub  = sub.compute_metrics(eq_sub)
            except Exception as exc:
                logger.warning("Sub-period %d backtest failed: %s", i + 1, exc)
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

    def walk_forward_validation(self, *args, **kwargs) -> pd.DataFrame:
        """Backward-compatible name for sub_period_stability (not an OOS test)."""
        return self.sub_period_stability(*args, **kwargs)

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
            "atm_to_vix_ratio": self.atm_to_vix_ratio,
            "pdv_horizon":     self.pdv_horizon,
            "s3_leg_direction": self.s3_leg_direction,
            "pdv_iv_spread_source": (self.signals_df.attrs.get("pdv_iv_spread_source")
                                     if self.signals_df is not None else None),
            "corrections":     {f.name: getattr(self.corrections, f.name)
                                for f in fields(self.corrections)},
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

        corr = state.get("corrections")
        engine            = cls(
            start_date      = state["start_date"],
            end_date        = state["end_date"],
            initial_capital = state["initial_capital"],
            r               = state.get("r", R_BACKTEST),
            q               = state.get("q", Q_BACKTEST),
            seed            = state.get("seed", RANDOM_SEED),
            # Results saved before C22 carry no β: they were priced at VIX (β=1)
            atm_to_vix_ratio = state.get("atm_to_vix_ratio", 1.0),
            corrections     = C22Corrections(**corr) if corr else C22Corrections.legacy(),
            pdv_horizon     = state.get("pdv_horizon"),
            s3_leg_direction = state.get("s3_leg_direction", S3_LEG_DIRECTION),
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


# ── S3 timing null test ────────────────────────────────────────────────────────

def s3_timing_null_test(
    engine:  BacktestEngine,
    n_paths: int = 200,
    seed:    int = RANDOM_SEED,
    method:  str = "random_walk",
) -> Dict:
    """
    Is the S3 timing rule better than random timing with the same mechanics?

    The real S3 rule (z of VIX/VVIX < −1 entry, > −0.3 exit, 30-day max hold,
    R2 gate) is re-run on ``n_paths`` synthetic VIX/VVIX ratio paths that
    carry no information about the market:
      method='random_walk' : Gaussian random walk in log(ratio) with the
                             real daily log-change std, started at the real
                             first value
      method='permutation' : the real daily log-changes in shuffled order
    Each path replaces ^VVIX by ^VIX / synthetic ratio over the WHOLE
    history the engine was given (sessions with no real VVIX print stay
    missing) and re-runs SignalEngine.generate exactly as engine.run() did:
    same spx/VIX history, precomputed regimes and PDV series, trailing
    z-scores warmed up on pre-window data, entries only inside the window.
    ^VVIX feeds nothing but S3, so every other input — the regime gate, the
    SPX/VIX market the straddles are priced on, β, the corrections and base
    (pass-1) sizing — is the real one. Only S3's timing changes.

    Two statistics are compared with their null distributions:
      index_leg : total P&L of S3 priced as a short SPX straddle (the
                  engine's S3 model) — the test of S3's timing
      z_proxy   : the pre-C22 z-unit proxy P&L. Its null distribution is
                  shifted far above zero: positive "P&L" on series with no
                  predictability, which is why it is not evidence of edge.

    p_value = (1 + #{null ≥ real}) / (n_paths + 1), one-sided.

    The real statistic goes through the same function on the real ^VVIX, so
    it IS the engine's pass-1 S3 P&L: ``engine_pass1_pnl`` (the engine's
    own number) must equal index_leg['real'] (z_proxy['real'] when the
    engine runs the legacy proxy). Before this was re-based on
    SignalEngine, the null re-built S3 from the backtest-window features
    alone, so its "real" S3 warmed its 252-day z-score up inside the window
    and was not the S3 the engine trades (2018-2025: 26 entries, first on
    2018-07-16, +$110,675, against the engine's first entry on 2018-01-04
    and +$41,952 at pass-1 size). Requires engine.run() to have been called.
    """
    if method not in ("random_walk", "permutation"):
        raise ValueError(f"unknown method {method!r}")
    if engine.signals_df is None or not engine._market:
        raise RuntimeError("Run the engine before the S3 null test.")
    mk = engine._market
    c = engine.corrections
    nav = mk["nav_base"]
    vix_col = mk["vix_col"]
    sig_engine = SignalEngine(clf=None)

    def _stats(vix_df: pd.DataFrame) -> Tuple[float, float, int]:
        sig = sig_engine.generate(
            mk["spx_col"], vix_df,
            start_date=engine.start_date, end_date=engine.end_date,
            precomputed_regimes=mk["regimes"], precomputed_pdv_vol=mk["pdv_vol"],
        )
        pos, kel = sig["s3_position"], sig["s3_kelly"]
        leg, _ = _simulate_straddle_pnl(
            engine.s3_leg_direction * pos, kel, nav, mk["spx_close"], mk["vix_idx"],
            r=mk["r"], q=engine.q,
            signal_label="s3", atm_to_vix_ratio=engine.atm_to_vix_ratio, corrections=c,
        )
        z = _build_s3_zscore(vix_df).reindex(sig.index)
        proxy, _ = _simulate_s3_pnl(pos, kel, nav, z, corrections=c)
        n_entries = int(((pos != 0) & (pos.shift(1, fill_value=0) == 0)).sum())
        return float(leg.sum()), float(proxy.sum()), n_entries

    se_logger = logging.getLogger(SignalEngine.__module__)
    level = se_logger.level
    se_logger.setLevel(max(level, logging.WARNING))     # one INFO line per path otherwise
    try:
        real_leg, real_proxy, real_n = _stats(vix_col)

        vix = pd.to_numeric(vix_col["^VIX"], errors="coerce").to_numpy(dtype=float)
        vvix = pd.to_numeric(vix_col["^VVIX"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(vix) & np.isfinite(vvix) & (vix > 0) & (vvix > 0)
        log_ratio = np.log(vix[ok] / vvix[ok])
        steps = np.diff(log_ratio)
        rng = np.random.default_rng(seed)
        null_leg, null_proxy, null_n = [], [], []
        for _ in range(n_paths):
            if method == "random_walk":
                inc = rng.normal(0.0, steps.std(ddof=1), len(steps))
            else:
                inc = rng.permutation(steps)
            path = log_ratio[0] + np.concatenate([[0.0], np.cumsum(inc)])
            syn = np.full(len(vix_col), np.nan)
            syn[ok] = vix[ok] / np.exp(path)
            f = vix_col.copy()
            f["^VVIX"] = syn
            a, b, n = _stats(f)
            null_leg.append(a)
            null_proxy.append(b)
            null_n.append(n)
    finally:
        se_logger.setLevel(level)

    def _summ(real: float, null: List[float]) -> Dict:
        arr = np.asarray(null)
        return {
            "real": real,
            "null_mean": float(arr.mean()),
            "null_std": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
            "null_frac_positive": float((arr > 0).mean()),
            "p_value": float((1 + (arr >= real).sum()) / (len(arr) + 1)),
        }

    return {
        "method": method, "n_paths": n_paths, "seed": seed,
        "engine_pass1_pnl": float(engine._pass1_pnl["s3"].sum()) if "s3" in engine._pass1_pnl else np.nan,
        # entries (flat → on), not trade records: an S3 position held past
        # the straddle's 21-session expiry is rolled into a new record
        "real_trades": real_n, "null_mean_trades": float(np.mean(null_n)) if null_n else np.nan,
        "index_leg": _summ(real_leg, null_leg),
        "z_proxy": _summ(real_proxy, null_proxy),
    }
