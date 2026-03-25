"""
delta_hedger.py — C7: Delta-Hedged P&L Simulation

Simulates daily delta-hedging of a long ATM SPX straddle over the 2020
calendar year (entry 2020-01-02, exit 2020-12-31, rebalance each close).

Portfolio structure
-------------------
  Long 1 ATM straddle (call + put)
    Entry:  2020-01-02  K = S_entry = 3257.85,  T = 1 year
    Exit:   2020-12-31
  Short Δ_{t-1} units of SPX delta (rebalanced daily close)

Two hedging runs
----------------
  Run A  — Heston-IV / market Greeks
              ATM vol = VIX term-structure interpolated to T_rem
  Run B  — PDV-adjusted vega
              Vega attribution scaled by  σ_PDV_{t-1} / σ_ATM_{t-1}
              Delta hedge identical to Run A

P&L attribution (per trading day)
----------------------------------
  V_t − V_{t-1} − Δ_{t-1}·ΔS  =  Γ-P&L + ν-P&L + Θ-P&L + Residual

  Γ-P&L      =  ½ · Γ_{t-1} · (ΔS)²
  ν-P&L_A    =  ν_{t-1} · Δσ
  ν-P&L_B    =  ν_{t-1} · (σ_PDV_{t-1} / σ_ATM_{t-1}) · Δσ
  Θ-P&L      =  θ_{t-1} / 252           (negative = decay)
  Residual_X  =  Total P&L − Γ − ν_X − Θ

  where Total P&L = V_t − V_{t-1} − Δ_{t-1}·(S_t − S_{t-1})

Vomma flags
-----------
  Daily vomma computed from BS formula at (S_{t-1}, K, T_rem_{t-1}, σ_ATM_{t-1}).
  z-score normalised against mean / std from C6 greeks_surface.parquet.
  Unstable day:  |z| > 2.0

Zero look-ahead guarantee
-------------------------
  All Greeks (Δ, Γ, ν, θ) are computed using close-of-day t-1 data.
  PDV features at t use EWMA of returns through t-1 only.
  VIX interpolated vol at t-1 is used — never day-t.

Outputs
-------
  data_store/backtest/delta_hedge_2020.parquet
  data_store/backtest/delta_hedge_attribution.png
"""

import logging
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import norm

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
from joint_vol_calibration.models.pdv import extract_pdv_features
from joint_vol_calibration.greeks.risk_monitor import (
    GREEKS_SURFACE_PATH,
    _bs_vomma,
)

logger = logging.getLogger(__name__)

# ── Simulation constants ───────────────────────────────────────────────────────

ENTRY_DATE: str = "2020-01-02"
EXIT_DATE: str  = "2020-12-31"
K_ENTRY: float  = 3257.85    # ATM strike at entry (SPX close 2020-01-02)
T_ENTRY: float  = 1.0        # initial time to expiry (years)
R: float        = 0.015      # 1-year risk-free rate (2020 US Treasury ~1.5%)
Q: float        = 0.013      # SPX annualised dividend yield

# Warmup start: give PDV EWMA (60d halflife) 6+ months to stabilise
PDV_WARMUP_START: str = "2019-07-01"

# Stress test date
COVID_CRASH_DATE: str = "2020-03-16"

# VIX term-structure tenors (calendar days)
VIX_TENORS: OrderedDict = OrderedDict([
    ("^VIX9D", 9.0),
    ("^VIX",  30.0),
    ("^VIX3M", 93.0),
    ("^VIX6M", 182.0),
])

# ── Output paths ──────────────────────────────────────────────────────────────

BACKTEST_DIR    = DATA_DIR / "backtest"
OUTPUT_PARQUET  = BACKTEST_DIR / "delta_hedge_2020.parquet"
OUTPUT_CHART    = BACKTEST_DIR / "delta_hedge_attribution.png"

PDV_MODEL_PATH  = DATA_DIR / "pdv_model.pkl"
VOMMA_THRESHOLD_SIGMA: float = 2.0


# ── BS straddle helpers ────────────────────────────────────────────────────────

def _bs_straddle_value(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> float:
    """
    Black-Scholes ATM straddle price = call + put.

    Uses put-call parity: P = C − S·exp(−qT) + K·exp(−rT)
    so straddle = 2C − S·exp(−qT) + K·exp(−rT).
    """
    if T <= 0 or sigma <= 0:
        # At or past expiry: intrinsic value only
        return abs(S - K)

    sqrtT = np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    eqT = np.exp(-q * T)
    erT = np.exp(-r * T)

    call = S * eqT * norm.cdf(d1) - K * erT * norm.cdf(d2)
    put  = call - S * eqT + K * erT   # put-call parity
    return float(call + put)


def _bs_straddle_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> Tuple[float, float, float, float]:
    """
    Black-Scholes Greeks for a long straddle (call + put).

    Returns (delta, gamma, vega, theta) where:

      delta  = exp(−qT)·(2N(d₁)−1)
               ≈ 0 for ATM,  > 0 for K < F,  < 0 for K > F

      gamma  = 2·exp(−qT)·N′(d₁) / (S·σ·√T)
               always positive (long convexity)

      vega   = 2·S·exp(−qT)·N′(d₁)·√T        (per unit change in σ)
               always positive

      theta  = −S·exp(−qT)·N′(d₁)·σ/√T
               − r·K·exp(−rT)·(2N(d₂)−1)
               + q·S·exp(−qT)·(2N(d₁)−1)
               expressed per year (negative = time decay for long straddle)
    """
    T = max(T, 1e-8)
    sqrtT = np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    eqT = np.exp(-q * T)
    erT = np.exp(-r * T)
    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)

    delta = eqT * (2.0 * Nd1 - 1.0)
    gamma = 2.0 * eqT * nd1 / (S * sigma * sqrtT)
    vega  = 2.0 * S * eqT * nd1 * sqrtT

    # Theta per year (sum of call theta + put theta)
    theta = (
        -(S * eqT * nd1 * sigma) / sqrtT
        - r * K * erT * (2.0 * Nd2 - 1.0)
        + q * S * eqT * (2.0 * Nd1 - 1.0)
    )
    return delta, gamma, vega, theta


# ── VIX term-structure interpolation ──────────────────────────────────────────

def _interp_atm_iv(
    vix_row: pd.Series,
    T_rem_years: float,
) -> float:
    """
    Interpolate ATM implied volatility from the VIX term structure.

    Method: linear interpolation in **total-variance** space
            TV(T) = σ(T)² · T  →  σ_interp = √(TV_interp / T)

    VIX columns used: ^VIX9D (9d), ^VIX (30d), ^VIX3M (93d), ^VIX6M (182d).
    VIX values are quoted as annualised % → divide by 100 to get σ decimal.

    Edge cases:
      T < 9d      : flat extrapolation below (use VIX9D)
      T > 182d    : flat extrapolation above (use VIX6M)
      Any NaN     : use nearest non-NaN tenor
    """
    T_days = T_rem_years * 365.25

    # Build list of valid (days, sigma) pairs
    pairs = []
    for col, days in VIX_TENORS.items():
        v = vix_row.get(col, np.nan)
        if pd.notna(v) and v > 0:
            pairs.append((float(days), float(v) / 100.0))

    if not pairs:
        return np.nan

    pairs.sort(key=lambda x: x[0])

    # Flat extrapolation below / above
    if T_days <= pairs[0][0]:
        return pairs[0][1]
    if T_days >= pairs[-1][0]:
        return pairs[-1][1]

    # Interpolate in total-variance space
    for i in range(len(pairs) - 1):
        t1, iv1 = pairs[i]
        t2, iv2 = pairs[i + 1]
        if t1 <= T_days <= t2:
            tv1 = iv1**2 * t1
            tv2 = iv2**2 * t2
            frac = (T_days - t1) / (t2 - t1)
            tv_interp = tv1 + frac * (tv2 - tv1)
            return float(np.sqrt(max(tv_interp / T_days, 1e-8)))

    return float(pairs[-1][1])  # fallback


# ── PDV forecast extraction ────────────────────────────────────────────────────

def _compute_pdv_forecasts(
    spx_df: pd.DataFrame,
    pdv_model,
) -> pd.Series:
    """
    Compute PDV linear vol forecasts for all dates in spx_df.

    Uses extract_pdv_features (EWMA with 60d halflife) then
    calls pdv_model.linear_.predict(X[['sigma1','sigma2','lev']]).

    Zero look-ahead: EWMA at time t uses returns r_1, ..., r_t only.
    The forecast σ_PDV(t) represents "vol expected from t+1 onwards"
    using only information available at close of day t.

    Parameters
    ----------
    spx_df    : DataFrame with columns ['date', 'log_return'] (date column, not index)
    pdv_model : PDVModel instance loaded from pdv_model.pkl

    Returns
    -------
    pd.Series indexed by date (Timestamp), values = annualised vol (decimal)
    """
    # extract_pdv_features expects a Series indexed by date
    log_ret = spx_df.set_index("date")["log_return"].dropna()
    feats = extract_pdv_features(log_ret)    # index = dates

    xcols = ["sigma1", "sigma2", "lev"]
    X = feats[xcols].dropna()

    vol_series = pdv_model.linear_.predict(X)   # pd.Series with same index
    return vol_series


# ── Vomma flag calibration from C6 surface ────────────────────────────────────

def _load_vomma_surface_stats() -> Tuple[float, float]:
    """
    Load the C6 greeks_surface.parquet and return (mean, std) of vomma.

    Used to z-score daily vomma values and flag unstable hedge days.
    Returns (np.nan, np.nan) if the parquet file does not exist.
    """
    if not GREEKS_SURFACE_PATH.exists():
        logger.warning("C6 greeks surface not found at %s; vomma flags disabled", GREEKS_SURFACE_PATH)
        return np.nan, np.nan

    df = pd.read_parquet(GREEKS_SURFACE_PATH)
    mu  = float(df["vomma"].mean())
    std = float(df["vomma"].std())
    return mu, std


# ── Main simulation ────────────────────────────────────────────────────────────

def run_simulation(
    entry_date: str = ENTRY_DATE,
    exit_date:  str = EXIT_DATE,
    k_entry:    float = K_ENTRY,
    r: float    = R,
    q: float    = Q,
    t_entry:    float = T_ENTRY,
    pdv_model_path: Optional[Path] = None,
    vomma_threshold: float = VOMMA_THRESHOLD_SIGMA,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Simulate daily delta-hedging of a long ATM SPX straddle.

    Returns a DataFrame with one row per trading day (entry to exit inclusive),
    containing P&L attribution columns for both Run A and Run B.

    Parameters
    ----------
    entry_date      : str 'YYYY-MM-DD' — simulation entry date
    exit_date       : str 'YYYY-MM-DD' — simulation exit date
    k_entry         : float — ATM strike at entry (S at entry close)
    r               : float — risk-free rate (annualised, decimal)
    q               : float — dividend yield (annualised, decimal)
    t_entry         : float — initial time to expiry (years)
    pdv_model_path  : Path  — path to pdv_model.pkl (None → use default)
    vomma_threshold : float — number of σ for unstable flag (default 2.0)

    Returns
    -------
    pd.DataFrame — see column list in module docstring.
    """
    if pdv_model_path is None:
        pdv_model_path = PDV_MODEL_PATH

    # ── Load data ──────────────────────────────────────────────────────────────
    # Need look-back for PDV EWMA warmup; get data from warmup start
    logger.info("Loading SPX and VIX data for simulation ...")
    spx_full = get_spx_ohlcv(as_of_date=exit_date, start_date=PDV_WARMUP_START)
    vix_full = get_vix_term_structure_wide(as_of_date=exit_date, start_date=PDV_WARMUP_START)

    # Align and index by date
    spx_full["date"] = pd.to_datetime(spx_full["date"])
    vix_full["date"] = pd.to_datetime(vix_full["date"])

    # Filter to simulation window for the main loop (inclusive)
    entry_ts = pd.Timestamp(entry_date)
    exit_ts  = pd.Timestamp(exit_date)

    spx_sim = spx_full[
        (spx_full["date"] >= entry_ts) & (spx_full["date"] <= exit_ts)
    ].reset_index(drop=True)

    vix_sim = vix_full[
        (vix_full["date"] >= entry_ts) & (vix_full["date"] <= exit_ts)
    ].set_index("date")

    if len(spx_sim) == 0:
        raise ValueError(f"No SPX data found between {entry_date} and {exit_date}")

    # ── PDV forecasts ──────────────────────────────────────────────────────────
    logger.info("Computing PDV forecasts ...")
    pdv_vol: pd.Series = pd.Series(dtype=float)
    if pdv_model_path.exists():
        with open(pdv_model_path, "rb") as fh:
            pdv_model = pickle.load(fh)
        try:
            pdv_vol = _compute_pdv_forecasts(spx_full, pdv_model)
        except Exception as exc:
            logger.warning("PDV forecast failed: %s — using NaN", exc)
    else:
        logger.warning("PDV model not found at %s — Run B will use NaN", pdv_model_path)

    # ── Vomma surface stats from C6 ────────────────────────────────────────────
    vomma_mu, vomma_std = _load_vomma_surface_stats()

    # ── Entry straddle value & state ──────────────────────────────────────────
    # entry_dates row: date = ENTRY_DATE
    entry_row = spx_sim.iloc[0]
    S_entry  = float(entry_row["close"])
    T0_rem   = t_entry    # initial T_rem

    # Get entry-day VIX IV
    if entry_ts in vix_sim.index:
        entry_vix_row = vix_sim.loc[entry_ts]
    else:
        entry_vix_row = pd.Series(dtype=float)
    sigma0 = _interp_atm_iv(entry_vix_row, T0_rem)

    # PDV forecast at entry
    pdv0 = float(pdv_vol.get(entry_ts, np.nan)) if len(pdv_vol) > 0 else np.nan

    V0 = _bs_straddle_value(S_entry, k_entry, T0_rem, r, q, sigma0 if not np.isnan(sigma0) else 0.2)

    # Compute entry-day Greeks (used to hedge on day 2)
    sigma0_for_greeks = sigma0 if not np.isnan(sigma0) else 0.2
    delta0, gamma0, vega0, theta0 = _bs_straddle_greeks(S_entry, k_entry, T0_rem, r, q, sigma0_for_greeks)

    # ── Simulation loop ────────────────────────────────────────────────────────
    records = []

    # Initialise "previous" state
    prev_S     = S_entry
    prev_sigma = sigma0_for_greeks
    prev_pdv   = pdv0
    prev_V     = V0
    prev_delta = delta0
    prev_gamma = gamma0
    prev_vega  = vega0
    prev_theta = theta0
    prev_T_rem = T0_rem
    prev_vomma = _compute_vomma_at(S_entry, k_entry, T0_rem, r, q, sigma0_for_greeks)

    # Entry day row — no P&L yet (this is the initialisation day)
    records.append({
        "date": entry_ts,
        "S": S_entry,
        "T_rem": T0_rem,
        "sigma_atm": prev_sigma,
        "sigma_pdv": prev_pdv,
        "pdv_ratio": prev_pdv / prev_sigma if (not np.isnan(prev_pdv) and prev_sigma > 0) else np.nan,
        "straddle_value": V0,
        "delta": prev_delta,
        "gamma": prev_gamma,
        "vega":  prev_vega,
        "theta": prev_theta,
        "dS": np.nan,
        "dsigma": np.nan,
        "pnl_total":      np.nan,
        "pnl_gamma":      np.nan,
        "pnl_vega_a":     np.nan,
        "pnl_vega_b":     np.nan,
        "pnl_theta":      np.nan,
        "pnl_residual_a": np.nan,
        "pnl_residual_b": np.nan,
        "vomma": prev_vomma,
        "vomma_zscore": _zscore(prev_vomma, vomma_mu, vomma_std),
        "is_unstable": _is_unstable(_zscore(prev_vomma, vomma_mu, vomma_std), vomma_threshold),
    })

    # Loop over days 2..N  (day 1 is entry, day 2+ is where we compute P&L)
    for i in range(1, len(spx_sim)):
        row  = spx_sim.iloc[i]
        t    = pd.Timestamp(row["date"])
        S_t  = float(row["close"])

        # Remaining time to expiry at current day
        days_elapsed = (t - entry_ts).days
        T_rem_t = max(t_entry - days_elapsed / 365.25, 1e-4)

        # ATM IV: use VIX term structure interpolated to T_rem_t
        if t in vix_sim.index:
            vix_row_t = vix_sim.loc[t]
        else:
            vix_row_t = pd.Series(dtype=float)
        sigma_t = _interp_atm_iv(vix_row_t, T_rem_t)
        sigma_t = sigma_t if not np.isnan(sigma_t) else prev_sigma  # carry-forward fallback

        # PDV forecast for this day (look-ahead guard: use prev_pdv for hedge)
        pdv_t = float(pdv_vol.get(t, np.nan)) if len(pdv_vol) > 0 else np.nan

        # Current straddle value
        V_t = _bs_straddle_value(S_t, k_entry, T_rem_t, r, q, sigma_t)

        # ── P&L attribution (using PREVIOUS day's Greeks for zero look-ahead) ──
        dS     = S_t - prev_S
        dsigma = sigma_t - prev_sigma

        # Total delta-hedged P&L
        pnl_total = (V_t - prev_V) - prev_delta * dS

        # Gamma P&L: ½ Γ_{t-1} ΔS²
        pnl_gamma = 0.5 * prev_gamma * dS**2

        # Theta P&L: θ_{t-1} / 252  (negative = decay)
        pnl_theta = prev_theta / 252.0

        # Vega P&L — Run A (raw Heston/market vega)
        pnl_vega_a = prev_vega * dsigma

        # Vega P&L — Run B (PDV-adjusted)
        if not np.isnan(prev_pdv) and prev_sigma > 0:
            pdv_ratio_prev = prev_pdv / prev_sigma
        else:
            pdv_ratio_prev = 1.0   # fallback: same as Run A
        pnl_vega_b = prev_vega * pdv_ratio_prev * dsigma

        # Residuals
        pnl_residual_a = pnl_total - pnl_gamma - pnl_vega_a - pnl_theta
        pnl_residual_b = pnl_total - pnl_gamma - pnl_vega_b - pnl_theta

        # ── Greeks for next day's hedge (zero look-ahead: computed now, used tomorrow) ──
        delta_t, gamma_t, vega_t, theta_t = _bs_straddle_greeks(S_t, k_entry, T_rem_t, r, q, sigma_t)

        # Vomma for stability flag (computed at current node)
        vomma_t = _compute_vomma_at(S_t, k_entry, T_rem_t, r, q, sigma_t)
        vomma_z = _zscore(vomma_t, vomma_mu, vomma_std)

        # PDV ratio (today, for labelling)
        pdv_ratio_t = pdv_t / sigma_t if (not np.isnan(pdv_t) and sigma_t > 0) else np.nan

        records.append({
            "date": t,
            "S": S_t,
            "T_rem": T_rem_t,
            "sigma_atm": sigma_t,
            "sigma_pdv": pdv_t,
            "pdv_ratio": pdv_ratio_t,
            "straddle_value": V_t,
            "delta": delta_t,
            "gamma": gamma_t,
            "vega":  vega_t,
            "theta": theta_t,
            "dS": dS,
            "dsigma": dsigma,
            "pnl_total":      pnl_total,
            "pnl_gamma":      pnl_gamma,
            "pnl_vega_a":     pnl_vega_a,
            "pnl_vega_b":     pnl_vega_b,
            "pnl_theta":      pnl_theta,
            "pnl_residual_a": pnl_residual_a,
            "pnl_residual_b": pnl_residual_b,
            "vomma": vomma_t,
            "vomma_zscore": vomma_z,
            "is_unstable": _is_unstable(vomma_z, vomma_threshold),
        })

        # Advance state
        prev_S     = S_t
        prev_sigma = sigma_t
        prev_pdv   = pdv_t
        prev_V     = V_t
        prev_delta = delta_t
        prev_gamma = gamma_t
        prev_vega  = vega_t
        prev_theta = theta_t
        prev_T_rem = T_rem_t

    # ── Build output DataFrame ─────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    # Cumulative P&L (sum of all days; NaN on entry day is treated as 0)
    df["cum_pnl_a"] = df["pnl_total"].fillna(0.0).cumsum()
    df["cum_pnl_b"] = (
        df["pnl_gamma"].fillna(0) +
        df["pnl_vega_b"].fillna(0) +
        df["pnl_theta"].fillna(0) +
        df["pnl_residual_b"].fillna(0)
    ).cumsum()

    # Hedge efficiency: Var(residual) / Var(total P&L)
    valid = df["pnl_total"].notna()
    if valid.sum() > 1:
        var_total   = df.loc[valid, "pnl_total"].var()
        df.attrs["hedge_efficiency_a"] = (
            df.loc[valid, "pnl_residual_a"].var() / var_total
            if var_total > 0 else np.nan
        )
        df.attrs["hedge_efficiency_b"] = (
            df.loc[valid, "pnl_residual_b"].var() / var_total
            if var_total > 0 else np.nan
        )
    else:
        df.attrs["hedge_efficiency_a"] = np.nan
        df.attrs["hedge_efficiency_b"] = np.nan

    logger.info(
        "Simulation complete: %d trading days | "
        "Hedge efficiency A=%.3f  B=%.3f | "
        "Unstable days: %d",
        len(df),
        df.attrs["hedge_efficiency_a"],
        df.attrs["hedge_efficiency_b"],
        int(df["is_unstable"].sum()),
    )
    return df


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_vomma_at(S, K, T, r, q, sigma) -> float:
    """Compute vomma (scalar) at a single (S, K, T, r, q, sigma) node."""
    return float(_bs_vomma(
        np.array([S]), np.array([K]), np.array([T]),
        np.array([r]), np.array([q]), np.array([sigma])
    )[0])


def _zscore(value: float, mu: float, std: float) -> float:
    """Z-score of value against (mu, std); returns NaN if std is invalid."""
    if np.isnan(mu) or np.isnan(std) or std <= 0:
        return np.nan
    return (value - mu) / std


def _is_unstable(zscore: float, threshold: float) -> bool:
    """True if |z| > threshold."""
    if np.isnan(zscore):
        return False
    return bool(abs(zscore) > threshold)


# ── Stress test ────────────────────────────────────────────────────────────────

def stress_test(df: pd.DataFrame, stress_date: str = COVID_CRASH_DATE) -> dict:
    """
    Extract daily P&L breakdown for a specific stress date.

    Parameters
    ----------
    df          : output of run_simulation()
    stress_date : str 'YYYY-MM-DD'

    Returns
    -------
    dict with P&L components, Greeks, and PDV ratio on that day.
    """
    ts = pd.Timestamp(stress_date)
    if ts not in df.index:
        raise KeyError(f"{stress_date} not found in simulation output (no trading day?)")

    row = df.loc[ts]
    result = {
        "date": stress_date,
        "S": row["S"],
        "dS": row["dS"],
        "dsigma": row["dsigma"],
        "sigma_atm": row["sigma_atm"],
        "sigma_pdv": row["sigma_pdv"],
        "pdv_ratio": row["pdv_ratio"],
        "straddle_value": row["straddle_value"],
        "delta_t_minus_1": df.iloc[df.index.get_loc(ts) - 1]["delta"] if df.index.get_loc(ts) > 0 else np.nan,
        "gamma_t_minus_1": df.iloc[df.index.get_loc(ts) - 1]["gamma"] if df.index.get_loc(ts) > 0 else np.nan,
        "vega_t_minus_1":  df.iloc[df.index.get_loc(ts) - 1]["vega"]  if df.index.get_loc(ts) > 0 else np.nan,
        "pnl_total":      row["pnl_total"],
        "pnl_gamma":      row["pnl_gamma"],
        "pnl_vega_a":     row["pnl_vega_a"],
        "pnl_vega_b":     row["pnl_vega_b"],
        "pnl_theta":      row["pnl_theta"],
        "pnl_residual_a": row["pnl_residual_a"],
        "pnl_residual_b": row["pnl_residual_b"],
        "is_unstable": row["is_unstable"],
    }
    return result


# ── P&L attribution chart ──────────────────────────────────────────────────────

def plot_attribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Plot stacked daily P&L attribution and cumulative P&L.

    Top panel:    Stacked bar chart — Gamma, Vega_A, Theta, Residual_A (Run A)
    Middle panel: Stacked bar chart — Gamma, Vega_B, Theta, Residual_B (Run B)
    Bottom panel: Cumulative P&L comparison A vs B, with vomma flags (red dots)

    Parameters
    ----------
    df        : output of run_simulation()
    save_path : Path — if given, save figure; else return without saving.
    """
    daily = df.dropna(subset=["pnl_total"]).copy()
    dates = daily.index

    # Colour palette
    c_gamma   = "#2196F3"   # blue
    c_vega    = "#4CAF50"   # green
    c_theta   = "#FF9800"   # orange
    c_res_a   = "#9C27B0"   # purple
    c_res_b   = "#E91E63"   # pink
    c_a       = "#1565C0"
    c_b       = "#2E7D32"

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    def _stacked_bar(ax, gamma, vega, theta, residual, vega_label, res_label, res_colour, title):
        """Draw a stacked P&L bar chart on ax."""
        x = np.arange(len(dates))
        ax.bar(x, gamma,    label="Gamma",   color=c_gamma,    alpha=0.85, zorder=2)
        ax.bar(x, vega,     label=vega_label, color=c_vega,    alpha=0.85, zorder=2,
               bottom=gamma)
        ax.bar(x, theta,    label="Theta",   color=c_theta,    alpha=0.85, zorder=2,
               bottom=gamma + vega)
        ax.bar(x, residual, label=res_label, color=res_colour, alpha=0.75, zorder=2,
               bottom=gamma + vega + theta)
        ax.axhline(0, color="black", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("P&L ($)")
        ax.set_xticks(x[::max(1, len(x)//12)])
        ax.set_xticklabels(
            [str(dates[j].date()) for j in range(0, len(dates), max(1, len(dates)//12))],
            rotation=35, ha="right", fontsize=8,
        )
        ax.legend(loc="upper left", fontsize=8, ncol=4)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # ── Run A
    ax0 = fig.add_subplot(gs[0])
    _stacked_bar(
        ax0,
        daily["pnl_gamma"].values,
        daily["pnl_vega_a"].values,
        daily["pnl_theta"].values,
        daily["pnl_residual_a"].values,
        vega_label="Vega (Heston IV)",
        res_label="Residual A",
        res_colour=c_res_a,
        title="Run A — Heston-IV Vega Attribution",
    )

    # ── Run B
    ax1 = fig.add_subplot(gs[1])
    _stacked_bar(
        ax1,
        daily["pnl_gamma"].values,
        daily["pnl_vega_b"].values,
        daily["pnl_theta"].values,
        daily["pnl_residual_b"].values,
        vega_label="Vega (PDV-adjusted)",
        res_label="Residual B",
        res_colour=c_res_b,
        title="Run B — PDV-Adjusted Vega Attribution",
    )

    # ── Cumulative P&L + vomma flags
    ax2 = fig.add_subplot(gs[2])
    cum_a = daily["cum_pnl_a"].values
    cum_b = daily["cum_pnl_b"].values
    x = np.arange(len(dates))

    ax2.plot(x, cum_a, color=c_a, linewidth=1.6, label="Cumulative P&L — Run A")
    ax2.plot(x, cum_b, color=c_b, linewidth=1.6, linestyle="--", label="Cumulative P&L — Run B")

    # Overlay vomma flags (red dots)
    flag_idx = np.where(daily["is_unstable"].values)[0]
    if len(flag_idx):
        ax2.scatter(
            flag_idx,
            [max(cum_a.max(), cum_b.max()) * 0.98] * len(flag_idx),
            color="red", s=25, zorder=5, label=f"Unstable hedge ({len(flag_idx)} days)",
        )

    # Mark COVID crash
    covid_ts = pd.Timestamp(COVID_CRASH_DATE)
    if covid_ts in daily.index:
        ci = daily.index.get_loc(covid_ts)
        ax2.axvline(ci, color="red", linewidth=1.0, linestyle=":", alpha=0.7)
        ax2.text(ci + 0.5, ax2.get_ylim()[0], "COVID\nCrash", fontsize=8, color="red", va="bottom")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Cumulative P&L: Run A vs Run B (with vomma flags)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.set_xticks(x[::max(1, len(x)//12)])
    ax2.set_xticklabels(
        [str(dates[j].date()) for j in range(0, len(dates), max(1, len(dates)//12))],
        rotation=35, ha="right", fontsize=8,
    )
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    fig.suptitle(
        f"Delta-Hedged ATM SPX Straddle P&L Attribution  ({ENTRY_DATE} → {EXIT_DATE})\n"
        f"K = {K_ENTRY:.0f}  |  T = 1 year  |  r = {R*100:.1f}%  |  q = {Q*100:.1f}%",
        fontsize=13, fontweight="bold",
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Attribution chart saved to %s", save_path)

    return fig


# ── Persistence ────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save simulation results DataFrame to parquet."""
    if path is None:
        path = OUTPUT_PARQUET
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Results saved to %s (%d rows)", path, len(df))
    return path


def load_results(path: Optional[Path] = None) -> pd.DataFrame:
    """Load simulation results from parquet."""
    if path is None:
        path = OUTPUT_PARQUET
    return pd.read_parquet(path)


# ── Summary metrics ────────────────────────────────────────────────────────────

def compute_hedge_metrics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics comparing Run A and Run B.

    Returns a dict with:
      n_days           : number of trading days (excluding entry day)
      cumulative_pnl_a : total P&L (Run A)
      cumulative_pnl_b : total P&L (Run B)
      hedge_efficiency_a : Var(resid_A) / Var(total)
      hedge_efficiency_b : Var(resid_B) / Var(total)
      pdv_improves_hedge : bool — True if Var(resid_B) < Var(resid_A)
      n_unstable_days  : count of vomma-flagged days
      covid_crash_pnl  : total P&L on COVID crash date (or NaN)
    """
    valid = df.dropna(subset=["pnl_total"])
    n = len(valid)

    var_total = valid["pnl_total"].var() if n > 1 else np.nan
    var_res_a = valid["pnl_residual_a"].var() if n > 1 else np.nan
    var_res_b = valid["pnl_residual_b"].var() if n > 1 else np.nan

    eff_a = var_res_a / var_total if (var_total and not np.isnan(var_total)) else np.nan
    eff_b = var_res_b / var_total if (var_total and not np.isnan(var_total)) else np.nan

    covid_ts = pd.Timestamp(COVID_CRASH_DATE)
    covid_pnl = float(df.loc[covid_ts, "pnl_total"]) if covid_ts in df.index else np.nan

    metrics = {
        "n_days": n,
        "cumulative_pnl_a": float(valid["pnl_total"].sum()),
        "cumulative_pnl_b": float(
            (valid["pnl_gamma"] + valid["pnl_vega_b"] + valid["pnl_theta"] + valid["pnl_residual_b"]).sum()
        ),
        "hedge_efficiency_a": eff_a,
        "hedge_efficiency_b": eff_b,
        "pdv_improves_hedge": bool(var_res_b < var_res_a) if not (np.isnan(var_res_a) or np.isnan(var_res_b)) else False,
        "n_unstable_days": int(df["is_unstable"].sum()),
        "covid_crash_pnl": covid_pnl,
    }
    return metrics


# ── DeltaHedger orchestration class ───────────────────────────────────────────

class DeltaHedger:
    """
    Orchestrates the full delta-hedge simulation workflow.

    Usage
    -----
      hedger = DeltaHedger()
      results = hedger.run()
      fig     = hedger.plot()
      path    = hedger.save()
      metrics = hedger.metrics()
      stress  = hedger.stress_test()
    """

    def __init__(
        self,
        entry_date: str = ENTRY_DATE,
        exit_date:  str = EXIT_DATE,
        k_entry:    float = K_ENTRY,
        r:          float = R,
        q:          float = Q,
        t_entry:    float = T_ENTRY,
        pdv_model_path: Optional[Path] = None,
        vomma_threshold: float = VOMMA_THRESHOLD_SIGMA,
    ):
        self.entry_date      = entry_date
        self.exit_date       = exit_date
        self.k_entry         = k_entry
        self.r               = r
        self.q               = q
        self.t_entry         = t_entry
        self.pdv_model_path  = pdv_model_path
        self.vomma_threshold = vomma_threshold
        self._results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Execute simulation and cache results."""
        self._results = run_simulation(
            entry_date=self.entry_date,
            exit_date=self.exit_date,
            k_entry=self.k_entry,
            r=self.r,
            q=self.q,
            t_entry=self.t_entry,
            pdv_model_path=self.pdv_model_path,
            vomma_threshold=self.vomma_threshold,
        )
        return self._results

    def _require_results(self):
        if self._results is None:
            raise RuntimeError("Call .run() before accessing results.")

    def plot(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Generate attribution chart."""
        self._require_results()
        return plot_attribution(
            self._results,
            save_path=save_path or OUTPUT_CHART,
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Save results to parquet."""
        self._require_results()
        return save_results(self._results, path=path or OUTPUT_PARQUET)

    def metrics(self) -> dict:
        """Return summary hedge metrics."""
        self._require_results()
        return compute_hedge_metrics(self._results)

    def stress_test(self, date: str = COVID_CRASH_DATE) -> dict:
        """Return P&L breakdown for a specific date."""
        self._require_results()
        return stress_test(self._results, stress_date=date)

    def __repr__(self) -> str:
        status = "ready" if self._results is not None else "not run"
        return (
            f"DeltaHedger(entry={self.entry_date}, exit={self.exit_date}, "
            f"K={self.k_entry:.0f}, status={status})"
        )
