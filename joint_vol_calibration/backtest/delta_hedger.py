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
  Self-financing cash account: −V + Δ·S accrues r; the stock leg pays q.

Two attribution runs (ONE hedge)
--------------------------------
  Run A  — VIX-term-structure Greeks
              vol = atm_to_vix_ratio × VIX term structure interpolated to
              T_rem. The default ratio is 1.0: the backtest's β (≈0.82) is
              measured at 30 days only and this straddle is 1-year, so the
              C7 study keeps the VIX-TS level and studies hedge mechanics,
              not the vol level.
  Run B  — PDV-scaled vega ATTRIBUTION
              Vega line scaled by  σ_PDV_{t-1} / σ_ATM_{t-1}
              The delta hedge, and therefore the P&L, is identical to Run A;
              Run B only re-labels how much of the P&L is called "vega".
              It is not a different hedge and cannot "over-hedge".

P&L attribution (per trading day)
----------------------------------
  Total = V_t − V_{t-1} − Δ_{t-1}·ΔS + Fin
        = Θ + Γ + ν_X + Vanna + Volga + Fin + Residual_X

  Θ-P&L      =  θ_{t-1} · (T_rem_{t-1} − T_rem_t)     (calendar clock; a
                                                        Friday→Monday step
                                                        decays three days)
  Γ-P&L      =  ½ · Γ_{t-1} · (ΔS)²
  ν-P&L_A    =  ν_{t-1} · Δσ
  ν-P&L_B    =  ν_{t-1} · (σ_PDV_{t-1} / σ_ATM_{t-1}) · Δσ
  Vanna-P&L  =  vanna_{t-1} · ΔS · Δσ
  Volga-P&L  =  ½ · volga_{t-1} · (Δσ)²
  Fin-P&L    =  [r·(Δ_{t-1}·S_{t-1} − V_{t-1}) − q·Δ_{t-1}·S_{t-1}] · ΔT
  Residual_X =  Total − (Θ + Γ + ν_X + Vanna + Volga + Fin)

  In a Black-Scholes world with constant σ the residual is third-order
  (speed, charm): Θ + Fin = −½σ²S²Γ·ΔT by the BS PDE, so the explained
  part is ½Γ(ΔS² − σ²S²ΔT), the textbook hedged-straddle P&L.

Unstable-hedge flag (vomma)
---------------------------
  Straddle vomma (volga) = ν · d1·d2/σ  (put vomma = call vomma, so the
  straddle carries twice a call's). The flag measures the EXPECTED volga
  P&L for the next day, ½·|volga_t|·Var_20d(Δσ), i.e. vol-of-vol risk in
  dollars, and z-scores it against the position's own trailing history
  (up to 252 sessions, at least 20; ex-ante: the window ends at t−1).
  Unstable day: z > 2. On the 2020 straddle the flag fires on 29 days, all
  in the Feb–Apr COVID window, and the next day's realised |volga P&L| on
  flagged days is ~20× that on other days.

  (Before C22 the flag z-scored a single CALL's vomma at 2020 spot against
  the C6 surface built at S≈6,581 across all strikes and maturities. Vomma
  is linear in S, so the 2020 book sat at z<0 every day and the flag could
  never fire — 0/253 days, including the COVID crash.)

Look-ahead
----------
  All Greeks (Δ, Γ, ν, θ, vanna, volga) are computed at close of day t-1.
  Run B's σ_PDV comes from a PDVLinear fitted ONLY on returns strictly
  before the entry date (walk-forward), not the full-sample pdv_model.pkl,
  whose coefficients were fitted on 2010–2025 including 2020. The training
  end date is recorded in ``df.attrs["pdv_train_end"]``.

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

from joint_vol_calibration.config import DATA_DIR, HIST_START, VIX_INDEX_TENOR_DAYS
from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
from joint_vol_calibration.models.pdv import extract_pdv_features
from joint_vol_calibration.greeks.risk_monitor import _bs_vomma

logger = logging.getLogger(__name__)

# ── Simulation constants ───────────────────────────────────────────────────────

ENTRY_DATE: str = "2020-01-02"
EXIT_DATE: str  = "2020-12-31"
K_ENTRY: float  = 3257.85    # ATM strike at entry (SPX close 2020-01-02)
T_ENTRY: float  = 1.0        # initial time to expiry (years)
R: float        = 0.015      # 1-year risk-free rate (2020 US Treasury ~1.5%)
Q: float        = 0.013      # SPX annualised dividend yield

# Calendar-day year used for time to expiry. Cboe quotes the VIX family on
# a 365-day year, and the backtest engine's tenors are calendar days / 365,
# so the interpolation and the C7 clock use the same convention.
DAYS_PER_YEAR: float = 365.0

# Warmup start for the simulation window's EWMA state (the walk-forward PDV
# fit itself reads the full history from HIST_START).
PDV_WARMUP_START: str = "2019-07-01"

# Minimum training sample for a walk-forward PDV fit (≈ 2 years of sessions)
PDV_MIN_TRAIN_OBS: int = 504

# Stress test date
COVID_CRASH_DATE: str = "2020-03-16"

# VIX term-structure tenors (calendar days) — the single source of truth is
# config.VIX_INDEX_TENOR_DAYS (Cboe methodology: 9 / 30 / 93 / 184 days).
VIX_TENORS: OrderedDict = OrderedDict(
    (tk, float(VIX_INDEX_TENOR_DAYS[tk])) for tk in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")
)

# ── Output paths ──────────────────────────────────────────────────────────────

BACKTEST_DIR    = DATA_DIR / "backtest"
OUTPUT_PARQUET  = BACKTEST_DIR / "delta_hedge_2020.parquet"
OUTPUT_CHART    = BACKTEST_DIR / "delta_hedge_attribution.png"

PDV_MODEL_PATH  = DATA_DIR / "pdv_model.pkl"
VOMMA_THRESHOLD_SIGMA: float = 2.0
VOMMA_VOLVOL_WINDOW: int = 20     # trailing window for Var(Δσ) (vol-of-vol)
VOMMA_Z_WINDOW: int = 252         # trailing window for the flag's z-score (≈ the position's history)
VOMMA_Z_MIN_PERIODS: int = 20


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


def _bs_straddle_cross_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> Tuple[float, float]:
    """
    Second-order vol Greeks of a long straddle: (vanna, volga).

      vanna = ∂²V/∂S∂σ = −2·exp(−qT)·N′(d₁)·d₂/σ
      volga = ∂²V/∂σ²  = vega_straddle · d₁·d₂/σ      (a.k.a. vomma)

    Call and put share both (put-call parity is linear in S and free of σ),
    so the straddle carries exactly twice a call's vanna and volga.
    """
    T = max(T, 1e-8)
    sqrtT = np.sqrt(T)
    F = S * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    eqT = np.exp(-q * T)
    nd1 = norm.pdf(d1)
    vega = 2.0 * S * eqT * nd1 * sqrtT
    vanna = -2.0 * eqT * nd1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return float(vanna), float(volga)


# ── VIX term-structure interpolation ──────────────────────────────────────────

def _interp_atm_iv(
    vix_row: pd.Series,
    T_rem_years: float,
) -> float:
    """
    Interpolate the VIX-family variance-swap vol to horizon T_rem.

    Method: linear interpolation in **total-variance** space
            TV(T) = σ(T)² · T  →  σ_interp = √(TV_interp / T)

    VIX columns used: ^VIX9D, ^VIX, ^VIX3M, ^VIX6M at the Cboe tenors in
    config.VIX_INDEX_TENOR_DAYS (9 / 30 / 93 / 184 calendar days); T is
    converted to days on the same 365-day year.
    VIX values are quoted as annualised % → divide by 100 to get σ decimal.

    NOTE: the result is a VARIANCE-SWAP vol, not ATM implied vol. VIX
    contains the skew/convexity premium and sits above ATM IV (about 0.8×
    on the SPX chains measured in backtest_engine.ATM_TO_VIX_RATIO).
    Callers that price ATM options must map it (σ_ATM = β·VIX).

    Edge cases:
      T < 9d      : flat extrapolation below (use VIX9D)
      T > 184d    : flat extrapolation above (use VIX6M)
      Any NaN     : use nearest non-NaN tenor
    """
    T_days = T_rem_years * DAYS_PER_YEAR

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

def fit_pdv_before(
    log_returns: pd.Series,
    cutoff: pd.Timestamp,
    horizon: int = 1,
    min_obs: int = PDV_MIN_TRAIN_OBS,
):
    """
    Fit a PDVLinear on log-returns STRICTLY before ``cutoff`` (walk-forward).

    Shared by the C7 Run B attribution and by BacktestEngine's per-year
    walk-forward PDV, so both use the same zero-look-ahead protocol.

    The regression target comes from models.pdv.build_xy(horizon=horizon):
    horizon 1 is |r_{t+1}|·√252 (an E|r| forecast, ≈0.8σ under a Gaussian);
    larger horizons are forward RMS vol. Rows whose target would need
    returns at or after the cutoff are NaN in build_xy and dropped, so no
    post-cutoff return enters the fit.

    Returns
    -------
    (model, train_end) — the fitted PDVLinear and the date of the last
    training return, or (None, None) when fewer than ``min_obs`` returns
    precede the cutoff.
    """
    from joint_vol_calibration.models.pdv import PDVLinear, build_xy

    r = log_returns.dropna()
    r = r[~r.index.duplicated(keep="first")].sort_index()
    r_train = r[r.index < pd.Timestamp(cutoff)]
    if len(r_train) < min_obs:
        return None, None
    feats_train = extract_pdv_features(r_train)
    X_tr, y_tr = build_xy(r_train, feats_train, horizon=horizon)
    model = PDVLinear().fit(X_tr, y_tr)
    return model, pd.Timestamp(r_train.index.max())


def _compute_pdv_forecasts(
    spx_df: pd.DataFrame,
    pdv_model,
) -> pd.Series:
    """
    Compute PDV linear vol forecasts for all dates in spx_df.

    Uses extract_pdv_features (EWMA with 60d halflife) then the model's
    predict on X[['sigma1','sigma2','lev']]. ``pdv_model`` may be a fitted
    PDVLinear (walk-forward, see fit_pdv_before) or a pickled PDVModel
    exposing ``.linear_``.

    Features at t use returns r_1..r_t only (EWMA). Whether the FORECAST
    is look-ahead-free depends on the model: a model fitted on data after
    t leaks the future through its coefficients.

    Parameters
    ----------
    spx_df    : DataFrame with columns ['date', 'log_return'] (date column, not index)
    pdv_model : PDVLinear, or PDVModel with a fitted ``linear_``

    Returns
    -------
    pd.Series indexed by date (Timestamp), values = annualised vol (decimal)
    """
    # extract_pdv_features expects a Series indexed by date
    log_ret = spx_df.set_index("date")["log_return"].dropna()
    feats = extract_pdv_features(log_ret)    # index = dates

    xcols = ["sigma1", "sigma2", "lev"]
    X = feats[xcols].dropna()

    model = getattr(pdv_model, "linear_", pdv_model)
    vol_series = model.predict(X)   # pd.Series with same index
    return vol_series


# ── Main simulation ────────────────────────────────────────────────────────────

def hedge_attribution(
    dates:     pd.DatetimeIndex,
    S:         np.ndarray,
    sigma:     np.ndarray,
    k_entry:   float,
    t_entry:   float,
    r:         float,
    q:         float,
    sigma_pdv: Optional[np.ndarray] = None,
    vomma_threshold: float = VOMMA_THRESHOLD_SIGMA,
) -> pd.DataFrame:
    """
    Daily delta-hedge P&L and Greek attribution for one long straddle.

    Pure function of the price/vol path (no I/O), so it can be run on real
    data (run_simulation) or on simulated Black-Scholes paths (tests).

    Parameters
    ----------
    dates     : trading dates; the first is the entry close
    S, sigma  : spot and ATM implied vol (decimal) at each close
    k_entry   : strike;  t_entry : years to expiry at the entry close
    r, q      : continuously-compounded rate and dividend yield
    sigma_pdv : optional PDV vol forecast per date (Run B attribution)

    Returns
    -------
    DataFrame indexed by date — see the module docstring for the columns.
    The entry row carries NaN P&L (nothing to attribute yet).
    """
    dates = pd.DatetimeIndex(dates)
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(dates)
    if sigma_pdv is None:
        sigma_pdv = np.full(n, np.nan)
    sigma_pdv = np.asarray(sigma_pdv, dtype=float)

    T = np.maximum(
        t_entry - (dates - dates[0]).days.values / DAYS_PER_YEAR, 1e-4
    )

    records = []
    prev = None
    for i in range(n):
        S_t, sig_t, T_t, pdv_t = S[i], sigma[i], T[i], sigma_pdv[i]
        V_t = _bs_straddle_value(S_t, k_entry, T_t, r, q, sig_t)
        delta_t, gamma_t, vega_t, theta_t = _bs_straddle_greeks(S_t, k_entry, T_t, r, q, sig_t)
        vanna_t, volga_t = _bs_straddle_cross_greeks(S_t, k_entry, T_t, r, q, sig_t)

        rec = {
            "date": dates[i], "S": S_t, "T_rem": T_t,
            "sigma_atm": sig_t, "sigma_pdv": pdv_t,
            "pdv_ratio": pdv_t / sig_t if (np.isfinite(pdv_t) and sig_t > 0) else np.nan,
            "straddle_value": V_t,
            "delta": delta_t, "gamma": gamma_t, "vega": vega_t, "theta": theta_t,
            "vanna": vanna_t, "vomma": volga_t,
        }
        if prev is None:
            rec.update({k: np.nan for k in (
                "dS", "dsigma", "dT", "pnl_total", "pnl_gamma", "pnl_vega_a",
                "pnl_vega_b", "pnl_theta", "pnl_vanna", "pnl_volga",
                "pnl_financing", "pnl_residual_a", "pnl_residual_b",
            )})
        else:
            dS = S_t - prev["S"]
            dsig = sig_t - prev["sigma_atm"]
            dT = prev["T_rem"] - T_t          # > 0: calendar time elapsed

            fin = (r * (prev["delta"] * prev["S"] - prev["straddle_value"])
                   - q * prev["delta"] * prev["S"]) * dT
            pnl_total = (V_t - prev["straddle_value"]) - prev["delta"] * dS + fin

            pnl_gamma = 0.5 * prev["gamma"] * dS ** 2
            pnl_theta = prev["theta"] * dT
            pnl_vega_a = prev["vega"] * dsig
            pdv_ratio_prev = prev["pdv_ratio"] if np.isfinite(prev["pdv_ratio"]) else 1.0
            pnl_vega_b = prev["vega"] * pdv_ratio_prev * dsig
            pnl_vanna = prev["vanna"] * dS * dsig
            pnl_volga = 0.5 * prev["vomma"] * dsig ** 2

            common = pnl_gamma + pnl_theta + pnl_vanna + pnl_volga + fin
            rec.update({
                "dS": dS, "dsigma": dsig, "dT": dT,
                "pnl_total": pnl_total, "pnl_gamma": pnl_gamma,
                "pnl_vega_a": pnl_vega_a, "pnl_vega_b": pnl_vega_b,
                "pnl_theta": pnl_theta, "pnl_vanna": pnl_vanna,
                "pnl_volga": pnl_volga, "pnl_financing": fin,
                "pnl_residual_a": pnl_total - common - pnl_vega_a,
                "pnl_residual_b": pnl_total - common - pnl_vega_b,
            })
        records.append(rec)
        prev = rec

    df = pd.DataFrame(records).set_index("date")
    df.index = pd.to_datetime(df.index)

    # ── Unstable-hedge flag: expected next-day volga P&L vs own history ────
    volvol_var = df["dsigma"].rolling(VOMMA_VOLVOL_WINDOW, min_periods=5).var()
    exp_volga = 0.5 * df["vomma"].abs() * volvol_var
    hist = exp_volga.shift(1).rolling(VOMMA_Z_WINDOW, min_periods=VOMMA_Z_MIN_PERIODS)
    mu, sd = hist.mean(), hist.std()
    z = (exp_volga - mu) / sd.where(sd > 0)
    df["exp_volga_pnl"] = exp_volga
    df["vomma_zscore"] = z
    df["is_unstable"] = (z > vomma_threshold).fillna(False).astype(bool)

    # Cumulative P&L. Run B re-attributes vega only, so its total is the same
    # hedged P&L; cum_pnl_b is kept for schema compatibility.
    df["cum_pnl_a"] = df["pnl_total"].fillna(0.0).cumsum()
    df["cum_pnl_b"] = df["cum_pnl_a"]

    valid = df["pnl_total"].notna()
    var_total = df.loc[valid, "pnl_total"].var() if valid.sum() > 1 else np.nan
    for run in ("a", "b"):
        res = df.loc[valid, f"pnl_residual_{run}"]
        ratio = res.var() / var_total if (valid.sum() > 1 and var_total > 0) else np.nan
        # "hedge_efficiency_*" is the historical name of this ratio; it is the
        # share of daily P&L variance the Greek expansion leaves unexplained.
        df.attrs[f"hedge_efficiency_{run}"] = ratio
        df.attrs[f"unexplained_var_ratio_{run}"] = ratio
        df.attrs[f"cum_residual_{run}"] = float(res.sum())
    return df


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
    pdv_horizon: int = 1,
    atm_to_vix_ratio: float = 1.0,
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
    pdv_model_path  : Path  — OPTIONAL pickled PDVModel for Run B. None (the
                      default) fits a walk-forward PDVLinear on returns
                      strictly before ``entry_date``. Passing the shipped
                      full-sample pdv_model.pkl reintroduces look-ahead and
                      is kept only to reproduce the pre-C22 numbers.
    vomma_threshold : float — z-score threshold for the unstable-hedge flag
    pdv_horizon     : int   — build_xy horizon of the walk-forward PDV fit
    atm_to_vix_ratio: float — vol = ratio × VIX-TS (see module docstring)

    Returns
    -------
    pd.DataFrame — see column list in module docstring. ``attrs`` carry the
    unexplained-variance ratios, cumulative residuals and the PDV training
    end date (``pdv_train_end``; None when the pickled model is used).
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    logger.info("Loading SPX and VIX data for simulation ...")
    spx_full = get_spx_ohlcv(as_of_date=exit_date, start_date=HIST_START)
    vix_full = get_vix_term_structure_wide(as_of_date=exit_date, start_date=PDV_WARMUP_START)

    spx_full["date"] = pd.to_datetime(spx_full["date"])
    vix_full["date"] = pd.to_datetime(vix_full["date"])

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

    dates = pd.DatetimeIndex(spx_sim["date"])
    S = spx_sim["close"].astype(float).values
    T = np.maximum(t_entry - (dates - dates[0]).days.values / DAYS_PER_YEAR, 1e-4)

    # ATM vol from the VIX term structure; carry the last valid value forward
    sigma = np.empty(len(dates))
    last = np.nan
    for i, (t, T_t) in enumerate(zip(dates, T)):
        row = vix_sim.loc[t] if t in vix_sim.index else pd.Series(dtype=float)
        s = atm_to_vix_ratio * _interp_atm_iv(row, T_t)
        if np.isnan(s):
            s = last if np.isfinite(last) else 0.2
        sigma[i] = last = s

    # ── PDV forecasts for Run B ────────────────────────────────────────────────
    pdv_train_end = None
    pdv_vol: pd.Series = pd.Series(dtype=float)
    try:
        if pdv_model_path is not None:
            logger.warning(
                "Run B uses the pickled PDV model at %s — its coefficients may "
                "contain data after %s (look-ahead).", pdv_model_path, entry_date,
            )
            with open(pdv_model_path, "rb") as fh:
                pdv_model = pickle.load(fh)
        else:
            log_ret = spx_full.set_index("date")["log_return"]
            pdv_model, pdv_train_end = fit_pdv_before(log_ret, entry_ts, horizon=pdv_horizon)
            if pdv_model is None:
                logger.warning("Too little pre-%s history for a PDV fit — Run B = Run A", entry_date)
        if pdv_model is not None:
            pdv_vol = _compute_pdv_forecasts(spx_full, pdv_model)
    except Exception as exc:
        logger.warning("PDV forecast failed: %s — Run B = Run A", exc)
    sigma_pdv = pdv_vol.reindex(dates).values if len(pdv_vol) else None

    df = hedge_attribution(
        dates, S, sigma, k_entry, t_entry, r, q,
        sigma_pdv=sigma_pdv, vomma_threshold=vomma_threshold,
    )
    df.attrs["pdv_train_end"] = None if pdv_train_end is None else str(pdv_train_end.date())

    logger.info(
        "Simulation complete: %d trading days | unexplained-variance ratio "
        "A=%.4f  B=%.4f | cum residual A=%.2f | unstable days: %d",
        len(df), df.attrs["unexplained_var_ratio_a"], df.attrs["unexplained_var_ratio_b"],
        df.attrs["cum_residual_a"], int(df["is_unstable"].sum()),
    )
    return df


# ── Internal helpers ───────────────────────────────────────────────────────────

def _compute_vomma_at(S, K, T, r, q, sigma) -> float:
    """Single-CALL vomma at one (S, K, T, r, q, sigma) node (straddle = 2×)."""
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
    loc = df.index.get_loc(ts)
    prev = df.iloc[loc - 1] if loc > 0 else None
    result = {
        "date": stress_date,
        "S": row["S"],
        "dS": row["dS"],
        "dsigma": row["dsigma"],
        "sigma_atm": row["sigma_atm"],
        "sigma_pdv": row["sigma_pdv"],
        "pdv_ratio": row["pdv_ratio"],
        "straddle_value": row["straddle_value"],
        "delta_t_minus_1": prev["delta"] if prev is not None else np.nan,
        "gamma_t_minus_1": prev["gamma"] if prev is not None else np.nan,
        "vega_t_minus_1":  prev["vega"]  if prev is not None else np.nan,
        "pnl_total":      row["pnl_total"],
        "pnl_gamma":      row["pnl_gamma"],
        "pnl_vega_a":     row["pnl_vega_a"],
        "pnl_vega_b":     row["pnl_vega_b"],
        "pnl_theta":      row["pnl_theta"],
        "pnl_residual_a": row["pnl_residual_a"],
        "pnl_residual_b": row["pnl_residual_b"],
        "is_unstable": row["is_unstable"],
    }
    for col in ("pnl_vanna", "pnl_volga", "pnl_financing"):
        if col in df.columns:
            result[col] = row[col]
    return result


# ── P&L attribution chart ──────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    """Column values, or zeros for frames written before the column existed."""
    return df[name].fillna(0.0).values if name in df.columns else np.zeros(len(df))


def plot_attribution(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """
    Plot stacked daily P&L attribution and cumulative P&L.

    Top panel:    Stacked bars — Gamma, Vega_A, Theta+Financing, Vanna+Volga, Residual_A
    Middle panel: Same with the PDV-scaled vega line (Run B)
    Bottom panel: Cumulative P&L and cumulative residuals, with unstable-hedge flags

    Parameters
    ----------
    df        : output of run_simulation()
    save_path : Path — if given, save figure; else return without saving.
    """
    daily = df.dropna(subset=["pnl_total"]).copy()
    dates = daily.index

    c_gamma   = "#2196F3"
    c_vega    = "#4CAF50"
    c_theta   = "#FF9800"
    c_cross   = "#795548"
    c_res_a   = "#9C27B0"
    c_res_b   = "#E91E63"
    c_a       = "#1565C0"

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    carry = _col(daily, "pnl_theta") + _col(daily, "pnl_financing")
    cross = _col(daily, "pnl_vanna") + _col(daily, "pnl_volga")

    def _stacked_bar(ax, vega, residual, vega_label, res_label, res_colour, title):
        x = np.arange(len(dates))
        gamma = _col(daily, "pnl_gamma")
        layers = [(gamma, "Gamma", c_gamma), (vega, vega_label, c_vega),
                  (carry, "Theta + financing", c_theta), (cross, "Vanna + volga", c_cross),
                  (residual, res_label, res_colour)]
        bottom = np.zeros(len(x))
        for vals, lbl, colour in layers:
            ax.bar(x, vals, bottom=bottom, label=lbl, color=colour, alpha=0.85, zorder=2)
            bottom = bottom + vals
        ax.axhline(0, color="black", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("P&L ($)")
        ax.set_xticks(x[::max(1, len(x)//12)])
        ax.set_xticklabels(
            [str(dates[j].date()) for j in range(0, len(dates), max(1, len(dates)//12))],
            rotation=35, ha="right", fontsize=8,
        )
        ax.legend(loc="upper left", fontsize=8, ncol=5)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    ax0 = fig.add_subplot(gs[0])
    _stacked_bar(ax0, _col(daily, "pnl_vega_a"), _col(daily, "pnl_residual_a"),
                 "Vega (VIX-TS IV)", "Residual A", c_res_a,
                 "Run A — VIX-Term-Structure Greek Attribution")

    ax1 = fig.add_subplot(gs[1])
    _stacked_bar(ax1, _col(daily, "pnl_vega_b"), _col(daily, "pnl_residual_b"),
                 "Vega × σ_PDV/σ_ATM", "Residual B", c_res_b,
                 "Run B — PDV-Scaled Vega Attribution (same hedge, same P&L)")

    ax2 = fig.add_subplot(gs[2])
    x = np.arange(len(dates))
    cum = daily["cum_pnl_a"].values
    ax2.plot(x, cum, color=c_a, linewidth=1.6, label="Cumulative hedged P&L")
    ax2.plot(x, np.cumsum(_col(daily, "pnl_residual_a")), color=c_res_a, linewidth=1.2,
             linestyle="--", label="Cumulative residual A")
    ax2.plot(x, np.cumsum(_col(daily, "pnl_residual_b")), color=c_res_b, linewidth=1.2,
             linestyle=":", label="Cumulative residual B")

    flag_idx = np.where(daily["is_unstable"].values)[0]
    if len(flag_idx):
        ax2.scatter(flag_idx, [np.nanmax(cum) * 0.98] * len(flag_idx), color="red", s=25,
                    zorder=5, label=f"Unstable hedge — vol-of-vol flag ({len(flag_idx)} days)")

    covid_ts = pd.Timestamp(COVID_CRASH_DATE)
    if covid_ts in daily.index:
        ci = daily.index.get_loc(covid_ts)
        ax2.axvline(ci, color="red", linewidth=1.0, linestyle=":", alpha=0.7)
        ax2.text(ci + 0.5, ax2.get_ylim()[0], "COVID\nCrash", fontsize=8, color="red", va="bottom")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Cumulative P&L and unexplained residuals", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative P&L ($)")
    ax2.set_xticks(x[::max(1, len(x)//12)])
    ax2.set_xticklabels(
        [str(dates[j].date()) for j in range(0, len(dates), max(1, len(dates)//12))],
        rotation=35, ha="right", fontsize=8,
    )
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    first = df.index[0].date() if len(df) else ENTRY_DATE
    last = df.index[-1].date() if len(df) else EXIT_DATE
    fig.suptitle(
        f"Delta-Hedged ATM SPX Straddle P&L Attribution  ({first} → {last})\n"
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
    Compute summary statistics comparing the Run A and Run B attributions.

    Returns a dict with:
      n_days             : number of trading days (excluding entry day)
      cumulative_pnl_a   : total hedged P&L
      cumulative_pnl_b   : same P&L (Run B re-attributes vega only)
      hedge_efficiency_a : Var(resid_A) / Var(total) — the UNEXPLAINED share
      hedge_efficiency_b : Var(resid_B) / Var(total)   (historical key names)
      unexplained_var_ratio_a / _b : the same ratios under an honest name
      cum_residual_a / _b : Σ residual — must be read next to the ratio: a
                            small variance ratio can hide a residual that
                            rivals the total P&L
      pdv_improves_hedge : bool — True if Var(resid_B) < Var(resid_A), i.e.
                           the PDV-scaled vega line explains MORE variance
      n_unstable_days    : count of vol-of-vol-flagged days
      covid_crash_pnl    : total P&L on COVID crash date (or NaN)
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

    total = float(valid["pnl_total"].sum())
    metrics = {
        "n_days": n,
        "cumulative_pnl_a": total,
        "cumulative_pnl_b": total,
        "hedge_efficiency_a": eff_a,
        "hedge_efficiency_b": eff_b,
        "unexplained_var_ratio_a": eff_a,
        "unexplained_var_ratio_b": eff_b,
        "cum_residual_a": float(valid["pnl_residual_a"].sum()),
        "cum_residual_b": float(valid["pnl_residual_b"].sum()),
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
        pdv_horizon: int = 1,
        atm_to_vix_ratio: float = 1.0,
    ):
        self.entry_date      = entry_date
        self.exit_date       = exit_date
        self.k_entry         = k_entry
        self.r               = r
        self.q               = q
        self.t_entry         = t_entry
        self.pdv_model_path  = pdv_model_path
        self.vomma_threshold = vomma_threshold
        self.pdv_horizon     = pdv_horizon
        self.atm_to_vix_ratio = atm_to_vix_ratio
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
            pdv_horizon=self.pdv_horizon,
            atm_to_vix_ratio=self.atm_to_vix_ratio,
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
