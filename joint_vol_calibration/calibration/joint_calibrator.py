"""
joint_calibrator.py — Joint SPX/VIX Smile Calibration Engine (C4).

Minimises a single weighted loss over three market observables simultaneously:

  L(kappa, theta, sigma, rho, v0) =
      w1 * MSE_SPX(implied_vols)          # SPX smile fit
    + w2 * MSE_VIX_futures(prices)        # VIX term structure fit
    + w3 * MSE_VIX_options(implied_vols)  # VIX options smile fit

This joint objective forces Heston to be consistent with:
  (a) The SPX skew and term structure
  (b) The level and slope of VIX futures
  (c) The vol-of-vol embedded in VIX options

Running all three legs together is what distinguishes a proper joint calibration
from a naïve SPX-only fit. The VIX legs prevent degenerate solutions where
kappa and sigma go to extremes to fit SPX at the expense of VIX.

Usage
-----
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
    cal = JointCalibrator(as_of_date='2026-03-24')
    result = cal.calibrate()
    print(result)

Requirements
------------
  - data_store/vol_system.db must contain options_snapshots + vix_term_structure
  - Run DataPipeline().daily_refresh() to ensure data is current
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

from joint_vol_calibration.config import (
    HESTON_BOUNDS, HESTON_DEFAULTS, JOINT_W1, JOINT_W2, JOINT_W3,
    RANDOM_SEED, DATA_DIR,
)
from joint_vol_calibration.data import database as db
from joint_vol_calibration.models.heston import (
    heston_call_batch,
    heston_vix_futures_curve,
    implied_vol_from_price,
    implied_vol_batch,
    black_scholes_call,
    bates_call_batch,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_VIX_WINDOW  = 30.0 / 365.0   # VIX 30-day integration window
_FELLER_PENALTY = 50.0         # large penalty keeps solver in Feller region
_MIN_T_DAYS  = 7               # minimum expiry to include (days)
_MAX_T_YEARS = 2.0             # maximum expiry to include (years)
_MONEYNESS_LO = 0.75           # K/S lower bound for filtered surface
_MONEYNESS_HI = 1.30           # K/S upper bound
_MIN_IV = 0.01                 # discard IVs below 1%
_MAX_IV = 2.00                 # discard IVs above 200%
_MIN_PRICE = 0.50              # discard options priced below $0.50

# Calibration subsampling: keep only the most informative anchor points
# (6 expiry buckets × 9 strikes = 54 options max — sufficient for 5-param Heston)
_CAL_EXPIRY_TARGETS_DAYS = [14, 30, 60, 91, 180, 365]  # target days
_CAL_STRIKES_PER_EXPIRY  = 9                             # strikes per expiry slice


# ── VIX term structure tenors (Yahoo Finance index → days to expiry) ──────────

_VIX_TS_TENORS = {
    "^VIX9D":  9.0  / 365.0,
    "^VIX":    30.0 / 365.0,
    "^VIX3M":  91.0 / 365.0,
    "^VIX6M":  182.0 / 365.0,
}


# ── Helper: VIX option pricing under Heston via CIR transition density ────────

def heston_vix_call_price(
    K_vix: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    r: float = 0.045,
    n_grid: int = 1000,
) -> float:
    """
    Price a VIX call option under Heston using numerical integration over the
    CIR transition density of the variance process.

    Under Heston, VIX^2(T) = theta + (v_T - theta) * f  where
      f = (1 - exp(-kappa*tau)) / (kappa*tau),  tau = 30/365.

    The conditional distribution of v_T given v_0 is:
      2*c*v_T | v_0 ~ non-central chi-squared(df=d, nc=lambda)
    where:
      c      = 2*kappa / (sigma^2 * (1 - exp(-kappa*T)))
      d      = 4*kappa*theta / sigma^2          (degrees of freedom)
      lambda = 2*c*v_0*exp(-kappa*T)            (non-centrality)

    VIX call = e^{-rT} * E[max(VIX(T) - K_vix, 0)]
             = e^{-rT} * integral max(100*sqrt(a + b*v) - K_vix, 0) * p(v) dv
    where a = theta*(1-f), b = f.

    Parameters
    ----------
    K_vix : VIX strike (e.g. 20.0 means VIX = 20)
    T     : time to expiry in years

    Returns
    -------
    float — VIX call price (in VIX points)
    """
    if T <= 0:
        vix_now = 100.0 * np.sqrt(max(theta, 1e-8))
        return max(vix_now - K_vix, 0.0)

    if sigma < 1e-8:
        # Deterministic limit: VIX(T) = sqrt(theta + (v0-theta)*exp(-kappa*T)*f + theta*(1-f))
        ev_T = theta + (v0 - theta) * np.exp(-kappa * T)
        f = (1.0 - np.exp(-kappa * _VIX_WINDOW)) / (kappa * _VIX_WINDOW + 1e-12)
        vix2 = theta * (1.0 - f) + ev_T * f
        vix_det = 100.0 * np.sqrt(max(vix2, 0.0))
        return max(vix_det - K_vix, 0.0) * np.exp(-r * T)

    # CIR parameters
    c      = 2.0 * kappa / (sigma**2 * (1.0 - np.exp(-kappa * T) + 1e-12))
    d      = 4.0 * kappa * theta / sigma**2
    lam    = 2.0 * c * v0 * np.exp(-kappa * T)

    # VIX^2(T) = a + b * v_T
    f = ((1.0 - np.exp(-kappa * _VIX_WINDOW))
         / (kappa * _VIX_WINDOW + 1e-12))
    a = theta * (1.0 - f)
    b = f

    # Integration grid over v_T: from 0 to v_max (99.9th percentile of dist)
    v_mean = (d + lam) / (2.0 * c)
    v_std  = np.sqrt((d + 2.0 * lam) / (2.0 * c**2))
    v_max  = v_mean + 6.0 * v_std
    v_grid = np.linspace(1e-8, max(v_max, 0.01), n_grid)

    # p(v_T) from non-central chi-squared: 2*c*v_T ~ ncx2(df=d, nc=lam)
    x_grid = 2.0 * c * v_grid
    pdf_x  = stats.ncx2.pdf(x_grid, df=d, nc=lam)
    pdf_v  = 2.0 * c * pdf_x                          # change of variables

    # VIX(T) at each grid point
    vix2_grid = a + b * v_grid
    vix_grid  = 100.0 * np.sqrt(np.maximum(vix2_grid, 0.0))

    # Payoff and integral
    payoff = np.maximum(vix_grid - K_vix, 0.0)
    price  = np.trapz(payoff * pdf_v, v_grid) * np.exp(-r * T)
    return float(max(price, 0.0))


def heston_vix_put_price(
    K_vix: float, T: float, kappa: float, theta: float,
    sigma: float, v0: float, r: float = 0.045,
) -> float:
    """VIX put via put-call parity: P = C - F + K*df, where F = model VIX forward."""
    call = heston_vix_call_price(K_vix, T, kappa, theta, sigma, v0, r)
    # Forward VIX: approx sqrt(E[VIX^2(T)])
    ev_T = theta + (v0 - theta) * np.exp(-kappa * T)
    f    = (1.0 - np.exp(-kappa * _VIX_WINDOW)) / (kappa * _VIX_WINDOW + 1e-12)
    fwd_vix = 100.0 * np.sqrt(max(theta * (1 - f) + ev_T * f, 0.0))
    return call - fwd_vix * np.exp(-r * T) + K_vix * np.exp(-r * T)


# ── Main Class ────────────────────────────────────────────────────────────────

class JointCalibrator:
    """
    Joint SPX/VIX smile calibration engine for the Heston model.

    Loads market data once at construction, then optimises a weighted
    three-leg loss function using differential evolution (global search)
    followed by L-BFGS-B (local polish).

    Attributes (set after calibrate())
    ------------------------------------
    params     : dict — calibrated Heston parameters
    result     : scipy OptimizeResult
    fit_time   : float — wall-clock seconds for calibration
    losses     : dict — per-leg loss at optimum
    """

    def __init__(
        self,
        as_of_date: str,
        r: float = 0.045,
        q: float = 0.013,
        w1: float = JOINT_W1,
        w2: float = JOINT_W2,
        w3: float = JOINT_W3,
    ):
        """
        Parameters
        ----------
        as_of_date : str 'YYYY-MM-DD' — calibration date
        r          : risk-free rate (continuously compounded)
        q          : SPX dividend yield
        w1, w2, w3 : loss weights (SPX ivol, VIX futures, VIX options)
        """
        self.as_of_date = as_of_date
        self.r = r
        self.q = q
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.params: Optional[dict] = None
        self.result = None
        self.fit_time: Optional[float] = None
        self.losses: Optional[dict] = None
        self._n_evals = 0

        # Load and prepare market data
        logger.info("JointCalibrator: loading market data for %s", as_of_date)
        self._load_market_data()

    # ── Market Data Preparation ────────────────────────────────────────────────

    def _load_market_data(self):
        """Load and clean all three data legs from the database."""
        # SPX spot
        spx_hist = db.get_spx_ohlcv(as_of_date=self.as_of_date)
        if spx_hist.empty:
            raise ValueError(f"No SPX OHLCV data available for {self.as_of_date}")
        self.S = float(spx_hist.iloc[-1]["close"])
        logger.info("  SPX spot S = %.2f (date: %s)",
                    self.S, spx_hist.iloc[-1]["date"])

        # SPX options surface
        self.spx_surface = self._prepare_spx_surface()
        logger.info("  SPX surface: %d liquid options across %d expiries",
                    len(self.spx_surface),
                    self.spx_surface["time_to_expiry"].nunique())

        # VIX term structure
        self.vix_ts = self._prepare_vix_term_structure()
        logger.info("  VIX term structure: %d tenors", len(self.vix_ts))

        # VIX options
        self.vix_options = self._prepare_vix_options()
        logger.info("  VIX options: %d liquid contracts", len(self.vix_options))

        # Adjust weights if VIX options are sparse
        if len(self.vix_options) < 5:
            logger.warning("  VIX options too sparse — setting w3=0")
            self.w3 = 0.0
            self.w1 = self.w1 / (self.w1 + self.w2)
            self.w2 = self.w2 / (self.w1 + self.w2) if (self.w1 + self.w2) > 0 else 0.0

    def _prepare_spx_surface(self) -> pd.DataFrame:
        """
        Filter and subsample SPX options to the ~54 most informative anchor points.

        Design:
          - 6 target expiry buckets: 14d, 30d, 60d, 91d, 180d, 365d
          - For each bucket: pick the closest available expiry
          - 9 strikes per expiry: evenly spaced across [moneyness_lo, hi]
            (closest observed strikes to log-uniform grid)
          - Prefer OTM options (calls for K≥S, puts for K<S)
          - Precompute BS vega for vega-normalized price error (no IV inversion in loop)

        This subsampling is the standard industry practice for Heston calibration.
        6×9=54 points is sufficient to constrain all 5 Heston parameters.
        The full surface is retained in self.spx_surface_full for validation.
        """
        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="SPX")
        if raw.empty:
            return pd.DataFrame()

        S = self.S
        df = raw.copy()

        # Standard quality filters
        df = df[df["time_to_expiry"] >= _MIN_T_DAYS / 365.0]
        df = df[df["time_to_expiry"] <= _MAX_T_YEARS]
        df = df[df["strike"].between(S * _MONEYNESS_LO, S * _MONEYNESS_HI)]
        df = df[df["mid_price"] > _MIN_PRICE]
        df = df[df["implied_vol"].between(_MIN_IV, _MAX_IV)]
        df = df.dropna(subset=["implied_vol"])

        # OTM convention: calls for K≥S, puts for K<S
        call_mask = (df["right"] == "C") & (df["strike"] >= S)
        put_mask  = (df["right"] == "P") & (df["strike"] <  S)
        df = df[call_mask | put_mask].copy()

        # Save full surface for validation
        self.spx_surface_full = df.sort_values(
            ["time_to_expiry", "strike"]).reset_index(drop=True)

        if df.empty:
            return df

        # ── Subsample to 6 expiry buckets × 9 strikes ──
        available_expiries = sorted(df["time_to_expiry"].unique())
        selected_rows = []

        for target_days in _CAL_EXPIRY_TARGETS_DAYS:
            target_T = target_days / 365.0
            # Find closest available expiry
            best_T = min(available_expiries,
                         key=lambda t: abs(t - target_T))
            if abs(best_T - target_T) > target_T * 0.5:
                continue  # skip if no expiry within 50% of target

            slice_df = df[df["time_to_expiry"] == best_T]
            if len(slice_df) < 2:
                continue

            # Select evenly-spaced strikes: log-uniform from lo to hi
            strikes_avail = sorted(slice_df["strike"].unique())
            k_lo = np.log(S * _MONEYNESS_LO)
            k_hi = np.log(S * _MONEYNESS_HI)
            target_ks = np.exp(np.linspace(k_lo, k_hi, _CAL_STRIKES_PER_EXPIRY))

            picked_strikes = set()
            for tk in target_ks:
                best_k = min(strikes_avail, key=lambda k: abs(k - tk))
                if best_k not in picked_strikes:
                    picked_strikes.add(best_k)

            for K in picked_strikes:
                row = slice_df[slice_df["strike"] == K]
                if not row.empty:
                    selected_rows.append(row.iloc[0])

        if not selected_rows:
            return pd.DataFrame()

        cal_df = pd.DataFrame(selected_rows).reset_index(drop=True)

        # ── Precompute BS vega (used to normalise price errors in _spx_leg) ──
        from scipy.stats import norm as _norm
        vegas = np.zeros(len(cal_df))
        for i, row in cal_df.iterrows():
            iv = float(row["implied_vol"])
            T  = float(row["time_to_expiry"])
            K  = float(row["strike"])
            if iv > 0 and T > 0:
                d1 = (np.log(S / K) + (self.r - self.q + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                v  = S * np.exp(-self.q * T) * _norm.pdf(d1) * np.sqrt(T)
                vegas[i] = max(v, 0.5)   # floor: prevent division by near-zero
            else:
                vegas[i] = 1.0

        cal_df = cal_df.copy()
        cal_df["bs_vega"]       = vegas
        cal_df["market_price_f"] = cal_df["mid_price"].astype(float)

        return cal_df.sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)

    def _prepare_vix_term_structure(self) -> pd.DataFrame:
        """
        Extract the most recent VIX term structure as market VIX futures proxies.

        Uses VIX9D (9d), VIX (30d), VIX3M (91d), VIX6M (182d) as the four term
        structure anchor points. These are the real-time VIX indices published by
        CBOE, which closely track the corresponding VIX futures prices.

        Returns
        -------
        DataFrame with columns [tenor_years, market_price]
        """
        ts_wide = db.get_vix_term_structure_wide(as_of_date=self.as_of_date)
        if ts_wide.empty:
            return pd.DataFrame(columns=["tenor_years", "market_price"])

        latest = ts_wide.iloc[-1]
        rows = []
        for col, tenor in _VIX_TS_TENORS.items():
            if col in latest and pd.notna(latest[col]) and latest[col] > 0:
                rows.append({"tenor_years": tenor, "market_price": float(latest[col])})

        if not rows:
            return pd.DataFrame(columns=["tenor_years", "market_price"])

        return pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)

    def _prepare_vix_options(self) -> pd.DataFrame:
        """
        Filter VIX options to liquid region: near-term, near-ATM strikes.

        VIX options: T ≤ 90 days, 70% ≤ K/VIX ≤ 150%, valid IV.
        """
        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="VIX")
        if raw.empty:
            return pd.DataFrame()

        # Get current VIX level from term structure
        ts = self._prepare_vix_term_structure()
        if ts.empty or len(ts) == 0:
            return pd.DataFrame()
        vix_now = float(ts[ts["tenor_years"] <= 31 / 365].iloc[-1]["market_price"]
                        if len(ts[ts["tenor_years"] <= 31 / 365]) > 0
                        else ts.iloc[0]["market_price"])

        df = raw.copy()
        df = df[df["time_to_expiry"] >= 7.0 / 365.0]
        df = df[df["time_to_expiry"] <= 91.0 / 365.0]
        df = df[df["strike"].between(vix_now * 0.70, vix_now * 1.50)]
        df = df[df["mid_price"] > 0.05]
        df = df[df["implied_vol"].between(_MIN_IV, _MAX_IV)]
        df = df.dropna(subset=["implied_vol"])

        # Subsample: use only 2 expiry buckets × 7 strikes (like SPX subsampling)
        # This keeps VIX options leg fast (14 options vs 100+)
        df = df.sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)
        available_expiries = sorted(df["time_to_expiry"].unique())
        target_vix_expiries = [30.0 / 365.0, 60.0 / 365.0]
        selected_rows = []
        for target_T in target_vix_expiries:
            if not available_expiries:
                continue
            best_T = min(available_expiries, key=lambda t: abs(t - target_T))
            if abs(best_T - target_T) > target_T * 0.7:
                continue
            slice_df = df[df["time_to_expiry"] == best_T]
            if len(slice_df) < 2:
                continue
            # 7 strikes evenly spaced in log-moneyness
            strikes_avail = sorted(slice_df["strike"].unique())
            k_lo = np.log(vix_now * 0.75)
            k_hi = np.log(vix_now * 1.40)
            target_ks = np.exp(np.linspace(k_lo, k_hi, 7))
            picked = set()
            for tk in target_ks:
                bk = min(strikes_avail, key=lambda k: abs(k - tk))
                if bk not in picked:
                    picked.add(bk)
            for K in picked:
                rows = slice_df[slice_df["strike"] == K]
                if not rows.empty:
                    selected_rows.append(rows.iloc[0])
        if not selected_rows:
            return pd.DataFrame()
        df = pd.DataFrame(selected_rows).reset_index(drop=True)

        # Precompute BS vega for vega-normalised price error (avoids IV inversion in loop)
        from scipy.stats import norm as _norm
        vegas = np.zeros(len(df))
        for i, row in df.iterrows():
            iv = float(row["implied_vol"])
            T  = float(row["time_to_expiry"])
            K  = float(row["strike"])
            if iv > 0 and T > 0:
                d1 = (np.log(vix_now / K) + (self.r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                v  = vix_now * _norm.pdf(d1) * np.sqrt(T)
                vegas[i] = max(v, 0.05)
            else:
                vegas[i] = 1.0
        df = df.copy()
        df["bs_vega"]        = vegas
        df["market_price_f"] = df["mid_price"].astype(float)
        return df

    # ── Loss Function ──────────────────────────────────────────────────────────

    def joint_loss(self, params_vec: np.ndarray) -> float:
        """
        Evaluate the weighted joint loss at parameter vector [kappa, theta, sigma, rho, v0].

        Returns a large value (1e6) if parameters violate hard bounds.
        Applies a soft Feller penalty to keep 2*kappa*theta > sigma^2.
        """
        self._n_evals += 1
        kappa, theta, sigma, rho, v0 = params_vec

        # Hard bounds check (DE should respect bounds, but polish may stray)
        bnds = HESTON_BOUNDS
        if not (bnds["kappa"][0] <= kappa <= bnds["kappa"][1]):  return 1e6
        if not (bnds["theta"][0] <= theta <= bnds["theta"][1]):  return 1e6
        if not (bnds["sigma"][0] <= sigma <= bnds["sigma"][1]):  return 1e6
        if not (bnds["rho"][0]   <= rho   <= bnds["rho"][1]):    return 1e6
        if not (bnds["v0"][0]    <= v0    <= bnds["v0"][1]):     return 1e6

        # Feller condition: 2*kappa*theta >= sigma^2 ensures variance stays positive
        feller_viol = max(0.0, sigma**2 - 2.0 * kappa * theta)
        feller_penalty = _FELLER_PENALTY * feller_viol**2

        l1 = self._spx_leg(kappa, theta, sigma, rho, v0)
        l2 = self._vix_futures_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0

        return self.w1 * l1 + self.w2 * l2 + self.w3 * l3 + feller_penalty

    def _spx_leg(
        self, kappa: float, theta: float, sigma: float, rho: float, v0: float
    ) -> float:
        """
        SPX leg: vega-normalised MSE between Heston model prices and market prices.

        Uses the approximation:
          (model_iv - market_iv) ≈ (model_price - market_price) / BS_vega
        which avoids per-option IV inversion in the calibration hot loop.
        BS_vega is precomputed once at market IVs in _prepare_spx_surface().

        Groups by expiry and batch-prices all strikes in one vectorised call.
        Each error is divided by BS_vega^2 so the objective is in vol² units,
        matching the IV-MSE interpretation.
        """
        if self.spx_surface.empty:
            return 0.0

        df = self.spx_surface
        sse = 0.0
        n   = 0

        for T_val, group in df.groupby("time_to_expiry"):
            T        = float(T_val)
            strikes  = group["strike"].values.astype(float)
            rights   = group["right"].values
            mkt_px   = group["market_price_f"].values.astype(float)
            bs_vegas = group["bs_vega"].values.astype(float)

            # Batch price all calls for this expiry
            model_calls = heston_call_batch(
                self.S, strikes, T, self.r, self.q,
                kappa, theta, sigma, rho, v0
            )

            # Convert to correct right via put-call parity (vectorised)
            put_mask    = (rights == "P")
            model_prices = model_calls.copy()
            model_prices[put_mask] = (
                model_calls[put_mask]
                - self.S * np.exp(-self.q * T)
                + strikes[put_mask] * np.exp(-self.r * T)
            )

            # Vega-normalised squared error ≈ (Δ IV)^2
            err = (model_prices - mkt_px) / bs_vegas
            sse += np.sum(err**2)
            n   += len(err)

        return sse / max(n, 1)

    def _vix_futures_leg(
        self, kappa: float, theta: float, sigma: float, v0: float
    ) -> float:
        """
        VIX term structure leg: MSE between Heston-implied VIX levels and
        the VIX term structure indices (^VIX9D, ^VIX, ^VIX3M, ^VIX6M).

        Loss is in VIX-point space (not vol-decimal), scaled by 1/100^2.
        A 1 VIX point error contributes 0.01^2 = 1e-4 to this leg.
        """
        if self.vix_ts.empty:
            return 0.0

        tenors = self.vix_ts["tenor_years"].values.astype(float)
        market = self.vix_ts["market_price"].values.astype(float)
        model  = heston_vix_futures_curve(kappa, theta, sigma, v0, tenors)

        # Normalise to vol-decimal space (divide by 100)
        errors = (model - market) / 100.0
        return float(np.mean(errors**2))

    def _vix_options_leg(
        self, kappa: float, theta: float, sigma: float, v0: float
    ) -> float:
        """
        VIX options leg: vega-normalised MSE between Heston VIX option prices
        and market prices. No IV inversion — uses precomputed BS vegas.

        Prices VIX options using the CIR transition density (semi-analytic).
        """
        if self.vix_options.empty:
            return 0.0

        df = self.vix_options
        sse = 0.0
        n   = 0

        for _, row in df.iterrows():
            K        = float(row["strike"])
            T        = float(row["time_to_expiry"])
            mkt_px   = float(row["market_price_f"])
            bs_vega  = float(row["bs_vega"])
            right    = str(row["right"])

            if right == "C":
                model_px = heston_vix_call_price(K, T, kappa, theta, sigma, v0, self.r)
            else:
                model_px = heston_vix_put_price(K, T, kappa, theta, sigma, v0, self.r)

            err = (model_px - mkt_px) / bs_vega
            sse += err**2
            n   += 1

        return sse / max(n, 1)

    # ── Calibration ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        de_maxiter: int = 200,
        de_popsize: int = 10,
        polish: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Calibrate Heston parameters by minimising joint_loss.

        Step 1: Differential Evolution (global search, avoids local minima)
        Step 2: L-BFGS-B polish on the DE result (fast local refinement)

        Parameters
        ----------
        de_maxiter : int — DE max iterations (200 gives good coverage in <30s)
        de_popsize : int — DE population size per dimension (10 × 5 params = 50 members)
        polish     : bool — run L-BFGS-B after DE
        verbose    : bool — print progress

        Returns
        -------
        dict with keys: params, loss, leg_losses, fit_time, n_evals, success
        """
        logger.info("Starting joint calibration for %s", self.as_of_date)
        logger.info("  SPX surface: %d options", len(self.spx_surface))
        logger.info("  VIX tenors:  %d", len(self.vix_ts))
        logger.info("  VIX options: %d", len(self.vix_options))
        logger.info("  Weights: w1=%.2f, w2=%.2f, w3=%.2f", self.w1, self.w2, self.w3)

        bounds = [
            HESTON_BOUNDS["kappa"],
            HESTON_BOUNDS["theta"],
            HESTON_BOUNDS["sigma"],
            HESTON_BOUNDS["rho"],
            HESTON_BOUNDS["v0"],
        ]

        self._n_evals = 0
        t0 = time.time()

        # ── Step 1: Differential Evolution ──
        if verbose:
            print(f"\n[C4] Joint calibration: {self.as_of_date}  |  "
                  f"S={self.S:.0f}  |  "
                  f"{len(self.spx_surface)} SPX opts  |  "
                  f"{len(self.vix_ts)} VIX tenors")
            print(f"     DE: maxiter={de_maxiter}, popsize={de_popsize} × 5 = "
                  f"{de_popsize*5} members...")

        de_result = optimize.differential_evolution(
            self.joint_loss,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-6,
            seed=RANDOM_SEED,
            mutation=(0.5, 1.5),
            recombination=0.9,
            init="latinhypercube",
            workers=1,              # single-threaded (CF is not thread-safe)
            callback=None,
        )

        best_params = de_result.x
        best_loss   = de_result.fun

        if verbose:
            print(f"     DE done: loss={best_loss:.6f}, evals={self._n_evals}")

        # ── Step 2: L-BFGS-B Polish ──
        if polish:
            if verbose:
                print("     Polishing with L-BFGS-B...")
            polish_result = optimize.minimize(
                self.joint_loss,
                best_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
            )
            if polish_result.fun < best_loss:
                best_params = polish_result.x
                best_loss   = polish_result.fun
            if verbose:
                print(f"     Polish done: loss={best_loss:.6f}, evals={self._n_evals}")

        self.fit_time = time.time() - t0

        # ── Store Results ──
        kappa, theta, sigma, rho, v0 = best_params
        self.params = {
            "kappa": float(kappa),
            "theta": float(theta),
            "sigma": float(sigma),
            "rho":   float(rho),
            "v0":    float(v0),
        }

        # Per-leg losses at optimum
        l1 = self._spx_leg(kappa, theta, sigma, rho, v0)
        l2 = self._vix_futures_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        self.losses = {
            "spx_iv_rmse":    float(np.sqrt(l1)) * 100,   # in vol points (pct)
            "vix_futures_rmse": float(np.sqrt(l2)) * 100, # in vol points (pct)
            "vix_options_rmse": float(np.sqrt(l3)) * 100,
            "total_loss":     float(best_loss),
        }

        result = {
            "params":     self.params,
            "loss":       best_loss,
            "leg_losses": self.losses,
            "fit_time":   self.fit_time,
            "n_evals":    self._n_evals,
            "success":    bool(de_result.success),
            "feller_ok":  bool(2 * kappa * theta >= sigma**2),
        }
        self.result = result

        if verbose:
            self._print_result(result)

        return result

    # ── Bates (1996) SVJ Calibration ──────────────────────────────────────────

    def _bates_joint_loss(self, params_vec: np.ndarray) -> float:
        """
        Joint loss for 8-parameter Bates SVJ model.

        Params: [kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j]
        """
        kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = params_vec

        # Hard bounds
        bnds = HESTON_BOUNDS
        if not (bnds["kappa"][0] <= kappa <= bnds["kappa"][1]):  return 1e6
        if not (bnds["theta"][0] <= theta <= bnds["theta"][1]):  return 1e6
        if not (bnds["sigma"][0] <= sigma <= bnds["sigma"][1]):  return 1e6
        if not (bnds["rho"][0]   <= rho   <= bnds["rho"][1]):    return 1e6
        if not (bnds["v0"][0]    <= v0    <= bnds["v0"][1]):     return 1e6
        if lam < 0 or lam > 10:      return 1e6
        if mu_j < -0.15 or mu_j > 0: return 1e6
        if sigma_j < 0.01 or sigma_j > 0.15: return 1e6

        # Feller condition penalty
        feller_viol    = max(0.0, sigma**2 - 2.0 * kappa * theta)
        feller_penalty = _FELLER_PENALTY * feller_viol**2

        l1 = self._bates_spx_leg(kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
        l2 = self._vix_futures_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0

        return self.w1 * l1 + self.w2 * l2 + self.w3 * l3 + feller_penalty

    def _bates_spx_leg(
        self, kappa: float, theta: float, sigma: float, rho: float, v0: float,
        lam: float, mu_j: float, sigma_j: float,
    ) -> float:
        """SPX leg using Bates batch pricer instead of Heston."""
        if self.spx_surface.empty:
            return 0.0

        df  = self.spx_surface
        sse = 0.0
        n   = 0

        for T_val, group in df.groupby("time_to_expiry"):
            T        = float(T_val)
            strikes  = group["strike"].values.astype(float)
            rights   = group["right"].values
            mkt_px   = group["market_price_f"].values.astype(float)
            bs_vegas = group["bs_vega"].values.astype(float)

            model_calls = bates_call_batch(
                self.S, strikes, T, self.r, self.q,
                kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j
            )

            put_mask     = (rights == "P")
            model_prices = model_calls.copy()
            model_prices[put_mask] = (
                model_calls[put_mask]
                - self.S * np.exp(-self.q * T)
                + strikes[put_mask] * np.exp(-self.r * T)
            )

            err  = (model_prices - mkt_px) / bs_vegas
            sse += np.sum(err**2)
            n   += len(err)

        return sse / max(n, 1)

    def calibrate_bates(
        self,
        de_maxiter: int = 200,
        de_popsize: int = 10,
        polish: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Calibrate Bates (1996) SVJ parameters by minimising the joint loss.

        Eight parameters:
          kappa, theta, sigma, rho, v0   — Heston diffusion (same as calibrate())
          lam                            — jump intensity (jumps/year) ∈ [0, 10]
          mu_j                           — mean log jump size ∈ [−0.15, 0]
          sigma_j                        — jump size std ∈ [0.01, 0.15]

        Steps: Differential Evolution (global) → L-BFGS-B polish.

        Returns
        -------
        dict with keys: params, loss, leg_losses, fit_time, n_evals, success,
                        heston_comparison (if Heston already calibrated)
        """
        logger.info("Starting Bates calibration for %s", self.as_of_date)

        bounds = [
            HESTON_BOUNDS["kappa"],
            HESTON_BOUNDS["theta"],
            HESTON_BOUNDS["sigma"],
            HESTON_BOUNDS["rho"],
            HESTON_BOUNDS["v0"],
            (0.0, 10.0),      # lam
            (-0.15, 0.0),     # mu_j
            (0.01, 0.15),     # sigma_j
        ]

        self._n_evals = 0
        t0 = time.time()

        if verbose:
            print(f"\n[C4-Bates] Bates SVJ calibration: {self.as_of_date}  |  "
                  f"S={self.S:.0f}  |  {len(self.spx_surface)} SPX opts")
            print(f"     DE: maxiter={de_maxiter}, popsize={de_popsize} × 8 = "
                  f"{de_popsize*8} members...")

        de_result = optimize.differential_evolution(
            self._bates_joint_loss,
            bounds=bounds,
            maxiter=de_maxiter,
            popsize=de_popsize,
            tol=1e-6,
            seed=RANDOM_SEED,
            mutation=(0.5, 1.5),
            recombination=0.9,
            init="latinhypercube",
            workers=1,
        )

        best_params = de_result.x
        best_loss   = de_result.fun

        if verbose:
            print(f"     DE done: loss={best_loss:.6f}, evals={self._n_evals}")

        if polish:
            if verbose:
                print("     Polishing with L-BFGS-B...")
            polish_result = optimize.minimize(
                self._bates_joint_loss,
                best_params,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7},
            )
            if polish_result.fun < best_loss:
                best_params = polish_result.x
                best_loss   = polish_result.fun
            if verbose:
                print(f"     Polish done: loss={best_loss:.6f}, evals={self._n_evals}")

        fit_time = time.time() - t0

        kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = best_params
        bates_params = {
            "kappa": float(kappa), "theta": float(theta),
            "sigma": float(sigma), "rho":   float(rho),
            "v0":    float(v0),    "lam":   float(lam),
            "mu_j":  float(mu_j),  "sigma_j": float(sigma_j),
        }

        l1 = self._bates_spx_leg(kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
        l2 = self._vix_futures_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        bates_losses = {
            "spx_iv_rmse":      float(np.sqrt(l1)) * 100,
            "vix_futures_rmse": float(np.sqrt(l2)) * 100,
            "vix_options_rmse": float(np.sqrt(l3)) * 100,
            "total_loss":       float(best_loss),
        }

        result = {
            "params":     bates_params,
            "loss":       best_loss,
            "leg_losses": bates_losses,
            "fit_time":   fit_time,
            "n_evals":    self._n_evals,
            "success":    bool(de_result.success),
            "feller_ok":  bool(2 * kappa * theta >= sigma**2),
        }

        # Attach Heston comparison if available
        if self.params is not None:
            result["heston_comparison"] = {
                "heston_spx_rmse": self.losses["spx_iv_rmse"],
                "bates_spx_rmse":  bates_losses["spx_iv_rmse"],
                "heston_vix_rmse": self.losses["vix_futures_rmse"],
                "bates_vix_rmse":  bates_losses["vix_futures_rmse"],
                "heston_rho":      self.params["rho"],
                "bates_rho":       float(rho),
            }

        if verbose:
            self._print_bates_result(result)

        return result

    def _print_bates_result(self, result: dict):
        p  = result["params"]
        ll = result["leg_losses"]
        print(f"\n── Calibrated Bates SVJ Parameters ({self.as_of_date}) ──")
        print(f"  kappa   = {p['kappa']:.4f}")
        print(f"  theta   = {p['theta']:.4f}  (vol={np.sqrt(p['theta'])*100:.1f}%)")
        print(f"  sigma   = {p['sigma']:.4f}")
        print(f"  rho     = {p['rho']:.4f}")
        print(f"  v0      = {p['v0']:.4f}  (vol={np.sqrt(p['v0'])*100:.1f}%)")
        print(f"  lam     = {p['lam']:.3f} jumps/yr")
        print(f"  mu_j    = {p['mu_j']*100:.2f}%")
        print(f"  sigma_j = {p['sigma_j']*100:.2f}%")
        print(f"  Feller  : 2κθ={2*p['kappa']*p['theta']:.4f} {'≥' if result['feller_ok'] else '<'} σ²={p['sigma']**2:.4f}")
        print(f"\n── Bates Loss Breakdown ─────────────────────────────")
        print(f"  SPX smile RMSE    : {ll['spx_iv_rmse']:.3f} vol pts")
        print(f"  VIX futures RMSE  : {ll['vix_futures_rmse']:.3f} vol pts")
        print(f"  VIX options RMSE  : {ll['vix_options_rmse']:.3f} vol pts")
        print(f"  Fit time          : {result['fit_time']:.1f}s")
        if "heston_comparison" in result:
            hc = result["heston_comparison"]
            print(f"\n── Bates vs Heston ──────────────────────────────────")
            print(f"  SPX RMSE: Heston {hc['heston_spx_rmse']:.3f} → Bates {hc['bates_spx_rmse']:.3f} vol pts")
            print(f"  VIX RMSE: Heston {hc['heston_vix_rmse']:.3f} → Bates {hc['bates_vix_rmse']:.3f} vol pts")
            print(f"  rho:      Heston {hc['heston_rho']:.4f} → Bates {hc['bates_rho']:.4f}")

    # ── Validation ────────────────────────────────────────────────────────────

    def smile_reconstruction_error(self) -> pd.DataFrame:
        """
        Compute model vs market implied vol at every calibration point.

        Returns a DataFrame with columns:
          [expiry, strike, moneyness, right, market_iv, model_iv, error, abs_error]

        This is the primary quality check: the residuals should be small and
        roughly symmetric around zero (no systematic bias in any region).
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() before validate()")

        p = self.params
        rows = []
        df = self.spx_surface

        for T, group in df.groupby("time_to_expiry"):
            strikes = group["strike"].values.astype(float)
            rights  = group["right"].values
            mkt_ivs = group["implied_vol"].values.astype(float)

            model_calls = heston_call_batch(
                self.S, strikes, float(T), self.r, self.q,
                p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"]
            )

            for j, (K, right, mkt_iv) in enumerate(zip(strikes, rights, mkt_ivs)):
                if right == "P":
                    mp = (model_calls[j] - self.S * np.exp(-self.q * T)
                          + K * np.exp(-self.r * T))
                else:
                    mp = model_calls[j]
                model_iv = implied_vol_from_price(mp, self.S, K, T, self.r, self.q, right)

                rows.append({
                    "expiry":     round(float(T) * 365),
                    "strike":     float(K),
                    "moneyness":  float(K) / self.S,
                    "right":      right,
                    "market_iv":  mkt_iv,
                    "model_iv":   model_iv if model_iv is not None else np.nan,
                })

        res = pd.DataFrame(rows)
        res["error"]     = res["model_iv"] - res["market_iv"]
        res["abs_error"] = res["error"].abs()
        return res

    def validate(self) -> dict:
        """
        Full calibration quality report.

        Returns
        -------
        dict with smile reconstruction metrics and VIX curve comparison.
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() first")

        smile_df = self.smile_reconstruction_error()

        # Smile metrics
        rmse  = float(np.sqrt(np.nanmean(smile_df["error"]**2))) * 100
        mae   = float(np.nanmean(smile_df["abs_error"])) * 100
        max_e = float(np.nanmax(smile_df["abs_error"])) * 100
        bias  = float(np.nanmean(smile_df["error"])) * 100

        # VIX curve comparison
        p = self.params
        vix_model = heston_vix_futures_curve(
            p["kappa"], p["theta"], p["sigma"], p["v0"],
            self.vix_ts["tenor_years"].values if not self.vix_ts.empty else np.array([])
        )
        vix_market = self.vix_ts["market_price"].values if not self.vix_ts.empty else np.array([])
        vix_rmse = float(np.sqrt(np.mean((vix_model - vix_market)**2))) if len(vix_market) else np.nan

        report = {
            "smile_rmse_vol_pts":   rmse,
            "smile_mae_vol_pts":    mae,
            "smile_max_err_vol_pts": max_e,
            "smile_bias_vol_pts":   bias,
            "vix_curve_rmse_pts":   vix_rmse,
            "n_spx_options":        len(smile_df),
            "feller_condition_ok":  2 * p["kappa"] * p["theta"] >= p["sigma"]**2,
        }

        print("\n── Calibration Quality Report ──────────────────────")
        print(f"  SPX smile RMSE : {rmse:.2f} vol pts")
        print(f"  SPX smile MAE  : {mae:.2f} vol pts")
        print(f"  SPX smile max  : {max_e:.2f} vol pts")
        print(f"  SPX smile bias : {bias:+.2f} vol pts")
        print(f"  VIX curve RMSE : {vix_rmse:.2f} VIX pts")
        print(f"  Feller OK      : {report['feller_condition_ok']}")
        print("─" * 50)

        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Save calibrated parameters and metadata to disk.

        Default path: data_store/calibrations/joint_cal_{as_of_date}.pkl
        """
        if self.params is None:
            raise RuntimeError("Nothing to save — run calibrate() first")

        if path is None:
            cal_dir = DATA_DIR / "calibrations"
            cal_dir.mkdir(parents=True, exist_ok=True)
            path = str(cal_dir / f"joint_cal_{self.as_of_date}.pkl")

        payload = {
            "as_of_date":  self.as_of_date,
            "params":      self.params,
            "losses":      self.losses,
            "fit_time":    self.fit_time,
            "n_evals":     self._n_evals,
            "r":           self.r,
            "q":           self.q,
            "S":           self.S,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Calibration saved to %s", path)
        return path

    @classmethod
    def load_params(cls, path: str) -> dict:
        """Load calibrated params dict from a saved pickle."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload

    # ── Display ───────────────────────────────────────────────────────────────

    def _print_result(self, result: dict):
        p = result["params"]
        ll = result["leg_losses"]
        print(f"\n── Calibrated Heston Parameters ({self.as_of_date}) ──")
        print(f"  kappa = {p['kappa']:.4f}  (mean-reversion speed)")
        print(f"  theta = {p['theta']:.4f}  (long-run var, vol={np.sqrt(p['theta'])*100:.1f}%)")
        print(f"  sigma = {p['sigma']:.4f}  (vol-of-vol)")
        print(f"  rho   = {p['rho']:.4f}  (spot-vol correlation)")
        print(f"  v0    = {p['v0']:.4f}  (spot var,  vol={np.sqrt(p['v0'])*100:.1f}%)")
        print(f"  Feller: 2κθ={2*p['kappa']*p['theta']:.4f} {'≥' if result['feller_ok'] else '<'} σ²={p['sigma']**2:.4f}")
        print(f"\n── Loss Breakdown ───────────────────────────────────")
        print(f"  SPX smile RMSE    : {ll['spx_iv_rmse']:.2f} vol pts")
        print(f"  VIX futures RMSE  : {ll['vix_futures_rmse']:.2f} vol pts")
        print(f"  VIX options RMSE  : {ll['vix_options_rmse']:.2f} vol pts")
        print(f"  Total loss        : {ll['total_loss']:.6f}")
        print(f"  Fit time          : {result['fit_time']:.1f}s  ({result['n_evals']} evals)")

    def __repr__(self) -> str:
        if self.params is None:
            return f"JointCalibrator(as_of={self.as_of_date}, not calibrated)"
        p = self.params
        return (f"JointCalibrator(as_of={self.as_of_date}, "
                f"kappa={p['kappa']:.3f}, theta={p['theta']:.4f}, "
                f"sigma={p['sigma']:.3f}, rho={p['rho']:.3f}, v0={p['v0']:.4f})")
