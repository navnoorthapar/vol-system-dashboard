"""
risk_monitor.py — Second-Order Greeks Risk Monitor (C6).

Computes the second-order Greeks surface across strike / maturity for one
Heston calibration, the exact quadratic-variation convexity of the CIR
variance process, and a per-node vomma-instability measure.

The 'sigma' that is bumped is the Black-Scholes implied volatility at each
(K, T) node, backed out from the Heston price. All Greek formulas are
therefore standard BS market Greeks evaluated at the model-implied vol.

Pricing and IV inversion
------------------------
  Heston prices come from the Lewis (2001) Fourier formula evaluated on an
  adaptive trapezoid grid (implied_vol.fourier_otm_price with the Heston
  characteristic function), accurate to ~1e-14 of the forward. The solver
  inverts the OUT-OF-THE-MONEY price (puts below the forward, calls above) with
  a bracketed, safeguarded Newton (implied_vol.implied_vol_black).

  A node whose OTM price is below `min_otm_price_rel × F` (default 5e-6 × F,
  about $0.03 at F = 6,600: below one $0.05 SPX tick) is not a meaningful
  risk node — its implied vol is set by a price nobody can trade — and is
  dropped. Dropping is never silent: every dropped node and the reason is
  recorded in the surface provenance (DataFrame.attrs["provenance"], also
  written into the parquet metadata) and logged. Any node that is above the
  floor but fails to invert raises, because that can only be a bug.

Greeks (units: per option on one unit of the index; ×100 per SPX contract)
-------------------------------------------------------------------------
  delta  : call delta e^{−qT}N(d1)                       [dimensionless]
  vega   : ∂V/∂σ = S e^{−qT} φ(d1) √T                     [index pts per 1.00 of σ;
                                                           ÷100 → per vol point]
  vomma  : ∂²V/∂σ² = Vega·d1·d2/σ                         [index pts per (1.00 of σ)²;
                                                           ÷1e4 → per vol-point²]
           "Volga" is another name for the same Greek; this module uses
           "vomma" only. validate() cross-checks the analytic value against a
           central finite difference of the BS price (not published).
  vanna  : ∂²V/∂S∂σ = −e^{−qT} φ(d1) d2/σ                 [delta per 1.00 of σ;
                                                           ÷100 → per vol point]
  qv_convexity : Var[∫₀ᵀ v_t dt] of the CIR variance, exact closed form
                 (a model quantity: it does not depend on any option's vomma)

Sign convention
---------------
  Vomma > 0 in both wings (d1·d2 > 0, i.e. |ln(K/F)| > σ²T/2)
  Vomma < 0 near the money (d1 > 0 > d2)
  Vanna > 0 for OTM calls (d2 < 0), < 0 for ITM calls (d2 > 0)

Vomma-instability measure (per node, absolute — not a cross-sectional z-score)
------------------------------------------------------------------------------
  For an implied-vol move Δσ, a vega-hedged position is left with the
  second-order P&L ½·vomma·Δσ². Relative to the first-order vega P&L this is

        vomma_pnl_ratio = ½·vomma·Δσ² / (vega·|Δσ|) = ½·d1·d2·|Δσ| / σ_IV

  Δσ is the model-implied move of that maturity's ATM implied vol over a
  hedge horizon h (default one trading day) at n standard deviations
  (default 2), from the Heston variance dynamics:

        Δσ(T) = n·σ_v·√(v0·h) · B(T) / (2·σ̄(T)),
        B(T) = (1 − e^{−κT})/(κT),   σ̄(T)² = θ + (v0 − θ)·B(T)

  (σ̄² is the expected average variance to T, the leading-order ATM implied
  variance; its sensitivity to v0 is B(T), and dv0 has standard deviation
  σ_v√(v0·h)). A node is flagged unstable when |vomma_pnl_ratio| exceeds the
  threshold (default 0.25: the convexity term is more than a quarter of the
  vega P&L, so a vega hedge sized at today's IV mis-hedges materially).
  The measure is zero when the model has no vol-of-vol (σ_v = 0), scales
  with the size of the vol shock, and needs no other nodes. The old pooled
  z-score is kept as the descriptive column `vomma_zscore` only.

Outputs
-------
  data_store/greeks/greeks_surface.parquet   — surface + provenance metadata
  data_store/greeks/vomma_heatmap.png        — heatmap with ×-marks

Usage
-----
  from joint_vol_calibration.greeks.risk_monitor import RiskMonitor

  monitor = RiskMonitor.from_calibration("data_store/calibrations/joint_cal_2026-09-11.pkl")
  surface = monitor.build()
  flagged = monitor.flag()
  path    = monitor.save()
  checks  = monitor.validate()

  # or regenerate the published surface from a calibration pickle:
  python -m joint_vol_calibration.greeks.risk_monitor \\
      --calibration data_store/calibrations/joint_cal_2026-09-11.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR, HESTON_BOUNDS, RANDOM_SEED, ROOT_DIR
from joint_vol_calibration.models.heston import characteristic_function
from joint_vol_calibration.models.implied_vol import (
    IV_BELOW_FLOOR,
    IV_NO_TIME_VALUE,
    IV_OK,
    IV_STATUS,
    bs_d1d2,
    bs_delta,
    bs_price,
    bs_vanna,
    bs_vega,
    bs_vomma,
    fourier_otm_price,
    implied_vol_black,
)

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

GREEKS_DIR          = DATA_DIR / "greeks"
GREEKS_SURFACE_PATH = GREEKS_DIR / "greeks_surface.parquet"
HEATMAP_PATH        = GREEKS_DIR / "vomma_heatmap.png"

# ── Default grid ───────────────────────────────────────────────────────────────

DEFAULT_MATURITIES_DAYS = np.array([14, 30, 60, 91, 180, 365])
DEFAULT_LOG_MONEYNESS   = np.linspace(-0.30, 0.25, 15)   # ln(K/F), 15 strikes per expiry

# Node floor: OTM option value / forward. 5e-6 × F ≈ $0.033 at F = 6,600,
# below the $0.05 SPX minimum tick; ~1e8 above the pricer's error.
MIN_OTM_PRICE_REL = 5e-6

# Vomma-instability measure defaults (see module docstring)
IV_SHOCK_HORIZON_YEARS = 1.0 / 252.0   # one trading day (daily re-hedge)
IV_SHOCK_N_SD          = 2.0           # 2-standard-deviation move
UNSTABLE_RATIO_THRESHOLD = 0.25        # |½·vomma·Δσ²| > 25% of |vega·Δσ|

# Finite-difference bump for the (unpublished) vomma cross-check, 10 bp of vol
_VOMMA_FD_BUMP = 0.001
_VOLGA_BUMP = _VOMMA_FD_BUMP           # backward-compatible name

SURFACE_SCHEMA_VERSION = 2
_PROVENANCE_KEY = b"greeks_provenance"

GREEKS_UNITS = {
    "K": "index points",
    "T": "years",
    "T_days": "calendar days",
    "log_moneyness": "ln(K/F), F = S·exp((r−q)T)",
    "strike_index": "position of the strike in the per-maturity strike grid",
    "iv": "Black-Scholes implied vol, decimal (0.20 = 20%)",
    "otm_price": "discounted OTM option value (put if K<F, call if K>=F), index points",
    "delta": "call delta, dimensionless",
    "vega": "index points per 1.00 of implied vol (÷100 → per vol point)",
    "vomma": "index points per (1.00 of implied vol)^2 (÷1e4 → per vol-point^2); a.k.a. volga",
    "vanna": "change in call delta per 1.00 of implied vol (÷100 → per vol point)",
    "qv_convexity": "Var[∫_0^T v_t dt] of the CIR variance, (variance·years)^2, exact",
    "iv_shock": "model-implied ATM implied-vol move used for the instability measure, decimal",
    "vomma_pnl_ratio": "½·vomma·Δσ² / (vega·|Δσ|), dimensionless",
    "vomma_zscore": "pooled cross-sectional z-score of vomma (descriptive only)",
    "is_unstable": "|vomma_pnl_ratio| > threshold",
    "scale": "per option on one unit of the index; ×100 for one SPX contract",
}


# ── Analytical BS Greeks (vectorised) — thin aliases of models.implied_vol ────
# Kept under their historical private names: delta_hedger imports _bs_vomma.

def _bs_d1d2(F, K, T, sigma) -> tuple:
    """d1, d2 in the forward-measure Black-Scholes formula."""
    return bs_d1d2(F, K, T, sigma)


def _bs_call_price(S, K, T, r, q, sigma) -> np.ndarray:
    """Vectorised Black-Scholes call price with dividend yield q."""
    return bs_price(S, K, T, r, q, sigma, True)


def _bs_delta(S, K, T, r, q, sigma) -> np.ndarray:
    """∂C/∂S = exp(−qT)·N(d1)  (always in [0, exp(-qT)])."""
    return bs_delta(S, K, T, r, q, sigma, True)


def _bs_vega(S, K, T, r, q, sigma) -> np.ndarray:
    """∂C/∂σ = S·exp(−qT)·N′(d1)·√T  (always positive)."""
    return bs_vega(S, K, T, r, q, sigma)


def _bs_vomma(S, K, T, r, q, sigma) -> np.ndarray:
    """Analytical vomma (= volga) = ∂²C/∂σ² = Vega · d1 · d2 / σ."""
    return bs_vomma(S, K, T, r, q, sigma)


def _bs_vanna(S, K, T, r, q, sigma) -> np.ndarray:
    """∂²C/∂S∂σ = ∂Vega/∂S = ∂Delta/∂σ = −exp(−qT)·N′(d1)·d2 / σ."""
    return bs_vanna(S, K, T, r, q, sigma)


def _bs_vomma_numerical(S, K, T, r, q, sigma, bump: float = _VOMMA_FD_BUMP) -> np.ndarray:
    """
    Vomma by central finite difference of the BS call price:
    (V(σ+δ) − 2V(σ) + V(σ−δ)) / δ². A numerical cross-check of the analytic
    vomma (same Greek), agreeing to O(δ²); not a second risk measure.
    """
    sig_up = np.clip(sigma + bump, 1e-4, 5.0)
    sig_dn = np.clip(sigma - bump, 1e-4, 5.0)
    V_up = _bs_call_price(S, K, T, r, q, sig_up)
    V_0  = _bs_call_price(S, K, T, r, q, sigma)
    V_dn = _bs_call_price(S, K, T, r, q, sig_dn)
    return (V_up - 2.0 * V_0 + V_dn) / bump**2


# Backward-compatible name: "volga" is the same Greek as vomma.
_bs_volga_numerical = _bs_vomma_numerical


# ── Heston OTM prices (Lewis / Fourier) ───────────────────────────────────────

def heston_otm_prices(
    S: float,
    K: np.ndarray,
    T: float,
    r: float,
    q: float,
    params: dict,
) -> tuple:
    """
    Discounted Heston OUT-OF-THE-MONEY option values for one maturity.

    Returns (otm_price, F, ok): put values for K < F, call values for K >= F,
    the forward, and whether the Fourier truncation converged. Accuracy is
    ~1e-14·F absolute (see implied_vol.fourier_otm_price), independent of
    heston_call_batch's fixed integration grid.
    """
    K = np.asarray(K, dtype=float)
    F = S * np.exp((r - q) * T)
    p = {k: float(params[k]) for k in ("kappa", "theta", "sigma", "rho", "v0")}

    def cf(u):   # CF of ln(S_T/F): unit spot, zero carry
        return characteristic_function(u, 1.0, T, 0.0, 0.0, **p)

    # Expected total variance to T: the control-variate variance for the pricer
    kT = p["kappa"] * T
    B = -np.expm1(-kT) / kT if kT > 1e-10 else 1.0
    w = max((p["theta"] + (p["v0"] - p["theta"]) * B) * T, 1e-12)
    otm_fwd, ok = fourier_otm_price(cf, np.log(K / F), variance_hint=w)
    return np.exp(-r * T) * F * otm_fwd, F, ok


# ── Quadratic Variation Convexity ─────────────────────────────────────────────

def _qv_series_coeffs(n_terms: int = 24) -> tuple:
    """Taylor coefficients of f_v(y), f_θ(y) (see qv_variance_exact)."""
    from math import factorial
    ks = np.arange(3, 3 + n_terms)
    # f_v(y) = Σ_{k≥3} c_k y^k,  c_k = (−1)^{k+1}(2^k − 2k)/k!
    c = np.array([(-1) ** (k + 1) * (2.0**k - 2.0 * k) / factorial(int(k)) for k in ks])
    # f_θ(y) = Σ_{k≥3} c_k/(k+1) · y^{k+1}
    d = c / (ks + 1)
    return ks, c, d


_QV_KS, _QV_CV, _QV_CT = _qv_series_coeffs()


def qv_variance_exact(kappa: float, theta: float, sigma: float, v0: float, T: float) -> float:
    """
    Exact Var[∫₀ᵀ v_t dt] for the CIR variance dv = κ(θ − v)dt + σ√v dW.

    Var[I_T] = 2∫₀ᵀ Var(v_s)·(1 − e^{−κ(T−s)})/κ ds with
    Var(v_s) = v0σ²/κ (e^{−κs} − e^{−2κs}) + θσ²/(2κ)(1 − e^{−κs})², which
    integrates to

        Var[I_T] = σ²/κ³ · [ v0·f_v(κT) + θ·f_θ(κT) ]
        f_v(y) = 1 − 2y e^{−y} − e^{−2y}
        f_θ(y) = y + 2y e^{−y} − 5/2 + 2e^{−y} + ½e^{−2y}

    For y < 0.5 both brackets are evaluated from their Taylor series
    (f_v ≈ y³/3, f_θ ≈ y⁴/12) to avoid cancellation; the limit κ → 0 is
    σ²(v0T³/3 + θ·0) as expected. Deterministic, exact, microseconds.
    """
    if T <= 0 or sigma == 0:
        return 0.0
    y = kappa * T
    if y < 0.5:
        # f/κ³ from the series, written as powers of κ and T so κ → 0 is finite
        fv_over_k3 = float(np.sum(_QV_CV * kappa ** (_QV_KS - 3) * T ** _QV_KS))
        ft_over_k3 = float(np.sum(_QV_CT * kappa ** (_QV_KS - 2) * T ** (_QV_KS + 1)))
    else:
        e1 = np.exp(-y)
        e2 = e1 * e1
        fv_over_k3 = (1.0 - 2.0 * y * e1 - e2) / kappa**3
        ft_over_k3 = (y + 2.0 * y * e1 - 2.5 + 2.0 * e1 + 0.5 * e2) / kappa**3
    return float(sigma**2 * (v0 * fv_over_k3 + theta * ft_over_k3))


def simulate_qv_convexity(
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    T: float,
    n_paths: int = 10_000,
    n_steps: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Monte Carlo estimate of Var[∫₀ᵀ v_t dt] (cross-check for qv_variance_exact).

    Full-truncation Euler (Lord, Koekkoek & van Dijk 2010): the auxiliary
    process ṽ is NOT floored; only its positive part enters drift, diffusion
    and the integral,

        ṽ_{t+Δt} = ṽ_t + κ(θ − ṽ_t⁺)Δt + σ√(ṽ_t⁺)·ΔW_t,   v_t = ṽ_t⁺

    and ∫v dt is accumulated with the trapezoid rule. Returns the sample
    variance (ddof=0). Seeded by `rng` (default RANDOM_SEED).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)

    v_aux        = np.full(n_paths, v0, dtype=np.float64)
    integrated_v = np.zeros(n_paths, dtype=np.float64)

    for _ in range(n_steps):
        v_pos = np.maximum(v_aux, 0.0)
        dW = rng.standard_normal(n_paths) * sqrt_dt
        v_next = v_aux + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * dW
        integrated_v += 0.5 * (v_pos + np.maximum(v_next, 0.0)) * dt
        v_aux = v_next

    return float(np.var(integrated_v, ddof=0))


# ── Vomma-instability measure ─────────────────────────────────────────────────

def model_iv_shock(
    params: dict,
    T: np.ndarray,
    horizon: float = IV_SHOCK_HORIZON_YEARS,
    n_sd: float = IV_SHOCK_N_SD,
) -> np.ndarray:
    """
    Model-implied n-standard-deviation move of the ATM implied vol at maturity
    T over `horizon` years (decimal vol). See the module docstring.
    """
    T = np.asarray(T, dtype=float)
    kappa, theta = float(params["kappa"]), float(params["theta"])
    sig_v, v0 = float(params["sigma"]), float(params["v0"])
    kT = kappa * T
    B = np.where(kT > 1e-8, -np.expm1(-kT) / np.where(kT > 1e-8, kT, 1.0), 1.0)
    var_bar = np.maximum(theta + (v0 - theta) * B, 1e-12)
    return n_sd * sig_v * np.sqrt(max(v0, 0.0) * horizon) * B / (2.0 * np.sqrt(var_bar))


# ── Core Surface Computation ───────────────────────────────────────────────────

def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR),
            capture_output=True, text=True, timeout=5,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def compute_greeks_surface(
    S: float,
    r: float,
    q: float,
    params: dict,
    strikes: Optional[np.ndarray] = None,
    maturities: Optional[np.ndarray] = None,
    as_of_date: Optional[str] = None,
    mc_n_paths: int = 10_000,
    mc_n_steps: int = 100,
    verbose: bool = True,
    *,
    log_moneyness: Optional[Sequence[float]] = None,
    min_otm_price_rel: float = MIN_OTM_PRICE_REL,
    qv_method: str = "exact",
    iv_shock_horizon: float = IV_SHOCK_HORIZON_YEARS,
    iv_shock_n_sd: float = IV_SHOCK_N_SD,
    source: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute the second-order Greeks surface for a Heston calibration.

    For each (K, T) node:
      1. Price the Heston OTM option (Lewis/Fourier) → invert to σ_IV.
      2. Drop the node if its OTM value is below min_otm_price_rel × F
         (recorded in attrs["provenance"]["dropped_nodes"], never silent).
      3. Analytical BS Greeks at σ_IV (delta, vega, vomma, vanna).
      4. Attach the maturity's QV convexity and model IV shock.

    Parameters
    ----------
    S, r, q       : spot, risk-free rate, dividend yield (continuous)
    params        : dict with keys {kappa, theta, sigma, rho, v0}
    strikes       : absolute strikes used for every maturity; default is the
                    log-moneyness grid below applied to each maturity's forward
    maturities    : expiry grid in years; default [14,30,60,91,180,365] days
    as_of_date    : MARKET date the calibration describes ('YYYY-MM-DD');
                    recorded as a column and in the provenance
    mc_n_paths, mc_n_steps : only used when qv_method="mc"
    log_moneyness : ln(K/F) grid (default DEFAULT_LOG_MONEYNESS)
    min_otm_price_rel : node floor on OTM value / F
    qv_method     : "exact" (closed form, default) or "mc" (cross-check)
    iv_shock_horizon, iv_shock_n_sd : sizing of the model IV shock
    source        : extra provenance (e.g. calibration file) to record

    Returns
    -------
    DataFrame with columns
      [as_of_date,] K, T, T_days, log_moneyness, strike_index, iv, otm_price,
      delta, vega, vomma, vanna, qv_convexity, iv_shock
    and attrs["provenance"] (see save_greeks_surface for persistence).
    """
    p = {k: float(params[k]) for k in ("kappa", "theta", "sigma", "rho", "v0")}
    kappa, theta, sigma, v0 = p["kappa"], p["theta"], p["sigma"], p["v0"]
    outside = {k: v for k, v in p.items()
               if not (HESTON_BOUNDS[k][0] <= v <= HESTON_BOUNDS[k][1])}
    if outside:
        logger.warning("Greeks surface built from parameters outside HESTON_BOUNDS: %s", outside)

    if maturities is None:
        maturities = DEFAULT_MATURITIES_DAYS / 365.0
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    lm_grid = (np.asarray(DEFAULT_LOG_MONEYNESS if log_moneyness is None else log_moneyness,
                          dtype=float))
    if qv_method not in ("exact", "mc"):
        raise ValueError(f"qv_method must be 'exact' or 'mc', got {qv_method!r}")

    rng = np.random.default_rng(RANDOM_SEED)
    t0  = time.time()
    frames: List[pd.DataFrame] = []
    dropped: List[dict] = []
    n_grid = 0

    for T in maturities:
        F = S * np.exp((r - q) * T)
        K_arr = (F * np.exp(lm_grid) if strikes is None
                 else np.asarray(strikes, dtype=float))
        n_K = len(K_arr)
        n_grid += n_K
        T_days = int(round(T * 365))
        x = np.log(K_arr / F)

        otm_disc, _, ok = heston_otm_prices(S, K_arr, T, r, q, p)
        if not ok:
            raise RuntimeError(
                f"Heston Fourier pricer did not converge at T={T_days}d for params {p}"
            )
        otm_fwd = otm_disc * np.exp(r * T)
        ivs, status = implied_vol_black(
            otm_fwd, F, K_arr, T, is_call=(x >= 0), discount=1.0,
            sigma_min=1e-4, sigma_max=10.0,
            min_otm_price=min_otm_price_rel * F, full_output=True,
        )

        ill_posed = (status == IV_BELOW_FLOOR) | (status == IV_NO_TIME_VALUE)
        failed = (status != IV_OK) & ~ill_posed
        if failed.any():
            bad = [(float(x[j]), IV_STATUS[int(status[j])]) for j in np.flatnonzero(failed)]
            raise RuntimeError(
                f"IV inversion failed for well-posed nodes at T={T_days}d "
                f"(ln(K/F), status): {bad}"
            )
        for j in np.flatnonzero(ill_posed):
            dropped.append({
                "T_days": T_days,
                "log_moneyness": round(float(x[j]), 6),
                "K": round(float(K_arr[j]), 4),
                "otm_price": float(otm_disc[j]),
                "otm_fwd_rel": float(otm_fwd[j] / F),
                "reason": f"OTM value below {min_otm_price_rel:g}·F",
            })

        valid = status == IV_OK
        if not valid.any():
            logger.warning("No well-posed nodes at T=%dd (all below the price floor)", T_days)
            continue

        K_v, iv_v = K_arr[valid], ivs[valid]
        n_v = int(valid.sum())
        Sv, Tv = np.full(n_v, S), np.full(n_v, T)
        rv, qv = np.full(n_v, r), np.full(n_v, q)

        if qv_method == "exact":
            qv_conv = qv_variance_exact(kappa, theta, sigma, v0, T)
        else:
            qv_conv = simulate_qv_convexity(kappa, theta, sigma, v0, T,
                                            mc_n_paths, mc_n_steps, rng)

        frame = pd.DataFrame({
            "K":             K_v,
            "T":             float(T),
            "T_days":        T_days,
            "log_moneyness": x[valid],
            "strike_index":  np.flatnonzero(valid).astype(int),
            "iv":            iv_v,
            "otm_price":     otm_disc[valid],
            "delta":         _bs_delta(Sv, K_v, Tv, rv, qv, iv_v),
            "vega":          _bs_vega(Sv, K_v, Tv, rv, qv, iv_v),
            "vomma":         _bs_vomma(Sv, K_v, Tv, rv, qv, iv_v),
            "vanna":         _bs_vanna(Sv, K_v, Tv, rv, qv, iv_v),
            "qv_convexity":  float(qv_conv),
            "iv_shock":      float(model_iv_shock(p, T, iv_shock_horizon, iv_shock_n_sd)),
        })
        if as_of_date:
            frame.insert(0, "as_of_date", as_of_date)
        frames.append(frame)

        if verbose:
            logger.info(
                "  T=%4dd: %d/%d nodes, IV=[%.1f%%..%.1f%%], QV_conv=%.3e",
                T_days, n_v, n_K, iv_v.min() * 100, iv_v.max() * 100, qv_conv,
            )

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if dropped:
        logger.warning(
            "Greeks surface: %d of %d grid nodes dropped (OTM value below %.0e·F): %s",
            len(dropped), n_grid, min_otm_price_rel,
            ", ".join(f"{d['T_days']}d@{d['log_moneyness']:+.3f}" for d in dropped),
        )

    df.attrs["provenance"] = {
        "schema_version": SURFACE_SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_revision": _git_revision(),
        "S": float(S), "r": float(r), "q": float(q),
        "params": p,
        "params_outside_heston_bounds": sorted(outside),
        "source": source or {},
        "grid": {
            "maturities_days": [int(round(t * 365)) for t in maturities],
            "log_moneyness": None if strikes is not None else [float(v) for v in lm_grid],
            "strikes": None if strikes is None else [float(k) for k in np.asarray(strikes)],
        },
        "n_grid_nodes": int(n_grid),
        "n_nodes": int(len(df)),
        "dropped_nodes": dropped,
        "pricer": "Heston Lewis (2001) Fourier formula, BS control variate, adaptive trapezoid (~1e-14·F)",
        "iv_solver": "implied_vol.implied_vol_black on the OTM price (bracketed Newton)",
        "min_otm_price_rel": float(min_otm_price_rel),
        "qv_method": qv_method,
        "iv_shock": {"horizon_years": float(iv_shock_horizon), "n_sd": float(iv_shock_n_sd)},
        "units": GREEKS_UNITS,
    }
    if verbose:
        logger.info("Greeks surface: %d/%d nodes in %.1fs", len(df), n_grid, time.time() - t0)
    return df


# ── Unstable Hedge Flagging ────────────────────────────────────────────────────

def flag_unstable_hedges(
    df: pd.DataFrame,
    threshold: float = UNSTABLE_RATIO_THRESHOLD,
    iv_shock: Optional[float] = None,
    *,
    greek: str = "vomma",
) -> pd.DataFrame:
    """
    Flag nodes where the vomma P&L of a model-sized IV move is a material
    fraction of the vega P&L (see the module docstring).

    Parameters
    ----------
    threshold       : flag when |vomma_pnl_ratio| > threshold (default 0.25)
    iv_shock        : IV move Δσ (decimal) for every node; default is the
                      per-maturity model shock in df["iv_shock"]
    greek           : column the descriptive z-score is computed on

    The signature changed with the metric: the first positional argument used
    to be a z-score threshold (threshold_sigma); it is now the ratio threshold.

    Columns added
    -------------
    vomma_pnl_ratio : ½·vomma·Δσ² / (vega·|Δσ|)  (dimensionless)
    is_unstable     : |vomma_pnl_ratio| > threshold
    vomma_zscore    : pooled cross-sectional z-score of `greek` — descriptive
                      only (scale-free; it says which nodes are extreme
                      relative to this surface, not whether any are risky)
    """
    out = df.copy()
    if iv_shock is None:
        if "iv_shock" not in out.columns:
            raise ValueError("df has no 'iv_shock' column; pass iv_shock explicitly")
        shock = out["iv_shock"].to_numpy(dtype=float)
    else:
        shock = np.full(len(out), float(iv_shock))
        out["iv_shock"] = shock

    vega = out["vega"].to_numpy(dtype=float)
    vomma = out["vomma"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(vega > 0, 0.5 * vomma * np.abs(shock) / vega, np.nan)
    out["vomma_pnl_ratio"] = ratio
    out["is_unstable"] = np.abs(np.nan_to_num(ratio, nan=0.0)) > threshold

    vals = out[greek].to_numpy(dtype=float)
    mu = float(np.mean(vals)) if len(vals) else 0.0
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    out["vomma_zscore"] = (vals - mu) / max(std, 1e-12)

    if isinstance(df.attrs.get("provenance"), dict):
        prov = dict(df.attrs["provenance"])
        prov["unstable_rule"] = {
            "metric": "vomma_pnl_ratio = 0.5*vomma*|iv_shock|/vega",
            "threshold": float(threshold),
            "iv_shock": "per-maturity model shock" if iv_shock is None else float(iv_shock),
        }
        out.attrs["provenance"] = prov

    logger.info(
        "Flagged %d / %d unstable nodes (|vomma P&L / vega P&L| > %.2f)",
        int(out["is_unstable"].sum()), len(out), threshold,
    )
    return out


# ── Vomma Heatmap ─────────────────────────────────────────────────────────────

def plot_vomma_heatmap(
    df: pd.DataFrame,
    greek: str = "vomma",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    2-D heatmap: log-moneyness ln(K/F) on x, maturity on y, {greek} as colour.

    Diverging RdBu_r map centred on zero. Unstable nodes (is_unstable) are
    marked with a black ×; grid nodes dropped below the price floor are blank.
    """
    df_p = df.copy()
    df_p["log_m_r"] = df_p["log_moneyness"].round(4)

    pivot = df_p.pivot_table(
        index="T_days", columns="log_m_r", values=greek, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(np.abs(vals).max(), 1e-6) if len(vals) > 0 else 1.0
    norm_ = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(
        pivot.values, aspect="auto",
        cmap=plt.get_cmap("RdBu_r"), norm=norm_,
        origin="lower",
    )

    x_labels = [f"{v:+.3f}" for v in pivot.columns]
    y_labels  = [f"{v}d"   for v in pivot.index]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Log-moneyness  ln(K/F)", fontsize=10)
    ax.set_ylabel("Maturity", fontsize=10)
    ax.set_title(
        f"Heston surface — {greek}\n"
        f"(red = positive  |  blue = negative  |  × = unstable  |  blank = below price floor)",
        fontsize=11,
    )
    unit = GREEKS_UNITS.get(greek, "")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{greek} [{unit}]" if unit else greek)

    if "is_unstable" in df_p.columns:
        col_list = list(pivot.columns)
        row_list = list(pivot.index)
        for _, row in df_p[df_p["is_unstable"]].iterrows():
            xi = col_list.index(row["log_m_r"])
            yi = row_list.index(row["T_days"])
            ax.plot(xi, yi, "kx", markersize=10, markeredgewidth=2)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap saved to %s", save_path)

    return fig


# ── Persistence ───────────────────────────────────────────────────────────────

def save_greeks_surface(
    df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Persist the Greeks surface to parquet. The provenance dict in
    df.attrs["provenance"] (params, S, r, q, market date, source calibration,
    grid, dropped nodes, units, code revision) is written into the parquet
    schema metadata under b"greeks_provenance"; read it back with
    load_greeks_surface(). Plain pd.read_parquet still returns the table.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path = Path(path) if path else GREEKS_SURFACE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    prov = df.attrs.get("provenance")
    if prov:
        meta = dict(table.schema.metadata or {})
        meta[_PROVENANCE_KEY] = json.dumps(prov, default=str).encode()
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, out_path)
    logger.info("Greeks surface saved: %s  (%d rows)", out_path, len(df))
    return out_path


def load_greeks_surface(path: Optional[Path] = None) -> pd.DataFrame:
    """Read a saved surface; provenance (if any) is put in attrs["provenance"]."""
    import pyarrow.parquet as pq

    in_path = Path(path) if path else GREEKS_SURFACE_PATH
    df = pd.read_parquet(in_path)
    meta = pq.read_schema(in_path).metadata or {}
    if _PROVENANCE_KEY in meta:
        df.attrs["provenance"] = json.loads(meta[_PROVENANCE_KEY].decode())
    else:
        df.attrs["provenance"] = None    # legacy file: no provenance recorded
    return df


# ── RiskMonitor Class ─────────────────────────────────────────────────────────

def _calibrator_default_q() -> float:
    """The flat dividend yield JointCalibrator assumes when none is recorded.

    Only legacy (pre-C22) results lack a recorded q; model_version 2 results
    store the parity-implied median q they actually used.
    """
    from joint_vol_calibration.calibration.joint_calibrator import FLAT_CARRY_Q
    return float(FLAT_CARRY_Q)


class RiskMonitor:
    """
    Orchestrates the second-order Greeks risk analysis.

    Typical workflow
    ----------------
      monitor = RiskMonitor.from_calibration(cal_path)   # or RiskMonitor(S, r, q, params, as_of_date)
      surface = monitor.build()            # compute Greeks surface
      flagged = monitor.flag()             # vomma-instability flags
      fig     = monitor.plot()             # heatmap (saved to disk)
      path    = monitor.save()             # persist to parquet (+ provenance)
      checks  = monitor.validate()         # automated sanity checks

    as_of_date is the MARKET date the parameters describe. It is no longer
    defaulted to today's date (the run date), which mislabelled the surface;
    when it is not given the surface carries no market date and the
    provenance says so.
    """

    def __init__(
        self,
        S: float,
        r: float,
        q: float,
        params: dict,
        as_of_date: Optional[str] = None,
        source: Optional[dict] = None,
    ):
        self.S          = S
        self.r          = r
        self.q          = q
        self.params     = params
        self.as_of_date = as_of_date
        self.source     = dict(source or {})
        self.surface: Optional[pd.DataFrame] = None
        self.flagged: Optional[pd.DataFrame] = None

    @classmethod
    def from_calibration(cls, path, q: Optional[float] = None) -> "RiskMonitor":
        """
        Build a monitor from a joint_cal_*.pkl: params, spot S, rate r, the
        dividend yield the calibrator used (recorded q, else JointCalibrator's
        default) and the market date (as_of_date) all come from the file.
        """
        path = Path(path)
        with open(path, "rb") as f:
            cal = pickle.load(f)
        params = {k: float(cal["params"][k]) for k in ("kappa", "theta", "sigma", "rho", "v0")}
        for key in ("S", "r", "as_of_date"):
            if cal.get(key) is None:
                raise ValueError(f"{path.name} has no '{key}'; cannot build a dated surface")
        if q is not None:
            q_val, q_src = float(q), "argument"
        elif cal.get("q") is not None:
            q_val, q_src = float(cal["q"]), "calibration file"
        else:
            q_val, q_src = _calibrator_default_q(), "JointCalibrator default (not recorded in file)"
        try:
            rel = str(path.resolve().relative_to(ROOT_DIR.resolve()))
        except ValueError:
            rel = str(path)
        losses = cal.get("leg_losses", cal.get("losses", {})) or {}
        source = {
            "calibration_file": rel,
            "q_source": q_src,
            "spx_iv_rmse_vp": (float(losses["spx_iv_rmse"])
                               if losses.get("spx_iv_rmse") is not None else None),
            "spot_date": cal.get("spot_date"),
            "run_date": cal.get("run_date"),
        }
        return cls(float(cal["S"]), float(cal["r"]), q_val, params,
                   as_of_date=str(cal["as_of_date"]), source=source)

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
        mc_n_paths: int = 10_000,
        mc_n_steps: int = 100,
        verbose: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute the Greeks surface. Must be called before flag/plot/save."""
        logger.info("Building Greeks surface (S=%.1f, date=%s)...", self.S, self.as_of_date)
        self.surface = compute_greeks_surface(
            S=self.S, r=self.r, q=self.q,
            params=self.params,
            strikes=strikes, maturities=maturities,
            as_of_date=self.as_of_date,
            mc_n_paths=mc_n_paths, mc_n_steps=mc_n_steps,
            verbose=verbose, source=self.source, **kwargs,
        )
        self.flagged = None
        return self.surface

    def flag(self, threshold: float = UNSTABLE_RATIO_THRESHOLD, **kwargs) -> pd.DataFrame:
        """Flag vomma-unstable nodes (see flag_unstable_hedges)."""
        if self.surface is None:
            raise RuntimeError("Call build() before flag()")
        self.flagged = flag_unstable_hedges(self.surface, threshold, **kwargs)
        return self.flagged

    def plot(
        self,
        greek: str = "vomma",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generate heatmap. Uses flagged surface when available."""
        df = self.flagged if self.flagged is not None else self.surface
        if df is None:
            raise RuntimeError("Call build() before plot()")
        return plot_vomma_heatmap(df, greek=greek,
                                   save_path=save_path or HEATMAP_PATH)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the (flagged, if available) surface to parquet with provenance."""
        df = self.flagged if self.flagged is not None else self.surface
        if df is None:
            raise RuntimeError("Call build() before save()")
        return save_greeks_surface(df, path)

    def validate(self) -> dict:
        """
        Automated sanity checks on the computed surface.

        Checks
        ------
        vega_positive          : BS vega > 0 everywhere
        vomma_positive_far_otm : vomma > 0 wherever d1·d2 > 0 is implied by
                                 |ln(K/F)| > 0.15 and σ²T/2 < 0.15
        iv_in_range            : all σ_IV ∈ (0.01, 3.0)
        qv_nonnegative         : Var[QV] ≥ 0
        vomma_fd_check         : analytic vomma vs central finite difference
                                 within 5% (+ small absolute tolerance)
        iv_reprices            : BS at σ_IV reproduces the Heston OTM price to 1e-8 (relative)
        no_well_posed_drops    : no node above the price floor was dropped
        """
        if self.surface is None:
            raise RuntimeError("Call build() before validate()")
        df = self.surface
        checks: dict = {}

        checks["vega_positive"] = bool((df["vega"] > 0).all())

        far = df[(np.abs(df["log_moneyness"]) > 0.15) & (df["iv"] ** 2 * df["T"] / 2 < 0.15)]
        checks["vomma_positive_far_otm"] = bool((far["vomma"] > 0).all())

        checks["iv_in_range"] = bool((df["iv"] > 0.01).all() and (df["iv"] < 3.0).all())

        checks["qv_nonnegative"] = bool((df["qv_convexity"] >= 0).all())

        args = [df[c].to_numpy(dtype=float) for c in ("K", "T")]
        n = len(df)
        S_, r_, q_ = np.full(n, self.S), np.full(n, self.r), np.full(n, self.q)
        fd = _bs_vomma_numerical(S_, args[0], args[1], r_, q_, df["iv"].to_numpy(dtype=float))
        rel_err = np.abs(fd - df["vomma"].to_numpy()) / (np.abs(df["vomma"].to_numpy()) + 1e-4)
        checks["vomma_fd_check"] = bool((rel_err < 0.05).all())

        if "otm_price" in df.columns:
            is_call = df["log_moneyness"].to_numpy() >= 0
            rep = bs_price(S_, args[0], args[1], r_, q_, df["iv"].to_numpy(dtype=float), is_call)
            otm = df["otm_price"].to_numpy(dtype=float)
            checks["iv_reprices"] = bool((np.abs(rep - otm) <= 1e-8 * otm + 1e-12 * self.S).all())

        prov = df.attrs.get("provenance") or {}
        floor = prov.get("min_otm_price_rel")
        if floor is not None:
            checks["no_well_posed_drops"] = all(
                d["otm_fwd_rel"] < floor for d in prov.get("dropped_nodes", [])
            )

        all_ok = all(checks.values())
        logger.info("Validation: %s — %s",
                    "ALL PASS" if all_ok else "FAILURES DETECTED", checks)
        return checks

    def __repr__(self) -> str:
        rows = len(self.surface) if self.surface is not None else 0
        return (f"RiskMonitor(S={self.S}, date={self.as_of_date}, "
                f"surface_rows={rows})")


# ── Regeneration entry point ──────────────────────────────────────────────────

def build_surface_from_calibration(
    calibration_path,
    out_path: Optional[Path] = None,
    plot_path: Optional[Path] = None,
    threshold: float = UNSTABLE_RATIO_THRESHOLD,
    q: Optional[float] = None,
) -> pd.DataFrame:
    """
    Regenerate the published Greeks surface from one calibration pickle:
    build, flag, validate (raises on any failed check) and save with
    provenance. Returns the flagged surface.
    """
    monitor = RiskMonitor.from_calibration(calibration_path, q=q)
    monitor.build(verbose=True)
    monitor.flag(threshold)
    checks = monitor.validate()
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Greeks surface failed validation: {failed}")
    monitor.save(out_path)
    if plot_path is not None:
        plt.close(monitor.plot(save_path=plot_path))
    return monitor.flagged


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the Greeks surface parquet from a joint_cal_*.pkl calibration.",
    )
    parser.add_argument("--calibration", required=True, help="path to a joint_cal_*.pkl")
    parser.add_argument("--out", default=str(GREEKS_SURFACE_PATH), help="output parquet path")
    parser.add_argument("--plot", default=None, help="optional heatmap PNG path")
    parser.add_argument("--threshold", type=float, default=UNSTABLE_RATIO_THRESHOLD,
                        help="vomma P&L / vega P&L instability threshold")
    parser.add_argument("--q", type=float, default=None,
                        help="override the dividend yield (default: from the file)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build_surface_from_calibration(args.calibration, Path(args.out),
                                        Path(args.plot) if args.plot else None,
                                        args.threshold, args.q)
    prov = df.attrs["provenance"]
    print(f"{args.out}: {prov['n_nodes']}/{prov['n_grid_nodes']} nodes, "
          f"{int(df['is_unstable'].sum())} unstable, as_of {prov['as_of_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
