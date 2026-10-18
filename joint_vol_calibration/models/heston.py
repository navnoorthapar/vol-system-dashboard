"""
heston.py — Heston (1993) / Bates (1996) stochastic-volatility engine.

The Heston (1993) model:
  dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
  dv_t = kappa(theta - v_t)dt + sigma * sqrt(v_t) dW_t^v
  dW_t^S * dW_t^v = rho dt

Parameters:
  kappa : mean-reversion speed of variance (> 0)
  theta : long-run variance (> 0)
  sigma : vol-of-vol — volatility of the variance process (> 0)
  rho   : correlation between spot and variance shocks (< 0 for equity = skew)
  v0    : initial instantaneous variance (= spot vol^2)

This module provides:
  1. characteristic_function() / bates_characteristic_function() — closed-form
     CFs ('little trap' form of Albrecher et al. 2007).
  2. heston_call_batch() / bates_call_batch() — the pricing engine: Lewis (2001)
     contour integral with a Black-Scholes control variate, evaluated by an
     adaptive trapezoid rule (vectorised over strikes). heston_call_price() and
     bates_call_price() are single-strike wrappers around the same engine.
     No FFT is used anywhere in this module.
  3. heston_vix_term_index() — what ^VIX9D / ^VIX / ^VIX3M / ^VIX6M measure.
  4. heston_vix_future_exact() / heston_vix_futures_curve() — exact VIX futures.
  5. calibrate_to_spx() — SPX-only differential-evolution calibration.
  6. heston_monte_carlo() — Andersen (2008) QE scheme (or full-truncation Euler).
  7. heston_greeks() — finite-difference Greeks with documented units.

VIX under Heston — index versus futures (two different instruments):
  The Cboe constant-maturity indices are SPOT-starting:
    I(tau)^2 = (1/tau) E_0[ int_0^tau v_s ds ]
             = theta + (v0 - theta) * (1 - e^{-kappa*tau}) / (kappa*tau)
  with tau = 9, 30, 93, 184 calendar days for ^VIX9D, ^VIX, ^VIX3M, ^VIX6M
  (config.VIX_INDEX_TENOR_DAYS). heston_vix_term_index() implements this; it
  does not depend on sigma.

  A VIX FUTURE expiring at T settles on VIX_T = sqrt(VIX_T^2), where
    VIX_T^2 = theta*(1 - b) + b * v_T,   b = (1 - e^{-kappa*tau})/(kappa*tau),
  tau = 30/365, and v_T follows the CIR transition law. Its fair value is
  E[VIX_T], which by Jensen's inequality is strictly below sqrt(E[VIX_T^2]);
  the gap grows with sigma (vol-of-vol) and T. heston_vix_future_exact()
  evaluates E[VIX_T] exactly from the CIR Laplace transform;
  heston_vix_futures_upper_bound() keeps the old sqrt(E[VIX_T^2]) value under
  an explicit name.

  Bates (1996) adds compound-Poisson jumps (intensity lam, log-jump
  N(mu_j, sigma_j^2)). The Cboe VIX is the log-contract (model-free) variance
  -(2/tau) E[ln(S_tau / F)], which under Bates picks up a deterministic jump
  term 2*lam*(e^{mu_j + sigma_j^2/2} - 1 - mu_j) on top of the diffusive part
  (Carr & Wu 2009); bates_vix_term_index() and bates_vix_future_exact()
  include it.
"""

import logging
import pickle
import warnings
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import ndtr

from joint_vol_calibration.config import (
    HESTON_DEFAULTS, HESTON_BOUNDS, MC_PATHS, MC_STEPS_PER_YEAR,
    MC_ANTITHETIC, RANDOM_SEED, MC_CACHE_DIR,
)

logger = logging.getLogger(__name__)

# Risk-free rate (we use a simple constant; replace with actual T-bill rate)
_DEFAULT_RATE = 0.045   # 4.5% — approximate 2023-2024 level
_DEFAULT_DIV  = 0.013   # 1.3% dividend yield on SPX

_VIX_TAU = 30.0 / 365.0   # VIX integration window (30 calendar days)


# ── Small numerical helpers ───────────────────────────────────────────────────

def _clog1p(z: np.ndarray) -> np.ndarray:
    """
    Complex log(1 + z) (principal branch) that keeps full relative accuracy for
    small |z|. numpy's complex log1p evaluates log(|1+z|) directly, which loses
    all precision once |z| ~ 1e-16 — exactly the regime of small vol-of-vol,
    where the CF divides this quantity by sigma^2.
    """
    z = np.asarray(z, dtype=complex)
    re, im = z.real, z.imag
    return 0.5 * np.log1p(2.0 * re + re * re + im * im) + 1j * np.arctan2(im, 1.0 + re)


def _one_minus_exp_over(x: np.ndarray) -> np.ndarray:
    """(1 - e^{-x}) / x for x >= 0, with the x -> 0 limit (= 1) handled exactly."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-8
    safe = np.where(small, 1.0, x)
    return np.where(small, 1.0 - 0.5 * x + x * x / 6.0, -np.expm1(-safe) / safe)


def _expected_integrated_variance(T, kappa: float, theta: float, v0: float):
    """E[int_0^T v_s ds] = theta*T + (v0 - theta)*(1 - e^{-kappa T})/kappa (vectorised)."""
    T = np.asarray(T, dtype=float)
    return T * (theta + (v0 - theta) * _one_minus_exp_over(kappa * T))


# ── Characteristic Function (Heston closed form) ──────────────────────────────

def _heston_log_cf(z, T: float, kappa: float, theta: float, sigma: float,
                   rho: float, v0: float) -> np.ndarray:
    """
    log E[exp(i z X_T)] with X_T = ln(S_T / F_T) (log-FORWARD coordinates), for
    complex z (vectorised).

    'Little trap' form (Albrecher et al. 2007), rewritten so that sigma only
    appears multiplied — never divided — outside a log1p:
        s    = z^2 + i z
        beta = kappa - i rho sigma z,      d = sqrt(beta^2 + sigma^2 s)
        g    = (beta - d)/(beta + d)  =  -sigma^2 s / (beta + d)^2
        D    = -s (1 - e^{-dT}) / ((beta + d)(1 - g e^{-dT}))
        C    = kappa theta [ -s T/(beta + d)
                             - (2/sigma^2) log((1 - g e^{-dT})/(1 - g)) ]
    The principal square root (Re d >= 0) keeps |e^{-dT}| <= 1. Algebraically
    identical to the textbook form but free of the (beta - d)/sigma^2
    cancellation, so it is stable down to sigma -> 0 (the Black-Scholes limit).
    """
    z = np.asarray(z, dtype=complex)
    sig = max(float(sigma), 1e-12)
    s = z * (z + 1j)
    beta = kappa - 1j * rho * sig * z
    d = np.sqrt(beta * beta + sig * sig * s)
    bpd = beta + d
    one_m_e = -np.expm1(-d * T)
    e = 1.0 - one_m_e
    g = -sig * sig * s / (bpd * bpd)
    D = -s * one_m_e / (bpd * (1.0 - g * e))
    # log((1 - g e)/(1 - g)) = log1p(g (1 - e)/(1 - g)); divided by sigma^2
    log_ratio = _clog1p(g * one_m_e / (1.0 - g))
    C = kappa * theta * (-s * T / bpd - 2.0 * log_ratio / (sig * sig))
    return C + D * v0


def _bates_jump_log_cf(z, T: float, lam: float, mu_j: float, sigma_j: float) -> np.ndarray:
    """
    Log-CF of the compensated compound-Poisson jump part of ln(S_T/F_T):
      lam T (E[e^{izJ}] - 1 - i z kbar),  J ~ N(mu_j, sigma_j^2),
      kbar = E[e^J - 1] = e^{mu_j + sigma_j^2/2} - 1.
    The -i z lam T kbar term is the risk-neutral drift compensator.
    """
    z = np.asarray(z, dtype=complex)
    kbar = np.expm1(mu_j + 0.5 * sigma_j**2)
    return lam * T * (np.exp(1j * z * mu_j - 0.5 * z * z * sigma_j**2) - 1.0 - 1j * z * kbar)


def characteristic_function(
    phi: complex,
    S: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
) -> complex:
    """
    Heston characteristic function E[e^{i*phi*ln(S_T)}] under risk-neutral measure.

    Uses the 'Little Trap' formulation (Albrecher et al. 2007) which avoids
    the branch discontinuity in the original Heston formula that causes
    numerical instability for long maturities or extreme parameters.

    Parameters
    ----------
    phi   : complex (or complex array) — characteristic function argument
    S     : float   — current spot price
    T     : float   — time to expiry in years
    r     : float   — risk-free rate (continuously compounded)
    q     : float   — continuous dividend yield
    kappa, theta, sigma, rho, v0 : Heston parameters

    Returns
    -------
    complex — E[exp(i*phi*ln(S_T))], the CF of the log terminal (dollar) price,
    i.e. exp(i*phi*ln F_T) times the log-forward CF.

    Note on coordinates: this CF is in dollar log-price, so any Fourier
    inversion against it must use dollar log-strike ln(K). The pricing engine
    below works in log-forward coordinates (x = ln(F/K)) instead.
    """
    phi = np.asarray(phi, dtype=complex)
    log_F = np.log(S) + (r - q) * T
    return np.exp(_heston_log_cf(phi, T, kappa, theta, sigma, rho, v0)
                  + 1j * phi * log_F)


# ── Pricing engine: Lewis (2001) contour + Black-Scholes control variate ─────
#
# For X = ln(S_T/F) with CF phi(u), Lewis (2001) gives, with x = ln(F/K),
#   C = DF [ F - sqrt(FK)/pi * int_0^inf Re(e^{iux} phi(u - i/2)) / (u^2 + 1/4) du ].
# The contour Im(u) = -1/2 needs only E[(S_T/F)^{1/2}], which is finite for
# every Heston/Bates parameter set and maturity — unlike Carr-Madan damping
# alpha = 1.5, which needs E[S_T^{2.5}] and silently breaks beyond the
# moment-explosion time (Andersen & Piterbarg 2007).
#
# Subtracting the same formula for Black-Scholes with total variance w (whose
# CF on the contour is the real exp(-w(u^2 + 1/4)/2)) gives
#   C = C_BS(w) + DF sqrt(FK)/pi * int_0^inf Re(e^{iux}(phi_BS - phi)) / (u^2+1/4) du.
# The control variate removes the slowly-decaying Laplace kernel from the
# aliasing error of the trapezoid rule, so the rule converges geometrically
# in the step h once 2*pi/h exceeds |x| plus the width of the distribution.
# Truncation (U) is read off |phi| on a geometric scan; the step is halved
# until two successive trapezoid estimates agree to _PRICE_RTOL * DF * F.
# This is what makes short maturities (1-7 days) and v0 at its 1e-4 floor
# accurate: the old fixed 512-point / v<=300 grid was not.

_PRICE_RTOL = 1e-10          # target quadrature error, as a fraction of DF*F
# Cap on trapezoid nodes per slice. Only near-degenerate corners reach it:
# theta = v0 = 1e-4 with sigma = 2 and kappa = 0.1 (CIR stuck near zero with
# rare huge excursions) needs both a slowly decaying CF (large U) and fat
# tails (small h), ~0.9M nodes at T = 1y. 2**16 was too low there (0.8 vp
# error at 1y, with a warning); 2**20 is accurate to 5e-9 * F on every corner
# of HESTON_BOUNDS at ~1 s per slice. Typical slices use a few hundred nodes.
_LEWIS_MAX_NODES = 2**20
_LEWIS_BLOCK = 4096          # nodes per vectorised block (bounds memory)


class _LewisGrid(NamedTuple):
    """Uniform trapezoid grid u_k = k*h, k = 0..n-1, on the Lewis contour."""
    h: float
    n: int


def _lewis_sum(x: np.ndarray, u: np.ndarray, gw: np.ndarray) -> np.ndarray:
    """Re( sum_k exp(i u_k x_j) * gw_k ) for every x_j, in memory-bounded blocks."""
    out = np.zeros(len(x))
    for start in range(0, len(u), _LEWIS_BLOCK):
        ub = u[start:start + _LEWIS_BLOCK]
        gb = gw[start:start + _LEWIS_BLOCK]
        # explicit multiply-sum rather than a BLAS matvec: same cost, and no
        # BLAS thread pool (which stalls badly on an oversubscribed machine)
        out += (np.exp(1j * np.outer(x, ub)) * gb).sum(axis=1).real
    return out


def _lewis_call_prices(
    S: float,
    strikes,
    T: float,
    r: float,
    q: float,
    log_cf: Callable[[np.ndarray], np.ndarray],
    total_var: float,
    jump_extent: float = 0.0,
    grid: Optional[_LewisGrid] = None,
    rtol: float = _PRICE_RTOL,
) -> Tuple[np.ndarray, _LewisGrid]:
    """
    Core engine: discounted call prices for one expiry, plus the grid used.

    Parameters
    ----------
    log_cf     : callable z -> log E[exp(i z ln(S_T/F))] (complex z, vectorised)
    total_var  : approximate Var[ln S_T] (sets the control-variate vol and the
                 initial step); accuracy does not depend on it being exact.
    jump_extent: extra half-width of the log-return distribution from jumps.
    grid       : if given, price on exactly this grid (no adaptation) — used by
                 the Greeks so that bumped prices share one quadrature rule.
    """
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    df = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    calls = np.empty(len(strikes))

    finite = np.isfinite(strikes)
    pos = finite & (strikes > 0)
    calls[~finite] = np.nan
    nonpos = finite & ~pos
    calls[nonpos] = df * (F - strikes[nonpos])      # call with K <= 0 = forward - K
    if not pos.any():
        return calls, grid if grid is not None else _LewisGrid(1.0, 1)
    K = strikes[pos]
    x = np.log(F / K)

    w = max(float(total_var), 1e-14)
    sd = np.sqrt(w)
    d1 = x / sd + 0.5 * sd
    bs_cv = df * (F * ndtr(d1) - K * ndtr(d1 - sd))
    pref = df * np.sqrt(F * K) / np.pi

    def g(u):
        """(phi_BS - phi)(u - i/2) / (u^2 + 1/4) on the Lewis contour."""
        uu = u * u + 0.25
        return (np.exp(-0.5 * w * uu) - np.exp(log_cf(u - 0.5j))) / uu

    if grid is not None:
        u = grid.h * np.arange(grid.n)
        gw = g(u) * grid.h
        gw[0] *= 0.5
        calls[pos] = bs_cv + pref * _lewis_sum(x, u, gw)
    else:
        tol_abs = rtol * df * F
        # ── Truncation: |integrand| <= (|phi| + phi_BS)/u^2, so the tail beyond
        #    U is bounded by sup_{u>=U}(|phi| + phi_BS) / U.
        scan = np.geomspace(1e-2, 1e7, 600) / sd
        amp = np.abs(np.exp(log_cf(scan - 0.5j))) + np.exp(-0.5 * w * (scan * scan + 0.25))
        amp = np.where(np.isfinite(amp), amp, np.inf)
        tail = np.maximum.accumulate(amp[::-1])[::-1] / scan
        ok = pref.max() * tail <= 0.1 * tol_abs
        U = scan[np.argmax(ok)] if ok.any() else scan[-1]
        truncated = not ok.any()

        # ── Step: start from the distribution width, halve until converged.
        xmax = float(np.max(np.abs(x)))
        h = 2.0 * np.pi / (xmax + 10.0 * sd + jump_extent)
        h = min(h, U / 16.0)
        n = int(np.ceil(U / h)) + 1
        if n > _LEWIS_MAX_NODES:
            # Keep the step (aliasing is unbounded if h is too coarse) and
            # truncate earlier instead: the tail is bounded by 2/U.
            n = _LEWIS_MAX_NODES
            truncated = True
        u = h * np.arange(n)
        gw = g(u) * h
        gw[0] *= 0.5
        integral = _lewis_sum(x, u, gw)
        converged = False
        for _ in range(12):
            if 2 * n - 1 > _LEWIS_MAX_NODES:
                break
            u_odd = h * (np.arange(n - 1) + 0.5)
            fine = 0.5 * integral + _lewis_sum(x, u_odd, g(u_odd) * (0.5 * h))
            err = np.max(pref * np.abs(fine - integral))
            integral = fine
            h *= 0.5
            n = 2 * n - 1
            if err <= tol_abs:
                converged = True
                break
        if truncated or not converged:
            warnings.warn("Lewis quadrature did not reach rtol=%.1e (T=%.4g, %s); "
                          "prices may be inaccurate" % (
                              rtol, T, "CF not decayed within the node budget"
                              if truncated else "step refinement not converged"),
                          RuntimeWarning, stacklevel=3)
        grid = _LewisGrid(h, n)
        calls[pos] = bs_cv + pref * integral

    return calls, grid


def _clip_calls(calls: np.ndarray, S: float, strikes: np.ndarray, T: float,
                r: float, q: float) -> np.ndarray:
    """Clip to the no-arbitrage band [DF*max(F-K,0), DF*F] (removes rounding noise)."""
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    return np.clip(calls, np.maximum(F - strikes, 0.0) * df, F * df)


def _warn_deprecated_grid_args(fn: str, **kwargs) -> None:
    passed = [k for k, v in kwargs.items() if v is not None]
    if passed:
        warnings.warn(
            "%s(): %s no longer used — the engine chooses its quadrature grid "
            "adaptively (Lewis contour, see module docstring)." % (fn, ", ".join(passed)),
            DeprecationWarning, stacklevel=3,
        )


def heston_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    damping_alpha: Optional[float] = None,
    n_integration_points: Optional[int] = None,
) -> float:
    """
    Price one European call under Heston (present value, i.e. discounted).

    Thin wrapper over heston_call_batch() so that there is a single pricing
    engine (Lewis contour + Black-Scholes control variate, adaptive trapezoid).

    Parameters
    ----------
    damping_alpha, n_integration_points : deprecated and ignored (they belonged
        to the former Carr-Madan quadrature); passing them emits a
        DeprecationWarning.

    Returns
    -------
    float — call price, discounted to today (do NOT discount again).
    """
    _warn_deprecated_grid_args("heston_call_price", damping_alpha=damping_alpha,
                               n_integration_points=n_integration_points)
    return float(heston_call_batch(S, np.array([float(K)]), T, r, q,
                                   kappa, theta, sigma, rho, v0)[0])


def heston_put_price(
    S: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
) -> float:
    """
    Put price via put-call parity: P = C - S*e^{-qT} + K*e^{-rT}.

    (Parity is an identity of the pricing measure, so this is exact given the
    call; tests compare it against an independently computed Lewis put.)
    """
    call = heston_call_price(S, K, T, r, q, kappa, theta, sigma, rho, v0)
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


def heston_call_batch(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    n_points: Optional[int] = None,
    v_upper: Optional[float] = None,
) -> np.ndarray:
    """
    Price European calls for one expiry, vectorised over strikes.

    Algorithm: Lewis (2001) contour integral with a Black-Scholes control
    variate, evaluated by a trapezoid rule whose truncation point and step are
    chosen per call (see the comment block above _LewisGrid): the step is
    halved until successive estimates agree to 1e-10 x DF*F. Validated against
    an independent Lewis/Gauss-Legendre reference to < 0.01 implied-vol points
    for T from 1 day to 1 year, K/F in [0.7, 1.3] (options worth at least
    1e-5 x F), on every archived fit including those with v0 at its 1e-4
    floor, the model_version 2 fits (sigma 1.6-2.0) and the slow-decay corners
    of HESTON_BOUNDS (tests/test_heston_reference.py). Typical cost: 1-3 ms
    per 25-strike slice (~10 ms at 1 day with v0 = 1e-4); the degenerate
    corner theta = v0 = 1e-4, sigma = 2, kappa = 0.1 needs up to ~1M nodes
    (~1 s per slice). A RuntimeWarning is emitted whenever the step
    refinement cannot confirm the 1e-10 target within the node cap.

    Parameters
    ----------
    strikes   : 1-D array of strike prices (any ordering)
    n_points, v_upper : deprecated and ignored (old fixed Carr-Madan grid).

    Returns
    -------
    np.ndarray, shape (M,) — discounted call prices, clipped to the
    no-arbitrage band [DF*max(F-K, 0), DF*F] (removes rounding-level noise; a
    price AT the lower bound means "zero time value to working precision",
    and implied_vol_from_price() returns NaN for it).
    """
    _warn_deprecated_grid_args("heston_call_batch", n_points=n_points, v_upper=v_upper)
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    if T <= 0:
        F = S * np.exp((r - q) * T)
        return np.maximum(F - strikes, 0.0) * np.exp(-r * T)

    calls, _ = _lewis_call_prices(
        S, strikes, T, r, q,
        lambda z: _heston_log_cf(z, T, kappa, theta, sigma, rho, v0),
        float(_expected_integrated_variance(T, kappa, theta, v0)),
    )
    return _clip_calls(calls, S, strikes, T, r, q)


def implied_vol_batch(
    prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    rights: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Black-Scholes implied vol inversion for an array of option prices.

    Parameters
    ----------
    prices  : array of market mid-prices
    rights  : array of 'C' or 'P' strings
    Returns : array of IVs; NaN where inversion fails or price is out of bounds.
    """
    ivs = np.full(len(prices), np.nan)
    for i, (p, K, right) in enumerate(zip(prices, strikes, rights)):
        ivs[i] = implied_vol_from_price(float(p), S, float(K), T, r, q, right)
    return ivs


# ── Implied Volatility Inversion ──────────────────────────────────────────────

def black_scholes_call(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price for implied vol inversion."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)


def _bs_otm_price(F: float, K: float, T: float, df: float, vol: float, right: str) -> float:
    """Black-76 price of a call ('C') or put ('P') on forward F, discounted by df."""
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    if right == "C":
        return df * (F * ndtr(d1) - K * ndtr(d2))
    return df * (K * ndtr(-d2) - F * ndtr(-d1))


_IV_LOWER = 1e-8
_IV_UPPER_MAX = 20.0        # 2000% — beyond this the inversion gives up (NaN)


def implied_vol_from_price(
    price: float, S: float, K: float, T: float, r: float, q: float,
    right: str = "C",
    tol: float = 1e-6,
) -> float:
    """
    Invert Black-Scholes to find the implied volatility for a given option price.

    The price is first mapped by put-call parity to the OUT-of-the-money option
    at this strike (call if K >= F, else put), whose price is pure time value;
    the inversion is done on that option, which keeps full relative precision
    for small prices. Brent's method brackets the root in [1e-8, 5], widening
    the upper end to 10 and 20 (2000% vol) for extreme short-dated quotes.

    Ill-posed inputs return NaN — never a silent boundary value:
      * time value <= 1e-12 * DF*F (price at, or below, intrinsic),
      * price at or above the no-arbitrage upper bound,
      * T <= 0, non-finite inputs, or no root below 2000% vol.

    Parameters
    ----------
    price : float — option price (present value)
    right : str — 'C' (call) or 'P' (put)
    tol   : float — absolute convergence tolerance on vol

    Returns
    -------
    float — implied vol (annualised decimal), or NaN.
    """
    try:
        price = float(price)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(price) and np.isfinite(S) and np.isfinite(K)) \
            or T <= 0 or K <= 0 or S <= 0:
        return float("nan")

    df = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    is_put = str(right).upper() == "P"
    if K >= F:          # OTM call
        otm, otm_right, upper = (price + df * (F - K) if is_put else price), "C", df * F
    else:               # OTM put
        otm, otm_right, upper = (price if is_put else price - df * (F - K)), "P", df * K

    if not (1e-12 * df * F < otm < upper):
        return float("nan")

    def objective(vol: float) -> float:
        return _bs_otm_price(F, K, T, df, vol, otm_right) - otm

    lo, hi = _IV_LOWER, 5.0
    if objective(lo) > 0:
        return float("nan")
    while objective(hi) < 0:
        if hi >= _IV_UPPER_MAX:
            return float("nan")
        hi = min(2.0 * hi, _IV_UPPER_MAX)
    try:
        return float(optimize.brentq(objective, lo, hi, xtol=tol, rtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


# ── Bates (1996) SVJ Model ────────────────────────────────────────────────────

def bates_characteristic_function(
    phi: complex,
    S: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
) -> complex:
    """
    Bates (1996) SVJ characteristic function E[e^{i*phi*ln(S_T)}].

    φ_Bates = φ_Heston × exp(λT(e^{iφμⱼ − ½φ²σⱼ²} − 1 − iφk̄)),  k̄ = e^{μⱼ+½σⱼ²} − 1

    The Heston CF already encodes the stochastic variance dynamics.  The Bates
    extension multiplies by the CF of a compound Poisson jump process where
    each jump is log-normally distributed: J ~ N(μⱼ, σⱼ²). The −iφλT·k̄ term
    is the risk-neutral compensator: without it φ_Bates(−i) = F·e^{λTk̄} and
    the discounted price is not a martingale.

    Parameters
    ----------
    phi     : complex — CF argument
    lam     : float  — annualised jump intensity λ (jumps per year)
    mu_j    : float  — mean log jump size (decimal; e.g. -0.05 = -5%)
    sigma_j : float  — std of log jump size (decimal; >= 0)

    All Heston parameters (kappa, theta, sigma, rho, v0) are unchanged.

    Returns
    -------
    complex — Bates CF evaluated at phi.
    """
    heston_cf = characteristic_function(
        phi, S, T, r, q, kappa, theta, sigma, rho, v0
    )
    return heston_cf * np.exp(_bates_jump_log_cf(phi, T, lam, mu_j, sigma_j))


def bates_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    damping_alpha: Optional[float] = None,
    n_integration_points: Optional[int] = None,
) -> float:
    """
    Price one European call under the Bates (1996) SVJ model.

    Thin wrapper over bates_call_batch() (same engine as Heston, Bates CF).
    damping_alpha / n_integration_points are deprecated and ignored.

    Returns
    -------
    float — call price (present value).
    """
    _warn_deprecated_grid_args("bates_call_price", damping_alpha=damping_alpha,
                               n_integration_points=n_integration_points)
    return float(bates_call_batch(S, np.array([float(K)]), T, r, q, kappa, theta,
                                  sigma, rho, v0, lam, mu_j, sigma_j)[0])


def bates_call_batch(
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    n_points: Optional[int] = None,
    v_upper: Optional[float] = None,
) -> np.ndarray:
    """
    Batch-price European calls under Bates (1996) SVJ, vectorised over strikes.

    Same engine as heston_call_batch() (Lewis contour + BS control variate,
    adaptive trapezoid) with the Bates CF. The control-variate variance and
    the initial step include the jump variance lam*T*(mu_j^2 + sigma_j^2) and
    the single-jump range |mu_j| + 10 sigma_j.

    Returns
    -------
    np.ndarray, shape (M,) — call prices, clipped to no-arbitrage bounds.
    """
    _warn_deprecated_grid_args("bates_call_batch", n_points=n_points, v_upper=v_upper)
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    if T <= 0:
        F = S * np.exp((r - q) * T)
        return np.maximum(F - strikes, 0.0) * np.exp(-r * T)

    jump_var = lam * T * (mu_j**2 + sigma_j**2)
    jump_extent = (abs(mu_j) + 10.0 * sigma_j) if lam * T > 1e-14 else 0.0
    calls, _ = _lewis_call_prices(
        S, strikes, T, r, q,
        lambda z: (_heston_log_cf(z, T, kappa, theta, sigma, rho, v0)
                   + _bates_jump_log_cf(z, T, lam, mu_j, sigma_j)),
        float(_expected_integrated_variance(T, kappa, theta, v0)) + jump_var,
        jump_extent=jump_extent,
    )
    return _clip_calls(calls, S, strikes, T, r, q)


# ── Greeks ────────────────────────────────────────────────────────────────────

def heston_greeks(
    S: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, v0: float,
    right: str = "C",
    bump: float = 1e-4,
    v0_bump: float = 1e-2,
) -> dict:
    """
    Compute option Greeks via central finite differences on the Heston price.

    "Vol" below means the instantaneous vol sigma_v = sqrt(v0) as a DECIMAL
    (0.20 = 20%); divide a per-unit-vol Greek by 100 for "per vol point".

    Greeks returned (all per ONE option on ONE unit of underlying):
      delta     : dV/dS
      gamma     : d²V/dS²
      vega      : dV/dsigma_v           ($ per 1.00 of vol; /100 per vol pt)
      theta     : V(T - 1/365) - V(T)   ($ per CALENDAR DAY of decay, < 0 long)
      theta_annual : theta * 365        (the annualised rate dV/dt)
      vanna     : d²V/(dS dsigma_v)     (per 1.00 of vol)
      vomma     : d²V/dsigma_v²         (per (1.00 of vol)²; "volga")
      vanna_v0  : d²V/(dS dv0)          (variance units)
      vomma_v0  : d²V/dv0²              (variance units)
    Chain rule: vega = 2 sigma_v V_v,  vanna = 2 sigma_v V_Sv,
                vomma = 4 v0 V_vv + 2 V_v.

    Parameters
    ----------
    bump    : relative spot bump (dS = S * bump)
    v0_bump : relative variance bump (dv = v0 * v0_bump). Every bumped price
              is evaluated on ONE shared quadrature grid, so differences are
              smooth in the parameters (no grid-switch noise) even at v0=1e-4.

    Returns
    -------
    dict with keys [delta, gamma, vega, theta, vanna, vomma,
                    theta_annual, vanna_v0, vomma_v0].

    Financial interpretation:
      Delta and Gamma drive the P&L of a delta-hedged book.
      Vega, Vanna, and Vomma are the vol risk — capturing exposure to
      vol level, skew, and convexity respectively.
    """
    is_put = right.upper() != "C"
    dS = S * bump
    dv = v0 * v0_bump
    dt = 1.0 / 365.0
    T_dn = max(T - dt, 0.0)

    def log_cf_for(T_, v0_):
        return lambda z: _heston_log_cf(z, T_, kappa, theta, sigma, rho, v0_)

    def var_for(T_, v0_):
        return float(_expected_integrated_variance(T_, kappa, theta, v0_))

    # One grid good for every bump: the finest step / longest truncation over
    # the base point and the shortest maturity used (theta bump).
    grid = None
    if T > 0:
        _, g_base = _lewis_call_prices(S, np.array([K]), T, r, q,
                                       log_cf_for(T, v0 - dv), var_for(T, v0 - dv))
        h, U = g_base.h, g_base.h * (g_base.n - 1)
        if T_dn > 0:
            _, g_dn = _lewis_call_prices(S, np.array([K]), T_dn, r, q,
                                         log_cf_for(T_dn, v0), var_for(T_dn, v0))
            h, U = min(h, g_dn.h), max(U, g_dn.h * (g_dn.n - 1))
        grid = _LewisGrid(h, min(int(np.ceil(U / h)) + 1, _LEWIS_MAX_NODES))

    def price(S_=S, T_=T, v0_=v0):
        if T_ <= 0:
            c = max(S_ - K, 0.0)
        else:
            c, _ = _lewis_call_prices(S_, np.array([K]), T_, r, q,
                                      log_cf_for(T_, v0_), var_for(T_, v0_), grid=grid)
            c = float(c[0])
        if is_put:
            return c - S_ * np.exp(-q * T_) + K * np.exp(-r * T_)
        return c

    p0 = price()
    p_up, p_dn = price(S_=S + dS), price(S_=S - dS)
    delta = (p_up - p_dn) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0 + p_dn) / (dS**2)

    p_vup, p_vdn = price(v0_=v0 + dv), price(v0_=v0 - dv)
    V_v = (p_vup - p_vdn) / (2.0 * dv)
    V_vv = (p_vup - 2.0 * p0 + p_vdn) / (dv**2)

    V_Sv = (price(S_=S + dS, v0_=v0 + dv) - price(S_=S + dS, v0_=v0 - dv)
            - price(S_=S - dS, v0_=v0 + dv) + price(S_=S - dS, v0_=v0 - dv)) / (4.0 * dS * dv)

    theta_day = price(T_=T_dn) - p0
    sig_v = np.sqrt(v0)

    return {
        "delta":        float(delta),
        "gamma":        float(gamma),
        "vega":         float(2.0 * sig_v * V_v),
        "theta":        float(theta_day),
        "vanna":        float(2.0 * sig_v * V_Sv),
        "vomma":        float(4.0 * v0 * V_vv + 2.0 * V_v),
        "theta_annual": float(theta_day * 365.0),
        "vanna_v0":     float(V_Sv),
        "vomma_v0":     float(V_vv),
    }


# ── VIX: spot index and futures under Heston / Bates ─────────────────────────

def heston_expected_variance(
    T: float,
    kappa: float,
    theta: float,
    v0: float,
) -> float:
    """
    Compute E_t[v_{t+T}] under Heston — expected instantaneous variance at T.

    Under Heston: E[v_T] = theta + (v0 - theta) * exp(-kappa * T)
    (the mean of the CIR transition law).

    Parameters
    ----------
    T : float — horizon in years

    Returns
    -------
    float — expected variance E[v_T]

    Financial note:
      This is NOT the VIX. VIX is the annualised sqrt of the expected
      integrated variance over the next 30 days, not the point-in-time variance.
    """
    return theta + (v0 - theta) * np.exp(-kappa * T)


def heston_integrated_variance(
    T,
    kappa: float,
    theta: float,
    v0: float,
):
    """
    Compute E_t[integral_t^{t+T} v_s ds] / T under Heston (vectorised over T).

    This is the expected average variance over [t, t+T]:
      theta + (v0 - theta) * (1 - e^{-kappa*T}) / (kappa*T)
    with the exact limit v0 as kappa*T -> 0 (computed via expm1, so there is
    no cancellation for small kappa*T).

    For T = 30/365 this is the model's spot VIX^2 (decimal variance).

    Returns
    -------
    float (or ndarray for array T) — expected average variance
    """
    out = theta + (v0 - theta) * _one_minus_exp_over(kappa * np.asarray(T, dtype=float))
    return float(out) if np.ndim(out) == 0 else out


def heston_vix_term_index(kappa: float, theta: float, sigma: float, v0: float, T):
    """
    Heston-implied level of a spot-starting VIX-style index with horizon T.

    I(T) = 100 * sqrt( (1/T) E_0[ int_0^T v_s ds ] )
         = 100 * sqrt( theta + (v0 - theta) * (1 - e^{-kappa T}) / (kappa T) )

    This is what the Cboe constant-maturity indices measure: ^VIX9D, ^VIX,
    ^VIX3M, ^VIX6M are this quantity with T = 9, 30, 93, 184 calendar days
    (config.VIX_INDEX_TENOR_DAYS). It is NOT a VIX future: a future expiring
    at T pays the 30-day vol over the FORWARD window [T, T+30d]
    (see heston_vix_future_exact).

    Parameters
    ----------
    sigma : vol-of-vol — accepted for signature symmetry with the futures
            functions; the spot index does not depend on it under Heston.
    T     : horizon(s) in years (scalar or array). T -> 0 gives 100*sqrt(v0),
            kappa -> 0 gives 100*sqrt(v0), kappa -> inf gives 100*sqrt(theta).

    Returns
    -------
    float or ndarray — index level in vol points (e.g. 20.5 means 20.5%).
    """
    avg = theta + (v0 - theta) * _one_minus_exp_over(kappa * np.asarray(T, dtype=float))
    out = 100.0 * np.sqrt(np.maximum(avg, 0.0))
    return float(out) if np.ndim(out) == 0 else out


def heston_vix_index(
    kappa: float,
    theta: float,
    v0: float,
    T: float = _VIX_TAU,
) -> float:
    """
    Heston model-implied spot VIX: heston_vix_term_index(..., T) with the
    default T = 30/365 (the ^VIX definition).

    Returns
    -------
    float — VIX level (e.g. 20.5 means 20.5%)
    """
    return heston_vix_term_index(kappa, theta, 0.0, v0, T)


def _bates_jump_vix_variance(lam: float, mu_j: float, sigma_j: float) -> float:
    """
    Jump contribution to the model-free (log-contract) VIX^2 under Bates:
      2 * lam * (kbar - mu_j),  kbar = e^{mu_j + sigma_j^2/2} - 1.
    Derivation: -(2/tau) E[ln(S_tau/F)] with d ln S = (... - lam kbar) dt + dJ
    gives (1/tau) E[int v] + 2 lam (kbar - mu_j). It is >= 0 and differs from
    the jumps' quadratic variation lam*(mu_j^2 + sigma_j^2) (Carr & Wu 2009).
    """
    return 2.0 * lam * (np.expm1(mu_j + 0.5 * sigma_j**2) - mu_j)


def bates_vix_term_index(kappa: float, theta: float, sigma: float, v0: float,
                         lam: float, mu_j: float, sigma_j: float, T):
    """
    Bates-implied level of a spot-starting VIX-style index with horizon T:
      100 * sqrt( heston average variance over [0,T] + 2 lam (kbar - mu_j) ).

    The jump term is constant in T because the jump intensity is constant.
    lam = 0 recovers heston_vix_term_index exactly.
    """
    avg = theta + (v0 - theta) * _one_minus_exp_over(kappa * np.asarray(T, dtype=float))
    avg = avg + _bates_jump_vix_variance(lam, mu_j, sigma_j)
    out = 100.0 * np.sqrt(np.maximum(avg, 0.0))
    return float(out) if np.ndim(out) == 0 else out


# Trapezoid grid in w = log(t * sqrt(m)) for E[sqrt(Y)] (see _expected_sqrt_affine_cir)
_VIXF_W = np.arange(-32.0, 32.0 + 1e-9, 0.1)


def _expected_sqrt_affine_cir(a, b, kappa, theta, sigma, v0, T):
    """
    E[sqrt(a + b v_T)] for CIR v_T (started at v0), vectorised over T.

    Uses sqrt(y) = (1/sqrt(pi)) int_0^inf (1 - e^{-t^2 y}) / t^2 dt and the
    closed-form CIR Laplace transform
      E[e^{-p v_T}] = (1 + 2cp)^{-2 kappa theta/sigma^2} exp(-v0 e^{-kappa T} p/(1 + 2cp)),
      c = sigma^2 (1 - e^{-kappa T}) / (4 kappa),
    rewritten so sigma never appears in a denominator (exact sigma -> 0 limit).
    With t = e^w / sqrt(m), m = E[a + b v_T], the integrand decays like e^{w}
    as w -> -inf and like e^{-w} as w -> +inf and is analytic in the strip
    |Im w| < pi/2, so the trapezoid rule with step 0.1 on [-32, 32] converges
    to machine precision; the two exponential tails are added analytically.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    ekT = np.exp(-kappa * T)
    one_m_ekT = -np.expm1(-kappa * T)
    c = sigma**2 * T * _one_minus_exp_over(kappa * T) / 4.0      # (n_T,)
    m = a + b * (theta * one_m_ekT + v0 * ekT)                   # E[Y]
    m = np.maximum(m, 1e-300)

    w = _VIXF_W[None, :]
    s = np.exp(2.0 * w) / m[:, None]                             # Laplace argument
    bs = b * s
    y = 2.0 * c[:, None] * bs
    l1p = np.where(y < 1e-8, 1.0 - 0.5 * y, np.log1p(y) / np.where(y < 1e-8, 1.0, y))
    log_L = -(a * s
              + theta * one_m_ekT[:, None] * bs * l1p
              + v0 * ekT[:, None] * bs / (1.0 + y))
    one_m_L = -np.expm1(log_L)
    integrand = one_m_L * np.exp(-w)
    hw = _VIXF_W[1] - _VIXF_W[0]
    body = hw * (integrand.sum(axis=1) - 0.5 * (integrand[:, 0] + integrand[:, -1]))
    tails = np.exp(_VIXF_W[0]) + one_m_L[:, -1] * np.exp(-_VIXF_W[-1])
    return np.sqrt(m / np.pi) * (body + tails)


def heston_vix_future_exact(kappa: float, theta: float, sigma: float, v0: float,
                            T, tau: float = _VIX_TAU):
    """
    Exact Heston VIX futures price E_0[VIX_T] in vol points (vectorised over T).

    VIX_T^2 = a + b v_T with b = (1 - e^{-kappa tau})/(kappa tau),
    a = theta (1 - b), and v_T | v0 the CIR transition law (scaled non-central
    chi-square). E[sqrt(a + b v_T)] is evaluated from the closed-form CIR
    Laplace transform (see _expected_sqrt_affine_cir); no sampling, no
    approximation beyond ~1e-12 quadrature error.

    Properties (tested): exact <= sqrt(E[VIX_T^2]) (Jensen) with equality as
    sigma -> 0; T -> 0 gives the spot index heston_vix_term_index(tau); it
    agrees with exact CIR sampling within Monte Carlo error.

    Parameters
    ----------
    T   : futures expiry/expiries in years (scalar or array)
    tau : VIX averaging window (30/365)

    Returns
    -------
    float or ndarray — VIX futures price(s), e.g. 20.5
    """
    b = float(_one_minus_exp_over(kappa * tau))
    a = theta * (1.0 - b)
    out = 100.0 * _expected_sqrt_affine_cir(a, b, kappa, theta, sigma, v0, T)
    return float(out[0]) if np.ndim(T) == 0 else out


def heston_vix_futures_upper_bound(
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    expiry_times: np.ndarray,
    vix_window: float = _VIX_TAU,
) -> np.ndarray:
    """
    sqrt(E[VIX_T^2]) — an UPPER BOUND on the VIX futures price (Jensen).

      sqrt(E[VIX^2(T)]) = 100 * sqrt(theta + (E[v_T] - theta) * b),
      E[v_T] = theta + (v0 - theta) e^{-kappa T},  b = (1 - e^{-kappa tau})/(kappa tau).

    This is the forward-starting 30-day variance-swap rate. It ignores the
    concavity of the square root and does not depend on sigma; the true futures
    price heston_vix_future_exact() is lower by an amount that grows with
    vol-of-vol and expiry (0.15-1.9 VIX points across the 38 archived joint
    fits at 30-182 day expiries; median 0.6 at 30d, 1.4 at 182d).
    Kept, under an explicit name, for the variance-swap interpretation.
    """
    T = np.asarray(expiry_times, dtype=float)
    ev_T = heston_expected_variance(T, kappa, theta, v0)
    b = _one_minus_exp_over(kappa * vix_window)
    avg_var = theta + (ev_T - theta) * b
    return 100.0 * np.sqrt(np.maximum(avg_var, 1e-8))


def heston_vix_futures_curve(
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    expiry_times: np.ndarray,
    vix_window: float = _VIX_TAU,
) -> np.ndarray:
    """
    Heston-implied VIX FUTURES term structure (exact), in VIX points.

    For each futures expiry T_i returns E[VIX_{T_i}] from
    heston_vix_future_exact(). Before C22 this returned
    sqrt(E[VIX^2(T_i)]), which is an upper bound (still available as
    heston_vix_futures_upper_bound) and ignores sigma entirely.

    Note: these are FUTURES. The Cboe spot indices ^VIX9D/^VIX/^VIX3M/^VIX6M
    are a different instrument — use heston_vix_term_index() for them.

    Parameters
    ----------
    expiry_times : np.ndarray — VIX futures expiry times in years [T_1, ..., T_n]
    vix_window   : float — the VIX integration window (30/365 by default)

    Returns
    -------
    np.ndarray — Heston-implied VIX futures prices (in VIX points, e.g. 20.5)
    """
    T = np.atleast_1d(np.asarray(expiry_times, dtype=float))
    if T.size == 0:
        return np.zeros(0)
    return heston_vix_future_exact(kappa, theta, sigma, v0, T, tau=vix_window)


def bates_vix_future_exact(kappa: float, theta: float, sigma: float, v0: float,
                           lam: float, mu_j: float, sigma_j: float,
                           T, tau: float = _VIX_TAU):
    """
    Exact Bates VIX futures price E[VIX_T] in vol points (vectorised over T).

    With constant jump intensity the jump part of VIX_T^2 is deterministic,
    VIX_T^2 = a + 2 lam (kbar - mu_j) + b v_T, so the Heston Laplace-transform
    evaluation applies with a shifted constant. lam = 0 recovers
    heston_vix_future_exact.
    """
    b = float(_one_minus_exp_over(kappa * tau))
    a = theta * (1.0 - b) + _bates_jump_vix_variance(lam, mu_j, sigma_j)
    out = 100.0 * _expected_sqrt_affine_cir(a, b, kappa, theta, sigma, v0, T)
    return float(out[0]) if np.ndim(T) == 0 else out


def bates_vix_futures_curve(kappa: float, theta: float, sigma: float, v0: float,
                            lam: float, mu_j: float, sigma_j: float,
                            expiry_times: np.ndarray,
                            vix_window: float = _VIX_TAU) -> np.ndarray:
    """Bates-implied VIX futures term structure (exact), in VIX points."""
    T = np.atleast_1d(np.asarray(expiry_times, dtype=float))
    if T.size == 0:
        return np.zeros(0)
    return bates_vix_future_exact(kappa, theta, sigma, v0, lam, mu_j, sigma_j,
                                  T, tau=vix_window)


def compute_heston_vix_gap(
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    market_vix_futures: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the Heston-VIX futures gap: model price vs market price.

    Model prices are exact Heston VIX futures (heston_vix_futures_curve). The
    input must be VIX FUTURES quotes; do not pass the spot term-structure
    indices (^VIX3M etc.) here — compare those with heston_vix_term_index().

    Parameters
    ----------
    market_vix_futures : DataFrame with columns [expiry_month, close, days_to_expiry]
                         as returned by database.get_vix_futures_curve()

    Returns
    -------
    DataFrame with columns:
      [expiry_month, days_to_expiry, market_price, model_price,
       gap, gap_pct, abs_gap]

    where gap = model_price - market_price.
    A positive gap means Heston over-prices VIX futures (predicts too high vol).
    A negative gap means Heston under-prices (predicts too low vol).
    """
    df = market_vix_futures.copy()
    df = df[df["days_to_expiry"] > 0].copy()

    expiry_times = df["days_to_expiry"].values / 365.0
    model_prices = heston_vix_futures_curve(kappa, theta, sigma, v0, expiry_times)

    df["model_price"] = model_prices
    df.rename(columns={"close": "market_price"}, inplace=True)
    df["gap"]      = df["model_price"] - df["market_price"]
    df["abs_gap"]  = df["gap"].abs()
    df["gap_pct"]  = (df["gap"] / df["market_price"] * 100.0).round(2)

    if len(df):
        rmse = np.sqrt((df["gap"]**2).mean())
        logger.info(
            "Heston VIX futures gap: RMSE=%.2f pts, max|gap|=%.2f pts, mean|gap|=%.2f pts",
            rmse, df["abs_gap"].max(), df["abs_gap"].mean()
        )
    return df


# ── Calibration ───────────────────────────────────────────────────────────────

_UNPRICEABLE_PENALTY = 1.0   # squared-IV penalty for an option whose model IV is NaN


def _surface_model_ivs(
    params: np.ndarray,
    surface: pd.DataFrame,
    S: float,
    r: float,
    q: float,
) -> np.ndarray:
    """
    Model implied vols for every row of `surface` (positional, NaN where the
    model price has no implied vol). One heston_call_batch call per expiry.
    """
    kappa, theta, sigma, rho, v0 = params
    strikes = surface["strike"].to_numpy(dtype=float)
    expiries = surface["time_to_expiry"].to_numpy(dtype=float)
    rights = (surface["right"].astype(str).str.upper().to_numpy()
              if "right" in surface.columns else np.full(len(surface), "C"))
    ivs = np.full(len(surface), np.nan)
    for T in np.unique(expiries[expiries > 0]):
        idx = np.flatnonzero((expiries == T) & (strikes > 0))
        if idx.size == 0:
            continue
        calls = heston_call_batch(S, strikes[idx], T, r, q, kappa, theta, sigma, rho, v0)
        for j, c in zip(idx, calls):
            K = strikes[j]
            px = c - S * np.exp(-q * T) + K * np.exp(-r * T) if rights[j] == "P" else c
            ivs[j] = implied_vol_from_price(px, S, K, T, r, q, rights[j])
    return ivs


def _implied_vol_surface_loss(
    params: np.ndarray,
    market_surface: pd.DataFrame,
    S: float,
    r: float,
    q: float,
    weights: Optional[np.ndarray] = None,
    feller_penalty: float = 0.0,
) -> float:
    """
    Loss function for Heston calibration to SPX implied vol surface.

    L = (1/N) * sum_i w_i * (sigma_model_i - sigma_market_i)^2
        + feller_penalty * max(sigma^2 - 2 kappa theta, 0)

    where i indexes each (K_i, T_i, right_i) row of market_surface (rows with
    T <= 0, K <= 0 or implied_vol <= 0 contribute nothing but count in N).
    `weights` is aligned POSITIONALLY with the rows of market_surface (the
    index labels are ignored). A row whose model price has no implied vol
    (NaN) adds a flat penalty of 1.0 (= 100 vol pts squared) instead of its
    weighted error.

    The Feller condition is not required for pricing (the CF is valid when it
    fails; v merely touches 0), so by default it is not imposed. Pass
    feller_penalty > 0 for a soft, continuous exterior penalty (the joint
    calibrator uses 50).

    Parameters
    ----------
    params : np.ndarray — [kappa, theta, sigma, rho, v0]
    market_surface : DataFrame with [strike, time_to_expiry, right, implied_vol, ...]
    S : float — current spot
    r, q : float — risk-free rate and dividend yield
    weights : np.ndarray or None — per-observation weights (positional)

    Returns
    -------
    float — weighted MSE loss (vol decimal squared)
    """
    kappa, theta, sigma, rho, v0 = params
    surface = market_surface.reset_index(drop=True)
    n = len(surface)
    if n == 0:
        return 0.0
    if weights is not None and len(weights) != n:
        raise ValueError("weights has length %d but the surface has %d rows"
                         % (len(weights), n))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)

    mkt = surface["implied_vol"].to_numpy(dtype=float)
    valid = ((surface["time_to_expiry"].to_numpy(dtype=float) > 0)
             & (surface["strike"].to_numpy(dtype=float) > 0) & (mkt > 0))
    model = _surface_model_ivs(params, surface[valid].reset_index(drop=True), S, r, q)
    err2 = np.where(np.isnan(model), _UNPRICEABLE_PENALTY,
                    w[valid] * (model - mkt[valid])**2)
    loss = float(np.sum(err2)) / n
    if feller_penalty > 0:
        loss += feller_penalty * max(sigma**2 - 2.0 * kappa * theta, 0.0)
    return loss


def calibrate_to_spx(
    market_surface: pd.DataFrame,
    S: float,
    r: float = _DEFAULT_RATE,
    q: float = _DEFAULT_DIV,
    method: str = "differential_evolution",
    seed: int = RANDOM_SEED,
    feller_penalty: float = 0.0,
) -> dict:
    """
    Calibrate Heston parameters to the SPX implied volatility surface.

    Minimises the weighted MSE between Heston-implied vols and market vols
    across all (strike, maturity) pairs in market_surface.

    Parameters
    ----------
    market_surface : DataFrame with [strike, time_to_expiry, right, implied_vol].
                     Should be filtered to liquid strikes (e.g. delta in [0.1, 0.9]).
    S   : float — current SPX spot price
    r   : float — risk-free rate
    q   : float — continuous dividend yield
    method : str — 'differential_evolution' (global, slow) or
                   'minimize' (local, fast, needs good initial guess)
    seed : int — random seed for reproducibility
    feller_penalty : float — weight of the soft Feller penalty (0 = no Feller
                   constraint; see _implied_vol_surface_loss)

    Returns
    -------
    dict with keys:
      params : dict[str → float] — calibrated {kappa, theta, sigma, rho, v0}
      loss   : float — final calibration loss (weighted MSE, vol decimal^2)
      success : bool — whether optimisation converged
      n_obs  : int — number of market observations used
      n_iv_failures : int — options whose model IV is NaN at the optimum

    Calibration note:
      Each loss evaluation prices one vectorised slice per expiry and inverts
      every option's IV, so differential evolution over a few dozen options
      takes minutes, not seconds. Bid/ask weights (1/spread, normalised to
      mean 1) are aligned with the filtered surface positionally.
    """
    # Filter to liquid options; positional index from here on
    surface = market_surface.copy()
    surface = surface[
        (surface["implied_vol"] > 0.01) &   # > 1% vol
        (surface["implied_vol"] < 3.0) &    # < 300% vol (remove illiquid)
        (surface["time_to_expiry"] > 0.02)  # > 1 week
    ].reset_index(drop=True)

    if len(surface) < 5:
        logger.warning("Only %d market observations for calibration — may be unreliable", len(surface))

    # Compute weights: inversely proportional to bid-ask spread
    if "bid" in surface.columns and "ask" in surface.columns:
        spread = (surface["ask"] - surface["bid"]).replace(0, np.nan)
        weights = (1.0 / spread).fillna(1.0).to_numpy(dtype=float)
        weights = weights / weights.sum() * len(weights)
    else:
        weights = None

    bounds_list = [
        HESTON_BOUNDS["kappa"],
        HESTON_BOUNDS["theta"],
        HESTON_BOUNDS["sigma"],
        HESTON_BOUNDS["rho"],
        HESTON_BOUNDS["v0"],
    ]

    def loss_fn(p):
        return _implied_vol_surface_loss(p, surface, S, r, q, weights,
                                         feller_penalty=feller_penalty)

    if method == "differential_evolution":
        logger.info("Running differential evolution calibration on %d options ...", len(surface))
        result = optimize.differential_evolution(
            loss_fn,
            bounds=bounds_list,
            seed=seed,
            maxiter=1000,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.9,
            popsize=15,
            workers=1,   # parallelise externally, not here
            polish=True,
            disp=False,
        )
    else:
        x0 = [
            HESTON_DEFAULTS["kappa"],
            HESTON_DEFAULTS["theta"],
            HESTON_DEFAULTS["sigma"],
            HESTON_DEFAULTS["rho"],
            HESTON_DEFAULTS["v0"],
        ]
        result = optimize.minimize(
            loss_fn,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": 2000, "ftol": 1e-12},
        )

    kappa, theta, sigma, rho, v0 = result.x
    params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}

    n_fail = 0
    if len(surface):
        valid = (surface["strike"] > 0) & (surface["implied_vol"] > 0)
        model_ivs = _surface_model_ivs(result.x, surface[valid].reset_index(drop=True), S, r, q)
        n_fail = int(np.isnan(model_ivs).sum())
        if n_fail:
            logger.warning("%d of %d options have no model implied vol at the optimum "
                           "(each adds a 1.0 penalty to the loss)", n_fail, len(model_ivs))

    logger.info(
        "Calibration complete. Loss=%.6f, success=%s. "
        "kappa=%.3f theta=%.4f sigma=%.3f rho=%.3f v0=%.4f",
        result.fun, result.success, kappa, theta, sigma, rho, v0
    )

    return {
        "params":  params,
        "loss":    float(result.fun),
        "success": bool(result.success),
        "n_obs":   len(surface),
        "n_iv_failures": n_fail,
    }


# ── Monte Carlo Pricing ───────────────────────────────────────────────────────

_MC_SCHEMES = ("qe", "full_truncation")
_QE_PSI_CRIT = 1.5


def _qe_variance_step(v, zv, kappa, theta, sigma, dt):
    """
    One Andersen (2008) Quadratic-Exponential step for the CIR variance.

    Moment-matches the exact conditional mean m and variance s^2 of v_{t+dt}:
    psi = s^2/m^2 <= 1.5 -> v' = a (b + Z)^2 (quadratic branch);
    psi > 1.5 -> v' = 0 w.p. p, else Exp(beta) (exponential branch), drawn by
    inverting U = Phi(Z) so that antithetic Z -> -Z maps U -> 1-U.
    Returns (v', branch data needed for the martingale correction).
    """
    one_m_e = -np.expm1(-kappa * dt)
    e = 1.0 - one_m_e
    om_over_k = dt * float(_one_minus_exp_over(kappa * dt))   # (1 - e)/kappa, kappa->0 safe
    m = theta + (v - theta) * e
    s2 = (v * sigma**2 * e * om_over_k
          + theta * sigma**2 * one_m_e * om_over_k / 2.0)
    m = np.maximum(m, 1e-300)
    psi = np.maximum(s2 / (m * m), 1e-200)
    quad = psi <= _QE_PSI_CRIT

    v_new = np.empty_like(v)
    # quadratic branch
    inv = 2.0 / psi[quad]
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
    a = m[quad] / (1.0 + b2)
    v_new[quad] = a * (np.sqrt(b2) + zv[quad])**2
    # exponential branch
    ex = ~quad
    p = (psi[ex] - 1.0) / (psi[ex] + 1.0)
    beta = (1.0 - p) / m[ex]
    u = ndtr(zv[ex])
    with np.errstate(divide="ignore"):
        tail = np.log((1.0 - p) / ndtr(-zv[ex])) / beta
    v_new[ex] = np.where(u <= p, 0.0, tail)
    return v_new, quad, a, b2, p, beta


def heston_monte_carlo(
    S: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    v0: float,
    n_paths: int = MC_PATHS,
    n_steps_per_year: int = MC_STEPS_PER_YEAR,
    seed: int = RANDOM_SEED,
    cache_key: Optional[str] = None,
    scheme: str = "qe",
    antithetic: bool = MC_ANTITHETIC,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Heston paths.

    Schemes
    -------
    'qe' (default) : Andersen (2008) Quadratic-Exponential scheme for v with
        the central (gamma1 = gamma2 = 1/2) log-spot discretisation and
        Andersen's martingale correction, so E[S_{t+dt} | S_t, v_t] =
        S_t e^{(r-q)dt} holds exactly per step. Accurate at daily steps even
        when the Feller condition fails or binds (2 kappa theta <= sigma^2),
        which is where most production fits sit.
    'full_truncation' : Euler with full truncation (Lord, Koekkoek & van Dijk
        2010): the raw state v~ is carried unfloored and only v~^+ enters the
        drift and diffusion; the returned paths_v is max(v~, 0). Simpler, but
        with a visible discretisation bias at Feller-binding parameters.
    (The pre-C22 code floored the stored state — the 'absorption' scheme in
    Lord et al.'s taxonomy — which over-prices OTM options several-fold when
    Feller binds. Neither scheme here is Broadie-Kaya exact simulation.)

    Antithetic variates: paths [0, n/2) use normals Z, paths [n/2, n) use -Z
    (same layout mc_option_price() assumes for its pair-mean standard error).
    Normals are drawn per time step; nothing of size n_paths x n_steps x 2 is
    allocated beyond the two returned path arrays.

    Parameters
    ----------
    n_paths : int — number of Monte Carlo paths (rounded down to even if antithetic)
    n_steps_per_year : int — time steps per year (252 = daily)
    seed : int — random seed for reproducibility
    cache_key : str or None — if set, results are cached to MC_CACHE_DIR under
                '<cache_key>__<scheme>[_anti].pkl' (the scheme is part of the
                file name so a cache from another scheme is never reused).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
      paths_S : shape (n_paths, n_steps+1) — spot price paths
      paths_v : shape (n_paths, n_steps+1) — variance paths (>= 0)
    """
    if scheme not in _MC_SCHEMES:
        raise ValueError("scheme must be one of %s, got %r" % (_MC_SCHEMES, scheme))

    cache_file = None
    if cache_key is not None:
        cache_file = MC_CACHE_DIR / ("%s__%s%s.pkl" % (cache_key, scheme,
                                                        "_anti" if antithetic else ""))
        if cache_file.exists():
            logger.info("Loading MC paths from cache: %s", cache_file)
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    n_steps = max(int(T * n_steps_per_year), 1)
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    half = n_paths // 2 if antithetic else n_paths
    n_paths = 2 * half if antithetic else n_paths

    paths_S = np.empty((n_paths, n_steps + 1))
    paths_v = np.empty((n_paths, n_steps + 1))
    paths_S[:, 0] = S
    paths_v[:, 0] = v0
    log_s = np.full(n_paths, np.log(S))
    drift = (r - q) * dt

    def normals():
        z = rng.standard_normal((half, 2))
        return np.concatenate([z, -z]) if antithetic else z

    if scheme == "full_truncation":
        v_raw = np.full(n_paths, float(v0))
        sq_dt = np.sqrt(dt)
        rho_c = np.sqrt(max(1.0 - rho * rho, 0.0))
        for i in range(n_steps):
            z = normals()
            vp = np.maximum(v_raw, 0.0)
            sv = np.sqrt(vp) * sq_dt
            log_s += drift - 0.5 * vp * dt + sv * z[:, 0]
            v_raw = v_raw + kappa * (theta - vp) * dt + sigma * sv * (rho * z[:, 0] + rho_c * z[:, 1])
            paths_S[:, i + 1] = np.exp(log_s)
            paths_v[:, i + 1] = np.maximum(v_raw, 0.0)
    else:
        sig = max(float(sigma), 1e-10)
        g1 = g2 = 0.5
        K0 = -rho * kappa * theta * dt / sig
        K1 = g1 * dt * (kappa * rho / sig - 0.5) - rho / sig
        K2 = g2 * dt * (kappa * rho / sig - 0.5) + rho / sig
        K3 = g1 * dt * (1.0 - rho * rho)
        K4 = g2 * dt * (1.0 - rho * rho)
        A = K2 + 0.5 * K4
        v = np.full(n_paths, float(v0))
        for i in range(n_steps):
            z = normals()
            v_new, quad, a, b2, p, beta = _qe_variance_step(v, z[:, 1], kappa, theta, sig, dt)
            # Martingale correction: K0* = -ln E[e^{A v'} | v] - (K1 + K3/2) v
            log_M = np.full(n_paths, np.nan)
            ok_q = A < 1.0 / (2.0 * a)
            qi = np.flatnonzero(quad)[ok_q]
            log_M[qi] = (A * b2[ok_q] * a[ok_q] / (1.0 - 2.0 * A * a[ok_q])
                         - 0.5 * np.log(1.0 - 2.0 * A * a[ok_q]))
            ok_e = A < beta
            ei = np.flatnonzero(~quad)[ok_e]
            log_M[ei] = np.log(p[ok_e] + beta[ok_e] * (1.0 - p[ok_e]) / (beta[ok_e] - A))
            k0 = np.where(np.isnan(log_M), K0, -log_M - (K1 + 0.5 * K3) * v)
            log_s += (drift + k0 + K1 * v + K2 * v_new
                      + np.sqrt(np.maximum(K3 * v + K4 * v_new, 0.0)) * z[:, 0])
            v = v_new
            paths_S[:, i + 1] = np.exp(log_s)
            paths_v[:, i + 1] = v

    if cache_file is not None:
        with open(cache_file, "wb") as f:
            pickle.dump((paths_S, paths_v), f)
        logger.info("Cached MC paths to %s", cache_file)

    return paths_S, paths_v


def mc_option_price(
    paths_S: np.ndarray,
    K: float,
    T: float,
    r: float,
    right: str = "C",
    antithetic: Optional[bool] = None,
) -> Tuple[float, float]:
    """
    Price a European option from pre-simulated paths.

    Standard error: with antithetic paths (layout of heston_monte_carlo:
    path i and path i + n/2 are a pair) the two halves are NOT independent,
    so the SE is computed from the n/2 i.i.d. pair means. That estimator is
    also valid when the halves are independent, so it is the default whenever
    n is even (antithetic=None). antithetic=False forces the naive i.i.d. SE.

    Returns
    -------
    Tuple[float, float] : (price, standard_error)
    """
    S_T = paths_S[:, -1]
    if right.upper() == "C":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)

    discounted = np.exp(-r * T) * payoffs
    n = len(discounted)
    pair = (n % 2 == 0 and n >= 4) if antithetic is None else bool(antithetic)
    if pair:
        if n % 2:
            raise ValueError("antithetic=True needs an even number of paths")
        half = n // 2
        pm = 0.5 * (discounted[:half] + discounted[half:])
        return float(pm.mean()), float(pm.std(ddof=1) / np.sqrt(half))
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n))


# ── HestonModel Class ─────────────────────────────────────────────────────────

class HestonModel:
    """
    Encapsulates a calibrated Heston model instance.

    Workflow:
      1. model = HestonModel()
      2. result = model.calibrate(surface_df, S, r, q)
      3. gaps = model.compute_vix_gap(market_vix_futures_df)
      4. greeks = model.greeks(S, K, T, r, q)
      5. model.save(path) / HestonModel.load(path)

    Attributes
    ----------
    params : dict — calibrated {kappa, theta, sigma, rho, v0}
    calib_result : dict — full calibration output including loss and n_obs
    is_calibrated : bool
    """

    def __init__(self):
        self.params: dict = dict(HESTON_DEFAULTS)
        self.calib_result: Optional[dict] = None
        self.is_calibrated: bool = False
        self._S: Optional[float] = None
        self._r: float = _DEFAULT_RATE
        self._q: float = _DEFAULT_DIV

    def calibrate(
        self,
        market_surface: pd.DataFrame,
        S: float,
        r: float = _DEFAULT_RATE,
        q: float = _DEFAULT_DIV,
        method: str = "differential_evolution",
        feller_penalty: float = 0.0,
    ) -> dict:
        """
        Calibrate to SPX implied vol surface and store results.

        Returns calibration result dict (see calibrate_to_spx).
        """
        self._S = S
        self._r = r
        self._q = q

        result = calibrate_to_spx(market_surface, S, r, q, method=method,
                                  feller_penalty=feller_penalty)
        self.params = result["params"]
        self.calib_result = result
        self.is_calibrated = True

        return result

    def _spot(self, S: Optional[float]) -> float:
        return self._S if S is None else S

    def price(
        self,
        K: float,
        T: float,
        right: str = "C",
        S: Optional[float] = None,
    ) -> float:
        """Price a European option under calibrated Heston (present value)."""
        self._check_calibrated()
        s = self._spot(S)
        p = self.params
        if right.upper() == "C":
            return heston_call_price(s, K, T, self._r, self._q,
                                     p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])
        else:
            return heston_put_price(s, K, T, self._r, self._q,
                                    p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])

    def implied_vol(
        self, K: float, T: float, right: str = "C", S: Optional[float] = None
    ) -> float:
        """Model-implied vol for a given strike/maturity (NaN if ill-posed)."""
        price = self.price(K, T, right, S)
        s = self._spot(S)
        return implied_vol_from_price(price, s, K, T, self._r, self._q, right=right)

    def greeks(
        self, K: float, T: float, right: str = "C", S: Optional[float] = None
    ) -> dict:
        """Compute all Greeks for a given option (units: see heston_greeks)."""
        self._check_calibrated()
        s = self._spot(S)
        p = self.params
        return heston_greeks(s, K, T, self._r, self._q,
                             p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"],
                             right=right)

    def compute_vix_gap(self, market_vix_futures: pd.DataFrame) -> pd.DataFrame:
        """
        Quantify the Heston-VIX futures mismatch (exact model futures vs market).

        Returns a DataFrame showing model vs market VIX futures prices.
        """
        self._check_calibrated()
        p = self.params
        return compute_heston_vix_gap(
            p["kappa"], p["theta"], p["sigma"], p["v0"],
            market_vix_futures,
        )

    def vix_index(self) -> float:
        """Return current Heston-implied (30-day spot) VIX level."""
        self._check_calibrated()
        p = self.params
        return heston_vix_index(p["kappa"], p["theta"], p["v0"])

    def smile_surface(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
    ) -> pd.DataFrame:
        """
        Compute the full Heston implied vol smile surface.

        Uses the calibrated rate/dividend (self._r, self._q), consistent with
        .price(), .implied_vol() and .greeks(). (A per-call rate override was
        removed: it was never threaded into the pricer, and Heston implied vol
        is essentially rate-invariant in forward-moneyness terms anyway.)

        Parameters
        ----------
        strikes    : 1D array of strike prices
        maturities : 1D array of maturities in years

        Returns
        -------
        DataFrame with columns [strike, maturity, moneyness, log_moneyness,
        implied_vol_c, implied_vol_p]
        """
        self._check_calibrated()
        rows = []
        for T in maturities:
            for K in strikes:
                iv_c = self.implied_vol(K, T, "C", S)
                iv_p = self.implied_vol(K, T, "P", S)
                rows.append({
                    "strike":         K,
                    "maturity":       T,
                    "moneyness":      K / S,
                    "log_moneyness":  np.log(K / S),
                    "implied_vol_c":  iv_c,
                    "implied_vol_p":  iv_p,
                })
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Persist calibrated model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "params":       self.params,
                "calib_result": self.calib_result,
                "S":            self._S,
                "r":            self._r,
                "q":            self._q,
            }, f)
        logger.info("Heston model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "HestonModel":
        """Load a previously calibrated model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        model = cls()
        model.params = state["params"]
        model.calib_result = state["calib_result"]
        model._S = state["S"]
        model._r = state["r"]
        model._q = state["q"]
        model.is_calibrated = True
        logger.info("Heston model loaded from %s", path)
        return model

    def _check_calibrated(self):
        if not self.is_calibrated:
            raise RuntimeError(
                "HestonModel is not yet calibrated. "
                "Call model.calibrate(surface_df, S) first."
            )

    def __repr__(self) -> str:
        if self.is_calibrated:
            p = self.params
            return (
                f"HestonModel(kappa={p['kappa']:.3f}, theta={p['theta']:.4f}, "
                f"sigma={p['sigma']:.3f}, rho={p['rho']:.3f}, v0={p['v0']:.4f})"
            )
        return "HestonModel(not calibrated)"
