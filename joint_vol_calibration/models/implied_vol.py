"""
implied_vol.py — torch-free Black-Scholes / Black-76 pricing, Greeks and a
robust implied-volatility solver.

Everything here is plain numpy/scipy so that risk tooling (the Greeks monitor,
the delta hedger, the backtest engine) can price and invert options without
importing the neural-network stack.

Contents
--------
  Prices      black_price            Black-76 (forward) price, call or put
              bs_price               Black-Scholes price with dividend yield q
              fourier_otm_price      OTM forward price from any characteristic
                                     function of ln(S_T/F) (Lewis 2001)

  Greeks      bs_d1d2, bs_delta, bs_gamma, bs_vega, bs_vomma, bs_vanna
              (spot Greeks with continuous dividend yield q; per 1.00 of vol)

  Inversion   implied_vol_black      vectorised, bracketed, safeguarded Newton
              implied_vol_bs         same, from Black-Scholes inputs (S, r, q)

Implied-vol solver
------------------
  The solver always inverts the OUT-OF-THE-MONEY forward price (a call price
  for K >= F is used as is; an in-the-money call is turned into the OTM put
  via put-call parity, and vice versa). In total-volatility space s = σ√T the
  OTM price is strictly increasing, convex below the Manaster-Koehler (1982)
  inflection point s* = sqrt(2|ln(K/F)|) and concave above it, and vega is
  largest at s*, so s* is the starting point. Newton on the raw price
  crawls at both ends — exponentially small prices below s*, prices
  saturating at their upper bound (F for calls, K for puts) far above it —
  so, as in Jäckel's "Let's Be Rational", it runs on transformed objectives:
  below s* on h(s) = −1/ln(p/F) (≈ 2s²/x² in the wing, so each step is close
  to a Babylonian square-root step), above s* on ln(q), q = (p_max − p)/F
  (≈ −s²/8 for large s). Every step is safeguarded by a bisection bracket
  that is tightened on each evaluation, so the method cannot diverge or
  cycle.

  An entry that is ill-posed (non-finite input, T <= 0, price at or below
  intrinsic, price above the sigma_max value, price below a caller-supplied
  floor), that float64 cannot resolve (see below) or that has not converged
  after max_iter iterations is returned as NaN — a non-converged iterate is
  never returned as an implied vol. With full_output=True a per-entry status
  code (IV_STATUS) says why.

  Resolution limit: the OTM target is rounded, by about one ulp of the input
  price, and also of F − K when an in-the-money input goes through parity.
  Its complement to the upper bound is rounded by one ulp of that bound. When
  either is within _MIN_SIGNIFICANCE (1,000) ulps of zero, the price does not
  determine σ. Examples are an ITM option whose time value is lost in the
  parity subtraction, or a total vol σ√T above ~14, where the price equals its
  upper bound to 13 digits. Such an entry is IV_UNRESOLVED instead of whatever
  σ happens to reproduce the rounded price. (Before this guard, a 900%-vol
  5-year put came back as 740% with status ok.) Above the limit the target has
  at least 3 significant digits, which bounds the relative σ error at about
  1e-5 in the wings and at most 1e-3 at the money.

  Why the old solver was replaced: the previous Newton loop in nn_pricer
  started from the Brenner-Subrahmanyam guess clipped to 0.05, which for OTM
  calls sits in the flat convex region where vega ≈ 0; the first step jumped
  to the 5.0 clip and the iterate then bounced between the clips and was
  returned as if converged (8.1% of 200k exact Black-Scholes prices over the
  NN training domain came back wrong by > 1e-6, 28.5% for ln(K/F) in
  [0.10, 0.25), typically as 1e-4).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from scipy.special import log_ndtr, ndtr

ArrayLike = Union[float, np.ndarray]
BoolLike = Union[bool, np.ndarray]
# Return annotations follow numpy's own convention: a function declared to
# return np.ndarray returns a numpy scalar of the same dtype (np.float64,
# np.int8) when every input is a scalar. numpy scalars support .shape, .ndim,
# [()] and float().

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
# Relative price residual |ln p(σ) − ln p_target| required (with a sub-tolerance
# Newton step) before an implied vol is accepted as converged.
_LN_RESIDUAL_TOL = 1e-7
# The OTM target and its complement to the upper bound must each exceed this
# many units of their float64 rounding (see "Resolution limit" above).
_MIN_SIGNIFICANCE = 1e3

# ── Status codes returned by implied_vol_black(..., full_output=True) ─────────

IV_OK = 0               # converged
IV_BAD_INPUT = 1        # non-finite input, F/K/T/discount <= 0
IV_NO_TIME_VALUE = 2    # OTM price <= 0: price at or below intrinsic value
IV_ABOVE_MAX = 3        # price >= Black price at sigma_max (or >= its upper bound)
IV_BELOW_MIN = 4        # price <= Black price at sigma_min
IV_BELOW_FLOOR = 5      # OTM price below the caller's min_otm_price floor
IV_NOT_CONVERGED = 6    # max_iter reached without meeting the tolerance
IV_UNRESOLVED = 7       # float64 rounding of the price does not determine σ

IV_STATUS = {
    IV_OK: "ok",
    IV_BAD_INPUT: "bad_input",
    IV_NO_TIME_VALUE: "no_time_value",
    IV_ABOVE_MAX: "above_sigma_max",
    IV_BELOW_MIN: "below_sigma_min",
    IV_BELOW_FLOOR: "below_price_floor",
    IV_NOT_CONVERGED: "not_converged",
    IV_UNRESOLVED: "unresolved_at_float64",
}


# ── Black-76 / Black-Scholes prices ───────────────────────────────────────────

def _as_float_arrays(*args: Union[ArrayLike, BoolLike]) -> Tuple[NDArray[np.float64], ...]:
    arrs = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])
    return tuple(np.array(a, dtype=float) for a in arrs)


def _otm_log_price_over_F(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    ln(OTM forward price / F) for log-moneyness x = ln(K/F) and total vol s = σ√T.

    OTM call (x >= 0):  p/F = N(d1) − e^x N(d2)
    OTM put  (x <  0):  p/F = e^x N(−d2) − N(−d1)
    with d1 = −x/s + s/2, d2 = d1 − s.

    Evaluated as log_ndtr(a) + ln(−expm1(log_ndtr(b) + c − log_ndtr(a))) so that
    neither underflow (|d| > 38) nor the cancellation between the two terms in
    the far wings destroys the result. Returns −inf where s <= 0.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s_safe = np.where(s > 0, s, 1.0)
        d1 = -x / s_safe + 0.5 * s_safe
        d2 = d1 - s_safe
        call = x >= 0
        # call: a = d1, b = d2, c = x ; put: a = −d2, b = −d1, c = −x, then + x
        la = np.where(call, log_ndtr(d1), log_ndtr(-d2))
        lb = np.where(call, log_ndtr(d2), log_ndtr(-d1))
        y = np.where(call, x + lb - la, lb - x - la)        # always < 0 in exact arithmetic
        y = np.minimum(y, -1e-300)
        out = la + np.log(-np.expm1(y)) + np.where(call, 0.0, x)
    return np.where(s > 0, out, -np.inf)


def _otm_log_complement_over_F(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    ln((p_max − p)/F) for the OTM price p, where p_max is its s → ∞ limit
    (F for the call, K for the put). For both sides the complement is
    N(−d1) + e^x N(d2), a sum of positive terms, evaluated in log space.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s_safe = np.where(s > 0, s, 1.0)
        d1 = -x / s_safe + 0.5 * s_safe
        d2 = d1 - s_safe
        out = np.logaddexp(log_ndtr(-d1), x + log_ndtr(d2))
    return np.where(s > 0, out, np.where(x >= 0, 0.0, x))


def _otm_log_vega_over_F(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """ln ∂(OTM price / F)/∂s = ln φ(d1) (identical for the call and the put)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s_safe = np.where(s > 0, s, 1.0)
        d1 = -x / s_safe + 0.5 * s_safe
        return np.where(s > 0, -0.5 * d1 * d1 - _LOG_SQRT_2PI, -np.inf)


def black_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    is_call: Union[bool, np.ndarray] = True,
    discount: ArrayLike = 1.0,
) -> np.ndarray:
    """
    Black-76 price of a European option on a forward F.

        call = D·[F N(d1) − K N(d2)],  put = D·[K N(−d2) − F N(−d1)]
        d1 = (ln(F/K) + σ²T/2)/(σ√T),  d2 = d1 − σ√T,  D = discount factor.

    With discount = 1 this is the undiscounted forward price. At T <= 0 or
    σ <= 0 the intrinsic value D·max(±(F−K), 0) is returned.
    """
    F, K, T, sigma, is_call, D = _as_float_arrays(F, K, T, sigma, is_call, discount)
    is_call = is_call.astype(bool)
    s = sigma * np.sqrt(np.maximum(T, 0.0))
    x = np.log(K / F)
    # exp(ln(p/F) + ln F), not F·exp(ln(p/F)): for F > 1 the latter goes through
    # a subnormal intermediate and loses the price's digits (7.5% at p ≈ 1e-319)
    with np.errstate(divide="ignore"):
        otm = np.exp(_otm_log_price_over_F(x, s) + np.log(F))
    otm = np.where(s > 0, otm, 0.0)
    # Recover the requested side from the OTM price via parity: C − P = F − K
    call = np.where(x >= 0, otm, otm + (F - K))
    put = np.where(x >= 0, otm + (K - F), otm)
    out: np.ndarray = D * np.where(is_call, call, put)
    return out


def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    sigma: ArrayLike,
    is_call: Union[bool, np.ndarray] = True,
) -> np.ndarray:
    """Black-Scholes price with continuous dividend yield q (F = S·e^{(r−q)T})."""
    S, K, T, r, q, sigma = _as_float_arrays(S, K, T, r, q, sigma)
    F = S * np.exp((r - q) * T)
    return black_price(F, K, T, sigma, is_call, np.exp(-r * T))


# ── Fourier pricing from a characteristic function (Lewis 2001) ──────────────

def fourier_otm_price(
    cf_logfwd: Callable[[np.ndarray], np.ndarray],
    x: ArrayLike,
    *,
    variance_hint: Optional[float] = None,
    tol: float = 1e-15,
    tail_tol: float = 1e-16,
    u_max: float = 1.0e5,
    h_max: float = 2.0,
    h_min: float = 0.01,
    n_min: int = 128,
    chunk: int = 8192,
) -> Tuple[np.ndarray, bool]:
    """
    Undiscounted OUT-OF-THE-MONEY price / F from the characteristic function of
    ln(S_T/F), for log-moneyness x = ln(K/F) (OTM call for x >= 0, put for x < 0).

    Lewis (2001):  C/F = 1 − (e^{x/2}/π) ∫₀^∞ Re[e^{−iux} φ(u − i/2)] / (u² + ¼) du
                   P/F = C/F − (1 − e^x)

    The integrand has poles at u = ±i/2, which limit a plain trapezoid rule to
    a step of about 0.1. Both poles have the same residue for every
    characteristic function (φ(0) = φ(−i) = 1), so a Black-Scholes control
    variate with total variance w removes them exactly:

        OTM/F = BS_OTM(x, w) − (e^{x/2}/π) ∫₀^∞ Re[e^{−iux}(φ − φ_BS)(u − i/2)] / (u² + ¼) du

    The remaining integrand is analytic in a wider strip, so the trapezoid
    rule (spectrally accurate for this even, smooth integrand) can use a much
    coarser step. The step is halved, reusing all previous nodes, until two
    successive estimates agree to `tol` (in units of F) for every strike;
    truncation U is the first point of a geometric grid beyond which
    |φ − φ_BS|/(u² + ¼) stays below tail_tol. Verified against adaptive
    quadrature to ~1e−14·F over the NN training domain, so prices far below
    that are numerically zero — callers must apply a price floor.

    Parameters
    ----------
    cf_logfwd     : callable u -> E[exp(i·u·ln(S_T/F))], vectorised over complex u
    x             : log-moneyness array
    variance_hint : total variance w for the control variate (any w > 0 is
                    exact; a w close to the model's makes the integrand small).
                    Default: second cumulant estimated from cf_logfwd.
    tol           : convergence tolerance (absolute, units of F)
    tail_tol      : truncation tolerance on the integrand modulus
    u_max         : cap on the truncation point U
    h_max, h_min  : largest initial / smallest allowed trapezoid step
    n_min         : minimum number of intervals on [0, U]

    Returns
    -------
    (otm_over_F, ok) — array shaped like x, and whether truncation and the
    step refinement both converged (if not, the prices are unreliable).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if variance_hint is None or not (np.isfinite(variance_hint) and variance_hint > 0):
        d = 0.05
        w = -2.0 * np.real(np.log(cf_logfwd(np.array([d + 0j]))[0])) / d**2
        variance_hint = w if (np.isfinite(w) and w > 0) else 0.01
    w = float(variance_hint)

    def _diff(z: np.ndarray) -> np.ndarray:
        out: np.ndarray = cf_logfwd(z) - np.exp(-0.5 * w * (1j * z + z * z))
        return out

    # Truncation point: first U on a geometric grid with tail below tail_tol at U and 1.25·U
    n_cand = int(np.ceil(np.log(u_max / 32.0) / np.log(1.25))) + 1
    cand = 32.0 * 1.25 ** np.arange(n_cand)
    with np.errstate(over="ignore", invalid="ignore"):
        tail = np.abs(_diff(cand - 0.5j)) / (cand**2 + 0.25)
    good = np.isfinite(tail) & (tail < tail_tol)
    both = good[:-1] & good[1:]
    if not both.any():
        return np.full(x.shape, np.nan), False
    U = float(cand[int(np.argmax(both))])

    def _sum(u: np.ndarray, weight: ArrayLike) -> np.ndarray:
        weight = np.broadcast_to(np.asarray(weight, dtype=float), u.shape)
        acc = np.zeros(x.shape)
        for i in range(0, u.size, chunk):
            uu = u[i:i + chunk]
            vals = _diff(uu - 0.5j) / (uu * uu + 0.25) * weight[i:i + chunk]
            acc += np.real(np.exp(-1j * np.outer(x, uu)) @ vals)
        return acc

    h = min(h_max, U / n_min, np.pi / (4.0 * max(float(np.max(np.abs(x))), 1e-3)))
    n = int(np.ceil(U / h))
    h = U / n
    u = np.arange(n + 1) * h
    wts = np.full(n + 1, h)
    wts[0] = wts[-1] = 0.5 * h
    integral = _sum(u, wts)
    scale = np.exp(0.5 * x) / np.pi
    converged = False
    while h >= h_min:
        mids = (np.arange(n) + 0.5) * h
        refined = 0.5 * integral + 0.5 * h * _sum(mids, 1.0)
        change = np.max(np.abs(refined - integral) * scale)
        integral, n, h = refined, 2 * n, 0.5 * h
        if change < tol:
            converged = True
            break

    s = np.sqrt(w)
    bs = np.where(x >= 0, black_price(1.0, np.exp(x), 1.0, s, True),
                  black_price(1.0, np.exp(x), 1.0, s, False))
    return bs - scale * integral, converged


# ── Greeks (spot, with continuous dividend yield) ─────────────────────────────

def bs_d1d2(F: ArrayLike, K: ArrayLike, T: ArrayLike,
            sigma: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """d1, d2 of the forward-measure Black formula."""
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1: np.ndarray = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2: np.ndarray = d1 - sigma * sqrtT
    return d1, d2


def _phi(d: ArrayLike) -> np.ndarray:
    out: np.ndarray = np.exp(-0.5 * d * d - _LOG_SQRT_2PI)
    return out


def bs_delta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike, is_call: BoolLike = True) -> np.ndarray:
    """∂V/∂S: call e^{−qT}N(d1), put −e^{−qT}N(−d1)."""
    F = S * np.exp((r - q) * T)
    d1, _ = bs_d1d2(F, K, T, sigma)
    dq = np.exp(-q * T)
    out: np.ndarray = np.where(is_call, dq * ndtr(d1), -dq * ndtr(-d1))
    return out


def bs_gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike) -> np.ndarray:
    """∂²V/∂S² = e^{−qT}φ(d1)/(S σ √T) (call = put)."""
    F = S * np.exp((r - q) * T)
    d1, _ = bs_d1d2(F, K, T, sigma)
    out: np.ndarray = np.exp(-q * T) * _phi(d1) / (S * sigma * np.sqrt(np.maximum(T, 1e-12)))
    return out


def bs_vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
            sigma: ArrayLike) -> np.ndarray:
    """∂V/∂σ = S e^{−qT} φ(d1) √T, per 1.00 (100 vol points) of σ (call = put)."""
    F = S * np.exp((r - q) * T)
    d1, _ = bs_d1d2(F, K, T, sigma)
    out: np.ndarray = S * np.exp(-q * T) * _phi(d1) * np.sqrt(np.maximum(T, 1e-12))
    return out


def bs_vomma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike) -> np.ndarray:
    """
    Vomma (a.k.a. volga) = ∂²V/∂σ² = Vega · d1 · d2 / σ, per (1.00 of σ)² (call = put).

    Sign: d1·d2 > 0 in both wings (convex in vol); d1·d2 < 0 near the money
    (locally concave in vol).
    """
    F = S * np.exp((r - q) * T)
    d1, d2 = bs_d1d2(F, K, T, sigma)
    out: np.ndarray = bs_vega(S, K, T, r, q, sigma) * d1 * d2 / np.maximum(sigma, 1e-8)
    return out


def bs_vanna(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike,
             sigma: ArrayLike) -> np.ndarray:
    """∂²V/∂S∂σ = −e^{−qT} φ(d1) d2 / σ, per 1.00 of σ (call = put)."""
    F = S * np.exp((r - q) * T)
    d1, d2 = bs_d1d2(F, K, T, sigma)
    out: np.ndarray = -np.exp(-q * T) * _phi(d1) * d2 / np.maximum(sigma, 1e-8)
    return out


# ── Implied volatility ────────────────────────────────────────────────────────

# The return type depends on full_output; the overloads let a type checker see
# which (they change nothing at run time).
@overload
def implied_vol_black(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike,
    is_call: BoolLike = ..., discount: ArrayLike = ..., *,
    sigma_min: float = ..., sigma_max: float = ..., min_otm_price: ArrayLike = ...,
    tol: float = ..., max_iter: int = ..., full_output: Literal[False] = ...,
) -> np.ndarray: ...
@overload
def implied_vol_black(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike,
    is_call: BoolLike = ..., discount: ArrayLike = ..., *,
    sigma_min: float = ..., sigma_max: float = ..., min_otm_price: ArrayLike = ...,
    tol: float = ..., max_iter: int = ..., full_output: Literal[True],
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def implied_vol_black(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike,
    is_call: BoolLike = ..., discount: ArrayLike = ..., *,
    sigma_min: float = ..., sigma_max: float = ..., min_otm_price: ArrayLike = ...,
    tol: float = ..., max_iter: int = ..., full_output: bool,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...
def implied_vol_black(
    price: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    is_call: Union[bool, np.ndarray] = True,
    discount: ArrayLike = 1.0,
    *,
    sigma_min: float = 1e-6,
    sigma_max: float = 10.0,
    min_otm_price: ArrayLike = 0.0,
    tol: float = 1e-12,
    max_iter: int = 100,
    full_output: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Black-76 implied volatility, vectorised.

    Parameters
    ----------
    price         : option prices (discounted with `discount`)
    F, K, T       : forward, strike, time to expiry (years)
    is_call       : True for calls, False for puts (scalar or array)
    discount      : discount factor applied to `price` (1.0 = undiscounted)
    sigma_min/max : search interval for σ; prices outside it give NaN
    min_otm_price : undiscounted OTM-price floor (same units as F). Inverting a
                    price that is below the accuracy of the pricer that
                    produced it fabricates an IV, so callers that know their
                    pricer's error should pass a floor; entries below it are NaN.
    tol           : absolute tolerance on σ
    max_iter      : iteration cap; entries not converged by then are NaN
    full_output   : also return the int status array (see IV_STATUS)

    Returns
    -------
    sigma, or (sigma, status) if full_output.
    """
    price, F, K, T, is_call, D, floor = _as_float_arrays(
        price, F, K, T, is_call, discount, min_otm_price
    )
    is_call = is_call.astype(bool)
    shape = price.shape
    status = np.full(shape, IV_OK, dtype=np.int8)

    finite = (np.isfinite(price) & np.isfinite(F) & np.isfinite(K) & np.isfinite(T)
              & np.isfinite(D) & (F > 0) & (K > 0) & (T > 0) & (D > 0))
    status[~finite] = IV_BAD_INPUT

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(finite, price / np.where(finite, D, 1.0), np.nan)
        Fs = np.where(finite, F, 1.0)
        Ks = np.where(finite, K, 1.0)
        Ts = np.where(finite, T, 1.0)
        x = np.log(Ks / Fs)
        # OTM undiscounted price via parity (C − P = F − K)
        otm = np.where(
            x >= 0,
            np.where(is_call, p, p - (Ks - Fs)),
            np.where(is_call, p - (Fs - Ks), p),
        )
        target = otm / Fs

    sqrtT = np.sqrt(Ts)
    s_lo = sigma_min * sqrtT
    s_hi = sigma_max * sqrtT

    pending = status == IV_OK
    no_tv = pending & ~(otm > 0)
    status[no_tv] = IV_NO_TIME_VALUE
    pending &= ~no_tv

    below_floor = pending & (otm < floor)
    status[below_floor] = IV_BELOW_FLOOR
    pending &= ~below_floor

    with np.errstate(divide="ignore", invalid="ignore"):
        # ln(otm/F), except where the quotient is subnormal and has lost its
        # significant digits: there ln(otm) − ln(F). (Not everywhere: near the
        # upper bound σ is so sensitive to ln p that the extra rounding of the
        # difference of two larger logs costs ~10x accuracy.)
        normal = target >= np.finfo(float).tiny
        ln_target = np.where(
            pending,
            np.where(normal, np.log(np.where(normal & pending, target, 1.0)),
                     np.log(np.where(pending, otm, 1.0)) - np.log(Fs)),
            0.0,
        )
    ln_hi = _otm_log_price_over_F(x, s_hi)
    ln_lo = _otm_log_price_over_F(x, s_lo)
    above = pending & (ln_target >= ln_hi)
    status[above] = IV_ABOVE_MAX
    pending &= ~above
    below = pending & (ln_target <= ln_lo)
    status[below] = IV_BELOW_MIN
    pending &= ~below

    # Float64 resolution of the target: one ulp of the input price,
    # plus one ulp of F − K when an ITM input went through parity; its
    # complement to the upper bound p_max (F for calls, K for puts) also
    # carries one ulp of p_max.
    # Compared in absolute terms (index points), so tiny prices do not underflow.
    eps, ulp_min = np.finfo(float).eps, np.finfo(float).smallest_subnormal
    with np.errstate(invalid="ignore"):
        via_parity = np.where(x >= 0, ~is_call, is_call)
        noise = eps * (np.abs(p) + np.where(via_parity, np.abs(Fs - Ks), 0.0)) + ulp_min
        p_max = np.where(x >= 0, Fs, Ks)
        unresolved = pending & ((otm <= _MIN_SIGNIFICANCE * noise)
                                | (p_max - otm <= _MIN_SIGNIFICANCE * (noise + eps * p_max)))
    status[unresolved] = IV_UNRESOLVED
    pending &= ~unresolved

    s_out = np.full(shape, np.nan)
    if pending.any():
        idx = np.flatnonzero(pending)
        xi = x.ravel()[idx]
        lt = ln_target.ravel()[idx]
        tgt = target.ravel()[idx]
        a = s_lo.ravel()[idx].copy()
        b = s_hi.ravel()[idx].copy()
        xtol = tol * sqrtT.ravel()[idx]

        # Manaster-Koehler start: inflection point of the price in s (max vega)
        s = np.clip(np.sqrt(2.0 * np.abs(xi)), a, b)
        lower_region = lt < _otm_log_price_over_F(xi, s)
        # Upper region works on the complement q = p_max/F − p/F (p_max: F or K)
        with np.errstate(divide="ignore", invalid="ignore"):
            lq_t = np.log(np.where(xi >= 0, 1.0, np.exp(xi)) - tgt)

        done = np.zeros(idx.size, dtype=bool)
        for _ in range(max_iter):
            act = ~done
            if not act.any():
                break
            sa, xa = s[act], xi[act]
            lp = _otm_log_price_over_F(xa, sa)
            lv = _otm_log_vega_over_F(xa, sa)        # derivatives in log space:
            diff_ln = lp - lt[act]                   # φ(d1) underflows in the wings
            # Tighten the bracket: the OTM price is increasing in s
            a[act] = np.where(diff_ln < 0, sa, a[act])
            b[act] = np.where(diff_ln > 0, sa, b[act])
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                # Below s*: Newton on h(s) = −1/ln(p/F), which behaves like
                # 2s²/x² in the wing (ln p ≈ −x²/2s²), so Newton is close to the
                # Babylonian square root and converges from far away.
                dlp = np.exp(lv - lp)                           # d ln p / ds
                step_h = (1.0 / lt[act] - 1.0 / lp) / (dlp / (lp * lp))
                # Above s*: Newton on ln q, q = complement to the upper bound,
                # ≈ −s²/8 for large s, so prices close to their upper bound
                # (huge total vol) converge as fast as ordinary ones.
                lq = _otm_log_complement_over_F(xa, sa)
                step_q = (lq - lq_t[act]) / (-np.exp(lv - lq))
            step = np.where(lower_region[act], step_h, step_q)
            s_new = sa - step
            aa, bb = a[act], b[act]
            # Converged once the Newton step itself is below tolerance AND the
            # price residual is small. The step is tested raw: at the root,
            # rounding can put a bracket end exactly on s, and a sub-tolerance
            # step "outside" it must not trigger a bisection that throws the
            # iterate away. The residual guard rejects spuriously tiny steps
            # far from the root (where the wing formulas lose all precision).
            small = (np.isfinite(step) & (np.abs(step) <= xtol[act])
                     & (np.abs(diff_ln) <= _LN_RESIDUAL_TOL))
            bad = ~small & (~np.isfinite(s_new) | (s_new <= aa) | (s_new >= bb)
                            | ((np.abs(step) <= xtol[act]) & (np.abs(diff_ln) > _LN_RESIDUAL_TOL)))
            s_new = np.where(bad, 0.5 * (aa + bb), s_new)       # bisect if Newton leaves the bracket
            conv = small | ((bb - aa) <= xtol[act]) | (diff_ln == 0)
            s[act] = np.where(diff_ln == 0, sa, s_new)
            done[act] = conv

        s_res = np.where(done, s, np.nan)
        flat_status = status.ravel()
        flat_status[idx[~done]] = IV_NOT_CONVERGED
        status = flat_status.reshape(shape)
        s_flat = s_out.ravel()
        s_flat[idx] = s_res
        s_out = s_flat.reshape(shape)

    sigma: np.ndarray = s_out / sqrtT
    sigma = np.where(status == IV_OK, sigma, np.nan)
    if sigma.ndim == 0:
        sigma = sigma[()]
        status = status[()]
    return (sigma, status) if full_output else sigma


@overload
def implied_vol_bs(
    price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
    q: ArrayLike, is_call: BoolLike = ..., *, full_output: Literal[False] = ...,
    **kwargs: Any,
) -> np.ndarray: ...
@overload
def implied_vol_bs(
    price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
    q: ArrayLike, is_call: BoolLike = ..., *, full_output: Literal[True],
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]: ...
def implied_vol_bs(
    price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    is_call: Union[bool, np.ndarray] = True,
    *,
    full_output: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Black-Scholes implied vol with dividend yield q (wraps implied_vol_black;
    keyword arguments are passed through)."""
    price, S, K, T, r, q = _as_float_arrays(price, S, K, T, r, q)
    F = S * np.exp((r - q) * T)
    return implied_vol_black(price, F, K, T, is_call, np.exp(-r * T),
                             full_output=full_output, **kwargs)


def _bs_iv_vectorized(
    call_prices: np.ndarray,
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Backward-compatible entry point (formerly nn_pricer._bs_iv_vectorized).

    Inverts DISCOUNTED call prices (as returned by heston_call_batch) to Black
    implied vols with the robust solver above, on the old search range
    σ ∈ (1e-4, 5.0). Returns NaN wherever the price is ill-posed or the solver
    has not converged to `tol` within `max_iter` iterations — unlike the old
    implementation, which returned the last (often clipped) iterate.
    """
    out: np.ndarray = implied_vol_black(
        call_prices, F, K, T, True, np.exp(-np.asarray(r, float) * np.asarray(T, float)),
        sigma_min=1e-4, sigma_max=5.0, tol=tol, max_iter=max_iter,
    )
    return out
