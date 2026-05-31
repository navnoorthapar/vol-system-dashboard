"""
quintic_ou.py — Two-factor Quintic Ornstein-Uhlenbeck volatility model (C13).

References
----------
  Bourgey & de Marco (2025): arXiv:2503.14158
    "Capturing Smile Dynamics with the Quintic Volatility Model: SPX, SSR and VIX"
  Carr, Pelts & Tehranchi (2022): arXiv:2212.10917
    "The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX & VIX smiles"

Model dynamics
--------------
    dS_t / S_t  = σ(t) dB_t
    σ(t)        = g_0(t) · p(Z_t)
    Z_t         = θ X_t + (1-θ) Y_t
    dX_t        = -λx X_t dt + dW_t
    dY_t        = -λy Y_t dt + dW_t          ← same BM W drives both factors
    d<B, W>_t   = ε dt                        ← leverage

    p(z) = α0 + α1·z + α3·z³ + α5·z⁵        ← degree-5, odd + constant
    g_0(t) = sqrt(ξ_0(t) / E[p(Z_t)²])      ← forward-variance normalisation

Parameters (8 total)
--------------------
    λx      fast OU mean-reversion speed  (λx > λy)
    λy      slow OU mean-reversion speed
    θ       weight of fast factor in Z_t   (θ ∈ ℝ, typically 0–2)
    α0      polynomial constant
    α1      polynomial linear coefficient
    α3      polynomial cubic coefficient
    α5      polynomial quintic coefficient
    ε       spot-vol correlation  ε ∈ (−1, 0)   (leverage, analogous to ρ in Heston)
"""

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── VIX constants ──────────────────────────────────────────────────────────────
_VIX_DELTA  = 30.0 / 365.0   # 30-calendar-day integration window
_VIX_SCALE  = 100.0           # VIX quoted as percentage points

# Default quadrature sizes
_N_QUAD_OUTER = 22    # GH points per dim for (X_T, Y_T) marginal → 22² = 484 ≈ 500
_N_QUAD_INNER = 20    # GL points for the time integral over [T, T+Δ]


# ── Parameter dataclass ────────────────────────────────────────────────────────

@dataclass
class QuinticOUParams:
    """8-parameter two-factor Quintic OU model."""
    lam_x:   float   # fast OU speed  (> lam_y)
    lam_y:   float   # slow OU speed
    theta:   float   # factor weight  (Z = θX + (1-θ)Y)
    alpha0:  float   # polynomial constant
    alpha1:  float   # linear coefficient
    alpha3:  float   # cubic coefficient
    alpha5:  float   # quintic coefficient
    epsilon: float   # spot-vol correlation  ε ∈ (−1, 1)

    # ── Conversion helpers ────────────────────────────────────────────────────

    def to_array(self) -> np.ndarray:
        return np.array([self.lam_x, self.lam_y, self.theta,
                         self.alpha0, self.alpha1, self.alpha3, self.alpha5,
                         self.epsilon], dtype=float)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "QuinticOUParams":
        return cls(lam_x=float(x[0]), lam_y=float(x[1]), theta=float(x[2]),
                   alpha0=float(x[3]), alpha1=float(x[4]),
                   alpha3=float(x[5]), alpha5=float(x[6]),
                   epsilon=float(x[7]))

    def alpha_vec(self) -> np.ndarray:
        """Full 6-element coefficient vector [α0, α1, 0, α3, 0, α5]."""
        return np.array([self.alpha0, self.alpha1, 0.0,
                         self.alpha3, 0.0, self.alpha5], dtype=float)

    def __repr__(self) -> str:
        return (f"QuinticOUParams(λx={self.lam_x:.4f}, λy={self.lam_y:.4f}, "
                f"θ={self.theta:.4f}, α0={self.alpha0:.4f}, α1={self.alpha1:.4f}, "
                f"α3={self.alpha3:.4f}, α5={self.alpha5:.4f}, ε={self.epsilon:.4f})")


# Default parameter starting point (calibration initial guess)
QUINTIC_DEFAULTS = QuinticOUParams(
    lam_x=35.0, lam_y=0.6, theta=0.94,
    alpha0=0.05, alpha1=0.10, alpha3=0.05, alpha5=0.02,
    epsilon=-0.75,
)

# Calibration bounds  (lower, upper) for each parameter
QUINTIC_BOUNDS = [
    (1.0,   500.0),   # lam_x  (extended from 200 — prior run hit 182/200)
    (0.01,  20.0),    # lam_y
    (0.0,   3.0),     # theta
    (-0.5,  0.5),     # alpha0
    (-2.0,  2.0),     # alpha1
    (-2.0,  2.0),     # alpha3
    (-1.0,  1.0),     # alpha5
    (-0.99, -0.01),   # epsilon  (leverage: always negative)
]


# ── Polynomial evaluation ──────────────────────────────────────────────────────

def eval_poly(z: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Evaluate p(z) = α0 + α1·z + α3·z³ + α5·z⁵.

    Parameters
    ----------
    z     : evaluation points (scalar or ndarray)
    alpha : 6-element array [α0, α1, 0, α3, 0, α5]
    """
    z = np.asarray(z, dtype=float)
    a0, a1, _, a3, _, a5 = alpha
    return a0 + a1 * z + a3 * z**3 + a5 * z**5


# ── Gaussian moments ───────────────────────────────────────────────────────────

def gaussian_moments(max_n: int, mu: float, sigma2: float) -> np.ndarray:
    """
    Compute E[Z^k] for k = 0, …, max_n when Z ~ N(mu, sigma2).

    Recurrence:  E[Z^n] = mu · E[Z^{n-1}] + (n-1) · σ² · E[Z^{n-2}]
    """
    m = np.zeros(max_n + 1, dtype=float)
    m[0] = 1.0
    if max_n >= 1:
        m[1] = mu
    for k in range(2, max_n + 1):
        m[k] = mu * m[k - 1] + (k - 1) * sigma2 * m[k - 2]
    return m


def ep_squared(mu: float, sigma2: float, alpha: np.ndarray) -> float:
    """
    Compute E[p(Z)²] analytically for Z ~ N(mu, sigma2).

    p(z) = Σ_k alpha[k] · z^k  (alpha is the 6-element vector).

    Expands the cross-product and uses the Gaussian moment recurrence.
    """
    m = gaussian_moments(10, mu, sigma2)       # need up to degree 5+5=10
    c = np.asarray(alpha, dtype=float)
    result = 0.0
    for j in range(6):
        if c[j] == 0.0:
            continue
        for k in range(6):
            if c[k] == 0.0:
                continue
            result += c[j] * c[k] * m[j + k]
    return float(result)


# ── OU variance / covariance functions ────────────────────────────────────────

def var_Z(t: float, lam_x: float, lam_y: float, theta: float) -> float:
    """
    Var[Z_t] = Var[θ X_t + (1-θ) Y_t],  X_0 = Y_0 = 0,  same BM.

    = θ² · Var[X_t]  +  (1-θ)² · Var[Y_t]  +  2θ(1-θ) · Cov[X_t, Y_t]
    """
    if t <= 0.0:
        return 0.0
    vx  = (1.0 - math.exp(-2.0 * lam_x * t)) / (2.0 * lam_x)
    vy  = (1.0 - math.exp(-2.0 * lam_y * t)) / (2.0 * lam_y)
    cxy = (1.0 - math.exp(-(lam_x + lam_y) * t)) / (lam_x + lam_y)
    return theta**2 * vx + (1.0 - theta)**2 * vy + 2.0 * theta * (1.0 - theta) * cxy


def _cond_var_Z(dt: float, lam_x: float, lam_y: float, theta: float) -> float:
    """
    Var[Z_{t+dt} | X_t, Y_t]  — conditional variance increment.
    Equals var_Z(dt, ...) since the conditional residual has the same structure.
    """
    return var_Z(dt, lam_x, lam_y, theta)


def _cond_var_components(dt: float, lam_x: float, lam_y: float
                          ) -> Tuple[float, float, float]:
    """Return (Var_X_cond, Var_Y_cond, Cov_XY_cond) for one OU step."""
    if dt <= 0.0:
        return 0.0, 0.0, 0.0
    vx  = (1.0 - math.exp(-2.0 * lam_x * dt)) / (2.0 * lam_x)
    vy  = (1.0 - math.exp(-2.0 * lam_y * dt)) / (2.0 * lam_y)
    cxy = (1.0 - math.exp(-(lam_x + lam_y) * dt)) / (lam_x + lam_y)
    return vx, vy, cxy


# ── Forward-variance normalisation ────────────────────────────────────────────

def g0_at(t: float, xi0_t: float, params: QuinticOUParams) -> float:
    """
    g_0(t) = sqrt( ξ_0(t) / E[p(Z_t)²] )

    Ensures E[σ(t)²] = ξ_0(t) unconditionally (Z_t ~ N(0, Var[Z_t])).
    """
    if xi0_t <= 0.0 or t < 0.0:
        return 1.0
    ep2 = ep_squared(0.0, var_Z(t, params.lam_x, params.lam_y, params.theta),
                     params.alpha_vec())
    if ep2 <= 0.0:
        return 1.0
    return math.sqrt(xi0_t / ep2)


# ── VIX² computation ──────────────────────────────────────────────────────────

def compute_vix2(
    T:            float,
    xt:           float,
    yt:           float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_quad:       int = _N_QUAD_INNER,
) -> float:
    """
    Compute VIX²(T; X_T=xt, Y_T=yt) via Gauss-Legendre integration.

    VIX²(T) = (100² / Δ) ∫_T^{T+Δ} ξ_0(s) · E[p(Z_s)² | X_T, Y_T] ds

    For each node s ∈ [T, T+Δ]:
      • Z_s | F_T ~ N(μ_Z(s), σ²_Z_cond(s))
      • μ_Z(s)  = θ·xt·e^{-λx(s-T)} + (1-θ)·yt·e^{-λy(s-T)}
      • σ²_cond = Var[Z_{s-T}] (same-BM OU formula starting from zero residual)
    """
    p = params
    a = p.alpha_vec()
    Delta = _VIX_DELTA

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # map [-1,1] → [T, T+Delta]
    s_vals = 0.5 * (nodes + 1.0) * Delta + T
    w_vals = weights * 0.5 * Delta

    integral = 0.0
    for s, w in zip(s_vals, w_vals):
        dt = float(s - T)
        mu_x = xt * math.exp(-p.lam_x * dt) if dt > 0.0 else xt
        mu_y = yt * math.exp(-p.lam_y * dt) if dt > 0.0 else yt
        mu_z = p.theta * mu_x + (1.0 - p.theta) * mu_y
        sig2_cond = _cond_var_Z(dt, p.lam_x, p.lam_y, p.theta)
        # g_0(s)² = ξ_0(s) / E[p(Z_s)²]_unconditional
        # E[σ(s)²|F_T] = g_0(s)² · E[p(Z_s)²|F_T] = ξ_0(s) · ep_cond / ep_uncond
        ep_uncond = ep_squared(0.0, var_Z(float(s), p.lam_x, p.lam_y, p.theta), a)
        if ep_uncond <= 0.0:
            continue
        ep_cond = ep_squared(mu_z, sig2_cond, a)
        integral += w * fwd_var_func(s) * (ep_cond / ep_uncond)

    return (_VIX_SCALE**2 / Delta) * integral


# ── 2-D Gauss-Hermite quadrature for (X_T, Y_T) marginal ─────────────────────

def _gauss_hermite_2d(
    n:      int,
    var_x:  float,
    var_y:  float,
    cov_xy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 2-D product Gauss-Hermite quadrature for BivNorm(0, Σ).

    Returns (x_pts, y_pts, weights) of length n².
    Weights are normalised to sum to 1.
    """
    nodes_1d, weights_1d = np.polynomial.hermite.hermgauss(n)
    # GH approximates ∫ f(x)exp(-x²)dx; for N(0,1) we need z = sqrt(2)*x
    z_std = math.sqrt(2.0) * nodes_1d
    w_1d  = weights_1d / math.sqrt(math.pi)   # normalise: sum → 1

    # Cholesky  [[var_x, cov_xy], [cov_xy, var_y]]
    L11 = math.sqrt(max(var_x, 1e-12))
    L21 = cov_xy / L11
    L22 = math.sqrt(max(var_y - L21**2, 1e-12))

    n2  = n * n
    xp  = np.empty(n2)
    yp  = np.empty(n2)
    wp  = np.empty(n2)
    idx = 0
    for i in range(n):
        for j in range(n):
            xp[idx] = L11 * z_std[i]
            yp[idx] = L21 * z_std[i] + L22 * z_std[j]
            wp[idx] = w_1d[i] * w_1d[j]
            idx += 1
    return xp, yp, wp


# ── VIX futures & options pricing ─────────────────────────────────────────────

def price_vix_futures(
    T:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> float:
    """
    VIX futures price ≈ E[VIX_T] = E[sqrt(VIX²_T)].

    Integrates over the bivariate-Normal marginal of (X_T, Y_T) via 2-D
    Gauss-Hermite quadrature (~500 points).
    """
    p = params
    var_x  = (1.0 - math.exp(-2.0 * p.lam_x * T)) / (2.0 * p.lam_x)
    var_y  = (1.0 - math.exp(-2.0 * p.lam_y * T)) / (2.0 * p.lam_y)
    cov_xy = (1.0 - math.exp(-(p.lam_x + p.lam_y) * T)) / (p.lam_x + p.lam_y)

    xp, yp, wp = _gauss_hermite_2d(n_outer, var_x, var_y, cov_xy)

    ev = 0.0
    for xi, yi, wi in zip(xp, yp, wp):
        vix2 = compute_vix2(T, float(xi), float(yi), params, fwd_var_func, n_inner)
        ev  += wi * math.sqrt(max(vix2, 0.0))
    return float(ev)


def price_vix_option(
    T:            float,
    K_vix:        float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> float:
    """
    VIX call option price = E[max(VIX_T − K, 0)] via 2-D Gauss-Hermite (~500 pts).

    Parameters
    ----------
    K_vix : strike in VIX-point units (e.g. 20.0 means VIX = 20)
    """
    p = params
    var_x  = (1.0 - math.exp(-2.0 * p.lam_x * T)) / (2.0 * p.lam_x)
    var_y  = (1.0 - math.exp(-2.0 * p.lam_y * T)) / (2.0 * p.lam_y)
    cov_xy = (1.0 - math.exp(-(p.lam_x + p.lam_y) * T)) / (p.lam_x + p.lam_y)

    xp, yp, wp = _gauss_hermite_2d(n_outer, var_x, var_y, cov_xy)

    call = 0.0
    for xi, yi, wi in zip(xp, yp, wp):
        vix2 = compute_vix2(T, float(xi), float(yi), params, fwd_var_func, n_inner)
        call += wi * max(math.sqrt(max(vix2, 0.0)) - K_vix, 0.0)
    return float(call)


# ── SPX Monte Carlo pricing ────────────────────────────────────────────────────

def _run_mc_paths(
    T:       float,
    params:  QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths: int,
    n_steps: int,
    rng:     np.random.RandomState,
    flip:    bool,
) -> np.ndarray:
    """
    Simulate log(S_T / S_0) for n_paths paths.

    Exact OU update (Gaussian):
        X_{t+dt} = e^{-λx dt} X_t  +  σ_x Z1
        Y_{t+dt} = e^{-λy dt} Y_t  +  σ_y Z1    ← same Z1 (same BM)
    Euler log-price:
        Δlog S  = -½ σ² dt  +  σ √dt (ε Z1 + √(1-ε²) Z2)
    """
    p = params
    a = p.alpha_vec()
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    eps_perp = math.sqrt(max(1.0 - p.epsilon**2, 0.0))

    ex  = math.exp(-p.lam_x * dt)
    ey  = math.exp(-p.lam_y * dt)
    sx  = math.sqrt((1.0 - math.exp(-2.0 * p.lam_x * dt)) / (2.0 * p.lam_x))
    sy  = math.sqrt((1.0 - math.exp(-2.0 * p.lam_y * dt)) / (2.0 * p.lam_y))

    Xt   = np.zeros(n_paths)
    Yt   = np.zeros(n_paths)
    logS = np.zeros(n_paths)

    for step in range(n_steps):
        t_mid = (step + 0.5) * dt
        Z1 = rng.randn(n_paths)
        Z2 = rng.randn(n_paths)
        if flip:
            Z1 = -Z1
            Z2 = -Z2

        Xt = ex * Xt + sx * Z1
        Yt = ey * Yt + sy * Z1    # same Z1

        Zt = p.theta * Xt + (1.0 - p.theta) * Yt

        # Forward-variance normalisation (unconditional at t_mid)
        xi0   = fwd_var_func(t_mid)
        ep2   = ep_squared(0.0, var_Z(t_mid, p.lam_x, p.lam_y, p.theta), a)
        g0    = math.sqrt(xi0 / max(ep2, 1e-12))

        sigma = np.maximum(g0 * eval_poly(Zt, a), 0.0)
        logS += -0.5 * sigma**2 * dt + sigma * sqrt_dt * (p.epsilon * Z1 + eps_perp * Z2)

    return logS


def price_spx_options_mc(
    S0:           float,
    T:            float,
    strikes:      np.ndarray,
    r:            float,
    q:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 20_000,
    n_steps:      int = 0,        # 0 → auto (50 steps/year, min 10)
    seed:         int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Price a batch of SPX calls at the same expiry T via MC (antithetic variates).

    Returns
    -------
    prices   : call prices (same order as strikes)
    std_errs : 1-sigma MC errors
    """
    strikes = np.asarray(strikes, dtype=float)
    if n_steps == 0:
        n_steps = max(10, int(round(50.0 * T)))
    half = max(1, n_paths // 2)

    rng = np.random.RandomState(seed)
    logS1 = _run_mc_paths(T, params, fwd_var_func, half, n_steps, rng, flip=False)
    logS2 = _run_mc_paths(T, params, fwd_var_func, half, n_steps, rng, flip=True)
    logS  = np.concatenate([logS1, logS2])

    S_T   = S0 * np.exp(logS + (r - q) * T)
    disc  = math.exp(-r * T)

    prices   = np.empty(len(strikes))
    std_errs = np.empty(len(strikes))
    for i, K in enumerate(strikes):
        payoff   = np.maximum(S_T - K, 0.0)
        dp       = disc * payoff
        prices[i]   = float(np.mean(dp))
        std_errs[i] = float(np.std(dp) / math.sqrt(len(dp)))
    return prices, std_errs


def implied_vol_surface_mc(
    S0:           float,
    r:            float,
    q:            float,
    expiries:     np.ndarray,   # per-row expiry (same length as strikes)
    strikes:      np.ndarray,   # per-row strike
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 10_000,
    seed:         int = 42,
) -> np.ndarray:
    """
    Compute SPX implied vols for multiple (K, T) pairs, sharing MC paths per expiry.

    Returns an array of implied vols (NaN where pricing fails).
    """
    from joint_vol_calibration.models.heston import implied_vol_from_price

    ivs = np.full(len(strikes), np.nan)
    base_seed = seed

    for T in np.unique(expiries):
        mask     = expiries == T
        K_slice  = strikes[mask]
        prices, _ = price_spx_options_mc(
            S0, T, K_slice, r, q, params, fwd_var_func,
            n_paths=n_paths, seed=base_seed,
        )
        disc = math.exp(-r * T)
        F    = S0 * math.exp((r - q) * T)

        row_ivs = np.full(len(K_slice), np.nan)
        for j, (K, price) in enumerate(zip(K_slice, prices)):
            lo = disc * max(F - K, 0.0)
            hi = disc * F
            if price > lo + 1e-6 and price < hi - 1e-6:
                iv = implied_vol_from_price(price, S0, K, T, r, q)
                if iv is not None:
                    row_ivs[j] = iv

        indices = np.where(mask)[0]
        for j, idx in enumerate(indices):
            ivs[idx] = row_ivs[j]
        base_seed += 1

    return ivs


# ── VIX option implied vol (Black futures model) ─────────────────────────────

def vix_option_implied_vol(
    T:            float,
    K_vix:        float,
    F_vix:        float,    # VIX futures price
    price:        float,    # call price in VIX pts
    r:            float = 0.045,
) -> Optional[float]:
    """
    Invert the Black-76 formula to get VIX option implied vol.
    F_vix is the VIX futures price (= E[VIX_T] from price_vix_futures).
    """
    from scipy.optimize import brentq

    disc = math.exp(-r * T)
    if T <= 0 or F_vix <= 0:
        return None

    def black76_call(sigma):
        d1 = (math.log(F_vix / K_vix) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        from scipy.stats import norm
        return disc * (F_vix * norm.cdf(d1) - K_vix * norm.cdf(d2))

    lo_price = disc * max(F_vix - K_vix, 0.0)
    hi_price = disc * F_vix
    if price <= lo_price + 1e-6 or price >= hi_price - 1e-6:
        return None
    try:
        return float(brentq(lambda s: black76_call(s) - price, 1e-4, 10.0, xtol=1e-6))
    except Exception:
        return None


# ── Model wrapper class ────────────────────────────────────────────────────────

class QuinticOUModel:
    """
    Convenience wrapper: holds parameters + forward-variance curve,
    exposes VIX and SPX pricing methods.
    """

    def __init__(
        self,
        params:       QuinticOUParams,
        fwd_var_func: Optional[Callable[[float], float]] = None,
        r:            float = 0.045,
        q:            float = 0.013,
        S0:           float = 100.0,
    ):
        self.params       = params
        self.r            = r
        self.q            = q
        self.S0           = S0
        self.fwd_var_func = fwd_var_func or (lambda t: 0.04)  # flat 20% vol

    def vix_futures(self, T: float, **kw) -> float:
        return price_vix_futures(T, self.params, self.fwd_var_func, **kw)

    def vix_call(self, T: float, K: float, **kw) -> float:
        return price_vix_option(T, K, self.params, self.fwd_var_func, **kw)

    def spx_calls(self, T: float, strikes: np.ndarray, **kw) -> Tuple[np.ndarray, np.ndarray]:
        return price_spx_options_mc(
            self.S0, T, strikes, self.r, self.q,
            self.params, self.fwd_var_func, **kw,
        )

    def spx_implied_vols(self, expiries: np.ndarray, strikes: np.ndarray, **kw) -> np.ndarray:
        return implied_vol_surface_mc(
            self.S0, self.r, self.q, expiries, strikes,
            self.params, self.fwd_var_func, **kw,
        )

    def var_Z_at(self, t: float) -> float:
        return var_Z(t, self.params.lam_x, self.params.lam_y, self.params.theta)

    def g0_at(self, t: float) -> float:
        return g0_at(t, self.fwd_var_func(t), self.params)
