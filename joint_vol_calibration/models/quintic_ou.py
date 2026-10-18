"""
quintic_ou.py — Two-factor Quintic Ornstein-Uhlenbeck volatility model (C13).

References
----------
  Abi Jaber, Illand & Li (2022): arXiv:2212.10917
    "The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates
     SPX & VIX smiles"  (one-factor model, p(x) = α0 + α1x + α3x³ + α5x⁵)
  Abi Jaber & Li (2025): arXiv:2503.14158
    "Capturing Smile Dynamics with the Quintic Volatility Model: SPX,
     Skew-Stickiness Ratio and VIX"  (two-factor model implemented here)
  Authors' reference implementation: github.com/shaunlinz02/two_factor_quintic_ou

Model dynamics
--------------
    dS_t / S_t  = σ_t dB_t,       B = ε W + sqrt(1-ε²) W⊥
    σ_t         = g_0(t) · p(Z_t)
    Z_t         = θ X_t + (1-θ) Y_t
    dX_t        = -λx X_t dt + dW_t,   X_0 = 0
    dY_t        = -λy Y_t dt + dW_t,   Y_0 = 0   ← same BM W drives both factors

    p(z)   = α0 + α1·z + α3·z³ + α5·z⁵             ← odd + constant
    g_0(t) = sqrt(ξ_0(t) / E[p(Z_t)²])           ← E[σ_t²] = ξ_0(t)

σ_t is SIGNED: there is no floor. dS = Sσ dB only sees σ², and a floor would
make the SPX leg a different model from the VIX leg (which uses p² through the
g_0 normalisation and VIX² formula).

Which variant this is, and where it departs from the papers
------------------------------------------------------------
  • Two factors (Abi Jaber & Li 2025) with the odd-plus-constant polynomial of
    Abi Jaber, Illand & Li (2022) — i.e. Section 3.2 of Abi Jaber & Li, which
    sets α2 = α4 = 0. The papers' ρ is called ε here.
  • p can be rescaled by any non-zero constant without changing the model
    (g_0 absorbs it; Abi Jaber & Li, Remark 2.1), so one coefficient is
    redundant. The calibrator fixes α5 = 1 as the paper does, keeps
    α0, α1, α3 ≥ 0 (Abi Jaber, Illand & Li: same-sign ATM skew as ε) and
    ε ≤ 0. S is a true martingale whenever α5 > 0 and ε ≤ 0 (proved for one
    factor with a flat-g_0 forward-variance curve in Abi Jaber, Illand & Li,
    Prop. A.1; stated for two factors in Abi Jaber & Li, Remark 2.1). The
    dataclass below accepts any real coefficients;
    `admissibility_violations` reports what falls outside that set.
  • θ is restricted to [0, 1] by the calibrator; the paper only requires
    θ ≥ 0 (its reported fits have θ between 0.678 and 0.964).
  • ξ_0 is a piecewise-constant curve bootstrapped from the Cboe VIX indices
    (the papers extract it from SPX log-contract replication or use a
    parametric curve). VIX window Δ = 30/365 (Cboe); the reference code
    uses 30/360.
  • VIX derivatives: VIX²_T is a degree-10 polynomial in (X_T, Y_T) whose
    coefficients come from a Gauss-Legendre time integral split at the jumps
    of ξ_0; it is integrated against the (X_T, Y_T) law with a uniform
    lattice rule on the standard-normal plane (the authors use a 1450-point
    optimal quantization grid).
  • SPX Monte Carlo: left-point Itô scheme with EXACT joint Gaussian steps
    for (∫dW, ∫e^{-λx(·)}dW, ∫e^{-λy(·)}dW) on a 252-steps/yr grid (≥ 100
    steps per expiry), Richardson-extrapolated against the half-step grid on
    the same Brownian path; the reference code reuses one normal for both OU
    innovations and the Brownian increment and does not extrapolate. Prices
    use the conditional (Romano–Touzi mixing) estimator with antithetic pairs
    and an empirical martingale correction (sample mean of S_T/F forced to 1,
    which keeps every simulated smile free of static arbitrage); the authors
    use a timer-option control variate.

Parameters (8 fields, 7 free once α5 is normalised)
---------------------------------------------------
    λx      fast OU mean-reversion speed  (per year)
    λy      slow OU mean-reversion speed  (per year)
    θ       weight of the fast factor in Z_t   (calibrated in [0, 1])
    α0      polynomial constant
    α1      polynomial linear coefficient
    α3      polynomial cubic coefficient
    α5      polynomial quintic coefficient  (scale-redundant; 1 after calibration)
    ε       correlation between B and W  ε ∈ [−1, 1]  (calibrated in [−1, 0])
"""

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr

logger = logging.getLogger(__name__)

# ── VIX constants ──────────────────────────────────────────────────────────────
_VIX_DELTA  = 30.0 / 365.0   # 30-calendar-day integration window
_VIX_SCALE  = 100.0           # VIX quoted as percentage points

# Default quadrature sizes. VIX² is evaluated from its polynomial coefficients,
# so a fine cubature is cheap (~6 ms per expiry). With 101 lattice points per
# dimension (h = 0.16, 7,845 nodes inside |z| ≤ 8) the Black-76 VIX IVs of the
# 2026-03-24 calibration set are within 0.03 vp of a 0.01-step reference; a
# 48² Gauss-Hermite product rule was off by up to 0.8 vp on the kinked payoff.
_N_QUAD_OUTER = 101   # lattice points per dimension of the (X_T, Y_T) plane rule
_N_QUAD_INNER = 20    # GL points per smooth piece of the time integral over [T, T+Δ]

# SPX Monte Carlo time grid, as in the authors' reference code: 252 steps per
# year, but never fewer than 100 steps to any expiry.
MC_STEPS_PER_YEAR = 252
MC_MIN_STEPS      = 100


# ── Parameter dataclass ────────────────────────────────────────────────────────

@dataclass
class QuinticOUParams:
    """8-field two-factor Quintic OU parameter set (α5 is scale-redundant)."""
    lam_x:   float   # fast OU speed
    lam_y:   float   # slow OU speed
    theta:   float   # factor weight  (Z = θX + (1-θ)Y)
    alpha0:  float   # polynomial constant
    alpha1:  float   # linear coefficient
    alpha3:  float   # cubic coefficient
    alpha5:  float   # quintic coefficient
    epsilon: float   # corr(B, W)  ε ∈ [−1, 1]

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

    def to_dict(self) -> dict:
        return {"lam_x": self.lam_x, "lam_y": self.lam_y, "theta": self.theta,
                "alpha0": self.alpha0, "alpha1": self.alpha1,
                "alpha3": self.alpha3, "alpha5": self.alpha5,
                "epsilon": self.epsilon}

    def alpha_vec(self) -> np.ndarray:
        """Full 6-element coefficient vector [α0, α1, 0, α3, 0, α5]."""
        return np.array([self.alpha0, self.alpha1, 0.0,
                         self.alpha3, 0.0, self.alpha5], dtype=float)

    def normalised(self) -> "QuinticOUParams":
        """Same model with p rescaled so that α5 = 1 (requires α5 ≠ 0)."""
        if self.alpha5 == 0.0:
            raise ValueError("cannot normalise: alpha5 == 0")
        s = self.alpha5
        return QuinticOUParams(self.lam_x, self.lam_y, self.theta,
                               self.alpha0 / s, self.alpha1 / s, self.alpha3 / s,
                               1.0, self.epsilon)

    def __repr__(self) -> str:
        return (f"QuinticOUParams(λx={self.lam_x:.4f}, λy={self.lam_y:.4f}, "
                f"θ={self.theta:.4f}, α0={self.alpha0:.6g}, α1={self.alpha1:.6g}, "
                f"α3={self.alpha3:.6g}, α5={self.alpha5:.6g}, ε={self.epsilon:.4f})")


# Default parameter point: the SPX+VIX fit of Abi Jaber & Li (2025), Section 3.2
# (23 Oct 2017, α2 = α4 = 0, α5 = 1). Used as a documented starting point and
# as the test fixture for "paper-like" dynamics.
QUINTIC_DEFAULTS = QuinticOUParams(
    lam_x=31.8, lam_y=0.659, theta=0.964,
    alpha0=0.0004, alpha1=0.0046, alpha3=0.0096, alpha5=1.0,
    epsilon=-0.765,
)

# Admissible set enforced by the calibrator, in raw parameter units
# (lower, upper) per field. α5 is pinned to 1 (scale normalisation); the other
# coefficients are non-negative; ε ≤ 0. The calibrator searches α0, α1, α3 in
# units of the stationary std of Z (see quintic_calibrator), so the α bounds
# here are only sign constraints.
QUINTIC_BOUNDS = [
    (1.0,    300.0),     # lam_x  (Richardson step convergence tested at 31.8 and 300)
    (0.01,   20.0),      # lam_y
    (0.0,    1.0),       # theta  (convex combination of the two factors)
    (0.0,    math.inf),  # alpha0
    (0.0,    math.inf),  # alpha1
    (0.0,    math.inf),  # alpha3
    (1.0,    1.0),       # alpha5 (normalisation, Abi Jaber & Li Remark 2.1)
    (-0.99,  0.0),       # epsilon (martingale condition ε ≤ 0)
]


def admissibility_violations(params: QuinticOUParams) -> list:
    """
    List the ways `params` leaves the set in which the model is well posed as
    published: S a true martingale (α5 > 0, ε ≤ 0), p increasing so that ε is
    the sign of the spot-vol correlation (α0, α1, α3 ≥ 0), θ ∈ [0, 1].
    Empty list ⇒ admissible.
    """
    out = []
    if not params.alpha5 > 0.0:
        out.append(f"alpha5={params.alpha5:.4g} <= 0: martingale property not guaranteed")
    if params.epsilon > 0.0:
        out.append(f"epsilon={params.epsilon:.4g} > 0: martingale property not guaranteed")
    for name in ("alpha0", "alpha1", "alpha3"):
        v = getattr(params, name)
        if v < 0.0:
            out.append(f"{name}={v:.4g} < 0: p not monotone, ε is not the leverage sign")
    if not 0.0 <= params.theta <= 1.0:
        out.append(f"theta={params.theta:.4g} outside [0, 1]")
    if params.lam_x <= 0.0 or params.lam_y <= 0.0:
        out.append("mean-reversion speeds must be positive")
    if abs(params.epsilon) > 1.0:
        out.append(f"|epsilon|={abs(params.epsilon):.4g} > 1")
    return out


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
    z2 = z * z
    return a0 + z * (a1 + z2 * (a3 + a5 * z2))


# ── Gaussian moments ───────────────────────────────────────────────────────────

def gaussian_moments(max_n: int, mu, sigma2) -> np.ndarray:
    """
    Compute E[Z^k] for k = 0, …, max_n when Z ~ N(mu, sigma2).

    Recurrence:  E[Z^n] = mu · E[Z^{n-1}] + (n-1) · σ² · E[Z^{n-2}]

    `mu` and `sigma2` may be broadcastable arrays; the result then has shape
    (max_n + 1, *broadcast_shape).
    """
    mu = np.asarray(mu, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    shape = np.broadcast(mu, sigma2).shape
    m = np.zeros((max_n + 1,) + shape, dtype=float)
    m[0] = 1.0
    if max_n >= 1:
        m[1] = mu
    for k in range(2, max_n + 1):
        m[k] = mu * m[k - 1] + (k - 1) * sigma2 * m[k - 2]
    return m


def _poly_square_coeffs(alpha: np.ndarray) -> np.ndarray:
    """Coefficients c_k of p(z)² = Σ_k c_k z^k (Cauchy product, length 11)."""
    a = np.asarray(alpha, dtype=float)
    return np.convolve(a, a)


def ep_squared(mu, sigma2, alpha: np.ndarray):
    """
    Compute E[p(Z)²] analytically for Z ~ N(mu, sigma2).

    p(z) = Σ_k alpha[k] · z^k  (alpha is the 6-element vector).
    Uses the Cauchy product of the coefficients and the Gaussian moment
    recurrence. Scalar inputs return a float; array inputs broadcast.
    """
    c = _poly_square_coeffs(alpha)
    m = gaussian_moments(len(c) - 1, mu, sigma2)
    out = np.tensordot(c, m, axes=(0, 0))
    return float(out) if np.ndim(out) == 0 else out


# ── OU variance / covariance functions ────────────────────────────────────────

def _one_minus_exp_over(a, lam):
    """(1 − e^{−a}) / lam, computed with expm1 for small a."""
    return -np.expm1(-a) / lam


def var_Z(t, lam_x: float, lam_y: float, theta: float):
    """
    Var[Z_t] = Var[θ X_t + (1-θ) Y_t],  X_0 = Y_0 = 0,  same BM.

    = θ² · Var[X_t]  +  (1-θ)² · Var[Y_t]  +  2θ(1-θ) · Cov[X_t, Y_t]

    Scalar t returns a float (0 for t ≤ 0); array t returns an array.
    """
    t_arr = np.maximum(np.asarray(t, dtype=float), 0.0)
    vx  = _one_minus_exp_over(2.0 * lam_x * t_arr, 2.0 * lam_x)
    vy  = _one_minus_exp_over(2.0 * lam_y * t_arr, 2.0 * lam_y)
    cxy = _one_minus_exp_over((lam_x + lam_y) * t_arr, lam_x + lam_y)
    out = theta**2 * vx + (1.0 - theta)**2 * vy + 2.0 * theta * (1.0 - theta) * cxy
    return float(out) if np.ndim(out) == 0 else out


def stationary_var_Z(lam_x: float, lam_y: float, theta: float) -> float:
    """lim_{t→∞} Var[Z_t] = θ²/(2λx) + (1-θ)²/(2λy) + 2θ(1-θ)/(λx+λy)."""
    return (theta**2 / (2.0 * lam_x) + (1.0 - theta)**2 / (2.0 * lam_y)
            + 2.0 * theta * (1.0 - theta) / (lam_x + lam_y))


def _cond_var_Z(dt: float, lam_x: float, lam_y: float, theta: float) -> float:
    """
    Var[Z_{t+dt} | X_t, Y_t].

    The conditional residual θ∫e^{-λx(t+dt-s)}dW + (1-θ)∫e^{-λy(t+dt-s)}dW over
    (t, t+dt] has the law of Z_dt started from zero, so this equals var_Z(dt).
    """
    return var_Z(dt, lam_x, lam_y, theta)


def _cond_var_components(dt: float, lam_x: float, lam_y: float
                          ) -> Tuple[float, float, float]:
    """Return (Var_X_cond, Var_Y_cond, Cov_XY_cond) for one OU step."""
    if dt <= 0.0:
        return 0.0, 0.0, 0.0
    vx  = float(_one_minus_exp_over(2.0 * lam_x * dt, 2.0 * lam_x))
    vy  = float(_one_minus_exp_over(2.0 * lam_y * dt, 2.0 * lam_y))
    cxy = float(_one_minus_exp_over((lam_x + lam_y) * dt, lam_x + lam_y))
    return vx, vy, cxy


def _step_cholesky(h: float, lam_x: float, lam_y: float) -> np.ndarray:
    """
    Lower Cholesky factor of Cov(ΔW, I_X, I_Y) over a step of length h, where
    ΔW = ∫dW, I_X = ∫e^{-λx(t+h-s)}dW, I_Y = ∫e^{-λy(t+h-s)}dW:

        Var ΔW = h,  Cov(ΔW, I_X) = (1-e^{-λx h})/λx,  Var I_X = (1-e^{-2λx h})/(2λx),
        Cov(I_X, I_Y) = (1-e^{-(λx+λy)h})/(λx+λy),  and likewise for Y.

    The matrix is nearly singular when λh ≪ 1 (I ≈ ΔW), so negative pivots
    produced by round-off are clipped to zero instead of failing.
    """
    vx, vy, cxy = _cond_var_components(h, lam_x, lam_y)
    cwx = float(_one_minus_exp_over(lam_x * h, lam_x))
    cwy = float(_one_minus_exp_over(lam_y * h, lam_y))
    L = np.zeros((3, 3))
    L[0, 0] = math.sqrt(h)
    L[1, 0] = cwx / L[0, 0]
    L[1, 1] = math.sqrt(max(vx - L[1, 0]**2, 0.0))
    L[2, 0] = cwy / L[0, 0]
    L[2, 1] = (cxy - L[2, 0] * L[1, 0]) / L[1, 1] if L[1, 1] > 0.0 else 0.0
    L[2, 2] = math.sqrt(max(vy - L[2, 0]**2 - L[2, 1]**2, 0.0))
    return L


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


def _fwd_var_breakpoints(fwd_var_func) -> np.ndarray:
    """Jump times of ξ_0 if the curve exposes them (piecewise-constant curves)."""
    bp = getattr(fwd_var_func, "breakpoints", None)
    if bp is None:
        return np.empty(0)
    return np.asarray(bp() if callable(bp) else bp, dtype=float)


def _gl_nodes(a: float, b: float, n: int, breakpoints: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composite Gauss-Legendre rule on [a, b], with one n-point panel per smooth
    piece of the integrand (panels split at the breakpoints inside (a, b)).
    """
    x, w = np.polynomial.legendre.leggauss(n)
    inner = [float(p) for p in np.sort(breakpoints) if a < p < b]
    edges = [a] + inner + [b]
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (x + 1.0) * (hi - lo) + lo)
        weights.append(0.5 * w * (hi - lo))
    return np.concatenate(nodes), np.concatenate(weights)


def _step_mean_fwd_var(fwd_var_func, t_grid: np.ndarray) -> np.ndarray:
    """
    Average of ξ_0 over each step [t_k, t_{k+1}]. Exact when the curve exposes
    its integral (`integrated(t)` = ∫_0^t ξ_0); otherwise the mid-point value.
    Using the step average makes E[Σ σ_k² h_k] = ∫_0^T ξ_0 hold exactly on any
    grid, including steps that straddle a jump of ξ_0.
    """
    integ = getattr(fwd_var_func, "integrated", None)
    h = np.diff(t_grid)
    if integ is not None:
        cum = np.array([integ(float(t)) for t in t_grid])
        return np.diff(cum) / h
    mids = 0.5 * (t_grid[:-1] + t_grid[1:])
    return np.array([fwd_var_func(float(t)) for t in mids])


# ── VIX² computation ──────────────────────────────────────────────────────────

def vix2_coefficients(
    T: float,
    params: QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_inner: int = _N_QUAD_INNER,
) -> np.ndarray:
    """
    Coefficients β of the bivariate polynomial VIX²_T = Σ_{i+l≤10} β[i, l] X_T^i Y_T^l.

    Given F_T, Z_s ~ N(μ, v) with μ = A(τ)X_T + B(τ)Y_T, A = θe^{-λx τ},
    B = (1-θ)e^{-λy τ}, v = var_Z(τ), τ = s - T. Writing p(z)² = Σ_k c_k z^k
    and G ~ N(0, v),
        E[p(Z_s)² | F_T] = Σ_j μ^j d_j(τ),   d_j = Σ_{k≥j} c_k C(k, j) E[G^{k-j}],
    and expanding μ^j binomially gives β after the time integral
        VIX² = (100²/Δ) ∫_T^{T+Δ} ξ_0(s) E[p(Z_s)² | F_T] / E[p(Z_s)²] ds
    (Gauss-Legendre, one panel per smooth piece of ξ_0). Computed once per
    expiry, so evaluating VIX at many cubature nodes is cheap.
    """
    p = params
    c = _poly_square_coeffs(p.alpha_vec())                 # degree 10
    deg = len(c) - 1
    s, ws = _gl_nodes(T, T + _VIX_DELTA, n_inner, _fwd_var_breakpoints(fwd_var_func))
    tau = s - T
    xi = np.array([fwd_var_func(float(si)) for si in s])
    ep_unc = np.tensordot(c, gaussian_moments(deg, 0.0, var_Z(s, p.lam_x, p.lam_y, p.theta)),
                          axes=(0, 0))
    wgt = np.where(ep_unc > 0.0,
                   (_VIX_SCALE**2 / _VIX_DELTA) * ws * xi / np.where(ep_unc > 0.0, ep_unc, 1.0),
                   0.0)
    g = gaussian_moments(deg, 0.0, var_Z(tau, p.lam_x, p.lam_y, p.theta))   # (deg+1, nodes)
    binom = np.array([[math.comb(k, j) for j in range(deg + 1)] for k in range(deg + 1)],
                     dtype=float)                           # binom[k, j] = C(k, j)
    # d[j, s] = Σ_{k≥j} c_k C(k,j) g_{k-j}(s)
    d = np.zeros((deg + 1, len(s)))
    for j in range(deg + 1):
        for k in range(j, deg + 1):
            if c[k] != 0.0:
                d[j] += c[k] * binom[k, j] * g[k - j]
    A = p.theta * np.exp(-p.lam_x * tau)
    B = (1.0 - p.theta) * np.exp(-p.lam_y * tau)
    beta = np.zeros((deg + 1, deg + 1))
    for j in range(deg + 1):
        dj = wgt * d[j]
        for i in range(j + 1):
            beta[i, j - i] = binom[j, i] * float(np.sum(dj * A**i * B**(j - i)))
    return beta


def _eval_bivariate(beta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Σ β[i, l] x^i y^l for arrays x, y of the same shape (Horner in y, then x)."""
    deg = beta.shape[0] - 1
    acc = np.zeros(np.broadcast(x, y).shape)
    for i in range(deg, -1, -1):
        row = np.full(acc.shape, beta[i, deg - i])
        for l in range(deg - i - 1, -1, -1):
            row = row * y + beta[i, l]
        acc = acc * x + row
    return acc


def _vix2_on_grid(
    T: float,
    xt: np.ndarray,
    yt: np.ndarray,
    params: QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_inner: int,
) -> np.ndarray:
    """Vectorised VIX²(T; X_T, Y_T) for arrays xt, yt (same shape)."""
    beta = vix2_coefficients(T, params, fwd_var_func, n_inner)
    return _eval_bivariate(beta, np.asarray(xt, dtype=float), np.asarray(yt, dtype=float))


def compute_vix2(
    T:            float,
    xt,
    yt,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_quad:       int = _N_QUAD_INNER,
):
    """
    Compute VIX²(T; X_T=xt, Y_T=yt) via Gauss-Legendre integration.

    VIX²(T) = (100² / Δ) ∫_T^{T+Δ} ξ_0(s) · E[p(Z_s)² | X_T, Y_T] / E[p(Z_s)²] ds

    For each node s ∈ [T, T+Δ]:
      • Z_s | F_T ~ N(μ_Z(s), σ²_Z_cond(s))
      • μ_Z(s)  = θ·xt·e^{-λx(s-T)} + (1-θ)·yt·e^{-λy(s-T)}
      • σ²_cond = Var[Z_{s-T}] (same-BM OU formula starting from zero residual)

    The time integral uses an n_quad-point GL panel per smooth piece of ξ_0.
    Scalar (xt, yt) return a float; arrays broadcast.
    """
    out = _vix2_on_grid(T, np.asarray(xt, dtype=float), np.asarray(yt, dtype=float),
                        params, fwd_var_func, n_quad)
    return float(out) if np.ndim(out) == 0 else out


# ── 2-D cubature for the (X_T, Y_T) marginal ──────────────────────────────────

_CUBATURE_RADIUS = 8.0     # standard-normal radius kept (tail mass ~ e^{-32})


@lru_cache(maxsize=16)
def _std_normal_plane_rule(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Uniform ("trapezoidal") rule for the standard bivariate normal: nodes on the
    lattice hℤ² with h = 2R/(n-1), R = _CUBATURE_RADIUS, kept inside the disk
    |z| ≤ R, weights ∝ φ(z1)φ(z2) normalised to sum to 1 (cached per n).

    For smooth integrands this rule converges exponentially fast in 1/h; for
    the kinked option payoff its error is O(h²) with a small constant, which
    makes it far more accurate than a product Gauss-Hermite rule of similar
    size (2026-03-24 VIX set: 0.002 vp max IV error at n = 161 vs a 0.01-step
    reference, against 0.14–0.8 vp for 128²–48² Gauss-Hermite nodes).
    """
    z = np.linspace(-_CUBATURE_RADIUS, _CUBATURE_RADIUS, n)
    z1, z2 = np.meshgrid(z, z, indexing="ij")
    keep = z1**2 + z2**2 <= _CUBATURE_RADIUS**2 + 1e-12
    z1, z2 = z1[keep], z2[keep]
    w = np.exp(-0.5 * (z1**2 + z2**2))
    w /= w.sum()
    for a in (z1, z2, w):
        a.setflags(write=False)
    return z1, z2, w


def _plane_cubature_2d(
    n:      int,
    var_x:  float,
    var_y:  float,
    cov_xy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cubature nodes and weights for BivNorm(0, [[var_x, cov_xy], [cov_xy, var_y]]):
    the standard-normal plane rule mapped through the Cholesky factor.
    Returns (x_pts, y_pts, weights); weights sum to 1.
    """
    z1, z2, w = _std_normal_plane_rule(int(n))
    L11 = math.sqrt(max(var_x, 1e-300))
    L21 = cov_xy / L11
    L22 = math.sqrt(max(var_y - L21**2, 0.0))
    return L11 * z1, L21 * z1 + L22 * z2, w


def vix_at_nodes(
    T:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    VIX_T at the cubature nodes of the (X_T, Y_T) law, and their weights.

    VIX²_T is evaluated from its polynomial coefficients (vix2_coefficients),
    so every VIX derivative at expiry T is Σ_i w_i Φ(VIX_i): futures (Φ(v) = v)
    and all strikes reuse the same node values. n_outer = lattice points per
    dimension of the plane rule (see _std_normal_plane_rule).
    """
    p = params
    var_x  = float(_one_minus_exp_over(2.0 * p.lam_x * T, 2.0 * p.lam_x))
    var_y  = float(_one_minus_exp_over(2.0 * p.lam_y * T, 2.0 * p.lam_y))
    cov_xy = float(_one_minus_exp_over((p.lam_x + p.lam_y) * T, p.lam_x + p.lam_y))
    xp, yp, wp = _plane_cubature_2d(n_outer, var_x, var_y, cov_xy)
    vix2 = _vix2_on_grid(T, xp, yp, params, fwd_var_func, n_inner)
    return np.sqrt(np.maximum(vix2, 0.0)), wp


# ── VIX futures & options pricing ─────────────────────────────────────────────

def price_vix_futures(
    T:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> float:
    """
    VIX futures price = E[VIX_T] = E[sqrt(VIX²_T)] (no discounting: a future).

    Integrates over the bivariate-Normal marginal of (X_T, Y_T) with the plane
    cubature of vix_at_nodes.
    """
    vix, w = vix_at_nodes(T, params, fwd_var_func, n_outer, n_inner)
    return float(np.dot(w, vix))


def vix_option_prices(
    T:            float,
    strikes,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Model VIX forward and UNDISCOUNTED call and put premia for many strikes.

    Returns (F, calls, puts) with F = E[VIX_T], calls = E[(VIX_T − K)+],
    puts = E[(K − VIX_T)+]. Put-call parity calls − puts = F − K holds exactly
    on the cubature. Multiply by e^{−rT} for present values.
    """
    vix, w = vix_at_nodes(T, params, fwd_var_func, n_outer, n_inner)
    K = np.atleast_1d(np.asarray(strikes, dtype=float))
    F = float(np.dot(w, vix))
    calls = np.maximum(vix[:, None] - K[None, :], 0.0).T @ w
    puts  = np.maximum(K[None, :] - vix[:, None], 0.0).T @ w
    return F, calls, puts


def price_vix_option(
    T:            float,
    K_vix:        float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_outer:      int = _N_QUAD_OUTER,
    n_inner:      int = _N_QUAD_INNER,
) -> float:
    """
    UNDISCOUNTED VIX call premium E[max(VIX_T − K, 0)] (plane cubature).

    This is the forward premium, not a present value: invert it with
    `black76_implied_vol(..., discount=1.0)` or `vix_option_implied_vol(..., r=0)`.

    Parameters
    ----------
    K_vix : strike in VIX-point units (e.g. 20.0 means VIX = 20)
    """
    _, calls, _ = vix_option_prices(T, [K_vix], params, fwd_var_func, n_outer, n_inner)
    return float(calls[0])


# ── Black-76 ──────────────────────────────────────────────────────────────────

def black76_price(F, K, T, sigma, is_call=True, discount=1.0):
    """
    Black-76 price discount·(F N(d1) − K N(d2)) for calls, and the put analogue.
    All arguments broadcast. discount = 1 gives the undiscounted (forward) price.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.where(sd > 0, (np.log(F / K) + 0.5 * sd**2) / np.where(sd > 0, sd, 1.0), 0.0)
    d2 = d1 - sd
    call = F * ndtr(d1) - K * ndtr(d2)
    put  = K * ndtr(-d2) - F * ndtr(-d1)
    intrinsic_c = np.maximum(F - K, 0.0)
    intrinsic_p = np.maximum(K - F, 0.0)
    call = np.where(sd > 0, call, intrinsic_c)
    put  = np.where(sd > 0, put, intrinsic_p)
    return np.asarray(discount, dtype=float) * np.where(is_call, call, put)


def black76_implied_vol(price, F, K, T, is_call=True, discount=1.0,
                        lo: float = 1e-6, hi: float = 10.0,
                        n_iter: int = 64) -> np.ndarray:
    """
    Vectorised Black-76 implied volatility by bisection.

    `price` is the present value at `discount` (pass discount = 1 for an
    undiscounted forward premium). The option is converted to its out-of-the-
    money side through put-call parity before inversion, so deep-ITM inputs do
    not lose precision. 64 halvings of [lo, hi] resolve σ to ~1e-18. Prices
    outside the no-arbitrage band (intrinsic, F or K), or implying a vol
    outside [lo, hi], return NaN.
    """
    price, F, K, T, is_call, discount = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(F, dtype=float),
        np.asarray(K, dtype=float), np.asarray(T, dtype=float),
        np.asarray(is_call, dtype=bool), np.asarray(discount, dtype=float))
    undisc = price / discount
    # OTM side: put if K < F else call
    otm_call = K >= F
    otm_price = np.where(is_call == otm_call, undisc,
                         np.where(is_call, undisc - (F - K), undisc + (F - K)))
    upper = np.where(otm_call, F, K)
    ok = (np.isfinite(otm_price) & (otm_price > 0.0) & (otm_price < upper)
          & (T > 0) & (F > 0) & (K > 0))
    a = np.full(otm_price.shape, lo)
    b = np.full(otm_price.shape, hi)
    f_hi = black76_price(F, K, T, b, otm_call) - otm_price
    ok &= f_hi >= 0.0
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        above = black76_price(F, K, T, m, otm_call) > otm_price
        b = np.where(above, m, b)
        a = np.where(above, a, m)
    iv = 0.5 * (a + b)
    ok &= iv > lo * 1.0001
    out = np.where(ok, iv, np.nan)
    return out if out.ndim else float(out)


def vix_option_implied_vol(
    T:            float,
    K_vix:        float,
    F_vix:        float,    # VIX forward (futures) price
    price:        float,    # call price in VIX pts, PRESENT VALUE at rate r
    r:            float = 0.045,
) -> Optional[float]:
    """
    Invert the Black-76 formula to get a VIX call implied vol.

    `price` must be discounted at `r` (a present value). Model premia from
    `price_vix_option` are undiscounted: pass r = 0 for them.
    """
    if T <= 0 or F_vix <= 0:
        return None
    iv = black76_implied_vol(price, F_vix, K_vix, T, True, math.exp(-r * T))
    return None if not np.isfinite(iv) else float(iv)


# ── SPX Monte Carlo pricing ────────────────────────────────────────────────────

def mc_time_grid(
    expiries: Sequence[float],
    steps_per_year: int = MC_STEPS_PER_YEAR,
    min_steps:      int = MC_MIN_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared simulation grid on [0, max(expiries)] containing every expiry.

    On (T_{i-1}, T_i] the step is h_i ≤ min(1/steps_per_year, T_i/min_steps),
    so each expiry T is reached in at least max(min_steps, steps_per_year·T)
    steps and no step is longer than 1/steps_per_year.

    Returns (t_grid, expiry_index) with t_grid[expiry_index[j]] == sorted
    unique expiries[j].
    """
    Ts = np.unique(np.asarray(expiries, dtype=float))
    if Ts.size == 0 or Ts[0] <= 0:
        raise ValueError("expiries must be positive")
    pieces = [np.array([0.0])]
    idx = []
    prev = 0.0
    n_total = 0
    for T in Ts:
        h_max = min(1.0 / steps_per_year, T / min_steps)
        n = max(1, int(math.ceil((T - prev) / h_max - 1e-9)))
        pieces.append(prev + (T - prev) * np.arange(1, n + 1) / n)
        n_total += n
        idx.append(n_total)
        prev = T
    t = np.concatenate(pieces)
    t[np.asarray(idx)] = Ts          # exact expiry nodes
    return t, np.asarray(idx)


def _halve_steps(t_grid: np.ndarray) -> np.ndarray:
    """Insert the mid-point of every step: the fine grid of a Richardson pair."""
    fine = np.empty(2 * len(t_grid) - 1)
    fine[0::2] = t_grid
    fine[1::2] = 0.5 * (t_grid[:-1] + t_grid[1:])
    return fine


def _g0_on_grid(params: QuinticOUParams, fwd_var_func, t_grid: np.ndarray) -> np.ndarray:
    """
    g0_k = sqrt(mean ξ_0 over [t_k, t_{k+1}] / E[p(Z_{t_k})²]) for every step,
    so that E[σ_k²] equals the step-average forward variance exactly.
    (At t = 0, Z_0 = 0 and E[p(Z_0)²] = α0²; α0 = 0 gives σ_0 = 0.)
    """
    p = params
    xi_bar = _step_mean_fwd_var(fwd_var_func, t_grid)
    ep2 = np.atleast_1d(ep_squared(0.0, var_Z(t_grid[:-1], p.lam_x, p.lam_y, p.theta),
                                   p.alpha_vec()))
    safe = np.where(ep2 > 0.0, ep2, 1.0)
    return np.where(ep2 > 0.0, np.sqrt(np.maximum(xi_bar, 0.0) / safe), 0.0)


_CRN_CACHE: dict = {}
_CRN_CACHE_MAX_BYTES = 128 * 2**20


def _crn_normals(seed: int, n_steps: int, k: int, n: int) -> Optional[np.ndarray]:
    """
    Standard normals of shape (n_steps, k, n) for seed, cached (one entry) so
    that repeated evaluations with common random numbers — every loss call of a
    calibration — skip the generator. Drawing the block at once yields exactly
    the same numbers as drawing (k, n) per step from default_rng(seed), so the
    cache never changes results. Returns None (draw per step) above
    _CRN_CACHE_MAX_BYTES.
    """
    if n_steps * k * n * 8 > _CRN_CACHE_MAX_BYTES:
        return None
    key = (int(seed), int(n_steps), int(k), int(n))
    arr = _CRN_CACHE.get(key)
    if arr is None:
        _CRN_CACHE.clear()
        arr = np.random.default_rng(seed).standard_normal((n_steps, k, n))
        arr.setflags(write=False)
        _CRN_CACHE[key] = arr
    return arr


def _simulate_integrals(
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    t_grid:       np.ndarray,
    record_idx:   np.ndarray,
    n_pairs:      int,
    seed:         int,
    with_perp:    bool = False,
    coarse:       bool = False,
) -> dict:
    """
    Left-point Itô simulation of the path integrals that determine S_T.

    For k = 0 … N-1, with h = t_{k+1} − t_k:
        σ_k     = g0_k · p(θ X_k + (1-θ) Y_k)          (pre-step state: adapted)
        (ΔW, I_X, I_Y) ~ exact joint Gaussian (see _step_cholesky)
        A      += σ_k ΔW                 → ∫σ dW
        V      += σ_k² h                 → ∫σ² dt
        P      += σ_k √h Z⊥  (optional)  → ∫σ dW⊥
        X       = e^{-λx h} X + I_X,   Y = e^{-λy h} Y + I_Y

    g0_k² = mean ξ_0 over the step / E[p(Z_{t_k})²], so E[Σσ_k²h] = ∫ξ_0 exactly.
    Since σ_k is F_{t_k}-measurable and ΔW, Z⊥ are independent of F_{t_k},
    E[exp(ε A + √(1-ε²) P − ½V)] = 1 exactly for the discrete scheme.

    coarse=True additionally runs the same scheme on the grid t_grid[::2]
    (every coarse step = two fine steps) driven by the SAME Brownian path —
    the coarse increments are sums of fine ones, so their joint law is again
    exact — and returns its integrals as 'Ac', 'Vc', for the Richardson
    extrapolation in spx_forward_prices_mc, which cancels the O(h) weak error.

    Antithetic: each step draws normals for n_pairs paths; paths n_pairs..2n_pairs
    use the same draws negated (true mirror paths). Mirror paths only reduce
    variance when p has an even part: for odd p, σ(−Z) = −σ(Z) and the mirror
    path has the same log-price, so the conditional estimator and the
    martingale correction do the work.

    Returns arrays of shape (len(record_idx), 2·n_pairs) for keys 'A', 'V'
    (and 'P' when with_perp; 'Ac', 'Vc' when coarse).
    """
    p = params
    a = p.alpha_vec()
    h_all = np.diff(t_grid)
    n_steps = len(h_all)
    if coarse and (n_steps % 2 or np.any(np.asarray(record_idx) % 2)):
        raise ValueError("coarse coupling needs an even number of steps to every record node")
    g0 = _g0_on_grid(params, fwd_var_func, t_grid)
    g0c = _g0_on_grid(params, fwd_var_func, t_grid[::2]) if coarse else None

    n = n_pairs
    normals = _crn_normals(seed, n_steps, 4 if with_perp else 3, n)
    rng = None if normals is not None else np.random.default_rng(seed)
    X = np.zeros(2 * n)
    Y = np.zeros(2 * n)
    A = np.zeros(2 * n)
    V = np.zeros(2 * n)
    P = np.zeros(2 * n) if with_perp else None
    Ac = np.zeros(2 * n) if coarse else None
    Vc = np.zeros(2 * n) if coarse else None
    dW, IX, IY = np.empty(2 * n), np.empty(2 * n), np.empty(2 * n)
    dP = np.empty(2 * n) if with_perp else None
    tmp = np.empty(2 * n)
    sig_c = np.empty(2 * n) if coarse else None
    a0, a1, _, a3, _, a5 = a
    rec = {j: i for i, j in enumerate(record_idx)}
    keys = ["A", "V"] + (["P"] if with_perp else []) + (["Ac", "Vc"] if coarse else [])
    out = {k: np.empty((len(record_idx), 2 * n)) for k in keys}

    def mirror(dst, half):
        dst[:n] = half
        np.negative(half, out=dst[n:])

    chol_cache = {}
    for k in range(n_steps):
        h = float(h_all[k])
        key = round(h, 15)
        L = chol_cache.get(key)
        if L is None:
            L = _step_cholesky(h, p.lam_x, p.lam_y)
            chol_cache[key] = L
        z = normals[k] if normals is not None else rng.standard_normal((4 if with_perp else 3, n))
        mirror(dW, L[0, 0] * z[0])
        mirror(IX, L[1, 0] * z[0] + L[1, 1] * z[1])
        mirror(IY, L[2, 0] * z[0] + L[2, 1] * z[1] + L[2, 2] * z[2])

        # σ_k = g0_k p(Z_k) on the pre-step state (Horner: a0 + z(a1 + z²(a3 + a5 z²)))
        Zk = p.theta * X + (1.0 - p.theta) * Y
        np.multiply(Zk, Zk, out=tmp)
        pz = a0 + Zk * (a1 + tmp * (a3 + a5 * tmp))
        sigma = g0[k] * pz
        A += sigma * dW
        V += (sigma * sigma) * h
        if with_perp:
            mirror(dP, math.sqrt(h) * z[3])
            P += sigma * dP
        if coarse:
            if k % 2 == 0:
                np.multiply(pz, g0c[k // 2], out=sig_c)   # frozen for this coarse step
            Ac += sig_c * dW
            Vc += (sig_c * sig_c) * h

        X *= math.exp(-p.lam_x * h)
        X += IX
        Y *= math.exp(-p.lam_y * h)
        Y += IY

        j = rec.get(k + 1)
        if j is not None:
            out["A"][j] = A
            out["V"][j] = V
            if with_perp:
                out["P"][j] = P
            if coarse:
                out["Ac"][j] = Ac
                out["Vc"][j] = Vc
    return out


def _pair_mean_se(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over antithetic pairs (rows i and i + n)."""
    n = x.shape[0] // 2
    pair = 0.5 * (x[:n] + x[n:])
    se = pair.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.full(pair.shape[1:], np.nan)
    return pair.mean(axis=0), se


def _mixing_otm_premia(A, V, eps, F, K, otm_call):
    """
    Conditional (Romano–Touzi) OTM premia per path given the W-integrals:
    S_T | W ~ F·M1·LN(variance (1−ε²)V), M1 = exp(εA − ½ε²V).

    M1 is rescaled by its sample mean (empirical martingale correction), so the
    simulated law of S_T is a positive mixture of lognormals with mean exactly
    F: every premium is then an expectation of a non-negative payoff under a
    measure with the right forward, i.e. free of static arbitrage (no
    negative or sub-intrinsic prices, exact put-call parity). The correction
    is a ratio estimator: O(1/N) bias, ~ the variance of an M1 control variate.
    Returns per-path premia of shape (paths, strikes).
    """
    M1 = np.exp(eps * A - 0.5 * eps**2 * V)
    M1 = M1 / M1.mean()
    Fm = M1[:, None] * F[None, :]
    sd = np.broadcast_to(np.sqrt(max(1.0 - eps**2, 0.0) * V)[:, None], Fm.shape)
    pos = sd > 0.0
    sd_safe = np.where(pos, sd, 1.0)
    Kb = K[None, :]
    d1 = (np.log(Fm / Kb) + 0.5 * sd_safe**2) / sd_safe
    d2 = d1 - sd_safe
    call = Fm * ndtr(d1) - Kb * ndtr(d2)
    put = Kb * ndtr(-d2) - Fm * ndtr(-d1)
    prem = np.where(otm_call[None, :], call, put)
    intrinsic = np.where(otm_call[None, :], np.maximum(Fm - Kb, 0.0), np.maximum(Kb - Fm, 0.0))
    return np.where(pos, prem, intrinsic)


def spx_forward_prices_mc(
    expiries:     Sequence[float],
    strikes:      Sequence[float],
    forwards:     Sequence[float],
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 20_000,
    seed:         int = 42,
    steps_per_year: int = MC_STEPS_PER_YEAR,
    min_steps:    int = MC_MIN_STEPS,
    method:       str = "conditional",
    richardson:   bool = True,
) -> dict:
    """
    UNDISCOUNTED SPX call and put premia E[(S_T − K)+], E[(K − S_T)+] for
    per-row (T, K, F) triples, with S_T = F·M_T and E[M_T] = 1.

    All expiries share one simulation on the grid of `mc_time_grid`
    (≥ steps_per_year steps per year, ≥ min_steps steps to every expiry),
    with n_paths/2 antithetic pairs.

    method = "conditional" (default): conditioning on W, log S_T is Gaussian
        with mean ε∫σdW − ½∫σ²dt and variance (1−ε²)∫σ²dt (Romano–Touzi), so
        each path contributes a Black-76 premium on F·M1, M1 = exp(ε∫σdW −
        ½ε²∫σ²dt), rescaled to sample mean 1 (see _mixing_otm_premia). The
        out-of-the-money side is priced; the other follows from parity.
        richardson=True (default) also runs the scheme on the half-step grid
        with the same Brownian path and extrapolates the implied vol,
        σ = 2σ(h/2) − σ(h): the left-point scheme's O(h) weak error cancels
        (~1 vp in the 1y wings at 252 steps/yr otherwise, from the fast
        factor). Prices are then Black-76 at the extrapolated vol.
    method = "plain": simulate W⊥ too and average the raw payoffs (antithetic
        pairs only, no correction, no extrapolation) — an independent,
        unbiased check of the conditional estimator.

    Returns dict with 'call', 'put', 'call_se', 'put_se' arrays and 'iv'
    (Black-76 vol of the row), all in row order. Standard errors come from
    the antithetic pair averages (for the extrapolated estimator, of the
    per-pair combination 2·fine − coarse).
    """
    if method not in ("conditional", "plain"):
        raise ValueError(f"unknown method {method!r}")
    T_rows = np.asarray(expiries, dtype=float)
    K_rows = np.asarray(strikes, dtype=float)
    F_rows = np.asarray(forwards, dtype=float)
    n_pairs = max(2, int(n_paths) // 2)
    t_grid, idx = mc_time_grid(T_rows, steps_per_year, min_steps)
    Ts = t_grid[idx]
    extrapolate = richardson and method == "conditional"
    if extrapolate:
        t_grid, idx = _halve_steps(t_grid), 2 * idx
    sims = _simulate_integrals(params, fwd_var_func, t_grid, idx, n_pairs, seed,
                               with_perp=(method == "plain"), coarse=extrapolate)
    eps = float(params.epsilon)

    out = {k: np.full(len(K_rows), np.nan) for k in ("call", "put", "call_se", "put_se", "iv")}
    for j, T in enumerate(Ts):
        mask = T_rows == T
        if not mask.any():
            continue
        K = K_rows[mask]
        F = F_rows[mask]
        otm_call = K >= F
        if method == "plain":
            logM = (eps * sims["A"][j] + math.sqrt(max(1.0 - eps**2, 0.0)) * sims["P"][j]
                    - 0.5 * sims["V"][j])
            ST = np.exp(logM)[:, None] * F[None, :]
            call, call_se = _pair_mean_se(np.maximum(ST - K[None, :], 0.0))
            put, put_se = _pair_mean_se(np.maximum(K[None, :] - ST, 0.0))
            otm = np.where(otm_call, call, put)
            iv = black76_implied_vol(otm, F, K, T, otm_call, 1.0)
        else:
            prem = _mixing_otm_premia(sims["A"][j], sims["V"][j], eps, F, K, otm_call)
            otm, otm_se = _pair_mean_se(prem)
            iv = black76_implied_vol(otm, F, K, T, otm_call, 1.0)
            if extrapolate:
                prem_c = _mixing_otm_premia(sims["Ac"][j], sims["Vc"][j], eps, F, K, otm_call)
                otm_c, _ = _pair_mean_se(prem_c)
                iv_c = black76_implied_vol(otm_c, F, K, T, otm_call, 1.0)
                iv = 2.0 * iv - iv_c
                _, otm_se = _pair_mean_se(2.0 * prem - prem_c)
                otm = black76_price(F, K, T, iv, otm_call)
            call = np.where(otm_call, otm, otm + (F - K))
            put = np.where(otm_call, otm - (F - K), otm)
            call_se = put_se = otm_se
        out["call"][mask], out["put"][mask] = call, put
        out["call_se"][mask], out["put_se"][mask] = call_se, put_se
        out["iv"][mask] = iv
    return out


def simulate_terminal_log_spot(
    T:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 20_000,
    seed:         int = 42,
    steps_per_year: int = MC_STEPS_PER_YEAR,
    min_steps:    int = MC_MIN_STEPS,
) -> np.ndarray:
    """
    Samples of log(S_T / F_T) from the full (plain) scheme, antithetic layout
    (rows i and i + n_paths/2 are mirror paths). E[exp(·)] = 1 exactly in the
    discrete scheme, so the sample mean of exp(·) is a martingale check.
    """
    t_grid, idx = mc_time_grid([T], steps_per_year, min_steps)
    sims = _simulate_integrals(params, fwd_var_func, t_grid, idx,
                               max(2, n_paths // 2), seed, with_perp=True)
    eps = float(params.epsilon)
    return (eps * sims["A"][0] + math.sqrt(max(1.0 - eps**2, 0.0)) * sims["P"][0]
            - 0.5 * sims["V"][0])


def price_spx_options_mc(
    S0:           float,
    T:            float,
    strikes:      np.ndarray,
    r:            float,
    q:            float,
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 20_000,
    n_steps:      int = 0,        # 0 → auto: max(MC_MIN_STEPS, ceil(252·T))
    seed:         int = 42,
    method:       str = "conditional",
    richardson:   bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Price a batch of SPX calls at the same expiry T by Monte Carlo
    (left-point Itô scheme, exact joint OU/Brownian steps, antithetic pairs,
    conditional estimator with an empirical martingale correction and
    Richardson extrapolation; see spx_forward_prices_mc). n_steps fixes the
    base grid.

    Returns
    -------
    prices   : discounted call prices e^{-rT}·E[(S_T − K)+], F = S0·e^{(r−q)T}
    std_errs : 1-sigma MC errors (from the antithetic pair averages)
    """
    strikes = np.asarray(strikes, dtype=float)
    if n_steps and n_steps > 0:
        spy, mns = max(1, int(math.ceil(n_steps / T))), int(n_steps)
    else:
        spy, mns = MC_STEPS_PER_YEAR, MC_MIN_STEPS
    F = S0 * math.exp((r - q) * T)
    res = spx_forward_prices_mc(np.full(len(strikes), T), strikes, np.full(len(strikes), F),
                                params, fwd_var_func, n_paths=n_paths, seed=seed,
                                steps_per_year=spy, min_steps=mns, method=method,
                                richardson=richardson)
    disc = math.exp(-r * T)
    return disc * res["call"], disc * res["call_se"]


def implied_vol_surface_forward_mc(
    expiries:     np.ndarray,   # per-row expiry
    strikes:      np.ndarray,   # per-row strike
    forwards:     np.ndarray,   # per-row forward F(T)
    params:       QuinticOUParams,
    fwd_var_func: Callable[[float], float],
    n_paths:      int = 10_000,
    seed:         int = 42,
    steps_per_year: int = MC_STEPS_PER_YEAR,
    min_steps:    int = MC_MIN_STEPS,
    return_stderr: bool = False,
    richardson:   bool = True,
):
    """
    Black-76 implied vols of the model smile, per row, on the given forwards.

    Each row's vol comes from its out-of-the-money premium (put for K < F,
    call for K ≥ F), Richardson-extrapolated by default. NaN only if a
    premium underflows to zero.
    With return_stderr=True also returns the MC standard error in vol units
    (price s.e. / Black-76 vega).
    """
    T = np.asarray(expiries, dtype=float)
    K = np.asarray(strikes, dtype=float)
    F = np.asarray(forwards, dtype=float)
    res = spx_forward_prices_mc(T, K, F, params, fwd_var_func, n_paths=n_paths, seed=seed,
                                steps_per_year=steps_per_year, min_steps=min_steps,
                                richardson=richardson)
    otm_call = K >= F
    se = np.where(otm_call, res["call_se"], res["put_se"])
    iv = res["iv"]
    if not return_stderr:
        return iv
    sd = np.where(np.isfinite(iv), iv, 0.2) * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    vega = F * np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * np.sqrt(T)
    return iv, np.where(np.isfinite(iv), se / vega, np.nan)


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
    SPX Black-Scholes implied vols (spot S0, rate r, yield q) for many (K, T)
    pairs, sharing one set of MC paths across expiries.

    Identical to Black-76 on F = S0·e^{(r−q)T}; see implied_vol_surface_forward_mc.
    Returns an array of implied vols (NaN where pricing fails).
    """
    T = np.asarray(expiries, dtype=float)
    F = S0 * np.exp((r - q) * T)
    return implied_vol_surface_forward_mc(T, np.asarray(strikes, dtype=float), F,
                                          params, fwd_var_func, n_paths=n_paths, seed=seed)


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
        """Undiscounted VIX call premium (see price_vix_option)."""
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
