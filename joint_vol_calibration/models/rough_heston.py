"""
rough_heston.py — rough Heston priced through its lifted multi-factor
approximation, on the repository's Lewis engine (C23 research model).

Model (El Euch & Rosenbaum 2019, arXiv:1609.02108), alpha = H + 1/2:
    dS_t / S_t = sqrt(V_t) dB_t
    V_t = V0 + int_0^t K(t-s) kappa (theta - V_s) ds
             + int_0^t K(t-s) sigma sqrt(V_s) dW_s,      d<B, W> = rho dt
    K(t) = t^(alpha-1) / Gamma(alpha)
H = 1/2 makes K = 1: classical Heston with the same (kappa, theta, sigma,
rho, v0). In log-forward coordinates X_T = ln(S_T / F_T),
    ln E[e^{i z X_T}] = kappa theta (I^1 h)(T) + V0 (I^{1-alpha} h)(T),
where h solves the fractional Riccati equation D^alpha h = F(z, h),
    F(z, h) = -z(z + i)/2 + (i rho sigma z - kappa) h + sigma^2 h^2 / 2.

Lifted approximation (Abi Jaber 2019, "Lifting the Heston model",
arXiv:1810.04868; Abi Jaber & El Euch 2019, SIAM J. Financial Math. 10):
K is completely monotone, K(t) = int_0^inf e^{-xt} mu(dx) with
mu(dx) = x^{-alpha} / (Gamma(alpha) Gamma(1-alpha)) dx. Collapsing mu on a
partition 0 = eta_0 < eta_1 < ... < eta_n onto each interval's barycentre gives
K_n(t) = sum_i c_i e^{-x_i t} with c_i, x_i >= 0, and h is replaced by
psi^n = sum_i c_i psi_i, where the psi_i solve the ODE system
    psi_i' = -x_i psi_i + F(z, psi^n),      psi_i(0) = 0,
and the log-CF becomes V0 int_0^T F(z, psi^n) + kappa theta int_0^T psi^n
(Abi Jaber 2019, Thm. 3.3: I^{1-alpha} h = I^1 F(z, h)).

Choices made here, and why:
  * The first interval is [0, eta_1], not (eta_0 > 0, ...) as in Abi Jaber &
    El Euch: then c_1 -> 1, x_1 -> 0 and every other c_i -> 0 as H -> 1/2,
    so the scheme is exactly Heston at H = 1/2 with no special case (tested).
    The endpoints eta_1 = 1/(sqrt(n) T), eta_n = 4 n^2.5 / T scale with the
    maturity T of the slice being priced; this is the partition QuantLib's
    FractionalKernelApproximation (BSD-3) documents from its own kernel-error
    study, adopted so the two implementations can be compared node for node.
  * Time stepping: each step integrates the linear part -x_i psi_i exactly
    and F as its linear interpolant over the step (an exponential
    integrator; weights Delta(phi1 - phi2) and Delta phi2 below). The F at
    the END of the step is treated implicitly: summing the factor updates
    gives psi^n_{k+1} = A + B F(psi^n_{k+1}), a scalar quadratic solved in
    closed form (stable root). The implicit step is what keeps very large
    |z| stable (an explicit step needs Delta * sigma |z| < O(1)). The grid
    t_k = T (k/N)^2 is refined near 0, where psi^n ~ t^alpha.
  * Pricing: the CF goes through heston._lewis_call_prices (Lewis contour,
    Black-Scholes control variate, adaptive trapezoid) exactly like Heston
    and Bates, so the three models share one quadrature.

Accuracy (measured, tests/test_rough_heston.py):
  * H = 1/2 reproduces heston_call_batch to ~5e-6 * F at the default 200
    steps and < 1e-6 * F at 800; the CF error falls 4x per doubling of the
    steps (second order);
  * against QuantLib 1.44's AnalyticRoughHestonEngine (dev-only oracle),
    Lifted route, same 20-factor partition: <= 1.1e-6 * F (800 steps here,
    2000 there) for H in {0.1, 0.3, 0.49}, 30 days and 1 year;
  * the lifting error itself, against QuantLib's Adams scheme for the exact
    fractional Riccati equation (2048 steps): 1.5e-7 to 8.5e-5 * F on the
    same sets, largest at H = 0.1; the mean variance converges to the
    Mittag-Leffler closed form like n^-1.5 (0.07%-0.26% at 20 factors).
  * at the H = 0.02 box floor, where the joint SPX + VIX-index fits land:
    mean-variance error 0.34%-0.44% at 20 factors (still n^-1.5); prices
    within 1.3e-4 * F of Adams at 8192 steps on the test sets (Adams at
    2048 steps is itself off by 6e-5 * F at sigma = 2 there) and within
    7.7e-5 * F / 0.085 vp of it at the 2026-10-16 joint fit, 14 days-1 year.
At a calibrated rough fit the anchor IVs move by <= 0.04 vp when factors
are doubled and steps quadrupled (JointCalibrator.calibrate_comparison,
numerics_check); with the n^-1.5 rate that self-convergence gap understates
the 20-factor error by a factor 1/(1 - 2^-1.5) = 1.55, which the check
reports as its estimate.

Counter-evidence to keep in view when reading any fit: Abi Jaber & Li (2024),
"Volatility models in practice: Rough, Path-dependent or Markovian?",
arXiv:2401.03345, find on SPX that rough models misfit the global smile
shape and do not beat one-factor Markovian models with the same parameter
count from one week to three months.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import cast

import numpy as np
from scipy.linalg import expm

from joint_vol_calibration.config import (
    ROUGH_HESTON_N_FACTORS,
    ROUGH_HESTON_TIME_STEPS,
)
from joint_vol_calibration.models.heston import (
    _clip_calls,
    _expected_integrated_variance,
    _lewis_call_derivs,
    _lewis_call_prices,
    _LewisGrid,
)

ROUGH_HESTON_PARAM_NAMES = ("kappa", "theta", "sigma", "rho", "v0", "H")
_GRADING = 2.0                 # t_k = T (k/N)^_GRADING


# ── Kernel approximation ─────────────────────────────────────────────────────

def _sinc_pi(b: float) -> float:
    """sin(pi b)/(pi b) = 1/(Gamma(1+b) Gamma(1-b)) without the Gamma
    overflow as b -> 0."""
    y = math.pi * b
    return 1.0 - y * y / 6.0 if abs(y) < 1e-8 else math.sin(y) / y


@lru_cache(maxsize=4096)
def _kernel_cached(H: float, n: int, T: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    beta = 0.5 - H                         # 1 - alpha
    norm = _sinc_pi(beta)
    eta_lo = 1.0 / (math.sqrt(n) * T)
    eta_hi = 4.0 * n ** 2.5 / T
    if n == 1:
        eta_lo = eta_hi
    c = np.empty(n)
    x = np.empty(n)
    # [0, eta_1]: mass eta_1^beta * norm, barycentre beta/(1+beta) eta_1
    c[0] = eta_lo ** beta * norm
    x[0] = beta / (1.0 + beta) * eta_lo
    if n > 1:
        dl = math.log(eta_hi / eta_lo) / (n - 1)
        # (b^beta - a^beta) and the barycentre through expm1: no cancellation
        # as beta -> 0 (H -> 1/2)
        em_b = math.expm1(beta * dl)
        em_1b = math.expm1((1.0 + beta) * dl)
        rate = (beta / (1.0 + beta)) * (em_1b / em_b) if beta > 0 else math.expm1(dl) / dl
        a = eta_lo
        for i in range(1, n):
            c[i] = a ** beta * em_b * norm
            x[i] = a * rate
            a *= math.exp(dl)
    return tuple(c), tuple(x)


def lifted_kernel(H: float, n_factors: int = ROUGH_HESTON_N_FACTORS,
                  T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(c, x): K_n(t) = sum_i c_i exp(-x_i t) approximating
    t^(H-1/2) / Gamma(H+1/2) for t in (0, T] (see the module docstring)."""
    if not 0.0 < H <= 0.5:
        raise ValueError(f"H must be in (0, 1/2], got {H}")
    if n_factors < 1:
        raise ValueError("at least one factor required")
    if T <= 0:
        raise ValueError("T must be positive")
    c, x = _kernel_cached(float(H), int(n_factors), float(T))
    return np.array(c), np.array(x)


def _phi12(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """phi1 = (1 - e^-t)/t, phi2 = (t - 1 + e^-t)/t^2 (series for small t)."""
    t = np.asarray(theta, dtype=float)
    small = t < 1e-4
    ts = np.where(small, 1.0, t)
    em = np.expm1(-ts)
    p1 = np.where(small, 1.0 - t / 2.0 + t * t / 6.0, -em / ts)
    p2 = np.where(small, 0.5 - t / 6.0 + t * t / 24.0, (ts + em) / (ts * ts))
    return p1, p2


# ── Riccati solver and characteristic function ──────────────────────────────

@lru_cache(maxsize=256)
def _unit_tables(H: float, n_factors: int, n_steps: int):
    """Step tables for maturity T = 1, shared by EVERY maturity.

    The partition endpoints scale as 1/T, so x_i(T) = x_i(1)/T and
    c_i(T) = c_i(1) T^(H-1/2); on t_k = T (k/N)^2 the step is dt = T d_k, so
    x_i dt = x_i(1) d_k does not depend on T. With psi_i = T phi_i the factor
    update phi_{k+1} = E phi_k + W0 F_k + W1 F_{k+1} uses the T = 1 weights
    for every slice, and psi^n = sum_i c_i(T) psi_i = T^(1/2+H) sum_i c_i(1) phi_i.
    Returns (d (N,), E (N, n), W0 (N, n), W1 (N, n), cE (N, n) = c E,
    cW0 (N,) = c.W0, B1 (N,) = c.W1), with c = c(1).
    """
    c, x = (np.array(v) for v in _kernel_cached(H, n_factors, 1.0))
    tk = (np.arange(n_steps + 1) / n_steps) ** _GRADING
    d = np.diff(tk)
    th = d[:, None] * x[None, :]
    p1, p2 = _phi12(th)
    W0 = d[:, None] * (p1 - p2)
    W1 = d[:, None] * p2
    E = np.exp(-th)
    return d, E, W0, W1, E * c[None, :], W0 @ c, W1 @ c


def _riccati_sweep(z: np.ndarray, T_col, kappa: float, sigma: float,
                   rho: float, H: float, n_factors: int, n_steps: int):
    """One time sweep for frequencies z[j] at maturities T_col[j] (columns
    may belong to different expiries; see _unit_tables for why they can share
    one sweep). Returns (int_0^T F, int_0^T psi^n, psi^n(T)) per column."""
    z = np.asarray(z, dtype=complex)
    T_col = np.broadcast_to(np.asarray(T_col, dtype=float), z.shape)
    d, E, W0, W1, cE, cW0, B1 = _unit_tables(float(H), int(n_factors), int(n_steps))
    s_col = T_col ** (0.5 + H)                 # psi^n = s_col * sum_i c_i(1) phi_i
    a0 = -0.5 * z * (z + 1j)
    a1 = 1j * rho * sigma * z - kappa
    a2 = 0.5 * sigma * sigma
    phi = np.zeros((E.shape[1], *z.shape), dtype=complex)
    psin = np.zeros(z.shape, dtype=complex)
    f = a0.copy()                              # F(z, 0)
    int_f = np.zeros(z.shape, dtype=complex)
    int_psi = np.zeros(z.shape, dtype=complex)
    for k in range(n_steps):
        # A = sum_i c_i (E_i phi_i + W0_i F_k), scaled to psi^n units
        A = s_col * (np.einsum("n,nm->m", cE[k], phi) + cW0[k] * f)
        Bk = s_col * B1[k]
        # psi^n = A + B F(psi^n)  <=>  B a2 psi^2 + (B a1 - 1) psi + (A + B a0) = 0;
        # the root continuous with psi^n = (A + B a0)/(1 - B a1) as B -> 0
        P = 1.0 - Bk * a1
        Qc = A + Bk * a0
        disc = np.sqrt(P * P - 4.0 * a2 * Bk * Qc)
        disc = np.where((P.conj() * disc).real >= 0, disc, -disc)
        psin_next = 2.0 * Qc / (P + disc)
        f_next = a0 + (a1 + a2 * psin_next) * psin_next
        # phi <- E phi + W0 F_k + W1 F_{k+1}, in place
        phi *= E[k][:, None]
        phi += np.multiply.outer(W0[k], f)
        phi += np.multiply.outer(W1[k], f_next)
        h = 0.5 * d[k] * T_col
        int_f += h * (f + f_next)
        int_psi += h * (psin + psin_next)
        psin, f = psin_next, f_next
    return int_f, int_psi, psin


def rough_heston_log_cf(z, T: float, kappa: float, theta: float, sigma: float, rho: float,
                        v0: float, H: float, n_factors: int = ROUGH_HESTON_N_FACTORS,
                        n_steps: int = ROUGH_HESTON_TIME_STEPS) -> np.ndarray:
    """ln E[exp(i z ln(S_T/F_T))] under the lifted rough Heston model,
    vectorised over complex z. The kernel is built for horizon T."""
    z = np.asarray(z, dtype=complex)
    if T <= 0:
        return np.zeros(z.shape, dtype=complex)
    if not 0.0 < H <= 0.5:
        raise ValueError(f"H must be in (0, 1/2], got {H}")
    int_f, int_psi, _ = _riccati_sweep(z.ravel(), T, kappa, sigma, rho, H, n_factors, n_steps)
    return (v0 * int_f + kappa * theta * int_psi).reshape(z.shape)


# ── Expected variance (VIX index leg) ────────────────────────────────────────

def _lifted_mean_integral(T: float, kappa: float, theta: float, v0: float,
                          c: np.ndarray, x: np.ndarray) -> float:
    """int_0^T E[V_t] dt under the lifted model with kernel (c, x).

    E[V_t] = V0 + sum_i c_i m_i(t) with m' = -diag(x) m + kappa (theta - E[V]) 1,
    m(0) = 0 (Abi Jaber 2019, the mean of the lifted factors). Integrated
    exactly with one matrix exponential of the augmented linear system
    (m, int m, 1).
    """
    n = len(c)
    A = -np.diag(x) - kappa * np.outer(np.ones(n), c)
    b = kappa * (theta - v0) * np.ones(n)
    M = np.zeros((2 * n + 1, 2 * n + 1))
    M[:n, :n] = A
    M[:n, 2 * n] = b
    M[n:2 * n, :n] = np.eye(n)
    y = expm(M * T)[:, 2 * n]
    return float(v0 * T + c @ y[n:2 * n])


def rough_heston_expected_variance_integral(T, kappa: float, theta: float, v0: float,
                                            H: float, n_factors: int = ROUGH_HESTON_N_FACTORS):
    """E[int_0^T V_t dt] (vectorised over T), kernel built per horizon.
    H = 1/2 is Heston's theta T + (v0 - theta)(1 - e^{-kappa T})/kappa."""
    Ts = np.atleast_1d(np.asarray(T, dtype=float))
    out = np.empty(len(Ts))
    for i, t in enumerate(Ts):
        if t <= 0:
            out[i] = 0.0
            continue
        c, x = lifted_kernel(H, n_factors, t)
        out[i] = _lifted_mean_integral(t, kappa, theta, v0, c, x)
    return float(out[0]) if np.ndim(T) == 0 else out


def rough_heston_vix_term_index(kappa: float, theta: float, sigma: float, v0: float,
                                H: float, T, n_factors: int = ROUGH_HESTON_N_FACTORS):
    """Spot-starting VIX-style index 100 sqrt(E[int_0^T V] / T) (the Cboe
    ^VIX9D/^VIX/^VIX3M/^VIX6M definition; paths are continuous, so the
    log-contract variance is the expected integrated variance). sigma is
    accepted for signature symmetry; the mean does not depend on it."""
    Ts = np.atleast_1d(np.asarray(T, dtype=float))
    integ = np.atleast_1d(rough_heston_expected_variance_integral(Ts, kappa, theta, v0, H,
                                                                  n_factors))
    out = 100.0 * np.sqrt(np.maximum(integ / Ts, 0.0))
    return float(out[0]) if np.ndim(T) == 0 else out


# ── Pricing ──────────────────────────────────────────────────────────────────

# Quadrature target for the Lewis trapezoid, as a fraction of DF*F. The
# Heston engine's 1e-10 would be wasted here: the Riccati time stepping is
# accurate to ~1e-6 of the CF at 200 steps (tests), so 1e-7 keeps the
# quadrature an order below it at a fraction of the nodes.
ROUGH_LEWIS_RTOL = 1e-7


# ln|phi| below which the CF carries no price weight at any quadrature target
# (e^-50 ~ 2e-22 of DF*F, against ROUGH_LEWIS_RTOL = 1e-7); see _mask_unresolved.
_LOG_CF_FLOOR = -50.0


def _mask_unresolved(log_cf: np.ndarray, u=None, slice_of=None) -> np.ndarray:
    """Set phi = 0 at far-tail nodes the time grid cannot resolve.

    |phi(u - i/2)| <= E[(S_T/F)^(1/2)] <= 1 on the Lewis contour, so a node
    reporting |phi| > 1 is unresolved. Past the frequency where the Riccati
    steps stop resolving psi^n (u ~ 2.5e5 at 200 steps, T = 0.47 y, where
    the true ln|phi| is ~ -1.6e4), the sweep returns noise whose real part
    lands anywhere, not only above 0: on 2026-10-27 noise at |phi| ~ 1e-12
    near u = 9e5 read as an undecayed tail to the Lewis truncation scan, which
    sized one slice at 976,333 nodes (205 at the neighbouring optimum) and
    made each frozen-grid residual 26 s instead of 0.02 s.
    So, given the frequencies ``u`` (the real part of z; ``slice_of`` labels
    the slice of each node when several are stacked), every node of a slice
    at or above the lowest frequency where ln|phi| has fallen below
    _LOG_CF_FLOOR is set to phi = 0 too: the CF of the lifted model decays and
    does not come back, so nothing beyond that point carries price weight.
    """
    log_cf = np.asarray(log_cf)
    bad = ~np.isfinite(log_cf) | (log_cf.real > 1e-12)
    if u is not None:
        u = np.broadcast_to(np.asarray(u, dtype=float), log_cf.shape)
        key = (np.zeros(log_cf.shape) if slice_of is None
               else np.broadcast_to(np.asarray(slice_of), log_cf.shape))
        low = np.isfinite(log_cf) & (log_cf.real < _LOG_CF_FLOOR)
        for k in np.unique(key[low]):
            same = key == k
            bad |= same & (u >= u[same & low].min())
    return np.where(bad, -np.inf + 0j, log_cf) if bad.any() else log_cf

def rough_heston_call_batch_grid(
    S: float, strikes, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, v0: float, H: float,
    n_factors: int = ROUGH_HESTON_N_FACTORS, n_steps: int = ROUGH_HESTON_TIME_STEPS,
    grid: _LewisGrid | None = None, rtol: float = ROUGH_LEWIS_RTOL,
) -> tuple[np.ndarray, _LewisGrid | None]:
    """Calls and the Lewis grid used (grid given: priced on it, no adaptation)."""
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    if T <= 0:
        F = S * np.exp((r - q) * T)
        return np.maximum(F - strikes, 0.0) * np.exp(-r * T), None
    # Black-Scholes control-variate variance: Heston's closed-form E[int v]
    # (the engine's accuracy does not depend on it being the rough model's
    # own mean; it only has to be close, and it is cheap)
    total_var = float(_expected_integrated_variance(T, kappa, theta, v0))

    def log_cf(z):
        z = np.asarray(z, dtype=complex)
        int_f, int_psi, _ = _riccati_sweep(z.ravel(), T, kappa, sigma, rho, H, n_factors,
                                           n_steps)
        return _mask_unresolved((v0 * int_f + kappa * theta * int_psi).reshape(z.shape),
                                z.real)

    calls, grid = _lewis_call_prices(S, strikes, T, r, q, log_cf, max(total_var, 1e-12),
                                     grid=grid, rtol=rtol)
    return _clip_calls(calls, S, strikes, T, r, q), grid


def rough_heston_call_batch(
    S: float, strikes, T: float, r: float, q: float,
    kappa: float, theta: float, sigma: float, rho: float, v0: float, H: float,
    n_factors: int = ROUGH_HESTON_N_FACTORS, n_steps: int = ROUGH_HESTON_TIME_STEPS,
) -> np.ndarray:
    """European calls for one expiry under lifted rough Heston, vectorised
    over strikes; heston_call_batch's signature plus H (so
    JointCalibrator._model_prices / _surface_errors / _out_of_sample accept it
    as a pricer). Discounted, clipped to the no-arbitrage band."""
    return rough_heston_call_batch_grid(S, strikes, T, r, q, kappa, theta, sigma, rho, v0, H,
                                        n_factors, n_steps)[0]


class RoughHestonSurfacePricer:
    """Every expiry of a quote surface in ONE stacked Riccati sweep on frozen
    Lewis grids: the calibration inner loop.

    slices: [(T, F, D, strikes)] priced off forward F and discount D.
    adapt(p) prices each slice with the adaptive engine and stores the grids;
    prices(p) then reuses them (no scan, no step halving), so a residual
    evaluation is one time sweep over all frequencies of all slices. On a
    frozen grid the prices are smooth in p (finite-difference Jacobians see
    no grid switching); a fit refreshes the grids at its optimum and checks
    the frozen prices against the adaptive ones there
    (JointCalibrator.calibrate_spx_only).
    """

    def __init__(self, slices, n_factors: int = ROUGH_HESTON_N_FACTORS,
                 n_steps: int = ROUGH_HESTON_TIME_STEPS, rtol: float = ROUGH_LEWIS_RTOL):
        self.slices = [(float(T), float(F), float(D), np.atleast_1d(np.asarray(K, float)))
                       for T, F, D, K in slices]
        self.n_factors, self.n_steps, self.rtol = int(n_factors), int(n_steps), float(rtol)
        self.grids: list | None = None

    def _price_slice(self, j, p, grid=None):
        T, F, D, K = self.slices[j]
        r_T = -math.log(D) / T
        return rough_heston_call_batch_grid(F, K, T, r_T, r_T, *p, n_factors=self.n_factors,
                                            n_steps=self.n_steps, grid=grid, rtol=self.rtol)

    def adapt(self, p) -> list:
        """Adaptive prices at p; stores their grids for prices()."""
        out, grids = [], []
        for j in range(len(self.slices)):
            calls, grid = self._price_slice(j, p)
            out.append(calls)
            grids.append(grid)
        self.grids = grids
        return out

    @property
    def n_nodes(self) -> int:
        return int(sum(g.n for g in self.grids)) if self.grids else 0

    def prices(self, p) -> list:
        """Calls per slice on the frozen grids (adapt() first)."""
        if self.grids is None:
            raise RuntimeError("adapt() before prices()")
        kappa, theta, sigma, rho, v0, H = (float(v) for v in p)
        z = np.concatenate([g.h * np.arange(g.n) - 0.5j for g in self.grids])
        T_col = np.concatenate([np.full(g.n, s[0]) for s, g in zip(self.slices, self.grids)])
        int_f, int_psi, _ = _riccati_sweep(z, T_col, kappa, sigma, rho, H, self.n_factors,
                                           self.n_steps)
        log_cf = _mask_unresolved(v0 * int_f + kappa * theta * int_psi, z.real,
                                  np.repeat(np.arange(len(self.grids)), [g.n for g in self.grids]))
        self._last: tuple[bytes, np.ndarray, np.ndarray, np.ndarray] = (
            np.asarray(p, dtype=float).tobytes(), log_cf, int_f, int_psi)
        out, start = [], 0
        for (T, F, D, K), grid in zip(self.slices, self.grids):
            vals = log_cf[start:start + grid.n]
            start += grid.n
            w = float(_expected_integrated_variance(T, kappa, theta, v0))
            r_T = -math.log(D) / T
            def log_cf_slice(_z, v=vals):
                return v
            calls, _ = _lewis_call_prices(F, K, T, r_T, r_T, log_cf_slice, max(w, 1e-12), grid=grid)
            out.append(_clip_calls(calls, F, K, T, r_T, r_T))
        return out

    def theta_v0_derivatives(self, p) -> list:
        """d(call)/d(theta, v0) per slice on the frozen grids, exactly: the
        log-CF is v0 int F + kappa theta int psi^n and neither integral
        depends on theta or v0. Reuses the sweep of prices(p) when it was the
        last one evaluated."""
        key = np.asarray(p, dtype=float).tobytes()
        if getattr(self, "_last", None) is None or self._last[0] != key:
            self.prices(p)
        _, log_cf, int_f, int_psi = self._last
        kappa = float(p[0])
        out, start = [], 0
        for (T, F, D, K), grid in zip(self.slices, cast(list, self.grids)):   # set by prices()
            sl = slice(start, start + grid.n)
            start += grid.n
            grad = np.vstack([kappa * int_psi[sl], int_f[sl]])
            vals = log_cf[sl]
            r_T = -math.log(D) / T
            out.append(_lewis_call_derivs(F, K, T, r_T, r_T,
                                          lambda _z, v=vals, g=grad: (v, g), grid))
        return out


# ── SPX-only least squares (C23 model comparison) ────────────────────────────

class RoughHestonSPXResiduals:
    """Residuals of the SPX anchors, optionally with the VIX spot-index leg,
    under lifted rough Heston, priced on frozen grids
    (RoughHestonSurfacePricer):
      SPX  sqrt(w_spx/n) (model - market) / vega          per anchor
      VIX  sqrt(w_vix/n_vix) (index - market) / 100       per spot index
    (the joint loss of JointCalibrator with the rough model; w_vix = 0 is the
    SPX-only problem). groups: JointCalibrator._spx_groups() tuples
    (T, F, D, K, is_put, market, vega). adapt(p) refreshes the grids at p and
    returns the residuals there; __call__(p) prices on the current grids.
    """

    names = ROUGH_HESTON_PARAM_NAMES

    def __init__(self, groups, w_spx: float = 1.0, n_factors: int = ROUGH_HESTON_N_FACTORS,
                 n_steps: int = ROUGH_HESTON_TIME_STEPS, vix_tenors=None, vix_market=None,
                 w_vix: float = 0.0):
        self.groups = [tuple(g) for g in groups]
        self.pricer = RoughHestonSurfacePricer([(g[0], g[1], g[2], g[3]) for g in self.groups],
                                               n_factors=n_factors, n_steps=n_steps)
        n = sum(len(g[3]) for g in self.groups)
        self.c = math.sqrt(w_spx / n)
        self.n_factors = int(n_factors)
        self.vix_tenors = np.asarray([] if vix_tenors is None else vix_tenors, dtype=float)
        self.vix_market = np.asarray([] if vix_market is None else vix_market, dtype=float)
        self.c_vix = (math.sqrt(w_vix / len(self.vix_tenors))
                      if (w_vix > 0 and len(self.vix_tenors)) else 0.0)
        self.n_calls = 0
        self.n_adapt = 0

    def _vix(self, p) -> np.ndarray:
        kappa, theta, sigma, _, v0, H = p
        idx = rough_heston_vix_term_index(kappa, theta, sigma, v0, H, self.vix_tenors,
                                          self.n_factors)
        return self.c_vix * (np.atleast_1d(idx) - self.vix_market) / 100.0

    def _resid(self, calls_list, p) -> np.ndarray:
        parts = []
        for (_, F, D, K, is_put, mkt, vega), calls in zip(self.groups, calls_list):
            model = calls.copy()
            model[is_put] = calls[is_put] - D * (F - K[is_put])
            parts.append(self.c * (model - mkt) / vega)
        if self.c_vix:
            parts.append(self._vix(p))
        return np.concatenate(parts)

    def adapt(self, p) -> np.ndarray:
        self.n_adapt += 1
        p = np.asarray(p, dtype=float)
        return self._resid(self.pricer.adapt(p), p)

    def __call__(self, p) -> np.ndarray:
        self.n_calls += 1
        p = np.asarray(p, dtype=float)
        r = self._resid(self.pricer.prices(p), p)
        self._last_r = (p.tobytes(), id(self.pricer.grids), r)
        return r

    def jac(self, p, lo=None, hi=None) -> np.ndarray:
        """d r / d p on the frozen grids: the SPX theta and v0 columns exact
        (theta_v0_derivatives), the VIX-index part of those two columns by
        central differences of the closed-form index (cheap), kappa / sigma /
        rho / H by one-sided differences of the whole vector that stay inside
        [lo, hi] (the price is smooth in p on a frozen grid, so a 1e-7
        relative step is safe)."""
        p = np.asarray(p, dtype=float)
        last = getattr(self, "_last_r", None)
        # least_squares evaluates r(p) right before jac(p): reuse it
        base = (last[2] if last is not None and last[0] == p.tobytes()
                and last[1] == id(self.pricer.grids) else self(p))
        J = np.empty((len(base), 6))
        n_spx = sum(len(g[3]) for g in self.groups)
        dtv = self.pricer.theta_v0_derivatives(p)
        J[:n_spx, 1] = np.concatenate([self.c * d[:, 0] / g[6] for d, g in zip(dtv, self.groups)])
        J[:n_spx, 4] = np.concatenate([self.c * d[:, 1] / g[6] for d, g in zip(dtv, self.groups)])
        if self.c_vix:
            for j in (1, 4):
                h = 1e-6 * abs(p[j])
                up, dn = p.copy(), p.copy()
                up[j] += h
                dn[j] -= h
                J[n_spx:, j] = (self._vix(up) - self._vix(dn)) / (2 * h)
        for j in (0, 2, 3, 5):
            h = 1e-7 * max(abs(p[j]), 0.1)
            if hi is not None and p[j] + h > hi[j]:
                h = -h
            pp = p.copy()
            pp[j] += h
            J[:, j] = (self(pp) - base) / h
        return J


def fit_trf_with_grid_refresh(resid: RoughHestonSPXResiduals, starts, labels, lo, hi, *,
                              max_nfev: int = 120, tol: float = 1e-10,
                              refresh_rtol: float = 1e-6, max_refresh: int = 2) -> tuple:
    """Multi-start TRF (RoughHestonSPXResiduals.jac on frozen grids) where each
    start adapts the grids at its x0, and after convergence re-adapts them at
    the optimum; if the adaptive loss there differs from the frozen one by
    more than refresh_rtol (relative), TRF restarts from that point on the new
    grids (at most max_refresh times). The loss reported per start is the
    ADAPTIVE one at its final point. Returns (best_x, best_loss, report)."""
    from scipy import optimize

    def jac(q):
        return resid.jac(q, lo, hi)

    runs = []
    best_x, best_loss = None, np.inf
    for x0, label in zip(starts, labels):
        x = np.clip(np.asarray(x0, dtype=float), lo, hi)
        t0 = time.perf_counter()
        calls0 = resid.n_calls
        resid.adapt(x)
        nfev, refreshes, status, frozen_loss = 0, 0, 0, np.nan
        for attempt in range(max_refresh + 1):
            refreshes = attempt
            sol = optimize.least_squares(resid, x, jac=jac,
                                         bounds=(lo, hi), method="trf", x_scale="jac",
                                         ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
            nfev += int(sol.nfev)
            status = int(sol.status)
            x = sol.x
            frozen_loss = float(2.0 * sol.cost)
            r = resid.adapt(x)
            adapt_loss = float(r @ r)
            if abs(adapt_loss - frozen_loss) <= refresh_rtol * max(frozen_loss, 1e-300):
                break
        runs.append({"label": label, "x0": [float(v) for v in np.clip(x0, lo, hi)],
                     "x": [float(v) for v in x], "loss": adapt_loss,
                     "frozen_loss": frozen_loss, "grid_refreshes": int(refreshes),
                     "nfev": nfev, "n_residual_evals": resid.n_calls - calls0,
                     "status": status,
                     "seconds": time.perf_counter() - t0})
        if adapt_loss < best_loss:
            best_x, best_loss = x.copy(), adapt_loss
    return best_x, best_loss, runs
