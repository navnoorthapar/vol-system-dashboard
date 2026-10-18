"""
test_quintic_ou.py — unit tests for the two-factor Quintic OU model (C13).

Coverage:
  • Gaussian moment recurrence, ep_squared, var_Z (incl. the law of total
    variance for the conditional OU step) and the exact (ΔW, I_X, I_Y) step law
  • eval_poly; g0 normalisation (E[σ²] = ξ_0)
  • VIX²: polynomial coefficients vs a direct moment-recursion integral and vs a
    brute-force conditional Monte Carlo; monotonicity in X_T
  • VIX futures/options: Jensen bound, put-call parity, cubature convergence,
    undiscounted premia + consistent Black-76 inversion
  • SPX Monte Carlo (C22 rewrite): martingale property, true antithetic pairs,
    E[σ²] = ξ_0 under the simulation (no floor), conditional vs plain estimator,
    no-arbitrage, step convergence of the Richardson-extrapolated scheme
  • Parameter dataclass, scale normalisation, admissibility checks
"""

import dataclasses
import math

import numpy as np
import pytest

from joint_vol_calibration.models import quintic_ou as q
from joint_vol_calibration.models.quintic_ou import (
    QuinticOUParams,
    eval_poly,
    gaussian_moments,
    ep_squared,
    var_Z,
    _cond_var_Z,
    _cond_var_components,
    g0_at,
    compute_vix2,
    price_vix_futures,
    price_vix_option,
    vix_option_prices,
    price_spx_options_mc,
    QUINTIC_DEFAULTS,
    QUINTIC_BOUNDS,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    # Large α0: p(z) > 0 over ±3 std of Z and a strong even part, so the
    # antithetic mirror path is genuinely different from the original.
    return QuinticOUParams(
        lam_x=35.0, lam_y=0.6, theta=0.94,
        alpha0=0.30, alpha1=0.10, alpha3=0.02, alpha5=0.01,
        epsilon=-0.75,
    )


@pytest.fixture
def flat_fwd_var():
    """Flat ξ_0 = 0.04  (20% vol)."""
    return lambda t: 0.04


class _StepCurve:
    """Piecewise-constant ξ_0 with exact integral and breakpoints (like the calibrator's)."""

    def __init__(self):
        self.t_nodes = np.array([0.05, 0.25, 0.5])
        self.xi_nodes = np.array([0.09, 0.05, 0.07])

    def __call__(self, t):
        i = min(int(np.searchsorted(self.t_nodes, t, side="left")), len(self.xi_nodes) - 1)
        return float(self.xi_nodes[i])

    @property
    def breakpoints(self):
        return self.t_nodes[:-1]

    def integrated(self, t):
        edges = np.concatenate([[0.0], self.t_nodes])
        tot = 0.0
        for i, xi in enumerate(self.xi_nodes):
            lo, hi = edges[i], edges[i + 1]
            if t <= lo:
                return tot
            tot += xi * (min(t, hi) - lo)
        return tot + (self.xi_nodes[-1] * (t - edges[-1]) if t > edges[-1] else 0.0)


# ── Gaussian moments ───────────────────────────────────────────────────────────

class TestGaussianMoments:

    def test_zeroth_moment_is_one(self):
        m = gaussian_moments(0, mu=1.5, sigma2=2.0)
        assert m[0] == pytest.approx(1.0)

    def test_first_moment_equals_mean(self):
        m = gaussian_moments(2, mu=3.7, sigma2=1.0)
        assert m[1] == pytest.approx(3.7)

    def test_second_moment_variance_plus_mean_squared(self):
        mu, s2 = 1.0, 4.0
        m = gaussian_moments(2, mu, s2)
        assert m[2] == pytest.approx(mu**2 + s2, rel=1e-10)

    def test_odd_moments_zero_for_zero_mean(self):
        m = gaussian_moments(9, mu=0.0, sigma2=3.0)
        for k in [1, 3, 5, 7, 9]:
            assert m[k] == pytest.approx(0.0, abs=1e-12)

    def test_fourth_moment_N01(self):
        # Z ~ N(0,1): E[Z^4] = 3
        m = gaussian_moments(4, mu=0.0, sigma2=1.0)
        assert m[4] == pytest.approx(3.0, rel=1e-10)

    def test_array_inputs_match_scalar(self):
        mus = np.array([0.0, 0.3, -1.2])
        m = gaussian_moments(10, mus, 0.5)
        for j, mu in enumerate(mus):
            np.testing.assert_allclose(m[:, j], gaussian_moments(10, mu, 0.5), rtol=1e-14)


# ── ep_squared ─────────────────────────────────────────────────────────────────

class TestEpSquared:

    def test_constant_poly(self):
        alpha = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert ep_squared(mu=0.0, sigma2=1.0, alpha=alpha) == pytest.approx(4.0, rel=1e-10)

    def test_linear_poly_zero_mean(self):
        alpha = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        assert ep_squared(mu=0.0, sigma2=2.0, alpha=alpha) == pytest.approx(18.0, rel=1e-10)

    def test_agrees_with_gauss_hermite_integral(self):
        # Independent check: 120-point Gauss-Hermite integral of p(z)² under N(mu, s2)
        alpha = np.array([0.05, 0.20, 0.0, 0.05, 0.0, 0.02])
        mu, s2 = 0.5, 1.5
        x, w = np.polynomial.hermite.hermgauss(120)
        z = mu + math.sqrt(2.0 * s2) * x
        expected = float(np.sum(w * eval_poly(z, alpha)**2) / math.sqrt(math.pi))
        assert ep_squared(mu, s2, alpha) == pytest.approx(expected, rel=1e-10)


# ── var_Z / conditional step law ──────────────────────────────────────────────

class TestVarZ:

    def test_zero_at_t0(self):
        assert var_Z(0.0, lam_x=10.0, lam_y=1.0, theta=0.9) == pytest.approx(0.0)

    def test_increases_with_t(self):
        v1 = var_Z(0.1, 10.0, 1.0, 0.9)
        v2 = var_Z(1.0, 10.0, 1.0, 0.9)
        assert v2 > v1 > 0.0

    def test_stationary_variance_limit(self):
        lam_x, lam_y, theta = 10.0, 1.0, 0.9
        assert var_Z(1e6, lam_x, lam_y, theta) == pytest.approx(
            q.stationary_var_Z(lam_x, lam_y, theta), rel=1e-10)

    def test_cond_var_Z_zero_dt(self):
        assert _cond_var_Z(0.0, 10.0, 1.0, 0.9) == pytest.approx(0.0)

    def test_law_of_total_variance_over_one_step(self):
        # Var Z_{t+h} = Var(θe^{-λx h}X_t + (1-θ)e^{-λy h}Y_t) + Var[Z_{t+h} | F_t],
        # with Var/Cov of (X_t, Y_t) from the one-step components at horizon t.
        lam_x, lam_y, theta, t, h = 12.0, 0.8, 0.7, 0.3, 0.07
        vx, vy, cxy = _cond_var_components(t, lam_x, lam_y)
        a, b = theta * math.exp(-lam_x * h), (1 - theta) * math.exp(-lam_y * h)
        carried = a * a * vx + b * b * vy + 2 * a * b * cxy
        assert var_Z(t + h, lam_x, lam_y, theta) == pytest.approx(
            carried + _cond_var_Z(h, lam_x, lam_y, theta), rel=1e-12)

    def test_step_cholesky_reproduces_exact_covariance(self):
        # (ΔW, I_X, I_Y) covariance from the closed forms, incl. a nearly
        # singular case (λy h ≪ 1, I_Y ≈ ΔW).
        for lam_x, lam_y, h in ((35.0, 0.6, 1 / 252), (150.0, 0.01, 1 / 2520)):
            L = q._step_cholesky(h, lam_x, lam_y)
            vx, vy, cxy = _cond_var_components(h, lam_x, lam_y)
            cwx = (1 - math.exp(-lam_x * h)) / lam_x
            cwy = (1 - math.exp(-lam_y * h)) / lam_y
            expected = np.array([[h, cwx, cwy], [cwx, vx, cxy], [cwy, cxy, vy]])
            np.testing.assert_allclose(L @ L.T, expected, rtol=1e-9, atol=1e-15)


# ── eval_poly ─────────────────────────────────────────────────────────────────

class TestEvalPoly:

    def test_constant_term_at_zero(self):
        alpha = np.array([3.0, 1.0, 0.0, 0.5, 0.0, 0.1])
        assert eval_poly(np.array([0.0]), alpha)[0] == pytest.approx(3.0)

    def test_odd_polynomial_antisymmetric_around_zero(self):
        alpha = np.array([0.0, 0.5, 0.0, 0.3, 0.0, 0.1])
        z = np.array([1.5])
        assert eval_poly(z, alpha)[0] == pytest.approx(-eval_poly(-z, alpha)[0])

    def test_matches_power_sum(self):
        alpha = np.array([0.05, 0.20, 0.0, 0.05, 0.0, 0.02])
        z = np.linspace(-2, 2, 50)
        expected = sum(alpha[k] * z**k for k in range(6))
        np.testing.assert_allclose(eval_poly(z, alpha), expected, rtol=1e-13, atol=1e-15)


# ── g0 normalisation ───────────────────────────────────────────────────────────

class TestG0Normalisation:

    def test_g0_positive(self, default_params, flat_fwd_var):
        assert g0_at(0.5, flat_fwd_var(0.5), default_params) > 0.0

    def test_g0_ensures_ep2_equals_xi0(self, default_params, flat_fwd_var):
        t = 0.25
        xi0 = flat_fwd_var(t)
        g = g0_at(t, xi0, default_params)
        ep2 = ep_squared(0.0, var_Z(t, default_params.lam_x, default_params.lam_y,
                                    default_params.theta), default_params.alpha_vec())
        assert g**2 * ep2 == pytest.approx(xi0, rel=1e-8)

    def test_g0_scales_with_xi0(self, default_params):
        g1 = g0_at(0.5, 0.04, default_params)
        g2 = g0_at(0.5, 0.16, default_params)
        assert g2 == pytest.approx(2.0 * g1, rel=1e-8)


# ── VIX² ──────────────────────────────────────────────────────────────────────

def _vix2_direct(T, xt, yt, p, fv, n=200):
    """Independent VIX²: GL time integral of ep_squared ratios (no polynomial expansion)."""
    x, w = np.polynomial.legendre.leggauss(n)
    s = T + 0.5 * (x + 1) * q._VIX_DELTA
    ws = 0.5 * w * q._VIX_DELTA
    a = p.alpha_vec()
    tot = 0.0
    for si, wi in zip(s, ws):
        tau = si - T
        mu = p.theta * xt * math.exp(-p.lam_x * tau) + (1 - p.theta) * yt * math.exp(-p.lam_y * tau)
        cond = ep_squared(mu, var_Z(tau, p.lam_x, p.lam_y, p.theta), a)
        unc = ep_squared(0.0, var_Z(si, p.lam_x, p.lam_y, p.theta), a)
        tot += wi * fv(si) * cond / unc
    return 1e4 / q._VIX_DELTA * tot


class TestComputeVix2:

    def test_vix2_positive(self, default_params, flat_fwd_var):
        assert compute_vix2(0.0833, 0.0, 0.0, default_params, flat_fwd_var) > 0.0

    def test_polynomial_coefficients_match_direct_integral(self, flat_fwd_var):
        p = QUINTIC_DEFAULTS
        for T, x, y in ((0.06, 0.1, -0.05), (0.4, -0.3, 0.2), (0.9, 0.25, 0.4)):
            assert compute_vix2(T, x, y, p, flat_fwd_var, n_quad=40) == pytest.approx(
                _vix2_direct(T, x, y, p, flat_fwd_var), rel=1e-10)

    def test_split_at_forward_variance_jumps_is_exact(self):
        # ξ_0 jumps at 0.25 inside [T, T+Δ]: the panelled GL integral of the
        # deterministic part (X_T = Y_T = 0 at large T ≈ stationary) must equal
        # the exact integral of ξ_0 scaled by the smooth ratio — check with a
        # constant ratio (p constant ⇒ ratio ≡ 1 ⇒ VIX² = 100²/Δ ∫ ξ_0).
        p = dataclasses.replace(QUINTIC_DEFAULTS, alpha1=0.0, alpha3=0.0, alpha5=0.0, alpha0=1.0)
        fv = _StepCurve()
        T = 0.23
        exact = 1e4 / q._VIX_DELTA * (fv.integrated(T + q._VIX_DELTA) - fv.integrated(T))
        assert compute_vix2(T, 0.0, 0.0, p, fv, n_quad=4) == pytest.approx(exact, rel=1e-12)

    def test_vix2_increases_with_xt_on_increasing_branch(self, default_params, flat_fwd_var):
        # p has non-negative coefficients ⇒ increasing; for X_T ≥ 0 the
        # conditional mean of Z_s moves up the positive branch, so VIX² rises.
        vals = [compute_vix2(0.0833, x, 0.0, default_params, flat_fwd_var) for x in (0.0, 0.5, 1.0, 2.0)]
        assert all(b > a for a, b in zip(vals, vals[1:]))

    def test_matches_conditional_monte_carlo(self, default_params):
        # Brute force: simulate (X_s, Y_s) exactly from (x_T, y_T) and average σ_s²/g0² ratios.
        p, fv = default_params, (lambda t: 0.05)
        T, xT, yT = 0.2, 0.15, -0.05
        rng = np.random.default_rng(7)
        n_t, n_paths = 30, 100_000
        s = T + (np.arange(n_t) + 0.5) * q._VIX_DELTA / n_t
        a = p.alpha_vec()
        acc = np.zeros(n_paths)
        for si in s:
            tau = si - T
            vx, vy, cxy = _cond_var_components(tau, p.lam_x, p.lam_y)
            L11 = math.sqrt(vx); L21 = cxy / L11; L22 = math.sqrt(vy - L21**2)
            z1, z2 = rng.standard_normal(n_paths), rng.standard_normal(n_paths)
            X = xT * math.exp(-p.lam_x * tau) + L11 * z1
            Y = yT * math.exp(-p.lam_y * tau) + L21 * z1 + L22 * z2
            Z = p.theta * X + (1 - p.theta) * Y
            unc = ep_squared(0.0, var_Z(si, p.lam_x, p.lam_y, p.theta), a)
            acc += fv(si) * eval_poly(Z, a)**2 / unc
        mc = 1e4 * acc / n_t           # (100²/Δ)·∫ ≈ 100²·mean over the window
        exact = compute_vix2(T, xT, yT, p, fv, n_quad=40)
        se = mc.std() / math.sqrt(n_paths)
        # midpoint-in-time bias is O(1/n_t²) of a smooth integrand: far below 3 se
        assert abs(mc.mean() - exact) < 3 * se


# ── VIX futures ───────────────────────────────────────────────────────────────

class TestVixFutures:

    def test_vix_futures_positive(self, default_params, flat_fwd_var):
        assert price_vix_futures(0.0833, default_params, flat_fwd_var, n_outer=41, n_inner=10) > 0.0

    def test_jensen_bound_and_variance_swap_level(self, default_params):
        # E[VIX_T] ≤ sqrt(E[VIX_T²]) and E[VIX_T²] = 100²/Δ ∫_T^{T+Δ} ξ_0 (model-free)
        fv = _StepCurve()
        T = 0.2
        vix, w = q.vix_at_nodes(T, default_params, fv)
        fwd_var_swap = 1e4 / q._VIX_DELTA * (fv.integrated(T + q._VIX_DELTA) - fv.integrated(T))
        assert float(w @ vix**2) == pytest.approx(fwd_var_swap, rel=1e-9)
        assert float(w @ vix) < math.sqrt(fwd_var_swap)

    def test_vix_futures_monotone_in_forward_var(self, default_params):
        F_low  = price_vix_futures(0.0833, default_params, lambda t: 0.01)
        F_high = price_vix_futures(0.0833, default_params, lambda t: 0.16)
        assert F_high > F_low

    def test_cubature_converged(self, flat_fwd_var):
        # The default lattice is converged: refining it moves the forward < 1e-6 rel.
        p = QUINTIC_DEFAULTS
        f0 = price_vix_futures(0.1, p, flat_fwd_var)
        f1 = price_vix_futures(0.1, p, flat_fwd_var, n_outer=201)
        assert f0 == pytest.approx(f1, rel=1e-6)


# ── VIX options ───────────────────────────────────────────────────────────────

class TestVixOption:

    def test_call_non_negative(self, default_params, flat_fwd_var):
        assert price_vix_option(0.0833, 25.0, default_params, flat_fwd_var) >= 0.0

    def test_call_bounded_by_futures(self, default_params, flat_fwd_var):
        F = price_vix_futures(0.0833, default_params, flat_fwd_var)
        assert price_vix_option(0.0833, 10.0, default_params, flat_fwd_var) <= F + 1e-9

    def test_call_decreases_with_strike(self, default_params, flat_fwd_var):
        c_lo = price_vix_option(0.0833, 15.0, default_params, flat_fwd_var)
        c_hi = price_vix_option(0.0833, 30.0, default_params, flat_fwd_var)
        assert c_lo >= c_hi

    def test_put_call_parity_on_the_cubature(self, flat_fwd_var):
        K = np.array([12.0, 18.0, 20.0, 25.0, 40.0])
        F, calls, puts = vix_option_prices(0.3, K, QUINTIC_DEFAULTS, flat_fwd_var)
        np.testing.assert_allclose(calls - puts, F - K, atol=1e-10)

    def test_iv_lattice_convergence(self, flat_fwd_var):
        # Black-76 IV on the kinked payoff: default lattice within 0.1 vp of a
        # 2.5× finer one (a 48² Gauss-Hermite rule was off by up to 0.8 vp).
        p, T = QUINTIC_DEFAULTS, 0.06
        K = np.array([18.0, 20.0, 23.0, 30.0, 40.0])
        ivs = []
        for n in (q._N_QUAD_OUTER, 251):
            F, calls, puts = vix_option_prices(T, K, p, flat_fwd_var, n_outer=n)
            is_call = K >= F
            ivs.append(q.black76_implied_vol(np.where(is_call, calls, puts), F, K, T, is_call))
        assert np.nanmax(np.abs(ivs[0] - ivs[1])) < 0.001

    def test_undiscounted_premium_inverts_without_discounting(self, default_params, flat_fwd_var):
        # price_vix_option is an undiscounted premium: inverting it with r = 0
        # (or discount = 1) and repricing must give the premium back, while the
        # old r = 4.5% inversion treated it as a present value and inflated the IV.
        T, K = 0.5, 18.0
        F = price_vix_futures(T, default_params, flat_fwd_var)
        prem = price_vix_option(T, K, default_params, flat_fwd_var)
        iv = q.vix_option_implied_vol(T, K, F, prem, r=0.0)
        assert float(q.black76_price(F, K, T, iv)) == pytest.approx(prem, rel=1e-10)
        iv_wrong = q.vix_option_implied_vol(T, K, F, prem, r=0.045)
        assert iv_wrong > iv + 0.005


# ── Black-76 ─────────────────────────────────────────────────────────────────

class TestBlack76:

    def test_round_trip_discounted_calls_and_puts(self):
        F, T, r = 24.0, 0.5, 0.045
        K = np.array([12.0, 18.0, 24.0, 30.0, 60.0])
        D = math.exp(-r * T)
        for is_call in (True, False):
            pv = q.black76_price(F, K, T, 0.8, is_call, D)
            np.testing.assert_allclose(q.black76_implied_vol(pv, F, K, T, is_call, D), 0.8, atol=1e-10)
        assert q.vix_option_implied_vol(T, 18.0, F, float(q.black76_price(F, 18.0, T, 0.8, True, D)),
                                        r) == pytest.approx(0.8, abs=1e-10)

    def test_outside_no_arbitrage_band_is_nan(self):
        F, T = 100.0, 0.5
        assert np.isnan(q.black76_implied_vol(9.99, F, 90.0, T, True))     # below intrinsic
        assert np.isnan(q.black76_implied_vol(100.0, F, 90.0, T, True))    # at the upper bound


# ── SPX Monte Carlo ───────────────────────────────────────────────────────────

class TestSpxMC:

    def test_call_non_negative(self, default_params, flat_fwd_var):
        prices, _ = price_spx_options_mc(
            S0=5000.0, T=0.25, strikes=np.array([5000.0]), r=0.045, q=0.013,
            params=default_params, fwd_var_func=flat_fwd_var, n_paths=4000, seed=42)
        assert prices[0] >= 0.0

    def test_atm_call_close_to_black_scholes_at_flat_variance(self, default_params, flat_fwd_var):
        # E[∫σ²] = ∫ξ_0 = 0.04·T exactly, so the ATM price is near BS at 20% vol
        S0, T, r, qd = 5000.0, 0.25, 0.045, 0.013
        prices, se = price_spx_options_mc(S0, T, np.array([S0]), r, qd, default_params,
                                          flat_fwd_var, n_paths=20_000, seed=42)
        F = S0 * math.exp((r - qd) * T)
        bs = float(q.black76_price(F, S0, T, 0.20, True, math.exp(-r * T)))
        assert prices[0] == pytest.approx(bs, rel=0.03)

    @pytest.mark.parametrize("which", ["fixture", "paper"])
    @pytest.mark.parametrize("T", [0.25, 1.0])
    def test_martingale(self, which, T, default_params):
        # E[S_T/F] = 1 for the discrete left-point scheme; the pre-C22 scheme
        # (σ from X_{t+dt} times the same shock) gave 0.963 ± 0.0004 at T = 1.
        p = default_params if which == "fixture" else QUINTIC_DEFAULTS
        x = np.exp(q.simulate_terminal_log_spot(T, p, _StepCurve(), n_paths=40_000, seed=11))
        n = len(x) // 2
        pair = 0.5 * (x[:n] + x[n:])
        se = pair.std(ddof=1) / math.sqrt(n)
        assert abs(pair.mean() - 1.0) < 4.0 * se
        assert se < 0.003

    def test_antithetic_pairs_are_mirror_paths(self, default_params):
        logS = q.simulate_terminal_log_spot(0.5, default_params, _StepCurve(), n_paths=20_000, seed=5)
        n = len(logS) // 2
        assert np.corrcoef(logS[:n], logS[n:])[0, 1] < -0.3
        # pair-average standard error beats iid paths of the same total count
        x = np.exp(logS)
        se_pairs = (0.5 * (x[:n] + x[n:])).std(ddof=1) / math.sqrt(n)
        se_iid = x.std(ddof=1) / math.sqrt(2 * n)
        assert se_pairs < 0.5 * se_iid

    def test_simulated_variance_matches_forward_variance(self):
        # E[∫_0^T σ² dt] = ∫_0^T ξ_0 exactly (no floor on σ). The paper's p has
        # a negative branch hit with material probability, which a max(σ, 0)
        # floor would remove.
        p, fv, T = QUINTIC_DEFAULTS, _StepCurve(), 0.6
        t_grid, idx = q.mc_time_grid([T])
        sims = q._simulate_integrals(p, fv, t_grid, idx, 20_000, seed=3)
        V = sims["V"][0]
        n = len(V) // 2
        pair = 0.5 * (V[:n] + V[n:])
        assert abs(pair.mean() - fv.integrated(T)) < 4 * pair.std(ddof=1) / math.sqrt(n)
        # and the paths really do visit p < 0
        z = np.random.default_rng(0).standard_normal(100_000) * math.sqrt(
            var_Z(T, p.lam_x, p.lam_y, p.theta))
        assert np.mean(eval_poly(z, p.alpha_vec()) < 0) > 0.05

    def test_conditional_estimator_agrees_with_plain(self, default_params):
        fv, T = _StepCurve(), 0.5
        K = np.array([85.0, 100.0, 110.0])
        F = np.full(3, 100.0)
        Ts = np.full(3, T)
        cond = q.spx_forward_prices_mc(Ts, K, F, default_params, fv, n_paths=20_000, seed=1,
                                       richardson=False)
        plain = q.spx_forward_prices_mc(Ts, K, F, default_params, fv, n_paths=200_000, seed=2,
                                        method="plain")
        se = np.sqrt(cond["call_se"]**2 + plain["call_se"]**2)
        assert np.all(np.abs(cond["call"] - plain["call"]) < 4 * se)

    def test_parity_and_no_arbitrage(self):
        # Exact put-call parity (control variate on M1), deep-ITM call above
        # intrinsic, and a convex decreasing call curve — at the paper's
        # parameters where the pre-C22 pricer returned sub-intrinsic prices.
        fv, T, F = _StepCurve(), 0.5, 100.0
        K = np.linspace(60.0, 140.0, 17)
        res = q.spx_forward_prices_mc(np.full(17, T), K, np.full(17, F), QUINTIC_DEFAULTS, fv,
                                      n_paths=10_000, seed=4)
        np.testing.assert_allclose(res["call"] - res["put"], F - K, atol=1e-9)
        assert np.all(res["call"] > np.maximum(F - K, 0.0))
        assert np.all(np.diff(res["call"]) < 0)
        assert np.all(np.diff(res["call"], 2) > 0)
        iv = q.black76_implied_vol(np.where(K >= F, res["call"], res["put"]), F, K, T, K >= F)
        assert np.all(np.isfinite(iv))

    def test_step_convergence_with_richardson(self):
        # The plain left-point scheme's O(h) bias is ~1 vp in the 1y call wing
        # at 252 steps/yr for the paper's λx = 31.8 (18.13 vs 17.10 % on a fine
        # grid, 0.3 vp ATM). The Richardson-extrapolated default must sit within
        # 0.3 vp of the same estimator on a 2× finer grid (seed noise of the
        # difference here ≈ 0.06–0.09 vp).
        p, fv = QUINTIC_DEFAULTS, (lambda t: 0.07)
        T = 1.0
        K = 6600.0 * np.exp(np.array([0.0, 0.15]))
        F = np.full(2, 6600.0)
        Ts = np.full(2, T)
        base = q.implied_vol_surface_forward_mc(Ts, K, F, p, fv, n_paths=100_000, seed=21)
        fine = q.implied_vol_surface_forward_mc(Ts, K, F, p, fv, n_paths=100_000, seed=22,
                                                steps_per_year=504, min_steps=200)
        assert np.max(np.abs(base - fine)) < 0.003

    def test_crn_cache_does_not_change_results(self, default_params, monkeypatch):
        fv = _StepCurve()
        args = (np.full(2, 0.3), np.array([90.0, 110.0]), np.full(2, 100.0), default_params, fv)
        q._CRN_CACHE.clear()
        cached = q.implied_vol_surface_forward_mc(*args, n_paths=2000, seed=9)
        monkeypatch.setattr(q, "_CRN_CACHE_MAX_BYTES", 0)
        q._CRN_CACHE.clear()
        drawn = q.implied_vol_surface_forward_mc(*args, n_paths=2000, seed=9)
        np.testing.assert_array_equal(cached, drawn)

    def test_time_grid_contains_expiries_and_respects_step_rules(self):
        Ts = [0.03, 0.2, 0.9]
        t, idx = q.mc_time_grid(Ts)
        np.testing.assert_array_equal(t[idx], Ts)
        h = np.diff(t)
        assert h.max() <= 1 / 252 + 1e-15
        assert idx[0] >= 100 and idx[1] >= 100 and idx[2] >= 0.9 * 252


# ── QuinticOUParams helpers / admissibility ──────────────────────────────────

class TestParams:

    def test_round_trip_array(self):
        p = QUINTIC_DEFAULTS
        p2 = QuinticOUParams.from_array(p.to_array())
        assert p2 == p

    def test_alpha_vec_structure(self):
        p = QuinticOUParams(lam_x=10.0, lam_y=1.0, theta=0.9,
                            alpha0=0.1, alpha1=0.2, alpha3=0.3, alpha5=0.4, epsilon=-0.5)
        np.testing.assert_allclose(p.alpha_vec(), [0.1, 0.2, 0.0, 0.3, 0.0, 0.4])

    def test_bounds_length(self):
        assert len(QUINTIC_BOUNDS) == 8

    def test_defaults_are_the_papers_admissible_fit(self):
        p = QUINTIC_DEFAULTS
        assert p.alpha5 == 1.0 and p.epsilon < 0.0 and p.lam_x > p.lam_y
        assert q.admissibility_violations(p) == []

    def test_scale_invariance_of_p(self, flat_fwd_var):
        # p and c·p (c > 0) are the same model — the reason α5 is fixed to 1.
        p = QUINTIC_DEFAULTS
        s = dataclasses.replace(p, alpha0=3 * p.alpha0, alpha1=3 * p.alpha1,
                                alpha3=3 * p.alpha3, alpha5=3.0)
        assert price_vix_futures(0.2, s, flat_fwd_var) == pytest.approx(
            price_vix_futures(0.2, p, flat_fwd_var), rel=1e-12)
        K = np.array([90.0, 100.0])
        a = q.implied_vol_surface_forward_mc(np.full(2, 0.3), K, np.full(2, 100.0), p,
                                             flat_fwd_var, n_paths=2000, seed=1)
        b = q.implied_vol_surface_forward_mc(np.full(2, 0.3), K, np.full(2, 100.0), s,
                                             flat_fwd_var, n_paths=2000, seed=1)
        np.testing.assert_allclose(a, b, rtol=1e-10)
        np.testing.assert_allclose(s.normalised().to_array(), p.to_array(), rtol=1e-12)

    def test_admissibility_flags(self):
        p = QUINTIC_DEFAULTS
        assert any("martingale" in v for v in q.admissibility_violations(
            dataclasses.replace(p, alpha5=-0.5)))
        assert any("martingale" in v for v in q.admissibility_violations(
            dataclasses.replace(p, epsilon=0.2)))
        assert any("alpha1" in v for v in q.admissibility_violations(
            dataclasses.replace(p, alpha1=-0.9)))
        assert any("theta" in v for v in q.admissibility_violations(
            dataclasses.replace(p, theta=2.85)))
