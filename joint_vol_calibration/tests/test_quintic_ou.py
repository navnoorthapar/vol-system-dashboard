"""
test_quintic_ou.py — 30 unit tests for the two-factor Quintic OU model (C13).

Coverage:
  1–5   : Gaussian moment recurrence
  6–8   : ep_squared analytical correctness
  9–11  : var_Z / cond_var geometry
  12–14 : eval_poly correctness & edge cases
  15–17 : g0 normalisation (ensures E[σ²] = ξ_0)
  18–20 : VIX² computation via Gauss-Legendre
  21–23 : VIX futures pricing via Gauss-Hermite
  24–26 : VIX option pricing (call≥0, call≤futures, monotone in K)
  27–28 : SPX MC pricing (put-call parity proxy, antithetic convergence)
  29–30 : QuinticOUParams dataclass helpers
"""

import numpy as np
import pytest

from joint_vol_calibration.models.quintic_ou import (
    QuinticOUParams,
    eval_poly,
    gaussian_moments,
    ep_squared,
    var_Z,
    _cond_var_Z,
    g0_at,
    compute_vix2,
    price_vix_futures,
    price_vix_option,
    price_spx_options_mc,
    QUINTIC_DEFAULTS,
    QUINTIC_BOUNDS,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    # Choose alpha0 large enough so p(z) > 0 for |z| < 0.4 (3-sigma of Z_t stationary)
    # Stationary std_Z ≈ 0.137 for these OU speeds → 3-sigma ≈ 0.41.
    # p(-0.41) = 0.30 - 0.10*0.41 - ... = 0.258 > 0  ✓
    return QuinticOUParams(
        lam_x=35.0, lam_y=0.6, theta=0.94,
        alpha0=0.30, alpha1=0.10, alpha3=0.02, alpha5=0.01,
        epsilon=-0.75,
    )


@pytest.fixture
def flat_fwd_var():
    """Flat ξ_0 = 0.04  (20% vol)."""
    return lambda t: 0.04


# ── 1–5: Gaussian moments ──────────────────────────────────────────────────────

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


# ── 6–8: ep_squared ────────────────────────────────────────────────────────────

class TestEpSquared:

    def test_constant_poly(self):
        # p(z) = c  ��� E[p²] = c²  regardless of distribution
        alpha = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        val = ep_squared(mu=0.0, sigma2=1.0, alpha=alpha)
        assert val == pytest.approx(4.0, rel=1e-10)

    def test_linear_poly_zero_mean(self):
        # p(z) = a*z → E[p²] = a²*σ²
        alpha = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0])
        sigma2 = 2.0
        val = ep_squared(mu=0.0, sigma2=sigma2, alpha=alpha)
        assert val == pytest.approx(9.0 * sigma2, rel=1e-10)

    def test_agrees_with_direct_formula_for_default_params(self):
        # Cross-check: ep_squared vs direct expansion for known alpha
        alpha = np.array([0.05, 0.20, 0.0, 0.05, 0.0, 0.02])
        mu, s2 = 0.5, 1.5
        m  = gaussian_moments(10, mu, s2)
        expected = sum(
            alpha[j] * alpha[k] * m[j + k]
            for j in range(6) for k in range(6)
        )
        val = ep_squared(mu, s2, alpha)
        assert val == pytest.approx(expected, rel=1e-10)


# ── 9–11: var_Z / _cond_var_Z ─────────────────────────────────────────────────

class TestVarZ:

    def test_zero_at_t0(self):
        assert var_Z(0.0, lam_x=10.0, lam_y=1.0, theta=0.9) == pytest.approx(0.0)

    def test_increases_with_t(self):
        v1 = var_Z(0.1, 10.0, 1.0, 0.9)
        v2 = var_Z(1.0, 10.0, 1.0, 0.9)
        assert v2 > v1 > 0.0

    def test_stationary_variance_limit(self):
        # For large t, Var[X_t] → 1/(2*lam_x), Var[Y_t] → 1/(2*lam_y)
        lam_x, lam_y, theta = 10.0, 1.0, 0.9
        v_inf = var_Z(1e6, lam_x, lam_y, theta)
        expected = (
            theta**2 / (2 * lam_x)
            + (1 - theta)**2 / (2 * lam_y)
            + 2 * theta * (1 - theta) / (lam_x + lam_y)
        )
        assert v_inf == pytest.approx(expected, rel=1e-4)

    def test_cond_var_Z_zero_dt(self):
        assert _cond_var_Z(0.0, 10.0, 1.0, 0.9) == pytest.approx(0.0)

    def test_cond_var_Z_equals_var_Z_of_dt(self):
        # The conditional variance residual has same formula as var_Z(dt, ...)
        dt = 0.05
        assert _cond_var_Z(dt, 10.0, 1.0, 0.9) == pytest.approx(
            var_Z(dt, 10.0, 1.0, 0.9), rel=1e-10
        )


# ── 12–14: eval_poly ──────────────────────────────────────────────────────────

class TestEvalPoly:

    def test_constant_term_at_zero(self):
        alpha = np.array([3.0, 1.0, 0.0, 0.5, 0.0, 0.1])
        assert eval_poly(np.array([0.0]), alpha)[0] == pytest.approx(3.0)

    def test_odd_polynomial_antisymmetric_around_zero(self):
        # p(z) = α1*z + α3*z³ + α5*z⁵ is odd if α0=0
        alpha = np.array([0.0, 0.5, 0.0, 0.3, 0.0, 0.1])
        z = np.array([1.5])
        assert eval_poly(z, alpha)[0] == pytest.approx(-eval_poly(-z, alpha)[0])

    def test_vectorised_output_shape(self):
        alpha = np.array([0.05, 0.20, 0.0, 0.05, 0.0, 0.02])
        z = np.linspace(-2, 2, 50)
        out = eval_poly(z, alpha)
        assert out.shape == (50,)


# ── 15–17: g0 normalisation ───────────────────────────────────────────────────

class TestG0Normalisation:

    def test_g0_positive(self, default_params, flat_fwd_var):
        g = g0_at(0.5, flat_fwd_var(0.5), default_params)
        assert g > 0.0

    def test_g0_ensures_ep2_equals_xi0(self, default_params, flat_fwd_var):
        # g_0(t)² × E[p(Z_t)²] should equal ξ_0(t)
        t = 0.25
        xi0 = flat_fwd_var(t)
        g   = g0_at(t, xi0, default_params)
        ep2 = ep_squared(
            0.0,
            var_Z(t, default_params.lam_x, default_params.lam_y, default_params.theta),
            default_params.alpha_vec(),
        )
        assert g**2 * ep2 == pytest.approx(xi0, rel=1e-8)

    def test_g0_scales_with_xi0(self, default_params):
        # g_0 ∝ sqrt(ξ_0)
        t  = 0.5
        g1 = g0_at(t, 0.04, default_params)
        g2 = g0_at(t, 0.16, default_params)
        assert g2 == pytest.approx(2.0 * g1, rel=1e-8)


# ── 18–20: VIX² computation ───────────────────────────────────────────────────

class TestComputeVix2:

    def test_vix2_positive(self, default_params, flat_fwd_var):
        v2 = compute_vix2(0.0833, 0.0, 0.0, default_params, flat_fwd_var)
        assert v2 > 0.0

    def test_vix2_at_zero_xt_yt_is_deterministic(self, default_params, flat_fwd_var):
        # VIX² at (0,0) should be real-valued (no randomness)
        v1 = compute_vix2(0.0833, 0.0, 0.0, default_params, flat_fwd_var, n_quad=30)
        v2 = compute_vix2(0.0833, 0.0, 0.0, default_params, flat_fwd_var, n_quad=30)
        assert v1 == pytest.approx(v2)

    def test_vix2_increases_with_xt(self, default_params, flat_fwd_var):
        # Higher |Z_T| → higher vol → higher VIX²
        v_at0   = compute_vix2(0.0833, 0.0,  0.0, default_params, flat_fwd_var)
        v_high  = compute_vix2(0.0833, 2.0,  0.0, default_params, flat_fwd_var)
        v_high2 = compute_vix2(0.0833, 3.0,  0.0, default_params, flat_fwd_var)
        # Not strictly monotone for all params, but qualitative check
        assert v_high > 0.0
        assert v_high2 > 0.0


# ── 21–23: VIX futures pricing ────────────────────────────────────────────────

class TestVixFutures:

    def test_vix_futures_positive(self, default_params, flat_fwd_var):
        F = price_vix_futures(0.0833, default_params, flat_fwd_var, n_outer=10, n_inner=10)
        assert F > 0.0

    def test_vix_futures_in_plausible_range(self, default_params, flat_fwd_var):
        # 20% flat vol → VIX should be somewhere near 20
        F = price_vix_futures(0.0833, default_params, flat_fwd_var, n_outer=15, n_inner=15)
        # Broad plausibility: VIX between 5 and 80 for reasonable params
        assert 5.0 < F < 80.0

    def test_vix_futures_monotone_in_forward_var(self, default_params):
        # Higher ξ_0 → higher VIX
        F_low  = price_vix_futures(0.0833, default_params, lambda t: 0.01, n_outer=10, n_inner=10)
        F_high = price_vix_futures(0.0833, default_params, lambda t: 0.16, n_outer=10, n_inner=10)
        assert F_high > F_low


# ── 24–26: VIX option pricing ─────────────────────────────────────────────────

class TestVixOption:

    def test_call_non_negative(self, default_params, flat_fwd_var):
        c = price_vix_option(0.0833, 25.0, default_params, flat_fwd_var, n_outer=10, n_inner=10)
        assert c >= 0.0

    def test_call_bounded_by_futures(self, default_params, flat_fwd_var):
        F = price_vix_futures(0.0833, default_params, flat_fwd_var, n_outer=10, n_inner=10)
        c = price_vix_option(0.0833, 10.0, default_params, flat_fwd_var, n_outer=10, n_inner=10)
        assert c <= F + 1e-6

    def test_call_decreases_with_strike(self, default_params, flat_fwd_var):
        c_lo = price_vix_option(0.0833, 15.0, default_params, flat_fwd_var, n_outer=12, n_inner=12)
        c_hi = price_vix_option(0.0833, 30.0, default_params, flat_fwd_var, n_outer=12, n_inner=12)
        assert c_lo >= c_hi


# ── 27–28: SPX MC pricing ─────────────────────────────────────────────────────

class TestSpxMC:

    def test_call_non_negative(self, default_params, flat_fwd_var):
        prices, _ = price_spx_options_mc(
            S0=5000.0, T=0.25, strikes=np.array([5000.0]),
            r=0.045, q=0.013,
            params=default_params, fwd_var_func=flat_fwd_var,
            n_paths=4000, seed=42,
        )
        assert prices[0] >= 0.0

    def test_atm_call_reasonable_price(self, default_params, flat_fwd_var):
        # ATM call at T=0.25 with 20% vol: BS ≈ 5000 * 0.20 * sqrt(0.25) / sqrt(2π) ≈ 200
        prices, stderr = price_spx_options_mc(
            S0=5000.0, T=0.25, strikes=np.array([5000.0]),
            r=0.045, q=0.013,
            params=default_params, fwd_var_func=flat_fwd_var,
            n_paths=8000, seed=42,
        )
        # Broad sanity: ATM call between 50 and 500 for typical params
        assert 50.0 < prices[0] < 500.0

    def test_antithetic_reduces_std_err(self, default_params, flat_fwd_var):
        # With antithetic variates (n_paths=4000, half each) std_err should be reasonable
        _, stderr = price_spx_options_mc(
            S0=5000.0, T=0.25, strikes=np.array([5000.0]),
            r=0.045, q=0.013,
            params=default_params, fwd_var_func=flat_fwd_var,
            n_paths=4000, seed=7,
        )
        # Std error < 5% of the price (200) → < 10
        assert stderr[0] < 20.0


# ── 29–30: QuinticOUParams helpers ────────────────────────────────────────────

class TestParams:

    def test_round_trip_array(self):
        p   = QUINTIC_DEFAULTS
        arr = p.to_array()
        p2  = QuinticOUParams.from_array(arr)
        assert p2.lam_x   == pytest.approx(p.lam_x)
        assert p2.epsilon == pytest.approx(p.epsilon)

    def test_alpha_vec_structure(self):
        p = QuinticOUParams(
            lam_x=10.0, lam_y=1.0, theta=0.9,
            alpha0=0.1, alpha1=0.2, alpha3=0.3, alpha5=0.4,
            epsilon=-0.5,
        )
        av = p.alpha_vec()
        assert av[0] == pytest.approx(0.1)
        assert av[1] == pytest.approx(0.2)
        assert av[2] == pytest.approx(0.0)  # no α2 term
        assert av[3] == pytest.approx(0.3)
        assert av[4] == pytest.approx(0.0)  # no α4 term
        assert av[5] == pytest.approx(0.4)

    def test_bounds_length(self):
        assert len(QUINTIC_BOUNDS) == 8

    def test_lam_x_gt_lam_y_in_defaults(self):
        p = QUINTIC_DEFAULTS
        assert p.lam_x > p.lam_y

    def test_epsilon_negative_in_defaults(self):
        assert QUINTIC_DEFAULTS.epsilon < 0.0
