"""
test_rough_heston.py — lifted rough Heston (models/rough_heston.py, C23).

Independent checks, none of which reuse the code under test as its own oracle:
  1. Kernel: non-negative weights/rates, the H = 1/2 collapse to one node
     (c = 1, x = 0), and convergence of K_n to t^(H-1/2)/Gamma(H+1/2) in n.
  2. H = 1/2 is classical Heston: the log-CF against the closed-form little
     trap, the prices against heston_call_batch, second-order convergence in
     the Riccati step count.
  3. The mean variance against the Mittag-Leffler closed form of the EXACT
     rough kernel (series, written here), so the lifting error is measured.
  4. Frozen-grid surface pricer: equal to the adaptive engine at the adapted
     point, smooth nearby; exact theta / v0 derivatives vs differences.
  5. No-arbitrage shape of rough prices (decreasing, convex in strike).
  6. A synthetic rough market is recovered by the grid-refreshing TRF.
  7. QuantLib 1.44's AnalyticRoughHestonEngine as a DEV-ONLY oracle
     (pytest.importorskip): the Lifted route agrees with ours node for node;
     the Adams route (exact fractional Riccati) measures the lifting error.
"""

import itertools
import math

import numpy as np
import pytest

from joint_vol_calibration.models import rough_heston as rh
from joint_vol_calibration.models.heston import _heston_log_cf, heston_call_batch

_P = (4.992, 0.0499, 2.0, -0.753, 0.0344)       # 2026-10-16 production fit (rounded)
_PR = (0.6, 0.05, 0.55, -0.7, 0.03)             # a rough-Heston-like set (sigma interior)
_F, _R, _Q = 100.0, 0.04, 0.01
_K = np.array([70.0, 80, 90, 95, 100, 105, 110, 120])


# ── 1. kernel ────────────────────────────────────────────────────────────────

def test_kernel_collapses_to_one_node_at_half():
    c, x = rh.lifted_kernel(0.5, 20, 1.0)
    assert c[0] == pytest.approx(1.0, abs=1e-15)
    assert x[0] == 0.0
    np.testing.assert_array_equal(c[1:], 0.0)


@pytest.mark.parametrize("H", [0.05, 0.1, 0.3, 0.45])
def test_kernel_nonnegative_and_converges_in_n(H):
    t = np.geomspace(1e-3, 1.0, 40)
    exact = t ** (H - 0.5) / math.gamma(H + 0.5)
    errs = []
    for n in (5, 20, 80):
        c, x = rh.lifted_kernel(H, n, 1.0)
        assert np.all(c > 0)
        assert np.all(x >= 0)
        assert np.all(np.diff(x) > 0)
        Kn = (c[None, :] * np.exp(-x[None, :] * t[:, None])).sum(axis=1)
        errs.append(float(np.max(np.abs(Kn / exact - 1))))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[2] < 5e-3, errs


def test_kernel_scales_with_maturity():
    """x_i(T) = x_i(1)/T and c_i(T) = c_i(1) T^(H-1/2): what lets every
    expiry share one sweep (rough_heston._unit_tables)."""
    H, T = 0.17, 0.0383
    c1, x1 = rh.lifted_kernel(H, 20, 1.0)
    cT, xT = rh.lifted_kernel(H, 20, T)
    np.testing.assert_allclose(xT, x1 / T, rtol=1e-12)
    np.testing.assert_allclose(cT, c1 * T ** (H - 0.5), rtol=1e-12)


def test_invalid_hurst_rejected():
    for H in (0.0, -0.1, 0.51):
        with pytest.raises(ValueError, match="H must be in"):
            rh.lifted_kernel(H, 20, 1.0)
    with pytest.raises(ValueError, match="H must be in"):
        rh.rough_heston_log_cf(np.array([1.0 - 0.5j]), 0.5, *_P, 0.6)


# ── 2. H = 1/2 is Heston ─────────────────────────────────────────────────────

@pytest.mark.parametrize("T", [14 / 365, 91 / 365, 1.0])
def test_half_is_heston_cf_with_second_order_convergence(T):
    z = np.linspace(0.0, 150.0, 301) - 0.5j
    ref = np.exp(_heston_log_cf(z, T, *_P))
    errs = [float(np.max(np.abs(np.exp(rh.rough_heston_log_cf(z, T, *_P, 0.5, n_steps=N)) - ref)))
            for N in (100, 200, 400)]
    assert errs[1] < 1e-5
    # implicit trapezoid: x4 per doubling of the steps
    for a, b in itertools.pairwise(errs):
        assert 3.0 < a / b < 5.0, errs


@pytest.mark.parametrize("T", [7 / 365, 30 / 365, 1.0])
def test_half_prices_match_heston_engine(T):
    rough = rh.rough_heston_call_batch(_F, _K, T, _R, _Q, *_P, 0.5)
    np.testing.assert_allclose(rough, heston_call_batch(_F, _K, T, _R, _Q, *_P), rtol=0, atol=1e-5 * _F)
    fine = rh.rough_heston_call_batch(_F, _K, T, _R, _Q, *_P, 0.5, n_steps=800)
    np.testing.assert_allclose(fine, heston_call_batch(_F, _K, T, _R, _Q, *_P), rtol=0, atol=1e-6 * _F)


def test_continuous_in_hurst_at_half():
    a = rh.rough_heston_call_batch(_F, _K, 0.25, _R, _Q, *_PR, 0.5)
    b = rh.rough_heston_call_batch(_F, _K, 0.25, _R, _Q, *_PR, 0.4999)
    assert np.max(np.abs(a - b)) < 1e-4 * _F


# ── 3. mean variance vs Mittag-Leffler (exact rough kernel) ───────────────────

def _mittag_leffler(alpha, beta, z, terms=400):
    """E_{alpha,beta}(z) = sum_k z^k / Gamma(alpha k + beta), in log space."""
    total = 0.0
    for k in range(terms):
        mag = k * math.log(abs(z)) - math.lgamma(alpha * k + beta) if z != 0 else (
            0.0 if k == 0 else -np.inf)
        total += (1.0 if (z >= 0 or k % 2 == 0) else -1.0) * math.exp(mag)
    return total


@pytest.mark.parametrize(("H", "tol"), [(0.02, 5e-3),   # the box floor, where joint fits land
                                        (0.1, 3e-3), (0.3, 3e-3)])
@pytest.mark.parametrize("T", [30 / 365, 1.0])
def test_expected_variance_matches_mittag_leffler(H, tol, T):
    """Exact rough Heston: E[V_t] = V0 + (theta - V0)(1 - E_a(-kappa t^a)),
    so int_0^T E[V] = theta T - (theta - V0) T E_{a,2}(-kappa T^a), a = H + 1/2."""
    kappa, theta, _, _, v0 = (2.0, 0.06, 0.5, -0.7, 0.02)
    a = H + 0.5
    exact = theta * T - (theta - v0) * T * _mittag_leffler(a, 2.0, -kappa * T ** a)
    errs = [abs(rh.rough_heston_expected_variance_integral(T, kappa, theta, v0, H, n_factors=n)
                / exact - 1) for n in (20, 40, 80)]
    # the 20-factor lifting error: measured 0.07%-0.26% at H = 0.1-0.3 and
    # 0.34%-0.44% at H = 0.02
    assert errs[0] < tol
    # and it shrinks like n^-1.5 (ratio 2^1.5 = 2.8 per doubling)
    for a, b in itertools.pairwise(errs):
        assert 2.2 < a / b < 3.5, errs


def test_vix_index_at_half_is_heston():
    from joint_vol_calibration.models.heston import heston_vix_term_index
    T = np.array([9, 30, 93, 184]) / 365.0
    np.testing.assert_allclose(rh.rough_heston_vix_term_index(*_P[:3], _P[4], 0.5, T),
                               heston_vix_term_index(*_P[:3], _P[4], T), rtol=1e-10)


# ── 4. frozen-grid surface pricer ─────────────────────────────────────────────

_SLICES = [(14 / 365, 7700.0, 0.998, np.linspace(7000, 7900, 7)),
           (91 / 365, 7760.0, 0.990, np.linspace(6500, 8100, 7)),
           (1.0, 7900.0, 0.960, np.linspace(5800, 8600, 7))]


def test_frozen_pricer_equals_adaptive_and_moves_smoothly():
    p = np.r_[_PR, 0.15]
    pr = rh.RoughHestonSurfacePricer(_SLICES)
    with pytest.raises(RuntimeError):
        pr.prices(p)
    adaptive = pr.adapt(p)
    frozen = pr.prices(p)
    for a, b, s in zip(adaptive, frozen, _SLICES):
        np.testing.assert_allclose(b, a, rtol=0, atol=1e-12 * s[1])
    # one sweep for all slices == each slice priced on its own
    for j, s in enumerate(_SLICES):
        own = rh.rough_heston_call_batch_grid(s[1], s[3], s[0], -math.log(s[2]) / s[0],
                                              -math.log(s[2]) / s[0], *p, grid=pr.grids[j])[0]
        np.testing.assert_allclose(frozen[j], own, rtol=0, atol=1e-12 * s[1])
    q = p * np.array([1.05, 1.03, 0.97, 1.02, 1.04, 1.1])
    moved = pr.prices(q)
    fresh = [pr._price_slice(j, q)[0] for j in range(len(_SLICES))]
    for a, b, s in zip(moved, fresh, _SLICES):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-7 * s[1])


def test_theta_v0_derivatives_are_exact():
    p = np.r_[_PR, 0.2]
    pr = rh.RoughHestonSurfacePricer(_SLICES)
    pr.adapt(p)
    d = pr.theta_v0_derivatives(p)
    for col, j in ((0, 1), (1, 4)):
        h = 1e-6 * p[j]
        up, dn = p.copy(), p.copy()
        up[j] += h
        dn[j] -= h
        fd = [(a - b) / (2 * h) for a, b in zip(pr.prices(up), pr.prices(dn))]
        for dj, fj in zip(d, fd):
            np.testing.assert_allclose(dj[:, col], fj, rtol=1e-6, atol=1e-6 * np.abs(fj).max())


# ── 5. shape ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("H", [0.05, 0.15, 0.35])
def test_rough_prices_decreasing_and_convex_in_strike(H):
    """No-arbitrage shape, to the quadrature target the engine promises
    (ROUGH_LEWIS_RTOL * F; 1-week far-wing prices are ~1e-9 * F)."""
    tol = rh.ROUGH_LEWIS_RTOL * _F
    K = np.linspace(60.0, 140.0, 41)
    for T in (14 / 365, 0.5):
        c = rh.rough_heston_call_batch(_F, K, T, _R, _Q, *_PR, H)
        assert np.all(np.diff(c) <= 2 * tol)
        assert np.all(np.diff(c, 2) >= -4 * tol)
        assert np.all(c <= _F * math.exp(-_Q * T) + 1e-12)
        # and far tighter than that where the price is not negligible
        big = c[1:-1] > 1e-3 * _F
        assert np.all(np.diff(c, 2)[big] > 0)


# ── 6. synthetic rough market recovered ─────────────────────────────────────

def test_grid_refreshing_trf_recovers_a_synthetic_rough_market():
    true = np.r_[_PR, 0.2]
    groups = []
    for T, F, D, K in _SLICES:
        r = -math.log(D) / T
        calls = rh.rough_heston_call_batch(F, K, T, r, r, *true, n_steps=100)
        is_put = K < F
        mkt = calls.copy()
        mkt[is_put] = calls[is_put] - D * (F - K[is_put])
        groups.append((T, F, D, K, is_put, mkt, np.full(len(K), 0.01 * F)))
    resid = rh.RoughHestonSPXResiduals(groups, n_steps=100)
    lo = np.array([0.1, 1e-4, 1e-3, -0.95, 1e-4, 0.02])
    hi = np.array([20.0, 1.0, 2.0, 0.0, 1.0, 0.5])
    x0 = true * np.array([1.4, 0.8, 1.3, 0.85, 1.2, 1.6])
    x, loss, runs = rh.fit_trf_with_grid_refresh(resid, [x0], ["near"], lo, hi)
    assert loss < 1e-14
    assert x[5] == pytest.approx(0.2, abs=0.01)
    assert runs[0]["loss"] == loss
    assert runs[0]["status"] > 0


# ── 7. QuantLib oracle (dev-only) ────────────────────────────────────────────

def _ql_rough_calls(ql, p, H, days, K, approx, steps):
    kappa, theta, sigma, rho, v0 = p
    today = ql.Date(16, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    rts = ql.YieldTermStructureHandle(ql.FlatForward(today, _R, dc))
    qts = ql.YieldTermStructureHandle(ql.FlatForward(today, _Q, dc))
    proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(_F)), v0, kappa, theta,
                            sigma, rho)
    eng = ql.AnalyticRoughHestonEngine(ql.RoughHestonModel(proc, H), 128, steps,
                                       getattr(ql.AnalyticRoughHestonEngine, approx), 20)
    out = []
    for k in K:
        o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, float(k)),
                             ql.EuropeanExercise(today + days))
        o.setPricingEngine(eng)
        out.append(o.NPV())
    return np.array(out)


@pytest.mark.parametrize("p", [_P, _PR])
@pytest.mark.parametrize("H", [0.1, 0.3, 0.49])
@pytest.mark.parametrize("days", [30, 365])
def test_matches_quantlib_lifted_engine(p, H, days):
    """Same kernel partition, independent ODE scheme (QuantLib: uniform steps,
    explicit predictor; here: graded steps, implicit quadratic step): they
    must agree to the time-stepping error. Integer days: QuantLib's Python
    dates are whole days, so T = days/365 exactly on both sides."""
    ql = pytest.importorskip("QuantLib")
    K = np.array([80.0, 90, 100, 110, 120])
    ours = rh.rough_heston_call_batch(_F, K, days / 365, _R, _Q, *p, H, n_steps=800)
    theirs = _ql_rough_calls(ql, p, H, days, K, "Lifted", 2000)
    assert np.max(np.abs(ours - theirs)) < 2e-6 * _F


@pytest.mark.parametrize("H", [0.1, 0.3])
def test_lifting_error_against_quantlib_adams(H):
    """QuantLib's Adams scheme solves the exact fractional Riccati equation;
    the gap to it is the 20-factor lifting error, measured (not hidden):
    <= 1e-4 * F on these sets (it was 2.3e-5 to 8.5e-5 * F)."""
    ql = pytest.importorskip("QuantLib")
    K = np.array([80.0, 90, 100, 110, 120])
    for p in (_P, _PR):
        for days in (30, 365):
            ours = rh.rough_heston_call_batch(_F, K, days / 365, _R, _Q, *p, H)
            adams = _ql_rough_calls(ql, p, H, days, K, "AdamsPredictorCorrector", 2048)
            assert np.max(np.abs(ours - adams)) < 1e-4 * _F


def test_lifting_error_at_the_floor_where_the_joint_fit_lands():
    """The joint (SPX + VIX index) rough fit of 2026-10-16 sits on the
    H = 0.02 floor (data_store/calibrations/diagnostics/2026-10-16.json), so
    the 20-factor lifting error is measured THERE against QuantLib's Adams
    scheme for the exact fractional Riccati equation, 14 days to 1 year:
    measured <= 7.7e-5 * F in price and <= 0.085 vp in implied vol (largest
    at 14 days), against a 1.08 vp in-sample misfit at that fit. The VIX
    index leg's error is the mean's: <= 0.02 VIX points (Mittag-Leffler).
    Adams at 2048 steps is converged here (to 6e-7 * F of 8192 steps); it
    is NOT at sigma = 2, H = 0.02 (6e-5 * F), which is why the generic
    Adams test above stops at H = 0.1."""
    ql = pytest.importorskip("QuantLib")
    from joint_vol_calibration.models.implied_vol import implied_vol_black
    p = (1.5791, 0.0538, 0.8772, -0.7123, 0.0352)       # 2026-10-16 joint rough fit (rounded)
    K = np.array([85.0, 90, 95, 100, 105])
    for days in (14, 30, 91, 365):
        T = days / 365
        ours = rh.rough_heston_call_batch(_F, K, T, _R, _Q, *p, 0.02)
        adams = _ql_rough_calls(ql, p, 0.02, days, K, "AdamsPredictorCorrector", 2048)
        assert np.max(np.abs(ours - adams)) < 1e-4 * _F
        Fw, D = _F * math.exp((_R - _Q) * T), math.exp(-_R * T)
        d_iv = [implied_vol_black(a, Fw, k, T, D, True) - implied_vol_black(b, Fw, k, T, D, True)
                for a, b, k in zip(ours, adams, K)]
        assert np.max(np.abs(d_iv)) * 100 < 0.1
    kappa, theta, _, _, v0 = p
    a = 0.52
    for days in (9, 30, 93, 184):
        T = days / 365
        exact = theta * T - (theta - v0) * T * _mittag_leffler(a, 2.0, -kappa * T ** a)
        lifted = rh.rough_heston_expected_variance_integral(T, kappa, theta, v0, 0.02)
        assert abs(100 * math.sqrt(lifted / T) - 100 * math.sqrt(exact / T)) < 0.02


def test_an_unresolved_far_frequency_does_not_inflate_the_lewis_grid():
    """Found when the 2026-10-27 calibration diagnostics timed out on the bot
    (20 min, then 12 min): near H = 1/2 the time grid cannot resolve one
    truncation-scan frequency (u = 4.95e5, |phi| = 8.6e-4 where the true value is
    ~0), which pushed the Lewis truncation of the 1-year slice to u = 5e4 and its
    grid to 80,293 nodes (1.02 million on 2026-10-26's parameters), so each
    residual evaluation took seconds. phi is zero above a frequency where it has
    decayed below e^-70; the prices do not move."""
    F, T, r = 7740.0, 0.9649, 0.04
    K = F * np.linspace(0.72, 1.27, 8)
    p = (3.8304, 0.0576, 1.6295, -0.6851, 0.0249)          # the 2026-10-27 joint fit
    calls, grid = rh.rough_heston_call_batch_grid(F, K, T, r, r, *p, 0.4888)
    assert grid is not None
    assert grid.n < 2_000
    # the same prices as a run with four times the Riccati steps (which resolves it)
    fine, _ = rh.rough_heston_call_batch_grid(F, K, T, r, r, *p, 0.4888,
                                              n_steps=4 * rh.ROUGH_HESTON_TIME_STEPS)
    np.testing.assert_allclose(calls, fine, atol=1e-6 * F)
    # the mask leaves a CF that decays monotonically, and anything below the floor, alone
    z = np.array([1.0, 2.0, 3.0, 4.0]) - 0.5j
    lc = np.array([-0.1, -80.0, -5.0, -90.0]) + 0j
    out = rh._mask_rebound(lc, z)
    assert np.array_equal(out[:2], lc[:2])
    assert np.isneginf(out[2:].real).all()
    smooth = np.array([-0.1, -1.0, -10.0, -60.0]) + 0j
    assert np.array_equal(rh._mask_rebound(smooth, z), smooth)
