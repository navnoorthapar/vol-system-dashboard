"""
eSSVI surface (surface/essvi.py): formulas against Gatheral-Jacquier,
Durrleman's g against a finite-difference density, the butterfly conditions
as a seeded property test, the measured counterexample that motivates the
slice-ordering rows, and fits on synthetic surfaces.
"""
import numpy as np
import pytest
from scipy.special import ndtr

from joint_vol_calibration.surface import essvi as es

K_WIDE = np.linspace(-3.0, 2.0, 2001)


def _gj_ssvi(k, theta, phi, rho):
    """Gatheral-Jacquier eq. 4.1 verbatim."""
    return theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))


def test_w_matches_gatheral_jacquier_formula():
    rng = np.random.default_rng(1)
    for _ in range(50):
        th, ps, rh = rng.uniform(1e-3, 0.2), rng.uniform(0.01, 1.5), rng.uniform(-0.95, 0.95)
        np.testing.assert_allclose(es.essvi_w(K_WIDE, th, ps, rh), _gj_ssvi(K_WIDE, th, ps / th, rh),
                                   rtol=1e-12, atol=1e-15)


def test_derivatives_match_finite_differences():
    th, ps, rh = 0.02, 0.3, -0.6
    h = 1e-5
    w, w1, w2 = es.essvi_derivatives(K_WIDE, th, ps, rh)
    wp, wm = es.essvi_w(K_WIDE + h, th, ps, rh), es.essvi_w(K_WIDE - h, th, ps, rh)
    np.testing.assert_allclose(w1, (wp - wm) / (2 * h), rtol=0, atol=1e-8)
    np.testing.assert_allclose(w2, (wp - 2 * w + wm) / h ** 2, rtol=0, atol=2e-4)


def test_durrleman_g_is_the_density_numerator():
    """p(k) = g(k) / sqrt(2 pi w) exp(-d_-^2 / 2) (GJ Lemma 2.2 proof) equals the
    second strike-derivative of forward-normalised call prices."""
    th, ps, rh = 0.03, 0.4, -0.5
    k = np.linspace(-1.0, 0.6, 801)
    w, w1, w2 = es.essvi_derivatives(k, th, ps, rh)
    g = es.durrleman_g(k, w, w1, w2)
    dm = -k / np.sqrt(w) - np.sqrt(w) / 2
    p_g = g / np.sqrt(2 * np.pi * w) * np.exp(-dm ** 2 / 2)

    def call(kk):
        ww = es.essvi_w(kk, th, ps, rh)
        sw = np.sqrt(ww)
        d1 = -kk / sw + sw / 2
        return ndtr(d1) - np.exp(kk) * ndtr(d1 - sw)
    # p is the density of ln(S_T/F): K x d2c/dK2 at K = e^k (F = 1)
    h = 1e-4
    K = np.exp(k)
    cK = lambda KK: call(np.log(KK))                            # noqa: E731
    d2 = (cK(K * (1 + h)) - 2 * cK(K) + cK(K * (1 - h))) / (K * h) ** 2
    np.testing.assert_allclose(K * d2, p_g, rtol=2e-4, atol=1e-6)


def test_butterfly_conditions_give_nonnegative_g_seeded_property():
    """GJ Theorem 4.2: psi(1+|rho|) < 4 and psi^2(1+|rho|) <= 4 theta => g >= 0."""
    rng = np.random.default_rng(20261020)
    n_checked = 0
    for _ in range(300):
        th = 10 ** rng.uniform(-4, -0.5)
        rh = rng.uniform(-0.99, 0.99)
        cap = min(4 / (1 + abs(rh)), 2 * np.sqrt(th / (1 + abs(rh))))
        ps = rng.uniform(0.01, 1.0) * cap
        bm = es.butterfly_margins(th, ps, rh)
        assert bm["lee_wing"] > 0 and bm["gj_curvature"] >= 0
        w, w1, w2 = es.essvi_derivatives(K_WIDE, th, ps, rh)
        assert es.durrleman_g(K_WIDE, w, w1, w2).min() >= -1e-12
        n_checked += 1
    assert n_checked == 300


def test_violating_the_curvature_bound_creates_butterfly_arbitrage():
    th, rh = 0.01, -0.5
    ps = 3.0 * 2 * np.sqrt(th / (1 + abs(rh)))             # 3x the GJ curvature bound
    assert ps * (1 + abs(rh)) < 4                            # the Lee wing bound still holds
    w, w1, w2 = es.essvi_derivatives(K_WIDE, th, ps, rh)
    assert es.durrleman_g(K_WIDE, w, w1, w2).min() < -1e-3


def test_monotone_theta_A_B_is_not_sufficient_for_calendar():
    """The 2026-06-18 pair quoted in the module docstring: theta up, A equal,
    B up -- yet the slices cross for k > 0.0046."""
    t1, p1, r1 = 0.0013236987654983404, 0.042391909254063304, -0.525280445618967
    t2, p2, r2 = 0.0013323408907600048, 0.04434556669120879, -0.5461943600680241
    cm = es.calendar_margins([t1, t2], [p1, p2], [r1, r2])
    assert cm["theta_increment"][0] > 0
    assert cm["A_increment"][0] >= -1e-15 and cm["B_increment"][0] > 0
    k = np.linspace(-0.5, 0.5, 100001)
    d = es.essvi_w(k, t2, p2, r2) - es.essvi_w(k, t1, p1, r1)
    assert d.min() < -2e-5
    assert k[d < 0].min() == pytest.approx(0.0046, abs=2e-4)


K_QUOTES = np.linspace(-0.4, 0.25, 40)


def _synthetic_quotes(surface, Ts, k=K_QUOTES, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    kk, TT, iv, lab = [], [], [], []
    for T in Ts:
        v = surface.iv(k, T)
        kk.append(k)
        TT.append(np.full(len(k), T))
        iv.append(v + noise * rng.standard_normal(len(k)))
        lab.append(np.full(len(k), f"T{T:.4f}"))
    return np.concatenate(kk), np.concatenate(TT), np.concatenate(iv), np.concatenate(lab)


def test_fit_recovers_an_arbitrage_free_surface():
    Ts = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
    theta = 0.04 * Ts * (1 + 0.3 * np.sqrt(Ts))
    psi = 0.8 * np.sqrt(theta)
    true = es.ESSVISurface(T=Ts, theta=theta, psi=psi, rho=np.full(5, -0.6))
    k, T, iv, lab = _synthetic_quotes(true, Ts)
    surf, info = es.fit_essvi(k, T, iv, np.full(len(k), 0.001), lab)
    fitted = np.concatenate([surf.iv(k[lab == e], float(T[lab == e][0])) for e in dict.fromkeys(lab)])
    assert np.max(np.abs(fitted - iv)) < 2e-4                 # 0.02 vol points
    c = surf.conditions()
    assert c["butterfly_ok"] and c["calendar_ok_on_check_grid"]
    assert info["max_constraint_violation"] < 1e-9


def test_fit_keeps_slices_ordered_when_the_data_cross():
    """Two expiries one day apart whose quotes cross (the later one has less
    total variance on the call wing): the fitted slices still never cross."""
    T1, T2 = 28 / 365, 29 / 365
    k = np.linspace(-0.3, 0.15, 46)
    iv1 = 0.16 - 0.5 * k + 1.2 * k * k
    iv2 = 0.16 - 0.62 * k + 1.2 * k * k                       # steeper: lower on the call side
    kk = np.concatenate([k, k])
    TT = np.concatenate([np.full(46, T1), np.full(46, T2)])
    iv = np.concatenate([iv1, iv2])
    lab = np.array(["a"] * 46 + ["b"] * 46)
    assert (iv2 ** 2 * T2 < iv1 ** 2 * T1).any()            # the data really cross
    surf, info = es.fit_essvi(kk, TT, iv, np.full(92, 0.001), lab)
    c = surf.conditions()
    assert c["calendar_ok_on_check_grid"], c
    assert c["slice_ordering_min_dw"] >= -es._ORDER_TOL         # touching, never crossing


def test_params_at_reproduces_slices_and_interpolates_linearly():
    s = es.ESSVISurface(T=[0.1, 0.5], theta=[0.004, 0.02], psi=[0.1, 0.2], rho=[-0.7, -0.5])
    assert s.params_at(0.1) == pytest.approx((0.004, 0.1, -0.7))
    assert s.params_at(0.5) == pytest.approx((0.02, 0.2, -0.5))
    th, ps, rh = s.params_at(0.3)
    assert th == pytest.approx(0.012) and ps == pytest.approx(0.15)
    assert rh * ps == pytest.approx(0.5 * (-0.07) + 0.5 * (-0.1))
    th0, ps0, rh0 = s.params_at(0.05)                          # short end: scale by T/T1
    assert (th0, ps0, rh0) == pytest.approx((0.002, 0.05, -0.7))
    assert s.atm_vol(0.5) == pytest.approx(np.sqrt(0.02 / 0.5))


def test_round_trip_dict():
    s = es.ESSVISurface(T=[0.1, 0.5], theta=[0.004, 0.02], psi=[0.1, 0.2], rho=[-0.7, -0.5],
                        expiries=["a", "b"])
    s2 = es.ESSVISurface.from_dict(s.to_dict())
    np.testing.assert_allclose(s2.w(K_WIDE, 0.3), s.w(K_WIDE, 0.3), rtol=1e-15)
