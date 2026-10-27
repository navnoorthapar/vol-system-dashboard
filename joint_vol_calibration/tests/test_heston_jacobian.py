"""
test_heston_jacobian.py — the analytic parameter Jacobian of the Heston engine (C23).

heston_call_batch_jac differentiates the Lewis quadrature on the grid the
adaptive engine converged on, through a hand-derived gradient of the
little-trap log-CF. Three independent checks:
  1. a 4th-order central finite difference of heston_call_batch itself
     (the production pricer, re-adapting its grid at every bump);
  2. forward-mode automatic differentiation (torch.func.jacfwd) of an
     independent torch transcription of the same CF on the same grid, which
     must agree to rounding: it checks the hand algebra, not the quadrature;
  3. structural cases: prices identical to heston_call_batch, a supplied
     grid reproduces the adaptive result, T <= 0 and clipped quotes have
     zero rows.
The VIX spot-index Jacobian is checked against central differences,
including the small-kappa*T series branch.
"""

import math

import numpy as np
import pytest

from joint_vol_calibration.models import heston as hm
from joint_vol_calibration.models.heston import (
    heston_call_batch,
    heston_call_batch_grid,
    heston_call_batch_jac,
    heston_vix_term_index,
    heston_vix_term_index_jac,
)
from joint_vol_calibration.tests.test_heston_reference import V2_FITS

_F, _R = 7600.0, 0.041
_K = _F * np.exp(np.linspace(-0.35, 0.15, 11))
_T_GRID = (7 / 365, 14 / 365, 0.25, 1.0)
_PARAM_SETS = [tuple(p) for p in V2_FITS.values()] + [
    (2.0, 0.04, 0.5, -0.7, 0.04),
    (1.0, 0.04, 0.3, -0.3, 0.02),
    (12.0, 0.03, 1.2, -0.9, 0.01),
]


def _fd4(p, T, K=_K, F=_F, r=_R, rel=1e-4):
    """4th-order central difference of the production pricer."""
    J = np.zeros((len(K), 5))
    for j in range(5):
        h = rel * abs(p[j])

        def f(dh, j=j):
            pp = list(p)
            pp[j] += dh
            return heston_call_batch(F, K, T, r, r, *pp)
        J[:, j] = (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)
    return J


@pytest.mark.parametrize("p", _PARAM_SETS)
@pytest.mark.parametrize("T", _T_GRID)
def test_matches_fourth_order_finite_differences(p, T):
    _calls, J = heston_call_batch_jac(_F, _K, T, _R, _R, *p)
    Jfd = _fd4(p, T)
    scale = np.abs(Jfd).max(axis=0)
    rel = np.abs(J - Jfd).max(axis=0) / scale
    # FD truncation and grid-switch noise dominate below ~1e-7 at 1 week
    assert rel.max() < 5e-6, rel


def test_small_sigma_floor_still_accurate():
    """sigma = 1e-3 is the box floor; the sigma column has a cancellation
    there (see _heston_log_cf_grad) but stays accurate."""
    K = np.array([60.0, 80, 95, 100, 105, 120, 150])
    for T in (0.1, 1.0):
        p = (2.0, 0.04, 1e-3, -0.5, 0.04)
        _, J = heston_call_batch_jac(100.0, K, T, 0.03, 0.03, *p)
        Jfd = _fd4(p, T, K=K, F=100.0, r=0.03)
        rel = np.abs(J - Jfd).max(axis=0) / np.abs(Jfd).max(axis=0)
        assert rel.max() < 1e-5, rel


def test_prices_are_exactly_the_pricers():
    for p in _PARAM_SETS[:3]:
        for T in _T_GRID:
            calls, _ = heston_call_batch_jac(_F, _K, T, _R, 0.012, *p)
            np.testing.assert_array_equal(calls, heston_call_batch(_F, _K, T, _R, 0.012, *p))
            c2, grid = heston_call_batch_grid(_F, _K, T, _R, 0.012, *p)
            np.testing.assert_array_equal(c2, calls)
            assert grid.n > 10


def test_supplied_grid_reproduces_adaptive_result():
    p = _PARAM_SETS[0]
    for T in _T_GRID:
        _calls, grid = heston_call_batch_grid(_F, _K, T, _R, _R, *p)
        c_a, J_a = heston_call_batch_jac(_F, _K, T, _R, _R, *p)
        c_g, J_g = heston_call_batch_jac(_F, _K, T, _R, _R, *p, grid=grid)
        # same trapezoid, summed in a different order (halving vs one pass)
        np.testing.assert_allclose(c_g, c_a, rtol=0, atol=1e-10 * _F)
        np.testing.assert_allclose(J_g, J_a, rtol=1e-12, atol=1e-12 * np.abs(J_a).max())


def test_expired_and_clipped_quotes_have_zero_rows():
    calls, J = heston_call_batch_jac(_F, _K, 0.0, _R, _R, *_PARAM_SETS[0])
    np.testing.assert_array_equal(J, 0.0)
    np.testing.assert_array_equal(calls, np.maximum(_F - _K, 0.0))
    # 1-week far-OTM calls: the quadrature returns -1e-12-level values that the
    # no-arbitrage clip sets to 0; a clipped price does not move with p
    K = np.array([60.0, 100.0, 150.0, 250.0])
    T, p = 7 / 365, (2.0, 0.04, 0.3, -0.5, 0.04)
    raw, _ = hm._lewis_call_prices(100.0, K, T, 0.03, 0.03,
                                   lambda z: hm._heston_log_cf(z, T, *p),
                                   float(hm._expected_integrated_variance(T, 2.0, 0.04, 0.04)))
    calls, J = heston_call_batch_jac(100.0, K, T, 0.03, 0.03, *p)
    clipped = calls != raw
    assert clipped.any()
    assert not clipped.all()
    np.testing.assert_array_equal(J[clipped], 0.0)
    assert np.abs(J[~clipped]).max() > 0


# ── forward-mode AD of an independent torch transcription ────────────────────

def _torch_log_cf(torch, z, T, p):
    """Little-trap Heston log-CF, transcribed from the textbook form
    (Albrecher et al. 2007) with g = (b - d)/(b + d): a different algebraic
    route from heston._heston_log_cf, differentiated by torch, not by hand."""
    kappa, theta, sigma, rho, v0 = p[0], p[1], p[2], p[3], p[4]
    b = kappa - 1j * rho * sigma * z
    d = torch.sqrt(b * b + sigma * sigma * (z * z + 1j * z))
    g = (b - d) / (b + d)
    e = torch.exp(-d * T)
    D = (b - d) / (sigma * sigma) * (1 - e) / (1 - g * e)
    C = kappa * theta / (sigma * sigma) * ((b - d) * T - 2 * torch.log((1 - g * e) / (1 - g)))
    return C + D * v0


@pytest.mark.parametrize("T", [14 / 365, 0.25, 1.0])
def test_matches_torch_jacfwd_on_the_same_grid(T):
    torch = pytest.importorskip("torch")
    p = _PARAM_SETS[-1] if T < 0.1 else _PARAM_SETS[0]
    _calls, grid = heston_call_batch_grid(_F, _K, T, _R, _R, *p)
    _, J = heston_call_batch_jac(_F, _K, T, _R, _R, *p, grid=grid)

    u = torch.arange(grid.n, dtype=torch.float64) * grid.h
    wts = torch.full((grid.n,), grid.h, dtype=torch.float64)
    wts[0] *= 0.5
    x = torch.log(_F / torch.tensor(_K, dtype=torch.float64))
    D = math.exp(-_R * T)
    pref = D * torch.sqrt(_F * torch.tensor(_K, dtype=torch.float64)) / math.pi
    z = u.to(torch.complex128) - 0.5j

    def calls_of(pt):
        # the w-dependent control variate is constant in p (held fixed)
        phi = torch.exp(_torch_log_cf(torch, z, T, pt))
        integrand = torch.real(torch.exp(1j * x[:, None] * u[None, :])
                               * (-phi / (u * u + 0.25))[None, :]) * wts[None, :]
        return pref * integrand.sum(dim=1)

    pt = torch.tensor(p, dtype=torch.float64)
    J_ad = torch.func.jacfwd(calls_of)(pt).numpy()
    scale = np.abs(J_ad).max(axis=0)
    rel = np.abs(J - J_ad).max(axis=0) / scale
    # an algebra slip would be O(1); the textbook form's own rounding (it
    # divides by sigma^2 and takes log of a ratio) is ~1e-9 at 14 days
    assert rel.max() < 1e-8, rel


# ── VIX spot-index Jacobian ──────────────────────────────────────────────────

@pytest.mark.parametrize("kappa", [5e-3, 0.1, 4.99, 20.0])
def test_vix_index_jacobian_matches_central_differences(kappa):
    T = np.array([9, 30, 93, 184]) / 365.0
    theta, sigma, v0 = 0.05, 1.5, 0.034
    idx, J = heston_vix_term_index_jac(kappa, theta, sigma, v0, T)
    np.testing.assert_allclose(idx, heston_vix_term_index(kappa, theta, sigma, v0, T), rtol=1e-15)
    base = [kappa, theta, sigma, -0.7, v0]
    for j in range(5):
        h = 1e-6 * max(abs(base[j]), 1e-3)
        up, dn = list(base), list(base)
        up[j] += h
        dn[j] -= h
        fd = (heston_vix_term_index(up[0], up[1], up[2], up[4], T)
              - heston_vix_term_index(dn[0], dn[1], dn[2], dn[4], T)) / (2 * h)
        np.testing.assert_allclose(J[:, j], fd, rtol=1e-6, atol=1e-6)
    assert np.all(J[:, 2] == 0)
    assert np.all(J[:, 3] == 0)


def test_d_one_minus_exp_over_branches_join():
    x = np.array([1e-3 * (1 - 1e-9), 1e-3 * (1 + 1e-9)])
    d = hm._d_one_minus_exp_over(x)
    assert abs(d[0] - d[1]) < 1e-12
    xs = np.array([1e-6, 1e-4, 0.5, 3.0])
    h = 1e-7
    fd = (hm._one_minus_exp_over(xs + h) - hm._one_minus_exp_over(xs - h)) / (2 * h)
    np.testing.assert_allclose(hm._d_one_minus_exp_over(xs), fd, rtol=1e-6, atol=1e-8)
