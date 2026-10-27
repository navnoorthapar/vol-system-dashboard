"""
test_heston_vvix.py — the model counterpart of Cboe's ^VVIX (C23).

heston_vvix(kappa, theta, sigma, v0) = 100 sqrt(-(2/T) E[ln(VIX_T / E VIX_T)])
at T = 30 days, from the closed-form CIR Laplace transform. Checked against
exact CIR sampling (scaled non-central chi-square, seeded) and its limits.
"""

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from joint_vol_calibration.models import heston as hm

_T = 30.0 / 365.0
_SETS = [
    (5.0, 0.05, 2.0, 0.034),          # 2026-10-16 production fit (rounded)
    (27.4, 0.0433, 6.9, 0.0364),      # 2026-10-16 with sigma <= 10
    (1.2, 0.075, 1.0, 0.033),
    (3.0, 0.04, 0.3, 0.04),
]


def _cir_sample(kappa, theta, sigma, v0, T, n, seed):
    c = 2 * kappa / (sigma**2 * -math.expm1(-kappa * T))
    d = 4 * kappa * theta / sigma**2
    lam = 2 * c * v0 * math.exp(-kappa * T)
    return stats.ncx2.rvs(d, lam, size=n, random_state=np.random.default_rng(seed)) / (2 * c)


@pytest.mark.parametrize("p", _SETS)
def test_expected_log_matches_exact_cir_sampling(p):
    kappa, theta, sigma, v0 = p
    b = float(hm._one_minus_exp_over(kappa * _T))
    a = theta * (1 - b)
    Y = a + b * _cir_sample(kappa, theta, sigma, v0, _T, 400_000, seed=11)
    ln = np.log(Y)
    exact = hm._expected_log_affine_cir(a, b, kappa, theta, sigma, v0, _T)[0]
    assert abs(exact - ln.mean()) < 4 * ln.std() / math.sqrt(len(ln))


@pytest.mark.parametrize("p", _SETS)
def test_vvix_matches_sampling(p):
    kappa, theta, sigma, v0 = p
    b = float(hm._one_minus_exp_over(kappa * _T))
    a = theta * (1 - b)
    Y = a + b * _cir_sample(kappa, theta, sigma, v0, _T, 400_000, seed=12)
    mc = 100 * math.sqrt((2 / _T) * (math.log(np.sqrt(Y).mean()) - 0.5 * np.log(Y).mean()))
    assert hm.heston_vvix(kappa, theta, sigma, v0) == pytest.approx(mc, rel=5e-3)


def test_vvix_limits_and_monotonicity():
    # sigma -> 0: VIX_T is deterministic, its log-contract vol vanishes
    assert hm.heston_vvix(3.0, 0.04, 1e-4, 0.04) < 0.05
    vals = [hm.heston_vvix(3.0, 0.04, s, 0.04) for s in (0.2, 0.5, 1.0, 1.5, 2.0)]
    assert all(x < y for x, y in itertools.pairwise(vals))
    # faster mean reversion averages variance over the 30-day window: lower VVIX
    assert hm.heston_vvix(27.0, 0.04, 2.0, 0.04) < hm.heston_vvix(5.0, 0.04, 2.0, 0.04)


def test_expected_log_jensen_and_degenerate_variance():
    kappa, theta, sigma, v0 = _SETS[0]
    b = float(hm._one_minus_exp_over(kappa * _T))
    a = theta * (1 - b)
    m = a + b * (theta + (v0 - theta) * math.exp(-kappa * _T))
    e_log = hm._expected_log_affine_cir(a, b, kappa, theta, sigma, v0, _T)[0]
    assert e_log < math.log(m)                                   # Jensen
    tiny = hm._expected_log_affine_cir(a, b, kappa, theta, 1e-6, v0, _T)[0]
    assert tiny == pytest.approx(math.log(m), abs=1e-9)          # no randomness
