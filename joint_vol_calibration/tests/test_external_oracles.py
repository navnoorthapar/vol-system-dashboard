"""
test_external_oracles.py — the pricing core against independent open-source
libraries (C23). DEV-ONLY oracles, never runtime dependencies
(requirements-dev.txt):

  * QuantLib 1.44 (BSD-3): AnalyticHestonEngine (adaptive Gauss-Lobatto,
    relative tolerance 1e-12) and BatesEngine against heston_call_batch /
    bates_call_batch on every committed model_version 2 fit and the stress
    corners of tests/test_heston_reference.py, 1 day to 2 years. QuantLib's
    COSHestonEngine at its defaults (L = 16, N = 200) is recorded as the
    reason the Lewis contour is kept: at sigma ~ 2 its truncation range is
    too tight (Le Floc'h 2020, arXiv:2005.13248).
  * py_lets_be_rational 1.1.2 (MIT, Jaeckel's "Let's Be Rational"): implied
    volatility against models.implied_vol.implied_vol_black and the
    calibrator's data.option_chain.black76_implied_vol.

Every module skips cleanly without the oracle (pytest.importorskip). QuantLib
Python dates are whole days, so maturities are integer days / 365 on both
sides (Actual365Fixed), and the flat curves are continuously compounded.
"""

import numpy as np
import pytest

from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.models.heston import bates_call_batch, heston_call_batch
from joint_vol_calibration.models.implied_vol import black_price, implied_vol_black
from joint_vol_calibration.tests.test_heston_reference import STRESS_SETS, V2_FITS

_S, _R, _Q = 100.0, 0.0443, 0.0093
_DAYS = (1, 2, 7, 30, 91, 365, 730)
_K = _S * np.exp(np.linspace(-0.5, 0.4, 19))
_HEADLINE = (4.991868659859623, 0.04994358952070586, 1.9999999726422244,
             -0.7528229284527355, 0.03439100744902689)           # joint_cal_2026-10-16
_SETS = {**V2_FITS, "v2:2026-10-16": _HEADLINE, **STRESS_SETS}


@pytest.fixture(scope="module")
def ql():
    return pytest.importorskip("QuantLib")


def _curves(ql):
    today = ql.Date(16, 10, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    return (today, ql.YieldTermStructureHandle(ql.FlatForward(today, _R, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(today, _Q, dc)))


def _ql_calls(ql, engine, today, days, K):
    out = []
    for k in K:
        o = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, float(k)),
                             ql.EuropeanExercise(today + days))
        o.setPricingEngine(engine)
        try:
            out.append(o.NPV())
        except RuntimeError:          # QuantLib's adaptive engine gives up at a few
            out.append(np.nan)        # corner strikes ("max number of iterations")
    return np.array(out)


@pytest.mark.parametrize("name", list(_SETS))
def test_heston_engine_matches_quantlib_analytic(ql, name):
    kappa, theta, sigma, rho, v0 = _SETS[name]
    today, rts, qts = _curves(ql)
    proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(_S)), v0, kappa, theta,
                            sigma, rho)
    eng = ql.AnalyticHestonEngine(ql.HestonModel(proc), 1e-12, 1_000_000)
    n_compared = 0
    for days in _DAYS:
        ours = heston_call_batch(_S, _K, days / 365, _R, _Q, kappa, theta, sigma, rho, v0)
        theirs = _ql_calls(ql, eng, today, days, _K)
        ok = np.isfinite(theirs)
        n_compared += int(ok.sum())
        # our engine targets 1e-10 * DF * F; measured worst 5.6e-12 * S
        # (sigma = 2, rho = -0.95), <= 3e-13 * S on every v2 fit incl. 1 day
        assert np.max(np.abs(ours[ok] - theirs[ok])) < 1e-10 * _S, days
    assert n_compared >= 0.9 * len(_DAYS) * len(_K)


def test_quantlib_default_cos_is_less_accurate_than_our_engine_at_sigma_two(ql):
    """Why the Lewis contour stays: at the headline fit (sigma = 2) QuantLib's
    COS engine at its defaults misses the adaptive reference by 1.7e-6 * S
    (measured), ours by < 1e-12 * S."""
    kappa, theta, sigma, rho, v0 = _HEADLINE
    today, rts, qts = _curves(ql)
    proc = ql.HestonProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(_S)), v0, kappa, theta,
                            sigma, rho)
    model = ql.HestonModel(proc)
    ref_eng = ql.AnalyticHestonEngine(model, 1e-12, 1_000_000)
    cos_eng = ql.COSHestonEngine(model, 16, 200)
    worst_ours = worst_cos = 0.0
    for days in (30, 91, 365):
        ref = _ql_calls(ql, ref_eng, today, days, _K)
        ours = heston_call_batch(_S, _K, days / 365, _R, _Q, *_HEADLINE)
        cos = _ql_calls(ql, cos_eng, today, days, _K)
        worst_ours = max(worst_ours, float(np.nanmax(np.abs(ours - ref))))
        worst_cos = max(worst_cos, float(np.nanmax(np.abs(cos - ref))))
    assert worst_ours < 1e-3 * worst_cos


@pytest.mark.parametrize("p", [(2.0, 0.04, 0.5, -0.7, 0.04, 0.8, -0.08, 0.12),
                               (5.0, 0.05, 1.8, -0.75, 0.034, 1.5, -0.05, 0.08)])
def test_bates_engine_matches_quantlib(ql, p):
    """QuantLib's BatesModel rejects lam = 0, so the nested Heston case is
    covered by the Heston test above."""
    today, rts, qts = _curves(ql)
    proc = ql.BatesProcess(rts, qts, ql.QuoteHandle(ql.SimpleQuote(_S)), p[4], p[0], p[1],
                           p[2], p[3], p[5], p[6], p[7])
    eng = ql.BatesEngine(ql.BatesModel(proc), 1e-12, 1_000_000)
    for days in (7, 30, 365):
        ours = bates_call_batch(_S, _K, days / 365, _R, _Q, *p)
        theirs = _ql_calls(ql, eng, today, days, _K)
        assert np.nanmax(np.abs(ours - theirs)) < 1e-10 * _S, days       # measured 1.6e-12 * S


# ── implied volatility vs Let's Be Rational ──────────────────────────────────

@pytest.fixture(scope="module")
def lbr_set():
    lbr = pytest.importorskip("py_lets_be_rational")
    rng = np.random.default_rng(20261020)
    n = 20_000
    x = rng.uniform(-1.5, 0.7, n)                        # ln(K/F): deep put wing to far calls
    T = np.exp(rng.uniform(np.log(1 / 365), np.log(3.0), n))
    sig = np.exp(rng.uniform(np.log(0.05), np.log(1.5), n))
    F = np.full(n, 100.0)
    K = F * np.exp(x)
    is_call = x >= 0                                      # OTM side
    px = black_price(F, K, T, sig, is_call, 1.0)
    ref = np.array([lbr.implied_volatility_from_a_transformed_rational_guess(
        float(p), 100.0, float(k), float(t), 1.0 if c else -1.0)
        for p, k, t, c in zip(px, K, T, is_call)])
    # subnormal prices (< 1e-300: deep wings at 1-day maturities) carry no
    # information; LBR still returns a number there, wrong in some cases
    solved = np.isfinite(ref) & (ref > 0) & (px > 1e-300)
    return dict(x=x, T=T, sig=sig, F=F, K=K, is_call=is_call, px=px, ref=ref, solved=solved)


def test_implied_vol_black_matches_lets_be_rational(lbr_set):
    d = lbr_set
    ours = implied_vol_black(d["px"], d["F"], d["K"], d["T"], d["is_call"], 1.0)
    s = d["solved"]
    assert s.sum() > 15_000
    # solves every quote LBR solves, and agrees with it
    assert not np.any(s & ~np.isfinite(ours))
    assert np.max(np.abs(ours[s] - d["ref"][s]) / d["sig"][s]) < 1e-10      # measured 1.3e-12
    assert np.max(np.abs(ours[s] / d["sig"][s] - 1)) < 1e-10


def test_calibrator_iv_matches_lets_be_rational_on_its_domain(lbr_set):
    """data.option_chain.black76_implied_vol returns NaN below a time value of
    1e-10 * D * F by design (documented); on the calibration domain (K/F in
    [0.70, 1.30], 7 days to 2 years, time value >= 1e-4 * F) it must solve
    every quote and agree with LBR."""
    d = lbr_set
    b76 = oc.black76_implied_vol(d["px"], d["F"], d["K"], d["T"], 1.0, d["is_call"])
    s = d["solved"]
    dom = s & (np.abs(d["x"]) <= np.log(1.30)) & (d["T"] >= 7 / 365) & (d["T"] <= 2.0) \
        & (d["px"] >= 1e-4 * d["F"])
    assert dom.sum() > 2000
    assert not np.any(dom & ~np.isfinite(b76))
    assert np.max(np.abs(b76[dom] - d["ref"][dom]) / d["sig"][dom]) < 1e-10
    both = s & np.isfinite(b76)
    assert np.max(np.abs(b76[both] - d["ref"][both]) / d["sig"][both]) < 1e-10
    # every NaN it returns where LBR solved is a sub-floor time value
    miss = s & ~np.isfinite(b76)
    assert np.all(d["px"][miss] <= 1e-10 * d["F"][miss] * 1.0000001)
