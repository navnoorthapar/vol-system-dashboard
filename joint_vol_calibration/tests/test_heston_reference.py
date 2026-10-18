"""
test_heston_reference.py — Heston/Bates engine checked against INDEPENDENT references.

Nothing in this file calls the module under test to produce an expected value.
The references are written from scratch here:

  * _ref_log_cf      — Heston/Bates CF in Gatheral's (2006) 'r_minus' algebra
                       (not the cancellation-free form the engine uses), itself
                       validated against a numerical solution of the Riccati
                       ODEs (_riccati_cf, scipy solve_ivp DOP853).
  * ref_otm_prices   — Lewis (2001) formula integrated by composite 64-point
                       Gauss-Legendre on [0, U] with no control variate and no
                       trapezoid (the engine uses neither of those), validated
                       against Gil-Pelaez inversion with scipy.quad.
  * ref_iv           — Black-76 inversion with brentq at 1e-15 tolerance.
  * scipy.stats.ncx2 quadrature and numpy's noncentral_chisquare sampler for the
    CIR transition law behind the exact VIX futures.

The parameter sets include every archived production fit as of 2026-10-12
(data_store/calibrations/joint_cal_*.pkl, captured below so the test does not
depend on files that change daily), six of which have v0 at its 1e-4 floor,
plus Feller-violating and moment-explosion corners.
"""

import numpy as np
import pytest
from scipy import integrate, stats
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ndtr

from joint_vol_calibration.models.heston import (
    bates_call_batch,
    bates_characteristic_function,
    bates_vix_future_exact,
    bates_vix_term_index,
    characteristic_function,
    heston_call_batch,
    heston_call_price,
    heston_put_price,
    heston_vix_future_exact,
    heston_vix_futures_curve,
    heston_vix_futures_upper_bound,
    heston_vix_index,
    heston_vix_term_index,
    implied_vol_from_price,
)

# ── Captured parameter sets ──────────────────────────────────────────────────

# (kappa, theta, sigma, rho, v0) of every joint_cal_*.pkl in data_store/calibrations
# on 2026-10-12.
ARCHIVED_FITS = {
    "2026-03-24": (4.6213176193778605, 0.07641449297851698, 0.8406653896239384, -0.99, 0.056129661711285944),
    "2026-05-31": (1.7684851080781756, 0.06656289853226985, 0.46029248544890183, -0.9499999978720693, 0.012652491606997066),
    "2026-06-16": (1.7283832637417436, 0.06538647365052157, 0.4754218159987066, -0.9499996778269761, 0.01846314999583544),
    "2026-06-23": (2.1570005741459655, 0.060827998165064116, 0.45495637761552704, -0.9499944468616698, 0.02071981972792559),
    "2026-07-01": (2.2203680755045427, 0.05844536655646888, 0.4268033820116558, -0.9499998833080837, 0.015771728420617488),
    "2026-07-02": (2.2330654716484664, 0.05814189482884961, 0.32019674494817363, -0.95, 0.016163379904155218),
    "2026-07-03": (2.151937235934189, 0.05824786979136873, 0.25459285884403027, -0.9499994674655204, 0.0152523520637579),
    "2026-07-07": (2.2463364664087986, 0.05881490392109623, 0.5140394216443579, -0.949998803961174, 0.011741625170483135),
    "2026-07-09": (1.9446453933159082, 0.06081561823685746, 0.38863919137669517, -0.9499999113760701, 0.018872040073900927),
    "2026-07-10": (3.1549210377251558, 0.05226254365596095, 0.5742552937784485, -0.9499999767803233, 0.009807360643616567),
    "2026-07-13": (2.1265546213160214, 0.05909961197014847, 0.5013577137729412, -0.949999977976662, 0.009142401403691495),
    "2026-07-14": (8.940231634950962, 0.03804279646714794, 0.8247563903707226, -0.9499999784144678, 0.01189752049585781),
    "2026-07-15": (9.917663288498344, 0.03500840702881859, 0.8333088933193414, -0.9499997311595196, 0.007494827599476848),
    "2026-07-16": (7.989543125647846, 0.03346641447461783, 0.7312753288705778, -0.9499996534578831, 0.008025797567913603),
    "2026-07-17": (9.622517861698112, 0.03595525557776162, 0.8318416947765234, -0.9499998925602368, 0.009048408388284865),
    "2026-07-20": (4.1442353549369395, 0.051601846603463265, 0.6539881331387477, -0.7270084242425414, 0.022001906435301265),
    "2026-07-22": (5.772939915573865, 0.041578933243444666, 0.6928676600759979, -0.9499998482861566, 0.016010690722884708),
    "2026-07-23": (2.262714027074097, 0.05481690128840488, 0.47448246046212955, -0.9499999722864539, 0.01902298007682205),
    "2026-08-04": (4.022199139627185, 0.04630341824281701, 0.6103173509096278, -0.9499998803823346, 0.005322082502569006),
    "2026-08-07": (3.631611164146436, 0.0438425031113854, 0.5643102214875947, -0.9499994551650994, 0.0010366473165818724),
    "2026-08-19": (12.369477378296962, 0.025734325206678954, 0.797899535565246, -0.9499993995449834, 0.00010003505987579733),
    "2026-08-20": (11.327698974258267, 0.023958744662554834, 0.7367481381271861, -0.9499997011599789, 0.00010000831044072954),
    "2026-08-21": (12.858096400673555, 0.027831733337673158, 0.8460073116118316, -0.9499992189986955, 0.00010012989374197607),
    "2026-08-24": (11.920021035790336, 0.0252733457771945, 0.7762217090539604, -0.9499995531267332, 0.0001000004466646971),
    "2026-08-25": (12.889500556952184, 0.026932780424920574, 0.8332479215832653, -0.949997366565157, 0.0001000193388556845),
    "2026-08-27": (2.9266857200472445, 0.046151288280873365, 0.5197593673638121, -0.9499994006717394, 0.004923750856494424),
    "2026-08-28": (12.088755481523034, 0.021779829878589726, 0.7256618944670605, -0.9499995249680864, 0.00010000366730152263),
    "2026-08-29": (2.941696056454989, 0.04429810733227592, 0.5105224123635487, -0.9499999694741437, 0.0009897697638569828),
    "2026-09-01": (2.677595779741257, 0.045842477542628685, 0.49548422970951866, -0.9499997575430701, 0.0047383032230568405),
    "2026-09-02": (3.1984176143561593, 0.044649062412011435, 0.5344342284216073, -0.9499993899880071, 0.009071545838036378),
    "2026-09-03": (2.7890181528817903, 0.0462861025781017, 0.5081277980778387, -0.9499994614411824, 0.005441093756265632),
    "2026-09-04": (2.6746472139431137, 0.044479857532856056, 0.4877978119084037, -0.9499997748025135, 0.0034751960324730944),
    "2026-09-11": (4.999651331591191, 0.043135521895477605, 0.6567535677361634, -0.9499997077161061, 0.010880541693622658),
    "2026-09-25": (3.3207382183333225, 0.0461714518215673, 0.2950430551172502, -0.9499999467256413, 0.009575874370681524),
    "2026-09-28": (3.9053979473871343, 0.04412024693032082, 0.5870386543460683, -0.9499994804896467, 0.013842953503267154),
    "2026-10-02": (3.3002106817827523, 0.04712767370234951, 0.23602604989579373, -0.9499994481977553, 0.00917812254112923),
    "2026-10-07": (3.393597772855002, 0.04544317608061593, 0.555365348888141, -0.9499999837317422, 0.00848292458835137),
    "2026-10-09": (3.045683566327438, 0.047224295186613204, 0.536340311688529, -0.9499983810735141, 0.00841088583081257),
}

STRESS_SETS = {
    "base":           (2.0, 0.04, 0.3, -0.7, 0.04),
    "feller_violated": (0.5, 0.04, 1.0, -0.9, 0.04),
    "sigma2_rho095":  (1.0, 0.04, 2.0, -0.95, 0.04),
    "sigma2_rho0":    (0.5, 0.04, 2.0, 0.0, 0.04),    # negative-moment explosion
    "small_sigma":    (2.0, 0.04, 1e-3, -0.5, 0.04),
}

ALL_SETS = {**ARCHIVED_FITS, **STRESS_SETS}

_S, _R, _Q = 7600.0, 0.037, 0.013

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid); requirements allow 1.26
_trapz = getattr(np, "trapezoid", None) or np.trapz
_MATURITIES_D = [1, 3, 7, 14, 30, 91, 365]
_MONEYNESS = np.linspace(0.7, 1.3, 13)


# ── Independent reference implementation ────────────────────────────────────

def _ref_log_cf(u, T, kappa, theta, sigma, rho, v0, lam=0.0, mu_j=0.0, sigma_j=0.0):
    """log E[exp(iuX)], X = ln(S_T/F_T): Gatheral (2006) 'r_minus' algebra."""
    u = np.asarray(u, dtype=complex)
    alpha = -0.5 * u * u - 0.5j * u
    beta = kappa - 1j * rho * sigma * u
    d = np.sqrt(beta * beta - 2.0 * alpha * sigma * sigma)
    r_minus = (beta - d) / sigma**2
    r_plus = (beta + d) / sigma**2
    g = r_minus / r_plus
    e = np.exp(-d * T)
    D = r_minus * (1.0 - e) / (1.0 - g * e)
    C = kappa * theta * (r_minus * T - 2.0 / sigma**2 * np.log((1.0 - g * e) / (1.0 - g)))
    out = C + D * v0
    if lam:
        kbar = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
        out = out + lam * T * (np.exp(1j * u * mu_j - 0.5 * u * u * sigma_j**2)
                               - 1.0 - 1j * u * kbar)
    return out


def _riccati_cf(u, T, kappa, theta, sigma, rho, v0):
    """CF by integrating the Heston Riccati ODEs numerically (no closed form)."""
    u = complex(u)

    def rhs(t, y):
        D = y[0] + 1j * y[1]
        dD = 0.5 * (-u * u - 1j * u) + (1j * rho * sigma * u - kappa) * D + 0.5 * sigma**2 * D * D
        dC = kappa * theta * D
        return [dD.real, dD.imag, dC.real, dC.imag]

    sol = solve_ivp(rhs, (0.0, T), [0.0, 0.0, 0.0, 0.0], rtol=1e-12, atol=1e-14,
                    method="DOP853")
    D = sol.y[0, -1] + 1j * sol.y[1, -1]
    C = sol.y[2, -1] + 1j * sol.y[3, -1]
    return np.exp(C + D * v0)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def ref_otm_prices(S, strikes, T, r, q, params, jumps=(0.0, 0.0, 0.0)):
    """
    Out-of-the-money option prices (call if K >= F else put) by the Lewis
    formula, integrated with composite 64-point Gauss-Legendre up to the point
    where |CF|/u < 1e-18. Returns (prices, rights).
    """
    kappa, theta, sigma, rho, v0 = params
    strikes = np.asarray(strikes, dtype=float)
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    x = np.log(F / strikes)
    w = (theta * T + (v0 - theta) * (-np.expm1(-kappa * T)) / kappa
         + jumps[0] * T * (jumps[1]**2 + jumps[2]**2))

    def cf(u):
        return np.exp(_ref_log_cf(u - 0.5j, T, kappa, theta, sigma, rho, v0, *jumps))

    scan = np.geomspace(1.0, 1e9, 600)
    tail = np.maximum.accumulate((np.abs(cf(scan)) / scan)[::-1])[::-1]
    U = scan[np.argmax(tail < 1e-18)] if (tail < 1e-18).any() else scan[-1]
    width_osc = 8.0 / (np.max(np.abs(x)) + 0.5 * w + 1e-3)
    edges = [0.0]
    while edges[-1] < U:
        edges.append(edges[-1] + min(width_osc, max(0.5, 0.25 * edges[-1])))
    a, b = np.array(edges[:-1])[:, None], np.array(edges[1:])[:, None]
    nodes = (0.5 * (b - a) * _GL_X + 0.5 * (a + b)).ravel()
    wts = (0.5 * (b - a) * _GL_W).ravel()
    vals = cf(nodes) / (nodes**2 + 0.25) * wts
    integral = np.array([(np.exp(1j * nodes * xj) * vals).real.sum() for xj in x])
    is_call = strikes >= F
    base = np.where(is_call, F, strikes)
    prices = df * (base - np.sqrt(F * strikes) / np.pi * integral)
    return prices, np.where(is_call, "C", "P")


def _black76(F, K, T, df, vol, right):
    sd = vol * np.sqrt(T)
    d1 = np.log(F / K) / sd + 0.5 * sd
    if right == "C":
        return df * (F * ndtr(d1) - K * ndtr(d1 - sd))
    return df * (K * ndtr(sd - d1) - F * ndtr(-d1))


def ref_iv(price, S, K, T, r, q, right):
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    try:
        return brentq(lambda v: _black76(F, K, T, df, v, right) - price, 1e-7, 20.0,
                      xtol=1e-15, rtol=1e-15, maxiter=500)
    except ValueError:
        return np.nan


def _otm_from_call(call, S, K, T, r, q, right):
    return call if right == "C" else call - S * np.exp(-q * T) + K * np.exp(-r * T)


# ── The reference itself is right ────────────────────────────────────────────

class TestReferenceIsSound:

    @pytest.mark.parametrize("name", ["base", "2026-08-19", "sigma2_rho095", "sigma2_rho0"])
    @pytest.mark.parametrize("T", [1 / 365, 30 / 365, 1.0, 3.0])
    def test_reference_cf_matches_riccati_ode(self, name, T):
        p = ALL_SETS[name]
        for u in [0.3 - 0.5j, 5.0 - 0.5j, 40.0 - 0.5j, 2.0 - 2.5j, 300.0 - 0.5j, 1.5 + 0.0j]:
            ref = np.exp(_ref_log_cf(u, T, *p))
            ode = _riccati_cf(u, T, *p)
            assert abs(ref - ode) <= 1e-10 * max(abs(ode), 1e-300)

    def test_reference_lewis_matches_gil_pelaez_quad(self):
        """Lewis/GL reference == Gil-Pelaez probabilities integrated by scipy.quad.

        (Gil-Pelaez in double precision with quad is good to ~5e-8 here; a
        30-digit mpmath evaluation of these three prices agreed with the GL
        reference to 1.3e-12 when this test was written.)"""
        p, T = STRESS_SETS["base"], 91 / 365
        F = _S * np.exp((_R - _Q) * T)
        strikes = np.array([0.85, 1.0, 1.15]) * F
        prices, rights = ref_otm_prices(_S, strikes, T, _R, _Q, p)
        for K, px, right in zip(strikes, prices, rights):
            k = np.log(K / F)
            phi = lambda u, sh: np.exp(_ref_log_cf(u - sh, T, *p))
            p2 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * k) * phi(u, 0j) / (1j * u)).real,
                                      0, np.inf, limit=2000, epsabs=1e-14)[0] / np.pi
            p1 = 0.5 + integrate.quad(lambda u: (np.exp(-1j * u * k) * phi(u, 1j) / (1j * u)).real,
                                      0, np.inf, limit=2000, epsabs=1e-14)[0] / np.pi
            call = np.exp(-_R * T) * (F * p1 - K * p2)
            gp = _otm_from_call(call, _S, K, T, _R, _Q, right)
            assert abs(gp - px) < 1e-7, (K / F, gp, px)


# ── Pricing engine vs reference ──────────────────────────────────────────────

class TestEngineCharacteristicFunction:

    @pytest.mark.parametrize("name", list(STRESS_SETS) + ["2026-03-24", "2026-08-19"])
    def test_cf_matches_reference(self, name):
        p = ALL_SETS[name]
        u = np.array([0.0, 0.7, 3.0, 25.0, 400.0]) - 0.5j
        for T in [1 / 365, 0.25, 2.0, 10.0]:
            F = _S * np.exp((_R - _Q) * T)
            eng = characteristic_function(u, _S, T, _R, _Q, *p)
            ref = np.exp(_ref_log_cf(u, T, *p) + 1j * u * np.log(F))
            # the reference's (beta - d)/sigma^2 form loses ~1e-11 at sigma=1e-3
            np.testing.assert_allclose(eng, ref, rtol=1e-9, atol=1e-300)

    @pytest.mark.parametrize("sigma", [1e-7, 1e-9])
    def test_cf_small_sigma_converges_to_black_scholes(self, sigma):
        """sigma -> 0 (rho = 0, so the model error is O(sigma^2)): deterministic
        variance, CF -> exp(-w(u^2 + iu)/2) in log-forward coordinates. The
        textbook (beta - d)/sigma^2 form is off by 0.4% at sigma=1e-7 and by
        500% at 1e-9 through cancellation."""
        kappa, theta, v0, T = 2.0, 0.04, 0.09, 0.5
        w = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
        u = np.array([0.5, 3.0, 10.0])
        F = _S * np.exp((_R - _Q) * T)
        eng = characteristic_function(u, _S, T, _R, _Q, kappa, theta, sigma, 0.0, v0)
        bs = np.exp(-0.5 * w * (u * u + 1j * u) + 1j * u * np.log(F))
        np.testing.assert_allclose(eng, bs, rtol=1e-9)


class TestBatchPricerAccuracy:
    """
    heston_call_batch must reproduce the reference to < 0.01 implied-vol points
    for 1d..1y and K/F in [0.7, 1.3]. The pre-C22 Carr-Madan grid (512 points,
    v <= 300) failed this at short maturities (0.8 vp at 7d, 3.4 vp at 1d on
    archived fits) and in the moment-explosion corner (sigma2_rho0).
    """

    @pytest.mark.parametrize("name", list(ALL_SETS))
    def test_iv_error_below_one_hundredth_vol_point(self, name):
        p = ALL_SETS[name]
        worst = (0.0, None)
        n_checked = 0
        for Td in _MATURITIES_D:
            T = Td / 365.0
            F = _S * np.exp((_R - _Q) * T)
            strikes = _MONEYNESS * F
            calls = heston_call_batch(_S, strikes, T, _R, _Q, *p)
            refs, rights = ref_otm_prices(_S, strikes, T, _R, _Q, p)
            for K, c, ref, right in zip(strikes, calls, refs, rights):
                model = _otm_from_call(c, _S, K, T, _R, _Q, right)
                # absolute price accuracy everywhere, including where the
                # option is worth (numerically) nothing
                assert abs(model - ref) < 1e-6 * F, (Td, K / F, model, ref)
                if ref < 1e-5 * F:          # IV of a ~zero price is not informative
                    continue
                err = abs(ref_iv(model, _S, K, T, _R, _Q, right)
                          - ref_iv(ref, _S, K, T, _R, _Q, right)) * 100.0
                n_checked += 1
                err = np.inf if np.isnan(err) else err
                if err > worst[0]:
                    worst = (err, (Td, round(K / F, 3), right, ref, model))
        assert n_checked >= 20
        assert worst[0] < 0.01, f"{name}: max IV error {worst[0]:.4f} vp at {worst[1]}"

    def test_moment_explosion_regime_three_years(self):
        """(kappa=0.5, sigma=2, rho=0): E[S^2.5] explodes before 3y, which broke
        Carr-Madan alpha=1.5 (old 0.8-strike call off by $93). Lewis contour is fine."""
        p, T = STRESS_SETS["sigma2_rho0"], 3.0
        F = _S * np.exp((_R - _Q) * T)
        strikes = np.array([0.6, 0.8, 1.0, 1.2, 1.5]) * F
        calls = heston_call_batch(_S, strikes, T, _R, _Q, *p)
        refs, rights = ref_otm_prices(_S, strikes, T, _R, _Q, p)
        model = [_otm_from_call(c, _S, K, T, _R, _Q, rt) for c, K, rt in zip(calls, strikes, rights)]
        np.testing.assert_allclose(model, refs, atol=0.01)

    @pytest.mark.parametrize("name", ["base", "2026-08-19", "sigma2_rho095"])
    def test_single_pricer_is_the_batch_engine(self, name):
        p = ALL_SETS[name]
        for T in [2 / 365, 0.25, 1.5]:
            F = _S * np.exp((_R - _Q) * T)
            strikes = np.array([0.8, 0.97, 1.0, 1.04, 1.25]) * F
            batch = heston_call_batch(_S, strikes, T, _R, _Q, *p)
            single = [heston_call_price(_S, K, T, _R, _Q, *p) for K in strikes]
            np.testing.assert_allclose(single, batch, rtol=0, atol=1e-8)

    @pytest.mark.parametrize("name", ["base", "2026-05-31", "feller_violated"])
    def test_put_matches_independent_lewis_put(self, name):
        """Puts (via parity in the module) vs puts computed directly by the
        Lewis PUT formula — parity is checked against an independent number."""
        p = ALL_SETS[name]
        T = 0.25
        F = _S * np.exp((_R - _Q) * T)
        for K in np.array([0.75, 0.9, 0.98]) * F:
            refs, rights = ref_otm_prices(_S, [K], T, _R, _Q, p)
            assert rights[0] == "P"
            put = heston_put_price(_S, K, T, _R, _Q, *p)
            assert abs(put - refs[0]) < 1e-6, (K / F, put, refs[0])


class TestBatesAgainstReference:

    @pytest.mark.parametrize("jumps", [(2.0, -0.05, 0.07), (0.5, -0.15, 0.10), (8.0, -0.10, 0.12)])
    @pytest.mark.parametrize("T", [3 / 365, 30 / 365, 0.5, 2.0])
    def test_bates_batch_matches_reference(self, jumps, T):
        p = STRESS_SETS["base"]
        F = _S * np.exp((_R - _Q) * T)
        strikes = np.array([0.7, 0.85, 0.95, 1.0, 1.05, 1.2, 1.3]) * F
        calls = bates_call_batch(_S, strikes, T, _R, _Q, *p, *jumps)
        refs, rights = ref_otm_prices(_S, strikes, T, _R, _Q, p, jumps=jumps)
        model = np.array([_otm_from_call(c, _S, K, T, _R, _Q, rt)
                          for c, K, rt in zip(calls, strikes, rights)])
        np.testing.assert_allclose(model, refs, rtol=0, atol=1e-6 * F)

    def test_bates_cf_is_a_martingale(self):
        """E[S_T] = F: CF at phi = -i must equal the forward."""
        for T in [0.01, 0.5, 3.0]:
            F = _S * np.exp((_R - _Q) * T)
            cf = bates_characteristic_function(-1j, _S, T, _R, _Q, *STRESS_SETS["base"],
                                               lam=3.0, mu_j=-0.08, sigma_j=0.1)
            assert abs(cf / F - 1.0) < 1e-13

    def test_t_zero_intrinsic_for_itm_strike(self):
        strikes = np.array([7000.0, 7600.0, 8000.0])
        prices = bates_call_batch(_S, strikes, 0.0, _R, _Q, *STRESS_SETS["base"],
                                  lam=2.0, mu_j=-0.05, sigma_j=0.07)
        np.testing.assert_allclose(prices, [600.0, 0.0, 0.0], atol=1e-12)


# ── VIX: spot term-structure index ───────────────────────────────────────────

def _log_contract_variance(log_cf_at, tau, h=1e-5):
    """-(2/tau) E[ln(S_tau/F)], E[X] = Im(log phi(h))/h from a log-forward CF."""
    return -2.0 / tau * np.imag(np.log(log_cf_at(h))) / h


class TestVixTermIndex:

    @pytest.mark.parametrize("name", ["base", "2026-05-31", "2026-08-19", "feller_violated"])
    @pytest.mark.parametrize("days", [9, 30, 93, 184])
    def test_index_equals_log_contract_variance_of_the_cf(self, name, days):
        """Cboe's VIX is the log-contract variance; compute it from the CF
        (independent of the closed form) and compare."""
        p = ALL_SETS[name]
        tau = days / 365.0
        F = _S * np.exp((_R - _Q) * tau)
        lc = _log_contract_variance(
            lambda h: characteristic_function(h, _S, tau, _R, _Q, *p) / np.exp(1j * h * np.log(F)),
            tau)
        kappa, theta, sigma, rho, v0 = p
        assert abs(heston_vix_term_index(kappa, theta, sigma, v0, tau) - 100 * np.sqrt(lc)) < 1e-5

    def test_index_from_otm_option_strip(self):
        """VIX^2 = (2 e^{r tau}/tau) * int OTM(K)/K^2 dK (Cboe model-free
        formula), with OTM prices from the reference pricer."""
        p = ARCHIVED_FITS["2026-05-31"]
        tau = 30 / 365
        F = _S * np.exp((_R - _Q) * tau)
        kgrid = np.linspace(np.log(0.2), np.log(2.5), 1501)
        strikes = F * np.exp(kgrid)
        otm, _ = ref_otm_prices(_S, strikes, tau, _R, _Q, p)
        var = 2.0 * np.exp(_R * tau) / tau * _trapz(otm / strikes, kgrid)
        kappa, theta, sigma, rho, v0 = p
        assert abs(heston_vix_term_index(kappa, theta, sigma, v0, tau) - 100 * np.sqrt(var)) < 2e-3

    def test_limits_and_vectorisation(self):
        kappa, theta, sigma, v0 = 3.0, 0.0625, 0.5, 0.04
        T = np.array([1e-12, 9 / 365, 30 / 365, 93 / 365, 184 / 365])
        out = heston_vix_term_index(kappa, theta, sigma, v0, T)
        assert out.shape == T.shape
        assert abs(out[0] - 20.0) < 1e-9                                   # T -> 0: sqrt(v0)
        assert np.all(np.diff(out) > 0)                                    # v0 < theta: upward
        assert abs(heston_vix_term_index(1e-12, theta, sigma, v0, 0.5) - 20.0) < 1e-9   # kappa -> 0
        assert abs(heston_vix_term_index(1e4, theta, sigma, v0, 0.5) - 25.0) < 1e-2     # kappa -> inf
        assert heston_vix_index(kappa, theta, v0) == pytest.approx(
            heston_vix_term_index(kappa, theta, sigma, v0, 30 / 365), abs=1e-12)

    def test_index_is_not_the_futures_price(self):
        """^VIX (spot, window [0,30d]) and the 1M future (window [30d,60d]) are
        different instruments whenever v0 != theta."""
        kappa, theta, sigma, rho, v0 = ARCHIVED_FITS["2026-05-31"]
        idx = heston_vix_term_index(kappa, theta, sigma, v0, 30 / 365)
        fut = heston_vix_futures_curve(kappa, theta, sigma, v0, np.array([30 / 365]))[0]
        assert abs(idx - fut) > 1.0


class TestBatesVixJumpTerm:

    @pytest.mark.parametrize("jumps", [(2.0, -0.05, 0.07), (8.0, -0.10, 0.12), (0.5, 0.03, 0.02)])
    def test_index_equals_log_contract_variance_of_bates_cf(self, jumps):
        kappa, theta, sigma, rho, v0 = STRESS_SETS["base"]
        tau = 30 / 365
        F = _S * np.exp((_R - _Q) * tau)
        lc = _log_contract_variance(
            lambda h: bates_characteristic_function(h, _S, tau, _R, _Q, kappa, theta, sigma,
                                                    rho, v0, *jumps) / np.exp(1j * h * np.log(F)),
            tau)
        got = bates_vix_term_index(kappa, theta, sigma, v0, *jumps, tau)
        assert abs(got - 100 * np.sqrt(lc)) < 1e-5
        # and the jump term is material: dropping it (pure Heston) is wrong
        assert got - heston_vix_term_index(kappa, theta, sigma, v0, tau) > 0.1

    def test_lam_zero_is_heston(self):
        kappa, theta, sigma, rho, v0 = ARCHIVED_FITS["2026-03-24"]
        T = np.array([9, 30, 93, 184]) / 365
        np.testing.assert_allclose(bates_vix_term_index(kappa, theta, sigma, v0, 0.0, -0.1, 0.1, T),
                                   heston_vix_term_index(kappa, theta, sigma, v0, T), rtol=1e-15)
        np.testing.assert_allclose(bates_vix_future_exact(kappa, theta, sigma, v0, 0.0, -0.1, 0.1, T),
                                   heston_vix_future_exact(kappa, theta, sigma, v0, T), rtol=1e-13)


# ── VIX futures: exact E[VIX_T] ──────────────────────────────────────────────

def _ncx2_expected_vix(kappa, theta, sigma, v0, T, tau=30 / 365, jump_var=0.0):
    """E[100 sqrt(a + b v_T)] by quadrature against the scipy.stats.ncx2 pdf."""
    b = (1 - np.exp(-kappa * tau)) / (kappa * tau)
    a = theta * (1 - b) + jump_var
    c = sigma**2 * (1 - np.exp(-kappa * T)) / (4 * kappa)
    dist = stats.ncx2(4 * kappa * theta / sigma**2, v0 * np.exp(-kappa * T) / c, scale=c)
    val = integrate.quad(lambda v: np.sqrt(a + b * v) * dist.pdf(v), 0, dist.ppf(1 - 1e-15),
                         limit=1000, epsabs=1e-14, epsrel=1e-12, points=[dist.mean()])[0]
    return 100.0 * val


class TestVixFuturesExact:

    _SETS = ["2026-05-31", "2026-03-24", "2026-08-19", "base", "feller_violated"]

    @pytest.mark.parametrize("name", _SETS)
    def test_matches_ncx2_quadrature(self, name):
        kappa, theta, sigma, rho, v0 = ALL_SETS[name]
        T = np.array([7, 30, 91, 182, 365]) / 365
        got = heston_vix_future_exact(kappa, theta, sigma, v0, T)
        ref = [_ncx2_expected_vix(kappa, theta, sigma, v0, t) for t in T]
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-8)

    @pytest.mark.parametrize("name", ["2026-05-31", "2026-03-24", "2026-08-19"])
    def test_matches_exact_cir_sampling(self, name):
        kappa, theta, sigma, rho, v0 = ALL_SETS[name]
        rng = np.random.default_rng(20261012)
        tau = 30 / 365
        b = (1 - np.exp(-kappa * tau)) / (kappa * tau)
        for T in [30 / 365, 182 / 365]:
            c = sigma**2 * (1 - np.exp(-kappa * T)) / (4 * kappa)
            vT = c * rng.noncentral_chisquare(4 * kappa * theta / sigma**2,
                                              v0 * np.exp(-kappa * T) / c, size=400_000)
            vix = 100 * np.sqrt(theta * (1 - b) + b * vT)
            se = vix.std(ddof=1) / np.sqrt(vix.size)
            got = heston_vix_future_exact(kappa, theta, sigma, v0, T)
            assert abs(got - vix.mean()) < 4 * se, (T, got, vix.mean(), se)

    @pytest.mark.parametrize("name", _SETS)
    def test_jensen_strictly_below_upper_bound(self, name):
        kappa, theta, sigma, rho, v0 = ALL_SETS[name]
        T = np.array([30, 91, 182]) / 365
        exact = heston_vix_futures_curve(kappa, theta, sigma, v0, T)
        upper = heston_vix_futures_upper_bound(kappa, theta, sigma, v0, T)
        assert np.all(exact < upper - 0.05)

    def test_sigma_to_zero_and_t_to_zero_limits(self):
        kappa, theta, v0 = 2.0, 0.04, 0.09
        T = np.array([0.1, 0.5, 1.0])
        np.testing.assert_allclose(heston_vix_future_exact(kappa, theta, 1e-9, v0, T),
                                   heston_vix_futures_upper_bound(kappa, theta, 1e-9, v0, T),
                                   rtol=1e-12)
        assert heston_vix_future_exact(kappa, theta, 0.8, v0, 1e-12) == pytest.approx(
            heston_vix_term_index(kappa, theta, 0.8, v0, 30 / 365), abs=1e-9)

    def test_flat_expected_variance_still_gives_downward_futures_curve(self):
        """v0 = theta makes E[VIX_T^2] flat, but Var[v_T] grows with T, so the
        exact futures curve slopes DOWN (convexity). The old sqrt(E) curve was
        exactly flat and could not show this."""
        T = np.array([1, 2, 3, 6, 12]) / 12
        fut = heston_vix_futures_curve(2.0, 0.04, 0.5, 0.04, T)
        assert np.all(np.diff(fut) < 0)
        assert fut[0] - fut[-1] > 0.3

    def test_bates_futures_include_jump_variance(self):
        kappa, theta, sigma, rho, v0 = STRESS_SETS["base"]
        lam, mu_j, sigma_j = 2.0, -0.05, 0.07
        jump_var = 2 * lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1 - mu_j)
        T = np.array([30, 182]) / 365
        got = bates_vix_future_exact(kappa, theta, sigma, v0, lam, mu_j, sigma_j, T)
        ref = [_ncx2_expected_vix(kappa, theta, sigma, v0, t, jump_var=jump_var) for t in T]
        np.testing.assert_allclose(got, ref, rtol=0, atol=1e-8)


# ── Implied-vol inversion against the reference inverter ────────────────────

class TestImpliedVolAgainstReference:

    @pytest.mark.parametrize("T", [1 / 365, 7 / 365, 0.25, 2.0])
    def test_calls_and_puts_round_trip(self, T):
        F = _S * np.exp((_R - _Q) * T)
        df = np.exp(-_R * T)
        for m in [0.7, 0.9, 1.0, 1.1, 1.3]:
            K = m * F
            for vol in [0.05, 0.2, 0.9, 6.0]:
                for right in "CP":
                    px = _black76(F, K, T, df, vol, right)
                    otm = _black76(F, K, T, df, vol, "C" if K >= F else "P")
                    got = implied_vol_from_price(px, _S, K, T, _R, _Q, right)
                    if otm <= 1e-12 * df * F:
                        assert np.isnan(got)          # zero time value: ill-posed
                    else:
                        assert abs(got - vol) < 1e-6, (T, m, vol, right, got)
