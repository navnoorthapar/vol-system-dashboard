"""
test_implied_vol.py — torch-free Black-Scholes / Black-76 pricing, Greeks,
Fourier OTM pricing and the robust implied-vol solver (models/implied_vol.py).

Coverage:
  1. Exact round trips over a dense (x, T, σ) grid, calls and puts, from the
     OTM side and from the ITM side (parity), with zero failures
  2. The cases the old Newton loop got wrong (OTM calls returned as 1e-4)
  3. Ill-posed inputs → NaN with the right status; non-converged → NaN
  4. Greeks with dividend yield vs finite differences of the price
  5. fourier_otm_price vs Black-76 (lognormal CF) and vs adaptive quadrature
"""

import numpy as np
import pytest
from scipy import integrate

from joint_vol_calibration.models.implied_vol import (
    IV_ABOVE_MAX,
    IV_BAD_INPUT,
    IV_BELOW_FLOOR,
    IV_NO_TIME_VALUE,
    IV_NOT_CONVERGED,
    IV_OK,
    IV_UNRESOLVED,
    _bs_iv_vectorized,
    black_price,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vanna,
    bs_vega,
    bs_vomma,
    fourier_otm_price,
    implied_vol_black,
    implied_vol_bs,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _grid():
    """Dense (x, T, σ) grid: 81 strikes × 7 maturities × 9 vols."""
    x = np.linspace(-1.0, 1.0, 81)
    T = np.array([1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 3.0, 5.0])
    s = np.array([0.01, 0.03, 0.08, 0.15, 0.30, 0.60, 1.0, 2.0, 4.0])
    X, TT, SS = np.meshgrid(x, T, s, indexing="ij")
    return X.ravel(), TT.ravel(), SS.ravel()


def _otm(F, K, T, s):
    """Undiscounted OTM price (put below the forward, call at/above)."""
    return np.where(K >= F, black_price(F, K, T, s, True), black_price(F, K, T, s, False))


# ── 1. Round trips ────────────────────────────────────────────────────────────

class TestRoundTrip:

    def test_dense_grid_from_otm_prices_exact(self):
        """Every well-posed OTM price on the dense grid inverts to σ within 1e-10."""
        x, T, s = _grid()
        F = np.full(x.size, 100.0)
        K = F * np.exp(x)
        otm = _otm(F, K, T, s)
        well_posed = otm > 1e-12 * F
        iv, status = implied_vol_black(otm, F, K, T, is_call=(K >= F), full_output=True)
        assert well_posed.sum() > 3000
        assert np.all(status[well_posed] == IV_OK), "a well-posed price failed to invert"
        err = np.abs(iv[well_posed] - s[well_posed])
        assert err.max() < 1e-10, f"max |σ error| = {err.max():.2e}"

    @pytest.mark.parametrize("is_call", [True, False])
    def test_dense_grid_from_any_side(self, is_call):
        """Calls (or puts) at every strike: ITM inputs go through parity; the only
        error left is the parity cancellation, bounded by eps·intrinsic/vega."""
        x, T, s = _grid()
        F = np.full(x.size, 100.0)
        K = F * np.exp(x)
        price = black_price(F, K, T, s, is_call)
        otm = _otm(F, K, T, s)
        well_posed = otm > 1e-8 * F
        iv = implied_vol_black(price, F, K, T, is_call=is_call)
        assert not np.isnan(iv[well_posed]).any()
        assert np.abs(iv[well_posed] - s[well_posed]).max() < 1e-6

    def test_random_nn_domain_compat_wrapper(self):
        """200k exact prices over the old NN training domain through the
        backward-compatible _bs_iv_vectorized: no NaN, max error < 1e-9.
        (The old Newton loop got 8.1% of these wrong, 28.5% for x ∈ [0.10, 0.25).)"""
        rng = np.random.default_rng(1)
        n = 200_000
        x = rng.uniform(-0.30, 0.25, n)
        T = rng.uniform(0.02, 2.0, n)
        s = rng.uniform(0.08, 0.60, n)
        r = 0.03
        F = np.full(n, 100.0)
        K = F * np.exp(x)
        C = black_price(F, K, T, s, True, np.exp(-r * T))
        ok = _otm(F, K, T, s) > 1e-8 * F
        iv = _bs_iv_vectorized(C[ok], F[ok], K[ok], T[ok], np.full(ok.sum(), r))
        assert not np.isnan(iv).any()
        assert np.abs(iv - s[ok]).max() < 1e-9

    def test_bs_wrapper_with_dividends(self):
        S, r, q = 6581.0, 0.045, 0.013
        K = np.array([5000.0, 6000.0, 6581.0, 7200.0, 8000.0])
        T = np.array([0.1, 0.5, 1.0, 0.25, 2.0])
        sig = np.array([0.35, 0.25, 0.2, 0.15, 0.18])
        for is_call in (True, False):
            p = bs_price(S, K, T, r, q, sig, is_call)
            iv = implied_vol_bs(p, S, K, T, r, q, is_call)
            np.testing.assert_allclose(iv, sig, atol=1e-10)

    def test_converges_in_few_iterations(self):
        """The transformed Newton (−1/ln p below s*, ln of the complement above)
        converges on the whole dense grid within 15 iterations."""
        x, T, s = _grid()
        F = np.full(x.size, 100.0)
        K = F * np.exp(x)
        otm = _otm(F, K, T, s)
        ok = otm > 1e-12 * F
        iv = implied_vol_black(otm[ok], F[ok], K[ok], T[ok], K[ok] >= F[ok], max_iter=15)
        assert not np.isnan(iv).any()
        assert np.abs(iv - s[ok]).max() < 1e-10

    @pytest.mark.parametrize("x, T, sigma", [
        (0.12268, 0.0463, 0.2925), (-0.22336, 0.8017, 0.1125), (-0.12418, 0.0278, 0.5216),
        (-0.80, 5.0, 0.08), (-0.15, 7 / 365, 0.30), (-0.025, 0.25, 0.01),
    ])
    def test_no_false_convergence_far_from_root(self, x, T, sigma):
        """A bisection step can land so far into the wing that φ(d1) underflows
        and the Newton step evaluates to ~0; that must not be read as converged."""
        F = 100.0
        K = F * np.exp(x)
        p = black_price(F, K, T, sigma, K >= F)
        assert implied_vol_black(p, F, K, T, K >= F) == pytest.approx(sigma, abs=1e-10)

    def test_extreme_vol_round_trips(self):
        """600% (short-dated) and 0.5% vols are inverted, not capped."""
        assert implied_vol_black(black_price(100, 100, 1 / 365, 6.0), 100, 100, 1 / 365) == \
            pytest.approx(6.0, abs=1e-10)
        assert implied_vol_black(black_price(100, 100.5, 1.0, 0.005), 100, 100.5, 1.0) == \
            pytest.approx(0.005, abs=1e-10)


# ── 2. Regressions from the old solver ────────────────────────────────────────

class TestOldSolverFailures:

    @pytest.mark.parametrize("x, T, sigma", [
        (0.169, 0.743, 0.133),     # test_roundtrip_skew_range sample returned as 1e-4
        (0.117, 0.25, 0.1572),     # audit Heston example (reference IV)
        (0.091, 0.25, 0.1773),
        (0.20, 0.10, 0.25),
    ])
    def test_otm_call_recovered(self, x, T, sigma):
        """The Brenner-Subrahmanyam start clipped to 0.05 put these OTM calls
        in the flat region; Newton bounced between the clips and 1e-4 came back."""
        F, r = 6600.0, 0.045
        K = F * np.exp(x)
        C = black_price(F, K, T, sigma, True, np.exp(-r * T))
        iv = _bs_iv_vectorized(np.array([C]), np.array([F]), np.array([K]),
                               np.array([T]), np.array([r]))
        assert iv[0] == pytest.approx(sigma, abs=1e-10)

    def test_non_converged_is_nan_not_iterate(self):
        """With too few iterations the result must be NaN (status not_converged),
        never the last iterate dressed up as an implied vol."""
        x = np.array([-0.2, 0.05, 0.15, 0.22])
        F = np.full(4, 100.0)
        K = F * np.exp(x)
        T = np.full(4, 0.5)
        price = black_price(F, K, T, 0.2, True)
        iv, status = implied_vol_black(price, F, K, T, max_iter=1, full_output=True)
        assert np.all(np.isnan(iv))
        assert np.all(status == IV_NOT_CONVERGED)
        iv_compat = _bs_iv_vectorized(price, F, K, T, np.zeros(4), max_iter=1)
        assert np.all(np.isnan(iv_compat))


# ── 3. Ill-posed inputs ───────────────────────────────────────────────────────

class TestIllPosed:

    def test_status_codes(self):
        F, K, T = 100.0, 110.0, 0.5
        cases = [
            (0.0, F, K, T, True, IV_NO_TIME_VALUE),        # zero OTM price
            (10.0, F, 90.0, T, True, IV_NO_TIME_VALUE),    # ITM call exactly at intrinsic
            (-1.0, F, 90.0, T, True, IV_NO_TIME_VALUE),    # below intrinsic
            (100.0, F, K, T, True, IV_ABOVE_MAX),          # call worth the forward
            (1.0, F, K, 0.0, True, IV_BAD_INPUT),          # expired
            (np.nan, F, K, T, True, IV_BAD_INPUT),
            (1.0, F, -5.0, T, True, IV_BAD_INPUT),
        ]
        for price, F_, K_, T_, call, expected in cases:
            iv, status = implied_vol_black(price, F_, K_, T_, call, full_output=True)
            assert np.isnan(iv) and status == expected, (price, K_, T_, status)

    def test_unresolvable_prices_are_nan_not_a_guess(self):
        """When float64 rounding of the price does not determine σ, the solver
        must say so instead of returning whichever σ reproduces the rounded
        price. Two cases: a price within a few ulps of its upper bound (total
        vol σ√T ≳ 14; a 900%-vol 5-year put came back as 7.49 with status ok),
        and an ITM input whose time value (~1e-13·F) is lost in the parity
        subtraction. Everything that is still returned is accurate."""
        F = 100.0
        x = np.linspace(-2.0, 2.0, 41)
        T = np.array([0.25, 1.0, 5.0, 20.0])
        s = np.array([2.0, 5.0, 9.0])
        X, TT, SS = (a.ravel() for a in np.meshgrid(x, T, s, indexing="ij"))
        K = F * np.exp(X)
        for is_call in (True, False):
            iv, status = implied_vol_black(black_price(F, K, TT, SS, is_call), F, K, TT,
                                           is_call, full_output=True)
            ok = status == IV_OK
            assert np.all(np.abs(iv[ok] - SS[ok]) <= 1e-5 * SS[ok]), "a wrong σ came back as ok"
            assert np.all(status[~ok] != IV_NOT_CONVERGED)
            huge = SS * np.sqrt(TT) >= 20
            assert huge.any() and not ok[huge].any()
        # prices a few ulps below their upper bound (F for the call, K for the
        # put): any σ above ~15 reproduces them
        for K, call, bound in [(F * np.exp(0.5), True, F), (F * np.exp(-0.5), False, F * np.exp(-0.5))]:
            for k in (2, 8, 64):
                price = bound * (1.0 - k * np.finfo(float).eps)
                iv, status = implied_vol_black(price, F, K, 4.0, call, sigma_max=50.0, full_output=True)
                assert np.isnan(iv) and status == IV_UNRESOLVED, (K, k, iv, status)
        # time value below 1000 ulps of the ITM call price
        K, T, sig = F * np.exp(-0.3), 7 / 365, 0.30
        c = black_price(F, K, T, sig, True)
        assert black_price(F, K, T, sig, False) < 1e-12 * F
        iv, status = implied_vol_black(c, F, K, T, True, full_output=True)
        assert np.isnan(iv) and status == IV_UNRESOLVED
        # the same option quoted as its (OTM) put is perfectly resolvable
        iv_put = implied_vol_black(black_price(F, K, T, sig, False), F, K, T, False)
        assert iv_put == pytest.approx(sig, abs=1e-10)

    def test_tiny_prices_keep_their_digits(self):
        """Prices near 1e-319 (F = 5,000): black_price formed F·exp(ln(p/F)), and
        the solver ln(otm/F), each through a subnormal intermediate with a few
        significant bits (7.5% price error, 5e-5 relative σ error). Both now
        work with ln p − ln F. Reference: 60-digit mpmath."""
        mp = pytest.importorskip("mpmath")
        mp.mp.dps = 60
        F, T = 5000.0, 1 / 365
        for x, sig in [(0.4, 0.2), (-0.4, 0.2), (2.0, 1.0)]:
            K = F * np.exp(x)
            sq = mp.mpf(sig) * mp.sqrt(T)
            d1 = (mp.log(mp.mpf(F) / mp.mpf(K)) + sq**2 / 2) / sq
            d2 = d1 - sq
            ref = (F * mp.ncdf(d1) - K * mp.ncdf(d2) if x >= 0
                   else K * mp.ncdf(-d2) - F * mp.ncdf(-d1))
            p = black_price(F, K, T, sig, x >= 0)
            assert 0 < p < 1e-300
            assert float(abs(p - ref) / ref) < 1e-4          # subnormal: ~15 bits left
            for price in (p, float(ref)):
                assert implied_vol_black(price, F, K, T, x >= 0) == pytest.approx(sig, rel=1e-7)

    def test_price_floor(self):
        price = black_price(100.0, 130.0, 0.05, 0.2)       # ~1e-13
        iv, status = implied_vol_black(price, 100.0, 130.0, 0.05,
                                       min_otm_price=1e-6, full_output=True)
        assert np.isnan(iv) and status == IV_BELOW_FLOOR

    def test_compat_nan_cases(self):
        F = np.array([100.0, 100.0, 100.0])
        K = np.array([90.0, 100.0, 110.0])
        T = np.array([0.25, 0.25, 0.25])
        r = np.array([0.05, 0.05, 0.05])
        assert np.all(np.isnan(_bs_iv_vectorized(np.array([-1.0, -0.5, -1.0]), F, K, T, r)))
        assert np.isnan(_bs_iv_vectorized(np.array([1.0]), np.array([100.0]), np.array([100.0]),
                                          np.array([0.0]), np.array([0.0]))[0])


# ── 4. Greeks ─────────────────────────────────────────────────────────────────

class TestGreeks:
    S, r, q = 100.0, 0.05, 0.02
    K = np.array([70.0, 90.0, 100.0, 115.0, 140.0])
    T = np.array([0.1, 0.5, 1.0, 0.25, 2.0])
    sig = np.array([0.4, 0.25, 0.2, 0.3, 0.22])

    def _p(self, S=None, sig=None, call=True):
        return bs_price(self.S if S is None else S, self.K, self.T, self.r, self.q,
                        self.sig if sig is None else sig, call)

    @pytest.mark.parametrize("call", [True, False])
    def test_delta_gamma(self, call):
        h = 1e-3
        fd_delta = (self._p(S=self.S + h, call=call) - self._p(S=self.S - h, call=call)) / (2 * h)
        np.testing.assert_allclose(bs_delta(self.S, self.K, self.T, self.r, self.q, self.sig, call),
                                   fd_delta, rtol=1e-6, atol=1e-9)
        h = 1e-2
        fd_gamma = (self._p(S=self.S + h, call=call) - 2 * self._p(call=call)
                    + self._p(S=self.S - h, call=call)) / h**2
        np.testing.assert_allclose(bs_gamma(self.S, self.K, self.T, self.r, self.q, self.sig),
                                   fd_gamma, rtol=1e-4, atol=1e-8)

    def test_vega_vomma_vanna(self):
        h = 1e-4
        up, dn = self._p(sig=self.sig + h), self._p(sig=self.sig - h)
        vega = bs_vega(self.S, self.K, self.T, self.r, self.q, self.sig)
        np.testing.assert_allclose(vega, (up - dn) / (2 * h), rtol=1e-6)
        vomma = bs_vomma(self.S, self.K, self.T, self.r, self.q, self.sig)
        np.testing.assert_allclose(vomma, (up - 2 * self._p() + dn) / h**2, rtol=1e-3, atol=1e-4)
        d_up = bs_delta(self.S, self.K, self.T, self.r, self.q, self.sig + h)
        d_dn = bs_delta(self.S, self.K, self.T, self.r, self.q, self.sig - h)
        np.testing.assert_allclose(bs_vanna(self.S, self.K, self.T, self.r, self.q, self.sig),
                                   (d_up - d_dn) / (2 * h), rtol=1e-6, atol=1e-10)

    def test_put_call_parity(self):
        F, K, T, s, D = 100.0, np.array([80.0, 100.0, 120.0]), 0.5, 0.3, 0.98
        c = black_price(F, K, T, s, True, D)
        p = black_price(F, K, T, s, False, D)
        np.testing.assert_allclose(c - p, D * (F - K), atol=1e-12)

    def test_zero_vol_is_intrinsic(self):
        np.testing.assert_allclose(black_price(100.0, np.array([90.0, 110.0]), 1.0, 0.0, True),
                                   [10.0, 0.0])


# ── 5. Fourier OTM pricing ────────────────────────────────────────────────────

def _lewis_quad(cf, x):
    """Independent reference: Lewis formula by adaptive quadrature to infinity."""
    def integrand(u):
        return np.real(np.exp(-1j * u * x) * cf(np.array([u - 0.5j]))[0]) / (u * u + 0.25)
    val, _ = integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-15, epsrel=1e-13)
    c = 1.0 - np.exp(x / 2) / np.pi * val
    return c if x >= 0 else c - (1.0 - np.exp(x))


class TestFourierOtmPrice:

    @pytest.mark.parametrize("T, s", [(1 / 365, 0.2), (14 / 365, 0.01), (0.25, 0.2), (2.0, 0.5)])
    def test_lognormal_cf_matches_black(self, T, s):
        """With a lognormal CF the Fourier price is the Black price (the control
        variate is given a different variance so the integral is non-trivial)."""
        w = s * s * T
        def cf(u):
            return np.exp(-0.5 * w * (1j * u + u * u))
        x = np.linspace(-0.5, 0.5, 41)
        otm, ok = fourier_otm_price(cf, x, variance_hint=0.7 * w)
        assert ok
        ref = _otm(1.0, np.exp(x), T, s)
        assert np.abs(otm - ref).max() < 1e-13

    def test_heston_cf_matches_quadrature(self):
        from joint_vol_calibration.models.heston import characteristic_function
        p = dict(kappa=12.86, theta=0.0278, sigma=0.846, rho=-0.95, v0=1e-4)   # 2026-08-21 fit
        for T in (7 / 365, 0.25):
            def cf(u, T=T):
                return characteristic_function(u, 1.0, T, 0.0, 0.0, **p)
            x = np.array([-0.08, -0.02, 0.0, 0.01, 0.03])
            otm, ok = fourier_otm_price(cf, x)
            assert ok
            ref = np.array([_lewis_quad(cf, xi) for xi in x])
            assert np.abs(otm - ref).max() < 1e-12

    def test_non_decaying_cf_reports_failure(self):
        otm, ok = fourier_otm_price(lambda u: np.ones_like(u, dtype=complex), np.array([0.0]),
                                    variance_hint=0.01)
        assert not ok
