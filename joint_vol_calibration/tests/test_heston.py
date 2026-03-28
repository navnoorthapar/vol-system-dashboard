"""
test_heston.py — Unit tests for the Heston model implementation.

Tests are designed to verify financial correctness, not just code correctness:
  1. Put-call parity (no-arbitrage identity)
  2. Known limiting cases (T→0, K→0, K→∞)
  3. Monotonicity of option prices (calls ↓ as K↑, puts ↑ as K↑)
  4. Greeks signs (delta ∈ (0,1) for calls, gamma > 0, vega > 0)
  5. VIX formula: at kappa→∞, VIX² → theta (long-run mean)
  6. VIX gap function returns correct shape

Run with:
  pytest tests/test_heston.py -v
"""

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.models.heston import (
    heston_call_price,
    heston_put_price,
    heston_greeks,
    heston_vix_index,
    heston_vix_futures_curve,
    heston_integrated_variance,
    compute_heston_vix_gap,
    implied_vol_from_price,
    black_scholes_call,
    HestonModel,
    heston_call_batch,
    bates_characteristic_function,
    bates_call_price,
    bates_call_batch,
    characteristic_function,
)

# Default parameters for tests (stable, near-ATM)
_PARAMS = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
_S = 4500.0
_r = 0.045
_q = 0.013
_K_atm = 4500.0
_T = 0.25   # 3 months


# ── Put-Call Parity ───────────────────────────────────────────────────────────

class TestPutCallParity:
    """C - P = S*e^{-qT} - K*e^{-rT} (no-arbitrage identity)."""

    def test_atm_put_call_parity(self):
        p = _PARAMS
        call = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        put  = heston_put_price(_S, _K_atm, _T, _r, _q, **p)
        lhs = call - put
        rhs = _S * np.exp(-_q * _T) - _K_atm * np.exp(-_r * _T)
        assert abs(lhs - rhs) < 0.10, (
            f"Put-call parity violated: C-P={lhs:.4f}, F-K*df={rhs:.4f}, diff={abs(lhs-rhs):.4f}"
        )

    @pytest.mark.parametrize("K", [4000, 4200, 4500, 4800, 5000])
    def test_parity_across_strikes(self, K):
        p = _PARAMS
        call = heston_call_price(_S, K, _T, _r, _q, **p)
        put  = heston_put_price(_S, K, _T, _r, _q, **p)
        lhs = call - put
        rhs = _S * np.exp(-_q * _T) - K * np.exp(-_r * _T)
        assert abs(lhs - rhs) < 0.50, (
            f"K={K}: put-call parity error = {abs(lhs-rhs):.4f}"
        )


# ── Option Price Bounds ───────────────────────────────────────────────────────

class TestOptionBounds:
    """Option prices must respect no-arbitrage bounds."""

    def test_call_lower_bound(self):
        p = _PARAMS
        call = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        lb = max(_S * np.exp(-_q * _T) - _K_atm * np.exp(-_r * _T), 0.0)
        assert call >= lb - 0.01, f"Call={call:.4f} below lower bound {lb:.4f}"

    def test_call_upper_bound(self):
        p = _PARAMS
        call = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        ub = _S * np.exp(-_q * _T)
        assert call <= ub + 0.01, f"Call={call:.4f} above upper bound {ub:.4f}"

    def test_call_positive(self):
        p = _PARAMS
        for K in [4000, 4250, 4500, 4750, 5000]:
            call = heston_call_price(_S, K, _T, _r, _q, **p)
            assert call >= 0.0, f"Negative call price at K={K}: {call:.4f}"

    def test_put_positive(self):
        p = _PARAMS
        for K in [4000, 4250, 4500, 4750, 5000]:
            put = heston_put_price(_S, K, _T, _r, _q, **p)
            assert put >= 0.0, f"Negative put price at K={K}: {put:.4f}"


# ── Monotonicity ──────────────────────────────────────────────────────────────

class TestMonotonicity:
    """Call prices decrease in K; put prices increase in K."""

    def test_call_decreasing_in_strike(self):
        p = _PARAMS
        strikes = [4000, 4200, 4400, 4600, 4800]
        calls = [heston_call_price(_S, K, _T, _r, _q, **p) for K in strikes]
        for i in range(len(calls) - 1):
            assert calls[i] >= calls[i + 1] - 0.01, (
                f"Call not monotone: C(K={strikes[i]})={calls[i]:.2f} "
                f"< C(K={strikes[i+1]})={calls[i+1]:.2f}"
            )

    def test_put_increasing_in_strike(self):
        p = _PARAMS
        strikes = [4000, 4200, 4400, 4600, 4800]
        puts = [heston_put_price(_S, K, _T, _r, _q, **p) for K in strikes]
        for i in range(len(puts) - 1):
            assert puts[i] <= puts[i + 1] + 0.01, (
                f"Put not monotone: P(K={strikes[i]})={puts[i]:.2f} "
                f"> P(K={strikes[i+1]})={puts[i+1]:.2f}"
            )

    def test_call_decreasing_in_maturity_deep_otm(self):
        """
        Deep OTM calls should be more expensive at longer maturities.
        This tests that time value is positive.
        """
        p = _PARAMS
        K_otm = 5500  # 22% OTM
        t1, t2 = 0.1, 0.5
        c1 = heston_call_price(_S, K_otm, t1, _r, _q, **p)
        c2 = heston_call_price(_S, K_otm, t2, _r, _q, **p)
        assert c2 >= c1 - 0.01, (
            f"OTM call at T={t2:.1f} ({c2:.4f}) < T={t1:.1f} ({c1:.4f})"
        )


# ── Implied Volatility Round-Trip ─────────────────────────────────────────────

class TestImpliedVolRoundTrip:
    """Price → IV → Price should be identical."""

    @pytest.mark.parametrize("K,sig", [
        (4300, 0.18), (4500, 0.20), (4700, 0.22), (4000, 0.25)
    ])
    def test_bs_roundtrip(self, K, sig):
        price = black_scholes_call(_S, K, _T, _r, _q, sig)
        iv = implied_vol_from_price(price, _S, K, _T, _r, _q, "C")
        assert iv is not None, f"IV inversion failed for K={K}, sig={sig}"
        assert abs(iv - sig) < 1e-4, f"Round-trip error: in={sig:.4f}, out={iv:.4f}"

    def test_heston_implied_vol_finite(self):
        """Heston model-implied vol should be a finite positive number."""
        p = _PARAMS
        call = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        iv = implied_vol_from_price(call, _S, _K_atm, _T, _r, _q, "C")
        assert iv is not None, "Heston ATM call IV inversion failed"
        assert 0.05 < iv < 1.0, f"IV={iv:.4f} outside (5%, 100%) range"


# ── Greeks Signs and Magnitudes ───────────────────────────────────────────────

class TestGreeks:
    """Greeks must have correct signs for long options."""

    def test_call_delta_range(self):
        """Call delta ∈ (0, 1)."""
        p = _PARAMS
        g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right="C")
        assert 0.0 < g["delta"] < 1.0, f"Call delta={g['delta']:.4f} not in (0,1)"

    def test_put_delta_range(self):
        """Put delta ∈ (-1, 0)."""
        p = _PARAMS
        g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right="P")
        assert -1.0 < g["delta"] < 0.0, f"Put delta={g['delta']:.4f} not in (-1,0)"

    def test_gamma_positive(self):
        """Gamma > 0 for both calls and puts (convexity always helps the buyer)."""
        p = _PARAMS
        for right in ["C", "P"]:
            g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right=right)
            assert g["gamma"] > 0, f"{right} gamma={g['gamma']:.6f} not positive"

    def test_vega_positive(self):
        """Vega > 0: higher vol → higher option value (for long options)."""
        p = _PARAMS
        for right in ["C", "P"]:
            g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right=right)
            assert g["vega"] > 0, f"{right} vega={g['vega']:.4f} not positive"

    def test_theta_negative(self):
        """Theta < 0: time decay hurts long option holder."""
        p = _PARAMS
        for right in ["C", "P"]:
            g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right=right)
            assert g["theta"] < 0, f"{right} theta={g['theta']:.4f} not negative"

    def test_atm_delta_near_half(self):
        """ATM call delta should be near 0.5 (before dividend adjustment)."""
        p = _PARAMS
        g = heston_greeks(_S, _K_atm, _T, _r, _q, **p, right="C")
        assert 0.4 < g["delta"] < 0.6, (
            f"ATM call delta={g['delta']:.4f} too far from 0.5"
        )


# ── VIX Model Tests ───────────────────────────────────────────────────────────

class TestHestonVIX:
    """Verify Heston VIX model formulas."""

    def test_vix_at_long_run_mean(self):
        """When kappa is large, VIX → sqrt(theta) * 100 regardless of v0."""
        large_kappa = 500.0   # extremely fast reversion: v0 irrelevant at 30d horizon
        theta = 0.0625        # = 25% vol
        v0 = 0.04             # = 20% vol (different from theta)
        vix = heston_vix_index(large_kappa, theta, v0)
        expected = np.sqrt(theta) * 100.0   # = 25.0
        assert abs(vix - expected) < 0.5, (
            f"High-kappa VIX={vix:.2f} should converge to {expected:.2f}"
        )

    def test_vix_at_current_vol(self):
        """When kappa→0, VIX ≈ sqrt(v0) * 100."""
        small_kappa = 0.001
        theta = 0.0625
        v0 = 0.0400   # = 20% vol
        vix = heston_vix_index(small_kappa, theta, v0)
        expected = np.sqrt(v0) * 100.0   # ≈ 20.0
        assert abs(vix - expected) < 0.5, (
            f"Low-kappa VIX={vix:.2f} should converge to {expected:.2f}"
        )

    def test_vix_futures_curve_shape(self):
        """VIX futures term structure should be monotone toward theta from v0."""
        p = _PARAMS
        expiries = np.array([1/12, 2/12, 3/12, 6/12, 1.0])  # 1m to 12m
        futures = heston_vix_futures_curve(
            p["kappa"], p["theta"], p["sigma"], p["v0"], expiries
        )
        assert len(futures) == len(expiries), "Wrong number of futures prices"
        assert all(f > 0 for f in futures), "All VIX futures must be positive"
        # When v0 = theta, term structure should be approximately flat
        p_flat = dict(kappa=2.0, theta=0.04, sigma=0.3, v0=0.04)
        futures_flat = heston_vix_futures_curve(
            p_flat["kappa"], p_flat["theta"], p_flat["sigma"], p_flat["v0"], expiries
        )
        spread = max(futures_flat) - min(futures_flat)
        assert spread < 2.0, f"Flat term structure spread {spread:.2f} too wide"

    def test_vix_gap_dataframe_structure(self):
        """compute_heston_vix_gap returns correct columns."""
        p = _PARAMS
        market = pd.DataFrame({
            "expiry_month":   ["2024-03", "2024-04", "2024-05"],
            "close":          [20.5, 21.0, 21.5],
            "days_to_expiry": [30, 60, 90],
        })
        gap_df = compute_heston_vix_gap(
            p["kappa"], p["theta"], p["sigma"], p["v0"], market
        )
        assert "model_price" in gap_df.columns
        assert "market_price" in gap_df.columns
        assert "gap" in gap_df.columns
        assert "abs_gap" in gap_df.columns
        assert len(gap_df) == 3

    def test_integrated_variance_reduces_to_v0_at_zero(self):
        """int_0^0 v_s ds / 0 → v0 (limit as T→0)."""
        p = _PARAMS
        very_small_T = 1e-6
        avg_var = heston_integrated_variance(very_small_T, p["kappa"], p["theta"], p["v0"])
        assert abs(avg_var - p["v0"]) < 1e-3, (
            f"avg_var at T→0 should → v0={p['v0']}, got {avg_var:.6f}"
        )


# ── HestonModel Class ─────────────────────────────────────────────────────────

class TestHestonModelClass:
    """Test the HestonModel OOP wrapper."""

    def test_uncalibrated_raises(self):
        model = HestonModel()
        with pytest.raises(RuntimeError, match="not yet calibrated"):
            model.price(K=4500, T=0.25)

    def test_repr_uncalibrated(self):
        model = HestonModel()
        assert "not calibrated" in repr(model)

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve all parameters exactly."""
        model = HestonModel()
        # Manually set params (skip calibration for speed)
        model.params = dict(kappa=1.5, theta=0.06, sigma=0.4, rho=-0.6, v0=0.05)
        model._S = 4500.0
        model._r = 0.045
        model._q = 0.013
        model.is_calibrated = True

        path = str(tmp_path / "heston_test.pkl")
        model.save(path)
        model2 = HestonModel.load(path)

        for key in ["kappa", "theta", "sigma", "rho", "v0"]:
            assert abs(model.params[key] - model2.params[key]) < 1e-10, (
                f"Parameter {key} changed after save/load"
            )


# ── Bates (1996) SVJ Model ────────────────────────────────────────────────────

# Shared Bates params (lam=0 → pure Heston)
_BATES_PARAMS = dict(**_PARAMS, lam=2.0, mu_j=-0.05, sigma_j=0.07)
_B_S = _S
_B_K = _K_atm
_B_T = _T
_B_r = _r
_B_q = _q


class TestBatesCharacteristicFunction:

    def _heston_cf(self, phi):
        return characteristic_function(
            phi, _B_S, _B_T, _B_r, _B_q,
            **_PARAMS
        )

    def _bates_cf(self, phi, lam, mu_j, sigma_j):
        return bates_characteristic_function(
            phi, _B_S, _B_T, _B_r, _B_q,
            **_PARAMS, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
        )

    def test_lam_zero_equals_heston(self):
        """When lam=0, Bates CF collapses to Heston CF (jump term = exp(0) = 1)."""
        phi = 2.5 + 0.0j
        h = self._heston_cf(phi)
        b = self._bates_cf(phi, lam=0.0, mu_j=-0.05, sigma_j=0.05)
        assert abs(b - h) < 1e-10, f"lam=0 Bates should equal Heston: {b} vs {h}"

    def test_returns_complex(self):
        """Output must be a complex number."""
        phi = 1.0 + 0.5j
        result = self._bates_cf(phi, lam=1.0, mu_j=-0.02, sigma_j=0.05)
        assert isinstance(result, complex)

    def test_nonzero_lam_differs_from_heston(self):
        """With lam>0, Bates CF should differ from Heston CF."""
        phi = 3.0 + 0.0j
        h = self._heston_cf(phi)
        b = self._bates_cf(phi, lam=3.0, mu_j=-0.05, sigma_j=0.08)
        assert abs(b - h) > 1e-8, "Non-zero lam should change CF"


class TestBatesCallPrice:

    def _call(self, K=_B_K, T=_B_T, lam=2.0, mu_j=-0.05, sigma_j=0.07):
        return bates_call_price(
            _B_S, K, T, _B_r, _B_q,
            **_PARAMS, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
        )

    def test_lam_zero_matches_heston_call_price(self):
        """Bates with lam=0 should price identically to Heston call price."""
        bates0 = self._call(lam=0.0, mu_j=0.0, sigma_j=0.01)
        heston = heston_call_price(_B_S, _B_K, _B_T, _B_r, _B_q, **_PARAMS)
        assert abs(bates0 - heston) < 0.05, (
            f"lam=0 Bates={bates0:.4f} vs Heston={heston:.4f}"
        )

    def test_nonnegative(self):
        """Call price must be >= 0."""
        price = self._call()
        assert price >= 0.0

    def test_t_zero_early_return(self):
        """T=0 → price is intrinsic value."""
        price = bates_call_price(
            _B_S, _B_K, 0.0, _B_r, _B_q,
            **_PARAMS, lam=2.0, mu_j=-0.05, sigma_j=0.07,
        )
        intrinsic = max(_B_S - _B_K, 0.0)
        assert abs(price - intrinsic) < 1e-8

    def test_otm_lower_than_atm(self):
        """Deep OTM call should be cheaper than ATM call."""
        atm  = self._call(K=_B_S)
        deep = self._call(K=_B_S * 1.30)
        assert deep < atm


class TestBatesCallBatch:

    def _strikes(self):
        return np.array([4000.0, 4250.0, 4500.0, 4750.0, 5000.0])

    def _batch(self, lam=2.0, mu_j=-0.05, sigma_j=0.07):
        return bates_call_batch(
            _B_S, self._strikes(), _B_T, _B_r, _B_q,
            **_PARAMS, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
        )

    def test_shape(self):
        strikes = self._strikes()
        result  = self._batch()
        assert result.shape == strikes.shape

    def test_lam_zero_matches_heston_batch(self):
        """Bates batch with lam=0 should closely match heston_call_batch."""
        strikes  = self._strikes()
        bates0   = bates_call_batch(
            _B_S, strikes, _B_T, _B_r, _B_q,
            **_PARAMS, lam=0.0, mu_j=0.0, sigma_j=0.01,
        )
        heston_b = heston_call_batch(
            _B_S, strikes, _B_T, _B_r, _B_q, **_PARAMS,
        )
        np.testing.assert_allclose(bates0, heston_b, atol=0.05,
                                   err_msg="lam=0 Bates batch should ≈ Heston batch")

    def test_no_arbitrage_all_nonnegative(self):
        """All batch prices must be non-negative."""
        prices = self._batch()
        assert np.all(prices >= 0.0), f"Negative prices found: {prices}"

    def test_monotone_decreasing_in_strike(self):
        """Call prices must be weakly decreasing in strike."""
        prices = self._batch()
        assert np.all(np.diff(prices) <= 1e-6), (
            f"Call prices not monotone in strike: {prices}"
        )

    def test_t_zero_early_return(self):
        """T=0 returns intrinsic values."""
        strikes = self._strikes()
        prices  = bates_call_batch(
            _B_S, strikes, 0.0, _B_r, _B_q,
            **_PARAMS, lam=2.0, mu_j=-0.05, sigma_j=0.07,
        )
        intrinsic = np.maximum(_B_S - strikes, 0.0)
        np.testing.assert_allclose(prices, intrinsic, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
