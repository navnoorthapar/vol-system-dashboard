"""
test_heston.py — Unit tests for the Heston model implementation.

Tests are designed to verify financial correctness, not just code correctness:
  1. Put-call parity against an INDEPENDENTLY computed put (Lewis put formula)
  2. Known limiting cases (T→0, K→0, K→∞) and no-arbitrage bounds
  3. Monotonicity of option prices (calls ↓ as K↑, puts ↑ as K↑)
  4. Implied-vol inversion: round trips, NaN for ill-posed inputs
  5. Greeks: signs, documented units, accuracy at v0 on its 1e-4 floor
  6. VIX index / futures formulas and the gap function
  7. Monte Carlo (Andersen QE) vs the analytic price at Feller-binding
     parameters, with a correctly computed antithetic standard error
  8. SPX calibration loss: positional weights, no Feller wall

Accuracy of the pricing engine against an independent reference over the
full maturity/strike grid lives in test_heston_reference.py.

Run with:
  pytest tests/test_heston.py -v
"""

import numpy as np
import pandas as pd
import pytest

import joint_vol_calibration.models.heston as heston_mod
from joint_vol_calibration.models.heston import (
    heston_call_price,
    heston_put_price,
    heston_greeks,
    heston_vix_index,
    heston_vix_futures_curve,
    heston_vix_futures_upper_bound,
    heston_integrated_variance,
    compute_heston_vix_gap,
    implied_vol_from_price,
    implied_vol_batch,
    black_scholes_call,
    HestonModel,
    heston_call_batch,
    bates_characteristic_function,
    bates_call_price,
    bates_call_batch,
    characteristic_function,
    heston_monte_carlo,
    mc_option_price,
    calibrate_to_spx,
    _implied_vol_surface_loss,
)
from joint_vol_calibration.tests.test_heston_reference import (
    ARCHIVED_FITS,
    ref_otm_prices,
)

# Default parameters for tests (stable, near-ATM)
_PARAMS = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
_S = 4500.0
_r = 0.045
_q = 0.013
_K_atm = 4500.0
_T = 0.25   # 3 months


def _p(name):
    kappa, theta, sigma, rho, v0 = ARCHIVED_FITS[name]
    return dict(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0)


def _tuple(p):
    return (p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])


# ── Put-Call Parity ───────────────────────────────────────────────────────────

class TestPutCallParity:
    """C - P = S*e^{-qT} - K*e^{-rT}, with P computed INDEPENDENTLY.

    heston_put_price is defined by parity from the call, so comparing the two
    only checks an identity. Here the put comes from the Lewis PUT formula in
    the reference module (different CF algebra, different quadrature).
    """

    @pytest.mark.parametrize("K", [4000, 4200, 4400, 4500])
    @pytest.mark.parametrize("params", [_PARAMS, _p("2026-08-19")], ids=["base", "v0floor"])
    def test_parity_with_independent_put(self, K, params):
        call = heston_call_price(_S, K, _T, _r, _q, **params)
        F = _S * np.exp((_r - _q) * _T)
        assert K < F
        put_ref, rights = ref_otm_prices(_S, [K], _T, _r, _q, _tuple(params))
        assert rights[0] == "P"
        lhs = call - put_ref[0]
        rhs = _S * np.exp(-_q * _T) - K * np.exp(-_r * _T)
        assert abs(lhs - rhs) < 1e-6, f"K={K}: parity error {abs(lhs - rhs):.2e}"


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

    def test_extreme_strikes(self):
        """K -> 0: call -> DF*F; K far OTM: call -> 0."""
        F = _S * np.exp((_r - _q) * _T)
        c = heston_call_batch(_S, np.array([1e-3, 50 * _S]), _T, _r, _q, **_PARAMS)
        assert c[0] == pytest.approx(np.exp(-_r * _T) * (F - 1e-3), rel=1e-12)
        assert c[1] < 1e-9

    def test_non_finite_strike_gives_nan_not_an_exception(self):
        c = heston_call_batch(_S, np.array([np.nan, _K_atm]), _T, _r, _q, **_PARAMS)
        assert np.isnan(c[0]) and c[1] == pytest.approx(heston_call_price(_S, _K_atm, _T, _r, _q, **_PARAMS))

    def test_deprecated_grid_arguments_warn(self):
        with pytest.warns(DeprecationWarning):
            heston_call_batch(_S, np.array([_K_atm]), _T, _r, _q, **_PARAMS, n_points=512)
        with pytest.warns(DeprecationWarning):
            heston_call_price(_S, _K_atm, _T, _r, _q, **_PARAMS, n_integration_points=128)


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

    def test_otm_call_value_increases_with_maturity(self):
        """
        Deep OTM calls are more expensive at longer maturities (positive time
        value). (Formerly misnamed 'decreasing_in_maturity'.)
        """
        p = _PARAMS
        K_otm = 5500  # 22% OTM
        c1 = heston_call_price(_S, K_otm, 0.1, _r, _q, **p)
        c2 = heston_call_price(_S, K_otm, 0.5, _r, _q, **p)
        assert c2 > c1 > 0.0


# ── Implied Volatility ────────────────────────────────────────────────────────

class TestImpliedVolRoundTrip:
    """Price → IV → Price should be identical; ill-posed inputs give NaN."""

    @pytest.mark.parametrize("K,sig", [
        (4300, 0.18), (4500, 0.20), (4700, 0.22), (4000, 0.25)
    ])
    def test_bs_roundtrip(self, K, sig):
        price = black_scholes_call(_S, K, _T, _r, _q, sig)
        iv = implied_vol_from_price(price, _S, K, _T, _r, _q, "C")
        assert abs(iv - sig) < 1e-6, f"Round-trip error: in={sig:.4f}, out={iv:.4f}"

    def test_heston_implied_vol_finite(self):
        """Heston model-implied vol should be a finite positive number."""
        p = _PARAMS
        call = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        iv = implied_vol_from_price(call, _S, _K_atm, _T, _r, _q, "C")
        assert 0.05 < iv < 1.0, f"IV={iv:.4f} outside (5%, 100%) range"

    def test_zero_otm_price_is_nan_not_a_tiny_vol(self):
        """Pre-C22 a clipped OTM price of exactly 0 inverted to 1e-4 (0.01%)."""
        assert np.isnan(implied_vol_from_price(0.0, _S, 5000.0, _T, _r, _q, "C"))
        assert np.isnan(implied_vol_from_price(0.0, _S, 4000.0, _T, _r, _q, "P"))

    def test_price_at_or_below_intrinsic_is_nan(self):
        F = _S * np.exp((_r - _q) * _T)
        intrinsic = np.exp(-_r * _T) * (F - 4000.0)
        for px in [intrinsic, intrinsic - 5e-5, intrinsic - 1.0]:
            assert np.isnan(implied_vol_from_price(px, _S, 4000.0, _T, _r, _q, "C"))

    def test_out_of_bounds_and_non_finite_are_nan(self):
        assert np.isnan(implied_vol_from_price(_S * 1.01, _S, _K_atm, _T, _r, _q, "C"))
        assert np.isnan(implied_vol_from_price(np.nan, _S, _K_atm, _T, _r, _q, "C"))
        assert np.isnan(implied_vol_from_price(np.inf, _S, _K_atm, _T, _r, _q, "C"))
        assert np.isnan(implied_vol_from_price(10.0, _S, _K_atm, 0.0, _r, _q, "C"))

    @pytest.mark.parametrize("vol", [6.0, 15.0])
    def test_extreme_short_dated_vol_round_trips(self, vol):
        """The old fixed [1e-4, 5] bracket returned None above 500% vol."""
        T = 1 / 365
        price = black_scholes_call(_S, _S, T, _r, _q, vol)
        assert implied_vol_from_price(price, _S, _S, T, _r, _q, "C") == pytest.approx(vol, abs=1e-6)

    @pytest.mark.parametrize("K", [3800.0, 4500.0, 5200.0])
    def test_call_and_put_give_the_same_vol(self, K):
        """Parity-consistent call/put prices must invert to one vol, on either
        side of the forward (the inversion works on the OTM option)."""
        call = black_scholes_call(_S, K, _T, _r, _q, 0.31)
        put = call - _S * np.exp(-_q * _T) + K * np.exp(-_r * _T)
        iv_c = implied_vol_from_price(call, _S, K, _T, _r, _q, "C")
        iv_p = implied_vol_from_price(put, _S, K, _T, _r, _q, "p")
        assert iv_c == pytest.approx(0.31, abs=1e-6)     # tol=1e-6 (documented)
        assert iv_p == pytest.approx(0.31, abs=1e-6)

    def test_batch_inversion_marks_failures_nan(self):
        ivs = implied_vol_batch(np.array([0.0, black_scholes_call(_S, 4700, _T, _r, _q, 0.2)]),
                                _S, np.array([5000.0, 4700.0]), _T, _r, _q, np.array(["C", "C"]))
        assert np.isnan(ivs[0]) and ivs[1] == pytest.approx(0.2, abs=1e-6)


# ── Greeks Signs and Magnitudes ───────────────────────────────────────────────

def _ref_call(S, K, T, p):
    """Independent call price from the reference Lewis/Gauss-Legendre pricer."""
    px, rights = ref_otm_prices(S, [K], T, _r, _q, _tuple(p))
    return px[0] + (S * np.exp(-_q * T) - K * np.exp(-_r * T) if rights[0] == "P" else 0.0)


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

    def test_theta_is_one_calendar_day_of_decay(self):
        """'theta' is $ per calendar day: V(T - 1d) - V(T) (pre-C22 it was the
        annualised rate, 365x larger, while the docstring said 'per day')."""
        g = heston_greeks(_S, _K_atm, _T, _r, _q, **_PARAMS, right="C")
        one_day = _ref_call(_S, _K_atm, _T - 1 / 365, _PARAMS) - _ref_call(_S, _K_atm, _T, _PARAMS)
        assert g["theta"] == pytest.approx(one_day, rel=1e-6)
        assert g["theta_annual"] == pytest.approx(365.0 * one_day, rel=1e-6)

    def test_vomma_in_vol_units_is_positive_for_atm(self):
        """Vomma (d²V/dσ_v², σ_v = sqrt(v0)) is positive ATM; the pre-C22 value
        was d²V/dv0² (variance units), which is negative there."""
        g = heston_greeks(_S, _K_atm, _T, _r, _q, **_PARAMS, right="C")
        assert g["vomma"] > 0
        assert g["vomma_v0"] < 0
        # chain rule linking the two unit systems
        vega_v0 = g["vega"] / (2.0 * np.sqrt(_PARAMS["v0"]))
        assert g["vomma"] == pytest.approx(4 * _PARAMS["v0"] * g["vomma_v0"] + 2 * vega_v0, rel=1e-10)

    @pytest.mark.parametrize("T,K", [(30 / 365, 7600.0), (91 / 365, 7200.0), (7 / 365, 7600.0)])
    def test_accuracy_at_v0_floor(self, T, K):
        """v0 = 1e-4 (six archived fits). Pre-C22: gamma 7% and vomma_v0 6-76%
        off, vega 20% off at 7d. Reference = Richardson-extrapolated finite
        differences of the independent reference pricer."""
        p = _p("2026-08-19")
        S, v0 = 7600.0, p["v0"]
        g = heston_greeks(S, K, T, _r, _q, **p, right="C")

        def f(S_=S, v_=v0):
            return _ref_call(S_, K, T, dict(p, v0=v_))

        def d2S(h):
            dS = S * h
            return (f(S_=S + dS) - 2 * f() + f(S_=S - dS)) / dS**2

        def dv(h):
            d = v0 * h
            return (f(v_=v0 + d) - f(v_=v0 - d)) / (2 * d)

        def d2v(h):
            d = v0 * h
            return (f(v_=v0 + d) - 2 * f() + f(v_=v0 - d)) / d**2

        gamma = (4 * d2S(0.001) - d2S(0.002)) / 3
        V_v = (4 * dv(0.025) - dv(0.05)) / 3
        V_vv = (4 * d2v(0.025) - d2v(0.05)) / 3
        assert g["gamma"] == pytest.approx(gamma, rel=1e-3)
        assert g["vega"] == pytest.approx(2 * np.sqrt(v0) * V_v, rel=1e-3)
        assert g["vomma_v0"] == pytest.approx(V_vv, rel=1e-2)
        assert g["vomma"] == pytest.approx(4 * v0 * V_vv + 2 * V_v, rel=1e-2)


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
        """When kappa→0, VIX → sqrt(v0) * 100."""
        theta = 0.0625
        v0 = 0.0400   # = 20% vol
        vix = heston_vix_index(1e-10, theta, v0)
        assert abs(vix - 20.0) < 1e-6, f"Low-kappa VIX={vix:.6f} should converge to 20"

    def test_vix_futures_curve_shape(self):
        """Exact futures: right length, positive, and strictly below the
        sqrt(E[VIX^2]) upper bound (Jensen) by a margin that grows with T."""
        p = _PARAMS
        expiries = np.array([1/12, 2/12, 3/12, 6/12, 1.0])  # 1m to 12m
        futures = heston_vix_futures_curve(
            p["kappa"], p["theta"], p["sigma"], p["v0"], expiries
        )
        upper = heston_vix_futures_upper_bound(
            p["kappa"], p["theta"], p["sigma"], p["v0"], expiries
        )
        assert len(futures) == len(expiries), "Wrong number of futures prices"
        assert all(f > 0 for f in futures), "All VIX futures must be positive"
        gap = upper - futures
        assert np.all(gap > 0) and np.all(np.diff(gap) > 0)

    def test_vix_futures_depend_on_vol_of_vol(self):
        """sqrt(E[VIX^2]) ignores sigma; the exact price must not."""
        T = np.array([0.25, 0.5])
        lo = heston_vix_futures_curve(2.0, 0.04, 0.2, 0.03, T)
        hi = heston_vix_futures_curve(2.0, 0.04, 0.8, 0.03, T)
        assert np.all(hi < lo - 0.1)

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
        np.testing.assert_allclose(
            gap_df["model_price"],
            heston_vix_futures_curve(p["kappa"], p["theta"], p["sigma"], p["v0"],
                                     np.array([30, 60, 90]) / 365.0))

    def test_integrated_variance_reduces_to_v0_at_zero(self):
        """int_0^0 v_s ds / 0 → v0 (limit as T→0)."""
        p = _PARAMS
        avg_var = heston_integrated_variance(1e-6, p["kappa"], 0.09, p["v0"])
        assert avg_var == pytest.approx(p["v0"] + (0.09 - p["v0"]) * p["kappa"] * 1e-6 / 2, rel=1e-9)

    def test_integrated_variance_no_cancellation_for_small_kappa_t(self):
        """kappa*T just above the old 1e-8 switch: (1-e^{-x})/x via expm1."""
        x = 3e-8
        got = heston_integrated_variance(1.0, x, 0.09, 0.04)
        assert got == pytest.approx(0.04 + 0.05 * x / 2, rel=1e-13)


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

    @staticmethod
    def _calibrated_model():
        model = HestonModel()
        model.params = dict(kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04)
        model._S = 4500.0
        model._r = 0.045
        model._q = 0.013
        model.is_calibrated = True
        return model

    def test_smile_surface_shape_and_columns(self):
        model = self._calibrated_model()
        strikes = np.array([4200.0, 4500.0, 4800.0])
        mats = np.array([0.25, 0.5])
        surf = model.smile_surface(S=4500.0, strikes=strikes, maturities=mats)
        assert len(surf) == len(strikes) * len(mats)   # 3 × 2 = 6
        for col in ("strike", "maturity", "moneyness", "log_moneyness",
                    "implied_vol_c", "implied_vol_p"):
            assert col in surf.columns
        # Inverted implied vols should be present, positive and sane.
        assert surf["implied_vol_c"].notna().all()
        assert (surf["implied_vol_c"] > 0).all()
        assert (surf["implied_vol_c"] < 2.0).all()
        assert (surf["implied_vol_p"] > 0).all()
        # call and put at the same strike are the same model: one vol
        np.testing.assert_allclose(surf["implied_vol_c"], surf["implied_vol_p"], atol=1e-6)


# ── SPX calibration loss ──────────────────────────────────────────────────────

def _synthetic_surface(params, S=_S, expiries=(0.01, 0.1, 0.25, 0.5), m=(0.9, 0.95, 1.0, 1.05, 1.1)):
    rows = []
    for T in expiries:
        F = S * np.exp((_r - _q) * T)
        strikes = np.array(m) * F
        calls = heston_call_batch(S, strikes, T, _r, _q, *_tuple(params))
        for K, c in zip(strikes, calls):
            right = "C" if K >= F else "P"
            px = c if right == "C" else c - S * np.exp(-_q * T) + K * np.exp(-_r * T)
            iv = implied_vol_from_price(px, S, K, T, _r, _q, right)
            rows.append(dict(strike=K, time_to_expiry=T, right=right, implied_vol=iv,
                             bid=px * 0.99, ask=px * 1.01))
    return pd.DataFrame(rows)


class TestCalibrationLoss:

    def test_loss_is_zero_at_generating_params_with_bidask_weights(self):
        """calibrate_to_spx drops the T=0.01 rows, leaving index labels 5..19;
        the old loss used those LABELS to index a 15-long weight array (wrong
        weights, then IndexError swallowed as +1.0 per row: loss 0.333)."""
        surf = _synthetic_surface(_PARAMS)
        filt = surf[surf["time_to_expiry"] > 0.02]
        assert filt.index[0] == 5                       # non-contiguous labels
        spread = (filt["ask"] - filt["bid"]).to_numpy()
        w = (1.0 / spread) / (1.0 / spread).sum() * len(spread)
        x = np.array(_tuple(_PARAMS))
        assert _implied_vol_surface_loss(x, filt, _S, _r, _q, w) < 1e-10
        assert _implied_vol_surface_loss(x, filt, _S, _r, _q, None) < 1e-10

    def test_weights_must_align_with_rows(self):
        surf = _synthetic_surface(_PARAMS)
        with pytest.raises(ValueError):
            _implied_vol_surface_loss(np.array(_tuple(_PARAMS)), surf, _S, _r, _q, np.ones(3))

    def test_no_feller_wall(self):
        """Feller-violating parameters are valid Heston parameters; the old loss
        returned 1e10 there even for a surface generated by them."""
        p = dict(kappa=2.5, theta=0.05, sigma=0.6, rho=-0.6, v0=0.03)
        assert p["sigma"]**2 > 2 * p["kappa"] * p["theta"]
        surf = _synthetic_surface(p)
        x = np.array(_tuple(p))
        assert _implied_vol_surface_loss(x, surf, _S, _r, _q) < 1e-10
        soft = _implied_vol_surface_loss(x, surf, _S, _r, _q, feller_penalty=50.0)
        assert soft == pytest.approx(50.0 * (p["sigma"]**2 - 2 * p["kappa"] * p["theta"]), rel=1e-6)

    def test_unpriceable_option_penalty_is_explicit(self):
        """A row whose model price has no IV adds exactly 1.0 (no bare except)."""
        surf = pd.DataFrame([dict(strike=2.0 * _S, time_to_expiry=7 / 365, right="C",
                                  implied_vol=0.2)])
        loss = _implied_vol_surface_loss(np.array(_tuple(_PARAMS)), surf, _S, _r, _q)
        assert loss == 1.0

    def test_calibrate_recovers_feller_violating_params(self):
        """End-to-end public workflow (L-BFGS-B from HESTON_DEFAULTS)."""
        p = dict(kappa=2.5, theta=0.05, sigma=0.6, rho=-0.6, v0=0.03)
        surf = _synthetic_surface(p, expiries=(0.01, 0.1, 0.3, 0.7))
        res = calibrate_to_spx(surf, _S, _r, _q, method="minimize")
        assert res["n_obs"] == 15
        assert res["n_iv_failures"] == 0
        assert res["loss"] < 1e-7          # RMSE < 0.03 vol pts
        model = HestonModel()
        res2 = model.calibrate(surf, _S, _r, _q, method="minimize")
        assert res2["loss"] == pytest.approx(res["loss"])


# ── Bates (1996) SVJ Model ────────────────────────────────────────────────────

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

    def test_returns_complex(self):
        """Output must be a complex number."""
        phi = 1.0 + 0.5j
        result = self._bates_cf(phi, lam=1.0, mu_j=-0.02, sigma_j=0.05)
        assert isinstance(result, complex)

    def test_jump_factor_matches_closed_form(self):
        """Bates CF / Heston CF = compound-Poisson CF with compensator,
        computed here by hand."""
        phi = 3.0 - 0.5j
        lam, mu_j, sigma_j = 3.0, -0.05, 0.08
        kbar = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        expected = np.exp(lam * _B_T * (np.exp(1j * phi * mu_j - 0.5 * phi**2 * sigma_j**2)
                                        - 1 - 1j * phi * kbar))
        ratio = self._bates_cf(phi, lam, mu_j, sigma_j) / self._heston_cf(phi)
        assert abs(ratio - expected) < 1e-13


class TestBatesCallPrice:

    def _call(self, K=_B_K, T=_B_T, lam=2.0, mu_j=-0.05, sigma_j=0.07):
        return bates_call_price(
            _B_S, K, T, _B_r, _B_q,
            **_PARAMS, lam=lam, mu_j=mu_j, sigma_j=sigma_j,
        )

    def test_vanishing_intensity_converges_to_heston(self):
        """lam -> 0+ exercises the jump code path and must converge to Heston."""
        bates = self._call(lam=1e-9, mu_j=-0.2, sigma_j=0.1)
        heston = heston_call_price(_B_S, _B_K, _B_T, _B_r, _B_q, **_PARAMS)
        assert abs(bates - heston) < 1e-6

    def test_negative_jumps_fatten_the_left_tail(self):
        """Downward jumps raise OTM put value (via parity) relative to Heston."""
        K = 0.8 * _B_S
        df_K = K * np.exp(-_B_r * _B_T) - _B_S * np.exp(-_B_q * _B_T)
        put_b = self._call(K=K, lam=2.0, mu_j=-0.10, sigma_j=0.05) + df_K
        put_h = heston_call_price(_B_S, K, _B_T, _B_r, _B_q, **_PARAMS) + df_K
        assert put_b > put_h + 1.0

    def test_nonnegative(self):
        """Call price must be >= 0."""
        price = self._call()
        assert price >= 0.0

    def test_t_zero_returns_intrinsic(self):
        """T=0 → price is intrinsic value (ITM strike, so it is not trivially 0)."""
        K = 4000.0
        price = bates_call_price(
            _B_S, K, 0.0, _B_r, _B_q,
            **_PARAMS, lam=2.0, mu_j=-0.05, sigma_j=0.07,
        )
        assert abs(price - (_B_S - K)) < 1e-8

    def test_otm_lower_than_atm(self):
        """Deep OTM call should be cheaper than ATM call."""
        atm  = self._call(K=_B_S)
        deep = self._call(K=_B_S * 1.30)
        assert deep < atm

    def test_single_equals_batch(self):
        strikes = np.array([4000.0, 4500.0, 5000.0])
        batch = bates_call_batch(_B_S, strikes, _B_T, _B_r, _B_q, **_BATES_PARAMS)
        single = [self._call(K=K) for K in strikes]
        np.testing.assert_allclose(single, batch, atol=1e-10)


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


# ── Monte Carlo pricer ↔ analytic cross-validation ────────────────────────────

_MC_CASES = {
    # Feller binding/violated: where the pre-C22 'absorption' scheme (state
    # floored at 0) was 5-30 standard errors too rich.
    "2026-08-19": _p("2026-08-19"),   # v0 = 1e-4, 2*kappa*theta ~ sigma^2
    "2026-09-11": _p("2026-09-11"),
    "feller_violated": dict(kappa=0.5, theta=0.04, sigma=1.0, rho=-0.9, v0=0.04),
}


@pytest.fixture(scope="module", params=list(_MC_CASES))
def case(request):
    p = _MC_CASES[request.param]
    pS, pV = heston_monte_carlo(_S, _T, _r, _q, **p,
                                n_paths=60_000, n_steps_per_year=252, seed=42)
    return p, pS, pV


class TestMonteCarlo:
    """An INDEPENDENT MC estimate must agree with the analytic price within
    Monte Carlo error, at parameters where discretisation actually matters
    (Feller binding or violated), using the antithetic pair-mean SE."""

    @pytest.mark.parametrize("m,right", [(1.0, "C"), (1.1, "C"), (0.9, "P")])
    def test_qe_matches_analytic(self, case, m, right):
        p, pS, _ = case
        F = _S * np.exp((_r - _q) * _T)
        K = m * F
        call = heston_call_price(_S, K, _T, _r, _q, **p)
        analytic = call if right == "C" else call - _S * np.exp(-_q * _T) + K * np.exp(-_r * _T)
        mc, se = mc_option_price(pS, K, _T, _r, right=right)
        assert abs(mc - analytic) < 4 * se, (
            f"K/F={m} {right}: MC={mc:.4f} vs analytic={analytic:.4f} "
            f"({(mc - analytic) / se:+.1f} SE)"
        )

    def test_variance_paths_nonnegative(self, case):
        _, _, pV = case
        assert (pV >= 0).all()

    def test_spot_paths_strictly_positive(self, case):
        _, pS, _ = case
        assert (pS > 0).all()

    def test_initial_conditions(self, case):
        p, pS, pV = case
        np.testing.assert_allclose(pS[:, 0], _S)
        np.testing.assert_allclose(pV[:, 0], p["v0"])

    def test_discrete_martingale(self, case):
        """QE with Andersen's martingale correction: E[S_T] = F within MC error."""
        _, pS, _ = case
        F = _S * np.exp((_r - _q) * _T)
        half = pS.shape[0] // 2
        pm = 0.5 * (pS[:half, -1] + pS[half:, -1])
        assert abs(pm.mean() - F) < 4 * pm.std(ddof=1) / np.sqrt(half)

    def test_full_truncation_matches_analytic_at_moderate_params(self):
        p = _p("2026-09-11")
        pS, pV = heston_monte_carlo(_S, _T, _r, _q, **p, n_paths=60_000, seed=3,
                                    scheme="full_truncation")
        assert (pV >= 0).all()
        analytic = heston_call_price(_S, _K_atm, _T, _r, _q, **p)
        mc, se = mc_option_price(pS, _K_atm, _T, _r, right="C")
        assert abs(mc - analytic) < 4 * se

    def test_reported_se_matches_seed_to_seed_dispersion(self):
        """With antithetic pairs the naive i.i.d. SE overstates the error (by
        ~1.5x ATM here). The reported SE must match the actual spread of
        estimates across independent seeds."""
        est, ses = [], []
        for seed in range(60):
            pS, _ = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=2_000,
                                       n_steps_per_year=52, seed=1000 + seed)
            mc, se = mc_option_price(pS, _K_atm, _T, _r, right="C")
            est.append(mc)
            ses.append(se)
        ratio = np.mean(ses) / np.std(est, ddof=1)
        assert 0.8 < ratio < 1.25, f"reported SE / empirical sd = {ratio:.2f}"

    def test_pair_se_formula(self):
        pS, _ = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=1_000, seed=5)
        disc = np.exp(-_r * _T) * np.maximum(pS[:, -1] - _K_atm, 0.0)
        pm = 0.5 * (disc[:500] + disc[500:])
        mc, se = mc_option_price(pS, _K_atm, _T, _r, right="C")
        assert mc == pytest.approx(disc.mean(), rel=1e-12)
        assert se == pytest.approx(pm.std(ddof=1) / np.sqrt(500), rel=1e-12)
        _, se_naive = mc_option_price(pS, _K_atm, _T, _r, right="C", antithetic=False)
        assert se_naive == pytest.approx(disc.std(ddof=1) / np.sqrt(1000), rel=1e-12)

    def test_odd_n_paths_antithetic_no_crash(self):
        """Regression: antithetic variates with an ODD n_paths previously
        allocated mismatched arrays and crashed on broadcast. Now n_paths is
        re-synced to the actual antithetic count (n_paths-1)."""
        pS, pV = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS,
                                    n_paths=9_999, seed=1)
        assert pS.shape[0] == 9_998        # 2 * (9999 // 2)
        assert pS.shape == pV.shape
        # And it still prices sanely.
        mc, se = mc_option_price(pS, _K_atm, _T, _r, right="C")
        assert mc > 0 and se > 0

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError):
            heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=100, scheme="euler")

    def test_cache_is_keyed_by_scheme(self, tmp_path, monkeypatch):
        monkeypatch.setattr(heston_mod, "MC_CACHE_DIR", tmp_path)
        a, _ = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=200, seed=9,
                                  cache_key="k", scheme="full_truncation")
        b, _ = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=200, seed=9,
                                  cache_key="k", scheme="qe")
        assert not np.array_equal(a, b)
        c, _ = heston_monte_carlo(_S, _T, _r, _q, **_PARAMS, n_paths=200, seed=123,
                                  cache_key="k", scheme="qe")
        np.testing.assert_array_equal(b, c)     # served from the qe cache file
        assert len(list(tmp_path.glob("k__*.pkl"))) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
