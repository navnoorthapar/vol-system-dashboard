"""
test_risk_monitor.py — Test suite for C6: Second-Order Greeks Risk Monitor.

Coverage:
  1. Analytical BS Greeks — shapes, signs, magnitudes
  2. Vomma sign analysis — positive in the wings, negative near ATM
  3. Vomma analytic vs central finite difference (same Greek, a.k.a. volga)
  4. QV convexity — exact CIR closed form vs quadrature; MC cross-check
  5. Greeks surface — every node vs an independent quad + brentq reference,
     no well-posed node missing, drops recorded, columns and units
  6. Vomma-instability measure — absolute, per node, scales with the shock
  7. Persistence and provenance (parquet metadata), calibration-file entry
     point and CLI
  8. RiskMonitor class; torch-free import
"""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize
from scipy.stats import norm

from joint_vol_calibration.greeks import risk_monitor as rm
from joint_vol_calibration.greeks.risk_monitor import (
    GREEKS_UNITS,
    MIN_OTM_PRICE_REL,
    RiskMonitor,
    _bs_delta,
    _bs_vanna,
    _bs_vega,
    _bs_volga_numerical,
    _bs_vomma,
    _bs_vomma_numerical,
    build_surface_from_calibration,
    compute_greeks_surface,
    flag_unstable_hedges,
    load_greeks_surface,
    model_iv_shock,
    qv_variance_exact,
    save_greeks_surface,
    simulate_qv_convexity,
)
from joint_vol_calibration.config import ROOT_DIR
from joint_vol_calibration.models.heston import characteristic_function
from joint_vol_calibration.models.implied_vol import IV_NOT_CONVERGED, IV_OK

# ── Shared fixtures ────────────────────────────────────────────────────────────

# 2026-03-24 legacy fit (the one the published surface was built from)
_PARAMS = {
    "kappa": 4.62, "theta": 0.0764, "sigma": 0.84, "rho": -0.99, "v0": 0.0561,
}
_S, _r, _q = 6581.0, 0.045, 0.013

# Small grid for fast surface tests
_FAST_MATURITIES = np.array([30, 91]) / 365.0
_FAST_STRIKES    = np.array([5800.0, 6200.0, 6581.0, 7000.0, 7400.0])


def _make_surface() -> pd.DataFrame:
    """Build a small surface for reuse across test classes (fast)."""
    return compute_greeks_surface(
        S=_S, r=_r, q=_q, params=_PARAMS,
        strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES,
        verbose=False,
    )


def _ref_otm_and_iv(S, K, T, r, q, p):
    """
    Independent reference for one node: Lewis formula by scipy.quad, then the
    OTM option inverted with scipy brentq on the textbook BS formula.
    Returns (undiscounted OTM price / F, implied vol or NaN).
    """
    F = S * np.exp((r - q) * T)
    x = np.log(K / F)

    def integrand(u):
        cf = characteristic_function(np.array([u - 0.5j]), 1.0, T, 0.0, 0.0, **p)[0]
        return np.real(np.exp(-1j * u * x) * cf) / (u * u + 0.25)

    val, _ = integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-15, epsrel=1e-13)
    c = 1.0 - np.exp(x / 2) / np.pi * val
    otm = c if x >= 0 else c - (1.0 - np.exp(x))

    def bs_otm(sig):
        d1 = (-x + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
        d2 = d1 - sig * np.sqrt(T)
        if x >= 0:
            return norm.cdf(d1) - np.exp(x) * norm.cdf(d2)
        return np.exp(x) * norm.cdf(-d2) - norm.cdf(-d1)

    if otm <= 1e-13:
        return otm, np.nan
    iv = optimize.brentq(lambda s: bs_otm(s) - otm, 1e-4, 5.0, xtol=1e-14, rtol=1e-14)
    return otm, iv


def _cal_pickle(tmp_path, q="missing", name="joint_cal_2026-09-11.pkl"):
    """A calibration pickle in the live format (2026-09-11 fit; q not recorded)."""
    cal = {
        "params": {"kappa": 4.999651331591191, "theta": 0.043135521895477605,
                   "sigma": 0.6567535677361634, "rho": -0.9499997077161061,
                   "v0": 0.010880541693622658},
        "leg_losses": {"spx_iv_rmse": 0.2152171959941077},
        "as_of_date": "2026-09-11", "spot_date": "2026-09-11", "run_date": "2026-09-11",
        "S": 7656.97998046875, "r": 0.03913000106811523,
    }
    if q != "missing":
        cal["q"] = q
    path = tmp_path / name
    with open(path, "wb") as f:
        pickle.dump(cal, f)
    return path


# ── 1. Analytical BS Greeks ───────────────────────────────────────────────────

class TestAnalyticBsGreeks:
    """Verify correctness of vectorised BS Greek formulas."""

    _S = np.array([100.0, 100.0, 100.0, 100.0])
    _K = np.array([85.0,  100.0, 115.0, 130.0])
    _T = np.array([0.25,  0.25,  0.25,  0.25])
    _r = np.array([0.05,  0.05,  0.05,  0.05])
    _q = np.array([0.02,  0.02,  0.02,  0.02])
    _s = np.array([0.25,  0.25,  0.25,  0.25])

    def test_vega_always_positive(self):
        """BS vega = S·exp(-qT)·N'(d1)·√T > 0 for all valid inputs."""
        v = _bs_vega(self._S, self._K, self._T, self._r, self._q, self._s)
        assert (v > 0).all(), f"Vega must be positive, got: {v}"

    def test_delta_between_0_and_exp_neg_qt(self):
        """Call delta = exp(-qT)·N(d1) ∈ (0, exp(-qT))."""
        d = _bs_delta(self._S, self._K, self._T, self._r, self._q, self._s)
        upper = np.exp(-self._q * self._T)
        assert (d > 0).all(),      f"Delta must be positive: {d}"
        assert (d < upper).all(),  f"Delta must be < exp(-qT): {d} vs {upper}"

    def test_delta_atm_approx_half(self):
        """ATM call delta ≈ 0.5 (with small dividend adjustment)."""
        d = _bs_delta(np.array([100.0]), np.array([100.0]), np.array([0.25]),
                      np.array([0.05]), np.array([0.0]), np.array([0.25]))[0]
        assert abs(d - 0.5) < 0.10, f"ATM delta {d:.4f} too far from 0.5"

    def test_itm_delta_greater_than_otm(self):
        """ITM call delta > OTM call delta (call delta monotone decreasing in K)."""
        args = (np.array([0.25]), np.array([0.05]), np.array([0.02]), np.array([0.25]))
        d_itm = _bs_delta(np.array([100.0]), np.array([80.0]), *args)[0]
        d_otm = _bs_delta(np.array([100.0]), np.array([120.0]), *args)[0]
        assert d_itm > d_otm, f"ITM delta {d_itm:.4f} must exceed OTM delta {d_otm:.4f}"

    def test_vanna_sign_otm_call(self):
        """OTM call (K > F): d2 < 0 → Vanna = -exp(-qT)·N'(d1)·d2/σ > 0."""
        van = _bs_vanna(np.array([100.0]), np.array([120.0]), np.array([0.25]),
                        np.array([0.05]), np.array([0.02]), np.array([0.25]))[0]
        assert van > 0, f"Vanna for OTM call must be positive, got {van:.6f}"

    def test_vanna_sign_itm_call(self):
        """Deep ITM call (K << F): d2 > 0 → Vanna < 0."""
        van = _bs_vanna(np.array([100.0]), np.array([70.0]), np.array([0.25]),
                        np.array([0.05]), np.array([0.02]), np.array([0.25]))[0]
        assert van < 0, f"Vanna for deep ITM call must be negative, got {van:.6f}"


# ── 2. Vomma Sign Analysis ────────────────────────────────────────────────────

class TestVommaSign:
    """Validate the key property: Vomma > 0 in both wings."""

    def _vomma(self, S, K, T, r, q, sigma):
        return _bs_vomma(
            np.array([S]), np.array([K]), np.array([T]),
            np.array([r]), np.array([q]), np.array([sigma]),
        )[0]

    def test_vomma_positive_deep_otm_call(self):
        v = self._vomma(S=100, K=130, T=0.5, r=0.05, q=0.01, sigma=0.25)
        assert v > 0, f"Deep OTM call Vomma should be positive, got {v:.6f}"

    def test_vomma_positive_deep_itm_call(self):
        v = self._vomma(S=100, K=75, T=0.5, r=0.05, q=0.01, sigma=0.25)
        assert v > 0, f"Deep ITM call Vomma should be positive, got {v:.6f}"

    def test_vomma_negative_near_atm(self):
        """Near-ATM (K=F): d1 > 0 > d2 → Vomma < 0 (locally concave in vol)."""
        v = self._vomma(S=100, K=100, T=0.10, r=0.0, q=0.0, sigma=0.20)
        assert v < 0, f"ATM call Vomma expected negative near ATM, got {v:.6f}"

    def test_vomma_full_surface_far_otm_positive(self):
        """The fast surface HAS far-OTM-call nodes (the old solver dropped or
        mangled them) and all of them have Vomma > 0."""
        df = _make_surface()
        far_otm = df[df["log_moneyness"] > 0.10]
        assert len(far_otm) >= 1, "far-OTM call nodes are missing from the surface"
        assert (far_otm["vomma"] > 0).all(), far_otm[["K", "T", "log_moneyness", "vomma"]]


# ── 3. Vomma analytic vs finite difference ───────────────────────────────────

class TestVommaFiniteDifference:
    """Central difference on the price must match analytic vomma (same Greek)."""

    def _single(self, S, K, T, r, q, sigma):
        a = [np.array([v]) for v in (S, K, T, r, q, sigma)]
        return _bs_vomma(*a)[0], _bs_vomma_numerical(*a)[0]

    def test_matches_atm(self):
        vomma, fd = self._single(100, 100, 0.5, 0.05, 0.02, 0.25)
        assert abs(fd - vomma) / (abs(vomma) + 1e-6) < 0.01

    def test_matches_otm(self):
        vomma, fd = self._single(100, 120, 0.5, 0.05, 0.02, 0.25)
        assert abs(fd - vomma) / (abs(vomma) + 1e-6) < 0.01

    def test_volga_name_is_the_same_function(self):
        assert _bs_volga_numerical is _bs_vomma_numerical

    def test_surface_publishes_one_name(self):
        """'volga' was published as a second column holding the same Greek."""
        df = _make_surface()
        assert "vomma" in df.columns and "volga" not in df.columns


# ── 4. QV Convexity ───────────────────────────────────────────────────────────

def _qv_quad(kappa, theta, sigma, v0, T):
    def var_v(s):
        return (v0 * sigma**2 / kappa * (np.exp(-kappa * s) - np.exp(-2 * kappa * s))
                + theta * sigma**2 / (2 * kappa) * (1 - np.exp(-kappa * s)) ** 2)
    return integrate.quad(lambda s: 2 * var_v(s) * (-np.expm1(-kappa * (T - s))) / kappa,
                          0, T, epsabs=0, epsrel=1e-13)[0]


class TestQvConvexity:

    _BASE = dict(kappa=4.62, theta=0.0764, sigma=0.84, v0=0.0561)

    @pytest.mark.parametrize("kappa", [0.1, 1.0, 4.62, 20.0])
    @pytest.mark.parametrize("T", [1 / 365, 14 / 365, 1.0, 5.0])
    def test_exact_matches_quadrature(self, kappa, T):
        for theta, sigma, v0 in [(0.0764, 0.84, 0.0561), (0.03, 0.3, 1e-4), (0.2, 1.5, 0.36)]:
            exact = qv_variance_exact(kappa, theta, sigma, v0, T)
            ref = _qv_quad(kappa, theta, sigma, v0, T)
            assert exact == pytest.approx(ref, rel=1e-11)

    def test_small_kappa_limit(self):
        """κ → 0: Var[∫v dt] → σ²·v0·T³/3."""
        assert qv_variance_exact(1e-12, 0.05, 0.5, 0.04, 1.0) == pytest.approx(0.25 * 0.04 / 3, rel=1e-9)

    def test_zero_for_zero_sigma(self):
        assert qv_variance_exact(4.62, 0.0764, 0.0, 0.0561, 1.0) == 0.0
        qv = simulate_qv_convexity(kappa=4.62, theta=0.0764, sigma=0.0, v0=0.0561,
                                   T=0.5, n_paths=2000, n_steps=50)
        assert qv < 1e-12

    def test_monotone_in_sigma_and_T(self):
        f = qv_variance_exact
        assert f(4.62, 0.0764, 1.2, 0.0561, 0.5) > f(4.62, 0.0764, 0.2, 0.0561, 0.5)
        assert f(4.62, 0.0764, 0.84, 0.0561, 1.0) > f(4.62, 0.0764, 0.84, 0.0561, 0.1)

    def test_mc_cross_check(self):
        """Full-truncation MC with the trapezoid rule agrees with the closed form
        within its sampling error (seeded)."""
        T = 0.25
        exact = qv_variance_exact(**self._BASE, T=T)
        mc = simulate_qv_convexity(**self._BASE, T=T, n_paths=40_000, n_steps=100,
                                   rng=np.random.default_rng(0))
        assert abs(mc / exact - 1) < 0.03

    def test_surface_uses_exact_value(self):
        """The published qv_convexity is the closed form (was a 10k-path MC
        estimate, 0.6-2.9% off the exact value)."""
        df = _make_surface()
        for T, grp in df.groupby("T"):
            expected = qv_variance_exact(_PARAMS["kappa"], _PARAMS["theta"], _PARAMS["sigma"],
                                         _PARAMS["v0"], T)
            np.testing.assert_allclose(grp["qv_convexity"], expected, rtol=1e-12)


# ── 5. Greeks Surface ─────────────────────────────────────────────────────────

class TestGreeksSurface:

    def test_default_grid_matches_independent_reference(self):
        """
        Every node of the default grid (4 maturities × 15 strikes) against an
        independent quad + brentq reference: each node is either present with
        |IV − ref| < 1e-8, or genuinely below the price floor and recorded as
        dropped. The old surface silently dropped 21/90 nodes (13 well-posed,
        up to $153) and published IVs fabricated from pricer noise in the
        14d/30d call wings (e.g. 28.7% where the true price is ~1e-14).
        """
        mats = np.array([14, 30, 180, 365]) / 365.0
        df = compute_greeks_surface(_S, _r, _q, _PARAMS, maturities=mats, verbose=False)
        prov = df.attrs["provenance"]
        dropped = {(d["T_days"], round(d["log_moneyness"], 4)) for d in prov["dropped_nodes"]}
        present = {(int(r.T_days), round(r.log_moneyness, 4)): r.iv for r in df.itertuples()}
        assert prov["n_grid_nodes"] == 60
        assert len(present) + len(dropped) == 60
        for T in mats:
            F = _S * np.exp((_r - _q) * T)
            for x in rm.DEFAULT_LOG_MONEYNESS:
                otm, ref_iv = _ref_otm_and_iv(_S, F * np.exp(x), T, _r, _q, _PARAMS)
                key = (int(round(T * 365)), round(float(x), 4))
                if otm >= MIN_OTM_PRICE_REL:
                    assert key in present, f"well-posed node {key} (OTM {otm:.2e}·F) missing"
                    assert abs(present[key] - ref_iv) < 1e-8, (key, present[key], ref_iv)
                else:
                    assert key in dropped, f"node {key} below the floor was not recorded"

    def test_fast_grid_nodes_accounted_for(self):
        """2 maturities × 5 strikes: every node is present or recorded as dropped."""
        df = _make_surface()
        prov = df.attrs["provenance"]
        assert prov["n_grid_nodes"] == 10
        assert prov["n_nodes"] + len(prov["dropped_nodes"]) == 10
        assert len(df) == prov["n_nodes"]
        for d in prov["dropped_nodes"]:
            assert d["otm_fwd_rel"] < MIN_OTM_PRICE_REL and d["reason"]

    def test_required_columns(self):
        df = _make_surface()
        required = {"K", "T", "T_days", "log_moneyness", "strike_index", "iv", "otm_price",
                    "delta", "vega", "vomma", "vanna", "qv_convexity", "iv_shock"}
        assert not (required - set(df.columns))

    def test_every_column_has_units(self):
        flagged = flag_unstable_hedges(_make_surface())
        for col in flagged.columns:
            assert col in GREEKS_UNITS, f"column {col} has no stated unit"

    def test_iv_reprices_heston(self):
        """BS at the published IV reproduces the Heston OTM value."""
        m = RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS, as_of_date="2026-03-24")
        m.build(strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES, verbose=False)
        assert m.validate()["iv_reprices"]

    def test_vega_strictly_positive(self):
        assert (_make_surface()["vega"] > 0).all()

    def test_iv_realistic_range(self):
        df = _make_surface()
        assert (df["iv"] > 0.01).all() and (df["iv"] < 3.0).all()

    def test_delta_in_valid_range(self):
        df = _make_surface()
        assert (df["delta"] > 0).all() and (df["delta"] < 1).all()

    def test_as_of_date_column_added(self):
        df = compute_greeks_surface(
            S=_S, r=_r, q=_q, params=_PARAMS,
            strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES,
            as_of_date="2026-03-24", verbose=False,
        )
        assert (df["as_of_date"] == "2026-03-24").all()
        assert df.attrs["provenance"]["as_of_date"] == "2026-03-24"

    def test_out_of_bounds_params_recorded(self):
        """The published surface used ρ = −0.99, outside HESTON_BOUNDS, with no
        record of it anywhere."""
        prov = _make_surface().attrs["provenance"]
        assert prov["params_outside_heston_bounds"] == ["rho"]
        ok = compute_greeks_surface(_S, _r, _q, {**_PARAMS, "rho": -0.9}, strikes=_FAST_STRIKES,
                                    maturities=_FAST_MATURITIES, verbose=False)
        assert ok.attrs["provenance"]["params_outside_heston_bounds"] == []

    def test_inversion_failure_raises(self, monkeypatch):
        """A node above the price floor that fails to invert is a bug: raise."""
        real = rm.implied_vol_black

        def broken(*args, **kwargs):
            iv, status = real(*args, **kwargs)
            status = np.array(status, copy=True)
            status[np.flatnonzero(status == IV_OK)[:1]] = IV_NOT_CONVERGED
            return np.where(status == IV_OK, iv, np.nan), status

        monkeypatch.setattr(rm, "implied_vol_black", broken)
        with pytest.raises(RuntimeError, match="IV inversion failed"):
            _make_surface()

    def test_qv_method_mc_still_available(self):
        df = compute_greeks_surface(_S, _r, _q, _PARAMS, strikes=_FAST_STRIKES,
                                    maturities=_FAST_MATURITIES, verbose=False,
                                    qv_method="mc", mc_n_paths=500, mc_n_steps=20)
        assert (df["qv_convexity"] > 0).all()


# ── 6. Vomma-instability measure ──────────────────────────────────────────────

class TestUnstableMeasure:

    def test_ratio_formula(self):
        """vomma_pnl_ratio = ½·vomma·|Δσ|/vega = ½·d1·d2·|Δσ|/σ."""
        df = flag_unstable_hedges(_make_surface())
        F = _S * np.exp((_r - _q) * df["T"])
        d1 = (np.log(F / df["K"]) + 0.5 * df["iv"] ** 2 * df["T"]) / (df["iv"] * np.sqrt(df["T"]))
        d2 = d1 - df["iv"] * np.sqrt(df["T"])
        np.testing.assert_allclose(df["vomma_pnl_ratio"],
                                   0.5 * d1 * d2 * df["iv_shock"] / df["iv"], rtol=1e-9)
        assert (df["is_unstable"] == (df["vomma_pnl_ratio"].abs() > rm.UNSTABLE_RATIO_THRESHOLD)).all()

    def test_no_flags_without_vol_shock(self):
        df = _make_surface()
        assert not flag_unstable_hedges(df, iv_shock=0.0)["is_unstable"].any()
        assert np.all(model_iv_shock({**_PARAMS, "sigma": 0.0}, np.array([0.1, 1.0])) == 0.0)

    def test_not_scale_invariant(self):
        """A pooled z-score flags the same nodes whatever the vomma level; the
        absolute measure must react when vomma is 10× larger."""
        df = _make_surface()
        base = flag_unstable_hedges(df)["is_unstable"].sum()
        scaled = flag_unstable_hedges(df.assign(vomma=df["vomma"] * 10))["is_unstable"].sum()
        assert scaled > base

    def test_per_node_not_cross_sectional(self):
        """A node's flag must not depend on which other nodes are on the surface
        (the z-score did: it always flagged the ~3 most extreme nodes)."""
        mats = np.array([14, 30, 91, 365]) / 365.0
        df = compute_greeks_surface(_S, _r, _q, _PARAMS, maturities=mats, verbose=False)
        full = flag_unstable_hedges(df)
        part = flag_unstable_hedges(df[df["T_days"] >= 91])
        np.testing.assert_array_equal(full.loc[part.index, "is_unstable"], part["is_unstable"])

    def test_calm_vs_stressed(self):
        """More vol-of-vol and higher variance → larger model shock and more flags."""
        calm = {**_PARAMS, "sigma": 0.3, "v0": 0.01, "rho": -0.7}
        stress = {**_PARAMS, "sigma": 1.5, "v0": 0.36, "rho": -0.7}
        T = np.array([14, 365]) / 365.0
        assert np.all(model_iv_shock(stress, T) > 3 * model_iv_shock(calm, T))
        n = [flag_unstable_hedges(compute_greeks_surface(7000.0, 0.04, 0.013, p,
                                                         verbose=False))["is_unstable"].sum()
             for p in (calm, stress)]
        assert n[0] != n[1]

    def test_model_iv_shock_formula(self):
        """Δσ = n·σ_v·√(v0·h)·B/(2σ̄) with B = (1−e^{−κT})/(κT)."""
        T, h, n = 0.25, 1 / 252, 2.0
        k, th, sv, v0 = (_PARAMS[k] for k in ("kappa", "theta", "sigma", "v0"))
        B = (1 - np.exp(-k * T)) / (k * T)
        expected = n * sv * np.sqrt(v0 * h) * B / (2 * np.sqrt(th + (v0 - th) * B))
        assert model_iv_shock(_PARAMS, T, h, n) == pytest.approx(expected, rel=1e-12)

    def test_injected_extreme_value_gets_flagged(self):
        df = _make_surface().copy()
        df.loc[df.index[0], "vomma"] = df["vega"].iloc[0] * 1e3
        assert flag_unstable_hedges(df).loc[df.index[0], "is_unstable"]

    def test_descriptive_zscore_kept(self):
        flagged = flag_unstable_hedges(_make_surface())
        z = flagged["vomma_zscore"]
        assert abs(z.mean()) < 1e-9 and z.std(ddof=1) == pytest.approx(1.0)

    def test_is_unstable_is_bool_and_original_not_mutated(self):
        df = _make_surface()
        cols = set(df.columns)
        flagged = flag_unstable_hedges(df)
        assert flagged["is_unstable"].dtype == bool
        assert set(df.columns) == cols


# ── 7. Persistence, provenance, entry points ─────────────────────────────────

class TestPersistence:

    def test_round_trip_with_provenance(self, tmp_path):
        df = flag_unstable_hedges(_make_surface())
        path = save_greeks_surface(df, tmp_path / "surface.parquet")
        plain = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), plain.reset_index(drop=True),
                                      check_like=True)
        loaded = load_greeks_surface(path)
        prov = loaded.attrs["provenance"]
        assert prov["params"] == {k: float(v) for k, v in _PARAMS.items()}
        assert (prov["S"], prov["r"], prov["q"]) == (_S, _r, _q)
        assert prov["n_grid_nodes"] == 10 and prov["units"]["vomma"]
        assert prov["unstable_rule"]["threshold"] == rm.UNSTABLE_RATIO_THRESHOLD

    def test_legacy_file_has_no_provenance(self, tmp_path):
        path = tmp_path / "legacy.parquet"
        _make_surface().to_parquet(path, index=False)
        assert load_greeks_surface(path).attrs["provenance"] is None


class TestCalibrationEntryPoint:

    def test_from_calibration_records_source(self, tmp_path):
        m = RiskMonitor.from_calibration(_cal_pickle(tmp_path))
        assert m.as_of_date == "2026-09-11"
        assert m.S == pytest.approx(7656.97998046875)
        # q not recorded in live pickles → the calibrator's own default
        from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
        import inspect
        from joint_vol_calibration.calibration.joint_calibrator import FLAT_CARRY_Q
        assert m.q == FLAT_CARRY_Q
        assert "JointCalibrator default" in m.source["q_source"]
        assert m.source["calibration_file"].endswith("joint_cal_2026-09-11.pkl")

    def test_recorded_q_is_used(self, tmp_path):
        m = RiskMonitor.from_calibration(_cal_pickle(tmp_path, q=0.0125))
        assert m.q == 0.0125 and m.source["q_source"] == "calibration file"

    def test_no_run_date_default(self):
        """as_of_date used to default to date.today() (the run date)."""
        assert RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS).as_of_date is None

    def test_build_surface_from_calibration(self, tmp_path):
        out = tmp_path / "g.parquet"
        df = build_surface_from_calibration(_cal_pickle(tmp_path), out)
        loaded = load_greeks_surface(out)
        prov = loaded.attrs["provenance"]
        assert prov["as_of_date"] == "2026-09-11"
        assert prov["source"]["calibration_file"].endswith("joint_cal_2026-09-11.pkl")
        assert prov["n_nodes"] == len(loaded) == len(df)
        assert prov["n_grid_nodes"] == 90
        assert "is_unstable" in loaded.columns and (loaded["as_of_date"] == "2026-09-11").all()

    def test_cli(self, tmp_path, capsys):
        out = tmp_path / "cli.parquet"
        assert rm.main(["--calibration", str(_cal_pickle(tmp_path)), "--out", str(out)]) == 0
        assert out.exists()
        assert "as_of 2026-09-11" in capsys.readouterr().out


# ── 8. RiskMonitor Class ──────────────────────────────────────────────────────

class TestRiskMonitor:

    def _monitor(self):
        m = RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS, as_of_date="2026-03-24")
        m.build(strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES, verbose=False)
        return m

    def test_build_returns_dataframe(self):
        m = self._monitor()
        assert isinstance(m.surface, pd.DataFrame) and len(m.surface) > 0

    def test_flag_adds_columns(self):
        flagged = self._monitor().flag()
        assert {"is_unstable", "vomma_pnl_ratio", "vomma_zscore"} <= set(flagged.columns)

    def test_validate_all_pass(self):
        checks = self._monitor().validate()
        failures = {k: v for k, v in checks.items() if not v}
        assert not failures, f"validate() failures: {failures}"
        assert "vomma_fd_check" in checks and "volga_close_to_vomma" not in checks

    def test_repr(self):
        m = self._monitor()
        r = repr(m)
        assert "RiskMonitor" in r and "2026-03-24" in r and str(len(m.surface)) in r

    def test_plot_returns_figure(self, tmp_path):
        import matplotlib.pyplot as plt
        m = self._monitor()
        m.flag()
        fig = m.plot(save_path=tmp_path / "heatmap.png")
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "heatmap.png").exists()
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        assert self._monitor().save(path=tmp_path / "surface.parquet").exists()

    def test_build_before_flag_raises(self):
        with pytest.raises(RuntimeError):
            RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS).flag()

    def test_build_before_plot_raises(self):
        with pytest.raises(RuntimeError):
            RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS).plot()


def test_import_does_not_load_torch():
    """The Greeks monitor (and the backtest engine through delta_hedger) used to
    import torch and sklearn only to reach a numpy IV helper in nn_pricer."""
    code = ("import sys, joint_vol_calibration.greeks.risk_monitor, "
            "joint_vol_calibration.models.implied_vol; "
            "sys.exit(int('torch' in sys.modules or 'sklearn' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], cwd=str(ROOT_DIR)).returncode == 0
