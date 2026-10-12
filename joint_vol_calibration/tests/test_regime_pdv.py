"""
test_regime_pdv.py — Tests for C11: Regime-Switching PDV with Merton Jump Component

Test coverage (13 test classes):
  1. merton_log_density          — output shape, finite, n=0 reduces to Normal,
                                   integrates to one
  2. calibrate_jump_mle          — types, bounds, parameter recovery on simulated
                                   Merton data, success=False handling, clipped
                                   fallback, bound-activity reporting
  3. MertonJumpParams            — properties (jump_variance_annual, jump_vol_annual, repr)
  4. RegimePDV.load              — FileNotFoundError on missing paths; returns
                                   taken from the pickle
  5. RegimePDV.calibrate_jump    — zero look-ahead, fallback when too few R2 days
  6. RegimePDV.calibrate_jump_tail — detected-jump estimator, cache key, fallback
  7. RegimePDV.forecast          — R0/R1 zero jump_adj, R2 positive jump_adj,
                                   result keys, staleness
  8. compare_covid_2020          — columns, scoring against the NEXT day's return
  9. Zero look-ahead             — future labels, future returns (poison), and the
                                   ex-ante diffusion vol
  10. Integration                — end-to-end on saved artifacts
  11. PDVLinear4F
  12. daily_jump_test            — size under Gaussian and GARCH nulls, power,
                                   causality
  13. calibrate_jump_tail alpha  — smaller alpha flags fewer jumps
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# np.trapz was removed in NumPy 2.0 (renamed np.trapezoid) — same cross-version
# helper used in heston.py / joint_calibrator.py.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# Module-level picklable stub (cannot be defined inside a test method)
class _FakePDVModel:
    is_fitted = True
    _features = pd.DataFrame(
        {"sigma1": [0.2]},
        index=pd.to_datetime(["2020-01-02"]),
    )


from joint_vol_calibration.models.pdv import PDVModel
from joint_vol_calibration.signals import regime_pdv as rp
from joint_vol_calibration.signals.regime_pdv import (
    ANNUALISE,
    MIN_R2_DAYS,
    MIN_TAIL_DAYS,
    _BOUNDS_FULL,
    _moment_match_fallback,
    MertonJumpParams,
    RegimePDV,
    calibrate_jump_mle,
    daily_jump_test,
    jump_test_critical_value,
    merton_log_density,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_returns(
    n: int = 600,
    seed: int = 42,
    start: str = "2010-01-04",
    jump_prob: float = 0.03,
    jump_mu: float = -0.02,
    jump_sd: float = 0.03,
) -> pd.Series:
    """GARCH(1,1) daily log returns with occasional Gaussian jumps."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = 1e-4
    for t in range(n):
        if t:
            s2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
        r[t] = rng.normal(0.0, np.sqrt(s2[t]))
        if rng.random() < jump_prob:
            r[t] += rng.normal(jump_mu, jump_sd)
    return pd.Series(r, index=pd.bdate_range(start, periods=n), name="log_return")


def _make_regime_labels(
    index: pd.DatetimeIndex,
    r2_fraction: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic regime labels (default R1, ~r2_fraction R2)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    regimes = np.ones(n, dtype=int)
    r2_idx = rng.choice(n, size=int(n * r2_fraction), replace=False)
    regimes[r2_idx] = 2
    return pd.DataFrame({"regime": regimes}, index=index)


def _build_regime_pdv(
    r2_fraction: float = 0.30,
    n_days: int = 600,
    seed: int = 42,
    jump_prob: float = 0.03,
    jump_mu: float = -0.02,
) -> RegimePDV:
    """A loaded RegimePDV on synthetic in-memory data — no disk I/O."""
    r = _make_returns(n=n_days, seed=seed, jump_prob=jump_prob, jump_mu=jump_mu)
    lbl = _make_regime_labels(r.index, r2_fraction=r2_fraction, seed=seed)
    return RegimePDV.from_data(r, lbl)


def _simulate_merton(T, lam, mu_j, sigma_j, sigma_d=0.15, seed=42):
    rng = np.random.default_rng(seed)
    sd = sigma_d / np.sqrt(ANNUALISE)
    r = rng.normal(0.0, sd, T)
    n_jumps = rng.poisson(lam / ANNUALISE, T)
    for t, k in enumerate(n_jumps):
        if k > 0:
            r[t] += rng.normal(mu_j, sigma_j, k).sum()
    return r, np.full(T, sd)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. merton_log_density
# ═══════════════════════════════════════════════════════════════════════════════

class TestMertonLogDensity:

    def _sigma_d(self, n: int) -> np.ndarray:
        return np.full(n, 0.15 / np.sqrt(ANNUALISE))  # daily std of 15% ann vol

    def test_output_shape(self):
        r = np.random.default_rng(0).normal(0, 0.01, 50)
        s = self._sigma_d(50)
        out = merton_log_density(r, s, lam_daily=2/ANNUALISE, mu_j=-0.01, sigma_j=0.02)
        assert out.shape == (50,), "Output must match input length"

    def test_all_finite(self):
        r = np.array([0.0, 0.01, -0.02, 0.005, -0.03])
        s = self._sigma_d(5)
        out = merton_log_density(r, s, lam_daily=1/ANNUALISE, mu_j=-0.005, sigma_j=0.01)
        assert np.all(np.isfinite(out)), "All log-densities must be finite"

    def test_zero_jump_reduces_to_normal(self):
        """With lam_daily → 0, must match log Normal(r; 0, sigma_d²)."""
        rng = np.random.default_rng(1)
        r = rng.normal(0, 0.01, 100)
        sd = self._sigma_d(100)
        out_jump = merton_log_density(r, sd, lam_daily=1e-9, mu_j=0.0, sigma_j=0.001)
        from scipy.stats import norm
        log_norm = norm.logpdf(r, 0.0, sd)
        np.testing.assert_allclose(out_jump, log_norm, atol=1e-4)

    def test_larger_sigma_j_increases_variance(self):
        """A bigger sigma_j should produce a flatter (lower peak) density at r=0."""
        r    = np.array([0.0])
        sd   = np.array([0.01 / np.sqrt(ANNUALISE)])
        lam  = 3.0 / ANNUALISE
        d_small = merton_log_density(r, sd, lam, mu_j=0.0, sigma_j=0.005)
        d_large = merton_log_density(r, sd, lam, mu_j=0.0, sigma_j=0.050)
        assert d_large[0] < d_small[0], "Larger sigma_j should reduce density at r=0"

    def test_density_integrates_to_one(self):
        """A probability density must integrate to 1 over the return support."""
        grid = np.linspace(-1.0, 1.0, 200_001)
        sigma_d = np.full_like(grid, 0.012)
        ld = merton_log_density(grid, sigma_d, lam_daily=0.05,
                                mu_j=-0.04, sigma_j=0.06)
        integral = _trapz(np.exp(ld), grid)
        assert integral == pytest.approx(1.0, abs=1e-4)

    def test_density_integrates_to_one_no_jumps(self):
        """With λ→0 the mixture collapses to a single Gaussian — still unit mass."""
        grid = np.linspace(-1.0, 1.0, 200_001)
        sigma_d = np.full_like(grid, 0.012)
        ld = merton_log_density(grid, sigma_d, lam_daily=1e-9,
                                mu_j=0.0, sigma_j=0.01)
        assert _trapz(np.exp(ld), grid) == pytest.approx(1.0, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. calibrate_jump_mle
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrateJumpMle:

    def test_output_type(self):
        r, sd = _simulate_merton(300, 4.0, -0.01, 0.025)
        assert isinstance(calibrate_jump_mle(r, sd), MertonJumpParams)

    def test_bounds_respected(self):
        r, sd = _simulate_merton(300, 4.0, -0.01, 0.025)
        params = calibrate_jump_mle(r, sd)
        (l0, l1), (m0, m1), (s0, s1) = _BOUNDS_FULL
        assert l0 <= params.lam <= l1
        assert m0 <= params.mu_j <= m1
        assert s0 <= params.sigma_j <= s1

    def test_n_r2_days_stored(self):
        r, sd = _simulate_merton(200, 4.0, -0.01, 0.025)
        assert calibrate_jump_mle(r, sd).n_r2_days == 200

    def test_nll_is_finite(self):
        r, sd = _simulate_merton(300, 4.0, -0.01, 0.025)
        assert np.isfinite(calibrate_jump_mle(r, sd).nll)

    def test_recovers_crash_parameters_on_simulated_merton_data(self):
        """λ=4/yr, μ_j=−4%, σ_j=3% over 5000 days (~80 jumps): every
        parameter within ±50% of the truth and no bound active."""
        r, sd = _simulate_merton(5000, 4.0, -0.04, 0.03, seed=7)
        p = calibrate_jump_mle(r, sd)
        assert p.lam == pytest.approx(4.0, rel=0.5)
        assert p.mu_j == pytest.approx(-0.04, rel=0.5)
        assert p.sigma_j == pytest.approx(0.03, rel=0.5)
        assert p.bound_active == ()

    def test_success_false_keeps_the_best_optimum(self, monkeypatch):
        """L-BFGS-B often reports success=False (line-search termination) at
        a good point. Pre-C22 that discarded the optimum for a moment-matching
        fallback that ignored the bounds (λ=50 under a λ≤20 box)."""
        rng = np.random.default_rng(3)
        r = rng.standard_t(3, 300) * 0.01
        sd = np.full(300, 0.01)
        bounds = [(0.1, 20.0), (-0.15, 0.05), (0.005, 0.15)]
        honest = calibrate_jump_mle(r, sd, bounds=bounds)

        real_minimize = rp.minimize

        def flaky(*a, **k):
            res = real_minimize(*a, **k)
            res.success = False
            res.message = "ABNORMAL_TERMINATION_IN_LNSRCH"
            return res

        monkeypatch.setattr(rp, "minimize", flaky)
        p = calibrate_jump_mle(r, sd, bounds=bounds)
        assert (p.lam, p.mu_j, p.sigma_j) == (honest.lam, honest.mu_j, honest.sigma_j)
        assert p.method == "mle"
        for v, (lo, hi) in zip((p.lam, p.mu_j, p.sigma_j), bounds):
            assert lo <= v <= hi

    def test_moment_fallback_is_clipped_into_bounds(self):
        rng = np.random.default_rng(3)
        r = rng.standard_t(3, 300) * 0.01
        sd = np.full(300, 0.001)
        bounds = [(0.1, 20.0), (-0.15, 0.05), (0.005, 0.15)]
        lam, mu, sig = _moment_match_fallback(r, sd, bounds)
        for v, (lo, hi) in zip((lam, mu, sig), bounds):
            assert lo <= v <= hi
        assert lam == 20.0          # unclipped it would exceed the box

    def test_bound_activity_is_reported(self):
        """A box that excludes the truth must be flagged, not silently used."""
        r, sd = _simulate_merton(3000, 40.0, -0.01, 0.02, seed=5)
        p = calibrate_jump_mle(r, sd, bounds=[(0.1, 2.0), (-0.2, 0.2), (1e-4, 0.25)])
        assert "lam" in p.bound_active


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MertonJumpParams
# ═══════════════════════════════════════════════════════════════════════════════

class TestMertonJumpParams:

    def test_jump_variance_annual(self):
        p = MertonJumpParams(lam=4.0, mu_j=-0.01, sigma_j=0.02)
        expected = 4.0 * ((-0.01)**2 + (0.02)**2)
        assert abs(p.jump_variance_annual - expected) < 1e-12

    def test_jump_vol_annual_nonneg(self):
        p = MertonJumpParams(lam=2.0, mu_j=0.0, sigma_j=0.01)
        assert p.jump_vol_annual >= 0

    def test_jump_vol_annual_correct(self):
        p = MertonJumpParams(lam=4.0, mu_j=0.0, sigma_j=0.02)
        expected = np.sqrt(4.0 * 0.02**2)
        assert abs(p.jump_vol_annual - expected) < 1e-12

    def test_repr_contains_key_fields(self):
        p = MertonJumpParams(lam=3.0, mu_j=-0.008, sigma_j=0.015, n_r2_days=500,
                             bound_active=("lam",))
        s = repr(p)
        assert "λ=" in s
        assert "n_days=500" in s
        assert "bound_active=['lam']" in s


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RegimePDV.load
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePDVLoad:

    def test_missing_pdv_raises(self, tmp_path):
        model = RegimePDV(
            pdv_model_path=tmp_path / "missing.pkl",
            regime_labels_path=tmp_path / "labels.parquet",
        )
        with pytest.raises(FileNotFoundError, match="PDV model not found"):
            model.load()

    def test_missing_labels_raises(self, tmp_path):
        pdv_path = tmp_path / "pdv_model.pkl"
        with open(pdv_path, "wb") as f:
            pickle.dump(_FakePDVModel(), f)
        model = RegimePDV(
            pdv_model_path=pdv_path,
            regime_labels_path=tmp_path / "missing_labels.parquet",
        )
        with pytest.raises(FileNotFoundError, match="Regime labels not found"):
            model.load()

    def test_load_uses_the_returns_stored_in_the_pickle(self, tmp_path):
        r = _make_returns(400)
        PDVModel().fit(r).save(str(tmp_path / "pdv.pkl"))
        _make_regime_labels(r.index).to_parquet(tmp_path / "lbl.parquet")
        m = RegimePDV(tmp_path / "pdv.pkl", tmp_path / "lbl.parquet").load()
        pd.testing.assert_series_equal(m._log_returns, r.astype(float),
                                       check_names=False, check_freq=False)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RegimePDV.calibrate_jump
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePDVCalibrateJump:

    def test_no_r2_data_before_date_returns_zero_jump(self):
        model = _build_regime_pdv(n_days=600)
        model._regime_labels["regime"] = 1
        dates = model._regime_labels.index
        model._regime_labels.loc[dates[:2], "regime"] = 2
        params = model.calibrate_jump(dates[400].strftime("%Y-%m-%d"))
        assert params.lam == 0.0 and params.method == "none"

    def test_result_is_cached(self):
        model = _build_regime_pdv(n_days=600)
        as_of = model._regime_labels.index[450].strftime("%Y-%m-%d")
        assert model.calibrate_jump(as_of) is model.calibrate_jump(as_of)

    def test_zero_lookahead_r2_days_are_strictly_before_cutoff(self):
        """R2 labels only on/after the cutoff → nothing to calibrate on."""
        model = _build_regime_pdv(n_days=500)
        dates = model._regime_labels.index
        cutoff_idx = 400
        model._regime_labels["regime"] = 1
        model._regime_labels.iloc[cutoff_idx:, 0] = 2
        params = model.calibrate_jump(dates[cutoff_idx].strftime("%Y-%m-%d"))
        assert params.lam == 0.0

    def test_sufficient_r2_days_produces_nonzero_lambda(self):
        model = _build_regime_pdv(r2_fraction=0.40, n_days=600)
        params = model.calibrate_jump(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        assert params.lam > 0.0
        assert params.n_r2_days >= MIN_R2_DAYS

    def test_calibration_sample_is_days_after_an_r2_close(self):
        """Conditioning matches forecast(): the label at t-1 selects r_t."""
        model = _build_regime_pdv(n_days=600)
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        _, _, _, dates = model._get_r2_data(as_of)
        lbl = model._regime_labels["regime"].reindex(model._log_returns.index)
        prev = lbl.shift(1)
        assert (prev.loc[dates] == 2).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RegimePDV.calibrate_jump_tail — detected-jump estimator
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrateJumpTail:

    def _model(self):
        return _build_regime_pdv(r2_fraction=0.5, n_days=900, jump_prob=0.06,
                                 jump_mu=-0.05)

    def test_tail_result_is_threshold_estimate(self):
        model = self._model()
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        p = model.calibrate_jump_tail(as_of)
        assert isinstance(p, MertonJumpParams)
        assert p.method == "threshold"

    def test_lambda_is_detected_jump_frequency_per_r2_day(self):
        """λ refers to every post-R2 day, not only to the flagged ones."""
        model = self._model()
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        p = model.calibrate_jump_tail(as_of)
        _, _, _, dates = model._get_r2_data(as_of)
        flags = model.compute_jump_flags(as_of)["jump"].reindex(dates).fillna(False)
        assert p.lam == pytest.approx(ANNUALISE * flags.sum() / len(dates))
        assert p.n_r2_days == len(dates)

    def test_cache_key_includes_alpha(self):
        model = self._model()
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        model.calibrate_jump_tail(as_of, alpha=0.01)
        model.calibrate_jump_tail(as_of, alpha=0.05)
        assert len(model._jump_tail_cache) == 2

    def test_tail_cached_second_call(self):
        model = self._model()
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        assert model.calibrate_jump_tail(as_of) is model.calibrate_jump_tail(as_of)

    def test_fallback_to_mle_when_too_few_jumps_detected(self):
        model = _build_regime_pdv(r2_fraction=0.30, n_days=500, jump_prob=0.0)
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        p = model.calibrate_jump_tail(as_of, alpha=1e-4)
        assert p.method in ("mle", "none")
        assert np.isfinite(p.lam)

    def test_tail_zero_jump_fallback_when_no_r2_data(self):
        model = _build_regime_pdv(n_days=500)
        model._regime_labels["regime"] = 1
        p = model.calibrate_jump_tail(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        assert p.lam == 0.0

    def test_tail_zero_lookahead(self):
        """Adding R2 labels and huge returns after as_of must not change it."""
        model = self._model()
        dates = model._regime_labels.index
        cutoff = dates[700].strftime("%Y-%m-%d")
        before = model.calibrate_jump_tail(cutoff)

        r = model._log_returns.copy()
        r.iloc[701:] *= 10
        lbl = model._regime_labels.copy()
        lbl.iloc[701:, 0] = 2
        after = RegimePDV.from_data(r, lbl).calibrate_jump_tail(cutoff)
        assert (before.lam, before.mu_j, before.sigma_j) == (after.lam, after.mu_j, after.sigma_j)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RegimePDV.forecast
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePDVForecast:

    def test_r1_returns_zero_jump_adj(self):
        model = _build_regime_pdv(n_days=500)
        model._regime_labels.iloc[-1, 0] = 1
        result = model.forecast(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        assert result["regime"] == "SHORT_GAMMA"
        assert result["jump_adj"] == 0.0
        assert result["total_vol"] == result["pdv_vol"]

    def test_r0_returns_zero_jump_adj(self):
        model = _build_regime_pdv(n_days=500)
        model._regime_labels.iloc[-1, 0] = 0
        result = model.forecast(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        assert result["regime"] == "LONG_GAMMA"
        assert result["jump_adj"] == 0.0

    def test_r2_adds_the_calibrated_jump_variance(self):
        """R2 with λ > 0: total_vol² − pdv_vol² must equal λ(μ_j²+σ_j²) > 0."""
        model = _build_regime_pdv(r2_fraction=0.35, n_days=600)
        model._regime_labels.iloc[-1, 0] = 2
        result = model.forecast(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        assert result["regime"] == "VOMMA_ACTIVE"
        assert result["lambda"] > 0
        assert result["jump_adj"] > 0
        jv = result["lambda"] * (result["mu_j"] ** 2 + result["sigma_j"] ** 2)
        assert result["total_vol"] ** 2 - result["pdv_vol"] ** 2 == pytest.approx(jv, rel=1e-9)

    def test_total_vol_greater_or_equal_pdv_vol(self):
        model = _build_regime_pdv(r2_fraction=0.30, n_days=500)
        last = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        for regime_code in [0, 1, 2]:
            model._regime_labels.iloc[-1, 0] = regime_code
            result = model.forecast(last)
            assert result["total_vol"] >= result["pdv_vol"] - 1e-10

    def test_forecast_result_keys(self):
        model = _build_regime_pdv(n_days=400)
        result = model.forecast(model._regime_labels.index[-1].strftime("%Y-%m-%d"))
        required = {
            "pdv_vol", "jump_adj", "total_vol", "regime",
            "lambda", "mu_j", "sigma_j", "n_r2_cal_days", "as_of_date",
            "feature_date", "regime_date", "stale", "bound_active",
            "jump_method", "diffusion_scale",
        }
        assert required.issubset(set(result.keys()))

    def test_pdv_vol_is_positive(self):
        model = _build_regime_pdv(n_days=400)
        assert model.forecast(model._regime_labels.index[-1].strftime("%Y-%m-%d"))["pdv_vol"] > 0.0

    def test_stale_features_raise_unless_allowed(self):
        """Pre-C22 forecast('2026-09-23') silently used 2025-12-30 features."""
        model = _build_regime_pdv(n_days=400)
        last = model._log_returns.index[-1]
        later = (last + pd.offsets.BDay(10)).strftime("%Y-%m-%d")
        model._regime_labels.loc[pd.Timestamp(later), "regime"] = 1
        with pytest.raises(ValueError, match="business days old"):
            model.forecast(later)
        res = model.forecast(later, allow_stale=True)
        assert res["stale"] is True
        assert res["feature_date"] == str(last.date())
        assert res["regime_date"] == later

    def test_one_day_gap_is_reported_not_stale(self):
        model = _build_regime_pdv(n_days=400)
        last = model._log_returns.index[-1]
        nxt = (last + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        res = model.forecast(nxt)
        assert res["stale"] is False and res["feature_date"] == str(last.date())

    def test_pdv_vol_is_scaled_forecast_from_pre_as_of_fit(self):
        """σ_PDV = k · PDVLinear(features at as_of) with PDVLinear fit only on
        returns before as_of (here: re-derived by hand)."""
        from joint_vol_calibration.models.pdv import PDVLinear, build_xy, extract_pdv_features
        from scipy.stats import norm
        model = _build_regime_pdv(n_days=500)
        as_of = model._log_returns.index[450]
        r = model._log_returns[model._log_returns.index < as_of]
        f = extract_pdv_features(r)
        X, y = build_xy(r, f, horizon=1, target="abs_return")
        lin = PDVLinear().fit(X, y)
        k = np.median(y.values / lin.predict(X).values) / norm.ppf(0.75)
        p = lin.predict(model._features.loc[[as_of]]).iloc[0]
        res = model.forecast(as_of.strftime("%Y-%m-%d"))
        assert res["pdv_vol"] == pytest.approx(k * p, rel=1e-10)
        assert res["diffusion_scale"] == pytest.approx(k, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. compare_covid_2020 / compare over a date range
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def frame():
    model = _build_regime_pdv(r2_fraction=0.25, n_days=900, seed=7)
    dates = model._features.index
    df = model.compare_covid_2020(start=dates[600].strftime("%Y-%m-%d"),
                                  end=dates[760].strftime("%Y-%m-%d"))
    return model, df


class TestCompareDateRange:

    def test_has_required_columns(self, frame):
        _, df = frame
        for col in ("regime", "pdv_vol", "jump_adj", "total_vol", "rv_next_1d",
                    "rv_fwd_21d", "qlike_pdv", "qlike_jump", "se_pdv",
                    "se_jump", "improvement"):
            assert col in df.columns, f"Missing column: {col}"

    def test_r2_days_have_positive_jump_adj_and_others_zero(self, frame):
        _, df = frame
        r2 = df[df["regime"] == "VOMMA_ACTIVE"]
        assert len(r2) > 0 and (r2["jump_adj"] > 0).all()
        assert (df[df["regime"] != "VOMMA_ACTIVE"]["jump_adj"] == 0.0).all()

    def test_scored_against_the_next_days_return(self, frame):
        """Pre-C22 the 'actual' was the BACKWARD 20-day RV of the same day."""
        model, df = frame
        t = df.index[40]
        nxt = model._log_returns.shift(-1).loc[t]
        x = ANNUALISE * nxt ** 2
        row = df.loc[t]
        assert row["rv_next_1d"] == pytest.approx(abs(nxt) * np.sqrt(ANNUALISE))
        assert row["qlike_pdv"] == pytest.approx(x / row["pdv_vol"] ** 2 + np.log(row["pdv_vol"] ** 2))
        assert row["improvement"] == pytest.approx(row["qlike_pdv"] - row["qlike_jump"])

    def test_forecasts_do_not_use_future_returns(self, frame):
        model, df = frame
        end = df.index[-1]
        r = model._log_returns.copy()
        r[r.index > end] *= 10
        m2 = RegimePDV.from_data(r, model._regime_labels)
        df2 = m2.compare_covid_2020(start=str(df.index[0].date()), end=str(end.date()))
        pd.testing.assert_series_equal(df["total_vol"], df2["total_vol"])


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Zero look-ahead — explicit audit
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroLookAhead:

    def test_calibration_uses_no_future_r2_data(self):
        model = _build_regime_pdv(r2_fraction=0.30, n_days=600)
        dates = model._regime_labels.index
        cutoff_idx = 450
        model._regime_labels.iloc[cutoff_idx:, 0] = 1
        cutoff_date = dates[cutoff_idx].strftime("%Y-%m-%d")
        params_before = model.calibrate_jump(cutoff_date)
        model._regime_labels.iloc[cutoff_idx + 1:, 0] = 2
        model._jump_cache.clear()
        params_after = model.calibrate_jump(cutoff_date)
        assert (params_before.lam, params_before.mu_j, params_before.sigma_j) == \
               (params_after.lam, params_after.mu_j, params_after.sigma_j)

    def test_future_returns_cannot_move_the_calibration(self, tmp_path):
        """Poison: ×10 every return on/after as_of and rebuild the pickled
        PDVModel from the poisoned history. Pre-C22 the full-sample pickle
        coefficients moved and so did the 'zero look-ahead' calibration."""
        r = _make_returns(700, seed=3)
        lbl = _make_regime_labels(r.index, r2_fraction=0.4, seed=3)
        lbl.to_parquet(tmp_path / "lbl.parquet")
        as_of = r.index[550]
        poisoned = r.copy()
        poisoned[poisoned.index >= as_of] *= 10
        out = []
        for name, series in (("clean", r), ("poison", poisoned)):
            PDVModel().fit(series).save(str(tmp_path / f"{name}.pkl"))
            m = RegimePDV(tmp_path / f"{name}.pkl", tmp_path / "lbl.parquet").load()
            p = m.calibrate_jump(as_of.strftime("%Y-%m-%d"))
            out.append((p.lam, p.mu_j, p.sigma_j))
        assert out[0] == out[1]

    def test_diffusion_vol_for_r_t_ignores_r_t(self):
        """σ_d for the return of day t is the forecast made at t−1. Pre-C22 it
        used the features AT t, so a big r_t inflated its own diffusion vol."""
        model = _build_regime_pdv(r2_fraction=0.5, n_days=600)
        r = model._log_returns
        lbl = model._regime_labels.copy()
        as_of = r.index[-1].strftime("%Y-%m-%d")
        k = 500
        lbl.iloc[k - 1, 0] = 2            # day k is in the sample
        lbl.iloc[k, 0] = 2                # and so is day k+1
        base = RegimePDV.from_data(r, lbl)
        shocked_r = r.copy()
        shocked_r.iloc[k] = -0.09
        shocked = RegimePDV.from_data(shocked_r, lbl)
        # use the same diffusion fit for both (isolate the timing question)
        shocked._diffusion_cache[as_of] = dict(base._diffusion(as_of))
        shocked._diffusion_cache[as_of]["p"] = base._diffusion(as_of)["model"].predict(
            shocked._features[["sigma1", "sigma2", "lev"]])
        _, sd0, _, d0 = base._get_r2_data(as_of)
        _, sd1, _, d1 = shocked._get_r2_data(as_of)
        s0 = pd.Series(sd0, index=d0)
        s1 = pd.Series(sd1, index=d1)
        day_k, day_k1 = r.index[k], r.index[k + 1]
        assert s1.loc[day_k] == s0.loc[day_k]          # σ_d,k does not see r_k
        assert s1.loc[day_k1] > 1.5 * s0.loc[day_k1]   # σ_d,k+1 does


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Integration — runs only if saved artifacts are present
# ═══════════════════════════════════════════════════════════════════════════════

from joint_vol_calibration.signals.regime_pdv import PDV_MODEL_PATH, REGIME_LBL_PATH


@pytest.fixture(scope="module")
def integration_model():
    m = RegimePDV()
    m.load()
    return m


@pytest.mark.skipif(
    not (PDV_MODEL_PATH.exists() and REGIME_LBL_PATH.exists()),
    reason="Saved PDV model / regime labels not found — skipping integration test",
)
class TestIntegration:

    def test_load_succeeds(self, integration_model):
        assert integration_model.is_loaded
        assert integration_model._log_returns is not None
        assert integration_model._regime_labels is not None

    def test_forecast_2026_03_24(self, integration_model):
        result = integration_model.forecast("2026-03-24")
        assert result["feature_date"] == "2026-03-24"
        assert result["stale"] is False
        assert result["pdv_vol"] > 0.05
        assert result["regime"] in {"LONG_GAMMA", "SHORT_GAMMA", "VOMMA_ACTIVE"}
        assert result["total_vol"] >= result["pdv_vol"]

    def test_covid_crash_r2(self, integration_model):
        """2020-03-16 (VIX=82, VVIX=207) must be classified as VOMMA_ACTIVE."""
        result = integration_model.forecast("2020-03-16")
        assert result["regime"] == "VOMMA_ACTIVE"
        assert result["lambda"] > 0 and result["jump_adj"] > 0

    def test_jump_mle_is_interior(self, integration_model):
        """Pre-C22 both fits ended ON their λ bound (20 and 50/yr) and the
        old test restated the bounds. No parameter may sit within 1% of a
        bound now."""
        p = integration_model.calibrate_jump("2026-03-24")
        assert p.bound_active == ()
        for v, (lo, hi) in zip((p.lam, p.mu_j, p.sigma_j), _BOUNDS_FULL):
            assert lo + 0.01 * (hi - lo) < v < hi - 0.01 * (hi - lo)

    def test_forecast_beyond_the_data_is_refused(self, integration_model):
        last = integration_model._log_returns.index[-1]
        with pytest.raises(ValueError, match="business days old"):
            integration_model.forecast((last + pd.offsets.BDay(30)).strftime("%Y-%m-%d"))


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PDVLinear4F
# ═══════════════════════════════════════════════════════════════════════════════

from joint_vol_calibration.models.pdv import PDVLinear4F


def _make_4f_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Synthetic DataFrame with the 4 required features."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=n)
    return pd.DataFrame(
        {
            "sigma1": rng.uniform(0.08, 0.35, n),
            "sigma2": rng.uniform(0.10, 0.30, n),
            "lev":    rng.uniform(-0.04, 0.04, n),
            "vix":    rng.uniform(0.10, 0.45, n),
        },
        index=idx,
    )


class TestPDVLinear4F:

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            PDVLinear4F().predict(_make_4f_df())

    def test_fit_returns_self(self):
        m = PDVLinear4F()
        X = _make_4f_df()
        y = pd.Series(np.random.default_rng(1).uniform(0.1, 0.4, len(X)), index=X.index)
        assert m.fit(X, y) is m

    def test_predict_returns_series_with_correct_index(self):
        X = _make_4f_df()
        y = pd.Series(np.random.default_rng(2).uniform(0.1, 0.4, len(X)), index=X.index)
        pred = PDVLinear4F().fit(X, y).predict(X)
        assert isinstance(pred, pd.Series)
        assert list(pred.index) == list(X.index)

    def test_predict_clips_to_1e4_minimum(self):
        X = _make_4f_df(n=20)
        y = pd.Series(np.full(20, -1.0), index=X.index)
        assert (PDVLinear4F().fit(X, y).predict(X) >= 1e-4).all()

    def test_coef_has_4_elements(self):
        X = _make_4f_df()
        y = pd.Series(np.random.default_rng(3).uniform(0.1, 0.4, len(X)), index=X.index)
        m = PDVLinear4F().fit(X, y)
        assert m.coef_ is not None and len(m.coef_) == 4

    def test_perfect_linear_fit(self):
        X = _make_4f_df(n=200, seed=7)
        y = (0.5 * X["sigma1"] + 0.3 * X["sigma2"] + 0.1 * X["lev"]
             + 0.2 * X["vix"] + 0.01)
        pred = PDVLinear4F().fit(X, y).predict(X)
        np.testing.assert_allclose(pred.values, y.values, atol=1e-6)

    def test_repr_fitted(self):
        X = _make_4f_df()
        y = pd.Series(np.random.default_rng(4).uniform(0.1, 0.4, len(X)), index=X.index)
        r = repr(PDVLinear4F().fit(X, y))
        assert "PDVLinear4F" in r and "sigma1" in r and "vix" in r


# ═══════════════════════════════════════════════════════════════════════════════
# 12. daily_jump_test (replaces the single-pair "BN-S ratio")
# ═══════════════════════════════════════════════════════════════════════════════

class TestDailyJumpTest:

    def test_critical_value_is_deterministic_and_above_gaussian(self):
        c = jump_test_critical_value(20, 0.01)
        assert c == jump_test_critical_value(20, 0.01)
        assert 2.576 < c < 3.2       # estimated scale fattens the tails

    def test_size_under_iid_gaussian_null(self):
        """The old ratio flagged 26.7% of pure-diffusion days."""
        rng = np.random.default_rng(123)
        r = pd.Series(rng.normal(0, 0.01, 100_000))
        rate = daily_jump_test(r, alpha=0.01)["jump"].mean()
        assert 0.008 < rate < 0.012

    def test_size_under_garch_null_within_twice_alpha(self):
        rng = np.random.default_rng(321)
        n = 50_000
        s2 = np.empty(n)
        x = np.empty(n)
        s2[0] = 1e-4
        for t in range(n):
            if t:
                s2[t] = 2e-6 + 0.09 * x[t - 1] ** 2 + 0.89 * s2[t - 1]
            x[t] = rng.normal(0, np.sqrt(s2[t]))
        rate = daily_jump_test(pd.Series(x), alpha=0.01)["jump"].mean()
        assert 0.005 < rate < 0.02

    def test_eight_sigma_return_is_flagged(self):
        rng = np.random.default_rng(9)
        r = pd.Series(rng.normal(0, 0.01, 300))
        r.iloc[200] = -0.08
        out = daily_jump_test(r)
        assert bool(out["jump"].iloc[200])
        assert out["stat"].iloc[200] < -out["crit"].iloc[200]

    def test_statistic_is_causal(self):
        """L_t uses r_t and returns strictly before t only."""
        rng = np.random.default_rng(10)
        r = pd.Series(rng.normal(0, 0.01, 300))
        a = daily_jump_test(r)
        r2 = r.copy()
        r2.iloc[151:] *= 5
        b = daily_jump_test(r2)
        pd.testing.assert_frame_equal(a.iloc[:151], b.iloc[:151])
        r3 = r.copy()
        r3.iloc[150] *= 5           # r_t itself changes only the numerator
        c = daily_jump_test(r3)
        assert c["sigma_local"].iloc[150] == a["sigma_local"].iloc[150]

    def test_flags_strictly_before_as_of(self):
        model = _build_regime_pdv(n_days=300)
        as_of = model._regime_labels.index[150]
        flags = model.compute_jump_flags(as_of.strftime("%Y-%m-%d"))
        assert (flags.index < as_of).all()

    def test_requires_loaded(self):
        model = RegimePDV(
            pdv_model_path=Path("/nonexistent/pdv.pkl"),
            regime_labels_path=Path("/nonexistent/lbl.parquet"),
        )
        with pytest.raises(RuntimeError, match="load"):
            model.compute_jump_flags("2022-01-01")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. calibrate_jump_tail — test level
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrateJumpTailAlpha:

    def test_smaller_alpha_flags_fewer_days(self):
        model = _build_regime_pdv(r2_fraction=0.5, n_days=900, jump_prob=0.06)
        as_of = model._regime_labels.index[-1].strftime("%Y-%m-%d")
        loose = model.compute_jump_flags(as_of, alpha=0.05)["jump"].sum()
        strict = model.compute_jump_flags(as_of, alpha=0.001)["jump"].sum()
        assert strict < loose
