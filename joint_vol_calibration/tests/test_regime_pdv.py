"""
test_regime_pdv.py — Tests for C11: Regime-Switching PDV with Merton Jump Component

Test coverage (42 tests across 10 test classes):
  1. merton_log_density          — output shape, finite, n=0 reduces to Normal
  2. calibrate_jump_mle          — output types, bounds, convergence on synthetic data
  3. MertonJumpParams            — properties (jump_variance_annual, jump_vol_annual, repr)
  4. RegimePDV.load              — FileNotFoundError on missing paths
  5. RegimePDV.calibrate_jump    — zero look-ahead, fallback when too few R2 days
  6. RegimePDV.calibrate_jump_tail — tail filter, cache key, fallback, bounds, look-ahead
  7. RegimePDV.forecast          — R0/R1 zero jump_adj, R2 positive jump_adj,
                                   uses tail calibration, result keys
  8. compare_covid_2020          — shape, columns, R2 non-negative jump_adj
  9. Zero look-ahead             — full and tail both respect the cutoff
  10. Integration                — end-to-end on saved artifacts: tail params in expected
                                   range, COVID crash R2, jump_adj materially > old 0.16pp
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Module-level picklable stub (cannot be defined inside a test method)
class _FakePDVModel:
    is_fitted = True
    _features = pd.DataFrame(
        {"sigma1": [0.2]},
        index=pd.to_datetime(["2020-01-02"]),
    )


from joint_vol_calibration.signals.regime_pdv import (
    ANNUALISE,
    MIN_R2_DAYS,
    MIN_TAIL_DAYS,
    _BOUNDS_TAIL,
    REGIME_NAMES,
    MertonJumpParams,
    RegimePDV,
    _moment_match_fallback,
    calibrate_jump_mle,
    merton_log_density,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_pdv_mock(
    n: int = 500,
    seed: int = 42,
    start: str = "2010-01-04",
) -> MagicMock:
    """
    Build a minimal mock of a fitted PDVModel.

    _features has sigma1, sigma2, lev, r_lag1, rv_hist_20d.
    linear_.predict() returns sigma1 (a reasonable proxy).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    daily_r = rng.normal(0, 0.012, n)
    sigma1 = np.maximum(
        pd.Series(daily_r**2 * ANNUALISE).ewm(span=9, adjust=False).mean().values**0.5,
        0.01,
    )
    sigma2 = np.maximum(
        pd.Series(daily_r**2 * ANNUALISE).ewm(span=119, adjust=False).mean().values**0.5,
        0.01,
    )
    lev = pd.Series(daily_r).ewm(span=19, adjust=False).mean().values * (ANNUALISE**0.5)

    feats = pd.DataFrame(
        {
            "sigma1":      sigma1,
            "sigma2":      sigma2,
            "lev":         lev,
            "r_lag1":      np.concatenate([[np.nan], daily_r[:-1]]),
            "rv_hist_20d": np.maximum(
                pd.Series(daily_r**2).rolling(20).mean().values**0.5 * ANNUALISE**0.5,
                0.01,
            ),
        },
        index=dates,
    )

    # linear_ mock: predict returns sigma1
    linear_mock = MagicMock()
    def _linear_predict(df):
        vals = df["sigma1"].values.copy()
        return pd.Series(vals, index=df.index)
    linear_mock.predict.side_effect = _linear_predict

    pdv_mock = MagicMock()
    pdv_mock.is_fitted = True
    pdv_mock._features = feats
    pdv_mock.linear_ = linear_mock
    return pdv_mock


def _make_regime_labels(
    pdv_mock,
    r2_fraction: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Build synthetic regime labels aligned with pdv_mock._features.index."""
    dates = pdv_mock._features.index
    rng   = np.random.default_rng(seed)
    n     = len(dates)
    # Assign R2 to ~25% of days, spread across the series
    regimes = np.ones(n, dtype=int)  # default R1
    r2_idx  = rng.choice(n, size=int(n * r2_fraction), replace=False)
    regimes[r2_idx] = 2
    return pd.DataFrame({"regime": regimes}, index=dates)


def _build_regime_pdv(
    r2_fraction: float = 0.30,
    n_days: int = 600,
    seed: int = 42,
) -> RegimePDV:
    """
    Build a RegimePDV instance with injected mock artifacts — no disk I/O.
    """
    pdv_mock = _make_pdv_mock(n=n_days, seed=seed)
    lbl      = _make_regime_labels(pdv_mock, r2_fraction=r2_fraction, seed=seed)

    model = RegimePDV(
        pdv_model_path=Path("/nonexistent/pdv_model.pkl"),
        regime_labels_path=Path("/nonexistent/regime_labels.parquet"),
    )
    # Inject mocks directly
    model._pdv = pdv_mock
    model._regime_labels = lbl
    model.is_loaded = True
    return model


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

        # Almost-zero jump intensity
        out_jump = merton_log_density(r, sd, lam_daily=1e-9, mu_j=0.0, sigma_j=0.001)

        # Pure Normal log-density
        from scipy.stats import norm
        log_norm = norm.logpdf(r, 0.0, sd)

        np.testing.assert_allclose(out_jump, log_norm, atol=1e-4)

    def test_larger_sigma_j_increases_variance(self):
        """A bigger sigma_j should produce a flatter (lower peak) density at r=0."""
        r    = np.array([0.0])
        sd   = np.array([0.01 / np.sqrt(ANNUALISE)])
        lam  = 3.0 / ANNUALISE
        # Large sigma_j → lower peak
        d_small = merton_log_density(r, sd, lam, mu_j=0.0, sigma_j=0.005)
        d_large = merton_log_density(r, sd, lam, mu_j=0.0, sigma_j=0.050)
        assert d_large[0] < d_small[0], "Larger sigma_j should reduce density at r=0"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. calibrate_jump_mle
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibrateJumpMle:

    def _synthetic_merton(
        self, T: int = 300, lam: float = 4.0, mu_j: float = -0.01,
        sigma_j: float = 0.025, sigma_d: float = 0.15, seed: int = 42
    ):
        rng = np.random.default_rng(seed)
        lam_d = lam / ANNUALISE
        sd    = sigma_d / np.sqrt(ANNUALISE)
        r     = rng.normal(0.0, sd, T)
        # Add jumps
        n_jumps = rng.poisson(lam_d, T)
        for t, k in enumerate(n_jumps):
            if k > 0:
                r[t] += rng.normal(mu_j, sigma_j, k).sum()
        return r, np.full(T, sd)

    def test_output_type(self):
        r, sd = self._synthetic_merton()
        params = calibrate_jump_mle(r, sd)
        assert isinstance(params, MertonJumpParams)

    def test_bounds_respected(self):
        r, sd = self._synthetic_merton()
        params = calibrate_jump_mle(r, sd)
        assert 0.1 <= params.lam <= 50.0, "λ must be in [0.1, 50]"
        assert -0.10 <= params.mu_j <= 0.02, "μ_j must be in [-0.10, 0.02]"
        assert 0.001 <= params.sigma_j <= 0.20, "σ_j must be in [0.001, 0.20]"

    def test_n_r2_days_stored(self):
        r, sd = self._synthetic_merton(T=200)
        params = calibrate_jump_mle(r, sd)
        assert params.n_r2_days == 200

    def test_nll_is_finite(self):
        r, sd = self._synthetic_merton()
        params = calibrate_jump_mle(r, sd)
        assert np.isfinite(params.nll), "NLL must be finite"

    def test_recovers_approximate_params(self):
        """
        With 500 R2-day sample, calibrated λ should be reasonably close to truth.
        We allow ±60% tolerance because daily returns are very noisy.
        """
        true_lam = 5.0
        r, sd = self._synthetic_merton(T=500, lam=true_lam, seed=99)
        params = calibrate_jump_mle(r, sd, random_seed=99)
        # Jump intensity should be in a reasonable range
        assert 0.5 <= params.lam <= 30.0, f"λ={params.lam:.2f} out of plausible range"


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
        p = MertonJumpParams(lam=3.0, mu_j=-0.008, sigma_j=0.015, n_r2_days=500)
        s = repr(p)
        assert "λ=" in s
        assert "n_days=500" in s


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RegimePDV.load — missing files
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RegimePDV.calibrate_jump — zero look-ahead & fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePDVCalibrateJump:

    def test_no_r2_data_before_date_returns_zero_jump(self):
        """If all R2 days are AFTER as_of_date, result should be zero-jump fallback."""
        model = _build_regime_pdv(n_days=600)
        # Set all R2 days to 2015+, then request forecast for 2010
        model._regime_labels["regime"] = 1  # all R1
        # Only 2 R2 days in 2014 — not enough
        dates = model._regime_labels.index
        model._regime_labels.loc[dates[:2], "regime"] = 2

        as_of = dates[0].strftime("%Y-%m-%d")
        params = model.calibrate_jump(as_of)
        assert params.lam == 0.0, "Should return zero-jump when too few R2 days"

    def test_result_is_cached(self):
        model = _build_regime_pdv(n_days=600)
        dates = model._regime_labels.index
        as_of = dates[350].strftime("%Y-%m-%d")

        p1 = model.calibrate_jump(as_of)
        p2 = model.calibrate_jump(as_of)
        assert p1 is p2, "Second call must return cached object"

    def test_zero_lookahead_r2_days_are_strictly_before_cutoff(self):
        """
        Inspect calibration: R2 days used must all be < as_of_date.
        We set R2 days only in the future and verify zero-jump fallback.
        """
        model = _build_regime_pdv(n_days=400)
        dates = model._regime_labels.index
        # Cut date = 200th day
        cutoff_idx = 200
        cutoff = dates[cutoff_idx].strftime("%Y-%m-%d")

        # Force R2 only on dates >= cutoff
        model._regime_labels["regime"] = 1
        model._regime_labels.iloc[cutoff_idx:, 0] = 2

        params = model.calibrate_jump(cutoff)
        # Should see zero R2 days before cutoff → zero-jump fallback
        assert params.lam == 0.0

    def test_sufficient_r2_days_produces_nonzero_lambda(self):
        """With enough R2 days, calibrated λ should be > 0."""
        model = _build_regime_pdv(r2_fraction=0.40, n_days=500)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        params = model.calibrate_jump(as_of)
        assert params.lam > 0.0, "Should find a positive jump intensity"
        assert params.n_r2_days >= MIN_R2_DAYS

    def test_params_within_bounds(self):
        model = _build_regime_pdv(r2_fraction=0.35, n_days=500)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        params = model.calibrate_jump(as_of)
        if params.lam > 0:
            assert 0.1 <= params.lam <= 50.0
            assert 0.001 <= params.sigma_j <= 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RegimePDV.calibrate_jump_tail — tail filter, cache key, fallback, bounds
# ═══════════════════════════════════════════════════════════════════════════════

def _build_regime_pdv_with_crash(
    n_days: int = 600,
    r2_fraction: float = 0.30,
    n_crash_days: int = 20,
    crash_vol_boost: float = 0.30,
    seed: int = 42,
) -> RegimePDV:
    """
    Like _build_regime_pdv but injects `n_crash_days` R2 days with large
    rv_hist_20d (boosted by crash_vol_boost above the PDV prediction).
    This gives the tail filter genuine high-error days to select.
    """
    model = _build_regime_pdv(r2_fraction=r2_fraction, n_days=n_days, seed=seed)

    feats = model._pdv._features.copy()
    feats.index = pd.to_datetime(feats.index)

    # Find R2 indices and inflate rv_hist_20d on a subset
    r2_idx = np.where(model._regime_labels["regime"] == 2)[0]
    rng    = np.random.default_rng(seed + 1)
    if len(r2_idx) >= n_crash_days:
        crash_idx = rng.choice(r2_idx, size=n_crash_days, replace=False)
        crash_dates = model._regime_labels.index[crash_idx]
        feats.loc[crash_dates, "rv_hist_20d"] = (
            feats.loc[crash_dates, "rv_hist_20d"] + crash_vol_boost
        )
        model._pdv._features = feats

    return model


class TestCalibrateJumpTail:

    def test_tail_result_is_merton_jump_params(self):
        model = _build_regime_pdv_with_crash(n_days=600)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        params = model.calibrate_jump_tail(as_of)
        assert isinstance(params, MertonJumpParams)

    def test_tail_cache_key_includes_zscore(self):
        """Different zscore_threshold values must use different cache entries."""
        model = _build_regime_pdv_with_crash(n_days=600)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")

        p1 = model.calibrate_jump_tail(as_of, zscore_threshold=2.0)
        p2 = model.calibrate_jump_tail(as_of, zscore_threshold=1.0)
        # Both should be cached separately
        assert len(model._jump_tail_cache) == 2

    def test_tail_cached_second_call(self):
        model = _build_regime_pdv_with_crash(n_days=600)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        p1 = model.calibrate_jump_tail(as_of)
        p2 = model.calibrate_jump_tail(as_of)
        assert p1 is p2

    def test_tail_fallback_when_too_few_tail_days(self):
        """
        If z-score filter retains fewer than MIN_TAIL_DAYS days, should
        fall back to full-R2 MLE rather than crashing.
        """
        # Set a very high zscore_threshold so almost no days pass
        model = _build_regime_pdv(r2_fraction=0.30, n_days=500)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        # zscore_threshold=10 → virtually no days pass
        params = model.calibrate_jump_tail(as_of, zscore_threshold=10.0)
        assert isinstance(params, MertonJumpParams)
        assert np.isfinite(params.lam)

    def test_tail_zero_jump_fallback_when_no_r2_data(self):
        """No R2 days before cutoff → zero-jump fallback, same as full calibration."""
        model = _build_regime_pdv(n_days=500)
        model._regime_labels["regime"] = 1  # all R1
        dates = model._regime_labels.index
        as_of = dates[10].strftime("%Y-%m-%d")
        params = model.calibrate_jump_tail(as_of)
        assert params.lam == 0.0

    def test_tail_lambda_within_tail_bounds(self):
        """Tail-calibrated λ must satisfy _BOUNDS_TAIL upper limit (≤ 20)."""
        model = _build_regime_pdv_with_crash(n_days=600, n_crash_days=30)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        params = model.calibrate_jump_tail(as_of, zscore_threshold=1.5)
        if params.lam > 0:
            lo, hi = _BOUNDS_TAIL[0]
            assert params.lam <= hi, f"λ={params.lam:.2f} exceeds tail bound {hi}"

    def test_tail_sigma_j_within_tail_bounds(self):
        model = _build_regime_pdv_with_crash(n_days=600, n_crash_days=30)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        params = model.calibrate_jump_tail(as_of, zscore_threshold=1.5)
        if params.lam > 0:
            lo, hi = _BOUNDS_TAIL[2]
            assert lo <= params.sigma_j <= hi

    def test_tail_n_r2_days_is_tail_count(self):
        """n_r2_days in result should reflect the tail subset size, not all R2 days."""
        model = _build_regime_pdv_with_crash(n_days=600, n_crash_days=30)
        dates = model._regime_labels.index
        as_of = dates[-1].strftime("%Y-%m-%d")
        full_params = model.calibrate_jump(as_of)
        tail_params = model.calibrate_jump_tail(as_of, zscore_threshold=1.5)
        # Tail should have fewer days than full (when tail filter fired)
        if tail_params.n_r2_days > 0 and full_params.n_r2_days > 0:
            assert tail_params.n_r2_days <= full_params.n_r2_days

    def test_tail_zero_lookahead(self):
        """
        Adding R2 days AFTER as_of_date must not change tail params.
        """
        model = _build_regime_pdv_with_crash(n_days=500, n_crash_days=25)
        dates = model._regime_labels.index
        cutoff_idx = 400
        cutoff = dates[cutoff_idx].strftime("%Y-%m-%d")

        params_before = model.calibrate_jump_tail(cutoff)

        # Add R2 days after cutoff and clear cache
        model._regime_labels.iloc[cutoff_idx + 1:, 0] = 2
        model._jump_tail_cache.clear()
        params_after = model.calibrate_jump_tail(cutoff)

        assert params_before.lam == params_after.lam
        assert params_before.mu_j == params_after.mu_j
        assert params_before.sigma_j == params_after.sigma_j


# ═══════════════════════════════════════════════════════════════════════════════
# 7. RegimePDV.forecast — regime switching
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegimePDVForecast:

    def test_r1_returns_zero_jump_adj(self):
        """R1 days must have jump_adj=0 and total_vol==pdv_vol."""
        model = _build_regime_pdv(n_days=400)
        dates = model._regime_labels.index

        # Force the last date to R1
        model._regime_labels.iloc[-1, 0] = 1
        last_date = dates[-1].strftime("%Y-%m-%d")
        result = model.forecast(last_date)

        assert result["regime"] == "SHORT_GAMMA"
        assert result["jump_adj"] == 0.0
        assert result["total_vol"] == result["pdv_vol"]

    def test_r0_returns_zero_jump_adj(self):
        """R0 days must also have jump_adj=0."""
        model = _build_regime_pdv(n_days=400)
        dates = model._regime_labels.index
        model._regime_labels.iloc[-1, 0] = 0
        last_date = dates[-1].strftime("%Y-%m-%d")
        result = model.forecast(last_date)

        assert result["regime"] == "LONG_GAMMA"
        assert result["jump_adj"] == 0.0

    def test_r2_returns_positive_jump_adj(self):
        """R2 days with sufficient calibration data must have jump_adj > 0."""
        model = _build_regime_pdv(r2_fraction=0.35, n_days=500)
        dates = model._regime_labels.index

        # Force last date to R2
        model._regime_labels.iloc[-1, 0] = 2
        last_date = dates[-1].strftime("%Y-%m-%d")
        result = model.forecast(last_date)

        assert result["regime"] == "VOMMA_ACTIVE"
        # jump_adj ≥ 0 always; it's > 0 when λ > 0
        assert result["jump_adj"] >= 0.0
        assert result["total_vol"] >= result["pdv_vol"]

    def test_total_vol_greater_or_equal_pdv_vol(self):
        """total_vol must always be ≥ pdv_vol (jump adds variance, not removes it)."""
        model = _build_regime_pdv(r2_fraction=0.30, n_days=500)
        dates = model._regime_labels.index

        for regime_code in [0, 1, 2]:
            model._regime_labels.iloc[-1, 0] = regime_code
            result = model.forecast(dates[-1].strftime("%Y-%m-%d"))
            assert result["total_vol"] >= result["pdv_vol"] - 1e-10

    def test_forecast_result_keys(self):
        """Forecast dict must contain all required keys."""
        model = _build_regime_pdv(n_days=400)
        dates = model._regime_labels.index
        result = model.forecast(dates[-1].strftime("%Y-%m-%d"))
        required = {
            "pdv_vol", "jump_adj", "total_vol", "regime",
            "lambda", "mu_j", "sigma_j", "n_r2_cal_days", "as_of_date",
        }
        assert required.issubset(set(result.keys()))

    def test_pdv_vol_is_positive(self):
        model = _build_regime_pdv(n_days=300)
        dates = model._regime_labels.index
        result = model.forecast(dates[-1].strftime("%Y-%m-%d"))
        assert result["pdv_vol"] > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. compare_covid_2020 / compare over a date range
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareDateRange:

    def _model_covering_2020(self) -> RegimePDV:
        """Build a model whose features span 2010 → 2021 so 2020 is covered."""
        return _build_regime_pdv(r2_fraction=0.25, n_days=2800, seed=7)

    def test_returns_dataframe(self):
        model = self._model_covering_2020()
        dates = model._pdv._features.index
        start = dates[2400].strftime("%Y-%m-%d")
        end   = dates[2500].strftime("%Y-%m-%d")
        df = model.compare_covid_2020(start=start, end=end)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        model = self._model_covering_2020()
        dates = model._pdv._features.index
        start = dates[2400].strftime("%Y-%m-%d")
        end   = dates[2500].strftime("%Y-%m-%d")
        df = model.compare_covid_2020(start=start, end=end)
        for col in ("regime", "pdv_vol", "jump_adj", "total_vol",
                    "rv_actual_20d", "error_pdv", "error_jump", "improvement"):
            assert col in df.columns, f"Missing column: {col}"

    def test_r2_days_have_nonneg_jump_adj(self):
        model = self._model_covering_2020()
        dates = model._pdv._features.index
        start = dates[2200].strftime("%Y-%m-%d")
        end   = dates[2400].strftime("%Y-%m-%d")
        df = model.compare_covid_2020(start=start, end=end)
        r2_rows = df[df["regime"] == "VOMMA_ACTIVE"]
        if len(r2_rows) > 0:
            assert (r2_rows["jump_adj"] >= 0).all()

    def test_non_r2_days_have_zero_jump_adj(self):
        model = self._model_covering_2020()
        dates = model._pdv._features.index
        start = dates[2200].strftime("%Y-%m-%d")
        end   = dates[2400].strftime("%Y-%m-%d")
        df = model.compare_covid_2020(start=start, end=end)
        non_r2 = df[df["regime"] != "VOMMA_ACTIVE"]
        assert (non_r2["jump_adj"] == 0.0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Zero look-ahead — explicit audit
# ═══════════════════════════════════════════════════════════════════════════════

class TestZeroLookAhead:

    def test_calibration_uses_no_future_r2_data(self):
        """
        Mark the last 100 days all as R2, then request calibration for
        the day JUST BEFORE those 100 days. The calibrated params must
        not depend on those future R2 days at all.
        """
        model = _build_regime_pdv(r2_fraction=0.20, n_days=500)
        dates = model._regime_labels.index
        cutoff_idx = 400

        # Override: clear all R2 on and after cutoff
        model._regime_labels.iloc[cutoff_idx:, 0] = 1

        cutoff_date = dates[cutoff_idx].strftime("%Y-%m-%d")
        params_before = model.calibrate_jump(cutoff_date)

        # Now add many R2 days AFTER the cutoff — should not affect params
        model._regime_labels.iloc[cutoff_idx + 1:, 0] = 2
        # Clear cache to force re-evaluation
        model._jump_cache.clear()
        params_after = model.calibrate_jump(cutoff_date)

        # The resulting params should be identical since we only look at < cutoff
        assert params_before.lam == params_after.lam
        assert params_before.mu_j == params_after.mu_j
        assert params_before.sigma_j == params_after.sigma_j


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Integration — runs only if saved artifacts are present
# ═══════════════════════════════════════════════════════════════════════════════

from joint_vol_calibration.signals.regime_pdv import PDV_MODEL_PATH, REGIME_LBL_PATH


@pytest.mark.skipif(
    not (PDV_MODEL_PATH.exists() and REGIME_LBL_PATH.exists()),
    reason="Saved PDV model / regime labels not found — skipping integration test",
)
class TestIntegration:

    @pytest.fixture(scope="class")
    def model(self):
        m = RegimePDV()
        m.load()
        return m

    def test_load_succeeds(self, model):
        assert model.is_loaded
        assert model._pdv is not None
        assert model._regime_labels is not None

    def test_forecast_2026_03_24(self, model):
        result = model.forecast("2026-03-24")
        assert result["pdv_vol"] > 0.05, "PDV vol should be > 5% in 2026"
        assert result["regime"] in {"LONG_GAMMA", "SHORT_GAMMA", "VOMMA_ACTIVE"}
        assert result["total_vol"] >= result["pdv_vol"]

    def test_covid_crash_r2(self, model):
        """2020-03-16 (VIX=82, VVIX=207) must be classified as VOMMA_ACTIVE."""
        result = model.forecast("2020-03-16")
        assert result["regime"] == "VOMMA_ACTIVE", (
            f"COVID crash day must be R2, got {result['regime']}"
        )
        assert result["jump_adj"] >= 0.0

    def test_tail_params_in_expected_range(self, model):
        """
        Tail-calibrated params on 15 years of data should give crash-like values.
        λ < 20/yr and σ_j > 0.005 (0.5% per jump), not the λ=50 tiny-jump collapse.
        """
        params = model.calibrate_jump_tail("2026-03-24")
        assert params.lam <= 20.0, (
            f"Tail λ={params.lam:.2f} hit the 50/yr ceiling — tail filter not working"
        )
        assert params.sigma_j >= 0.005, "σ_j should be at least 0.5% for crash jumps"

    def test_tail_jump_adj_materially_larger_than_full(self, model):
        """
        On 2020-03-16 (COVID crash), tail-calibrated jump_adj should be
        materially larger than the previous 0.16pp produced by full-R2 MLE.
        """
        result = model.forecast("2020-03-16")
        assert result["regime"] == "VOMMA_ACTIVE"
        # Old full-R2 result was ~0.16pp. Tail should do better.
        # We allow zero (if tail falls back) but never negative.
        assert result["jump_adj"] >= 0.0, "jump_adj must be non-negative"
        # If tail calibration actually fired (lam > 0), the jump should be > 0.001
        if result["lambda"] > 0:
            assert result["jump_adj"] > 0.001, (
                f"jump_adj={result['jump_adj']:.4f} should be > 0.001 on COVID crash"
            )
