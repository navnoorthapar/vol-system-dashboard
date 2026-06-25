"""
test_pdv.py — Tests for C3: Path-Dependent Volatility model (Guyon-Lekeufack 2023)

The PDV model produces the headline R² = 0.31 walk-forward result and feeds
every downstream trading signal, yet it had NO dedicated unit tests — it was
only exercised incidentally through regime_pdv / signal_engine / delta_hedger.
This file locks down the core behaviour and, crucially, converts the project's
central credibility claim — *zero look-ahead bias* — from a docstring comment
into an executable proof.

Test coverage (8 classes):
  1. extract_pdv_features  — columns, positivity, NO-LOOK-AHEAD on features
  2. build_xy              — target is genuinely next-day vol; no-look-ahead
  3. PDVLinear             — fit/predict, positivity, exact OLS recovery
  4. PDVKernel             — Nadaraya-Watson weights sum to 1, convex-combo bound
  5. GARCH11               — valid persistence, positive vol forecast
  6. compute_metrics       — R²=1 on perfect fit, R²≈0 for mean predictor
  7. walk_forward_predict  — burn-in NaNs, predictions finite
  8. NO-LOOK-AHEAD (poison)— future targets cannot change past predictions
"""

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.models.pdv import (
    extract_pdv_features,
    build_xy,
    PDVLinear,
    PDVKernel,
    GARCH11,
    walk_forward_predict,
    compute_metrics,
    forecast_skill_comparison,
)

ANNUALISE = 252


# ── Shared synthetic data (fixed seed, mild vol clustering) ───────────────────

def _make_returns(n: int = 800, seed: int = 7) -> pd.Series:
    """Synthetic daily log returns with GARCH-like vol clustering."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=n)
    # Simple vol-clustering process so features + GARCH have signal to fit
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = 1e-4
    r[0] = rng.normal(0.0, np.sqrt(sigma2[0]))
    for t in range(1, n):
        sigma2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * sigma2[t - 1]
        r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
    return pd.Series(r, index=dates, name="log_return")


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return _make_returns()


@pytest.fixture(scope="module")
def xy(returns):
    feats = extract_pdv_features(returns)
    X, y = build_xy(returns, feats, horizon=1)
    return X, y


# ── 1. extract_pdv_features ───────────────────────────────────────────────────

class TestExtractFeatures:
    def test_expected_columns_present(self, returns):
        feats = extract_pdv_features(returns)
        for col in ("f1", "f2", "sigma1", "sigma2", "lev", "ts_slope",
                    "rv_hist_5d", "rv_hist_20d"):
            assert col in feats.columns

    def test_variance_features_nonnegative(self, returns):
        feats = extract_pdv_features(returns)
        # f1, f2 are EMAs of squared returns → strictly non-negative
        assert (feats["f1"].dropna() >= 0).all()
        assert (feats["f2"].dropna() >= 0).all()
        assert (feats["sigma1"].dropna() >= 0).all()
        assert (feats["sigma2"].dropna() >= 0).all()

    def test_no_lookahead_on_features(self, returns):
        """Feature row t must be byte-identical whether or not future returns
        exist in the input. This is the property the docstring claims."""
        cut = 500
        feats_full = extract_pdv_features(returns)
        feats_trunc = extract_pdv_features(returns.iloc[:cut])
        # Compare the overlap region [0, cut) on the path-dependent columns
        cols = ["f1", "f2", "sigma1", "sigma2", "lev", "ts_slope"]
        a = feats_full[cols].iloc[:cut]
        b = feats_trunc[cols].iloc[:cut]
        pd.testing.assert_frame_equal(a, b)


# ── 2. build_xy ───────────────────────────────────────────────────────────────

class TestBuildXY:
    def test_target_is_next_day_abs_return(self, returns):
        """y(t) must equal |r(t+1)| * sqrt(252) — the next day's realised vol."""
        feats = extract_pdv_features(returns)
        X, y = build_xy(returns, feats, horizon=1)
        # Pick an interior date present in y, verify against the raw next return
        t_date = y.index[100]
        pos = returns.index.get_loc(t_date)
        expected = abs(returns.iloc[pos + 1]) * np.sqrt(ANNUALISE)
        assert y.loc[t_date] == pytest.approx(expected, rel=1e-9)

    def test_no_nans_in_output(self, xy):
        X, y = xy
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_x_excludes_label_leak_column(self, xy):
        X, _ = xy
        # abs_r_ann is |r_t| itself — must be dropped to avoid trivial leakage
        assert "abs_r_ann" not in X.columns


# ── 3. PDVLinear ──────────────────────────────────────────────────────────────

class TestPDVLinear:
    def test_fit_predict_shapes(self, xy):
        X, y = xy
        m = PDVLinear().fit(X, y)
        preds = m.predict(X)
        assert len(preds) == len(X)
        assert (preds.index == X.index).all()

    def test_predictions_nonnegative(self, xy):
        X, y = xy
        m = PDVLinear().fit(X, y)
        assert (m.predict(X) >= 0).all()

    def test_recovers_known_linear_relationship(self):
        """With a noise-free linear target, OLS must recover the coefficients."""
        rng = np.random.default_rng(0)
        n = 300
        idx = pd.bdate_range("2018-01-01", periods=n)
        X = pd.DataFrame({
            "sigma1": np.abs(rng.normal(0.2, 0.05, n)),
            "sigma2": np.abs(rng.normal(0.18, 0.04, n)),
            "lev":    rng.normal(0.0, 0.02, n),
        }, index=idx)
        y = (0.40 * X["sigma1"] + 0.30 * X["sigma2"]
             - 0.50 * X["lev"] + 0.02)
        m = PDVLinear().fit(X, y)
        a, b, c = m.coef_
        assert a == pytest.approx(0.40, abs=1e-6)
        assert b == pytest.approx(0.30, abs=1e-6)
        assert c == pytest.approx(-0.50, abs=1e-6)
        assert m.intercept_ == pytest.approx(0.02, abs=1e-6)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            PDVLinear().predict(pd.DataFrame({"sigma1": [0.2], "sigma2": [0.2],
                                              "lev": [0.0]}))


# ── 4. PDVKernel (Nadaraya-Watson) ────────────────────────────────────────────

class TestPDVKernel:
    def test_weights_sum_to_one(self, xy):
        X, y = xy
        m = PDVKernel().fit(X.iloc[:200], y.iloc[:200])
        w = m._gaussian_kernel_weights(m._X_train[0])
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= 0).all()

    def test_prediction_is_convex_combination(self, xy):
        """NW prediction is a weighted average of training targets, so it must
        lie within [min(y_train), max(y_train)]."""
        X, y = xy
        ytr = y.iloc[:200]
        m = PDVKernel().fit(X.iloc[:200], ytr)
        preds = m.predict(X.iloc[200:260])
        assert preds.min() >= ytr.min() - 1e-9
        assert preds.max() <= ytr.max() + 1e-9


# ── 5. GARCH11 ────────────────────────────────────────────────────────────────

class TestGARCH11:
    def test_fit_produces_valid_persistence(self, returns):
        g = GARCH11().fit(returns)
        assert g.omega_ > 0
        assert 0 < g.alpha_ < 1
        assert 0 < g.beta_ < 1
        # Stationarity: alpha + beta < 1
        assert g.alpha_ + g.beta_ < 1.0

    def test_predict_positive_annualised_vol(self, returns):
        g = GARCH11().fit(returns)
        vol = g.predict().dropna()
        assert (vol > 0).all()
        # Annualised vol should be in a sane range for the synthetic series
        assert vol.median() < 5.0


# ── 6. compute_metrics ────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_prediction(self):
        idx = pd.bdate_range("2019-01-01", periods=100)
        y = pd.Series(np.linspace(0.1, 0.3, 100), index=idx)
        m = compute_metrics(y, y.copy(), label="perfect")
        assert m["r2"] == pytest.approx(1.0, abs=1e-9)
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)
        assert m["corr"] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.filterwarnings("ignore:An input array is constant")
    def test_mean_predictor_r2_zero(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2019-01-01", periods=200)
        y = pd.Series(rng.normal(0.2, 0.05, 200), index=idx)
        y_hat = pd.Series(np.full(200, y.mean()), index=idx)
        m = compute_metrics(y, y_hat, label="mean")
        # Predicting the mean gives ss_res = ss_tot → R² = 0
        assert m["r2"] == pytest.approx(0.0, abs=1e-9)

    def test_too_few_obs_returns_empty(self):
        idx = pd.bdate_range("2019-01-01", periods=5)
        y = pd.Series(np.arange(5.0), index=idx)
        assert compute_metrics(y, y, label="tiny") == {}


# ── 7. walk_forward_predict — structure ───────────────────────────────────────

class TestWalkForwardStructure:
    def test_burn_in_is_nan(self, xy):
        X, y = xy
        tm = 300
        yhat = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=tm)
        assert yhat.iloc[:tm].isna().all()
        assert yhat.iloc[tm:].notna().all()

    def test_predictions_finite_and_positive(self, xy):
        X, y = xy
        yhat = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=300)
        oos = yhat.dropna()
        assert np.isfinite(oos.values).all()
        assert (oos > 0).all()


# ── 8. NO-LOOK-AHEAD — the central credibility claim, proven ──────────────────

class TestNoLookAhead:
    def test_walk_forward_no_lookahead_poison(self, xy):
        """Corrupt every target AFTER a cutoff with garbage values and re-run
        the walk-forward. Every prediction BEFORE the cutoff must be unchanged
        — proving that a prediction at t depends only on data strictly before t.
        If any future leakage existed, the poisoned targets would shift the
        earlier predictions and this test would fail."""
        X, y = xy
        tm = 200
        poison_at = 350  # position; predictions [tm, poison_at) must be immune

        clean = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=tm)

        y_poison = y.copy()
        y_poison.iloc[poison_at:] = 999.0  # absurd future targets
        poisoned = walk_forward_predict(PDVLinear, {}, X, y_poison,
                                        train_min_days=tm)

        # Predictions at positions [tm, poison_at) train on y[:t] ⊆ y[:poison_at]
        # (all clean) → must be byte-identical.
        pd.testing.assert_series_equal(
            clean.iloc[tm:poison_at],
            poisoned.iloc[tm:poison_at],
            check_exact=True,
        )

    def test_embargo_purges_overlapping_labels(self, xy):
        """For a horizon-h target, consecutive labels overlap by h-1 days, so the
        walk-forward must PURGE the last h-1 training rows (embargo). Corrupting a
        target at position j must NOT move predictions at j+1..j+h-1 (purged), but
        MUST move the prediction at j+h. The contrast with horizon=1 (where the
        very next prediction DOES move) proves the embargo is actually applied."""
        X, y = xy
        tm, j, h = 200, 400, 5
        clean = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=tm, horizon=h)
        yp = y.copy(); yp.iloc[j] = 999.0
        pois = walk_forward_predict(PDVLinear, {}, X, yp, train_min_days=tm, horizon=h)
        # Embargo protects the h-1 predictions immediately after the poisoned label
        pd.testing.assert_series_equal(
            clean.iloc[j + 1:j + h], pois.iloc[j + 1:j + h], check_exact=True
        )
        # The poison bites exactly at j+h (non-vacuous)
        assert not np.isclose(clean.iloc[j + h], pois.iloc[j + h])
        # Contrast: with horizon=1 there is no embargo, so the very next
        # prediction DOES change — confirming the purge is what protects j+1..j+h-1.
        c1 = walk_forward_predict(PDVLinear, {}, X, y,  train_min_days=tm, horizon=1)
        p1 = walk_forward_predict(PDVLinear, {}, X, yp, train_min_days=tm, horizon=1)
        assert not np.isclose(c1.iloc[j + 1], p1.iloc[j + 1])

    def test_poison_actually_bites_later(self, xy):
        """Guard against a vacuous test: predictions that DO train on poisoned
        targets must visibly change, confirming the poison is potent."""
        X, y = xy
        tm = 200
        poison_at = 350
        clean = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=tm)
        y_poison = y.copy()
        y_poison.iloc[poison_at:] = 999.0
        poisoned = walk_forward_predict(PDVLinear, {}, X, y_poison,
                                        train_min_days=tm)
        # Well past the cutoff the training window includes poisoned rows.
        assert not np.isclose(clean.iloc[450], poisoned.iloc[450])


# ── Honest effect size: PDV vs simple baselines, scale-invariant ──────────────

class TestForecastSkill:
    """Locks the correction to the oversold 'R^2=0.31 vs 0.08 naive, 4x' claim.
    The 0.08 was the naive predictor's UNCALIBRATED R^2; on a fair scale-invariant
    basis (corr^2) the gap to a trivial EWMA/GARCH is modest, not 4x. Real-data
    magnitudes (PDV ~0.30, GARCH/EWMA ~0.24, 20d MA ~0.20) are reproducible via
    forecast_skill_comparison(); here we lock the structural insight on the
    synthetic fixture (absolute numbers are small by construction)."""

    def test_comparison_structure(self, returns):
        out = forecast_skill_comparison(returns, train_min_days=300)
        for k in ("pdv_walk_forward", "naive_rv20",
                  "ewma_riskmetrics_0p94", "garch11_in_sample"):
            assert k in out
            assert np.isfinite(out[k]["corr2"])

    def test_naive_uncalibrated_r2_understates_information(self, returns):
        """A 20-day RV runs hot vs E|r| (RV ~ sigma, E|r| ~ 0.8 sigma), so its
        uncalibrated R^2 sits FAR below its corr^2. That scale penalty is exactly
        what made '0.08 naive' an apples-to-oranges comparison against PDV's
        OLS-calibrated R^2. The same penalty hits the raw EWMA forecast."""
        out = forecast_skill_comparison(returns, train_min_days=300)
        assert out["naive_rv20"]["r2_uncalibrated"] < out["naive_rv20"]["corr2"]
        assert (out["ewma_riskmetrics_0p94"]["r2_uncalibrated"]
                < out["ewma_riskmetrics_0p94"]["corr2"])
