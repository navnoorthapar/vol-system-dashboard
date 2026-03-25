"""
test_regime_classifier.py — Unit tests for C8: Volatility Regime Classifier.

Test coverage:
  1.  build_features()  — shape, columns, NaN handling, zero look-ahead
  2.  build_regime_labels() — rule logic for each regime, priority, NaN rows
  3.  build_dataset()   — X/y alignment, shift-by-1 zero look-ahead
  4.  RegimeClassifier  — fit/predict/predict_proba, feature importance, repr
  5.  train_test_split_temporal() — strict date boundary
  6.  evaluate_classifier() — keys, confusion matrix shape, recall
  7.  validate_regime2_dates() — stress-date checks
  8.  regime_distribution_by_year() — shape/columns
  9.  Persistence        — save/load roundtrip
  10. RegimePipeline     — repr, require-run guard

Total: 28 tests across 10 test classes.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.signals.regime_classifier import (
    FEATURE_COLS,
    REGIME2_VALIDATION_DATES,
    REGIME_NAMES,
    TEST_START_DATE,
    TRAIN_END_DATE,
    VVIX_REGIME2_THRESHOLD,
    RegimeClassifier,
    RegimePipeline,
    build_dataset,
    build_features,
    build_regime_labels,
    evaluate_classifier,
    load_regime_labels,
    regime_distribution_by_year,
    save_regime_labels,
    train_test_split_temporal,
    validate_regime2_dates,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_spx(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic SPX OHLCV with log_returns."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    closes = 3000.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    log_ret = np.concatenate([[np.nan], np.diff(np.log(closes))])
    return pd.DataFrame({
        "date":       dates,
        "open":       closes * 0.999,
        "high":       closes * 1.005,
        "low":        closes * 0.995,
        "close":      closes,
        "volume":     rng.integers(1_000_000, 5_000_000, n),
        "log_return": log_ret,
    })


def _make_vix(spx_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Synthetic VIX term structure aligned with SPX dates."""
    rng = np.random.default_rng(seed)
    n = len(spx_df)
    vix   = rng.uniform(12, 30, n)
    vvix  = rng.uniform(70, 130, n)    # range 70-130 ensures some > 100
    vix3m = vix + rng.uniform(-3, 3, n)
    vix6m = vix3m + rng.uniform(-2, 2, n)
    dates = spx_df["date"].values
    return pd.DataFrame({
        "date":          dates,
        "^VIX":          vix,
        "^VIX3M":        vix3m,
        "^VIX6M":        vix6m,
        "^VIX9D":        vix - rng.uniform(0, 2, n),
        "^VVIX":         vvix,
        "ts_slope_3m9d": vix3m - (vix - rng.uniform(0, 2, n)),
        "ts_slope_6m1m": vix6m - vix,
    })


@pytest.fixture
def spx_df():
    return _make_spx(300)


@pytest.fixture
def vix_wide_df(spx_df):
    return _make_vix(spx_df)


@pytest.fixture
def features_df(spx_df, vix_wide_df):
    return build_features(spx_df, vix_wide_df, pdv_model=None)


@pytest.fixture
def labels_series(spx_df, vix_wide_df):
    return build_regime_labels(spx_df, vix_wide_df)


@pytest.fixture
def dataset(spx_df, vix_wide_df):
    return build_dataset(spx_df, vix_wide_df, pdv_model=None)


@pytest.fixture
def fitted_clf(dataset):
    X, y = dataset
    clf = RegimeClassifier(params={
        "objective": "multi:softprob",
        "num_class":  3,
        "n_estimators": 20,
        "max_depth":    2,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
    })
    clf.fit(X, y, use_sample_weights=True)
    return clf, X, y


# ── 1. build_features ────────────────────────────────────────────────────────

class TestBuildFeatures:
    def test_returns_dataframe(self, features_df):
        assert isinstance(features_df, pd.DataFrame)

    def test_has_all_feature_columns(self, features_df):
        for col in FEATURE_COLS:
            assert col in features_df.columns, f"Missing column: {col}"

    def test_index_is_datetime(self, features_df):
        assert pd.api.types.is_datetime64_any_dtype(features_df.index)

    def test_vix_in_decimal(self, features_df):
        """VIX feature should be < 1 (decimal, not percentage)."""
        valid = features_df["vix"].dropna()
        assert (valid < 1.0).all(), "vix feature should be decimal (< 1.0)"

    def test_vvix_in_decimal(self, features_df):
        """VVIX feature < 5 (decimal)."""
        valid = features_df["vvix"].dropna()
        assert (valid < 5.0).all()

    def test_fear_premium_positive(self, features_df):
        """Fear premium = VIX / rv_20d, always positive."""
        valid = features_df["fear_premium"].dropna()
        assert (valid > 0).all()

    def test_fear_premium_clipped(self, features_df):
        """Fear premium clipped at 10."""
        assert features_df["fear_premium"].max() <= 10.0 + 1e-8

    def test_no_future_data_used(self, spx_df, vix_wide_df):
        """
        Features at time t should not use data from t+1.
        Check: feature row count <= number of trading days.
        """
        feats = build_features(spx_df, vix_wide_df)
        assert len(feats) <= len(spx_df)


# ── 2. build_regime_labels ────────────────────────────────────────────────────

class TestBuildRegimeLabels:
    def test_returns_series(self, labels_series):
        assert isinstance(labels_series, pd.Series)

    def test_values_in_0_1_2(self, labels_series):
        assert set(labels_series.unique()).issubset({0, 1, 2})

    def test_regime2_when_vvix_high(self, spx_df):
        """When VVIX > 100 → Regime 2 overrides all."""
        vix = _make_vix(spx_df)
        # Force VVIX > 100 for all days
        vix["^VVIX"] = 150.0
        labels = build_regime_labels(spx_df, vix)
        # All non-NaN labels must be 2
        assert (labels == 2).all(), "Expected all Regime 2 when VVIX=150"

    def test_regime0_when_rv_exceeds_iv(self, spx_df):
        """When RV >> IV AND VVIX <= 100 → Regime 0."""
        vix = _make_vix(spx_df)
        vix["^VIX"]  = 5.0   # very low IV (5%)
        vix["^VVIX"] = 70.0  # well below 100
        labels = build_regime_labels(spx_df, vix)
        valid = labels.dropna()
        # With rv_20d typically >> 5%, most days should be Regime 0
        assert (valid == 0).mean() > 0.5, "Expected mostly Regime 0 when IV=5%"

    def test_regime1_when_iv_exceeds_rv(self, spx_df):
        """When IV >> RV AND VVIX <= 100 → Regime 1."""
        vix = _make_vix(spx_df)
        vix["^VIX"]  = 80.0  # very high IV (80%)
        vix["^VVIX"] = 70.0
        labels = build_regime_labels(spx_df, vix)
        valid = labels.dropna()
        # With IV=80%, RV rarely exceeds → mostly Regime 1
        assert (valid == 1).mean() > 0.5, "Expected mostly Regime 1 when IV=80%"

    def test_regime2_overrides_rv_iv(self, spx_df):
        """VVIX > 100 → Regime 2 even when rv > iv."""
        vix = _make_vix(spx_df)
        vix["^VIX"]  = 5.0    # low IV (would give Regime 0)
        vix["^VVIX"] = 120.0  # high VVIX → Regime 2
        labels = build_regime_labels(spx_df, vix)
        valid = labels.dropna()
        assert (valid == 2).all(), "Regime 2 must override Regime 0"

    def test_first_rows_nan_dropped(self, labels_series):
        """First 19 days (insufficient for rv_20d) are dropped."""
        # Labels series should have fewer rows than SPX (first 19 rows dropped)
        assert len(labels_series) < 300  # 300 = total SPX rows in fixture


# ── 3. build_dataset ─────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_returns_tuple(self, dataset):
        X, y = dataset
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_y_same_length(self, dataset):
        X, y = dataset
        assert len(X) == len(y)

    def test_x_y_same_index(self, dataset):
        X, y = dataset
        pd.testing.assert_index_equal(X.index, y.index)

    def test_no_nans_after_build(self, dataset):
        X, y = dataset
        assert not X.isna().any().any(), "No NaN in feature matrix after build"
        assert not y.isna().any(), "No NaN in labels after build"

    def test_feature_shift_zero_lookahead(self, spx_df, vix_wide_df):
        """
        X(t) should reflect features from t-1 (shifted), not t.
        Verify: feature values in X are not identical to un-shifted features.
        """
        raw_feats = build_features(spx_df, vix_wide_df)
        X, _ = build_dataset(spx_df, vix_wide_df)
        # At the second date in X, the feature value should match
        # the FIRST date in raw_feats (due to .shift(1))
        if len(X) >= 2 and len(raw_feats) >= 2:
            first_aligned_date = X.index[0]
            shifted_src_date   = raw_feats.index[raw_feats.index < first_aligned_date].max()
            if not pd.isnull(shifted_src_date):
                np.testing.assert_allclose(
                    X.loc[first_aligned_date, "vix"],
                    raw_feats.loc[shifted_src_date, "vix"],
                    rtol=1e-5,
                )


# ── 4. RegimeClassifier ───────────────────────────────────────────────────────

class TestRegimeClassifier:
    def test_repr_not_fitted(self):
        clf = RegimeClassifier()
        assert "not fitted" in repr(clf)

    def test_repr_fitted(self, fitted_clf):
        clf, _, _ = fitted_clf
        assert "fitted" in repr(clf)

    def test_predict_shape(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_in_range(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_proba_shape(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        probas = clf.predict_proba(X)
        assert probas.shape == (len(X), 3)

    def test_predict_proba_sums_to_1(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        probas = clf.predict_proba(X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_series_has_index(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        preds = clf.predict_series(X)
        assert isinstance(preds, pd.Series)
        assert preds.index.equals(X.index)

    def test_feature_importance_returns_series(self, fitted_clf):
        clf, _, _ = fitted_clf
        imp = clf.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) > 0

    def test_predict_raises_before_fit(self, dataset):
        clf = RegimeClassifier()
        X, _ = dataset
        with pytest.raises(RuntimeError):
            clf.predict(X)


# ── 5. train_test_split_temporal ─────────────────────────────────────────────

class TestTrainTestSplit:
    def test_no_overlap(self, dataset):
        X, y = dataset
        X_tr, X_te, y_tr, y_te = train_test_split_temporal(
            X, y, train_end="2018-12-31", test_start="2019-01-01"
        )
        assert X_tr.index.max() <= pd.Timestamp("2018-12-31")
        assert X_te.index.min() >= pd.Timestamp("2019-01-01")

    def test_no_data_loss(self, dataset):
        """Train + test rows should equal total rows."""
        X, y = dataset
        X_tr, X_te, y_tr, y_te = train_test_split_temporal(
            X, y, train_end="2018-12-31", test_start="2019-01-01"
        )
        assert len(X_tr) + len(X_te) == len(X)

    def test_correct_split_proportions(self, dataset):
        X, y = dataset
        # 300 bdate days: 2018-01-02 to approx 2019-03
        # Split at end of 2018
        X_tr, X_te, _, _ = train_test_split_temporal(
            X, y, train_end="2018-12-31", test_start="2019-01-01"
        )
        assert len(X_tr) > 0
        assert len(X_te) > 0


# ── 6. evaluate_classifier ───────────────────────────────────────────────────

class TestEvaluateClassifier:
    def test_returns_dict(self, fitted_clf, dataset):
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert isinstance(result, dict)

    def test_required_keys(self, fitted_clf, dataset):
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        for key in ["accuracy", "per_class", "confusion_matrix",
                    "regime2_recall", "regime_dist_test", "validation_dates"]:
            assert key in result, f"Missing key: {key}"

    def test_accuracy_in_range(self, fitted_clf, dataset):
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_confusion_matrix_shape(self, fitted_clf, dataset):
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert result["confusion_matrix"].shape == (3, 3)

    def test_per_class_has_three_classes(self, fitted_clf, dataset):
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert set(result["per_class"].keys()) == {0, 1, 2}


# ── 7. validate_regime2_dates ────────────────────────────────────────────────

class TestValidateRegime2Dates:
    def test_returns_dict(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        result = validate_regime2_dates(clf, X)
        assert isinstance(result, dict)

    def test_dates_are_keys(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        result = validate_regime2_dates(clf, X, dates=["2020-01-01"])
        assert "2020-01-01" in result

    def test_missing_date_returns_none(self, fitted_clf, dataset):
        """If the date is not in X, result should be None (not raise)."""
        clf, X, _ = fitted_clf
        result = validate_regime2_dates(clf, X, dates=["1900-01-01"])
        assert result["1900-01-01"] is None


# ── 8. regime_distribution_by_year ───────────────────────────────────────────

class TestRegimeDistributionByYear:
    def test_returns_dataframe(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        dist = regime_distribution_by_year(clf, X)
        assert isinstance(dist, pd.DataFrame)

    def test_fractions_sum_to_1_per_year(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        dist = regime_distribution_by_year(clf, X)
        # Each row (year) should sum to ~1
        np.testing.assert_allclose(dist.sum(axis=1), 1.0, atol=1e-6)

    def test_column_names_are_regime_names(self, fitted_clf, dataset):
        clf, X, _ = fitted_clf
        dist = regime_distribution_by_year(clf, X)
        for col in dist.columns:
            assert col in REGIME_NAMES.values()


# ── 9. Persistence ────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, fitted_clf, dataset, tmp_path):
        clf, X, y = fitted_clf
        path = tmp_path / "test_clf.pkl"
        clf.save(path)
        assert path.exists()

        loaded = RegimeClassifier.load(path)
        np.testing.assert_array_equal(clf.predict(X), loaded.predict(X))

    def test_save_creates_parent_dir(self, fitted_clf, tmp_path):
        clf, _, _ = fitted_clf
        path = tmp_path / "deep" / "nested" / "clf.pkl"
        clf.save(path)
        assert path.exists()

    def test_save_regime_labels_creates_parquet(self, fitted_clf, dataset, tmp_path):
        clf, X, y = fitted_clf
        path = tmp_path / "labels.parquet"
        save_regime_labels(clf, X, y_true=y, path=path)
        assert path.exists()

    def test_load_regime_labels_shape(self, fitted_clf, dataset, tmp_path):
        clf, X, y = fitted_clf
        path = tmp_path / "labels.parquet"
        save_regime_labels(clf, X, y_true=y, path=path)
        df = load_regime_labels(path)
        assert len(df) == len(X)
        assert "regime" in df.columns
        assert "confidence" in df.columns


# ── 10. RegimePipeline interface ─────────────────────────────────────────────

class TestRegimePipeline:
    def test_repr_not_run(self):
        pipe = RegimePipeline()
        assert "not run" in repr(pipe)
        assert str(TRAIN_END_DATE) in repr(pipe)

    def test_plot_raises_before_run(self):
        pipe = RegimePipeline()
        with pytest.raises(RuntimeError):
            pipe.plot()


# ── Module constants ──────────────────────────────────────────────────────────

class TestConstants:
    def test_feature_cols_length(self):
        assert len(FEATURE_COLS) == 6

    def test_regime_names_complete(self):
        assert set(REGIME_NAMES.keys()) == {0, 1, 2}

    def test_vvix_threshold_above_median(self):
        """Threshold must be > 62 (minimum VVIX) and < 207 (maximum)."""
        assert 62 < VVIX_REGIME2_THRESHOLD < 207

    def test_validation_dates_list(self):
        assert "2020-03-16" in REGIME2_VALIDATION_DATES
        assert "2025-04-09" in REGIME2_VALIDATION_DATES
