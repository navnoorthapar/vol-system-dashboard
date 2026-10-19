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
  9.  Persistence        — save/load roundtrip, version-robust pickle,
                           predictions never written over the rule labels
  10. RegimePipeline     — repr, require-run guard, no full-sample PDV
  11. The ONE R2 gate     — adaptive default, VVIX gaps, noise audit on the
                           same labels (C22)
  12. HMM                — rule-semantic state mapping (causality is in
                           test_lookahead.py)
  13. Probability calibration, evaluation provenance, accuracy report,
      rule-label frame, predictions-table rows, forward-premium table, σ̂
      injection hook, train/serve feature parity, the shipped evaluation record
"""


import numpy as np
import pandas as pd
import pytest

import pickle

import xgboost as xgb

import joint_vol_calibration.signals.regime_classifier as rc
from joint_vol_calibration.signals.regime_classifier import (
    CLASSIFIER_FEATURE_COLS,
    FEATURE_COLS,
    PDV_SOURCE_INJECTED,
    PDV_SOURCE_RV_PROXY,
    R2_GATE,
    REGIME2_VALIDATION_DATES,
    REGIME_NAMES,
    TRAIN_END_DATE,
    VVIX_REGIME2_THRESHOLD,
    RegimeClassifier,
    RegimePipeline,
    HMMRegimeClassifier,
    _expected_calibration_error,
    build_dataset,
    build_features,
    build_regime_labels,
    build_rule_label_frame,
    classifier_accuracy_report,
    classifier_evaluation,
    classifier_features,
    classifier_prediction_db_rows,
    gate_name,
    r2_gate_state,
    r2_gate_threshold,
    regime_forward_premium_table,
    regime_label_noise_audit,
    regime_label_summary,
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
        Features at time t must not use data from t+1: poisoning every row
        AFTER t leaves every feature dated ≤ t bit-identical.
        """
        clean = build_features(spx_df, vix_wide_df)
        cutoff = clean.index[200]
        spx_p, vix_p = spx_df.copy(), vix_wide_df.copy()
        spx_p.loc[spx_p["date"] > cutoff, "log_return"] = 0.3
        for c in ("^VIX", "^VIX3M", "^VVIX"):
            vix_p.loc[pd.to_datetime(vix_p["date"]) > cutoff, c] = 777.0
        pois = build_features(spx_p, vix_p)
        pd.testing.assert_frame_equal(clean.loc[:cutoff], pois.loc[:cutoff], check_exact=True)
        assert not clean.loc[clean.index > cutoff].equals(pois.loc[pois.index > cutoff])

    def test_vvix_missing_print_carried_one_session(self, spx_df, vix_wide_df):
        """One missing VVIX print uses the previous print; two leave a gap."""
        vix = vix_wide_df.copy()
        vix.loc[100, "^VVIX"] = np.nan
        vix.loc[[150, 151], "^VVIX"] = np.nan
        feats = build_features(spx_df, vix)
        d = pd.to_datetime(vix["date"])
        assert feats.loc[d[100], "vvix"] == pytest.approx(vix.loc[99, "^VVIX"] / 100.0)
        assert feats.loc[d[150], "vvix"] == pytest.approx(vix.loc[149, "^VVIX"] / 100.0)
        assert np.isnan(feats.loc[d[151], "vvix"])


# ── 1b. the σ̂ injection hook for pdv_iv_spread ──────────────────────────────

class TestVolForecastInjection:
    def test_injected_forecast_defines_spread(self, spx_df, vix_wide_df):
        idx = pd.to_datetime(spx_df["date"])
        sigma = pd.Series(np.linspace(0.12, 0.30, len(idx)), index=idx)
        sigma.iloc[:30] = np.nan                     # warm-up: no forecast yet
        feats = build_features(spx_df, vix_wide_df, pdv_vol_series=sigma)
        vix = vix_wide_df.set_index(pd.to_datetime(vix_wide_df["date"]))["^VIX"] / 100.0
        have = sigma.dropna().index
        np.testing.assert_allclose(feats.loc[have, "pdv_iv_spread"],
                                   (sigma - vix).loc[have], rtol=0, atol=1e-15)
        assert feats.attrs["pdv_iv_spread_source"] == PDV_SOURCE_INJECTED
        # forecast-less days fall back to the rv proxy, and are counted
        proxy = build_features(spx_df, vix_wide_df)["pdv_iv_spread"].iloc[:30]
        pd.testing.assert_series_equal(feats["pdv_iv_spread"].iloc[:30], proxy)
        assert feats.attrs["pdv_iv_spread_n_proxy"] == int(proxy.notna().sum()) > 0

    def test_default_source_is_rv_proxy(self, features_df):
        assert features_df.attrs["pdv_iv_spread_source"] == PDV_SOURCE_RV_PROXY

    def test_rejects_vol_points(self, spx_df, vix_wide_df):
        """σ̂ in vol POINTS (18.0) would make the spread meaningless."""
        idx = pd.to_datetime(spx_df["date"])
        with pytest.raises(ValueError, match="POINTS"):
            build_features(spx_df, vix_wide_df,
                           pdv_vol_series=pd.Series(18.0, index=idx))

    def test_duplicate_forecast_dates_keep_last(self, spx_df, vix_wide_df):
        idx = pd.to_datetime(spx_df["date"])
        sigma = pd.Series(0.2, index=idx)
        dup = pd.Series([0.5], index=[idx[50]])
        feats = build_features(spx_df, vix_wide_df, pdv_vol_series=pd.concat([sigma, dup]))
        vix50 = vix_wide_df.loc[50, "^VIX"] / 100.0
        assert feats.loc[idx[50], "pdv_iv_spread"] == pytest.approx(0.5 - vix50)


# ── 2. build_regime_labels ────────────────────────────────────────────────────

class TestBuildRegimeLabels:
    def test_returns_series(self, labels_series):
        assert isinstance(labels_series, pd.Series)

    def test_values_in_0_1_2(self, labels_series):
        assert set(labels_series.unique()).issubset({0, 1, 2})

    def test_regime2_when_vvix_high(self, spx_df):
        """When VVIX > 100 → Regime 2 overrides all (fixed threshold)."""
        vix = _make_vix(spx_df)
        # Force VVIX > 100 for all days; use explicit threshold=100 (not adaptive)
        vix["^VVIX"] = 150.0
        labels = build_regime_labels(spx_df, vix, vvix_threshold=100.0)
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
        """VVIX > 100 → Regime 2 even when rv > iv (fixed threshold)."""
        vix = _make_vix(spx_df)
        vix["^VIX"]  = 5.0    # low IV (would give Regime 0)
        vix["^VVIX"] = 120.0  # high VVIX → Regime 2 with explicit threshold
        labels = build_regime_labels(spx_df, vix, vvix_threshold=100.0)
        valid = labels.dropna()
        assert (valid == 2).all(), "Regime 2 must override Regime 0"

    def test_first_rows_nan_dropped(self, labels_series):
        """First 19 days (insufficient for rv_20d) are dropped."""
        # Labels series should have fewer rows than SPX (first 19 rows dropped)
        assert len(labels_series) < 300  # 300 = total SPX rows in fixture


# ── 2b. The ONE R2 gate (C22) ────────────────────────────────────────────────

class TestR2Gate:
    def test_default_gate_is_adaptive(self, spx_df, vix_wide_df):
        """Every label builder defaults to the adaptive gate — the gate the
        published labels, the dashboard and the backtest all describe."""
        assert R2_GATE is None
        assert gate_name() == "adaptive_p80_252d"
        default = build_regime_labels(spx_df, vix_wide_df)
        adaptive = build_regime_labels(spx_df, vix_wide_df, vvix_threshold=None)
        fixed = build_regime_labels(spx_df, vix_wide_df, vvix_threshold=100.0)
        pd.testing.assert_series_equal(default, adaptive)
        common = default.index.intersection(fixed.index)
        assert (default[common] != fixed[common]).any(), "fixture cannot tell the gates apart"

    def test_r2_exactly_where_gate_state_is_on(self, spx_df, vix_wide_df):
        labels = build_regime_labels(spx_df, vix_wide_df)
        vvix = vix_wide_df.set_index(pd.to_datetime(vix_wide_df["date"]))["^VVIX"]
        state = r2_gate_state(vvix).reindex(labels.index)
        assert ((labels == 2) == (state == 1.0)).all()
        gate = r2_gate_threshold(vvix)
        # adaptive gate = trailing 252-session 80th percentile, ≥ 63 prints
        assert gate.iloc[:62].isna().all() and gate.iloc[62:].notna().all()
        assert gate.iloc[100] == pytest.approx(vvix.iloc[:101].quantile(0.80))

    def test_missing_vvix_print_does_not_fail_open(self, spx_df, vix_wide_df):
        """VVIX far above the gate with ONE missing print: that day is R2,
        not silently R0/R1 (NaN > gate used to evaluate False)."""
        vix = vix_wide_df.copy()
        vix["^VVIX"] = 150.0
        vix.loc[120, "^VVIX"] = np.nan
        labels = build_regime_labels(spx_df, vix, vvix_threshold=100.0)
        day = pd.Timestamp(vix.loc[120, "date"])
        assert labels.loc[day] == 2

    def test_vvix_outage_leaves_label_unknown(self, spx_df, vix_wide_df):
        """Two consecutive missing prints: the first is carried forward, the
        second has no gate state and is dropped rather than guessed."""
        vix = vix_wide_df.copy()
        vix["^VVIX"] = 150.0
        vix.loc[[120, 121], "^VVIX"] = np.nan
        labels = build_regime_labels(spx_df, vix, vvix_threshold=100.0)
        d120, d121 = (pd.Timestamp(vix.loc[i, "date"]) for i in (120, 121))
        assert labels.loc[d120] == 2
        assert d121 not in labels.index

    def test_label_summary_persistence(self):
        idx = pd.bdate_range("2021-01-04", periods=6)
        lab = pd.Series([1, 1, 2, 2, 0, 0], index=idx)
        out = regime_label_summary(lab)
        assert out["persistence_acc"] == pytest.approx(3 / 5)
        assert out["counts"] == {0: 2, 1: 2, 2: 2}
        assert regime_label_summary(lab, start="2021-01-06")["n"] == 4


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

    def test_default_fit_returns_raw_booster_probabilities(self, fitted_clf):
        """No calibrator is fit on the booster's own training rows any more."""
        clf, X, _ = fitted_clf
        assert isinstance(clf.model_, xgb.XGBClassifier)
        assert clf.calibration_ is None
        np.testing.assert_array_equal(clf.predict_proba(X),
                                      clf.model_.predict_proba(X[clf.feature_names_].values))


_FAST = {"objective": "multi:softprob", "num_class": 3, "n_estimators": 60,
         "max_depth": 3, "learning_rate": 0.1, "random_state": 0, "n_jobs": 1,
         "verbosity": 0}


def _imbalanced_classes(n: int, seed: int = 0):
    """Stationary 3-class data with a rare class 2 (~13%)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = np.column_stack([0.8 * X[:, 0], 0.6 + 0.4 * X[:, 1], -1.4 + 1.2 * X[:, 2]])
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(X, index=idx, columns=["a", "b", "c"]), pd.Series(y, index=idx)


class TestProbabilityCalibration:
    def test_holdout_calibrator_sees_only_later_unseen_rows(self):
        X, y = _imbalanced_classes(1000)
        clf = RegimeClassifier(_FAST).fit(X, y, calibration="holdout", calibration_frac=0.25)
        meta = clf.metadata_
        assert meta["fit_rows"] == 750 and meta["calibration_rows"] == 250
        assert pd.Timestamp(meta["fit_window"][1]) < pd.Timestamp(meta["calibration_window"][0])

    def test_holdout_calibration_improves_out_of_sample_ece(self):
        """Balanced class weights tilt raw P(class 2) far above its frequency;
        an UNWEIGHTED calibrator fit on later rows the booster never saw
        repairs it out of sample."""
        X, y = _imbalanced_classes(3000)
        Xtr, ytr, Xte, yte = X.iloc[:2000], y.iloc[:2000], X.iloc[2000:], y.iloc[2000:]
        raw = RegimeClassifier(_FAST).fit(Xtr, ytr)
        cal = RegimeClassifier(_FAST).fit(Xtr, ytr, calibration="holdout")
        hit = (yte.values == 2).astype(float)
        e_raw = _expected_calibration_error(raw.predict_proba(Xte)[:, 2], hit)
        e_cal = _expected_calibration_error(cal.predict_proba(Xte)[:, 2], hit)
        assert e_raw > 0.10
        assert e_cal < 0.5 * e_raw
        np.testing.assert_allclose(cal.predict_proba(Xte).sum(axis=1), 1.0, atol=1e-12)

    def test_rejects_unknown_calibration(self, dataset):
        X, y = dataset
        with pytest.raises(ValueError):
            RegimeClassifier(_FAST).fit(X, y, calibration="isotonic_in_sample")


class TestVersionRobustPickle:
    def test_pickle_holds_no_sklearn_objects(self):
        X, y = _imbalanced_classes(600)
        clf = RegimeClassifier(_FAST).fit(X, y, calibration="holdout")
        blob = pickle.dumps(clf)
        assert b"sklearn" not in blob, "pickle must not depend on sklearn class layouts"
        back = pickle.loads(blob)
        np.testing.assert_array_equal(back.predict_proba(X), clf.predict_proba(X))
        np.testing.assert_array_equal(back.predict(X), clf.predict(X))
        assert back.metadata_["calibration"] == "holdout"

    def test_legacy_wrapped_model_state_still_loads(self, dataset):
        """A pre-C22 pickle (plain __dict__ with a sklearn wrapper) loads."""
        from sklearn.linear_model import LogisticRegression

        X, y = dataset
        cols = CLASSIFIER_FEATURE_COLS
        legacy = LogisticRegression(max_iter=500).fit(X[cols].values, y.values)
        obj = RegimeClassifier.__new__(RegimeClassifier)
        obj.__setstate__({"params": {}, "model_": legacy, "feature_names_": cols,
                          "train_end_date_": None})
        np.testing.assert_array_equal(obj.predict(X), legacy.predict(X[cols].values))
        assert obj.metadata_ == {"legacy_pickle": True}

    def test_shipped_classifier_is_built_on_the_canonical_gate(self):
        """The shipped research classifier records how it was built and was
        trained on labels from the gate every trading path uses."""
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "data_store" / "signals" / "regime_classifier.pkl"
        if not path.exists():
            pytest.skip("no shipped classifier")
        blob = path.read_bytes()
        assert b"sklearn" not in blob
        clf = RegimeClassifier.load(path)
        meta = clf.metadata_
        assert meta["r2_gate"] == gate_name(R2_GATE)
        assert meta["feature_cols"] == CLASSIFIER_FEATURE_COLS
        assert meta["pdv_iv_spread_source"] == PDV_SOURCE_RV_PROXY
        assert meta["calibration"] == "holdout"
        ev = meta["evaluation"]
        assert 0.0 < ev["oos_acc"] < ev["persistence_acc"] < 1.0

    def test_shipped_evaluation_record_matches_the_shipped_data(self):
        """The dashboard prints metadata_["evaluation"]. It was first written on
        the pre-migration DB, whose spurious 2025-05-08 NULL log_return dropped 20
        test rows (1,488: OOS 75.1% vs persistence 86.8%); on the shipped data the
        same model scores 75.0% vs 86.7% on 1,508 rows."""
        from pathlib import Path
        from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide

        path = Path(__file__).resolve().parents[2] / "data_store" / "signals" / "regime_classifier.pkl"
        spx = get_spx_ohlcv("2025-12-31", "2010-01-01")
        if not path.exists() or len(spx) < 3000:
            pytest.skip("shipped classifier or market history unavailable")
        clf = RegimeClassifier.load(path)
        ev = classifier_evaluation(clf, spx, get_vix_term_structure_wide("2025-12-31", "2010-01-01"))
        stored = clf.metadata_["evaluation"]
        assert stored["n_test"] == ev["n_test"]
        assert tuple(stored["test_window"]) == tuple(ev["test_window"])
        for k, v in ev.items():
            if k not in ("n_test", "test_window"):
                assert stored[k] == pytest.approx(v, abs=1e-12), k


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

    def test_persistence_baseline_reported(self, fitted_clf, dataset):
        """C17: eval must report the no-skill persistence baseline y_hat(t)=y(t-1)."""
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert "persistence_baseline_accuracy" in result
        assert 0.0 <= result["persistence_baseline_accuracy"] <= 1.0
        assert isinstance(result["beats_persistence"], bool)

    def test_reports_provenance(self, fitted_clf):
        """An accuracy number must say which rows, features and gate it is on."""
        clf, X, y = fitted_clf
        result = evaluate_classifier(clf, X, y)
        assert result["n_test"] == len(y)
        assert result["test_window"] == (str(X.index.min().date()), str(X.index.max().date()))
        assert result["feature_cols"] == list(X.columns)
        assert result["r2_gate"] == gate_name(R2_GATE)
        assert result["persistence_n"] == len(y) - 1

    def test_persistence_uses_previous_session_label(self):
        """With the full label series, ŷ(t) is the previous SESSION's label even
        when that session's feature row was dropped from the test set."""
        idx = pd.bdate_range("2021-01-04", periods=6)
        labels = pd.Series([1, 1, 2, 1, 1, 1], index=idx)
        y_test = labels.drop(idx[2])
        acc, n = rc._persistence(y_test, labels)
        assert (acc, n) == (pytest.approx(0.75), 4)
        assert rc._persistence(y_test)[0] == pytest.approx(1.0)   # row-shift view


class TestAccuracyReport:
    def test_report_is_computed_with_matching_baseline(self):
        spx = _make_spx(1100, seed=3)
        vix = _make_vix(spx, seed=3)
        rep = classifier_accuracy_report(
            spx, vix, train_end="2019-12-31", test_start="2020-01-01",
            test_end="2022-03-31", wf_years=[2021], params=_FAST,
        )
        assert rep["r2_gate"] == gate_name(R2_GATE)
        assert rep["pdv_iv_spread_source"] == PDV_SOURCE_RV_PROXY
        X, y = build_dataset(spx, vix)
        in_test = (X.index >= "2020-01-01") & (X.index <= "2022-03-31")
        single = rep["single_split"]
        assert single["n_test"] == int(in_test.sum())
        labels = build_regime_labels(spx, vix)
        assert single["persistence_acc"] == pytest.approx(rc._persistence(y[in_test], labels)[0])
        assert set(rep["walk_forward"]["by_year"]) == {2021}
        for k in ("in_sample_acc", "oos_acc"):
            assert 0.0 <= single[k] <= 1.0
            assert 0.0 <= rep["benchmarks"]["with_vvix"][k] <= 1.0
            assert 0.0 <= rep["benchmarks"]["with_vvix_and_lagged_label"][k] <= 1.0


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
        assert "pred_regime" in df.columns
        assert "confidence" in df.columns
        assert (df["true_regime"] == y).all()

    def test_predictions_never_overwrite_rule_labels(self, fitted_clf, tmp_path, monkeypatch):
        """RegimePipeline used to write classifier PREDICTIONS over the RULE
        label file the dashboard headlines; that path is now refused, and the
        default goes to a separate predictions file."""
        clf, X, y = fitted_clf
        rule_file = tmp_path / "regime_labels.parquet"
        preds_file = tmp_path / "regime_classifier_predictions.parquet"
        monkeypatch.setattr(rc, "LABELS_PARQUET", rule_file)
        monkeypatch.setattr(rc, "CLASSIFIER_PREDICTIONS_PARQUET", preds_file)
        with pytest.raises(ValueError):
            save_regime_labels(clf, X, y_true=y, path=rule_file)
        assert not rule_file.exists()
        assert save_regime_labels(clf, X, y_true=y) == preds_file
        assert preds_file.exists() and not rule_file.exists()


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

    def test_defaults_avoid_look_ahead(self):
        """No full-sample PDV pickle (its coefficients saw the test period),
        and labels from the canonical gate."""
        pipe = RegimePipeline()
        assert pipe.pdv_model_path is None
        assert pipe.vvix_threshold is R2_GATE
        assert gate_name(R2_GATE) in repr(pipe)


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


# ── 11. predict_proba_series ──────────────────────────────────────────────────

class TestPredictProbaSeries:

    def test_returns_dataframe_with_3_columns(self, fitted_clf):
        clf, X, _ = fitted_clf
        proba_df = clf.predict_proba_series(X)
        assert isinstance(proba_df, pd.DataFrame)
        assert set(proba_df.columns) == {"prob_r0", "prob_r1", "prob_r2"}

    def test_proba_rows_sum_to_one(self, fitted_clf):
        clf, X, _ = fitted_clf
        proba_df = clf.predict_proba_series(X)
        row_sums = proba_df.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6,
                                   err_msg="Each row of predict_proba_series must sum to 1")

    def test_index_matches_input(self, fitted_clf):
        clf, X, _ = fitted_clf
        proba_df = clf.predict_proba_series(X)
        assert list(proba_df.index) == list(X.index)


# ── 12. HMMRegimeClassifier ───────────────────────────────────────────────────

def _make_hmm_features(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic HMM feature DataFrame with 3 'regimes': low/mid/high VIX."""
    rng   = np.random.default_rng(seed)
    idx   = pd.bdate_range("2010-01-04", periods=n)
    # Three blocks with distinct VIX levels to help HMM distinguish states
    block = n // 3
    vix   = np.concatenate([
        rng.uniform(0.10, 0.18, block),      # low-vol block
        rng.uniform(0.18, 0.28, block),      # mid-vol block
        rng.uniform(0.28, 0.45, n - 2*block),  # high-vol block
    ])
    vvix      = vix * rng.uniform(3.5, 5.0, n)
    ts_slope  = rng.uniform(-0.005, 0.005, n)
    fear_prem = rng.uniform(0.95, 1.15, n)
    return pd.DataFrame(
        {"vix": vix, "vvix": vvix, "ts_slope": ts_slope, "fear_premium": fear_prem},
        index=idx,
    )


@pytest.fixture
def fitted_hmm():
    X   = _make_hmm_features(n=400)
    hmm = HMMRegimeClassifier(n_components=3, n_iter=50, random_state=42)
    hmm.fit(X)
    return hmm, X


class TestHMMRegimeClassifier:

    def test_require_fitted_raises_before_fit(self):
        hmm = HMMRegimeClassifier()
        X   = _make_hmm_features(n=10)
        with pytest.raises(RuntimeError, match="not fitted"):
            hmm.predict(X)

    def test_fit_sets_is_fitted(self, fitted_hmm):
        hmm, _ = fitted_hmm
        assert hmm.is_fitted is True

    def test_state_to_regime_covers_all_3(self, fitted_hmm):
        hmm, _ = fitted_hmm
        assert hmm._state_to_regime is not None
        assert set(hmm._state_to_regime.values()) == {0, 1, 2}

    def test_predict_returns_array_in_012(self, fitted_hmm):
        hmm, X = fitted_hmm
        labels = hmm.predict(X)
        assert isinstance(labels, np.ndarray)
        assert set(labels).issubset({0, 1, 2})

    def test_predict_shape_matches_input(self, fitted_hmm):
        hmm, X = fitted_hmm
        labels = hmm.predict(X)
        assert len(labels) == len(X)

    def test_hmm_predict_proba_columns(self, fitted_hmm):
        hmm, X = fitted_hmm
        proba_df = hmm.hmm_predict_proba(X)
        assert isinstance(proba_df, pd.DataFrame)
        assert set(proba_df.columns) == {"prob_r0", "prob_r1", "prob_r2"}

    def test_hmm_predict_proba_rows_sum_to_1(self, fitted_hmm):
        hmm, X = fitted_hmm
        proba_df = hmm.hmm_predict_proba(X)
        row_sums = proba_df.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_hmm_predict_proba_index_matches_input(self, fitted_hmm):
        hmm, X = fitted_hmm
        proba_df = hmm.hmm_predict_proba(X)
        assert list(proba_df.index) == list(X.index)

    def test_save_load_roundtrip(self, fitted_hmm, tmp_path):
        hmm, X = fitted_hmm
        path = tmp_path / "hmm_test.pkl"
        hmm.save(path)
        hmm2 = HMMRegimeClassifier.load(path)
        # Predictions should be identical after reload
        labels1 = hmm.predict(X)
        labels2 = hmm2.predict(X)
        np.testing.assert_array_equal(labels1, labels2)

    def test_hmm_features_constant(self, fitted_hmm):
        """HMM_FEATURES class attribute must contain the 4 expected columns."""
        hmm, _ = fitted_hmm
        assert set(hmm.HMM_FEATURES) == {"vix", "vvix", "ts_slope", "fear_premium"}

    def test_state_mapping_follows_rule_semantics(self):
        """Under the rule, R1 (implied ≥ trailing realised) has the LOWEST mean
        VIX; R0 is the state where trailing realised exceeds implied
        (fear_premium < 1); R2 the highest-VVIX state. Ranking states by mean
        VIX (the pre-C22 mapping) swapped R0 and R1."""
        #                 vix   vvix  ts_slope fear_premium
        means = np.array([[0.30, 1.35, -0.010, 1.20],    # state 0: vol-of-vol spike
                          [0.13, 0.85,  0.015, 1.45],    # state 1: calm, IV ≥ RV
                          [0.20, 0.95,  0.002, 0.85]])   # state 2: RV > IV
        assert HMMRegimeClassifier.map_states_to_regimes(means) == {0: 2, 1: 1, 2: 0}

    def test_fit_uses_rule_semantic_mapping(self, fitted_hmm):
        hmm, _ = fitted_hmm
        assert hmm._state_to_regime == HMMRegimeClassifier.map_states_to_regimes(
            hmm._model.means_)

    def test_requires_three_components(self):
        with pytest.raises(ValueError):
            HMMRegimeClassifier(n_components=2)

    def test_multi_start_fit_separates_distinct_clusters(self):
        """Three well-separated clusters: single-start EM (seed 42) merges two
        of them and splits the third; the best of n_init starts separates all
        three, and the mapping names them by rule semantics."""
        rng = np.random.default_rng(0)
        state = np.repeat([1, 0, 2, 1, 0, 2, 1, 0, 2], 100)
        mu = {0: (0.20, 0.95, 0.002, 0.85),      # trailing RV > IV  → R0
              1: (0.13, 0.85, 0.015, 1.45),      # calm, IV ≥ RV     → R1
              2: (0.30, 1.35, -0.010, 1.20)}     # vol-of-vol spike  → R2
        sd = (0.01, 0.03, 0.002, 0.04)
        X = pd.DataFrame([[rng.normal(m, s_) for m, s_ in zip(mu[k], sd)] for k in state],
                         columns=HMMRegimeClassifier.HMM_FEATURES,
                         index=pd.bdate_range("2012-01-02", periods=len(state)))
        single = HMMRegimeClassifier(n_iter=100, n_init=1).fit(X)
        multi = HMMRegimeClassifier(n_iter=100).fit(X)
        assert multi.log_likelihood_ > single.log_likelihood_
        pred = multi.predict(X)
        np.testing.assert_array_equal(pred, state)


# ── Label ontology noise floor (why the classifier was demoted) ───────────────

class TestLabelNoiseAudit:
    """`regime_label_noise_audit` quantifies the noise floor set by the LABEL
    ontology — the ceiling no classifier trained on these labels can beat. The
    labels are same-day observables (rv_20d vs VIX/100; VVIX > gate), so a share
    of days sit on a decision boundary and flip under ordinary measurement noise.
    On the real series that flip rate is ~6% and persistence ~87%; here we lock
    the structure on synthetic data with a REALISTIC (persistent, near-realized)
    VIX so both properties are exercised."""

    @staticmethod
    def _persistent_market(n: int = 500, seed: int = 3):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2016-01-04", periods=n)
        s2 = np.empty(n); r = np.empty(n)
        s2[0] = 1e-4; r[0] = rng.normal(0.0, np.sqrt(s2[0]))
        for t in range(1, n):
            s2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
            r[t] = rng.normal(0.0, np.sqrt(s2[t]))
        spx = pd.DataFrame({"date": dates,
                            "close": 3000.0 * np.exp(np.cumsum(r)),
                            "log_return": r})
        rv = np.sqrt(pd.Series(r ** 2).rolling(20, min_periods=15).mean().bfill()
                     .values * 252)
        vix = np.empty(n); vix[0] = rv[0] * 100
        vvix = np.empty(n); vvix[0] = 95.0
        for t in range(1, n):
            # AR(1) VIX that tracks realized vol → persistent AND near the R0/R1 cut
            vix[t] = 0.90 * vix[t - 1] + 0.10 * (rv[t] * 100) + rng.normal(0, 0.8)
            vvix[t] = 0.90 * vvix[t - 1] + 0.10 * 95.0 + rng.normal(0, 3)
        vix = np.clip(vix, 7, 70); vvix = np.clip(vvix, 75, 150)
        vixw = pd.DataFrame({"date": dates, "^VIX": vix,
                             "^VIX3M": vix + 1.5, "^VVIX": vvix})
        return spx, vixw

    def test_returns_all_keys(self):
        spx, vix = self._persistent_market()
        a = regime_label_noise_audit(spx, vix, vvix_threshold=100.0)
        for k in ("n_days", "boundary_within_1vp", "boundary_within_0p5vp",
                  "label_flip_rate", "perturbation_stable", "r2_gate_fragile",
                  "persistence_acc"):
            assert k in a
        assert a["n_days"] > 100

    def test_audits_the_labels_actually_used(self):
        """Same gate, same labels, same persistence as build_regime_labels —
        the noise floor must describe the labels that are displayed/traded."""
        spx, vix = self._persistent_market()
        for thr in (R2_GATE, 100.0):
            a = regime_label_noise_audit(spx, vix, vvix_threshold=thr)
            labels = build_regime_labels(spx, vix, vvix_threshold=thr)
            assert a["r2_gate"] == gate_name(thr)
            assert a["n_days"] == len(labels)
            assert a["persistence_acc"] == pytest.approx(
                round(regime_label_summary(labels)["persistence_acc"], 4))
        assert regime_label_noise_audit(spx, vix)["r2_gate"] == gate_name(R2_GATE)

    def test_bounded_noise_floor_and_persistence(self):
        spx, vix = self._persistent_market()
        a = regime_label_noise_audit(spx, vix, vvix_threshold=100.0)
        # A real but bounded noise floor: some labels flip, most are stable.
        assert 0.0 < a["label_flip_rate"] < 0.5
        assert a["perturbation_stable"] == pytest.approx(1.0 - a["label_flip_rate"], abs=1e-9)
        # Persistent (same-day) labels → "predict yesterday" is a strong baseline.
        assert a["persistence_acc"] > 0.6
        # Fractions are valid probabilities.
        for k in ("boundary_within_1vp", "boundary_within_0p5vp", "r2_gate_fragile"):
            assert 0.0 <= a[k] <= 1.0


# ── Rule-label frame, DB rows, forward-premium table ──────────────────────────

class TestRuleLabelFrame:
    def test_frame_pairs_label_with_same_day_observables(self, spx_df, vix_wide_df):
        frame = build_rule_label_frame(spx_df, vix_wide_df)
        labels = build_regime_labels(spx_df, vix_wide_df)
        feats = build_features(spx_df, vix_wide_df)
        pd.testing.assert_series_equal(frame["regime"], labels.rename("regime"))
        pd.testing.assert_frame_equal(frame[FEATURE_COLS], feats.loc[labels.index])
        d = labels.index[50]
        r = spx_df.set_index(pd.to_datetime(spx_df["date"]))["log_return"].loc[:d].iloc[-20:]
        assert frame.loc[d, "rv_20d"] == pytest.approx(np.sqrt((r ** 2).mean() * 252))
        assert frame.attrs["r2_gate"] == gate_name(R2_GATE)

    def test_frame_records_the_spread_definition_on_every_row(self, spx_df, vix_wide_df):
        """Label files are built by appending; the proxy and the PDV-21 spread
        share the column name, so each row says which one it holds."""
        frame = build_rule_label_frame(spx_df, vix_wide_df)
        assert set(frame["pdv_iv_spread_source"]) == {PDV_SOURCE_RV_PROXY}
        idx = pd.to_datetime(spx_df["date"])
        sigma = pd.Series(0.15, index=idx)
        injected = build_rule_label_frame(spx_df, vix_wide_df, pdv_vol_series=sigma)
        assert set(injected["pdv_iv_spread_source"]) == {PDV_SOURCE_INJECTED}


class TestPredictionsTable:
    """classifier_predictions holds the research classifier's forecasts. Pre-C22
    (as regime_labels) it stored t−1 features under date t and rv_20d = 2·VIX − RV;
    mid-C22 the writer was pointed at RULE labels while the table was renamed
    for predictions."""

    def test_rows_are_the_forecast_for_t_with_session_t_observables(
            self, spx_df, vix_wide_df, fitted_clf):
        clf, X, _ = fitted_clf
        frame = build_rule_label_frame(spx_df, vix_wide_df)
        rows = classifier_prediction_db_rows(clf, X, frame).set_index("date")
        assert len(rows) == len(X)
        d = X.index[60]
        key = d.strftime("%Y-%m-%d")
        assert rows.loc[key, "regime"] == clf.predict(X.loc[[d]])[0]
        assert rows.loc[key, "confidence"] == pytest.approx(clf.predict_proba(X.loc[[d]]).max())
        vix = vix_wide_df.set_index(pd.to_datetime(vix_wide_df["date"]))
        assert rows.loc[key, "vix_level"] == pytest.approx(vix.loc[d, "^VIX"])
        assert rows.loc[key, "vix_level"] != pytest.approx(X.loc[d, "vix"] * 100)  # X holds t−1
        assert rows.loc[key, "vix_slope"] == pytest.approx(vix.loc[d, "^VIX3M"] - vix.loc[d, "^VIX"])
        assert rows.loc[key, "rv_20d"] == pytest.approx(frame.loc[d, "rv_20d"])
        assert (rows["confidence"] < 1.0).any()

    def test_pipeline_writes_predictions_not_rule_labels(self, tmp_path, monkeypatch):
        spx = _make_spx(600, seed=7)                 # 2018-01-02 .. 2020-04: has test rows
        vix = _make_vix(spx, seed=7)
        monkeypatch.setattr(rc, "get_spx_ohlcv", lambda **k: spx)
        monkeypatch.setattr(rc, "get_vix_term_structure_wide", lambda **k: vix)
        monkeypatch.setattr(rc, "MODEL_PATH", tmp_path / "clf.pkl")
        monkeypatch.setattr(rc, "CLASSIFIER_PREDICTIONS_PARQUET", tmp_path / "preds.parquet")
        written = []
        monkeypatch.setattr(rc, "insert_classifier_predictions", lambda df: written.append(df) or len(df))

        pipe = RegimePipeline(xgb_params=_FAST)
        pipe.run()
        rows = written[0].set_index("date")
        assert list(rows.index) == list(pipe.X_.index.strftime("%Y-%m-%d"))
        np.testing.assert_array_equal(rows["regime"].to_numpy(), pipe.clf_.predict(pipe.X_))
        rule = build_regime_labels(spx, vix).reindex(pipe.X_.index).to_numpy()
        assert (rows["regime"].to_numpy() != rule).any()     # forecasts, not the rule
        ev = pipe.clf_.metadata_["evaluation"]
        assert ev == classifier_evaluation(pipe.clf_, spx, vix)
        assert ev["n_test"] == int((pipe.X_.index >= "2020-01-01").sum())


class TestClassifierFeatures:
    """Train/serve parity: the inputs are rebuilt from the classifier's own record."""

    def _clf(self, fitted_clf, src):
        clf = fitted_clf[0]
        clf.metadata_["pdv_iv_spread_source"] = src
        return clf

    def test_rv_proxy_classifier_gets_the_proxy(self, spx_df, vix_wide_df, fitted_clf):
        clf = self._clf(fitted_clf, PDV_SOURCE_RV_PROXY)
        feats = classifier_features(clf, spx_df, vix_wide_df)
        pd.testing.assert_frame_equal(feats, build_features(spx_df, vix_wide_df))
        assert feats.attrs["pdv_iv_spread_source"] == PDV_SOURCE_RV_PROXY

    def test_walk_forward_classifier_gets_the_walk_forward_spread(
            self, spx_df, vix_wide_df, fitted_clf, monkeypatch):
        idx = pd.to_datetime(spx_df["date"])
        sigma = pd.Series(np.linspace(0.10, 0.30, len(idx)), index=idx)
        monkeypatch.setattr(rc, "walk_forward_pdv_vol", lambda spx, **k: sigma)
        clf = self._clf(fitted_clf, rc.PDV_SOURCE_WALK_FORWARD)
        feats = classifier_features(clf, spx_df, vix_wide_df)
        vix = vix_wide_df.set_index(pd.to_datetime(vix_wide_df["date"]))["^VIX"] / 100
        np.testing.assert_allclose(feats["pdv_iv_spread"], (sigma - vix).reindex(feats.index))
        assert feats.attrs["pdv_iv_spread_source"] == rc.PDV_SOURCE_WALK_FORWARD

    @pytest.mark.parametrize("src", [PDV_SOURCE_INJECTED, "pdv_model_full_sample", None])
    def test_sources_that_cannot_be_rebuilt_raise(self, spx_df, vix_wide_df, fitted_clf, src):
        clf = self._clf(fitted_clf, src)
        with pytest.raises(ValueError, match="pdv_iv_spread source"):
            classifier_features(clf, spx_df, vix_wide_df)


class TestForwardPremiumTable:
    def test_forward_rv_is_t_plus_1_to_t_plus_h(self, spx_df, vix_wide_df):
        labels = build_regime_labels(spx_df, vix_wide_df)
        tab = regime_forward_premium_table(spx_df, vix_wide_df, labels, horizon=21)
        assert set(tab.index) <= {0, 1, 2}
        assert tab["p_fwd_rv_gt_vix"].between(0, 1).all()
        # recompute one regime's numbers by hand
        r = spx_df.set_index(pd.to_datetime(spx_df["date"]))["log_return"]
        v = vix_wide_df.set_index(pd.to_datetime(vix_wide_df["date"]))["^VIX"] / 100
        hits, n = 0, 0
        for d in labels.index[labels == 1]:
            i = r.index.get_loc(d)
            fwd = r.iloc[i + 1:i + 22]
            if len(fwd) < 21:
                continue
            n += 1
            hits += np.sqrt((fwd ** 2).mean() * 252) > v.loc[d]
        assert tab.loc[1, "n"] == n
        assert tab.loc[1, "p_fwd_rv_gt_vix"] == pytest.approx(hits / n)
