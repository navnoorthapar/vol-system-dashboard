"""
test_pdv.py — Tests for C3: Path-Dependent Volatility model (an EWMA
approximation of Guyon-Lekeufack 2023; see models/pdv.py for how it differs)

This file locks down the core behaviour and converts the project's central
credibility claim — *zero look-ahead bias* — from a docstring comment into an
executable proof. The C22 classes (9-15) pin the corrections of that round:
each was checked to FAIL on the pre-C22 code and pass on the fix.

Test coverage:
  1. extract_pdv_features  — columns, positivity, NO-LOOK-AHEAD on features
  2. build_xy              — target is genuinely next-day vol; no-look-ahead
  3. PDVLinear             — fit/predict, positivity, exact OLS recovery
  4. PDVKernel             — Nadaraya-Watson weights sum to 1, convex-combo bound
  5. GARCH11               — valid persistence, positive vol forecast
  6. compute_metrics       — R²=1 on perfect fit, R²≈0 for mean predictor
  7. walk_forward_predict  — burn-in NaNs, predictions finite
  8. NO-LOOK-AHEAD (poison)— future targets cannot change past predictions
  9. forward target window — horizon>1 label covers [t+1, t+h] exactly
 10. EWM start-up          — one atypical first return cannot dominate sigma2
 11. GARCH timing          — predict() on row t is the forecast for t+1
 12. kernel support        — no silent 1-nearest-neighbour extrapolation
 13. stress test alignment — forecast for day d never uses r_d or later labels
 14. horizon-matched vol forecast — walk_forward_vol_forecast / latest_vol_forecast
 15. real SPX fixture      — units vs realised vol and the VIX
 16. shipped artifact      — the CLI-built pickle loads with the consumer API
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.models.pdv import (
    EWM_BURN_IN,
    extract_pdv_features,
    build_xy,
    forward_target,
    PDVLinear,
    PDVKernel,
    PDVModel,
    GARCH11,
    walk_forward_predict,
    walk_forward_garch,
    walk_forward_vol_forecast,
    latest_vol_forecast,
    walk_forward_skill_report,
    compute_metrics,
    forecast_skill_comparison,
    stress_test_date,
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
    magnitudes on the common 2012-2025 OOS sample (C22): PDV 0.31, GARCH 0.25,
    EWMA 0.21, 20d RV 0.21 — reproducible via forecast_skill_comparison(); here
    we lock the structural insight on the synthetic fixture (absolute numbers
    are small by construction)."""

    def test_comparison_structure(self, returns):
        out = forecast_skill_comparison(returns, train_min_days=300)
        for k in ("pdv_walk_forward", "naive_rv20", "ewma_riskmetrics_0p94",
                  "garch11_in_sample", "garch11_walk_forward"):
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

    def test_every_model_scored_on_the_same_rows(self, returns):
        """C22: PDV was scored on its OOS window while the baselines were scored
        on the full sample (EWMA warm-up included). All n must now agree."""
        out = forecast_skill_comparison(returns, train_min_days=300)
        ns = {k: v["n"] for k, v in out.items()}
        assert len(set(ns.values())) == 1, ns


# ── 9. Forward target window (C22: horizon>1 label leakage) ───────────────────

class TestForwardTarget:
    def _spike(self, n=160, k=100, size=0.10):
        idx = pd.bdate_range("2020-01-01", periods=n)
        r = pd.Series(0.001, index=idx)
        r.iloc[k] = size
        return r, idx, k

    @pytest.mark.parametrize("h", [1, 5, 21])
    def test_spike_enters_label_exactly_for_t_in_k_minus_h_to_k_minus_1(self, h):
        """y(t) must cover [t+1, t+h]: a single large return at position k is
        inside the window exactly for t = k-h .. k-1. The pre-C22 window
        r2.shift(-1).rolling(h) covered [t-h+2, t+1] and lit up t = k-1..k+h-2
        (h-1 of those labels were already in the feature information set)."""
        r, idx, k = self._spike()
        feats = extract_pdv_features(r)
        _, y = build_xy(r, feats, horizon=h)
        hit = [idx.get_loc(d) for d in y[y > 0.2].index]
        assert hit == list(range(k - h, k))

    def test_rv_target_formula(self):
        r = _make_returns(200, seed=3)
        y = forward_target(r, horizon=21, target="rv")
        t = 50
        expected = np.sqrt(ANNUALISE / 21 * np.sum(r.iloc[t + 1:t + 22] ** 2))
        assert y.iloc[t] == pytest.approx(expected, rel=1e-12)
        assert y.iloc[-21:].isna().all() and np.isfinite(y.iloc[-22])

    def test_abs_and_rv_coincide_at_horizon_one(self):
        r = _make_returns(100, seed=4)
        a = forward_target(r, 1, "abs_return")
        b = forward_target(r, 1, "rv")
        pd.testing.assert_series_equal(a, b, check_names=False)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            forward_target(_make_returns(50), 1, "variance")

    def test_build_xy_burn_in_drops_leading_rows(self, returns):
        feats = extract_pdv_features(returns)
        X0, _ = build_xy(returns, feats, horizon=1, burn_in=0)
        X1, _ = build_xy(returns, feats, horizon=1)
        assert X1.index[0] == returns.index[EWM_BURN_IN]
        assert X0.index[0] < X1.index[0]

    def test_validate_reports_purged_metrics_for_horizon_5(self, returns):
        """validate() must pass the horizon to the walk-forward (embargo) and
        start the OOS window after it."""
        m = PDVModel(horizon=5, train_min_days=300).fit(returns)
        out = m.validate(returns, verbose=False)
        lin = out["predictions"]["linear"]
        # first OOS forecast sits at row train_min_days + (h-1) of X
        assert lin.notna().idxmax() == out["y_oos"].index[0]

    def test_verbose_reports_go_to_logging_not_stdout(self, returns, capsys, caplog):
        """print() output is block-buffered in CI and landed after later log
        lines; the verbose reports are one logging record each."""
        import logging
        m = PDVModel(horizon=1, train_min_days=300).fit(returns)
        with caplog.at_level(logging.INFO, logger="joint_vol_calibration.models.pdv"):
            m.validate(returns, verbose=True)
            m.stress_test(returns, stress_date=str(returns.index[-5].date()), verbose=True)
        assert capsys.readouterr().out == ""
        text = caplog.text
        assert "PDV WALK-FORWARD VALIDATION RESULTS" in text
        assert "PDVLinear coefficients" in text
        assert "PDV STRESS TEST" in text


# ── 10. EWM start-up (C22: the first return seeded sigma2) ────────────────────

class TestEWMStartUp:
    def test_atypical_first_return_does_not_dominate_slow_ewma(self):
        """SPX 2010 started with a 0.31% day; the recursive EWMA seeded f2 with
        that single r_0^2 and still gave it 70% of the weight 30 days later
        (sigma2 0.087 at row 20 vs a trailing RMS of 0.16). With a bias-free
        start-up, f2 after 30 iid N(0, s^2) days must average ~s^2 even when
        r_0 = 0.1 s."""
        rng = np.random.default_rng(11)
        s = 0.01
        vals = []
        for _ in range(150):
            r = rng.normal(0.0, s, 31)
            r[0] = 0.1 * s
            f = extract_pdv_features(pd.Series(r, index=pd.bdate_range("2010-01-04", periods=31)))
            vals.append(f["f2"].iloc[30] / ANNUALISE / s ** 2)
        assert np.mean(vals) == pytest.approx(1.0, abs=0.1)

    def test_start_up_is_still_causal(self, returns):
        a = extract_pdv_features(returns.iloc[:40])
        b = extract_pdv_features(returns)
        pd.testing.assert_frame_equal(a[["f1", "f2", "lev"]], b[["f1", "f2", "lev"]].iloc[:40])


# ── 11. GARCH timing (C22: predict() was sigma_t, the variance of r_t) ─────────

class TestGARCHTiming:
    def test_predict_on_row_t_is_forecast_for_t_plus_1(self, returns):
        g = GARCH11().fit(returns)
        r = returns.values
        sig2 = np.empty(len(r))
        sig2[0] = np.var(r)
        for t in range(1, len(r)):          # independent explicit recursion
            sig2[t] = g.omega_ + g.alpha_ * r[t - 1] ** 2 + g.beta_ * sig2[t - 1]
        t = 400
        nxt = g.omega_ + g.alpha_ * r[t] ** 2 + g.beta_ * sig2[t]
        assert g.predict().iloc[t] == pytest.approx(np.sqrt(nxt * ANNUALISE), rel=1e-10)

    def test_forecast_reacts_to_todays_return_on_todays_row(self, returns):
        """A shock on day t must raise the row-t forecast (it is the forecast
        for t+1) and leave row t-1 untouched."""
        t = 500
        shocked = returns.copy()
        shocked.iloc[t] = 0.08
        g0 = GARCH11().fit(returns)
        v0 = g0.filter(returns)
        v1 = g0.filter(shocked)              # same parameters, shocked path
        assert v1.iloc[t] > 2 * v0.iloc[t]
        assert v1.iloc[t - 1] == pytest.approx(v0.iloc[t - 1], rel=1e-12)
        pd.testing.assert_series_equal(g0.predict(), v0, check_names=False)

    def test_walk_forward_garch_no_lookahead(self, returns):
        cut = 600
        a = walk_forward_garch(returns, train_min_days=300, refit_every=100)
        poisoned = returns.copy()
        poisoned.iloc[cut:] *= 10
        b = walk_forward_garch(poisoned, train_min_days=300, refit_every=100)
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])
        assert a.iloc[:300].isna().all() and a.iloc[300:].notna().all()


# ── 12. Kernel support (C22: NW silently became 1-nearest-neighbour) ──────────

class TestKernelSupport:
    def test_far_query_falls_back_to_ols_not_nearest_neighbour(self, xy):
        X, y = xy
        m = PDVKernel(use_leverage=True).fit(X.iloc[:400], y.iloc[:400])
        q = X.iloc[[10]].copy()
        q["sigma1"] = X["sigma1"].iloc[:400].max() + 20 * m._h[0]
        d_min, ess = m.support_diagnostics(q[m._xcols].values[0])
        assert d_min > m.max_support_dist
        pred = m.predict(q).iloc[0]
        nearest = m._y_train[np.argmax(m._log_kernel(q[m._xcols].values[0]))]
        Xb = np.column_stack([m._X_train, np.ones(len(m._X_train))])
        beta = np.linalg.lstsq(Xb, m._y_train, rcond=None)[0]
        ols = float(np.r_[q[m._xcols].values[0], 1.0] @ beta)
        assert pred == pytest.approx(max(ols, 1e-4), rel=1e-10)
        assert not np.isclose(pred, nearest)
        assert m.n_fallback_ == 1

    def test_interior_query_uses_kernel_average(self, xy):
        X, y = xy
        m = PDVKernel().fit(X.iloc[:400], y.iloc[:400])
        q = X.iloc[[200]]
        d_min, ess = m.support_diagnostics(q[m._xcols].values[0])
        assert d_min == pytest.approx(0.0, abs=1e-12) and ess >= m.min_ess
        w = m._gaussian_kernel_weights(q[m._xcols].values[0])
        assert m.predict(q).iloc[0] == pytest.approx(float(w @ m._y_train))
        assert m.n_fallback_ == 0

    def test_isolated_training_point_does_not_replay_its_label(self):
        """In-sample, an isolated crisis row carried ~100% of its own NW weight,
        so the 'prediction' was its stored label (2020-03-16: 92.42% =
        |r_2020-03-17|·sqrt(252)). Low ESS must route it to the OLS fallback."""
        rng = np.random.default_rng(5)
        n = 300
        X = pd.DataFrame({"sigma1": rng.uniform(0.1, 0.2, n),
                          "sigma2": rng.uniform(0.1, 0.2, n)},
                         index=pd.bdate_range("2019-01-01", periods=n))
        y = pd.Series(0.15 + rng.normal(0, 0.01, n), index=X.index)
        X.iloc[-1] = [0.9, 0.5]        # isolated crisis point
        y.iloc[-1] = 2.0               # its (future) label
        m = PDVKernel().fit(X, y)
        pred = m.predict(X.iloc[[-1]]).iloc[0]
        assert abs(pred - 2.0) > 0.5


# ── 13. Stress-test alignment (C22: the published COVID number) ───────────────

class TestStressTestAlignment:
    def _setup(self):
        r = _make_returns(800, seed=9)
        k = 700
        return r, r.index[k], k

    def test_forecast_for_day_d_does_not_depend_on_r_d(self):
        r, d, k = self._setup()
        preds = []
        for shock in (-0.10, 0.002):
            rr = r.copy()
            rr.iloc[k] = shock
            feats = extract_pdv_features(rr)
            X, y = build_xy(rr, feats)
            fitted = PDVLinear().fit(X, y)          # full-sample fit passed in
            out = stress_test_date(rr, feats, {"linear": fitted}, str(d.date()))
            row = out[out["date"] == str(d.date())].iloc[0]
            assert row["forecast_origin"] == str(r.index[k - 1].date())
            preds.append(row["pred_linear"])
        assert preds[0] == preds[1]

    def test_actual_is_the_move_of_the_target_day(self):
        r, d, k = self._setup()
        feats = extract_pdv_features(r)
        out = stress_test_date(r, feats, {"linear": PDVLinear()}, str(d.date()))
        row = out[out["date"] == str(d.date())].iloc[0]
        assert row["actual_ann_vol"] == pytest.approx(
            round(abs(r.iloc[k]) * np.sqrt(ANNUALISE) * 100, 2))

    def test_walk_forward_matches_manual_causal_fit(self):
        """The model behind the forecast for day d is fit only on labels known
        at the close of d-1 (rows up to d-2 for horizon 1)."""
        r, d, k = self._setup()
        feats = extract_pdv_features(r)
        out = stress_test_date(r, feats, {"linear": PDVLinear()}, str(d.date()))
        X, y = build_xy(r, feats)
        tr = X.index <= r.index[k - 2]
        manual = PDVLinear().fit(X[tr], y[tr]).predict(feats.iloc[[k - 1]]).iloc[0]
        row = out[out["date"] == str(d.date())].iloc[0]
        assert row["pred_linear"] == pytest.approx(round(manual * 100, 2))

    def test_kernel_stress_forecast_is_not_its_own_stored_label(self):
        """Pre-C22, the in-sample kernel's prediction on an isolated crisis row
        was (almost exactly) that row's stored training label |r_{d+1}|. Build
        a crisis whose most extreme row d is followed by a flat day (label 0):
        replaying the label gives ~1%, while an honest forecast after five
        5-8% moves must be crisis-sized."""
        r = _make_returns(800, seed=9)
        r.iloc[690:695] = [-0.06, 0.05, -0.07, 0.06, -0.08]
        r.iloc[695] = 0.0
        feats = extract_pdv_features(r)
        X, y = build_xy(r, feats)
        kern = PDVKernel(use_leverage=True).fit(X, y)
        d = X["sigma1"].idxmax()
        assert y.loc[d] == 0.0
        out = stress_test_date(r, feats, {"kernel": kern}, str(d.date()))
        row = out[out["date"] == str(d.date())].iloc[0]
        assert row["pred_kernel"] > 20.0
        assert out.attrs["fit"] == "walk-forward"


# ── 14. Horizon-matched vol forecast (C22: E|r| was subtracted from the VIX) ──

def _fat_tailed_returns(n=2500, seed=21):
    """GARCH(1,1) with Student-t(4) shocks: E|r|/RMS well below sqrt(2/pi)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(4, n) / np.sqrt(2.0)          # unit variance
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = 1e-4
    for t in range(n):
        if t:
            s2[t] = 2e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return pd.Series(r, index=pd.bdate_range("2012-01-02", periods=n))


class TestWalkForwardVolForecast:
    def test_no_lookahead_poison(self, returns):
        cut = 600
        a = walk_forward_vol_forecast(returns, horizon=21, train_min_days=300)
        p = returns.copy()
        p.iloc[cut:] *= 10
        b = walk_forward_vol_forecast(p, horizon=21, train_min_days=300)
        before = a.index[a.index < returns.index[cut]]
        assert len(before) > 100
        pd.testing.assert_series_equal(a.loc[before], b.loc[before])

    def test_embargo_must_cover_the_horizon(self, returns):
        with pytest.raises(ValueError):
            walk_forward_vol_forecast(returns, horizon=21, embargo=5)

    def test_one_day_abs_target_equals_walk_forward_predict(self, returns):
        a = walk_forward_vol_forecast(returns, horizon=1, target="abs_return",
                                      train_min_days=300)
        feats = extract_pdv_features(returns)
        X, y = build_xy(returns, feats, horizon=1)
        b = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=300).dropna()
        common = a.index.intersection(b.index)
        assert len(common) == len(b)
        np.testing.assert_allclose(a.loc[common], b.loc[common], rtol=1e-10)

    def test_forecasts_reach_the_last_date(self, returns):
        """The last h rows have no label yet but must still get a forecast:
        the final value is the live forecast."""
        a = walk_forward_vol_forecast(returns, horizon=21, train_min_days=300)
        assert a.index[-1] == returns.index[-1]
        live = latest_vol_forecast(returns, horizon=21)
        assert live["vol"] == pytest.approx(a.iloc[-1], rel=1e-10)
        assert live["feature_date"] == returns.index[-1]
        assert live["train_end"] == returns.index[-1 - 21]

    def test_rms_target_is_variance_consistent_and_eabs_is_not(self):
        r = _fat_tailed_returns()
        f21 = walk_forward_vol_forecast(r, horizon=21, train_min_days=500)
        f1 = walk_forward_vol_forecast(r, horizon=1, target="abs_return",
                                       train_min_days=500)
        y21 = forward_target(r, 21, "rv")
        both = pd.concat([f21.rename("f21"), f1.rename("f1"), y21.rename("y")],
                         axis=1, join="inner").dropna()
        assert both["y"].mean() / both["f21"].mean() == pytest.approx(1.0, abs=0.1)
        assert both["f1"].mean() / both["y"].mean() < 0.85


class TestSkillReport:
    def test_pooled_r2_matches_compute_metrics(self, returns):
        rep = walk_forward_skill_report(returns, train_min_days=300, horizons=(1,))
        feats = extract_pdv_features(returns)
        X, y = build_xy(returns, feats)
        p = walk_forward_predict(PDVLinear, {}, X, y, train_min_days=300)
        m = compute_metrics(y, p)
        row = rep["summary"].iloc[0]
        assert row["r2"] == pytest.approx(m["r2"], abs=1e-12)
        assert row["n_obs"] == m["n_obs"]


# ── 15. Real SPX fixture (captured from the project DB, 2010-2025) ────────────

_FIXTURE = Path(__file__).parent / "fixtures" / "pdv_spx_vix_daily_2010_2025.csv.gz"


@pytest.fixture(scope="module")
def spx():
    df = pd.read_csv(_FIXTURE, index_col=0, parse_dates=True)
    return df[df.index <= "2025-12-30"]


class TestRealSPX:
    def test_21d_forecast_matches_realised_scale(self, spx):
        r = spx["log_return"]
        f21 = walk_forward_vol_forecast(r, horizon=21)
        y21 = forward_target(r, 21, "rv")
        both = pd.concat([f21.rename("f"), y21.rename("y")], axis=1, join="inner").dropna()
        assert both["y"].mean() / both["f"].mean() == pytest.approx(1.0, abs=0.1)

    def test_vix_spread_long_trigger_is_not_permanent(self, spx):
        """With the 1-day E|r| model, 'VIX > PDV + 2pp' held on 96% of days —
        a units artefact. The horizon-matched forecast must not."""
        r, vix = spx["log_return"], spx["vix_close"] / 100.0
        f21 = walk_forward_vol_forecast(r, horizon=21)
        d = pd.concat([f21.rename("f"), vix.rename("v")], axis=1, join="inner").dropna()
        share = ((d["f"] - d["v"]) < -0.02).mean()
        assert share < 0.8

    def test_covid_stress_forecast_is_ex_ante(self, spx):
        """Forecast for 2020-03-16 made at the 2020-03-13 close by the
        walk-forward linear model: 70.0% (realised 202.6%). The pre-C22 card
        said 92.4% — the in-sample kernel replaying |r_2020-03-17|."""
        r = spx["log_return"]
        feats = extract_pdv_features(r)
        out = stress_test_date(r, feats, {"linear": PDVLinear()}, "2020-03-16")
        row = out[out["date"] == "2020-03-16"].iloc[0]
        assert row["forecast_origin"] == "2020-03-13"
        assert row["actual_ann_vol"] == pytest.approx(202.64, abs=0.01)
        assert row["pred_linear"] == pytest.approx(70.0, abs=0.5)


# ── 16. The shipped artifact ──────────────────────────────────────────────────

_DB = Path(__file__).resolve().parents[2] / "data_store" / "vol_system.db"


@pytest.mark.skipif(not _DB.exists(), reason="project DB not present")
class TestBuildArtifact:
    def test_cli_pickle_is_loadable_by_consumers(self, tmp_path):
        """`python -m joint_vol_calibration.models.pdv` must pickle the
        importable classes (not __main__.PDVModel) with the consumer API:
        _features, linear_.coef_ (3), linear_.predict, horizon 21, stored
        returns. Reads the DB, writes only to tmp_path."""
        import pickle
        import subprocess
        import sys

        out = tmp_path / "pdv_model.pkl"
        root = Path(__file__).resolve().parents[2]
        subprocess.run(
            [sys.executable, "-m", "joint_vol_calibration.models.pdv",
             "--as-of", "2015-12-31", "--out", str(out)],
            cwd=root, check=True, capture_output=True, text=True, timeout=600,
        )
        raw = out.read_bytes()
        assert b"__main__" not in raw
        m = pickle.loads(raw)
        assert type(m).__module__ == "joint_vol_calibration.models.pdv"
        assert m.horizon == 21 and len(m.linear_.coef_) == 3
        assert m._features.index[-1] == m._log_returns.index[-1]
        pred = m.linear_.predict(m._features[["sigma1", "sigma2", "lev"]].iloc[[-1]])
        assert 0.05 < pred.iloc[0] < 1.0
