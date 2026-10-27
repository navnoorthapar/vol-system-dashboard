"""
Closed-form Sharpe inference (stats/sharpe.py): every formula is checked
against a simulation of the quantity it approximates, not against itself.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from joint_vol_calibration.stats import sharpe as sh


def test_returns_from_nav_reproduce_the_engines_sharpe():
    """Same convention as backtest_engine.compute_metrics (base NAV first)."""
    from joint_vol_calibration.backtest.backtest_engine import INITIAL_CAPITAL, RESULTS_PATH, compute_metrics

    eq = pd.read_parquet(RESULTS_PATH)
    for col in ("nav", "nav_s3", "nav_s1c"):
        m = compute_metrics(pd.DataFrame({"nav": eq[col]}), base_nav=INITIAL_CAPITAL)
        r = sh.returns_from_nav(eq[col].to_numpy(), INITIAL_CAPITAL)
        assert sh.annualise(sh.sharpe(r)) == pytest.approx(m["sharpe"], rel=1e-12)
        assert sh.annualise(sh.sortino(r)) == pytest.approx(m["sortino"], rel=1e-12)
    pnl = eq["pnl_s3"].to_numpy()
    np.testing.assert_allclose(sh.returns_from_pnl(pnl, INITIAL_CAPITAL),
                               sh.returns_from_nav(eq["nav_s3"].to_numpy(), INITIAL_CAPITAL),
                               rtol=1e-10, atol=1e-15)


def test_sample_moments_of_a_normal_sample():
    rng = np.random.default_rng(0)
    m = sh.sample_moments(rng.normal(0.001, 0.01, 200_000))
    assert m["skew"] == pytest.approx(0.0, abs=0.02)
    assert m["kurtosis"] == pytest.approx(3.0, abs=0.05)


def test_sharpe_variance_matches_simulation_for_skewed_fat_tailed_returns():
    """Mertens' Var(SR̂) for IID non-normal returns, against Monte Carlo."""
    rng = np.random.default_rng(1)
    n, sims = 500, 6000
    # Negatively skewed (−√2), kurtosis 6, per-period SR ≈ 0.35: the
    # non-normal terms raise Var(SR̂) by ~65% over the normal 1/(n − 1).
    x = 2.5 - rng.gamma(2.0, 1.0, size=(sims, n))
    srs = x.mean(axis=1) / x.std(axis=1, ddof=1)
    sr_true = 0.5 / np.sqrt(2.0)
    v = sh.sharpe_variance(sr_true, n, -np.sqrt(2.0), 6.0)
    assert srs.var(ddof=1) == pytest.approx(v, rel=0.06)
    assert srs.var(ddof=1) > 1.4 / (n - 1)          # the normal-theory variance is far off
    pooled = sh.sample_moments(x.ravel())
    assert pooled["skew"] == pytest.approx(-np.sqrt(2.0), abs=0.02)
    assert pooled["kurtosis"] == pytest.approx(6.0, abs=0.1)


def test_psr_is_calibrated_under_the_null():
    """Under SR = SR*, PSR is ~Uniform(0, 1): P(PSR > 0.95) ≈ 5%."""
    rng = np.random.default_rng(2)
    n, sims = 500, 3000
    x = rng.standard_t(df=6, size=(sims, n)) * 0.01     # true SR = 0
    psr = np.array([sh.probabilistic_sharpe_ratio(
        sh.sharpe(r), n, *(lambda m: (m["skew"], m["kurtosis"]))(sh.sample_moments(r))) for r in x])
    assert (psr > 0.95).mean() == pytest.approx(0.05, abs=0.015)
    assert (psr > 0.50).mean() == pytest.approx(0.50, abs=0.04)


def test_psr_basic_properties():
    assert sh.probabilistic_sharpe_ratio(0.05, 1000, 0.0, 3.0, 0.05) == pytest.approx(0.5)
    lo = sh.probabilistic_sharpe_ratio(0.02, 500, 0.0, 3.0)
    hi = sh.probabilistic_sharpe_ratio(0.02, 2000, 0.0, 3.0)
    assert 0.5 < lo < hi < 1.0
    # negative skew and fat tails widen the SR distribution → lower PSR at SR > 0
    assert (sh.probabilistic_sharpe_ratio(0.05, 1000, -3.0, 50.0)
            < sh.probabilistic_sharpe_ratio(0.05, 1000, 0.0, 3.0))


def test_min_track_record_length_is_where_psr_reaches_the_confidence_level():
    sr, sk, ku = 0.04, -1.2, 9.0
    n_star = sh.min_track_record_length(sr, sk, ku, 0.0, alpha=0.05)
    assert sh.probabilistic_sharpe_ratio(sr, n_star, sk, ku, 0.0) == pytest.approx(0.95, abs=1e-12)
    assert sh.probabilistic_sharpe_ratio(sr, n_star * 0.9, sk, ku, 0.0) < 0.95
    assert sh.min_track_record_length(-0.01, 0.0, 3.0) == float("inf")


@pytest.mark.parametrize("n_trials", [10, 100, 1000])
def test_expected_max_sharpe_matches_the_simulated_maximum(n_trials):
    """E[max of N IID N(0, V)] — the DSR benchmark — against Monte Carlo."""
    rng = np.random.default_rng(3)
    V = 0.0004
    mx = (rng.normal(0.0, np.sqrt(V), size=(20_000, n_trials))).max(axis=1).mean()
    assert sh.expected_max_sharpe(n_trials, V) == pytest.approx(mx, rel=0.03)


def test_deflated_sharpe_ratio_deflates_with_more_trials_and_more_dispersion():
    rng = np.random.default_rng(4)
    r = rng.normal(0.0006, 0.01, 1800)
    base = sh.deflated_sharpe_ratio(r, 1, 0.001)
    psr = sh.probabilistic_sharpe_ratio(sh.sharpe(r), r.size,
                                        *[sh.sample_moments(r)[k] for k in ("skew", "kurtosis")])
    assert base["dsr"] == pytest.approx(psr)          # one trial: no deflation
    d10 = sh.deflated_sharpe_ratio(r, 10, 1e-4)["dsr"]
    d100 = sh.deflated_sharpe_ratio(r, 100, 1e-4)["dsr"]
    d100w = sh.deflated_sharpe_ratio(r, 100, 4e-4)["dsr"]
    assert base["dsr"] > d10 > d100 > d100w


def test_effective_trials():
    assert sh.effective_trials(40, 0.0) == pytest.approx(40)
    assert sh.effective_trials(40, 1.0) == pytest.approx(1)
    assert sh.effective_trials(40, 0.5) == pytest.approx(20.5)
    assert sh.effective_trials(40, -0.3) == pytest.approx(40)      # clipped


def test_newey_west_tstat():
    rng = np.random.default_rng(5)
    x = rng.normal(0.1, 1.0, 5000)
    t0 = sh.newey_west_tstat(x, lags=0)
    classic = x.mean() / (x.std(ddof=0) / np.sqrt(x.size))
    assert t0["t"] == pytest.approx(classic, rel=1e-12)
    # positively autocorrelated series: HAC SE larger than the IID one
    e = rng.normal(size=5000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, e.size):
        ar[i] = 0.7 * ar[i - 1] + e[i]
    assert sh.newey_west_tstat(ar, lags=30)["se"] > 2.0 * sh.newey_west_tstat(ar, lags=0)["se"]
    assert sh.newey_west_tstat(x, lags=5)["p_two_sided"] == pytest.approx(
        2 * norm.sf(abs(sh.newey_west_tstat(x, lags=5)["t"])))


@pytest.mark.parametrize("lags", [5, 21])
def test_newey_west_matches_statsmodels_hac(lags):
    """The Bartlett weights at lags > 0, against statsmodels' HAC standard
    error of a regression on a constant (no small-sample correction): the
    published newey_west_mean_return_t_lag21 is this statistic."""
    import statsmodels.api as smapi

    rng = np.random.default_rng(11)
    e = rng.normal(size=3000)
    ar = np.empty_like(e)
    ar[0] = e[0]
    for i in range(1, e.size):
        ar[i] = 0.4 * ar[i - 1] + e[i]
    x = ar * 0.01 + 0.002
    fit = smapi.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags,
                                                                     "use_correction": False})
    nw = sh.newey_west_tstat(x, lags)
    assert nw["se"] == pytest.approx(float(fit.bse[0]), rel=1e-10)
    assert nw["t"] == pytest.approx(float(fit.tvalues[0]), rel=1e-10)


def test_sharpe_inference_bundle():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0005, 0.01, 2000)
    inf = sh.sharpe_inference(r)
    assert inf.sr_annual == pytest.approx(inf.sr * np.sqrt(252))
    assert inf.min_trl_years == pytest.approx(inf.min_trl_periods / 252)
    assert 0 < inf.psr_gt_0 < 1
    d = inf.to_dict()
    assert set(d) >= {"sr", "se", "psr_gt_0", "min_trl_periods"}
