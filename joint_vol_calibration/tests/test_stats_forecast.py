"""
Forecast evaluation (stats/forecast.py): losses, look-ahead-free walk-forward
baselines, the GARCH plumbing against a hand-written recursion, the DM test
against statsmodels and against its own nominal size, and the MCS.
"""

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.stats import forecast as fc


def _garch_returns(n=900, seed=0):
    rng = np.random.default_rng(seed)
    w, a, b = 2e-6, 0.08, 0.9
    v = np.empty(n)
    r = np.empty(n)
    v[0] = w / (1 - a - b)
    for t in range(n):
        if t:
            v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
        r[t] = np.sqrt(v[t]) * rng.standard_normal()
    return pd.Series(r, index=pd.bdate_range("2015-01-02", periods=n))


def test_losses():
    y = np.array([0.04, 0.09, 0.01])
    assert np.allclose(fc.qlike(y, y), 0.0)
    assert np.all(fc.qlike(y, y * 1.3) > 0)
    assert np.all(fc.qlike(y, y * 0.7) > 0)
    assert np.allclose(fc.qlike(3 * y, 3 * y * 1.2), fc.qlike(y, y * 1.2))     # scale-free
    # QLIKE punishes under-prediction more than the same relative over-prediction
    assert fc.qlike(y, 0.5 * y)[0] > fc.qlike(y, 1.5 * y)[0]
    assert np.allclose(fc.mse(y, y + 0.01), 1e-4)


def test_target_is_the_next_h_sessions():
    r = pd.Series(np.arange(1.0, 11.0), index=pd.bdate_range("2020-01-01", periods=10))
    y = fc.target_variance(r, horizon=3, annualise=1)
    assert y.iloc[0] == pytest.approx((2 ** 2 + 3 ** 2 + 4 ** 2) / 3)
    assert y.iloc[-3:].isna().all()
    assert y.iloc[:-3].notna().all()


@pytest.mark.parametrize("builder", ["har", "vix_mz", "garch"])
def test_walk_forward_baselines_ignore_the_future(builder):
    """Poison every return after session p: forecasts before p must not move."""
    r = _garch_returns()
    vix = pd.Series(np.sqrt(r.rolling(10, min_periods=1).var().bfill() * 252) * 100 + 3, index=r.index)
    p = 700
    r2 = r.copy()
    r2.iloc[p:] *= 4.0
    vix2 = vix.copy()
    vix2.iloc[p:] *= 2.0
    run = {
        "har": lambda x, v: fc.har_forecast(x, train_min=300),
        "vix_mz": lambda x, v: fc.vix_mz_forecast(x, v, train_min=300),
        "garch": lambda x, v: fc.garch_forecast(x, train_min=500, refit_every=21),
    }[builder]
    a, b = run(r, vix), run(r2, vix2)
    assert a.iloc[:p].notna().sum() > 100
    np.testing.assert_array_equal(a.iloc[:p].to_numpy(), b.iloc[:p].to_numpy())
    assert not np.allclose(a.iloc[p + 25:].to_numpy(), b.iloc[p + 25:].to_numpy())


def test_garch_forecast_is_the_summed_analytic_variance_path():
    """garch_forecast's units and timing, checked against the GARCH(1,1)
    recursion written out by hand with the parameters arch estimates."""
    from arch import arch_model

    r = _garch_returns(seed=1)
    h, s = 21, 600
    f = fc.garch_forecast(r, horizon=h, train_min=s, refit_every=21)
    x = r.to_numpy() * 100
    params = arch_model(x[:s + 1], mean="Zero", vol="GARCH", p=1, o=0, q=1).fit(
        disp="off", show_warning=False).params
    w, a, b = params["omega"], params["alpha[1]"], params["beta[1]"]
    fixed = arch_model(x[:s + 21], mean="Zero", vol="GARCH", p=1, o=0, q=1).fix(params.to_numpy())
    sig2 = fixed.conditional_volatility ** 2           # σ²_t | t-1 on each row
    for t in (s, s + 7, s + 20):
        nxt = w + a * x[t] ** 2 + b * sig2[t]          # σ²_{t+1|t}
        path, cur = [], nxt
        for _ in range(h):
            path.append(cur)
            cur = w + (a + b) * cur
        assert f.iloc[t] == pytest.approx(np.sum(path) / 1e4 * 252 / h, rel=1e-9)
    assert f.iloc[:s].isna().all()


def test_dm_statistic_matches_statsmodels():
    st = pytest.importorskip("statsmodels.tsa.stattools")
    if not hasattr(st, "diebold_mariano_test"):
        pytest.skip("statsmodels < 0.15 has no diebold_mariano_test")
    rng = np.random.default_rng(3)
    n = 800
    y = rng.normal(size=n)
    fa = y + rng.normal(scale=1.0, size=n)
    fb = y + rng.normal(scale=1.1, size=n)
    la, lb = (y - fa) ** 2, (y - fb) ** 2
    ours = fc.dm_test(la, lb, horizon=1, kernel="bartlett", lags=6, harvey=False)
    ref = st.diebold_mariano_test(y, fa, fb, lags=6, criterion="mse")
    assert ours["stat"] == pytest.approx(float(ref.statistic), rel=1e-3)
    assert ours["p_two_sided"] == pytest.approx(float(ref.pvalue), rel=1e-2, abs=1e-4)


def test_dm_size_with_overlapping_targets():
    """Two equally accurate 5-step forecasts with MA(4) errors: the HLN DM
    test rejects at the 5% level about 5% of the time, not far more."""
    rng = np.random.default_rng(4)
    h, n, sims = 5, 600, 300
    rej = 0
    for _ in range(sims):
        e1 = np.convolve(rng.normal(size=n + h), np.ones(h), "valid")[:n]
        e2 = np.convolve(rng.normal(size=n + h), np.ones(h), "valid")[:n]
        rej += fc.dm_test(e1 ** 2, e2 ** 2, horizon=h)["p_two_sided"] < 0.05
    assert rej / sims < 0.10


def test_dm_detects_a_better_forecast():
    rng = np.random.default_rng(5)
    y = rng.chisquare(3, 2000) / 3
    good = np.full_like(y, 1.0)
    bad = np.full_like(y, 2.0)
    res = fc.dm_test(fc.qlike(y, good), fc.qlike(y, bad), horizon=1)
    assert res["stat"] < -5
    assert res["p_a_better"] < 1e-6


def test_mcs_excludes_a_clearly_worse_model():
    rng = np.random.default_rng(6)
    n = 1500
    base = rng.chisquare(2, n)
    L = pd.DataFrame({"a": base + rng.normal(0, 0.1, n), "b": base + rng.normal(0, 0.1, n),
                      "bad": base + 0.8 + rng.normal(0, 0.1, n)})
    res = fc.mcs(L, size=0.10, block_size=5, reps=500, seed=1)
    assert "bad" in res["excluded"]
    assert {"a", "b"} <= set(res["included"])


def test_evaluate_ranks_the_true_conditional_variance_first():
    """The model's own h-step conditional variance beats a noisy copy of it
    and a constant (the sample mean of the target, i.e. with hindsight)."""
    n, h = 1600, 21
    rng = np.random.default_rng(7)
    w, a, b = 2e-6, 0.08, 0.9
    v, r = np.empty(n), np.empty(n)
    v[0] = w / (1 - a - b)
    for t in range(n):
        if t:
            v[t] = w + a * r[t - 1] ** 2 + b * v[t - 1]
        r[t] = np.sqrt(v[t]) * rng.standard_normal()
    idx = pd.bdate_range("2015-01-02", periods=n)
    r = pd.Series(r, index=idx)
    nxt = w + a * r.to_numpy() ** 2 + b * v                  # σ²_{t+1|t}
    phi, vbar = a + b, w / (1 - a - b)
    oracle = (vbar + (nxt - vbar) * (1 - phi ** h) / ((1 - phi) * h)) * 252
    y2 = fc.target_variance(r)
    F = pd.DataFrame({"oracle": oracle, "noisy": oracle * np.exp(rng.normal(0, 0.6, n)),
                      "flat": float(y2.mean())}, index=idx)
    ev = fc.evaluate(F, y2, horizon=h, reference="oracle", mcs_reps=300)
    assert ev["rank"]["qlike"]["oracle"] == 1
    assert ev["losses"]["qlike"]["oracle"] < ev["losses"]["qlike"]["noisy"]
    assert ev["dm"]["qlike"]["noisy"]["rectangular_hln"]["stat"] < 0
    assert "noisy" in ev["mcs"]["qlike"]["90"]["excluded"]
    assert set(ev["mcs"]["qlike"]) == {"90", "75"}
    assert ev["n"] == len(pd.concat([F, y2], axis=1).dropna())
    assert len(fc.non_overlapping(F, h)) == int(np.ceil(len(F) / h))
