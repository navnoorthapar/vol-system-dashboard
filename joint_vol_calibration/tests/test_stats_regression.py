"""OLS with Newey-West standard errors (stats/regression.py)."""

import numpy as np
import pytest

from joint_vol_calibration.stats.regression import ols_hac


def test_coefficients_and_r2_are_ordinary_least_squares():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = 0.3 + x @ np.array([1.5, -0.7]) + rng.normal(scale=0.5, size=500)
    res = ols_hac(y, x, lags=0)
    X = np.column_stack([np.ones(500), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(res["coef"], b, rtol=1e-10)
    u = y - X @ b
    assert res["r2"] == pytest.approx(1 - (u @ u) / ((y - y.mean()) @ (y - y.mean())))


def test_zero_lag_hac_is_white_and_lags_widen_it_for_autocorrelated_errors():
    rng = np.random.default_rng(1)
    n = 3000
    x = np.cumsum(rng.normal(size=n)) * 0.0 + rng.normal(size=n)
    e = np.empty(n)
    e[0] = rng.normal()
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    # the regressor is persistent too, so the score x·e is autocorrelated
    xp = np.empty(n)
    xp[0] = 0.0
    for i in range(1, n):
        xp[i] = 0.8 * xp[i - 1] + x[i]
    y = 1.0 + 0.5 * xp + e
    r0 = ols_hac(y, xp, lags=0)
    X = np.column_stack([np.ones(n), xp])
    u = y - X @ np.array(r0["coef"])
    XtXi = np.linalg.inv(X.T @ X)
    white = XtXi @ (X.T * u ** 2) @ X @ XtXi
    np.testing.assert_allclose(r0["se"], np.sqrt(np.diag(white)), rtol=1e-8)
    r30 = ols_hac(y, xp, lags=30)
    assert r30["se"][1] > 1.8 * r0["se"][1]


def test_nan_rows_dropped_and_too_few_rows_rejected():
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.5])
    x = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0])
    res = ols_hac(y, x, lags=1)
    assert res["n"] == 4
    with pytest.raises(ValueError, match="not enough observations"):
        ols_hac(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
