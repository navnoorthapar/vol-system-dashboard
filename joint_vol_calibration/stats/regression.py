"""
regression.py — OLS with Newey-West (HAC) standard errors.

Used for the factor attributions (sleeve P&L on an always-on short-straddle
"VRP" factor; sleeve returns on Cboe strategy-index returns). Bartlett
kernel; the default lag is the Newey & West (1994) rule floor(4·(n/100)^(2/9)).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm


def ols_hac(y: Sequence[float], X: np.ndarray, lags: Optional[int] = None,
            add_constant: bool = True) -> Dict:
    """OLS of y on X with HAC covariance.

    Returns coef, se, t, p (two-sided, normal), r2, n, lags. With
    add_constant the first coefficient is the intercept.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_constant:
        X = np.column_stack([np.ones(len(X)), X])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    if n <= k:
        raise ValueError("not enough observations")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    g = X * u[:, None]
    S = g.T @ g / n
    for j in range(1, int(lags) + 1):
        if j >= n:
            break
        w = 1.0 - j / (lags + 1.0)
        G = g[j:].T @ g[:-j] / n
        S += w * (G + G.T)
    cov = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(u @ u) / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": beta.tolist(), "se": se.tolist(), "t": t.tolist(),
            "p": [float(2.0 * norm.sf(abs(v))) if np.isfinite(v) else float("nan") for v in t],
            "r2": r2, "n": int(n), "lags": int(lags)}
