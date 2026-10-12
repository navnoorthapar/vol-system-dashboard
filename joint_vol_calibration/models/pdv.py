"""
pdv.py — Path-Dependent Volatility (PDV) Model
            Guyon & Lekeufack, "Volatility is (Mostly) Path-Dependent" (2023)

Core claim: instantaneous vol is a deterministic function of recent realized
returns — not a hidden stochastic factor.

    sigma(t) = f(r_{t-1}, r_{t-2}, ..., r_{t-L})

In the Guyon-Lekeufack formulation, f is parameterised via two "path-dependent"
state variables:

    f1(t) = EMA_{fast}( r² )   — short-memory variance component (halflife ~5d)
    f2(t) = EMA_{slow}( r² )   — long-memory variance component (halflife ~60d)

The conditional expected vol is then estimated non-parametrically by
Nadaraya-Watson kernel regression.

This module provides three estimators:

  1. PDVNaive      — rolling historical vol, baseline (no params)
  2. PDVLinear     — OLS on (f1, f2, leverage) — interpretable, competitive
  3. PDVKernel     — Nadaraya-Watson in 2D (f1, f2) — the main model

All three share the same feature extraction, fit interface, and validation
framework so results are directly comparable.

Financial interpretation of the features:
  f1 (fast): reacts to yesterday's big move. If the market dropped 5% today,
             f1 spikes. This drives short-horizon vol clustering (GARCH effect).
  f2 (slow): reacts to the sustained regime shift. The COVID crisis (Feb-Mar 2020)
             elevated f2 for months, not days.
  leverage:  EMA of r (not r²). Negative leverage → negative recent drift →
             vol tends to rise (the leverage effect / volatility feedback).

The Heston model needs BOTH f1 and f2 as STOCHASTIC factors (v_t).
PDV replaces that hidden stochasticity with OBSERVED history. That is the
entire conceptual difference.
"""

import copy
import logging
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import pearsonr


logger = logging.getLogger(__name__)

# Number of leading feature rows dropped from every training set. The slow
# EWMA (60-day half-life) needs about one half-life of data before it is a
# meaningful 60-day average; rows before that are estimated from too few
# returns. See extract_pdv_features() and build_xy().
EWM_BURN_IN = 60



# ── Feature Engineering ───────────────────────────────────────────────────────

def extract_pdv_features(
    log_returns: pd.Series,
    halflife_fast: float = 5.0,
    halflife_slow: float = 60.0,
    halflife_lev:  float = 10.0,
    n_lags: int = 5,
    annualise: int = 252,
) -> pd.DataFrame:
    """
    Compute the PDV state variables from a log-return series.

    Parameters
    ----------
    log_returns    : pd.Series indexed by date, values = ln(S_t / S_{t-1})
    halflife_fast  : float — EMA half-life for fast variance (days). Default 5.
    halflife_slow  : float — EMA half-life for slow variance (days). Default 60.
    halflife_lev   : float — EMA half-life for leverage term (days). Default 10.
    n_lags         : int   — number of raw lagged returns to include (default 5).
    annualise      : int   — trading days per year for annualisation.

    Returns
    -------
    DataFrame with columns:
      f1      : fast EMA of r² (annualised daily variance)
      f2      : slow EMA of r² (annualised daily variance)
      sigma1  : sqrt(f1) — fast vol estimate
      sigma2  : sqrt(f2) — slow vol estimate
      lev     : EMA of r (leverage indicator; negative = recent down-drift)
      r_lag1 … r_lag{n_lags} : raw lagged returns
      rv_hist_5d   : 5-day rolling RV (sqrt of mean r² over last 5 days)
      rv_hist_20d  : 20-day rolling RV
      ts_slope     : sigma2 - sigma1 (term structure of PDV; > 0 = contango)

    All variance quantities are annualised (multiply by `annualise`).
    LOOK-AHEAD: every feature on row t uses only information from t and earlier.
                f1(t) is computed from r_1,...,r_t — no r_{t+1} ever used.

    START-UP: the EWMAs are the bias-free finite-sample form (pandas
    ``adjust=True``): f(t) = sum_{i<=t} w^(t-i) x_i / sum_{i<=t} w^(t-i). The
    recursive form (``adjust=False``) seeds the average with the FIRST squared
    return alone and lets that single value carry weight w^t — with a 60-day
    half-life the seed still held 25% of the weight after 120 days, which
    biased the first ~6 months of sigma2 (0.087 at row 20 vs a trailing
    60-day RMS of 0.161 on SPX 2010). Both forms coincide once w^t is
    negligible. Early rows are still estimated from few returns, so training
    sets built with build_xy() drop the first EWM_BURN_IN rows.
    """
    r = log_returns.copy().rename("r")

    feats = pd.DataFrame(index=r.index)

    # Squared returns
    r2 = r ** 2

    # EMA half-life → per-step weight alpha = 1 − 2^(−1/halflife)
    alpha_fast = 1.0 - np.exp(-np.log(2.0) / halflife_fast)
    alpha_slow = 1.0 - np.exp(-np.log(2.0) / halflife_slow)
    alpha_lev  = 1.0 - np.exp(-np.log(2.0) / halflife_lev)

    # f1, f2: annualised EMA of squared returns (bias-free start-up)
    feats["f1"] = r2.ewm(alpha=alpha_fast, adjust=True).mean() * annualise
    feats["f2"] = r2.ewm(alpha=alpha_slow, adjust=True).mean() * annualise

    # sigma = sqrt of variance estimates
    feats["sigma1"] = np.sqrt(feats["f1"])
    feats["sigma2"] = np.sqrt(feats["f2"])

    # Leverage term: EMA of signed return (captures vol-return correlation)
    feats["lev"] = r.ewm(alpha=alpha_lev, adjust=True).mean() * np.sqrt(annualise)

    # Raw lagged returns (preserve sign: negative lags matter for leverage)
    for lag in range(1, n_lags + 1):
        feats[f"r_lag{lag}"] = r.shift(lag)

    # Rolling realised vol (non-parametric vol estimators)
    feats["rv_hist_5d"]  = np.sqrt(r2.rolling(5).mean()  * annualise)
    feats["rv_hist_20d"] = np.sqrt(r2.rolling(20).mean() * annualise)

    # Term structure slope in vol-space (> 0 = market in vol-contango)
    feats["ts_slope"] = feats["sigma2"] - feats["sigma1"]

    # Annualised absolute return for reference
    feats["abs_r_ann"] = r.abs() * np.sqrt(annualise)

    return feats


def forward_target(
    log_returns: pd.Series,
    horizon: int = 1,
    target: Optional[str] = None,
    annualise: int = 252,
) -> pd.Series:
    """
    Forward realised-volatility target y(t) over the window [t+1, t+horizon].

    y(t) uses ONLY returns strictly after t, so it is the quantity a forecast
    made at the close of day t is scored against. The last `horizon` rows are
    NaN (their window is not complete).

    target
    ------
    'rv' (alias 'rms') : sqrt(annualise/h * sum_{i=1..h} r_{t+i}^2) — forward
                         RMS volatility, variance-consistent; this is the
                         target to use whenever the forecast is compared
                         with an implied volatility such as the VIX.
    'abs_return'       : sqrt(annualise) * mean_{i=1..h} |r_{t+i}| — an
                         E|r| target (for h=1: |r_{t+1}|*sqrt(252)). A model
                         fit on it forecasts E|r|, about 0.66-0.8 of the RMS
                         volatility; label it as such.
    None (default)     : 'abs_return' for horizon 1, 'rv' otherwise (the
                         historical behaviour of build_xy).
    For horizon 1 both targets are identical (|r_{t+1}|*sqrt(252)).
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if target is None:
        target = "abs_return" if horizon == 1 else "rv"
    r = log_returns.astype(float)
    if target in ("rv", "rms"):
        # shift(-h).rolling(h) at row t averages rows t+1..t+h
        m2 = (r ** 2).shift(-horizon).rolling(horizon).mean()
        return np.sqrt(m2 * annualise)
    if target == "abs_return":
        m1 = r.abs().shift(-horizon).rolling(horizon).mean()
        return m1 * np.sqrt(annualise)
    raise ValueError(f"unknown target {target!r}; use 'rv' or 'abs_return'")


def build_xy(
    log_returns: pd.Series,
    features_df: pd.DataFrame,
    horizon: int = 1,
    target: Optional[str] = None,
    annualise: int = 252,
    burn_in: int = EWM_BURN_IN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for regression.

    The target y(t) is the realised vol over the FORWARD window
    [t+1, t+horizon] (see forward_target()). X(t) holds features known at the
    close of day t, so row t pairs information through t with what happens
    on days t+1 .. t+horizon.

    This is a strict no-look-ahead construction: the features on row t only
    use information through day t, and the label on row t only uses returns
    after t. (Before C22 the horizon>1 label averaged r^2 over [t-h+2, t+1],
    i.e. h-1 of its h returns were already inside the feature information
    set; on SPX that inflated the horizon-5 walk-forward R^2 from 0.50 to
    0.81. horizon=1 was unaffected.)

    Parameters
    ----------
    horizon : int — forecast horizon in trading days.
                   1  → |r_{t+1}| * sqrt(252)            (E|r| target)
                   5  → 5-day forward RMS vol
                   21 → 21-day forward RMS vol (matches the 30-day VIX)
    target  : 'rv' | 'abs_return' | None — see forward_target().
    burn_in : int — drop the first `burn_in` rows of the feature frame, while
              the slow EWMA is still averaging only a handful of returns.
              Default EWM_BURN_IN (60 = one slow half-life). Pass 0 to keep
              every row.

    Returns
    -------
    X : DataFrame — feature matrix (rows = dates, columns = features)
    y : Series    — target vector (forward realised vol, annualised decimal)
    """
    y = forward_target(log_returns, horizon=horizon, target=target,
                       annualise=annualise)

    # Feature columns (exclude the raw abs_r_ann to avoid label leakage)
    xcols = [c for c in features_df.columns if c not in ("abs_r_ann",)]
    X = features_df[xcols].copy()
    if burn_in:
        X = X.iloc[int(burn_in):]

    # Align and drop NaNs
    combined = pd.concat([X, y.rename("y")], axis=1, join="inner").dropna()
    return combined[xcols], combined["y"]


# ── Model 0: Naive Rolling Vol ────────────────────────────────────────────────

class PDVNaive:
    """
    Baseline: predict tomorrow's vol = today's 20-day rolling realised vol.

    This is the simplest conceivable benchmark. Any useful model must beat it.
    Financial meaning: "vol clusters but I don't know anything about regime shifts."
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PDVNaive":
        return self   # no parameters to fit

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X["rv_hist_20d"].copy()

    def __repr__(self) -> str:
        return "PDVNaive(20d rolling vol)"


# ── Model 1: PDV Linear (OLS) ─────────────────────────────────────────────────

class PDVLinear:
    """
    Parametric PDV: OLS regression on (sigma1, sigma2, lev).

    y_hat(t) = a*sigma1(t) + b*sigma2(t) + c*lev(t) + d

    Financial interpretation:
      a : weight on fast vol component  (typically 0.3–0.6)
      b : weight on slow vol component  (typically 0.2–0.5)
      c : leverage coefficient          (typically < 0: high negative lev → high future vol)
      d : intercept (floor vol)

    This is analogous to a two-factor variance model:
      sigma²_PDV ≈ a*f1 + b*f2 + c*lev + d
    but fit in vol space (not variance) to reduce heteroscedasticity.
    """

    _XCOLS = ["sigma1", "sigma2", "lev"]

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PDVLinear":
        """Unconstrained OLS (least squares via SVD; no sign constraints).

        The forecast is floored at 1e-4 in predict(), not in the fit.
        """
        Xm = X[self._XCOLS].values
        ym = y.values
        # Add bias column
        Xb = np.column_stack([Xm, np.ones(len(Xm))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Xb, ym, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(Xb.shape[1])
        self.coef_      = coeffs[:-1]
        self.intercept_ = coeffs[-1]
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("PDVLinear not fitted. Call .fit() first.")
        Xm = X[self._XCOLS].values
        y_hat = Xm @ self.coef_ + self.intercept_
        # Vol predictions must be non-negative
        return pd.Series(np.maximum(y_hat, 1e-4), index=X.index)

    def __repr__(self) -> str:
        if self.coef_ is None:
            return "PDVLinear(not fitted)"
        a, b, c = self.coef_
        return (
            f"PDVLinear(sigma1_coef={a:.3f}, sigma2_coef={b:.3f}, "
            f"lev_coef={c:.3f}, intercept={self.intercept_:.4f})"
        )


# ── Model 1b: PDV Linear 4-Factor (OLS + VIX) ────────────────────────────────

class PDVLinear4F:
    """
    Four-factor PDV: OLS regression on (sigma1, sigma2, lev, vix).

    y_hat(t) = a*sigma1(t) + b*sigma2(t) + c*lev(t) + e*vix(t) + d

    Adds VIX (decimal, e.g. 0.20 = VIX 20) as a fourth regressor to
    capture the forward-looking volatility expectation embedded in options
    prices. VIX is far from orthogonal to the path-dependent features: an
    OLS of the same-day VIX on (σ₁, σ₂, lev) fit on 2010-2018 explains about
    85% of its variance on 2019-2025 (this is the Guyon-Lekeufack finding).
    Only the residual — market expectations about jumps and macro
    uncertainty that return EMAs cannot see — can add information, and the
    coefficient on `vix` should be read with that collinearity in mind.

    Zero look-ahead: the 'vix' column in the feature DataFrame is already
    the lagged (t-1) VIX value when features are shifted inside SignalEngine /
    build_dataset.  No extra shifting is done here.

    Usage
    -----
    Model is fit and accessed identically to PDVLinear:
        m4 = PDVLinear4F().fit(X, y)
        preds = m4.predict(X)

    The PDVModel wrapper uses PDVLinear internally; to use this model in
    compare_covid_2020() or forecast(), pass it directly to RegimePDV or
    use it standalone.
    """

    _XCOLS = ["sigma1", "sigma2", "lev", "vix"]

    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PDVLinear4F":
        """OLS on four regressors."""
        Xm = X[self._XCOLS].values
        ym = y.values
        Xb = np.column_stack([Xm, np.ones(len(Xm))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(Xb, ym, rcond=None)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(Xb.shape[1])
        self.coef_      = coeffs[:-1]
        self.intercept_ = float(coeffs[-1])
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("PDVLinear4F not fitted. Call .fit() first.")
        Xm    = X[self._XCOLS].values
        y_hat = Xm @ self.coef_ + self.intercept_
        return pd.Series(np.maximum(y_hat, 1e-4), index=X.index)

    def __repr__(self) -> str:
        if self.coef_ is None:
            return "PDVLinear4F(not fitted)"
        a, b, c, e = self.coef_
        return (
            f"PDVLinear4F(sigma1={a:.3f}, sigma2={b:.3f}, "
            f"lev={c:.3f}, vix={e:.3f}, intercept={self.intercept_:.4f})"
        )


# ── Model 2: PDV Kernel (Nadaraya-Watson) ─────────────────────────────────────

class PDVKernel:
    """
    Non-parametric PDV via Nadaraya-Watson kernel regression.

    E[y | x] = sum_i K_h(x - x_i) * y_i / sum_i K_h(x - x_i)

    where K_h is a product Gaussian kernel with bandwidth h.

    State space: 2D — (sigma1, sigma2). The leverage term is included as
    a third dimension if use_leverage=True.

    Bandwidth: rule of thumb per dimension (no cross-validation is done),
                h_j = bandwidth_scale * 1.06 * std(x_j) * n^{-1/(d+4)}

    Out-of-support queries
    ----------------------
    NW cannot extrapolate: a query far outside the training cloud puts
    (numerically) all weight on the single nearest training row, and the
    "prediction" becomes that one day's realised target. Before C22 this hit
    crisis onsets (2020-03-13: the walk-forward kernel, last refit on data to
    2020-02-05, saw a query 17.8 bandwidths beyond its training data, ESS
    1.35, and forecast 18% vol where the linear model said 70%), and
    in-sample it let the kernel replay a row's own label. Two diagnostics are
    therefore computed for every query:
      d_min — distance to the nearest training row, in bandwidth units;
      ESS   — effective sample size 1 / sum(w_i^2) of the NW weights.
    If d_min > max_support_dist or ESS < min_ess the query is outside the
    support the kernel can average over, and the prediction falls back to an
    OLS fit on the same features and training rows (fit at .fit() time).
    The number of fallbacks of the last predict() call is in n_fallback_.

    Prediction complexity: O(N) per query (full pass over training data).
    """

    _XCOLS_2D = ["sigma1", "sigma2"]
    _XCOLS_3D = ["sigma1", "sigma2", "lev"]

    def __init__(
        self,
        use_leverage: bool = False,
        bandwidth_scale: float = 1.0,
        min_ess: float = 10.0,
        max_support_dist: float = 3.0,
    ):
        """
        Parameters
        ----------
        use_leverage     : bool  — include leverage term as 3rd dimension.
        bandwidth_scale  : float — multiply the rule-of-thumb bandwidth by this.
                                   > 1 = smoother (more bias, less variance)
                                   < 1 = sharper  (more variance, less bias)
        min_ess          : float — fall back to OLS when the effective number
                                   of training rows behind a prediction is
                                   below this.
        max_support_dist : float — fall back to OLS when the nearest training
                                   row is more than this many bandwidths away.
        """
        self.use_leverage     = use_leverage
        self.bandwidth_scale  = bandwidth_scale
        self.min_ess          = float(min_ess)
        self.max_support_dist = float(max_support_dist)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._h:       Optional[np.ndarray] = None
        self._ols:     Optional[np.ndarray] = None
        self.n_fallback_: int = 0
        self._xcols = self._XCOLS_3D if use_leverage else self._XCOLS_2D

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PDVKernel":
        """
        Store training data, compute bandwidths and the OLS fallback.

        Note: NW kernel regression is fully non-parametric — all training
        data is retained. Prediction requires a full pass over training data.
        """
        Xm = X[self._xcols].values.astype(float)
        ym = y.values.astype(float)

        # Remove any NaN rows
        mask = ~(np.isnan(Xm).any(axis=1) | np.isnan(ym))
        self._X_train = Xm[mask]
        self._y_train = ym[mask]

        n, d = self._X_train.shape

        # Rule of thumb: h_j = 1.06 * sigma_j * n^{-1/(d+4)}
        sigmas = self._X_train.std(axis=0)
        self._h = self.bandwidth_scale * 1.06 * sigmas * n ** (-1.0 / (d + 4))

        # Guard against zero bandwidth (constant features)
        self._h = np.maximum(self._h, 1e-8)

        self._ols = self._fit_ols()

        logger.info(
            "PDVKernel fitted on %d points. Bandwidths: %s",
            n, dict(zip(self._xcols, self._h.round(4)))
        )
        return self

    def _fit_ols(self) -> np.ndarray:
        """OLS (with intercept) on the kernel's own features: the fallback."""
        Xb = np.column_stack([self._X_train, np.ones(len(self._X_train))])
        coeffs, _, _, _ = np.linalg.lstsq(Xb, self._y_train, rcond=None)
        return coeffs

    def _log_kernel(self, x_query: np.ndarray) -> np.ndarray:
        """Unnormalised log kernel -0.5 * |(x_i - x) / h|^2 for every row."""
        diff = (self._X_train - x_query) / self._h
        return -0.5 * (diff ** 2).sum(axis=1)

    def _gaussian_kernel_weights(self, x_query: np.ndarray) -> np.ndarray:
        """
        Compute Nadaraya-Watson weights for a single query point x_query.

        K_h(x, x_i) = prod_j phi((x_j - x_ij) / h_j)
        where phi is the standard Gaussian PDF.

        Returns weight vector of shape (n_train,), summing to 1. The largest
        kernel value is normalised to exp(0) = 1 before summing, so the sum is
        always >= 1 and the normalisation never underflows — however far the
        query lies from the data (see support_diagnostics()).
        """
        log_k = self._log_kernel(x_query)
        k = np.exp(log_k - log_k.max())
        return k / k.sum()

    def support_diagnostics(self, x_query: np.ndarray) -> Tuple[float, float]:
        """Return (d_min in bandwidth units, ESS of the NW weights)."""
        _, d_min, ess = self._nw(np.asarray(x_query, dtype=float))
        return d_min, ess

    def _nw(self, x_query: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """NW weights plus the two support diagnostics, in one pass."""
        log_k = self._log_kernel(x_query)
        top = log_k.max()
        k = np.exp(log_k - top)
        w = k / k.sum()
        return w, float(np.sqrt(-2.0 * top)), float(1.0 / np.sum(w ** 2))

    def _in_support(self, d_min: float, ess: float) -> bool:
        return (d_min <= getattr(self, "max_support_dist", 3.0)
                and ess >= getattr(self, "min_ess", 10.0))

    def _ols_predict(self, x_query: np.ndarray) -> float:
        if getattr(self, "_ols", None) is None:      # pickles predating C22
            self._ols = self._fit_ols()
        return float(np.dot(self._ols[:-1], x_query) + self._ols[-1])

    def _predict_one(self, x_query: np.ndarray) -> Tuple[float, bool]:
        """(prediction, used_fallback) for one feature vector. O(N_train)."""
        x_query = np.asarray(x_query, dtype=float)
        w, d_min, ess = self._nw(x_query)
        if self._in_support(d_min, ess):
            return float(np.dot(w, self._y_train)), False
        return self._ols_predict(x_query), True

    def predict_single(self, x_query: np.ndarray) -> float:
        """Predict for a single feature vector. O(N_train)."""
        return self._predict_one(x_query)[0]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict for all rows of X.

        Parameters
        ----------
        X : DataFrame — must contain the same feature columns as training.

        Returns
        -------
        pd.Series of predicted vol (annualised), same index as X. Rows
        outside the kernel's support use the OLS fallback; their count is
        stored in self.n_fallback_.
        """
        if self._X_train is None:
            raise RuntimeError("PDVKernel not fitted. Call .fit() first.")

        Xm = X[self._xcols].values.astype(float)
        out = [self._predict_one(Xm[i]) for i in range(len(Xm))]
        y_hat = np.array([v for v, _ in out], dtype=float)
        self.n_fallback_ = int(sum(fb for _, fb in out))
        return pd.Series(np.maximum(y_hat, 1e-4), index=X.index)

    def __repr__(self) -> str:
        if self._X_train is None:
            return "PDVKernel(not fitted)"
        n = len(self._X_train)
        h_str = ", ".join(f"{h:.4f}" for h in self._h)
        return f"PDVKernel(n_train={n}, dims={self._xcols}, h=[{h_str}])"


# ── GARCH(1,1) Baseline ───────────────────────────────────────────────────────

def _garch_filter(
    r: np.ndarray, omega: float, alpha: float, beta: float, sigma2_0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the GARCH(1,1) variance recursion over a return array.

    Returns (sigma2, sigma2_next):
      sigma2[t]      = Var(r_t | r_0..r_{t-1})   = omega + alpha r_{t-1}^2 + beta sigma2[t-1]
      sigma2_next[t] = Var(r_{t+1} | r_0..r_t)   = omega + alpha r_t^2     + beta sigma2[t]
    with sigma2[0] = sigma2_0. sigma2_next[t] is the one-step-ahead forecast
    available at the close of day t. Implemented as a first-order linear
    filter (identical to the explicit loop, without the Python overhead).
    """
    x = omega + alpha * r ** 2
    sigma2_next, _ = lfilter([1.0], [1.0, -beta], x, zi=[beta * sigma2_0])
    sigma2 = np.concatenate([[sigma2_0], sigma2_next[:-1]])
    return sigma2, sigma2_next


class GARCH11:
    """
    GARCH(1,1) baseline: sigma²(t) = ω + α*r²(t-1) + β*sigma²(t-1)

    MLE estimation via scipy.optimize.minimize on the Gaussian log-likelihood:
      L = -0.5 * sum_t [log(2π) + log(σ²_t) + r²_t/σ²_t]

    Included as a benchmark. GARCH(1,1) is itself an exponential kernel on
    past squared returns (a single decay rate β); PDV adds a second time
    scale and a leverage term, which is where any edge over GARCH comes from.

    Timing: predict() returns, on the row of day t, the forecast of the
    variance of r_{t+1} made at the close of t (ω + α r_t² + β σ²_t) — the
    same information set and target day as the PDV forecasts. (Before C22 it
    returned σ²_t, the variance of r_t known at t−1, so GARCH was scored as a
    two-step forecast against PDV's one-step.)

    Typical SPX GARCH(1,1) parameters:
      ω ≈ 1e-6, α ≈ 0.08–0.12, β ≈ 0.85–0.92
      α + β ≈ 0.95–0.99 (high persistence = slow vol mean-reversion)
    """

    def __init__(self):
        self.omega_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.beta_:  Optional[float] = None
        self.sigma2_0_: Optional[float] = None
        self._sigma2_series: Optional[pd.Series] = None   # Var(r_t | t-1), on row t
        self._sigma2_next:   Optional[pd.Series] = None   # Var(r_{t+1} | t), on row t

    def fit(self, log_returns: pd.Series) -> "GARCH11":
        """Fit by maximum likelihood (L-BFGS-B, finite-difference gradients)."""
        r_s = log_returns.dropna()
        r = r_s.values.astype(float)
        sigma2_0 = float(np.var(r))   # initialise at the sample variance

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
                return 1e10
            sigma2, _ = _garch_filter(r, omega, alpha, beta, sigma2_0)
            ll = -0.5 * np.sum(np.log(sigma2) + r**2 / sigma2)
            return -ll

        # Starting values: persistence near 0.95
        x0 = [1e-6, 0.09, 0.89]
        bounds = [(1e-9, 1e-2), (0.001, 0.4), (0.5, 0.999)]
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})

        self.omega_, self.alpha_, self.beta_ = (float(v) for v in result.x)
        self.sigma2_0_ = sigma2_0
        sigma2, sigma2_next = _garch_filter(
            r, self.omega_, self.alpha_, self.beta_, sigma2_0
        )
        self._sigma2_series = pd.Series(sigma2, index=r_s.index)
        self._sigma2_next = pd.Series(sigma2_next, index=r_s.index)
        logger.info(
            "GARCH(1,1) fitted: ω=%.2e  α=%.4f  β=%.4f  (α+β=%.4f)",
            self.omega_, self.alpha_, self.beta_, self.alpha_ + self.beta_
        )
        return self

    def filter(
        self, log_returns: pd.Series, annualise: int = 252, horizon: int = 1,
    ) -> pd.Series:
        """
        Apply the fitted parameters to `log_returns` (e.g. a longer series)
        and return, on row t, the annualised vol forecast made with returns
        through t: for horizon 1 sqrt(σ²_{t+1|t}·252); for horizon h the RMS
        of the GARCH variance path over t+1..t+h,
            σ̄² + (σ²_{t+1|t} − σ̄²)·(1 − φ^h) / ((1 − φ)·h),  φ = α + β,
        σ̄² = ω / (1 − φ). Causal: row t depends on returns up to t only. The
        recursion starts from the variance used at fit time (sigma2_0_, a
        statistic of the training sample).
        """
        if self.omega_ is None:
            raise RuntimeError("GARCH11 not fitted.")
        r_s = log_returns.dropna()
        _, nxt = _garch_filter(r_s.values.astype(float), self.omega_,
                               self.alpha_, self.beta_, self.sigma2_0_)
        h = int(horizon)
        if h > 1:
            phi = self.alpha_ + self.beta_
            vbar = self.omega_ / (1.0 - phi)
            nxt = vbar + (nxt - vbar) * (1.0 - phi ** h) / ((1.0 - phi) * h)
        return pd.Series(np.sqrt(nxt * annualise), index=r_s.index)

    def predict(self, dates: Optional[pd.Index] = None) -> pd.Series:
        """
        Return the GARCH one-step-ahead conditional vol (annualised).

        Row t holds sqrt(σ²_{t+1|t} · 252) with σ²_{t+1|t} = ω + α r_t² + β σ²_t:
        the forecast for day t+1 made with returns through the close of t.
        Parameters are the ones from .fit() (in-sample if the fit used the
        evaluation period; see walk_forward_garch() for an OOS version).
        """
        if self._sigma2_series is None:
            raise RuntimeError("GARCH11 not fitted.")
        nxt = getattr(self, "_sigma2_next", None)
        if nxt is None:   # object pickled before C22: rebuild from sigma2_t
            s2 = self._sigma2_series
            nxt = s2.shift(-1)   # sigma2_{t+1} computed from r_t, on row t
        ann_vol = np.sqrt(nxt * 252)
        if dates is not None:
            ann_vol = ann_vol.reindex(dates)
        return ann_vol

    def __repr__(self) -> str:
        if self.omega_ is None:
            return "GARCH11(not fitted)"
        return (
            f"GARCH11(ω={self.omega_:.2e}, α={self.alpha_:.4f}, "
            f"β={self.beta_:.4f}, persistence={self.alpha_+self.beta_:.4f})"
        )


def walk_forward_garch(
    log_returns: pd.Series,
    train_min_days: int = 504,
    refit_every: int = 252,
    annualise: int = 252,
    horizon: int = 1,
) -> pd.Series:
    """
    Out-of-sample GARCH(1,1) vol forecasts (see GARCH11.filter for horizon>1).

    Parameters are re-estimated every `refit_every` rows on returns up to and
    including the refit origin, then the recursion is run forward with those
    frozen parameters. Row t holds the forecast made at the close of t for
    the next `horizon` days; it depends only on returns through t. NaN for
    the first `train_min_days` rows.
    """
    r = log_returns.dropna()
    out = pd.Series(np.nan, index=r.index)
    n = len(r)
    for s in range(train_min_days, n, int(refit_every)):
        g = GARCH11().fit(r.iloc[:s + 1])
        block_end = min(s + int(refit_every), n)
        vol = g.filter(r.iloc[:block_end], annualise=annualise, horizon=horizon)
        out.iloc[s:block_end] = vol.iloc[s:block_end].values
    return out


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_predict(
    model_cls,
    model_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_min_days: int = 504,
    horizon: int = 1,
    refit_every: int = 1,
) -> pd.Series:
    """
    Walk-forward (expanding window) prediction for kernel/linear models.

    At each row t (from train_min_days + embargo onwards):
      - Fit model on X[0:t-embargo], y[0:t-embargo]
      - Predict y_hat[t] using X[t]

    This is the proper no-look-ahead evaluation. No test data ever
    touches the model before its prediction date.

    Parameters
    ----------
    model_cls     : class — PDVNaive, PDVLinear, or PDVKernel
    model_kwargs  : dict  — kwargs passed to model constructor
    X             : full feature DataFrame (all dates)
    y             : full target Series (all dates)
    train_min_days: int   — minimum training observations before first prediction
    horizon       : int   — forecast horizon of the target (trading days). For
                            horizon h, y(t) is realised vol over [t+1, t+h], so
                            consecutive targets OVERLAP by h-1 days. We PURGE the
                            last (h-1) training rows before each fit (embargo): a
                            label y(i) is used to predict y(t) only if its window
                            ends at or before t, i.e. i <= t-h. For horizon=1
                            (the default and the headline target) embargo=0 and
                            this is the standard 1-step walk-forward. Skipping
                            this purge inflates OOS R^2 via overlapping windows
                            (Lopez de Prado, Advances in Financial ML, 2018, ch 7).
                            Rows are assumed consecutive trading days (X, y as
                            returned by build_xy, which only drops leading rows
                            and the trailing rows without a complete label).
    refit_every   : int   — refit every this many rows (default 1 = daily).
                            Between refits the model is stale, never
                            look-ahead. Fitting is cheap for every model here
                            (the kernel only stores its data; its cost is the
                            O(N) prediction, paid daily either way), so the
                            default refits daily for all models. Before C22
                            the kernel was refit only every 63 rows, which
                            left it extrapolating far outside its training
                            data at every crisis onset.

    Returns
    -------
    pd.Series of out-of-sample predictions, indexed by date.
    Same length as y, NaN for the first (train_min_days + embargo) observations.

    The no-look-ahead property is verified in test_pdv.py
    (test_walk_forward_no_lookahead_poison).
    """
    n = len(X)
    dates = X.index
    y_hat = pd.Series(np.nan, index=dates)

    embargo = max(int(horizon) - 1, 0)    # purge overlapping target windows
    refit_every = max(int(refit_every), 1)

    model = None
    start = train_min_days + embargo
    for t in range(start, n):
        # Training rows whose target window ends at or before t: iloc[:t-embargo]
        fit_end = t - embargo
        if model is None or (t - start) % refit_every == 0:
            model = model_cls(**model_kwargs)
            model.fit(X.iloc[:fit_end], y.iloc[:fit_end])

        # Predict at time t using features X[t]
        pred = model.predict(X.iloc[[t]])
        y_hat.iloc[t] = pred.iloc[0]

    return y_hat


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series,
    y_hat: pd.Series,
    label: str = "",
) -> dict:
    """
    Compute regression metrics for vol prediction quality.

    Metrics:
      R²       : coefficient of determination (higher = better)
      MAE      : mean absolute error in vol points (annualised %)
      RMSE     : root mean squared error in vol points
      Corr     : Pearson correlation(y_true, y_hat)
      MedAE    : median absolute error (robust to spikes)
      Bias     : mean(y_hat - y_true) — positive = over-prediction

    Reference points on SPX 2012-2025 (walk-forward, 1-day |r| target):
    PDVLinear R² ≈ 0.31, GARCH(1,1) ≈ 0.16 (scale-penalised; corr² 0.25),
    20-day RV ≈ 0.08 (corr² 0.21). The pooled R² of this noisy target is
    dominated by a few crisis days (≈ 0.19 without 2020).

    Parameters
    ----------
    y_true, y_hat : pd.Series — must be aligned by index. Both in annualised
                                decimal vol (e.g. 0.20 = 20% vol).
    label         : str       — model name for logging.

    Returns
    -------
    dict with keys [label, r2, mae, rmse, corr, med_ae, bias, n_obs].
    """
    # Align on common non-NaN index
    both = pd.concat([y_true.rename("true"), y_hat.rename("hat")], axis=1).dropna()
    yt = both["true"].values
    yh = both["hat"].values
    n  = len(yt)

    if n < 10:
        logger.warning("Only %d observations for metric computation", n)
        return {}

    ss_res = np.sum((yt - yh) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    mae   = np.mean(np.abs(yt - yh))
    rmse  = np.sqrt(np.mean((yt - yh) ** 2))
    corr  = pearsonr(yt, yh)[0]
    med_ae = np.median(np.abs(yt - yh))
    bias  = np.mean(yh - yt)

    metrics = dict(
        label=label, n_obs=n,
        r2=round(r2, 4), mae=round(mae * 100, 4),
        rmse=round(rmse * 100, 4), corr=round(corr, 4),
        med_ae=round(med_ae * 100, 4), bias=round(bias * 100, 4),
    )
    return metrics


def forecast_skill_comparison(
    log_returns: pd.Series,
    train_min_days: int = 504,
    annualise: int = 252,
    garch_refit_every: int = 252,
) -> dict:
    """Honest, scale-invariant comparison of next-day |return| forecast skill.

    Target: y(t) = |r_{t+1}| * sqrt(252)  (next-day annualised absolute return).

    We report squared correlation (corr^2) for each forecaster, which measures
    information content AFTER optimal linear rescaling. This matters because a
    raw 20-day realised-vol forecast runs systematically hot (RV ~ sigma, but
    E|r| = sigma*sqrt(2/pi) ~ 0.8*sigma), so its uncalibrated R^2 (1 - SSres/SStot)
    badly understates its information. PDVLinear is OLS-fit, so it gets that
    rescaling for free; comparing PDV's calibrated R^2 to a naive predictor's
    *uncalibrated* R^2 is apples-to-oranges and overstates the structural edge.

    Every forecaster is scored on the SAME rows: the out-of-sample window of
    the PDV walk-forward (before C22 the baselines were scored on the full
    sample, EWMA warm-up included, while PDV was scored on its OOS window).
    Every forecast on row t uses returns through t and targets day t+1.

    Returns a dict keyed by model name with {corr2, r2_uncalibrated, n}:
      pdv_walk_forward      — PDVLinear, daily walk-forward refit
      naive_rv20            — trailing 20-day realised vol
      ewma_riskmetrics_0p94 — RiskMetrics EWMA, lambda = 0.94
      garch11_in_sample     — GARCH(1,1), ONE fit on the whole sample
                              (parameters see the evaluation period)
      garch11_walk_forward  — GARCH(1,1) refit every `garch_refit_every` rows
                              on past data only
    The honest read is the corr2 column: PDV beats simple baselines, but by a
    modest margin over a trivial EWMA/GARCH, not the ~4x that the
    uncalibrated-naive comparison implies.
    """
    r = log_returns.dropna()
    target = r.abs().shift(-1) * np.sqrt(annualise)

    # PDVLinear, genuine walk-forward (calibrated by OLS)
    feats = extract_pdv_features(r)
    X, y = build_xy(r, feats, horizon=1)
    pdv_pred = walk_forward_predict(PDVLinear, {}, X, y,
                                    train_min_days=train_min_days, horizon=1)
    common = pdv_pred.dropna().index.intersection(target.dropna().index)
    t_c = target.reindex(common)

    def _both(f: pd.Series) -> dict:
        d = pd.concat([f.reindex(common).rename("f"), t_c.rename("t")],
                      axis=1).dropna()
        if len(d) < 100:
            return {"corr2": np.nan, "r2_uncalibrated": np.nan, "n": len(d)}
        c2 = float(np.corrcoef(d["f"], d["t"])[0, 1] ** 2)
        ss_res = float(np.sum((d["t"] - d["f"]) ** 2))
        ss_tot = float(np.sum((d["t"] - d["t"].mean()) ** 2))
        r2u = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"corr2": round(c2, 4), "r2_uncalibrated": round(r2u, 4), "n": len(d)}

    out = {"pdv_walk_forward": _both(pdv_pred)}

    # Naive: trailing 20-day realised vol (the strawman baseline; note how far
    # its uncalibrated R^2 sits below its corr^2 — that gap is pure scale bias)
    out["naive_rv20"] = _both(feats["rv_hist_20d"])

    # RiskMetrics EWMA, lambda = 0.94 (parameter-free industry standard)
    ewma94 = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean() * annualise)
    out["ewma_riskmetrics_0p94"] = _both(ewma94)

    # GARCH(1,1): one in-sample fit, and an honest walk-forward version
    try:
        g = GARCH11().fit(r)
        out["garch11_in_sample"] = _both(g.predict())
    except Exception as exc:
        logger.warning("in-sample GARCH failed: %s", exc)
        out["garch11_in_sample"] = {"corr2": np.nan, "r2_uncalibrated": np.nan, "n": 0}
    try:
        out["garch11_walk_forward"] = _both(walk_forward_garch(
            r, train_min_days=train_min_days, refit_every=garch_refit_every,
            annualise=annualise,
        ))
    except Exception as exc:
        logger.warning("walk-forward GARCH failed: %s", exc)
        out["garch11_walk_forward"] = {"corr2": np.nan, "r2_uncalibrated": np.nan, "n": 0}

    return out


def stress_test_date(
    log_returns: pd.Series,
    features: pd.DataFrame,
    models: dict,
    stress_date: str = "2020-03-16",
    horizon: int = 1,
    annualise: int = 252,
    walk_forward: bool = True,
    target: Optional[str] = None,
    burn_in: int = EWM_BURN_IN,
) -> pd.DataFrame:
    """
    What each model forecast, BEFORE the move, for the days around a stress date.

    For every target day d in the window (5 trading days before the stress
    date through 3 after):
      - the forecast origin o is the previous trading day: the prediction is
        model(features at o), and features at o use returns through o only,
        never r_d;
      - with walk_forward=True (default) each model is a copy re-fit on the
        (X, y) rows whose labels are fully observed at the close of o (label
        window ending on or before o), so neither the coefficients nor a
        kernel's stored targets know anything about d or later;
      - the forecast is compared with the realised value of the same target
        over [d, d+horizon-1]; for horizon 1 that is |r_d| * sqrt(252).

    Before C22 this function used the features AT d (which already contain
    r_d) and the full-sample models, and compared them with |r_d|: the
    published "92.4% on 2020-03-16" was the in-sample kernel replaying its own
    stored label |r_2020-03-17|.

    Parameters
    ----------
    stress_date  : str 'YYYY-MM-DD' — the crisis day to examine.
    models       : dict {label: model with .fit(X, y) and .predict(X)}. With
                   walk_forward=True only their configuration is used (they
                   are deep-copied and re-fit); with walk_forward=False they
                   are used as given — in-sample if they were fit on the full
                   sample (the result's attrs['fit'] says which).
    horizon, target, burn_in : as in build_xy().

    Returns
    -------
    DataFrame with columns [date, forecast_origin, actual_ann_vol,
    pred_{label}, err_{label}] in annualised vol percent, one row per target
    day.
    """
    r = log_returns.dropna().copy()
    r.index = pd.to_datetime(r.index)
    feats = features.copy()
    feats.index = pd.to_datetime(feats.index)
    idx = r.index
    stress_dt = pd.to_datetime(stress_date)
    horizon = int(horizon)

    y_full = forward_target(r, horizon=horizon, target=target, annualise=annualise)
    if walk_forward:
        X_all, y_all = build_xy(r, feats.reindex(idx), horizon=horizon,
                                target=target, annualise=annualise,
                                burn_in=burn_in)

    # Window: 5 days before and 3 after the stress date (target days)
    pos = idx.searchsorted(stress_dt)
    window_start = max(1, pos - 5)            # a target day needs an origin
    window_end   = min(len(idx), pos + 4)

    rows = []
    for p in range(window_start, window_end):
        d, o = idx[p], idx[p - 1]
        actual = float(y_full.loc[o]) if np.isfinite(y_full.loc[o]) else np.nan
        row = {
            "date": d.strftime("%Y-%m-%d"),
            "forecast_origin": o.strftime("%Y-%m-%d"),
            "actual_ann_vol": round(actual * 100, 2) if np.isfinite(actual) else np.nan,
        }
        feat_row = feats.loc[[o]] if o in feats.index else None
        if walk_forward and p - 1 - horizon >= 0:
            train_end = idx[p - 1 - horizon]   # last label row known at o
            tr = X_all.index <= train_end
            X_tr, y_tr = X_all[tr], y_all[tr]
        for label, model in models.items():
            pred = np.nan
            try:
                if feat_row is not None:
                    if walk_forward:
                        if p - 1 - horizon < 0 or len(X_tr) < 30:
                            raise ValueError("too little history before origin")
                        m = copy.deepcopy(model).fit(X_tr, y_tr)
                    else:
                        m = model
                    pred = float(m.predict(feat_row).iloc[0])
            except Exception as exc:
                logger.warning("stress_test_date: %s at %s failed: %s", label, o.date(), exc)
                pred = np.nan
            row[f"pred_{label}"] = round(pred * 100, 2) if np.isfinite(pred) else np.nan
            row[f"err_{label}"] = (round((pred - actual) * 100, 2)
                                   if np.isfinite(pred) and np.isfinite(actual) else np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["fit"] = "walk-forward" if walk_forward else "as given (in-sample if fit on the full sample)"
    df.attrs["horizon"] = horizon
    return df


# ── Main PDV class ────────────────────────────────────────────────────────────

def _target_description(horizon: int, target: Optional[str]) -> str:
    t = target or ("abs_return" if int(horizon) == 1 else "rv")
    if t == "abs_return":
        if int(horizon) == 1:
            return "next-day |r| x sqrt(252) (an E|r| forecast, NOT a volatility)"
        return f"mean |r| over the next {horizon} days x sqrt(252) (E|r| scale)"
    return f"forward {horizon}-day RMS volatility (variance-consistent)"


class PDVModel:
    """
    Unified PDV model: feature extraction + all estimators + validation.

    Targets
    -------
    horizon=1 (default)        : |r_{t+1}|·sqrt(252) — an E|r| forecast; this is
                                 the target of the walk-forward R^2 table.
    horizon=h, target='rv'     : forward h-day RMS vol (variance-consistent).

    Usage:
      model = PDVModel()
      model.fit(log_returns)
      results = model.validate(log_returns, verbose=True)
      stress  = model.stress_test(log_returns, "2020-03-16")
      model.save("pdv_model.pkl")
    """

    def __init__(
        self,
        halflife_fast:  float = 5.0,
        halflife_slow:  float = 60.0,
        halflife_lev:   float = 10.0,
        n_lags:         int   = 5,
        bw_scale:       float = 1.0,
        use_leverage:   bool  = True,
        horizon:        int   = 1,
        train_min_days: int   = 504,
        target:         Optional[str] = None,
        burn_in:        int   = EWM_BURN_IN,
    ):
        self.halflife_fast  = halflife_fast
        self.halflife_slow  = halflife_slow
        self.halflife_lev   = halflife_lev
        self.n_lags         = n_lags
        self.bw_scale       = bw_scale
        self.use_leverage   = use_leverage
        self.horizon        = int(horizon)
        self.train_min_days = train_min_days
        self.target         = target
        self.burn_in        = int(burn_in)

        self.naive_  = PDVNaive()
        self.linear_ = PDVLinear()
        self.kernel_ = PDVKernel(
            use_leverage=use_leverage, bandwidth_scale=bw_scale
        )
        self.garch_  = GARCH11()

        self._features:    Optional[pd.DataFrame] = None
        self._X:           Optional[pd.DataFrame] = None
        self._y:           Optional[pd.Series]    = None
        self._log_returns: Optional[pd.Series]    = None
        self.is_fitted: bool = False

    def __setstate__(self, state: dict) -> None:
        # Objects pickled before C22 lack the newer attributes.
        state.setdefault("target", None)
        state.setdefault("burn_in", 0)
        state.setdefault("_log_returns", None)
        self.__dict__.update(state)

    @property
    def target_description(self) -> str:
        return _target_description(self.horizon, self.target)

    def _extract(self, log_returns: pd.Series) -> pd.DataFrame:
        return extract_pdv_features(
            log_returns,
            halflife_fast=self.halflife_fast,
            halflife_slow=self.halflife_slow,
            halflife_lev=self.halflife_lev,
            n_lags=self.n_lags,
        )

    def fit(self, log_returns: pd.Series) -> "PDVModel":
        """
        Fit all sub-models on the full log-return series.

        This is the IN-SAMPLE fit: its coefficients have seen every date in
        `log_returns`, so its predictions for those dates are NOT forecasts.
        For out-of-sample numbers use .validate().

        Parameters
        ----------
        log_returns : pd.Series of log returns, indexed by date.
        """
        log_returns = log_returns.dropna()
        logger.info("Extracting PDV features from %d return observations ...", len(log_returns))

        self._log_returns = log_returns.copy()
        self._features = self._extract(log_returns)

        self._X, self._y = build_xy(
            log_returns, self._features, horizon=self.horizon,
            target=self.target, burn_in=self.burn_in,
        )

        logger.info(
            "Built dataset: X shape=%s, y shape=%s. "
            "Fitting all models ...",
            self._X.shape, self._y.shape
        )

        # In-sample fits
        self.naive_.fit(self._X, self._y)
        self.linear_.fit(self._X, self._y)
        self.kernel_.fit(self._X, self._y)
        self.garch_.fit(log_returns)

        self.is_fitted = True
        logger.info(
            "PDV models fitted (target: %s). Linear: %s",
            self.target_description, self.linear_
        )
        return self

    def validate(
        self,
        log_returns: pd.Series,
        verbose: bool = True,
        garch_refit_every: int = 252,
    ) -> dict:
        """
        Walk-forward out-of-sample validation.

        Every model, GARCH included, is scored on the same out-of-sample rows,
        with forecasts that only use information available at the forecast
        origin: naive / linear / kernel via walk_forward_predict() (purged for
        horizon > 1), GARCH via walk_forward_garch() (re-estimated every
        `garch_refit_every` rows). Before C22 the GARCH row was a single
        in-sample fit, mistimed by one day, printed inside this table.

        Note: GARCH and the naive 20-day RV forecast a volatility (RMS scale).
        Against the 1-day E|r| target their uncalibrated R^2 is penalised for
        scale; forecast_skill_comparison() reports the scale-free corr^2.

        Returns a summary dict with R², MAE, etc. for each model,
        plus the full prediction time-series for each model.
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted. Call .fit() first.")

        log_returns = log_returns.dropna()
        features = self._extract(log_returns)
        X, y = build_xy(log_returns, features, horizon=self.horizon,
                        target=self.target, burn_in=self.burn_in)

        logger.info("Running walk-forward validation (train_min=%d days) ...",
                    self.train_min_days)

        # Walk-forward predictions (purged for horizon > 1)
        yhat_naive  = walk_forward_predict(
            PDVNaive,  {}, X, y, self.train_min_days, horizon=self.horizon
        )
        yhat_linear = walk_forward_predict(
            PDVLinear, {}, X, y, self.train_min_days, horizon=self.horizon
        )
        yhat_kernel = walk_forward_predict(
            PDVKernel, {"use_leverage": self.use_leverage,
                        "bandwidth_scale": self.bw_scale},
            X, y, self.train_min_days, horizon=self.horizon
        )
        yhat_garch = walk_forward_garch(
            log_returns, train_min_days=self.train_min_days,
            refit_every=garch_refit_every, horizon=self.horizon,
        ).reindex(X.index)

        predictions = {
            "naive":  yhat_naive,
            "linear": yhat_linear,
            "kernel": yhat_kernel,
            "garch":  yhat_garch,
        }

        # OOS period: the rows where the walk-forward produced a forecast
        embargo  = max(self.horizon - 1, 0)
        oos_mask = y.index >= y.index[self.train_min_days + embargo]
        y_oos    = y[oos_mask]

        metrics_list = []
        for label, yhat in predictions.items():
            m = compute_metrics(y_oos, yhat[oos_mask], label=label)
            metrics_list.append(m)

        metrics_df = pd.DataFrame(metrics_list).set_index("label")

        if verbose:
            print("\n" + "=" * 70)
            print("  PDV WALK-FORWARD VALIDATION RESULTS")
            print(f"  Target: {self.target_description}")
            print(f"  OOS period: {y_oos.index[0].date()} -> {y_oos.index[-1].date()}"
                  f" ({len(y_oos)} rows)")
            print("=" * 70)
            print(metrics_df[["r2", "mae", "rmse", "corr", "bias"]].to_string())
            print()
            _print_model_interpretation(self.linear_)

        return {
            "metrics":     metrics_df,
            "predictions": predictions,
            "y_oos":       y_oos,
        }

    def stress_test(
        self,
        log_returns: pd.Series,
        stress_date: str = "2020-03-16",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Examine out-of-sample model forecasts on and around a stress date.

        For 2020-03-16 (worst COVID crash day): the forecast for March 16 is
        made at the 2020-03-13 close, from 2020-03-13 features, by models
        re-fit on labels known at that close (see stress_test_date()), and
        compared with the realised |r_2020-03-16|.

        This is the definitive test: did the model warn us?
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted. Call .fit() first.")

        log_returns = log_returns.dropna()
        features = self._extract(log_returns)
        features.index = pd.to_datetime(features.index)

        models_for_stress = {
            "naive":  self.naive_,
            "linear": self.linear_,
            "kernel": self.kernel_,
        }

        result = stress_test_date(
            log_returns, features, models_for_stress, stress_date,
            horizon=self.horizon, target=self.target, burn_in=self.burn_in,
            walk_forward=True,
        )

        if verbose:
            print("\n" + "=" * 70)
            print(f"  PDV STRESS TEST: {stress_date}  (walk-forward, prior-close origin)")
            print(f"  Target: {self.target_description}")
            print("=" * 70)
            print(result.to_string(index=False))

            # Feature breakdown at the forecast origin of the stress date
            hit = result[result["date"] == pd.to_datetime(stress_date).strftime("%Y-%m-%d")]
            if len(hit):
                origin = pd.to_datetime(hit["forecast_origin"].iloc[0])
                feat_row = features.loc[origin]
                print(f"\n  Features on {origin.date()} (forecast origin for {stress_date}):")
                key_feats = {
                    "sigma1 (fast vol, ann)": f"{feat_row['sigma1']*100:.1f}%",
                    "sigma2 (slow vol, ann)": f"{feat_row['sigma2']*100:.1f}%",
                    "leverage (EMA return)":  f"{feat_row['lev']*100:.2f}%",
                    "r_lag1":                 f"{feat_row['r_lag1']*100:.2f}%",
                    "r_lag2":                 f"{feat_row['r_lag2']*100:.2f}%",
                }
                for k, v in key_feats.items():
                    print(f"    {k:<30s}: {v}")

        return result

    def predict_vol(
        self,
        features_row: pd.DataFrame,
        model: str = "kernel",
    ) -> float:
        """
        Predict the model's target (see target_description) for one feature row.

        Uses the IN-SAMPLE fitted sub-models.

        Parameters
        ----------
        features_row : 1-row DataFrame from extract_pdv_features()
        model        : 'naive', 'linear', or 'kernel'

        Returns
        -------
        float — annualised forecast (decimal, e.g. 0.20 = 20%)
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted.")
        m = {"naive": self.naive_, "linear": self.linear_, "kernel": self.kernel_}[model]
        return float(m.predict(features_row).iloc[0])

    def describe(self) -> dict:
        """Plain-data summary of the fitted model (for dashboards / reports)."""
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted.")
        a, b, c = (float(v) for v in self.linear_.coef_)
        return {
            "horizon":        self.horizon,
            "target":         self.target_description,
            "feature_start":  str(self._features.index[0].date()),
            "feature_end":    str(self._features.index[-1].date()),
            "n_train":        int(len(self._X)),
            "linear":         {"sigma1": a, "sigma2": b, "lev": c,
                               "intercept": float(self.linear_.intercept_)},
            "garch":          {"omega": self.garch_.omega_, "alpha": self.garch_.alpha_,
                               "beta": self.garch_.beta_,
                               "persistence": self.garch_.alpha_ + self.garch_.beta_},
            "halflives_days": {"fast": self.halflife_fast, "slow": self.halflife_slow,
                               "lev": self.halflife_lev},
        }

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("PDVModel saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "PDVModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("PDVModel loaded from %s", path)
        return model


# ── Interpretation helper ─────────────────────────────────────────────────────

def _print_model_interpretation(model: PDVLinear) -> None:
    if model.coef_ is None:
        return
    a, b, c = model.coef_
    d = model.intercept_
    print("  PDVLinear coefficients (vol-space):")
    print(f"    sigma1 (fast, ~5d EWMA vol) : {a:.4f}  "
          f"{'↑' if a>0 else '↓'} short-term vol drives forecast")
    print(f"    sigma2 (slow, ~60d EWMA vol): {b:.4f}  "
          f"{'↑' if b>0 else '↓'} long-term regime matters")
    print(f"    lev    (EMA of return)       : {c:.4f}  "
          f"{'negative return → higher vol' if c<0 else 'positive return → higher vol'}")
    print(f"    intercept (floor vol)        : {d*100:.2f}%")
    total_w = a + b
    print(f"    fast/slow split              : {a/total_w*100:.0f}% / {b/total_w*100:.0f}%")

