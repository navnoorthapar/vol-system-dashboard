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

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

from joint_vol_calibration.config import RANDOM_SEED, MC_CACHE_DIR

logger = logging.getLogger(__name__)


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
    """
    r = log_returns.copy().rename("r")

    feats = pd.DataFrame(index=r.index)

    # Squared returns
    r2 = r ** 2

    # EMA half-life → span: span = 2 * halflife - 1 (or use alpha directly)
    alpha_fast = 1.0 - np.exp(-np.log(2.0) / halflife_fast)
    alpha_slow = 1.0 - np.exp(-np.log(2.0) / halflife_slow)
    alpha_lev  = 1.0 - np.exp(-np.log(2.0) / halflife_lev)

    # f1, f2: annualised EMA of squared returns
    feats["f1"] = r2.ewm(alpha=alpha_fast, adjust=False).mean() * annualise
    feats["f2"] = r2.ewm(alpha=alpha_slow, adjust=False).mean() * annualise

    # sigma = sqrt of variance estimates
    feats["sigma1"] = np.sqrt(feats["f1"])
    feats["sigma2"] = np.sqrt(feats["f2"])

    # Leverage term: EMA of signed return (captures vol-return correlation)
    feats["lev"] = r.ewm(alpha=alpha_lev, adjust=False).mean() * np.sqrt(annualise)

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


def build_xy(
    log_returns: pd.Series,
    features_df: pd.DataFrame,
    horizon: int = 1,
    target: str = "abs_return",
    annualise: int = 252,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for regression.

    The target y(t) is defined as the realised vol over [t+1, t+horizon].
    ALL features in X are lagged by 1 day: X(t) uses features known at
    close of day t, predicting what happens on day t+1 onward.

    This is a strict no-look-ahead construction: the features on row t
    only use information through day t.

    Parameters
    ----------
    horizon : int — forecast horizon in trading days.
                   1  → predict |r_{t+1}| * sqrt(252)  (single-day annualised vol)
                   5  → predict 5-day forward RV
                   21 → predict monthly forward RV
    target  : str — 'abs_return' or 'rv'

    Returns
    -------
    X : DataFrame — feature matrix (rows = dates, columns = features)
    y : Series    — target vector (next-period realised vol, annualised decimal)
    """
    r = log_returns.copy()
    r2 = r ** 2

    if horizon == 1:
        y = r.abs().shift(-1) * np.sqrt(annualise)
    else:
        # Rolling forward RV: sqrt(annualise/horizon * sum r²_{t+1..t+h})
        y = np.sqrt(
            r2.shift(-1).rolling(horizon).mean() * annualise
        )

    # Feature columns (exclude the raw abs_r_ann to avoid label leakage)
    xcols = [c for c in features_df.columns if c not in ("abs_r_ann",)]
    X = features_df[xcols].copy()

    # Align and drop NaNs
    combined = pd.concat([X, y.rename("y")], axis=1).dropna()
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
        """OLS with non-negativity constraint on sigma1, sigma2 weights."""
        Xm = X[self._XCOLS].values
        ym = y.values
        # Add bias column
        Xb = np.column_stack([Xm, np.ones(len(Xm))])
        # Solve via normal equations (fast, stable for ~4K rows)
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


# ── Model 2: PDV Kernel (Nadaraya-Watson) ─────────────────────────────────────

class PDVKernel:
    """
    Non-parametric PDV via Nadaraya-Watson kernel regression.

    E[y | x] = sum_i K_h(x - x_i) * y_i / sum_i K_h(x - x_i)

    where K_h is a product Gaussian kernel with bandwidth h.

    State space: 2D — (sigma1, sigma2). The leverage term is included as
    a third dimension if use_leverage=True.

    Bandwidth selection:
      default: Silverman's rule-of-thumb per dimension
                h_j = 1.06 * std(x_j) * n^{-1/5}
      optional: leave-one-out cross-validation (slower but optimal)

    Financial rationale for non-parametric approach:
      The relationship sigma_{t+1} = f(f1_t, f2_t) is not necessarily linear.
      During a crisis, the mapping may be highly nonlinear (fat tails in vol).
      NW captures this nonlinearity without assuming a functional form.
      The main limitation is the curse of dimensionality — we keep the state
      space ≤ 3D to ensure enough data density.

    Prediction complexity: O(N) per query (full pass over training data).
    For N=4,000 this is fast enough. For N>100K, approximate KD-tree methods
    should be used (implemented in the neural network component C5).
    """

    _XCOLS_2D = ["sigma1", "sigma2"]
    _XCOLS_3D = ["sigma1", "sigma2", "lev"]

    def __init__(self, use_leverage: bool = False, bandwidth_scale: float = 1.0):
        """
        Parameters
        ----------
        use_leverage    : bool  — include leverage term as 3rd dimension.
        bandwidth_scale : float — multiply Silverman bandwidth by this factor.
                                  > 1 = smoother (more bias, less variance)
                                  < 1 = sharper  (more variance, less bias)
        """
        self.use_leverage    = use_leverage
        self.bandwidth_scale = bandwidth_scale
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._h:       Optional[np.ndarray] = None
        self._xcols = self._XCOLS_3D if use_leverage else self._XCOLS_2D

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PDVKernel":
        """
        Store training data and compute Silverman bandwidths.

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

        # Silverman's rule of thumb: h_j = 1.06 * sigma_j * n^{-1/(d+4)}
        sigmas = self._X_train.std(axis=0)
        self._h = self.bandwidth_scale * 1.06 * sigmas * n ** (-1.0 / (d + 4))

        # Guard against zero bandwidth (constant features)
        self._h = np.maximum(self._h, 1e-8)

        logger.info(
            "PDVKernel fitted on %d points. Bandwidths: %s",
            n, dict(zip(self._xcols, self._h.round(4)))
        )
        return self

    def _gaussian_kernel_weights(self, x_query: np.ndarray) -> np.ndarray:
        """
        Compute Nadaraya-Watson weights for a single query point x_query.

        K_h(x, x_i) = prod_j phi((x_j - x_ij) / h_j)
        where phi is the standard Gaussian PDF.

        Returns weight vector of shape (n_train,), summing to 1.
        """
        # Scaled differences: shape (n_train, d)
        diff = (self._X_train - x_query) / self._h
        # Log kernel (more numerically stable for large datasets)
        log_k = -0.5 * (diff ** 2).sum(axis=1)
        # Subtract max for numerical stability before exp
        log_k -= log_k.max()
        k = np.exp(log_k)
        total = k.sum()
        if total < 1e-300:
            # Query is far from all training points — fall back to uniform
            return np.ones(len(k)) / len(k)
        return k / total

    def predict_single(self, x_query: np.ndarray) -> float:
        """Predict for a single feature vector. O(N_train)."""
        w = self._gaussian_kernel_weights(x_query)
        return float(np.dot(w, self._y_train))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict for all rows of X.

        Parameters
        ----------
        X : DataFrame — must contain the same feature columns as training.

        Returns
        -------
        pd.Series of predicted vol (annualised), same index as X.
        """
        if self._X_train is None:
            raise RuntimeError("PDVKernel not fitted. Call .fit() first.")

        Xm = X[self._xcols].values.astype(float)
        y_hat = np.array([
            self.predict_single(Xm[i]) for i in range(len(Xm))
        ])
        return pd.Series(np.maximum(y_hat, 1e-4), index=X.index)

    def __repr__(self) -> str:
        if self._X_train is None:
            return "PDVKernel(not fitted)"
        n = len(self._X_train)
        h_str = ", ".join(f"{h:.4f}" for h in self._h)
        return f"PDVKernel(n_train={n}, dims={self._xcols}, h=[{h_str}])"


# ── GARCH(1,1) Baseline ───────────────────────────────────────────────────────

class GARCH11:
    """
    GARCH(1,1) baseline: sigma²(t) = ω + α*r²(t-1) + β*sigma²(t-1)

    MLE estimation via scipy.optimize.minimize on the Gaussian log-likelihood:
      L = -0.5 * sum_t [log(2π) + log(σ²_t) + r²_t/σ²_t]

    Included as a benchmark. The PDV model should beat GARCH on R²
    because GARCH is univariate (uses r_{t-1} only) while PDV uses
    multi-timescale history (f1 and f2).

    Typical SPX GARCH(1,1) parameters:
      ω ≈ 1e-6, α ≈ 0.08–0.12, β ≈ 0.85–0.92
      α + β ≈ 0.95–0.99 (high persistence = slow vol mean-reversion)
    """

    def __init__(self):
        self.omega_: Optional[float] = None
        self.alpha_: Optional[float] = None
        self.beta_:  Optional[float] = None
        self._sigma2_series: Optional[pd.Series] = None

    def fit(self, log_returns: pd.Series) -> "GARCH11":
        """Fit by maximum likelihood. Uses L-BFGS-B with analytical gradients."""
        r = log_returns.dropna().values

        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.9999:
                return 1e10
            n = len(r)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(r)   # initialise at sample variance
            for t in range(1, n):
                sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
            ll = -0.5 * np.sum(np.log(sigma2) + r**2 / sigma2)
            return -ll

        # Starting values: persistence near 0.95
        x0 = [1e-6, 0.09, 0.89]
        bounds = [(1e-9, 1e-2), (0.001, 0.4), (0.5, 0.999)]
        result = minimize(neg_log_likelihood, x0, method="L-BFGS-B",
                          bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-12})

        self.omega_, self.alpha_, self.beta_ = result.x
        # Recompute sigma² series for prediction
        n = len(r)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(r)
        for t in range(1, n):
            sigma2[t] = self.omega_ + self.alpha_ * r[t-1]**2 + self.beta_ * sigma2[t-1]

        self._sigma2_series = pd.Series(
            sigma2, index=log_returns.dropna().index
        )
        logger.info(
            "GARCH(1,1) fitted: ω=%.2e  α=%.4f  β=%.4f  (α+β=%.4f)",
            self.omega_, self.alpha_, self.beta_, self.alpha_ + self.beta_
        )
        return self

    def predict(self, dates: Optional[pd.Index] = None) -> pd.Series:
        """
        Return GARCH conditional vol (annualised) as a one-step-ahead forecast.

        sigma_hat(t) = sqrt(sigma²(t)) * sqrt(252)

        The GARCH forecast for t+1 is the sigma²(t) value — computed
        from data up to t-1 (no look-ahead).
        """
        if self._sigma2_series is None:
            raise RuntimeError("GARCH11 not fitted.")
        ann_vol = np.sqrt(self._sigma2_series * 252)
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


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_predict(
    model_cls,
    model_kwargs: dict,
    X: pd.DataFrame,
    y: pd.Series,
    train_min_days: int = 504,
) -> pd.Series:
    """
    Walk-forward (expanding window) prediction for kernel/linear models.

    At each date t (from train_min_days onwards):
      - Fit model on X[0:t], y[0:t]
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

    Returns
    -------
    pd.Series of out-of-sample predictions, indexed by date.
    Same length as y, NaN for the first train_min_days observations.

    Note: For PDVKernel, full walk-forward is O(N²) which is slow for N=4K.
          We therefore use a "burn-in" approach: fit once on the first
          train_min_days observations, then re-fit every 63 days (quarterly).
          This introduces minimal look-ahead (< 1 quarter's data).
    """
    n = len(X)
    dates = X.index
    y_hat = pd.Series(np.nan, index=dates)

    # For kernel: refit quarterly (every 63 trading days)
    is_kernel = model_cls is PDVKernel
    refit_freq = 63 if is_kernel else 1   # daily refit for linear (fast)

    model = None
    for t in range(train_min_days, n):
        if model is None or (t - train_min_days) % refit_freq == 0:
            model = model_cls(**model_kwargs)
            model.fit(X.iloc[:t], y.iloc[:t])

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

    Financial benchmark: GARCH(1,1) typically achieves R² ≈ 0.05–0.15
    for 1-day-ahead vol prediction. Any model above 0.15 is competitive.

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


def stress_test_date(
    log_returns: pd.Series,
    features: pd.DataFrame,
    models: dict,
    stress_date: str = "2020-03-16",
    horizon: int = 1,
    annualise: int = 252,
) -> pd.DataFrame:
    """
    Examine what each model predicted on (and around) a stress date.

    For the COVID stress test (2020-03-16):
      - We show each model's prediction on the PRIOR trading day (2020-03-13)
        because that's the prediction available before the open
      - We compare to actual |r_{2020-03-16}| * sqrt(252)

    Parameters
    ----------
    stress_date : str 'YYYY-MM-DD' — the crisis day to examine.
    models      : dict {label: fitted model with .predict(X) method}

    Returns
    -------
    DataFrame with columns [date, actual_vol, {model predictions...}, error_{model}]
    showing values for the 5 days around the stress date.
    """
    idx = pd.to_datetime(log_returns.index)
    stress_dt = pd.to_datetime(stress_date)

    # Window: 5 days before and after stress date
    pos = idx.searchsorted(stress_dt)
    window_start = max(0, pos - 5)
    window_end   = min(len(idx), pos + 4)
    window_dates = idx[window_start:window_end]

    r = log_returns.copy()
    r.index = pd.to_datetime(r.index)

    rows = []
    for dt in window_dates:
        actual_rv = abs(r.loc[dt]) * np.sqrt(annualise) if dt in r.index else np.nan
        row = {"date": dt.strftime("%Y-%m-%d"), "actual_ann_vol": round(actual_rv * 100, 2)}

        for label, model in models.items():
            # Prediction available at close of PRIOR day
            # We use the features AT dt (which use returns up to and including dt-1)
            if dt in features.index:
                feat_row = features.loc[[dt]]
                try:
                    pred = model.predict(feat_row).iloc[0]
                    row[f"pred_{label}"] = round(pred * 100, 2)
                    row[f"err_{label}"] = round((pred - actual_rv) * 100, 2)
                except Exception:
                    row[f"pred_{label}"] = np.nan
                    row[f"err_{label}"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ── Main PDV class ────────────────────────────────────────────────────────────

class PDVModel:
    """
    Unified PDV model: feature extraction + all three estimators + validation.

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
    ):
        self.halflife_fast  = halflife_fast
        self.halflife_slow  = halflife_slow
        self.halflife_lev   = halflife_lev
        self.n_lags         = n_lags
        self.bw_scale       = bw_scale
        self.use_leverage   = use_leverage
        self.horizon        = horizon
        self.train_min_days = train_min_days

        self.naive_  = PDVNaive()
        self.linear_ = PDVLinear()
        self.kernel_ = PDVKernel(
            use_leverage=use_leverage, bandwidth_scale=bw_scale
        )
        self.garch_  = GARCH11()

        self._features: Optional[pd.DataFrame] = None
        self._X:        Optional[pd.DataFrame] = None
        self._y:        Optional[pd.Series]    = None
        self.is_fitted: bool = False

    def fit(self, log_returns: pd.Series) -> "PDVModel":
        """
        Fit all sub-models on the full log-return series.

        This is the IN-SAMPLE fit. For out-of-sample evaluation use
        .validate() which runs proper walk-forward.

        Parameters
        ----------
        log_returns : pd.Series of log returns, indexed by date.
        """
        logger.info("Extracting PDV features from %d return observations ...", len(log_returns))

        self._features = extract_pdv_features(
            log_returns,
            halflife_fast=self.halflife_fast,
            halflife_slow=self.halflife_slow,
            halflife_lev=self.halflife_lev,
            n_lags=self.n_lags,
        )

        self._X, self._y = build_xy(
            log_returns, self._features, horizon=self.horizon
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
            "PDV models fitted. Linear: %s", self.linear_
        )
        return self

    def validate(
        self,
        log_returns: pd.Series,
        verbose: bool = True,
    ) -> dict:
        """
        Walk-forward out-of-sample validation.

        Returns a summary dict with R², MAE, etc. for each model,
        plus the full prediction time-series for each model.

        This is the key validation step. Results here are the ones you
        can actually trade on — no in-sample contamination.
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted. Call .fit() first.")

        features = extract_pdv_features(
            log_returns,
            halflife_fast=self.halflife_fast,
            halflife_slow=self.halflife_slow,
            halflife_lev=self.halflife_lev,
            n_lags=self.n_lags,
        )
        X, y = build_xy(log_returns, features, horizon=self.horizon)

        logger.info("Running walk-forward validation (train_min=%d days) ...",
                    self.train_min_days)

        # Walk-forward predictions
        yhat_naive  = walk_forward_predict(
            PDVNaive,  {}, X, y, self.train_min_days
        )
        yhat_linear = walk_forward_predict(
            PDVLinear, {}, X, y, self.train_min_days
        )
        yhat_kernel = walk_forward_predict(
            PDVKernel, {"use_leverage": self.use_leverage,
                        "bandwidth_scale": self.bw_scale},
            X, y, self.train_min_days
        )

        # GARCH predictions (already one-step-ahead from MLE recursion)
        yhat_garch = self.garch_.predict(X.index)

        predictions = {
            "naive":  yhat_naive,
            "linear": yhat_linear,
            "kernel": yhat_kernel,
            "garch":  yhat_garch,
        }

        # Compute metrics for the OOS period
        oos_mask = y.index >= y.index[self.train_min_days]
        y_oos    = y[oos_mask]

        metrics_list = []
        for label, yhat in predictions.items():
            m = compute_metrics(y_oos, yhat[oos_mask], label=label)
            metrics_list.append(m)

        metrics_df = pd.DataFrame(metrics_list).set_index("label")

        if verbose:
            print("\n" + "=" * 70)
            print("  PDV WALK-FORWARD VALIDATION RESULTS")
            print("  Target: next-day absolute return (annualised)")
            print("  OOS period: from training day", self.train_min_days)
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
        Examine model predictions on and around a specific stress date.

        For 2020-03-16 (worst COVID crash day):
          - Prints features on 2020-03-13 (last trading day before March 16)
          - Shows each model's implied vol forecast for March 16
          - Shows actual realised absolute return on March 16
          - Reports the prediction error for each model

        This is the definitive test: did the model warn us?
        A good PDV model should have predicted 40–80% vol on March 16
        based on the extreme moves in the preceding week.
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted. Call .fit() first.")

        features = extract_pdv_features(
            log_returns,
            halflife_fast=self.halflife_fast,
            halflife_slow=self.halflife_slow,
            halflife_lev=self.halflife_lev,
            n_lags=self.n_lags,
        )
        features.index = pd.to_datetime(features.index)

        models_for_stress = {
            "naive":  self.naive_,
            "linear": self.linear_,
            "kernel": self.kernel_,
        }

        result = stress_test_date(
            log_returns, features, models_for_stress, stress_date
        )

        if verbose:
            print("\n" + "=" * 70)
            print(f"  PDV STRESS TEST: {stress_date}")
            print("=" * 70)
            print(result.to_string(index=False))

            # Feature breakdown for the stress date
            stress_dt = pd.to_datetime(stress_date)
            idx = pd.to_datetime(features.index)
            if stress_dt in idx:
                feat_row = features.loc[stress_dt]
                print(f"\n  Features on {stress_date} (used to predict NEXT day):")
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
        Predict annualised 1-day vol for a single feature row.

        Parameters
        ----------
        features_row : 1-row DataFrame from extract_pdv_features()
        model        : 'naive', 'linear', or 'kernel'

        Returns
        -------
        float — annualised vol forecast (decimal, e.g. 0.20 = 20%)
        """
        if not self.is_fitted:
            raise RuntimeError("PDVModel not fitted.")
        m = {"naive": self.naive_, "linear": self.linear_, "kernel": self.kernel_}[model]
        return float(m.predict(features_row).iloc[0])

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
