"""
regime_pdv.py — C11: Regime-Switching PDV with Merton Jump Component

Overview
--------
Combines two components:
  • C3 PDV Linear — OLS on the path features (σ₁, σ₂, lev) of models/pdv.py,
    RE-FIT on returns strictly before the calibration date (never the
    full-sample coefficients of pdv_model.pkl, which have seen the future).
  • C8 regime labels — the deterministic rule labels stored in
    data_store/signals/regime_labels.parquet (R0=LONG_GAMMA, R1=SHORT_GAMMA,
    R2=VOMMA_ACTIVE; R2 ⇔ VVIX above its threshold at the close).

Forecast logic (a forecast made at the close of as_of_date for the next day)
---------------------------------------------------------------------------
  R0 / R1:  total_vol = σ_PDV                       (plain diffusion)
  R2:       total_vol = sqrt(σ²_PDV + λ·(μ_j² + σ_j²))

Diffusion volatility σ_PDV
--------------------------
PDVLinear is fit on |r_{t+1}|·√252, so it forecasts E|r|, not σ (on SPX
E|r| ≈ 0.66 × RMS vol). Treating it as the diffusion σ made the likelihood
fill the missing variance with "jumps". It is therefore rescaled by

    k = median_t( |r_{t+1}|·√252 / p_t ) / Φ⁻¹(0.75)

on the training sample: the scale at which the median standardised return
equals the Gaussian median (0.6745). The median ignores the rare large moves
that the jump component exists to explain; a variance-matching scale would
absorb them into the diffusion. (k ≈ 1.19 on SPX; √(π/2) = 1.25 is the
value for exactly Gaussian returns.)

The diffusion vol applied to the return of day t is the forecast made at the
close of t−1:  σ_d,t = k·p_{t−1} / √252.  Before C22 it was the prediction
from the features AT t, which already contain r_t²: every large return
inflated its own diffusion vol and hid itself from the jump likelihood.

Merton (1976) jump component
-----------------------------
    r_t = σ_d,t · ε_t + Σ_{k=1}^{N_t} J_k,   ε_t ~ N(0, 1)
    N_t ~ Poisson(λ/252),                    J_k ~ N(μ_j, σ_j²)
    Annualised jump variance = λ·(μ_j² + σ_j²)

Calibration sample: every day t strictly before as_of_date whose PREVIOUS
close carried an R2 label — the same conditioning forecast() applies (the
regime known at the forecast origin). No day is selected on its outcome.

  calibrate_jump()      — maximum likelihood for (λ, μ_j, σ_j) on that whole
                          sample, λ ∈ [0.001, 252] per year. A parameter that
                          ends on a bound is reported in `bound_active`
                          (the pre-C22 boxes λ ≤ 20 / λ ≤ 50 were always
                          binding). This is what forecast() uses by default.
  calibrate_jump_tail() — detected-jump (threshold) estimator: the R2 days
                          flagged by daily_jump_test() at level α. λ is the
                          flag frequency per R2 day × 252, and (μ_j, σ_j) are
                          the moments of the flagged returns net of the
                          diffusion variance. It describes the large,
                          statistically detectable jumps only.

Result on SPX, as_of 2026-03-24 (900 post-R2 days): the MLE is interior at
λ ≈ 61 jumps/yr, μ_j ≈ −0.25%, σ_j ≈ 1.0% — the data prefer frequent small
jumps (a fat-tailed scale mixture) over rare crashes. The profile likelihood
is flat within one log-likelihood point for λ ≈ 50-150/yr, and λ = 1-5/yr is
rejected by 18-36 points. The detected-jump estimator gives λ ≈ 15/yr,
μ_j ≈ −1.2%, σ_j ≈ 3.2% for the large jumps alone. (The pre-C22 docstring
promised λ = 2-8/yr, μ_j = −3..−8%, σ_j = 3-6%, and its fits sat on their
λ bounds of 20 and 50/yr.)

Daily jump test (replaces the pre-C22 "BN-S jump filter")
---------------------------------------------------------
The Barndorff-Nielsen-Shephard test needs many intraday returns per day; this
project has daily closes only. The old filter computed 1 − (π/2)|r_{t−1}|/|r_t|
from ONE pair of daily returns and flagged > 0.3 — it only asked whether
today's move was 2.24× yesterday's, and flagged 26.7% of days under a pure
Gaussian diffusion. daily_jump_test() is a Lee-Mykland (2008)-style daily
statistic instead:

    L_t = r_t / σ̂_{t−1},   σ̂²_{t−1} = (π/2) · mean_{j}( |r_j||r_{j−1}| )

over the `window` return pairs strictly before t (local bipower variation,
robust to an isolated jump in the window). |L_t| is compared with a critical
value calibrated by simulation under an iid Gaussian null for the same
window, so the test has size α exactly under that null (scale-free) and close
to α under a GARCH(1,1) null (0.9% at α = 1% in a 200k-day simulation).

Zero look-ahead
---------------
  • A calibration at as_of_date uses returns, regime labels and PDV
    coefficients from dates strictly before as_of_date.
  • forecast(as_of_date) additionally uses the features and the label AT
    as_of_date (information at that close); its target is the next day.
  • forecast() refuses features or labels older than `max_stale_bdays`
    business days unless allow_stale=True, and always reports the dates used.

Outputs (from .forecast())
--------------------------
  {
    'pdv_vol':        float — diffusion forecast σ_PDV (annualised, decimal)
    'jump_adj':       float — Merton jump addition in vol units (≥ 0)
    'total_vol':      float — sqrt(pdv_vol² + λ·(μ_j²+σ_j²))
    'regime':         str   — 'LONG_GAMMA' | 'SHORT_GAMMA' | 'VOMMA_ACTIVE'
    'lambda', 'mu_j', 'sigma_j' — jump parameters (NaN for R0/R1)
    'n_r2_cal_days':  int   — days used by the jump calibration (0 for R0/R1)
    'bound_active':   tuple — jump parameters that ended on a bound
    'jump_method':    str   — 'mle' or 'threshold'
    'diffusion_scale':float — k
    'as_of_date', 'feature_date', 'regime_date', 'stale'
  }

COVID comparison (compare_covid_2020)
--------------------------------------
Scores the plain and the jump-augmented forecasts made at each close of the
period against the NEXT day's realised squared return with QLIKE and squared
error (Patton 2011: both rank variance forecasts consistently when the proxy
is noisy but conditionally unbiased). Before C22 it scored them against the
BACKWARD 20-day RV, where adding any variance to a downward-biased forecast
looked like an improvement.
"""

import logging
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import poisson as scipy_poisson

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.models.pdv import (
    EWM_BURN_IN,
    PDVLinear,
    PDVModel,
    build_xy,
    extract_pdv_features,
    forward_target,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SIGNALS_DIR     = DATA_DIR / "signals"
PDV_MODEL_PATH  = DATA_DIR / "pdv_model.pkl"
REGIME_LBL_PATH = SIGNALS_DIR / "regime_labels.parquet"

# ── Constants ─────────────────────────────────────────────────────────────────

ANNUALISE    = 252          # trading days per year
N_MAX_JUMPS  = 5            # Poisson truncation for MLE density
MIN_R2_DAYS  = 30           # minimum R2 days for a jump calibration
MIN_TAIL_DAYS = 10          # minimum detected jumps for the threshold estimator
MIN_DIFFUSION_TRAIN = 252   # minimum training rows for the diffusion model

JUMP_TEST_WINDOW = 20       # return pairs in the local bipower variance
JUMP_TEST_ALPHA  = 0.01     # size of the daily jump test

REGIME_NAMES: Dict[int, str] = {
    0: "LONG_GAMMA",
    1: "SHORT_GAMMA",
    2: "VOMMA_ACTIVE",
}

_PARAM_NAMES = ("lam", "mu_j", "sigma_j")

# MLE bounds. λ may range up to one jump per day on average: the optimum is
# reported, not forced into a "crash-like" box (bound activity is flagged).
_BOUNDS_FULL: List[Tuple[float, float]] = [
    (1e-3, 252.0),  # λ (annualised jumps per year)
    (-0.20, 0.20),  # μ_j: mean log jump size
    (1e-4, 0.25),   # σ_j: jump size std
]

# ── Jump parameter container ───────────────────────────────────────────────────

@dataclass
class MertonJumpParams:
    """
    Calibrated Merton (1976) jump parameters.

    lam    : float — annualised jump intensity λ (expected jumps per year)
    mu_j   : float — mean log jump size (in return space, decimal)
    sigma_j: float — std of log jump size (decimal; > 0)
    n_r2_days   : days used for the calibration
    nll         : negative log-likelihood at the optimum (MLE only)
    bound_active: names of parameters that ended on (within 0.1% of the
                  width of) an optimisation bound — the estimate is then a
                  property of the box, not of the data
    method      : 'mle' | 'threshold' | 'moment_fallback' | 'none'

    Annualised jump variance contribution: lam * (mu_j² + sigma_j²)
    """
    lam:       float
    mu_j:      float
    sigma_j:   float
    n_r2_days: int   = 0
    nll:       float = 0.0
    bound_active: Tuple[str, ...] = ()
    method:    str   = "mle"

    @property
    def jump_variance_annual(self) -> float:
        """Annual variance added by the jump component."""
        return self.lam * (self.mu_j ** 2 + self.sigma_j ** 2)

    @property
    def jump_vol_annual(self) -> float:
        """
        Annualised vol contribution from jumps alone.
        Not the same as total vol increase — use with pdv_vol:
          total_vol = sqrt(pdv_vol² + jump_variance_annual)
        """
        return np.sqrt(max(self.jump_variance_annual, 0.0))

    def __repr__(self) -> str:
        bounds = f", bound_active={list(self.bound_active)}" if self.bound_active else ""
        return (
            f"MertonJumpParams(λ={self.lam:.3f}/yr, "
            f"μ_j={self.mu_j*100:.2f}%, "
            f"σ_j={self.sigma_j*100:.2f}%, "
            f"jump_var_ann={self.jump_variance_annual*100:.2f}% vol pts², "
            f"n_days={self.n_r2_days}, method={self.method}{bounds})"
        )


def _no_jump(n_days: int) -> MertonJumpParams:
    return MertonJumpParams(lam=0.0, mu_j=0.0, sigma_j=1e-4,
                            n_r2_days=int(n_days), nll=0.0, method="none")


# ── Merton density (MLE building block) ───────────────────────────────────────

def merton_log_density(
    r: np.ndarray,
    sigma_d: np.ndarray,
    lam_daily: float,
    mu_j: float,
    sigma_j: float,
    n_max: int = N_MAX_JUMPS,
) -> np.ndarray:
    """
    Log-density of the Merton (1976) mixture for daily log-returns.

    p(r_t | σ_d,t, λ_d, μ_j, σ_j) =
        Σ_{n=0}^{n_max} P(N=n; λ_d) · Normal(r_t; n·μ_j, σ_d,t² + n·σ_j²)

    Parameters
    ----------
    r          : array (T,)  — daily log-returns
    sigma_d    : array (T,)  — daily diffusion std
    lam_daily  : float       — daily jump intensity  (λ/252)
    mu_j       : float       — mean log jump size
    sigma_j    : float       — std log jump size
    n_max      : int         — Poisson truncation

    Returns
    -------
    log_dens   : array (T,)  — log p(r_t | ...) for each observation
    """
    ns = np.arange(0, n_max + 1)
    log_pois = scipy_poisson.logpmf(ns, lam_daily)  # (n_max+1,)

    r_2d     = r[:, None]           # (T, 1)
    sigma_2d = sigma_d[:, None]     # (T, 1)

    mix_mu  = ns[None, :] * mu_j                           # (1, n_max+1)
    mix_var = sigma_2d ** 2 + ns[None, :] * sigma_j ** 2  # (T, n_max+1)
    mix_var = np.maximum(mix_var, 1e-16)

    log_normal = (
        -0.5 * np.log(2.0 * np.pi * mix_var)
        - 0.5 * ((r_2d - mix_mu) ** 2) / mix_var
    )

    # log-sum-exp over the mixture components
    log_terms = log_pois[None, :] + log_normal   # (T, n_max+1)
    max_terms = log_terms.max(axis=1, keepdims=True)
    log_dens  = max_terms[:, 0] + np.log(
        np.exp(log_terms - max_terms).sum(axis=1)
    )
    return log_dens


def _active_bounds(x: np.ndarray, bounds: List[Tuple[float, float]],
                   rel_tol: float = 1e-3) -> Tuple[str, ...]:
    """Names of the parameters lying within rel_tol·(hi−lo) of a bound."""
    out = []
    for name, v, (lo, hi) in zip(_PARAM_NAMES, x, bounds):
        tol = rel_tol * (hi - lo)
        if v <= lo + tol or v >= hi - tol:
            out.append(name)
    return tuple(out)


def calibrate_jump_mle(
    daily_returns: np.ndarray,
    daily_sigma_d: np.ndarray,
    n_max: int = N_MAX_JUMPS,
    random_seed: int = RANDOM_SEED,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> MertonJumpParams:
    """
    Calibrate Merton jump parameters by maximum likelihood.

    Fits λ, μ_j, σ_j to minimise -Σ_t log p(r_t | σ_d,t, λ/252, μ_j, σ_j),
    with L-BFGS-B from several starting points spread over the λ range.

    The lowest finite negative log-likelihood across starts is returned even
    when L-BFGS-B reports success=False (it often stops with
    ABNORMAL_TERMINATION_IN_LNSRCH at a perfectly good point); the message is
    logged. Only if no start produces a finite value is the moment-matching
    fallback used, and its output is clipped into `bounds`.

    Parameters
    ----------
    daily_returns  : array (T,)  — daily log-returns
    daily_sigma_d  : array (T,)  — daily diffusion std
    n_max          : int         — Poisson truncation
    random_seed    : int
    bounds         : list of (lo, hi) tuples for [λ, μ_j, σ_j].
                     Defaults to _BOUNDS_FULL.

    Returns
    -------
    MertonJumpParams — calibrated parameters (see .bound_active)
    """
    if bounds is None:
        bounds = _BOUNDS_FULL
    bounds = [tuple(map(float, b)) for b in bounds]
    daily_returns = np.asarray(daily_returns, dtype=float)
    daily_sigma_d = np.asarray(daily_sigma_d, dtype=float)

    rng = np.random.default_rng(random_seed)

    def neg_ll(params: np.ndarray) -> float:
        lam_ann, mu_j, sigma_j = params
        lam_daily = lam_ann / ANNUALISE
        if lam_daily <= 0 or sigma_j <= 0:
            return 1e12
        log_dens = merton_log_density(
            daily_returns, daily_sigma_d, lam_daily, mu_j, sigma_j, n_max
        )
        return -np.sum(log_dens)

    # Starting points from rare/large to frequent/small jumps, clipped into
    # the bounds so every start is feasible.
    raw_starts = [
        [2.0,   -0.030, 0.040],
        [10.0,  -0.010, 0.020],
        [50.0,  -0.005, 0.010],
        [150.0,  0.000, 0.005],
        [rng.uniform(1, 100), rng.uniform(-0.03, 0.0), rng.uniform(0.005, 0.04)],
    ]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    starts = [np.clip(np.asarray(x0, dtype=float), lo, hi) for x0 in raw_starts]

    best_result = None
    best_nll    = np.inf

    for x0 in starts:
        try:
            res = minimize(
                neg_ll, x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
        except Exception as exc:
            logger.debug("Merton MLE start %s failed: %s", x0, exc)
            continue
        if np.isfinite(res.fun) and res.fun < best_nll:
            best_nll    = float(res.fun)
            best_result = res

    if best_result is None or best_nll >= 1e12:
        logger.warning(
            "Merton MLE produced no finite likelihood; falling back to "
            "moment matching (clipped into the bounds)."
        )
        lam_ann, mu_j, sigma_j = _moment_match_fallback(
            daily_returns, daily_sigma_d, bounds
        )
        nll = neg_ll([lam_ann, mu_j, sigma_j])
        return MertonJumpParams(
            lam=lam_ann, mu_j=mu_j, sigma_j=sigma_j,
            n_r2_days=len(daily_returns), nll=float(nll),
            bound_active=_active_bounds(np.array([lam_ann, mu_j, sigma_j]), bounds),
            method="moment_fallback",
        )

    if not best_result.success:
        logger.info(
            "Merton MLE: best start reports success=False (%s); using its "
            "optimum (NLL %.3f).", best_result.message, best_nll,
        )

    x = np.asarray(best_result.x, dtype=float)
    active = _active_bounds(x, bounds)
    if active:
        logger.warning(
            "Merton MLE optimum on the bound for %s: %s — the estimate "
            "reflects the box, not the data.", list(active), x,
        )
    lam_ann, mu_j, sigma_j = x
    return MertonJumpParams(
        lam=float(lam_ann),
        mu_j=float(mu_j),
        sigma_j=float(sigma_j),
        n_r2_days=len(daily_returns),
        nll=float(best_nll),
        bound_active=active,
        method="mle",
    )


def _moment_match_fallback(
    daily_returns: np.ndarray,
    daily_sigma_d: np.ndarray,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float, float]:
    """
    Crude moment matching used only when the MLE cannot be evaluated.

    Uses the excess variance  E[r_t² − σ_d,t²] = λ_d · (μ_j² + σ_j²)  with
    (μ_j, σ_j) set to the mean / std of the returns. Every parameter is
    clipped into `bounds` (default _BOUNDS_FULL).
    """
    if bounds is None:
        bounds = _BOUNDS_FULL
    excess_var_daily = daily_returns ** 2 - daily_sigma_d ** 2
    mean_excess      = np.maximum(np.mean(excess_var_daily), 1e-8)

    mu_j    = float(np.mean(daily_returns))
    sigma_j = max(float(np.std(daily_returns)), 1e-4)

    denom   = max(mu_j ** 2 + sigma_j ** 2, 1e-10)
    lam_ann = mean_excess / denom * ANNUALISE

    (l_lo, l_hi), (m_lo, m_hi), (s_lo, s_hi) = bounds
    return (float(np.clip(lam_ann, l_lo, l_hi)),
            float(np.clip(mu_j, m_lo, m_hi)),
            float(np.clip(sigma_j, s_lo, s_hi)))


# ── Daily jump test ────────────────────────────────────────────────────────────

@lru_cache(maxsize=32)
def jump_test_critical_value(
    window: int = JUMP_TEST_WINDOW,
    alpha: float = JUMP_TEST_ALPHA,
    n_sim: int = 200_000,
    seed: int = RANDOM_SEED,
) -> float:
    """
    Two-sided critical value c with P(|L_t| > c) = alpha under an iid
    Gaussian null, for the daily_jump_test() statistic with `window` pairs.

    L_t is scale-free under that null, so one simulation serves every
    volatility level. Deterministic (fixed seed) and cached. For
    window = 20, alpha = 1% it is ≈ 2.91 (vs 2.58 for a known variance):
    estimating σ from 20 pairs fattens the tails of L.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((int(n_sim), int(window) + 2))
    bv = (np.pi / 2.0) * np.mean(np.abs(z[:, 1:window + 1] * z[:, 0:window]), axis=1)
    stat = np.abs(z[:, window + 1]) / np.sqrt(bv)
    return float(np.quantile(stat, 1.0 - alpha))


def daily_jump_test(
    log_returns: pd.Series,
    window: int = JUMP_TEST_WINDOW,
    alpha: float = JUMP_TEST_ALPHA,
) -> pd.DataFrame:
    """
    Lee-Mykland (2008)-style jump test on daily returns.

        L_t = r_t / σ̂_{t−1},   σ̂²_{t−1} = (π/2) · mean(|r_j||r_{j−1}|)

    over the `window` most recent return pairs strictly before t (local
    bipower variance: an isolated jump in the window inflates only two of its
    products, and each by a factor of one neighbouring diffusive return).
    Day t is flagged when |L_t| exceeds jump_test_critical_value(window,
    alpha), so the per-day size is alpha under an iid Gaussian null.

    This is NOT the Barndorff-Nielsen-Shephard test, which needs intraday
    returns within each day.

    Returns
    -------
    DataFrame indexed like the (NaN-free) returns with columns
      stat        : L_t (signed)
      sigma_local : σ̂_{t−1} (daily, decimal)
      crit        : critical value
      jump        : bool flag (False where the statistic is undefined)
    """
    r = log_returns.dropna().astype(float)
    a = r.abs()
    bv = (np.pi / 2.0) * (a * a.shift(1)).rolling(int(window)).mean().shift(1)
    sigma_local = np.sqrt(bv)
    stat = r / sigma_local.where(sigma_local > 0)
    crit = jump_test_critical_value(int(window), float(alpha))
    return pd.DataFrame({
        "stat": stat,
        "sigma_local": sigma_local,
        "crit": crit,
        "jump": (stat.abs() > crit).fillna(False).astype(bool),
    })


# ── Main class ─────────────────────────────────────────────────────────────────

class RegimePDV:
    """
    Regime-switching PDV with Merton jump component.

    Wraps:
      - SPX log returns (from the fitted PDVModel pickle, or injected) and the
        PDV features computed from them
      - Regime labels parquet (C8 output)
      - A diffusion model and jump calibration per as_of_date (lazy, cached,
        zero look-ahead)

    Typical usage
    -------------
    model = RegimePDV().load()
    result = model.forecast("2026-03-24")
    comparison = model.compare_covid_2020()

    Or, without files:
    model = RegimePDV.from_data(log_returns, regime_labels)
    """

    def __init__(
        self,
        pdv_model_path:     Optional[Path] = None,
        regime_labels_path: Optional[Path] = None,
        max_stale_bdays:    int = 3,
    ):
        self.pdv_model_path     = Path(pdv_model_path    or PDV_MODEL_PATH)
        self.regime_labels_path = Path(regime_labels_path or REGIME_LBL_PATH)
        self.max_stale_bdays    = int(max_stale_bdays)

        # Loaded artifacts
        self._pdv:           Optional[PDVModel]      = None
        self._regime_labels: Optional[pd.DataFrame]  = None
        self._log_returns:   Optional[pd.Series]     = None
        self._features:      Optional[pd.DataFrame]  = None

        # Caches keyed by as_of_date (and test settings)
        self._diffusion_cache: Dict[str, Optional[dict]] = {}
        self._jump_cache:      Dict[str, MertonJumpParams] = {}
        self._jump_tail_cache: Dict[str, MertonJumpParams] = {}

        self.is_loaded: bool = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "RegimePDV":
        """
        Load the PDV model (for its return history) and the regime labels.

        Must be called before forecast().
        """
        if not self.pdv_model_path.exists():
            raise FileNotFoundError(
                f"PDV model not found: {self.pdv_model_path}\n"
                "Build it with `python -m joint_vol_calibration.models.pdv`."
            )
        if not self.regime_labels_path.exists():
            raise FileNotFoundError(
                f"Regime labels not found: {self.regime_labels_path}\n"
                "Run C8 (regime_classifier.py) first."
            )

        with open(self.pdv_model_path, "rb") as fh:
            self._pdv = pickle.load(fh)

        if not self._pdv.is_fitted:
            raise RuntimeError("Loaded PDVModel is not fitted.")

        labels = pd.read_parquet(self.regime_labels_path)
        self._set_data(self._returns_from_pdv(self._pdv), labels)

        logger.info(
            "RegimePDV loaded: returns %s→%s (%d rows) | "
            "Regime labels %s→%s (%d rows, R2=%d)",
            self._log_returns.index.min().date(),
            self._log_returns.index.max().date(),
            len(self._log_returns),
            self._regime_labels.index.min().date(),
            self._regime_labels.index.max().date(),
            len(self._regime_labels),
            (self._regime_labels["regime"] == 2).sum(),
        )
        return self

    @classmethod
    def from_data(
        cls,
        log_returns: pd.Series,
        regime_labels: pd.DataFrame,
        max_stale_bdays: int = 3,
    ) -> "RegimePDV":
        """Build a loaded instance from in-memory returns and labels."""
        m = cls(max_stale_bdays=max_stale_bdays)
        m._set_data(log_returns, regime_labels)
        return m

    def _set_data(self, log_returns: pd.Series, labels: pd.DataFrame) -> None:
        r = log_returns.dropna().astype(float).copy()
        r.index = pd.to_datetime(r.index)
        r = r[~r.index.duplicated(keep="last")].sort_index()
        lbl = labels.copy()
        lbl.index = pd.to_datetime(lbl.index)
        self._log_returns   = r
        self._features      = extract_pdv_features(r)
        self._regime_labels = lbl.sort_index()
        self._diffusion_cache.clear()
        self._jump_cache.clear()
        self._jump_tail_cache.clear()
        self.is_loaded = True

    @staticmethod
    def _returns_from_pdv(pdv) -> pd.Series:
        """Log returns behind a fitted PDVModel (pickles predating C22 only
        store features: r_t is then rebuilt as r_lag1 of the next row, which
        loses the final day)."""
        r = getattr(pdv, "_log_returns", None)
        if isinstance(r, pd.Series) and len(r):
            return r
        feats = pdv._features.copy()
        feats.index = pd.to_datetime(feats.index)
        logger.warning(
            "PDVModel pickle has no stored returns; rebuilding them from "
            "r_lag1 (the last day %s is lost). Rebuild the pickle.",
            feats.index[-1].date(),
        )
        return feats["r_lag1"].shift(-1).dropna()

    def _require_loaded(self, what: str) -> None:
        if not self.is_loaded:
            raise RuntimeError(f"Call .load() before .{what}().")

    # ── Diffusion model (PDV re-fit strictly before as_of_date) ───────────────

    def _diffusion(self, as_of_date: str) -> Optional[dict]:
        """
        PDVLinear fit on returns strictly before `as_of_date`, its scale k
        (E|r| → diffusion σ, see module docstring) and its forecasts p_t
        (annualised, E|r| scale) for every feature date.

        Returns None if fewer than MIN_DIFFUSION_TRAIN training rows exist.
        """
        if as_of_date in self._diffusion_cache:
            return self._diffusion_cache[as_of_date]
        cutoff = pd.to_datetime(as_of_date)
        r = self._log_returns
        r_before = r[r.index < cutoff]
        out = None
        if len(r_before) > 0:
            feats_before = self._features.loc[r_before.index]
            X, y = build_xy(r_before, feats_before, horizon=1, target="abs_return")
            if len(X) >= MIN_DIFFUSION_TRAIN:
                lin = PDVLinear().fit(X, y)
                fitted = lin.predict(X)
                k = float(np.median(y.values / fitted.values) / norm.ppf(0.75))
                # forecasts from the burn-in rows (slow EWMA still averaging
                # a handful of returns) are not used as diffusion vols
                cols = PDVLinear._XCOLS
                p = lin.predict(self._features[cols].iloc[EWM_BURN_IN:].dropna())
                out = {"model": lin, "scale": k, "p": p,
                       "train_end": X.index[-1], "n_train": len(X)}
        self._diffusion_cache[as_of_date] = out
        return out

    # ── Shared data extraction ────────────────────────────────────────────────

    def _get_r2_data(
        self, as_of_date: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """
        Calibration arrays for all days t < as_of_date whose previous close
        was labelled R2.

        Returns
        -------
        (r_t, sigma_d_daily, pdv_vol_ann, dates) if ≥ MIN_R2_DAYS days, else None.
          r_t           : daily log-return of day t
          sigma_d_daily : diffusion std for r_t = k·p_{t−1}/√252 (ex ante)
          pdv_vol_ann   : k·p_{t−1} (annualised)
          dates         : the days t
        """
        cutoff = pd.to_datetime(as_of_date)
        diff = self._diffusion(as_of_date)
        if diff is None:
            return None

        r = self._log_returns
        r = r[r.index < cutoff]
        lbl = self._regime_labels["regime"].reindex(r.index)
        prev_r2 = (lbl.shift(1) == 2).to_numpy()

        pdv_prev = (diff["scale"] * diff["p"]).reindex(r.index).shift(1)
        sel = prev_r2 & pdv_prev.notna().to_numpy() & np.isfinite(r.to_numpy())
        if sel.sum() < MIN_R2_DAYS:
            return None
        pdv_vol_ann   = pdv_prev.to_numpy()[sel]
        sigma_d_daily = pdv_vol_ann / np.sqrt(ANNUALISE)
        return r.to_numpy()[sel], sigma_d_daily, pdv_vol_ann, r.index[sel]

    def _r2_count(self, as_of_date: str) -> int:
        lbl = self._regime_labels
        return int(((lbl["regime"] == 2) & (lbl.index < pd.to_datetime(as_of_date))).sum())

    # ── Daily jump flags ──────────────────────────────────────────────────────

    def compute_jump_flags(
        self,
        as_of_date: str,
        alpha: float = JUMP_TEST_ALPHA,
        window: int = JUMP_TEST_WINDOW,
    ) -> pd.DataFrame:
        """
        daily_jump_test() on the returns strictly before `as_of_date`.

        Returns the test DataFrame (stat, sigma_local, crit, jump) for those
        days; rows whose statistic is undefined (first `window`+1 days) are
        dropped.
        """
        self._require_loaded("compute_jump_flags")
        r = self._log_returns
        r = r[r.index < pd.to_datetime(as_of_date)]
        return daily_jump_test(r, window=window, alpha=alpha).dropna(subset=["stat"])

    # ── Full-R2 calibration (all R2 days) ─────────────────────────────────────

    def calibrate_jump(self, as_of_date: str) -> MertonJumpParams:
        """
        Merton MLE on ALL days before `as_of_date` that follow an R2 close.

        No outcome-based selection; ex-ante diffusion vol; λ free in
        [0.001, 252]/yr with bound activity reported. Used by forecast().

        Result is cached — repeated calls with the same date are free.
        """
        self._require_loaded("calibrate_jump")

        if as_of_date in self._jump_cache:
            return self._jump_cache[as_of_date]

        data = self._get_r2_data(as_of_date)
        if data is None:
            r2_count = self._r2_count(as_of_date)
            logger.warning(
                "Fewer than %d usable R2 days before %s (%d labelled). "
                "Zero-jump fallback.", MIN_R2_DAYS, as_of_date, r2_count,
            )
            params = _no_jump(r2_count)
            self._jump_cache[as_of_date] = params
            return params

        r_r2, sigma_d_daily, _, _ = data
        logger.info(
            "Calibrating jump params on %d post-R2 days before %s ...",
            len(r_r2), as_of_date,
        )
        params = calibrate_jump_mle(r_r2, sigma_d_daily, bounds=_BOUNDS_FULL)
        logger.info("Jump MLE: %s", params)
        self._jump_cache[as_of_date] = params
        return params

    # ── Detected-jump (threshold) calibration ─────────────────────────────────

    def calibrate_jump_tail(
        self,
        as_of_date: str,
        alpha: float = JUMP_TEST_ALPHA,
        window: int = JUMP_TEST_WINDOW,
    ) -> MertonJumpParams:
        """
        Detected-jump estimator on the days after an R2 close.

        Steps
        -----
        1. Calibration days: t < as_of_date with an R2 label at t−1
           (_get_r2_data), ex-ante diffusion vol σ_d,t.
        2. Flag the days whose return is a jump by daily_jump_test(alpha,
           window) — a test with a stated size, using only returns before t
           for its scale.
        3. λ = 252 · n_flagged / n_R2  (detected jumps per R2 day, annualised:
           the estimate refers to every R2 day, not only to flagged ones).
           μ_j = mean(r_flagged);  σ_j² = var(r_flagged) − mean(σ_d,flagged²),
           floored at (1e-4)² — the moments of the flagged returns with the
           diffusion variance removed.

        Because only jumps large enough to be detected are counted, λ is a
        lower bound on the jump frequency and (μ_j, σ_j) describe large
        jumps. Falls back to calibrate_jump() (MLE) when fewer than
        MIN_TAIL_DAYS jumps are detected.

        Zero look-ahead: every input is from dates strictly before
        as_of_date. Cached by (as_of_date, alpha, window).
        """
        self._require_loaded("calibrate_jump_tail")

        cache_key = f"{as_of_date}|a{alpha:.4f}|w{int(window)}"
        if cache_key in self._jump_tail_cache:
            return self._jump_tail_cache[cache_key]

        data = self._get_r2_data(as_of_date)
        if data is None:
            r2_count = self._r2_count(as_of_date)
            logger.warning(
                "Fewer than %d usable R2 days before %s. Zero-jump fallback.",
                MIN_R2_DAYS, as_of_date,
            )
            params = _no_jump(r2_count)
            self._jump_tail_cache[cache_key] = params
            return params

        r_r2, sigma_d_daily, _, dates = data
        flags = self.compute_jump_flags(as_of_date, alpha=alpha, window=window)
        flagged = flags["jump"].reindex(dates).fillna(False).to_numpy(dtype=bool)
        n_flag, n_r2 = int(flagged.sum()), len(r_r2)
        logger.info(
            "Jump test (alpha=%.3f, window=%d) on %d post-R2 days before %s: "
            "%d flagged (%.1f%%)", alpha, window, n_r2, as_of_date,
            n_flag, 100.0 * n_flag / max(n_r2, 1),
        )

        if n_flag < MIN_TAIL_DAYS:
            logger.warning(
                "Only %d detected jumps (< %d) — falling back to the full MLE.",
                n_flag, MIN_TAIL_DAYS,
            )
            params = self.calibrate_jump(as_of_date)
            self._jump_tail_cache[cache_key] = params
            return params

        jr  = r_r2[flagged]
        sdj = sigma_d_daily[flagged]
        mu_j = float(np.mean(jr))
        var_j = float(np.var(jr) - np.mean(sdj ** 2))
        sigma_j = float(np.sqrt(max(var_j, 1e-8)))
        params = MertonJumpParams(
            lam=ANNUALISE * n_flag / n_r2, mu_j=mu_j, sigma_j=sigma_j,
            n_r2_days=n_r2, nll=0.0, method="threshold",
        )
        logger.info("Detected-jump estimate: %s", params)
        self._jump_tail_cache[cache_key] = params
        return params

    # ── Forecast ──────────────────────────────────────────────────────────────

    def _stale_bdays(self, used: pd.Timestamp, requested: pd.Timestamp) -> int:
        return max(len(pd.bdate_range(used, requested)) - 1, 0)

    def forecast(
        self,
        as_of_date: str,
        jump_method: str = "mle",
        allow_stale: bool = False,
    ) -> dict:
        """
        Regime-switched vol forecast made at the close of `as_of_date`.

        Logic
        -----
        1. Diffusion model fit on data strictly before as_of_date; σ_PDV =
           k·p(features at as_of_date) — the next-day diffusion vol.
        2. Regime label at as_of_date.
        3. If R2: add the jump variance from calibrate_jump() (jump_method=
           'mle', default) or calibrate_jump_tail() ('threshold').

        If as_of_date has no features or no label, the nearest prior date is
        used, and the gap is reported. A gap of more than max_stale_bdays
        business days raises ValueError unless allow_stale=True (then
        'stale' is True in the result).

        Returns
        -------
        dict — see module docstring.
        """
        self._require_loaded("forecast")
        if jump_method not in ("mle", "threshold"):
            raise ValueError("jump_method must be 'mle' or 'threshold'")

        dt = pd.to_datetime(as_of_date)

        # ── Features at the forecast origin ──────────────────────────────────
        feats = self._features
        prior = feats.index[feats.index <= dt]
        if len(prior) == 0:
            raise ValueError(f"No PDV features available on or before {as_of_date}.")
        dt_feat = prior[-1]

        # ── Regime ───────────────────────────────────────────────────────────
        lbl = self._regime_labels
        prior_lbl = lbl.index[lbl.index <= dt]
        if len(prior_lbl) == 0:
            raise ValueError(f"No regime label on or before {as_of_date}.")
        dt_lbl = prior_lbl[-1]

        gap_feat = self._stale_bdays(dt_feat, dt)
        gap_lbl  = self._stale_bdays(dt_lbl, dt)
        stale = max(gap_feat, gap_lbl) > self.max_stale_bdays
        if gap_feat or gap_lbl:
            logger.warning(
                "forecast(%s): using features from %s and regime label from %s.",
                as_of_date, dt_feat.date(), dt_lbl.date(),
            )
        if stale and not allow_stale:
            raise ValueError(
                f"forecast({as_of_date}): latest features {dt_feat.date()} / "
                f"label {dt_lbl.date()} are more than {self.max_stale_bdays} "
                "business days old. Refresh the data, or pass allow_stale=True."
            )

        # ── PDV diffusion vol ────────────────────────────────────────────────
        diff = self._diffusion(as_of_date)
        if diff is None:
            raise ValueError(
                f"Fewer than {MIN_DIFFUSION_TRAIN} training rows before {as_of_date}."
            )
        pdv_vol = float(diff["scale"] * diff["p"].loc[dt_feat])

        regime_int  = int(lbl.loc[dt_lbl, "regime"])
        regime_name = REGIME_NAMES[regime_int]

        # ── Jump adjustment ──────────────────────────────────────────────────
        if regime_int == 2:
            params = (self.calibrate_jump(as_of_date) if jump_method == "mle"
                      else self.calibrate_jump_tail(as_of_date))
            jump_var  = params.jump_variance_annual
            total_vol = float(np.sqrt(pdv_vol ** 2 + jump_var))
            jump_adj  = float(total_vol - pdv_vol)
            lam, mu_j, sigma_j = params.lam, params.mu_j, params.sigma_j
            n_cal     = params.n_r2_days
            bound_active = params.bound_active
        else:
            jump_adj  = 0.0
            total_vol = pdv_vol
            lam = mu_j = sigma_j = float("nan")
            n_cal     = 0
            bound_active = ()

        return {
            "pdv_vol":         pdv_vol,
            "jump_adj":        jump_adj,
            "total_vol":       total_vol,
            "regime":          regime_name,
            "lambda":          lam,
            "mu_j":            mu_j,
            "sigma_j":         sigma_j,
            "n_r2_cal_days":   n_cal,
            "bound_active":    bound_active,
            "jump_method":     jump_method,
            "diffusion_scale": diff["scale"],
            "as_of_date":      as_of_date,
            "feature_date":    str(dt_feat.date()),
            "regime_date":     str(dt_lbl.date()),
            "stale":           bool(stale),
        }

    # ── COVID comparison ──────────────────────────────────────────────────────

    def compare_covid_2020(
        self,
        start: str = "2020-01-02",
        end:   str = "2020-12-31",
        jump_method: str = "mle",
    ) -> pd.DataFrame:
        """
        Out-of-sample comparison of plain PDV vs regime-switched (Merton) PDV.

        For each trading day t in [start, end] the forecasts are made at the
        close of t for day t+1. The diffusion model and the jump parameters
        are calibrated at the first trading day of t's quarter (inside the
        window) on data strictly before it, so every forecast uses only
        information available at its close.

        Each forecast variance h is scored against the next day's realised
        annualised squared return x = 252·r_{t+1}² with
          QLIKE  L = x/h + log h   (Patton 2011; = x/h − log(x/h) − 1 up to a
                                    term in x only, so loss differences are
                                    the same and x = 0 is allowed)
          SE     (x − h)²
        Both are consistent for noisy but conditionally unbiased proxies.

        Returns
        -------
        pd.DataFrame indexed by date with columns:
          regime, pdv_vol, jump_adj, total_vol, rv_next_1d, rv_fwd_21d,
          qlike_pdv, qlike_jump, se_pdv, se_jump, improvement
        where
          rv_next_1d  : |r_{t+1}|·√252 (the realised outcome being scored)
          rv_fwd_21d  : forward 21-day RMS vol (context only, not scored)
          improvement : qlike_pdv − qlike_jump (> 0 means the jump term helps)
        """
        self._require_loaded("compare_covid_2020")

        r = self._log_returns
        feats = self._features
        date_range = feats.loc[start:end].index

        r_next = r.shift(-1)
        fwd21 = forward_target(r, horizon=21, target="rv")
        lbl = self._regime_labels["regime"]

        rows = []
        quarter_origin: Dict[str, str] = {}
        for dt in date_range:
            qkey = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            if qkey not in quarter_origin:
                quarter_origin[qkey] = dt.strftime("%Y-%m-%d")
            origin = quarter_origin[qkey]

            diff = self._diffusion(origin)
            if diff is None or dt not in diff["p"].index:
                continue
            pdv_vol = float(diff["scale"] * diff["p"].loc[dt])

            regime_int = int(lbl.loc[dt]) if dt in lbl.index else -1
            regime_name = REGIME_NAMES.get(regime_int, "UNKNOWN")

            if regime_int == 2:
                jp = (self.calibrate_jump(origin) if jump_method == "mle"
                      else self.calibrate_jump_tail(origin))
                total_vol = float(np.sqrt(pdv_vol ** 2 + jp.jump_variance_annual))
            else:
                total_vol = pdv_vol
            jump_adj = total_vol - pdv_vol

            rn = r_next.get(dt, np.nan)
            x = ANNUALISE * rn ** 2 if np.isfinite(rn) else np.nan
            h0, h1 = pdv_vol ** 2, total_vol ** 2
            q0 = x / h0 + np.log(h0) if np.isfinite(x) else np.nan
            q1 = x / h1 + np.log(h1) if np.isfinite(x) else np.nan
            rows.append({
                "date":        dt,
                "regime":      regime_name,
                "pdv_vol":     pdv_vol,
                "jump_adj":    jump_adj,
                "total_vol":   total_vol,
                "rv_next_1d":  abs(rn) * np.sqrt(ANNUALISE) if np.isfinite(rn) else np.nan,
                "rv_fwd_21d":  float(fwd21.get(dt, np.nan)),
                "qlike_pdv":   q0,
                "qlike_jump":  q1,
                "se_pdv":      (x - h0) ** 2 if np.isfinite(x) else np.nan,
                "se_jump":     (x - h1) ** 2 if np.isfinite(x) else np.nan,
                "improvement": q0 - q1,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.set_index("date")

        r2_rows = df[(df["regime"] == "VOMMA_ACTIVE")].dropna(subset=["qlike_pdv"])
        if len(r2_rows) > 0:
            logger.info(
                "Comparison %s→%s on %d R2 days: mean QLIKE PDV=%.4f → "
                "jump=%.4f (Δ=%.4f, >0 favours jumps)",
                start, end, len(r2_rows), r2_rows["qlike_pdv"].mean(),
                r2_rows["qlike_jump"].mean(), r2_rows["improvement"].mean(),
            )
        return df

    # ── Convenience ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "RegimePDV(not loaded)"
        n_r2  = (self._regime_labels["regime"] == 2).sum()
        n_lbl = len(self._regime_labels)
        return (
            f"RegimePDV(loaded | returns {self._log_returns.index[0].date()}→"
            f"{self._log_returns.index[-1].date()} | "
            f"regime_labels={n_lbl} rows | R2={n_r2} | "
            f"jump_cache={len(self._jump_cache)} | "
            f"tail_cache={len(self._jump_tail_cache)})"
        )
