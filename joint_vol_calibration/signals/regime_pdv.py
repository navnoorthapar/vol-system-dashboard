"""
regime_pdv.py — C11: Regime-Switching PDV with Merton Jump Component

Overview
--------
Combines two already-fitted components:
  • C3 PDV Linear  — σ_diffusion(t) = 0.354·σ₁ + 0.241·σ₂ − 1.496·lev + 0.0346
  • C8 Regime      — XGBoost 3-way classifier (R0=LONG_GAMMA, R1=SHORT_GAMMA,
                     R2=VOMMA_ACTIVE)

Forecast logic
--------------
  R0 / R1 (VVIX ≤ 100):
      total_vol = σ_PDV                 (plain diffusion, no jump)
      jump_adj  = 0

  R2 (VOMMA_ACTIVE, VVIX > 100):
      total_vol = sqrt(σ²_PDV + λ·(μ_j² + σ_j²))
      jump_adj  = total_vol − σ_PDV

      where λ, μ_j, σ_j are the Merton (1976) jump parameters calibrated on
      the TAIL of R2 days (see Tail Calibration below).

Merton (1976) jump component
-----------------------------
The daily log-return on R2 days is modelled as:

    r_t = σ_d,t · ε_t + Σ_{k=1}^{N_t} J_k
    ε_t ~ N(0, 1/252)
    N_t ~ Poisson(λ/252)
    J_k ~ N(μ_j, σ_j²)

Annualised variance:
    Var_annual = σ²_PDV + λ·(μ_j² + σ_j²)

Parameters λ, μ_j, σ_j are fitted by maximum likelihood.
The log-likelihood uses the Merton mixture density (truncated at n_max jumps).

Tail Calibration (calibrate_jump_tail)
---------------------------------------
Naive MLE on all 1,465 R2 days collapses to λ≈50/yr with tiny σ_j because
elevated-VVIX days include many "calm R2" periods where PDV explains vol well.
The solution: calibrate only on R2 days where PDV severely underpredicted.

Steps:
  1. For each R2 day t, compute error = rv_actual(t) − σ_PDV(t)
  2. Standardise: error_z = (error − mean(error)) / std(error)
  3. Keep only days where error_z > zscore_threshold (default 2.0)
  4. Run MLE on the tail only, using crash-appropriate bounds:
       λ ∈ [0.1, 20]   — crash jumps happen 2–8 times/year, not 50
       μ_j ∈ [−0.15, 0.05]  — allow large negative jumps
       σ_j ∈ [0.005, 0.15]  — jump size std 0.5–15%

Expected result: λ=2–8/yr, μ_j negative (−3% to −8%), σ_j=3–6%.
Falls back to full-R2 calibration if tail has fewer than MIN_TAIL_DAYS=10.

Zero look-ahead
---------------
  • `as_of_date` is the last date for which a forecast is requested.
  • Jump calibration uses ONLY R2 days with date < as_of_date.
  • PDV features at date t use only returns up through and including day t.
  • Regime labels at date t are determined by day-t observables (same as C8).

Outputs (from .forecast())
--------------------------
  {
    'pdv_vol':       float — PDV diffusion forecast (annualised, decimal)
    'jump_adj':      float — Merton jump addition in vol units (≥ 0)
    'total_vol':     float — sqrt(pdv_vol² + λ·(μ_j²+σ_j²))
    'regime':        str   — 'LONG_GAMMA' | 'SHORT_GAMMA' | 'VOMMA_ACTIVE'
    'lambda':        float — jump intensity (jumps/year, NaN for R0/R1)
    'mu_j':          float — mean log jump size (decimal, NaN for R0/R1)
    'sigma_j':       float — std log jump size (decimal, NaN for R0/R1)
    'n_r2_cal_days': int   — tail days used for MLE (0 for R0/R1)
  }

COVID comparison (compare_covid_2020)
--------------------------------------
Returns a DataFrame of daily forecasts over 2020, comparing:
  - σ_PDV  (plain diffusion)
  - σ_jump (regime-switched with tail-Merton component)
  - σ_actual (ex-post 20-day RV from PDV features)
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson as scipy_poisson

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.models.pdv import PDVModel, extract_pdv_features

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

SIGNALS_DIR     = DATA_DIR / "signals"
PDV_MODEL_PATH  = DATA_DIR / "pdv_model.pkl"
REGIME_CLF_PATH = SIGNALS_DIR / "regime_classifier.pkl"
REGIME_LBL_PATH = SIGNALS_DIR / "regime_labels.parquet"

# ── Constants ─────────────────────────────────────────────────────────────────

ANNUALISE    = 252          # trading days per year
N_MAX_JUMPS  = 5            # Poisson truncation for MLE density
MIN_R2_DAYS  = 30           # minimum R2 days for full calibration
MIN_TAIL_DAYS = 10          # minimum tail days for tail calibration

REGIME_NAMES: Dict[int, str] = {
    0: "LONG_GAMMA",
    1: "SHORT_GAMMA",
    2: "VOMMA_ACTIVE",
}

# MLE bounds for full-R2 calibration (all elevated-VVIX days)
_BOUNDS_FULL: List[Tuple[float, float]] = [
    (0.1,  50.0),   # λ (annualised): 0.1 to 50 jumps/year
    (-0.10, 0.02),  # μ_j: typical small-jump mean
    (0.001, 0.20),  # σ_j: jump size std
]

# MLE bounds for tail calibration (genuine crash days only)
_BOUNDS_TAIL: List[Tuple[float, float]] = [
    (0.1,  20.0),   # λ: crash jumps are rarer (< 20/yr)
    (-0.15, 0.05),  # μ_j: allow large negative crash jumps
    (0.005, 0.15),  # σ_j: crash jump size std (0.5–15%)
]

# ── Jump parameter container ───────────────────────────────────────────────────

@dataclass
class MertonJumpParams:
    """
    Calibrated Merton (1976) jump parameters.

    lam    : float — annualised jump intensity λ (expected jumps per year)
    mu_j   : float — mean log jump size (in return space, decimal)
    sigma_j: float — std of log jump size (decimal; > 0)

    Annualised jump variance contribution: lam * (mu_j² + sigma_j²)
    """
    lam:       float
    mu_j:      float
    sigma_j:   float
    n_r2_days: int   = 0    # days used for MLE (tail days for tail calibration)
    nll:       float = 0.0  # negative log-likelihood at optimum

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
        return (
            f"MertonJumpParams(λ={self.lam:.3f}/yr, "
            f"μ_j={self.mu_j*100:.2f}%, "
            f"σ_j={self.sigma_j*100:.2f}%, "
            f"jump_var_ann={self.jump_variance_annual*100:.2f}% vol pts², "
            f"n_days={self.n_r2_days})"
        )


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
    sigma_d    : array (T,)  — daily diffusion std (pdv_vol / sqrt(252))
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


def calibrate_jump_mle(
    daily_returns: np.ndarray,
    daily_sigma_d: np.ndarray,
    n_max: int = N_MAX_JUMPS,
    random_seed: int = RANDOM_SEED,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> MertonJumpParams:
    """
    Calibrate Merton jump parameters by maximum likelihood.

    Fits λ, μ_j, σ_j to minimise -Σ_t log p(r_t | σ_d,t, λ/252, μ_j, σ_j).

    Parameters
    ----------
    daily_returns  : array (T,)  — daily log-returns
    daily_sigma_d  : array (T,)  — daily diffusion std (pdv_vol / sqrt(252))
    n_max          : int         — Poisson truncation
    random_seed    : int
    bounds         : list of (lo, hi) tuples for [λ, μ_j, σ_j].
                     Defaults to _BOUNDS_FULL. Pass _BOUNDS_TAIL for tail MLE.

    Returns
    -------
    MertonJumpParams — calibrated parameters
    """
    if bounds is None:
        bounds = _BOUNDS_FULL

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

    # Starting points — crash-biased when tail bounds are used
    is_tail_bounds = bounds is _BOUNDS_TAIL
    if is_tail_bounds:
        starts = [
            [2.0,  -0.030, 0.040],   # 2 jumps/yr, −3%, 4% std
            [4.0,  -0.050, 0.060],   # 4 jumps/yr, −5%, 6% std
            [1.0,  -0.080, 0.080],   # 1 jump/yr,  −8%, 8% std
            [rng.uniform(1, 6), rng.uniform(-0.10, -0.01),
             rng.uniform(0.02, 0.08)],
        ]
    else:
        starts = [
            [2.0,  -0.005, 0.015],
            [5.0,  -0.010, 0.025],
            [10.0, -0.003, 0.010],
            [rng.uniform(1, 8), rng.uniform(-0.02, 0),
             rng.uniform(0.005, 0.04)],
        ]

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
            if res.fun < best_nll:
                best_nll    = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or not best_result.success:
        logger.warning(
            "Merton MLE optimisation failed; falling back to moment matching."
        )
        lam_ann, mu_j, sigma_j = _moment_match_fallback(
            daily_returns, daily_sigma_d
        )
        nll = neg_ll([lam_ann, mu_j, sigma_j])
        return MertonJumpParams(
            lam=lam_ann, mu_j=mu_j, sigma_j=sigma_j,
            n_r2_days=len(daily_returns), nll=float(nll)
        )

    lam_ann, mu_j, sigma_j = best_result.x
    return MertonJumpParams(
        lam=float(lam_ann),
        mu_j=float(mu_j),
        sigma_j=float(sigma_j),
        n_r2_days=len(daily_returns),
        nll=float(best_nll),
    )


def _moment_match_fallback(
    daily_returns: np.ndarray,
    daily_sigma_d: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Moment matching fallback when MLE fails.

    Match first central moment of the excess variance:
        excess_var_t = r_t² − σ_d,t²
    E[excess_var] = λ_d · (μ_j² + σ_j²)
    """
    excess_var_daily = daily_returns ** 2 - daily_sigma_d ** 2
    mean_excess      = np.maximum(np.mean(excess_var_daily), 1e-8)

    mu_j    = float(np.mean(daily_returns))
    sigma_j = float(np.std(daily_returns))
    sigma_j = max(sigma_j, 0.001)

    denom  = mu_j ** 2 + sigma_j ** 2
    if denom <= 0:
        denom = sigma_j ** 2
    lam_d   = mean_excess / max(denom, 1e-10)
    lam_ann = float(np.clip(lam_d * ANNUALISE, 0.1, 50.0))

    return lam_ann, mu_j, sigma_j


# ── Main class ─────────────────────────────────────────────────────────────────

class RegimePDV:
    """
    Regime-switching PDV with Merton jump component.

    Wraps:
      - Fitted PDVModel (linear component only, for speed)
      - Regime labels parquet (C8 output)
      - Merton jump calibration — tail-only (lazy, cached, zero look-ahead)

    Typical usage
    -------------
    model = RegimePDV()
    model.load()
    result = model.forecast("2026-03-24")
    # {'pdv_vol': 0.247, 'jump_adj': 0.083, 'total_vol': 0.330, 'regime': 'VOMMA_ACTIVE', ...}

    comparison = model.compare_covid_2020()
    # DataFrame with daily σ_PDV, σ_jump, σ_actual for all of 2020
    """

    def __init__(
        self,
        pdv_model_path:     Optional[Path] = None,
        regime_labels_path: Optional[Path] = None,
    ):
        self.pdv_model_path     = Path(pdv_model_path    or PDV_MODEL_PATH)
        self.regime_labels_path = Path(regime_labels_path or REGIME_LBL_PATH)

        # Loaded artifacts
        self._pdv:           Optional[PDVModel]      = None
        self._regime_labels: Optional[pd.DataFrame]  = None

        # Separate caches for full vs tail calibration
        self._jump_cache:      Dict[str, MertonJumpParams] = {}
        self._jump_tail_cache: Dict[str, MertonJumpParams] = {}

        self.is_loaded: bool = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self) -> "RegimePDV":
        """
        Load PDV model and regime labels from disk.

        Must be called before forecast().
        """
        if not self.pdv_model_path.exists():
            raise FileNotFoundError(
                f"PDV model not found: {self.pdv_model_path}\n"
                "Run C3 (pdv.py PDVModel.save()) first."
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

        self._regime_labels = pd.read_parquet(self.regime_labels_path)
        self._regime_labels.index = pd.to_datetime(self._regime_labels.index)
        self._regime_labels = self._regime_labels.sort_index()

        logger.info(
            "RegimePDV loaded: PDV features %s→%s (%d rows) | "
            "Regime labels %s→%s (%d rows, R2=%d)",
            self._pdv._features.index.min().date(),
            self._pdv._features.index.max().date(),
            len(self._pdv._features),
            self._regime_labels.index.min().date(),
            self._regime_labels.index.max().date(),
            len(self._regime_labels),
            (self._regime_labels["regime"] == 2).sum(),
        )

        self.is_loaded = True
        return self

    # ── Shared data extraction ────────────────────────────────────────────────

    def _get_r2_data(
        self, as_of_date: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]]:
        """
        Extract arrays for all R2 days strictly before `as_of_date`.

        Returns
        -------
        (r_r2, sigma_d_daily, pdv_vol_ann, dates) if ≥ MIN_R2_DAYS days available.
        None otherwise.

        Where:
          r_r2          : daily log-returns on R2 days
          sigma_d_daily : daily diffusion std = pdv_vol_ann / sqrt(252)
          pdv_vol_ann   : annualised PDV diffusion vol on R2 days
          dates         : DatetimeIndex of valid R2 days (after NaN drop)
        """
        cutoff = pd.to_datetime(as_of_date)

        lbl = self._regime_labels
        r2_mask  = (lbl["regime"] == 2) & (lbl.index < cutoff)
        r2_dates = lbl.index[r2_mask]

        if len(r2_dates) < MIN_R2_DAYS:
            return None

        feats = self._pdv._features.copy()
        feats.index = pd.to_datetime(feats.index)

        common = r2_dates.intersection(feats.index)
        if len(common) < MIN_R2_DAYS:
            return None

        feats_r2 = feats.loc[common]

        # PDV diffusion vol (annualised)
        pdv_vol_ann = self._pdv.linear_.predict(
            feats_r2[["sigma1", "sigma2", "lev"]]
        ).values

        # Daily diffusion std
        sigma_d_daily = pdv_vol_ann / np.sqrt(ANNUALISE)

        # Daily log-return at date t: r_lag1(t) = r_{t-1}, so shift(-1) gives r_t
        r_series = feats["r_lag1"].shift(-1)
        r_r2     = r_series.reindex(common).values

        # Drop NaN (last trading day edge)
        valid = ~np.isnan(r_r2) & ~np.isnan(sigma_d_daily)
        r_r2          = r_r2[valid]
        sigma_d_daily = sigma_d_daily[valid]
        pdv_vol_ann   = pdv_vol_ann[valid]
        valid_dates   = common[valid]

        if len(r_r2) < MIN_R2_DAYS:
            return None

        return r_r2, sigma_d_daily, pdv_vol_ann, valid_dates

    # ── Full-R2 calibration (all R2 days) ─────────────────────────────────────

    def calibrate_jump(self, as_of_date: str) -> MertonJumpParams:
        """
        Calibrate Merton jump parameters on ALL R2 days strictly before `as_of_date`.

        This tends to find λ≈50/yr with tiny σ_j because it includes calm R2 days
        (VVIX slightly above 100 with no real crash). Use calibrate_jump_tail() for
        the preferred crash-jump calibration used in forecast().

        Result is cached — repeated calls with the same date are free.
        """
        if not self.is_loaded:
            raise RuntimeError("Call .load() before .calibrate_jump().")

        if as_of_date in self._jump_cache:
            return self._jump_cache[as_of_date]

        data = self._get_r2_data(as_of_date)
        if data is None:
            r2_count = int((
                (self._regime_labels["regime"] == 2) &
                (self._regime_labels.index < pd.to_datetime(as_of_date))
            ).sum())
            logger.warning(
                "Only %d R2 days before %s (minimum %d). Zero-jump fallback.",
                r2_count, as_of_date, MIN_R2_DAYS,
            )
            params = MertonJumpParams(lam=0.0, mu_j=0.0, sigma_j=1e-4,
                                      n_r2_days=r2_count, nll=0.0)
            self._jump_cache[as_of_date] = params
            return params

        r_r2, sigma_d_daily, _, _ = data
        logger.info(
            "Calibrating full-R2 jump params on %d days before %s ...",
            len(r_r2), as_of_date,
        )
        params = calibrate_jump_mle(r_r2, sigma_d_daily, bounds=_BOUNDS_FULL)
        logger.info("Full-R2 calibrated: %s", params)
        self._jump_cache[as_of_date] = params
        return params

    # ── Tail calibration (crash days only) ────────────────────────────────────

    def calibrate_jump_tail(
        self,
        as_of_date: str,
        zscore_threshold: float = 2.0,
    ) -> MertonJumpParams:
        """
        Calibrate Merton jump parameters on the TAIL of R2 days.

        Motivation
        ----------
        MLE on all R2 days collapses to many tiny jumps (λ≈50) because calm R2
        days (VVIX just above 100) dominate the sample. Filtering to the tail —
        days where PDV most severely under-predicted — isolates genuine crash days
        and produces physically meaningful crash-jump parameters.

        Steps
        -----
        1. Get all R2 days before as_of_date via _get_r2_data().
        2. For each R2 day, compute:
               actual_rv  = rv_hist_20d  (backward-looking 20-day RV, known at close)
               error      = actual_rv − σ_PDV  (positive = PDV underpredicted)
        3. Standardise: error_z = (error − mean(error)) / std(error)
        4. Keep only days where error_z > zscore_threshold.
        5. Run MLE on tail using _BOUNDS_TAIL (crash-appropriate bounds).

        Falls back to full-R2 calibration if tail has < MIN_TAIL_DAYS observations.

        Zero look-ahead guarantee
        -------------------------
        Only R2 days with date < as_of_date enter the filter and the MLE.
        rv_hist_20d(t) is backward-looking (uses returns through day t only).

        Result is cached by (as_of_date, zscore_threshold).

        Parameters
        ----------
        as_of_date        : str 'YYYY-MM-DD'
        zscore_threshold  : float — keep R2 days with error_z > this. Default 2.0.

        Returns
        -------
        MertonJumpParams — tail-calibrated parameters
        """
        if not self.is_loaded:
            raise RuntimeError("Call .load() before .calibrate_jump_tail().")

        cache_key = f"{as_of_date}|z{zscore_threshold:.2f}"
        if cache_key in self._jump_tail_cache:
            return self._jump_tail_cache[cache_key]

        # ── Get all R2 data ───────────────────────────────────────────────────
        data = self._get_r2_data(as_of_date)
        if data is None:
            r2_count = int((
                (self._regime_labels["regime"] == 2) &
                (self._regime_labels.index < pd.to_datetime(as_of_date))
            ).sum())
            logger.warning(
                "Only %d R2 days before %s. Zero-jump fallback.", r2_count, as_of_date
            )
            params = MertonJumpParams(lam=0.0, mu_j=0.0, sigma_j=1e-4,
                                      n_r2_days=r2_count, nll=0.0)
            self._jump_tail_cache[cache_key] = params
            return params

        r_r2, sigma_d_daily, pdv_vol_ann, valid_dates = data

        # ── Compute actual RV on R2 days ──────────────────────────────────────
        feats = self._pdv._features.copy()
        feats.index = pd.to_datetime(feats.index)
        actual_rv = feats.reindex(valid_dates)["rv_hist_20d"].values  # annualised

        # Drop NaN actual_rv (rolling 20-day window produces NaN for first ~20 rows)
        valid_rv = ~np.isnan(actual_rv)
        if valid_rv.sum() < MIN_R2_DAYS:
            logger.warning(
                "After NaN drop in rv_hist_20d only %d R2 rows — zero-jump fallback.",
                valid_rv.sum(),
            )
            params = MertonJumpParams(lam=0.0, mu_j=0.0, sigma_j=1e-4,
                                      n_r2_days=int(valid_rv.sum()), nll=0.0)
            self._jump_tail_cache[cache_key] = params
            return params

        actual_rv     = actual_rv[valid_rv]
        pdv_vol_ann   = pdv_vol_ann[valid_rv]
        r_r2          = r_r2[valid_rv]
        sigma_d_daily = sigma_d_daily[valid_rv]

        # Error in vol space: positive = PDV underpredicted realized vol
        error = actual_rv - pdv_vol_ann

        err_std = error.std()
        if err_std < 1e-10:
            # Degenerate (all errors identical) — fall back to full calibration
            logger.warning(
                "Zero error variance on R2 days before %s; using full-R2 params.",
                as_of_date,
            )
            params = calibrate_jump_mle(r_r2, sigma_d_daily, bounds=_BOUNDS_FULL)
            params = MertonJumpParams(
                lam=params.lam, mu_j=params.mu_j, sigma_j=params.sigma_j,
                n_r2_days=params.n_r2_days, nll=params.nll,
            )
            self._jump_tail_cache[cache_key] = params
            return params

        # ── Filter to tail: days where PDV severely under-predicted ──────────
        error_z    = (error - error.mean()) / err_std
        tail_mask  = error_z > zscore_threshold

        n_tail = int(tail_mask.sum())
        logger.info(
            "Tail filter z>%.1f on %d R2 days before %s → %d tail days (%.1f%%)",
            zscore_threshold, len(r_r2), as_of_date,
            n_tail, 100 * n_tail / len(r_r2),
        )

        if n_tail < MIN_TAIL_DAYS:
            # Too few tail days — fall back to full-R2 calibration
            logger.warning(
                "Only %d tail days (z > %.1f) — falling back to full-R2 MLE.",
                n_tail, zscore_threshold,
            )
            params = calibrate_jump_mle(r_r2, sigma_d_daily, bounds=_BOUNDS_FULL)
            self._jump_tail_cache[cache_key] = params
            return params

        # ── MLE on tail only ──────────────────────────────────────────────────
        r_tail     = r_r2[tail_mask]
        sigma_tail = sigma_d_daily[tail_mask]

        logger.info(
            "Running tail MLE on %d days (mean_err=%.2f%%, mean_ret=%.2f%%) ...",
            n_tail,
            error[tail_mask].mean() * 100,
            r_tail.mean() * 100,
        )
        params = calibrate_jump_mle(r_tail, sigma_tail, bounds=_BOUNDS_TAIL)
        params = MertonJumpParams(
            lam=params.lam, mu_j=params.mu_j, sigma_j=params.sigma_j,
            n_r2_days=n_tail, nll=params.nll,
        )
        logger.info("Tail-calibrated: %s", params)
        self._jump_tail_cache[cache_key] = params
        return params

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast(self, as_of_date: str) -> dict:
        """
        Return regime-switched vol forecast for `as_of_date`.

        Logic
        -----
        1. Look up features at `as_of_date` in PDV model.
        2. Compute σ_PDV via the fitted linear model.
        3. Look up regime at `as_of_date` in regime_labels.
        4. If R2: use tail-calibrated jump params (zero look-ahead).
           If R0/R1: no adjustment.

        Parameters
        ----------
        as_of_date : str 'YYYY-MM-DD'

        Returns
        -------
        dict with keys: pdv_vol, jump_adj, total_vol, regime,
                        lambda, mu_j, sigma_j, n_r2_cal_days, as_of_date
        """
        if not self.is_loaded:
            raise RuntimeError("Call .load() before .forecast().")

        dt = pd.to_datetime(as_of_date)

        # ── PDV diffusion vol ──────────────────────────────────────────────────
        feats = self._pdv._features.copy()
        feats.index = pd.to_datetime(feats.index)

        if dt not in feats.index:
            prior = feats.index[feats.index <= dt]
            if len(prior) == 0:
                raise ValueError(
                    f"No PDV features available on or before {as_of_date}."
                )
            dt_feat = prior[-1]
            logger.warning(
                "%s not in PDV features; using nearest prior date %s.",
                as_of_date, dt_feat.date(),
            )
        else:
            dt_feat = dt

        feat_row = feats.loc[[dt_feat], ["sigma1", "sigma2", "lev"]]
        pdv_vol  = float(self._pdv.linear_.predict(feat_row).iloc[0])

        # ── Regime ────────────────────────────────────────────────────────────
        lbl = self._regime_labels
        if dt in lbl.index:
            regime_int = int(lbl.loc[dt, "regime"])
        else:
            prior_lbl = lbl.index[lbl.index <= dt]
            if len(prior_lbl) == 0:
                regime_int = 1
                logger.warning(
                    "No regime label on or before %s; defaulting to R1.", as_of_date
                )
            else:
                regime_int = int(lbl.loc[prior_lbl[-1], "regime"])
                logger.warning(
                    "No regime label for %s; using nearest prior %s.",
                    as_of_date, prior_lbl[-1].date(),
                )

        regime_name = REGIME_NAMES[regime_int]

        # ── Jump adjustment (tail-calibrated) ─────────────────────────────────
        if regime_int == 2:
            params    = self.calibrate_jump_tail(as_of_date)
            jump_var  = params.jump_variance_annual
            total_vol = float(np.sqrt(pdv_vol ** 2 + jump_var))
            jump_adj  = float(total_vol - pdv_vol)
            lam       = params.lam
            mu_j      = params.mu_j
            sigma_j   = params.sigma_j
            n_cal     = params.n_r2_days
        else:
            jump_adj  = 0.0
            total_vol = pdv_vol
            lam       = float("nan")
            mu_j      = float("nan")
            sigma_j   = float("nan")
            n_cal     = 0

        return {
            "pdv_vol":        pdv_vol,
            "jump_adj":       jump_adj,
            "total_vol":      total_vol,
            "regime":         regime_name,
            "lambda":         lam,
            "mu_j":           mu_j,
            "sigma_j":        sigma_j,
            "n_r2_cal_days":  n_cal,
            "as_of_date":     as_of_date,
        }

    # ── COVID comparison ──────────────────────────────────────────────────────

    def compare_covid_2020(
        self,
        start: str = "2020-01-02",
        end:   str = "2020-12-31",
    ) -> pd.DataFrame:
        """
        Compare plain PDV vs regime-switched (tail-Merton) PDV vs actual RV over 2020.

        For each trading day in [start, end]:
          - Uses ONLY data available strictly before that date for jump calibration
            (zero look-ahead on the calibration window)
          - Jump parameters are tail-calibrated (z-score filter on PDV errors)
          - PDV features at date t already use only returns through t

        Returns
        -------
        pd.DataFrame with columns:
          date, regime, pdv_vol, jump_adj, total_vol, rv_actual_20d,
          error_pdv, error_jump, improvement
        where:
          rv_actual_20d : 20-day backward-looking RV (no look-ahead)
          error_pdv     : abs(pdv_vol - rv_actual_20d)
          error_jump    : abs(total_vol - rv_actual_20d)
          improvement   : error_pdv - error_jump  (>0 means jump model helps)
        """
        if not self.is_loaded:
            raise RuntimeError("Call .load() before .compare_covid_2020().")

        feats = self._pdv._features.copy()
        feats.index = pd.to_datetime(feats.index)
        date_range = feats.loc[start:end].index

        rows = []
        # Cache tail params per quarter (zero look-ahead: calibrate at quarter start)
        _qcache: Dict[str, MertonJumpParams] = {}

        for dt in date_range:
            dt_str = dt.strftime("%Y-%m-%d")

            quarter_key = f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
            if quarter_key not in _qcache:
                _qcache[quarter_key] = self.calibrate_jump_tail(dt_str)
            jparams = _qcache[quarter_key]

            # PDV vol
            feat_row = feats.loc[[dt], ["sigma1", "sigma2", "lev"]]
            pdv_vol  = float(self._pdv.linear_.predict(feat_row).iloc[0])

            # Regime
            lbl = self._regime_labels
            if dt in lbl.index:
                regime_int = int(lbl.loc[dt, "regime"])
            else:
                regime_int = 1
            regime_name = REGIME_NAMES[regime_int]

            # Total vol: apply jump only in R2 with valid (lam > 0) params
            if regime_int == 2 and jparams.lam > 0:
                jump_var  = jparams.jump_variance_annual
                total_vol = float(np.sqrt(pdv_vol ** 2 + jump_var))
                jump_adj  = float(total_vol - pdv_vol)
            else:
                total_vol = pdv_vol
                jump_adj  = 0.0

            rv_actual = float(feats.loc[dt, "rv_hist_20d"])

            rows.append({
                "date":          dt_str,
                "regime":        regime_name,
                "pdv_vol":       pdv_vol,
                "jump_adj":      jump_adj,
                "total_vol":     total_vol,
                "rv_actual_20d": rv_actual,
                "error_pdv":     abs(pdv_vol  - rv_actual),
                "error_jump":    abs(total_vol - rv_actual),
                "improvement":   abs(pdv_vol  - rv_actual) - abs(total_vol - rv_actual),
            })

        df = pd.DataFrame(rows).set_index("date")
        df.index = pd.to_datetime(df.index)

        # Summary log
        r2_rows = df[df["regime"] == "VOMMA_ACTIVE"]
        n_r2    = len(r2_rows)
        if n_r2 > 0:
            mae_pdv  = r2_rows["error_pdv"].mean()
            mae_jump = r2_rows["error_jump"].mean()
            logger.info(
                "COVID 2020 comparison on %d R2 days: "
                "MAE_PDV=%.2f%% → MAE_jump=%.2f%% (Δ=%.2f pp)",
                n_r2, mae_pdv * 100, mae_jump * 100,
                (mae_pdv - mae_jump) * 100,
            )

        return df

    # ── Convenience ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "RegimePDV(not loaded)"
        n_r2  = (self._regime_labels["regime"] == 2).sum()
        n_lbl = len(self._regime_labels)
        return (
            f"RegimePDV(loaded | "
            f"regime_labels={n_lbl} rows | R2={n_r2} | "
            f"jump_cache={len(self._jump_cache)} | "
            f"tail_cache={len(self._jump_tail_cache)} | "
            f"linear={self._pdv.linear_})"
        )
