"""
regime_classifier.py — C8: Volatility Regime Labels and (research) Classifier

Two things live here, and they must not be confused:

  1. The deterministic RULE label (build_regime_labels). Every trading path
     (the backtest, the signal engine, the daily bot, the dashboard headline)
     uses this label, lagged one session. It is the only regime that trades.
  2. An XGBoost classifier that tries to predict tomorrow's rule label from
     today's features. It loses to "predict yesterday's label" and is
     research-only (C16/C17 demotion); see classifier_accuracy_report().

Three regimes (descriptive states, not profitability regimes)
-------------------------------------------------------------
  REGIME 0  LONG_GAMMA    : trailing realised vol above implied
                             rv_20d(t) > VIX(t)/100  AND  not R2
  REGIME 1  SHORT_GAMMA   : implied vol at or above trailing realised
                             rv_20d(t) ≤ VIX(t)/100  AND  not R2
  REGIME 2  VOMMA_ACTIVE  : elevated vol-of-vol (overrides 0/1)
                             VVIX(t) > gate(t)

  The names are historical. R0/R1 describe the TRAILING 20-day window; they
  are not forecasts. Measured ex post (2010-2025, canonical gate, forward RV
  over t+1..t+21): after an R0 day forward realised vol still ends up BELOW
  VIX 81% of the time (P(fwd RV > VIX) = 0.19 vs 0.15 after R1; mean gap
  −3.3 vs −3.5 vol points). With the old fixed-100 gate R0 was even lower
  than R1 (0.14 vs 0.16). So "R0 → long gamma is profitable" is not
  supported; regime_forward_premium_table() recomputes the check on any data.

  The R2 gate (ONE gate, used everywhere)
  ---------------------------------------
  gate(t) = rolling 252-session 80th percentile of VVIX, using VVIX prints
  dated ≤ t only (min 63 prints). See R2_GATE / r2_gate_threshold(). A single
  missing VVIX print is carried forward one session (VVIX_FFILL_LIMIT); longer
  gaps leave the gate state, and therefore the label, unknown.

  History: until C22 the backtest used a FIXED VVIX > 100 gate (and the signal
  engine OR-ed a second hard-coded VVIX > 100 override on top of whatever
  labels it was given), while the published labels used the adaptive gate.
  100 is not "the 75th percentile of VVIX history" as was claimed: it is the
  ~67th percentile of 2010-2026 VVIX and the ~52nd of the 2018-01..2025-03
  backtest window, so the fixed gate labelled about half of the backtest
  window R2. VVIX_REGIME2_THRESHOLD is kept only for research comparisons.

Features (computed at close of t; every consumer lags them one session)
------------------------------------------------------------------------
  vix             : ^VIX / 100                  (implied vol level, decimal)
  ts_slope        : (^VIX3M − ^VIX) / 100       (term structure slope; < 0 = backwardation)
  fear_premium    : VIX / rv_20d                 (> 1 = implied above trailing realised)
  rv_change_5d    : rv_5d(t) − rv_5d(t−5)       (realized vol momentum)
  pdv_iv_spread   : σ̂(t) − VIX(t)/100           (vol forecast minus market IV)
  vvix            : ^VVIX / 100                  (vol-of-vol level, decimal)

  σ̂ is whatever forecast is injected through build_features(pdv_vol_series=...)
  (annualised decimal σ at the VIX horizon). The canonical one is
  pdv_vol_series=PDV_WALK_FORWARD: walk_forward_pdv_vol(), the purged
  walk-forward PDVLinear forecast of the next-21-session RMS vol (the default
  in SignalEngine.generate and the backtest). Without one, the trailing rv_20d
  is used as a proxy (build_features' default, kept for the live label job). "IVR" in older docs referred to this spread; no
  implied-vol RANK is computed anywhere.

Classifier (research only)
--------------------------
  XGBoost multi-class (objective='multi:softprob', num_class=3)
  Train:  2010 → 2019-12-31   Test: 2020-01-01 → 2025-12-31
  Class imbalance handled via sample_weight (sklearn 'balanced'), so the RAW
  booster output is not a calibrated probability; the shipped model adds an
  isotonic map fit on a later holdout it never trained on (RegimeClassifier).
  Accuracy is never hard-coded: classifier_accuracy_report() computes it
  (single split, per-year walk-forward, benchmarks, persistence on the same
  rows), and the shipped pickle records its own OOS accuracy and baseline in
  metadata_["evaluation"].

Zero look-ahead
---------------
  Feature X(t) uses data available at close of day t.
  Label y(t) is the regime defined by same-day observables (no future data).
  Training uses X_shift = X.shift(1) so that X(t-1) → y(t).
  At inference: use X(t) to predict y(t+1).

Outputs
-------
  data_store/signals/regime_classifier.pkl            (research classifier)
  data_store/signals/regime_classifier_predictions.parquet
  data_store/signals/regime_confusion_matrix.png
  data_store/signals/regime_distribution.png
  The RULE labels in data_store/signals/regime_labels.parquet are written by
  scripts/cloud_refresh.py, never by the classifier pipeline.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.data.database import (
    get_spx_ohlcv,
    get_vix_term_structure_wide,
    insert_regime_labels,
)
from joint_vol_calibration.models.pdv import (
    VIX_HORIZON_DAYS,
    extract_pdv_features,
    walk_forward_vol_forecast,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REGIME_NAMES: Dict[int, str] = {
    0: "LONG_GAMMA",
    1: "SHORT_GAMMA",
    2: "VOMMA_ACTIVE",
}

# ── The R2 gate: ONE definition, used by every label builder ──────────────────
# R2_GATE is the default `vvix_threshold` of build_regime_labels and of every
# function that builds labels through it. None selects the ADAPTIVE gate:
# VVIX(t) above the rolling R2_GATE_WINDOW-session R2_GATE_QUANTILE quantile of
# VVIX prints dated ≤ t. A float selects a fixed level (research only).
R2_GATE: Optional[float] = None
R2_GATE_WINDOW:      int   = 252
R2_GATE_QUANTILE:    float = 0.80
R2_GATE_MIN_PERIODS: int   = 63
# A single missing VVIX print (10 such sessions in 2010-2026) is carried
# forward from the previous session; this is causal (the last print is known
# at t). Longer outages leave the gate state unknown instead of failing open.
VVIX_FFILL_LIMIT:    int   = 1

# Legacy FIXED gate. NOT used by any trading path since C22; kept so research
# code can reproduce the pre-C22 labels (vvix_threshold=VVIX_REGIME2_THRESHOLD).
# It is the ~67th percentile of 2010-2026 VVIX, not the 75th as once stated.
VVIX_REGIME2_THRESHOLD: float = 100.0
TRAIN_END_DATE:  str = "2019-12-31"
TEST_START_DATE: str = "2020-01-01"
TEST_END_DATE:   str = "2025-12-31"

FEATURE_COLS: List[str] = [
    "vix",
    "ts_slope",
    "fear_premium",
    "rv_change_5d",
    "pdv_iv_spread",
    "vvix",
]

# Training feature set for the research classifier: excludes 'vvix'.
# Features are lagged (X(t−1) → y(t)), so vvix(t−1) is a LEGITIMATE predictor
# of the R2 gate at t — it carries the same information the persistence
# baseline uses, and with it the model scores ~87% OOS, still below
# persistence. It is excluded so the benchmark answers a narrower question:
# do the OTHER features carry regime information beyond the gate variable?
# (Earlier docs called vvix "circular"; that is only true for unlagged
# features.) build_features() still outputs vvix (Signal 3 and the HMM use
# it), and pickled classifiers select columns via feature_names_.
CLASSIFIER_FEATURE_COLS: List[str] = [c for c in FEATURE_COLS if c != "vvix"]

# Stress dates that must be classified as Regime 2
REGIME2_VALIDATION_DATES: List[str] = [
    "2020-03-16",   # COVID crash: VIX=82.7, VVIX=207.6
    "2025-04-09",   # Tariff spike: VIX=33.6, VVIX=142.5
]

# ── Paths ─────────────────────────────────────────────────────────────────────

SIGNALS_DIR        = DATA_DIR / "signals"
MODEL_PATH         = SIGNALS_DIR / "regime_classifier.pkl"
# RULE labels (written by scripts/cloud_refresh.py; read by the dashboard and
# regime_pdv). The classifier pipeline must never write here.
LABELS_PARQUET     = SIGNALS_DIR / "regime_labels.parquet"
# Research classifier PREDICTIONS (save_regime_labels default).
CLASSIFIER_PREDICTIONS_PARQUET = SIGNALS_DIR / "regime_classifier_predictions.parquet"
CONFUSION_PLOT     = SIGNALS_DIR / "regime_confusion_matrix.png"
DISTRIBUTION_PLOT  = SIGNALS_DIR / "regime_distribution.png"

# ── XGBoost hyperparameters ───────────────────────────────────────────────────

XGB_PARAMS: dict = {
    "objective":          "multi:softprob",
    "num_class":          3,
    "n_estimators":       300,
    "max_depth":          4,
    "learning_rate":      0.05,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "min_child_weight":   10,
    "gamma":              1.0,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "use_label_encoder":  False,
    "eval_metric":        "mlogloss",
    "random_state":       RANDOM_SEED,
    "n_jobs":             -1,
    "verbosity":          0,
}


# ── Market-data helpers ───────────────────────────────────────────────────────

def _join_market(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    spx_cols: List[str],
    vix_cols: List[str],
) -> pd.DataFrame:
    """Inner-join SPX and VIX-TS columns on date (SPX sessions drive the index)."""
    spx = spx_df.copy()
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.set_index("date").sort_index()

    vix = vix_wide_df.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date").sort_index()

    return spx[spx_cols].join(
        vix[[c for c in vix_cols if c in vix.columns]],
        how="inner",
    )


def _realised_vol(log_ret: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """Trailing annualised realised vol sqrt(mean(r²)·252) over `window` sessions."""
    return np.sqrt((log_ret ** 2).rolling(window, min_periods=min_periods).mean() * 252)


def _label_rv(log_ret: pd.Series) -> pd.Series:
    """The rv_20d used by the RULE label (min 15 of 20 returns)."""
    return _realised_vol(log_ret, 20, min_periods=15)


def fill_vvix(vvix: pd.Series) -> pd.Series:
    """Carry a missing VVIX print forward at most VVIX_FFILL_LIMIT sessions.

    Causal: the value used at t is the last print dated ≤ t. Without this a
    single missing print made the R2 gate fail OPEN (NaN > x is False, so an
    R2 day between two R2 days was labelled R1) and, in the signal engine,
    dropped the following session from the backtest entirely.
    """
    return vvix.ffill(limit=VVIX_FFILL_LIMIT)


def r2_gate_threshold(
    vvix: pd.Series,
    vvix_threshold: Optional[float] = R2_GATE,
) -> pd.Series:
    """Per-session VVIX level (index points) above which the session is R2.

    vvix_threshold=None (the canonical R2_GATE) gives the adaptive gate: the
    rolling R2_GATE_WINDOW-session R2_GATE_QUANTILE quantile of the VVIX
    prints dated ≤ t (missing prints are skipped, not filled, so the gate is
    the same whether or not a gap is later back-filled). A float gives a
    constant level. NaN where the gate is not yet defined (adaptive warm-up).
    """
    vvix = pd.to_numeric(vvix, errors="coerce")
    if vvix_threshold is None:
        return vvix.rolling(R2_GATE_WINDOW, min_periods=R2_GATE_MIN_PERIODS).quantile(
            R2_GATE_QUANTILE
        )
    return pd.Series(float(vvix_threshold), index=vvix.index)


def r2_gate_state(
    vvix: pd.Series,
    vvix_threshold: Optional[float] = R2_GATE,
) -> pd.Series:
    """R2 gate state per session: 1.0 = R2, 0.0 = not R2, NaN = unknown.

    Unknown when the (one-session forward-filled) VVIX print or the gate
    itself is missing — the caller decides what to do with an unknown day
    instead of silently treating it as "not R2".
    """
    level = fill_vvix(pd.to_numeric(vvix, errors="coerce"))
    gate = r2_gate_threshold(vvix, vvix_threshold)
    state = (level > gate).astype(float)
    state[level.isna() | gate.isna()] = np.nan
    return state


def gate_name(vvix_threshold: Optional[float] = R2_GATE) -> str:
    """Human-readable provenance tag for a gate setting."""
    if vvix_threshold is None:
        return (f"adaptive_p{int(round(R2_GATE_QUANTILE * 100))}"
                f"_{R2_GATE_WINDOW}d")
    return f"fixed_{float(vvix_threshold):g}"


# ── Feature engineering ───────────────────────────────────────────────────────

# Provenance tags written to build_features(...).attrs["pdv_iv_spread_source"]
PDV_SOURCE_INJECTED = "injected_forecast"
PDV_SOURCE_MODEL    = "pdv_model_full_sample"
PDV_SOURCE_RV_PROXY = "rv20_proxy"
PDV_SOURCE_WALK_FORWARD = "pdv21_walk_forward"

# Pass as build_features(pdv_vol_series=PDV_WALK_FORWARD) (or as
# SignalEngine.generate(precomputed_pdv_vol=PDV_WALK_FORWARD), its default) to
# have the σ̂ built here by walk_forward_pdv_vol(): the purged walk-forward
# PDVLinear forecast of the next-21-session RMS vol (variance-consistent and
# horizon-matched with the 30-day VIX), dated at its forecast origin.
PDV_WALK_FORWARD = "walk_forward"


def _spx_log_returns(spx_df: pd.DataFrame) -> pd.Series:
    """Daily SPX log returns rebuilt from closes, one per session.

    The DB's log_return column has isolated NaNs where both closes exist
    (e.g. 2025-05-08), which walk_forward_vol_forecast would DROP, silently
    joining two returns into one session. Closes are the source of truth;
    the stored column is used only when no close column exists.
    """
    spx = spx_df.copy()
    if "date" in spx.columns:
        spx = spx.set_index(pd.to_datetime(spx["date"]))
    spx.index = pd.to_datetime(spx.index)
    spx = spx[~spx.index.duplicated(keep="last")].sort_index()
    if "close" in spx.columns:
        r = np.log(spx["close"].astype(float)).diff()
    else:
        r = spx["log_return"].astype(float)
    return r.dropna().rename("log_return")


def walk_forward_pdv_vol(
    spx_df: pd.DataFrame,
    horizon: int = VIX_HORIZON_DAYS,
    **kwargs,
) -> pd.Series:
    """
    The σ̂ behind pdv_iv_spread: models.pdv.walk_forward_vol_forecast on the
    SPX returns in ``spx_df``.

    Row t is the forecast, made at the close of t from returns dated ≤ t and
    a PDVLinear refit daily on labels fully observed by t (purged, embargo ≥
    horizon), of sqrt(252/h · Σ_{i=1..h} r²_{t+i}) — the annualised RMS vol of
    the next ``horizon`` sessions (21 = the 30-calendar-day VIX horizon), so
    σ̂(t) − VIX(t)/100 compares like with like. Sessions before the first
    forecast (burn-in + train_min_days + embargo ≈ 585 sessions of history)
    are absent; build_features fills them with the rv_20d proxy and counts
    them in .attrs["pdv_iv_spread_n_proxy"].

    kwargs are passed to walk_forward_vol_forecast (train_min_days, …).
    """
    r = _spx_log_returns(spx_df)
    if len(r) == 0:
        return pd.Series(dtype=float, name=f"pdv_vol_{int(horizon)}d")
    return walk_forward_vol_forecast(r, horizon=horizon, **kwargs)


def _validate_vol_forecast(pdv_vol_series: pd.Series) -> pd.Series:
    """Coerce an injected σ forecast to a clean, date-indexed float Series."""
    if not isinstance(pdv_vol_series, pd.Series):
        raise TypeError("pdv_vol_series must be a date-indexed pd.Series of "
                        "annualised decimal vol forecasts")
    s = pd.to_numeric(pdv_vol_series, errors="coerce").copy()
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    finite = s.dropna()
    if len(finite) and float(finite.median()) > 3.0:
        # 3.0 = 300% annualised vol: anything above is a percent-unit series
        raise ValueError(
            "pdv_vol_series looks like vol POINTS (median %.2f); pass annualised "
            "decimal vol, e.g. 0.18 for 18%%" % float(finite.median())
        )
    if len(finite) and float(finite.min()) < 0.0:
        raise ValueError("pdv_vol_series has negative values; it must be a vol (σ), "
                         "not a variance difference or a spread")
    return s


def build_features(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
    pdv_vol_series: "Optional[pd.Series]" = None,
) -> pd.DataFrame:
    """
    Build the six-column feature matrix for the regime classifier and signals.

    Parameters
    ----------
    spx_df      : SPX OHLCV DataFrame with columns ['date', 'close', 'log_return']
    vix_wide_df : Wide VIX TS DataFrame with columns ['^VIX', '^VIX3M', '^VVIX', ...]
    pdv_model   : Optional fitted PDVModel. Its full-sample coefficients see
                  the future, so it must NOT be used for backtest or evaluation
                  features (kept for live/legacy callers only). Ignored when
                  pdv_vol_series is given.
    pdv_vol_series : THE injection hook for the pdv_iv_spread feature.
                  PDV_WALK_FORWARD ("walk_forward") builds the canonical σ̂
                  here with walk_forward_pdv_vol(spx_df) (21-session RMS-vol
                  forecast, purged walk-forward; source tag
                  PDV_SOURCE_WALK_FORWARD). Otherwise
                  a date-indexed Series of vol forecasts σ̂(t) that:
                    • are annualised DECIMAL vol (σ, not variance, not points),
                    • target the VIX horizon (30 calendar days ≈ 21 sessions),
                      so σ̂ − VIX/100 compares like with like,
                    • use only information dated ≤ t (walk-forward fit).
                  Sessions without a forecast (e.g. warm-up) fall back to the
                  trailing rv_20d proxy; how many did is recorded in
                  .attrs["pdv_iv_spread_n_proxy"].

    Returns
    -------
    pd.DataFrame indexed by date with columns = FEATURE_COLS, values at close
    of t (unshifted — every consumer lags them one session). NaN rows remain
    for early dates (rolling-window warm-up). `.attrs["pdv_iv_spread_source"]`
    records which σ̂ the spread was built from.
    """
    df = _join_market(spx_df, vix_wide_df, ["close", "log_return"],
                      ["^VIX", "^VIX3M", "^VIX6M", "^VVIX"])

    log_ret = df["log_return"]

    # ── Realized vol rolling windows ─────────────────────────────────────────
    rv_20d = _realised_vol(log_ret, 20)
    rv_5d  = _realised_vol(log_ret, 5)

    # ── Feature 1: VIX level ──────────────────────────────────────────────────
    df["vix"] = df["^VIX"] / 100.0

    # ── Feature 2: VIX term structure slope (3M − 1M) ────────────────────────
    if "^VIX3M" in df.columns and "^VIX" in df.columns:
        df["ts_slope"] = (df["^VIX3M"] - df["^VIX"]) / 100.0
    else:
        df["ts_slope"] = 0.0

    # ── Feature 3: Fear premium  VIX / rv_20d ────────────────────────────────
    # Clip to avoid division by zero / extreme values
    df["fear_premium"] = (df["^VIX"] / 100.0) / rv_20d.clip(lower=0.01)
    df["fear_premium"] = df["fear_premium"].clip(upper=10.0)

    # ── Feature 4: 5-day realized vol change (RV momentum) ───────────────────
    df["rv_change_5d"] = rv_5d - rv_5d.shift(5)

    # ── Feature 5: vol forecast vs ATM IV spread ─────────────────────────────
    rv_proxy_spread = rv_20d - df["^VIX"] / 100.0
    n_proxy = 0
    source_injected = PDV_SOURCE_INJECTED
    if isinstance(pdv_vol_series, str):
        if pdv_vol_series != PDV_WALK_FORWARD:
            raise ValueError(f"pdv_vol_series={pdv_vol_series!r}: pass a Series, None "
                             f"or PDV_WALK_FORWARD ({PDV_WALK_FORWARD!r})")
        pdv_vol_series = walk_forward_pdv_vol(spx_df)
        source_injected = PDV_SOURCE_WALK_FORWARD
    if pdv_vol_series is not None:
        # Walk-forward forecasts (zero look-ahead: σ̂(t) uses data dated ≤ t).
        pdv_vol = _validate_vol_forecast(pdv_vol_series).reindex(df.index)
        spread  = pdv_vol - df["^VIX"] / 100.0
        use_proxy = spread.isna() & rv_proxy_spread.notna()
        n_proxy = int(use_proxy.sum())
        df["pdv_iv_spread"] = spread.fillna(rv_proxy_spread)
        source = source_injected
        if n_proxy:
            logger.info("pdv_iv_spread: %d session(s) without an injected forecast "
                        "use the rv_20d proxy", n_proxy)
    elif pdv_model is not None:
        try:
            log_ret_s = log_ret.dropna()
            pdv_feats  = extract_pdv_features(log_ret_s)
            pdv_vol    = pdv_model.linear_.predict(pdv_feats[["sigma1", "sigma2", "lev"]])
            pdv_vol    = pdv_vol.reindex(df.index)
            df["pdv_iv_spread"] = pdv_vol - df["^VIX"] / 100.0
            source = PDV_SOURCE_MODEL
        except Exception as exc:
            logger.warning("PDV spread feature failed (%s); using rv_20d fallback", exc)
            df["pdv_iv_spread"] = rv_proxy_spread
            source = PDV_SOURCE_RV_PROXY
    else:
        # Fallback: (trailing rv_20d − VIX) is a realized-vs-implied proxy
        df["pdv_iv_spread"] = rv_proxy_spread
        source = PDV_SOURCE_RV_PROXY

    # ── Feature 6: VVIX level (one missing print carried forward) ────────────
    if "^VVIX" in df.columns:
        df["vvix"] = fill_vvix(df["^VVIX"]) / 100.0
    else:
        df["vvix"] = 0.0

    out = df[FEATURE_COLS].copy()
    out.attrs["pdv_iv_spread_source"] = source
    out.attrs["pdv_iv_spread_n_proxy"] = n_proxy
    return out


# ── Regime label generation ────────────────────────────────────────────────────

def build_regime_labels(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    vvix_threshold: "Optional[float]" = R2_GATE,
) -> pd.Series:
    """
    Assign rule-based regime labels to each trading day.

    Rules (applied in priority order)
    ----------------------------------
    REGIME 2  : VVIX(t) > gate(t)           (overrides 0/1)
    REGIME 0  : rv_20d(t) > VIX(t)/100      (trailing realized > implied)
    REGIME 1  : rv_20d(t) ≤ VIX(t)/100      (implied ≥ trailing realized)

    Labels are defined by same-day observables — no look-ahead bias.

    Parameters
    ----------
    vvix_threshold : None (default, = R2_GATE) for the canonical ADAPTIVE gate
        — rolling 252-session 80th percentile of VVIX dated ≤ t (≈23% R2 days
        in 2018-01..2025-03); or a float for a FIXED level (research only;
        100.0 reproduces the pre-C22 backtest labels, ≈48% R2 in that window).

    Sessions are dropped (not guessed) when rv_20d lacks 15 returns, or when
    the R2 gate state is unknown: adaptive-gate warm-up (< 63 VVIX prints) or
    a VVIX outage longer than VVIX_FFILL_LIMIT sessions.

    Returns
    -------
    pd.Series[int] indexed by date (Timestamp), values in {0, 1, 2}.
    """
    df = _join_market(spx_df, vix_wide_df, ["log_return"], ["^VIX", "^VVIX"])

    rv_20d = _label_rv(df["log_return"])
    vix_iv = df["^VIX"] / 100.0

    if "^VVIX" in df.columns:
        r2 = r2_gate_state(df["^VVIX"], vvix_threshold)
    else:
        logger.warning("No ^VVIX column: R2 gate cannot fire; labels are R0/R1 only")
        r2 = pd.Series(0.0, index=df.index)

    # Default: Regime 1 (short gamma / neutral)
    labels = pd.Series(1.0, index=df.index)

    # Regime 0: trailing realized vol exceeds implied vol
    labels[rv_20d > vix_iv] = 0.0

    # Regime 2: elevated vol-of-vol (overrides 0/1)
    labels[r2 == 1.0] = 2.0

    # Unknown: insufficient rv history, unknown gate, or (non-R2) missing VIX
    labels[rv_20d.isna() | r2.isna() | ((r2 == 0.0) & vix_iv.isna())] = np.nan

    return labels.dropna().astype(int)


def build_rule_label_frame(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    vvix_threshold: "Optional[float]" = R2_GATE,
    pdv_vol_series: "Optional[pd.Series]" = None,
) -> pd.DataFrame:
    """Same-day (UNSHIFTED) features, gate and RULE label, one row per session.

    The one place that pairs a session's rule label with the observables that
    produced it — for label files, the regime_labels DB table and dashboards.
    Columns: FEATURE_COLS + rv_20d (the label's own realised vol) +
    vvix_gate (index points) + regime (int). Only labelled sessions are kept.
    """
    feats = build_features(spx_df, vix_wide_df, pdv_vol_series=pdv_vol_series)
    labels = build_regime_labels(spx_df, vix_wide_df, vvix_threshold)
    mkt = _join_market(spx_df, vix_wide_df, ["log_return"], ["^VVIX"])
    out = feats.reindex(labels.index).copy()
    out["rv_20d"] = _label_rv(mkt["log_return"]).reindex(labels.index)
    if "^VVIX" in mkt.columns:
        out["vvix_gate"] = r2_gate_threshold(mkt["^VVIX"], vvix_threshold).reindex(labels.index)
    else:
        out["vvix_gate"] = np.nan
    out["regime"] = labels.astype(int)
    out.attrs["r2_gate"] = gate_name(vvix_threshold)
    out.attrs["pdv_iv_spread_source"] = feats.attrs.get("pdv_iv_spread_source")
    return out


def regime_label_summary(
    labels: pd.Series,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Share of each regime and the persistence baseline, on ONE label series.

    persistence_acc = P[y_t == y_{t−1}] over consecutive labelled sessions in
    [start, end] — the accuracy of "predict yesterday's label", which any
    classifier evaluated on these same labels must beat. Compute it on the
    labels that are actually displayed or traded, never on another gate's.
    """
    lab = labels.dropna().astype(int).sort_index()
    if start is not None:
        lab = lab[lab.index >= pd.Timestamp(start)]
    if end is not None:
        lab = lab[lab.index <= pd.Timestamp(end)]
    n = int(len(lab))
    counts = {k: int((lab == k).sum()) for k in (0, 1, 2)}
    persist = float((lab.shift(1) == lab).iloc[1:].mean()) if n > 1 else float("nan")
    return {
        "n":               n,
        "start":           str(lab.index[0].date()) if n else None,
        "end":             str(lab.index[-1].date()) if n else None,
        "counts":          counts,
        "shares":          {k: (counts[k] / n if n else float("nan")) for k in (0, 1, 2)},
        "persistence_acc": persist,
    }


def regime_forward_premium_table(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    horizon: int = 21,
) -> pd.DataFrame:
    """EX-POST check of the regime ontology's economic premise.

    For each rule regime: how often realised vol over the NEXT `horizon`
    sessions (t+1..t+h) exceeded VIX(t), the mean forward RV − VIX gap (vol
    points), mean VIX and the share of sessions in backwardation. If R0 is a
    "long gamma" state, P(fwd RV > VIX | R0) should exceed P(· | R1).

    This deliberately uses FUTURE returns — it is an evaluation table, never a
    feature. Labels default to build_regime_labels(spx_df, vix_wide_df).
    """
    if labels is None:
        labels = build_regime_labels(spx_df, vix_wide_df)
    df = _join_market(spx_df, vix_wide_df, ["log_return"], ["^VIX", "^VIX3M"])
    r2 = df["log_return"] ** 2
    # mean of r² over t+1..t+h (reverse rolling of the shifted series)
    fwd_var = r2[::-1].rolling(horizon, min_periods=horizon).mean()[::-1].shift(-1)
    fwd_rv = np.sqrt(fwd_var * 252)
    vix = df["^VIX"] / 100.0
    slope = (df["^VIX3M"] - df["^VIX"]) if "^VIX3M" in df.columns else pd.Series(np.nan, index=df.index)
    t = pd.DataFrame({"regime": labels.reindex(df.index), "fwd_rv": fwd_rv,
                      "vix": vix, "slope": slope}).dropna(subset=["regime", "fwd_rv", "vix"])
    t["regime"] = t["regime"].astype(int)
    t["fwd_gt_vix"] = (t["fwd_rv"] > t["vix"]).astype(float)
    t["gap_vp"] = (t["fwd_rv"] - t["vix"]) * 100.0
    t["backwardation"] = (t["slope"] < 0).astype(float)
    return t.groupby("regime").agg(
        n=("fwd_gt_vix", "size"),
        p_fwd_rv_gt_vix=("fwd_gt_vix", "mean"),
        mean_fwd_rv_minus_vix_vp=("gap_vp", "mean"),
        mean_vix=("vix", "mean"),
        share_backwardation=("backwardation", "mean"),
    )


def regime_label_noise_audit(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    vvix_threshold: Optional[float] = R2_GATE,
) -> dict:
    """Measure the NOISE FLOOR of the R0/R1/R2 label ontology.

    "The model cannot lower the noise floor — that's the optimal bound of the
    encoding of your data." In an ML project the labels set that floor, and here
    the labels are defined by same-day observables: rv_20d vs VIX/100 for R0/R1,
    and VVIX > gate for R2. Two consequences fall out, and both are the reason
    the C8 classifier was demoted rather than tuned:

      1. Because the label at t is a function of information known at t,
         "predict yesterday's label" (the persistence baseline) is a legitimate
         causal predictor and is very hard to beat.
      2. Days sitting on a decision boundary flip their label under ordinary
         measurement noise. That flip rate is a hard ceiling on ANY classifier
         trained on these labels, set by the ONTOLOGY, not the model.

    This function quantifies (2) by relabelling under perturbations that are all
    within the data's own measurement noise:
      - realized-vol window: 20d -> 19d and 21d,
      - VIX level: +/- 0.5 vol points.

    Returns a dict:
      n_days, boundary_within_1vp, boundary_within_0p5vp  (fraction of days near
      the R0/R1 cut), label_flip_rate, perturbation_stable (= 1 - flip_rate),
      r2_gate_fragile (fraction within 3 VVIX pts of the R2 gate),
      persistence_acc (P[y_t == y_{t-1}] over consecutive labelled sessions),
      r2_gate (provenance tag of the gate the labels were built with).

    The unperturbed labels are exactly build_regime_labels(..., vvix_threshold)
    — same gate, same VVIX gap handling — so the noise floor and persistence
    describe the labels that are actually used. The flip rate is a
    SENSITIVITY measure of the ontology, not a Bayes-error ceiling.
    """
    df = _join_market(spx_df, vix_wide_df, ["log_return"], ["^VIX", "^VVIX"])
    log_ret = df["log_return"]
    vix_iv  = df["^VIX"] / 100.0
    if "^VVIX" in df.columns:
        r2    = r2_gate_state(df["^VVIX"], vvix_threshold)
        vvix  = fill_vvix(df["^VVIX"])
        gate  = r2_gate_threshold(df["^VVIX"], vvix_threshold)
    else:
        r2    = pd.Series(0.0, index=df.index)
        vvix  = pd.Series(np.nan, index=df.index)
        gate  = pd.Series(np.nan, index=df.index)

    def _rv(w: int) -> pd.Series:
        return _realised_vol(log_ret, w, min_periods=max(w - 5, 1))

    def _label(rv: pd.Series, iv: pd.Series) -> pd.Series:
        lab = pd.Series(1.0, index=df.index)
        lab[rv > iv] = 0.0
        lab[r2 == 1.0] = 2.0       # R2 overrides, matches build_regime_labels
        lab[rv.isna() | r2.isna()] = np.nan
        return lab

    rv20 = _rv(20)
    base = _label(rv20, vix_iv)
    mask = base.notna()
    n    = int(mask.sum())
    gap  = (rv20 - vix_iv)[mask].abs()

    flip = pd.Series(False, index=df.index)
    for w in (19, 21):
        pert = _label(_rv(w), vix_iv)
        flip |= (pert != base) & pert.notna() & base.notna()
    for dv in (0.005, -0.005):
        pert = _label(rv20, vix_iv + dv)
        flip |= (pert != base) & pert.notna() & base.notna()
    flip = flip[mask]

    r2_fragile = float(((vvix[mask] - gate[mask]).abs() < 3.0).mean())
    persist    = regime_label_summary(base[mask])["persistence_acc"]

    return {
        "n_days":                n,
        "boundary_within_1vp":   round(float((gap < 0.01).mean()), 4),
        "boundary_within_0p5vp": round(float((gap < 0.005).mean()), 4),
        "label_flip_rate":       round(float(flip.mean()), 4),
        "perturbation_stable":   round(float(1.0 - flip.mean()), 4),
        "r2_gate_fragile":       round(r2_fragile, 4),
        "persistence_acc":       round(persist, 4),
        "r2_gate":               gate_name(vvix_threshold),
    }


# ── Probability calibration helpers (plain arrays → version-robust) ───────────

def _expected_calibration_error(p: np.ndarray, y01: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin expected calibration error of p against 0/1 outcomes."""
    p = np.asarray(p, dtype=float)
    y01 = np.asarray(y01, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y01[m].mean())
    return float(err)


def _fit_isotonic_ovr(proba: np.ndarray, y: np.ndarray, classes) -> List[Tuple[np.ndarray, np.ndarray]]:
    """One-vs-rest isotonic maps, stored as (x_thresholds, y_thresholds) arrays."""
    from sklearn.isotonic import IsotonicRegression

    maps = []
    for j, k in enumerate(classes):
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(proba[:, j], (np.asarray(y) == k).astype(float))
        maps.append((np.asarray(ir.X_thresholds_, dtype=float),
                     np.asarray(ir.y_thresholds_, dtype=float)))
    return maps


def _apply_isotonic_ovr(proba: np.ndarray, maps: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply one-vs-rest isotonic maps and renormalise rows to sum to 1."""
    cal = np.column_stack([np.interp(proba[:, j], x, yv) for j, (x, yv) in enumerate(maps)])
    tot = cal.sum(axis=1, keepdims=True)
    k = cal.shape[1]
    return np.where(tot > 0, cal / np.where(tot > 0, tot, 1.0), 1.0 / k)


def _window(index: pd.Index) -> Optional[Tuple[str, str]]:
    if len(index) == 0 or not isinstance(index, pd.DatetimeIndex):
        return None
    return (str(index.min().date()), str(index.max().date()))


# ── Regime classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    XGBoost multi-class classifier for the three volatility regimes.
    RESEARCH ONLY: it loses to the persistence baseline and drives nothing.

    Probabilities
    -------------
    calibration=None (fit default): predict_proba is the raw booster output.
        With the default 'balanced' sample weights it is tilted toward the
        rarer classes, so it is a score, NOT a calibrated frequency.
    calibration="holdout": the booster is fit on the earlier
        (1 − calibration_frac) of the time-ordered rows, and a one-vs-rest
        isotonic map is fit, UNWEIGHTED, on the later calibration_frac — rows
        the booster never saw. Pre-C22 the isotonic map was fit on the
        booster's own training rows with the balanced weights; it learned the
        overfit training scores and made out-of-sample P(R2) worse
        (2020-2025, adaptive-gate labels: ECE 0.150 raw → 0.177 calibrated).

    Persistence
    -----------
    Pickles store the booster as XGBoost's own JSON model plus plain Python
    lists, so a saved classifier does not depend on sklearn/xgboost private
    class layouts (the old CalibratedClassifierCV(FrozenEstimator) pickle
    needed sklearn ≥ 1.6 just to import). Pre-C22 pickles still load.

    Usage
    -----
      clf = RegimeClassifier()
      clf.fit(X_train, y_train, calibration="holdout")
      preds   = clf.predict(X_test)
      probas  = clf.predict_proba(X_test)   # shape (N, 3)
      clf.save(path)

      clf2 = RegimeClassifier.load(path)
    """

    _STATE_VERSION = 2

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS.copy()
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.train_end_date_: Optional[str] = None
        self.calibration_: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.metadata_: Dict = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_sample_weights: bool = True,
        calibration: Optional[str] = None,
        calibration_frac: float = 0.2,
    ) -> "RegimeClassifier":
        """
        Fit XGBoost on (X, y), optionally with a time-ordered holdout calibrator.

        Parameters
        ----------
        X                  : feature DataFrame (rows × feature columns)
        y                  : integer regime labels (0/1/2), aligned with X
        use_sample_weights : if True, balance classes via sklearn 'balanced'
                             weights (booster only; never the calibrator)
        calibration        : None or "holdout" (see class docstring)
        calibration_frac   : share of the LATEST rows held out for calibration
        """
        if calibration not in (None, "holdout"):
            raise ValueError("calibration must be None or 'holdout'")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.feature_names_ = list(X.columns)

        # Time order matters for the holdout: the calibrator must see LATER rows.
        if isinstance(X.index, pd.DatetimeIndex):
            order = np.argsort(X.index.values, kind="stable")
            X, y = X.iloc[order], y.iloc[order]

        n = len(X)
        n_cal = int(round(n * calibration_frac)) if calibration == "holdout" else 0
        if calibration == "holdout" and (n_cal < 20 or n - n_cal < 20):
            raise ValueError(
                f"holdout calibration needs ≥ 20 rows on each side (n={n}, "
                f"calibration_frac={calibration_frac})"
            )
        n_fit = n - n_cal
        X_fit, y_fit = X.iloc[:n_fit], y.iloc[:n_fit]

        sample_weight = None
        if use_sample_weights:
            sample_weight = compute_sample_weight("balanced", y_fit.values)

        model = xgb.XGBClassifier(**self.params)
        model.fit(X_fit.values, y_fit.values, sample_weight=sample_weight, verbose=False)
        self.model_ = model
        self.calibration_ = None

        meta = {
            "use_sample_weights": bool(use_sample_weights),
            "calibration":        calibration,
            "fit_rows":           int(n_fit),
            "fit_window":         _window(X_fit.index),
            "calibration_rows":   0,
            "calibration_window": None,
        }
        if n_cal:
            X_cal, y_cal = X.iloc[n_fit:], y.iloc[n_fit:]
            raw = model.predict_proba(X_cal.values)
            self.calibration_ = _fit_isotonic_ovr(raw, y_cal.values, model.classes_)
            meta["calibration_rows"] = int(n_cal)
            meta["calibration_window"] = _window(X_cal.index)
        self.metadata_.update(meta)

        logger.info(
            "RegimeClassifier fitted on %d rows (calibration=%s, %d holdout rows), classes %s",
            n_fit, calibration, n_cal, np.unique(y.values).tolist(),
        )
        return self

    # ── inference ────────────────────────────────────────────────────────────

    def _is_legacy(self) -> bool:
        """True for a pre-C22 pickle whose model_ is a sklearn wrapper."""
        return self.model_ is not None and not isinstance(self.model_, xgb.XGBClassifier)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape (N, 3) (see class docstring)."""
        self._require_fitted()
        Xv = X[self.feature_names_].values
        if self._is_legacy():
            return self.model_.predict_proba(Xv)
        proba = self.model_.predict_proba(Xv)
        if self.calibration_ is not None:
            proba = _apply_isotonic_ovr(proba, self.calibration_)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted regime labels (int array): argmax of predict_proba."""
        self._require_fitted()
        if self._is_legacy():
            return self.model_.predict(X[self.feature_names_].values)
        proba = self.predict_proba(X)
        return np.asarray(self.model_.classes_)[proba.argmax(axis=1)]

    def predict_series(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as a Series indexed like X."""
        return pd.Series(self.predict(X), index=X.index, name="regime")

    def predict_proba_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities as DataFrame indexed like X.

        Returns
        -------
        pd.DataFrame with columns ['prob_r0', 'prob_r1', 'prob_r2'], indexed like X.
        """
        proba = self.predict_proba(X)
        return pd.DataFrame(
            proba, index=X.index, columns=["prob_r0", "prob_r1", "prob_r2"]
        )

    def feature_importance(self) -> pd.Series:
        """Return XGBoost feature importances, indexed by feature name."""
        self._require_fitted()
        base_clf = self.model_
        if self._is_legacy():
            # CalibratedClassifierCV(FrozenEstimator(xgb)) → unwrap twice
            inner = getattr(base_clf, "estimator", base_clf)
            base_clf = getattr(inner, "estimator", inner)
        return pd.Series(
            base_clf.feature_importances_,
            index=self.feature_names_,
        ).sort_values(ascending=False)

    # ── persistence ──────────────────────────────────────────────────────────

    def __getstate__(self) -> dict:
        state = {
            "state_version":   self._STATE_VERSION,
            "params":          dict(self.params),
            "feature_names_":  list(self.feature_names_) if self.feature_names_ else None,
            "train_end_date_": self.train_end_date_,
            "metadata_":       dict(getattr(self, "metadata_", {}) or {}),
            "calibration_":    None,
            "booster_json":    None,
            "legacy_model":    None,
        }
        cal = getattr(self, "calibration_", None)
        if cal is not None:
            state["calibration_"] = [
                (np.asarray(x, float).tolist(), np.asarray(yv, float).tolist()) for x, yv in cal
            ]
        if isinstance(self.model_, xgb.XGBClassifier):
            state["booster_json"] = bytes(self.model_.get_booster().save_raw("json"))
        elif self.model_ is not None:
            state["legacy_model"] = self.model_
        return state

    def __setstate__(self, state: dict) -> None:
        if "state_version" not in state:
            # Pre-C22 pickle: a plain __dict__ holding a sklearn wrapper.
            self.__dict__.update(state)
            self.calibration_ = None
            self.metadata_ = {"legacy_pickle": True}
            return
        self.params = state["params"]
        self.feature_names_ = state["feature_names_"]
        self.train_end_date_ = state["train_end_date_"]
        self.metadata_ = state.get("metadata_") or {}
        cal = state.get("calibration_")
        self.calibration_ = (
            [(np.asarray(x, float), np.asarray(yv, float)) for x, yv in cal]
            if cal is not None else None
        )
        self.model_ = None
        if state.get("booster_json") is not None:
            model = xgb.XGBClassifier()
            model.load_model(bytearray(state["booster_json"]))
            self.model_ = model
        elif state.get("legacy_model") is not None:
            self.model_ = state["legacy_model"]

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the classifier (booster stored as XGBoost JSON)."""
        if path is None:
            path = MODEL_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=4)
        logger.info("RegimeClassifier saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RegimeClassifier":
        """Load a pickled RegimeClassifier."""
        if path is None:
            path = MODEL_PATH
        with open(path, "rb") as fh:
            clf = pickle.load(fh)
        if not isinstance(clf, cls):
            raise TypeError(f"Expected RegimeClassifier, got {type(clf)}")
        return clf

    def _require_fitted(self):
        if self.model_ is None:
            raise RuntimeError("RegimeClassifier not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.model_ is not None else "not fitted"
        cal = (getattr(self, "metadata_", {}) or {}).get("calibration")
        return (
            f"RegimeClassifier(n_estimators={self.params.get('n_estimators', '?')}, "
            f"max_depth={self.params.get('max_depth', '?')}, "
            f"calibration={cal}, status={status})"
        )


# ── Training pipeline ─────────────────────────────────────────────────────────

def build_dataset(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
    vvix_threshold: "Optional[float]" = R2_GATE,
    pdv_vol_series: "Optional[pd.Series]" = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) aligned dataset.

    Features are shifted by 1 day (X_{t-1} predicts y_t) to enforce
    zero look-ahead at inference time. Labels use the canonical R2 gate unless
    vvix_threshold says otherwise. For evaluation, pass walk-forward σ̂ via
    pdv_vol_series (or neither, for the rv proxy) — never a full-sample
    pdv_model, whose coefficients have seen the test period.

    Returns
    -------
    X : DataFrame — shifted feature matrix, NaN rows dropped
    y : Series    — regime labels aligned to X
    X.attrs carries "r2_gate" and "pdv_iv_spread_source" provenance.
    """
    feats  = build_features(spx_df, vix_wide_df, pdv_model, pdv_vol_series=pdv_vol_series)
    labels = build_regime_labels(spx_df, vix_wide_df, vvix_threshold)

    # Shift features forward 1 day: X(t-1) predicts y(t)
    X_shifted = feats.shift(1)

    # Align on common index, drop NaN
    combined = pd.concat([X_shifted, labels.rename("regime")], axis=1).dropna()
    X = combined[FEATURE_COLS].copy()
    y = combined["regime"].astype(int)
    X.attrs["r2_gate"] = gate_name(vvix_threshold)
    X.attrs["pdv_iv_spread_source"] = feats.attrs.get("pdv_iv_spread_source")

    return X, y


def train_test_split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Time-series aware train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    train_end_ts  = pd.Timestamp(train_end)
    test_start_ts = pd.Timestamp(test_start)

    train_mask = X.index <= train_end_ts
    test_mask  = X.index >= test_start_ts

    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
    )


# The configuration the shipped research classifier is trained with.
SHIPPED_FIT_KWARGS: dict = {"use_sample_weights": True, "calibration": "holdout"}


def train_classifier(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
    train_end: str = TRAIN_END_DATE,
    params: Optional[dict] = None,
    vvix_threshold: "Optional[float]" = R2_GATE,
    pdv_vol_series: "Optional[pd.Series]" = None,
    feature_cols: Optional[List[str]] = None,
    fit_kwargs: Optional[dict] = None,
) -> Tuple["RegimeClassifier", pd.DataFrame, pd.Series]:
    """
    Full training pipeline: build dataset → split → fit → return.

    Trains on CLASSIFIER_FEATURE_COLS (no vvix) by default with
    SHIPPED_FIT_KWARGS (balanced weights, time-later holdout calibration).
    clf.metadata_ records the gate, features and PDV source it was built on.

    Returns
    -------
    (clf, X_test, y_test)
    """
    logger.info("Building dataset ...")
    X, y = build_dataset(spx_df, vix_wide_df, pdv_model, vvix_threshold,
                         pdv_vol_series=pdv_vol_series)
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, train_end)

    logger.info(
        "Train: %d samples (%s → %s) | Test: %d samples (%s → %s)",
        len(X_train), X_train.index.min().date(), X_train.index.max().date(),
        len(X_test),  X_test.index.min().date(),  X_test.index.max().date(),
    )
    logger.info("Train regime dist: %s", y_train.value_counts().sort_index().to_dict())
    logger.info("Test  regime dist: %s", y_test.value_counts().sort_index().to_dict())

    cols = list(feature_cols or CLASSIFIER_FEATURE_COLS)
    clf = RegimeClassifier(params=params)
    clf.fit(X_train[cols], y_train, **(fit_kwargs or SHIPPED_FIT_KWARGS))
    clf.train_end_date_ = train_end
    clf.metadata_.update({
        "r2_gate":              X.attrs.get("r2_gate"),
        "pdv_iv_spread_source": X.attrs.get("pdv_iv_spread_source"),
        "feature_cols":         cols,
        "train_window":         _window(X_train.index),
    })
    return clf, X_test, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────

def _persistence(y: pd.Series, labels: Optional[pd.Series] = None) -> Tuple[float, int]:
    """Accuracy of ŷ(t) = label of the previous labelled session, on y's rows."""
    if labels is not None:
        prev = labels.dropna().astype(int).sort_index().shift(1).reindex(y.index)
    else:
        prev = y.shift(1)
    m = prev.notna()
    if not m.any():
        return float("nan"), 0
    return float((y[m].astype(int) == prev[m].astype(int)).mean()), int(m.sum())


def evaluate_classifier(
    clf: RegimeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: Optional[pd.Series] = None,
) -> dict:
    """
    Compute evaluation metrics on the test set.

    Parameters
    ----------
    labels : optional full rule-label series the test labels come from. When
             given, the persistence predictor for day t is the label of the
             previous labelled SESSION (even if that session's feature row was
             dropped); otherwise the previous row of y_test.

    Returns
    -------
    dict with:
      accuracy           : overall accuracy
      persistence_baseline_accuracy : accuracy of the no-skill rule
                           y_hat(t) = y(t-1), on the same rows. The label is
                           built from same-day observables, so yesterday's
                           label is known at close of t-1 — a legitimate
                           causal predictor and the floor any model must beat.
      persistence_n      : rows the persistence accuracy is measured on
      beats_persistence  : bool — accuracy > persistence baseline
      per_class          : dict of {class: {precision, recall, f1}}
      confusion_matrix   : np.ndarray (3×3)
      regime2_recall     : recall for Regime 2 specifically
      regime_dist_test   : {0: count, 1: count, 2: count}
      validation_dates   : {date: predicted_regime} for key stress dates
      n_test, test_window, feature_cols, r2_gate, pdv_iv_spread_source :
                           provenance — which rows, features and label gate
                           the numbers describe.
    """
    y_pred = clf.predict(X_test)

    acc   = accuracy_score(y_test.values, y_pred)

    # No-skill persistence baseline on the same rows, same labels.
    persistence_acc, persistence_n = _persistence(y_test, labels)
    cm    = confusion_matrix(y_test.values, y_pred, labels=[0, 1, 2])
    report_dict = classification_report(
        y_test.values, y_pred, labels=[0, 1, 2],
        target_names=[REGIME_NAMES[i] for i in range(3)],
        output_dict=True,
        zero_division=0,
    )

    # Regime 2 recall: critical for stress detection
    regime2_recall = report_dict.get("VOMMA_ACTIVE", {}).get("recall", np.nan)

    # Validation dates check
    val_preds = {}
    for date_str in REGIME2_VALIDATION_DATES:
        ts = pd.Timestamp(date_str)
        if ts in X_test.index:
            val_preds[date_str] = int(clf.predict(X_test.loc[[ts]])[0])
        else:
            val_preds[date_str] = None

    meta = getattr(clf, "metadata_", {}) or {}
    return {
        "accuracy":          acc,
        "persistence_baseline_accuracy": persistence_acc,
        "persistence_n":     persistence_n,
        "beats_persistence": bool(pd.notna(persistence_acc) and acc > persistence_acc),
        "per_class":         {
            i: {
                "precision": report_dict.get(REGIME_NAMES[i], {}).get("precision", np.nan),
                "recall":    report_dict.get(REGIME_NAMES[i], {}).get("recall", np.nan),
                "f1":        report_dict.get(REGIME_NAMES[i], {}).get("f1-score", np.nan),
            }
            for i in range(3)
        },
        "confusion_matrix":  cm,
        "regime2_recall":    regime2_recall,
        "regime_dist_test":  y_test.value_counts().sort_index().to_dict(),
        "validation_dates":  val_preds,
        "n_test":            int(len(y_test)),
        "test_window":       _window(y_test.index),
        "feature_cols":      list(clf.feature_names_ or []),
        "r2_gate":           meta.get("r2_gate", X_test.attrs.get("r2_gate")),
        "pdv_iv_spread_source": meta.get("pdv_iv_spread_source",
                                         X_test.attrs.get("pdv_iv_spread_source")),
    }


def classifier_accuracy_report(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_vol_series: "Optional[pd.Series]" = None,
    vvix_threshold: "Optional[float]" = R2_GATE,
    train_end: str = TRAIN_END_DATE,
    test_start: str = TEST_START_DATE,
    test_end: str = TEST_END_DATE,
    wf_years: Optional[List[int]] = None,
    params: Optional[dict] = None,
    fit_kwargs: Optional[dict] = None,
    include_benchmarks: bool = True,
) -> dict:
    """The classifier accuracy numbers, computed — never hard-coded.

    No look-ahead anywhere: features are lagged one session; pdv_iv_spread is
    the injected walk-forward σ̂ (or the rv proxy when none is given — never
    the full-sample pdv_model.pkl); labels use `vvix_threshold` (canonical
    gate by default). Every accuracy is paired with the persistence baseline
    on the SAME rows and the SAME labels.

    Returns a dict:
      r2_gate, pdv_iv_spread_source, feature_cols, fit_kwargs
      single_split : train ≤ train_end → test [test_start, test_end]:
                     n_train, n_test, train_window, test_window,
                     in_sample_acc (on the training window rows),
                     oos_acc, persistence_acc
      walk_forward : per calendar year Y in wf_years: train on all rows
                     before Y, test on Y. by_year {Y: {n, acc,
                     persistence_acc}}, pooled acc / persistence_acc / n
      benchmarks   : single split with vvix(t−1) added, and with vvix(t−1)
                     plus the lagged label y(t−1) added (both legitimate,
                     causal predictors) — in_sample_acc / oos_acc
    """
    fit_kwargs = dict(fit_kwargs or SHIPPED_FIT_KWARGS)
    X, y = build_dataset(spx_df, vix_wide_df, None, vvix_threshold,
                         pdv_vol_series=pdv_vol_series)
    labels = build_regime_labels(spx_df, vix_wide_df, vvix_threshold)
    cols = list(CLASSIFIER_FEATURE_COLS)
    in_test = (X.index >= pd.Timestamp(test_start)) & (X.index <= pd.Timestamp(test_end))
    in_train = X.index <= pd.Timestamp(train_end)

    def _split(feature_frame: pd.DataFrame) -> dict:
        Xtr, ytr = feature_frame[in_train], y[in_train]
        Xte, yte = feature_frame[in_test], y[in_test]
        clf = RegimeClassifier(params=params).fit(Xtr, ytr, **fit_kwargs)
        return {
            "in_sample_acc": float((clf.predict(Xtr) == ytr.values).mean()),
            "oos_acc":       float((clf.predict(Xte) == yte.values).mean()),
        }

    single = _split(X[cols])
    pers, pers_n = _persistence(y[in_test], labels)
    single.update({
        "n_train":         int(in_train.sum()),
        "n_test":          int(in_test.sum()),
        "train_window":    _window(X.index[in_train]),
        "test_window":     _window(X.index[in_test]),
        "persistence_acc": pers,
        "persistence_n":   pers_n,
    })

    if wf_years is None:
        wf_years = list(range(pd.Timestamp(test_start).year, pd.Timestamp(test_end).year + 1))
    by_year = {}
    hits = hits_p = n_tot = n_p = 0
    for year in wf_years:
        cut = pd.Timestamp(f"{year}-01-01")
        yr = (X.index >= cut) & (X.index <= min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(test_end)))
        tr = X.index < cut
        if yr.sum() == 0 or tr.sum() < 500:
            continue
        clf = RegimeClassifier(params=params).fit(X.loc[tr, cols], y[tr], **fit_kwargs)
        correct = clf.predict(X.loc[yr, cols]) == y[yr].values
        p_acc, p_n = _persistence(y[yr], labels)
        by_year[int(year)] = {"n": int(yr.sum()), "acc": float(correct.mean()),
                              "persistence_acc": p_acc}
        hits += int(correct.sum()); n_tot += int(yr.sum())
        hits_p += int(round(p_acc * p_n)); n_p += p_n
    walk_forward = {
        "by_year":         by_year,
        "n":               n_tot,
        "acc":             hits / n_tot if n_tot else float("nan"),
        "persistence_acc": hits_p / n_p if n_p else float("nan"),
    }

    benchmarks = {}
    if include_benchmarks:
        benchmarks["with_vvix"] = _split(X[cols + ["vvix"]])
        X_lag = X[cols + ["vvix"]].copy()
        X_lag["regime_lag1"] = labels.sort_index().shift(1).reindex(X.index)
        X_lag = X_lag.fillna({"regime_lag1": -1})
        benchmarks["with_vvix_and_lagged_label"] = _split(X_lag)

    return {
        "r2_gate":              X.attrs.get("r2_gate"),
        "pdv_iv_spread_source": X.attrs.get("pdv_iv_spread_source"),
        "feature_cols":         cols,
        "fit_kwargs":           fit_kwargs,
        "single_split":         single,
        "walk_forward":         walk_forward,
        "benchmarks":           benchmarks,
    }


def regime_distribution_by_year(
    clf: RegimeClassifier,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fraction of each regime per calendar year.

    Returns
    -------
    pd.DataFrame — rows=year, columns=Regime {0,1,2}, values=fraction
    """
    preds = clf.predict_series(X)
    df = preds.to_frame("regime")
    df["year"] = df.index.year
    dist = df.groupby("year")["regime"].value_counts(normalize=True).unstack(fill_value=0)
    dist.columns = [REGIME_NAMES[c] for c in dist.columns]
    return dist


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_regime2_dates(
    clf: RegimeClassifier,
    X: pd.DataFrame,
    dates: Optional[List[str]] = None,
) -> Dict[str, bool]:
    """
    For each date in `dates`, check that the classifier predicts Regime 2.

    Returns dict {date_str: bool_correct}.
    """
    if dates is None:
        dates = REGIME2_VALIDATION_DATES

    results = {}
    for date_str in dates:
        ts = pd.Timestamp(date_str)
        if ts not in X.index:
            results[date_str] = None   # date not in test window
            continue
        pred = int(clf.predict(X.loc[[ts]])[0])
        results[date_str] = (pred == 2)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """
    Plot labelled confusion matrix heatmap.

    Parameters
    ----------
    cm        : 3×3 confusion matrix from evaluate_classifier()
    save_path : if provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"R{i}\n{REGIME_NAMES[i]}" for i in range(3)],
    )
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")

    ax.set_title(
        "Regime Classifier — Confusion Matrix\n"
        f"Test set: {TEST_START_DATE} → 2025-12-31",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("Predicted Regime", fontsize=10)
    ax.set_ylabel("True Regime", fontsize=10)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_regime_distribution(
    dist_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Stacked-bar chart showing regime fraction per year.

    Parameters
    ----------
    dist_df   : output of regime_distribution_by_year()
    """
    colours = {"LONG_GAMMA": "#2196F3", "SHORT_GAMMA": "#4CAF50", "VOMMA_ACTIVE": "#F44336"}

    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(dist_df))
    for col in ["LONG_GAMMA", "SHORT_GAMMA", "VOMMA_ACTIVE"]:
        if col in dist_df.columns:
            vals = dist_df[col].values
            ax.bar(dist_df.index, vals, bottom=bottom, label=col,
                   color=colours.get(col, "grey"), alpha=0.85, edgecolor="white")
            bottom += vals

    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of trading days")
    ax.set_title("Predicted Regime Distribution by Year", fontsize=12, fontweight="bold")
    ax.set_xticks(dist_df.index)
    ax.set_xticklabels([str(y) for y in dist_df.index], rotation=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=9)

    # Mark the train/test boundary
    test_start_year = int(TEST_START_DATE[:4])
    ax.axvline(test_start_year - 0.5, color="black", linewidth=1.5, linestyle="--")
    ax.text(
        test_start_year - 0.4, 0.97,
        "← Train | Test →",
        transform=ax.get_xaxis_transform(),
        fontsize=8, va="top",
    )

    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Regime distribution chart saved to %s", save_path)

    return fig


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_regime_labels(
    clf: RegimeClassifier,
    X: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    path: Optional[Path] = None,
) -> Path:
    """
    Save the research classifier's PREDICTIONS (not rule labels) to parquet.

    Columns: pred_regime, pred_regime_name, confidence, prob_0..prob_2,
    true_regime (the rule label, if y_true is given) and the features.

    The default path is CLASSIFIER_PREDICTIONS_PARQUET. It used to be
    LABELS_PARQUET, so running RegimePipeline overwrote the shipped RULE
    labels (which the dashboard headlines as "the label the backtest trades
    on") with the demoted classifier's predictions. Writing predictions to
    LABELS_PARQUET is refused.

    Parameters
    ----------
    clf    : fitted RegimeClassifier
    X      : full feature matrix (all dates)
    y_true : optional ground-truth labels (for comparison)
    """
    if path is None:
        path = CLASSIFIER_PREDICTIONS_PARQUET
    path = Path(path)
    if path.resolve() == Path(LABELS_PARQUET).resolve():
        raise ValueError(
            f"refusing to write classifier predictions over the rule-label file {path}"
        )

    preds  = clf.predict(X)
    probas = clf.predict_proba(X)          # (N, 3)

    df = pd.DataFrame({
        "pred_regime":      preds,
        "pred_regime_name": [REGIME_NAMES[p] for p in preds],
        "confidence":       probas.max(axis=1),
        "prob_0":           probas[:, 0],
        "prob_1":           probas[:, 1],
        "prob_2":           probas[:, 2],
    }, index=X.index)

    if y_true is not None:
        df["true_regime"] = y_true.reindex(X.index)

    for col in FEATURE_COLS:
        if col in X.columns:
            df[col] = X[col].values

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Classifier predictions saved to %s (%d rows)", path, len(df))
    return path


def load_regime_labels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the RULE regime labels parquet (or any labels/predictions parquet)."""
    if path is None:
        path = LABELS_PARQUET
    return pd.read_parquet(path)


# ── RegimePipeline orchestration ──────────────────────────────────────────────

class RegimePipeline:
    """
    End-to-end research-classifier pipeline.

    Usage
    -----
      pipe = RegimePipeline()
      pipe.run()          # load data → build features → train → evaluate → save
      pipe.plot()         # confusion matrix + distribution chart
      results = pipe.results_

    Writes the classifier (MODEL_PATH), its predictions
    (CLASSIFIER_PREDICTIONS_PARQUET) and the RULE labels to the regime_labels
    DB table. It never touches the rule-label parquet.

    pdv_vol_series injects a walk-forward σ̂ for pdv_iv_spread. pdv_model_path
    (a full-sample PDV pickle) is accepted for legacy use only — its
    coefficients have seen the test period, so evaluation numbers built on it
    are look-ahead contaminated; the default is neither (rv_20d proxy).
    """

    def __init__(
        self,
        train_end:      str   = TRAIN_END_DATE,
        vvix_threshold: Optional[float] = R2_GATE,
        pdv_model_path: Optional[Path] = None,
        xgb_params:     Optional[dict] = None,
        pdv_vol_series: Optional[pd.Series] = None,
    ):
        self.train_end      = train_end
        self.vvix_threshold = vvix_threshold
        self.pdv_model_path = Path(pdv_model_path) if pdv_model_path is not None else None
        self.xgb_params     = xgb_params
        self.pdv_vol_series = pdv_vol_series

        self.clf_:    Optional[RegimeClassifier] = None
        self.X_:      Optional[pd.DataFrame]     = None
        self.y_:      Optional[pd.Series]        = None
        self.X_test_: Optional[pd.DataFrame]     = None
        self.y_test_: Optional[pd.Series]        = None
        self.results_: Optional[dict]            = None

    def run(self) -> dict:
        """Execute full pipeline: data → train → evaluate → save."""
        logger.info("RegimePipeline.run() starting ...")

        # Load data
        spx = get_spx_ohlcv(as_of_date=TEST_END_DATE, start_date="2010-01-01")
        vix = get_vix_term_structure_wide(as_of_date=TEST_END_DATE, start_date="2010-01-01")

        pdv_model = None
        if self.pdv_vol_series is None and self.pdv_model_path is not None:
            logger.warning("Using full-sample PDV model %s: evaluation is look-ahead "
                           "contaminated", self.pdv_model_path)
            with open(self.pdv_model_path, "rb") as fh:
                pdv_model = pickle.load(fh)

        # Train
        self.clf_, self.X_test_, self.y_test_ = train_classifier(
            spx, vix, pdv_model,
            train_end=self.train_end,
            params=self.xgb_params,
            vvix_threshold=self.vvix_threshold,
            pdv_vol_series=self.pdv_vol_series,
        )

        # Build full aligned dataset for labelling all dates
        self.X_, self.y_ = build_dataset(spx, vix, pdv_model, self.vvix_threshold,
                                         pdv_vol_series=self.pdv_vol_series)
        labels = build_regime_labels(spx, vix, self.vvix_threshold)

        # Evaluate (persistence on the same labels and rows)
        self.results_ = evaluate_classifier(self.clf_, self.X_test_, self.y_test_, labels=labels)
        dist = regime_distribution_by_year(self.clf_, self.X_)
        self.results_["regime_dist_by_year"] = dist
        self.results_["label_summary_test"] = regime_label_summary(
            labels, start=TEST_START_DATE, end=TEST_END_DATE)
        self.results_["forward_premium_by_regime"] = regime_forward_premium_table(
            spx, vix, labels)

        # Validate stress dates
        val = validate_regime2_dates(self.clf_, self.X_)
        self.results_["validation"] = val

        logger.info("Test accuracy: %.4f (persistence %.4f)", self.results_["accuracy"],
                    self.results_["persistence_baseline_accuracy"])
        logger.info("Regime 2 recall: %.4f", self.results_["regime2_recall"])
        logger.info("Validation dates: %s", val)

        # Save model and its predictions (never over the rule labels)
        self.clf_.metadata_["evaluation"] = {
            "oos_acc":         float(self.results_["accuracy"]),
            "persistence_acc": float(self.results_["persistence_baseline_accuracy"]),
            "n_test":          int(self.results_["n_test"]),
            "test_window":     self.results_["test_window"],
        }
        self.clf_.save(MODEL_PATH)
        save_regime_labels(self.clf_, self.X_, self.y_)

        # Write the RULE labels (same-day observables) to SQLite
        _write_labels_to_db(build_rule_label_frame(spx, vix, self.vvix_threshold))

        return self.results_

    def plot(self) -> Tuple[plt.Figure, plt.Figure]:
        """Generate confusion matrix and distribution charts."""
        self._require_run()
        fig_cm   = plot_confusion_matrix(
            self.results_["confusion_matrix"], save_path=CONFUSION_PLOT
        )
        fig_dist = plot_regime_distribution(
            self.results_["regime_dist_by_year"], save_path=DISTRIBUTION_PLOT
        )
        return fig_cm, fig_dist

    def _require_run(self):
        if self.clf_ is None:
            raise RuntimeError("Call .run() before accessing results.")

    def __repr__(self) -> str:
        status = "complete" if self.results_ is not None else "not run"
        return (
            f"RegimePipeline(train_end={self.train_end}, "
            f"r2_gate={gate_name(self.vvix_threshold)}, status={status})"
        )


# ── DB write helper ────────────────────────────────────────────────────────────

def regime_label_db_rows(label_frame: pd.DataFrame) -> pd.DataFrame:
    """Rows for the regime_labels DB table from build_rule_label_frame output.

    Each row holds the RULE label and the SAME-DAY observables that define it:
    vix_level = VIX(t) points, vix_slope = VIX3M(t) − VIX(t) points, rv_20d =
    the label's own trailing realised vol (decimal). confidence is 1.0: the
    rule is deterministic, it is not a model probability.

    (Pre-C22 the table held classifier predictions labelled 'regime', t−1
    features under date t, and rv_20d = vix − spread = 2·VIX − RV.)
    """
    return pd.DataFrame({
        "date":       label_frame.index.strftime("%Y-%m-%d"),
        "regime":     label_frame["regime"].astype(int).values,
        "confidence": 1.0,
        "vix_level":  label_frame["vix"].values * 100.0,
        "vix_slope":  label_frame["ts_slope"].values * 100.0,
        "rv_20d":     label_frame["rv_20d"].values,
    })


def _write_labels_to_db(label_frame: pd.DataFrame) -> int:
    """Write rule labels (see regime_label_db_rows) to the regime_labels table."""
    rows = regime_label_db_rows(label_frame)
    try:
        n = insert_regime_labels(rows)
        logger.info("Wrote %d regime label rows to SQLite", n)
        return n
    except Exception as exc:
        logger.warning("DB write failed: %s", exc)
        return 0


# ── HMM Regime Classifier (C5 variant) ────────────────────────────────────────

class HMMRegimeClassifier:
    """
    3-state Gaussian HMM regime classifier using hmmlearn (RESEARCH ONLY).

    Learns latent market states from four observable features:
        (vix, vvix, ts_slope, fear_premium)

    Causal inference only
    ---------------------
    Every prediction is a FORWARD-FILTERED posterior P(state_t | x_1..x_t):
    the probability (and label) for day t never depends on rows after t, so
    hmm_predict_proba(X[:k]).iloc[-1] == hmm_predict_proba(X).iloc[k-1].
    (Pre-C22, predict() ran Viterbi over the whole input and
    predict_proba_raw() used forward–backward smoothing; both condition state
    t on later observations — e.g. P(R2) on 2020-02-21 was 0.08 filtered but
    0.53 when the rest of 2020 was in the input.) The filter starts from the
    fitted start distribution at the first row of X.

    Fitting: EM from n_init starts (seeds random_state, random_state+1, …),
    keeping the highest log-likelihood. A single start (pre-C22) often
    converged to a local optimum with two states on the same means (2010-2019
    features, seed 42: two states at VIX 0.141/0.142, log-lik 14,129 vs
    15,271 with 5 starts), leaving only two distinct "regimes". Even with
    distinct states the filtered labels match the rule labels on only ~28%
    of 2020-2025 sessions: the HMM finds its own clusters.

    State → regime mapping (follows the RULE semantics)
    ---------------------------------------------------
      R2 (VOMMA_ACTIVE) : the state with the highest mean vvix
      R0 (LONG_GAMMA)   : of the other two, the one with the lower mean
                          fear_premium (VIX/rv_20d < 1 ⇔ rv_20d > VIX)
      R1 (SHORT_GAMMA)  : the remaining state
    (Pre-C22 states were ranked by mean VIX, lowest → R0; but under the rule
    R1 has the LOWEST mean VIX, so R0 and R1 came out swapped.) HMM states are
    still latent clusters: the mapping is a naming convention, not a claim
    that they reproduce the rule labels.

    Usage
    -----
      hmm = HMMRegimeClassifier()
      hmm.fit(X_train)                       # X must contain the 4 HMM features
      proba_df = hmm.hmm_predict_proba(X)    # DataFrame (prob_r0, prob_r1, prob_r2)
      labels   = hmm.predict(X)              # int array aligned to regime labelling
    """

    HMM_FEATURES = ["vix", "vvix", "ts_slope", "fear_premium"]

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 200,
        random_state: int = RANDOM_SEED,
        n_init: int = 5,
    ):
        if n_components != 3:
            raise ValueError("HMMRegimeClassifier maps states onto R0/R1/R2 and "
                             "needs exactly 3 components")
        self.n_components  = n_components
        self.n_iter        = n_iter
        self.random_state  = random_state
        self.n_init        = max(int(n_init), 1)
        self.log_likelihood_: Optional[float] = None
        self._model        = None
        self._state_to_regime: Optional[Dict[int, int]] = None
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "HMMRegimeClassifier":
        """
        Fit Gaussian HMM on features X (rows with any NaN feature are dropped).

        After fitting, states are mapped to R0/R1/R2 by the rule semantics
        described in the class docstring.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as exc:
            raise ImportError("hmmlearn is required: pip install hmmlearn") from exc

        Xm = X[self.HMM_FEATURES].dropna().values.astype(float)

        best, best_ll = None, -np.inf
        for k in range(self.n_init):
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state + k,
            )
            model.fit(Xm)
            ll = float(model.score(Xm))
            if ll > best_ll:
                best, best_ll = model, ll
        model = best
        self._model = model
        self.log_likelihood_ = best_ll

        col = {f: i for i, f in enumerate(self.HMM_FEATURES)}
        means = np.asarray(model.means_, dtype=float)
        self._state_to_regime = self.map_states_to_regimes(means)

        self.is_fitted = True
        logger.info(
            "HMMRegimeClassifier fitted on %d rows. State→Regime mapping: %s | "
            "mean vvix per state: %s | mean fear_premium per state: %s",
            len(Xm),
            self._state_to_regime,
            {s: round(float(means[s, col["vvix"]]), 4) for s in range(self.n_components)},
            {s: round(float(means[s, col["fear_premium"]]), 4) for s in range(self.n_components)},
        )
        return self

    @classmethod
    def map_states_to_regimes(cls, means: np.ndarray) -> Dict[int, int]:
        """{hmm_state: regime} from per-state feature means (rows = states,
        columns = HMM_FEATURES), following the rule semantics: highest mean
        vvix → R2; of the rest, lower mean fear_premium → R0; other → R1."""
        col = {f: i for i, f in enumerate(cls.HMM_FEATURES)}
        means = np.asarray(means, dtype=float)
        r2_state = int(np.argmax(means[:, col["vvix"]]))
        rest = [s for s in range(len(means)) if s != r2_state]
        r0_state = min(rest, key=lambda s: means[s, col["fear_premium"]])
        r1_state = next(s for s in rest if s != r0_state)
        return {r0_state: 0, r1_state: 1, r2_state: 2}

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("HMMRegimeClassifier not fitted. Call .fit() first.")

    def _log_emission(self, Xm: np.ndarray) -> np.ndarray:
        """log N(x_t; μ_s, Σ_s) for every row and state, shape (N, n_components)."""
        from scipy.stats import multivariate_normal

        covs = np.asarray(self._model.covars_, dtype=float)   # full (n, d, d)
        out = np.full((len(Xm), self.n_components), np.nan)
        ok = np.isfinite(Xm).all(axis=1)
        if ok.any():
            for s in range(self.n_components):
                out[ok, s] = multivariate_normal(
                    mean=self._model.means_[s], cov=covs[s], allow_singular=True
                ).logpdf(Xm[ok])
        return out

    def _forward_filter(self, X: pd.DataFrame) -> np.ndarray:
        """Filtered posteriors P(state_t | x_1..x_t) in raw HMM-state order.

        Missing features are forward-filled (causal); a row that is still
        incomplete gets the one-step prediction with no measurement update.
        """
        Xm = X[self.HMM_FEATURES].ffill().values.astype(float)
        log_b = self._log_emission(Xm)
        trans = np.asarray(self._model.transmat_, dtype=float)
        prior = np.asarray(self._model.startprob_, dtype=float)
        alpha = np.empty((len(Xm), self.n_components))
        for t in range(len(Xm)):
            if t > 0:
                prior = alpha[t - 1] @ trans
            if np.isfinite(log_b[t]).all():
                w = np.log(np.clip(prior, 1e-300, None)) + log_b[t]
                w = np.exp(w - w.max())
                alpha[t] = w / w.sum()
            else:
                alpha[t] = prior / prior.sum()
        return alpha

    def _predict_raw_states(self, X: pd.DataFrame) -> np.ndarray:
        """Most likely raw HMM state per row under the FILTERED posterior."""
        self._require_fitted()
        return self._forward_filter(X).argmax(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return regime labels (0/1/2), shape (N,), from filtered posteriors."""
        self._require_fitted()
        raw = self._predict_raw_states(X)
        return np.array([self._state_to_regime[s] for s in raw])

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return FILTERED state probabilities, shape (N, n_components)."""
        self._require_fitted()
        return self._forward_filter(X)

    def hmm_predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return regime probabilities as DataFrame indexed like X.

        The forward-filtered posteriors (indexed by HMM state) are reordered
        so that column 0 = P(R0), column 1 = P(R1), column 2 = P(R2). Row t
        uses only rows ≤ t of X.

        Returns
        -------
        pd.DataFrame with columns ['prob_r0', 'prob_r1', 'prob_r2'], indexed like X.
        """
        self._require_fitted()
        raw_proba = self.predict_proba_raw(X)  # shape (N, n_components)

        # Reorder columns: col i in output = P(regime i)
        # _state_to_regime maps {hmm_state: regime_label}
        # We need the inverse: {regime_label: hmm_state}
        regime_to_state = {v: k for k, v in self._state_to_regime.items()}

        ordered = np.column_stack([
            raw_proba[:, regime_to_state[r]]
            for r in range(self.n_components)
        ])

        return pd.DataFrame(
            ordered, index=X.index, columns=["prob_r0", "prob_r1", "prob_r2"]
        )

    def compare_with_xgboost(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        xgb_clf: "RegimeClassifier",
    ) -> dict:
        """
        Compare HMM predictions with XGBoost predictions on the same data.

        Both are causal (filtered HMM; lagged-feature XGBoost); pass X/y_true
        from build_dataset so the rule labels line up with lagged features.

        Returns
        -------
        dict with accuracy, per-class metrics, and average P(R2) comparison.
        """
        self._require_fitted()
        xgb_clf._require_fitted()

        hmm_preds = self.predict(X)
        xgb_preds = xgb_clf.predict(X)
        y          = y_true.values

        hmm_acc = float(accuracy_score(y, hmm_preds))
        xgb_acc = float(accuracy_score(y, xgb_preds))

        # Average P(R2) on R2 days
        r2_mask = y == 2
        hmm_p_r2 = self.hmm_predict_proba(X)["prob_r2"]
        xgb_p_r2 = pd.DataFrame(
            xgb_clf.predict_proba(X), index=X.index, columns=["prob_r0", "prob_r1", "prob_r2"]
        )["prob_r2"]

        return {
            "hmm_accuracy":        hmm_acc,
            "xgb_accuracy":        xgb_acc,
            "hmm_avg_p_r2_on_r2":  float(hmm_p_r2[r2_mask].mean()) if r2_mask.any() else float("nan"),
            "xgb_avg_p_r2_on_r2":  float(xgb_p_r2[r2_mask].mean()) if r2_mask.any() else float("nan"),
            "n_r2_days":           int(r2_mask.sum()),
            "n_total":             int(len(y)),
        }

    def save(self, path: Path) -> Path:
        """Pickle the HMM classifier."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("HMMRegimeClassifier saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "HMMRegimeClassifier":
        """Load a pickled HMMRegimeClassifier."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected HMMRegimeClassifier, got {type(obj)}")
        return obj

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"HMMRegimeClassifier(n_components={self.n_components}, "
            f"n_iter={self.n_iter}, status={status})"
        )
