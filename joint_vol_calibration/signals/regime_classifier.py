"""
regime_classifier.py — C8: Volatility Regime Classifier

Classifies each trading day into one of three volatility regimes using
an XGBoost multi-class model trained on 2010-2019 and tested on 2020-2025.

Three regimes
-------------
  REGIME 0  LONG_GAMMA    : Realized vol > Implied vol
                             rv_20d(t) > VIX(t)/100  AND  VVIX(t) ≤ 100
                             → Long gamma trades are profitable.
                               Sell variance-swaps / sell straddles loses money.

  REGIME 1  SHORT_GAMMA   : Implied vol > Realized vol
                             rv_20d(t) ≤ VIX(t)/100  AND  VVIX(t) ≤ 100
                             → Short gamma earns the variance risk premium.
                               Standard systematic vol-selling works.

  REGIME 2  VOMMA_ACTIVE  : Elevated vol-of-vol
                             VVIX(t) > 100   (above 75th percentile of VVIX history)
                             → Vol-of-vol is elevated; standard delta/vega hedges
                               are dangerous. Vomma (∂Vega/∂σ) trades are active.
                               This regime overrides 0/1 — it is independent of
                               the IV vs RV comparison.

  Note on VVIX threshold
  ----------------------
  The CBOE VVIX (Volatility-of-VIX) ranges from 62 to 207 in our sample.
  A threshold of 100 captures the 75th percentile (elevated) and correctly
  identifies COVID March 2020 (VVIX=207), 2025-04-09 tariff spike (VVIX=142),
  and 2022 selloff (VVIX~110-130) while excluding routine market fluctuations.

Features (computed at close of t, shifted 1 day for zero look-ahead)
----------------------------------------------------------------------
  vix             : ^VIX / 100                  (implied vol level, decimal)
  ts_slope        : (^VIX3M − ^VIX) / 100       (term structure slope; < 0 = backwardation)
  fear_premium    : VIX / rv_20d                 (vol risk premium; > 1 = overpaying for hedges)
  rv_change_5d    : rv_5d(t) − rv_5d(t−5)       (realized vol momentum)
  pdv_iv_spread   : σ_PDV(t) − VIX(t)/100       (path-dep model over/under market IV)
  vvix            : ^VVIX / 100                  (vol-of-vol level, decimal)

Model
-----
  XGBoost multi-class (objective='multi:softprob', num_class=3)
  Train:  2010-01-04 → 2019-12-31  (~2,300 samples after NaN drop)
  Test:   2020-01-01 → 2025-12-31  (~1,500 samples)

  Class imbalance handled via sample_weight (sklearn 'balanced')

Zero look-ahead
---------------
  Feature X(t) uses data available at close of day t.
  Label y(t) is the regime defined by same-day observables (no future data).
  Training uses X_shift = X.shift(1) so that X(t-1) → y(t).
  At inference: use X(t) to predict y(t+1).

Key validations
---------------
  • Regime 2 must fire on 2020-03-16  (VVIX=207, COVID crash)
  • Regime 2 must fire on 2025-04-09  (VVIX=142, tariff spike)
  • Confusion matrix, per-class precision/recall
  • Regime distribution by year

Outputs
-------
  data_store/regime_classifier.pkl
  data_store/regime_labels.parquet
  data_store/regime_confusion_matrix.png
  data_store/regime_distribution.png
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
from joint_vol_calibration.models.pdv import extract_pdv_features

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REGIME_NAMES: Dict[int, str] = {
    0: "LONG_GAMMA",
    1: "SHORT_GAMMA",
    2: "VOMMA_ACTIVE",
}

VVIX_REGIME2_THRESHOLD: float = 100.0   # VVIX > 100 → Regime 2
TRAIN_END_DATE:  str = "2019-12-31"
TEST_START_DATE: str = "2020-01-01"

FEATURE_COLS: List[str] = [
    "vix",
    "ts_slope",
    "fear_premium",
    "rv_change_5d",
    "pdv_iv_spread",
    "vvix",
]

# Stress dates that must be classified as Regime 2
REGIME2_VALIDATION_DATES: List[str] = [
    "2020-03-16",   # COVID crash: VIX=82.7, VVIX=207.6
    "2025-04-09",   # Tariff spike: VIX=33.6, VVIX=142.5
]

# ── Paths ─────────────────────────────────────────────────────────────────────

SIGNALS_DIR        = DATA_DIR / "signals"
MODEL_PATH         = SIGNALS_DIR / "regime_classifier.pkl"
LABELS_PARQUET     = SIGNALS_DIR / "regime_labels.parquet"
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


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
) -> pd.DataFrame:
    """
    Build the six-column feature matrix for the regime classifier.

    Parameters
    ----------
    spx_df      : SPX OHLCV DataFrame with columns ['date', 'close', 'log_return']
    vix_wide_df : Wide VIX TS DataFrame with columns ['^VIX', '^VIX3M', '^VVIX', ...]
    pdv_model   : Optional PDVModel instance (for pdv_iv_spread feature).
                  If None, falls back to rv_20d - vix as spread proxy.

    Returns
    -------
    pd.DataFrame indexed by date with columns = FEATURE_COLS.
    NaN rows present for early dates (EWMA warmup, rolling windows).
    """
    spx = spx_df.copy()
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.set_index("date").sort_index()

    vix = vix_wide_df.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date").sort_index()

    # Inner join on date (SPX trading days drive the index)
    df = spx[["close", "log_return"]].join(
        vix[[c for c in ["^VIX", "^VIX3M", "^VIX6M", "^VVIX"] if c in vix.columns]],
        how="inner",
    )

    log_ret = df["log_return"]

    # ── Realized vol rolling windows ─────────────────────────────────────────
    rv_20d = np.sqrt((log_ret**2).rolling(20).mean() * 252)
    rv_5d  = np.sqrt((log_ret**2).rolling(5).mean()  * 252)

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

    # ── Feature 5: PDV vs ATM IV spread ──────────────────────────────────────
    if pdv_model is not None:
        try:
            log_ret_s = log_ret.dropna()
            pdv_feats  = extract_pdv_features(log_ret_s)
            pdv_vol    = pdv_model.linear_.predict(pdv_feats[["sigma1", "sigma2", "lev"]])
            pdv_vol    = pdv_vol.reindex(df.index)
            df["pdv_iv_spread"] = pdv_vol - df["^VIX"] / 100.0
        except Exception as exc:
            logger.warning("PDV spread feature failed (%s); using rv_20d fallback", exc)
            df["pdv_iv_spread"] = rv_20d - df["^VIX"] / 100.0
    else:
        # Fallback: (trailing rv_20d − VIX) is a reasonable realized-vs-implied proxy
        df["pdv_iv_spread"] = rv_20d - df["^VIX"] / 100.0

    # ── Feature 6: VVIX level ─────────────────────────────────────────────────
    df["vvix"] = df["^VVIX"] / 100.0 if "^VVIX" in df.columns else 0.0

    return df[FEATURE_COLS].copy()


# ── Regime label generation ────────────────────────────────────────────────────

def build_regime_labels(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    vvix_threshold: float = VVIX_REGIME2_THRESHOLD,
) -> pd.Series:
    """
    Assign rule-based regime labels to each trading day.

    Rules (applied in priority order)
    ----------------------------------
    REGIME 2  : VVIX(t) > vvix_threshold   (overrides 0/1)
    REGIME 0  : rv_20d(t) > VIX(t)/100     (realized > implied)
    REGIME 1  : rv_20d(t) ≤ VIX(t)/100     (implied > realized)

    Labels are defined by same-day observables — no look-ahead bias.

    Returns
    -------
    pd.Series[int] indexed by date (Timestamp), values in {0, 1, 2}.
    """
    spx = spx_df.copy()
    spx["date"] = pd.to_datetime(spx["date"])
    spx = spx.set_index("date").sort_index()

    vix = vix_wide_df.copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date").sort_index()

    df = spx[["log_return"]].join(
        vix[[c for c in ["^VIX", "^VVIX"] if c in vix.columns]],
        how="inner",
    )

    log_ret = df["log_return"]
    rv_20d  = np.sqrt((log_ret**2).rolling(20).mean() * 252)

    vix_iv = df["^VIX"] / 100.0
    vvix   = df["^VVIX"] if "^VVIX" in df.columns else pd.Series(0.0, index=df.index)

    # Default: Regime 1 (short gamma / neutral)
    labels = pd.Series(1, index=df.index, dtype=int)

    # Regime 0: trailing realized vol exceeds implied vol
    labels[rv_20d > vix_iv] = 0

    # Regime 2: elevated vol-of-vol (overrides 0/1)
    labels[vvix > vvix_threshold] = 2

    # Drop days with insufficient history (rv_20d NaN = first 19 trading days)
    labels[rv_20d.isna()] = pd.NA

    return labels.dropna().astype(int)


# ── Regime classifier ─────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    XGBoost multi-class classifier for the three volatility regimes.

    Usage
    -----
      clf = RegimeClassifier()
      clf.fit(X_train, y_train)
      preds   = clf.predict(X_test)
      probas  = clf.predict_proba(X_test)   # shape (N, 3)
      clf.save(path)

      clf2 = RegimeClassifier.load(path)
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS.copy()
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: Optional[List[str]] = None
        self.train_end_date_: Optional[str] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_sample_weights: bool = True,
    ) -> "RegimeClassifier":
        """
        Fit XGBoost on (X, y).

        Parameters
        ----------
        X                  : feature DataFrame (rows × FEATURE_COLS)
        y                  : integer regime labels (0/1/2)
        use_sample_weights : if True, balance classes via sklearn 'balanced' weights
        """
        self.feature_names_ = list(X.columns)

        sample_weight = None
        if use_sample_weights:
            sample_weight = compute_sample_weight("balanced", y.values)

        self.model_ = xgb.XGBClassifier(**self.params)
        self.model_.fit(
            X.values,
            y.values,
            sample_weight=sample_weight,
            verbose=False,
        )
        logger.info(
            "RegimeClassifier fitted on %d samples, classes %s",
            len(y),
            np.unique(y.values).tolist(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted regime labels (int array)."""
        self._require_fitted()
        return self.model_.predict(X[self.feature_names_].values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape (N, 3)."""
        self._require_fitted()
        return self.model_.predict_proba(X[self.feature_names_].values)

    def predict_series(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as a Series indexed like X."""
        return pd.Series(self.predict(X), index=X.index, name="regime")

    def feature_importance(self) -> pd.Series:
        """Return XGBoost feature importances (weight/gain)."""
        self._require_fitted()
        # XGBoost 3.x: use feature_importances_ directly (sklearn API)
        imp_values = self.model_.feature_importances_
        return pd.Series(
            imp_values,
            index=self.feature_names_,
        ).sort_values(ascending=False)

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the classifier."""
        if path is None:
            path = MODEL_PATH
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
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
        return (
            f"RegimeClassifier(n_estimators={self.params.get('n_estimators', '?')}, "
            f"max_depth={self.params.get('max_depth', '?')}, "
            f"status={status})"
        )


# ── Training pipeline ─────────────────────────────────────────────────────────

def build_dataset(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
    vvix_threshold: float = VVIX_REGIME2_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) aligned dataset.

    Features are shifted by 1 day (X_{t-1} predicts y_t) to enforce
    zero look-ahead at inference time.

    Returns
    -------
    X : DataFrame — shifted feature matrix, NaN rows dropped
    y : Series    — regime labels aligned to X
    """
    feats  = build_features(spx_df, vix_wide_df, pdv_model)
    labels = build_regime_labels(spx_df, vix_wide_df, vvix_threshold)

    # Shift features forward 1 day: X(t-1) predicts y(t)
    X_shifted = feats.shift(1)

    # Align on common index, drop NaN
    combined = pd.concat([X_shifted, labels.rename("regime")], axis=1).dropna()
    X = combined[FEATURE_COLS]
    y = combined["regime"].astype(int)

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


def train_classifier(
    spx_df: pd.DataFrame,
    vix_wide_df: pd.DataFrame,
    pdv_model=None,
    train_end: str = TRAIN_END_DATE,
    params: Optional[dict] = None,
    vvix_threshold: float = VVIX_REGIME2_THRESHOLD,
) -> Tuple["RegimeClassifier", pd.DataFrame, pd.Series]:
    """
    Full training pipeline: build dataset → split → fit → return.

    Returns
    -------
    (clf, X_test, y_test)
    """
    logger.info("Building dataset ...")
    X, y = build_dataset(spx_df, vix_wide_df, pdv_model, vvix_threshold)
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, train_end)

    logger.info(
        "Train: %d samples (%s → %s) | Test: %d samples (%s → %s)",
        len(X_train), X_train.index.min().date(), X_train.index.max().date(),
        len(X_test),  X_test.index.min().date(),  X_test.index.max().date(),
    )
    logger.info("Train regime dist: %s", y_train.value_counts().sort_index().to_dict())
    logger.info("Test  regime dist: %s", y_test.value_counts().sort_index().to_dict())

    clf = RegimeClassifier(params=params)
    clf.fit(X_train, y_train)
    clf.train_end_date_ = train_end
    return clf, X_test, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_classifier(
    clf: RegimeClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute evaluation metrics on the test set.

    Returns
    -------
    dict with:
      accuracy           : overall accuracy
      per_class          : dict of {class: {precision, recall, f1}}
      confusion_matrix   : np.ndarray (3×3)
      regime2_recall     : recall for Regime 2 specifically
      regime_dist_test   : {0: count, 1: count, 2: count}
      validation_dates   : {date: predicted_regime} for key stress dates
    """
    y_pred = clf.predict(X_test)

    acc   = accuracy_score(y_test.values, y_pred)
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

    return {
        "accuracy":          acc,
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
    Save (date, predicted_regime, confidence, true_regime) to parquet.

    Parameters
    ----------
    clf    : fitted RegimeClassifier
    X      : full feature matrix (all dates)
    y_true : optional ground-truth labels (for comparison)
    """
    if path is None:
        path = LABELS_PARQUET

    preds  = clf.predict(X)
    probas = clf.predict_proba(X)          # (N, 3)

    df = pd.DataFrame({
        "regime":        preds,
        "regime_name":   [REGIME_NAMES[p] for p in preds],
        "confidence":    probas.max(axis=1),
        "prob_0":        probas[:, 0],
        "prob_1":        probas[:, 1],
        "prob_2":        probas[:, 2],
    }, index=X.index)

    if y_true is not None:
        df["true_regime"] = y_true.reindex(X.index)

    for col in FEATURE_COLS:
        df[col] = X[col].values

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info("Regime labels saved to %s (%d rows)", path, len(df))
    return path


def load_regime_labels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load regime labels parquet."""
    if path is None:
        path = LABELS_PARQUET
    return pd.read_parquet(path)


# ── RegimePipeline orchestration ──────────────────────────────────────────────

class RegimePipeline:
    """
    End-to-end regime classification pipeline.

    Usage
    -----
      pipe = RegimePipeline()
      pipe.run()          # load data → build features → train → evaluate → save
      pipe.plot()         # confusion matrix + distribution chart
      results = pipe.results_
    """

    def __init__(
        self,
        train_end:      str   = TRAIN_END_DATE,
        vvix_threshold: float = VVIX_REGIME2_THRESHOLD,
        pdv_model_path: Optional[Path] = None,
        xgb_params:     Optional[dict] = None,
    ):
        self.train_end      = train_end
        self.vvix_threshold = vvix_threshold
        self.pdv_model_path = pdv_model_path or (DATA_DIR / "pdv_model.pkl")
        self.xgb_params     = xgb_params

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
        spx = get_spx_ohlcv(as_of_date="2025-12-31", start_date="2010-01-01")
        vix = get_vix_term_structure_wide(as_of_date="2025-12-31", start_date="2010-01-01")

        # Load PDV model
        pdv_model = None
        if self.pdv_model_path.exists():
            with open(self.pdv_model_path, "rb") as fh:
                pdv_model = pickle.load(fh)
        else:
            logger.warning("PDV model not found at %s; using rv fallback", self.pdv_model_path)

        # Train
        self.clf_, self.X_test_, self.y_test_ = train_classifier(
            spx, vix, pdv_model,
            train_end=self.train_end,
            params=self.xgb_params,
            vvix_threshold=self.vvix_threshold,
        )

        # Build full aligned dataset for labelling all dates
        self.X_, self.y_ = build_dataset(spx, vix, pdv_model, self.vvix_threshold)

        # Evaluate
        self.results_ = evaluate_classifier(self.clf_, self.X_test_, self.y_test_)
        dist = regime_distribution_by_year(self.clf_, self.X_)
        self.results_["regime_dist_by_year"] = dist

        # Validate stress dates
        val = validate_regime2_dates(self.clf_, self.X_)
        self.results_["validation"] = val

        logger.info("Test accuracy: %.4f", self.results_["accuracy"])
        logger.info("Regime 2 recall: %.4f", self.results_["regime2_recall"])
        logger.info("Validation dates: %s", val)

        # Save model and labels
        self.clf_.save(MODEL_PATH)
        save_regime_labels(self.clf_, self.X_, self.y_)

        # Write to SQLite
        _write_labels_to_db(self.clf_, self.X_, self.y_)

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
            f"vvix_threshold={self.vvix_threshold}, status={status})"
        )


# ── DB write helper ────────────────────────────────────────────────────────────

def _write_labels_to_db(
    clf: RegimeClassifier,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
):
    """Write predictions to the regime_labels SQLite table."""
    preds  = clf.predict(X)
    probas = clf.predict_proba(X)

    # Build the rows expected by insert_regime_labels
    rows = pd.DataFrame({
        "date":       X.index.strftime("%Y-%m-%d"),
        "regime":     preds,
        "confidence": probas.max(axis=1),
        "vix_level":  X["vix"].values * 100,   # back to VIX points
        "vix_slope":  X["ts_slope"].values * 100,
        "rv_20d":     (X["vix"].values - X["pdv_iv_spread"].values) if "pdv_iv_spread" in X.columns else X["vix"].values,
    })
    try:
        n = insert_regime_labels(rows)
        logger.info("Wrote %d regime label rows to SQLite", n)
    except Exception as exc:
        logger.warning("DB write failed: %s", exc)
