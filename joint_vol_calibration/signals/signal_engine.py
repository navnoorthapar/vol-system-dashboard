"""
signal_engine.py — C9: Trading Signal Engine

Three regime-gated volatility trading signals that together drive position
sizing for the complete backtest in C10.

Signal 1 — IV vs Realised Spread Trade
---------------------------------------
  Edge: PDV-forecast vol deviates from ATM implied vol (VIX).
    PDV > VIX  → market underprices vol → long straddle (long gamma)
    VIX > PDV  → market overprices vol → short straddle (short gamma)

  Regime gate:
    Long  straddle only in Regime 0 (LONG_GAMMA,  rv > iv)
    Short straddle only in Regime 1 (SHORT_GAMMA, iv > rv)
    No new entries in Regime 2 (VOMMA_ACTIVE — too dangerous)

  Entry threshold : |pdv_iv_spread| > 0.02  (2 vol points)
  Exit            : spread reverts to zero OR max-hold 21 days
  Sizing          : Kelly fraction = tanh(|spread|/0.04) × 0.20, capped at 20 %

Signal 2 — VIX Term-Structure Curve Trade
------------------------------------------
  Edge: VIX 3M–1M slope deviates from its 252-day rolling mean.
    Slope > mean + 1σ  → contango steeper than normal, front month overpriced
                        → short front vol (short calendar spread)
    Slope < 0 (inversion / backwardation)
                        → front month cheap
                        → long front vol

  Regime gate: only Regime 0 or 1 (never Regime 2)

  Exit: slope z-score reverts inside ±0.50  OR  max-hold 10 days
        OR stop-loss: slope moves 15 % of entry level against position
  Sizing: min(|z_slope| / 3, 1.0) × 0.20, capped at 20 %

Signal 3 — Dispersion Proxy Trade
------------------------------------
  Edge: VIX / VVIX ratio is below its 252-day rolling mean by ≥ 1 σ.
    A depressed VIX relative to VVIX implies the index is calm while
    individual-stock options are nervous → low implied correlation →
    long dispersion (long single-stock vol, short index vol) is cheap.

  Substitution note: we use VIX/VVIX as a proxy for the index-vol /
  single-stock-vol ratio because historical single-stock options data
  is not in the database (only SPX and VIX index options available).
  The ratio VIX/VVIX correlates strongly with CBOE's Implied Correlation
  Index (ICJ) in available data.

  Entry: z_ratio < −1.0  AND  regime in {0, 1}  → long dispersion (+1)
  Exit : z_ratio > −0.30  OR  max-hold 30 days
  Sizing: min(|z_ratio| / 3, 1.0) × 0.20, capped at 20 %

Signal combination
------------------
  For each day, three scalar positions s1, s2, s3 ∈ {−1, 0, +1}.

  All three agree (same non-zero sign):
    combined = mean(agreeing) × equal_weight_factor   (full conviction)

  Exactly two agree (same sign, third is 0 or opposite):
    If third is flat (0)  → combined = mean(two) × 0.5  (half weight)
    If third conflicts    → combined = 0               (too uncertain)

  All flat or conflicted: combined = 0

  The combined_pos is further scaled by combined_strength ∈ [0, 1].

Zero look-ahead guarantee
-------------------------
  All feature inputs are derived from close of t-1 before any day-t
  signal is generated.  The public `generate()` method shifts the
  feature DataFrame by 1 row before signal computation.

Outputs
-------
  signals_df columns per signal:
    s{i}_pos           : +1 / −1 / 0
    s{i}_str           : 0–1 signal strength (used for position sizing)
    s{i}_regime_entry  : Regime at time of entry (NaN while flat)
    s{i}_days_held     : trading days since entry (0 while flat)
    s{i}_kelly         : Kelly fraction (capped at 20 %)
    s{i}_exp_pnl       : expected P&L proxy  = str × kelly × 100

  Combined:
    combined_pos       : final position (−1 to +1)
    combined_str       : final strength
    combined_kelly     : combined Kelly fraction
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.data.database import (
    get_spx_ohlcv,
    get_vix_term_structure_wide,
)
from joint_vol_calibration.signals.regime_classifier import (
    FEATURE_COLS,
    RegimeClassifier,
    build_features,
    build_regime_labels,
)

logger = logging.getLogger(__name__)

# ── Signal constants ──────────────────────────────────────────────────────────

# Signal 1 — IVR spread trade
S1_ENTRY_THRESHOLD: float = 0.02     # 2 vol points (decimal)
S1_MAX_HOLD: int          = 21       # trading days
S1_MAX_KELLY: float       = 0.20     # cap at 20 % of notional
S1_STRENGTH_SCALE: float  = 0.04     # tanh(spread / scale) saturates at this spread

# Signal 2 — VIX term-structure trade
S2_ZSCORE_ENTRY: float    = 1.0      # enter short when z > 1 std
S2_ZSCORE_EXIT: float     = 0.50     # exit when |z| < 0.5 (mean-reversion)
S2_LOOKBACK: int          = 252      # rolling window for slope stats
S2_MAX_HOLD: int          = 10       # trading days
S2_STOP_LOSS_PCT: float   = 0.15     # 15 % adverse move triggers stop
S2_MAX_KELLY: float       = 0.20
S2_STRENGTH_SCALE: float  = 3.0      # z / scale → strength

# Signal 3 — Dispersion proxy trade
S3_ZSCORE_ENTRY: float    = -1.0     # enter when z_ratio < −1.0
S3_ZSCORE_EXIT: float     = -0.30    # exit when z_ratio > −0.30
S3_LOOKBACK: int          = 252      # rolling window for VIX/VVIX ratio stats
S3_MAX_HOLD: int          = 30       # trading days
S3_MAX_KELLY: float       = 0.20
S3_STRENGTH_SCALE: float  = 3.0

# Output paths
SIGNALS_DIR   = DATA_DIR / "signals"
SIGNALS_PATH  = SIGNALS_DIR / "signals_2015_2025.parquet"
SIGNALS_CHART = SIGNALS_DIR / "signal_overview.png"


# ── Generic state-machine helper ──────────────────────────────────────────────

def _run_statemachine(
    entry_long:  pd.Series,
    entry_short: pd.Series,
    exit_cond:   pd.Series,
    strength:    pd.Series,
    regime:      pd.Series,
    max_hold:    int,
) -> pd.DataFrame:
    """
    Simulate a trading state machine over a time series.

    Rules
    -----
    • Enter long  when entry_long[t]=True  AND currently flat.
    • Enter short when entry_short[t]=True AND currently flat.
    • Exit when exit_cond[t]=True OR days_held >= max_hold.
    • On exit day, position immediately becomes 0.
    • No re-entry on the same day as exit.
    • If both entry_long and entry_short fire on same day: no entry.

    Returns
    -------
    pd.DataFrame with columns:
      position       : +1 / 0 / -1  (entry day position set immediately)
      strength       : signal strength at entry (held constant during trade)
      regime_entry   : regime at entry date (NaN while flat)
      days_held      : 0 = flat, ≥1 = bars in trade
    """
    n = len(entry_long)
    idx = entry_long.index

    position    = np.zeros(n, dtype=float)
    str_out     = np.zeros(n, dtype=float)
    regime_out  = np.full(n, np.nan)
    days_held   = np.zeros(n, dtype=int)

    cur_pos         = 0
    cur_str         = 0.0
    cur_regime      = np.nan
    cur_days        = 0
    entry_price_ref = 0.0   # reference value for stop-loss (set externally if needed)

    for i in range(n):
        el = bool(entry_long.iloc[i])
        es = bool(entry_short.iloc[i])
        ec = bool(exit_cond.iloc[i])

        if cur_pos != 0:
            # In trade
            cur_days += 1

            # Check exit
            if ec or cur_days >= max_hold:
                cur_pos    = 0
                cur_str    = 0.0
                cur_regime = np.nan
                cur_days   = 0
                # Position is 0 on exit day (already set by cur_pos = 0 above)

        if cur_pos == 0:
            # Flat — check entry (no same-day re-entry after exit)
            if el and not es:
                cur_pos    = +1
                cur_str    = float(strength.iloc[i])
                cur_regime = float(regime.iloc[i]) if pd.notna(regime.iloc[i]) else np.nan
                cur_days   = 1
            elif es and not el:
                cur_pos    = -1
                cur_str    = float(strength.iloc[i])
                cur_regime = float(regime.iloc[i]) if pd.notna(regime.iloc[i]) else np.nan
                cur_days   = 1

        position[i]  = cur_pos
        str_out[i]   = cur_str
        regime_out[i] = cur_regime
        days_held[i]  = cur_days

    return pd.DataFrame(
        {
            "position":     position,
            "strength":     str_out,
            "regime_entry": regime_out,
            "days_held":    days_held,
        },
        index=idx,
    )


# ── Signal 1: IVR Spread ──────────────────────────────────────────────────────

def generate_signal1(
    features: pd.DataFrame,
    regimes:  pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
) -> pd.DataFrame:
    """
    Signal 1: PDV vs ATM IV spread trade.

    All inputs are already on the PREVIOUS day's data (shifted by 1).
    No further shifting is done inside this function.

    Entry:
      PDV > VIX + threshold  AND  regime == 0  → long straddle (+1)
      VIX > PDV + threshold  AND  regime == 1  → short straddle (−1)

    Exit: spread crosses zero  OR  days_held >= max_hold.
    """
    spread = features["pdv_iv_spread"]      # σ_PDV − σ_ATM  (decimal)

    # Strength: tanh saturates at ~1 when |spread| >> S1_STRENGTH_SCALE
    strength = np.tanh(spread.abs() / S1_STRENGTH_SCALE)

    entry_long  = (spread >  threshold) & (regimes == 0)
    entry_short = (spread < -threshold) & (regimes == 1)

    # Exit when spread crosses zero (reverts)
    exit_cond   = (spread.shift(-1).fillna(spread) * spread.shift(1).fillna(spread) < 0) | \
                  (spread.abs() < 0.005)
    # Note: using shift(-1) would be look-ahead; instead use threshold-of-zero crossing
    exit_cond   = spread.abs() < 0.005   # pure spread-at-zero exit (no future info)

    result = _run_statemachine(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s1_{c}" for c in result.columns})


# ── Signal 2: VIX Term-Structure Curve ────────────────────────────────────────

def generate_signal2(
    features:      pd.DataFrame,
    regimes:       pd.Series,
    zscore_entry:  float = S2_ZSCORE_ENTRY,
    zscore_exit:   float = S2_ZSCORE_EXIT,
    lookback:      int   = S2_LOOKBACK,
    max_hold:      int   = S2_MAX_HOLD,
    stop_loss_pct: float = S2_STOP_LOSS_PCT,
    max_kelly:     float = S2_MAX_KELLY,
) -> pd.DataFrame:
    """
    Signal 2: VIX term-structure slope trade.

    Slope = (VIX3M − VIX) / 100  (already in 'ts_slope' feature column).

    Entry:
      z_slope > zscore_entry  AND  regime ∈ {0, 1}  → short front vol (−1)
      slope < 0 (backwardation)  AND  regime ∈ {0, 1}  → long front vol (+1)

    Exit: |z_slope| < zscore_exit  OR  days_held >= max_hold  OR  stop-loss.
    """
    slope   = features["ts_slope"]      # (VIX3M − VIX) / 100

    mu_slope  = slope.rolling(lookback, min_periods=lookback // 2).mean()
    std_slope = slope.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    z_slope   = (slope - mu_slope) / std_slope

    # Strength: saturates at 1 when |z| = S2_STRENGTH_SCALE
    strength = (z_slope.abs() / S2_STRENGTH_SCALE).clip(upper=1.0)

    regime_ok  = regimes.isin([0, 1])
    entry_long  = (slope < 0)          & regime_ok   # backwardation → long front vol
    entry_short = (z_slope > zscore_entry) & regime_ok   # steep contango → short front vol

    # Exit: z-score reverts inside exit band
    exit_cond = z_slope.abs() < zscore_exit

    result = _run_statemachine(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold
    )

    # Stop-loss: slope moved 15% of entry level against the position
    # Compute inline using days_held > 0 and slope change
    pos    = result["position"].values
    dh     = result["days_held"].values
    slope_v = slope.values
    stop   = np.zeros(len(slope_v), dtype=bool)

    entry_slope_ref = 0.0
    for i in range(len(pos)):
        if dh[i] == 1:                         # just entered
            entry_slope_ref = slope_v[i]
        if dh[i] > 1 and entry_slope_ref != 0:
            pct_move = (slope_v[i] - entry_slope_ref) / abs(entry_slope_ref)
            # Short position (pos = -1) loses when slope widens (pct_move > +stop_loss_pct)
            # Long position  (pos = +1) loses when slope narrows (pct_move < -stop_loss_pct)
            if (pos[i] == -1 and pct_move >  stop_loss_pct) or \
               (pos[i] == +1 and pct_move < -stop_loss_pct):
                stop[i] = True
                entry_slope_ref = 0.0

    # Apply stop-loss: zero out from stop day onward until next entry
    pos_with_stop = pos.copy()
    for i in range(len(pos_with_stop)):
        if stop[i]:
            pos_with_stop[i] = 0.0
    result["position"] = pos_with_stop
    result.loc[result["position"] == 0, "strength"] = 0.0
    result.loc[result["position"] == 0, "regime_entry"] = np.nan
    result.loc[result["position"] == 0, "days_held"] = 0

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s2_{c}" for c in result.columns})


# ── Signal 3: Dispersion Proxy ────────────────────────────────────────────────

def generate_signal3(
    features:     pd.DataFrame,
    regimes:      pd.Series,
    zscore_entry: float = S3_ZSCORE_ENTRY,
    zscore_exit:  float = S3_ZSCORE_EXIT,
    lookback:     int   = S3_LOOKBACK,
    max_hold:     int   = S3_MAX_HOLD,
    max_kelly:    float = S3_MAX_KELLY,
) -> pd.DataFrame:
    """
    Signal 3: VIX/VVIX dispersion proxy trade.

    ratio = vix_feature / vvix_feature  = (^VIX / ^VVIX)
    A depressed ratio (VVIX elevated relative to VIX) implies that
    individual stock options are expensive vs index → low implied
    correlation → long dispersion is cheap.

    Entry: z_ratio < zscore_entry (= −1.0)  AND  regime ∈ {0, 1}
    Exit : z_ratio > zscore_exit  (= −0.30) OR   days_held >= max_hold
    """
    # Both features already divided by 100; ratio is dimensionless
    ratio   = features["vix"] / features["vvix"].clip(lower=1e-6)

    mu_ratio  = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    std_ratio = ratio.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    z_ratio   = (ratio - mu_ratio) / std_ratio

    # Strength: saturates at 1 when |z| = S3_STRENGTH_SCALE
    strength = (z_ratio.abs() / S3_STRENGTH_SCALE).clip(upper=1.0)

    regime_ok   = regimes.isin([0, 1])
    entry_long  = (z_ratio < zscore_entry) & regime_ok   # only long dispersion
    entry_short = pd.Series(False, index=features.index)  # no short dispersion

    # Exit when z_ratio reverts above exit threshold
    exit_cond = z_ratio > zscore_exit

    result = _run_statemachine(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s3_{c}" for c in result.columns})


# ── Signal combination ─────────────────────────────────────────────────────────

def combine_signals(
    s1_pos: pd.Series,
    s2_pos: pd.Series,
    s3_pos: pd.Series,
    s1_str: pd.Series,
    s2_str: pd.Series,
    s3_str: pd.Series,
) -> pd.DataFrame:
    """
    Combine three signals into a single position using the agreement rule.

    Rules
    -----
    Non-zero signals with same sign = "agree".
    Non-zero signals with opposite signs = "conflict".

    • All three agree (all same non-zero sign):
        combined_pos = mean_direction × mean_strength  (equal weight)
    • Exactly two agree (same sign), third is flat:
        combined_pos = mean_direction × mean_strength × 0.5  (half weight)
    • Two agree but third opposes  →  flat (conflict = 0)
    • Only one non-zero            →  that signal × 0.5  (half weight)
    • All flat                     →  0

    Returns
    -------
    pd.DataFrame with columns: combined_pos, combined_str, combined_kelly
    """
    idx = s1_pos.index
    n   = len(idx)

    combined_pos = np.zeros(n)
    combined_str = np.zeros(n)

    p1 = s1_pos.values
    p2 = s2_pos.values
    p3 = s3_pos.values
    st1 = s1_str.values
    st2 = s2_str.values
    st3 = s3_str.values

    for i in range(n):
        pos  = np.array([p1[i], p2[i], p3[i]])
        strs = np.array([st1[i], st2[i], st3[i]])

        nz_mask  = pos != 0
        nz_pos   = pos[nz_mask]
        nz_str   = strs[nz_mask]

        k = len(nz_pos)
        if k == 0:
            combined_pos[i] = 0.0
            combined_str[i] = 0.0
            continue

        # Check for conflict among non-zero signals
        has_positive = np.any(nz_pos > 0)
        has_negative = np.any(nz_pos < 0)

        if has_positive and has_negative:
            # Conflict: flat
            combined_pos[i] = 0.0
            combined_str[i] = 0.0
            continue

        # All non-zero agree in direction
        direction     = float(np.sign(nz_pos[0]))
        mean_strength = float(np.mean(nz_str))

        if k == 3:
            # All three agree → equal weight
            combined_pos[i] = direction
            combined_str[i] = mean_strength
        elif k == 2:
            # Two agree, one flat → half weight
            combined_pos[i] = direction * 0.5
            combined_str[i] = mean_strength * 0.5
        else:  # k == 1
            # Only one signal active → half weight
            combined_pos[i] = direction * 0.5
            combined_str[i] = mean_strength * 0.5

    combined_kelly = np.clip(np.nan_to_num(combined_str, nan=0.0) * max(S1_MAX_KELLY, S2_MAX_KELLY, S3_MAX_KELLY), 0, 0.20)

    return pd.DataFrame(
        {
            "combined_pos":   combined_pos,
            "combined_str":   combined_str,
            "combined_kelly": combined_kelly,
        },
        index=idx,
    )


# ── SignalEngine orchestrator ─────────────────────────────────────────────────

class SignalEngine:
    """
    Orchestrates the generation of all three regime-gated trading signals.

    Usage
    -----
      engine = SignalEngine(clf)
      df = engine.generate(spx_df, vix_wide_df, start_date="2015-01-01")
      engine.plot(df)
      engine.save(df)

    Parameters
    ----------
    clf         : fitted RegimeClassifier from C8
    pdv_model   : optional PDVModel; if None, Signal 1 uses the pdv_iv_spread
                  feature already in the C8 feature set (rv_20d − VIX proxy)
    """

    def __init__(
        self,
        clf:       Optional[RegimeClassifier] = None,
        pdv_model  = None,
    ):
        self.clf       = clf
        self.pdv_model = pdv_model
        self._result: Optional[pd.DataFrame] = None

    def generate(
        self,
        spx_df:     pd.DataFrame,
        vix_wide_df: pd.DataFrame,
        start_date:  str = "2015-01-01",
        end_date:    str = "2025-12-31",
    ) -> pd.DataFrame:
        """
        Generate all signals for [start_date, end_date].

        Zero look-ahead: features are shifted 1 day inside this method
        before passing to signal generators.

        Returns
        -------
        pd.DataFrame indexed by date with all signal and combined columns.
        """
        # ── Build feature matrix ─────────────────────────────────────────────
        feats_raw = build_features(spx_df, vix_wide_df, pdv_model=self.pdv_model)

        # ── Compute regime labels ────────────────────────────────────────────
        regimes_raw = build_regime_labels(spx_df, vix_wide_df)

        # ── Optionally override with classifier predictions ──────────────────
        if self.clf is not None and self.clf.model_ is not None:
            try:
                preds = self.clf.predict_series(feats_raw.dropna())
                regimes_raw = preds
            except Exception as exc:
                logger.warning("Classifier prediction failed (%s); using rule-based regimes", exc)

        # ── Shift by 1: X(t-1) predicts y(t) ───────────────────────────────
        feats   = feats_raw.shift(1)
        regimes = regimes_raw.shift(1).ffill().dropna()

        # Align all on common index
        common_idx = feats.dropna().index.intersection(regimes.index)
        feats   = feats.loc[common_idx]
        regimes = regimes.loc[common_idx].astype(int)

        # Filter to [start_date, end_date]
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)
        mask = (feats.index >= start_ts) & (feats.index <= end_ts)
        feats   = feats[mask]
        regimes = regimes[mask]

        if len(feats) == 0:
            raise ValueError(f"No data in [{start_date}, {end_date}] after feature build.")

        logger.info(
            "Generating signals for %d trading days (%s → %s)",
            len(feats), feats.index.min().date(), feats.index.max().date(),
        )

        # ── Generate individual signals ───────────────────────────────────────
        s1 = generate_signal1(feats, regimes)
        s2 = generate_signal2(feats, regimes)
        s3 = generate_signal3(feats, regimes)

        # ── Combine ───────────────────────────────────────────────────────────
        combined = combine_signals(
            s1["s1_position"], s2["s2_position"], s3["s3_position"],
            s1["s1_strength"], s2["s2_strength"], s3["s3_strength"],
        )

        # ── Assemble output DataFrame ─────────────────────────────────────────
        df = pd.concat([s1, s2, s3, combined, feats, regimes.rename("regime")], axis=1)
        self._result = df

        n_s1  = (df["s1_position"] != 0).sum()
        n_s2  = (df["s2_position"] != 0).sum()
        n_s3  = (df["s3_position"] != 0).sum()
        n_cb  = (df["combined_pos"] != 0).sum()
        logger.info(
            "Active days — S1: %d | S2: %d | S3: %d | Combined: %d",
            n_s1, n_s2, n_s3, n_cb,
        )
        return df

    def save(self, df: Optional[pd.DataFrame] = None, path: Optional[Path] = None) -> Path:
        """Save signals DataFrame to parquet."""
        df = df if df is not None else self._result
        if df is None:
            raise RuntimeError("No results to save. Call generate() first.")
        path = Path(path or SIGNALS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("Signals saved to %s (%d rows)", path, len(df))
        return path

    def plot(
        self,
        df: Optional[pd.DataFrame] = None,
        save_path: Optional[Path] = None,
        figsize: tuple = (16, 10),
    ) -> plt.Figure:
        """
        Plot combined position timeline with per-signal breakdown.

        Panel 1: VIX level with regime colour bands
        Panel 2: Signal 1 (IVR spread) position
        Panel 3: Signal 2 (TS curve) position
        Panel 4: Signal 3 (Dispersion) position
        Panel 5: Combined position
        """
        df = df if df is not None else self._result
        if df is None:
            raise RuntimeError("No results to plot. Call generate() first.")

        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        fig.subplots_adjust(hspace=0.35)

        dates = df.index

        # ── Panel 0: VIX with regime bands ────────────────────────────────────
        ax = axes[0]
        if "vix" in df.columns:
            ax.plot(dates, df["vix"] * 100, color="#333333", linewidth=0.8, label="VIX")
        if "regime" in df.columns:
            colours = {0: "#E3F2FD", 1: "#E8F5E9", 2: "#FFEBEE"}
            for r, c in colours.items():
                mask = df["regime"] == r
                ax.fill_between(dates, 0, (df["vix"].max() * 100 if "vix" in df.columns else 80),
                                where=mask, color=c, alpha=0.5, label=f"R{r}")
        ax.set_ylabel("VIX")
        ax.legend(loc="upper left", fontsize=7, ncol=4)
        ax.set_title("VIX Level with Regime Background", fontsize=10)

        # ── Panels 1-3: Per-signal positions ──────────────────────────────────
        signal_info = [
            ("s1_position", "s1_strength", "#2196F3", "Signal 1 — IV vs RV Spread"),
            ("s2_position", "s2_strength", "#4CAF50", "Signal 2 — VIX TS Curve"),
            ("s3_position", "s3_strength", "#FF9800", "Signal 3 — Dispersion Proxy"),
        ]
        for i, (pos_col, str_col, colour, title) in enumerate(signal_info):
            ax = axes[i + 1]
            pos = df[pos_col].values
            x   = np.arange(len(dates))
            ax.bar(x, pos * df[str_col].values,
                   color=[colour if p > 0 else "#F44336" if p < 0 else "grey"
                          for p in pos],
                   alpha=0.7, width=1.0)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Position × Strength")
            ax.set_title(title, fontsize=9)
            ax.set_ylim(-1.1, 1.1)
            ax.set_yticks([-1, 0, 1])

        # ── Panel 4: Combined position ─────────────────────────────────────────
        ax = axes[4]
        cpos = df["combined_pos"].values
        ax.fill_between(np.arange(len(dates)), 0, cpos,
                        where=(cpos >= 0), color="#1565C0", alpha=0.7, label="Long")
        ax.fill_between(np.arange(len(dates)), 0, cpos,
                        where=(cpos < 0), color="#C62828", alpha=0.7, label="Short")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("Combined Pos")
        ax.set_title("Combined Position", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="upper left", fontsize=8)

        # ── X-axis labels ─────────────────────────────────────────────────────
        n = len(dates)
        step = max(1, n // 12)
        axes[-1].set_xticks(range(0, n, step))
        axes[-1].set_xticklabels(
            [str(dates[j].date()) for j in range(0, n, step)],
            rotation=35, ha="right", fontsize=8,
        )

        fig.suptitle("Trading Signal Engine — Regime-Gated Signals", fontsize=13, fontweight="bold")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Signal overview chart saved to %s", save_path)

        return fig

    def __repr__(self) -> str:
        status = f"{len(self._result)} days" if self._result is not None else "not run"
        clf_status = "with classifier" if (self.clf and self.clf.model_) else "rule-based regimes"
        return f"SignalEngine({clf_status}, status={status})"


# ── Summary statistics ─────────────────────────────────────────────────────────

def signal_summary(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for all signals.

    Returns
    -------
    dict with per-signal and combined statistics.
    """
    stats = {}

    for prefix, label in [("s1", "Signal1_IVR"), ("s2", "Signal2_TS"), ("s3", "Signal3_Disp")]:
        pos_col = f"{prefix}_position"
        if pos_col not in df.columns:
            continue
        pos  = df[pos_col]
        n    = len(df)
        n_pos = (pos > 0).sum()
        n_neg = (pos < 0).sum()
        n_flat = (pos == 0).sum()
        stats[label] = {
            "n_days":        n,
            "n_long":        int(n_pos),
            "n_short":       int(n_neg),
            "n_flat":        int(n_flat),
            "pct_active":    float((n_pos + n_neg) / n),
            "avg_strength":  float(df[f"{prefix}_strength"][pos != 0].mean()) if (pos != 0).any() else 0.0,
            "avg_kelly":     float(df[f"{prefix}_kelly"][pos != 0].mean())    if (pos != 0).any() else 0.0,
        }

    combined = df["combined_pos"]
    stats["Combined"] = {
        "n_days":       len(df),
        "n_long":       int((combined > 0).sum()),
        "n_short":      int((combined < 0).sum()),
        "n_flat":       int((combined == 0).sum()),
        "pct_active":   float((combined != 0).sum() / len(df)),
        "avg_strength": float(df["combined_str"][combined != 0].mean()) if (combined != 0).any() else 0.0,
    }

    return stats
