"""
signal_engine.py — C9: Trading Signal Engine

Regime-gated volatility trading signals that drive position sizing for the
backtest in C10. Every signal at session t is computed from features and
regime labels dated t−1 (see "Zero look-ahead" below).

Signal 1 — PDV–IV Spread Trade (formerly "IVR"; no implied-vol rank exists)
---------------------------------------------------------------------------
  Spread = σ̂(t) − VIX(t)/100, where σ̂ is the injected vol forecast
  (build_features(pdv_vol_series=...)) or the rv_20d proxy.
    σ̂ > VIX  → market underprices vol → long straddle (long gamma)
    VIX > σ̂  → market overprices vol → short straddle (short gamma)

  Regime gate:
    Long  straddle only in Regime 0 (LONG_GAMMA,  trailing rv > iv)
    Short straddle only in Regime 1 (SHORT_GAMMA, iv ≥ trailing rv)
    No new entries in Regime 2 (VOMMA_ACTIVE)

  Entry threshold : |spread| > 0.02  (2 vol points)
  Exit            : the spread reverts INTO the ±0.5 vp band OR crosses
                    through zero to the other side (the entry side is
                    remembered), OR max-hold 21 days
  Sizing          : size fraction = tanh(|spread|/0.04) × 0.20, capped at 20 %

Signal 1 — Regime-Filtered Variant (S1_regime_filtered)
---------------------------------------------------------
  Identical to Signal 1 except that on any day the current regime is R2
  (VOMMA_ACTIVE), the effective position output is scaled to 0.0 regardless
  of whether a trade is open.

  Motivation: S1 entry gates block new positions in R2, but an open trade
  entered in R0 or R1 can persist through a regime transition to R2. The
  original backtest (2018-2025) showed that such "regime overhang" trades
  contributed significantly to the −$503K S1 loss. The filtered variant
  zero-scales the position on R2 days while leaving the state machine running
  so that positions resume in R0/R1 if the max-hold has not elapsed.

  Output columns: s1rf_* (position, strength, regime_entry, days_held,
                           kelly, exp_pnl)

Signal 2 — VIX Term-Structure Curve Trade
------------------------------------------
  Edge: VIX 3M–1M slope deviates from its 252-day rolling mean.
    Slope > mean + 1σ  → contango steeper than normal, front month overpriced
                        → short front vol (SHORT SPX straddle, −1)
    Slope < 0 (inversion / backwardation)
                        → front month cheap
                        → long front vol (LONG SPX straddle, +1)

  Regime gate: only Regime 0 or 1 (never Regime 2)

  Exit: slope z-score reverts inside ±0.50  OR  max-hold 10 days
        OR stop-loss: front-month implied vol (VIX) moves 15 % of its entry
        level AGAINST the straddle actually traded — up for a short
        straddle, down for a long one. (Pre-C22 the stop was written on the
        slope with the sign inverted: it fired when the trade was winning.)
  Sizing: min(|z_slope| / 3, 1.0) × 0.20, capped at 20 %

Signal 3 — VIX/VVIX Ratio ("Dispersion Proxy") Trade
------------------------------------------------------
  Entry when the VIX / VVIX ratio is below its 252-day rolling mean by ≥ 1σ:
  VIX is calm relative to the vol-of-vol market.

  What +1 means operationally: a bet that the ratio REVERTS UP — VIX rising
  relative to VVIX. That is what the backtest's S3 P&L proxy rewards
  (P&L ∝ Δz of the ratio) and how the straddle combiner below counts it
  (long index vol). The original "long dispersion (short index vol)"
  reading would be the opposite sign; it is a hypothesis, not what is traded.

  Substitution note: no single-stock options or implied-correlation data are
  in the database, so the link to true dispersion pricing is unverified.

  Entry: z_ratio < −1.0  AND  regime in {0, 1}  → +1
  Exit : z_ratio > −0.30  OR  max-hold 30 days
  Sizing: min(|z_ratio| / 3, 1.0) × 0.20, capped at 20 %

Signal combination (research only; the published portfolio does not use it)
---------------------------------------------------------------------------
  Positions s1c, s3, s4 ∈ {−1, 0, +1}, each read as an index-vol direction.

    All three non-zero and agree      → direction × 1.0, mean strength
    Two agree, third flat             → direction × 0.5, mean strength × 0.5
    One non-zero, two flat            → direction × 0.5, strength × 0.5
    Any two non-zero signals conflict → 0
    All flat                          → 0

  (One active signal earns the same half weight as two agreeing ones.)

Zero look-ahead guarantee
-------------------------
  SignalEngine.generate() lags the feature matrix AND the regime labels by
  one SESSION before any signal is computed, so the signal for session t uses
  only data dated ≤ t−1 and can trade at the close of t. Rolling statistics
  are trailing and use all history supplied (entries start at start_date);
  regime labels passed in are only lagged, never modified.

Outputs
-------
  signals_df columns per signal:
    s{i}_position      : +1 / −1 / 0
    s{i}_strength      : 0–1 signal strength (used for position sizing)
    s{i}_regime_entry  : Regime at time of entry (NaN while flat)
    s{i}_days_held     : trading days since entry (0 while flat)
    s{i}_kelly         : size fraction (capped at 20 %; a heuristic, not a
                         Kelly-criterion estimate — the name is historical)
    s{i}_exp_pnl       : expected P&L proxy  = str × kelly × 100
    s2_z, s3_z, s4_z   : the continuous z-scores the signals threshold on

  Combined:
    combined_pos       : final position (−1 to +1)
    combined_str       : final strength
    combined_kelly     : combined size fraction
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.signals.regime_classifier import (
    R2_GATE,
    RegimeClassifier,
    build_features,
    build_regime_labels,
)

logger = logging.getLogger(__name__)

# ── Signal constants ──────────────────────────────────────────────────────────

# Signal 1 — PDV–IV spread trade
S1_ENTRY_THRESHOLD: float = 0.02     # 2 vol points (decimal)
S1_EXIT_BAND: float       = 0.005    # exit inside ±0.5 vp, or on crossing zero
S1_MAX_HOLD: int          = 21       # trading days
S1_MAX_KELLY: float       = 0.20     # cap at 20 % of notional
S1_STRENGTH_SCALE: float  = 0.04     # tanh(spread / scale) saturates at this spread
S1_RF_REGIME2_SCALE: float = 0.0    # position multiplier on R2 days for regime-filtered variant

# Signal 2 — VIX term-structure trade
S2_ZSCORE_ENTRY: float    = 1.0      # enter short when z > 1 std
S2_ZSCORE_EXIT: float     = 0.50     # exit when |z| < 0.5 (mean-reversion)
S2_LOOKBACK: int          = 252      # rolling window for slope stats
S2_MAX_HOLD: int          = 10       # trading days
S2_STOP_LOSS_PCT: float   = 0.15     # stop when VIX moves 15 % of its entry
                                     # level against the straddle held
S2_MAX_KELLY: float       = 0.20
S2_STRENGTH_SCALE: float  = 3.0      # z / scale → strength

# Signal 3 — VIX/VVIX ratio ("dispersion proxy") trade
S3_ZSCORE_ENTRY: float    = -1.0     # enter when z_ratio < −1.0
S3_ZSCORE_EXIT: float     = -0.30    # exit when z_ratio > −0.30
S3_LOOKBACK: int          = 252      # rolling window for VIX/VVIX ratio stats
S3_MAX_HOLD: int          = 30       # trading days
S3_MAX_KELLY: float       = 0.20
S3_STRENGTH_SCALE: float  = 3.0

# Signal 4 — Volatility Risk Premium (VRP)
# Edge: VIX (30-day implied) historically trades ~30% above 20-day realised vol.
# When the premium is abnormally elevated (z > 1), sell vol to collect it.
# When premium collapses (z < −0.5, realised running hot), buy cheap vol.
# Regime gate: shorts only in R1 (confirmed IV > RV); longs only in R0 (confirmed RV > IV).
# Hard exit on R2 transition (jump risk) via _run_statemachine_r2exit.
# Reference: Bakshi & Kapadia (2003), Carr & Wu (2009).
S4_Z_ENTRY_SHORT: float   = 1.0     # enter short when z_vrp > 1.0 (IV unusually rich)
S4_Z_ENTRY_LONG: float    = -0.5    # enter long  when z_vrp < −0.5 (IV unusually cheap)
S4_Z_EXIT: float          = 0.3     # exit when |z_vrp| < 0.3 (reverted to mean)
S4_LOOKBACK: int          = 252     # rolling window for VRP z-score
S4_MAX_HOLD: int          = 15      # shorter hold to reduce regime-change exposure
S4_MAX_KELLY: float       = 0.20
S4_STRENGTH_SCALE: float  = 2.0    # |z| / scale → strength (saturates at 2σ)

# Output paths. SignalEngine.save() names its default file after the date
# span actually saved (default_signals_path); a fixed "signals_2015_2025"
# name once labelled a 2018-01..2025-03 file that nothing read.
SIGNALS_DIR   = DATA_DIR / "signals"
SIGNALS_CHART = SIGNALS_DIR / "signal_overview.png"


def default_signals_path(df: pd.DataFrame) -> Path:
    """data_store/signals/signals_<first>_<last>.parquet for a signals frame."""
    idx = pd.to_datetime(df.index)
    return SIGNALS_DIR / f"signals_{idx.min():%Y%m%d}_{idx.max():%Y%m%d}.parquet"


# ── Generic state-machine helper ──────────────────────────────────────────────

def _side_exit(
    exit_cond:  pd.Series,
    exit_long:  Optional[pd.Series],
    exit_short: Optional[pd.Series],
) -> "tuple[np.ndarray, np.ndarray]":
    """Per-side exit flags: exit_cond OR the side-specific condition."""
    base = exit_cond.fillna(False).astype(bool).to_numpy()
    el = base | (exit_long.fillna(False).astype(bool).to_numpy() if exit_long is not None else False)
    es = base | (exit_short.fillna(False).astype(bool).to_numpy() if exit_short is not None else False)
    return el, es


def _entry_allowed(entry_allowed: Optional[pd.Series], n: int) -> np.ndarray:
    if entry_allowed is None:
        return np.ones(n, dtype=bool)
    return entry_allowed.fillna(False).astype(bool).to_numpy()


def _run_statemachine(
    entry_long:  pd.Series,
    entry_short: pd.Series,
    exit_cond:   pd.Series,
    strength:    pd.Series,
    regime:      pd.Series,
    max_hold:    int,
    exit_long:   Optional[pd.Series] = None,
    exit_short:  Optional[pd.Series] = None,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Simulate a trading state machine over a time series.

    Rules
    -----
    • Enter long  when entry_long[t]=True  AND currently flat.
    • Enter short when entry_short[t]=True AND currently flat.
    • Exit when exit_cond[t]=True OR days_held >= max_hold, or — for a long
      position — exit_long[t]=True, for a short — exit_short[t]=True
      (optional side-specific exits; NaN counts as False).
    • On exit day, position immediately becomes 0.
    • No re-entry on the same day as exit.
    • If both entry_long and entry_short fire on same day: no entry.
    • No entry on a day where entry_allowed is False (optional; lets
      trailing statistics use history before a backtest window while the
      book still starts flat at the window's first session).

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
    exit_l, exit_s = _side_exit(exit_cond, exit_long, exit_short)
    allow = _entry_allowed(entry_allowed, n)

    position    = np.zeros(n, dtype=float)
    str_out     = np.zeros(n, dtype=float)
    regime_out  = np.full(n, np.nan)
    days_held   = np.zeros(n, dtype=int)

    cur_pos         = 0
    cur_str         = 0.0
    cur_regime      = np.nan
    cur_days        = 0

    for i in range(n):
        el = bool(entry_long.iloc[i]) and allow[i]
        es = bool(entry_short.iloc[i]) and allow[i]
        ec = bool(exit_l[i]) if cur_pos > 0 else bool(exit_s[i])

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

        # `elif`, not `if`: when we started the day in a trade we never open a
        # new one the same day. This enforces the documented "no same-day
        # re-entry after exit" (otherwise a max_hold exit that still meets the
        # entry condition reopens instantly, defeating max_hold) and prevents a
        # direct +1↔−1 flip with no flat day in between.
        elif cur_pos == 0:
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


# ── R2-Exit State Machine ─────────────────────────────────────────────────────

def _run_statemachine_r2exit(
    entry_long:  pd.Series,
    entry_short: pd.Series,
    exit_cond:   pd.Series,
    strength:    pd.Series,
    regime:      pd.Series,
    max_hold:    int,
    exit_long:   Optional[pd.Series] = None,
    exit_short:  Optional[pd.Series] = None,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    State machine variant that EXITS positions when regime transitions INTO R2.

    When the regime transitions from a non-R2 value to R2 (VOMMA_ACTIVE) while
    a position is open, the position is immediately closed and the state machine
    reset.  Entry gates remain: no new entries are taken in R2.

    Differs from the S1RF (regime-filtered) variant, which zeroes the output
    on R2 days but leaves the state machine running.  This variant TERMINATES
    the trade at the R2 boundary — the position is gone, not paused.
    exit_long / exit_short / entry_allowed behave as in _run_statemachine.

    Returns
    -------
    pd.DataFrame with columns:
      position, strength, regime_entry, days_held, r2_exit (bool flag)
    """
    n   = len(entry_long)
    idx = entry_long.index
    exit_l, exit_s = _side_exit(exit_cond, exit_long, exit_short)
    allow = _entry_allowed(entry_allowed, n)

    position   = np.zeros(n, dtype=float)
    str_out    = np.zeros(n, dtype=float)
    regime_out = np.full(n, np.nan)
    days_held  = np.zeros(n, dtype=int)
    r2_exit    = np.zeros(n, dtype=bool)

    cur_pos    = 0
    cur_str    = 0.0
    cur_regime = np.nan
    cur_days   = 0
    prev_reg   = -1.0   # -1 sentinel = "no prior regime"

    for i in range(n):
        el      = bool(entry_long.iloc[i]) and allow[i]
        es      = bool(entry_short.iloc[i]) and allow[i]
        ec      = bool(exit_l[i]) if cur_pos > 0 else bool(exit_s[i])
        cur_reg = float(regime.iloc[i]) if pd.notna(regime.iloc[i]) else -1.0

        if cur_pos != 0:
            cur_days += 1

            # R2 transition: previous day was NOT R2, today IS R2 → forced exit
            if prev_reg != 2.0 and cur_reg == 2.0:
                cur_pos    = 0
                cur_str    = 0.0
                cur_regime = np.nan
                cur_days   = 0
                r2_exit[i] = True

            # Normal exit
            elif ec or cur_days >= max_hold:
                cur_pos    = 0
                cur_str    = 0.0
                cur_regime = np.nan
                cur_days   = 0

        # `elif` (not `if`): no same-day re-entry after any exit — including the
        # R2 forced exit, which must leave the book flat that day, not reopen.
        elif cur_pos == 0:
            if el and not es:
                cur_pos    = +1
                cur_str    = float(strength.iloc[i])
                cur_regime = cur_reg
                cur_days   = 1
            elif es and not el:
                cur_pos    = -1
                cur_str    = float(strength.iloc[i])
                cur_regime = cur_reg
                cur_days   = 1

        position[i]   = cur_pos
        str_out[i]    = cur_str
        regime_out[i] = cur_regime
        days_held[i]  = cur_days
        prev_reg      = cur_reg

    return pd.DataFrame(
        {
            "position":     position,
            "strength":     str_out,
            "regime_entry": regime_out,
            "days_held":    days_held,
            "r2_exit":      r2_exit,
        },
        index=idx,
    )


# ── Signal 1: PDV–IV Spread ───────────────────────────────────────────────────

def _s1_spread_exits(spread: pd.Series) -> "tuple[pd.Series, pd.Series]":
    """Exit flags keyed to the side of zero the spread was on at entry.

    A trade entered on a POSITIVE spread (> +threshold) exits once the spread
    is back below +S1_EXIT_BAND — inside the band or already through zero;
    a trade entered on a NEGATIVE spread exits once it is above −S1_EXIT_BAND.
    The old rule (|spread| < band only) missed sign flips that jumped over the
    ±0.5 vp band (10 of 23 in 2018-2025) and rode them to max-hold.

    Returns (exit_from_positive, exit_from_negative).
    """
    return spread < S1_EXIT_BAND, spread > -S1_EXIT_BAND


def generate_signal1(
    features: pd.DataFrame,
    regimes:  pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 1: PDV vs ATM IV spread trade.

    All inputs are already on the PREVIOUS day's data (shifted by 1).
    No further shifting is done inside this function.

    Entry:
      PDV > VIX + threshold  AND  regime == 0  → long straddle (+1)
      VIX > PDV + threshold  AND  regime == 1  → short straddle (−1)

    Exit: spread back inside ±S1_EXIT_BAND or through zero (see
    _s1_spread_exits)  OR  days_held >= max_hold.
    """
    spread = features["pdv_iv_spread"]      # σ_PDV − σ_ATM  (decimal)

    # Strength: tanh saturates at ~1 when |spread| >> S1_STRENGTH_SCALE
    strength = np.tanh(spread.abs() / S1_STRENGTH_SCALE)

    entry_long  = (spread >  threshold) & (regimes == 0)
    entry_short = (spread < -threshold) & (regimes == 1)

    # long entered on a positive spread, short on a negative one
    from_pos, from_neg = _s1_spread_exits(spread)
    no_exit = pd.Series(False, index=spread.index)

    result = _run_statemachine(
        entry_long, entry_short, no_exit, strength, regimes, max_hold,
        exit_long=from_pos, exit_short=from_neg, entry_allowed=entry_allowed,
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s1_{c}" for c in result.columns})


# ── Signal 1 Contrarian Variant (S1C) ─────────────────────────────────────────

def generate_signal1_contrarian(
    features:  pd.DataFrame,
    regimes:   pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 1 Contrarian (S1C): PDV model mean-reversion trade.

    Empirical finding (2018-2025 backtest): the PDV spread is a CONTRARIAN
    indicator. When PDV > ATM IV (spread > 0) in R0, the model overestimates
    fear — vol subsequently falls → SHORT straddle is profitable. When PDV <
    ATM IV in R1, the model underestimates the risk premium — buy the straddle
    at discount.

    Entry (opposite of S1):
      PDV > VIX + threshold  AND  regime == 0  → SHORT straddle (−1)
      VIX > PDV + threshold  AND  regime == 1  → LONG  straddle (+1)

    Exit: spread back inside ±S1_EXIT_BAND or through zero (see
    _s1_spread_exits)  OR  max_hold  OR  regime transitions INTO
    R2 (forced close via _run_statemachine_r2exit — C17). An open short
    straddle riding through a VVIX spike for up to 21 days with no stop
    was the single largest production tail risk; the R2 boundary now
    terminates the trade the same way S1X/S2X/S4 do.

    DATA-SNOOPING WARNING
    ---------------------
    This signal was constructed by observing that S1 LOST money over
    2018-2025 and flipping its direction. The "contrarian" hypothesis was
    formed on the SAME data used to evaluate it, so the backtest P&L of
    S1C is in-sample by construction and overstates expected live
    performance. Treat results as a hypothesis to be confirmed on data
    after the discovery date (2026-06), not as validated edge.
    """
    spread = features["pdv_iv_spread"]

    strength = np.tanh(spread.abs() / S1_STRENGTH_SCALE)

    # CONTRARIAN: short when PDV overestimates (spread > threshold in R0)
    #             long  when PDV underestimates (spread < -threshold in R1)
    entry_short = (spread >  threshold) & (regimes == 0)
    entry_long  = (spread < -threshold) & (regimes == 1)

    # short entered on a positive spread, long on a negative one
    from_pos, from_neg = _s1_spread_exits(spread)
    no_exit = pd.Series(False, index=spread.index)

    result = _run_statemachine_r2exit(
        entry_long, entry_short, no_exit, strength, regimes, max_hold,
        exit_long=from_neg, exit_short=from_pos, entry_allowed=entry_allowed,
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]   = kelly
    result["exp_pnl"] = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s1c_{c}" for c in result.columns})


# ── Signal 1 Regime-Filtered Variant ──────────────────────────────────────────

def generate_signal1_regime_filtered(
    features:  pd.DataFrame,
    regimes:   pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 1 — Regime-Filtered Variant (S1RF).

    Runs the identical state machine as generate_signal1() but zeroes the
    effective output (position, strength, kelly, exp_pnl) on every day where
    regime == 2 (VOMMA_ACTIVE).

    The underlying state machine keeps advancing through R2 days so that a
    trade entered in R0 or R1 can resume when the regime returns to R0/R1 —
    provided max_hold has not elapsed.

    Returns
    -------
    pd.DataFrame with columns prefixed ``s1rf_``:
        s1rf_position, s1rf_strength, s1rf_regime_entry, s1rf_days_held,
        s1rf_kelly, s1rf_exp_pnl
    """
    base = generate_signal1(features, regimes, threshold=threshold,
                            max_hold=max_hold, max_kelly=max_kelly,
                            entry_allowed=entry_allowed)

    result = base.rename(columns={c: c.replace("s1_", "s1rf_", 1) for c in base.columns})

    r2_mask = (regimes == 2).reindex(result.index, fill_value=False)
    result.loc[r2_mask, "s1rf_position"] = S1_RF_REGIME2_SCALE
    result.loc[r2_mask, "s1rf_strength"] = 0.0
    result.loc[r2_mask, "s1rf_kelly"]    = 0.0
    result.loc[r2_mask, "s1rf_exp_pnl"]  = 0.0

    return result


def compare_s1_regime_filter(
    signals_df: pd.DataFrame,
    start_date: str   = "2018-01-01",
    end_date:   str   = "2025-03-24",
    notional:   float = 1_000_000.0,
) -> dict:
    """
    Compare S1 (original) vs S1_regime_filtered over a date range.

    Edge-proxy P&L (first-order approximation):
        daily_pnl = position × pdv_iv_spread × kelly × notional / 252

    Classifies each S1 trade by whether it had any Regime-2 exposure while
    open (trade persisted through R2 after entering in R0/R1).

    Parameters
    ----------
    signals_df  : output of SignalEngine.generate() — must contain s1_*,
                  s1rf_*, pdv_iv_spread, regime columns.
    start_date  : inclusive left bound (YYYY-MM-DD).
    end_date    : inclusive right bound (YYYY-MM-DD).
    notional    : portfolio notional in dollars.

    Returns
    -------
    dict with keys:
        s1_total_pnl         float   total proxy P&L, original S1
        s1rf_total_pnl       float   total proxy P&L, S1 regime-filtered
        pnl_difference       float   s1rf_total_pnl − s1_total_pnl
        n_trades             int     distinct S1 trades in window
        n_trades_with_r2     int     trades that had ≥1 R2 day while open
        n_trades_without_r2  int     trades with no R2 exposure
        r2_days_zeroed       int     days where S1 non-zero but S1RF forced 0
        s1_pnl_series        pd.Series  daily S1 proxy P&L
        s1rf_pnl_series      pd.Series  daily S1RF proxy P&L
    """
    df = signals_df.copy()
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    df   = df[mask]

    if len(df) == 0:
        raise ValueError(f"No data in [{start_date}, {end_date}]")

    required = [
        "s1_position", "s1_kelly", "s1_days_held",
        "s1rf_position", "s1rf_kelly",
        "pdv_iv_spread", "regime",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    spread = df["pdv_iv_spread"].fillna(0.0)

    s1_pnl   = df["s1_position"]   * spread * df["s1_kelly"]   * notional / 252.0
    s1rf_pnl = df["s1rf_position"] * spread * df["s1rf_kelly"] * notional / 252.0

    # Identify distinct trades: days_held == 1 marks a new trade start
    dh          = df["s1_days_held"].values
    regime_vals = df["regime"].values

    trade_id   = np.zeros(len(df), dtype=int)
    current_id = 0
    for i in range(len(dh)):
        if dh[i] == 1:
            current_id += 1
        if dh[i] > 0:
            trade_id[i] = current_id

    n_with_r2    = 0
    n_without_r2 = 0
    for tid in range(1, current_id + 1):
        mask_tid = trade_id == tid
        if np.any(regime_vals[mask_tid] == 2):
            n_with_r2 += 1
        else:
            n_without_r2 += 1

    r2_days_zeroed = int(((df["regime"] == 2) & (df["s1_position"] != 0)).sum())

    return {
        "s1_total_pnl":        float(s1_pnl.sum()),
        "s1rf_total_pnl":      float(s1rf_pnl.sum()),
        "pnl_difference":      float(s1rf_pnl.sum() - s1_pnl.sum()),
        "n_trades":            current_id,
        "n_trades_with_r2":    n_with_r2,
        "n_trades_without_r2": n_without_r2,
        "r2_days_zeroed":      r2_days_zeroed,
        "s1_pnl_series":       s1_pnl,
        "s1rf_pnl_series":     s1rf_pnl,
    }


# ── Signal 1 R2-Exit Variant ───────────────────────────────────────────────────

def generate_signal1_r2exit(
    features:  pd.DataFrame,
    regimes:   pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 1 — R2-Exit Variant (S1X).

    When regime TRANSITIONS into R2 (prev day not R2, today R2) while a
    position is open, the position is closed immediately and the state machine
    is reset.  Entry gates remain: no new entries are allowed in R2.

    Compare with S1RF: that variant zeroes the output but keeps the state
    machine running so trades can resume when regime returns to R0/R1.
    S1X TERMINATES the trade at the R2 boundary.

    Returns
    -------
    pd.DataFrame with columns prefixed ``s1x_``:
        s1x_position, s1x_strength, s1x_regime_entry, s1x_days_held,
        s1x_kelly, s1x_exp_pnl, s1x_r2_exit
    """
    spread   = features["pdv_iv_spread"]
    strength = np.tanh(spread.abs() / S1_STRENGTH_SCALE)

    entry_long  = (spread >  threshold) & (regimes == 0)
    entry_short = (spread < -threshold) & (regimes == 1)
    from_pos, from_neg = _s1_spread_exits(spread)
    no_exit = pd.Series(False, index=spread.index)

    result = _run_statemachine_r2exit(
        entry_long, entry_short, no_exit, strength, regimes, max_hold,
        exit_long=from_pos, exit_short=from_neg, entry_allowed=entry_allowed,
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]   = kelly
    result["exp_pnl"] = result["strength"].fillna(0.0) * kelly * 100.0
    return result.rename(columns={c: f"s1x_{c}" for c in result.columns})


# ── Signal 1 Soft Regime-Scaling Variant ──────────────────────────────────────

def generate_signal1_soft(
    features:  pd.DataFrame,
    regimes:   pd.Series,
    prob_r2:   pd.Series,
    threshold: float = S1_ENTRY_THRESHOLD,
    max_hold:  int   = S1_MAX_HOLD,
    max_kelly: float = S1_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 1 — Soft Regime-Scaling Variant (S1S).

    Position is scaled by a step function of calibrated P(Regime 2):
      - P(R2) < 0.4  →  scale = 1.0  (full position)
      - P(R2) ∈ [0.4, 0.6)  →  scale = 0.5  (half position)
      - P(R2) ≥ 0.6  →  scale = 0.0  (flat / no trade)

    The 0.4 / 0.6 cut-offs only mean something if prob_r2 is a calibrated
    probability: use a RegimeClassifier fit with calibration="holdout"
    (isotonic map fit on a later window the booster never saw). The raw
    booster output, trained with balanced class weights, overstates P(R2).
    Unlike continuous (1 − prob_r2), the three-level step gives cleaner
    risk-sizing behaviour.

    Parameters
    ----------
    prob_r2 : pd.Series — predicted P(regime == 2), aligned with features index.
              Must already be shifted 1 day (same as features) — no extra shift.

    Returns
    -------
    pd.DataFrame with columns prefixed ``s1s_``:
        s1s_position, s1s_strength, s1s_regime_entry, s1s_days_held,
        s1s_kelly, s1s_exp_pnl, s1s_scale
    """
    base = generate_signal1(
        features, regimes, threshold=threshold, max_hold=max_hold, max_kelly=max_kelly,
        entry_allowed=entry_allowed,
    )
    result = base.rename(columns={c: c.replace("s1_", "s1s_", 1) for c in base.columns})

    # Align prob_r2 to result index; clip to [0, 1]
    prob_r2_a = prob_r2.reindex(result.index).fillna(0.0).clip(0.0, 1.0).values

    # Three-level step function: full / half / flat
    scale = np.where(prob_r2_a < 0.4, 1.0, np.where(prob_r2_a < 0.6, 0.5, 0.0))
    scale_s = pd.Series(scale, index=result.index)

    result["s1s_scale"]    = scale_s
    result["s1s_position"] = result["s1s_position"] * scale_s
    result["s1s_strength"] = result["s1s_strength"] * scale_s
    result["s1s_kelly"]    = result["s1s_kelly"]    * scale_s
    result["s1s_exp_pnl"]  = result["s1s_exp_pnl"]  * scale_s

    return result


# ── Signal 2: VIX Term-Structure Curve ────────────────────────────────────────

def _s2_slope_zscore(slope: pd.Series, lookback: int) -> pd.Series:
    """Trailing z-score of the VIX3M − VIX slope (window ends at t)."""
    mu_slope  = slope.rolling(lookback, min_periods=lookback // 2).mean()
    std_slope = slope.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    return (slope - mu_slope) / std_slope


def _apply_s2_stop(
    result:        pd.DataFrame,
    front_vol:     pd.Series,
    stop_loss_pct: float,
) -> pd.DataFrame:
    """Stop an S2 trade when front-month implied vol moves against the straddle.

    S2 trades an SPX straddle: +1 = LONG straddle (loses when front vol
    falls), −1 = SHORT straddle (loses when front vol rises). With
    v_entry = VIX on the entry session, the adverse move on a later session is

        adverse_t = −position × (VIX_t / v_entry − 1)

    and the stop fires when adverse_t > stop_loss_pct. The position is then
    zeroed from the stop session until the state machine starts a NEW trade
    (a single-day zero would let the trade resume on the next bar).

    Pre-C22 the stop was written on the SLOPE (VIX3M − VIX) with the sign
    inverted: it stopped shorts when contango steepened and longs when
    backwardation deepened — both favourable for the straddle actually held —
    so all 8 stops in 2018-2025 fired on winning trades.
    """
    pos  = result["position"].to_numpy()
    dh   = result["days_held"].to_numpy()
    vol  = front_vol.to_numpy(dtype=float)
    stop = np.zeros(len(pos), dtype=bool)

    entry_vol = np.nan
    for i in range(len(pos)):
        if dh[i] == 1:                         # just entered
            entry_vol = vol[i]
        if dh[i] > 1 and pos[i] != 0 and np.isfinite(entry_vol) and entry_vol > 0 \
                and np.isfinite(vol[i]):
            adverse = -pos[i] * (vol[i] / entry_vol - 1.0)
            if adverse > stop_loss_pct:
                stop[i] = True
                entry_vol = np.nan

    pos_with_stop = pos.copy()
    stopped = False
    for i in range(len(pos_with_stop)):
        if stop[i]:
            pos_with_stop[i] = 0.0
            stopped = True
        elif stopped:
            if dh[i] == 1:  # new entry in the state machine → stop cleared
                stopped = False
            else:
                pos_with_stop[i] = 0.0
    result = result.copy()
    result["position"] = pos_with_stop
    result.loc[result["position"] == 0, "strength"] = 0.0
    result.loc[result["position"] == 0, "regime_entry"] = np.nan
    result.loc[result["position"] == 0, "days_held"] = 0
    result["stop"] = stop
    return result


def _generate_signal2_core(
    features:      pd.DataFrame,
    regimes:       pd.Series,
    zscore_entry:  float,
    zscore_exit:   float,
    lookback:      int,
    max_hold:      int,
    stop_loss_pct: float,
    max_kelly:     float,
    statemachine,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    slope   = features["ts_slope"]      # (VIX3M − VIX) / 100
    z_slope = _s2_slope_zscore(slope, lookback)

    # Strength: saturates at 1 when |z| = S2_STRENGTH_SCALE
    strength = (z_slope.abs() / S2_STRENGTH_SCALE).clip(upper=1.0)

    regime_ok   = regimes.isin([0, 1])
    entry_long  = (slope < 0)              & regime_ok   # backwardation → long front vol
    entry_short = (z_slope > zscore_entry) & regime_ok   # steep contango → short front vol

    # Exit: z-score reverts inside exit band
    exit_cond = z_slope.abs() < zscore_exit

    result = statemachine(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold,
        entry_allowed=entry_allowed,
    )
    result = _apply_s2_stop(result, features["vix"], stop_loss_pct)

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    result["z"]        = z_slope
    return result


def generate_signal2(
    features:      pd.DataFrame,
    regimes:       pd.Series,
    zscore_entry:  float = S2_ZSCORE_ENTRY,
    zscore_exit:   float = S2_ZSCORE_EXIT,
    lookback:      int   = S2_LOOKBACK,
    max_hold:      int   = S2_MAX_HOLD,
    stop_loss_pct: float = S2_STOP_LOSS_PCT,
    max_kelly:     float = S2_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 2: VIX term-structure slope trade, traded as an SPX straddle.

    Slope = (VIX3M − VIX) / 100  (already in 'ts_slope' feature column).

    Entry:
      z_slope > zscore_entry  AND  regime ∈ {0, 1}  → short straddle (−1)
      slope < 0 (backwardation)  AND  regime ∈ {0, 1}  → long straddle (+1)

    Exit: |z_slope| < zscore_exit  OR  days_held >= max_hold  OR  stop-loss
    (VIX moves stop_loss_pct of its entry level against the straddle; see
    _apply_s2_stop). Output adds s2_stop (stop sessions) and s2_z (z_slope).
    """
    result = _generate_signal2_core(
        features, regimes, zscore_entry, zscore_exit, lookback, max_hold,
        stop_loss_pct, max_kelly, _run_statemachine, entry_allowed=entry_allowed,
    )
    return result.rename(columns={c: f"s2_{c}" for c in result.columns})


# ── Signal 2 R2-Exit Variant ───────────────────────────────────────────────────

def generate_signal2_r2exit(
    features:      pd.DataFrame,
    regimes:       pd.Series,
    zscore_entry:  float = S2_ZSCORE_ENTRY,
    zscore_exit:   float = S2_ZSCORE_EXIT,
    lookback:      int   = S2_LOOKBACK,
    max_hold:      int   = S2_MAX_HOLD,
    stop_loss_pct: float = S2_STOP_LOSS_PCT,
    max_kelly:     float = S2_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 2 — R2-Exit Variant (S2X).

    Identical to generate_signal2() (including the VIX-based stop) except
    that positions are forcibly closed when regime transitions INTO R2
    (prev day not R2, today R2).

    Returns
    -------
    pd.DataFrame with columns prefixed ``s2x_``.
    """
    result = _generate_signal2_core(
        features, regimes, zscore_entry, zscore_exit, lookback, max_hold,
        stop_loss_pct, max_kelly, _run_statemachine_r2exit, entry_allowed=entry_allowed,
    )
    return result.rename(columns={c: f"s2x_{c}" for c in result.columns})


# ── Signal 3: VIX/VVIX Ratio ("Dispersion Proxy") ────────────────────────────

def generate_signal3(
    features:     pd.DataFrame,
    regimes:      pd.Series,
    zscore_entry: float = S3_ZSCORE_ENTRY,
    zscore_exit:  float = S3_ZSCORE_EXIT,
    lookback:     int   = S3_LOOKBACK,
    max_hold:     int   = S3_MAX_HOLD,
    max_kelly:    float = S3_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 3: VIX/VVIX ratio mean-reversion ("dispersion proxy").

    ratio = vix_feature / vvix_feature  = (^VIX / ^VVIX)
    A depressed ratio means VIX is calm relative to the vol-of-vol market.
    +1 is a bet that the ratio reverts UP (VIX rising relative to VVIX) —
    what the backtest's Δz P&L proxy rewards; see the module docstring for
    why the "long dispersion (short index vol)" reading is not what is traded.

    Entry: z_ratio < zscore_entry (= −1.0)  AND  regime ∈ {0, 1}
    Exit : z_ratio > zscore_exit  (= −0.30) OR   days_held >= max_hold
    Output adds s3_z (z_ratio, trailing window ending at t).
    """
    # Both features already divided by 100; ratio is dimensionless
    ratio   = features["vix"] / features["vvix"].clip(lower=1e-6)

    mu_ratio  = ratio.rolling(lookback, min_periods=lookback // 2).mean()
    std_ratio = ratio.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    z_ratio   = (ratio - mu_ratio) / std_ratio

    # Strength: saturates at 1 when |z| = S3_STRENGTH_SCALE
    strength = (z_ratio.abs() / S3_STRENGTH_SCALE).clip(upper=1.0)

    regime_ok   = regimes.isin([0, 1])
    entry_long  = (z_ratio < zscore_entry) & regime_ok   # only the +1 side
    entry_short = pd.Series(False, index=features.index)  # never −1

    # Exit when z_ratio reverts above exit threshold
    exit_cond = z_ratio > zscore_exit

    result = _run_statemachine(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold,
        entry_allowed=entry_allowed,
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]    = kelly
    result["exp_pnl"]  = result["strength"].fillna(0.0) * kelly * 100.0
    result["z"]        = z_ratio
    return result.rename(columns={c: f"s3_{c}" for c in result.columns})


# ── Signal 4: Volatility Risk Premium (VRP) ───────────────────────────────────

def generate_signal4(
    features:      pd.DataFrame,
    regimes:       pd.Series,
    z_entry_short: float = S4_Z_ENTRY_SHORT,
    z_entry_long:  float = S4_Z_ENTRY_LONG,
    z_exit:        float = S4_Z_EXIT,
    lookback:      int   = S4_LOOKBACK,
    max_hold:      int   = S4_MAX_HOLD,
    max_kelly:     float = S4_MAX_KELLY,
    entry_allowed: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Signal 4: Volatility Risk Premium (VRP) mean-reversion.

    VRP = VIX_implied − RV_20d (both annualised decimal vol).
    Reconstructed from existing features: vrp = vix × (1 − 1/fear_premium).

    Entry:
      z_vrp >  z_entry_short  AND  regime == 1  → short straddle (IV unusually rich)
      z_vrp <  z_entry_long   AND  regime == 0  → long straddle  (IV unusually cheap)

    Exit: |z_vrp| < z_exit  OR  days_held >= max_hold  OR  regime → R2 (jump risk).
    Output adds s4_z (z_vrp, trailing window ending at t).
    """
    fear = features["fear_premium"].clip(lower=0.01)
    vrp  = features["vix"] * (1.0 - 1.0 / fear)   # positive when IV > RV (normal)

    mu_vrp  = vrp.rolling(lookback, min_periods=lookback // 2).mean()
    std_vrp = vrp.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-6)
    z_vrp   = (vrp - mu_vrp) / std_vrp
    z_vrp_prev = z_vrp.shift(1)  # yesterday's z-score to detect peak/trough

    strength = (z_vrp.abs() / S4_STRENGTH_SCALE).clip(upper=1.0)

    # Regime-specific entries: require z_vrp to have PEAKED and be declining
    # before selling vol (or TROUGHED and rising before buying).
    # Rationale: z_vrp > threshold fires at the peak of fear — entering a short
    # straddle INTO a vol spike is negative-expectancy. Waiting for the z to turn
    # (start declining) means the event has happened and we are in mean-reversion.
    entry_short = (z_vrp >  z_entry_short) & (z_vrp < z_vrp_prev) & (regimes == 1)
    entry_long  = (z_vrp <  z_entry_long)  & (z_vrp > z_vrp_prev) & (regimes == 0)

    exit_cond = z_vrp.abs() < z_exit

    # R2-exit state machine: immediately close on regime → R2 (jump protection)
    result = _run_statemachine_r2exit(
        entry_long, entry_short, exit_cond, strength, regimes, max_hold,
        entry_allowed=entry_allowed,
    )

    kelly = (result["strength"] * max_kelly).clip(upper=max_kelly).fillna(0.0)
    result["kelly"]   = kelly
    result["exp_pnl"] = result["strength"].fillna(0.0) * kelly * 100.0
    result["z"]       = z_vrp

    return result.rename(columns={c: f"s4_{c}" for c in result.columns})


# ── Signal combination ─────────────────────────────────────────────────────────

def combine_signals(
    s1_pos: pd.Series,
    s2_pos: pd.Series,
    s3_pos: pd.Series,
    s1_str: pd.Series,
    s2_str: pd.Series,
    s3_str: pd.Series,
    max_kelly: Optional[float] = None,
) -> pd.DataFrame:
    """
    Combine three signals into a single position using the agreement rule.

    Every input position must be expressed in the same units — the direction
    of the index straddle (+1 long vol, −1 short vol) — because combined_pos
    is traded as one SPX straddle.

    Rules
    -----
    Non-zero signals with same sign = "agree".
    Non-zero signals with opposite signs = "conflict".

    • All three agree (all same non-zero sign):
        combined_pos = direction,        combined_str = mean_strength
    • Exactly two agree (same sign), third is flat:
        combined_pos = direction × 0.5,  combined_str = mean_strength × 0.5
    • Only one non-zero:
        combined_pos = direction × 0.5,  combined_str = strength × 0.5
      (one active signal earns the same half weight as two agreeing ones)
    • Any conflict among the non-zero signals  →  0
    • All flat                     →  0

    combined_kelly = combined_str × max_kelly, capped at max_kelly
    (default: the largest per-signal cap, S1–S4).

    Returns
    -------
    pd.DataFrame with columns: combined_pos, combined_str, combined_kelly
    """
    if max_kelly is None:
        max_kelly = max(S1_MAX_KELLY, S2_MAX_KELLY, S3_MAX_KELLY, S4_MAX_KELLY)
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

    combined_kelly = np.clip(np.nan_to_num(combined_str, nan=0.0) * max_kelly, 0, max_kelly)

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
    Orchestrates the generation of all regime-gated trading signals.

    Usage
    -----
      engine = SignalEngine()
      df = engine.generate(spx_df, vix_wide_df, start_date="2018-01-01")
      engine.plot(df)
      engine.save(df)

    Parameters
    ----------
    clf         : optional fitted RegimeClassifier (RESEARCH ONLY). Used for
                  regimes only when no precomputed_regimes are given; R2 is
                  then still set by the canonical rule gate.
    pdv_model   : optional full-sample PDVModel (look-ahead; legacy). Prefer
                  generate(precomputed_pdv_vol=...) with walk-forward σ̂.
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
        spx_df:              pd.DataFrame,
        vix_wide_df:         pd.DataFrame,
        start_date:          str = "2015-01-01",
        end_date:            str = "2025-12-31",
        precomputed_regimes: "Optional[pd.Series]" = None,
        precomputed_pdv_vol: "Optional[pd.Series]" = None,
    ) -> pd.DataFrame:
        """
        Generate all signals for [start_date, end_date].

        Zero look-ahead: features AND regime labels are lagged one SESSION
        here, before any signal generator sees them; the row for session t
        holds only information dated ≤ t−1.

        Parameters
        ----------
        precomputed_regimes : optional regime labels dated at the session they
            describe (e.g. build_regime_labels(...) with the canonical gate).
            They are used exactly as given — lagged one session, never
            modified. (Pre-C22 a hard-coded VVIX > 100 override was OR-ed on
            top, so labels built with any other gate never took effect.)
            Without them: the classifier's predictions if a fitted clf was
            given (with R2 forced wherever the canonical rule label is R2),
            else build_regime_labels(spx_df, vix_wide_df).
        precomputed_pdv_vol : optional walk-forward σ̂ forecast Series — the
            pdv_iv_spread injection hook; see build_features(pdv_vol_series).

        Every session in the window with a known lagged regime is kept, even
        if some feature is NaN: each signal treats a NaN input as "no entry /
        no exit" for that session. (Pre-C22 any NaN feature — e.g. one missing
        VVIX print, which only S3 uses — removed the whole session from the
        output and hence from the backtest P&L path.)

        Trailing statistics (the S2/S3/S4 z-scores) are computed over ALL the
        history supplied, but entries are allowed only inside [start_date,
        end_date], so the book starts flat on the window's first session.
        (Pre-C22 the features were cut to the window first, so the 252-day
        z-scores warmed up INSIDE it: in the 2018-2025 backtest S3/S4 never
        traded, and S2 traded once, in 2018-01..2018-07, although data were
        loaded from 2015 precisely to warm them.)

        Returns
        -------
        pd.DataFrame indexed by date with all signal columns (s1, s1c, s1rf,
        s1x, s2, s2x, s3, s4 prefixes), the combined columns, the lagged
        features and the lagged regime.
        .attrs: regime_source, pdv_iv_spread_source, n_sessions_dropped.
        """
        # ── Build feature matrix (values at the close of each session) ───────
        feats_raw = build_features(
            spx_df, vix_wide_df,
            pdv_model=self.pdv_model,
            pdv_vol_series=precomputed_pdv_vol,
        )
        sessions = feats_raw.index

        # ── Regime labels (dated at the session they describe) ────────────────
        if precomputed_regimes is not None:
            regimes_raw = pd.Series(precomputed_regimes, copy=True)
            regimes_raw.index = pd.to_datetime(regimes_raw.index)
            regimes_raw = regimes_raw[~regimes_raw.index.duplicated(keep="last")]
            regime_source = "precomputed"
        elif self.clf is not None and self.clf.model_ is not None:
            try:
                cols = list(self.clf.feature_names_ or feats_raw.columns)
                preds = self.clf.predict_series(feats_raw.dropna(subset=cols))
                # R2 is a deterministic risk rule, not a prediction problem:
                # wherever the canonical rule label is R2, the regime is R2.
                rule = build_regime_labels(spx_df, vix_wide_df, R2_GATE)
                regimes_raw = preds.copy()
                regimes_raw[rule.reindex(preds.index) == 2] = 2
                regime_source = "classifier+rule_r2"
            except Exception as exc:
                logger.warning("Classifier prediction failed (%s); using rule-based regimes", exc)
                regimes_raw = build_regime_labels(spx_df, vix_wide_df)
                regime_source = "rule"
        else:
            regimes_raw = build_regime_labels(spx_df, vix_wide_df)
            regime_source = "rule"

        # ── Lag by one SESSION: information dated t−1 drives session t ────────
        # Reindex labels to the session calendar first so the shift moves one
        # session, not one position of whatever index they arrived with; the
        # ffill only carries an already-known label over a session that has
        # none (never anything dated after t−1).
        feats   = feats_raw.shift(1)
        regimes = regimes_raw.reindex(sessions).shift(1).ffill()

        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date)
        has_regime = regimes.notna().to_numpy()
        in_window = (sessions >= start_ts) & (sessions <= end_ts)

        n_dropped = int((in_window & ~has_regime).sum())
        if n_dropped:
            dropped = sessions[in_window & ~has_regime]
            logger.warning(
                "%d session(s) in [%s, %s] have no lagged regime label and are "
                "dropped (first: %s)", n_dropped, start_date, end_date,
                ", ".join(str(d.date()) for d in dropped[:5]),
            )

        # Full history (sessions with a regime) for the trailing statistics;
        # entries only inside the window.
        feats   = feats[has_regime]
        regimes = regimes[has_regime].astype(int)
        window  = pd.Series(in_window[has_regime], index=feats.index)

        if not window.any():
            raise ValueError(f"No data in [{start_date}, {end_date}] after feature build.")

        logger.info(
            "Generating signals for %d trading days (%s → %s), regimes: %s",
            int(window.sum()), feats.index[window.values].min().date(),
            feats.index[window.values].max().date(), regime_source,
        )

        # ── Generate individual signals ───────────────────────────────────────
        kw = {"entry_allowed": window}
        s1   = generate_signal1(feats, regimes, **kw)
        s1c  = generate_signal1_contrarian(feats, regimes, **kw)
        s1rf = generate_signal1_regime_filtered(feats, regimes, **kw)
        s1x  = generate_signal1_r2exit(feats, regimes, **kw)
        s2   = generate_signal2(feats, regimes, **kw)
        s2x  = generate_signal2_r2exit(feats, regimes, **kw)
        s3   = generate_signal3(feats, regimes, **kw)
        s4   = generate_signal4(feats, regimes, **kw)

        # ── Combine S1C + S3 + S4 as index-vol directions (research only) ─────
        # S1 removed: PDV spread is a lagging indicator — backwards as directional signal.
        # S2 removed: no mechanistic basis for TS slope predicting vol direction.
        # S1C: contrarian use of the spread (±1 = long/short straddle).
        # S3: +1 = VIX/VVIX ratio expected to revert up (VIX rising relative to
        #     VVIX), counted as long index vol — the direction its P&L proxy pays.
        # S4: VRP mean-reversion after z peaks (±1 = long/short straddle).
        combined = combine_signals(
            s1c["s1c_position"], s3["s3_position"], s4["s4_position"],
            s1c["s1c_strength"], s3["s3_strength"], s4["s4_strength"],
            max_kelly=max(S1_MAX_KELLY, S3_MAX_KELLY, S4_MAX_KELLY),
        )

        # ── Assemble output DataFrame (window only) ───────────────────────────
        df = pd.concat(
            [s1, s1c, s1rf, s1x, s2, s2x, s3, s4, combined, feats, regimes.rename("regime")],
            axis=1,
        ).loc[window.values]
        df.attrs["regime_source"] = regime_source
        df.attrs["pdv_iv_spread_source"] = feats_raw.attrs.get("pdv_iv_spread_source")
        df.attrs["n_sessions_dropped"] = n_dropped
        self._result = df

        n_s1   = (df["s1_position"]   != 0).sum()
        n_s1c  = (df["s1c_position"]  != 0).sum()
        n_s1rf = (df["s1rf_position"] != 0).sum()
        n_s2   = (df["s2_position"]   != 0).sum()
        n_s3   = (df["s3_position"]   != 0).sum()
        n_s4   = (df["s4_position"]   != 0).sum()
        n_cb   = (df["combined_pos"]  != 0).sum()
        logger.info(
            "Active days — S1: %d | S1C: %d | S1RF: %d | S2: %d | S3: %d | S4: %d | Combined: %d",
            n_s1, n_s1c, n_s1rf, n_s2, n_s3, n_s4, n_cb,
        )
        return df

    def save(self, df: Optional[pd.DataFrame] = None, path: Optional[Path] = None) -> Path:
        """Save signals DataFrame to parquet (default: default_signals_path(df),
        named after the date span actually saved)."""
        df = df if df is not None else self._result
        if df is None:
            raise RuntimeError("No results to save. Call generate() first.")
        path = Path(path or default_signals_path(df))
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
        Panel 2: Signal 1 (PDV–IV spread) position
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
            ("s1_position", "s1_strength", "#2196F3", "Signal 1 — PDV–IV Spread"),
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

    for prefix, label in [
        ("s1",   "Signal1_PDV_IV"),
        ("s1c",  "Signal1C_Contrarian"),
        ("s1rf", "Signal1_RF"),
        ("s2",   "Signal2_TS"),
        ("s3",   "Signal3_Disp"),
        ("s4",   "Signal4_VRP"),
    ]:
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
