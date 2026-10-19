"""
config.py — Global configuration for the Joint SPX/VIX Calibration System.

Paths, seeds, date ranges, tickers and hyperparameters that more than one
module needs. A constant here is a promise: every module that needs the value
imports it from here rather than restating it. (Not every constant has reached
every caller yet — modules that still hard-code a value this file defines are
listed in the C22 integration notes.)

Importing this module has no side effects: it creates no directories. Code
that writes a file creates the file's parent directory itself.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data_store"
DB_PATH    = DATA_DIR / "vol_system.db"
MC_CACHE_DIR = DATA_DIR / "mc_cache"   # pre-computed Monte Carlo results (gitignored)

# ── Data Date Range ──────────────────────────────────────────────────────────
HIST_START = "2010-01-04"   # first day of data we pull
HIST_END   = None           # inclusive upper bound; None = the last closed NYSE session

# The window every published backtest metric is computed over (inclusive).
BT_START = "2018-01-01"
BT_END   = "2025-03-24"

# ── Tickers (Yahoo Finance) ──────────────────────────────────────────────────
SPX_TICKER          = "^GSPC"   # S&P 500 index daily OHLCV
VIX_TICKER          = "^VIX"    # VIX index daily OHLC
TBILL_TICKER        = "^IRX"    # 13-week T-bill, bank-discount yield in percent
SPX_OPTIONS_TICKER  = "^SPX"    # Yahoo option chain for SPX (mixes SPX and SPXW roots)
VIX_OPTIONS_TICKER  = "^VIX"    # Yahoo option chain for VIX (VIX and VIXW roots)

# ── Heston Model Defaults ─────────────────────────────────────────────────────
HESTON_DEFAULTS = {
    "kappa": 2.0,    # mean-reversion speed
    "theta": 0.04,   # long-run variance (~20% vol)
    "sigma": 0.5,    # vol-of-vol
    "rho":  -0.7,    # spot-vol correlation (negative = skew)
    "v0":   0.04,    # initial variance
}
HESTON_BOUNDS = {
    "kappa": (0.1,  20.0),
    "theta": (1e-4,  1.0),
    "sigma": (1e-3,  2.0),
    "rho":   (-0.95, 0.0),
    "v0":    (1e-4,  1.0),
}

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_PATHS = 30_000
MC_STEPS_PER_YEAR = 252
MC_ANTITHETIC = True

# ── Joint Calibration Loss Weights ────────────────────────────────────────────
# Raw weights; JointCalibrator normalises the ACTIVE legs to sum to one
# (0.5/0.6 -> 0.455/0.545 with w3 = 0). Units differ by leg: the SPX term is
# an IV MSE in decimal vol, the VIX term an MSE of index levels in decimal
# vol, so the ratio is a judgement call, not an estimate.
JOINT_W1 = 0.5   # SPX implied-vol MSE weight
JOINT_W2 = 0.6   # VIX index leg (^VIX9D/^VIX/^VIX3M/^VIX6M vs heston_vix_term_index).
                 # Raised from 0.3 before C22 when the leg still fitted these spot
                 # indices with the futures mapping; kept at 0.6 for C22, and not
                 # re-tuned against any RMSE target.
# VIX options leg: off. Its pre-C22 RMSE (37.14 vol pts) came from a broken
# put pricer, spot-VIX vegas and a stale chain, so it said nothing about the
# model; the pricer is fixed (CIR transition law), but the DB holds VIX option
# quotes for too few sessions to make the leg part of the daily fit.
JOINT_W3 = 0.0   # was 0.2

# ── VIX term-structure indices ────────────────────────────────────────────────
# Cboe constant-maturity volatility indices. Each one is a SPOT-starting
# variance-swap rate: expected volatility from TODAY to its horizon, i.e.
# VIX_T^2 = (1/T) * integral_0^T E[v_s] ds. They are NOT VIX futures (a future
# expiring at T settles on the 30-day vol over [T, T+30d]). Horizons are in
# calendar days per Cboe's index methodology. Single source of truth for every
# module that maps a ticker to a tenor.
VIX_INDEX_TENOR_DAYS = {
    "^VIX9D": 9,
    "^VIX":   30,
    "^VIX3M": 93,
    "^VIX6M": 184,
}
# ^VVIX is the 30-day implied volatility of VIX (from VIX options); it is
# stored alongside the indices above with this horizon.
VVIX_TENOR_DAYS = 30

# ── Neural Network ────────────────────────────────────────────────────────────
NN_HIDDEN_DIM   = 256
NN_N_LAYERS     = 4
NN_LEARNING_RATE = 1e-3
NN_BATCH_SIZE   = 2048
NN_MAX_EPOCHS   = 200
NN_TRAIN_SAMPLES = 500_000   # MC samples for NN training
NN_VOL_ERROR_TARGET = 0.001  # 0.1 vol point in decimal (accuracy target quoted in nn_pricer)
