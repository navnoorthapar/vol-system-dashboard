"""
config.py — Global configuration for the Joint SPX/VIX Calibration System.

All paths, seeds, date ranges, and hyperparameters live here.
Changing one value here propagates everywhere — no magic numbers in model code.
"""

import os
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data_store"
DB_PATH    = DATA_DIR / "vol_system.db"
PARQUET_DIR = DATA_DIR / "parquet"
MC_CACHE_DIR = DATA_DIR / "mc_cache"   # pre-computed Monte Carlo results

for _dir in [DATA_DIR, PARQUET_DIR, MC_CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Data Date Range ──────────────────────────────────────────────────────────
HIST_START = "2010-01-04"   # first day of data we pull
HIST_END   = "2025-12-31"   # inclusive upper bound for historical work

# Backtest windows (walk-forward)
BACKTEST_START  = "2018-01-02"
BACKTEST_END    = "2025-12-31"
TRAIN_YEARS     = 2          # rolling training window
TEST_MONTHS     = 6          # out-of-sample test window

# ── Tickers ──────────────────────────────────────────────────────────────────
SPX_TICKER  = "^GSPC"
VIX_TICKER  = "^VIX"
SPXW_TICKER = "^SPX"        # yfinance options chain ticker for SPX

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
    "rho":   (-0.99, 0.0),
    "v0":    (1e-4,  1.0),
}

# ── Monte Carlo ───────────────────────────────────────────────────────────────
MC_PATHS = 30_000
MC_STEPS_PER_YEAR = 252
MC_ANTITHETIC = True

# ── Joint Calibration Loss Weights ────────────────────────────────────────────
JOINT_W1 = 0.5   # SPX implied vol MSE weight
JOINT_W2 = 0.3   # VIX futures price MSE weight
JOINT_W3 = 0.2   # VIX options implied vol MSE weight

# ── Neural Network ────────────────────────────────────────────────────────────
NN_HIDDEN_DIM   = 256
NN_N_LAYERS     = 4
NN_LEARNING_RATE = 1e-3
NN_BATCH_SIZE   = 2048
NN_MAX_EPOCHS   = 200
NN_TRAIN_SAMPLES = 500_000   # MC samples for NN training
NN_VOL_ERROR_TARGET = 0.001  # 0.1 vol point in decimal

# ── Risk Thresholds ───────────────────────────────────────────────────────────
SIGNAL_VOL_EDGE_THRESHOLD = 0.02   # 2 vol points to trigger gamma trade
SIGNAL_VIX_EDGE_THRESHOLD = 0.50   # $0.50 VIX futures mispricing threshold

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_DIR   = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
