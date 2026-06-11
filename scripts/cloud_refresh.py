#!/usr/bin/env python3
"""
Daily auto-refresh for GitHub Actions.

Runs every weekday after US market close:
  1. Download last 400 days of SPX/VIX/T-bill from yfinance -> SQLite DB
  2. Extend regime_labels.parquet with new trading days
  3. Download SPX options + recalibrate Heston -> save joint_cal_YYYY-MM-DD.pkl
  4. Freeze Flask app to static HTML -> .site-build/

The workflow then commits updated data files and pushes .site-build/ to gh-pages.
"""
import sys, os, pathlib, logging, subprocess, pickle
from datetime import date, timedelta

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cloud_refresh")

import pandas as pd
import numpy as np

TODAY = date.today().isoformat()
DATA  = ROOT / "data_store"
DB    = DATA / "vol_system.db"


# -- Step 1: Market data -------------------------------------------------------

def refresh_market_data() -> bool:
    """Download last 400 calendar days into the SQLite DB. Returns True on success."""
    from joint_vol_calibration.data import yf_downloader as dl
    from joint_vol_calibration.data.database import (
        insert_spx_ohlcv, insert_vix_daily,
        insert_vix_term_structure, insert_tbill_rates,
    )

    start = (date.today() - timedelta(days=400)).isoformat()
    log.info("[1/4] Downloading market data from %s -> %s", start, TODAY)

    ok = True

    try:
        spx = dl.download_spx_ohlcv(start=start, end=TODAY)
        if not spx.empty:
            n = insert_spx_ohlcv(spx)
            log.info("  SPX OHLCV: %d rows", n)
        else:
            log.warning("  SPX OHLCV: empty download")
            ok = False
    except Exception as e:
        log.error("  SPX OHLCV failed: %s", e)
        ok = False

    try:
        vix = dl.download_vix_index(start=start, end=TODAY)
        if not vix.empty:
            n = insert_vix_daily(vix)
            log.info("  VIX daily: %d rows", n)
    except Exception as e:
        log.warning("  VIX daily failed: %s", e)

    try:
        ts = dl.download_vix_term_structure(start=start, end=TODAY)
        if not ts.empty:
            n = insert_vix_term_structure(ts)
            log.info("  VIX term structure: %d rows", n)
    except Exception as e:
        log.warning("  VIX term structure failed: %s", e)

    try:
        tbill = dl.download_tbill_rate(start=start, end=TODAY)
        if not tbill.empty:
            n = insert_tbill_rates(tbill)
            log.info("  T-bill rates: %d rows", n)
    except Exception as e:
        log.warning("  T-bill rates failed: %s", e)

    return ok


# -- Step 2: Regime labels -----------------------------------------------------

def extend_regime_labels() -> int:
    """Append predictions for dates missing from regime_labels.parquet. Returns new row count."""
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import build_features, FEATURE_COLS

    rl_path = DATA / "signals" / "regime_labels.parquet"
    rl      = pd.read_parquet(rl_path)
    last_dt = rl.index.max()
    log.info("[2/4] Regime labels last date: %s -- extending to %s", last_dt.date(), TODAY)

    with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
        clf = pickle.load(f)

    # Fetch enough history for rolling-window features (>=20 days lookback)
    lookback_start = str((last_dt - timedelta(days=60)).date())
    spx = get_spx_ohlcv(as_of_date=TODAY, start_date=lookback_start)
    vix = get_vix_term_structure_wide(as_of_date=TODAY, start_date=lookback_start)

    if spx.empty or vix.empty:
        log.warning("  Insufficient market data for feature computation; skipping")
        return 0

    feats = build_features(spx, vix)

    # Forward-fill slow-moving features across data gaps (VIX TS weekends, etc.)
    FFILL = ["fear_premium", "rv_change_5d", "ts_slope", "vvix", "pdv_iv_spread"]
    for col in FFILL:
        if col in feats.columns:
            feats[col] = feats[col].ffill()
    feats.dropna(inplace=True)

    new_feats = feats[feats.index > last_dt][FEATURE_COLS]
    if new_feats.empty:
        log.info("  No new dates to predict; regime labels already current")
        return 0

    # Pass the DataFrame, not .values: RegimeClassifier.predict() selects the
    # columns it was trained on via feature_names_ (works for both the legacy
    # 6-feature pickle and the C16 5-feature pickle that excludes vvix).
    regime_vals = clf.predict(new_feats)

    new_rows = new_feats.copy()
    new_rows["regime"] = regime_vals

    combined = pd.concat([rl, new_rows])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(rl_path)

    n = len(new_rows)
    log.info("  Added %d new rows -> regime_labels now at %s", n, combined.index.max().date())
    return n


# -- Step 3: Heston recalibration ----------------------------------------------

def recalibrate_heston() -> bool:
    """Download SPX options via yfinance + run JointCalibrator. Returns True on success."""
    import yfinance as yf
    import sqlite3
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_tbill_rate

    log.info("[3/4] Downloading SPX options for %s", TODAY)

    # Skip if already calibrated today
    existing = list((DATA / "calibrations").glob(f"joint_cal_{TODAY}.pkl"))
    if existing:
        log.info("  Calibration already exists for today; skipping")
        return True

    try:
        spx_ticker = yf.Ticker("^SPX")
        exps = spx_ticker.options
        if not exps:
            log.warning("  ^SPX returned no expirations; trying ^GSPC")
            spx_ticker = yf.Ticker("^GSPC")
            exps = spx_ticker.options

        exps = [e for e in (exps or [])[:8]]
        frames = []
        for exp in exps:
            try:
                chain = spx_ticker.option_chain(exp)
                for side, df in [("call", chain.calls), ("put", chain.puts)]:
                    df = df.copy()
                    df["expiration"]  = exp
                    df["option_type"] = side
                    frames.append(df)
            except Exception as ee:
                log.warning("  Expiry %s failed: %s", exp, ee)

        if not frames:
            log.warning("  No options data available; skipping recalibration")
            return False

        opts = pd.concat(frames, ignore_index=True)
        opts = opts[opts["impliedVolatility"] > 0.01].copy()
        opts["implied_vol"]   = opts["impliedVolatility"]
        opts["snapshot_date"] = TODAY

        con = sqlite3.connect(str(DB))
        opts[["strike", "expiration", "option_type", "implied_vol", "snapshot_date"]].to_sql(
            "spx_options", con, if_exists="replace", index=False
        )
        con.close()
        log.info("  Saved %d option contracts", len(opts))

    except Exception as e:
        log.error("  Options download failed: %s; skipping calibration", e)
        return False

    log.info("  Running JointCalibrator...")
    try:
        cal = JointCalibrator(as_of_date=TODAY)
        result = cal.calibrate()

        spx_df = get_spx_ohlcv(as_of_date=TODAY)
        result["as_of_date"] = TODAY
        result["S"] = float(spx_df["close"].iloc[-1]) if not spx_df.empty else 0.0
        result["r"] = get_tbill_rate(TODAY)

        out = DATA / "calibrations" / f"joint_cal_{TODAY}.pkl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(result, f)

        p = result.get("params", {})
        l = result.get("leg_losses", {})
        log.info(
            "  Done: kappa=%.4f sigma=%.4f rho=%.4f | SPX RMSE=%.3f vp | VIX RMSE=%.3f",
            p.get("kappa", 0), p.get("sigma", 0), p.get("rho", 0),
            l.get("spx_iv_rmse", 0), l.get("vix_futures_rmse", 0),
        )
        return True

    except Exception as e:
        log.error("  Calibration failed: %s", e)
        return False


# -- Step 4: Freeze static site ------------------------------------------------

def freeze_site() -> None:
    log.info("[4/4] Freezing static site -> .site-build/")
    result = subprocess.run(
        [sys.executable, "dashboard/freeze_site.py"],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        log.info("  %s", line)
    if result.returncode != 0:
        log.error("freeze_site.py exited %d:\n%s", result.returncode, result.stderr)
        sys.exit(1)


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Daily auto-refresh -- %s", TODAY)
    log.info("=" * 60)

    market_ok = refresh_market_data()
    if not market_ok:
        log.warning("Market data incomplete -- continuing with existing DB")

    extend_regime_labels()
    recalibrate_heston()
    freeze_site()

    log.info("=" * 60)
    log.info("Refresh complete. Site rebuilt at .site-build/")
    log.info("=" * 60)
