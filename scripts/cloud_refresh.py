#!/usr/bin/env python3
"""
Daily auto-refresh for GitHub Actions.

Runs every weekday after US market close:
  1. Download last 400 days of SPX/VIX/T-bill from yfinance -> SQLite DB
  2. Extend regime_labels.parquet with new trading days
  3. Download SPX options + recalibrate Heston -> save joint_cal_<session>.pkl
     (named by the closed SPX session the fit describes, not the run date)
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

# Calendar days of history fetched before relabelling. The R2 gate is a rolling
# 252-TRADING-day quantile, so this must comfortably exceed 252 business days.
LABEL_LOOKBACK_DAYS = 900
_ADAPTIVE_WINDOW = 252


def latest_session_date(spx_df) -> "str | None":
    """Date (YYYY-MM-DD) of the most recent CLOSED SPX session in ``spx_df``.

    This is what a calibration actually describes. It equals the run date only
    when the job fires before midnight UTC; a cron delayed past midnight (which
    GitHub does routinely) would otherwise mint a pickle named for a day that
    has not traded yet.
    """
    if spx_df is None or len(spx_df) == 0:
        return None
    return str(spx_df["date"].iloc[-1])[:10]


def calibration_path(session_date: str) -> pathlib.Path:
    """Canonical location of the joint calibration for one closed session."""
    return DATA / "calibrations" / f"joint_cal_{session_date}.pkl"


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
    """Extend regime_labels.parquet with DETERMINISTIC RULE labels, and repair
    any stored label that disagrees with the rule. Returns rows added.

    This used to append ``clf.predict(...)`` — the XGBoost classifier's output.
    That is the model C16/C17 demoted: it scores 63.4% out-of-sample and loses
    to a 90.0% "predict-yesterday" baseline, so it is research-only and is not
    supposed to drive anything. But dashboard/app.py headlines the LAST ROW of
    this parquet as "the deterministic rule label, the label the backtest
    trades on", so every day the bot appended a prediction, the front page
    published the demoted model under a caption saying it was the rule.

    Two details matter for reproducing the historical labels exactly:
      * the R2 gate is ADAPTIVE (rolling 252-day 80th percentile of VVIX),
        i.e. ``vvix_threshold=None``. The fixed 100.0 default disagrees with
        the shipped history on 19.7% of days; adaptive reproduces all 4,125
        of them exactly.
      * that rolling gate therefore needs >= 252 trading days of VVIX history
        before it is meaningful, so the fetch window is ~900 calendar days
        rather than the 60 days the feature builder alone would need.
    """
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import (
        build_features, build_regime_labels, FEATURE_COLS,
    )

    rl_path = DATA / "signals" / "regime_labels.parquet"
    rl      = pd.read_parquet(rl_path)
    last_dt = rl.index.max()
    log.info("[2/4] Regime labels last date: %s -- extending to %s", last_dt.date(), TODAY)

    # Long enough that the trailing dates get a FULL 252-day adaptive window.
    lookback_start = str((last_dt - timedelta(days=LABEL_LOOKBACK_DAYS)).date())
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

    labels = build_regime_labels(spx, vix, vvix_threshold=None)
    if labels.empty:
        log.warning("  Rule labels could not be computed; skipping")
        return 0

    # Only trust dates whose adaptive gate saw a full 252-day window inside
    # this fetch; earlier ones would be computed from a truncated history.
    if len(labels) <= _ADAPTIVE_WINDOW:
        log.warning("  Only %d labelled days fetched (< %d needed for the adaptive "
                    "gate); skipping to avoid rewriting history from a short window",
                    len(labels), _ADAPTIVE_WINDOW)
        return 0
    safe_from = labels.index[_ADAPTIVE_WINDOW]

    new_idx = feats.index[(feats.index > last_dt) & feats.index.isin(labels.index)]
    n = 0
    if len(new_idx):
        new_rows = feats.loc[new_idx, FEATURE_COLS].copy()
        new_rows["regime"] = labels.reindex(new_idx).astype(int)
        rl = pd.concat([rl, new_rows])
        rl = rl[~rl.index.duplicated(keep="last")].sort_index()
        n = len(new_rows)
    else:
        log.info("  No new dates to label; regime labels already current")

    # Self-heal: overwrite any stored label in the trustworthy window that the
    # rule disagrees with. This is what repairs the rows the classifier wrote.
    repair_idx = rl.index[(rl.index >= safe_from) & rl.index.isin(labels.index)]
    if len(repair_idx):
        stored = rl.loc[repair_idx, "regime"].astype(int)
        truth  = labels.reindex(repair_idx).astype(int)
        differ = stored != truth
        if differ.any():
            rl.loc[repair_idx[differ], "regime"] = truth[differ]
            log.info("  Repaired %d stored label(s) that disagreed with the rule "
                     "(%s → %s)", int(differ.sum()),
                     repair_idx[differ][0].date(), repair_idx[differ][-1].date())

    rl.to_parquet(rl_path)
    log.info("  Added %d new rows -> regime_labels now at %s", n, rl.index.max().date())
    return n


# -- Step 3: Heston recalibration ----------------------------------------------

def recalibrate_heston() -> bool:
    """Download SPX options via yfinance + run JointCalibrator. Returns True on success."""
    import yfinance as yf
    import sqlite3
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
    from joint_vol_calibration.data.database import (
        get_spx_ohlcv, get_tbill_rate, insert_options_snapshot,
    )

    # Name the fit by the session it describes, not by the run date: exact
    # provenance, holiday runs skip naturally, and a delayed cron cannot write a
    # second fit for one close. (Pickles dated before 2026-09-08 carry the run
    # date — see scripts/backfill_calibration_provenance.py.)
    spot_date = latest_session_date(get_spx_ohlcv(as_of_date=TODAY))
    if spot_date is None:
        log.warning("  No SPX sessions in DB; skipping recalibration")
        return False
    log.info("[3/4] Downloading SPX options for session %s (run date %s)", spot_date, TODAY)

    out = calibration_path(spot_date)
    if out.exists():
        log.info("  Calibration for session %s already exists; skipping", spot_date)
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

        # This used to write a 5-column frame to a table called `spx_options`.
        # Nothing reads that table: JointCalibrator reads `options_snapshots`
        # via db.get_options_surface, which returns the most recent snapshot
        # <= as_of_date. So every "daily recalibration" was in fact refitting
        # the newest COMMITTED snapshot (2026-06-20) with today's spot and
        # today's time-to-expiry -- pricing a 14-day expiry as a 90-day one
        # against a June quote. Map the chain onto the real schema instead.
        opts["snapshot_date"]  = spot_date
        opts["expiry"]         = pd.to_datetime(opts["expiration"]).dt.strftime("%Y-%m-%d")
        opts["right"]          = opts["option_type"].str[0].str.upper()   # call/put -> C/P
        opts["implied_vol"]    = opts["impliedVolatility"]
        _bid = pd.to_numeric(opts.get("bid"), errors="coerce")
        _ask = pd.to_numeric(opts.get("ask"), errors="coerce")
        _last = pd.to_numeric(opts.get("lastPrice"), errors="coerce")
        _mid = (_bid + _ask) / 2.0
        opts["mid_price"]      = _mid.where((_bid > 0) & (_ask > 0), _last)
        opts["time_to_expiry"] = (
            (pd.to_datetime(opts["expiry"]) - pd.Timestamp(spot_date)).dt.days / 365.0
        )
        opts = opts[(opts["mid_price"] > 0) & (opts["time_to_expiry"] > 0)]
        opts = opts.dropna(subset=["strike", "mid_price", "implied_vol", "right"])

        if opts.empty:
            log.warning("  No usable option rows after cleaning; skipping calibration")
            return False

        # insert_options_snapshot uses a plain INSERT, so clear any partial
        # write for this session first (re-runs must not duplicate the chain).
        con = sqlite3.connect(str(DB))
        con.execute(
            "DELETE FROM options_snapshots WHERE underlying='SPX' AND snapshot_date=?",
            (spot_date,),
        )
        con.commit()
        con.close()

        n_opt = insert_options_snapshot(opts, "SPX")
        log.info("  Saved %d option contracts as the %s SPX snapshot", n_opt, spot_date)

    except Exception as e:
        log.error("  Options download failed: %s; skipping calibration", e)
        return False

    log.info("  Running JointCalibrator...")
    try:
        from joint_vol_calibration.calibration.joint_calibrator import is_acceptable_calibration

        cal = JointCalibrator(as_of_date=TODAY)
        result = cal.calibrate()

        spx_df = get_spx_ohlcv(as_of_date=TODAY)
        result["as_of_date"] = spot_date   # the session the fit describes
        result["run_date"]   = TODAY
        result["S"] = float(spx_df["close"].iloc[-1]) if not spx_df.empty else 0.0
        result["r"] = get_tbill_rate(TODAY)

        p = result.get("params", {})
        l = result.get("leg_losses", {})
        spx_rmse = l.get("spx_iv_rmse", None)

        # Quality gate: a thin/noisy live snapshot can drive Heston into a
        # degenerate corner (sigma->0, rho->0). Do NOT let that overwrite the
        # showcased calibration — the dashboard then keeps the last good fit.
        ok, reason = is_acceptable_calibration(
            p, spx_rmse, result.get("n_vix_tenors")
        )
        # A fit is only "today's" if its SPX leg really is today's chain.
        _snap = result.get("spx_snapshot_date")
        if ok and _snap != spot_date:
            ok, reason = False, (
                f"SPX surface is stale (snapshot {_snap}, session {spot_date}) — "
                "refusing to publish a fit built on an old chain"
            )
        if not ok:
            log.warning(
                "  Calibration REJECTED (%s): kappa=%.4f sigma=%.4f rho=%.4f SPX RMSE=%.3f vp "
                "-- keeping previous good calibration, not writing today's pkl",
                reason, p.get("kappa", 0), p.get("sigma", 0), p.get("rho", 0),
                spx_rmse if spx_rmse is not None else -1,
            )
            return False

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(result, f)

        log.info(
            "  Done (accepted): kappa=%.4f sigma=%.4f rho=%.4f | SPX RMSE=%.3f vp | "
            "VIX RMSE=%.3f | %s SPX opts, %s VIX tenors | Feller %s | session %s",
            p.get("kappa", 0), p.get("sigma", 0), p.get("rho", 0),
            l.get("spx_iv_rmse", 0), l.get("vix_futures_rmse", 0),
            result.get("n_spx_options", "?"), result.get("n_vix_tenors", "?"),
            result.get("feller_state", "?"), spot_date,
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
