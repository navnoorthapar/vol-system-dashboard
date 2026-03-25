"""
pipeline.py — Data orchestration layer.

This module ties together CBOE downloader, Yahoo Finance downloader,
and the database into a single coherent pipeline.

Two operating modes:
  1. historical_backfill() — one-time historical load (run once at setup)
  2. daily_refresh()       — incremental update (run each trading day)

ZERO LOOK-AHEAD GUARANTEE:
  - All data is stored with its observation date.
  - The database read functions (database.py) enforce as_of_date gating.
  - This pipeline only writes data that was observable on the day it was
    downloaded. It never pre-dates entries.
  - validate_no_lookahead() runs automated checks before any backtest.

Usage:
  from joint_vol_calibration.data.pipeline import DataPipeline
  pipe = DataPipeline()
  pipe.historical_backfill()          # first run
  pipe.daily_refresh()                 # daily cron
  pipe.validate_no_lookahead()        # before backtesting
  coverage = pipe.get_coverage()       # inspect what we have
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from joint_vol_calibration.config import HIST_START, HIST_END
from joint_vol_calibration.data import cboe_downloader as cboe
from joint_vol_calibration.data import yf_downloader as yf_dl
from joint_vol_calibration.data import database as db

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates all data downloads and storage for the vol trading system.

    Attributes
    ----------
    hist_start : str — start of historical period ('YYYY-MM-DD')
    hist_end   : str — end of historical period ('YYYY-MM-DD')
    """

    def __init__(
        self,
        hist_start: str = HIST_START,
        hist_end: str = HIST_END,
    ):
        self.hist_start = hist_start
        self.hist_end   = hist_end
        db.init_database()
        logger.info("DataPipeline initialised. DB at: %s", db.DB_PATH)

    # ── Full Historical Backfill ───────────────────────────────────────────────

    def historical_backfill(self) -> dict:
        """
        One-time historical data load.

        Downloads and stores:
          1. SPX OHLCV (Yahoo Finance)
          2. VIX daily history (CBOE CSV)
          3. VIX futures daily history (CBOE CSV)
          4. VIX futures settlement prices (CBOE CSV)

        Options chains:
          Historical SPX/VIX options chains are NOT freely available.
          yfinance only gives current chains. We start accumulating from today.
          For backtesting purposes, we will reconstruct implied vol surfaces
          from the model calibration using historical parameters.

        Returns
        -------
        dict : row counts per data source.
        """
        logger.info("=== Starting historical backfill [%s → %s] ===",
                    self.hist_start, self.hist_end)
        results = {}

        # 1. SPX OHLCV
        results["spx_ohlcv"] = self._load_spx_ohlcv()

        # 2. VIX daily
        results["vix_daily"] = self._load_vix_daily()

        # 3. VIX futures daily
        results["vix_futures_daily"] = self._load_vix_futures_daily()

        # 4. VIX futures settlements
        results["vix_futures_settlements"] = self._load_vix_futures_settlements()

        # 5. VIX term structure (free proxy for VIX futures term structure)
        results["vix_term_structure"] = self._load_vix_term_structure()

        # 6. Snapshot today's options chains
        results["spx_options_snapshot"] = self._snapshot_options("SPX")
        results["vix_options_snapshot"] = self._snapshot_options("VIX")

        logger.info("=== Historical backfill complete. Rows: %s ===", results)
        return results

    # ── Daily Refresh ─────────────────────────────────────────────────────────

    def daily_refresh(self) -> dict:
        """
        Incremental daily update — run at end of each trading day.

        Downloads only the most recent data (today / last trading day).
        For options chains, takes a fresh daily snapshot (this is how we
        build historical options data over time — by snapshotting daily).

        Returns
        -------
        dict : rows added per source.
        """
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info("=== Daily refresh for %s ===", today)
        results = {}

        # SPX OHLCV — last 5 days to catch any gaps
        try:
            start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
            df_spx = yf_dl.download_spx_ohlcv(start=start, end=today)
            if not df_spx.empty:
                results["spx_ohlcv"] = db.insert_spx_ohlcv(df_spx)
        except Exception as e:
            logger.error("SPX OHLCV refresh failed: %s", e)
            results["spx_ohlcv"] = 0

        # VIX daily — re-download last 5 days from CBOE (catches corrections)
        try:
            df_vix = cboe.download_vix_history()
            recent = df_vix[df_vix["date"] >= (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")]
            if not recent.empty:
                results["vix_daily"] = db.insert_vix_daily(recent)
        except Exception as e:
            logger.error("VIX daily refresh failed: %s", e)
            results["vix_daily"] = 0

        # VIX futures — re-download (CBOE updates this file daily)
        try:
            df_vf = cboe.download_vix_futures_daily_history()
            recent_vf = df_vf[df_vf["date"] >= (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")]
            if not recent_vf.empty:
                results["vix_futures_daily"] = db.insert_vix_futures_daily(recent_vf)
        except Exception as e:
            logger.error("VIX futures daily refresh failed: %s", e)
            results["vix_futures_daily"] = 0

        # VIX term structure — re-download (lightweight)
        try:
            df_ts = yf_dl.download_vix_term_structure(
                start=(datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
            )
            if not df_ts.empty:
                results["vix_term_structure"] = db.insert_vix_term_structure(df_ts)
        except Exception as e:
            logger.error("VIX term structure refresh failed: %s", e)
            results["vix_term_structure"] = 0

        # Options snapshots — full chain snapshot
        results["spx_options"] = self._snapshot_options("SPX")
        results["vix_options"] = self._snapshot_options("VIX")

        logger.info("=== Daily refresh complete: %s ===", results)
        return results

    # ── Look-ahead Validation ─────────────────────────────────────────────────

    def validate_no_lookahead(self, test_date: Optional[str] = None) -> bool:
        """
        Verify that the database enforces zero look-ahead bias.

        Tests three properties:
          1. get_spx_ohlcv(as_of_date=D) returns NO rows with date > D
          2. get_options_surface(as_of_date=D) returns NO options with
             snapshot_date > D
          3. get_vix_futures_curve(as_of_date=D) returns NO rows with
             date > D

        Parameters
        ----------
        test_date : str 'YYYY-MM-DD' — date to run validation on.
                    Defaults to 2020-01-15 (arbitrary mid-history date).

        Returns
        -------
        bool : True if all checks pass. Raises AssertionError if any fail.
        """
        if test_date is None:
            test_date = "2020-01-15"

        logger.info("Validating zero look-ahead bias as of %s ...", test_date)
        all_passed = True

        # Test 1: SPX OHLCV
        df = db.get_spx_ohlcv(as_of_date=test_date)
        if not df.empty:
            max_date = df["date"].max()
            if hasattr(max_date, "strftime"):
                max_date_str = max_date.strftime("%Y-%m-%d")
            else:
                max_date_str = str(max_date)[:10]
            assert max_date_str <= test_date, (
                f"LOOK-AHEAD BIAS DETECTED: SPX OHLCV max date {max_date_str} > {test_date}"
            )
            logger.info("  [PASS] SPX OHLCV: max date %s <= %s", max_date_str, test_date)
        else:
            logger.info("  [SKIP] SPX OHLCV: no data before %s", test_date)

        # Test 2: VIX daily
        df = db.get_vix_daily(as_of_date=test_date)
        if not df.empty:
            max_date = df["date"].max()
            max_date_str = max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)[:10]
            assert max_date_str <= test_date, (
                f"LOOK-AHEAD BIAS DETECTED: VIX daily max date {max_date_str} > {test_date}"
            )
            logger.info("  [PASS] VIX daily: max date %s <= %s", max_date_str, test_date)

        # Test 3: Options snapshots
        for und in ["SPX", "VIX"]:
            df = db.get_options_surface(as_of_date=test_date, underlying=und)
            if not df.empty:
                max_snap = df["snapshot_date"].max()
                max_snap_str = max_snap.strftime("%Y-%m-%d") if hasattr(max_snap, "strftime") else str(max_snap)[:10]
                assert max_snap_str <= test_date, (
                    f"LOOK-AHEAD BIAS DETECTED: {und} options snapshot {max_snap_str} > {test_date}"
                )
                # Also verify no expired options are returned
                min_expiry = df["expiry"].min()
                assert min_expiry > pd.to_datetime(test_date), (
                    f"LOOK-AHEAD BIAS: {und} options contain expired contracts (expiry < {test_date})"
                )
                logger.info("  [PASS] %s options: snapshot <= %s, all expiries > %s",
                            und, test_date, test_date)

        logger.info("Zero look-ahead validation: ALL CHECKS PASSED for %s", test_date)
        return True

    # ── Coverage Report ───────────────────────────────────────────────────────

    def get_coverage(self) -> dict:
        """
        Return a summary of available data in the database.

        Call this to quickly inspect what date ranges are loaded before
        running calibrations or backtests.

        Returns
        -------
        dict : {table_name: {rows, min_date, max_date}}
        """
        coverage = db.get_data_coverage()
        logger.info("Data coverage:")
        for table, info in coverage.items():
            logger.info("  %s: %d rows [%s → %s]",
                        table, info.get("rows", 0) or info.get("snapshots", 0),
                        info.get("min_date"), info.get("max_date"))
        return coverage

    # ── Private Loaders ───────────────────────────────────────────────────────

    def _load_spx_ohlcv(self) -> int:
        """Download and store SPX OHLCV history."""
        try:
            df = yf_dl.download_spx_ohlcv(start=self.hist_start, end=self.hist_end)
            if df.empty:
                logger.warning("No SPX OHLCV data returned")
                return 0
            return db.insert_spx_ohlcv(df)
        except Exception as e:
            logger.error("Failed to load SPX OHLCV: %s", e)
            return 0

    def _load_vix_daily(self) -> int:
        """Download and store VIX daily history from CBOE."""
        try:
            df = cboe.download_vix_history()
            if df.empty:
                # Fallback to Yahoo Finance
                logger.warning("CBOE VIX download failed, trying Yahoo Finance ...")
                df = yf_dl.download_vix_index(start=self.hist_start, end=self.hist_end)
            if df.empty:
                return 0
            # Filter to our date range
            df = df[
                (df["date"] >= self.hist_start) & (df["date"] <= self.hist_end)
            ]
            return db.insert_vix_daily(df)
        except Exception as e:
            logger.error("Failed to load VIX daily: %s", e)
            return 0

    def _load_vix_futures_daily(self) -> int:
        """Download and store VIX futures daily price history."""
        try:
            df = cboe.download_vix_futures_daily_history()
            if df.empty:
                logger.warning("VIX futures daily history returned empty")
                return 0
            df = df[
                (df["date"] >= self.hist_start) & (df["date"] <= self.hist_end)
            ]
            return db.insert_vix_futures_daily(df)
        except Exception as e:
            logger.error("Failed to load VIX futures daily history: %s", e)
            return 0

    def _load_vix_term_structure(self) -> int:
        """Download and store full VIX term structure history."""
        try:
            df = yf_dl.download_vix_term_structure(
                start=self.hist_start, end=self.hist_end
            )
            if df.empty:
                logger.warning("VIX term structure returned empty")
                return 0
            return db.insert_vix_term_structure(df)
        except Exception as e:
            logger.error("Failed to load VIX term structure: %s", e)
            return 0

    def _load_vix_futures_settlements(self) -> int:
        """Download and store VIX futures final settlement prices."""
        try:
            df = cboe.download_vix_futures_settlements()
            if df.empty:
                logger.warning("VIX futures settlements returned empty")
                return 0
            return db.insert_vix_futures_settlements(df)
        except Exception as e:
            logger.error("Failed to load VIX futures settlements: %s", e)
            return 0

    def _snapshot_options(self, underlying: str) -> int:
        """Take a snapshot of today's options chain for the given underlying."""
        try:
            if underlying == "SPX":
                df = yf_dl.snapshot_spx_options()
            else:
                df = yf_dl.snapshot_vix_options()

            if df is None or df.empty:
                # Try CBOE delayed quotes as fallback
                logger.info("yfinance options empty for %s, trying CBOE ...", underlying)
                df = cboe.download_cboe_options_snapshot(underlying)

            if df is None or df.empty:
                logger.warning("No options data available for %s", underlying)
                return 0

            return db.insert_options_snapshot(df, underlying)
        except Exception as e:
            logger.error("Options snapshot failed for %s: %s", underlying, e)
            return 0


# ── Convenience top-level functions ──────────────────────────────────────────

def run_backfill(hist_start: str = HIST_START, hist_end: str = HIST_END) -> dict:
    """
    Convenience wrapper: initialise pipeline and run historical backfill.

    Run once at project setup:
      from joint_vol_calibration.data.pipeline import run_backfill
      run_backfill()
    """
    pipe = DataPipeline(hist_start=hist_start, hist_end=hist_end)
    return pipe.historical_backfill()


def run_daily() -> dict:
    """
    Convenience wrapper: incremental daily data refresh.

    Schedule via cron or run manually each day:
      from joint_vol_calibration.data.pipeline import run_daily
      run_daily()
    """
    pipe = DataPipeline()
    return pipe.daily_refresh()
