"""
pipeline.py — Data orchestration layer.

This module ties together the Cboe downloader, the Yahoo Finance downloader,
and the database into a single coherent pipeline.

Operating modes:
  1. historical_backfill() — one-time historical load (run once at setup)
  2. daily_refresh()       — incremental update (run each trading day, after
                             the 16:15 ET data cutoff)
  3. restore_persisted() / persist() — load / write the parquet record of
     everything captured (data_store/market, data_store/options), so the
     SQLite file can be treated as a rebuildable cache

ZERO LOOK-AHEAD GUARANTEE:
  - All data is stored with its observation date: the NY market session the
    values belong to (database.quote_session / last_closed_session), never the
    wall-clock date of the machine that ran the download.
  - The database read functions (database.py) enforce as_of_date gating.
  - This pipeline only writes data that was observable on the session it is
    dated by. It never pre-dates entries.
  - validate_no_lookahead() runs automated checks before any backtest.

DATA VALIDATION (warnings, never silent drops):
  - every download is checked against the NYSE session calendar;
  - Yahoo's VIX-family series are cross-checked against Cboe's official index
    files (database.cross_check_vix);
  - an upsert that no longer sees a stored session keeps it and says so.

Usage:
  from joint_vol_calibration.data.pipeline import DataPipeline
  pipe = DataPipeline()
  pipe.historical_backfill()          # first run
  pipe.daily_refresh()                 # daily cron
  pipe.validate_no_lookahead()        # before backtesting
  coverage = pipe.get_coverage()       # inspect what we have
"""

import logging
from typing import Optional

import pandas as pd

from joint_vol_calibration.config import HIST_START, HIST_END
from joint_vol_calibration.data import cboe_downloader as cboe
from joint_vol_calibration.data import yf_downloader as yf_dl
from joint_vol_calibration.data import database as db

logger = logging.getLogger(__name__)

# Calendar days re-downloaded by daily_refresh before the last closed session:
# covers a long weekend plus a couple of sessions a source served late.
REFRESH_LOOKBACK_DAYS = 10


def _sessions_only(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop rows dated on days the NYSE was closed, saying which.

    Cboe's index files carry values on some exchange holidays (e.g. VIX on
    2026-01-19, MLK day); the store is keyed by NYSE session, so those rows are
    not ingested.
    """
    if df is None or df.empty:
        return df
    keep = df["date"].map(db.is_nyse_session).astype(bool)
    if not keep.all():
        logger.info("%s: not storing %d row(s) dated on NYSE holidays: %s", label,
                    int((~keep).sum()), ", ".join(df.loc[~keep, "date"].astype(str).head(10)))
    return df[keep]


class DataPipeline:
    """
    Orchestrates all data downloads and storage for the vol trading system.

    Attributes
    ----------
    hist_start : str — start of historical period ('YYYY-MM-DD')
    hist_end   : str — end of historical period ('YYYY-MM-DD'); None (the
                 default, config.HIST_END) means the last closed NYSE session
                 at construction time
    """

    def __init__(
        self,
        hist_start: str = HIST_START,
        hist_end: Optional[str] = HIST_END,
    ):
        self.hist_start = hist_start
        self.hist_end   = hist_end if hist_end is not None else db.last_closed_session()
        db.init_database()
        logger.info("DataPipeline initialised. DB at: %s", db.DB_PATH)

    # ── Full Historical Backfill ───────────────────────────────────────────────

    def historical_backfill(self) -> dict:
        """
        One-time historical data load.

        Downloads and stores:
          1. SPX OHLCV (Yahoo Finance)
          2. VIX daily history (Cboe CSV; Yahoo if Cboe fails)
          3. VX futures daily history (Cboe per-contract CSVs, monthly contracts)
          4. VX final settlements (each contract's expiry-day settle; Cboe VRO
             where no contract file exists)
          5. VIX term-structure indices (Yahoo), cross-checked against Cboe
          6. T-bill rates (^IRX)
          7. The current option chains, unless a session is still trading

        Options chains:
          Historical SPX/VIX options chains are NOT freely available.
          Only current chains are; they accumulate one session at a time.

        Returns
        -------
        dict : row counts per data source.
        """
        logger.info("=== Starting historical backfill [%s → %s] ===",
                    self.hist_start, self.hist_end)
        results = {}

        results["spx_ohlcv"] = self._load_spx_ohlcv()
        results["vix_daily"] = self._load_vix_daily()
        results["vix_futures_daily"], results["vix_futures_settlements"] = self._load_vix_futures()
        results["vix_term_structure"] = self._load_vix_term_structure()
        results["tbill_rates"] = self._load_tbill_rates()
        results["spx_options_snapshot"] = self._snapshot_options("SPX")
        results["vix_options_snapshot"] = self._snapshot_options("VIX")

        logger.info("=== Historical backfill complete. Rows: %s ===", results)
        return results

    # ── Daily Refresh ─────────────────────────────────────────────────────────

    def daily_refresh(self, persist: bool = False, allow_intraday_options: bool = False) -> dict:
        """
        Incremental daily update — run after the session's 16:15 ET data cutoff.

        Every download ends at the LAST CLOSED NYSE SESSION (inclusive) and
        starts REFRESH_LOOKBACK_DAYS calendar days earlier — or at the table's
        last stored session if that is earlier — so the session that just
        closed is always requested, a late or revised bar is picked up, and a
        database that missed runs is filled rather than left with a gap.
        Option chains are snapshotted for that session (Cboe first, Yahoo as
        fallback); each snapshot replaces any earlier chain for the session.

        Parameters
        ----------
        persist : also write the parquet record (see persist()).
        allow_intraday_options : snapshot chains even while a session is
            trading (they would be dated that session but hold intraday quotes).

        Returns
        -------
        dict : rows written per source.
        """
        end = db.last_closed_session()
        start = (pd.Timestamp(end) - pd.Timedelta(days=REFRESH_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        logger.info("=== Daily refresh: sessions %s .. %s ===", start, end)
        results = {}

        def since_last(table: str) -> str:
            # Yahoo serves any span in one request, so reach back to the last
            # stored session when that is earlier than the window: a database
            # that missed runs is filled instead of keeping a silent gap.
            last = db.get_latest_date(table)
            return last if last is not None and last < start else start

        # SPX OHLCV
        try:
            df_spx = yf_dl.download_spx_ohlcv(start=since_last("spx_ohlcv"), end=end)
            results["spx_ohlcv"] = db.insert_spx_ohlcv(df_spx) if not df_spx.empty else 0
            if not df_spx.empty:
                db.validate_sessions(df_spx["date"], df_spx["date"].min(), end, label="SPX download")
        except Exception as e:
            logger.error("SPX OHLCV refresh failed: %s", e)
            results["spx_ohlcv"] = 0

        # VIX daily — Yahoo for the same-evening row, then Cboe's official file
        # on top wherever it already has the session.
        results["vix_daily"] = self._refresh_vix_daily(since_last("vix_daily"), end)

        # VX futures — the monthly contracts' per-contract files (one request per
        # listed contract, whatever the span), from the window start or from the
        # last stored VX session if that is earlier, so a gap since the last
        # run is filled; then the final settlements of contracts that expired.
        vx_start = start
        last_vx = db.get_latest_date("vix_futures_daily")
        if last_vx is not None and last_vx < start:
            vx_start = last_vx
        try:
            vx = cboe.download_vix_futures_daily_history(start=vx_start, end=end)
            results["vix_futures_daily"] = db.insert_vix_futures_daily(vx) if not vx.empty else 0
        except Exception as e:
            logger.warning("VX futures refresh failed: %s", e)
            vx = pd.DataFrame()
            results["vix_futures_daily"] = 0
        try:
            final = cboe.download_vix_futures_settlements(start=vx_start, end=end, daily_history=vx)
            results["vix_futures_settlements"] = db.insert_vix_futures_settlements(final) if not final.empty else 0
        except Exception as e:
            logger.warning("VX final settlements unavailable: %s", e)
            results["vix_futures_settlements"] = 0

        # VIX term structure (Yahoo), cross-checked against Cboe's index files
        try:
            df_ts = yf_dl.download_vix_term_structure(start=since_last("vix_term_structure"), end=end)
            results["vix_term_structure"] = db.insert_vix_term_structure(df_ts) if not df_ts.empty else 0
        except Exception as e:
            logger.error("VIX term structure refresh failed: %s", e)
            results["vix_term_structure"] = 0
        self._cross_check_vix(start, end)

        # T-bill rates
        try:
            df_tb = yf_dl.download_tbill_rate(start=since_last("tbill_rates"), end=end)
            results["tbill_rates"] = db.insert_tbill_rates(df_tb) if not df_tb.empty else 0
        except Exception as e:
            logger.error("T-bill rate refresh failed: %s", e)
            results["tbill_rates"] = 0

        # Options snapshots — full chain, dated by the quote session
        results["spx_options"] = self._snapshot_options("SPX", allow_intraday=allow_intraday_options)
        results["vix_options"] = self._snapshot_options("VIX", allow_intraday=allow_intraday_options)

        if persist:
            results["persisted_files"] = len(self.persist(since=start))

        logger.info("=== Daily refresh complete: %s ===", results)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def restore_persisted(self, market_root=None, options_root=None) -> dict:
        """Load the parquet record into the database (run before downloading)."""
        counts = db.import_market_partitions(market_root)
        counts["options"] = db.import_options_snapshots(options_root)
        logger.info("Restored persisted data: %s", counts)
        return counts

    def persist(self, since: Optional[str] = None, market_root=None, options_root=None) -> list:
        """Write the parquet record: market partitions for years >= since, chains for sessions >= since."""
        paths = db.export_market_partitions(market_root, since=since)
        paths += db.export_options_snapshots(options_root, since=since)
        return paths

    # ── Look-ahead Validation ─────────────────────────────────────────────────

    def validate_no_lookahead(self, test_date: Optional[str] = None) -> bool:
        """
        Verify that the database enforces zero look-ahead bias.

        Tests three properties:
          1. get_spx_ohlcv(as_of_date=D) returns NO rows with date > D
          2. get_vix_daily(as_of_date=D) returns NO rows with date > D
          3. get_options_surface(as_of_date=D) returns NO options with
             snapshot_date > D or expiry <= D

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
        dict : see database.get_data_coverage
        """
        coverage = db.get_data_coverage()
        logger.info("Data coverage:")
        for table, info in coverage.items():
            if "min_date" not in info:
                continue
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
        """Download and store VIX daily history: Cboe's official file, Yahoo if Cboe fails."""
        df = pd.DataFrame()
        try:
            df = _sessions_only(cboe.download_vix_history(), "Cboe VIX history")
        except Exception as e:
            logger.warning("Cboe VIX history failed (%s); falling back to Yahoo Finance", e)
        try:
            if df.empty:
                df = yf_dl.download_vix_index(start=self.hist_start, end=self.hist_end)
            if df.empty:
                return 0
            df = df[(df["date"] >= self.hist_start) & (df["date"] <= self.hist_end)]
            return db.insert_vix_daily(df)
        except Exception as e:
            logger.error("Failed to load VIX daily: %s", e)
            return 0

    def _refresh_vix_daily(self, start: str, end: str) -> int:
        """Yahoo rows for the window, then Cboe's rows on top (Cboe is the official print)."""
        n = 0
        try:
            yv = yf_dl.download_vix_index(start=start, end=end)
            if not yv.empty:
                n = db.insert_vix_daily(yv)
        except Exception as e:
            logger.warning("Yahoo VIX refresh failed: %s", e)
        try:
            cv = cboe.download_vix_history()
            cv = _sessions_only(cv[(cv["date"] >= start) & (cv["date"] <= end)], "Cboe VIX")
            if not cv.empty:
                n = max(n, db.insert_vix_daily(cv, warn_dropped=False))
        except Exception as e:
            logger.warning("Cboe VIX refresh failed (Yahoo rows kept): %s", e)
        return n

    def _cross_check_vix(self, start: str, end: str) -> None:
        """Log any Yahoo-vs-Cboe disagreement in the VIX family over [start, end]."""
        try:
            ref = cboe.download_vix_term_structure_history(start=start, end=end)
            db.cross_check_vix(ref, start=start, end=end)
        except Exception as e:
            logger.warning("VIX cross-check against Cboe skipped: %s", e)

    def _load_vix_futures(self) -> tuple:
        """VX daily history (monthly contracts) and their final settlements. Returns (daily rows, settlements)."""
        n_daily = n_final = 0
        daily = pd.DataFrame()
        try:
            daily = cboe.download_vix_futures_daily_history(start=self.hist_start, end=self.hist_end)
            if daily.empty:
                logger.warning("VIX futures daily history returned empty")
            else:
                n_daily = db.insert_vix_futures_daily(daily)
        except Exception as e:
            logger.error("Failed to load VIX futures daily history: %s", e)
        try:
            final = cboe.download_vix_futures_settlements(start=self.hist_start, end=self.hist_end,
                                                          daily_history=daily)
            if final.empty:
                logger.warning("VIX futures settlements returned empty")
            else:
                n_final = db.insert_vix_futures_settlements(final)
        except Exception as e:
            logger.error("Failed to load VIX futures settlements: %s", e)
        return n_daily, n_final

    def _load_vix_term_structure(self) -> int:
        """Download and store full VIX term structure history, then cross-check it."""
        try:
            df = yf_dl.download_vix_term_structure(
                start=self.hist_start, end=self.hist_end
            )
            if df.empty:
                logger.warning("VIX term structure returned empty")
                return 0
            n = db.insert_vix_term_structure(df)
        except Exception as e:
            logger.error("Failed to load VIX term structure: %s", e)
            return 0
        self._cross_check_vix(self.hist_start, self.hist_end)
        return n

    def _load_tbill_rates(self) -> int:
        """Download and store 13-week T-bill rate history (^IRX)."""
        try:
            df = yf_dl.download_tbill_rate(start=self.hist_start, end=self.hist_end)
            if df.empty:
                logger.warning("T-bill rate download returned empty")
                return 0
            return db.insert_tbill_rates(df)
        except Exception as e:
            logger.error("Failed to load T-bill rates: %s", e)
            return 0

    def _snapshot_options(self, underlying: str, allow_intraday: bool = False) -> int:
        """Snapshot the current chain: Cboe (one request, all expiries, greeks), Yahoo as fallback.

        Skipped while a session is trading unless ``allow_intraday``: the chain
        would be dated that session but hold intraday quotes, and nothing
        guarantees a post-close pull replaces it.
        """
        if db.is_intraday() and not allow_intraday:
            logger.info("Skipping %s options snapshot: session %s is still trading (before %s ET)",
                        underlying, db.quote_session(), db.CLOSE_DATA_CUTOFF.strftime("%H:%M"))
            return 0
        df = pd.DataFrame()
        try:
            df = cboe.download_cboe_options_snapshot(underlying)
        except Exception as e:
            logger.warning("Cboe options snapshot failed for %s: %s", underlying, e)
        if df is None or df.empty:
            logger.info("Cboe options empty for %s, trying Yahoo Finance ...", underlying)
            try:
                df = (yf_dl.snapshot_spx_options() if underlying == "SPX"
                      else yf_dl.snapshot_vix_options())
            except Exception as e:
                logger.error("Yahoo options snapshot failed for %s: %s", underlying, e)
                df = pd.DataFrame()
        if df is None or df.empty:
            logger.warning("No options data available for %s", underlying)
            return 0
        try:
            return db.insert_options_snapshot(df, underlying)
        except Exception as e:
            logger.error("Options snapshot failed for %s: %s", underlying, e)
            return 0


# ── Convenience top-level functions ──────────────────────────────────────────

def run_backfill(hist_start: str = HIST_START, hist_end: Optional[str] = HIST_END) -> dict:
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
