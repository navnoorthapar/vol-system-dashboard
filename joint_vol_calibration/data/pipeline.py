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
  - every download is checked against the NYSE session calendar, and rows
    dated on days the NYSE was closed are not stored in the session-keyed
    tables (logged);
  - the VIX index family comes from Cboe's official files, with Yahoo only
    for sessions Cboe has not posted; the two are cross-checked on every
    refresh (refresh_vix_index_family, database.compare_sources);
  - an upsert that no longer sees a stored session keeps it and says so.

Usage:
  from joint_vol_calibration.data.pipeline import DataPipeline
  pipe = DataPipeline()
  pipe.historical_backfill()          # first run
  pipe.daily_refresh()                 # daily cron
  pipe.validate_no_lookahead()        # before backtesting
  coverage = pipe.get_coverage()       # inspect what we have
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joint_vol_calibration.config import (
    BT_END,
    BT_START,
    HIST_END,
    HIST_START,
    VIX_INDEX_TENOR_DAYS,
    VVIX_TENOR_DAYS,
)
from joint_vol_calibration.data import cboe_downloader as cboe
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import yf_downloader as yf_dl

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


# ══════════════════════════════════════════════════════════════════════════════
# The VIX index family: Cboe primary, Yahoo provisional (C23)
# ══════════════════════════════════════════════════════════════════════════════

# Pts above which a Yahoo and a Cboe print of the same index on the same
# session are reported (the C22 cross-check threshold).
CROSS_SOURCE_TOL = 0.25


def refresh_vix_index_family(start: str, end: str, *, yahoo: bool = True, vix_daily: bool = True,
                             tol: float = CROSS_SOURCE_TOL) -> dict:
    """
    Store ^VIX9D/^VIX/^VIX3M/^VIX6M/^VVIX (vix_term_structure) and VIX OHLC
    (vix_daily) for sessions [start, end] with Cboe's official files as the
    primary source.

    Order: Yahoo's rows first, then Cboe's on top. A Yahoo row therefore only
    survives for a session Cboe's file does not have yet (Cboe usually posts
    the evening's close within hours; Yahoo the same evening); it is stored
    with source='yfinance' and replaced by the Cboe print on the first run
    after Cboe publishes it — database.insert_vix_term_structure and
    insert_vix_daily never let a non-Cboe row replace a stored Cboe one.
    Rows on days the NYSE was closed are dropped by the writers.
    ``vix_daily=False`` leaves the VIX OHLC table alone.

    Cross-source check: every session both sources print in the window is
    compared (database.compare_sources); rows differing by more than ``tol``
    are returned and logged. Whether a mismatch fails a run is the caller's
    decision.

    Returns
    -------
    dict with keys
      ok           False if Cboe's term-structure download failed (the Yahoo
                   rows, if any, were still stored as provisional)
      rows         {'vix_term_structure_yfinance', 'vix_term_structure_cboe',
                    'vix_daily_yfinance', 'vix_daily_cboe'} -> rows written
      provisional  [(date, ticker), ...] stored from Yahoo because Cboe's
                   file lacks them
      mismatches   DataFrame [date, ticker, cboe, yfinance, diff] (|diff| > tol)
      errors       list of download error strings
    """
    out: dict[str, Any] = {"ok": True, "rows": {}, "provisional": [], "errors": [],
                           "mismatches": pd.DataFrame(columns=["date", "ticker", "cboe", "yfinance", "diff"])}
    yts = yvd = pd.DataFrame()
    if yahoo:
        try:
            yts = yf_dl.download_vix_term_structure(start=start, end=end)
        except Exception as e:
            out["errors"].append(f"yahoo term structure: {e}")
            logger.warning("Yahoo VIX term structure failed: %s", e)
        try:
            if vix_daily:
                yvd = yf_dl.download_vix_index(start=start, end=end)
        except Exception as e:
            out["errors"].append(f"yahoo ^VIX: {e}")
            logger.warning("Yahoo ^VIX failed: %s", e)
    cts = cvd = pd.DataFrame()
    try:
        cts = cboe.download_vix_term_structure_history(start=start, end=end)
    except Exception as e:
        out["ok"] = False
        out["errors"].append(f"cboe term structure: {e}")
        logger.warning("Cboe VIX-family download failed (Yahoo rows stored as provisional): %s", e)
    try:
        if vix_daily:
            cvd = cboe.download_vix_history()
            cvd = cvd[(cvd["date"] >= start) & (cvd["date"] <= end)]
    except Exception as e:
        out["errors"].append(f"cboe VIX OHLC: {e}")
        logger.warning("Cboe VIX OHLC download failed: %s", e)

    def window(df):
        return df if df.empty else df[(df["date"] >= start) & (df["date"] <= end)]
    yts, yvd, cts, cvd = window(yts), window(yvd), window(cts), window(cvd)

    if not yts.empty and not cts.empty:
        out["mismatches"] = db.compare_sources(cts, yts, tol, keys=("date", "ticker"),
                                               labels=("cboe", "yfinance"))
    if not yts.empty:
        yk = db.drop_non_sessions(yts, "Yahoo term structure")
        ck = set() if cts.empty else set(zip(cts["date"], cts["ticker"]))
        out["provisional"] = sorted((d, t) for d, t in zip(yk["date"], yk["ticker"]) if (d, t) not in ck)

    for key, frame, ins in (("vix_term_structure_yfinance", yts, db.insert_vix_term_structure),
                            ("vix_daily_yfinance", yvd, db.insert_vix_daily),
                            ("vix_term_structure_cboe", cts, db.insert_vix_term_structure),
                            ("vix_daily_cboe", cvd, db.insert_vix_daily)):
        out["rows"][key] = ins(frame, warn_dropped=key.endswith("_cboe")) if not frame.empty else 0
    if out["provisional"]:
        logger.info("VIX family: %d Yahoo row(s) stored as provisional (not in Cboe's files yet): %s",
                    len(out["provisional"]),
                    ", ".join(f"{d} {t}" for d, t in out["provisional"][:10]))
    return out


def refresh_cboe_indices(symbols=None, start: str | None = None, end: str | None = None) -> dict:
    """Download and store the non-VIX-family Cboe indices (cboe_downloader.CBOE_INDEX_CATALOG).

    ``symbols`` defaults to cboe_downloader.DEFAULT_EXTRA_INDICES (VIX1D,
    VIX1Y, SKEW, COR1M, COR3M, DSPX); start defaults to config.HIST_START.
    One request per symbol. A failed symbol is logged and skipped. Returns
    {symbol: {rows, min_date, max_date}} plus 'errors'.
    """
    symbols = tuple(symbols or cboe.DEFAULT_EXTRA_INDICES)
    start = start or HIST_START
    out: dict = {"errors": []}
    for sym in symbols:
        try:
            df = cboe.download_cboe_indices([sym], start=start, end=end)
        except Exception as e:
            out["errors"].append(f"{sym}: {e}")
            logger.warning("Cboe %s history failed: %s", sym, e)
            continue
        n = db.insert_cboe_index(df) if not df.empty else 0
        stored = db.drop_non_sessions(df, sym)
        out[sym] = {"rows": n, "min_date": stored["date"].min() if n else None,
                    "max_date": stored["date"].max() if n else None}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# C23 scripted correction of the persisted VIX index family
# ══════════════════════════════════════════════════════════════════════════════

# |new - old| at or below this is a re-statement of the same 2-decimal print
# (Yahoo serves float32: 22.129999 for Cboe's 22.13), not a revision.
SAME_PRINT_TOL = 0.005

_CORRECTION_LOG_COLS = ["table", "date", "ticker", "old_close", "old_source", "new_close", "new_source",
                        "diff", "action", "in_backtest_window"]


def _correction_frame(stored: pd.DataFrame, ref: pd.DataFrame, keys: list, value_cols: list,
                      table: str, lo: str, hi: str) -> tuple:
    """(corrected rows, per-row log) for one table: Cboe's value wherever Cboe has the
    session, the stored row (tagged 'yfinance' if unrecorded) where it does not, Cboe
    rows the store lacks added, rows on non-session days removed."""
    ref = ref[(ref["date"] >= lo) & (ref["date"] <= hi)]
    ref = db.drop_non_sessions(ref, f"Cboe {table}")
    ref = ref.drop_duplicates(keys, keep="last")
    m = stored.merge(ref[keys + value_cols], on=keys, how="outer", suffixes=("_old", "_ref"),
                     indicator=True)
    sess = db.is_session_array(m["date"])
    act = pd.Series("unchanged", index=m.index, dtype=object)
    both = (m["_merge"] == "both").to_numpy()
    diff = (m["close_ref"] - m["close_old"]).to_numpy(float)
    # the largest change over every value column (open/high/low too for vix_daily)
    worst = np.nanmax(np.abs(np.column_stack([(m[f"{c}_ref"] - m[f"{c}_old"]).to_numpy(float)
                                              for c in value_cols])), axis=1, initial=0.0)
    old_src = db.canonical_source(m["source"]) if "source" in m.columns else pd.Series(None, index=m.index)
    act[both & (worst > SAME_PRINT_TOL)] = "replaced_by_cboe"
    act[both & (worst <= SAME_PRINT_TOL) & ((worst > 1e-12)
                                            | old_src.ne("cboe").to_numpy())] = "restated_as_cboe"
    act[(m["_merge"] == "right_only").to_numpy()] = "added_from_cboe"
    left = (m["_merge"] == "left_only").to_numpy()
    act[left] = "kept_not_in_cboe"
    act[~sess] = "removed_non_session"
    use_ref = both | (m["_merge"] == "right_only").to_numpy()
    out = m.copy()
    for c in value_cols:
        out[c] = np.where(use_ref, m[f"{c}_ref"], m[f"{c}_old"])
    out["source"] = np.where(use_ref, "cboe", old_src.fillna("yfinance").to_numpy())
    log = pd.DataFrame({
        "table": table, "date": m["date"], "ticker": m["ticker"],
        "old_close": m["close_old"], "old_source": old_src,
        "new_close": np.where(sess, out["close"], np.nan),
        "new_source": np.where(sess, out["source"], None),  # type: ignore[call-overload]  # object column
        "diff": diff, "action": act,
        "in_backtest_window": m["date"].between(BT_START, BT_END),
    })
    return out[sess].reset_index(drop=True), log


def correct_vix_index_family(cboe_term: pd.DataFrame | None = None,
                             cboe_vix: pd.DataFrame | None = None, *,
                             root=None, apply: bool = False, apply_db: bool = False,
                             log_dir=None, log_name: str = "c23_vix_index_family") -> dict:
    """
    The C23 correction of the persisted VIX index family: make Cboe's official
    files the source of every vix_term_structure and vix_daily row in the
    yearly partitions (data_store/market/), as refresh_vix_index_family does
    for new sessions.

    For each table, over the stored date span:
      replaced_by_cboe   the stored (Yahoo) value differs from Cboe's by more
                         than SAME_PRINT_TOL -> Cboe's value
      restated_as_cboe   the same 2-decimal print (Yahoo's float32 22.129999 vs
                         Cboe's 22.13), or unrecorded provenance -> Cboe's value
                         and source='cboe'
      added_from_cboe    a session Cboe prints and the store lacks (VVIX gaps)
      kept_not_in_cboe   the store has it, Cboe's file does not -> kept, an
                         unrecorded source becomes 'yfinance' (the only
                         pre-C22 writer of these tables was yf_downloader)
      removed_non_session  dated on a day the NYSE was closed
    ``cboe_term``/``cboe_vix`` default to a live download of the full files.

    apply      rewrite the partitions under ``root`` (default MARKET_DIR) with
               database.replace_market_partitions
    apply_db   also bring the database (DB_PATH) in line: non-session rows
               deleted, every corrected row upserted
    log_dir    write <log_dir>/<log_name>.csv (every row whose value changed
               by more than SAME_PRINT_TOL, added, removed or kept without
               Cboe) and <log_name>.json (counts per action, table and ticker;
               the backtest-window counts). The CLI never overwrites the
               committed C23 log: a re-run writes a dated <log_name>.

    Returns the summary dict (also what the JSON holds) with 'log' (the full
    per-row DataFrame) added.
    """
    root = Path(root) if root is not None else db.MARKET_DIR
    if cboe_term is None:
        cboe_term = cboe.download_vix_term_structure_history()
    if cboe_vix is None:
        cboe_vix = cboe.download_vix_history()
    ts_old = db.read_market_partitions("vix_term_structure", root)
    vd_old = db.read_market_partitions("vix_daily", root)
    if ts_old.empty or vd_old.empty:
        raise FileNotFoundError(f"no vix_term_structure/vix_daily partitions under {root}")

    lo, hi = ts_old["date"].min(), ts_old["date"].max()
    ts_ref = cboe_term[["date", "ticker", "close"]].copy()
    ts_new, ts_log = _correction_frame(ts_old, ts_ref, ["date", "ticker"], ["close"],
                                       "vix_term_structure", lo, hi)
    tenors = {**VIX_INDEX_TENOR_DAYS, "^VVIX": VVIX_TENOR_DAYS}
    ts_new["tenor_days"] = ts_new["ticker"].map(tenors).astype(int)
    ts_new["is_vvix"] = ts_new["ticker"].eq("^VVIX").astype(int)

    lo_d, hi_d = vd_old["date"].min(), vd_old["date"].max()
    vd_ref = cboe_vix[["date", "open", "high", "low", "close"]].copy()
    vd_new, vd_log = _correction_frame(vd_old.assign(ticker="^VIX"), vd_ref.assign(ticker="^VIX"),
                                       ["date", "ticker"], ["open", "high", "low", "close"],
                                       "vix_daily", lo_d, hi_d)
    vd_new = vd_new.drop(columns=["ticker"])

    log = pd.concat([ts_log, vd_log], ignore_index=True)
    summary: dict[str, Any] = {
        "generated_by": "joint_vol_calibration.data.pipeline.correct_vix_index_family",
        "same_print_tol": SAME_PRINT_TOL,
        "backtest_window": [BT_START, BT_END],
        "cboe_reference": {
            "term_structure_rows": len(cboe_term), "term_structure_max_date": str(cboe_term["date"].max()),
            "vix_rows": len(cboe_vix), "vix_max_date": str(cboe_vix["date"].max()),
        },
        "tables": {},
    }
    for table, tl, new, old in (("vix_term_structure", ts_log, ts_new, ts_old),
                                ("vix_daily", vd_log, vd_new, vd_old)):
        counts = tl["action"].value_counts().to_dict()
        in_bt = tl[tl["in_backtest_window"]]["action"].value_counts().to_dict()
        big = tl[(tl["action"] == "replaced_by_cboe") & (tl["diff"].abs() > CROSS_SOURCE_TOL)]
        summary["tables"][table] = {
            "rows_before": len(old), "rows_after": len(new),
            "actions": {k: int(v) for k, v in sorted(counts.items())},
            "actions_in_backtest_window": {k: int(v) for k, v in sorted(in_bt.items())},
            "replaced_over_cross_source_tol": len(big),
            "replaced_over_cross_source_tol_in_backtest_window": int(big["in_backtest_window"].sum()),
            "by_ticker": {t: {k: int(v) for k, v in g["action"].value_counts().sort_index().items()}
                          for t, g in tl.groupby("ticker")},
        }
    material = log[log["action"].isin(["replaced_by_cboe", "added_from_cboe", "removed_non_session",
                                       "kept_not_in_cboe"])]
    summary["material_rows"] = len(material)
    summary["material_rows_in_backtest_window"] = int(material["in_backtest_window"].sum())

    if apply:
        written = db.replace_market_partitions("vix_term_structure", ts_new, root)
        written += db.replace_market_partitions("vix_daily", vd_new, root)
        summary["partitions_rewritten"] = len(written)
    if apply_db:
        summary["db"] = _apply_vix_correction_to_db(ts_new, vd_new)
    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        material.sort_values(["table", "date", "ticker"])[_CORRECTION_LOG_COLS].to_csv(
            d / f"{log_name}.csv", index=False, float_format="%.6f")
        (d / f"{log_name}.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    summary["log"] = log
    return summary


def _apply_vix_correction_to_db(ts_new: pd.DataFrame, vd_new: pd.DataFrame) -> dict:
    """Delete non-session rows from the two tables in DB_PATH and upsert the corrected rows."""
    db.init_database()
    removed = {}
    with db._connection() as conn:
        for table in ("vix_term_structure", "vix_daily"):      # literal table names; dates are bound
            dates = [r[0] for r in conn.execute(f"SELECT DISTINCT date FROM {table}")]  # noqa: S608
            bad = [d for d, ok in zip(dates, db.is_session_array(dates)) if not ok]
            removed[table] = sum(conn.execute(f"DELETE FROM {table} WHERE date = ?", (d,)).rowcount  # noqa: S608
                                 for d in bad)
    n_ts = db.insert_vix_term_structure(ts_new, warn_dropped=False)
    n_vd = db.insert_vix_daily(vd_new, warn_dropped=False)
    return {"removed_non_session": removed, "upserted": {"vix_term_structure": n_ts, "vix_daily": n_vd}}


def resync_database_from_record(*, options: bool = True) -> dict:
    """
    Bring the database cache (DB_PATH) in line with the persisted record,
    offline: no download.

    1. delete every row dated on a day the NYSE was closed from the
       session-keyed tables (database.SESSION_KEYED_TABLES);
    2. upsert every market partition (database.MARKET_TABLES): the corrected
       Cboe rows replace the stored Yahoo ones (the writers' Cboe precedence
       lets a Cboe row replace any other), and provenance comes with them;
    3. with ``options``, re-import every committed chain
       (import_options_snapshots(overwrite=True)), which stores one source
       spelling ('yahoo' -> 'yfinance').

    This is what the C23 correction needs after the partitions were rewritten
    (correct_vix_index_family --apply). It CHANGES what every consumer of the
    database reads (labels, backtest, hedger, calibrator VIX leg), so it is a
    deliberate step, run once, followed by regenerating those artifacts.
    Returns counts, including rows still without a source per table.
    """
    db.init_database()
    removed = {}
    with db._connection() as conn:
        for table in db.SESSION_KEYED_TABLES:              # the schema's own table names; dates are bound
            dates = [r[0] for r in conn.execute(f"SELECT DISTINCT date FROM {table}")]  # noqa: S608
            bad = [d for d, ok in zip(dates, db.is_session_array(dates)) if not ok]
            removed[table] = sum(conn.execute(f"DELETE FROM {table} WHERE date = ?", (d,)).rowcount  # noqa: S608
                                 for d in bad)
    out: dict[str, Any] = {"removed_non_session": removed, "imported": db.import_market_partitions()}
    if options:
        out["options_rows_reimported"] = db.import_options_snapshots(overwrite=True)
    with db._connection() as conn:
        out["null_source_rows"] = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE source IS NULL").fetchone()[0]  # noqa: S608
            for t in ("vix_daily", "vix_term_structure", "options_snapshots")}
        out["option_sources"] = {r[0]: r[1] for r in conn.execute(
            "SELECT source, COUNT(*) FROM options_snapshots GROUP BY source")}
    logger.info("resync_database_from_record: %s", out)
    return out


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
        hist_end: str | None = HIST_END,
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
          5. VIX term-structure indices (Cboe's files; Yahoo where Cboe has
             no row), cross-checked
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

        # The VIX index family (vix_daily + vix_term_structure): Cboe's
        # official files primary, Yahoo for the sessions Cboe has not posted
        # yet, cross-checked (refresh_vix_index_family).
        vix_start = min(since_last("vix_daily"), since_last("vix_term_structure"))
        fam = refresh_vix_index_family(vix_start, end)
        results["vix_daily"] = max(fam["rows"]["vix_daily_yfinance"], fam["rows"]["vix_daily_cboe"])
        results["vix_term_structure"] = max(fam["rows"]["vix_term_structure_yfinance"],
                                            fam["rows"]["vix_term_structure_cboe"])
        results["vix_cross_source_mismatches"] = len(fam["mismatches"])

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

    def persist(self, since: str | None = None, market_root=None, options_root=None) -> list:
        """Write the parquet record: market partitions for years >= since, chains for sessions >= since."""
        paths = db.export_market_partitions(market_root, since=since)
        paths += db.export_options_snapshots(options_root, since=since)
        return paths

    # ── Look-ahead Validation ─────────────────────────────────────────────────

    def validate_no_lookahead(self, test_date: str | None = None) -> bool:
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
        """Store the full VIX term-structure history: Cboe's files primary, Yahoo
        for the rows Cboe lacks (refresh_vix_index_family, term structure only)."""
        fam = refresh_vix_index_family(self.hist_start, self.hist_end, vix_daily=False)
        return max(fam["rows"]["vix_term_structure_yfinance"], fam["rows"]["vix_term_structure_cboe"])

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

def run_backfill(hist_start: str = HIST_START, hist_end: str | None = HIST_END) -> dict:
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


# ── Command line ──────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    """``python -m joint_vol_calibration.data.pipeline <command>``

    correct-vix-index-family  the C23 correction (dry run unless --apply);
                              --apply-db also rewrites DB_PATH's two tables;
                              the log goes to --log-dir (default with --apply:
                              data_store/market/corrections)
    resync-db                 bring DB_PATH in line with the persisted record,
                              offline (resync_database_from_record)
    cboe-indices              download VIX1D/VIX1Y/SKEW/COR1M/COR3M/DSPX (or
                              --symbols) into the database; --persist also
                              writes data_store/market/cboe_index_daily/
    """
    ap = argparse.ArgumentParser(prog="python -m joint_vol_calibration.data.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correct-vix-index-family")
    c.add_argument("--apply", action="store_true")
    c.add_argument("--apply-db", action="store_true")
    c.add_argument("--log-dir", default=None)
    r = sub.add_parser("resync-db")
    r.add_argument("--no-options", action="store_true", help="leave options_snapshots alone")
    i = sub.add_parser("cboe-indices")
    i.add_argument("--symbols", nargs="*", default=None)
    i.add_argument("--start", default=None)
    i.add_argument("--persist", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "correct-vix-index-family":
        log_dir = args.log_dir or (db.MARKET_DIR / "corrections" if args.apply else None)
        log_name = "c23_vix_index_family"
        if args.log_dir is None and log_dir is not None and (Path(log_dir) / f"{log_name}.csv").exists():
            # The committed C23 log records the correction that was made; a later
            # re-run (a no-op, or a Cboe revision) gets its own dated log beside it.
            log_name = f"{log_name}_{db._utcnow().strftime('%Y%m%dT%H%M%SZ')}"
        res = correct_vix_index_family(apply=args.apply, apply_db=args.apply_db, log_dir=log_dir,
                                       log_name=log_name)
        res.pop("log")
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0
    if args.cmd == "resync-db":
        print(json.dumps(resync_database_from_record(options=not args.no_options), indent=2, sort_keys=True))
        return 0
    res = refresh_cboe_indices(args.symbols, start=args.start)
    if args.persist:
        res["persisted"] = [str(p) for p in db.export_market_partitions(tables=["cboe_index_daily"])]
    print(json.dumps(res, indent=2, sort_keys=True, default=str))
    return 1 if res["errors"] else 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
