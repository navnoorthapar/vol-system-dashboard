"""
database.py — SQLite schema, all read/write operations, the NYSE session
calendar the data layer dates everything by, validation hooks, and the
parquet persistence helpers the daily bot uses to keep what it captures.

Design philosophy:
- EVERY read function accepts an `as_of_date` parameter.
  Data returned is ALWAYS filtered to `date <= as_of_date`.
  This is the single enforcement point for zero look-ahead bias.
- Every date is a NEW YORK MARKET SESSION: the NYSE session the values belong
  to, never the wall-clock date of the machine that downloaded them. An option
  chain pulled on a Saturday holds Friday's quotes and is dated Friday
  (``quote_session``); a download's default end is the last session whose
  close has printed (``last_closed_session``).
- Writes are idempotent:
    * time series (spx_ohlcv, vix_daily, vix_term_structure, tbill_rates,
      vix_futures_*) are keyed upserts (INSERT OR REPLACE on the natural key).
      An upsert never deletes: a row the remote source later drops stays
      stored, and the writer logs a warning naming it. A vix_daily row from
      Cboe (the official print) is only replaced by another Cboe row.
    * an options snapshot REPLACES that session's chain for that underlying in
      one transaction (delete-then-insert), so a re-run never duplicates or
      mixes quote sets. A UNIQUE index on the natural key
      (snapshot_date, underlying, root, expiry, strike, right) backs this up.
- Every connection is closed when its operation ends, and the database is kept
  in rollback-journal (DELETE) mode: the file is git-tracked, and a WAL-mode
  file can be committed while its un-checkpointed pages sit in an ignored
  ``-wal`` sidecar.
- Validation hooks (``validate_sessions``, ``find_bad_prints``,
  ``cross_check_vix``, ``compare_sources``) LOG WARNINGS. They never drop
  or rewrite data. Two writer rules do (C23), each logged: a session-keyed
  table (SESSION_KEYED_TABLES) stores no row dated on a day the NYSE was
  closed, and in the VIX-index tables a stored Cboe value (the official
  print) is replaced only by another Cboe value.

Tables:
  spx_ohlcv               — SPX daily OHLCV (Yahoo ^GSPC); log_return derived
  vix_daily               — VIX index daily OHLC (Cboe or Yahoo; see `source`)
  vix_term_structure      — Cboe constant-maturity indices ^VIX9D/^VIX/^VIX3M/
                            ^VIX6M (spot-starting variance-swap rates, NOT VIX
                            futures) plus ^VVIX
  vix_futures_daily       — VX futures daily settlement/OHLC per contract,
                            keyed (date, expiry_date)
  vix_futures_settlements — VX final settlement values (monthly contracts:
                            expiry-day settle, Cboe VRO before mid-2013)
  options_snapshots       — SPX/VIX option chains keyed by quote session
  tbill_rates             — ^IRX 13-week bill: continuously compounded rate
                            plus the raw bank-discount yield
  cboe_index_daily        — other Cboe indices (VIX1D, VIX1Y, SKEW, COR1M,
                            COR3M, DSPX, strategy benchmarks), one row per
                            session and symbol (C23)
  rates_daily             — rate quotes as published: Treasury bills and par
                            curve, SOFR, H.15 3-month bill; the inputs of
                            data/rates.py's zero curve (C23)
  classifier_predictions  — XGBoost regime-classifier output (research only;
                            this is NOT the rule-label series the dashboard
                            and backtest use — that lives in
                            data_store/signals/regime_labels.parquet)

Persistence for the daily bot (the SQLite file is a rebuildable cache; the
parquet files are what a run should commit):
  export_market_partitions / import_market_partitions
      yearly partitions  data_store/market/<table>/<table>_<YYYY>.parquet
      for spx_ohlcv, vix_daily, vix_term_structure, tbill_rates (by
      default) and, on request, cboe_index_daily and rates_daily.
  export_options_snapshot(s) / import_options_snapshots
      one file per chain  data_store/options/<spx|vix>_<YYYY-MM-DD>.parquet
  Export MERGES into an existing partition (database wins on a key clash) and
  never removes a row the file already holds; import upserts. So a session the
  remote source later drops survives every later run. (VX futures are not
  partitioned: Cboe keeps every contract's full history online, and
  pipeline.daily_refresh refills them from the last stored session.)
"""

import logging
import math
import os
import re
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from datetime import time as dtime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil.relativedelta import MO
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

from joint_vol_calibration.config import (
    DATA_DIR,
    DB_PATH,
    VIX_INDEX_TENOR_DAYS,
    VVIX_TENOR_DAYS,
)

logger = logging.getLogger(__name__)

# Where the persistence helpers read and write. Module attributes (not
# defaults baked into signatures) so tests can redirect them.
MARKET_DIR = DATA_DIR / "market"
OPTIONS_DIR = DATA_DIR / "options"

# Deprecated. The write-only "parquet mirror" of options snapshots that lived
# here was never read by anything and diverged from SQLite; it was removed in
# C22 (the persisted chains now live in OPTIONS_DIR). The name survives only so
# existing test monkeypatches of ``database.PARQUET_DIR`` still resolve;
# nothing reads or writes it.
PARQUET_DIR = DATA_DIR / "parquet"


class StaleDataError(LookupError):
    """The newest stored data is older than the caller allows."""


# ══════════════════════════════════════════════════════════════════════════════
# NYSE session calendar
# ══════════════════════════════════════════════════════════════════════════════
#
# Rule-based, because no exchange-calendar package is a dependency. Full-day
# NYSE closures only (early 13:00 closes are ordinary sessions here). On
# 2026-10-12 it reproduced all 6,481 ^GSPC sessions Yahoo serves for
# 2001-01-02..2026-10-09 exactly; tests/test_data_layer.py re-checks it against
# the shipped SPX history. Extend _NYSE_SPECIAL_CLOSURES if the exchange closes
# for an unscheduled event.

NY_TZ = ZoneInfo("America/New_York")
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
# Index closes and option settlement marks are final a few minutes after the
# 16:00 bell (SPX options trade to 16:15). A session counts as "closed" for
# data purposes only from here on.
CLOSE_DATA_CUTOFF = dtime(16, 15)

_NYSE_SPECIAL_CLOSURES = (
    "2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14",   # September 11
    "2004-06-11",                                             # President Reagan
    "2007-01-02",                                             # President Ford
    "2012-10-29", "2012-10-30",                               # Hurricane Sandy
    "2018-12-05",                                             # President G.H.W. Bush
    "2025-01-09",                                             # President Carter
)


class _NYSEHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        # NYSE does not close on Friday Dec 31 when Jan 1 is a Saturday.
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        Holiday("Martin Luther King Jr. Day", month=1, day=1,
                offset=DateOffset(weekday=MO(3)), start_date="1998-01-01"),
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


_CAL_START, _CAL_END = "1990-01-01", "2045-12-31"


@lru_cache(maxsize=1)
def _holiday_index() -> pd.DatetimeIndex:
    regular = _NYSEHolidayCalendar().holidays(_CAL_START, _CAL_END)
    special = pd.DatetimeIndex(pd.to_datetime(list(_NYSE_SPECIAL_CLOSURES)))
    return regular.union(special)


@lru_cache(maxsize=1)
def _session_offset() -> CustomBusinessDay:
    return CustomBusinessDay(holidays=_holiday_index())


def _day(d) -> pd.Timestamp:
    """Normalise a date-like (str, date, Timestamp) to a naive midnight Timestamp."""
    ts = pd.Timestamp(d)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts.normalize()


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def nyse_holidays(start, end) -> list:
    """Full-day NYSE closures that fall on weekdays in [start, end] (YYYY-MM-DD strings)."""
    s, e = _day(start), _day(end)
    hol = _holiday_index()
    hol = hol[(hol >= s) & (hol <= e) & (hol.weekday < 5)]
    return [_iso(h) for h in hol]


def nyse_sessions(start, end) -> pd.DatetimeIndex:
    """NYSE trading sessions in [start, end], inclusive (naive midnight timestamps)."""
    return pd.date_range(_day(start), _day(end), freq=_session_offset())


def is_nyse_session(d) -> bool:
    """True if ``d`` (any date-like) is an NYSE trading session."""
    ts = _day(d)
    return ts.weekday() < 5 and ts not in _holiday_index()


def previous_session(d) -> str:
    """The last NYSE session strictly before ``d``."""
    ts = _day(d)
    if is_nyse_session(ts):
        return _iso(ts - _session_offset())
    # From a closed day, one session back is the last session before it.
    return _iso(_session_offset().rollback(ts))


def session_on_or_before(d) -> str:
    """``d`` itself if it is a session, else the last session before it."""
    ts = _day(d)
    return _iso(ts) if is_nyse_session(ts) else previous_session(ts)


def sessions_between(earlier, later) -> int:
    """Number of NYSE sessions in (earlier, later] — 0 when they are the same session."""
    a, b = _day(earlier), _day(later)
    if b <= a:
        return 0
    return len(nyse_sessions(a + pd.Timedelta(days=1), b))


def _utcnow() -> datetime:
    """Clock used by the session helpers (tests monkeypatch this)."""
    return datetime.now(UTC)


def _now_ny(now: datetime | None = None) -> datetime:
    """``now`` (default: the clock) in New York time. Naive datetimes are taken as UTC."""
    if now is None:
        now = _utcnow()
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(NY_TZ)


def last_closed_session(now: datetime | None = None,
                        cutoff: dtime = CLOSE_DATA_CUTOFF) -> str:
    """The most recent NYSE session whose close has printed, as of ``now``.

    A session day counts from ``cutoff`` (16:15 ET) onwards; before that — or
    on a weekend/holiday — the answer is the previous session. This is the
    right inclusive ``end`` for a daily download: it never asks for a bar that
    is still forming, and it does not depend on the machine's time zone.
    """
    ny = _now_ny(now)
    today = pd.Timestamp(ny.date())
    if is_nyse_session(today) and ny.time() >= cutoff:
        return _iso(today)
    return previous_session(today)


def quote_session(now: datetime | None = None) -> str:
    """The NYSE session whose quotes a live feed is showing at ``now``.

    From the 09:30 ET open of a session day it is that day (intraday quotes
    until 16:00, closing quotes after); before the open, and on weekends and
    holidays, it is the previous session. Snapshot dates come from here.
    """
    ny = _now_ny(now)
    today = pd.Timestamp(ny.date())
    if is_nyse_session(today) and ny.time() >= SESSION_OPEN:
        return _iso(today)
    return previous_session(today)


def utc_iso(when: datetime) -> str:
    """ISO-8601 UTC timestamp ('2026-10-09T20:15:00Z'); naive datetimes are taken as UTC."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_intraday(now: datetime | None = None) -> bool:
    """True between the 09:30 open and the data cutoff (16:15 ET) of a session day."""
    ny = _now_ny(now)
    return (is_nyse_session(pd.Timestamp(ny.date()))
            and SESSION_OPEN <= ny.time() < CLOSE_DATA_CUTOFF)


def is_session_array(dates) -> np.ndarray:
    """Vectorised ``is_nyse_session`` over date-likes (unparseable -> False)."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)), errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    idx = idx.normalize()
    return np.asarray(idx.notna() & (idx.weekday < 5) & ~idx.isin(_holiday_index()), dtype=bool)


def drop_non_sessions(df: pd.DataFrame, label: str = "series", *, col: str = "date") -> pd.DataFrame:
    """``df`` without the rows dated on days the NYSE was closed, saying which.

    The session-keyed tables (SESSION_KEYED_TABLES) store one row per NYSE
    session. Both sources print values on exchange holidays: Cboe's
    VIX_History.csv carries a VIX close on 33 NYSE holidays from 2022-05-30
    to 2026-09-07 (counted 2026-10-20; its VIX9D/VIX3M/VIX6M/VVIX files
    carry none since 2010), and Yahoo served ^VIX on 2026-05-25 and
    2026-09-07. Before C23 such rows were stored (29 in vix_daily, 2 in
    vix_term_structure). Returns ``df`` itself when nothing is dropped.
    """
    if df is None or df.empty:
        return df
    keep = pd.Series(is_session_array(df[col]), index=df.index)
    if keep.all():
        return df
    dropped = df.loc[~keep, col].astype(str)
    logger.info("%s: not storing %d row(s) dated on days the NYSE was closed: %s%s", label,
                int((~keep).sum()), ", ".join(sorted(set(dropped))[:10]),
                " ..." if dropped.nunique() > 10 else "")
    return df[keep]


def validate_sessions(dates: Iterable, start=None, end=None, *,
                      label: str = "series", log: bool = True) -> dict:
    """Compare ``dates`` with the NYSE calendar over [start, end].

    ``start``/``end`` default to the span of ``dates``. Returns
    ``{"missing": [...], "extra": [...]}``: sessions with no row, and rows on
    days the NYSE was closed. Logs one warning per non-empty list; never
    raises — the caller decides what a gap means.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)), errors="coerce")).dropna()
    if idx.tz is not None:
        idx = idx.tz_convert(NY_TZ).tz_localize(None)
    idx = idx.normalize().unique()
    if len(idx) == 0 and (start is None or end is None):
        return {"missing": [], "extra": []}
    lo = _day(start) if start is not None else idx.min()
    hi = _day(end) if end is not None else idx.max()
    sessions = nyse_sessions(lo, hi)
    window = idx[(idx >= lo) & (idx <= hi)]
    missing = [_iso(d) for d in sessions.difference(window)]
    extra = [_iso(d) for d in window.difference(sessions)]
    if log:
        if missing:
            logger.warning("%s: %d NYSE session(s) missing in %s..%s: %s%s", label,
                           len(missing), _iso(lo), _iso(hi), ", ".join(missing[:10]),
                           " ..." if len(missing) > 10 else "")
        if extra:
            logger.warning("%s: %d row(s) on non-session days: %s%s", label, len(extra),
                           ", ".join(extra[:10]), " ..." if len(extra) > 10 else "")
    return {"missing": missing, "extra": extra}


# ── VX futures expiry calendar ────────────────────────────────────────────────

def vx_monthly_expiry(year: int, month: int) -> str:
    """Final settlement date of the standard (monthly) VX future for a contract month.

    Cboe rule: the Wednesday 30 days before the third Friday of the FOLLOWING
    calendar month. If that Friday is an exchange holiday, count 30 days back
    from the business day before it; if the resulting Wednesday is a holiday,
    use the business day before it. E.g. Oct 2026 -> 2026-10-21; May 2027 ->
    2027-05-18 (third Friday of June 2027 is the observed Juneteenth holiday).
    """
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    first = pd.Timestamp(ny, nm, 1)
    third_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
    anchor = third_friday if is_nyse_session(third_friday) else _day(previous_session(third_friday))
    expiry = anchor - pd.Timedelta(days=30)
    return session_on_or_before(expiry)


def vx_expiry_for_month(expiry_month: str) -> str:
    """``vx_monthly_expiry`` for a 'YYYY-MM' contract month."""
    y, m = str(expiry_month)[:7].split("-")
    return vx_monthly_expiry(int(y), int(m))


# ══════════════════════════════════════════════════════════════════════════════
# Option contract conventions
# ══════════════════════════════════════════════════════════════════════════════

# OCC option symbol as Cboe and Yahoo publish it (no padding spaces):
# root, YYMMDD expiry, C/P, strike x 1000 in 8 digits. e.g. SPXW261016C07700000.
_OCC_RE = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<ymd>\d{6})(?P<cp>[CP])(?P<k>\d{8})$")

# Settlement style by root. SPX monthlies and all VIX options settle on the
# 09:30 ET special opening quotation (AM); SPXW settles on the 16:00 close (PM).
ROOT_SETTLEMENT = {"SPX": "AM", "SPXW": "PM", "VIX": "AM", "VIXW": "AM"}

_MINUTES_PER_YEAR = 525_600.0   # Cboe VIX methodology: T in minutes / 525,600


def parse_occ_symbol(symbol) -> dict | None:
    """Split an OCC option symbol into root, expiry (YYYY-MM-DD), right ('C'/'P') and strike.

    Returns None when ``symbol`` is not an OCC symbol.
    """
    if not isinstance(symbol, str):
        return None
    m = _OCC_RE.match(symbol.strip().replace(" ", ""))
    if not m:
        return None
    ymd = m.group("ymd")
    return {
        "root": m.group("root"),
        "expiry": f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:]}",
        "right": m.group("cp"),
        "strike": int(m.group("k")) / 1000.0,
    }


def settlement_for_root(root) -> str | None:
    """'AM' or 'PM' for a known option root, else None."""
    return ROOT_SETTLEMENT.get(str(root).upper()) if root else None


def time_to_settlement(snapshot_date, expiry, settlement: str | None = None) -> float:
    """Years from the 16:00 ET close of ``snapshot_date`` to the contract's settlement.

    AM-settled contracts settle at 09:30 ET on ``expiry``, PM-settled at 16:00
    ET. The interval is counted in New York wall-clock minutes (a DST change
    neither adds nor removes an hour) and divided by 525,600, as in the Cboe
    VIX methodology. With no known settlement style it falls back to whole
    calendar days / 365.
    """
    snap, exp = _day(snapshot_date), _day(expiry)
    if settlement not in ("AM", "PM"):
        return (exp - snap).days / 365.0
    start = datetime.combine(snap.date(), SESSION_CLOSE, NY_TZ)
    end = datetime.combine(exp.date(), SESSION_OPEN if settlement == "AM" else SESSION_CLOSE, NY_TZ)
    return (end - start).total_seconds() / 60.0 / _MINUTES_PER_YEAR


def _tte_column(df: pd.DataFrame) -> pd.Series:
    """Vectorised ``time_to_settlement`` over a frame with snapshot_date/expiry/settlement."""
    settle = df["settlement"] if "settlement" in df.columns else pd.Series(None, index=df.index)
    return pd.Series(
        [time_to_settlement(s, e, st if isinstance(st, str) else None)
         for s, e, st in zip(df["snapshot_date"], df["expiry"], settle)],
        index=df.index, dtype=float,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Rates
# ══════════════════════════════════════════════════════════════════════════════

TBILL_TENOR_DAYS = 91   # a 13-week bill


def tbill_discount_to_cc(discount_yield, days: int = TBILL_TENOR_DAYS):
    """Convert a T-bill bank-discount yield (decimal) to a continuously compounded rate.

    ^IRX quotes the 13-week bill on the BANK-DISCOUNT basis: the price per 1 of
    face is P = 1 - d * days / 360. The pricers here discount with exp(-r T)
    and T in ACT/365 years, so the matching rate is the continuously
    compounded ACT/365 yield of that price:

        r_cc = -ln(1 - d * days / 360) * 365 / days

    (For reference, the same price gives a money-market yield of
    d / (1 - d*days/360) on ACT/360, and a bond-equivalent yield of
    365 d / (360 - d*days).) d = 4% gives r_cc = 4.076%. Zero and negative
    discount yields are valid (the 2020-03-19..27 prints were negative) and
    map to zero and negative rates. Accepts a scalar or an array/Series.
    """
    d = np.asarray(discount_yield, dtype=float)
    price = 1.0 - d * days / 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(price > 0, -np.log(np.where(price > 0, price, 1.0)) * 365.0 / days, np.nan) + 0.0
    if isinstance(discount_yield, pd.Series):
        return pd.Series(r, index=discount_yield.index, name=discount_yield.name)
    return float(r) if r.ndim == 0 else r


# ══════════════════════════════════════════════════════════════════════════════
# Schema
# ══════════════════════════════════════════════════════════════════════════════

# Bumped when a migration in `_migrate` must run on an existing file.
# v2 (C23): tables cboe_index_daily and rates_daily (created, nothing rewritten).
SCHEMA_VERSION = 2

_TABLE_DDL = {
    "spx_ohlcv": """
    -- SPX daily OHLCV (Yahoo ^GSPC). log_return = ln(close_t / close_{t-1})
    -- against the previous STORED session; NULL only on the table's first row.
    CREATE TABLE IF NOT EXISTS spx_ohlcv (
        date        TEXT PRIMARY KEY,   -- NY session, YYYY-MM-DD
        open        REAL NOT NULL,
        high        REAL NOT NULL,
        low         REAL NOT NULL,
        close       REAL NOT NULL,
        volume      INTEGER NOT NULL,
        log_return  REAL
    )""",
    "vix_daily": """
    -- VIX index daily levels (Cboe VIX_History.csv or Yahoo ^VIX; see source)
    CREATE TABLE IF NOT EXISTS vix_daily (
        date        TEXT PRIMARY KEY,
        open        REAL NOT NULL,
        high        REAL NOT NULL,
        low         REAL NOT NULL,
        close       REAL NOT NULL,
        source      TEXT              -- 'cboe' | 'yfinance'; NULL = unrecorded (pre-C22)
    )""",
    "vix_futures_settlements": """
    -- VX final settlement values: the special opening quotation of VIX on the
    -- expiry morning, from the contract's own expiry-day settle (Cboe's VRO
    -- index where no contract file exists). One row per expiry.
    CREATE TABLE IF NOT EXISTS vix_futures_settlements (
        settlement_date  TEXT NOT NULL,   -- the contract's expiry date
        expiry_month     TEXT NOT NULL,   -- YYYY-MM contract month
        settlement_price REAL NOT NULL,
        is_weekly        INTEGER NOT NULL DEFAULT 0,
        source           TEXT,
        PRIMARY KEY (settlement_date, expiry_month)
    )""",
    "vix_futures_daily": """
    -- VX futures end-of-day data, one row per (trade date, contract). Keyed on
    -- the contract's expiry DATE so weekly and monthly contracts of the same
    -- month cannot collide.
    CREATE TABLE IF NOT EXISTS vix_futures_daily (
        date            TEXT NOT NULL,    -- NY session
        expiry_date     TEXT NOT NULL,    -- final settlement date
        expiry_month    TEXT NOT NULL,    -- YYYY-MM contract month
        symbol          TEXT,             -- Cboe symbol, e.g. 'VX/V6' or 'VX41/V6'
        is_weekly       INTEGER NOT NULL DEFAULT 0,
        open            REAL,
        high            REAL,
        low             REAL,
        close           REAL,             -- last trade; NULL if it did not trade
        settle          REAL,             -- Cboe daily settlement (the official mark)
        volume          INTEGER,
        open_interest   INTEGER,
        days_to_expiry  INTEGER,          -- calendar days from date to expiry_date
        source          TEXT,
        PRIMARY KEY (date, expiry_date)
    )""",
    "options_snapshots": """
    -- Option chain snapshots (SPX and VIX), one chain per (snapshot_date, underlying).
    CREATE TABLE IF NOT EXISTS options_snapshots (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date   TEXT NOT NULL,    -- NY session the QUOTES belong to (never the wall-clock date)
        underlying      TEXT NOT NULL,    -- 'SPX' | 'VIX'
        expiry          TEXT NOT NULL,    -- YYYY-MM-DD
        strike          REAL NOT NULL,
        right           TEXT NOT NULL,    -- 'C' | 'P'
        bid             REAL,
        ask             REAL,
        mid_price       REAL,
        implied_vol     REAL,             -- vendor IV as delivered (annualised decimal); not re-derived from mid_price
        open_interest   INTEGER,
        volume          INTEGER,
        delta           REAL,             -- vendor greeks (Cboe); NULL from Yahoo
        gamma           REAL,
        vega            REAL,
        theta           REAL,
        time_to_expiry  REAL,             -- years; see time_to_settlement()
        root            TEXT NOT NULL DEFAULT '',   -- OCC root (SPX/SPXW/VIX/VIXW); '' = unrecorded (pre-C22)
        contract_symbol TEXT,             -- OCC symbol
        settlement      TEXT,             -- 'AM' | 'PM'; NULL = unrecorded
        underlying_price REAL,            -- underlying level the quotes were taken against
        quote_time      TEXT,             -- when the quotes were observed (ISO-8601 UTC)
        source          TEXT              -- 'cboe' | 'yfinance'
    )""",
    "vix_term_structure": """
    -- Cboe VIX term-structure indices (^VIX9D, ^VIX, ^VIX3M, ^VIX6M) and ^VVIX.
    -- Each index is a SPOT-starting variance-swap rate over its horizon
    -- (tenor_days, from config.VIX_INDEX_TENOR_DAYS): expected volatility from
    -- today to T. They are NOT VIX futures, which settle on the 30-day
    -- volatility starting at their expiry; real VX futures are in
    -- vix_futures_daily. ^VVIX is the 30-day implied vol of VIX (is_vvix=1).
    CREATE TABLE IF NOT EXISTS vix_term_structure (
        date        TEXT NOT NULL,
        ticker      TEXT NOT NULL,
        tenor_days  INTEGER NOT NULL,
        close       REAL NOT NULL,
        is_vvix     INTEGER NOT NULL DEFAULT 0,
        source      TEXT,
        PRIMARY KEY (date, ticker)
    )""",
    "classifier_predictions": """
    -- XGBoost regime-classifier predictions (research output; the demoted
    -- model). NOT the deterministic rule labels the dashboard and backtest use.
    CREATE TABLE IF NOT EXISTS classifier_predictions (
        date            TEXT PRIMARY KEY,
        regime          INTEGER NOT NULL,  -- 0=A, 1=B, 2=C
        confidence      REAL NOT NULL,
        vix_level       REAL,
        vix_slope       REAL,
        rv_20d          REAL
    )""",
    "tbill_rates": """
    -- 13-week T-bill (^IRX via Yahoo). Zero look-ahead: read via get_tbill_rate(as_of_date).
    CREATE TABLE IF NOT EXISTS tbill_rates (
        date            TEXT PRIMARY KEY,
        rate            REAL NOT NULL,     -- continuously compounded, ACT/365, decimal (tbill_discount_to_cc)
        discount_yield  REAL               -- ^IRX as quoted (bank-discount basis), decimal; may be <= 0
    )""",
    "cboe_index_daily": """
    -- Other Cboe indices from their official history files (C23): VIX1D, VIX1Y,
    -- SKEW, the implied-correlation indices COR1M/COR3M, the dispersion index
    -- DSPX, and any strategy benchmark (PUT, WPUT, BXM, BFLY, CNDR, ...).
    -- One row per NYSE session and index. Single-value files (SKEW, DSPX)
    -- leave open/high/low NULL. Read via get_cboe_index(as_of_date).
    CREATE TABLE IF NOT EXISTS cboe_index_daily (
        date        TEXT NOT NULL,       -- NY session
        symbol      TEXT NOT NULL,       -- Cboe file name, e.g. 'SKEW', 'COR3M'
        open        REAL,
        high        REAL,
        low         REAL,
        close       REAL NOT NULL,
        source      TEXT,                -- 'cboe'
        PRIMARY KEY (date, symbol)
    )""",
    "rates_daily": """
    -- Daily interest-rate quotes as published (C23), the inputs of
    -- data/rates.py's zero curve: US Treasury bill rates (bank discount and
    -- coupon equivalent) and the par yield curve (home.treasury.gov), SOFR
    -- (NY Fed), and on request FRED copies (DGS*, SOFR, the H.15 3-month
    -- bill DTB3). value is the published quote / 100; its basis is fixed by
    -- the series name (rates.bill_series, rates.PAR_MONTHS). Dated by the
    -- publisher's business day, which is NOT always an NYSE session (the
    -- bond market closes on Columbus and Veterans Day). Read via
    -- get_rates_daily(as_of_date).
    CREATE TABLE IF NOT EXISTS rates_daily (
        date        TEXT NOT NULL,
        series      TEXT NOT NULL,       -- e.g. 'UST_BILL_13W_BD', 'UST_PAR_2Y', 'SOFR'
        value       REAL NOT NULL,       -- decimal (4.10% -> 0.041), basis per series
        source      TEXT NOT NULL,       -- 'treasury' | 'nyfed' | 'fred'
        PRIMARY KEY (date, series)
    )""",
}

_INDEX_DDL = [
    # The natural key. Its (snapshot_date, underlying) prefix also serves the
    # per-chain lookups, and vix_term_structure's (date, ticker) primary key
    # serves date-range scans, so the two older single-purpose indexes
    # (idx_options_snap_date_und, idx_vts_date) are dropped as redundant.
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_options_snapshot_key ON options_snapshots"
    "(snapshot_date, underlying, root, expiry, strike, right)",
    "DROP INDEX IF EXISTS idx_options_snap_date_und",
    "DROP INDEX IF EXISTS idx_vts_date",
]

# Columns added after a table first shipped: ALTERed into pre-C22 files.
_ADDED_COLUMNS = {
    "vix_daily": [("source", "TEXT")],
    "vix_term_structure": [("source", "TEXT")],
    "vix_futures_settlements": [("is_weekly", "INTEGER NOT NULL DEFAULT 0"), ("source", "TEXT")],
    "options_snapshots": [
        ("root", "TEXT NOT NULL DEFAULT ''"),
        ("contract_symbol", "TEXT"),
        ("settlement", "TEXT"),
        ("underlying_price", "REAL"),
        ("quote_time", "TEXT"),
        ("source", "TEXT"),
    ],
    "tbill_rates": [("discount_yield", "REAL")],
}

_OPTION_COLUMNS = [
    "snapshot_date", "underlying", "expiry", "strike", "right",
    "bid", "ask", "mid_price", "implied_vol",
    "open_interest", "volume", "delta", "gamma", "vega", "theta",
    "time_to_expiry", "root", "contract_symbol", "settlement",
    "underlying_price", "quote_time", "source",
]
_OPTION_KEY = ["snapshot_date", "underlying", "root", "expiry", "strike", "right"]


# ── Connections ───────────────────────────────────────────────────────────────

@contextmanager
def _connection():
    """Open DB_PATH, commit on success, roll back on error, and ALWAYS close.

    (``with sqlite3.connect(...)`` only commits/rolls back; it never closes.)
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table: str) -> list:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


_SCHEMA_READY: set = set()


def _db_key():
    p = Path(DB_PATH)
    try:
        return (str(p.resolve()), p.stat().st_ino)
    except OSError:
        return None


def _schema_is_current(conn) -> bool:
    """True if the file is at SCHEMA_VERSION with every table, column and index in place."""
    if conn.execute("PRAGMA user_version").fetchone()[0] < SCHEMA_VERSION:
        return False
    if not set(_TABLE_DDL) <= _tables(conn):
        return False
    for table, cols in _ADDED_COLUMNS.items():
        if not {c for c, _ in cols} <= set(_columns(conn, table)):
            return False
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    return "ux_options_snapshot_key" in indexes and not {"idx_options_snap_date_und", "idx_vts_date"} & indexes


def _ensure_schema() -> None:
    """Bring DB_PATH to the current schema once per process (cheap after the first call)."""
    key = _db_key()
    if key is not None and key in _SCHEMA_READY:
        return
    init_database()


def init_database() -> None:
    """Create all tables and migrate an older file to the current schema.

    Safe to call any number of times. On a file that is already current it
    only reads: it does not open a write transaction, so the git-tracked file
    is not touched (even a no-op write transaction bumps SQLite's header
    change counter). Otherwise the migration runs in one transaction; the
    schema version lives in ``PRAGMA user_version``. The file is left in
    rollback-journal (DELETE) mode.
    """
    path = Path(DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        wal = conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        if wal:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError as exc:   # another process holds the file
                logger.warning("Could not switch %s out of WAL mode: %s", path, exc)
        migrated = not _schema_is_current(conn)
        if migrated:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            conn.execute("BEGIN IMMEDIATE")
            try:
                vacuum = _migrate(conn, version)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            if vacuum:
                conn.execute("VACUUM")
    finally:
        conn.close()
    key = _db_key()
    if key is not None:
        _SCHEMA_READY.add(key)
    if migrated:
        logger.info("Database initialised at %s (schema v%d)", path, SCHEMA_VERSION)


def _migrate(conn, version: int) -> bool:
    """Apply every migration step newer than ``version``. Returns True if a VACUUM is due."""
    tables = _tables(conn)
    legacy = version < 1 and bool(tables)     # a file written by the pre-C22 schema
    vacuum = False

    # 1. Renames/rebuilds that must precede CREATE TABLE IF NOT EXISTS.
    if "regime_labels" in tables and "classifier_predictions" not in tables:
        # The table always held the demoted classifier's predictions (it has a
        # `confidence` column), never the rule labels its name implied.
        conn.execute("ALTER TABLE regime_labels RENAME TO classifier_predictions")
        logger.info("migration: regime_labels -> classifier_predictions")
    if "vix_futures_daily" in tables and "expiry_date" not in _columns(conn, "vix_futures_daily"):
        _rebuild_legacy_vix_futures_daily(conn)

    # 2. Missing tables, 3. columns added after a table first shipped.
    for stmt in _TABLE_DDL.values():
        conn.execute(stmt)
    for table, cols in _ADDED_COLUMNS.items():
        have = set(_columns(conn, table))
        for col, decl in cols:
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    # 4. One-time data repairs of a pre-C22 file.
    if legacy:
        _migrate_tbill_to_cc(conn)
        n = conn.execute(
            "UPDATE options_snapshots SET source='yfinance' "
            "WHERE source IS NULL AND delta IS NULL AND gamma IS NULL AND vega IS NULL"
        ).rowcount
        if n:
            # Inferred, not recorded: Cboe always supplies greeks, Yahoo never does.
            logger.info("migration: %d legacy option rows (no greeks) tagged source='yfinance'", n)
        _redate_nonsession_snapshots(conn)
        _sync_vix_index_tenors(conn)
        _recompute_log_returns(conn, from_date=None)
        if "spx_options" in tables:
            # Dead table left by the pre-C20 bot; nothing ever read it.
            conn.execute("DROP TABLE spx_options")
            logger.info("migration: dropped dead table spx_options")
            vacuum = True

    # 5. Indexes (the unique key needs `root`, so after step 3).
    _drop_duplicate_option_keys(conn)
    for stmt in _INDEX_DDL:
        conn.execute(stmt)
    return vacuum


def _rebuild_legacy_vix_futures_daily(conn) -> None:
    """Move a pre-C22 vix_futures_daily (keyed on expiry_month) to the expiry_date key."""
    rows = conn.execute("SELECT * FROM vix_futures_daily").fetchall()
    conn.execute("ALTER TABLE vix_futures_daily RENAME TO vix_futures_daily_pre_c22")
    conn.execute(_TABLE_DDL["vix_futures_daily"])
    for r in rows:
        r = dict(r)
        conn.execute(
            "INSERT OR REPLACE INTO vix_futures_daily (date, expiry_date, expiry_month, is_weekly, "
            "open, high, low, close, settle, volume, open_interest, days_to_expiry) "
            "VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
            (r["date"], vx_expiry_for_month(r["expiry_month"]), r["expiry_month"],
             r.get("open"), r.get("high"), r.get("low"), r.get("close"), r.get("close"),
             r.get("volume"), r.get("open_interest"), r.get("days_to_expiry")),
        )
    conn.execute("DROP TABLE vix_futures_daily_pre_c22")
    logger.info("migration: vix_futures_daily re-keyed on (date, expiry_date); %d row(s) moved", len(rows))


def _migrate_tbill_to_cc(conn) -> None:
    """Pre-C22 files stored ^IRX/100 (a discount yield) in `rate`. Convert to continuous compounding."""
    rows = conn.execute("SELECT date, rate FROM tbill_rates WHERE discount_yield IS NULL").fetchall()
    if not rows:
        return
    d = np.array([r["rate"] for r in rows], dtype=float)
    cc = tbill_discount_to_cc(d)
    conn.executemany(
        "UPDATE tbill_rates SET discount_yield = ?, rate = ? WHERE date = ?",
        [(float(di), float(ri), r["date"]) for di, ri, r in zip(d, cc, rows)],
    )
    logger.info("migration: %d tbill_rates rows converted from discount yield to "
                "continuously compounded (mean shift %+.1f bp)", len(rows), float(np.mean(cc - d)) * 1e4)


def _sync_vix_index_tenors(conn) -> None:
    for ticker, days in {**VIX_INDEX_TENOR_DAYS, "^VVIX": VVIX_TENOR_DAYS}.items():
        n = conn.execute(
            "UPDATE vix_term_structure SET tenor_days = ? WHERE ticker = ? AND tenor_days != ?",
            (days, ticker, days),
        ).rowcount
        if n:
            logger.info("migration: %s tenor_days -> %d on %d row(s)", ticker, days, n)


def _drop_duplicate_option_keys(conn) -> None:
    n = conn.execute(
        "DELETE FROM options_snapshots WHERE id NOT IN ("
        " SELECT MAX(id) FROM options_snapshots"
        " GROUP BY snapshot_date, underlying, root, expiry, strike, right)"
    ).rowcount
    if n:
        logger.warning("options_snapshots: removed %d duplicate row(s) on the natural key "
                       "(kept the latest insert of each)", n)


def _redate_nonsession_snapshots(conn) -> list:
    """Re-date every snapshot whose date is not an NYSE session to the session its quotes are from.

    A chain pulled on a weekend or holiday holds the previous session's quotes;
    the old downloaders dated it by the wall clock, so every time-to-expiry was
    short by the gap. time_to_expiry is recomputed from the new date. Returns
    [(underlying, old_date, new_date, rows)].
    """
    moved = []
    pairs = conn.execute(
        "SELECT underlying, snapshot_date, COUNT(*) FROM options_snapshots "
        "GROUP BY underlying, snapshot_date ORDER BY snapshot_date, underlying"
    ).fetchall()
    for und, old, n in pairs:
        if is_nyse_session(old):
            continue
        new = previous_session(old)
        clash = conn.execute(
            "SELECT COUNT(*) FROM options_snapshots WHERE underlying=? AND snapshot_date=?",
            (und, new),
        ).fetchone()[0]
        if clash:
            logger.warning("%s snapshot %s is dated on a closed day, but session %s already has "
                           "a chain; leaving it for manual review", und, old, new)
            continue
        rows = conn.execute(
            "SELECT id, expiry, settlement FROM options_snapshots WHERE underlying=? AND snapshot_date=?",
            (und, old),
        ).fetchall()
        conn.executemany(
            "UPDATE options_snapshots SET snapshot_date = ?, time_to_expiry = ? WHERE id = ?",
            [(new, time_to_settlement(new, r["expiry"], r["settlement"]), r["id"]) for r in rows],
        )
        moved.append((und, old, new, n))
        logger.warning("re-dated %s snapshot %s (%s) -> quote session %s (%s); %d rows, "
                       "time_to_expiry recomputed", und, old, _day(old).day_name(), new,
                       _day(new).day_name(), n)
    return moved


def redate_nonsession_snapshots() -> list:
    """Public wrapper: re-date weekend/holiday-dated option snapshots in DB_PATH (see above)."""
    _ensure_schema()
    with _connection() as conn:
        return _redate_nonsession_snapshots(conn)


# ── Record helpers ────────────────────────────────────────────────────────────

def _records(df: pd.DataFrame, cols: Sequence[str]) -> list:
    """Rows as dicts with NaN/NA/NaT -> None and numpy scalars -> Python (sqlite3-safe)."""
    out = []
    for rec in df[list(cols)].to_dict("records"):
        clean = {}
        for k, v in rec.items():
            if v is None or v is pd.NA or v is pd.NaT:
                clean[k] = None
            elif isinstance(v, (float, np.floating)):
                clean[k] = None if math.isnan(v) else float(v)
            elif isinstance(v, (np.integer, np.bool_, bool)):
                clean[k] = int(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def _warn_dropped(conn, table: str, incoming: pd.DataFrame, keys: Sequence[str],
                  label: str) -> list:
    """Warn about stored rows inside the incoming window that the source no longer returns.

    The upsert keeps them (never deletes); this only makes the drop visible.
    """
    if incoming.empty:
        return []
    lo, hi = incoming["date"].min(), incoming["date"].max()
    stored = pd.read_sql_query(
        f"SELECT {', '.join(keys)} FROM {table} WHERE date BETWEEN ? AND ?", conn, params=[lo, hi]
    )
    if stored.empty:
        return []
    if "ticker" in keys:
        stored = stored[stored["ticker"].isin(incoming["ticker"].unique())]
    have = set(map(tuple, incoming[list(keys)].astype(str).to_numpy()))
    gone = [tuple(r) for r in stored[list(keys)].astype(str).to_numpy() if tuple(r) not in have]
    if gone:
        shown = ", ".join("/".join(g) for g in gone[:10])
        logger.warning("%s: source no longer returns %d stored row(s) in %s..%s (kept): %s%s",
                       label, len(gone), lo, hi, shown, " ..." if len(gone) > 10 else "")
    return gone


def find_bad_prints(df: pd.DataFrame, label: str = "series", *, log: bool = True) -> pd.DataFrame:
    """Rows whose OHLC are internally inconsistent: close <= 0, low > high, or close outside [low, high].

    Logs a warning naming them and returns them; the caller decides what to do.
    """
    if df.empty or "close" not in df.columns:
        return df.iloc[0:0]
    bad = df["close"].le(0) | df["close"].isna()
    if {"low", "high"}.issubset(df.columns):
        lo, hi, c = df["low"], df["high"], df["close"]
        tol = 1e-6 * c.abs()
        bad |= (lo > hi + tol) | (c > hi + tol) | (c < lo - tol)
    flagged = df[bad.fillna(False)]
    if log and not flagged.empty:
        logger.warning("%s: %d bad print(s) (inconsistent OHLC or non-positive close) on %s", label,
                       len(flagged), ", ".join(map(str, flagged["date"].head(10))))
    return flagged


# ══════════════════════════════════════════════════════════════════════════════
# SPX OHLCV
# ══════════════════════════════════════════════════════════════════════════════

def insert_spx_ohlcv(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """
    Upsert SPX daily OHLCV rows.

    Parameters
    ----------
    df : DataFrame with columns [date, open, high, low, close, volume].
         date must be a string 'YYYY-MM-DD' or datetime-like.
    warn_dropped : log stored sessions inside the incoming window that the
         incoming frame lacks (they are kept either way).

    log_return is (re)computed in the database against the previous STORED
    session — including the first row of this batch and every row after an
    inserted gap — so an overlapping re-download never nulls a stored return.

    Returns
    -------
    int : number of rows inserted/replaced.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    find_bad_prints(df, "spx_ohlcv")
    validate_sessions(df["date"], label="spx_ohlcv")
    rows = _records(df, ["date", "open", "high", "low", "close", "volume"])
    sql = """
        INSERT OR REPLACE INTO spx_ohlcv
            (date, open, high, low, close, volume, log_return)
        VALUES
            (:date, :open, :high, :low, :close, :volume, NULL)
    """
    with _connection() as conn:
        if warn_dropped:
            _warn_dropped(conn, "spx_ohlcv", df, ["date"], "spx_ohlcv")
        conn.executemany(sql, rows)
        _recompute_log_returns(conn, from_date=df["date"].min())
    logger.info("Upserted %d SPX OHLCV rows", len(rows))
    return len(rows)


def _recompute_log_returns(conn, from_date: str | None) -> int:
    """Recompute log_return for every row from ``from_date`` on, against the previous stored close."""
    anchor = None
    if from_date is not None:
        anchor = conn.execute(
            "SELECT MAX(date) FROM spx_ohlcv WHERE date < ?", (from_date,)
        ).fetchone()[0]
    lo = anchor if anchor is not None else (from_date or "")
    rows = conn.execute(
        "SELECT date, close FROM spx_ohlcv WHERE date >= ? ORDER BY date", (lo,)
    ).fetchall()
    if not rows:
        return 0
    closes = np.array([r["close"] for r in rows], dtype=float)
    lr = np.full(len(rows), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr[1:] = np.log(closes[1:] / closes[:-1])
    start = 1 if anchor is not None else 0          # the anchor keeps its own return
    updates = [(None if not np.isfinite(lr[i]) else float(lr[i]), rows[i]["date"])
               for i in range(start, len(rows))]
    conn.executemany("UPDATE spx_ohlcv SET log_return = ? WHERE date = ?", updates)
    return len(updates)


def get_spx_ohlcv(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Return SPX OHLCV for [start_date, as_of_date].

    LOOK-AHEAD GUARD: no data beyond as_of_date is ever returned.

    Parameters
    ----------
    as_of_date : str 'YYYY-MM-DD' — upper bound (inclusive).
    start_date : str 'YYYY-MM-DD' — lower bound (inclusive). If None,
                 returns all data up to as_of_date.

    Returns
    -------
    DataFrame sorted ascending by date.
    """
    return _get_dated("spx_ohlcv", as_of_date, start_date)


def _get_dated(table: str, as_of_date: str, start_date: str | None,
               order: str = "date ASC") -> pd.DataFrame:
    _validate_date(as_of_date)
    sql = f"SELECT * FROM {table} WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    sql += f" ORDER BY {order}"
    with _connection() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VIX daily
# ══════════════════════════════════════════════════════════════════════════════

# ── Source vocabulary and precedence ──────────────────────────────────────────

# One spelling per data source in every `source` column. option_chain's
# observed_quotes() tags its rows 'yahoo' and yf_downloader 'yfinance' (both
# Yahoo through yfinance): the writers below store 'yfinance' for either.
_SOURCE_ALIASES = {"yahoo": "yfinance", "yf": "yfinance", "y!": "yfinance"}

# The official publisher of the VIX index family. In the VIX-index tables
# (vix_daily, vix_term_structure, cboe_index_daily) a stored row from this
# source is replaced only by another row from it.
OFFICIAL_INDEX_SOURCE = "cboe"


def canonical_source(values) -> pd.Series:
    """``values`` (a Series) with known aliases mapped to one spelling; None stays None."""
    s = pd.Series(values) if not isinstance(values, pd.Series) else values

    def one(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
        return _SOURCE_ALIASES.get(str(v).strip().lower(), v)
    return pd.Series([one(v) for v in s.astype(object)], index=s.index, dtype=object)


def _skip_over_official(conn, table: str, df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Drop incoming non-Cboe rows whose key is stored with source 'cboe'.

    Cboe's index files are the official print. The bot writes Yahoo's window
    first and Cboe's on top; when the Cboe download failed, the Yahoo rows
    used to replace the official closes for the whole 400-day window
    (2026-02-06: Yahoo ^VIX 20.37, Cboe 17.76) and the run committed them.
    """
    not_official = df["source"].astype(object).ne(OFFICIAL_INDEX_SOURCE).to_numpy()
    if not not_official.any():
        return df
    key_sql = ", ".join(keys)
    # table and key names come from this module's constants; the values are bound
    official = {tuple(map(str, r)) for r in conn.execute(
        f"SELECT {key_sql} FROM {table} WHERE source = ? AND date BETWEEN ? AND ?",  # noqa: S608
        (OFFICIAL_INDEX_SOURCE, df["date"].min(), df["date"].max()))}
    if not official:
        return df
    incoming = [tuple(map(str, r)) for r in df[list(keys)].to_numpy()]
    keep = ~(not_official & np.array([k in official for k in incoming], dtype=bool))
    if not keep.all():
        logger.info("%s: kept %d stored Cboe close(s); the incoming non-Cboe rows for those "
                    "keys were skipped", table, int((~keep).sum()))
    return df[keep]


def insert_vix_daily(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """Upsert VIX index daily OHLC. An optional `source` column ('cboe'/'yfinance') is stored.

    A stored row whose source is 'cboe' (VIX_History.csv, the official
    print) is only ever replaced by another Cboe row: incoming rows from any
    other source (Yahoo's ^VIX, or unrecorded) for those dates are skipped
    (_skip_over_official). Rows dated on days the NYSE was closed are not
    stored (drop_non_sessions). Returns the rows written.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df = drop_non_sessions(df, "vix_daily")
    if df.empty:
        return 0
    df["source"] = canonical_source(df["source"]) if "source" in df.columns else None
    find_bad_prints(df, "vix_daily")
    validate_sessions(df["date"], label="vix_daily")
    sql = """
        INSERT OR REPLACE INTO vix_daily (date, open, high, low, close, source)
        VALUES (:date, :open, :high, :low, :close, :source)
    """
    with _connection() as conn:
        if warn_dropped:
            _warn_dropped(conn, "vix_daily", df, ["date"], "vix_daily")
        df = _skip_over_official(conn, "vix_daily", df, ["date"])
        rows = _records(df, ["date", "open", "high", "low", "close", "source"])
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX daily rows", len(rows))
    return len(rows)


def get_vix_daily(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Return VIX daily levels up to as_of_date.

    LOOK-AHEAD GUARD: strictly enforced.
    """
    return _get_dated("vix_daily", as_of_date, start_date)


# ══════════════════════════════════════════════════════════════════════════════
# VX futures
# ══════════════════════════════════════════════════════════════════════════════

def _expiry_dates_for_months(months: pd.Series) -> pd.Series:
    lookup = {m: vx_expiry_for_month(m) for m in months.dropna().unique()}
    return months.map(lookup)


def insert_vix_futures_settlements(df: pd.DataFrame) -> int:
    """
    Upsert VX final settlement values.

    df columns: [settlement_date, settlement_price] plus optional expiry_month
    ('YYYY-MM', defaults to the settlement date's month — a VX contract always
    expires inside its contract month), is_weekly (defaults to "settlement_date
    is not the standard monthly expiry") and source.
    """
    _ensure_schema()
    df = df.copy()
    if df.empty:
        return 0
    df["settlement_date"] = _to_date_str(df["settlement_date"])
    if "expiry_month" not in df.columns:
        df["expiry_month"] = df["settlement_date"].str[:7]
    if "is_weekly" not in df.columns:
        df["is_weekly"] = (df["settlement_date"] != _expiry_dates_for_months(df["expiry_month"])).astype(int)
    if "source" not in df.columns:
        df["source"] = None
    rows = _records(df, ["settlement_date", "expiry_month", "settlement_price", "is_weekly", "source"])
    sql = """
        INSERT OR REPLACE INTO vix_futures_settlements
            (settlement_date, expiry_month, settlement_price, is_weekly, source)
        VALUES (:settlement_date, :expiry_month, :settlement_price, :is_weekly, :source)
    """
    with _connection() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX futures settlement rows", len(rows))
    return len(rows)


_VX_DAILY_COLUMNS = ["date", "expiry_date", "expiry_month", "symbol", "is_weekly", "open", "high",
                     "low", "close", "settle", "volume", "open_interest", "days_to_expiry", "source"]


def insert_vix_futures_daily(df: pd.DataFrame) -> int:
    """
    Upsert VX futures end-of-day rows, keyed (date, expiry_date), merging
    column-wise with any row already stored for that key (non-NULL wins).

    Accepts the Cboe-shaped frame from cboe_downloader (date, expiry_date,
    settle, ...). For backward compatibility it also accepts the legacy shape
    [date, expiry_month, open, high, low, close, volume, open_interest,
    days_to_expiry]: the expiry date is then the standard monthly expiry of
    that contract month, and `close` doubles as the settlement price (the old
    parser stored Cboe's Settle column under that name).

    Cboe writes 0 for "no trade" in its price columns; those become NULL.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    if "expiry_date" not in df.columns:
        if "expiry_month" not in df.columns:
            raise ValueError("VX rows need expiry_date or expiry_month")
        df["expiry_date"] = _expiry_dates_for_months(df["expiry_month"].astype(str).str[:7])
    df["expiry_date"] = _to_date_str(df["expiry_date"])
    if "expiry_month" not in df.columns:
        df["expiry_month"] = df["expiry_date"].str[:7]
    if "settle" not in df.columns:
        df["settle"] = df["close"] if "close" in df.columns else np.nan
    for col in ("open", "high", "low", "close", "settle"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            df[col] = v.where(v > 0)
        else:
            df[col] = np.nan
    df = df[df["settle"].notna() | df["close"].notna()].copy()
    if "days_to_expiry" not in df.columns:
        df["days_to_expiry"] = (pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["date"])).dt.days
    if "is_weekly" not in df.columns:
        df["is_weekly"] = (df["expiry_date"] != _expiry_dates_for_months(df["expiry_month"])).astype(int)
    for col in ("symbol", "source", "volume", "open_interest"):
        if col not in df.columns:
            df[col] = None
    rows = _records(df, _VX_DAILY_COLUMNS)
    # Upsert that MERGES: the daily settlement file carries symbol + settle,
    # the per-contract file carries OHLC/volume/OI + settle. A NULL in the
    # incoming row never erases a value another source already stored.
    updates = ", ".join(f"{c} = COALESCE(excluded.{c}, vix_futures_daily.{c})"
                        for c in _VX_DAILY_COLUMNS if c not in ("date", "expiry_date"))
    sql = (f"INSERT INTO vix_futures_daily ({', '.join(_VX_DAILY_COLUMNS)}) "
           f"VALUES ({', '.join(':' + c for c in _VX_DAILY_COLUMNS)}) "
           f"ON CONFLICT(date, expiry_date) DO UPDATE SET {updates}")
    with _connection() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX futures daily rows", len(rows))
    return len(rows)


def get_vix_futures_curve(as_of_date: str, include_weeklies: bool = False) -> pd.DataFrame:
    """
    Return the VX futures curve observed on exactly ``as_of_date``.

    Live contracts only (expiry_date > as_of_date), front month first. Weekly
    contracts are excluded by default: on days a weekly does not trade, Cboe
    publishes a monthly contract's settle as its daily settlement (every
    untraded weekly in the 2026-10 files; on 2026-10-16 the November-month
    weeklies VX44/X6 and VX45/X6 carried the OCTOBER monthly's 19.9075), which
    is not an independent price.

    LOOK-AHEAD GUARD: only prices observed on as_of_date are returned.

    Returns
    -------
    DataFrame with columns [expiry_date, expiry_month, symbol, is_weekly,
    settle, close, price, days_to_expiry]; ``price`` is the settle where Cboe
    published one (trade dates from mid-2013 on) and the last trade otherwise.
    """
    _validate_date(as_of_date)
    _ensure_schema()
    sql = """
        SELECT expiry_date, expiry_month, symbol, is_weekly, settle, close,
               COALESCE(settle, close) AS price, days_to_expiry
        FROM vix_futures_daily
        WHERE date = ? AND expiry_date > ?
    """
    if not include_weeklies:
        sql += " AND is_weekly = 0"
    sql += " ORDER BY expiry_date ASC"
    with _connection() as conn:
        return pd.read_sql_query(sql, conn, params=[as_of_date, as_of_date])


def get_vix_futures_settlements(as_of_date: str) -> pd.DataFrame:
    """
    Return all VIX futures settlement prices settled on or before as_of_date.

    LOOK-AHEAD GUARD: settlement_date <= as_of_date strictly.
    """
    _validate_date(as_of_date)
    sql = """
        SELECT * FROM vix_futures_settlements
        WHERE settlement_date <= ?
        ORDER BY settlement_date ASC, expiry_month ASC
    """
    with _connection() as conn:
        return pd.read_sql_query(sql, conn, params=[as_of_date])


# ══════════════════════════════════════════════════════════════════════════════
# Options snapshots
# ══════════════════════════════════════════════════════════════════════════════

def insert_options_snapshot(df: pd.DataFrame, underlying: str) -> int:
    """
    Store an options chain snapshot, REPLACING any chain already stored for the
    same (snapshot_date, underlying).

    Parameters
    ----------
    df : DataFrame with columns matching options_snapshots (snapshot_date must
         be set and should be the NY session the quotes belong to). Optional
         columns are filled: `root` from `contract_symbol` (or yfinance's
         `contractSymbol`) when that is an OCC symbol (else ''), `settlement`
         from the root, `time_to_expiry` from ``time_to_settlement`` — which
         also replaces a supplied value wherever the settlement style is known.
    underlying : 'SPX' or 'VIX'

    The delete and the insert run in one transaction, so a re-run of the same
    session (e.g. a later, post-close pull) leaves exactly one chain. Rows that
    repeat the natural key (snapshot_date, underlying, root, expiry, strike,
    right) keep the last occurrence, with a warning.

    Returns
    -------
    int : rows stored.
    """
    _ensure_schema()
    if df is None or df.empty:
        return 0
    df = df.copy()
    und = underlying.upper()
    df["underlying"] = und
    df["snapshot_date"] = _to_date_str(df["snapshot_date"])
    df["expiry"] = _to_date_str(df["expiry"])
    if "contract_symbol" not in df.columns and "contractSymbol" in df.columns:
        df["contract_symbol"] = df["contractSymbol"]          # raw yfinance column name

    if "root" not in df.columns:
        if "contract_symbol" in df.columns:
            df["root"] = [(parse_occ_symbol(s) or {}).get("root", "") for s in df["contract_symbol"]]
        else:
            df["root"] = ""
    df["root"] = df["root"].fillna("").astype(str)
    if "settlement" not in df.columns:
        df["settlement"] = [settlement_for_root(r) for r in df["root"]]
    # time_to_expiry follows time_to_settlement(): computed where missing, and
    # where the settlement style is known a writer's own value is replaced so
    # the column never contradicts `settlement` (e.g. whole days for an AM
    # contract overstate T by 6.5 hours).
    tte = _tte_column(df)
    if "time_to_expiry" not in df.columns:
        df["time_to_expiry"] = tte
    else:
        given = pd.to_numeric(df["time_to_expiry"], errors="coerce")
        known = df["settlement"].isin(["AM", "PM"])
        off = known & ~np.isclose(given, tte, rtol=0.0, atol=1e-9)
        if off.any():
            logger.warning("%s snapshot: %d time_to_expiry value(s) recomputed from the AM/PM "
                           "settlement time", und, int(off.sum()))
        df["time_to_expiry"] = given.where(~known & given.notna(), tte)

    if "source" in df.columns:
        spelled = df["source"].astype(str).str.strip().str.lower()
        aliased = df["source"].notna() & spelled.isin(_SOURCE_ALIASES)
        if aliased.any():
            logger.info("%s snapshot: source %s stored as %s", und,
                        sorted(df.loc[aliased, "source"].astype(str).unique()),
                        sorted(set(canonical_source(df.loc[aliased, "source"]))))
        df["source"] = canonical_source(df["source"])

    for snap in sorted(df["snapshot_date"].unique()):
        if not is_nyse_session(snap):
            logger.warning("%s snapshot dated %s, which is not an NYSE session — snapshot dates "
                           "should be the session the quotes belong to (database.quote_session)",
                           und, snap)

    dup = df.duplicated(_OPTION_KEY, keep="last")
    if dup.any():
        logger.warning("%s snapshot: dropped %d row(s) repeating the natural key %s (kept the last)",
                       und, int(dup.sum()), _OPTION_KEY)
        df = df[~dup]

    cols = [c for c in _OPTION_COLUMNS if c in df.columns]
    rows = _records(df, cols)
    sql = (f"INSERT INTO options_snapshots ({', '.join(cols)}) "
           f"VALUES ({', '.join(':' + c for c in cols)})")
    snaps = sorted(df["snapshot_date"].unique())
    with _connection() as conn:
        replaced = 0
        for snap in snaps:
            replaced += conn.execute(
                "DELETE FROM options_snapshots WHERE snapshot_date = ? AND underlying = ?", (snap, und)
            ).rowcount
        conn.executemany(sql, rows)
    logger.info("Stored %d %s option rows for %s%s", len(rows), und, ", ".join(snaps),
                f" (replaced {replaced})" if replaced else "")
    return len(rows)


def get_options_surface(as_of_date: str, underlying: str,
                        max_age_sessions: int | None = None) -> pd.DataFrame:
    """
    Return the most recent options chain for an underlying as of a date.

    LOOK-AHEAD GUARD: snapshot_date <= as_of_date AND expiry > as_of_date
    (we only return options that haven't expired yet as of the query date,
    AND whose price was observed no later than as_of_date).

    STALENESS: the chain can be older than ``as_of_date``. Its
    ``time_to_expiry`` is measured from ITS ``snapshot_date`` (returned as a
    column) — pair it with the spot of that session, not of as_of_date. The
    age in NYSE sessions is checked against the session of as_of_date: with
    ``max_age_sessions`` set (0 = same session only) an older chain raises
    StaleDataError; without it a stale chain is returned with a warning.

    Parameters
    ----------
    as_of_date : str 'YYYY-MM-DD'
    underlying : 'SPX' or 'VIX'
    max_age_sessions : int or None

    Returns
    -------
    DataFrame with full chain, sorted by expiry, strike.
    """
    _validate_date(as_of_date)
    sql = """
        SELECT * FROM options_snapshots
        WHERE underlying = ?
          AND snapshot_date = (
              SELECT MAX(snapshot_date)
              FROM options_snapshots
              WHERE underlying = ?
                AND snapshot_date <= ?
          )
          AND expiry > ?
        ORDER BY expiry ASC, strike ASC
    """
    params = [underlying.upper(), underlying.upper(), as_of_date, as_of_date]
    with _connection() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    if not df.empty:
        snap = _iso(df["snapshot_date"].iloc[0])
        age = sessions_between(snap, session_on_or_before(as_of_date))
        if max_age_sessions is not None and age > max_age_sessions:
            raise StaleDataError(
                f"newest {underlying.upper()} chain on or before {as_of_date} is {snap}, "
                f"{age} session(s) old (allowed: {max_age_sessions})")
        if age > 0:
            logger.warning("get_options_surface(%s, %s): returning the %s chain, %d session(s) "
                           "stale; its time_to_expiry is measured from %s", as_of_date,
                           underlying.upper(), snap, age, snap)
    return df


_DATED_TABLES = {"spx_ohlcv": "date", "vix_daily": "date", "vix_term_structure": "date",
                 "tbill_rates": "date", "vix_futures_daily": "date",
                 "vix_futures_settlements": "settlement_date", "options_snapshots": "snapshot_date",
                 "cboe_index_daily": "date", "rates_daily": "date"}


def get_latest_date(table: str) -> str | None:
    """Latest stored date in one of the dated tables (None if it is empty)."""
    if table not in _DATED_TABLES:
        raise ValueError(f"not a dated table: {table}")
    _ensure_schema()
    with _connection() as conn:
        return conn.execute(f"SELECT MAX({_DATED_TABLES[table]}) FROM {table}").fetchone()[0]


def get_options_snapshot_dates(underlying: str | None = None) -> list:
    """Sorted snapshot dates stored (optionally for one underlying)."""
    _ensure_schema()
    sql = "SELECT DISTINCT snapshot_date FROM options_snapshots"
    params: list = []
    if underlying:
        sql += " WHERE underlying = ?"
        params.append(underlying.upper())
    with _connection() as conn:
        return sorted(r[0] for r in conn.execute(sql + " ORDER BY snapshot_date", params))


# ══════════════════════════════════════════════════════════════════════════════
# VIX term structure
# ══════════════════════════════════════════════════════════════════════════════

_TERM_TENORS = {**VIX_INDEX_TENOR_DAYS, "^VVIX": VVIX_TENOR_DAYS}


def insert_vix_term_structure(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """Upsert VIX term-structure index rows [date, ticker, close, (tenor_days), (is_vvix), (source)].

    tenor_days for ^VIX9D/^VIX/^VIX3M/^VIX6M/^VVIX always comes from config
    (VIX_INDEX_TENOR_DAYS / VVIX_TENOR_DAYS); a conflicting value in ``df`` is
    overridden with a warning, so no writer can store a second tenor map.

    Cboe is the primary source (C23): a stored (date, ticker) row from Cboe's
    official file is only replaced by another Cboe row, so a Yahoo row is a
    provisional value that stands until Cboe publishes that session
    (pipeline.refresh_vix_index_family writes both in that order). Rows
    dated on days the NYSE was closed are not stored.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    df = df.drop_duplicates(["date", "ticker"], keep="last").copy()
    df = drop_non_sessions(df, "vix_term_structure").copy()
    if df.empty:
        return 0
    canon = df["ticker"].map(_TERM_TENORS)
    if "tenor_days" in df.columns:
        given = pd.to_numeric(df["tenor_days"], errors="coerce")
        clash = canon.notna() & given.notna() & (given != canon)
        if clash.any():
            logger.warning("vix_term_structure: overriding tenor_days for %s with config values %s",
                           sorted(df.loc[clash, "ticker"].unique()),
                           {t: _TERM_TENORS[t] for t in df.loc[clash, "ticker"].unique()})
        df["tenor_days"] = canon.fillna(given)
    else:
        df["tenor_days"] = canon
    if df["tenor_days"].isna().any():
        raise ValueError(f"unknown tenor for tickers {sorted(df.loc[df['tenor_days'].isna(), 'ticker'].unique())}")
    df["tenor_days"] = df["tenor_days"].astype(int)
    df["is_vvix"] = (df["is_vvix"].astype(bool) if "is_vvix" in df.columns
                     else df["ticker"].eq("^VVIX")).astype(int)
    df["source"] = canonical_source(df["source"]) if "source" in df.columns else None
    for ticker, sub in df.groupby("ticker"):
        validate_sessions(sub["date"], label=f"vix_term_structure {ticker}")
    sql = """
        INSERT OR REPLACE INTO vix_term_structure
            (date, ticker, tenor_days, close, is_vvix, source)
        VALUES (:date, :ticker, :tenor_days, :close, :is_vvix, :source)
    """
    with _connection() as conn:
        if warn_dropped:
            _warn_dropped(conn, "vix_term_structure", df, ["date", "ticker"], "vix_term_structure")
        df = _skip_over_official(conn, "vix_term_structure", df, ["date", "ticker"])
        rows = _records(df, ["date", "ticker", "tenor_days", "close", "is_vvix", "source"])
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX term structure rows", len(rows))
    return len(rows)


def get_vix_term_structure(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Return VIX term structure up to as_of_date (long format, one row per date and ticker).

    LOOK-AHEAD GUARD: date <= as_of_date strictly.
    """
    return _get_dated("vix_term_structure", as_of_date, start_date, order="date ASC, tenor_days ASC")


def get_vix_term_structure_wide(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Return VIX term structure in wide format: one column per ticker.

    Columns: [^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^VVIX] (plus derived slope).
    Useful for regime classifier feature construction.
    """
    return vix_term_structure_wide(get_vix_term_structure(as_of_date, start_date))


def vix_term_structure_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """get_vix_term_structure_wide's pivot of long (date, ticker, close) rows:
    one column per ticker plus the two slope columns, ``date`` a column.
    Public so a caller that edits long rows (backtest.c23_bridge) builds the
    exact frame the database reader would."""
    if df_long.empty:
        return pd.DataFrame()
    wide = df_long.pivot(index="date", columns="ticker", values="close")
    wide.columns.name = None
    # Term structure slope: VIX3M - VIX9D (or VIX6M - VIX if 9D unavailable)
    if "^VIX3M" in wide.columns and "^VIX9D" in wide.columns:
        wide["ts_slope_3m9d"] = wide["^VIX3M"] - wide["^VIX9D"]
    if "^VIX6M" in wide.columns and "^VIX" in wide.columns:
        wide["ts_slope_6m1m"] = wide["^VIX6M"] - wide["^VIX"]
    return wide.reset_index()


def compare_sources(primary: pd.DataFrame, secondary: pd.DataFrame, tol: float = 0.25, *,
                    keys: Sequence[str] = ("date", "ticker"), value: str = "close",
                    labels: tuple = ("cboe", "yfinance"), log: bool = True) -> pd.DataFrame:
    """Rows where two downloads of the same series disagree by more than ``tol``.

    ``primary`` and ``secondary`` are long frames with ``keys`` and ``value``
    columns (e.g. cboe_downloader.download_vix_term_structure_history and
    yf_downloader.download_vix_term_structure for the same window). Only NYSE
    sessions are compared. Read-only: nothing is stored or dropped.

    Returns rows [*keys, <labels[0]>, <labels[1]>, diff] with
    |diff| = |primary - secondary| > tol, sorted by key, and logs one
    warning per ticker. (Measured 2026-10-20: the pre-C23 stored term
    structure, downloaded from Yahoo, differs from Cboe's files by more than
    0.25 pts on 38 of 20,859 (date, ticker) rows; see
    data_store/market/corrections/.)
    """
    keys = list(keys)
    a, b = labels
    cols = [*keys, a, b, "diff"]
    if primary is None or secondary is None or primary.empty or secondary.empty:
        return pd.DataFrame(columns=cols)
    p = primary[[*keys, value]].copy()
    s = secondary[[*keys, value]].copy()
    for f in (p, s):
        f["date"] = _to_date_str(f["date"])
    p = drop_non_sessions(p.drop_duplicates(keys, keep="last"), f"compare_sources {a}")
    s = drop_non_sessions(s.drop_duplicates(keys, keep="last"), f"compare_sources {b}")
    m = p.merge(s, on=keys, suffixes=("_a", "_b"))
    m["diff"] = m[f"{value}_a"] - m[f"{value}_b"]
    out = m[m["diff"].abs() > tol].rename(columns={f"{value}_a": a, f"{value}_b": b})
    out = out[cols].sort_values(keys).reset_index(drop=True)
    if log and not out.empty:
        group = keys[1] if len(keys) > 1 else keys[0]
        for g, grp in out.groupby(group):
            logger.warning("%s vs %s %s: %d row(s) differ by more than %.2f: %s%s", a, b, g, len(grp), tol,
                           ", ".join(f"{r[0]} ({r[1]:.2f} vs {r[2]:.2f})"
                                     for r in grp[["date", a, b]].head(8).itertuples(index=False)),
                           " ..." if len(grp) > 8 else "")
    return out


def cross_check_vix(reference: pd.DataFrame, tol: float = 0.25,
                    start: str | None = None, end: str | None = None,
                    *, log: bool = True) -> pd.DataFrame:
    """Compare stored VIX-family closes with an independent reference.

    ``reference`` is a long frame [date, ticker, close] — normally Cboe's
    official index files via cboe_downloader.download_vix_term_structure_history().
    Checks vix_term_structure for every reference ticker it stores, and
    vix_daily against the reference '^VIX', on NYSE sessions inside both
    series' spans (and [start, end] if given).

    Returns rows [table, date, ticker, stored, reference, diff, issue] with
    issue in {'mismatch' (|diff| > tol), 'missing_in_db', 'missing_in_reference'}
    and logs a warning per (table, ticker, issue). Read-only.
    """
    ref = reference.copy()
    ref["date"] = _to_date_str(ref["date"])
    out = []
    with _connection() as conn:
        stored_ts = pd.read_sql_query("SELECT date, ticker, close FROM vix_term_structure", conn)
        stored_vd = pd.read_sql_query("SELECT date, close FROM vix_daily", conn)
    stored_vd["ticker"] = "^VIX"
    for table, stored in (("vix_term_structure", stored_ts), ("vix_daily", stored_vd)):
        for ticker in sorted(set(stored["ticker"]) & set(ref["ticker"])):
            s = stored[stored["ticker"] == ticker].set_index("date")["close"]
            r = ref[ref["ticker"] == ticker].drop_duplicates("date", keep="last").set_index("date")["close"]
            lo = max(s.index.min(), r.index.min(), start or "")
            hi = min(s.index.max(), r.index.max(), end or "9999-12-31")
            if lo > hi:
                continue
            sess = {_iso(d) for d in nyse_sessions(lo, hi)}
            s, r = s[s.index.isin(sess)], r[r.index.isin(sess)]
            both = s.index.intersection(r.index)
            diff = (s[both] - r[both])
            for d in diff.index[diff.abs() > tol]:
                out.append((table, d, ticker, float(s[d]), float(r[d]), float(diff[d]), "mismatch"))
            for d in sorted(set(r.index) - set(s.index)):
                out.append((table, d, ticker, np.nan, float(r[d]), np.nan, "missing_in_db"))
            for d in sorted(set(s.index) - set(r.index)):
                out.append((table, d, ticker, float(s[d]), np.nan, np.nan, "missing_in_reference"))
    res = pd.DataFrame(out, columns=["table", "date", "ticker", "stored", "reference", "diff", "issue"])
    if log and not res.empty:
        for (table, ticker, issue), grp in res.groupby(["table", "ticker", "issue"]):
            detail = ", ".join(
                f"{r.date} ({r.stored:.2f} vs {r.reference:.2f})" if issue == "mismatch" else str(r.date)
                for r in grp.head(8).itertuples())
            logger.warning("cross-check %s %s: %d %s (tol %.2f): %s%s", table, ticker, len(grp),
                           issue, tol, detail, " ..." if len(grp) > 8 else "")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# Classifier predictions (formerly the misleadingly named `regime_labels` table)
# ══════════════════════════════════════════════════════════════════════════════

def insert_classifier_predictions(df: pd.DataFrame) -> int:
    """Write XGBoost regime-classifier predictions [date, regime, confidence, vix_level, vix_slope, rv_20d]."""
    _ensure_schema()
    df = _normalise_date_column(df)
    rows = _records(df, ["date", "regime", "confidence", "vix_level", "vix_slope", "rv_20d"])
    sql = """
        INSERT OR REPLACE INTO classifier_predictions
            (date, regime, confidence, vix_level, vix_slope, rv_20d)
        VALUES
            (:date, :regime, :confidence, :vix_level, :vix_slope, :rv_20d)
    """
    with _connection() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def get_classifier_predictions(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """Return classifier predictions strictly up to as_of_date (look-ahead safe)."""
    _ensure_schema()
    return _get_dated("classifier_predictions", as_of_date, start_date)


def insert_regime_labels(df: pd.DataFrame) -> int:
    """Deprecated alias of insert_classifier_predictions.

    Despite the name this never stored the deterministic rule labels; it stores
    the demoted XGBoost classifier's predictions. Rule labels:
    data_store/signals/regime_labels.parquet (build_regime_labels).
    """
    return insert_classifier_predictions(df)


def get_regime_labels(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """Deprecated alias of get_classifier_predictions (classifier output, not rule labels)."""
    return get_classifier_predictions(as_of_date, start_date)


# ══════════════════════════════════════════════════════════════════════════════
# T-bill rates
# ══════════════════════════════════════════════════════════════════════════════

def insert_tbill_rates(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """
    Upsert 13-week T-bill daily rates.

    Parameters
    ----------
    df : DataFrame with columns [date, rate] and optionally discount_yield.
         rate must be a continuously compounded decimal (e.g. 0.045 for 4.5%);
         yf_downloader.download_tbill_rate() delivers it converted from ^IRX
         with tbill_discount_to_cc(). Zero and negative rates are valid.

    Returns
    -------
    int : number of rows inserted/replaced.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    df = df.drop_duplicates("date", keep="last").sort_values("date")
    if "discount_yield" not in df.columns:
        df["discount_yield"] = None
    wild = pd.to_numeric(df["rate"], errors="coerce").abs() > 0.25
    if wild.any():
        logger.warning("tbill_rates: %d rate(s) beyond +/-25%% — percent vs decimal mix-up? %s",
                       int(wild.sum()), ", ".join(df.loc[wild, "date"].head(5)))
    rows = _records(df, ["date", "rate", "discount_yield"])
    sql = """
        INSERT OR REPLACE INTO tbill_rates (date, rate, discount_yield)
        VALUES (:date, :rate, :discount_yield)
    """
    with _connection() as conn:
        if warn_dropped:
            _warn_dropped(conn, "tbill_rates", df, ["date"], "tbill_rates")
        conn.executemany(sql, rows)
    logger.info("Upserted %d T-bill rate rows", len(rows))
    return len(rows)


def get_tbill_rate(as_of_date: str, fallback: float = 0.045) -> float:
    """
    Return the 13-week T-bill rate as of as_of_date (most recent on or before).

    LOOK-AHEAD GUARD: date <= as_of_date strictly.

    Parameters
    ----------
    as_of_date : str 'YYYY-MM-DD' — upper bound (inclusive).
    fallback   : float — default rate if no data found (historical default 4.5%).

    Returns
    -------
    float : continuously compounded decimal rate (e.g. 0.045).
    """
    _validate_date(as_of_date)
    sql = """
        SELECT rate FROM tbill_rates
        WHERE date <= ?
        ORDER BY date DESC
        LIMIT 1
    """
    with _connection() as conn:
        row = conn.execute(sql, [as_of_date]).fetchone()
    if row is None:
        logger.warning("No T-bill rate found for %s — using fallback %.4f", as_of_date, fallback)
        return fallback
    return float(row["rate"])


def get_tbill_rates_series(as_of_date: str, start_date: str | None = None) -> pd.DataFrame:
    """
    Return full T-bill rate series up to as_of_date.

    LOOK-AHEAD GUARD: date <= as_of_date strictly.

    Returns
    -------
    DataFrame with columns [date, rate, discount_yield], sorted ascending by
    date; rate is continuously compounded.
    """
    return _get_dated("tbill_rates", as_of_date, start_date)


# ══════════════════════════════════════════════════════════════════════════════
# Other Cboe indices (VIX1D, VIX1Y, SKEW, COR1M, COR3M, DSPX, benchmarks)
# ══════════════════════════════════════════════════════════════════════════════

def insert_cboe_index(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """Upsert Cboe index closes [date, symbol, close, (open, high, low), (source)].

    Same rules as the VIX family: rows dated on days the NYSE was closed are
    not stored, and a stored Cboe row is only replaced by another Cboe row.
    Returns the rows written.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    if "symbol" not in df.columns:
        raise ValueError("cboe_index_daily rows need a 'symbol' column")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.lstrip("^")
    df = df.drop_duplicates(["date", "symbol"], keep="last")
    df = drop_non_sessions(df, "cboe_index_daily").copy()
    if df.empty:
        return 0
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = np.nan
    df["source"] = canonical_source(df["source"]) if "source" in df.columns else None
    df = df[pd.to_numeric(df["close"], errors="coerce").notna()]
    cols = ["date", "symbol", "open", "high", "low", "close", "source"]
    with _connection() as conn:
        if warn_dropped:
            for sym, sub in df.groupby("symbol"):
                stored = conn.execute("SELECT COUNT(*) FROM cboe_index_daily WHERE symbol = ?",
                                      (sym,)).fetchone()[0]
                if stored:
                    _warn_dropped_keyed(conn, "cboe_index_daily", sub, "symbol", str(sym))
        df = _skip_over_official(conn, "cboe_index_daily", df, ["date", "symbol"])
        rows = _records(df, cols)
        conn.executemany(  # the column names are the schema's own, the values are bound
            f"INSERT OR REPLACE INTO cboe_index_daily ({', '.join(cols)}) "  # noqa: S608
            f"VALUES ({', '.join(':' + c for c in cols)})", rows)
    logger.info("Upserted %d Cboe index rows (%s)", len(rows), ", ".join(sorted(df["symbol"].unique())))
    return len(rows)


def _warn_dropped_keyed(conn, table: str, incoming: pd.DataFrame, col: str, value: str) -> list:
    """_warn_dropped for one series of a (date, <col>)-keyed table."""
    lo, hi = incoming["date"].min(), incoming["date"].max()
    stored = {r[0] for r in conn.execute(  # table and column from the callers' constants
        f"SELECT date FROM {table} WHERE {col} = ? AND date BETWEEN ? AND ?", (value, lo, hi))}  # noqa: S608
    gone = sorted(stored - set(incoming["date"]))
    if gone:
        logger.warning("%s %s: source no longer returns %d stored row(s) in %s..%s (kept): %s%s",
                       table, value, len(gone), lo, hi, ", ".join(gone[:10]),
                       " ..." if len(gone) > 10 else "")
    return gone


def get_cboe_index(as_of_date: str, symbols: Iterable[str] | None = None,
                   start_date: str | None = None) -> pd.DataFrame:
    """Cboe index rows with date <= as_of_date (optionally from start_date), long format.

    LOOK-AHEAD GUARD: date <= as_of_date strictly. Columns [date (datetime),
    symbol, open, high, low, close, source], sorted by (date, symbol).
    """
    _validate_date(as_of_date)
    _ensure_schema()
    sql = "SELECT * FROM cboe_index_daily WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    if symbols:
        syms = [str(s).upper().lstrip("^") for s in symbols]
        sql += f" AND symbol IN ({', '.join('?' * len(syms))})"
        params += syms
    sql += " ORDER BY date ASC, symbol ASC"
    with _connection() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_cboe_index_wide(as_of_date: str, symbols: Iterable[str] | None = None,
                        start_date: str | None = None) -> pd.DataFrame:
    """get_cboe_index pivoted to one close column per symbol (index: date)."""
    long = get_cboe_index(as_of_date, symbols, start_date)
    if long.empty:
        return pd.DataFrame()
    # pivot, not pivot_table: (date, symbol) is the key, and a duplicate must raise, not be averaged
    wide = long.pivot(index="date", columns="symbol", values="close")  # noqa: PD010
    wide.columns.name = None
    return wide


# ══════════════════════════════════════════════════════════════════════════════
# Rate quotes (Treasury bills, par curve, SOFR): inputs of data/rates.py
# ══════════════════════════════════════════════════════════════════════════════

# A stored row from a higher-ranked source is replaced only by a row of the
# same or a higher rank: Treasury publishes the bill and par-curve quotes, and
# FRED republishes the par curve (DGS*) as a fallback.
_RATE_SOURCE_RANK = {"treasury": 2, "nyfed": 2, "fred": 1}


def insert_rates_daily(df: pd.DataFrame, *, warn_dropped: bool = True) -> int:
    """Upsert rate quotes [date, series, value, source]; value is the published quote / 100.

    Not session-filtered: the publishers' business days are not NYSE
    sessions. A stored Treasury row is not replaced by a FRED copy of it.
    """
    _ensure_schema()
    df = _normalise_date_column(df)
    if df.empty:
        return 0
    missing = {"series", "value", "source"} - set(df.columns)
    if missing:
        raise ValueError(f"rates_daily rows need columns {sorted(missing)}")
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).drop_duplicates(["date", "series"], keep="last")
    if df.empty:
        return 0
    wild = df["value"].abs() > 0.25
    if wild.any():
        logger.warning("rates_daily: %d value(s) beyond +/-25%% — percent stored as decimal? %s",
                       int(wild.sum()), ", ".join(df.loc[wild, "date"].head(5)))
    rank = df["source"].map(_RATE_SOURCE_RANK).fillna(0).to_numpy()
    with _connection() as conn:
        stored = pd.read_sql_query(
            "SELECT date, series, source FROM rates_daily WHERE date BETWEEN ? AND ?", conn,
            params=[df["date"].min(), df["date"].max()])
        if warn_dropped and not stored.empty:
            got = set(zip(df["date"], df["series"]))
            mine = stored[stored["series"].isin(df["series"].unique())
                          & stored["source"].isin(df["source"].unique())]
            gone = [f"{d}/{s}" for d, s in zip(mine["date"], mine["series"]) if (d, s) not in got]
            if gone:
                logger.warning("rates_daily: source no longer returns %d stored row(s) (kept): %s%s",
                               len(gone), ", ".join(gone[:10]), " ..." if len(gone) > 10 else "")
        if not stored.empty:
            have = {(d, s): _RATE_SOURCE_RANK.get(src, 0)
                    for d, s, src in stored.itertuples(index=False)}
            outranked = np.array([have.get((d, s), -1) > r
                                  for d, s, r in zip(df["date"], df["series"], rank)], dtype=bool)
            if outranked.any():
                logger.info("rates_daily: kept %d stored higher-ranked quote(s)", int(outranked.sum()))
                df = df[~outranked]
        rows = _records(df, ["date", "series", "value", "source"])
        conn.executemany("INSERT OR REPLACE INTO rates_daily (date, series, value, source) "
                         "VALUES (:date, :series, :value, :source)", rows)
    logger.info("Upserted %d rate quotes (%d series)", len(rows), df["series"].nunique())
    return len(rows)


def get_rates_daily(as_of_date: str, series: Iterable[str] | None = None,
                    start_date: str | None = None) -> pd.DataFrame:
    """Rate quotes dated on or before as_of_date, long format [date (str), series, value, source].

    LOOK-AHEAD GUARD: date <= as_of_date. This is the quote's own date; a
    series published the NEXT business day (SOFR, H.15) needs a stricter
    gate, which data/rates.py applies.
    """
    _validate_date(as_of_date)
    _ensure_schema()
    sql = "SELECT date, series, value, source FROM rates_daily WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    if series:
        ser = list(series)
        sql += f" AND series IN ({', '.join('?' * len(ser))})"
        params += ser
    sql += " ORDER BY date ASC, series ASC"
    with _connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


# ══════════════════════════════════════════════════════════════════════════════
# Persistence: parquet partitions the daily bot commits
# ══════════════════════════════════════════════════════════════════════════════

_MARKET_TABLES = {
    # table: (natural key, persisted columns, column dtypes)
    "spx_ohlcv": (["date"], ["date", "open", "high", "low", "close", "volume"],
                  {"open": "float64", "high": "float64", "low": "float64", "close": "float64",
                   "volume": "int64"}),
    "vix_daily": (["date"], ["date", "open", "high", "low", "close", "source"],
                  {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}),
    "vix_term_structure": (["date", "ticker"], ["date", "ticker", "tenor_days", "close", "is_vvix", "source"],
                           {"tenor_days": "int64", "close": "float64", "is_vvix": "int64"}),
    "tbill_rates": (["date"], ["date", "rate", "discount_yield"],
                    {"rate": "float64", "discount_yield": "float64"}),
    "cboe_index_daily": (["date", "symbol"], ["date", "symbol", "open", "high", "low", "close", "source"],
                         {"open": "float64", "high": "float64", "low": "float64", "close": "float64"}),
    "rates_daily": (["date", "series"], ["date", "series", "value", "source"], {"value": "float64"}),
}
# What export/import_market_partitions handle by default — the four tables
# the daily bot restores and commits. The C23 reference tables are opt-in
# (tables=REFERENCE_TABLES), so the bot's restore/persist steps are unchanged
# until it is wired to download them.
MARKET_TABLES = ("spx_ohlcv", "vix_daily", "vix_term_structure", "tbill_rates")
REFERENCE_TABLES = ("cboe_index_daily", "rates_daily")

# Tables keyed by NYSE session: their writers never store, and the export
# never writes, a row dated on a day the NYSE was closed.
SESSION_KEYED_TABLES = ("vix_daily", "vix_term_structure", "cboe_index_daily")


def _canonical(df: pd.DataFrame, cols: Sequence[str], dtypes: dict, keys: Sequence[str]) -> pd.DataFrame:
    df = df[list(cols)].copy()
    for c in cols:
        if c in dtypes:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(dtypes[c])
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.sort_values(list(keys)).reset_index(drop=True)


# Relative tolerance under which two float values in a persisted file count as
# the same value. Derived columns (tbill_rates.rate = -ln(1 - d*91/360)*365/91)
# differ in the last bit between runner CPUs: in 61f7252 three
# tbill_rates_2025 rows changed by 6.9e-18 and the file was rewritten, and
# both 2026-10-20 book runs flagged it as changed again. Market prints carry
# at most 6 significant decimals, so 1e-12 can never hide a real revision.
_PARQUET_FLOAT_RTOL = 1e-12


def _same_content(old: pd.DataFrame, new: pd.DataFrame, rtol: float = _PARQUET_FLOAT_RTOL) -> bool:
    """True if ``old`` and ``new`` hold the same rows: float columns to ``rtol``
    (NaN equals NaN), every other column exactly (compared as text)."""
    if list(old.columns) != list(new.columns) or len(old) != len(new):
        return False
    old, new = old.reset_index(drop=True), new.reset_index(drop=True)
    for col in new.columns:
        a, b = old[col], new[col]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                return False
            ok = ~np.isnan(x)
            if not np.allclose(x[ok], y[ok], rtol=rtol, atol=1e-15):
                return False
        elif a.astype(str).tolist() != b.astype(str).tolist():
            return False
    return True


def _write_parquet_if_changed(df: pd.DataFrame, path: Path) -> bool:
    """Write ``df`` to ``path`` (zstd) unless the file already holds this content
    (_same_content: floats to a relative 1e-12, everything else exactly)."""
    if path.exists():
        try:
            old = pd.read_parquet(path)
            if _same_content(old, df):
                return False
        except Exception:   # unreadable file: rewrite it
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)
    return True


def export_market_partitions(root: Path | None = None,
                             tables: Iterable[str] | None = None,
                             since: str | None = None) -> list:
    """Persist market tables as yearly parquet partitions.

    Layout: ``<root>/<table>/<table>_<YYYY>.parquet`` (root defaults to
    MARKET_DIR = data_store/market). Only years with rows on/after ``since``
    are touched. Each partition is MERGED with the file already there: on a
    key clash the database row wins, and rows the file holds that the database
    lacks are KEPT — so a session a remote source later drops, or a database
    rebuilt from a shorter download, can never erase what was captured. The
    one exception: in the session-keyed tables (SESSION_KEYED_TABLES) a row
    dated on a day the NYSE was closed is not a capture of a session, and is
    left out of the file whichever side holds it. A file is rewritten only
    if its content changes (_same_content). ``tables`` defaults to
    MARKET_TABLES; pass REFERENCE_TABLES (or any _MARKET_TABLES name) to
    persist the C23 reference tables. Returns the paths written.
    """
    root = Path(root) if root is not None else MARKET_DIR
    written = []
    for table in (tables or MARKET_TABLES):
        keys, cols, dtypes = _MARKET_TABLES[table]
        sql = f"SELECT {', '.join(cols)} FROM {table}"
        params: list = []
        if since:
            _validate_date(since)
            sql += " WHERE date >= ?"
            params.append(f"{since[:4]}-01-01")
        with _connection() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if df.empty:
            continue
        df = _canonical(df, cols, dtypes, keys)
        session_keyed = table in SESSION_KEYED_TABLES
        if session_keyed:
            df = drop_non_sessions(df, f"export {table}")
        for year, part in df.groupby(df["date"].str[:4]):
            path = root / table / f"{table}_{year}.parquet"
            if path.exists():
                old = _canonical(pd.read_parquet(path).reindex(columns=cols), cols, dtypes, keys)
                if session_keyed:
                    old = drop_non_sessions(old, f"{path.name}")
                part = pd.concat([old, part], ignore_index=True)
                part = _canonical(part.drop_duplicates(keys, keep="last"), cols, dtypes, keys)
            if _write_parquet_if_changed(part, path):
                written.append(path)
    logger.info("Exported %d market partition file(s) under %s", len(written), root)
    return written


def read_market_partitions(table: str, root: Path | None = None) -> pd.DataFrame:
    """Every persisted row of ``table`` (all years), canonical columns and dtypes; empty if none."""
    root = Path(root) if root is not None else MARKET_DIR
    keys, cols, dtypes = _MARKET_TABLES[table]
    files = sorted((root / table).glob(f"{table}_*.parquet"))
    if not files:
        return pd.DataFrame(columns=cols)
    df = pd.concat([pd.read_parquet(f).reindex(columns=cols) for f in files], ignore_index=True)
    return _canonical(df, cols, dtypes, keys)


def replace_market_partitions(table: str, df: pd.DataFrame, root: Path | None = None) -> list:
    """Make the persisted partitions of ``table`` hold exactly ``df``'s rows.

    FOR DELIBERATE, SCRIPTED CORRECTIONS ONLY (pipeline.correct_vix_index_family):
    unlike export_market_partitions this does not merge. Rows a file holds
    that ``df`` lacks are removed, and the file of a year with no row left in
    ``df`` is deleted. ``df`` must therefore be the WHOLE corrected table; an
    empty ``df`` is refused. Returns the paths rewritten or deleted.
    """
    root = Path(root) if root is not None else MARKET_DIR
    keys, cols, dtypes = _MARKET_TABLES[table]
    if df is None or df.empty:
        raise ValueError(f"replace_market_partitions({table}): refusing to empty the record")
    df = _canonical(df.drop_duplicates(list(keys), keep="last"), cols, dtypes, keys)
    written = []
    years = set()
    for year, part in df.groupby(df["date"].str[:4]):
        years.add(year)
        path = root / table / f"{table}_{year}.parquet"
        if _write_parquet_if_changed(part.reset_index(drop=True), path):
            written.append(path)
    for path in sorted((root / table).glob(f"{table}_*.parquet")):
        if path.stem.rsplit("_", 1)[-1] not in years:
            path.unlink()
            written.append(path)
            logger.warning("replace_market_partitions(%s): deleted %s (no row left for that year)",
                           table, path.name)
    logger.info("replace_market_partitions(%s): %d file(s) rewritten under %s", table, len(written), root)
    return written


def import_market_partitions(root: Path | None = None,
                             tables: Iterable[str] | None = None) -> dict:
    """Upsert every persisted market partition under ``root`` into the database.

    Use at the start of a run, before downloading: the files are the record of
    what earlier runs captured. ``tables`` defaults to MARKET_TABLES. Returns
    {table: rows upserted}.
    """
    root = Path(root) if root is not None else MARKET_DIR
    inserters = {
        "spx_ohlcv": insert_spx_ohlcv,
        "vix_daily": insert_vix_daily,
        "vix_term_structure": insert_vix_term_structure,
        "tbill_rates": insert_tbill_rates,
        "cboe_index_daily": insert_cboe_index,
        "rates_daily": insert_rates_daily,
    }
    counts = {}
    for table in (tables or MARKET_TABLES):
        files = sorted((root / table).glob(f"{table}_*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        counts[table] = inserters[table](df, warn_dropped=False)
    return counts


def options_snapshot_path(snapshot_date: str, underlying: str, root: Path | None = None) -> Path:
    root = Path(root) if root is not None else OPTIONS_DIR
    return root / f"{underlying.lower()}_{snapshot_date}.parquet"


def export_options_snapshot(snapshot_date: str, underlying: str,
                            root: Path | None = None, *,
                            moneyness: tuple | None = None,
                            max_years: float | None = None,
                            two_sided_only: bool = False) -> Path | None:
    """Persist one stored chain to ``<root>/<underlying>_<snapshot_date>.parquet``.

    All columns except the surrogate id, sorted by (root, expiry, right,
    strike). By default the whole chain is kept (a full Cboe SPX chain,
    ~30k rows, is ~0.85 MB as zstd parquet). Optional size filters, for a
    caller that commits one file per session: ``moneyness=(lo, hi)`` keeps
    strikes within [lo, hi] x underlying_price (rows without an underlying
    price are kept), ``max_years`` caps time_to_expiry, and
    ``two_sided_only`` keeps rows with bid > 0 and ask > 0. Returns the path
    if written (None if there is no such chain or the file already holds it).
    """
    _validate_date(snapshot_date)
    with _connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM options_snapshots WHERE snapshot_date = ? AND underlying = ?",
            conn, params=[snapshot_date, underlying.upper()])
    if df.empty:
        return None
    if moneyness is not None:
        lo, hi = moneyness
        k = df["strike"] / df["underlying_price"]
        df = df[df["underlying_price"].isna() | k.between(lo, hi)]
    if max_years is not None:
        df = df[df["time_to_expiry"] <= max_years]
    if two_sided_only:
        df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df.drop(columns=["id"]).sort_values(["root", "expiry", "right", "strike"]).reset_index(drop=True)
    path = options_snapshot_path(snapshot_date, underlying, root)
    return path if _write_parquet_if_changed(df, path) else None


def export_options_snapshots(root: Path | None = None, since: str | None = None,
                             **filters) -> list:
    """``export_options_snapshot`` for every stored chain (optionally only sessions >= since).

    ``filters`` (moneyness, max_years, two_sided_only) are passed through.
    """
    with _connection() as conn:
        pairs = conn.execute(
            "SELECT DISTINCT snapshot_date, underlying FROM options_snapshots "
            "WHERE snapshot_date >= ? ORDER BY snapshot_date, underlying", (since or "",)
        ).fetchall()
    out = []
    for snap, und in pairs:
        p = export_options_snapshot(snap, und, root, **filters)
        if p is not None:
            out.append(p)
    return out


def import_options_snapshots(root: Path | None = None, overwrite: bool = False) -> int:
    """Load persisted chains into the database.

    A chain the database already holds for that (session, underlying) is left
    alone unless ``overwrite``. Returns rows inserted.
    """
    root = Path(root) if root is not None else OPTIONS_DIR
    have = set()
    with _connection() as conn:
        have = {(r[0], r[1]) for r in conn.execute(
            "SELECT DISTINCT snapshot_date, underlying FROM options_snapshots")}
    n = 0
    for path in sorted(root.glob("*_*.parquet")):
        m = re.match(r"^([a-z]+)_(\d{4}-\d{2}-\d{2})$", path.stem)
        if not m:
            continue
        und, snap = m.group(1).upper(), m.group(2)
        if (snap, und) in have and not overwrite:
            continue
        n += insert_options_snapshot(pd.read_parquet(path), und)
    return n


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _to_date_str(values) -> pd.Series:
    """Datetime-likes -> 'YYYY-MM-DD'. Tz-aware instants are converted to New York time first."""
    s = pd.Series(values) if not isinstance(values, pd.Series) else values
    try:
        dt = pd.to_datetime(s)
    except (ValueError, TypeError):
        dt = pd.to_datetime(s, utc=True)       # mixed UTC offsets
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(NY_TZ)
    return dt.dt.strftime("%Y-%m-%d")


def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a string 'date' column in 'YYYY-MM-DD' format (NY calendar date).

    Accepts a 'date' (or 'Date') column, or a DatetimeIndex / index named
    date/Date (as yfinance returns). Tz-aware timestamps are instants and are
    converted to America/New_York before the date is taken, so an evening NY
    print stamped in UTC (after 00:00 UTC) keeps its session date.
    """
    df = df.copy()
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("date", "Date", "Datetime"):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
        else:
            raise ValueError("DataFrame has no 'date' column and index is not DatetimeIndex")
    if df.empty:
        df["date"] = df["date"].astype(str)
        return df
    df["date"] = _to_date_str(df["date"]).to_numpy()
    return df


def _validate_date(date_str: str) -> None:
    """Raise ValueError if date_str is not a valid YYYY-MM-DD string."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")


def get_data_coverage() -> dict:
    """
    Return a summary of what data is loaded in the database.
    Useful for quickly checking data completeness before a backtest run, and
    the source for any published data table.

    Keys: one per time-series table ({rows, min_date, max_date}); per
    underlying ``options_spx`` / ``options_vix`` ({snapshots, min_date,
    max_date, per_snapshot: {date: {rows, source}}}); and
    ``vix_term_structure_by_ticker`` ({ticker: {rows, min_date, max_date,
    tenor_days}}).
    """
    _ensure_schema()
    with _connection() as conn:
        summary = {}
        for table, col in [
            ("spx_ohlcv",              "date"),
            ("vix_daily",              "date"),
            ("vix_term_structure",     "date"),
            ("vix_futures_daily",      "date"),
            ("vix_futures_settlements","settlement_date"),
            ("tbill_rates",            "date"),
            ("classifier_predictions", "date"),
            ("cboe_index_daily",       "date"),
            ("rates_daily",            "date"),
        ]:
            row = conn.execute(
                f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {table}"
            ).fetchone()
            summary[table] = {
                "rows": row[0],
                "min_date": row[1],
                "max_date": row[2],
            }

        summary["vix_term_structure_by_ticker"] = {
            r[0]: {"rows": r[1], "min_date": r[2], "max_date": r[3], "tenor_days": r[4]}
            for r in conn.execute(
                "SELECT ticker, COUNT(*), MIN(date), MAX(date), MAX(tenor_days) "
                "FROM vix_term_structure GROUP BY ticker ORDER BY MAX(tenor_days), ticker")
        }

        for und in ("SPX", "VIX"):
            row = conn.execute(
                "SELECT COUNT(DISTINCT snapshot_date), MIN(snapshot_date), MAX(snapshot_date) "
                "FROM options_snapshots WHERE underlying=?",
                (und,)
            ).fetchone()
            per = {
                r[0]: {"rows": r[1], "source": r[2]}
                for r in conn.execute(
                    "SELECT snapshot_date, COUNT(*), GROUP_CONCAT(DISTINCT source) "
                    "FROM options_snapshots WHERE underlying=? GROUP BY snapshot_date "
                    "ORDER BY snapshot_date", (und,))
            }
            summary[f"options_{und.lower()}"] = {
                "snapshots": row[0],
                "min_date": row[1],
                "max_date": row[2],
                "per_snapshot": per,
            }
    return summary
