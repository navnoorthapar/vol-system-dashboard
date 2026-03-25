"""
database.py — SQLite schema and all read/write operations.

Design philosophy:
- EVERY read function accepts an `as_of_date` parameter.
  Data returned is ALWAYS filtered to `date <= as_of_date`.
  This is the single enforcement point for zero look-ahead bias.
- Write functions are idempotent (INSERT OR IGNORE / INSERT OR REPLACE).
- Parquet files mirror SQLite for options chains (faster columnar reads).

Tables:
  spx_ohlcv              — SPX daily OHLCV from Yahoo Finance
  vix_daily              — VIX index daily levels from CBOE
  vix_futures_settlements — VIX futures final settlement prices
  options_snapshots      — SPX/VIX options chains keyed by snapshot_date
"""

import sqlite3
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from joint_vol_calibration.config import DB_PATH, PARQUET_DIR

logger = logging.getLogger(__name__)


# ── Schema DDL ────────────────────────────────────────────────────────────────

_DDL = """
-- SPX daily OHLCV (Yahoo Finance)
CREATE TABLE IF NOT EXISTS spx_ohlcv (
    date        TEXT PRIMARY KEY,   -- YYYY-MM-DD
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL,
    volume      INTEGER NOT NULL,
    log_return  REAL                -- ln(close_t / close_{t-1}), NULL for first row
);

-- VIX index daily levels (CBOE)
CREATE TABLE IF NOT EXISTS vix_daily (
    date        TEXT PRIMARY KEY,
    open        REAL NOT NULL,
    high        REAL NOT NULL,
    low         REAL NOT NULL,
    close       REAL NOT NULL
);

-- VIX futures settlement prices (CBOE)
-- Each row = one futures contract's final settlement on its settlement_date.
-- expiry_month: YYYY-MM (e.g. "2024-01") — identifies the contract
CREATE TABLE IF NOT EXISTS vix_futures_settlements (
    settlement_date TEXT NOT NULL,
    expiry_month    TEXT NOT NULL,
    settlement_price REAL NOT NULL,
    PRIMARY KEY (settlement_date, expiry_month)
);

-- VIX futures daily price history (end-of-day, multiple contracts)
CREATE TABLE IF NOT EXISTS vix_futures_daily (
    date            TEXT NOT NULL,
    expiry_month    TEXT NOT NULL,  -- YYYY-MM
    open            REAL,
    high            REAL,
    low             REAL,
    close           REAL NOT NULL,
    volume          INTEGER,
    open_interest   INTEGER,
    days_to_expiry  INTEGER,        -- calendar days from date to expiry
    PRIMARY KEY (date, expiry_month)
);

-- Options chain snapshots (SPX and VIX)
-- snapshot_date: the trading date this chain was pulled on
-- underlying: 'SPX' or 'VIX'
-- expiry: option expiration YYYY-MM-DD
-- strike, right, mid_price, implied_vol stored per row
CREATE TABLE IF NOT EXISTS options_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date   TEXT NOT NULL,
    underlying      TEXT NOT NULL,
    expiry          TEXT NOT NULL,
    strike          REAL NOT NULL,
    right           TEXT NOT NULL,  -- 'C' or 'P'
    bid             REAL,
    ask             REAL,
    mid_price       REAL,
    implied_vol     REAL,           -- market IV, annualised decimal
    open_interest   INTEGER,
    volume          INTEGER,
    delta           REAL,
    gamma           REAL,
    vega            REAL,
    theta           REAL,
    time_to_expiry  REAL            -- years to expiry from snapshot_date
);
CREATE INDEX IF NOT EXISTS idx_options_snap_date_und
    ON options_snapshots(snapshot_date, underlying);

-- VIX term structure indices (^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^VVIX)
-- Primary free proxy for VIX futures term structure (CBOE CDN is Cloudflare-blocked)
CREATE TABLE IF NOT EXISTS vix_term_structure (
    date        TEXT NOT NULL,
    ticker      TEXT NOT NULL,
    tenor_days  INTEGER NOT NULL,
    close       REAL NOT NULL,
    is_vvix     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (date, ticker)
);
CREATE INDEX IF NOT EXISTS idx_vts_date ON vix_term_structure(date);

-- Regime labels (written by C8 classifier)
CREATE TABLE IF NOT EXISTS regime_labels (
    date            TEXT PRIMARY KEY,
    regime          INTEGER NOT NULL,  -- 0=A, 1=B, 2=C
    confidence      REAL NOT NULL,
    vix_level       REAL,
    vix_slope       REAL,
    rv_20d          REAL
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Return a connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """
    Create all tables if they don't exist.
    Safe to call multiple times (idempotent).
    """
    with _connect() as conn:
        conn.executescript(_DDL)
    logger.info("Database initialised at %s", DB_PATH)


# ── SPX OHLCV ─────────────────────────────────────────────────────────────────

def insert_spx_ohlcv(df: pd.DataFrame) -> int:
    """
    Upsert SPX daily OHLCV rows.

    Parameters
    ----------
    df : DataFrame with columns [date, open, high, low, close, volume].
         date must be a string 'YYYY-MM-DD' or datetime-like.

    Returns
    -------
    int : number of rows inserted/replaced.
    """
    df = _normalise_date_column(df)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    rows = df[["date", "open", "high", "low", "close", "volume", "log_return"]].to_dict("records")
    sql = """
        INSERT OR REPLACE INTO spx_ohlcv
            (date, open, high, low, close, volume, log_return)
        VALUES
            (:date, :open, :high, :low, :close, :volume, :log_return)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d SPX OHLCV rows", len(rows))
    return len(rows)


def get_spx_ohlcv(as_of_date: str, start_date: Optional[str] = None) -> pd.DataFrame:
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
    _validate_date(as_of_date)
    sql = "SELECT * FROM spx_ohlcv WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    sql += " ORDER BY date ASC"
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── VIX Daily ─────────────────────────────────────────────────────────────────

def insert_vix_daily(df: pd.DataFrame) -> int:
    """Upsert VIX index daily levels from CBOE CSV."""
    df = _normalise_date_column(df)
    rows = df[["date", "open", "high", "low", "close"]].to_dict("records")
    sql = """
        INSERT OR REPLACE INTO vix_daily (date, open, high, low, close)
        VALUES (:date, :open, :high, :low, :close)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX daily rows", len(rows))
    return len(rows)


def get_vix_daily(as_of_date: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return VIX daily levels up to as_of_date.

    LOOK-AHEAD GUARD: strictly enforced.
    """
    _validate_date(as_of_date)
    sql = "SELECT * FROM vix_daily WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    sql += " ORDER BY date ASC"
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── VIX Futures ───────────────────────────────────────────────────────────────

def insert_vix_futures_settlements(df: pd.DataFrame) -> int:
    """
    Upsert VIX futures final settlement prices.

    df columns: [settlement_date, expiry_month, settlement_price]
    expiry_month format: 'YYYY-MM'
    """
    df = df.copy()
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.strftime("%Y-%m-%d")
    rows = df[["settlement_date", "expiry_month", "settlement_price"]].to_dict("records")
    sql = """
        INSERT OR REPLACE INTO vix_futures_settlements
            (settlement_date, expiry_month, settlement_price)
        VALUES (:settlement_date, :expiry_month, :settlement_price)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX futures settlement rows", len(rows))
    return len(rows)


def insert_vix_futures_daily(df: pd.DataFrame) -> int:
    """
    Upsert VIX futures end-of-day prices for multiple contracts.

    df columns: [date, expiry_month, open, high, low, close, volume,
                 open_interest, days_to_expiry]
    """
    df = _normalise_date_column(df)
    rows = df.to_dict("records")
    sql = """
        INSERT OR REPLACE INTO vix_futures_daily
            (date, expiry_month, open, high, low, close,
             volume, open_interest, days_to_expiry)
        VALUES (:date, :expiry_month, :open, :high, :low, :close,
                :volume, :open_interest, :days_to_expiry)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX futures daily rows", len(rows))
    return len(rows)


def get_vix_futures_curve(as_of_date: str) -> pd.DataFrame:
    """
    Return the VIX futures term structure as of as_of_date.

    Fetches all contracts with known daily prices on exactly as_of_date,
    sorted by days_to_expiry ascending (front month first).

    LOOK-AHEAD GUARD: only prices from <= as_of_date used.
    Settlement prices (final) also never bleed forward because they are
    stored with their actual settlement_date.

    Returns
    -------
    DataFrame with columns [expiry_month, close, days_to_expiry].
    """
    _validate_date(as_of_date)
    sql = """
        SELECT expiry_month, close, days_to_expiry
        FROM vix_futures_daily
        WHERE date = ?
        ORDER BY days_to_expiry ASC
    """
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=[as_of_date])
    return df


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
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=[as_of_date])
    return df


# ── Options Snapshots ─────────────────────────────────────────────────────────

def insert_options_snapshot(df: pd.DataFrame, underlying: str) -> int:
    """
    Insert a full options chain snapshot.

    Parameters
    ----------
    df : DataFrame with columns matching options_snapshots table.
         snapshot_date must already be set in the dataframe.
    underlying : 'SPX' or 'VIX'

    Returns
    -------
    int : rows inserted.
    """
    df = df.copy()
    df["underlying"] = underlying.upper()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")

    cols = [
        "snapshot_date", "underlying", "expiry", "strike", "right",
        "bid", "ask", "mid_price", "implied_vol",
        "open_interest", "volume", "delta", "gamma", "vega", "theta",
        "time_to_expiry",
    ]
    # Only keep columns that exist in df
    cols = [c for c in cols if c in df.columns]
    rows = df[cols].to_dict("records")

    placeholders = ", ".join(f":{c}" for c in cols)
    col_names = ", ".join(cols)
    sql = f"INSERT INTO options_snapshots ({col_names}) VALUES ({placeholders})"

    with _connect() as conn:
        conn.executemany(sql, rows)

    # Mirror to parquet for fast bulk reads
    parquet_path = PARQUET_DIR / f"options_{underlying.lower()}_{df['snapshot_date'].iloc[0]}.parquet"
    df[cols].to_parquet(parquet_path, index=False)

    logger.info("Inserted %d %s options rows for %s", len(rows), underlying, df["snapshot_date"].iloc[0])
    return len(rows)


def get_options_surface(as_of_date: str, underlying: str) -> pd.DataFrame:
    """
    Return the full options surface for a given underlying as of a date.

    LOOK-AHEAD GUARD: snapshot_date <= as_of_date AND expiry > as_of_date
    (we only return options that haven't expired yet as of the query date,
    AND whose price was observed no later than as_of_date).

    Parameters
    ----------
    as_of_date : str 'YYYY-MM-DD'
    underlying : 'SPX' or 'VIX'

    Returns
    -------
    DataFrame with full chain, sorted by expiry, strike.
    """
    _validate_date(as_of_date)
    # Most recent snapshot on or before as_of_date
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
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


# ── VIX Term Structure ────────────────────────────────────────────────────────

def insert_vix_term_structure(df: pd.DataFrame) -> int:
    """Upsert VIX term structure index rows."""
    df = _normalise_date_column(df)
    rows = df[["date", "ticker", "tenor_days", "close", "is_vvix"]].to_dict("records")
    sql = """
        INSERT OR REPLACE INTO vix_term_structure
            (date, ticker, tenor_days, close, is_vvix)
        VALUES (:date, :ticker, :tenor_days, :close, :is_vvix)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    logger.info("Upserted %d VIX term structure rows", len(rows))
    return len(rows)


def get_vix_term_structure(as_of_date: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return VIX term structure up to as_of_date.

    LOOK-AHEAD GUARD: date <= as_of_date strictly.

    Returns wide-format DataFrame: columns = ticker symbols,
    index = date. Each cell = closing level.
    """
    _validate_date(as_of_date)
    sql = "SELECT * FROM vix_term_structure WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        _validate_date(start_date)
        sql += " AND date >= ?"
        params.append(start_date)
    sql += " ORDER BY date ASC, tenor_days ASC"
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_vix_term_structure_wide(as_of_date: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Return VIX term structure in wide format: one column per ticker.

    Columns: [^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^VVIX] (plus derived slope).
    Useful for regime classifier feature construction.
    """
    df_long = get_vix_term_structure(as_of_date, start_date)
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


# ── Regime Labels ─────────────────────────────────────────────────────────────

def insert_regime_labels(df: pd.DataFrame) -> int:
    """Write regime classifications from Component 8 classifier."""
    df = _normalise_date_column(df)
    rows = df.to_dict("records")
    sql = """
        INSERT OR REPLACE INTO regime_labels
            (date, regime, confidence, vix_level, vix_slope, rv_20d)
        VALUES
            (:date, :regime, :confidence, :vix_level, :vix_slope, :rv_20d)
    """
    with _connect() as conn:
        conn.executemany(sql, rows)
    return len(rows)


def get_regime_labels(as_of_date: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Return regime labels strictly up to as_of_date (look-ahead safe)."""
    _validate_date(as_of_date)
    sql = "SELECT * FROM regime_labels WHERE date <= ?"
    params: list = [as_of_date]
    if start_date:
        sql += " AND date >= ?"
        params.append(start_date)
    sql += " ORDER BY date ASC"
    with _connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalise_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a string 'date' column in 'YYYY-MM-DD' format."""
    df = df.copy()
    if "date" not in df.columns:
        if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={"index": "date"}, inplace=True)
        else:
            raise ValueError("DataFrame has no 'date' column and index is not DatetimeIndex")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def _validate_date(date_str: str) -> None:
    """Raise ValueError if date_str is not a valid YYYY-MM-DD string."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")


def get_data_coverage() -> dict:
    """
    Return a summary of what data is loaded in the database.
    Useful for quickly checking data completeness before a backtest run.
    """
    with _connect() as conn:
        summary = {}
        for table, col in [
            ("spx_ohlcv",              "date"),
            ("vix_daily",              "date"),
            ("vix_term_structure",     "date"),
            ("vix_futures_daily",      "date"),
            ("vix_futures_settlements","settlement_date"),
        ]:
            row = conn.execute(
                f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {table}"
            ).fetchone()
            summary[table] = {
                "rows": row[0],
                "min_date": row[1],
                "max_date": row[2],
            }

        for und in ("SPX", "VIX"):
            row = conn.execute(
                "SELECT COUNT(DISTINCT snapshot_date), MIN(snapshot_date), MAX(snapshot_date) "
                "FROM options_snapshots WHERE underlying=?",
                (und,)
            ).fetchone()
            summary[f"options_{und.lower()}"] = {
                "snapshots": row[0],
                "min_date": row[1],
                "max_date": row[2],
            }
    return summary
