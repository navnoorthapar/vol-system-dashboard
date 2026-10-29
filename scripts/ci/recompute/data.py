"""
data.py: the recompute audit's own readers of data_store and its own math.

Read-only. The site renders from the committed record (app.record_db): the
SQLite cache data_store/vol_system.db with the parquet record (market
partitions under data_store/market, option chains under data_store/options)
upserted over it. Store.sql queries the same view, built here by its own code
in an in-memory database: every cache table, each market table with the
record's rows replacing the cache's on the record's keys (SPX log returns
recomputed from the closes), and each recorded chain the cache lacks.
cache_vs_record reports how far the cache lags the record. The NYSE calendar
and the SVI and Black formulas are written here again from their definitions,
not imported from the dashboard.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pickle
import sqlite3
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm as _norm

DATE_FMT = "%Y-%m-%d"
# the market tables the record holds, with their keys
RECORD_KEYS = {"spx_ohlcv": ("date",), "vix_daily": ("date",), "vix_term_structure": ("date", "ticker"),
               "tbill_rates": ("date",)}
# a cached value may differ from the record by float32 noise (Yahoo's provisional print kept
# until Cboe's official one replaces it in the record), never by a printed digit
CACHE_ATOL = 1e-4


class Store:
    """Lazy, cached reads of one data_store directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, Any] = {}

    # ── files ────────────────────────────────────────────────────────────────
    def json(self, rel: str) -> Any:
        key = "json:" + rel
        if key not in self._cache:
            self._cache[key] = json.loads((self.root / rel).read_text())
        return self._cache[key]

    def pickle(self, rel: str) -> Any:
        key = "pkl:" + rel
        if key not in self._cache:
            with open(self.root / rel, "rb") as f:
                self._cache[key] = pickle.load(f)  # noqa: S301  (the repository's own artifacts)
        return self._cache[key]

    def parquet(self, rel: str, **kw: Any) -> pd.DataFrame:
        key = "pq:" + rel + repr(sorted(kw.items()))
        if key not in self._cache:
            self._cache[key] = pd.read_parquet(self.root / rel, **kw)
        return self._cache[key]

    def sql(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """``query`` on the cache brought up to the committed record (record_view)."""
        key = "sql:" + query + repr(params)
        if key not in self._cache:
            if "db" not in self._cache:
                self._cache["db"] = record_view(self.root)
            self._cache[key] = pd.read_sql_query(query, self._cache["db"], params=params)
        return self._cache[key]

    def cache_sql(self, query: str, params: tuple = ()) -> pd.DataFrame:
        """``query`` on the checkout's SQLite cache as it is (no record applied)."""
        con = sqlite3.connect(f"file:{self.root / 'vol_system.db'}?mode=ro", uri=True)
        try:
            return pd.read_sql_query(query, con, params=params)
        finally:
            con.close()

    # ── market series ─────────────────────────────────────────────────────────
    def spx(self) -> pd.DataFrame:
        df = self.sql("SELECT date, open, high, low, close, log_return FROM spx_ohlcv ORDER BY date")
        df["date"] = df["date"].astype(str).str[:10]
        return df.set_index("date")

    def vix_close(self) -> pd.Series:
        df = self.sql("SELECT date, close FROM vix_daily ORDER BY date")
        return pd.Series(df["close"].to_numpy(float), index=df["date"].astype(str).str[:10], name="VIX")

    def vix_family(self) -> pd.DataFrame:
        """vix_term_structure wide: one column per ticker (^VIX9D, ^VIX, ^VIX3M, ^VIX6M, ^VVIX)."""
        df = self.sql("SELECT date, ticker, close FROM vix_term_structure ORDER BY date")
        df["date"] = df["date"].astype(str).str[:10]
        return df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()

    def vx_futures(self) -> pd.DataFrame:
        df = self.sql("SELECT date, expiry_date, expiry_month, symbol, is_weekly, close, settle "
                      "FROM vix_futures_daily ORDER BY date, expiry_date")
        df["date"] = df["date"].astype(str).str[:10]
        df["expiry_date"] = df["expiry_date"].astype(str).str[:10]
        return df

    def surface_sessions(self) -> list[str]:
        root = self.root / "surface"
        return sorted(p.name for p in root.iterdir()
                      if p.is_dir() and len(p.name) == 10 and (p / "residuals.parquet").exists()
                      and (p / "surface.json").exists())


def _utc_minutes(ts: Any) -> str:
    t = str(ts)
    if t.endswith(("Z", "+00:00")) and len(t) >= 16 and t[10] == "T":
        return f"{t[:10]} {t[11:16]} UTC"
    return t.replace("T", " ")


def chain_words(inputs: dict, universe: dict | None = None) -> tuple[str, str, list[str]] | None:
    """(name, time clause, the clause's badge parts) a page must use for a stored
    chain, from surface.json inputs:
    a Yahoo chain has no quote times, so one whose quote_time holds the job's fetch
    instant is a download 'fetched …' (with 'T measured from the fetch' when its T
    origin was that instant), one without is a close snapshot with 'no quote time';
    a Cboe book prints 'quotes at …' its recorded quote time."""
    uni = universe or {}
    src = str(inputs.get("source") or uni.get("source") or "").strip().lower()
    if not src:
        return None
    qt = inputs.get("quote_time") or uni.get("quote_time")
    if src in ("yfinance", "yahoo"):
        if not qt:
            return "Yahoo Finance close snapshot", "no quote time", ["no quote time"]
        parts = [f"fetched {_utc_minutes(qt)}", "Yahoo reports no quote times"]
        if str(inputs.get("t_origin_basis") or "").startswith("quote_time"):
            parts.append("T measured from the fetch")
        return "Yahoo Finance download", f"{parts[0]}; {', '.join(parts[1:])}", parts
    name = "Cboe" if src == "cboe" else src
    clause = f"quotes at {_utc_minutes(qt)}" if qt else "no quote time"
    return name, clause, [clause]


def _record(root: Path, table: str) -> pd.DataFrame:
    files = sorted((root / "market" / table).glob(f"{table}_*.parquet"))
    if not files:
        return pd.DataFrame()
    rec = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rec["date"] = rec["date"].astype(str).str[:10]
    return rec


def record_view(root: Path) -> sqlite3.Connection:
    """An in-memory copy of ``root``/vol_system.db with the committed record applied:
    for each market table the record's rows replace the cache's on its keys (the
    last recorded row of a key wins) and the cache's other rows stay; SPX log
    returns are recomputed from consecutive closes; a recorded chain
    (options/<underlying>_<session>.parquet) the cache does not hold is added."""
    src = sqlite3.connect(f"file:{root / 'vol_system.db'}?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")
    try:
        tables = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        for t in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', src)  # noqa: S608  (names from the schema)
            keys = RECORD_KEYS.get(t)
            if keys:
                rec = _record(root, t)
                if not rec.empty:
                    df["date"] = df["date"].astype(str).str[:10]
                    rec = rec.drop_duplicates(list(keys), keep="last")
                    cols = [c for c in df.columns if c in rec.columns]
                    old = df.set_index(list(keys))
                    new = rec[cols].set_index(list(keys))
                    df = pd.concat([old[~old.index.isin(new.index)], new]).reset_index()
                    df = df.sort_values(list(keys)).reset_index(drop=True)
            if t == "spx_ohlcv" and "log_return" in df.columns:
                c = df["close"].astype(float).to_numpy()
                df["log_return"] = np.concatenate([[np.nan], np.log(c[1:] / c[:-1])])
            if t == "options_snapshots":
                have = set(zip(df["snapshot_date"].astype(str).str[:10], df["underlying"].astype(str)))
                extra = []
                for f in sorted((root / "options").glob("*_*.parquet")):
                    und, _, sess = f.stem.partition("_")
                    if len(sess) == 10 and (sess, und.upper()) not in have:
                        ch = pd.read_parquet(f)
                        ch["snapshot_date"], ch["underlying"] = sess, und.upper()
                        extra.append(ch[[c for c in df.columns if c in ch.columns]])
                if extra:
                    df = pd.concat([df, *extra], ignore_index=True)
            df.to_sql(t, mem, index=False)
    finally:
        src.close()
    return mem


def cache_vs_record(store: Store, table: str, cols: tuple[str, ...]) -> dict[str, Any]:
    """Every row of the SQLite cache's ``table`` against the committed parquet record:
    rows that differ, rows the record lacks, and the record's dates beyond the cache."""
    files = sorted((store.root / "market" / table).glob(f"{table}_*.parquet"))
    rec = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    key = ["date"] + (["ticker"] if table == "vix_term_structure" else [])
    q = f"SELECT {', '.join(key + list(cols))} FROM {table}"  # noqa: S608  (names from the driver's constant list)
    cache_df = store.cache_sql(q)
    for df in (rec, cache_df):
        if not df.empty:
            df["date"] = df["date"].astype(str).str[:10]
    m = cache_df.merge(rec[key + list(cols)], on=key, how="left", suffixes=("", "_rec"), indicator=True)
    missing = int((m["_merge"] == "left_only").sum())
    both = m[m["_merge"] == "both"]
    diff = 0
    for c in cols:
        a, b = both[c].astype(float).to_numpy(), both[c + "_rec"].astype(float).to_numpy()
        bad = ~(np.isclose(a, b, rtol=0.0, atol=CACHE_ATOL) | (np.isnan(a) & np.isnan(b)))
        diff += int(bad.sum())
    ahead = sorted(set(rec["date"]) - set(cache_df["date"])) if not rec.empty else []
    last_cache = cache_df["date"].max() if not cache_df.empty else None
    ahead = [d for d in ahead if last_cache is None or d > last_cache]
    return {"rows": len(cache_df), "differ": diff, "not_in_record": missing, "record_ahead": ahead}


# ── NYSE calendar (full-day closures, from the exchange's rules) ──────────────
def _observed(d: dt.date) -> dt.date | None:
    if d.weekday() == 5:                     # Saturday: the Friday before, except New Year
        return None if (d.month, d.day) == (1, 1) else d - dt.timedelta(days=1)
    if d.weekday() == 6:                     # Sunday: the Monday after
        return d + dt.timedelta(days=1)
    return d


def _nth(year: int, month: int, weekday: int, n: int) -> dt.date:
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def _last(year: int, month: int, weekday: int) -> dt.date:
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> dt.date:
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return dt.date(year, month, day)


@cache
def nyse_closures(year: int) -> frozenset[dt.date]:
    days = [_observed(dt.date(year, 1, 1)), _nth(year, 1, 0, 3), _nth(year, 2, 0, 3),
            _easter(year) - dt.timedelta(days=2), _last(year, 5, 0),
            _observed(dt.date(year, 6, 19)) if year >= 2022 else None,
            _observed(dt.date(year, 7, 4)), _nth(year, 9, 0, 1), _nth(year, 11, 3, 4),
            _observed(dt.date(year, 12, 25))]
    out = {d for d in days if d is not None and d.year == year}
    # New Year's Day on a Saturday of the NEXT year closes nothing this year (NYSE rule)
    return frozenset(out)


def is_session(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in nyse_closures(d.year)


def sessions_between(a: str, b: str) -> int:
    """NYSE sessions in (a, b]: the age of a date ``a`` seen from ``b`` (0 when b ≤ a)."""
    da, db = dt.date.fromisoformat(a[:10]), dt.date.fromisoformat(b[:10])
    if db <= da:
        return 0
    n, d = 0, da
    while d < db:
        d += dt.timedelta(days=1)
        n += is_session(d)
    return n


# ── SVI and Black ─────────────────────────────────────────────────────────────
def svi_total_variance(k: float | np.ndarray, sl: dict) -> Any:
    x = np.asarray(k, float) - sl["m"]
    return sl["a"] + sl["b"] * (sl["rho"] * x + np.sqrt(x * x + sl["sigma"] ** 2))


def svi_vol(k: float, sl: dict) -> float:
    """Implied vol (decimal) of a raw-SVI slice at ln-moneyness k."""
    return math.sqrt(float(svi_total_variance(k, sl)) / sl["T"])


def fwd_call_delta(k: float, w: float) -> float:
    s = math.sqrt(w)
    return float(_norm.cdf(-k / s + 0.5 * s))


def strike_at_delta(sl: dict, delta: float) -> float:
    """ln-moneyness where the undiscounted forward delta of the slice is ``delta``
    (calls positive, puts negative: a put delta d is a call delta 1 + d)."""
    want = delta if delta > 0 else 1.0 + delta
    return float(brentq(lambda k: fwd_call_delta(k, float(svi_total_variance(k, sl))) - want,
                        -3.0, 2.0, xtol=1e-14, rtol=1e-14, maxiter=500))


def rms_vol(log_returns: np.ndarray) -> float:
    """Annualised RMS vol in percent of a window of daily log returns (no demeaning)."""
    r = np.asarray(log_returns, float)
    return 100.0 * math.sqrt(float(np.mean(r * r)) * 252.0)
