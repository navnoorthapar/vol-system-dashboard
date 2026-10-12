"""
Data-layer regression tests (C22 audit).

Each class pins one verified finding against the data layer. Every test runs
against a temporary database (fixture ``tmp_db``) — never the tracked
data_store/vol_system.db — except ``test_calendar_matches_every_shipped_spx_session``,
which opens the shipped file READ-ONLY. Clocks are frozen by monkeypatching
``database._utcnow``; Yahoo and Cboe are replaced by in-process fakes or by
captured fixtures (tests/fixtures/cboe_*). No test touches the network.
"""
import ast
import importlib.util
import logging
import pickle
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration import config
from joint_vol_calibration.data import cboe_downloader as cb
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import pipeline as pl
from joint_vol_calibration.data import yf_downloader as yfd

REPO = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures"


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """A fresh database (and persistence roots) under tmp_path."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(db, "MARKET_DIR", tmp_path / "market", raising=False)
    monkeypatch.setattr(db, "OPTIONS_DIR", tmp_path / "options", raising=False)
    monkeypatch.setattr(db, "PARQUET_DIR", tmp_path / "parquet")
    db.init_database()
    return tmp_path / "t.db"


def freeze(monkeypatch, iso: str) -> datetime:
    now = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    monkeypatch.setattr(db, "_utcnow", lambda: now)
    return now


def spx_frame(dates, closes=None) -> pd.DataFrame:
    closes = np.asarray(closes if closes is not None else np.linspace(7000, 7100, len(dates)), float)
    return pd.DataFrame({"date": list(dates), "open": closes, "high": closes * 1.01,
                         "low": closes * 0.99, "close": closes, "volume": 1})


def chain(snapshot_date, rows, **extra) -> pd.DataFrame:
    """rows: (expiry, strike, right, root, bid, ask)."""
    df = pd.DataFrame(rows, columns=["expiry", "strike", "right", "root", "bid", "ask"])
    df["snapshot_date"] = snapshot_date
    df["mid_price"] = (df["bid"] + df["ask"]) / 2
    df["implied_vol"] = 0.15
    for k, v in extra.items():
        df[k] = v
    return df


def raw_conn(path) -> sqlite3.Connection:
    return sqlite3.connect(str(path))


# The pre-C22 schema, verbatim, so the migration is tested on the real starting point.
_LEGACY_DDL = """
CREATE TABLE spx_ohlcv (date TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL, volume INTEGER NOT NULL, log_return REAL);
CREATE TABLE vix_daily (date TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL,
    low REAL NOT NULL, close REAL NOT NULL);
CREATE TABLE vix_futures_settlements (settlement_date TEXT NOT NULL, expiry_month TEXT NOT NULL,
    settlement_price REAL NOT NULL, PRIMARY KEY (settlement_date, expiry_month));
CREATE TABLE vix_futures_daily (date TEXT NOT NULL, expiry_month TEXT NOT NULL, open REAL,
    high REAL, low REAL, close REAL NOT NULL, volume INTEGER, open_interest INTEGER,
    days_to_expiry INTEGER, PRIMARY KEY (date, expiry_month));
CREATE TABLE options_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_date TEXT NOT NULL,
    underlying TEXT NOT NULL, expiry TEXT NOT NULL, strike REAL NOT NULL, right TEXT NOT NULL,
    bid REAL, ask REAL, mid_price REAL, implied_vol REAL, open_interest INTEGER, volume INTEGER,
    delta REAL, gamma REAL, vega REAL, theta REAL, time_to_expiry REAL);
CREATE INDEX idx_options_snap_date_und ON options_snapshots(snapshot_date, underlying);
CREATE TABLE vix_term_structure (date TEXT NOT NULL, ticker TEXT NOT NULL, tenor_days INTEGER NOT NULL,
    close REAL NOT NULL, is_vvix INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (date, ticker));
CREATE INDEX idx_vts_date ON vix_term_structure(date);
CREATE TABLE regime_labels (date TEXT PRIMARY KEY, regime INTEGER NOT NULL, confidence REAL NOT NULL,
    vix_level REAL, vix_slope REAL, rv_20d REAL);
CREATE TABLE tbill_rates (date TEXT PRIMARY KEY, rate REAL NOT NULL);
CREATE TABLE spx_options (snapshot_date TEXT, expiry TEXT, strike REAL, right TEXT, bid REAL,
    ask REAL, mid_price REAL, implied_vol REAL, open_interest INTEGER, volume INTEGER,
    time_to_expiry REAL, as_of_date TEXT);
"""


@pytest.fixture
def legacy_db(tmp_path, monkeypatch):
    """A pre-C22 file: WAL mode, weekend snapshot, discount-yield rates, 182-day VIX6M,
    a spurious NULL log_return, the dead spx_options table and the misnamed regime_labels."""
    path = tmp_path / "legacy.db"
    con = raw_conn(path)
    con.executescript(_LEGACY_DDL)
    con.execute("PRAGMA journal_mode=WAL")
    con.executemany("INSERT INTO spx_ohlcv VALUES (?,?,?,?,?,?,?)", [
        ("2025-05-06", 1, 1, 1, 100.0, 1, None),
        ("2025-05-07", 1, 1, 1, 101.0, 1, float(np.log(101 / 100))),
        ("2025-05-08", 1, 1, 1, 102.0, 1, None),                 # the spurious NULL
    ])
    con.executemany("INSERT INTO options_snapshots (snapshot_date, underlying, expiry, strike, right, "
                    "bid, ask, mid_price, implied_vol, time_to_expiry) VALUES (?,?,?,?,?,?,?,?,?,?)", [
        ("2026-05-31", "SPX", "2026-06-15", 7500.0, "C", 10, 11, 10.5, 0.15, 15 / 365),
        ("2026-05-31", "SPX", "2026-06-15", 7500.0, "P", 9, 10, 9.5, 0.15, 15 / 365),
        ("2026-03-24", "SPX", "2026-04-17", 6500.0, "C", 50, 51, 50.5, 0.2, 24 / 365),
    ])
    con.executemany("INSERT INTO vix_term_structure VALUES (?,?,?,?,?)", [
        ("2026-06-18", "^VIX6M", 182, 20.0, 0), ("2026-06-18", "^VIX3M", 93, 18.0, 0),
        ("2026-06-18", "^VVIX", 30, 90.0, 1)])
    con.executemany("INSERT INTO tbill_rates VALUES (?,?)", [("2024-06-03", 0.04), ("2020-03-18", 0.00003)])
    con.execute("INSERT INTO regime_labels VALUES ('2025-12-30', 1, 0.8, 15.0, 1.0, 0.12)")
    con.execute("INSERT INTO spx_options (snapshot_date) VALUES ('2026-06-05')")
    con.commit()
    con.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Session calendar and snapshot dating (data-snapshot-date-wallclock-weekend)
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionCalendar:
    def test_calendar_matches_every_shipped_spx_session(self):
        """The rule-based NYSE calendar reproduces every ^GSPC session in the shipped DB."""
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        try:
            dates = [r[0] for r in con.execute("SELECT date FROM spx_ohlcv")]
        finally:
            con.close()
        if len(dates) < 1000:
            pytest.skip("shipped SPX history unavailable")
        res = db.validate_sessions(dates, log=False)
        assert res == {"missing": [], "extra": []}

    def test_2026_holidays(self):
        assert db.nyse_holidays("2026-01-01", "2026-12-31") == [
            "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
            "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"]

    @pytest.mark.parametrize("now,last_closed,quote", [
        ("2026-05-31T15:00:00Z", "2026-05-29", "2026-05-29"),   # Sunday
        ("2026-06-20T10:00:00Z", "2026-06-18", "2026-06-18"),   # Saturday after Juneteenth
        ("2026-03-28T12:00:00Z", "2026-03-27", "2026-03-27"),   # Saturday
        ("2026-10-09T23:30:00Z", "2026-10-09", "2026-10-09"),   # Friday 19:30 ET
        ("2026-10-10T02:00:00Z", "2026-10-09", "2026-10-09"),   # Friday 22:00 ET, Saturday in UTC
        ("2026-10-09T20:05:00Z", "2026-10-08", "2026-10-09"),   # Friday 16:05 ET: before the cutoff
        ("2026-10-12T08:15:00Z", "2026-10-09", "2026-10-09"),   # Monday pre-open
        ("2026-10-12T14:00:00Z", "2026-10-09", "2026-10-12"),   # Monday intraday
    ])
    def test_last_closed_and_quote_session(self, now, last_closed, quote):
        t = datetime.fromisoformat(now.replace("Z", "+00:00"))
        assert db.last_closed_session(t) == last_closed
        assert db.quote_session(t) == quote

    def test_yahoo_chain_fetched_on_a_saturday_is_dated_friday(self, monkeypatch):
        monkeypatch.setattr(yfd.yf, "Ticker", _FakeOptionTicker)
        sat = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
        df = yfd._fetch_yf_options("^SPX", "SPX", now=sat)
        assert set(df["snapshot_date"]) == {"2026-10-09"}
        am = df[(df["root"] == "SPX") & (df["right"] == "C")].iloc[0]
        pm = df[(df["root"] == "SPXW") & (df["right"] == "C")].iloc[0]
        assert (am["settlement"], pm["settlement"]) == ("AM", "PM")
        assert pm["time_to_expiry"] == pytest.approx(7 / 365)            # Fri close -> Fri close
        assert am["time_to_expiry"] == pytest.approx((7 * 1440 - 390) / 525_600)
        assert set(df["underlying_price"]) == {7811.54}
        assert set(df["quote_time"]) == {"2026-10-10T14:00:00Z"}

    def test_migration_redates_weekend_snapshots(self, legacy_db):
        db.init_database()
        con = raw_conn(legacy_db)
        rows = con.execute("SELECT snapshot_date, expiry, time_to_expiry FROM options_snapshots "
                           "ORDER BY snapshot_date").fetchall()
        con.close()
        assert {r[0] for r in rows} == {"2026-03-24", "2026-05-29"}     # Sunday 05-31 -> Friday 05-29
        for snap, exp, tte in rows:
            assert tte == pytest.approx((pd.Timestamp(exp) - pd.Timestamp(snap)).days / 365)
        assert [r[2] * 365 for r in rows if r[0] == "2026-05-29"] == [pytest.approx(17)] * 2

    def test_insert_warns_on_a_non_session_snapshot_date(self, tmp_db, caplog):
        with caplog.at_level(logging.WARNING):
            db.insert_options_snapshot(chain("2026-05-31", [("2026-06-15", 7500, "C", "SPX", 1, 2)]), "SPX")
        assert "not an NYSE session" in caplog.text


class _FakeChain:
    def __init__(self, calls, puts):
        self.calls, self.puts = calls, puts


class _FakeOptionTicker:
    """Yahoo ^SPX chain shaped like yfinance's: one expiry, an SPX and an SPXW contract per side."""
    options = ("2026-10-16",)

    def __init__(self, symbol):
        self.symbol = symbol

    def option_chain(self, expiry):
        def side(cp, bids, asks):
            return pd.DataFrame({
                "contractSymbol": [f"SPX261016{cp}07800000", f"SPXW261016{cp}07800000"],
                "strike": [7800.0, 7800.0], "bid": bids, "ask": asks,
                "impliedVolatility": [0.117, 0.119], "openInterest": [28305, 3349],
                "volume": [100, 50], "lastTradeDate": pd.to_datetime(["2026-10-09 20:11:52"] * 2, utc=True),
            })
        return _FakeChain(side("C", [59.7, 61.7], [60.4, 62.0]), side("P", [23.5, 25.4], [24.0, 25.8]))

    def history(self, **kw):
        idx = pd.DatetimeIndex(["2026-10-08", "2026-10-09"]).tz_localize("America/New_York")
        return pd.DataFrame({"Close": [7764.70, 7811.54]}, index=idx)


# ══════════════════════════════════════════════════════════════════════════════
# Idempotent option snapshots (data-options-snapshot-not-idempotent)
# ══════════════════════════════════════════════════════════════════════════════

class TestOptionSnapshots:
    ROWS = [("2026-10-16", 7800.0, "C", "SPX", 59.7, 60.4), ("2026-10-16", 7800.0, "P", "SPX", 23.5, 24.0)]

    def test_reinserting_a_session_replaces_its_chain(self, tmp_db):
        db.insert_options_snapshot(chain("2026-10-09", self.ROWS), "SPX")
        later = chain("2026-10-09", [(e, k, r, root, b + 1, a + 1) for e, k, r, root, b, a in self.ROWS])
        db.insert_options_snapshot(later, "SPX")
        surf = db.get_options_surface("2026-10-09", "SPX")
        assert len(surf) == 2                                   # was 4: plain INSERT appended
        assert sorted(surf["bid"]) == [24.5, 60.7]              # the later pull wins

    def test_am_and_pm_contracts_at_one_strike_coexist(self, tmp_db):
        rows = self.ROWS + [("2026-10-16", 7800.0, "C", "SPXW", 61.7, 62.0)]
        assert db.insert_options_snapshot(chain("2026-10-09", rows), "SPX") == 3
        surf = db.get_options_surface("2026-10-09", "SPX")
        calls = surf[surf["right"] == "C"].set_index("root")
        assert calls.loc["SPX", "settlement"] == "AM" and calls.loc["SPXW", "settlement"] == "PM"
        assert calls.loc["SPXW", "time_to_expiry"] > calls.loc["SPX", "time_to_expiry"]

    def test_repeated_key_in_one_chain_keeps_the_last(self, tmp_db, caplog):
        rows = self.ROWS + [("2026-10-16", 7800.0, "C", "SPX", 1.0, 2.0)]
        with caplog.at_level(logging.WARNING):
            assert db.insert_options_snapshot(chain("2026-10-09", rows), "SPX") == 2
        assert "repeating the natural key" in caplog.text

    def test_unique_index_backs_the_natural_key(self, tmp_db):
        db.insert_options_snapshot(chain("2026-10-09", self.ROWS[:1]), "SPX")
        con = raw_conn(tmp_db)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute("INSERT INTO options_snapshots (snapshot_date, underlying, root, expiry, strike, right) "
                        "VALUES ('2026-10-09', 'SPX', 'SPX', '2026-10-16', 7800.0, 'C')")
        con.close()

    def test_no_write_only_parquet_mirror(self, tmp_db, tmp_path):
        db.insert_options_snapshot(chain("2026-10-09", self.ROWS), "SPX")
        assert not (tmp_path / "parquet").exists() or not any((tmp_path / "parquet").iterdir())


class TestOptionsSurfaceStaleness:
    """data-get-options-surface-staleness"""

    def _stored(self):
        db.insert_options_snapshot(chain("2026-09-23", [("2026-10-16", 7000.0, "C", "SPXW", 5, 6)]), "SPX")

    def test_stale_chain_raises_when_the_caller_limits_age(self, tmp_db):
        self._stored()
        with pytest.raises(db.StaleDataError, match="12 session"):
            db.get_options_surface("2026-10-09", "SPX", max_age_sessions=0)

    def test_stale_chain_is_returned_with_a_warning_by_default(self, tmp_db, caplog):
        self._stored()
        with caplog.at_level(logging.WARNING):
            surf = db.get_options_surface("2026-10-09", "SPX")
        assert len(surf) == 1 and "12 session(s) stale" in caplog.text
        assert str(surf["snapshot_date"].iloc[0].date()) == "2026-09-23"

    def test_weekend_query_of_friday_chain_is_not_stale(self, tmp_db):
        db.insert_options_snapshot(chain("2026-10-09", [("2026-10-16", 7000.0, "C", "SPXW", 5, 6)]), "SPX")
        assert len(db.get_options_surface("2026-10-11", "SPX", max_age_sessions=0)) == 1


# ══════════════════════════════════════════════════════════════════════════════
# T-bill units (data-rq-units-and-provenance, data side)
# ══════════════════════════════════════════════════════════════════════════════

class TestTbillUnits:
    def test_cc_rate_reprices_the_bill(self):
        d = np.array([0.0005, 0.02, 0.04, 0.055, -0.00105])
        r = db.tbill_discount_to_cc(d)
        price = 1 - d * 91 / 360                     # bank-discount price of a 91-day bill
        assert np.allclose(np.exp(-r * 91 / 365), price, rtol=0, atol=1e-15)
        assert db.tbill_discount_to_cc(0.04) == pytest.approx(0.0407620, abs=1e-7)

    def test_download_keeps_zero_and_negative_yields_and_converts(self, monkeypatch):
        idx = pd.DatetimeIndex(["2020-03-18", "2020-03-19", "2020-03-26", "2020-03-30"], name="Date")
        quotes = [0.003, -0.028, -0.105, 0.0]
        monkeypatch.setattr(yfd.yf, "download", lambda *a, **k: pd.DataFrame({"Close": quotes}, index=idx))
        df = yfd.download_tbill_rate(start="2020-03-16", end="2020-03-31")
        assert df["date"].tolist() == ["2020-03-18", "2020-03-19", "2020-03-26", "2020-03-30"]
        assert np.allclose(df["discount_yield"], np.array(quotes) / 100)
        assert np.allclose(df["rate"], db.tbill_discount_to_cc(np.array(quotes) / 100))
        assert (df["rate"] < 0).sum() == 2

    def test_negative_rate_round_trips(self, tmp_db):
        db.insert_tbill_rates(pd.DataFrame({"date": ["2020-03-26"], "rate": [-0.00105]}))
        assert db.get_tbill_rate("2020-03-27") == pytest.approx(-0.00105)

    def test_migration_converts_legacy_discount_rates_exactly_once(self, legacy_db):
        db.init_database()
        db.init_database()
        con = raw_conn(legacy_db)
        rate, disc = con.execute("SELECT rate, discount_yield FROM tbill_rates WHERE date='2024-06-03'").fetchone()
        con.close()
        assert disc == pytest.approx(0.04)
        assert rate == pytest.approx(0.0407620, abs=1e-7)

    def test_backfill_guard_admits_march_2020_but_not_unit_errors(self):
        mod = _load_script("backfill_tbill_history")
        rates = pd.DataFrame({"date": ["a", "b", "c", "d"], "rate": [-0.00105, 0.05, 0.30, -0.02]})
        assert mod.out_of_range(rates)["date"].tolist() == ["c", "d"]
        # The backtest window comes from config, not a second literal.
        tree = ast.parse((REPO / "scripts" / "backfill_tbill_history.py").read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                    and n.module == "joint_vol_calibration.config" for a in n.names}
        assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
                    if isinstance(t, ast.Name)}
        assert {"BT_START", "BT_END"} <= imported and not {"BT_START", "BT_END"} & assigned


# ══════════════════════════════════════════════════════════════════════════════
# Download windows (data-daily-refresh-drops-current-session)
# ══════════════════════════════════════════════════════════════════════════════

class _Recorder:
    def __init__(self):
        self.calls = []

    def download(self, tickers, **kw):
        self.calls.append((tickers, kw))
        return pd.DataFrame()

    def ticker(self, symbol):
        rec = self

        class _T:
            options = ()

            def history(self, **kw):
                rec.calls.append((symbol, kw))
                return pd.DataFrame()
        return _T()


@pytest.fixture
def offline_pipeline(tmp_db, monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(yfd.yf, "download", rec.download)
    monkeypatch.setattr(yfd.yf, "Ticker", rec.ticker)
    snaps = []

    def no_cboe(*a, **k):
        raise cb.requests.HTTPError("offline")
    monkeypatch.setattr(cb, "download_vix_history", no_cboe)
    monkeypatch.setattr(cb, "download_vix_term_structure_history", no_cboe)
    vx_calls = []
    monkeypatch.setattr(cb, "download_vix_futures_daily_history",
                        lambda **k: vx_calls.append(k) or pd.DataFrame())
    monkeypatch.setattr(cb, "download_vix_futures_settlements", lambda *a, **k: pd.DataFrame())
    rec.vx_calls = vx_calls
    monkeypatch.setattr(cb, "download_cboe_options_snapshot", lambda u, now=None: snaps.append(u) or pd.DataFrame())
    monkeypatch.setattr(yfd, "snapshot_spx_options", lambda: snaps.append("yf-SPX") or pd.DataFrame())
    monkeypatch.setattr(yfd, "snapshot_vix_options", lambda: snaps.append("yf-VIX") or pd.DataFrame())
    return rec, snaps


class TestDownloadWindows:
    def test_evening_refresh_requests_the_session_that_just_closed(self, offline_pipeline, monkeypatch):
        rec, snaps = offline_pipeline
        freeze(monkeypatch, "2026-10-09T22:00:00Z")            # Friday 18:00 ET
        pl.DataPipeline().daily_refresh()
        assert {t for t, _ in rec.calls} == {"^GSPC", "^VIX", "^VIX9D ^VIX ^VIX3M ^VIX6M ^VVIX", "^IRX"}
        for ticker, kw in rec.calls:
            assert kw["end"] == "2026-10-10", f"{ticker}: the 10-09 session must be inside the exclusive end"
            assert kw["start"] == "2026-09-29"
        assert snaps == ["SPX", "yf-SPX", "VIX", "yf-VIX"]       # Cboe first, Yahoo fallback
        assert rec.vx_calls == [{"start": "2026-09-29", "end": "2026-10-09"}]

    def test_vx_refresh_fills_the_gap_since_the_last_stored_session(self, offline_pipeline, monkeypatch):
        rec, _ = offline_pipeline
        db.insert_vix_futures_daily(pd.DataFrame({"date": ["2026-08-14"], "expiry_date": ["2026-09-16"],
                                                  "settle": [18.0]}))
        freeze(monkeypatch, "2026-10-09T22:00:00Z")
        pl.DataPipeline().daily_refresh()
        assert rec.vx_calls == [{"start": "2026-08-14", "end": "2026-10-09"}]

    def test_refresh_reaches_back_to_the_last_stored_session(self, offline_pipeline, monkeypatch):
        rec, _ = offline_pipeline
        db.insert_spx_ohlcv(spx_frame(["2026-06-17", "2026-06-18"]))
        freeze(monkeypatch, "2026-10-09T22:00:00Z")
        pl.DataPipeline().daily_refresh()
        starts = {t: kw["start"] for t, kw in rec.calls}
        assert starts["^GSPC"] == "2026-06-18"                  # not 2026-09-29: no silent gap
        assert starts["^IRX"] == "2026-09-29"                   # empty table: the normal window

    def test_intraday_refresh_stops_at_the_previous_session_and_skips_chains(self, offline_pipeline, monkeypatch):
        rec, snaps = offline_pipeline
        freeze(monkeypatch, "2026-10-09T15:00:00Z")            # Friday 11:00 ET
        pl.DataPipeline().daily_refresh()
        assert {kw["end"] for _, kw in rec.calls} == {"2026-10-09"}   # inclusive end 10-08
        assert snaps == []

    @pytest.mark.parametrize("fn", [yfd.download_vix_index, yfd.download_tbill_rate,
                                    yfd.download_vix_term_structure])
    def test_default_end_is_the_last_closed_session(self, monkeypatch, fn):
        rec = _Recorder()
        monkeypatch.setattr(yfd.yf, "download", rec.download)
        freeze(monkeypatch, "2026-10-11T12:00:00Z")            # Sunday
        fn(start="2026-10-01")
        assert rec.calls[0][1]["end"] == "2026-10-10"           # Friday 10-09 inclusive

    def test_hist_end_none_means_last_closed_session(self, tmp_db, monkeypatch):
        freeze(monkeypatch, "2026-10-11T12:00:00Z")
        assert config.HIST_END is None
        assert pl.DataPipeline().hist_end == "2026-10-09"


# ══════════════════════════════════════════════════════════════════════════════
# Validation hooks (data-no-cross-source-or-calendar-validation)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidationHooks:
    def test_validate_sessions_reports_missing_and_extra(self, caplog):
        with caplog.at_level(logging.WARNING):
            res = db.validate_sessions(["2026-06-11", "2026-06-15", "2026-06-19"], label="t")
        assert res == {"missing": ["2026-06-12", "2026-06-16", "2026-06-17", "2026-06-18"],
                       "extra": ["2026-06-19"]}
        assert "missing" in caplog.text and "non-session" in caplog.text

    def test_insert_logs_a_missing_session(self, tmp_db, caplog):
        with caplog.at_level(logging.WARNING):
            db.insert_vix_daily(pd.DataFrame({"date": ["2026-06-11", "2026-06-15"], "open": 1.0,
                                              "high": 1.0, "low": 1.0, "close": 1.0}))
        assert "2026-06-12" in caplog.text

    def test_find_bad_prints(self, caplog):
        df = pd.DataFrame({"date": ["a", "b", "c"], "low": [1, 5, 1], "high": [2, 4, 2], "close": [1.5, 4.5, 0]})
        with caplog.at_level(logging.WARNING):
            assert db.find_bad_prints(df, "t")["date"].tolist() == ["b", "c"]

    def _store_yahoo_vix(self):
        vix = pd.DataFrame({"date": ["2026-02-05", "2026-02-06", "2026-02-09", "2026-06-11", "2026-06-15"],
                            "close": [21.77, 20.37, 17.36, 19.44, 16.20]})
        for c in ("open", "high", "low"):
            vix[c] = vix["close"]
        db.insert_vix_daily(vix.assign(source="yfinance"))
        db.insert_vix_term_structure(vix[["date", "close"]].assign(ticker="^VIX"))

    def test_cross_check_flags_the_yahoo_bad_print_and_the_gap(self, tmp_db, monkeypatch, caplog):
        self._store_yahoo_vix()
        ref = cb.download_index_history.__wrapped__ if hasattr(cb.download_index_history, "__wrapped__") else None
        monkeypatch.setattr(cb._SESSION, "get", lambda url, timeout=None: _FileResp(FIX / "cboe_VIX_History.csv"))
        ref = cb.download_index_history("VIX").assign(ticker="^VIX")
        with caplog.at_level(logging.WARNING):
            res = db.cross_check_vix(ref)
        mism = res[res["issue"] == "mismatch"]
        assert set(mism["table"]) == {"vix_daily", "vix_term_structure"}
        assert set(mism["date"]) == {"2026-02-06"}
        assert mism["stored"].iloc[0] == pytest.approx(20.37) and mism["reference"].iloc[0] == pytest.approx(17.76)
        gaps = res[res["issue"] == "missing_in_db"]
        assert "2026-06-12" in set(gaps["date"])
        assert "2026-01-19" not in set(res["date"])             # Cboe's MLK-day print is not a session
        assert "20.37 vs 17.76" in caplog.text

    def test_pipeline_does_not_ingest_cboe_holiday_prints(self):
        df = pd.DataFrame({"date": ["2026-01-16", "2026-01-19", "2026-01-20"], "close": [1.0, 2.0, 3.0]})
        assert pl._sessions_only(df, "t")["date"].tolist() == ["2026-01-16", "2026-01-20"]

    def test_upsert_keeps_a_session_the_source_dropped(self, tmp_db, caplog):
        dates = ["2026-09-21", "2026-09-22", "2026-09-23"]
        db.insert_spx_ohlcv(spx_frame(dates))
        with caplog.at_level(logging.WARNING):
            db.insert_spx_ohlcv(spx_frame(["2026-09-21", "2026-09-23"], [7000, 7100]))
        stored = db.get_spx_ohlcv("2026-09-30")["date"].dt.strftime("%Y-%m-%d").tolist()
        assert stored == dates
        assert "no longer returns 1 stored row" in caplog.text and "2026-09-22" in caplog.text


class _FileResp:
    def __init__(self, path):
        self.text, self.status_code = Path(path).read_text(), 200

    def raise_for_status(self):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# log_return on upsert (data-log-return-null-on-upsert)
# ══════════════════════════════════════════════════════════════════════════════

class TestLogReturns:
    def test_overlapping_reinsert_keeps_every_return(self, tmp_db):
        dates = pd.bdate_range("2026-09-01", "2026-09-10").strftime("%Y-%m-%d")
        db.insert_spx_ohlcv(spx_frame(dates))
        db.insert_spx_ohlcv(spx_frame(dates[4:], np.linspace(7000, 7100, len(dates))[4:]))
        df = db.get_spx_ohlcv("2026-09-10")
        assert df["log_return"].isna().tolist() == [True] + [False] * (len(df) - 1)
        assert np.allclose(df["log_return"].iloc[1:], np.log(df["close"]).diff().iloc[1:])

    def test_filling_a_gap_recomputes_the_next_return(self, tmp_db):
        db.insert_spx_ohlcv(spx_frame(["2026-09-21", "2026-09-23"], [100.0, 110.0]))
        db.insert_spx_ohlcv(spx_frame(["2026-09-22"], [105.0]))
        lr = db.get_spx_ohlcv("2026-09-23").set_index("date")["log_return"]
        assert lr.iloc[2] == pytest.approx(np.log(110 / 105))   # not ln(110/100)

    def test_migration_repairs_a_stored_null(self, legacy_db):
        db.init_database()
        con = raw_conn(legacy_db)
        lr = con.execute("SELECT log_return FROM spx_ohlcv WHERE date='2025-05-08'").fetchone()[0]
        first = con.execute("SELECT log_return FROM spx_ohlcv WHERE date='2025-05-06'").fetchone()[0]
        con.close()
        assert lr == pytest.approx(np.log(102 / 101)) and first is None


# ══════════════════════════════════════════════════════════════════════════════
# Connections and journal mode (data-sqlite-connection-wal)
# ══════════════════════════════════════════════════════════════════════════════

class TestConnections:
    def test_every_connection_is_closed(self, tmp_db, monkeypatch):
        opened = []
        real = sqlite3.connect

        def tracking(*a, **k):
            c = real(*a, **k)
            opened.append(c)
            return c
        monkeypatch.setattr(db.sqlite3, "connect", tracking)
        db.init_database()
        db.insert_spx_ohlcv(spx_frame(["2026-09-21", "2026-09-22"]))
        db.get_spx_ohlcv("2026-09-22")
        db.insert_tbill_rates(pd.DataFrame({"date": ["2026-09-21"], "rate": [0.04]}))
        db.get_tbill_rate("2026-09-22")
        db.insert_options_snapshot(chain("2026-09-22", TestOptionSnapshots.ROWS), "SPX")
        db.get_options_surface("2026-09-22", "SPX")
        db.get_data_coverage()
        assert len(opened) >= 8
        for c in opened:
            with pytest.raises(sqlite3.ProgrammingError):
                c.execute("SELECT 1")

    def test_init_on_a_current_file_changes_no_byte(self, tmp_db):
        """DataPipeline() calls init_database(); on the tracked file that must not dirty git."""
        db.insert_spx_ohlcv(spx_frame(["2026-09-21"]))
        before = tmp_db.read_bytes()
        db.init_database()
        db._SCHEMA_READY.clear()
        db.get_latest_date("spx_ohlcv")                     # re-checks the schema
        assert tmp_db.read_bytes() == before

    def test_file_is_left_in_rollback_journal_mode(self, legacy_db):
        db.init_database()
        db.insert_spx_ohlcv(spx_frame(["2026-09-21"]))
        con = raw_conn(legacy_db)
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
        assert mode == "delete"
        assert not Path(str(legacy_db) + "-wal").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Orphan tables and the misnamed regime API (data-orphan-tables-apis)
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaMigration:
    def test_dead_and_misnamed_tables(self, legacy_db):
        db.init_database()
        con = raw_conn(legacy_db)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n_pred = con.execute("SELECT COUNT(*) FROM classifier_predictions").fetchone()[0]
        version = con.execute("PRAGMA user_version").fetchone()[0]
        indexes = {r[1] for r in con.execute("PRAGMA index_list(options_snapshots)")}
        con.close()
        assert "spx_options" not in tables and "regime_labels" not in tables
        assert n_pred == 1 and version == db.SCHEMA_VERSION
        assert indexes == {"ux_options_snapshot_key"}

    def test_legacy_option_rows_get_inferred_source_and_unrecorded_root(self, legacy_db):
        db.init_database()
        surf = db.get_options_surface("2026-05-29", "SPX")
        assert set(surf["source"]) == {"yfinance"} and set(surf["root"]) == {""}
        assert surf["settlement"].isna().all()

    def test_regime_label_alias_reads_classifier_predictions(self, tmp_db):
        db.insert_regime_labels(pd.DataFrame({"date": ["2025-12-30"], "regime": [1], "confidence": [0.8],
                                              "vix_level": [15.0], "vix_slope": [1.0], "rv_20d": [0.1]}))
        assert len(db.get_classifier_predictions("2025-12-31")) == 1
        assert len(db.get_regime_labels("2025-12-31")) == 1
        with pytest.raises(ValueError, match="Invalid date"):
            db.get_regime_labels("2025-12-31", start_date="garbage")

    def test_legacy_vix_futures_rows_still_insert_and_read(self, tmp_db):
        df = pd.DataFrame({"date": ["2020-02-10", "2020-02-11"], "expiry_month": "2020-03",
                           "open": 20.0, "high": 21.0, "low": 19.0, "close": [20.5, 20.7],
                           "volume": 1, "open_interest": 1, "days_to_expiry": [37, 36]})
        db.insert_vix_futures_daily(df)
        curve = db.get_vix_futures_curve("2020-02-11")
        assert curve["expiry_date"].tolist() == ["2020-03-18"] and curve["price"].tolist() == [20.7]

    def test_coverage_reports_each_snapshot(self, tmp_db):
        db.insert_options_snapshot(chain("2026-10-09", TestOptionSnapshots.ROWS, source="cboe"), "SPX")
        cov = db.get_data_coverage()
        assert cov["options_spx"]["per_snapshot"] == {"2026-10-09": {"rows": 2, "source": "cboe"}}
        assert "tbill_rates" in cov and "vix_term_structure_by_ticker" in cov


# ══════════════════════════════════════════════════════════════════════════════
# VIX term-structure tenors (data-vix-ts-index-fit-as-futures, data side)
# ══════════════════════════════════════════════════════════════════════════════

class TestTermStructureTenors:
    def test_download_uses_the_config_tenor_map(self, monkeypatch):
        tickers = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M", "^VVIX"]
        cols = pd.MultiIndex.from_product([["Close"], tickers], names=["Price", "Ticker"])
        frame = pd.DataFrame([[12.0, 14.0, 17.0, 19.0, 85.0]], columns=cols,
                             index=pd.DatetimeIndex(["2026-10-09"], name="Date"))
        monkeypatch.setattr(yfd.yf, "download", lambda *a, **k: frame)
        ts = yfd.download_vix_term_structure(start="2026-10-09", end="2026-10-09")
        got = ts.set_index("ticker")["tenor_days"].to_dict()
        assert got == {**config.VIX_INDEX_TENOR_DAYS, "^VVIX": config.VVIX_TENOR_DAYS}
        assert got["^VIX6M"] == 184 and got["^VIX3M"] == 93

    def test_writer_cannot_store_a_second_tenor_map(self, tmp_db, caplog):
        with caplog.at_level(logging.WARNING):
            db.insert_vix_term_structure(pd.DataFrame({"date": ["2026-10-09"], "ticker": ["^VIX6M"],
                                                       "tenor_days": [182], "close": [19.0]}))
        assert db.get_vix_term_structure("2026-10-09")["tenor_days"].tolist() == [184]
        assert "overriding tenor_days" in caplog.text

    def test_migration_fixes_the_stored_vix6m_tenor(self, legacy_db):
        db.init_database()
        ts = db.get_vix_term_structure("2026-06-18").set_index("ticker")["tenor_days"]
        assert ts.to_dict() == {"^VIX3M": 93, "^VIX6M": 184, "^VVIX": 30}


# ══════════════════════════════════════════════════════════════════════════════
# Cboe -> Yahoo fallbacks (data-cboe-downloader-broken, pipeline side)
# ══════════════════════════════════════════════════════════════════════════════

class TestPipelineFallbacks:
    def test_vix_backfill_falls_back_to_yahoo_when_cboe_fails(self, tmp_db, monkeypatch):
        def boom():
            raise cb.requests.HTTPError("403")
        monkeypatch.setattr(cb, "download_vix_history", boom)
        yahoo = pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "open": 12.0, "high": 13.0,
                              "low": 11.0, "close": [12.5, 14.0], "source": "yfinance"})
        monkeypatch.setattr(yfd, "download_vix_index", lambda start, end: yahoo)
        pipe = pl.DataPipeline(hist_start="2020-01-01", hist_end="2020-01-31")
        assert pipe._load_vix_daily() == 2                     # was 0: the fallback was unreachable
        assert db.get_vix_daily("2020-01-31")["close"].tolist() == [12.5, 14.0]

    def test_snapshot_prefers_cboe_and_falls_back_to_yahoo(self, tmp_db, monkeypatch):
        freeze(monkeypatch, "2026-10-09T22:00:00Z")
        pipe = pl.DataPipeline()
        cboe_chain = chain("2026-10-09", TestOptionSnapshots.ROWS, source="cboe")
        monkeypatch.setattr(cb, "download_cboe_options_snapshot", lambda u, now=None: cboe_chain)
        monkeypatch.setattr(yfd, "snapshot_spx_options", lambda: pytest.fail("Yahoo must not be called"))
        assert pipe._snapshot_options("SPX") == 2

        def down(u, now=None):
            raise cb.requests.HTTPError("403")
        yahoo_chain = chain("2026-10-09", TestOptionSnapshots.ROWS[:1], source="yfinance")
        monkeypatch.setattr(cb, "download_cboe_options_snapshot", down)
        monkeypatch.setattr(yfd, "snapshot_spx_options", lambda: yahoo_chain)
        assert pipe._snapshot_options("SPX") == 1
        assert set(db.get_options_surface("2026-10-09", "SPX")["source"]) == {"yfinance"}


# ══════════════════════════════════════════════════════════════════════════════
# Persistence (ops-db-not-persisted / data-daily-chains-discarded, data side)
# ══════════════════════════════════════════════════════════════════════════════

class TestPersistence:
    def _fresh_db(self, monkeypatch, path):
        monkeypatch.setattr(db, "DB_PATH", path)
        db.init_database()

    def test_partition_layout_and_round_trip(self, tmp_db, tmp_path, monkeypatch):
        db.insert_spx_ohlcv(spx_frame(["2025-12-31", "2026-01-02"], [6800.0, 6850.0]))
        db.insert_tbill_rates(pd.DataFrame({"date": ["2026-01-02"], "rate": [0.0365], "discount_yield": [0.036]}))
        written = db.export_market_partitions()
        names = sorted(p.relative_to(tmp_path / "market").as_posix() for p in written)
        assert names == ["spx_ohlcv/spx_ohlcv_2025.parquet", "spx_ohlcv/spx_ohlcv_2026.parquet",
                         "tbill_rates/tbill_rates_2026.parquet"]
        self._fresh_db(monkeypatch, tmp_path / "fresh.db")
        assert db.import_market_partitions() == {"spx_ohlcv": 2, "tbill_rates": 1}
        df = db.get_spx_ohlcv("2026-01-02")
        assert df["close"].tolist() == [6800.0, 6850.0]
        assert df["log_return"].iloc[1] == pytest.approx(np.log(6850 / 6800))
        assert db.get_tbill_rates_series("2026-01-02")["discount_yield"].tolist() == [0.036]

    def test_a_session_the_source_later_drops_survives(self, tmp_db, tmp_path, monkeypatch):
        db.insert_spx_ohlcv(spx_frame(["2026-09-21", "2026-09-22", "2026-09-23"]))
        db.export_market_partitions()
        # Next run: the stale committed DB, plus a download that no longer has 09-22.
        self._fresh_db(monkeypatch, tmp_path / "next_run.db")
        db.insert_spx_ohlcv(spx_frame(["2026-09-21", "2026-09-23", "2026-09-24"]))
        db.export_market_partitions(since="2026-01-01")
        part = pd.read_parquet(tmp_path / "market" / "spx_ohlcv" / "spx_ohlcv_2026.parquet")
        assert part["date"].tolist() == ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]

    def test_unchanged_data_rewrites_nothing(self, tmp_db, tmp_path):
        db.insert_vix_term_structure(pd.DataFrame({"date": ["2026-10-09"] * 2, "ticker": ["^VIX", "^VVIX"],
                                                   "close": [14.84, 84.88], "source": "cboe"}))
        first = db.export_market_partitions()
        path = tmp_path / "market" / "vix_term_structure" / "vix_term_structure_2026.parquet"
        assert first == [path]
        blob = path.read_bytes()
        assert db.export_market_partitions() == []
        assert path.read_bytes() == blob

    def test_option_chain_export_import(self, tmp_db, tmp_path, monkeypatch):
        rows = TestOptionSnapshots.ROWS + [("2026-10-16", 7800.0, "C", "SPXW", 61.7, 62.0)]
        db.insert_options_snapshot(chain("2026-10-09", rows, source="cboe", underlying_price=7811.54,
                                         quote_time="2026-10-09T22:00:00Z"), "SPX")
        path = db.export_options_snapshot("2026-10-09", "SPX")
        assert path == tmp_path / "options" / "spx_2026-10-09.parquet"
        assert db.export_options_snapshot("2026-10-09", "SPX") is None      # unchanged: not rewritten
        self._fresh_db(monkeypatch, tmp_path / "fresh.db")
        assert db.import_options_snapshots() == 3
        assert db.import_options_snapshots() == 0                           # already there
        surf = db.get_options_surface("2026-10-09", "SPX")
        assert set(surf["root"]) == {"SPX", "SPXW"} and set(surf["underlying_price"]) == {7811.54}


# ══════════════════════════════════════════════════════════════════════════════
# Dates and config (data-normalise-date-robustness, data-config-dead-window-sideeffects)
# ══════════════════════════════════════════════════════════════════════════════

    def test_option_export_size_filters(self, tmp_db, tmp_path):
        rows = [("2026-10-16", 7800.0, "C", "SPXW", 61.7, 62.0),     # ATM, two-sided
                ("2026-10-16", 4000.0, "P", "SPXW", 0.0, 0.05),      # one-sided, 0.51x
                ("2028-12-15", 7800.0, "C", "SPX", 900.0, 910.0)]    # ~2.2 years out
        db.insert_options_snapshot(chain("2026-10-09", rows, underlying_price=7811.54), "SPX")
        path = db.export_options_snapshot("2026-10-09", "SPX", tmp_path / "f", moneyness=(0.7, 1.3),
                                          max_years=2.0, two_sided_only=True)
        kept = pd.read_parquet(path)
        assert kept[["expiry", "strike"]].values.tolist() == [["2026-10-16", 7800.0]]
        full = pd.read_parquet(db.export_options_snapshot("2026-10-09", "SPX", tmp_path / "all"))
        assert len(full) == 3


class TestDatesAndConfig:
    def test_datetime_index_named_date_is_accepted(self):
        df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-09-01"], name="Date"))
        assert db._normalise_date_column(df)["date"].tolist() == ["2026-09-01"]

    def test_utc_instant_after_midnight_keeps_the_ny_session_date(self):
        ts = pd.to_datetime(["2026-10-10 01:30"]).tz_localize("UTC")     # Fri 21:30 ET
        assert db._normalise_date_column(pd.DataFrame({"date": ts}))["date"].tolist() == ["2026-10-09"]

    def test_importing_config_creates_no_directories(self, tmp_path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        shutil.copy(REPO / "joint_vol_calibration" / "config.py", pkg / "config.py")
        spec = importlib.util.spec_from_file_location("config_under_test", pkg / "config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["pkg"]
        assert sorted(p.name for p in pkg.iterdir() if p.name != "__pycache__") == ["config.py"]

    def test_backtest_window_and_dead_constants(self):
        assert (config.BT_START, config.BT_END) == ("2018-01-01", "2025-03-24")
        for dead in ("BACKTEST_START", "BACKTEST_END", "TRAIN_YEARS", "TEST_MONTHS", "SPXW_TICKER",
                     "SIGNAL_VOL_EDGE_THRESHOLD", "SIGNAL_VIX_EDGE_THRESHOLD", "LOG_DIR", "LOG_LEVEL"):
            assert not hasattr(config, dead), dead


# ══════════════════════════════════════════════════════════════════════════════
# Provenance backfill script (data-provenance-stale-chain-fits)
# ══════════════════════════════════════════════════════════════════════════════

def _load_script(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestProvenanceBackfill:
    PARAMS = {"kappa": 2.0, "theta": 0.05, "sigma": 0.4, "rho": -0.7, "v0": 0.02}
    SNAPS = ["2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-09-11"]

    def _write(self, d, date, **extra):
        with open(d / f"joint_cal_{date}.pkl", "wb") as f:
            pickle.dump({"params": dict(self.PARAMS), "as_of_date": date, **extra}, f)

    def _bytes(self, d):
        return {p.name: p.read_bytes() for p in sorted(d.glob("*.pkl"))}

    def _read(self, d, date):
        with open(d / f"joint_cal_{date}.pkl", "rb") as f:
            return pickle.load(f)

    def test_marks_stale_chains_never_overwrites_and_is_idempotent(self, tmp_path):
        mod = _load_script("backfill_calibration_provenance")
        d = tmp_path / "cal"
        d.mkdir()
        self._write(d, "2026-05-31")                                       # legacy showcase (a Sunday)
        self._write(d, "2026-06-23", n_spx_options=54, n_vix_tenors=4)
        self._write(d, "2026-09-04", spot_date="2026-09-04", n_spx_options=40, n_vix_tenors=4)
        # a live-recorded fit, shaped like the real 09-11 pickle (no `provenance` key)
        self._write(d, "2026-09-11", spot_date="2026-09-11", spx_snapshot_date="2026-09-11",
                    n_spx_options=9, n_vix_tenors=4, feller_state="PASS", feller_margin=0.1,
                    pinned_params=[])
        untouched = self._bytes(d)["joint_cal_2026-09-11.pkl"]

        assert mod.backfill(apply=True, cal_dir=d, snapshots=self.SNAPS) == 3
        after_first = self._bytes(d)
        assert after_first["joint_cal_2026-09-11.pkl"] == untouched
        assert "provenance" not in self._read(d, "2026-09-04")             # its counts were live

        showcase = self._read(d, "2026-05-31")
        assert (showcase["spx_snapshot_date"], showcase["stale_spx_chain"]) == ("2026-05-29", False)
        assert showcase["n_spx_options"] is None
        jun23 = self._read(d, "2026-06-23")
        assert (jun23["spx_snapshot_date"], jun23["spx_chain_age_days"], jun23["stale_spx_chain"]) == \
            ("2026-06-18", 5, True)
        sep04 = self._read(d, "2026-09-04")
        assert (sep04["spx_snapshot_date"], sep04["spx_chain_age_days"]) == ("2026-06-18", 78)
        assert sep04["n_spx_options"] == 40                                 # recorded value kept

        assert mod.backfill(apply=True, cal_dir=d, snapshots=self.SNAPS) == 0
        assert self._bytes(d) == after_first
