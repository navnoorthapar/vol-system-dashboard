"""
Tests for scripts/cloud_refresh.py — the daily pipeline (C22 outage fixes).

What broke in production (2026-09-08..09-23): every run was green while the
calibration was rejected (or degenerate) on an empty after-hours book; an
empty SPX download carried on with the committed DB's June close; rejected
chains were stored as "the latest snapshot"; importing the module changed the
process CWD. These tests run main() end to end against a temporary SQLite DB
and a fake yfinance ticker — no network.
"""
import importlib.util
import json
import logging
import os
import pathlib
import pickle
import sqlite3
import sys
from datetime import UTC, date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest import forward_ledger as fl
from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
from joint_vol_calibration.data import cboe_downloader as cboe
from joint_vol_calibration.data import database as dbm
from joint_vol_calibration.data import rates
from joint_vol_calibration.data import yf_downloader as dl
from joint_vol_calibration.models.heston import heston_vix_term_index
from joint_vol_calibration.tests.test_joint_calibrator import _TRUE, _heston_chain

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cloud_refresh.py"
NOW = datetime(2026, 9, 24, 1, 30, tzinfo=timezone.utc)      # Wed 09-23 21:30 ET
SESSION = "2026-09-23"


def _load():
    spec = importlib.util.spec_from_file_location("cloud_refresh_test_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Import hygiene ────────────────────────────────────────────────────────────

def test_import_has_no_side_effects(tmp_path, monkeypatch):
    """Pre-C22 the import chdir'ed to the repo root, configured root logging
    and froze TODAY, making other tests order-dependent."""
    monkeypatch.chdir(tmp_path)
    handlers = list(logging.getLogger().handlers)
    mod = _load()
    assert pathlib.Path.cwd() == tmp_path
    assert logging.getLogger().handlers == handlers
    assert not hasattr(mod, "TODAY")


def test_latest_session_ignores_intraday_bar():
    cr = _load()
    spx = pd.DataFrame({"date": ["2026-09-22", "2026-09-23"], "close": [1.0, 2.0]})
    during = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)       # 14:00 ET
    after = datetime(2026, 9, 23, 20, 30, tzinfo=timezone.utc)       # 16:30 ET
    assert cr.latest_session_date(spx, during) == "2026-09-22"
    assert cr.latest_session_date(spx, after) == "2026-09-23"


def test_unknown_step_rejected():
    cr = _load()
    with pytest.raises(SystemExit):
        cr._parse_args(["--steps", "market,bogus"])


# ── End-to-end harness ────────────────────────────────────────────────────────

def _business_days(end: str, n: int):
    d, out = date.fromisoformat(end), []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d -= timedelta(days=1)
    return sorted(out)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temporary DB + data dir; market downloads faked; returns the module."""
    cr = _load()
    data = tmp_path / "data_store"
    (data / "calibrations").mkdir(parents=True)
    (data / "parquet").mkdir()
    monkeypatch.setattr(cr, "DATA", data)
    monkeypatch.setattr(dbm, "DB_PATH", data / "vol_system.db")
    monkeypatch.setattr(dbm, "PARQUET_DIR", data / "parquet")
    monkeypatch.setattr(dbm, "MARKET_DIR", data / "market")
    monkeypatch.setattr(dbm, "OPTIONS_DIR", data / "options")
    monkeypatch.chdir(tmp_path)              # main() chdirs; restored afterwards
    dbm.init_database()

    days = [d for d in _business_days(SESSION, 30) if d != "2026-09-07"]   # Labor Day
    spx = pd.DataFrame({"date": days, "open": 7600.0, "high": 7650.0, "low": 7550.0,
                        "close": np.linspace(7500, 7600, len(days)), "volume": 1e9})
    ts_rows = []
    for d in days:
        for t, n in VIX_INDEX_TENOR_DAYS.items():
            ts_rows.append({"date": d, "ticker": t, "tenor_days": n, "is_vvix": 0,
                            "close": heston_vix_term_index(_TRUE["kappa"], _TRUE["theta"],
                                                           _TRUE["sigma"], _TRUE["v0"], n / 365)})
    ts = pd.DataFrame(ts_rows).assign(source="yfinance")      # as yf_downloader labels it
    tb = pd.DataFrame({"date": days, "rate": 0.0400})

    monkeypatch.setattr(dl, "download_spx_ohlcv", lambda start, end: spx.copy())
    monkeypatch.setattr(dl, "download_vix_index", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(dl, "download_vix_term_structure", lambda start, end: ts.copy())
    monkeypatch.setattr(dl, "download_tbill_rate", lambda start, end: tb.copy())
    monkeypatch.setattr(cboe, "download_vix_history", lambda: pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "source"]))       # no network
    # Cboe's VIX-family files have not posted these sessions (so every Yahoo
    # row is provisional), and the Treasury / NY Fed feeds return nothing (so
    # there is no discount curve and fits fall back to ^IRX): no network.
    monkeypatch.setattr(cboe, "download_vix_term_structure_history",
                        lambda start=None, end=None, tickers=None: pd.DataFrame(
                            columns=["date", "ticker", "tenor_days", "close", "is_vvix", "source"]))
    monkeypatch.setattr(rates, "download_treasury_year", lambda year, kind: pd.DataFrame(
        columns=["date", "series", "value", "source"]))
    monkeypatch.setattr(rates, "download_nyfed_sofr", lambda start=None, end=None: pd.DataFrame(
        columns=["date", "series", "value", "source"]))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("REFRESH_SLOT", raising=False)          # no run record unless a test asks
    monkeypatch.delenv("CBOE_BOOKS_DIR", raising=False)        # no private book store unless a test asks
    # The post-fit steps have their own tests below; here an accepted fit records
    # a ledger from fixed signals (the synthetic DB is too short for the real
    # ones) and the derived artifacts are a no-op.
    monkeypatch.setattr(fl, "signals_for_session", lambda session, start_date=None: _SIGNALS)
    monkeypatch.setattr(cr, "_real_refresh_session_artifacts", cr.refresh_session_artifacts,
                        raising=False)
    monkeypatch.setattr(cr, "refresh_session_artifacts", lambda session, headline, **kw: {})
    def no_capture(und, now=None, **kw):
        raise OSError("no network in tests")
    monkeypatch.setattr(cboe, "capture_option_chain", no_capture)
    # Cboe's CFE settlement file of any session (_vx_settlement_csv); the VX
    # per-contract files are unreachable: no network
    monkeypatch.setattr(cboe, "download_vx_settlements", lambda trade_date: cboe.parse_vx_settlement_csv(
        _vx_settlement_csv(trade_date), trade_date))
    monkeypatch.setattr(cboe, "download_vix_futures_daily_history",
                        lambda start=None, end=None, expiries=None: pd.DataFrame())
    return cr


_MONTH_CODES = "FGHJKMNQUVXZ"


def _vx_settlement_csv(session: str) -> str:
    """A CFE settlement file for ``session`` in Cboe's format: the nine monthly VX
    contracts from the session's month on, in contango, at 4 dp and moving from
    session to session, plus a weekly and a VXM mini line the job leaves out."""
    y, m = int(session[:4]), int(session[5:7])
    exps = [e for e in (dbm.vx_monthly_expiry(y + (m - 1 + k) // 12, (m - 1 + k) % 12 + 1) for k in range(11))
            if e >= session][:9]
    shift = 0.0137 * (date.fromisoformat(session).toordinal() % 29)
    lines = ["Product,Symbol,Expiration Date,Price"]
    for i, e in enumerate(exps):
        lines.append(f"VX,VX/{_MONTH_CODES[int(e[5:7]) - 1]}{e[3]},{e},{17.5 + 0.45 * i + shift:.4f}")
    lines.append(f"VX,VX40/X6,{exps[0]},{30.0:.4f}")                 # a weekly: not stored
    lines.append(f"VXM,VXM/X6,{exps[1]},{17.0:.4f}")                 # the mini: not VX
    return "\n".join(lines) + "\n"


def _vx_monthlies(session: str) -> list[tuple[str, float]]:
    """(expiry, settle) of the monthly contracts in _vx_settlement_csv(session), from its text."""
    out = []
    for ln in _vx_settlement_csv(session).splitlines()[1:]:
        prod, sym, exp, px = ln.split(",")
        if prod == "VX" and sym.startswith("VX/"):
            out.append((exp, float(px)))
    return out


# every sleeve flat except S4 short: the ledger opens one straddle
_SIGNALS = {s: {"position": 0.0, "kelly": 0.0} for s in ("s1", "s1c", "s2", "s3")}
_SIGNALS["s4"] = {"position": -1.0, "kelly": 0.10}


class _FakeTicker:
    """yfinance.Ticker stand-in serving a Heston-priced (or empty) book."""
    empty_book = False

    def __init__(self, symbol):
        start = date.fromisoformat(SESSION)
        self.options = tuple((start + timedelta(days=i)).isoformat()
                             for i in range(1, 400) if (start + timedelta(days=i)).weekday() == 4)

    def option_chain(self, exp):
        df = _heston_chain(expiries=[exp], S=7600.0)
        if self.empty_book:
            df["bid"] = 0.0
            df["ask"] = 0.0
        calls = df[df.option_type == "call"].drop(columns=["expiration", "option_type"])
        puts = df[df.option_type == "put"].drop(columns=["expiration", "option_type"])
        return type("Chain", (), {"calls": calls, "puts": puts})()


class _EmptyBookTicker(_FakeTicker):
    empty_book = True


class _DailyBookTicker(_FakeTicker):
    """The real SPX listing: one expiry EVERY weekday (SPXW dailies), so the
    first 8 listed expiries span ~1.5 weeks (the pre-C22 outage). Each slice
    also carries one two-sided quote too wide for the cleaner (100% relative
    spread) and one one-sided line, like a real book."""

    def __init__(self, symbol):
        start = date.fromisoformat(SESSION)
        self.options = tuple((start + timedelta(days=i)).isoformat()
                             for i in range(1, 400) if (start + timedelta(days=i)).weekday() < 5)

    def option_chain(self, exp):
        ch = super().option_chain(exp)
        calls = ch.calls.reset_index(drop=True)
        far = calls.index[calls["strike"] > 1.15 * 7600.0][0]
        calls.loc[far, "ask"] = 3.0 * calls.loc[far, "bid"]          # wide: rel spread 100%
        calls.loc[far + 1, "bid"] = 0.0                               # one-sided
        return type("Chain", (), {"calls": calls, "puts": ch.puts})()


def _spx_snapshot_rows(session):
    import sqlite3
    con = sqlite3.connect(str(dbm.DB_PATH))
    try:
        return con.execute("SELECT COUNT(*), SUM(bid > 0 AND ask >= bid) FROM options_snapshots "
                           "WHERE underlying='SPX' AND snapshot_date=?", (session,)).fetchone()
    finally:
        con.close()


def _accepted_pickle(cr, session):
    res = {"model_version": 2, "spot_date": session, "as_of_date": session,
           "params": {"kappa": 5.0, "theta": 0.05, "sigma": 0.7, "rho": -0.9, "v0": 0.015},
           "leg_losses": {"spx_iv_rmse": 1.4, "vix_index_rmse": 0.6},
           "n_spx_expiries": 6, "n_spx_options": 54, "n_put_wing_expiries": 6,
           "n_vix_tenors": 4, "spx_snapshot_date": session, "spx_snapshot_session": session}
    with open(cr.DATA / "calibrations" / f"joint_cal_{session}.pkl", "wb") as f:
        pickle.dump(res, f)


def _run(cr, steps, ticker=_FakeTicker, **kw):
    status = cr.DATA.parent / "status.json"
    code = cr.main(["--steps", steps, "--status-file", str(status)], now_utc=NOW,
                   ticker_factory=ticker, **kw)
    return code, json.loads(status.read_text())


# ── Exit status and fail-closed behaviour ─────────────────────────────────────

def test_empty_spx_download_is_a_hard_failure(env, monkeypatch):
    monkeypatch.setattr(dl, "download_spx_ohlcv", lambda start, end: pd.DataFrame())
    code, st = _run(env, "market,labels,calibrate")
    assert code == 1
    assert any("SPX OHLCV" in e for e in st["errors"])
    assert st["calibration"]["status"] == "skipped"          # never labels a stale session
    assert _spx_snapshot_rows(SESSION)[0] == 0


def test_expected_session_missing_is_a_hard_failure(env, monkeypatch):
    stale = pd.DataFrame({"date": ["2026-06-17", "2026-06-18"], "open": 1.0, "high": 1.0,
                          "low": 1.0, "close": [7500.0, 7500.58], "volume": 1.0})
    monkeypatch.setattr(dl, "download_spx_ohlcv", lambda start, end: stale)
    code, st = _run(env, "market,calibrate")
    assert code == 1
    assert any("expected NYSE session 2026-09-23 is missing" in e for e in st["errors"])
    assert _spx_snapshot_rows("2026-06-18")[0] == 0           # no mislabelled chain


def test_empty_book_skips_and_persists_nothing(env):
    """Empty after-hours book: fail closed at the quote-state gate, store no
    snapshot, warn; exit 0 while the headline calibration is fresh."""
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    cal = st["calibration"]
    assert cal["status"] == "skipped" and "no live two-sided quotes" in cal["reason"]
    assert _spx_snapshot_rows(SESSION)[0] == 0
    assert not (env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl").exists()
    assert pathlib.Path(cal["diagnostic_chain"]).exists()       # kept for post-mortem, outside the DB
    assert st["headline_calibration"]["date"] == "2026-09-22"
    assert st["headline_calibration"]["age_sessions"] == 1
    # a dead book is not a ledger session either: nothing registered, so a
    # later slot with a live book can still record it
    assert st["forward_ledger"]["status"] == "no_live_book"
    assert "not a live book" in st["forward_ledger"]["reason"]
    assert not (env.DATA / "forward").exists()
    assert any("forward ledger" in w for w in st["warnings"])
    assert code == 0


def test_rejection_with_stale_headline_fails_the_run(env):
    _accepted_pickle(env, "2026-09-11")                          # 8 sessions old
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    assert st["headline_calibration"]["age_sessions"] == 8
    assert code == 1
    assert any("sessions old" in e for e in st["errors"])


def test_legacy_pickles_do_not_count_as_accepted(env):
    legacy = {"params": {"kappa": 5.0, "theta": 0.0431, "sigma": 0.657, "rho": -0.95,
                         "v0": 0.01088},
              "leg_losses": {"spx_iv_rmse": 0.215, "vix_futures_rmse": 0.27},
              "n_spx_options": 9, "n_vix_tenors": 4, "spot_date": "2026-09-22",
              "spx_snapshot_date": "2026-09-22"}
    with open(env.DATA / "calibrations" / "joint_cal_2026-09-22.pkl", "wb") as f:
        pickle.dump(legacy, f)
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    assert st["headline_calibration"]["date"] is None
    assert code == 1


def test_accepted_calibration_end_to_end(env, monkeypatch, tmp_path):
    summary, output = tmp_path / "summary.md", tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker,
                    calibrate_kwargs={"de_maxiter": 15, "de_popsize": 6, "verbose": False})
    cal = st["calibration"]
    assert cal["status"] == "accepted", cal["reason"]
    assert code == 0
    # a daily listing: only the target expiries (+ neighbours) are downloaded
    assert len(cal["chain"]["requested"]) <= 3 * 6
    pkl = env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl"
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    assert res["model_version"] == 2 and res["spot_date"] == SESSION
    assert res["n_spx_expiries"] == 6
    # Rates are the ones the fit used: the medians of the per-expiry parity
    # rates it priced off (the market's 4.2%), not a post-hoc overwrite with
    # the bill rate (4.00%).
    assert res["r"] == pytest.approx(float(np.median([e["r"] for e in res["anchors_by_expiry"]])),
                                     rel=1e-12)
    assert res["r"] == pytest.approx(0.042, abs=3e-4)
    # tbill_rates.rate is already continuously compounded: used as is (the
    # first C22 version converted it again, 4.00% -> 4.075%)
    assert res["r_reference"] == pytest.approx(0.0400, abs=1e-12)
    assert res["r_reference_source"].startswith("^IRX") and "cc" in res["r_reference_source"]
    assert res["chain_diagnostics"]["r_ref"] == res["r_reference"]     # one rate, cleaning and fit
    # Wed 21:30 ET, inside the GTH window, but Yahoo records no quote time
    # (QUANT-01): the run used to stamp its own clock as the quote time and
    # measure T from it. The download instant is fetched_at, T runs from the
    # 16:00 close, and the stored snapshot records no quote time.
    fetched = cal["chain"]["fetched_at"]
    assert NOW <= datetime.fromisoformat(fetched) < NOW + timedelta(minutes=10)
    assert cal["chain"]["quote_time"] is None
    assert cal["chain"]["source"] == "yfinance"
    assert cal["chain"]["t_origin"] == "2026-09-23T16:00:00-04:00"
    assert res["methods"]["T_origin"] == "2026-09-23T16:00:00-04:00"
    assert f"records no quote time; downloaded {fetched}" in res["methods"]["T"]
    assert "Global Trading Hours" not in res["methods"]["T"]
    assert "residual timing mismatch" not in res["methods"]["vix_leg_timing"]
    assert res["anchors_by_expiry"][0]["T_days"] == pytest.approx((date.fromisoformat(
        res["anchors_by_expiry"][0]["expiry"]) - date.fromisoformat(SESSION)).days, abs=1e-9)
    # Stored: the observed two-sided quotes (incl. the one too wide for the
    # cleaner), never a one-sided line.
    n, n_two_sided = _spx_snapshot_rows(SESSION)
    assert n > 0 and n_two_sided == n
    assert n > res["chain_diagnostics"]["n_clean"]
    assert st["session"] == SESSION and st["headline_calibration"]["date"] == SESSION
    out = output.read_text()
    assert "calibration=accepted" in out and f"session={SESSION}" in out
    assert "publishable=false" in out                   # no freeze step in this run
    text = summary.read_text()
    assert "accepted" in text and "out-of-sample" in text and "Feller" in text
    # the accepted chain is persisted for the workflow to commit
    chain_file = dbm.options_snapshot_path(SESSION, "SPX")
    assert chain_file.exists() and str(chain_file) in st["persisted"]
    assert len(pd.read_parquet(chain_file)) == n
    stored = pd.read_parquet(chain_file)
    assert stored["quote_time"].isna().all()                   # never the download instant
    assert res["chain_diagnostics"]["fetched_at"] == fetched   # which the fit records
    # The committed chain reproduces the committed fit's inputs: a calibrator
    # built from the stored snapshot (fresh DB loaded from the parquet) has the
    # same carry, anchors and out-of-sample surface as the live run.
    dbm.DB_PATH.unlink()
    dbm._SCHEMA_READY.clear()
    dbm.init_database()
    env.restore_captures()
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
    again = JointCalibrator(as_of_date=SESSION)
    prov = again._provenance()
    assert prov["chain_diagnostics"]["t_origin"] == res["chain_diagnostics"]["t_origin"]
    assert prov["carry_by_expiry"] == res["carry_by_expiry"]
    assert prov["anchors_by_expiry"] == res["anchors_by_expiry"]
    assert len(again.spx_surface_full) - len(again.spx_surface) == res["oos"]["n_spx_oos"]
    # idempotent re-run
    code2, st2 = _run(env, "calibrate")
    assert code2 == 0 and st2["calibration"]["status"] == "exists"


# ── Book slot: forward ledger, Cboe captures, derived artifacts (C23) ─────────

def test_an_accepted_session_is_recorded_in_the_forward_ledger(env):
    """The forward test at executable prices: one canonical, hash-chained file
    per session (forward_ledger.record_session). With no Cboe capture stored
    the book is the Yahoo chain the calibrate step downloaded."""
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 0, st["errors"]
    led = st["forward_ledger"]
    assert led["ok"], led
    assert led["status"] == "recorded"
    assert led["book"] == "yahoo"
    path = env.DATA / "forward" / f"ledger_{SESSION}.json"
    assert led["path"] == str(path)
    ledger = fl.load_ledger(path)
    fl.validate_ledger(ledger)
    assert ledger["previous"] is None
    # Yahoo records no quote time (QUANT-01): none is claimed; the book was
    # downloaded after the 16:00 close (a pre-close download is refused)
    assert ledger["inputs"]["quote_after_close"] is None
    assert ledger["inputs"]["quote_time"] is None
    assert ledger["inputs"]["book"]["fetched_at"] == st["calibration"]["chain"]["fetched_at"]
    assert [p["sleeve"] for p in ledger["positions"]] == ["s4"]
    assert led["actions"] == ["s4 open"]
    # the ledger's chain hash is the committed snapshot's
    stored = pd.read_parquet(dbm.options_snapshot_path(SESSION, "SPX"))
    assert ledger["inputs"]["chain_sha256"] == fl.chain_sha256(fl.observed_book(stored))
    assert ledger["inputs"]["book"]["source"] == "yahoo"
    assert ledger["inputs"]["book"]["extra_expiries"] == []          # no open position yet
    # the retry slot: the session exists, nothing is rewritten
    before = path.read_bytes()
    _, st2 = _run(env, "calibrate")
    assert st2["calibration"]["status"] == "exists"
    assert st2["forward_ledger"]["status"] == "exists"
    assert path.read_bytes() == before


def test_a_ledger_failure_is_loud_but_keeps_the_fit_and_chain(env, monkeypatch, capsys):
    def clash(*a, **k):
        raise FileExistsError("a different ledger is registered for 2026-09-23")
    monkeypatch.setattr(fl, "record_session", clash)
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 1
    assert st["calibration"]["status"] == "accepted"
    assert (env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl").exists()
    assert str(dbm.options_snapshot_path(SESSION, "SPX")) in st["persisted"]
    assert any("forward ledger" in e for e in st["errors"])
    assert "::error title=Forward ledger::" in capsys.readouterr().out
    assert not st["publish_blockers"]


def _full_spx_capture(session=SESSION, quote_time="2026-09-24T01:15:00Z"):
    """A Cboe SPX capture as the bot stores it: every listed expiry (weekday
    dailies for two months, then monthlies) two-sided, in the
    options_snapshots schema."""
    start = date.fromisoformat(session)
    exps = [(start + timedelta(days=i)).isoformat() for i in range(1, 62)
            if (start + timedelta(days=i)).weekday() < 5]
    exps += ["2026-12-18", "2027-03-19"]
    raw = _heston_chain(expiries=exps, S=7600.0)
    frame = pd.DataFrame({
        "snapshot_date": session, "expiry": raw["expiration"], "strike": raw["strike"],
        "right": np.where(raw["option_type"] == "call", "C", "P"), "bid": raw["bid"], "ask": raw["ask"],
        "mid_price": 0.5 * (raw["bid"] + raw["ask"]), "root": "SPXW",
        "contract_symbol": raw["contractSymbol"],
        "settlement": "PM", "underlying_price": 7600.0, "quote_time": quote_time, "source": "cboe"})
    return cboe.ChainCapture(
        underlying="SPX", frame=frame, session=session, quote_time=quote_time, book_state="gth",
        fetched_at="2026-09-24T01:30:05Z", payload_timestamp="2026-09-24T01:30:00Z",
        underlying_last_trade_time="2026-09-23T16:14:59", underlying_price=7600.0,
        url="https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json", sha256="cd" * 32,
        n_bytes=4321)


def _fit_fails(monkeypatch):
    from joint_vol_calibration.calibration import joint_calibrator as jc

    def boom(self, *a, **k):
        raise RuntimeError("optimiser diverged")
    monkeypatch.setattr(jc.JointCalibrator, "calibrate", boom)


def test_the_ledger_trades_on_the_full_cboe_capture_whatever_the_fit_does(env, monkeypatch):
    """STATS-1/STATS-2: the ledger's book is the session's full Cboe capture
    (every listed expiry), not the calibration-expiry Yahoo download, and the
    session is recorded although the fit failed."""
    monkeypatch.setattr(cboe, "capture_option_chain",
                        lambda und, now=None, **kw: _full_spx_capture() if und == "SPX" else _capture(und))
    _fit_fails(monkeypatch)
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    assert st["calibration"]["status"] == "failed"
    assert code == 0                                         # headline 1 session old: warning only
    led = st["forward_ledger"]
    assert led["status"] == "recorded", led
    assert led["book"] == "cboe"
    ledger = fl.load_ledger(led["path"])
    pq_path = env.DATA / "options" / "cboe" / f"spx_{SESSION}_013005Z.parquet"
    capture = pd.read_parquet(pq_path)
    assert ledger["inputs"]["book"]["source"] == "cboe"
    assert ledger["inputs"]["book"]["file"] == f"options/cboe/{pq_path.name}"
    assert ledger["inputs"]["book"]["raw_sha256"] == "cd" * 32
    assert ledger["inputs"]["chain_sha256"] == fl.chain_sha256(fl.observed_book(capture))
    assert ledger["inputs"]["quote_time"] == "2026-09-24T01:15:00Z"
    assert ledger["inputs"]["chain_expiries"] == capture["expiry"].nunique()
    (pos,) = ledger["positions"]
    assert pos["sleeve"] == "s4"
    assert pos["contract"]["expiry"] in set(capture["expiry"])


def test_books_held_in_the_private_store_are_found_and_not_captured_again(env, monkeypatch, tmp_path):
    """CITSEC-06: with the private data repository configured, the workflow moves
    each raw book there (scripts/ci/cboe_private.py) and the public tree keeps a
    manifest. A later slot of the same session must neither capture the book
    again (a manifest, or the book in the private checkout, counts) nor lose it:
    the forward ledger marks on the book from the private checkout
    (CBOE_BOOKS_DIR, cboe_downloader.find_books)."""
    calls = []

    def fake(und, now=None, **kw):
        calls.append(und)
        return _full_spx_capture() if und == "SPX" else _capture(und)
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    _fit_fails(monkeypatch)
    _accepted_pickle(env, "2026-09-22")
    _code, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    assert st["forward_ledger"]["status"] == "recorded"
    first = fl.load_ledger(st["forward_ledger"]["path"])
    # what the workflow does next when configured: books to the private checkout, a manifest here
    pub = env.DATA / "options" / "cboe"
    priv = tmp_path / "private" / cboe.PRIVATE_BOOKS_SUBDIR
    priv.mkdir(parents=True)
    for pq_path in sorted(pub.glob("*.parquet")):
        cboe.write_manifest(pq_path, cboe.manifest_dir(env.DATA), "owner/books")
        for f in (pq_path, pq_path.with_suffix(".json")):
            f.rename(priv / f.name)
    assert not list(pub.glob("*"))
    pathlib.Path(st["forward_ledger"]["path"]).unlink()            # its commit failed
    monkeypatch.setenv("CBOE_BOOKS_DIR", str(priv))
    _code, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    assert calls == ["SPX", "VIX"]                                 # nothing fetched again
    assert st["capture"]["status"] == "exists"
    assert st["capture"]["captures"]["SPX"]["file"].startswith(str(priv))
    assert not list(pub.glob("*.parquet"))                         # nothing raw written publicly
    led = st["forward_ledger"]
    assert (led["status"], led["book"]) == ("recorded", "cboe")
    again = fl.load_ledger(led["path"])
    assert again["inputs"]["book"] == first["inputs"]["book"]      # the same book, read privately
    assert again["inputs"]["chain_sha256"] == first["inputs"]["chain_sha256"]
    # without the private checkout the manifest alone still says the book is stored
    monkeypatch.delenv("CBOE_BOOKS_DIR")
    _code, st = _run(env, "capture")
    assert st["capture"]["status"] == "exists"
    assert st["capture"]["captures"]["VIX"]["file"].endswith(".json")
    assert calls == ["SPX", "VIX"]


def test_every_later_book_slot_can_mark_what_the_ledger_opened():
    """The defect behind STATS-1, on the real expiry rule: a 30-day straddle
    opened on a session's book is outside the next sessions' calibration
    download (option_chain.select_calibration_expiries with fetch_spx_chain's
    one neighbour) on most sessions of its life, while a full listing (the
    Cboe capture) holds it on every session until it expires."""
    from joint_vol_calibration.data import option_chain as oc

    start = date.fromisoformat("2026-10-16")
    listing = [(start + timedelta(days=i)).isoformat() for i in range(120)
               if (start + timedelta(days=i)).weekday() < 5]
    opened = "2026-11-16"                                 # 31 days: the ledger's pick on 10-16
    sessions = [d for d in listing if "2026-10-19" <= d < opened]
    in_calibration_download = [d for d in sessions if opened in oc.select_calibration_expiries(
        [e for e in listing if e > d], d, neighbours=1)]
    assert len(in_calibration_download) <= len(sessions) // 3
    assert all(opened in [e for e in listing if e > d] for d in sessions)


def test_the_yahoo_fallback_adds_the_open_positions_expiries(env, monkeypatch):
    """Without a Cboe capture the ledger uses the Yahoo chain plus the
    expiries of its open straddles, so a target change closes the straddle at
    this session's bids instead of waiting for the expiry to re-enter the
    calibration download."""
    from joint_vol_calibration.data import option_chain as oc

    code, _ = _run(env, "market")                        # the DB: closes, VIX rows, ^IRX
    assert code == 0
    prev_session, exp = "2026-09-16", "2026-10-16"
    book = _heston_chain(expiries=[exp], S=7600.0)
    flat = {s: {"position": 0.0, "kelly": 0.0} for s in fl.SLEEVES}
    short_s4 = flat | {"s4": {"position": -1.0, "kelly": 0.1}}
    first = fl.step(None, prev_session, book, short_s4, r_ref=0.04)
    fl.write_ledger(first, env.DATA / "forward")
    assert [p["contract"]["expiry"] for p in first["positions"]] == [exp]
    raw, meta = oc.fetch_spx_chain(SESSION, ticker_factory=_DailyBookTicker)
    assert exp not in meta["requested"]                  # not a calibration expiry on 09-23
    monkeypatch.setattr(fl, "signals_for_session", lambda session, start_date=None: flat)
    # the defect: the calibration download alone cannot price the straddle
    alone = fl.step(first, SESSION, raw, flat, r_ref=0.04)
    assert {a["sleeve"]: a for a in alone["actions"]}["s4"]["reason_code"] == "expiry_not_in_book"
    led = env.record_forward_ledger(SESSION, {"source": "yahoo", "raw": raw, "symbol": meta["symbol"],
                                              "quote_time": "2026-09-24T01:30:00Z"},
                                    ticker_factory=_DailyBookTicker)
    assert led["status"] == "recorded", led
    ledger = fl.load_ledger(led["path"])
    assert ledger["inputs"]["book"]["extra_expiries"] == [exp]
    assert ledger["inputs"]["open_expiries_not_in_book"] == []
    assert {a["sleeve"]: a["action"] for a in ledger["actions"]}["s4"] == "close"
    (c,) = ledger["closed"]
    assert c["closed"]["how"] == "close"
    assert c["closed"]["pnl_exec"] is not None
    assert fl.verify_chain([first, ledger]) == []


def test_a_book_quoted_before_the_close_is_not_the_ledgers(env, monkeypatch):
    """A capture whose quotes predate the session's 16:00 ET close is skipped
    for the next candidate book, not booked at prices older than the signal."""
    early = _full_spx_capture(quote_time="2026-09-23T19:45:00Z")          # 15:45 ET
    monkeypatch.setattr(cboe, "capture_option_chain",
                        lambda und, now=None, **kw: early if und == "SPX" else _capture(und))
    _fit_fails(monkeypatch)
    _accepted_pickle(env, "2026-09-22")
    _, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    led = st["forward_ledger"]
    assert led["status"] == "recorded"
    assert led["book"] == "yahoo"                          # the next candidate
    assert any("before the 2026-09-23 close" in r for r in led["refused_books"])


def test_a_yahoo_book_downloaded_before_the_close_is_not_the_ledgers(env):
    """Yahoo records no quote time (QUANT-01), so the ledger can no longer
    read one off the Yahoo book; a book downloaded before the 16:00 ET close
    still cannot hold the closing quotes and is refused on its fetched_at."""
    from joint_vol_calibration.data import option_chain as oc

    code, _ = _run(env, "market")
    assert code == 0
    raw, meta = oc.fetch_spx_chain(SESSION, ticker_factory=_DailyBookTicker)
    early = {"source": "yahoo", "raw": raw, "symbol": meta["symbol"], "quote_time": None,
             "fetched_at": "2026-09-23T19:45:00Z"}                      # 15:45 ET
    led = env.record_forward_ledger(SESSION, early, ticker_factory=_DailyBookTicker)
    assert led["status"] == "no_live_book"
    assert "downloaded at 2026-09-23T19:45:00Z, before the 2026-09-23 close" in led["reason"]


def _capture(und, session=SESSION, rows=4):
    """A ChainCapture as cboe_downloader.capture_option_chain returns it."""
    frame = pd.DataFrame({"snapshot_date": session, "underlying": und, "root": f"{und}W",
                          "expiry": "2026-10-16", "strike": 7000.0 + 25 * np.arange(rows),
                          "right": "C", "bid": 10.0, "ask": 10.5, "quote_time": "2026-09-24T01:15:00Z",
                          "source": "cboe", "contract_symbol": [f"{und}W261016C{i}" for i in range(rows)]})
    return cboe.ChainCapture(
        underlying=und, frame=frame, session=session, quote_time="2026-09-24T01:15:00Z",
        book_state="gth", fetched_at="2026-09-24T01:30:05Z", payload_timestamp="2026-09-24T01:30:00Z",
        underlying_last_trade_time="2026-09-23T16:14:59", underlying_price=7600.0,
        url=f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{und}.json",
        sha256="ab" * 32, n_bytes=1234)


def test_the_cboe_books_are_captured_whatever_the_fit_does(env, monkeypatch, capsys):
    """The full SPX and VIX books, timestamped, under data_store/options/cboe/,
    stored even when the Yahoo book is empty and the calibration is skipped;
    one capture per underlying and session (the retry slot skips them)."""
    calls = []

    def fake(und, now=None, **kw):
        calls.append(und)
        return _capture(und)
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,capture,calibrate", ticker=_EmptyBookTicker)
    assert code == 0
    assert st["calibration"]["status"] == "skipped"
    capt = st["capture"]
    assert capt["status"] == "captured"
    d = env.DATA / "options" / "cboe"
    for und in ("spx", "vix"):
        pq_path = d / f"{und}_{SESSION}_013005Z.parquet"
        js = json.loads((d / f"{und}_{SESSION}_013005Z.json").read_text())
        assert str(pq_path) in st["persisted"]
        assert len(pd.read_parquet(pq_path)) == js["rows"] == 4
        assert js["book_state"] == "gth"
        assert js["sha256"] == "ab" * 32
        assert capt["captures"][und.upper()]["quote_time"] == "2026-09-24T01:15:00Z"
    assert not list(d.glob(".*.tmp"))
    # the captures are not options snapshots: the DB and the calibrator never see them
    assert _spx_snapshot_rows(SESSION)[0] == 0
    # retry slot: nothing fetched again, nothing rewritten
    code, st = _run(env, "capture")
    assert st["capture"]["status"] == "exists"
    assert calls == ["SPX", "VIX"]
    assert "persisted" not in st or not st["persisted"]


def _capture_fetched_at(cap, fetched):
    """``cap`` as the delayed feed serves it when fetched at ``fetched`` (UTC): its
    quotes QUOTE_DELAY older, in the book state those quotes are in."""
    import dataclasses
    quoted = fetched - cboe.QUOTE_DELAY
    qt = quoted.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dataclasses.replace(
        cap, frame=cap.frame.assign(quote_time=qt), quote_time=qt,
        book_state=cboe.book_state_at(quoted, cap.session),
        fetched_at=fetched.strftime("%Y-%m-%dT%H:%M:%SZ"),
        payload_timestamp=fetched.strftime("%Y-%m-%dT%H:%M:%SZ"))


def test_the_close_slot_stores_the_closing_book_and_the_book_slot_the_overnight_one(env, monkeypatch):
    """FUND_GRADE_PLAN 0.8: the close slot (market,labels,closebook) stores the
    session's closing books, the regular session's last quotes, which Cboe's
    delayed feed shows until Global Trading Hours open at 20:15 ET; GitHub starts
    that slot hours late, here at 20:05 ET, past midnight UTC. The book slot
    after it must still capture the overnight book (a closing book is not the
    book its retry skips), its retry must skip that one, and the forward ledger
    must trade on the book slot's book as before, not on the closing book, whose
    file name now sorts first."""
    clock = {"now": datetime(2026, 9, 24, 0, 5, 5, tzinfo=UTC)}     # Wed 20:05:05 ET
    calls = []

    def fake(und, now=None, **kw):
        calls.append(und)
        cap = _full_spx_capture() if und == "SPX" else _live_capture(und)
        return _capture_fetched_at(cap, clock["now"])
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    _fit_fails(monkeypatch)
    _accepted_pickle(env, "2026-09-22")
    monkeypatch.setattr(env, "extend_regime_labels", lambda as_of: 1)
    status = env.DATA.parent / "status_close.json"
    code = env.main(["--steps", "market,labels,closebook", "--status-file", str(status)],
                    now_utc=clock["now"])
    st = json.loads(status.read_text())
    assert code == 0
    capt = st["capture"]
    assert capt["status"] == "captured", capt
    d = env.DATA / "options" / "cboe"
    closing = d / f"spx_{SESSION}_000505Z.parquet"
    assert capt["captures"]["SPX"]["file"] == str(closing)
    assert capt["captures"]["SPX"]["book_state"] == capt["captures"]["VIX"]["book_state"] == "closing"
    assert str(closing) in st["persisted"]
    assert "forward_ledger" not in st                     # the ledger waits for the book slot
    # the book slot: Wednesday 21:30 ET, the Global Trading Hours book
    clock["now"] = NOW + timedelta(seconds=5)
    code, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    capt = st["capture"]
    assert capt["status"] == "captured", capt             # not "exists": the closing book is another
    overnight = d / f"spx_{SESSION}_013005Z.parquet"
    assert capt["captures"]["SPX"]["file"] == str(overnight)
    assert capt["captures"]["SPX"]["book_state"] == "gth"
    assert calls == ["SPX", "VIX", "SPX", "VIX"]
    assert sorted(p.name for p in d.glob("spx_*.parquet")) == [closing.name, overnight.name]
    led = st["forward_ledger"]
    assert led["status"] == "recorded", led
    assert led["book_file"] == f"options/cboe/{overnight.name}"
    assert fl.load_ledger(led["path"])["inputs"]["book"]["book_state"] == "gth"
    # the retry slot: both books stored, nothing fetched again
    clock["now"] = NOW + timedelta(hours=2)
    code, st = _run(env, "capture")
    assert st["capture"]["status"] == "exists"
    assert calls == ["SPX", "VIX", "SPX", "VIX"]


def _live_capture(und, session=SESSION):
    """_capture with its strikes about the underlying's price: a two-sided book
    by the quote-state gate (every near-ATM line two-sided)."""
    import dataclasses
    cap = _capture(und, session=session)
    return dataclasses.replace(cap, underlying_price=float(cap.frame["strike"].mean()))


def test_the_closebook_step_does_not_store_a_closing_book_that_is_not_live(env, monkeypatch, capsys):
    """MAJOR 2 of the 0.8 review: the closebook step stored any closing book of
    the session, even an emptied or one-sided one, which no surface or ledger
    can use (it fails their quote-state gate) and which would add about 0.9 MB of
    raw quotes to the public record every weekday. It now applies that gate
    first: a book whose near-ATM lines are not two-sided is rejected with the
    reason, which the run record keeps and a warning annotation shows; the
    other underlying's live book is stored."""
    import dataclasses
    at = datetime(2026, 9, 24, 0, 5, 5, tzinfo=UTC)                # Wed 20:05:05 ET: the closing book

    def fake(und, now=None, **kw):
        if und == "SPX":                                           # asks only: the book has emptied
            cap = _full_spx_capture()
            cap = dataclasses.replace(cap, frame=cap.frame.assign(bid=0.0))
        else:
            cap = _live_capture(und)
        return _capture_fetched_at(cap, at)
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    monkeypatch.setenv("REFRESH_SLOT", "close")
    status = env.DATA.parent / "status_close.json"
    env.main(["--steps", "closebook", "--status-file", str(status)], now_utc=at)
    st = json.loads(status.read_text())
    capt = st["capture"]
    assert capt["status"] == "partial", capt
    spx, vix = capt["captures"]["SPX"], capt["captures"]["VIX"]
    assert spx["status"] == "rejected"
    assert spx["reason"].startswith("not a live book (0% of near-ATM lines two-sided)")
    assert (vix["status"], vix["book_state"]) == ("captured", "closing")
    d = env.DATA / "options" / "cboe"
    assert [p.name.split("_")[0] for p in d.glob("*.parquet")] == ["vix"]
    rec = json.loads(next((env.DATA / "runs").glob("run_*_close.json")).read_text())
    assert rec["capture"]["captures"]["SPX"]["status"] == "rejected"
    assert rec["capture"]["captures"]["SPX"]["reason"] == spx["reason"]
    assert "::warning title=Cboe capture::" in capsys.readouterr().out
    assert any("SPX rejected (not a live book" in w for w in st["warnings"])


def test_the_ledger_keeps_the_book_slots_books_before_a_weekday_closing_book(env, monkeypatch):
    """MINOR 4 of the 0.8 review: the forward ledger's pre-registered protocol
    trades on the book slot's book, its Cboe capture else its Yahoo chain. On a
    Monday-Thursday session the close slot now also stores the closing book; when
    the book slot's capture failed, that closing book was taken ahead of the Yahoo
    chain. It now comes after it, and is used only when the Yahoo chain is not a
    live book either. After a Friday the closing book is the book slot's own book
    (the overnight book opens only on Sunday evening) and keeps its place."""
    import dataclasses
    clock = {"now": datetime(2026, 9, 24, 0, 5, 5, tzinfo=UTC)}      # Wed 20:05:05 ET: closing book

    def close_slot_only(und, now=None, **kw):
        if clock["now"] >= NOW:                                     # the book slot's download fails
            raise OSError("cdn.cboe.com unreachable")
        cap = _full_spx_capture() if und == "SPX" else _live_capture(und)
        return _capture_fetched_at(cap, clock["now"])
    monkeypatch.setattr(cboe, "capture_option_chain", close_slot_only)
    _fit_fails(monkeypatch)
    _accepted_pickle(env, "2026-09-22")
    status = env.DATA.parent / "status_close.json"
    env.main(["--steps", "market,labels,closebook", "--status-file", str(status)], now_utc=clock["now"])
    closing = json.loads(status.read_text())["capture"]["captures"]["SPX"]
    assert (closing["status"], closing["book_state"]) == ("captured", "closing")
    clock["now"] = NOW                                              # Wed 21:30 ET: the book slot
    _, st = _run(env, "market,capture,calibrate", ticker=_DailyBookTicker)
    assert st["capture"]["captures"]["SPX"]["status"] == "failed"
    assert st["forward_ledger"]["status"] == "recorded", st["forward_ledger"]
    assert st["forward_ledger"]["book"] == "yahoo"                  # not the close slot's closing book
    # neither of the book slot's books is live: the closing book, rather than no ledger
    (env.DATA / "forward" / f"ledger_{SESSION}.json").unlink()
    _, st = _run(env, "market,capture,calibrate", ticker=_EmptyBookTicker)
    assert st["forward_ledger"]["book_file"] == f"options/cboe/{pathlib.Path(closing['file']).name}"
    # Friday 2026-09-18: the closing book is the book slot's own, ahead of the Yahoo chain
    fri = "2026-09-18"
    cap = dataclasses.replace(_full_spx_capture(session=fri, quote_time="2026-09-18T22:10:00Z"),
                              book_state="closing", fetched_at="2026-09-19T06:30:00Z")
    pq_path, _ = cboe.save_capture(cap, env.DATA / "options" / "cboe")
    raw = _full_spx_capture(session=fri).frame
    led = env.record_forward_ledger(fri, {"source": "yahoo", "raw": raw, "symbol": "^SPX",
                                          "quote_time": None, "fetched_at": "2026-09-19T06:30:00Z"})
    assert (led["book"], led["book_file"]) == ("cboe", f"options/cboe/{pq_path.name}")


def test_the_closebook_step_stores_nothing_once_the_feed_shows_another_book(env, monkeypatch):
    """A close slot that GitHub starts after the feed has moved to the Global
    Trading Hours book stores nothing (skipped, with the reason): the book slot
    captures that book as before. A download that turns out to be newer than
    the clock said is not stored as the closing book either."""
    calls = []
    served = {"at": NOW}

    def fake(und, now=None, **kw):
        calls.append(und)
        return _capture_fetched_at(_capture(und), served["at"])
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    status = env.DATA.parent / "status.json"
    env.main(["--steps", "closebook", "--status-file", str(status)], now_utc=NOW)     # 21:30 ET
    capt = json.loads(status.read_text())["capture"]
    assert capt["status"] == "skipped"
    assert "not its closing book" in capt["reason"]
    assert "(quotes of 2026-09-23 21:15 ET)" in capt["reason"]
    assert calls == []
    # the clock says closing book (20:20 ET: quotes of 20:05), the payload is 20:31's
    at = datetime(2026, 9, 24, 0, 20, tzinfo=UTC)
    served["at"] = at + timedelta(minutes=11)
    env.main(["--steps", "closebook", "--status-file", str(status)], now_utc=at)
    capt = json.loads(status.read_text())["capture"]
    assert capt["status"] == "skipped", capt
    assert {c["status"] for c in capt["captures"].values()} == {"skipped"}
    assert "is the gth book" in capt["captures"]["SPX"]["reason"]
    assert calls == ["SPX", "VIX"]
    assert not list((env.DATA / "options" / "cboe").glob("*.parquet"))


def test_a_capture_of_another_session_or_a_failed_download_is_not_stored(env, monkeypatch, capsys):
    def fake(und, now=None, **kw):
        if und == "VIX":
            raise OSError("cdn.cboe.com unreachable")
        return _capture(und, session="2026-09-24")             # the book rolled already
    monkeypatch.setattr(cboe, "capture_option_chain", fake)
    code, st = _run(env, "capture")
    assert code == 0
    capt = st["capture"]
    assert capt["status"] == "failed"
    assert capt["captures"]["SPX"]["status"] == "rejected"
    assert "session 2026-09-24's, not 2026-09-23's" in capt["captures"]["SPX"]["reason"]
    assert capt["captures"]["VIX"]["status"] == "failed"
    assert not (env.DATA / "options" / "cboe").exists()
    assert "::warning title=Cboe capture::" in capsys.readouterr().out


def test_a_capture_write_error_fails_that_capture_not_the_run(env, monkeypatch, capsys):
    """PROD-7: only the download was guarded, so an OSError from save_capture (a
    full disk) propagated out of main(): no market or calibrate step, no status
    file, no GITHUB_OUTPUT. Now that underlying is "failed", what it left of its
    capture is removed (the retry slot captures it whole), and the run goes on."""
    monkeypatch.setattr(cboe, "capture_option_chain", lambda und, **kw: _capture(und))
    real = cboe.save_capture

    def full_disk(cap, out_dir):
        if cap.underlying != "SPX":
            return real(cap, out_dir)
        d = pathlib.Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        stem = cboe.capture_stem(cap)
        cap.frame.to_parquet(d / f"{stem}.parquet", index=False)      # renamed into place ...
        (d / f".{stem}.json.tmp").write_text("{")                     # ... then the JSON fails
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(cboe, "save_capture", full_disk)
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,capture,calibrate", ticker=_EmptyBookTicker)
    capt = st["capture"]
    assert capt["status"] == "partial"
    assert capt["captures"]["SPX"]["status"] == "failed"
    assert "No space left on device" in capt["captures"]["SPX"]["reason"]
    assert capt["captures"]["VIX"]["status"] == "captured"
    d = env.DATA / "options" / "cboe"
    assert sorted(p.name for p in d.iterdir()) == [f"vix_{SESSION}_013005Z.json",
                                                   f"vix_{SESSION}_013005Z.parquet"]
    stem = d / f"vix_{SESSION}_013005Z"
    assert capt["files"] == [str(stem.with_suffix(".parquet")), str(stem.with_suffix(".json"))]
    assert st["market"]["ok"]
    assert st["calibration"]["status"] == "skipped"  # the later steps ran
    assert code == 0
    assert not st["publish_blockers"]
    assert "::warning title=Cboe capture::Cboe chain capture partial" in capsys.readouterr().out
    # the retry slot captures SPX again (nothing of the failed capture is left to say "exists")
    monkeypatch.setattr(cboe, "save_capture", real)
    code, st = _run(env, "capture")
    assert st["capture"]["captures"]["SPX"]["status"] == "captured"
    assert st["capture"]["captures"]["VIX"]["status"] == "exists"


def test_a_crash_in_the_capture_step_does_not_skip_the_run(env, monkeypatch, capsys):
    def crash(now_utc, expected, out_dir=None, **kw):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(env, "capture_cboe_chains", crash)
    _accepted_pickle(env, "2026-09-22")
    _, st = _run(env, "market,capture,calibrate", ticker=_EmptyBookTicker)
    assert st["capture"]["status"] == "failed"
    assert "RuntimeError: unexpected" in st["capture"]["reason"]
    assert st["market"]["ok"]
    assert st["calibration"]["status"] == "skipped"
    assert st["session"] == SESSION
    assert "exit_code" in st
    assert "::warning title=Cboe capture::" in capsys.readouterr().out


def test_no_capture_while_the_book_is_another_sessions(env, monkeypatch):
    """Same refusal as the calibration: from the 09:30 ET open the live book
    is the new session's."""
    monkeypatch.setattr(cboe, "capture_option_chain", lambda und, **kw: _capture(und))
    status = env.DATA.parent / "status.json"
    during = NOW + timedelta(hours=13, minutes=30)                 # Thu 11:00 ET
    env.main(["--steps", "capture", "--status-file", str(status)], now_utc=during)
    capt = json.loads(status.read_text())["capture"]
    assert capt["status"] == "skipped"
    assert "refusing" in capt["reason"]


def test_session_artifacts_are_rebuilt_from_an_accepted_fit(env, monkeypatch):
    """After an accepted fit: the session's surface report, its diagnostics
    (the script, stubbed here), and the Greeks surface and scenario grid of the
    new headline fit, all written under data_store/."""
    monkeypatch.setattr(env, "refresh_session_artifacts", env._real_refresh_session_artifacts)
    ran = []

    def diagnostics(session, out_dir, timeout):
        ran.append((session, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{session}.json").write_text("{}")
    monkeypatch.setattr(env, "_run_diagnostics", diagnostics)
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 0, st["errors"]
    arts = st["artifacts"]
    assert {k: v["ok"] for k, v in arts.items()} == {"surface": True, "diagnostics": True,
                                                    "greeks": True, "scenario": True}
    sdir = env.DATA / "surface" / SESSION
    assert sorted(p.name for p in sdir.iterdir()) == ["residuals.parquet", "scorecard.json",
                                                     "surface.json", "vix_replication.json"]
    idx = json.loads((env.DATA / "surface" / "index.json").read_text())
    assert [r["session"] for r in idx["sessions"]] == [SESSION]
    assert ran == [(SESSION, env.DATA / "calibrations" / "diagnostics")]
    from joint_vol_calibration.greeks.risk_monitor import load_greeks_surface
    from joint_vol_calibration.greeks.scenario_grid import load_scenario_grid
    g = load_greeks_surface(env.DATA / "greeks" / "greeks_surface.parquet").attrs["provenance"]
    sgp = load_scenario_grid(env.DATA / "greeks" / "scenario_grid.parquet").attrs["provenance"]
    for prov in (g, sgp):
        assert prov["as_of_date"] == SESSION
        assert prov["source"]["calibration_file"].endswith(f"joint_cal_{SESSION}.pkl")


def test_a_failed_artifact_is_reported_and_loses_nothing(env, monkeypatch, capsys):
    """A surface-report crash fails the job (exit 1, annotated) but the fit,
    the chain, the ledger and the other artifacts stand, and the site may
    still publish."""
    from joint_vol_calibration.surface import report
    monkeypatch.setattr(env, "refresh_session_artifacts", env._real_refresh_session_artifacts)
    monkeypatch.setattr(env, "_run_diagnostics", lambda s, d, t: None)

    def boom(*a, **k):
        raise ValueError("SVI fit diverged")
    monkeypatch.setattr(report, "write_session", boom)
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 1
    assert st["artifacts"]["surface"] == {"ok": False, "error": "ValueError: SVI fit diverged",
                                          "seconds": st["artifacts"]["surface"]["seconds"]}
    assert st["artifacts"]["greeks"]["ok"]
    assert st["artifacts"]["scenario"]["ok"]
    assert st["calibration"]["status"] == "accepted"
    assert st["forward_ledger"]["ok"]
    assert str(dbm.options_snapshot_path(SESSION, "SPX")) in st["persisted"]
    assert not st["publish_blockers"]
    assert "::error title=Session artifacts::surface artifact" in capsys.readouterr().out


def _greeks_source(cr, name="greeks_surface.parquet", key=b"greeks_provenance"):
    import pyarrow.parquet as pq
    meta = pq.read_schema(cr.DATA / "greeks" / name).metadata
    return json.loads(meta[key])["source"]["calibration_file"]


def test_a_failed_artifact_is_rebuilt_by_the_next_run(env, monkeypatch):
    """PROD-5 / ENG-6: the artifacts were built only by the run that accepted the
    fit. The diagnostics timed out there; the retry slot found the pickle
    ('exists') and rebuilt nothing, so the session's diagnostics never existed.
    Every run with the calibrate step now backfills what is missing."""
    monkeypatch.setattr(env, "refresh_session_artifacts", env._real_refresh_session_artifacts)
    calls = []

    def timeout(session, out_dir, timeout):
        calls.append(session)
        raise TimeoutError("calibration_diagnostics.py: 1200 s")
    monkeypatch.setattr(env, "_run_diagnostics", timeout)
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 1
    assert st["calibration"]["status"] == "accepted"
    assert {k: v["ok"] for k, v in st["artifacts"].items()} == {
        "surface": True, "diagnostics": False, "greeks": True, "scenario": True}
    # the failed part is not retried in the same run, and nothing else is missing
    assert st["artifact_backfill"]["gaps"] == {}
    assert calls == [SESSION]
    diag = env.DATA / "calibrations" / "diagnostics" / f"{SESSION}.json"
    assert not diag.exists()

    def works(session, out_dir, timeout):
        calls.append(session)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{session}.json").write_text("{}")
    monkeypatch.setattr(env, "_run_diagnostics", works)
    code, st = _run(env, "calibrate")                          # the 04:30 retry
    assert st["calibration"]["status"] == "exists"
    bf = st["artifact_backfill"]
    assert bf["gaps"] == {SESSION: ["diagnostics"]}
    assert list(bf["rebuilt"]) == [SESSION]
    assert list(bf["rebuilt"][SESSION]) == ["diagnostics"]
    assert bf["rebuilt"][SESSION]["diagnostics"]["ok"]
    assert code == 0
    assert diag.exists()
    assert calls == [SESSION, SESSION]
    code, st = _run(env, "calibrate")                          # repaired: nothing to do
    assert code == 0
    assert st["artifact_backfill"] == {"gaps": {}, "rebuilt": {}, "deferred": []}
    assert calls == [SESSION, SESSION]
    code, st = _run(env, "market,labels")                      # the close slot never backfills
    assert "artifact_backfill" not in st


def _committed_surface(env, sess, version=None, vix_version=None):
    """data_store/surface/<sess>/ as a committed session holds it (surface.json
    and vix_replication.json stamped ``version``; None: no report_version, as
    every artifact written before the field), its chain, its diagnostics and
    its index row."""
    from joint_vol_calibration.data import database as dbm
    d = env.DATA / "surface" / sess
    d.mkdir(parents=True, exist_ok=True)
    for name, v in (("surface.json", version), ("vix_replication.json", vix_version or version)):
        doc = {"schema_version": 1, "session": sess}
        if v is not None:
            doc["report_version"] = v
        (d / name).write_text(json.dumps(doc))
    chain = dbm.options_snapshot_path(sess, "SPX")
    chain.parent.mkdir(parents=True, exist_ok=True)
    chain.write_bytes(b"chain")
    diag = env.DATA / "calibrations" / "diagnostics" / f"{sess}.json"
    diag.parent.mkdir(parents=True, exist_ok=True)
    diag.write_text("{}")
    idx = env.DATA / "surface" / "index.json"
    rows = json.loads(idx.read_text())["sessions"] if idx.exists() else []
    idx.write_text(json.dumps({"sessions": [*[r for r in rows if r["session"] != sess],
                                            {"session": sess}]}))


def test_a_surface_written_under_an_older_method_is_rebuilt(env, monkeypatch):
    """0.7 changed what a session's surface report says (its book, its T
    origin, its provenance). artifact_gaps rebuilt only MISSING surfaces, so
    a method change reached new sessions only. A committed surface.json or
    vix_replication.json stored under an older report_version (none: the
    artifacts before the field) is now a 'surface' gap, which the backfill
    rebuilds MAX_BACKFILL_SESSIONS a run; a current one is not."""
    from joint_vol_calibration.surface import report
    cur = report.REPORT_VERSION
    old, mixed, fresh = "2026-09-17", "2026-09-18", "2026-09-21"
    monkeypatch.setattr(env, "accepted_sessions",
                        lambda: {s: f"joint_cal_{s}.pkl" for s in (old, mixed, fresh)})
    _committed_surface(env, old)                                # predates report_version
    _committed_surface(env, mixed, cur, vix_version=cur - 1)    # one file left behind
    _committed_surface(env, fresh, cur)
    assert env.artifact_gaps(None) == {mixed: ["surface"], old: ["surface"]}
    (env.DATA / "surface" / old / "surface.json").write_text("{not json")
    assert env.artifact_gaps(None)[old] == ["surface"]          # unreadable: rebuilt too
    assert report.stored_report_version({}) == 1
    assert report.stored_report_version({"report_version": cur}) == cur
    _committed_surface(env, old, cur)
    _committed_surface(env, mixed, cur)
    assert env.artifact_gaps(None) == {}
    _committed_surface(env, fresh, cur + 1)                     # written by newer code: left alone
    assert env.artifact_gaps(None) == {}


def test_a_captured_session_without_an_accepted_fit_gets_its_surface(env, monkeypatch):
    """CITSEC-05: the surface was built only after an accepted Heston fit, so a
    session with a full Cboe book and a rejected (or no) fit had none. Now a
    session whose stored Cboe capture passes the quote-state gates
    (surface.quotes.cboe_book_refusal) is a 'surface' gap without a fit (no
    diagnostics: they need one); a session whose captures all fail the gates
    is not, and a current surface is left alone."""
    from joint_vol_calibration.surface import quotes as sqm
    from joint_vol_calibration.surface import report
    live, empty, done = "2026-09-17", "2026-09-18", "2026-09-21"
    cdir = env.DATA / "options" / "cboe"
    cdir.mkdir(parents=True)
    for s_ in (live, empty, done):
        (cdir / f"spx_{s_}_013000Z.parquet").write_bytes(b"capture")
        (cdir / f"vix_{s_}_013001Z.parquet").write_bytes(b"capture")
    monkeypatch.setattr(env, "accepted_sessions", dict)
    asked = []

    def refusal(s_):
        asked.append(s_)
        return "not a live book (12% of near-ATM lines two-sided)" if s_ == empty else None
    monkeypatch.setattr(sqm, "cboe_book_refusal", refusal)
    _committed_surface(env, done, report.REPORT_VERSION)
    assert sqm.captured_sessions() == [done, empty, live]
    assert env.artifact_gaps(None) == {live: ["surface"]}
    assert sorted(asked) == [live, empty]                      # a current surface is not re-checked
    _committed_surface(env, done)                              # ... a stale one is
    assert env.artifact_gaps(None) == {done: ["surface"], live: ["surface"]}
    # what the backfill then builds for it: the surface alone
    seen = []
    monkeypatch.setattr(env, "refresh_session_artifacts",
                        lambda s_, h, parts=None, **kw: seen.append((s_, parts)) or
                        {p: {"ok": True} for p in parts})
    env.backfill_session_artifacts(None)
    assert seen == [(done, ("surface",)), (live, ("surface",))]


def test_a_capture_whose_check_raises_is_no_gap_and_no_crash(env, monkeypatch):
    """artifact_gaps runs on every calibrate run: a capture whose quote-state
    check raises (an unforeseen read error) must not crash the backfill (which
    fails the run) on every run after it; the session is logged and skipped."""
    from joint_vol_calibration.surface import quotes as sqm
    cdir = env.DATA / "options" / "cboe"
    cdir.mkdir(parents=True)
    (cdir / "spx_2026-09-17_013000Z.parquet").write_bytes(b"capture")
    monkeypatch.setattr(env, "accepted_sessions", dict)

    def boom(s_):
        raise KeyError("strike")
    monkeypatch.setattr(sqm, "cboe_book_refusal", boom)
    assert env.artifact_gaps(None) == {}


def test_greeks_of_another_fit_are_rebuilt_from_the_headline(env, monkeypatch):
    """The artifact commit was dropped (gate failure) or a part failed: the
    Greeks surface and scenario grid named the previous headline fit. The next
    run rebuilds both from the headline, and a missing file the same way."""
    monkeypatch.setattr(env, "refresh_session_artifacts", env._real_refresh_session_artifacts)
    monkeypatch.setattr(env, "_run_diagnostics", lambda s, d, t: (d.mkdir(parents=True, exist_ok=True),
                                                                  (d / f"{s}.json").write_text("{}")))
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert code == 0, st["errors"]
    head = f"joint_cal_{SESSION}.pkl"
    assert _greeks_source(env).endswith(head)
    import pyarrow.parquet as pq
    g = env.DATA / "greeks" / "greeks_surface.parquet"
    t = pq.read_table(g)
    meta = dict(t.schema.metadata)
    prov = json.loads(meta[b"greeks_provenance"])
    prov["source"]["calibration_file"] = "data_store/calibrations/joint_cal_2026-09-22.pkl"
    meta[b"greeks_provenance"] = json.dumps(prov).encode()
    pq.write_table(t.replace_schema_metadata(meta), g)          # the previous headline's surface
    (env.DATA / "greeks" / "scenario_grid.parquet").unlink()   # and no scenario grid
    code, st = _run(env, "calibrate")
    bf = st["artifact_backfill"]
    assert bf["gaps"] == {SESSION: ["greeks", "scenario"]}
    assert {k: v["ok"] for k, v in bf["rebuilt"][SESSION].items()} == {"greeks": True, "scenario": True}
    assert code == 0
    assert _greeks_source(env).endswith(head)
    assert _greeks_source(env, "scenario_grid.parquet", b"scenario_provenance").endswith(head)
    assert not list((env.DATA / "greeks").glob(".*.tmp"))


def test_backfill_rebuilds_newest_first_and_defers_the_rest(env, monkeypatch):
    gaps = {"2026-09-23": ["diagnostics"], "2026-09-22": ["surface", "diagnostics"],
            "2026-09-21": ["surface"]}
    monkeypatch.setattr(env, "artifact_gaps", lambda head: dict(gaps))
    seen = []
    def rebuild(s, h, parts=None, **kw):
        seen.append((s, parts))
        return {p: {"ok": True} for p in parts}
    monkeypatch.setattr(env, "refresh_session_artifacts", rebuild)
    out = env.backfill_session_artifacts("h.pkl", skip=["2026-09-23"], max_sessions=1)
    assert seen == [("2026-09-22", ("surface", "diagnostics"))]
    assert out["deferred"] == ["2026-09-21"]
    assert list(out["gaps"]) == ["2026-09-22", "2026-09-21"]
    with pytest.raises(ValueError, match="unknown artifact part"):
        env._real_refresh_session_artifacts(SESSION, None, parts=("bogus",))


def test_the_backfill_keeps_to_its_time_budget(env, monkeypatch):
    """A job that reaches its 45-minute timeout is cancelled before the step that
    uploads the captures, so the backfill has a budget: a session is started only
    with a minute left, its diagnostics get at most what is left, the rest waits
    for the next run; main() shrinks the budget by what the run already used."""
    gaps = {"2026-09-23": ["diagnostics"], "2026-09-22": ["diagnostics"]}
    monkeypatch.setattr(env, "artifact_gaps", lambda head: dict(gaps))
    now = [0.0]
    seen = []

    def rebuild(s, h, parts=None, diagnostics_timeout=None, **kw):
        seen.append((s, diagnostics_timeout))
        now[0] += 500.0                                   # this session took 500 s
        return {p: {"ok": True} for p in parts}
    monkeypatch.setattr(env, "refresh_session_artifacts", rebuild)
    out = env.backfill_session_artifacts("h.pkl", budget_s=530.0, clock=lambda: now[0])
    assert seen == [("2026-09-23", 530.0)]
    assert out["deferred"] == ["2026-09-22"]                 # 30 s left: not started
    seen.clear()
    out = env.backfill_session_artifacts("h.pkl", budget_s=5000.0, clock=lambda: now[0])
    assert [s for s, _ in seen] == ["2026-09-23", "2026-09-22"]
    assert seen[0][1] == env.DIAGNOSTICS_TIMEOUT_S           # capped by the part's own timeout
    budgets = []
    monkeypatch.setattr(env, "backfill_session_artifacts",
                        lambda head, skip=(), budget_s=None, **kw: budgets.append(budget_s) or
                        {"gaps": {}, "rebuilt": {}, "deferred": []})
    _run(env, "calibrate")
    assert len(budgets) == 1
    assert 0 < budgets[0] <= env.BACKFILL_BUDGET_S


def test_a_greeks_write_that_dies_keeps_the_previous_surface(env, monkeypatch):
    """PROD-3: save_greeks_surface writes in place; a crash mid-write left a
    surface without its provenance, which the data gate then tripped on. The
    bot writes it under a temporary name and renames it into place."""
    from joint_vol_calibration.greeks import risk_monitor
    g = env.DATA / "greeks" / "greeks_surface.parquet"
    g.parent.mkdir(parents=True)
    g.write_bytes(b"previous surface")

    def dies(cal, out_path, *a, **k):
        assert pathlib.Path(out_path) != g
        pathlib.Path(out_path).write_bytes(b"half a parq")
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(risk_monitor, "build_surface_from_calibration", dies)
    res = env._real_refresh_session_artifacts(SESSION, "h.pkl", parts=("greeks",))
    assert not res["greeks"]["ok"]
    assert "No space left" in res["greeks"]["error"]
    assert g.read_bytes() == b"previous surface"
    assert sorted(p.name for p in g.parent.iterdir()) == ["greeks_surface.parquet"]


def test_gate_rejection_after_a_fit_persists_nothing(env, monkeypatch):
    """A chain that passes the quote-state gate but whose FIT the shared gate
    rejects (here: a one-tenor VIX leg) must leave no snapshot and no pickle;
    the raw chain goes to a diagnostics file outside the DB."""
    ts_all = dl.download_vix_term_structure(None, None)
    monkeypatch.setattr(dl, "download_vix_term_structure",
                        lambda start, end: ts_all[ts_all["ticker"] == "^VIX"].copy())
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,calibrate",
                    calibrate_kwargs={"de_maxiter": 5, "de_popsize": 5, "verbose": False})
    cal = st["calibration"]
    assert cal["status"] == "rejected" and "VIX leg under-identified" in cal["reason"]
    assert cal["summary"]["n_vix_tenors"] == 1
    assert _spx_snapshot_rows(SESSION)[0] == 0
    assert not (env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl").exists()
    assert not dbm.options_snapshot_path(SESSION, "SPX").exists()
    assert pathlib.Path(cal["diagnostic_chain"]).exists()
    assert code == 0                           # headline 1 session old: warning only
    # the strategy's trades do not depend on the fit (STATS-2): the session is
    # recorded in the forward ledger on the book it downloaded
    led = st["forward_ledger"]
    assert led["status"] == "recorded", led
    assert fl.load_ledger(led["path"])["session"] == SESSION


def test_accepted_fit_without_its_stored_chain_fails_the_run(env, monkeypatch):
    """The stored chain is the fit's committed input record: if it cannot be
    written the run must say so (exit 1), not log a warning."""
    def broken(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(dbm, "insert_options_snapshot", broken)
    code, st = _run(env, "market,calibrate",
                    calibrate_kwargs={"de_maxiter": 5, "de_popsize": 5, "verbose": False})
    assert st["calibration"]["status"] == "accepted"
    assert code == 1 and any("could not store the accepted" in e for e in st["errors"])


def test_intraday_run_never_stores_or_labels_the_open_session(env, monkeypatch):
    """A manual dispatch at 14:00 ET must download through the PREVIOUS
    session and label as of it: yfinance's daily history includes the bar
    still forming."""
    seen = {}
    full = dl.download_spx_ohlcv(None, None)

    def spx(start, end):
        seen["end"] = end
        return full[full["date"] <= end].copy()

    def labels(as_of):
        seen["labels_as_of"] = str(as_of)
        return 0
    monkeypatch.setattr(dl, "download_spx_ohlcv", spx)
    monkeypatch.setattr(env, "extend_regime_labels", labels)
    during = datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc)          # Wed 14:00 ET
    status = env.DATA.parent / "status.json"
    code = env.main(["--steps", "market,labels", "--status-file", str(status)], now_utc=during)
    assert code == 0
    assert seen["end"] == "2026-09-22" and seen["labels_as_of"] == "2026-09-22"
    assert SESSION not in _spx_dates()


# ── Session guard: the live book must belong to the session it is labelled ────

_GUARD_CASES = [
    # manual dispatch at Thu 14:00 ET (09-23 has no fit yet): Thursday's intraday book
    (datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc), "2026-09-23",
     "holds session 2026-09-24's quotes"),
    # Tue 02:30 UTC slot after the MLK holiday: the Mon-evening GTH book trades for Tuesday
    (datetime(2027, 1, 19, 2, 30, tzinfo=timezone.utc), "2027-01-15",
     "NYSE holiday 2027-01-18"),
    (datetime(2027, 1, 19, 13, 0, tzinfo=timezone.utc), "2027-01-15",     # Tue 08:00 ET
     "NYSE holiday 2027-01-18"),
    # normal Tue 02:30 UTC slot: Monday's session, GTH book (T from the quote time)
    (datetime(2026, 10, 20, 2, 30, tzinfo=timezone.utc), "2026-10-19", None),
    # weekend run: Friday's closing book
    (datetime(2026, 10, 18, 15, 0, tzinfo=timezone.utc), "2026-10-16", None),
    # Saturday 02:30 UTC slot: Friday evening keeps the closing book
    (datetime(2026, 10, 17, 2, 30, tzinfo=timezone.utc), "2026-10-16", None),
]


@pytest.mark.parametrize("now, expected, refused", _GUARD_CASES)
def test_session_guard(now, expected, refused):
    cr = _load()
    from joint_vol_calibration.data import option_chain as oc
    assert oc.last_closed_session(now).isoformat() == expected
    reason = cr.session_guard(now, expected)
    if refused is None:
        assert reason is None
    else:
        assert refused in reason and f"as {expected}" in reason


@pytest.mark.parametrize("now, expected, refused", _GUARD_CASES)
def test_calibrate_refuses_a_book_of_another_session(env, now, expected, refused):
    """recalibrate_heston with the DB current through ``expected``: a refused
    run is "skipped" before any download (no chain, no pickle, no snapshot);
    an allowed one downloads the book (here empty, so the quote-state gate
    skips it)."""
    days = [d.strftime("%Y-%m-%d") for d in dbm.nyse_sessions("2026-09-01", expected)]
    dbm.insert_spx_ohlcv(pd.DataFrame({"date": days, "open": 7600.0, "high": 7650.0,
                                       "low": 7550.0, "close": 7600.0, "volume": 1e9}))
    built = []

    def ticker(symbol):
        built.append(symbol)
        return _EmptyBookTicker(symbol)
    st = env.recalibrate_heston(now, ticker_factory=ticker)
    assert st["status"] == "skipped" and st["session"] == expected
    if refused is None:
        assert built and "no live two-sided quotes" in st["reason"]
    else:
        assert not built and refused in st["reason"]
    assert _spx_snapshot_rows(expected)[0] == 0
    assert not (env.DATA / "calibrations" / f"joint_cal_{expected}.pkl").exists()


def test_intraday_dispatch_does_not_store_todays_book_as_yesterdays(env):
    """Review finding: market,calibrate,freeze dispatched at 14:00 ET while
    the previous session had no fit downloaded TODAY's intraday chain and
    saved it as the previous session's calibration and snapshot (accepted,
    session 2026-09-23). Now: skipped with the reason, nothing stored, and no
    hard failure while the headline fit is fresh."""
    _accepted_pickle(env, "2026-09-22")
    during = datetime(2026, 9, 24, 18, 0, tzinfo=timezone.utc)          # Thu 14:00 ET
    status = env.DATA.parent / "status.json"
    code = env.main(["--steps", "market,calibrate", "--status-file", str(status)],
                    now_utc=during, ticker_factory=_FakeTicker,
                    calibrate_kwargs={"de_maxiter": 5, "de_popsize": 5, "verbose": False})
    st = json.loads(status.read_text())
    cal = st["calibration"]
    assert cal["status"] == "skipped", cal
    assert "holds session 2026-09-24's quotes" in cal["reason"]
    assert not (env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl").exists()
    assert _spx_snapshot_rows(SESSION)[0] == 0
    assert not dbm.options_snapshot_path(SESSION, "SPX").exists()
    assert code == 0 and st["headline_calibration"]["date"] == "2026-09-22"


def test_labels_store_missing_observables_as_nan_not_yesterdays(env, monkeypatch):
    """Label rows come from build_rule_label_frame. The forward-fill the step
    used to apply stored the previous session's ts_slope under later dates
    when ^VIX3M was missing; the rows must carry NaN, the rule label and the
    pdv_iv_spread source tag."""
    from joint_vol_calibration.signals.regime_classifier import FEATURE_COLS, build_rule_label_frame
    idx = pd.bdate_range("2024-01-01", periods=420)
    rng = np.random.default_rng(7)
    ret = rng.normal(0, 0.01, len(idx))
    spx = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "log_return": ret,
                        "close": 4000 * np.exp(np.cumsum(ret))})
    vix = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                        "^VIX": 16 + rng.normal(0, 1.5, len(idx)).cumsum() * 0.05,
                        "^VIX9D": 15.0, "^VIX3M": 18.0, "^VIX6M": 19.0,
                        "^VVIX": 95 + rng.normal(0, 4, len(idx))})
    vix.loc[vix.index[-2:], "^VIX3M"] = np.nan                # two missing prints
    monkeypatch.setattr(dbm, "get_spx_ohlcv", lambda as_of_date, start_date=None:
                        spx[spx["date"] <= as_of_date].reset_index(drop=True))
    monkeypatch.setattr(dbm, "get_vix_term_structure_wide", lambda as_of_date, start_date=None:
                        vix[vix["date"] <= as_of_date].reset_index(drop=True))
    frame = build_rule_label_frame(spx, vix, vvix_threshold=None)
    seed = frame.iloc[:-5][list(FEATURE_COLS) + ["regime"]]        # stored history
    (env.DATA / "signals").mkdir()
    seed.to_parquet(env.DATA / "signals" / "regime_labels.parquet")

    added = env.extend_regime_labels(date.fromisoformat(spx["date"].iloc[-1]))
    rl = pd.read_parquet(env.DATA / "signals" / "regime_labels.parquet")
    assert added == 5 and rl.index.max() == frame.index.max()
    tail = rl.iloc[-5:]
    assert tail["ts_slope"].iloc[-2:].isna().all()               # not the 3rd-last value
    assert (tail["regime"] == frame["regime"].iloc[-5:]).all()
    assert tail["pdv_iv_spread_source"].notna().all()


def test_rebuild_labels_restates_history_extend_cannot_reach(env, monkeypatch):
    """After a restatement of the record far back (C23: Cboe's VIX family
    replaced Yahoo's to 2010), every stored label row is rebuilt from the
    database. extend_regime_labels only heals the trailing window a 900-day
    fetch can label with a full 252-day gate, so an old wrong label survives
    it; rebuild_regime_labels repairs it and the features, and refuses (writing
    nothing) when the database cannot rebuild a stored date."""
    from joint_vol_calibration.signals.regime_classifier import FEATURE_COLS, build_rule_label_frame
    idx = pd.bdate_range("2023-01-02", periods=700)
    rng = np.random.default_rng(11)
    ret = rng.normal(0, 0.01, len(idx))
    spx = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "log_return": ret,
                        "close": 4000 * np.exp(np.cumsum(ret))})
    vix = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                        "^VIX": 16 + rng.normal(0, 1.5, len(idx)).cumsum() * 0.05,
                        "^VIX9D": 15.0, "^VIX3M": 18.0, "^VIX6M": 19.0,
                        "^VVIX": 95 + rng.normal(0, 4, len(idx))})
    served = {"vix": vix}
    monkeypatch.setattr(dbm, "get_spx_ohlcv", lambda as_of_date, start_date=None:
                        spx[spx["date"] <= as_of_date].reset_index(drop=True))
    monkeypatch.setattr(dbm, "get_vix_term_structure_wide", lambda as_of_date, start_date=None:
                        served["vix"][served["vix"]["date"] <= as_of_date].reset_index(drop=True))
    truth = build_rule_label_frame(spx, vix, vvix_threshold=None)
    stored = truth[[*FEATURE_COLS, "regime", "pdv_iv_spread_source"]].copy()
    old = stored.index[40]                                   # long before any heal window
    stored.loc[old, "regime"] = (int(stored.loc[old, "regime"]) + 1) % 3
    stored.loc[stored.index[41], "vvix"] += 0.5
    (env.DATA / "signals").mkdir()
    path = env.DATA / "signals" / "regime_labels.parquet"
    stored.to_parquet(path)
    as_of = date.fromisoformat(spx["date"].iloc[-1])

    assert env.extend_regime_labels(as_of) == 0
    assert int(pd.read_parquet(path).loc[old, "regime"]) != int(truth.loc[old, "regime"])

    out = env.rebuild_regime_labels(as_of)
    rl = pd.read_parquet(path)
    assert out["labels_changed"] == 1
    assert out["changed"][0][0] == str(old.date())
    assert out["features_changed"] >= 1
    assert out["rows_added"] == 0
    assert (rl["regime"] == truth["regime"]).all()
    np.testing.assert_allclose(rl["vvix"].to_numpy(float), truth["vvix"].to_numpy(float))

    served["vix"] = vix.iloc[300:]                           # a store that lost its history
    before = path.read_bytes()
    with pytest.raises(ValueError, match="cannot rebuild"):
        env.rebuild_regime_labels(as_of)
    assert path.read_bytes() == before


# ── Publishable vs exit status (the workflow deploys on publishable) ──────────

class _FreezeOK:
    returncode, stdout, stderr = 0, "", ""


def test_stale_calibration_fails_the_job_but_stays_publishable(env, monkeypatch, tmp_path):
    """Right after C22 merges the newest model_version 2 fit is months old: a
    rejected book-slot calibration must turn the run red without freezing the
    site on old market data; the dashboard shows the staleness banner."""
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _FreezeOK())
    _accepted_pickle(env, "2026-09-11")                           # 8 sessions old
    code, st = _run(env, "market,calibrate,freeze", ticker=_EmptyBookTicker)
    assert code == 1
    assert st["publishable"] is True and st["publish_blockers"] == []
    assert "publishable=true" in output.read_text()


@pytest.mark.parametrize("failure", ["market", "freeze", "no_freeze_step"])
def test_publish_blocking_failures(env, monkeypatch, failure):
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _FreezeOK())
    _accepted_pickle(env, "2026-09-22")
    steps = "market,calibrate,freeze"
    if failure == "market":
        monkeypatch.setattr(dl, "download_spx_ohlcv", lambda start, end: pd.DataFrame())
    elif failure == "freeze":
        class _R:
            returncode, stdout, stderr = 2, "", "boom"
        monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _R())
    else:
        steps = "market,calibrate"
    code, st = _run(env, steps, ticker=_EmptyBookTicker)
    assert st["publishable"] is False
    if failure != "no_freeze_step":
        assert code == 1 and st["publish_blockers"]


def test_close_slot_reports_the_headline(env, monkeypatch, tmp_path):
    """The close slot (market,labels) commits too; its message must name the
    real headline calibration, not 'none'."""
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(env, "extend_regime_labels", lambda as_of: 1)
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,labels")
    assert code == 0 and st["headline_calibration"]["date"] == "2026-09-22"
    assert "headline_calibration=2026-09-22" in output.read_text()


def test_each_workflow_run_writes_its_record(env, monkeypatch):
    """FUND_GRADE_PLAN 0.9 / VALIDATOR-15: what the bot attempted and why the gate
    said no lived only in the gitignored status JSON and in Actions logs that
    expire. A workflow run (REFRESH_SLOT) writes data_store/runs/run_<run_utc>_<slot>.json
    for the capture commit; every field is the run's own (compared with its
    status JSON, nothing pinned), and it is on disk before the freeze so the site
    this run renders reads it."""
    monkeypatch.setenv("REFRESH_SLOT", "book")
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    at_freeze = []

    def freeze(*a, **k):
        at_freeze.extend(json.loads(p.read_text()) for p in (env.DATA / "runs").glob("run_*.json"))
        return _FreezeOK()
    monkeypatch.setattr(env.subprocess, "run", freeze)
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,calibrate,freeze", ticker=_EmptyBookTicker)
    files = sorted((env.DATA / "runs").glob("run_*.json"))
    assert [f.name for f in files] == [env.run_record_name(st["run_utc"], "book")]
    rec = json.loads(files[0].read_text())
    assert (rec["schema"], rec["slot"], rec["complete"]) == ("run_ledger/1", "book", True)
    for key in ("run_utc", "steps", "session", "expected_session", "exit_code", "publishable",
                "publish_blockers", "errors", "warnings"):
        assert rec[key] == st[key], key
    cal = st["calibration"]
    assert rec["calibration"]["status"] == cal["status"] != "accepted"
    assert rec["calibration"]["reason"] == cal["reason"].splitlines()[0]
    assert rec["calibration"]["session"] == cal["session"]
    assert "near_atm_two_sided_share" in rec["calibration"]["chain"]
    assert all(cal["chain"][k] == v for k, v in rec["calibration"]["chain"].items())
    assert rec["forward_ledger"]["status"] == st["forward_ledger"]["status"]
    assert rec["exit_code"] == code
    assert rec["headline_calibration"]["session"] == st["headline_calibration"]["date"]
    assert (rec["github"]["sha"], rec["github"]["run_id"]) == ("f" * 40, "4242")
    assert {"restore", "market", "calibrate", "forward_ledger", "freeze", "total"} <= set(rec["timings_s"])
    assert all(v >= 0 for v in rec["timings_s"].values())
    assert st["run_record"].endswith(files[0].name)
    # before the freeze: the same run, not yet complete
    assert len(at_freeze) == 1
    assert (at_freeze[0]["complete"], at_freeze[0]["exit_code"]) == (False, None)
    assert at_freeze[0]["calibration"]["status"] == cal["status"]


def test_a_run_without_a_known_slot_writes_no_record(env, monkeypatch):
    """A local run (no REFRESH_SLOT) leaves the store as it was; an unknown slot
    name is a warning, not a file name."""
    _accepted_pickle(env, "2026-09-22")
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    assert code == 0
    assert "run_record" not in st
    assert not (env.DATA / "runs").exists()
    monkeypatch.setenv("REFRESH_SLOT", "../book")
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    assert not (env.DATA / "runs").exists()
    assert any("REFRESH_SLOT" in w for w in st["warnings"])


def test_a_run_record_that_cannot_be_written_costs_nothing_else(env, monkeypatch):
    monkeypatch.setenv("REFRESH_SLOT", "book")
    _accepted_pickle(env, "2026-09-22")

    def broken(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(env, "write_run_record", broken)
    code, st = _run(env, "market,calibrate", ticker=_EmptyBookTicker)
    assert code == 0
    assert any("could not write the run record: OSError: disk full" in w for w in st["warnings"])


def test_the_run_record_is_json_whatever_the_status_holds(env, tmp_path):
    """numpy scalars from the chain diagnostics serialise as numbers."""
    status = {"run_utc": "2026-10-28T02:31:05+00:00", "steps": ["calibrate"], "errors": [], "warnings": [],
              "calibration": {"status": "rejected", "reason": "a\nb", "session": "2026-10-27",
                              "chain": {"n_raw": np.int64(7), "near_atm_two_sided_share": np.float64(0.5)}}}
    p = env.write_run_record(status, "book", directory=tmp_path)
    assert p.name == "run_20261028T023105Z_book.json"
    rec = json.loads(p.read_text())
    assert rec["calibration"]["chain"] == {"n_raw": 7, "near_atm_two_sided_share": 0.5}
    assert rec["calibration"]["reason"] == "a"
    assert (rec["capture"], rec["forward_ledger"]) == (None, None)


def test_freeze_failure_is_a_hard_failure(env, monkeypatch):
    class _R:
        returncode, stdout, stderr = 2, "", "boom"
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _R())
    code, st = _run(env, "freeze")
    assert code == 1 and st["freeze"]["ok"] is False


def test_freeze_stderr_logged_on_success(env, monkeypatch, caplog):
    class _R:
        returncode, stdout, stderr = 0, "ok", "InconsistentVersionWarning: sklearn"
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **k: _R())
    with caplog.at_level(logging.WARNING, logger="cloud_refresh"):
        code, _ = _run(env, "freeze")
    assert code == 0
    assert any("InconsistentVersionWarning" in r.getMessage() for r in caplog.records)


# ── Persistence of captures (data_store/market, data_store/options) ───────────

def _spx_dates():
    return set(dbm.get_spx_ohlcv(as_of_date="2100-01-01")["date"].astype(str).str[:10])


def test_market_captures_survive_a_dropped_session_and_a_fresh_db(env, monkeypatch):
    """A session Yahoo later drops must survive: the run restores the committed
    partitions before downloading and merges its download back into them."""
    code, st = _run(env, "market")
    assert code == 0
    part = dbm.MARKET_DIR / "spx_ohlcv" / "spx_ohlcv_2026.parquet"
    assert part.exists() and str(part) in st["persisted"]
    assert "2026-09-22" in set(pd.read_parquet(part)["date"])

    # Next run: clean clone (fresh, empty DB) and Yahoo no longer serves 09-22.
    full = dl.download_spx_ohlcv(None, None)
    monkeypatch.setattr(dl, "download_spx_ohlcv",
                        lambda start, end: full[full["date"] != "2026-09-22"].copy())
    dbm.DB_PATH.unlink()
    dbm._SCHEMA_READY.clear()
    dbm.init_database()
    code, st = _run(env, "market")
    assert code == 0
    assert st["restored"]["market"]["spx_ohlcv"] > 0
    assert "2026-09-22" in _spx_dates()
    assert "2026-09-22" in set(pd.read_parquet(part)["date"])


def test_rederive_rebuilds_the_partitions_and_labels_over_mains_files(env, monkeypatch):
    """PROD-2: the push loop's answer to a binary conflict (commit_and_push.sh
    REDERIVE). After a market run, main is given a different copy of the 2026
    SPX partition (a row the run never downloaded; the run's last session
    missing). --rederive, with no download and no restore, merges the run's
    database into main's file, so main's extra row and the run's session are
    both there, and re-extends the labels as of the run's session."""
    code, _ = _run(env, "market")
    assert code == 0
    part = dbm.MARKET_DIR / "spx_ohlcv" / "spx_ohlcv_2026.parquet"
    run_part = pd.read_parquet(part)
    mains = run_part[run_part["date"] != SESSION].copy()
    extra = mains.iloc[[0]].assign(date="2026-01-02", close=1234.5)
    pd.concat([extra, mains]).sort_values("date").to_parquet(part, index=False)
    status = env.DATA.parent / "status.json"
    before = status.read_text()

    def no_network(*a, **k):
        raise AssertionError("rederive must not download or restore")
    monkeypatch.setattr(dl, "download_spx_ohlcv", no_network)
    monkeypatch.setattr(env, "restore_captures", no_network)
    labelled = []
    monkeypatch.setattr(env, "extend_regime_labels", lambda as_of: labelled.append(as_of) or 1)
    code = env.main(["--rederive", "--steps", "market,labels", "--status-file", str(status)], now_utc=NOW)
    assert code == 0
    got = pd.read_parquet(part)
    assert SESSION in set(got["date"])
    assert "2026-01-02" in set(got["date"])
    assert got.loc[got["date"] == "2026-01-02", "close"].item() == 1234.5
    assert len(got) == len(run_part) + 1
    assert labelled == [date.fromisoformat(SESSION)]
    assert status.read_text() == before                        # the run's status file is the run's
    # the market step alone: the labels are not touched
    labelled.clear()
    assert env.main(["--rederive", "--steps", "market,capture,calibrate,freeze"], now_utc=NOW) == 0
    assert labelled == []


def test_merge_main_data_rederives_mains_sessions_over_the_branchs_record(env, tmp_path, monkeypatch):
    """ENG-1 (scripts/ci/merge_main_data.py rederive): main's partition (the bot's
    provisional Yahoo rows) is upserted into
    the branch's database and the partition re-exported: main's new session
    is in, the branch's Cboe row for the same session stays, a row main
    stored on an NYSE holiday does not come in."""
    spec = importlib.util.spec_from_file_location(
        "merge_main_data_test_mod", SCRIPT.parent / "ci" / "merge_main_data.py")
    merge = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, merge)
    spec.loader.exec_module(merge)
    code, _ = _run(env, "market")
    assert code == 0
    rel = "data_store/market/vix_term_structure/vix_term_structure_2026.parquet"
    part = dbm.MARKET_DIR / "vix_term_structure" / "vix_term_structure_2026.parquet"
    cboe = pd.DataFrame([{"date": SESSION, "ticker": "^VIX", "tenor_days": 30, "close": 17.76,
                          "is_vvix": 0, "source": "cboe"}])
    dbm.insert_vix_term_structure(cboe)                        # the branch's restated print
    dbm.export_market_partitions(tables=("vix_term_structure",), since=SESSION)
    mains = pd.read_parquet(part)
    mains.loc[(mains["date"] == SESSION) & (mains["ticker"] == "^VIX"), ["close", "source"]] = \
        [20.37, "yfinance"]
    extra = mains[mains["date"] == SESSION].assign(date="2026-09-24", source="yfinance")
    holiday = mains[mains["date"] == SESSION].assign(date="2026-09-07", source="yfinance")
    theirs = tmp_path / "theirs"
    (theirs / rel).parent.mkdir(parents=True)
    pd.concat([mains, extra, holiday], ignore_index=True).to_parquet(theirs / rel, index=False)
    out = merge.rederive(theirs, [rel], env)
    assert out["tables"] == ["vix_term_structure"]
    assert "labels_added" not in out
    got = pd.read_parquet(part)
    vix = got[(got["date"] == SESSION) & (got["ticker"] == "^VIX")].iloc[0]
    assert (vix["close"], vix["source"]) == (17.76, "cboe")
    assert set(got.loc[got["date"] == "2026-09-24", "ticker"]) == set(extra["ticker"])
    assert "2026-09-07" not in set(got["date"])


def _vx_rows(session: str) -> list[tuple[str, float]]:
    con = sqlite3.connect(str(dbm.DB_PATH))
    try:
        return con.execute("SELECT expiry_date, settle FROM vix_futures_daily WHERE date = ? "
                           "ORDER BY expiry_date", (session,)).fetchall()
    finally:
        con.close()


def test_close_slot_stores_the_sessions_vx_settlements_in_the_record(env, monkeypatch):
    """FUND_GRADE_PLAN 1.12 / CITSEC-11: no step of the bot refreshed
    vix_futures_daily, so the Overview's VX curve was of 2026-10-09 on a page of
    2026-10-27. The close slot (market,labels) now stores the session's monthly VX
    settlements (the weekly and the VXM mini left out) and, from an empty
    partition directory, exports them to data_store/market/vix_futures_daily/,
    so the record holds them; the next run restores them from there."""
    monkeypatch.setattr(env, "extend_regime_labels", lambda as_of: 1)   # the env stores no labels
    vx_dir = dbm.MARKET_DIR / "vix_futures_daily"
    assert not vx_dir.exists()
    code, st = _run(env, "market,labels")
    assert code == 0
    vx = st["market"]["vx_futures"]
    want = _vx_monthlies(SESSION)
    assert (vx["ok"], vx["stored"], vx["last"], vx["missing"]) == (True, [SESSION], SESSION, [])
    assert vx["rows"] == len(want) == st["market"]["rows"]["vix_futures_daily"]
    assert _vx_rows(SESSION) == want
    part = vx_dir / f"vix_futures_daily_{SESSION[:4]}.parquet"
    assert str(part) in st["persisted"]
    rec = pd.read_parquet(part)
    assert sorted(zip(rec["expiry_date"], rec["settle"])) == want
    assert set(rec["date"]) == {SESSION}
    assert (rec["is_weekly"] == 0).all()
    assert not st["warnings"] or not any("VX" in w for w in st["warnings"])
    # a clean clone: the next run restores the session's settlements from the record
    dbm.DB_PATH.unlink()
    dbm._SCHEMA_READY.clear()
    dbm.init_database()
    code, st = _run(env, "market")
    assert code == 0
    assert st["restored"]["market"]["vix_futures_daily"] == len(want)
    assert st["market"]["vx_futures"]["sessions"] == []          # nothing left to fetch
    assert _vx_rows(SESSION) == want


def test_vx_refresh_fills_the_gap_from_the_newest_stored_session(env, monkeypatch):
    """The job fetches every NYSE session after the newest one the table holds
    (one settlement file each), not only today's: a slot that failed is made up
    by the next one."""
    stale = "2026-09-17"
    dbm.insert_vix_futures_daily(cboe.parse_vx_settlement_csv(_vx_settlement_csv(stale), stale))
    asked = []
    real = cboe.download_vx_settlements
    monkeypatch.setattr(cboe, "download_vx_settlements", lambda d: asked.append(d) or real(d))
    code, st = _run(env, "market")
    assert code == 0
    gap = ["2026-09-18", "2026-09-21", "2026-09-22", SESSION]
    assert asked == gap
    assert st["market"]["vx_futures"]["stored"] == gap
    for d in gap:
        assert _vx_rows(d) == _vx_monthlies(d)


@pytest.mark.parametrize("failure", ["raises", "not_posted", "stale_copy"])
def test_a_failed_vx_download_never_fails_the_close_slot(env, monkeypatch, failure):
    """A VX download that fails, a file Cboe has not posted, or a repeat of the
    previous session's prices stores nothing for the session, is reported as a
    warning, and the close slot's other steps (SPX, labels, the partitions) run
    and pass as before; the next slot retries."""
    prev = "2026-09-22"
    dbm.insert_vix_futures_daily(cboe.parse_vx_settlement_csv(_vx_settlement_csv(prev), prev))

    def broken(trade_date):
        if failure == "raises":
            raise OSError("Cboe is down")
        if failure == "not_posted":
            return cboe.parse_vx_settlement_csv("Product,Symbol,Expiration Date,Price\n", trade_date)
        return cboe.parse_vx_settlement_csv(_vx_settlement_csv(prev), trade_date)   # yesterday's prices
    monkeypatch.setattr(cboe, "download_vx_settlements", broken)
    monkeypatch.setattr(env, "extend_regime_labels", lambda as_of: 1)   # the env stores no labels
    code, st = _run(env, "market,labels")
    assert code == 0
    assert st["labels_added"] == 1
    assert not st["publish_blockers"]
    assert not st["errors"]
    vx = st["market"]["vx_futures"]
    assert (vx["ok"], vx["stored"], vx["missing"], vx["last"]) == (False, [], [SESSION], prev)
    assert _vx_rows(SESSION) == []
    warn = [w for w in st["warnings"] if "VX futures table ends " + prev in w]
    assert len(warn) == 1
    assert SESSION in warn[0]
    assert "the next slot retries" in warn[0]
    if failure == "stale_copy":
        assert "every price equals the previous session's" in warn[0]
    assert (dbm.MARKET_DIR / "spx_ohlcv" / "spx_ohlcv_2026.parquet").exists()


def test_no_persist_flag_leaves_capture_dirs_alone(env):
    status = env.DATA.parent / "status.json"
    code = env.main(["--steps", "market", "--status-file", str(status), "--no-persist"],
                    now_utc=NOW)
    assert code == 0
    assert not dbm.MARKET_DIR.exists()


def _vix_close(day):
    import sqlite3
    con = sqlite3.connect(str(dbm.DB_PATH))
    try:
        return con.execute("SELECT close, source FROM vix_daily WHERE date=?", (day,)).fetchone()
    finally:
        con.close()


def test_cboe_vix_print_overrides_yahoo(env, monkeypatch):
    """Cboe's VIX_History.csv is the official print (2026-02-06: Yahoo 20.37,
    Cboe 17.76); the bot overlays it like DataPipeline does. A Cboe failure
    keeps Yahoo's rows."""
    row = {"date": SESSION, "open": 17.0, "high": 21.0, "low": 16.0}
    monkeypatch.setattr(dl, "download_vix_index", lambda start, end: pd.DataFrame(
        [{**row, "close": 20.37, "source": "yfinance"}]))
    monkeypatch.setattr(cboe, "download_vix_history", lambda: pd.DataFrame(
        [{**row, "close": 17.76, "source": "cboe"},
         {**row, "date": "2026-09-07", "close": 99.0, "source": "cboe"}]))   # Labor Day
    assert _run(env, "market")[0] == 0
    assert _vix_close(SESSION) == (17.76, "cboe")
    assert _vix_close("2026-09-07") is None                  # not an NYSE session

    def boom():
        raise OSError("cdn.cboe.com unreachable")
    monkeypatch.setattr(cboe, "download_vix_history", boom)
    monkeypatch.setattr(dl, "download_vix_index", lambda start, end: pd.DataFrame(
        [{**row, "date": "2026-09-22", "close": 15.0, "source": "yfinance"},
         {**row, "close": 20.37, "source": "yfinance"}]))
    assert _run(env, "market")[0] == 0
    assert _vix_close("2026-09-22") == (15.0, "yfinance")       # no Cboe row: Yahoo fills it
    # Review finding: the outage run replaced the official closes with Yahoo's
    # for the whole window (275 rows) and committed them.
    assert _vix_close(SESSION) == (17.76, "cboe")
    part = pd.read_parquet(dbm.MARKET_DIR / "vix_daily" / "vix_daily_2026.parquet")
    assert part.loc[part["date"] == SESSION, ["close", "source"]].values.tolist() == [[17.76, "cboe"]]


def _ts_row(day, ticker):
    con = sqlite3.connect(str(dbm.DB_PATH))
    try:
        return con.execute("SELECT close, source FROM vix_term_structure WHERE date=? AND ticker=?",
                           (day, ticker)).fetchone()
    finally:
        con.close()


def test_vix_term_structure_is_cboe_primary(env, monkeypatch, capsys):
    """Every consumer (labels, backtest, calibrator VIX leg) reads
    vix_term_structure, and the bot used to fill it from Yahoo only (float32
    copies, NULL source, holiday rows); the Cboe overlay corrected vix_daily
    alone. Now Cboe's files are written on top: a session both print stores
    Cboe's value and a > 0.25 pt difference is reported (not a failure); a
    session Cboe has not posted yet keeps Yahoo's row, labelled 'yfinance'
    (provisional), until the next run after Cboe posts it."""
    yahoo = dl.download_vix_term_structure(None, None)
    yahoo.loc[(yahoo["date"] == "2026-09-22") & (yahoo["ticker"] == "^VIX"), "close"] = 20.37
    monkeypatch.setattr(dl, "download_vix_term_structure", lambda start, end: yahoo.copy())
    cb = yahoo[yahoo["date"] < SESSION].copy()                # Cboe lags one session
    cb["close"] = cb["close"].round(2)
    cb.loc[(cb["date"] == "2026-09-22") & (cb["ticker"] == "^VIX"), "close"] = 17.76
    cb["source"] = "cboe"
    monkeypatch.setattr(cboe, "download_vix_term_structure_history",
                        lambda start=None, end=None, tickers=None: cb.copy())
    code, st = _run(env, "market")
    assert code == 0
    fam = st["market"]["vix_family"]
    assert fam["ok"]
    assert not fam["errors"]
    assert _ts_row("2026-09-22", "^VIX") == (17.76, "cboe")
    assert _ts_row(SESSION, "^VIX")[1] == "yfinance"
    assert fam["n_provisional"] == 4
    assert all(p.startswith(SESSION) for p in fam["provisional"])
    assert fam["n_mismatches"] == 1
    assert fam["mismatches"][0] == {"date": "2026-09-22", "ticker": "^VIX", "cboe": 17.76,
                                    "yfinance": 20.37, "diff": pytest.approx(17.76 - 20.37)}
    out = capsys.readouterr().out
    assert "::warning title=VIX family cross-check::1 VIX-family print(s) differ" in out
    assert any("Cboe 17.76 vs Yahoo 20.37" in w for w in st["warnings"])
    part = pd.read_parquet(dbm.MARKET_DIR / "vix_term_structure" / "vix_term_structure_2026.parquet")
    assert part["source"].notna().all()                      # every persisted row says where it came from
    assert set(part.loc[part["date"] == SESSION, "source"]) == {"yfinance"}

    # Next run: Cboe has posted the session; its print replaces the provisional row.
    cb2 = pd.concat([cb, yahoo[yahoo["date"] == SESSION].assign(source="cboe", close=19.5)])
    monkeypatch.setattr(cboe, "download_vix_term_structure_history",
                        lambda start=None, end=None, tickers=None: cb2.copy())
    code, st = _run(env, "market")
    assert code == 0
    assert _ts_row(SESSION, "^VIX") == (19.5, "cboe")
    assert st["market"]["vix_family"]["n_provisional"] == 0


def test_a_cboe_outage_stores_yahoo_as_provisional_and_warns(env, monkeypatch, capsys):
    """The run still stores the session (Yahoo, labelled 'yfinance') and does
    not fail; a stored Cboe print is never replaced by Yahoo's."""
    def down(start=None, end=None, tickers=None):
        raise OSError("cdn.cboe.com unreachable")
    yahoo = dl.download_vix_term_structure(None, None)
    good = yahoo[yahoo["date"] == "2026-09-22"].assign(source="cboe", close=18.0)
    monkeypatch.setattr(cboe, "download_vix_term_structure_history",
                        lambda start=None, end=None, tickers=None: good.copy())
    assert _run(env, "market")[0] == 0
    monkeypatch.setattr(cboe, "download_vix_term_structure_history", down)
    code, st = _run(env, "market")
    assert code == 0
    fam = st["market"]["vix_family"]
    assert fam["ok"] is False
    assert "cdn.cboe.com unreachable" in fam["errors"][0]
    assert _ts_row(SESSION, "^VIX")[1] == "yfinance"
    assert _ts_row("2026-09-22", "^VIX") == (18.0, "cboe")
    assert "::warning title=VIX family download::" in capsys.readouterr().out


def _treasury_rows(days, level=0.0405):
    """A flat-ish Treasury day: bills (bank discount) and par yields."""
    rows = []
    for d in days:
        for w in (4, 8, 13, 26, 52):
            n = 7 * w
            # the bank-discount quote of a bill whose cc yield is `level`
            disc = (1 - np.exp(-level * n / 365)) * 360 / n
            rows.append({"date": d, "series": rates.bill_series(w, "BD"), "value": round(disc, 6)})
        rows.extend({"date": d, "series": f"UST_PAR_{tenor}", "value": level + 0.001}
                    for tenor in ("2Y", "5Y", "10Y"))
    return pd.DataFrame(rows).assign(source="treasury")


def test_rates_feed_the_curve_the_bot_fits_with(env, monkeypatch, tmp_path):
    """The bot's market step refreshes rates_daily (Treasury + NY Fed) and
    persists it; without that the zero curve goes stale 7 days after the last
    committed quote and every fit falls back to ^IRX. With a curve, the fit
    records it as its reference and cleans the chain with r(T) per expiry."""
    calls = []

    def treasury(year, kind):
        calls.append((year, kind))
        return _treasury_rows(_business_days(SESSION, 5)) if kind == "daily_treasury_bill_rates" \
            else pd.DataFrame(columns=["date", "series", "value", "source"])
    monkeypatch.setattr(rates, "download_treasury_year", treasury)
    code, st = _run(env, "market,calibrate", ticker=_DailyBookTicker)
    assert sorted(set(calls)) == [(2026, k) for k in rates.TREASURY_KINDS]
    curve = st["market"]["rates"]["curve"]
    assert curve["ok"]
    assert curve["curve_date"] == SESSION
    assert curve["r_91d"] == pytest.approx(0.0405, abs=1e-6)
    part = dbm.MARKET_DIR / "rates_daily" / "rates_daily_2026.parquet"
    assert str(part) in st["persisted"]
    assert len(pd.read_parquet(part)) == len(_treasury_rows(_business_days(SESSION, 5)))
    cal = st["calibration"]
    assert cal["status"] == "accepted", cal["reason"]
    assert code == 0
    with open(env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl", "rb") as f:
        res = pickle.load(f)
    assert res["rate_reference"]["kind"] == "treasury_curve"
    assert res["rate_reference"]["curve_date"] == SESSION
    assert res["r_reference"] == pytest.approx(curve["r_91d"], abs=1e-12)
    assert res["r_reference_source"].startswith(f"Treasury zero curve {SESSION}")
    assert res["chain_diagnostics"]["r_ref"] == "r(T) per expiry"
    assert cal["summary"]["r_reference_source"] == res["r_reference_source"]


def test_no_curve_is_a_warning_and_the_fit_falls_back_to_irx(env, capsys):
    code, st = _run(env, "market")
    assert code == 0
    assert st["market"]["rates"]["curve"]["ok"] is False
    assert "::warning title=Discount curve::no usable Treasury zero curve" in capsys.readouterr().out


def test_download_window_reaches_back_to_the_last_stored_session():
    """Pre-C22 the window was always today-400d, so once the stored history
    ended more than 400 days ago a permanent gap opened (the 2027-07-28 bomb)."""
    cr = _load()
    today = date(2027, 9, 1)
    assert cr.download_start(today, "2026-06-18") == date(2026, 6, 8)
    assert cr.download_start(today, "2027-08-31") == today - timedelta(days=400)
    assert cr.download_start(today, None) == today - timedelta(days=400)


def test_freeze_only_run_renders_from_the_restored_captures(env, monkeypatch):
    """A re-render (steps=freeze) must load the committed captures into the DB
    first; otherwise it publishes whatever stale history the DB file holds."""
    assert _run(env, "market")[0] == 0
    dbm.DB_PATH.unlink()
    dbm._SCHEMA_READY.clear()
    dbm.init_database()
    seen = {}

    class _R:
        returncode, stdout, stderr = 0, "", ""

    def fake_freeze(*a, **k):
        seen["dates"] = _spx_dates()
        return _R()
    monkeypatch.setattr(env.subprocess, "run", fake_freeze)
    code, st = _run(env, "freeze")
    assert code == 0
    assert SESSION in seen["dates"]
    assert "persisted" not in st                 # a re-render exports nothing
