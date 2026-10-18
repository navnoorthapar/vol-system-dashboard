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
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
from joint_vol_calibration.data import database as dbm
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
    ts = pd.DataFrame(ts_rows)
    tb = pd.DataFrame({"date": days, "rate": 0.0400})

    monkeypatch.setattr(dl, "download_spx_ohlcv", lambda start, end: spx.copy())
    monkeypatch.setattr(dl, "download_vix_index", lambda start, end: pd.DataFrame())
    monkeypatch.setattr(dl, "download_vix_term_structure", lambda start, end: ts.copy())
    monkeypatch.setattr(dl, "download_tbill_rate", lambda start, end: tb.copy())
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    return cr


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
    code, st = _run(env, "market,calibrate",
                    calibrate_kwargs={"de_maxiter": 15, "de_popsize": 6, "verbose": False})
    cal = st["calibration"]
    assert cal["status"] == "accepted", cal["reason"]
    assert code == 0
    pkl = env.DATA / "calibrations" / f"joint_cal_{SESSION}.pkl"
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    assert res["model_version"] == 2 and res["spot_date"] == SESSION
    assert res["n_spx_expiries"] == 6
    # rates are the ones the fit used (parity), not a post-hoc overwrite
    assert res["r"] == pytest.approx(0.042, abs=2e-3)
    assert res["r_reference"] == pytest.approx(0.04075, abs=1e-4)          # ^IRX 4.00% -> cc
    assert res["r_reference_source"].startswith("^IRX")
    n, n_two_sided = _spx_snapshot_rows(SESSION)
    assert n > 0 and n_two_sided == n                           # only clean quotes stored
    assert st["session"] == SESSION and st["headline_calibration"]["date"] == SESSION
    assert "calibration=accepted" in output.read_text()
    assert f"session={SESSION}" in output.read_text()
    assert "accepted" in summary.read_text()
    # idempotent re-run
    code2, st2 = _run(env, "calibrate")
    assert code2 == 0 and st2["calibration"]["status"] == "exists"


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
