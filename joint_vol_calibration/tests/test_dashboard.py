"""
Regression tests for the dashboard display layer (dashboard/app.py) and the
static-site freezer (dashboard/freeze_site.py) — correction round C22.

Every test pins a defect the C22 audit found on the live site:

  * the calibration selector headlined the LOWEST-RMSE archived fit, which
    rewards thin data (one SPX expiry, one VIX tenor);
  * the PDV percentile counted 4,125 NaN history rows as "not below" and read
    '1.1th percentile, vol cheap' for a spread above its own mean;
  * the PDV forecast was frozen at the pickle's 2025-12-30 training features and
    its staleness flag was computed but never rendered;
  * `python dashboard/app.py` overwrote the tracked regime_labels.parquet with the
    demoted classifier's predictions;
  * a broken research-classifier pickle returned HTTP 500 for the front page, and
    a missing label file made it headline the classifier as the rule label;
  * the classifier card was fed pdv_iv_spread = −VIX;
  * loader failures rendered zeros with confident prose and still deployed;
  * the smile chart and Greeks came from a different, gate-rejected fit;
  * KPIs double-counted round trips, the 'walk-forward' chart was an in-sample
    slice at a flat 5% hurdle clipped at ±4, annual returns did not compound;
  * the heatmap silently dropped 16 of 69 nodes;
  * every page declared the home page as its canonical URL.

Synthetic market data is seeded; no test touches the network or writes under
data_store/ (tests that need different inputs point the app at a temp copy).
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import itertools
import json
import pickle
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "data_store"


def _surface_file(name: str) -> Path:
    """``name`` in the session the Surface page renders (desk_artifacts.load_surface:
    the newest data_store/surface/<session>/ with a residual table), so a test
    holds on the bot's next session as well as on the committed one."""
    root = SHIPPED / "surface"
    sessions = sorted(p.name for p in root.iterdir()
                      if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
                      and (p / "residuals.parquet").exists())
    return root / sessions[-1] / name


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dash():
    """dashboard/app.py, imported once per module (tests monkeypatch DATA)."""
    return _load("dashboard_app_under_test", "dashboard/app.py")


@pytest.fixture(scope="module")
def freezer():
    return _load("freeze_site_under_test", "dashboard/freeze_site.py")


# ── synthetic inputs ──────────────────────────────────────────────────────────

CURVE = (("^VIX9D", 9), ("^VIX", 30), ("^VIX3M", 93), ("^VIX6M", 184))


# NYSE full-day closures as published by the exchange (independent of app.py's
# rule-based calendar, which test_nyse_calendar_matches_published_closures checks).
NYSE_CLOSURES_2022_2024 = [
    "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04",
    "2022-09-05", "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19",
    "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
]


def _sessions(dash, start: str, end: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, freq="C", holidays=NYSE_CLOSURES_2022_2024)


def test_nyse_calendar_matches_published_closures(dash):
    got = [str(h.date()) for h in dash.nyse_holidays("2022-01-01", "2024-12-31")]
    assert got == NYSE_CLOSURES_2022_2024
    assert dash.session_age("2026-09-04", "2026-09-08") == 1      # Labor Day 09-07
    assert dash.session_age("2026-09-08", "2026-09-04") == 0
    assert dash.stale_level(1) == "ok"
    assert dash.stale_level(2) == "warn"
    assert dash.stale_level(5) == "error"


def _write_market_db(path: Path, dates: pd.DatetimeIndex, seed: int = 11,
                     drop_subtenors_on_last: bool = False) -> dict:
    """A small vol_system.db with the tables the dashboard reads."""
    rng = np.random.default_rng(seed)
    n = len(dates)
    r = rng.normal(0.0, 0.01, n)
    r[0] = 0.0
    close = 4000.0 * np.exp(np.cumsum(r))
    vix = np.clip(16.0 + np.cumsum(rng.normal(0.0, 0.3, n)), 10.0, 45.0)
    vvix = np.clip(90.0 + rng.normal(0.0, 6.0, n), 60.0, 160.0)
    ds = [d.strftime("%Y-%m-%d") for d in dates]
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE spx_ohlcv (date TEXT PRIMARY KEY, open REAL, high REAL, "
                "low REAL, close REAL, volume INTEGER, log_return REAL)")
    con.execute("CREATE TABLE vix_daily (date TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL)")
    con.execute("CREATE TABLE vix_term_structure (date TEXT, ticker TEXT, tenor_days INTEGER, "
                "close REAL, is_vvix INTEGER, PRIMARY KEY (date, ticker))")
    # the research page's change-log facts average this per backtest year
    con.execute("CREATE TABLE tbill_rates (date TEXT PRIMARY KEY, rate REAL NOT NULL, discount_yield REAL)")
    con.executemany("INSERT INTO tbill_rates VALUES (?,?,?)",
                    [(d, 0.02 + 0.0001 * (i % 50), None) for i, d in enumerate(ds)])
    con.executemany("INSERT INTO spx_ohlcv VALUES (?,?,?,?,?,?,?)",
                    [(d, c, c, c, c, 1, (None if i == 0 else float(x)))
                     for i, (d, c, x) in enumerate(zip(ds, close, r))])
    con.executemany("INSERT INTO vix_daily VALUES (?,?,?,?,?)", [(d, v, v, v, v) for d, v in zip(ds, vix)])
    rows = []
    for i, (d, v, vv) in enumerate(zip(ds, vix, vvix)):
        for tk, ten in CURVE:
            if drop_subtenors_on_last and i == n - 1 and tk != "^VIX":
                continue
            rows.append((d, tk, ten, float(v * (0.9 + ten / 300.0)), 0))
        rows.append((d, "^VVIX", 30, float(vv), 1))
    con.executemany("INSERT INTO vix_term_structure VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return {"dates": dates, "r": pd.Series(r, index=dates), "vix": pd.Series(vix, index=dates)}


def _write_labels(path: Path, dates: pd.DatetimeIndex, seed: int = 3) -> pd.DataFrame:
    """Rule labels whose pdv_iv_spread is NaN except on the last 20 rows (as shipped)."""
    rng = np.random.default_rng(seed)
    rl = pd.DataFrame({"regime": rng.choice([0, 1, 2], size=len(dates), p=[0.1, 0.7, 0.2])},
                      index=pd.DatetimeIndex(dates, name="date"))
    spread = np.full(len(dates), np.nan)
    spread[-20:] = rng.normal(-0.04, 0.01, 20)
    rl["pdv_iv_spread"] = spread
    path.parent.mkdir(parents=True, exist_ok=True)
    rl.to_parquet(path)
    return rl


def _cal(session: str, rmse: float, n_vix: int = 4, n_spx: int = 54, **extra) -> dict:
    """A complete, well-identified model_version 2 result (Feller PASS, fresh snapshot, 6 expiries)."""
    cal = {
        "params": {"kappa": 2.2, "theta": 0.05, "sigma": 0.45, "rho": -0.8, "v0": 0.02},
        "leg_losses": {"spx_iv_rmse": rmse, "vix_futures_rmse": 0.3, "vix_options_rmse": 0.0},
        "n_spx_options": n_spx, "n_spx_expiries": max(1, n_spx // 9), "n_vix_tenors": n_vix,
        "spot_date": session, "spx_snapshot_date": session, "as_of_date": session,
        "S": 6000.0, "fit_time": 5.0, "n_evals": 1000, "success": True,
        "weights": {"w1": 0.5, "w2": 0.5, "w3": 0.0},
        "model_version": 2, "feller": "off",
    }
    cal.update(extra)
    return cal


def _write_cals(cal_dir: Path, cals: dict) -> None:
    cal_dir.mkdir(parents=True, exist_ok=True)
    for name, cal in cals.items():
        with open(cal_dir / name, "wb") as f:
            pickle.dump(cal, f)


@pytest.fixture
def synth(dash, tmp_path, monkeypatch):
    """A complete, small data_store in tmp_path; DATA points at it."""
    data = tmp_path / "data_store"
    (data / "signals").mkdir(parents=True)
    # long enough for the purged walk-forward PDV forecast to have > 252 sessions
    dates = _sessions(dash, "2019-01-02", "2024-06-28")
    mkt = _write_market_db(data / "vol_system.db", dates)
    labels = _write_labels(data / "signals" / "regime_labels.parquet", dates)
    shutil.copy(SHIPPED / "signals" / "regime_classifier.pkl", data / "signals")
    shutil.copy(SHIPPED / "pdv_model.pkl", data)
    for sub in ("greeks", "backtest"):
        shutil.copytree(SHIPPED / sub, data / sub)
    last = str(dates[-1].date())
    _write_cals(data / "calibrations", {f"joint_cal_{last}.pkl": _cal(last, 1.0)})
    monkeypatch.setattr(dash, "DATA", data)
    return {"data": data, "dates": dates, "labels": labels, **mkt}


def _tree_digest(root: Path) -> dict:
    """sha256 of every file under root, except SQLite's transient WAL sidecars.

    vol_system.db is in WAL mode, so any reader (read-only included) creates the
    gitignored -wal/-shm files; the database file itself must stay unchanged.
    """
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and not p.name.endswith(("-wal", "-shm"))}


# ── rendering is side-effect free ─────────────────────────────────────────────

def test_rendering_writes_nothing(dash):
    """Every loader and route on the shipped data leaves data_store byte-identical."""
    before = _tree_digest(SHIPPED)
    client = dash.app.test_client()
    for route in dash.ROUTES:
        assert client.get(route).status_code == 200
    assert _tree_digest(SHIPPED) == before


def test_app_has_no_writers_or_refresh_threads():
    """The C21 fix lived only in cloud_refresh: app.py still relabelled the
    tracked parquet with the DEMOTED classifier from a startup thread."""
    src = (ROOT / "dashboard" / "app.py").read_text()
    for forbidden in (".to_parquet(", "predict_series", "threading", "daily_refresh",
                      "_refresh_regime_labels", ".to_sql(", "pickle.dump"):
        assert forbidden not in src, f"display layer contains a writer: {forbidden}"


def test_db_connection_is_read_only(dash, tmp_path, monkeypatch):
    data = tmp_path / "data_store"
    data.mkdir()
    _write_market_db(data / "vol_system.db", _sessions(dash, "2024-01-02", "2024-02-28"))
    monkeypatch.setattr(dash, "DATA", data)
    with pytest.raises(sqlite3.OperationalError), dash._db() as con:
        con.execute("CREATE TABLE should_fail (a INTEGER)")


def test_local_preview_binds_loopback(monkeypatch):
    """`python dashboard/app.py` serves on 127.0.0.1 and starts no background work."""
    import runpy
    import threading

    import flask

    seen = {}
    monkeypatch.setattr(flask.Flask, "run", lambda self, **kw: seen.update(kw))
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("DASHBOARD_HOST", raising=False)
    n_threads = threading.active_count()
    runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
    assert seen.get("host") == "127.0.0.1"
    assert threading.active_count() == n_threads


# ── calibration selection ─────────────────────────────────────────────────────

def test_selector_takes_newest_acceptable_fit_not_lowest_rmse(dash, tmp_path, monkeypatch):
    data = tmp_path / "data_store"
    _write_cals(data / "calibrations", {
        "joint_cal_2026-07-01.pkl": _cal("2026-07-01", 0.20),            # best RMSE, older
        "joint_cal_2026-09-04.pkl": _cal("2026-09-04", 1.90),            # newest acceptable
        "joint_cal_2026-09-10.pkl": _cal("2026-09-10", 0.05, n_vix=1),   # newest, gate-rejected
    })
    monkeypatch.setattr(dash, "DATA", data)
    assert Path(dash.latest_good_calibration_path()).name == "joint_cal_2026-09-04.pkl"


def test_selector_never_falls_back_to_a_rejected_fit(dash, tmp_path, monkeypatch):
    data = tmp_path / "data_store"
    _write_cals(data / "calibrations", {
        "joint_cal_2026-09-10.pkl": _cal("2026-09-10", 0.05, n_vix=1),
        "joint_cal_2026-09-11.pkl": _cal("2026-09-11", 4.00),
    })
    monkeypatch.setattr(dash, "DATA", data)
    assert dash.latest_good_calibration_path() is None


def test_selector_calls_the_gate_with_the_full_result(dash, tmp_path, monkeypatch):
    """The gate receives the whole stored result (its single-source-of-truth
    form), so every check it knows about is applied, including ones whose
    keys the dashboard never names."""
    from joint_vol_calibration.calibration import joint_calibrator as jc

    seen = []

    def gate(result, *a, **k):
        seen.append(result)
        if result.get("n_spx_expiries", 9) < 4:
            return False, "too few expiries"
        return True, "ok"

    monkeypatch.setattr(jc, "is_acceptable_calibration", gate)
    data = tmp_path / "data_store"
    _write_cals(data / "calibrations", {
        "joint_cal_2026-09-04.pkl": _cal("2026-09-04", 1.9, n_spx_expiries=4),
        "joint_cal_2026-09-11.pkl": _cal("2026-09-11", 0.2, n_spx=9, n_spx_expiries=1),
    })
    monkeypatch.setattr(dash, "DATA", data)
    assert Path(dash.latest_good_calibration_path()).name == "joint_cal_2026-09-04.pkl"
    assert seen
    assert all("params" in r and "spot_date" in r for r in seen)


def test_selector_never_ranks_legacy_fits(dash, tmp_path, monkeypatch):
    """A pre-C22 (model_version 1) fit is newer and passes every numeric check,
    but its RMSEs are not comparable: it is labelled legacy and not ranked."""
    legacy = _cal("2026-10-09", 0.3)
    del legacy["model_version"]
    data = tmp_path / "data_store"
    _write_cals(data / "calibrations", {
        "joint_cal_2026-06-18.pkl": _cal("2026-06-18", 0.9),
        "joint_cal_2026-10-09.pkl": legacy,
    })
    monkeypatch.setattr(dash, "DATA", data)
    assert Path(dash.latest_good_calibration_path()).name == "joint_cal_2026-06-18.pkl"
    rows = {Path(r["path"]).name: r for r in dash.scan_calibrations()}
    assert rows["joint_cal_2026-10-09.pkl"]["legacy"] is True
    assert rows["joint_cal_2026-06-18.pkl"]["legacy"] is False


def test_calibration_page_shows_age_and_staleness_banner(dash, synth):
    """A headline fit 10 sessions older than the newest session is flagged."""
    dates = synth["dates"]
    old = str(dates[-11].date())
    cal_dir = synth["data"] / "calibrations"
    for p in cal_dir.glob("*.pkl"):
        p.unlink()
    _write_cals(cal_dir, {f"joint_cal_{old}.pkl": _cal(old, 1.0)})
    d = dash.load_page2()
    assert d["session"] == old
    assert d["calib_age"] == 10
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert 'class="banner banner-stale-error"' in html
    # no verb: 'Surface artifacts is 5 NYSE sessions old' read wrong for a plural label (F17)
    assert "Heston calibration: 10 NYSE sessions old" in html
    assert " is 10 NYSE sessions old" not in html


def test_legacy_fits_are_counted_and_never_headlined(dash, synth):
    """The page reports how many archived fits are legacy; none is the headline."""
    cal_dir = synth["data"] / "calibrations"
    old = str(synth["dates"][-2].date())
    last = str(synth["dates"][-1].date())
    legacy = _cal(last, 0.3)
    del legacy["model_version"]
    for p in cal_dir.glob("*.pkl"):
        p.unlink()
    _write_cals(cal_dir, {f"joint_cal_{old}.pkl": _cal(old, 1.0),
                          f"joint_cal_{last}.pkl": legacy})
    d = dash.load_page2()
    assert d["session"] == old
    assert d["legacy"] is False
    assert d["archive"] == {"n_total": 2, "n_legacy": 1, "n_current": 1, "n_current_ok": 1}
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "PRE-C22 FIT" not in html
    assert "1 pre-C22 fit is labelled legacy and never ranked" in html
    assert "1 pre-C22 fits" not in html                 # one fit, singular (the "1 sessions" defect)


def test_carry_is_shown_only_when_the_fit_recorded_it(dash, synth):
    """A lone stored r is a post-fit T-bill close, and q=0 was a display default."""
    cal_dir = synth["data"] / "calibrations"
    last = str(synth["dates"][-1].date())
    _write_cals(cal_dir, {f"joint_cal_{last}.pkl": _cal(last, 1.0, r=0.03913)})
    d = dash.load_page2()
    assert d["r_pct"] is None
    assert d["q_pct"] is None
    assert d["carry_recorded"] is False
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "SOFR" not in html
    assert "r = n/a" in html

    _write_cals(cal_dir, {f"joint_cal_{last}.pkl": _cal(last, 1.0, r=0.045, q=0.013)})
    d = dash.load_page2()
    assert (d["r_pct"], d["q_pct"]) == (4.5, 1.3)


@pytest.mark.parametrize("kind", ["treasury_curve", None])
def test_reference_rate_is_labelled_by_the_fits_record(dash, synth, kind):
    """Every fit before C23 discounted its fallback expiries at the ^IRX bill; a
    bot fit since then records the Treasury zero curve (rate_reference). The
    page printed '^IRX' for whatever r_reference held."""
    last = str(synth["dates"][-1].date())
    extra = {"r": 0.045, "q": 0.013, "r_reference": 0.0402,
             "r_reference_source": "^IRX 13-week bill, 4.0200% cc"}
    if kind:
        extra.update(r_reference_source="Treasury zero curve ...",
                     rate_reference={"kind": kind, "r_91d": 0.0402, "label": "x"})
    _write_cals(synth["data"] / "calibrations", {f"joint_cal_{last}.pkl": _cal(last, 1.0, **extra)})
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    text = _visible_text(html)
    if kind:
        assert "Treasury r(91d) 4.02%" in text
        assert "^IRX" not in text
    else:
        assert "^IRX 4.02%" in text
        assert "Treasury r(91d)" not in text


@pytest.mark.parametrize("recorded", [True, False])
def test_single_expiry_fit_is_flagged_not_called_good(dash, synth, monkeypatch, recorded):
    """Display check, independent of whether the gate admits such a fit."""
    from joint_vol_calibration.calibration import joint_calibrator as jc

    monkeypatch.setattr(jc, "is_acceptable_calibration", lambda params, *a, **k: (True, "ok"))
    last = str(synth["dates"][-1].date())
    extra = {} if recorded else {"n_spx_expiries": None}      # legacy: count not stored
    _write_cals(synth["data"] / "calibrations",
                {f"joint_cal_{last}.pkl": _cal(last, 0.2, n_spx=9, **extra)})
    d = dash.load_page2()
    assert d["n_spx_expiries"] == 1
    assert d["spx_underidentified"]
    assert d["n_spx_expiries_inferred"] is (not recorded)
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "Good fit" not in html
    assert "Not comparable" in html


def test_model_smile_is_the_headline_fits_heston_smile(dash, synth):
    """The chart showed the Greeks parquet (another, gate-rejected fit)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    from joint_vol_calibration.models.heston import heston_call_batch

    last = str(synth["dates"][-1].date())
    cal = _cal(last, 1.0, params={"kappa": 3.0, "theta": 0.047, "sigma": 0.5, "rho": -0.9, "v0": 0.012})
    _write_cals(synth["data"] / "calibrations", {f"joint_cal_{last}.pkl": cal})
    p = cal["params"]
    d = dash.load_page2()
    assert d["kappa"] == p["kappa"]
    assert d["mat_smiles"], "no smile rendered"
    for s in d["mat_smiles"]:
        T = s["T"] / 365.0
        k = np.array(s["lm"])
        K = np.exp(k)
        # the engine's prices (its quadrature is adaptive), inverted here
        # independently by brentq on Black-Scholes
        calls = heston_call_batch(1.0, K, T, 0.0, 0.0, p["kappa"], p["theta"], p["sigma"],
                                  p["rho"], p["v0"])
        for ki, Ki, c, iv_pct in zip(k, K, calls, s["iv"]):
            def bs(vol):
                d1 = (-np.log(Ki) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
                return norm.cdf(d1) - Ki * norm.cdf(d1 - vol * np.sqrt(T)) - c
            ref = brentq(bs, 1e-4, 5.0, xtol=1e-12)
            assert abs(iv_pct - ref * 100) < 2e-3, (s["T"], ki, iv_pct, ref * 100)


def test_quintic_is_compared_with_the_same_snapshot_heston(dash):
    q = dash.load_page2()["quintic"]
    assert q is not None
    assert q["heston"] is not None
    assert q["heston"]["file"] == f"joint_cal_{q['as_of']}.pkl"
    with open(SHIPPED / "calibrations" / q["heston"]["file"], "rb") as f:
        h = pickle.load(f)
    assert q["heston"]["spx_rmse"] == pytest.approx(dash._cal_losses(h)["spx_iv_rmse"])
    # The Δ is taken on the SAME targets: the Heston parameters re-priced on
    # the Quintic's strikes (baseline_<session>.json), not two strike sets.
    base = json.loads((SHIPPED / "calibrations" / f"baseline_{q['as_of']}.json").read_text())
    assert base["heston"]["file"] == q["heston"]["file"]
    assert base["params"] == h["params"]
    assert q["heston"]["spx_delta"] == pytest.approx(
        q["spx_rmse"] - base["on_quintic_targets"]["spx_iv_rmse"], abs=1e-12)


# ── PDV card ──────────────────────────────────────────────────────────────────

def _synth_walk_forward(synth) -> pd.Series:
    """The causal PDV forecast the signals trade on, rebuilt independently for the synthetic DB."""
    from joint_vol_calibration.models.pdv import walk_forward_vol_forecast

    with open(synth["data"] / "pdv_model.pkl", "rb") as f:
        pdv = pickle.load(f)
    con = sqlite3.connect(synth["data"] / "vol_system.db")
    close = pd.Series(dict(con.execute("SELECT date, close FROM spx_ohlcv ORDER BY date").fetchall()))
    con.close()
    close.index = pd.to_datetime(close.index)
    return walk_forward_vol_forecast(np.log(close).diff().dropna(), horizon=pdv.horizon,
                                     target=pdv.target, train_min_days=pdv.train_min_days)


def test_pdv_percentile_excludes_nan_and_ranks_like_for_like(dash, synth):
    """History = VIX − the causal (walk-forward) forecast on prior sessions, NaNs
    excluded; full-sample coefficients on past dates would be look-ahead."""
    d = dash.load_page1()
    fc = _synth_walk_forward(synth)
    spread = (synth["vix"] - 100 * fc).dropna()
    assert len(spread) > 260
    expected = float((spread.iloc[:-1] < spread.iloc[-1]).mean() * 100)
    assert d["pdv_spread_pct"] == pytest.approx(expected, abs=0.051)
    assert d["pdv_hist_n"] == len(spread) - 1
    assert d["pdv_spread_mean"] == pytest.approx(spread.iloc[:-1].mean(), abs=0.006)
    # the parquet's mostly-NaN column must play no part
    nan_based = float(((-synth["labels"]["pdv_iv_spread"] * 100) < d["pdv_spread"]).mean() * 100)
    assert abs(d["pdv_spread_pct"] - nan_based) > 1.0


def test_spread_percentile_helper_drops_nan(dash):
    hist = pd.Series([1.0, 2.0, 3.0, 4.0] + [np.nan] * 96)
    pct, n = dash.spread_percentile(hist, 3.5)
    assert (pct, n) == (75.0, 4)


def test_pdv_forecast_uses_current_features_and_its_own_date(dash, synth):
    """The forecast was the pickle's 2025-12-30 training row paired with today's VIX;
    it is now the live walk-forward forecast (= models.pdv.latest_vol_forecast)."""
    from joint_vol_calibration.models.pdv import latest_vol_forecast

    d = dash.load_page1()
    last = synth["dates"][-1]
    assert d["pdv_date"] == str(last.date())
    fc = _synth_walk_forward(synth).iloc[-1]
    with open(synth["data"] / "pdv_model.pkl", "rb") as f:
        pdv = pickle.load(f)
    con = sqlite3.connect(synth["data"] / "vol_system.db")
    close = pd.Series(dict(con.execute("SELECT date, close FROM spx_ohlcv ORDER BY date").fetchall()))
    con.close()
    close.index = pd.to_datetime(close.index)
    live = latest_vol_forecast(np.log(close).diff().dropna(), horizon=pdv.horizon)["vol"]
    assert fc == pytest.approx(live, rel=1e-9)
    assert d["pdv_forecast"] == pytest.approx(fc * 100, abs=0.006)
    assert d["pdv_spread"] == pytest.approx(synth["vix"].iloc[-1] - fc * 100, abs=0.006)
    assert d["pdv_stale"] is False


def test_stale_pdv_forecast_is_flagged_on_the_page(dash, synth):
    """Labels newer than the market data -> the forecast is stale and says so."""
    dates = _sessions(dash, "2019-01-02", "2024-07-15")
    _write_labels(synth["data"] / "signals" / "regime_labels.parquet", dates)
    d = dash.load_page1()
    assert d["pdv_stale"] is True
    assert d["pdv_age"] == len(dates) - len(synth["dates"])
    html = dash.app.test_client().get("/").get_data(as_text=True)
    assert 'id="pdv-stale-flag"' in html
    assert "STALE FORECAST" in html
    assert "Within this spread's usual range" not in html
    assert "rich versus" not in html


def test_covid_stress_forecast_is_ex_ante(dash):
    """Changing the stress-day return must not move its own forecast."""
    with open(SHIPPED / "pdv_model.pkl", "rb") as f:
        pdv = pickle.load(f)
    dates = pd.bdate_range("2017-01-02", "2020-06-30")
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, len(dates)), index=dates)
    a = dash.pdv_stress_forecast(pdv, r, "2020-03-16")
    r2 = r.copy()
    r2.loc["2020-03-16"] = -0.12
    b = dash.pdv_stress_forecast(pdv, r2, "2020-03-16")
    assert a["origin"] == "2020-03-13"
    assert a["forecast_pct"] == pytest.approx(b["forecast_pct"], abs=1e-12)
    # The realised side covers the forecast's own horizon (the shipped model is
    # the 21-session RMS-vol forecast), starting on the stress day.
    h = int(getattr(pdv, "horizon", 1))
    assert b["horizon"] == h
    window = r2.loc["2020-03-16":].iloc[:h]
    assert b["actual_pct"] == pytest.approx(
        100.0 * np.sqrt((window ** 2).mean() * 252), rel=1e-12)
    assert b["actual_pct"] != pytest.approx(a["actual_pct"])   # the stress move counts


def test_shipped_front_page_drops_the_hardcoded_stress_and_window_claims(dash):
    html = dash.app.test_client().get("/").get_data(as_text=True)
    for stale in ("92.4%", "−110pp", "2015–2025", "10-year history", "OLS + GARCH(1,1)"):
        assert stale not in html


# ── front page: rule label, classifier, term structure ───────────────────────

def test_classifier_card_uses_the_training_feature_path(dash):
    """The card fed pdv_iv_spread = −VIX (PDV read before it was computed), and
    later the PDV forecast to a classifier whose training set used the rv₂₀
    proxy. The card must rebuild the feature exactly as the training set did,
    from the source the model records."""
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import build_features

    d = dash.load_classifier_card()
    as_of = d["clf_feature_date"]
    with open(SHIPPED / "signals" / "regime_classifier.pkl", "rb") as f:
        clf = pickle.load(f)
    source = clf.metadata_["pdv_iv_spread_source"]
    assert source == "rv20_proxy"                     # what the shipped model records
    assert d["clf_feature_source"] == source
    feats = build_features(get_spx_ohlcv(as_of_date=as_of), get_vix_term_structure_wide(as_of_date=as_of))
    assert feats.attrs["pdv_iv_spread_source"] == source
    proba = clf.predict_proba(feats.dropna(subset=clf.feature_names_).iloc[[-1]])[0] * 100
    assert [d["prob_r0"], d["prob_r1"], d["prob_r2"]] == pytest.approx(list(proba), abs=0.051)


def test_classifier_card_refuses_a_source_it_cannot_rebuild(dash, synth):
    """A classifier whose training spread cannot be rebuilt is reported, not fed a substitute."""
    path = synth["data"] / "signals" / "regime_classifier.pkl"
    with open(path, "rb") as f:
        clf = pickle.load(f)
    clf.metadata_ = {**clf.metadata_, "pdv_iv_spread_source": "pdv_model_full_sample"}
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    d = dash.load_classifier_card()
    assert d["clf_available"] is False
    assert d["prob_r1"] is None
    assert any("research classifier" in e["component"] for e in d["errors"])
    assert d["regime"] is not None                 # the rule label is unaffected


def test_rule_inputs_reproduce_the_stored_label(dash):
    """The regime card shows the rule's own inputs (rv₂₀, VIX, VVIX, adaptive gate)."""
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import build_rule_label_frame

    d = dash.load_page1()
    r = d["rule"]
    frame = build_rule_label_frame(get_spx_ohlcv(as_of_date=r["date"]),
                                   get_vix_term_structure_wide(as_of_date=r["date"]))
    last = frame.iloc[-1]
    assert str(frame.index[-1].date()) == r["date"]
    assert r["rv20"] == pytest.approx(last["rv_20d"], abs=1e-12)
    assert r["gate"] == pytest.approx(last["vvix_gate"], abs=1e-12)
    assert r["label"] == int(last["regime"])
    rl = pd.read_parquet(SHIPPED / "signals" / "regime_labels.parquet")
    if pd.Timestamp(r["date"]) in rl.index:
        assert r["label"] == int(rl["regime"].loc[r["date"]])     # stored label on that session
    else:
        # the book slot ran before the close slot stored the session's label (a late
        # or failed close run): the stored labels lag the market data, never lead it
        assert rl.index.max() < pd.Timestamp(r["date"])
    html = dash.app.test_client().get("/").get_data(as_text=True)
    assert f"Rule inputs on {r['date']}" in html


def test_corrupt_classifier_keeps_the_rule_label_and_reports_an_error(dash, synth):
    """The classifier is a Research-page card: a broken pickle is that card's
    loader error (it blocks the publish) and never touches the rule label."""
    (synth["data"] / "signals" / "regime_classifier.pkl").write_bytes(b"not a pickle")
    resp = dash.app.test_client().get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    lab = int(synth["labels"]["regime"].iloc[-1])
    assert f"{['R0', 'R1', 'R2'][lab]} · {['LONG GAMMA', 'SHORT GAMMA', 'VOMMA ACTIVE'][lab]}" in html
    assert "research classifier" not in html            # not on the Overview at all
    card = dash.load_classifier_card()
    assert card["clf_available"] is False
    assert [e["component"] for e in card["errors"]] == ["research classifier (regime_classifier.pkl)"]
    assert card["regime"] == lab                          # the rule label is unaffected


def test_missing_labels_never_headline_the_classifier(dash, synth):
    (synth["data"] / "signals" / "regime_labels.parquet").unlink()
    resp = dash.app.test_client().get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "Rule label unavailable" in html
    assert dash.load_page1()["regime"] is None


def test_term_structure_is_taken_from_one_date(dash, tmp_path, monkeypatch, synth):
    """When the newest session lacks the sub-tenors the curve must not mix dates."""
    data = synth["data"]
    (data / "vol_system.db").unlink()
    _write_market_db(data / "vol_system.db", synth["dates"], drop_subtenors_on_last=True)
    d = dash.load_page1()
    assert d["vix_date"] == str(synth["dates"][-1].date())
    assert d["ts_date"] == str(synth["dates"][-2].date())
    con = sqlite3.connect(data / "vol_system.db")
    want = [r[0] for r in con.execute(
        "SELECT close FROM vix_term_structure WHERE date=? AND is_vvix=0 ORDER BY tenor_days",
        (d["ts_date"],))]
    con.close()
    assert d["ts_values"] == pytest.approx([round(v, 2) for v in want])


# ── loader failures: visible, and never deployed ──────────────────────────────

@pytest.mark.parametrize(("victim", "route", "component"), [
    ("greeks/greeks_surface.parquet", "/risk", "Greeks surface"),
    ("backtest/full_results.pkl", "/research", "backtest engine results"),
    ("pdv_model.pkl", "/", "PDV forecast"),
])
def test_missing_input_renders_an_error_not_zeros(dash, synth, route, victim, component):
    (synth["data"] / victim).unlink()
    resp = dash.app.test_client().get(route)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    # the victim is the only failing loader: the synthetic store has every other
    # input the page reads (it lacked tbill_rates, a second error hidden by ">= 1")
    assert int(resp.headers["X-Dashboard-Loader-Errors"]) == 1
    assert component in resp.headers["X-Dashboard-Error-Components"]
    assert "Data unavailable" in html
    # a zero substituted over a zero ("0 / 0", "0.00 / 0.0"), not a number pair such
    # as the inference headline's PSR / DSR "s2 0.10 / 0.004"
    assert not re.search(r"(?<![\d.])0(\.0+)? / 0(\.0+)?(?![\d.])", _visible_text(html))
    assert "NAV: $0" not in html
    # no headline tile shows a substituted zero (or a unit hanging off a missing value)
    tiles = [re.sub(r"<[^>]+>", "", v).strip()
             for v in re.findall(r'<div class="kpi-value[^"]*">(.*?)</div>', html, re.DOTALL)]
    assert not [v for v in tiles if re.fullmatch(r"[+−]?\$?0(\.0+)?\s*%?", v)], tiles
    assert "—%" not in _visible_text(html)


def test_missing_calibration_renders_an_error_not_a_500(dash, synth):
    for p in (synth["data"] / "calibrations").glob("*.pkl"):
        p.unlink()
    resp = dash.app.test_client().get("/calibration")
    assert resp.status_code == 200
    assert "headline calibration" in resp.headers["X-Dashboard-Error-Components"]


def test_any_loader_error_blocks_the_publish(dash, freezer, tmp_path, monkeypatch):
    """freeze_site only failed on HTTP 500 or a 'Traceback' string, so pages
    whose loaders had swallowed an exception were deployed."""
    real = dash.load_page3

    def broken():
        d = real()
        d.setdefault("errors", []).append({"component": "greeks", "message": "boom"})
        return d

    monkeypatch.setattr(dash, "load_page3", broken)
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 1
    assert not out.exists() or not any(out.iterdir())


def test_freeze_fails_closed_and_writes_nothing(dash, freezer, synth, tmp_path):
    (synth["data"] / "greeks" / "greeks_surface.parquet").unlink()
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 1
    assert not out.exists() or not any(out.iterdir())


def test_freeze_of_shipped_data_canonicals_and_sitemap(dash, freezer, tmp_path):
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 0
    sitemap = (out / "sitemap.xml").read_text()
    locs = re.findall(r"<loc>([^<]+)</loc>", sitemap)
    assert len(set(locs)) == len(locs) == len(dash.ROUTES)
    for route, fname in dash.ROUTES.items():
        html = (out / fname).read_text()
        canon = re.search(r'<link rel="canonical"\s+href="([^"]+)"', html).group(1)
        og = re.search(r'<meta property="og:url"\s+content="([^"]+)"', html).group(1)
        assert canon == og == dash.canonical_url(route)
        assert canon in locs
        assert 'href="/' not in html.replace('href="/"', "")   # nav links are relative
    # lastmod is the date of the data each page shows, never "today"
    p1 = dash.load_page1()
    assert f"<loc>{dash.SITE_URL}</loc><lastmod>{p1['latest_session']}</lastmod>" in sitemap
    p2 = dash.load_page2()
    assert f"calibration.html</loc><lastmod>{p2['session']}</lastmod>" in sitemap
    g = dash.load_page3()
    assert f"risk.html</loc><lastmod>{g['surface_date']}</lastmod>" in sitemap
    # the research page's data date is the last session of its equity curve
    # (it read a "run_date" key full_results.pkl never had, so it had no lastmod)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    last = str(pd.Timestamp(eq.index[-1]).date())
    assert f"research.html</loc><lastmod>{last}</lastmod>" in sitemap
    assert dash.app.test_client().get("/research").headers["X-Dashboard-Data-Date"] == last
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    last_round = max(r.get("date_end") or r["date"] for r in reg["rounds"])
    assert f"corrections.html</loc><lastmod>{last_round}</lastmod>" in sitemap
    # a page without a data date of its own gets no lastmod rather than a wrong one
    assert "model-risk.html</loc></url>" in sitemap
    has_surface = bool(dash.artifacts.surface_sessions(dash.DATA))
    assert ("surface.html</loc></url>" in sitemap) is (not has_surface)
    assert sitemap.count("<lastmod>") == 5 + has_surface


def test_retired_urls_redirect_to_their_successors(dash, freezer, tmp_path):
    """greeks.html and backtest.html are linked from outside the site; they
    must keep working after the C23 rename (static meta-refresh, noindex)."""
    assert dash.LEGACY_REDIRECTS == {"greeks.html": "risk.html", "backtest.html": "research.html"}
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 0
    for old, new in dash.LEGACY_REDIRECTS.items():
        html = (out / old).read_text()
        assert f"content='0; url={new}'" in html
        assert f"href='{dash.SITE_URL}{new}'" in html
        assert "noindex" in html
        assert (out / new).exists()
    client = dash.app.test_client()
    assert client.get("/greeks").status_code == 301
    assert client.get("/greeks").location.endswith("/risk")
    assert client.get("/backtest").location.endswith("/research")
    # the 404 page is in the desk palette, not the retired neon one
    page404 = (out / "404.html").read_text()
    assert "#111418" in page404
    assert "#8abbff" in page404
    for old in ("#0a0a0a", "#00ff88", "#00d4ff"):
        assert old not in page404.lower()


def test_no_third_party_script_and_vendored_scripts_match_their_sri(dash, freezer, pages):
    """Every script is served from the site itself with an integrity hash that
    equals the vendored file's; no page loads anything from a CDN."""
    for name, v in dash.VENDORED.items():
        vendored = ROOT / "dashboard" / v["src"]
        assert vendored.exists(), name
        assert freezer.sri_sha384(str(vendored)) == v["sri"], name
        assert v["version"] in v["src"]
        assert (ROOT / "dashboard" / "static" / "vendor" / "licenses" / v["license_file"]).exists(), name
    ec = dash.VENDORED["echarts"]
    for route, html in pages.items():
        assert not re.findall(r"<script[^>]+src=\"(?:https?:)?//", html), route
        # the chart library is loaded, with its hash, exactly on the pages that draw a chart
        draws = 'data-spec="' in html or 'data-load-3d="' in html
        tag = f'<script src="{ec["src"]}" integrity="{ec["sri"]}" crossorigin="anonymous"></script>'
        assert (tag in html) is draws, route
        assert html.count(ec["src"]) == (1 if draws else 0), route
        assert "cdn.jsdelivr" not in html
        assert "unpkg.com" not in html
        assert "chart.umd" not in html
        assert not freezer.page_policy_problems(html), route
    # echarts-gl is never a page script: the Surface page loads it on demand, with its hash
    gl = dash.VENDORED["echarts-gl"]
    for route, html in pages.items():
        assert f'<script src="{gl["src"]}"' not in html, route
    surf = pages["/surface"]
    if 'data-load-3d="surface3d"' in surf:
        assert f'data-src="{gl["src"]}" data-sri="{gl["sri"]}"' in surf
    js = (ROOT / "dashboard" / "static" / "desk.js").read_text()
    assert "s.integrity = " in js
    assert "s.crossOrigin = 'anonymous'" in js


def test_freeze_copies_static_and_refuses_a_script_that_fails_its_sri(dash, freezer, tmp_path, monkeypatch):
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 0
    for v in dash.VENDORED.values():
        assert (out / v["src"]).read_bytes() == (ROOT / "dashboard" / v["src"]).read_bytes()
    for name in ("echarts", "echarts-gl"):
        monkeypatch.setitem(dash.VENDORED[name], "sri", "sha384-" + "A" * 64)
        out2 = tmp_path / f"site-{name}"
        assert freezer.freeze(str(out2), dash) == 1
        assert not out2.exists() or not any(out2.iterdir())
        monkeypatch.undo()


# ── Greeks page ───────────────────────────────────────────────────────────────

def test_heatmap_draws_every_node(dash):
    """0.05-wide bins on a 0.039-spaced grid hid 16 of 69 nodes (one unstable)."""
    d = dash.load_page3()
    gdf = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    cells = [c for row in d["heatmap_rows"] for c in row["cells"] if not c.get("empty")]
    assert len(cells) == d["n_cells"] == len(gdf)
    # the surface's own flag (vomma P&L ratio rule), not a re-derived pooled z-score
    assert sum(c["unstable"] for c in cells) == d["n_unstable"] == int(gdf["is_unstable"].sum())
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    assert len(re.findall(r'class="node unstable flag"', html)) == d["n_unstable"]
    assert "ln(K/S)" not in html
    assert "2026-03-24" not in html
    assert "All three unstable" not in html


def test_greeks_badge_and_staleness_come_from_the_surface(dash):
    d = dash.load_page3()
    gdf = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    assert d["surface_date"] == str(gdf["as_of_date"].iloc[0])[:10]
    assert d["greeks_age"] == dash.session_age(d["surface_date"], dash._latest_market_session())
    assert d["qv_values"]
    assert len(d["qv_values"]) == d["n_maturities"]


@pytest.mark.parametrize("same_fit", [True, False])
def test_greeks_provenance_metadata_is_read_and_checked(dash, synth, same_fit):
    """Contract for the surface builder: parquet metadata key b"provenance"."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    last = str(synth["dates"][-1].date())
    with open(synth["data"] / "calibrations" / f"joint_cal_{last}.pkl", "rb") as f:
        params = dict(pickle.load(f)["params"])
    if not same_fit:
        params["kappa"] += 0.5
    path = synth["data"] / "greeks" / "greeks_surface.parquet"
    table = pa.Table.from_pandas(pd.read_parquet(path))
    prov = {"source_calibration": f"joint_cal_{last}.pkl", "spot_date": last,
            "S": 6000.0, "r": 0.04, "q": 0.013, "params": params}
    meta = dict(table.schema.metadata or {})
    meta[b"provenance"] = json.dumps(prov).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)

    d = dash.load_page3()
    assert d["provenance"]["source_calibration"] == f"joint_cal_{last}.pkl"
    assert d["source_matches_headline"] is same_fit
    assert d["surface_date"] == last
    assert d["greeks_age"] == 0
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    assert ("not the parameters of the calibration-page headline fit" in html) is (not same_fit)


# ── backtest page: one source for every metric ───────────────────────────────

@pytest.fixture(scope="module")
def engine_state():
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        return pickle.load(f)


def test_kpis_are_round_trips_of_the_portfolio_sleeves(dash, engine_state):
    """283 counted entries AND exits; the per-signal table shows 88+22+31 = 141."""
    m = engine_state["metrics"]
    d = dash.load_page4()
    rows = {r["label"]: r["value"] for r in d["metrics_rows"]}
    n = sum(m[f"{s}_n_trades"] for s in dash.PORTFOLIO_SLEEVES)
    trades = [t for t in engine_state["all_trades"] if t.signal in dash.PORTFOLIO_SLEEVES]
    closed = [t for t in trades if not t.is_open]
    assert d["kpi"]["n_trades"] == n == len(trades)
    assert rows["Trades"] == f"{n:,}" + (f" ({n - len(closed)} open)" if len(closed) < n else "")
    assert n == sum(int(s["n_trades"].split()[0]) for s in d["signal_metrics"] if not s["is_research"])
    assert d["kpi"]["win_rate"] == pytest.approx(np.mean([t.net_pnl > 0 for t in closed]), abs=1e-12)
    assert d["kpi"]["avg_pnl"] == pytest.approx(sum(t.net_pnl for t in closed) / 3 / len(closed), abs=1e-9)
    assert rows["Sharpe"] == f"{m['sharpe']:.3f}".replace("-", "\u2212")


def test_s3_sharpe_is_the_engine_value_not_hidden(dash, engine_state):
    s3 = next(s for s in dash.load_page4()["signal_metrics"] if s["label"].startswith("S3"))
    assert s3["sharpe"] == f"{engine_state['metrics']['s3_sharpe']:.3f}"
    html = dash.app.test_client().get("/backtest").get_data(as_text=True)
    assert "N/A*" not in html
    assert "ratio diverges" not in html


def test_half_year_sharpes_use_engine_metrics_unclipped(dash):
    from joint_vol_calibration.backtest.backtest_engine import compute_metrics

    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    nav = eq["nav"]
    nav.index = pd.to_datetime(nav.index)
    wf = dash.half_year_sharpes(nav)
    for w in wf:
        sub = nav.loc[w["start"]:w["end"]]
        prior = nav.loc[:pd.Timestamp(w["start"]) - pd.Timedelta(days=1)]
        win = pd.concat([prior.iloc[-1:], sub])
        if w["flat"]:
            assert w["sharpe"] is None
        else:
            ref = compute_metrics(pd.DataFrame({"nav": win}), rf=0.0)["sharpe"]
            assert w["sharpe"] == pytest.approx(ref, abs=1e-12)
    html = dash.app.test_client().get("/backtest").get_data(as_text=True)
    assert "OOS Windows" not in html
    assert "Walk-Forward Sharpe" not in html

    # no clipping: a steady riser has a Sharpe far above the old ±4 cap
    idx = pd.bdate_range("2021-01-04", "2021-12-31")
    steady = pd.Series(1e6 * np.exp(np.cumsum(np.random.default_rng(2).normal(4e-4, 1e-4, len(idx)))),
                       index=idx)
    hy = dash.half_year_sharpes(steady)
    assert len(hy) == 2
    assert min(w["sharpe"] for w in hy) > 20


def test_annual_returns_compound_to_the_total(dash):
    d = dash.load_page4()
    total = np.prod([1 + a["ret_pct"] / 100 for a in d["ann"]]) - 1
    assert total == pytest.approx(d["kpi"]["cum_ret"], abs=1e-10)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    assert d["eq_dates"][-1] == str(pd.Timestamp(eq.index[-1]).date())


def test_backtest_prose_is_rendered_from_the_results(dash, engine_state):
    html = dash.app.test_client().get("/research").get_data(as_text=True)
    for stale in ("+$463K, 76% win", "S1C collapses to −$404K", "−$189K", "−$333K COVID",
                  "73% of profit", "R2-Exit P&amp;L"):
        assert stale not in html
    m = engine_state["metrics"]
    for key in ("s1_total_pnl", "s1c_total_pnl", "s4_total_pnl"):
        assert dash._usd_k(m[key]) in html


# ── C22 step 2: structure, accessibility and copy ─────────────────────────────

@pytest.fixture(scope="module")
def pages(dash):
    """Rendered HTML of every route from the shipped data."""
    client = dash.app.test_client()
    return {route: client.get(route).get_data(as_text=True) for route in dash.ROUTES}


def _attrs(tag: str) -> dict:
    return dict(re.findall(r'([\w:-]+)="([^"]*)"', tag))


def test_every_page_has_one_h1_landmarks_and_a_skip_link(pages):
    """The pages had zero h1–h6, no landmarks around the banner and no skip link."""
    for route, html in pages.items():
        assert len(re.findall(r"<h1[\s>]", html)) == 1, route
        assert len(re.findall(r"<h2[\s>]", html)) >= 3, route
        assert html.count("<main") == 1
        assert 'id="main"' in html
        assert re.search(r'<nav class="nav" aria-label="Primary">', html)
        assert html.count("<header") == 1
        assert '<a class="skip-link" href="#main">' in html
        # the nav is a direct child of <body> (sticky works for the whole page),
        # and nothing overrides its position inline
        assert '<nav class="nav" style=' not in html
        assert "position:relative;\">" not in html


def test_nav_marks_exactly_the_current_page(dash, pages):
    for route, html in pages.items():
        current = re.findall(r'<a href="([^"]+)" class="nav-link" data-key="\w" aria-current="page">', html)
        assert current == [route], (route, current)
        assert len(re.findall(r'<a [^>]*aria-current="page"', html)) == 1
    # every page is in the nav, with a unique g-shortcut key
    keys = [k for _, _, k, _ in dash.NAV]
    assert [r for r, *_ in dash.NAV] == list(dash.ROUTES)
    assert len(set(keys)) == len(keys)


def test_every_chart_has_an_accessible_name_and_a_data_alternative(pages):
    """Canvases had role=img and a generic label (the PDV sparkline was called an
    'SPX price sparkline'); none had a text alternative carrying the numbers.
    Every chart is now a role=img container named by its title, described by a
    sentence and by a server-rendered table of its data, with a JSON spec."""
    n = 0
    for route, html in pages.items():
        for tag in re.findall(r'<div class="chart[^"]*" id="[^"]+" data-spec="[^"]+"[^>]*>', html):
            a = _attrs(tag)
            cid = a["id"]
            assert a.get("role") == "img", (route, tag)
            assert len(a.get("aria-label", "")) > 20, (route, tag)
            assert a.get("aria-describedby") == f"{cid}-desc {cid}-data", (route, tag)
            desc = re.search(rf'<p class="sr-only" id="{re.escape(cid)}-desc">([^<]+)</p>', html)
            assert desc and len(desc.group(1).strip()) > 20, (route, cid)
            m = re.search(rf'<div class="chart-data sr-only" id="{re.escape(cid)}-data">(.*?)</div>', html,
                          re.DOTALL)
            assert m and "<table>" in m.group(1) and "<caption>" in m.group(1), (route, cid)
            assert re.search(r"<td[^>]*>[^<]*\d", m.group(1)), (route, cid)        # it carries numbers
            spec = json.loads(_html_unescape(re.search(
                rf'<script type="application/json" id="{re.escape(a["data-spec"])}">(.*?)</script>', html,
                re.DOTALL).group(1)))
            assert spec["id"] == cid and spec["series"], (route, cid)
            # the table toggle and the CSV button name this chart, or (a small
            # multiple, P4.4) its group's one CSV lists it beside the group's table
            groups = [g.split() for g in re.findall(r'data-csv-group="([^"]+)"', html)]
            if any(cid in g for g in groups):
                assert 'data-table-for="smile-group-data"' in html
            else:
                assert f'data-table-for="{cid}-data"' in html
                assert f'data-csv-for="{cid}"' in html
            n += 1
        assert "SPX price sparkline" not in html
        for svg in re.findall(r'<svg class="spark[^"]*"[^>]*>', html):
            assert 'role="img"' in svg and len(_attrs(svg).get("aria-label", "")) > 20, (route, svg)
    assert n >= 20


def _html_unescape(s: str) -> str:
    import html as _h

    return _h.unescape(s)


def test_scroll_regions_are_keyboard_focusable(pages):
    """Tables and heatmaps scroll horizontally on phones; axe flagged the
    scroll regions as unreachable by keyboard. Every visible table sits in a
    focusable, named scroll region (chart-data tables are the charts' text
    alternatives and are toggled, not scrolled in place)."""
    n = 0
    for route, html in pages.items():
        for m in re.finditer(r'<table class="(tbl[^"]*|heat)"', html):
            before = html[:m.start()]
            opener = before[before.rfind("<div"):]
            assert re.match(r'<div class="(tbl-wrap|heat-wrap)[^"]*" tabindex="0" role="region" '
                            r'aria-(label|labelledby)="[^"]+">\s*$', opener), (route, opener[:160])
            n += 1
    assert n >= 15


DESK_CSS = ROOT / "dashboard" / "static" / "desk.css"
DESK_JS = ROOT / "dashboard" / "static" / "desk.js"


def _tokens() -> dict:
    root = re.search(r":root\s*\{(.*?)\n\}", DESK_CSS.read_text(), re.DOTALL).group(1)
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\b", root))


def test_colour_tokens_meet_wcag_aa(dash, pages):
    """--text3 was #555 (2.5:1 on the cards) and chart ticks/axis titles #555/#444.
    Text tokens need 4.5:1 on every surface they sit on (WCAG 1.4.3); series,
    fills and the focus ring need 3:1 against the panels (1.4.11)."""
    tok = _tokens()
    backgrounds = [tok[b] for b in ("bg", "bg-deep", "surface", "surface2", "tooltip",
                                    "tint-ok", "tint-watch", "tint-breach", "tint-info")]
    for name in ("text", "text2", "text3", "pos", "neg", "ok", "watch", "breach", "info"):
        for bg in backgrounds:
            assert dash.contrast_ratio(tok[name], bg) >= 4.5, (name, tok[name], bg)
    for name in [f"cat-{i}" for i in range(6)] + ["series-headline", "series-muted", "pos-fill", "neg-fill",
                                                  "tail-down", "tail-up", "focus"]:
        for bg in (tok["bg"], tok["surface"], tok["surface2"]):
            assert dash.contrast_ratio(tok[name], bg) >= 3.0, (name, tok[name], bg)
    # a pressed toggle / selected tab is marked by a 3:1 border, not by colour of text alone
    assert re.search(r'\.btn\[aria-pressed="true"\][^{]*\{[^}]*border-color: var\(--pos-fill\)', DESK_CSS.read_text())
    # charts take their colours from these tokens (read at run time), not from literals
    js = DESK_JS.read_text()
    assert "getComputedStyle(root).getPropertyValue(name)" in js
    screen = js[:js.index("if (print)")]
    assert not re.findall(r"'#[0-9a-fA-F]{3,6}'", screen)
    # the print theme is dark-on-white and also meets 4.5:1
    pr = dict(re.findall(r"t\.(text3|text2|text) = '(#[0-9a-fA-F]{6})'", js))
    assert set(pr) == {"text", "text2", "text3"}
    for c in pr.values():
        assert dash.contrast_ratio(c, "#ffffff") >= 4.5
    for route, h in pages.items():
        for literal in ("#555", "#444", "#00ff88"):
            assert f"color:{literal}" not in h and f"color: {literal}" not in h, (route, literal)


def test_categorical_series_are_the_okabe_ito_set_and_never_neon(dash):
    tok = _tokens()
    assert [tok[f"cat-{i}"] for i in range(6)] == ["#56b4e9", "#e69f00", "#009e73", "#f0e442", "#cc79a7",
                                                   "#d55e00"]
    css = DESK_CSS.read_text().lower()
    for neon in ("#00ff88", "#00d4ff", "#ff3366", "#0a0a0a"):
        assert neon not in css
    # gains and losses are blue / orange (colour-vision safe), each with a sign glyph
    assert tok["pos"] == "#8abbff"
    assert tok["neg"] == "#fbb360"


def test_heatmap_text_contrasts_with_every_cell(dash):
    """White text on the bright-green cells was 2.7:1; the z labels sat at 0.7 opacity."""
    d = dash.load_page3()
    cells = [c for row in d["heatmap_rows"] for c in row["cells"] if not c.get("empty")]
    assert cells
    for c in cells:
        assert dash.contrast_ratio(c["fg"], c["bg"]) >= 4.5, c
    for z in np.linspace(-3, 3, 241):
        bg = dash._vomma_color(float(z))
        assert dash.contrast_ratio(dash._text_colour_on(bg), bg) >= 4.5
    # every ramp a heatmap cell can take: cividis (IV), berlin (residuals), P&L
    for i in range(101):
        for bg in (dash.seq_colour(i, 0, 100), dash.div_colour(i / 50 - 1, 1.0),
                   dash.pnl_colour(i / 50 - 1, 1.0)):
            assert dash.contrast_ratio(dash._text_colour_on(bg), bg) >= 4.5, bg
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    assert "opacity:0.7" not in html
    # unstable is flagged by outline + text marker, not by a fill alone
    n_flagged = len(re.findall(r'class="node unstable flag"', html))
    assert n_flagged == len(re.findall(r'<div class="vv">⚠ ', html)) == d["n_unstable"]
    assert '<span class="sym">(∂²V/∂S∂σ)</span>' in html       # σ is not upper-cased to Σ
    # cell details are reachable by keyboard and screen reader, not only a title tooltip
    assert ' title="' not in "".join(re.findall(r'<td class="node[^>]*>', html))
    assert all("data-detail=" in td for td in re.findall(r'<td class="node[^>]*>', html))
    assert 'aria-live="polite"' in html
    assert "Hover or focus a cell (arrow keys move)" in html


def test_motion_print_and_font_fallbacks(pages):
    css, js = DESK_CSS.read_text(), DESK_JS.read_text()
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "animation: false" in js
    assert "@media print" in css
    assert "beforeprint" in js
    assert "afterprint" in js
    assert re.search(r"--mono:[^;]*monospace;", css)
    assert re.search(r"--sans:[^;]*sans-serif;", css)
    for route, h in pages.items():
        assert '<meta name="color-scheme" content="dark">' in h
        assert "family: 'Menlo'" not in h and "font-family:'Menlo',monospace" not in h, route


def test_fonts_are_self_hosted_subsets_that_match_their_manifest(pages):
    """Inter was declared in the CSS but never loaded (system fallback everywhere).
    The fonts are now OFL subsets served from the site, preloaded, and each file
    matches the manifest build_fonts.py wrote (sha256 and the source it came from)."""
    fdir = ROOT / "dashboard" / "static" / "fonts"
    manifest = json.loads((fdir / "fonts.json").read_text())
    css = DESK_CSS.read_text()
    files = sorted(p.name for p in fdir.glob("*.woff2"))
    assert files == sorted(manifest["files"])
    for name, meta in manifest["files"].items():
        data = (fdir / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == meta["sha256"], name
        assert data[:4] == b"wOF2" and len(data) < 60_000, name
        assert any(f"url({q}fonts/{name}{q})" in css for q in ("", "'", '"')), name
    for lic in ("OFL-Inter.txt", "OFL-IBM-Plex-Mono.txt"):
        assert "SIL OPEN FONT LICENSE" in (fdir / lic).read_text().upper()
    # the IBM Plex derivative is renamed (Reserved Font Name), so it is "Desk Mono"
    assert '"Desk Mono"' in css
    assert "font-family: \"IBM Plex Mono\"" not in css
    for h in pages.values():
        assert '<link rel="preload" href="static/fonts/InterDesk-4.1-wght400-700.woff2" as="font" type="font/woff2" crossorigin>' in h
        assert "fonts.googleapis" not in h
        assert "fonts.gstatic" not in h


def test_charts_fail_soft_without_echarts(pages):
    """A failed Chart.js load threw 'Chart is not defined' and blanked every chart.
    Without ECharts (blocked, or JavaScript off) every chart's table is shown."""
    js = DESK_JS.read_text()
    assert "if (!window.echarts) {" in js
    assert "root.classList.add('charts-failed')" in js
    assert re.search(r"\.charts-failed \.chart-data\.sr-only[^{]*\{", DESK_CSS.read_text())
    for html in pages.values():
        assert "new Chart(" not in html
        assert "Chart.defaults" not in html
        # ECharts loads before the page script, which only reads JSON specs
        if 'data-spec="' in html:
            assert html.index("echarts-6.1.0-desk.min.js") < html.index("static/desk.js?v=")
    assert "if (!$$('.chart').length) return;" in js
    assert "echarts-6.1.0-desk.min.js" not in pages["/model-risk"]        # no chart, no library


def test_social_meta_and_card_carry_no_test_count(pages):
    """The card said '631 tests', the meta '666', CI '642 passed, 24 skipped'."""
    og = _load("og_card_under_test", "dashboard/make_og_image.py")
    assert not any(re.search(r"\btests?\b", label) for _, label in og.STATS)
    for route, html in pages.items():
        assert not re.search(r"\d+\s+tests", html), route
        title = re.search(r'<meta property="og:title"\s+content="([^"]+)"', html).group(1)
        desc = re.search(r'<meta property="og:description"\s+content="([^"]+)"', html).group(1)
        assert title
        assert desc
    titles = {re.search(r'og:title"\s+content="([^"]+)"', h).group(1) for h in pages.values()}
    assert len(titles) == len(pages)                      # per-page, not one for all


def test_titles_do_not_promise_a_vix_smile_fit(pages):
    """CLM-05: 'Joint SPX/VIX smile calibration' means fitting both option
    smiles (Guyon 2020); every committed fit has the VIX-options leg off
    (w3 = 0) and fits the four VIX indices. One honest title everywhere: the
    README, CITATION.cff, every page's og:title and the Overview heading, and
    the social card; the card's correction range is the registry's."""
    readme_h1 = (ROOT / "README.md").read_text().splitlines()[0]
    cff_title = re.search(r'^title: "([^"]+)"', (ROOT / "CITATION.cff").read_text(), re.MULTILINE).group(1)
    og = _load("og_card_title_under_test", "dashboard/make_og_image.py")
    texts = {"README": readme_h1, "CITATION": cff_title, "card": " ".join(og.TITLE_LINES)}
    for route, html in pages.items():
        texts[route] = re.search(r'<meta property="og:title"\s+content="([^"]+)"', html).group(1)
        texts[route + " alt"] = re.search(r'<meta property="og:image:alt"\s+content="([^"]+)"', html).group(1)
    texts["overview h1"] = re.search(r'<h1 class="page-title">([^<]+)</h1>', pages["/"]).group(1)
    for where, text in texts.items():
        assert "smile calibration" not in text.lower(), (where, text)
    assert readme_h1 == f"# {cff_title}"
    assert cff_title == "Joint SPX/VIX Calibration System"
    assert "VIX index term structure" in og.SUBTITLE
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    span = f"{reg['rounds'][0]['id']}–{reg['rounds'][-1]['id']}"
    assert (span, "correction rounds logged") in og.STATS


def test_regime_and_calibration_copy_state_the_actual_rules(pages):
    idx, cal, bt = pages["/"], pages["/calibration"], pages["/research"]
    assert "rolling 252-day 80th percentile" in idx
    assert "rv₂₀ &gt; VIX" in idx
    for stale in ("Low VIX / backwardation", "Normal contango", "VVIX &gt; 100"):
        assert stale not in idx
        assert stale not in bt
    for stale in ("non-affine model (e.g. rough Heston)", "degenerate on daily data",
                  "Barndorff-Nielsen &amp; Shephard 2004", "production classification"):
        assert stale not in cal
    # The ρ-boundary banner (rough-Heston / Bates copy) renders only when the
    # headline fit has ρ on its bound; the shipped headline does not.
    if "AT THE LOWER BOUNDARY" in cal:
        assert "affine Volterra" in cal
        assert "did not converge" in cal
    # the pre-C22 claim that Heston was fitted to an SVI/SSVI-smoothed surface
    # (never wired in). P2.2's skew panel names the market's SVI-cal surface,
    # so the bare string 'SVI' is no longer the test: the stale claims are.
    for stale in ("structural failure", "Feller-satisfying Heston", "SOFR",
                  "VIX Futures RMSE", "SVI smoothing", "SSVI-smoothed", "SSVI surface", "SVI / SSVI",
                  "SVI/SSVI"):
        assert stale not in cal, stale
    # the pre-C22 copy's "VIX options RMSE was 37.14 (vol) pts"; the quote smiles'
    # chart data may hold an IV of 37.14 %
    assert not re.search(r"37\.14 (vol )?pts", cal)


def _raw_specs(html: str) -> dict:
    return {m.group(1): json.loads(_html_unescape(m.group(2))) for m in re.finditer(
        r'<script type="application/json" id="([\w-]+)-spec">(.*?)</script>', html, re.DOTALL)}


def _spec(html: str, chart_id: str) -> dict:
    """A chart's spec as desk.js draws it: compact x values expanded, shared x values
    and derived series filled in (desk_charts.resolve, the twin of desk.js resolveSpec)."""
    raw = _raw_specs(html)
    assert chart_id in raw, chart_id
    return _charts().resolve(raw[chart_id], raw)


def _charts():
    import sys
    if str(ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(ROOT / "dashboard"))
    import desk_charts
    return desk_charts


def test_term_structure_is_drawn_on_a_linear_tenor_axis(dash, pages):
    """Tenors were evenly spaced categories (9d, 30d, 93d, 184d) on a smoothed line."""
    from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS

    d = dash.load_page1()
    cid = re.search(r'<div class="chart[^"]*" id="([^"]+)" data-spec="[^"]+" role="img" '
                    r'aria-label="VIX index term structure', pages["/"]).group(1)
    spec = _spec(pages["/"], cid)
    assert spec["x"]["type"] == "value"
    assert spec["x"].get("min") == 0
    line = spec["series"][0]
    assert line["type"] == "line"
    assert not line.get("smooth")
    assert [p[0] for p in line["data"]] == [VIX_INDEX_TENOR_DAYS[t]
                                            for t in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")]
    assert [p[1] for p in line["data"]] == pytest.approx(d["ts_values"])


def test_backtest_audit_heading_counts_its_rows(pages):
    """'Ten Flaws Found, Eight Fixed, Two Quantified' sat above an 11-row table.
    The Corrections page heading is counted from the postmortems it shows."""
    cp = pages["/corrections"]
    arts = re.findall(r'<article class="pm[^"]*" id="c\d+" data-round="(C\d+)".*?</article>', cp, re.DOTALL)
    bodies = re.findall(r'<article class="pm[^"]*" id="c\d+" data-round="C\d+".*?</article>', cp, re.DOTALL)
    statuses = [s for b in bodies for s in re.findall(r'data-status="(fixed|quantified)"', b)]
    n_fixed, n_quant = statuses.count("fixed"), statuses.count("quantified")
    heading = re.search(r'id="audit-title">([^<]+)<', cp).group(1)
    assert heading == (f"{arts[0]}–{arts[-1]} corrections — {len(statuses)} findings, "
                       f"{n_fixed} fixed, {n_quant} quantified")
    assert arts == [f"C{i}" for i in range(16, 24)]
    assert "Ten Flaws Found" not in cp
    assert "C17 pipeline" not in cp


def test_backtest_window_crisis_and_data_claims_come_from_the_data(dash, pages):
    bt = pages["/research"]
    d = dash.load_page4()
    assert re.search(rf"<h1 class=\"page-title\">Research · delta-hedged straddle backtest, "
                     rf"<span class=\"nowrap\">{d['start_date']} → {d['end_date']}</span>", bt)
    assert "2018 → 2025<" not in bt
    tariffs = next(c for c in d["crisis"] if c["name"] == "Tariffs_2025")
    assert tariffs["end"] <= d["end_date"]
    assert f"Tariffs 2025 ({tariffs['start']} → {tariffs['end']})" in bt
    assert "April 2025 tariff shock is <em>not</em> in it" in bt
    assert "single options snapshot" not in bt
    assert "22 times in 7 years" not in bt
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        n_dates = con.execute("SELECT COUNT(DISTINCT snapshot_date) FROM options_snapshots "
                              "WHERE underlying='SPX'").fetchone()[0]
    assert f"the database's {n_dates} SPX chain" in bt
    assert "$-" not in bt
    assert "'$' + v + 'k'" not in bt


def test_greeks_provenance_written_by_risk_monitor_is_read(dash, synth):
    """The dashboard read b"provenance" while risk_monitor.save_greeks_surface
    writes b"greeks_provenance" (nested source): a regenerated surface with
    full provenance was shown as 'records no source calibration'."""
    from joint_vol_calibration.greeks.risk_monitor import save_greeks_surface

    last = str(synth["dates"][-1].date())
    with open(synth["data"] / "calibrations" / f"joint_cal_{last}.pkl", "rb") as f:
        params = dict(pickle.load(f)["params"])
    path = synth["data"] / "greeks" / "greeks_surface.parquet"
    df = pd.read_parquet(path)
    df.attrs["provenance"] = {
        "schema_version": 2, "as_of_date": last, "S": 6000.0, "r": 0.04, "q": 0.007,
        "params": params,
        "source": {"calibration_file": f"data_store/calibrations/joint_cal_{last}.pkl",
                   "spot_date": last},
    }
    save_greeks_surface(df, path)

    d = dash.load_page3()
    assert d["provenance"]["source_calibration"] == f"joint_cal_{last}.pkl"
    assert d["source_matches_headline"] is True
    assert d["surface_date"] == last
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    assert f"joint_cal_{last}.pkl" in html
    assert "records no source calibration" not in html


def test_classifier_accuracy_is_read_from_the_shipped_model(pages):
    """The front page hard-coded '63.4%' vs '90.0%' (pre-C22 fixed-gate labels with a
    look-ahead PDV feature) beside a classifier retrained in C22 that records its own
    evaluation. The figures must come from the pickle's metadata."""
    import pickle

    with open(ROOT / "data_store" / "signals" / "regime_classifier.pkl", "rb") as f:
        ev = pickle.load(f).metadata_["evaluation"]
    idx = pages["/research"]
    oos = f"{round(ev['oos_acc'] * 100, 1)}%"
    pers = f"{round(ev['persistence_acc'] * 100, 1)}%"
    # visible text: 'out-of-sample' is the page's first use of a glossary term, a link (Q14)
    assert re.search(rf"{re.escape(oos)} out-of-sample ?, it loses to the {re.escape(pers)}",
                     _visible_text(idx))
    assert re.search(r"<dt>Out-of-sample accuracy:</dt><dd><span[^>]*>" + re.escape(oos), idx)
    assert re.search(r"<dt>Persistence baseline:</dt><dd><span[^>]*>" + re.escape(pers), idx)


def test_backtest_audit_table_has_the_c22_round(pages):
    assert 'data-round="C22"' in pages["/corrections"]
    for route in ("/research", "/corrections"):
        assert "from the C21 run" not in pages[route]


# ── C22 verification round: presentation guarantees ──────────────────────────

def _visible_text(html: str) -> str:
    import html as _html

    body = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", html, flags=re.DOTALL)
    body = body[body.index("<body"):]
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)))


def _assert_number_style(route: str, html: str) -> None:
    text = _visible_text(html)
    hyphen_numbers = re.findall(r"(?<![\w/.\-–])-\$?\.?\d[\d,.]*", text)
    assert not hyphen_numbers, (route, hyphen_numbers[:5])
    for bad in (r"\bNaN\b", r"\bnan%", r"\bundefined\b", r"\bInfinity\b",
                r"[:=]\s*(None|null)\b", r"(^|\s)None(\s|$)"):
        assert not re.search(bad, text), (route, bad)


def test_numbers_use_true_minus_signs_and_no_placeholders(pages):
    """'-0.571', '$-923' and 'z=-1.0' printed hyphens; a missing value must read '—',
    never 'nan', 'None' or 'undefined'."""
    for route, html in pages.items():
        _assert_number_style(route, html)


def test_pdv_skill_is_the_library_walk_forward_r2(dash, pages):
    """The card hard-coded 'Walk-fwd R² 0.31 / kernel 0.23' (the superseded next-day
    model) beside the 21-day model; the figure is now recomputed from the DB."""
    from joint_vol_calibration.models.pdv import spx_log_returns_from_db, walk_forward_skill_report

    d = dash.load_page1()
    s = d["pdv_skill"]
    assert s["horizon"] == 21
    r = spx_log_returns_from_db(as_of_date=d["spx_date"])
    rep = walk_forward_skill_report(r, horizons=(21,))["summary"].iloc[0]
    assert s["r2"] == pytest.approx(rep["r2"], abs=5e-5)          # the report rounds to 4 dp
    assert s["r2_ex"] == pytest.approx(rep["r2_ex"], abs=5e-5)
    html = pages["/"]
    assert f"Walk-forward R²: <span class=\"v\">{s['r2']:.2f}" in html
    assert "Walk-fwd R² (kernel)" not in html
    assert ">0.31<" not in html


def test_stress_card_describes_the_models_own_target(dash, pages):
    """The 21-day model's outcome is the RMS vol over its horizon, not 'that day's |r|'."""
    st = dash.load_page1()["stress"]
    text = _visible_text(pages["/"])
    assert st["horizon"] == 21
    assert st["n_window"] == 21
    assert f"Realised over its 21-session horizon ({st['stress_date']} → {st['window_end']}" in text
    assert "that day's |r|" not in text


def test_audit_evidence_is_read_from_the_c22_bridge(dash, pages):
    """'S1 +$722K → −$508K from β alone' quoted the published C21 figure as the
    pre-β value; the β step alone moved S1 from the previous bridge row."""
    bridge = pd.read_csv(SHIPPED / "backtest" / "c22_bridge.csv").set_index("step")
    grid = pd.read_csv(SHIPPED / "backtest" / "beta_sensitivity.csv").set_index("beta")
    rows = list(bridge.index)
    prev = bridge.loc[rows[rows.index("beta") - 1]]
    text = _visible_text(pages["/corrections"])
    # the bridge's pre-β S1 is at β = 1 but before the later C22 steps; the grid's
    # β = 1 row is the full C22 configuration, and the evidence says which is which
    want = (f"C22 bridge row {int(bridge.loc['beta', 'row'])}: S1 {dash._usd_k(prev['s1_pnl'])} → "
            f"{dash._usd_k(bridge.loc['beta', 's1_pnl'])} from β alone, before the later C22 steps; "
            f"at β = 1 under the full C22 configuration (β grid), S1 shows "
            f"{dash._usd_k(grid.loc[1.0, 's1_pnl'])}")
    assert want in text
    assert "S1 would show" not in text
    assert f"C22 bridge row {int(bridge.loc['r2_gate', 'row'])}: portfolio {dash._usd_k(bridge.loc['r2_gate', 'd_pnl'])}" in text
    assert "+$722K → −$508K" not in text
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        beta = pickle.load(f)["atm_to_vix_ratio"]
    assert f"β = {beta:.2f}" in text


def test_a_bridge_from_another_run_is_a_loader_error(dash, synth):
    path = synth["data"] / "backtest" / "c22_bridge.csv"
    df = pd.read_csv(path)
    df.loc[df.index[-1], "pnl"] += 1000.0
    df.to_csv(path, index=False)
    d = dash.load_page4()
    assert any("C22 bridge" in e["component"] for e in d["errors"])
    assert d["bridge"] is None


def test_change_log_quotes_facts_from_the_db(dash, pages):
    """The log said '2018: ~1.5%' (^IRX averaged 1.95%) and R2 '~53% → ~25%'."""
    d = dash.load_page4()
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        t = pd.read_sql("SELECT date, rate FROM tbill_rates WHERE date BETWEEN ? AND ?", con,
                        params=(d["start_date"], d["end_date"]), parse_dates=["date"])
    m2018 = t[t["date"].dt.year == 2018]["rate"].mean()
    assert d["tbill_by_year"][2018] == pytest.approx(m2018, abs=1e-12)
    text = _visible_text(pages["/research"])
    assert f"2018: {m2018 * 100:.2f}%" in text
    sh = d["r2_share"]
    assert f"R2 on {sh['fixed'] * 100:.1f}% of backtest sessions" in text
    assert f"R2 on {sh['adaptive'] * 100:.1f}% of backtest sessions" in text
    assert 0.40 < sh["fixed"] < 0.55
    assert 0.18 < sh["adaptive"] < 0.28
    for stale in ("~1.5%", "~53%", "~25%", "18 new tests"):
        assert stale not in text


def test_no_js_fallback_and_scroll_affordance(pages):
    for route, html in pages.items():
        nos = re.search(r"<noscript><style>(.*?)</style></noscript>", html, re.DOTALL).group(1)
        assert ".chart {" in nos and ".chart-data.sr-only" in nos and "position: static" in nos, route
        assert ".js-only" in nos
    css = DESK_CSS.read_text()
    rule = css[css.index(".tbl-wrap, .heat-wrap {"):]
    rule = rule[:rule.index("}")]
    assert "local" in rule
    assert "scroll" in rule
    assert "position: relative" in rule
    assert "overflow-x: auto" in rule
    assert "Scroll sideways for more columns" in DESK_JS.read_text()


def test_unverified_or_contradicted_copy_is_gone(pages):
    cal, bt = _visible_text(pages["/calibration"]), _visible_text(pages["/research"])
    assert "misclassifies 2025-04-09" not in cal          # no stored HMM backs it
    assert "10 vp target" not in bt                      # the Quintic VIX-options RMSE is below it
    assert "Good fit" not in cal


def test_pdv_features_use_returns_rebuilt_from_closes(dash, synth):
    """The stored log_return column has holes (2025-05-08 in the shipped DB);
    dropping that row removed a session from the PDV state variables."""
    from joint_vol_calibration.models.pdv import extract_pdv_features

    hole = str(synth["dates"][-30].date())
    con = sqlite3.connect(synth["data"] / "vol_system.db")
    con.execute("UPDATE spx_ohlcv SET log_return = NULL WHERE date = ?", (hole,))
    con.commit()
    close = pd.Series(dict(con.execute("SELECT date, close FROM spx_ohlcv ORDER BY date").fetchall()))
    con.close()
    close.index = pd.to_datetime(close.index)
    feats = extract_pdv_features(np.log(close).diff().dropna())
    d = dash.load_page1()
    assert d["pdv_sigma1"] == pytest.approx(feats["sigma1"].iloc[-1] * 100, abs=0.006)
    assert d["pdv_forecast"] == pytest.approx(_synth_walk_forward(synth).iloc[-1] * 100, abs=0.006)


def test_scenario_panel_renders_the_shipped_grid(dash, pages):
    """The Risk page's scenario panel shows the producer's grid
    (greeks/scenario_grid.py): the sticky-strike cells, every one, with the
    book, date and units from the file's provenance. Read from the file, not
    typed: the daily bot regenerates it from each new headline fit."""
    from joint_vol_calibration.greeks.scenario_grid import load_scenario_grid
    path = SHIPPED / "greeks" / "scenario_grid.parquet"
    if not path.exists():
        pytest.skip("scenario_grid.parquet not generated")
    g = load_scenario_grid(path)
    prov = g.attrs["provenance"]
    html = pages["/risk"]
    table = re.search(r'<table class="heat" id="scenario-grid".*?</table>', html, re.DOTALL).group(0)
    ss = g[g["dynamics"] == "sticky_strike"]
    assert table.count("<td ") == len(ss)
    for row in ss.itertuples():
        assert f"P&amp;L {dash.signed(row.pnl, ',.0f')}" in table or \
            f"P&L {dash.signed(row.pnl, ',.0f')}" in table
    text = _visible_text(html)
    assert prov["book"] in text
    assert f"as of {prov['as_of_date']}" in text
    assert "sticky-strike dynamics shown (the file also holds sticky-moneyness)" in text
    assert prov["units"] in text
    assert "Artifact not yet generated" not in text.split("Scenario grid")[1].split("QV convexity")[0]
    # the sticky-moneyness grid, every cell, behind its own disclosure on the same colour scale
    sm = g[g["dynamics"] == "sticky_moneyness"]
    t2 = re.search(r'<table class="heat" id="scenario-grid-sticky-moneyness".*?</table>', html,
                   re.DOTALL).group(0)
    assert t2.count("<td ") == len(sm) > 0
    for row in sm.itertuples():
        assert f"P&L {dash.signed(row.pnl, ',.0f')}" in t2.replace("&amp;", "&")


def test_unstable_nodes_follow_the_surfaces_recorded_rule(dash, pages):
    """load_page3 re-derived the flag as |z| > 2 (5 of 79 nodes when this was
    found) instead of the surface's own vomma-P&L-ratio rule; the expected count
    is read from the stored surface, not typed here."""
    import pyarrow.parquet as pq

    path = SHIPPED / "greeks" / "greeks_surface.parquet"
    gdf = pd.read_parquet(path)
    rule = json.loads(pq.read_schema(path).metadata[b"greeks_provenance"])["unstable_rule"]
    d = dash.load_page3()
    assert d["unstable_rule"]["kind"] == "pnl_ratio"
    assert d["unstable_rule"]["threshold"] == rule["threshold"]
    assert (d["n_unstable"] == int(gdf["is_unstable"].sum())
            == int((gdf["vomma_pnl_ratio"].abs() > rule["threshold"]).sum()))
    text = _visible_text(pages["/risk"])
    assert f"{d['n_unstable']} of {d['n_cells']}" in text
    assert f"vomma P&L ratio ½·vomma·|Δσ|/vega > {rule['threshold']:.2f}" in text
    assert "|vomma z-score| > 2" not in text
    assert "pooled score" not in text
    # every dropped grid node says why
    empties = [c for row in d["heatmap_rows"] for c in row["cells"] if c.get("empty")]
    assert len(empties) == d["n_missing"]
    assert all(c["reason"] for c in empties)


def test_unstable_falls_back_to_z_only_without_the_ratio_and_says_so(dash, synth):
    path = synth["data"] / "greeks" / "greeks_surface.parquet"
    gdf = pd.read_parquet(path).drop(columns=["vomma_pnl_ratio"])
    gdf.to_parquet(path)
    d = dash.load_page3()
    assert d["unstable_rule"]["kind"] == "zscore"
    assert d["n_unstable"] == int((gdf["vomma_zscore"].abs() > 2).sum())
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    assert "fallback: the surface records no vomma P&amp;L ratio" in html


def test_model_smile_never_carries_a_nan(dash, monkeypatch):
    """implied_vol_from_price returns NaN (not None) for ill-posed inputs."""
    from joint_vol_calibration.models import heston

    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.7, "v0": 0.04}
    # a deep out-of-the-money node with a near-zero price is omitted, never NaN
    deep = dash.heston_model_smile(params, maturities_days=(14,), log_moneyness=(-1.5, -0.8, 0.0, 0.1))
    assert all(np.isfinite(v) for s in deep for v in s["iv"])
    assert 0.0 in deep[0]["lm"]
    assert -1.5 not in deep[0]["lm"]
    real = heston.implied_vol_from_price
    monkeypatch.setattr(heston, "implied_vol_from_price",
                        lambda price, *a, **k: float("nan") if a[1] > 1.05 else real(price, *a, **k))
    out = dash.heston_model_smile(params, maturities_days=(30,), log_moneyness=(-0.1, 0.0, 0.1))
    assert out[0]["lm"] == [-0.1, 0.0]
    assert all(np.isfinite(v) for v in out[0]["iv"])


def _without_inference(base: dict) -> dict:
    """A Research page dict as it is before the C23 inference report has run."""
    return {**base, "null_rows": [], "inference": {"state": "absent", "paths": [], "producer": None},
            "sharpe_ci": None}


@pytest.mark.parametrize(("p", "says"), [
    (0.0746, "does not beat random entry dates at the 5% level (p = 0.075;"),
    (0.0348, "beats random entry dates at the 5% level (p = 0.035;")])
def test_s3_timing_claim_follows_the_stored_null_test(dash, monkeypatch, p, says):
    """'Only S3's timing survives' / 'its timing beats a random-walk null' were
    hard-coded; the regenerated null test gives p = 0.075, not significant.
    Without the C23 inference report the legacy 200-path null is quoted, and
    labelled as superseded."""
    base = _without_inference(dash.load_page4())
    monkeypatch.setattr(dash, "load_page4", lambda: {**base, "kpi": {**base["kpi"], "s3_null_p": p}})
    text = _visible_text(dash.app.test_client().get("/research").get_data(as_text=True))
    assert says in text
    assert "the legacy 200-path null, superseded in C23" in text
    assert ("not significant at 5%" in text) is (p >= 0.05)
    assert "Timing Survives" not in text


@pytest.mark.parametrize("route", ["/", "/surface", "/calibration", "/risk", "/research", "/corrections",
                                   "/model-risk"])
def test_page_header_dates_never_break_across_lines(dash, route):
    """A date in the page header must sit inside a no-wrap element: at phone
    width the browser otherwise breaks '2026-10-16' at a hyphen."""
    html = dash.app.test_client().get(route).get_data(as_text=True)
    badge = re.search(r'<div class="page-badge"[^>]*>(.*?)</div>', html, re.DOTALL)
    assert badge, route
    body = badge.group(1)
    # Strip the no-wrap spans; no ISO date may remain outside one.
    outside = re.sub(r'<span class="nowrap"[^>]*>.*?</span>', "", body, flags=re.DOTALL)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", outside), (route, outside)


@pytest.mark.parametrize("route", ["/", "/surface", "/calibration", "/risk", "/research", "/corrections",
                                   "/model-risk"])
def test_table_dates_never_break_across_lines(pages, route):
    """C23 verify: at 390 px the Calibration page's gate-passing fits and rough
    Heston tables broke '2026-10-16' into '2026-' and '10-16'. A table cell that
    opens with a date keeps it on one line: the cell is no-wrap, or the date sits
    in a no-wrap span when a chip follows it. Chart data tables are screen-reader
    only and never laid out."""
    assert _breakable_table_dates(pages[route]) == [], route


def test_table_dates_never_break_with_every_artifact_present(dash, art):
    """The same with every C23 artifact in the store: the forward ledger's
    position cell ('2026-11-19 7,760 · short') appears only once a ledger is
    written, and the simulated next session caught its date wrapping."""
    for route in ("/", "/surface", "/calibration", "/risk", "/research", "/corrections", "/model-risk"):
        _resp, html = _get(dash, route)
        assert _breakable_table_dates(html) == [], route


def _breakable_table_dates(html: str) -> list:
    html = re.sub(r'<div class="chart-data sr-only".*?\n</div>', "", html, flags=re.DOTALL)
    bad = []
    for table in re.findall(r'<table class="tbl[^"]*"[^>]*>.*?</table>', html, flags=re.DOTALL):
        for attrs, body in re.findall(r"<td([^>]*)>(.*?)</td>", table, flags=re.DOTALL):
            text = re.sub(r"<[^>]+>", "", body).strip()
            if (re.match(r"\d{4}-\d{2}-\d{2}", text) and "nowrap" not in attrs
                    and not re.match(r'\s*<span class="nowrap">\d{4}-\d{2}-\d{2}', body)):
                bad.append(text[:40])
    return bad


def test_artifact_paths_break_between_names_never_inside_one(dash, pages):
    """C23 verify: the footer's freshness table and the Model Risk artifact and
    module columns broke a path anywhere ('data_store/vol_syste' over 'm.db',
    'joint_vol_calibration/m' over 'odels/heston.py'): word-break: break-all and
    overflow-wrap: anywhere. A path now breaks only after '/' or '_', where the
    path filter puts a <wbr>; its names stay whole (a CamelCase test class may
    also break between its words: test_path_filter_breaks_between_camel_case_words)."""
    assert str(dash.path_breaks("data_store/vol_system.db")) == "data_<wbr>store/<wbr>vol_<wbr>system.db"
    assert str(dash.path_breaks("data_store/surface/<session>/")) == \
        "data_<wbr>store/<wbr>surface/<wbr>&lt;session&gt;/"
    assert str(dash.path_breaks("calibrations/joint_cal_2026-10-16.pkl")) == \
        'calibrations/<wbr>joint_<wbr>cal_<wbr><span class="nowrap">2026-10-16.pkl</span>'
    assert str(dash.path_breaks("test_pdv.py::TestNoLookAhead::test_x")) == \
        "test_<wbr>pdv.py::<wbr>Test<wbr>No<wbr>Look<wbr>Ahead::<wbr>test_<wbr>x"
    assert str(dash.path_breaks(None)) == ""

    def paths(fragment):
        return re.findall(r"<code([^>]*)>(.*?)</code>", fragment, flags=re.DOTALL)

    def whole(body):
        import html as _html
        return _html.unescape(re.sub(r"<[^>]+>", "", body))

    for route, html in pages.items():
        foot = re.search(r'<footer class="site-footer".*?</footer>', html, flags=re.DOTALL)
        assert foot, route
        cells = [p for p in paths(foot.group(0)) if whole(p[1]).startswith("data_store/")]
        assert cells, route
        for attrs, body in cells:
            assert 'class="path"' in attrs, (route, body)
            assert str(dash.path_breaks(whole(body))) == body, (route, body)
    mr = pages["/model-risk"]
    for table_id in ("monitor-table", "inventory-table"):
        table = re.search(rf'<table[^>]*id="{table_id}".*?</table>', mr, flags=re.DOTALL)
        assert table, table_id
        cells = paths(table.group(0))
        assert cells, table_id
        for attrs, body in cells:
            assert 'class="path"' in attrs, (table_id, body)
            assert str(dash.path_breaks(whole(body))) == body, (table_id, body)
    css = (ROOT / "dashboard" / "static" / "desk.css").read_text()
    assert "break-all" not in css
    assert ".tbl code.path { overflow-wrap: break-word; }" in css


# ── Independent review of 8bf9aa6: every displayed number equals its artifact ─

def test_stress_windows_are_measured_from_the_prior_close(dash, pages):
    """compute_crisis_performance divided by the window's FIRST NAV, dropping the
    first session's return: 'FedHikes 2022 (2022-01-01 → 2022-12-31) +5.1%' sat
    beside a 2022 annual return of +4.71% for the same sessions."""
    d = dash.load_page4()
    crisis = {c["name"]: c for c in d["crisis"]}
    ann = {a["year"]: a["ret_pct"] / 100 for a in d["ann"]}
    assert crisis["FedHikes_2022"]["total_return"] == pytest.approx(ann[2022], abs=1e-12)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    nav = eq["nav"]
    nav.index = pd.to_datetime(nav.index)
    for c in crisis.values():
        base = nav[nav.index < pd.Timestamp(c["start"])].iloc[-1]
        assert c["total_return"] == pytest.approx(nav.loc[c["start"]:c["end"]].iloc[-1] / base - 1, abs=1e-12)
    # C23 record (Cboe VIX family); C22: +4.71%, +1.69%, -1.13%
    assert crisis["FedHikes_2022"]["total_return"] == pytest.approx(0.0413374, abs=1e-6)
    assert crisis["COVID_2020"]["total_return"] == pytest.approx(-0.0018125, abs=1e-6)
    assert crisis["Tariffs_2025"]["total_return"] == pytest.approx(-0.0114006, abs=1e-6)
    text = _visible_text(pages["/research"])
    assert "FedHikes 2022 (2022-01-01 → 2022-12-31) +4.1%" in text
    assert "COVID 2020 (2020-02-19 → 2020-04-30) −0.2%" in text
    assert "+5.1%" not in text
    assert "+1.8%" not in text
    # a jump on the window's first session counts; a window at the curve's start uses its first NAV
    idx = pd.bdate_range("2021-12-27", "2022-01-07")
    toy = pd.Series([100.0] * 5 + [110.0] * 5, index=idx)
    got = dash.window_returns(toy, {"jump": ("2022-01-01", "2022-01-07"),
                                    "head": ("2021-12-01", "2021-12-31")})
    assert got[0]["total_return"] == pytest.approx(0.10)
    assert got[0]["n_days"] == 5
    assert got[1]["total_return"] == pytest.approx(0.0)
    assert got[1]["n_days"] == 4


LIVE_BETA_CAPTURE = ROOT / "joint_vol_calibration" / "tests" / "fixtures" / "spx_chain_live_20261012.json"


def test_beta_copy_names_its_chains_and_the_later_check(dash, pages):
    """'β = 0.82, estimated from five real 2026 SPX chains (0.80–0.84)': the five
    were 4 DB chains and a live capture, the range was typed in, and the later
    2026-10-16 DB chain (0.70; mean over every chain 0.80) went unmentioned."""
    from joint_vol_calibration.backtest.backtest_engine import BETA_ESTIMATION_SESSIONS
    from joint_vol_calibration.backtest.run_backtest import estimate_beta

    t = estimate_beta(include_live=False)
    inc = t[t["status"].str.startswith("included")]
    oos = t[t["status"].str.startswith("out-of-sample")].sort_values("session")
    assert sorted(inc["session"]) == sorted(BETA_ESTIMATION_SESSIONS)
    est = list(inc["ratio"]) + [json.loads(LIVE_BETA_CAPTURE.read_text())["ratio_full_chain"]]
    every = est + list(oos["ratio"])
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        beta = pickle.load(f)["atm_to_vix_ratio"]
    assert round(float(np.mean(est)), 2) == beta
    first = oos.iloc[0]
    # The estimation set and the first later chain are frozen (BETA_ESTIMATION_SESSIONS), so
    # their wording is pinned. The all-chain mean grows with every chain the daily bot stores,
    # so it is derived, and the page must only call it a grid row when it really is one.
    stem = (f"β = {beta:.2f} = mean of {len(inc)} DB chains and 1 live capture "
            f"({min(est):.2f}–{max(est):.2f}); the first later chain, {first['session']}, gives "
            f"{first['ratio']:.2f}")
    assert stem == ("β = 0.82 = mean of 4 DB chains and 1 live capture (0.80–0.84); the first later "
                    "chain, 2026-10-16, gives 0.70")
    grid_betas = pd.read_csv(SHIPPED / "backtest" / "beta_sensitivity.csv")["beta"]
    in_grid = any(abs(g - round(float(np.mean(every)), 2)) < 1e-9 for g in grid_betas)
    want = (f"{stem} — the mean over all {len(every)} chains, {np.mean(every):.2f}, "
            + ("is a row of the β grid" if in_grid else "is not in the β grid"))
    text = _visible_text(pages["/corrections"])
    assert want in text
    for route in ("/research", "/corrections"):
        assert "five real 2026 SPX chains" not in _visible_text(pages[route])
    # every chain is listed with its ratio and use
    d = dash.load_page4()
    db_chains = [(c["session"], round(c["ratio"], 3)) for c in d["beta_est"]["chains"] if c["kind"] == "DB chain"]
    assert db_chains == [(s, round(r, 3)) for s, r in zip(t["session"], t["ratio"])]
    # the grid on the page is the CSV, including the row at the mean over every chain
    grid = pd.read_csv(SHIPPED / "backtest" / "beta_sensitivity.csv")
    html = pages["/research"]
    assert len(re.findall(r'<tr data-beta="', html)) == len(grid)
    # Every row of the grid is the CSV row, and the ◆ (mean over every chain) marks a row only
    # when that mean is one: each chain the bot stores moves it, often off the grid.
    for _, row in grid.iterrows():
        cells = re.search(rf'<tr data-beta="{row["beta"]:.2f}"[^>]*>(.*?)</tr>', html, re.DOTALL).group(1)
        for col in ("pnl", "s1c_pnl", "s3_pnl", "s4_pnl", "s1_pnl"):
            assert dash._usd_k(row[col]) in cells, (row["beta"], col)
        assert ("◆" in cells) == (in_grid and abs(row["beta"] - round(float(np.mean(every)), 2)) < 1e-9)


def test_a_beta_that_disagrees_with_its_chains_is_a_loader_error(dash, synth):
    path = synth["data"] / "backtest" / "full_results.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["atm_to_vix_ratio"] = 0.90
    with open(path, "wb") as f:
        pickle.dump(state, f)
    d = dash.load_page4()
    assert any(e["component"].startswith("β estimation chains") for e in d["errors"])
    assert d["beta_est"] is None


def test_trade_kpis_keep_the_open_trade_out_of_the_win_rate(dash, engine_state, pages):
    """'Round trips 208, win rate 38.0%, average −$1,039' counted the S4 straddle
    opened on the last session (is_open, net = its −$433 entry cost) as a closed,
    losing round trip."""
    trades = [t for t in engine_state["all_trades"] if t.signal in dash.PORTFOLIO_SLEEVES]
    open_ = [t for t in trades if t.is_open]
    closed = [t for t in trades if not t.is_open]
    assert [(t.signal, str(t.entry_date.date())) for t in open_] == [("s4", "2025-03-24")]
    assert (len(trades), len(closed)) == (206, 205)       # C23 record; C22: 208 / 207
    win = np.mean([t.net_pnl > 0 for t in closed])
    avg = sum(t.net_pnl for t in closed) / 3 / len(closed)
    assert (round(win * 100, 1), round(avg)) == (37.6, -1083)     # C22: 38.2%, -$1,043
    text = _visible_text(pages["/research"])
    assert (f"Trades {len(trades)} 1 open at the end; over the {len(closed)} closed: "
            f"win rate {win * 100:.1f}%, "
            f"average {dash._usd(avg)} of portfolio P&L per trade") in text
    assert "Round trips" not in text
    assert "38.0%" not in text
    assert "−$1,039" not in text
    assert "S4 VRP −$273,733" in text                     # C22: −$273,544
    assert "46 (1 open)" in text
    assert "S4's count includes a straddle opened on the last session (2025-03-24)" in text


def test_displayed_numbers_are_rounded_once(dash, pages):
    """Values rounded in the loader and again in the template printed S3's final
    +5.647% as '+5.7%', the 365-day smile's 12.4249 as '12.43', and Greeks vomma
    3,267.496 as '3,268' (via 3,267.5)."""
    m = dash.MINUS
    # equity-curve table: final / lowest / highest from the unrounded curves
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    table = re.search(r'id="chart-equity-data">(.*?)</div>', pages["/research"], re.DOTALL).group(1)
    for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C contrarian"),
                      ("nav_s3", "S3 short SPX straddle"), ("nav_s4", "S4 VRP"), ("nav_s1", "S1 PDV (ref)")):
        rel = eq[col] / eq[col].iloc[0] - 1
        want = "".join(f'<td class="num">{dash._pct_prose(x)}</td>'
                       for x in (rel.iloc[-1], rel.min(), rel.max()))
        assert f"<tr><td>{name}</td>{want}</tr>" in table, name
    # the C22 case (+5.647% printed as +5.7%) is now +6.105% on the C23 record
    assert '<tr><td>S3 short SPX straddle</td><td class="num">+6.1%</td>' in table
    # model smile table: the Heston smile of the headline fit, formatted once
    _, cal = dash.load_headline_calibration()
    smile = {s["T"]: s for s in dash.heston_model_smile(cal["params"])}
    cal_html = pages["/calibration"]
    s365 = smile[365]
    assert f"{s365['lm'][-1]:+.3f}: {s365['iv'][-1] * 100:.2f}" in cal_html
    # Rounded once: the loader hands the template the unrounded model smile for every
    # maturity, so each printed cell is a single rounding of the model value. (Checked on
    # the live headline fit rather than one day's number: the bot replaces that fit.)
    d2 = dash.load_page2()
    for row in d2["mat_smiles"]:
        model = smile[row["T"]]
        assert list(row["lm"]) == list(model["lm"])
        assert [x / 100.0 for x in row["iv"]] == pytest.approx(list(model["iv"]), rel=0, abs=1e-15)
        for i in (0, -1):
            assert f"{row['lm'][i]:+.3f}: {row['iv'][i]:.2f}".replace("-", m) in cal_html
    # Greeks heatmap: every cell's vomma and flag metric from the stored surface
    gdf = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    gdf = gdf.assign(lm_key=gdf["log_moneyness"].round(4)).sort_values(["T_days", "lm_key"])
    g_html = pages["/risk"]
    # (P4.5: the flag metric is the cell's first line, vomma the second)
    vv = re.findall(r'<div class="z">([^<]+)</div>', g_html)
    assert vv == [f"{x:,.0f}".replace("-", m) for x in gdf["vomma"]]
    zz = [re.sub(r"^⚠ ", "", z).strip() for z in re.findall(r'<div class="vv">([^<]+)</div>', g_html)]
    assert zz == [f"{abs(x):.2f}" for x in gdf["vomma_pnl_ratio"]]


def test_a_vomma_just_below_a_half_is_rounded_once(dash, synth):
    """The C22 case: vomma 3,267.496 printed '3,268' (rounded to 3,267.5 in the
    loader, then again in the template). Checked on a written surface, not on the
    shipped one: the daily bot regenerates that from each new headline fit."""
    from joint_vol_calibration.greeks.risk_monitor import load_greeks_surface, save_greeks_surface
    path = synth["data"] / "greeks" / "greeks_surface.parquet"
    gdf = load_greeks_surface(path)
    gdf.loc[gdf.index[0], "vomma"] = 3267.496
    save_greeks_surface(gdf, path)
    html = dash.app.test_client().get("/risk").get_data(as_text=True)
    vv = re.findall(r'<div class="z">([^<]+)</div>', html)           # vomma: the second line
    assert "3,267" in vv
    assert "3,268" not in vv


def test_vix_index_tenors_come_from_config(dash, pages):
    """The front page hard-coded ^VIX6M at 182 days ('9d 20.12 → 182d 21.67' and
    the chart's x position); the system's tenor map is 184 days."""
    from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS

    d = dash.load_page1()
    assert d["ts_tenors"] == [f"{VIX_INDEX_TENOR_DAYS[t]}d" for t in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")]
    assert d["ts_tenors"][-1] == "184d"
    text = _visible_text(pages["/"])
    assert f"9d {d['ts_values'][0]:.2f} → 184d {d['ts_values'][-1]:.2f}" in text
    assert "182d" not in pages["/"]


@pytest.mark.parametrize(("last_vix", "big", "tail", "lead"), [
    (95.0, "Record high", "Implied vol is rich versus this spread's own history (highest on record)",
     "above all"),
    (0.5, "Record low", "Implied vol is cheap versus this spread's own history (lowest on record)",
     "at or below all"),
])
def test_record_pdv_spreads_read_as_records(dash, synth, last_vix, big, tail, lead):
    """A spread above (below) its whole history rendered 'top 0%' and '100th
    percentile' ('bottom 0%', '0th'). Synthetic history: today's VIX is pushed
    far above (below) anything before it."""
    con = sqlite3.connect(synth["data"] / "vol_system.db")
    con.execute("UPDATE vix_daily SET close = ? WHERE date = ?", (last_vix, str(synth["dates"][-1].date())))
    con.commit()
    con.close()
    d = dash.load_page1()
    assert d["pdv_spread_pct"] == (100.0 if last_vix > 50 else 0.0)
    text = _visible_text(dash.app.test_client().get("/").get_data(as_text=True))
    assert tail in text
    assert f"{big} {lead} {d['pdv_hist_n']:,} prior sessions" in text
    for bad in (r"\btop 0%", r"\bbottom 0%", r"\b100th\b", r"\b0th\b"):
        assert not re.search(bad, text), bad


def test_percentile_wording_is_capped_short_of_a_record(dash):
    assert dash.percentile_rank(99.6) == {"record": None, "big": "99th", "tail": "top 1%"}
    assert dash.percentile_rank(0.3) == {"record": None, "big": "1st", "tail": "bottom 1%"}
    assert dash.percentile_rank(85.2) == {"record": None, "big": "85th", "tail": "top 15%"}
    assert dash.percentile_rank(50.0)["tail"] is None
    assert dash.percentile_rank(None) is None


def test_pre_c22_spread_statistics_are_computed_from_the_data(dash, pages, monkeypatch):
    """'σ̂ averaged 0.71× ... held on 95% of sessions' was typed text (the code
    comments say ~96% and ≈0.66×). Both figures are now the engine's own pre-C22
    replay (pdv_horizon = 1) on the engine's data window."""
    from joint_vol_calibration.backtest.backtest_engine import BacktestEngine
    from joint_vol_calibration.backtest.run_backtest import _load_data
    from joint_vol_calibration.models.pdv import forward_target

    d = dash.load_page4()
    pc = d["pre_c22_spread"]
    spx, vix_wide = _load_data()                       # the engine's loader, independent of app.py
    sig = BacktestEngine(start_date=d["start_date"], end_date=d["end_date"],
                         pdv_horizon=1)._build_walkforward_pdv_vol(spx)
    vix = vix_wide.set_index(pd.to_datetime(vix_wide["date"]))["^VIX"] / 100
    w = pd.concat([sig.rename("f"), vix.rename("v")], axis=1, join="inner")
    w = w.loc[d["start_date"]:d["end_date"]].dropna()
    assert pc["share"] == pytest.approx(((w["v"] - w["f"]) > 0.02).mean(), abs=1e-12)
    assert pc["n"] == len(w) == len(pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet"))
    close = spx.set_index(pd.to_datetime(spx["date"]))["close"]
    y = forward_target(np.log(close).diff().dropna(), horizon=21, target="rv")
    both = pd.concat([sig.rename("f"), y.rename("y")], axis=1, join="inner")
    both = both.loc[d["start_date"]:d["end_date"]].dropna()
    assert pc["ratio"] == pytest.approx(both["f"].mean() / both["y"].mean(), abs=1e-12)
    text = _visible_text(pages["/corrections"])
    assert (f"averaged {pc['ratio']:.2f}× the realised RMS vol of the next 21 sessions in the backtest window, "
            f"so “VIX > σ̂ + 2 vol pts” held on {pc['share'] * 100:.0f}% of its {pc['n']:,} sessions") in text
    # the page follows the computation, not a typed figure
    monkeypatch.setattr(dash, "pre_c22_spread_facts", lambda *a, **k: {**pc, "share": 0.5, "ratio": 0.9})
    text = _visible_text(dash.app.test_client().get("/corrections").get_data(as_text=True))
    assert "averaged 0.90×" in text
    assert "held on 50% of its" in text


def test_s3_null_test_is_labelled_as_the_base_size_test(dash, engine_state, pages, synth):
    """The p = 0.075 sat next to S3's +$56,471, but s3_timing_null_test sizes every
    path, the real one included, at S3's base (pass-1) kelly: its real statistic is
    +$41,952, before the sizing overlay. The page says so, and shows the base-size
    figure only when the stored results record it."""
    m = engine_state["metrics"]
    text = _visible_text(pages["/corrections"])
    p = f"{m['s3_null_p_value']:.3f}"
    # the C22 finding quotes its own (legacy) test, named as superseded
    assert f"timing null test on S3 at base (pass-1) size, before the sizing overlay: p = {p}" in text
    assert "(the C22 200-path null; C23 replaced it with exposure-matched nulls" in text
    # the Research page quotes it only when the C23 nulls are missing
    base = _without_inference(dash.load_page4())
    import unittest.mock as _mock
    with _mock.patch.object(dash, "load_page4", lambda: base):
        legacy = _visible_text(dash.app.test_client().get("/research").get_data(as_text=True))
    text += " | " + legacy
    assert ("its timing, tested on S3 at base (pass-1) size, before the sizing overlay, does not beat "
            f"random entry dates at the 5% level (p = {p};") in text
    assert "S3 at base (pass-1) size, before the sizing overlay, vs random entry dates: not significant at 5%" in text
    assert "null p =" not in text
    assert "S3 entries vs random entry dates" not in text
    if "s3_null_real_pnl" not in m:
        assert "+$41,952" not in text                    # never typed in
    # when the results record the base-size statistic, it is quoted
    path = synth["data"] / "backtest" / "full_results.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["metrics"]["s3_null_real_pnl"] = 41951.87
    with open(path, "wb") as f:
        pickle.dump(state, f)
    client = dash.app.test_client()
    text = _visible_text(client.get("/corrections").get_data(as_text=True))
    assert "on S3 at base (pass-1) size (+$41,952), before the sizing overlay" in text
    base = _without_inference(dash.load_page4())
    import unittest.mock as _mock
    with _mock.patch.object(dash, "load_page4", lambda: base):
        text = _visible_text(client.get("/research").get_data(as_text=True))
    assert "on S3 at base (pass-1) size (+$41,952), before the sizing overlay" in text


def test_s3_trade_count_is_explained_as_entries_plus_rolls(dash, engine_state, pages):
    """S3's 39 trade records are 26 entries and 13 expiry rolls; 'S3 fires 39 times'
    overstated its breadth."""
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    pos = eq["s3_position"]
    entries = int(((pos != 0) & (pos.shift(1, fill_value=0) == 0)).sum())
    s3 = [t for t in engine_state["all_trades"] if t.signal == "s3"]
    exp = {t.exit_date for t in s3 if t.exit_reason == "expiry"}
    rolls = sum(1 for t in s3 if t.entry_date in exp)
    assert (len(s3), entries, rolls) == (39, 26, 13)
    text = _visible_text(pages["/research"])
    assert (f"{len(s3)} trades in 2018–2025 ({entries} entries plus {rolls} rolls of a position held "
            f"past its straddle's expiry)") in text
    assert f"S3 enters {entries} times in 2018–2025" in text
    assert "S3 fires 39 times" not in text


def test_no_double_negative_losses(pages):
    """'it loses −$274K' / 'It loses −$431K' read as gains."""
    for route, html in pages.items():
        assert not re.search(r"\blos(?:es|t|e)\s+[−-]\$", _visible_text(html)), route


def test_page_header_can_wrap_at_every_separator():
    """Header items are no-wrap spans joined by '·' separators with no spaces;
    without a break opportunity at the separator the whole line is one
    unbreakable run (497 px wide at a 390 px viewport). The separator must be
    an inline-block, which browsers may break around."""
    rule = re.search(r"\.page-badge \.sep\s*\{([^}]*)\}", DESK_CSS.read_text())
    assert rule
    assert "display: inline-block" in rule.group(1)


# ── C23 artifacts: the producers' outputs, absent / ok / malformed ────────────
#
# tests/fixtures/dashboard/ holds trimmed samples of what each producer wrote
# (dashboard/capture_artifact_fixtures.py): the surface report for 2026-10-16
# (3 of 17 expiries, every 4th quote, all models; option prices dropped), the
# inference summary / benchmarks / forecast evaluation, three forward-ledger
# files written by backtest/forward_ledger.py from its own test chains, and the
# calibration diagnostics of 2026-10-16 plus the summary of every diagnosed
# session (trimmed real files). Each test builds a data_store that links the shipped
# directories and adds the artifacts, so nothing under data_store/ changes.
# The store's calibrations stop at the fixtures' session: the daily bot adds
# newer fits (and the headline moves to them), while these samples describe
# 2026-10-16, so a newer shipped fit must not become the store's headline.

FIX = ROOT / "joint_vol_calibration" / "tests" / "fixtures" / "dashboard"
SESSION = "2026-10-16"


def _artifact_store(tmp_path: Path, surface=True, inference=True, forward=True, diagnostics=True) -> Path:
    data = tmp_path / "data_store"
    data.mkdir()
    for child in SHIPPED.iterdir():
        if child.name in ("backtest", "calibrations", "surface", "forward"):
            continue
        (data / child.name).symlink_to(child)
    for sub in ("backtest", "calibrations"):
        (data / sub).mkdir()
        for child in (SHIPPED / sub).iterdir():
            if child.name in ("inference", "diagnostics"):
                continue
            m = re.fullmatch(r"joint_cal_(\d{4}-\d{2}-\d{2})\.pkl", child.name)
            if sub == "calibrations" and m and m.group(1) > SESSION:
                continue                      # a fit the bot added after the fixtures' session
            (data / sub / child.name).symlink_to(child)
    if surface:
        shutil.copytree(FIX / "surface", data / "surface")
    if inference:
        shutil.copytree(FIX / "inference", data / "backtest" / "inference")
    if forward:
        shutil.copytree(FIX / "forward", data / "forward")
    if diagnostics:
        shutil.copytree(FIX / "diagnostics", data / "calibrations" / "diagnostics")
    return data


@pytest.fixture
def art(dash, tmp_path, monkeypatch):
    data = _artifact_store(tmp_path)
    monkeypatch.setattr(dash, "DATA", data)
    return data


def _get(dash, route):
    r = dash.app.test_client().get(route)
    assert r.status_code == 200, route
    return r, r.get_data(as_text=True)


def _errors(resp) -> int:
    return int(resp.headers.get("X-Dashboard-Loader-Errors", "0"))


@pytest.fixture
def no_art(dash, tmp_path, monkeypatch):
    """The shipped data_store without any C23 producer's artifact (data_store/
    now ships the surface, inference and diagnostics outputs)."""
    data = _artifact_store(tmp_path, surface=False, inference=False, forward=False, diagnostics=False)
    monkeypatch.setattr(dash, "DATA", data)
    return data


def test_artifacts_absent_render_named_placeholders_and_still_publish(dash, no_art):
    """Before a producer runs, its panels say 'not yet generated', name the file
    and the producer, and show no number in its place; the pages still publish."""
    client = dash.app.test_client()
    pages = {route: client.get(route).get_data(as_text=True) for route in ("/", "/surface", "/research")}
    surf = pages["/surface"]
    assert 'data-artifact-state="absent"' in surf
    assert "data_store/surface/&lt;session&gt;/residuals.parquet" in surf
    assert "scripts/surface_report.py" in surf
    res = pages["/research"].replace("<wbr>", "")
    for path in ("data_store/backtest/inference/summary.json",
                 "data_store/forward/ledger_&lt;session&gt;.json",
                 "data_store/backtest/inference/forecast_eval.json"):
        assert path in res, path
    assert "the bootstrap interval needs the inference artifact" in pages["/"]
    for route in dash.ROUTES:
        assert _errors(client.get(route)) == 0, route


def test_surface_page_renders_the_producers_numbers(dash, art):
    resp, html = _get(dash, "/surface")
    assert _errors(resp) == 0
    sc = json.loads((FIX / "surface" / SESSION / "scorecard.json").read_text())
    text = _visible_text(html)
    table = re.search(r'<table class="tbl dense[^"]*" id="scorecard-table">(.*?)</table>', html,
                      re.DOTALL).group(1)
    rank = [r["model"] for r in sc["ranking_by_oos_median_abs_hs"]]
    rows = re.findall(r"<tr[^>]*><td[^>]*>([^<]+)</td>", table.split("<tbody>")[1])
    assert [dash.model_label(m) for m in rank] == rows            # the producer's ranking order
    for m in sc["models"]:
        o = m["metrics"]["out_of_sample"]
        row = re.search(rf"<td[^>]*>{re.escape(dash.model_label(m['model']))}</td>(.*?)</tr>", table,
                        re.DOTALL).group(1)
        cells = re.findall(r'<td class="num">([^<]*)</td>', row)
        assert cells[2] == f"{o['n']:,}" and cells[3] == f"{o['median_abs_hs']:.2f}", m["model"]
        assert cells[4] == f"{o['inside_spread_pct']:.1f}%" and cells[6] == f"{o['rmse_vp']:.2f}", m["model"]
    for note in sc["notes"]:
        assert note in text
    # the market heatmap is the median mid IV of the quotes in each bucket, computed here independently
    res = pd.read_parquet(FIX / "surface" / SESSION / "residuals.parquet")
    mkt = res[res["model"] == res["model"].iloc[0]]
    e0 = min(mkt["expiry"].unique())
    q = mkt[(mkt["expiry"] == e0) & (mkt["k"] > -0.02) & (mkt["k"] <= 0.0)]
    heat = re.search(r'<table class="heat" id="iv-heatmap"[^>]*>(.*?)</table>', html, re.DOTALL).group(1)
    first = re.search(r"<tr[^>]*><th scope=\"row\">[^<]*</th>(.*?)</tr>", heat.split("<tbody>")[1],
                      re.DOTALL).group(1)
    tds = re.findall(r"<td[^>]*>(.*?)</td>", first, re.DOTALL)
    used = dash.k_buckets_used(mkt["k"])            # K_EDGES' buckets, plus a wing that has quotes
    assert len(tds) == len(used)
    atm = tds[used.index(dash.K_EDGES.index(0.0))]                # the (−0.02, 0] bucket
    assert re.match(rf"{q['mid_iv'].median() * 100:.1f}\b", atm), (atm, q["mid_iv"].median())
    # VIX replication and the arbitrage screen are the files' numbers
    vx = json.loads((FIX / "surface" / SESSION / "vix_replication.json").read_text())["indices"]["VIX"]
    assert f"{vx['replicated']:.2f}" in text
    assert f"Cboe close {vx['official']['close']:.2f}" in text
    arb = json.loads((FIX / "surface" / SESSION / "surface.json").read_text())["arbitrage"]
    # (P2.5: the long 'Constraint rows' heads wrap: the class gained wrap-head)
    atab = re.search(r'<table class="tbl dense[^"]*" id="arb-table">(.*?)</table>', html, re.DOTALL).group(1)
    assert f"{arb['executable_violations']['C3_vertical_butterfly']:,}" in re.search(
        r"<td>Butterfly</td>(.*?)</tr>", atab, re.DOTALL).group(1)
    # one residual grid per model, of that model's out-of-sample quotes only
    for m in res["model"].unique():
        n_oos = int((~res.loc[res["model"] == m, "in_sample"]).sum())
        assert f"of its {n_oos:,} out-of-sample quotes" in html, m
    assert "Every session the surface report covers" in html
    assert "2026-06-18" in html


def _card(html: str, href: str) -> str:
    card = re.search(rf'<a class="result" href="{re.escape(href)}"(.*?)</a>', html, re.DOTALL)
    assert card, href
    return _visible_text("<body>" + card.group(1))


def test_surface_key_result_is_the_arbitrage_free_surface_in_half_spreads(dash, art):
    """The Overview's first result is the arbitrage-free SVI surface's
    out-of-sample median |model − mid| in half-spreads from the scorecard, with
    Heston on its own held-out quotes beside it; never an in-sample figure, and
    the card says why it carries no sampling interval."""
    _, html = _get(dash, "/")
    sc = json.loads((FIX / "surface" / SESSION / "scorecard.json").read_text())
    by = {m["model"]: m for m in sc["models"]}
    s = by["svi_cal"]["metrics"]["out_of_sample"]
    h = next(m for k, m in by.items() if k.startswith("heston"))["metrics"]["out_of_sample"]
    t = _card(html, "surface.html#scorecard")
    # P1.3: named for what it is and how it is tested (surface.json fit_set.rule), and the snapshot it scores
    rule = json.loads((FIX / "surface" / SESSION / "surface.json").read_text())["fit_set"]["rule"]
    assert rule.startswith("alternate strikes")
    assert "Arbitrage-free SVI surface · alternate strikes held out" in t
    assert "Quote-grade surface" not in t
    assert "Yahoo close snapshot, no quote time" in t
    assert f"{s['median_abs_hs']:.2f} half-spreads" in t
    assert f"{s['n']:,} held-out quotes of {SESSION}" in t
    assert f"{s['inside_spread_pct']:.1f}% inside the bid–ask" in t
    dg = by["svi_cal"]["dense_grid"]
    assert f"{dg['butterfly_violations']} butterfly and {dg['calendar_violations']} calendar violations" in t
    assert f"Heston on its own held-out quotes of that session: {h['median_abs_hs']:.1f} half-spreads" in t
    # one stored session in this store: no range, and the card says what the figure is not
    assert "not a sampling interval" in t
    assert "interpolation test" in t


def test_overview_key_results_are_read_from_the_shipped_artifacts(dash, pages):
    """The three results on the shipped data: the numbers are recomputed here
    from whatever data_store holds (the bot adds sessions), never typed."""
    t1 = _card(pages["/"], "surface.html#scorecard")
    rows = dash.artifacts.surface_oos_by_session(dash.DATA)
    svi = [r for r in rows if r["model"] == "svi_cal"]
    head = dash.load_page2()["session"]
    cur = next((r for r in svi if r["session"] == head), svi[-1])
    assert f"{cur['median_abs_hs']:.2f} half-spreads" in t1
    lo, hi = min(r["median_abs_hs"] for r in svi), max(r["median_abs_hs"] for r in svi)
    assert f"Range over {len(svi)} stored sessions: {lo:.2f}–{hi:.2f} half-spreads" in t1
    t2 = _card(pages["/"], "calibration.html#rough-vs-heston")
    summ = json.loads((SHIPPED / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]
    # the headline fit's own row; the newest diagnosed session when the bot's (fail-soft)
    # diagnostics for a new fit are missing, as the card itself falls back
    row = next((r for r in summ if r["session"] == head), max(summ, key=lambda r: r["session"]))
    assert "Joint SPX/VIX fit · out of sample" in t2
    assert f"{row['sigma_cap2_oos']:.2f} vol pts, Heston" in t2
    assert f"Rough Heston on the same quotes: {row['joint_rough_oos']:.2f}" in t2
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in summ)
    assert f"lower on {lower} of {len(summ)} sessions" in t2
    hes = [r["sigma_cap2_oos"] for r in summ]
    assert f"Heston {min(hes):.2f}–{max(hes):.2f}" in t2
    t3 = _card(pages["/"], "research.html#inference")
    inf = json.loads((SHIPPED / "backtest" / "inference" / "summary.json").read_text())
    b = inf["bootstrap"]["portfolio"]["sharpe"]
    fmt = lambda x: f"{x:+.2f}".replace("-", "−")  # noqa: E731
    assert f"95% CI {fmt(b['ci95'][0])} to {fmt(b['ci95'][1])}" in t3
    assert f"p {inf['multiple_testing']['spa_published']['p_consistent']:.2f}" in t3
    assert "mark-to-model" in t3.lower()


def test_overview_keeps_research_prototypes_off_the_first_screen(dash, pages):
    """The demoted ML classifier (it loses to 'predict yesterday') sat on the
    Overview next to the rule label; it is now a Research-page card, and the
    Overview's loader no longer even computes it. (P1.7: it is named once, as
    a negative result, in the 'Does not work' column of the findings strip.)"""
    strip = re.search(r'<div class="findings" id="findings".*?\n</div>', pages["/"], re.DOTALL)
    if strip:
        cols = _findings(pages["/"])
        where = [h for _, h, items in cols if any("classifier" in it for it in items)]
        assert where in ([], ["✕ Does not work"]), where
    idx = _visible_text(pages["/"].replace(strip.group(0), "") if strip else pages["/"])
    for word in ("classifier", "XGBoost", "P(R0", "fear_premium"):
        assert word not in idx.replace("ML regime classifier, the NN pricer", ""), word
    assert 'id="research-classifier"' not in pages["/"]
    assert 'id="research-classifier"' in pages["/research"]
    assert 'id="label-history"' in pages["/research"]
    assert not any(k.startswith("clf_") for k in dash.load_page1())
    # the Overview names where the prototypes are, and says none feeds its numbers
    assert "none feeds a number on this page" in idx


def test_overview_pipeline_panel_reads_the_stored_records(dash, pages):
    """Cboe book captures, accepted fits, artifacts and the forward ledger as
    the stored records state them."""
    pl = dash.pipeline_state(dash.artifacts.load_diagnostics_summary(dash.DATA))
    text = _visible_text(pages["/"])
    cboe = SHIPPED / "options" / "cboe"
    caps = sorted(cboe.glob("*.json")) if cboe.is_dir() else []
    for rec in (json.loads(f.read_text()) for f in caps):
        latest = next(c for c in pl["captures"] if c["underlying"] == rec["underlying"])
        if latest["file"].startswith(f"{rec['underlying'].lower()}_{rec['session']}"):
            assert f"{rec['underlying']} {rec['session']}: {rec['rows']:,} lines" in text
    assert f"{pl['fits']} (model_version 2), newest {pl['fits_newest']}" in text
    assert f"{len(pl['surface_sessions'])} sessions" in text
    if pl["forward"]["state"] != "ok":
        assert dash.forward_ledger_protocol()["pending"] in text


def test_rough_vs_heston_panel_lists_every_diagnosed_session(dash, art):
    resp, html = _get(dash, "/calibration")
    assert _errors(resp) == 0
    rows = json.loads((FIX / "diagnostics" / "summary.json").read_text())["rows"]
    tab = re.search(r'<table class="tbl dense" id="rough-heston-table">(.*?)</table>', html,
                    re.DOTALL).group(1)
    body = tab.split("<tbody>")[1]
    assert len(re.findall(r"<tr[ >]", body)) == len(rows)
    for r in rows:
        cell = re.search(rf'<td><span class="nowrap">{r["session"]}</span>(.*?)</tr>', body, re.DOTALL)
        line = _visible_text("<body>" + cell.group(1))
        assert f"{r['sigma_cap2_oos']:.2f}" in line
        assert f"{r['joint_rough_oos']:.2f}" in line
        assert f"{r['joint_rough_H']:.2f}" in line
    verdict = _visible_text("<body>" + re.search(r'id="rough-verdict">(.*?)</p>', html, re.DOTALL).group(1))
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in rows)
    same = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in rows)
    assert f"lower on {lower} of {len(rows)} sessions" in verdict
    assert f"on {same} it returns H = ½" in verdict
    # Greek letters in upper-cased headings and chips keep their case (σ, not Σ)
    assert re.search(r'<span class="chip chip-watch"><span class="sym">σ</span> at cap</span>', html)
    assert '<th scope="col" class="num">σ' not in html


def test_inventory_does_not_claim_flat_carry_for_the_per_maturity_greeks():
    """M2's limitation said 'Flat carry across maturities' after the Greeks surface
    moved to per-maturity carry (C23); the inventory must describe the shipped surface."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(SHIPPED / "greeks" / "greeks_surface.parquet").metadata
    prov = json.loads(meta[b"greeks_provenance"])
    models = json.loads((ROOT / "dashboard" / "model_inventory.json").read_text())["models"]
    greeks = next(m for m in models if m["module"].endswith("greeks/risk_monitor.py"))
    if (prov.get("carry") or {}).get("by_maturity"):
        assert "flat carry" not in greeks["limitations"].lower()
        assert "per maturity" in greeks["limitations"]


def test_risk_provenance_states_the_per_maturity_carry(dash, pages):
    """The C23 Greeks surface has one r(T), q(T) per maturity and no scalar r, q;
    the banner printed 'r = —%, q = —%'."""
    import pyarrow.parquet as pq

    meta = pq.read_schema(SHIPPED / "greeks" / "greeks_surface.parquet").metadata
    prov = json.loads(meta[b"greeks_provenance"])
    text = _visible_text(pages["/risk"])
    assert "r = —%" not in text
    rows = (prov.get("carry") or {}).get("by_maturity") or []
    if prov.get("r") is None and rows:
        r_lo, r_hi = min(c["r"] for c in rows) * 100, max(c["r"] for c in rows) * 100
        q_lo, q_hi = min(c["q"] for c in rows) * 100, max(c["q"] for c in rows) * 100
        # the site's formatter: a negative bound takes the typographic minus (q(T) on 2026-10-26)
        f = dash._fmt
        assert f"r(T) {f(r_lo, '.2f')}–{f(r_hi, '.2f')}%" in text
        assert f"q(T) {f(q_lo, '.2f')}–{f(q_hi, '.2f')}% from the fit's parity forwards" in text


def test_corrections_page_shows_the_c23_waterfall_from_its_file(pages):
    bridge = pd.read_csv(SHIPPED / "backtest" / "c23_bridge.csv")
    tab = re.search(r'<table class="tbl dense[^"]*" id="c23-bridge"[^>]*>(.*?)</table>',
                    pages["/corrections"],
                    re.DOTALL).group(1)
    body = _visible_text("<body>" + tab)
    assert len(re.findall(r"<tr>", tab.split("<tbody>")[1])) == len(bridge)
    for _, r in bridge.iterrows():
        assert r["label"] in body
        pnl = f"${abs(r['pnl']):,.0f}"
        assert ("−" if r["pnl"] < 0 else "+") + pnl in body
    assert 'data-round="C23"' in pages["/corrections"]


def test_diagnostics_summary_contract(dash, tmp_path):
    """summary.json rows: finite numbers or null, H in [0, 1/2], one row per session."""
    da = dash.artifacts
    good = json.loads((FIX / "diagnostics" / "summary.json").read_text())
    tmp = tmp_path / "data_store"
    (tmp / "calibrations" / "diagnostics").mkdir(parents=True)
    path = tmp / "calibrations" / "diagnostics" / "summary.json"
    path.write_text(json.dumps(good))
    assert da.load_diagnostics_summary(tmp)["state"] == "ok"
    for breaker, msg in ((lambda o: o["rows"][0].update(joint_rough_H=0.7), "outside [0, 1/2]"),
                         (lambda o: o["rows"].append(dict(o["rows"][0])), "duplicate sessions"),
                         (lambda o: o["rows"][0].update(sigma_cap2_oos="n/a"), "not a finite number"),
                         (lambda o: o.update(rows=[]), "non-empty 'rows'")):
        bad = json.loads(json.dumps(good))
        breaker(bad)
        path.write_text(json.dumps(bad))
        with pytest.raises(da.ArtifactError, match=re.escape(msg)):
            da.load_diagnostics_summary(tmp)
    path.unlink()
    assert da.load_diagnostics_summary(tmp)["state"] == "absent"


def test_residual_stats_score_each_model_on_its_own_out_of_sample_quotes(dash):
    """Two models on the same three quotes; each has its own in-sample set."""
    rows = []
    for model, ivs, ins in (("heston_v2", (0.21, 0.20, 0.30), (True, False, False)),
                            ("svi", (0.20, 0.205, 0.25), (False, True, False))):
        for k, mid, bid, ask, iv, i in zip((-0.1, 0.0, 0.1), (0.20, 0.20, 0.25), (0.19, 0.195, 0.24),
                                           (0.21, 0.205, 0.26), ivs, ins):
            rows.append({"model": model, "expiry": "2026-11-20", "T_days": 35.0, "k": k, "mid_iv": mid,
                         "bid_iv": bid, "ask_iv": ask, "model_iv": iv, "in_sample": i})
    p = Path(__import__("tempfile").mkdtemp()) / "residuals.parquet"
    pd.DataFrame(rows).to_parquet(p)
    res = dash.artifacts.read_residuals(p, SESSION)
    h = dash.artifacts.residual_stats(res, "heston_v2")
    s = dash.artifacts.residual_stats(res, "svi")
    assert h["scope"] == s["scope"] == "out-of-sample"
    assert h["n"] == s["n"] == 2
    # heston OOS: k=0 (0.20 vs 0.20, inside) and k=0.1 (0.30 vs mid 0.25: 5 vp, half-spread
    # (0.26 − 0.24) × 50 = 1 vp, so 5 half-spreads); its anchor at k=−0.1 is not scored
    assert h["inside_spread_pct"] == 50.0
    assert h["rmse_vp"] == pytest.approx(np.sqrt((0 + 5.0 ** 2) / 2))
    assert h["median_abs_hs"] == pytest.approx((0.0 + 5.0) / 2)
    # svi OOS: k=−0.1 (inside) and k=0.1 (0.25 vs 0.25, inside)
    assert s["inside_spread_pct"] == 100.0
    assert s["median_abs_hs"] == pytest.approx(0.0)


def _rewrite_json(path: Path, fn) -> None:
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _bad_residuals(kind):
    def go(data):
        p = data / "surface" / SESSION / "residuals.parquet"
        df = pd.read_parquet(p)
        if kind == "vol_points":
            df["mid_iv"] = df["mid_iv"] * 100.0
        elif kind == "other_session":
            df["session"] = "2026-06-18"
        elif kind == "unequal_models":
            df = df.iloc[1:]
        elif kind == "negative_spread":
            df.loc[df.index[0], "half_spread_vp"] = -0.1
        df.to_parquet(p)
    return go


MALFORMED = {
    "residuals in vol points": ("/surface", _bad_residuals("vol_points"), "looks like vol points"),
    "residuals from another session": ("/surface", _bad_residuals("other_session"),
                                       "rows from another session"),
    "models on different quotes": ("/surface", _bad_residuals("unequal_models"), "different quote sets"),
    "negative half-spread": ("/surface", _bad_residuals("negative_spread"), "half_spread_vp must be finite"),
    "inside share above 100%": ("/", lambda d: _rewrite_json(
        d / "surface" / SESSION / "scorecard.json",
        lambda o: o["models"][0]["metrics"]["out_of_sample"].update(inside_spread_pct=120.0)), "not a percentage"),
    "scorecard without out-of-sample metrics": ("/surface", lambda d: _rewrite_json(
        d / "surface" / SESSION / "scorecard.json",
        lambda o: o["models"][0]["metrics"].pop("out_of_sample")), "no out_of_sample metrics"),
    "VIX replicated as a decimal": ("/surface", lambda d: _rewrite_json(
        d / "surface" / SESSION / "vix_replication.json",
        lambda o: o["indices"]["VIX"].update(replicated=0.197)), "is not in index points"),
    "negative arbitrage count": ("/model-risk", lambda d: _rewrite_json(
        d / "surface" / SESSION / "surface.json",
        lambda o: o["arbitrage"]["executable_violations"].update(C3_vertical_butterfly=-1)), "must be a count"),
    "scorecard of another session": ("/surface", lambda d: _rewrite_json(
        d / "surface" / SESSION / "scorecard.json", lambda o: o.update(session="2026-06-18")), "session 2026-06-18"),
    "inverted bootstrap interval": ("/research", lambda d: _rewrite_json(
        d / "backtest" / "inference" / "summary.json",
        lambda o: o["bootstrap"]["portfolio"]["sharpe"].update(ci95=[0.5, -0.5])), "is not an interval"),
    "p-value above one": ("/research", lambda d: _rewrite_json(
        d / "backtest" / "inference" / "summary.json",
        lambda o: o["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"].update(p_value=1.3)), "not a probability"),
    "unknown inference schema": ("/", lambda d: _rewrite_json(
        d / "backtest" / "inference" / "summary.json", lambda o: o.update(schema="something_else/2")), "schema"),
    "edited forward ledger": ("/research", lambda d: _rewrite_json(
        d / "forward" / "ledger_2026-03-24.json",
        lambda o: o["summary"]["s3"].update(realised_pnl_net=1.0)), "does not match"),
    "missing forward ledger": ("/research", lambda d: (d / "forward" / "ledger_2026-03-27.json").unlink(),
                               "previous session"),
    "unsorted singular values": ("/calibration", lambda d: _rewrite_json(
        d / "calibrations" / "diagnostics" / f"{SESSION}.json",
        lambda o: o["identifiability"].update(rel_jacobian_singular_values=[0.1, 2.0, 0.5, 0.2, 0.01])),
        "largest first"),
    "rough H above one half": ("/calibration", lambda d: _rewrite_json(
        d / "calibrations" / "diagnostics" / "summary.json",
        lambda o: o["rows"][-1].update(joint_rough_H=0.9)), "outside [0, 1/2]"),
    "model VVIX not a number": ("/calibration", lambda d: _rewrite_json(
        d / "calibrations" / "diagnostics" / f"{SESSION}.json",
        lambda o: o["sigma_profile"]["rows"][0].update(vvix_model="n/a")),
        "vvix_model is not a finite number"),
    "diagnostics of another session": ("/calibration", lambda d: _rewrite_json(
        d / "calibrations" / "diagnostics" / f"{SESSION}.json", lambda o: o.update(session="2026-06-18")),
        "session 2026-06-18"),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_a_malformed_artifact_is_a_loader_error_not_a_quiet_gap(dash, tmp_path, monkeypatch, case):
    route, breaker, message = MALFORMED[case]
    data = _artifact_store(tmp_path)
    breaker(data)
    monkeypatch.setattr(dash, "DATA", data)
    resp, html = _get(dash, route)
    assert _errors(resp) >= 1, case
    assert message in _visible_text(html), (case, message)


def test_a_malformed_artifact_blocks_the_freeze(dash, freezer, tmp_path, monkeypatch):
    data = _artifact_store(tmp_path)
    _bad_residuals("vol_points")(data)
    monkeypatch.setattr(dash, "DATA", data)
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 1
    assert not out.exists() or not any(out.iterdir())


def test_valid_artifacts_freeze_and_stay_within_the_page_budget(dash, freezer, art, tmp_path):
    """Every page publishes with every artifact present, and no page is heavier
    than its budget (raw HTML; GitHub Pages serves it gzipped at about a fifth)."""
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 0
    # research.html: 400 kB until B12 added every portfolio round trip (206 points), two
    # histograms and the trade and drawdown tables, about 30 kB raw (404 kB with them)
    budget = {"index.html": 120_000, "surface.html": 600_000, "calibration.html": 300_000,
              "risk.html": 300_000, "research.html": 430_000, "corrections.html": 120_000,
              "model-risk.html": 80_000}
    for name, cap in budget.items():
        size = (out / name).stat().st_size
        assert size <= cap, (name, size)
    # shared assets: the chart library is the only large script, fonts are subsets
    import gzip
    static = ROOT / "dashboard" / "static"
    assert len(gzip.compress((static / "vendor" / "echarts-6.1.0-desk.min.js").read_bytes())) < 240_000
    # P4.1: the quote inspector, the residual strip and the linked chart groups
    # raised the page script from 51 KB; it is served gzipped like the pages. F4's keyboard
    # readout of the per-trade strip took it from 29.8 to 30.5 KB gzipped; R3-T9's table CSV
    # numbers (one-number cells parsed, block children spaced) past 31 KB
    assert (static / "desk.js").stat().st_size < 110_000
    assert len(gzip.compress((static / "desk.js").read_bytes())) < 31_500
    assert (static / "desk.css").stat().st_size < 60_000
    assert sum(p.stat().st_size for p in (static / "fonts").glob("*.woff2")) < 80_000


def test_research_inference_panels_show_the_summary(dash, art):
    resp, html = _get(dash, "/research")
    assert _errors(resp) == 0
    s = json.loads((FIX / "inference" / "summary.json").read_text())
    b = s["bootstrap"]["portfolio"]["sharpe"]
    boot = re.search(r'<table class="tbl dense" id="bootstrap-table">(.*?)</table>', html, re.DOTALL).group(1)
    port = _visible_text("<body>" + re.search(r"<td[^>]*>Portfolio</td>(.*?)</tr>", boot,
                                               re.DOTALL).group(1))
    def fmt(x):
        return f"{x:+.2f}".replace("-", "−")
    assert f"{fmt(b['ci95'][0])} to {fmt(b['ci95'][1])}" in port
    nul = re.search(r'<table class="tbl dense" id="null-table">(.*?)</table>', html, re.DOTALL).group(1)
    s3 = s["nulls"]["sleeves"]["s3"]
    row = _visible_text("<body>" + re.search(r"<td[^>]*>S3 short SPX straddle</td>(.*?)</tr>", nul,
                                              re.DOTALL).group(1))
    assert f"{s3['circular']['total_pnl']['p_value']:.3f}" in row
    assert f"{s3['shuffle']['total_pnl']['p_value']:.3f}" in row
    # the KPI interval and the overview key result use the same bootstrap row
    assert f"95% CI {fmt(b['ci95'][0])} to {fmt(b['ci95'][1])} (bootstrap)" in _visible_text(html)
    _, idx = _get(dash, "/")
    assert f"95% CI {fmt(b['ci95'][0])} to {fmt(b['ci95'][1])}" in _visible_text(idx)
    # forward ledger: three registered sessions whose hash chain checks out
    assert "3 sessions registered, 2026-03-24 → 2026-05-29" in _visible_text(html)
    # forecast table: PDV's QLIKE rank as the file states
    fe = json.loads((FIX / "inference" / "forecast_eval.json").read_text())["backtest_window"]
    ftab = re.search(r'<table class="tbl dense" id="forecast-table">(.*?)</table>', html, re.DOTALL).group(1)
    pdv = _visible_text("<body>" + re.search(r"<td[^>]*>PDV</td>(.*?)</tr>", ftab,
                                              re.DOTALL).group(1))
    assert f"{fe['losses']['qlike']['pdv']:.3f}" in pdv
    assert str(fe["rank"]["qlike"]["pdv"]) in pdv


def test_forward_ledger_hash_chain_matches_the_producer(dash, tmp_path):
    """The reader's canonical SHA-256 is the producer's: the committed files chain."""
    data = _artifact_store(tmp_path, surface=False, inference=False, diagnostics=False)
    led = dash.artifacts.load_forward_ledger(data)
    assert led["state"] == "ok"
    assert led["sessions"] == ["2026-03-24", "2026-03-27", "2026-05-29"]
    files = sorted((FIX / "forward").glob("ledger_*.json"))
    for a, b in itertools.pairwise(files):
        prev = json.loads(b.read_text())["previous"]
        assert prev["sha256"] == dash.artifacts._canonical_sha256(json.loads(a.read_text()))


def test_calibration_page_reads_the_diagnostics_and_the_quote_smiles(dash, art):
    resp, html = _get(dash, "/calibration")
    assert _errors(resp) == 0
    diag = json.loads((FIX / "diagnostics" / f"{SESSION}.json").read_text())
    text = _visible_text(html)
    ident = diag["identifiability"]
    assert f"{ident['rel_jtj_condition_number']:,.0f}" in text
    ms = diag["trf"]["multistart"]
    assert f"{ms['n_starts_at_best']} / {ms['n_starts']}" in text
    prof = re.search(r'<table class="tbl dense" id="sigma-profile">(.*?)</table>', html, re.DOTALL).group(1)
    # every refit a row (the production box's row is highlighted: <tr class="em">)
    assert len(re.findall(r"<tr[ >]", prof.split("<tbody>")[1])) == len(diag["sigma_profile"]["rows"])
    assert "at cap" in prof
    assert "Rough Heston (lifted)" in html
    assert f"{diag['spx_only']['rough_heston']['params']['H']:.3f}" in html
    # the κ tile carries the diagnostics' standard error
    assert f"SE {ident['standard_errors']['kappa']:.3g}" in text
    # the anchor expiry present in the residual table is drawn from quotes, with Heston's anchors counted
    res = pd.read_parquet(FIX / "surface" / SESSION / "residuals.parquet")
    h = res[res["model"].str.startswith("heston")]
    with open(SHIPPED / "calibrations" / f"joint_cal_{SESSION}.pkl", "rb") as f:
        cal = pickle.load(f)
    shared = sorted({a["expiry"] for a in cal["anchors_by_expiry"]} & set(h["expiry"]))
    assert shared
    for e in shared:
        q = h[h["expiry"] == e]
        assert (f"{len(q)} quotes, {int(q['in_sample'].sum())} of them fit anchors; Heston inside the bid–ask on "
                f"{q['inside_spread'].mean() * 100:.0f}%") in text, e


def test_model_risk_checks_use_the_surface_artifacts(dash, art):
    resp, _html = _get(dash, "/model-risk")
    assert _errors(resp) == 0
    d = dash.load_model_risk()
    mon = {m["check"]: m for m in d["monitors"]}
    sc = json.loads((FIX / "surface" / SESSION / "scorecard.json").read_text())
    h = next(m for m in sc["models"] if m["model"].startswith("heston"))["metrics"]["out_of_sample"]
    hs = mon["Heston median |model − mid|, out of sample (half-spreads)"]
    lim = dash.MONITOR_LIMITS["median_abs_hs"]
    want = "ok" if h["median_abs_hs"] <= lim[0] else "watch" if h["median_abs_hs"] <= lim[1] else "breach"
    assert hs["status"] == want
    arb = json.loads((FIX / "surface" / SESSION / "surface.json").read_text())["arbitrage"]
    share = arb["n_quotes_in_executable_violation"] / arb["n_quotes"] * 100
    arb_row = mon["Quotes failing static arbitrage at bid/ask (% of scored quotes)"]
    assert arb_row["value"].startswith(f"{share:.1f}%")


def test_model_risk_v1_counts_the_single_quote_outliers_apart(dash):
    """R3-D1: V1 read '6.8% (198 of 2,913) BREACH' under 'executable static-arbitrage
    row', the label the Surface tile gave one bad Yahoo quote. The check is now
    named for what the screen measures, and its note counts the largest rows a
    single-quote outlier explains, with the leg they share."""
    row = next(m for m in dash.load_model_risk()["monitors"] if m["key"] == "arb_exec_share")
    assert row["check"] == "Quotes failing static arbitrage at bid/ask (% of scored quotes)"
    assert "executable" not in row["check"].lower()
    lv = dash.arb_violations(json.loads(_surface_file("surface.json").read_text())["arbitrage"])
    bad = [v["bad_quote"] for v in lv if v["bad_quote"]]
    if bad:
        assert (f"Single-quote outliers: {len(bad)} of its {len(lv)} largest mid violations "
                f"(bad quote {bad[0]}") in row["note"]
    else:
        assert "outlier" not in row["note"]


def test_model_risk_without_artifacts_marks_their_checks_not_applicable(dash, no_art):
    d = dash.load_model_risk()
    na = [m for m in d["monitors"] if m["value"] == "artifact not yet generated"]
    assert len(na) == 3
    assert all(m["status"] == "na" for m in na)


# ── Corrections registry and model inventory ─────────────────────────────────

def test_corrections_name_each_pinning_test_by_its_file(dash, pages):
    """C23 verify: 'Pinned by' printed 35 of the 64 tests as Class::test with no
    file (TestNoLookAhead::test_walk_forward_no_lookahead_poison) and the other
    29 with the whole joint_vol_calibration/tests/ path, which broke letter by
    letter at phone width. Each is now its file and test, in one form, wrapping
    after '/' or '_'."""
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    want = [str(dash.path_breaks(t.split("/")[-1])) for r in reg["rounds"] for f in r["findings"]
            for t in f.get("tests", [])]
    got = []
    for dd in re.findall(r"<dt>Pinned by</dt><dd>(.*?)</dd>", pages["/corrections"], flags=re.DOTALL):
        got += re.findall(r'<code class="path">(.*?)</code>', dd, flags=re.DOTALL)
    assert got == want
    assert all(re.sub(r"<[^>]+>", "", g).startswith("test_") and ".py::" in g for g in got)


def test_every_correction_links_a_real_commit_and_existing_tests():
    """The Corrections page ties each round to its commit(s) and pinning tests;
    a renamed test or a mistyped hash must fail here, not 404 on the site."""
    import ast
    import subprocess

    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    rounds = [r["id"] for r in reg["rounds"]]
    assert rounds == [f"C{i}" for i in range(16, 24)]
    shallow = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
                             capture_output=True, text=True, check=False).stdout.strip() != "false"
    for r in reg["rounds"]:
        assert r["commits"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["date"]), r["id"]
        for c in r["commits"]:
            assert re.fullmatch(r"[0-9a-f]{40}", c), (r["id"], c)
            if not shallow:
                exists = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{c}^{{commit}}"],
                                        check=False)
                assert exists.returncode == 0
                day = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%ad", "--date=short", c],
                                     capture_output=True, text=True, check=False).stdout.strip()
                # a round may span days (C23: 2026-10-20 to 2026-10-21)
                assert r["date"] <= day <= r.get("date_end", r["date"]), (r["id"], c, day)
        for f in r["findings"]:
            assert f["status"] in ("fixed", "quantified"), f["flaw"]
            for node in f.get("tests", []):
                path, *names = node.split("::")
                tree = ast.parse((ROOT / path).read_text())
                scope = tree.body
                for name in names:
                    hit = next((n for n in scope
                                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
                    assert hit is not None, node
                    scope = hit.body


def test_model_inventory_paths_exist(dash):
    """The Model Risk page says this test checks every listed module and artifact."""
    inv = json.loads((ROOT / "dashboard" / "model_inventory.json").read_text())
    assert "test_dashboard.py checks that every listed module and artifact path exists" in inv["about"]
    ids = [m["id"] for m in inv["models"]]
    assert ids == [f"M{i}" for i in range(1, len(ids) + 1)]
    for m in inv["models"]:
        assert (ROOT / m["module"]).is_file(), m["id"]
        # an output not yet written says so and why (the forward ledger's first entry)
        outs = [m["outputs"]] if isinstance(m["outputs"], str) else m["outputs"]
        assert (outs and all((ROOT / o).exists() for o in outs)) or m.get("outputs_pending"), (m["id"], outs)
        assert m["tier"] in ("production", "backtest", "research"), m["id"]
        assert m["materiality"] in ("high", "medium", "low"), m["id"]
        assert m.get("page") is None or m["page"] in dash.ROUTES, m["id"]
    vendors = [v["id"] for v in inv["vendor_inputs"]]
    assert vendors == [f"V{i}" for i in range(1, len(vendors) + 1)]


def test_artifact_pages_keep_the_number_style(dash, art):
    """Producer prose (inference headlines, scorecard notes, VIX gap notes) is
    printed through the prose filter: '-0.15' and '$-158' become '−0.15', '−$158'."""
    client = dash.app.test_client()
    for route in dash.ROUTES:
        _assert_number_style(route, client.get(route).get_data(as_text=True))
    assert dash.prose("CI [-0.68, 0.50]; P&L $-158; C16-C22; 2018-01-02") == \
        "CI [−0.68, 0.50]; P&L −$158; C16-C22; 2018-01-02"


# ── C23 adversarial review, UI findings (UI-1 … UI-18, CLM-01/08/09) ──────────

def _cells(html: str, table_id: str) -> list[list[str]]:
    """Visible cell texts of a heat table, one list per body row (row head first)."""
    tbl = re.search(rf'<table class="heat" id="{re.escape(table_id)}"[^>]*>(.*?)</table>', html, re.DOTALL)
    assert tbl, table_id
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", tbl.group(1).split("<tbody>")[1], re.DOTALL):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL)
        rows.append([re.sub(r"<span class=\"sr-only\">.*?</span>", "", c).strip() for c in cells])
    return rows


def test_risk_delta_grid_is_the_otm_options_forward_delta(dash, pages):
    """UI-3: the grid said 'OTM option, forward delta' over the stored call spot
    delta e^{−qT}N(d₁), so a 30%-OTM put read +0.99. Each cell is now the OTM
    option's forward delta: N(d₁) − 1 for the put below the forward, N(d₁) for
    the call above it, recomputed here from the stored surface."""
    g = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    html = pages["/risk"]
    cap = re.search(r'<caption id="heat-delta-cap">(.*?)</caption>', html, re.DOTALL).group(1)
    assert "N(d₁) − 1 for the put at K &lt; F" in cap
    assert "N(d₁) for the call at K ≥ F" in cap
    rows = {r[0]: r[1:] for r in _cells(html, "heat-delta")}
    cols = sorted(float(x) for x in g["log_moneyness"].round(4).unique())
    n_put = n_call = 0
    for r in g.itertuples():
        nd1 = r.delta * np.exp(r.q * r.T)
        want = nd1 - 1.0 if r.log_moneyness < 0 else nd1
        got = rows[f"{int(r.T_days)}d"][cols.index(round(float(r.log_moneyness), 4))]
        assert got == dash.signed(want, ".2f"), (r.T_days, r.log_moneyness, got, want)
        n_put += r.log_moneyness < 0
        n_call += r.log_moneyness >= 0
    assert n_put
    assert n_call
    # the 30-day, 30%-OTM put: a small negative delta, never the +0.99 the grid showed
    deep_put = g[(g["T_days"] == 30) & (g["log_moneyness"].round(2) == -0.30)].iloc[0]
    assert dash.otm_forward_delta(deep_put.delta, deep_put.q, deep_put.T, deep_put.log_moneyness) < 0
    assert float(rows["30d"][0].replace("−", "-")) <= 0          # '−0.01' or '0.00', never '+0.99'


def test_one_colour_per_model_on_every_surface_chart(dash, pages):
    """UI-2: the slice drew Heston in the orange the by-expiry chart beside it used
    for SVI, and had no legend. Each model now has one colour on every chart, and
    the slice has one toggle per model (all four), the market quotes and the
    Heston anchors, with the forward marked."""
    html = pages["/surface"]
    sl, be = _spec(html, "chart-slices"), _spec(html, "chart-by-expiry")
    by_name = {s["name"]: s["color"] for s in be["series"]}
    models = json.loads(_surface_file("scorecard.json").read_text())["models"]
    keys = [m["model"] for m in models]
    assert len({dash.model_colour(k) for k in keys}) == len(keys)            # distinct
    for k in keys:
        label = dash.model_label(k)
        assert by_name[label] == dash.model_colour(k), k
        slice_cols = {s["color"] for s in sl["series"] if s.get("legendKey") == label}
        assert slice_cols == {dash.model_colour(k)}, (k, slice_cols)
    legend = [it["name"] for it in sl["legend_items"]]
    assert {dash.model_label(k) for k in keys} <= set(legend)
    assert {"Bid–ask", "Mid", "Heston fit anchors"} <= set(legend)
    assert len(legend) == len(set(legend))                                   # one button per key
    shown = {it["name"] for it in sl["legend_items"] if not it["hidden"]}
    assert {dash.model_label("heston_v2"), dash.model_label("svi_cal")} <= shown
    assert sl["marks"]["x"][0]["value"] == 0                                  # the forward
    # the same model keeps its colour on the calibration page
    cal = pages["/calibration"]
    smile = next(_spec(cal, cid) for cid in re.findall(r'id="(chart-smile-[^"]+)" data-spec', cal))
    heston = next(s for s in smile["series"] if s.get("legendKey", s["name"]) == dash.model_label("heston_v2")
                  and s["type"] == "line")
    assert heston["color"] == dash.model_colour("heston_v2")


def test_smile_slice_model_points_share_the_quotes_x_values(dash, pages):
    """UI-2/UI-17: all four models on the slice within the page budget: a model's
    points are at its expiry's quotes (P4.1: every series is filled from the one
    packed quote block, desk_charts.resolve as desk.js does), and the IVs are the
    residual table's model IVs at those strikes."""
    sl = _spec(pages["/surface"], "chart-slices")
    by = {(s["group"], s["quote"]): s for s in sl["series"] if s.get("quote") in ("mid", "ba")}
    res = pd.read_parquet(_surface_file("residuals.parquet"))
    with_x = [s for s in sl["series"] if s.get("quote") == "model"]
    assert len(with_x) == res["model"].nunique() * res["expiry"].nunique()      # every model, every expiry
    for s in with_x:
        mids = by[(s["group"], "mid")]["data"]
        assert len(mids) == len(s["data"]), s["name"]
        e, key = s["group"], s["model"]
        assert dash.model_label(key) == s["legendKey"]
        q = res[(res["model"] == key) & (res["expiry"] == e)].sort_values("k", kind="mergesort")
        assert [p[0] for p in s["data"]] == [m[0] for m in mids], s["name"]
        assert [m[0] for m in mids] == [round(float(k), 6) for k in q["k"]], s["name"]
        assert [p[1] for p in s["data"]] == pytest.approx([float(v) * 100 for v in q["model_iv"]], abs=1e-4)
        assert [v for p in by[(e, "ba")]["data"] for v in p[1:]] == pytest.approx(
            [v for b, a in zip(q["bid_iv"], q["ask_iv"]) for v in (b * 100, a * 100)], abs=1e-5)


def test_smile_slice_has_a_per_expiry_table_of_its_data(dash, pages):
    """UI-17: without the chart the slice offered 'the values listed below', a
    table of quote counts. Its data block now has one table per expiry (bid,
    mid, ask and every model's IV by ln(K/F) bucket); the default expiry's is
    shown, the others are hidden until their tab is chosen (all show without JS)."""
    html = pages["/surface"]
    block = re.search(r'<div class="chart-data sr-only" id="chart-slices-data">(.*?)\n</div>', html,
                      re.DOTALL).group(1)
    groups = re.findall(r'<div class="chart-data-group" data-group="([^"]+)"( hidden)?>', block)
    exps = re.findall(r'data-value="(\d{4}-\d{2}-\d{2})">\d+d</button>', html)
    assert [g for g, _ in groups] == exps
    assert sum(1 for _, h in groups if not h) == 1
    head = re.search(r'data-group="[^"]+"[^>]*><table>\s*<caption>[^<]+</caption>\s*<thead><tr>(.*?)</tr>',
                     block, re.DOTALL).group(1)
    for col in ("Bid", "Mid", "Ask", "Heston", "SVI-cal", "SVI", "eSSVI"):
        assert f">{col}</th>" in head, col
    assert ('[role="tabpanel"][hidden], .chart-data [data-group][hidden] { display: block !important; }'
            in html)


def test_iv_and_residual_grids_hold_every_quote(dash, pages):
    """UI-14: pd.cut over ±0.30 dropped the 108 quotes at ln(K/F) ≤ −0.30 (the deep
    put wing) from every grid, and the colour ramp spanned quotes no cell showed.
    The grids now carry an open wing column; their cells add up to the quotes."""
    d = dash.load_surface()
    res = pd.read_parquet(_surface_file("residuals.parquet"))
    mkt = res[res["model"] == res["model"].iloc[0]]
    assert (mkt["k"] <= -0.30).sum() > 0
    assert d["iv_heat"]["n"] == len(mkt) == d["n_quotes"]
    assert d["iv_heat"]["cols"][0] == "≤ −0.30"
    shown = [float(c["text"]) for r in d["iv_heat"]["rows"] for c in r["cells"] if not c.get("empty")]
    assert d["iv_heat"]["hi"] == pytest.approx(max(shown), abs=0.051)       # the ramp spans the cells
    assert d["iv_heat"]["lo"] == pytest.approx(min(shown), abs=0.051)
    for r in d["resid_heat"]:
        assert sum(c["n"] for row in r["rows"] for c in row["cells"]) == r["n"], r["key"]
    assert dash.k_bucket_label(0) == "≤ −0.30"
    assert dash.k_bucket_label(len(dash.K_EDGES)) == "> +0.30"
    assert list(dash.k_bucket(pd.Series([-0.355, -0.30, -0.25, 0.31]))) == [0, 0, 1, len(dash.K_EDGES)]


def test_scorecard_and_by_expiry_chart_say_which_quotes_they_score(dash, pages):
    """UI-16: 'same quotes, same metric' sat above 1,496 vs 2,860 scored quotes, and
    the by-expiry chart beside it mixed in the fitted quotes. The title follows the
    quote counts, the chart is each model's out-of-sample quotes (recomputed here),
    and the history column names the repair's moves."""
    html = pages["/surface"]
    sc = json.loads(_surface_file("scorecard.json").read_text())
    ns = {m["metrics"]["out_of_sample"]["n"] for m in sc["models"]}
    title = re.search(r'<h2 class="panel-title" id="scorecard-t">(.*?)</h2>', html).group(1)
    assert ("same quotes" in title) is (len(ns) == 1)
    be = _spec(html, "chart-by-expiry")
    res = pd.read_parquet(_surface_file("residuals.parquet"))
    for s in be["series"]:
        key = next(k for k in res["model"].unique() if dash.model_label(k) == s["name"])
        q = res[(res["model"] == key) & ~res["in_sample"].astype(bool)]
        med = q["error_hs"].abs().groupby(q["expiry"]).median()
        exps = sorted(med.index)
        got = [v for v in s["data"] if v is not None]
        assert got == pytest.approx([float(med[e]) for e in exps if e in med.index], rel=1e-5), s["name"]
    assert "Out-of-sample quotes only" in html
    assert "Repair moves outside spread" in html
    assert "Arb. outside spread" not in html


def test_surface_atm_over_vix_is_named_by_its_method(dash, pages):
    """UI-5: Surface showed '30-day ATM vol / VIX' = 0.712 and linked to Research,
    whose table gives 0.702 for the same chain. Each number now carries its own
    method's name, and the β estimator's value equals the Research page's."""
    surf, res = _visible_text(pages["/surface"]), pages["/research"]
    vb = json.loads(_surface_file("vix_replication.json").read_text())["beta"]
    assert "30-day ATM vol / VIX on this chain" not in surf
    if "atm_over_vix_official" not in vb:       # the chain does not bracket 30 days: no VIX-term ATM
        assert "VIX-term ATM / VIX on this chain" not in surf
    else:
        assert f"VIX-term ATM / VIX on this chain: {vb['atm_over_vix_official']:.3f}" in surf
        session = _surface_file("residuals.parquet").parent.name
        row = re.search(rf'<td class="nowrap">{session}</td><td>[^<]*</td><td class="num">([0-9.]+)</td>',
                        res)
        if row:     # the session's chain is one the β estimator read
            assert f"gives {row.group(1)} on the same chain" in surf
        else:
            assert "on the same chain" not in surf
    assert "β estimator ATM 30d / VIX" in surf
    assert "VIX-term ATM / VIX" in surf


def test_research_s3_timing_headline_is_the_c23_exposure_matched_null(dash, pages):
    """UI-6: the hero and KPI quoted the 200-path null C23 calls inadequate
    (p = 0.065) while the Inference panel below said 0.078. Both now read the
    inference summary's circular-shift test."""
    summ = json.loads((SHIPPED / "backtest" / "inference" / "summary.json").read_text())
    circ = summ["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
    text = _visible_text(pages["/research"])
    p = f"{circ['p_value']:.3f}"
    assert f"exposure-matched circular shifts at the 5% level (p = {p};" in text
    kpi = re.search(r'id="s3-timing-kpi">(.*?)</div></div>', pages["/research"], re.DOTALL).group(1)
    assert f"p = {p}" in kpi
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as fh:
        legacy = pickle.load(fh)["metrics"]["s3_null_p_value"]
    if f"{legacy:.3f}" != p:
        assert f"p = {legacy:.3f}" not in text


def test_research_t_bills_are_collateral_income_not_a_hurdle(dash, pages):
    """R3-T5: the NAV is already an excess return over T-bills (rf = 0), yet the
    KPI row called the T-bill return a 'benchmark', the equity chart drew it as
    'T-bill (collateral)' among the sleeves and the prose put S3 'below the T-bill
    return'. Each now says the T-bill income is earned on top of the NAV."""
    html = pages["/research"]
    text = _visible_text(html)
    for old in ("T-bill benchmark", "below the T-bill return", "T-bill (collateral)"):
        assert old not in html, old
    # the KPI keeps the collateral's compounded ^IRX return, read from the metrics
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as fh:
        cash = pickle.load(fh)["metrics"]["cash_cum_return"]
    kpi = re.search(r'kpi-label">T-bills on the collateral \(earned on top of the NAV\)</div>\s*'
                    r'<div class="kpi-value">(.*?)<span', html, re.DOTALL)
    assert kpi, "T-bill KPI missing"
    assert _visible_text("<body>" + kpi.group(1)).strip() == f"{cash * 100:+.1f}".replace("-", dash.MINUS)
    assert "an excess return over T-bills (the hurdle is 0)" in text
    names = [s["name"] for s in _spec(html, "chart-equity")["series"]]
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    if "nav_tbill" in eq.columns and eq["nav_tbill"].notna().any():
        assert "T-bill return (not in the NAV)" in names


def test_corrections_trend_sentence_agrees_with_the_published_series(dash, pages):
    """UI-4: 'Every successive fix made the backtest worse' sat beside a chart on
    which C22 raised the result from −26.1% to −21.6%. The sentence is now read
    off the series: it names every round that raised the result."""
    html = pages["/corrections"]
    sub = _visible_text("<body>" + re.search(r'id="audit-sub">(.*?)</p>', html, re.DOTALL).group(1))
    series = _spec(html, "chart-headline")
    vals = list(series["series"][0]["data"])
    rounds = series["xs"]
    ups = [rounds[i + 1] for i in range(len(vals) - 1)
           if vals[i] is not None and vals[i + 1] is not None and vals[i + 1] - vals[i] > 0.05]
    assert "Every successive fix made the backtest worse" not in html
    if ups:
        for r in ups:
            assert f"{r} raised it from" in sub, r
        assert not sub.startswith("No correction")
    f = dash.headline_trend_sentence
    assert f([("C17", -9.1), ("C18", -19.0), ("C21", -26.1)]).startswith("No correction raised")
    assert "C22 raised it from −26.1% to −21.6%" in f([("C17", -9.1), ("C21", -26.1), ("C22", -21.6),
                                                       ("C23", -22.2)])


def test_waterfall_exports_steps_not_bar_encodings(dash, pages):
    """UI-9: the CSV exported the stacked-bar geometry (a step that raised P&L by
    $107.9K as −107.87) and the tooltip listed every step as both 'raises' and
    'lowers'. The spec now carries its own CSV rows (Δ P&L, P&L after) and each
    step bar's tip only where that bar is drawn."""
    w = _spec(pages["/corrections"], "chart-waterfall")
    bridge = sorted(dash._read_csv_rows(SHIPPED / "backtest" / "c22_bridge.csv"), key=lambda r: int(r["row"]))
    rows = {r[1]: r for r in w["csv_rows"]["rows"]}
    assert w["csv_rows"]["columns"][:4] == ["row", "step", "d_pnl_usd", "pnl_after_usd"]
    for b in bridge:
        assert rows[b["label"]][2] == pytest.approx(float(b["d_pnl"]), abs=0.01), b["label"]
        assert rows[b["label"]][3] == pytest.approx(float(b["pnl"]), abs=0.01), b["label"]
    atm = next(b for b in bridge if "0.82" in b["label"])
    assert float(atm["d_pnl"]) > 0
    assert rows[atm["label"]][2] > 0
    up = next(s for s in w["series"] if s["name"] == "Step raises P&L")
    down = next(s for s in w["series"] if s["name"] == "Step lowers P&L")
    for i, (tu, td) in enumerate(zip(up["tip"], down["tip"])):
        assert (tu is None) or (td is None), i                    # never both at one step
        assert (tu is None) == (up["data"][i] is None), i
        assert (td is None) == (down["data"][i] is None), i
    assert w["horizontal"] is True                              # step names never collide
    assert w.get("labelShare")
    js = DESK_JS.read_text()
    assert "if (spec.csv_rows)" in js
    assert "(spec.tipRows || [])" in js


def test_waterfall_end_bar_is_named_for_the_record_it_holds(dash, pages):
    """R3-D5: the end bar read 'C22 canonical' but held the bridge's last level,
    replayed on the C23 record (today's canonical run), while the caption beside it
    quoted the C22 run as published, so one name carried two values on one page.
    The end category now names today's run on the C23 record, and the caption's
    published C22 figure comes from c23_bridge.csv instead of the template."""
    w = _spec(pages["/corrections"], "chart-waterfall")
    c22 = sorted(dash._read_csv_rows(SHIPPED / "backtest" / "c22_bridge.csv"), key=lambda r: int(r["row"]))
    c23 = sorted(dash._read_csv_rows(SHIPPED / "backtest" / "c23_bridge.csv"), key=lambda r: int(r["row"]))
    end = w["x"]["categories"][-1]
    assert "C22" not in end, end
    assert "C23" in end, end
    assert end == dash.WATERFALL_END
    assert w["xs"][-1] == end
    total = next(s for s in w["series"] if s["name"] == "Run total")
    assert total["data"][-1] == pytest.approx(float(c22[-1]["pnl"]) / 1e3, rel=1e-4)
    assert "canonical today" in w["describe"]
    note = _visible_text("<body>" + re.search(r'id="waterfall".*?<p class="note">(.*?)</p>',
                                              pages["/corrections"], re.DOTALL).group(1))
    assert f"the published C22 run ended at {dash._usd(float(c23[0]['pnl']), signed=True)}" in note


def test_surface_badge_lists_the_models(pages):
    """UI-10: the badge read 'models:' and nothing else."""
    badge = _visible_text("<body>" + re.search(r'<div class="page-badge">(.*?)</div>', pages["/surface"],
                                                re.DOTALL).group(1))
    for m in ("eSSVI", "SVI", "SVI-cal", "Heston"):
        assert re.search(rf"models:.*\b{re.escape(m)}\b", badge), (m, badge)


def test_surface_key_result_shows_its_range_caveat(dash, pages):
    """UI-11: key result 1 showed 'Range over 5 stored sessions' in the interval
    slot and dropped the note that it is not a sampling interval and that
    held-out strikes are an interpolation test (rendered only without a range)."""
    card = _card(pages["/"], "surface.html#scorecard")
    assert "Range over" in card
    assert "not a sampling interval" in card
    assert "interpolation test" in card
    assert 'class="kpi-ci kpi-range"' in pages["/"]


def test_calibration_smile_anchors_stand_apart_from_the_out_of_sample_mids(dash, pages):
    """UI-12: the 9 fit anchors were filled dots the colour of 338 hollow mids,
    and both legend swatches were the same white dot."""
    cal = pages["/calibration"]
    n = 0
    for cid in re.findall(r'id="(chart-smile-[^"]+)" data-spec', cal):
        spec = _spec(cal, cid)
        ser = {s.get("legendKey", s["name"]): s for s in spec["series"] if not s.get("strip")}
        if "Mid (fit anchor)" not in ser:
            continue
        a, o = ser["Mid (fit anchor)"], ser["Mid (out of sample)"]
        assert a["color"] != o["color"]
        assert a["symbol"] == "diamond"
        assert a["size"] > o["size"]
        assert a.get("z", 2) > 2
        items = {it["name"]: it for it in spec["legend_items"]}
        assert items["Mid (fit anchor)"]["diamond"]
        assert items["Mid (out of sample)"]["hollow"]
        n += 1
    assert n
    assert ".btn .key-swatch.hollow" in DESK_CSS.read_text()
    assert ".btn .key-swatch.diamond" in DESK_CSS.read_text()


def test_heat_focus_ring_is_two_tone(dash):
    """UI-13: the 2 px #8abbff ring drawn inside a cell was ~1.05:1 against the
    light ends of the cividis and berlin ramps. The ring is black with a white
    inner band: one of the two reaches 3:1 on any cell colour."""
    css = DESK_CSS.read_text()
    rule = re.search(r"\.heat td:focus-visible \{([^}]*)\}", css).group(1)
    assert "outline: 2px solid #000" in rule
    assert "inset 0 0 0 4px #fff" in rule
    for i in range(101):
        for bg in (dash.seq_colour(i, 0, 100), dash.div_colour(i / 50 - 1, 1.0),
                   dash.pnl_colour(i / 50 - 1, 1.0)):
            assert max(dash.contrast_ratio("#000000", bg), dash.contrast_ratio("#ffffff", bg)) >= 3.0, bg


def test_print_keeps_cell_colours_opens_drawers_and_every_tab(dash):
    """UI-7: heat-cell colours did not print (white text on white), methodology
    drawers printed closed and a capped table overlapped the chart below it."""
    css, js = DESK_CSS.read_text(), DESK_JS.read_text()
    pr = css[css.index("@media print"):]
    assert "print-color-adjust: exact" in pr
    assert ".heat td" in pr
    assert '[role="tabpanel"][hidden] { display: block !important; }' in pr
    assert ".tbl-wrap.capped { max-height: none !important;" in pr
    # every closed disclosure opens for print: the methodology drawers and (P1.8) the
    # collapsed postmortem rounds and findings of the Corrections page
    assert "$$('details:not([open])').forEach(function (d) { d.open = true; opened.push(d); });" in js
    assert "opened.forEach(function (d) { d.open = false; })" in js


def test_status_bar_freshness_never_reflows_the_page(dash, pages):
    """UI-8: '1 session behind (refreshes after the next close)' replaced 'as of
    <date>' after first paint and wrapped the status bar (CLS 0.15–0.63). The
    text fits the reserved width; the explanation moved to its title. The risk
    page's other Greek grids are hidden server-side (hiding them after paint
    shifted the page), and the smile slice is drawn once, not twice."""
    js, css = DESK_JS.read_text(), DESK_CSS.read_text()
    assert "(refreshes after the next close)'" not in js
    assert "'1 session behind'" in js
    assert "el.title =" in js
    assert ".statusbar #freshness { display: inline-block; min-width: 18ch; }" in css
    risk = pages["/risk"]
    for key in ("iv", "vega", "vanna", "delta"):
        want = rf'<div id="grid-{key}" role="tabpanel" aria-labelledby="tab-{key}" hidden>'
        assert re.search(want, risk), key
    assert '<div id="grid-vomma" role="tabpanel" aria-labelledby="tab-vomma">' in risk
    assert "else if (!Desk.isPending(el)) render(el);" in js
    assert "if (Desk.charts[e.target.id]) return;" in js


def test_captions_wrap_to_their_scroll_region(dash):
    """UI-15: a caption inside a horizontal scroll region was as wide as its table,
    so on a phone the middle of each line was off-screen."""
    css = DESK_CSS.read_text()
    assert ".tbl-wrap, .heat-wrap { container-type: inline-size; }" in css
    assert (".tbl-wrap > table > caption, .heat-wrap > table > caption { position: sticky; left: 0; "
            "max-width: 100cqw; }") in css


def test_chart_fallback_hides_chart_only_controls(dash):
    """UI-17: with ECharts blocked, the 3D button and legend toggles stayed and did
    nothing, and a fallback notice sat under the hidden 3D container."""
    js, css = DESK_JS.read_text(), DESK_CSS.read_text()
    assert "if (w.hidden || w.closest('[hidden]')) return;" in js
    assert (".charts-failed [data-load-3d], .charts-failed [data-legend-for], "
            ".charts-failed [data-zoom-for]") in css


def test_both_first_screen_fonts_are_preloaded(pages):
    """UI-18: only Inter was preloaded; canvas charts drawn before Desk Mono arrived
    kept the fallback font. Both are preloaded and early charts are redrawn once
    the fonts are ready."""
    for route, h in pages.items():
        assert ('<link rel="preload" href="static/fonts/DeskMono-400.woff2" as="font" type="font/woff2" '
                'crossorigin>') in h, route
    js = DESK_JS.read_text()
    assert "doc.fonts.ready.then(" in js
    assert "if (c.early) render(c.el);" in js


def test_no_identification_count_where_the_sigma_profile_contradicts_it(dash, pages):
    """CLM-01: '4 of 5 parameters are set by the data' / '4 of 5 effective degrees
    of freedom' beside a σ profile on which κ rides the σ cap (1.2 → 27.4). The
    pages count interior parameters and say that κ moves with the cap."""
    idx, cal = _visible_text(pages["/"]), _visible_text(pages["/calibration"])
    for t in (idx, cal):
        assert "parameters are set by the data" not in t
        assert "effective degrees of freedom" not in t
    d = dash.load_page2()
    rg = d["ridge"]
    prof = d["ident"]["sigma_profile"]
    kap = [r["params"]["kappa"] for r in sorted(prof, key=lambda r: r["sigma_cap"])]
    assert rg["kappa_lo"] == pytest.approx(kap[0])
    assert rg["kappa_top"] == pytest.approx(kap[-1])
    if rg["moves"]:
        assert "moves with the σ cap along a κ–σ ridge" in idx
        assert "moves with the σ cap along a κ–σ ridge" in cal
        assert f"κ ≤ {d['ident']['kappa_max']:g}" in cal
    assert "of 5 parameters interior (not on a bound)" in cal
    # the free-σ refit names its wider box (κ ≤ 60, not production's 20)
    assert re.search(r"Refitted in a wider box \(σ ≤ \d+, κ ≤ \d+; production has κ ≤ \d+\)", idx)


def test_overview_states_what_is_stored_not_who_stored_it(pages):
    """CLM-08: 'What the daily job has stored' listed records committed from
    development runs; the copy now describes design intent and the stored record."""
    idx = _visible_text(pages["/"])
    assert "What the daily job has stored" not in pages["/"]
    assert "Pipeline · what is stored" in idx
    assert "Each U.S. session it stores the option books" not in idx
    assert "whoever wrote it" in idx


def test_model_inventory_is_the_validation_packs(dash, pages):
    """CLM-09: the pack (19 models, M14 = surface) and this inventory (13 models,
    M10 = surface) used one ID for different models while the pack called them the
    same inventory. The inventory is canonical: every pack ID is here under the
    pack's name, and the Model Risk checks cite these IDs."""
    inv = json.loads((ROOT / "dashboard" / "model_inventory.json").read_text())
    ours = {m["id"]: m["name"] for m in inv["models"]}
    ours.update({v["id"]: v["source"] for v in inv["vendor_inputs"]})
    pack = (ROOT / "docs" / "model_validation" / "README.md").read_text()
    sec = pack[pack.index("## 1. Model inventory"):pack.index("## 2.")]
    rows = re.findall(r"^\| (M\d+|V\d+) \| ([^|]+) \|", sec, re.MULTILINE)
    assert len(rows) >= 19

    def stem(name: str) -> str:                   # the name up to its first parenthesis or colon
        return re.split(r" \(|: ", name.strip())[0].strip().strip("`")

    for mid, cell in rows:
        assert mid in ours, mid
        assert stem(ours[mid]) == stem(cell), (mid, ours[mid], cell)
    # and the other way: the pack lists every model the page carries (it lacked M20, the C7 simulator)
    assert sorted(ours) == sorted(mid for mid, _ in rows), sorted(set(ours) ^ {mid for mid, _ in rows})
    d = dash.load_model_risk()
    ids = set(ours) | {"Data"}
    for m in d["monitors"]:
        assert m["model"] in ids, m
    by = {m["check"]: m["model"] for m in d["monitors"]}
    assert by["Headline fit age (NYSE sessions)"] == "M4"                       # the joint calibrator
    assert by["Greeks surface built from the headline fit"] == "M6"
    assert by["β drift: newest chain's ATM vol / VIX against the frozen β"] == "M7"
    assert by["Market database lag behind the rule labels (sessions)"] == "M10"
    assert re.search(r'<tr id="inv-m14"><td[^>]*>M14</td><td class="wrap"[^>]*>'
                     r'<strong[^>]*>Quote-grade surface</strong>', pages["/model-risk"])


# ── C23 polish: what the screenshot review found ─────────────────────────────

def test_partial_years_are_marked_ytd_with_their_span(dash, pages):
    """STATS-5: the implied-β table headed its last column '2025' although the
    year runs only to the backtest's end, while the annual chart said '2025 YTD'.
    A year the record marks partial (benchmarks.json implied_beta.<index>.partial_years,
    the NAV's last calendar year) is headed 'YTD' wherever a year heads a column or
    a row, and a note gives the span the artifact records; full years are unmarked."""
    html = pages["/research"]
    bm = json.loads((SHIPPED / "backtest" / "inference" / "benchmarks.json").read_text())
    ib = bm["implied_beta"]
    years = [str(y) for y in ib["PUT"]["by_year"]]
    partial = {str(y): p for ix in ("PUT", "WPUT") for y, p in (ib[ix].get("partial_years") or {}).items()}
    assert partial, "the shipped benchmark window ends mid-year"
    tab = re.search(r'<table class="tbl dense" id="implied-beta">(.*?)</table>', html, re.DOTALL).group(1)
    heads = [_visible_text("<body>" + re.sub(r'<(\w+) aria-hidden="true">.*?</\1>', "", h)).strip()
             for h in re.findall(r"<th[^>]*>(.*?)</th>", tab.split("</thead>")[0], re.DOTALL)][2:]
    assert heads == [f"{y} YTD" if y in partial else y for y in years]
    unwrap = re.sub(r'<span class="nowrap">(.*?)</span>', r"\1", html)
    note = _visible_text("<body>" + re.search(r'<p class="note" id="implied-beta-note">(.*?)</p>',
                                               unwrap, re.DOTALL).group(1))
    for y, p in partial.items():
        lo, hi = p["window"].split("..")
        assert f"{y} YTD covers {lo} → {hi}" in note
    for ix in ("PUT", "WPUT"):
        for p in (ib[ix].get("partial_years") or {}).values():
            legs = f"{ix} {p['n_periods']:,} roll periods (expiries {p['first_expiry']} → {p['last_expiry']})"
            assert legs in note
    # the monthly grid's row heads and the annual chart: the NAV's partial year, with its span
    d = dash.load_page4()
    ann = {str(a["year"]): a for a in d["ann"]}
    assert any(a["partial"] for a in ann.values())
    rows = [r[0] for r in _cells(html, "monthly-grid")]
    assert rows == [f"{y} YTD" if ann[y]["partial"] else y for y in ann]
    text = _visible_text(unwrap)
    for y, a in ann.items():
        if a["partial"]:
            assert f"{y} YTD: {a['first']} → {a['last']}, {a['sessions']:,} sessions" in text
            assert a["first"][:4] == a["last"][:4] == y
            assert a["last"] == d["eq_dates"][-1]
    assert re.search(r"Per-date rate \(annual mean, cc: .*\b\d{4} YTD: \d", text)
    # the notes' dates never break at their hyphens ('2025-' / '01-17' at 390 px)
    for raw in (re.search(r'id="implied-beta-note">(.*?)</p>', html, re.DOTALL).group(1),
                re.search(r"Years compound exactly(.*?)</p>", html, re.DOTALL).group(1)):
        assert len(re.findall(r"\d{4}-\d{2}-\d{2}", raw)) == \
            len(re.findall(r'<span class="nowrap">\d{4}-\d{2}-\d{2}</span>', raw)) > 0


def _path_cells(html: str, table_id: str) -> dict:
    """{first visible text of a row's second cell or its id: [paths cited in the row]}."""
    tbl = re.search(rf'<table class="tbl[^"]*" id="{table_id}">(.*?)</table>', html, re.DOTALL)
    assert tbl, table_id
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl.group(1).split("<tbody>")[1], re.DOTALL):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)
        key = _visible_text("<body>" + cells[1]).strip()
        out[key] = [re.sub(r"<[^>]+>", "", c)
                    for c in re.findall(r'<code class="path">(.*?)</code>', tr, re.DOTALL)]
    return out


def test_artifact_citations_name_what_the_code_reads(dash, pages):
    """P5: the M7 β-drift check cited full_results.pkl while the footer cited
    full_results.parquet, with nothing saying both exist: the engine writes the
    NAV to the parquet and the run's state (β, trades, metrics) to the pickle
    (BacktestEngine.save). Each citation names every file its number is read from,
    the inventory lists both outputs of the engine, and the Quintic panel names the
    Quintic fit it reports."""
    from joint_vol_calibration.backtest import backtest_engine as be
    html = pages["/model-risk"]
    mon = _path_cells(html, "monitor-table")
    beta = next(v for k, v in mon.items() if k.startswith("β drift"))
    assert beta == ["data_store/backtest/full_results.pkl", "data_store/vol_system.db"]
    st = pickle.loads((SHIPPED / "backtest" / "full_results.pkl").read_bytes())
    assert "atm_to_vix_ratio" in st                    # the β the check compares with
    lag = next(v for k, v in mon.items() if k.startswith("Market database lag"))
    assert lag == ["data_store/signals/regime_labels.parquet", "data_store/vol_system.db"]
    assert re.search(r'<td class="wrap"[^>]*>Labels <span class="nowrap">', html)      # its dates stay whole
    rel = be.RESULTS_PATH.relative_to(ROOT).as_posix()
    inv = {k.split(" ")[0]: v for k, v in _path_cells(html, "inventory-table").items()}
    engine = next(v for k, v in inv.items() if k.startswith("Backtest"))
    assert engine[1:] == [rel, rel[:-len(".parquet")] + ".pkl"]       # module, then both outputs
    footer = re.search(r"<tr>(?:(?!</tr>).)*?Backtest window end \(fixed\)"
                       r"(?:(?!</tr>).)*?<code[^>]*>(.*?)</code>", html, re.DOTALL)
    assert re.sub(r"<[^>]+>", "", footer.group(1)) == rel
    # every path any page cites exists, or is a pattern, or is an output the inventory marks pending
    inv_models = json.loads((ROOT / "dashboard" / "model_inventory.json").read_text())["models"]
    pending = {m["outputs"] for m in inv_models if m.get("outputs_pending")}
    for route, page in pages.items():
        for c in re.findall(r'<code class="path">(.*?)</code>', page, re.DOTALL):
            p = re.sub(r"<[^>]+>", "", c).replace("&lt;", "<").replace("&gt;", ">")
            if p.startswith(("data_store/", "docs/", "dashboard/", "joint_vol_calibration/", "scripts/")) \
                    and "<" not in p and p not in pending:
                assert (ROOT / p).exists(), (route, p)
    cal = _visible_text(re.sub(r'<wbr>|<span class="nowrap">|</span>', "", pages["/calibration"]))
    q = max((SHIPPED / "calibrations").glob("quintic_cal_*.pkl")).name
    assert f"Quintic: data_store/calibrations/{q}" in cal


# ── browser checks (Chromium via Playwright; skipped where it is not installed) ──

@pytest.fixture(scope="module")
def site_browser(dash, freezer, tmp_path_factory):
    """The frozen site served on loopback and a headless Chromium."""
    sync_api = pytest.importorskip("playwright.sync_api")
    import functools
    import http.server
    import socketserver
    import threading

    out = tmp_path_factory.mktemp("site")
    assert freezer.freeze(str(out), dash) == 0

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(out)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw = sync_api.sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as exc:              # no browser binary on this machine
        pw.stop()
        httpd.shutdown()
        pytest.skip(f"Chromium unavailable: {exc}")
    yield {"base": f"http://127.0.0.1:{httpd.server_address[1]}/", "browser": browser, "site": out}
    browser.close()
    pw.stop()
    httpd.shutdown()


BROWSER_TIMEOUT_MS = 120_000          # generous: the suite shares its CPUs with other jobs


def _context(sb, width=1440, height=900):
    ctx = sb["browser"].new_context(viewport={"width": width, "height": height})
    ctx.set_default_timeout(BROWSER_TIMEOUT_MS)
    ctx.set_default_navigation_timeout(BROWSER_TIMEOUT_MS)
    return ctx


def _page(sb, width=1440, height=900):
    ctx = _context(sb, width, height)
    return ctx, ctx.new_page()


_CHART_DRAWN_JS = """([id, ms]) => new Promise((resolve, reject) => {
  const t0 = Date.now();
  (function poll() {
    if (window.Desk && Desk.charts[id] && document.fonts.status === 'loaded') return resolve(true);
    if (Date.now() - t0 > ms) return reject(new Error('chart ' + id + ' not drawn in ' + ms + ' ms'));
    setTimeout(poll, 50);
  })();
})"""


def _wait_for_chart(page, chart_id: str) -> None:
    """Wait until desk.js has drawn ``chart_id`` with the web fonts loaded.

    Not page.wait_for_function: when its predicate is not already true, Playwright
    polls it by evaluating the string in the page, which the pages' CSP
    (script-src 'self', no 'unsafe-eval') refuses, so a chart drawn a moment late
    (by the IntersectionObserver) failed the test with an EvalError. A function
    passed to page.evaluate is not evaluated from a string in the page."""
    page.evaluate(_CHART_DRAWN_JS, [chart_id, BROWSER_TIMEOUT_MS])


def test_browser_tests_never_poll_a_string_predicate():
    """FINAL-3: the waterfall and period-label tests waited with
    page.wait_for_function on a string, which fails with an EvalError under the
    pages' CSP whenever the chart is not drawn at the first check (a full-suite
    run failed so). Browser tests wait with _wait_for_chart."""
    call = "wait_for_function" + "("
    offenders = [f"{f.name}:{i}" for f in sorted(Path(__file__).parent.glob("test_*.py"))
                 for i, line in enumerate(f.read_text().splitlines(), 1) if call in line]
    assert not offenders, offenders


def _a_slice_expiry(sb) -> str:
    """An expiry of the frozen Surface page's smile tabs, read from the page: the
    daily bot replaces the session and so its expiries (a pinned date passed on
    the committed data and failed in the simulated next session)."""
    exps = re.findall(r'aria-controls="slices-panel" data-value="(\d{4}-\d{2}-\d{2})"',
                      (sb["site"] / "surface.html").read_text())
    assert len(exps) >= 2, exps
    return exps[len(exps) // 2]


def test_browser_panel_deep_links_land_on_their_panel(site_browser):
    """UI-1: desk.js rewrote '#scorecard' to '#exp=…&model=…' before the browser
    scrolled, so every panel link on Surface and Risk (the Overview's first key
    result included) landed at the top. Every '<page>#<panel>' loads at its panel,
    a plain load keeps its URL, and a copied link keeps the tab state."""
    sb = site_browser
    links = set()
    for f in sorted(sb["site"].glob("*.html")):
        h = f.read_text()
        links |= set(re.findall(r'href="([a-z\-]+\.html#[a-z][^"&]*)"', h))
        links |= {f"{f.name}#{pid}" for pid in re.findall(r'data-copy-link="([^"]+)"', h)}
    exp = _a_slice_expiry(sb)
    links |= {f"surface.html#scorecard&exp={exp}", "risk.html#scenario&greek=vega"}
    ctx, page = _page(sb)
    bad = []
    for href in sorted(links):
        page.goto(sb["base"] + href, wait_until="load")
        page.wait_for_timeout(150)
        anchor = href.split("#", 1)[1].split("&")[0]
        top = page.evaluate(f"(() => {{ const e = document.getElementById({anchor!r}); "
                            "return e ? e.getBoundingClientRect().top : null; })()")
        at_bottom = page.evaluate(
            "Math.abs(scrollY - (document.documentElement.scrollHeight - innerHeight)) < 2")
        if top is None or not (-5 <= top <= 120 or at_bottom):
            bad.append((href, top, page.url))
    assert not bad, bad
    page.goto(sb["base"] + "surface.html", wait_until="load")
    assert page.url.endswith("/surface.html")                   # a plain load keeps its URL
    page.goto(sb["base"] + f"surface.html#exp={exp}", wait_until="load")
    page.click("[data-copy-link='scorecard']")
    assert page.url.endswith(f"#scorecard&exp={exp}")
    ctx.close()


def test_browser_slice_is_drawn_once_and_tables_follow_the_tab(site_browser):
    """UI-8 / UI-17: the slice was drawn twice at boot (tab init and the
    IntersectionObserver); with ECharts blocked, the tabs still switch the
    per-expiry tables, and no notice sits under the hidden 3D view."""
    sb = site_browser
    ctx, page = _page(sb)
    page.add_init_script("""window.__so = {}; document.addEventListener('DOMContentLoaded', () => {
        const tmp = window.echarts.init(document.createElement('div'));
        const proto = Object.getPrototypeOf(tmp); const so = proto.setOption; tmp.dispose();
        proto.setOption = function () {
          const id = this.getDom().id; window.__so[id] = (window.__so[id] || 0) + 1;
          return so.apply(this, arguments); }; });""")
    page.goto(sb["base"] + "surface.html", wait_until="load")
    # the slice sits below the delta grid (P2.1), out of the first viewport: the
    # IntersectionObserver draws it when it is scrolled to
    page.locator("#chart-slices").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-slices")
    page.wait_for_timeout(200)
    n = page.evaluate("window.__so['chart-slices']")
    early = page.evaluate("Desk.charts['chart-slices'].early")
    assert n == (2 if early else 1), (n, early)
    ctx.close()
    ctx = _context(sb)
    ctx.route("**/echarts-6.1.0-desk.min.js*",
              lambda r: r.fulfill(status=200, body="/* blocked */",
                                  headers={"content-type": "text/javascript"}))
    page = ctx.new_page()
    page.goto(sb["base"] + "surface.html", wait_until="load")
    st = page.evaluate("""({failed: document.documentElement.classList.contains('charts-failed'),
        btn3d: document.querySelector('[data-load-3d]').offsetParent !== null,
        notes: [...document.querySelectorAll('.chart-fallback')].map(p => p.previousElementSibling.id)})""")
    assert st["failed"], st
    assert not st["btn3d"], st
    assert "surface3d" not in st["notes"], st
    exp = _a_slice_expiry(sb)
    page.click(f"[data-chart-filter] [data-value='{exp}']")
    shown = page.evaluate("[...document.querySelectorAll('#chart-slices-data [data-group]')]"
                          ".filter(g => !g.hidden).map(g => g.dataset.group)")
    assert shown == [exp]
    ctx.close()


def test_browser_layout_is_stable_and_fits_at_desk_and_phone_widths(site_browser):
    """UI-8 / UI-15: no layout shift after first paint; at 390 px no caption runs
    past its scroll region; at 1440 px the surface history table fits."""
    sb = site_browser
    for width, height in ((1440, 900), (390, 844)):
        ctx, page = _page(sb, width, height)
        page.add_init_script("window.__cls = 0; new PerformanceObserver(l => {"
                             " for (const e of l.getEntries())"
                             " if (!e.hadRecentInput) window.__cls += e.value; })"
                             ".observe({type: 'layout-shift', buffered: true});")
        for name in ("index.html", "surface.html", "risk.html"):
            page.goto(sb["base"] + name, wait_until="load")
            page.wait_for_timeout(400)
            assert page.evaluate("window.__cls") < 0.05, (width, name, page.evaluate("window.__cls"))
            over = page.evaluate("""
                [...document.querySelectorAll('.tbl-wrap > table > caption, .heat-wrap > table > caption')]
                .filter(c => c.offsetParent !== null && c.getBoundingClientRect().width > 0)
                .filter(c => c.getBoundingClientRect().right >
                             c.closest('.tbl-wrap, .heat-wrap').getBoundingClientRect().right + 1)
                .map(c => c.textContent.slice(0, 40))""")
            assert not over, (width, name, over)
        if width == 1440:
            page.goto(sb["base"] + "surface.html", wait_until="load")
            fits = page.evaluate("(() => { const w = document.getElementById('surface-history-table')"
                                 ".closest('.tbl-wrap'); return w.scrollWidth <= w.clientWidth + 1; })()")
            assert fits
        ctx.close()


_LINE_BREAKS_JS = """els => els.map(el => {
  // every place the rendered text starts a new line, with the characters either side
  const out = [];
  const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  let top = null, prev = '', n;
  while ((n = walk.nextNode())) {
    for (let i = 0; i < n.length; i++) {
      const r = document.createRange(); r.setStart(n, i); r.setEnd(n, i + 1);
      const rect = r.getClientRects()[0];
      if (!rect) continue;
      if (top !== null && rect.top > top + 2) out.push([el.textContent, prev, n.data[i]]);
      top = rect.top; prev = n.data[i];
    }
  }
  return out;
}).flat()"""

_PANEL_HEADS_JS = """() => Array.from(document.querySelectorAll('.panel-head')).map(h => {
  const b = h.querySelector('[data-copy-link]');
  if (!b || !b.offsetParent) return null;
  const hr = h.getBoundingClientRect(), cs = getComputedStyle(h);
  const right = hr.right - parseFloat(cs.paddingRight) - parseFloat(cs.borderRightWidth);
  const gap = right - b.getBoundingClientRect().right;
  return gap > 2 ? h.parentElement.id + ' ' + gap.toFixed(0) + 'px' : null;
}).filter(Boolean)"""


def _between_names(prev: str, nxt: str) -> bool:
    """A line break the path filter allows: after '/', '_' or '::', or between
    the words of a CamelCase name (a lower-case letter or digit, then a capital)."""
    return prev in "/_:" or (bool(re.fullmatch(r"[a-z0-9]", prev)) and nxt.isupper())


def test_path_filter_breaks_between_camel_case_words(dash):
    """P8: at 320 px Corrections broke 'TestGenerateSignal1Contrarian' inside a
    word; the path filter offers a break between its words as well as after
    '/', '_' and '::'."""
    out = str(dash.path_breaks("test_signal_engine.py::TestGenerateSignal1Contrarian::test_r2_exit"))
    assert "::<wbr>Test<wbr>Generate<wbr>Signal1<wbr>Contrarian::<wbr>test_<wbr>r2_<wbr>exit" in out
    assert str(dash.path_breaks("data_store/vol_system.db")) == "data_<wbr>store/<wbr>vol_<wbr>system.db"
    assert str(dash.path_breaks("README.md")) == "README.md"


def test_browser_paths_and_panel_links_at_phone_width(site_browser):
    """C23 verify, at 390 and 320 px: a path in the freshness table or the Model
    Risk columns, or a test ID on Corrections, wraps only after '/', '_' or '::'
    or between the words of a CamelCase test class (they broke as 'vol_syste' /
    'm.db', 'test_surf' / 'ace_svi.py' and, at 320 px, inside
    'TestGenerateSignal1Contrarian'), and at 390 px every panel's '#' link sits at
    the right of its header (it dropped to the left of a second line alone when
    the title and its date filled the first)."""
    ctx, page = _page(site_browser, 390, 844)
    try:
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 844})
            for name in ("index", "surface", "calibration", "risk", "research", "corrections", "model-risk"):
                page.goto(site_browser["base"] + f"{name}.html", wait_until="load")
                codes = page.locator("code.path")
                assert codes.count(), name
                breaks = page.evaluate(_LINE_BREAKS_JS, codes.element_handles())
                assert [b for b in breaks if not _between_names(b[1], b[2])] == [], (width, name, breaks)
                if width == 390:
                    assert page.evaluate(_PANEL_HEADS_JS) == [], name
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(site_browser["base"] + "model-risk.html", wait_until="load")
        breaks = page.evaluate(_LINE_BREAKS_JS, page.locator("#monitor-table code.path").element_handles())
        assert breaks, "the monitor table's paths do wrap at 390 px"        # the check above saw breaks
    finally:
        ctx.close()


def test_browser_waterfall_csv_and_print(site_browser):
    """UI-9 / UI-7: the downloaded waterfall CSV has the steps' signed Δ P&L; a
    print opens every methodology drawer and restores them afterwards."""
    sb = site_browser
    ctx, page = _page(sb)
    page.goto(sb["base"] + "corrections.html", wait_until="load")
    csv = page.evaluate(
        "Desk.specCsv(JSON.parse(document.getElementById('chart-waterfall-spec').textContent))")
    atm = next(line for line in csv.splitlines() if "0.82" in line)
    assert float(next(c for c in atm.split(",")[2:] if c)) > 0, atm
    page.goto(sb["base"] + "risk.html", wait_until="load")
    page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
    assert page.evaluate("[...document.querySelectorAll('details.method')].every(d => d.open)")
    assert page.evaluate("[...document.querySelectorAll('[role=tabpanel]')].every(p => !p.hidden)")
    page.evaluate("window.dispatchEvent(new Event('afterprint'))")
    assert page.evaluate("[...document.querySelectorAll('details.method')].every(d => !d.open)")
    ctx.close()


def test_long_daily_series_are_downsampled_for_drawing(pages):
    """UI-8: the Research page's daily lines (~1,800 points each, six on the equity
    chart) were drawn point by point; they are drawn with LTTB downsampling (the
    CSV and the table keep every point)."""
    for cid in ("chart-equity", "chart-dd", "chart-rolling"):
        spec = _spec(pages["/research"], cid)
        assert all(s.get("sampling") == "lttb" for s in spec["series"]), cid
        assert len(spec["series"][0]["data"]) == len(spec["xs"]) > 1000, cid
    assert "sampling: s.sampling || undefined" in DESK_JS.read_text()


# the cells that name a model or a strategy, in every table that lists them
_NAME_CELLS = {
    "surface.html": ["#scorecard-table tbody td:first-child"],
    "calibration.html": ["#spx-only tbody td:first-child"],
    "research.html": ["#sleeve-table tbody td:first-child", "#bootstrap-table tbody td:first-child",
                      "#null-table tbody td:first-child", "#bench-table tbody td:first-child",
                      "#forecast-table tbody td:first-child"],
    "model-risk.html": ["#monitor-table tbody td:first-child .faint",
                        "#inventory-table tbody td:nth-child(2) strong"],
}
_ONE_LINE_JS = """sel => [...document.querySelectorAll(sel)].map(el => {
  const r = document.createRange(); r.selectNodeContents(el);
  const w = el.closest('.tbl-wrap');
  return [el.textContent.trim(), new Set([...r.getClientRects()].map(q => Math.round(q.top))).size,
          w.scrollWidth <= w.clientWidth + 1];
})"""


def test_browser_model_and_strategy_names_stay_on_one_line_at_desk_width(site_browser):
    """P2: at 1440 px the Surface scorecard wrapped 'SVI, calendar-ordered' onto
    three lines and 'Heston (v2 fit)' onto two. Every table that names models or
    strategies keeps each name on one line at desk width, and still fits its
    region (no sideways scroll traded for it); at 390 px they scroll as before."""
    ctx, page = _page(site_browser, 1440, 900)
    try:
        for name, sels in _NAME_CELLS.items():
            page.goto(site_browser["base"] + name, wait_until="load")
            for sel in sels:
                cells = page.evaluate(_ONE_LINE_JS, sel)
                assert cells, (name, sel)
                assert [c[0] for c in cells if c[1] != 1] == [], (name, sel)
                assert all(c[2] for c in cells), (name, sel, "the table no longer fits its region")
    finally:
        ctx.close()


# text-heavy tables that become cards in a narrow region
_CARD_TABLES = {"model-risk.html": ["monitor-table", "inventory-table", "vendor-table"],
                "calibration.html": ["quintic-vs-heston"],
                "research.html": ["improvements-table", "sleeve-spec-table"],     # P3.6: the sleeve spec too
                "corrections.html": ["c23-bridge"]}
_CARDS_JS = """id => {
  const t = document.getElementById(id), w = t.closest('.tbl-wrap');
  const heads = [...t.querySelectorAll('thead th')].map(h => h.textContent.trim().toLowerCase());
  const rows = [...t.querySelectorAll('tbody tr')];
  return {display: getComputedStyle(t).display, fits: w.scrollWidth <= w.clientWidth + 1,
    role: t.getAttribute('role'), n_cells: t.querySelectorAll('tbody td').length,
    labels_ok: rows.every(r => [...r.children].every((c, i) =>
      (c.getAttribute('data-label') || '').toLowerCase() === heads[i]))};
}"""


def _ax_count(cdp, element_id: str, role: str) -> int:
    """Nodes of ``role`` under #element_id in Chromium's own accessibility tree."""
    root = cdp.send("DOM.getDocument")["root"]["nodeId"]
    node = cdp.send("DOM.querySelector", {"nodeId": root, "selector": f"#{element_id}"})["nodeId"]
    return len(cdp.send("Accessibility.queryAXTree", {"nodeId": node, "role": role})["nodes"])


def test_browser_text_tables_become_labelled_cards_on_a_phone(site_browser):
    """P3: at 390 and 320 px the Model Risk monitoring and inventory tables and the
    Quintic table were rows scrolled sideways, several screens tall, with large
    empty areas. In a region narrower than 640 px each text-heavy table is a stack
    of cards, one per row, every field labelled with its column's header; nothing
    scrolls sideways, and the table keeps its semantics (explicit roles, and in
    Chromium's accessibility tree a column header per column and a cell per
    cell). At 1440 px the wide-region tables stay tables."""
    sb = site_browser
    for width in (390, 320, 1440):
        ctx, page = _page(sb, width, 844)
        cdp = ctx.new_cdp_session(page)
        try:
            for name, ids in _CARD_TABLES.items():
                page.goto(sb["base"] + name, wait_until="load")
                page.evaluate("document.querySelectorAll('details').forEach(d => { d.open = true; })")
                for tid in ids:
                    st = page.evaluate(_CARDS_JS, tid)
                    assert st["labels_ok"], (width, tid)
                    assert st["fits"], (width, tid, st)
                    assert st["role"] == "table", (width, tid)
                    if width < 640:
                        assert st["display"] == "block", (width, tid, st)
                    elif tid in ("monitor-table", "inventory-table", "quintic-vs-heston", "c23-bridge"):
                        assert st["display"] == "table", (width, tid, st)
                    if width == 390:
                        heads = page.evaluate(f"document.querySelectorAll('#{tid} thead th').length")
                        assert _ax_count(cdp, tid, "columnheader") == heads, (tid, "column headers")
                        assert _ax_count(cdp, tid, "cell") == st["n_cells"], (tid, "cells")
        finally:
            ctx.close()


_CARD_CAPTIONS_JS = """() => [...document.querySelectorAll('.tbl.cards > caption')].map(c => {
  const t = c.closest('table'), r = c.getBoundingClientRect();
  const lh = parseFloat(getComputedStyle(c).lineHeight);
  return {id: t.id, ratio: r.width / t.getBoundingClientRect().width, lines: Math.round(r.height / lh)};
})"""


def test_browser_card_table_captions_span_their_table(site_browser):
    """FINAL-2: at 390 and 320 px the Model Risk vendor table's caption ('Vendor
    and third-party inputs (the IDs the data checks above carry)') stood one word
    to a line in a 40 px column beside its cards: a table caption inside a table
    displayed as a block is boxed in an anonymous table of min-content width. In
    the card layout a caption is a block as wide as its cards, at every width."""
    sb = site_browser
    seen = 0
    for width in (390, 320, 1440):
        ctx, page = _page(sb, width, 844)
        try:
            for name in _CARD_TABLES:
                page.goto(sb["base"] + name, wait_until="load")
                page.evaluate("document.querySelectorAll('details').forEach(d => { d.open = true; })")
                for cap in page.evaluate(_CARD_CAPTIONS_JS):
                    seen += 1
                    assert cap["ratio"] >= 0.95, (width, cap)
                    assert cap["lines"] <= 3, (width, cap)
        finally:
            ctx.close()
    assert seen                                  # at least one card table carries a caption


_WATERFALL_LABELS_JS = """() => {
  const c = Desk.charts['chart-waterfall'], L = c.labels, spec = c.spec;
  const ctx = document.createElement('canvas').getContext('2d'); ctx.font = L.font;
  return {cats: spec.x.categories, short: spec.labelsShort || [], lines: L.lines, used: L.short,
          maxLines: L.maxLines, band: L.band, lineHeight: L.lineHeight, width: L.width,
          widest: Math.max(...L.lines.flat().map(x => ctx.measureText(x).width)),
          overflow: c.chart.getOption().yAxis[0].axisLabel.overflow,
          gridLeft: c.chart.getOption().grid[0].left};
}"""


def test_browser_waterfall_step_names_are_never_truncated(site_browser):
    """P4: the C22 waterfall cut its step names with an ellipsis ('PDV spread:
    21-day RMS-vol forecast (daily pur...') at every width. desk.js now wraps each
    name into the lines its band holds; where the full name would need more, the
    concise name (without its trailing parenthetical) is drawn and the tooltip and
    table keep the full one. At 1440 px every name is drawn in full; at no width
    is a name cut, wider than its column or taller than its band."""
    sb = site_browser
    for width in (1440, 1024, 390, 320):
        ctx, page = _page(sb, width, 900)
        try:
            page.goto(sb["base"] + "corrections.html", wait_until="load")
            _wait_for_chart(page, "chart-waterfall")
            page.wait_for_timeout(200)
            st = page.evaluate(_WATERFALL_LABELS_JS)
            assert st["overflow"] != "truncate", width
            for i, (cat, lines) in enumerate(zip(st["cats"], st["lines"])):
                drawn = " ".join(lines)
                assert drawn == (st["short"][i] if i in st["used"] else cat), (width, cat, lines)
                assert not re.search(r"…|\.\.\.", drawn), (width, drawn)
                assert len(lines) <= st["maxLines"], (width, cat, lines)
            assert st["maxLines"] * st["lineHeight"] <= st["band"] + 0.5, (width, st)
            assert st["widest"] <= st["width"] + 1, (width, st["widest"], st["width"])
            assert st["gridLeft"] >= st["width"], width
            if width == 1440:
                assert st["used"] == [], st
            table = page.evaluate("[...document.querySelectorAll("
                                  "'#chart-waterfall-data tbody td:nth-child(2)')].map(td => td.textContent)")
            assert set(st["cats"][1:-1]) <= set(table)          # the table keeps every full name
        finally:
            ctx.close()


_X_LABELS_JS = """(id) => {
  // the axis labels ECharts draws: zrender Text elements not hidden as overlaps
  const c = Desk.charts[id], cats = c.spec.x.categories.map(String), drawn = [];
  const walk = (el, hidden) => {
    hidden = hidden || el.ignore || el.invisible;
    const t = el.type === 'text' && !hidden ? String(el.style.text) : null;
    if (t !== null && cats.includes(t.split('\\n').join(' '))) {
      const r = el.getBoundingRect().clone(); if (el.transform) r.applyTransform(el.transform);
      drawn.push({text: t, x: r.x, w: r.width});
    }
    if (el.childrenRef) el.childrenRef().forEach(ch => walk(ch, hidden));
  };
  c.chart.getZr().storage.getRoots().forEach(r => walk(r, false));
  return {cats: cats, drawn: drawn.sort((a, b) => a.x - b.x), width: c.el.clientWidth};
}"""


def test_browser_newest_period_of_the_annual_and_half_year_bars_is_labelled(site_browser):
    """FINAL-1: at 390 px the annual-return chart drew no label under its last
    bar, the partial year the P1 note marks '2025 YTD': the label was wider than
    its band, so ECharts hid it as an overlap (at 320 px every other year went
    too, and on the half-year chart the newest half-year was never labelled). A
    label wider than its band now wraps at its spaces, and where labels must be
    thinned they are thinned counting back from the newest, so the last bar is
    always labelled in full, with no label overlapping another or the edge."""
    sb = site_browser
    for width in (1440, 390, 320):
        ctx, page = _page(sb, width, 900)
        try:
            page.goto(sb["base"] + "research.html", wait_until="load")
            for cid in ("chart-annual", "chart-halfyear"):
                page.evaluate(f"document.getElementById('{cid}').scrollIntoView()")
                _wait_for_chart(page, cid)
                page.wait_for_timeout(200)
                st = page.evaluate(_X_LABELS_JS, cid)
                texts = [d["text"].replace("\n", " ") for d in st["drawn"]]
                assert texts, (width, cid)
                assert texts[-1] == st["cats"][-1], (width, cid, texts)
                for a, b in zip(st["drawn"], st["drawn"][1:]):
                    assert a["x"] + a["w"] <= b["x"], (width, cid, a, b)
                first, last = st["drawn"][0], st["drawn"][-1]
                assert first["x"] >= 0, (width, cid, first)
                assert last["x"] + last["w"] <= st["width"], (width, cid, last)
                if width == 1440 and cid == "chart-annual":
                    assert texts == st["cats"], texts                  # room for every year
        finally:
            ctx.close()


def test_waterfall_concise_names_drop_only_a_trailing_parenthetical(dash):
    assert dash.concise_label("R2 gate = upstream default (adaptive (rolling 80th pct))") == \
        "R2 gate = upstream default"
    assert dash.concise_label("PDV spread: 21-day RMS-vol forecast (daily purged refit)") == \
        "PDV spread: 21-day RMS-vol forecast"
    assert dash.concise_label("VIX tenors 9/30/93/184d, 365-day year") == \
        "VIX tenors 9/30/93/184d, 365-day year"
    assert dash.concise_label("(only a parenthetical)") == "(only a parenthetical)"


def test_forward_panel_states_its_protocol_while_awaiting_the_first_entry(dash, no_art):
    """P6: with no ledger registered the forward panel was a bare 'artifact not
    yet generated' above a blank block, and the Overview said the first entry
    comes 'with the next accepted fit', which the job does not wait for (it
    records every session with a live captured book, whatever the fit does). The
    panel now states the protocol from forward_ledger.py and option_chain.py and
    the one status sentence the Overview's pipeline row also shows; no entry is
    made up."""
    from joint_vol_calibration.backtest import forward_ledger as fl
    from joint_vol_calibration.data import option_chain as oc
    _, res = _get(dash, "/research")
    _, idx = _get(dash, "/")
    panel = re.search(r'<section class="panel[^"]*" id="forward".*?</section>', res, re.DOTALL).group(0)
    text = _visible_text("<body>" + re.sub(r"<wbr>|</?span[^>]*>", "", panel))
    fp = dash.forward_ledger_protocol()
    pending = fp["pending"]
    assert "Awaiting first entry" in text
    assert pending[:1].upper() + pending[1:] + "." in text
    assert "Artifact not yet generated" not in text
    assert "forward-table" not in panel
    assert pending in _visible_text(idx)                     # the Overview's pipeline row says the same
    assert "next accepted fit" not in res + idx
    for k in ("Records", "Prices", "Hash chain", "Marks", "Invalidated by"):
        assert f" {k} " in text, k
    for fact in (f"nearest {fl.TARGET_DAYS} calendar days out (at least {fl.MIN_DAYS})",
                 f"${fl.NAV_PER_SLEEVE:,.0f} of sleeve NAV",
                 f"${fl.COMMISSION_PER_LEG:.2f} per contract per leg",
                 f"schema {fl.SCHEMA}", f"{oc.SESSION_CLOSE_ET:%H:%M} ET close",
                 f"at least {oc.MIN_TWO_SIDED_SHARE:.0%} of near-the-money lines two-sided",
                 "data_store/forward/ledger_<session>.json"):
        assert fact in text, fact
    assert len(fp["rows"]) == 5


def test_carry_panel_lists_the_anchors_and_carry_the_fit_priced(dash, pages):
    """P7: the Carry and inputs panel left a large blank beside the model smile.
    It now lists, per expiry, the anchors the headline fit priced and the parity
    rate and carry it priced them with, read from the fit's anchors_by_expiry, and
    the anchor and expiry rules the fit records."""
    d = dash.load_page2()
    with open(SHIPPED / "calibrations" / d["calib_file"], "rb") as f:
        cal = pickle.load(f)
    anchors = cal["anchors_by_expiry"]
    html = pages["/calibration"]
    body = re.search(r'id="anchors-table">(.*?)</table>', html, re.DOTALL).group(1).split("<tbody>")[1]
    rows = [[_visible_text("<body>" + c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL)]
            for tr in re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)]
    assert len(rows) == len(anchors)
    for row, a in zip(rows, anchors):
        assert row[0] == str(a["expiry"])[:10]
        assert row[2:5] == [str(a["n"]), str(a["n_puts"]), str(a["n_wing_puts"])]
        assert row[6] == dash._fmt(a["r"] * 100, ".2f") + "%"
        assert row[7] == dash._fmt(a["q"] * 100, ".2f") + "%"
    assert sum(a["n"] for a in anchors) == cal["n_spx_options"]       # every anchor the fit priced
    text = _visible_text(html)
    assert "Anchor rule: " + dash.prose(cal["methods"]["anchors"]) in text


_BLANK_JS = """id => {
  const p = document.getElementById(id), b = p.querySelector('.panel-body');
  const kids = [...b.querySelectorAll(':scope > *, :scope > * > *')]
    .filter(e => e.offsetParent && e.getBoundingClientRect().height);
  const bottom = Math.max(...kids.map(e => e.getBoundingClientRect().bottom));
  const r = p.getBoundingClientRect();
  return (r.bottom - bottom) / r.height;
}"""


def test_browser_panels_beside_taller_ones_carry_no_large_blank(site_browser):
    """P6 / P7: at 1440 px the forward panel (awaiting its first entry) and the
    Carry and inputs panel each stood beside a taller panel with most of their
    height blank. Neither leaves more than a third of its height empty."""
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "calibration.html", wait_until="load")
        assert page.evaluate(_BLANK_JS, "carry") < 0.34
        if not (SHIPPED / "forward").is_dir():                   # the panel awaits the first entry
            page.goto(site_browser["base"] + "research.html", wait_until="load")
            assert page.evaluate(_BLANK_JS, "forward") < 0.34
    finally:
        ctx.close()


_OPEN_ALL_JS = """() => { document.querySelectorAll('.g details').forEach(d => { d.open = true; }); }"""
_PANEL_FIT_JS = """id => { const p = document.getElementById(id);
  let last = -Infinity;
  p.querySelectorAll('.panel-body *').forEach(el => { const b = el.getBoundingClientRect();
    if (b.width > 0 && b.height > 0 && !el.closest('details:not([open]) > :not(summary)'))
      last = Math.max(last, b.bottom); });
  const charts = [...p.querySelectorAll('.chart.grow')]
    .map(c => Math.round(c.getBoundingClientRect().height));
  return {blank: Math.round(p.getBoundingClientRect().bottom - last), charts: charts}; }"""


def test_browser_an_open_disclosure_stretches_no_chart_and_leaves_no_blank_beside_it(site_browser):
    """F3: opening a <details> made its panel taller than the row; the panel
    beside it stretched with it: at 1440 px 'Why the edge disappeared' ended
    1,250 px below its text beside the open change log, and the implied-β chart
    grew to 1,359 px for a 0.55–0.95 range beside the open inference notes,
    exaggerating its year-to-year swings. A row with an open disclosure now lets
    each panel end with its own content, and no grow chart exceeds 420 px."""
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "research.html", wait_until="load")
        page.evaluate(_OPEN_ALL_JS)
        page.locator("#chart-implied-beta").scroll_into_view_if_needed()
        _wait_for_chart(page, "chart-implied-beta")
        page.wait_for_timeout(300)
        for pid in ("why", "limits", "benchmarks", "inference"):
            fit = page.evaluate(_PANEL_FIT_JS, pid)
            assert fit["blank"] <= 96, (pid, fit)
            assert all(h <= 420 for h in fit["charts"]), (pid, fit)
        # the Surface's variance-swap strip beside its open replication notes (F9: 1,250 px)
        page.goto(site_browser["base"] + "surface.html", wait_until="load")
        page.evaluate(_OPEN_ALL_JS)
        rows = page.evaluate("""() => [...document.querySelectorAll('.g.pair, .g.level, .g.trio')]
          .filter(g => g.querySelector('details[open]'))
          .flatMap(g => [...g.children].filter(p => p.classList.contains('panel')).map(p => p.id))""")
        assert "varswap" in rows, rows
        for pid in rows:
            page.locator("#" + pid).scroll_into_view_if_needed()
            page.wait_for_timeout(150)
            fit = page.evaluate(_PANEL_FIT_JS, pid)
            assert fit["blank"] <= 96, (pid, fit)
            assert all(h <= 420 for h in fit["charts"]), (pid, fit)
    finally:
        ctx.close()


_ELLIPSIS_CUT_JS = """() => [...document.querySelectorAll('body *')].filter(el => {
  const s = getComputedStyle(el);
  if (s.textOverflow !== 'ellipsis' || s.overflowX === 'visible' || el.closest('[hidden]')) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && el.scrollWidth > el.clientWidth + 1 && !el.getAttribute('title');
}).map(el => el.className + ': ' + el.textContent.trim())"""


def test_browser_no_label_is_cut_short_with_an_ellipsis(site_browser):
    """F5: a .kpi-label was one line with text-overflow: ellipsis and no title,
    so 'SPX RMSE in sample' read 'SPX RMSE IN SA…' and 'VVIX model / market'
    'VVIX MODEL / MA…' on the Overview at 1440 and 768 px (and eleven more
    tiles at 768 or 1024 px on Surface, Calibration and Risk). The page did
    not scroll, so the overflow gate never saw it; the label now wraps."""
    for width in (768, 1024, 1440):
        ctx, page = _page(site_browser, width, 900)
        try:
            for name in ("index.html", "surface.html", "calibration.html", "risk.html"):
                page.goto(site_browser["base"] + name, wait_until="load")
                page.evaluate("() => document.fonts.ready.then(() => true)")
                assert page.evaluate(_ELLIPSIS_CUT_JS) == [], (name, width)
        finally:
            ctx.close()


def test_browser_every_trade_is_reachable_from_the_keyboard(site_browser):
    """F4: the per-trade strip ('Every trade · P&L by entry date') was a canvas
    with no focusable element, and its Table summarises by sleeve and year, so
    one trade's P&L and entry date were reachable only by mouse. A focusable
    readout under it steps through the shown trades by entry date with the arrow
    keys (Home / End, Page Up / Down by 10), says the sleeve, the entry date and
    the P&L, and shows the tooltip at that dot; a sleeve toggled off in the
    legend is skipped."""
    sb = site_browser
    raw = _raw_specs((sb["site"] / "research.html").read_text())["chart-round-trips"]
    pts = sorted((str(d[0]), float(d[1]), s_["name"]) for s_ in raw["series"] for d in s_["data"])
    ctx, page = _page(sb, 1440, 900)
    try:
        page.goto(sb["base"] + "research.html", wait_until="load")
        page.locator("#chart-round-trips").scroll_into_view_if_needed()
        _wait_for_chart(page, "chart-round-trips")
        r = page.locator("#chart-round-trips-readout")
        assert r.get_attribute("tabindex") == "0"
        r.focus()
        page.keyboard.press("Home")
        first = r.inner_text()
        x0, y0, n0 = pts[0]
        assert n0 in first, first
        assert x0[:10] in first, first
        assert dash_num(y0) in first, (first, y0)
        assert f"1 of {len(pts)}" in first, first
        page.keyboard.press("ArrowRight")
        assert f"2 of {len(pts)}" in r.inner_text()
        page.keyboard.press("End")
        last = r.inner_text()
        assert pts[-1][2] in last, last
        assert pts[-1][0][:10] in last, last
        tip = page.evaluate("""() => [...document.querySelectorAll('#chart-round-trips > div')]
            .some(d => d.style.position === 'absolute' && getComputedStyle(d).display !== 'none'
                  && d.innerText.trim())""")
        assert tip, "no tooltip at the stepped trade"
        # a sleeve toggled off is skipped
        s3 = next(n for _x, _y, n in pts if n.startswith("S3"))
        page.click(f"[data-legend-for~='chart-round-trips'][data-series='{s3}']")
        r.focus()
        page.keyboard.press("Home")
        left = sum(1 for _x, _y, n in pts if n != s3)
        assert f"1 of {left}" in r.inner_text(), r.inner_text()
    finally:
        ctx.close()


def dash_num(v: float) -> str:
    """A $000s P&L as the readout prints it: one decimal, sign, Unicode minus."""
    return ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):,.1f}"


# ── P1 trust pass ─────────────────────────────────────────────────────────────

_SIGNED_FILLS_JS = """cid => {
  const c = Desk.charts[cid], root = getComputedStyle(document.documentElement);
  const pos = root.getPropertyValue('--pos-fill').trim(), neg = root.getPropertyValue('--neg-fill').trim();
  const out = [];
  (c.spec.series || []).forEach((s, i) => {
    if (!s.signed) return;
    const data = c.chart.getModel().getSeriesByIndex(i).getData();
    for (let j = 0; j < data.count(); j++) {
      const raw = s.data[j], v = Array.isArray(raw) ? raw[1] : raw;
      if (v === null || v === undefined) continue;
      out.push([String((c.spec.xs || [])[j]), v, data.getItemVisual(j, 'style').fill, v < 0 ? neg : pos]);
    }
  });
  return out;
}"""


_SIGNED_CHARTS_JS = """() => Array.from(document.querySelectorAll('.chart[data-spec]')).filter(el =>
  (JSON.parse(document.getElementById(el.getAttribute('data-spec')).textContent).series || [])
    .some(s => s.signed) && !el.closest('[hidden]')).map(el => el.id)"""
_TONED_KPIS_JS = """() => Array.from(document.querySelectorAll('.kpi-value.pos, .kpi-value.neg')).map(el =>
  [el.classList.contains('neg') ? 'neg' : 'pos', getComputedStyle(el).color, el.textContent.trim()])"""
_DIALOG_BOX_JS = """() => { const d = document.getElementById('keysDialog'), r = d.getBoundingClientRect();
  return [d.open, r.left, r.width, document.documentElement.clientWidth]; }"""


def test_browser_signed_bars_take_the_colour_of_their_sign(site_browser):
    """P1.1: desk.js read a signed bar's value as p.value[1], which on a
    horizontal chart (points swapped to [value, category]) is the category name,
    so the sleeve chart drew S1C and S4, both losses, in the gain colour. Every
    bar of every signed series is drawn in the loss fill below zero and the gain
    fill otherwise (the chart's own spec gives the values)."""
    sb = site_browser
    ctx, page = _page(sb)
    seen = {}
    try:
        for name in ("research.html", "risk.html"):
            page.goto(sb["base"] + name, wait_until="load")
            ids = page.evaluate(_SIGNED_CHARTS_JS)
            for cid in ids:
                page.evaluate(f"document.getElementById('{cid}').scrollIntoView()")
                _wait_for_chart(page, cid)
                bars = page.evaluate(_SIGNED_FILLS_JS, cid)
                assert bars, cid
                for label, v, fill, want in bars:
                    assert fill.lower() == want.lower(), (name, cid, label, v, fill, want)
                seen[cid] = bars
    finally:
        ctx.close()
    assert "chart-sleeves" in seen, sorted(seen)
    assert "chart-annual" in seen, sorted(seen)
    sleeves = seen["chart-sleeves"]
    assert any(v < 0 for _, v, *_ in sleeves)
    assert any(v >= 0 for _, v, *_ in sleeves)


def test_browser_kpi_tones_and_the_shortcut_dialog(site_browser):
    """P1.1: .kpi-value (colour --text) came after .pos / .neg at the same
    specificity, so no signed KPI showed its tone; and the universal margin
    reset opened the '?' dialog in the top-left corner instead of centred."""
    sb = site_browser
    tok = _tokens()

    def rgb(h: str) -> str:
        return f"rgb({int(h[1:3], 16)}, {int(h[3:5], 16)}, {int(h[5:7], 16)})"
    want = {"pos": rgb(tok["pos"]), "neg": rgb(tok["neg"])}
    ctx, page = _page(sb)
    tiles = []
    try:
        for name in ("index.html", "risk.html", "research.html"):
            page.goto(sb["base"] + name, wait_until="load")
            tiles += [(name, *t) for t in page.evaluate(_TONED_KPIS_JS)]
    finally:
        ctx.close()
    assert {t[1] for t in tiles} == {"pos", "neg"}, tiles
    for name, cls, colour, text in tiles:
        assert colour == want[cls], (name, cls, colour, text)
        assert text[0] in "+−", (name, text)                # the sign glyph stays: colour is not the only cue
    for width, height in ((1440, 900), (390, 844)):
        ctx, page = _page(sb, width, height)
        try:
            page.goto(sb["base"] + "surface.html", wait_until="load")
            page.keyboard.press("Shift+Slash")
            box = page.evaluate(_DIALOG_BOX_JS)
            opened, left, w, vw = box
            assert opened, width
            assert abs(left - (vw - w) / 2) <= 2, (width, box)
        finally:
            ctx.close()


def _table_rows(html: str, table_id: str) -> list[list[str]]:
    """Visible cell texts of a .tbl table's body rows (chips and sr-only spans dropped)."""
    tbl = re.search(rf'<table class="tbl[^"]*" id="{re.escape(table_id)}"[^>]*>(.*?)</table>', html,
                    re.DOTALL)
    assert tbl, table_id
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", tbl.group(1).split("<tbody>")[1], re.DOTALL):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.DOTALL)
        out.append([_html_unescape(re.sub(r"<span class=\"(chip|sr-only)[^\"]*\">.*?</span>|<[^>]+>", "", c))
                    .strip() for c in cells])
    return out


def _diag_files(root: Path) -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(root.glob("*.json"))
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.stem)}


def test_sigma_profile_lists_model_vvix_against_the_market(dash, art):
    """P1.2: the σ-profile table showed what each cap does to the SPX legs but
    not to VVIX, the joint fit's other side. It now lists the model VVIX and
    VVIX − market per refit, with the ^VVIX close in its caption (the fixture's
    2026-10-16 profile: 226 … 113 against 110)."""
    resp, html = _get(dash, "/calibration")
    assert _errors(resp) == 0
    rows = json.loads((FIX / "diagnostics" / f"{SESSION}.json").read_text())["sigma_profile"]["rows"]
    cells = _table_rows(html, "sigma-profile")
    assert len(cells) == len(rows) == 8
    assert [c[8] for c in cells] == [f"{r['vvix_model']:.0f}" for r in rows]
    assert [c[8] for c in cells] == ["226", "221", "205", "176", "153", "123", "113", "113"]
    assert [c[9] for c in cells] == [dash.signed(r["vvix_model"] - r["vvix_market"], ".0f") for r in rows]
    cap = re.search(r'id="sigma-profile"><caption>(.*?)</caption>', html, re.DOTALL).group(1)
    assert f"^VVIX close of {rows[0]['vvix_market']:.0f}" in cap
    assert f"^VVIX close of {rows[0]['vvix_market']:.0f}" == "^VVIX close of 110"


def _expected_tradeoff(files: dict) -> dict:
    """The trade-off recomputed from the diagnostics files, independently of app.py."""
    out: dict[str, list] = {"far": [], "esc": []}
    for s, d in sorted(files.items()):
        rows = sorted(d["sigma_profile"]["rows"], key=lambda r: r["sigma_cap"])
        cap = d["sigma_profile"]["production_box"]["sigma"][1]
        prod = next(r for r in rows if r["sigma_cap"] == cap)
        top = rows[-1]
        key = "far" if top["vvix_model"] > 1.1 * top["vvix_market"] else "esc"
        out[key].append((s, prod, top))
    return out


def test_vvix_tradeoff_sentence_is_computed_from_every_sigma_profile(dash, pages):
    """P1.2: the Calibration page said, unconditionally, that Heston 'needs a
    vol-of-vol that prices VVIX far above the market', which the σ ≤ 10 refit of
    2026-10-16 contradicts (VVIX 112.7 against 109.9). The sentence is generated
    from every diagnosed session's σ profile; each number in it is recomputed
    here from the files, the count included."""
    html = pages["/calibration"]
    text = _visible_text(html)
    assert "Heston needs a vol-of-vol that prices VVIX far above the market" not in text
    sent = re.search(r'id="vvix-tradeoff">(.*?)</p>', html, re.DOTALL)
    assert sent
    sent = _html_unescape(re.sub(r"<[^>]+>", "", sent.group(1)))
    files = _diag_files(SHIPPED / "calibrations" / "diagnostics")
    exp = _expected_tradeoff(files)
    n, far, esc = len(files), exp["far"], exp["esc"]
    assert n >= 1
    if far:
        lead = "all " + str(n) if len(far) == n else f"{len(far)} of {n}"
        assert f"On {lead if n > 1 else 'the one'} diagnosed session" in sent
        pairs = ", ".join(f"{t['vvix_model']:.0f} vs {t['vvix_market']:.0f}" for _, _, t in far)
        assert f"more than 10% above the ^VVIX close ({pairs})" in sent
    for s, p, t in esc:
        clause = (f"the σ ≤ {t['sigma_cap']:g} refit (σ {t['params']['sigma']:.2f}, "
                  f"κ {t['params']['kappa']:.1f}, Feller margin {dash.signed(t['feller_margin'], '.1f')})")
        assert f"n {s} " in sent, (s, sent)
        assert clause in sent, (s, sent)
        assert f"{p['spx_oos_iv_rmse']:.2f} → {t['spx_oos_iv_rmse']:.2f} vol pts" in sent
        assert (f"VVIX {p['vvix_model']:.0f} → {t['vvix_model']:.1f} against {t['vvix_market']:.1f}"
                in sent)
        if p["sigma_at_cap"]:
            assert f"{s} the σ ≤ {p['sigma_cap']:g} box binds" in sent
    if len(esc) == 1 and far:
        assert sent.endswith(f"; one session, the other {dash.count_word(len(far))} do not escape.")


def test_vvix_tradeoff_sentence_reads_every_configuration(dash):
    """The generated sentence on configurations the store may reach: one session,
    none escaping, two escaping (grammar and counts follow the data)."""
    def row(cap, oos, vm, mkt=100.0, sig=None, kap=5.0, feller=-1.0, at_cap=False):
        return {"sigma_cap": cap, "spx_oos_iv_rmse": oos, "vvix_model": vm, "vvix_market": mkt,
                "sigma": sig if sig is not None else min(cap, 1.9), "kappa": kap, "feller_margin": feller,
                "sigma_at_cap": at_cap}

    def prof(s, top_vm, top_oos=1.0):
        return {"session": s, "production_cap": 2.0,
                "rows": [row(1.0, 3.0, 200.0), row(2.0, 2.0, 180.0, at_cap=True), row(10.0, top_oos, top_vm)]}
    one = dash.vvix_tradeoff_sentence(dash.vvix_tradeoff([prof("2026-01-02", 150.0)]))
    assert one.startswith("On the one diagnosed session even the σ ≤ 10 refit prices VVIX more than 10%")
    assert "(150 vs 100)" in one
    none_ = dash.vvix_tradeoff_sentence(dash.vvix_tradeoff([prof("2026-01-02", 150.0),
                                                            prof("2026-01-05", 160.0)]))
    assert none_.startswith("On all 2 diagnosed sessions")
    assert "escape" not in none_
    two = dash.vvix_tradeoff_sentence(dash.vvix_tradeoff(
        [prof("2026-01-02", 105.0), prof("2026-01-05", 104.0, top_oos=2.5), prof("2026-01-06", 170.0)]))
    assert two.startswith("On 1 of 3 diagnosed sessions")
    assert "On 2026-01-02 the σ ≤ 2 box binds, and the σ ≤ 10 refit" in two
    assert "lowers OOS RMSE 2.00 → 1.00 vol pts and VVIX 180 → 105.0 against 100.0" in two
    assert "raises OOS RMSE 2.00 → 2.50 vol pts and lowers VVIX 180 → 104.0" in two
    assert two.endswith("; two sessions, the other one does not escape.")
    assert dash.vvix_tradeoff_sentence([]) is None


def test_sigma_frontier_is_every_sessions_sigma_profile(dash, pages):
    """P1.2: the σ frontier plots, per diagnosed session, each σ-profile refit's
    (SPX out-of-sample RMSE, model VVIX − market): the trade-off between the two
    legs of the joint fit. The headline session is drawn in the headline colour,
    the production cap's point larger and caps above it hollow; the table behind
    it lists every point with its session."""
    html = pages["/calibration"]
    spec = _spec(html, "sigma-frontier")
    files = {s: d for s, d in _diag_files(SHIPPED / "calibrations" / "diagnostics").items()
             if (d.get("sigma_profile") or {}).get("rows")}
    _, cal = dash.load_headline_calibration()
    headline = dash._calibration_session(cal, "")
    assert {s["name"] for s in spec["series"]} == set(files)
    assert len(spec["series"]) == len(files) >= 1
    for s in spec["series"]:
        d = files[s["name"]]
        rows = sorted(d["sigma_profile"]["rows"], key=lambda r: r["sigma_cap"])
        cap = d["sigma_profile"]["production_box"]["sigma"][1]
        assert len(s["data"]) == len(rows) == 8
        want = [[dash.round_sig(r["spx_oos_iv_rmse"]), dash.round_sig(r["vvix_model"] - r["vvix_market"])]
                for r in rows]
        assert s["data"] == want, s["name"]
        assert s["pointLabels"] == [f"{r['sigma_cap']:g}" for r in rows]
        assert s["hollowPoints"] == [r["sigma_cap"] > cap for r in rows]
        assert s["bigPoints"] == [r["sigma_cap"] == cap for r in rows]
        assert s["color"] == ("headline" if s["name"] == headline else "muted")
    tbl = spec["table"]
    assert tbl["columns"][0]["label"] == "Session"
    assert len(tbl["rows"]) == sum(len(d["sigma_profile"]["rows"]) for d in files.values())
    # beside the table it reads at desk width: one panel, the table and the chart in one .split
    panel = html[html.index('id="sigma-tradeoff"'):html.index('id="rough-vs-heston"')]
    split = panel[panel.index('class="split"'):]
    assert split.index('id="sigma-profile"') < split.index('id="sigma-frontier"')


def test_vvix_kpi_is_the_published_fits_model_vvix(dash, pages):
    """P1.2: the Calibration KPI strip carried the optimiser's run time where the
    joint fit's VVIX belongs; the tile now reads model / market VVIX at the
    published fit (stored_fit of its diagnostics file), on Calibration and on the
    Overview's headline-fit panel, and the optimiser time moved to its sub-line."""
    path, cal = dash.load_headline_calibration()
    session = dash._calibration_session(cal, path)
    f = SHIPPED / "calibrations" / "diagnostics" / f"{session}.json"
    if not f.exists():
        pytest.skip("the headline fit's diagnostics are not generated yet")
    sf = json.loads(f.read_text())["stored_fit"]
    assert sf["file"] == Path(path).name
    want = f"{sf['vvix_model']:.0f} / {sf['vvix_market']:.0f}"
    for route in ("/calibration", "/"):
        tile = re.search(r'<div class="kpi-label">VVIX model / market</div>\s*'
                         r'<div class="kpi-value">(.*?)</div>', pages[route])
        assert tile, route
        assert tile.group(1) == want, route
        assert tile.group(1).split(" / ")[0] == format(sf["vvix_model"], ".0f")
    assert "Optimiser" not in re.findall(r'<div class="kpi-label">(.*?)</div>', pages["/calibration"])
    assert f"optimiser {cal['fit_time']:.1f} s" in _visible_text(pages["/calibration"])


def test_model_risk_box_bound_check_names_the_wide_refit(dash, pages):
    """P1.2: the 'Parameters on a box bound' check now says what the widest σ
    refit of the headline session gives (OOS and VVIX against the close) and
    whether it is the only diagnosed session where the refit escapes."""
    d = dash.load_page2()
    t = next((x for x in d["vvix_tradeoff"] if x["session"] == d["session"]), None)
    if t is None:
        pytest.skip("the headline fit's σ profile is not generated yet")
    tp = t["top"]
    text = _visible_text(re.sub(r"</?span[^>]*>", "", pages["/model-risk"]))
    assert (f"σ ≤ {tp['sigma_cap']:g} refit: OOS {tp['spx_oos_iv_rmse']:.2f}, VVIX {tp['vvix_model']:.1f} vs "
            f"{tp['vvix_market']:.1f} ({d['session']}") in text


def test_risk_hedge_is_a_taylor_check_not_hedge_effectiveness(dash, pages):
    """P1.3a: a 99.99% 'Hedge P&L explained' tile headlined the Risk page, though
    the marks and the Greeks share one vol input (the VIX term structure, Run A),
    so the share tests the second-order expansion, not a hedge against market
    prices. The top tile is gone; #hedge shows the Taylor check as the residual's
    sd against the daily P&L's, recomputed here from the parquet."""
    html = pages["/risk"]
    text = _visible_text(html)
    assert "Hedge P&L explained" not in text
    assert "Explained variance" not in text
    h = pd.read_parquet(SHIPPED / "backtest" / "delta_hedge_2020.parquet").dropna(subset=["pnl_total"])
    hedge = _visible_text("<body>" + html[html.index('id="hedge"'):html.index('id="chart-hedge"')])
    assert "Second-order Taylor check" in hedge
    assert f"{h['pnl_residual_a'].std(ddof=1):.2f} idx pts residual sd" in hedge
    assert f"vs daily P&L sd {h['pnl_total'].std(ddof=1):.1f} idx pts, {len(h)} days" in hedge
    assert "tests the expansion, not hedge effectiveness against market prices" in hedge


def test_stale_calibration_banner_names_its_cause_from_the_store(dash, synth):
    """P1.3b: the stale banner said 'No newer fit passed the quality gate' whether
    or not a newer fit existed. Its cause is read from the store: a later fit that
    failed the gate (with the gate's reason), a later chain with no fit, or
    neither (and then it says the store cannot tell a rejected fit from none)."""
    data, dates = synth["data"], synth["dates"]
    cal_dir = data / "calibrations"
    for p in cal_dir.glob("*.pkl"):
        p.unlink()
    old, newer = str(dates[-11].date()), str(dates[-3].date())
    _write_cals(cal_dir, {f"joint_cal_{old}.pkl": _cal(old, 1.0)})

    def banner() -> str:
        html = dash.app.test_client().get("/calibration").get_data(as_text=True)
        b = re.search(r'data-stale="calibration"(.*?)</div>\s*</div>\s*</div>', html, re.DOTALL)
        assert b
        return _visible_text("<body>" + b.group(1))
    causes = []
    t = banner()                                   # neither a later chain nor a later fit
    assert f"No SPX closing chain or fit after {old} is stored" in t
    assert "whether a later session was fitted and rejected, or not fitted" in t
    causes.append(t)
    (data / "options").mkdir(exist_ok=True)        # a later chain, no fit of it
    (data / "options" / f"spx_{newer}.parquet").write_bytes(b"")
    t = banner()
    assert f"An SPX closing chain for {newer} is stored but no fit of it; the {old} fit is shown." in t
    causes.append(t)
    bad = _cal(newer, 1.0, n_vix=1)                # a later fit the gate rejects
    _write_cals(cal_dir, {f"joint_cal_{newer}.pkl": bad})
    ok, reason = dash.calibration_gate(bad)
    assert not ok
    t = banner()
    assert f"The {newer} fit is stored but failed the quality gate: {reason}; the {old} fit is shown." in t
    causes.append(t)
    assert len(set(causes)) == 3
    assert not any("No newer fit passed the quality gate" in c for c in causes)


def test_stale_calibration_banner_on_the_shipped_store(dash, pages):
    """The shipped store's cause is computed from its own chains and fits."""
    path, cal = dash.load_headline_calibration()
    session = dash._calibration_session(cal, path)
    cause = dash.calibration_stale_cause(session)
    assert session in cause
    later_chains = [f for f in (SHIPPED / "options").glob("spx_*.parquet") if f.stem[4:] > session]
    later_fits = [r for r in dash.scan_calibrations() if r["session"] and r["session"] > session]
    if not later_chains and not later_fits:
        assert cause.startswith(f"No SPX closing chain or fit after {session} is stored.")
    if dash.load_page2()["calib_age"] >= dash.STALE_WARN_SESSIONS:
        assert cause in _visible_text(pages["/calibration"])


def test_s3_is_named_for_what_it_trades(dash, pages):
    """P1.3c: 'S3 dispersion' named a trade the backtest does not hold: S3 is the
    index leg alone, a short SPX straddle (backtest_engine.S3_LEG_DIRECTION = −1,
    every S3 trade short), with no single-name leg. No page says 'S3 dispersion';
    every chart legend and category for S3 says what it is."""
    import importlib
    engine = importlib.import_module("joint_vol_calibration.backtest.backtest_engine")
    assert engine.S3_LEG_DIRECTION == -1
    for route, html in pages.items():
        assert not re.search(r"S3[ ·]+dispersion", _visible_text(html), re.IGNORECASE), route
        assert not re.search(r"S3 dispersion", html, re.IGNORECASE), route
    html = pages["/research"]
    labels = []
    for m in re.finditer(r'<script type="application/json" id="([\w-]+)-spec">', html):
        spec = _spec(html, m.group(1))
        labels += [s["name"] for s in spec.get("series") or [] if str(s.get("name", "")).startswith("S3")]
        labels += [c for c in ((spec.get("x") or {}).get("categories") or []) if str(c).startswith("S3")]
    assert labels
    assert all("short SPX straddle" in x for x in labels), labels
    text = _visible_text(html)
    assert "S3 short SPX straddle (index leg of a dispersion trade; no single-name leg)" in text


def test_m5_vx_futures_range_is_computed_from_the_gated_fits(dash, pages):
    """P1.3d: model_inventory.json typed M5's 'misses real VX futures by
    1.85–3.18 points'. The range is the min and max of losses.vx_futures_rmse_oos
    over the gate-passing fits, with the sessions that store none, recomputed
    here from the pickles."""
    inv = json.loads((ROOT / "dashboard" / "model_inventory.json").read_text())
    m5 = next(m for m in inv["models"] if m["id"] == "M5")
    assert "{vx_futures_oos_rmse}" in m5["limitations"]
    assert not re.search(r"\d\.\d\d–\d\.\d\d", m5["limitations"])
    have, none = [], []
    for r in dash.scan_calibrations():
        if r["legacy"] is False and r["ok"] and r["session"]:
            with open(r["path"], "rb") as f:
                v = dash._cal_losses(pickle.load(f)).get("vx_futures_rmse_oos")
            (have if v is not None else none).append((r["session"], v))
    assert have
    vals = [v for _, v in have]
    want = f"{min(vals):.2f}–{max(vals):.2f}" if len(vals) > 1 else f"{vals[0]:.2f}"
    tail = f"; none stored for {', '.join(sorted(s for s, _ in none))}" if none else ""
    sess = "session" if len(have) == 1 else "sessions"
    # P3.1: the text names the table the settlements come from, through its last date
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        last = con.execute("SELECT MAX(date) FROM vix_futures_daily").fetchone()[0]
    text = (f"Heston misses real VX futures by {want} index pts RMSE out of sample "
            f"({len(have)} {sess}{tail}; settlements from vix_futures_daily in data_store/vol_system.db, "
            f"through {last}).")
    row = re.search(r'<tr id="inv-m5">(.*?)</tr>', pages["/model-risk"], re.DOTALL).group(1)
    assert text in _visible_text("<body>" + re.sub(r"</?span[^>]*>", "", row))


def test_overview_first_result_names_the_surface_and_its_test(dash, pages):
    """P1.3e: the first Overview result read 'Quote-grade surface · held-out
    strikes'; it is the arbitrage-free SVI surface with alternate strikes held
    out (surface.json fit_set.rule), on a Yahoo close snapshot with no quote time."""
    labels = re.findall(r'<span class="result-k">(.*?)</span>', pages["/"])
    assert labels
    assert "alternate strikes held out" in _html_unescape(labels[0])
    assert _html_unescape(labels[0]).startswith("Arbitrage-free SVI surface")


def _scenario_prov() -> dict:
    import pyarrow.parquet as pq
    raw = pq.read_schema(SHIPPED / "greeks" / "scenario_grid.parquet").metadata[b"scenario_provenance"]
    return json.loads(raw.decode())


def test_cash_greeks_are_unit_conversions_of_the_reference_straddle(dash, pages):
    """P1.4: the Risk KPIs printed raw vomma (max 19,931, per unit index, per
    (1.00 vol)²), a number no desk reads. They are now the reference 30-day ATM
    straddle's cash Greeks per SPX contract, recomputed here independently
    (scipy's normal density, the stated conversion constants) to 1e-9, checked
    against the grid's own delta and revaluation, and found in the tiles."""
    from scipy.stats import norm
    pv = _scenario_prov()
    S, F, K, T, sig, D = (pv[k] for k in ("S", "forward", "strike", "T", "iv_atm_forward", "discount"))
    d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    vega = 2 * D * F * norm.pdf(d1) * np.sqrt(T)
    gamma = (F / S) ** 2 * 2 * D * norm.pdf(d1) / (F * sig * np.sqrt(T))
    vanna = -(F / S) * 2 * D * norm.pdf(d1) * d2 / sig
    volga = vega * d1 * d2 / sig
    want = {"vega_usd": vega * 0.01 * 100, "gamma_usd": 0.5 * gamma * (0.01 * S) ** 2 * 100,
            "vanna_usd": vanna * (0.01 * S) * 0.01 * 100, "volga_usd": volga * 0.01 ** 2 * 100}
    got = dash.cash_greeks(S, F, K, T, sig, D)
    for k, v in want.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    # the same Black-76 straddle the grid revalues: its delta, and its P&L for small moves
    assert D * (F / S) * (2 * norm.cdf(d1) - 1) == pytest.approx(pv["hedge_delta"], rel=1e-9)
    g = pd.read_parquet(SHIPPED / "greeks" / "scenario_grid.parquet")
    g = g[g["dynamics"] == "sticky_strike"] if "dynamics" in g.columns else g
    at = {(r.spot_shock_pct, r.vol_shock_vp): r.pnl for r in g.itertuples()}
    if (0.0, 2.0) in at:
        want_2vp = 2 * want["vega_usd"] + 0.5 * want["volga_usd"] * 4
        assert at[(0.0, 2.0)] * 100 == pytest.approx(want_2vp, rel=2e-3)
    if (1.0, 0.0) in at:
        assert at[(1.0, 0.0)] * 100 == pytest.approx(want["gamma_usd"], rel=0.05)
    html = pages["/risk"]
    for k, v in want.items():
        tile = re.search(rf'id="kpi-{k}">.*?<div class="kpi-value">(.*?)</div>', html, re.DOTALL)
        assert tile, k
        assert tile.group(1) == dash.usd_desk(v), (k, tile.group(1))
    strip = html[html.index('id="risk-kpis"'):html.index('id="greeks-grid"')]
    assert "per unit index, per (1.00 vol)²" not in strip
    assert "Max vomma" not in strip
    # the staleness of the fit the Greeks come from stays on every tile
    age = dash.session_age(pv["as_of_date"], dash._latest_market_session())
    for k in want:
        sub = re.search(rf'id="kpi-{k}">.*?<div class="kpi-sub">(.*?)</div></div>', html, re.DOTALL).group(1)
        assert f"fit of {pv['as_of_date']}" in sub
        if age:
            assert f"{dash.plural(age, 'session')} old" in sub


def test_convexity_dominated_vega_share_is_vega_weighted(dash, pages):
    """P1.4: 'Unstable nodes 41 of 83' weighed a deep-wing node with no vega like
    an ATM one; the KPI is the vega-weighted share of the flagged nodes."""
    g = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    u = g["is_unstable"].astype(bool)
    share = float(g.loc[u, "vega"].sum() / g["vega"].sum()) * 100
    far = g["log_moneyness"].abs() >= 0.09 - 1e-12
    tile = re.search(r'id="kpi-convexity">.*?<div class="kpi-value">(.*?)<span', pages["/risk"], re.DOTALL)
    assert tile
    assert tile.group(1) == f"{share:.1f}"
    sub = _visible_text("<body>" + re.search(r'id="kpi-convexity">(.*?)</div></div>', pages["/risk"],
                                              re.DOTALL).group(1))
    assert f"{int(u.sum())} of {len(g)} nodes" in sub
    assert f"{int((u & far).sum())} of them at |ln(K/F)| ≥ 0.09" in sub
    assert f"{g.loc[far, 'vega'].mean():,.0f} against {g.loc[~far, 'vega'].mean():,.0f}" in sub


def test_scenario_grid_reads_in_dollars_per_contract_on_a_toggle(dash, pages):
    """P1.4: the scenario grid gains a units toggle (aria-pressed buttons): index
    points per straddle, or dollars per SPX contract (× 100). Without JavaScript
    the index-point cells show; the dollar cells equal the grid's pnl × 100."""
    html = pages["/risk"]
    panel = html[html.index('id="scenario"'):html.index('id="qv-vanna"')]
    btns = re.findall(r'<button class="btn" type="button" data-units-for="scenario-units" data-show="(\w+)" '
                      r'aria-pressed="(true|false)">(.*?)</button>', panel)
    assert [(b[0], b[1]) for b in btns] == [("base", "true"), ("alt", "false")]
    assert btns[1][2] == "$ per SPX contract"
    g = pd.read_parquet(SHIPPED / "greeks" / "scenario_grid.parquet")
    g = g[g["dynamics"] == "sticky_strike"] if "dynamics" in g.columns else g
    spots = sorted(g["spot_shock_pct"].unique())
    vols = sorted(g["vol_shock_vp"].unique())
    lut = {(r.spot_shock_pct, r.vol_shock_vp): r.pnl for r in g.itertuples()}
    tbl = re.search(r'<table class="heat" id="scenario-grid"[^>]*>(.*?)</table>', html, re.DOTALL).group(1)
    rows = re.findall(r"<tr><th scope=\"row\">(.*?)</th>(.*?)</tr>", tbl.split("<tbody>")[1], re.DOTALL)
    assert len(rows) == len(vols)
    for (label, body), v in zip(rows, vols):
        assert label == dash.signed(v, ".0f", " vol pts")
        alts = re.findall(r'<span data-unit="alt">(.*?)</span>', body)
        base = re.findall(r'<span data-unit="base">(.*?)</span>', body)
        assert alts == [dash._usd(lut[(s, v)] * 100, signed=True) for s in spots]
        for a, s in zip(alts, spots):                      # pnl × 100, rounded to the dollar
            assert abs(int(a.replace("$", "").replace(",", "").replace("−", "-")) - lut[(s, v)] * 100) <= 0.5
        assert base == [dash.signed(lut[(s, v)], ",.0f") for s in spots]
    css = DESK_CSS.read_text()
    assert '.units:not([data-show="alt"]) [data-unit="alt"]' in css
    assert "initUnits" in DESK_JS.read_text()


def test_browser_scenario_units_toggle(site_browser):
    """The units buttons switch every cell and the readout, and keep aria-pressed in step."""
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "risk.html", wait_until="load")
        cell = page.locator("#scenario-grid tbody td").first
        assert cell.locator('[data-unit="base"]:not(.sr-only)').is_visible()
        assert not cell.locator('[data-unit="alt"]:not(.sr-only)').is_visible()
        page.locator('[data-units-for="scenario-units"][data-show="alt"]').click()
        alt_btn = page.locator('[data-units-for="scenario-units"][data-show="alt"]')
        assert alt_btn.get_attribute("aria-pressed") == "true"
        assert cell.locator('[data-unit="alt"]:not(.sr-only)').is_visible()
        assert not cell.locator('[data-unit="base"]:not(.sr-only)').is_visible()
        cell.hover()
        assert "per SPX contract" in page.locator("#scenario-grid-readout").inner_text()
    finally:
        ctx.close()


def test_overview_pdv_forecast_discloses_its_rank_on_both_losses(dash, pages):
    """P1.5: the Overview showed the PDV forecast without saying how it ranks
    among the evaluated forecasters. Under the forecast it now says, from
    forecast_eval.json, its rank on 21-day MSE and on QLIKE, its membership of
    the 90% model confidence set and the best model on both; only the bad rank
    would be selective the other way. The line links to the Research panel."""
    fe = json.loads((SHIPPED / "backtest" / "inference" / "forecast_eval.json").read_text())
    bw = fe["backtest_window"]
    html = pages["/"]
    line = re.search(r'<p class="note" id="pdv-forecast-rank">(.*?)</p>', html, re.DOTALL)
    assert line
    assert '<a href="research.html#forecast">' in line.group(1)
    text = _visible_text("<body>" + re.sub(r"</?span[^>]*>", "", line.group(1)))
    for key, label in (("mse", "21-day MSE"), ("qlike", "QLIKE")):
        r = bw["rank"][key]
        assert f"{r['pdv']} of {len(r)} on {label}" in text
    mcs = {k: "pdv" in bw["mcs"][k]["90"]["included"] for k in ("mse", "qlike")}
    if len(set(mcs.values())) == 1:
        assert f"in the 90% model confidence set: {'yes' if mcs['mse'] else 'no'} on both" in text
    else:
        for k, lab in (("mse", "21-day MSE"), ("qlike", "QLIKE")):
            assert f"{'yes' if mcs[k] else 'no'} on {lab}" in text
    best = {k: min(bw["rank"][k], key=bw["rank"][k].get) for k in ("mse", "qlike")}
    if best["mse"] == best["qlike"] == "vix2":
        assert "best on both: VIX² (risk-neutral, contains the variance premium)" in text
    assert f"({bw['start']} → {bw['end']})" in text
    # the link target exists on the Research page
    assert 'id="forecast"' in pages["/research"]


_VIX_FAMILY = r"(?:VIX|VIX9D|VIX3M|VIX6M|VVIX)"


def unit_lint(html: str) -> list[str]:
    """The site's unit house style, as a lint of one rendered page.

    * 'vp' never appears in what a reader sees: the page text, the heatmap
      readouts (data-detail), labels and titles, and the chart specs' visible
      strings (axis names, units, series, the tables behind the charts); code
      elements are excluded (an artifact field such as rmse_vp is a name).
    * A VIX-family level (VIX, VIX9D, VIX3M, VIX6M, VVIX) is an index level,
      never a '%': a KPI tile with that label and a '%' unit span, a dt/dd pair,
      or the level followed by '%' in the text.
    * 'pts' always says which points: 'vol pts', 'index pts', 'idx pts' (SPX
      index points of P&L) or 'percentage pts', never bare (the Overview's
      '0.82 pts RMSE', Calibration's 'VIX indices (pts)'), in the text, the
      KPI unit spans and the chart specs; and an index level is never in 'vol
      pts' (the VIX term-structure axes read 'Index level (vol pts)').
    """
    import html as _h
    out = []
    body = re.sub(r"<(style|noscript)\b.*?</\1>", " ", html, flags=re.DOTALL)
    specs = re.findall(r'<script type="application/json"[^>]*>(.*?)</script>', body, re.DOTALL)
    body = re.sub(r"<script\b.*?</script>", " ", body, flags=re.DOTALL)
    body = re.sub(r"<code\b.*?</code>", " ", body, flags=re.DOTALL)
    attrs = " ".join(_h.unescape(v) for v in re.findall(
        r'(?:data-detail|data-detail-alt|aria-label|title)="([^"]*)"', body))
    text = _h.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)))
    for src, s in (("text", text), ("attribute", attrs)):
        out += [f"'vp' in {src}: …{s[max(0, m.start() - 40):m.end() + 5]}…"
                for m in re.finditer(r"\bvp\b", s)]
    for raw in specs:
        spec = json.loads(_h.unescape(raw))
        vis = [spec.get("title", ""), spec.get("describe", "")]
        for ax in ("x", "y", "y2"):
            a = spec.get(ax) or {}
            vis += [a.get("name", ""), a.get("unit", ""), a.get("tickUnit", ""), a.get("label", "")]
        vis += [s_.get("name", "") for s_ in spec.get("series") or []]
        tbl = spec.get("table") or {}
        vis += [tbl.get("caption", "")] + [c.get("label", "") for c in tbl.get("columns") or []]
        out += [f"'vp' in chart {spec.get('id')}: {v}" for v in vis if re.search(r"\bvp\b", str(v))]
    lv = re.findall(rf'<div class="kpi-label">\s*({_VIX_FAMILY})\s*</div>\s*<div class="kpi-value[^"]*">[^<]*'
                    r'<span class="unit">\s*%', body)
    out += [f"KPI {x} in %" for x in lv]
    out += [f"{x} in % ({y})"
            for x, y in re.findall(rf"<dt>\s*({_VIX_FAMILY})\s*</dt>\s*<dd>([^<]*%)\s*</dd>", body)]
    out += [f"{m.group(0)!r}" for m in re.finditer(
        rf"(?<![\w/])({_VIX_FAMILY})\b[ :=]*[+−-]?\d[\d,]*(?:\.\d+)?\s*%", text)]
    bare = r"(?<!vol )(?<!index )(?<!idx )(?<!percentage )\bpts\b"
    level = r"[Ii]ndex level[^()]*\(vol pts\)"
    for src, s in (("text", text), ("attribute", attrs)):
        out += [f"bare 'pts' in {src}: …{s[max(0, m.start() - 40):m.end() + 5]}…"
                for m in re.finditer(bare, s)]
    for raw in specs:
        blob = json.dumps(json.loads(_h.unescape(raw)), ensure_ascii=False)
        out += [f"bare 'pts' in a chart: …{blob[max(0, m.start() - 40):m.end() + 5]}…"
                for m in re.finditer(bare, blob)]
        out += [f"index level in vol pts: {m.group(0)}" for m in re.finditer(level, blob)]
    return out


def test_unit_house_style_on_every_page(pages):
    """P1.6: VIX-family levels are index levels, never '%' (the Overview printed
    'VIX 19.45 %'); vol differences and errors are 'vol pts' and 'vp' is never
    shown (the Surface KPIs read 'RMSE 0.48 vp', Calibration 'Lifting error
    (vp)', the Risk scenario rows '−5 vp'). Every page passes the lint."""
    for route, html in pages.items():
        assert unit_lint(html) == [], route


def test_unit_lint_catches_a_seeded_offender(pages):
    """The lint fails on each kind of offender seeded into a page that passes."""
    html = pages["/"]
    assert unit_lint(html) == []
    seeds = {
        "kpi": html.replace("</main>", '<div class="kpi"><div class="kpi-label">VVIX</div>'
                                       '<div class="kpi-value">109.70<span class="unit">%</span></div>'
                                       '</div></main>'),
        "text": html.replace("</main>", "<p>RMSE 0.48 vp on the held-out quotes</p></main>"),
        "detail": html.replace("</main>",
                               '<table><tr><td data-detail="half-spread 0.6 vp">x</td></tr></table></main>'),
        "dl": html.replace("</main>", "<dl><dt>VIX9D</dt><dd>18.20%</dd></dl></main>"),
        "prose": html.replace("</main>", "<p>with VIX3M 21.3% on the day</p></main>"),
        "bare pts": html.replace("</main>", '<div class="kpi"><div class="kpi-label">VIX indices</div>'
                                            '<div class="kpi-value">0.82<span class="unit">pts RMSE</span>'
                                            '</div></div></main>'),
        "level axis": html.replace("</main>", '<script type="application/json" id="x-spec">'
                                              '{"y": {"name": "Index level (vol pts)"}}</script></main>'),
    }
    for kind, seeded in seeds.items():
        assert unit_lint(seeded), kind
    # names in code elements are not units
    assert unit_lint(html.replace("</main>", "<p><code>rmse_vp</code> and <code>vp</code></p></main>")) == []


_ROUTE_OF = {"index.html": "/", "surface.html": "/surface", "calibration.html": "/calibration",
             "risk.html": "/risk", "research.html": "/research", "corrections.html": "/corrections",
             "model-risk.html": "/model-risk"}


def _findings(html: str) -> list[tuple[str, str, list[str]]]:
    """(class attribute, head, [item html]) for each column of #findings."""
    strip = re.search(r'<div class="findings" id="findings"[^>]*>(.*?)\n</div>', html, re.DOTALL)
    assert strip
    cols = re.findall(r'<section class="([^"]+)"[^>]*>\s*<h3 class="find-head"[^>]*>(.*?)</h3>\s*'
                      r'<ul class="find-list[^"]*"[^>]*>(.*?)</ul>', strip.group(1), re.DOTALL)
    return [(c, _visible_text("<body>" + h).strip(),
             re.findall(r'<li class="find-item">(.*?)</li>', ul, re.DOTALL))
            for c, h, ul in cols]


def test_overview_findings_at_a_glance_are_computed_and_linked(dash, pages):
    """P1.7: the Overview's first screen said what the system is, not what it
    found. #findings lists what works, what does not and what is next, three
    columns of equal weight, each line a number read from an artifact (recomputed
    here) and one link to a panel that exists on the frozen page."""
    html = pages["/"]
    cols = _findings(html)
    assert [h for _, h, _ in cols] == ["✓ Works", "✕ Does not work", "→ Next · not done yet"]
    assert len({c for c, _, _ in cols}) == 1                        # one class set: equal weight
    for _, head, items in cols:
        assert 2 <= len(items) <= 4, (head, len(items))
        for it in items:
            links = re.findall(r'<a href="([^"]+)">', it)
            assert len(links) == 1, it
            page, _, anchor = links[0].partition("#")
            assert anchor, links[0]
            assert f'id="{anchor}"' in pages[_ROUTE_OF[page]], links[0]
    text = {head: [_visible_text("<body>" + it).strip() for it in items] for _, head, items in cols}
    works, fails, nxt = (" ".join(text[h]) for h in ("✓ Works", "✕ Does not work", "→ Next · not done yet"))
    # works: the arbitrage-free surface on held-out strikes, the VIX replication, the gate
    path, cal = dash.load_headline_calibration()
    session = dash._calibration_session(cal, path)
    sc = json.loads(_surface_file("scorecard.json").read_text())
    svi = next(m for m in sc["models"] if m["model"] == "svi_cal")["metrics"]["out_of_sample"]
    assert (f"median {svi['median_abs_hs']:.2f} half-spreads from the mid, "
            f"{svi['inside_spread_pct']:.1f}% inside") in works
    rep = json.loads(_surface_file("vix_replication.json").read_text())
    gap = (rep["indices"].get("VIX") or {}).get("gap_vs_official")
    if gap is None:     # a session whose chain does not replicate the VIX: no replication line
        assert "index pts from the Cboe close" not in works
    else:
        assert f"{dash.signed(gap, '.2f')} index pts from the Cboe close" in works
    rows = dash.scan_calibrations()
    n_legacy = sum(1 for r in rows if r["legacy"])
    assert f"{n_legacy:,} pre-C22 fit" in works
    assert "none published" in works
    # does not work: Heston's worst short expiry with σ on its cap and its VVIX; the backtest; the classifier
    t_by = {str(c["expiry"]): float(c["T"]) * 365 for c in cal["carry_by_expiry"]}
    short = {e: v for e, v in cal["oos"]["spx_oos_rmse_by_expiry"].items() if t_by.get(e, 99) <= 30}
    e_w = max(short, key=short.get)
    assert f"{short[e_w]:.2f} vol pts out of sample on the {e_w} expiry" in fails
    diag = SHIPPED / "calibrations" / "diagnostics" / f"{session}.json"
    if diag.exists():
        sf = json.loads(diag.read_text())["stored_fit"]
        assert f"VVIX {sf['vvix_model']:.0f} against {sf['vvix_market']:.0f}" in fails
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        em = pickle.load(f)["metrics"]
    assert dash._pct_prose(em["cumulative_return"], 1) in fails
    summ = json.loads((SHIPPED / "backtest" / "inference" / "summary.json").read_text())
    b = summ["bootstrap"]["portfolio"]["sharpe"]
    ci = dash.artifacts.bootstrap_ci(dash.artifacts.load_inference(SHIPPED), "portfolio", "sharpe")
    assert f"Sharpe {ci['point']:+.2f}".replace("-", "−") in fails
    assert f"95% CI {b['ci95'][0]:+.2f} to {b['ci95'][1]:+.2f}".replace("-", "−") in fails
    with open(SHIPPED / "signals" / "regime_classifier.pkl", "rb") as f:
        ev = pickle.load(f).metadata_["evaluation"]
    assert (f"{ev['oos_acc'] * 100:.1f}% out-of-sample accuracy against {ev['persistence_acc'] * 100:.1f}% "
            "for the predict-yesterday baseline") in fails
    # next: phrased as not done
    n_fw = len(list((SHIPPED / "forward").glob("ledger_*.json"))) if (SHIPPED / "forward").is_dir() else 0
    assert f"{n_fw} registered session" in nxt
    assert all(t.split(": ", 1)[1].startswith(("not ", "not yet")) for t in text["→ Next · not done yet"])


def test_browser_findings_strip_is_on_the_first_screen_and_stacks_on_a_phone(site_browser):
    """P1.7: at 1440×900 the strip starts above y = 600 (the first screen); at
    390 px its three columns stack, full width, in the same type."""
    for width, height in ((1440, 900), (390, 844)):
        ctx, page = _page(site_browser, width, height)
        try:
            page.goto(site_browser["base"] + "index.html", wait_until="load")
            box = page.evaluate("""() => { const s = document.getElementById('findings');
              const cols = [...s.querySelectorAll('.find-col')].map(c => {
                const r = c.getBoundingClientRect();
                return [r.left, r.top, r.width, getComputedStyle(c.querySelector('.find-item')).fontSize]; });
              return {top: s.getBoundingClientRect().top + window.scrollY, cols}; }""")
            assert len(box["cols"]) == 3
            assert len({c[3] for c in box["cols"]}) == 1
            if width == 1440:
                assert box["top"] < 600, box
                assert len({round(c[1]) for c in box["cols"]}) == 1          # side by side
            else:
                assert len({round(c[0]) for c in box["cols"]}) == 1          # stacked
                assert box["cols"][0][1] < box["cols"][1][1] < box["cols"][2][1]
        finally:
            ctx.close()


def test_corrections_register_lists_every_finding(dash, pages):
    """P1.8: the Corrections page was 7,685 px of postmortems with no overview.
    A register above them lists every finding of corrections.json: its id,
    round, flaw, status, the portfolio P&L effect where a bridge CSV measures it
    (else '—'), the number of tests that pin it ('gap' at 0) and a link that
    resolves to its postmortem entry. Without JavaScript every row shows."""
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    fs = [(r["id"], n, f) for r in reg["rounds"] for n, f in enumerate(r["findings"], 1)]
    html = pages["/corrections"]
    tbl = re.search(r'<table class="tbl dense cards" id="register-table">(.*?)</table>', html, re.DOTALL)
    assert tbl
    rows = re.findall(r'<tr data-status="(\w+)" data-tests="(\d+)">(.*?)</tr>', tbl.group(1), re.DOTALL)
    assert len(rows) == len(fs) == 33
    assert [s for s, _, _ in rows] == [f["status"] for _, _, f in fs]
    assert (sum(s == "fixed" for s, _, _ in rows), sum(s == "quantified" for s, _, _ in rows)) == (30, 3)
    gaps = [i for i, (_, n, _) in enumerate(rows) if n == "0"]
    assert gaps
    assert all("chip-breach\">gap</span>" in rows[i][2] for i in gaps)
    assert [int(n) for _, n, _ in rows] == [len(f.get("tests") or []) for _, _, f in fs]
    c22 = pd.read_csv(SHIPPED / "backtest" / "c22_bridge.csv").set_index("step")
    c23 = pd.read_csv(SHIPPED / "backtest" / "c23_bridge.csv").set_index("step")
    for (rid, n, f), (_, _, body) in zip(fs, rows):
        cells = [_visible_text("<body>" + c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", body, re.DOTALL)]
        assert cells[0] == (f.get("id") or f"{rid}.{n}")
        if f.get("bridge_steps"):
            want = float(c22.loc[f["bridge_steps"], "d_pnl"].sum())
        elif f.get("c23_bridge_steps"):
            want = float(c23.loc[f["c23_bridge_steps"], "d_pnl"].sum())
        else:
            want = None
        effect = "—" if want is None else ("unchanged" if abs(want) < 500 else dash._usd_k(want))
        assert cells[4] == effect, rid
        link = re.search(r'<a href="#([\w-]+)">', body).group(1)
        assert link == f"{rid.lower()}-f{n}"
        assert re.search(rf'<details class="pm-cell pm-finding" id="{link}"', html), link
    assert "hidden" not in tbl.group(1)
    btns = re.findall(r'data-filter-for="register-table" data-filter="(\w+)" aria-pressed="(true|false)"',
                      html)
    assert btns == [("all", "true"), ("quantified", "false"), ("notest", "false")]
    # the rounds are disclosures, the newest open; every finding stays in the page
    opened = re.findall(r'<details class="pm-details"( open)?>', html)
    assert opened == [""] * (len(reg["rounds"]) - 1) + [" open"]


def test_browser_corrections_register_filters_links_and_height(site_browser):
    """P1.8: the filter chips show the quantified findings and the ones no test
    pins; an entry link opens its collapsed round and finding and scrolls to it;
    the page is under 3,500 px at 1440 (7,685 before)."""
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "corrections.html", wait_until="load")
        assert page.evaluate("document.documentElement.scrollHeight") < 3500
        shown = ("() => [...document.querySelectorAll('#register-table tbody tr')]"
                 ".filter(t => t.offsetParent).length")
        n_all = page.evaluate(shown)
        page.locator('[data-filter="notest"]').click()
        assert 1 <= page.evaluate(shown) < n_all
        assert page.locator('[data-filter="notest"]').get_attribute("aria-pressed") == "true"
        page.locator('[data-filter="quantified"]').click()
        n_quant = page.locator('#register-table tbody tr[data-status="quantified"]').count()
        assert page.evaluate(shown) == n_quant
        page.locator('[data-filter="all"]').click()
        assert page.evaluate(shown) == n_all
        first = page.locator("#register-table tbody tr a").first
        target = first.get_attribute("href")[1:]
        assert page.evaluate(f"!document.getElementById('{target}').closest('details.pm-details').open")
        first.click()
        page.wait_for_timeout(200)
        st = page.evaluate(f"""() => {{ const el = document.getElementById('{target}');
          const r = el.getBoundingClientRect();
          return [el.open, el.closest('details.pm-details').open, r.top >= 0 && r.top < innerHeight]; }}""")
        assert st == [True, True, True], st
    finally:
        ctx.close()


# ── P2: the surface and skew in desk coordinates ─────────────────────────────

def _desk_inputs() -> tuple[dict, pd.DataFrame, dict]:
    """surface.json, residuals.parquet and vix_replication.json of the session the
    Surface page renders (the newest stored one)."""
    return (json.loads(_surface_file("surface.json").read_text()),
            pd.read_parquet(_surface_file("residuals.parquet")),
            json.loads(_surface_file("vix_replication.json").read_text()))


def _desk_module():
    """dashboard/desk_surface.py (dashboard/ is on sys.path once the dash fixture imported app.py)."""
    return importlib.import_module("desk_surface")


def _table_heads(html: str, table_id: str) -> list[str]:
    tbl = re.search(rf'<table class="tbl[^"]*" id="{re.escape(table_id)}"[^>]*>(.*?)</table>', html,
                    re.DOTALL)
    assert tbl, table_id
    head = tbl.group(1).split("<thead>")[1].split("</thead>")[0]
    return [_html_unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<th[^>]*>(.*?)</th>", head, re.DOTALL)]


def _kpi_tiles(html: str, block_id: str) -> dict[str, str]:
    start = html.index(f'id="{block_id}"')
    ends = [i for i in (html.find('<div class="kpis"', start), html.find('<div class="g"', start)) if i > 0]
    blk = html[start:min(ends)]
    out = {}
    pat = r'<div class="kpi-label">(.*?)</div>\s*<div class="kpi-value[^"]*">(.*?)</div>'
    for lab, val in re.findall(pat, blk, re.DOTALL):
        out[_html_unescape(re.sub(r"<[^>]+>", "", lab)).strip()] = _html_unescape(
            re.sub(r'<span class="unit">.*?</span>', "", val)).strip()
    return out


DELTA_GRID_HEADS = ["Expiry", "T (d)", "F", "10ΔP", "25ΔP", "ATMF", "25ΔC", "10ΔC", "RR25", "BF25", "RR10",
                    "BF10", "ATM skew", "VS vol", "VS − ATMF", "Median half-spread", "Quotes"]


def test_surface_delta_grid_panel_lists_every_expiry_and_five_tenors(dash, pages):
    """P2.1: the Surface page opens with the SVI-cal surface in desk coordinates:
    one row per listed expiry (the slices of the rendered session) and five
    constant-maturity rows, each with exactly the desk's columns, the conventions
    in the caption, and the values desk_surface computes from the artifacts."""
    ds = _desk_module()
    html = pages["/surface"]
    sj, res, vix = _desk_inputs()
    listed = ds.delta_grid(sj, res, vix)
    cm = ds.constant_maturity(listed)
    first = re.search(r'<section class="panel [^"]*" id="([\w-]+)"', html).group(1)
    assert first == "delta-grid"
    for tid in ("delta-grid-listed", "delta-grid-cm"):
        assert _table_heads(html, tid) == DELTA_GRID_HEADS, tid
        cap = re.search(rf'id="{tid}">\s*<caption>(.*?)</caption>', html, re.DOTALL).group(1)
        assert "RR = call − put" in _html_unescape(cap)
        assert "not premium-adjusted" in _html_unescape(cap)
    rows_l = _table_rows(html, "delta-grid-listed")
    rows_c = _table_rows(html, "delta-grid-cm")
    assert len(rows_l) == len(sj["surfaces"]["svi_cal"]["params"]["slices"]) == len(listed)
    assert len(rows_c) == 5
    assert [r[0][:2] for r in rows_c] == ["1M", "2M", "3M", "6M", "1Y"]
    col = {h: i for i, h in enumerate(DELTA_GRID_HEADS)}
    for r, g in zip(rows_l, listed):
        assert r[0] == g["expiry"]
        assert r[col["ATMF"]].startswith(dash._fmt(g["atmf"], ".2f"))
        assert r[col["RR25"]] == dash.signed(g["rr25"], ".2f")
        assert r[col["BF25"]] == dash.signed(g["bf25"], ".2f")
    for r, g in zip(rows_c, cm):
        assert r[col["ATMF"]].startswith(dash._fmt(g["atmf"], ".2f"))
    # the tab toggle names both row sets with their counts
    assert f">Listed expiries ({len(listed)})</button>" in html
    assert ">Constant maturity (5)</button>" in html
    # the grid sits in a labelled, focusable scroll region (phones scroll it, not the page)
    assert html.count('role="region" aria-label="Delta grid, listed expiries (scrollable)"') == 1


# the heads as text: S_T and g_min are set as S<sub>T</sub> and g<sub>min</sub>
RND_HEADS = ["Expiry", "Days", "Quoted K/F", "ST < 0.80 F", "ST < 0.90 F", "Outside 0.95–1.05 F",
             "ST > 1.10 F", "Mass below quoted", "Mass above quoted", "∫p", "E[ST/F]", "gmin"]


def test_surface_rnd_panels_print_every_slices_risk_neutral_tails(dash, pages):
    """B9: the Surface page reads the certified SVI-cal slices as distributions
    (desk_density.py, Breeden–Litzenberger). The table lists every listed slice
    of the rendered session with the module's numbers; a threshold beyond the
    quoted strikes prints 'beyond quoted strikes', never a number; the captions
    say risk-neutral and no higher moment is printed; the density chart draws
    each picked expiry only inside its quoted strikes, and the tail chart has a
    gap wherever the table says 'beyond quoted strikes'."""
    dd = importlib.import_module("desk_density")
    sj, res, _ = _desk_inputs()
    rows = dd.tail_rows(sj, res)
    html = pages["/surface"]
    assert _table_heads(html, "rnd-table-tbl") == RND_HEADS
    got = _table_rows(html, "rnd-table-tbl")
    assert len(got) == len(rows) == len(sj["surfaces"]["svi_cal"]["params"]["slices"])
    col = {h: i for i, h in enumerate(RND_HEADS)}
    keys = {"ST < 0.80 F": "lt80", "ST < 0.90 F": "lt90", "Outside 0.95–1.05 F": "out5",
            "ST > 1.10 F": "gt110"}
    assert "<th scope=\"col\" class=\"num\"><i>S</i><sub>T</sub> &lt; 0.90 F</th>" in html
    n_beyond = 0
    for cells, r in zip(got, rows):
        assert cells[0].split(" ")[0] == r["expiry"]
        for head, key in keys.items():
            p = r["prob"][key]
            if p is None:
                n_beyond += 1
                assert cells[col[head]] == "beyond quoted strikes", (r["expiry"], head)
            else:
                assert cells[col[head]] == dash._fmt(100 * p, ".2f"), (r["expiry"], head)
        assert cells[col["Mass below quoted"]] == dash._fmt(100 * r["below_quoted"], ".2f")
        assert cells[col["∫p"]] == dash._fmt(r["mass"], ".4f")
        assert cells[col["E[ST/F]"]] == dash._fmt(r["mean"], ".4f")
        if r["certified"]:                     # a certified slice is a distribution with mean F
            assert cells[col["∫p"]] == "1.0000"
            assert cells[col["E[ST/F]"]] == "1.0000"
        assert cells[col["gmin"]] == dd.fmt_g(r["g_min"])
        assert not cells[col["gmin"]].startswith(("0.0000", "−"))   # positive: never a bare 0.0000
    assert n_beyond == sum(1 for r in rows for v in r["prob"].values() if v is None)
    panel = html[html.index('id="rnd"'):html.index('id="iv-heat"')]
    text = _visible_text("<body>" + panel)
    assert "Risk-neutral" in text
    assert "not a forecast" in text
    assert re.findall(r"skewness|kurtosis", text, re.IGNORECASE) == ["skewness", "kurtosis"]   # "no … shown"
    assert "no skewness or kurtosis is shown" in text
    specs = _raw_specs(html)
    picks = {p["row"]["expiry"]: p["row"] for p in dd.chart_expiries(rows)}
    dens = specs["chart-rnd"]["series"]
    assert len(dens) == len(picks) >= 2
    for s_ in dens:
        r = next(v for e, v in picks.items() if e in s_["name"])
        ks = [x for x, _ in s_["data"]]
        assert min(ks) >= r["k_lo"] - 1e-6, s_["name"]
        assert max(ks) <= r["k_hi"] + 1e-6, s_["name"]
        assert s_["role"] == "tenor"
    tails = {s_["name"]: s_["data"] for s_ in specs["chart-rnd-tails"]["series"]}
    for name, key in (("SVI-cal P(S_T < 0.90 F)", "lt90"), ("SVI-cal P(S_T > 1.10 F)", "gt110")):
        assert [v is None for _, v in tails[name]] == [r["prob"][key] is None for r in rows], name


def test_surface_state_tiles_are_the_one_month_constant_maturity_row(dash, pages):
    ds = _desk_module()
    listed = ds.delta_grid(*_desk_inputs())
    cm = {r["label"]: r for r in ds.constant_maturity(listed)}
    tiles = _kpi_tiles(pages["/surface"], "surface-state")
    m1, y1 = cm["1M"], cm["1Y"]
    assert tiles["1M ATMF"] == dash._fmt(m1["atmf"], ".2f")
    assert tiles["1M RR25"] == dash.signed(m1["rr25"], ".2f")
    assert tiles["1M BF25"] == dash.signed(m1["bf25"], ".2f")
    assert tiles["1M VS − ATMF"] == dash.signed(m1["vs_minus_atmf"], ".2f")
    slope = (y1["atmf"] - m1["atmf"]) if (y1["available"] and m1["available"]) else None
    assert tiles["ATMF slope 1M → 1Y"] == dash.signed(slope, ".2f")
    # the state row sits above the model-error tiles
    html = pages["/surface"]
    assert html.index('id="surface-state"') < html.index('id="surface-scores"')


def _kpi_sub(html: str, block_id: str, label: str) -> str:
    """The visible sub-line of the KPI tile ``label`` in the tile row ``block_id``."""
    start = html.index(f'id="{block_id}"')
    ends = [i for i in (html.find('<div class="kpis"', start), html.find('<div class="g"', start)) if i > 0]
    blk = html[start:min(ends)]
    for tile in blk.split('<div class="kpi">')[1:]:
        lab = re.search(r'<div class="kpi-label">(.*?)</div>', tile, re.DOTALL)
        if lab and _html_unescape(re.sub(r"<[^>]+>", "", lab.group(1))).strip() == label:
            sub = re.search(r'<div class="kpi-sub">(.*?)</div>', tile, re.DOTALL)
            assert sub, f"{label}: no sub-line"
            return _html_unescape(re.sub(r"<[^>]+>", "", sub.group(1))).strip()
    raise AssertionError(f"no {label!r} tile in #{block_id}")


def test_surface_vs_tile_names_its_vs_beside_the_replicated_vix_and_the_cboe_close(dash, pages):
    """R3-T1: the '1M VS − ATMF' tile printed '+5.18' with no VS level, on a page
    that also prints the 30-day variance as the VIX replicated from this chain
    and the Cboe close (19.11, 19.70 and 19.99 on 2026-10-16). Its sub-line names
    all three: the 1M row's VS and vix_replication.json's VIX."""
    ds = _desk_module()
    sj, res, vx = _desk_inputs()
    m1 = ds.constant_maturity(ds.delta_grid(sj, res, vx))[0]
    assert m1["label"] == "1M"
    sub = _kpi_sub(pages["/surface"], "surface-state", "1M VS − ATMF")
    if m1["vs"] is not None:
        assert f"VS {dash._fmt(m1['vs'], '.2f')}% (the bracketing strips at 30 days)" in sub, sub
    vix = (vx.get("indices") or {}).get("VIX")
    if vix:
        off = vix.get("official")
        close = off.get("close") if isinstance(off, dict) else off
        assert f"VIX replicated {dash._fmt(vix.get('replicated'), '.2f')}" in sub, sub
        assert f"Cboe close {dash._fmt(close, '.2f')}" in sub, sub


def test_constant_maturity_grid_names_the_strike_counts_of_its_two_vs_strips(dash, pages):
    """R3-T1: a constant-maturity VS vol interpolates two strips, and a thin one
    moves it (the 2026-10-16 1M VS read a 194-strike and an 86-strike strip).
    Under each tenor: the two strips' strike counts, the 'Strikes' column of the
    variance-swap strip table."""
    ds = _desk_module()
    cm = ds.constant_maturity(ds.delta_grid(*_desk_inputs()))
    html = pages["/surface"]
    strip = _raw_specs(html)["chart-strip"]["table"]
    cols = [c["label"] for c in strip["columns"]]
    n = {r[0]: r[cols.index("Strikes")] for r in strip["rows"]}
    tbl = html[html.index('id="delta-grid-cm"'):]
    tbl = tbl[tbl.index("<tbody>"):tbl.index("</table>")]
    heads = re.findall(r'<th scope="row">(.*?)</th>', tbl, re.DOTALL)
    assert len(heads) == len(cm)
    checked = 0
    for r, th in zip(cm, heads):
        subs = [_html_unescape(re.sub(r"<[^>]+>", "", x)).strip()
                for x in re.findall(r'<span class="dg-sub">(.*?)</span>', th, re.DOTALL)]
        if not r["available"] or r["lower"] not in n or r["upper"] not in n:
            assert not any("strikes" in x for x in subs), (r["label"], subs)
            continue
        want = (f"{n[r['lower']]} VS strikes" if r["lower"] == r["upper"]
                else f"{n[r['lower']]} / {n[r['upper']]} VS strikes")
        assert want in subs, (r["label"], subs)
        checked += 1
    assert checked
    assert cm[0]["label"] == "1M"
    assert cm[0]["available"]


def test_overview_surface_line_is_the_one_month_row_with_its_age(dash, pages):
    ds = _desk_module()
    m1 = ds.constant_maturity(ds.delta_grid(*_desk_inputs()))[0]
    assert m1["label"] == "1M"
    line = re.search(r'<p class="note" id="surface-line">(.*?)</p>', pages["/"], re.DOTALL)
    assert line, "the Overview's headline-fit panel lost its surface line"
    text = re.sub(r"\s+([,.)])", r"\1", _visible_text("<body>" + line.group(1)))
    session = _surface_file("surface.json").parent.name
    age = dash.session_age(session, dash._latest_market_session())
    assert (f"1M ATMF {dash._fmt(m1['atmf'], '.2f')}% · RR25 {dash.signed(m1['rr25'], '.2f')} · "
            f"BF25 {dash.signed(m1['bf25'], '.2f')} vol pts") in text
    assert f"surface {session}, " in text
    assert ("the newest session" if age == 0 else f"{dash.plural(age, 'session')} old") in text
    assert 'href="/surface#delta-grid"' in line.group(1)


def test_surface_stale_banner_still_renders_above_the_new_panels(dash, pages):
    session = _surface_file("surface.json").parent.name
    age = dash.session_age(session, dash._latest_market_session())
    html = pages["/surface"]
    banner = re.search(r'<div class="banner banner-stale-(\w+)" data-stale="surface" data-age="(\d+)"', html)
    if dash.stale_level(age) == "ok":
        assert banner is None
    else:
        assert banner
        assert banner.group(1) == dash.stale_level(age)
        assert int(banner.group(2)) == age
        assert html.index('data-stale="surface"') < html.index('id="delta-grid"')


def test_delta_grid_renders_an_extrapolated_pillar_and_dashes_its_rr_bf(dash):
    """A synthetic slice quoted only up to between its 25Δ and 10Δ call strikes:
    the 10ΔC cell carries '◇ extrapolated' and RR10 / BF10 print '—'."""
    ds = _desk_module()
    sl = {"a": 0.0004, "b": 0.05, "rho": -0.6, "m": 0.0, "sigma": 0.12, "T": 28 / 365, "expiry": "2030-01-28"}
    k25, k10 = ds.delta_strike(sl, 0.25), ds.delta_strike(sl, 0.10)
    sj = {"surfaces": {"svi_cal": {"params": {"slices": [sl]}}},
          "term_structure": [{"expiry": sl["expiry"], "T_days": 28.0, "forward": 5000.0, "n_quotes": 12,
                              "median_half_spread_vp": 0.2}]}
    res = pd.DataFrame({"model": "svi_cal", "expiry": sl["expiry"], "k": [-0.5, (k25 + k10) / 2]})
    listed = ds.delta_grid(sj, res, {"strip": []})
    view = dash.desk_grid_view(listed, ds.constant_maturity(listed))
    macros = dash.app.jinja_env.get_template("_macros.html").module
    html = str(macros.desk_grid(view, view["listed"], "t-grid", "caption", "label"))
    (row,) = _table_rows(html, "t-grid")
    col = {h: i for i, h in enumerate(DELTA_GRID_HEADS)}
    assert row[col["10ΔC"]].endswith("◇ extrapolated")
    assert "extrapolated" not in "".join(row[col["10ΔP"]:col["25ΔC"] + 1])
    assert row[col["RR10"]] == row[col["BF10"]] == "—"
    assert row[col["RR25"]] != "—"
    assert row[col["BF25"]] != "—"
    assert row[col["VS vol"]] == "—"                     # no strip row for the expiry: not invented


def _independent_heston_skew(params: dict, t_days: float, h: float) -> float:
    """Central difference of Heston implied vol at ln(K/F) = ±h: prices from the
    pricer, inverted here with brentq on Black-76 (not the app's smile code)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    from joint_vol_calibration.models.heston import heston_call_batch

    T = t_days / 365.0
    K = np.exp(np.array([-h, h]))
    calls = heston_call_batch(1.0, K, T, 0.0, 0.0, params["kappa"], params["theta"], params["sigma"],
                              params["rho"], params["v0"])

    def black_call(k_: float, s: float) -> float:
        d1 = (-np.log(k_) + 0.5 * s * s * T) / (s * np.sqrt(T))
        return float(norm.cdf(d1) - k_ * norm.cdf(d1 - s * np.sqrt(T)))

    iv = [brentq(lambda s, k_=k_, c=c: black_call(k_, s) - c, 1e-4, 5.0, xtol=1e-14)
          for k_, c in zip(K, calls)]
    return (iv[1] - iv[0]) / (2 * h)


def test_skew_term_heston_series_is_the_published_fit_and_its_difference_converges(dash, pages):
    """P2.2: the Heston skew is drawn at the parameters the Calibration page
    shows, at the headline session's listed maturities, and at the expiry
    nearest one year it agrees with an independent central difference at
    h = 0.005 to 1e-4 (a Richardson check of the step)."""
    d = dash.load_page2()
    sk = d["skew_term"]
    _, cal = dash.load_headline_calibration()
    assert sk["heston_params"] == cal["params"]
    tiles = {t["key"]: t["value"] for t in d["param_tiles"]}
    assert tiles == {k: float(cal["params"][k]) for k in tiles}
    spec = _spec(pages["/calibration"], "chart-skew-term")
    ser = {s["name"]: s for s in spec["series"]}
    days = [r["T_days"] for r in sk["rows"]]
    want = dash.heston_atm_skew(cal["params"], days)
    got = [y for _, y in ser["Heston, published fit"]["data"]]
    assert got == pytest.approx([abs(w) for w in want], rel=1e-5)
    yr = min(sk["rows"], key=lambda r: abs(r["T_days"] - 365))
    ref = _independent_heston_skew(cal["params"], yr["T_days"], 0.005)
    assert yr["heston"] == pytest.approx(ref, abs=1e-4)
    # the σ-profile refit is the diagnostics file's widest cap
    diag = SHIPPED / "calibrations" / "diagnostics" / f"{d['session']}.json"
    if diag.exists():
        rows = json.loads(diag.read_text())["sigma_profile"]["rows"]
        top = max(rows, key=lambda r: r["sigma_cap"])
        keys = ("kappa", "theta", "sigma", "rho", "v0")
        assert sk["refit"]["params"] == {k: top["params"][k] for k in keys}



def _gil_pelaez_call(params: dict, k: float, T: float) -> float:
    """Forward-normalised Heston call (S = F = 1, r = q = 0) by Gil-Pelaez
    inversion of the characteristic function (Albrecher et al.'s 'little trap'
    form), integrated with scipy quad: no code shared with the repository's
    Lewis-contour pricer."""
    from scipy.integrate import quad

    ka, th, sg, rho, v0 = (params[x] for x in ("kappa", "theta", "sigma", "rho", "v0"))

    def cf(u: complex) -> complex:
        b = ka - rho * sg * 1j * u
        d = np.sqrt(b * b + sg * sg * (1j * u + u * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        return np.exp(ka * th / sg ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
                      + v0 * (b - d) / sg ** 2 * (1 - e) / (1 - g * e))

    def prob(shift: complex) -> float:
        f = lambda u: (np.exp(-1j * u * k) * cf(u + shift) / (1j * u)).real  # noqa: E731
        return 0.5 + quad(f, 1e-10, 2000.0, limit=2000, epsabs=1e-13, epsrel=1e-12)[0] / np.pi

    return prob(-1j) - np.exp(k) * prob(0.0)


def test_heston_short_end_skew_agrees_with_an_independent_pricer(dash):
    """P2.2: the plan expected the market's short-dated ATM skew to lie above the
    published Heston fit's; the page shows the opposite and says so in a
    'does not work' finding. That finding rests on heston_atm_skew, so at the
    headline session's shortest listed expiry it is checked against a
    Gil-Pelaez pricer that shares no code with the repository's (Richardson
    central difference at h = 0.01 / 0.005, Black-76 inverted with brentq), and
    the side of the SVI-cal market skew it falls on is the side the rows say."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    sk = dash.load_page2()["skew_term"]
    first = next(r for r in sk["rows"] if r["heston"] is not None)
    T = first["T_days"] / 365.0

    def iv(k: float) -> float:
        c = _gil_pelaez_call(sk["heston_params"], k, T)
        black = lambda s: (norm.cdf(-k / (s * np.sqrt(T)) + s * np.sqrt(T) / 2)  # noqa: E731
                           - np.exp(k) * norm.cdf(-k / (s * np.sqrt(T)) - s * np.sqrt(T) / 2))
        return brentq(lambda s: black(s) - c, 1e-4, 5.0, xtol=1e-14)

    d_h = (iv(0.01) - iv(-0.01)) / 0.02
    d_hh = (iv(0.005) - iv(-0.005)) / 0.01
    ref = (4 * d_hh - d_h) / 3
    assert first["heston"] == pytest.approx(ref, rel=2e-3), (first["T_days"], first["heston"], ref)
    assert (abs(ref) > abs(first["svi"])) == (first["ratio"] > 1)

def test_skew_term_chart_is_log_log_with_units_and_the_slope_table_is_polyfit(dash, pages):
    ds = _desk_module()
    html = pages["/calibration"]
    spec = _spec(html, "chart-skew-term")
    assert spec["x"]["log"] is True
    assert spec["y"]["log"] is True
    assert "years" in spec["x"]["name"]
    assert "vol per unit ln(K/F)" in spec["y"]["name"]
    cap = _visible_text("<body>" + re.search(r'id="skew-term-caption">(.*?)</p>', html, re.DOTALL).group(1))
    assert "not an estimate of H" in cap
    assert "T^(H−1/2)" in cap
    rows = {r[0]: r for r in _table_rows(html, "skew-slopes")}
    sessions = dash.artifacts.load_surface_desk_inputs(SHIPPED)
    assert sorted(rows) == [s["session"] for s in sessions]
    for s in sessions:
        sj = s["surface"]
        sl = sorted(sj["surfaces"]["svi_cal"]["params"]["slices"], key=lambda x: x["T"])
        ts = sorted(sj["term_structure"], key=lambda t: t["T_days"])
        t1, k1 = np.array([x["T"] for x in sl]), np.array([ds.svi_atm_skew(x) for x in sl])
        t2, k2 = np.array([t["T_days"] / 365 for t in ts]), np.array([t["atm_skew_essvi"] for t in ts])
        cells = rows[s["session"]]
        for col, (t, k, lo) in zip((2, 3, 5, 6), ((t1, k1, 0), (t2, k2, 0), (t1, k1, 30), (t2, k2, 30))):
            keep = t * 365 >= lo
            ref = np.polyfit(np.log(t[keep]), np.log(np.abs(k[keep])), 1)[0]
            assert cells[col] == dash.signed(ref, ".3f"), (s["session"], col)      # printed unclipped
        assert cells[1].startswith(f"{len(sl)} · ")


def test_skew_term_sentence_states_what_the_data_shows(dash, pages):
    """The sentence under the chart is computed: whether the published Heston
    skew is steeper or flatter than the SVI-cal market skew, its ratio range,
    and on how many expiries under 30 days the market line is above Heston."""
    sk = dash.load_page2()["skew_term"]
    rows = sk["rows"]
    ratios = [abs(r["heston"]) / abs(r["svi"]) for r in rows]
    short = [r for r in rows if r["T_days"] < 30]
    above = sum(1 for r in short if abs(r["svi"]) > abs(r["heston"]))
    sent = re.search(r'id="skew-term-sentence">(.*?)</p>', pages["/calibration"], re.DOTALL)
    text = _visible_text("<body>" + sent.group(1))
    assert f"×{min(ratios):.2f} to ×{max(ratios):.2f}" in text
    if all(x > 1 for x in ratios):
        assert "steeper than the SVI-cal market skew at every listed expiry" in text
    elif all(x < 1 for x in ratios):
        assert "flatter than the SVI-cal market skew at every listed expiry" in text
    if short:
        assert f"market line lies above the Heston line on {above} of {len(short)} expiries" in text
    # the pair carries its unit, the decimal skew per unit ln(K/F), not the grid's ×10 (R3-T8)
    f = next(r for r in rows if r["ratio"] is not None)
    assert (f"|skew| {abs(f['heston']):.2f} against {abs(f['svi']):.2f} vol per unit ln(K/F) at "
            f"{f['T_days']:.0f} d") in text


def test_findings_cite_the_skew_power_law_and_link_to_its_panel(dash, pages):
    sk = dash.load_page2()["skew_term"]
    slopes = [p["slopes"]["svi_all"]["slope"] for p in sk["sessions"] if p["slopes"]["svi_all"]]
    items = [it for _, _, its in _findings(pages["/"]) for it in its if "calibration.html#skew-term" in it]
    assert len(items) == 1
    text = _visible_text("<body>" + items[0])
    assert (f"slope in T is {dash.signed(min(slopes), '.2f')} to {dash.signed(max(slopes), '.2f')} over "
            f"{len(slopes)} sessions") in text
    # the Overview's pair names its unit and that it is a magnitude (R3-T8): the
    # Surface grid prints the same skew ×10, in vol pts per 0.1 of ln(K/F)
    f = next(r for r in sk["rows"] if r["ratio"] is not None)
    assert (f"(|skew| {abs(f['heston']):.2f} against {abs(f['svi']):.2f} vol per unit ln(K/F) at "
            f"{f['T_days']:.0f} d)") in text
    # the Surface grid's skew column and the rough-vs-Heston panel link to it
    assert '<a href="calibration.html#skew-term">ATM skew</a>' in pages["/surface"].replace(
        'href="/calibration#', 'href="calibration.html#')
    rv = re.search(r'id="rough-vs-heston".*?</section>', pages["/calibration"], re.DOTALL).group(0)
    assert 'href="#skew-term"' in rv


def test_browser_skew_term_axes_are_logarithmic(site_browser):
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "calibration.html#skew-term", wait_until="load")
        page.locator("#chart-skew-term").scroll_into_view_if_needed()
        _wait_for_chart(page, "chart-skew-term")
        types = page.evaluate("""() => { const o = Desk.charts['chart-skew-term'].chart.getOption();
          return [o.xAxis[0].type, o.yAxis[0].type, o.xAxis[0].name, o.yAxis[0].name]; }""")
        assert types[:2] == ["log", "log"], types
        assert "years" in types[2]
        assert "ln(K/F)" in types[3]
    finally:
        ctx.close()


def _cm_of(ds, s: dict) -> list[dict]:
    return ds.constant_maturity(ds.delta_grid(s["surface"], s["quotes"], s["vix"]))


def test_surface_change_against_every_earlier_session_states_its_gap(dash, pages):
    """P2.3: the headline's constant-maturity grid against each earlier stored
    session (newest first, the newest selected), each table headed with the gap
    in NYSE sessions (session_age) and its Δ cells the differences of the two
    grids desk_surface computes from the artifacts."""
    ds = _desk_module()
    html = pages["/surface"]
    inputs = dash.artifacts.load_surface_desk_inputs(SHIPPED)
    head = _surface_file("surface.json").parent.name
    earlier = sorted((s["session"] for s in inputs if s["session"] < head), reverse=True)
    sec = re.search(r'id="surface-change".*?</section>', html, re.DOTALL).group(0)
    tabs = re.findall(r'role="tab" aria-selected="(true|false)" aria-controls="vs-([\d-]+)"', sec)
    assert [t for _, t in tabs] == earlier
    assert [t for sel, t in tabs if sel == "true"] == earlier[:1]
    assert 'data-chart-filter="chart-atm-ts chart-rr-term"' in sec
    new = _cm_of(ds, next(s for s in inputs if s["session"] == head))
    for old in earlier:
        cap = _html_unescape(re.search(rf'id="vs-table-{old}">\s*<caption>(.*?)</caption>', sec,
                                       re.DOTALL).group(1))
        gap = dash.session_age(old, head)
        assert f"vs {old}, {dash.plural(gap, 'NYSE session')} earlier" in re.sub(r"<[^>]+>", "", cap)
        rows = _table_rows(sec, f"vs-table-{old}")
        delta = ds.cm_delta(new, _cm_of(ds, next(s for s in inputs if s["session"] == old)))
        assert [r[0] for r in rows] == [r["label"] for r in delta]
        for cells, r in zip(rows, delta):
            assert cells[2] == dash.signed(r["atmf"]["delta"], ".2f")
            assert cells[4] == dash.signed(r["rr25"]["delta"], ".2f")
            assert cells[6] == dash.signed(r["bf25"]["delta"], ".2f")
    # the term charts name both dates
    atm = {it["name"] for it in _spec(html, "chart-atm-ts")["legend_items"]}
    rr = {it["name"] for it in _spec(html, "chart-rr-term")["legend_items"]}
    assert f"SVI-cal ATM {head}" in atm
    assert f"SVI-cal RR25 {head}" in rr
    for old in earlier:
        assert f"SVI-cal ATM {old}" in atm
        assert f"SVI-cal RR25 {old}" in rr


def test_the_surface_page_never_calls_a_session_change_a_day_or_the_prior_session(pages):
    """The stored surface sessions are irregular (March to October): a change
    between two of them is labelled with its gap, never as a day's or the prior
    session's. (The Overview's 'percentile of N prior sessions' is a history.)"""
    html = pages["/surface"]
    text = _visible_text(html).lower()
    for bad in (r"(?<!\d)1-day", r"\bone-day", r"\bprior session", r"\bprevious session", r"day-over-day"):
        assert not re.search(bad, text), bad
    assert not re.search(r"(?<!\d)1-day|prior session", html, re.IGNORECASE)


def test_a_tenor_the_earlier_session_lacks_prints_a_dash(dash):
    ds = _desk_module()

    def surface(days):
        sl = [{"a": 0.0004 * t / 30, "b": 0.05, "rho": -0.6, "m": 0.0, "sigma": 0.12, "T": t / 365,
               "expiry": f"2030-{i + 1:02d}-01"} for i, t in enumerate(days)]
        sj = {"surfaces": {"svi_cal": {"params": {"slices": sl}}},
              "term_structure": [{"expiry": s["expiry"], "T_days": s["T"] * 365} for s in sl]}
        res = pd.DataFrame([{"model": "svi_cal", "expiry": s["expiry"], "k": k} for s in sl for k in (-1, 1)])
        return ds.constant_maturity(ds.delta_grid(sj, res, None))

    v = {"session": "2030-01-01", "gap": 7, "rows": ds.cm_delta(surface((20, 45, 100, 200, 400)),
                                                               surface((20, 45, 100, 200)))}
    macros = dash.app.jinja_env.get_template("_macros.html").module
    html = str(macros.vs_table("2030-03-01", v))
    rows = {r[0]: r for r in _table_rows(html, "vs-table-2030-01-01")}
    assert rows["1Y"][2] == rows["1Y"][4] == rows["1Y"][6] == "—"
    assert rows["1Y"][1].startswith("— → ")
    assert rows["3M"][2] != "—"


def test_browser_session_tabs_switch_the_table_and_both_charts_without_a_request(site_browser):
    ctx, page = _page(site_browser, 1440, 900)
    try:
        page.goto(site_browser["base"] + "surface.html", wait_until="load")
        tabs = page.locator('#surface-change [role="tab"]')
        n = tabs.count()
        assert n >= 1
        last = tabs.nth(n - 1).get_attribute("data-value")
        page.locator("#chart-atm-ts").scroll_into_view_if_needed()
        _wait_for_chart(page, "chart-atm-ts")
        page.locator("#chart-rr-term").scroll_into_view_if_needed()
        _wait_for_chart(page, "chart-rr-term")
        requests = []
        page.on("request", lambda r: requests.append(r.url))
        tabs.nth(n - 1).click()
        page.wait_for_timeout(300)
        st = page.evaluate("""() => {
          const shown = [...document.querySelectorAll('#surface-change [role=tabpanel]')]
            .filter(p => !p.hidden).map(p => p.id);
          const sel = id => { const o = Desk.charts[id].chart.getOption(), s = o.legend[0].selected;
            return Object.keys(s).filter(k => s[k]); };
          const btns = id => [...document.querySelectorAll('[data-legend-for="' + id + '"][data-group]')]
            .filter(b => !b.hidden).map(b => b.dataset.group);
          return {shown, atm: sel('chart-atm-ts'), rr: sel('chart-rr-term'), atmBtns: btns('chart-atm-ts'),
                  rrBtns: btns('chart-rr-term'), hash: location.hash}; }""")
        assert st["shown"] == [f"vs-{last}"], st
        assert f"SVI-cal ATM {last}" in st["atm"]
        assert not [s for s in st["atm"] if s.startswith("SVI-cal ATM ") and s != f"SVI-cal ATM {last}"
                    and s.split()[-1] < last], st["atm"]
        assert f"SVI-cal RR25 {last}" in st["rr"]
        assert st["atmBtns"] == [last]
        assert st["rrBtns"] == [last]
        assert f"vs={last}" in st["hash"]
        assert requests == []          # every session's data is in the page
    finally:
        ctx.close()


def test_atm_term_forward_steps_and_a_calendar_flag_marker(dash, pages):
    """P2.4: the ATM chart draws the forward ATM vol between the constant
    maturities as steps (the values desk_surface computes); a negative forward
    variance is a 'calendar' flag marker, never a NaN in the spec."""
    ds = _desk_module()
    html = pages["/surface"]
    spec = _spec(html, "chart-atm-ts")
    step = next(s for s in spec["series"] if s["name"].startswith("SVI-cal forward ATM vol"))
    fw = ds.forward_vols(ds.constant_maturity(ds.delta_grid(*_desk_inputs())))
    want = [[f["t1_days"], f["fwd_vol"]] for f in fw]
    got = step["data"][0::2]
    assert [x for x, _ in got] == pytest.approx([x for x, _ in want], rel=1e-5)
    assert [y for _, y in got] == pytest.approx([y for _, y in want], rel=1e-5)
    note = _visible_text("<body>" + re.search(r'id="forward-vols">(.*?)</p>', html, re.DOTALL).group(1))
    for f in fw:
        assert f"{f['from']} → {f['to']} {dash._fmt(f['fwd_vol'], '.2f')}%" in note
    legend = {it["name"] for it in spec["legend_items"]}
    assert step["name"] in legend
    # a synthetic falling total variance: a flag item, no NaN anywhere
    bad = ds.forward_vols([{"label": "1M", "T": 30 / 365, "atmf": 20.0},
                           {"label": "2M", "T": 61 / 365, "atmf": 13.0}])
    series = dash.forward_vol_series(bad)
    flag = [s for s in series if s["type"] == "scatter"]
    assert len(flag) == 1
    assert flag[0]["name"].startswith("Calendar flag")
    assert all(v is None or np.isfinite(v) for s in series for pt in s["data"] for v in pt)
    assert "NaN" not in json.dumps(series)


def test_varswap_strip_marks_the_four_official_closes(dash, pages):
    html = pages["/surface"]
    spec = _spec(html, "chart-strip")
    mk = [s for s in spec["series"] if s["name"].startswith("Cboe close (16:15 ET)")]
    assert len(mk) == 1
    assert mk[0]["type"] == "scatter"
    assert mk[0].get("hollow") is True
    closes = [r[2] for r in _table_rows(html, "vix-rep-table") if r[2] != "—"]
    assert len(closes) == 4
    assert [dash._fmt(y, ".2f") for _, y in mk[0]["data"]] == closes
    assert [x for x, _ in mk[0]["data"]] == [9, 30, 93, 184]
    legend = {it["name"] for it in spec["legend_items"]}
    assert {"Variance-swap vol (Cboe formula)", mk[0]["name"]} <= legend
    assert re.search(r'id="vix-markers">', html)



def test_vix_terms_table_prints_each_index_terms_and_its_gap_split(dash, pages):
    """P2.7: the replication panel stood beside the taller variance-swap strip
    with its bottom blank. It now lists, per index of the session the page
    renders, the two listed expiries the replication interpolates and the near
    weight, and the gap split as vix_replication.json's gap_attribution splits
    it (gap = −truncation_pts − residual_pts); read here from the artifact."""
    html = pages["/surface"]
    session = dash.load_surface()["session"]
    vix = json.loads((SHIPPED / "surface" / session / "vix_replication.json").read_text())
    rows = _table_rows(html, "vix-terms-table")
    gaps = {r[0]: r[3] for r in _table_rows(html, "vix-rep-table")}
    assert [r[0] for r in rows] == [t for t in ("VIX9D", "VIX", "VIX3M", "VIX6M") if t in vix["indices"]]

    def days(x: float) -> str:
        return dash._fmt(x, ".0f") if abs(x - round(x)) < 1e-6 else dash._fmt(x, ".1f")

    def num(txt: str) -> float:
        return float(txt.replace("−", "-").replace("+", ""))

    n_split = 0
    for r in rows:
        ix = vix["indices"][r[0]]
        m = ix.get(ix.get("headline_method") or "cboe_rules") or {}
        # an index the chain cannot replicate may record no terms at all ("fewer
        # than two usable terms"): the row then prints a dash, as the page does
        if m.get("near") and m.get("next"):
            assert r[1] == f"{days(m['near']['days'])} / {days(m['next']['days'])}", r
        else:
            assert ix.get("replicated") is None, r
            assert r[1] == "—", r
        assert r[2] == dash._fmt(m.get("near_weight"), ".3f"), r
        ga = ix.get("gap_attribution") or {}
        if ix.get("replicated") is None or not ga.get("available"):
            assert r[3] == r[4] == "—", r
            continue
        n_split += 1
        assert r[3] == dash.signed(-ga["truncation_pts"], ".2f"), r
        assert r[4] == dash.signed(-ga["residual_pts"], ".2f"), r
        # the two parts add up to the gap beside them, up to each one's rounding
        assert abs(num(r[3]) + num(r[4]) - num(gaps[r[0]])) <= 0.011, (r, gaps[r[0]])
    note = _visible_text("<body>" + re.search(r'<p class="note" id="vix-terms-note">.*?</p>', html,
                                              re.DOTALL).group(0))
    for t, ix in vix["indices"].items():
        assert (f"{t} is not replicated" in note) == (ix.get("replicated") is None), t
        m = ix.get(ix.get("headline_method") or "cboe_rules") or {}
        if ix.get("replicated") is None and not m.get("extrapolated") and m.get("reason"):
            assert f"{t} is not replicated: {m['reason']}." in note, t
    if n_split:
        assert "Missing-strike shortfall: the replication on the stored strip less the same" in note
        bias = [ix["gap_attribution"]["far_wing_market_minus_model_pts"] for ix in vix["indices"].values()
                if ix.get("replicated") is not None and (ix.get("gap_attribution") or {}).get("available")]
        assert ("likely understates" in note) == any(b > 0 for b in bias)
        assert ("likely overstates" in note) == any(b < 0 for b in bias)


def test_vix_replication_prose_gives_the_gap_split_with_the_tables_signs(dash, pages):
    """R3-T7: under the terms table (gap = replicated − Cboe close) the method
    prose gave the same split as official − replicated, each part with the
    opposite sign. The page's sentence now carries the table's own VIX row, and
    the column names its convention."""
    html = pages["/surface"]
    session = dash.load_surface()["session"]
    vix = json.loads((SHIPPED / "surface" / session / "vix_replication.json").read_text())
    head = re.search(r'<table[^>]*id="vix-terms-table".*?</thead>', html, re.DOTALL).group(0)
    assert "Missing-strike shortfall (replicated − full strip)" in _visible_text("<body>" + head)
    prose = _visible_text("<body>" + re.search(r'<ul class="prose"[^>]*>.*?</ul>', html[html.index(
        'id="vix-terms-table"'):], re.DOTALL).group(0))
    assert "official − replicated" not in prose
    assert "official - replicated" not in prose
    row = {r[0]: r for r in _table_rows(html, "vix-terms-table")}.get("VIX")
    ga = (vix["indices"].get("VIX") or {}).get("gap_attribution") or {}
    if row is None or row[3] == "—":
        assert "missing-strike shortfall" not in prose
        return
    assert ga.get("available")
    gap = {r[0]: r[3] for r in _table_rows(html, "vix-rep-table")}["VIX"]
    assert f"{gap} = missing-strike shortfall {row[3]} + unexplained {row[4]}" in prose
    assert f"Unexplained by the chain: {row[4]}," in prose


def test_vix_terms_table_renders_a_partial_artifact_without_inventing_a_split(dash):
    """The terms table on rows a future replication may write: an index with no
    terms and no replication, a split without a far-wing figure (it was
    compared with 0 and broke the page), and an AM term's fractional days.
    Missing values print '—', and the note speaks only of what the rows hold."""
    macros = dash.app.jinja_env.get_template("_macros.html").module
    none = {"replicated": False, "extrapolated": False, "near": None, "next": None, "near_weight": None,
            "gap": None, "truncation": None, "unexplained": None, "far_wing": None, "surface": None,
            "surface_label": None, "reason": None}
    rows = [{**none, "ticker": "VIX9D", "reason": "fewer than two usable terms"},
            {**none, "ticker": "VIX", "replicated": True, "near": {"expiry": "2030-01-28", "days": 28.0},
             "next": {"expiry": "2030-02-18", "days": 49.0}, "near_weight": 19 / 21, "gap": -0.30,
             "truncation": -0.05, "unexplained": -0.25, "surface": "svi_cal", "surface_label": "SVI-cal"},
            {**none, "ticker": "VIX6M", "replicated": True,
             "near": {"expiry": "2030-07-19", "days": 181.7291667},
             "next": {"expiry": "2030-08-16", "days": 216.7291667}, "near_weight": 0.935, "gap": -0.10,
             "truncation": -0.01, "unexplained": -0.09, "far_wing": -0.40, "surface": "svi_cal",
             "surface_label": "SVI-cal"}]
    html = str(macros.vix_terms(rows))
    got = _table_rows(html, "vix-terms-table")
    assert got[0] == ["VIX9D", "—", "—", "—", "—"]
    assert got[1] == ["VIX", "28 / 49", "0.905", "−0.05", "−0.25"]
    assert got[2] == ["VIX6M", "181.7 / 216.7", "0.935", "−0.01", "−0.09"]
    note = _visible_text("<body>" + html)
    assert "VIX9D is not replicated: fewer than two usable terms." in note
    assert "SVI-cal surface’s wing" in note
    assert "The stored quotes beyond the surface’s fitted range price below that wing (VIX6M −0.40" in note
    assert "likely understates" not in note
    assert "None" not in note

def test_chain_badge_names_the_source_and_the_quote_time(dash):
    """P2.5: the badge is generic: a Yahoo snapshot says so and that it has no
    quote time; a Cboe-sourced chain prints 'Cboe' and its quote time."""
    yahoo = {"inputs": {"source": "yfinance", "quote_time": None, "chain_rows": 6191},
             "universe": {"n_universe": 2913, "n_expiries": 17}}
    assert dash.chain_badge(yahoo, [43, 40, 41, 43]) == (
        "Chain: Yahoo Finance close snapshot · no quote time · 6,191 rows → 2,913 scored · "
        "17 expiries (other stored sessions 40–43)")
    cboe = {"inputs": {"source": "cboe", "quote_time": "2026-10-23T15:59:30-04:00", "chain_rows": 9000},
            "universe": {"n_universe": 4000, "n_expiries": 30}}
    txt = dash.chain_badge(cboe, [17])
    assert txt.startswith("Chain: Cboe · quotes at 2026-10-23 15:59:30-04:00 · 9,000 rows → 4,000 scored")
    assert txt.endswith("30 expiries (other stored session 17)")
    assert "Yahoo" not in txt
    assert "no quote time" not in txt


def test_surface_and_calibration_badges_and_the_expiries_column_read_the_artifacts(dash, pages):
    inputs = dash.artifacts.load_surface_desk_inputs(SHIPPED)
    head = _surface_file("surface.json").parent.name
    sj = next(s["surface"] for s in inputs if s["session"] == head)
    others = [dash.session_expiries(s["surface"]) for s in inputs if s["session"] != head]
    want = dash.chain_badge(sj, others)
    for route in ("/surface", "/calibration"):
        badge = re.search(r'<div class="page-badge" id="chain-badge">(.*?)</div>', pages[route], re.DOTALL)
        if route == "/calibration" and dash.load_page2().get("session") != head:
            continue                       # the fit's own session's chain, when it has a surface report
        assert badge, route
        text = re.sub(r"\s+", " ", _visible_text("<body>" + badge.group(1))).strip()
        assert text == want, route
        if sj["inputs"]["source"] == "yfinance" and not sj["inputs"].get("quote_time"):
            assert "Yahoo" in text
            assert "no quote time" in text
    heads = _table_heads(pages["/surface"], "surface-history-table")
    col = heads.index("Expiries")
    rows = _table_rows(pages["/surface"], "surface-history-table")
    by = {s["session"]: dash.session_expiries(s["surface"]) for s in inputs}
    assert len(rows) == len(by)
    for r in rows:
        assert r[col] == str(by[r[0]]), r[0]


def _widest_strike_gap(legs: list[dict]) -> float:
    ks = sorted({float(lg["strike"]) for lg in legs})
    return max((b - a for a, b in itertools.pairwise(ks)), default=0.0)


def test_arbitrage_tile_names_the_largest_violation_and_the_table_lists_them(dash, pages):
    """R3-T4: the tile read 'Executable static arbitrage … Largest mid violation
    95.0 idx pts ($9,497 per contract), C2 vertical spread, 2027-03-31 P
    6260/6265': a 5-point vertical priced 95 points through its bound by one bad
    Yahoo quote (P6265 151.4/154.1 between P6260 53.5/54.3 and P6270 62.6/65.1),
    and 24 of the 25 listed rows were tagged 'executable'. A row whose shortfall
    is larger than its widest strike gap now reads 'bad quote: <leg>' with no
    dollar value, and the tile headlines the largest row without one."""
    html = pages["/surface"]
    arb = json.loads(_surface_file("surface.json").read_text())["arbitrage"]
    tile_m = re.search(r'id="arb-tile">(.*?)</div>\s*</div>', html, re.DOTALL)
    tile = _visible_text("<body>" + tile_m.group(1))
    assert "Quotes failing static arbitrage at bid/ask" in tile
    assert "Executable" not in tile
    assert "constraint row" in tile
    assert "no quote time" in tile or "quotes at" in tile
    assert 'href="#largest-violations"' in html
    rows = _table_rows(html, "largest-violations")
    assert len(rows) == len(arb["largest_mid_violations"])
    ordered = sorted(arb["largest_mid_violations"], key=lambda v: -v["shortfall_index_points"])
    named = set()
    for r, v in zip(rows, ordered):
        assert r[2] == f"{v['shortfall_index_points']:.2f}"
        for lg in v["legs"]:
            assert f"{lg['right']}{lg['strike']:.0f}" in r[1]
            assert str(lg["expiry"])[:10] in r[1]
        if r[4].startswith("bad quote: "):
            assert r[3] == "—"
            exp, leg = r[4][len("bad quote: "):].split()
            assert any(str(lg["expiry"])[:10] == exp and f"{lg['right']}{lg['strike']:.0f}" == leg
                       for lg in v["legs"]), r
            named.add(r[4])
        else:
            # no row wider in shortfall than its strike gap passes as tradeable
            assert not 0.0 < _widest_strike_gap(v["legs"]) < v["shortfall_index_points"], r
            assert r[3] == dash.usd_desk(v["shortfall_index_points"] * 100, signed=False)
            assert r[4] == ("executable" if v["executable"] else "mids only")
    # every flagged row that holds a named leg carries it: flagged rows share their leg
    for r, v in zip(rows, ordered):
        held = {f"bad quote: {str(lg['expiry'])[:10]} {lg['right']}{lg['strike']:.0f}" for lg in v["legs"]}
        assert not (held & named) or r[4] in held & named, r
    n_bad = sum(1 for r in rows if r[4].startswith("bad quote: "))
    assert f"{n_bad} of the {len(rows)} largest rows from single-quote outliers" in tile
    clean = next((v for r, v in zip(rows, ordered) if not r[4].startswith("bad quote: ")), None)
    if clean:
        sf = clean["shortfall_index_points"]
        assert (f"Largest mid violation without an outlier leg: {sf:.1f} idx pts "
                f"({dash.usd_desk(sf * 100, signed=False)} per contract)") in tile
    else:
        assert "Largest mid violation" not in tile
    # the panel table counts constraint rows, and says so
    assert _table_heads(html, "arb-table")[1:] == ["Constraint rows", "Violated at mids",
                                                   "Violated at bid/ask"]


def test_bad_quote_flag_names_the_leg_the_impossible_rows_share(dash):
    """flag_bad_quotes on the 2026-10-16 shape: one off quote (P6265) breaks a
    vertical, a butterfly and a calendar spread by more than their strike gaps,
    and a wide calendar butterfly that holds it is flagged with it; a violation
    inside its strike gap and a one-strike calendar row stay tradeable."""
    def row(typ, sf, *legs):
        return dash.violation_view({"type": typ, "shortfall_index_points": sf, "executable": True,
                                    "legs": [{"expiry": e, "right": "P", "strike": k, "weight": w}
                                             for e, k, w in legs]})
    t1, t2 = "2027-03-31", "2027-04-16"
    views = [row("C2_vertical_spread", 95.0, (t1, 6260.0, 1.0), (t1, 6265.0, -1.0)),
             row("C3_vertical_butterfly", 94.9, (t1, 6260.0, 0.5), (t1, 6265.0, -1.0), (t1, 6270.0, 0.5)),
             row("C5_calendar_vertical_spread", 79.0, (t1, 6265.0, -1.0), (t2, 6275.0, 1.0)),
             row("C6_calendar_butterfly", 58.0, (t1, 6265.0, -1.0), (t2, 6200.0, 0.5), (t2, 6400.0, 0.5)),
             row("C2_vertical_spread", 20.0, (t1, 7000.0, 1.0), (t1, 7010.0, -1.0)),
             row("C3_vertical_butterfly", 9.4, (t1, 6245.0, 1 / 3), (t1, 6255.0, -1.0), (t1, 6260.0, 2 / 3)),
             row("C4_calendar_spread", 3.0, (t1, 6500.0, -1.0), (t2, 6500.0, 1.0))]
    moved = [{"expiry": t1, "right": "P", "strike": 7010.0, "repair_dprice": -15.0}]
    out = dash.flag_bad_quotes(views, moved)
    assert [v["bad_quote"] for v in out] == [f"{t1} P6265"] * 4 + [f"{t1} P7010", None, None]
    # a lone two-leg row with no repair move either way still names one of its own legs
    lone = dash.flag_bad_quotes([row("C2_vertical_spread", 20.0, (t1, 7000.0, 1.0), (t1, 7010.0, -1.0))])
    assert lone[0]["bad_quote"] in (f"{t1} P7000", f"{t1} P7010")


def _kpi_blocks(html: str, block_id: str) -> dict[str, str]:
    """{label: the tile's markup} of one KPI strip."""
    start = html.index(f'id="{block_id}"')
    ends = [i for i in (html.find('<div class="kpis"', start + 1), html.find('<div class="g"', start))
            if i > 0]
    blk = html[start:min(ends)]
    out = {}
    for tile in re.split(r'(?=<div class="kpi"[ >])', blk)[1:]:
        lab = re.search(r'<div class="kpi-label">(.*?)</div>', tile, re.DOTALL)
        if lab:
            out[_html_unescape(re.sub(r"<[^>]+>", "", lab.group(1))).strip()] = tile
    return out


def _range_span(tile: str) -> str | None:
    m = re.search(r'<div class="kpi-ci kpi-range"[^>]*>(.*?)</div>', tile, re.DOTALL)
    return _html_unescape(m.group(1)).strip() if m else None


def test_surface_model_tiles_carry_their_range_over_every_stored_session(dash, pages):
    """P2.6: every Surface tile of a per-session model metric shows that model's
    out-of-sample median |z| over every stored session (each scorecard.json),
    marked as a range across sessions."""
    html = pages["/surface"]
    oos = dash.artifacts.surface_oos_by_session(SHIPPED)
    tiles = _kpi_blocks(html, "surface-scores")
    model_tiles = {k: v for k, v in tiles.items() if k.endswith("· median |z|")}
    assert len(model_tiles) >= 4
    score_rows = dash.load_surface()["score_rows"]
    for label, tile in model_tiles.items():
        key = next(r["key"] for r in score_rows if r["short"] + " · median |z|" == label)
        xs = [r["median_abs_hs"] for r in oos if r["model"] == key]
        span = _range_span(tile)
        assert span == f"{len(xs)} sessions: {min(xs):.2f}–{max(xs):.2f} (out of sample)", label
        assert "A range across sessions, not a sampling interval" in tile
    for label, tile in _kpi_blocks(html, "surface-state").items():
        assert "sessions:" in (_range_span(tile) or ""), label


def test_calibration_fit_tiles_rank_this_fit_among_the_gated_fits(dash, pages):
    """P2.6: each fit-quality tile says where this fit ranks among the gated
    fits and their range (the OOS tile on 2026-10-16: 'worst of 5'), at the
    site's one precision for vol-pt RMSE (.2f)."""
    html = pages["/calibration"]
    tiles = _kpi_blocks(html, "fit-quality")
    hist = dash._gated_history()
    d = dash.load_page2()
    own = next(h for h in hist if h["session"] == d["session"])
    for label, key in (("SPX IV RMSE", "rmse"), ("Out of sample", "oos"), ("Put wing OOS", "wing"),
                       ("VIX index RMSE", "vix")):
        xs = [h[key] for h in hist]
        better = sum(1 for x in xs if x < own[key] - 1e-12)
        word = "best" if better == 0 else "worst" if better == len(xs) - 1 else dash.ordinal(better + 1)
        span = _range_span(tiles[label])
        assert span == f"{word} of {len(xs)} gated sessions ({min(xs):.2f}–{max(xs):.2f})", label
        value = re.search(r'<div class="kpi-value[^"]*">([^<]*)', tiles[label]).group(1)
        assert value == f"{own[key]:.2f}", label
    if max(h["oos"] for h in hist) == own["oos"]:
        assert _range_span(tiles["Out of sample"]).startswith(f"worst of {len(hist)}")
    if "VVIX model / market" in tiles and d.get("vvix"):
        assert "diagnosed sessions" in (_range_span(tiles["VVIX model / market"]) or "")


def test_fit_losses_reach_the_tiles_unrounded(dash):
    """The tiles print .2f; load_page2 pre-rounded the losses to .3f, so the
    2026-10-26 put-wing OOS RMSE 2.30497 became 2.305 and printed 2.31 beside
    its rank range (2.30)."""
    _, cal = dash.load_headline_calibration()
    losses = dash._cal_losses(cal)
    d = dash.load_page2()
    for key, src in (("spx_rmse", "spx_iv_rmse"), ("spx_oos_rmse", "spx_oos_iv_rmse"),
                     ("spx_oos_wing_rmse", "spx_oos_put_wing_rmse"), ("vix_index_rmse", "vix_index_rmse")):
        if dash._num(losses.get(src)) is not None:
            assert d[key] == float(losses[src]), key


def test_one_rendering_of_the_heston_oos_rmse_and_the_svi_cal_median_z_on_every_page(dash, pages):
    """P2.6: Calibration printed the Heston OOS RMSE as 2.955 where the Overview
    and Model Risk print 2.96. Every page renders it, and the SVI-cal median |z|,
    at one precision."""
    _, cal = dash.load_headline_calibration()
    heston = dash._cal_losses(cal)["spx_oos_iv_rmse"]
    sc = json.loads(_surface_file("scorecard.json").read_text())
    svi_m = next(m for m in sc["models"] if m["model"] == "svi_cal")
    svi = svi_m["metrics"]["out_of_sample"]["median_abs_hs"]
    for name, v, must in (("Heston OOS RMSE", heston, ("/", "/calibration", "/model-risk")),
                          ("SVI-cal median |z|", svi, ("/", "/surface"))):
        seen = {}
        for route, html in pages.items():
            text = _visible_text(html)
            found = {f"{v:.{p}f}" for p in (2, 3, 4)
                     if re.search(r"(?<![\d.])" + re.escape(f"{v:.{p}f}") + r"(?!\d)", text)}
            if found:
                seen[route] = found
        assert set(must) <= set(seen), (name, seen)
        assert set().union(*seen.values()) == {f"{v:.2f}"}, (name, seen)


def test_surface_history_heads_name_their_population(dash, pages):
    heads = _table_heads(pages["/surface"], "surface-history-table")
    models = [h for h in heads if "|z|" in h]
    assert len(models) >= 4
    assert all(h.endswith("|z|, out of sample") for h in models)
    assert "Scored quotes" in heads
    assert "Quotes" not in heads
    cap = re.search(r'id="surface-history-table"><caption>(.*?)</caption>', pages["/surface"], re.DOTALL)
    assert "out-of-sample quotes" in _html_unescape(cap.group(1))
    assert "all scored quotes" not in _html_unescape(cap.group(1))
    oos = dash.artifacts.surface_oos_by_session(SHIPPED)
    rows = _table_rows(pages["/surface"], "surface-history-table")
    history_models = dash.load_surface()["history_models"]
    for r in rows:
        for h, cell in zip(heads, r):
            if "|z|" in h:
                key = next(m["key"] for m in history_models if h.startswith(m["label"] + " "))
                v = next((x["median_abs_hs"] for x in oos
                          if x["session"] == r[0] and x["model"] == key), None)
                assert cell == dash._fmt(v, ".2f"), (r[0], h)


# ── P2.7: no half-empty grid rows, no blank panel bottoms ────────────────────

def test_every_row_of_two_unequal_panels_is_marked_as_a_pair(pages):
    """P2.7: below 1181 px an s7 + s5 (or s8 + s4) row became 12 + 6, a
    half-empty row. Every grid row of two panels with unequal spans now carries
    the template-set .pair class (no :has() dependency), and a row of three s4
    panels .trio, so desk.css can stack or split them."""
    n = 0
    for route, html in pages.items():
        # each grid row runs to the next one (a panel is a <section>; no section sits between rows)
        for chunk in re.split(r'(?=<div class="g[ "])', html)[1:]:
            cls = re.match(r'<div class="([^"]*)"', chunk).group(1).split()
            body = chunk.split("</main>")[0]
            spans = re.findall(r'<section class="panel (s\d+)', body)
            spans = [s for s in spans if s != "s12"]
            if len(spans) == 2 and spans[0] != spans[1]:
                assert "pair" in cls, (route, spans, body[:200])
                n += 1
            if len(spans) == 3 and set(spans) == {"s4"}:
                assert "trio" in cls, (route, spans)
    assert n >= 10


_LAYOUT_JS = """([maxUnused, maxBlank, exempt]) => {
  const vis = el => { const b = el.getBoundingClientRect(), s = getComputedStyle(el);
    return b.width > 0 && b.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  const shut = el => { const d = el.closest('details:not([open])'), s = el.closest('summary');
    return !!d && d !== el && !(s && s.parentElement === d); };
  const holes = [], blanks = [];
  document.querySelectorAll('.g').forEach(g => {
    const gap = parseFloat(getComputedStyle(g).columnGap) || 0, rows = {};
    [...g.children].filter(vis).forEach(k => {
      const t = Math.round(k.getBoundingClientRect().top); (rows[t] = rows[t] || []).push(k); });
    Object.values(rows).forEach(ks => {
      const used = ks.reduce((a, k) => a + k.getBoundingClientRect().width, 0) + gap * (ks.length - 1);
      const unused = g.getBoundingClientRect().width - used;
      if (unused > maxUnused) holes.push(ks.map(k => k.id).join('+') + ' ' + Math.round(unused));
    });
  });
  document.querySelectorAll('.panel').forEach(p => {
    if (!vis(p) || exempt.includes(p.id)) return;
    let last = -Infinity;
    p.querySelectorAll('.panel-body *').forEach(el => {
      if (!vis(el) || el.closest('[hidden]') || el.closest('.sr-only') || shut(el)) return;
      last = Math.max(last, el.getBoundingClientRect().bottom);
    });
    if (last !== -Infinity && p.getBoundingClientRect().bottom - last > maxBlank)
      blanks.push(p.id + ' ' + Math.round(p.getBoundingClientRect().bottom - last));
  });
  return {holes, blanks};
}"""


@pytest.mark.parametrize("width", [1024, 1440, 1920])
def test_browser_grid_rows_are_full_and_panels_end_with_their_content(site_browser, width):
    """P2.7 (also in scripts/ci/a11y/interact.mjs at 1024, 1280, 1440 and 1920 px):
    no grid row leaves more than 60 px unused, and no panel ends more than 96 px
    below its last visible content once its charts are drawn (#carry and #forward
    have their own fixes in flight). The body of a closed <details> is not
    visible content: Chromium keeps a layout box for it, and counting it hid
    the blank below the replication panel's closed disclosure (135 px at 1440)
    and the scenario grid's (111 px at 1920)."""
    ctx, page = _page(site_browser, width, 900)
    try:
        for name in ("index.html", "surface.html", "calibration.html", "risk.html", "research.html",
                     "corrections.html", "model-risk.html"):
            page.goto(site_browser["base"] + name, wait_until="load")
            h = page.evaluate("document.documentElement.scrollHeight")
            for y in range(0, h, 700):
                page.evaluate(f"window.scrollTo(0, {y})")
                page.wait_for_timeout(40)
            page.wait_for_timeout(300)
            r = page.evaluate(_LAYOUT_JS, [60, 96, ["carry", "forward"]])
            assert r == {"holes": [], "blanks": []}, (name, width, r)
    finally:
        ctx.close()


def test_browser_heatmap_row_head_stays_in_view_on_a_phone(site_browser):
    ctx, page = _page(site_browser, 390, 844)
    try:
        page.goto(site_browser["base"] + "surface.html", wait_until="load")
        st = page.evaluate("""async () => {
          const wrap = document.getElementById('iv-heatmap').closest('.heat-wrap');
          wrap.scrollIntoView({block: 'center'});
          const before = wrap.querySelector('tbody th[scope=row]').getBoundingClientRect().left;
          wrap.scrollLeft = 200;
          await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
          const th = wrap.querySelector('tbody th[scope=row]').getBoundingClientRect();
          return {scrolled: wrap.scrollLeft, before, left: th.left, right: th.right,
                  wrapLeft: wrap.getBoundingClientRect().left, vw: innerWidth}; }""")
        assert st["scrolled"] > 100, st
        assert st["left"] >= st["wrapLeft"] - 1, st
        assert st["right"] <= st["vw"], st
        assert abs(st["left"] - st["before"]) < 2, st
    finally:
        ctx.close()


# ── P3.1: the joint fit's VX futures leg ─────────────────────────────────────

def _vx_last() -> str | None:
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        return con.execute("SELECT MAX(date) FROM vix_futures_daily").fetchone()[0]


def _gated_pickles(dash) -> list[tuple[str, dict]]:
    out = []
    for r in dash.scan_calibrations():
        if r["legacy"] is False and r["ok"] and r["session"]:
            with open(r["path"], "rb") as f:
                out.append((r["session"], pickle.load(f)))
    return sorted(out, key=lambda x: x[0])


def test_vx_leg_is_each_fits_own_out_of_sample_score(dash, pages):
    """P3.1: Calibration #vixleg redraws every gated fit's VX futures leg from the
    settlements of its own session: the RMSE it prints is the fit's recorded
    losses.vx_futures_rmse_oos (to 1e-6), a fit with no settlements on its
    session gets the empty state and no futures series, and nothing is spliced
    from an earlier session."""
    d = dash.load_page2()
    views = {v["session"]: v for v in d["vx_leg"]["views"]}
    last = _vx_last()
    html = pages["/calibration"]
    gated = _gated_pickles(dash)
    assert sorted(views) == [s for s, _ in gated]
    for session, cal in gated:
        v, rec = views[session], dash._cal_losses(cal).get("vx_futures_rmse_oos")
        a = html.index(f'<div id="vx-{session}" role="tabpanel"')
        panel = html[a:min(i for i in (html.find('<div id="vx-', a + 10), html.find('id="vx-caption"', a))
                           if i > 0)]
        if v["leg"] is not None:
            assert rec is not None, session
            assert abs(v["leg"]["rmse"] - rec) <= 1e-6, session
            assert f'id="chart-vx-{session}"' in panel
            assert f"RMSE {rec:.2f} index pts" in _html_unescape(panel)
        else:
            assert f"chart-vx-{session}" not in html, session
            assert v["empty"] in _html_unescape(re.sub(r"</?span[^>]*>", "", panel))
            if last is not None and session > last:
                assert v["empty"] == f"VX futures table ends {last}; none stored for {session}."
    # the headline fit is the newest gated session; it has a leg only if its session has settlements
    assert d["vx_leg"]["n_with"] == sum(1 for v in views.values() if v["leg"])


def test_gated_fits_table_has_the_vx_out_of_sample_column(dash, pages):
    """P3.1: #history gains 'VX OOS (index pts)', each row the fit's own record or '—'."""
    html = pages["/calibration"]
    hist = html[html.index('id="history"'):html.index('id="history"') + 20000]
    assert "VX OOS (index pts)" in hist
    got = re.findall(r'<td class="num" data-vx-oos>(.*?)</td>', hist)
    want = []
    for _session, cal in reversed(_gated_pickles(dash)):           # the table is newest first
        v = dash._cal_losses(cal).get("vx_futures_rmse_oos")
        want.append("—" if v is None else f"{v:.2f}")
    assert got == want


def test_model_risk_checks_the_vx_futures_table_lag(dash, pages):
    """P3.1: a Model Risk check on the VX futures table's lag behind the market
    DB, in NYSE sessions, with the existing label-lag limits (no new threshold)."""
    d = dash.load_model_risk()
    row = next(m for m in d["monitors"]
               if m["check"] == "VX futures table lag behind the market DB (NYSE sessions)")
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        db_date = con.execute("SELECT MAX(date) FROM spx_ohlcv").fetchone()[0][:10]
    lag = dash.session_age(_vx_last(), db_date)
    assert row["model"] == "M5"
    assert row["value"] == str(lag)
    assert row["status"] == dash._rag(float(lag), dash.MONITOR_LIMITS["label_lag"])
    assert row["limits"] == "OK ≤ 0 · watch ≤ 1 · breach above"
    assert "VX futures table lag behind the market DB" in _html_unescape(pages["/model-risk"])


def test_freshness_table_lists_vx_futures_on_their_own_date(dash, pages):
    """P3.1: the footer's freshness table lists the VX futures settlements with
    their as-of date, their age in NYSE sessions and the staleness styling of
    that age. No headline reads them, so they stay out of the status bar."""
    c = dash.load_chrome()
    src = next(s for s in c["sources"] if s["short"] == "VX")
    last = _vx_last()
    assert src["date"] == last
    assert src["age"] == dash.session_age(last, c["latest"])
    assert src["level"] == dash.stale_level(src["age"])
    assert src["bar"] is False
    html = pages["/"]
    foot = html[html.index('<footer'):]
    row = re.search(r"<tr><td>VX futures settlements</td><td class=\"lvl-(\w+) nowrap\">([\d-]+)</td>\s*"
                    r'<td class="num">(\d+)</td>', foot)
    assert row, "no VX futures row in the freshness table"
    assert row.groups() == (src["level"], last, str(src["age"]))
    bar = html[html.index('<header class="statusbar"'):html.index("</header>")]
    assert ">VX<" not in bar


# ── P3.2: implied against subsequently realised vol ───────────────────────────

def _vol_frame() -> pd.DataFrame:
    """VIX, RV21 trailing and forward from the shipped DB, written out here (not desk_market)."""
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        spx = pd.read_sql("SELECT date, close FROM spx_ohlcv ORDER BY date", con, index_col="date")
        vix = pd.read_sql("SELECT date, close FROM vix_daily ORDER BY date", con, index_col="date")["close"]
    spx.index, vix.index = pd.to_datetime(spx.index), pd.to_datetime(vix.index)
    r2 = pd.Series(np.log(spx["close"].to_numpy(float)), index=spx.index).diff() ** 2
    trl = np.sqrt(r2.rolling(21).mean() * 252) * 100
    return pd.DataFrame({"vix": vix, "rv_trl": trl.reindex(vix.index),
                         "rv_fwd": trl.shift(-21).reindex(vix.index)})


def test_vrp_beta_table_is_recomputed_from_the_market_db(dash, pages):
    """P3.2: the β table of Research #vrp: mean, median, share above zero and the
    variance-space mean of β·VIX − RV21 forward over the backtest window, for
    β = 1, the backtest's β, 0.80 and the newest chain's ATM / VIX, recomputed
    here from vol_system.db."""
    html = pages["/research"]
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        st = pickle.load(f)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet", columns=["nav"])
    start, end = str(eq.index[0].date()), str(eq.index[-1].date())
    d4 = dash.load_page4()
    newest = d4["beta_est"]["later"][-1]["ratio"]
    betas = [1.0, round(float(st["atm_to_vix_ratio"]), 2), 0.80, round(float(newest), 2)]
    f = _vol_frame().loc[start:end].dropna(subset=["rv_fwd"])
    tbl = html[html.index('id="vrp-beta-table"'):]
    tbl = tbl[:tbl.index("</table>")]
    rows = re.findall(r'<tr[^>]*data-vrp-beta="([\d.]+)">(.*?)</tr>', tbl, re.DOTALL)
    assert [float(b) for b, _ in rows] == list(dict.fromkeys(betas))
    for b, cells in rows:
        beta = float(b)
        v = beta * f["vix"] - f["rv_fwd"]
        var = (beta * f["vix"]) ** 2 - f["rv_fwd"] ** 2
        got = [_html_unescape(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<td[^>]*>(.*?)</td>", cells)]
        assert got[2:] == [dash.signed(v.mean(), ".2f"), dash.signed(v.median(), ".2f"),
                           f"{(v > 0).mean() * 100:.1f}%", dash.signed(var.mean(), ".1f")], b
    assert f"{start} → {end}, {len(f):,} sessions" in _html_unescape(tbl)


def test_vrp_marks_every_s4_entry_and_says_what_it_is_not(dash, pages):
    """P3.2: the S4 entries of the trade log are marked (count and sides from
    all_trades); the caption defines both horizons and says it is not P&L."""
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        trades = [t for t in pickle.load(f)["all_trades"] if t.signal == "s4"]
    spec = _spec(pages["/research"], "chart-vrp-levels")
    by = {s["name"]: s for s in spec["series"]}
    assert len(by["S4 short entry"]["data"]) == sum(1 for t in trades if t.direction < 0)
    assert len(by["S4 long entry"]["data"]) == sum(1 for t in trades if t.direction > 0)
    assert len(by["S4 short entry"]["data"]) + len(by["S4 long entry"]["data"]) == len(trades)
    assert {p[0] for s in ("S4 short entry", "S4 long entry") for p in by[s]["data"]} == \
        {str(pd.Timestamp(t.entry_date).date()) for t in trades}
    cap = _html_unescape(re.search(r'<p class="note" id="vrp-caption">(.*?)</p>', pages["/research"],
                                   re.DOTALL).group(1))
    assert "not P&L" in cap
    assert "21 sessions to the date" in cap
    assert "t+1 … t+21" in cap
    # the second chart zooms with the first (same x values)
    bars = _spec(pages["/research"], "chart-vrp-bars")
    assert bars["link"] == spec["link"] == "vrp"
    assert bars["xs"] == spec["xs"]
    # the forward series has no value in the last 21 sessions
    fwd = by["RV21 forward"]["data"]
    assert all(x is None for x in fwd[-21:])
    assert fwd[-22] is not None


def test_vrp_tables_and_csv_carry_the_same_numbers(dash, pages):
    """P3.2: each #vrp chart's table (the no-JS fallback) and its CSV (built from
    the spec's series) carry the same numbers."""
    html = pages["/research"]
    yrs = _spec(html, "chart-vrp-years")
    rows = yrs["table"]["rows"]
    assert [r[0] for r in rows] == yrs["xs"]
    assert [r[1] for r in rows] == [dash.signed(v, ".2f") for v in yrs["series"][0]["data"]]
    lev = _spec(html, "chart-vrp-levels")
    by = {s["name"]: s["data"] for s in lev["series"]}
    xs = pd.to_datetime(lev["xs"])
    for row in lev["table"]["rows"]:
        year = int(str(row[0])[:4])
        sel = xs.year == year
        for col, name in ((1, "VIX"), (2, "RV21 trailing"), (3, "RV21 forward")):
            vals = [v for v, s in zip(by[name], sel) if s and v is not None]
            assert abs(float(row[col]) - float(np.mean(vals))) <= 0.01, (year, name)
    f = _vol_frame()
    years = f.groupby(f.index.year)["vix"].mean()
    assert [int(str(r[0])[:4]) for r in lev["table"]["rows"]] == list(years.index)


def test_overview_vix_minus_trailing_rv_equals_the_computed_spread(pages):
    """P3.2: the Overview's 'VIX − RV21 (trailing)' line is the newest session's
    VIX close minus the RMS realised vol of its last 21 sessions."""
    f = _vol_frame().dropna(subset=["rv_trl"])
    last = f.iloc[-1]
    line = _html_unescape(re.sub(r"<[^>]+>", "", re.search(r'<p class="note" id="vix-rv-line">(.*?)</p>',
                                                            pages["/"], re.DOTALL).group(1)))
    assert f"VIX − RV21 (trailing) {_signed2(last['vix'] - last['rv_trl'])} vol pts" in line
    assert str(f.index[-1].date()) in line
    assert f"VIX {last['vix']:.2f}" in line
    assert f"{last['rv_trl']:.2f}%" in line
    assert 'href="research.html#vrp"' in pages["/"]


def _signed2(x: float) -> str:
    return ("+" if x >= 0 else "−") + f"{abs(x):.2f}"


def test_compact_specs_resolve_to_the_full_series(pages):
    """P3.2: a long daily chart stores its dates as day gaps, a linked chart names the
    chart whose dates it shares, and a derived series names its inputs; resolved,
    each equals what the page used to embed in full (desk_charts.resolve, the twin
    of desk.js resolveSpec)."""
    ch = _charts()
    xs = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    days = pd.bdate_range("2024-01-02", periods=300)
    iso = [str(d.date()) for d in days]
    c = ch.compact_dates(iso)
    assert c is not None
    assert ch.expand_dates(c) == iso
    assert ch.compact_dates(xs) is None                       # short: kept as it is
    html = pages["/research"]
    raw = _raw_specs(html)
    assert "xs" not in raw["chart-equity"]
    assert "xsDates" in raw["chart-equity"]
    eq = _spec(html, "chart-equity")
    for cid in ("chart-dd", "chart-rolling"):
        assert raw[cid]["xsFrom"] == "chart-equity"
        assert _spec(html, cid)["xs"] == eq["xs"]
    eqp = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet", columns=["nav"])
    assert eq["xs"] == [str(d.date()) for d in eqp.index]
    lev, bars = _spec(html, "chart-vrp-levels"), _spec(html, "chart-vrp-bars")
    by = {s["name"]: s["data"] for s in lev["series"]}
    assert bars["series"][0]["data"] == [None if (a is None or b is None) else round(a - b, 2)
                                         for a, b in zip(by["VIX"], by["RV21 forward"])]
    scaled = next(s for s in lev["series"] if s["name"].endswith("·VIX (backtest mark)"))
    names = [s["name"] for s in lev["series"]]
    k = raw["chart-vrp-levels"]["series"][names.index(scaled["name"])]["derive"]["k"]
    assert scaled["data"] == [round(k * v, 4) for v in by["VIX"]]


def test_browser_resolves_compact_specs_as_the_server_does(site_browser):
    """P3.2: desk.js expands the compact specs to the same x values and data the
    server-side resolver gives, and the two #vrp charts zoom together."""
    sb = site_browser
    html = (sb["site"] / "research.html").read_text()
    want_bars, want_lev = _spec(html, "chart-vrp-bars"), _spec(html, "chart-vrp-levels")
    ctx, page = _page(sb)
    page.goto(sb["base"] + "research.html#vrp", wait_until="load")
    page.locator("#chart-vrp-bars").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-vrp-levels")
    _wait_for_chart(page, "chart-vrp-bars")
    got = page.evaluate("""() => ({bx: Desk.charts['chart-vrp-bars'].spec.xs,
        bd: Desk.charts['chart-vrp-bars'].spec.series[0].data,
        lx: Desk.charts['chart-vrp-levels'].spec.xs,
        g1: Desk.charts['chart-vrp-levels'].chart.group, g2: Desk.charts['chart-vrp-bars'].chart.group})""")
    assert got["lx"] == want_lev["xs"]
    assert got["bx"] == want_bars["xs"]
    assert got["bd"] == want_bars["series"][0]["data"]
    assert got["g1"] == got["g2"] == "vrp"
    page.click('[data-zoom-for="chart-vrp-levels"][data-years="3"]')
    zoom = page.evaluate("""() => ['chart-vrp-levels', 'chart-vrp-bars'].map(
        id => Desk.charts[id].chart.getOption().dataZoom[0].start)""")
    assert zoom[0] > 50, zoom
    assert abs(zoom[0] - zoom[1]) < 1e-6, zoom
    ctx.close()


# ── P3.3: Model Risk as an exceptions view ────────────────────────────────────

_SEVERITY = ("breach", "watch", "ok", "info", "na")


def _monitor_rows(html: str) -> list[tuple[str, str, str]]:
    m = re.search(r'<table class="tbl cards" id="monitor-table">.*?<tbody>(.*?)</tbody>', html, re.DOTALL)
    assert m is not None
    tbody = m.group(1)
    return re.findall(r'<tr id="(check-[\w-]+)" data-status="(\w+)">(.*?)</tr>', tbody, re.DOTALL)


def test_model_risk_lists_breaches_first_and_counts_them_in_its_filters(dash, pages):
    """P3.3: the rows run BREACH, WATCH, OK, INFO (then N/A), so the first n rows,
    n = the BREACH tally, are all BREACH; within a status, inventory order. The
    summary strip's filter chips carry the tally."""
    html = pages["/model-risk"]
    rows = _monitor_rows(html)
    statuses = [st for _, st, _ in rows]
    assert statuses == sorted(statuses, key=_SEVERITY.index)
    d = dash.load_model_risk()
    n_breach = d["rag_counts"]["breach"]
    assert n_breach >= 1
    assert statuses[:n_breach] == ["breach"] * n_breach
    inv = [m["id"] for m in d["inventory"]] + [v["id"] for v in d["vendor_inputs"]] + ["Data"]
    for st in _SEVERITY:
        models = [m["model"] for m in d["monitors"] if m["status"] == st]
        assert models == sorted(models, key=inv.index), st
    chips = dict(re.findall(r'data-filter="status:(\w+)" data-count="(\d+)"', html))
    assert {k: int(v) for k, v in chips.items()} == {k: v for k, v in d["rag_counts"].items() if v}
    assert {k: int(v) for k, v in chips.items()} == {st: statuses.count(st) for st in set(statuses)}
    assert 'data-filter-for="monitor-table" data-filter="all" aria-pressed="true"' in html


def test_model_risk_history_ends_with_todays_value_and_uses_todays_limits(dash, pages):
    """P3.3: each check's history is the function that gives today's value, applied
    to every stored session; its last dot is today's value. The out-of-sample
    history is every gated fit's spx_oos_iv_rmse, recomputed here from the pickles,
    with today's rank among them."""
    d = dash.load_model_risk()
    with_history = [m for m in d["monitors"] if m["history"]]
    assert len(with_history) >= 6
    for m in with_history:
        if m["num"] is None:        # today's session cannot compute it: no dot stands in for today
            assert m["history"]["last"] is None, m["check"]
            assert m["history"]["today_missing"], m["check"]
        else:
            assert m["history"]["last"] == pytest.approx(m["num"], abs=1e-12), m["check"]
        lim = dash.MONITOR_LIMITS[m["limits_key"]]
        assert [x["status"] for x in m["history"]["dots"]] == [dash._rag(x["value"], lim)
                                                               for x in m["history"]["dots"]]
    oos = next(m for m in d["monitors"] if m["key"] == "oos_rmse")
    want = [dash._cal_losses(c).get("spx_oos_iv_rmse") for _, c in _gated_pickles(dash)]
    assert [x["value"] for x in oos["history"]["dots"]] == pytest.approx(want)
    assert [x["session"] for x in oos["history"]["dots"]] == [s for s, _ in _gated_pickles(dash)]
    today = want[-1]
    worse = sum(1 for v in want if v > today + 1e-12)
    better = sum(1 for v in want if v < today - 1e-12)
    if worse == 0:
        assert oos["history"]["rank"].endswith(f"worst of {len(want)}")
    elif better == 0:
        assert oos["history"]["rank"].endswith(f"best of {len(want)}")
    html = pages["/model-risk"]
    row = {i: body for i, _, body in _monitor_rows(html)}[oos["id"]]
    assert f"rank today: {oos['history']['rank']}" in _html_unescape(row)
    assert "today’s limits applied retroactively" in _html_unescape(html)
    assert "control limit" not in _visible_text(html).lower()
    for m in d["monitors"]:
        if not m["history"]:
            body = {i: b for i, _, b in _monitor_rows(html)}[m["id"]]
            assert "single observation" in body, m["check"]


def test_model_risk_bullets_scale_the_value_against_the_limits(dash, pages):
    """P3.3: each numeric check has a server-rendered bullet (role=img) whose tick x
    is proportional to the value on [0, max(1.5·a, value)] and whose label names
    the value and both limits."""
    html = pages["/model-risk"]
    d = {m["id"]: m for m in dash.load_model_risk()["monitors"]}
    n = 0
    for rid, _st, body in _monitor_rows(html):
        svg = re.search(r'<svg class="bullet"[^>]*>', body)
        m = d[rid]
        lim = dash.MONITOR_LIMITS.get(m["limits_key"] or "")
        if lim is None or m["num"] is None:
            assert svg is None, rid
            continue
        a = _attrs(svg.group(0))
        assert a["role"] == "img"
        top = max(1.5 * lim[1], m["num"])
        assert float(a["data-tick"]) == pytest.approx(m["num"] / top * 120, abs=0.01), rid
        label = _html_unescape(a["aria-label"])
        assert f"OK ≤ {lim[0]:g}" in label, rid
        assert f"watch ≤ {lim[1]:g}" in label, rid
        assert label.startswith("value "), rid
        n += 1
    assert n >= 10


def test_model_risk_values_link_to_their_evidence_and_the_evidence_links_back(dash, pages):
    """P3.3: every value cell links to the panel that shows its evidence (an id that
    exists on that page), and that panel carries a 'Monitored · <model> ·
    <status>' chip back to the check's row."""
    html = pages["/model-risk"]
    files = {v: k for k, v in dash.ROUTES.items()}
    d = {m["id"]: m for m in dash.load_model_risk()["monitors"]}
    linked = 0
    for rid, st, body in _monitor_rows(html):
        href = re.search(r'<a href="([\w.-]+\.html)#([\w-]+)" class="mon-value"', body)
        assert href, rid
        page, anchor = href.groups()
        target = pages[files[page]]
        assert f'id="{anchor}"' in target, (rid, page, anchor)
        chip = re.search(rf'<a class="chip chip-\w+ mon-link" href="model-risk.html#{rid}"[^>]*>([^<]+)</a>',
                         target)
        assert chip, (rid, page)
        word = {"ok": "OK", "na": "N/A"}.get(st, st.upper())
        assert _html_unescape(chip.group(1)) == f"Monitored · {d[rid]['model']} · {word}", rid
        # the chip sits inside the evidence element
        start = target.index(f'id="{anchor}"')
        assert target.index(f'href="model-risk.html#{rid}"') > start
        linked += 1
    assert linked == len(d)


# ── P3.4: the evidence graph ──────────────────────────────────────────────────

def _site_links():
    import sys
    if str(ROOT / "dashboard") not in sys.path:
        sys.path.insert(0, str(ROOT / "dashboard"))
    import site_links
    return site_links


def _pages_by_file(dash, pages) -> dict:
    return {dash.ROUTES[r]: html for r, html in pages.items()}


def test_every_internal_fragment_link_resolves_and_a_broken_one_is_caught(dash, pages):
    """P3.4: every internal link with a fragment ('page.html#id', '/route#id',
    '#id') names an id that exists on its target page; a seeded broken fragment
    is reported."""
    sl = _site_links()
    by_file = _pages_by_file(dash, pages)
    rep = sl.link_report(by_file, dash.ROUTES)
    assert rep["broken"] == []
    seeded = dict(by_file)
    seeded["index.html"] = seeded["index.html"].replace(
        "</main>", '<a href="calibration.html#no-such-panel">x</a>'
                   '<a href="/surface#nor-this">y</a></main>', 1)
    bad = sl.link_report(seeded, dash.ROUTES)["broken"]
    assert ("index.html", "calibration.html#no-such-panel") in bad
    assert ("index.html", "/surface#nor-this") in bad


def test_every_page_but_model_risk_links_to_evidence_on_three_other_panels(dash, pages):
    """P3.4: each page except Model Risk links from inside <main> to at least three
    distinct panels of other pages (the 'Monitored' chips are not counted)."""
    rep = _site_links().link_report(_pages_by_file(dash, pages), dash.ROUTES)
    for f in ("index.html", "surface.html", "calibration.html", "risk.html", "research.html",
              "corrections.html"):
        assert len(rep["cross"][f]) >= 3, (f, sorted(rep["cross"][f]))


def test_cited_on_footers_list_exactly_the_citing_pages(dash, pages):
    """P3.4: a panel restated elsewhere through m.cite() carries a 'Cited on'
    footer, and the footer lists exactly the pages whose rendered cite links
    point at it; every cross-page cite of a panel has its footer."""
    by_file = _pages_by_file(dash, pages)
    labels = {dash.ROUTES[r]: label for r, label, *_ in dash.NAV}
    cites: dict[str, set] = {}
    for f, html in by_file.items():
        for href in re.findall(r'<a class="cite" href="([\w.-]+\.html#[\w-]+)"', html):
            if not href.startswith(f + "#"):
                cites.setdefault(href, set()).add(f)
    assert len(cites) >= 10
    footers = {}
    for f, html in by_file.items():
        for pid, body in re.findall(r'<div class="panel-foot cite-foot" id="([\w-]+)-cited">(.*?)</div>',
                                    html, re.DOTALL):
            footers[f"{f}#{pid}"] = {labels[re.match(r"[\w-]+\.html", h).group(0)]: h
                                     for h in re.findall(r'<a href="([^"]+)">', body)}
    reg = dash.cite_registry()
    # every rendered cite of a panel is in the registry, and the panel's footer names its page
    for target, sources in cites.items():
        assert sources <= {e["page"] for e in reg[target]}, target
        page, pid = target.split("#")
        if f'id="{pid}" aria-labelledby' in by_file[page]:
            assert target in footers, target
            assert {labels[x] for x in sources} <= set(footers[target]), target
    # a footer lists exactly the pages whose cite() calls point at its panel
    for target, srcs in footers.items():
        assert set(srcs) == {e["label"] for e in reg[target]}, target
        for href in srcs.values():
            page = href.split("#")[0]
            assert any(e["page"] == page and (e["panel"] is None or href.endswith("#" + e["panel"]))
                       for e in reg[target]), (target, href)
    # a footer's links resolve too
    assert _site_links().link_report(by_file, dash.ROUTES)["broken"] == []


def test_cite_registry_reads_every_cite_call_in_the_templates(dash):
    """The registry is the templates' own m.cite calls with a literal target."""
    reg = dash.cite_registry()
    n_calls = 0
    for f in (ROOT / "dashboard" / "templates").glob("*.html"):
        for page, _anchor in re.findall(r"m\.cite\([^\n]*?'([\w.-]+\.html)#([\w-]+)'\)", f.read_text()):
            n_calls += page != f.name
    assert sum(len(v) for v in reg.values()) <= n_calls
    assert {e["page"] for e in reg["calibration.html#oos"]} >= {"index.html"}


# ── P3.5: the VIX complex on the Overview ─────────────────────────────────────

def _ts_wide() -> pd.DataFrame:
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        df = pd.read_sql("SELECT date, ticker, close FROM vix_term_structure", con)
    return df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()


def test_overview_term_structure_overlays_carry_their_dates(pages):
    """P3.5: the term-structure chart's dashed overlays are the curves 1, 5 and 21
    sessions before the shown one, each named with its date."""
    w = _ts_wide()[["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]].dropna()
    spec = _spec(pages["/"], "chart-ts")
    names = [s["name"] for s in spec["series"]]
    assert names[0] == f"{w.index[-1]} (latest)"
    want = [f"{w.index[-1 - k]} ({k} session{'s' if k > 1 else ''} earlier)" for k in (1, 5, 21)]
    assert names[1:] == want
    for s, k in zip(spec["series"][1:], (1, 5, 21)):
        row = w.iloc[-1 - k]
        want_vals = [round(float(row[t]), 2) for t in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")]
        assert [p[1] for p in s["data"]] == want_vals
    legend = [it["name"] for it in spec["legend_items"]]
    assert legend == names


def test_overview_vx_curve_is_on_its_own_date_with_its_m1_m2(dash, pages):
    """P3.5: the VX curve is the futures table's newest date, never the spot date:
    M1 → M2 = VX2 / VX1 − 1, recomputed here from vol_system.db; its age in NYSE
    sessions is shown in the stale styling of that age, and the curve is not on
    the spot chart."""
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        last = con.execute("SELECT MAX(date) FROM vix_futures_daily").fetchone()[0]
        px = [r[0] for r in con.execute(
            "SELECT COALESCE(settle, close) FROM vix_futures_daily WHERE date = ? AND expiry_date > ? "
            "AND is_weekly = 0 AND days_to_expiry > 0 ORDER BY expiry_date", (last, last))]
        db_last = con.execute("SELECT MAX(date) FROM spx_ohlcv").fetchone()[0][:10]
    html = pages["/"]
    line = _html_unescape(re.sub(r"<[^>]+>", "", re.search(r'<p class="note" id="vx-m1m2">(.*?)</p>', html,
                                                            re.DOTALL).group(1)))
    m1m2 = (px[1] / px[0] - 1) * 100
    assert line.startswith(f"M1 → M2 {_signed2(m1m2)}%")
    spec = _spec(html, "chart-vx-curve")
    assert spec["series"][0]["name"] == f"{last} settlements"
    assert spec["series"][0]["data"] == pytest.approx(px)
    age = dash.session_age(last, db_last)
    badge = re.search(r'<p class="note" id="vx-curve-age"><span class="age-(\w+)">(.*?)</span>;', html,
                      re.DOTALL)
    assert badge.group(1) == dash.stale_level(age)
    assert f"{age} NYSE session" in _html_unescape(re.sub(r"<[^>]+>", "", badge.group(2)))
    ts = _spec(html, "chart-ts")
    assert not any("settlement" in s["name"] for s in ts["series"])


def test_overview_vix_over_vix3m_share_and_percentile(dash, pages):
    """P3.5: the share of sessions with VIX / VIX3M above 1 and today's percentile
    among the prior sessions (percentile_rank wording), recomputed from the DB."""
    w = _ts_wide()
    r = (w["^VIX"] / w["^VIX3M"]).dropna()
    share = (r > 1).mean() * 100
    pct = (r.iloc[:-1] < r.iloc[-1]).mean() * 100
    line = _html_unescape(re.sub(r"<[^>]+>", "", re.search(r'<p class="note" id="vix-ratio-line">(.*?)</p>',
                                                            pages["/"], re.DOTALL).group(1)))
    assert f"{share:.1f}% of {len(r):,} sessions" in line
    assert f"Today {r.iloc[-1]:.3f}" in line
    rank = dash.percentile_rank(pct)
    assert rank["big"] in line
    assert f"{len(r) - 1:,} prior sessions since {r.index[0]}" in line


def test_regime_panel_states_the_distance_to_the_r2_gate(dash, pages):
    """P3.5: 'VVIX x vs R2 gate g (±d above/below)', with d = VVIX − gate."""
    r = dash.load_page1()["rule"]
    got = _html_unescape(re.sub(r"<[^>]+>", "", re.search(r'<dd id="r2-gate-distance">(.*?)</dd>', pages["/"],
                                                          re.DOTALL).group(1)))
    diff = r["vvix"] - r["gate"]
    side = "above" if diff > 0 else "below" if diff < 0 else "at the gate"
    assert got == f"VVIX {r['vvix']:.2f} vs R2 gate {r['gate']:.2f} ({_signed2(diff)} {side})"



# ── P3.6: the sleeve specification ────────────────────────────────────────────

def _spec_rows(html: str) -> dict:
    tbl = re.search(r'<table class="tbl dense cards spec" id="sleeve-spec-table">(.*?)</table>', html,
                    re.DOTALL)
    assert tbl is not None
    out = {}
    for key, body in re.findall(r'<tr data-sleeve="(\w+)"[^>]*>(.*?)</tr>', tbl.group(1), re.DOTALL):
        cells = re.findall(r'<td[^>]*data-label="([^"]+)"[^>]*>(.*?)</td>', body, re.DOTALL)
        out[key] = {k: _html_unescape(re.sub(r"<[^>]+>", " ", v)) for k, v in cells}
    return out


def test_sleeve_spec_sits_above_the_equity_curve_and_counts_the_trade_log(dash, pages):
    """P3.6: #sleeve-spec comes right after the KPI strip, before #equity; each
    sleeve's side counts are its trade-log records by direction; the Instrument
    cell names the β·VIX mark; the 'combined' records are explained, not counted."""
    html = pages["/research"]
    assert html.index('id="sleeve-spec"') < html.index('id="equity"')
    assert html.index('id="kpi-strip"') < html.index('id="sleeve-spec"')
    with open(SHIPPED / "backtest" / "full_results.pkl", "rb") as f:
        trades = pickle.load(f)["all_trades"]
    rows = _spec_rows(html)
    assert list(rows) == ["s1c", "s3", "s4", "s1", "s2"]
    for key, r in rows.items():
        mine = [t for t in trades if t.signal == key]
        n_long = sum(1 for t in mine if t.direction > 0)
        n_short = sum(1 for t in mine if t.direction < 0)
        parts = ((f"short {n_short}" if n_short else ""), (f"long {n_long}" if n_long else ""))
        want = " / ".join(x for x in parts if x)
        assert r["Side (trades)"].split("exits")[0].strip() == want, key
        assert "β·VIX" in r["Instrument"], key
    comb = [t for t in trades if t.signal == "combined"]
    m_ = re.search(r'<p class="note" id="sleeve-spec-note">(.*?)</p>', html, re.DOTALL)
    assert m_ is not None
    note = _html_unescape(re.sub(r"<[^>]+>", "", m_.group(1)))
    n_l, n_s = sum(t.direction > 0 for t in comb), sum(t.direction < 0 for t in comb)
    assert f"{len(comb)} “combined” records ({n_l} long, {n_s} short)" in note
    assert "z_vrp:" in note


def test_sleeve_spec_constants_are_read_from_the_modules(dash, monkeypatch):
    """P3.6: every constant in the table is the engine's or the signal module's
    value at render time; changing one changes the page."""
    from joint_vol_calibration.backtest import backtest_engine as be
    from joint_vol_calibration.signals import signal_engine as se
    _resp, html = _get(dash, "/research")
    rows = _spec_rows(html)
    vp = lambda x: f"{x * 100:g}"            # noqa: E731
    assert f"{be.STRADDLE_TENOR_DAYS}-calendar-day" in rows["s4"]["Instrument"]
    assert f"VIX + {vp(se.S1_ENTRY_THRESHOLD)} vol pts" in rows["s1c"]["Entry rule"]
    assert rows["s1c"]["Max hold"].strip() == f"{se.S1_MAX_HOLD} sessions"
    assert rows["s3"]["Max hold"].strip() == f"{se.S3_MAX_HOLD} sessions"
    gm = lambda x: f"{x:g}".replace("-", "−")            # noqa: E731 (a true minus sign)
    assert f"z_vrp > {gm(se.S4_Z_ENTRY_SHORT)} and falling" in rows["s4"]["Entry rule"]
    assert f"z_vrp < {gm(se.S4_Z_ENTRY_LONG)} and rising" in rows["s4"]["Entry rule"]
    assert f"|z_vrp| < {gm(se.S4_Z_EXIT)}" in rows["s4"]["Exit rule"]
    assert f"< {gm(se.S3_ZSCORE_ENTRY)} in R0 or R1" in rows["s3"]["Entry rule"]
    assert f"{se.S1_MAX_KELLY:.0%} of sleeve NAV" in rows["s1c"]["Sizing"]
    assert f"hedge slippage {be.SLIPPAGE_PCT:.2%}" in rows["s3"]["Costs"]
    note = _html_unescape(re.search(r'id="sleeve-spec-note">(.*?)</p>', html, re.DOTALL).group(1))
    assert f"NAV / ({be.MARGIN_RATIO:.0%} × K × {be.CONTRACT_MULT})" in note
    assert f"S3_PROXY_COST ({be.S3_PROXY_COST * 100:g}% of notional)" in note
    monkeypatch.setattr(se, "S4_Z_EXIT", 0.45)
    monkeypatch.setattr(be, "MARGIN_RATIO", 0.25)
    _resp, html2 = _get(dash, "/research")
    assert "|z_vrp| < 0.45" in _spec_rows(html2)["s4"]["Exit rule"]
    assert "NAV / (25% × K × 100)" in _html_unescape(html2)


def test_browser_sleeve_spec_is_on_the_first_screen_and_cards_on_a_phone(site_browser):
    """P3.6: at 1440 px the sleeve table starts within the first 900 px."""
    sb = site_browser
    ctx, page = _page(sb, 1440, 900)
    page.goto(sb["base"] + "research.html", wait_until="load")
    top = page.evaluate("document.getElementById('sleeve-spec').getBoundingClientRect().top + scrollY")
    assert top < 900, top
    ctx.close()


# ── P3.7: parameter tiles as dot plots with ±2·SE ─────────────────────────────

def _dot_svgs(html: str) -> dict:
    return {k: (attrs, body) for attrs, k, body in
            ((m.group(0), m.group(1), m.group(2)) for m in re.finditer(
                r'<svg class="dots"[^>]*data-param="(\w+)"[^>]*>(.*?)</svg>', html, re.DOTALL))}


def test_parameter_dot_plots_are_the_gated_fits_with_their_own_standard_errors(dash, pages):
    """P3.7: under each parameter, one dot per gate-passing fit (oldest first) with
    a ±2·SE whisker from that fit's identifiability (diagnostics/<session>.json or
    its pickle); a parameter on its bound has no whisker; role=img with an
    aria-label listing the dated values."""
    html = pages["/calibration"]
    svgs = _dot_svgs(html)
    assert sorted(svgs) == sorted(k for k, *_ in dash.PARAM_META)
    gated = _gated_pickles(dash)
    specs = {k: sp for k, _sym, _name, sp in dash.PARAM_META}
    for key, (tag, body) in svgs.items():
        attrs = _attrs(tag[:tag.index(">") + 1])
        assert attrs["role"] == "img"
        label = _html_unescape(attrs["aria-label"])
        n_whiskers = 0
        for session, cal in gated:
            diag_path = SHIPPED / "calibrations" / "diagnostics" / f"{session}.json"
            ident = cal.get("identifiability") or (json.loads(diag_path.read_text())["identifiability"]
                                                   if diag_path.exists() else {})
            v = float(cal["params"][key])
            text = f"{v:{specs[key]}}".replace("-", "−")
            if key in (ident.get("on_bound") or []):
                assert f"{session} {text} (on its bound, no SE)" in label, (key, session)
            else:
                se = (ident.get("standard_errors") or {}).get(key)
                if se is None:
                    assert f"{session} {text} (no SE stored)" in label, (key, session)
                else:
                    assert f"{session} {text} ± {2 * se:{specs[key]}}" in label, (key, session)
                    n_whiskers += 1
        assert body.count('class="dp-se"') == n_whiskers, key
        assert body.count("<circle") == len(gated), key
    # σ on its bound in the headline fit: the newest dot sits on the dashed line, without a whisker
    d = dash.load_page2()
    if "sigma" in (d.get("pinned_params") or []):
        sig = next(t for t in d["param_tiles"] if t["key"] == "sigma")["dots"]
        assert sig["rows"][-1]["x0"] is None
        assert any(abs(b["x"] - sig["rows"][-1]["x"]) < 0.01 for b in sig["bounds"])
    assert "Gauss–Newton curvature scale, not a confidence interval" in _html_unescape(html)


def test_parameter_tiles_speak_trader_units(dash, pages):
    """P3.7: √θ as the long-run vol, √v₀ as the spot vol, κ as a half-life in sessions."""
    import math
    _path, cal = dash.load_headline_calibration()
    p = cal["params"]
    text = _html_unescape(pages["/calibration"])
    assert f"√θ = {math.sqrt(p['theta']) * 100:.2f}% long-run vol" in text
    assert f"√v₀ = {math.sqrt(p['v0']) * 100:.2f}% spot vol" in text
    assert f"half-life {math.log(2) / p['kappa'] * 252:.1f} sessions (ln2/κ × 252)" in text


def test_dot_plot_geometry():
    """The forest plot's x scale is linear; a near bound is drawn, a far one is not."""
    ch = _charts()
    pts = [{"session": "2026-01-02", "value": 1.0, "se": 0.1},
           {"session": "2026-02-02", "value": 2.0, "se": None, "on_bound": True}]
    g = ch.dot_plot(pts, 0.0, 2.0)
    r0, r1 = g["rows"]
    scale = (r1["x"] - r0["x"]) / 1.0
    assert r0["x1"] - r0["x"] == pytest.approx(0.2 * scale, abs=0.02)
    assert r1["x0"] is None
    assert [b["value"] for b in g["bounds"]] == [2.0]          # 0.0 lies 0.8 below a span of 1.2: off scale
    assert ch.dot_plot([], 0, 1) is None
    # one value and no whisker: no division by zero
    one = ch.dot_plot([{"session": "2026-01-02", "value": 0.5, "se": None}], 0.0, 10.0)
    assert one["left"] <= one["rows"][0]["x"] <= one["right"]


# ── P4.1: the quote inspector on every smile ──────────────────────────────────

def _client_quotes(g: dict) -> list[dict]:
    """The packed quotes of one smile expiry, unpacked with desk.js's formulas
    (decodeQuotes), written out here independently of desk_quotes.decode."""
    from scipy.stats import norm
    out, K = [], g["K0"]
    for i in range(len(g["mid"])):
        if i:
            K += g["dK"][i - 1] * g["step"]
        mid, db, da = g["mid"][i] / 1e5, g["db"][i] / 1e6, g["da"][i] / 1e6
        hs = (db + da) / 2
        k = np.log(K / g["F"])
        sw = np.sqrt((mid / 100) ** 2 * g["T"] / 365.0)
        d1 = -k / sw + sw / 2
        right = g["r"][i]
        models = {m: (None if e[i] is None else {"iv": mid + e[i] / 1e4, "z": (e[i] / 1e4) / max(hs, 0.01),
                                                  "st": int(g["s"][m][i])})
                  for m, e in g["m"].items()}
        out.append({"strike": K, "right": right, "k": float(k), "mid": mid,
                    "delta": float(norm.cdf(d1)) - (1.0 if right == "P" else 0.0), "models": models})
    return out


def _producer_delta(p) -> float:
    from scipy.stats import norm
    w = p.mid_iv ** 2 * p.T_days / 365.0
    return float(norm.cdf(-p.k / np.sqrt(w) + np.sqrt(w) / 2)) - (1.0 if p.right == "P" else 0.0)


def _smile_specs(pages) -> list[tuple[str, dict]]:
    out = [("/surface", _raw_specs(pages["/surface"])["chart-slices"])]
    cal = _raw_specs(pages["/calibration"])
    out += [("/calibration", s) for cid, s in cal.items()
            if cid.startswith("chart-smile-") and s.get("quotes")]
    return out


def test_quote_inspector_formulas_reproduce_the_residual_table(dash, pages):
    """P4.1: the inspector names a quote's strike, right, forward delta and each
    model's z. Three quotes per expiry (first, middle, last) of every smile on
    Surface and Calibration, unpacked with the client's formulas from the page's
    own spec, give residuals.parquet's strike (exactly), right, the producer's
    forward delta (1e-6) and each model's error_hs (±0.05), in_sample and
    inside_spread."""
    res = pd.read_parquet(_surface_file("residuals.parquet"))
    specs = _smile_specs(pages)
    assert len(specs) >= 2, [r for r, _ in specs]
    n = 0
    for route, spec in specs:
        groups = spec["quotes"]["groups"]
        assert groups, route
        for e, g in groups.items():
            qs = _client_quotes(g)
            for i in sorted({0, len(qs) // 2, len(qs) - 1}):
                q = qs[i]
                for m, mm in q["models"].items():
                    rows = res[(res["model"] == m) & (res["expiry"] == e)].sort_values("k", kind="mergesort")
                    p = next(rows.iloc[[i]].itertuples())
                    assert (q["strike"], q["right"]) == (p.strike, p.right), (route, e, i)
                    assert q["k"] == pytest.approx(p.k, abs=1e-9)
                    assert abs(q["delta"] - _producer_delta(p)) <= 1e-6, (route, e, i)
                    if mm is None:
                        assert not np.isfinite(p.model_iv)
                        continue
                    assert abs(mm["z"] - p.error_hs) <= 0.05, (route, e, m, i, mm["z"], p.error_hs)
                    flags = (bool(mm["st"] & 1), bool(mm["st"] & 2))
                    assert flags == (bool(p.in_sample), bool(p.inside_spread)), (route, e, m, i)
                    n += 1
    assert n >= 3 * len(specs[0][1]["quotes"]["groups"]) * len(specs[0][1]["quotes"]["models"])


def test_quote_inspector_footnote_names_the_chain_and_its_quote_time(dash, pages):
    """The tooltip's footnote is the chain's own source and quote time (surface.json),
    'Yahoo close, no quote time' for a Yahoo snapshot without one."""
    sj = json.loads(_surface_file("surface.json").read_text())
    want = dash.quote_source_note(sj)
    inp = sj.get("inputs") or {}
    if str(inp.get("source")).lower() in ("yfinance", "yahoo") and not inp.get("quote_time"):
        assert want == "Yahoo close, no quote time"
    for route, spec in _smile_specs(pages):
        assert spec["quotes"]["source"] == want, route
    assert dash.quote_source_note({"inputs": {"source": "cboe", "quote_time": "2026-10-16T15:45:00"}}) \
        == "Cboe, quotes at 2026-10-16 15:45:00"


def test_smile_series_hold_no_raw_series_name_in_their_legend(pages):
    """The tooltip leaked a raw series name ('2026-11-16 mid'); every legend button
    of a smile is a key without the expiry, and the residual strip's series draw
    on the second grid."""
    for route, spec in _smile_specs(pages):
        assert not any(re.match(r"\d{4}-\d{2}-\d{2} ", it["name"]) for it in spec["legend_items"]), route
        strip = [s for s in spec["series"] if s.get("strip")]
        assert strip, route
        assert all(s["quote"] in ("z", "zfit") for s in strip), route


def test_browser_quote_inspector_hover_keyboard_and_delta_axis(site_browser):
    """P4.1: hovering a quote of the shown expiry gives its strike, bid and ask,
    every model's IV and z (all four toggled on) and no raw series name; the
    arrow-key readout says the same as the tooltip; the residual strip shares the
    smile's x range; the forward-delta switch keeps the expiry; desk.js unpacks
    the quotes as desk_quotes.decode does; the CSV has one row per quote."""
    sb = site_browser
    html = (sb["site"] / "surface.html").read_text()
    spec = _raw_specs(html)["chart-slices"]
    import desk_quotes
    ctx, page = _page(sb)
    page.goto(sb["base"] + "surface.html", wait_until="load")
    page.locator("#chart-slices").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-slices")
    exp = page.evaluate("Desk.filters['chart-slices']")
    assert exp in spec["quotes"]["groups"]
    # every model on
    for label in [m["label"] for m in spec["quotes"]["models"]]:
        b = page.locator(f"[data-legend-for='chart-slices'][data-series='{label}']")
        if b.get_attribute("aria-pressed") == "false":
            b.click()
    js = page.evaluate("""(e) => Desk.charts['chart-slices'].spec._rows[e].map(q => [q.strike, q.right,
        q.k, q.delta, q.xd, q.bid, q.ask,
        Object.fromEntries(Object.entries(q.models).map(([m, v]) => [m, v.z]))])""", exp)
    py = desk_quotes.decode(spec["quotes"]["groups"][exp])
    assert len(js) == len(py)
    for a, b in zip(js, py):
        assert (a[0], a[1]) == (b["strike"], b["right"])
        assert a[2] == pytest.approx(b["k"], abs=1e-12)
        assert a[3] == pytest.approx(b["delta"], abs=1e-9)
        assert a[4] == pytest.approx(b["x_delta"], abs=1e-9)
        for m, z in a[7].items():
            assert (z is None) == (b["models"][m]["z"] is None)
            if z is not None:
                assert z == pytest.approx(b["models"][m]["z"], abs=1e-9)
    i = len(py) // 2
    pos = page.evaluate("""(i) => { const c = Desk.charts['chart-slices'];
        const q = c.spec._rows[Desk.filters['chart-slices']][i];
        const px = c.chart.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, [q.k, q.mid]);
        const r = c.el.getBoundingClientRect(); return {x: r.left + px[0], y: r.top + px[1]}; }""", i)
    page.mouse.move(pos["x"], pos["y"])
    page.wait_for_timeout(250)
    tip_js = """() => { const t = [...document.querySelectorAll('#chart-slices > div')]
        .filter(d => getComputedStyle(d).display !== 'none' && d.style.position === 'absolute'
                && d.innerText.startsWith('K '))[0]; return t ? t.innerText : null; }"""
    tip = page.evaluate(tip_js)
    assert tip, "no tooltip"
    lines = [x.strip() for x in tip.split("\n") if x.strip()]
    q = py[i]
    # the hovered quote or a neighbour within a pixel or two
    head = r"K [\d,]+ [PC] · ln\(K/F\) [−+]?\d\.\d{3} · Δ [−+]\d\.\d{2} · \d+ d$"
    assert re.match(head, lines[0]), lines[0]
    assert re.search(r"Bid / mid / ask \d+\.\d\d / \d+\.\d\d / \d+\.\d\d%", tip)
    assert sum(1 for x in lines if re.search(r" · z [−+]?\d+\.\d", x)) == len(spec["quotes"]["models"]), lines
    assert not re.search(r"\d{4}-\d{2}-\d{2} mid", tip)
    assert lines[-1] == spec["quotes"]["source"]
    readout = page.locator("#chart-slices-readout").inner_text()
    assert readout == " · ".join(lines)
    # the keyboard: the readout steps one quote and says what the tooltip then says
    page.focus("#chart-slices-readout")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    readout2 = page.locator("#chart-slices-readout").inner_text()
    tip2 = page.evaluate(tip_js)
    assert readout2 != readout
    assert readout2 == " · ".join(x.strip() for x in tip2.split("\n") if x.strip())
    qi = page.evaluate("Desk.charts['chart-slices'].qi")
    assert readout2.startswith(f"K {py[qi]['strike']:,} {py[qi]['right']}")
    # the strip shares the smile's x range
    ax = page.evaluate("Desk.charts['chart-slices'].chart.getOption().xAxis.map(a => [a.min, a.max])")
    assert len(ax) == 2
    assert ax[0] == ax[1]
    assert ax[0][0] <= min(x["k"] for x in py)
    assert ax[0][1] >= max(x["k"] for x in py)
    # forward delta: the axis is renamed and the expiry kept
    page.click("[data-xmode-for='chart-slices'][data-xmode='delta']")
    page.wait_for_timeout(200)
    names = page.evaluate("Desk.charts['chart-slices'].chart.getOption().xAxis.map(a => a.name)")
    assert "Forward delta" in names
    tab = page.locator(f"[data-chart-filter='chart-slices'] [data-value='{exp}']")
    assert tab.get_attribute("aria-selected") == "true"
    assert page.evaluate("Desk.filters['chart-slices']") == exp
    assert "x=delta" in page.url
    assert q["k"] < 0 or q["x_delta"] > 0                       # (the quote exists on the delta axis too)
    csv = page.evaluate("Desk.specCsv(Desk.charts['chart-slices'].spec)")
    assert len(csv.strip().split("\n")) == 1 + sum(len(g["mid"]) for g in spec["quotes"]["groups"].values())
    ctx.close()


def test_browser_smile_expiry_switch_keeps_the_spec_and_draws_the_shown_group_only(site_browser):
    """P4.2: every expiry-tab click re-parsed the 250 KB smile spec and rebuilt all
    17 × 12 series (64–112 ms, 304–448 ms at 4× CPU). A switch now keeps the parsed
    spec, draws the shown expiry's series only (a model toggled off carries no
    data), keeps the forward line on a drawn series of that expiry, draws the strip's
    clipped points as triangles and leaves no earlier expiry marked shown in the
    legend. (The timing itself is the interact.mjs gate.)"""
    sb = site_browser
    html = (sb["site"] / "surface.html").read_text()
    spec = _raw_specs(html)["chart-slices"]
    exps = list(spec["quotes"]["groups"])
    ctx, page = _page(sb)
    page.goto(sb["base"] + "surface.html", wait_until="load")
    page.locator("#chart-slices").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-slices")
    page.evaluate("Desk.charts['chart-slices'].spec.__kept = true")
    first = page.evaluate("Desk.filters['chart-slices']")
    state_js = """(exp) => {
      const c = Desk.charts['chart-slices'], o = c.chart.getOption(), sel = o.legend[0].selected;
      const off = [...document.querySelectorAll('[data-legend-for="chart-slices"][aria-pressed="false"]')]
        .map(b => b.dataset.series);
      const keyOf = n => (c.spec.series.find(x => x.name === n || x.name + ' beyond the clip' === n) || {});
      const drawn = o.series.filter(s => s.id !== 'zband');
      const clip = c.spec.quotes.zClip || 10;
      return {
        kept: c.spec.__kept === true, filter: Desk.filters['chart-slices'],
        names: drawn.map(s => s.name),
        marked: drawn.filter(s => s.markLine && (s.markLine.data || []).length).map(s => s.name),
        shownElsewhere: Object.keys(sel).filter(k => sel[k] && !k.startsWith(exp + ' ')),
        offWithData: drawn.filter(s => (s.data || []).length).map(s => s.name)
          .filter(n => off.includes(keyOf(n).legendKey || keyOf(n).name)),
        rows: c.spec._rows[exp].length,
        bars: ((o.series.find(s => s.name === exp + ' bid–ask') || {}).data || []).length,
        clipped: c.spec._rows[exp].filter(q => Object.values(q.models)
          .some(m => m.z !== null && Math.abs(m.z) > clip)).length,
        triangles: o.series.filter(s => / beyond the clip$/.test(s.name))
          .reduce((n, s) => n + (s.data || []).filter(p => p.symbol === 'triangle').length, 0) }; }"""
    for exp in [*[e for e in exps if e != first][:3], first]:
        page.click(f"[data-chart-filter='chart-slices'] [data-value='{exp}']")
        page.wait_for_timeout(250)
        st = page.evaluate(state_js, exp)
        assert st["kept"], "the spec was parsed again"
        assert st["filter"] == exp
        assert st["names"], st
        assert all(n.startswith(exp + " ") for n in st["names"]), st["names"]
        assert st["marked"], st
        assert all(n.startswith(exp + " ") for n in st["marked"]), st["marked"]
        assert not st["shownElsewhere"], st["shownElsewhere"]
        assert not st["offWithData"], st["offWithData"]
        # one bar per quote: lower cap, the bar, upper cap and a break (7 points)
        assert st["bars"] == 7 * st["rows"]
        assert st["triangles"] <= st["clipped"] * len(spec["quotes"]["models"])
    ctx.close()


def test_browser_expiry_is_the_shared_cursor_on_surface(site_browser):
    """P4.3: the expiry selected one place on Surface is selected everywhere. A
    click on an IV-heatmap row head selects that expiry's smile tab, writes exp=
    into the hash, marks its rows aria-current in the IV heatmap, every residual
    heatmap and the delta grid (only those), draws a line at its category on
    #chart-by-expiry and names it in the readout; ']' and '[' step to the next
    and previous expiry; a click on a by-expiry category selects it. Without
    JavaScript the row heads stay plain text. Expiries are read from the page:
    the daily bot replaces them."""
    sb = site_browser
    ctx, page = _page(sb)
    page.goto(sb["base"] + "surface.html", wait_until="load")
    tabs = page.evaluate("""[...document.querySelectorAll('[data-chart-filter="chart-slices"] [role=tab]')]
        .map(t => t.dataset.value)""")
    rows = page.evaluate("[...document.querySelectorAll('#iv-heatmap tr[data-exp]')].map(r => r.dataset.exp)")
    assert rows == tabs
    exp = rows[len(rows) // 2]
    label = page.locator(f"#iv-heatmap tr[data-exp='{exp}'] th").inner_text().strip()
    btn = page.locator(f"#iv-heatmap tr[data-exp='{exp}'] th[scope=row] button.row-pick")
    btn.scroll_into_view_if_needed()
    btn.click()
    page.wait_for_timeout(300)
    page.locator("#chart-by-expiry").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-by-expiry")
    st = page.evaluate("""() => ({
      sel: [...document.querySelectorAll('[data-chart-filter="chart-slices"] [aria-selected=true]')]
        .map(t => t.dataset.value),
      hash: location.hash,
      current: [...document.querySelectorAll('tr[aria-current]')]
        .map(r => [r.closest('table').id, r.dataset.exp, r.getAttribute('aria-current')]),
      tables: [...new Set([...document.querySelectorAll('tr[data-exp]')].map(r => r.closest('table').id))],
      lines: Desk.charts['chart-by-expiry'].chart.getOption().series
        .flatMap(s => ((s.markLine || {}).data || []).map(d => d.xAxis)).filter(x => x !== undefined),
      readout: document.getElementById('iv-heatmap-readout').textContent })""")
    assert st["sel"] == [exp]
    assert f"exp={exp}" in st["hash"]
    assert "iv-heatmap" in st["tables"]
    assert "delta-grid-listed" in st["tables"]
    assert any(t.startswith("resid-heat-") for t in st["tables"])
    assert sorted(c[0] for c in st["current"]) == sorted(st["tables"])     # one row per table
    assert all(c[1] == exp and c[2] == "true" for c in st["current"]), st["current"]
    assert st["lines"] == [label], st["lines"]
    assert exp in st["readout"]
    # ']' steps to the next expiry, '[' back
    page.keyboard.press("]")
    page.wait_for_timeout(200)
    nxt = tabs[tabs.index(exp) + 1]
    assert page.evaluate("Desk.filters['chart-slices']") == nxt
    assert f"exp={nxt}" in page.evaluate("location.hash")
    assert page.locator(f"#delta-grid-listed tr[data-exp='{nxt}']").get_attribute("aria-current") == "true"
    page.keyboard.press("[")
    page.wait_for_timeout(200)
    assert page.evaluate("Desk.filters['chart-slices']") == exp
    # a click on a by-expiry category selects it
    i = 1 if tabs.index(exp) != 1 else 2
    pos = page.evaluate("""(i) => { const c = Desk.charts['chart-by-expiry'];
        const x = c.chart.convertToPixel({xAxisIndex: 0}, c.spec.x.categories[i]);
        const r = c.el.getBoundingClientRect(); return {x: r.left + x, y: r.top + r.height / 2}; }""", i)
    page.mouse.click(pos["x"], pos["y"])
    page.wait_for_timeout(300)
    assert page.evaluate("Desk.filters['chart-slices']") == tabs[i]
    assert "[" in page.locator("#keysDialog").inner_text()
    ctx.close()
    # without JavaScript the row heads are the labels they were
    ctx = sb["browser"].new_context(viewport={"width": 1440, "height": 900}, java_script_enabled=False)
    page = ctx.new_page()
    page.goto(sb["base"] + "surface.html", wait_until="load")
    assert page.locator(".row-pick").count() == 0
    assert page.locator("tr[aria-current]").count() == 0
    assert page.locator(f"#iv-heatmap tr[data-exp='{exp}'] th").inner_text().strip() == label
    ctx.close()
    ctx, page = _page(sb)
    page.goto(sb["base"] + "index.html", wait_until="load")
    assert "Previous / next expiry" not in page.locator("#keysDialog").inner_text()
    ctx.close()


def test_chart_group_domains_and_year_windows_are_pure(dash):
    """P4.4: the helpers behind the chart groups. nice_step is a 1-2-2.5-5 step;
    shared_domain covers every range, padded, on that step; year_windows gives
    each year's first and last date (None for a year without one); group_smiles
    sets one x domain and one y step on every panel, links them, merges their
    legends and prefixes the expiry to every table row."""
    assert dash.nice_step(0.8) == pytest.approx(0.2)
    assert dash.nice_step(60, 6) == pytest.approx(10)
    assert dash.nice_step(0.37) == pytest.approx(0.1)
    lo, hi, step = dash.shared_domain([(-0.33, 0.05), (-0.12, 0.31), (-0.2, 0.1)])
    assert step == pytest.approx(0.2)
    assert lo <= -0.33
    assert hi >= 0.31
    assert lo / step == pytest.approx(round(lo / step))
    assert hi / step == pytest.approx(round(hi / step))
    dates = ["2021-12-31", "2022-01-03", "2022-06-01", "2022-12-30", "2023-01-03"]
    assert dash.year_windows(dates, [2022, 2023, 2024]) == [["2022-01-03", "2022-12-30"],
                                                            ["2023-01-03", "2023-01-03"], None]

    def panel(exp, k, y, items):
        return {"expiry": exp, "source": "quotes", "k_range": k, "y_range": y,
                "spec": {"id": f"chart-smile-{exp}", "x": {"name": "ln(K/F)"}, "y": {"name": "IV (%)"},
                         "legend_items": [{"name": n} for n in items],
                         "table": dash.charts.table("t", ["ln(K/F)", "Quotes#"], [["−0.1", 3]])}}
    sm = [panel("2026-11-16", (-0.3, 0.05), (11.0, 58.0), ["Bid–ask", "Mid"]),
          panel("2027-01-15", (-0.15, 0.2), (12.0, 31.0), ["Bid–ask", "Mid", "Heston"])]
    g = dash.group_smiles(sm, "2026-10-16")
    assert g["ids"] == ["chart-smile-2026-11-16", "chart-smile-2027-01-15"]
    assert {(s["spec"]["x"]["min"], s["spec"]["x"]["max"]) for s in sm} == {tuple(g["x"])}
    assert g["x"][0] <= -0.3
    assert g["x"][1] >= 0.2
    assert all(s["spec"]["y"]["step"] == g["y_step"] == 10 for s in sm)
    assert [(s["spec"]["y"]["min"], s["spec"]["y"]["max"]) for s in sm] == [(10, 60), (10, 40)]
    assert all(s["spec"]["link"] == dash.SMILE_GROUP for s in sm)
    assert [i["name"] for i in g["legend_items"]] == ["Bid–ask", "Mid", "Heston"]
    assert g["table"]["rows"] == [["2026-11-16", "−0.1", "3"], ["2027-01-15", "−0.1", "3"]]
    assert [c["label"] for c in g["table"]["columns"]] == ["Expiry", "ln(K/F)", "Quotes"]
    assert dash.group_smiles(sm[:1], "x") is None


def test_browser_research_time_series_are_one_chart_group(site_browser):
    """P4.4: drawdown and the rolling Sharpe sit under the equity curve, full
    width, as one group: a pointer on the equity curve shows the same date on all
    three, a preset, the year menu, a click on a year's bar and the wheel on any
    member give all three the same window, R2 is shaded on each, their plot areas
    line up within 1 px at 1440 px and a preset click makes no long task over
    100 ms. Dates are read from the specs."""
    sb = site_browser
    html = (sb["site"] / "research.html").read_text()
    raw = _raw_specs(html)
    ids = ["chart-equity", "chart-dd", "chart-rolling"]
    assert all(raw[i].get("link") == "research-ts" for i in ids)
    assert all((raw[i].get("marks") or {}).get("areas") for i in ids)          # R2 on each
    assert all(f'id="{i}" data-spec="{i}-spec" data-chart-group="research-ts"' in html for i in ids)
    ctx, page = _page(sb)
    page.goto(sb["base"] + "research.html", wait_until="load")
    for i in ids:
        page.locator("#" + i).scroll_into_view_if_needed()
        _wait_for_chart(page, i)
    win_js = """(ids) => ids.map(id => {
        const r = Desk.charts[id].chart.getModel().getComponent('dataZoom', 0).getValueRange();
        return r.map(v => new Date(v).toISOString().slice(0, 10)); })"""
    rect_js = """(ids) => ids.map(id => { const g = Desk.charts[id].chart.getModel().getComponent('grid', 0)
        .coordinateSystem.getRect(); const r = Desk.charts[id].el.getBoundingClientRect();
        return [r.left + g.x, r.left + g.x + g.width]; })"""
    rects = page.evaluate(rect_js, ids)
    assert max(r[0] for r in rects) - min(r[0] for r in rects) <= 1, rects
    assert max(r[1] for r in rects) - min(r[1] for r in rects) <= 1, rects
    xs = page.evaluate("Desk.charts['chart-equity'].spec.xs")
    # the pointer: a date in the middle of the window
    day = xs[len(xs) // 2]
    page.locator("#chart-equity").scroll_into_view_if_needed()
    pt = page.evaluate("""(d) => { const c = Desk.charts['chart-equity'];
        const x = c.chart.convertToPixel({xAxisIndex: 0}, d), r = c.el.getBoundingClientRect();
        return {x: r.left + x, y: r.top + r.height / 3}; }""", day)
    page.mouse.move(pt["x"], pt["y"])
    page.wait_for_timeout(300)
    heads = page.evaluate("""(ids) => ids.map(id => {
        const t = [...document.querySelectorAll('#' + id + ' > div')]
          .filter(d => d.style.position === 'absolute' && getComputedStyle(d).display !== 'none'
            && d.innerText.trim());
        return t.length ? t[0].innerText.split('\\n')[0].trim() : null; })""", ids)
    # the same session on all three (the pointer lands on this day or a neighbour)
    assert len(set(heads)) == 1, heads
    assert heads[0] in xs[xs.index(day) - 1:xs.index(day) + 2], (day, heads)
    # 3Y: one window on all three, and no long task over 100 ms
    page.evaluate("""() => { window.__lt = []; new PerformanceObserver(l => l.getEntries()
        .forEach(e => window.__lt.push(e.duration))).observe({type: 'longtask'}); }""")
    page.click("[data-zoom-for='chart-equity'][data-years='3']")
    page.wait_for_timeout(400)
    w3 = page.evaluate(win_js, ids)
    assert len({tuple(w) for w in w3}) == 1, w3
    assert w3[0][1] == xs[-1]
    assert max(page.evaluate("window.__lt") or [0]) <= 100
    # a year's bar on the annual chart: that year's first and last session
    page.locator("#chart-annual").scroll_into_view_if_needed()
    _wait_for_chart(page, "chart-annual")
    cats = raw["chart-annual"]["x"]["categories"]
    year = next(c for c in cats[1:] if c.isdigit())
    want = [next(x for x in xs if x.startswith(year)), [x for x in xs if x.startswith(year)][-1]]
    pos = page.evaluate("""(cat) => { const c = Desk.charts['chart-annual'];
        const x = c.chart.convertToPixel({xAxisIndex: 0}, cat), r = c.el.getBoundingClientRect();
        return {x: r.left + x, y: r.top + r.height / 2}; }""", year)
    page.mouse.click(pos["x"], pos["y"])
    page.wait_for_timeout(400)
    assert page.evaluate(win_js, ids) == [want] * 3
    assert page.get_attribute("[data-zoom-for='chart-equity'][data-years='3']", "aria-pressed") == "false"
    assert page.input_value("[data-zoom-year-for='chart-equity']") == "|".join(want)
    # the year menu (the keyboard's way to the same windows)
    opts = page.evaluate("[...document.querySelectorAll('[data-zoom-year-for=\"chart-equity\"] option')]"
                         ".map(o => o.value)")
    page.select_option("[data-zoom-year-for='chart-equity']", opts[1])
    page.wait_for_timeout(400)
    assert page.evaluate(win_js, ids) == [opts[1].split("|")] * 3
    # the wheel on the drawdown chart zooms the group too
    page.locator("#chart-dd").scroll_into_view_if_needed()
    box = page.locator("#chart-dd").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.wheel(0, 400)
    page.wait_for_timeout(500)
    ww = page.evaluate(win_js, ids)
    assert len({tuple(w) for w in ww}) == 1, ww
    assert ww[0] != opts[1].split("|")
    ctx.close()


def test_browser_calibration_smiles_are_small_multiples(site_browser):
    """P4.4: the six Calibration smiles share one ln(K/F) domain and one y tick
    step, one legend bar drives all of them (Heston toggled once is off in every
    panel), each panel's head carries Heston's inside-spread share and median |z|,
    and one Table / CSV replaces the six pairs: the CSV has one row per quote of
    every panel. The page's tab stops fall by the 30 the six pairs and legends
    held (135 → 105 on the committed data)."""
    sb = site_browser
    html = (sb["site"] / "calibration.html").read_text()
    raw = _raw_specs(html)
    ids = [i for i in re.findall(r'id="(chart-smile-[^"]+)" data-spec', html) if raw[i].get("quotes")]
    assert len(ids) >= 2
    assert len({(raw[i]["x"]["min"], raw[i]["x"]["max"]) for i in ids}) == 1
    assert len({raw[i]["y"]["step"] for i in ids}) == 1
    assert not any(f'data-csv-for="{i}"' in html or f'data-table-for="{i}-data"' in html for i in ids)
    assert html.count("data-csv-group=") == 1
    for i in ids:
        exp = i[len("chart-smile-"):]
        head = re.search(rf'id="smile-{exp}".*?<span class="panel-meta">([^<]*)</span>', html,
                         re.DOTALL).group(1)
        assert re.match(r"Heston inside \d+% · median \|z\| \d+\.\d$", head), head
    ctx, page = _page(sb)
    page.goto(sb["base"] + "calibration.html", wait_until="load")
    for i in ids:
        page.locator("#" + i).scroll_into_view_if_needed()
        _wait_for_chart(page, i)
    ax = page.evaluate("(ids) => ids.map(id => Desk.charts[id].chart.getOption().xAxis"
                       ".map(a => [a.min, a.max]))", ids)
    assert len({json.dumps(a) for a in ax}) == 1, ax
    hk = next(s.get("legendKey") or s["name"] for s in raw[ids[0]]["series"] if s.get("quote") == "model")
    page.click(f"#smile-group-tools [data-series='{hk}']")
    page.wait_for_timeout(400)
    left = page.evaluate("""([ids, hk]) => ids.map(id => Desk.charts[id].chart.getOption().series
        .filter(s => s.name.endsWith(' ' + hk)).reduce((n, s) => n + (s.data || []).length, 0))""", [ids, hk])
    assert left == [0] * len(ids), left
    with page.expect_download() as dl:
        page.click("[data-csv-group]")
    rows = Path(dl.value.path()).read_text().strip().split("\n")
    assert rows[0].startswith("expiry,")
    n_quotes = sum(len(g["mid"]) for i in ids for g in raw[i]["quotes"]["groups"].values())
    assert len(rows) - 1 == n_quotes
    ctx.close()


def _delta_e76(a: str, b: str) -> float:
    """CIE76 ΔE between two '#rrggbb' colours (sRGB, D65)."""
    def lab(h: str) -> tuple[float, float, float]:
        rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b_ = (((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb)
        x = (0.4124 * r + 0.3576 * g + 0.1805 * b_) / 0.95047
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b_
        z = (0.0193 * r + 0.1192 * g + 0.9505 * b_) / 1.08883

        def f(t: float) -> float:
            return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
        return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))
    la, lb = lab(a), lab(b)
    return float(sum((p - q) ** 2 for p, q in zip(la, lb)) ** 0.5)


def _lch_hue(h: str) -> tuple[float, float]:
    """(chroma, hue in degrees) of '#rrggbb' in CIE LCh (the Lab of _delta_e76)."""
    import math
    rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b_ = (((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb)
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b_) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b_
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b_) / 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    a, bb = 500 * (f(x) - f(y)), 200 * (f(y) - f(z))
    return math.hypot(a, bb), math.degrees(math.atan2(bb, a)) % 360


def test_every_series_follows_the_encoding_registry(dash, pages):
    """P4.5 (extends test_one_colour_per_model_on_every_surface_chart): over every
    page's chart specs, (a) a series that stands for a model (role 'model') has
    its model's colour and label (desk_charts.MODEL_ENCODING: 'Heston', not
    'Heston (v2 fit)'); (b) on a page that draws models no other series takes a
    model colour (the Calibration tenors were cat0 / cat1 / cat4); (c) series
    indexed by maturity take the tenor ramp, lighter to darker with T; (d) on a
    P&L or return chart no series is within ΔE76 25 of the gain or loss fill
    (S3 was cat1, 16.6 from --neg-fill), nor, unless it is neutral (LCh chroma
    under 15), within 30° of hue of either: the Risk P&L explain drew Vega in
    vermilion cat5 (13° from the loss orange) under a gain-blue +403.5 tile and
    Gamma in sky-blue cat0 (29° from the gain blue), and Research drew S1C and
    the rolling Sharpe in cat0 (F7); (e) market observables and Cboe indices
    are 'headline'; (f) no series uses 'info' (the gain blue); (g) a risk-neutral
    tail (role 'tail') takes its side's TAIL_COLOURS hue: the Surface tails were
    all three SVI-cal cat0, 29° of hue from the gain blue, the upside told from
    the downside by its dash alone (B3-F13); the two sides are 60° or more of LCh
    hue apart and each 30° or more from the gain and loss fills. The ramp reaches
    3:1 on the panels."""
    ch = dash.charts
    tok = _tokens()
    hex_of = {"headline": tok["series-headline"], "muted": tok["series-muted"], "pos": tok["pos-fill"],
              "neg": tok["neg-fill"], **{f"cat{i}": tok[f"cat-{i}"] for i in range(6)},
              **{f"ramp{i}": tok[f"ramp-{i}"] for i in range(6)}, "cat-default": tok["cat-0"],
              **{c: tok[c] for c in ch.TAIL_COLOURS.values()}}

    def hue_gap(a: str, b: str) -> float:
        return abs((_lch_hue(a)[1] - _lch_hue(b)[1] + 180) % 360 - 180)
    down, up = (hex_of[ch.TAIL_COLOURS[side]] for side in ("below", "above"))
    assert hue_gap(down, up) >= 60, (down, up)
    for c in (down, up):
        assert _lch_hue(c)[0] >= 15, c
        for fill in ("pos-fill", "neg-fill"):
            assert hue_gap(c, tok[fill]) >= 30, (c, fill)
    assert not set(ch.TAIL_COLOURS.values()) & ch.MODEL_COLOUR_TOKENS
    for i in range(6):
        for bg in (tok["bg"], tok["surface"], tok["surface2"]):
            assert dash.contrast_ratio(tok[f"ramp-{i}"], bg) >= 3.0, (i, bg)
    assert dash.model_label("heston_v2") == "Heston"
    assert dash.model_label("svi_cal") == "SVI-cal"
    seen = collections.Counter()
    # a y axis that is P&L or a return (not a ratio that mentions P&L, as the vomma flag's)
    pnl_axis = re.compile(r"^(Cumulative )?(P&L|[Rr]eturn|Drawdown|Sharpe)\b")
    for route, html in pages.items():
        specs = _raw_specs(html)
        every = [(cid, i, s) for cid, sp in specs.items() for i, s in enumerate(sp.get("series") or [])]
        model_page = any(s.get("role") == "model" for _, _, s in every)
        for cid, i, s in every:
            c = s.get("color") or f"cat{i % 6}"
            role = s.get("role")
            where = (route, cid, s.get("name"))
            assert c != "info", where                                                        # (f)
            if role == "model":                                                              # (a)
                seen["model"] += 1
                assert c == ch.model_colour(s["model"]), where
                label = re.sub(r"^\d{4}-\d{2}-\d{2} ", "", s.get("legendKey") or s["name"])
                ml = ch.model_label(s["model"])
                assert label == ml or label.startswith((ml + " ", ml + ",")), (where, label, ml)
            elif model_page and not s.get("signed"):                                          # (b)
                assert c not in ch.MODEL_COLOUR_TOKENS, where
            if role in ("market", "cboe"):                                                  # (e)
                seen["market"] += 1
                assert c == "headline", where
            if role == "tail":                                                              # (g)
                seen["tail"] += 1
                assert c == ch.TAIL_COLOURS[s["side"]], where
        for cid, sp in specs.items():
            tenors = sorted((s["T"], s["color"]) for s in sp.get("series") or [] if s.get("role") == "tenor")
            if tenors:                                                                      # (c)
                seen["tenor"] += 1
                assert all(c in ch.TENOR_RAMP for _, c in tenors), (route, cid)
                idx = [ch.TENOR_RAMP.index(c) for _, c in tenors]
                assert idx == sorted(idx), (route, cid, tenors)
                assert idx[0] < idx[-1], (route, cid, tenors)
            if pnl_axis.search((sp.get("y") or {}).get("name") or ""):                     # (d)
                seen["pnl"] += 1
                for i, s in enumerate(sp.get("series") or []):
                    c = s.get("color") or f"cat{i % 6}"
                    if s.get("signed") or s.get("invisible") or c in ("pos", "neg"):
                        continue                    # the sign colours themselves, by design
                    for fill in ("pos-fill", "neg-fill"):
                        assert _delta_e76(hex_of[c], tok[fill]) >= 25, (route, cid, s["name"], c, fill)
                        chroma, hue = _lch_hue(hex_of[c])
                        gap = abs((hue - _lch_hue(tok[fill])[1] + 180) % 360 - 180)
                        assert chroma < 15 or gap >= 30, (route, cid, s["name"], c, fill, round(gap))
    assert min(seen["model"] / 20, seen["market"] / 8, seen["tenor"] / 2, seen["pnl"] / 3) >= 1, seen
    sides = {s.get("side") for s in _raw_specs(pages["/surface"])["chart-rnd-tails"]["series"]}
    assert seen["tail"] >= 2, seen
    assert sides == set(ch.TAIL_COLOURS), sides


# a subscript written out raw, as a reader would see it: S_T, VIX_T, g_min, RV_h (a
# symbol of up to four letters, '_', an index of up to three; not a path or a key)
_RAW_SUBSCRIPT = re.compile(r"(?<![\w/.-])[A-Za-z]{1,4}_[A-Za-z0-9]{1,3}(?![\w/.-])")


def _spec_names(sp: dict) -> list[tuple[str, str]]:
    """(field, text) of every name a reader of the chart sees: its title and
    description, the axis names, the series and legend names, the tooltip rows
    and the data table's caption and heads."""
    out = [("title", sp.get("title") or ""), ("describe", sp.get("describe") or "")]
    for ax in ("x", "y", "y2"):
        a = sp.get(ax) or {}
        out += [(f"{ax}.name", a.get("name") or ""), (f"{ax}.label", a.get("label") or "")]
    for s in sp.get("series") or []:
        out += [("series", s.get("name") or ""), ("series", s.get("legendKey") or "")]
    out += [("legend", it["name"]) for it in sp.get("legend_items") or []]
    out += [("tipRows", r["name"]) for r in sp.get("tipRows") or []]
    t = sp.get("table") or {}
    for tb in (t, *(t.get("groups") or [])):
        out.append(("caption", tb.get("caption") or ""))
        out += [("column", c["label"]) for c in tb.get("columns") or []]
    return out


def test_no_chart_name_axis_legend_or_tooltip_shows_a_raw_subscript(dash, pages):
    """B3-F13: the Surface tail chart's legend buttons and tooltips read 'SVI-cal
    P(S_T < 0.90 F)', the density's y axis 'Density of ln(S_T/F)' and the
    Calibration VX charts 'Heston E[VIX_T]', where the tables beside them set
    S<sub>T</sub>. Over every chart spec on every page, every subscript in a name
    the reader sees is one app.subs sets (desk.js sets the axis names and the
    tooltip rows with the same pattern), and the page leaves none raw: not in a
    chart's accessible name, its legend buttons, its description, nor in a
    table's caption or heads."""
    js = DESK_JS.read_text()
    assert re.search(r"var SUBS = /(.*?)/g;", js).group(1) == dash._SUBSCRIPTS.pattern
    for use in ("name: subs(a.name || '', 1)", "subs(p.seriesName)", "subs(r.name)"):
        assert use in js, use
    seen: collections.Counter = collections.Counter()
    n_charts = 0
    for route, html in pages.items():
        legends: dict[str, list[str]] = collections.defaultdict(list)
        buttons = re.findall(r'<button[^>]*data-legend-for="([^"]*)"[^>]*>(.*?)</button>', html, re.DOTALL)
        for ids, inner in buttons:
            for i in ids.split():
                legends[i].append(inner)
            # a set subscript sits in one span with its label: the button is a flex
            # box, and the bare <i> and <sub> stood apart as items ('P( S T < 0.80 F)')
            one_label = r'<span class="key-swatch[^>]*></span><span>.*</span>'
            assert "<sub>" not in inner or re.fullmatch(one_label, inner, re.DOTALL), (route, inner)
        for cid, sp in _raw_specs(html).items():
            n_charts += 1
            for field, text in _spec_names(sp):
                for tok in _RAW_SUBSCRIPT.findall(text):
                    seen[tok] += 1
                    assert dash._SUBSCRIPTS.fullmatch(tok), (route, cid, field, tok)
            shown = list(legends.get(cid, []))
            div = re.search(rf'<div class="chart[^"]*" id="{re.escape(cid)}"[^>]*>', html)
            if div:
                shown.append(_html_unescape(_attrs(div.group(0)).get("aria-label", "")))
            desc = re.search(rf'<p class="sr-only" id="{re.escape(cid)}-desc">(.*?)</p>', html, re.DOTALL)
            if desc:
                shown.append(desc.group(1))
            for text in shown:
                plain = _html_unescape(re.sub(r"<[^>]+>", "", text))
                assert not _RAW_SUBSCRIPT.search(plain), (route, cid, plain)
        for cell in re.findall(r"<(?:caption|th)\b[^>]*>(.*?)</(?:caption|th)>", html, re.DOTALL):
            plain = _html_unescape(re.sub(r"<[^>]+>", "", cell))
            assert not _RAW_SUBSCRIPT.search(plain), (route, plain)
    assert n_charts >= len(pages), n_charts
    assert seen["S_T"] >= 1, seen
    # the plain-text twin (an aria-label) and the markup agree on what they set
    assert dash.subs_text("E[VIX_T] and S_T") == "E[VIX T] and S T"
    assert str(dash.subs("E[VIX_T] and S_T")) == "E[VIX<sub>T</sub>] and <i>S</i><sub>T</sub>"


def test_browser_tail_hues_and_chart_subscripts_as_drawn(site_browser):
    """B3-F13 in the browser: the Surface tail chart draws the two downside tails
    in one colour (--tail-down) and the upside tail in another (--tail-up); the
    density's y axis name is rich text (an italic S, a subscript T), a tail's
    tooltip row sets S<sub>T</sub>, and the tails' legend buttons hold their swatch
    and one label, no bare text or <sub> as a flex item of its own."""
    sb = site_browser
    ctx, page = _page(sb)
    page.goto(sb["base"] + "surface.html", wait_until="load")
    for cid in ("chart-rnd", "chart-rnd-tails"):
        page.locator("#" + cid).scroll_into_view_if_needed()
        _wait_for_chart(page, cid)
    got = page.evaluate("""() => {
      const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const d = Desk.charts['chart-rnd'].chart.getOption();
      const t = Desk.charts['chart-rnd-tails'].chart.getOption();
      const tip = t.series.map(s => {
        const i = s.data.findIndex(v => v && v[1] !== null && v[1] !== undefined);
        const p = {seriesName: s.name, value: s.data[i], dataIndex: i, color: '#fff'};
        return i < 0 ? null : t.tooltip[0].formatter(p);
      });
      const bare = b => [...b.childNodes].filter(n => n.nodeType === 3 && n.textContent.trim()).length;
      const btns = [...document.querySelectorAll('[data-legend-for="chart-rnd-tails"]')];
      return {axis: d.yAxis[0].name, rich: Object.keys(d.yAxis[0].nameTextStyle.rich || {}),
              colours: t.series.map(s => [s.name, s.lineStyle.color]),
              down: css('--tail-down'), up: css('--tail-up'),
              tip: tip, kids: btns.map(b => [b.childElementCount, bare(b)])};
    }""")
    ctx.close()
    assert "_" not in got["axis"], got["axis"]
    assert "{i|S}{s|T}" in got["axis"], got["axis"]
    assert {"i", "s"} <= set(got["rich"])
    by_side = {">": set(), "<": set()}
    for name, colour in got["colours"]:
        by_side[">" if " > " in name else "<"].add(colour.lower())
    want = {"<": {got["down"].lower()}, ">": {got["up"].lower()}}
    assert by_side == want, (by_side, want)
    tips = [t for t in got["tip"] if t]
    assert tips, got["tip"]
    for t in tips:
        assert "<i>S</i><sub>T</sub>" in t, t
        assert "S_T" not in t, t
    assert got["kids"], got
    assert all(k == [2, 0] for k in got["kids"]), got["kids"]


def _on_ramp(colour: str, stops: tuple[str, ...] | list[str], tol: float = 2.5) -> bool:
    """Whether '#rrggbb' lies on the piecewise-linear sRGB ramp through ``stops``."""
    def rgb(h: str) -> np.ndarray:
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
    p = rgb(colour)
    for a, b in itertools.pairwise(stops):
        a_, b_ = rgb(a), rgb(b)
        d = b_ - a_
        t = 0.0 if not d.any() else float(np.clip(np.dot(p - a_, d) / np.dot(d, d), 0, 1))
        if np.linalg.norm(a_ + t * d - p) <= tol:
            return True
    return False


def test_heat_ramps_one_meaning_numeric_keys_and_no_small_text(dash, pages, site_browser):
    """P4.5: only the P&L heatmaps (Risk scenario grid, Research monthly returns)
    take the P&L ramp; every other heatmap cell is on cividis or the diverging
    ramp (DIVERGE_11, whose ends are neither a P&L hue nor breach red: berlin's
    blue end sat next to --pos), the vomma flag view on cividis (it ran to the
    loss orange); every ramp key has a number at both ends; no visible text in a
    heatmap is under 10.5 px (the vomma cells' second line was 9.5 px)."""
    pnl_tables = {"scenario-grid", "monthly-grid"}
    n = collections.Counter()
    for route, html in pages.items():
        for tid, body in re.findall(r'<table class="heat" id="([^"]+)"[^>]*>(.*?)</table>', html, re.DOTALL):
            for bg in re.findall(r'<td[^>]*style="background:(#[0-9a-fA-F]{6})', body):
                if tid in pnl_tables or tid.startswith("scenario"):
                    n["pnl"] += 1
                    assert _on_ramp(bg, dash.PNL_DIV), (route, tid, bg)
                else:
                    n["other"] += 1
                    assert _on_ramp(bg, dash.CIVIDIS_11) or _on_ramp(bg, dash.DIVERGE_11), (route, tid, bg)
        # each ramp key: a number before and after each bar
        for key in re.findall(r'<div class="ramp[^"]*"[^>]*>(.*?)</div>', html, re.DOTALL):
            bars = r'<span>([^<]*)</span><span class="ramp-bar"[^>]*></span><span>([^<]*)</span>'
            for lo, hi in re.findall(bars, key):
                n["keys"] += 1
                assert re.search(r"\d", lo), (route, lo, hi)
                assert re.search(r"\d", hi), (route, lo, hi)
    for e in dash.DIVERGE_11[:2] + dash.DIVERGE_11[-2:]:
        for token in ("pos", "pos-fill", "neg", "neg-fill", "breach"):
            assert _delta_e76(e, _tokens()[token]) >= 25, (e, token)
    assert min(n["pnl"], n["other"] / 10, n["keys"] / 8) >= 1, n
    sb = site_browser
    ctx, page = _page(sb, 320, 900)
    small = []
    for name in ("risk.html", "surface.html", "research.html"):
        page.goto(sb["base"] + name, wait_until="load")
        small += page.evaluate("""(name) => [...document.querySelectorAll('table.heat *')]
          .filter(e => [...e.childNodes].some(t => t.nodeType === 3 && t.textContent.trim())
            && e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden'
            && !e.closest('.sr-only') && parseFloat(getComputedStyle(e).fontSize) < 10.5)
          .map(e => name + ' ' + e.tagName + '.' + e.className + ' ' + getComputedStyle(e).fontSize)""", name)
    ctx.close()
    assert not small, small[:10]


def test_browser_tooltips_name_the_point_and_carry_units(site_browser):
    """P4.5: a tooltip's head carries the x value with its unit and the index
    name where there is one ('VIX3M · 93 d', not '93.0'), and every row its unit:
    Overview #chart-ts at the 93-day point (index pts), Calibration #chart-oos (vol pts) and
    Surface #chart-strip (%)."""
    sb = site_browser
    tip_js = """(id) => { const t = [...document.querySelectorAll('#' + id + ' > div')]
        .filter(d => d.style.position === 'absolute' && getComputedStyle(d).display !== 'none'
          && d.innerText.trim()); return t.length ? t[0].innerText : null; }"""
    ctx, page = _page(sb)

    def hover(name, cid, js_point):
        page.goto(sb["base"] + name, wait_until="load")
        page.locator("#" + cid).scroll_into_view_if_needed()
        _wait_for_chart(page, cid)
        pt = page.evaluate(js_point, cid)
        page.mouse.move(pt["x"], pt["y"])
        page.wait_for_timeout(300)
        return page.evaluate(tip_js, cid)

    at_point = """(cid) => { const c = Desk.charts[cid], s = c.spec.series[0];
        const p = s.data.find(d => d[0] === 93) || s.data[Math.floor(s.data.length / 2)];
        const px = c.chart.convertToPixel({seriesIndex: 0}, p), r = c.el.getBoundingClientRect();
        return {x: r.left + px[0], y: r.top + px[1]}; }"""
    tip = hover("index.html", "chart-ts", at_point)
    assert tip, "no tooltip on #chart-ts"
    assert "VIX3M · 93 d" in tip, tip
    assert re.search(r"\d\.\d+ index pts", tip), tip          # an index level (F6: was 'vol pts')
    mid = """(cid) => { const c = Desk.charts[cid], r = c.el.getBoundingClientRect();
        const g = c.chart.getModel().getComponent('grid', 0).coordinateSystem.getRect();
        return {x: r.left + g.x + g.width * 0.4, y: r.top + g.y + g.height * 0.6}; }"""
    tip = hover("calibration.html", "chart-oos", mid)
    assert tip, "no tooltip on #chart-oos"
    assert re.search(r"\d\.\d+ vol pts", tip), tip
    tip = hover("surface.html", "chart-strip", mid)
    assert tip, "no tooltip on #chart-strip"
    assert re.search(r"\d+ d", tip.split("\n")[0]), tip
    rows = [x for x in tip.split("\n")[1:] if x.strip()]
    assert rows, tip
    assert all(re.search(r"\d%$", x.strip()) for x in rows), rows
    ctx.close()


# ── Q20: the phone first screen ───────────────────────────────────────────────

def test_oldest_source_names_the_status_bars_oldest_input(dash):
    """Q20: the one-line phone status strip names the input with the most NYSE
    sessions of age; a missing input comes first; a reference-only source (VX,
    not in the bar) and an unaged one (the fixed backtest window) never."""
    src = [{"short": "SPX", "bar": True, "aged": True, "age": 0, "level": "ok"},
           {"short": "FIT", "bar": True, "aged": True, "age": 5, "level": "error"},
           {"short": "GREEKS", "bar": True, "aged": True, "age": 5, "level": "error"},
           {"short": "BACKTEST", "bar": True, "aged": False, "age": None, "level": "ok"},
           {"short": "VX", "bar": False, "aged": True, "age": 40, "level": "error"}]
    assert dash.oldest_source(src)["short"] == "FIT"              # ties: the earlier row
    assert dash.oldest_source([*src, {"short": "LABELS", "bar": True, "aged": True, "age": None,
                                      "level": "missing"}])["short"] == "LABELS"
    assert dash.oldest_source(src[3:]) is None


def test_every_page_carries_the_phone_status_strip_and_fold_hooks(dash, pages):
    """Q20, server side: every page's status bar holds a <details> whose summary
    names the oldest input and its age (load_chrome's sources) and whose list
    gives every status-bar source its date; each stale banner keeps its full
    text and role=status, its title naming the body it folds; the Overview's
    findings columns name their lists. Without JavaScript nothing is folded:
    no list or banner body is rendered hidden."""
    c = dash.load_chrome()
    o = c["oldest"]
    assert o is not None
    assert o["short"] in {s["short"] for s in c["sources"] if s["bar"]}
    want = ("missing" if o["level"] == "missing" else
            (dash.plural(o["age"], "session") + " old") if o["age"] else "current")
    for route, html in pages.items():
        bar = html[html.index('<header class="statusbar"'):html.index("</header>")]
        det = re.search(r'<details class="sb-phone">\s*<summary class="sb-item">(.*?)</summary>'
                        r'(.*?)</details>', bar, re.DOTALL)
        assert det, route
        summary = _visible_text("<body>" + det.group(1)).split(":")[0].split()
        assert summary == [o["short"], *want.split()], route
        listed = re.findall(r'<li><span class="sb-k">([^<]+)</span>', det.group(2))
        assert listed[:-3] == [s["short"] for s in c["sources"] if s["bar"]], route
        assert listed[-3:] == ["MODEL", "BUILD", "VIEW"], route
        banners = list(re.finditer(
            r'<div class="banner banner-stale-\w+" data-stale="(\w+)" data-age="\d+" role="status">'
            r'.*?<div class="banner-title" data-disclose="([\w-]+)">.*?'
            r'<div class="banner-body fold-body" id="([\w-]+)">', html, re.DOTALL))
        for m in banners:
            assert m.group(2) == m.group(3) == f"stale-{m.group(1)}-body", route
        assert len(banners) == html.count('class="banner banner-stale-'), route   # each one a status
        assert not re.search(r'<(ul|div) class="(find-list|banner-body)[^"]*"[^>]* hidden', html), route
        if "fold-body" in html:                    # no JavaScript: the <noscript> style unfolds every body
            assert ".banner-body.fold-body { display: block !important; }" in html, route
    assert ('<div class="ticker" role="region" aria-label="Market ticker, last close" tabindex="0">'
            in pages["/"])
    heads = re.findall(r'<h3 class="find-head" id="find-(\w+)" data-disclose="find-list-(\w+)" '
                       r'data-count="(\d+)">', pages["/"])
    assert len(heads) == 3
    assert all(a == b for a, b, _ in heads)
    for (_, head, items), (_, _, n) in zip(_findings(pages["/"]), heads):
        assert int(n) == len(items), head


def test_browser_phone_first_screen_shows_the_subject_and_first_number(site_browser):
    """Q20 acceptance at 390 x 844: on every page <main> starts at y <= 110 (it was
    164: a 3-line status strip and a 3-line ticker), the h1 at y <= 200 (292 on
    Overview, Surface and Calibration, under a 108-198 px stale banner) and the
    first key number (.kpi-value, .result-v, or a Calibration parameter tile)
    above y = 844 (Overview 1,484, Calibration 2,043 at 'Fit quality'). The
    staleness stays on screen without a tap: the strip's state reads STALE and
    names the oldest input; the stale banner's title stays. The strip's
    <details> and each folded heading open from the keyboard. At 1440 nothing
    is folded and the one-line strip is not shown."""
    sb = site_browser
    ctx, page = _page(sb, 390, 844)
    geo_js = """() => {
      const top = e => e ? Math.round(e.getBoundingClientRect().top + scrollY) : null;
      const vis = e => e && e.offsetParent !== null && getComputedStyle(e).visibility !== 'hidden';
      const num = [...document.querySelectorAll('main .kpi-value, main .result-v, main .param-val')]
        .filter(vis)[0];
      const state = document.querySelector('.statusbar .sb-state');
      const phone = document.querySelector('.statusbar .sb-phone > summary');
      const stale = [...document.querySelectorAll('.banner[data-stale]')];
      return {main: top(document.querySelector('main')), h1: top(document.querySelector('h1')),
              num: top(num), state: vis(state) ? state.innerText.trim() : null,
              phone: vis(phone) ? phone.innerText.trim() : null,
              stale_titles: stale.map(b => { const t = b.querySelector('.banner-title');
                const r = t.getBoundingClientRect(); return r.height > 0 ? t.innerText.trim() : null; }),
              sw: document.documentElement.scrollWidth}; }"""
    try:
        for name in _ROUTE_OF:
            page.goto(sb["base"] + name, wait_until="load")
            g = page.evaluate(geo_js)
            assert g["main"] <= 110, (name, g)
            assert g["h1"] <= 200, (name, g)
            if g["num"] is not None or name in ("index.html", "surface.html", "calibration.html",
                                                "risk.html", "research.html"):
                assert g["num"] is not None, (name, g)
                assert g["num"] < 844, (name, g)
            assert g["sw"] <= 390, (name, g)
            assert g["phone"], (name, g)
            assert re.search(r"\b(old|current|missing)\b", g["phone"]), (name, g)
            if g["stale_titles"]:
                assert re.search(r"STALE|LAGGING", g["state"] or ""), (name, g)
                assert all(t and " old" in t for t in g["stale_titles"]), (name, g)
        # the strip's details from the keyboard: skip link, then the summary
        page.goto(sb["base"] + "index.html", wait_until="load")
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")
        assert page.evaluate("document.activeElement.matches('.sb-phone > summary')")
        page.keyboard.press("Enter")
        assert page.evaluate("document.querySelector('.sb-phone').open")
        assert page.locator(".sb-list").is_visible()
        page.keyboard.press("Enter")
        assert not page.evaluate("document.querySelector('.sb-phone').open")
        # a folded heading is a button; Enter unfolds it
        btn = page.locator("#find-works button.disclose")
        assert btn.get_attribute("aria-expanded") == "false"
        assert not page.locator("#find-list-works").is_visible()
        btn.focus()
        page.keyboard.press("Enter")
        assert btn.get_attribute("aria-expanded") == "true"
        assert page.locator("#find-list-works").is_visible()
        stale = page.locator(".banner[data-stale] button.disclose")
        if stale.count():
            body = page.locator(".banner[data-stale] .banner-body").first
            assert not body.is_visible()
            stale.first.focus()
            page.keyboard.press("Space")
            assert body.is_visible()
    finally:
        ctx.close()
    ctx, page = _page(sb, 1440, 900)
    try:
        page.goto(sb["base"] + "index.html", wait_until="load")
        assert not page.locator(".sb-phone").is_visible()
        assert page.locator(".statusbar .sb-right").is_visible()
        assert page.locator("#find-list-works").is_visible()
        assert all(b.is_visible() for b in page.locator(".banner[data-stale] .banner-body").all())
        assert page.locator("button.disclose").count() == 0      # nothing folds: headings stay headings
        page.set_viewport_size({"width": 390, "height": 844})     # turned narrow: folds
        page.wait_for_timeout(200)
        assert not page.locator("#find-list-works").is_visible()
        assert page.locator("#find-works button.disclose").get_attribute("aria-expanded") == "false"
    finally:
        ctx.close()
    ctx = sb["browser"].new_context(viewport={"width": 390, "height": 844}, java_script_enabled=False)
    try:                                           # no JavaScript: nothing folds
        page = ctx.new_page()
        page.goto(sb["base"] + "index.html", wait_until="load")
        assert page.locator("#find-list-works").is_visible()
        assert all(b.is_visible() for b in page.locator(".banner[data-stale] .banner-body").all())
    finally:
        ctx.close()


def test_browser_phone_status_list_closes_on_escape_and_outside(site_browser):
    """A1 (F16): the phone strip's source list (details.sb-phone) stayed open on
    Escape and on a click or a tap elsewhere, over the page head. At 390 px
    Escape closes it and focus is back on its summary; a click or a tap
    outside closes it; a click or a tap on its own list does not."""
    sb = site_browser
    ctx = sb["browser"].new_context(viewport={"width": 390, "height": 844}, has_touch=True)
    ctx.set_default_timeout(BROWSER_TIMEOUT_MS)
    is_open = "document.querySelector('.sb-phone').open"
    try:
        page = ctx.new_page()
        for name in ("index.html", "calibration.html"):
            page.goto(sb["base"] + name, wait_until="load")
            summary = page.locator(".sb-phone > summary")
            summary.focus()
            page.keyboard.press("Enter")
            assert page.evaluate(is_open), name
            page.locator(".sb-list li").first.click()            # its own list: stays open
            assert page.evaluate(is_open), name
            page.keyboard.press("Escape")
            assert not page.evaluate(is_open), name
            assert page.evaluate("document.activeElement.matches('.sb-phone > summary')"), name
            summary.click()
            assert page.evaluate(is_open), name
            # a point in the left gutter below the open list: a long list covers the h1
            lb = page.locator(".sb-phone .sb-list").bounding_box()
            out = (4, min(840, lb["y"] + lb["height"] + 24))
            page.mouse.click(*out)
            assert not page.evaluate(is_open), name
            summary.tap()
            assert page.evaluate(is_open), name
            page.locator(".sb-list li").first.tap()
            assert page.evaluate(is_open), name
            page.touchscreen.tap(*out)
            assert not page.evaluate(is_open), name
    finally:
        ctx.close()


# ── D12: sortable, copyable tables ────────────────────────────────────────────

def test_tables_are_published_in_the_servers_order_without_sort_controls(pages):
    """D12: sorting and copying are desk.js enhancements; the HTML (what a page
    shows without JavaScript and in print) carries no sort button, arrow or copy
    control; the only aria-sort is the Risk vomma table's, which the server
    publishes sorted."""
    for route, html in pages.items():
        for token in ("th-sort", "tbl-tools", "tbl-act"):
            assert token not in html, (route, token)
        assert html.count("aria-sort=") == (1 if route == "/risk" else 0), route
    # a table whose order is its meaning (the cumulative C23 waterfall) is left out
    assert '<table class="tbl dense cards" id="c23-bridge" data-nosort>' in pages["/corrections"]


def test_browser_tables_sort_copy_and_export(site_browser, tmp_path):
    """D12 acceptance on research.html: clicking the 'Sharpe' head of Per-sleeve
    performance orders the rows ascending, then descending, then back to the
    published order, and aria-sort follows; 'Copy' writes a TSV whose cells are
    the cells' text as written; 'CSV' downloads the head and every row. The
    number parser reads U+2212, $, %, $…k, 'm × 10⁻ⁿ', 'a / b', 'a to b' and
    'a–b', and refuses dates and tenors; on Surface's tail table the 'beyond
    quoted strikes' cells sort last both ways."""
    sb = site_browser
    ctx = _context(sb)
    ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=sb["base"].rstrip("/"))
    page = ctx.new_page()
    try:
        page.goto(sb["base"] + "research.html", wait_until="load")
        parse = page.evaluate("""() => ['\\u2212$273,733', '12.5%', '$1.2k', '$3.4M',
            '9.5 \\u00d7 10\\u207b\\u2079', '205 / 110', '\\u22121.81 to \\u22120.30', '0.703\\u20131.061',
            '46 (1 open)', '2026-10-16', '1M',
            '\\u2014', 'S3 \\u00b7 short SPX straddle', '+0.42'].map(Desk.cellNumber)""")
        assert parse[:9] == [-273733, 12.5, 1200, 3.4e6, pytest.approx(9.5e-9), 205, -1.81, 0.703, 46]
        assert parse[9:13] == [None, None, None, None]
        assert parse[13] == 0.42
        tbl = page.locator("#sleeve-table")
        heads = [h.strip() for h in tbl.locator("thead th").all_text_contents()]
        col = heads.index("Sharpe")
        published = [r.strip() for r in tbl.locator("tbody tr td:first-child").all_text_contents()]

        def sharpe() -> list[float]:
            return page.evaluate("""(j) => [...document.querySelectorAll('#sleeve-table tbody tr')]
                .map(r => Desk.cellNumber(r.cells[j].textContent))""", col)
        th = tbl.locator("thead th").nth(col)
        btn = th.locator("button.th-sort")
        btn.click()
        assert th.get_attribute("aria-sort") == "ascending"
        assert sharpe() == sorted(sharpe())
        btn.click()
        assert th.get_attribute("aria-sort") == "descending"
        assert sharpe() == sorted(sharpe(), reverse=True)
        btn.press("Enter")                                  # the keyboard, a third time: published order
        assert th.get_attribute("aria-sort") is None
        assert [r.strip() for r in tbl.locator("tbody tr td:first-child").all_text_contents()] == published
        # Copy: the cells as written (the chip's 'portfolio', not the CSS capitals), tab-separated
        want = page.evaluate("""() => { const t = document.getElementById('sleeve-table');
            const txt = c => c.textContent.replace(/\\s+/g, ' ').trim();
            return [[...t.tHead.rows[0].cells].map(txt),
                    ...[...t.tBodies[0].rows].map(r => [...r.cells].map(txt))]
              .map(r => r.join('\\t')).join('\\n') + '\\n'; }""")
        page.locator("#sleeves .panel-actions button.tbl-act", has_text="Copy").click()
        page.wait_for_timeout(200)
        got = page.evaluate("navigator.clipboard.readText()")
        assert got == want
        assert got.splitlines()[0].split("\t")[:3] == ["Signal", "Total P&L", "Sharpe"]
        n = tbl.locator("tbody tr").count()
        with page.expect_download() as dl:
            page.locator("#sleeves .panel-actions button.tbl-act", has_text="CSV").click()
        path = tmp_path / "sleeves.csv"
        dl.value.save_as(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 + n
        assert lines[0].startswith("Signal,Total P&L,Sharpe,")
        assert dl.value.suggested_filename == "sleeve-table.csv"
        # numbers before text: 'beyond quoted strikes' sorts last both ways
        page.goto(sb["base"] + "surface.html", wait_until="load")
        t2 = page.locator("#rnd-table-tbl")
        h2 = [h.strip() for h in t2.locator("thead th").all_text_contents()]
        j = h2.index("ST > 1.10 F")
        cells = """(j) => [...document.querySelectorAll('#rnd-table-tbl tbody tr')]
            .map(r => r.cells[j].textContent.trim())"""
        before = page.evaluate(cells, j)
        n_beyond = before.count("beyond quoted strikes")
        assert 0 < n_beyond < len(before)
        b2 = t2.locator("thead th").nth(j).locator("button.th-sort")
        for _ in range(2):
            b2.click()
            after = page.evaluate(cells, j)
            assert after[-n_beyond:] == ["beyond quoted strikes"] * n_beyond
            nums = [float(x) for x in after[:-n_beyond]]
            assert nums in (sorted(nums), sorted(nums, reverse=True))
    finally:
        ctx.close()


_TABLE_CSV_JS = """(id) => { const t = document.getElementById(id), w = t.closest('.tbl-wrap') || t;
  const own = w.previousElementSibling, bar = own && own.classList.contains('tbl-tools') ? own
    : t.closest('.panel').querySelector('.panel-actions');
  [...bar.querySelectorAll('button.tbl-act')].find(b => b.textContent === 'CSV').click(); }"""


def _table_csv(page, table_id: str, tmp_path) -> list[list[str]]:
    """The rows of the CSV a table's own CSV button downloads."""
    import csv

    with page.expect_download() as dl:
        page.evaluate(_TABLE_CSV_JS, table_id)
    path = tmp_path / f"{table_id}.csv"
    dl.value.save_as(path)
    text = path.read_text(encoding="utf-8")
    assert "\u2212" not in text, table_id
    return list(csv.reader(text.splitlines()))


def test_browser_table_csv_carries_numbers_not_display_strings(site_browser, tmp_path):
    """R3-T9: a table's CSV carried the cells as displayed, so a spreadsheet read
    '\u22120.99' (U+2212) as text and the delta grid's tenor ran into its
    bracketing expiries ('1M26-11-13 \u2192 \u2026'). A cell that is one number
    exports as that number, any other cell as its text with an ASCII minus (a
    'lo to hi' interval stays text), and a block child is a word of its own."""
    sb = site_browser
    ctx = _context(sb)
    page = ctx.new_page()
    try:
        page.goto(sb["base"] + "research.html", wait_until="load")
        got = page.evaluate("""() => ['\u22120.99', '+0.84', '\u2212$273,733', '12.5%', '$1.2k',
            '9.5 \u00d7 10\u207b\u2079', '\u22121.81 to \u22120.30', '1M', '2026-10-16', '\u2014',
            '46 (1 open)'].map(Desk.csvValue)""")
        assert got == [-0.99, 0.84, -273733, 12.5, 1200, 9.5e-9, "-1.81 to -0.30", "1M",
                       "2026-10-16", "\u2014", "46 (1 open)"]
        shown = page.evaluate("""() => [...document.querySelectorAll('#bootstrap-table tbody tr')]
            .map(r => [...r.cells].map(c => c.textContent.replace(/\\s+/g, ' ').trim()))""")
        rows = _table_csv(page, "bootstrap-table", tmp_path)
        assert len(rows) == 1 + len(shown)
        j = rows[0].index("Interval")
        one = re.compile(r"[+\u2212-]?\d+(?:\.\d+)?")
        n_num = 0
        for want, row in zip(shown, rows[1:]):
            assert len(row) == len(want)
            for w, cell in zip(want, row):
                if one.fullmatch(w):            # a number: written plainly, its sign ASCII
                    n_num += 1
                    assert float(cell) == float(w.replace("\u2212", "-")), (w, cell)
                    assert not cell.startswith("+")
                else:
                    assert cell == w.replace("\u2212", "-"), (w, cell)
            assert " to " in row[j]
            assert not one.fullmatch(row[j])           # the interval stays text
        assert n_num > 0
        # Surface's constant-maturity grid: the tenor, its expiry pair and the
        # strike count are separate words, and every quote cell is a number
        page.goto(sb["base"] + "surface.html", wait_until="load")
        heads = page.evaluate("""() => [...document.querySelectorAll('#delta-grid-cm tbody th')]
            .map(th => [...th.childNodes].map(n => n.textContent.trim()).filter(Boolean).join(' '))""")
        rows = _table_csv(page, "delta-grid-cm", tmp_path)
        assert [r[0] for r in rows[1:]] == heads
        assert any(" " in h for h in heads)
        shown = page.evaluate("""() => [...document.querySelectorAll('#delta-grid-cm tbody tr')]
            .map(r => [...r.cells].slice(1).map(c => c.textContent.trim()))""")
        for want, row in zip(shown, rows[1:]):
            for w, cell in zip(want, row[1:]):
                if one.fullmatch(w):
                    assert float(cell) == float(w.replace("\u2212", "-")), (w, cell)
    finally:
        ctx.close()


# ── Q14: conventions and definitions, linked from first use ──────────────────

def _glossary():
    return _load("desk_glossary_under_test", "dashboard/desk_glossary.py")


def test_definitions_section_lists_every_term_with_an_anchor(pages):
    """Model Risk #definitions: the units and signs, then one anchor per term
    (≥ 20) with its <dfn>, expansion, units and where it is used; the footer and
    the '?' overlay of every page link to it."""
    gl = _glossary()
    mr = pages["/model-risk"]
    sec = mr[mr.index('id="definitions"'):]
    sec = sec[:sec.index("</section>")] if "</section>" in sec else sec
    assert 'id="conventions"' in sec
    for k, _v in gl.CONVENTIONS:
        assert f"<dt>{k}</dt>" in sec, k
    assert len(gl.TERMS) >= 20
    for t in gl.TERMS:
        m = re.search(rf'<div class="def" id="def-{re.escape(t["id"])}"[^>]*><dt><dfn[^>]*>(.*?)</dfn>', sec)
        assert m, t["id"]
        assert _html_unescape(m.group(1)) == t["term"]
        block = sec[m.start():sec.index("</div>", m.start())]
        assert "Units: " in block, t["id"]
        assert "used on " in block, t["id"]
    for route, html in pages.items():
        footer = html[html.index("<footer"):]
        assert 'href="model-risk.html#definitions"' in footer, route
        keys = html[html.index('id="keysDialog"'):]
        assert 'href="model-risk.html#definitions"' in keys[:keys.index("</dialog>")], route


class _ProseRuns:
    """The text runs of <main>'s running prose (p, li, dd) a reader sees, in
    page order, each tagged with the glossary link it sits in ('' outside one)
    and the definitions it is part of (a definition never links its own term):
    an html.parser walk independent of desk_glossary's regex tokenizer.
    Headings, tables, controls, code, SVG, <dt> and hidden, aria-hidden,
    sr-only and chip text are not prose."""

    SKIP = frozenset({"h1", "h2", "h3", "h4", "th", "td", "button", "summary", "code", "pre", "kbd",
                      "script", "style", "svg", "title", "option", "select", "textarea", "label",
                      "noscript", "dt"})
    VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
                      "source", "track", "wbr"})

    def __init__(self, html: str):
        from html.parser import HTMLParser

        runs: list[tuple[str, str, set]] = []
        stack: list[tuple[str, bool, str, str]] = []
        skip, void = self.SKIP, self.VOID

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag in void:
                    return
                a = dict(attrs)
                cls = (a.get("class") or "").split()
                hide = (tag in skip or "hidden" in a or a.get("aria-hidden") == "true"
                        or "sr-only" in cls or "chip" in cls)
                term = ""
                if tag == "a":
                    m = re.fullmatch(r"[^#]*#def-([\w-]+)", a.get("href") or "")
                    term = m.group(1) if m and "term" in cls else ""
                    hide = hide or not term                     # an ordinary link's text is not linked
                stack.append((tag, hide, term, a.get("data-term") or ""))

            def handle_endtag(self, tag):
                for k in range(len(stack) - 1, -1, -1):
                    if stack[k][0] == tag:
                        del stack[k:]
                        break

            def handle_data(self, data):
                if any(e[1] for e in stack) or not any(e[0] in ("p", "li", "dd") for e in stack):
                    return
                runs.append((data, next((e[2] for e in reversed(stack) if e[2]), ""),
                             {e[3] for e in stack if e[3]}))

        main = html[html.index("<main"):html.rindex("</main>")]
        P(convert_charrefs=True).feed(main)
        self.runs = runs


def test_every_page_links_the_first_use_of_its_terms(pages):
    """Every page links each glossary term once, at its first use in the prose
    of <main>, to its definition (on Model Risk, the same page); an acronym
    carries its expansion in <abbr>. Before Q14 no page had an <abbr> or <dfn>."""
    gl = _glossary()
    by_id = {t["id"]: t for t in gl.TERMS}
    seen: set[str] = set()
    for route, html in pages.items():
        main = html[html.index("<main"):html.rindex("</main>")]
        links = re.findall(r'<a class="term" href="([^"]*)#def-([\w-]+)">(.*?)</a>', main)
        assert links, route
        ids = [i for _h, i, _x in links]
        assert len(ids) == len(set(ids)), (route, ids)            # the first use only
        for href, i, inner in links:
            assert i in by_id, (route, i)
            assert href == ("" if route == "/model-risk" else "model-risk.html"), (route, href)
            t = by_id[i]
            if t["abbr"]:
                assert inner.startswith('<abbr title="'), (route, i)
                assert _html_unescape(_attrs(inner[:inner.index(">") + 1])["title"]) == t["abbr"]
            else:
                assert "<abbr" not in inner
            assert re.fullmatch(t["pat"], re.sub(r"<[^>]+>", "", inner)), (route, i, inner)
        # each term's first prose use is its link; a term in the prose is linked
        runs = _ProseRuns(html).runs
        for t in gl.TERMS:
            pat = re.compile(t["pat"].replace("&amp;", "&"))
            hits = [(k, link) for k, (text, link, own) in enumerate(runs)
                    if pat.search(text) and t["id"] not in own]
            if not hits:
                assert t["id"] not in ids, (route, t["id"])
                continue
            assert hits[0][1] == t["id"], (route, t["id"], runs[hits[0][0]][0][:80])
        seen |= set(ids)
    # the acronym load the review counted is covered where a reader meets it
    assert {"psr", "dsr", "spa", "stepm", "pbo", "holm", "qlike", "z-vrp"} <= set(
        gl.first_uses(pages["/research"]))
    assert {"svi-cal", "half-spread", "atmf", "rr25"} <= set(gl.first_uses(pages["/surface"]))
    assert len(seen) >= 20


def test_a_definition_never_links_to_itself(pages):
    mr = pages["/model-risk"]
    for m in re.finditer(r'<div class="def" id="def-([\w-]+)">(.*?)</div>', mr, flags=re.DOTALL):
        assert f'href="#def-{m.group(1)}"' not in m.group(2), m.group(1)


# ── B8: the volatility cone with the surface's ATM term structure ────────────

def test_vol_cone_ends_on_the_surface_session_and_matches_the_module(dash, pages):
    """#vol-cone: the percentiles of SPX RV at 10–252 sessions in both samples, each
    ending on the surface's session (no look-ahead: the database runs later), the
    RV of the windows ending that day and the SVI-cal ATMF at T·252/365 sessions,
    named with the session; the table lists the percentiles by horizon; the
    rank table reads the module. Before B8 the Surface had no realised vol."""
    cone = _load("desk_cone_under_test", "dashboard/desk_cone.py")
    surf = _surface_file("surface.json")
    session = surf.parent.name
    html = pages["/surface"]
    assert 'id="vol-cone"' in html
    assert 'id="vol-cone-rank"' in html
    spec = _spec(html, "chart-vol-cone")
    close = dash._read_spx()["close"]
    assert close.index.max() >= pd.Timestamp(session)
    by_name = {s["name"]: s for s in spec["series"]}
    assert by_name[f"SVI-cal ATMF {session}"]["role"] == "model"
    rv_now = by_name[f"SPX RV, windows ending {session}"]
    sj = json.loads(surf.read_text())
    res = pd.read_parquet(surf.parent / "residuals.parquet")
    listed = _load("desk_surface_under_test", "dashboard/desk_surface.py").delta_grid(sj, res, None)
    table = {(r[0], r[2]): r for r in spec["table"]["rows"]}
    tab_values = re.findall(r'role="tab" aria-selected="(?:true|false)" aria-controls="vol-cone-panel" '
                            r'data-value="([\w-]+)"', html)
    assert tab_values == [k for k, _l, _s in cone.SAMPLES]
    for key, label, _short in cone.SAMPLES:
        start = cone.sample_start(key, close, session)
        r = cone.sample_returns(close, start, session)
        assert str(r.index[-1].date()) == session
        rows = cone.cone(r)
        for c in rows:
            assert c["last"] == session
            for pc in cone.PCTS:
                s = by_name[f"{pc}th pct RV · {label.lower()}"]
                assert s["group"] == key
                got = {int(x): y for x, y in s["data"]}[c["h"]]
                assert got == pytest.approx(c["pct"][pc], rel=1e-3)
            row = table[(str(c["h"]), label)]
            assert row[4:9] == [f"{c['pct'][pc]:.1f}" for pc in cone.PCTS]
            assert row[3] == f"{c['n']:,}"
        if key == "since-2010":
            assert [y for _x, y in rv_now["data"]] == pytest.approx([c["current"] for c in rows], rel=1e-3)
            pts, beyond = cone.implied_points(listed, r)
            got = by_name[f"SVI-cal ATMF {session}"]["data"]
            assert [x for x, _ in got] == pytest.approx([p["sessions"] for p in pts], rel=1e-3)
            assert [y for _, y in got] == pytest.approx([p["atmf"] for p in pts], rel=1e-3)
            assert all(p["sessions"] <= 252 for p in pts)
            assert len(pts) + beyond == sum(1 for x in listed if x.get("atmf") is not None)
            # the rank table, row by row
            tbl = html[html.index('id="vol-cone-rank-tbl"'):]
            tbl = tbl[:tbl.index("</table>")]
            body = [[_html_unescape(re.sub(r"<[^>]+>", "", c_)).strip()
                     for c_ in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
                    for tr in re.findall(r"<tr>(.*?)</tr>", tbl[tbl.index("<tbody>"):])]
            assert [b[0] for b in body] == [p["expiry"] for p in pts]
            assert [b[3] for b in body] == [f"{p['atmf']:.2f}" for p in pts]
            assert [b[4] for b in body] == [f"{p['rank']:.0f}" for p in pts]
    assert len(spec["table"]["rows"]) == len(cone.HORIZONS) * len(cone.SAMPLES)
    def txt(frag: str) -> str:
        return _html_unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", frag)))
    note = txt(html[html.index('id="vol-cone-note"'):html.index('id="vol-cone-rank"')])
    assert f"since 2010 2010-01-04 → {session}" in note
    assert "T·252/365 sessions" in note
    rank_note = txt(html[html.index('id="vol-cone-rank-note"'):][:3000])
    assert "variance risk premium" in rank_note
    assert "intraday" in rank_note


def test_cone_percentile_labels_are_not_vol_levels(pages):
    """F12: the cone's end labels read '95%', '50%', '5%' on a vol-in-% axis, so
    '95%' looked like a vol level; they are 'p95' … 'p5'."""
    spec = _raw_specs(pages["/surface"])["chart-vol-cone"]
    ends = [s_["endLabel"] for s_ in spec["series"] if s_.get("endLabel")]
    assert ends
    assert all(re.fullmatch(r"p\d{1,2}", e) for e in ends), ends


def test_every_1m_on_the_surface_is_the_30_day_constant_maturity(pages):
    """'1M' is the SVI-cal surface interpolated to exactly 30 days (the KPI tile,
    the constant-maturity grid); a listed expiry is named by its date and days,
    never '1M'. Before the fix the cone headline read '1M: … 2026-11-16 (31 d) is
    13.82%' beside the 13.92% tile, and the density legend, its table and the
    tail table called the 31-day slice '1M'."""
    html = pages["/surface"]
    specs = _raw_specs(html)
    strings = [_visible_text(html)]
    for sp in specs.values():
        strings += [str(s_.get("name", "")) for s_ in sp.get("series", [])]
        strings += [str(c["label"]) for c in (sp.get("table") or {}).get("columns", [])]
        strings.append(str(sp.get("title", "")))
    text = " ¦ ".join(strings)
    tile = re.search(r"1M ATMF (\d+\.\d{2}) ?%", text)
    assert tile, "the 1M ATMF tile"
    v30 = tile.group(1)
    # '1M' never names a listed date (a constant-maturity row's 'lower → upper' is the bracket)
    assert not re.findall(r"\b1M\b\s*[·:(]?\s*\d{2,4}-\d{2}-\d{2}(?!\s*→)", text)
    # (a tile's sub-line ending in a date runs into the next tile's '1M RR25' label)
    assert not re.findall(r"\d{4}-\d{2}-\d{2}\)?\s*1M\b(?! (?:ATMF|RR25|BF25|VS))", text)
    assert all("1M" not in r[0] for r in _tbody_rows(html, "rnd-table-tbl"))
    assert not re.findall(r"\b1M\s*:", text)
    # every percentage printed against '1M' is the 30-day value
    vals = re.findall(r"\b1M\b[^0-9¦]{0,40}?(\d+\.\d{2}) ?%", text)
    assert vals
    assert set(vals) == {v30}, vals
    head = _visible_text("<body>" + html[html.index('id="vol-cone-1m"'):].split("</p>")[0])
    listed = re.search(r"ATMF of (\d+\.\d{2})%", head)
    assert listed
    assert "nearest 30 days" in head
    if listed.group(1) != v30:                 # the two are told apart where both print
        assert f"1M tile’s {v30}%" in head


def test_each_sleeve_has_one_name_on_every_page(dash, pages):
    """One registry label per sleeve (dashboard.SLEEVE_LABELS), read by every
    table, legend, category and sentence. Before, research.html called S3 'S3
    index-leg short' in the B12 tables and charts and 'S3 · short SPX straddle'
    in the per-sleeve table right above them, S1C 'S1C Contrarian' and 'S1C
    contrarian', and S1 'S1 PDV (ref)', 'S1 (reference)' and 'S1 · PDV short-VRP'."""
    labels = dash.SLEEVE_LABELS
    assert len(set(labels.values())) == len(labels)
    code = re.compile(r"^(S1C|S1|S2|S3|S4)\b")
    # a sleeve code followed by a descriptor is a name of that sleeve
    descr = r"\s*(?:·\s*)?(?:[Cc]ontrarian|PDV|VRP|VIX|index-leg|short SPX|[Dd]ispersion|\(ref(?:erence)?\))"
    marks = ("S4 short entry", "S4 long entry")         # the VRP chart's entry markers, not a sleeve name
    for route, html in pages.items():
        strings = [_visible_text(html)]
        for sp in _raw_specs(html).values():
            names = [str(x.get("name", "")) for x in sp.get("series", [])]
            names += [str(c) for ax in ("x", "y") for c in ((sp.get(ax) or {}).get("categories") or [])]
            names += [str(r[0]) for r in (sp.get("table") or {}).get("rows", []) if r]
            for n in names if route == "/research" else ():   # elsewhere a row may be a finding's title
                m = code.match(n)
                if m and not n.startswith(marks):
                    key = m.group(1).lower()
                    assert n.startswith(labels[key]), (route, n)
            strings += names
        heads = [_html_unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<tr[^>]*>\s*<t[dh][^>]*>(.*?)</t[dh]>", html, re.DOTALL)]
        for n in heads if route == "/research" else ():
            m = code.match(n)
            if m:
                assert n.startswith(labels[m.group(1).lower()]), (route, n)
        text = " ¦ ".join(strings + heads)
        for key, label in labels.items():
            if key == "portfolio":
                continue
            c = key.upper()
            for m in re.finditer(rf"\b{c}\b{descr}", text):
                assert text[m.start():].startswith(label), (route, text[m.start():m.start() + 40])
    research = _visible_text(pages["/research"])
    for key in ("s1c", "s3", "s4", "s1", "s2"):
        assert labels[key] in research, key


# ── B12: round trips and worst drawdowns ──────────────────────────────────────

def _tbody_rows(html: str, table_id: str) -> list[list[str]]:
    t = html[html.index(f'id="{table_id}"'):]
    t = t[t.index("<tbody>"):t.index("</table>")]
    return [[_html_unescape(re.sub(r"<[^>]+>", "", c)).strip()
             for c in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.DOTALL)]


def test_trade_pnl_histogram_puts_gains_on_top(pages):
    """F18: the horizontal P&L histogram listed '< −60' first, so the largest
    losses sat at the top, the reverse of every other P&L axis on the page; the
    first category (drawn at the top) is now the largest gains."""
    spec = _raw_specs(pages["/research"])["chart-trade-pnl-hist"]
    xs = spec["xs"]
    assert xs[0].startswith("≥ +"), xs
    assert xs[-1].startswith("< −"), xs
    assert [r[0] for r in spec["table"]["rows"]] == xs


def test_trade_stats_mean_names_both_dollar_bases(dash, pages):
    """F8: the Trades tile's 'average −$1,083 of portfolio P&L per trade' and the
    trade-statistics note's 'a mean of −3.2k per trade' are the same trades in
    two units (the portfolio is the equal-weight mean of three sleeves); the
    note now says both, and the two agree."""
    html = pages["/research"]
    note = _visible_text("<body>" + html[html.index('id="trade-stats-note"'):].split("</p>")[0])
    m = re.search(r"a mean of ([+−-][\d,.]+)k per trade in sleeve dollars \(\s*(−?\$[\d,]+) of portfolio P&L",
                  note)
    assert m, note
    tile = re.search(r"average (−?\$[\d,]+) of portfolio P&L per trade", _visible_text(html))
    assert tile
    assert m.group(2) == tile.group(1)
    mean_k = float(m.group(1).replace("−", "-").replace(",", ""))
    tile_usd = float(tile.group(1).replace("−", "-").replace("$", "").replace(",", ""))
    assert mean_k * 1000 / 3 == pytest.approx(tile_usd, abs=0.05 * 1000 / 3 + 1)


def test_round_trips_reconcile_with_the_sleeve_table_and_the_engine(dash, pages):
    """#round-trips, #trade-stats and #drawdowns: every sleeve's count equals the
    per-sleeve table's (the open S4 straddle counted apart), the round trips add
    up to the engine's sleeve P&L, the strip plot holds every closed trade, the
    'combined' records are not read, each series' deepest drawdown is the
    engine's max drawdown (the portfolio's is the published KPI) and every
    unrecovered drawdown prints 'not recovered'. Before B12 the page showed only
    aggregates."""
    tr_mod = _load("desk_trades_under_test", "dashboard/desk_trades.py")
    state = pickle.loads((SHIPPED / "backtest" / "full_results.pkl").read_bytes())
    em, trades = state["metrics"], state["all_trades"]
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    html = pages["/research"]
    for pid in ("round-trips", "trade-hists", "trade-stats", "drawdowns"):
        assert f'id="{pid}"' in html, pid
    sleeve_rows = {r[0]: r for r in _tbody_rows(html, "sleeve-table")}
    stats = _tbody_rows(html, "trade-stats-tbl")
    labels = {k: dash.SLEEVE_LABELS[k] for k in ("s1c", "s3", "s4", "s1", "s2")}
    table_label = labels
    assert [r[0] for r in stats] == list(labels.values())
    spec = _spec(html, "chart-round-trips")
    by_name = {s["name"]: s for s in spec["series"]}
    n_closed_port = 0
    for (sig, lab), row in zip(labels.items(), stats, strict=True):
        rts = tr_mod.round_trips(trades, (sig,))
        n_open = sum(r["open"] for r in rts)
        assert int(row[1]) + int(row[2]) == len(rts) == int(em[f"{sig}_n_trades"])
        assert int(row[2]) == n_open
        # the per-sleeve table's 'Trades' cell: '46 (1 open)'
        want = f"{len(rts)}" + (f" ({n_open} open)" if n_open else "")
        assert sleeve_rows[table_label[sig]][5] == want, (sig, sleeve_rows[table_label[sig]])
        assert sum(r["pnl"] for r in rts) == pytest.approx(em[f"{sig}_total_pnl"], abs=1.0)
        assert row[3] == dash.signed(em[f"{sig}_total_pnl"] / 1000, ",.1f")
        if sig in dash.PORTFOLIO_SLEEVES:
            pts = by_name[lab]["data"]
            assert len(pts) == len(rts) - n_open
            assert sum(y for _x, y in pts) == pytest.approx(
                sum(r["pnl"] for r in rts if not r["open"]) / 1000, abs=0.05 * len(pts))
            n_closed_port += len(rts) - n_open
        else:
            assert lab not in by_name                   # reference sleeves: the statistics only
    assert not any("combined" in s["name"].lower() for s in spec["series"])
    assert f"{n_closed_port} closed" in html
    hist = _spec(html, "chart-trade-pnl-hist")
    assert sum(int(r[1]) for r in hist["table"]["rows"]) == n_closed_port
    days = _spec(html, "chart-trade-days-hist")
    assert sum(int(r[1]) for r in days["table"]["rows"]) == n_closed_port
    # the strip's data alternative: closed trades by sleeve and entry year, adding up to the sleeve
    by_sleeve = collections.Counter()
    for r in spec["table"]["rows"]:
        by_sleeve[r[0]] += int(r[2])
    for sig in dash.PORTFOLIO_SLEEVES:
        rts = tr_mod.round_trips(trades, (sig,))
        assert by_sleeve[labels[sig]] == sum(1 for r in rts if not r["open"]), sig
    # drawdowns: the deepest of each series is the engine's; 'not recovered' where it is so
    dd = _tbody_rows(html, "drawdowns-tbl")
    assert dd[0][0] == "Portfolio"
    assert dd[0][5] == f"{em['max_drawdown'] * 100:.1f}%".replace("-", "−")
    # the published max drawdown (the KPI tile) is the table's top row
    mdd = f"{em['max_drawdown'] * 100:.1f}".replace("-", "−")
    assert re.search(r"Max drawdown\s*" + re.escape(mdd) + r"\s*%", _visible_text(html))
    for col, lab, key in (("nav", "Portfolio", "max_drawdown"),
                          ("nav_s1c", labels["s1c"], "s1c_max_drawdown"),
                          ("nav_s3", labels["s3"], "s3_max_drawdown"),
                          ("nav_s4", labels["s4"], "s4_max_drawdown")):
        rows = [r for r in dd if r[0] == lab]
        periods = tr_mod.worst_drawdowns(eq[col])
        assert periods[0]["depth"] == pytest.approx(em[key], abs=1e-12)
        assert len(rows) == len(periods) == min(5, len(tr_mod.drawdown_periods(eq[col])))
        for r, p in zip(rows, periods, strict=True):
            assert r[2:5] == [p["peak"], p["valley"], p["recovery"] or "not recovered"]
            assert r[5] == f"{p['depth'] * 100:.1f}%".replace("-", "−")


# ── B13: the vol-context strip in the ticker ──────────────────────────────────

def _ticker(c: dict, label: str) -> dict | None:
    return next((t for t in c["ticker"] if t["label"] == label), None)


def test_ticker_vol_context_is_computed_from_the_market_db(dash, synth):
    """B13: after VIX/VIX3M the ticker reads RV21 (RMS of the last 21 daily SPX
    log returns, annualised ×252, %), VIX − RV21 (vol pts) and VIX's percentile
    among the 252 sessions before its newest close (percent strictly lower),
    each recomputed here from the store's own rows; every ticker item's title
    names its as-of date and the percentile's states its 252-session window.
    A store without surface artifacts has no surface item (nothing in its place)."""
    c = dash.load_chrome()
    labels = [t["label"] for t in c["ticker"]]
    i = labels.index("VIX/VIX3M")
    assert labels[i + 1:i + 4] == ["RV21", "VIX−RV21", "VIX 1Y PCTL"]
    assert labels[i + 4] == "RULE"
    last = str(synth["dates"][-1].date())
    r, vix = synth["r"].iloc[1:], synth["vix"]
    rv = 100.0 * float(np.sqrt(np.mean(r.iloc[-21:].to_numpy() ** 2) * 252))
    assert _ticker(c, "RV21")["value"] == f"{rv:.2f}"
    assert _ticker(c, "VIX−RV21")["value"] == dash.signed(float(vix.iloc[-1]) - rv, ".2f")
    n_below = int((vix.iloc[-253:-1] < vix.iloc[-1]).sum())
    pc = _ticker(c, "VIX 1Y PCTL")
    assert pc["value"] == dash.percentile_rank(n_below / 252 * 100)["big"]
    assert f"{n_below} of the 252 sessions before it" in pc["title"]
    assert "252 sessions" in pc["sr"]
    for t in c["ticker"]:
        assert re.search(r"\d{4}-\d{2}-\d{2}", t.get("title") or ""), t["label"]
    for lab in ("RV21", "VIX−RV21", "VIX 1Y PCTL"):
        assert last in _ticker(c, lab)["title"], lab
        assert _ticker(c, lab)["note"] is None           # the SPX close's date: no note
    assert _ticker(c, "1M ATMF") is None


def test_ticker_surface_item_carries_the_surface_date_and_its_age(dash, pages):
    """B13: the SVI-cal 1M ATMF and RR25 in the ticker are the Overview's surface
    line (the same 30-day constant-maturity row), on the surface's own session,
    always shown as a note styled by its age against the newest session; the
    title names the session, its age and the expiries it interpolates between."""
    c = dash.load_chrome()
    item = _ticker(c, "1M ATMF")
    assert item is not None
    assert c["ticker"][-1] is item
    surf = dash.artifacts.load_surface(dash.DATA)
    session = surf["session"]
    age = dash.session_age(session, c["latest"])
    assert item["note"] == session
    assert item["note_level"] == dash.stale_level(age)
    assert session in item["title"]
    assert re.search(r"between the \d{4}-\d{2}-\d{2} and \d{4}-\d{2}-\d{2} expiries|"
                     r"the \d{4}-\d{2}-\d{2} expiry", item["title"])
    ov = _visible_text(pages["/"])
    line = re.search(r"1M ATMF ([\d.]+)% · RR25 ([+−]?[\d.]+) · BF25", ov)
    assert line, "no surface line on the Overview"
    assert item["value"] == line.group(1) + "%"
    assert item["more"] == [{"label": "RR25", "value": line.group(2)}]
    for route, html in pages.items():
        tk = html[html.index('<div class="ticker"'):html.index("<main")]
        assert (f'<span class="tk-label">1M ATMF</span><span class="tk-val">{item["value"]}</span>'
                f'<span class="tk-label">RR25</span><span class="tk-val">{line.group(2)}</span>'
                f'<span class="tk-note lvl-{item["note_level"]}">{session}</span>') in tk, route
        titled = re.findall(r'<span class="tk" title="[^"]*\d{4}-\d{2}-\d{2}[^"]*">', tk)
        assert len(titled) == len(c["ticker"]), route


def test_browser_ticker_is_one_scrolling_row_to_860_px_and_at_most_two_lines_above(site_browser):
    """B13 acceptance 3: with ten items the ticker would wrap to four lines at 601
    px and three at 768; it is one row up to 860 px that scrolls inside itself
    (from the keyboard too: it is a focusable region) and at most two lines from
    861 to 1920 px, and the page never scrolls sideways (the screen-reader text
    of the percentile item stays inside the row)."""
    sb = site_browser
    geo = """() => { const t = document.querySelector('.ticker');
      const tops = new Set([...t.querySelectorAll('.tk')]
        .map(e => Math.round(e.getBoundingClientRect().top)));
      return {lines: tops.size, sw: t.scrollWidth, cw: t.clientWidth, left: t.scrollLeft,
              dsw: document.documentElement.scrollWidth}; }"""
    ctx, page = _page(sb, 390, 844)
    try:
        for w in (320, 390, 601, 768, 860, 861, 1024, 1280, 1440, 1920):
            page.set_viewport_size({"width": w, "height": 844})
            page.goto(sb["base"] + "index.html", wait_until="load")
            g = page.evaluate(geo)
            assert g["dsw"] <= w, (w, g)
            if w <= 860:
                assert g["lines"] == 1, (w, g)
                assert g["sw"] > g["cw"], (w, g)
            else:
                assert g["lines"] <= 2, (w, g)
        page.set_viewport_size({"width": 768, "height": 844})
        page.goto(sb["base"] + "index.html", wait_until="load")
        page.focus(".ticker")
        for _ in range(4):
            page.keyboard.press("ArrowRight")
        page.wait_for_timeout(400)
        g = page.evaluate(geo)
        assert g["left"] > 0, g
        # the surface date takes the status bar's age colour (lvl-ok / warn / error), not the grey note
        colours = page.evaluate("""() => {
          const n = document.querySelector('.ticker .tk-note[class*="lvl-"]');
          if (!n) return null;
          const p = document.createElement('span');
          p.className = [...n.classList].find(c => c.startsWith('lvl-'));
          document.body.appendChild(p);
          const out = [getComputedStyle(n).color, getComputedStyle(p).color];
          p.remove(); return out; }""")
        assert colours, "no dated ticker note"
        assert colours[0] == colours[1], colours
    finally:
        ctx.close()
