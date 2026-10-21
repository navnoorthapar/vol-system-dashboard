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
    assert "Heston calibration is 10 NYSE sessions old" in html


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
            # the table toggle and the CSV button name this chart
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
                                                  "focus"]:
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
    assert n_flagged == len(re.findall(r'<div class="z">⚠ ', html)) == d["n_unstable"]
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
    for stale in ("structural failure", "Feller-satisfying Heston", "SOFR",
                  "VIX Futures RMSE", "SVI"):
        assert stale not in cal, stale
    # the pre-C22 copy's "VIX options RMSE was 37.14 (vol) pts"; the quote smiles'
    # chart data may hold an IV of 37.14 %
    assert not re.search(r"37\.14 (vol )?pts", cal)


def _spec(html: str, chart_id: str) -> dict:
    raw = re.search(rf'<script type="application/json" id="{re.escape(chart_id)}-spec">(.*?)</script>', html,
                    re.DOTALL)
    assert raw, chart_id
    return json.loads(_html_unescape(raw.group(1)))


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
    assert f"{oos} out-of-sample, it loses to the {pers}" in idx
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


@pytest.mark.parametrize(("p", "says"), [
    (0.0746, "does not beat random entry dates at the 5% level (p = 0.075)"),
    (0.0348, "beats random entry dates at the 5% level (p = 0.035)")])
def test_s3_timing_claim_follows_the_stored_null_test(dash, monkeypatch, p, says):
    """'Only S3's timing survives' / 'its timing beats a random-walk null' were
    hard-coded; the regenerated null test gives p = 0.075, not significant."""
    base = dash.load_page4()
    monkeypatch.setattr(dash, "load_page4", lambda: {**base, "kpi": {**base["kpi"], "s3_null_p": p}})
    text = _visible_text(dash.app.test_client().get("/research").get_data(as_text=True))
    assert says in text
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
    for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C Contrarian"), ("nav_s3", "S3 Dispersion"),
                      ("nav_s4", "S4 VRP"), ("nav_s1", "S1 PDV (reference)")):
        rel = eq[col] / eq[col].iloc[0] - 1
        want = "".join(f'<td class="num">{dash._pct_prose(x)}</td>'
                       for x in (rel.iloc[-1], rel.min(), rel.max()))
        assert f"<tr><td>{name}</td>{want}</tr>" in table, name
    # the C22 case (+5.647% printed as +5.7%) is now +6.105% on the C23 record
    assert '<tr><td>S3 Dispersion</td><td class="num">+6.1%</td>' in table
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
    vv = re.findall(r'<div class="vv">([^<]+)</div>', g_html)
    assert vv == [f"{x:,.0f}".replace("-", m) for x in gdf["vomma"]]
    zz = [re.sub(r"^⚠ ", "", z).strip() for z in re.findall(r'<div class="z">([^<]+)</div>', g_html)]
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
    vv = re.findall(r'<div class="vv">([^<]+)</div>', html)
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
    text = _visible_text(pages["/research"]) + " | " + _visible_text(pages["/corrections"])
    p = f"{m['s3_null_p_value']:.3f}"
    assert ("its timing, tested on S3 at base (pass-1) size, before the sizing overlay, does not beat "
            f"random entry dates at the 5% level (p = {p})") in text
    assert "S3 at base (pass-1) size, before the sizing overlay, vs random entry dates: not significant at 5%" in text
    assert f"timing null test on S3 at base (pass-1) size, before the sizing overlay: p = {p}" in text
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
    for route in ("/research", "/corrections"):
        text = _visible_text(client.get(route).get_data(as_text=True))
        assert "on S3 at base (pass-1) size (+$41,952), before the sizing overlay" in text, route


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
    res = pages["/research"]
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
    table = re.search(r'<table class="tbl dense" id="scorecard-table">(.*?)</table>', html,
                      re.DOTALL).group(1)
    rank = [r["model"] for r in sc["ranking_by_oos_median_abs_hs"]]
    rows = re.findall(r"<tr[^>]*><td>([^<]+)</td>", table.split("<tbody>")[1])
    assert [dash.model_label(m) for m in rank] == rows            # the producer's ranking order
    for m in sc["models"]:
        o = m["metrics"]["out_of_sample"]
        row = re.search(rf"<td>{re.escape(dash.model_label(m['model']))}</td>(.*?)</tr>", table,
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
    first = re.search(r"<tr><th scope=\"row\">[^<]*</th>(.*?)</tr>", heat.split("<tbody>")[1],
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
    atab = re.search(r'<table class="tbl dense" id="arb-table">(.*?)</table>', html, re.DOTALL).group(1)
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
    assert "Quote-grade surface · held-out strikes" in t
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
    Overview's loader no longer even computes it."""
    idx = _visible_text(pages["/"])
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
        assert "no entry yet: the first is written with the next accepted fit" in text


def test_rough_vs_heston_panel_lists_every_diagnosed_session(dash, art):
    resp, html = _get(dash, "/calibration")
    assert _errors(resp) == 0
    rows = json.loads((FIX / "diagnostics" / "summary.json").read_text())["rows"]
    tab = re.search(r'<table class="tbl dense" id="rough-heston-table">(.*?)</table>', html,
                    re.DOTALL).group(1)
    body = tab.split("<tbody>")[1]
    assert len(re.findall(r"<tr[ >]", body)) == len(rows)
    for r in rows:
        line = _visible_text("<body>" + re.search(rf"<td>{r['session']}(.*?)</tr>", body, re.DOTALL).group(1))
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
    inv = {m["id"]: m for m in models}
    if (prov.get("carry") or {}).get("by_maturity"):
        assert "flat carry" not in inv["M2"]["limitations"].lower()
        assert "per maturity" in inv["M2"]["limitations"]


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
        assert f"r(T) {r_lo:.2f}–{r_hi:.2f}%" in text
        assert f"q(T) {q_lo:.2f}–{q_hi:.2f}% from the fit's parity forwards" in text


def test_corrections_page_shows_the_c23_waterfall_from_its_file(pages):
    bridge = pd.read_csv(SHIPPED / "backtest" / "c23_bridge.csv")
    tab = re.search(r'<table class="tbl dense" id="c23-bridge">(.*?)</table>', pages["/corrections"],
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
    budget = {"index.html": 120_000, "surface.html": 600_000, "calibration.html": 300_000,
              "risk.html": 300_000, "research.html": 400_000, "corrections.html": 120_000,
              "model-risk.html": 80_000}
    for name, cap in budget.items():
        size = (out / name).stat().st_size
        assert size <= cap, (name, size)
    # shared assets: the chart library is the only large script, fonts are subsets
    import gzip
    static = ROOT / "dashboard" / "static"
    assert len(gzip.compress((static / "vendor" / "echarts-6.1.0-desk.min.js").read_bytes())) < 240_000
    assert (static / "desk.js").stat().st_size < 60_000
    assert (static / "desk.css").stat().st_size < 60_000
    assert sum(p.stat().st_size for p in (static / "fonts").glob("*.woff2")) < 80_000


def test_research_inference_panels_show_the_summary(dash, art):
    resp, html = _get(dash, "/research")
    assert _errors(resp) == 0
    s = json.loads((FIX / "inference" / "summary.json").read_text())
    b = s["bootstrap"]["portfolio"]["sharpe"]
    boot = re.search(r'<table class="tbl dense" id="bootstrap-table">(.*?)</table>', html, re.DOTALL).group(1)
    port = _visible_text("<body>" + re.search(r"<td>Portfolio</td>(.*?)</tr>", boot, re.DOTALL).group(1))
    def fmt(x):
        return f"{x:+.2f}".replace("-", "−")
    assert f"{fmt(b['ci95'][0])} to {fmt(b['ci95'][1])}" in port
    nul = re.search(r'<table class="tbl dense" id="null-table">(.*?)</table>', html, re.DOTALL).group(1)
    s3 = s["nulls"]["sleeves"]["s3"]
    row = _visible_text("<body>" + re.search(r"<td>S3 dispersion</td>(.*?)</tr>", nul, re.DOTALL).group(1))
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
    pdv = _visible_text("<body>" + re.search(r"<td>PDV</td>(.*?)</tr>", ftab, re.DOTALL).group(1))
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
    assert len(re.findall(r"<tr>", prof.split("<tbody>")[1])) == len(diag["sigma_profile"]["rows"])
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
    arb_row = mon["Quotes in an executable static-arbitrage row (% of scored quotes)"]
    assert arb_row["value"].startswith(f"{share:.1f}%")


def test_model_risk_without_artifacts_marks_their_checks_not_applicable(dash, no_art):
    d = dash.load_model_risk()
    na = [m for m in d["monitors"] if m["value"] == "artifact not yet generated"]
    assert len(na) == 3
    assert all(m["status"] == "na" for m in na)


# ── Corrections registry and model inventory ─────────────────────────────────

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
        assert (ROOT / m["outputs"]).exists(), (m["id"], m["outputs"])
        assert m["tier"] in ("production", "backtest", "research"), m["id"]
        assert m.get("page") is None or m["page"] in dash.ROUTES, m["id"]


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
    assert rows["30d"][0].startswith("−")


def test_one_colour_per_model_on_every_surface_chart(dash, pages):
    """UI-2: the slice drew Heston in the orange the by-expiry chart beside it used
    for SVI, and had no legend. Each model now has one colour on every chart, and
    the slice has one toggle per model (all four), the market quotes and the
    Heston anchors, with the forward marked."""
    html = pages["/surface"]
    sl, be = _spec(html, "chart-slices"), _spec(html, "chart-by-expiry")
    by_name = {s["name"]: s["color"] for s in be["series"]}
    models = json.loads((SHIPPED / "surface" / "2026-10-16" / "scorecard.json").read_text())["models"]
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
    heston = next(s for s in smile["series"] if s["name"] == "Heston")
    assert heston["color"] == dash.model_colour("heston_v2")


def test_smile_slice_model_points_share_the_quotes_x_values(dash, pages):
    """UI-2/UI-17: all four models on the slice within the page budget: a model's
    points list only IVs, aligned with that expiry's mids (desk.js takes x from
    them), and the IVs are the residual table's model IVs at those strikes."""
    sl = _spec(pages["/surface"], "chart-slices")
    by = {s["name"]: s for s in sl["series"]}
    res = pd.read_parquet(SHIPPED / "surface" / "2026-10-16" / "residuals.parquet")
    with_x = [s for s in sl["series"] if s.get("xFrom")]
    assert len(with_x) == res["model"].nunique() * res["expiry"].nunique()      # every model, every expiry
    for s in with_x:
        mids = by[s["xFrom"]]["data"]
        assert len(mids) == len(s["data"]), s["name"]
        e, label = s["group"], s["legendKey"]
        key = next(k for k in res["model"].unique() if dash.model_label(k) == label)
        q = res[(res["model"] == key) & (res["expiry"] == e)].sort_values("k")
        assert [round(float(k), 3) for k in q["k"]] == [m[0] for m in mids], s["name"]
        assert s["data"] == [round(float(v) * 100, 2) for v in q["model_iv"]], s["name"]


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
    res = pd.read_parquet(SHIPPED / "surface" / "2026-10-16" / "residuals.parquet")
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
    sc = json.loads((SHIPPED / "surface" / "2026-10-16" / "scorecard.json").read_text())
    ns = {m["metrics"]["out_of_sample"]["n"] for m in sc["models"]}
    title = re.search(r'<h2 class="panel-title" id="scorecard-t">(.*?)</h2>', html).group(1)
    assert ("same quotes" in title) is (len(ns) == 1)
    be = _spec(html, "chart-by-expiry")
    res = pd.read_parquet(SHIPPED / "surface" / "2026-10-16" / "residuals.parquet")
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
    vb = json.loads((SHIPPED / "surface" / "2026-10-16" / "vix_replication.json").read_text())["beta"]
    assert f"VIX-term ATM / VIX on this chain: {vb['atm_over_vix_official']:.3f}" in surf
    assert "30-day ATM vol / VIX on this chain" not in surf
    row = re.search(r'<td class="nowrap">2026-10-16</td><td>[^<]*</td><td class="num">([0-9.]+)</td>', res)
    assert row, "research β chain row"
    assert f"gives {row.group(1)} on the same chain" in surf
    assert "β estimator ATM 30d / VIX" in surf
    assert "VIX-term ATM / VIX" in surf


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


def test_surface_badge_lists_the_models(pages):
    """UI-10: the badge read 'models:' and nothing else."""
    badge = _visible_text("<body>" + re.search(r'<div class="page-badge">(.*?)</div>', pages["/surface"],
                                                re.DOTALL).group(1))
    for m in ("eSSVI", "SVI", "SVI-cal", "Heston"):
        assert re.search(rf"models:.*\b{re.escape(m)}\b", badge), (m, badge)


def test_calibration_smile_anchors_stand_apart_from_the_out_of_sample_mids(dash, pages):
    """UI-12: the 9 fit anchors were filled dots the colour of 338 hollow mids,
    and both legend swatches were the same white dot."""
    cal = pages["/calibration"]
    n = 0
    for cid in re.findall(r'id="(chart-smile-[^"]+)" data-spec', cal):
        spec = _spec(cal, cid)
        ser = {s["name"]: s for s in spec["series"]}
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
    assert "$$('details.method:not([open])').forEach(function (d) { d.open = true; opened.push(d); });" in js
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


def _page(sb, width=1440, height=900):
    ctx = sb["browser"].new_context(viewport={"width": width, "height": height})
    return ctx, ctx.new_page()


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
    links |= {"surface.html#scorecard&exp=2026-12-18", "risk.html#scenario&greek=vega"}
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
    page.goto(sb["base"] + "surface.html#exp=2026-12-18", wait_until="load")
    page.click("[data-copy-link='scorecard']")
    assert page.url.endswith("#scorecard&exp=2026-12-18")
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
    page.wait_for_timeout(300)
    page.evaluate("document.fonts.ready.then(() => true)")
    page.wait_for_timeout(200)
    n = page.evaluate("window.__so['chart-slices']")
    early = page.evaluate("Desk.charts['chart-slices'].early")
    assert n == (2 if early else 1), (n, early)
    ctx.close()
    ctx = sb["browser"].new_context(viewport={"width": 1440, "height": 900})
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
    page.click("[data-chart-filter] [data-value='2026-12-18']")
    shown = page.evaluate("[...document.querySelectorAll('#chart-slices-data [data-group]')]"
                          ".filter(g => !g.hidden).map(g => g.dataset.group)")
    assert shown == ["2026-12-18"]
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
