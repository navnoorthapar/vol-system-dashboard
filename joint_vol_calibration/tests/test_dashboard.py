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
    assert dash.stale_level(1) == "ok" and dash.stale_level(2) == "warn" and dash.stale_level(5) == "error"


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
    with pytest.raises(sqlite3.OperationalError):
        with dash._db() as con:
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
    assert seen and all("params" in r and "spot_date" in r for r in seen)


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
    assert d["session"] == old and d["calib_age"] == 10
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
    assert d["session"] == old and d["legacy"] is False
    assert d["archive"] == {"n_total": 2, "n_legacy": 1, "n_current": 1, "n_current_ok": 1}
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "PRE-C22 FIT" not in html
    assert "1 pre-C22 fits are labelled legacy and never ranked" in html


def test_carry_is_shown_only_when_the_fit_recorded_it(dash, synth):
    """A lone stored r is a post-fit T-bill close, and q=0 was a display default."""
    cal_dir = synth["data"] / "calibrations"
    last = str(synth["dates"][-1].date())
    _write_cals(cal_dir, {f"joint_cal_{last}.pkl": _cal(last, 1.0, r=0.03913)})
    d = dash.load_page2()
    assert d["r_pct"] is None and d["q_pct"] is None and d["carry_recorded"] is False
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "SOFR" not in html and "r = n/a" in html

    _write_cals(cal_dir, {f"joint_cal_{last}.pkl": _cal(last, 1.0, r=0.045, q=0.013)})
    d = dash.load_page2()
    assert (d["r_pct"], d["q_pct"]) == (4.5, 1.3)


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
    assert d["n_spx_expiries"] == 1 and d["spx_underidentified"]
    assert d["n_spx_expiries_inferred"] is (not recorded)
    html = dash.app.test_client().get("/calibration").get_data(as_text=True)
    assert "Good fit" not in html and "Not comparable" in html


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
        # independent reference: finer integration grid + direct BS inversion
        calls = heston_call_batch(1.0, K, T, 0.0, 0.0, p["kappa"], p["theta"], p["sigma"],
                                  p["rho"], p["v0"], n_points=20000, v_upper=2000.0)
        for ki, Ki, c, iv_pct in zip(k, K, calls, s["iv"]):
            def bs(vol):
                d1 = (-np.log(Ki) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
                return norm.cdf(d1) - Ki * norm.cdf(d1 - vol * np.sqrt(T)) - c
            ref = brentq(bs, 1e-4, 5.0, xtol=1e-12)
            assert abs(iv_pct - ref * 100) < 2e-3, (s["T"], ki, iv_pct, ref * 100)


def test_quintic_is_compared_with_the_same_snapshot_heston(dash):
    q = dash.load_page2()["quintic"]
    assert q is not None and q["heston"] is not None
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
    assert d["pdv_stale"] is True and d["pdv_age"] == len(dates) - len(synth["dates"])
    html = dash.app.test_client().get("/").get_data(as_text=True)
    assert 'id="pdv-stale-flag"' in html and "STALE FORECAST" in html
    assert "Within this spread's usual range" not in html and "rich versus" not in html


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

    d = dash.load_page1()
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
    d = dash.load_page1()
    assert d["clf_available"] is False and d["prob_r1"] is None
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
    assert r["label"] == int(rl["regime"].loc[r["date"]])     # stored label on that session
    html = dash.app.test_client().get("/").get_data(as_text=True)
    assert f"Rule inputs on {r['date']}" in html


def test_corrupt_classifier_keeps_the_rule_label_and_reports_an_error(dash, synth):
    (synth["data"] / "signals" / "regime_classifier.pkl").write_bytes(b"not a pickle")
    resp = dash.app.test_client().get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert resp.headers["X-Dashboard-Loader-Errors"] == "1"
    lab = int(synth["labels"]["regime"].iloc[-1])
    assert f"{['R0', 'R1', 'R2'][lab]} · {['LONG GAMMA', 'SHORT GAMMA', 'VOMMA ACTIVE'][lab]}" in html
    assert "research classifier" in html and "Data unavailable" in html


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

@pytest.mark.parametrize("victim, route, component", [
    ("greeks/greeks_surface.parquet", "/greeks", "Greeks surface"),
    ("backtest/full_results.pkl", "/backtest", "backtest engine results"),
    ("pdv_model.pkl", "/", "PDV forecast"),
])
def test_missing_input_renders_an_error_not_zeros(dash, synth, route, victim, component):
    (synth["data"] / victim).unlink()
    resp = dash.app.test_client().get(route)
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert int(resp.headers["X-Dashboard-Loader-Errors"]) >= 1
    assert component in resp.headers["X-Dashboard-Error-Components"]
    assert "Data unavailable" in html
    assert "0 / 0" not in html and "NAV: $0" not in html and "0.00%" not in html


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
    assert len(set(locs)) == len(dash.ROUTES)
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
    assert f"greeks.html</loc><lastmod>{g['surface_date']}</lastmod>" in sitemap
    # the backtest page's data date is the last session of its equity curve
    # (it read a "run_date" key full_results.pkl never had, so it had no lastmod)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    last = str(pd.Timestamp(eq.index[-1]).date())
    assert f"backtest.html</loc><lastmod>{last}</lastmod>" in sitemap
    assert sitemap.count("<lastmod>") == len(dash.ROUTES)
    assert dash.app.test_client().get("/backtest").headers["X-Dashboard-Data-Date"] == last


def test_no_third_party_script_and_the_vendored_chartjs_matches_its_sri(dash, freezer):
    """Chart.js is served from the site itself; the CDN copy is only a fallback
    loaded from JS with the same SRI hash, which must equal the vendored file's."""
    html = dash.app.test_client().get("/").get_data(as_text=True)
    assert not re.findall(r"<script[^>]+src=\"(?:https?:)?//", html)
    assert f'<script src="{dash.CHARTJS["src"]}">' in html
    vendored = ROOT / "dashboard" / dash.CHARTJS["src"]
    assert vendored.exists()
    assert freezer.sri_sha384(str(vendored)) == dash.CHARTJS["sri"]
    assert f'"{dash.CHARTJS["cdn"]}"' in html and f'"{dash.CHARTJS["sri"]}"' in html
    assert "s.crossOrigin = 'anonymous'" in html
    assert "@4.4.3/" in dash.CHARTJS["cdn"] and dash.CHARTJS["version"] in dash.CHARTJS["src"]


def test_freeze_copies_static_and_refuses_a_chartjs_that_fails_its_sri(dash, freezer, tmp_path, monkeypatch):
    out = tmp_path / "site"
    assert freezer.freeze(str(out), dash) == 0
    assert (out / dash.CHARTJS["src"]).read_bytes() == (ROOT / "dashboard" / dash.CHARTJS["src"]).read_bytes()
    monkeypatch.setitem(dash.CHARTJS, "sri", "sha384-" + "A" * 64)
    out2 = tmp_path / "site2"
    assert freezer.freeze(str(out2), dash) == 1
    assert not out2.exists() or not any(out2.iterdir())


# ── Greeks page ───────────────────────────────────────────────────────────────

def test_heatmap_draws_every_node(dash):
    """0.05-wide bins on a 0.039-spaced grid hid 16 of 69 nodes (one unstable)."""
    d = dash.load_page3()
    gdf = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    cells = [c for row in d["heatmap_rows"] for c in row["cells"] if not c.get("empty")]
    assert len(cells) == d["n_cells"] == len(gdf)
    # the surface's own flag (vomma P&L ratio rule), not a re-derived pooled z-score
    assert sum(c["unstable"] for c in cells) == d["n_unstable"] == int(gdf["is_unstable"].sum())
    html = dash.app.test_client().get("/greeks").get_data(as_text=True)
    assert len(re.findall(r'class="node unstable"', html)) == d["n_unstable"]
    assert "ln(K/S)" not in html and "2026-03-24" not in html and "All three unstable" not in html


def test_greeks_badge_and_staleness_come_from_the_surface(dash):
    d = dash.load_page3()
    gdf = pd.read_parquet(SHIPPED / "greeks" / "greeks_surface.parquet")
    assert d["surface_date"] == str(gdf["as_of_date"].iloc[0])[:10]
    assert d["greeks_age"] == dash.session_age(d["surface_date"], dash._latest_market_session())
    assert d["qv_values"] and len(d["qv_values"]) == d["n_maturities"]


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
    assert d["surface_date"] == last and d["greeks_age"] == 0
    html = dash.app.test_client().get("/greeks").get_data(as_text=True)
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
    assert "N/A*" not in html and "ratio diverges" not in html


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
    assert "OOS Windows" not in html and "Walk-Forward Sharpe" not in html

    # no clipping: a steady riser has a Sharpe far above the old ±4 cap
    idx = pd.bdate_range("2021-01-04", "2021-12-31")
    steady = pd.Series(1e6 * np.exp(np.cumsum(np.random.default_rng(2).normal(4e-4, 1e-4, len(idx)))), index=idx)
    hy = dash.half_year_sharpes(steady)
    assert len(hy) == 2 and min(w["sharpe"] for w in hy) > 20


def test_annual_returns_compound_to_the_total(dash):
    d = dash.load_page4()
    total = np.prod([1 + a["ret_pct"] / 100 for a in d["ann"]]) - 1
    assert total == pytest.approx(d["kpi"]["cum_ret"], abs=1e-10)
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    assert d["eq_dates"][-1] == str(pd.Timestamp(eq.index[-1]).date())


def test_backtest_prose_is_rendered_from_the_results(dash, engine_state):
    html = dash.app.test_client().get("/backtest").get_data(as_text=True)
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
        assert html.count("<main") == 1 and 'id="main"' in html
        assert re.search(r'<nav class="nav" aria-label="Primary">', html)
        assert html.count("<header") == 1
        assert '<a class="skip-link" href="#main">' in html
        # the nav is a direct child of <body> (sticky works for the whole page),
        # and nothing overrides its position inline
        assert '<nav class="nav" style=' not in html and "position:relative;\">" not in html


def test_nav_marks_exactly_the_current_page(dash, pages):
    for route, html in pages.items():
        current = re.findall(r'<a href="([^"]+)" class="nav-link active" aria-current="page">', html)
        assert current == [route], (route, current)
        assert len(re.findall(r'<a [^>]*aria-current="page"', html)) == 1


def test_every_chart_has_an_accessible_name_and_a_data_alternative(pages):
    """Canvases had role=img and a generic label (the PDV sparkline was called an
    'SPX price sparkline'); none had a text alternative carrying the numbers."""
    n = 0
    for route, html in pages.items():
        for tag in re.findall(r"<canvas[^>]*>", html):
            a = _attrs(tag)
            assert a.get("role") == "img", (route, tag)
            assert len(a.get("aria-label", "")) > 20, (route, tag)
            ref = a.get("aria-describedby")
            assert ref, (route, tag)
            m = re.search(rf'<(\w+) class="chart-data sr-only" id="{re.escape(ref)}">(.*?)</\1>', html, re.S)
            assert m, (route, ref)
            assert re.search(r"\d", m.group(2)), (route, ref)        # it carries numbers
            n += 1
        assert "SPX price sparkline" not in html
    assert n == 10


def test_scroll_regions_are_keyboard_focusable(pages):
    """Tables and the heatmap scroll horizontally on phones; axe flagged the
    scroll regions as unreachable by keyboard."""
    for route, html in pages.items():
        for m in re.finditer(r'<table class="(data-table|heatmap)"', html):
            before = html[:m.start()]
            opener = before[before.rfind("<div"):]
            assert re.match(r'<div class="(table-scroll|heatmap-wrap)[^"]*" tabindex="0" role="region" aria-(label|labelledby)="[^"]+">\s*$', opener), (route, opener[:120])


def test_colour_tokens_meet_wcag_aa(dash, pages):
    """--text3 was #555 (2.5:1 on the cards) and chart ticks/axis titles #555/#444."""
    html = pages["/"]
    root = re.search(r":root \{(.*?)\}", html, re.S).group(1)
    tok = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", root))
    backgrounds = [tok["bg"], tok["surface"], tok["surface2"], tok["tooltip"],
                   tok["tint-red"], tok["tint-amber"], tok["tint-green"], tok["tint-blue"]]
    for name in ("text", "text2", "text3", "accent", "accent2", "red", "blue", "yellow",
                 "amber", "orange", "purple", "cyan", "pink"):
        for bg in backgrounds:
            assert dash.contrast_ratio(tok[name], bg) >= 4.5, (name, tok[name], bg)
    vs_block = html[html.index("var VS = {"):html.index("};", html.index("var VS = {"))]
    vs = dict(re.findall(r"(tick|title):\s*'(#[0-9a-fA-F]{6})'", vs_block))
    assert set(vs) == {"tick", "title"}
    for c in vs.values():
        assert dash.contrast_ratio(c, tok["surface"]) >= 4.5
    for literal in ("#555", "#444"):
        for route, h in pages.items():
            assert f"color:{literal}" not in h and f"color: '{literal}'" not in h, (route, literal)


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
    html = dash.app.test_client().get("/greeks").get_data(as_text=True)
    assert "opacity:0.7" not in html
    # unstable is flagged by outline + text marker, not by a red fill alone
    assert len(re.findall(r'class="node unstable"', html)) == len(re.findall(r'<div class="z">⚠ ', html)) == d["n_unstable"]
    assert '<span class="sym">(∂²V/∂S∂σ)</span>' in html       # σ is not upper-cased to Σ


def test_motion_print_and_font_fallbacks(pages):
    html = pages["/"]
    assert "@media (prefers-reduced-motion: reduce)" in html
    assert "Chart.defaults.animation = false" in html
    assert "@media print" in html and "beforeprint" in html and "afterprint" in html
    assert '<meta name="color-scheme" content="dark">' in html
    for route, h in pages.items():
        assert "family: 'Menlo'" not in h and "font-family:'Menlo',monospace" not in h, route
        assert "monospace" in re.search(r"--mono:([^;]+);", h).group(1)


def test_page_charts_wait_for_chartjs(pages):
    """A failed Chart.js load threw 'Chart is not defined' and blanked every chart."""
    for route, html in pages.items():
        page_js = html[html.rindex("function makeBarOpts"):]
        first_new = page_js.find("new Chart(")
        assert first_new > page_js.find("vsCharts(function () {") > 0, route
    assert "if (!window.Chart) return fail();" in pages["/"]


def test_social_meta_and_card_carry_no_test_count(pages):
    """The card said '631 tests', the meta '666', CI '642 passed, 24 skipped'."""
    og = _load("og_card_under_test", "dashboard/make_og_image.py")
    assert not any(re.search(r"\btests?\b", label) for _, label in og.STATS)
    for route, html in pages.items():
        assert not re.search(r"\d+\s+tests", html), route
        title = re.search(r'<meta property="og:title"\s+content="([^"]+)"', html).group(1)
        desc = re.search(r'<meta property="og:description"\s+content="([^"]+)"', html).group(1)
        assert title and desc
    titles = {re.search(r'og:title"\s+content="([^"]+)"', h).group(1) for h in pages.values()}
    assert len(titles) == len(pages)                      # per-page, not one for all


def test_regime_and_calibration_copy_state_the_actual_rules(pages):
    idx, cal, bt = pages["/"], pages["/calibration"], pages["/backtest"]
    assert "rolling 252-day 80th percentile" in idx
    assert "rv₂₀ &gt; VIX" in idx
    for stale in ("Low VIX / backwardation", "Normal contango", "VVIX &gt; 100"):
        assert stale not in idx and stale not in bt
    for stale in ("non-affine model (e.g. rough Heston)", "degenerate on daily data",
                  "Barndorff-Nielsen &amp; Shephard 2004", "production classification"):
        assert stale not in cal
    # The ρ-boundary banner (rough-Heston / Bates copy) renders only when the
    # headline fit has ρ on its bound; the shipped headline does not.
    if "AT THE LOWER BOUNDARY" in cal:
        assert "affine Volterra" in cal and "did not converge" in cal
    for stale in ("37.14", "structural failure", "Feller-satisfying Heston", "SOFR",
                  "VIX Futures RMSE", "SVI"):
        assert stale not in cal, stale


def test_vix_colour_and_words_come_from_one_threshold_table(dash, monkeypatch):
    """VIX 19 was painted yellow beside 'NORMAL range'; VIX 27 red beside 'caution'."""
    base = dash.load_page1()
    client = dash.app.test_client()
    for vix, cls, words in ((19.0, "green", "NORMAL"), (27.0, "yellow", "ELEVATED"),
                            (31.0, "red", "STRESS"), (12.0, "dim", "LOW")):
        monkeypatch.setattr(dash, "load_page1", lambda v=vix: {**base, "vix": v})
        html = client.get("/").get_data(as_text=True)
        block = html[html.index("VIX  <span"):html.index("<!-- VVIX -->")]
        assert f'<span class="chip chip-{cls}">{words}' in block, (vix, block)


def test_term_structure_is_drawn_on_a_linear_tenor_axis(pages):
    js = pages["/"][pages["/"].index("// VIX Term Structure chart"):]
    js = js[:js.index("// Sparkline")]
    assert "type: 'linear'" in js and "tension: 0," in js and "tension: 0.35" not in js


def test_backtest_audit_heading_counts_its_rows(pages):
    """'Ten Flaws Found, Eight Fixed, Two Quantified' sat above an 11-row table."""
    bt = pages["/backtest"]
    statuses = re.findall(r'<tr data-status="(fixed|quantified)" data-round="(C\d+)"', bt)
    rounds = sorted({r for _, r in statuses}, key=lambda r: int(r[1:]))
    n_fixed = sum(1 for s, _ in statuses if s == "fixed")
    n_quant = sum(1 for s, _ in statuses if s == "quantified")
    heading = re.search(r'id="audit-title">([^<]+)<', bt).group(1)
    assert heading == (f"{rounds[0]}–{rounds[-1]} Methodology Audit — {len(statuses)} Flaws Found, "
                       f"{n_fixed} Fixed, {n_quant} Quantified")
    assert {"C19", "C20", "C21"} <= set(rounds)
    assert "Ten Flaws Found" not in bt and "C17 pipeline" not in bt


def test_backtest_window_crisis_and_data_claims_come_from_the_data(dash, pages):
    bt = pages["/backtest"]
    d = dash.load_page4()
    assert re.search(rf"<h1 class=\"page-title\">C10 Backtest Results — (?:<span class=\"nowrap\">)?"
                     rf"{d['start_date']} → {d['end_date']}", bt)
    assert "2018 → 2025<" not in bt
    tariffs = next(c for c in d["crisis"] if c["name"] == "Tariffs_2025")
    assert tariffs["end"] <= d["end_date"]
    assert f"Tariffs 2025 ({tariffs['start']} → {tariffs['end']})" in bt
    assert "April 2025 tariff shock is <em>not</em> in it" in bt
    assert "single options snapshot" not in bt and "22 times in 7 years" not in bt
    with sqlite3.connect(f"file:{SHIPPED / 'vol_system.db'}?mode=ro", uri=True) as con:
        n_dates = con.execute("SELECT COUNT(DISTINCT snapshot_date) FROM options_snapshots "
                              "WHERE underlying='SPX'").fetchone()[0]
    assert f"its {n_dates} SPX chain" in bt
    assert "$-" not in bt and "'$' + v + 'k'" not in bt


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
    html = dash.app.test_client().get("/greeks").get_data(as_text=True)
    assert f"joint_cal_{last}.pkl" in html and "records no source calibration" not in html


def test_classifier_accuracy_is_read_from_the_shipped_model(pages):
    """The front page hard-coded '63.4%' vs '90.0%' (pre-C22 fixed-gate labels with a
    look-ahead PDV feature) beside a classifier retrained in C22 that records its own
    evaluation. The figures must come from the pickle's metadata."""
    import pickle

    with open(ROOT / "data_store" / "signals" / "regime_classifier.pkl", "rb") as f:
        ev = pickle.load(f).metadata_["evaluation"]
    idx = pages["/"]
    oos = f"{round(ev['oos_acc'] * 100, 1)}%"
    pers = f"{round(ev['persistence_acc'] * 100, 1)}%"
    assert f"{oos} out-of-sample, it loses to the {pers}" in idx
    assert re.search(r"Out-of-sample accuracy: <span[^>]*>" + re.escape(oos), idx)
    assert re.search(r"Persistence baseline: <span[^>]*>" + re.escape(pers), idx)


def test_backtest_audit_table_has_the_c22_round(pages):
    bt = pages["/backtest"]
    assert 'data-round="C22"' in bt
    assert "from the C21 run" not in bt


# ── C22 verification round: presentation guarantees ──────────────────────────

def _visible_text(html: str) -> str:
    import html as _html

    body = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", html, flags=re.S)
    body = body[body.index("<body"):]
    return _html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)))


def test_numbers_use_true_minus_signs_and_no_placeholders(pages):
    """'-0.571', '$-923' and 'z=-1.0' printed hyphens; a missing value must read '—',
    never 'nan', 'None' or 'undefined'."""
    for route, html in pages.items():
        text = _visible_text(html)
        hyphen_numbers = re.findall(r"(?<![\w/.\-–])-\$?\d[\d,.]*", text)
        assert not hyphen_numbers, (route, hyphen_numbers[:5])
        for bad in (r"\bNaN\b", r"\bnan%", r"\bundefined\b", r"\bInfinity\b",
                    r"[:=]\s*(None|null)\b", r"(^|\s)None(\s|$)"):
            assert not re.search(bad, text), (route, bad)


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
    assert "Walk-fwd R² (kernel)" not in html and ">0.31<" not in html


def test_stress_card_describes_the_models_own_target(dash, pages):
    """The 21-day model's outcome is the RMS vol over its horizon, not 'that day's |r|'."""
    st = dash.load_page1()["stress"]
    text = _visible_text(pages["/"])
    assert st["horizon"] == 21 and st["n_window"] == 21
    assert f"Realised over its 21-session horizon ({st['stress_date']} → {st['window_end']}" in text
    assert "that day's |r|" not in text


def test_audit_evidence_is_read_from_the_c22_bridge(dash, pages):
    """'S1 +$722K → −$508K from β alone' quoted the published C21 figure as the
    pre-β value; the β step alone moved S1 from the previous bridge row."""
    bridge = pd.read_csv(SHIPPED / "backtest" / "c22_bridge.csv").set_index("step")
    grid = pd.read_csv(SHIPPED / "backtest" / "beta_sensitivity.csv").set_index("beta")
    rows = list(bridge.index)
    prev = bridge.loc[rows[rows.index("beta") - 1]]
    text = _visible_text(pages["/backtest"])
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
    text = _visible_text(pages["/backtest"])
    assert f"2018: {m2018 * 100:.2f}%" in text
    sh = d["r2_share"]
    assert f"R2 on {sh['fixed'] * 100:.1f}% of backtest sessions" in text
    assert f"R2 on {sh['adaptive'] * 100:.1f}% of backtest sessions" in text
    assert 0.40 < sh["fixed"] < 0.55 and 0.18 < sh["adaptive"] < 0.28
    for stale in ("~1.5%", "~53%", "~25%", "18 new tests"):
        assert stale not in text


def test_no_js_fallback_and_scroll_affordance(pages):
    html = pages["/backtest"]
    nos = re.search(r"<noscript><style>(.*?)</style></noscript>", html, re.S).group(1)
    assert ".chart-wrap" in nos and ".chart-data.sr-only" in nos and "position: static" in nos
    css = html[html.index(".table-scroll, .heatmap-wrap {"):]
    css = css[:css.index("}")]
    assert "local" in css and "scroll" in css and "position: relative" in css
    assert "Scroll sideways for more columns" in html


def test_unverified_or_contradicted_copy_is_gone(pages):
    cal, bt = _visible_text(pages["/calibration"]), _visible_text(pages["/backtest"])
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
    with open(synth["data"] / "pdv_model.pkl", "rb") as f:
        pdv = pickle.load(f)
    feats = extract_pdv_features(np.log(close).diff().dropna())
    d = dash.load_page1()
    assert d["pdv_sigma1"] == pytest.approx(feats["sigma1"].iloc[-1] * 100, abs=0.006)
    assert d["pdv_forecast"] == pytest.approx(_synth_walk_forward(synth).iloc[-1] * 100, abs=0.006)


def test_unstable_nodes_follow_the_surfaces_recorded_rule(dash, pages):
    """load_page3 re-derived the flag as |z| > 2 (5 of 79) over the surface's own
    vomma-P&L-ratio rule (31 of 79, as the README states)."""
    import pyarrow.parquet as pq

    path = SHIPPED / "greeks" / "greeks_surface.parquet"
    gdf = pd.read_parquet(path)
    rule = json.loads(pq.read_schema(path).metadata[b"greeks_provenance"])["unstable_rule"]
    d = dash.load_page3()
    assert d["unstable_rule"]["kind"] == "pnl_ratio"
    assert d["unstable_rule"]["threshold"] == rule["threshold"]
    assert d["n_unstable"] == int(gdf["is_unstable"].sum()) == int((gdf["vomma_pnl_ratio"].abs() > rule["threshold"]).sum())
    text = _visible_text(pages["/greeks"])
    assert f"{d['n_unstable']} of {d['n_cells']}" in text
    assert f"vomma P&L ratio ½·vomma·|Δσ|/vega > {rule['threshold']:.2f}" in text
    assert "|vomma z-score| > 2" not in text and "pooled score" not in text
    # every dropped grid node says why
    empties = [c for row in d["heatmap_rows"] for c in row["cells"] if c.get("empty")]
    assert len(empties) == d["n_missing"] and all(c["reason"] for c in empties)


def test_unstable_falls_back_to_z_only_without_the_ratio_and_says_so(dash, synth):
    path = synth["data"] / "greeks" / "greeks_surface.parquet"
    gdf = pd.read_parquet(path).drop(columns=["vomma_pnl_ratio"])
    gdf.to_parquet(path)
    d = dash.load_page3()
    assert d["unstable_rule"]["kind"] == "zscore"
    assert d["n_unstable"] == int((gdf["vomma_zscore"].abs() > 2).sum())
    html = dash.app.test_client().get("/greeks").get_data(as_text=True)
    assert "fallback: the surface records no vomma P&amp;L ratio" in html


def test_model_smile_never_carries_a_nan(dash, monkeypatch):
    """implied_vol_from_price returns NaN (not None) for ill-posed inputs."""
    from joint_vol_calibration.models import heston

    params = {"kappa": 2.0, "theta": 0.04, "sigma": 0.5, "rho": -0.7, "v0": 0.04}
    # a deep out-of-the-money node with a near-zero price is omitted, never NaN
    deep = dash.heston_model_smile(params, maturities_days=(14,), log_moneyness=(-1.5, -0.8, 0.0, 0.1))
    assert all(np.isfinite(v) for s in deep for v in s["iv"]) and 0.0 in deep[0]["lm"]
    assert -1.5 not in deep[0]["lm"]
    real = heston.implied_vol_from_price
    monkeypatch.setattr(heston, "implied_vol_from_price",
                        lambda price, *a, **k: float("nan") if a[1] > 1.05 else real(price, *a, **k))
    out = dash.heston_model_smile(params, maturities_days=(30,), log_moneyness=(-0.1, 0.0, 0.1))
    assert out[0]["lm"] == [-0.1, 0.0]
    assert all(np.isfinite(v) for v in out[0]["iv"])


@pytest.mark.parametrize("p, says", [(0.0746, "does not beat random entry dates at the 5% level (p = 0.075)"),
                                     (0.0348, "beats random entry dates at the 5% level (p = 0.035)")])
def test_s3_timing_claim_follows_the_stored_null_test(dash, monkeypatch, p, says):
    """'Only S3's timing survives' / 'its timing beats a random-walk null' were
    hard-coded; the regenerated null test gives p = 0.075, not significant."""
    base = dash.load_page4()
    monkeypatch.setattr(dash, "load_page4", lambda: {**base, "kpi": {**base["kpi"], "s3_null_p": p}})
    text = _visible_text(dash.app.test_client().get("/backtest").get_data(as_text=True))
    assert says in text
    assert ("not significant at 5%" in text) is (p >= 0.05)
    assert "Timing Survives" not in text


@pytest.mark.parametrize("route", ["/", "/calibration", "/greeks", "/backtest"])
def test_page_header_dates_never_break_across_lines(dash, route):
    """A date in the page header must sit inside a no-wrap element: at phone
    width the browser otherwise breaks '2026-10-16' at a hyphen."""
    html = dash.app.test_client().get(route).get_data(as_text=True)
    badge = re.search(r'<div class="page-badge"[^>]*>(.*?)</div>', html, re.S)
    assert badge, route
    body = badge.group(1)
    # Strip the no-wrap spans; no ISO date may remain outside one.
    outside = re.sub(r'<span class="nowrap"[^>]*>.*?</span>', "", body, flags=re.S)
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
    for name, c in crisis.items():
        base = nav[nav.index < pd.Timestamp(c["start"])].iloc[-1]
        assert c["total_return"] == pytest.approx(nav.loc[c["start"]:c["end"]].iloc[-1] / base - 1, abs=1e-12)
    assert crisis["FedHikes_2022"]["total_return"] == pytest.approx(0.0470983, abs=1e-6)
    assert crisis["COVID_2020"]["total_return"] == pytest.approx(0.0168672, abs=1e-6)
    assert crisis["Tariffs_2025"]["total_return"] == pytest.approx(-0.0113131, abs=1e-6)
    text = _visible_text(pages["/backtest"])
    assert "FedHikes 2022 (2022-01-01 → 2022-12-31) +4.7%" in text
    assert "COVID 2020 (2020-02-19 → 2020-04-30) +1.7%" in text
    assert "+5.1%" not in text and "+1.8%" not in text
    # a jump on the window's first session counts; a window at the curve's start uses its first NAV
    idx = pd.bdate_range("2021-12-27", "2022-01-07")
    toy = pd.Series([100.0] * 5 + [110.0] * 5, index=idx)
    got = dash.window_returns(toy, {"jump": ("2022-01-01", "2022-01-07"), "head": ("2021-12-01", "2021-12-31")})
    assert got[0]["total_return"] == pytest.approx(0.10) and got[0]["n_days"] == 5
    assert got[1]["total_return"] == pytest.approx(0.0) and got[1]["n_days"] == 4


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
    text = _visible_text(pages["/backtest"])
    assert want in text
    assert "five real 2026 SPX chains" not in text
    # every chain is listed with its ratio and use
    d = dash.load_page4()
    assert [(c["session"], round(c["ratio"], 3)) for c in d["beta_est"]["chains"] if c["kind"] == "DB chain"] == \
        [(s, round(r, 3)) for s, r in zip(t["session"], t["ratio"])]
    # the grid on the page is the CSV, including the row at the mean over every chain
    grid = pd.read_csv(SHIPPED / "backtest" / "beta_sensitivity.csv")
    html = pages["/backtest"]
    assert len(re.findall(r'<tr data-beta="', html)) == len(grid)
    # Every row of the grid is the CSV row, and the ◆ (mean over every chain) marks a row only
    # when that mean is one: each chain the bot stores moves it, often off the grid.
    for _, row in grid.iterrows():
        cells = re.search(rf'<tr data-beta="{row["beta"]:.2f}"[^>]*>(.*?)</tr>', html, re.S).group(1)
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
    assert (len(trades), len(closed)) == (208, 207)
    win = np.mean([t.net_pnl > 0 for t in closed])
    avg = sum(t.net_pnl for t in closed) / 3 / len(closed)
    assert (round(win * 100, 1), round(avg)) == (38.2, -1043)
    text = _visible_text(pages["/backtest"])
    assert (f"Trades 208 1 open at the end; over the 207 closed: win rate {win * 100:.1f}%, "
            f"average {dash._usd(avg)} of portfolio P&L per trade") in text
    assert "Round trips" not in text and "38.0%" not in text and "−$1,039" not in text
    assert "S4 VRP −$273,544" in text and "46 (1 open)" in text
    assert "S4's count includes a straddle opened on the last session (2025-03-24)" in text


def test_displayed_numbers_are_rounded_once(dash, pages):
    """Values rounded in the loader and again in the template printed S3's final
    +5.647% as '+5.7%', the 365-day smile's 12.4249 as '12.43', and Greeks vomma
    3,267.496 as '3,268' (via 3,267.5)."""
    m = dash.MINUS
    # equity-curve table: final / lowest / highest from the unrounded curves
    eq = pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet")
    table = re.search(r'id="equityChart-data">(.*?)</div>', pages["/backtest"], re.S).group(1)
    for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C Contrarian"), ("nav_s3", "S3 Dispersion"),
                      ("nav_s4", "S4 VRP"), ("nav_s1", "S1 PDV (reference)")):
        rel = eq[col] / eq[col].iloc[0] - 1
        want = "".join(f"<td>{dash._pct_prose(x)}</td>" for x in (rel.iloc[-1], rel.min(), rel.max()))
        assert f"<tr><td>{name}</td>{want}</tr>" in table, name
    assert "<tr><td>S3 Dispersion</td><td>+5.6%</td>" in table
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
    g_html = pages["/greeks"]
    vv = re.findall(r'<div class="vv">([^<]+)</div>', g_html)
    assert vv == [f"{x:,.0f}".replace("-", m) for x in gdf["vomma"]]
    zz = [re.sub(r"^⚠ ", "", z).strip() for z in re.findall(r'<div class="z">([^<]+)</div>', g_html)]
    assert zz == [f"{abs(x):.2f}" for x in gdf["vomma_pnl_ratio"]]
    assert "3,267" in vv and "3,268" not in vv


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


@pytest.mark.parametrize("last_vix, big, tail, lead", [
    (95.0, "Record high", "Implied vol is rich versus this spread's own history (highest on record)", "above all"),
    (0.5, "Record low", "Implied vol is cheap versus this spread's own history (lowest on record)", "at or below all"),
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
    assert dash.percentile_rank(50.0)["tail"] is None and dash.percentile_rank(None) is None


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
    w = pd.concat([sig.rename("f"), vix.rename("v")], axis=1, join="inner").loc[d["start_date"]:d["end_date"]].dropna()
    assert pc["share"] == pytest.approx(((w["v"] - w["f"]) > 0.02).mean(), abs=1e-12)
    assert pc["n"] == len(w) == len(pd.read_parquet(SHIPPED / "backtest" / "full_results.parquet"))
    close = spx.set_index(pd.to_datetime(spx["date"]))["close"]
    y = forward_target(np.log(close).diff().dropna(), horizon=21, target="rv")
    both = pd.concat([sig.rename("f"), y.rename("y")], axis=1, join="inner").loc[d["start_date"]:d["end_date"]].dropna()
    assert pc["ratio"] == pytest.approx(both["f"].mean() / both["y"].mean(), abs=1e-12)
    text = _visible_text(pages["/backtest"])
    assert (f"averaged {pc['ratio']:.2f}× the realised RMS vol of the next 21 sessions in the backtest window, "
            f"so “VIX > σ̂ + 2 vol pts” held on {pc['share'] * 100:.0f}% of its {pc['n']:,} sessions") in text
    # the page follows the computation, not a typed figure
    monkeypatch.setattr(dash, "pre_c22_spread_facts", lambda *a, **k: {**pc, "share": 0.5, "ratio": 0.9})
    text = _visible_text(dash.app.test_client().get("/backtest").get_data(as_text=True))
    assert "averaged 0.90×" in text and "held on 50% of its" in text


def test_s3_null_test_is_labelled_as_the_base_size_test(dash, engine_state, pages, synth):
    """The p = 0.075 sat next to S3's +$56,471, but s3_timing_null_test sizes every
    path, the real one included, at S3's base (pass-1) kelly: its real statistic is
    +$41,952, before the sizing overlay. The page says so, and shows the base-size
    figure only when the stored results record it."""
    m = engine_state["metrics"]
    text = _visible_text(pages["/backtest"])
    p = f"{m['s3_null_p_value']:.3f}"
    assert ("its timing, tested on S3 at base (pass-1) size, before the sizing overlay, does not beat "
            f"random entry dates at the 5% level (p = {p})") in text
    assert "S3 at base (pass-1) size, before the sizing overlay, vs random entry dates: not significant at 5%" in text
    assert f"timing null test on S3 at base (pass-1) size, before the sizing overlay: p = {p}" in text
    assert "null p =" not in text and "S3 entries vs random entry dates" not in text
    if "s3_null_real_pnl" not in m:
        assert "+$41,952" not in text                    # never typed in
    # when the results record the base-size statistic, it is quoted
    path = synth["data"] / "backtest" / "full_results.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    state["metrics"]["s3_null_real_pnl"] = 41951.87
    with open(path, "wb") as f:
        pickle.dump(state, f)
    text = _visible_text(dash.app.test_client().get("/backtest").get_data(as_text=True))
    assert "tested on S3 at base (pass-1) size (+$41,952), before the sizing overlay" in text


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
    text = _visible_text(pages["/backtest"])
    assert (f"{len(s3)} trades in 2018–2025 ({entries} entries plus {rolls} rolls of a position held "
            f"past its straddle's expiry)") in text
    assert f"S3 enters {entries} times in 2018–2025" in text and "S3 fires 39 times" not in text


def test_no_double_negative_losses(pages):
    """'it loses −$274K' / 'It loses −$431K' read as gains."""
    for route, html in pages.items():
        assert not re.search(r"\blos(?:es|t|e)\s+[−-]\$", _visible_text(html)), route


def test_page_header_can_wrap_at_every_separator():
    """Header items are no-wrap spans joined by '·' separators with no spaces;
    without a break opportunity at the separator the whole line is one
    unbreakable run (497 px wide at a 390 px viewport). The separator must be
    an inline-block, which browsers may break around."""
    css = (ROOT / "dashboard" / "templates" / "base.html").read_text()
    rule = re.search(r"\.page-badge \.sep\s*\{([^}]*)\}", css)
    assert rule and "display: inline-block" in rule.group(1)
