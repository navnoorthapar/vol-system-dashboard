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
    for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C Contrarian"), ("nav_s3", "S3 index-leg short"),
                      ("nav_s4", "S4 VRP"), ("nav_s1", "S1 PDV (reference)")):
        rel = eq[col] / eq[col].iloc[0] - 1
        want = "".join(f'<td class="num">{dash._pct_prose(x)}</td>'
                       for x in (rel.iloc[-1], rel.min(), rel.max()))
        assert f"<tr><td>{name}</td>{want}</tr>" in table, name
    # the C22 case (+5.647% printed as +5.7%) is now +6.105% on the C23 record
    assert '<tr><td>S3 index-leg short</td><td class="num">+6.1%</td>' in table
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
        assert f"r(T) {r_lo:.2f}–{r_hi:.2f}%" in text
        assert f"q(T) {q_lo:.2f}–{q_hi:.2f}% from the fit's parity forwards" in text


def test_corrections_page_shows_the_c23_waterfall_from_its_file(pages):
    bridge = pd.read_csv(SHIPPED / "backtest" / "c23_bridge.csv")
    tab = re.search(r'<table class="tbl dense[^"]*" id="c23-bridge">(.*?)</table>',
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
    port = _visible_text("<body>" + re.search(r"<td[^>]*>Portfolio</td>(.*?)</tr>", boot,
                                               re.DOTALL).group(1))
    def fmt(x):
        return f"{x:+.2f}".replace("-", "−")
    assert f"{fmt(b['ci95'][0])} to {fmt(b['ci95'][1])}" in port
    nul = re.search(r'<table class="tbl dense" id="null-table">(.*?)</table>', html, re.DOTALL).group(1)
    s3 = s["nulls"]["sleeves"]["s3"]
    row = _visible_text("<body>" + re.search(r"<td[^>]*>S3 · short SPX straddle</td>(.*?)</tr>", nul,
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
    arb_row = mon["Quotes in an executable static-arbitrage row (% of scored quotes)"]
    assert arb_row["value"].startswith(f"{share:.1f}%")


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
    heston = next(s for s in smile["series"] if s["name"] == "Heston")
    assert heston["color"] == dash.model_colour("heston_v2")


def test_smile_slice_model_points_share_the_quotes_x_values(dash, pages):
    """UI-2/UI-17: all four models on the slice within the page budget: a model's
    points list only IVs, aligned with that expiry's mids (desk.js takes x from
    them), and the IVs are the residual table's model IVs at those strikes."""
    sl = _spec(pages["/surface"], "chart-slices")
    by = {s["name"]: s for s in sl["series"]}
    res = pd.read_parquet(_surface_file("residuals.parquet"))
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
                "calibration.html": ["quintic-vs-heston"], "research.html": ["improvements-table"],
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
    assert all("short SPX straddle" in x or "index-leg short" in x for x in labels), labels
    text = _visible_text(html)
    assert "S3 · short SPX straddle (index leg of a dispersion trade; no single-name leg)" in text


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
    text = (f"Heston misses real VX futures by {want} index pts RMSE out of sample "
            f"({len(have)} {sess}{tail}).")
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
                      r'<ul class="find-list">(.*?)</ul>', strip.group(1), re.DOTALL)
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
    gap = rep["indices"]["VIX"]["gap_vs_official"]
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


def test_findings_cite_the_skew_power_law_and_link_to_its_panel(dash, pages):
    sk = dash.load_page2()["skew_term"]
    slopes = [p["slopes"]["svi_all"]["slope"] for p in sk["sessions"] if p["slopes"]["svi_all"]]
    items = [it for _, _, its in _findings(pages["/"]) for it in its if "calibration.html#skew-term" in it]
    assert len(items) == 1
    text = _visible_text("<body>" + items[0])
    assert (f"slope in T is {dash.signed(min(slopes), '.2f')} to {dash.signed(max(slopes), '.2f')} over "
            f"{len(slopes)} sessions") in text
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
    assert f"RR25 {head}" in rr
    for old in earlier:
        assert f"SVI-cal ATM {old}" in atm
        assert f"RR25 {old}" in rr


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
        assert f"RR25 {last}" in st["rr"]
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
    step = next(s for s in spec["series"] if s["name"].startswith("Forward ATM vol"))
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


def test_arbitrage_tile_names_the_largest_violation_and_the_table_lists_them(dash, pages):
    html = pages["/surface"]
    arb = json.loads(_surface_file("surface.json").read_text())["arbitrage"]
    tile_m = re.search(r'id="arb-tile">(.*?)</div>\s*</div>', html, re.DOTALL)
    tile = _visible_text("<body>" + tile_m.group(1))
    top = max(arb["largest_mid_violations"], key=lambda v: v["shortfall_index_points"])
    assert f"Largest mid violation {max(arb['max_mid_violation_index_points'].values()):.1f} idx pts" in tile
    assert dash.usd_desk(top["shortfall_index_points"] * 100, signed=False) + " per contract" in tile
    assert "constraint row" in tile
    assert 'href="#largest-violations"' in html
    rows = _table_rows(html, "largest-violations")
    assert len(rows) == len(arb["largest_mid_violations"])
    ordered = sorted(arb["largest_mid_violations"], key=lambda v: -v["shortfall_index_points"])
    for r, v in zip(rows, ordered):
        assert r[2] == f"{v['shortfall_index_points']:.2f}"
        assert r[3] == dash.usd_desk(v["shortfall_index_points"] * 100, signed=False)
        for lg in v["legs"]:
            assert f"{lg['right']}{lg['strike']:.0f}" in r[1]
            assert str(lg["expiry"])[:10] in r[1]
    # the panel table counts constraint rows, and says so
    assert _table_heads(html, "arb-table")[1:] == ["Constraint rows", "Violated at mids",
                                                   "Violated at bid/ask"]


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
