"""
dashboard/desk_market.py: market series beside the models.

The VX futures leg (P3.1): the Calibration page redraws each gated fit's
out-of-sample VX score from the stored settlements. These tests recompute it
with their own SQL and the library pricer and check it against the fit's own
record (losses.vx_futures_rmse_oos), session by session.

Not a deploy-gate file: it pins values of the backtest window and of the
committed fits verified when the panels were specified (2018-01-02 → 2025-03-24
and the four 2026 fits with settlements), which later sessions do not change.
"""
from __future__ import annotations

import math
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
DB = ROOT / "data_store" / "vol_system.db"
CALS = ROOT / "data_store" / "calibrations"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_market as dm  # noqa: E402

from joint_vol_calibration.models.heston import (  # noqa: E402
    heston_vix_future_exact,
    heston_vix_term_index,
)

# the four committed fits with VX settlements on their session: (session, contracts, RMSE)
KNOWN_VX = (("2026-03-24", 9, 2.6639), ("2026-03-27", 9, 3.1761), ("2026-05-29", 9, 2.5556),
            ("2026-06-18", 8, 1.8535))


def _losses(cal: dict) -> dict:
    return cal.get("leg_losses") or cal.get("losses") or {}


def _settles(con, session: str) -> list[dict]:
    """The test's own query of the rows the calibrator scores (database.get_vix_futures_curve)."""
    cur = con.execute("SELECT expiry_date, expiry_month, is_weekly, settle, close, days_to_expiry "
                      "FROM vix_futures_daily WHERE date = ? AND expiry_date > ? AND is_weekly = 0",
                      (session, session))
    return [{"expiry_date": e, "expiry_month": m, "is_weekly": w,
             "price": s if s is not None else c, "days_to_expiry": d} for e, m, w, s, c, d in cur]


def _gated_cals() -> list[tuple[str, dict]]:
    dash = DASH / "app.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("dashboard_app_for_desk_market", dash)
    assert spec is not None
    assert spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    out = []
    for r in app.scan_calibrations():
        if r["legacy"] is False and r["ok"] and r["session"]:
            with open(r["path"], "rb") as f:
                out.append((r["session"], pickle.load(f)))
    return out


@pytest.fixture(scope="module")
def gated():
    return _gated_cals()


def test_vx_leg_reproduces_each_fits_recorded_out_of_sample_rmse(gated):
    """Acceptance 1: for every gated fit that recorded a VX score, the model curve
    recomputed from its parameters against the stored settlements of its own
    session gives losses.vx_futures_rmse_oos to 1e-6."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    checked = 0
    for session, cal in gated:
        rec = _losses(cal).get("vx_futures_rmse_oos")
        rows = _settles(con, session)
        leg = dm.vx_model_leg(cal["params"], rows)
        if rec is None:
            assert leg is None, session          # a fit with no record had nothing to score
            continue
        assert leg is not None, session
        p = cal["params"]
        T = np.array([r["days_to_expiry"] for r in rows if r["price"] and r["days_to_expiry"] > 0]) / 365.0
        px = np.array([r["price"] for r in rows if r["price"] and r["days_to_expiry"] > 0])
        ref = math.sqrt(float(np.mean((heston_vix_future_exact(p["kappa"], p["theta"], p["sigma"], p["v0"], T)
                                       - px) ** 2)))
        assert leg["rmse"] == pytest.approx(rec, abs=1e-6), session
        assert leg["rmse"] == pytest.approx(ref, abs=1e-12), session
        assert leg["n"] == len(px)
        checked += 1
    assert checked >= 4


@pytest.mark.parametrize(("session", "n", "rmse"), KNOWN_VX)
def test_vx_leg_known_values(session, n, rmse):
    with open(CALS / f"joint_cal_{session}.pkl", "rb") as f:
        cal = pickle.load(f)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    leg = dm.vx_model_leg(cal["params"], _settles(con, session))
    assert leg["n"] == n
    assert round(leg["rmse"], 4) == rmse
    for r in leg["rows"]:
        assert r["gap"] == pytest.approx(r["model"] - r["settle"])


def test_vx_leg_is_none_without_settlements_and_never_borrows_a_neighbour():
    """A fit whose session stores no settlements has no leg: nothing is put in its place."""
    p = {"kappa": 5.0, "theta": 0.05, "sigma": 1.5, "v0": 0.03}
    assert dm.vx_model_leg(p, []) is None
    # only weeklies, expired or unpriced contracts: nothing the calibrator scores
    rows = [{"expiry_date": "2026-11-18", "expiry_month": "2026-11", "is_weekly": 1, "price": 18.0,
             "days_to_expiry": 30},
            {"expiry_date": "2026-10-21", "expiry_month": "2026-10", "is_weekly": 0, "price": 17.0,
             "days_to_expiry": 0},
            {"expiry_date": "2026-12-16", "expiry_month": "2026-12", "is_weekly": 0, "price": None,
             "days_to_expiry": 60}]
    assert dm.live_monthly(rows) == []
    assert dm.vx_model_leg(p, rows) is None


def test_vx_leg_curve_starts_at_the_models_spot_vix_and_passes_through_each_contract():
    p = {"kappa": 4.0, "theta": 0.06, "sigma": 1.2, "v0": 0.025}
    rows = [{"expiry_date": f"2027-0{i}-15", "expiry_month": f"2027-0{i}", "is_weekly": 0,
             "price": 18.0 + i, "days_to_expiry": 30 * i} for i in range(1, 6)]
    leg = dm.vx_model_leg(p, rows)
    curve = dict(leg["curve"])
    assert curve[0.0] == pytest.approx(heston_vix_term_index(4.0, 0.06, 1.2, 0.025, 30 / 365), rel=1e-8)
    for r in leg["rows"]:
        assert curve[r["days"]] == pytest.approx(r["model"], rel=1e-12)
    assert [r["days"] for r in leg["rows"]] == sorted(r["days"] for r in leg["rows"])
    assert leg["n_below"] == sum(1 for r in leg["rows"] if r["gap"] < 0)


# ── implied against subsequently realised vol (P3.2) ─────────────────────────

def _ref_rv(r: np.ndarray, n: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """A loop reference: RMS of the n returns to t (trailing) and of t+1..t+n (forward)."""
    trl = np.full(len(r), np.nan)
    fwd = np.full(len(r), np.nan)
    for t in range(len(r)):
        if t >= n - 1 and np.all(np.isfinite(r[t - n + 1:t + 1])):
            trl[t] = math.sqrt(252.0 / n * float(np.sum(r[t - n + 1:t + 1] ** 2))) * 100
        if t + n < len(r):
            fwd[t] = math.sqrt(252.0 / n * float(np.sum(r[t + 1:t + n + 1] ** 2))) * 100
    return trl, fwd


def test_rv_trailing_and_forward_match_a_reference_on_a_fixture():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2020-01-01", periods=120)
    r = pd.Series(rng.normal(0, 0.012, len(idx)), index=idx)
    trl, fwd = _ref_rv(r.to_numpy())
    np.testing.assert_allclose(dm.rv_trailing(r).to_numpy(), trl, rtol=1e-12, equal_nan=True)
    np.testing.assert_allclose(dm.rv_forward(r).to_numpy(), fwd, rtol=1e-12, equal_nan=True)
    f = dm.rv_forward(r)
    assert f.iloc[-21:].isna().all()
    assert f.iloc[:-21].notna().all()
    assert dm.rv_trailing(r).iloc[:20].isna().all()


def test_rv_forward_is_the_forecast_evaluations_target():
    """The forward RV is forecast_eval.json's y(t), as a vol: y² = 252/21 Σ r²_{t+i}."""
    from joint_vol_calibration.stats.forecast import target_variance
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0, 0.01, 300), index=pd.bdate_range("2021-01-01", periods=300))
    np.testing.assert_allclose((dm.rv_forward(r) / 100) ** 2, target_variance(r), rtol=1e-12, equal_nan=True)


def _db_frame() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    spx = pd.read_sql("SELECT date, close FROM spx_ohlcv ORDER BY date", con, index_col="date")
    vix = pd.read_sql("SELECT date, close FROM vix_daily ORDER BY date", con, index_col="date")["close"]
    spx.index, vix.index = pd.to_datetime(spx.index), pd.to_datetime(vix.index)
    r = pd.Series(np.log(spx["close"].to_numpy(float)), index=spx.index).diff()
    return pd.DataFrame({"vix": vix, "rv_fwd": dm.rv_forward(r).reindex(vix.index),
                         "r": r.reindex(vix.index)})


def test_rv_forward_on_the_committed_db_ends_21_sessions_before_its_last_session():
    """Acceptance 1: no forward value in the last 21 sessions (2026-09-24 when the
    DB ended 2026-10-23; the bot appends sessions, so the property is checked)."""
    f = _db_frame()
    last_fwd = f["rv_fwd"].dropna().index[-1]
    assert list(f.index).index(last_fwd) == len(f) - 22
    assert f["rv_fwd"].iloc[-21:].isna().all()


# the backtest window's ex-post premium, verified by hand when the panel was specified
BETA_TABLE = {1.00: (3.11, 4.32, 79.2, 50.1), 0.82: (-0.46, 1.19, 61.6, -97.7),
              0.80: (-0.85, 0.86, 58.2, -112.3), 0.70: (-2.84, -1.01, 39.6, -180.0)}


@pytest.mark.parametrize("beta", sorted(BETA_TABLE))
def test_vrp_beta_table_over_the_backtest_window(beta):
    """Acceptance 2: the 16 numbers of the β table (±0.01; the variance-space mean
    ±0.1), recomputed here from vol_system.db without desk_market's helpers."""
    f = _db_frame().loc["2018-01-02":"2025-03-24"].dropna(subset=["rv_fwd"])
    assert len(f) == 1816
    v = beta * f["vix"] - f["rv_fwd"]
    var = (beta * f["vix"]) ** 2 - f["rv_fwd"] ** 2
    mean, med, share, vmean = BETA_TABLE[beta]
    assert abs(v.mean() - mean) <= 0.01
    assert abs(v.median() - med) <= 0.01
    assert abs((v > 0).mean() * 100 - share) <= 0.05
    assert abs(var.mean() - vmean) <= 0.1
    st = dm.vrp_stats(_db_frame()["vix"], _db_frame()["rv_fwd"], beta, "2018-01-02", "2025-03-24")
    assert st["n"] == len(f)
    assert st["mean"] == pytest.approx(v.mean(), abs=1e-12)
    assert st["median"] == pytest.approx(v.median(), abs=1e-12)
    assert st["share_pos"] == pytest.approx((v > 0).mean() * 100, abs=1e-12)
    assert st["var_mean"] == pytest.approx(var.mean(), abs=1e-9)


def test_vrp_full_sample_line():
    f = _db_frame().dropna(subset=["rv_fwd"])
    v = f["vix"] - f["rv_fwd"]
    st = dm.vrp_stats(_db_frame()["vix"], _db_frame()["rv_fwd"], 1.0)
    assert st["mean"] == pytest.approx(v.mean(), abs=1e-12)
    assert st["share_pos"] == pytest.approx((v > 0).mean() * 100, abs=1e-12)
    # the backtest window's sessions are a fixed part of it: the full sample through 2026-09-24
    # gave +3.75 and 84.6% when the panel was specified; later sessions move it little
    assert 3.0 < st["mean"] < 4.5


def test_yearly_means_mark_a_partial_last_year_with_its_span():
    idx = pd.bdate_range("2023-12-20", "2025-03-24")
    x = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    ys = dm.yearly_means(x)
    assert [y["year"] for y in ys] == [2023, 2024, 2025]
    assert [y["partial"] for y in ys] == [False, False, True]
    assert ys[-1]["last"] == "2025-03-24"
    assert ys[-1]["n"] == len(idx[idx.year == 2025])
    assert ys[1]["mean"] == pytest.approx(x[x.index.year == 2024].mean())


# ── the VIX complex on the Overview (P3.5) ───────────────────────────────────

def test_vx_curve_on_a_fixture_day_is_front_first_monthly_and_its_contango():
    rows = [{"expiry_date": "2026-12-16", "expiry_month": "2026-12", "is_weekly": 0, "price": 19.0,
             "days_to_expiry": 60},
            {"expiry_date": "2026-11-18", "expiry_month": "2026-11", "is_weekly": 0, "price": 18.0,
             "days_to_expiry": 32},
            {"expiry_date": "2026-11-04", "expiry_month": "2026-11", "is_weekly": 1, "price": 30.0,
             "days_to_expiry": 18},
            {"expiry_date": "2026-10-21", "expiry_month": "2026-10", "is_weekly": 0, "price": 17.0,
             "days_to_expiry": 4}]
    c = dm.vx_curve(rows)
    assert [r["expiry"] for r in c["rows"]] == ["2026-10-21", "2026-11-18", "2026-12-16"]
    assert c["m1m2"] == pytest.approx(18.0 / 17.0 - 1)
    one = dm.vx_curve(rows[:1])
    assert one["m1m2"] is None
    assert dm.vx_curve([]) is None


def test_vx_curve_m1_m2_on_the_committed_db():
    """Acceptance 2: M1 → M2 = +4.28% on 2026-10-09 (17.12 → 17.85)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c = dm.vx_curve(_settles(con, "2026-10-09"))
    assert round(c["m1m2"] * 100, 2) == 4.28
    assert round(c["m1"]["settle"], 2) == 17.12
    assert round(c["m2"]["settle"], 2) == 17.85


def test_ratio_view_on_a_fixture():
    r = pd.Series([0.9, 1.1, 1.2, 0.95, 1.05, 0.8], index=pd.bdate_range("2026-01-01", periods=6))
    v = dm.ratio_view(r)
    assert v["share_above"] == pytest.approx(50.0)
    assert v["pct"] == pytest.approx(0.0)
    assert v["runs"] == [("2026-01-02", "2026-01-05"), ("2026-01-07", "2026-01-07")]
    v2 = dm.ratio_view(pd.Series([0.9, 0.8, 1.0, 0.85], index=pd.bdate_range("2026-01-01", periods=4)))
    assert v2["pct"] == pytest.approx(100 / 3)        # strictly below today (0.80), among the prior three


def test_vix_over_vix3m_share_and_percentile_on_the_committed_db():
    """Acceptance 4: backwardation on 7.6% of 4,228 sessions and today's 0.950 at the
    80.8th percentile, on the DB as it stood through 2026-10-23 (later sessions
    are the bot's; the snapshot is fixed here)."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    df = pd.read_sql("SELECT date, ticker, close FROM vix_term_structure WHERE ticker IN ('^VIX', '^VIX3M') "
                     "AND date <= '2026-10-23'", con)
    w = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    v = dm.ratio_view(w["^VIX"] / w["^VIX3M"])
    assert v["n"] == 4228
    assert round(v["share_above"], 1) == 7.6
    assert round(v["today"], 3) == 0.950
    assert round(v["pct"], 1) == 80.8


def test_pctile_prior_ranks_the_newest_value_against_the_window_before_it():
    """B13: the ticker's VIX percentile: of the ``window`` sessions before the
    newest (NaN dropped, the newest itself left out), how many closed strictly
    lower; ties do not count as below. Fewer earlier sessions than the window:
    None (no shorter window is put in its place)."""
    idx = pd.bdate_range("2026-01-01", periods=7)
    x = pd.Series([30.0, 10.0, np.nan, 12.0, 15.0, 15.0, 15.0], index=idx)
    p = dm.pctile_prior(x, window=4)
    # the four sessions before the newest (NaN dropped): 10, 12, 15, 15 → two strictly below 15
    assert (p["n_below"], p["n"], p["pct"]) == (2, 4, 50.0)
    assert p["value"] == 15.0
    assert (p["date"], p["first"], p["last"]) == ("2026-01-09", "2026-01-02", "2026-01-08")
    assert dm.pctile_prior(x, window=5)["n_below"] == 2        # 30 enters the window, above 15
    assert dm.pctile_prior(x, window=6) is None                  # only five earlier values
    assert dm.PCTILE_WINDOW == 252
    big = pd.Series(np.arange(300.0), index=pd.bdate_range("2025-01-01", periods=300))
    assert dm.pctile_prior(big)["pct"] == 100.0                  # above all 252 before it
    assert dm.pctile_prior(big[::-1].set_axis(big.index))["pct"] == 0.0
