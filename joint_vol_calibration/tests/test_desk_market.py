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
