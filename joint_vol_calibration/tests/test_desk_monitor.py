"""
dashboard/desk_monitor.py: Model Risk as an exceptions view (P3.3).

Pure functions: the worst-first order of the checks, the status tally, the
bullet geometry (bands at the limits, a tick proportional to the value) and the
history of a check over the stored sessions (dots coloured by today's limits,
and where today ranks). The page-level checks against the artifacts are in
test_dashboard.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_monitor as dmo  # noqa: E402


def test_checks_run_worst_first_then_in_inventory_order():
    spec = [("M7", "ok", "a"), ("M4", "breach", "b"), ("V1", "breach", "c"), ("M4", "info", "d"),
            ("M4", "watch", "e"), ("M4", "breach", "f"), ("M5", "na", "g")]
    rows = [{"model": m, "status": st, "check": c} for m, st, c in spec]
    got = dmo.order_checks(rows, ["M4", "M5", "M7", "V1"])
    assert [r["check"] for r in got] == ["b", "f", "c", "e", "a", "d", "g"]
    assert dmo.tally(got) == {"breach": 3, "watch": 1, "ok": 1, "info": 1, "na": 1}


@pytest.mark.parametrize(("value", "limits"), [(2.96, (1.5, 3.0)), (0.0, (0.0, 1.0)), (10.0, (0.0, 1.0)),
                                               (0.29, (0.5, 1.5)), (7.46, (2.0, 5.0))])
def test_bullet_bands_sit_at_the_limits_and_the_tick_is_proportional(value, limits):
    b = dmo.bullet(value, limits)
    g, a = limits
    top = max(1.5 * a, value)
    assert b["top"] == pytest.approx(top)
    assert b["tick"] == pytest.approx(value / top * dmo.BULLET_W, abs=0.01)
    ok, watch, breach = b["bands"]
    assert ok["x"] == 0
    assert ok["w"] == pytest.approx(g / top * dmo.BULLET_W, abs=0.01)
    assert watch["x"] + watch["w"] == pytest.approx(a / top * dmo.BULLET_W, abs=0.02)
    assert breach["x"] + breach["w"] == pytest.approx(dmo.BULLET_W, abs=0.02)
    assert f"value {value:.2f}" in b["label"]
    assert f"OK ≤ {g:g}" in b["label"]
    assert f"watch ≤ {a:g}" in b["label"]


def test_bullet_needs_a_non_negative_value():
    assert dmo.bullet(None, (1, 2)) is None
    assert dmo.bullet(-1.0, (1, 2)) is None


def test_history_colours_each_session_by_todays_limits_and_ranks_today():
    pts = [("2026-03-24", 1.98), ("2026-03-27", 1.54), ("2026-05-29", 1.28), ("2026-06-18", 1.83),
           ("2026-10-16", 2.96)]
    h = dmo.history_view(pts, (1.5, 3.0))
    assert [d["status"] for d in h["dots"]] == ["watch", "watch", "ok", "watch", "watch"]
    assert h["rank"] == "worst of 5"
    assert h["range"] == "1.28–2.96"
    assert h["last"] == 2.96
    assert dmo.history_view([("a", 3.0), ("b", 1.0)], (1, 2))["rank"] == "best of 2"
    assert dmo.history_view([("a", 1.0), ("b", 3.0), ("c", 2.0)], (1, 2))["rank"] == "2nd of 3"
    assert dmo.history_view([("a", 1.0), ("b", 0.0), ("c", 1.0)], (0, 1))["rank"] == "tied worst of 3"
    # one stored value is not a history
    assert dmo.history_view([("a", 1.0)], (1, 2)) is None
    assert dmo.history_view([("a", None), ("b", 1.0)], (1, 2)) is None


def test_slugs_are_ascii_ids():
    assert dmo.slug("oos_rmse") == "oos-rmse"
    assert dmo.slug("β drift: newest chain's ATM vol / VIX") == "beta-drift-newest-chain-s-atm-vol-vix"
    assert dmo.slug("") == "check"


def test_history_never_shows_an_earlier_session_as_today():
    """A check today's session cannot compute (None) keeps the earlier dots but
    ranks nothing and reports no value for today, instead of promoting the
    previous session's value to 'today'."""
    h = dmo.history_view([("a", 1.0), ("b", 2.0), ("c", None)], (1.5, 3.0))
    assert h["last"] is None
    assert h["today_missing"]
    assert h["rank"] is None
    assert [d["session"] for d in h["dots"]] == ["a", "b"]
    assert h["last_session"] == "b"
    full = dmo.history_view([("a", 1.0), ("b", 2.0), ("c", 0.5)], (1.5, 3.0))
    assert full["last"] == 0.5
    assert not full["today_missing"]
    assert full["rank"] == "best of 3"

