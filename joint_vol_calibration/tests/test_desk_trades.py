"""
dashboard/desk_trades.py: the backtest's round trips and worst drawdowns (B12).

Drawdown periods are checked on hand-built NAV paths (peak, valley, recovery,
'not recovered' and depth), the deepest period against the running-maximum
drawdown on a random path, and the round-trip statistics on a small trade log
whose answers are worked out by hand (the open trade is kept out of the closed
statistics but in the total; 'combined' records are never read).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_trades as dt  # noqa: E402


def _nav(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_drawdown_periods_on_a_hand_built_path():
    nav = _nav([100, 110, 99, 105, 110, 112, 100.8, 101, 90, 95])
    ps = dt.drawdown_periods(nav)
    days = nav.index.strftime("%Y-%m-%d")
    assert len(ps) == 2
    deep, first = ps                                   # deepest first
    assert (deep["peak"], deep["valley"], deep["recovery"]) == (days[5], days[8], None)
    assert deep["depth"] == pytest.approx(90 / 112 - 1)
    assert (deep["to_valley"], deep["sessions"]) == (3, 4)    # not recovered: to the last session
    assert (first["peak"], first["valley"], first["recovery"]) == (days[1], days[2], days[4])
    assert first["depth"] == pytest.approx(99 / 110 - 1)
    assert (first["to_valley"], first["sessions"]) == (1, 3)
    assert dt.worst_drawdowns(nav, top=1) == [deep]
    assert dt.drawdown_periods(_nav([1, 2, 3])) == []
    assert dt.drawdown_periods(pd.Series([], dtype=float)) == []


def test_the_deepest_period_is_the_running_maximum_drawdown():
    rng = np.random.default_rng(3)
    nav = _nav(1e6 * np.exp(np.cumsum(rng.standard_normal(900) * 0.01)))
    ps = dt.drawdown_periods(nav)
    assert ps[0]["depth"] == pytest.approx(float((nav / nav.cummax() - 1).min()), rel=1e-12)
    assert [p["depth"] for p in ps] == sorted(p["depth"] for p in ps)
    # the periods do not overlap, and every one but possibly the last recovered
    spans = sorted((p["peak"], p["recovery"] or "9999") for p in ps)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))
    assert sum(1 for p in ps if p["recovery"] is None) <= 1


def _t(sig, entry, exit_, pnl, open_=False, direction=-1):
    return SimpleNamespace(signal=sig, entry_date=pd.Timestamp(entry), exit_date=pd.Timestamp(exit_),
                           net_pnl=np.float64(pnl), is_open=open_, direction=direction)


def test_round_trip_statistics_by_hand():
    log = [_t("s4", "2024-01-02", "2024-01-10", -10.0), _t("s4", "2024-02-01", "2024-02-03", 30.0),
           _t("s4", "2024-03-01", "2024-03-21", -50.0),
           _t("s4", "2024-04-01", "2024-04-01", -2.0, open_=True),
           _t("combined", "2024-01-02", "2024-01-03", 1e6), _t("s3", "2024-01-05", "2024-01-06", 7.0)]
    rts = dt.round_trips(log, ("s4",))
    assert [r["entry"] for r in rts] == ["2024-01-02", "2024-02-01", "2024-03-01", "2024-04-01"]
    assert [r["days"] for r in rts] == [8, 2, 20, 0]
    st = dt.sleeve_stats(rts)
    assert (st["n"], st["n_closed"], st["n_open"]) == (4, 3, 1)
    assert st["total"] == pytest.approx(-32.0)             # the open trade's entry cost included
    assert st["p_profit"] == pytest.approx(1 / 3)          # closed trades only
    assert st["mean"] == pytest.approx(-10.0)
    assert st["median"] == pytest.approx(-10.0)
    assert (st["best"], st["worst"]) == (30.0, -50.0)
    assert st["worst_share"] == pytest.approx(1.0)         # two losses, both in the five worst
    assert dt.sleeve_stats(rts, top=1)["worst_share"] == pytest.approx(50 / 60)
    assert st["median_days"] == pytest.approx(8.0)
    assert all(r["signal"] != "combined" for r in dt.round_trips(log, ("s3", "s4")))


def test_histogram_bins_are_closed_on_the_left_and_open_at_the_ends():
    assert dt.histogram([-70, -60, -0.1, 0, 9.99, 10, 60, 99], (-60, 0, 10, 60)) == [1, 2, 2, 1, 2]
    assert dt.histogram([0, 4, 5, 31], (0, 5, 30), open_low=False) == [2, 1, 1]
    assert sum(dt.histogram(np.arange(-100, 100), dt.PNL_BINS_K)) == 200
