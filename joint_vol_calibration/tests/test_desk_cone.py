"""
dashboard/desk_cone.py: the volatility cone of SPX realised vol and the SVI-cal
ATM term structure on it (B8).

The percentiles are checked against a window-by-window recomputation on a
fixture, RV_h at 21 sessions against the site's RV21 (desk_market), the
sample's end against look-ahead (a close after the surface session moves
nothing), and the calendar-day to session mapping (T·252/365).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_cone as dc  # noqa: E402
import desk_market as dm  # noqa: E402


@pytest.fixture
def closes() -> pd.Series:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2009-06-01", "2026-10-30")
    # a calm and a stressed regime, so the percentiles spread
    vol = np.where((idx >= pd.Timestamp("2020-02-20")) & (idx <= pd.Timestamp("2020-06-30")), 0.04, 0.008)
    r = rng.standard_normal(len(idx)) * vol
    return pd.Series(4000.0 * np.exp(np.cumsum(r)), index=idx)


def _windows_by_hand(c: pd.Series, h: int) -> np.ndarray:
    r = np.diff(np.log(c.to_numpy(float)))
    return np.array([100.0 * math.sqrt(252.0 / h * float(np.sum(r[i - h:i] ** 2)))
                     for i in range(h, len(r) + 1)])


def test_cone_percentiles_match_a_window_by_window_recomputation(closes):
    end = "2026-10-16"
    start = dc.sample_start("since-2010", closes, end)
    assert start == pd.Timestamp("2010-01-04")
    r = dc.sample_returns(closes, start, end)
    sub = closes[(closes.index >= start) & (closes.index <= end)]
    for row in dc.cone(r):
        want = _windows_by_hand(sub, row["h"])
        assert row["n"] == len(want) == len(sub) - row["h"]
        for p in dc.PCTS:
            assert row["pct"][p] == pytest.approx(np.percentile(want, p), rel=1e-12)
        assert row["current"] == pytest.approx(want[-1], rel=1e-12)
        assert row["last"] == end                     # the cone ends on the sample's last session
        assert row["pct"][5] < row["pct"][25] < row["pct"][50] < row["pct"][75] < row["pct"][95]


def test_rv_at_21_sessions_is_the_sites_rv21(closes):
    r = np.log(closes).diff()
    a = dc.rv_windows(r.dropna(), 21)
    b = dm.rv_trailing(r).dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 4000
    np.testing.assert_allclose(a[common].to_numpy(), b[common].to_numpy(), rtol=1e-10)


def test_constant_returns_give_a_flat_cone():
    idx = pd.bdate_range("2010-01-04", periods=600)
    a = 0.01
    c = pd.Series(100.0 * np.exp(np.cumsum(np.r_[0.0, np.tile([a, -a], 300)[:599]])), index=idx)
    r = dc.sample_returns(c, idx[0], idx[-1])
    for row in dc.cone(r):
        for p in dc.PCTS:
            assert row["pct"][p] == pytest.approx(100.0 * a * math.sqrt(252.0), rel=1e-9)


def test_nothing_after_the_surface_session_enters(closes):
    end = pd.Timestamp("2026-10-16")
    shocked = closes.copy()
    shocked[shocked.index > end] *= np.exp(np.linspace(0.0, 0.5, int((shocked.index > end).sum())))
    for name, _label, _short in dc.SAMPLES:
        a = dc.cone(dc.sample_returns(closes, dc.sample_start(name, closes, end), end))
        b = dc.cone(dc.sample_returns(shocked, dc.sample_start(name, shocked, end), end))
        assert a == b


def test_the_last_three_years_start_on_the_first_session_three_years_back(closes):
    s = dc.sample_start("last-3y", closes, "2026-10-16")
    assert s == pd.Timestamp("2023-10-16")
    weekend = closes[closes.index != pd.Timestamp("2023-10-16")]
    assert dc.sample_start("last-3y", weekend, "2026-10-16") == pd.Timestamp("2023-10-17")
    with pytest.raises(ValueError, match="no close on or after"):
        dc.sample_start("since-2010", closes[closes.index < "2010-01-01"], "2026-10-16")


def test_implied_points_map_calendar_days_to_sessions_and_rank(closes):
    end = "2026-10-16"
    r = dc.sample_returns(closes, dc.sample_start("since-2010", closes, end), end)
    listed = [{"expiry": "a", "T_days": 30.0, "atmf": 13.8}, {"expiry": "b", "T_days": 365.0, "atmf": 16.0},
              {"expiry": "c", "T_days": 400.0, "atmf": 17.0}, {"expiry": "d", "T_days": 2.0, "atmf": 20.0},
              {"expiry": "e", "T_days": None, "atmf": 15.0}]
    pts, beyond = dc.implied_points(listed, r)
    assert beyond == 1                                 # 400 d = 276 sessions: past the cone
    by = {p["expiry"]: p for p in pts}
    assert set(by) == {"a", "b", "d"}
    assert by["a"]["sessions"] == pytest.approx(30 * 252 / 365)
    assert by["a"]["h"] == 21
    assert by["b"]["h"] == 252
    assert by["d"]["h"] == 2                           # at least two returns in a window
    rv = dc.rv_windows(r, 21).to_numpy()
    assert by["a"]["rank"] == pytest.approx(100.0 * float(np.mean(rv < 13.8)))
    assert dc.rank(r, 21, float("nan")) is None
