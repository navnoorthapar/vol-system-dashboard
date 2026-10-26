"""
dashboard/desk_surface.py: the fitted SVI-cal surface in desk coordinates.

The Surface page's delta grid (P2.1) quotes each listed expiry and five
constant maturities by forward delta, with RR/BF, the ATM skew and the
variance-swap vol of the same chain. These tests check the numbers against the
committed 2026-10-16 surface report (values verified by hand when the panel was
specified) and against independent re-computations on every stored session:
the pillar strikes recover their target deltas, ATMF is the term structure's
SVI-cal ATM vol, constant-maturity rows are a linear-in-total-variance
interpolation and never an extrapolation, and an extrapolated wing is flagged
and kept out of RR/BF.

Not a deploy-gate file: it reads the committed 2026-10-16 session by name (the
surface report keeps every session it wrote), as a fixture of known values.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
SURF = ROOT / "data_store" / "surface"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_surface as ds  # noqa: E402

KNOWN = "2026-10-16"
SESSIONS = sorted(p.name for p in SURF.iterdir() if p.is_dir() and (p / "surface.json").exists())


def _inputs(session: str) -> tuple[dict, pd.DataFrame, dict]:
    root = SURF / session
    return (json.loads((root / "surface.json").read_text()), pd.read_parquet(root / "residuals.parquet"),
            json.loads((root / "vix_replication.json").read_text()))


@pytest.fixture(scope="module")
def grids() -> dict[str, list[dict]]:
    return {s: ds.delta_grid(*_inputs(s)) for s in SESSIONS}


def _row(grid: list[dict], expiry: str) -> dict:
    return next(r for r in grid if r["expiry"] == expiry)


def test_the_known_session_matches_the_values_verified_by_hand(grids):
    g = grids[KNOWN]
    r = _row(g, "2026-11-16")
    assert r["T_days"] == pytest.approx(31.0, abs=1e-9)
    for key, want in (("atmf", 13.82), ("rr25", -5.82), ("bf25", 0.84), ("rr10", -13.91), ("bf10", 3.89)):
        assert r[key] == pytest.approx(want, abs=0.01), key
    # the put pillars' strikes at 31 days
    assert r["k"]["25P"] == pytest.approx(-0.033, abs=5e-4)
    assert r["k"]["10P"] == pytest.approx(-0.090, abs=5e-4)
    r = _row(g, "2027-04-16")
    for key, want in (("atmf", 15.27), ("rr25", -6.36), ("bf25", 0.88)):
        assert r[key] == pytest.approx(want, abs=0.01), key
    assert len(g) == len(json.loads((SURF / KNOWN / "surface.json").read_text())["term_structure"])


@pytest.mark.parametrize("session", SESSIONS)
def test_atmf_is_the_term_structures_svi_cal_atm_vol(grids, session):
    """surface.json stores atm_vol_svi_cal to 10 significant figures: equal to 1e-9 relative."""
    ts = {t["expiry"]: t for t in _inputs(session)[0]["term_structure"]}
    for r in grids[session]:
        assert r["atmf"] == pytest.approx(ts[r["expiry"]]["atm_vol_svi_cal"], rel=1e-9, abs=0), r["expiry"]


@pytest.mark.parametrize("session", SESSIONS)
def test_every_pillar_strike_recovers_its_delta(grids, session):
    """Black forward delta re-computed from the slice's own total variance at the
    pillar's k (scipy.stats.norm, not the module's ndtr) equals the target."""
    slices = {s["expiry"]: s for s in _inputs(session)[0]["surfaces"]["svi_cal"]["params"]["slices"]}
    for r in grids[session]:
        sl = slices[r["expiry"]]
        for key, _label, target in ds.PILLARS:
            k = r["k"][key]
            x = k - sl["m"]
            w = sl["a"] + sl["b"] * (sl["rho"] * x + math.sqrt(x * x + sl["sigma"] ** 2))
            nd1 = norm.cdf(-k / math.sqrt(w) + math.sqrt(w) / 2)
            if target is None:
                assert k == 0.0
                continue
            got = nd1 if target > 0 else nd1 - 1.0
            assert got == pytest.approx(target, abs=1e-6), (r["expiry"], key)
            assert r["vol"][key] == pytest.approx(100 * math.sqrt(w / sl["T"]), rel=1e-12)


@pytest.mark.parametrize("session", SESSIONS)
def test_rr_and_bf_follow_their_definitions(grids, session):
    for r in grids[session]:
        v = r["vol"]
        for wing in ("25", "10"):
            if r["extrap"][wing + "C"] or r["extrap"][wing + "P"]:
                assert r["rr" + wing] is None
                assert r["bf" + wing] is None
                continue
            assert r["rr" + wing] == pytest.approx(v[wing + "C"] - v[wing + "P"], abs=1e-12)
            assert r["bf" + wing] == pytest.approx((v[wing + "C"] + v[wing + "P"]) / 2 - v["ATM"], abs=1e-12)


def _reference_cm(grid: list[dict], days: int) -> dict | None:
    """Linear in total variance at fixed delta, the bracket found by searchsorted."""
    ts = np.array([r["T"] for r in grid])
    t = days / 365.0
    if t < ts[0] or t > ts[-1]:
        return None
    j = int(np.searchsorted(ts, t, side="left"))
    if ts[j] == t:
        lo = hi = grid[j]
    else:
        lo, hi = grid[j - 1], grid[j]
    out = {}
    for key, _, _ in ds.PILLARS:
        w1, w2 = (lo["vol"][key] / 100) ** 2 * lo["T"], (hi["vol"][key] / 100) ** 2 * hi["T"]
        w = w1 if hi is lo else w1 + (w2 - w1) * (t - lo["T"]) / (hi["T"] - lo["T"])
        out[key] = 100 * math.sqrt(w / t)
    out["lower"], out["upper"] = lo["expiry"], hi["expiry"]
    return out


@pytest.mark.parametrize("session", SESSIONS)
def test_constant_maturity_rows_are_linear_in_total_variance(grids, session):
    cm = ds.constant_maturity(grids[session])
    assert [r["label"] for r in cm] == ["1M", "2M", "3M", "6M", "1Y"]
    assert [r["days"] for r in cm] == [30, 61, 91, 182, 365]
    for r in cm:
        ref = _reference_cm(grids[session], r["days"])
        if ref is None:
            assert not r["available"]
            assert all(v is None for v in r["vol"].values())
            continue
        assert r["available"]
        assert (r["lower"], r["upper"]) == (ref["lower"], ref["upper"])
        for key, _, _ in ds.PILLARS:
            assert r["vol"][key] == pytest.approx(ref[key], abs=1e-9), (r["label"], key)
        assert r["atmf"] == r["vol"]["ATM"]


def _slice(T: float, **kw) -> dict:
    sl = {"a": 0.0004, "b": 0.05, "rho": -0.6, "m": 0.0, "sigma": 0.12, "T": T,
          "expiry": f"2030-01-{int(T * 365):02d}"}
    sl.update(kw)
    return sl


def _surface(slices: list[dict], kranges: dict[str, tuple[float, float]]) -> tuple[dict, pd.DataFrame, dict]:
    sj = {"surfaces": {"svi_cal": {"params": {"slices": slices}}},
          "term_structure": [{"expiry": s["expiry"], "T_days": s["T"] * 365, "forward": 5000.0,
                              "n_quotes": 10, "median_half_spread_vp": 0.2} for s in slices]}
    res = pd.DataFrame([{"model": "svi_cal", "expiry": e, "k": k} for e, (lo, hi) in kranges.items()
                        for k in (lo, hi)])
    vix = {"strip": [{"expiry": s["expiry"], "var_swap_vol": 20.0} for s in slices]}
    return sj, res, vix


def test_a_pillar_beyond_the_quoted_strikes_is_flagged_and_left_out_of_rr_bf():
    sl = _slice(28 / 365)
    k10c = ds.delta_strike(sl, 0.10)
    k25c = ds.delta_strike(sl, 0.25)
    assert k25c < k10c
    # quoted up to between the 25Δ and the 10Δ call strike
    grid = ds.delta_grid(*_surface([sl], {sl["expiry"]: (-0.5, (k25c + k10c) / 2)}))
    r = grid[0]
    assert r["extrap"] == {"10P": False, "25P": False, "ATM": False, "25C": False, "10C": True}
    assert r["rr10"] is None
    assert r["bf10"] is None
    assert r["rr25"] is not None
    assert r["bf25"] is not None
    assert r["vol"]["10C"] is not None              # the value is shown, flagged


def test_constant_maturity_never_extrapolates_in_maturity():
    """A session whose longest expiry is 200 days has no 1Y row; one that starts
    at 40 days has no 1M row (the older sessions start at 7 days)."""
    slices = [_slice(T / 365, a=0.0004 * T / 30) for T in (40, 70, 120, 200)]
    grid = ds.delta_grid(*_surface(slices, {s["expiry"]: (-1.0, 1.0) for s in slices}))
    cm = {r["label"]: r for r in ds.constant_maturity(grid)}
    for lab in ("1M", "1Y"):
        assert not cm[lab]["available"]
        assert cm[lab]["atmf"] is None
        assert cm[lab]["rr25"] is None
    for lab in ("2M", "3M", "6M"):
        assert cm[lab]["available"]
        assert cm[lab]["atmf"] is not None


def test_delta_inversion_fails_loudly_instead_of_clipping():
    with pytest.raises(ValueError, match="is not in"):
        ds.delta_strike(_slice(0.1), 0.0)
    with pytest.raises(ValueError, match="is not in"):
        ds.delta_strike(_slice(0.1), 1.2)
    # total variance negative somewhere on [−3, 2]: no inversion
    with pytest.raises(ValueError, match="total variance not positive"):
        ds.delta_strike(_slice(0.1, a=-0.05), 0.25)


def test_the_svi_atm_skew_is_the_slope_of_the_slice_at_the_forward():
    sl = _slice(0.25)
    h = 1e-6
    fd = (math.sqrt(ds.svi_w(h, sl) / sl["T"]) - math.sqrt(ds.svi_w(-h, sl) / sl["T"])) / (2 * h)
    assert ds.svi_atm_skew(sl) == pytest.approx(fd, abs=1e-7)
