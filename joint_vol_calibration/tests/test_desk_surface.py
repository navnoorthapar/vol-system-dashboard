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

import itertools
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


# ── P2.2: the ATM-skew term structure ────────────────────────────────────────

@pytest.mark.parametrize("session", SESSIONS)
def test_the_analytic_svi_atm_skew_equals_a_finite_difference_on_every_slice(session):
    h = 1e-5
    for sl in _inputs(session)[0]["surfaces"]["svi_cal"]["params"]["slices"]:
        fd = (math.sqrt(ds.svi_w(h, sl) / sl["T"]) - math.sqrt(ds.svi_w(-h, sl) / sl["T"])) / (2 * h)
        assert ds.svi_atm_skew(sl) == pytest.approx(fd, abs=1e-6), sl["expiry"]


# the slopes verified when the panel was specified (B2 labelled the all-expiry
# eSSVI values 'T >= 30 d'): all expiries, then T >= 30 d with its expiry count
SLOPES = {
    "2026-03-24": (-0.409, -0.439, -0.465, -0.493, 27),
    "2026-03-27": (-0.315, -0.358, -0.427, -0.441, 25),
    "2026-05-29": (-0.336, -0.392, -0.310, -0.415, 26),
    "2026-06-18": (-0.357, -0.422, -0.325, -0.454, 27),
    "2026-10-16": (-0.450, -0.557, -0.467, -0.583, 13),
}


@pytest.mark.parametrize("session", sorted(SLOPES))
def test_skew_power_law_slopes_match_polyfit_and_the_verified_values(session):
    sj = _inputs(session)[0]
    sl = sorted(sj["surfaces"]["svi_cal"]["params"]["slices"], key=lambda s: s["T"])
    ts = sorted(sj["term_structure"], key=lambda t: t["T_days"])
    t_svi = np.array([s["T"] for s in sl])
    k_svi = np.array([ds.svi_atm_skew(s) for s in sl])
    t_ess = np.array([t["T_days"] / 365 for t in ts])
    k_ess = np.array([t["atm_skew_essvi"] for t in ts])
    got = []
    cases = ((t_svi, k_svi, None), (t_ess, k_ess, None), (t_svi, k_svi, 30), (t_ess, k_ess, 30))
    for t, k, min_days in cases:
        keep = np.ones(len(t), bool) if min_days is None else t * 365 >= min_days
        ref = np.polyfit(np.log(t[keep]), np.log(np.abs(k[keep])), 1)[0]
        s = ds.skew_slope(t, k, min_days)
        assert s["slope"] == pytest.approx(ref, abs=1e-12)
        assert s["n"] == int(keep.sum())
        got.append(s)
    want = SLOPES[session]
    for s, w in zip(got, want[:4]):
        assert s["slope"] == pytest.approx(w, abs=0.001)
    assert got[2]["n"] == want[4]
    assert got[2]["t_lo_days"] >= 30


def test_the_steepest_slope_prints_unclipped():
    """2026-10-16's all-expiry eSSVI slope, the steepest of the twenty, prints as
    −0.557 (the page's signed .3f), not clipped to a display range."""
    import desk_format

    sj = _inputs(KNOWN)[0]
    ts = sorted(sj["term_structure"], key=lambda t: t["T_days"])
    s = ds.skew_slope([t["T_days"] / 365 for t in ts], [t["atm_skew_essvi"] for t in ts])
    assert desk_format.signed(s["slope"], ".3f") == "−0.557"


def test_skew_slope_needs_three_expiries_and_ignores_zero_skews():
    assert ds.skew_slope([0.1, 0.2], [-1.0, -0.8]) is None
    s = ds.skew_slope([0.1, 0.2, 0.4, 0.8], [-1.0, 0.0, -0.25, -0.125])
    assert s["n"] == 3
    assert s["slope"] == pytest.approx(-1.0, abs=1e-12)


# ── P2.3: one session against an earlier one ─────────────────────────────────

def test_the_change_grid_is_the_cell_by_cell_difference_of_two_constant_maturity_grids(grids):
    head = SESSIONS[-1]
    new = ds.constant_maturity(grids[head])
    for old_s in SESSIONS[:-1]:
        old = ds.constant_maturity(grids[old_s])
        rows = ds.cm_delta(new, old)
        assert [r["label"] for r in rows] == [r["label"] for r in new]
        for r, a, b in zip(rows, new, old):
            for key in ("atmf", "rr25", "bf25"):
                assert r[key]["new"] == a[key]
                assert r[key]["old"] == b[key]
                want = None if (a[key] is None or b[key] is None) else a[key] - b[key]
                assert r[key]["delta"] == want, (old_s, r["label"], key)


def test_a_tenor_one_session_lacks_has_no_change():
    """The earlier session's longest expiry is 200 days: its 1Y is not
    extrapolated, so the 1Y change is None (printed '—')."""
    long_ = [_slice(T / 365, a=0.0004 * T / 30) for T in (20, 45, 100, 200, 400)]
    short = [_slice(T / 365, a=0.0005 * T / 30) for T in (20, 45, 100, 200)]
    new = ds.constant_maturity(ds.delta_grid(*_surface(long_, {s["expiry"]: (-1.0, 1.0) for s in long_})))
    old = ds.constant_maturity(ds.delta_grid(*_surface(short, {s["expiry"]: (-1.0, 1.0) for s in short})))
    rows = {r["label"]: r for r in ds.cm_delta(new, old)}
    assert rows["1Y"]["available"] is False
    for key in ("atmf", "rr25", "bf25"):
        assert rows["1Y"][key]["delta"] is None
        assert rows["1Y"][key]["old"] is None
        assert rows["1Y"][key]["new"] is not None
        assert rows["3M"][key]["delta"] is not None


# ── P2.4: forward ATM vol between constant maturities ────────────────────────

@pytest.mark.parametrize("session", SESSIONS)
def test_forward_vols_are_the_formula_on_the_constant_maturity_total_variances(grids, session):
    cm = ds.constant_maturity(grids[session])
    fw = ds.forward_vols(cm)
    pts = [r for r in cm if r["available"]]
    pairs = list(itertools.pairwise(pts))
    assert [(f["from"], f["to"]) for f in fw] == [(a["label"], b["label"]) for a, b in pairs]
    for f, (a, b) in zip(fw, pairs):
        w1, w2 = (a["atmf"] / 100) ** 2 * a["T"], (b["atmf"] / 100) ** 2 * b["T"]
        assert f["t2_days"] - f["t1_days"] >= 7
        assert f["flag"] is None                      # SVI-cal is calendar-ordered
        assert f["fwd_vol"] == pytest.approx(100 * math.sqrt((w2 - w1) / (b["T"] - a["T"])), rel=1e-12)


def test_a_falling_total_variance_is_a_calendar_flag_not_a_nan():
    pts = [{"label": "1M", "T": 30 / 365, "atmf": 20.0}, {"label": "2M", "T": 61 / 365, "atmf": 13.0},
           {"label": "3M", "T": 91 / 365, "atmf": 14.0}]
    fw = ds.forward_vols(pts)
    assert fw[0]["flag"] == "calendar"
    assert fw[0]["fwd_vol"] is None
    assert fw[0]["fwd_var"] < 0
    assert fw[1]["flag"] is None
    assert math.isfinite(fw[1]["fwd_vol"])


def test_a_pair_closer_than_seven_days_is_never_a_forward_period():
    pts = [{"label": "a", "T": 13 / 365, "atmf": 14.0}, {"label": "b", "T": 14 / 365, "atmf": 14.3},
           {"label": "c", "T": 30 / 365, "atmf": 14.5}]
    fw = ds.forward_vols(pts)
    assert [(f["from"], f["to"]) for f in fw] == [("a", "c")]


# ── P2.5: provenance of the committed chains ─────────────────────────────────

def test_the_committed_sessions_are_yahoo_snapshots_without_a_quote_time_and_their_expiry_counts():
    """The facts the page badge and the Expiries column print, on the committed
    sessions: every chain is a Yahoo snapshot with no quote time, and the
    sessions list 43 / 40 / 41 / 43 / 17 expiries."""
    want = {"2026-03-24": 43, "2026-03-27": 40, "2026-05-29": 41, "2026-06-18": 43, "2026-10-16": 17}
    for session, n in want.items():
        sj = _inputs(session)[0]
        assert sj["inputs"]["source"] == "yfinance"
        assert sj["inputs"]["quote_time"] is None
        assert sj["universe"]["n_expiries"] == n == len(sj["term_structure"])
    sj = _inputs(KNOWN)[0]
    assert (sj["inputs"]["chain_rows"], sj["universe"]["n_universe"]) == (6191, 2913)
    arb = sj["arbitrage"]
    assert max(arb["max_mid_violation_index_points"].values()) == pytest.approx(94.97, abs=0.005)
    assert len(arb["largest_mid_violations"]) == 25


# ── P2.7: each replicated index's terms and its gap split ────────────────────

@pytest.mark.parametrize("session", SESSIONS)
def test_vix_terms_are_the_headline_methods_and_the_split_adds_up_to_the_gap(session):
    """The terms and weight are those of the index's headline method, and where
    the artifact attributes the gap, −truncation − residual equals the stored
    gap (official − replicated = truncation_pts + residual_pts)."""
    vix = _inputs(session)[2]
    rows = ds.vix_terms(vix)
    assert [r["ticker"] for r in rows] == [t for t in ds.VIX_FAMILY if t in vix["indices"]]
    for r in rows:
        ix = vix["indices"][r["ticker"]]
        m = ix.get(ix.get("headline_method") or "cboe_rules") or {}
        if m.get("near") and m.get("next"):
            assert (r["near"]["days"], r["next"]["days"]) == (m["near"]["days"], m["next"]["days"])
            assert r["near"]["expiry"] == str(m["near"]["expiry"])[:10]
        else:
            # no terms recorded ('fewer than two usable terms'): only an index
            # the chain cannot replicate (the bot's sessions include such chains)
            assert ix.get("replicated") is None
            assert (r["near"], r["next"]) == (None, None)
        assert r["near_weight"] == m.get("near_weight")
        # an index that is not replicated carries its method's own reason, if any
        assert r["reason"] == (str(m["reason"]) if ix.get("replicated") is None and m.get("reason")
                               else None)
        if ix.get("replicated") is None:
            assert (r["gap"], r["truncation"], r["unexplained"]) == (None, None, None)
            continue
        assert r["gap"] == ix["gap_vs_official"]
        ga = ix["gap_attribution"]
        if not ga.get("available"):
            assert (r["truncation"], r["unexplained"]) == (None, None)
            continue
        assert (r["truncation"], r["unexplained"]) == (-ga["truncation_pts"], -ga["residual_pts"])
        assert r["truncation"] + r["unexplained"] == pytest.approx(r["gap"], abs=1e-9)
        assert r["far_wing"] == ga["far_wing_market_minus_model_pts"]


def test_vix_terms_split_is_exercised():
    """The split branch above is not vacuous: across the stored sessions at
    least three indices carry an attributed gap. Counted over every session,
    not per session: a later session whose chain replicates fewer indices (the
    bot stores whatever the book allows) must not fail the suite."""
    n = sum(r["truncation"] is not None for s in SESSIONS for r in ds.vix_terms(_inputs(s)[2]))
    assert n >= 3


def test_vix_terms_of_the_known_session():
    """2026-10-16: VIX interpolates 2026-11-13 (28 d) and 2026-12-04 (49 d) with
    near weight 19/21; its −0.29 gap is −0.04 of strike truncation and −0.24
    unexplained. VIX9D is not replicated: its pair 14 / 28 d would extrapolate."""
    rows = {r["ticker"]: r for r in ds.vix_terms(_inputs(KNOWN)[2])}
    v = rows["VIX"]
    assert v["near"] == {"expiry": "2026-11-13", "days": 28.0}
    assert v["next"] == {"expiry": "2026-12-04", "days": 49.0}
    assert v["near_weight"] == pytest.approx((49 - 30) / (49 - 28), abs=1e-9)
    assert [round(v[k], 2) for k in ("gap", "truncation", "unexplained")] == [-0.29, -0.04, -0.24]
    assert v["surface"] == "svi_cal"
    assert v["far_wing"] > 0
    n = rows["VIX9D"]
    assert (n["replicated"], n["extrapolated"]) == (False, True)
    assert n["near_weight"] > 1
    assert (n["near"]["days"], n["next"]["days"]) == (14.0, 28.0)


def test_a_gap_split_that_does_not_add_up_is_not_shown():
    """The split is printed only when the artifact records it and its parts add
    up to the stored gap; otherwise both parts are None, never a residual
    made up to fit."""
    base = {"headline_method": "cboe_rules", "replicated": 20.0, "gap_vs_official": -0.30,
            "cboe_rules": {"near": {"expiry": "2026-11-13", "days": 28},
                           "next": {"expiry": "2026-12-04", "days": 49},
                           "near_weight": 0.9, "extrapolated": False, "constant_maturity_days": 30},
            "gap_attribution": {"available": True, "truncation_pts": 0.05, "residual_pts": 0.25,
                                "far_wing_market_minus_model_pts": -0.1, "surface": "svi_cal"}}
    ok = ds.vix_terms({"indices": {"VIX": base}})[0]
    assert (ok["truncation"], ok["unexplained"], ok["far_wing"]) == (-0.05, -0.25, -0.1)
    off = json.loads(json.dumps(base))
    off["gap_attribution"]["residual_pts"] = 0.20
    assert ds.vix_terms({"indices": {"VIX": off}})[0]["truncation"] is None
    gone = json.loads(json.dumps(base))
    gone["gap_attribution"] = {"available": False, "reason": "term not fitted"}
    r = ds.vix_terms({"indices": {"VIX": gone}})[0]
    assert (r["gap"], r["truncation"], r["unexplained"]) == (-0.30, None, None)
    nopair = json.loads(json.dumps(base))
    nopair.update({"headline_method": None, "replicated": None, "gap_vs_official": None,
                   "cboe_rules": {"ok": False, "reason": "fewer than two usable terms"},
                   "gap_attribution": {"available": False, "reason": "index not replicated"}})
    r = ds.vix_terms({"indices": {"VIX6M": nopair}})[0]
    assert (r["near"], r["next"], r["near_weight"], r["gap"], r["truncation"]) == (None,) * 5
    assert r["reason"] == "fewer than two usable terms"
    assert ds.vix_terms(None) == []
    assert ds.vix_terms({"indices": {}}) == []
