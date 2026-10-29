"""
dashboard/desk_density.py: the risk-neutral density and tail probabilities of
the certified SVI-cal surface (B9).

Breeden–Litzenberger, checked independently of the closed forms the module
uses: on every certified slice of every stored session the density integrates
to 1 and has the forward as its mean (E[S_T/F] = 1), equals the second strike
derivative of the Black call priced on the slice, and P(S_T < K) equals the
first strike derivative of the put and the density's integral up to K. A
threshold outside an expiry's quoted strikes has no probability.

Not a deploy-gate file: the known-session values read the committed 2026-10-16
surface report by name (the surface report keeps every session it wrote).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
SURF = ROOT / "data_store" / "surface"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_density as dd  # noqa: E402
import desk_surface as ds  # noqa: E402

KNOWN = "2026-10-16"
SESSIONS = sorted(p.name for p in SURF.iterdir() if p.is_dir() and (p / "surface.json").exists())


def _inputs(session: str) -> tuple[dict, pd.DataFrame]:
    root = SURF / session
    return json.loads((root / "surface.json").read_text()), pd.read_parquet(root / "residuals.parquet")


def _slices(session: str) -> list[dict]:
    return _inputs(session)[0]["surfaces"]["svi_cal"]["params"]["slices"]


@pytest.fixture(scope="module")
def tails() -> dict[str, list[dict]]:
    return {s: dd.tail_rows(*_inputs(s)) for s in SESSIONS}


def _black_call(k: np.ndarray, sl: dict) -> np.ndarray:
    """Undiscounted Black call / F on the slice's own vol: N(d₊) − eᵏ·N(d₋)."""
    w = np.asarray(ds.svi_w(k, sl))
    sw = np.sqrt(w)
    return norm.cdf(-k / sw + sw / 2) - np.exp(k) * norm.cdf(-k / sw - sw / 2)


def test_every_session_has_certified_slices():
    assert SESSIONS, "no stored surface sessions"
    for s in SESSIONS:
        assert any(x.get("certified") for x in _slices(s)), s


def test_an_uncertified_slice_is_never_a_density_curve():
    rows = [{"expiry": "a", "T_days": 30.0, "k_lo": -0.3, "certified": False},
            {"expiry": "b", "T_days": 35.0, "k_lo": -0.3, "certified": True}]
    assert [p["row"]["expiry"] for p in dd.chart_expiries(rows)] == ["b"]


@pytest.mark.parametrize("session", SESSIONS)
def test_every_certified_slice_is_a_distribution_with_the_forward_as_its_mean(tails, session):
    """Acceptance 1: mass = 1 ± 1e-3 and E[S_T/F] = 1 ± 1e-3 on every certified
    slice; the module's integrals agree with an independent quad over the whole line."""
    rows = {r["expiry"]: r for r in tails[session]}
    for sl in _slices(session):
        if not sl.get("certified"):
            continue
        r = rows[str(sl["expiry"])[:10]]
        assert r["mass"] == pytest.approx(1.0, abs=1e-3), (session, sl["expiry"])
        assert r["mean"] == pytest.approx(1.0, abs=1e-3), (session, sl["expiry"])
        sd = math.sqrt(ds.svi_w(0.0, sl))
        m = sum(quad(dd.density, a, b, args=(sl,), limit=200)[0]
                for a, b in ((-60 * sd - 5, -8 * sd), (-8 * sd, 8 * sd), (8 * sd, 60 * sd + 5)))
        assert m == pytest.approx(r["mass"], abs=1e-6), (session, sl["expiry"])


@pytest.mark.parametrize("session", SESSIONS)
def test_the_density_is_the_second_strike_derivative_of_the_slices_call(session):
    """Breeden–Litzenberger by finite differences: with C(K)/F = c(k), K = F·eᵏ,
    the density of ln(S_T/F) is K·∂²C/∂K² = e⁻ᵏ·(c″(k) − c′(k)) (independent of
    Durrleman's g); compared inside each expiry's quoted strikes. The step is a
    fiftieth of the slice's SVI σ at most: σ is the width of the smile's turn, and
    a fixed 10⁻³ missed the 22.5 peak of a σ = 0.0054 slice (a simulated session)
    by 0.4%."""
    sj, res = _inputs(session)
    kr = ds.quoted_k_range(res)
    for sl in sj["surfaces"]["svi_cal"]["params"]["slices"]:
        h = min(1e-3, float(sl["sigma"]) / 50.0)
        lo, hi = kr[str(sl["expiry"])[:10]]
        ks = np.linspace(lo, hi, 9)
        c0, cp, cm = _black_call(ks, sl), _black_call(ks + h, sl), _black_call(ks - h, sl)
        fd = np.exp(-ks) * ((cp - 2 * c0 + cm) / h ** 2 - (cp - cm) / (2 * h))
        an = np.asarray(dd.density(ks, sl))
        assert np.allclose(an, fd, rtol=2e-3, atol=2e-4 * an.max()), (session, sl["expiry"])


def _put(x: float, sl: dict) -> float:
    return float(_black_call(np.array([x]), sl)[0]) - 1.0 + math.exp(x)


@pytest.mark.parametrize("session", SESSIONS)
def test_prob_below_is_the_put_slope_and_the_densitys_integral(session):
    """P(S_T < K) = ∂P/∂K (finite difference of the put on the slice) = ∫_{−∞}^{k} p."""
    sj, res = _inputs(session)
    kr = ds.quoted_k_range(res)
    h = 1e-5
    for sl in sj["surfaces"]["svi_cal"]["params"]["slices"][::3]:
        lo, hi = kr[str(sl["expiry"])[:10]]
        for k in (lo, 0.5 * lo, 0.0, 0.5 * hi, hi):
            # put / F = c(k) − 1 + eᵏ; ∂P/∂K = e⁻ᵏ·∂(P/F)/∂k
            fd = math.exp(-k) * (_put(k + h, sl) - _put(k - h, sl)) / (2 * h)
            pb = dd.prob_below(k, sl)
            assert pb == pytest.approx(fd, abs=2e-6), (session, sl["expiry"], k)
            sd = math.sqrt(ds.svi_w(0.0, sl))
            integ = quad(dd.density, -60 * sd - 5, k, args=(sl,), limit=400, points=[k - 8 * sd])[0]
            assert pb == pytest.approx(integ, abs=1e-6), (session, sl["expiry"], k)


@pytest.mark.parametrize("session", SESSIONS)
def test_a_threshold_beyond_the_quoted_strikes_has_no_probability(tails, session):
    """Acceptance 2: P(S_T < 0.80 F), P(S_T < 0.90 F), P(S_T > 1.10 F) and
    P(outside 0.95–1.05 F) are None exactly when a threshold lies outside the
    expiry's quoted ln(K/F) range; the masses beyond the quotes are the closed
    form at the range's ends."""
    by = {str(s["expiry"])[:10]: s for s in _slices(session)}
    for r in tails[session]:
        lo, hi = r["k_lo"], r["k_hi"]
        inside = {x: lo <= math.log(x) <= hi for x in (0.80, 0.90, 0.95, 1.05, 1.10)}
        for key, _, side, x in dd.THRESHOLDS:
            if inside[x]:
                want = dd.prob_below(math.log(x), by[r["expiry"]])
                assert r["prob"][key] == pytest.approx(want if side == "below" else 1 - want, abs=1e-12)
            else:
                assert r["prob"][key] is None, (session, r["expiry"], key)
        assert (r["prob"]["out5"] is None) == (not (inside[0.95] and inside[1.05]))
        assert r["below_quoted"] == pytest.approx(dd.prob_below(lo, by[r["expiry"]]), abs=1e-12)
        assert 0.0 < r["below_quoted"] < 1.0
        assert 0.0 < r["above_quoted"] < 1.0


def test_beyond_quoted_strikes_on_a_synthetic_slice():
    sl = {"T": 0.1, "a": 0.002, "b": 0.05, "rho": -0.5, "m": 0.0, "sigma": 0.1, "expiry": "2030-01-01",
          "certified": True, "g_min": 0.1}
    res = pd.DataFrame({"model": ["svi_cal"] * 2, "expiry": ["2030-01-01"] * 2, "k": [-0.05, 0.03]})
    sj = {"surfaces": {"svi_cal": {"params": {"slices": [sl]}}}}
    (r,) = dd.tail_rows(sj, res)
    assert all(v is None for v in r["prob"].values())
    res2 = res.assign(k=[-0.4, 0.2])
    (r2,) = dd.tail_rows(sj, res2)
    assert all(v is not None for v in r2["prob"].values())
    assert r2["prob"]["lt80"] < r2["prob"]["lt90"]


def test_the_known_session_matches_the_values_verified_by_hand(tails):
    """The review's check on 2026-10-16 (B9 detail): 31 d 3.31%; 91 d 8.35% / 4.28%;
    182 d 13.86% / 15.01% (the hand check read rounded parameters: within 0.05)."""
    rows = {r["expiry"]: r for r in tails[KNOWN]}
    m1, m3, m6 = rows["2026-11-16"], rows["2027-01-15"], rows["2027-04-16"]
    assert 100 * m1["prob"]["lt90"] == pytest.approx(3.31, abs=0.01)
    assert m1["prob"]["gt110"] is None                  # 1.10 F lies above the highest quoted strike
    assert 100 * m3["prob"]["lt90"] == pytest.approx(8.35, abs=0.01)
    assert 100 * m3["prob"]["gt110"] == pytest.approx(4.28, abs=0.02)
    assert 100 * m6["prob"]["lt90"] == pytest.approx(13.86, abs=0.05)
    assert 100 * m6["prob"]["gt110"] == pytest.approx(15.01, abs=0.05)
    picks = dd.chart_expiries(tails[KNOWN])
    assert [(p["label"], p["row"]["expiry"]) for p in picks] == [
        ("nearest 30 d", "2026-11-16"), ("nearest 91 d", "2027-01-15"), ("nearest 182 d", "2027-04-16")]


def test_chart_expiries_take_the_nearest_listed_expiry_within_tolerance():
    rows = [{"expiry": e, "T_days": t, "k_lo": -0.3} for e, t in
            (("a", 10.0), ("b", 28.0), ("c", 33.0), ("d", 300.0))]
    picks = dd.chart_expiries(rows)
    # none within 35% of 91 or 182 days
    assert [(p["label"], p["row"]["expiry"]) for p in picks] == [("nearest 30 d", "b")]


def test_the_density_curve_stays_inside_the_quoted_range():
    sl = _slices(KNOWN)[4]
    pts = dd.density_curve(sl, -0.3, 0.05)
    assert pts[0][0] == pytest.approx(-0.3)
    assert pts[-1][0] == pytest.approx(0.05)
    assert all(-0.3 - 1e-12 <= k <= 0.05 + 1e-12 and v >= 0 for k, v in pts)


def test_durrleman_g_matches_the_certificates_g_min_where_it_is_attained():
    """surface.json records each slice's certificate g_min; g on a fine grid over
    [−3, 2] never goes below it (the certificate also takes the |k| → ∞ limit)."""
    for sl in _slices(KNOWN):
        g = np.asarray(dd.durrleman_g(np.linspace(-3, 2, 50001), sl))
        assert g.min() >= sl["g_min"] - 1e-9, sl["expiry"]


def test_fmt_g_keeps_a_tiny_positive_certificate_visible():
    """g_min of 9.5·10⁻⁹ (the 13-day slice of 2026-10-16) printed as '0.0000' would
    read as a slice on the edge of arbitrage; it prints in powers of ten."""
    assert dd.fmt_g(9.517184506e-09) == "9.5 × 10⁻⁹"
    assert dd.fmt_g(3.372e-4) == "3.4 × 10⁻⁴"
    assert dd.fmt_g(9.99e-5) == "1.0 × 10⁻⁴"
    assert dd.fmt_g(0.0071213) == "0.0071"
    assert dd.fmt_g(-2e-5) == "−2.0 × 10⁻⁵"
    assert dd.fmt_g(None) == "—"
