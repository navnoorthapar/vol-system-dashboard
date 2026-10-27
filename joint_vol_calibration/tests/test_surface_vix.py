"""
Cboe VIX methodology (surface/vix.py): the methodology's own sample
calculation as an exact fixture, the rule-level behaviour (zero-bid stop,
K0, dK at the ends, bracket and candidate set), model-free consistency on a
synthetic Black-Scholes chain, and the stored-session replication.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.surface import essvi as es
from joint_vol_calibration.surface import vix

FIXTURE = Path(__file__).parent / "fixtures" / "surface_cboe_vix_sample.csv"
# the sample's calculation time, naive New York wall-clock time as vix.py takes it
CALC = datetime(2022, 9, 27, 10, 45, 15)  # noqa: DTZ001
R1, R2 = 0.031664e-2, 0.028797e-2                  # published R1, R2


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(FIXTURE, comment="#")


def _term(sample, term, M, R, **kw):
    g = sample[sample["term"] == term]
    return vix.single_term_variance(g["strike"], g["call_bid"], g["call_ask"], g["put_bid"],
                                    g["put_ask"], M / vix.MINUTES_PER_YEAR, R, **kw)


def test_sample_minutes_match_the_methodology():
    # SPX AM 2022-10-21 09:30 and SPXW PM 2022-10-28 16:00 from 10:45:15
    assert vix.minutes_to_settlement(CALC, "2022-10-21", "AM") == 34484
    assert vix.minutes_to_settlement(CALC, "2022-10-28", "PM") == 44954


def test_sample_forwards_k0_and_variances(sample):
    near = _term(sample, "near", 34484, R1)
    nxt = _term(sample, "next", 44954, R2)
    assert near["atm_strike"] == 1965
    assert nxt["atm_strike"] == 1960
    assert near["F"] == pytest.approx(1962.89996, abs=5e-6)
    assert nxt["F"] == pytest.approx(1962.40006, abs=5e-6)
    assert near["K0"] == nxt["K0"] == 1960
    assert near["sum_contributions"] == pytest.approx(0.0006320516, abs=5e-11)
    assert nxt["sum_contributions"] == pytest.approx(0.0008314016, abs=5e-11)
    assert near["sigma2"] == pytest.approx(0.019233906, abs=5e-10)
    assert nxt["sigma2"] == pytest.approx(0.019423884, abs=5e-10)
    level = vix.constant_maturity_index(near["sigma2"], 34484, nxt["sigma2"], 44954, 30 * 1440)
    assert level == pytest.approx(13.927842, abs=5e-7)          # "VIX = 100 x 0.13927842"
    assert round(level, 2) == 13.93


def test_sample_individual_contributions_and_delta_k(sample):
    """Appendix 5 rows: dK at the strip ends, after excluded strikes, at K0."""
    near = _term(sample, "near", 34484, R1)
    nxt = _term(sample, "next", 44954, R2)
    cn = dict(zip(near["strikes"], zip(near["dK"], near["contributions"])))
    cx = dict(zip(nxt["strikes"], zip(nxt["dK"], nxt["contributions"])))
    expected_near = {1370: (5, 0.0000005328), 1375: (5, 0.0000003306), 1400: (7.5, 0.0000004783),
                     1960: (5, 0.0000296432), 2100: (15, 0.0000003401), 2125: (25, 0.0000005536)}
    expected_next = {1275: (50, 0.0000023069), 1325: (37.5, 0.0000032041),
                     1960: (5, 0.0000339711), 2200: (50, 0.0000007748)}
    for K, (dk, c) in expected_near.items():
        assert cn[K][0] == dk, K
        assert cn[K][1] == pytest.approx(c, abs=5e-11), K
    for K, (dk, c) in expected_next.items():
        assert cx[K][0] == dk, K
        assert cx[K][1] == pytest.approx(c, abs=5e-11), K


def test_two_consecutive_zero_bids_stop_the_walk(sample):
    """Methodology's own examples: puts 1350/1355 and the 2225 call have bids
    but lie beyond two consecutive zero bids, so they are excluded; isolated
    zero bids (1405 put, 2120 call) are skipped without stopping."""
    near = _term(sample, "near", 34484, R1)
    ks = set(near["strikes"])
    assert near["lowest_strike"] == 1370
    assert near["highest_strike"] == 2125
    assert not ks & {1350, 1355, 1360, 1365, 2225, 2120, 1405}
    assert {1370, 1400, 1410, 2100, 2125} <= ks
    nxt = _term(sample, "next", 44954, R2)
    assert nxt["lowest_strike"] == 1275
    assert 1300 not in set(nxt["strikes"])


def test_k0_without_both_sides_fails_closed_or_falls_back_flagged(sample):
    g = sample[sample["term"] == "near"].copy()
    g.loc[g["strike"] == 1960, ["call_bid", "call_ask"]] = np.nan
    args = (g["strike"], g["call_bid"], g["call_ask"], g["put_bid"], g["put_ask"],
            34484 / vix.MINUTES_PER_YEAR, R1)
    strict = vix.single_term_variance(*args, k0_fallback=False)
    assert not strict["ok"]
    assert "K0 = 1960" in strict["reason"]
    fb = vix.single_term_variance(*args)
    assert fb["ok"]
    assert fb["k0_fallback"]
    assert fb["K0"] == 1955
    assert fb["literal_K0"] == 1960


def test_given_forward_replaces_the_single_strike_rule(sample):
    a = _term(sample, "near", 34484, R1)
    b = _term(sample, "near", 34484, R1, forward=a["F"])
    assert b["forward_method"] == "given"
    assert b["sigma2"] == pytest.approx(a["sigma2"], rel=1e-14)


def _bs_chain(F, T, vol, strikes, D=1.0):
    K = np.asarray(strikes, dtype=float)
    c = oc.black76_price(F, K, T, D, vol, True)
    p = oc.black76_price(F, K, T, D, vol, False)
    return K, c, p


def test_flat_vol_chain_recovers_the_variance():
    """Model-free check: a dense, wide Black-Scholes strip at 20% vol gives
    sigma^2 = 0.04 up to the strike discretisation."""
    F, T = 5000.0, 30 / 365
    K, c, p = _bs_chain(F, T, 0.20, np.arange(2000, 9001, 5.0))
    out = vix.single_term_variance(K, c, c, p, p, T, 0.0, forward=F)
    assert np.sqrt(out["sigma2"]) == pytest.approx(0.20, abs=2e-4)


def test_truncated_wing_biases_the_strip_down():
    F, T = 5000.0, 30 / 365
    K, c, p = _bs_chain(F, T, 0.20, np.arange(4500, 5501, 5.0))
    out = vix.single_term_variance(K, c, c, p, p, T, 0.0, forward=F)
    assert np.sqrt(out["sigma2"]) < 0.199                      # strikes beyond +-10% missing


def test_end_of_week_candidates():
    assert vix.is_end_of_week_expiry("2026-10-16")              # Friday
    assert vix.is_end_of_week_expiry("2026-04-02")              # Thursday before Good Friday
    assert not vix.is_end_of_week_expiry("2026-04-01")          # Wednesday
    assert not vix.is_end_of_week_expiry("2026-11-16")          # Monday


def test_bracket_picks_last_term_within_and_first_after():
    terms = [{"minutes": m, "candidate": True, "expiry": str(m)} for m in (20000, 40000, 45000, 60000)]
    near, nxt = vix.bracket(terms, 43200)
    assert near["minutes"] == 40000
    assert nxt["minutes"] == 45000
    near, nxt = vix.bracket(terms, 10000)                       # nothing <= target: nearest + next
    assert near["minutes"] == 20000
    assert nxt["minutes"] == 40000


def test_constant_maturity_interpolates_total_variance():
    # equal sigma^2 on both terms -> the index is that vol
    assert vix.constant_maturity_index(0.04, 30000, 0.04, 50000, 43200) == pytest.approx(20.0, rel=1e-12)


def test_surface_variance_swap_of_flat_surface_is_its_variance():
    T = 0.25
    s = es.ESSVISurface(T=[T], theta=[0.04 * T], psi=[1e-8], rho=[0.0])
    vs = vix.surface_variance_swap(s, T)
    assert vs == pytest.approx(0.04, rel=1e-9)


def test_stored_session_replication_brackets_and_flags():
    """2026-10-16 stored chain: the Cboe next-term candidate (2026-11-20) is not
    stored, so the bracket widens to 2026-11-13 / 2026-12-04 and the note says so."""
    from joint_vol_calibration.surface import quotes as sqm
    sq = sqm.load_session_quotes("2026-10-16")
    calc = datetime.fromisoformat(sq.clean_diagnostics["t_origin"]).astimezone(oc.NY_TZ).replace(tzinfo=None)
    terms = vix.term_tables(sq.raw, "2026-10-16", calc)
    assert {t["expiry"] for t in terms if t["candidate"]} >= {"2026-10-30", "2026-11-13", "2026-12-04"}
    assert "2026-11-20" not in {t["expiry"] for t in terms}
    rep = vix.replicate_index(terms, sq.r_ref, 30, spot=sq.spot)
    assert rep["ok"]
    assert rep["near"]["expiry"] == "2026-11-13"
    assert rep["next"]["expiry"] == "2026-12-04"
    assert 0.0 < rep["near_weight"] < 1.0
    off = vix.official_closes("2026-10-16")
    assert off["VIX"]["close"] == pytest.approx(19.99)
    assert off["VIX"]["source"] == "cboe"
    assert abs(rep["level"] - off["VIX"]["close"]) < 1.0


class _FlatSurface:
    def __init__(self, vol):
        self.vol = vol

    def w(self, k, T):
        return np.full(np.shape(k), self.vol ** 2 * float(T))

    def iv(self, k, T):
        return np.full(np.shape(k), self.vol)


def _flat_terms(vol, F, kf_lo, kf_hi, minutes=(28 * 1440, 35 * 1440), step=5.0):
    """Two candidate terms of a Black-Scholes chain (R = 0) quoted only on
    K/F in [kf_lo, kf_hi], bid = ask = model price."""
    terms = []
    for i, M in enumerate(minutes):
        T = M / vix.MINUTES_PER_YEAR
        K, c, p = _bs_chain(F, T, vol, np.arange(np.ceil(kf_lo * F / step) * step, kf_hi * F + 1e-9, step))
        tab = pd.DataFrame({"bid_c": c, "ask_c": c, "bid_p": p, "ask_p": p}, index=K)
        terms.append({"expiry": f"2026-11-{13 + 7 * i:02d}", "settlement": "PM", "minutes": int(M),
                      "T": T, "has_root": True, "candidate": True, "table": tab})
    return terms


def _bs_vs_outside(vol, T, kf_lo, kf_hi):
    """Variance-swap variance carried by OTM options outside K/F [lo, hi]
    (Black-Scholes, F = 1), by adaptive quadrature: independent of vix.py."""
    from scipy.integrate import quad
    from scipy.stats import norm
    sd = vol * np.sqrt(T)

    def otm(k):
        d1 = -k / sd + sd / 2
        if k >= 0:
            return norm.cdf(d1) - np.exp(k) * norm.cdf(d1 - sd)
        return np.exp(k) * norm.cdf(-(d1 - sd)) - norm.cdf(-d1)
    f = lambda k: otm(k) * np.exp(-k)                                   # noqa: E731
    lo = quad(f, -3.0, np.log(kf_lo), limit=200, epsabs=1e-14)[0]
    hi = quad(f, np.log(kf_hi), 2.0, limit=200, epsabs=1e-14)[0]
    return 2.0 * (lo + hi) / T


def test_gap_attribution_on_a_truncated_flat_vol_chain():
    """Market = surface at every stored strike, so the far-wing and in-range
    parts vanish; the full-range model index is the flat vol; the truncation
    matches an independent quadrature of the variance beyond K/F 0.85-1.15,
    and residual + gap + truncation close."""
    vol, F = 0.20, 5000.0
    terms = _flat_terms(vol, F, 0.85, 1.15)
    rep = vix.replicate_index(terms, 0.0, 30)
    assert rep["ok"]
    a = vix.gap_attribution(rep, terms, 0.0, _FlatSurface(vol), [t["expiry"] for t in terms],
                            official=20.0)
    assert a["available"]
    assert a["far_wing_market_minus_model_pts"] == pytest.approx(0.0, abs=1e-9)
    assert a["fit_inside_range_pts"] == pytest.approx(0.0, abs=1e-9)
    assert a["levels"]["L3_model_continuous_k_-3_to_2"] == pytest.approx(20.0, abs=1e-3)
    s_strip = [vol ** 2 - _bs_vs_outside(vol, t["T"], 0.85, 1.15) for t in terms]
    expect = 20.0 - vix.constant_maturity_index(s_strip[0], terms[0]["minutes"], s_strip[1],
                                                terms[1]["minutes"], 30 * 1440)
    assert expect > 0.01
    assert a["truncation_pts"] == pytest.approx(expect, rel=2e-3, abs=2e-4)
    assert a["gap_pts"] == pytest.approx(rep["level"] - 20.0)
    assert a["residual_pts"] + a["gap_pts"] + a["truncation_pts"] == pytest.approx(0.0, abs=1e-12)
    assert abs(a["discretisation_pts"]) < 0.02                 # 5-point strikes


def test_gap_attribution_sees_a_rich_far_wing():
    """Puts beyond the fitted K/F 0.70 priced at 60% vol against a 20%
    surface: L1 (surface prices beyond the fitted range) is exactly the
    replication of the unmodified 20% chain, and the excess is far-wing
    richness, not fit error."""
    vol, F = 0.20, 5000.0
    flat = _flat_terms(vol, F, 0.55, 1.20)
    terms = _flat_terms(vol, F, 0.55, 1.20)
    for t in terms:
        tab = t["table"]
        far = tab.index.to_numpy() < 0.70 * F
        _, _, p_rich = _bs_chain(F, t["T"], 0.60, tab.index.to_numpy()[far])
        tab.loc[far, ["bid_p", "ask_p"]] = np.column_stack([p_rich, p_rich])
    rep = vix.replicate_index(terms, 0.0, 30)
    a = vix.gap_attribution(rep, terms, 0.0, _FlatSurface(vol), [t["expiry"] for t in terms],
                            official=None)
    base = vix.replicate_index(flat, 0.0, 30)["level"]
    assert a["levels"]["L1_model_outside_fit_range"] == pytest.approx(base, abs=1e-9)
    assert a["far_wing_market_minus_model_pts"] == pytest.approx(rep["level"] - base, abs=1e-9)
    assert a["far_wing_market_minus_model_pts"] > 0.1
    assert a["fit_inside_range_pts"] == pytest.approx(0.0, abs=1e-9)
    assert "residual_pts" not in a


def test_gap_attribution_needs_fitted_terms():
    terms = _flat_terms(0.2, 5000.0, 0.8, 1.2)
    rep = vix.replicate_index(terms, 0.0, 30)
    a = vix.gap_attribution(rep, terms, 0.0, _FlatSurface(0.2), {}, official=20.0)
    assert not a["available"]
    assert "not fitted" in a["reason"]


# ── C23f: AM-date SPXW proxies (review QSM-4) and extrapolated brackets (QSM-3) ─

def test_spx_am_settlement_dates():
    assert vix.is_spx_am_settlement_date("2026-10-16")          # third Friday
    assert vix.is_spx_am_settlement_date("2026-12-18")
    assert vix.is_spx_am_settlement_date("2027-01-15")
    assert not vix.is_spx_am_settlement_date("2026-12-17")
    assert not vix.is_spx_am_settlement_date("2026-12-11")      # second Friday
    assert vix.is_spx_am_settlement_date("2026-06-18")          # Juneteenth Friday: Thursday
    assert not vix.is_spx_am_settlement_date("2026-06-19")
    assert vix.is_spx_am_settlement_date("2027-06-17")          # 2027-06-18 observed holiday


def _lines(expiry, root, strikes=(95.0, 100.0, 105.0)):
    return [{"expiry": expiry, "strike": K, "right": right, "bid": 1.0, "ask": 1.2, "root": root}
            for K in strikes for right in ("C", "P")]


def test_term_tables_flag_spxw_lines_standing_in_for_the_am_monthly():
    """2026-10-16's chain stores 2026-12-18 and 2027-01-15 (third Fridays) as
    SPXW only. They stay PM-timed candidates, flagged am_date_pm_proxy; an AM
    date with SPX lines is AM; a root-less chain is not flagged (its own
    deviation line covers it)."""
    calc = datetime(2026, 10, 16, 16, 0)  # noqa: DTZ001
    raw = pd.DataFrame(_lines("2026-12-18", "SPXW") + _lines("2027-01-15", "SPXW")
                       + _lines("2027-04-16", "SPX") + _lines("2027-04-16", "SPXW")
                       + _lines("2026-12-04", "SPXW"))
    by = {t["expiry"]: t for t in vix.term_tables(raw, "2026-10-16", calc)}
    assert by["2026-12-18"]["settlement"] == "PM"
    assert by["2026-12-18"]["am_date_pm_proxy"]
    assert by["2027-01-15"]["am_date_pm_proxy"]
    assert by["2027-01-15"]["candidate"]
    assert by["2027-04-16"]["settlement"] == "AM"
    assert not by["2027-04-16"]["am_date_pm_proxy"]
    assert not by["2026-12-04"]["am_date_pm_proxy"]
    rootless = pd.DataFrame(_lines("2026-12-18", ""))
    t = vix.term_tables(rootless, "2026-10-16", calc)[0]
    assert t["settlement"] == "PM"
    assert not t["am_date_pm_proxy"]
    assert not t["has_root"]


def _rep(ok=True, w=0.5, days=9):
    near = {"expiry": "2026-10-30", "days": 14.0}
    nxt = {"expiry": "2026-11-13", "days": 28.0}
    return {"ok": ok, "level": 20.0, "near_weight": w, "extrapolated": not 0.0 <= w <= 1.0,
            "constant_maturity_days": days, "near": near, "next": nxt, "reason": "x"}


def test_headline_method_refuses_an_extrapolated_bracket():
    assert vix.headline_method({"cboe_rules": _rep(), "parity_forward": _rep()}) == ("cboe_rules", None)
    pf = vix.headline_method({"cboe_rules": _rep(ok=False), "parity_forward": _rep()})
    assert pf[0] == "parity_forward"
    m, why = vix.headline_method({"cboe_rules": _rep(w=1.357), "parity_forward": _rep(w=1.357)})
    assert m is None
    assert "2026-10-30 (14.0d) / 2026-11-13 (28.0d)" in why
    assert "1.357" in why
    assert vix.headline_method({"cboe_rules": _rep(ok=False), "parity_forward": _rep(ok=False)})[0] is None
