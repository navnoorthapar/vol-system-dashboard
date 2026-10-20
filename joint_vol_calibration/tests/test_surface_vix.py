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
CALC = datetime(2022, 9, 27, 10, 45, 15)            # the sample's calculation time (ET)
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
    assert near["atm_strike"] == 1965 and nxt["atm_strike"] == 1960
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
        assert cn[K][0] == dk and cn[K][1] == pytest.approx(c, abs=5e-11), K
    for K, (dk, c) in expected_next.items():
        assert cx[K][0] == dk and cx[K][1] == pytest.approx(c, abs=5e-11), K


def test_two_consecutive_zero_bids_stop_the_walk(sample):
    """Methodology's own examples: puts 1350/1355 and the 2225 call have bids
    but lie beyond two consecutive zero bids, so they are excluded; isolated
    zero bids (1405 put, 2120 call) are skipped without stopping."""
    near = _term(sample, "near", 34484, R1)
    ks = set(near["strikes"])
    assert near["lowest_strike"] == 1370 and near["highest_strike"] == 2125
    assert not ks & {1350, 1355, 1360, 1365, 2225, 2120, 1405}
    assert {1370, 1400, 1410, 2100, 2125} <= ks
    nxt = _term(sample, "next", 44954, R2)
    assert nxt["lowest_strike"] == 1275 and 1300 not in set(nxt["strikes"])   # isolated zero bid


def test_k0_without_both_sides_fails_closed_or_falls_back_flagged(sample):
    g = sample[sample["term"] == "near"].copy()
    g.loc[g["strike"] == 1960, ["call_bid", "call_ask"]] = np.nan
    args = (g["strike"], g["call_bid"], g["call_ask"], g["put_bid"], g["put_ask"],
            34484 / vix.MINUTES_PER_YEAR, R1)
    strict = vix.single_term_variance(*args, k0_fallback=False)
    assert not strict["ok"] and "K0 = 1960" in strict["reason"]
    fb = vix.single_term_variance(*args)
    assert fb["ok"] and fb["k0_fallback"] and fb["K0"] == 1955 and fb["literal_K0"] == 1960


def test_given_forward_replaces_the_single_strike_rule(sample):
    a = _term(sample, "near", 34484, R1)
    b = _term(sample, "near", 34484, R1, forward=a["F"])
    assert b["forward_method"] == "given" and b["sigma2"] == pytest.approx(a["sigma2"], rel=1e-14)


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
    assert near["minutes"] == 40000 and nxt["minutes"] == 45000
    near, nxt = vix.bracket(terms, 10000)                       # nothing <= target: nearest + next
    assert near["minutes"] == 20000 and nxt["minutes"] == 40000


def test_constant_maturity_interpolates_total_variance():
    # equal sigma^2 on both terms -> the index is that vol
    assert vix.constant_maturity_index(0.04, 30000, 0.04, 50000, 43200) == pytest.approx(20.0, rel=1e-12)


def test_surface_variance_swap_of_flat_surface_is_its_variance():
    T = 0.25
    s = es.ESSVISurface(T=[T], theta=[0.04 * T], psi=[1e-8], rho=[0.0])
    vs = vix.surface_variance_swap(s, T)
    assert vs == pytest.approx(0.04, rel=2e-5)


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
    assert rep["ok"] and rep["near"]["expiry"] == "2026-11-13" and rep["next"]["expiry"] == "2026-12-04"
    assert 0.0 < rep["near_weight"] < 1.0
    off = vix.official_closes("2026-10-16")
    assert off["VIX"]["close"] == pytest.approx(19.99) and off["VIX"]["source"] == "cboe"
    assert abs(rep["level"] - off["VIX"]["close"]) < 1.0
