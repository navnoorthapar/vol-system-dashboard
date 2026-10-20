"""
test_properties_iv.py: property-based tests (Hypothesis) of the implied-vol
solvers and of option_chain's time-to-expiry clock.

implied_vol.implied_vol_black (safeguarded Newton, used by the Greeks monitor
and the NN surrogate) and option_chain.black76_implied_vol (bisection, used by
the chain cleaner and the calibrator) are checked over T from one day to three
years, total vol from 0.3% to 300% and both sides of the forward:

  round trip     OTM, price >= 1e-10 DF F: status ok and sigma recovered to
                 1e-9 relative (measured worst 2.4e-12 over 1e5 random inputs);
                 ITM: either status ok within the documented resolution bound
                 (1e-3 relative; measured worst 2.5e-5), or an explicit
                 non-ok status, never a wrong number reported as ok
  monotonicity   a higher price never gives a lower implied vol
  call/put       the call and the parity-equivalent put give the same vol
  agreement      inside its documented domain (sigma in (1e-4, 5), time value
                 above 1e-10 DF F) the bisection inverter is finite and matches
                 the Newton solver (OTM to 1e-9 relative)

and for option_chain.year_fraction (minutes / 525,600, Cboe convention):

  an origin that holds no time (None, '' or NaN, as a missing quote_time
  cell reads) and the session's own 16:00 ET close give the same T as the
  whole-day count; AM settlement is exactly 390 minutes earlier than PM;
  T is strictly increasing in the expiry date.

Example budgets come from the Hypothesis profile set up in
test_properties_pricing (HYPOTHESIS_PROFILE=ci|dev|thorough).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, event, given, note, settings
from hypothesis import strategies as st

from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.models.implied_vol import (
    IV_OK,
    IV_STATUS,
    black_price,
    implied_vol_black,
)
from joint_vol_calibration.tests.test_properties_pricing import budget, log_uniform

pytestmark = pytest.mark.property

OTM_RTOL = 1e-9          # relative sigma error for a well-posed OTM price
ITM_RTOL = 1e-3          # documented resolution bound once parity is involved


@st.composite
def quotes(draw, side: str = "any"):
    """(F, K, T, sigma, is_call, DF): K within +-3 total standard deviations."""
    F = draw(log_uniform(1.0, 1e4))
    T = draw(log_uniform(1.0 / 365.0, 3.0))
    sigma = draw(log_uniform(0.01, 3.0))
    z = draw(st.floats(-3.0, 3.0))
    K = F * math.exp(z * sigma * math.sqrt(T))
    DF = math.exp(-draw(st.floats(-0.01, 0.08)) * T)
    if side == "otm":
        is_call = K >= F
    elif side == "itm":
        is_call = K < F
    else:
        is_call = draw(st.booleans())
    return F, K, T, sigma, is_call, DF


@settings(max_examples=budget(4.0))
@given(q=quotes("otm"))
def test_otm_round_trip_recovers_sigma(q):
    F, K, T, sigma, is_call, DF = q
    p = float(black_price(F, K, T, sigma, is_call, DF))
    assume(p >= 1e-10 * DF * F)                     # the solver's well-posed domain
    iv, status = implied_vol_black(p, F, K, T, is_call, DF, full_output=True)
    note(f"p={p!r} iv={iv!r} status={IV_STATUS[int(status)]}")
    assert int(status) == IV_OK
    assert abs(float(iv) - sigma) <= OTM_RTOL * sigma


@settings(max_examples=budget(4.0))
@given(q=quotes("itm"))
def test_itm_inversion_is_within_its_resolution_bound_or_says_why_not(q):
    """An ITM price carries the forward in its digits: the time value left
    after parity may not determine sigma. The solver must then return a
    non-ok status (NaN), not a wrong vol with status ok."""
    F, K, T, sigma, is_call, DF = q
    p = float(black_price(F, K, T, sigma, is_call, DF))
    iv, status = implied_vol_black(p, F, K, T, is_call, DF, full_output=True)
    note(f"p={p!r} iv={iv!r} status={IV_STATUS[int(status)]}")
    event(f"status {IV_STATUS[int(status)]}")
    if int(status) == IV_OK:
        assert abs(float(iv) - sigma) <= ITM_RTOL * sigma
    else:
        assert math.isnan(float(iv))


@settings(max_examples=budget(2.0))
@given(q=quotes("any"), bump=st.floats(1e-6, 0.5))
def test_a_higher_price_never_gives_a_lower_vol(q, bump):
    F, K, T, sigma, is_call, DF = q
    s2 = sigma * (1.0 + bump)
    p1, p2 = (float(black_price(F, K, T, s, is_call, DF)) for s in (sigma, s2))
    assume(p2 > p1)
    (v1, st1), (v2, st2) = (implied_vol_black(p, F, K, T, is_call, DF, full_output=True)
                            for p in (p1, p2))
    assume(int(st1) == IV_OK and int(st2) == IV_OK)
    assert float(v2) >= float(v1)


@settings(max_examples=budget(2.0))
@given(q=quotes("any"))
def test_call_and_parity_put_give_the_same_vol(q):
    F, K, T, sigma, _, DF = q
    c = float(black_price(F, K, T, sigma, True, DF))
    p = c - DF * (F - K)                             # put by parity
    (vc, sc), (vp, sp) = (implied_vol_black(x, F, K, T, flag, DF, full_output=True)
                          for x, flag in ((c, True), (p, False)))
    assume(int(sc) == IV_OK and int(sp) == IV_OK)
    tol = OTM_RTOL if abs(math.log(K / F)) < 1e-12 else ITM_RTOL
    assert abs(float(vc) - float(vp)) <= tol * sigma


@settings(max_examples=budget(4.0))
@given(q=quotes("any"))
def test_chain_bisection_inverter_matches_the_newton_solver_in_its_domain(q):
    F, K, T, sigma, is_call, DF = q
    assume(1e-4 * 1.0001 < sigma < 5.0)
    p = float(black_price(F, K, T, sigma, is_call, DF))
    otm_call = K >= F
    tv = p if is_call == otm_call else (p - DF * (F - K) if is_call else p + DF * (F - K))
    assume(tv > 1e-10 * DF * F)
    bis = float(oc.black76_implied_vol(p, F, K, T, DF, is_call)[()])
    newton, status = implied_vol_black(p, F, K, T, is_call, DF, full_output=True)
    note(f"tv/DF F={tv / (DF * F)!r} bisection={bis!r} newton={newton!r}")
    assert math.isfinite(bis)
    if int(status) == IV_OK:
        tol = OTM_RTOL if is_call == otm_call else ITM_RTOL
        assert abs(bis - float(newton)) <= tol * sigma


# ── option_chain.year_fraction ───────────────────────────────────────────────

sessions = st.dates(min_value=date(2005, 1, 3), max_value=date(2035, 12, 28)).filter(oc.is_nyse_session)


@settings(max_examples=budget(2.0))
@given(session=sessions, days=st.integers(0, 800),
       missing=st.sampled_from([None, "", float("nan")]), settlement=st.sampled_from(["PM", "AM"]))
def test_year_fraction_clock(session, days, missing, settlement):
    expiry = session + timedelta(days=days)
    whole_days = oc.year_fraction(session, expiry, settlement)
    # an origin with no time in it is the session close (it used to raise)
    assert oc.year_fraction(session, expiry, settlement, origin=missing) == whole_days
    # the session's own 16:00 ET close as an explicit origin: same T (New York
    # wall-clock minutes, so a DST change in between does not count)
    explicit = oc.year_fraction(session, expiry, settlement, origin=oc.session_close(session))
    assert abs(explicit - whole_days) <= 1e-15
    # AM (SOQ, 09:30 ET) settles 390 minutes before PM (16:00 ET)
    pm = oc.year_fraction(session, expiry, "PM")
    am = oc.year_fraction(session, expiry, "AM")
    assert abs((pm - am) * 525_600.0 - 390.0) <= 1e-9
    # strictly increasing in the expiry date
    assert oc.year_fraction(session, expiry + timedelta(days=1), settlement) > whole_days
