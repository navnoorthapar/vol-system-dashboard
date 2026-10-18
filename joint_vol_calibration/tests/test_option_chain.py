"""
Tests for data/option_chain.py — SPX chain hygiene for the joint calibrator (C22).

The live pipeline fitted Heston to an empty after-hours book for two weeks
(2026-09-08..09-23): Yahoo IV bisection artefacts on the wings, lastPrice
mids, and only the first 8 listed expiries (one usable slice). These tests pin
the replacement: expiry selection by target tenor, two-sided quotes only,
parity forwards, own Black-76 IVs and a fail-closed quote-state gate.

Fixture: tests/fixtures/spx_chain_2026-10-16_subset.csv — two expiries
(2026-10-30, 2027-01-15) of the live ^SPX chain fetched 2026-10-18 08:10 UTC
(Friday 10-16 closing book; SPX close 7659.44), strikes 0.85-1.10 S on a
25-point grid. No network is used.
"""
import math
import pathlib
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.data import option_chain as oc

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "spx_chain_2026-10-16_subset.csv"
S_1016 = 7659.44


def _live():
    return pd.read_csv(FIXTURE)


# ── Expiry selection ──────────────────────────────────────────────────────────

def _daily_expiries(start: date, n_days: int):
    out, d = [], start
    while len(out) < n_days:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d = date.fromordinal(d.toordinal() + 1)
    return out


def test_select_expiries_one_per_target_within_tolerance():
    session = date(2026, 9, 23)
    listed = _daily_expiries(date(2026, 9, 24), 60)          # ~12 weeks of dailies
    listed += ["2026-12-31", "2027-01-15", "2027-03-19", "2027-06-17", "2027-09-17"]
    got = oc.select_calibration_expiries(listed, session)
    assert len(got) == 6 == len(set(got))
    for tgt, e in zip(oc.CAL_EXPIRY_TARGETS_DAYS, got):
        days = (date.fromisoformat(e) - session).days
        assert days >= 7
        assert abs(days - tgt) <= 0.5 * tgt, (tgt, e, days)


def test_select_expiries_never_reuses_an_expiry():
    # One listed expiry near 20d would be "nearest" to both 14d and 30d.
    session = date(2026, 9, 1)
    listed = ["2026-09-21", "2026-11-30"]
    got = oc.select_calibration_expiries(listed, session)
    assert got == ["2026-09-21", "2026-11-30"]
    assert len(got) == len(set(got))


def test_first_eight_listed_expiries_were_one_bucket():
    """The pre-C22 bug: the first 8 dailies from 09-23 give one >=7d bucket."""
    session = date(2026, 9, 23)
    first8 = ["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29",
              "2026-09-30", "2026-10-01", "2026-10-02", "2026-10-05"]
    assert len(oc.select_calibration_expiries(first8, session)) == 1


def test_select_expiries_neighbours_superset():
    session = date(2026, 9, 23)
    listed = _daily_expiries(date(2026, 9, 24), 300)
    base = oc.select_calibration_expiries(listed, session)
    wide = oc.select_calibration_expiries(listed, session, neighbours=1)
    assert set(base) < set(wide) and len(wide) <= 3 * len(base)


# ── NYSE calendar ─────────────────────────────────────────────────────────────

def test_known_holidays_and_sessions():
    assert not oc.is_nyse_session(date(2026, 6, 19))         # Juneteenth (Fri)
    assert not oc.is_nyse_session(date(2026, 4, 3))          # Good Friday 2026
    assert not oc.is_nyse_session(date(2026, 11, 26))        # Thanksgiving
    assert not oc.is_nyse_session(date(2026, 12, 25))
    assert not oc.is_nyse_session(date(2027, 7, 5))          # July 4th observed (Mon)
    assert oc.is_nyse_session(date(2026, 9, 23))
    assert oc.session_on_or_before(date(2026, 6, 20)) == date(2026, 6, 18)
    assert oc.session_on_or_before(date(2026, 5, 31)) == date(2026, 5, 29)


@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 9, 19, 15, 0, tzinfo=timezone.utc), date(2026, 9, 18)),   # Saturday
    (datetime(2026, 9, 18, 23, 30, tzinfo=timezone.utc), date(2026, 9, 18)),  # Fri 19:30 ET
    (datetime(2026, 9, 23, 18, 0, tzinfo=timezone.utc), date(2026, 9, 22)),   # Wed 14:00 ET
    (datetime(2026, 9, 24, 0, 30, tzinfo=timezone.utc), date(2026, 9, 23)),   # after midnight UTC
    (datetime(2026, 6, 20, 1, 0, tzinfo=timezone.utc), date(2026, 6, 18)),    # Juneteenth eve
])
def test_last_closed_session(now, expected):
    assert oc.last_closed_session(now) == expected


def test_sessions_between_skips_weekends_and_holidays():
    assert oc.sessions_between("2026-06-17", "2026-06-22") == 2   # 18, 22 (19 holiday)
    assert oc.sessions_between("2026-09-11", "2026-09-11") == 0


def test_year_fraction_am_settlement_is_earlier():
    pm = oc.year_fraction("2026-10-16", "2026-11-20", "PM")
    am = oc.year_fraction("2026-10-16", "2026-11-20", "AM")
    assert pm == pytest.approx(35 / 365.0)
    assert (pm - am) * 365 * 24 == pytest.approx(6.5)


def test_tbill_discount_to_cc():
    assert oc.tbill_discount_to_cc(0.04) == pytest.approx(0.04075, abs=2e-5)


# ── Black-76 inversion ────────────────────────────────────────────────────────

def test_black76_iv_round_trip_puts_and_calls():
    F, T, D = 7700.0, 0.25, 0.99
    K = np.array([6000., 7000., 7700., 8200., 9000.])
    vol = np.array([0.32, 0.22, 0.16, 0.13, 0.12])
    for is_call in (True, False):
        px = oc.black76_price(F, K, T, D, vol, is_call)
        iv = oc.black76_implied_vol(px, F, K, T, D, np.full(K.shape, is_call))
        assert np.max(np.abs(iv - vol)) < 1e-9


def test_black76_iv_nan_below_intrinsic():
    iv = oc.black76_implied_vol(np.array([5.0]), 7700.0, 7000.0, 0.25, 0.99, np.array([True]))
    assert np.isnan(iv[0])


# ── Chain hygiene on the captured live book ──────────────────────────────────

def test_live_chain_put_call_iv_agree_after_cleaning():
    """Own Black-76 IVs on parity (F, D) agree across put/call at a strike.

    Yahoo's impliedVolatility disagrees by ~1.5-5 vp on the same quotes (its
    forward violates parity); after cleaning the gap is < 0.2 vp.
    """
    cc = oc.clean_option_chain(_live(), "2026-10-16", S_1016, r_ref=0.041)
    assert cc.ok, cc.reason
    q = cc.quotes
    worst_own, worst_yahoo = 0.0, 0.0
    for _, g in q.groupby("expiry"):
        F = g["forward"].iloc[0]
        c = g[g.right == "C"].set_index("strike")
        p = g[g.right == "P"].set_index("strike")
        k = [x for x in c.index.intersection(p.index) if abs(math.log(x / F)) < 0.04]
        assert len(k) >= 5
        worst_own = max(worst_own, float(np.mean(np.abs(c.implied_vol[k] - p.implied_vol[k]))))
        worst_yahoo = max(worst_yahoo, float(np.mean(np.abs(c.yahoo_iv[k] - p.yahoo_iv[k]))))
    assert worst_own < 0.002, worst_own
    assert worst_yahoo > 0.01            # the diagnostic column shows the old defect


def test_live_chain_forwards_and_carry_are_sane():
    cc = oc.clean_option_chain(_live(), "2026-10-16", S_1016, r_ref=0.041)
    carry = cc.carry.set_index("expiry")
    assert set(carry.index) == {"2026-10-30", "2027-01-15"}
    for exp, row in carry.iterrows():
        assert 1.0 < row["forward"] / S_1016 < 1.02
        assert 0.0 < row["r"] < 0.08 and -0.03 < row["q"] < 0.03
    # 2026-10-30 holds stale strikes that violate parity by tens of points;
    # they must be dropped, not averaged into the forward.
    assert cc.diagnostics["n_dropped_parity_inconsistent"] > 0


def test_never_uses_last_price_and_requires_two_sided():
    raw = _live()
    raw.loc[raw.index[:40], "bid"] = 0.0           # one-sided lines
    raw["lastPrice"] = 1e6                          # would wreck any fallback
    cc = oc.clean_option_chain(raw, "2026-10-16", S_1016, r_ref=0.041)
    assert cc.ok
    assert (cc.quotes["bid"] > 0).all()
    assert (cc.quotes["mid_price"] < 1e5).all()
    assert np.allclose(cc.quotes["mid_price"], 0.5 * (cc.quotes["bid"] + cc.quotes["ask"]))


def test_empty_book_fails_closed_with_reason():
    """Emulated after-hours book: bid = ask = 0, Yahoo IV bisection artefacts
    2^-k on the wings and 1e-5 near ATM, lastPrice stale. The old pipeline fit
    this (176-273 vp RMSE); the cleaner must refuse it."""
    raw = _live()
    raw["bid"] = 0.0
    raw["ask"] = 0.0
    m = raw["strike"] / S_1016
    artefacts = np.array([0.0156, 0.0313, 0.0625, 0.125, 0.5])
    raw["impliedVolatility"] = np.where((m < 0.9) | (m > 1.1),
                                        artefacts[np.arange(len(raw)) % 5], 1e-5)
    cc = oc.clean_option_chain(raw, "2026-10-16", S_1016, r_ref=0.041)
    assert not cc.ok
    assert "no live two-sided quotes" in cc.reason
    assert cc.quotes.empty


def test_spread_cap_applies():
    raw = _live()
    i = raw.index[(raw.strike == 7650) & (raw.expiration == "2027-01-15")
                  & (raw.option_type == "call")][0]
    raw.loc[i, "ask"] = raw.loc[i, "bid"] * 3          # 100% relative spread
    cc = oc.clean_option_chain(raw, "2026-10-16", S_1016, r_ref=0.041)
    hit = cc.quotes[(cc.quotes.strike == 7650) & (cc.quotes.expiry == "2027-01-15")
                    & (cc.quotes.right == "C")]
    assert hit.empty


# ── Synthetic chain: exact recovery of F_T, D_T ──────────────────────────────

def _synthetic_chain(F, D, T_days, session="2026-09-23", root="SPXW", spread=0.2):
    K = np.arange(round(F * 0.85 / 5) * 5, F * 1.15, 5.0)
    T = T_days / 365.0
    vol = 0.18 - 0.4 * np.log(K / F)                       # skewed smile
    expiry = date.fromordinal(date.fromisoformat(session).toordinal() + T_days).isoformat()
    rows = []
    for right in ("C", "P"):
        px = oc.black76_price(F, K, T, D, vol, right == "C")
        for k, p in zip(K, px):
            yymmdd = expiry[2:4] + expiry[5:7] + expiry[8:10]
            rows.append({"contractSymbol": f"{root}{yymmdd}{right}{int(k * 1000):08d}",
                         "strike": k, "bid": max(p - spread / 2, 0.0), "ask": p + spread / 2,
                         "lastPrice": 0.01, "impliedVolatility": 0.5,
                         "expiration": expiry, "option_type": "call" if right == "C" else "put"})
    return pd.DataFrame(rows)


def test_parity_regression_recovers_forward_and_discount():
    S = 7700.0
    r, q, T_days = 0.042, 0.011, 180
    T = T_days / 365.0
    F, D = S * math.exp((r - q) * T), math.exp(-r * T)
    cc = oc.clean_option_chain(_synthetic_chain(F, D, T_days), "2026-09-23", S, r_ref=0.04)
    row = cc.carry.iloc[0]
    assert row["method"] == "parity_regression"
    assert row["forward"] == pytest.approx(F, rel=1e-6)
    assert row["discount"] == pytest.approx(D, rel=1e-7)
    assert row["q"] == pytest.approx(q, abs=1e-4)
    assert row["r"] == pytest.approx(r, abs=1e-4)


def test_am_lines_dropped_when_pm_lines_exist_and_am_only_gets_am_time():
    S = 7700.0
    F, D = 7720.0, 0.995
    pm = _synthetic_chain(F, D, 30, root="SPXW")
    am = _synthetic_chain(F, D, 30, root="SPX")
    mixed = pd.concat([pm.iloc[::2], am.iloc[1::2]], ignore_index=True)
    cc = oc.clean_option_chain(mixed, "2026-09-23", S, r_ref=0.04)
    assert set(cc.quotes["root"]) == {"SPXW"}
    assert cc.quotes["time_to_expiry"].iloc[0] == pytest.approx(30 / 365.0)
    only_am = oc.clean_option_chain(am, "2026-09-23", S, r_ref=0.04)
    assert set(only_am.quotes["settlement"]) == {"AM"}
    assert only_am.quotes["time_to_expiry"].iloc[0] == pytest.approx((30 - 6.5 / 24) / 365.0)


def test_fetch_downloads_only_selected_expiries():
    class _Chain:
        def __init__(self, exp):
            df = _synthetic_chain(7720.0, 0.995, 30)
            self.calls = df[df.option_type == "call"].drop(columns=["expiration", "option_type"])
            self.puts = df[df.option_type == "put"].drop(columns=["expiration", "option_type"])

    asked = []

    class _Ticker:
        options = tuple(_daily_expiries(date(2026, 9, 24), 300))

        def __init__(self, sym):
            pass

        def option_chain(self, exp):
            asked.append(exp)
            return _Chain(exp)

    raw, meta = oc.fetch_spx_chain("2026-09-23", ticker_factory=_Ticker, neighbours=0)
    assert asked == meta["requested"]
    assert len(asked) == 6
    assert set(raw["expiration"]) == set(asked)
