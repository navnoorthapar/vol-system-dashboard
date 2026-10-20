"""
Static-arbitrage screen and Cohen-Reisinger-Wang l1-BA repair
(surface/arbitrage.py) on synthetic chains with known arbitrage, plus the
committed 2026-10-16 chain.
"""
import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.surface import arbitrage as arb


STRIKES = np.arange(4000.0, 6001.0, 50.0)


def _chain(vol_fn, expiries=((30, 5000.0, 0.997), (60, 5010.0, 0.994), (120, 5030.0, 0.988)),
           strikes=STRIKES, half=0.25):
    """OTM quotes priced by Black-76 at vol_fn(k, T); bid/ask = mid -+ half."""
    rows = []
    for days, F, D in expiries:
        T = days / 365.0
        for K in strikes:
            right = "C" if K >= F else "P"
            k = np.log(K / F)
            mid = float(oc.black76_price(F, K, T, D, vol_fn(k, T), right == "C"))
            rows.append({"expiry": f"E{days:03d}", "T": T, "strike": K, "right": right,
                         "forward": F, "discount": D, "mid": mid,
                         "bid": max(mid - half, 0.0), "ask": mid + half})
    return pd.DataFrame(rows)


def _skew(k, T):
    return 0.18 - 0.25 * k + 0.4 * k * k


def test_arbitrage_free_chain_has_no_violation_and_no_repair():
    q = _chain(_skew)
    rep = arb.screen(q)
    assert sum(rep.mid_violations.values()) == 0
    assert sum(rep.mid_violations_gt_tick.values()) == 0
    assert sum(rep.executable_violations.values()) == 0
    assert rep.repair["status"] == 0
    assert rep.repair["n_perturbed_outside_spread"] == 0
    assert rep.repair["max_abs_dprice"] < 1e-6


def test_put_and_call_quotes_normalise_to_the_same_call_price():
    """C = P + D (F - K): the same strike quoted as a put or as a call gives the
    same normalised call-equivalent price."""
    q = _chain(_skew).iloc[:5].copy()
    F, D, T = q["forward"].to_numpy(), q["discount"].to_numpy(), q["T"].to_numpy()
    K = q["strike"].to_numpy()
    as_call = q.assign(right="C", mid=oc.black76_price(F, K, T, D, _skew(np.log(K / F), T), True))
    as_call["bid"], as_call["ask"] = as_call["mid"] - 0.25, as_call["mid"] + 0.25
    a = arb.build_problem(q, scale=5000.0)
    b = arb.build_problem(as_call, scale=5000.0)
    np.testing.assert_allclose(a.y, b.y, rtol=0, atol=1e-9)


def test_injected_butterfly_is_found_executable_and_repaired_minimally():
    q = _chain(_skew, half=0.05)
    i = int(q.index[(q["expiry"] == "E060") & (q["strike"] == 5500.0)][0])
    q.loc[i, ["mid", "bid", "ask"]] += 3.0               # rich body: convexity broken by ~3 points
    rep = arb.screen(q)
    assert rep.mid_violations["C3_vertical_butterfly"] >= 1
    assert rep.mid_violations_gt_tick["C3_vertical_butterfly"] >= 1
    assert rep.executable_violations["C3_vertical_butterfly"] >= 1
    assert bool(rep.quote_flags.loc[i, "arb_exec"])
    # the repair moves the rich quote back (down) outside its spread ...
    assert bool(rep.quote_flags.loc[i, "repair_outside_spread"])
    assert rep.quote_flags.loc[i, "repair_dprice"] < -2.0
    # ... and essentially nothing else outside its spread
    assert rep.repair["n_perturbed_outside_spread"] <= 3
    assert rep.repair["max_row_shortfall_after"] < 1e-9
    worst = rep.violations[0]
    assert {leg["strike"] for leg in worst["legs"]} & {5500.0}


def test_mid_violation_inside_the_spread_is_not_executable():
    q = _chain(_skew, half=2.0)
    i = int(q.index[(q["expiry"] == "E060") & (q["strike"] == 5500.0)][0])
    q.loc[i, ["mid", "bid", "ask"]] += 3.0               # convexity broken at mids by ~1 point,
    #                                                      less than the 4-point bid/ask cost
    rep = arb.screen(q)
    assert rep.mid_violations["C3_vertical_butterfly"] >= 1
    assert sum(rep.executable_violations.values()) == 0
    assert rep.repair["n_perturbed_outside_spread"] == 0   # fixable inside the spread


def test_vertical_spread_violation():
    q = _chain(_skew, half=0.05)
    sub = q[(q["expiry"] == "E030") & (q["right"] == "C")].sort_values("strike")
    i, j = int(sub.index[2]), int(sub.index[3])
    q.loc[j, ["mid", "bid", "ask"]] = q.loc[i, ["mid", "bid", "ask"]].to_numpy() + 1.0
    rep = arb.screen(q)
    assert rep.mid_violations["C1_outright"] == 0
    assert rep.mid_violations["C2_vertical_spread"] >= 1
    assert rep.executable_violations["C2_vertical_spread"] >= 1


def test_calendar_violation_between_expiries():
    """A later expiry priced at a much lower vol than the earlier one breaks
    c(T, k) non-decreasing in T (convex order)."""
    def vol(k, T):
        return 0.30 if T < 45 / 365 else 0.12
    q = _chain(vol, expiries=((30, 5000.0, 0.997), (60, 5000.0, 0.994)), half=0.05)
    rep = arb.screen(q)
    cal = (rep.mid_violations["C4_calendar_spread"] + rep.mid_violations["C5_calendar_vertical_spread"]
           + rep.mid_violations["C6_calendar_butterfly"])
    assert cal > 0
    assert rep.mid_violations["C3_vertical_butterfly"] == 0      # each slice alone is fine
    assert rep.repair["max_row_shortfall_after"] < 1e-9


def test_repair_cost_is_the_l1_ba_objective():
    """Hand-checkable case: one quote outside the no-arbitrage band by x with
    tiny spreads costs about x (marginal cost 1 outside the spread)."""
    q = _chain(_skew, half=0.01)
    i = int(q.index[(q["expiry"] == "E120") & (q["strike"] == 4500.0)][0])
    q.loc[i, ["mid", "bid", "ask"]] += 2.0
    rep = arb.screen(q)
    moved = rep.quote_flags["repair_dprice"].abs()
    assert moved.loc[i] == pytest.approx(moved.max())
    assert 1.5 < moved.loc[i] < 2.05


def test_committed_session_screen_counts():
    """2026-10-16: the screen finds executable arbitrage among a few percent of
    the scoring universe (stale Yahoo lines); the repair LP is feasible and
    leaves no violated row."""
    from joint_vol_calibration.surface import quotes as sqm
    sq = sqm.load_session_quotes("2026-10-16")
    rep = arb.screen(sq.quotes)
    n = len(sq.quotes)
    assert n == 2913
    out = rep.repair["n_perturbed_outside_spread"]
    assert 0 < out < 0.05 * n
    assert rep.repair["max_row_shortfall_after"] < 1e-8
    assert rep.n_rows["C3_vertical_butterfly"] == n - 17            # one per interior strike + forward
    # part of the mid violations are below one tick (rounding of quoted prices)
    assert (sum(rep.mid_violations_gt_tick.values()) < sum(rep.mid_violations.values()))
