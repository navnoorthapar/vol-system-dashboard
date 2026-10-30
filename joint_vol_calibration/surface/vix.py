"""
vix.py — the Cboe VIX calculation from a stored SPX chain, the variance-swap
strip by expiry, and the ATM-vol / variance-swap-vol ratio (the empirical
beta the backtest's beta x VIX mark assumes).

Methodology implemented (Cboe Volatility Index Methodology v6.0, 2026-02-26,
and the Cboe Volatility Index Mathematics Methodology, sec. 2.1 and 3):
  T        minutes from the calculation time to settlement (09:30 ET for
           AM-settled SPX, 16:00 ET for PM-settled SPXW), rounded down,
           / 525,600
  F        the strike with the smallest |call mid - put mid| (ties: lowest
           strike; both sides must be non-null and not crossed), then
           F = K* + e^{RT} (C - P)
  K0       the listed strike equal to or immediately below F
  strikes  OTM puts below K0 walking down, OTM calls above K0 walking up;
           a zero bid or zero ask is skipped, two consecutive ones stop the
           walk; both options at K0 (their mids averaged)
  dK       half the distance between neighbouring selected strikes (the
           adjacent distance at the ends)
  sigma^2  (2/T) sum dK/K^2 e^{RT} Q(K) - (1/T) (F/K0 - 1)^2
  index    100 sqrt{[T1 s1 (M2 - Mcm)/(M2 - M1) + T2 s2 (Mcm - M1)/(M2 - M1)]
                    x 525,600 / Mcm}
  terms    the Bracket Method on the VIX candidate set: AM-settled SPX
           expiries and PM-settled SPXW expiries at the end of a week
           (Friday, or the session before a Friday holiday), excluding SPXW
           on an AM expiry date; near = the latest candidate with M <= Mcm,
           next = the first candidate after it (the methodology text in v6.0
           lists only the near-term rule; this next-term rule reproduces its
           sample calculation). An index whose stored candidates do not
           bracket its constant maturity (the near weight would fall outside
           [0, 1]) is not replicated from the stored chain: headline_method
           returns no method, and the extrapolated level stays in its
           method's record, flagged.
  R        per term (cmt_rates; Mathematics Methodology sec. 2.1): the
           Treasury par yields (CMT, bond-equivalent) of the newest Treasury
           date before the session, at the twelve tenors of Cboe's table
           (1, 2, 3, 6 months, 1, 2, 3, 5, 7, 10, 20, 30 years as 30, 60,
           91, 182, 365, ..., 10,950 days; the 1.5- and 4-month yields the
           Treasury now also publishes are not in the table and are not
           used); a natural cubic spline in days, evaluated at the term's
           expiration date counted in calendar days from the curve's date,
           held between two consecutive tenors within their two yields and,
           before the first tenor, between Cboe's two lines through it;
           then APY = (1 + BEY/2)^2 - 1 and R = ln(1 + APY). The curve is
           the previous business day's, as in the sample (it computes on
           2022-09-27 with the curve "available on September 26th at the
           end of the day", its expiries 25 and 32 days from that date):
           the one the session's official 16:15 ET close was computed with.
The sample calculation in the methodology's appendices is a test fixture
(tests/fixtures/surface_cboe_vix_sample.csv): this code reproduces its
rates R1 = 0.031664% and R2 = 0.028797% from its yield curve, its sigma^2
values and VIX = 13.93 (tests/test_surface_vix.py).

Deviations forced by the stored data (reported per session, never hidden):
  * R: when the stored Treasury par curve has no date within 7 days before
    the session, R is the session's reference rate (the ^IRX bill or the
    zero curve's 91-day rate, cc) for both terms, and the session's
    deviations say so. dVIX/dR ~ VIX T / 2: a 25 bp error moves the 30-day
    index by ~0.002 points. Beyond the 30-year tenor Cboe states no rule;
    the yield is held at the last one (no VIX-family term reaches it).
  * timing: a chain that records no quote time (Yahoo) is calculated at
    option_chain.t_origin (16:00 ET on the session); the official close is
    the 16:15 ET print. The snapshot's own timing is unknown, which is the
    largest unquantified gap. A Cboe book is calculated at the instant its
    quotes stand at (report._calc_time_ny: its quote time, or a closing
    book's 16:15 ET close), as Cboe times its own calculation: no departure,
    though a book quoted in Global Trading Hours is still compared with the
    previous 16:15 ET print (a gap note).
  * zero bids: the C22 pipeline stores two-sided lines only (bid > 0), so on
    those chains the two-consecutive-zero-bid stop cannot trigger and the
    strip runs to the last stored strike. Older (pre-C22) snapshots keep
    zero bids and the rule applies.
  * missing expiries: the bot's Yahoo download stores only the calibration
    expiries and their neighbours, so a Cboe candidate term can be absent;
    the bracket then uses the nearest stored candidates (wider bracket).
    cboe_term_window checks the pair against Cboe's window (more than 23
    and fewer than 37 days) and the report lists a pair outside it, with the
    missing candidates, among the session's deviations. A session's Cboe
    capture holds the full listing, so its bracket is Cboe's. A second,
    diagnostic replication brackets with any stored expiry (any weekday).
  * settlement of root-less chains: pre-C22 snapshots carry no OCC root;
    their lines are treated as PM-settled (the cleaner's convention), which
    misplaces T by 6.5 hours on AM-only dates.
  * AM dates stored as SPXW only: when a rooted chain holds SPXW (PM) lines
    but no SPX (AM) lines on an SPX monthly settlement date (the third
    Friday, or the session before it when that Friday is a holiday; the
    2026-10-16 Yahoo capture stored 2026-12-18 and 2027-01-15 this way), the
    stated rule would drop the date from the candidate set and widen the
    bracket. The SPXW lines are kept as a proxy for the AM monthly instead,
    on their own PM timing (T to 16:00, 6.5 hours past the 09:30 SOQ the AM
    series settles on), flagged am_date_pm_proxy per term and listed in the
    session's deviations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc

MINUTES_PER_YEAR = 525_600.0
CONSTANT_MATURITY_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 184}
OFFICIAL_TICKERS = {"VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M"}


# ── risk-free rate per term (Mathematics Methodology sec. 2.1) ────────────────

# The CMT tenors of Cboe's spline and the number of days it gives each (the
# methodology's table). The 1.5- and 4-month par yields the Treasury file has
# carried since 2025-02-18 and 2022-10-19 are not in the table, so not used.
CBOE_CMT_DAYS = {"UST_PAR_1M": 30, "UST_PAR_2M": 60, "UST_PAR_3M": 91, "UST_PAR_6M": 182,
                 "UST_PAR_1Y": 365, "UST_PAR_2Y": 730, "UST_PAR_3Y": 1095, "UST_PAR_5Y": 1825,
                 "UST_PAR_7Y": 2555, "UST_PAR_10Y": 3650, "UST_PAR_20Y": 7300, "UST_PAR_30Y": 10950}
# How old the newest par curve before a session may be (calendar days), as
# for the calibrator's zero curve (rates.get_zero_curve).
CMT_MAX_AGE_DAYS = 7


def cboe_bounded_spline_bey(days, bey, t: float) -> float:
    """The bond-equivalent yield at ``t`` days by Cboe's bounded cubic spline
    (Mathematics Methodology sec. 2.1) through the CMT points (days, bey).

    A natural cubic spline in days, bounded: between two tenors by the lower
    and the higher of their yields; before the shortest tenor (t1, y1) from
    below by the line through it and the next tenor whose yield is >= y1,
    from above by the line through it and the next tenor whose yield is
    <= y1 (a bound with no such tenor is flat at y1). Beyond the longest
    tenor the methodology states no rule; the last yield is held."""
    from scipy.interpolate import CubicSpline
    d, y = np.asarray(days, dtype=float), np.asarray(bey, dtype=float)
    o = np.argsort(d, kind="mergesort")
    d, y = d[o], y[o]
    if len(d) < 2:
        raise ValueError("Cboe's spline needs at least two CMT points")
    t = float(t)
    if t >= d[-1]:
        return float(y[-1])
    val = float(CubicSpline(d, y, bc_type="natural")(t))
    if t >= d[0]:
        i = int(np.searchsorted(d, t, side="right")) - 1          # d[i] <= t < d[i + 1]
        lo, hi = min(y[i], y[i + 1]), max(y[i], y[i + 1])
    else:
        def slope(reaches) -> float:
            j = next((j for j in range(1, len(d)) if reaches(y[j])), None)
            return 0.0 if j is None else float((y[j] - y[0]) / (d[j] - d[0]))
        lo = float(y[0] + slope(lambda v: v >= y[0]) * (t - d[0]))
        hi = float(y[0] + slope(lambda v: v <= y[0]) * (t - d[0]))
    return float(min(max(val, lo), hi))


def bey_to_cc(bey: float) -> float:
    """Cboe's conversion of a bond-equivalent yield to the continuously
    compounded rate: APY = (1 + BEY/2)^2 - 1, then r = ln(1 + APY)."""
    apy = (1.0 + bey / 2.0) ** 2 - 1.0
    return math.log1p(apy)


@dataclass(frozen=True)
class CMTRates:
    """The risk-free rate of each term by Cboe's rule: the CMT par yields of
    one Treasury date (days -> bond-equivalent yield, Cboe's tenors only), a
    term's rate the bounded spline at its expiration date, counted in days
    from that date, converted to continuous compounding. Called with an
    expiry (YYYY-MM-DD) it returns that rate."""
    curve_date: str
    days: tuple
    bey: tuple

    def t_days(self, expiry: str) -> int:
        return (date.fromisoformat(str(expiry)[:10]) - date.fromisoformat(self.curve_date)).days

    def __call__(self, expiry: str) -> float:
        return bey_to_cc(cboe_bounded_spline_bey(self.days, self.bey, self.t_days(expiry)))

    def record(self) -> dict:
        return {"curve_date": self.curve_date,
                "cmt_yields_by_days": {str(int(d)): float(v) for d, v in zip(self.days, self.bey)}}


def cmt_rates(session: str, *, root=None) -> CMTRates:
    """Cboe's rate input for ``session``'s replication: the stored Treasury
    par yields (rates.par_yields) of the newest Treasury date before the
    session, at most CMT_MAX_AGE_DAYS old, at Cboe's tenors. The session's
    own curve is not used: the methodology's sample computes on 2022-09-27
    with the curve of 2022-09-26. Raises LookupError (no curve, or fewer than
    two of Cboe's tenors) or database.StaleDataError (too old)."""
    from joint_vol_calibration.data import rates
    curve_date, par = rates.par_yields(str(session)[:10], same_day=False,
                                       max_age_days=CMT_MAX_AGE_DAYS, root=root)
    pts = sorted((CBOE_CMT_DAYS[s], v) for s, v in par.items() if s in CBOE_CMT_DAYS)
    if len(pts) < 2:
        raise LookupError(f"the Treasury par curve of {curve_date} has {len(pts)} of Cboe's CMT tenors")
    return CMTRates(curve_date, tuple(float(d) for d, _ in pts), tuple(float(v) for _, v in pts))


Rate = float | Callable[[str], float]


def _rate(R: Rate, expiry: str) -> float:
    """The rate of ``expiry``'s term: R itself, or R(expiry) for a rate per term (CMTRates)."""
    return float(R(str(expiry)[:10])) if callable(R) else float(R)


# ── single term ───────────────────────────────────────────────────────────────

def _valid(bid, ask):
    return np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)


def single_term_variance(strikes, call_bid, call_ask, put_bid, put_ask,
                         T: float, R: float, *, forward: float | None = None,
                         k0_fallback: bool = True) -> dict:
    """Cboe single-term variance (Mathematics Methodology sec. 3(a)).

    Arrays are per strike (NaN = no quote). ``forward`` given: use it instead
    of the single-strike rule (and e^{RT} with the caller's R, which should
    then be the same discount, R = -ln D / T). ``k0_fallback``: when the
    listed strike immediately below F lacks a valid put or call (a stored
    chain that kept two-sided lines only), use the nearest lower strike that
    has both, and flag it; without the fallback that is the methodology's
    "cannot be calculated" case.

    Returns a dict with sigma2, F, K0, the selected strikes with dK, Q and
    contributions, counts and flags; or {'ok': False, 'reason': ...}.
    """
    K = np.asarray(strikes, dtype=float)
    o = np.argsort(K, kind="mergesort")
    K = K[o]
    cb, ca, pb, pa = (np.asarray(x, dtype=float)[o] for x in (call_bid, call_ask, put_bid, put_ask))
    cv, pv = _valid(cb, ca), _valid(pb, pa)
    cmid = np.where(cv, 0.5 * (cb + ca), np.nan)
    pmid = np.where(pv, 0.5 * (pb + pa), np.nan)
    both = cv & pv
    if not both.any():
        return {"ok": False, "reason": "no strike with both a call and a put quote"}
    growth = math.exp(R * T)
    if forward is None:
        diff = np.where(both, np.abs(cmid - pmid), np.inf)
        i_atm = int(np.argmin(diff))                 # argmin returns the lowest strike on ties
        F = float(K[i_atm] + growth * (cmid[i_atm] - pmid[i_atm]))
        atm_strike, f_method = float(K[i_atm]), "cboe_single_strike"
    else:
        F, atm_strike, f_method = float(forward), None, "given"
    below = np.where(K <= F)[0]
    if not len(below):
        return {"ok": False, "reason": "no strike at or below the forward"}
    literal_K0 = float(K[below[-1]])
    ok_below = [j for j in below if both[j]]
    if both[below[-1]]:
        i0, fallback = int(below[-1]), False
    elif k0_fallback and ok_below:
        i0, fallback = int(ok_below[-1]), True
    else:
        return {"ok": False, "reason": f"K0 = {literal_K0:g} lacks a valid put or call quote"}
    K0 = float(K[i0])

    def walk(idx, bid, ask, valid):
        """Strike indices selected walking away from K0 (Cboe 3(a)(iii))."""
        chosen, zeros = [], 0
        for j in idx:
            if not valid[j]:
                continue                             # null quote: strike removed
            if bid[j] == 0 or ask[j] == 0:
                zeros += 1
                if zeros >= 2:
                    break
                continue
            zeros = 0
            chosen.append(j)
        return chosen

    puts = walk(range(i0 - 1, -1, -1), pb, pa, pv)
    calls = walk(range(i0 + 1, len(K)), cb, ca, cv)
    if not puts:
        return {"ok": False, "reason": "every OTM put excluded"}
    if not calls:
        return {"ok": False, "reason": "every OTM call excluded"}
    sel = [*sorted(puts), i0, *sorted(calls)]
    Ks = K[sel]
    Q = np.array([pmid[j] if j < i0 else (cmid[j] if j > i0 else 0.5 * (pmid[j] + cmid[j]))
                  for j in sel])
    dK = np.empty(len(Ks))
    dK[1:-1] = 0.5 * (Ks[2:] - Ks[:-2])
    dK[0] = Ks[1] - Ks[0]
    dK[-1] = Ks[-1] - Ks[-2]
    contrib = dK / Ks ** 2 * growth * Q
    sigma2 = 2.0 / T * float(contrib.sum()) - (F / K0 - 1.0) ** 2 / T
    return {
        "ok": True, "sigma2": sigma2, "F": F, "K0": K0, "atm_strike": atm_strike,
        "T": float(T), "R": float(R), "growth": growth,
        "forward_method": f_method, "k0_fallback": fallback, "literal_K0": literal_K0,
        "n_puts": len(puts), "n_calls": len(calls),
        "lowest_strike": float(Ks[0]), "highest_strike": float(Ks[-1]),
        "sum_contributions": float(contrib.sum()),
        "strikes": Ks, "dK": dK, "Q": Q, "contributions": contrib,
    }


def constant_maturity_index(sigma2_near: float, M_near: float, sigma2_next: float,
                            M_next: float, M_cm: float) -> float:
    """Cboe constant-maturity interpolation (sec. 3(b)); returns the index level."""
    T1, T2 = M_near / MINUTES_PER_YEAR, M_next / MINUTES_PER_YEAR
    w1 = (M_next - M_cm) / (M_next - M_near)
    w2 = (M_cm - M_near) / (M_next - M_near)
    var = (T1 * sigma2_near * w1 + T2 * sigma2_next * w2) * MINUTES_PER_YEAR / M_cm
    return 100.0 * math.sqrt(var) if var > 0 else float("nan")


# ── chain plumbing ────────────────────────────────────────────────────────────

def _settle_dt(expiry: str, settlement: str) -> datetime:
    t = oc.AM_SETTLE_ET if str(settlement).upper() == "AM" else oc.PM_SETTLE_ET
    return datetime.combine(date.fromisoformat(str(expiry)[:10]), t)


def minutes_to_settlement(calc_time_ny: datetime, expiry: str, settlement: str) -> int:
    """Whole minutes (rounded down) from a naive New York wall-clock time."""
    return math.floor((_settle_dt(expiry, settlement) - calc_time_ny).total_seconds() / 60.0)


def is_end_of_week_expiry(expiry: str) -> bool:
    """An NYSE session that is a Friday, or a Thursday whose Friday is an NYSE
    holiday (a Friday holiday itself lists no expiry)."""
    d = date.fromisoformat(str(expiry)[:10])
    if not oc.is_nyse_session(d):
        return False
    if d.weekday() == 4:
        return True
    return d.weekday() == 3 and not oc.is_nyse_session(d + timedelta(days=1))


def is_spx_am_settlement_date(expiry: str) -> bool:
    """True on the SPX monthly (AM, SOQ) settlement date of the expiry's
    month: the third Friday, or the NYSE session before it when that Friday
    is a holiday."""
    d = date.fromisoformat(str(expiry)[:10])
    first = d.replace(day=1)
    third_friday = first + timedelta(days=(4 - first.weekday()) % 7 + 14)
    settle = third_friday if oc.is_nyse_session(third_friday) else oc.previous_session(third_friday)
    return d == settle


def term_tables(raw: pd.DataFrame, session: str, calc_time_ny: datetime) -> list:
    """Per expiry of a stored chain: settlement, minutes and the strike table.

    One settlement per expiry: AM-settled SPX lines when the expiry has any
    (the VIX uses the SPX monthly, not the same-day SPXW), else PM. Root-less
    lines are PM (option_chain convention). On an SPX monthly settlement
    date with SPXW lines but no SPX lines stored, the PM SPXW lines stand in
    for the AM monthly (am_date_pm_proxy; module docstring). Crossed quotes
    (bid > ask) are treated as null.
    """
    df = oc._normalise(raw)
    df["settlement"] = np.where(df["root"] == "SPX", "AM", "PM")
    out: list[dict[str, Any]] = []
    for key, g in df.groupby("expiry"):
        exp = cast(str, key)
        has_am = (g["settlement"] == "AM").any()
        sett = "AM" if has_am else "PM"
        lines = g[g["settlement"] == sett]
        M = minutes_to_settlement(calc_time_ny, exp, sett)
        if M <= 0:
            continue
        c = lines[lines["right"] == "C"].groupby("strike")[["bid", "ask"]].last()
        p = lines[lines["right"] == "P"].groupby("strike")[["bid", "ask"]].last()
        tab = c.join(p, how="outer", lsuffix="_c", rsuffix="_p").sort_index()
        has_root = bool((lines["root"] != "").any())
        out.append({"expiry": str(exp)[:10], "settlement": sett, "minutes": M,
                    "T": M / MINUTES_PER_YEAR, "has_root": has_root,
                    "am_date_pm_proxy": bool(not has_am and has_root
                                             and is_spx_am_settlement_date(exp)),
                    "candidate": bool(has_am or is_end_of_week_expiry(exp)),
                    "table": tab})
    out.sort(key=lambda r: r["minutes"])
    return out


def _variance_of(term: dict, R: Rate, carry: dict | None = None) -> dict:
    """Single-term variance of one term. With ``carry`` (expiry -> (F, D) of
    the cleaned chain's parity regression) and the expiry in it, F and the
    discount come from there ("parity_regression"); otherwise the Cboe
    single-strike forward with the term's rate (R, or R(expiry) for a rate
    per term)."""
    tab = term["table"]
    args = (tab.index.to_numpy(float), tab["bid_c"].to_numpy(float), tab["ask_c"].to_numpy(float),
            tab["bid_p"].to_numpy(float), tab["ask_p"].to_numpy(float), term["T"])
    if carry is not None and term["expiry"] in carry:
        F, D = carry[term["expiry"]]
        out = single_term_variance(*args, -math.log(D) / term["T"], forward=F)
        if out["ok"]:
            out["forward_method"] = "parity_regression"
        return out
    return single_term_variance(*args, _rate(R, term["expiry"]))


def bracket(terms: list, M_cm: float) -> tuple:
    """(near, next) per the Bracket Method; None when a side is missing."""
    if not terms:
        return None, None
    le = [t for t in terms if t["minutes"] <= M_cm]
    near = le[-1] if le else terms[0]
    after = [t for t in terms if t["minutes"] > near["minutes"]]
    nxt = after[0] if after else None
    return near, nxt


def _term_summary(t: dict, v: dict) -> dict:
    return {"expiry": t["expiry"], "settlement": t["settlement"], "minutes": t["minutes"],
            "am_date_pm_proxy": bool(t.get("am_date_pm_proxy", False)),
            "days": t["minutes"] / 1440.0, "R": v["R"], "F": v["F"], "K0": v["K0"],
            "forward_method": v["forward_method"], "k0_fallback": v["k0_fallback"],
            "sigma2": v["sigma2"], "single_term_vol": 100.0 * math.sqrt(max(v["sigma2"], 0.0)),
            "n_puts": v["n_puts"], "n_calls": v["n_calls"],
            "lowest_K_over_F": v["lowest_strike"] / v["F"],
            "highest_K_over_F": v["highest_strike"] / v["F"]}


def replicate_index(terms: list, R: Rate, days: int, *, candidates_only: bool = True,
                    carry: dict | None = None, spot: float | None = None) -> dict:
    """One constant-maturity index from ``terms`` (term_tables output).

    R     : the risk-free rate, one for every term or a function of the
            term's expiry (CMTRates: Cboe's rate per term); each term's rate
            is recorded in its summary
    carry : expiry -> (F, D) to use the parity forwards (see _variance_of)
    spot  : when given, a term whose forward lies outside [0.8, 1.25] x spot
            is rejected (a stale strike can make the single-strike rule pick
            a far wing: 2026-05-29, 2026-12-18 expiry, F = 2,396)
    """
    pool = [t for t in terms if t["candidate"]] if candidates_only else list(terms)
    M_cm = days * 1440.0
    near, nxt = bracket(pool, M_cm)
    if near is None or nxt is None:
        return {"ok": False, "reason": "fewer than two usable terms"}
    v1, v2 = _variance_of(near, R, carry), _variance_of(nxt, R, carry)
    if not (v1["ok"] and v2["ok"]):
        return {"ok": False, "reason": f"near {near['expiry']}: {v1.get('reason', 'ok')}; "
                                       f"next {nxt['expiry']}: {v2.get('reason', 'ok')}"}
    if spot is not None:
        for t, v in ((near, v1), (nxt, v2)):
            if not (0.8 * spot < v["F"] < 1.25 * spot):
                return {"ok": False, "reason": f"{t['expiry']}: implausible forward {v['F']:.1f} "
                                               f"(spot {spot:.1f}) from the single-strike rule"}
    level = constant_maturity_index(v1["sigma2"], near["minutes"], v2["sigma2"], nxt["minutes"], M_cm)
    w_near = (nxt["minutes"] - M_cm) / (nxt["minutes"] - near["minutes"])
    return {"ok": True, "level": level, "constant_maturity_days": days,
            "near": _term_summary(near, v1), "next": _term_summary(nxt, v2),
            "near_weight": w_near, "extrapolated": bool(w_near < 0 or w_near > 1),
            "candidates_only": candidates_only,
            "forward_source": "parity_regression" if carry is not None else "cboe_single_strike"}


# Cboe VIX methodology: the near- and next-term options have more than 23
# and less than 37 days to expiration.
CBOE_VIX_TERM_WINDOW_DAYS = (23.0, 37.0)


def cboe_term_window(rep: dict, terms: list, calc_time_ny: datetime,
                     window: tuple = CBOE_VIX_TERM_WINDOW_DAYS) -> dict | None:
    """Whether a replicated VIX's two terms are Cboe's: each with more than
    window[0] and fewer than window[1] days to settlement. With a full
    listing the bracket method picks such a pair; a stored chain that lacks
    the candidates in the window (the bot's Yahoo download keeps 17
    expiries: on 2026-10-26 the near term was 10.6 days out) brackets wider.

    rep   : replicate_index output (None when it did not replicate)
    terms : term_tables output of the same chain at ``calc_time_ny``

    Returns {"window_days", "near", "next" ({expiry, days}), "inside",
    "missing_candidates"}: the VIX candidate dates (AM monthly settlement
    dates, end-of-week PM dates) whose settlement falls inside the window and
    that the stored chain does not hold, each with its days. None when ``rep``
    did not replicate."""
    if not (rep or {}).get("ok"):
        return None
    lo, hi = float(window[0]), float(window[1])
    held = {t["expiry"] for t in terms if t.get("candidate")}
    missing = []
    start = calc_time_ny.date()
    for i in range(int(hi) + 2):
        d = (start + timedelta(days=i)).isoformat()
        sett = "AM" if is_spx_am_settlement_date(d) else "PM" if is_end_of_week_expiry(d) else None
        if sett is None or d in held:
            continue
        days = minutes_to_settlement(calc_time_ny, d, sett) / 1440.0
        if lo < days < hi:
            missing.append({"expiry": d, "settlement": sett, "days": days})
    near, nxt = rep["near"], rep["next"]
    inside = all(lo < t["days"] < hi for t in (near, nxt))
    return {"window_days": [lo, hi],
            "near": {"expiry": near["expiry"], "days": near["days"]},
            "next": {"expiry": nxt["expiry"], "days": nxt["days"]},
            "inside": bool(inside), "missing_candidates": missing}


def term_window_deviation(win: dict | None, name: str = "VIX") -> str | None:
    """The deviation line for a bracket outside Cboe's term window (None when
    the bracket is Cboe's or nothing was replicated)."""
    if not win or win["inside"]:
        return None
    lo, hi = win["window_days"]
    n, x = win["near"], win["next"]
    miss = win["missing_candidates"]
    return (f"{name} bracket {n['expiry']} ({n['days']:.1f} d) / {x['expiry']} ({x['days']:.1f} d) "
            f"is not Cboe's: Cboe's near and next terms have more than {lo:.0f} and fewer than "
            f"{hi:.0f} days to expiration, and the stored chain "
            + (("lacks the candidate expiries " if len(miss) > 1 else "lacks the candidate expiry ")
               + ", ".join(f"{m['expiry']} ({m['days']:.1f} d)" for m in miss)
               if miss else "holds no candidate expiry in that window")
            + "; the level interpolates a wider bracket.")


HEADLINE_METHODS = ("cboe_rules", "parity_forward")


def headline_method(rep: dict) -> tuple:
    """(method, reason) for one index's replications {method: replicate_index
    output}: the first of HEADLINE_METHODS that is ok and interpolates
    (0 <= near weight <= 1); (None, reason) when none does."""
    for m in HEADLINE_METHODS:
        r = rep.get(m) or {}
        if r.get("ok") and not r.get("extrapolated"):
            return m, None
    ok = [rep[m] for m in HEADLINE_METHODS if (rep.get(m) or {}).get("ok")]
    if ok:
        r = ok[0]
        n, x = r["near"], r["next"]
        return None, (f"no stored Cboe candidate expiries on both sides of "
                      f"{r['constant_maturity_days']} days: the nearest pair {n['expiry']} "
                      f"({n['days']:.1f}d) / {x['expiry']} ({x['days']:.1f}d) would extrapolate "
                      f"(near weight {r['near_weight']:.3f}, outside [0, 1]), which is not "
                      "Cboe's interpolation")
    reasons = "; ".join(f"{m}: {(rep.get(m) or {}).get('reason', 'not run')}" for m in HEADLINE_METHODS)
    return None, f"no replication: {reasons}"


# ── variance-swap strip and the ATM / variance-swap ratio ─────────────────────

def _bs_otm_normalised(k, w):
    from scipy.special import ndtr
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = -k / sw + 0.5 * sw
    d2 = d1 - sw
    call = ndtr(d1) - np.exp(k) * ndtr(d2)
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
    return np.where(k >= 0, call, put)


def surface_variance_swap(surface, T: float, k_lo: float = -3.0, k_hi: float = 2.0,
                          n: int = 6001) -> float:
    """Variance-swap variance of an iv-surface at T: sigma^2 T = 2 int otm(k) e^{-k} dk
    over [k_lo, k_hi] (forward-normalised OTM prices). Simpson's rule on each
    side of k = 0, where the OTM integrand has its put / call kink, with about
    n points in all (flat 20% vol, 30 days: relative error 2e-12 in sigma^2,
    against 4e-5 for one trapezoid on 6,001 points)."""
    from scipy.integrate import simpson
    total = 0.0
    span = k_hi - k_lo
    for a, b in ((k_lo, min(0.0, k_hi)), (max(0.0, k_lo), k_hi)):
        if b <= a:
            continue
        m = max(int(n * (b - a) / span) | 1, 101)          # odd point count
        k = np.linspace(a, b, m)
        f = _bs_otm_normalised(k, surface.w(k, T)) * np.exp(-k)
        total += float(simpson(f, x=k))
    return 2.0 * total / T


def variance_swap_strip(terms: list, R: Rate, quotes: pd.DataFrame | None = None,
                        carry: dict | None = None, surfaces: dict | None = None) -> list:
    """Per expiry: the Cboe single-term variance-swap vol from the stored
    quotes (on the parity forward when ``carry`` has the expiry), the
    model-free ATM vol (mid IVs interpolated to k = 0 in the scoring
    universe), their difference and ratio ATM / VS (the empirical beta), and
    for each fitted surface in ``surfaces`` (name -> object with w(k, T) and
    iv(k, T)) its ATM vol and model variance-swap vol, on k in [-3, 2] and on
    the quoted k-range. Model variance swaps extrapolate the fitted wings
    beyond K/F 0.70-1.30 (the universe) and inherit each model's wing bias."""
    rows = []
    qk = {}
    if quotes is not None:
        for exp, g in quotes.groupby("expiry", sort=False):
            o = np.argsort(g["k"].to_numpy(float))
            qk[str(exp)] = (g["k"].to_numpy(float)[o], g["mid_iv"].to_numpy(float)[o],
                            float(g["T"].iloc[0]))
    for t in terms:
        v = _variance_of(t, R, carry)
        row = {"expiry": t["expiry"], "settlement": t["settlement"], "days": t["minutes"] / 1440.0,
               "cboe_candidate": t["candidate"], "ok": v["ok"]}
        if not v["ok"]:
            row["reason"] = v["reason"]
            rows.append(row)
            continue
        vs = 100.0 * math.sqrt(max(v["sigma2"], 0.0))
        row.update({"R": v["R"], "F": v["F"], "forward_method": v["forward_method"], "K0": v["K0"],
                    "k0_fallback": v["k0_fallback"], "n_puts": v["n_puts"], "n_calls": v["n_calls"],
                    "lowest_K_over_F": v["lowest_strike"] / v["F"],
                    "highest_K_over_F": v["highest_strike"] / v["F"], "var_swap_vol": vs})
        if t["expiry"] in qk:
            kq, ivq, Tq = qk[t["expiry"]]
            atm_q = 100.0 * float(np.interp(0.0, kq, ivq))
            row.update({"T_surface": Tq, "atm_vol_quotes": atm_q, "vs_minus_atm_vp": vs - atm_q,
                        "atm_over_vs": atm_q / vs})
            for name, surf in (surfaces or {}).items():
                full = surface_variance_swap(surf, Tq)
                inq = surface_variance_swap(surf, Tq, float(kq.min()), float(kq.max()), 2001)
                row[f"atm_vol_{name}"] = 100.0 * float(surf.iv(np.array([0.0]), Tq)[0])
                row[f"var_swap_vol_{name}_k_-3_to_2"] = 100.0 * math.sqrt(max(full, 0.0))
                row[f"var_swap_vol_{name}_quoted_range"] = 100.0 * math.sqrt(max(inq, 0.0))
        rows.append(row)
    return rows


FIT_RANGE_K_OVER_F = (0.70, 1.30)     # the scoring universe the surfaces are fitted on


def _model_strip(v: dict, surface, T_surface: float, lo_kf: float, hi_kf: float) -> tuple:
    """sigma^2 of one term with the Cboe discrete sum, market mids replaced by
    the surface's forward OTM prices at the strikes OUTSIDE [lo_kf, hi_kf]
    (lo_kf = hi_kf = F: every strike modelled). Same strikes, dK, F and K0
    as the market strip ``v``. Returns (sigma2_mixed, sigma2_all_model)."""
    K, dK, F, K0 = v["strikes"], v["dK"], v["F"], v["K0"]
    k = np.log(K / F)
    w = surface.w(k, T_surface)
    otm = _bs_otm_normalised(k, w)                       # OTM vs F, forward-normalised
    put_n = np.where(k < 0, otm, otm - (1.0 - np.exp(k)))
    call_n = put_n + (1.0 - np.exp(k))
    q_model = F * np.where(K < K0, put_n, np.where(K > K0, call_n, 0.5 * (put_n + call_n)))
    T = v["T"]
    growth = v["growth"]
    q_mkt_fwd = growth * v["Q"]                          # forward (undiscounted) market prices
    kf = K / F
    outside = (kf < lo_kf) | (kf > hi_kf)
    mixed = np.where(outside, q_model, q_mkt_fwd)
    corr = (F / K0 - 1.0) ** 2 / T
    s_mixed = 2.0 / T * float(np.sum(dK / K ** 2 * mixed)) - corr
    s_model = 2.0 / T * float(np.sum(dK / K ** 2 * q_model)) - corr
    return s_mixed, s_model


def gap_attribution(rep: dict, terms: list, R: Rate, surface, fitted_expiries,
                    official: float | None, *, carry: dict | None = None,
                    name: str = "svi_cal") -> dict:
    """Decompose one replicated index against a fitted arbitrage-free surface
    on the replication's own near / next terms. Levels (index points), each
    with the Cboe constant-maturity weights of ``rep``:

      L0  replicated: Cboe discrete sum on the stored market mids
      L1  same strikes, market mids inside the fitted K/F range 0.70-1.30,
          the surface's prices outside it
      L2  same strikes, the surface's prices everywhere
      L2c the surface's continuous variance swap on the stored strip's K/F range
      L3  the surface's continuous variance swap on k in [-3, 2]

    Only truncation_pts is a term of the gap: official - L0 = truncation_pts +
    residual_pts. The far-wing, fit and discretisation terms sum to L0 - L2c,
    the replication against the surface; residual_share_of_official_minus_
    replicated is the part of the gap the stored chain does not explain.

    Reported: far_wing_market_minus_model_pts = L0 - L1 (how much richer the
    stored quotes beyond the fitted range are than the surface's extrapolated
    wing), fit_inside_range_pts = L1 - L2, discretisation_pts = L2 - L2c,
    truncation_pts = L3 - L2c (what strikes beyond the stored strip add under
    the surface's wing; with a positive far-wing difference the surface's
    wing is thinner than the market's, so this likely understates the
    truncation), and residual_pts = official - L0 - truncation_pts (snapshot
    timing vs the 16:15 ET print, stale quotes, a rate other than Cboe's,
    missing Cboe series).

    The surface is evaluated at each term's own T (minutes to its Cboe
    settlement / 525,600): on a third Friday the VIX uses the AM SPX lines
    while the cleaned chain (and so the fitted slice) keeps the PM SPXW lines,
    6.5 hours later. ``fitted_expiries``: the expiries the surface was fitted
    on; a term outside them (e.g. a 0-6 DTE term) makes the attribution
    unavailable. ``carry``: expiry -> (F, D) when ``rep`` used the parity
    forwards.
    """
    if not rep.get("ok"):
        return {"available": False, "reason": "replication not available"}
    sel = (rep["near"], rep["next"])
    missing = [t["expiry"] for t in sel if t["expiry"] not in set(fitted_expiries)]
    if missing:
        return {"available": False,
                "reason": f"term(s) {', '.join(missing)} not fitted by the surface "
                          "(outside the scoring universe)"}
    by_exp = {t["expiry"]: t for t in terms}
    use_carry = carry if rep.get("forward_source") == "parity_regression" else None
    lo_kf, hi_kf = FIT_RANGE_K_OVER_F
    s0, s1, s2, s2c, s3 = [], [], [], [], []
    for t in sel:
        term = by_exp[t["expiry"]]
        v = _variance_of(term, R, use_carry)
        if not v["ok"]:
            return {"available": False, "reason": f"{t['expiry']}: {v['reason']}"}
        Ts = term["T"]
        mixed, model = _model_strip(v, surface, Ts, lo_kf, hi_kf)
        s0.append(v["sigma2"])
        s1.append(mixed)
        s2.append(model)
        s2c.append(surface_variance_swap(surface, Ts, math.log(t["lowest_K_over_F"]),
                                         math.log(t["highest_K_over_F"]), 2001))
        s3.append(surface_variance_swap(surface, Ts))
    M1, M2 = rep["near"]["minutes"], rep["next"]["minutes"]
    Mcm = rep["constant_maturity_days"] * 1440.0

    def lvl(x):
        return constant_maturity_index(x[0], M1, x[1], M2, Mcm)
    L0, L1, L2, L2c, L3 = lvl(s0), lvl(s1), lvl(s2), lvl(s2c), lvl(s3)
    out = {"available": True, "surface": name, "fit_range_K_over_F": list(FIT_RANGE_K_OVER_F),
           "levels": {"L0_replicated": L0, "L1_model_outside_fit_range": L1,
                      "L2_model_all_strikes": L2, "L2c_model_continuous_on_strip": L2c,
                      "L3_model_continuous_k_-3_to_2": L3},
           "far_wing_market_minus_model_pts": L0 - L1,
           "fit_inside_range_pts": L1 - L2,
           "discretisation_pts": L2 - L2c,
           "truncation_pts": L3 - L2c}
    out["identity"] = ("official - L0 = truncation_pts + residual_pts: only the truncation term "
                       "explains the gap; far_wing + fit_inside_range + discretisation = L0 - L2c "
                       "decompose the replication against the surface, not the gap")
    if official is not None:
        out["official"] = official
        out["gap_pts"] = L0 - official
        out["residual_pts"] = official - L0 - (L3 - L2c)
        if abs(official - L0) > 1e-12:
            out["residual_share_of_official_minus_replicated"] = out["residual_pts"] / (official - L0)
    return out


def constant_maturity_atm(rep: dict, strip: list) -> float | None:
    """The model-free ATM vol at the index's constant maturity: the near / next
    terms' quote ATM vols combined with the index's own weights (total
    variance, as in the VIX formula); None when a term has no ATM quote."""
    if not rep.get("ok"):
        return None
    by = {r["expiry"]: r for r in strip}
    n, x = by.get(rep["near"]["expiry"], {}), by.get(rep["next"]["expiry"], {})
    if "atm_vol_quotes" not in n or "atm_vol_quotes" not in x:
        return None
    M1, M2 = rep["near"]["minutes"], rep["next"]["minutes"]
    Mcm = rep["constant_maturity_days"] * 1440.0
    s1, s2 = (n["atm_vol_quotes"] / 100.0) ** 2, (x["atm_vol_quotes"] / 100.0) ** 2
    return constant_maturity_index(s1, M1, s2, M2, Mcm)


def official_closes(session: str) -> dict:
    """Official index closes stored for ``session`` (look-ahead gated)."""
    out = {}
    vd = db.get_vix_daily(as_of_date=session, start_date=session)
    if len(vd):
        r = vd.iloc[-1]
        out["VIX"] = {"close": float(r["close"]), "high": float(r["high"]), "low": float(r["low"]),
                      "open": float(r["open"]), "source": str(r.get("source")), "table": "vix_daily"}
    ts = db.get_vix_term_structure(as_of_date=session, start_date=session)
    for name, tic in OFFICIAL_TICKERS.items():
        row = ts[ts["ticker"] == tic]
        if len(row) and name not in out:
            out[name] = {"close": float(row["close"].iloc[-1]), "source": str(row["source"].iloc[-1]),
                         "table": "vix_term_structure"}
        elif len(row):
            out[name]["term_structure_close"] = float(row["close"].iloc[-1])
    return out
