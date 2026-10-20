"""
vix.py — the Cboe VIX calculation from a stored SPX chain, the variance-swap
strip by expiry, and the ATM-vol / variance-swap-vol ratio (the empirical
beta the backtest's beta x VIX mark assumes).

Methodology implemented (Cboe Volatility Index Methodology v6.0, 2026-02-26,
and the Cboe Volatility Index Mathematics Methodology, sec. 3):
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
           sample calculation)
The sample calculation in the methodology's appendices is a test fixture
(tests/fixtures/surface_cboe_vix_sample.csv): this code reproduces its
sigma^2 values and VIX = 13.93 (tests/test_surface_vix.py).

Deviations forced by the stored data (reported per session, never hidden):
  * R: Cboe interpolates Treasury CMT yields with a bounded cubic spline per
    expiry; here R = the ^IRX 13-week bill rate of the session (cc), the
    only rate the database stores. dVIX/dR ~ VIX T / 2: a 25 bp error moves
    the 30-day index by ~0.002 points.
  * timing: a stored chain has no quote time unless it is a GTH capture, so
    the calculation time is option_chain.t_origin (16:00 ET on the session);
    the official close is the 16:15 ET print. The snapshot's own timing is
    unknown (Yahoo), which is the largest unquantified gap.
  * zero bids: the C22 pipeline stores two-sided lines only (bid > 0), so on
    those chains the two-consecutive-zero-bid stop cannot trigger and the
    strip runs to the last stored strike. Older (pre-C22) snapshots keep
    zero bids and the rule applies.
  * missing expiries: the bot stores only the calibration expiries and their
    neighbours, so a Cboe candidate term can be absent; the bracket then
    uses the nearest stored candidates (wider bracket), and the result is
    flagged. A second, diagnostic replication brackets with any stored
    expiry (any weekday).
  * settlement of root-less chains: pre-C22 snapshots carry no OCC root;
    their lines are treated as PM-settled (the cleaner's convention), which
    misplaces T by 6.5 hours on AM-only dates.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc

MINUTES_PER_YEAR = 525_600.0
CONSTANT_MATURITY_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 184}
OFFICIAL_TICKERS = {"VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M"}


# ── single term ───────────────────────────────────────────────────────────────

def _valid(bid, ask):
    return np.isfinite(bid) & np.isfinite(ask) & (ask >= bid)


def single_term_variance(strikes, call_bid, call_ask, put_bid, put_ask,
                         T: float, R: float, *, forward: Optional[float] = None,
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
    sel = sorted(puts) + [i0] + sorted(calls)
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
        "T": float(T), "growth": growth,
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
    return int(math.floor((_settle_dt(expiry, settlement) - calc_time_ny).total_seconds() / 60.0))


def is_end_of_week_expiry(expiry: str) -> bool:
    """An NYSE session that is a Friday, or a Thursday whose Friday is an NYSE
    holiday (a Friday holiday itself lists no expiry)."""
    d = date.fromisoformat(str(expiry)[:10])
    if not oc.is_nyse_session(d):
        return False
    if d.weekday() == 4:
        return True
    return d.weekday() == 3 and not oc.is_nyse_session(d + timedelta(days=1))


def term_tables(raw: pd.DataFrame, session: str, calc_time_ny: datetime) -> list:
    """Per expiry of a stored chain: settlement, minutes and the strike table.

    One settlement per expiry: AM-settled SPX lines when the expiry has any
    (the VIX uses the SPX monthly, not the same-day SPXW), else PM. Root-less
    lines are PM (option_chain convention). Crossed quotes (bid > ask) are
    treated as null.
    """
    df = oc._normalise(raw)
    df["settlement"] = np.where(df["root"] == "SPX", "AM", "PM")
    out = []
    for exp, g in df.groupby("expiry"):
        has_am = (g["settlement"] == "AM").any()
        sett = "AM" if has_am else "PM"
        lines = g[g["settlement"] == sett]
        M = minutes_to_settlement(calc_time_ny, exp, sett)
        if M <= 0:
            continue
        c = lines[lines["right"] == "C"].groupby("strike")[["bid", "ask"]].last()
        p = lines[lines["right"] == "P"].groupby("strike")[["bid", "ask"]].last()
        tab = c.join(p, how="outer", lsuffix="_c", rsuffix="_p").sort_index()
        out.append({"expiry": str(exp)[:10], "settlement": sett, "minutes": M,
                    "T": M / MINUTES_PER_YEAR, "has_root": bool((lines["root"] != "").any()),
                    "candidate": bool(has_am or is_end_of_week_expiry(exp)),
                    "table": tab})
    out.sort(key=lambda r: r["minutes"])
    return out


def _variance_of(term: dict, R: float, carry: Optional[dict] = None) -> dict:
    """Single-term variance of one term. With ``carry`` (expiry -> (F, D) of
    the cleaned chain's parity regression) and the expiry in it, F and the
    discount come from there ("parity_regression"); otherwise the Cboe
    single-strike forward with rate R."""
    tab = term["table"]
    args = (tab.index.to_numpy(float), tab["bid_c"].to_numpy(float), tab["ask_c"].to_numpy(float),
            tab["bid_p"].to_numpy(float), tab["ask_p"].to_numpy(float), term["T"])
    if carry is not None and term["expiry"] in carry:
        F, D = carry[term["expiry"]]
        out = single_term_variance(*args, -math.log(D) / term["T"], forward=F)
        if out["ok"]:
            out["forward_method"] = "parity_regression"
        return out
    return single_term_variance(*args, R)


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
            "days": t["minutes"] / 1440.0, "F": v["F"], "K0": v["K0"],
            "forward_method": v["forward_method"], "k0_fallback": v["k0_fallback"],
            "sigma2": v["sigma2"], "single_term_vol": 100.0 * math.sqrt(max(v["sigma2"], 0.0)),
            "n_puts": v["n_puts"], "n_calls": v["n_calls"],
            "lowest_K_over_F": v["lowest_strike"] / v["F"],
            "highest_K_over_F": v["highest_strike"] / v["F"]}


def replicate_index(terms: list, R: float, days: int, *, candidates_only: bool = True,
                    carry: Optional[dict] = None, spot: Optional[float] = None) -> dict:
    """One constant-maturity index from ``terms`` (term_tables output).

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


def variance_swap_strip(terms: list, R: float, quotes: Optional[pd.DataFrame] = None,
                        carry: Optional[dict] = None, surfaces: Optional[dict] = None) -> list:
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
        row.update({"F": v["F"], "forward_method": v["forward_method"], "K0": v["K0"],
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


def gap_attribution(rep: dict, terms: list, R: float, surface, fitted_expiries,
                    official: Optional[float], *, carry: Optional[dict] = None,
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

    Reported: far_wing_market_minus_model_pts = L0 - L1 (how much richer the
    stored quotes beyond the fitted range are than the surface's extrapolated
    wing), fit_inside_range_pts = L1 - L2, discretisation_pts = L2 - L2c,
    truncation_pts = L3 - L2c (what strikes beyond the stored strip add under
    the surface's wing; with a positive far-wing difference the surface's
    wing is thinner than the market's, so this likely understates the
    truncation), and residual_pts = official - L0 - truncation_pts (snapshot
    timing vs the 16:15 ET print, stale quotes, rate, missing Cboe series).

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
    if official is not None:
        out["official"] = official
        out["gap_pts"] = L0 - official
        out["residual_pts"] = official - L0 - (L3 - L2c)
    return out


def constant_maturity_atm(rep: dict, strip: list) -> Optional[float]:
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
