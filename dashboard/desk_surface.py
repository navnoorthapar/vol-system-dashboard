"""
desk_surface.py — the fitted SVI-cal surface in desk coordinates (pure functions;
no Flask, no I/O).

A desk quotes a smile by delta, not by strike: the vol at the 10- and 25-delta
put, the at-the-money forward and the 25- and 10-delta call, and from them the
risk reversal and the butterfly. This module reads those numbers off the
calibration-ordered SVI surface the surface report stores (surface.json
``surfaces.svi_cal.params.slices``), per listed expiry and at constant
maturity, with the variance-swap vol of the same chain beside them.

Conventions (every caption on the page states them):
  * k = ln(K / F_T); total variance w(k) = σ(k)²·T, T in years (calendar days /
    365, the surface report's clock);
  * delta is the undiscounted forward Black call delta on the slice's own vol at
    that strike, N(d₁) with d₁ = −k/√w + √w/2; a put's is N(d₁) − 1. It is not
    premium-adjusted;
  * ATMF is k = 0 (strike at the forward), not the delta-neutral straddle;
  * RR = σ_call − σ_put (negative: puts over calls); BF = (σ_call + σ_put)/2 −
    σ_ATMF; both in vol points;
  * the ATM skew is ∂σ/∂k at k = 0 of the slice, w′(0) / (2·√(w(0)·T)), in vol
    (decimal) per unit of ln-moneyness; the grid prints it ×10 in vol points
    (vol pts per 0.1 of ln-moneyness);
  * a pillar whose strike lies outside its expiry's quoted ln(K/F) range is an
    extrapolation of the fitted wings: it is flagged, and the RR/BF that would
    use it is not computed;
  * constant-maturity rows interpolate linearly in total variance at fixed delta
    between the two bracketing listed expiries, and are never extrapolated: a
    tenor outside the listed range has no row values.
These are values of the fitted surface, not raw quotes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from desk_format import signed
from scipy.optimize import brentq
from scipy.special import ndtr

K_LO, K_HI = -3.0, 2.0              # the surface certificates' k range (CERT_GRID)
DAYS_PER_YEAR = 365.0               # the surface report's T = calendar days / 365
# (key, label, signed delta): put deltas negative, call deltas positive, ATMF k = 0
PILLARS: tuple[tuple[str, str, float | None], ...] = (
    ("10P", "10ΔP", -0.10), ("25P", "25ΔP", -0.25), ("ATM", "ATMF", None),
    ("25C", "25ΔC", 0.25), ("10C", "10ΔC", 0.10))
CM_DAYS = (30, 61, 91, 182, 365)
CM_LABELS = {30: "1M", 61: "2M", 91: "3M", 182: "6M", 365: "1Y"}
SKEW_UNIT_K = 0.1                   # the grid's skew: vol pts per 0.1 of ln-moneyness
_MONO_GRID = np.linspace(K_LO, K_HI, 5001)
_MONO_TOL = 1e-12


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── one SVI slice ─────────────────────────────────────────────────────────────
def svi_w(k: Any, sl: Mapping) -> Any:
    """Raw-SVI total variance w(k) = a + b(ρ(k − m) + √((k − m)² + σ²))."""
    x = np.asarray(k, dtype=float) - sl["m"]
    out = sl["a"] + sl["b"] * (sl["rho"] * x + np.sqrt(x * x + sl["sigma"] ** 2))
    return float(out) if np.ndim(out) == 0 else out


def svi_w_prime(k: Any, sl: Mapping) -> Any:
    """dw/dk = b(ρ + (k − m)/√((k − m)² + σ²))."""
    x = np.asarray(k, dtype=float) - sl["m"]
    out = sl["b"] * (sl["rho"] + x / np.sqrt(x * x + sl["sigma"] ** 2))
    return float(out) if np.ndim(out) == 0 else out


def svi_atm_skew(sl: Mapping) -> float:
    """∂σ/∂k at k = 0, in vol (decimal) per unit ln-moneyness: w′(0)/(2√(w(0)·T))."""
    return float(svi_w_prime(0.0, sl)) / (2.0 * math.sqrt(float(svi_w(0.0, sl)) * sl["T"]))


def forward_call_delta(k: Any, w: Any) -> Any:
    """Undiscounted forward Black call delta N(d₁), d₁ = −k/√w + √w/2."""
    sw = np.sqrt(np.asarray(w, dtype=float))
    out = ndtr(-np.asarray(k, dtype=float) / sw + 0.5 * sw)
    return float(out) if np.ndim(out) == 0 else out


def forward_delta(k: float, sl: Mapping, call: bool) -> float:
    """The slice's forward delta at k: N(d₁) for a call, N(d₁) − 1 for a put."""
    d = float(forward_call_delta(k, svi_w(k, sl)))
    return d if call else d - 1.0


def delta_strike(sl: Mapping, target: float) -> float:
    """The k at which the slice's forward delta equals ``target`` (calls > 0, puts < 0).

    Root-finding with brentq on k ∈ [−3, 2]. The call delta must be
    non-increasing in k on that range (it is on a butterfly-free slice) and the
    target must be bracketed; otherwise this raises ValueError rather than
    clipping to an edge."""
    if not -1.0 < target < 1.0 or target == 0.0:
        raise ValueError(f"delta target {target} is not in (−1, 0) ∪ (0, 1)")
    want = target if target > 0 else 1.0 + target
    w = np.asarray(svi_w(_MONO_GRID, sl))
    if not np.all(w > 0):
        raise ValueError(f"slice {sl.get('expiry')}: total variance not positive on k ∈ [{K_LO}, {K_HI}]")
    d = forward_call_delta(_MONO_GRID, w)
    if np.any(np.diff(d) > _MONO_TOL):
        raise ValueError(f"slice {sl.get('expiry')}: forward delta is not monotone in k on [{K_LO}, {K_HI}]")

    def f(k: float) -> float:
        return float(forward_call_delta(k, svi_w(k, sl))) - want

    lo, hi = f(K_LO), f(K_HI)
    if not (lo > 0 > hi):
        raise ValueError(f"slice {sl.get('expiry')}: delta {target} not bracketed on [{K_LO}, {K_HI}]")
    return float(brentq(f, K_LO, K_HI, xtol=1e-14, rtol=1e-14, maxiter=500))


def rr_bf(vol: Mapping[str, Any], extrap: Mapping[str, bool], wing: str) -> tuple:
    """(RR, BF) of the ``wing`` ('25' or '10') pillars in vol points; (None, None)
    when a pillar is missing or extrapolated beyond the quoted strikes."""
    c, p, a = vol.get(wing + "C"), vol.get(wing + "P"), vol.get("ATM")
    if c is None or p is None or a is None or extrap.get(wing + "C") or extrap.get(wing + "P"):
        return None, None
    return c - p, 0.5 * (c + p) - a


# ── one session ───────────────────────────────────────────────────────────────
def quoted_k_range(residuals: Any, model: str = "svi_cal") -> dict[str, tuple[float, float]]:
    """{expiry: (min k, max k)} of the quotes scored for ``model`` (every model is
    scored on the same quotes; the market rows when the model is absent)."""
    q = residuals
    if "model" in q.columns and (q["model"] == model).any():
        q = q[q["model"] == model]
    g = q.groupby(q["expiry"].astype(str).str[:10])["k"].agg(["min", "max"])
    return {str(e): (float(r["min"]), float(r["max"])) for e, r in g.iterrows()}


def _strip_by_expiry(vix_rep: Mapping | None) -> dict[str, dict]:
    return {str(r.get("expiry"))[:10]: r for r in (vix_rep or {}).get("strip") or [] if isinstance(r, dict)}


def delta_grid(surface_json: Mapping, residuals: Any, vix_rep: Mapping | None) -> list[dict]:
    """One row per listed expiry of the SVI-cal surface, shortest first.

    Each row: expiry, T (years), T_days, forward, the pillars' k, vol (vol pts)
    and extrapolation flag, ATMF, RR25/BF25/RR10/BF10 (vol pts), the ATM skew
    (decimal vol per unit k, and ×10 in vol pts as ``skew_vp``), w(0) and w′(0)
    (for constant-maturity interpolation), the variance-swap vol of the same
    expiry from the replication strip, its spread over this ATMF and the
    strip's strike count (puts + calls), and the term structure's quote count
    and median half-spread."""
    svi_cal = (surface_json.get("surfaces") or {}).get("svi_cal") or {}
    slices = (svi_cal.get("params") or {}).get("slices") or []
    ts = {str(t.get("expiry"))[:10]: t for t in surface_json.get("term_structure") or []}
    kr = quoted_k_range(residuals)
    strip = _strip_by_expiry(vix_rep)
    rows = []
    for sl in sorted(slices, key=lambda s: float(s["T"])):
        e = str(sl["expiry"])[:10]
        T = float(sl["T"])
        k_lo, k_hi = kr.get(e, (None, None))
        ks: dict[str, float] = {}
        vol: dict[str, float] = {}
        extrap: dict[str, bool] = {}
        for key, _label, target in PILLARS:
            k = 0.0 if target is None else delta_strike(sl, target)
            ks[key] = k
            vol[key] = 100.0 * math.sqrt(float(svi_w(k, sl)) / T)
            # no quotes on record for the expiry: every pillar is beyond them
            extrap[key] = k_lo is None or k_hi is None or not (k_lo <= k <= k_hi)
        rr25, bf25 = rr_bf(vol, extrap, "25")
        rr10, bf10 = rr_bf(vol, extrap, "10")
        skew = svi_atm_skew(sl)
        t = ts.get(e) or {}
        st = strip.get(e)
        vs = _num((st or {}).get("var_swap_vol"))
        rows.append({
            "expiry": e, "T": T, "T_days": T * DAYS_PER_YEAR, "forward": _num(t.get("forward")),
            "k": ks, "vol": vol, "extrap": extrap, "atmf": vol["ATM"],
            "rr25": rr25, "bf25": bf25, "rr10": rr10, "bf10": bf10,
            "skew": skew, "skew_vp": skew * 100.0 * SKEW_UNIT_K,
            "w0": float(svi_w(0.0, sl)), "w0p": float(svi_w_prime(0.0, sl)),
            "vs": vs, "vs_minus_atmf": (vs - vol["ATM"]) if vs is not None else None,
            "n_strikes": ((st.get("n_puts") or 0) + (st.get("n_calls") or 0)) if st else None,
            "half_spread_vp": _num(t.get("median_half_spread_vp")),
            "n_quotes": int(t["n_quotes"]) if _num(t.get("n_quotes")) is not None else None,
            "k_lo": k_lo, "k_hi": k_hi})
    return rows


def _lin_w(v1: float | None, T1: float, v2: float | None, T2: float, T: float) -> float | None:
    """Vol (pts) at T, linear in total variance between (T1, v1) and (T2, v2)."""
    if v1 is None or v2 is None:
        return None
    w1, w2 = (v1 / 100.0) ** 2 * T1, (v2 / 100.0) ** 2 * T2
    a = 0.0 if T2 == T1 else (T - T1) / (T2 - T1)
    return 100.0 * math.sqrt(((1.0 - a) * w1 + a * w2) / T)


def bracket(grid: Sequence[Mapping], t: float) -> tuple[Mapping, Mapping] | None:
    """The listed rows (lower, upper) with lower.T ≤ t ≤ upper.T, nearest on each
    side; None outside the listed range (no extrapolation)."""
    rows = sorted(grid, key=lambda r: r["T"])
    if not rows or t < rows[0]["T"] - 1e-12 or t > rows[-1]["T"] + 1e-12:
        return None
    lower = max((r for r in rows if r["T"] <= t + 1e-12), key=lambda r: r["T"])
    upper = min((r for r in rows if r["T"] >= t - 1e-12), key=lambda r: r["T"])
    return lower, upper


def constant_maturity(grid: Sequence[Mapping], days: Iterable[int] = CM_DAYS) -> list[dict]:
    """Rows at fixed tenors (1M, 2M, 3M, 6M, 1Y by default) from the listed grid.

    Every pillar is interpolated linearly in total variance at fixed delta
    between the bracketing listed expiries (an exact match is taken as listed);
    the ATM skew from w(0) and w′(0) interpolated linearly in T; the
    variance-swap vol linearly in total variance, with the strike counts of
    the two strips it reads (``strikes``: lower, upper). A pillar extrapolated
    on either side stays flagged. A tenor outside the listed T range has
    ``available`` False and no values (never extrapolated)."""
    out = []
    for D in days:
        T = D / DAYS_PER_YEAR
        row: dict[str, Any] = {"days": D, "label": CM_LABELS.get(D, f"{D}d"), "T": T, "T_days": float(D),
                               "available": False, "lower": None, "upper": None, "forward": None,
                               "vol": {key: None for key, _, _ in PILLARS},
                               "extrap": {key: False for key, _, _ in PILLARS},
                               "atmf": None, "rr25": None, "bf25": None, "rr10": None, "bf10": None,
                               "skew": None, "skew_vp": None, "vs": None, "vs_minus_atmf": None,
                               "strikes": (None, None), "half_spread_vp": None, "n_quotes": None}
        br = bracket(grid, T)
        if br is not None:
            lo, hi = br
            row.update(available=True, lower=lo["expiry"], upper=hi["expiry"])
            for key, _, _ in PILLARS:
                row["vol"][key] = _lin_w(lo["vol"][key], lo["T"], hi["vol"][key], hi["T"], T)
                row["extrap"][key] = bool(lo["extrap"][key] or hi["extrap"][key])
            row["atmf"] = row["vol"]["ATM"]
            row["rr25"], row["bf25"] = rr_bf(row["vol"], row["extrap"], "25")
            row["rr10"], row["bf10"] = rr_bf(row["vol"], row["extrap"], "10")
            a = 0.0 if hi["T"] == lo["T"] else (T - lo["T"]) / (hi["T"] - lo["T"])
            w0 = (1.0 - a) * lo["w0"] + a * hi["w0"]
            w0p = (1.0 - a) * lo["w0p"] + a * hi["w0p"]
            row["skew"] = w0p / (2.0 * math.sqrt(w0 * T))
            row["skew_vp"] = row["skew"] * 100.0 * SKEW_UNIT_K
            row["vs"] = _lin_w(lo["vs"], lo["T"], hi["vs"], hi["T"], T)
            row["vs_minus_atmf"] = (row["vs"] - row["atmf"]) if row["vs"] is not None else None
            row["strikes"] = (lo.get("n_strikes"), hi.get("n_strikes"))
        out.append(row)
    return out


# ── the ATM-skew term structure ──────────────────────────────────────────────
SKEW_SLOPE_MIN_POINTS = 3


def skew_slope(t_years: Sequence[float], skews: Sequence[float],
               min_days: float | None = None) -> dict | None:
    """OLS slope of log|skew| on log T (a power law |skew| ∝ T^slope), with the
    number of expiries and their T range in days; expiries shorter than
    ``min_days`` are left out. None with fewer than three usable expiries.

    A descriptive slope of one fitted surface: inside a rough-volatility model
    the skew scales as T^(H − 1/2), but this slope is not an estimate of H."""
    pts = [(float(t), abs(float(s))) for t, s in zip(t_years, skews)
           if _num(t) is not None and _num(s) is not None and float(t) > 0 and float(s) != 0
           and (min_days is None or float(t) * DAYS_PER_YEAR >= min_days - 1e-9)]
    if len(pts) < SKEW_SLOPE_MIN_POINTS:
        return None
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    slope = float(np.polyfit(x, y, 1)[0])
    days = [p[0] * DAYS_PER_YEAR for p in pts]
    return {"slope": slope, "n": len(pts), "t_lo_days": min(days), "t_hi_days": max(days)}


# ── one session against an earlier one ───────────────────────────────────────
CM_DELTA_KEYS = ("atmf", "rr25", "bf25")


def cm_delta(new: Sequence[Mapping], old: Sequence[Mapping]) -> list[dict]:
    """Cell-by-cell change of two constant-maturity grids (``new`` − ``old``) in
    ATMF, RR25 and BF25, vol points, per tenor of ``new``. A change is None when
    either session lacks the tenor or the value (an extrapolated RR/BF)."""
    by_old = {r["days"]: r for r in old}
    out = []
    for r in new:
        o = by_old.get(r["days"]) or {}
        row: dict[str, Any] = {"days": r["days"], "label": r["label"],
                               "available": bool(r.get("available") and o.get("available"))}
        for key in CM_DELTA_KEYS:
            a, b = r.get(key), o.get(key)
            row[key] = {"new": a, "old": b, "delta": (a - b) if (a is not None and b is not None) else None}
        out.append(row)
    return out


# ── forward ATM vol between constant maturities ──────────────────────────────
FWD_MIN_DAYS = 7.0      # an AM/PM pair (10-29 / 10-30) is never a forward period


def forward_vols(points: Sequence[Mapping], min_days: float = FWD_MIN_DAYS) -> list[dict]:
    """Forward ATM vol between consecutive constant-maturity points (1M → 2M →
    3M → 6M → 1Y): σ_fwd = √((w₂ − w₁)/(T₂ − T₁)) with w = ATMF²·T, in vol pts.

    A point closer than ``min_days`` to the previous one is skipped (the next
    pair spans further). A negative forward variance (w falling with T: calendar
    arbitrage) is never a NaN: its vol is None and the item carries the flag
    'calendar'."""
    pts = [p for p in points if p.get("available", True) and p.get("atmf") is not None]
    out: list[dict] = []
    prev = None
    for p in pts:
        if prev is not None and (p["T"] - prev["T"]) * DAYS_PER_YEAR < min_days:
            continue
        if prev is not None:
            w1 = (prev["atmf"] / 100.0) ** 2 * prev["T"]
            w2 = (p["atmf"] / 100.0) ** 2 * p["T"]
            fvar = (w2 - w1) / (p["T"] - prev["T"])
            out.append({"from": prev["label"], "to": p["label"], "atm1": prev["atmf"], "atm2": p["atmf"],
                        "t1_days": prev["T"] * DAYS_PER_YEAR,
                        "t2_days": p["T"] * DAYS_PER_YEAR, "fwd_var": fvar,
                        "fwd_vol": 100.0 * math.sqrt(fvar) if fvar >= 0 else None,
                        "flag": None if fvar >= 0 else "calendar"})
        prev = p
    return out


# ── the VIX family replicated from the same chain ────────────────────────────
VIX_FAMILY = ("VIX9D", "VIX", "VIX3M", "VIX6M")
GAP_SPLIT_TOL = 1e-6    # index points: the two parts must add up to the stored gap


def _term(t: Any) -> dict | None:
    if not isinstance(t, Mapping) or _num(t.get("days")) is None:
        return None
    return {"expiry": str(t.get("expiry"))[:10], "days": float(t["days"])}


def vix_terms(vix_rep: Mapping | None) -> list[dict]:
    """Per VIX-family index of vix_replication.json: the two listed expiries its
    replication interpolates and the near term's weight, and the gap to the Cboe
    close split the way the artifact's gap_attribution splits it.

    The terms are those of the index's headline method; an index that is not
    replicated shows the Cboe-rules pair it would have used, whose
    ``extrapolated`` flag says why it was refused (a near weight outside [0, 1]),
    or no pair and the method's own ``reason`` (e.g. fewer than two usable
    terms).

    gap_attribution records official − replicated = truncation_pts +
    residual_pts. In the page's sign (gap = replicated − Cboe close) the parts
    are −truncation_pts, the missing-strike shortfall (replicated − full strip),
    valued on the attribution surface's wing, and −residual_pts, what the chain
    leaves unexplained. The split is None when the artifact has no attribution
    for the index or its parts do not add up to the stored gap within
    GAP_SPLIT_TOL; ``far_wing`` is the stored quotes beyond the surface's fitted
    range minus that wing (index points), the sign of the truncation's bias."""
    out: list[dict] = []
    for tick in VIX_FAMILY:
        ix = ((vix_rep or {}).get("indices") or {}).get(tick)
        if not isinstance(ix, Mapping):
            continue
        method = ix.get("headline_method")
        terms = ix.get(method) if method else ix.get("cboe_rules")
        terms = terms if isinstance(terms, Mapping) else {}
        gap = _num(ix.get("gap_vs_official"))
        replicated = _num(ix.get("replicated")) is not None
        ga_raw = ix.get("gap_attribution")
        ga: Mapping = ga_raw if isinstance(ga_raw, Mapping) else {}
        trunc, resid = _num(ga.get("truncation_pts")), _num(ga.get("residual_pts"))
        split = (replicated and bool(ga.get("available")) and gap is not None and trunc is not None
                 and resid is not None and abs(-trunc - resid - gap) <= GAP_SPLIT_TOL)
        out.append({"ticker": tick, "days": _num(terms.get("constant_maturity_days")), "method": method,
                    "replicated": replicated,
                    "near": _term(terms.get("near")), "next": _term(terms.get("next")),
                    "near_weight": _num(terms.get("near_weight")),
                    "extrapolated": bool(terms.get("extrapolated")),
                    "gap": gap if replicated else None,
                    "truncation": -trunc if split and trunc is not None else None,
                    "unexplained": -resid if split and resid is not None else None,
                    "far_wing": _num(ga.get("far_wing_market_minus_model_pts")) if split else None,
                    "fit_inside": _num(ga.get("fit_inside_range_pts")) if split else None,
                    "fit_range": _fit_range(ga.get("fit_range_K_over_F")) if split else None,
                    "surface": ga.get("surface") if split else None,
                    "reason": str(terms["reason"]) if (not replicated and terms.get("reason")) else None})
    return out


def _fit_range(fr: Any) -> tuple[float, float] | None:
    """gap_attribution.fit_range_K_over_F as (low, high), or None."""
    if not isinstance(fr, Sequence) or isinstance(fr, str) or len(fr) != 2:
        return None
    lo, hi = _num(fr[0]), _num(fr[1])
    return (lo, hi) if lo is not None and hi is not None else None


# the producer's gap-attribution sentence (surface/report.py gap notes): it
# states the split as official − replicated, against the page's convention
GAP_NOTE_PREFIX = "Gap attribution ("


def vix_gap_notes(notes: Iterable[Any] | None, terms: Sequence[Mapping]) -> list[str]:
    """vix_replication.json's gap_notes with the producer's gap-attribution
    sentence rebuilt from ``terms`` (vix_terms rows, optionally carrying a
    ``surface_label``), so the prose gives the VIX split the #vix-terms-table
    prints, on the page's convention: gap = replicated − Cboe close =
    missing-strike shortfall (replicated − full strip) + unexplained.

    The producer's sentence is dropped whatever the split; the rebuilt one is
    added only where vix_terms shows the VIX split (it adds up to the gap)."""
    out = [str(n) for n in (notes or []) if not str(n).startswith(GAP_NOTE_PREFIX)]
    r = next((t for t in terms if t.get("ticker") == "VIX"), None)
    if r is None or r.get("truncation") is None or r.get("unexplained") is None or r.get("gap") is None:
        return out
    label = r.get("surface_label") or r.get("surface") or "fitted"
    gap, short, unexp = r["gap"], r["truncation"], r["unexplained"]
    fr = r.get("fit_range")
    fitted = f"the fitted K/F {fr[0]:.2f}–{fr[1]:.2f}" if fr else "the fitted range"
    s = (f"Gap attribution (VIX, {label} surface; gap = replicated − Cboe close): "
         f"{signed(gap)} = missing-strike shortfall {signed(short)} + unexplained "
         f"{signed(unexp)} (each part rounded on its own). Missing-strike shortfall "
         f"(replicated − full strip): what the replication lacks without the strikes beyond "
         f"the stored strip, valued on the surface’s wing.")
    fw = r.get("far_wing")
    if fw is not None:
        if abs(fw) < 0.005:
            s += f" The stored quotes beyond {fitted} price in line with that wing."
        else:
            s += (f" The stored quotes beyond {fitted} price {abs(fw):.2f} index points "
                  f"{'above' if fw > 0 else 'below'} that wing, so the shortfall likely "
                  f"{'understates' if fw > 0 else 'overstates'} what the strip lacks.")
    fi = r.get("fit_inside")
    if fi is not None:
        s += (f" Inside {fitted}, market minus surface is {signed(fi)}; with the far-wing "
              f"term it sets the replication against the surface and is not part of the gap.")
    share = f", {100.0 * unexp / gap:.0f}% of the gap" if abs(gap) > 1e-12 else ""
    s += (f" Unexplained by the chain: {signed(unexp)}{share} (snapshot timing against the "
          f"16:15 ET print, stale quotes, the rate).")
    out.append(s)
    return out
