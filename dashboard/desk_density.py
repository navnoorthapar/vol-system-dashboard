"""
desk_density.py — the risk-neutral density and tail probabilities of the
certified SVI-cal surface (pure functions; no Flask, no I/O).

Breeden–Litzenberger: the undiscounted forward price of a call, C(K) =
E[(S_T − K)⁺], has ∂²C/∂K² = the density of S_T, and −∂C/∂K = P(S_T > K). On
one fitted SVI slice, with k = ln(K/F), total variance w(k) = σ(k)²·T and the
Black call priced on it, both derivatives are closed forms in w, w′ and w″
(Gatheral & Jacquier 2014, "Arbitrage-free SVI volatility surfaces", §2):

  * density of x = ln(S_T/F):  p(k) = g(k) / √(2π·w(k)) · exp(−d₋(k)²/2),
    d₋ = −k/√w − √w/2, where g is Durrleman's
    g(k) = (1 − k·w′/(2w))² − (w′²/4)·(1/w + 1/4) + w″/2,
    the function the surface report's butterfly certificate proves ≥ 0;
  * P(S_T < K) = N(−d₋) + φ(d₋)·w′(k) / (2√w(k)) (the slope of the put price;
    the vol skew adds the second term to the flat-smile N(−d₋)).

Conventions (the page states them):
  * probabilities are risk-neutral, under the T-forward measure of each
    expiry; they price claims, they do not forecast; F is the expiry's
    parity forward of the surface report, so a move is measured from F, not
    from spot;
  * P(S_T < K) is a function of the fitted smile at K alone (its level and
    slope), so a probability is printed only for a threshold K inside the
    expiry's quoted ln(K/F) range; one beyond it is "beyond quoted strikes";
  * the mass below the lowest quoted strike (and above the highest) is the
    same closed form at the range's edge: a number set by the quotes there.
    How that mass is spread beyond the edge is the SVI linear wing, an
    extrapolation, so the density is drawn only inside the quoted range and
    no moment beyond the mean is reported (skewness and kurtosis are set by
    the wings);
  * the checks ∫p = 1 and E[S_T/F] = ∫eᵏp = 1 integrate the slice over the
    whole line (wings included): they test the slice as a distribution (Lee's
    moment bounds and the Rogers–Tehranchi large-moneyness limit), not the
    quotes.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import desk_surface
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

DAYS_PER_YEAR = desk_surface.DAYS_PER_YEAR
SQRT_2PI = math.sqrt(2.0 * math.pi)
# (key, label, side, K/F): below = P(S_T < K), above = P(S_T > K)
THRESHOLDS: tuple[tuple[str, str, str, float], ...] = (
    ("lt80", "S_T < 0.80 F", "below", 0.80),
    ("lt90", "S_T < 0.90 F", "below", 0.90),
    ("gt110", "S_T > 1.10 F", "above", 1.10),
)
BAND = (0.95, 1.05)            # P(S_T outside 0.95–1.05 F)
CHART_TENORS = (30, 91, 182)   # the density chart: the listed expiries nearest 1M, 3M and 6M
CHART_LABELS = {30: "1M", 91: "3M", 182: "6M"}
TENOR_TOL = 0.35               # a listed expiry stands for a tenor within 35% of its days
CHART_STEP = 0.005             # k spacing of the drawn density
INTEGRATION_RANGE = (-40.0, 40.0)   # p and eᵏp are below 1e-30 well inside it on every stored slice


def svi_w2(k: Any, sl: Mapping) -> Any:
    """d²w/dk² = b·σ² / ((k − m)² + σ²)^{3/2}."""
    x = np.asarray(k, dtype=float) - sl["m"]
    out = sl["b"] * sl["sigma"] ** 2 / (x * x + sl["sigma"] ** 2) ** 1.5
    return float(out) if np.ndim(out) == 0 else out


def durrleman_g(k: Any, sl: Mapping) -> Any:
    """g(k) = (1 − k·w′/(2w))² − (w′²/4)(1/w + 1/4) + w″/2 of the slice."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(desk_surface.svi_w(k, sl))
    w1 = np.asarray(desk_surface.svi_w_prime(k, sl))
    w2 = np.asarray(svi_w2(k, sl))
    out = (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    return float(out) if np.ndim(out) == 0 else out


def _d_minus(k: Any, w: Any) -> Any:
    sw = np.sqrt(w)
    return -np.asarray(k, dtype=float) / sw - 0.5 * sw


def density(k: Any, sl: Mapping) -> Any:
    """Risk-neutral density of ln(S_T/F) at k (per unit of ln-moneyness)."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(desk_surface.svi_w(k, sl))
    dm = _d_minus(k, w)
    out = np.asarray(durrleman_g(k, sl)) / (SQRT_2PI * np.sqrt(w)) * np.exp(-0.5 * dm * dm)
    return float(out) if np.ndim(out) == 0 else out


def prob_below(k: float, sl: Mapping) -> float:
    """P(S_T < F·eᵏ) = N(−d₋) + φ(d₋)·w′/(2√w): the slope of the slice's put price."""
    w = float(desk_surface.svi_w(k, sl))
    sw = math.sqrt(w)
    dm = float(_d_minus(k, w))
    w1 = float(desk_surface.svi_w_prime(k, sl))
    return float(ndtr(-dm)) + math.exp(-0.5 * dm * dm) / SQRT_2PI * w1 / (2.0 * sw)


def _integral(f: Any, sl: Mapping) -> float:
    """∫f over INTEGRATION_RANGE, split where the density lives (around ±6 total vols)."""
    sd = math.sqrt(max(float(desk_surface.svi_w(0.0, sl)), 1e-12))
    lo, hi = INTEGRATION_RANGE
    cuts = sorted({lo, max(lo, -12.0 * sd), max(lo, -6.0 * sd), 0.0,
                   min(hi, 6.0 * sd), min(hi, 12.0 * sd), hi})
    return float(sum(quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
                     for a, b in itertools.pairwise(cuts)))


def checks(sl: Mapping) -> dict:
    """{mass: ∫p dk, mean: E[S_T/F] = ∫eᵏ·p dk} over the whole line (wings included)."""
    def p(k: float) -> float:
        return float(density(k, sl))

    def ep(k: float) -> float:
        return math.exp(k) * p(k)

    return {"mass": _integral(p, sl), "mean": _integral(ep, sl)}


def _prob(sl: Mapping, kk: float, side: str, k_lo: float | None, k_hi: float | None) -> float | None:
    if k_lo is None or k_hi is None or not (k_lo <= kk <= k_hi):
        return None                                   # beyond quoted strikes
    pb = prob_below(kk, sl)
    return pb if side == "below" else 1.0 - pb


def tail_rows(surface_json: Mapping, residuals: Any) -> list[dict]:
    """One row per listed SVI-cal slice, shortest first: expiry, T (years), T_days,
    the quoted ln(K/F) range (and as K/F), each THRESHOLDS probability and
    P(outside BAND) (None when a threshold lies beyond the quoted strikes), the
    mass below the lowest and above the highest quoted strike, the slice's
    certificate (certified, g_min) and the checks (∫p, E[S_T/F])."""
    svi_cal = (surface_json.get("surfaces") or {}).get("svi_cal") or {}
    slices = (svi_cal.get("params") or {}).get("slices") or []
    kr = desk_surface.quoted_k_range(residuals)
    rows = []
    for sl in sorted(slices, key=lambda s: float(s["T"])):
        e = str(sl["expiry"])[:10]
        k_lo, k_hi = kr.get(e, (None, None))
        probs = {key: _prob(sl, math.log(x), side, k_lo, k_hi) for key, _, side, x in THRESHOLDS}
        b_lo, b_hi = (_prob(sl, math.log(x), side, k_lo, k_hi) for x, side in zip(BAND, ("below", "above")))
        probs["out5"] = (b_lo + b_hi) if (b_lo is not None and b_hi is not None) else None
        rows.append({
            "expiry": e, "T": float(sl["T"]), "T_days": float(sl["T"]) * DAYS_PER_YEAR,
            "k_lo": k_lo, "k_hi": k_hi,
            "kf_lo": math.exp(k_lo) if k_lo is not None else None,
            "kf_hi": math.exp(k_hi) if k_hi is not None else None,
            "prob": probs,
            "below_quoted": prob_below(k_lo, sl) if k_lo is not None else None,
            "above_quoted": (1.0 - prob_below(k_hi, sl)) if k_hi is not None else None,
            "certified": bool(sl.get("certified")), "g_min": desk_surface._num(sl.get("g_min")),
            **checks(sl)})
    return rows


def chart_expiries(rows: Sequence[Mapping], tenors: Sequence[int] = CHART_TENORS) -> list[dict]:
    """The certified listed expiry nearest each tenor (within TENOR_TOL of its
    days), each expiry once: [{tenor, label, row}]. A listed expiry, not an
    interpolation, so every curve is one butterfly-certified slice."""
    out: list[dict] = []
    used: set[str] = set()
    for D in tenors:
        cands = [r for r in rows if r["expiry"] not in used and r["k_lo"] is not None
                 and r.get("certified", True) and abs(r["T_days"] - D) <= TENOR_TOL * D]
        if not cands:
            continue
        r = min(cands, key=lambda r: abs(r["T_days"] - D))
        used.add(r["expiry"])
        out.append({"tenor": D, "label": CHART_LABELS.get(D, f"{D}d"), "row": r})
    return out


def density_curve(sl: Mapping, k_lo: float, k_hi: float, step: float = CHART_STEP) -> list[list[float]]:
    """[[k, p(k)], …] on [k_lo, k_hi] (both ends included), the quoted range only."""
    n = max(2, math.ceil((k_hi - k_lo) / step) + 1)
    ks = np.linspace(k_lo, k_hi, n)
    return [[float(k), float(v)] for k, v in zip(ks, np.asarray(density(ks, sl)))]


def slice_of(surface_json: Mapping, expiry: str) -> Mapping:
    svi_cal = (surface_json.get("surfaces") or {}).get("svi_cal") or {}
    slices = (svi_cal.get("params") or {}).get("slices") or []
    return next(s for s in slices if str(s["expiry"])[:10] == expiry)


_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def fmt_g(x: float | None) -> str:
    """The certificate's g_min as printed: 4 decimals from 0.001 up, else
    'm.m × 10⁻ⁿ' (a value of 10⁻⁸ is positive, which '0.0000' would hide)."""
    if x is None or not math.isfinite(x):
        return "—"
    if x == 0.0 or abs(x) >= 1e-3:
        return format(x, ".4f").replace("-", "−")
    e = math.floor(math.log10(abs(x)))
    m = x / 10.0 ** e
    if round(abs(m), 1) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{m:.1f}".replace("-", "−") + " × 10" + str(e).translate(_SUPER)
