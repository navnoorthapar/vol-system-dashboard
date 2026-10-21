"""
svi.py — raw SVI per expiry under no-arbitrage constraints.

Raw SVI (Gatheral 2004; Gatheral & Jacquier, arXiv:1204.0646, eq. 3.1):

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)),

total implied variance at log-moneyness k = ln(K/F_T).

Constraints imposed per slice (SLSQP):
  * butterfly: Durrleman's g(k) >= 1e-8 (GJ Lemma 2.2, with a small margin:
    SLSQP leaves binding rows at about -1e-10) on SVI_GRID = 481 points on k
    in [-3, 2] (K/F from 0.05 to 7.4, spacing 0.0104; far wider than every
    quoted SPX strike range in the stored chains), plus cutting planes
    (below). There is no closed-form parameter condition for raw SVI (GJ
    sec. 3), hence the rows. Until C23f the rows covered only [-1.5, 1]
    (241 points); with the certificate below, short slices then chased a
    negative-g pocket out in (1, 3.3] round after round (2026-03-27,
    2026-04-06: g = -3.3e-5 at k = 3.28, moving 0.13 inward per cut) and
    fell back to their eSSVI start, while rows over all of [-3, 2] certify
    every slice of the five stored sessions (184 of 184, none at its start;
    summed objective 731.4 -> 695.8 against the uncertified fits). Rows on a
    1/k grid beyond [-3, 2] were tried and made SLSQP stall on two slices;
    the tails are left to the cutting planes. The fixed wide grid is
    deliberate: SVI fits a short-dated quoted range best in its near-parabolic
    regime (large sigma, m at its bound), which
    turns g negative far outside the quotes (2026-03-24, 31-day slice: the
    unconstrained optimum has g < 0 on the grid, and imposing g >= 0 raises
    the mean squared half-spread error from 4.1 to 6.8). A grid that only
    widened each quoted range (by half its span, at least z in [-6, 4]) let
    SLSQP wander into that degenerate regime on 2026-03-24 (iteration limits,
    incompatible constraints, a 39 vp RMSE), so it was rejected.
  * wings: b (1 + rho) < 2 and b (1 - rho) < 2 (Roger Lee's moment bound;
    with them g(k) -> 1/4 - (b(1+-rho))^2/16 > 0 as |k| -> inf and
    d_+(k) -> -inf, GJ Lemma 2.2's second condition)
  * min_k w(k) = a + b sigma sqrt(1 - rho^2) >= 0

Certificates and cutting planes. Rows at grid nodes are not a certificate:
with g >= 0 imposed only at the 241 nodes the constraint binds at the
optimum and g dips below zero between nodes (C23 review: g = -2.6e-5 at
k = 0.190, K/F 1.21, between nodes 0.1875 and 0.1979, 2026-10-16; 16 slices
on 3 of 5 sessions; ordering crossings of -1.5e-6 at k = 6.6 and of -5.6e-7
at k = 40 off the old check grid). A slice is therefore accepted only when
it passes butterfly_certificate (and, against its neighbours,
ordering_certificate), each evaluated on CERT_GRID (k in [-3, 2] at 1e-4
spacing; tails on a 1/k grid out to k = 2,000 and -3,000) with every
discrete local minimum refined by a golden-section search on its two
neighbouring cells, plus the |k| -> inf limits. Where a certificate fails,
the refined minimisers become new constraint rows (g rows, ordering rows,
scaled by 30 / |k| beyond |k| = 30 so that far rows stay well conditioned,
and the asymptotic-intercept row of a wing whose slopes tie) and the slice
is re-solved from its start, at most _MAX_CUT_ROUNDS times; a slice that
never certifies keeps its start (an eSSVI slice, certified by Gatheral-
Jacquier Theorem 4.2, or the previously accepted fit). These are numerical
certificates on the stated grids, not closed-form ones.

Two surfaces are built from these slices:

fit_svi_surface — per-expiry SVI, no calendar constraint between slices: the
  best-fitting butterfly-free smile per expiry. Its calendar violations are
  measured and reported (metrics.dense_grid_checks).

fit_svi_calendar_surface — the same slices fitted under calendar ordering
  w_{i-1}(k) <= w_i(k) <= w_{i+1}(k) against BOTH neighbours (a "sandwich"),
  by block coordinate descent over the slices, started from the fitted
  eSSVI surface. The eSSVI start is butterfly-free and calendar-ordered, so
  every slice fit starts feasible, and each accepted update keeps the whole
  surface feasible: a slice is only replaced by a fit that passes
  butterfly_certificate and ordering_certificate against its current
  neighbours. ordering_certificate: w_{i+1}(k) - w_i(k) >= -1e-10 (total
  variance) on CERT_GRID with refined minima; the wing slopes b (1 +- rho)
  non-decreasing to 1e-10 (the |k| -> inf order); and, on a wing whose slopes
  tie, the asymptotic intercepts a - b (1 + rho) m (right) and
  a + b (1 - rho) m (left) ordered to 1e-10 (the limit of w_{i+1} - w_i). A
  slope shortfall within the 1e-10 tolerance is not extrapolated linearly
  over the 1/k grid (it is treated as a tie). Ordering rows are imposed on
  CALENDAR_ROWS_GRID and at the cutting planes. Between quoted expiries
  total variance is interpolated linearly in T, which keeps the order.
  A sequential variant constrained only above the previous fitted slice was
  tried first and rejected: on 2026-03-27 a few short slices ended
  infeasible (the eSSVI start lay below the previous SVI slice) and the error
  propagated forward (inside-spread share 54% -> 14%).

The objective matches essvi.fit_essvi: the mean squared implied-vol residual
in half-spread units (weights = 1 / max(half-spread, floor)). Each slice
starts from the corresponding slice of a fitted eSSVI surface (an SSVI slice
is a raw SVI slice: GJ sec. 4, a = theta (1 - rho^2)/2, b = theta phi / 2,
m = -rho / phi, sigma = sqrt(1 - rho^2) / phi, rho unchanged), which is
butterfly- and calendar-free.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from joint_vol_calibration.surface._blas import blas_single_threaded
from joint_vol_calibration.surface.essvi import CHECK_GRID, durrleman_g

SVI_GRID = np.linspace(-3.0, 2.0, 481)          # butterfly rows (spacing 0.0104)
CALENDAR_ROWS_GRID = np.unique(np.concatenate([
    np.linspace(-1.5, 1.0, 51), np.linspace(-0.3, 0.3, 31),
    [-30.0, -10.0, -3.0, -2.5, -2.0, 1.5, 2.0, 10.0, 30.0]]))
_LEE_SLACK = 1e-6
_ORDER_TOL = 1e-10            # w_{i+1} - w_i >= -tol (and wing slopes, intercepts): ordered
_G_TOL = 0.0                  # butterfly certificate: g >= -tol everywhere checked
_G_MARGIN = 1e-8              # g rows in the fit: g >= margin at every row
_FEAS_TOL = 1e-9              # max constraint violation of an accepted slice
_MAX_CUT_ROUNDS = 6           # cutting-plane re-solves per slice fit
_FAR_ROW_K = 30.0             # ordering rows beyond |k| = 30 are scaled by 30 / |k|

# Certificate grid: k in [-3, 2] at 1e-4 spacing, and the tails on a grid
# uniform in 1/k out to k = 2,000 (right) and -3,000 (left).
_CERT_U = np.linspace(1.0, 1e-3, 1000)[1:]
CERT_CORE = (-3.0, 2.0)
CERT_GRID = np.concatenate([CERT_CORE[0] / _CERT_U[::-1],
                            np.linspace(CERT_CORE[0], CERT_CORE[1], 50001),
                            CERT_CORE[1] / _CERT_U])
_GOLD = (math.sqrt(5.0) - 1.0) / 2.0
_GOLD_ITERS = 48              # bracket shrinks by 0.618^48 = 1e-10 of its width
_BOUNDS = [(-1.0, 1.0),        # a
           (0.0, 2.0),         # b
           (-0.999, 0.999),    # rho
           (-1.0, 1.0),        # m
           (1e-4, 2.0)]        # sigma


def svi_w(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    d = k - m
    return a + b * (rho * d + np.sqrt(d * d + sigma * sigma))


def svi_derivatives(k, a, b, rho, m, sigma):
    """(w, dw/dk, d2w/dk2) of a raw SVI slice."""
    k = np.asarray(k, dtype=float)
    d = k - m
    S = np.sqrt(d * d + sigma * sigma)
    w = a + b * (rho * d + S)
    w1 = b * (rho + d / S)
    w2 = b * sigma * sigma / (S * S * S)
    return w, w1, w2


def svi_g(k, params):
    w, w1, w2 = svi_derivatives(k, *params)
    return durrleman_g(k, w, w1, w2)


def svi_param_jacobians(k, a, b, rho, m, sigma):
    """d(w, w', w'')/d(a, b, rho, m, sigma) on grid k: three (len(k), 5) arrays."""
    k = np.asarray(k, dtype=float)
    d = k - m
    S = np.sqrt(d * d + sigma * sigma)
    S3 = S ** 3
    S5 = S ** 5
    one, zero = np.ones_like(k), np.zeros_like(k)
    jw = np.stack([one, rho * d + S, b * d, b * (-rho - d / S), b * sigma / S], axis=1)
    jw1 = np.stack([zero, rho + d / S, b * one, -b * sigma * sigma / S3,
                    -b * d * sigma / S3], axis=1)
    jw2 = np.stack([zero, sigma * sigma / S3, zero, 3.0 * b * sigma * sigma * d / S5,
                    b * (2.0 * sigma / S3 - 3.0 * sigma ** 3 / S5)], axis=1)
    return jw, jw1, jw2


def svi_g_jacobian(k, params):
    """(g(k), dg/d(a, b, rho, m, sigma)) of a raw SVI slice on grid k."""
    k = np.asarray(k, dtype=float)
    w, w1, w2 = svi_derivatives(k, *params)
    jw, jw1, jw2 = svi_param_jacobians(k, *params)
    u = 1.0 - k * w1 / (2.0 * w)
    g = u * u - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2
    dg_dw = 2.0 * u * k * w1 / (2.0 * w * w) + 0.25 * w1 * w1 / (w * w)
    dg_dw1 = -u * k / w - 0.5 * w1 * (1.0 / w + 0.25)
    return g, dg_dw[:, None] * jw + dg_dw1[:, None] * jw1 + 0.5 * jw2


def wing_slopes(params) -> tuple:
    """(right, left) asymptotic slopes of w: b (1 + rho), b (1 - rho)."""
    b, rho = params[1], params[2]
    return b * (1.0 + rho), b * (1.0 - rho)


def wing_intercepts(params) -> tuple:
    """(right, left) asymptotic intercepts of w: w(k) - b (1 + rho) k ->
    a - b (1 + rho) m as k -> +inf, and w(k) - b (1 - rho) |k| ->
    a + b (1 - rho) m as k -> -inf."""
    a, b, rho, m = params[0], params[1], params[2], params[3]
    return a - b * (1.0 + rho) * m, a + b * (1.0 - rho) * m


# ── numerical no-arbitrage certificates ──────────────────────────────────────

def _refined_minima(f, x, v) -> tuple:
    """Every discrete local minimum of v = f(x) on the sorted grid x (one per
    plateau; the grid ends count), refined by a golden-section search over
    its two neighbouring cells, vectorised over the minima. Returns (k, f(k)):
    per minimum the lower of the refined and the grid value."""
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(x)
    left = np.concatenate([[np.inf], v[:-1]])
    right = np.concatenate([v[1:], [np.inf]])
    idx = np.flatnonzero((v < left) & (v <= right))
    if not len(idx):
        idx = np.array([int(np.argmin(v))])
    a, b = x[np.maximum(idx - 1, 0)], x[np.minimum(idx + 1, n - 1)]
    c, d = b - _GOLD * (b - a), a + _GOLD * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(_GOLD_ITERS):
        lo_side = fc < fd                        # the minimum lies in [a, d]
        b = np.where(lo_side, d, b)
        a = np.where(lo_side, a, c)
        new = np.where(lo_side, b - _GOLD * (b - a), a + _GOLD * (b - a))
        fn = f(new)
        c, d = np.where(lo_side, new, d), np.where(lo_side, c, new)
        fc, fd = np.where(lo_side, fn, fd), np.where(lo_side, fc, fn)
    xr, fr = np.where(fc < fd, c, d), np.minimum(fc, fd)
    better = fr < v[idx]
    return np.where(better, xr, x[idx]), np.where(better, fr, v[idx])


def _finite_or_neg_inf(f):
    def g(k):
        with np.errstate(divide="ignore", invalid="ignore"):
            y = f(k)
        return np.where(np.isfinite(y), y, -np.inf)
    return g


def _butterfly_check(params, grid=CERT_GRID) -> tuple:
    """(summary, cut points) of butterfly_certificate."""
    p = tuple(float(v) for v in params)
    f = _finite_or_neg_inf(lambda k: svi_g(k, p))
    v = f(grid)
    ks, gs = _refined_minima(f, grid, v)
    r, l_ = wing_slopes(p)
    limit = float(min(0.25 - r * r / 16.0, 0.25 - l_ * l_ / 16.0))
    j = int(np.argmin(gs))
    lo, hi = CERT_CORE
    core_grid = (grid >= lo) & (grid <= hi)
    core_min = (ks >= lo) & (ks <= hi)
    g_core = float(min(v[core_grid].min(), gs[core_min].min() if core_min.any() else np.inf))
    gmin = float(min(gs[j], limit))
    summary = {"ok": bool(gmin >= -_G_TOL), "g_min": gmin,
               "k_at_g_min": float(ks[j]) if gs[j] <= limit else None,
               "g_min_k_-3_to_2": g_core, "g_limit_abs_k_to_inf": limit}
    cuts = ks[(gs < -_G_TOL) & np.isfinite(gs)]
    return summary, cuts


def butterfly_certificate(params) -> dict:
    """Durrleman's g >= 0 for one raw SVI slice over the real line, as a
    numerical certificate: g on CERT_GRID (k in [-3, 2] at 1e-4 spacing,
    tails uniform in 1/k out to k = 2,000 and -3,000), every discrete local
    minimum refined by golden-section search, and the limit
    g(+-inf) = 1/4 - (b (1 +- rho))^2 / 16. ok: the minimum is >= -_G_TOL (0).
    Also g_min_k_-3_to_2, the minimum restricted to the core range."""
    return _butterfly_check(params)[0]


def _ordering_check(lower, upper, grid=CERT_GRID) -> tuple:
    """(summary, cut points, tied wings with a crossing intercept) of
    ordering_certificate."""
    lo = tuple(float(v) for v in lower)
    hi = tuple(float(v) for v in upper)
    rl, ll = wing_slopes(lo)
    rh, lh = wing_slopes(hi)
    dsr, dsl = rh - rl, lh - ll
    fr, fl = max(-dsr, 0.0), max(-dsl, 0.0)        # a tied slope's shortfall (<= tol when ok)

    def d(k):
        k = np.asarray(k, dtype=float)
        return (svi_w(k, *hi) - svi_w(k, *lo)
                + fr * np.maximum(k, 0.0) + fl * np.maximum(-k, 0.0))

    f = _finite_or_neg_inf(d)
    ks, ds = _refined_minima(f, grid, f(grid))
    irl, ill = wing_intercepts(lo)
    irh, ilh = wing_intercepts(hi)
    tie_r, tie_l = abs(dsr) <= _ORDER_TOL, abs(dsl) <= _ORDER_TOL
    lim_r = irh - irl if tie_r else None
    lim_l = ilh - ill if tie_l else None
    j = int(np.argmin(ds))
    lims = [x for x in (lim_r, lim_l) if x is not None]
    ok = bool(dsr >= -_ORDER_TOL and dsl >= -_ORDER_TOL and ds[j] >= -_ORDER_TOL
              and all(x >= -_ORDER_TOL for x in lims))
    summary = {"ok": ok, "min_dw": float(ds[j]), "k_at_min_dw": float(ks[j]),
               "wing_slope_increments": [float(dsr), float(dsl)],
               "tied_wing_intercept_increments": [None if lim_r is None else float(lim_r),
                                                  None if lim_l is None else float(lim_l)]}
    cuts = ks[(ds < -_ORDER_TOL) & np.isfinite(ds)]
    crossing_intercepts = (lim_r is not None and lim_r < -_ORDER_TOL,
                           lim_l is not None and lim_l < -_ORDER_TOL)
    return summary, cuts, crossing_intercepts


def ordering_certificate(lower, upper) -> dict:
    """Calendar order w_lower(k) <= w_upper(k) of two raw SVI slices over the
    real line, as a numerical certificate: the wing slopes b (1 +- rho)
    non-decreasing to _ORDER_TOL (1e-10); on a wing whose slopes tie to
    1e-10, the asymptotic intercept (wing_intercepts, the limit of
    w_upper - w_lower) >= -1e-10; and w_upper - w_lower >= -1e-10 on CERT_GRID
    with every discrete local minimum refined (a tied slope's shortfall is
    not extrapolated linearly over the tail grid)."""
    return _ordering_check(lower, upper)[0]


def _certify(params, lower=None, upper=None) -> tuple:
    """(ok, g cut points, ordering cut points, intercept rows to add): the
    butterfly certificate of ``params`` and its ordering certificate against
    each given neighbour. Intercept rows are (neighbour, wing) pairs, e.g.
    ("lower", "right")."""
    bfly, g_cuts = _butterfly_check(params)
    ok = bfly["ok"]
    c_cuts, rows = [], set()
    for which, pair in (("lower", (lower, params)), ("upper", (params, upper))):
        if pair[0] is None or pair[1] is None:
            continue
        o, cc, (cr, cl) = _ordering_check(*pair)
        ok = ok and o["ok"]
        c_cuts.append(cc)
        if cr:
            rows.add((which, "right"))
        if cl:
            rows.add((which, "left"))
    c = np.concatenate(c_cuts) if c_cuts else np.zeros(0)
    return bool(ok), g_cuts, c, rows


def ssvi_slice_to_raw(theta: float, psi: float, rho: float) -> tuple:
    """Raw SVI parameters (a, b, rho, m, sigma) of an (e)SSVI slice."""
    phi = psi / theta
    return (theta * (1.0 - rho * rho) / 2.0, theta * phi / 2.0, rho,
            -rho / phi, math.sqrt(1.0 - rho * rho) / phi)


@dataclass
class SVISlice:
    expiry: str
    T: float
    params: tuple                      # (a, b, rho, m, sigma)
    info: dict = field(default_factory=dict)

    def w(self, k):
        return svi_w(k, *self.params)

    def iv(self, k):
        return np.sqrt(np.maximum(self.w(k), 0.0) / self.T)


@dataclass
class SVISurface:
    """Per-expiry SVI slices; between slices total variance is interpolated
    linearly in T at fixed k (non-decreasing in T wherever the two slices are
    ordered; no butterfly certificate in between); before the first slice w
    scales by T/T_1, after the last slice by T/T_N."""
    slices: list
    name: str = "SVI"

    @property
    def T(self) -> np.ndarray:
        return np.array([s.T for s in self.slices])

    def w(self, k, T: float):
        Ts = self.T
        T = float(T)
        if T <= Ts[0]:  # noqa: SIM300 (T is a maturity, not a constant)
            return self.slices[0].w(k) * T / Ts[0]
        if T >= Ts[-1]:  # noqa: SIM300 (T is a maturity, not a constant)
            return self.slices[-1].w(k) * T / Ts[-1]
        i = int(np.searchsorted(Ts, T, side="right") - 1)
        lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        return (1 - lam) * self.slices[i].w(k) + lam * self.slices[i + 1].w(k)

    def iv(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / float(T))

    def to_dict(self) -> dict:
        names = ("a", "b", "rho", "m", "sigma")
        return {"model": self.name, "slices": [
            {"expiry": s.expiry, "T": float(s.T),
             **{n: float(v) for n, v in zip(names, s.params)},
             "success": bool(s.info.get("success", False)),
             "certified": s.info.get("certified"),
             "g_min": s.info.get("g_min")} for s in self.slices]}

    @classmethod
    def from_dict(cls, d: dict) -> SVISurface:
        names = ("a", "b", "rho", "m", "sigma")
        return cls([SVISlice(s["expiry"], s["T"], tuple(s[n] for n in names))
                    for s in d["slices"]], name=d.get("model", "SVI"))

    def butterfly(self) -> dict:
        """butterfly_certificate of every slice, summarised."""
        rows = [butterfly_certificate(s.params) for s in self.slices]
        if not rows:
            return {"certified": True, "slices_with_g_negative": 0}
        j = int(np.argmin([r["g_min"] for r in rows]))
        bad = [s.expiry for s, r in zip(self.slices, rows) if not r["ok"]]
        return {"certified": not bad, "g_min": rows[j]["g_min"],
                "k_at_g_min": rows[j]["k_at_g_min"], "expiry_at_g_min": self.slices[j].expiry,
                "slices_with_g_negative": len(bad), "expiries_with_g_negative": bad,
                "g_min_k_-3_to_2": float(min(r["g_min_k_-3_to_2"] for r in rows)),
                "slices_with_g_negative_on_k_-3_to_2": int(sum(
                    r["g_min_k_-3_to_2"] < -_G_TOL for r in rows)),
                "g_limit_abs_k_to_inf_min": float(min(r["g_limit_abs_k_to_inf"] for r in rows)),
                "certificate": ("svi.butterfly_certificate: Durrleman g on CERT_GRID (k in [-3, 2] "
                                "at 1e-4 spacing; tails uniform in 1/k out to k = 2,000 and "
                                "-3,000), every local minimum refined by golden-section search, "
                                "and the |k| -> inf limit 1/4 - (b(1 +- rho))^2/16"),
                "tolerance_g": _G_TOL}

    def ordering(self, grid=CHECK_GRID) -> dict:
        """Calendar order of consecutive slices: ordering_certificate per
        pair, summarised; plus the minimum of w_{i+1} - w_i on ``grid``
        (essvi.CHECK_GRID: the pre-C23f check, kept for comparison)."""
        pairs = list(zip(self.slices[:-1], self.slices[1:]))
        dgrid, rows = None, []
        for s1, s2 in pairs:
            d = float(np.min(s2.w(grid) - s1.w(grid)))
            dgrid = d if dgrid is None else min(dgrid, d)
            rows.append(ordering_certificate(s1.params, s2.params))
        out: dict = {"ordered": all(r["ok"] for r in rows), "min_dw_on_check_grid": dgrid,
                     "certificate": ("svi.ordering_certificate: w_{i+1} - w_i >= -1e-10 on "
                                     "CERT_GRID (k in [-3, 2] at 1e-4 spacing; tails uniform in "
                                     "1/k out to k = 2,000 and -3,000) with every local minimum "
                                     "refined; wing slopes b(1 +- rho) non-decreasing to 1e-10; "
                                     "on a wing whose slopes tie, the asymptotic intercept "
                                     "ordered to 1e-10"),
                     "check_grid": "essvi.CHECK_GRID (4,001 points on k in [-3, 2] plus |k| = 10, 30)",
                     "tolerance_total_variance": _ORDER_TOL}
        if rows:
            j = int(np.argmin([r["min_dw"] for r in rows]))
            ties = [x for r in rows for x in r["tied_wing_intercept_increments"] if x is not None]
            out.update({"min_dw": rows[j]["min_dw"], "k_at_min_dw": rows[j]["k_at_min_dw"],
                        "pair_at_min_dw": [pairs[j][0].expiry, pairs[j][1].expiry],
                        "min_wing_slope_increment": float(min(min(r["wing_slope_increments"])
                                                              for r in rows)),
                        "tied_wings": len(ties),
                        "min_tied_wing_intercept_increment": float(min(ties)) if ties else None,
                        "pairs_unordered": [[a.expiry, b.expiry] for (a, b), r in zip(pairs, rows)
                                            if not r["ok"]]})
        else:
            out.update({"min_dw": None, "min_wing_slope_increment": None, "pairs_unordered": []})
        return out

    def certificate(self) -> dict:
        """{'butterfly': butterfly(), 'calendar': ordering(), 'arbitrage_free': both certified}."""
        b, c = self.butterfly(), self.ordering()
        return {"butterfly": b, "calendar": c,
                "arbitrage_free": bool(b["certified"] and c["ordered"])}


def _objective(x, k, T, mid, inv_h):
    a, b, rho, m, sig = x
    d = k - m
    S = np.sqrt(d * d + sig * sig)
    w = np.maximum(a + b * (rho * d + S), 1e-14)
    iv = np.sqrt(w / T)
    r = (iv - mid) * inv_h
    n = len(k)
    f = float(np.dot(r, r)) / n
    c = 2.0 * r * inv_h / n / (2.0 * iv * T)
    grad = np.array([
        c.sum(),
        np.dot(c, rho * d + S),
        np.dot(c, b * d),
        np.dot(c, b * (-rho - d / S)),
        np.dot(c, b * sig / S),
    ])
    return f, grad


def _intercept_jacobian(x, wing: str) -> np.ndarray:
    _a, b, rho, m, _sig = x
    if wing == "right":                            # a - b (1 + rho) m
        return np.array([1.0, -(1.0 + rho) * m, -b * m, -b * (1.0 + rho), 0.0])
    return np.array([1.0, (1.0 - rho) * m, -b * m, b * (1.0 - rho), 0.0])   # a + b (1 - rho) m


def _constraints(grid, lower=None, upper=None, cal_grid=CALENDAR_ROWS_GRID, *,
                 g_margin: float = 0.0, intercepts=()):
    """Inequality constraints c(x) >= 0 of one slice and their Jacobian:
    g - ``g_margin`` on ``grid``, the two Lee wing bounds, min w >= 0 and,
    against a ``lower`` / ``upper`` neighbour slice (raw SVI params), the
    ordering rows on ``cal_grid`` (rows beyond |k| = 30 scaled by 30 / |k|,
    which keeps far cutting planes well conditioned and leaves the feasible
    set unchanged), the wing-slope order and the asymptotic-intercept rows
    named in ``intercepts`` ((neighbour, wing) pairs, e.g. ("lower",
    "right"); see module docstring)."""
    lee = 2.0 * (1.0 - _LEE_SLACK)
    cal_grid = np.asarray(cal_grid, dtype=float)
    ak = np.abs(cal_grid)
    row_scale = np.where(ak > _FAR_ROW_K, _FAR_ROW_K / np.maximum(ak, _FAR_ROW_K), 1.0)
    nb = {"lower": lower, "upper": upper}
    w_lo = None if lower is None else svi_w(cal_grid, *lower)
    w_hi = None if upper is None else svi_w(cal_grid, *upper)
    s_lo = None if lower is None else wing_slopes(lower)
    s_hi = None if upper is None else wing_slopes(upper)
    icpt = sorted((w_, wing) for w_, wing in intercepts if nb.get(w_) is not None)

    def fun(x):
        a, b, rho, _m, sig = x
        parts = [svi_g(grid, x) - g_margin,
                 [lee - b * (1 + rho), lee - b * (1 - rho),
                  a + b * sig * math.sqrt(max(1 - rho * rho, 0.0))]]
        if lower is not None or upper is not None:
            w = svi_w(cal_grid, *x)
            r, l_ = wing_slopes(x)
            if lower is not None:
                parts += [row_scale * (w - w_lo), [r - s_lo[0], l_ - s_lo[1]]]
            if upper is not None:
                parts += [row_scale * (w_hi - w), [s_hi[0] - r, s_hi[1] - l_]]
            ir, il = wing_intercepts(x)
            for which, wing in icpt:
                n_r, n_l = wing_intercepts(nb[which])
                mine, theirs = (ir, n_r) if wing == "right" else (il, n_l)
                parts.append([mine - theirs if which == "lower" else theirs - mine])
        return np.concatenate(parts)

    def jac(x):
        _a, b, rho, _m, sig = x
        _, jg = svi_g_jacobian(grid, x)
        sq = math.sqrt(max(1 - rho * rho, 0.0))
        rows = [jg,
                np.array([[0.0, -(1 + rho), -b, 0.0, 0.0],
                          [0.0, -(1 - rho), b, 0.0, 0.0],
                          [1.0, sig * sq, (-b * sig * rho / sq) if sq > 0 else 0.0, 0.0, b * sq]])]
        if lower is not None or upper is not None:
            jw, _, _ = svi_param_jacobians(cal_grid, *x)
            jw = row_scale[:, None] * jw
            js = np.array([[0.0, 1 + rho, b, 0.0, 0.0], [0.0, 1 - rho, -b, 0.0, 0.0]])
            if lower is not None:
                rows += [jw, js]
            if upper is not None:
                rows += [-jw, -js]
            for which, wing in icpt:
                ji = _intercept_jacobian(x, wing)
                rows.append((ji if which == "lower" else -ji)[None, :])
        return np.vstack(rows)

    return fun, jac


def fit_svi_slice(k, T, mid_iv, half_spread, x0, *, h_floor: float = 5e-4,
                  grid=SVI_GRID, maxiter: int = 500, lower=None, upper=None,
                  cal_grid=CALENDAR_ROWS_GRID, max_cut_rounds: int = _MAX_CUT_ROUNDS) -> tuple:
    """Fit one raw SVI slice from the start ``x0`` (feasible for the
    constraints, e.g. an eSSVI slice); ``lower`` / ``upper``: neighbour slices
    the fit must stay above / below (ordering rows on ``cal_grid``, the
    wing-slope order and, where needed, asymptotic-intercept rows).

    SLSQP runs on scaled variables (a by the slice's ATM total variance, b and
    sigma by their start values, rho by 0.1, m by 0.05) with the objective
    divided by its start value; unscaled, SLSQP's line search failed on most
    short slices. Constraint Jacobians are analytic; g rows carry the margin
    _G_MARGIN (1e-8). SLSQP can also stop at a degenerate stationary point
    with b -> 0 (a flat smile: the gradient in rho, m and sigma vanishes with
    b; seen on a 2026-03-27 slice, 13.5 vp RMSE): when the result is not
    better than the start or b collapses, the fit is rerun with 10x smaller
    scales.

    Cutting planes: a run is certified when its rows hold (max violation
    <= 1e-9, no margins) and it passes butterfly_certificate and, against
    each neighbour, ordering_certificate. While an uncertified run beats
    every certified candidate (the start included), the refined minimisers
    where its certificates fail (and the intercept rows of tied wings that
    cross) are added as rows and the slice is re-solved from ``x0``, at most
    ``max_cut_rounds`` times. The returned slice is the best
    certified of the runs and the start; when none is certified, the
    least-violating candidate with info['certified'] = False. The parameter
    box (_BOUNDS) is widened to contain ``x0``, so the start is used as
    given. Returns (params, info)."""
    k = np.asarray(k, dtype=float)
    mid = np.asarray(mid_iv, dtype=float)
    inv_h = 1.0 / np.maximum(np.asarray(half_spread, dtype=float), h_floor)
    x0 = np.asarray(x0, dtype=float)
    grid = np.asarray(grid, dtype=float)
    cal_grid = np.asarray(cal_grid, dtype=float)
    lo = np.minimum(np.array([b_[0] for b_ in _BOUNDS]), x0)
    hi = np.maximum(np.array([b_[1] for b_ in _BOUNDS]), x0)
    f0, _ = _objective(x0, k, T, mid, inv_h)
    fs = max(f0, 1e-12)
    theta_atm = max(float(svi_w(0.0, *x0)), 1e-6)
    sc0 = np.array([theta_atm, max(x0[1], 1e-3), 0.1, 0.05, max(x0[4], 1e-3)])

    def run(sc, cfun_m, cjac_m, cfun0):
        def f_scaled(u):
            f, g = _objective(x0 + u * sc, k, T, mid, inv_h)
            return f / fs, g * sc / fs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(f_scaled, np.zeros(5), jac=True, method="SLSQP",
                           bounds=list(zip((lo - x0) / sc, (hi - x0) / sc)),
                           constraints=[{"type": "ineq", "fun": lambda u: cfun_m(x0 + u * sc),
                                         "jac": lambda u: cjac_m(x0 + u * sc) * sc[None, :]}],
                           options={"maxiter": maxiter, "ftol": 1e-12})
        x = np.clip(x0 + res.x * sc, lo, hi)
        f, _ = _objective(x, k, T, mid, inv_h)
        return {"x": x, "f": f, "viol": float(max(0.0, -cfun0(x).min())), "res": res}

    g_rows, c_rows = grid, cal_grid
    icpt: set[tuple[str, str]] = set()
    cfun0, _ = _constraints(g_rows, lower, upper, c_rows)
    start = {"x": x0, "f": f0, "viol": float(max(0.0, -cfun0(x0).min())), "res": None}
    start["certified"] = bool(_certify(x0, lower, upper)[0] and start["viol"] <= _FEAS_TOL)
    best_cert_f = f0 if start["certified"] else np.inf
    all_runs: list = []
    rounds = 0
    for rnd in range(max_cut_rounds + 1):
        cfun_m, cjac_m = _constraints(g_rows, lower, upper, c_rows, g_margin=_G_MARGIN,
                                      intercepts=icpt)
        cfun0, _ = _constraints(g_rows, lower, upper, c_rows, intercepts=icpt)
        runs = [run(sc0, cfun_m, cjac_m, cfun0)]
        if not (runs[0]["f"] < f0) or runs[0]["x"][1] < 1e-5:
            runs.append(run(0.1 * sc0, cfun_m, cjac_m, cfun0))
        all_runs += runs
        new_g, new_c, new_i = [], [], set()
        for r in runs:
            ok, gc, cc, ic = _certify(r["x"], lower, upper)
            r["certified"] = bool(ok and np.isfinite(r["f"]) and r["viol"] <= _FEAS_TOL)
            if r["certified"]:
                best_cert_f = min(best_cert_f, r["f"])
            else:
                r["cuts"] = (gc, cc, ic)
        # keep cutting while an uncertified run beats every certified candidate
        pending = [r for r in runs if not r["certified"] and np.isfinite(r["f"])
                   and r["f"] < best_cert_f]
        if not pending or rnd == max_cut_rounds:
            break
        for r in pending:
            gc, cc, ic = r["cuts"]
            new_g.append(gc)
            new_c.append(cc)
            new_i |= ic
        add_g = np.setdiff1d(np.concatenate(new_g), g_rows)
        add_c = np.setdiff1d(np.concatenate(new_c), c_rows)
        add_i = new_i - icpt
        if not (len(add_g) or len(add_c) or add_i):
            break
        rounds += 1
        g_rows = np.unique(np.concatenate([g_rows, add_g]))
        c_rows = np.unique(np.concatenate([c_rows, add_c]))
        icpt |= add_i
    feasible = [r for r in [*all_runs, start] if r["certified"]]
    best = (min(feasible, key=lambda r: r["f"]) if feasible
            else min([*all_runs, start], key=lambda r: r["viol"]))
    res = all_runs[0]["res"] if best["res"] is None else best["res"]
    x = best["x"]
    info = {"success": best["res"] is not None and bool(best["res"].success),
            "message": str(res.message), "nit": int(sum(r["res"].nit for r in all_runs)),
            "runs": len(all_runs), "objective_start": f0, "objective": best["f"],
            "max_constraint_violation": best["viol"], "returned_start": best["res"] is None,
            "certified": bool(best["certified"]), "cut_rounds": rounds,
            "butterfly_cut_rows": int(len(g_rows) - len(grid)),
            "calendar_cut_rows": int(len(c_rows) - len(cal_grid)),
            "intercept_rows": [list(t) for t in sorted(icpt)],
            "g_min": butterfly_certificate(x)["g_min"],
            "g_min_on_rows": float(svi_g(g_rows, x).min()),
            "grid_k": [float(grid[0]), float(grid[-1])]}
    return tuple(float(v) for v in x), info


@blas_single_threaded
def fit_svi_surface(groups, essvi_surface, *, h_floor: float = 5e-4,
                    grid=SVI_GRID) -> SVISurface:
    """Fit every expiry. ``groups`` = iterable of (expiry, T, k, mid_iv,
    half_spread) in increasing T; each slice starts from the eSSVI slice at
    the same T."""
    slices = []
    for expiry, T, k, mid, hs in groups:
        x0 = ssvi_slice_to_raw(*essvi_surface.params_at(T))
        params, info = fit_svi_slice(k, T, mid, hs, x0, h_floor=h_floor, grid=grid)
        slices.append(SVISlice(str(expiry), float(T), params, info))
    return SVISurface(slices)


@blas_single_threaded
def fit_svi_calendar_surface(groups, essvi_surface, *, h_floor: float = 5e-4,
                             grid=SVI_GRID, max_sweeps: int = 4,
                             rel_tol: float = 1e-4, starts=None) -> tuple:
    """Calendar-ordered SVI slices by block coordinate descent (module
    docstring). ``groups`` as for fit_svi_surface. Sweeps alternate forward /
    backward over the slices and stop when a sweep lowers the summed slice
    objective by less than ``rel_tol`` (relative) or after ``max_sweeps``.

    Each slice update is solved from the slice's current parameters (the
    eSSVI slice in the first sweep) and from ``starts[i]`` when given (e.g.
    the free per-slice SVI fit), and the best certified result is kept: one
    SLSQP start per update left slices stuck behind their neighbours
    (2026-03-27, 2026-12-31 slice: objective 1.166 from its current
    parameters, 0.648 from its free fit, both certified).

    Returns (SVISurface named "SVI-cal", info)."""
    groups = list(groups)
    n = len(groups)
    data = [(str(e), float(T), np.asarray(k, float), np.asarray(m, float), np.asarray(h, float))
            for e, T, k, m, h in groups]
    params = [tuple(float(v) for v in ssvi_slice_to_raw(*essvi_surface.params_at(T)))
              for _, T, *_ in data]
    extra = [[tuple(float(v) for v in starts[i])] if starts is not None and starts[i] is not None
             else [] for i in range(n)]
    inv = [1.0 / np.maximum(h, h_floor) for *_, h in data]
    objs = [_objective(np.asarray(params[i]), data[i][2], data[i][1], data[i][3], inv[i])[0]
            for i in range(n)]
    start_total = float(sum(objs))
    infos: list[dict] = [{} for _ in range(n)]
    n_rejected = n_cut_rounds = n_fits = 0
    sweeps = 0
    totals = [start_total]
    for sweep in range(max_sweeps):
        order = range(n) if sweep % 2 == 0 else range(n - 1, -1, -1)
        for i in order:
            e, T, k, mid, hs = data[i]
            lower = params[i - 1] if i > 0 else None
            upper = params[i + 1] if i < n - 1 else None
            best = None
            for x0 in dict.fromkeys([params[i], *extra[i]]):        # distinct starts, in order
                p_new, info = fit_svi_slice(k, T, mid, hs, x0, h_floor=h_floor, grid=grid,
                                            lower=lower, upper=upper)
                n_cut_rounds += info["cut_rounds"]
                n_fits += 1
                if info["certified"] and (best is None or info["objective"] < best[1]["objective"]):
                    best = (p_new, info)
            if best is None:
                n_rejected += 1
            elif best[1]["objective"] <= objs[i]:
                params[i], objs[i] = best[0], best[1]["objective"]
                infos[i] = best[1]
        sweeps += 1
        totals.append(float(sum(objs)))
        if totals[-2] - totals[-1] <= rel_tol * max(totals[-2], 1e-12):
            break
    slices = []
    for i, (e, T, *_rest) in enumerate(data):
        inf = dict(infos[i]) if infos[i] else {"success": False, "returned_start": True,
                                               "objective": objs[i]}
        inf["g_min"] = butterfly_certificate(params[i])["g_min"]
        inf["certified"] = _certify(params[i], params[i - 1] if i else None,
                                    params[i + 1] if i < n - 1 else None)[0]
        inf["grid_k"] = [float(grid[0]), float(grid[-1])]
        inf["objective"] = objs[i]
        slices.append(SVISlice(e, T, params[i], inf))
    surf = SVISurface(slices, name="SVI-cal")
    cert = surf.certificate()
    info = {"sweeps": sweeps, "summed_objective_by_sweep": totals,
            "objective_start_essvi": start_total, "objective": float(sum(objs)),
            "slices_kept_at_essvi_start": int(sum(1 for x in infos if not x)),
            "slice_fits": n_fits, "updates_without_a_certified_fit": n_rejected,
            "cutting_plane_rounds": n_cut_rounds,
            "ordering": cert["calendar"], "butterfly": cert["butterfly"],
            "arbitrage_free_certified": cert["arbitrage_free"],
            "max_constraint_violation": float(max(
                max(0.0, -float(_constraints(grid, params[i - 1] if i else None,
                                             params[i + 1] if i < n - 1 else None)[0](
                    np.asarray(params[i])).min())) for i in range(n)))}
    return surf, info
