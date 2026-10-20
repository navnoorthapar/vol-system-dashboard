"""
svi.py — raw SVI per expiry under no-arbitrage constraints.

Raw SVI (Gatheral 2004; Gatheral & Jacquier, arXiv:1204.0646, eq. 3.1):

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)),

total implied variance at log-moneyness k = ln(K/F_T).

Constraints imposed per slice (SLSQP):
  * butterfly: Durrleman's g(k) >= 0 (GJ Lemma 2.2) on a dense grid,
    SVI_GRID = 241 points on k in [-1.5, 1.0] (K/F from 0.22 to 2.72, wider
    than every quoted SPX strike range in the stored chains). There is no
    closed-form parameter condition for raw SVI (GJ sec. 3), hence the grid.
    The fixed wide grid is deliberate: SVI fits a short-dated quoted range
    best in its near-parabolic regime (large sigma, m at its bound), which
    turns g negative far outside the quotes (2026-03-24, 31-day slice: the
    unconstrained optimum has g < 0 on the grid, and imposing g >= 0 raises
    the mean squared half-spread error from 4.1 to 6.8). A grid that only
    widened each quoted range (by half its span, at least z in [-6, 4]) let
    SLSQP wander into that degenerate regime on 2026-03-24 (iteration limits,
    incompatible constraints, a 39 vp RMSE), so it was rejected.
  * wings: b (1 + rho) < 2 and b (1 - rho) < 2 (Roger Lee's moment bound;
    with them g(k) -> 1/4 - (b(1+-rho))^2/16 >= 0 as |k| -> inf and
    d_+(k) -> -inf, GJ Lemma 2.2's second condition)
  * min_k w(k) = a + b sigma sqrt(1 - rho^2) >= 0
Two surfaces are built from these slices:

fit_svi_surface — per-expiry SVI, no calendar constraint between slices: the
  best-fitting butterfly-free smile per expiry. Its calendar violations are
  measured and reported (metrics.dense_grid_checks).

fit_svi_calendar_surface — the same slices fitted under calendar ordering
  w_{i-1}(k) <= w_i(k) <= w_{i+1}(k) against BOTH neighbours (a "sandwich"),
  by block coordinate descent over the slices, started from the fitted
  eSSVI surface. The eSSVI start is butterfly-free and calendar-ordered, so
  every slice fit starts feasible, and each accepted update keeps the whole
  surface feasible: a slice is only replaced by a fit that is ordered against
  its current neighbours on essvi.CHECK_GRID (4,001 points on k in [-3, 2]
  plus |k| = 10, 30, tolerance 1e-10 in total variance) and in the wing
  slopes b (1 +- rho) (the |k| -> inf order). Ordering rows are imposed on
  CALENDAR_ROWS_GRID and, where the check grid still finds a crossing,
  at the crossing points (cutting planes, at most 3 rounds per slice). The
  ordering certificate is a grid certificate. Between quoted expiries total
  variance is interpolated linearly in T, which keeps the order.
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

SVI_GRID = np.linspace(-1.5, 1.0, 241)
CALENDAR_ROWS_GRID = np.unique(np.concatenate([
    np.linspace(-1.5, 1.0, 51), np.linspace(-0.3, 0.3, 31),
    [-30.0, -10.0, -3.0, -2.5, -2.0, 1.5, 2.0, 10.0, 30.0]]))
_LEE_SLACK = 1e-6
_ORDER_TOL = 1e-10            # w_{i+1} - w_i >= -tol on CHECK_GRID: ordered
_FEAS_TOL = 1e-9              # max constraint violation of an accepted slice
_MAX_CUT_ROUNDS = 3
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
             "g_min_on_grid": s.info.get("g_min"),
             "butterfly_grid_k": s.info.get("grid_k")} for s in self.slices]}

    @classmethod
    def from_dict(cls, d: dict) -> SVISurface:
        names = ("a", "b", "rho", "m", "sigma")
        return cls([SVISlice(s["expiry"], s["T"], tuple(s[n] for n in names))
                    for s in d["slices"]], name=d.get("model", "SVI"))

    def ordering(self, grid=CHECK_GRID) -> dict:
        """Calendar order of consecutive slices: min over the grid of
        w_{i+1} - w_i, and the wing-slope increments (>= 0 holds)."""
        dmin, smin = None, None
        for s1, s2 in zip(self.slices[:-1], self.slices[1:]):
            d = float(np.min(s2.w(grid) - s1.w(grid)))
            r1, l1 = wing_slopes(s1.params)
            r2, l2 = wing_slopes(s2.params)
            sl = min(r2 - r1, l2 - l1)
            dmin = d if dmin is None else min(dmin, d)
            smin = sl if smin is None else min(smin, sl)
        return {"min_dw_on_check_grid": dmin, "min_wing_slope_increment": smin,
                "ordered": bool((dmin is None or dmin >= -_ORDER_TOL)
                                and (smin is None or smin >= -_ORDER_TOL)),
                "check_grid": "essvi.CHECK_GRID (4,001 points on k in [-3, 2] plus |k| = 10, 30)",
                "tolerance_total_variance": _ORDER_TOL}


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


def _constraints(grid, lower=None, upper=None, cal_grid=CALENDAR_ROWS_GRID):
    """Inequality constraints c(x) >= 0 of one slice and their Jacobian:
    g on ``grid``, the two Lee wing bounds, min w >= 0 and, against a
    ``lower`` / ``upper`` neighbour slice (raw SVI params), the ordering rows
    on ``cal_grid`` and the wing-slope order (see module docstring)."""
    lee = 2.0 * (1.0 - _LEE_SLACK)
    cal_grid = np.asarray(cal_grid, dtype=float)
    w_lo = None if lower is None else svi_w(cal_grid, *lower)
    w_hi = None if upper is None else svi_w(cal_grid, *upper)
    s_lo = None if lower is None else wing_slopes(lower)
    s_hi = None if upper is None else wing_slopes(upper)

    def fun(x):
        a, b, rho, _m, sig = x
        parts = [svi_g(grid, x),
                 [lee - b * (1 + rho), lee - b * (1 - rho),
                  a + b * sig * math.sqrt(max(1 - rho * rho, 0.0))]]
        if lower is not None or upper is not None:
            w = svi_w(cal_grid, *x)
            r, l_ = wing_slopes(x)
            if lower is not None:
                parts += [w - w_lo, [r - s_lo[0], l_ - s_lo[1]]]
            if upper is not None:
                parts += [w_hi - w, [s_hi[0] - r, s_hi[1] - l_]]
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
            js = np.array([[0.0, 1 + rho, b, 0.0, 0.0], [0.0, 1 - rho, -b, 0.0, 0.0]])
            if lower is not None:
                rows += [jw, js]
            if upper is not None:
                rows += [-jw, -js]
        return np.vstack(rows)

    return fun, jac


def fit_svi_slice(k, T, mid_iv, half_spread, x0, *, h_floor: float = 5e-4,
                  grid=SVI_GRID, maxiter: int = 500, lower=None, upper=None,
                  cal_grid=CALENDAR_ROWS_GRID) -> tuple:
    """Fit one raw SVI slice from the start ``x0`` (feasible for the
    constraints, e.g. an eSSVI slice); ``lower`` / ``upper``: neighbour slices
    the fit must stay above / below (ordering rows on ``cal_grid`` and the
    wing-slope order).

    SLSQP runs on scaled variables (a by the slice's ATM total variance, b and
    sigma by their start values, rho by 0.1, m by 0.05) with the objective
    divided by its start value; unscaled, SLSQP's line search failed on most
    short slices. Constraint Jacobians are analytic. SLSQP can also stop at a
    degenerate stationary point with b -> 0 (a flat smile: the gradient in
    rho, m and sigma vanishes with b; seen on a 2026-03-27 slice, 13.5 vp
    RMSE): when the result is not better than the start or b collapses, the
    fit is rerun with 10x smaller scales. The returned slice is the best
    feasible (max constraint violation <= 1e-9) of the runs and the start.
    The parameter box (_BOUNDS) is widened to contain ``x0``, so the start is
    used as given. Returns (params, info)."""
    k = np.asarray(k, dtype=float)
    mid = np.asarray(mid_iv, dtype=float)
    inv_h = 1.0 / np.maximum(np.asarray(half_spread, dtype=float), h_floor)
    x0 = np.asarray(x0, dtype=float)
    lo = np.minimum(np.array([b_[0] for b_ in _BOUNDS]), x0)
    hi = np.maximum(np.array([b_[1] for b_ in _BOUNDS]), x0)
    cfun, cjac = _constraints(grid, lower, upper, cal_grid)
    f0, _ = _objective(x0, k, T, mid, inv_h)
    fs = max(f0, 1e-12)
    theta_atm = max(float(svi_w(0.0, *x0)), 1e-6)
    sc0 = np.array([theta_atm, max(x0[1], 1e-3), 0.1, 0.05, max(x0[4], 1e-3)])

    def run(sc):
        def f_scaled(u):
            f, g = _objective(x0 + u * sc, k, T, mid, inv_h)
            return f / fs, g * sc / fs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = minimize(f_scaled, np.zeros(5), jac=True, method="SLSQP",
                           bounds=list(zip((lo - x0) / sc, (hi - x0) / sc)),
                           constraints=[{"type": "ineq", "fun": lambda u: cfun(x0 + u * sc),
                                         "jac": lambda u: cjac(x0 + u * sc) * sc[None, :]}],
                           options={"maxiter": maxiter, "ftol": 1e-12})
        x = np.clip(x0 + res.x * sc, lo, hi)
        f, _ = _objective(x, k, T, mid, inv_h)
        return {"x": x, "f": f, "viol": float(max(0.0, -cfun(x).min())), "res": res}

    runs = [run(sc0)]
    if not (runs[0]["f"] < f0) or runs[0]["x"][1] < 1e-5:
        runs.append(run(0.1 * sc0))
    start = {"x": x0, "f": f0, "viol": float(max(0.0, -cfun(x0).min())), "res": None}
    feasible = [r for r in [*runs, start] if np.isfinite(r["f"]) and r["viol"] <= _FEAS_TOL]
    best = min(feasible, key=lambda r: r["f"]) if feasible else min([*runs, start], key=lambda r: r["viol"])
    res = runs[0]["res"] if best["res"] is None else best["res"]
    x = best["x"]
    info = {"success": best["res"] is not None and bool(best["res"].success),
            "message": str(res.message), "nit": int(sum(r["res"].nit for r in runs)),
            "runs": len(runs), "objective_start": f0, "objective": best["f"],
            "max_constraint_violation": best["viol"], "returned_start": best["res"] is None,
            "g_min": float(svi_g(grid, x).min()),
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


def _ordered_against(params, lower, upper, grid=CHECK_GRID) -> tuple:
    """(ok, crossing k points): ``params`` against its neighbours on ``grid``
    and in the wing slopes. The crossing points are the per-run minima of the
    ordering gap, used as cutting planes."""
    w = svi_w(grid, *params)
    pts, ok = [], True
    r, l_ = wing_slopes(params)
    for nb, sign in ((lower, 1.0), (upper, -1.0)):
        if nb is None:
            continue
        d = sign * (w - svi_w(grid, *nb))
        rn, ln = wing_slopes(nb)
        if sign * (r - rn) < -_ORDER_TOL or sign * (l_ - ln) < -_ORDER_TOL:
            ok = False
        neg = d < -_ORDER_TOL
        if neg.any():
            ok = False
            idx = np.where(neg)[0]
            cuts = np.where(np.diff(idx) > 1)[0]
            for a_, b_ in zip(np.concatenate([[idx[0]], idx[cuts + 1]]),
                              np.concatenate([idx[cuts], [idx[-1]]])):
                j = a_ + int(np.argmin(d[a_:b_ + 1]))
                pts.extend([float(grid[a_]), float(grid[j]), float(grid[b_])])
    return ok, np.unique(np.asarray(pts, dtype=float))


@blas_single_threaded
def fit_svi_calendar_surface(groups, essvi_surface, *, h_floor: float = 5e-4,
                             grid=SVI_GRID, max_sweeps: int = 4,
                             rel_tol: float = 1e-4) -> tuple:
    """Calendar-ordered SVI slices by block coordinate descent (module
    docstring). ``groups`` as for fit_svi_surface. Sweeps alternate forward /
    backward over the slices and stop when a sweep lowers the summed slice
    objective by less than ``rel_tol`` (relative) or after ``max_sweeps``.

    Returns (SVISurface named "SVI-cal", info)."""
    groups = list(groups)
    n = len(groups)
    data = [(str(e), float(T), np.asarray(k, float), np.asarray(m, float), np.asarray(h, float))
            for e, T, k, m, h in groups]
    params = [tuple(float(v) for v in ssvi_slice_to_raw(*essvi_surface.params_at(T)))
              for _, T, *_ in data]
    inv = [1.0 / np.maximum(h, h_floor) for *_, h in data]
    objs = [_objective(np.asarray(params[i]), data[i][2], data[i][1], data[i][3], inv[i])[0]
            for i in range(n)]
    start_total = float(sum(objs))
    infos = [{} for _ in range(n)]
    n_rejected = n_cut_rounds = 0
    sweeps = 0
    totals = [start_total]
    for sweep in range(max_sweeps):
        order = range(n) if sweep % 2 == 0 else range(n - 1, -1, -1)
        for i in order:
            e, T, k, mid, hs = data[i]
            lower = params[i - 1] if i > 0 else None
            upper = params[i + 1] if i < n - 1 else None
            cal = CALENDAR_ROWS_GRID
            accepted = False
            for rnd in range(_MAX_CUT_ROUNDS + 1):
                p_new, info = fit_svi_slice(k, T, mid, hs, params[i], h_floor=h_floor, grid=grid,
                                            lower=lower, upper=upper, cal_grid=cal)
                ok, pts = _ordered_against(p_new, lower, upper)
                if ok and info["max_constraint_violation"] <= _FEAS_TOL:
                    accepted = True
                    break
                if not len(pts) or rnd == _MAX_CUT_ROUNDS:
                    break
                n_cut_rounds += 1
                cal = np.unique(np.concatenate([cal, pts]))
            if accepted and info["objective"] <= objs[i]:
                params[i], objs[i] = p_new, info["objective"]
                infos[i] = {**info, "cut_rows": int(len(cal) - len(CALENDAR_ROWS_GRID))}
            elif not accepted:
                n_rejected += 1
        sweeps += 1
        totals.append(float(sum(objs)))
        if totals[-2] - totals[-1] <= rel_tol * max(totals[-2], 1e-12):
            break
    slices = []
    for i, (e, T, *_rest) in enumerate(data):
        inf = dict(infos[i]) if infos[i] else {"success": False, "returned_start": True,
                                               "objective": objs[i]}
        inf["g_min"] = float(svi_g(grid, params[i]).min())
        inf["grid_k"] = [float(grid[0]), float(grid[-1])]
        inf["objective"] = objs[i]
        slices.append(SVISlice(e, T, params[i], inf))
    surf = SVISurface(slices, name="SVI-cal")
    info = {"sweeps": sweeps, "summed_objective_by_sweep": totals,
            "objective_start_essvi": start_total, "objective": float(sum(objs)),
            "slices_kept_at_essvi_start": int(sum(1 for x in infos if not x)),
            "fits_rejected_unordered": n_rejected, "cutting_plane_rounds": n_cut_rounds,
            "ordering": surf.ordering(),
            "max_constraint_violation": float(max(
                max(0.0, -float(_constraints(grid, params[i - 1] if i else None,
                                             params[i + 1] if i < n - 1 else None)[0](
                    np.asarray(params[i])).min())) for i in range(n)))}
    return surf, info
