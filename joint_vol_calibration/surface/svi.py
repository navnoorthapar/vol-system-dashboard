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
No calendar constraint is imposed between SVI slices: per-expiry SVI is the
best-fitting butterfly-free smile per expiry, and its calendar violations are
measured and reported (metrics.dense_grid_checks); the calendar-consistent
surface is essvi.py. A sequential variant (each slice constrained above the
previous one, on the wide grid or on the two slices' quoted range) was tried
and rejected: on 2026-03-27 a few short slices ended infeasible and the
error propagated forward (inside-spread share 54% -> 14% wide grid, RMSE
0.69 -> 7.3 vp quoted range).

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

from joint_vol_calibration.surface.essvi import durrleman_g

SVI_GRID = np.linspace(-1.5, 1.0, 241)
_LEE_SLACK = 1e-6
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

    @property
    def T(self) -> np.ndarray:
        return np.array([s.T for s in self.slices])

    def w(self, k, T: float):
        Ts = self.T
        T = float(T)
        if T <= Ts[0]:
            return self.slices[0].w(k) * T / Ts[0]
        if T >= Ts[-1]:
            return self.slices[-1].w(k) * T / Ts[-1]
        i = int(np.searchsorted(Ts, T, side="right") - 1)
        lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        return (1 - lam) * self.slices[i].w(k) + lam * self.slices[i + 1].w(k)

    def iv(self, k, T: float):
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / float(T))

    def to_dict(self) -> dict:
        names = ("a", "b", "rho", "m", "sigma")
        return {"model": "SVI", "slices": [
            {"expiry": s.expiry, "T": float(s.T),
             **{n: float(v) for n, v in zip(names, s.params)},
             "success": bool(s.info.get("success", False)),
             "g_min_on_grid": s.info.get("g_min"),
             "butterfly_grid_k": s.info.get("grid_k")} for s in self.slices]}

    @classmethod
    def from_dict(cls, d: dict) -> "SVISurface":
        names = ("a", "b", "rho", "m", "sigma")
        return cls([SVISlice(s["expiry"], s["T"], tuple(s[n] for n in names))
                    for s in d["slices"]])


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


def _constraints(grid):
    """Inequality constraints c(x) >= 0 of one slice (see module docstring)."""
    lee = 2.0 * (1.0 - _LEE_SLACK)

    def fun(x):
        return np.concatenate([
            svi_g(grid, x),
            [lee - x[1] * (1 + x[2]), lee - x[1] * (1 - x[2]),
             x[0] + x[1] * x[4] * math.sqrt(max(1 - x[2] * x[2], 0.0))]])
    return fun


def fit_svi_slice(k, T, mid_iv, half_spread, x0, *, h_floor: float = 5e-4,
                  grid=SVI_GRID, maxiter: int = 500) -> tuple:
    """Fit one raw SVI slice from the start ``x0`` (feasible for the butterfly
    constraints, e.g. an eSSVI slice).

    SLSQP runs on scaled variables (a by the slice's ATM total variance, b and
    sigma by their start values, rho by 0.1, m by 0.05) with the objective
    divided by its start value; unscaled, SLSQP's line search failed on most
    short slices. SLSQP can also stop at a degenerate stationary point with
    b -> 0 (a flat smile: the gradient in rho, m and sigma vanishes with b;
    seen on a 2026-03-27 slice, 13.5 vp RMSE): when the result is not better
    than the start or b collapses, the fit is rerun with 10x smaller scales.
    The returned slice is the best feasible (max constraint violation <=
    1e-9) of the runs and the start (an eSSVI start is always feasible).
    Returns (params, info)."""
    k = np.asarray(k, dtype=float)
    mid = np.asarray(mid_iv, dtype=float)
    inv_h = 1.0 / np.maximum(np.asarray(half_spread, dtype=float), h_floor)
    lo = np.array([b_[0] for b_ in _BOUNDS])
    hi = np.array([b_[1] for b_ in _BOUNDS])
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    cfun = _constraints(grid)
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
                           constraints=[{"type": "ineq", "fun": lambda u: cfun(x0 + u * sc)}],
                           options={"maxiter": maxiter, "ftol": 1e-12})
        x = np.clip(x0 + res.x * sc, lo, hi)
        f, _ = _objective(x, k, T, mid, inv_h)
        return {"x": x, "f": f, "viol": float(max(0.0, -cfun(x).min())), "res": res}

    runs = [run(sc0)]
    if not (runs[0]["f"] < f0) or runs[0]["x"][1] < 1e-5:
        runs.append(run(0.1 * sc0))
    start = {"x": x0, "f": f0, "viol": float(max(0.0, -cfun(x0).min())), "res": None}
    feasible = [r for r in runs + [start] if np.isfinite(r["f"]) and r["viol"] <= 1e-9]
    best = min(feasible, key=lambda r: r["f"]) if feasible else min(runs + [start], key=lambda r: r["viol"])
    res = runs[0]["res"] if best["res"] is None else best["res"]
    x = best["x"]
    info = {"success": best["res"] is not None and bool(best["res"].success),
            "message": str(res.message), "nit": int(sum(r["res"].nit for r in runs)),
            "runs": len(runs), "objective_start": f0, "objective": best["f"],
            "max_constraint_violation": best["viol"], "returned_start": best["res"] is None,
            "g_min": float(svi_g(grid, x).min()),
            "grid_k": [float(grid[0]), float(grid[-1])]}
    return tuple(float(v) for v in x), info


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
