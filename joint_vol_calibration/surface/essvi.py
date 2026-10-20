"""
essvi.py — the extended SSVI (eSSVI) implied-variance surface, fitted as one
butterfly- and calendar-arbitrage-free surface.

Slice i (expiry T_i) has ATM total variance theta_i, "curvature" psi_i =
theta_i phi_i and skew rho_i; its total implied variance at log-moneyness
k = ln(K/F_T) is (Gatheral-Jacquier 2014, eq. 4.1, with psi = theta phi)

    w(k) = 1/2 (theta + rho psi k + sqrt(psi^2 k^2 + 2 rho psi theta k + theta^2)).

No-arbitrage conditions imposed
-------------------------------
Butterfly, per slice (Gatheral & Jacquier, arXiv:1204.0646, Theorem 4.2):
    psi (1 + |rho|) < 4      and      psi^2 (1 + |rho|) <= 4 theta.
With A = psi (1 + rho) and B = psi (1 - rho) (twice the right / left wing
slopes of w) they read max(A, B) < 4 and psi max(A, B) <= 4 theta.

Calendar, consecutive slices: theta, A and B non-decreasing (the condition
Corbetta, Cohort, Laachir & Martini, arXiv:1804.04924 sec. 2.2, quote from
Hendriks & Martini: theta, psi non-decreasing and |rho_{i+1} psi_{i+1} -
rho_i psi_i| <= psi_{i+1} - psi_i) PLUS the slice ordering itself,
w_{i+1}(k) >= w_i(k), imposed by cutting planes: rows on CALENDAR_GRID (25
points, dense near the money, out to |k| = 30), then, wherever a pair still
crosses on CHECK_GRID (4,001 points on [-3, 2] plus |k| = 10, 30), the ends
of each crossing run and the precise (bounded 1-D) minimiser of w_{i+1} - w_i
in it are added as rows and the fit is re-solved from where it stopped,
until no pair crosses by more than 1e-10 in total variance (at most 8 rounds;
rounds and rows are in the fit info). The extra rows are needed: the monotone (theta, A, B) condition alone is
NOT sufficient. Measured on the 2026-06-18 fit without them: the 2026-07-16
and 07-17 slices (theta 0.0013237 -> 0.0013323, A equal, B 0.0647 -> 0.0686)
cross for every k > 0.0046 (min w_2 - w_1 = -2.4e-5 at k = 0.033); there
d psi / d theta = 226 exceeds Gatheral-Jacquier's continuous-time bound
(Theorem 4.1, constant rho) (1 + sqrt(1 - rho^2)) / rho^2 x psi / theta = 210.
Far grid points (|k| = 3, 10, 30) catch the equal-wing-slope case, where the
limit of w_{i+1} - w_i is (theta (1 +- rho))_{i+1} - (theta (1 +- rho))_i.
The ordering is a grid certificate, not a closed-form one.

Fit: theta, A and B are cumulative sums of non-negative increments (the
monotone conditions hold by construction); butterfly and ordering rows are
SLSQP inequality constraints with analytic Jacobians; variables are scaled.
The objective is the mean squared implied-vol residual in half-spread units,
    (1/N) sum_j ((sigma_model_j - sigma_mid_j) / max(h_j, h_floor))^2,
h_j = (ask_iv_j - bid_iv_j) / 2, i.e. weights are inverse half-spreads.

Between and beyond the fitted expiries the surface uses the Corbetta et al.
(sec. 5) interpolation: theta linear in T, psi and rho psi linear in the same
weight; before T_1, theta and psi scale by T/T_1 at fixed rho; after T_N,
psi and rho are held and theta = theta_N T / T_N. Their proof that the
interpolation is calendar-free rests on the monotone condition above, which
is not sufficient, so interpolated maturities carry no certificate; use the
fitted (quoted) expiries where arbitrage-freeness matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

_BUTTERFLY_SLACK = 1e-6          # psi (1+|rho|) <= 4 (1 - slack): strict "< 4"
_MIN_FIRST = 1e-6
CALENDAR_GRID = np.unique(np.concatenate([
    np.linspace(-0.3, 0.3, 13), np.linspace(-1.5, 1.0, 6),
    [-30.0, -10.0, -3.0, 3.0, 10.0, 30.0]]))
CHECK_GRID = np.concatenate([np.linspace(-3.0, 2.0, 4001), [-30.0, -10.0, 10.0, 30.0]])
_ORDER_TOL = 1e-10            # w_{i+1} - w_i >= -tol everywhere checked ends the refinement
_MAX_ROUNDS = 8


# ── slice formulas ────────────────────────────────────────────────────────────

def essvi_w(k, theta, psi, rho):
    """Total implied variance of an eSSVI slice (broadcasts)."""
    k = np.asarray(k, dtype=float)
    q = rho * psi
    R = np.sqrt(psi * psi * k * k + 2.0 * q * theta * k + theta * theta)
    return 0.5 * (theta + q * k + R)


def essvi_derivatives(k, theta, psi, rho):
    """(w, dw/dk, d2w/dk2) of an eSSVI slice."""
    k = np.asarray(k, dtype=float)
    q = rho * psi
    R2 = psi * psi * k * k + 2.0 * q * theta * k + theta * theta
    R = np.sqrt(R2)
    u = psi * psi * k + q * theta
    w = 0.5 * (theta + q * k + R)
    w1 = 0.5 * (q + u / R)
    w2 = 0.5 * (psi * psi * R2 - u * u) / (R2 * R)
    return w, w1, w2


def durrleman_g(k, w, w1, w2):
    """Durrleman's g(k) (Gatheral-Jacquier eq. 2.1); the risk-neutral density
    is g(k) / sqrt(2 pi w) exp(-d_-^2 / 2), so g >= 0 <=> no butterfly
    arbitrage on the slice (with Lee's wing condition)."""
    k = np.asarray(k, dtype=float)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2


def butterfly_margins(theta, psi, rho) -> dict:
    """Slack of the two Gatheral-Jacquier butterfly conditions (>= 0 holds)."""
    theta, psi, rho = (np.asarray(x, dtype=float) for x in (theta, psi, rho))
    return {"lee_wing": 4.0 - psi * (1.0 + np.abs(rho)),
            "gj_curvature": 4.0 * theta - psi * psi * (1.0 + np.abs(rho))}


def calendar_margins(theta, psi, rho) -> dict:
    """Slack of the Hendriks-Martini calendar conditions between consecutive
    slices (each array has n-1 entries; >= 0 holds)."""
    theta, psi, rho = (np.asarray(x, dtype=float) for x in (theta, psi, rho))
    A, B = psi * (1.0 + rho), psi * (1.0 - rho)
    return {"theta_increment": np.diff(theta), "A_increment": np.diff(A),
            "B_increment": np.diff(B)}


# ── the surface ───────────────────────────────────────────────────────────────

@dataclass
class ESSVISurface:
    """Fitted eSSVI slices at expiries T (sorted) plus the interpolation rule."""
    T: np.ndarray
    theta: np.ndarray
    psi: np.ndarray
    rho: np.ndarray
    expiries: list = field(default_factory=list)

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)
        self.psi = np.asarray(self.psi, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        if not np.all(np.diff(self.T) > 0):
            raise ValueError("eSSVI slices need strictly increasing T")

    def params_at(self, T: float) -> tuple:
        """(theta, psi, rho) at maturity T (Corbetta et al. sec. 5)."""
        T = float(T)
        Ts = self.T
        if T <= Ts[0]:
            lam = T / Ts[0]
            return self.theta[0] * lam, self.psi[0] * lam, self.rho[0]
        if T >= Ts[-1]:
            return self.theta[-1] * T / Ts[-1], self.psi[-1], self.rho[-1]
        i = int(np.searchsorted(Ts, T, side="right") - 1)
        lam = (T - Ts[i]) / (Ts[i + 1] - Ts[i])
        th = (1 - lam) * self.theta[i] + lam * self.theta[i + 1]
        ps = (1 - lam) * self.psi[i] + lam * self.psi[i + 1]
        rp = (1 - lam) * self.rho[i] * self.psi[i] + lam * self.rho[i + 1] * self.psi[i + 1]
        return th, ps, rp / ps

    def w(self, k, T: float):
        th, ps, rh = self.params_at(T)
        return essvi_w(k, th, ps, rh)

    def iv(self, k, T: float):
        """Black-76 implied vol (decimal) at log-moneyness k, maturity T."""
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / float(T))

    def g(self, k, T: float):
        th, ps, rh = self.params_at(T)
        w, w1, w2 = essvi_derivatives(k, th, ps, rh)
        return durrleman_g(k, w, w1, w2)

    def atm_vol(self, T: float) -> float:
        th, _, _ = self.params_at(T)
        return math.sqrt(th / float(T))

    def atm_skew(self, T: float) -> float:
        """d sigma_BS / dk at k = 0: w'(0) / (2 sqrt(theta T)) = rho psi / (2 sqrt(theta T))."""
        th, ps, rh = self.params_at(T)
        return rh * ps / (2.0 * math.sqrt(th * float(T)))

    def conditions(self) -> dict:
        """No-arbitrage margins of the fitted slices: the analytic butterfly
        bounds, the monotone (theta, A, B) increments, and the minimum of
        w_{i+1} - w_i over CHECK_GRID (4,001 points on [-3, 2] plus |k| = 10,
        30) refined by a bounded 1-D search around each pair's worst point."""
        bm = butterfly_margins(self.theta, self.psi, self.rho)
        cm = calendar_margins(self.theta, self.psi, self.rho)
        out = {f"butterfly_{k}_min": float(v.min()) for k, v in bm.items()}
        out.update({f"monotone_{k}_min": (float(v.min()) if len(v) else None) for k, v in cm.items()})
        dmin = None
        A, B = self.psi * (1 + self.rho), self.psi * (1 - self.rho)
        for i in range(len(self.T) - 1):
            d = (essvi_w(CHECK_GRID, self.theta[i + 1], self.psi[i + 1], self.rho[i + 1])
                 - essvi_w(CHECK_GRID, self.theta[i], self.psi[i], self.rho[i]))
            j = int(np.argmin(d))
            lo = CHECK_GRID[max(j - 2, 0)]
            hi = CHECK_GRID[min(j + 2, len(CHECK_GRID) - 1)]
            fx = _pair_min(self.theta, A, B, i, min(lo, hi), max(lo, hi))[1] if hi != lo else d[j]
            m = min(float(d.min()), float(fx))
            dmin = m if dmin is None else min(dmin, m)
        out["slice_ordering_min_dw"] = dmin
        out["butterfly_ok"] = bool(all(v.min() >= -1e-12 for v in bm.values()))
        out["calendar_ok_on_check_grid"] = bool(
            all((len(v) == 0) or v.min() >= -1e-12 for v in cm.values())
            and (dmin is None or dmin >= -_ORDER_TOL))
        return out

    def to_dict(self) -> dict:
        return {"model": "eSSVI", "slices": [
            {"expiry": (self.expiries[i] if i < len(self.expiries) else None),
             "T": float(self.T[i]), "theta": float(self.theta[i]),
             "psi": float(self.psi[i]), "rho": float(self.rho[i]),
             "atm_vol": float(math.sqrt(self.theta[i] / self.T[i]))}
            for i in range(len(self.T))]}

    @classmethod
    def from_dict(cls, d: dict) -> "ESSVISurface":
        sl = d["slices"]
        return cls(T=[s["T"] for s in sl], theta=[s["theta"] for s in sl],
                   psi=[s["psi"] for s in sl], rho=[s["rho"] for s in sl],
                   expiries=[s.get("expiry") for s in sl])


# ── fitting ───────────────────────────────────────────────────────────────────

def _atm_total_variance(k: np.ndarray, iv: np.ndarray, T: float) -> float:
    o = np.argsort(k)
    return float(np.interp(0.0, k[o], iv[o]) ** 2 * T)


class _Objective:
    """Mean squared half-spread residual and its gradient in the increment
    parametrisation v = [d theta (n), dA (n), dB (n)]."""

    def __init__(self, k, T, mid_iv, inv_h, slice_idx, n):
        self.k, self.T, self.mid, self.inv_h = k, T, mid_iv, inv_h
        self.s, self.n = slice_idx, n
        self.N = len(k)

    @staticmethod
    def unpack(v, n):
        theta = np.cumsum(v[:n])
        A = np.cumsum(v[n:2 * n])
        B = np.cumsum(v[2 * n:])
        return theta, A, B

    def __call__(self, v):
        n, s, k = self.n, self.s, self.k
        theta, A, B = self.unpack(v, n)
        P = 0.5 * (A + B)
        Q = 0.5 * (A - B)
        th, p, q = theta[s], P[s], Q[s]
        R = np.sqrt(np.maximum(p * p * k * k + 2.0 * q * th * k + th * th, 1e-300))
        w = np.maximum(0.5 * (th + q * k + R), 1e-14)
        iv = np.sqrt(w / self.T)
        r = (iv - self.mid) * self.inv_h
        f = float(np.dot(r, r)) / self.N
        # gradient
        c = 2.0 * r * self.inv_h / self.N / (2.0 * iv * self.T)     # df/dw per quote
        dth = 0.5 * (1.0 + (q * k + th) / R)
        dP = 0.5 * p * k * k / R
        dQ = 0.5 * (k + th * k / R)
        g_th = np.bincount(s, weights=c * dth, minlength=n)
        g_A = np.bincount(s, weights=c * 0.5 * (dP + dQ), minlength=n)
        g_B = np.bincount(s, weights=c * 0.5 * (dP - dQ), minlength=n)
        rc = lambda a: np.cumsum(a[::-1])[::-1]                      # noqa: E731
        return f, np.concatenate([rc(g_th), rc(g_A), rc(g_B)])


def _butterfly_constraints(n):
    L = np.tril(np.ones((n, n)))
    lim = 4.0 * (1.0 - _BUTTERFLY_SLACK)

    def fun(v):
        theta, A, B = _Objective.unpack(v, n)
        psi = 0.5 * (A + B)
        return np.concatenate([lim - A, lim - B, 4.0 * theta - psi * A, 4.0 * theta - psi * B])

    def jac(v):
        theta, A, B = _Objective.unpack(v, n)
        Z = np.zeros((n, n))
        rows = []
        rows.append(np.hstack([Z, -L, Z]))
        rows.append(np.hstack([Z, Z, -L]))
        dA3 = -(A + 0.5 * B)                    # d(psi A)/dA = A + B/2
        dB3 = -0.5 * A
        rows.append(np.hstack([4.0 * L, dA3[:, None] * L, dB3[:, None] * L]))
        dA4 = -0.5 * B
        dB4 = -(0.5 * A + B)                    # d(psi B)/dB = A/2 + B
        rows.append(np.hstack([4.0 * L, dA4[:, None] * L, dB4[:, None] * L]))
        return np.vstack(rows)

    return {"type": "ineq", "fun": fun, "jac": jac}


def _w_and_grads(k, theta, A, B):
    """w(k) per slice (rows) on grid k (cols) and dw/d(theta, A, B)."""
    P = 0.5 * (A + B)[:, None]
    Q = 0.5 * (A - B)[:, None]
    th = theta[:, None]
    kk = k[None, :]
    R = np.sqrt(np.maximum(P * P * kk * kk + 2.0 * Q * th * kk + th * th, 1e-300))
    w = 0.5 * (th + Q * kk + R)
    dth = 0.5 * (1.0 + (Q * kk + th) / R)
    dP = 0.5 * P * kk * kk / R
    dQ = 0.5 * (kk + th * kk / R)
    return w, dth, 0.5 * (dP + dQ), 0.5 * (dP - dQ)


def _ordering_constraints(n, pair, kk):
    """Rows w_{pair+1}(kk) - w_pair(kk) >= 0 (one row per (pair, k) entry)."""
    pair = np.asarray(pair, dtype=int)
    kk = np.asarray(kk, dtype=float)
    lidx = np.arange(n)
    up_hi = (lidx[None, :] <= (pair + 1)[:, None]).astype(float)
    up_lo = (lidx[None, :] <= pair[:, None]).astype(float)

    def parts(v):
        theta, A, B = _Objective.unpack(v, n)
        lo = _w_and_grads_rows(kk, theta[pair], A[pair], B[pair])
        hi = _w_and_grads_rows(kk, theta[pair + 1], A[pair + 1], B[pair + 1])
        return lo, hi

    def fun(v):
        lo, hi = parts(v)
        return hi[0] - lo[0]

    def jac(v):
        lo, hi = parts(v)
        return np.hstack([hi[j][:, None] * up_hi - lo[j][:, None] * up_lo for j in (1, 2, 3)])

    return {"type": "ineq", "fun": fun, "jac": jac}


def _w_and_grads_rows(k, theta, A, B):
    """Row-wise w(k_r; slice_r) and dw/d(theta, A, B)."""
    P, Q = 0.5 * (A + B), 0.5 * (A - B)
    R = np.sqrt(np.maximum(P * P * k * k + 2.0 * Q * theta * k + theta * theta, 1e-300))
    w = 0.5 * (theta + Q * k + R)
    dth = 0.5 * (1.0 + (Q * k + theta) / R)
    dP = 0.5 * P * k * k / R
    dQ = 0.5 * (k + theta * k / R)
    return w, dth, 0.5 * (dP + dQ), 0.5 * (dP - dQ)


def _pair_min(theta, A, B, i, k_lo, k_hi):
    """Bounded 1-D minimiser of w_{i+1}(k) - w_i(k) on [k_lo, k_hi]."""
    P, Q = 0.5 * (A + B), 0.5 * (A - B)
    rho = Q / P

    def d(x):
        return float(essvi_w(x, theta[i + 1], P[i + 1], rho[i + 1]) - essvi_w(x, theta[i], P[i], rho[i]))
    r = minimize_scalar(d, bounds=(k_lo, k_hi), method="bounded", options={"xatol": 1e-10})
    return float(r.x), float(r.fun)


def _ordering_shortfalls(v, n):
    """Cutting-plane candidates per consecutive pair: for every run of
    CHECK_GRID points where w_{i+1} - w_i < -_ORDER_TOL, the run's end points
    and the precise (bounded 1-D) minimiser of w_{i+1} - w_i inside it; plus
    the minimiser near the pair's worst grid point when only it crosses."""
    theta, A, B = _Objective.unpack(v, n)
    w, *_ = _w_and_grads(CHECK_GRID, theta, A, B)
    d = w[1:] - w[:-1]
    G = len(CHECK_GRID)
    out = []
    for i in range(n - 1):
        neg = d[i] < -_ORDER_TOL
        ks = []
        j = int(np.argmin(d[i]))
        runs = []
        if neg.any():
            idx = np.where(neg)[0]
            cuts = np.where(np.diff(idx) > 1)[0]
            starts = np.concatenate([[idx[0]], idx[cuts + 1]])
            ends = np.concatenate([idx[cuts], [idx[-1]]])
            runs = list(zip(starts, ends))
        else:
            runs = [(j, j)]
        for a_, b_ in runs:
            lo, hi = CHECK_GRID[max(a_ - 1, 0)], CHECK_GRID[min(b_ + 1, G - 1)]
            lo, hi = min(lo, hi), max(lo, hi)
            kx, fx = _pair_min(theta, A, B, i, lo, hi) if hi > lo else (float(CHECK_GRID[a_]), float(d[i][a_]))
            if fx < -_ORDER_TOL:
                ks.append(kx)
            if neg.any():
                ks.extend([float(CHECK_GRID[a_]), float(CHECK_GRID[b_])])
        out.append(np.unique(np.asarray(ks, dtype=float)))
    return out


def _initial_point(k, mid_iv, T_slice, s, n, rho0=-0.7, eta0=1.0):
    """A feasible start: theta from the ATM quotes (made increasing), power-law
    psi = eta0 sqrt(theta) (Gatheral-Jacquier's phi = eta theta^-1/2, inside both
    butterfly bounds for eta0 <= 2/sqrt(1+|rho0|)), constant rho0. With
    constant rho, d psi / d theta = psi / (2 theta) is inside GJ Theorem 4.1's
    calendar bound, so the slices are ordered."""
    theta = np.array([_atm_total_variance(k[s == i], mid_iv[s == i], T_slice[i]) for i in range(n)])
    theta = np.maximum.accumulate(np.maximum(theta, 1e-6))
    theta = theta + 1e-9 * np.arange(n)
    psi = eta0 * np.sqrt(theta)
    psi = np.minimum(psi, 0.9 * 4.0 / (1.0 + abs(rho0)))
    psi = np.maximum.accumulate(psi)
    A, B = psi * (1 + rho0), psi * (1 - rho0)
    inc = lambda a: np.concatenate([[a[0]], np.diff(a)])               # noqa: E731
    return np.concatenate([inc(theta), inc(A), inc(B)])


def fit_essvi(k, T, mid_iv, half_spread, expiries, *, h_floor: float = 5e-4,
              weights=None, maxiter: int = 1000, x0=None) -> tuple:
    """Fit one eSSVI surface to quotes.

    k, T, mid_iv, half_spread : per-quote arrays (decimal vols; half_spread
        = (ask_iv - bid_iv)/2, floored at h_floor = 0.05 vol points)
    expiries : per-quote expiry labels (one T per label)
    weights  : optional extra per-quote multipliers of the residual (e.g.
               robust weights); default 1
    Returns (ESSVISurface, info dict).
    """
    k = np.asarray(k, dtype=float)
    T = np.asarray(T, dtype=float)
    mid = np.asarray(mid_iv, dtype=float)
    h = np.maximum(np.asarray(half_spread, dtype=float), h_floor)
    exp_arr = np.asarray(expiries)
    labels, first = np.unique(exp_arr, return_index=True)
    order = np.argsort(T[first], kind="mergesort")
    labels = labels[order]
    T_slice = T[first][order]
    lab_idx = {lab: i for i, lab in enumerate(labels)}
    s = np.array([lab_idx[e] for e in exp_arr])
    n = len(labels)
    inv_h = 1.0 / h
    if weights is not None:
        inv_h = inv_h * np.asarray(weights, dtype=float)
    obj = _Objective(k, T, mid, inv_h, s, n)
    v0 = _initial_point(k, mid, T_slice, s, n) if x0 is None else np.asarray(x0, dtype=float)
    lb = np.concatenate([[_MIN_FIRST], np.zeros(n - 1)] * 3)
    # SLSQP on scaled variables u = v / sc: the increments differ by orders of
    # magnitude (theta ~1e-4..1e-1, A, B ~1e-1..1); unscaled, SLSQP stops early
    # (measured on 2026-03-24: objective 45.6 unscaled vs 10.4 scaled).
    sc = np.abs(v0) + 1e-3 * np.abs(v0).max()

    def f_scaled(u):
        f, g = obj(u * sc)
        return f, g * sc

    def scaled(con):
        return {"type": "ineq", "fun": lambda u: con["fun"](u * sc),
                "jac": lambda u: con["jac"](u * sc) * sc[None, :]}

    pair = np.repeat(np.arange(max(n - 1, 0)), len(CALENDAR_GRID))
    kk = np.tile(CALENDAR_GRID, max(n - 1, 0))
    u = v0 / sc
    rounds, nit, nfev, res = 0, 0, 0, None
    while True:
        cons = [scaled(_butterfly_constraints(n))]
        if n > 1:
            cons.append(scaled(_ordering_constraints(n, pair, kk)))
        res = minimize(f_scaled, u, jac=True, method="SLSQP",
                       bounds=list(zip(lb / sc, [None] * (3 * n))), constraints=cons,
                       options={"maxiter": maxiter, "ftol": 1e-14})
        u = res.x
        rounds += 1
        nit += int(res.nit)
        nfev += int(res.nfev)
        bad = _ordering_shortfalls(np.maximum(u * sc, lb), n) if n > 1 else []
        if not any(len(b_) for b_ in bad) or rounds >= _MAX_ROUNDS:
            break
        # cutting planes at the crossing points found
        add_p, add_k = [], []
        for i, ks in enumerate(bad):
            if len(ks):
                add_p.append(np.full(len(ks), i))
                add_k.append(ks)
        pair = np.concatenate([pair] + add_p)
        kk = np.concatenate([kk] + add_k)
    v = np.maximum(res.x * sc, lb)
    theta, A, B = _Objective.unpack(v, n)
    psi = 0.5 * (A + B)
    rho = (A - B) / (A + B)
    surf = ESSVISurface(T=T_slice, theta=theta, psi=psi, rho=rho, expiries=[str(x) for x in labels])
    f0, _ = obj(v0)
    f1, _ = obj(v)
    viol = 0.0
    for c in cons:
        viol = max(viol, float(max(0.0, -np.min(c["fun"](v / sc)))))

    bad = _ordering_shortfalls(v, n) if n > 1 else []
    info = {"success": bool(res.success), "message": str(res.message), "nit": nit,
            "nfev": nfev, "objective_start": f0, "objective": f1,
            "max_constraint_violation": viol, "ordering_rounds": rounds,
            "ordering_rows": int(len(kk)),
            "ordering_check_violations": int(sum(len(b_) for b_ in bad)),
            "h_floor_vp": h_floor * 100.0, "n_quotes": int(len(k)), "n_slices": n,
            "x": v}
    return surf, info
