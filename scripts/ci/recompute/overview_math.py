"""
overview_math.py: the Overview's definitions, written again from data_store.

Each function restates one definition the Overview page prints (the docstring
names it) and computes it from the raw store by a code path of its own: the PDV
forecast as an expanding least-squares fit carried in cumulative normal
equations, the rule label from its rolling windows, Heston prices by the Lewis
integral, E[VIX_T] by integrating the CIR survival function, the SVI desk grid
from the slices. Nothing here imports dashboard/ or joint_vol_calibration/;
pickles written by those packages are read with their classes replaced by inert
records (``load_inert``), so no audited code runs while they load.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

from .data import Store, is_session, strike_at_delta, svi_total_variance

ANNUALISE = 252
DAYS_PER_YEAR = 365.0
VIX_TENOR_DAYS = {"^VIX9D": 9, "^VIX": 30, "^VIX3M": 93, "^VIX6M": 184}   # Cboe's index tenors
# the calibration gate's thresholds (joint_calibrator.is_acceptable_calibration), restated
GATE: dict[str, Any] = {
    "model_version": 2, "sigma_min": 0.05, "rho_abs_min": 0.10, "spx_rmse_max": 3.0, "vix_tenors_min": 3,
    "spx_expiries_min": 4, "spx_options_min": 30, "put_wing_expiries_min": 3,
    "box": {"kappa": (0.1, 20.0), "theta": (1e-4, 1.0), "sigma": (1e-3, 2.0), "rho": (-0.95, 0.0),
            "v0": (1e-4, 1.0)}}
# the rule label's windows: rv over 20 sessions (15 present), the R2 gate the 0.80
# quantile of 252 VVIX prints (63 present), VVIX carried one session at most
RULE: dict[str, Any] = {"rv_window": 20, "rv_min": 15, "gate_window": 252, "gate_q": 0.80, "gate_min": 63,
                        "vvix_ffill": 1}
VIX_WINDOW_YEARS = 30.0 / 365.0
FELLER_TOL = 1e-4


# ── inert pickles ─────────────────────────────────────────────────────────────
_SAFE_ROOTS = ("builtins", "copyreg", "collections", "datetime", "numpy", "pandas", "pyarrow", "_codecs")


class Inert:
    """Stands in for a class of the audited packages: keeps the pickled state, runs nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            state = {**(state[0] or {}), **state[1]}
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state


class _InertUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.split(".")[0] in _SAFE_ROOTS:
            return super().find_class(module, name)
        return type(name, (Inert,), {"__module__": "inert." + module})


def load_inert(path: Path) -> Any:
    with open(path, "rb") as f:
        return _InertUnpickler(f).load()


def attr(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def finite(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── calendar and wording ──────────────────────────────────────────────────────
def session_on_or_before(day: str) -> str:
    d = dt.date.fromisoformat(str(day)[:10])
    while not is_session(d):
        d -= dt.timedelta(days=1)
    return d.isoformat()


def stale_level(age: int | None) -> str:
    """The pages' staleness level: warn from 2 NYSE sessions, error from 5."""
    if age is None:
        return "unknown"
    return "error" if age >= 5 else ("warn" if age >= 2 else "ok")


def ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def rank_words(pct: float) -> dict:
    """A percentile (percent of prior sessions strictly below) as the pages word it:
    'Record high' / 'Record low' at 100 / 0, else an integer 1st..99th; the tail
    'top N%' from 80, 'bottom N%' to 20."""
    if pct >= 100.0:
        return {"record": "high", "big": "Record high", "k": None, "tail": "highest on record"}
    if pct <= 0.0:
        return {"record": "low", "big": "Record low", "k": None, "tail": "lowest on record"}
    k = min(99, max(1, math.floor(pct + 0.5)))
    tail = f"top {100 - k}%" if pct >= 80 else (f"bottom {k}%" if pct <= 20 else None)
    return {"record": None, "big": ordinal(k), "k": k, "tail": tail}


def plural(n: int, word: str) -> str:
    return f"{n:,} {word if n == 1 else word + 's'}"


# ── market series ─────────────────────────────────────────────────────────────
def spx_closes(store: Store) -> pd.DataFrame:
    """spx_ohlcv by session (the last row of a duplicated date kept), with the
    close-to-close log return beside the stored one."""
    df = store.spx()
    df = df[~df.index.duplicated(keep="last")].copy()
    df["close_return"] = pd.Series(np.log(df["close"].to_numpy(float)), index=df.index).diff()
    return df


def vix_daily(store: Store) -> pd.Series:
    s = store.vix_close().astype(float)
    return s[~s.index.duplicated(keep="last")]


def stored_labels(store: Store) -> pd.Series:
    rl = store.parquet("signals/regime_labels.parquet")["regime"].dropna().astype(int)
    rl.index = pd.to_datetime(rl.index).strftime("%Y-%m-%d")
    return rl


# ── the rule label (R0 / R1 / R2) ─────────────────────────────────────────────
def rule_inputs(store: Store) -> dict | None:
    """The newest labelled session of the rule, from its definition: SPX sessions
    joined with the VIX-family rows; rv20 = √(mean r²·252) over the last 20 rows
    (stored log_return, at least 15 present); the R2 gate = the 0.80 quantile of
    the last 252 VVIX prints (at least 63; gaps skipped); VVIX carried one session
    at most; R2 when VVIX > gate, else R0 when rv20 > VIX/100, else R1."""
    spx = spx_closes(store)
    fam = store.vix_family()
    j = spx[["log_return"]].join(fam, how="inner")
    r2 = j["log_return"].astype(float).to_numpy() ** 2
    vvix = j["^VVIX"].astype(float).to_numpy() if "^VVIX" in j else np.full(len(j), np.nan)
    vix = j["^VIX"].astype(float).to_numpy() / 100.0
    n = len(j)
    for t in range(n - 1, max(n - 30, 0) - 1, -1):
        win = r2[max(0, t - RULE["rv_window"] + 1):t + 1]
        win = win[np.isfinite(win)]
        rv = math.sqrt(win.mean() * ANNUALISE) if len(win) >= RULE["rv_min"] else None
        g = vvix[max(0, t - RULE["gate_window"] + 1):t + 1]
        g = g[np.isfinite(g)]
        gate = float(np.quantile(g, RULE["gate_q"])) if len(g) >= RULE["gate_min"] else None
        level = vvix[t]
        if not np.isfinite(level) and t >= 1 and RULE["vvix_ffill"] >= 1:
            level = vvix[t - 1]
        if rv is None or gate is None or not np.isfinite(level):
            continue
        r2_on = level > gate
        if not r2_on and not np.isfinite(vix[t]):
            continue
        label = 2 if r2_on else (0 if rv > vix[t] else 1)
        raw = vvix[t]
        return {"date": j.index[t], "rv20": rv, "vix": float(vix[t]) if np.isfinite(vix[t]) else None,
                "vvix": float(raw) if np.isfinite(raw) else None, "gate": gate, "label": label}
    return None


# ── PDV: features, target, purged walk-forward forecast ───────────────────────
def ewm_adjusted(x: np.ndarray, halflife: float) -> np.ndarray:
    """The bias-free EWMA Σ wⁱ x_{t−i} / Σ wⁱ, w = 2^(−1/halflife), by its two recursions."""
    w = math.exp(-math.log(2.0) / halflife)
    num = np.empty(len(x))
    den = np.empty(len(x))
    a = b = 0.0
    for i, v in enumerate(x):
        a = v + w * a
        b = 1.0 + w * b
        num[i], den[i] = a, b
    return num / den


def pdv_state(r: pd.Series, hl_fast: float, hl_slow: float, hl_lev: float) -> pd.DataFrame:
    """σ₁ = √(252·EWMA_fast(r²)), σ₂ = √(252·EWMA_slow(r²)), lev = √252·EWMA_lev(r)."""
    x = r.to_numpy(float)
    return pd.DataFrame({"sigma1": np.sqrt(ewm_adjusted(x * x, hl_fast) * ANNUALISE),
                         "sigma2": np.sqrt(ewm_adjusted(x * x, hl_slow) * ANNUALISE),
                         "lev": ewm_adjusted(x, hl_lev) * math.sqrt(ANNUALISE)}, index=r.index)


def forward_rv(r: pd.Series, h: int) -> pd.Series:
    """y(t) = √(252/h · Σ_{i=1..h} r²_{t+i}); NaN where the window is incomplete."""
    c = np.concatenate([[0.0], np.cumsum(r.to_numpy(float) ** 2)])
    n = len(r)
    y = np.full(n, np.nan)
    t = np.arange(n - h)
    y[t] = np.sqrt((c[t + h + 1] - c[t + 1]) / h * ANNUALISE)
    return pd.Series(y, index=r.index)


def pdv_walk_forward(r: pd.Series, feats: pd.DataFrame, h: int, train_min: int, burn_in: int) -> pd.Series:
    """The forecast at t: OLS of y on [σ₁, σ₂, lev, 1] over the label rows i ≤ t − h
    (i ≥ burn_in), at least ``train_min`` of them, evaluated at row t, floored at 1e-4.
    Carried as cumulative normal equations (XᵀX, Xᵀy) instead of a refit per row."""
    X = np.column_stack([feats[["sigma1", "sigma2", "lev"]].to_numpy(float), np.ones(len(feats))])
    y = forward_rv(r, h).to_numpy(float)
    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    ok[:burn_in] = False
    xtx = np.zeros((4, 4))
    xty = np.zeros(4)
    n_used = 0
    out = np.full(len(r), np.nan)
    for t in range(len(r)):
        i = t - h
        if i >= 0 and ok[i]:
            xtx += np.outer(X[i], X[i])
            xty += X[i] * y[i]
            n_used += 1
        if i < 0 or n_used < train_min or not np.isfinite(X[t]).all():
            continue
        beta = np.linalg.solve(xtx, xty)
        out[t] = max(float(X[t] @ beta), 1e-4)
    return pd.Series(out, index=r.index).dropna()


def pdv_stress(r: pd.Series, feats: pd.DataFrame, h: int, burn_in: int, day: str) -> dict | None:
    """The ex-ante forecast for ``day`` made at the close before it: OLS on the label
    rows whose window ended by that close, against the RMS vol realised over the h
    sessions from ``day``."""
    days = [str(d) for d in r.index]
    pos = days.index(day) if day in days else 0
    if pos < 1:
        return None
    origin = days[pos - 1]
    X = np.column_stack([feats[["sigma1", "sigma2", "lev"]].to_numpy(float), np.ones(len(feats))])
    y = forward_rv(r, h).to_numpy(float)
    rows = np.arange(len(r))
    first = max(burn_in, 19)       # the lagged and 20-day columns of the training frame exist from row 19
    train = (rows >= first) & np.isfinite(y) & (rows + h <= pos - 1)
    beta, *_ = np.linalg.lstsq(X[train], y[train], rcond=None)
    fc = max(float(X[pos - 1] @ beta), 1e-4)
    win = r.iloc[pos:pos + h].to_numpy(float)
    act = math.sqrt(float(np.mean(win ** 2)) * ANNUALISE)
    return {"origin": origin, "n_train": int(train.sum()), "forecast": fc * 100, "actual": act * 100,
            "error": (fc - act) * 100, "n_window": len(win), "window_end": days[min(pos + h, len(r)) - 1]}


def r_squared(y: np.ndarray, p: np.ndarray) -> float:
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# ── Heston: Lewis prices, Black inversion, the ATM skew ───────────────────────
def _heston_cf(u: complex, T: float, kappa: float, theta: float, sigma: float, rho: float,
               v0: float) -> complex:
    """E[exp(iu ln(F_T/F_0))] (the 'little trap' form, no branch cut)."""
    b = kappa - rho * sigma * 1j * u
    d = np.sqrt(b * b + sigma * sigma * (1j * u + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = kappa * theta / sigma ** 2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / sigma ** 2 * (1.0 - e) / (1.0 - g * e)
    return complex(np.exp(C + D * v0))


def heston_call(k: float, T: float, p: dict) -> float:
    """Undiscounted call on F = 1 at strike e^k: 1 − e^{k/2}/π ∫₀^∞ Re[e^{−iuk} φ(u − i/2)] / (u² + ¼) du."""
    args = (T, p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])

    def f(u: float) -> float:
        return (np.exp(-1j * u * k) * _heston_cf(u - 0.5j, *args)).real / (u * u + 0.25)
    val, _ = integrate.quad(f, 0.0, np.inf, limit=800, epsabs=1e-13, epsrel=1e-11)
    return 1.0 - math.exp(0.5 * k) / math.pi * val


def black_call(k: float, vol: float, T: float) -> float:
    s = vol * math.sqrt(T)
    d1 = -k / s + 0.5 * s
    return float(stats.norm.cdf(d1) - math.exp(k) * stats.norm.cdf(d1 - s))


def implied_vol(k: float, T: float, call: float) -> float | None:
    lo = max(1.0 - math.exp(k), 0.0)
    if not lo < call < 1.0:
        return None
    return float(optimize.brentq(lambda v: black_call(k, v, T) - call, 1e-6, 5.0, xtol=1e-14, rtol=1e-14))


def heston_atm_skew(p: dict, T: float, h: float = 0.01) -> float | None:
    """∂σ/∂k at k = 0 of the Heston smile: central differences at h and h/2,
    Richardson-combined, (4·D(h/2) − D(h))/3 (the page's FD scheme)."""
    vols = []
    for k in (-h, -h / 2, h / 2, h):
        v = implied_vol(k, T, heston_call(k, T, p))
        if v is None:
            return None
        vols.append(v)
    d_h = (vols[3] - vols[0]) / (2 * h)
    d_hh = (vols[2] - vols[1]) / h
    return (4.0 * d_hh - d_h) / 3.0


# ── Heston: E^Q[VIX_T] for a VX future ────────────────────────────────────────
def heston_vix_future(p: dict, T: float) -> float:
    """E[√(a + b v_T)]·100 with VIX² = a + b·v (b = (1 − e^{−κτ})/(κτ), a = θ(1 − b),
    τ = 30/365) and v_T the CIR law: 2c·v_T ~ χ'²(4κθ/σ², 2c·v₀e^{−κT}),
    c = 2κ/(σ²(1 − e^{−κT})). By parts: E[g(X)] = g(0) + ∫ g′(x)·P(X > x) dx."""
    kappa, theta, sigma, v0 = (float(p[x]) for x in ("kappa", "theta", "sigma", "v0"))
    kt = kappa * VIX_WINDOW_YEARS
    b = -math.expm1(-kt) / kt
    a = theta * (1.0 - b)
    if T <= 0:
        return 100.0 * math.sqrt(a + b * v0)
    c = 2.0 * kappa / (sigma ** 2 * -math.expm1(-kappa * T))
    df, nc = 4.0 * kappa * theta / sigma ** 2, 2.0 * c * v0 * math.exp(-kappa * T)
    bb = b / (2.0 * c)
    mean = df + nc
    sd = math.sqrt(2.0 * (df + 2.0 * nc))
    # Gauss–Legendre on cells graded geometrically towards 0 (where the density is
    # singular when 4κθ/σ² < 2), then uniform out to 60 sd past the mean
    edges = np.concatenate([[0.0], mean * np.geomspace(1e-14, 1.0, 29),
                            mean + sd * np.linspace(0.0, 60.0, 61)[1:]])
    lo, hi = edges[:-1, None], edges[1:, None]
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _GL_NODES[None, :]
    w = 0.5 * (hi - lo) * _GL_WEIGHTS[None, :]
    total = float(np.sum(w * bb / (2.0 * np.sqrt(a + bb * x)) * stats.ncx2.sf(x, df, nc)))
    return 100.0 * (math.sqrt(a) + total)


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(12)


# ── SVI desk coordinates ──────────────────────────────────────────────────────
def svi_atm_skew(sl: dict) -> float:
    """∂σ/∂k at k = 0 of a raw-SVI slice: w′(0) / (2√(w(0)·T))."""
    x = -sl["m"]
    w0 = float(svi_total_variance(0.0, sl))
    w0p = sl["b"] * (sl["rho"] + x / math.sqrt(x * x + sl["sigma"] ** 2))
    return w0p / (2.0 * math.sqrt(w0 * sl["T"]))


def power_slope(T: list[float], skew: list[float]) -> float | None:
    """OLS slope of ln|skew| on ln T (at least three expiries)."""
    pts = [(t, abs(s)) for t, s in zip(T, skew) if t > 0 and s != 0 and math.isfinite(s)]
    if len(pts) < 3:
        return None
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    xm, ym = x.mean(), y.mean()
    return float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))


def svi_slices(sj: dict) -> list[dict]:
    s = (((sj.get("surfaces") or {}).get("svi_cal") or {}).get("params") or {}).get("slices") or []
    return sorted(s, key=lambda x: float(x["T"]))


PILLARS = {"10P": -0.10, "25P": -0.25, "ATM": None, "25C": 0.25, "10C": 0.10}


def desk_row(sl: dict, k_range: tuple[float, float] | None) -> dict[str, Any]:
    """One listed slice in desk coordinates: each pillar's total variance and whether
    its strike lies outside the expiry's quoted ln(K/F) range."""
    w: dict[str, float] = {}
    extrap: dict[str, bool] = {}
    for key, delta in PILLARS.items():
        k = 0.0 if delta is None else strike_at_delta(sl, delta)
        w[key] = float(svi_total_variance(k, sl))
        extrap[key] = k_range is None or not (k_range[0] <= k <= k_range[1])
    return {"T": float(sl["T"]), "expiry": str(sl["expiry"])[:10], "w": w, "extrap": extrap}


def constant_maturity(rows: list[dict], days: float) -> dict | None:
    """Pillar vols at ``days`` (T = days/365), linear in total variance at fixed delta
    between the bracketing listed expiries, never extrapolated; RR25 = 25ΔC − 25ΔP and
    BF25 = (25ΔC + 25ΔP)/2 − ATMF, None when a 25Δ strike is beyond the quotes."""
    T = days / DAYS_PER_YEAR
    lo = max((r for r in rows if r["T"] <= T + 1e-12), key=lambda r: r["T"], default=None)
    hi = min((r for r in rows if r["T"] >= T - 1e-12), key=lambda r: r["T"], default=None)
    if lo is None or hi is None:
        return None
    x = 0.0 if hi["T"] == lo["T"] else (T - lo["T"]) / (hi["T"] - lo["T"])
    vol = {key: 100.0 * math.sqrt(((1 - x) * lo["w"][key] + x * hi["w"][key]) / T) for key in PILLARS}
    ext = {key: lo["extrap"][key] or hi["extrap"][key] for key in PILLARS}
    wing = not (ext["25C"] or ext["25P"])
    return {"lower": lo["expiry"], "upper": hi["expiry"], "atmf": vol["ATM"],
            "rr25": vol["25C"] - vol["25P"] if wing else None,
            "bf25": 0.5 * (vol["25C"] + vol["25P"]) - vol["ATM"] if wing else None}


def quoted_k_ranges(res: pd.DataFrame, model: str = "svi_cal") -> dict[str, tuple[float, float]]:
    q = res[res["model"].astype(str) == model] if (res["model"].astype(str) == model).any() else res
    out: dict[str, tuple[float, float]] = {}
    for e, k in zip(q["expiry"].astype(str).str[:10], q["k"].astype(float)):
        lo, hi = out.get(e, (k, k))
        out[e] = (min(lo, k), max(hi, k))
    return out


# ── calibration archive ───────────────────────────────────────────────────────
def calibration_session(cal: dict, name: str) -> str | None:
    for key in ("spot_date", "as_of_date"):
        v = cal.get(key)
        if v is not None and len(str(v)) >= 10 and str(v)[4] == "-":
            return str(v)[:10]
    m = name.removeprefix("joint_cal_").removesuffix(".pkl")
    return m if len(m) == 10 else None


def is_legacy(cal: dict) -> bool:
    try:
        return int(cal.get("model_version", 1)) < GATE["model_version"]
    except (TypeError, ValueError):
        return True


def passes_gate(cal: dict) -> bool:
    """The acceptance gate of a current-schema fit, from the result's own fields."""
    if is_legacy(cal):
        return False
    p = cal["params"]
    ll = cal.get("leg_losses") or cal.get("losses") or {}
    snap = cal.get("spx_snapshot_session") or (session_on_or_before(cal["spx_snapshot_date"])
                                               if cal.get("spx_snapshot_date") else None)
    spot = cal.get("spot_date")
    if spot is not None and snap is not None and str(snap)[:10] != str(spot)[:10]:
        return False
    box = GATE["box"]
    k = float(p.get("kappa", 0.0))
    if float(p.get("sigma", 0.0)) < GATE["sigma_min"] or abs(float(p.get("rho", 0.0))) < GATE["rho_abs_min"]:
        return False
    if k <= 1.1 * box["kappa"][0] or k >= 0.995 * box["kappa"][1]:
        return False
    if any(name in p and float(p[name]) <= 2.0 * box[name][0] for name in ("v0", "theta")):
        return False
    checks = ((ll.get("spx_iv_rmse"), lambda v: v > GATE["spx_rmse_max"]),
              (cal.get("n_vix_tenors"), lambda v: v < GATE["vix_tenors_min"]),
              (cal.get("n_spx_expiries"), lambda v: v < GATE["spx_expiries_min"]),
              (cal.get("n_spx_options"), lambda v: v < GATE["spx_options_min"]),
              (cal.get("n_put_wing_expiries"), lambda v: v < GATE["put_wing_expiries_min"]))
    return not any(v is not None and bad(v) for v, bad in checks)


UNREADABLE = (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError, KeyError,
              TypeError, ValueError)


def calibration_archive(store: Store) -> list[dict]:
    """Every readable joint_cal_*.pkl (an unreadable one is left out of every count,
    as the page leaves it out): session, legacy flag, gate verdict, params and losses."""
    out = []
    for p in sorted((store.root / "calibrations").glob("joint_cal_*.pkl")):
        try:
            cal = load_inert(p)
        except UNREADABLE:
            cal = None
        if not isinstance(cal, dict) or not isinstance(cal.get("params"), dict):
            continue
        out.append({"file": p.name, "session": calibration_session(cal, p.name), "legacy": is_legacy(cal),
                    "ok": passes_gate(cal), "cal": cal})
    return out


# ── forward ledger and the VIX replication's departures ───────────────────────
def forward_ledger(store: Store) -> dict | None:
    """The forward ledger files, when they form an unbroken hash chain (each names
    the previous session and the SHA-256 of its canonical JSON)."""
    files = sorted((store.root / "forward").glob("ledger_*.json"))
    if not files:
        return None
    led = [json.loads(f.read_text()) for f in files]
    if any(x.get("schema") != "forward_ledger/1" or str(x.get("session")) != f.stem[7:]
           for x, f in zip(led, files)):
        return None
    for older, newer in itertools.pairwise(led):
        canon = json.dumps(older, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        prev = newer.get("previous") or {}
        if (prev.get("session") != older["session"]
                or prev.get("sha256") != hashlib.sha256(canon.encode()).hexdigest()):
            return None
    if led[0].get("previous") is not None:
        return None
    return {"n": len(led), "first": led[0]["session"], "last": led[-1]["session"]}


def vix_departures(rep: dict) -> int:
    """The lines Surface lists where the replication departs from Cboe: the
    deviations, the gap notes without the producer's gap-attribution sentence, and
    one rebuilt attribution line when the VIX gap splits exactly into its parts."""
    notes = [n for n in rep.get("gap_notes") or [] if not str(n).startswith("Gap attribution (")]
    ix = (rep.get("indices") or {}).get("VIX") or {}
    ga: dict = ix["gap_attribution"] if isinstance(ix.get("gap_attribution"), dict) else {}
    gap, tr, rs = (finite(ix.get("gap_vs_official")), finite(ga.get("truncation_pts")),
                   finite(ga.get("residual_pts")))
    split = (finite(ix.get("replicated")) is not None and bool(ga.get("available"))
             and gap is not None and tr is not None and rs is not None and abs(-tr - rs - gap) <= 1e-6)
    return len(rep.get("deviations") or []) + len(notes) + int(split)
