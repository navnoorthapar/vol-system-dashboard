"""
metrics.py — quote-grade fit metrics in spread units, and dense-grid static-
arbitrage checks for any model given as iv(k, T).

Per quote, with h = (ask_iv - bid_iv) / 2 (vol points):
  error_vp      model_iv - mid_iv (vol points)
  error_hs      error_vp / max(h, H_FLOOR_VP)   ("half-spreads")
  inside        bid_iv <= model_iv <= ask_iv    (equivalently the model price
                lies inside [bid, ask]: price is increasing in vol)
  outside_vp    distance from model_iv to the nearer spread edge when
                outside it (0 inside)

H_FLOOR_VP = 0.01 vol points only guards against zero-width (locked)
quotes; it is not a fitting weight. RMSE / median |error| are reported in
both vol points and half-spreads, overall, in- and out-of-sample, by expiry,
by tenor bucket, by delta bucket and on the tenor x delta grid.

Dense-grid checks (the same for every model, so a model cannot pass by
construction of its own test):
  butterfly  per quoted expiry, 201 strikes uniform in k over the quoted
             k-range widened by 10% of its span on each side; the model's
             forward-normalised call prices c(k) = C / (D F) must have no
             negative discrete butterfly (convexity in K = F e^k) beyond
             BUTTERFLY_TOL = 1e-9 (in units of D F: 7.7e-6 index points at
             F = 7,700), and no call-spread slope outside [-1, 0]
  calendar   between consecutive quoted expiries, on 201 points of the
             overlap of their k-ranges: total variance w(k, T) = iv^2 T must
             not decrease in T beyond CALENDAR_TOL = 1e-10
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr

from joint_vol_calibration.surface.quotes import DELTA_BUCKETS, TENOR_LABELS

H_FLOOR_VP = 0.01
BUTTERFLY_TOL = 1e-9
CALENDAR_TOL = 1e-10
GRID_POINTS = 201
GRID_PAD = 0.10


def _stats(err_vp: np.ndarray, err_hs: np.ndarray, inside: np.ndarray,
           outside_vp: np.ndarray) -> dict:
    n = int(len(err_vp))
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "inside_spread_pct": float(100.0 * inside.mean()),
        "rmse_vp": float(math.sqrt(np.mean(err_vp ** 2))),
        "median_abs_vp": float(np.median(np.abs(err_vp))),
        "mean_error_vp": float(np.mean(err_vp)),
        "rmse_hs": float(math.sqrt(np.mean(err_hs ** 2))),
        "median_abs_hs": float(np.median(np.abs(err_hs))),
        "within_2hs_pct": float(100.0 * np.mean(np.abs(err_hs) <= 2.0)),
        "rmse_outside_spread_vp": float(math.sqrt(np.mean(outside_vp ** 2))),
    }


def residual_frame(quotes: pd.DataFrame, model_iv, *, model: str,
                   in_sample=None, extra: Optional[dict] = None) -> pd.DataFrame:
    """Per-quote residuals of one model (the residuals.parquet rows)."""
    q = quotes
    miv = np.asarray(model_iv, dtype=float)
    hs = q["half_spread_vp"].to_numpy(float)
    err = 100.0 * (miv - q["mid_iv"].to_numpy(float))
    bid_iv = q["bid_iv"].to_numpy(float)
    ask_iv = q["ask_iv"].to_numpy(float)
    inside = (miv >= bid_iv) & (miv <= ask_iv)
    outside = 100.0 * np.where(miv > ask_iv, miv - ask_iv, np.where(miv < bid_iv, bid_iv - miv, 0.0))
    out = pd.DataFrame({
        "session": q["session"].to_numpy(),
        "model": model,
        "expiry": q["expiry"].to_numpy(),
        "T_days": q["T_days"].to_numpy(float),
        "strike": q["strike"].to_numpy(float),
        "right": q["right"].to_numpy(),
        "k": q["k"].to_numpy(float),
        "forward": q["forward"].to_numpy(float),
        "bid": q["bid"].to_numpy(float),
        "ask": q["ask"].to_numpy(float),
        "bid_iv": bid_iv,
        "mid_iv": q["mid_iv"].to_numpy(float),
        "ask_iv": ask_iv,
        "model_iv": miv,
        "vega": q["vega"].to_numpy(float),
        "half_spread_vp": hs,
        "error_vp": err,
        "error_hs": err / np.maximum(hs, H_FLOOR_VP),
        "inside_spread": inside,
        "outside_spread_vp": outside,
        "in_sample": (np.zeros(len(q), dtype=bool) if in_sample is None
                      else np.asarray(in_sample, dtype=bool)),
        "delta_bucket": q["delta_bucket"].to_numpy(),
        "tenor_bucket": q["tenor_bucket"].to_numpy(),
    })
    if extra:
        for col, vals in extra.items():
            out[col] = np.asarray(vals)
    return out


def summarise(res: pd.DataFrame) -> dict:
    """Metrics of one model's residual_frame (finite model IVs only; the
    count of quotes the model could not price is reported)."""
    ok = np.isfinite(res["model_iv"].to_numpy(float))
    r = res[ok]

    def st(mask):
        sub = r[mask]
        return _stats(sub["error_vp"].to_numpy(float), sub["error_hs"].to_numpy(float),
                      sub["inside_spread"].to_numpy(bool), sub["outside_spread_vp"].to_numpy(float))

    allm = np.ones(len(r), dtype=bool)
    ins = r["in_sample"].to_numpy(bool)
    out = {"all": st(allm), "in_sample": st(ins), "out_of_sample": st(~ins),
           "n_unpriced": int((~ok).sum())}
    if "arb_flag" in r.columns:
        out["excluding_arbitrage_flagged"] = st(~r["arb_flag"].to_numpy(bool))
    by_exp = {}
    for exp, g in r.groupby("expiry", sort=False):
        d = _stats(g["error_vp"].to_numpy(float), g["error_hs"].to_numpy(float),
                   g["inside_spread"].to_numpy(bool), g["outside_spread_vp"].to_numpy(float))
        d["T_days"] = float(g["T_days"].iloc[0])
        by_exp[str(exp)] = d
    out["by_expiry"] = by_exp
    out["by_tenor"] = {t: st((r["tenor_bucket"] == t).to_numpy()) for t in TENOR_LABELS
                       if (r["tenor_bucket"] == t).any()}
    out["by_delta"] = {b: st((r["delta_bucket"] == b).to_numpy()) for b in DELTA_BUCKETS
                       if (r["delta_bucket"] == b).any()}
    grid = {}
    for t in TENOR_LABELS:
        for b in DELTA_BUCKETS:
            m = ((r["tenor_bucket"] == t) & (r["delta_bucket"] == b)).to_numpy()
            if m.any():
                s = st(m)
                grid[f"{t}|{b}"] = {key: s[key] for key in
                                    ("n", "inside_spread_pct", "rmse_vp", "rmse_hs",
                                     "median_abs_hs", "mean_error_vp")}
    out["by_tenor_delta"] = grid
    return out


# ── dense-grid static-arbitrage checks ────────────────────────────────────────

def _norm_call(k, w):
    """Forward-normalised Black call price c = C / (D F) at total variance w."""
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = -k / sw + 0.5 * sw
    return ndtr(d1) - np.exp(k) * ndtr(d1 - sw)


def expiry_ranges(quotes: pd.DataFrame) -> list:
    """[(expiry, T, k_lo, k_hi)] in T order: quoted k-range widened by GRID_PAD."""
    out = []
    for exp, g in quotes.groupby("expiry", sort=False):
        lo, hi = float(g["k"].min()), float(g["k"].max())
        pad = GRID_PAD * max(hi - lo, 1e-3)
        out.append((str(exp), float(g["T"].iloc[0]), lo - pad, hi + pad))
    out.sort(key=lambda r: r[1])
    return out


def dense_grid_checks(iv_fn: Callable, ranges: list, n: int = GRID_POINTS) -> dict:
    """Butterfly / call-spread / calendar violations of iv_fn(k, T) on the
    dense grids of ``ranges`` (see module docstring)."""
    n_bf = n_vs = n_cal = 0
    min_bf, min_cal = np.inf, np.inf
    n_bf_pts = n_cal_pts = 0
    bad_bf, bad_cal = [], []
    nonfinite = 0
    for exp, T, lo, hi in ranges:
        k = np.linspace(lo, hi, n)
        iv = np.asarray(iv_fn(k, T), dtype=float)
        fin = np.isfinite(iv)
        nonfinite += int((~fin).sum())
        if fin.sum() < 3:
            continue
        kk, w = k[fin], iv[fin] ** 2 * T
        c = _norm_call(kk, w)
        K = np.exp(kk)
        slope = np.diff(c) / np.diff(K)
        n_vs += int(((slope > 1e-12) | (slope < -1.0 - 1e-12)).sum())
        # discrete butterfly on the (non-uniform in K) grid, per unit of D F
        hl, hr = np.diff(K)[:-1], np.diff(K)[1:]
        bf = (hr * c[:-2] - (hl + hr) * c[1:-1] + hl * c[2:]) / (hl + hr)
        n_bf_pts += len(bf)
        n_bad = int((bf < -BUTTERFLY_TOL).sum())
        n_bf += n_bad
        min_bf = min(min_bf, float(bf.min()))
        if n_bad:
            bad_bf.append(exp)
    for (e1, T1, lo1, hi1), (e2, T2, lo2, hi2) in zip(ranges[:-1], ranges[1:]):
        lo, hi = max(lo1, lo2), min(hi1, hi2)
        if hi <= lo:
            continue
        k = np.linspace(lo, hi, n)
        w1 = np.asarray(iv_fn(k, T1), dtype=float) ** 2 * T1
        w2 = np.asarray(iv_fn(k, T2), dtype=float) ** 2 * T2
        fin = np.isfinite(w1) & np.isfinite(w2)
        d = (w2 - w1)[fin]
        if not len(d):
            continue
        n_cal_pts += len(d)
        n_bad = int((d < -CALENDAR_TOL).sum())
        n_cal += n_bad
        min_cal = min(min_cal, float(d.min()))
        if n_bad:
            bad_cal.append(f"{e1}->{e2}")
    return {
        "grid_points_per_expiry": n,
        "butterfly_points": n_bf_pts, "butterfly_violations": n_bf,
        "butterfly_min_per_DF": (float(min_bf) if np.isfinite(min_bf) else None),
        "butterfly_violating_expiries": bad_bf,
        "call_spread_slope_violations": n_vs,
        "calendar_points": n_cal_pts, "calendar_violations": n_cal,
        "calendar_min_dw": (float(min_cal) if np.isfinite(min_cal) else None),
        "calendar_violating_pairs": bad_cal,
        "nonfinite_iv_points": nonfinite,
        "tolerances": {"butterfly_per_DF": BUTTERFLY_TOL, "calendar_total_variance": CALENDAR_TOL},
    }
