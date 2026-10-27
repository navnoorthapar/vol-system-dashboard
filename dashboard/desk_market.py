"""
desk_market.py — market series the desk reads beside the models (pure functions;
no Flask, no I/O: the caller reads the rows from data_store/vol_system.db).

The VX futures leg of a joint fit (Calibration #vixleg)
  The joint calibrator scores the fitted Heston parameters against the VX
  monthly futures settlements of its own session, out of sample: the loss has
  no VX-futures term. ``vx_model_leg`` repeats that score from the stored rows
  so the page can draw it: the model price of each contract is
  E^Q[VIX_T] = heston_vix_future_exact(κ, θ, σ, v₀, days_to_expiry / 365) in
  index points, against the contract's daily settlement (COALESCE(settle,
  close), monthly contracts only, live on the session: the rows of the
  calibrator's _vx_futures_oos), and the RMSE over them equals the fit's
  recorded losses.vx_futures_rmse_oos. Settlements are only ever paired with
  the fit of the same session: a fit whose session has no stored settlements
  has no VX leg (an older curve is never spliced onto a newer fit).

Implied against subsequently realised vol (Research #vrp)
  r_t = ln(close_t / close_{t−1}) of SPX (the returns the forecast evaluation
  uses). Both realised vols are root-mean-square, annualised, in vol pts:
    RV21 trailing(t) = 100·√(252/21 · Σ_{i=0..20} r²_{t−i})   (known at the close of t)
    RV21 forward(t)  = 100·√(252/21 · Σ_{i=1..21} r²_{t+i})   (the next 21 sessions)
  the forward one being forecast_eval.json's target y(t), so it has no value in
  the last 21 sessions. The ex-post premium of a mark at β·VIX is
  vrp_β(t) = β·VIX_t − RV21 forward(t) (vol pts) and its variance-space form
  (β·VIX_t)² − RV21 forward(t)² (vol pts²). VIX is a 30-calendar-day
  variance-swap rate, skew premium included, so VIX − RV overstates what an
  at-the-money straddle earns; 21-session windows overlap, so shares and means
  over consecutive sessions are not independent observations. A description,
  not a P&L.

The ticker's VIX percentile (B13)
  pctile_prior: the newest close against the 252 sessions before it, the
  percent of them strictly lower (ratio_view's convention, on a fixed window).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0          # the calibrator's T = days_to_expiry / 365
RV_WINDOW = 21                 # sessions: the forecast target's horizon (forecast_eval.json target)
ANNUALISE = 252
CURVE_STEP_DAYS = 5            # spacing of the drawn model curve E^Q[VIX_T]
PCTILE_WINDOW = 252            # sessions (about a year): the ticker's VIX percentile


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def live_monthly(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """The rows the calibrator scores: monthly contracts with a positive price and
    a positive time to expiry, front month first. ``price`` is the settlement, else
    the last trade (the DB's COALESCE(settle, close))."""
    out = []
    for r in rows:
        if int(r.get("is_weekly") or 0):
            continue
        px, days = _num(r.get("price")), _num(r.get("days_to_expiry"))
        if px is None or days is None or px <= 0 or days <= 0:
            continue
        out.append({"expiry": str(r.get("expiry_date") or ""), "month": str(r.get("expiry_month") or ""),
                    "days": days, "settle": px})
    return sorted(out, key=lambda r: (r["days"], r["expiry"]))


def vx_model_leg(params: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                 future_fn: Any = None) -> dict | None:
    """The fit's VX futures leg: per contract the settlement, the model E^Q[VIX_T]
    and the gap (model − settlement, index pts), their RMSE, and the model curve
    from T = 0 (the model's spot VIX) to the last contract, for drawing.

    ``rows``: the session's futures rows (see live_monthly). None when no contract
    is left: the session stores no settlements, and nothing is put in their place.
    """
    live = live_monthly(rows)
    if not live:
        return None
    if future_fn is None:
        from joint_vol_calibration.models.heston import heston_vix_future_exact as future_fn
    k, th, sg, v0 = (float(params[x]) for x in ("kappa", "theta", "sigma", "v0"))
    T = np.array([r["days"] for r in live], dtype=float) / DAYS_PER_YEAR
    model = np.atleast_1d(future_fn(k, th, sg, v0, T)).astype(float)
    out_rows = []
    for r, m in zip(live, model):
        out_rows.append({**r, "model": float(m), "gap": float(m) - r["settle"]})
    gaps = np.array([r["gap"] for r in out_rows])
    last = max(r["days"] for r in live)
    grid = sorted({0.0, *np.arange(0.0, last + 1e-9, CURVE_STEP_DAYS).tolist(), *(r["days"] for r in live)})
    curve_vals = np.atleast_1d(future_fn(k, th, sg, v0, np.array(grid) / DAYS_PER_YEAR)).astype(float)
    return {"rows": out_rows, "n": len(out_rows), "rmse": float(np.sqrt(np.mean(gaps ** 2))),
            "mean_gap": float(np.mean(gaps)), "n_below": int(np.sum(gaps < 0)),
            "curve": [[float(d), float(v)] for d, v in zip(grid, curve_vals)]}


# ── implied against subsequently realised vol ────────────────────────────────

def rv_trailing(r: pd.Series, n: int = RV_WINDOW) -> pd.Series:
    """RMS realised vol of the n sessions ending at t (including t), annualised, vol pts."""
    return np.sqrt((r.astype(float) ** 2).rolling(n).mean() * ANNUALISE) * 100.0


def rv_forward(r: pd.Series, n: int = RV_WINDOW) -> pd.Series:
    """RMS realised vol of the n sessions after t (t+1 … t+n), annualised, vol pts;
    NaN for the last n sessions (forecast_eval.json's target, as a vol)."""
    return rv_trailing(r, n).shift(-n)


def vrp(vix: pd.Series, rv_fwd: pd.Series, beta: float = 1.0) -> pd.Series:
    """Ex-post premium of a mark at β·VIX: β·VIX_t − RV21 forward(t), vol pts."""
    return beta * vix.astype(float) - rv_fwd


def vrp_stats(vix: pd.Series, rv_fwd: pd.Series, beta: float, start: str | None = None,
              end: str | None = None) -> dict | None:
    """Mean, median and share above zero of vrp_β over [start, end] (sessions with a
    forward value), and the variance-space mean (β·VIX)² − RV², vol pts²."""
    v = vrp(vix, rv_fwd, beta).loc[start:end].dropna()
    if v.empty:
        return None
    rf = rv_fwd.loc[v.index]
    var = (beta * vix.loc[v.index].astype(float)) ** 2 - rf ** 2
    return {"beta": float(beta), "n": len(v), "mean": float(v.mean()), "median": float(v.median()),
            "share_pos": float((v > 0).mean() * 100.0), "var_mean": float(var.mean()),
            "first": str(pd.Timestamp(v.index[0]).date()), "last": str(pd.Timestamp(v.index[-1]).date())}


def pctile_prior(x: pd.Series, window: int = PCTILE_WINDOW) -> dict | None:
    """The newest value against the ``window`` sessions before it (NaN dropped,
    the newest left out): ``n_below`` of them strictly lower (a tie is not
    below) and ``pct`` = 100·n_below / window, the convention of ratio_view and
    the Overview's percentiles. None with fewer than ``window`` earlier values:
    no shorter window is put in its place."""
    s = x.dropna().astype(float)
    if len(s) < window + 1:
        return None
    today = float(s.iloc[-1])
    prior = s.iloc[-window - 1:-1]
    n_below = int((prior < today).sum())
    return {"date": str(pd.Timestamp(s.index[-1]).date()), "value": today, "n_below": n_below,
            "n": int(window), "pct": n_below / window * 100.0,
            "first": str(pd.Timestamp(prior.index[0]).date()),
            "last": str(pd.Timestamp(prior.index[-1]).date())}


def yearly_means(x: pd.Series) -> list[dict]:
    """Calendar-year means of a dated series (NaN dropped); a last year that stops
    before December is marked partial with the span it covers (a 'YTD' bar)."""
    x = x.dropna()
    if x.empty:
        return []
    idx = pd.DatetimeIndex(x.index)
    out = []
    for yr, g in x.groupby(idx.year):
        gi = pd.DatetimeIndex(g.index)
        out.append({"year": int(yr), "mean": float(g.mean()), "n": len(g),
                    "first": str(gi[0].date()), "last": str(gi[-1].date()),
                    "partial": bool(yr == idx[-1].year and idx[-1].month < 12)})
    return out


# ── the VIX complex on the Overview (P3.5) ───────────────────────────────────

def vx_curve(rows: Sequence[Mapping[str, Any]]) -> dict | None:
    """One session's VX monthly curve (live_monthly: front month first) and its
    M1 → M2 slope, VX2 / VX1 − 1 (positive: contango). None without a contract."""
    live = live_monthly(rows)
    if not live:
        return None
    m1m2 = (live[1]["settle"] / live[0]["settle"] - 1.0) if len(live) >= 2 else None
    return {"rows": live, "n": len(live), "m1m2": m1m2,
            "m1": live[0], "m2": live[1] if len(live) >= 2 else None}


def ratio_view(ratio: pd.Series, threshold: float = 1.0) -> dict | None:
    """VIX/VIX3M: the share of sessions above ``threshold`` (backwardation), the
    newest value and its percentile among the PRIOR sessions (percent strictly
    below it, NaN dropped), and the runs above the threshold (for shading)."""
    r = ratio.dropna().astype(float)
    if len(r) < 2:
        return None
    above = r > threshold
    today = float(r.iloc[-1])
    prior = r.iloc[:-1]
    runs: list[tuple[str, str]] = []
    start = None
    idx = [str(pd.Timestamp(t).date()) for t in r.index]
    for i, on in enumerate(above.to_numpy()):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(r) - 1):
            end = i if on else i - 1
            runs.append((idx[start], idx[end]))
            start = None
    return {"n": len(r), "first": idx[0], "last": idx[-1], "today": today,
            "share_above": float(above.mean() * 100.0), "n_above": int(above.sum()),
            "pct": float((prior < today).mean() * 100.0), "n_prior": len(prior), "runs": runs}
