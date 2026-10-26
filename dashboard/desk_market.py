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
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DAYS_PER_YEAR = 365.0          # the calibrator's T = days_to_expiry / 365
CURVE_STEP_DAYS = 5            # spacing of the drawn model curve E^Q[VIX_T]


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
