#!/usr/bin/env python3
"""Like-for-like Heston baseline for the Quintic OU research fit.

Writes data_store/calibrations/baseline_<session>.json for the session of the
stored quintic_cal_<session>.pkl. The Heston side is the model_version 2 joint
calibration of the SAME session (joint_cal_<session>.pkl), so both models see
the same SPX chain, the same VIX indices and the same day.

Two Heston views are recorded:

  * own_targets: the joint fit's own metrics (in-sample z-grid anchors, the
    out-of-sample full OTM surface, the VIX spot-index leg).
  * on_quintic_targets: the SAME Heston parameters re-priced on the Quintic
    calibrator's exact targets, with the Quintic's definitions:
      - SPX: Black-76 IV on each row's parity forward, model vs market, in vol
        points (the Quintic's spx_iv_rmse);
      - VIX forwards: Heston E[VIX_T] (heston_vix_future_exact) vs the
        VIX-option parity forward, in VIX points (the Quintic's
        vix_futures_rmse, which is not a VX-futures fit);
      - VIX options: Black-76 IVs at equal K/F, market on its parity forward,
        model on its own forward (the Quintic's vix_options_rmse).
    The Quintic was fitted to these targets and Heston was not (its w3 = 0 and
    its SPX anchors are a different strike set), so this view favours the
    Quintic; the own_targets view is the Heston fit on its own terms.

``params`` stays at the top level: nn_pricer reads it as the reference fit.

Usage:  python scripts/heston_quintic_baseline.py [SESSION]   (default: newest quintic_cal_*.pkl)
"""
from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_vol_calibration.calibration.joint_calibrator import (  # noqa: E402
    heston_vix_call_price, heston_vix_put_price, is_acceptable_calibration,
)
from joint_vol_calibration.models.heston import (  # noqa: E402
    heston_call_batch, heston_vix_future_exact,
)
from joint_vol_calibration.models.implied_vol import implied_vol_black  # noqa: E402

CAL_DIR = ROOT / "data_store" / "calibrations"


def _rmse(a) -> float:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(math.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def heston_spx_ivs(p: dict, rows: list) -> np.ndarray:
    """Black-76 IVs of Heston OTM prices at each row's (F, K, T); forward-normalised."""
    out = np.full(len(rows), np.nan)
    for T in sorted({r["T"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["T"] == T]
        kf = np.array([rows[i]["K"] / rows[i]["forward"] for i in idx])
        calls = heston_call_batch(1.0, kf, T, 0.0, 0.0, p["kappa"], p["theta"],
                                  p["sigma"], p["rho"], p["v0"])
        is_call = kf >= 1.0
        otm = np.where(is_call, calls, calls - (1.0 - kf))
        out[idx] = implied_vol_black(otm, 1.0, kf, T, is_call=is_call, min_otm_price=1e-10)
    return out


def heston_vix_ivs(p: dict, rows: list) -> tuple[np.ndarray, dict]:
    """Model VIX-option IVs at equal K/F on the model's own forward E[VIX_T]."""
    fwd = {}
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        T = r["T"]
        if T not in fwd:
            fwd[T] = float(heston_vix_future_exact(p["kappa"], p["theta"], p["sigma"], p["v0"], T))
        F_mod = fwd[T]
        K_mod = r["K"] * F_mod / r["forward"]
        is_call = r["right"] == "C"
        fn = heston_vix_call_price if is_call else heston_vix_put_price
        price = fn(K_mod, T, p["kappa"], p["theta"], p["sigma"], p["v0"], r=0.0)
        out[i] = float(implied_vol_black(price, F_mod, K_mod, T, is_call=is_call))
    return out, fwd


def compute_baseline(session: str | None = None, cal_dir: Path = CAL_DIR) -> dict:
    """The baseline record for `session` (reads the two pickles, writes nothing).

    Raises ValueError unless the Quintic and Heston fits describe the same spot
    session and the same SPX option snapshot.
    """
    cal_dir = Path(cal_dir)
    if session is None:
        qpath = sorted(cal_dir.glob("quintic_cal_*.pkl"))[-1]
        session = qpath.stem.replace("quintic_cal_", "")
    qpath = cal_dir / f"quintic_cal_{session}.pkl"
    hpath = cal_dir / f"joint_cal_{session}.pkl"
    with open(qpath, "rb") as f:
        q = pickle.load(f)
    with open(hpath, "rb") as f:
        h = pickle.load(f)
    qprov = q.get("provenance") or {}
    if qprov.get("spot_date") != h.get("spot_date") or \
            qprov.get("spx_snapshot_date") != h.get("spx_snapshot_date"):
        raise ValueError(f"not like-for-like: quintic {qprov.get('spot_date')}/"
                         f"{qprov.get('spx_snapshot_date')} vs heston {h.get('spot_date')}/"
                         f"{h.get('spx_snapshot_date')}")
    p = h["params"]
    hl = h["leg_losses"]

    spx_model = heston_spx_ivs(p, q["spx_rows"])
    spx_mkt = np.array([r["market_iv"] for r in q["spx_rows"]])
    vix_model, _ = heston_vix_ivs(p, q["vix_rows"])
    vix_mkt = np.array([r["market_iv"] for r in q["vix_rows"]])
    fwd_model = np.array([float(heston_vix_future_exact(p["kappa"], p["theta"], p["sigma"],
                                                        p["v0"], r["T"]))
                          for r in q["vix_forward_rows"]])
    fwd_mkt = np.array([r["forward_mkt"] for r in q["vix_forward_rows"]])
    ql = q["leg_losses"]
    gate_ok, gate_reason = is_acceptable_calibration(h)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10).stdout.strip() or None
    except Exception:
        sha = None

    out = {
        "session": session,
        "as_of_date": session,
        "description": ("Like-for-like Heston baseline for the Quintic OU research fit: the "
                        "model_version 2 joint Heston calibration of the same session, on "
                        "its own targets and re-priced on the Quintic's targets"),
        "params": p,
        "heston": {
            "file": hpath.name,
            "model_version": h.get("model_version"),
            "gate_ok": bool(gate_ok), "gate_reason": gate_reason,
            "feller": h.get("feller"), "feller_state": h.get("feller_state"),
            "feller_margin": h.get("feller_margin"),
            "active_constraints": h.get("active_constraints"),
            "S": h.get("S"), "r": h.get("r"), "q": h.get("q"),
            "r_reference": h.get("r_reference"),
            "spx_snapshot_date": h.get("spx_snapshot_date"),
            "n_spx_options": h.get("n_spx_options"), "n_spx_expiries": h.get("n_spx_expiries"),
            "n_vix_tenors": h.get("n_vix_tenors"),
        },
        "own_targets": {
            "spx_iv_rmse": hl.get("spx_iv_rmse"),
            "spx_iv_rmse_definition": "vol points, in-sample z-grid anchors (vega-approximated)",
            "spx_oos_iv_rmse": hl.get("spx_oos_iv_rmse"),
            "spx_oos_put_wing_rmse": hl.get("spx_oos_put_wing_rmse"),
            "n_spx_oos": (h.get("oos") or {}).get("n_spx_oos"),
            "vix_index_rmse": hl.get("vix_index_rmse"),
            "vix_index_definition": "VIX points, ^VIX9D/^VIX/^VIX3M/^VIX6M vs heston_vix_term_index",
            "vx_futures_rmse_oos": hl.get("vx_futures_rmse_oos"),
        },
        "on_quintic_targets": {
            "spx_iv_rmse": 100.0 * _rmse(spx_model - spx_mkt),
            "n_spx": int(np.isfinite(spx_model).sum()), "n_spx_rows": len(spx_mkt),
            "vix_forward_rmse": _rmse(fwd_model - fwd_mkt),
            "vix_forward_rel_rmse": _rmse((fwd_model - fwd_mkt) / fwd_mkt),
            "n_vix_forwards": len(fwd_mkt),
            "vix_options_rmse": 100.0 * _rmse(vix_model - vix_mkt),
            "n_vix_options": int(np.isfinite(vix_model).sum()), "n_vix_option_rows": len(vix_mkt),
            "note": ("Heston was not fitted to these targets (VIX-options weight 0, different "
                     "SPX strike set); the Quintic was."),
        },
        "quintic": {
            "file": qpath.name,
            "spx_iv_rmse": ql.get("spx_iv_rmse"),
            "spx_iv_rmse_fresh_paths": ql.get("spx_iv_rmse_fresh_paths"),
            "vix_forward_rmse": ql.get("vix_futures_rmse"),
            "vix_forward_rel_rmse": ql.get("vix_futures_rel_rmse"),
            "vix_options_rmse": ql.get("vix_options_rmse"),
            "xi0_index_reproduction_rmse": ql.get("xi0_index_reproduction_rmse"),
            "success": q.get("success"),
            "at_bound": [k for k, v in (q.get("at_bound") or {}).items() if v],
            "n_spx_options": qprov.get("n_spx_options"),
            "n_vix_options": qprov.get("n_vix_options"),
            "spx_snapshot_date": qprov.get("spx_snapshot_date"),
            "vix_snapshot_date": qprov.get("vix_snapshot_date"),
            "r": qprov.get("r"), "r_source": qprov.get("r_source"),
            "code_sha": (qprov.get("code") or {}).get("sha"),
        },
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_sha": sha,
        "generator": "scripts/heston_quintic_baseline.py",
    }
    return out


def build(session: str | None = None, cal_dir: Path = CAL_DIR) -> Path:
    """Write cal_dir/baseline_<session>.json (see compute_baseline)."""
    out = compute_baseline(session, cal_dir)
    path = Path(cal_dir) / f"baseline_{out['session']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path


if __name__ == "__main__":
    p = build(sys.argv[1] if len(sys.argv) > 1 else None)
    d = json.loads(p.read_text())
    print(p)
    print(json.dumps({k: d[k] for k in ("own_targets", "on_quintic_targets", "quintic")}, indent=2))
