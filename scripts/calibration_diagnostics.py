#!/usr/bin/env python
"""
calibration_diagnostics.py — measured calibration diagnostics for every
committed model_version 2 session (C23).

For each session with an accepted joint_cal_<session>.pkl and its stored SPX
chain, writes data_store/calibrations/diagnostics/<session>.json:

  stored_fit        the committed DE + L-BFGS-B fit (params, loss, time, evals)
                    and the loss recomputed from the residual vector
  trf               the default multi-start TRF fit: loss vs stored, params,
                    time, evaluations, every start, basins, warm start
  trf_finite_diff   the same TRF with a '2-point' Jacobian (time, evals, loss)
  jacobian          analytic vs finite-difference Jacobian at the optimum:
                    accuracy against a central difference and time per call
  de                DE (same settings as the committed fits) in 4 processes,
                    TRF-polished: the optional global check, timed; with
                    --time-serial-de also the serial run on this machine
  identifiability   singular values, cond(J^T J), weakest direction, Gauss-
                    Newton standard errors / correlations, parameters on bounds
  sigma_profile     the joint fit under every config.SIGMA_PROFILE_CAPS cap
                    (kappa <= PROFILE_KAPPA_MAX), with SPX out-of-sample, VIX
                    index, VX futures and VVIX checks
  joint_rough_heston  the production joint loss (SPX anchors + VIX indices)
                    with lifted rough Heston; compare with "trf" and the
                    last sigma_profile row (the joint Heston fits)
  spx_only          Heston (production box and the wide profile box) and
                    lifted rough Heston fitted SPX-only with the same
                    residuals and optimiser: in-sample, out-of-
                    sample (all, put wing, by expiry), VIX index (out of
                    sample), fit time, starts, numerics check (rough)

and data_store/calibrations/diagnostics/summary.json (one row per session).
Everything is derived from committed data (the DB copy and the stored
chains); nothing is downloaded and no calibration pickle is written.

Usage:
  python scripts/calibration_diagnostics.py                 # every v2 session
  python scripts/calibration_diagnostics.py --session 2026-10-16
  python scripts/calibration_diagnostics.py --summary-only  # rebuild summary.json
Each session takes about 3-6 minutes on 4 CPUs (most of it the rough
Heston fit); run sessions as separate invocations to stay under a job time
limit. Timings are wall clock and the file records the load average: compare
methods within one file (they ran back to back), not seconds across files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import pickle
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_vol_calibration import config  # noqa: E402
from joint_vol_calibration.calibration import joint_calibrator as jc  # noqa: E402

OUT_DIR = config.DATA_DIR / "calibrations" / "diagnostics"
SCHEMA_VERSION = 1
# Sobol starts per rough fit (plus 2 warm starts from the Heston fit at
# H = 0.30 and 0.10). On 2026-10-16 all 6 starts of a 4-Sobol run reached one
# basin; 2 keeps a session under ~15 min on a loaded 4-CPU machine.
ROUGH_SOBOL_STARTS = 2
log = logging.getLogger("calibration_diagnostics")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        # strict JSON: NaN / inf (np.float64 is a float subclass) become null
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def committed_v2_sessions(cal_dir: Path | None = None) -> list:
    """Sessions with an ACCEPTED model_version >= 2 joint fit and a stored chain."""
    cal_dir = cal_dir or config.DATA_DIR / "calibrations"
    out = []
    for p in sorted(cal_dir.glob("joint_cal_*.pkl")):
        with open(p, "rb") as f:
            res = pickle.load(f)
        if jc.calibration_is_legacy(res) or not jc.is_acceptable_calibration(res)[0]:
            continue
        session = p.stem[len("joint_cal_"):]
        if (config.DATA_DIR / "options" / f"spx_{session}.parquet").exists():
            out.append(session)
    return out


def _cpu_seconds() -> float:
    """CPU seconds of this process (all threads) plus its reaped children
    (the DE worker processes, which scipy's pool joins on exit)."""
    s = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


def _timed(fn, *args, **kwargs):
    """(result, wall seconds, CPU seconds). CPU time is the robust number on
    a shared machine: wall clock grows with the load, CPU time much less."""
    w0, c0 = time.perf_counter(), _cpu_seconds()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - w0, _cpu_seconds() - c0


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _jacobian_timing(model, x, reps: int = 5) -> dict:
    """Analytic vs finite-difference Jacobian at x: accuracy and time."""
    from scipy.optimize._numdiff import approx_derivative

    x = np.asarray(x, dtype=float)
    lo, hi = jc._box_arrays(config.HESTON_BOUNDS, jc.HESTON_PARAM_NAMES)
    model(x)
    J = model.jac(x)
    # central differences with a step well above the 1e-10 quadrature noise
    Jc = np.zeros_like(J)
    for j in range(5):
        h = 1e-5 * abs(x[j])
        up, dn = x.copy(), x.copy()
        up[j] = min(up[j] + h, hi[j])
        dn[j] = max(dn[j] - h, lo[j])
        Jc[:, j] = (model(up) - model(dn)) / (up[j] - dn[j])
    rel = np.abs(J - Jc).max(axis=0) / np.abs(Jc).max(axis=0)
    t_a, t_f = [], []
    for _ in range(reps):
        model(x)                       # least_squares calls r(x) before jac(x)
        t0 = time.perf_counter()
        model.jac(x)
        t_a.append(time.perf_counter() - t0)
        f0 = model(x)
        t0 = time.perf_counter()
        approx_derivative(model, x, method="2-point", f0=f0, bounds=(lo, hi))
        t_f.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    model(x)
    t_r = time.perf_counter() - t0
    return {
        "analytic_vs_central_fd_max_rel_diff_by_param": dict(zip(jc.HESTON_PARAM_NAMES,
                                                                 rel.tolist())),
        "seconds_analytic_jacobian": float(np.median(t_a)),
        "seconds_2point_fd_jacobian": float(np.median(t_f)),
        "seconds_one_residual_eval": float(t_r),
        "speedup_vs_2point": float(np.median(t_f) / np.median(t_a)),
    }


def diagnose_session(session: str, time_serial_de: bool = False, de_workers: int = 4) -> dict:
    t_all = time.perf_counter()
    with open(config.DATA_DIR / "calibrations" / f"joint_cal_{session}.pkl", "rb") as f:
        stored = pickle.load(f)
    cal = jc.JointCalibrator(as_of_date=session)
    names = jc.HESTON_PARAM_NAMES
    x_stored = np.array([stored["params"][k] for k in names])
    r = cal.residuals(x_stored)
    out = {
        "schema_version": SCHEMA_VERSION,
        "session": session,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "code_commit": _git_head(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__,
                        "cpus": os.cpu_count(), "machine": platform.machine(),
                        # timings below are wall-clock on a shared machine:
                        # compare methods within one file (run back to back),
                        # not seconds across files or machines
                        "loadavg_start": list(os.getloadavg())},
        "data": {"n_spx_anchors": int(len(cal.spx_surface)),
                 "n_spx_expiries": int(cal.spx_surface["expiry"].nunique()),
                 "n_spx_oos": int(len(cal.spx_surface_full) - len(cal.spx_surface)),
                 "n_vix_tenors": int(len(cal.vix_ts)), "n_vx_futures": int(len(cal.vx_futures)),
                 "vvix_market": cal.vvix_market},
        "stored_fit": {
            "file": f"joint_cal_{session}.pkl",
            "params": stored["params"], "loss": stored["loss"],
            "loss_recomputed_from_residuals": float(r @ r),
            "fit_time_seconds": stored.get("fit_time"), "n_evals": stored.get("n_evals"),
            "optimizer": (stored.get("methods") or {}).get("optimizer"),
            "pinned_params": stored.get("pinned_params"),
            "spx_iv_rmse": stored["leg_losses"].get("spx_iv_rmse"),
            "spx_oos_iv_rmse": stored["leg_losses"].get("spx_oos_iv_rmse"),
            **cal._vvix_check(*(stored["params"][k] for k in ("kappa", "theta", "sigma", "v0"))),
        },
    }

    # 1. default TRF
    res, trf_wall, trf_cpu = _timed(cal.calibrate, verbose=False)
    trf_x = np.array([res["params"][k] for k in names])
    out["trf"] = {
        "params": res["params"], "loss": res["loss"],
        "loss_rel_diff_vs_stored": res["loss"] / stored["loss"] - 1.0,
        "param_rel_diff_vs_stored": {k: res["params"][k] / stored["params"][k] - 1.0
                                     for k in names},
        "fit_time_seconds": res["fit_time"], "n_evals": res["n_evals"],
        "fit_cpu_seconds": res["fit_cpu_time"],
        "wall_seconds_incl_diagnostics": trf_wall, "cpu_seconds_incl_diagnostics": trf_cpu,
        "success": res["success"], "message": res["optimizer_message"],
        "warm_start": res["methods"]["optimizer"]["warm_start"],
        "multistart": res["multistart"],
        "spx_iv_rmse": res["leg_losses"]["spx_iv_rmse"],
        "spx_oos_iv_rmse": res["leg_losses"]["spx_oos_iv_rmse"],
        "pinned_params": res["pinned_params"],
        "gate": list(jc.is_acceptable_calibration(res)),
    }
    out["identifiability"] = res["identifiability"]
    lo, hi = jc._box_arrays(config.HESTON_BOUNDS, names)
    out["identifiability"]["profile_check"] = jc.gauss_newton_profile_check(
        cal.residual_model(), trf_x, res["identifiability"], lo, hi)
    log.info("%s TRF: loss %.8g (stored %.8g), %.1fs, %d evals", session, res["loss"],
             stored["loss"], res["fit_time"], res["n_evals"])

    # 2. TRF with a finite-difference Jacobian (same starts)
    fd, _, fd_cpu = _timed(cal.calibrate, verbose=False, jacobian="2-point")
    out["trf_finite_diff"] = {"loss": fd["loss"], "fit_time_seconds": fd["fit_time"],
                              "fit_cpu_seconds": fd["fit_cpu_time"],
                              "cpu_seconds_incl_diagnostics": fd_cpu,
                              "n_evals": fd["n_evals"],
                              "loss_rel_diff_vs_analytic": fd["loss"] / res["loss"] - 1.0,
                              "n_distinct_basins": fd["multistart"]["n_distinct_basins"]}
    out["jacobian"] = _jacobian_timing(cal.residual_model(), trf_x)

    # 3. DE as the optional global check (processes), and optionally serial
    chk_res, _, chk_cpu = _timed(cal.calibrate, verbose=False, de_check=True,
                                 de_workers=de_workers)
    chk = chk_res["de_check"]
    out["de"] = {"settings": {"maxiter": 200, "popsize": 10, "seed": config.RANDOM_SEED,
                              "workers": de_workers, "updating": "deferred"},
                 # TRF fit + DE in worker processes + TRF polish, all CPU
                 "cpu_seconds_trf_plus_de_check": chk_cpu,
                 **chk}
    if time_serial_de:
        ser, ser_wall, ser_cpu = _timed(cal.calibrate, verbose=False, optimizer="de",
                                        de_workers=1)
        out["de"]["serial_rerun"] = {"seconds": ser_wall, "cpu_seconds": ser_cpu,
                                     "fit_time_seconds": ser["fit_time"],
                                     "fit_cpu_seconds": ser["fit_cpu_time"],
                                     "loss": ser["loss"], "n_evals": ser["n_evals"],
                                     "loss_rel_diff_vs_stored": ser["loss"] / stored["loss"] - 1}
    log.info("%s DE check (%d workers): %.1fs, loss %.8g, agrees=%s", session, de_workers,
             chk["de_seconds"], chk["de_polished_loss"], chk["agrees"])

    # 4. sigma profile (diagnostic; the production fit is untouched)
    cal.calibrate(verbose=False)
    out["sigma_profile"] = cal.sigma_profile(base=stored["params"])
    log.info("%s sigma profile done", session)

    # 5. SPX-only Heston vs lifted rough Heston, same residuals and optimiser
    spx = {}
    for key, model, box in (("heston", "heston", "production"),
                            ("heston_wide_box", "heston", "wide"),
                            ("rough_heston", "rough_heston", "production")):
        spx[key] = cal.calibrate_spx_only(model, warm=stored["params"], box=box,
                                          n_starts=ROUGH_SOBOL_STARTS if model != "heston" else 4)
        log.info("%s SPX-only %s: in %.3f vp, OOS %.3f vp, %.0fs", session, key,
                 spx[key]["spx_iv_rmse_exact"], spx[key]["spx_oos_iv_rmse"],
                 spx[key]["fit_seconds"])
    out["spx_only"] = spx

    # 6. the joint loss (SPX anchors + VIX spot indices) with rough Heston;
    #    its Heston counterparts are "trf" (production box) and the last
    #    sigma_profile row (sigma <= 10, kappa <= 60)
    jr = cal.calibrate_comparison("rough_heston", legs="joint", warm=stored["params"],
                                  n_starts=ROUGH_SOBOL_STARTS)
    out["joint_rough_heston"] = jr
    log.info("%s joint rough Heston: loss %.4g, in %.3f vp, OOS %.3f vp, VIX %.3f, H %.3f, %.0fs",
             session, jr["loss"], jr["spx_iv_rmse_exact"], jr["spx_oos_iv_rmse"],
             jr["vix_index_rmse"], jr["params"]["H"], jr["fit_seconds"])
    out["seconds_total"] = time.perf_counter() - t_all
    out["environment"]["loadavg_end"] = list(os.getloadavg())
    return _jsonable(out)


def summarise(paths) -> dict:
    rows = []
    for p in sorted(paths):
        d = json.loads(Path(p).read_text())
        prof = {r["sigma_cap"]: r for r in d["sigma_profile"]["rows"]}
        free = d["sigma_profile"]["rows"][-1]
        h, rgh = d["spx_only"]["heston"], d["spx_only"]["rough_heston"]
        hw = d["spx_only"].get("heston_wide_box") or {}
        rows.append({
            "session": d["session"],
            "stored_loss": d["stored_fit"]["loss"], "trf_loss": d["trf"]["loss"],
            "trf_loss_rel_diff": d["trf"]["loss_rel_diff_vs_stored"],
            "trf_seconds": d["trf"]["fit_time_seconds"], "trf_evals": d["trf"]["n_evals"],
            "trf_cpu_seconds": d["trf"].get("fit_cpu_seconds"),
            "trf_fd_seconds": d["trf_finite_diff"]["fit_time_seconds"],
            "trf_fd_cpu_seconds": d["trf_finite_diff"].get("fit_cpu_seconds"),
            "trf_fd_evals": d["trf_finite_diff"]["n_evals"],
            "de_serial_seconds": (d["de"].get("serial_rerun") or {}).get("fit_time_seconds"),
            "de_serial_cpu_seconds": (d["de"].get("serial_rerun") or {}).get("fit_cpu_seconds"),
            "de_serial_evals": (d["de"].get("serial_rerun") or {}).get("n_evals"),
            "de_serial_loss_rel_diff_vs_stored": (d["de"].get("serial_rerun") or {}).get(
                "loss_rel_diff_vs_stored"),
            "loadavg_start": d["environment"]["loadavg_start"],
            "trf_basins": d["trf"]["multistart"]["n_distinct_basins"],
            "stored_de_seconds": d["stored_fit"]["fit_time_seconds"],
            "stored_de_evals": d["stored_fit"]["n_evals"],
            "de4_seconds": d["de"]["de_seconds"], "de_agrees": d["de"]["agrees"],
            "jacobian_speedup_vs_2point": d["jacobian"]["speedup_vs_2point"],
            "rel_jtj_condition_number": d["identifiability"]["rel_jtj_condition_number"],
            "relative_standard_errors": d["identifiability"]["relative_standard_errors"],
            "on_bound": d["identifiability"]["on_bound"],
            "sigma_cap2_oos": prof[2.0]["spx_oos_iv_rmse"],
            "sigma_free": free["params"]["sigma"], "kappa_free": free["params"]["kappa"],
            "sigma_free_loss_rel": free["loss"] / prof[2.0]["loss"] - 1.0,
            "sigma_free_oos": free["spx_oos_iv_rmse"],
            "vvix_market": d["data"]["vvix_market"],
            "vvix_model_cap2": prof[2.0]["vvix_model"], "vvix_model_free": free["vvix_model"],
            "heston_spx_in": h["spx_iv_rmse_exact"], "heston_spx_oos": h["spx_oos_iv_rmse"],
            "heston_spx_wing": h["spx_oos_put_wing_rmse"],
            "heston_spx_vix_index_oos": h["vix_index_rmse_oos"],
            "heston_spx_sigma": h["params"]["sigma"],
            "heston_wide_spx_in": hw.get("spx_iv_rmse_exact"),
            "heston_wide_spx_oos": hw.get("spx_oos_iv_rmse"),
            "heston_wide_spx_sigma": (hw.get("params") or {}).get("sigma"),
            "heston_wide_spx_kappa": (hw.get("params") or {}).get("kappa"),
            "rough_in": rgh["spx_iv_rmse_exact"], "rough_oos": rgh["spx_oos_iv_rmse"],
            "rough_wing": rgh["spx_oos_put_wing_rmse"],
            "rough_vix_index_oos": rgh["vix_index_rmse_oos"],
            "rough_H": rgh["params"]["H"], "rough_sigma": rgh["params"]["sigma"],
            "rough_H_se": rgh["identifiability"]["standard_errors"].get("H"),
            "rough_rel_jtj_condition_number": rgh["identifiability"][
                "rel_jtj_condition_number"],
            "rough_seconds": rgh["fit_seconds"],
            "rough_basins": rgh["multistart"]["n_distinct_basins"],
            "rough_numerics_max_iv_vp": (rgh.get("numerics") or {}).get(
                "anchor_iv_max_abs_diff_vp"),
            "joint_heston_loss": d["trf"]["loss"],
            "joint_heston_free_sigma_loss": free["loss"],
            "joint_heston_vix_index": prof[2.0]["vix_index_rmse"],
            "joint_heston_free_sigma_vix_index": free["vix_index_rmse"],
            "joint_rough_loss": d["joint_rough_heston"]["loss"],
            "joint_rough_in": d["joint_rough_heston"]["spx_iv_rmse_exact"],
            "joint_rough_oos": d["joint_rough_heston"]["spx_oos_iv_rmse"],
            "joint_rough_wing": d["joint_rough_heston"]["spx_oos_put_wing_rmse"],
            "joint_rough_vix_index": d["joint_rough_heston"]["vix_index_rmse"],
            "joint_rough_H": d["joint_rough_heston"]["params"]["H"],
            "joint_rough_sigma": d["joint_rough_heston"]["params"]["sigma"],
            "joint_rough_on_bound": d["joint_rough_heston"]["on_bound"],
        })
    return {"schema_version": SCHEMA_VERSION,
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "n_sessions": len(rows), "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", action="append", help="session(s) to run (default: all v2)")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--time-serial-de", action="store_true",
                    help="also re-run the serial DE on this machine (about 45 s per session)")
    ap.add_argument("--de-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ("joint_vol_calibration",):
        logging.getLogger(name).setLevel(logging.WARNING)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        sessions = args.session or committed_v2_sessions()
        for s in sessions:
            d = diagnose_session(s, time_serial_de=args.time_serial_de,
                                 de_workers=args.de_workers)
            path = args.out_dir / f"{s}.json"
            path.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
            log.info("wrote %s (%.0fs)", path, d["seconds_total"])
    paths = sorted(p for p in args.out_dir.glob("20*.json"))
    if paths:
        (args.out_dir / "summary.json").write_text(
            json.dumps(_jsonable(summarise(paths)), indent=1, allow_nan=False) + "\n")
        log.info("wrote summary of %d session(s)", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
