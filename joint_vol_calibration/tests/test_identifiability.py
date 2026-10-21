"""
test_identifiability.py — what the identifiability report claims, checked
against independent evidence, plus the committed diagnostics artifacts (C23).

1. Gauss-Newton standard errors are a curvature scale: holding a
   parameter one standard error away and re-fitting the others raises the
   sum of squares by one residual variance s^2 exactly on a linear problem,
   and on 2026-10-16 by 0.67-1.04 s^2 (gauss_newton_profile_check, which
   the diagnostics report per parameter).
2. Under iid implied-vol noise (seeded) on the synthetic Heston market, the
   GN standard errors match the Monte Carlo spread of refitted parameters.
3. data_store/calibrations/diagnostics/<session>.json (written by
   scripts/calibration_diagnostics.py) is internally consistent, covers every
   committed model_version 2 session, and its numbers are reproduced by the
   code at spot checks.
"""

import itertools
import json
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration import config
from joint_vol_calibration.calibration import joint_calibrator as jc
from joint_vol_calibration.models.heston import heston_vvix
from joint_vol_calibration.tests import test_joint_calibrator as tjc

_NAMES = jc.HESTON_PARAM_NAMES
_DIAG = config.DATA_DIR / "calibrations" / "diagnostics"
_V2 = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")


# ── 1. profile check of the standard errors (real session) ───────────────────

def test_gauss_newton_profile_check_on_2026_10_16():
    """Holding a parameter one SE off and re-fitting the rest raises the sum
    of squares by ~s^2 when the Gauss-Newton picture holds. Measured on
    2026-10-16: rho and v0 0.95-1.04; kappa and theta 0.67-1.02 (the loss is
    flatter than J^T J says there, so those SEs are optimistic by up to
    ~20%). The bounds pin that picture without hiding it."""
    cal = jc.JointCalibrator(as_of_date="2026-10-16")
    res = cal.calibrate(verbose=False)
    idf = res["identifiability"]
    x = np.array([res["params"][k] for k in _NAMES])
    lo, hi = jc._box_arrays(config.HESTON_BOUNDS, _NAMES)
    chk = jc.gauss_newton_profile_check(cal.residual_model(), x, idf, lo, hi)
    assert set(chk) == set(idf["free_params"]) == {"kappa", "theta", "rho", "v0"}
    for name, ratios in chk.items():
        for m, ratio in ratios.items():
            assert 0.5 < ratio < 1.6, (name, m, ratio)
    for name in ("rho", "v0"):                    # the well-described directions
        assert all(0.85 < r < 1.15 for r in chk[name].values()), chk[name]
    assert min(chk["kappa"].values()) < 0.9       # and the flat one is not


def test_profile_check_is_exactly_one_on_a_linear_problem():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(50, 3))
    b = A @ np.array([1.0, -2.0, 0.5]) + 0.05 * rng.normal(size=50)
    x_hat, *_ = np.linalg.lstsq(A, b, rcond=None)

    def fun(x):
        return A @ x - b
    lo, hi = np.full(3, -10.0), np.full(3, 10.0)
    rep = jc.identifiability_report(A, fun(x_hat), x_hat, ("a", "b", "c"), lo, hi)
    chk = jc.gauss_newton_profile_check(fun, x_hat, rep, lo, hi, steps=(-2.0, 1.0))
    for ratios in chk.values():
        for r in ratios.values():
            assert r == pytest.approx(1.0, rel=1e-6)


# ── 2. Monte Carlo check of the standard errors (synthetic market) ───────────

def _noisy_fit(seed, iv_noise):
    """Synthetic Heston chain with iid N(0, iv_noise^2) implied-vol noise
    added to every quote (price shift = noise * Black vega), refitted."""
    from joint_vol_calibration.data import option_chain as oc
    rng = np.random.default_rng(seed)
    chain = tjc._heston_chain()
    T = np.array([oc.year_fraction(tjc._SESSION, e, "PM") for e in chain["expiration"]])
    F = tjc._SPOT * np.exp((tjc._R_TRUE - tjc._Q_TRUE) * T)
    D = np.exp(-tjc._R_TRUE * T)
    mid = 0.5 * (chain["bid"] + chain["ask"]).to_numpy()
    is_call = (chain["option_type"] == "call").to_numpy()
    iv = oc.black76_implied_vol(mid, F, chain["strike"].to_numpy(), T, D, is_call)
    vega = oc.black76_vega(F, chain["strike"].to_numpy(), T, D, np.where(np.isfinite(iv), iv, 0.2))
    shift = rng.normal(0.0, iv_noise, len(chain)) * vega
    chain = chain.assign(bid=chain["bid"] + shift, ask=chain["ask"] + shift)
    with pytest.MonkeyPatch.context() as mp:
        tjc._patch_db(mp)
        cal = jc.JointCalibrator(as_of_date=tjc._SESSION, spx_chain=chain)
        return cal.calibrate(n_starts=0, warm_start=dict(tjc._TRUE), verbose=False)


def test_standard_errors_match_the_monte_carlo_spread_of_refits():
    noise = 0.002                                     # 0.2 vol points per quote
    fits = [_noisy_fit(seed, noise) for seed in range(16)]
    est = pd.DataFrame([f["params"] for f in fits])
    mc_sd = est.std(ddof=1)
    se = pd.DataFrame([f["identifiability"]["standard_errors"] for f in fits]).median()
    for k in ("theta", "rho", "v0", "sigma"):
        ratio = se[k] / mc_sd[k]
        # 16 replicates: the sample sd itself is uncertain by ~18%
        assert 0.55 < ratio < 1.8, (k, ratio, se[k], mc_sd[k])


# ── 3. committed diagnostics artifacts ───────────────────────────────────────

def _load(session):
    path = _DIAG / f"{session}.json"
    if not path.exists():
        pytest.skip(f"{path} not generated (scripts/calibration_diagnostics.py)")
    return json.loads(path.read_text())


@pytest.mark.parametrize("session", _V2)
def test_diagnostics_artifact_is_consistent(session):
    d = _load(session)
    assert d["schema_version"] == 1
    assert d["session"] == session
    # both parts ran, from committed code (no uncommitted non-test changes)
    assert set(d["provenance"]) == {"heston", "rough"}
    for prov in d["provenance"].values():
        assert prov["code_commit"], prov
        assert prov["code_dirty_files"] == [], prov
    st = d["stored_fit"]
    assert st["loss_recomputed_from_residuals"] == pytest.approx(st["loss"], rel=1e-12)
    trf = d["trf"]
    assert trf["loss"] <= st["loss"] * (1 + 1e-7)
    assert trf["multistart"]["n_distinct_basins"] == 1
    assert trf["gate"] == [True, "ok"]
    assert d["trf_finite_diff"]["loss_rel_diff_vs_analytic"] == pytest.approx(0, abs=1e-6)
    jac = d["jacobian"]
    assert max(jac["analytic_vs_central_fd_max_rel_diff_by_param"].values()) < 1e-5
    assert jac["speedup_vs_2point"] > 1.0
    de = d["de"]
    assert de["used"] == "trf"
    assert de["de_polished_loss"] >= trf["loss"] * (1 - 1e-6)

    idf = d["identifiability"]
    sv = idf["rel_jacobian_singular_values"]
    assert sv == sorted(sv, reverse=True)
    assert idf["rel_jtj_condition_number"] == pytest.approx((sv[0] / sv[-1]) ** 2)
    assert idf["on_bound"] == trf["pinned_params"]
    for k in idf["on_bound"]:
        assert idf["standard_errors"][k] is None
    assert set(idf["profile_check"]) == set(idf["free_params"])
    for ratios in idf["profile_check"].values():
        assert all(0.4 < r < 2.0 for r in ratios.values())

    prof = d["sigma_profile"]
    assert prof["caps"] == list(config.SIGMA_PROFILE_CAPS)
    assert prof["kappa_max"] == config.PROFILE_KAPPA_MAX
    rows = prof["rows"]
    losses = [r["loss"] for r in rows]
    assert all(b <= a * (1 + 1e-9) for a, b in itertools.pairwise(losses))
    cap2 = next(r for r in rows if r["sigma_cap"] == 2.0)
    # the cap-2 refit is the production box whenever kappa stays below 20
    if cap2["params"]["kappa"] < config.HESTON_BOUNDS["kappa"][1]:
        assert cap2["loss"] == pytest.approx(trf["loss"], rel=1e-6)
    for r in rows:
        assert r["sigma_at_cap"] == (abs(r["params"]["sigma"] - r["sigma_cap"])
                                     <= 1e-3 * r["sigma_cap"])
        p = r["params"]
        assert r["vvix_model"] == pytest.approx(heston_vvix(p["kappa"], p["theta"],
                                                            p["sigma"], p["v0"]), rel=1e-9)

    h, hw, rgh = (d["spx_only"][k] for k in ("heston", "heston_wide_box", "rough_heston"))
    jr = d["joint_rough_heston"]
    for m in (h, hw, rgh, jr):
        assert np.isfinite(m["spx_iv_rmse_exact"])
        assert np.isfinite(m["spx_oos_iv_rmse"])
        assert m["n_spx_oos"] == d["data"]["n_spx_oos"]
    assert h["legs"] == hw["legs"] == rgh["legs"] == "spx"
    assert jr["legs"] == "joint"
    # the wide box nests the production box: never a worse SPX-only optimum
    assert hw["loss"] <= h["loss"] * (1 + 1e-6)
    lo, hi = config.ROUGH_HESTON_H_BOUNDS
    for m in (rgh, jr):
        assert lo <= m["params"]["H"] <= hi
        num = m["numerics"]
        # the lifting / time-step error is far below the misfit it is compared with
        assert num["anchor_iv_max_abs_diff_vp"] < 0.05
        assert num["lifting_error_estimate_max_vp"] == pytest.approx(
            num["anchor_iv_max_abs_diff_vp"] / (1 - 2 ** -1.5))
        assert num["frozen_vs_adaptive_rel_loss_diff"] < 1e-5
        assert ("H" in m["on_bound"]) == (m["params"]["H"] <= lo * (1 + 1e-3)
                                          or m["params"]["H"] >= hi * (1 - 1e-3))
    # rough Heston nests Heston (H = 1/2): its joint loss is not above the
    # production Heston fit's, up to the lifting error
    assert jr["loss"] <= trf["loss"] * 1.01


def test_diagnostics_cover_every_committed_v2_session_and_summary_matches():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "calibration_diagnostics", config.ROOT_DIR / "scripts" / "calibration_diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The five C23 sessions are reviewed here (each check above); the daily bot
    # adds a session's diagnostics after each accepted fit, so the committed set
    # grows and is checked structurally (test_session_artifacts.py).
    sessions = mod.committed_v2_sessions()
    assert set(_V2) <= set(sessions)
    summary_path = _DIAG / "summary.json"
    if not summary_path.exists():
        pytest.skip("summary.json not generated")
    summary = json.loads(summary_path.read_text())
    rows = [r["session"] for r in summary["rows"]]
    assert set(_V2) <= set(rows) <= set(sessions)
    assert rows == sorted(rows)
    for row in summary["rows"]:
        d = _load(row["session"])
        assert row["trf_loss"] == d["trf"]["loss"]
        assert row["rough_oos"] == d["spx_only"]["rough_heston"]["spx_oos_iv_rmse"]
        assert row["heston_spx_oos"] == d["spx_only"]["heston"]["spx_oos_iv_rmse"]
        assert row["joint_rough_oos"] == d["joint_rough_heston"]["spx_oos_iv_rmse"]


def test_artifact_numbers_are_reproduced_by_the_code():
    """Spot check on 2026-10-16: the sigma-free profile row's out-of-sample
    RMSE and the rough Heston fit's in-sample RMSE are recomputed live from
    the parameters stored in the artifact."""
    from joint_vol_calibration.models import rough_heston as rh
    d = _load("2026-10-16")
    # the diagnostics run on the committed fit's recorded VIX targets
    # (calibration_diagnostics.py; joint_calibrator.recorded_vix_targets)
    with open(config.DATA_DIR / "calibrations" / "joint_cal_2026-10-16.pkl", "rb") as f:
        stored = pickle.load(f)
    cal = jc.JointCalibrator(as_of_date="2026-10-16", **jc.recorded_inputs(stored))
    free = d["sigma_profile"]["rows"][-1]
    live = cal.fit_summary([free["params"][k] for k in _NAMES])
    assert live["spx_oos_iv_rmse"] == pytest.approx(free["spx_oos_iv_rmse"], rel=1e-9)
    assert live["loss"] == pytest.approx(free["loss"], rel=1e-9)
    rgh = d["spx_only"]["rough_heston"]
    x = tuple(rgh["params"][k] for k in rh.ROUGH_HESTON_PARAM_NAMES)
    err = cal._surface_errors(cal.spx_surface, rh.rough_heston_call_batch, x)["error"].dropna()
    assert math.sqrt(float(np.mean(err ** 2))) * 100 == pytest.approx(
        rgh["spx_iv_rmse_exact"], rel=1e-9)


def test_sigma_box_rationale_in_config_matches_the_artifacts():
    """config.HESTON_BOUNDS documents why sigma <= 2 is kept with numbers
    measured here; this pins those numbers to the committed diagnostics so
    the comment cannot drift from the data."""
    d = {s: _load(s) for s in _V2}
    binds = [s for s in _V2 if "sigma" in d[s]["trf"]["pinned_params"]]
    assert binds == ["2026-03-24", "2026-10-16"]
    interior = [round(d[s]["trf"]["params"]["sigma"], 2) for s in _V2 if s not in binds]
    assert interior == [1.83, 1.71, 1.59]

    def prof(s):
        rows = d[s]["sigma_profile"]["rows"]
        return next(r for r in rows if r["sigma_cap"] == 2.0), rows[-1]

    cap2, free = prof("2026-03-24")
    assert free["loss"] / cap2["loss"] - 1 == pytest.approx(-0.02, abs=0.005)
    assert round(free["params"]["sigma"], 2) == 2.21
    assert (round(cap2["spx_oos_iv_rmse"], 2), round(free["spx_oos_iv_rmse"], 2)) == (1.98, 1.82)
    cap2, free = prof("2026-10-16")
    assert free["loss"] / cap2["loss"] - 1 == pytest.approx(-0.28, abs=0.005)
    assert (round(free["params"]["sigma"], 1), round(free["params"]["kappa"])) == (6.9, 27)
    assert (round(cap2["spx_oos_iv_rmse"], 2), round(free["spx_oos_iv_rmse"], 2)) == (2.96, 1.19)
    assert (round(cap2["vvix_model"]), round(free["vvix_model"]),
            round(free["vvix_market"])) == (205, 113, 110)
    assert round(free["feller_margin"]) == -45
    # rough Heston: sigma interior everywhere, lower joint loss on 10-16,
    # and H = 1/2 (Heston itself) on two sessions
    jr = {s: d[s]["joint_rough_heston"] for s in _V2}
    sig = [jr[s]["params"]["sigma"] for s in _V2]
    assert "sigma" not in [b for s in _V2 for b in jr[s]["on_bound"]]
    assert (round(min(sig), 2), round(max(sig), 2)) == (0.88, 1.71)
    assert (round(jr["2026-10-16"]["loss"], 6), round(free["loss"], 6),
            round(d["2026-10-16"]["trf"]["loss"], 6)) == (7.3e-5, 8.5e-5, 1.18e-4)
    at_half = [s for s in _V2 if "H" in jr[s]["on_bound"] and jr[s]["params"]["H"] > 0.4]
    assert at_half == ["2026-05-29", "2026-06-18"]
