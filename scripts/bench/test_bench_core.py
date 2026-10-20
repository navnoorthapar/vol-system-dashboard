"""
test_bench_core.py: pytest-benchmark timings of the production numerical paths.

    make bench                                   # full run, JSON in .bench/
    python -m pytest scripts/bench --benchmark-disable -m "not slow"   # CI smoke

Each benchmark first checks that it measures the real computation (the
result is compared with an independent recomputation or a committed artifact)
and only then times it; nothing asserts on time. Inputs are the committed
headline fit (data_store/calibrations/joint_cal_2026-10-16.pkl) and the
2026-10-16 SPX chain it was fitted to.

  heston_slice_*        heston_call_batch, one expiry, 25 strikes K/F 0.70..1.30,
                        at 7, 30 and 365 days (adaptive Lewis quadrature)
  heston_surface_7x15   7 expiries x 15 strikes, the research surface shape
  iv_newton_chain       implied_vol.implied_vol_black on every clean OTM quote
  iv_bisection_chain    option_chain.black76_implied_vol (what the cleaner and
                        the calibrator use) on the same quotes
  joint_loss            one JointCalibrator.joint_loss evaluation (SPX anchors
                        and the VIX index leg) at the headline parameters
  full_calibration      JointCalibrator.calibrate(): DE + L-BFGS-B, seed 42
                        (slow, ~30-50 s; excluded from the CI smoke)
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.models.heston import heston_call_batch
from joint_vol_calibration.models.implied_vol import black_price, implied_vol_black

# the `benchmark` fixture; without the plugin the file skips instead of erroring
pytest.importorskip("pytest_benchmark")

HEADLINE = Path(DATA_DIR) / "calibrations" / "joint_cal_2026-10-16.pkl"
SESSION = "2026-10-16"


@pytest.fixture(scope="module")
def headline() -> dict:
    with open(HEADLINE, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="module")
def chain_quotes(headline):
    raw = db.get_options_surface(as_of_date=SESSION, underlying="SPX")
    clean = oc.clean_option_chain(raw, SESSION, float(headline["S"]), r_ref=float(headline["r_reference"]))
    assert clean.ok, clean.reason
    q = clean.quotes
    return q[q["otm"].astype(bool)].reset_index(drop=True)


def _params(headline) -> tuple:
    p = headline["params"]
    return p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"]


@pytest.mark.parametrize("days", [7, 30, 365])
def test_heston_slice_25_strikes(benchmark, headline, days):
    S, r, q, T = float(headline["S"]), 0.04, 0.013, days / 365.0
    F = S * np.exp((r - q) * T)
    K = F * np.linspace(0.70, 1.30, 25)
    prices = heston_call_batch(S, K, T, r, q, *_params(headline))
    # non-increasing in K to the engine's quadrature target (1e-10 DF F): deep
    # OTM 7-day calls carry ~2e-14 DF F of noise around zero
    assert np.all(np.diff(prices) <= 1e-10 * np.exp(-r * T) * F)
    assert np.all(prices >= 0)
    benchmark.extra_info.update(n_strikes=25, T_days=days)
    benchmark(heston_call_batch, S, K, T, r, q, *_params(headline))


def test_heston_surface_7x15(benchmark, headline):
    S, r, q = float(headline["S"]), 0.04, 0.013
    Ts = np.array([7, 14, 30, 60, 91, 182, 365]) / 365.0

    def surface():
        out = []
        for T in Ts:
            F = S * np.exp((r - q) * T)
            out.append(heston_call_batch(S, F * np.exp(np.linspace(-0.3, 0.1, 15)), T, r, q,
                                         *_params(headline)))
        return out

    assert all(np.all(np.isfinite(s)) for s in surface())
    benchmark.extra_info.update(n_expiries=7, n_strikes=15)
    benchmark(surface)


def _chain_inputs(chain_quotes):
    q = chain_quotes
    return (q["mid_price"].to_numpy(float), q["forward"].to_numpy(float), q["strike"].to_numpy(float),
            q["time_to_expiry"].to_numpy(float), q["discount"].to_numpy(float),
            (q["right"] == "C").to_numpy())


def test_iv_newton_chain(benchmark, chain_quotes):
    price, F, K, T, D, is_call = _chain_inputs(chain_quotes)
    iv = implied_vol_black(price, F, K, T, is_call, D)
    # re-prices every quote (an inversion that returns a wrong vol would not)
    ok = np.isfinite(iv)
    assert ok.mean() > 0.99
    np.testing.assert_allclose(black_price(F[ok], K[ok], T[ok], iv[ok], is_call[ok], D[ok]), price[ok],
                               rtol=1e-9, atol=1e-9)
    benchmark.extra_info.update(n_quotes=len(price))
    benchmark(implied_vol_black, price, F, K, T, is_call, D)


def test_iv_bisection_chain(benchmark, chain_quotes):
    price, F, K, T, D, is_call = _chain_inputs(chain_quotes)
    iv = oc.black76_implied_vol(price, F, K, T, D, is_call)
    np.testing.assert_allclose(iv, chain_quotes["implied_vol"].to_numpy(float), rtol=0, atol=1e-12)
    benchmark.extra_info.update(n_quotes=len(price))
    benchmark(oc.black76_implied_vol, price, F, K, T, D, is_call)


@pytest.fixture(scope="module")
def calibrator(headline):
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator, recorded_vix_targets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # the committed fit's objective: its recorded VIX targets (C23 restated the DB's)
        return JointCalibrator(as_of_date=SESSION, vix_targets=recorded_vix_targets(headline))


def test_joint_loss_evaluation(benchmark, calibrator, headline):
    x = np.array(_params(headline))
    loss = calibrator.joint_loss(x)
    # the committed fit's loss, recomputed (the stored value is this function at x)
    assert loss == pytest.approx(headline["loss"], rel=1e-9)
    benchmark.extra_info.update(n_anchors=len(calibrator.spx_surface),
                                n_vix_tenors=len(calibrator.vix_ts))
    benchmark(calibrator.joint_loss, x)


@pytest.mark.slow
def test_full_calibration(benchmark, calibrator, headline):
    result = benchmark.pedantic(calibrator.calibrate, kwargs={"verbose": False}, rounds=1, iterations=1)
    for k, v in headline["params"].items():
        assert result["params"][k] == pytest.approx(v, rel=1e-6), k
    benchmark.extra_info.update(n_evals=int(result["n_evals"]), fit_time_s=float(result["fit_time"]))
