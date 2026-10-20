"""
Quote-grade metrics and dense-grid arbitrage checks (surface/metrics.py).
"""
import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.surface import metrics as mt
from joint_vol_calibration.surface import svi as sv


def _quotes():
    # four quotes: half-spreads 0.1, 0.2, 0.1, 0.5 vol points
    return pd.DataFrame({
        "session": "2026-01-02", "expiry": ["E1", "E1", "E2", "E2"], "T": [0.1, 0.1, 0.5, 0.5],
        "T_days": [36.5, 36.5, 182.5, 182.5], "strike": [90.0, 110.0, 90.0, 110.0],
        "right": ["P", "C", "P", "C"], "k": [-0.1, 0.1, -0.1, 0.1], "forward": 100.0,
        "bid": 1.0, "ask": 1.1,
        "bid_iv": [0.199, 0.148, 0.209, 0.155], "mid_iv": [0.200, 0.150, 0.210, 0.160],
        "ask_iv": [0.201, 0.152, 0.211, 0.165], "half_spread_vp": [0.1, 0.2, 0.1, 0.5],
        "vega": 1.0, "delta_bucket": ["25P", "25C", "25P", "25C"],
        "tenor_bucket": ["30-90d", "30-90d", "180-365d", "180-365d"]})


def test_spread_unit_metrics_by_hand():
    q = _quotes()
    model = np.array([0.2005, 0.153, 0.2115, 0.160])       # +0.05, +0.3, +0.15, 0.0 vp
    res = mt.residual_frame(q, model, model="m", in_sample=[True, True, False, False])
    np.testing.assert_allclose(res["error_vp"], [0.05, 0.3, 0.15, 0.0], atol=1e-12)
    np.testing.assert_allclose(res["error_hs"], [0.5, 1.5, 1.5, 0.0], atol=1e-10)
    assert res["inside_spread"].tolist() == [True, False, False, True]
    np.testing.assert_allclose(res["outside_spread_vp"], [0.0, 0.1, 0.05, 0.0], atol=1e-12)
    s = mt.summarise(res)
    assert s["all"]["n"] == 4 and s["all"]["inside_spread_pct"] == pytest.approx(50.0)
    assert s["all"]["rmse_vp"] == pytest.approx(np.sqrt((0.05 ** 2 + 0.3 ** 2 + 0.15 ** 2) / 4))
    assert s["all"]["median_abs_hs"] == pytest.approx(1.0)
    assert s["in_sample"]["n"] == 2 and s["out_of_sample"]["inside_spread_pct"] == pytest.approx(50.0)
    assert s["by_expiry"]["E1"]["rmse_hs"] == pytest.approx(np.sqrt((0.25 + 2.25) / 2))
    assert s["by_tenor_delta"]["180-365d|25P"]["n"] == 1
    assert s["all"]["within_2hs_pct"] == pytest.approx(100.0)


def test_zero_width_spread_is_floored_not_infinite():
    q = _quotes()
    q.loc[0, ["bid_iv", "ask_iv", "half_spread_vp"]] = [0.2, 0.2, 0.0]
    res = mt.residual_frame(q, q["mid_iv"].to_numpy() + 0.0001, model="m")
    assert res.loc[0, "error_hs"] == pytest.approx(0.01 / mt.H_FLOOR_VP)


RANGES = [("a", 0.1, -0.4, 0.2), ("b", 0.25, -0.5, 0.3), ("c", 0.5, -0.6, 0.35)]


def test_flat_vol_has_no_dense_grid_violation():
    out = mt.dense_grid_checks(lambda k, T: np.full(len(k), 0.2), RANGES)
    assert out["butterfly_violations"] == 0 and out["calendar_violations"] == 0
    assert out["call_spread_slope_violations"] == 0
    assert out["butterfly_points"] == 3 * (mt.GRID_POINTS - 2)


def test_butterfly_arbitrage_is_detected_on_prices():
    vogt = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)
    iv = lambda k, T: np.sqrt(np.maximum(sv.svi_w(k, *vogt), 1e-12) / 1.0)   # noqa: E731
    out = mt.dense_grid_checks(iv, [("v", 1.0, -0.2, 0.9)])
    assert out["butterfly_violations"] > 0 and out["butterfly_violating_expiries"] == ["v"]


def test_calendar_arbitrage_is_detected():
    # total variance falls from T = 0.25 to T = 0.5
    iv = lambda k, T: np.full(len(k), 0.3 if T < 0.3 else 0.2)              # noqa: E731
    out = mt.dense_grid_checks(iv, RANGES)
    assert out["calendar_violations"] > 0 and out["calendar_violating_pairs"] == ["b->c"]
