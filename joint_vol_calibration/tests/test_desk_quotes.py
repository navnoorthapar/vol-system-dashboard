"""
dashboard/desk_quotes.py: the quotes behind a smile chart, packed for the quote
inspector (P4.1).

The packed block is what desk.js reads to say, for the quote under the pointer,
its strike, right, forward delta, bid / mid / ask and each model's IV and z.
These tests decode it on every stored surface session and check each value
against the producer's own residuals.parquet columns, so the inspector cannot
drift from the artifact on the bot's next session either.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
SURFACE = ROOT / "data_store" / "surface"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_quotes as dq  # noqa: E402

from joint_vol_calibration.surface import metrics  # noqa: E402


def _sessions() -> list[str]:
    return sorted(p.name for p in SURFACE.iterdir()
                  if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
                  and (p / "residuals.parquet").exists())


def _producer_delta(k: float, mid_iv: float, t_days: float, right: str) -> float:
    """The producer's forward Black delta (quotes.black76_delta) on its own columns."""
    w = mid_iv ** 2 * t_days / 365.0
    d1 = -k / math.sqrt(w) + math.sqrt(w) / 2.0
    return float(norm.cdf(d1)) - (1.0 if right == "P" else 0.0)


def test_the_half_spread_floor_is_the_producers():
    assert dq.H_FLOOR_VP == metrics.H_FLOOR_VP


@pytest.mark.parametrize("session", _sessions())
def test_decoded_quotes_reproduce_the_residual_table(session):
    """Every quote of every expiry and model: strike exact, right, forward delta
    within 1e-6 of the producer's formula on the parquet's own values, error_hs
    within 0.05 (the floor included: a locked quote divides by 0.01 vol pts),
    model IV within 10⁻⁴ vol pts, and the producer's in_sample / inside_spread."""
    res = pd.read_parquet(SURFACE / session / "residuals.parquet")
    models = list(res["model"].unique())
    packed = dq.encode(res, models)
    assert sorted(packed) == sorted(res["expiry"].unique())
    worst = {"delta": 0.0, "z": 0.0, "iv": 0.0}
    for e, g in packed.items():
        rows = dq.decode(g)
        for m in models:
            q = res[(res["model"] == m) & (res["expiry"] == e)].sort_values("k", kind="mergesort")
            assert len(q) == len(rows), (e, m)
            for row, p in zip(rows, q.itertuples()):
                assert row["strike"] == p.strike, (e, m, p.strike)
                assert row["right"] == p.right, (e, m, p.strike)
                want = _producer_delta(p.k, p.mid_iv, p.T_days, p.right)
                worst["delta"] = max(worst["delta"], abs(row["delta"] - want))
                mm = row["models"][m]
                assert mm["fit"] == bool(p.in_sample), (e, m, p.strike)
                if not np.isfinite(p.model_iv):
                    assert mm["iv"] is None
                    assert mm["z"] is None
                    continue
                assert mm["inside"] == bool(p.inside_spread), (e, m, p.strike)
                worst["z"] = max(worst["z"], abs(mm["z"] - p.error_hs))
                worst["iv"] = max(worst["iv"], abs(mm["iv"] - p.model_iv * 100.0))
            # the market columns: bid / ask IVs and the half-spread
            for row, p in zip(rows, q.itertuples()):
                assert row["bid"] == pytest.approx(p.bid_iv * 100.0, abs=1e-5)
                assert row["ask"] == pytest.approx(p.ask_iv * 100.0, abs=1e-5)
                assert row["half_spread"] == pytest.approx(p.half_spread_vp, abs=1e-5)
    assert worst["delta"] <= 1e-6, worst
    assert worst["z"] <= 0.05, worst
    assert worst["iv"] <= 1e-4, worst


@pytest.mark.parametrize("session", _sessions())
def test_flags_are_the_producers_arbitrage_columns(session):
    """arb_exec, a repair move (the producer's n_perturbed counts the same quotes)
    and arb_flag (moved outside the spread) per quote."""
    import json
    res = pd.read_parquet(SURFACE / session / "residuals.parquet")
    m0 = res["model"].iloc[0]
    packed = dq.encode(res, [m0])
    rows = [r for g in packed.values() for r in dq.decode(g)]
    base = res[res["model"] == m0]
    assert sum(r["exec"] for r in rows) == int(base["arb_exec"].sum())
    assert sum(r["outside"] for r in rows) == int(base["arb_flag"].sum())
    sj = json.loads((SURFACE / session / "surface.json").read_text())
    rep = (sj.get("arbitrage") or {}).get("repair") or {}
    if rep.get("n_perturbed") is not None:
        assert sum(r["repaired"] for r in rows) == rep["n_perturbed"]
        assert sum(r["outside"] for r in rows) == rep["n_perturbed_outside_spread"]


def test_delta_axis_puts_left_calls_right_and_matches_the_delta():
    """N(−d₁): a put's |Δ|, one minus a call's Δ; 0.5 lies just right of the forward."""
    for k, iv, t, right in ((-0.10, 25.0, 30.0, "P"), (0.05, 12.0, 30.0, "C"), (-0.30, 40.0, 13.0, "P")):
        x = dq.delta_coord(k, iv, t)
        d = dq.forward_delta(k, iv, t, right)
        assert x == pytest.approx(-d if right == "P" else 1.0 - d, abs=1e-12)
    assert dq.delta_coord(-0.2, 20.0, 30.0) < dq.delta_coord(0.0, 20.0, 30.0) < 0.5
    assert dq.delta_coord(0.05, 15.0, 30.0) > 0.5


def test_series_data_of_each_field():
    g = {"F": 100.0, "T": 30.0, "step": 5, "K0": 90, "dK": [2, 2], "r": "PCC",
         "mid": [2_500_000, 2_000_000, 1_800_000], "db": [500_000, 400_000, 100_000],
         "da": [500_000, 400_000, 300_000],
         "m": {"a": [10_000, None, -150_000]}, "s": {"a": "301"}, "f": "014"}
    rows = dq.decode(g)
    assert [r["strike"] for r in rows] == [90, 100, 110]
    assert rows[0]["models"]["a"]["fit"]
    assert rows[0]["models"]["a"]["inside"]
    assert rows[1]["models"]["a"]["iv"] is None
    assert rows[1]["models"]["a"]["z"] is None
    assert rows[2]["models"]["a"]["z"] == pytest.approx(-15.0 / 0.2)
    assert [r["exec"] for r in rows] == [False, True, False]
    assert [(r["outside"], r["repaired"]) for r in rows] == [(False, False), (False, False), (True, False)]
    assert dq.series_data(rows, "ba")[0][1:] == [24.5, 25.5]
    assert [p[1] for p in dq.series_data(rows, "model", "a")] == [26.0, None, 3.0]
    assert (len(dq.series_data(rows, "fit", "a")), len(dq.series_data(rows, "oos", "a"))) == (2, 1)
    z = dq.series_data(rows, "z", "a")
    assert [(p[1], round(p[2], 6)) for p in z] == [(round(1.0 / 0.5, 3), 2.0), (-dq.Z_CLIP, -75.0)]
    xd = [p[0] for p in dq.series_data(rows, "mid", x="delta")]
    assert xd == sorted(xd)


def test_a_strike_off_the_index_grid_is_refused():
    q = pd.DataFrame({"k": [0.0], "strike": [100.5], "right": ["C"], "forward": [100.0], "T_days": [30.0],
                      "mid_iv": [0.2], "bid_iv": [0.19], "ask_iv": [0.21]})
    with pytest.raises(ValueError, match="whole index points"):
        dq.encode_group(q, {})
