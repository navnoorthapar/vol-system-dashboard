"""
Surface report artifacts (surface/report.py, scripts/surface_report.py): the
schema, byte-for-byte determinism, and agreement of the committed
data_store/surface artifacts with a recomputation.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.surface import report

REPO = Path(__file__).resolve().parents[2]
SESSION = "2026-10-16"
COMMITTED = REPO / "data_store" / "surface"
FILES = ("residuals.parquet", "surface.json", "scorecard.json", "vix_replication.json")
RESID_COLS = ["session", "model", "expiry", "T_days", "strike", "right", "k", "forward", "bid", "ask",
              "bid_iv", "mid_iv", "ask_iv", "model_iv", "vega", "half_spread_vp", "error_vp",
              "error_hs", "inside_spread", "outside_spread_vp", "in_sample", "delta_bucket",
              "tenor_bucket", "arb_flag", "arb_exec", "arb_mid", "repair_dprice"]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("surface")
    row = report.write_session(SESSION, out)
    report.write_index([row], out)
    return out, row


def test_artifacts_and_schema(written):
    out, row = written
    d = out / SESSION
    for f in FILES:
        assert (d / f).exists(), f
    res = pd.read_parquet(d / "residuals.parquet")
    assert list(res.columns) == RESID_COLS
    assert set(res["model"]) == {"essvi", "svi", "svi_cal", "heston_v2"}
    assert len(res) == 4 * row["n_quotes"]
    surf = json.loads((d / "surface.json").read_text())
    assert surf["schema_version"] == report.SCHEMA_VERSION
    assert surf["session"] == SESSION
    assert surf["inputs"]["chain"] == f"data_store/options/spx_{SESSION}.parquet"
    assert len(surf["inputs"]["chain_sha256"]) == 64
    assert set(surf["surfaces"]) == {"essvi", "svi", "svi_cal"}
    ess = surf["surfaces"]["essvi"]
    assert ess["analytic_conditions"]["butterfly_ok"]
    assert ess["analytic_conditions"]["calendar_ok_on_check_grid"]
    assert ess["dense_grid"]["butterfly_violations"] == 0
    assert ess["dense_grid"]["calendar_violations"] == 0
    assert surf["surfaces"]["svi"]["dense_grid"]["butterfly_violations"] == 0
    cal = surf["surfaces"]["svi_cal"]
    assert cal["dense_grid"]["butterfly_violations"] == 0
    assert cal["dense_grid"]["calendar_violations"] == 0
    assert cal["calendar_ordering"]["ordered"]
    arbd = surf["arbitrage"]
    cert = arbd["executable_arbitrage_certificate"]
    assert cert["executable_arbitrage"]
    assert cert["bid_ask_box_feasible"] is False
    rq = arbd["quotes_repaired_outside_spread"]
    assert rq["n"] == arbd["repair"]["n_perturbed_outside_spread"] == sum(rq["by_expiry"].values())
    moves = [abs(r["repair_dprice"]) for r in rq["largest"]]
    assert moves == sorted(moves, reverse=True)
    card = json.loads((d / "scorecard.json").read_text())
    names = [m["model"] for m in card["models"]]
    assert names == ["essvi", "svi", "svi_cal", "heston_v2"]
    vixd = json.loads((d / "vix_replication.json").read_text())
    assert set(vixd["indices"]) == {"VIX9D", "VIX", "VIX3M", "VIX6M"}
    assert vixd["indices"]["VIX"]["headline_method"] == "cboe_rules"
    assert any("2026-11-20" in n for n in vixd["gap_notes"])
    ga = vixd["indices"]["VIX"]["gap_attribution"]
    assert ga["available"]
    assert ga["surface"] == "svi_cal"
    assert ga["residual_pts"] + ga["gap_pts"] + ga["truncation_pts"] == pytest.approx(0.0, abs=1e-8)
    idx = json.loads((out / "index.json").read_text())
    assert [r["session"] for r in idx["sessions"]] == [SESSION]


def test_spread_unit_claims_hold(written):
    """The orderings the README / dashboard will quote, all on the same quotes:
    the arbitrage-free SVI-cal surface prices most out-of-sample quotes inside
    the spread, between free per-slice SVI (calendar violations) and eSSVI;
    Heston misses by several half-spreads (medians)."""
    out, _ = written
    card = json.loads((out / SESSION / "scorecard.json").read_text())
    rows = {r["model"]: r for r in card["models"]}
    m = {k: r["metrics"]["out_of_sample"] for k, r in rows.items()}
    assert (m["svi"]["inside_spread_pct"] >= m["svi_cal"]["inside_spread_pct"]
            > m["essvi"]["inside_spread_pct"] > m["heston_v2"]["inside_spread_pct"])
    assert m["svi_cal"]["inside_spread_pct"] > 50.0
    assert rows["svi_cal"]["dense_grid"]["calendar_violations"] == 0
    assert rows["svi"]["dense_grid"]["calendar_violations"] > 0
    assert m["heston_v2"]["median_abs_hs"] > 3.0
    assert m["svi"]["median_abs_hs"] < 1.0
    assert m["svi_cal"]["median_abs_hs"] < 1.0


def test_rerun_is_byte_identical(written, tmp_path):
    out, _ = written
    report.write_session(SESSION, tmp_path)
    for f in FILES:
        assert (tmp_path / SESSION / f).read_bytes() == (out / SESSION / f).read_bytes(), f


def _models(path):
    return {m["model"]: m for m in json.loads(path.read_text())["models"]}


def test_committed_artifacts_match_a_recomputation(written):
    """data_store/surface/<session>/ is what this code writes. Closed-form
    numbers (Heston, VIX) agree to 1e-9. SLSQP fits can take a different path
    after a last-bit difference between CPUs (AVX-512 vs AVX2), so fitted
    models get 1e-4 relative on RMSEs and medians and, on the inside-spread
    share, a margin of 3 quotes."""
    out, _ = written
    if not (COMMITTED / SESSION / "scorecard.json").exists():
        pytest.skip("no committed surface artifacts")
    new, old = _models(out / SESSION / "scorecard.json"), _models(COMMITTED / SESSION / "scorecard.json")
    assert set(new) == set(old)
    for name in new:
        exact = name == "heston_v2"
        for split in ("all", "in_sample", "out_of_sample"):
            n = new[name]["metrics"][split]["n"]
            assert n == old[name]["metrics"][split]["n"], (name, split)
            for key in ("rmse_vp", "rmse_hs", "median_abs_hs", "inside_spread_pct"):
                a, b = new[name]["metrics"][split][key], old[name]["metrics"][split][key]
                if exact:
                    assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (name, split, key)
                elif key == "inside_spread_pct":
                    assert abs(a - b) <= 100.0 * 3 / n + 1e-9, (name, split, key)
                else:
                    assert a == pytest.approx(b, rel=1e-4), (name, split, key)
    vn = json.loads((out / SESSION / "vix_replication.json").read_text())
    vo = json.loads((COMMITTED / SESSION / "vix_replication.json").read_text())
    for name in ("VIX9D", "VIX", "VIX3M", "VIX6M"):
        a, b = vn["indices"][name].get("replicated"), vo["indices"][name].get("replicated")
        assert (a is None and b is None) or a == pytest.approx(b, rel=1e-9)
    rn = pd.read_parquet(out / SESSION / "residuals.parquet")
    ro = pd.read_parquet(COMMITTED / SESSION / "residuals.parquet")
    assert list(rn.columns) == list(ro.columns)
    assert len(rn) == len(ro)
    h = rn["model"] == "heston_v2"
    np.testing.assert_allclose(rn.loc[h, "model_iv"], ro.loc[h, "model_iv"], rtol=1e-9)


def test_script_writes_the_index(tmp_path):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "surface_report.py"),
                        "--sessions", SESSION, "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=REPO, timeout=600, check=False)
    assert r.returncode == 0, r.stderr
    assert SESSION in r.stdout
    idx = json.loads((tmp_path / "index.json").read_text())
    assert idx["sessions"][0]["session"] == SESSION


def test_round_sig_is_deterministic_and_bounded():
    x = np.array([0.0, 1.23456789012345, -9.87654321098e-7, np.nan, np.inf, 123456789.123])
    r = report.round_sig(x)
    assert r[0] == 0.0
    assert np.isnan(r[3])
    assert np.isinf(r[4])
    np.testing.assert_allclose(r[[1, 2, 5]], x[[1, 2, 5]], rtol=1e-9)
    np.testing.assert_array_equal(report.round_sig(r), r)
