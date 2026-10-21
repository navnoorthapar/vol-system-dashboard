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
from joint_vol_calibration.surface import svi as sv

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


# ── C23f: the certificate is the label's gate (QSM-1, CLM-02), same-quotes
# scoring (CLM-15), VIX9D extrapolation (QSM-3) and SPXW AM-date proxies (QSM-4)

def _recomputed_certificates(sec: dict) -> dict:
    return sv.SVISurface.from_dict(sec["params"]).certificate()


def test_svi_certificates_recompute_from_the_published_parameters(written):
    """surface.json's certificate fields are those of the parameters it
    publishes (written at full precision), and the scorecard's svi_cal label
    is the gate's verdict."""
    out, _ = written
    surf = json.loads((out / SESSION / "surface.json").read_text())
    for name in ("svi", "svi_cal"):
        sec = surf["surfaces"][name]
        cert = _recomputed_certificates(sec)
        assert report._clean(cert["butterfly"]) == sec["butterfly_certificate"]   # as written
        assert report._clean(cert["calendar"]) == sec["calendar_ordering"]
        assert sec["butterfly_free_certified"] is cert["butterfly"]["certified"]
        assert sec["arbitrage_free_certified"] is cert["arbitrage_free"]
        assert sec["slices_with_g_negative_on_k_-3_to_2"] == 0
        assert sec["butterfly_free_certified"]
    cal = surf["surfaces"]["svi_cal"]
    assert cal["arbitrage_free_certified"]
    assert cal["fit"]["arbitrage_free_certified"]
    card = json.loads((out / SESSION / "scorecard.json").read_text())
    assert card["notes"][2].startswith("svi_cal is the arbitrage-free surface on this session")
    sl = cal["params"]["slices"][0]
    assert len(repr(sl["b"]).replace("0.", "").lstrip("0")) > 11     # not rounded to 10 digits


def test_svi_cal_label_follows_the_certificate():
    bad = {"arbitrage_free_certified": False,
           "butterfly_certificate": {"slices_with_g_negative": 2, "g_min": -2.6e-5},
           "calendar_ordering": {"pairs_unordered": [["a", "b"]], "min_dw": -1e-6}}
    note = report._svi_cal_note(bad)
    assert note.startswith("svi_cal is NOT certified arbitrage-free")
    assert "2 slice(s)" in note
    assert "1 consecutive pair(s)" in note
    assert "is the arbitrage-free surface" not in note


def test_every_model_is_also_scored_on_the_same_quotes(written):
    """CLM-15: each model's out_of_sample set differs (Heston: non-anchors;
    surfaces: the strikes between fitted ones); common_out_of_sample is the
    same quotes for every model, those no model was fitted to."""
    out, _ = written
    card = json.loads((out / SESSION / "scorecard.json").read_text())
    res = pd.read_parquet(out / SESSION / "residuals.parquet")
    models = [m for m in card["models"] if m.get("available")]
    own = {m["model"]: m["metrics"]["out_of_sample"]["n"] for m in models}
    assert len(set(own.values())) > 1                               # the held-out sets differ
    fitted = res.groupby(["expiry", "strike", "right"])["in_sample"].any()
    n_common = int((~fitted).sum())
    assert card["common_out_of_sample"]["n"] == n_common
    for m in models:
        assert m["metrics"]["common_out_of_sample"]["n"] == n_common
    assert "the same quotes (the calibrator's OOS universe)" not in card["notes"][0]
    assert f"the same {n_common:,} quotes" in card["notes"][0]
    assert [r["model"] for r in card["ranking_by_common_oos_median_abs_hs"]]


def test_extrapolated_vix9d_is_not_reported_as_replicated(written):
    """QSM-3: on 2026-10-16 no stored candidate lies within 9 days, so the
    bracket 2026-10-30 / 2026-11-13 would extrapolate (weight 1.357)."""
    out, _ = written
    vixd = json.loads((out / SESSION / "vix_replication.json").read_text())
    v9 = vixd["indices"]["VIX9D"]
    assert v9["cboe_rules"]["extrapolated"]
    assert v9["headline_method"] is None
    assert "replicated" not in v9
    assert "gap_vs_official" not in v9
    assert "1.357" in v9["not_replicated_reason"]
    assert any(n.startswith("VIX9D is not replicated from the stored chain") for n in vixd["gap_notes"])
    for name in ("VIX", "VIX3M", "VIX6M"):
        assert vixd["indices"][name]["headline_method"] == "cboe_rules"
        assert not vixd["indices"][name]["cboe_rules"]["extrapolated"]


def test_spxw_am_date_proxies_are_flagged_and_listed(written):
    """QSM-4: 2026-12-18 and 2027-01-15 are stored as SPXW only; VIX3M's near
    term 2027-01-15 is the PM proxy for the AM monthly, and the deviations
    say so (the root-less line is not written for a rooted chain)."""
    out, _ = written
    vixd = json.loads((out / SESSION / "vix_replication.json").read_text())
    near = vixd["indices"]["VIX3M"]["cboe_rules"]["near"]
    assert near["expiry"] == "2027-01-15"
    assert near["am_date_pm_proxy"]
    assert near["settlement"] == "PM"
    assert any("SPXW PM lines stand in for the AM SPX monthly on 2026-12-18, 2027-01-15" in d
               for d in vixd["deviations"])
    assert not any(d.startswith("Root-less") for d in vixd["deviations"])


@pytest.mark.parametrize("session", ["2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16"])
def test_committed_svi_cal_surfaces_are_certified_from_their_json(session):
    """The committed surfaces recompute as arbitrage-free from the parameters
    they publish (QSM-1: before C23f three of them had g < 0 between nodes
    and four crossed off the check grid)."""
    path = COMMITTED / session / "surface.json"
    if not path.exists():
        pytest.skip("no committed surface artifacts")
    sec = json.loads(path.read_text())["surfaces"]["svi_cal"]
    cert = _recomputed_certificates(sec)
    assert cert["arbitrage_free"], (cert["butterfly"].get("expiries_with_g_negative"),
                                    cert["calendar"].get("pairs_unordered"))
    assert sec["arbitrage_free_certified"]


# ── PROD-6: residuals.parquet, the largest file a book session adds, is written
# compactly and losslessly (byte-stream-split floats, one page per column)

def _encodings(path: Path) -> dict:
    import pyarrow.parquet as pq
    md = pq.ParquetFile(path).metadata
    rg = md.row_group(0)
    return {rg.column(i).path_in_schema: set(rg.column(i).encodings) for i in range(md.num_columns)}


def test_residuals_writer_is_lossless_and_compact(tmp_path):
    rng = np.random.default_rng(7)
    n = 3000
    quote = pd.DataFrame({"k": report.round_sig(rng.normal(0, 0.2, n)),
                          "bid_iv": report.round_sig(rng.uniform(0.1, 0.6, n))})
    frames = []
    for m in ("essvi", "svi", "svi_cal", "heston_v2"):          # the quote columns repeat per model
        f = quote.copy()
        f.insert(0, "model", m)
        f["model_iv"] = report.round_sig(f["bid_iv"].to_numpy() + rng.normal(0, 0.01, n))
        f["in_sample"] = rng.random(n) < 0.5
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    plain, compact = tmp_path / "plain.parquet", tmp_path / "compact.parquet"
    df.to_parquet(plain, index=False, compression="zstd")        # the writer before C23f
    report.write_residuals(df, compact)
    back = pd.read_parquet(compact)
    assert list(back.columns) == list(df.columns)
    for c in df.columns:
        a, b = df[c].to_numpy(), back[c].to_numpy()
        if df[c].dtype.kind == "f":
            assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), c     # bit for bit
        else:
            assert (a == b).all(), c
    enc = _encodings(compact)
    for c in ("k", "bid_iv", "model_iv"):
        assert "BYTE_STREAM_SPLIT" in enc[c], (c, enc[c])
    assert compact.stat().st_size < 0.8 * plain.stat().st_size


@pytest.mark.parametrize("session", ["2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16"])
def test_committed_residuals_are_the_compact_writers_output(session, tmp_path):
    """The committed files are what write_residuals writes (no value moved:
    test_committed_artifacts_match_a_recomputation and the README figure check
    the values), so a regenerated session cannot grow them back."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = COMMITTED / session / "residuals.parquet"
    if not path.exists():
        pytest.skip("no committed surface artifacts")
    df = pd.read_parquet(path)
    enc = _encodings(path)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            assert "BYTE_STREAM_SPLIT" in enc[c], (session, c, enc[c])
    if pq.ParquetFile(path).metadata.created_by.endswith(f"version {pa.__version__}"):
        again = tmp_path / "again.parquet"
        report.write_residuals(df, again)
        assert again.read_bytes() == path.read_bytes()
