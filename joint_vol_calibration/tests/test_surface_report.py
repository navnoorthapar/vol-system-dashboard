"""
Surface report artifacts (surface/report.py, scripts/surface_report.py): the
schema, byte-for-byte determinism, and agreement of the committed
data_store/surface artifacts with a recomputation.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.surface import report
from joint_vol_calibration.surface import svi as sv
from joint_vol_calibration.surface import vix as vix_mod

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
    # provenance (CITSEC-02): the vendor, when it was quoted (a Yahoo chain:
    # never; T from the close) and downloaded, and the quote-state gate's share
    inp = surf["inputs"]
    assert inp["source"] == "yfinance"
    assert inp["quote_time"] is None
    assert {"fetched_at", "book_state"} <= set(inp)
    assert inp["t_origin_basis"] == "session close 16:00 ET"
    assert 0.8 <= inp["near_atm_two_sided_share"] <= 1.0
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
    # the Yahoo chain lacks the SPX monthly 2026-11-20: the bracket is wider than
    # Cboe's 23-37 day window, a listed deviation (not a gap note)
    win = vixd["indices"]["VIX"]["cboe_term_window"]
    assert not win["inside"]
    assert "2026-11-20" in [m["expiry"] for m in win["missing_candidates"]]
    assert [d for d in vixd["deviations"] if d.startswith("VIX bracket ")] == [
        vix_mod.term_window_deviation(win)]
    assert not any("2026-11-20" in n for n in vixd["gap_notes"])
    ga = vixd["indices"]["VIX"]["gap_attribution"]
    assert ga["available"]
    assert ga["surface"] == "svi_cal"
    assert ga["residual_pts"] + ga["gap_pts"] + ga["truncation_pts"] == pytest.approx(0.0, abs=1e-8)
    # plan 0.5: each term at Cboe's rate (the CMT spline of the par curve before the
    # session), not one 91-day rate for both, and no rate departure listed
    rate = vixd["rate"]
    cmt = vix_mod.cmt_rates(SESSION)
    assert rate["method"] == report.CMT_RATE_METHOD
    assert rate["curve_date"] == cmt.curve_date < SESSION
    assert rate["cmt_yields_by_days"] == pytest.approx(cmt.record()["cmt_yields_by_days"], rel=1e-9)
    for name, ix in vixd["indices"].items():
        for meth in ("cboe_rules", "any_expiry_parity_forward"):
            rep = ix.get(meth) or {}
            for t in (rep.get("near"), rep.get("next")) if rep.get("ok") else ():
                if t["forward_method"] == "cboe_single_strike":
                    assert t["R"] == pytest.approx(cmt(t["expiry"]), rel=1e-9), (name, meth, t["expiry"])
    head = vixd["indices"]["VIX"]["cboe_rules"]
    assert head["near"]["R"] != head["next"]["R"]
    # tier 0.5: a chain with no quote time is calculated at its T origin, the 16:00 ET
    # close, and that stays a listed departure
    assert vixd["calculation_time_basis"] == report.CALC_T_ORIGIN
    assert vixd["calculation_time_ny"] == f"{SESSION}T16:00"
    assert sum(d.startswith("Calculation time is the session's 16:00 ET close: the chain records no "
                            "quote time (Yahoo)") for d in vixd["deviations"]) == 1
    assert not any(d.startswith(("R is", "The rate is")) for d in vixd["deviations"])
    assert vixd["gap_notes"][-1].endswith("snapshot timing vs the 16:15 ET print, stale quotes.")
    idx = json.loads((out / "index.json").read_text())
    assert [r["session"] for r in idx["sessions"]] == [SESSION]
    # the method version the bot's backfill compares (cloud_refresh.artifact_gaps)
    for doc in (surf, card, vixd, idx["sessions"][0]):
        assert doc["report_version"] == report.REPORT_VERSION
        assert report.stored_report_version(doc) == report.REPORT_VERSION


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
    # the replicated levels moved at report_version 4 (Cboe's rate per term): a
    # committed artifact written before is stale until the bot's backfill
    # rebuilds it (cloud_refresh.artifact_gaps), and is compared from then on
    if report.stored_report_version(vo) >= 4:
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


# ── 0.7: the session's Cboe book is the surface's source; Yahoo the fallback ──

def _capture_store(tmp_path, monkeypatch, quote_time, *, book_state="closing", empty_atm=False):
    """A data_store/options/ with SESSION's committed Yahoo chain and a Cboe
    capture of SESSION made of the same quotes in the capture layout
    (cboe_downloader.parse_cboe_options_payload: no 'underlying', Cboe's own
    IV in implied_vol, nullable integer counts), stamped ``quote_time``."""
    import shutil

    from joint_vol_calibration.data import database as db

    opts = tmp_path / "options"
    (opts / "cboe").mkdir(parents=True)
    yahoo = db.options_snapshot_path(SESSION, "SPX")
    shutil.copy2(yahoo, opts / yahoo.name)
    frame = pd.read_parquet(yahoo).drop(columns=["underlying"])
    frame = frame.assign(source="cboe", quote_time=quote_time, implied_vol=0.5,
                         open_interest=frame["open_interest"].round().astype("Int64"),
                         volume=frame["volume"].round().astype("Int64"))
    if empty_atm:                                   # an empty book: no near-ATM bids
        near = (frame["strike"] / frame["underlying_price"] - 1.0).abs() < 0.03
        frame.loc[near, "bid"] = 0.0
    stem = f"spx_{SESSION}_013000Z"
    frame.to_parquet(opts / "cboe" / f"{stem}.parquet", index=False)
    side = {"underlying": "SPX", "session": SESSION, "quote_time": quote_time,
            "book_state": book_state, "fetched_at": "2026-10-17T01:30:00Z",
            "sha256": hashlib.sha256(b"payload").hexdigest(), "rows": len(frame),
            "two_sided_rows": int(((frame["bid"] > 0) & (frame["ask"] >= frame["bid"])).sum())}
    (opts / "cboe" / f"{stem}.json").write_text(json.dumps(side))
    monkeypatch.setattr(db, "OPTIONS_DIR", opts)
    return opts / "cboe" / f"{stem}.parquet", side


def test_the_surface_is_built_from_the_sessions_cboe_book(tmp_path, monkeypatch):
    """CITSEC-01 / QUANT-03 / VALIDATOR-02: the surface was built from the
    Yahoo chain while the session's Cboe book sat next to it. With a stored
    capture that passes the forward ledger's quote-state gates, the report
    reads the capture (mapped to the stored-chain layout), measures T from its
    quote time, and surface.json, the scorecard notes, the Heston row and the
    VIX timing line say so."""
    from joint_vol_calibration.surface import quotes as sqm
    from joint_vol_calibration.surface import scorecard as sc

    qt = "2026-10-17T00:30:00Z"                                       # Fri 20:30 ET: closing book
    path, side = _capture_store(tmp_path, monkeypatch, qt)
    out = report.run_session(SESSION, models=["essvi", "heston_v2"])
    inp = out["surface"]["inputs"]
    assert inp["source"] == "cboe"
    assert inp["chain"] == f"data_store/options/cboe/{path.name}"
    assert inp["chain_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inp["raw_sha256"] == side["sha256"]
    assert inp["quote_time"] == qt
    assert inp["fetched_at"] == side["fetched_at"]
    assert inp["book_state"] == side["book_state"]
    assert inp["source_reason"].startswith("the newest stored Cboe book of the session")
    assert inp["refused_books"] == []
    assert inp["chain_rows"] == side["rows"]
    # a closing book (after the close, outside the next GTH window): T from the close
    assert inp["t_origin_basis"] == "session close 16:00 ET"
    assert any(n.startswith("Quotes are Cboe's delayed SPX book") for n in out["scorecard"]["notes"])
    heston = {m["model"]: m for m in out["scorecard"]["models"]}["heston_v2"]
    assert "The fit was made on the session's Yahoo chain" in heston["description"]
    assert heston["fit"]["scored_on_fit_chain"] is False
    # tier 0.5: the replication is calculated at the instant the book's quotes stand at,
    # Cboe's own timing: for a closing book the 16:15 ET close, not the chain's 16:00 ET
    # T origin, and that is no departure; the note says when the book was quoted
    vixd = out["vix_replication"]
    assert vixd["calculation_time_ny"] == f"{SESSION}T16:15"
    assert vixd["calculation_time_basis"] == report.CALC_CLOSING_BOOK_CLOSE
    assert qt in vixd["calculation_time_source"]
    assert not any(d.startswith("Calculation time") for d in vixd["deviations"])
    assert not any("Yahoo" in d for d in vixd["deviations"])
    assert sum(n.startswith("The book is the session's closing book, quoted at 2026-10-16 20:30 ET after "
                            f"the close: the replication is timed at the 16:15 ET close of {SESSION}")
               for n in vixd["gap_notes"]) == 1
    # the snapshot's timing against the 16:15 ET print is then no cause of the gap
    att = [n for n in vixd["gap_notes"] if n.startswith("Gap attribution (")]
    assert all("snapshot timing" not in n for n in att)
    # the mapping: Cboe's IV is not read as this repository's
    raw = sqm.stored_chain_from_capture(pd.read_parquet(path))
    assert set(raw["underlying"]) == {"SPX"}
    assert raw["implied_vol"].isna().all()
    assert (raw["vendor_iv"] == 0.5).all()
    # the same quotes from the Yahoo chain (forced): the universe is identical
    yq = sqm.load_session_quotes(SESSION, source="yahoo")
    cq = sqm.load_session_quotes(SESSION)
    pd.testing.assert_frame_equal(yq.quotes, cq.quotes)
    assert yq.inputs["source"] == "yfinance"
    assert "Yahoo chain requested" in yq.inputs["source_reason"]
    none = np.zeros(len(yq.quotes), dtype=bool)
    ctx = sc.ScoreContext(yq, none, sc.surface_fit_mask(yq.quotes, none))
    assert "The fit was made" not in sc.heston_v2_factory(ctx).description


def test_a_gth_book_is_calculated_at_its_quote_time_and_departs_from_nothing(tmp_path, monkeypatch):
    """Tier 0.5: a Global Trading Hours book (the Mon-Thu book slots) is
    calculated at its quote time, as Cboe times a calculation on the quotes
    standing then, and vix_replication.json says so: the timing is no listed
    departure; the note says the Cboe close it is compared with is the earlier
    16:15 ET print, and the gap attribution keeps that timing among its causes."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    qt = "2026-10-19T02:30:00Z"                     # Sun 22:30 ET: GTH for Monday 2026-10-19
    _capture_store(tmp_path, monkeypatch, qt, book_state="gth")
    out = report.run_session(SESSION, models=["essvi"])
    vixd = out["vix_replication"]
    ny = datetime.fromisoformat(qt).astimezone(ZoneInfo("America/New_York")).replace(tzinfo=None)
    assert out["surface"]["inputs"]["t_origin_basis"].startswith("quote_time")
    assert vixd["calculation_time_ny"] == ny.isoformat(timespec="minutes")
    assert vixd["calculation_time_basis"] == report.CALC_BOOK_QUOTE_TIME
    assert vixd["calculation_time_source"].startswith(f"the Cboe book's quote time {qt}")
    assert not any(d.startswith("Calculation time") for d in vixd["deviations"])
    assert sum(n.startswith(f"The replication is timed at the book's quote time, {ny:%Y-%m-%d %H:%M} ET")
               and n.endswith(f"16:15 ET print of {SESSION}, so the gap includes the market's move "
                              "between the two instants.") for n in vixd["gap_notes"]) == 1
    att = [n for n in vixd["gap_notes"] if n.startswith("Gap attribution (")]
    assert all("snapshot timing vs the 16:15 ET print" in n for n in att)


@pytest.mark.parametrize(("quote_time", "empty_atm", "why"), [
    ("2026-10-16T19:00:00Z", False, "before the 2026-10-16 close"),        # 15:00 ET
    ("2026-10-17T00:30:00Z", True, "not a live book"),                     # no near-ATM bids
])
def test_a_cboe_book_failing_the_quote_state_gates_falls_back_to_yahoo(
        tmp_path, monkeypatch, quote_time, empty_atm, why):
    """The forward ledger's gates decide (forward_ledger.book_time_refusal,
    book_share_refusal): a capture quoted before the close, or an empty book,
    is not the session's; the Yahoo chain is used and the report says why."""
    from joint_vol_calibration.surface import quotes as sqm

    _capture_store(tmp_path, monkeypatch, quote_time, empty_atm=empty_atm)
    sq = sqm.load_session_quotes(SESSION)
    assert sq.inputs["source"] == "yfinance"
    assert sq.inputs["chain"] == f"data_store/options/spx_{SESSION}.parquet"
    assert why in sq.inputs["source_reason"]
    assert sq.inputs["source_reason"].endswith("the session's Yahoo chain (the calibration's "
                                               "download) is the fallback")
    assert len(sq.inputs["refused"]) == 1
    with pytest.raises(ValueError, match="no stored Cboe book"):
        sqm.load_session_quotes(SESSION, source="cboe")


def test_committed_cboe_books_are_the_surface_source_of_their_sessions():
    """Every session with a stored Cboe capture that passes the quote-state
    gates is read from that capture, with its quote time as the T origin of a
    GTH book (whatever sessions the bot has captured)."""
    from joint_vol_calibration.backtest import forward_ledger as fl
    from joint_vol_calibration.data import database as db
    from joint_vol_calibration.data import option_chain as oc
    from joint_vol_calibration.surface import quotes as sqm

    sessions = sorted({p.name.split("_")[1] for p in (db.OPTIONS_DIR / "cboe").glob("spx_*.parquet")})
    for s in sessions:
        try:
            sqm.session_spot(s)
        except ValueError:
            continue                                  # the test database lacks the session's close
        path, side = sqm.session_captures(s)[0]
        if fl.book_time_refusal(s, side.get("quote_time")):
            continue
        sq = sqm.load_session_quotes(s)
        if sq.inputs["source"] != "cboe":            # refused by the near-ATM gate: said so
            assert "no stored Cboe book passes" in sq.inputs["source_reason"], s
            continue
        assert sq.inputs["chain"] == f"data_store/options/cboe/{path.name}"
        assert sq.diagnostics["quote_time"] == side["quote_time"]
        gth = oc.t_origin(side["quote_time"], s) != oc.session_close(s)
        assert sq.diagnostics["t_origin_basis"] == (oc.GTH_BASIS if gth else oc.CLOSE_BASIS)
        assert sq.diagnostics["n_expiries"] > 17                      # the full listing
        # CITSEC-03: with the full listing the VIX bracket is Cboe's (both terms 23-37 days out)
        calc, basis = report._calc_time_ny(sq)
        if gth:                                       # tier 0.5: a GTH book at its quote time
            assert basis == report.CALC_BOOK_QUOTE_TIME
            assert calc.isoformat() == sq.diagnostics["t_origin"][:19]
        terms = vix_mod.term_tables(sq.raw, s, calc)
        rep = vix_mod.replicate_index(terms, sq.r_ref, 30, spot=sq.spot)
        win = vix_mod.cboe_term_window(rep, terms, calc)
        assert win["inside"], (s, win)
        assert win["missing_candidates"] == []


def test_a_session_with_a_live_cboe_book_and_no_fit_gets_a_surface(tmp_path, monkeypatch):
    """CITSEC-05: the surface was built only after the Heston fit passed its
    gate (2026-10-20 has a full Cboe book, no accepted fit and no surface).
    With no stored fit, the report is still written from the session's Cboe
    book; the scorecard's Heston row is absent and says why, and the other
    models, the arbitrage screen and the VIX replication are unchanged."""
    from joint_vol_calibration.surface import quotes as sqm

    _capture_store(tmp_path, monkeypatch, "2026-10-17T00:30:00Z")
    assert sqm.captured_sessions() == [SESSION]
    assert sqm.cboe_book_refusal(SESSION) is None
    nofit = tmp_path / "calibrations"
    nofit.mkdir()
    out = report.run_session(SESSION, calibrations_dir=nofit, models=["essvi", "heston_v2"])
    assert out["surface"]["inputs"]["source"] == "cboe"
    rows = {m["model"]: m for m in out["scorecard"]["models"]}
    assert rows["essvi"]["available"]
    assert rows["heston_v2"]["available"] is False
    assert rows["heston_v2"]["reason"].startswith(f"no joint SPX/VIX Heston fit is stored for {SESSION}")
    assert set(out["residuals"]["model"]) == {"essvi"}
    assert out["scorecard"]["common_out_of_sample"]["models"] == ["essvi"]
    assert any(n.endswith("Heston is not scored on this session: " + rows["heston_v2"]["reason"])
               for n in out["scorecard"]["notes"])
    assert out["vix_replication"]["indices"]["VIX"].get("replicated") is not None
    row = report.index_row(out)
    assert set(row["models"]) == {"essvi"}
    with_fit = report.run_session(SESSION, models=["essvi", "heston_v2"])
    assert {m["model"]: m for m in with_fit["scorecard"]["models"]}["heston_v2"]["available"]
    assert with_fit["vix_replication"]["indices"] == out["vix_replication"]["indices"]
    assert with_fit["surface"]["arbitrage"] == out["surface"]["arbitrage"]


def test_a_session_whose_books_fail_the_gates_has_no_cboe_surface(tmp_path, monkeypatch):
    from joint_vol_calibration.surface import quotes as sqm

    _capture_store(tmp_path, monkeypatch, "2026-10-17T00:30:00Z", empty_atm=True)
    assert "not a live book" in sqm.cboe_book_refusal(SESSION)
    assert sqm.cboe_book_refusal("2026-10-15").startswith("no stored Cboe capture")


def test_a_capture_that_cannot_be_read_is_refused_not_a_crash(tmp_path, monkeypatch):
    """A truncated capture parquet or sidecar (a disk or commit gone wrong)
    crashed the session's surface report, and through
    cloud_refresh.artifact_gaps every backfill run. It is now a refused book:
    the report falls back to the Yahoo chain and says why, and the Cboe-only
    check gives a reason instead of raising."""
    from joint_vol_calibration.surface import quotes as sqm

    path, _side = _capture_store(tmp_path, monkeypatch, "2026-10-17T00:30:00Z")
    path.write_bytes(b"PAR1 truncated")
    path.with_suffix(".json").write_text("{not json")
    sq = sqm.load_session_quotes(SESSION)
    assert sq.inputs["source"] == "yfinance"
    assert f"options/cboe/{path.name}: unreadable (" in sq.inputs["source_reason"]
    assert "unreadable" in sqm.cboe_book_refusal(SESSION)


def test_the_bracket_checked_is_the_one_the_vix_headline_uses():
    """The term-window deviation read cboe_rules alone: when its single-strike
    forward is refused and the headline is parity_forward, the replication the
    page prints was not checked against Cboe's window."""
    from datetime import datetime

    from joint_vol_calibration.data import option_chain as oc

    calc = datetime(2026, 10, 26, 16, 0, tzinfo=oc.NY_TZ).replace(tzinfo=None)
    terms = [{"expiry": e, "candidate": True} for e in ("2026-11-06", "2026-11-27")]
    pf = {"ok": True, "near": {"expiry": "2026-11-06", "days": 10.6},
          "next": {"expiry": "2026-11-27", "days": 31.6}}
    rep = {"cboe_rules": {"ok": False, "reason": "implausible forward"}, "parity_forward": pf,
           "headline_method": "parity_forward"}
    win = report._vix_term_window(rep, terms, calc)
    assert win is not None
    assert not win["inside"]
    assert win["near"] == {"expiry": "2026-11-06", "days": 10.6}
    assert report._vix_term_window({**rep, "headline_method": None}, terms, calc) is None


def test_an_expiry_with_no_fit_set_quote_is_dropped_not_fatal(monkeypatch, tmp_path):
    """Found by simulate_next_session.py (2026-10-29): on a book whose expiries a
    day apart carried independent noise, the l1 repair moved 186 of 188 quotes of
    one expiry outside their spread, so it had no fit-set quote and the scorecard
    raised 'surface fit set misses 1 quoted expiries': the session lost its whole
    surface report, and the deploy gate failed on the missing smiles. Such an
    expiry now leaves the scored universe with its reason; the rest is scored."""
    real = report.arb.screen
    victim: dict = {}

    def screen(q):
        out = real(q)
        exps = list(dict.fromkeys(q["expiry"]))
        victim.setdefault("expiry", exps[len(exps) // 2])
        hit = (q["expiry"] == victim["expiry"]).to_numpy()
        if hit.any():
            out.quote_flags.loc[hit, "repair_outside_spread"] = True
        return out

    monkeypatch.setattr(report.arb, "screen", screen)
    arts = report.run_session(SESSION)
    uni = arts["surface"]["universe"]
    e = victim["expiry"]
    assert [d["expiry"] for d in uni["dropped_expiries"]] == [e]
    assert "outside its spread" in uni["dropped_expiries"][0]["reason"]
    res = arts["residuals"]
    assert e not in set(res["expiry"])
    n_scored = len(res[res["model"] == res["model"].iloc[0]])
    assert uni["n_universe"] == n_scored == arts["scorecard"]["n_quotes"]
    assert uni["n_dropped_no_fit_set"] == uni["dropped_expiries"][0]["n_quotes"] > 0
    assert uni["n_expiries"] == res["expiry"].nunique()
    assert e not in {t["expiry"] for t in arts["surface"]["term_structure"]}


def test_a_session_without_a_recent_treasury_curve_lists_the_rate_departure(monkeypatch):
    """plan 0.5: when the stored Treasury par curve has no date within the
    allowed age before the session, the replication falls back to the
    session's reference rate for both terms and the deviations say so and
    why; with the curve, the rate is Cboe's per term and no line is listed."""
    from types import SimpleNamespace

    from joint_vol_calibration.data import database as db
    from joint_vol_calibration.surface import quotes as sqm

    sq = SimpleNamespace(session=SESSION, r_ref=0.0415, r_ref_source=sqm.IRX_SOURCE)
    rate, doc, line = report._vix_rate(sq)
    assert isinstance(rate, vix_mod.CMTRates)
    assert line is None
    assert doc["method"] == report.CMT_RATE_METHOD

    def stale(session, **kw):
        raise db.StaleDataError(f"newest Treasury par curve before {session} is 2026-10-01 (15 days old; "
                                "allowed 7)")
    monkeypatch.setattr(vix_mod, "cmt_rates", stale)
    rate, doc, line = report._vix_rate(sq)
    assert rate == sq.r_ref
    assert doc == {"method": "one_rate_for_both_terms", "R_cc": sq.r_ref, "source": sq.r_ref_source,
                   "cmt_unavailable": f"newest Treasury par curve before {SESSION} is 2026-10-01 "
                                      "(15 days old; allowed 7)"}
    assert line.startswith("The rate is the ^IRX bill rate for both terms, not Cboe's spline on the "
                           "Treasury CMT yields at each term's expiration: newest Treasury par curve")
    zc = SimpleNamespace(session=SESSION, r_ref=0.0415, r_ref_source="Treasury zero curve 2026-10-16")
    assert report._vix_rate(zc)[2].startswith("The rate is the Treasury zero curve's 91-day rate")


def test_the_vix_sentence_names_the_rate_when_its_terms_took_the_parity_forward():
    """Tier 0.5 review: the producer's gap-attribution sentence dropped the
    rate from the causes of the VIX's unexplained part whenever the session
    had Cboe's curve, but a VIX replicated on the parity forward discounts each
    term at its parity regression's D (R = -ln D / T), not at that curve. The
    rate is left out only when both of the VIX's headline terms took Cboe's
    single-strike forward."""
    ss, par = "cboe_single_strike", "parity_regression"

    def ix(method, near, nxt):
        return {"headline_method": method,
                method: {"near": {"forward_method": near}, "next": {"forward_method": nxt}}}
    assert report._headline_at_cboe_rate(ix("cboe_rules", ss, ss))
    assert report._headline_at_cboe_rate(ix("parity_forward", ss, ss))     # no parity fit for either
    assert not report._headline_at_cboe_rate(ix("parity_forward", par, par))
    assert not report._headline_at_cboe_rate(ix("parity_forward", ss, par))
    assert not report._headline_at_cboe_rate({"headline_method": None, "not_replicated_reason": "..."})
