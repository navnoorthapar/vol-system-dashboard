"""
test_session_artifacts.py — the committed per-session artifacts agree with
their inputs and with each other (C23).

The daily bot rebuilds these after each accepted fit (scripts/cloud_refresh.py
refresh_session_artifacts) and commits them behind this module, its artifact
gate. So nothing here types a value a new session replaces: every check is
structural or a consistency between committed files, and holds for any number
of sessions.

  surface       data_store/surface/<session>/ has its four files for exactly
                the sessions index.json lists, each built from the committed
                chain it names (sha256)
  diagnostics   data_store/calibrations/diagnostics/<session>.json is the
                session's own committed v2 fit, and summary.json has one row
                per complete file
  greeks        the Greeks surface and the scenario grid come from the same
                committed, gate-passing model_version 2 fit
"""
import hashlib
import json
import pickle

import pytest

from joint_vol_calibration.calibration.joint_calibrator import (
    calibration_is_legacy,
    is_acceptable_calibration,
)
from joint_vol_calibration.config import DATA_DIR, ROOT_DIR

SURFACE = DATA_DIR / "surface"
DIAG = DATA_DIR / "calibrations" / "diagnostics"
SURFACE_FILES = ("residuals.parquet", "scorecard.json", "surface.json", "vix_replication.json")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sessions(root, pattern):
    return sorted(p.name for p in root.glob(pattern) if p.is_dir())


def test_surface_index_lists_exactly_the_session_directories():
    if not (SURFACE / "index.json").exists():
        pytest.skip("no surface artifacts")
    idx = json.loads((SURFACE / "index.json").read_text())
    listed = [r["session"] for r in idx["sessions"]]
    assert listed == sorted(listed)
    assert listed == _sessions(SURFACE, "20*")
    for s in listed:
        assert sorted(p.name for p in (SURFACE / s).iterdir()) == list(SURFACE_FILES), s


def surface_input_problems(inp: dict, session: str, root) -> list:
    """What is wrong with a surface.json ``inputs`` block (empty: nothing):
    it names the committed book it was built from, under ``root``, with that
    file's sha256. The book is the session's Cboe capture
    (data_store/options/cboe/spx_<session>_<HHMMSS>Z.parquet, whose sidecar
    holds its quote time, book state, download instant and payload hash) or,
    as the fallback, its Yahoo chain data_store/options/spx_<session>.parquet."""
    import re

    out = []
    chain = root / str(inp.get("chain"))
    if inp.get("source") == "cboe":
        want = rf"data_store/options/cboe/spx_{session}_\d{{6}}Z\.parquet"
        if not re.fullmatch(want, str(inp.get("chain"))):
            out.append(f"a Cboe-sourced surface names {inp.get('chain')!r}, not a Cboe capture of {session}")
        side = chain.with_suffix(".json")
        if side.exists():
            sc = json.loads(side.read_text())
            for key, want in (("quote_time", sc.get("quote_time")), ("raw_sha256", sc.get("sha256")),
                              ("book_state", sc.get("book_state")), ("fetched_at", sc.get("fetched_at"))):
                if inp.get(key) != want:
                    out.append(f"inputs.{key} {inp.get(key)!r} is not the capture's {want!r}")
        else:
            out.append(f"no sidecar {side.name} next to the capture")
    else:
        if inp.get("chain") != f"data_store/options/spx_{session}.parquet":
            out.append(f"a {inp.get('source')}-sourced surface names {inp.get('chain')!r}")
    if not chain.exists():
        out.append(f"{inp.get('chain')} is not committed")
    elif inp.get("chain_sha256") != _sha256(chain):
        out.append(f"inputs.chain_sha256 is not the sha256 of {inp.get('chain')}")
    return out


@pytest.mark.parametrize("session", _sessions(SURFACE, "20*") or ["none"])
def test_surface_artifacts_are_built_from_the_committed_chain(session):
    if session == "none":
        pytest.skip("no surface artifacts")
    doc = json.loads((SURFACE / session / "surface.json").read_text())
    assert surface_input_problems(doc["inputs"], session, ROOT_DIR) == []
    card = json.loads((SURFACE / session / "scorecard.json").read_text())
    vix = json.loads((SURFACE / session / "vix_replication.json").read_text())
    assert doc["session"] == card["session"] == vix["session"] == session
    idx = {r["session"]: r for r in json.loads((SURFACE / "index.json").read_text())["sessions"]}
    assert idx[session]["n_quotes"] == card["n_quotes"]


def test_a_surface_built_from_a_cboe_capture_passes_the_artifact_gate(tmp_path):
    """0.7: the bot builds a session's surface from its Cboe capture when that
    passes the quote-state gates (surface.quotes.select_session_chain). The
    gate accepts it when it names the capture, its sha256 and its sidecar's
    quote time, book state, download instant and payload hash, and refuses a
    capture of another session or a hash that is not the file's."""
    s = "2026-01-05"
    cap = tmp_path / "data_store" / "options" / "cboe" / f"spx_{s}_013000Z.parquet"
    cap.parent.mkdir(parents=True)
    cap.write_bytes(b"capture")
    side = {"quote_time": "2026-01-06T01:15:00Z", "sha256": "ab" * 32, "book_state": "gth",
            "fetched_at": "2026-01-06T01:30:00Z"}
    cap.with_suffix(".json").write_text(json.dumps(side))
    inp = {"source": "cboe", "chain": f"data_store/options/cboe/{cap.name}",
           "chain_sha256": _sha256(cap), "quote_time": side["quote_time"],
           "raw_sha256": side["sha256"], "book_state": "gth", "fetched_at": side["fetched_at"]}
    assert surface_input_problems(inp, s, tmp_path) == []
    assert surface_input_problems(inp, "2026-01-06", tmp_path)            # another session's book
    assert surface_input_problems({**inp, "chain_sha256": "0" * 64}, s, tmp_path)
    assert surface_input_problems({**inp, "quote_time": None}, s, tmp_path)
    yahoo = tmp_path / "data_store" / "options" / f"spx_{s}.parquet"
    yahoo.write_bytes(b"chain")
    yin = {"source": "yfinance", "chain": f"data_store/options/spx_{s}.parquet",
           "chain_sha256": _sha256(yahoo)}
    assert surface_input_problems(yin, s, tmp_path) == []
    assert surface_input_problems({**yin, "chain": inp["chain"]}, s, tmp_path)


def _diag_files():
    return sorted(p for p in DIAG.glob("20*.json"))


@pytest.mark.parametrize("path", _diag_files() or [None], ids=lambda p: p.stem if p else "none")
def test_diagnostics_belong_to_the_sessions_own_fit(path):
    if path is None:
        pytest.skip("no diagnostics artifacts")
    d = json.loads(path.read_text())
    assert d["schema_version"] == 1
    assert d["session"] == path.stem
    with open(DATA_DIR / "calibrations" / f"joint_cal_{path.stem}.pkl", "rb") as f:
        fit = pickle.load(f)
    assert fit.get("model_version") == 2
    st = d["stored_fit"]
    assert st["loss"] == fit["loss"]
    assert st["params"] == {k: fit["params"][k] for k in st["params"]}
    # the stored loss is recomputed from the residuals on the fit's own inputs
    assert st["loss_recomputed_from_residuals"] == pytest.approx(st["loss"], rel=1e-12)
    for prov in d["provenance"].values():
        assert prov["code_commit"]


def test_diagnostics_summary_has_one_row_per_complete_session():
    if not (DIAG / "summary.json").exists():
        pytest.skip("no diagnostics summary")
    summary = json.loads((DIAG / "summary.json").read_text())
    complete = [p.stem for p in _diag_files()
                if {"heston", "rough"} <= set(json.loads(p.read_text())["provenance"])]
    assert [r["session"] for r in summary["rows"]] == complete
    assert summary["n_sessions"] == len(complete)


def _provenance(path, key):
    import pyarrow.parquet as pq
    return json.loads((pq.read_schema(path).metadata or {})[key])


def test_greeks_and_scenario_grid_come_from_one_accepted_fit():
    g_path = DATA_DIR / "greeks" / "greeks_surface.parquet"
    s_path = DATA_DIR / "greeks" / "scenario_grid.parquet"
    if not s_path.exists():
        pytest.skip("scenario grid not generated")
    g = _provenance(g_path, b"greeks_provenance")
    s = _provenance(s_path, b"scenario_provenance")
    src = g["source"]["calibration_file"]
    assert s["source"]["calibration_file"] == src
    assert g["as_of_date"] == s["as_of_date"]
    with open(ROOT_DIR / src, "rb") as f:
        fit = pickle.load(f)
    assert not calibration_is_legacy(fit)
    assert is_acceptable_calibration(fit)[0]
    assert str(fit.get("as_of_date") or fit.get("spot_date"))[:10] == g["as_of_date"]
    assert g["params"] == {k: fit["params"][k] for k in g["params"]}
