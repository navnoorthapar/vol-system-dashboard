"""
test_readme.py — every number the README front door quotes is the committed artifact's.

The README's first screen quotes three results with their intervals, and its
calibration, backtest, data and limitation sections quote tables and figures;
the validation pack quotes per-session figures. They are a committed snapshot,
docs/results/readme_results.json, written by scripts/readme_results.py from
data_store: every figure as the documents print it, the artifacts it was
computed from (path, sha256, report_version) and the date and commit it was
taken at. The documents say 'as of <date>' and name the live site for the
newest session.

It used to be recomputed here from data_store directly, so the daily job
rebuilding a quoted artifact (the surface method's report_version 3 backfill
moves 2026-10-26 from 1.60 to 1.66 half-spreads, rebuilt from its Cboe book)
turned every pull request red until someone edited the README. Now:

  (a) every figure is in its document as the snapshot holds it, and both
      documents state its date: always;
  (b) the snapshot is what the current data_store gives: a figure that moved
      fails when none of its artifacts changed (the code changed) or when one
      the daily job never writes changed (a commit did); when only artifacts
      the job writes changed (what its commit steps stage, and the cache every
      run rewrites), it is a StaleSnapshot warning naming them and
      `make readme-results`, never a failure.

The results range over every stored session in the span the snapshot names
with a surface scorecard and a model_version 2 fit (VALIDATOR-06, CODE-04,
CIO-05), so a session the job adds after the span moves nothing. C23_SESSIONS
are the five sessions the C23 round measured: the claims about that round (the
DE timings, the Data table's option counts, the article's C23 note) stay pinned
to them, and the article's note is still recomputed here. The Data table's rows
are counted as of the end date each states (CLM-03). test_readme.py is in no
gate of the daily job (daily_refresh.yml).

Also: the figure is drawn from the surface report it names (its sidecar's
numbers and source hash), docs/corrections.md and the validation pack's
findings log are renders of the corrections registry, every relative link
resolves, the badges carry no hand-typed counts, and QuantLib is a dev-only
test oracle.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import importlib.util
import json
import pickle
import re
import shutil
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_store"
README_RAW = (ROOT / "README.md").read_text()
FIRST_SCREEN_RAW = README_RAW.split("\n---\n")[0]
# whitespace-collapsed copies: a re-wrapped paragraph must not fail a number check
README = " ".join(README_RAW.split())
FIRST_SCREEN = " ".join(FIRST_SCREEN_RAW.split())
C23_SESSIONS = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")
C23_HEADLINE = "2026-10-16"
FIGURE = "2026-10-16"                       # the session docs/figures/smile_<session>.png draws
MINUS = "−"
PACK_RAW = (ROOT / "docs" / "model_validation" / "README.md").read_text()
PACK = " ".join(PACK_RAW.split())


def _script(name: str):
    """scripts/<name>.py as a module (registered, so its dataclasses resolve)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _script("readme_results")
SNAP = json.loads(rr.SNAPSHOT.read_text())
RESULT_SESSIONS = tuple(SNAP["sessions"])
LAST = RESULT_SESSIONS[-1]


def _cal(s: str) -> dict:
    with open(DATA / "calibrations" / f"joint_cal_{s}.pkl", "rb") as fh:
        return pickle.load(fh)


def fmt(x: float, spec: str) -> str:
    return format(x, spec).replace("-", MINUS)


def rng(values, spec: str) -> str:
    v = list(values)
    return f"{fmt(min(v), spec)}–{fmt(max(v), spec)}"


def _claim(cid: str) -> dict:
    return next(k for k in SNAP["claims"] if k["id"] == cid)


def _moved(text: str) -> str:
    """``text`` with its first decimal figure moved by one."""
    def up(m: re.Match) -> str:
        return f"{float(m.group()) + 1:.{len(m.group().split('.')[1])}f}"

    out = re.sub(r"\d+\.\d+", up, text, count=1)
    assert out != text, text
    return out


@pytest.fixture
def store_copy(tmp_path):
    """A private copy of data_store/ to change; the snapshot's Store reads it."""
    dst = tmp_path / "data_store"
    shutil.copytree(DATA, dst)
    return dst


@pytest.fixture(scope="module")
def inference():
    return json.loads((DATA / "backtest" / "inference" / "summary.json").read_text())


# ── (a) the documents say what the snapshot says ──────────────────────────────

def test_documents_quote_the_snapshot():
    """Every figure the README, the validation pack and DATA_SOURCES.md quote
    is in its place as the snapshot holds it, and both documents state the
    snapshot's date."""
    assert rr.doc_problems(SNAP) == []


def test_a_document_that_disagrees_with_the_snapshot_fails():
    claim = _claim("result_1/last")
    pattern = r"\s+".join(map(re.escape, claim["text"].split()))
    moved, n = re.subn(pattern, lambda m: _moved(claim["text"]), README_RAW)
    assert n == 1

    def read(doc: str) -> str:
        return moved if doc == rr.README else rr.read_doc(doc)

    problems = rr.doc_problems(SNAP, read)
    assert [p.split(":")[0] for p in problems] == ["result_1/last"]
    asof = f"as of {SNAP['generated']['date']}"
    undated = README_RAW.replace(asof, "as of 1999-01-01")

    def read_undated(doc: str) -> str:
        return undated if doc == rr.README else rr.read_doc(doc)

    assert [p for p in rr.doc_problems(SNAP, read_undated) if p.startswith("generated")]


def test_results_name_their_date_and_the_live_site():
    """The results' preamble links the snapshot it quotes with its date, and
    names the live site for the newest session; the pack links it too."""
    date = SNAP["generated"]["date"]
    assert f"[as of {date}](docs/results/readme_results.json)" in FIRST_SCREEN
    assert f"[as of {date}](../results/readme_results.json)" in PACK
    live = "[live site](https://navnoorbawa.me) shows the newest session"
    assert live in FIRST_SCREEN_RAW.replace("\n", " ")
    assert dt.date.fromisoformat(date) >= dt.date.fromisoformat(SNAP["params"]["sessions_through"])
    assert dt.date.fromisoformat(date) >= dt.date.fromisoformat(SNAP["params"]["data_through"])


# ── (b) the snapshot is what the artifacts give ───────────────────────────────

def test_snapshot_is_what_the_artifacts_give():
    """A figure whose artifacts are unchanged since the snapshot recomputes to
    its text; one the daily job's rebuilds moved is a StaleSnapshot warning
    (naming the artifacts and make readme-results), not a failure."""
    assert rr.check(SNAP) == []


def _part_problems(topic: str) -> list[str]:
    """(a) and (b) for the claims of one part of the documents."""
    part = {**SNAP, "claims": [k for k in SNAP["claims"] if k["id"].startswith(f"{topic}/")]}
    assert part["claims"], topic
    return rr.doc_problems(part) + rr.check(SNAP, topics=[topic])


def test_result_2_joint_calibration_heston_and_rough():
    """Result 2 (C24-I1 cites this test): the joint Heston fit's held-out
    range, the newest session's fit, the σ cap and rough Heston, over the
    sessions the README names, are the snapshot's and the snapshot is the
    fits' and the diagnostics'."""
    assert _part_problems("result_2") == []


def test_validation_pack_counts_and_outcomes_are_the_artifacts():
    """VALIDATOR-10 (C24-I1 cites this test): the pack was frozen at five
    sessions and disagreed with the site. Its span, the §1 limitations that
    count sessions, §2's identifiability, held-out and replication counts and
    the §3 outcome rows are the snapshot's, and the snapshot is the artifacts'."""
    assert _part_problems("pack") == []


def test_snapshot_names_each_artifact_with_its_hash_and_method():
    gen = SNAP["generated"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", gen["date"])
    assert re.fullmatch(r"[0-9a-f]{40}", gen["commit"])
    assert gen["command"] == rr.COMMAND == "make readme-results"
    assert "readme-results:" in (ROOT / "Makefile").read_text()
    assert SNAP["schema_version"] == rr.SCHEMA_VERSION
    assert SNAP["sessions"]
    assert SNAP["claims"]
    for key, entry in SNAP["artifacts"].items():
        assert key.startswith("data_store/"), key
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), key
        if key.startswith("data_store/surface/"):
            assert isinstance(entry["report_version"], int), key
        else:
            assert entry["report_version"] is None, key
    ids = [k["id"] for k in SNAP["claims"]]
    assert len(ids) == len(set(ids))
    for k in SNAP["claims"]:
        assert k["text"], k["id"]
        assert k["sources"] or k["id"] in {"backtest_prose/beta", "limitations/beta"}, k["id"]
        assert set(k["sources"]) <= set(SNAP["artifacts"]), k["id"]
        assert k["doc"] in {rr.README, rr.PACK, rr.SOURCES_DOC}, k["id"]


def test_a_figure_whose_artifacts_are_unchanged_must_recompute():
    """An edited snapshot (or code that now computes another figure) fails:
    nothing it was computed from changed."""
    snap = copy.deepcopy(SNAP)
    k = next(c for c in snap["claims"] if c["id"] == "result_3/sharpe")
    k["text"] = _moved(k["text"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rr.StaleSnapshot)
        failures = rr.check(snap, topics=["result_3"])
    assert len(failures) == 1
    assert failures[0].startswith("result_3/sharpe ")
    assert "every artifact it reads unchanged" in failures[0]


def test_an_artifact_changed_outside_the_daily_job_fails(store_copy):
    """The daily job never writes the backtest's inference: a change there that
    moves a figure is a commit that must regenerate the snapshot."""
    rel = "backtest/inference/summary.json"
    doc = json.loads((store_copy / rel).read_text())
    doc["bootstrap"]["portfolio"]["sharpe"]["point"] += 1.0
    (store_copy / rel).write_text(json.dumps(doc))
    assert not any(f"data_store/{rel}".startswith(p) for p in rr.bot_written())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rr.StaleSnapshot)
        failures = rr.check(SNAP, rr.Store(store_copy), topics=["result_3"])
    assert [f.split(" ")[0] for f in failures] == ["result_3/sharpe"]
    assert f"data_store/{rel} changed, which the daily job never writes" in failures[0]
    assert rr.COMMAND in failures[0]


def test_an_artifact_the_daily_job_rebuilt_warns_and_passes(store_copy):
    """The report_version backfill rebuilds the newest session's surface from
    its Cboe book: its figures move and its chain is no longer a Yahoo
    download. The test warns, naming the rebuilt artifacts and the command,
    and passes; the snapshot cannot be regenerated until the Yahoo sentence
    is reworded."""
    rebuilt = {}
    for name in ("scorecard.json", "surface.json"):
        rel = f"surface/{LAST}/{name}"
        doc = json.loads((store_copy / rel).read_text())
        was = SNAP["artifacts"][f"data_store/{rel}"]["report_version"]
        doc["report_version"] = max(was, doc.get("report_version") or 1) + 1
        rebuilt[f"data_store/{rel}"] = (was, doc["report_version"])
        if name == "scorecard.json":
            svi = next(m for m in doc["models"] if m["model"] == "svi_cal")
            svi["metrics"]["out_of_sample"]["median_abs_hs"] += 0.5
        else:
            doc["inputs"]["source"] = "cboe"
        (store_copy / rel).write_text(json.dumps(doc))
    store = rr.Store(store_copy)
    with pytest.warns(rr.StaleSnapshot) as caught:
        failures = rr.check(SNAP, store, topics=["result_1", "limitations", "pack"])
    assert failures == []
    message = "\n".join(str(w.message) for w in caught if issubclass(w.category, rr.StaleSnapshot))
    for key, (was, now) in rebuilt.items():
        assert f"{key} (report_version {was} → {now})" in message
    for cid in ("result_1/last", "limitations/quote_quality", "pack/m14_certified"):
        assert f"  {cid} " in message, cid
    assert rr.COMMAND in message
    _, bad = rr.build(SNAP["params"], rr.Store(store_copy), date="2000-01-01", commit="0" * 40)
    assert "limitations/quote_quality" in [k.id for k in bad]


def test_a_session_without_a_replicated_vix_is_named(store_copy):
    """A session the daily job adds can have no replicated VIX (2026-10-27:
    no candidate expiries on both sides of 30 days). A span over it stops the
    regeneration on the two sentences that list or range over the replicated
    values and names the session, instead of a TypeError on a missing number;
    checked against the snapshot, the rebuilt row is a warning."""
    rel = "surface/index.json"
    doc = json.loads((store_copy / rel).read_text())
    row = next(r for r in doc["sessions"] if r["session"] == LAST)
    row["vix_replicated"] = None
    (store_copy / rel).write_text(json.dumps(doc))
    _, bad = rr.build(SNAP["params"], rr.Store(store_copy), date="2000-01-01", commit="0" * 40)
    errors = {k.id: k.error for k in bad}            # other claims may refuse after a rebuild too
    for cid in ("calibration/vix_gap", "pack/m14_vix"):
        assert errors.get(cid) == f"the VIX is not replicated on ['{LAST}']", cid
    with pytest.warns(rr.StaleSnapshot, match=f"data_store/{rel}#session={LAST}"):
        assert rr.check(SNAP, rr.Store(store_copy), topics=["calibration", "pack"]) == []


def test_the_daily_job_writes_the_surface_and_fit_artifacts():
    """bot_written() reads the paths from the workflow's commit steps (a
    workflow change that hid them would turn a routine rebuild into a failure)
    and adds the cache every run rewrites without committing it."""
    bot = rr.bot_written()
    for key in SNAP["artifacts"]:
        if key.startswith(("data_store/surface", "data_store/calibrations", "data_store/market",
                           "data_store/options", "data_store/forward", "data_store/vol_system.db")):
            assert rr._under(key, bot), key
        if key.startswith(("data_store/backtest", "data_store/benchmarks")):
            assert not rr._under(key, bot), key


@pytest.mark.parametrize("cache", [True, False], ids=["the job's tree", "a clean checkout of its commit"])
def test_a_vx_session_the_daily_job_stores_warns_and_passes(store_copy, tmp_path, monkeypatch, cache):
    """FUND_GRADE_PLAN 1.12: the daily job stores each session's VX
    settlements in its cache (vol_system.db, rewritten by every run and never
    committed) and commits them as data_store/market/vix_futures_daily/.
    Counted from the cache alone, the Data table's VX row would fail in the
    tree the job leaves (the suite simulate_next_session.py runs) as a change
    to a file the job never writes, and a clean checkout of the job's commit
    would go on quoting the cache's rows without a word. It counts the record
    (the cache's rows and the partitions, as app.record_db merges them), so
    both trees warn, naming the figure that moved, and pass."""
    from joint_vol_calibration.data import database as dbm

    through = SNAP["params"]["data_through"]
    record = f"data_store/market/vix_futures_daily#date<={through}"
    assert record in _claim("data_table/vx_futures")["sources"]   # read while empty: the first one moves it
    # whatever the record holds already (the job may have stored sessions since
    # the snapshot): one contract listed after its newest, on its last session
    before = rr.Claims(rr.Store(store_copy), SNAP["params"]).vx_futures(through)
    month = str(pd.Period(max(m for _, m in before), "M") + 1)
    row = pd.DataFrame({"date": [max(d for d, _ in before)], "expiry_date": [f"{month}-15"],
                        "expiry_month": [month], "is_weekly": [0], "settle": [20.0]})
    job = tmp_path / "job.db"
    shutil.copy2(store_copy / "vol_system.db", job)
    monkeypatch.setattr(dbm, "DB_PATH", job)
    assert dbm.insert_vix_futures_daily(row) == 1                      # the market step
    assert dbm.export_market_partitions(store_copy / "market", tables=["vix_futures_daily"])  # the commit
    if cache:
        shutil.copy2(job, store_copy / "vol_system.db")
    with pytest.warns(rr.StaleSnapshot) as caught:
        failures = rr.check(SNAP, rr.Store(store_copy), topics=["data_table"])
    assert failures == []
    message = "\n".join(str(w.message) for w in caught if issubclass(w.category, rr.StaleSnapshot))
    cache_key = f"data_store/vol_system.db#vix_futures_daily dated on or before {through}"
    assert f"  {record}\n" in message                    # the job's commit, listed with no note
    assert f"  {cache_key} (" not in message              # and its cache, when changed, the same way
    if cache:
        assert f"  {cache_key}\n" in message
    assert "  data_table/vx_futures now reads " in message
    now = next(k for k in rr.compute(SNAP["params"], rr.Store(store_copy), ["data_table"])
               if k.id == "data_table/vx_futures")
    assert now.text.endswith(f"| {len(before) + 1:,} |")
    assert f" → {month}) | " in now.text


def test_regeneration_rewrites_only_the_figures_that_moved(tmp_path):
    """make readme-results rewrites a moved figure in the part of the document
    its claim names (keeping the line breaks) and the 'as of' dates."""
    for doc in (rr.README, rr.PACK, rr.SOURCES_DOC):
        (tmp_path / doc).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / doc, tmp_path / doc)
    old_last = _claim("result_1/last")["text"]
    # the same words outside the first screen are not the claim's
    with open(tmp_path / rr.README, "a") as fh:
        fh.write(f"\n## Elsewhere\n\n{old_last}\n")
    new = copy.deepcopy(SNAP)
    new["generated"]["date"] = "2099-12-31"
    for k in new["claims"]:
        if k["id"] in ("result_1/last", "pack/m14_certified"):
            k["text"] = _moved(k["text"])
    done = rr.update_docs(SNAP, new, root=tmp_path)
    assert len(done) == 4, done
    assert rr.doc_problems(new, lambda doc: (tmp_path / doc).read_text()) == []
    text = (tmp_path / rr.README).read_text()
    assert text.endswith(f"\n## Elsewhere\n\n{old_last}\n")
    before, after = README_RAW.splitlines(), text.splitlines()[:len(README_RAW.splitlines())]
    assert len(before) == len(after)                     # line breaks kept
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert 1 <= len(changed) <= 4, changed


# ── what the snapshot does not hold ───────────────────────────────────────────

def test_first_screen_is_short_and_names_its_parts():
    body = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", FIRST_SCREEN_RAW)
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    words = re.findall(r"[A-Za-z0-9][\w.,'’%·−/-]*", body)
    assert len(words) <= 340, len(words)
    for part in ("**What it is, and is not.**", "A model-validation record", "It is not an alpha claim",
                 "the backtest result is negative and no signal is deployable", "**Three results**",
                 "**Limitations.**", "**Links.**",
                 f"docs/figures/smile_{FIGURE}.png",
                 "https://navnoorbawa.me", "docs/model_validation/README.md", "docs/reproduce.md",
                 "docs/corrections.md", "docs/free_resources_scorecard.md", "CITATION.cff",
                 "docs/install.md"):
        assert part in FIRST_SCREEN, part
    # research prototypes are not on the first screen
    for word in ("classifier", "XGBoost", "neural", "NN "):
        assert word not in FIRST_SCREEN, word


def test_regeneration_command_reproduces_the_committed_fits():
    """CLM-07: the README's command for joint_cal_<session>.pkl was a new TRF
    fit on the Treasury curve, not the committed DE fit on ^IRX. It is now the
    refit make reproduce runs: the recorded inputs and the optimiser that made it
    (DE before C23, multi-start TRF from 2026-10-26)."""
    line = next(x for x in README_RAW.splitlines() if x.startswith("| `data_store/calibrations/joint_cal_"))
    assert "**recorded_inputs(ref)" in line
    choose = "'de' if ref.get('optimizer', 'de') == 'de' else 'trf'"
    assert f"c.calibrate(optimizer={choose})" in line
    refit = (ROOT / "scripts" / "ci" / "reproduce.py").read_text()
    assert "**recorded_inputs(ref)" in refit
    assert f"opt = {choose}" in refit
    for s in C23_SESSIONS:
        assert _cal(s).get("optimizer", "de") == "de", s      # every fit C23 measured is DE
    assert {_cal(s).get("optimizer", "de") for s in RESULT_SESSIONS} <= {"de", "trf_multistart"}


def test_validation_pack_keeps_monitoring_status_and_rule_claims_out():
    """VALIDATOR-10 and T3: the pack copies no monitoring status the daily job
    changes, and says 'the Cboe formula, with its listed departures', never
    that the replication follows Cboe's rules."""
    mon = PACK.split("## 4. Ongoing monitoring")[1].split("## 5.")[0]
    assert not re.search(r"\b\d+ of the \d+ checks breach", mon)
    assert "(Cboe rules)" not in PACK
    assert "with Cboe's rules" not in PACK


def test_figure_is_drawn_from_the_surface_report():
    mrf = _script("make_readme_figure")
    side = json.loads((ROOT / "docs" / "figures" / f"smile_{FIGURE}.json").read_text())
    src = ROOT / side["source"]
    assert side["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    res = pd.read_parquet(src)
    want = mrf.figure_numbers(res, FIGURE, mrf.pick_expiries(res))
    assert {k: side[k] for k in want} == want
    png = (ROOT / "docs" / "figures" / f"smile_{FIGURE}.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert f"![SPX smiles on {FIGURE}:" in FIRST_SCREEN       # the caption names its session
    assert f"](docs/figures/smile_{FIGURE}.png)" in FIRST_SCREEN


def test_corrections_doc_is_the_render_of_the_registry():
    rc = _script("render_corrections")
    doc = (ROOT / "docs" / "corrections.md").read_text()
    assert rc.splice(doc, rc.render()) == doc, "run python scripts/render_corrections.py"
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    for r in reg["rounds"]:
        assert f"### {r['id']}" in doc
        for c in r["commits"]:
            assert f"/commit/{c})" in doc


def test_validation_pack_findings_log_is_the_render_of_the_registry():
    """CLM-14: the pack's findings log named docs/corrections.md as its source
    yet disagreed with it (C17's severity, C20's dates, four C23 findings and
    six commits missing). It is now rendered from the same registry."""
    rc = _script("render_corrections")
    pack = (ROOT / "docs" / "model_validation" / "README.md").read_text()
    assert rc.LOG_BEGIN in pack, "the pack's findings log is not the generated block"
    rendered = rc.splice(pack, rc.render_findings_log(), rc.LOG_BEGIN)
    assert rendered == pack, "run python scripts/render_corrections.py"
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    log = pack[pack.index(rc.LOG_BEGIN):]
    doc = (ROOT / "docs" / "corrections.md").read_text()
    for r in reg["rounds"]:
        ids = [f["id"] for f in r["findings"] if f.get("id")] or [r["id"]]
        for i in ids:
            assert f"| {i} |" in log, i
            if i != r["id"]:
                assert f"[{i}]" in doc, i                  # the same ID in the postmortem
        for c in r["commits"]:
            assert f"`{c[:7]}`" in log, (r["id"], c)       # every commit of the round is in the log
        for f in r["findings"]:
            assert set(f.get("commits") or []) <= set(r["commits"]), (r["id"], f.get("id"))
    for line in log.splitlines():
        for sha in re.findall(r"`([0-9a-f]{7})`", line):
            assert f"/commit/{sha}" in doc, sha            # every commit in the log is in corrections.md


def test_relative_links_resolve():
    docs = [ROOT / "README.md", ROOT / "docs" / "corrections.md",
            ROOT / "docs" / "free_resources_scorecard.md",
            ROOT / "docs" / "model_validation" / "README.md", ROOT / "docs" / "README.md",
            ROOT / "substack_article.md"]
    for doc in docs:
        text = doc.read_text()
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            assert path.exists(), (doc.name, target)


def test_badges_are_live_and_carry_no_hand_typed_counts():
    badges = re.findall(r"\[!\[[^\]]*\]\(([^)]+)\)\]", FIRST_SCREEN_RAW)
    assert badges
    for url in badges:
        assert not re.search(r"\d+(%25|%)?[-_ ](passed|tests|coverage)", url), url
        m = re.search(r"actions/workflows/([\w.-]+)/badge\.svg", url)
        if m:
            assert (ROOT / ".github" / "workflows" / m.group(1)).exists(), url
    assert not re.search(r"\b\d[\d,]* (tests )?passed\b", FIRST_SCREEN)


def test_quantlib_is_a_dev_only_test_oracle():
    assert "QuantLib is used only as a dev-only test oracle" in README.replace("**", "")
    runtime = (ROOT / "requirements.txt").read_text().lower()
    assert "quantlib" not in runtime
    assert "quantlib" in (ROOT / "requirements-dev.txt").read_text().lower()
    for base in ("joint_vol_calibration", "dashboard", "scripts"):
        for py in (ROOT / base).rglob("*.py"):
            if "tests" in py.parts or "bench" in py.parts:
                continue
            assert not re.search(r"^\s*(import QuantLib|from QuantLib)", py.read_text(), re.MULTILINE), py


def test_article_c23_note_carries_the_canonical_numbers(inference):
    """substack_article.md opens with a dated C23 note; the numbers C23 moved are the artifacts'."""
    # the notes are block quotes: drop the '>' that starts each line before joining
    art = " ".join(re.sub(r"(?m)^>[ ]?", "", (ROOT / "substack_article.md").read_text()).split())
    note = art.split("**C23 note")[1].split("**Correction (October 2026, C22).**")[0]
    with open(DATA / "backtest" / "full_results.pkl", "rb") as fh:
        m = pickle.load(fh)["metrics"]
    nav = 1_000_000 * (1 + m["cumulative_return"])
    cum, sharpe = fmt(m["cumulative_return"] * 100, ".1f"), fmt(m["sharpe"], ".2f")
    head = f"**{cum}% (NAV ${nav:,.0f}), Sharpe {sharpe}**"
    assert head in note
    b = inference["bootstrap"]["portfolio"]["sharpe"]["ci95"]
    assert f"{fmt(b[0], '.2f')} to {fmt(b[1], '.2f')}" in note
    for key in ("s1", "s1c", "s3"):
        k = f"{abs(m[key + '_total_pnl']) / 1000:,.0f}K"
        assert k in note, key
    # CLM-11: the note says every number in it is checked by the tests; these are the rest
    spa = inference["multiple_testing"]["spa_published"]["p_consistent"]
    assert f"Hansen SPA p {spa:.2f}" in note
    s3 = inference["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
    assert (f"Against {s3['n_paths']:,} exposure-matched circular shifts, S3's entry timing has p "
            f"{s3['p_value']:.3f} ({s3['p_holm_6_sleeves']:.2f} after a Holm correction over the six "
            "series)") in note
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    c22 = next(pt for pt in reg["headline_series"]["points"] if pt["round"] == "C22")
    assert f"against {fmt(c22['cum_ret_pct'], '.1f')}% and −0.88 in the C22 note below" in note
    assert "Sharpe −1.80 → −0.88" in next(r for r in reg["rounds"] if r["id"] == "C22")["effect"]
    fam = json.loads((DATA / "market" / "corrections" / "c23_vix_index_family.json").read_text())
    vvix = fam["tables"]["vix_term_structure"]["by_ticker"]["^VVIX"]["added_from_cboe"]
    assert f"{vvix} VVIX sessions missing" in note
    rows = {r["session"]: r for r in json.loads(
        (DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]}
    diag = [rows[s] for s in C23_SESSIONS]
    sc = {s: {m["model"]: m for m in json.loads((DATA / "surface" / s / "scorecard.json").read_text())
              ["models"]} for s in C23_SESSIONS}
    hes = rng((sc[s]["heston_v2"]["metrics"]["out_of_sample"]["median_abs_hs"] for s in C23_SESSIONS), ".1f")
    svi = rng((sc[s]["svi_cal"]["metrics"]["out_of_sample"]["median_abs_hs"] for s in C23_SESSIONS), ".2f")
    assert f"misses by a median {hes} half-spreads" in note
    assert f"arbitrage-free SVI surface on the same sessions misses by {svi}" in note
    cur = rows[C23_HEADLINE]
    assert (f"cuts its out-of-sample error from {cur['sigma_cap2_oos']:.2f} to {cur['sigma_free_oos']:.2f} "
            f"vol pts, at σ = {cur['sigma_free']:.1f}") in note
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in diag)
    same = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)
    assert (f"lower on {lower} of {len(diag)} sessions and identical to Heston on the other {same}") in note
    art_full = " ".join((ROOT / "substack_article.md").read_text().split())
    worst = max(abs(r["trf_loss_rel_diff"]) for r in diag)
    assert f"whose loss it matches to within {worst:.1e} relative".replace("e-0", "e-") in art_full


def test_citation_is_this_version():
    """CLM-12: CITATION.cff cited 0.22.0 'released' 2026-10-19 while its
    abstract described C23 work committed after that date. The cited version is
    the package's, released no earlier than the last correction round it
    describes, and the title is the README's."""
    cff = (ROOT / "CITATION.cff").read_text()
    version = re.search(r"^version: (\S+)$", cff, re.MULTILINE).group(1)
    released = re.search(r"^date-released: (\d{4}-\d{2}-\d{2})$", cff, re.MULTILINE).group(1)
    toml = (ROOT / "pyproject.toml").read_text()
    pyproject = re.search(r'^version = "([^"]+)"$', toml, re.MULTILINE).group(1)
    assert version == pyproject
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    last = max(r.get("date_end") or r["date"] for r in reg["rounds"])
    assert released >= last, (released, last)
    title = re.search(r'^title: "([^"]+)"$', cff, re.MULTILINE).group(1)
    assert README_RAW.splitlines()[0] == f"# {title}"
    assert "no release tag" in " ".join(cff.split())       # nothing claims a tag or DOI that does not exist
    assert "doi" not in cff.lower()


def test_ui_plan_marks_its_withdrawn_acceptance_text_as_history():
    """HANDOFF D: docs/roadmap/ui_plan.json still read 'the tile shows 95.0' as
    P2.5's acceptance after R3-T4 withdrew that number (a single bad Yahoo quote,
    not executable arbitrage). The plan is history: a superseded acceptance line
    is marked, its item says why in a history_note, and the withdrawal it cites
    is in the corrections register."""
    plan = json.loads((ROOT / "docs" / "roadmap" / "ui_plan.json").read_text())
    items = {i["id"]: i for pk in plan["packages"] for i in pk["items"]}
    p25 = items["P2.5"]
    line = next(x for x in p25["acceptance"].splitlines() if "95.0" in x)
    assert "[Superseded, see history_note]" in line
    assert "3da6928" in p25["history_note"]
    for item in items.values():
        if "[Superseded" in item.get("acceptance", ""):
            assert item.get("history_note"), item["id"]
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    shas = [c for r in reg["rounds"] for c in r["commits"]]
    assert any(c.startswith("3da6928") for c in shas)
