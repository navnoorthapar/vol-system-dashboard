"""
Regression guards for the deployment configuration (C22 ops findings).

Text checks on the workflow files and scripts, so they run without PyYAML or
network: each one pins a property whose absence once broke or misreported the
published site.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DAILY = (ROOT / ".github" / "workflows" / "daily_refresh.yml").read_text()
TESTS = (ROOT / ".github" / "workflows" / "tests.yml").read_text()


def _crons(text):
    return re.findall(r'-\s*cron:\s*"([^"]+)"', text)


def test_daily_cron_is_split_by_data_source():
    crons = _crons(DAILY)
    book = [c for c in crons if c.split()[4] == "2-6"]               # live Cboe book + retry
    assert len(book) == 2
    # Global Trading Hours open at 01:15 UTC under EST and quotes reach Yahoo ~15 min
    # late: a book slot must start at least an hour after that open.
    assert all(int(c.split()[1]) * 60 + int(c.split()[0]) >= 2 * 60 + 15 for c in book)
    close = [c for c in crons if c.split()[1] == "22"]
    assert close and close[0].split()[0] != "0"                    # off the top of the hour
    assert '"market,labels"' in DAILY and "market,capture,calibrate,freeze" in DAILY


def test_daily_job_persists_captures_and_rebases_before_push():
    assert "data_store/market" in DAILY and "data_store/options" in DAILY
    assert "pull --rebase" in DAILY
    assert re.search(r"concurrency:\s*\n\s*group: daily-refresh", DAILY)


def test_deploy_messages_are_not_literal_shell():
    # peaceiris does not shell-expand: the old message published "$(date -u ...)"
    assert "$(date" not in DAILY
    assert "steps.refresh.outputs.session" in DAILY


def test_daily_job_tests_before_publishing_and_uses_the_lock():
    assert DAILY.index("pytest") < DAILY.index("Commit data files") < DAILY.index("gh-pages@")
    for wf in (DAILY, TESTS):
        assert "requirements-lock.txt" in wf
        assert "pip install -r requirements.txt" not in wf
        assert "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" not in wf


def test_actions_are_pinned_to_commit_shas():
    for wf in (DAILY, TESTS):
        uses = re.findall(r"uses:\s*(\S+)", wf)
        assert uses
        for u in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", u), u


def test_no_hand_typed_test_counts_in_ci():
    assert not re.search(r"\d+ passed", TESTS + DAILY)


def test_lock_pins_every_package_and_uses_cpu_torch():
    lines = [ln.strip() for ln in (ROOT / "requirements-lock.txt").read_text().splitlines()
             if ln.strip() and not ln.startswith(("#", "--"))]
    assert lines and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==\S+", ln) for ln in lines), lines
    names = {ln.split("==")[0].lower() for ln in lines}
    assert not any(n.startswith(("nvidia-", "triton")) for n in names)
    assert any(ln.startswith("torch==") and ln.endswith("+cpu") for ln in lines)


def test_lock_keeps_the_security_floors():
    """C23-E2: pip-audit found advisories in idna 3.11, urllib3 2.6.3 and
    werkzeug 3.1.8; requirements.txt floors them and the lock must honour the
    floors (a recompile that drops one fails here)."""
    from packaging.version import Version

    floors = {"idna": "3.15", "urllib3": "2.8.0", "werkzeug": "3.1.9"}
    req = (ROOT / "requirements.txt").read_text()
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==(\S+)$", (ROOT / "requirements-lock.txt").read_text(),
                           re.MULTILINE))
    pins = {k.lower(): v for k, v in pins.items()}
    for name, floor in floors.items():
        assert re.search(rf"^{name}>={re.escape(floor)}$", req, re.MULTILINE), name
        assert Version(pins[name]) >= Version(floor), (name, pins[name])


def test_redeploy_script_cannot_push_gh_pages():
    sh = (ROOT / "redeploy_site.sh").read_text()
    code = "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))
    assert "git push" not in code and "git init" not in code
    assert "gh workflow run" in code


def test_dead_procfile_is_gone():
    assert not (ROOT / "Procfile").exists()
    assert "gunicorn" not in (ROOT / "requirements.txt").read_text()


def _step(text, name):
    """The YAML block of the workflow step called ``name`` (up to the next step)."""
    m = re.search(rf"- name: {re.escape(name)}\n(.*?)(?=\n\s*- name: |\Z)", text, re.S)
    assert m, name
    return m.group(1)


def test_a_stale_calibration_does_not_block_the_deploy():
    # cloud_refresh exits 1 when the newest accepted fit is too old, which made
    # "steps.refresh.outcome == 'success'" skip the deploy: the site (market data,
    # labels, staleness banner) froze until a calibration passed. The deploy keys
    # on the publishable output instead, and needs always() to run after a red step.
    deploy = _step(DAILY, "Deploy to GitHub Pages")
    cond = re.search(r"if:\s*>?\s*(.*?)\n\s*uses:", deploy, re.S).group(1)
    assert "always()" in cond
    assert "steps.refresh.outputs.publishable == 'true'" in cond
    assert "steps.refresh.outcome" not in cond
    assert "steps.tests.outcome == 'success'" in cond


def test_a_presentation_test_cannot_lose_the_captures():
    # On 2026-10-20 one combined test gate decided both the data commit and the
    # deploy. Two dashboard copy tests failed on a legitimately new fit, so the
    # accepted overnight book of 2026-10-19 (not re-fetchable) was never committed.
    # The commit now keys only on the data gate; the dashboard tests gate the deploy.
    # C23 adds a third gate: the artifacts derived from a new fit (surface report,
    # diagnostics, Greeks surface, scenario grid) are a second commit behind their
    # own tests, so a failed artifact check cannot hold back the captures either.
    data = _step(DAILY, "Data tests (gate the capture commit)")
    art = _step(DAILY, "Artifact tests (gate the artifact commit)")
    pres = _step(DAILY, "Presentation tests (gate the deploy)")
    assert "id: datatests" in data
    assert "id: artifacttests" in art
    assert "id: tests" in pres
    assert "test_dashboard.py" not in data + art
    assert "test_dashboard.py" in pres
    assert "test_session_artifacts.py" in art
    assert "test_session_artifacts.py" not in data
    for f in ("test_data_layer.py", "test_option_chain.py", "test_cloud_refresh.py", "test_lookahead.py",
              "test_cboe_downloader.py", "test_forward_ledger.py"):
        assert f in data, f
    commit = _step(DAILY, "Commit data files")
    commit_if = re.search(r"if:\s*(.*)", commit).group(1)
    assert "steps.datatests.outcome == 'success'" in commit_if
    assert "steps.tests.outcome" not in commit_if and "artifacttests" not in commit_if
    # PROD-1: the captures are committed before any test gate runs, behind
    # integrity checks only, and leave the runner as an artifact before that
    caps = _step(DAILY, "Commit captures")
    caps_if = re.search(r"if:\s*(.*)", caps).group(1)
    assert "outcome == 'success'" not in caps_if
    assert "datatests" not in caps_if
    assert "check_captures.py" in _step(DAILY, "Check the captures (integrity only)")
    assert DAILY.index("- name: Commit captures") < DAILY.index("- name: Data tests")
    arts = _step(DAILY, "Commit derived artifacts")
    arts_if = re.search(r"if:\s*>?\s*(.*?)\n\s*env:", arts, re.S).group(1)
    assert "steps.artifacttests.outcome == 'success'" in arts_if
    assert "steps.commit.outcome == 'success'" in arts_if        # after the captures, never instead
    assert "steps.tests.outcome" not in arts_if
    deploy = _step(DAILY, "Deploy to GitHub Pages")
    cond = re.search(r"if:\s*>?\s*(.*?)\n\s*uses:", deploy, re.S).group(1)
    assert "steps.datatests.outcome == 'success'" in cond and "steps.tests.outcome == 'success'" in cond
    assert "steps.commitartifacts.outcome == 'success'" in cond


def test_each_commit_adds_its_own_paths():
    """The capture commit holds the record (incl. the Cboe books under
    data_store/options/cboe and the forward ledger), the artifact commit the
    derived files; the diagnostics live under data_store/calibrations, so the
    capture commit excludes them explicitly."""
    commit = _step(DAILY, "Commit data files")
    for p in ("data_store/signals/regime_labels.parquet", "data_store/market", "data_store/options",
              "data_store/calibrations", "data_store/forward"):
        assert p in commit, p
    assert "':(exclude)data_store/calibrations/diagnostics'" in commit
    for p in ("data_store/surface", "data_store/greeks"):
        assert p not in commit, p
    arts = _step(DAILY, "Commit derived artifacts")
    for p in ("data_store/surface", "data_store/calibrations/diagnostics", "data_store/greeks"):
        assert p in arts, p
    assert "vol_system.db" not in commit + arts


def test_push_retry_rebase_survives_the_unstaged_database():
    # data_store/vol_system.db is tracked, every run modifies it and the bot never
    # stages it. A plain "git pull --rebase" then refuses ("You have unstaged
    # changes"), "rebase --abort" finds no rebase, and the step exited 1 -- losing
    # the run's captures -- whenever main moved during the run.
    push = (ROOT / "scripts" / "ci" / "commit_and_push.sh").read_text()
    code = "\n".join(ln for ln in push.splitlines() if not ln.lstrip().startswith("#"))
    for name in ("Commit captures", "Commit data files", "Commit derived artifacts"):
        assert "scripts/ci/commit_and_push.sh" in _step(DAILY, name), name
    assert (ROOT / "data_store" / "vol_system.db").exists()
    rebases = re.findall(r"git rebase [^\n]*origin/main[^\n]*", code)
    assert rebases, rebases
    assert all("--autostash" in r for r in rebases), rebases
    assert "pull --rebase" not in code          # a failed pull's fetch ended the loop (PROD-2)
    # and the comment no longer claims the database is not committed/tracked
    assert "rebuildable cache and is not committed" not in DAILY
    assert "git-tracked" in DAILY


def test_commit_and_push_against_a_moved_remote(tmp_path):
    """scripts/ci/commit_and_push.sh on a scratch repository with a local
    remote: it commits only the paths given (a missing one skipped, the
    exclusion honoured, the tracked database never), rebases onto a main that
    moved during the run with the database still modified, and with nothing to
    add commits nothing."""
    import os
    import shutil
    import subprocess
    if not (shutil.which("git") and shutil.which("base64") and shutil.which("bash")):
        import pytest
        pytest.skip("git/base64/bash not available")
    script = ROOT / "scripts" / "ci" / "commit_and_push.sh"
    env = {**os.environ, "GH_TOKEN": "x", "REPO": "o/r", "PUSH_RETRY_SLEEP": "0",
           "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("GITHUB_OUTPUT", None)

    def git(*a, cwd):
        return subprocess.run(["git", *a], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout

    (tmp_path / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@example.com\n"
                                        "[init]\n\tdefaultBranch = main\n")
    remote, work, other = tmp_path / "remote.git", tmp_path / "work", tmp_path / "other"
    git("init", "--bare", str(remote), cwd=tmp_path)
    git("clone", str(remote), str(work), cwd=tmp_path)
    ds = work / "data_store"
    for rel, text in (("vol_system.db", "db0"), ("market/a.csv", "a0"), ("calibrations/c0.pkl", "c0")):
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_text(text)
    git("add", "-A", cwd=work)
    git("commit", "-qm", "init", cwd=work)
    git("push", "-q", "origin", "HEAD:main", cwd=work)
    git("clone", str(remote), str(other), cwd=tmp_path)
    (other / "README").write_text("moved")
    git("add", "README", cwd=other)
    git("commit", "-qm", "main moved", cwd=other)
    git("push", "-q", "origin", "HEAD:main", cwd=other)

    (ds / "vol_system.db").write_text("db1")                      # tracked, never committed
    (ds / "market" / "a.csv").write_text("a1")
    (ds / "calibrations" / "c1.pkl").write_text("c1")
    (ds / "calibrations" / "diagnostics").mkdir()
    (ds / "calibrations" / "diagnostics" / "d.json").write_text("{}")
    r = subprocess.run(["bash", str(script), "title", "body", "data_store/market", "data_store/missing",
                        "data_store/calibrations", ":(exclude)data_store/calibrations/diagnostics"],
                       cwd=work, env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "fetching origin/main to rebase onto it" in r.stdout
    log = git("log", "--format=%s", "origin/main", cwd=work).split("\n")
    assert log[:3] == ["title", "main moved", "init"]
    files = git("show", "--name-only", "--format=", "origin/main", cwd=work).split()
    assert sorted(files) == ["data_store/calibrations/c1.pkl", "data_store/market/a.csv"]
    status = git("status", "--porcelain", cwd=work)
    assert " M data_store/vol_system.db" in status                 # autostash restored it
    assert "?? data_store/calibrations/diagnostics/" in status

    head = git("rev-parse", "origin/main", cwd=work)
    r = subprocess.run(["bash", str(script), "t2", "b2", ":(exclude)data_store/calibrations/diagnostics"],
                       cwd=work, env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert "No changes to commit." in r.stdout
    assert git("rev-parse", "origin/main", cwd=work) == head
    assert git("diff", "--staged", "--name-only", cwd=work) == ""


def test_reproduce_refits_each_fit_from_its_committed_record(monkeypatch):
    """make reproduce deletes each artifact in its scratch copy before the step
    that rebuilds it, so a refit must read the fit's recorded inputs from the
    committed pickle in the checkout (it read the copy's, already deleted: all
    five calibration steps failed). And it must use the optimiser the fit was
    made with: the calibrator's default became multi-start TRF in C23, which
    lands on the committed DE optima to about 1e-7 in loss but not to 1e-6 in
    every output."""
    import importlib.util
    import pickle
    spec = importlib.util.spec_from_file_location("reproduce_under_test",
                                                  ROOT / "scripts" / "ci" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, mod)             # its dataclasses look it up
    spec.loader.exec_module(mod)
    # the refits only: this test is in the data gate, which must not depend on a
    # derived artifact the bot writes (the Greeks surface, the scenario grid)
    steps = {s.name: s for s in mod.build_steps(ROOT, parts=("calibration",))}
    refits = [n for n in steps if n.startswith("calibration-")]
    assert refits
    assert set(steps) == set(refits)
    from joint_vol_calibration.calibration import joint_calibrator as jc
    seen = []

    class FakeCalibrator:
        def __init__(self, **kw):
            seen.append(("init", kw))

        def calibrate(self, **kw):
            seen.append(("calibrate", kw))

        def save(self):
            seen.append(("save", {}))
    monkeypatch.setattr(jc, "JointCalibrator", FakeCalibrator)
    for name in refits:
        session = name.removeprefix("calibration-")
        code = steps[name].argv[-1]
        committed = ROOT / "data_store" / "calibrations" / f"joint_cal_{session}.pkl"
        assert repr(str(committed)) in code
        assert "DATA_DIR" not in code
        with open(committed, "rb") as f:
            ref = pickle.load(f)
        seen.clear()
        exec(compile(code, name, "exec"), {})                     # noqa: S102 (the step's own code)
        (_, init), (_, cal), _ = seen
        assert init == {"as_of_date": session, **jc.recorded_inputs(ref)}
        assert cal["optimizer"] == ("de" if ref.get("optimizer", "de") == "de" else "trf")


def test_reproduce_steps_do_not_read_derived_artifacts_for_the_refits(tmp_path, monkeypatch):
    """PROD-3: build_steps read the provenance of the bot-written Greeks surface and
    scenario grid unconditionally, so a surface written without b'greeks_provenance'
    (or missing, or unreadable) raised KeyError in the data gate's refit test above
    and blocked the capture commit. The refit steps now read nothing derived, and a
    full build reports such an artifact as a step that cannot run instead of raising."""
    import importlib.util
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq
    spec = importlib.util.spec_from_file_location("reproduce_derived_under_test",
                                                  ROOT / "scripts" / "ci" / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    root = tmp_path / "repo"
    (root / "data_store" / "greeks").mkdir(parents=True)
    (root / "data_store" / "calibrations").symlink_to(ROOT / "data_store" / "calibrations")
    greeks = root / "data_store" / "greeks" / "greeks_surface.parquet"
    pq.write_table(pa.table({"x": [1.0]}), greeks)                    # no provenance key
    (root / "data_store" / "greeks" / "scenario_grid.parquet").write_bytes(b"not a parquet file")

    refits = mod.build_steps(root, parts=("calibration",))
    assert refits
    assert all(s.name.startswith("calibration-") and not s.error for s in refits)
    steps = {s.name: s for s in mod.build_steps(root)}
    assert "greeks_provenance" in steps["greeks"].error
    assert steps["greeks"].argv == []
    assert steps["scenario"].error
    assert steps["scenario"].argv == []
    assert not steps["calibration-" + refits[0].name.removeprefix("calibration-")].error
    greeks.unlink()                                                   # a missing file too
    assert "FileNotFoundError" in {s.name: s for s in mod.build_steps(root)}["greeks"].error
    # the committed artifacts name their source: no step is flagged
    assert not [s.name for s in mod.build_steps(ROOT) if s.error]


def test_the_captures_leave_the_runner_before_any_gate():
    """PROD-1: the Cboe books, the accepted chain and pickle and the ledger lived
    only in the runner until the one data push succeeded; the only upload was the
    status JSON. Everything the run created or changed under the record's
    directories is uploaded (90 days) right after the refresh, before any gate can
    fail or time out, and a run whose captures reached neither commit says so."""
    bundle = _step(DAILY, "Bundle this run's captures")
    upload = _step(DAILY, "Upload the captures (before the gates)")
    assert "if: always() && steps.refresh.outcome != 'skipped'" in bundle
    for d in ("data_store/options", "data_store/calibrations", "data_store/forward", "data_store/market",
              "data_store/signals"):
        assert d in bundle, d
    assert "--others --modified" in bundle                     # new and changed files, not history
    assert "always()" in upload
    assert "retention-days: 90" in upload
    assert "runner.temp }}/captures" in upload
    first_gate = DAILY.index("- name: Data tests")
    assert DAILY.index("- name: Run cloud_refresh.py") < DAILY.index("- name: Bundle this run's captures") \
        < DAILY.index("- name: Upload the captures") < first_gate
    report = _step(DAILY, "Report captures that were not committed")
    assert "::error" in report
    assert "steps.commit.outcome != 'success'" in report
    assert "steps.commitcaptures.outcome != 'success'" in report
    commit = _step(DAILY, "Commit data files")
    assert "REDERIVE=" in commit
    assert "--rederive" in commit
    assert "steps.refresh.outputs.run_utc" in commit


def _repo(tmp_path):
    """A scratch repository with a local bare remote, as commit_and_push.sh sees CI."""
    import os
    import shutil
    import subprocess
    if not (shutil.which("git") and shutil.which("base64") and shutil.which("bash")):
        import pytest
        pytest.skip("git/base64/bash not available")
    env = {**os.environ, "GH_TOKEN": "x", "REPO": "o/r", "PUSH_RETRY_SLEEP": "0",
           "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("GITHUB_OUTPUT", None)
    (tmp_path / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@example.com\n"
                                        "[init]\n\tdefaultBranch = main\n")

    def git(*a, cwd):
        return subprocess.run(["git", *a], cwd=cwd, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()
    remote, work, other = tmp_path / "remote.git", tmp_path / "work", tmp_path / "other"
    git("init", "--bare", str(remote), cwd=tmp_path)
    git("clone", str(remote), str(work), cwd=tmp_path)
    ds = work / "data_store"
    for rel, text in (("vol_system.db", "db0"), ("market/vix_daily/vix_daily_2026.parquet", "v0\n"),
                      ("signals/regime_labels.parquet", "l0\n"), ("options/spx_old.parquet", "o0")):
        (ds / rel).parent.mkdir(parents=True, exist_ok=True)
        (ds / rel).write_text(text)
    git("add", "-A", cwd=work)
    git("commit", "-qm", "init", cwd=work)
    git("push", "-q", "origin", "HEAD:main", cwd=work)
    git("clone", str(remote), str(other), cwd=tmp_path)
    return env, git, work, other


def _main_moves(git, other, rel, text, msg="main moved"):
    (other / rel).parent.mkdir(parents=True, exist_ok=True)
    (other / rel).write_text(text)
    git("add", "-A", cwd=other)
    git("commit", "-qm", msg, cwd=other)
    git("push", "-q", "origin", "HEAD:main", cwd=other)


def _push_script(env, work, *args, extra_env=None):
    import subprocess
    return subprocess.run(["bash", str(ROOT / "scripts" / "ci" / "commit_and_push.sh"), *args],
                          cwd=work, env={**env, **(extra_env or {})}, capture_output=True, text=True,
                          check=False)


def test_a_transient_remote_outage_is_retried(tmp_path):
    """PROD-2: after a failed push the script ran `git pull --rebase`; when that
    pull's fetch failed too (a GitHub 5xx, a network blip spanning both) it exited
    1 at once ("rebase onto main failed") and the run's captures were lost. The
    fetch is now retried with a growing pause, and only a real rebase conflict is
    a conflict. A git wrapper fails the first push and the first two fetches."""
    import os
    env, git, work, other = _repo(tmp_path)
    _main_moves(git, other, "README", "moved")
    real_git = __import__("shutil").which("git")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counts = tmp_path / "counts"
    counts.mkdir()
    (bin_dir / "git").write_text(f"""#!/usr/bin/env bash
for a in "$@"; do
  case "$a" in
    push|fetch|pull)
      n=$(cat "{counts}/$a" 2>/dev/null || echo 0); echo $((n + 1)) > "{counts}/$a"
      limit=0; [ "$a" = push ] && limit=1; [ "$a" = fetch ] && limit=2; [ "$a" = pull ] && limit=1
      if [ "$n" -lt "$limit" ]; then echo "fatal: Could not read from remote repository." >&2; exit 128; fi
      break ;;
  esac
done
exec {real_git} "$@"
""")
    os.chmod(bin_dir / "git", 0o700)
    cap = work / "data_store" / "options" / "cboe" / "spx_2026-10-20_023000Z.parquet"
    cap.parent.mkdir(parents=True)
    cap.write_text("book")
    r = _push_script(env, work, "captures", "b", "data_store/options",
                     extra_env={"PATH": f"{bin_dir}:{env['PATH']}"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Fetching origin/main failed (try 1 of 4)" in r.stdout
    assert (counts / "fetch").read_text().strip() == "3"
    assert (counts / "push").read_text().strip() == "2"
    files = git("show", "--name-only", "--format=", "origin/main", cwd=work).split()
    assert files == ["data_store/options/cboe/spx_2026-10-20_023000Z.parquet"]
    assert git("log", "--format=%s", "origin/main", cwd=work).split("\n")[:2] == ["captures", "main moved"]


_PART = "data_store/market/vix_daily/vix_daily_2026.parquet"
# a stand-in for cloud_refresh.py --rederive: merge the run's row into whatever the
# partition file now holds, from the run's database (here a text file)
_REDERIVE = ("python3 -c \"import pathlib; p = pathlib.Path('" + _PART + "'); "
             "p.write_text(p.read_text() + 'run-row\\n'); "
             "pathlib.Path('rederived.log').write_text(pathlib.Path('data_store/vol_system.db').read_text())\"")


@pytest.mark.parametrize("rederive", [False, True])
def test_new_captures_land_although_main_rewrote_a_partition(tmp_path, rederive):
    """PROD-1's reproduction: main moved by a commit touching vix_daily_2026.parquet,
    which the run also rewrote (uncommitted, for the data commit). The capture
    commit holds only new files and lands; the run's partition is not left as
    --autostash's failed pop left it (main's copy): re-derived over main's with
    REDERIVE (the workflow), the run's own copy without."""
    env, git, work, other = _repo(tmp_path)
    _main_moves(git, other, _PART, "v0\nmain-row\n")
    (work / _PART).write_text("v0\nrun-row\n")
    (work / "data_store" / "vol_system.db").write_text("db-run")
    cap = work / "data_store" / "options" / "cboe" / "spx_2026-10-20_023000Z.parquet"
    cap.parent.mkdir(parents=True)
    cap.write_text("book")
    r = _push_script(env, work, "captures", "b", str(cap.relative_to(work)),
                     extra_env={"REDERIVE": _REDERIVE} if rederive else {})
    assert r.returncode == 0, r.stdout + r.stderr
    assert git("show", "--name-only", "--format=", "origin/main", cwd=work) == str(cap.relative_to(work))
    assert git("log", "--format=%s", "origin/main", cwd=work).split("\n")[:2] == ["captures", "main moved"]
    status = git("status", "--porcelain", cwd=work)
    assert "UU" not in status
    assert not git("stash", "list", cwd=work)
    part = (work / _PART).read_text()
    assert part == ("v0\nmain-row\nrun-row\n" if rederive else "v0\nrun-row\n")
    assert (work / "data_store" / "vol_system.db").read_text() == "db-run"


def test_a_failed_re_derive_fails_the_push_loudly(tmp_path):
    """CODE-01: sync_onto_main ran under `if !`, where bash ignores `set -e`, so a
    REDERIVE that crashed fell through to "Nothing left to commit" and the script
    exited 0: the data step reported success, the deploy went ahead, and the run's
    rows never reached main. A failed step now fails the script with an error."""
    env, git, work, other = _repo(tmp_path)
    _main_moves(git, other, _PART, "v0\nmain-row\n")
    ds = work / "data_store"
    (ds / "vol_system.db").write_text("db-run")
    (ds / _PART.removeprefix("data_store/")).write_text("v0\nrun-row\n")
    r = _push_script(env, work, "data", "b", "data_store/market", extra_env={"REDERIVE": "exit 7"})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "::error::" in r.stdout
    assert "Nothing left to commit" not in r.stdout
    assert git("log", "--format=%s", "origin/main", cwd=work).split("\n")[0] == "main moved"


def test_a_conflict_in_a_re_derivable_file_is_re_derived_over_main(tmp_path):
    """PROD-2: the data commit (partitions, labels) conflicted with main on a
    binary partition and the whole commit was dropped. It is now rebuilt over
    main's version by REDERIVE (cloud_refresh.py --rederive in the workflow),
    keeping the run's other files and its database, and pushed on top of main."""
    env, git, work, other = _repo(tmp_path)
    part = _PART
    _main_moves(git, other, part, "v0\nmain-row\n")
    ds = work / "data_store"
    (ds / "vol_system.db").write_text("db-run")                       # unstaged, never committed
    (ds / part.removeprefix("data_store/")).write_text("v0\nrun-row\n")
    (ds / "signals" / "regime_labels.parquet").write_text("l0\nl1\n")
    (ds / "options" / "spx_2026-10-20.parquet").write_text("chain")   # a second-chance capture
    r = _push_script(env, work, "data", "b", "data_store/market", "data_store/signals/regime_labels.parquet",
                     "data_store/options", extra_env={"REDERIVE": _REDERIVE})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Main also changed re-derivable file(s): " + part in r.stdout
    log = git("log", "--format=%s", "origin/main", cwd=work).split("\n")
    assert log[:3] == ["data", "main moved", "init"]
    assert git("show", f"origin/main:{part}", cwd=work) == "v0\nmain-row\nrun-row"
    assert git("show", "origin/main:data_store/options/spx_2026-10-20.parquet", cwd=work) == "chain"
    assert git("show", "origin/main:data_store/signals/regime_labels.parquet", cwd=work) == "l0\nl1"
    assert (work / "rederived.log").read_text() == "db-run"           # REDERIVE saw the run's database
    assert (ds / "vol_system.db").read_text() == "db-run"
    assert git("diff", "--name-only", cwd=work) == "data_store/vol_system.db"     # still unstaged
    assert git("diff", "--staged", "--name-only", cwd=work) == ""


def test_a_conflict_elsewhere_fails_loudly_and_names_the_file(tmp_path):
    env, git, work, other = _repo(tmp_path)
    _main_moves(git, other, "data_store/options/spx_old.parquet", "o-main")
    (work / "data_store" / "options" / "spx_old.parquet").write_text("o-run")
    r = _push_script(env, work, "data", "b", "data_store/options", extra_env={"REDERIVE": "true"})
    assert r.returncode == 1
    assert "main changed what this commit changes: data_store/options/spx_old.parquet" in r.stdout
    head = git("rev-parse", "origin/main", cwd=work)
    assert git("log", "-1", "--format=%s", head, cwd=work) == "main moved"
    assert (work / "data_store" / "options" / "spx_old.parquet").read_text() == "o-run"   # untouched
    # without REDERIVE even a partition conflict is an error (the derived-artifact commit)
    (tmp_path / "b").mkdir()
    env2, git2, work2, other2 = _repo(tmp_path / "b")
    part = "data_store/market/vix_daily/vix_daily_2026.parquet"
    _main_moves(git2, other2, part, "v-main\n")
    (work2 / part).write_text("v-run\n")
    r = _push_script(env2, work2, "data", "b", "data_store/market")
    assert r.returncode == 1
    assert "REDERIVE unset" in r.stdout


def _capture_files(root, session="2026-10-20", rows=3, json_rows=None):
    import json as _json

    import pandas as pd
    d = root / "data_store" / "options" / "cboe"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"snapshot_date": [session] * rows, "strike": range(rows)}).to_parquet(
        d / f"spx_{session}_023000Z.parquet")
    (d / f"spx_{session}_023000Z.json").write_text(_json.dumps(
        {"session": session, "rows": rows if json_rows is None else json_rows, "sha256": "ab" * 32}))


def test_check_captures_reads_every_new_capture_back(tmp_path):
    """The capture commit's gate (scripts/ci/check_captures.py): each new file is
    recognised by its name and read back whole; a broken one is reported and left
    out, another new file is left to the data commit."""
    import json as _json
    import pickle
    import subprocess

    import pandas as pd
    *_, work, _ = _repo(tmp_path)
    _capture_files(work)
    _capture_files(work, session="2026-10-21", json_rows=99)          # rows disagree
    (work / "data_store" / "options" / "cboe" / "vix_2026-10-20_023001Z.parquet").write_bytes(b"PAR1")
    pd.DataFrame({"snapshot_date": ["2026-10-20"], "strike": [1.0]}).to_parquet(
        work / "data_store" / "options" / "spx_2026-10-20.parquet")
    pd.DataFrame({"snapshot_date": ["2026-10-17"]}).to_parquet(
        work / "data_store" / "options" / "spx_2026-10-19.parquet")   # another session's rows
    cal = work / "data_store" / "calibrations"
    cal.mkdir()
    fit = {"model_version": 2, "spot_date": "2026-10-20", "as_of_date": "2026-10-20",
           "params": {"kappa": 5.0, "theta": 0.05, "sigma": 0.7, "rho": -0.9, "v0": 0.015},
           "leg_losses": {"spx_iv_rmse": 1.4, "vix_index_rmse": 0.6},
           "n_spx_expiries": 6, "n_spx_options": 54, "n_put_wing_expiries": 6,
           "n_vix_tenors": 4, "spx_snapshot_date": "2026-10-20", "spx_snapshot_session": "2026-10-20"}
    with open(cal / "joint_cal_2026-10-20.pkl", "wb") as f:
        pickle.dump(fit, f)
    (cal / "diagnostics").mkdir()
    (cal / "diagnostics" / "2026-10-20.json").write_text("{}")         # derived: not a capture
    (work / "data_store" / "forward").mkdir()
    (work / "data_store" / "forward" / "ledger_2026-10-20.json").write_text(_json.dumps({"session": "x"}))
    (work / "data_store" / "options" / "notes.txt").write_text("?")
    out = tmp_path / "ok.txt"
    r = subprocess.run([__import__("sys").executable, str(ROOT / "scripts" / "ci" / "check_captures.py"),
                        "--out", str(out), "--root", str(work)], capture_output=True, text=True, check=False)
    assert r.returncode == 1, r.stdout + r.stderr
    assert out.read_text().split() == [
        "data_store/calibrations/joint_cal_2026-10-20.pkl",
        "data_store/options/cboe/spx_2026-10-20_023000Z.json",
        "data_store/options/cboe/spx_2026-10-20_023000Z.parquet",
        "data_store/options/spx_2026-10-20.parquet"]
    errors = [ln for ln in r.stdout.splitlines() if ln.startswith("::error")]
    assert len(errors) == 5, errors
    for bad in ("spx_2026-10-21_023000Z.json", "spx_2026-10-21_023000Z.parquet", "vix_2026-10-20_023001Z",
                "spx_2026-10-19.parquet", "ledger_2026-10-20.json"):
        assert any(bad in e for e in errors), bad
    assert "other  data_store/options/notes.txt" in r.stdout
    assert "diagnostics" not in out.read_text()
