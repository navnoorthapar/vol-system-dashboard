"""
test_ci_tools.py: the developer and CI tools under scripts/ci and scripts/bench
that are not part of the daily bot's gates (those are in test_infra_config.py).

  * mutation_check.py (`make mutation`): every curated mutant still names a
    site that occurs exactly once, so a refactor that copies or rewrites a
    mutated line fails here instead of turning the weekly run red as "stale".
  * baseline_gate.py --against (CI's lint job): a baseline may only shrink
    against the base branch's, so adding findings and re-baselining in the
    same change fails.
  * simulate_next_session.py (CI's simulate job): the simulated session's
    book is priced by parameters moved off the headline fit, with noise, so
    a test pinning a headline-derived value fails there, not in the bot.
  * merge_main_data.py (merging live main's bot data into a branch that
    restated the record): only the re-derivable data files may conflict,
    they resolve to the branch's version, and main's rows are re-derived
    into them from the database (main's new sessions in, the branch's
    restated rows kept, holiday artefacts out).
  * repo_growth.py (the daily workflow's last step): what a run's commits add
    to the repository, by kind of file, with a warning above a budget.
  * pip_audit_gate.py (CI's audit job): an advisory against a locked pin fails
    unless a dated, reasoned allowlist entry covers it; an expired or
    malformed entry fails; an unreachable advisory service is retried and
    then fails, never passes.
  * bot_alert.py (the daily workflow's last step, on failure): two failed
    scheduled runs in a row open, or comment on, one bot-failure issue.
  * deploy_streak.py (the daily workflow, before the freeze): consecutive NYSE
    sessions with a deploy commit on gh-pages, across holidays.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)             # dataclasses look the module up
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mutation(monkeypatch):
    return _load("scripts/ci/mutation_check.py", "mutation_check_under_test", monkeypatch)


class TestMutationCheck:
    def test_every_mutant_site_occurs_exactly_once(self, mutation):
        """ENG-2: C23's analytic Jacobian copied `beta = kappa - 1j * rho * sig * z`
        into _heston_log_cf_grad, the rho mutant matched twice, `make mutation`
        reported it 'stale' and exited 1. Each site now carries the next line."""
        assert mutation.stale_sites() == {}
        for name, rel, old, new in mutation.MUTANTS:
            assert old != new, name
            assert (ROOT / rel).read_text().count(new) == 0, f"{name}: the mutated text is already there"

    def test_both_copies_of_the_cf_line_are_mutated(self, mutation):
        src = (ROOT / mutation.HESTON).read_text()
        line = "beta = kappa - 1j * rho * sig * z"
        sites = [old for name, rel, old, _ in mutation.MUTANTS if line in old]
        assert src.count(line) == len(sites) == 2
        assert len({src.index(s) for s in sites}) == 2               # two different places

    def test_mutant_tests_exist_and_name_known_mutants(self, mutation):
        names = [m[0] for m in mutation.MUTANTS]
        assert len(names) == len(set(names))
        assert set(mutation.MUTANT_TESTS) <= set(names)
        assert set(mutation.EQUIVALENT) <= set(names)
        for rel in mutation.all_tests():
            assert (ROOT / rel).is_file(), rel

    def test_a_duplicated_site_is_reported_stale(self, mutation, tmp_path):
        name, rel, old, _ = mutation.MUTANTS[0]
        for _, r, *_ in mutation.MUTANTS:
            (tmp_path / r).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / r).write_text((ROOT / r).read_text())
        (tmp_path / rel).write_text((ROOT / rel).read_text() + "\n# " + old + "\n")
        assert mutation.stale_sites(tmp_path) == {name: 2}


def _baseline(counts: dict, version="ruff 0.15.0") -> str:
    nested: dict = {}
    for (path, code), n in counts.items():
        nested.setdefault(path, {})[code] = n
    return json.dumps({"tool": "ruff", "version": version, "total": sum(counts.values()), "counts": nested})


def _against(tmp_path, head: dict, base, *extra):
    hp, bp = tmp_path / "head.json", tmp_path / "base.json"
    hp.write_text(_baseline(head))
    if base is not None:
        bp.write_text(_baseline(base))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ci" / "baseline_gate.py"), "ruff",
                        "--baseline", str(hp), "--against", str(bp), *extra],
                       capture_output=True, text=True, check=False, cwd=ROOT)
    return r.returncode, r.stdout + r.stderr


_BASE = {("a.py", "F401"): 2, ("b.py", "E501"): 5}


class TestBaselineAgainstBase:
    BASE = _BASE      # read-only

    def test_a_re_baseline_that_grows_a_count_fails(self, tmp_path):
        """ENG-4: `--update --allow-growth` after adding an unused import moved the
        baseline 1300 -> 1302 and the gate then passed; the lint job now compares
        the committed baseline with the base branch's."""
        grown = {**self.BASE, ("a.py", "F401"): 3, ("c.py", "E402"): 1}
        rc, out = _against(tmp_path, grown, self.BASE)
        assert rc == 1, out
        assert "GREW  a.py F401: base baseline allows 2, this baseline 3" in out
        assert "GREW  c.py E402: base baseline allows 0, this baseline 1" in out
        assert "b.py" not in out.split("baseline-vs-base")[0]
        assert "FAILED" in out
        assert "2 pair(s) grew" in out

    def test_shrinking_or_equal_passes(self, tmp_path):
        rc, out = _against(tmp_path, self.BASE, self.BASE)
        assert rc == 0, out
        assert "OK" in out, out
        rc, out = _against(tmp_path, {("a.py", "F401"): 1}, self.BASE)
        assert rc == 0, out
        assert "OK" in out, out

    def test_a_labelled_re_baseline_passes_and_says_so(self, tmp_path):
        rc, out = _against(tmp_path, {**self.BASE, ("c.py", "E402"): 1}, self.BASE, "--allow-growth")
        assert rc == 0, out
        assert "GREW  c.py E402" in out
        assert "reviewed re-baseline" in out

    def test_no_base_baseline_is_a_new_baseline(self, tmp_path):
        rc, out = _against(tmp_path, self.BASE, None)
        assert rc == 0, out
        assert "nothing to compare" in out, out

    def test_growth_helper(self, monkeypatch):
        from collections import Counter
        mod = _load("scripts/ci/baseline_gate.py", "baseline_gate_under_test", monkeypatch)
        assert mod.baseline_growth(Counter({("x", "A"): 2}), Counter({("x", "A"): 2, ("y", "B"): 0})) == {}
        assert mod.baseline_growth(Counter(), Counter({("y", "B"): 1})) == {("y", "B"): (0, 1)}

    def test_the_lint_job_runs_it_against_the_base_branch(self):
        wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
        lint = wf[wf.index("\n  lint:"):wf.index("\n  bench:")]
        assert "--against" in lint
        for tool in ("ruff", "mypy"):
            assert f"scripts/ci/baseline_gate.py {tool}" in lint
        assert "github.base_ref" in lint
        assert "github.event.before" in lint
        assert "sparse-checkout: scripts/ci" in lint
        assert "'re-baseline'" in lint
        # after the gates, so ruff's `check .` never scans the base checkout
        assert lint.index("--against") > lint.index("Type gate")
        step = re.search(r"- name: Baselines may only shrink.*", lint, re.DOTALL).group(0)
        assert "|| rc=1" in step
        assert "exit $rc" in step


@pytest.fixture
def sim(monkeypatch):
    return _load("scripts/ci/simulate_next_session.py", "simulate_under_test", monkeypatch)


class TestSimulatedSession:
    def test_the_book_is_priced_off_the_headline_inside_the_box(self, sim):
        """PROD-8: the synthetic book was priced by the committed headline's own
        parameters, so the new fit reproduced it (kappa 5.0093 vs 4.9919, RMSE
        0.057 vp) and a test pinning the headline passed the simulation."""
        import pickle

        from joint_vol_calibration.calibration.joint_calibrator import is_acceptable_calibration
        from joint_vol_calibration.config import DATA_DIR, HESTON_BOUNDS
        with open(DATA_DIR / "calibrations" / "joint_cal_2026-10-16.pkl", "rb") as fh:
            head = pickle.load(fh)["params"]
        heads = [head, {**head, "sigma": 2.0, "rho": -0.12, "kappa": 19.9},
                 {"kappa": 0.2, "theta": 1e-4, "sigma": 0.06, "rho": -0.94, "v0": 1e-4}]
        for h in heads:
            p = sim.simulated_params(h)
            assert sim.simulated_params(h) == p                       # deterministic
            for k, (lo, hi) in HESTON_BOUNDS.items():
                assert lo < p[k] < hi, (k, p[k])
            assert is_acceptable_calibration(p)[0], (h, p)
            assert p["sigma"] <= 0.7 * 1.9 + 1e-12
        moves = sim.param_moves(sim.simulated_params(head), head)
        assert set(moves) == set(sim.PARAMS)
        assert min(moves.values()) > 2 * sim.MIN_PARAM_MOVE, moves

    def test_mid_noise_is_about_one_vol_point_seeded_and_keeps_parity(self, sim):
        import numpy as np

        from joint_vol_calibration.data import option_chain as oc
        F, T, D = 6000.0, 0.25, 0.99
        K = np.arange(4800.0, 7000.0, 25.0)
        vol = 0.18 + 0.15 * np.maximum(-np.log(K / F), 0.0)
        calls, puts = (oc.black76_price(F, K, T, D, vol, c) for c in (True, False))
        a = sim.noisy_mid(calls, F, K, T, D, True, seed=7)
        np.testing.assert_array_equal(a, sim.noisy_mid(calls, F, K, T, D, True, seed=7))
        assert not np.array_equal(a, sim.noisy_mid(calls, F, K, T, D, True, seed=8))
        # the call and the put at a strike move by the same vol: parity holds
        p = sim.noisy_mid(puts, F, K, T, D, False, seed=7)
        np.testing.assert_allclose(a - p, D * (F - K), rtol=0, atol=1e-6 * F)
        rms = []
        inside = np.abs(np.log(K / F)) / (0.2 * np.sqrt(T)) < 2.5
        for seed in range(30):
            iv = oc.black76_implied_vol(sim.noisy_mid(calls, F, K, T, D, True, seed=seed), F, K, T, D, True)
            dv = (iv - vol)[inside & np.isfinite(iv)] * 100
            rms.append(float(np.sqrt(np.mean(dv ** 2))))
        assert 0.8 < float(np.mean(rms)) < 1.2

    def test_no_gate_test_reads_a_shipped_session_by_its_date(self, sim):
        """C23 verify: five Surface tests opened data_store/surface/2026-10-16/ by
        name while the page renders the newest session. On the simulated next
        session (2026-10-20) the page showed that session, the tests compared it
        with 2026-10-16's files and the deploy gate failed on a legitimate new fit
        (the 2026-10-20 incident class). A gate test reads the session the page
        renders; a fixed session is a fixture's, under a temporary directory."""
        dated = re.compile(r'SHIPPED\s*/\s*"[\w.-]+"\s*/\s*"\d{4}-\d{2}-\d{2}"'
                           r'|["\']data_store/[\w./-]*?/\d{4}-\d{2}-\d{2}/')
        gates = sim.gate_tests(ROOT / ".github" / "workflows" / "daily_refresh.yml")
        files = sorted({f for fs in gates.values() for f in fs})
        assert "test_dashboard.py" in files
        hits = [f"{f}:{n}: {line.strip()}" for f in files
                for n, line in enumerate((ROOT / "joint_vol_calibration" / "tests" / f).read_text()
                                         .splitlines(), 1) if dated.search(line)]
        assert hits == []

    def test_each_simulated_slot_must_leave_its_run_record(self, sim, tmp_path):
        """FUND_GRADE_PLAN 0.9: the simulation runs the close and book slots with
        REFRESH_SLOT set, so each must leave one complete record that agrees with
        its status; the capture check then runs on what they wrote."""
        import inspect
        runs = tmp_path / "data_store" / "runs"
        runs.mkdir(parents=True)
        book = {"run_utc": "2026-10-28T02:30:00+00:00", "exit_code": 0, "calibration": {"status": "accepted"}}
        close = {"run_utc": "2026-10-27T22:07:00+00:00", "exit_code": 0}
        statuses = {"close": close, "book": book}
        assert len(sim.run_record_problems(tmp_path, statuses)) == 2              # nothing written
        for slot, st in statuses.items():
            (runs / f"run_x_{slot}.json").write_text(json.dumps(
                {"slot": slot, "run_utc": st["run_utc"], "complete": True, "exit_code": st["exit_code"],
                 "calibration": st.get("calibration")}))
        assert sim.run_record_problems(tmp_path, statuses) == []
        bad = {**book, "calibration": {"status": "rejected"}}
        assert "is not the run's" in sim.run_record_problems(tmp_path, {"book": bad})[0]
        src = inspect.getsource(sim.drive)
        assert src.index('os.environ["REFRESH_SLOT"] = "close"') < src.index('"market,labels"')
        assert src.index('os.environ["REFRESH_SLOT"] = "book"') < src.index('"market,capture,calibrate')
        main = inspect.getsource(sim.main)
        assert "run_record_problems(" in main
        assert "check_captures.py" in main

    def test_the_driver_prices_book_and_vix_indices_with_the_moved_parameters(self, sim):
        import inspect
        src = inspect.getsource(sim.drive)
        assert 'p = simulated_params(cal["params"])' in src
        assert "p = cal[" not in src.replace('p = simulated_params(cal["params"])', "")
        assert "noisy_mid(" in src
        assert 'heston_vix_term_index(p["kappa"]' in src                # the VIX leg of the same p
        main = inspect.getsource(sim.main)
        assert "MIN_PARAM_MOVE" in main
        assert "param_moves(" in main


# ── merge_main_data.py ────────────────────────────────────────────────────────

@pytest.fixture
def merge(monkeypatch):
    return _load("scripts/ci/merge_main_data.py", "merge_main_data_under_test", monkeypatch)


def _git_repo(tmp_path):
    import os
    import shutil
    if not shutil.which("git"):
        pytest.skip("git not available")
    env_ = {**os.environ, "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"), "GIT_CONFIG_NOSYSTEM": "1"}
    (tmp_path / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@example.com\n"
                                        "[init]\n\tdefaultBranch = main\n")
    repo = tmp_path / "repo"

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, env=env_, check=True, capture_output=True,
                              text=True).stdout.strip()
    repo.mkdir()
    git("init", "-q")
    files = {"data_store/market/vix_daily/vix_daily_2026.parquet": "v0\n",
             "data_store/signals/regime_labels.parquet": "l0\n", "README": "r0\n",
             "data_store/market/spx_ohlcv/spx_ohlcv_2026.parquet": "s0\n"}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git("add", "-A")
    git("commit", "-qm", "base")
    git("checkout", "-qb", "branch")
    return repo, git, env_


def _write_commit(repo, git, msg, **files):
    for rel, text in files.items():
        (repo / rel).write_text(text)
    git("add", "-A")
    git("commit", "-qm", msg)


class TestMergeMainData:
    VIX = "data_store/market/vix_daily/vix_daily_2026.parquet"
    LABELS = "data_store/signals/regime_labels.parquet"
    SPX = "data_store/market/spx_ohlcv/spx_ohlcv_2026.parquet"

    def test_data_conflicts_resolve_to_the_branch_with_mains_copy_set_aside(self, merge, tmp_path,
                                                                             monkeypatch):
        """ENG-1 / PROD-4: the C23 branch and main's bot both rewrote the 2026 VIX
        partition and the labels; git cannot merge them, and main's side undoes the
        restatement. The merge takes the branch's copy and sets main's aside for
        the re-derivation; a file only main changed merges as usual."""
        repo, git, env_ = _git_repo(tmp_path)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", env_["GIT_CONFIG_GLOBAL"])
        _write_commit(repo, git, "C23 restates", **{self.VIX: "v-cboe\n", self.LABELS: "l-restated\n"})
        git("checkout", "-q", "main")
        _write_commit(repo, git, "bot close slot", **{self.VIX: "v0\nv-1020\n", self.LABELS: "l0\nl-1020\n",
                                                       self.SPX: "s0\ns-1020\n"})
        git("checkout", "-q", "branch")
        state = merge.start_merge("main", cwd=repo)
        assert sorted(state["conflicts"]) == sorted([self.VIX, self.LABELS])
        assert sorted(state["main_changed"]) == sorted([self.VIX, self.LABELS, self.SPX])
        theirs = tmp_path / "theirs"
        merge.take_ours_and_set_theirs_aside(state, "main", theirs, cwd=repo)
        assert (repo / self.VIX).read_text() == "v-cboe\n"
        assert (repo / self.LABELS).read_text() == "l-restated\n"
        assert (repo / self.SPX).read_text() == "s0\ns-1020\n"            # clean: main's
        assert (theirs / self.VIX).read_text() == "v0\nv-1020\n"
        assert (theirs / self.LABELS).read_text() == "l0\nl-1020\n"
        assert (theirs / self.SPX).read_text() == "s0\ns-1020\n"
        git("add", "-A")
        assert git("diff", "--name-only", "--diff-filter=U") == ""

    def test_any_other_conflict_aborts_the_merge(self, merge, tmp_path, monkeypatch):
        repo, git, env_ = _git_repo(tmp_path)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", env_["GIT_CONFIG_GLOBAL"])
        _write_commit(repo, git, "branch", **{"README": "r-branch\n", self.VIX: "v-cboe\n"})
        git("checkout", "-q", "main")
        _write_commit(repo, git, "main", **{"README": "r-main\n", self.VIX: "v-main\n"})
        git("checkout", "-q", "branch")
        head = git("rev-parse", "HEAD")
        with pytest.raises(ValueError, match="README"):
            merge.start_merge("main", cwd=repo)
        assert git("rev-parse", "HEAD") == head
        assert git("status", "--porcelain") == ""                       # aborted, nothing left
        assert subprocess.run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=repo,
                              capture_output=True, check=False).returncode != 0

    def test_a_dirty_tree_or_nothing_to_merge_is_refused(self, merge, tmp_path, monkeypatch):
        repo, _, env_ = _git_repo(tmp_path)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", env_["GIT_CONFIG_GLOBAL"])
        with pytest.raises(ValueError, match="nothing to merge"):
            merge.start_merge("main", cwd=repo)
        (repo / "README").write_text("dirty\n")
        with pytest.raises(ValueError, match="uncommitted"):
            merge.start_merge("main", cwd=repo)

    def test_compare_labels_names_the_restatements_and_mains_sessions(self, merge, tmp_path):
        import pandas as pd
        idx = pd.to_datetime(["2019-08-09", "2020-02-20", "2026-10-19"])
        ours = pd.DataFrame({"regime": [1, 2, 0]}, index=idx)
        theirs = pd.DataFrame({"regime": [0, 2, 0, 1]}, index=idx.append(pd.to_datetime(["2026-10-20"])))
        merged = pd.DataFrame({"regime": [1, 2, 0, 1]}, index=theirs.index)
        paths = []
        for name, df in (("o", ours), ("t", theirs), ("m", merged)):
            paths.append(tmp_path / f"{name}.parquet")
            df.to_parquet(paths[-1])
        rep = merge.compare_labels(*paths)
        assert rep["main_only_dates"] == ["2026-10-20"]
        assert rep["main_only_in_merge"] is True
        assert rep["labels_differing_from_main"] == ["2019-08-09"]
        assert rep["branch_labels_changed_by_merge"] == []


class TestRepoGrowth:
    def test_a_runs_commits_are_measured_by_kind_with_a_budget_warning(self, tmp_path, monkeypatch):
        import os
        import shutil
        if not shutil.which("git"):
            pytest.skip("git not available")
        growth = _load("scripts/ci/repo_growth.py", "repo_growth_under_test", monkeypatch)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        (tmp_path / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@example.com\n")
        repo = tmp_path / "repo"
        (repo / "data_store" / "options" / "cboe").mkdir(parents=True)
        (repo / "data_store" / "signals").mkdir()

        def git(*a):
            return subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True).stdout
        git("init", "-q")
        (repo / "data_store" / "signals" / "regime_labels.parquet").write_bytes(os.urandom(1000))
        git("add", "-A")
        git("commit", "-qm", "base")
        base = git("rev-parse", "HEAD").strip()
        (repo / "data_store" / "options" / "cboe" / "spx_2026-10-20_023000Z.parquet").write_bytes(
            os.urandom(200_000))
        (repo / "data_store" / "signals" / "regime_labels.parquet").write_bytes(os.urandom(50_000))
        git("add", "-A")
        git("commit", "-qm", "bot")
        g = growth.range_growth(f"{base}..HEAD", cwd=repo)
        assert g["commits"] == 1
        assert g["blobs"] == {"cboe books": {"files": 1, "bytes": 200_000},
                              "labels": {"files": 1, "bytes": 50_000}}
        assert 250_000 < g["pack_bytes"] < 260_000                  # random bytes: incompressible
        lines = growth.report_range(g, budget_mb=0.2)
        assert lines[-1].startswith("::warning title=Repository growth::this run added 0.25 MB")
        assert not growth.report_range(g, budget_mb=1)[-1].startswith("::warning")

    def test_kinds(self, monkeypatch):
        growth = _load("scripts/ci/repo_growth.py", "repo_growth_kinds", monkeypatch)
        assert growth.kind_of("data_store/surface/2026-10-16/residuals.parquet") == "surface residuals"
        assert growth.kind_of("data_store/surface/2026-10-16/surface.json") == "surface json"
        assert growth.kind_of("data_store/options/spx_2026-10-16.parquet") == "SPX chain"
        assert growth.kind_of("data_store/options/cboe/vix_2026-10-20_040625Z.json") == "cboe books"
        assert growth.kind_of("data_store/runs/run_20261028T023000Z_book.json") == "run records"   # 0.9
        manifest = "data_store/options/cboe_manifest/spx_2026-10-27_023000Z.json"
        assert growth.kind_of(manifest) == "cboe manifests"
        assert growth.kind_of("README.md") == "other"

    def test_the_pull_request_gate_fails_on_data_store_files_and_growth(self, tmp_path, monkeypatch):
        """FUND_GRADE_PLAN 0.11: a pull request may not add any file under data_store/
        (the bot owns it) nor grow the repository past the budget (unless labelled
        allow-growth, which never lifts the data_store/ rule). Changing an existing
        data file is not an addition; main's own commits are not the PR's."""
        import os
        import shutil
        if not shutil.which("git"):
            pytest.skip("git not available")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        (tmp_path / "gitconfig").write_text("[user]\n\tname = t\n\temail = t@example.com\n"
                                            "[init]\n\tdefaultBranch = main\n")
        repo = tmp_path / "repo"
        (repo / "data_store" / "signals").mkdir(parents=True)

        def git(*a):
            r = subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)
            return r.stdout.strip()

        def gate(*extra, budget="0.1"):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ci" / "repo_growth.py"), "--range",
                                f"main..{git('rev-parse', 'HEAD')}", "--gate", "--budget-mb", budget,
                                "--repo", str(repo), *extra],
                               cwd=repo, capture_output=True, text=True, check=False)
            return r.returncode, r.stdout
        git("init", "-q")
        (repo / "data_store" / "signals" / "regime_labels.parquet").write_bytes(os.urandom(1000))
        (repo / "app.py").write_text("x = 1\n")
        git("add", "-A")
        git("commit", "-qm", "base")
        git("checkout", "-q", "-b", "feature")
        (repo / "app.py").write_text("x = 2\n")
        (repo / "data_store" / "signals" / "regime_labels.parquet").write_bytes(os.urandom(2000))  # a change
        git("commit", "-qam", "code")
        git("checkout", "-q", "main")                      # the bot adds a file on main meanwhile
        (repo / "data_store" / "runs").mkdir()
        (repo / "data_store" / "runs" / "run_20261028T023000Z_book.json").write_text("{}")
        git("add", "-A")
        git("commit", "-qm", "bot")
        git("checkout", "-q", "feature")
        git("merge", "-q", "--no-edit", "main")            # the branch merges main: not its own addition
        code, out = gate()
        assert code == 0, out
        assert "| files added under data_store/ | 0 |" in out
        (repo / "data_store" / "forward").mkdir()
        (repo / "data_store" / "forward" / "ledger_2026-10-27.json").write_text("{}")
        git("add", "-A")
        git("commit", "-qm", "an agent commits data")
        code, out = gate()
        assert code == 1
        assert "::error title=Repository growth::data_store/forward/ledger_2026-10-27.json" in out
        assert gate("--allow-growth")[0] == 1                    # the label never lifts this rule
        git("rm", "-q", "data_store/forward/ledger_2026-10-27.json")
        (repo / "big.bin").write_bytes(os.urandom(300_000))
        git("add", "-A")
        git("commit", "-qm", "a large file")
        code, out = gate()
        assert code == 1
        assert "budget 0.1 MB" in out
        assert "| files added under data_store/ | 0 |" in out
        assert gate("--allow-growth")[0] == 0
        assert gate(budget="1")[0] == 0
        mv = repo / "data_store" / "market"
        mv.mkdir()
        git("mv", "big.bin", "data_store/market/big.bin")         # a rename into data_store/ is an addition
        git("commit", "-qm", "move")
        assert gate(budget="1")[0] == 1

    def test_pull_requests_run_the_gate(self):
        import yaml
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
        job = wf["jobs"]["growth"]
        assert job["if"] == "github.event_name == 'pull_request'"
        assert "continue-on-error" not in job
        checkout, _python, step = job["steps"]
        assert checkout["with"]["fetch-depth"] == 0
        assert step["env"]["BASE"] == "${{ github.event.pull_request.base.sha }}"
        assert step["env"]["HEAD"] == "${{ github.event.pull_request.head.sha }}"
        assert "allow-growth" in step["env"]["ALLOW_GROWTH"]
        run = step["run"]
        assert 'repo_growth.py --range "$BASE..$HEAD" --gate --budget-mb' in run
        assert "|| true" not in run
        assert "continue-on-error" not in step

    def test_the_daily_workflow_reports_it(self):
        wf = (ROOT / ".github" / "workflows" / "daily_refresh.yml").read_text()
        step = wf[wf.index("- name: Repository growth of this run"):]
        step = step[:step.index("\n      - name:", 10)]
        assert "repo_growth.py --range" in step
        assert "--budget-mb" in step
        assert "always()" in step
        assert "|| true" in step  # reporting never fails the run


# ── pip_audit_gate.py ─────────────────────────────────────────────────────────

@pytest.fixture
def audit(monkeypatch):
    return _load("scripts/ci/pip_audit_gate.py", "pip_audit_gate_under_test", monkeypatch)


def _report(*deps):
    """A pip-audit JSON report; each dep is (name, version, [(id, [aliases])]),
    or (name, version, None, skip reason)."""
    out = []
    for d in deps:
        if len(d) == 4:
            out.append({"name": d[0], "version": d[1], "skip_reason": d[3]})
        else:
            out.append({"name": d[0], "version": d[1],
                        "vulns": [{"id": i, "aliases": a, "fix_versions": ["9.9"]} for i, a in d[2]]})
    return {"dependencies": out, "fixes": []}


def _entry(id_="GHSA-aaaa", package="pillow", reason="nothing here decodes an untrusted image file",
           expires="2026-12-01"):
    return {"id": id_, "package": package, "reason": reason, "expires": expires}


class TestPipAuditGate:
    TODAY = __import__("datetime").date(2026, 11, 1)

    def test_an_unlisted_finding_fails(self, audit):
        """FUND_GRADE_PLAN 0.6: the audit job ran `|| true` with continue-on-error,
        so no advisory could ever fail CI."""
        rep = _report(("Pillow", "12.3.0", [("GHSA-aaaa", ["CVE-1"])]))
        v = audit.judge(rep, {"entries": []}, self.TODAY)
        assert not v.ok
        assert [(f.package, f.id) for f in v.uncovered] == [("pillow", "GHSA-aaaa")]

    def test_a_listed_unexpired_finding_passes_by_id_or_alias(self, audit):
        rep = _report(("Pillow", "12.3.0", [("GHSA-aaaa", ["CVE-1"]), ("PYSEC-2", ["CVE-2"])]))
        v = audit.judge(rep, {"entries": [_entry(), _entry(id_="CVE-2")]}, self.TODAY)
        assert v.ok, v
        assert len(v.covered) == 2
        assert v.unused == []

    def test_the_entry_must_name_the_package(self, audit):
        rep = _report(("urllib3", "2.8.0", [("GHSA-aaaa", [])]))
        v = audit.judge(rep, {"entries": [_entry(package="pillow")]}, self.TODAY)
        assert not v.ok
        assert [e.package for e in v.unused] == ["pillow"]

    def test_an_expired_entry_fails_even_when_it_still_matches(self, audit):
        rep = _report(("pillow", "12.3.0", [("GHSA-aaaa", [])]))
        v = audit.judge(rep, {"entries": [_entry(expires="2026-10-31")]}, self.TODAY)
        assert not v.ok
        assert any("expired on 2026-10-31" in p for p in v.problems)
        assert [f.id for f in v.uncovered] == ["GHSA-aaaa"]
        # and when nothing matches it any more: the stale exception still has to go
        v = audit.judge(_report(), {"entries": [_entry(expires="2026-10-31")]}, self.TODAY)
        assert not v.ok

    @pytest.mark.parametrize(("bad", "why"), [
        ({"reason": ""}, "missing reason"),
        ({"reason": "n/a"}, "20 characters"),
        ({"expires": "soon"}, "not a YYYY-MM-DD"),
        ({"expires": None}, "missing expires"),
        ({"package": ""}, "missing package"),
        ({"expires": "2027-12-01"}, "more than 180 days"),
    ])
    def test_a_malformed_entry_fails(self, audit, bad, why):
        rep = _report(("pillow", "12.3.0", [("GHSA-aaaa", [])]))
        v = audit.judge(rep, {"entries": [{**_entry(), **bad}]}, self.TODAY)
        assert not v.ok
        assert any(why in p for p in v.problems), v.problems

    def test_the_allowlist_must_be_an_object_with_entries(self, audit):
        assert not audit.judge(_report(), [], self.TODAY).ok
        assert audit.judge(_report(), {"entries": []}, self.TODAY).ok

    def test_skipped_dependencies_are_reported_not_failed(self, audit):
        v = audit.judge(_report(("foo", "1.0", None, "not on PyPI")), {"entries": []}, self.TODAY)
        assert v.ok
        assert v.skipped == [("foo", "not on PyPI")]
        assert "- not audited: foo (not on PyPI)" in audit.report_lines(v)

    def test_the_lock_is_audited_at_public_versions(self, audit):
        h = "    --hash=sha256:"
        lock = ("--index-url https://pypi.org/simple\n\narch==8.0.0 \\\n" + h + "a" * 64 + " \\\n"
                + h + "b" * 64 + "\nTorch==2.14.0+cpu \\\n" + h + "c" * 64 + "\n# a comment==1.0\n")
        pins = audit.lock_pins(lock)
        assert pins == {"arch": "8.0.0", "torch": "2.14.0+cpu"}
        assert audit.audit_input(pins) == "arch==8.0.0\ntorch==2.14.0\n"

    def test_a_network_error_is_retried_then_fails_closed(self, audit):
        calls, pauses, inputs = [], [], []

        def flaky(cmd, ok_on):
            calls.append(cmd)
            inputs.append(Path(cmd[cmd.index("-r") + 1]).read_text())
            if len(calls) >= ok_on:
                out = cmd[cmd.index("--output") + 1]
                Path(out).write_text(json.dumps(_report(("pillow", "12.3.0", []))))
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return subprocess.CompletedProcess(cmd, 1, "", "ConnectionError: HTTPSConnectionPool(pypi.org)")

        rep = audit.run_audit("pillow==12.3.0 \\\n  --hash=sha256:" + "a" * 64 + "\n",
                              runner=lambda c: flaky(c, 3), retries=4, sleep=pauses.append, pause_s=1.0)
        assert rep is not None
        assert len(calls) == 3
        assert pauses == [1.0, 2.0]
        assert inputs[0] == "pillow==12.3.0\n"                 # pins only: no hashes, no index options
        assert {"--no-deps", "--disable-pip"} <= set(calls[0])
        calls.clear()
        pauses.clear()
        assert audit.run_audit("pillow==12.3.0\n", runner=lambda c: flaky(c, 99), retries=3,
                               sleep=pauses.append, pause_s=1.0) is None
        assert len(calls) == 3
        assert pauses == [1.0, 2.0]

    def test_main_exit_codes(self, audit, tmp_path, monkeypatch):
        allow = tmp_path / "allow.json"
        allow.write_text(json.dumps({"entries": [_entry(expires="2026-12-01")]}))
        rep = tmp_path / "rep.json"
        summary = tmp_path / "summary.md"
        rep.write_text(json.dumps(_report(("pillow", "12.3.0", [("GHSA-aaaa", [])]))))
        assert audit.main(["--report", str(rep), "--allowlist", str(allow), "--summary", str(summary)],
                          today=self.TODAY) == 0
        assert "OK: no uncovered advisory" in summary.read_text()
        rep.write_text(json.dumps(_report(("pillow", "12.3.0", [("GHSA-bbbb", [])]))))
        assert audit.main(["--report", str(rep), "--allowlist", str(allow)], today=self.TODAY) == 1
        lock = tmp_path / "lock.txt"
        lock.write_text("pillow==12.3.0\n")
        monkeypatch.setattr(audit, "RETRY_SLEEP_S", 0.0)
        no_report = lambda c: subprocess.CompletedProcess(c, 1, "", "network down")  # noqa: E731
        assert audit.main(["--lock", str(lock), "--allowlist", str(allow)], runner=no_report,
                          today=self.TODAY) == 2

    def test_the_committed_allowlist_is_well_formed(self, audit):
        """Every committed entry has a reason and a valid date; whether an entry has
        expired is the CI job's question (it depends on the day it runs)."""
        from datetime import date
        data = json.loads((ROOT / "scripts" / "ci" / "pip_audit_allowlist.json").read_text())
        for e in data["entries"]:
            entries, problems = audit.load_allowlist({"entries": [e]}, date.fromisoformat(e["expires"]))
            assert problems == [], e
            assert len(entries) == 1
            assert e["package"] == audit.normalise(e["package"]), e
        keys = [(e["package"], e["id"]) for e in data["entries"]]
        assert len(keys) == len(set(keys))

    def test_the_audit_job_is_a_gate(self):
        wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
        job = wf[wf.index("\n  audit:"):wf.index("\n  mutation:")]
        job = "\n".join(ln for ln in job.splitlines() if not ln.lstrip().startswith("#"))
        assert "continue-on-error" not in job
        assert "|| true" not in job
        assert "scripts/ci/pip_audit_gate.py" in job
        assert "--require-hashes -r scripts/ci/pip_audit_lock.txt" in job
        mk = (ROOT / "Makefile").read_text()
        audit_target = mk[mk.index("\naudit:"):mk.index("\ndocker-build:")]
        assert "|| true" not in audit_target
        assert "pip_audit_gate.py" in audit_target


@pytest.fixture
def alert(monkeypatch):
    return _load("scripts/ci/bot_alert.py", "bot_alert_under_test", monkeypatch)


class _FakeGh:
    """A fake `gh`: canned run history and open issues, every call recorded."""

    def __init__(self, runs, open_issues=(), fail=0):
        self.runs, self.open_issues, self.fail, self.calls = list(runs), list(open_issues), fail, []

    def __call__(self, args):
        args = list(args)
        self.calls.append(args)
        if self.fail:
            self.fail -= 1
            raise subprocess.CalledProcessError(1, ["gh", *args], stderr="HTTP 502")
        if args[:2] == ["run", "list"]:
            return json.dumps(self.runs)
        if args[:2] == ["issue", "list"]:
            return json.dumps(self.open_issues)
        if args[:2] == ["issue", "create"]:
            return "https://github.com/o/r/issues/77\n"
        return ""

    def made(self, *verb):
        return [c for c in self.calls if c[:len(verb)] == list(verb)]


def _runs(*conclusions, current=None):
    """gh run list --json rows, newest first; ``current`` is this (in-progress) run."""
    rows = ([{"databaseId": current, "event": "schedule", "status": "in_progress", "conclusion": ""}]
            if current else [])
    rows += [{"databaseId": 100 - i, "event": "schedule", "status": "completed", "conclusion": c,
              "createdAt": f"run {i}", "url": f"https://x/runs/{100 - i}"} for i, c in enumerate(conclusions)]
    return rows


class TestBotAlert:
    """FUND_GRADE_PLAN 0.4: two failed scheduled runs in a row open (or comment on)
    one bot-failure issue; never a second open issue; a manual run never alerts."""

    def _alert(self, alert, gh, event="schedule"):
        return alert.alert("200", event, "https://x/runs/200", "book", runner=gh, sleep=lambda s: None)

    def test_two_failures_in_a_row_open_one_issue(self, alert):
        gh = _FakeGh(_runs("failure", "success", current=200))
        res = self._alert(alert, gh)
        assert (res["action"], res["streak"]) == ("created", 2)
        assert len(gh.made("issue", "create")) == 1
        create = gh.made("issue", "create")[0]
        assert create[create.index("--label") + 1] == alert.LABEL
        assert "https://x/runs/200" in create[create.index("--body") + 1]
        assert "https://x/runs/100" in create[create.index("--body") + 1]
        assert gh.made("label", "create")              # the label exists before the issue uses it
        assert gh.calls.index(gh.made("label", "create")[0]) < gh.calls.index(create)
        assert gh.made("issue", "comment") == []

    def test_a_failure_after_a_success_does_not_alert(self, alert):
        gh = _FakeGh(_runs("success", "failure", "failure", current=200))
        res = self._alert(alert, gh)
        assert (res["action"], res["streak"]) == ("none", 1)
        assert [c[:2] for c in gh.calls] == [["run", "list"]]

    def test_an_open_issue_gets_a_comment_not_a_second_issue(self, alert):
        gh = _FakeGh(_runs("failure", "failure", "success"), open_issues=[{"number": 12, "url": "u"},
                                                                            {"number": 9, "url": "v"}])
        res = self._alert(alert, gh)
        assert (res["action"], res["streak"], res["issue"]) == ("commented", 3, 9)
        assert gh.made("issue", "create") == []
        assert gh.made("issue", "comment")[0][2] == "9"

    def test_runs_that_say_nothing_about_the_bot_are_passed_over(self, alert):
        """Cancelled, queued and manual runs neither extend nor break a streak."""
        runs = _runs("cancelled", "failure", "skipped", "success", current=200)
        runs.insert(1, {"databaseId": 150, "event": "workflow_dispatch", "status": "completed",
                        "conclusion": "success"})
        runs.insert(1, {"databaseId": 151, "event": "schedule", "status": "queued", "conclusion": ""})
        assert alert.failure_streak(runs, "200")[0] == 2
        assert alert.failure_streak(_runs("timed_out", "startup_failure", "success"), "200")[0] == 3

    def test_a_manual_run_never_alerts(self, alert):
        gh = _FakeGh(_runs("failure", "failure"))
        res = self._alert(alert, gh, event="workflow_dispatch")
        assert res["action"] == "none"
        assert gh.calls == []

    def test_a_gh_failure_is_retried_then_reported(self, alert, monkeypatch, capsys):
        gh = _FakeGh(_runs("failure", current=200), fail=1)
        assert self._alert(alert, gh)["action"] == "created"         # one 502, then through
        monkeypatch.setattr(alert, "gh", _FakeGh([], fail=99))
        monkeypatch.setattr(alert.time, "sleep", lambda s: None)
        assert alert.main(["--run-id", "200", "--event", "schedule"]) == 1
        assert "::warning title=Failure alert::" in capsys.readouterr().out

    def test_dry_run_decides_without_writing(self, alert):
        gh = _FakeGh(_runs("failure", current=200))
        res = alert.alert("200", "schedule", "u", runner=gh, sleep=lambda s: None, dry_run=True)
        assert res["action"] == "created"
        assert [c[:2] for c in gh.calls] == [["run", "list"], ["issue", "list"]]


    def test_another_workflow_alerts_on_every_failure_with_its_report(self, alert, tmp_path, monkeypatch):
        """The weekly reproduction's use: its own label, every failure (a manual run's
        too), the report as the body; no run history is read."""
        gh = _FakeGh([])
        res = alert.alert("9", "workflow_dispatch", "https://x/runs/9", workflow="reproduce.yml", runner=gh,
                          sleep=lambda s: None, label="reproduce-failure", alert_after=1, all_events=True,
                          title="make reproduce failed", details="| artifact | FAIL |")
        assert (res["action"], res["streak"]) == ("created", 1)
        assert gh.made("run", "list") == []
        create = gh.made("issue", "create")[0]
        assert create[create.index("--label") + 1] == "reproduce-failure"
        assert create[create.index("--title") + 1] == "make reproduce failed"
        body = create[create.index("--body") + 1]
        assert "https://x/runs/9" in body
        assert "| artifact | FAIL |" in body
        listed = gh.made("issue", "list")[0]
        assert listed[listed.index("--label") + 1] == "reproduce-failure"
        missing = tmp_path / "none.md"
        gh2 = _FakeGh([])
        monkeypatch.setattr(alert, "gh", gh2)
        assert alert.main(["--run-id", "9", "--event", "schedule", "--label", "reproduce-failure",
                           "--alert-after", "1", "--body-file", str(missing)]) == 0
        body = gh2.made("issue", "create")[0][gh2.made("issue", "create")[0].index("--body") + 1]
        assert f"no report: {missing}" in body


@pytest.fixture
def streak(monkeypatch):
    return _load("scripts/ci/deploy_streak.py", "deploy_streak_under_test", monkeypatch)


class TestDeployStreak:
    """FUND_GRADE_PLAN 0.4: 'N consecutive NYSE sessions deployed' from the deploy
    commits on gh-pages; a holiday does not break the run, a missing session does."""

    @staticmethod
    def _log(*sessions):
        return [f"Deploy session {s} from {'a' * 40} (calibration accepted; run {i})"
                for i, s in enumerate(sessions)]

    def test_a_holiday_and_a_weekend_do_not_break_the_streak(self, streak):
        from joint_vol_calibration.data import database as db
        holiday = "2026-11-26"                         # Thanksgiving: a calendar fact, not data
        assert not db.is_nyse_session(holiday)
        after = [db.session_on_or_before("2026-11-30")]            # Monday after the Friday
        before = [db.previous_session(holiday)]                     # Wednesday
        before.insert(0, db.previous_session(before[0]))
        friday = db.session_on_or_before("2026-11-29")
        sessions = [*before, friday, *after]
        rec = streak.record(["Initial commit", *self._log(*reversed(sessions)), *self._log(sessions[-1])])
        assert (rec["streak"], rec["first"], rec["through"]) == (len(sessions), sessions[0], sessions[-1])
        assert rec["deploys"] == len(sessions)                      # a session deployed twice counts once
        assert rec["commits_read"] == len(sessions) + 2

    def test_a_missing_session_ends_it(self, streak):
        from joint_vol_calibration.data import database as db
        s3 = "2026-10-28"
        s2 = db.previous_session(s3)
        s1 = db.previous_session(s2)
        s0 = db.previous_session(s1)
        rec = streak.record(self._log(s3, s2, s0))                  # s1 never deployed
        assert (rec["streak"], rec["first"], rec["through"], rec["deploys"]) == (2, s2, s3, 3)

    def test_other_subjects_and_no_deploys(self, streak, tmp_path):
        assert streak.deployed_sessions(["Deploy session unknown from x", "Deploy session 2026-13-01 from x",
                                         "Merge branch", "  Deploy session 2026-10-27 from x (y)"]) == {
            __import__("datetime").date(2026, 10, 27)}
        log, out = tmp_path / "log.txt", tmp_path / ".deploy-streak.json"
        log.write_text("Initial commit\n")
        assert streak.main(["--log", str(log), "--out", str(out)]) == 0
        rec = json.loads(out.read_text())
        assert (rec["schema"], rec["streak"], rec["first"]) == ("deploy_streak/1", 0, None)
        log.write_text("\n".join(self._log("2026-10-27")) + "\n")
        streak.main(["--log", str(log), "--out", str(out)])
        assert json.loads(out.read_text())["streak"] == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == [".deploy-streak.json", "log.txt"]   # no .tmp

    def test_it_parses_the_workflows_own_deploy_message(self, streak):
        import yaml
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / "daily_refresh.yml").read_text())
        step = next(s for s in wf["jobs"]["refresh"]["steps"] if s.get("name") == "Deploy to GitHub Pages")
        msg = re.sub(r"\$\{\{ steps\.refresh\.outputs\.session \}\}", "2026-10-27",
                     step["with"]["full_commit_message"])
        msg = re.sub(r"\$\{\{[^}]*\}\}", "x", msg)
        assert streak.deployed_sessions([msg]) == {__import__("datetime").date(2026, 10, 27)}


@pytest.fixture
def repro(monkeypatch):
    return _load("scripts/ci/reproduce.py", "reproduce_status_under_test", monkeypatch)


class TestReproduceStatus:
    """FUND_GRADE_PLAN 0.3: make reproduce writes its result as a status JSON
    (scripts/ci/reproduce.py --status), committed weekly by reproduce.yml and
    printed on Model Risk; every count is derived from the checks."""

    def _checks(self, repro):
        exact = repro.Check("greeks: a.parquet", 1e-10)
        close = repro.Check("calibration: b.pkl", 1e-6)
        close.note(3e-8, "params.kappa")
        over = repro.Check("c7: c.parquet", 1e-10)
        over.note(2e-9, "pnl[4]")
        broken = repro.Check("backtest: d.pkl", 1e-10)
        broken.note(float("nan"), "nav[0]")
        broken.problems.append("missing key 'metrics'")
        return [exact, close, over, broken]

    def test_the_status_counts_come_from_the_checks(self, repro, tmp_path):
        checks = self._checks(repro)
        rec = repro.status_record(checks, commit="c" * 40, run_url="https://github.com/o/r/actions/runs/1",
                                  finished_utc="2026-11-02T07:30:00+00:00")
        assert rec["schema"] == "reproduce_status/1"
        assert (rec["n_checks"], rec["n_ok"], rec["n_exact"]) == (
            len(checks), sum(c.ok for c in checks), sum(c.ok and c.deviation == 0 for c in checks))
        finite = [c for c in checks if c.deviation == c.deviation]
        worst = max(finite, key=lambda c: c.deviation)
        assert (rec["max_deviation"], rec["max_deviation_artifact"]) == (worst.deviation, worst.artifact)
        assert [f["artifact"] for f in rec["failed"]] == [c.artifact for c in checks if not c.ok]
        assert rec["failed"][-1]["deviation"] is None                   # NaN: null, never invalid JSON
        assert rec["failed"][-1]["problems"] == ["missing key 'metrics'"]
        path = tmp_path / "reproduce" / "status.json"
        repro.write_status(path, rec)
        assert json.loads(path.read_text()) == rec
        assert [p.name for p in path.parent.iterdir()] == ["status.json"]

    def test_main_writes_the_status_whatever_the_outcome(self, repro, tmp_path, monkeypatch):
        failing = repro.Step("greeks", [], [], error="no headline fit named")
        monkeypatch.setattr(repro, "build_steps", lambda root, parts=None: [failing])
        monkeypatch.setattr(repro, "copy_tree", lambda src, dst: dst.mkdir(parents=True, exist_ok=True))
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
        monkeypatch.setenv("GITHUB_RUN_ID", "77")
        status = tmp_path / "status.json"
        assert repro.main(["--out", str(tmp_path / "out"), "--status", str(status)]) == 1
        rec = json.loads(status.read_text())
        assert (rec["n_checks"], rec["n_ok"], rec["run_url"]) == (1, 0, "https://github.com/o/r/actions/runs/77")
        assert rec["failed"][0]["problems"] == ["not run: no headline fit named"]
        assert rec["commit"] == subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                               text=True, check=True).stdout.strip()

    def test_the_weekly_workflow_commits_it_and_raises_an_issue_on_failure(self):
        import yaml
        text = (ROOT / ".github" / "workflows" / "reproduce.yml").read_text()
        wf = yaml.safe_load(text)
        on = wf[True]                                   # YAML 1.1 reads the key `on` as True
        assert {"schedule", "workflow_dispatch"} <= set(on)
        assert wf["concurrency"] == {"group": "reproduce", "cancel-in-progress": False}
        assert (wf["permissions"]["contents"], wf["permissions"]["issues"]) == ("write", "write")
        (job,) = wf["jobs"].values()
        assert "if" not in job
        steps = {s.get("name"): s for s in job["steps"]}
        run = steps["Rebuild in a scratch copy and diff against the committed artifacts"]["run"]
        assert "make reproduce" in run
        assert "STATUS=data_store/reproduce/status.json" in run
        commit = steps["Commit the result"]
        assert commit["if"].startswith("always()")
        assert "scripts/ci/commit_and_push.sh" in commit["run"]
        assert "data_store/reproduce/status.json" in commit["run"]
        issue = job["steps"][-1]
        assert (issue["if"], issue["continue-on-error"]) == ("failure()", True)
        for part in ("scripts/ci/bot_alert.py", "--label reproduce-failure", "--alert-after 1",
                     "--all-events", "--body-file"):
            assert part in issue["run"], part
        mk = (ROOT / "Makefile").read_text()
        assert "$(if $(STATUS),--status $(STATUS))" in mk[mk.index("\nreproduce:"):mk.index("\nmutation:")]
        tests = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
        assert "reproduce" not in tests["jobs"]
        assert "make reproduce" not in "\n".join(ln for ln in (ROOT / ".github" / "workflows" / "tests.yml")
                                                 .read_text().splitlines() if not ln.lstrip().startswith("#"))
