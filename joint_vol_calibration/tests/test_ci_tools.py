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

