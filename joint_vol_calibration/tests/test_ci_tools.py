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
    Its --force-fail mode, run here in full, fails the run and checks that
    bot_alert.py would open the bot-failure issue after two failures.
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
  * page_budget.py (CI's a11y job): the Surface page's weight against 450,000
    bytes plus 7,000 an expiry past 45 and less 5,500 an expiry short of it, the
    expiry count read from the page; its gzipped size at a stated level.
"""
from __future__ import annotations

import gzip
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


# bot_alert.py's mistakes the forced-failure check must catch (each a change of the
# repository's script): the 0.4 review's, run through the alert's own command line
_FAILED_SET = 'FAILED = frozenset({"failure", "timed_out", "startup_failure"})'
ALERT_MUTANTS = {
    "alerts on a third failure": ("ALERT_AFTER = 2 ", "ALERT_AFTER = 3 "),
    "alerts on one failure": ("ALERT_AFTER = 2 ", "ALERT_AFTER = 1 "),
    "a success does not end the streak": ("        if c in SUCCEEDED:\n            break",
                                          "        if c in SUCCEEDED:\n            continue"),
    "a cancelled run counts as failed": (_FAILED_SET, _FAILED_SET.replace('"startup_failure"',
                                                                          '"startup_failure", "cancelled"')),
    "a second open issue": ("    if open_issues:\n", "    if False:\n"),
    "main() drops the event": ("        res = alert(args.run_id, args.event,",
                               "        res = alert(args.run_id, 'workflow_dispatch',"),
    "main() alerts on nothing": ("    try:\n        res = alert(",
                                 "    return 0\n    try:\n        res = alert("),
    "the title loses the streak": ('title = title or f"Daily bot: {streak} scheduled runs failed in a row"',
                                   'title = title or "Daily bot failed"'),
    "a manual run reads the history": ('    if event != "schedule" and not all_events:', "    if False:"),
}


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

    def test_a_no_fit_run_must_skip_the_fit_and_still_build_the_surface(self, sim, tmp_path):
        """The simulation always produced an accepted fit, so CI never saw a newest
        surface session without one, although every Friday is one (the Saturday slot
        finds Yahoo's book empty and skips the fit; tier 0.7 item 4 builds the surface
        from the Cboe book anyway). Such a session failed ten presentation tests. With
        --no-fit the book slot sees an empty Yahoo book, and the run must skip the fit,
        keep the headline, build the session's surface with Heston marked not scored,
        record the ledger from the Cboe book and stay publishable."""
        s = "2026-11-02"
        (tmp_path / "data_store" / "market" / "vix_futures_daily").mkdir(parents=True)
        (tmp_path / "data_store" / "market" / "vix_futures_daily" / "vix_futures_daily_2026.parquet").touch()
        surf = tmp_path / "data_store" / "surface" / s
        surf.mkdir(parents=True)
        why = "no joint SPX/VIX Heston fit is stored for 2026-11-02"
        (surf / "scorecard.json").write_text(json.dumps({"models": [
            {"model": "svi_cal", "available": True},
            {"model": "heston_v2", "available": False, "reason": why}]}))
        st = {"simulated": {"session": s, "headline_before": "2026-10-30", "headline_params": {}},
              "calibration": {"status": "skipped", "reason": "no SPX option chain returned"},
              "headline_calibration": {"date": "2026-10-30"},
              "artifact_backfill": {"rebuilt": {s: {"surface": {"ok": True}}}},
              "forward_ledger": {"status": "recorded"}, "capture": {"status": "captured"},
              "market": {"vx_futures": {"last": s}}, "publishable": True}
        assert sim.outcome_problems(st, tmp_path, no_fit=True) == []
        # the same run is a failure of the mode that expects a new fit
        assert sim.outcome_problems(st, tmp_path) == [
            "the simulated fit was not accepted: skipped no SPX option chain returned"]
        accepted = {**st, "calibration": {"status": "accepted"}}
        assert sim.outcome_problems(accepted, tmp_path, no_fit=True)[0].startswith(
            "the empty Yahoo book was not skipped")
        moved = {**st, "headline_calibration": {"date": s}}
        assert sim.outcome_problems(moved, tmp_path, no_fit=True)[0].startswith("the headline fit moved")
        unbuilt = {**st, "artifact_backfill": {"rebuilt": {}}}
        assert sim.outcome_problems(unbuilt, tmp_path, no_fit=True)[0].startswith(
            f"the backfill did not build {s}'s surface")
        (surf / "scorecard.json").write_text(json.dumps({"models": [{"model": "heston_v2",
                                                                     "available": True}]}))
        assert sim.outcome_problems(st, tmp_path, no_fit=True)[0].startswith(
            f"{s}'s scorecard does not list Heston as not scored")
        (surf / "scorecard.json").write_text(json.dumps({"models": [
            {"model": "heston_v2", "available": False, "reason": why}]}))
        rebuilt = {s: {"surface": {"ok": True}}, "2026-10-28": {"surface": {"ok": False, "error": "x"}}}
        failed = {**st, "artifact_backfill": {"rebuilt": rebuilt}}
        assert sim.outcome_problems(failed, tmp_path, no_fit=True) == ["artifact 2026-10-28 surface: x"]
        # the Ticker the driver hands the bot lists no expiry in this mode
        src = (ROOT / "scripts" / "ci" / "simulate_next_session.py").read_text()
        assert "self.options = () if no_fit else" in src
        assert '"--drive", str(status_file), *(["--no-fit"] if args.no_fit else [])' in src
        # and CI runs it beside the run with a fit
        import yaml
        jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())["jobs"]
        runs = [st.get("run", "") for j in jobs.values() for st in j.get("steps", [])]
        assert any("simulate_next_session.py --full" in r for r in runs)
        assert any("simulate_next_session.py --no-fit" in r for r in runs)

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
        assert src.index('os.environ["REFRESH_SLOT"] = "close"') < src.index('slot_steps(')
        assert src.index('os.environ["REFRESH_SLOT"] = "book"') < src.index('"market,capture,calibrate')
        main = inspect.getsource(sim.main)
        assert "run_record_problems(" in main
        assert "check_captures.py" in main

    def test_a_forced_failure_fails_the_run_and_the_alert_opens_the_issue(self, tmp_path):
        """FUND_GRADE_PLAN 0.4: the failure alert was tested only on a fake history
        (TestBotAlert), never after a run of the bot that failed. --force-fail runs both
        scheduled slots of the simulated next session in a copy of this checkout with
        Yahoo's SPX download raising: each must exit 1 on that outage with a complete run
        record, the book run (the one with the freeze step) must not be publishable, and
        the copy's bot_alert.py, run through its command line with a fake gh first on
        PATH answering from the two runs' own records, must open the one bot-failure
        issue, comment on one already open, leave it alone when a success came between,
        and not read GitHub on a manual run. About two minutes."""
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ci" / "simulate_next_session.py"),
                            "--force-fail", "--out", str(tmp_path)], capture_output=True, text=True,
                           timeout=900, check=False)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        report = (tmp_path / "report.md").read_text()
        assert "| bot runs (forced failure: Yahoo's SPX download raises in both slots) | 1 |" in report
        assert "close exit 1, book exit 1, publishable False" in report
        assert "simulated outage (--force-fail)" in report
        assert ("| failure alert (two failed in a row) | 0 | None | created issue "
                "https://github.invalid/issues/1 (streak 2; 4 gh calls) |") in report
        assert "| failure alert (an issue already open) | 0 | None | commented issue 7 (streak 2;" in report
        assert "| failure alert (a success between) | 0 | None | none (streak 1; 1 gh calls) |" in report
        assert "| failure alert (a manual run) | 0 | None | none (streak None; 0 gh calls) |" in report
        assert "PROBLEM" not in report
        st = json.loads((tmp_path / "status.json").read_text())
        assert st["exit_code"] == 1
        assert st["simulated"]["force_fail"] is True
        assert "freeze" in st["steps"]
        assert not (tmp_path / "tree").exists()                       # removed when it passes

    @staticmethod
    def _failed_runs(sim, root: Path) -> tuple[dict, dict]:
        """Two scheduled runs failed on the forced outage, as the slots record them: the
        close run, then the book run (with the freeze step, not publishable)."""
        (root / "data_store" / "runs").mkdir(parents=True)
        why = f"SPX OHLCV download failed: {sim.OUTAGE}"
        sts = {}
        for slot, utc, steps in (("close", "2026-11-02T23:07:00+00:00", ["market", "labels"]),
                                 ("book", "2026-11-03T03:30:00+00:00",
                                  ["market", "capture", "calibrate", "freeze"])):
            sts[slot] = {"run_utc": utc, "exit_code": 1, "publishable": False, "errors": [why],
                         "publish_blockers": [why], "steps": steps}
            (root / "data_store" / "runs" / f"run_{utc[:10]}_{slot}.json").write_text(json.dumps(
                {"slot": slot, "run_utc": utc, "complete": True, "exit_code": 1, "calibration": None}))
        return sts["close"], sts["book"]

    def test_the_forced_failure_check_fails_when_the_run_or_the_alert_does_not(self, sim, tmp_path):
        """The checks behind --force-fail, on hand-made failed runs: they pass with the
        repository's bot_alert.py and fail when the book run did not fail, failed on
        something else, could be published, or has no run record."""
        import shutil
        (tmp_path / "scripts" / "ci").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "ci" / "bot_alert.py", tmp_path / "scripts" / "ci" / "bot_alert.py")
        close, book = self._failed_runs(sim, tmp_path)
        work = tmp_path / "alert"
        problems, rows = sim.force_fail_problems(1, book, tmp_path, close, work)
        assert problems == []
        assert [r[3] for r in rows] == [
            "created issue https://github.invalid/issues/1 (streak 2; 4 gh calls)",
            "commented issue 7 (streak 2; 3 gh calls)", "none (streak 1; 1 gh calls)",
            "none (streak None; 0 gh calls)"]
        # the issue names both failed runs, and the fake gh, not the real one, was called
        log = work / "two_failed_in_a_row" / "gh_calls.jsonl"
        calls = [json.loads(ln) for ln in log.read_text().splitlines() if ln]
        body = calls[-1][calls[-1].index("--body") + 1]
        assert "actions/runs/9000000" in body
        assert "actions/runs/9000001" in body
        assert sim.force_fail_problems(0, {**book, "exit_code": 0}, tmp_path, close, work)[0][0].startswith(
            "the forced failure did not fail the book run")
        assert sim.force_fail_problems(1, {**book, "errors": ["freeze failed"]}, tmp_path, close, work)[0][
            0].startswith("the book run did not fail on the forced outage")
        published = sim.force_fail_problems(1, {**book, "publishable": True}, tmp_path, close, work)[0]
        assert published[0].startswith("the failed book run is publishable")
        (tmp_path / "data_store" / "runs" / "run_2026-11-03_book.json").unlink()
        assert sim.force_fail_problems(1, book, tmp_path, close, work)[0][0].startswith(
            "0 book run record(s)")

    @pytest.mark.parametrize("mutant", list(ALERT_MUTANTS))
    def test_the_forced_failure_check_catches_a_wrong_alert(self, sim, tmp_path, mutant):
        """Each mistake of bot_alert.py in ALERT_MUTANTS, in a copy of the script,
        makes the forced-failure check report a problem. Two of the review's are left
        out as they change nothing gh can show: counting manual runs (the history is
        read with --event schedule, so none is listed) and counting the run itself (it
        is in progress while its last step runs)."""
        (tmp_path / "scripts" / "ci").mkdir(parents=True)
        src = (ROOT / "scripts" / "ci" / "bot_alert.py").read_text()
        old, new = ALERT_MUTANTS[mutant]
        assert src.count(old) == 1, mutant
        (tmp_path / "scripts" / "ci" / "bot_alert.py").write_text(src.replace(old, new))
        close, book = self._failed_runs(sim, tmp_path)
        problems, _ = sim.force_fail_problems(1, book, tmp_path, close, tmp_path / "alert")
        assert problems, mutant

    def test_the_close_slot_runs_the_workflows_steps_and_must_store_the_closing_book(self, sim, tmp_path):
        """FUND_GRADE_PLAN 0.8: the simulation ran the close slot as 'market,labels'
        written into the driver, so a close-slot step the workflow added was never run
        before merge. It now reads the close slot's steps from the workflow (with the
        closebook step) and fails unless the close slot stored the closing books, the
        book slot captured the overnight ones (after a Friday, when the feed still
        shows the closing book, found the close slot's stored instead), the forward
        ledger traded on the book slot's book and the session's surface was built from
        the closing book. Checked here on the status files and surface.json the
        simulated slots write (book_problems, which main applies); the driver that
        writes them runs only in the simulation itself (CI's simulate job)."""
        steps = sim.slot_steps(ROOT / ".github" / "workflows" / "daily_refresh.yml", "close")
        assert "closebook" in steps.split(",")

        s, d = "2026-10-29", "/x/data_store/options/cboe"
        status = tmp_path / "status.json"

        def slot(status_, state, stamp, ledger=None):
            caps = {u: {"status": status_, "file": f"{d}/{u.lower()}_{s}_{stamp}.parquet",
                        **({"book_state": state} if status_ == "captured" else {})} for u in ("SPX", "VIX")}
            return {"capture": {"status": status_, "captures": caps},
                    **({"forward_ledger": {"book_file": f"options/cboe/{ledger}"}} if ledger else {})}

        def problems(close, book, feed_state, chain):
            """book_problems on the files the simulated run would leave."""
            sim_rec = {"session": s, "book_slot_state": feed_state}
            status.write_text(json.dumps({**book, "simulated": sim_rec}))
            close_file = status.with_name("status_close.json")
            close_file.unlink(missing_ok=True)
            if close is not None:
                close_file.write_text(json.dumps(close))
            sj = tmp_path / "tree" / "data_store" / "surface" / s / "surface.json"
            sj.parent.mkdir(parents=True, exist_ok=True)
            sj.write_text(json.dumps({"inputs": {"chain": chain}}))
            return sim.book_problems(tmp_path / "tree", status)

        close = slot("captured", "closing", "220700Z")
        surface = f"data_store/options/cboe/spx_{s}_220700Z.parquet"
        # a Monday-Thursday session: the book slot captures the overnight books
        gth = slot("captured", "gth", "023000Z", ledger=f"spx_{s}_023000Z.parquet")
        assert problems(close, gth, "gth", surface) == []
        # after a Friday the feed still shows the closing book: found stored, not fetched again
        fri = slot("exists", None, "220700Z", ledger=f"spx_{s}_220700Z.parquet")
        assert problems(close, fri, "closing", surface) == []
        assert "want the gth books captured" in problems(close, fri, "gth", surface)[0]
        assert "want the close slot's books found stored" in problems(close, gth, "closing", surface)[0]
        # the close slot stored nothing (no status file), the ledger or the surface on the wrong book
        assert "close slot capture" in problems(None, gth, "gth", surface)[0]
        other = slot("captured", "gth", "023000Z", ledger="spx_x.parquet")
        assert problems(close, other, "gth", surface) == [
            ("the forward ledger traded on options/cboe/spx_x.parquet, not the book slot's "
             f"spx_{s}_023000Z.parquet")]
        yahoo = f"data_store/options/spx_{s}.parquet"
        assert problems(close, gth, "gth", yahoo) == [
            (f"the session's surface was built from {yahoo}, not its closing book "
             f"spx_{s}_220700Z.parquet")]

    def test_a_replayed_book_has_the_state_its_quote_time_gives(self, sim):
        """The simulated 2026-10-30 session (a Friday) failed the data gate: the replay
        typed every book 'gth', but 22:15 ET on a Friday is the closing book (Monday's
        GTH opens on Sunday evening), and test_cboe_downloader checks each committed
        capture's state against book_state_at."""
        from datetime import UTC, datetime

        import pandas as pd

        from joint_vol_calibration.data import cboe_downloader as cb
        from joint_vol_calibration.data import option_chain as oc
        meta = {"underlying_price": 7000.0, "url": "u", "sha256": "0" * 64, "n_bytes": 1}
        frame = pd.DataFrame({"snapshot_date": ["2026-10-01"], "quote_time": ["x"]})
        states = {}
        for session in ("2026-10-29", "2026-10-30"):                  # a Thursday, a Friday
            now = datetime.fromisoformat(f"{session}T22:30:00").replace(tzinfo=oc.NY_TZ).astimezone(UTC)
            cap = sim.replayed_capture(cb, "SPX", frame, meta, session, now)
            quoted = datetime.fromisoformat(cap.quote_time)
            assert cap.book_state == cb.book_state_at(quoted, session)
            assert list(cap.frame["quote_time"]) == [cap.quote_time]
            assert list(cap.frame["snapshot_date"]) == [session]
            states[session] = cap.book_state
        assert states == {"2026-10-29": "gth", "2026-10-30": "closing"}

    def test_the_close_slots_replay_is_a_different_listing_from_the_book_slots(self, sim):
        """The simulated session's closing and overnight books were two replays of one
        committed file, with the same line counts, so the presentation gate's check of
        the Overview's books passed on a weekday session here and failed in the bot.
        The close slot's replay now leaves out the furthest expiry's lines."""
        import pandas as pd

        from joint_vol_calibration.config import DATA_DIR
        books = sorted((DATA_DIR / "options" / "cboe").glob("spx_*.parquet"))
        if not books:
            pytest.skip("no committed Cboe book")
        frame = pd.read_parquet(books[-1])
        closing = sim.closing_listing(frame)
        assert 0 < len(closing) < len(frame)
        assert closing["expiry"].nunique() == frame["expiry"].nunique() - 1
        assert closing["expiry"].max() < frame["expiry"].max()

    def test_the_driver_prices_book_and_vix_indices_with_the_moved_parameters(self, sim):
        import inspect
        src = inspect.getsource(sim.drive)
        assert 'p = simulated_params(cal["params"])' in src
        assert "p = cal[" not in src.replace('p = simulated_params(cal["params"])', "")
        assert "noisy_mid(" in src
        assert 'heston_vix_term_index(p["kappa"]' in src                # the VIX leg of the same p
        main = inspect.getsource(sim.main)
        assert "outcome_problems(" in main                              # whose checks follow
        checks = inspect.getsource(sim.outcome_problems)
        assert "MIN_PARAM_MOVE" in checks
        assert "param_moves(" in checks


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
        # the VX settlements' partitions (FUND_GRADE_PLAN 1.12) apart from the other market tables
        assert growth.kind_of("data_store/market/vix_futures_daily/vix_futures_daily_2026.parquet") == \
            "VX futures partitions"
        assert growth.kind_of("data_store/market/spx_ohlcv/spx_ohlcv_2026.parquet") == "market partitions"
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


# ── recompute_audit.py (tests.yml and the bot's presentation gate) ────────────

def _recompute(monkeypatch):
    """scripts/ci/recompute (the package) importable as ``recompute``."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts" / "ci"))
    for name in [m for m in sys.modules if m == "recompute" or m.startswith("recompute.")]:
        monkeypatch.delitem(sys.modules, name)
    import recompute.core
    import recompute.data
    return recompute


class TestRecomputeAudit:
    def test_no_module_imports_the_code_it_audits(self):
        """Tier 0.1: the audit recomputes by an independent path. A check that called
        the dashboard's functions would only test that the code agrees with itself."""
        import ast

        audited = {p.stem for p in (ROOT / "dashboard").glob("*.py")} | {"joint_vol_calibration", "dashboard"}
        files = [*sorted((ROOT / "scripts" / "ci" / "recompute").glob("*.py")),
                 ROOT / "scripts" / "ci" / "recompute_audit.py"]
        assert len(files) >= 5
        bad = []
        for f in files:
            for node in ast.walk(ast.parse(f.read_text(), str(f))):
                names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level
                         else [])
                bad += [f"{f.name}: {n}" for n in names if n.split(".")[0] in audited]
            assert "spec_from_file_location" not in f.read_text(), f.name
        assert bad == []

    def test_a_number_passes_only_when_the_value_rounds_to_it(self, monkeypatch):
        core = _recompute(monkeypatch).core
        assert core.parse_num("−1,573.20 pts") == (-1573.2, 2)
        assert core.parse_num("▲ +0.70%") == (0.7, 2)
        assert core.parse_num("67th") == (67.0, 0)
        assert [x[0] for x in core.numbers_in("2026-10-26: 1.60 half-spreads, 31.0%")] == [1.6, 31.0]
        assert core.rounds_to(1.604999, 1.60, 2)
        assert core.rounds_to(1.5951, 1.60, 2)
        assert not core.rounds_to(1.6051, 1.60, 2)                  # one display unit off fails
        assert not core.rounds_to(None, 1.60, 2)
        assert core.rounds_to(0.0960, 9.60, 2, scale=100.0)

    def test_a_check_built_from_numpy_values_serialises_to_the_json_report(self, monkeypatch):
        """Found when the 2026-10-27 session first ran the unreplicated-VIX checks: a
        check whose verdict was a numpy comparison made --json, which CI uploads,
        raise 'Object of type bool is not JSON serializable'."""
        import numpy as np
        core = _recompute(monkeypatch).core
        c = core.Check("p", "panel", "what", np.float64(1.5), np.float64(1.5), np.float64(1.5) <= 2.0)
        assert json.loads(json.dumps(c.__dict__))["ok"] is True

    def test_a_missing_or_wrong_number_fails_and_is_never_skipped(self, monkeypatch):
        core = _recompute(monkeypatch).core
        from bs4 import BeautifulSoup
        soup = BeautifulSoup('<main><p id="a">ATMF 14.03%</p><p id="b">RR −4.80</p></main>', "lxml")
        a = core.Audit("t", soup)
        assert a.num("p", "ATMF", a.need("#a", "p"), 14.0349)
        assert not a.num("p", "RR", a.need("#b", "p"), -4.81)
        assert a.need("#c", "p") is None
        assert [c.ok for c in a.checks] == [True, False, False]
        cov = a.coverage()
        assert cov["total"] == 2
        assert cov["covered"] == 2

    def test_its_nyse_calendar_matches_the_published_closures(self, monkeypatch):
        data = _recompute(monkeypatch).data
        import datetime as dt
        published = {  # NYSE full-day closures 2022-2026 as the exchange lists them
            "2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30", "2022-06-20", "2022-07-04", "2022-09-05",
            "2022-11-24", "2022-12-26", "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29",
            "2023-06-19", "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25", "2024-01-01", "2024-01-15",
            "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28",
            "2024-12-25", "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19",
            "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
            "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25"}
        got = {str(d) for y in range(2022, 2027) for d in data.nyse_closures(y)}
        assert got == published
        assert data.sessions_between("2026-09-04", "2026-09-08") == 1    # Labor Day
        assert data.sessions_between("2026-10-23", "2026-10-26") == 1
        assert data.sessions_between("2026-10-26", "2026-10-26") == 0
        assert data.is_session(dt.date(2026, 10, 26))
        assert not data.is_session(dt.date(2026, 11, 26))

    def test_its_svi_delta_strike_inverts_the_forward_delta(self, monkeypatch):
        data = _recompute(monkeypatch).data
        sl = {"a": 0.002, "b": 0.03, "rho": -0.7, "m": 0.01, "sigma": 0.05, "T": 0.08}
        for d in (-0.10, -0.25, 0.25, 0.10):
            k = data.strike_at_delta(sl, d)
            want = d if d > 0 else 1 + d
            w = float(data.svi_total_variance(k, sl))
            assert data.fwd_call_delta(k, w) == pytest.approx(want, abs=1e-12)
        assert data.svi_vol(0.0, sl) ** 2 * sl["T"] == pytest.approx(float(data.svi_total_variance(0.0, sl)))

    def test_its_cboe_rate_reproduces_the_methodology_sample_from_the_record(self, monkeypatch, tmp_path):
        """Tier 0.5: the page names the rate as a cause of the unexplained gap only
        where the replication priced each term at Cboe's rate, and the audit checks
        that claim with its own code: Cboe's bounded spline on the Treasury par
        yields of the newest date before the session in the record. On the
        methodology's sample curve (the yields of 2022-09-26, terms 25 and 32 days
        on) it gives the published R1 = 0.031664% and R2 = 0.028797%."""
        import importlib

        import pandas as pd
        _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        data = importlib.import_module("recompute.data")
        pct = dict(zip(("1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                       (0.03, 0.02, 0.04, 0.05, 0.08, 0.11, 0.22, 0.59, 1.00, 1.37, 2.03, 2.21)))
        rows = [("2022-09-23", f"UST_PAR_{t}", v / 100.0 + 0.01) for t, v in pct.items()]
        rows += [("2022-09-26", f"UST_PAR_{t}", v / 100.0) for t, v in pct.items()]
        rows += [("2022-09-26", "UST_PAR_4M", 0.0005), ("2022-09-27", "UST_PAR_1M", 0.05)]
        part = tmp_path / "market" / "rates_daily"
        part.mkdir(parents=True)
        pd.DataFrame(rows, columns=["date", "series", "value"]).assign(source="treasury").to_parquet(
            part / "rates_daily_2022.parquet")
        day, curve = M.cmt_curve(data.Store(tmp_path), "2022-09-27")
        assert day == "2022-09-26"                     # the newest before the session, not its own
        assert curve == {M.CBOE_CMT_DAYS[t]: v / 100.0 for t, v in pct.items()}    # no 4-month point
        assert round(100.0 * M.cboe_cmt_rate(curve, 25.0), 6) == 0.031664
        assert round(100.0 * M.cboe_cmt_rate(curve, 32.0), 6) == 0.028797
        assert M.cmt_curve(data.Store(tmp_path), "2022-10-10") is None           # 14 days old

    def test_it_names_the_rate_for_an_index_on_the_parity_forward(self, monkeypatch, tmp_path):
        """Tier 0.5 review: the audit left the rate out of the causes of every
        index's unexplained gap once the single-strike terms' rates were Cboe's,
        but an index replicated on the parity forward discounts at its parity's
        own D (a rebuilt 2026-05-29: VIX3M and VIX6M). It now leaves the rate out
        only for indices whose two headline terms were each discounted at the rate
        it recomputes from the record's par yields."""
        import importlib
        import types

        import pandas as pd
        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        S = importlib.import_module("recompute.surface")
        pct = dict(zip(("1M", "2M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"),
                       (0.03, 0.02, 0.04, 0.05, 0.08, 0.11, 0.22, 0.59, 1.00, 1.37, 2.03, 2.21)))
        part = tmp_path / "market" / "rates_daily"
        part.mkdir(parents=True)
        pd.DataFrame([("2022-09-26", f"UST_PAR_{t}", v / 100.0) for t, v in pct.items()],
                     columns=["date", "series", "value"]).assign(source="treasury").to_parquet(
            part / "rates_daily_2022.parquet")
        curve = {M.CBOE_CMT_DAYS[t]: v / 100.0 for t, v in pct.items()}

        def term(expiry, days, forward, R=None):
            return {"expiry": expiry, "forward_method": forward,
                    "R": M.cboe_cmt_rate(curve, float(days)) if R is None else R}

        def index(method, near, nxt):
            return {"headline_method": method, method: {"ok": True, "near": near, "next": nxt}}

        ss = "cboe_single_strike"
        vix = index("cboe_rules", term("2022-10-21", 25, ss), term("2022-10-28", 32, ss))
        v3m = index("parity_forward", term("2022-12-16", 81, "parity_regression", 0.0081),
                    term("2023-01-20", 116, "parity_regression", 0.0093))
        v3m["cboe_rules"] = {"ok": True, "near": term("2022-12-16", 81, ss),
                             "next": term("2023-01-20", 116, ss)}
        vx = {"rate": {"method": S.CBOE_RATE_METHOD, "curve_date": "2022-09-26"},
              "indices": {"VIX": vix, "VIX3M": v3m}}
        ctx = types.SimpleNamespace(vx=vx, store=rc.data.Store(tmp_path), session="2022-09-27")
        a = rc.core.Audit("t", BeautifulSoup("", "lxml"))
        at = S.vix_rates(a, ctx)
        assert len(a.checks) == 2
        assert all(c.ok for c in a.checks)
        assert at == {"VIX"}
        T, St, R = S.CAUSE_TIMING, S.CAUSE_STALE, S.CAUSE_RATE
        assert S.vix_causes("book_quote_time", at, ["VIX"]) == [T, St]
        assert S.vix_causes("book_quote_time", at, ["VIX", "VIX3M"]) == [T, St, R]
        assert S.vix_causes("closing_book_close", at, ["VIX3M"]) == [St, R]
        # both 3-month terms on the single-strike forward at Cboe's rate: no rate cause
        v3m["headline_method"] = "cboe_rules"
        assert S.vix_rates(rc.core.Audit("t", BeautifulSoup("", "lxml")), ctx) == {"VIX", "VIX3M"}
        # a single-strike rate that is not Cboe's fails a check and vouches for no index
        vix["cboe_rules"]["next"]["R"] *= 1.01
        a = rc.core.Audit("t", BeautifulSoup("", "lxml"))
        assert S.vix_rates(a, ctx) == set()
        assert not all(c.ok for c in a.checks)
        # the record claims one rate for both terms: nothing to check, the rate is named
        vx["rate"] = {"method": "one_rate_for_both_terms"}
        assert S.vix_rates(rc.core.Audit("t", BeautifulSoup("", "lxml")), ctx) == set()

    def test_its_gap_sentence_reads_the_fitted_range_of_a_far_wing_in_line(self, monkeypatch):
        """Found freezing a rebuilt 2026-05-29: where the stored quotes beyond the
        fitted range price in line with the surface's wing, the page's gap sentence
        still names that range ('beyond the fitted K/F 0.70–1.30 price in line'),
        and the audit, expecting its numbers only beside a far-wing figure, failed
        the sentence (12 numbers printed, 10 recomputed)."""
        import importlib
        import types

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        sentence = ("Gap attribution (VIX, SVI-cal surface; gap = replicated − Cboe close): −0.30 = "
                    "missing-strike shortfall −0.24 + unexplained −0.05 (each part rounded on its own). "
                    "Missing-strike shortfall (replicated − full strip): what the replication lacks "
                    "without the strikes beyond the stored strip, valued on the surface’s wing.{wing} "
                    "Inside the fitted K/F 0.70–1.30, market minus surface is +0.01; with the far-wing "
                    "term it sets the replication against the surface and is not part of the gap. "
                    "Unexplained by the chain: −0.05, 18% of the gap (snapshot timing against the "
                    "16:15 ET print, stale quotes).")
        ctx = types.SimpleNamespace(vx={"deviations": [], "gap_notes": []})
        cases = {0.003: " The stored quotes beyond the fitted K/F 0.70–1.30 price in line with that wing.",
                 0.21: (" The stored quotes beyond the fitted K/F 0.70–1.30 price 0.21 index points above "
                        "that wing, so the shortfall likely understates what the strip lacks.")}
        for far, wing in cases.items():
            html = ('<section id="vix-replication"><details class="method"><ul class="prose"><li>'
                    + sentence.format(wing=wing) + "</li></ul></details></section>")
            a = rc.core.Audit("t", BeautifulSoup(html, "lxml"))
            split = S.GapSplit(gap=-0.2953, short=-0.2425, unexp=-0.0528, far=far, inside=0.0115,
                               fit=(0.70, 1.30))
            S.deviations(a, ctx, split, [S.CAUSE_TIMING, S.CAUSE_STALE])
            assert [c.label for c in a.checks if not c.ok] == [], far

    def test_it_rebuilds_the_overnight_books_quote_time_and_hours_after_the_close(
            self, monkeypatch, tmp_path):
        """FUND_GRADE_PLAN 0.8: a replication built from an overnight book says when the
        book was quoted and how many hours after the session's 16:15 ET print. The
        audit rebuilds both from surface.json's quote time and its own calendar, and,
        from report_version 6, requires the note exactly on an overnight book. Why no
        closing book was used (none stored, or it failed the quote-state gates) was
        copied through; the audit now counts the session's stored closing books by its
        own clock, checks surface.json's closing_books against that count and the
        note's reason clause against it."""
        import importlib
        import types

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        root = tmp_path / "data_store"
        (root / "options" / "cboe").mkdir(parents=True)
        store = rc.data.Store(root)
        none_stored = ("(no closing book of the session is stored: Cboe's delayed feed shows it only until "
                       "Global Trading Hours open at 20:15 ET)")
        failed_gates = "(the session's closing book did not pass the quote-state gates)"

        def note(quoted, hours, why=none_stored):
            return (f"{S.OVERNIGHT_NOTE} {why}, quoted at {quoted} ET, "
                    f"{hours} hours after the 16:15 ET print of 2026-10-29 it is compared with: the gap "
                    "mixes the replication’s error with the market’s move overnight.")

        def failed(notes, quote_time="2026-10-30T08:36:08Z", version=6, closing_books=None):
            inputs = {"source": "cboe", "quote_time": quote_time, "book_state": "gth",
                      **({"closing_books": closing_books} if closing_books is not None else {})}
            ctx = types.SimpleNamespace(vx={"report_version": version}, session="2026-10-29",
                                        sj={"inputs": inputs}, store=store)
            html = "".join(f"<li>{n}</li>" for n in notes)
            a = rc.core.Audit("t", BeautifulSoup(f"<ul>{html}</ul>", "lxml"))
            S.overnight_note(a, ctx, a.soup.select("li"))
            return [c.label for c in a.checks if not c.ok]

        # Fri 04:36:08 ET: 12 h 21 min after Thursday's 16:15 ET print
        assert failed([note("2026-10-30 04:36", "12.4")]) == []
        assert failed([note("2026-10-30 04:36", "12.3")]) == [
            "overnight-book note: hours from the Cboe close to the book's quote time"]
        assert failed([note("2026-10-30 04:46", "12.4")]) == [
            "overnight-book note: when the book was quoted (surface.json quote time, ET)"]
        shown = "overnight-book note shown (the chain is an overnight book by this audit's clock)"
        assert failed([]) == [shown]
        assert failed([], version=5) == []                  # written before the note existed
        # Thu 17:52 ET: the closing book, which has no such note
        assert failed([], quote_time="2026-10-29T21:52:00Z") == []
        assert failed([note("2026-10-29 17:52", "1.6")], quote_time="2026-10-29T21:52:00Z")[0] == shown
        # the reason clause and surface.json's count follow the stored records
        why = "overnight-book note: why no closing book was used (the stored records)"
        count = "surface.json closing_books = the session's stored closing books (this audit's clock)"
        assert failed([note("2026-10-30 04:36", "12.4", failed_gates)]) == [why]
        assert failed([note("2026-10-30 04:36", "12.4")], closing_books=0) == []
        (root / "options" / "cboe" / "spx_2026-10-29_220705Z.json").write_text(json.dumps(
            {"underlying": "SPX", "session": "2026-10-29", "quote_time": "2026-10-29T21:52:05Z"}))
        (root / "options" / "cboe" / "spx_2026-10-29_085136Z.json").write_text(json.dumps(
            {"underlying": "SPX", "session": "2026-10-29", "quote_time": "2026-10-30T08:36:08Z"}))
        assert failed([note("2026-10-30 04:36", "12.4", failed_gates)], closing_books=1) == []
        assert failed([note("2026-10-30 04:36", "12.4")], closing_books=0) == [count, why]

    def test_it_accepts_the_overnight_note_as_the_producer_writes_it(self, monkeypatch, tmp_path):
        """FUND_GRADE_PLAN 0.8: the audit's two reasons for building a session from its
        overnight book were written with a typographic apostrophe (Cboe’s, session’s);
        the producer writes a straight one and the page prints its note as written. So
        the first session built from an overnight book under report_version 6 (any
        weekday whose close slot stores no closing book) would fail the audit, and with
        it the deploy. The audit is now fed the note report._timing_note writes, by the
        producer's own clock, with no closing book stored and with one stored."""
        import html
        import importlib
        import types

        from bs4 import BeautifulSoup

        from joint_vol_calibration.surface import report as R
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        root = tmp_path / "data_store"
        (root / "options" / "cboe").mkdir(parents=True)
        session, quoted = "2026-10-29", "2026-10-30T08:36:08Z"   # Fri 04:36 ET: the overnight book

        def failed(closing_books):
            inputs = {"source": "cboe", "quote_time": quoted, "book_state": "gth",
                      "closing_books": closing_books}
            sq = types.SimpleNamespace(session=session, inputs=inputs,
                                       clean_diagnostics={"quote_time": quoted})
            calc, basis = R._calc_time_ny(sq)
            note = R._timing_note(sq, calc, basis)
            assert note is not None
            assert note.startswith(S.OVERNIGHT_NOTE)
            ctx = types.SimpleNamespace(vx={"report_version": R.REPORT_VERSION}, session=session,
                                        sj={"inputs": inputs}, store=rc.data.Store(root))
            a = rc.core.Audit("t", BeautifulSoup(f"<ul><li>{html.escape(note)}</li></ul>", "lxml"))
            S.overnight_note(a, ctx, a.soup.select("li"))
            return [c.label for c in a.checks if not c.ok]

        assert failed(0) == []
        (root / "options" / "cboe" / f"spx_{session}_220705Z.json").write_text(json.dumps(
            {"underlying": "SPX", "session": session, "quote_time": "2026-10-29T21:52:05Z"}))
        assert failed(1) == []

    def test_it_rebuilds_the_close_slots_reason_for_storing_no_closing_book(self, monkeypatch, tmp_path):
        """Minor 2 of the 0.8 check: a session whose closing book the close slot fetched
        and refused (the quote-state gate) is built from its overnight book, and the note
        said no closing book is stored because of the feed's hours, while the run record
        says the book was caught and refused; the audit, counting stored books only,
        agreed. The note now gives the run record's reason, and the audit rebuilds it from
        data_store/runs itself: the newest run of the close slot's step alone that did not
        store the session's SPX book. It checks surface.json's record of it too."""
        import html
        import importlib
        import types

        from bs4 import BeautifulSoup

        from joint_vol_calibration.surface import report as R
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        M = importlib.import_module("recompute.surface_math")
        root = tmp_path / "data_store"
        (root / "options" / "cboe").mkdir(parents=True)
        (root / "runs").mkdir()
        store = rc.data.Store(root)
        session, quoted = "2026-10-29", "2026-10-30T08:36:08Z"   # Fri 04:36 ET: the overnight book
        why = "not a live book (31% of near-ATM lines two-sided): 52 near-ATM lines"

        def record(name, run_utc, steps, status, reason, sess=session):
            (root / "runs" / f"run_{name}.json").write_text(json.dumps({
                "schema": "run_ledger/1", "run_utc": run_utc, "steps": steps,
                "capture": {"session": sess, "captures": {"SPX": {"status": status, "reason": reason}}}}))

        close = ["market", "labels", "closebook"]
        record("a", "2026-10-29T22:07:05+00:00", close, "rejected", why)
        record("b", "2026-10-30T02:30:00+00:00", ["market", "capture", "calibrate", "freeze"], "rejected",
               "empty chain")                                       # the book slot's: not the close slot
        record("c", "2026-10-29T23:00:00+00:00", close, "skipped", "the feed shows the gth book")
        record("d", "2026-10-28T22:07:05+00:00", close, "failed", "OSError: x", sess="2026-10-28")
        assert M.closing_book_refusal(store, session) == why
        assert M.closing_book_refusal(store, "2026-10-28") == "OSError: x"
        assert M.closing_book_refusal(store, "2026-10-27") is None

        def failed(refusal, note=None, run=None, version=R.REPORT_VERSION):
            inputs = {"source": "cboe", "quote_time": quoted, "book_state": "gth", "closing_books": 0,
                      "closing_book_refusal": refusal, **({"closing_book_run": run} if run else {})}
            if note is None:                                         # the note the producer writes
                sq = types.SimpleNamespace(session=session, inputs=inputs,
                                           clean_diagnostics={"quote_time": quoted})
                note = R._timing_note(sq, *R._calc_time_ny(sq))
            ctx = types.SimpleNamespace(vx={"report_version": version}, session=session,
                                        sj={"inputs": inputs}, store=store)
            a = rc.core.Audit("t", BeautifulSoup(f"<ul><li>{html.escape(str(note))}</li></ul>", "lxml"))
            S.overnight_note(a, ctx, a.soup.select("li"))
            return [c.label for c in a.checks if not c.ok]

        reason = "overnight-book note: why no closing book was used (the stored records)"
        record_ = "surface.json closing_book_refusal = the close slot's run record (data_store/runs)"
        assert failed(why) == []
        # the timing reason where the record says the book was refused
        timing = ("The book is the overnight (Global Trading Hours) book " + S.OVERNIGHT_WHY_NONE
                  + ", quoted at 2026-10-30 04:36 ET, 12.4 hours after the 16:15 ET print of 2026-10-29 "
                  "it is compared with: the gap mixes the replication's error with the market's move "
                  "overnight.")
        assert failed(why, timing) == [reason]
        run_ = "surface.json closing_book_run = the close slot's run records (data_store/runs)"
        assert failed(None, run="skipped") == [record_, run_, reason]   # surface.json lost the reason
        (root / "runs" / "run_a.json").unlink()                      # no record of the refusal
        assert M.closing_book_run(store, session) == "skipped"       # run c: the feed showed another book
        assert failed(None, run="skipped") == []
        # w5 review, minor 1: report_version 6's timing reason (a claim about the feed's hours
        # that is not observed) is that version's; from 7 the reason is the run records'
        assert failed(None, timing, version=6) == []
        assert failed(None, timing, run="skipped") == [reason]
        (root / "runs" / "run_c.json").unlink()
        assert M.closing_book_run(store, session) == "none"
        assert failed(None, run="none") == []
        assert failed(None, run="skipped") == [run_, reason]
        record("e", "2026-10-29T22:07:05+00:00", close, "captured", None)   # caught, then not kept
        assert M.closing_book_run(store, session) == "captured"
        assert failed(None, run="captured") == []

    def test_it_rebuilds_the_heston_rows_fit_time_clause_from_the_fits_record(self, monkeypatch, tmp_path):
        """FUND_GRADE_PLAN 0.8: a session scored on its closing book whose Heston fit was
        made on a chain downloaded after that book gave way says so in the Heston row,
        with the download's UTC minute. The audit compared the row with scorecard.json
        only; it now rebuilds the clause from the fit's chain record and its own clock."""
        import importlib
        import pickle
        import types

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        M = importlib.import_module("recompute.surface_math")
        root = tmp_path / "data_store"
        (root / "calibrations").mkdir(parents=True)
        clause = ("; that chain was downloaded at 2026-10-30 08:51 UTC, after the closing book scored here "
                  "had given way to the overnight one, so part of its error here is the market's move "
                  "between the two")
        row = ("Heston at the stored model_version-2 joint SPX/VIX-index fit. The fit was made on the "
               "Yahoo chain")

        def failed(text, fetched="2026-10-30T08:51:33Z", book=M.BOOK_CLOSING, version=6):
            with open(root / "calibrations" / "joint_cal_2026-10-29.pkl", "wb") as fh:
                pickle.dump({"chain_diagnostics": {"source": "yfinance", "fetched_at": fetched}}, fh)
            ctx = types.SimpleNamespace(vx={"report_version": version}, session="2026-10-29", s=None,
                                        store=rc.data.Store(root), book=lambda s: book)
            a = rc.core.Audit("t", BeautifulSoup(f"<dd>{text}</dd>", "lxml"))
            S.heston_fit_time_note(a, ctx, a.soup.select_one("dd"))
            return [c.label for c in a.checks if not c.ok]

        shown = ("Heston row: fit-time clause shown (scored on the closing book, fit downloaded after it "
                 "gave way, by the fit's record and this audit's clock)")
        assert failed(row + clause) == []                                   # Fri 04:36 ET quotes
        assert failed(row) == [shown]
        assert failed(row + clause.replace("08:51", "08:41")) == [
            "Heston row: when the fit's chain was downloaded (its record, UTC)"]
        assert failed(row, fetched="2026-10-29T22:00:00Z") == []            # Thu 17:45 ET: still closing
        assert failed(row + clause, book=M.BOOK_OVERNIGHT) == [shown]        # scored on the overnight book
        assert failed(row, version=5) == []

    def test_it_checks_every_book_of_the_newest_session_on_the_overview(self, monkeypatch, tmp_path):
        """FUND_GRADE_PLAN 0.8: a weekday session can hold its closing book (the close
        slot's) and its overnight book (the book slot's). The audit read only the
        newest book per underlying, and only the public tree's records; it now
        requires every book of each underlying's newest session, oldest quote first,
        each named by its recorded state, counting the manifests of books held in
        the private store too."""
        import importlib

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        ov = importlib.import_module("recompute.overview")
        root = tmp_path / "data_store"
        for sub in ("cboe", "cboe_manifest"):
            (root / "options" / sub).mkdir(parents=True)
        s = "2026-11-02"                                  # synthetic records: any weekday session
        recs = {("cboe", "spx_2026-10-30_084019Z"): ("2026-10-30", "2026-10-30T23:49:50Z", "closing", 9),
                ("cboe", f"spx_{s}_000705Z"): (s, "2026-11-02T23:52:05Z", "closing", 7),
                ("cboe_manifest", f"spx_{s}_073204Z"): (s, "2026-11-03T07:16:06Z", "gth", 8)}
        for (sub, stem), (sess, qt, state, n) in recs.items():
            (root / "options" / sub / f"{stem}.json").write_text(json.dumps(
                {"underlying": "SPX", "session": sess, "quote_time": qt, "book_state": state,
                 "rows": n, "two_sided_rows": n - 1, "expiries": 2}))

        def failed(cell):
            a = rc.core.Audit("t", BeautifulSoup(f"<table><tr><td>{cell}</td></tr></table>", "lxml"))
            ov.check_pipeline_books(a, rc.data.Store(root), a.soup.select_one("td"))
            return [c.label for c in a.checks if not c.ok]

        # each book names its chain (tier 0.2): Cboe, its quote time, the share of its lines
        # two-sided (6 of 7; the manifest's book, held privately, by its record: 7 of 8)
        closing = ("7 lines (6 two-sided, 2 expiries), Cboe, quotes at 2026-11-02 23:52 UTC, 85.7% of its "
                   "lines two-sided (closing book)")
        overnight = ("8 lines (7 two-sided, 2 expiries), Cboe, quotes at 2026-11-03 07:16 UTC, 87.5% of its "
                     "lines two-sided (overnight book)")
        counts = "Cboe option books (counts from the stored quotes; each book's chain)"
        assert failed(f"SPX {s}: {closing} and {overnight}") == []
        assert failed(f"SPX {s}: {overnight}") == [            # the closing book left out
            counts, f"{counts}: dates", f"SPX {s}: every book of the session named",
            "SPX book 1: named by its recorded state (closing)", "SPX book 1: its chain (tier 0.2)"]
        assert "SPX book 2: named by its recorded state (gth)" in failed(
            f"SPX {s}: {closing} and {overnight.replace('(overnight book)', '(closing book)')}")
        # a book that names another share of its lines two-sided, or no chain at all
        assert "SPX book 2: its chain (tier 0.2)" in failed(f"SPX {s}: {closing} and "
                                                            f"{overnight.replace('87.5%', '97.5%')}")
        bare = closing.replace(", Cboe, quotes at 2026-11-02 23:52 UTC, 85.7% of its lines two-sided", "")
        assert "SPX book 1: its chain (tier 0.2)" in failed(f"SPX {s}: {bare} and {overnight}")

    def test_it_leaves_a_model_the_scorecard_did_not_score_off_the_page(self, monkeypatch):
        """A session's surface is built whatever its Heston fit did (CITSEC-05); with no
        fit stored, scorecard.json lists heston_v2 as not available and the page gives
        it no tile, row or description (the scorecard's note says why). The audit
        expected all four models, so a session with no fit (every Friday session the
        Saturday book slot cannot fit: Yahoo's book is empty then) failed five checks
        and the presentation gate. Found on a copy of the record with 2026-10-30 built
        from its closing book (FUND_GRADE_PLAN 0.8)."""
        import importlib
        import types

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        sc = {"models": [{"model": "svi"}, {"model": "heston_v2", "available": False, "reason": "no fit"}],
              "ranking_by_oos_median_abs_hs": [{"model": "svi"}]}
        assert S.Ctx._score_order(types.SimpleNamespace(sc=sc, s=types.SimpleNamespace(models=["svi"]))) == [
            "svi"]

        def failed(tile):
            html = (f'<div id="surface-scores"><div class="kpi"><span class="kpi-label">{tile}</span>'
                    "</div></div>")
            ctx = types.SimpleNamespace(sc=sc, score_models=[], vx={})
            a = rc.core.Audit("t", BeautifulSoup(html, "lxml"))
            S.score_tiles(a, ctx)
            return [c.label for c in a.checks if not c.ok]

        assert failed("SVI-cal · median |z|") == []
        assert failed("Heston · median |z|") == ["Heston · median |z|: no tile (not scored on this session)"]

    def test_it_checks_the_badge_names_a_model_the_scorecard_did_not_score(self, monkeypatch):
        """The Surface badge listed the scored models and said nothing of Heston on a
        session without a fit; it now adds 'Heston not scored on this session', linked
        to the scorecard. The audit expected exactly four badge items; it now expects
        one more per model the scorecard marks not available, with that text and link."""
        import importlib
        import types

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        monkeypatch.setattr(S, "chain_badge", lambda a, c: None)
        sc = {"models": [{"model": "svi"}, {"model": "heston_v2", "available": False, "reason": "no fit"}]}
        ctx = types.SimpleNamespace(sc=sc, s=types.SimpleNamespace(models=["svi"]), session="2026-10-30",
                                    exps=["2026-11-20"], mkt=[0, 1, 2])

        def failed(extra):
            html = ('<div class="page-head"><div class="page-badge">'
                    '<span class="nowrap">Session 2026-10-30</span><span class="nowrap">1 expiries</span>'
                    '<span class="nowrap">3 quotes</span>'
                    f'<span class="nowrap">models: SVI</span>{extra}</div></div>')
            a = rc.core.Audit("t", BeautifulSoup(html, "lxml"))
            S.page_head(a, ctx)
            return [c.label for c in a.checks if not c.ok]

        good = '<span class="nowrap"><a href="#scorecard">Heston not scored on this session</a></span>'
        assert failed(good) == []
        assert failed("") == ["badge items: count"]
        assert failed(good.replace("#scorecard", "#scores")) == [
            "Heston not scored: links to the scorecard and its note"]
        assert failed(good.replace("not scored", "unscored")) == [
            "Heston: not scored (scorecard.json available: false)"]

    def test_its_history_keeps_a_column_for_every_model_a_session_scores(self, monkeypatch):
        """A newest session without a Heston fit dropped Heston's column from the whole
        Surface history table, and the audit, taking its columns from the same session,
        agreed. Its columns are now the page session's models, then every other model a
        stored session's residual table holds, by name."""
        import importlib
        import types

        _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        ns = types.SimpleNamespace
        own = ns(models=["essvi", "svi", "svi_cal"])
        older = [ns(models=["essvi", "svi", "svi_cal", "heston_v2"]), ns(models=["svi", "zeta"])]
        ctx = ns(s=own, all=[*older, own])
        assert S.history_models(ctx) == ["essvi", "svi", "svi_cal", "heston_v2", "zeta"]
        assert S.history_models(ns(s=own, all=[own])) == ["essvi", "svi", "svi_cal"]
        assert "the model is not scored on that session" in S.HIST_DASH

    def test_its_history_note_gives_the_run_records_reason_for_a_session_without_a_fit(self, monkeypatch,
                                                                                       tmp_path):
        """The Surface history's note now says why a session has no Heston figure where the
        job's run records say (a rejection first, else the newest run that stored no fit),
        grouped by what happened and the recorded reason. The audit rebuilds that text from
        data_store/runs itself, so its sessions and numbers are checked, not only counted."""
        import importlib

        rc = _recompute(monkeypatch)
        S = importlib.import_module("recompute.surface")
        runs = tmp_path / "runs"
        runs.mkdir()
        for i, (s, status, why) in enumerate((
                ("2026-10-30", "skipped", "no SPX option chain returned"),
                ("2026-11-02", "rejected", "kappa pinned at bound (kappa=0.10)."),
                ("2026-11-02", "skipped", "a later slot found no book"),
                ("2026-11-06", "skipped", "no SPX option chain returned"),
                ("2026-11-09", "ok", None))):
            (runs / f"run_{i}.json").write_text(json.dumps({
                "schema": "run_ledger/1", "run_utc": f"2026-11-1{i}T00:00:00+00:00",
                "calibration": {"session": s, "status": status, "reason": why}}))
        (runs / "run_9.json").write_text("not json")
        store = rc.data.Store(tmp_path)
        assert S.unfitted_causes(store, ["2026-11-06", "2026-10-30", "2026-11-02", "2026-11-09"]) == (
            "the job did not fit 2026-10-30, 2026-11-06 — no SPX option chain returned; "
            "the gate rejected the fit of 2026-11-02 — kappa pinned at bound (kappa=0.10)")
        assert S.unfitted_causes(store, ["2026-11-09"]) == ""
        # a later run that found the database stale names the older session too: the
        # session's own run gives the reason
        for i, (exp, why) in enumerate((("2026-11-13", "no SPX option chain returned"),
                                        ("2026-11-16", "DB's latest session is stale"))):
            (runs / f"run_s{i}.json").write_text(json.dumps({
                "schema": "run_ledger/1", "run_utc": f"2026-11-2{i}T00:00:00+00:00", "expected_session": exp,
                "calibration": {"session": "2026-11-13", "status": "skipped", "reason": why}}))
        assert S.unfitted_causes(rc.data.Store(tmp_path), ["2026-11-13"]) == (
            "the job did not fit 2026-11-13 — no SPX option chain returned")
        assert S.HIST_UNFIT == "Where the job’s run records say why Heston has none: "

    def test_its_overview_cites_the_fit_and_names_a_newer_surface_session(self, monkeypatch):
        """The Overview's surface item and key result cite the headline fit's session (the
        surface and Heston on one session's quotes). When a newer surface session has no
        fit, the page names it ('the headline fit's session (Surface shows the newer …)'),
        so the audit expects that date after the cited one, and the clause only then."""
        import importlib

        _recompute(monkeypatch)
        ov = importlib.import_module("recompute.overview")

        def row(s, m, mhs):
            return {"session": s, "model": m, "n": 10, "mhs": mhs, "inside": 30.0, "butterfly": 0,
                    "calendar": 0}
        scores = [row("2026-10-29", "svi_cal", 1.74), row("2026-10-29", "heston_v2", 10.1),
                  row("2026-10-30", "svi_cal", 1.56)]
        sh = ov.surface_headline({"scores": scores, "head": {"session": "2026-10-29"}})
        assert (sh["sess"], sh["cur"]["mhs"], sh["heston"]["mhs"], sh["newer"]) == (
            "2026-10-29", 1.74, 10.1, "2026-10-30")
        sh = ov.surface_headline({"scores": scores[:2], "head": {"session": "2026-10-29"}})
        assert sh["newer"] is None
        assert ov.NEWER.startswith("the headline fit’s session (Surface shows the newer")

    def test_its_replication_item_is_the_session_surface_shows(self, monkeypatch, tmp_path):
        """The Overview's VIX-replication item links to Surface, which shows the newest
        surface session, and no Heston figure is paired with it, so the page cites that
        session's replication even when the headline fit's session is older. The audit
        expected the headline fit's session's gap, date and departures; it now expects the
        session Surface shows (the newest stored surface session)."""
        import importlib

        from bs4 import BeautifulSoup

        rc = _recompute(monkeypatch)
        ov = importlib.import_module("recompute.overview")
        for s, gap, devs in (("2026-10-29", 1.32, 6), ("2026-11-02", 1.92, 3)):
            d = tmp_path / "surface" / s
            d.mkdir(parents=True)
            (d / "residuals.parquet").write_text("")
            share = {"near_atm_two_sided_share": 1.0} if s == "2026-11-02" else {}
            (d / "surface.json").write_text(json.dumps({"inputs": {"source": "cboe",
                                                                   "quote_time": f"{s}T21:14:30Z", **share}}))
            (d / "vix_replication.json").write_text(json.dumps({
                "indices": {"VIX": {"gap_vs_official": gap, "replicated": 20.0 + gap, "official": 20.0}},
                "deviations": [f"departure {i}" for i in range(devs)]}))

        def row(s, m, mhs):
            return {"session": s, "model": m, "n": 10, "mhs": mhs, "inside": 30.0, "butterfly": 0,
                    "calendar": 0}
        f = {"scores": [row("2026-10-29", "svi_cal", 1.74), row("2026-10-29", "heston_v2", 10.1),
                        row("2026-11-02", "svi_cal", 1.56)],
             "head": {"session": "2026-10-29"}, "arc": [], "skew": None}
        a = rc.core.Audit("t", BeautifulSoup("", "lxml"))
        items = {link: parts for link, _, parts, _ in ov.works_items(a, rc.data.Store(tmp_path), f)}
        # each names its own session's chain and its two-sided share, or that none is recorded (0.2)
        assert items["surface.html#vix-replication"] == [
            1.92, "2026-11-02", 3.0,
            " Chain: Cboe, quotes at 2026-11-02 21:14 UTC, 100.0% two-sided near the money."]
        assert items["surface.html#scorecard"][2:] == [
            "2026-10-29", "2026-11-02",
            " Chain: Cboe, quotes at 2026-10-29 21:14 UTC, two-sided share not recorded."]

    def test_it_checks_the_overviews_stale_fit_banner_and_its_cause(self, monkeypatch, tmp_path):
        """A Friday without a fit followed by a skipped one left the headline fit 2 NYSE
        sessions old; the Overview then shows a stale banner the audit never checked, its
        5 numbers fell outside every verified element and the coverage bar failed the
        presentation gate (the no-fit simulation on 2026-11-02). The audit now checks
        each banner: shown exactly from 2 sessions, its age, level and dates, and for the
        fit the cause the page reads from the run records."""
        import importlib

        from bs4 import BeautifulSoup
        rc = _recompute(monkeypatch)
        ov = importlib.import_module("recompute.overview")
        (tmp_path / "runs").mkdir()
        for utc, sess in (("20261031T023000Z", "2026-10-30"), ("20261103T023000Z", "2026-11-02")):
            (tmp_path / "runs" / f"run_{utc}_book.json").write_text(json.dumps(
                {"schema": "run_ledger/1", "run_utc": utc, "slot": "book",
                 "calibration": {"session": sess, "status": "skipped",
                                 "reason": "no SPX option chain returned"}}))
        store = rc.data.Store(tmp_path)
        f = {"ref_page": "2026-11-02", "spx_date": "2026-11-02", "label_date": "2026-11-02",
             "head": {"session": "2026-10-29"}, "arc": []}
        cause = ("The book run did not fit 2026-11-02: no SPX option chain returned (2 runs after 2026-10-29 "
                 "tried a fit: 2 skipped); the 2026-10-29 fit is shown.")
        assert ov.stale_cause(store, f, "2026-10-29") == cause

        def failed(body, age="2", title="Heston calibration: 2 NYSE sessions old"):
            html = (f'<main><div class="banner banner-stale-warn" data-stale="calibration" data-age="{age}">'
                    f'<div class="banner-title">{title}</div>'
                    f'<div class="banner-body">{body}</div></div></main>')
            a = rc.core.Audit("t", BeautifulSoup(html, "lxml"))
            ov.check_stale_banners(a, store, f, {"date": "2026-11-02"})
            return [c.label for c in a.checks if not c.ok]

        lead = "Heston calibration as of 2026-10-29; the newest session the site has data for is 2026-11-02."
        assert failed(f"{lead} {cause}") == []
        assert failed(lead) == ["Heston calibration: body, with why the fit is older"]
        assert failed(f"{lead} {cause}", age="1") == ["Heston calibration: data-age"]
        assert failed(f"{lead} {cause}", title="Heston calibration: 3 NYSE sessions old") == [
            "Heston calibration: title"]
        fresh = {**f, "head": {"session": "2026-10-30"}}
        a = rc.core.Audit("t", BeautifulSoup('<div data-stale="calibration"></div>', "lxml"))
        ov.check_stale_banners(a, store, fresh, {"date": "2026-11-02"})
        assert [c.label for c in a.checks if not c.ok] == [
            "Heston calibration: banner (1 NYSE sessions old): shown"]

    def test_its_replication_instant_follows_the_books_record(self, monkeypatch):
        """Tier 0.5: the audit rebuilds the replication's calculation time from
        surface.json's record of the chain by its own calendar: a Cboe book at its
        quote time while its quotes are live (regular hours to the 16:15 cutoff,
        the next session's Global Trading Hours and the pause before its open), a
        closing book at the session's 16:15 ET close, a Yahoo chain at its T origin."""
        import importlib
        _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        cboe = {"source": "cboe", "t_origin": "2026-10-16T16:00:00-04:00"}
        cases = {"2026-10-19T02:30:00Z": ("2026-10-18T22:30", "book_quote_time"),    # Sun 22:30 ET, GTH
                 "2026-10-19T13:20:00Z": ("2026-10-19T09:20", "book_quote_time"),    # Mon 09:20 ET, pause
                 "2026-10-16T20:10:00Z": ("2026-10-16T16:10", "book_quote_time"),    # Fri 16:10 ET
                 "2026-10-17T00:30:00Z": ("2026-10-16T16:15", "closing_book_close"),  # Fri 20:30 ET
                 "2026-10-18T16:00:00Z": ("2026-10-16T16:15", "closing_book_close")}  # Sunday noon
        for qt, want in cases.items():
            assert M.replication_instant({**cboe, "quote_time": qt}, "2026-10-16") == want, qt
        # Monday 2026-10-12 to 13 (Columbus Day trades): GTH for the Tuesday
        assert M.replication_instant({**cboe, "quote_time": "2026-10-13T01:00:00Z"}, "2026-10-12") == (
            "2026-10-12T21:00", "book_quote_time")
        yahoo = {"source": "yfinance", "quote_time": None, "t_origin": "2026-10-16T16:00:00-04:00"}
        assert M.replication_instant(yahoo, "2026-10-16") == ("2026-10-16T16:00", "chain_t_origin")
        assert M.replication_instant({"source": "yfinance"}, "2026-10-16") is None

    def test_it_names_each_sessions_book_by_its_own_clock(self, monkeypatch):
        """FUND_GRADE_PLAN 0.8: the Surface page names the book each session's VIX gap
        was measured on and gives the gap's median and range per book. The audit
        classifies each session from surface.json's source and quote time by its own
        calendar (not the capture's recorded state): a Cboe book quoted after the
        session's close and before the next Global Trading Hours open is its closing
        book (a Friday's lasts the weekend), one inside those hours its overnight book,
        any other Cboe instant 'other', any other source a Yahoo chain."""
        import importlib
        _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        cases = {"2026-10-19T02:30:00Z": M.BOOK_OVERNIGHT,        # Sun 22:30 ET: Monday's GTH
                 "2026-10-17T00:30:00Z": M.BOOK_CLOSING,          # Fri 20:30 ET
                 "2026-10-18T16:00:00Z": M.BOOK_CLOSING,          # Sunday noon
                 "2026-10-16T21:55:00Z": M.BOOK_CLOSING,          # Fri 17:55 ET, the close slot
                 "2026-10-16T20:10:00Z": M.BOOK_OTHER,            # Fri 16:10 ET, before the cutoff
                 "2026-10-19T13:20:00Z": M.BOOK_OTHER}            # Mon 09:20 ET, the pause
        for qt, want in cases.items():
            assert M.replication_book({"source": "cboe", "quote_time": qt}, "2026-10-16") == want, qt
        # a weekday: the closing book until 20:15 ET, the overnight one from then
        assert M.replication_book({"source": "cboe", "quote_time": "2026-10-28T00:10:00Z"},
                                  "2026-10-27") == M.BOOK_CLOSING            # Tue 20:10 ET
        assert M.replication_book({"source": "cboe", "quote_time": "2026-10-28T00:20:00Z"},
                                  "2026-10-27") == M.BOOK_OVERNIGHT          # Tue 20:20 ET
        assert M.replication_book({}, "2026-10-26") is None

    def test_it_names_a_closing_book_by_its_quote_time_not_its_underlyings_last_trade(self, monkeypatch):
        """w5w2 check, F2 and F3: the audit took a Cboe book quoted after the close and
        before Global Trading Hours open for the closing book only when surface.json's
        underlying last trade was at or before the close. That check tells nothing
        apart (Cboe stamps the VIX close 16:15:01; SPX overnight books record the
        session's 16:14:59, as its closing book does), and the producer no longer
        makes it: the audit's clock reads the quote time alone, whatever last trade a
        record holds."""
        import importlib
        _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        rec = {"source": "cboe", "quote_time": "2026-10-30T23:49:50Z"}               # Fri 19:49 ET
        for last in (None, "2026-10-30T16:14:59", "2026-10-30T16:15:01", "2026-10-30T16:40:00",
                     "2026-10-29T16:14:59"):
            assert M.replication_book({**rec, "underlying_last_trade_time": last}, "2026-10-30") == (
                M.BOOK_CLOSING), last
        overnight = {"source": "cboe", "quote_time": "2026-10-28T03:15:00Z"}         # Tue 23:15 ET
        assert M.replication_book({**overnight, "underlying_last_trade_time": "2026-10-27T16:14:59"},
                                  "2026-10-27") == M.BOOK_OVERNIGHT
        assert not hasattr(M, "last_trade_by_close")

    def test_it_names_a_yahoo_download_by_when_it_was_fetched(self, monkeypatch):
        """w5 review, major 3: the Surface page put the bot's Yahoo download of 2026-11-02,
        fetched at 03:30 UTC in the next session's overnight hours, in one 'Yahoo chain'
        row with the C23 close snapshots. A Yahoo chain records no quote time, so the
        audit classifies it by when it was downloaded, by its own calendar, as it does a
        Cboe book by its quote time: inside the next session's Global Trading Hours an
        overnight download, any other instant (or none recorded) a Yahoo chain. The
        instant is a pre-QUANT-01 chain's quote time (the download instant), else its
        fetch time, from inputs or universe."""
        import importlib
        _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        cases = {"2026-11-03T03:30:03Z": M.BOOK_YAHOO_OVERNIGHT,   # Mon 22:30 ET: Tuesday's GTH
                 "2026-11-02T22:00:00Z": M.BOOK_YAHOO,             # Mon 17:00 ET, after the close
                 "2026-11-02T18:00:00Z": M.BOOK_YAHOO,             # Mon 13:00 ET, the session itself
                 "2026-11-03T14:20:00Z": M.BOOK_YAHOO}             # Tue 09:20 ET, the pause
        for at, want in cases.items():
            assert M.replication_book({"source": "yfinance", "fetched_at": at}, "2026-11-02") == want, at
            assert M.replication_book({"source": "yfinance"}, "2026-11-02", {"fetched_at": at}) == want, at
        # a Friday's: Saturday morning is the weekend, Sunday 21:01 ET Monday's GTH
        assert M.replication_book({"source": "yfinance", "fetched_at": "2026-10-31T08:50:33Z"},
                                  "2026-10-30") == M.BOOK_YAHOO
        assert M.replication_book({"source": "yfinance", "fetched_at": "2026-11-02T02:01:00Z"},
                                  "2026-10-30") == M.BOOK_YAHOO_OVERNIGHT
        # a pre-QUANT-01 chain stamped its download instant (Tue 01:26 ET) as its quote time
        assert M.replication_book({"source": "yfinance", "quote_time": "2026-10-27T05:26:50Z"},
                                  "2026-10-26") == M.BOOK_YAHOO_OVERNIGHT
        assert M.replication_book({"source": "yfinance"}, "2026-10-26") == M.BOOK_YAHOO

    def test_it_checks_the_book_the_overviews_replication_finding_names(self, monkeypatch, tmp_path):
        """w5 review, major 3: the Overview's 'VIX replicated from the stored chain'
        finding gave the gap without saying whether it was taken on a closing book (the
        replication's own error) or an overnight one (which also holds the overnight
        move). The audit classifies the session Surface shows by its own clock and
        requires that book's sentence in the finding, and no other book's."""
        import importlib

        from bs4 import BeautifulSoup

        rc = _recompute(monkeypatch)
        ov = importlib.import_module("recompute.overview")
        M = importlib.import_module("recompute.surface_math")
        d = tmp_path / "surface" / "2026-11-02"
        d.mkdir(parents=True)
        (d / "residuals.parquet").write_text("")
        (d / "surface.json").write_text(json.dumps({"inputs": {"source": "yfinance",
                                                               "fetched_at": "2026-11-03T03:30:03Z"}}))
        assert set(ov.BOOK_BUILT) == set(M.BOOK_ORDER)

        def failed(sentence: str) -> list[str]:
            html = ('<ul id="find-list-works"><li class="find-item"><a href="surface.html#vix-replication">'
                    f"VIX replicated from the stored chain</a>: +1.92 index pts. {sentence}</li></ul>")
            a = rc.core.Audit("t", BeautifulSoup(html, "lxml"))
            ov.check_replication_book(a, rc.data.Store(tmp_path))
            assert len(a.checks) == len(ov.BOOK_BUILT)
            return [c.label for c in a.checks if not c.ok]
        assert failed(ov.BOOK_BUILT[M.BOOK_YAHOO_OVERNIGHT]) == []
        assert failed(ov.BOOK_BUILT[M.BOOK_YAHOO])            # the close-snapshot sentence is wrong here
        assert failed("")                                     # and naming none is too

    def test_its_calendar_times_an_early_closes_closing_book_at_1315(self, monkeypatch):
        """Tier 0.5 review: on an NYSE early close SPX options and the VIX close at
        13:15 ET. The audit's own calendar holds the exchange's published early
        closes; a closing book of 2026-11-27 (or one quoted between 13:15 and 16:15)
        is calculated at 13:15, and the Cboe closes print then."""
        import datetime as dt
        import importlib
        rc = _recompute(monkeypatch)
        M = importlib.import_module("recompute.surface_math")
        published = {
            "2018-07-03", "2018-11-23", "2018-12-24", "2019-07-03", "2019-11-29", "2019-12-24",
            "2020-11-27", "2020-12-24", "2021-11-26", "2022-11-25", "2023-07-03", "2023-11-24",
            "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24",
            "2026-11-27", "2026-12-24"}
        assert {str(d) for y in range(2018, 2027) for d in rc.data.nyse_early_closes(y)} == published
        cboe = {"source": "cboe", "t_origin": "2026-11-27T16:00:00-05:00"}
        cases = {"2026-11-28T02:30:00Z": ("2026-11-27T13:15", "closing_book_close"),   # Fri 21:30 EST
                 "2026-11-27T19:00:00Z": ("2026-11-27T13:15", "closing_book_close"),   # 14:00, after 13:15
                 "2026-11-27T17:00:00Z": ("2026-11-27T12:00", "book_quote_time"),      # 12:00, live
                 "2026-11-30T02:00:00Z": ("2026-11-29T21:00", "book_quote_time")}      # Sun 21:00, GTH
        for qt, want in cases.items():
            assert M.replication_instant({**cboe, "quote_time": qt}, "2026-11-27") == want, qt
        assert M.cboe_close_time("2026-11-27") == dt.time(13, 15)
        assert M.cboe_close_time("2026-10-30") == dt.time(16, 15)

    def test_ci_runs_it_on_the_rendered_site(self):
        wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text()
        step = wf[wf.index("- name: Recompute audit"):]
        step = step[:step.index("\n      - name:", 10)]
        assert 'scripts/ci/recompute_audit.py "$RUNNER_TEMP/site"' in step
        assert "GITHUB_STEP_SUMMARY" in step
        assert "|| true" not in step                                  # a mismatch fails the job

    def test_its_record_view_applies_the_committed_record(self, monkeypatch, tmp_path):
        """The pages render from the cache brought up to the committed record
        (app.record_db); the audit builds the same view by its own code, so a
        cache that lags the record does not make the audit read stale rows."""
        import sqlite3

        import pandas as pd
        data = _recompute(monkeypatch).data
        root = tmp_path / "data_store"
        (root / "market" / "spx_ohlcv").mkdir(parents=True)
        con = sqlite3.connect(root / "vol_system.db")
        con.execute("CREATE TABLE spx_ohlcv (date TEXT, open REAL, high REAL, low REAL, close REAL, "
                    "volume INTEGER, log_return REAL)")
        con.executemany("INSERT INTO spx_ohlcv VALUES (?, 1, 1, 1, ?, 0, NULL)",
                        [("2026-10-22", 100.0), ("2026-10-23", 101.0)])
        con.commit()
        con.close()
        pd.DataFrame({"date": ["2026-10-23", "2026-10-26"], "open": [1.0, 1.0], "high": [1.0, 1.0],
                      "low": [1.0, 1.0], "close": [102.0, 103.0], "volume": [0, 0]}).to_parquet(
            root / "market" / "spx_ohlcv" / "spx_ohlcv_2026.parquet")
        spx = data.Store(root).spx()
        assert list(spx.index) == ["2026-10-22", "2026-10-23", "2026-10-26"]
        assert list(spx["close"]) == [100.0, 102.0, 103.0]               # the record's value wins
        assert spx["log_return"].iloc[-1] == pytest.approx(__import__("math").log(103 / 102))
        r = data.cache_vs_record(data.Store(root), "spx_ohlcv", ("close",))
        assert r["record_ahead"] == ["2026-10-26"]
        assert r["not_in_record"] == 1                                    # 2026-10-22 is only cached

    def test_its_record_view_holds_the_recorded_vx_settlements(self, monkeypatch, tmp_path):
        """FUND_GRADE_PLAN 1.12: the bot records VX settlements under
        market/vix_futures_daily/; the audit's view applies them on (date,
        expiry_date) as app.record_db does, so it recomputes the Overview's VX
        panel from the sessions the page shows, not from the cache's older ones."""
        import sqlite3

        import pandas as pd
        data = _recompute(monkeypatch).data
        root = tmp_path / "data_store"
        (root / "market" / "vix_futures_daily").mkdir(parents=True)
        cols = ("date", "expiry_date", "expiry_month", "symbol", "is_weekly", "open", "high", "low", "close",
                "settle", "volume", "open_interest", "days_to_expiry", "source")
        con = sqlite3.connect(root / "vol_system.db")
        con.execute(f"CREATE TABLE vix_futures_daily ({', '.join(cols)})")
        con.executemany("INSERT INTO vix_futures_daily VALUES "
                        "(?, ?, ?, NULL, 0, 1, 1, 1, 1, ?, 5, 5, 9, 'cboe')",
                        [("2026-10-09", "2026-10-21", "2026-10", 17.1186),
                         ("2026-10-09", "2026-11-18", "2026-11", 17.8517)])
        con.commit()
        con.close()
        pd.DataFrame({"date": ["2026-10-09", "2026-10-28"], "expiry_date": ["2026-11-18", "2026-11-18"],
                      "expiry_month": ["2026-11", "2026-11"], "symbol": ["VX/X6", "VX/X6"],
                      "is_weekly": [0, 0], "settle": [17.8500, 18.5982], "days_to_expiry": [40, 21],
                      "source": ["cboe_settlement"] * 2}).to_parquet(
            root / "market" / "vix_futures_daily" / "vix_futures_daily_2026.parquet")
        vx = data.Store(root).vx_futures()
        assert list(zip(vx["date"], vx["expiry_date"], vx["settle"])) == [
            ("2026-10-09", "2026-10-21", 17.1186),
            ("2026-10-09", "2026-11-18", 17.85),                  # the record's value wins
            ("2026-10-28", "2026-11-18", 18.5982)]                # the record's own session


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


# ── page_budget.py (CI's a11y job) ────────────────────────────────────────────

def _surface_page(n: int, pad: int) -> str:
    tab = ('<button class="btn" type="button" role="tab" aria-selected="false" '
           'aria-controls="slices-panel" data-value="2026-11-{:02d}">x</button>')
    body = "".join(tab.format(i % 28 + 1) for i in range(n))
    return "<html>" + body + " " * max(0, pad - len(body) - 13) + "</html>"


class TestPageBudget:
    """D2: a full Cboe book made the Surface page 1.67 MB. The a11y job holds it to
    450,000 bytes at 45 listed expiries plus 7,000 for each expiry past 45 and less
    5,500 for each one short (a flat budget below 45 let a 17-expiry page grow by
    150 kB unnoticed), counting the expiries on the page (its smile's tabs), so a
    day that lists more expiries moves the budget with the page; the bot's deploy
    never runs this. It prints the gzipped size at its stated level, ungated."""

    def test_the_budget_moves_with_the_expiry_count_on_the_page(self, monkeypatch, tmp_path):
        pb = _load("scripts/ci/page_budget.py", "page_budget_under_test", monkeypatch)
        assert [pb.budget(n) for n in (0, 17, 44, 45, 46, 80)] == [
            202_500, 296_000, 444_500, 450_000, 457_000, 695_000]
        for n, size, ok in ((45, 450_000, True), (45, 450_001, False), (50, 485_000, True),
                            (50, 485_001, False), (17, 296_000, True), (17, 296_001, False)):
            site = tmp_path / f"{n}-{size}"
            site.mkdir()
            (site / "surface.html").write_text(_surface_page(n, size), encoding="utf-8")
            assert len((site / "surface.html").read_bytes()) == size
            got, msg = pb.check(site)
            assert got is ok, msg
            assert f"at {n} listed expiries" in msg
            raw = (site / "surface.html").read_bytes()
            gz = len(gzip.compress(raw, compresslevel=pb.GZIP_LEVEL, mtime=0))
            assert f"{gz:,} bytes gzipped at level {pb.GZIP_LEVEL}" in msg, msg
            assert pb.main([str(site)]) == (0 if ok else 1)

    def test_the_budget_grows_with_the_default_expirys_quotes_and_the_stored_sessions(self, monkeypatch,
                                                                                    tmp_path):
        """D2 check: four compared sessions and a 30-day default expiry of 582 quotes
        (2026-12-31 on the 2026-10-28 book) put the 45-expiry page at 454,224 bytes, over
        a budget that counted only expiries, and the a11y job would have gone red about
        2026-12-01 with no markup change. The default expiry's quotes (drawn from the
        page) and the "Every session" rows grow with the data, so the budget grows
        with them; the markup's own weight is still held to the expiry formula."""
        pb = _load("scripts/ci/page_budget.py", "page_budget_under_test", monkeypatch)
        assert pb.budget(45) == pb.budget(45, pb.DEFAULT_QUOTES, pb.HISTORY_ROWS) == 450_000
        assert pb.budget(45, 582, 14) == (450_000 + pb.PER_DEFAULT_QUOTE * (582 - pb.DEFAULT_QUOTES)
                                          + pb.PER_HISTORY_ROW * (14 - pb.HISTORY_ROWS))
        status = ('<p class="exp-status js-only" id="slices-status" role="status" data-from="x">Expiry 16 of '
                  '45 · 2026-12-31 · Thu · 30 days · F 7,745.53 · 582 quotes · median half-spread 0.05 '
                  'vol pts</p>')
        rows = "".join(f"<tr><td>2026-10-{i + 1:02d}</td></tr>" for i in range(14))
        table = ('<table class="tbl" id="surface-history-table"><caption>c</caption><thead><tr><th>S</th>'
                 f'</tr></thead><tbody>{rows}</tbody></table>')
        for size, ok in ((454_224, True), (pb.budget(45, 582, 14) + 1, False)):
            page = _surface_page(45, size - len(status.encode()) - len(table))
            page = page.replace("</html>", status + table + "</html>")
            site = tmp_path / f"s{size}"
            site.mkdir()
            (site / "surface.html").write_text(page, encoding="utf-8")
            assert len((site / "surface.html").read_bytes()) == size
            assert pb.counts(page) == (45, 582, 14)
            got, msg = pb.check(site)
            assert got is ok, msg
            assert "a 582-quote default expiry and 14 stored sessions" in msg

    def test_its_tab_pattern_is_the_surface_templates(self, monkeypatch):
        pb = _load("scripts/ci/page_budget.py", "page_budget_under_test", monkeypatch)
        tpl = (ROOT / "dashboard" / "templates" / "surface.html").read_text()
        line = next(ln for ln in tpl.splitlines() if 'aria-controls="slices-panel"' in ln)
        rendered = (line.replace("{% if e.month_start %} mo{% endif %}", " mo")
                    .replace("{{ 'true' if e.expiry == d.default_exp else 'false' }}", "true")
                    .replace("{{ e.expiry }}", "2026-11-25"))
        assert len(pb.TAB.findall(rendered)) == 1

    def test_ci_runs_it_in_the_a11y_job(self):
        import yaml
        wf = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
        steps = wf["jobs"]["a11y"]["steps"]
        names = [s.get("name") for s in steps]
        i = next(j for j, n in enumerate(names) if str(n).startswith("Surface page weight"))
        assert names.index("Render the static site") < i
        assert steps[i]["run"] == 'python scripts/ci/page_budget.py "$RUNNER_TEMP/site"'
        assert "continue-on-error" not in steps[i]
