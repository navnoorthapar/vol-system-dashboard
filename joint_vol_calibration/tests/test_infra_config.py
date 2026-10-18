"""
Regression guards for the deployment configuration (C22 ops findings).

Text checks on the workflow files and scripts, so they run without PyYAML or
network: each one pins a property whose absence once broke or misreported the
published site.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAILY = (ROOT / ".github" / "workflows" / "daily_refresh.yml").read_text()
TESTS = (ROOT / ".github" / "workflows" / "tests.yml").read_text()


def _crons(text):
    return re.findall(r'-\s*cron:\s*"([^"]+)"', text)


def test_daily_cron_is_split_by_data_source():
    crons = _crons(DAILY)
    assert "30 1 * * 2-6" in crons and "30 3 * * 2-6" in crons     # live Cboe book + retry
    close = [c for c in crons if c.split()[1] == "22"]
    assert close and close[0].split()[0] != "0"                    # off the top of the hour
    assert '"market,labels"' in DAILY and "market,calibrate,freeze" in DAILY


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


def test_redeploy_script_cannot_push_gh_pages():
    sh = (ROOT / "redeploy_site.sh").read_text()
    code = "\n".join(ln for ln in sh.splitlines() if not ln.lstrip().startswith("#"))
    assert "git push" not in code and "git init" not in code
    assert "gh workflow run" in code


def test_dead_procfile_is_gone():
    assert not (ROOT / "Procfile").exists()
    assert "gunicorn" not in (ROOT / "requirements.txt").read_text()
