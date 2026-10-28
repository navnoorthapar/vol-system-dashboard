#!/usr/bin/env python3
"""
bot_alert.py: one GitHub issue when the daily bot's scheduled runs fail twice
in a row (FUND_GRADE_PLAN 0.4).

    python scripts/ci/bot_alert.py --run-id "$GITHUB_RUN_ID" --event "$GITHUB_EVENT_NAME" \
        [--workflow daily_refresh.yml] [--run-url URL] [--slot SLOT] [--dry-run]

The last step of .github/workflows/daily_refresh.yml runs it when the job failed
(`if: failure()`, `continue-on-error: true`: it comes after every commit and the
deploy and can never cost either). A failed run used to be a red mark in the
Actions tab and nothing else: the bot lost three overnight books in a row
(2026-10-19 to 10-21) before anyone looked.

  * Only scheduled runs count. A manual dispatch has someone watching it.
  * This run failed: that is why the step runs. The scheduled runs before it
    are read newest first (`gh run list --workflow <file> --event schedule`);
    a run still queued or running, or one that ended cancelled, skipped or
    neutral, says nothing about the bot and is passed over. Each failed one
    (failure, timed_out, startup_failure) adds one to the streak; the first
    success ends it.
  * A streak of ALERT_AFTER (2) or more comments on the open issue labelled
    LABEL (bot-failure), or opens one (the label is created if missing).
    There is never a second open issue: closing it re-arms the alert.

The weekly reproduction (.github/workflows/reproduce.yml) uses the same script
with its own label (reproduce-failure), every failure alerting (--alert-after 1
--all-events) and its report as the body (--body-file).

The GitHub CLI is the only dependency (GH_TOKEN; the workflow grants
`issues: write` and `actions: read`). The runner that calls it is injectable,
so the decision and the issue calls are tested on a fake run history
(joint_vol_calibration/tests/test_ci_tools.py::TestBotAlert). A `gh` call that
fails is retried with a pause; the script then exits 1 (the step is
continue-on-error: the alert failing is visible, never fatal).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

LABEL = "bot-failure"
ALERT_AFTER = 2                  # consecutive failed scheduled runs, this one included
HISTORY = 30                     # scheduled runs read back
FAILED = frozenset({"failure", "timed_out", "startup_failure"})
SUCCEEDED = frozenset({"success"})
RETRIES = 3
RETRY_SLEEP_S = 10.0

Runner = Callable[[Sequence[str]], str]


def gh(args: Sequence[str]) -> str:
    """Run the GitHub CLI; its stdout, or CalledProcessError."""
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def _retry(runner: Runner, args: Sequence[str], sleep: Callable[[float], None]) -> str:
    for attempt in range(1, RETRIES + 1):
        try:
            return runner(args)
        except subprocess.CalledProcessError as e:
            if attempt == RETRIES:
                raise
            print(f"gh {' '.join(args[:2])} failed (try {attempt} of {RETRIES}): {e.stderr or e}",
                  flush=True)
            sleep(RETRY_SLEEP_S * attempt)
    raise AssertionError("unreachable")


def failure_streak(runs: list[dict], run_id: str) -> tuple[int, list[dict]]:
    """(the streak, its earlier failed runs): this run (failed) plus each
    finished scheduled run before it that failed, back to the first success.
    ``runs`` is `gh run list --json databaseId,event,status,conclusion,...`,
    newest first."""
    earlier = []
    for r in runs:
        if str(r.get("databaseId")) == str(run_id) or r.get("event") != "schedule":
            continue
        if r.get("status") != "completed":
            continue
        c = r.get("conclusion")
        if c in SUCCEEDED:
            break
        if c in FAILED:
            earlier.append(r)
    return 1 + len(earlier), earlier


def issue_body(streak: int, run_url: str, slot: str, earlier: list[dict]) -> str:
    lines = [f"The daily bot's scheduled runs have failed {streak} times in a row.", "",
             f"- this run ({slot or 'scheduled'}): {run_url}"]
    lines += [f"- {r.get('createdAt', '')}: {r.get('url') or r.get('databaseId')} ({r.get('conclusion')})"
              for r in earlier]
    lines += ["", ("Each run's record is in data_store/runs/ when its capture commit landed "
                   "(the calibration outcome, the gate's reason, publish blockers); the job log has the "
                   "rest. HANDOFF.md 'Ship' step 6 says how to recover a run that failed its presentation "
                   "gate."),
              "", (f"Opened by scripts/ci/bot_alert.py; later failures comment here. Close this issue once "
                   f"the bot is green: the next {ALERT_AFTER} failures in a row open a new one.")]
    return "\n".join(lines)


MAX_BODY = 60_000                # GitHub's issue body limit is 65,536 characters


def alert(run_id: str, event: str, run_url: str, slot: str = "", workflow: str = "daily_refresh.yml",
          runner: Runner | None = None, sleep: Callable[[float], None] | None = None, dry_run: bool = False,
          label: str = LABEL, alert_after: int = ALERT_AFTER, all_events: bool = False,
          title: str | None = None, details: str | None = None) -> dict:
    """Decide and act; returns what was done ({action, streak, issue}).

    The defaults are the daily bot's (two failed scheduled runs in a row).
    Another workflow names its own label and threshold: the weekly
    reproduction (.github/workflows/reproduce.yml) alerts on every failure,
    a manual run's included (alert_after 1, all_events), with its report as
    ``details``. ``runner`` and ``sleep`` default to the GitHub CLI and
    time.sleep (looked up at call time, so a test can replace them)."""
    runner = runner or gh
    sleep = sleep or time.sleep
    if event != "schedule" and not all_events:
        return {"action": "none", "streak": None, "issue": None,
                "why": f"a {event or 'manual'} run: only scheduled runs count"}
    earlier: list[dict] = []
    streak = 1
    if alert_after > 1:
        runs = json.loads(_retry(runner, ["run", "list", "--workflow", workflow, "--event", "schedule",
                                          "--limit", str(HISTORY), "--json",
                                          "databaseId,event,status,conclusion,createdAt,url"], sleep) or "[]")
        streak, earlier = failure_streak(runs, run_id)
    out: dict = {"action": "none", "streak": streak, "issue": None}
    if streak < alert_after:
        out["why"] = f"{streak} failed scheduled run in a row (alert at {alert_after})"
        return out
    if details is None:
        body = issue_body(streak, run_url, slot, earlier)
    else:
        body = f"{title or workflow + ' failed'}: {run_url}\n\n{details}"
    body = body[:MAX_BODY]
    open_issues = json.loads(_retry(runner, ["issue", "list", "--label", label, "--state", "open",
                                             "--json", "number,url", "--limit", "10"], sleep) or "[]")
    if open_issues:
        n = min(int(i["number"]) for i in open_issues)
        out.update(action="commented", issue=n)
        if not dry_run:
            _retry(runner, ["issue", "comment", str(n), "--body", body], sleep)
        return out
    out["action"] = "created"
    if not dry_run:
        _retry(runner, ["label", "create", label, "--color", "B60205", "--force", "--description",
                        f"{workflow} failed (scripts/ci/bot_alert.py)"], sleep)
        title = title or f"Daily bot: {streak} scheduled runs failed in a row"
        made = _retry(runner, ["issue", "create", "--label", label, "--title", title, "--body", body], sleep)
        out["issue"] = made.strip().splitlines()[-1] if made.strip() else None
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--workflow", default="daily_refresh.yml")
    ap.add_argument("--run-url", default="")
    ap.add_argument("--slot", default="")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--alert-after", type=int, default=ALERT_AFTER,
                    help="consecutive failed scheduled runs (this one included) that raise the alert")
    ap.add_argument("--all-events", action="store_true", help="a manual run alerts too")
    ap.add_argument("--title", default=None, help="the issue title (default: the daily bot's)")
    ap.add_argument("--body-file", type=Path, default=None,
                    help="the issue body after the run link (default: the daily bot's text); a missing "
                         "file is said so in the body")
    ap.add_argument("--dry-run", action="store_true", help="decide and print, open or comment on nothing")
    args = ap.parse_args(argv)
    details = None
    if args.body_file is not None:
        details = (args.body_file.read_text() if args.body_file.is_file()
                   else f"(no report: {args.body_file} was not written; see the job log)")
    try:
        res = alert(args.run_id, args.event, args.run_url, args.slot, args.workflow, dry_run=args.dry_run,
                    label=args.label, alert_after=args.alert_after, all_events=args.all_events,
                    title=args.title, details=details)
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        detail = getattr(e, "stderr", "") or e
        print(f"::warning title=Failure alert::could not check or raise the failure alert: {detail}",
              flush=True)
        return 1
    print(json.dumps(res), flush=True)
    if res["action"] != "none":
        print(f"::warning title=Failure alert::{res['streak']} failed run(s) in a row: issue "
              f"{res['action']} ({res['issue']}, label {args.label})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
