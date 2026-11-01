#!/usr/bin/env python3
"""
deploy_streak.py: how many consecutive NYSE sessions the site was deployed for,
from the daily workflow's deploy commits on gh-pages (FUND_GRADE_PLAN 0.4).

    git log --format=%s <gh-pages> | python scripts/ci/deploy_streak.py --out .deploy-streak.json
    python scripts/ci/deploy_streak.py --log subjects.txt --out .deploy-streak.json

The deploy step of .github/workflows/daily_refresh.yml commits to gh-pages with
the message 'Deploy session <s> from <sha> (calibration ...; run <id>)'. The
sessions those subjects name are the sessions the site was published for. The
streak ends at the newest of them and counts back through NYSE sessions (the
project's one calendar, joint_vol_calibration.data.database: a weekend or a
holiday is not a session and does not break it); the first session with no
deploy commit ends it. Several deploys of one session (the book slot, its
retry, a manual run) count once.

The workflow fetches gh-pages' commit messages only (no trees, in a scratch
repository) and writes the JSON before the freeze, so the site the run renders
counts the deploys before it; the file is gitignored, never committed. The
Overview's pipeline panel prints 'Deployed (gh-pages): N consecutive NYSE
sessions, <first> -> <through>' from it (dashboard/app.py deploy_streak); a
local freeze has no file and no row.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "deploy_streak/1"
DEPLOY = re.compile(r"^Deploy session (\d{4}-\d{2}-\d{2}) from ")


def deployed_sessions(subjects: Iterable[str]) -> set[date]:
    """The sessions named by deploy-commit subjects (any other subject is ignored)."""
    out = set()
    for s in subjects:
        m = DEPLOY.match(s.strip())
        if m:
            try:
                out.add(date.fromisoformat(m.group(1)))
            except ValueError:
                continue
    return out


def _previous_session() -> Callable[[date], date]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from joint_vol_calibration.data import database as db
    return lambda d: date.fromisoformat(db.previous_session(d))


def streak(sessions: set[date], previous_session: Callable[[date], date] | None = None) -> dict | None:
    """{streak, first, through, deploys} of the run of consecutive NYSE sessions
    ending at the newest deployed one; None with no deploy at all."""
    if not sessions:
        return None
    prev = previous_session or _previous_session()
    through = max(sessions)
    first, n = through, 1
    while (p := prev(first)) in sessions:
        first, n = p, n + 1
    return {"streak": n, "first": first.isoformat(), "through": through.isoformat(), "deploys": len(sessions)}


def record(subjects: Iterable[str], now: datetime | None = None,
           previous_session: Callable[[date], date] | None = None) -> dict:
    subjects = list(subjects)
    res = streak(deployed_sessions(subjects), previous_session)
    return {"schema": SCHEMA, "computed_utc": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
            "source": "gh-pages deploy commits ('Deploy session <s> from ...')",
            "commits_read": len(subjects), **(res or {"streak": 0, "first": None, "through": None,
                                                       "deploys": 0})}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("--log", type=Path, default=None, help="commit subjects, one per line (default: stdin)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    text = args.log.read_text() if args.log else sys.stdin.read()
    rec = record(text.splitlines())
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    tmp.replace(args.out)
    print(f"{rec['streak']} consecutive NYSE session(s) deployed, {rec['first']} -> {rec['through']} "
          f"({rec['deploys']} deployed sessions in {rec['commits_read']} gh-pages commits)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
