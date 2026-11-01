#!/usr/bin/env python3
"""
pip_audit_gate.py: known vulnerabilities in the locked set fail CI unless a
reviewed, dated exception covers them (FUND_GRADE_PLAN 0.6).

    python scripts/ci/pip_audit_gate.py [--lock requirements-lock.txt]
        [--allowlist scripts/ci/pip_audit_allowlist.json] [--summary FILE]
    python scripts/ci/pip_audit_gate.py --report audit.json      # judge a saved report

The audit job used to run pip-audit with `|| true` and continue-on-error, so a
published advisory against a package in the job that pushes to main and
gh-pages changed nothing. Now:

  * every pin of the lock is audited at its public version (a local label such
    as torch's `+cpu` is dropped: the advisory database knows `torch==2.14.0`,
    and the CPU build carries the same code), without installing anything
    (`--no-deps --disable-pip`; the lock is complete);
  * a finding fails the job unless scripts/ci/pip_audit_allowlist.json has an
    entry for that package and that advisory (its id or any alias), with a
    reason and an expiry date no more than MAX_EXPIRY_DAYS away;
  * an expired or malformed entry fails the job even if nothing matches it,
    so an exception cannot outlive its review; an entry nothing matches any
    more is reported for removal;
  * pip-audit itself failing (the advisory service or PyPI unreachable: no
    JSON report) is retried with a growing pause, and fails the job only when
    every attempt failed. It never passes for want of an answer.

Exit status: 0 clean (or every finding covered), 1 an uncovered finding or a
bad allowlist entry, 2 pip-audit gave no report after every retry.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "requirements-lock.txt"
ALLOWLIST = ROOT / "scripts" / "ci" / "pip_audit_allowlist.json"
MAX_EXPIRY_DAYS = 180           # an exception is re-reviewed at least twice a year
RETRIES = 4                     # attempts at a pip-audit run that returns no report
RETRY_SLEEP_S = 15.0            # 15 s, 30 s, 45 s between them

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)==([^\s\\;]+)")


def normalise(name: str) -> str:
    """PEP 503 name normalisation (`Foo_Bar` and `foo-bar` are one package)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_pins(text: str) -> dict[str, str]:
    """``{normalised name: version}`` from a pip requirements lock, hashed or not."""
    pins: dict[str, str] = {}
    for line in text.splitlines():
        m = _PIN.match(line.strip())
        if m:
            pins[normalise(m.group(1))] = m.group(2)
    return pins


def public_version(version: str) -> str:
    """The version the advisory database knows: without a local label (`2.14.0+cpu`)."""
    return version.split("+", 1)[0]


def audit_input(pins: dict[str, str]) -> str:
    return "".join(f"{name}=={public_version(v)}\n" for name, v in sorted(pins.items()))


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    id: str
    aliases: tuple[str, ...]
    fix_versions: tuple[str, ...]

    def ids(self) -> set[str]:
        return {self.id, *self.aliases}


def findings(report: dict) -> tuple[list[Finding], list[tuple[str, str]]]:
    """The vulnerabilities in a pip-audit JSON report, and the dependencies it
    skipped (name, reason)."""
    found: list[Finding] = []
    skipped: list[tuple[str, str]] = []
    for dep in report.get("dependencies", []):
        name = normalise(str(dep.get("name", "")))
        if dep.get("skip_reason"):
            skipped.append((name, str(dep["skip_reason"])))
            continue
        found.extend(Finding(name, str(dep.get("version", "")), str(v["id"]),
                             tuple(str(a) for a in v.get("aliases", []) or []),
                             tuple(str(f) for f in v.get("fix_versions", []) or []))
                     for v in dep.get("vulns", []) or [])
    return found, skipped


@dataclass(frozen=True)
class Entry:
    id: str
    package: str
    reason: str
    expires: date


def load_allowlist(data: object, today: date) -> tuple[list[Entry], list[str]]:
    """The allowlist's entries, and a problem line for every entry that is
    malformed or expired (each fails the gate)."""
    problems: list[str] = []
    raw = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], ["the allowlist must be an object with an 'entries' list"]
    entries: list[Entry] = []
    for i, e in enumerate(raw):
        where = f"entry {i}"
        if not isinstance(e, dict):
            problems.append(f"{where}: not an object")
            continue
        missing = [k for k in ("id", "package", "reason", "expires") if not str(e.get(k) or "").strip()]
        if missing:
            problems.append(f"{where} ({e.get('id', '?')}): missing {', '.join(missing)}")
            continue
        where = f"{e['id']} ({e['package']})"
        if len(str(e["reason"]).strip()) < 20:
            problems.append(f"{where}: the reason must say why the finding does not apply or cannot be "
                            "fixed yet (20 characters at least)")
            continue
        try:
            expires = date.fromisoformat(str(e["expires"]))
        except ValueError:
            problems.append(f"{where}: expires {e['expires']!r} is not a YYYY-MM-DD date")
            continue
        if expires < today:
            problems.append(f"{where}: the exception expired on {expires.isoformat()}; review it "
                            "(fix the pin, or renew the entry with a new reason)")
            continue
        if (expires - today).days > MAX_EXPIRY_DAYS:
            problems.append(f"{where}: expires {expires.isoformat()} is more than {MAX_EXPIRY_DAYS} days "
                            "away; exceptions are reviewed at least twice a year")
            continue
        entries.append(Entry(str(e["id"]).strip(), normalise(str(e["package"])), str(e["reason"]).strip(),
                             expires))
    return entries, problems


@dataclass
class Verdict:
    uncovered: list[Finding]
    covered: list[tuple[Finding, Entry]]
    problems: list[str]
    unused: list[Entry]
    skipped: list[tuple[str, str]]

    @property
    def ok(self) -> bool:
        return not self.uncovered and not self.problems


def judge(report: dict, allowlist: object, today: date) -> Verdict:
    found, skipped = findings(report)
    entries, problems = load_allowlist(allowlist, today)
    uncovered, covered, used = [], [], set()
    for f in found:
        match = next((e for e in entries if e.package == f.package and e.id in f.ids()), None)
        if match is None:
            uncovered.append(f)
        else:
            covered.append((f, match))
            used.add(id(match))
    unused = [e for e in entries if id(e) not in used]
    return Verdict(uncovered, covered, problems, unused, skipped)


def report_lines(v: Verdict) -> list[str]:
    lines = ["## pip-audit gate (requirements-lock.txt)", ""]
    if v.ok:
        lines.append(f"OK: no uncovered advisory ({len(v.covered)} covered by the allowlist).")
    else:
        lines.append(f"FAILED: {len(v.uncovered)} uncovered advisory(ies), {len(v.problems)} allowlist "
                     "problem(s).")
    if v.uncovered or v.covered:
        lines += ["", "| package | version | advisory | aliases | fixed in | status |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {f.package} | {f.version} | {f.id} | {', '.join(f.aliases) or '-'} | "
                  f"{', '.join(f.fix_versions) or '-'} | **not covered** |" for f in v.uncovered]
        lines += [f"| {f.package} | {f.version} | {f.id} | {', '.join(f.aliases) or '-'} | "
                  f"{', '.join(f.fix_versions) or '-'} | allowed until {e.expires.isoformat()}: "
                  f"{e.reason} |" for f, e in v.covered]
    lines += [f"- allowlist: {p}" for p in v.problems]
    lines += [f"- allowlist entry {e.id} ({e.package}) matches no finding any more: remove it"
              for e in v.unused]
    lines += [f"- not audited: {name} ({why})" for name, why in v.skipped]
    return lines


Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, check=False)


def run_audit(lock_text: str, runner: Runner = _run, retries: int = RETRIES,
              sleep: Callable[[float], None] = time.sleep, pause_s: float | None = None) -> dict | None:
    """pip-audit's JSON report on the lock's pins, or None when no attempt
    produced one. pip-audit exits 1 when it finds something; only a missing or
    unreadable report counts as a failed attempt (a network error)."""
    pause = RETRY_SLEEP_S if pause_s is None else pause_s
    with tempfile.TemporaryDirectory() as tmp:
        req, out = Path(tmp) / "audit-input.txt", Path(tmp) / "audit.json"
        req.write_text(audit_input(lock_pins(lock_text)))
        cmd = [sys.executable, "-m", "pip_audit", "-r", str(req), "--no-deps", "--disable-pip",
               "--progress-spinner", "off", "--format", "json", "--output", str(out)]
        for attempt in range(1, retries + 1):
            out.unlink(missing_ok=True)
            r = runner(cmd)
            try:
                report = json.loads(out.read_text())
                if isinstance(report, dict) and "dependencies" in report:
                    return report
            except (OSError, ValueError):
                pass
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["no output"]
            print(f"pip-audit gave no report (attempt {attempt} of {retries}, exit {r.returncode}): "
                  f"{tail[0]}", flush=True)
            if attempt < retries:
                sleep(pause * attempt)
    return None


def main(argv: list[str] | None = None, runner: Runner = _run, today: date | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", type=Path, default=LOCK)
    ap.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    ap.add_argument("--report", type=Path, default=None,
                    help="judge this saved pip-audit JSON report instead")
    ap.add_argument("--summary", type=Path, default=None, help="append the Markdown result here")
    args = ap.parse_args(argv)
    today = today or datetime.now(UTC).date()
    if args.report is not None:
        report = json.loads(args.report.read_text())
    else:
        report = run_audit(args.lock.read_text(), runner=runner)
    if report is None:
        msg = "pip-audit gave no report after every retry (the advisory service or PyPI unreachable?)"
        print(f"::error title=pip-audit::{msg}")
        if args.summary:
            with open(args.summary, "a") as fh:
                fh.write(f"## pip-audit gate\n\nNo result: {msg}\n")
        return 2
    verdict = judge(report, json.loads(args.allowlist.read_text()), today)
    lines = report_lines(verdict)
    print("\n".join(lines))
    for f in verdict.uncovered:
        print(f"::error title=pip-audit::{f.package} {f.version}: {f.id} "
              f"(fixed in {', '.join(f.fix_versions) or 'no release yet'}); bump the pin or add a dated "
              "exception to scripts/ci/pip_audit_allowlist.json")
    for p in verdict.problems:
        print(f"::error title=pip-audit allowlist::{p}")
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    sys.exit(main())
