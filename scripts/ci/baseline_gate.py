#!/usr/bin/env python3
"""
baseline_gate.py: lint and type-check gates that accept the existing backlog
but fail on anything new.

    python scripts/ci/baseline_gate.py ruff            # gate (CI, `make lint`)
    python scripts/ci/baseline_gate.py mypy            # gate (CI, `make typecheck`)
    python scripts/ci/baseline_gate.py ruff --update   # rewrite the baseline after fixes

The rule sets (pyproject.toml [tool.ruff.lint], [tool.mypy]) are broader than
what the code passed when they were introduced. Each tool's findings are
counted per (file, rule/error code) and compared with a committed baseline
(scripts/ci/{ruff,mypy}_baseline.json):

  * a count above the baseline fails the gate, and every finding of that
    (file, code) pair is printed (counts, not line numbers, are pinned, so
    the gate survives unrelated edits that move lines);
  * a count below the baseline passes, with a note that the baseline can
    shrink (`--update` writes the lower counts);
  * --update refuses to raise any count unless --allow-growth is given. That
    flag exists for a reviewed re-baseline (a tool upgrade that adds rules),
    so in normal use the baseline can only shrink;
  * STRICT_FILES may never appear in a baseline: one error there fails.

Both tools run through the current interpreter (python -m ruff / python -m
mypy), i.e. the versions pinned in requirements-lock.txt; the baseline records
the version it was written with, and a mismatch is reported because a new
version can move counts on its own.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = Path(__file__).resolve().parent

# Modules gated at zero errors under mypy's strict flags (pyproject overrides).
STRICT_FILES = {
    "mypy": (
        "joint_vol_calibration/models/implied_vol.py",
        "joint_vol_calibration/data/option_chain.py",
    ),
    "ruff": (),
}

Key = tuple[str, str]          # (repo-relative posix path, rule / error code)

_MYPY_LINE = re.compile(
    r"^(?P<path>[^:\n]+?):(?P<line>\d+)(?::(?P<col>\d+))?: (?P<sev>error|note|warning): "
    r"(?P<msg>.*?)(?:\s+\[(?P<code>[A-Za-z0-9_-]+)\])?\s*$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    code: str
    message: str

    @property
    def key(self) -> Key:
        return (self.path, self.code)


@dataclass
class Comparison:
    grown: dict[Key, tuple[int, int]] = field(default_factory=dict)    # key -> (baseline, now)
    shrunk: dict[Key, tuple[int, int]] = field(default_factory=dict)
    strict_violations: list[Key] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.grown and not self.strict_violations


def _relpath(path: str, root: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(root.resolve())
        except ValueError:
            return p.as_posix()
    return p.as_posix()


def parse_ruff_json(text: str, root: Path = ROOT) -> list[Finding]:
    """Findings from `ruff check --output-format json`."""
    out = []
    for item in json.loads(text or "[]"):
        code = item.get("code") or "syntax-error"       # ruff reports syntax errors with code null
        row = int((item.get("location") or {}).get("row") or 0)
        out.append(Finding(_relpath(item["filename"], root), row, code, item.get("message", "")))
    return out


def parse_mypy_output(text: str, root: Path = ROOT) -> list[Finding]:
    """Errors from mypy's default text output (notes and the summary line are
    ignored; an error without a code is counted as 'misc')."""
    out = []
    for line in text.splitlines():
        m = _MYPY_LINE.match(line)
        if not m or m.group("sev") != "error":
            continue
        out.append(Finding(_relpath(m.group("path"), root), int(m.group("line")),
                           m.group("code") or "misc", m.group("msg")))
    return out


def count(findings: Iterable[Finding]) -> Counter[Key]:
    return Counter(f.key for f in findings)


def compare(baseline: Counter[Key], current: Counter[Key],
            strict_files: Iterable[str] = ()) -> Comparison:
    res = Comparison()
    for key in sorted(set(baseline) | set(current)):
        b, c = baseline.get(key, 0), current.get(key, 0)
        if c > b:
            res.grown[key] = (b, c)
        elif c < b:
            res.shrunk[key] = (b, c)
    strict = set(strict_files)
    res.strict_violations = sorted(k for k in set(baseline) | set(current)
                                   if k[0] in strict and (baseline.get(k, 0) or current.get(k, 0)))
    return res


def updated_baseline(baseline: Counter[Key], current: Counter[Key],
                     allow_growth: bool = False) -> Counter[Key]:
    """The baseline --update writes: the current counts. Raises ValueError if
    that would raise any count and allow_growth is False."""
    grown = {k: (baseline.get(k, 0), n) for k, n in current.items() if n > baseline.get(k, 0)}
    if grown and not allow_growth:
        raise ValueError("refusing to grow the baseline: " + ", ".join(
            f"{p} {c}: {b} -> {n}" for (p, c), (b, n) in sorted(grown.items())))
    return Counter({k: n for k, n in current.items() if n > 0})


def to_json(counts: Counter[Key], tool: str, version: str) -> str:
    nested: dict[str, dict[str, int]] = {}
    for (path, code), n in sorted(counts.items()):
        if n > 0:
            nested.setdefault(path, {})[code] = n
    total = sum(n for codes in nested.values() for n in codes.values())
    doc = {"tool": tool, "version": version, "total": total, "counts": nested}
    return json.dumps(doc, indent=1, sort_keys=True) + "\n"


def from_json(text: str) -> tuple[Counter[Key], dict[str, Any]]:
    doc = json.loads(text)
    c = Counter({(path, code): int(n) for path, codes in doc.get("counts", {}).items()
                 for code, n in codes.items()})
    return c, doc


# ── running the tools ────────────────────────────────────────────────────────

def _tool_version(tool: str) -> str:
    mod = {"ruff": "ruff", "mypy": "mypy"}[tool]
    r = subprocess.run([sys.executable, "-m", mod, "--version"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def run_tool(tool: str) -> list[Finding]:
    if tool == "ruff":
        r = subprocess.run([sys.executable, "-m", "ruff", "check", "--output-format", "json",
                            "--exit-zero", "--no-cache", "."],
                           cwd=ROOT, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"ruff failed ({r.returncode}): {r.stderr.strip()}")
        return parse_ruff_json(r.stdout)
    if tool == "mypy":
        r = subprocess.run([sys.executable, "-m", "mypy", "--no-error-summary",
                            "--show-error-codes", "--no-pretty", "--no-color-output"],
                           cwd=ROOT, capture_output=True, text=True, check=False)
        if r.returncode not in (0, 1):        # 2 = usage error or crash
            raise RuntimeError(f"mypy failed ({r.returncode}): {r.stderr.strip() or r.stdout[-2000:]}")
        return parse_mypy_output(r.stdout)
    raise ValueError(tool)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tool", choices=("ruff", "mypy"))
    ap.add_argument("--baseline", type=Path, help="default: scripts/ci/<tool>_baseline.json")
    ap.add_argument("--update", action="store_true", help="rewrite the baseline to the current counts")
    ap.add_argument("--allow-growth", action="store_true",
                    help="with --update: allow counts to rise (reviewed re-baseline only)")
    args = ap.parse_args(argv)

    path = args.baseline or BASELINE_DIR / f"{args.tool}_baseline.json"
    version = _tool_version(args.tool)
    findings = run_tool(args.tool)
    current = count(findings)
    baseline, doc = (from_json(path.read_text()) if path.exists() else (Counter(), {}))

    if args.update:
        try:
            new = updated_baseline(baseline, current, allow_growth=args.allow_growth or not path.exists())
        except ValueError as exc:
            print(f"{args.tool}: {exc}", file=sys.stderr)
            return 1
        strict_hits = [k for k in new if k[0] in STRICT_FILES[args.tool]]
        if strict_hits:
            print(f"{args.tool}: strict files cannot be baselined: {strict_hits}", file=sys.stderr)
            return 1
        path.write_text(to_json(new, args.tool, version))
        print(f"{args.tool}: baseline written to {path.relative_to(ROOT)} "
              f"({sum(baseline.values())} -> {sum(new.values())} findings)")
        return 0

    if doc.get("version") and doc["version"] != version:
        print(f"note: baseline was written with {doc['version']}, running {version}; "
              "a version change can move counts by itself")
    res = compare(baseline, current, STRICT_FILES[args.tool])
    by_key: dict[Key, list[Finding]] = {}
    for f in findings:
        by_key.setdefault(f.key, []).append(f)
    for key, (b, c) in res.grown.items():
        print(f"NEW  {key[0]} {key[1]}: baseline allows {b}, found {c}")
        for f in sorted(by_key.get(key, []), key=lambda f: f.line):
            print(f"     {f.path}:{f.line}: {f.message}")
    for key in res.strict_violations:
        print(f"STRICT  {key[0]} must have no {args.tool} findings ({key[1]}: {current.get(key, 0)})")
    if res.shrunk:
        n = sum(b - c for b, c in res.shrunk.values())
        print(f"{args.tool}: {n} finding(s) fixed in {len(res.shrunk)} (file, code) pair(s); "
              f"shrink the baseline with: python scripts/ci/baseline_gate.py {args.tool} --update")
    total_b, total_c = sum(baseline.values()), sum(current.values())
    status = "OK" if res.ok else "FAILED"
    print(f"{args.tool} gate {status}: {total_c} findings (baseline {total_b}), "
          f"{len(res.grown)} pair(s) grew, strict files clean: {not res.strict_violations}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
