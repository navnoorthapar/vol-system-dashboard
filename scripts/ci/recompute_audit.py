#!/usr/bin/env python3
"""
recompute_audit.py: recompute every number the Overview and Surface pages print,
from data_store, by a code path of its own, and diff against the frozen HTML.

    python scripts/ci/recompute_audit.py SITE_DIR [--data data_store] [--md FILE] [--json FILE]

What an independent validator does first (FUND_GRADE_PLAN.md tier 0.1). The
checks live in scripts/ci/recompute/ (chrome.py: the status strip and ticker
every page carries; overview.py; surface.py). They read data_store with pandas,
SQLite and json and redo the arithmetic from the definitions the pages state
(SVI, forward delta, RMS realised vol, NYSE sessions); none imports dashboard/
or joint_vol_calibration/, which a test enforces. A printed number passes when
the recomputed value rounds to it at the printed precision.

The report is a table of every check (page, panel, what, printed, recomputed)
and a coverage line per page: how many of the numbers and dates printed inside
<main> sit in an element a check verified, and where the rest are.

Exit 1 on any mismatch, on any number a check expected and the page lacks, or
when the SQLite cache the site renders from disagrees with the committed parquet
record on a row both hold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from recompute import chrome, overview, surface  # noqa: E402
from recompute.core import Audit, Check  # noqa: E402
from recompute.data import Store, cache_vs_record  # noqa: E402

PAGES = (("index.html", "Overview", (chrome.run, overview.run)),
         ("surface.html", "Surface", (chrome.run, surface.run)))
CACHE_TABLES = (("spx_ohlcv", ("open", "high", "low", "close")), ("vix_daily", ("close",)),
                ("vix_term_structure", ("close",)))


def run(site: Path, data: Path) -> dict:
    from bs4 import BeautifulSoup

    store = Store(data)
    checks: list[Check] = []
    coverage: dict[str, dict] = {}
    for fname, name, fns in PAGES:
        soup = BeautifulSoup((site / fname).read_text(encoding="utf-8"), "lxml")
        audit = Audit(name, soup)
        for fn in fns:
            try:
                fn(audit, store)
            except Exception as exc:  # noqa: BLE001  (a check that crashes fails; it is never skipped)
                audit.fail(fn.__module__.rsplit(".", 1)[-1], "check raised", "no exception",
                           f"{type(exc).__name__}: {exc}")
        checks.extend(audit.checks)
        coverage[name] = audit.coverage()
    for table, cols in CACHE_TABLES:
        r = cache_vs_record(store, table, cols)
        # the pages render from the record (app.record_db), so a cached value the
        # record replaces is reported, not failed; a cached row the record lacks
        # could not be rebuilt from the repository, and fails
        ok = r["not_in_record"] == 0
        note = f"{r['rows']:,} rows; {r['differ']} cached value(s) the record replaces"
        if r["record_ahead"]:
            note += (f"; the record holds {len(r['record_ahead'])} later session(s) the cache does not "
                     f"({', '.join(r['record_ahead'][:3])})")
        what = f"vol_system.db {table} rows in market/{table}/*.parquet"
        checks.append(Check("data", "cache vs record", what,
                            f"{r['not_in_record']} not in the record", "0 not in the record", ok, note))
    return {"checks": checks, "coverage": coverage}


def markdown(res: dict) -> str:
    checks: list[Check] = res["checks"]
    bad = [c for c in checks if not c.ok]
    lines = [f"**Recompute audit:** {len(checks) - len(bad)} of {len(checks)} checks match"
             + (f", **{len(bad)} mismatch**" if bad else "") + ".", ""]
    for page, cov in res["coverage"].items():
        pct = 100.0 * cov["covered"] / cov["total"] if cov["total"] else 100.0
        miss = ", ".join(f"{k} {v}" for k, v in sorted(cov["missed"].items(), key=lambda kv: -kv[1])[:8])
        lines.append(f"- {page}: {cov['covered']:,} of {cov['total']:,} printed numbers and dates "
                     f"inside a verified element ({pct:.1f}%)"
                     + (f"; not yet checked: {miss}" if miss else ""))
    lines += ["", "| | Page | Panel | What | Printed | Recomputed | Note |", "|---|---|---|---|---|---|---|"]

    def cell(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")
    for c in sorted(checks, key=lambda c: c.ok):
        lines.append(f"| {'✓' if c.ok else '✗'} | {c.page} | {cell(c.panel)} | {cell(c.label)} | "
                     f"{cell(c.printed)} | {cell(c.expected)} | {cell(c.note)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("site", type=Path, help="the frozen site (dashboard/freeze_site.py output)")
    ap.add_argument("--data", type=Path, default=ROOT / "data_store")
    ap.add_argument("--md", type=Path, help="append the markdown report here (e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--json", type=Path, help="write every check as JSON here")
    a = ap.parse_args(argv)
    res = run(a.site, a.data)
    md = markdown(res)
    print(md)
    if a.md:
        with open(a.md, "a", encoding="utf-8") as f:
            f.write("## Recompute audit (Overview, Surface)\n\n" + md)
    if a.json:
        a.json.write_text(json.dumps({"checks": [c.__dict__ for c in res["checks"]],
                                      "coverage": res["coverage"]}, indent=1))
    return 1 if any(not c.ok for c in res["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
