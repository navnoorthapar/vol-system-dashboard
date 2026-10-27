#!/usr/bin/env python3
"""
render_corrections.py — docs/corrections.md from the corrections registry.

The registry (dashboard/corrections.json) is the one source of the correction
rounds C16 onward; the Corrections page renders it on the site and this
script renders the same rounds into docs/corrections.md, between the markers

    <!-- BEGIN GENERATED: scripts/render_corrections.py -->
    <!-- END GENERATED -->

so the prose around them (narrative notes) is kept. The C22 and C23
waterfalls are rendered from data_store/backtest/c22_bridge.csv and
c23_bridge.csv. The same registry renders the findings log of the
validation pack (docs/model_validation/README.md §6) between

    <!-- BEGIN GENERATED: findings log, scripts/render_corrections.py -->
    <!-- END GENERATED -->

one row per round, or per finding where the finding carries its own id.
test_readme.py fails when either committed file differs from a fresh render.

    python scripts/render_corrections.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "dashboard" / "corrections.json"
DOC = ROOT / "docs" / "corrections.md"
PACK = ROOT / "docs" / "model_validation" / "README.md"
BEGIN = "<!-- BEGIN GENERATED: scripts/render_corrections.py -->"
END = "<!-- END GENERATED -->"
LOG_BEGIN = "<!-- BEGIN GENERATED: findings log, scripts/render_corrections.py -->"
MINUS = "−"


def usd(x: float, signed: bool = True) -> str:
    s = f"${abs(x):,.0f}"
    if x < 0:
        return MINUS + s
    return ("+" + s) if signed and x > 0 else s


def num(x: float, spec: str) -> str:
    return format(x, spec).replace("-", MINUS)


def read_bridge(name: str) -> list[dict]:
    with open(ROOT / "data_store" / "backtest" / name, newline="") as fh:
        return sorted(csv.DictReader(fh), key=lambda r: int(r["row"]))


def waterfall(rows: list[dict], first_label: str | None = None) -> list[str]:
    out = ["| Row | Correction applied (cumulative) | NAV | P&L | Δ P&L | Sharpe |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        label = r["label"] if not (first_label and int(r["row"]) == 0) else first_label
        d = float(r["d_pnl"])
        out.append(f"| {r['row']} | {label} | {usd(float(r['nav']), False)} | {usd(float(r['pnl']))} | "
                   f"{usd(d) if int(r['row']) else ''} | {num(float(r['sharpe']), '.2f')} |")
    return out


def finding_evidence(f: dict, steps: dict) -> str:
    if f.get("evidence"):
        return f["evidence"]
    rows = [steps[s] for s in f.get("bridge_steps") or [] if s in steps]
    if not rows:
        return "computed on the Corrections page"
    first = min(int(r["row"]) for r in rows)
    prev = next((r for r in steps.values() if int(r["row"]) == first - 1), None)
    last = max(rows, key=lambda r: int(r["row"]))
    d = sum(float(r["d_pnl"]) for r in rows)
    sh = (f", Sharpe {num(float(prev['sharpe']), '.2f')} → {num(float(last['sharpe']), '.2f')}"
          if prev is not None else "")
    which = f"row{'s' if len(rows) > 1 else ''} {', '.join(r['row'] for r in rows)}"
    return f"C22 bridge {which}: portfolio P&L {usd(d)}{sh} (the bridge replayed on the C23-restated record)"


def render() -> str:
    reg = json.loads(REGISTRY.read_text())
    repo = reg["repo"]
    c22 = read_bridge("c22_bridge.csv")
    c23 = read_bridge("c23_bridge.csv")
    steps = {r["step"]: r for r in c22}
    rounds = reg["rounds"]
    n_find = sum(len(r["findings"]) for r in rounds)
    n_fixed = sum(f["status"] == "fixed" for r in rounds for f in r["findings"])
    n_tests = sum(len(f.get("tests") or []) for r in rounds for f in r["findings"])
    intro = (f"{rounds[0]['id']}–{rounds[-1]['id']}: {len(rounds)} rounds, {n_find} findings "
             f"({n_fixed} fixed, {n_find - n_fixed} quantified), {n_tests} regression tests cited. "
             "Rendered from "
             "`dashboard/corrections.json`, the registry the site's Corrections page reads.")
    out = [BEGIN, "", intro, "",
           "| Round | Date | Severity | Title | Effect on the published result |", "|---|---|---|---|---|"]
    def span(r: dict) -> str:
        end = r.get("date_end")
        return r["date"] + (f" → {end}" if end and end != r["date"] else "")

    for r in rounds:
        eff = r.get("effect") or ""
        link = f"[{r['id']}](#{r['id'].lower()})"
        out.append(f"| {link} | {span(r)} | {r['severity']} | {r['title']} | {eff} |")
    for r in rounds:
        head = (f"**{r['title']}** · {span(r)} · severity {r['severity']} · "
                f"components: {', '.join(r['components'])}")
        out += ["", f"### {r['id']}", "", head, "",
                f"- **Detection.** {r['detection']}",
                f"- **Effect on the published result.** {r.get('effect') or ''}",
                "- **Commits.** " + ", ".join(f"[`{c[:7]}`]({repo}/commit/{c})" for c in r["commits"])]
        if r["id"] == "C22":
            out += ["", ("The C22 waterfall, one correction at a time (`data_store/backtest/c22_bridge.csv`, "
                         "`python -m joint_vol_calibration.backtest.c22_bridge`). It is replayed on the "
                         "C23-restated VIX record, so its last row is today's canonical run; the published "
                         "C22 run ended at NAV $783,908, P&L −$216,092, Sharpe −0.88."), ""]
            out += waterfall(c22)
        if r["id"] == "C23":
            out += ["", ("The C23 waterfall: the published C22 run restated row by row on the Cboe record "
                         "(`data_store/backtest/c23_bridge.csv`, "
                         "`python -m joint_vol_calibration.backtest.c23_bridge`)."), ""]
            out += waterfall(c23)
        for i, f in enumerate(r["findings"], 1):
            tests = f.get("tests") or []
            pinned = (", ".join(f"`{t}`" for t in tests) if tests
                      else "no dedicated regression test (a gap)")
            fid = f" [{f['id']}]" if f.get("id") else ""
            out += ["", f"**Finding {i}{fid} ({f['status']}): {f['flaw']}**", "",
                    f"- Root cause: {f['root_cause']}",
                    f"- Fix: {f['fix']}",
                    f"- Evidence: {finding_evidence(f, steps)}"]
            if f.get("effect"):
                out.append(f"- Effect on the published result: {f['effect']}")
            if f.get("commits"):
                out.append("- Commits: " + ", ".join(f"[`{c[:7]}`]({repo}/commit/{c})" for c in f["commits"]))
            out.append(f"- Pinned by: {pinned}")
    out += ["", END]
    return "\n".join(out)


def _status(findings: list[dict]) -> str:
    kinds = sorted({f["status"] for f in findings}, key=("fixed", "quantified").index)
    return ", ".join(kinds)


def render_findings_log() -> str:
    """The validation pack's findings log (§6): one row per round, or one per
    finding where the finding carries its own id (the C23 sub-findings)."""
    reg = json.loads(REGISTRY.read_text())
    head = "| ID | Date | Component | Severity | Finding | Effect on the published result | Status | Commit |"
    out = [LOG_BEGIN, "", head, "|---|---|---|---|---|---|---|---|"]

    def span(x: dict) -> str:
        end = x.get("date_end")
        return x["date"] + (f" → {end}" if end and end != x["date"] else "")

    def commits(shas: list[str]) -> str:
        return ", ".join(f"`{c[:7]}`" for c in shas)

    for r in reg["rounds"]:
        own = [f for f in r["findings"] if f.get("id")]
        if own:
            out.extend(f"| {f['id']} | {span(f)} | {', '.join(f['models'])} | {f['severity']} | "
                       f"{f['flaw']} | {f['effect']} | {f['status']} | "
                       f"{commits(f['commits'])} |" for f in own)
        else:
            models = ", ".join(r.get("models") or r["components"])
            out.append(f"| {r['id']} | {span(r)} | {models} | {r['severity']} | {r['title']} | "
                       f"{r.get('effect') or ''} | {_status(r['findings'])} | {commits(r['commits'])} |")
    out += ["", END]
    return "\n".join(out)


def splice(doc: str, block: str, begin: str = BEGIN) -> str:
    i = doc.index(begin)
    j = doc.index(END, i) + len(END)
    return doc[:i] + block + doc[j:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if docs/corrections.md or the pack's findings log is not up to date")
    a = ap.parse_args(argv)
    targets = ((DOC, render(), BEGIN), (PACK, render_findings_log(), LOG_BEGIN))
    stale = 0
    for path, block, begin in targets:
        doc = path.read_text()
        new = splice(doc, block, begin)
        if a.check:
            if new != doc:
                print(f"{path.relative_to(ROOT)} is stale: run python scripts/render_corrections.py",
                      file=sys.stderr)
                stale = 1
            continue
        path.write_text(new)
        print(f"wrote {path.relative_to(ROOT)}")
    return stale


if __name__ == "__main__":
    raise SystemExit(main())
