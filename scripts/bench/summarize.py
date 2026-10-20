#!/usr/bin/env python3
"""
summarize.py: compact summary of a pytest-benchmark JSON (`make bench` writes
.bench/benchmarks.json) for docs/benchmarks/.

    python scripts/bench/summarize.py .bench/benchmarks.json --json docs/benchmarks/<date>.json

Keeps the machine description, the commit, and per benchmark min / median /
mean / stddev / rounds and extra_info (node counts, parity), and drops the raw
per-round samples (2 MB). Prints a markdown table (min and median, in
microseconds or seconds).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def summarize(doc: dict[str, Any]) -> dict[str, Any]:
    mi = doc["machine_info"]
    cpu = mi.get("cpu", {})
    out: dict[str, Any] = {
        "datetime": doc["datetime"],
        "pytest_benchmark": doc["version"],
        "commit": {k: doc["commit_info"].get(k) for k in ("id", "dirty", "branch")},
        "machine": {"cpu": cpu.get("brand_raw"), "logical_cpus": cpu.get("count"),
                    "avx512f": "avx512f" in cpu.get("flags", []), "system": mi.get("system"),
                    "release": mi.get("release"), "python": mi.get("python_version")},
        "benchmarks": [],
    }
    for b in doc["benchmarks"]:
        s = b["stats"]
        out["benchmarks"].append({
            "name": b["fullname"].split("::")[-1], "group": b.get("group"),
            "min_s": s["min"], "median_s": s["median"], "mean_s": s["mean"],
            "stddev_s": s["stddev"], "rounds": s["rounds"], "extra_info": b.get("extra_info", {}),
        })
    return out


def _fmt(sec: float) -> str:
    return f"{sec:.1f} s" if sec >= 1.0 else f"{sec * 1e6:,.0f} µs"


def table(summary: dict[str, Any]) -> str:
    lines = ["| Benchmark | Group | Min | Median | Rounds |", "|---|---|---|---|---|"]
    lines.extend(f"| {b['name']} | {b['group'] or '-'} | {_fmt(b['min_s'])} | {_fmt(b['median_s'])} "
                 f"| {b['rounds']} |" for b in summary["benchmarks"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("benchmark_json", type=Path)
    ap.add_argument("--json", type=Path, help="write the compact summary here")
    args = ap.parse_args(argv)
    summary = summarize(json.loads(args.benchmark_json.read_text()))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    print(table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
