#!/usr/bin/env python3
"""
repo_growth.py: how much the daily bot adds to the repository (`git` objects).

    python scripts/ci/repo_growth.py --range BASE..HEAD [--budget-mb 5] [--summary FILE]
    python scripts/ci/repo_growth.py --session 2026-10-16 [--capture-session 2026-10-20]

Every accepted session commits new, mostly already-compressed files (the Cboe
books, the SPX chain, the fit, the surface report's per-quote residuals, the
diagnostics, the ledger) and every slot rewrites others (the 2026 market
partitions, regime_labels.parquet, the Greeks surface and scenario grid, the
surface index and the diagnostics summary). Git keeps every version, so the
repository only grows; nothing measured it.

--range: the objects the commits in BASE..HEAD add (git rev-list --objects),
         their sizes (git cat-file --batch-check) and the size of the thin pack
         that carries them (git pack-objects --revs --thin: zlib, and deltas
         against what BASE has, as a push sends them), grouped by kind. With
         --budget-mb, a pack above the budget is reported as a GitHub
         ::warning:: (the daily workflow's last step), never a failure.
--session: the files a committed accepted session added and the per-slot
         rewrites, measured on the checkout (raw and zlib level-6 bytes; parquet
         and pickles are already compressed), so the growth per accepted
         session and per year can be stated from measurements.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (kind, regex on the repo-relative path); the first match wins
KINDS = [
    ("cboe books", r"^data_store/options/cboe/"),
    ("SPX chain", r"^data_store/options/[a-z]+_\d{4}-\d{2}-\d{2}\.parquet$"),
    ("fit", r"^data_store/calibrations/joint_cal_"),
    ("diagnostics", r"^data_store/calibrations/diagnostics/"),
    ("surface residuals", r"^data_store/surface/[^/]+/residuals\.parquet$"),
    ("surface json", r"^data_store/surface/"),
    ("Greeks / scenario", r"^data_store/greeks/"),
    ("forward ledger", r"^data_store/forward/"),
    ("market partitions", r"^data_store/market/"),
    ("labels", r"^data_store/signals/"),
    ("other", r""),
]
# per-slot rewrites (the same paths every slot), measured for --session
REWRITES = ["data_store/signals/regime_labels.parquet", "data_store/greeks/greeks_surface.parquet",
            "data_store/greeks/scenario_grid.parquet", "data_store/surface/index.json",
            "data_store/calibrations/diagnostics/summary.json"]


def kind_of(path: str) -> str:
    for kind, pat in KINDS:
        if re.search(pat, path):
            return kind
    return "other"                       # pragma: no cover (the last pattern matches everything)


def git(*args: str, cwd: Path = ROOT, input_: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, input=input_, capture_output=True, check=True).stdout


def range_growth(rng: str, cwd: Path = ROOT) -> dict:
    """Objects added by the commits in ``rng`` (BASE..HEAD): blob bytes by kind
    and the size of the pack that carries them."""
    base, head = rng.split("..", 1)
    objs = git("rev-list", "--objects", rng, cwd=cwd).decode().splitlines()
    paths = {}
    for line in objs:
        sha, _, path = line.partition(" ")
        if path:
            paths[sha] = path
    if paths:
        batch = git("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
                    cwd=cwd, input_=("\n".join(paths) + "\n").encode()).decode().splitlines()
    else:
        batch = []
    by_kind: dict = {}
    for row in batch:
        sha, typ, size = row.split()
        if typ != "blob":
            continue
        k = kind_of(paths[sha])
        n, b = by_kind.get(k, (0, 0))
        by_kind[k] = (n + 1, b + int(size))
    # --thin: deltas against objects BASE already has, as a push sends them
    pack = git("pack-objects", "--revs", "--thin", "--stdout", "-q", cwd=cwd,
               input_=f"{head}\n^{base}\n".encode())
    return {"range": rng, "commits": len(git("rev-list", rng, cwd=cwd).split()),
            "blobs": {k: {"files": n, "bytes": b} for k, (n, b) in sorted(by_kind.items())},
            "blob_bytes": sum(b for _, b in by_kind.values()), "pack_bytes": len(pack)}


def file_sizes(paths, root: Path = ROOT) -> dict:
    out: dict = {}
    for rel in paths:
        p = root / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        k = kind_of(rel)
        e = out.setdefault(k, {"files": 0, "bytes": 0, "zlib_bytes": 0})
        e["files"] += 1
        e["bytes"] += len(data)
        e["zlib_bytes"] += len(zlib.compress(data, 6))
    return out


def session_growth(session: str, capture_session: str | None = None, root: Path = ROOT) -> dict:
    """The files committed for accepted session ``session`` (and the Cboe books of
    ``capture_session``, the newest captured session when the accepted one
    predates the capture step), and the per-slot rewrites."""
    cap = capture_session or session
    new = [f"data_store/options/spx_{session}.parquet", f"data_store/calibrations/joint_cal_{session}.pkl",
           f"data_store/calibrations/diagnostics/{session}.json", f"data_store/forward/ledger_{session}.json"]
    new += [str(p.relative_to(root)) for p in sorted((root / "data_store" / "surface" / session).glob("*"))]
    new += [str(p.relative_to(root)) for p in sorted((root / "data_store" / "options" / "cboe").glob(
        f"*_{cap}_*"))]
    year = session[:4]
    market = root / "data_store" / "market"
    parts = [str(p.relative_to(root)) for p in sorted(market.glob(f"*/*_{year}.parquet"))]
    return {"session": session, "capture_session": cap, "new": file_sizes(new, root),
            "rewrites": file_sizes(REWRITES + parts, root)}


def _mb(n: float) -> str:
    return f"{n / 1e6:.2f} MB"


def report_range(g: dict, budget_mb: float | None) -> list[str]:
    n_blobs = sum(v["files"] for v in g["blobs"].values())
    head = (f"| repository growth, {g['commits']} commit(s) | {_mb(g['pack_bytes'])} packed "
            f"({_mb(g['blob_bytes'])} in {n_blobs} new blob(s)) |")
    lines = [head, "|---|---|"]
    lines += [f"| {k} | {v['files']} file(s), {_mb(v['bytes'])} |" for k, v in g["blobs"].items()]
    if budget_mb is not None and g["pack_bytes"] > budget_mb * 1e6:
        lines.append(f"::warning title=Repository growth::this run added {_mb(g['pack_bytes'])} "
                     f"(budget {budget_mb:g} MB a run): see scripts/ci/repo_growth.py")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--range", help="BASE..HEAD")
    ap.add_argument("--budget-mb", type=float, default=None)
    ap.add_argument("--summary", type=Path, default=None, help="append the Markdown table here")
    ap.add_argument("--session", help="measure a committed accepted session's files")
    ap.add_argument("--capture-session", default=None)
    args = ap.parse_args(argv)
    if bool(args.range) == bool(args.session):
        ap.error("give one of --range or --session")
    if args.range:
        lines = report_range(range_growth(args.range), args.budget_mb)
    else:
        g = session_growth(args.session, args.capture_session)
        new = sum(v["bytes"] for v in g["new"].values())
        rew = sum(v["bytes"] for v in g["rewrites"].values())
        lines = [f"| session {g['session']} (Cboe books of {g['capture_session']}) | bytes | zlib |",
                 "|---|---|---|"]
        lines += [f"| new: {k} ({v['files']}) | {v['bytes']:,} | {v['zlib_bytes']:,} |"
                  for k, v in g["new"].items()]
        lines += [f"| rewritten each slot: {k} ({v['files']}) | {v['bytes']:,} | {v['zlib_bytes']:,} |"
                  for k, v in g["rewrites"].items()]
        new_z = sum(v["zlib_bytes"] for v in g["new"].values())
        rew_z = sum(v["zlib_bytes"] for v in g["rewrites"].values())
        lines += [f"| **new per accepted session** | {new:,} | {new_z:,} |",
                  f"| **rewritten per slot** | {rew:,} | {rew_z:,} |"]
    text = "\n".join(lines)
    print(text)
    if args.summary:
        with open(args.summary, "a") as fh:
            fh.write("\n".join(ln for ln in lines if not ln.startswith("::")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
