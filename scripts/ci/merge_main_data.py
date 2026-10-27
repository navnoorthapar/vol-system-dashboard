#!/usr/bin/env python3
"""
merge_main_data.py: merge live main into this branch when main's daily bot has
rewritten the binary data files this branch also rewrote.

    python scripts/ci/merge_main_data.py [--ref origin/main] [--fetch] [--commit]

git cannot merge parquet. C23 restated the VIX index family from Cboe's
official files and rebuilt every regime label from the restated record; main's
bot keeps appending a session a day to the same yearly partitions and to
regime_labels.parquet. Every weekday's close slot therefore adds a conflict on
data_store/market/vix_daily/vix_daily_<year>.parquet,
vix_term_structure_<year>.parquet and data_store/signals/regime_labels.parquet,
and either wholesale side is wrong: main's undoes the restatement (its 2026 term
structure is pure Yahoo; 6 labels from 2019-08-09 to 2021-07-08 revert, and
extend_regime_labels never heals them because they lie before its 900-day
window), the branch's drops main's new sessions.

So the conflicting files are not chosen, they are re-derived from the merged
record:

  1. `git merge --no-commit --no-ff REF`. A conflict outside the re-derivable
     files (data_store/market/, data_store/signals/regime_labels.parquet)
     aborts the merge and exits 1: that needs a person.
  2. Each conflicting file is resolved to this branch's version (the restated
     record), and main's version of every market partition main changed is
     set aside.
  3. Main's partitions are upserted into this branch's database
     (data_store/vol_system.db, database.import_market_partitions): its new
     sessions go in, and the inserters' rules decide a clash (a Yahoo row
     never replaces a Cboe one; a row dated on an NYSE holiday is not stored,
     which drops the 2026-05-25 / 2026-09-07 Yahoo artefacts).
  4. The partitions are re-exported from the database over the merged files
     (database.export_market_partitions merges: the database wins a key
     clash, no stored row is dropped), and the labels are extended from the
     database as the bot does (cloud_refresh.extend_regime_labels: appends the
     sessions the file lacks, heals only the trailing window), so the
     restated labels stay and main's new sessions get their rule label.
  5. The resolved files and the database are staged, and a report (what each
     side held, what the merge holds) is printed and written to --report.
     Nothing is committed without --commit.

Run it in a clean checkout of the branch (it refuses a dirty tree), then run
the full suite: test_backtest_reproducibility's
test_rule_label_file_reproduces_from_the_database checks that the labels and
the database agree. The next bot run then replaces any provisional Yahoo row
with Cboe's print (the market step downloads 400 days, Cboe-primary).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REDERIVABLE = ("data_store/market/", "data_store/signals/regime_labels.parquet")
LABELS = "data_store/signals/regime_labels.parquet"
DB = "data_store/vol_system.db"


def git(*args: str, cwd: Path = ROOT, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode}): {r.stderr.strip()}")
    return r.stdout


def is_rederivable(path: str) -> bool:
    return any(path == p or (p.endswith("/") and path.startswith(p)) for p in REDERIVABLE)


def start_merge(ref: str, cwd: Path = ROOT) -> dict:
    """`git merge --no-commit --no-ff ref`; returns {"conflicts", "main_changed",
    "base"}. A conflict outside REDERIVABLE aborts the merge (ValueError)."""
    if git("status", "--porcelain", "--untracked-files=no", cwd=cwd).strip() or \
            git("status", "--porcelain", "--", "data_store", cwd=cwd).strip():
        raise ValueError("the working tree has uncommitted changes (or untracked files under "
                         "data_store/): run this in a clean checkout")
    base = git("merge-base", "HEAD", ref, cwd=cwd).strip()
    main_changed = [p for p in git("diff", "--name-only", base, ref, cwd=cwd).splitlines() if p]
    subprocess.run(["git", "merge", "--no-commit", "--no-ff", ref], cwd=cwd, capture_output=True,
                   text=True, check=False)
    merging = subprocess.run(["git", "rev-parse", "-q", "--verify", "MERGE_HEAD"], cwd=cwd,
                             capture_output=True, check=False).returncode == 0
    if not merging:
        raise ValueError(f"nothing to merge: {ref} is already in HEAD")
    conflicts = [p for p in git("diff", "--name-only", "--diff-filter=U", cwd=cwd).splitlines() if p]
    other = [p for p in conflicts if not is_rederivable(p)]
    if other:
        git("merge", "--abort", cwd=cwd)
        raise ValueError(f"conflicts outside the re-derivable data files, resolve by hand: {other}")
    return {"conflicts": conflicts, "main_changed": main_changed, "base": base}


def take_ours_and_set_theirs_aside(state: dict, ref: str, theirs_root: Path, cwd: Path = ROOT) -> None:
    """Resolve each conflict to HEAD's version; write main's version of every
    market partition main changed (and of the labels) under ``theirs_root``."""
    for p in state["conflicts"]:
        git("checkout", "--ours", "--", p, cwd=cwd)
    for p in state["main_changed"]:
        if not is_rederivable(p):
            continue
        blob = subprocess.run(["git", "show", f"{ref}:{p}"], cwd=cwd, capture_output=True, check=False)
        if blob.returncode != 0:                      # deleted on main: nothing to import
            continue
        out = theirs_root / p
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob.stdout)


def _load_cloud_refresh(root: Path):
    script = root / "scripts" / "cloud_refresh.py"
    spec = importlib.util.spec_from_file_location("cloud_refresh_merge", script)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/cloud_refresh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _frame_summary(path: Path) -> dict:
    import pandas as pd
    if not path.exists():
        return {"rows": 0}
    df = pd.read_parquet(path)
    dates = (df["date"].astype(str).str[:10] if "date" in df.columns
             else pd.Index(df.index).astype(str).str[:10])
    out: dict = {"rows": len(df), "first": str(dates.min()) if len(df) else None,
           "last": str(dates.max()) if len(df) else None}
    if "source" in df.columns:
        out["sources"] = {str(k): int(v) for k, v in df["source"].value_counts().items()}
    return out


def rederive(theirs_root: Path, main_changed: list, cr, *, as_of: str | None = None) -> dict:
    """Steps 3 and 4: main's partitions into the database, then the partitions
    re-exported and the labels extended from it. ``cr`` is scripts/cloud_refresh.py
    (its DATA and the database module's paths point at the checkout)."""
    from joint_vol_calibration.data import database as dbm
    tables = sorted({Path(p).parts[2] for p in main_changed if p.startswith("data_store/market/")
                     and len(Path(p).parts) > 3})
    report: dict = {"tables": tables}
    if tables:
        report["imported"] = dbm.import_market_partitions(root=theirs_root / "data_store" / "market",
                                                          tables=tables)
        years = sorted({Path(p).stem.rsplit("_", 1)[-1] for p in main_changed
                        if p.startswith("data_store/market/") and p.endswith(".parquet")})
        since = f"{years[0]}-01-01" if years else None
        report["exported"] = [str(p) for p in dbm.export_market_partitions(tables=tables, since=since)]
    labels_touched = any(p == LABELS for p in main_changed)
    if labels_touched:
        last = as_of or dbm.get_latest_date("spx_ohlcv")
        report["labels_added"] = cr.extend_regime_labels(date.fromisoformat(str(last)[:10]))
    return report


def compare_labels(ours: Path, theirs: Path, merged: Path) -> dict:
    """What the merge did to the labels: rows only main had (and whether they
    are in the merge), and the rows where the merged label differs from main's."""
    import pandas as pd
    o, t, m = (pd.read_parquet(p) for p in (ours, theirs, merged))
    only_main = t.index.difference(o.index)
    common = t.index.intersection(m.index)
    differ = common[(t.loc[common, "regime"].astype(int) != m.loc[common, "regime"].astype(int)).to_numpy()]
    kept = o.index.intersection(m.index)
    changed_ours = kept[(o.loc[kept, "regime"].astype(int) != m.loc[kept, "regime"].astype(int)).to_numpy()]
    return {"rows": {"branch": len(o), "main": len(t), "merged": len(m)},
            "main_only_dates": [str(d.date()) for d in only_main],
            "main_only_in_merge": bool(only_main.isin(m.index).all()),
            "labels_differing_from_main": [str(d.date()) for d in differ],
            "branch_labels_changed_by_merge": [str(d.date()) for d in changed_ours]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="origin/main", help="what to merge (default origin/main)")
    ap.add_argument("--fetch", action="store_true", help="git fetch origin main first")
    ap.add_argument("--commit", action="store_true", help="commit the merge (default: leave it staged)")
    ap.add_argument("--report", type=Path, default=None, help="also write the report as JSON here")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    if args.fetch:
        git("fetch", "-q", "origin", "main")
    try:
        state = start_merge(args.ref)
    except ValueError as exc:
        print(f"merge_main_data: {exc}", file=sys.stderr)
        return 1
    report: dict = {"ref": args.ref, "ref_sha": git("rev-parse", args.ref).strip(), "head": git(
        "rev-parse", "HEAD").strip(), "conflicts": state["conflicts"],
        "main_changed_data": [p for p in state["main_changed"] if is_rederivable(p)]}
    with tempfile.TemporaryDirectory(prefix="merge-main-") as tmp:
        theirs = Path(tmp)
        take_ours_and_set_theirs_aside(state, args.ref, theirs)
        before = {p: _frame_summary(ROOT / p) for p in report["main_changed_data"]}
        cr = _load_cloud_refresh(ROOT)
        report["rederive"] = rederive(theirs, state["main_changed"], cr)
        report["files"] = {p: {"branch": before[p], "main": _frame_summary(theirs / p),
                               "merged": _frame_summary(ROOT / p)} for p in report["main_changed_data"]}
        if (theirs / LABELS).exists():
            ours_labels = Path(tmp) / "ours_labels.parquet"
            ours_labels.write_bytes(subprocess.run(["git", "show", f"HEAD:{LABELS}"], cwd=ROOT,
                                                   capture_output=True, check=True).stdout)
            report["labels"] = compare_labels(ours_labels, theirs / LABELS, ROOT / LABELS)
    git("add", "-A", "--", "data_store/market", LABELS, *([DB] if (ROOT / DB).exists() else []))
    left = [p for p in git("diff", "--name-only", "--diff-filter=U").splitlines() if p]
    if left:
        print(f"merge_main_data: still unmerged: {left}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if args.report:
        args.report.write_text(text + "\n")
    if args.commit:
        git("commit", "-q", "-m", f"Merge {args.ref} ({report['ref_sha'][:7]}): its data re-derived "
            "over the branch's record (scripts/ci/merge_main_data.py)")
        print(f"committed {git('rev-parse', '--short', 'HEAD').strip()}")
    else:
        print("merge staged, not committed (review, run the suite, then `git commit`)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
