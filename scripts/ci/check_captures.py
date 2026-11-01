#!/usr/bin/env python3
"""
check_captures.py: the minimal integrity checks that gate the daily bot's
capture commit (.github/workflows/daily_refresh.yml, "Commit captures").

    python scripts/ci/check_captures.py --out LIST [--root DIR]

The captures are what a later run cannot fetch again: the Cboe GTH SPX and VIX
books, the accepted SPX chain, the accepted fit and the forward ledger. They
used to be committed only behind the full data gate (~355 tests), so a test
that failed on anything, and a push that failed for any reason, discarded the
session's book with the runner (2026-10-19, 2026-10-20, 2026-10-21). They are
now committed first and on their own, behind these checks only: each file the
run created (untracked under data_store/options, data_store/calibrations
without its diagnostics, data_store/forward, data_store/runs) is recognised
by its name and read back:

  options/cboe_manifest/<stem>.json                 the public record of a raw book held
                                                    in the private store: schema, stem,
                                                    session, hashes, rows, private path
  options/cboe/<und>_<session>_<HHMMSS>Z.parquet   readable, >= 1 row, its JSON
           + .json                                  pair present with the same row
                                                    count, session and a sha256
  options/<und>_<session>.parquet                   readable, >= 1 row, every
                                                    snapshot_date the session's
  calibrations/joint_cal_<session>.pkl              unpickles to a fit of that
                                                    session that passes the gate
  forward/ledger_<session>.json                     a valid ledger of that session that
                                                    links to the head of the chain (the
                                                    newest ledger before it; none after
                                                    it is committed)
  runs/run_<YYYYmmddTHHMMSSZ>_<slot>.json           the run record (cloud_refresh
                                                    run_record): its schema, and the
                                                    run time and slot of its name

The market partitions (data_store/market, the VX settlements' included) are
not captures: Cboe and Yahoo serve them again, so they go in the data commit
and are re-derived from the run's database on a rebase conflict
(commit_and_push.sh REDERIVE).

The paths that pass are written to LIST (one per line) for
scripts/ci/commit_and_push.sh. A file that fails is reported (::error::) and
left out, a file with another name is left to the data commit; the exit status
is 1 if any recognised file failed, so the step shows red while the good
captures are still committed. New files cannot conflict on a rebase, so this
commit lands even when main moved.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATE = r"(?P<session>\d{4}-\d{2}-\d{2})"
PATTERNS = {
    "cboe_book": re.compile(rf"^data_store/options/cboe/(?P<und>[a-z]+)_{DATE}_\d{{6}}Z"
                            r"\.(?P<ext>parquet|json)$"),
    "cboe_manifest": re.compile(rf"^data_store/options/cboe_manifest/(?P<und>[a-z]+)_{DATE}_\d{{6}}Z\.json$"),
    "chain": re.compile(rf"^data_store/options/(?P<und>[a-z]+)_{DATE}\.parquet$"),
    "fit": re.compile(rf"^data_store/calibrations/joint_cal_{DATE}\.pkl$"),
    "ledger": re.compile(rf"^data_store/forward/ledger_{DATE}\.json$"),
    "run": re.compile(r"^data_store/runs/run_(?P<stamp>\d{8}T\d{6}Z)_(?P<slot>[a-z]+)\.json$"),
}
CAPTURE_DIRS = ("data_store/options", "data_store/calibrations", "data_store/forward", "data_store/runs")
RUN_SCHEMA = "run_ledger/1"
RUN_KEYS = ("schema", "run_utc", "slot", "complete", "steps", "expected_session", "calibration", "capture",
            "forward_ledger", "exit_code", "publishable", "publish_blockers", "timings_s", "github")


def new_files(root: Path) -> list[str]:
    """Untracked, not ignored files under the capture directories (repo-relative)."""
    r = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard", "--", *CAPTURE_DIRS,
                        ":(exclude)data_store/calibrations/diagnostics"],
                       cwd=root, capture_output=True, check=True)
    return sorted(p for p in r.stdout.decode().split("\0") if p)


def classify(rel: str) -> tuple[str | None, dict]:
    for kind, pat in PATTERNS.items():
        m = pat.match(rel)
        if m:
            return kind, m.groupdict()
    return None, {}


def _rows(path: Path) -> int:
    import pyarrow.parquet as pq
    return int(pq.read_metadata(path).num_rows)


def check_run_record(path: Path, stamp: str, slot: str) -> None:
    """A run record is whole: valid JSON of the run-ledger schema whose run time
    and slot are those of its name. A record left by a run that stopped before
    its end (complete: false, the pre-freeze write) is kept: it says so."""
    from datetime import UTC, datetime
    rec = json.loads(path.read_text())
    if not isinstance(rec, dict) or rec.get("schema") != RUN_SCHEMA:
        raise ValueError(f"not a {RUN_SCHEMA} record")
    missing = [k for k in RUN_KEYS if k not in rec]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")
    t = datetime.fromisoformat(str(rec["run_utc"])).astimezone(UTC)
    if f"{t:%Y%m%dT%H%M%SZ}" != stamp or rec["slot"] != slot:
        raise ValueError(f"the record is run {rec['run_utc']} ({rec['slot']}), the file {stamp} ({slot})")
    if rec["complete"] not in (True, False) or (rec["complete"] and not isinstance(rec["exit_code"], int)):
        raise ValueError(f"complete {rec['complete']!r} with exit code {rec['exit_code']!r}")


def _ledger_session(name: str) -> str:
    return name[len("ledger_"):-len(".json")]


def check_ledger_link(root: Path, path: Path, led: dict) -> None:
    """The forward ledger is a hash chain that only grows at its head
    (FUND_GRADE_PLAN 0.10): a new ledger must name the newest ledger before it
    (its session and the SHA-256 of its canonical JSON, forward_ledger.
    ledger_sha256), or null when there is none, and no committed ledger may
    come after it. A broken link is rejected here, before the capture commit,
    rather than found by test_forward_ledger.py's chain test on main."""
    from joint_vol_calibration.backtest import forward_ledger as fl
    s = led["session"]
    others = {p.name: p for p in path.parent.glob("ledger_*.json") if p != path}
    r = subprocess.run(["git", "ls-files", "-z", "--", str(path.parent.relative_to(root))], cwd=root,
                       capture_output=True, check=True)
    committed = {Path(x).name for x in r.stdout.decode().split("\0") if x}
    later = sorted(n for n in others if n in committed and _ledger_session(n) > s)
    if later:
        raise ValueError(f"{later[-1]} is committed after session {s}: "
                         "the ledger chain only grows at its head")
    before = sorted(n for n in others if _ledger_session(n) < s)
    head = fl.load_ledger(others[before[-1]]) if before else None
    want = None if head is None else {"session": head["session"], "sha256": fl.ledger_sha256(head)}
    if led.get("previous") != want:
        raise ValueError(f"previous {led.get('previous')} does not link to the head of the chain: {want}")


def check_manifest(path: Path, session: str) -> None:
    """The public record of a raw book held in the private store
    (cboe_downloader.write_manifest): its schema, stem and session, the three
    SHA-256s, a positive row count and the private path of its parquet."""
    from joint_vol_calibration.data import cboe_downloader as cboe
    man = json.loads(path.read_text())
    if not isinstance(man, dict) or man.get("schema") != cboe.MANIFEST_SCHEMA:
        raise ValueError(f"not a {cboe.MANIFEST_SCHEMA} manifest")
    if man.get("stem") != path.stem or man.get("session") != session:
        raise ValueError(f"the manifest is {man.get('stem')} of {man.get('session')}, the file {path.stem}")
    for k in ("sha256", "parquet_sha256", "sidecar_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(man.get(k, ""))):
            raise ValueError(f"no {k}")
    if not (isinstance(man.get("rows"), int) and man["rows"] >= 1):
        raise ValueError(f"rows {man.get('rows')!r}")
    want = f"{cboe.PRIVATE_BOOKS_SUBDIR}/{path.stem}.parquet"
    if man.get("private_path") != want or not man.get("private_repo"):
        raise ValueError(f"private path {man.get('private_path')!r} in {man.get('private_repo')!r}")


def check(root: Path, rel: str, kind: str, g: dict) -> None:
    """Raise ValueError (or an I/O / format error) unless ``rel`` is a whole capture."""
    path = root / rel
    if kind == "run":
        check_run_record(path, g["stamp"], g["slot"])
        return
    if kind == "cboe_manifest":
        check_manifest(path, g["session"])
        return
    s = g["session"]
    if kind == "cboe_book":
        pq_path, js_path = path.with_suffix(".parquet"), path.with_suffix(".json")
        if not (pq_path.exists() and js_path.exists()):
            raise ValueError("a Cboe capture is a parquet and its JSON; one of them is missing")
        meta = json.loads(js_path.read_text())
        n = _rows(pq_path)
        if n < 1 or meta.get("rows") != n:
            raise ValueError(f"{n} rows in the parquet, {meta.get('rows')} in the JSON")
        if meta.get("session") != s:
            raise ValueError(f"the JSON names session {meta.get('session')}, the file {s}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("sha256", ""))):
            raise ValueError("no sha256 of the raw response")
    elif kind == "chain":
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=["snapshot_date"])
        if t.num_rows < 1:
            raise ValueError("empty chain")
        dates = {str(d)[:10] for d in t.column("snapshot_date").to_pylist()}
        if dates != {s}:
            raise ValueError(f"snapshot_date {sorted(dates)[:3]} is not the file's session {s}")
    elif kind == "fit":
        from joint_vol_calibration.calibration.joint_calibrator import is_acceptable_calibration
        with open(path, "rb") as fh:
            cal = pickle.load(fh)  # noqa: S301 (the bot's own fit, written minutes ago)
        if not isinstance(cal, dict):
            raise ValueError(f"not a calibration result ({type(cal).__name__})")
        d = str(cal.get("spot_date") or cal.get("as_of_date"))[:10]
        if d != s:
            raise ValueError(f"the fit describes session {d}, the file {s}")
        ok, why = is_acceptable_calibration(cal)
        if not ok:
            raise ValueError(f"the fit does not pass the gate: {why}")
    elif kind == "ledger":
        from joint_vol_calibration.backtest import forward_ledger as fl
        led = fl.load_ledger(path)
        if led.get("session") != s:
            raise ValueError(f"the ledger is session {led.get('session')}'s, the file {s}")
        check_ledger_link(root, path, led)
    else:                                   # pragma: no cover (classify returns only these)
        raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="write the verified paths here")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--no-raw-cboe", action="store_true",
                    help="the private Cboe store is configured: a raw book is never listed (it is "
                         "reported, and left out of every public commit)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    ok, bad, other, private = [], [], [], []
    for rel in new_files(root):
        kind, g = classify(rel)
        if kind is None:
            other.append(rel)
            continue
        if kind == "cboe_book" and args.no_raw_cboe:
            private.append(rel)
            print(f"::warning title=Raw Cboe book::{rel} is still in the public tree: the private store "
                  "did not take it; it is left out of the public commit", flush=True)
            continue
        try:
            check(root, rel, kind, g)
        except Exception as exc:  # noqa: BLE001 (any failure to read it back fails the file)
            bad.append(rel)
            print(f"::error title=Capture integrity::{rel}: {type(exc).__name__}: {exc}", flush=True)
            continue
        ok.append(rel)
    args.out.write_text("".join(f"{p}\n" for p in ok))
    for p in ok:
        print(f"  ok     {p}")
    for p in other:
        print(f"  other  {p} (not a capture: left to the data commit)")
    for p in private:
        print(f"  private {p} (raw book: not for this public repository)")
    print(f"{len(ok)} capture file(s) verified, {len(bad)} failed, {len(other)} other new file(s)"
          + (f", {len(private)} raw book file(s) kept out" if private else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
