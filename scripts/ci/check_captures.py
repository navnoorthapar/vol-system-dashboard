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
without its diagnostics, data_store/forward) is recognised by its name and
read back:

  options/cboe/<und>_<session>_<HHMMSS>Z.parquet   readable, >= 1 row, its JSON
           + .json                                  pair present with the same row
                                                    count, session and a sha256
  options/<und>_<session>.parquet                   readable, >= 1 row, every
                                                    snapshot_date the session's
  calibrations/joint_cal_<session>.pkl              unpickles to a fit of that
                                                    session that passes the gate
  forward/ledger_<session>.json                     a valid ledger of that session

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
    "chain": re.compile(rf"^data_store/options/(?P<und>[a-z]+)_{DATE}\.parquet$"),
    "fit": re.compile(rf"^data_store/calibrations/joint_cal_{DATE}\.pkl$"),
    "ledger": re.compile(rf"^data_store/forward/ledger_{DATE}\.json$"),
}
CAPTURE_DIRS = ("data_store/options", "data_store/calibrations", "data_store/forward")


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


def check(root: Path, rel: str, kind: str, g: dict) -> None:
    """Raise ValueError (or an I/O / format error) unless ``rel`` is a whole capture."""
    path = root / rel
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
    else:                                   # pragma: no cover (classify returns only these)
        raise ValueError(kind)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="write the verified paths here")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    ok, bad, other = [], [], []
    for rel in new_files(root):
        kind, g = classify(rel)
        if kind is None:
            other.append(rel)
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
    print(f"{len(ok)} capture file(s) verified, {len(bad)} failed, {len(other)} other new file(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
