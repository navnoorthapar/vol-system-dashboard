#!/usr/bin/env python3
"""
cboe_private.py: keep the raw Cboe option books out of this public repository,
in a private data repository the owner creates (CITSEC-06; DATA_SOURCES.md).

    python scripts/ci/cboe_private.py checkout --dir DIR --repo OWNER/NAME
    python scripts/ci/cboe_private.py export   --dir DIR --repo OWNER/NAME --list LIST
    python scripts/ci/cboe_private.py push     --dir DIR --repo OWNER/NAME --message MSG
    python scripts/ci/cboe_private.py finish   --dir DIR --repo OWNER/NAME --list LIST

Cboe's terms bar storing and redistributing its quotes, and the daily bot
committed every full SPX and VIX book it captured to this public repository.
When the repository secrets CBOE_DATA_REPO (owner/name of a PRIVATE repository)
and CBOE_DATA_TOKEN (a fine-grained token with contents read/write on that
repository only) are both set, .github/workflows/daily_refresh.yml:

  checkout  clones the private repository (depth 1) into DIR, a directory under
            $RUNNER_TEMP, and prints DIR/options/cboe: the run passes it to
            cloud_refresh.py as CBOE_BOOKS_DIR, so every reader of raw books
            (cboe_downloader.find_books / book_captured: the forward ledger's
            marks, the retry slot's "already captured" test, later backfills and
            reproductions) finds the books stored there;
  export    copies each raw book this run captured (an untracked
            data_store/options/cboe/<stem>.parquet and its <stem>.json) into
            DIR/options/cboe/; LIST receives the public raw paths;
  push      commits DIR and pushes it, retrying (a rejected push is rebased onto
            the private repository's branch first);
  finish    after the push: for each book in LIST, checks that the private copy
            is byte-identical (SHA-256), writes its public manifest,
            data_store/options/cboe_manifest/<stem>.json (cboe_downloader.
            write_manifest: hashes, counts, timestamps, the private path; never
            a quote), and deletes the public raw copies. A manifest exists only
            for a book the private repository holds: after a failed push there
            is none, so a later slot captures the session again (its "already
            captured" test counts manifests), and the raw files stay on the
            runner, out of every public commit and artifact.

The token reaches git only as a one-command http.extraheader (never written to
.git/config, where later steps running PyPI code could read it). With the
secrets unset nothing here runs, and the workflow warns that raw books are
still committed publicly. Books committed before the move are not removed:
rewriting the public history is the owner's decision.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PUBLIC_BOOKS = "data_store/options/cboe"
STEM = re.compile(r"^[a-z]+_\d{4}-\d{2}-\d{2}_\d{6}Z$")
ATTEMPTS = 5
PAUSE_S = 10.0
USER = ("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com")


def repo_url(repo: str) -> str:
    """https://github.com/<owner>/<name>.git; a local path or file:// URL (the
    tests' bare repositories) is used as it is."""
    if "://" in repo or repo.startswith("/"):
        return repo
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"CBOE_DATA_REPO must be owner/name, got {repo!r}")
    return f"https://github.com/{repo}.git"


def _git(args: list[str], cwd: Path | None = None, token: str | None = None) -> str:
    cmd = ["git"]
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}"]
    cmd += ["-c", f"user.name={USER[0]}", "-c", f"user.email={USER[1]}", *args]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def books_dir(private: Path) -> Path:
    from joint_vol_calibration.data import cboe_downloader as cboe
    return Path(private) / cboe.PRIVATE_BOOKS_SUBDIR


def checkout(private: Path, repo: str, token: str | None, attempts: int = ATTEMPTS,
             pause: float = PAUSE_S) -> Path:
    """Clone the private repository (depth 1) into ``private``; its books directory."""
    private = Path(private)
    for i in range(1, attempts + 1):
        shutil.rmtree(private, ignore_errors=True)
        try:
            _git(["clone", "-q", "--depth", "1", repo_url(repo), str(private)], token=token)
            break
        except subprocess.CalledProcessError as e:
            if i == attempts:
                raise
            print(f"clone failed (try {i} of {attempts}): {e.stderr.strip()}", flush=True)
            time.sleep(pause * i)
    d = books_dir(private)
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_raw_books(root: Path) -> list[Path]:
    """This run's raw books: untracked <stem>.parquet under data_store/options/cboe
    whose <stem>.json is there too (a capture is the pair)."""
    r = subprocess.run(["git", "ls-files", "-z", "--others", "--exclude-standard", "--", PUBLIC_BOOKS],
                       cwd=root, capture_output=True, check=True)
    out = []
    for rel in sorted(p for p in r.stdout.decode().split("\0") if p):
        p = root / rel
        if p.suffix == ".parquet" and STEM.match(p.stem) and p.with_suffix(".json").is_file():
            out.append(p)
    return out


def export(root: Path, private: Path) -> list[Path]:
    """Copy each new raw book pair into the private checkout; returns the
    public raw paths (parquet and json)."""
    dest = books_dir(private)
    dest.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for pq_path in new_raw_books(root):
        for src in (pq_path, pq_path.with_suffix(".json")):
            tgt = dest / src.name
            if tgt.exists() and _sha256(tgt) != _sha256(src):
                raise ValueError(f"{tgt} exists in the private store with different content")
            shutil.copy2(src, tgt)
            moved.append(src)
    return moved


def push(private: Path, token: str | None, message: str, attempts: int = ATTEMPTS,
         pause: float = PAUSE_S) -> bool:
    """Commit the private checkout and push it; False when there was nothing to commit."""
    private = Path(private)
    _git(["add", "-A", "--", str(Path(books_dir(private)).relative_to(private))], cwd=private)
    if not _git(["status", "--porcelain"], cwd=private).strip():
        return False
    _git(["commit", "-qm", message], cwd=private)
    try:
        branch = _git(["symbolic-ref", "--short", "HEAD"], cwd=private).strip() or "main"
    except subprocess.CalledProcessError:
        branch = "main"
    for i in range(1, attempts + 1):
        try:
            _git(["push", "-q", "origin", f"HEAD:refs/heads/{branch}"], cwd=private, token=token)
            return True
        except subprocess.CalledProcessError as e:
            if i == attempts:
                raise
            print(f"push failed (try {i} of {attempts}): {e.stderr.strip()}; rebasing", flush=True)
            try:
                _git(["pull", "-q", "--rebase", "origin", branch], cwd=private, token=token)
            except subprocess.CalledProcessError as e2:
                print(f"  rebase failed: {e2.stderr.strip()}", flush=True)
            time.sleep(pause * i)
    return True


def finish(root: Path, private: Path, repo: str, listed: list[Path]) -> list[Path]:
    """After the push: check each listed public raw file against its private
    copy (byte-identical), write the manifest of each book, then delete the
    public raw copies. Nothing is written or deleted unless every copy checks."""
    from joint_vol_calibration.data import cboe_downloader as cboe
    dest = books_dir(private)
    for p in listed:
        q = dest / p.name
        if not (q.is_file() and _sha256(q) == _sha256(p)):
            raise ValueError(f"{p.name}: the private copy is missing or differs; the public copy is kept")
    for pq_path in (p for p in listed if p.suffix == ".parquet"):
        cboe.write_manifest(pq_path, cboe.manifest_dir(root / "data_store"), repo)
    for p in listed:
        p.unlink()
    return listed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument("command", choices=("checkout", "export", "push", "finish"))
    ap.add_argument("--dir", type=Path, required=True, help="the private checkout")
    ap.add_argument("--repo", default=os.environ.get("CBOE_DATA_REPO", ""))
    ap.add_argument("--list", type=Path, default=None,
                    help="export writes, finish reads, the public raw paths")
    ap.add_argument("--message", default="Raw Cboe books")
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    token = os.environ.get("CBOE_DATA_TOKEN") or None
    root = args.root.resolve()
    try:
        if args.command == "checkout":
            print(checkout(args.dir, args.repo, token))
        elif args.command == "export":
            moved = export(root, args.dir)
            if args.list:
                args.list.write_text("".join(f"{p.relative_to(root)}\n" for p in moved))
            print(f"{len(moved)} raw file(s) copied to the private checkout", flush=True)
        elif args.command == "push":
            print("pushed" if push(args.dir, token, args.message) else "nothing to commit", flush=True)
        else:
            listed = [root / ln for ln in (args.list.read_text().split() if args.list else [])]
            done = finish(root, args.dir, args.repo, listed)
            print(f"{len(done)} public raw file(s) replaced by manifests in "
                  "data_store/options/cboe_manifest/", flush=True)
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        detail = getattr(e, "stderr", "") or e
        print(f"::error title=Private Cboe store::{args.command} failed: {detail}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
