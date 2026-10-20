#!/usr/bin/env python3
"""Download Cboe strategy-benchmark index levels into data_store/benchmarks/.

What is fetched (Cboe's public index-history CSVs; cdn.cboe.com 307-redirects
to cdn-api.cboe.com and the redirect is followed and recorded):

    PUT   WPUT  BXM  BFLY  CNDR  VXTH    strategy indices (stats/benchmarks.py)
    SET                                  the SPX exercise-settlement value, used
                                         to settle the monthly puts of the PUT
                                         replication (AM settlement)

What is stored: for each index a two-column CSV (date, close) of daily
closing LEVELS from ``--start`` (default 2017-01-01, one year before the
backtest window) to the last published day, plus manifest.json recording, per
file, the requested and final URL, HTTP status, retrieval time (UTC), the
SHA-256 of the downloaded bytes and of the stored file, row counts and date
range. Only index levels are kept, and only from --start: these are Cboe's
published values, used here under Cboe's website terms (personal,
non-commercial use); the data-licensing register decides whether they stay
in the public repo.

Usage:
    python scripts/fetch_cboe_benchmarks.py [--start 2017-01-01] [--out data_store/benchmarks]
                                            [--index PUT --index BFLY ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests                                                     # noqa: E402

from joint_vol_calibration.stats.benchmarks import (               # noqa: E402
    BENCHMARK_INDICES,
    CBOE_INDEX_URL,
    parse_cboe_index_csv,
)

SETTLEMENT_SERIES = ("SET",)
DEFAULT_OUT = ROOT / "data_store" / "benchmarks"
USER_AGENT = "Mozilla/5.0 (research; vol-system-dashboard benchmark fetch)"


def fetch(index: str, session: requests.Session, timeout: float = 60.0) -> dict:
    url = CBOE_INDEX_URL.format(index=index)
    resp = session.get(url, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return {"index": index, "url": url, "final_url": resp.url, "status": resp.status_code,
            "redirects": [r.status_code for r in resp.history], "content": resp.content,
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def store(got: dict, out_dir: pathlib.Path, start: str) -> dict:
    df = parse_cboe_index_csv(got["content"].decode("utf-8"), got["index"])
    n_all = len(df)
    df = df[df["date"] >= start].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{got['index']}: no rows on or after {start}")
    path = out_dir / f"{got['index']}.csv"
    text = df.to_csv(index=False, float_format="%.6f", lineterminator="\n")
    path.write_text(text)
    return {
        "file": path.name, "url": got["url"], "final_url": got["final_url"],
        "http_status": got["status"], "redirect_statuses": got["redirects"],
        "retrieved_at_utc": got["retrieved_at"],
        "sha256_download": hashlib.sha256(got["content"]).hexdigest(),
        "sha256_stored": hashlib.sha256(text.encode()).hexdigest(),
        "rows_published": n_all, "rows_stored": len(df),
        "first_date": df["date"].iloc[0], "last_date": df["date"].iloc[-1],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--index", action="append", help="index to fetch (repeatable); default: all")
    args = ap.parse_args(argv)
    names = args.index or list(BENCHMARK_INDICES) + list(SETTLEMENT_SERIES)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("source", "Cboe Global Markets index history CSVs (cdn.cboe.com)")
    manifest.setdefault("terms", "https://www.cboe.com/terms (personal, non-commercial use); "
                                 "levels only, stored from the start date below")
    manifest["start"] = args.start
    files = manifest.setdefault("files", {})
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,*/*;q=0.8"})
    failed = []
    for nm in names:
        try:
            files[nm] = store(fetch(nm, s), args.out, args.start)
            f = files[nm]
            print(f"{nm:5s} {f['rows_stored']:5d} rows {f['first_date']} → {f['last_date']}  "
                  f"({f['final_url']})")
        except Exception as exc:                      # noqa: BLE001 — report every index, then fail
            failed.append(nm)
            print(f"{nm:5s} FAILED: {exc}", file=sys.stderr)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
