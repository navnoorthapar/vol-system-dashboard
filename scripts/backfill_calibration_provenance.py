#!/usr/bin/env python3
"""Backfill data-provenance fields into historical calibration pickles.

Calibrations written before the provenance fix recorded only parameters and
losses — not how much market data produced them, nor WHICH option chain.
That omission is what let an RMSE-ranked selector headline the 2026-07-23 fit,
which was produced on a day the VIX term structure had collapsed to a single
tenor (see ``latest_good_calibration_path``).

Counts. They are recovered from the Daily Market Refresh CI logs, which print

    [C4] Joint calibration: <date>  |  S=...  |  <n> SPX opts  |  <m> VIX tenors

and are the authoritative record of what each fit actually saw. Fits that
predate the daily bot (2026-03-24, 2026-05-31) have no CI run; their counts are
left as None rather than guessed, and are marked as legacy.

The SPX chain. Until 2026-09-08 the bot never stored the chain it downloaded
(it wrote to a table nothing read), so every fit refitted the newest COMMITTED
snapshot on or before its date: the 2026-05-29 quotes (stored as "2026-05-31"
before C22 re-dated weekend snapshots to their session) for the 06-16 fit, and
the 2026-06-18 quotes (stored as "2026-06-20") for every fit from 06-23 to
09-04 — up to 78 days stale, paired with a same-day VIX leg. That is derived
deterministically from the committed snapshot list and recorded as
``spx_snapshot_date``, ``spx_chain_age_days`` (calendar days from the chain's
session to the fit's spot session) and ``stale_spx_chain``. Pickles that
recorded ``spx_snapshot_date`` live (from 2026-09-08) are never touched.

Derived diagnostics (feller_state, feller_margin, pinned_params) are computed
from the stored parameters — no new information is invented.

Non-destructive and idempotent: a key is written only when it is missing or
None, a value already recorded is never overwritten, and a pickle with nothing
to add is not rewritten — so a second run changes no byte.

Usage:  python scripts/backfill_calibration_provenance.py [--apply]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_vol_calibration.calibration.joint_calibrator import (  # noqa: E402
    classify_feller,
    count_pinned_params,
)
from joint_vol_calibration.data import database as db  # noqa: E402

CAL_DIR = ROOT / "data_store" / "calibrations"

# From this date on, cloud_refresh stores the chain it downloads and records
# spx_snapshot_date itself; earlier fits used the newest committed snapshot.
CHAIN_PERSISTED_FROM = "2026-09-08"

# Recovered from CI logs (run IDs in the Daily Market Refresh workflow).
# date -> (n_spx_options, n_vix_tenors)
CI_COUNTS: dict[str, tuple[int, int]] = {
    "2026-06-16": (54, 4),
    "2026-06-23": (54, 4),
    "2026-07-01": (54, 4),
    "2026-07-02": (54, 4),
    "2026-07-03": (54, 4),
    "2026-07-07": (54, 4),
    "2026-07-09": (54, 4),
    "2026-07-10": (45, 4),
    "2026-07-13": (45, 4),
    "2026-07-14": (45, 1),
    "2026-07-15": (45, 1),
    "2026-07-16": (45, 1),
    "2026-07-17": (45, 1),
    "2026-07-20": (45, 4),
    "2026-07-22": (45, 1),
    "2026-07-23": (45, 1),
    "2026-08-04": (36, 4),
    "2026-08-07": (36, 4),
    "2026-08-19": (36, 1),
    "2026-08-20": (36, 1),
    "2026-08-21": (36, 1),
    "2026-08-24": (36, 1),
    "2026-08-25": (36, 1),
    "2026-08-27": (36, 4),
    "2026-08-28": (36, 1),
    "2026-08-29": (36, 4),
    "2026-09-01": (36, 4),
    "2026-09-02": (36, 4),
    "2026-09-03": (36, 4),
    "2026-09-04": (36, 4),
}


def committed_spx_snapshots() -> list[str]:
    """SPX snapshot dates in the database (the committed chains a pre-fix fit could see)."""
    return db.get_options_snapshot_dates("SPX")


def derived_chain(fit_date: str, snapshots: list[str]) -> dict:
    """Which committed chain a pre-2026-09-08 fit dated ``fit_date`` refitted, and how stale it was."""
    used = max((s for s in snapshots if s <= fit_date), default=None)
    if used is None:
        return {}
    spot = db.session_on_or_before(fit_date)
    age = (db._day(spot) - db._day(used)).days
    return {
        "spx_snapshot_date": used,
        "spx_chain_age_days": int(age),
        "stale_spx_chain": used != spot,
        "spx_snapshot_provenance": (
            "derived: newest committed SPX snapshot on or before the fit date "
            "(the bot did not store its own chains before 2026-09-08)"),
    }


def proposed_updates(date: str, cal: dict, snapshots: list[str]) -> dict:
    """Every provenance key this pickle lacks, with its value. Never includes a recorded key."""
    p = cal.get("params", {})
    candidates: dict = {}
    if p:
        state, margin = classify_feller(p["kappa"], p["theta"], p["sigma"])
        candidates.update({
            "feller_state":  state,
            "feller_margin": float(margin),
            "pinned_params": count_pinned_params(p, state),
        })
    counts_missing = cal.get("n_spx_options") is None or cal.get("n_vix_tenors") is None
    if counts_missing and date in CI_COUNTS:
        n_spx, n_ten = CI_COUNTS[date]
        candidates.update({"n_spx_options": n_spx, "n_vix_tenors": n_ten,
                           "provenance": "counts backfilled from CI logs"})
    elif counts_missing and date < CHAIN_PERSISTED_FROM:
        # Predates the daily bot — do not invent counts.
        candidates.update({"n_spx_options": None, "n_vix_tenors": None,
                           "provenance": "legacy fit; data counts unrecorded"})
    # A fit whose counts were recorded live gets no provenance label at all.
    if date < CHAIN_PERSISTED_FROM and "spx_snapshot_date" not in cal:
        candidates.update(derived_chain(date, snapshots))

    def missing(key):
        return key not in cal or (cal[key] is None and candidates[key] is not None)

    return {k: v for k, v in candidates.items() if missing(k)}


def backfill(apply: bool = False, cal_dir: Path | None = None,
             snapshots: list[str] | None = None) -> int:
    """Fill missing provenance keys. Returns the number of pickles (that would be) rewritten."""
    cal_dir = Path(cal_dir) if cal_dir is not None else CAL_DIR
    snapshots = committed_spx_snapshots() if snapshots is None else snapshots
    changed = 0
    for path in sorted(cal_dir.glob("joint_cal_*.pkl")):
        date = path.stem.replace("joint_cal_", "")
        with open(path, "rb") as f:
            cal = pickle.load(f)
        if not cal.get("params"):
            continue

        updates = proposed_updates(date, cal, snapshots)
        merged = {**cal, **updates}
        stale = merged.get("stale_spx_chain")
        print(
            f"{date}  tenors={str(merged.get('n_vix_tenors')):>4}  "
            f"chain={merged.get('spx_snapshot_date')}"
            f"{' (STALE, ' + str(merged.get('spx_chain_age_days')) + 'd)' if stale else ''}  "
            f"adds={','.join(sorted(updates)) or '-'}"
        )
        if not updates:
            continue
        changed += 1
        if apply:
            with open(path, "wb") as f:
                pickle.dump(merged, f)
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes to disk")
    args = ap.parse_args()
    n = backfill(apply=args.apply)
    print(f"\n{'Updated' if args.apply else 'Would update'} {n} file(s).")
    if not args.apply:
        print("Dry run — re-run with --apply to write.")
