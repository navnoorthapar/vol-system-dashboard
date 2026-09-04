#!/usr/bin/env python3
"""Backfill data-provenance fields into historical calibration pickles.

Calibrations written before the provenance fix recorded only parameters and
losses — not how much market data produced them. That omission is what let an
RMSE-ranked selector headline the 2026-07-23 fit, which was produced on a day
the VIX term structure had collapsed to a single tenor (see
``latest_good_calibration_path``).

The counts are recovered from the Daily Market Refresh CI logs, which print

    [C4] Joint calibration: <date>  |  S=...  |  <n> SPX opts  |  <m> VIX tenors

and are the authoritative record of what each fit actually saw. Fits that
predate the daily bot (2026-03-24, 2026-05-31) have no CI run; their counts are
left as None rather than guessed, and are marked as legacy.

Derived diagnostics (feller_state, feller_margin, pinned_params) are recomputed
from the stored parameters — no new information is invented.

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

CAL_DIR = ROOT / "data_store" / "calibrations"

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


def backfill(apply: bool = False) -> int:
    changed = 0
    for path in sorted(CAL_DIR.glob("joint_cal_*.pkl")):
        date = path.stem.replace("joint_cal_", "")
        with open(path, "rb") as f:
            cal = pickle.load(f)

        p = cal.get("params", {})
        if not p:
            continue

        state, margin = classify_feller(p["kappa"], p["theta"], p["sigma"])
        updates = {
            "feller_state":  state,
            "feller_margin": float(margin),
            "pinned_params": count_pinned_params(p, state),
        }

        if date in CI_COUNTS:
            n_spx, n_ten = CI_COUNTS[date]
            updates["n_spx_options"] = n_spx
            updates["n_vix_tenors"] = n_ten
            updates["provenance"] = "counts backfilled from CI logs"
        else:
            # Predates the daily bot — do not invent counts.
            updates["n_spx_options"] = None
            updates["n_vix_tenors"] = None
            updates["provenance"] = "legacy fit; data counts unrecorded"

        pinned = ", ".join(updates["pinned_params"]) or "none"
        print(
            f"{date}  tenors={str(updates['n_vix_tenors']):>4}  "
            f"Feller={state:<8} margin={margin:+.2e}  pinned={pinned}"
        )

        if apply:
            cal.update(updates)
            with open(path, "wb") as f:
                pickle.dump(cal, f)
            changed += 1
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes to disk")
    args = ap.parse_args()
    n = backfill(apply=args.apply)
    print(f"\n{'Updated' if args.apply else 'Would update'} "
          f"{n if args.apply else len(list(CAL_DIR.glob('joint_cal_*.pkl')))} file(s).")
    if not args.apply:
        print("Dry run — re-run with --apply to write.")
