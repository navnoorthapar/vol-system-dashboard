#!/usr/bin/env python3
"""
surface_report.py — the quote-grade surface layer on every committed SPX chain.

For each data_store/options/spx_<session>.parquet: clean the chain (the
calibrator's universe), screen and repair static arbitrage, fit the eSSVI,
per-expiry SVI and calendar-ordered SVI (SVI-cal) surfaces, score them and the
stored model_version-2 Heston fit in spread units, and replicate the VIX from
the same chain. Writes
data_store/surface/<session>/{residuals.parquet, surface.json,
scorecard.json, vix_replication.json} and data_store/surface/index.json
(schema: joint_vol_calibration/surface/report.py).

    python scripts/surface_report.py                    # every stored session
    python scripts/surface_report.py --sessions 2026-10-16
    python scripts/surface_report.py --out /tmp/surface  # somewhere else

Deterministic: same inputs -> same bytes on the same machine (floats rounded
to 10 significant digits, sorted keys, no timestamps). Reads only committed
data through the repo's look-ahead-gated loaders; writes nothing but the
artifacts.
"""
import os

# Single-threaded BLAS before numpy loads: the problems are small and dense,
# and on a shared machine OpenBLAS thread oversubscription slowed the eSSVI
# fit by two orders of magnitude.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse                                                        # noqa: E402
import logging                                                         # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402
from pathlib import Path                                               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint_vol_calibration.surface import quotes as sqm               # noqa: E402
from joint_vol_calibration.surface import report                       # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sessions", nargs="*", help="sessions (default: every stored SPX chain)")
    ap.add_argument("--out", type=Path, default=report.SURFACE_DIR, help="output root")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR,
                        format="%(levelname)s %(name)s: %(message)s")
    sessions = args.sessions or sqm.stored_sessions("SPX")
    rows, failed = [], []
    for s in sessions:
        t0 = time.perf_counter()
        try:
            row = report.write_session(s, args.out)
        except (ValueError, FileNotFoundError) as exc:   # unusable chain: report, keep going
            failed.append(s)
            print(f"{s}: FAILED: {exc}", file=sys.stderr)
            continue
        dt = time.perf_counter() - t0
        rows.append(row)
        m = row["models"]
        line = "  ".join(f"{k} {v['inside_spread_pct']:.1f}% in / {v['rmse_vp']:.2f}vp"
                         for k, v in m.items())
        vix = (f"VIX {row['vix_replicated']:.2f} vs {row['vix_official']:.2f}"
               if row.get("vix_replicated") and row.get("vix_official") else "VIX n/a")
        print(f"{s}: {row['n_quotes']} quotes, {row['arbitrage_quotes_outside_spread']} need a "
              f"repair outside their spread | {line} | {vix} | {dt:.1f}s")
    if rows:
        existing = []
        idx = Path(args.out) / "index.json"
        if idx.exists():
            import json
            existing = [r for r in json.loads(idx.read_text())["sessions"]
                        if r["session"] not in {x["session"] for x in rows}]
        report.write_index(existing + rows, args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
