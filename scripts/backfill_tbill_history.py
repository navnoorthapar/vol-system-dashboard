#!/usr/bin/env python3
"""Backfill the full ^IRX (13-week T-bill) history into the database.

Why this exists
---------------
The backtest reports Sharpe and Sortino against a per-date cash hurdle and the
README advertises that hurdle as "Yahoo (^IRX), 2010 -> 2026, daily". That was
not true of the shipped database. Inside the 2018-01-01..2025-03-24 backtest
window it held exactly SEVEN rows -- 2020-01-02..2020-01-10, values
0.015, 0.015833, ... 0.020 -- which is ``np.linspace(0.015, 0.020, 7)`` from
``joint_vol_calibration/tests/test_rates.py``. A unit-test fixture wrote itself
into the tracked production database and was never cleaned up, and because
``compute_metrics`` forward-fills the rate series, those seven synthetic days
became the cash hurdle for five years: a flat 2.0% from 2020-01-13 through
2025-03-24 (when bills actually paid ~5% in 2023-24), and the 5% scalar
fallback across 2018-19 (when they actually paid ~2%).

The daily refresh bot only pulls a 400-day window, so it can never repair
history. This script does the one-time load; the bot's upserts then keep the
tail current. ``test_rates.py`` now snapshots and restores the rows it touches,
so the fixture cannot re-pollute the window.

Usage:  python scripts/backfill_tbill_history.py [--start 2010-01-01] [--apply]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import date


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_vol_calibration.data import yf_downloader as dl        # noqa: E402
from joint_vol_calibration.data.database import (                 # noqa: E402
    get_tbill_rates_series,
    init_database,
    insert_tbill_rates,
)

# The window every published backtest metric is computed over.
BT_START, BT_END = "2018-01-01", "2025-03-24"


def _coverage(as_of: str) -> tuple:
    """(row count, first date, last date) inside the backtest window."""
    df = get_tbill_rates_series(as_of_date=as_of, start_date=BT_START)
    if df.empty:
        return 0, None, None
    return len(df), str(df["date"].iloc[0])[:10], str(df["date"].iloc[-1])[:10]


def main(start: str, apply: bool) -> int:
    init_database()

    before = _coverage(BT_END)
    print(f"before: {before[0]} rows in {BT_START}..{BT_END} (first={before[1]} last={before[2]})")

    end = date.today().isoformat()
    rates = dl.download_tbill_rate(start=start, end=end)
    if rates.empty:
        print("download returned nothing; aborting")
        return 1

    # ^IRX is quoted as an annualised percentage and the downloader already
    # converts to decimal. Guard the unit anyway: a 4.2 here rather than 0.042
    # would silently make the cash hurdle 420%.
    bad = rates[(rates["rate"] <= 0) | (rates["rate"] > 0.25)]
    if len(bad):
        print(f"refusing to insert: {len(bad)} row(s) outside (0, 0.25] decimal range")
        print(bad.head().to_string())
        return 1

    print(f"downloaded {len(rates)} ^IRX rows ({rates['date'].iloc[0]} -> {rates['date'].iloc[-1]})")
    yearly = rates.assign(y=rates["date"].str[:4]).groupby("y")["rate"].mean() * 100
    print("annual mean (%):", ", ".join(f"{y}:{v:.2f}" for y, v in yearly.items()))

    if not apply:
        print("\ndry run -- pass --apply to write")
        return 0

    n = insert_tbill_rates(rates)
    after = _coverage(BT_END)
    print(f"inserted/replaced {n} rows")
    print(f"after:  {after[0]} rows in {BT_START}..{BT_END} (first={after[1]} last={after[2]})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--apply", action="store_true", help="write to the database")
    a = ap.parse_args()
    raise SystemExit(main(a.start, a.apply))
