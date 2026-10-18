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
tail current. ``test_rates.py`` now runs against a temporary copy of the
schema and never opens the tracked database.

Persistence: the database is not the record. Every bot run starts by
upserting the committed yearly partitions (data_store/market/tbill_rates/,
database.import_market_partitions) into it, so a fix written only to the
database is silently undone for every year older than the bot's 400-day
export window. With --apply the script therefore also merges what it wrote
into those partitions (export_market_partitions, every year from --start);
commit them together with the database.

Units (C22): ^IRX is a bank-discount yield in percent. The downloader now
stores the continuously compounded rate the pricers assume
(``database.tbill_discount_to_cc``) next to the raw quote, and keeps zero and
negative prints -- the 2020-03-19..27 bills printed below zero, and the old
"> 0" filter silently left those seven sessions without a rate.

Usage:  python scripts/backfill_tbill_history.py [--start 2010-01-01] [--end YYYY-MM-DD] [--apply]
"""
from __future__ import annotations

import argparse
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joint_vol_calibration.config import BT_END, BT_START          # noqa: E402
from joint_vol_calibration.data import yf_downloader as dl        # noqa: E402
from joint_vol_calibration.data.database import (                 # noqa: E402
    export_market_partitions,
    get_tbill_rates_series,
    init_database,
    insert_tbill_rates,
    last_closed_session,
)

# Plausible range for a continuously compounded 13-week bill rate, as a
# decimal. The upper bound catches a percent/decimal mix-up (4.2 rather than
# 0.042 would make the cash hurdle 420%); the lower bound admits the genuine
# negative prints of March 2020 (as low as -0.105%) but not a sign error.
RATE_RANGE = (-0.01, 0.25)


def out_of_range(rates):
    """Rows whose rate is outside (RATE_RANGE[0], RATE_RANGE[1]]."""
    lo, hi = RATE_RANGE
    return rates[(rates["rate"] <= lo) | (rates["rate"] > hi) | rates["rate"].isna()]


def _coverage(as_of: str) -> tuple:
    """(row count, first date, last date) inside the backtest window."""
    df = get_tbill_rates_series(as_of_date=as_of, start_date=BT_START)
    if df.empty:
        return 0, None, None
    return len(df), str(df["date"].iloc[0])[:10], str(df["date"].iloc[-1])[:10]


def main(start: str, apply: bool, end: str | None = None) -> int:
    init_database()

    before = _coverage(BT_END)
    print(f"before: {before[0]} rows in {BT_START}..{BT_END} (first={before[1]} last={before[2]})")

    # Not the machine's date: during a session that would fetch a still-forming bar.
    end = end or last_closed_session()
    rates = dl.download_tbill_rate(start=start, end=end)
    if rates.empty:
        print("download returned nothing; aborting")
        return 1

    bad = out_of_range(rates)
    if len(bad):
        lo, hi = RATE_RANGE
        print(f"refusing to insert: {len(bad)} row(s) outside ({lo}, {hi}] decimal range")
        print(bad.head().to_string())
        return 1

    print(f"downloaded {len(rates)} ^IRX rows ({rates['date'].iloc[0]} -> {rates['date'].iloc[-1]}); "
          f"{int((rates['rate'] <= 0).sum())} at or below zero")
    yearly = rates.assign(y=rates["date"].str[:4]).groupby("y")["rate"].mean() * 100
    print("annual mean, continuously compounded (%):",
          ", ".join(f"{y}:{v:.2f}" for y, v in yearly.items()))

    if not apply:
        print("\ndry run -- pass --apply to write")
        return 0

    n = insert_tbill_rates(rates)
    after = _coverage(BT_END)
    print(f"inserted/replaced {n} rows")
    print(f"after:  {after[0]} rows in {BT_START}..{BT_END} (first={after[1]} last={after[2]})")
    # The committed partitions are what the bot restores from: without this
    # its next run re-imports the old values for every year it does not export.
    written = export_market_partitions(tables=["tbill_rates"], since=start)
    if written:
        print(f"merged into {len(written)} partition file(s); commit them:"
              + "".join(f"\n  {p}" for p in written))
    else:
        print("the data_store/market/tbill_rates partitions already hold these rows")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="inclusive end date (default: the last closed NYSE session)")
    ap.add_argument("--apply", action="store_true",
                    help="write to the database and the data_store/market partitions")
    a = ap.parse_args()
    raise SystemExit(main(a.start, a.apply, a.end))
