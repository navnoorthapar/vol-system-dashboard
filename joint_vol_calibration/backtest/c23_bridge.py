"""
c23_bridge.py — C23 data-correction waterfall from the published C22 headline.

C23 made Cboe's official index files the source of the VIX index family
(^VIX9D/^VIX/^VIX3M/^VIX6M/^VVIX) the backtest reads, in place of Yahoo's
copies (pipeline.correct_vix_index_family; the per-row record is
data_store/market/corrections/c23_vix_index_family.csv), and re-synced the
database cache from that record (pipeline resync-db). The engine did not
change. This module measures what the data correction alone moves, one part
at a time, in a fixed order (cumulative; each row includes every row above):

   0  c22_published  the C22 canonical headline as published (PUBLISHED_C22,
                     c22_bridge.csv row 14 before C23): not re-run, the
                     pre-C23 database is not shipped
   1  restated       the corrected record with every material correction of
                     the log reverted: old values back on the replaced rows,
                     the Cboe-only VVIX sessions removed, the holiday rows
                     put back. What is left is the restatement of the same
                     2-decimal print (Yahoo serves float32, 22.129999 for
                     Cboe's 22.13; |change| <= 0.005, not logged per row), so
                     row 1 - row 0 is the effect of those restatements
   2  holidays       term-structure rows dated on NYSE holidays removed
   3  vvix_gaps      VVIX sessions Yahoo lacks, added from Cboe (the label
                     builder no longer carries the previous print forward)
   4  cboe_values    term-structure values replaced by Cboe's (each differed
                     from the Yahoo print by more than 0.005)

Row 4 runs the canonical engine on the corrected database, so it is the
canonical headline (run_backtest; c22_bridge.csv's last row) to the byte.

Rows 1-3 are built from the database plus the committed correction log, at
the long (date, ticker, close) level, and pivoted with the reader's own
database.vix_term_structure_wide, so the engine sees exactly the frame the
database would serve for that state of the record.

Usage (repository root):
  python -m joint_vol_calibration.backtest.c23_bridge
  python -m joint_vol_calibration.backtest.c23_bridge --csv data_store/backtest/c23_bridge.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.backtest.c22_bridge import SLEEVES, result_row
from joint_vol_calibration.data import database as db

logger = logging.getLogger("c23_bridge")

CORRECTION_LOG = db.MARKET_DIR / "corrections" / "c23_vix_index_family.csv"

# The C22 canonical run as published: data_store/backtest/c22_bridge.csv row 14
# (= full_results.pkl) at commit 44c4f0f, before the C23 database resync.
PUBLISHED_C22 = {
    "nav": 783908.214, "pnl": -216091.786, "sharpe": -0.8793707534, "max_dd": -0.216091786,
    "s1c_pnl": -431202.5601, "s1c_sharpe": -0.5704914886, "s1c_win": 0.3495934959, "s1c_trades": 123,
    "s3_pnl": 56471.40628, "s3_sharpe": 0.1809473688, "s3_win": 0.4871794872, "s3_trades": 39,
    "s4_pnl": -273544.2042, "s4_sharpe": -0.787253591, "s4_win": 0.3695652174, "s4_trades": 46,
    "s1_pnl": -287644.4945, "s1_sharpe": -0.1487085509, "s1_win": 0.5942028986, "s1_trades": 69,
    "s2_pnl": -149899.9393, "s2_sharpe": -0.3802019245, "s2_win": 0.3777777778, "s2_trades": 45,
    "combined_pnl": -343653.1048, "combined_sharpe": -0.8179197211, "combined_win": 0.3153153153,
    "combined_trades": 111,
}

STEPS = (
    ("c22_published", "C22 canonical as published (pre-C23 record)"),
    ("restated", "same 2-decimal prints restated as Cboe's (|change| <= 0.005)"),
    ("holidays", "term-structure rows on NYSE holidays removed"),
    ("vvix_gaps", "VVIX sessions missing from Yahoo added from Cboe"),
    ("cboe_values", "term-structure values replaced by Cboe's (|change| > 0.005)"),
)


def load_correction_log(path: Path | None = None) -> pd.DataFrame:
    """The vix_term_structure rows of the committed C23 correction log."""
    log = pd.read_csv(path or CORRECTION_LOG, dtype={"date": str, "ticker": str})
    return log[log["table"] == "vix_term_structure"].reset_index(drop=True)


def record_states(long_now: pd.DataFrame, log: pd.DataFrame) -> dict:
    """Long term-structure frames for rows 1-4, from the corrected rows and the log.

    Built backwards from the corrected record (row 4): revert the replaced
    values (row 3), drop the Cboe-only additions (row 2), restore the
    holiday rows (row 1). Returns {step: long frame [date, ticker, close]}.
    """
    keys = ["date", "ticker"]
    now = long_now[["date", "ticker", "close"]].copy()
    now["date"] = now["date"].astype(str).str[:10]
    lo, hi = now["date"].min(), now["date"].max()
    log = log[(log["date"] >= lo) & (log["date"] <= hi)]

    def by(action: str) -> pd.DataFrame:
        return log.loc[log["action"] == action, ["date", "ticker", "old_close"]]

    states = {"cboe_values": now}
    rep = by("replaced_by_cboe").set_index(keys)["old_close"]
    s3 = now.set_index(keys)
    missing = rep.index.difference(s3.index)
    if len(missing):
        raise ValueError(f"correction log rows not in the database: {list(missing)[:5]}")
    s3.loc[rep.index, "close"] = rep.to_numpy(float)
    states["vvix_gaps"] = s3.reset_index()
    add = pd.MultiIndex.from_frame(by("added_from_cboe")[keys])
    s2 = s3.drop(index=s3.index.intersection(add))
    states["holidays"] = s2.reset_index()
    hol = by("removed_non_session").rename(columns={"old_close": "close"})
    s1 = pd.concat([s2.reset_index(), hol], ignore_index=True)
    states["restated"] = s1.sort_values(keys).reset_index(drop=True)
    return {k: v.sort_values(keys).reset_index(drop=True) for k, v in states.items()}


def run_bridge(log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows 0-4 (see the module docstring); P&L in $ on $1M."""
    from joint_vol_calibration.backtest.run_backtest import BT_END, DATA_START, _load_data, load_pdv_vol

    spx, vix_now = _load_data()
    pdv = load_pdv_vol()
    long_now = db.get_vix_term_structure(BT_END, DATA_START)
    states = record_states(long_now, load_correction_log() if log is None else log)
    rows = [{"step": STEPS[0][0], "label": STEPS[0][1], **PUBLISHED_C22}]
    for key, label in STEPS[1:]:
        vix = vix_now if key == "cboe_values" else db.vix_term_structure_wide(states[key])
        t0 = time.time()
        eng = bte.BacktestEngine(pdv_vol_series=pdv)
        eq = eng.run(spx, vix)
        rows.append({"step": key, "label": label, **result_row(eng, eq, eng.compute_metrics())})
        logger.info("step %-12s done in %.1fs", key, time.time() - t0)
    df = pd.DataFrame(rows)
    df["d_pnl"] = df["pnl"].diff().fillna(0.0)
    return df


def main(argv: list | None = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)
    ap = argparse.ArgumentParser(description="C23 data-correction waterfall")
    ap.add_argument("--csv", help="write the rows as CSV (canonical: data_store/backtest/c23_bridge.csv)")
    args = ap.parse_args(argv)
    df = run_bridge()
    cols = ["step", "nav", "pnl", "d_pnl", "sharpe", "max_dd"] + [f"{s}_pnl" for s in SLEEVES]
    print(df[cols].to_string(float_format=lambda x: f"{x:,.4f}"))
    if args.csv:
        df.to_csv(args.csv, index_label="row", float_format=bte.ARTIFACT_CSV_FLOAT_FORMAT)
    return df


if __name__ == "__main__":
    main()
