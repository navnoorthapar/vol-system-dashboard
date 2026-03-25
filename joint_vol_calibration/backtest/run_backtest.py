"""
run_backtest.py — C10: Live Backtest Runner

Loads real 2018-2025 data, runs BacktestEngine, optionally runs walk-forward
validation, generates full_report.html, and saves results to parquet.

Usage:
  cd "Volatility Trading System"
  python -m joint_vol_calibration.backtest.run_backtest
  python -m joint_vol_calibration.backtest.run_backtest --no-wf   # skip walk-forward
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")

# ── Imports ───────────────────────────────────────────────────────────────────
from joint_vol_calibration.data.database import (
    get_spx_ohlcv,
    get_vix_term_structure_wide,
)
from joint_vol_calibration.backtest.backtest_engine import BacktestEngine
from joint_vol_calibration.backtest.report_generator import ReportGenerator


# ── Data range ────────────────────────────────────────────────────────────────
# Load from 2015 so signal rolling windows (252-day lookbacks) are warm by 2018
DATA_START = "2015-01-01"
BT_END     = "2025-03-24"


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPX OHLCV and VIX wide from SQLite."""
    logger.info("Loading SPX OHLCV (%s → %s) …", DATA_START, BT_END)
    spx_df = get_spx_ohlcv(as_of_date=BT_END, start_date=DATA_START)
    logger.info("  SPX rows: %d  (%.4f → %.4f)", len(spx_df),
                spx_df["close"].iloc[0], spx_df["close"].iloc[-1])

    logger.info("Loading VIX term structure (%s → %s) …", DATA_START, BT_END)
    vix_wide = get_vix_term_structure_wide(as_of_date=BT_END, start_date=DATA_START)
    logger.info("  VIX rows: %d  cols: %s", len(vix_wide), list(vix_wide.columns))

    return spx_df, vix_wide


def _print_summary(metrics: dict, crisis: dict) -> None:
    """Print key metrics to stdout."""
    m = metrics
    sep = "─" * 60

    def _p(x):  return f"{x*100:.2f}%"   if x is not None and pd.notna(x) else "—"
    def _f(x):  return f"{x:.3f}"         if x is not None and pd.notna(x) else "—"
    def _d(x):  return f"${x:,.0f}"       if x is not None and pd.notna(x) else "—"
    def _i(x):  return str(int(x))         if x is not None and pd.notna(x) else "—"

    print(f"\n{sep}")
    print("  C10 BACKTEST RESULTS — 2018-01-01 → 2025-03-24")
    print(sep)
    print(f"  Cumulative return :  {_p(m.get('cumulative_return'))}")
    print(f"  Ann. return       :  {_p(m.get('ann_return'))}")
    print(f"  Sharpe (rf=5%)    :  {_f(m.get('sharpe'))}")
    print(f"  Sortino           :  {_f(m.get('sortino'))}")
    print(f"  Max Drawdown      :  {_p(m.get('max_drawdown'))}")
    print(f"  DD Duration       :  {_i(m.get('dd_duration_days'))} days")
    print(f"  Calmar            :  {_f(m.get('calmar'))}")
    print(f"  Win Rate          :  {_p(m.get('win_rate'))}")
    print(f"  Avg P&L / trade   :  {_d(m.get('avg_pnl_per_trade'))}")
    print(f"  Best day          :  {_p(m.get('best_day'))}")
    print(f"  Worst day         :  {_p(m.get('worst_day'))}")
    print(f"  N trades          :  {_i(m.get('n_trades'))}")
    print(f"  N days            :  {_i(m.get('n_days'))}")
    print(sep)

    print("\n  Per-signal breakdown:")
    hdr = f"  {'Signal':<12} {'Sharpe':>8} {'AnnRet':>8} {'WinRate':>9} {'NTrades':>9} {'TotalP&L':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for sig, lbl in [("s1","S1 IVR"),("s2","S2 VIX TS"),("s3","S3 Disp"),("combined","Combined")]:
        sh = m.get(f"{sig}_sharpe")
        ar = m.get(f"{sig}_ann_return")
        wr = m.get(f"{sig}_win_rate")
        nt = m.get(f"{sig}_n_trades")
        tp = m.get(f"{sig}_total_pnl")
        print(f"  {lbl:<12} {_f(sh):>8} {_p(ar):>8} {_p(wr):>9} {_i(nt):>9} {_d(tp):>12}")

    print(f"\n  Crisis performance:")
    for name, d in crisis.items():
        if d.get("n_days", 0) == 0:
            print(f"  {name:<20} — no data")
            continue
        tr = d.get("total_return", 0) * 100
        dd = d.get("max_drawdown", 0) * 100
        sh = d.get("sharpe", 0)
        sign = "+" if tr >= 0 else ""
        print(f"  {name:<20}  return={sign}{tr:.1f}%  maxDD={dd:.1f}%  Sharpe={sh:.2f}")
    print(sep)


def main(run_wf: bool = True) -> None:
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    spx_df, vix_wide = _load_data()

    # ── 2. Run full backtest ──────────────────────────────────────────────────
    logger.info("Running BacktestEngine (2018-01-01 → 2025-03-24) …")
    engine    = BacktestEngine()
    equity_df = engine.run(spx_df, vix_wide)
    logger.info("Backtest done: %d days | NAV $%.0f → $%.0f | trades: %d",
                len(equity_df),
                engine.initial_capital,
                float(equity_df["nav"].iloc[-1]),
                len(engine.all_trades))

    # ── 3. Metrics + crisis ───────────────────────────────────────────────────
    metrics = engine.compute_metrics()
    crisis  = engine.crisis_performance()
    _print_summary(metrics, crisis)

    # ── 4. Walk-forward validation ────────────────────────────────────────────
    wf_df: pd.DataFrame | None = None
    if run_wf:
        logger.info("Running walk-forward validation (12 × 6-month windows) …")
        t_wf = time.time()
        wf_df = engine.walk_forward_validation(spx_df, vix_wide)
        logger.info("Walk-forward done in %.1fs", time.time() - t_wf)
        n_neg = int((wf_df["below_zero_sharpe"]).sum())
        n_tot = len(wf_df)
        logger.info("  Negative Sharpe windows: %d/%d", n_neg, n_tot)
        print(f"\n  Walk-Forward Summary ({n_neg}/{n_tot} windows with Sharpe < 0):")
        print(f"  {'Win':<5} {'Test Period':<25} {'Sharpe':>8} {'AnnRet':>8} {'Flag'}")
        print("  " + "─" * 60)
        for _, r in wf_df.iterrows():
            sh = r.get("sharpe", float("nan"))
            ar = r.get("ann_return", float("nan"))
            flag = "⚠ NEG" if r.get("below_zero_sharpe") else "OK"
            sh_s = f"{sh:.2f}" if pd.notna(sh) else "—"
            ar_s = f"{ar*100:.1f}%" if pd.notna(ar) else "—"
            period = f"{r['test_start']} → {r['test_end']}"
            print(f"  {int(r['window']):<5} {period:<25} {sh_s:>8} {ar_s:>8}  {flag}")

    # ── 5. Save results to disk ───────────────────────────────────────────────
    logger.info("Saving results to disk …")
    engine.save()
    logger.info("Saved: full_results.parquet + full_results.pkl")

    # ── 6. Generate HTML report ───────────────────────────────────────────────
    logger.info("Generating HTML report …")
    rg   = ReportGenerator(
        engine      = engine,
        equity_df   = equity_df,
        metrics     = metrics,
        crisis_perf = crisis,
        wf_df       = wf_df,
        signals_df  = engine.signals_df,
    )
    report_path = rg.generate(auto_open=True)
    logger.info("Report saved → %s", report_path)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C10: Run full backtest")
    parser.add_argument(
        "--no-wf", dest="no_wf", action="store_true",
        help="Skip walk-forward validation (saves ~5 min)"
    )
    args = parser.parse_args()
    main(run_wf=not args.no_wf)
