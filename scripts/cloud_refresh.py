#!/usr/bin/env python3
"""
Daily auto-refresh for GitHub Actions.

Steps (run all, or a subset with --steps):
  market     Download the last 400 days of SPX/VIX/T-bill from yfinance -> SQLite DB
  labels     Extend regime_labels.parquet with new trading days (rule labels)
  calibrate  Download the SPX option chain for the last CLOSED NYSE session,
             clean it (data/option_chain.py), fit the joint Heston model and,
             if the quality gate accepts it, save joint_cal_<session>.pkl and
             only then store the chain as that session's options snapshot
  freeze     Freeze the Flask app to static HTML -> .site-build/

Persistence (unless --no-persist): the SQLite DB is a rebuildable cache and is
not what the bot commits. Before any download the run upserts the committed
captures into it (yearly market partitions under data_store/market/ and the
per-session SPX chains under data_store/options/, via the data layer's
import_* helpers), and after the steps it exports what it captured back to
those files (export_* helpers merge and never drop a stored row). The
workflow commits those two directories, so a session Yahoo later drops, or a
chain that is only quoted once, survives every later run and every clean
clone.

The option step needs a book with live two-sided quotes. After the 16:15 ET
close Cboe's book empties until Global Trading Hours open (20:15 ET; Friday
evenings keep the closing book), so the workflow should run `market,labels`
after the close and `calibrate` when the book is live; a mistimed run fails
closed at the quote-state gate instead of fitting a stale/empty book.

Exit status (main() returns it):
  0  success, or a calibration REJECTED/SKIPPED while the newest accepted
     calibration is at most --max-calibration-age sessions old
  1  hard failure: empty SPX download, the expected NYSE session missing from
     the DB, a step that crashed, freeze failure, or a rejected/skipped
     calibration while the newest accepted one is too old

Every run writes a machine-readable status JSON (--status-file, default
.refresh-status.json at the repo root) with the session, the steps run, the
calibration outcome and reason, and the headline calibration's date and age;
a Markdown summary goes to $GITHUB_STEP_SUMMARY and key=value pairs to
$GITHUB_OUTPUT when those are set.

Importing this module has no side effects (no chdir, no logging setup, no
frozen date): everything happens in main().
"""
import argparse
import json
import logging
import os
import pathlib
import pickle
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data_store"
DEFAULT_STATUS_FILE = ROOT / ".refresh-status.json"

log = logging.getLogger("cloud_refresh")

# Calendar days of history fetched before relabelling. The R2 gate is a rolling
# 252-TRADING-day quantile, so this must comfortably exceed 252 business days.
LABEL_LOOKBACK_DAYS = 900
_ADAPTIVE_WINDOW = 252
DEFAULT_MAX_CAL_AGE_SESSIONS = 3
ALL_STEPS = ("market", "labels", "calibrate", "freeze")


def _bootstrap_path() -> None:
    """Make the repo importable when run as a script (called from main())."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def latest_session_date(spx_df, now_utc: "datetime | None" = None) -> "str | None":
    """Date (YYYY-MM-DD) of the most recent CLOSED SPX session in ``spx_df``.

    This is what a calibration describes. yfinance's daily history includes
    the in-progress bar during the session, so a last row dated today (New
    York) is dropped until 16:15 ET (a workflow_dispatch at 14:00 ET must not
    label an intraday bar as the close). With ``now_utc`` None the wall clock
    is used.
    """
    if spx_df is None or len(spx_df) == 0:
        return None
    from joint_vol_calibration.data import option_chain as oc
    dates = [str(d)[:10] for d in spx_df["date"]]
    closed = oc.last_closed_session(now_utc).isoformat()
    dates = [d for d in dates if d <= closed]
    return dates[-1] if dates else None


def calibration_path(session_date: str) -> pathlib.Path:
    """Canonical location of the joint calibration for one closed session."""
    return DATA / "calibrations" / f"joint_cal_{session_date}.pkl"


def _annotate(level: str, title: str, message: str) -> None:
    """GitHub Actions workflow command (shows as an annotation on the run)."""
    msg = str(message).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level} title={title}::{msg}", flush=True)


class _Group:
    """::group:: / ::endgroup:: around one step's log lines."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::group::{self.name}", flush=True)
        return self

    def __exit__(self, *exc):
        if os.environ.get("GITHUB_ACTIONS"):
            print("::endgroup::", flush=True)
        return False


# -- Step 1: Market data -------------------------------------------------------

def download_start(today: date, last_stored: "str | None") -> date:
    """First date to download: 400 days back, or earlier when the newest stored
    session is older than that, so no gap can open between what is stored
    (DB + persisted partitions) and the download window."""
    start = today - timedelta(days=400)
    if last_stored:
        start = min(start, date.fromisoformat(last_stored[:10]) - timedelta(days=10))
    return start


def refresh_market_data(today: "date | None" = None) -> dict:
    """Download the market series into the SQLite DB, from download_start().

    Returns {"ok", "hard_failure", "reason", "rows"}; an empty or failed SPX
    download is a hard failure (the run must not publish the committed DB's
    stale close as current).
    """
    from joint_vol_calibration.data import yf_downloader as dl
    from joint_vol_calibration.data.database import (
        get_latest_date, insert_spx_ohlcv, insert_vix_daily,
        insert_vix_term_structure, insert_tbill_rates,
    )

    today = today or date.today()
    start = download_start(today, get_latest_date("spx_ohlcv")).isoformat()
    end = today.isoformat()
    log.info("[market] Downloading market data from %s -> %s", start, end)
    out = {"ok": True, "hard_failure": False, "reason": "ok", "rows": {}, "start": start}

    try:
        spx = dl.download_spx_ohlcv(start=start, end=end)
        if spx is not None and not spx.empty:
            out["rows"]["spx_ohlcv"] = insert_spx_ohlcv(spx)
            log.info("  SPX OHLCV: %d rows", out["rows"]["spx_ohlcv"])
        else:
            out.update(ok=False, hard_failure=True, reason="SPX OHLCV download returned no rows")
            log.error("  SPX OHLCV: empty download")
    except Exception as e:
        out.update(ok=False, hard_failure=True, reason=f"SPX OHLCV download failed: {e}")
        log.error("  SPX OHLCV failed: %s", e)

    for key, fn, ins in (("vix_daily", dl.download_vix_index, insert_vix_daily),
                         ("vix_term_structure", dl.download_vix_term_structure,
                          insert_vix_term_structure),
                         ("tbill_rates", dl.download_tbill_rate, insert_tbill_rates)):
        try:
            df = fn(start=start, end=end)
            if df is not None and not df.empty:
                out["rows"][key] = ins(df)
                log.info("  %s: %d rows", key, out["rows"][key])
            else:
                log.warning("  %s: empty download", key)
        except Exception as e:
            log.warning("  %s failed: %s", key, e)
    return out


def check_expected_session(now_utc: datetime) -> dict:
    """The DB must contain the last closed NYSE session (by the calendar)."""
    from joint_vol_calibration.data import option_chain as oc
    from joint_vol_calibration.data.database import get_spx_ohlcv

    expected = oc.last_closed_session(now_utc).isoformat()
    session = latest_session_date(get_spx_ohlcv(as_of_date=expected), now_utc)
    if session != expected:
        return {"ok": False, "expected": expected, "session": session,
                "reason": (f"expected NYSE session {expected} is missing from the DB "
                           f"(latest SPX row: {session})")}
    return {"ok": True, "expected": expected, "session": session, "reason": "ok"}


# -- Step 2: Regime labels -----------------------------------------------------

def extend_regime_labels(today: "date | None" = None) -> int:
    """Extend regime_labels.parquet with DETERMINISTIC RULE labels, and repair
    any stored label that disagrees with the rule. Returns rows added.

    This used to append ``clf.predict(...)`` — the XGBoost classifier's output.
    That is the model C16/C17 demoted: it scores 63.4% out-of-sample and loses
    to a 90.0% "predict-yesterday" baseline, so it is research-only and is not
    supposed to drive anything. But dashboard/app.py headlines the LAST ROW of
    this parquet as "the deterministic rule label, the label the backtest
    trades on", so every day the bot appended a prediction, the front page
    published the demoted model under a caption saying it was the rule.

    Two details matter for reproducing the historical labels exactly:
      * the R2 gate is ADAPTIVE (rolling 252-day 80th percentile of VVIX),
        i.e. ``vvix_threshold=None``. The fixed 100.0 default disagrees with
        the shipped history on 19.7% of days; adaptive reproduces all 4,125
        of them exactly.
      * that rolling gate therefore needs >= 252 trading days of VVIX history
        before it is meaningful, so the fetch window is ~900 calendar days
        rather than the 60 days the feature builder alone would need.
    """
    import pandas as pd
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import (
        build_features, build_regime_labels, FEATURE_COLS,
    )

    today_s = (today or date.today()).isoformat()
    rl_path = DATA / "signals" / "regime_labels.parquet"
    rl      = pd.read_parquet(rl_path)
    last_dt = rl.index.max()
    log.info("[labels] Regime labels last date: %s -- extending to %s", last_dt.date(), today_s)

    # Long enough that the trailing dates get a FULL 252-day adaptive window.
    lookback_start = str((last_dt - timedelta(days=LABEL_LOOKBACK_DAYS)).date())
    spx = get_spx_ohlcv(as_of_date=today_s, start_date=lookback_start)
    vix = get_vix_term_structure_wide(as_of_date=today_s, start_date=lookback_start)

    if spx.empty or vix.empty:
        log.warning("  Insufficient market data for feature computation; skipping")
        return 0

    feats = build_features(spx, vix)

    # Forward-fill slow-moving features across data gaps (VIX TS weekends, etc.)
    FFILL = ["fear_premium", "rv_change_5d", "ts_slope", "vvix", "pdv_iv_spread"]
    for col in FFILL:
        if col in feats.columns:
            feats[col] = feats[col].ffill()
    feats.dropna(inplace=True)

    labels = build_regime_labels(spx, vix, vvix_threshold=None)
    if labels.empty:
        log.warning("  Rule labels could not be computed; skipping")
        return 0

    # Only trust dates whose adaptive gate saw a full 252-day window inside
    # this fetch; earlier ones would be computed from a truncated history.
    if len(labels) <= _ADAPTIVE_WINDOW:
        log.warning("  Only %d labelled days fetched (< %d needed for the adaptive "
                    "gate); skipping to avoid rewriting history from a short window",
                    len(labels), _ADAPTIVE_WINDOW)
        return 0
    safe_from = labels.index[_ADAPTIVE_WINDOW]

    new_idx = feats.index[(feats.index > last_dt) & feats.index.isin(labels.index)]
    n = 0
    if len(new_idx):
        new_rows = feats.loc[new_idx, FEATURE_COLS].copy()
        new_rows["regime"] = labels.reindex(new_idx).astype(int)
        rl = pd.concat([rl, new_rows])
        rl = rl[~rl.index.duplicated(keep="last")].sort_index()
        n = len(new_rows)
    else:
        log.info("  No new dates to label; regime labels already current")

    # Self-heal: overwrite any stored label in the trustworthy window that the
    # rule disagrees with. This is what repairs the rows the classifier wrote.
    repair_idx = rl.index[(rl.index >= safe_from) & rl.index.isin(labels.index)]
    if len(repair_idx):
        stored = rl.loc[repair_idx, "regime"].astype(int)
        truth  = labels.reindex(repair_idx).astype(int)
        differ = stored != truth
        if differ.any():
            rl.loc[repair_idx[differ], "regime"] = truth[differ]
            log.info("  Repaired %d stored label(s) that disagreed with the rule "
                     "(%s → %s)", int(differ.sum()),
                     repair_idx[differ][0].date(), repair_idx[differ][-1].date())

    rl.to_parquet(rl_path)
    log.info("  Added %d new rows -> regime_labels now at %s", n, rl.index.max().date())
    return n


# -- Step 3: Heston recalibration ----------------------------------------------

def _persist_snapshot(clean, session: str) -> int:
    """Store an ACCEPTED, cleaned chain as the session's SPX snapshot.

    Called only after the quality gate accepted the fit, so a rejected or
    empty-book chain never becomes "the most recent snapshot <= as_of" for a
    later caller. insert_options_snapshot uses a plain INSERT, so any partial
    write for this session is cleared first (re-runs must not duplicate).
    """
    import sqlite3
    from joint_vol_calibration.data import database as dbm

    opts = clean.quotes.copy()
    opts["snapshot_date"] = session
    con = sqlite3.connect(str(dbm.DB_PATH))
    try:
        con.execute("DELETE FROM options_snapshots WHERE underlying='SPX' AND snapshot_date=?",
                    (session,))
        con.commit()
    finally:
        con.close()
    return dbm.insert_options_snapshot(opts, "SPX")


def _save_rejected_chain(raw, session: str, reason: str) -> "str | None":
    """Keep a rejected raw chain for post-mortems, OUTSIDE options_snapshots
    (data_store/parquet/ is gitignored and nothing reads diag_* files)."""
    try:
        d = DATA / "parquet"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"diag_options_spx_{session}_rejected.parquet"
        df = raw.copy()
        df["rejection_reason"] = reason
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        df.to_parquet(path, index=False)
        return str(path)
    except Exception as e:
        log.warning("  could not save the rejected chain for diagnostics: %s", e)
        return None


def recalibrate_heston(now_utc: "datetime | None" = None, ticker_factory=None,
                       calibrate_kwargs: "dict | None" = None) -> dict:
    """Download the SPX chain for the last closed session and fit the joint model.

    Returns a status dict: {"status": accepted|rejected|skipped|exists|failed,
    "reason", "session", "path", "summary" (fit numbers), "chain" (quote-state
    diagnostics)}. Nothing is written to options_snapshots unless accepted.
    """
    import pandas as pd
    from joint_vol_calibration.calibration.joint_calibrator import (
        JointCalibrator, is_acceptable_calibration,
    )
    from joint_vol_calibration.data import option_chain as oc
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_tbill_rate

    now_utc = now_utc or _utcnow()
    expected = oc.last_closed_session(now_utc).isoformat()
    spot_date = latest_session_date(get_spx_ohlcv(as_of_date=expected), now_utc)
    st = {"status": "skipped", "reason": "", "session": spot_date, "path": None,
          "summary": {}, "chain": {}}
    if spot_date is None:
        st["reason"] = "no SPX sessions in DB"
        return st
    if spot_date != expected:
        st["reason"] = (f"DB's latest session {spot_date} is not the last closed NYSE "
                        f"session {expected}; refusing to label today's chain as {spot_date}")
        return st
    log.info("[calibrate] SPX options for session %s (run %s UTC)", spot_date,
             now_utc.strftime("%Y-%m-%d %H:%M"))

    out = calibration_path(spot_date)
    st["path"] = str(out)
    if out.exists():
        st.update(status="exists", reason=f"calibration for session {spot_date} already exists")
        log.info("  %s; skipping", st["reason"])
        return st

    spx = get_spx_ohlcv(as_of_date=spot_date)
    S = float(spx["close"].iloc[-1])
    r_ref = oc.tbill_discount_to_cc(get_tbill_rate(spot_date))

    try:
        raw, meta = oc.fetch_spx_chain(spot_date, ticker_factory=ticker_factory)
    except Exception as e:
        st.update(status="failed", reason=f"option chain download failed: {e}")
        return st
    st["chain"] = {"symbol": meta.get("symbol"), "n_listed": len(meta.get("listed", [])),
                   "requested": meta.get("requested", []), "failed": meta.get("failed", [])}
    if raw.empty:
        st["reason"] = "no SPX option chain returned"
        return st

    clean = oc.clean_option_chain(raw, spot_date, S, r_ref=r_ref)
    d = clean.diagnostics
    st["chain"].update({k: d.get(k) for k in ("n_raw", "n_two_sided", "n_near_atm",
                                              "near_atm_two_sided_share", "n_clean",
                                              "n_expiries_with_carry")})
    log.info("  Chain hygiene: %s raw -> %s two-sided -> %s clean rows; near-ATM two-sided "
             "share %.0f%%", d.get("n_raw"), d.get("n_two_sided"), d.get("n_clean", 0),
             100 * (d.get("near_atm_two_sided_share") or 0))
    if not clean.ok:
        st.update(status="skipped", reason=clean.reason)
        st["diagnostic_chain"] = _save_rejected_chain(raw, spot_date, clean.reason)
        return st

    try:
        # r=None: the calibrator reads the same ^IRX row and records its source
        cal = JointCalibrator(as_of_date=spot_date, spx_chain=clean)
        result = cal.calibrate(**(calibrate_kwargs or {}))
    except Exception as e:
        st.update(status="failed", reason=f"calibration failed: {e}")
        log.error("  %s", st["reason"])
        return st

    result["as_of_date"] = spot_date              # the session the fit describes
    result["run_utc"] = now_utc.isoformat(timespec="seconds")
    p, ll = result["params"], result["leg_losses"]
    st["summary"] = {
        "params": p, "spx_iv_rmse": ll.get("spx_iv_rmse"),
        "spx_oos_iv_rmse": ll.get("spx_oos_iv_rmse"),
        "vix_index_rmse": ll.get("vix_index_rmse"),
        "n_spx_expiries": result.get("n_spx_expiries"),
        "n_spx_options": result.get("n_spx_options"),
        "n_vix_tenors": result.get("n_vix_tenors"),
        "feller_state": result.get("feller_state"), "r": result.get("r"), "q": result.get("q"),
    }

    ok, reason = is_acceptable_calibration(result)
    if not ok:
        st.update(status="rejected", reason=reason)
        st["diagnostic_chain"] = _save_rejected_chain(raw, spot_date, reason)
        log.warning("  Calibration REJECTED (%s): kappa=%.4f sigma=%.4f rho=%.4f "
                    "SPX RMSE=%.3f vp -- keeping previous good calibration",
                    reason, p["kappa"], p["sigma"], p["rho"], ll.get("spx_iv_rmse", -1))
        return st

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(result, f)
    try:
        n_opt = _persist_snapshot(clean, spot_date)
        log.info("  Saved %d cleaned option quotes as the %s SPX snapshot", n_opt, spot_date)
    except Exception as e:     # the fit stands; the snapshot is a convenience
        log.warning("  could not store the accepted chain: %s", e)
    st.update(status="accepted", reason="ok")
    log.info("  Done (accepted): kappa=%.4f sigma=%.4f rho=%.4f | SPX RMSE %.3f vp "
             "(OOS %.3f) | VIX index RMSE %.3f | %d opts / %d expiries, %d VIX tenors | "
             "Feller %s | session %s", p["kappa"], p["sigma"], p["rho"],
             ll["spx_iv_rmse"], ll["spx_oos_iv_rmse"], ll["vix_index_rmse"],
             result["n_spx_options"], result["n_spx_expiries"], result["n_vix_tenors"],
             result["feller_state"], spot_date)
    return st


# -- Persistence of captures -----------------------------------------------------

def restore_captures() -> dict:
    """Upsert the committed captures (market partitions, SPX/VIX chains) into the DB."""
    from joint_vol_calibration.data import database as dbm
    out = {"market": dbm.import_market_partitions(), "options_rows": dbm.import_options_snapshots()}
    log.info("[persist] Restored committed captures into the DB: market %s, option rows %d",
             out["market"], out["options_rows"])
    return out


def persist_captures(market_since: "str | None", chain_sessions=()) -> list:
    """Export what this run captured to the committed parquet files.

    Market partitions are merged for every year >= ``market_since`` (None:
    skip); each session in ``chain_sessions`` gets its stored SPX chain
    written to data_store/options/spx_<session>.parquet. Returns the paths
    that changed.
    """
    from joint_vol_calibration.data import database as dbm
    written = []
    if market_since:
        written += dbm.export_market_partitions(since=market_since)
    for s in chain_sessions:
        p = dbm.export_options_snapshot(s, "SPX")
        if p is not None:
            written.append(p)
    written = [str(p) for p in written]
    log.info("[persist] %d capture file(s) changed%s", len(written),
             "".join(f"\n  {p}" for p in written))
    return written


def headline_calibration(session: "str | None") -> dict:
    """Newest calibration that passes the CURRENT gate, and its age in NYSE
    sessions relative to ``session``. Legacy (pre-C22) pickles do not count."""
    from joint_vol_calibration.calibration.joint_calibrator import is_acceptable_calibration
    from joint_vol_calibration.data import option_chain as oc

    best = None
    for path in sorted((DATA / "calibrations").glob("joint_cal_*.pkl")):
        try:
            with open(path, "rb") as f:
                cal = pickle.load(f)
        except Exception:
            continue
        if not isinstance(cal, dict) or not is_acceptable_calibration(cal)[0]:
            continue
        d = str(cal.get("spot_date") or cal.get("as_of_date") or path.stem[-10:])[:10]
        if best is None or d > best["date"]:
            best = {"date": d, "path": str(path)}
    if best is None:
        return {"date": None, "path": None, "age_sessions": None}
    best["age_sessions"] = (oc.sessions_between(best["date"], session)
                            if session else None)
    return best


# -- Step 4: Freeze static site ------------------------------------------------

def freeze_site() -> dict:
    log.info("[freeze] Freezing static site -> .site-build/")
    result = subprocess.run(
        [sys.executable, "dashboard/freeze_site.py"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    for line in result.stdout.splitlines():
        log.info("  %s", line)
    for line in result.stderr.splitlines():          # warnings matter on success too
        log.warning("  [freeze stderr] %s", line)
    if result.returncode != 0:
        return {"ok": False, "reason": f"freeze_site.py exited {result.returncode}"}
    return {"ok": True, "reason": "ok"}


# -- Reporting -----------------------------------------------------------------

def _write_reports(status: dict, status_file: "pathlib.Path | None") -> None:
    if status_file is not None:
        status_file = pathlib.Path(status_file)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(json.dumps(status, indent=2, default=str))
        log.info("Status written to %s", status_file)

    cal = status.get("calibration") or {}
    head = status.get("headline_calibration") or {}
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a") as f:
            f.write(f"session={status.get('session') or ''}\n")
            f.write(f"calibration={cal.get('status', 'not-run')}\n")
            f.write(f"reason={str(cal.get('reason', '')).splitlines()[0] if cal.get('reason') else ''}\n")
            f.write(f"labels_added={status.get('labels_added', 0)}\n")
            f.write(f"persisted={len(status.get('persisted') or [])}\n")
            f.write(f"headline_calibration={head.get('date') or ''}\n")
            f.write(f"exit_code={status.get('exit_code')}\n")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        s = cal.get("summary") or {}
        ch = cal.get("chain") or {}
        lines = [
            f"## Daily refresh — session {status.get('session')}",
            "",
            "| item | value |", "|---|---|",
            f"| steps | {', '.join(status.get('steps', []))} |",
            f"| exit code | {status.get('exit_code')} |",
            f"| market rows | {status.get('market', {}).get('rows', {})} |",
            f"| labels added | {status.get('labels_added', '—')} |",
            f"| capture files changed | {len(status.get('persisted') or [])} |",
            f"| calibration | **{cal.get('status', 'not run')}** {cal.get('reason', '')} |",
            f"| chain | {ch.get('n_raw', '—')} raw, {ch.get('n_two_sided', '—')} two-sided, "
            f"near-ATM two-sided {ch.get('near_atm_two_sided_share', '—')} |",
            f"| fit | {s.get('n_spx_options', '—')} opts / {s.get('n_spx_expiries', '—')} expiries; "
            f"SPX {s.get('spx_iv_rmse', '—')} vp (OOS {s.get('spx_oos_iv_rmse', '—')}); "
            f"VIX index {s.get('vix_index_rmse', '—')} |",
            f"| headline calibration | {head.get('date')} ({head.get('age_sessions')} sessions old) |",
        ]
        for e in status.get("errors", []):
            lines.append(f"| error | {e} |")
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")


# -- Entry point ---------------------------------------------------------------

def _parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--steps", default=",".join(ALL_STEPS),
                    help="comma-separated subset of: " + ",".join(ALL_STEPS))
    ap.add_argument("--status-file", default=str(DEFAULT_STATUS_FILE),
                    help="machine-readable status JSON ('' to disable)")
    ap.add_argument("--max-calibration-age", type=int, default=DEFAULT_MAX_CAL_AGE_SESSIONS,
                    help="sessions a rejected/skipped run tolerates before failing")
    ap.add_argument("--no-persist", action="store_true",
                    help="do not restore/export the committed parquet captures")
    ap.add_argument("--now", default=None,
                    help="override the clock (ISO UTC, e.g. 2026-10-17T01:30:00Z)")
    args = ap.parse_args(argv)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in ALL_STEPS]
    if bad:
        ap.error(f"unknown step(s): {bad}")
    return args, steps


def main(argv=None, now_utc: "datetime | None" = None, ticker_factory=None,
         calibrate_kwargs: "dict | None" = None) -> int:
    """Run the selected steps; return the process exit status."""
    args, steps = _parse_args(argv if argv is not None else sys.argv[1:])
    _bootstrap_path()
    os.chdir(ROOT)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format="%(asctime)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S")
    if args.now:
        now_utc = datetime.fromisoformat(args.now.replace("Z", "+00:00"))
    now_utc = now_utc or _utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    from joint_vol_calibration.data import option_chain as oc
    today = oc.ny_date(now_utc)
    status = {"run_utc": now_utc.isoformat(timespec="seconds"), "steps": steps,
              "expected_session": oc.last_closed_session(now_utc).isoformat(),
              "session": None, "errors": [], "warnings": []}
    hard_fail = False
    log.info("=" * 60)
    log.info("Daily auto-refresh -- %s UTC -- steps: %s", status["run_utc"], ",".join(steps))
    log.info("=" * 60)

    persist = not args.no_persist and any(s in steps for s in ("market", "labels", "calibrate"))
    if persist:
        with _Group("restore committed captures"):
            try:
                status["restored"] = restore_captures()
            except Exception as e:
                hard_fail = True
                msg = f"could not restore the committed captures: {e}"
                status["errors"].append(msg)
                _annotate("error", "Persistence", msg)
                log.exception(msg)

    market_ok = True
    if "market" in steps:
        with _Group("market data"):
            m = refresh_market_data(today)
            status["market"] = m
            if m["hard_failure"]:
                hard_fail, market_ok = True, False
                status["errors"].append(m["reason"])
                _annotate("error", "Market data", m["reason"])

    sess = None
    if market_ok:
        try:
            chk = check_expected_session(now_utc)
        except Exception as e:
            chk = {"ok": False, "session": None, "reason": f"session check failed: {e}"}
        sess = chk.get("session")
        status["session"] = sess
        if not chk["ok"] and any(s in steps for s in ("market", "calibrate")):
            hard_fail, market_ok = True, False
            status["errors"].append(chk["reason"])
            _annotate("error", "Missing session", chk["reason"])

    if "labels" in steps:
        with _Group("regime labels"):
            if market_ok:
                try:
                    status["labels_added"] = extend_regime_labels(today)
                except Exception as e:
                    hard_fail = True
                    status["errors"].append(f"labels step failed: {e}")
                    log.exception("labels step failed")
            else:
                status["labels_added"] = 0
                log.warning("  skipped: market data is not current")

    if "calibrate" in steps:
        with _Group("option chain + joint calibration"):
            if market_ok:
                try:
                    cal = recalibrate_heston(now_utc, ticker_factory, calibrate_kwargs)
                except Exception as e:
                    cal = {"status": "failed", "reason": f"calibrate step crashed: {e}"}
                    log.exception("calibrate step crashed")
            else:
                cal = {"status": "skipped", "reason": "market data is not current"}
            status["calibration"] = cal
            head = headline_calibration(sess or status["expected_session"])
            status["headline_calibration"] = head
            if cal["status"] in ("rejected", "skipped", "failed"):
                _annotate("warning", f"Calibration {cal['status']}", cal["reason"])
                status["warnings"].append(f"calibration {cal['status']}: {cal['reason']}")
                age = head.get("age_sessions")
                if head.get("date") is None or age is None or age > args.max_calibration_age:
                    hard_fail = True
                    msg = (f"newest accepted calibration is {head.get('date')} "
                           f"({age} sessions old > {args.max_calibration_age})"
                           if head.get("date") else "no calibration passes the current gate")
                    status["errors"].append(msg)
                    _annotate("error", "Calibration stale", msg)

    if persist:
        with _Group("persist captures"):
            since = status.get("market", {}).get("start") if "market" in steps else None
            chains = ([status["calibration"]["session"]]
                      if (status.get("calibration") or {}).get("status") == "accepted" else [])
            try:
                status["persisted"] = persist_captures(since, chains)
            except Exception as e:
                hard_fail = True
                msg = f"could not export the captures: {e}"
                status["errors"].append(msg)
                _annotate("error", "Persistence", msg)
                log.exception(msg)

    if "freeze" in steps:
        with _Group("freeze site"):
            fz = freeze_site()
            status["freeze"] = fz
            if not fz["ok"]:
                hard_fail = True
                status["errors"].append(fz["reason"])
                _annotate("error", "Freeze", fz["reason"])

    status["exit_code"] = 1 if hard_fail else 0
    _write_reports(status, pathlib.Path(args.status_file) if args.status_file else None)
    log.info("=" * 60)
    log.info("Refresh %s (exit %d)", "FAILED" if hard_fail else "complete", status["exit_code"])
    log.info("=" * 60)
    return status["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
