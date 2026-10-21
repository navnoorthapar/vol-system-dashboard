#!/usr/bin/env python3
"""
simulate_next_session.py — the daily bot's next session, offline, on a copy of
this checkout, followed by the bot's own test gates.

The incident class this guards against (2026-10-20): tests that pinned values
of the committed data passed on every pull request, then failed inside the bot
on the first legitimately new fit, and the accepted overnight book was never
committed. Here a new session happens before merge:

  1. The tracked files (the working tree's versions) are copied to a scratch
     directory, which is made a git repository (the diagnostics record the
     code commit).
  2. In the copy, scripts/cloud_refresh.py main() runs the book slot
     (market,capture,calibrate,freeze) for the session after the database's
     last SPX session, at 22:30 ET (the Global Trading Hours book), with every
     download faked from the committed data:
       market     Yahoo, Cboe and Treasury return the stored window plus one
                  row per series for the new session (the last value carried
                  forward; the VIX indices are the headline Heston fit's model
                  values, so the joint fit is consistent)
       capture    the newest committed Cboe books, re-dated to the new session
       calibrate  a Yahoo-format book priced by the headline fit's Heston
                  parameters at the new spot; the real calibrator, gate,
                  forward ledger, surface report, diagnostics, Greeks surface
                  and scenario grid follow
  3. The three gates of .github/workflows/daily_refresh.yml (their test lists
     are read from the workflow) run in the copy, and with --full the whole
     suite as the weekly tests.yml run would see main after the bot.

    python scripts/ci/simulate_next_session.py [--out DIR] [--full] [--keep]

Exit 1 when the simulated run does not accept and record the session, or when
any gate fails. Nothing outside --out is written. About 10 minutes (+9 with
--full) on 4 CPUs; the diagnostics take about 4.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES = ("Data tests (gate the capture commit)", "Artifact tests (gate the artifact commit)",
         "Presentation tests (gate the deploy)")


def gate_tests(workflow: Path) -> dict:
    """{gate step name: [test files]} from the daily workflow."""
    text = workflow.read_text()
    out = {}
    for name in GATES:
        m = re.search(rf"- name: {re.escape(name)}\n(.*?)(?=\n\s*- name: |\Z)", text, re.DOTALL)
        if not m:
            raise SystemExit(f"{workflow}: no step {name!r}")
        out[name] = re.findall(r"\btest_\w+\.py\b", m.group(1))
    return out


def copy_checkout(src: Path, dst: Path) -> None:
    """The working tree's tracked files and its new, not ignored ones (so a local
    run sees uncommitted work), made a git repository."""
    files = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                           cwd=src, capture_output=True, check=True).stdout
    for rel in filter(None, files.decode().split("\0")):
        s = src / rel
        if s.is_file():
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    env = {**os.environ, "GIT_AUTHOR_NAME": "sim", "GIT_AUTHOR_EMAIL": "sim@example.com",
           "GIT_COMMITTER_NAME": "sim", "GIT_COMMITTER_EMAIL": "sim@example.com"}
    for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                ["git", "commit", "-qm", "simulation base"]):
        subprocess.run(cmd, cwd=dst, check=True, env=env, capture_output=True)
    # No history, like CI's fetch-depth 1 checkout: mark the base commit a shallow
    # boundary so the tests that need history skip there as they do in CI.
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=dst, check=True, capture_output=True,
                          text=True).stdout.strip()
    (dst / ".git" / "shallow").write_text(head + "\n")


def _patch(module, **functions) -> None:
    """Replace a module's downloaders for this (driver) process only."""
    for name, fn in functions.items():
        if not hasattr(module, name):
            raise AttributeError(f"{module.__name__} has no {name}: the simulation is out of date")
        setattr(module, name, fn)


# ── the driver: runs INSIDE the copy (python <copy>/scripts/ci/simulate_next_session.py --drive) ──

def drive(status_file: Path) -> int:
    root = ROOT
    sys.path.insert(0, str(root))
    import importlib.util
    import pickle
    from datetime import UTC, datetime, timedelta

    import pandas as pd

    from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
    from joint_vol_calibration.data import cboe_downloader as cboe
    from joint_vol_calibration.data import database as dbm
    from joint_vol_calibration.data import option_chain as oc
    from joint_vol_calibration.data import rates
    from joint_vol_calibration.data import yf_downloader as dl
    from joint_vol_calibration.models.heston import heston_vix_term_index
    from joint_vol_calibration.tests.test_joint_calibrator import _heston_chain

    spec = importlib.util.spec_from_file_location("cloud_refresh_sim", root / "scripts" / "cloud_refresh.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load scripts/cloud_refresh.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)

    cr.restore_captures()                       # the committed record, as the bot's first step
    last = dbm.get_latest_date("spx_ohlcv")
    if last is None:
        raise SystemExit("the database holds no SPX session")
    session = oc.next_session(last).isoformat()
    now = datetime.combine(oc.next_session(last), datetime.min.time()).replace(
        hour=22, minute=30, tzinfo=oc.NY_TZ).astimezone(UTC)
    head = cr.headline_calibration(last)
    with open(head["path"], "rb") as f:
        cal = pickle.load(f)  # noqa: S301 (the checkout's own calibration pickle)
    p = cal["params"]
    spx = dbm.get_spx_ohlcv(as_of_date=last)
    S = float(spx["close"].iloc[-1])
    print(f"simulating session {session} (after {last}) at {now:%Y-%m-%dT%H:%MZ}; "
          f"chain priced by the {head['date']} fit at S = {S:.2f}", flush=True)

    def window(df, start, end):
        d = df["date"].astype(str).str[:10]
        return df[(d >= str(start)[:10]) & (d <= str(end)[:10])].copy()

    def spx_dl(start, end):
        df = dbm.get_spx_ohlcv(as_of_date=last, start_date=start)[
            ["date", "open", "high", "low", "close", "volume"]]
        new = df.iloc[[-1]].copy()
        new["date"] = session
        return pd.concat([df, new], ignore_index=True)

    def tbill_dl(start, end):
        df = dbm.get_tbill_rates_series(as_of_date=last, start_date=start)
        new = df.iloc[[-1]].copy()
        new["date"] = session
        return pd.concat([df, new], ignore_index=True)

    ts_cols = ["date", "ticker", "tenor_days", "close", "is_vvix", "source"]
    with dbm._connection() as conn:
        ts_all = pd.read_sql_query("SELECT date, ticker, tenor_days, close, is_vvix, source "
                                   "FROM vix_term_structure", conn)
        vd_all = pd.read_sql_query("SELECT date, open, high, low, close, source FROM vix_daily", conn)
    new_ts = []
    for t, n in {**VIX_INDEX_TENOR_DAYS}.items():
        new_ts.append({"date": session, "ticker": t, "tenor_days": n, "is_vvix": 0,
                       "close": float(heston_vix_term_index(p["kappa"], p["theta"], p["sigma"], p["v0"],
                                                            n / 365.0))})
    vv = ts_all[ts_all["ticker"] == "^VVIX"].iloc[-1]
    new_ts.append({"date": session, "ticker": "^VVIX", "tenor_days": int(vv["tenor_days"]),
                   "is_vvix": 1, "close": float(vv["close"])})
    ts_new = pd.DataFrame(new_ts)
    vix_close = float(ts_new.loc[ts_new["ticker"] == "^VIX", "close"].iloc[0])
    vd_new = pd.DataFrame([{"date": session, "open": vix_close, "high": vix_close, "low": vix_close,
                            "close": vix_close}])

    def ts_frame(start, end, source):
        return pd.concat([window(ts_all, start, end), ts_new], ignore_index=True)[ts_cols[:-1]] \
            .assign(source=source)

    def vd_frame(start, end, source):
        return pd.concat([window(vd_all, start, end), vd_new], ignore_index=True)[
            ["date", "open", "high", "low", "close"]].assign(source=source)

    _patch(dl, download_spx_ohlcv=spx_dl, download_tbill_rate=tbill_dl,
           download_vix_term_structure=lambda start, end: ts_frame(start, end, "yfinance"),
           download_vix_index=lambda start, end: vd_frame(start, end, "yfinance"))
    _patch(cboe, download_vix_term_structure_history=(
        lambda start=None, end=None, tickers=None: ts_frame(start or "2000-01-01", end or session, "cboe")),
        download_vix_history=lambda: vd_frame("2000-01-01", session, "cboe"))

    quotes = rates.load_rate_quotes()

    def treasury(year, kind):
        q = quotes[(quotes["source"] == "treasury") & quotes["date"].astype(str).str.startswith(str(year))]
        bills = q["series"].str.startswith("UST_BILL")
        q = q[bills] if kind == "daily_treasury_bill_rates" else q[~bills]
        if q.empty:
            return q
        new = q[q["date"] == q["date"].max()].assign(date=session)
        return pd.concat([q, new], ignore_index=True)
    _patch(rates, download_treasury_year=treasury,
           download_nyfed_sofr=lambda start=None, end=None: quotes[quotes["source"] == "nyfed"].copy())

    cap_dir = root / "data_store" / "options" / "cboe"
    replay = {}
    for und in ("SPX", "VIX"):
        src = sorted(cap_dir.glob(f"{und.lower()}_*.parquet"))
        if src:
            replay[und] = (pd.read_parquet(src[-1]), json.loads(src[-1].with_suffix(".json").read_text()))
    for old in cap_dir.glob(f"*_{session}_*"):   # the copy only: the session's capture must happen
        old.unlink()

    def capture(und, now_=None, **kw):
        if und not in replay:
            raise OSError(f"no committed {und} capture to replay")
        frame, meta = replay[und]
        frame = frame.assign(snapshot_date=session)
        qt = (now - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
        frame["quote_time"] = qt
        stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        return cboe.ChainCapture(
            underlying=und, frame=frame, session=session, quote_time=qt, book_state="gth",
            fetched_at=stamp, payload_timestamp=stamp,
            underlying_last_trade_time=None, underlying_price=float(meta["underlying_price"]),
            url=meta["url"], sha256=meta["sha256"], n_bytes=int(meta["n_bytes"]))
    _patch(cboe, capture_option_chain=capture)

    r = float(cal.get("r_reference") or 0.04)
    q = float(cal.get("q") or 0.013)
    start = oc.next_session(last)

    class Ticker:
        def __init__(self, symbol):
            self.options = tuple((start + timedelta(days=i)).isoformat() for i in range(1, 420)
                                 if (start + timedelta(days=i)).weekday() < 5)

        def option_chain(self, exp):
            df = _heston_chain(params=p, session=session, S=S, r=r, q=q, expiries=[exp], spread=0.2)
            calls = df[df.option_type == "call"].drop(columns=["expiration", "option_type"])
            puts = df[df.option_type == "put"].drop(columns=["expiration", "option_type"])
            return type("Chain", (), {"calls": calls, "puts": puts})()

    t0 = time.perf_counter()
    # The day as production runs it: the close slot (22:07 UTC) stores the
    # session's market data and label, then the book slot (02:30 UTC).
    close = datetime.combine(oc.next_session(last), datetime.min.time()).replace(
        hour=18, minute=7, tzinfo=oc.NY_TZ).astimezone(UTC)
    code_close = cr.main(["--steps", "market,labels", "--status-file", str(status_file.with_name(
        "status_close.json"))], now_utc=close)
    print(f"close slot exit {code_close}", flush=True)
    code = cr.main(["--steps", "market,capture,calibrate,freeze", "--status-file", str(status_file)],
                   now_utc=now, ticker_factory=Ticker)
    code = max(code, code_close)
    st = json.loads(status_file.read_text())
    st["simulated"] = {"session": session, "after": last, "now_utc": now.isoformat(),
                       "headline_before": head["date"], "seconds": round(time.perf_counter() - t0, 1)}
    status_file.write_text(json.dumps(st, indent=2, default=str))
    print(f"cloud_refresh exit {code} in {st['simulated']['seconds']} s", flush=True)
    return code


# ── the outer run ─────────────────────────────────────────────────────────────

def _pytest(cwd: Path, files: list, log: Path) -> tuple:
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
                           cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    tail = log.read_text().strip().splitlines()[-1:] or [""]
    return r.returncode, round(time.perf_counter() - t0, 1), tail[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="scratch directory (default: a temp dir)")
    ap.add_argument("--full", action="store_true", help="also run the whole suite in the copy")
    ap.add_argument("--keep", action="store_true", help="keep the copy")
    ap.add_argument("--drive", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.drive:
        return drive(args.drive)

    out = Path(args.out or tempfile.mkdtemp(prefix="sim-next-session-")).resolve()
    tree = out / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"copying the checkout to {tree} ...", flush=True)
    copy_checkout(ROOT, tree)
    env = {**os.environ, "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg"}
    for k in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_ACTIONS"):
        env.pop(k, None)
    status_file = out / "status.json"
    with open(out / "logs" / "bot.log", "w") as fh:
        r = subprocess.run([sys.executable, str(tree / "scripts" / "ci" / "simulate_next_session.py"),
                            "--drive", str(status_file)], cwd=tree, env=env, stdout=fh,
                           stderr=subprocess.STDOUT, check=False)
    problems = []
    st = json.loads(status_file.read_text()) if status_file.exists() else {}
    sim = st.get("simulated", {})
    cal = st.get("calibration") or {}
    if r.returncode != 0:
        problems.append(f"cloud_refresh exited {r.returncode}: {st.get('errors')}")
    if cal.get("status") != "accepted":
        problems.append(f"the simulated fit was not accepted: {cal.get('status')} {cal.get('reason')}")
    if not (cal.get("forward_ledger") or {}).get("ok"):
        problems.append(f"forward ledger: {cal.get('forward_ledger')}")
    if (st.get("capture") or {}).get("status") != "captured":
        problems.append(f"capture: {st.get('capture')}")
    for name, res in (st.get("artifacts") or {}).items():
        if not res.get("ok"):
            problems.append(f"artifact {name}: {res.get('error')}")
    if not st.get("publishable"):
        problems.append(f"not publishable: {st.get('publish_blockers')}")
    ok_arts = sorted(k for k, v in (st.get("artifacts") or {}).items() if v.get("ok"))
    rows = [("bot run", r.returncode, sim.get("seconds"),
             f"session {sim.get('session')}: calibration {cal.get('status')}, artifacts {ok_arts}")]
    gates = gate_tests(tree / ".github" / "workflows" / "daily_refresh.yml")
    if "--full" in (argv or sys.argv[1:]) or args.full:
        gates["Full suite"] = ["joint_vol_calibration/tests"]
    for name, files in gates.items():
        cwd = tree if name == "Full suite" else tree / "joint_vol_calibration" / "tests"
        log = out / "logs" / (re.sub(r"\W+", "_", name).strip("_").lower() + ".log")
        code, secs, tail = _pytest(cwd, files, log)
        rows.append((name, code, secs, tail))
        if code != 0:
            problems.append(f"{name} failed ({tail}); log {log}")
    report = ["| step | exit | seconds | result |", "|---|---|---|---|"]
    report += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    report += ["", *(f"- PROBLEM: {x}" for x in problems)] if problems else ["", "All gates pass."]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)
    if not args.keep and not problems:
        shutil.rmtree(tree, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
