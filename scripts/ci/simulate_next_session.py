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
                  forward; the VIX indices are the model values of the
                  session's parameters below, so the joint fit is consistent);
                  Cboe's VX settlement file of each session the futures table
                  lacks, up to the new one, prices the nine monthly contracts
                  at the model's E^Q[VIX_T] for those parameters (vx_file)
       capture    the newest committed Cboe books, re-dated to the new session
       calibrate  a Yahoo-format book priced at the new spot by Heston with
                  the session's parameters: the headline fit's moved by a
                  fixed step inside the box and off its bounds
                  (simulated_params: kappa x0.6, theta x1.2, sigma x0.7 below
                  1.9, rho +0.1, v0 x1.3), each mid moved by ~1 vol point
                  of seeded smile error that keeps put-call parity
                  (noisy_mid), so the fit is a new one with a
                  production-like RMSE; the real calibrator, gate, forward
                  ledger, surface report, diagnostics, Greeks surface and
                  scenario grid follow. A book priced by the headline itself
                  refitted it to 1e-3 (RMSE 0.06 vp), so a test that pinned
                  the headline's parameters, or values derived from them
                  only (Greeks nodes, scenario P&L, the model VIX term),
                  passed here and failed in the bot. The run fails unless
                  every parameter of the new fit moved by MIN_PARAM_MOVE.
     Before it the close slot runs its workflow steps at 18:07 ET: market data,
     labels and the closebook step, which must store the session's closing books;
     the book slot must still capture the Global Trading Hours book after them
     (after a Friday, when the feed still shows the closing book, find the close
     slot's books stored instead), the forward ledger must trade on the book
     slot's book and the session's surface report must be built from the
     closing book (slot_capture_problems).
     Each slot writes its run record (data_store/runs/, REFRESH_SLOT close and
     book), which must be complete and agree with the slot's status, and the
     capture commit's integrity check (scripts/ci/check_captures.py) must pass
     on everything the two runs wrote.
  3. The three gates of .github/workflows/daily_refresh.yml (their test lists
     are read from the workflow) run in the copy, and with --full the whole
     suite as the weekly tests.yml run would see main after the bot.

With --no-fit the book slot finds Yahoo's book empty, as the Saturday slot
does after every Friday session, so the calibration is skipped and no fit is
stored. The session's surface is still built from its Cboe book (the
backfill: tier 0.7 item 4), and the gates then run on a newest surface
session whose scorecard marks Heston not available, beside an older headline fit. The
run must skip the fit, record the forward ledger from the Cboe book, build
that surface with Heston marked not scored, stay publishable and exit 0. A
Friday session built that way failed ten presentation tests (pages and tests
that assumed a Heston row, and pages that cited different sessions), so the
site would not have deployed on any weekend.

With --force-fail the runs are made to fail (FUND_GRADE_PLAN 0.4): Yahoo's SPX
download raises, as in an outage, in both scheduled slots of the session. The
close slot and then the book slot (market,capture,calibrate,freeze: the one
whose status says whether the site may be published) fail hard on the market
step, exit 1, and each writes a complete run record saying so; the book run
must not be publishable. The workflow's last step (`if: failure()`) then runs
scripts/ci/bot_alert.py; here the copy's script is run as that step runs it,
through its command line, with a fake `gh` first on PATH that answers from a
run history built from the two runs' own records and logs every call (nothing
is sent). With the close run failed before the book run it must open the one
bot-failure issue naming both; with an issue already open it must comment on
it; with a success between them (a cancelled run after it, a failure before)
it must do nothing; a manual run must not even read the history. No fit, no
gates: about two minutes, so test_ci_tools.py runs this mode in full.

    python scripts/ci/simulate_next_session.py [--out DIR] [--full] [--keep] [--no-fit | --force-fail]

Exit 1 when the simulated run does not accept (with --no-fit: skip) the fit and
record the session, or when any gate fails; with --force-fail, when the run does
not fail or the alert does not do what it must. Nothing outside --out is written.
About 10 minutes (+9 with --full) on 4 CPUs; the diagnostics take about 4.
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
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES = ("Data tests (gate the capture commit)", "Artifact tests (gate the artifact commit)",
         "Presentation tests (gate the deploy)")
PARAMS = ("kappa", "theta", "sigma", "rho", "v0")
# the step from the headline fit to the simulated session's parameters
PARAM_STEP = {"kappa": 0.6, "theta": 1.2, "sigma": 0.7, "rho": 0.1, "v0": 1.3}
SIGMA_BELOW = 1.9            # the step is taken from below the sigma <= 2 prior
NOISE_VP = 1.0               # RMS of the mid noise, vol points
# noisy_mid's smile error per unit of NOISE_VP: level, slope, curvature (in z),
# per-strike; an RMS of ~1 over the strikes a book quotes
NOISE_SHAPE = (0.9, 0.35, 0.12, 0.25)
# every parameter of the new fit must differ from the headline by this much
# (relative): 20x the 1e-3 a test would pin a committed value with. Not more:
# kappa is the least identified parameter (its Gauss-Newton SE is ~19% on the
# headline session) and the calibrator warm-starts from the previous fit, so a
# priced book with kappa x0.6 can still be refitted with kappa a few percent
# off the headline (4.3% on the simulated 2026-10-27 session, which failed the
# earlier 5% bar while theta, sigma, rho and v0 moved 12-30%).
MIN_PARAM_MOVE = 0.02


def simulated_params(head: dict, bounds: dict | None = None) -> dict:
    """The simulated session's Heston parameters: the headline fit's moved by
    PARAM_STEP, inside the calibrator's box and at least 5% of its width off
    each bound, with |rho| >= 0.2 (the gate's no-skew check is 0.10)."""
    if bounds is None:
        from joint_vol_calibration.config import HESTON_BOUNDS
        bounds = HESTON_BOUNDS
    raw = {"kappa": head["kappa"] * PARAM_STEP["kappa"], "theta": head["theta"] * PARAM_STEP["theta"],
           "sigma": min(head["sigma"], SIGMA_BELOW) * PARAM_STEP["sigma"],
           "rho": min(head["rho"] + PARAM_STEP["rho"], -0.2), "v0": head["v0"] * PARAM_STEP["v0"]}
    out = {}
    for k in PARAMS:
        lo, hi = bounds[k]
        pad = 0.05 * (hi - lo)
        out[k] = float(min(max(raw[k], lo + pad if k in ("kappa", "sigma", "rho") else 3.0 * lo), hi - pad))
    return out


def noisy_mid(mid, F, K, T, D, is_call, seed: int, noise_vp: float = NOISE_VP):
    """``mid`` moved in Black-76 implied volatility by about ``noise_vp`` vol
    points (RMS), seeded by ``seed``: per expiry a smooth smile error (level,
    slope and curvature in z = ln(K/F) / (0.2 sqrt T), z clipped to +-3) plus a
    small per-strike part. The move depends on the strike only, so a call and
    the put at its strike move by the same vol: put-call parity holds, as it
    does in a real book (independent noise on each made the cleaner drop the
    pair as parity-inconsistent). A quote whose vol cannot be inverted keeps
    its price."""
    import numpy as np

    from joint_vol_calibration.data import option_chain as oc
    mid, K = np.asarray(mid, float), np.asarray(K, float)
    iv = oc.black76_implied_vol(mid, F, K, T, D, is_call)
    rng = np.random.default_rng(seed)
    uk, inv = np.unique(K, return_inverse=True)
    z = np.clip(np.log(uk / F) / (0.2 * np.sqrt(T)), -3.0, 3.0)
    a, b, c = rng.normal(0.0, 1.0, 3) * np.array(NOISE_SHAPE[:3])
    eps = a + b * z + c * (z * z - 1.0) + rng.normal(0.0, NOISE_SHAPE[3], uk.shape)
    moved = oc.black76_price(F, K, T, D, np.maximum(iv + eps[inv] * noise_vp / 100.0, 0.01), is_call)
    return np.where(np.isfinite(iv), moved, mid)


def param_moves(new: dict, head: dict) -> dict:
    """Relative change of each parameter from the headline fit."""
    return {k: abs(float(new[k]) - float(head[k])) / max(abs(float(head[k])), 1e-12) for k in PARAMS}


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


def slot_steps(workflow: Path, slot: str) -> str:
    """The --steps the daily workflow's "Choose steps" case gives a scheduled slot."""
    m = re.search(rf'\)\s*steps="([a-z,]+)"; slot="{re.escape(slot)}"', workflow.read_text())
    if not m:
        raise SystemExit(f"{workflow}: no {slot} slot in the steps case")
    return m.group(1)


def replayed_capture(cboe, und: str, frame, meta: dict, session: str, now):
    """A committed Cboe book replayed as ``session``'s capture, fetched at ``now``
    (tz-aware): the delayed feed's quotes are 15 minutes old, and the book state is
    what cboe.book_state_at gives for them, the rule test_cboe_downloader checks on
    every committed capture. The replay had typed 'gth', which is wrong on a Friday
    evening: that is the closing book, since Monday's GTH opens on Sunday evening."""
    quoted = now - timedelta(minutes=15)
    qt = quoted.strftime("%Y-%m-%dT%H:%M:%SZ")
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    return cboe.ChainCapture(
        underlying=und, frame=frame.assign(snapshot_date=session, quote_time=qt), session=session,
        quote_time=qt, book_state=cboe.book_state_at(quoted, session), fetched_at=stamp,
        payload_timestamp=stamp, underlying_last_trade_time=None,
        underlying_price=float(meta["underlying_price"]), url=meta["url"], sha256=meta["sha256"],
        n_bytes=int(meta["n_bytes"]))


def closing_listing(frame):
    """The close slot's replayed closing book: the committed book less the lines of its
    furthest expiry, so the session's two replayed books differ in lines and expiries as a
    real session's do (2026-10-30's closing book: 27,760 SPX lines over 53 expiries; the
    overnight books before it about 28,700-29,000). Two replays of one file had the same
    counts, so a check that took one book's counts for the other's passed here and failed
    in the bot (the presentation gate on a session with both books, FUND_GRADE_PLAN 0.8)."""
    last = frame["expiry"].max()
    return frame[frame["expiry"] != last].reset_index(drop=True)


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

# --force-fail: what the outage raises (the run must fail on it, and say so)
OUTAGE = "simulated outage (--force-fail): Yahoo Finance unreachable"


def drive(status_file: Path, no_fit: bool = False, force_fail: bool = False) -> int:
    root = ROOT
    sys.path.insert(0, str(root))
    import importlib.util
    import pickle
    from datetime import UTC, datetime, timedelta

    import numpy as np
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
    p = simulated_params(cal["params"])
    spx = dbm.get_spx_ohlcv(as_of_date=last)
    S = float(spx["close"].iloc[-1])
    print(f"simulating session {session} (after {last}) at {now:%Y-%m-%dT%H:%MZ}; chain priced at "
          f"S = {S:.2f} by {p} (the {head['date']} fit {cal['params']} moved), "
          f"{NOISE_VP:g} vp mid noise", flush=True)

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
    if force_fail:
        def outage(*args, **kwargs):
            raise OSError(OUTAGE)
        _patch(dl, download_spx_ohlcv=outage)
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

    from joint_vol_calibration.models.heston import heston_vix_future_exact
    codes = "FGHJKMNQUVXZ"

    def vx_file(trade_date):
        """The CFE settlement file of ``trade_date`` (the close slot asks for every
        session the VX table lacks): the nine monthly contracts from its month on,
        at the model's E^Q[VIX_T] for the session's parameters, to Cboe's 4 dp."""
        if trade_date > session:
            raise OSError(f"no VX settlements after the simulated session ({trade_date})")
        y, m = int(trade_date[:4]), int(trade_date[5:7])
        exps = [e for e in (dbm.vx_monthly_expiry(y + (m - 1 + k) // 12, (m - 1 + k) % 12 + 1)
                            for k in range(11)) if e >= trade_date][:9]
        T = np.array([(pd.Timestamp(e) - pd.Timestamp(trade_date)).days / 365.0 for e in exps])
        px = heston_vix_future_exact(p["kappa"], p["theta"], p["sigma"], p["v0"], T)
        lines = ["Product,Symbol,Expiration Date,Price"] + [
            f"VX,VX/{codes[int(e[5:7]) - 1]}{e[3]},{e},{v:.4f}" for e, v in zip(exps, px)]
        return cboe.parse_vx_settlement_csv("\n".join(lines) + "\n", trade_date)
    _patch(cboe, download_vx_settlements=vx_file,
           download_vix_futures_daily_history=lambda start=None, end=None, expiries=None: pd.DataFrame())

    cap_dir = root / "data_store" / "options" / "cboe"
    replay = {}
    for und in ("SPX", "VIX"):
        src = sorted(cap_dir.glob(f"{und.lower()}_*.parquet"))
        if src:
            replay[und] = (pd.read_parquet(src[-1]), json.loads(src[-1].with_suffix(".json").read_text()))
    for old in cap_dir.glob(f"*_{session}_*"):   # the copy only: the session's capture must happen
        old.unlink()

    # the instant of the slot running: each slot's replayed book is the one the
    # delayed feed shows then (the close slot's closing book, the book slot's GTH book)
    clock = {"now": now}
    close_slot = {"on": False}                  # the close slot replays a closing_listing

    def capture(und, now_=None, **kw):
        if und not in replay:
            raise OSError(f"no committed {und} capture to replay")
        frame, meta = replay[und]
        if close_slot["on"]:
            frame = closing_listing(frame)
        return replayed_capture(cboe, und, frame, meta, session, clock["now"])
    _patch(cboe, capture_option_chain=capture)

    r = float(cal.get("r_reference") or 0.04)
    q = float(cal.get("q") or 0.013)
    start = oc.next_session(last)

    class Ticker:
        def __init__(self, symbol):
            # --no-fit: Yahoo lists no expiry, as on the Saturday slot after a Friday
            self.options = () if no_fit else tuple(
                (start + timedelta(days=i)).isoformat() for i in range(1, 420)
                if (start + timedelta(days=i)).weekday() < 5)

        def option_chain(self, exp):
            import zlib
            spread = 0.2
            df = _heston_chain(params=p, session=session, S=S, r=r, q=q, expiries=[exp], spread=spread)
            T = oc.year_fraction(session, exp, "PM")
            mid = noisy_mid((df["bid"] + df["ask"]) / 2, S * np.exp((r - q) * T), df["strike"], T,
                            np.exp(-r * T), (df["option_type"] == "call").to_numpy(),
                            seed=zlib.crc32(exp.encode()))
            df["bid"] = np.maximum(mid - spread / 2, 0.05)
            df["ask"] = np.maximum(mid + spread / 2, df["bid"] + 0.05)
            calls = df[df.option_type == "call"].drop(columns=["expiration", "option_type"])
            puts = df[df.option_type == "put"].drop(columns=["expiration", "option_type"])
            return type("Chain", (), {"calls": calls, "puts": puts})()

    t0 = time.perf_counter()
    # The day as production runs it: the close slot (22:07 UTC, its steps read from
    # the workflow) stores the session's market data, label and closing books, then
    # the book slot (02:30 UTC) the Global Trading Hours book, the fit and the rest.
    close = datetime.combine(oc.next_session(last), datetime.min.time()).replace(
        hour=18, minute=7, tzinfo=oc.NY_TZ).astimezone(UTC)
    # Each slot writes its run record (data_store/runs/) as the workflow's slots do.
    os.environ["REFRESH_SLOT"] = "close"
    clock["now"], close_slot["on"] = close, True
    code_close = cr.main(["--steps", slot_steps(root / ".github" / "workflows" / "daily_refresh.yml",
                                                "close"),
                          "--status-file", str(status_file.with_name("status_close.json"))], now_utc=close)
    print(f"close slot exit {code_close}", flush=True)
    os.environ["REFRESH_SLOT"] = "book"
    clock["now"], close_slot["on"] = now, False
    code = cr.main(["--steps", "market,capture,calibrate,freeze", "--status-file", str(status_file)],
                   now_utc=now, ticker_factory=Ticker)
    if force_fail:
        # the failed book slot is the run the alert step ends (its status is the outer
        # run's); the close slot, failed on the same outage, is the scheduled run before
        st = json.loads(status_file.read_text())
        st["simulated"] = {"session": session, "after": last, "now_utc": now.isoformat(),
                           "close_utc": close.isoformat(), "close_exit": code_close, "force_fail": True,
                           "seconds": round(time.perf_counter() - t0, 1)}
        status_file.write_text(json.dumps(st, indent=2, default=str))
        print(f"book slot exit {code} (forced failure)", flush=True)
        return code
    code = max(code, code_close)
    st = json.loads(status_file.read_text())
    st["simulated"] = {"session": session, "after": last, "now_utc": now.isoformat(), "no_fit": no_fit,
                       "book_slot_state": cboe.feed_book_state(now, session),
                       "headline_before": head["date"], "headline_params": cal["params"],
                       "book_params": p, "noise_vp": NOISE_VP,
                       "seconds": round(time.perf_counter() - t0, 1)}
    status_file.write_text(json.dumps(st, indent=2, default=str))
    print(f"cloud_refresh exit {code} in {st['simulated']['seconds']} s", flush=True)
    return code


# ── the outer run ─────────────────────────────────────────────────────────────

def run_record_problems(tree: Path, statuses: dict) -> list[str]:
    """The simulated slots' run records (data_store/runs/, FUND_GRADE_PLAN 0.9):
    for each slot's status ({slot: status dict}), one complete record of that
    run whose calibration outcome and exit code are the run's."""
    recs = [json.loads(p.read_text()) for p in sorted((tree / "data_store" / "runs").glob("run_*.json"))]
    problems = []
    for slot, st in statuses.items():
        mine = [r for r in recs if r.get("slot") == slot and r.get("run_utc") == st.get("run_utc")]
        if len(mine) != 1:
            problems.append(f"{len(mine)} {slot} run record(s) for run {st.get('run_utc')}, expected 1")
            continue
        r = mine[0]
        want = (st.get("calibration") or {}).get("status")
        got = (r.get("calibration") or {}).get("status")
        if not r.get("complete") or r.get("exit_code") != st.get("exit_code") or got != want:
            problems.append(f"the {slot} run record (complete {r.get('complete')}, exit "
                            f"{r.get('exit_code')}, calibration {got}) is not the run's "
                            f"(exit {st.get('exit_code')}, calibration {want})")
    return problems


def _file(st: dict, und: str) -> str | None:
    """The name of the book a slot's capture step stored, or found stored, for ``und``."""
    f = ((((st.get("capture") or {}).get("captures") or {}).get(und)) or {}).get("file")
    return Path(f).name if f else None


def slot_capture_problems(close_st: dict, st: dict, book_state: str | None,
                          chain: str | None) -> list[str]:
    """The simulated session's books (FUND_GRADE_PLAN 0.8), from the close slot's
    status, the book slot's status, the state the delayed feed shows at the book
    slot's instant (``book_state``: drive records it) and the chain the session's
    surface.json records. The close slot must store the session's closing books.
    After a Monday-Thursday session the book slot must capture the Global Trading
    Hours books beside them; after a Friday or a holiday eve the feed still shows
    the closing book (no Global Trading Hours until the next session's evening
    before), so it must find the close slot's books stored and fetch nothing. The
    forward ledger trades on the book slot's SPX book, and the surface, with its
    VIX replication, is built from the closing one."""
    problems = []

    def states(s: dict) -> set:
        return {c.get("book_state") for c in ((s.get("capture") or {}).get("captures") or {}).values()}

    close_capt, capt = close_st.get("capture") or {}, st.get("capture") or {}
    if close_capt.get("status") != "captured" or states(close_st) != {"closing"}:
        problems.append(f"close slot capture (want the closing books captured): {close_capt}")
    if book_state == "closing":
        stored = {u: _file(close_st, u) for u in (close_capt.get("captures") or {})}
        if capt.get("status") != "exists" or {u: _file(st, u) for u in stored} != stored:
            problems.append("book slot capture (the feed still shows the closing book: want the close "
                            f"slot's books found stored, {sorted(map(str, stored.values()))}): {capt}")
    elif capt.get("status") != "captured" or states(st) != {"gth"}:
        problems.append(f"book slot capture (want the gth books captured): {capt}")
    led_book = (st.get("forward_ledger") or {}).get("book_file")
    if _file(st, "SPX") and led_book != f"options/cboe/{_file(st, 'SPX')}":
        problems.append(f"the forward ledger traded on {led_book}, not the book slot's {_file(st, 'SPX')}")
    closing = _file(close_st, "SPX")
    if closing and chain != f"data_store/options/cboe/{closing}":
        problems.append(f"the session's surface was built from {chain}, not its closing book {closing}")
    return problems


def book_problems(tree: Path, status_file: Path) -> list[str]:
    """slot_capture_problems on what the simulated slots wrote: the book slot's
    status (``status_file``, with the session and the book state the feed showed
    at its instant, which drive records), the close slot's beside it
    (status_close.json) and the chain the session's surface.json in ``tree``
    records."""
    st = json.loads(status_file.read_text()) if status_file.exists() else {}
    sim = st.get("simulated") or {}
    close_file = status_file.with_name("status_close.json")
    close_st = json.loads(close_file.read_text()) if close_file.exists() else {}
    sj = tree / "data_store" / "surface" / str(sim.get("session")) / "surface.json"
    chain = (json.loads(sj.read_text()).get("inputs") or {}).get("chain") if sj.exists() else None
    return slot_capture_problems(close_st, st, sim.get("book_slot_state"), chain)


def outcome_problems(st: dict, tree: Path, no_fit: bool = False) -> list[str]:
    """What the simulated book slot's status (``st``) must show. A new fit: the fit
    accepted, every parameter moved off the headline, the session's artifacts built.
    --no-fit: the fit skipped (Yahoo's book empty) and the headline left as it was,
    while the session's surface is built from its Cboe book by the backfill, with a
    scorecard that lists Heston as not scored and gives the reason. Both: the forward
    ledger recorded, the session's VX settlements stored, every artifact built and the
    run publishable. The slots' captures are book_problems' (a Friday's book slot finds
    the close slot's books stored rather than capturing one)."""
    problems = []
    sim = st.get("simulated", {})
    session = sim.get("session")
    cal = st.get("calibration") or {}
    if no_fit:
        if cal.get("status") != "skipped":
            problems.append(f"the empty Yahoo book was not skipped: calibration {cal.get('status')} "
                            f"{cal.get('reason')}")
        head = (st.get("headline_calibration") or {}).get("date")
        if head != sim.get("headline_before"):
            problems.append(f"the headline fit moved to {head} without a fit "
                            f"(was {sim.get('headline_before')})")
        built = ((st.get("artifact_backfill") or {}).get("rebuilt") or {}).get(session) or {}
        if not (built.get("surface") or {}).get("ok"):
            problems.append(f"the backfill did not build {session}'s surface from its book: {built}")
        sc_path = tree / "data_store" / "surface" / str(session) / "scorecard.json"
        rows = json.loads(sc_path.read_text())["models"] if sc_path.exists() else []
        heston = [m for m in rows if str(m.get("model", "")).startswith("heston")]
        if not heston or any(m.get("available") is not False or not m.get("reason") for m in heston):
            problems.append(f"{session}'s scorecard does not list Heston as not scored, with its reason: "
                            f"{heston}")
    else:
        if cal.get("status") != "accepted":
            problems.append(f"the simulated fit was not accepted: {cal.get('status')} {cal.get('reason')}")
        fit = (cal.get("summary") or {}).get("params")
        moves = param_moves(fit, sim["headline_params"]) if fit and sim.get("headline_params") else {}
        still = sorted(k for k, m in moves.items() if m < MIN_PARAM_MOVE)
        if fit and still:
            problems.append(f"the new fit did not move off the headline in {still} (relative moves "
                            f"{ {k: round(m, 4) for k, m in moves.items()} } < {MIN_PARAM_MOVE}): "
                            "a test pinning the headline would not be caught")
    if (st.get("forward_ledger") or {}).get("status") != "recorded":
        problems.append(f"forward ledger: {st.get('forward_ledger')}")
    vx = (st.get("market") or {}).get("vx_futures") or {}
    if vx.get("last") != session or not any(
            (tree / "data_store" / "market" / "vix_futures_daily").glob("vix_futures_daily_*.parquet")):
        problems.append(f"VX settlements of the session not stored and persisted: {vx}")
    arts = {**(st.get("artifacts") or {}),
            **{f"{s} {k}": v for s, a in ((st.get("artifact_backfill") or {}).get("rebuilt") or {}).items()
               for k, v in a.items()}}
    for name, res in arts.items():
        if not res.get("ok"):
            problems.append(f"artifact {name}: {res.get('error')}")
    if not st.get("publishable"):
        problems.append(f"not publishable: {st.get('publish_blockers')}")
    return problems


# A stand-in for the GitHub CLI, first on PATH when bot_alert.py runs here. It answers
# from a state file (the run history; the open issues) as `gh` would: `run list` keeps
# the --event and --limit it is given, `issue list` the open issues with the label. It
# logs every call (one JSON line each) and refuses any other command.
FAKE_GH = r"""#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as fh:
    fh.write(json.dumps(args) + "\n")
state = json.load(open(os.environ["FAKE_GH_STATE"]))
def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default
verb = args[:2]
if verb == ["run", "list"]:
    runs = [r for r in state["runs"] if opt("--event") in (None, r["event"])]
    print(json.dumps(runs[:int(opt("--limit", "20"))]))
elif verb == ["issue", "list"]:
    print(json.dumps([i for i in state["open_issues"] if opt("--label") in i["labels"]]))
elif verb == ["label", "create"]:
    pass
elif verb == ["issue", "create"]:
    print("https://github.invalid/issues/1")
elif verb == ["issue", "comment"]:
    print("https://github.invalid/issues/" + args[2] + "#issuecomment-1")
else:
    sys.stderr.write("fake gh: unexpected command " + " ".join(args) + "\n")
    sys.exit(1)
"""
RUNS_URL = "https://github.invalid/actions/runs/"


def run_history(tree: Path, statuses: dict) -> dict:
    """The two forced runs as `gh run list` lists them, from their own run records
    (data_store/runs/): {slot: entry}. Each is a scheduled run with an id of its own
    and its record's start time; the book run, whose last step is the alert, is still
    in progress; the close run is completed, failed or succeeded as its record's exit
    code says."""
    recs = [json.loads(p.read_text()) for p in sorted((tree / "data_store" / "runs").glob("run_*.json"))]
    out = {}
    for n, (slot, st) in enumerate(statuses.items()):
        rec = next(r for r in recs if r.get("slot") == slot and r.get("run_utc") == st.get("run_utc"))
        run_id = 9_000_000 + n
        done = {"status": "completed", "conclusion": "failure" if rec.get("exit_code") else "success"}
        out[slot] = {"databaseId": run_id, "event": "schedule", "createdAt": rec["run_utc"],
                     "url": f"{RUNS_URL}{run_id}",
                     **({"status": "in_progress", "conclusion": ""} if slot == "book" else done)}
    return out


def alert_cli(script: Path, work: Path, runs: list[dict], open_issues: list[dict], run_id: int,
              event: str = "schedule", slot: str = "book") -> tuple[int, dict, list[list[str]]]:
    """Run bot_alert.py (the file ``script``) as the workflow's `if: failure()` step does,
    through its command line, with FAKE_GH first on PATH answering from ``runs`` and
    ``open_issues``. The real gh is never reached: the fake is checked to be the gh
    that PATH finds, and the run has no GitHub token or configuration. Returns (exit
    code, the JSON line it prints, every gh call)."""
    work.mkdir(parents=True, exist_ok=True)
    bin_dir = work / "bin"
    bin_dir.mkdir(exist_ok=True)
    fake = bin_dir / "gh"
    fake.write_text(FAKE_GH.replace("{python}", sys.executable))
    fake.chmod(0o755)
    state, log = work / "gh_state.json", work / "gh_calls.jsonl"
    state.write_text(json.dumps({"runs": runs, "open_issues": open_issues}))
    log.write_text("")
    (work / "gh_config").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GH_", "GITHUB_"))}
    env.update(PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", FAKE_GH_STATE=str(state),
               FAKE_GH_LOG=str(log), GH_CONFIG_DIR=str(work / "gh_config"), GH_HOST="github.invalid")
    if shutil.which("gh", path=env["PATH"]) != str(fake):
        raise RuntimeError(f"the fake gh is not the one PATH finds: {shutil.which('gh', path=env['PATH'])}")
    r = subprocess.run([sys.executable, str(script), "--run-id", str(run_id), "--event", event,
                        "--run-url", f"{RUNS_URL}{run_id}", "--slot", slot,
                        "--workflow", "daily_refresh.yml"],
                       env=env, capture_output=True, text=True, timeout=120, check=False)
    printed: dict = next((json.loads(ln) for ln in r.stdout.splitlines() if ln.startswith("{")), {})
    calls = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    return r.returncode, printed, calls


def _flag(call: list[str], name: str) -> str | None:
    return call[call.index(name) + 1] if name in call and call.index(name) + 1 < len(call) else None


def alert_problems(tree: Path, statuses: dict, work: Path) -> tuple[list[str], list[tuple]]:
    """What the copy's bot_alert.py does at the end of the failed book run, by its
    command line on histories built from the two forced runs' records (run_history):
      - the close run failed before it: create the bot-failure label and open one
        issue, its title the streak, its body naming the close run;
      - the same with a bot-failure issue already open: comment on it, open none;
      - a success between them, a cancelled run after it and the failed close run
        before it: the success ends the streak, so nothing is opened or touched;
      - a manual run of the same failure: not even the history is read.
    Returns (problems, report rows)."""
    script = tree / "scripts" / "ci" / "bot_alert.py"
    label = "bot-failure"
    h = run_history(tree, statuses)
    book, close = h["book"], h["close"]
    if close["conclusion"] != "failure":
        return [f"the close run's record says it did not fail: {close}"], []
    older = {"databaseId": 8_999_990, "event": "schedule", "status": "completed", "conclusion": "success",
             "createdAt": "2026-01-01T00:00:00+00:00", "url": f"{RUNS_URL}8999990"}
    problems: list[str] = []
    rows: list[tuple] = []
    issue = {"number": 7, "url": "https://github.invalid/issues/7", "labels": [label]}
    cases = {
        "two failed in a row": ([book, close, older], [], "schedule"),
        "an issue already open": ([book, close, older], [issue], "schedule"),
        "a success between": ([book, {**older, "databaseId": 8_999_992, "conclusion": "cancelled",
                                      "url": f"{RUNS_URL}8999992"},
                               {**older, "databaseId": 8_999_991, "url": f"{RUNS_URL}8999991"}, close], [],
                              "schedule"),
        "a manual run": ([book, close, older], [], "workflow_dispatch"),
    }
    for name, (runs, issues, event) in cases.items():
        try:
            code, res, calls = alert_cli(script, work / re.sub(r"\W+", "_", name), runs, issues,
                                         book["databaseId"], event=event)
        except Exception as e:  # noqa: BLE001 (reported: the check failed, whatever raised)
            problems.append(f"{name}: bot_alert.py could not be run: {type(e).__name__}: {e}")
            continue
        made = [c for c in calls if c[:2] == ["issue", "create"]]
        said = [c for c in calls if c[:2] == ["issue", "comment"]]
        touched = [c for c in calls if c[0] in ("issue", "label") and c[:2] != ["issue", "list"]]
        if code != 0:
            problems.append(f"{name}: bot_alert.py exited {code}")
        if name == "two failed in a row":
            body = (_flag(made[0], "--body") or "") if made else ""
            if res.get("action") != "created" or res.get("streak") != 2 or len(made) != 1:
                problems.append(f"{name}: one issue was not opened: {res}, gh calls {calls}")
            elif not (_flag(made[0], "--label") == label
                      and any(c[:3] == ["label", "create", label] for c in calls)
                      and close["url"] in body and book["url"] in body
                      and "2 scheduled runs failed" in (_flag(made[0], "--title") or "")):
                problems.append(f"{name}: not the {label} issue naming both runs and the streak: {made[0]}")
        elif name == "an issue already open":
            if res.get("action") != "commented" or made or [c[2] for c in said] != ["7"]:
                problems.append(f"{name}: did not comment on the open issue (and only on it): {res}, "
                                f"gh calls {calls}")
        elif name == "a success between":
            if res.get("action") != "none" or res.get("streak") != 1 or touched:
                problems.append(f"{name}: the success did not end the streak: {res}, gh calls {calls}")
        elif res.get("action") != "none" or calls:
            problems.append(f"{name}: a manual run read or touched GitHub: {res}, gh calls {calls}")
        rows.append((f"failure alert ({name})", code, None,
                     f"{res.get('action')}" + (f" issue {res.get('issue')}" if res.get("issue") else "")
                     + f" (streak {res.get('streak')}; {len(calls)} gh calls)"))
    return problems, rows


def force_fail_problems(code: int, st: dict, tree: Path, close_st: dict,
                        work: Path) -> tuple[list[str], list[tuple]]:
    """--force-fail: both scheduled slots failed on the outage and recorded it, the
    book run is not publishable, and bot_alert.py does what it must on histories built
    from the two runs' records (alert_problems). Returns (problems, report rows)."""
    problems: list[str] = []
    if code == 0 or st.get("exit_code") != 1:
        problems.append(f"the forced failure did not fail the book run (exit {code}, status exit "
                        f"{st.get('exit_code')}): the alert step would not run")
        return problems, []
    for slot, s in (("close", close_st), ("book", st)):
        errs = [str(e) for e in s.get("errors") or []]
        if s.get("exit_code") != 1 or not any(OUTAGE in e for e in errs):
            problems.append(f"the {slot} run did not fail on the forced outage: exit {s.get('exit_code')}, "
                            f"errors {errs}")
    if "freeze" not in (st.get("steps") or []):
        problems.append(f"the book run did not include the freeze step: {st.get('steps')}")
    if st.get("publishable") or not any(OUTAGE in str(b) for b in st.get("publish_blockers") or []):
        problems.append(f"the failed book run is publishable or does not say why not: publishable "
                        f"{st.get('publishable')}, blockers {st.get('publish_blockers')}")
    statuses = {"close": close_st, "book": st}
    rec = run_record_problems(tree, statuses)
    if rec:
        return [*problems, *rec], []
    more, rows = alert_problems(tree, statuses, work)
    return [*problems, *more], rows


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
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--no-fit", action="store_true",
                      help="Yahoo's book empty (the Saturday slot): no fit, the surface from the Cboe book")
    mode.add_argument("--force-fail", action="store_true",
                      help="Yahoo down in both slots: the runs must fail, the book run unpublishable, and "
                           "bot_alert.py, run with a fake gh on the runs' own records, must open the "
                           "bot-failure issue")
    ap.add_argument("--drive", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.drive:
        return drive(args.drive, no_fit=args.no_fit, force_fail=args.force_fail)

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
                            "--drive", str(status_file), *(["--no-fit"] if args.no_fit else []),
                            *(["--force-fail"] if args.force_fail else [])],
                           cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False)
    problems: list[str] = []
    st = json.loads(status_file.read_text()) if status_file.exists() else {}
    sim = st.get("simulated", {})
    if args.force_fail:
        close_file = status_file.with_name("status_close.json")
        close_st = json.loads(close_file.read_text()) if close_file.exists() else {}
        problems, rows = force_fail_problems(r.returncode, st, tree, close_st, out / "alert")
        rows.insert(0, ("bot runs (forced failure: Yahoo's SPX download raises in both slots)", r.returncode,
                        sim.get("seconds"),
                        (f"session {sim.get('session')}: close exit {close_st.get('exit_code')}, book exit "
                         f"{st.get('exit_code')}, publishable {st.get('publishable')}; errors "
                         f"{st.get('errors')}")))
        return _report(out, tree, rows, problems, args.keep,
                       "Both runs failed as forced; the alert opens, comments on or leaves the issue as it "
                       "must.")
    cal = st.get("calibration") or {}
    if r.returncode != 0:
        problems.append(f"cloud_refresh exited {r.returncode}: {st.get('errors')}")
    problems += outcome_problems(st, tree, no_fit=args.no_fit)
    fit = (cal.get("summary") or {}).get("params")
    moves = param_moves(fit, sim["headline_params"]) if fit and sim.get("headline_params") else {}
    close_file = status_file.with_name("status_close.json")
    close_st = json.loads(close_file.read_text()) if close_file.exists() else {}
    problems += book_problems(tree, status_file)
    problems += run_record_problems(tree, {"close": close_st, "book": st})
    # the capture commit's integrity gate, on what the simulated runs wrote
    ok_list = out / "captures.ok"
    with open(out / "logs" / "check_captures.log", "w") as fh:
        rc = subprocess.run([sys.executable, str(tree / "scripts" / "ci" / "check_captures.py"), "--out",
                             str(ok_list), "--root", str(tree)], cwd=tree, env=env, stdout=fh,
                            stderr=subprocess.STDOUT, check=False).returncode
    if rc != 0:
        problems.append(f"check_captures.py exited {rc} on the simulated captures (logs/check_captures.log)")
    ok_arts = sorted(k for k, v in (st.get("artifacts") or {}).items() if v.get("ok")) + sorted(
        f"{k} (backfill)" for k, v in (((st.get("artifact_backfill") or {}).get("rebuilt") or {})
                                       .get(sim.get("session")) or {}).items() if v.get("ok"))
    rmse = (cal.get("summary") or {}).get("spx_iv_rmse")
    rows = [("bot run" + (" (no fit: Yahoo's book empty)" if args.no_fit else ""), r.returncode,
             sim.get("seconds"), f"session {sim.get('session')}: calibration {cal.get('status')}"
             + (f" (SPX RMSE {rmse:.2f} vp; parameters moved "
                + ", ".join(f"{k} {m:.0%}" for k, m in moves.items()) + ")" if moves and rmse else "")
             + f", artifacts {ok_arts}")]
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
    return _report(out, tree, rows, problems, args.keep)


def _report(out: Path, tree: Path, rows: list, problems: list[str], keep: bool,
            passed: str = "All gates pass.") -> int:
    report = ["| step | exit | seconds | result |", "|---|---|---|---|"]
    report += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    report += ["", *(f"- PROBLEM: {x}" for x in problems)] if problems else ["", passed]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print("\n".join(report), flush=True)
    if not keep and not problems:
        shutil.rmtree(tree, ignore_errors=True)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
