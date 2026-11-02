"""
quotes.py — the scoring universe of one session: cleaned OTM SPX quotes with
bid / mid / ask implied vols and spread units.

The universe is exactly the quote set the joint calibrator scores out of
sample (JointCalibrator.spx_surface_full): two-sided quotes cleaned by
option_chain.clean_option_chain (per-expiry parity F_T, D_T; own Black-76
IVs), OTM against F_T, 7 days <= T <= 2 years, 0.70 <= K/F <= 1.30,
mid >= $0.50, 1% <= mid IV <= 200%. Using the calibrator's own filter
constants means a Heston fit scored here reproduces the OOS RMSE stored in its
pickle (tests/test_surface_scorecard.py pins this), so every model in the
scorecard is measured on the same quotes.

Added per quote:
  bid_iv, ask_iv   Black-76 IVs of the bid and ask on the same F_T, D_T (an
                   OTM bid whose time value is below the IV solver's floor
                   gets bid_iv = 0, i.e. the spread's lower edge is 0 vol)
  half_spread_vp   (ask_iv - bid_iv) / 2 in vol points: a model IV is inside
                   the quoted spread iff bid_iv <= model_iv <= ask_iv
  vega             discounted Black-76 vega at mid IV, price per 1 vol point
  delta            forward (undiscounted) Black-76 delta of the quoted option
  delta_bucket     10P / 25P / ATM-P / ATM-C / 25C / 10C by |delta|
                   (<= 0.10, (0.10, 0.25], > 0.25) and side
  tenor_bucket     calendar days to expiry: 7-30d, 30-90d, 90-180d,
                   180-365d, 365-730d (right-closed)

Inputs are read through the repo's look-ahead-gated getters: the session's
closing Cboe book, else its newest stored one (data_store/options/cboe/
spx_<session>_<HHMMSS>Z.parquet, its sidecar's quote time and book state, the
full listing) when it passes the forward ledger's quote-state gates, else the
session's Yahoo chain
data_store/options/spx_<session>.parquet (database.options_snapshot_path;
select_session_chain records which and why), the session's SPX close
(get_spx_ohlcv(as_of_date=session)) and the reference rate the session's
joint fit used (joint_calibrator.session_rate_reference: ^IRX for every fit
committed before C23, the Treasury zero curve r(T) after), so the universe is
the calibrator's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from joint_vol_calibration.calibration import joint_calibrator as _jc
from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc

# The calibrator's universe filters (single source of truth: joint_calibrator).
MIN_T_DAYS: int = _jc._MIN_T_DAYS
MAX_T_YEARS: float = _jc._MAX_T_YEARS
MONEYNESS_LO: float = _jc._MONEYNESS_LO
MONEYNESS_HI: float = _jc._MONEYNESS_HI
MIN_PRICE: float = _jc._MIN_PRICE
MIN_IV: float = _jc._MIN_IV
MAX_IV: float = _jc._MAX_IV

IRX_SOURCE: str = _jc._IRX_SOURCE    # the label of every pre-C23 session's rate

TENOR_EDGES_DAYS = (7.0, 30.0, 90.0, 180.0, 365.0, 730.0)
TENOR_LABELS = ("7-30d", "30-90d", "90-180d", "180-365d", "365-730d")
DELTA_BUCKETS = ("10P", "25P", "ATM-P", "ATM-C", "25C", "10C")

QUOTE_COLUMNS = [
    "session", "expiry", "T", "T_days", "settlement", "root", "strike", "right",
    "forward", "discount", "k", "bid", "ask", "mid", "bid_iv", "mid_iv", "ask_iv",
    "half_spread_vp", "vega", "delta", "delta_bucket", "tenor_bucket",
]


@dataclass
class SessionQuotes:
    """One session's scoring universe and the inputs it was built from.

    quotes : DataFrame with QUOTE_COLUMNS, sorted by (T, strike, right)
    carry  : per-expiry parity carry of the cleaned chain (expiry, T, forward,
             discount, r, q, method, n_pairs, resid_rmse)
    raw    : the stored chain as read (every stored line, incl. zero bids
             where the snapshot kept them) — the VIX replication reads this
    inputs : which chain it is and why (select_session_chain): source,
             source_reason, chain (repository path), chain_sha256,
             raw_sha256 (a Cboe capture's payload hash), book_state,
             fetched_at, refused (books that failed the quote-state gates)
    """
    session: str
    spot: float
    r_ref: float
    quotes: pd.DataFrame
    carry: pd.DataFrame
    raw: pd.DataFrame
    clean_diagnostics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    r_ref_source: str = _jc._IRX_SOURCE
    inputs: dict = field(default_factory=dict)

    @property
    def expiries(self) -> list:
        return list(dict.fromkeys(self.quotes["expiry"]))


# ── which chain a session's surface is built from ─────────────────────────────

# Columns of a Cboe capture (cboe_downloader.parse_cboe_options_payload) that
# the stored-chain schema does not have, or holds with another meaning.
CAPTURE_EXTRA_COLUMNS = ("bid_size", "ask_size", "theo", "rho", "last_trade_price",
                         "option_last_trade_time")


def _book_files(pattern: str) -> list:
    """The stored Cboe books matching ``pattern``, wherever they are: the private
    store the daily workflow checks out (cboe_downloader.BOOKS_DIR_ENV) first, then
    data_store/options/cboe/; a book in both is taken once, from the first. The
    close slot's closing book is moved to the private store when its run ends, so
    the book slot that builds the session's surface finds it only there.
    data_store/ is the parent of database.OPTIONS_DIR."""
    from joint_vol_calibration.data import cboe_downloader as cd
    seen: dict = {}
    for d in cd.book_dirs(Path(db.OPTIONS_DIR).parent):
        for p in sorted(d.glob(pattern)):
            seen.setdefault(p.name, p)
    return [seen[n] for n in sorted(seen)]


# cboe_downloader.book_state_at: the session's closing book, quoted after its close
CLOSING_BOOK = "closing"


def _fetched(cap: tuple) -> tuple:
    """session_captures' sort key: (fetched_at, file name)."""
    return (str(cap[1].get("fetched_at") or ""), cap[0].name)


def session_captures(session: str, underlying: str = "SPX") -> list:
    """Every stored Cboe capture of ``session`` (_book_files: the private store
    and the public tree), newest first by fetch time:
    [(parquet path, sidecar dict)] (cloud_refresh.capture_cboe_chains writes
    <und>_<session>_<HHMMSS>Z.parquet and .json; the sidecar holds quote_time,
    book_state, fetched_at, rows, two_sided_rows and the raw payload's sha256)."""
    import json

    out = []
    for p in _book_files(f"{underlying.lower()}_{session}_*.parquet"):
        js = p.with_suffix(".json")
        try:
            side = json.loads(js.read_text()) if js.exists() else {}
        except (OSError, ValueError):
            side = {}                 # no sidecar record: the frame's own quote_time is read
        out.append((p, side if isinstance(side, dict) else {}))
    return sorted(out, key=_fetched, reverse=True)


def _close_slot_runs(session: str) -> list:
    """The runs of the closebook step alone (the close slot's steps: "closebook"
    without "capture") whose capture is of ``session``, from their run records
    (data_store/runs/run_*.json, scripts/cloud_refresh.py run_record):
    [(run_utc, the SPX capture's record or {})], oldest first. data_store/ is
    the parent of database.OPTIONS_DIR."""
    import json

    runs = Path(db.OPTIONS_DIR).parent / "runs"
    found = []
    for p in sorted(runs.glob("run_*.json")) if runs.is_dir() else []:
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or rec.get("schema") != "run_ledger/1":
            continue
        steps = rec.get("steps") or []
        capt = rec.get("capture") or {}
        if "closebook" in steps and "capture" not in steps and str(capt.get("session")) == session:
            spx = (capt.get("captures") or {}).get("SPX")
            found.append((str(rec.get("run_utc")), spx if isinstance(spx, dict) else {}))
    return sorted(found, key=lambda x: x[0])


def closing_book_refusal(session: str) -> str | None:
    """Why the job did not store ``session``'s closing SPX book, as its run
    records say: the reason of the newest close-slot run (_close_slot_runs)
    whose SPX capture was "rejected" (it failed the quote-state gate, was
    empty or was another session's book) or "failed". None when no such run
    says so (closing_book_run says what the records do hold)."""
    found = [(t, str(spx["reason"])) for t, spx in _close_slot_runs(session)
             if spx.get("status") in ("rejected", "failed") and spx.get("reason")]
    return max(found)[1] if found else None


# closing_book_run statuses: what the close slot's run records say of a session
# whose closing book is not stored
CLOSE_RUN_REFUSED, CLOSE_RUN_CAPTURED, CLOSE_RUN_SKIPPED, CLOSE_RUN_NONE = (
    "refused", "captured", "skipped", "none")


def closing_book_run(session: str) -> str:
    """What the close slot's run records (_close_slot_runs) say of ``session``'s
    closing SPX book, for a session whose store holds none: "refused" (a run
    did not store it and says why: closing_book_refusal), "captured" (a run
    captured it, or found it stored, yet the store holds none: it was not kept,
    as when the move to the private store fails), "skipped" (the runs stored
    none because, by the clock or by the quote time of the book they
    downloaded, the delayed feed showed another book) or "none" (no run of the
    close slot is recorded for the session: it had not run yet, or did not
    exist then). Nothing here says how long the feed shows a closing book:
    that is not observed."""
    if closing_book_refusal(session):
        return CLOSE_RUN_REFUSED
    runs = _close_slot_runs(session)
    if any(spx.get("status") in ("captured", "exists") for _, spx in runs):
        return CLOSE_RUN_CAPTURED
    return CLOSE_RUN_SKIPPED if runs else CLOSE_RUN_NONE


def captured_sessions(underlying: str = "SPX") -> list:
    """Every session with a stored Cboe capture of ``underlying``, newest first."""
    names = (p.name for p in _book_files(f"{underlying.lower()}_*.parquet"))
    return sorted({n.split("_")[1] for n in names if len(n.split("_")) == 3}, reverse=True)


def cboe_book_refusal(session: str) -> str | None:
    """None when ``session``'s surface can be built from one of its stored Cboe
    captures (select_session_chain(source="cboe") finds one that passes the
    quote-state gates and cleans), whatever its Heston fit did; otherwise
    why not (no capture, no stored close, every capture refused)."""
    session = str(pd.Timestamp(session).date())
    if not session_captures(session):
        return f"no stored Cboe capture of {session}"
    try:
        spot = session_spot(session)
        ref = _jc.session_rate_reference(session)
        select_session_chain(session, spot, ref.for_clean(), source="cboe")
    except (ValueError, OSError) as e:
        return str(e)
    return None


def stored_chain_from_capture(frame: pd.DataFrame, underlying: str = "SPX") -> pd.DataFrame:
    """A stored Cboe capture in the stored-chain layout the surface layer
    reads (options_snapshots, as data_store/options/spx_<s>.parquet holds a
    Yahoo chain): the one place the two layouts are mapped.

      underlying          added (a capture's file name carries it)
      implied_vol         Cboe's own IV in a capture; a stored chain's column
                          holds this repository's Black-76 IV (the cleaner
                          recomputes it), so Cboe's moves to vendor_iv
      open_interest,      nullable Int64 in a capture, float here (NaN when
      volume              missing)
      extras              CAPTURE_EXTRA_COLUMNS are dropped
      quote_time, source  kept: the capture's delayed-quote instant and 'cboe'

    Every line is kept, zero bids and one-sided lines included: the cleaner
    keeps two-sided lines only, and the VIX replication's zero-bid stop reads
    the rest. delta..theta stay as Cboe delivers them."""
    df = frame.drop(columns=[c for c in CAPTURE_EXTRA_COLUMNS if c in frame.columns]).copy()
    df.insert(1, "underlying", underlying.upper())
    df["vendor_iv"] = (pd.to_numeric(df["implied_vol"], errors="coerce") if "implied_vol" in df.columns
                       else np.nan)
    df["implied_vol"] = np.nan
    for col in ("open_interest", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def file_sha256(path: Path) -> str | None:
    import hashlib

    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unreadable_closing_book(session: str, readable: set) -> None:
    """Raise FileNotFoundError when a manifest (data_store/options/cboe_manifest,
    the public record of a book held in the private store) records a closing
    SPX book of ``session`` that is not among the books this run can read
    (``readable``: their parquet file names): the private store was not checked
    out, or its clone failed. Building the session from its other books would
    put the overnight book where the record holds the closing one, and the
    surface would be committed and never rebuilt, so the session is not built
    (it fails closed; a later run with the store builds it)."""
    import json

    from joint_vol_calibration.data import cboe_downloader as cd

    for mp in sorted(cd.manifest_dir(Path(db.OPTIONS_DIR).parent).glob(f"spx_{session}_*.json")):
        try:
            rec = json.loads(mp.read_text())
        except (OSError, ValueError):
            continue
        if (isinstance(rec, dict) and rec.get("book_state") == CLOSING_BOOK
                and f"{mp.stem}.parquet" not in readable):
            raise FileNotFoundError(
                f"the closing book {mp.stem} of {session} is held in the private store (its manifest "
                f"options/cboe_manifest/{mp.name}) and this run cannot read it: not building the session "
                "from another book")


def select_session_chain(session: str, spot: float, r_clean, *,
                         source: str = "auto") -> tuple:
    """(raw, clean, inputs): the chain ``session``'s surface is built from.

    The session's closing book when one is stored (session_captures, book
    state 'closing': the regular session's last quotes, standing at the 16:15
    ET close, 13:15 on an early close, when the official VIX-family closes are
    computed, so the VIX replication is set against them at one instant;
    FUND_GRADE_PLAN 0.8), else its newest stored Cboe SPX capture (the
    overnight Global Trading Hours book), the first of them that passes the
    forward ledger's quote-state gates (forward_ledger.book_time_refusal:
    quoted at or after the 16:00 ET close; book_share_refusal: >=
    option_chain.MIN_TWO_SIDED_SHARE of near-ATM lines two-sided) and cleans,
    with T from its quote time (option_chain.t_origin); else the session's
    Yahoo chain data_store/options/spx_<session>.parquet, the calibration's
    download, as the labelled fallback. ``source``: "auto" (that order),
    "cboe" (a capture or ValueError), "yahoo" (the stored Yahoo chain only).
    ``inputs`` records which and why."""
    from joint_vol_calibration.backtest import forward_ledger as fl
    from joint_vol_calibration.data import cboe_downloader as cd

    if source not in ("auto", "cboe", "yahoo"):
        raise ValueError(f"source must be 'auto', 'cboe' or 'yahoo', got {source!r}")
    refused: list[str] = []
    # closing books first, each kind newest first
    caps = sorted(session_captures(session) if source != "yahoo" else [],
                  key=lambda c: c[1].get("book_state") != CLOSING_BOOK)
    n_closing = sum(c[1].get("book_state") == CLOSING_BOOK for c in caps)
    if source != "yahoo":
        unreadable_closing_book(session, {p.name for p, _ in caps})
    for path, side in caps:
        label = f"options/cboe/{path.name}"
        try:
            frame = pd.read_parquet(path)
            qt = side.get("quote_time")
            if not qt and "quote_time" in frame.columns and frame["quote_time"].notna().any():
                qt = str(frame["quote_time"].dropna().iloc[0])
            why = fl.book_time_refusal(session, qt, side.get("fetched_at"))
            if why:
                refused.append(f"{label}: {why}")
                continue
            raw = stored_chain_from_capture(frame)
            clean = oc.clean_option_chain(raw, session, spot, r_ref=r_clean, quote_time=qt,
                                          fetched_at=side.get("fetched_at"))
        except Exception as e:  # noqa: BLE001 (a capture that cannot be read or cleaned is refused, never the book)
            refused.append(f"{label}: unreadable ({type(e).__name__}: {e})")
            continue
        why = fl.book_share_refusal(clean) or (None if clean.ok else clean.reason)
        # a closing-state book (by its quote time) is the closing book only if its
        # underlying last traded on the session at or before the close (w5 review)
        check = (cd.closing_book_check(side.get("underlying_last_trade_time"), session)
                 if side.get("book_state") == CLOSING_BOOK else None)
        if not why and check and check["result"] == cd.LAST_TRADE_LATE:
            why = (f"its underlying's last trade, {check['last_trade']} ET, is not at or before the "
                   f"session's {check['cutoff']} ET close, so it is not the closing book")
        if why:
            refused.append(f"{label}: {why}")
            continue
        if check:
            traded = (f"its underlying's last trade {check['last_trade']} ET, at or before that close"
                      if check["result"] == cd.LAST_TRADE_OK else
                      "its record holds no underlying last-trade time, so that could not be checked")
            which = (f"the session's closing book (its quotes the ones standing at the "
                     f"{db.close_data_cutoff(session):%H:%M} ET close, when the official VIX-family "
                     f"closes are computed; {traded})")
        else:
            newest = "the newest" if path == max(caps, key=_fetched)[0] else "an earlier"
            which = (f"{newest} stored Cboe book of the session"
                     + ("" if n_closing else " (no closing book of the session is stored)"))
        return raw, clean, {
            "source": "cboe",
            "source_reason": (f"{which}; it passes the "
                              "quote-state gates (quoted at or after the 16:00 ET close, "
                              f"{clean.diagnostics['near_atm_two_sided_share']:.1%} of near-ATM "
                              "lines two-sided)"
                              + (f"; refused: {'; '.join(refused)}" if refused else "")),
            # the book's name in either store (options/cboe/<name> in the private
            # repository); once it is held there, its public record is the manifest
            # data_store/options/cboe_manifest/<stem>.json, whose parquet_sha256 the
            # artifact gate matches against chain_sha256
            "chain": f"data_store/options/cboe/{path.name}", "chain_sha256": file_sha256(path),
            "raw_sha256": side.get("sha256"), "book_state": side.get("book_state"),
            "fetched_at": side.get("fetched_at"), "capture_rows": side.get("rows"),
            "capture_two_sided_rows": side.get("two_sided_rows"), "refused": refused,
            "underlying_last_trade_time": side.get("underlying_last_trade_time"),
            "closing_book_check": check,
            "closing_books": n_closing,
            "closing_book_refusal": None if n_closing else closing_book_refusal(session),
            "closing_book_run": None if n_closing else closing_book_run(session)}
    if source == "cboe":
        raise ValueError(f"no stored Cboe book of {session} passes the quote-state gates"
                         + (f": {'; '.join(refused)}" if refused else ""))
    path = db.options_snapshot_path(session, "SPX")
    if not path.exists():
        raise FileNotFoundError(f"no stored SPX chain for {session}: {path}"
                                + (f" (Cboe books refused: {'; '.join(refused)})" if refused else ""))
    raw = pd.read_parquet(path)
    clean = oc.clean_option_chain(raw, session, spot, r_ref=r_clean)
    why = ("Yahoo chain requested" if source == "yahoo" else
           "no stored Cboe book of the session" if not caps else
           f"no stored Cboe book passes the quote-state gates ({'; '.join(refused)})")
    return raw, clean, {
        "source": clean.diagnostics.get("source"),
        "source_reason": f"{why}; the session's Yahoo chain (the calibration's download) is the fallback",
        "chain": f"data_store/options/spx_{session}.parquet", "chain_sha256": file_sha256(path),
        "raw_sha256": None, "book_state": None, "fetched_at": None, "refused": refused,
        "closing_books": n_closing,
        "closing_book_refusal": None if n_closing else closing_book_refusal(session),
        "closing_book_run": None if n_closing else closing_book_run(session)}


def read_stored_chain(session: str, underlying: str = "SPX") -> pd.DataFrame:
    """The committed chain data_store/options/<und>_<session>.parquet."""
    path = db.options_snapshot_path(session, underlying)
    if not path.exists():
        raise FileNotFoundError(f"no stored {underlying} chain for {session}: {path}")
    return pd.read_parquet(path)


def fit_fetched_at(session: str) -> str | None:
    """When the chain of ``session``'s committed joint fit was downloaded
    (its chain_diagnostics fetched_at), or None. A Yahoo chain records no
    quote time and options_snapshots no download instant, so the fit's record
    is where a stored Yahoo chain's fetch time lives."""
    import pickle

    path = Path(DATA_DIR) / "calibrations" / f"joint_cal_{session}.pkl"
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            cal = pickle.load(f)  # noqa: S301 (trusted: the calibrator's own committed result)
    except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, ValueError, TypeError):
        return None
    diag = cal.get("chain_diagnostics") if isinstance(cal, dict) else None
    got = (diag or {}).get("fetched_at")
    return str(got) if got else None


def stored_sessions(underlying: str = "SPX") -> list:
    """Sessions with a committed chain under data_store/options/, sorted."""
    root = db.options_snapshot_path("2000-01-01", underlying).parent
    out = []
    for p in sorted(root.glob(f"{underlying.lower()}_*.parquet")):
        s = p.stem.split("_", 1)[1]
        try:
            pd.Timestamp(s)
        except ValueError:
            continue
        out.append(s)
    return out


def session_spot(session: str) -> float:
    """SPX close of ``session`` (look-ahead gated: as_of_date = session)."""
    hist = db.get_spx_ohlcv(as_of_date=session)
    if hist.empty or str(hist.iloc[-1]["date"])[:10] != session:
        raise ValueError(f"no SPX close stored for session {session}")
    return float(hist.iloc[-1]["close"])


def black76_delta(F, K, T, vol, is_call):
    """Forward (undiscounted) Black-76 delta, vectorised."""
    F, K, T, vol = (np.asarray(x, dtype=float) for x in (F, K, T, vol))
    sd = np.maximum(vol * np.sqrt(np.maximum(T, 0.0)), 1e-300)
    d1 = np.log(F / K) / sd + 0.5 * sd
    return np.where(np.asarray(is_call, dtype=bool), ndtr(d1), ndtr(d1) - 1.0)


def delta_bucket(delta, is_call) -> np.ndarray:
    a = np.abs(np.asarray(delta, dtype=float))
    is_call = np.asarray(is_call, dtype=bool)
    lvl = np.where(a <= 0.10, 0, np.where(a <= 0.25, 1, 2))
    put_lab = np.array(["10P", "25P", "ATM-P"])
    call_lab = np.array(["10C", "25C", "ATM-C"])
    return np.where(is_call, call_lab[lvl], put_lab[lvl])


def tenor_bucket(T_days) -> np.ndarray:
    d = np.asarray(T_days, dtype=float)
    idx = np.searchsorted(np.asarray(TENOR_EDGES_DAYS[1:-1]), d, side="left")
    return np.asarray(TENOR_LABELS)[np.clip(idx, 0, len(TENOR_LABELS) - 1)]


def universe_from_clean(clean: oc.CleanChain, session: str) -> tuple:
    """(quotes, counts): the calibrator's scoring universe of a cleaned chain,
    with spread units added. Same filters, same order as
    JointCalibrator._prepare_spx_surface (spx_surface_full)."""
    q = clean.quotes.copy()
    kf = q["strike"] / q["forward"]
    full = q[q["otm"].astype(bool)
             & (q["time_to_expiry"] >= MIN_T_DAYS / 365.0)
             & (q["time_to_expiry"] <= MAX_T_YEARS)
             & kf.between(MONEYNESS_LO, MONEYNESS_HI)
             & (q["mid_price"] > MIN_PRICE)
             & q["implied_vol"].between(MIN_IV, MAX_IV)].copy()
    full = full.sort_values(["time_to_expiry", "strike"], kind="mergesort").reset_index(drop=True)
    counts = {"n_clean": len(q), "n_universe": len(full)}

    F = full["forward"].to_numpy(float)
    D = full["discount"].to_numpy(float)
    K = full["strike"].to_numpy(float)
    T = full["time_to_expiry"].to_numpy(float)
    is_call = (full["right"] == "C").to_numpy()
    bid_iv = oc.black76_implied_vol(full["bid"].to_numpy(float), F, K, T, D, is_call)
    ask_iv = oc.black76_implied_vol(full["ask"].to_numpy(float), F, K, T, D, is_call)
    mid_iv = full["implied_vol"].to_numpy(float)
    counts["n_bid_iv_floor"] = int(np.isnan(bid_iv).sum())
    bid_iv = np.where(np.isnan(bid_iv), 0.0, bid_iv)
    bad_ask = ~np.isfinite(ask_iv)
    counts["n_dropped_no_ask_iv"] = int(bad_ask.sum())
    keep = ~bad_ask
    vega = oc.black76_vega(F, K, T, D, mid_iv) / 100.0
    delta = black76_delta(F, K, T, mid_iv, is_call)

    out = pd.DataFrame({
        "session": session,
        "expiry": full["expiry"].astype(str).to_numpy(),
        "T": T,
        "T_days": T * 365.0,
        "settlement": full["settlement"].astype(str).to_numpy(),
        "root": full["root"].astype(str).to_numpy(),
        "strike": K,
        "right": full["right"].astype(str).to_numpy(),
        "forward": F,
        "discount": D,
        "k": np.log(K / F),
        "bid": full["bid"].to_numpy(float),
        "ask": full["ask"].to_numpy(float),
        "mid": full["mid_price"].to_numpy(float),
        "bid_iv": bid_iv,
        "mid_iv": mid_iv,
        "ask_iv": ask_iv,
        "half_spread_vp": 50.0 * (ask_iv - bid_iv),
        "vega": vega,
        "delta": delta,
        "delta_bucket": delta_bucket(delta, is_call),
        "tenor_bucket": tenor_bucket(T * 365.0),
    })[keep].reset_index(drop=True)
    return out[QUOTE_COLUMNS], counts


def load_session_quotes(session: str, *, chain: pd.DataFrame | None = None,
                        spot: float | None = None,
                        r_ref: float | None = None, source: str = "auto") -> SessionQuotes:
    """Build the scoring universe of ``session`` from its best stored book.

    chain  : a stored/raw chain frame to use instead of the stored books
    spot   : SPX close override (default: the stored close of ``session``)
    r_ref  : reference rate override (one cc rate for every expiry). Default:
             the rate reference of the session's joint fit
             (joint_calibrator.session_rate_reference)
    source : which stored book (select_session_chain): "auto" (the session's
             Cboe capture when it passes the quote-state gates, else its
             Yahoo chain), "cboe" or "yahoo"

    Raises ValueError when the cleaned chain is unusable (the cleaner's
    fail-closed reason is passed through).
    """
    session = str(pd.Timestamp(session).date())
    S = session_spot(session) if spot is None else float(spot)
    if r_ref is None:
        ref = _jc.session_rate_reference(session)
        r, r_clean = float(ref.r_91d), ref.for_clean()
        r_src = _jc._IRX_SOURCE if ref.kind == "irx" else ref.label
    else:
        r = r_clean = float(r_ref)
        r_src = "argument"
    if chain is None:
        raw, clean, inputs = select_session_chain(session, S, r_clean, source=source)
    else:
        raw = chain
        clean = oc.clean_option_chain(raw, session, S, r_ref=r_clean)
        inputs = {"source": clean.diagnostics.get("source"), "source_reason": "chain argument",
                  "chain": "argument", "chain_sha256": None, "raw_sha256": None,
                  "book_state": None, "fetched_at": None, "refused": []}
    if not clean.ok:
        raise ValueError(f"SPX chain for {session} unusable: {clean.reason}")
    quotes, counts = universe_from_clean(clean, session)
    if quotes.empty:
        raise ValueError(f"SPX chain for {session}: no quote inside the scoring universe")
    diag = {
        **counts,
        "n_expiries": int(quotes["expiry"].nunique()),
        "T_days_min": float(quotes["T_days"].min()),
        "T_days_max": float(quotes["T_days"].max()),
        "median_half_spread_vp": float(np.median(quotes["half_spread_vp"])),
        "t_origin": clean.diagnostics.get("t_origin"),
        "t_origin_basis": clean.diagnostics.get("t_origin_basis"),
        # what the chain is and when it was quoted / downloaded (a Yahoo
        # chain records no quote time: quote_time None, fetched_at set)
        "source": clean.diagnostics.get("source"),
        "quote_time": clean.diagnostics.get("quote_time"),
        "fetched_at": (clean.diagnostics.get("fetched_at")
                       or (fit_fetched_at(session)
                           if not oc.records_quote_time(clean.diagnostics.get("source")) else None)),
        "n_raw": clean.diagnostics.get("n_raw"),
        "n_two_sided": clean.diagnostics.get("n_two_sided"),
        "near_atm_two_sided_share": clean.diagnostics.get("near_atm_two_sided_share"),
        "universe_filters": {
            "otm_vs_parity_forward": True, "min_T_days": MIN_T_DAYS,
            "max_T_years": MAX_T_YEARS, "K_over_F": [MONEYNESS_LO, MONEYNESS_HI],
            "min_mid_price": MIN_PRICE, "mid_iv_range": [MIN_IV, MAX_IV],
            "note": ("identical to JointCalibrator.spx_surface_full; 0-6 DTE "
                     "expiries are excluded, as in the calibrator"),
        },
    }
    inputs = {**inputs, "fetched_at": inputs.get("fetched_at") or diag["fetched_at"]}
    return SessionQuotes(session=session, spot=S, r_ref=r, quotes=quotes,
                         carry=clean.carry.copy(), raw=raw,
                         clean_diagnostics=dict(clean.diagnostics), diagnostics=diag,
                         r_ref_source=r_src, inputs=inputs)


def expiry_groups(quotes: pd.DataFrame):
    """Yield (expiry, T, F, D, frame) per expiry in T order (one T per expiry)."""
    for exp, g in quotes.groupby("expiry", sort=False):
        yield (exp, float(g["T"].iloc[0]), float(g["forward"].iloc[0]),
               float(g["discount"].iloc[0]), g)
