"""
forward_ledger.py — a pre-registered forward test at executable prices.

The backtest marks every straddle to a model (Black-Scholes at β·VIX). This
module does the opposite, going forward: on each session the daily bot
captures, it turns that session's signals into REAL listed SPX straddles,
records their entry bid/ask from the captured chain, and on every later
captured session marks each open straddle at that chain's bid, ask and mid.
P&L is booked at executable prices: a long straddle is bought at the asks and
sold at the bids, a short one sold at the bids and bought back at the asks; a
straddle held to expiry settles at its intrinsic value on the official SPX
close of a PM-settled expiry (no exit spread).

Each session writes one file, ledger_<session>.json, in a deterministic
schema (LEDGER_SCHEMA; keys sorted, floats to 10 significant digits). Every
file carries the SHA-256 of the previous session's file and of the captured
chain it used, so the chain of files is append-only by construction: editing
any past file breaks every later hash, and the git commit time of a file is
the time its targets were registered — before any later price existed.

Rules (the backtest's rules, on listed contracts):
    target      the sleeve's position for the session (signal_engine; S3's
                straddle is short when its signal is on, as in the backtest)
    contract    the PM-settled expiry whose calendar days to expiry are
                closest to STRADDLE_TENOR_DAYS (30), at least MIN_DAYS (14)
                away, and the listed strike nearest that expiry's parity
                forward with both legs two-sided (ties: lower strike)
    size        contracts per $1M = max(1, int(kelly·1e6 / (entry price·100))),
                short straddles capped at 1e6 / (0.20·K·100) (the engine's
                sizing and margin rule, at the executable entry price)
    hold        while the target keeps the same direction
    close       at the next captured book after the target changes or goes
                flat; a straddle with a one-sided or missing leg is not
                closed at a made-up price: it stays open and is retried
    settle      on or after its expiry date, at |S_close(expiry) − K|; if the
                target is still on, a new straddle is opened the same session
    hedge       optional bookkeeping of a delta hedge in the forward, set at
                each mark to the Black-76 straddle delta at the mid IVs, with
                the forward implied by the straddle's own legs
                (F = K + (C − P)/D): hedge P&L is reported separately
    model mark  with the session's VIX term structure, the backtest's own
                β·VIX straddle value of the same contract is recorded beside
                the market mid, so the mark-to-model error accumulates too
    timing      the signals use the session's 16:00 ET close, so the book they
                trade on must be quoted after it (the bot's Global Trading
                Hours capture is); a ``quote_time`` before the close is refused
                rather than booked at a price that predates the signal

Commissions ($0.65 per contract per leg, the backtest's rate) are recorded
separately; *_exec P&L is before commission, *_net after.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR

SCHEMA = "forward_ledger/1"
FORWARD_DIR = DATA_DIR / "forward"
TARGET_DAYS = 30
MIN_DAYS = 14
CONTRACT_MULT = 100
COMMISSION_PER_LEG = 0.65
MARGIN_RATIO = 0.20
NAV_PER_SLEEVE = 1_000_000.0
SLEEVES: tuple[str, ...] = ("s1", "s1c", "s2", "s3", "s4")
# straddle direction per unit of signal position (S3 on = long dispersion = short index straddle)
STRADDLE_SIGN: dict[str, int] = {"s1": 1, "s1c": 1, "s2": 1, "s3": -1, "s4": 1}
_SIG = 10   # significant digits kept for floats


# ── deterministic JSON ────────────────────────────────────────────────────────

def _clean(obj: Any) -> Any:
    """Floats to _SIG significant digits (NaN/inf → None), numpy → Python,
    tuples → lists, keys → str; recursively."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{_SIG}g}")
    if isinstance(obj, (pd.Timestamp, date)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    return obj


def canonical_json(obj: Any) -> str:
    """The canonical serialisation the hashes are taken over."""
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ledger_sha256(ledger: Mapping) -> str:
    return sha256_text(canonical_json(ledger))


def chain_sha256(book: pd.DataFrame) -> str:
    """SHA-256 of the captured two-sided book, independent of file format:
    CSV of (expiry, strike, right, root, bid, ask) sorted by contract."""
    cols = ["expiry", "strike", "right", "root", "bid", "ask"]
    b = book[cols].copy()
    b["root"] = b["root"].fillna("").astype(str)
    b = b.sort_values(["expiry", "right", "strike", "root"]).reset_index(drop=True)
    return sha256_text(b.to_csv(index=False, float_format="%.6f", lineterminator="\n"))


# ── the captured book ─────────────────────────────────────────────────────────

def observed_book(raw: pd.DataFrame) -> pd.DataFrame:
    """Every two-sided line of a raw or stored chain, one settlement style per
    expiry (option_chain.observed_quotes; the same rows the bot stores)."""
    from joint_vol_calibration.data import option_chain as oc

    book = oc.observed_quotes(raw)
    if "root" not in book.columns:
        book["root"] = ""
    book["root"] = book["root"].fillna("").astype(str)
    if "settlement" not in book.columns:
        book["settlement"] = "PM"
    return book.reset_index(drop=True)


def _b76(F: float, K: float, T: float, D: float, sig: float) -> dict[str, float]:
    from scipy.stats import norm
    sq = sig * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sq * sq) / sq
    nd1 = norm.pdf(d1)
    return {"delta_F": D * (2.0 * norm.cdf(d1) - 1.0),   # ∂(C+P)/∂F
            "vega": 2.0 * D * F * nd1 * math.sqrt(T)}


def _leg(book: pd.DataFrame, expiry: str, strike: float, right: str, root: str) -> dict | None:
    m = book[(book["expiry"] == expiry) & (np.isclose(book["strike"], strike)) &
             (book["right"] == right) & (book["root"] == root)]
    if m.empty:
        return None
    row = m.iloc[0]
    bid, ask = float(row["bid"]), float(row["ask"])
    if not (bid > 0 and ask >= bid):
        return None
    return {"bid": bid, "ask": ask, "mid": 0.5 * (bid + ask)}


def _discount(T: float, r_ref: float | None, carry: pd.DataFrame | None, expiry: str) -> float:
    if carry is not None and len(carry) and expiry in set(carry["expiry"]):
        return float(carry.set_index("expiry").loc[expiry, "discount"])
    return math.exp(-(r_ref if r_ref is not None and math.isfinite(r_ref) else 0.0) * T)


def _straddle_quote(book: pd.DataFrame, expiry: str, strike: float, root: str, T: float,
                    D: float) -> dict | None:
    """Both legs of a straddle with the implied forward, mid IVs and delta."""
    from joint_vol_calibration.models.implied_vol import implied_vol_black

    c = _leg(book, expiry, strike, "C", root)
    p = _leg(book, expiry, strike, "P", root)
    if c is None or p is None:
        return None
    F = strike + (c["mid"] - p["mid"]) / D
    out = {"call": c, "put": p, "forward": F, "discount": D, "T": T,
           "bid": c["bid"] + p["bid"], "ask": c["ask"] + p["ask"], "mid": c["mid"] + p["mid"]}
    if T > 0 and F > 0:
        iv = implied_vol_black(np.array([c["mid"], p["mid"]]), F, strike, T,
                               np.array([True, False]), D)
        out["iv_call"], out["iv_put"] = float(iv[0]), float(iv[1])
        s = float(np.nanmean(iv)) if np.isfinite(iv).any() else float("nan")
        out["iv_mid"] = s
        if math.isfinite(s) and s > 0:
            out.update(_b76(F, strike, T, D, s))
    return out


# ── contract selection ────────────────────────────────────────────────────────

def _year_fraction(session: str, expiry: str, settlement: str, t_origin) -> float:
    from joint_vol_calibration.data.option_chain import year_fraction
    return float(year_fraction(session, expiry, settlement, origin=t_origin))


def select_straddle(book: pd.DataFrame, session: str, r_ref: float | None = None,
                    carry: pd.DataFrame | None = None, t_origin=None,
                    target_days: int = TARGET_DAYS, min_days: int = MIN_DAYS) -> dict | None:
    """The ATM-forward straddle on the PM expiry closest to ``target_days``."""
    sess = pd.Timestamp(session)
    pm = book[book["settlement"] == "PM"]
    exps = sorted(set(pm["expiry"]))
    cands = [(abs((pd.Timestamp(e) - sess).days - target_days), -(pd.Timestamp(e) - sess).days, e)
             for e in exps if (pd.Timestamp(e) - sess).days >= min_days]
    for _, _, expiry in sorted(cands):
        g = pm[pm["expiry"] == expiry]
        roots = sorted(set(g["root"]))
        root = "SPXW" if "SPXW" in roots else roots[0]
        g = g[g["root"] == root]
        ks = sorted(set(g[g["right"] == "C"]["strike"]) & set(g[g["right"] == "P"]["strike"]))
        if len(ks) < 2:
            continue
        T = _year_fraction(session, expiry, "PM", t_origin)
        D = _discount(T, r_ref, carry, expiry)
        # forward from the put-call parity of the near-spot strikes (median)
        c = g[g["right"] == "C"].set_index("strike")["mid_price"]
        p = g[g["right"] == "P"].set_index("strike")["mid_price"]
        Fk = (pd.Series(ks, index=ks) + (c.reindex(ks) - p.reindex(ks)) / D)
        Fk = Fk.dropna()
        if Fk.empty:
            continue
        near = np.argsort(np.abs(Fk.index.to_numpy(float) - float(Fk.median())))[:6]
        F0 = float(Fk.iloc[near].median())
        K = min(ks, key=lambda k: (abs(k - F0), k))
        q = _straddle_quote(book, expiry, K, root, T, D)
        if q is None:
            continue
        return {"expiry": expiry, "strike": float(K), "root": root, "settlement": "PM",
                "days_to_expiry": int((pd.Timestamp(expiry) - sess).days), "parity_forward": F0, **q}
    return None


# ── positions ─────────────────────────────────────────────────────────────────

def _contracts(kelly: float, price: float, K: float, direction: int) -> int:
    n = max(1, int(kelly * NAV_PER_SLEEVE / (price * CONTRACT_MULT)))
    if direction < 0:
        cap = int(NAV_PER_SLEEVE / (MARGIN_RATIO * K * CONTRACT_MULT))
        n = max(0, min(n, cap))
    return n


def open_position(sleeve: str, direction: int, kelly: float, quote: dict, session: str) -> dict:
    entry = quote["ask"] if direction > 0 else quote["bid"]
    n = _contracts(kelly, entry, quote["strike"], direction)
    pid = f"{sleeve}:{session}:{quote['root']}:{quote['expiry']}:{quote['strike']:g}"
    hedge_units = -direction * quote.get("delta_F", float("nan"))
    return {
        "id": pid, "sleeve": sleeve, "direction": int(direction), "kelly": float(kelly),
        "contracts_per_1m": int(n),
        "contract": {"expiry": quote["expiry"], "strike": quote["strike"], "root": quote["root"],
                     "settlement": quote["settlement"]},
        "entry": {"session": session, "price_exec": entry, "price_mid": quote["mid"],
                  "bid": quote["bid"], "ask": quote["ask"],
                  "call": quote["call"], "put": quote["put"], "forward": quote["forward"],
                  "iv_mid": quote.get("iv_mid"), "delta_F": quote.get("delta_F"),
                  "days_to_expiry": quote["days_to_expiry"]},
        "mark": None,
        "hedge": {"units": hedge_units, "forward": quote["forward"], "pnl": 0.0},
        "sessions_marked": 0,
    }


def _pnl(direction: int, entry_exec: float, entry_mid: float, exit_exec: float | None,
         exit_mid: float | None, n: int) -> dict[str, float | None]:
    per_exec = None if exit_exec is None else direction * (exit_exec - entry_exec) * CONTRACT_MULT
    per_mid = None if exit_mid is None else direction * (exit_mid - entry_mid) * CONTRACT_MULT
    comm = 2 * COMMISSION_PER_LEG                       # one straddle, one side
    return {"pnl_exec_per_straddle": per_exec, "pnl_mid_per_straddle": per_mid,
            "pnl_exec": None if per_exec is None else per_exec * n,
            "pnl_mid": None if per_mid is None else per_mid * n,
            "commission": comm * n}


def mark_position(pos: dict, book: pd.DataFrame, session: str, r_ref: float | None = None,
                  carry: pd.DataFrame | None = None, t_origin=None,
                  vix_row: pd.Series | None = None, beta: float | None = None) -> dict:
    """The position marked on this session's book (a new dict; ``pos`` unchanged)."""
    pos = json.loads(canonical_json(pos))
    c = pos["contract"]
    T = _year_fraction(session, c["expiry"], c["settlement"], t_origin)
    D = _discount(T, r_ref, carry, c["expiry"])
    q = _straddle_quote(book, c["expiry"], c["strike"], c["root"], T, D) if T > 0 else None
    d, n = pos["direction"], pos["contracts_per_1m"]
    e = pos["entry"]
    if q is None:
        pos["mark"] = {"session": session, "status": "unmarked",
                       "reason": "expired" if T <= 0 else "a leg is missing or one-sided"}
        return pos
    exit_exec = q["bid"] if d > 0 else q["ask"]
    mark = {"session": session, "status": "marked", "bid": q["bid"], "ask": q["ask"], "mid": q["mid"],
            "call": q["call"], "put": q["put"], "forward": q["forward"], "T": T,
            "iv_mid": q.get("iv_mid"), "delta_F": q.get("delta_F"),
            "close_value_exec": exit_exec,
            **_pnl(d, e["price_exec"], e["price_mid"], exit_exec, q["mid"], n)}
    h = pos["hedge"]
    if h["units"] is not None and h["forward"] is not None:
        h["pnl"] = (float(h["pnl"])
                    + float(h["units"]) * (q["forward"] - float(h["forward"])) * CONTRACT_MULT * n)
        h["forward"] = q["forward"]
        if q.get("delta_F") is not None and math.isfinite(q["delta_F"]):
            h["units"] = -d * q["delta_F"]
    if vix_row is not None:
        from joint_vol_calibration.backtest import backtest_engine as bte
        from joint_vol_calibration.backtest.delta_hedger import _bs_straddle_value
        b = bte.ATM_TO_VIX_RATIO if beta is None else float(beta)
        S = q["forward"] * D * math.exp(bte.Q_BACKTEST * T)       # spot implied by F and D
        r = -math.log(D) / T if T > 0 else 0.0
        sig = bte._atm_sigma(vix_row, T, b)
        mark["model_value"] = float(_bs_straddle_value(S, c["strike"], T, r, bte.Q_BACKTEST, sig))
        mark["model_sigma"] = sig
        mark["model_minus_mid"] = mark["model_value"] - q["mid"]
    pos["mark"] = mark
    pos["sessions_marked"] = int(pos["sessions_marked"]) + 1
    return pos


def settle_position(pos: dict, session: str, settle_close: float) -> dict:
    """Settle at intrinsic on the official close of the (PM) expiry date."""
    pos = json.loads(canonical_json(pos))
    K = float(pos["contract"]["strike"])
    intrinsic = abs(float(settle_close) - K)
    e = pos["entry"]
    res = _pnl(pos["direction"], e["price_exec"], e["price_mid"], intrinsic, intrinsic,
               pos["contracts_per_1m"])
    res["commission"] = 0.0                              # no exit trade at settlement
    h = pos["hedge"]
    if h["units"] is not None and h["forward"] is not None:
        h["pnl"] = float(h["pnl"]) + float(h["units"]) * (float(settle_close) - float(h["forward"])) \
            * CONTRACT_MULT * pos["contracts_per_1m"]
        h["forward"] = float(settle_close)
        h["units"] = 0.0
    pos["mark"] = {"session": session, "status": "settled", "settle_close": float(settle_close),
                   "close_value_exec": intrinsic, "mid": intrinsic, **res}
    return pos


def _realised(pos: dict, how: str, session: str) -> dict:
    m = pos["mark"]
    entry_comm = 2 * COMMISSION_PER_LEG * pos["contracts_per_1m"]
    exit_comm = 0.0 if how == "settle" else 2 * COMMISSION_PER_LEG * pos["contracts_per_1m"]
    pnl_exec = m.get("pnl_exec")
    return {**pos, "closed": {"session": session, "how": how, "pnl_exec": pnl_exec,
                              "pnl_mid": m.get("pnl_mid"), "commission": entry_comm + exit_comm,
                              "pnl_net": None if pnl_exec is None else pnl_exec - entry_comm - exit_comm,
                              "hedge_pnl": pos["hedge"]["pnl"]}}


# ── one session ───────────────────────────────────────────────────────────────

def _summary_template() -> dict:
    return {"realised_pnl_exec": 0.0, "realised_pnl_mid": 0.0, "realised_pnl_net": 0.0,
            "realised_hedge_pnl": 0.0, "n_closed": 0, "n_opened": 0}


def quote_after_close(session: str, quote_time) -> bool | None:
    """True if ``quote_time`` (ISO string or datetime; naive = UTC) is at or
    after the session's 16:00 ET close; None when no quote time is given.
    Raises ValueError when it is earlier: the session's signals use that
    close, so trading them on an earlier book would use a price from before
    the information."""
    from datetime import datetime as _dt

    from joint_vol_calibration.data.option_chain import NY_TZ, SESSION_CLOSE_ET, _as_instant

    q = _as_instant(quote_time)
    if q is None:
        return None
    close = _dt.combine(pd.Timestamp(session).date(), SESSION_CLOSE_ET, tzinfo=NY_TZ)
    if q < close:
        raise ValueError(f"book quoted at {q.isoformat()} is before the {session} close "
                         f"({close.isoformat()}) that the session's signals use")
    return True


def step(prev: Mapping | None, session: str, raw_chain: pd.DataFrame,
         signals: Mapping[str, Mapping[str, float]], spot: float | None = None,
         r_ref: float | None = None, carry: pd.DataFrame | None = None,
         t_origin=None, quote_time: str | None = None,
         settlement_closes: Mapping[str, float] | None = None,
         vix_row: pd.Series | None = None, beta: float | None = None) -> dict:
    """The ledger record of ``session``.

    prev              : the previous session's ledger (None for the first)
    raw_chain         : this session's captured chain (raw or stored frame)
    signals           : {sleeve: {"position": −1/0/+1, "kelly": f}}
    settlement_closes : {"YYYY-MM-DD": official SPX close} for expiry dates
                        up to this session (to settle expired straddles)
    """
    session = str(pd.Timestamp(session).date())
    if prev is not None and str(prev["session"]) >= session:
        raise ValueError(f"ledger for {session} must follow {prev['session']}")
    after_close = quote_after_close(session, quote_time)
    book = observed_book(raw_chain)
    settle = {str(k): float(v) for k, v in (settlement_closes or {}).items()}
    open_prev = {p["sleeve"]: p for p in (prev or {}).get("positions", [])}
    summary = {}
    for s in SLEEVES:
        sm = dict(_summary_template())
        sm.update({k: v for k, v in (prev or {}).get("summary", {}).get(s, {}).items()
                   if k in sm})
        sm.update({"open_id": None, "open_pnl_exec": None, "open_pnl_mid": None})
        summary[s] = sm
    positions, closed, actions = [], [], []
    for s in SLEEVES:
        sig = signals.get(s, {"position": 0, "kelly": 0.0})
        target = int(np.sign(float(sig.get("position", 0)))) * STRADDLE_SIGN[s]
        kelly = float(sig.get("kelly", 0.0) or 0.0)
        pos = open_prev.get(s)
        just_closed = False
        settled_today = False
        if pos is not None:
            exp = pos["contract"]["expiry"]
            if session >= exp:
                if exp in settle:
                    pos = settle_position(pos, session, settle[exp])
                    closed.append(_realised(pos, "settle", session))
                    actions.append({"sleeve": s, "action": "settle", "id": pos["id"]})
                    settled_today, pos = True, None
                else:
                    pos = dict(pos, mark={"session": session, "status": "awaiting_settlement",
                                          "reason": f"no official close for {exp} supplied"})
                    actions.append({"sleeve": s, "action": "await_settlement", "id": pos["id"]})
            else:
                pos = mark_position(pos, book, session, r_ref, carry, t_origin, vix_row, beta)
                if target != pos["direction"]:
                    if pos["mark"]["status"] == "marked":
                        closed.append(_realised(pos, "close", session))
                        actions.append({"sleeve": s, "action": "close", "id": pos["id"]})
                        pos, just_closed = None, True
                    else:
                        actions.append({"sleeve": s, "action": "close_pending", "id": pos["id"]})
                else:
                    actions.append({"sleeve": s, "action": "hold", "id": pos["id"]})
        if pos is None and target != 0 and not just_closed:
            q = select_straddle(book, session, r_ref, carry, t_origin)
            if q is None:
                actions.append({"sleeve": s, "action": "no_contract"})
            else:
                newp = open_position(s, target, kelly, q, session)
                if newp["contracts_per_1m"] < 1:
                    actions.append({"sleeve": s, "action": "margin_blocked"})
                else:
                    pos = newp
                    summary[s]["n_opened"] += 1
                    actions.append({"sleeve": s, "action": "roll" if settled_today else "open",
                                    "id": pos["id"]})
        if pos is not None:
            positions.append(pos)
    for c in closed:
        sm = summary[c["sleeve"]]
        sm["n_closed"] += 1
        for k_from, k_to in (("pnl_exec", "realised_pnl_exec"), ("pnl_mid", "realised_pnl_mid"),
                             ("pnl_net", "realised_pnl_net"), ("hedge_pnl", "realised_hedge_pnl")):
            v = c["closed"][k_from]
            if v is not None:
                sm[k_to] = float(sm[k_to]) + float(v)
    for p in positions:
        sm = summary[p["sleeve"]]
        m = p.get("mark") or {}
        sm["open_pnl_exec"] = m.get("pnl_exec")
        sm["open_pnl_mid"] = m.get("pnl_mid")
        sm["open_id"] = p["id"]
    ledger = {
        "schema": SCHEMA,
        "session": session,
        "previous": None if prev is None else {"session": str(prev["session"]),
                                               "sha256": ledger_sha256(prev)},
        "inputs": {"chain_sha256": chain_sha256(book), "chain_rows_two_sided": len(book),
                   "quote_time": None if quote_time is None else str(quote_time),
                   "quote_after_close": after_close,
                   "t_origin": None if t_origin is None else str(t_origin),
                   "spot": spot, "r_ref": r_ref,
                   "beta": None if vix_row is None else (beta if beta is not None else _default_beta())},
        "signals": {s: {"position": int(np.sign(float(signals.get(s, {}).get("position", 0)))),
                        "kelly": float(signals.get(s, {}).get("kelly", 0.0) or 0.0),
                        "straddle_direction": int(np.sign(float(signals.get(s, {}).get("position", 0))))
                        * STRADDLE_SIGN[s]} for s in SLEEVES},
        "actions": actions,
        "positions": positions,
        "closed": closed,
        "summary": summary,
    }
    ledger = json.loads(canonical_json(ledger))
    validate_ledger(ledger)
    return ledger


def _default_beta() -> float:
    from joint_vol_calibration.backtest.backtest_engine import ATM_TO_VIX_RATIO
    return float(ATM_TO_VIX_RATIO)


# ── files ─────────────────────────────────────────────────────────────────────

def ledger_path(directory: Path, session: str) -> Path:
    return Path(directory) / f"ledger_{session}.json"


def write_ledger(ledger: Mapping, directory: Path = FORWARD_DIR) -> Path:
    """Write ledger_<session>.json (pretty-printed canonical JSON); refuses to
    overwrite a different file for the same session."""
    validate_ledger(ledger)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = ledger_path(directory, ledger["session"])
    text = json.dumps(_clean(ledger), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() and path.read_text() != text:
        raise FileExistsError(f"{path} exists with different content; a registered ledger is never rewritten")
    path.write_text(text)
    return path


def load_ledger(path: Path) -> dict:
    led = json.loads(Path(path).read_text())
    validate_ledger(led)
    return led


def latest_ledger(directory: Path = FORWARD_DIR, before: str | None = None) -> dict | None:
    """The newest ledger strictly before ``before`` (or the newest of all)."""
    files = sorted(Path(directory).glob("ledger_*.json"))
    if before is not None:
        files = [f for f in files if f.stem[len("ledger_"):] < str(before)]
    return load_ledger(files[-1]) if files else None


def verify_chain(ledgers: Sequence[Mapping]) -> list[str]:
    """Problems with a sequence of ledgers (ordered by session): every
    'previous' must name the prior file's session and hash."""
    problems = []
    for a, b in itertools.pairwise(ledgers):
        prev = b.get("previous") or {}
        if prev.get("session") != a["session"]:
            problems.append(f"{b['session']}: previous session {prev.get('session')} != {a['session']}")
        elif prev.get("sha256") != ledger_sha256(a):
            problems.append(f"{b['session']}: hash of {a['session']} does not match")
    return problems


# ── schema ────────────────────────────────────────────────────────────────────

_NUM = {"type": ["number", "null"]}
_LEG = {"type": "object", "required": ["bid", "ask", "mid"],
        "properties": {"bid": {"type": "number"}, "ask": {"type": "number"}, "mid": {"type": "number"}}}
LEDGER_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": SCHEMA,
    "type": "object",
    "required": ["schema", "session", "previous", "inputs", "signals", "actions", "positions",
                 "closed", "summary"],
    "properties": {
        "schema": {"const": SCHEMA},
        "session": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "previous": {"type": ["object", "null"], "required": ["session", "sha256"]},
        "inputs": {"type": "object", "required": ["chain_sha256", "chain_rows_two_sided"],
                   "properties": {"chain_sha256": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                                  "chain_rows_two_sided": {"type": "integer"}}},
        "signals": {"type": "object"},
        "actions": {"type": "array", "items": {
            "type": "object", "required": ["sleeve", "action"],
            "properties": {"action": {"enum": ["open", "roll", "hold", "close", "close_pending", "settle",
                                               "await_settlement", "no_contract", "margin_blocked"]}}}},
        "positions": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "sleeve", "direction", "kelly", "contracts_per_1m", "contract", "entry",
                         "mark", "hedge"],
            "properties": {
                "direction": {"enum": [-1, 1]},
                "contracts_per_1m": {"type": "integer"},
                "contract": {"type": "object", "required": ["expiry", "strike", "root", "settlement"]},
                "entry": {"type": "object", "required": ["session", "price_exec", "price_mid", "bid", "ask",
                                                         "call", "put"],
                          "properties": {"call": _LEG, "put": _LEG, "price_exec": {"type": "number"}}},
                "mark": {"type": ["object", "null"]},
                "hedge": {"type": "object", "required": ["units", "forward", "pnl"],
                          "properties": {"pnl": _NUM}},
            }}},
        "closed": {"type": "array", "items": {"type": "object", "required": ["id", "closed"]}},
        "summary": {"type": "object"},
    },
}


def _type_ok(value: Any, t: Any) -> bool:
    types = t if isinstance(t, list) else [t]
    for tt in types:
        if tt == "null" and value is None:
            return True
        if tt == "object" and isinstance(value, dict):
            return True
        if tt == "array" and isinstance(value, list):
            return True
        if tt == "string" and isinstance(value, str):
            return True
        if tt == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if tt == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if tt == "boolean" and isinstance(value, bool):
            return True
    return False


def _validate(value: Any, schema: Mapping, path: str, errors: list[str]) -> None:
    import re
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")
    if "type" in schema and not _type_ok(value, schema["type"]):
        errors.append(f"{path}: {type(value).__name__} is not {schema['type']}")
        return
    if "pattern" in schema and isinstance(value, str) and not re.match(schema["pattern"], value):
        errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, dict):
        errors.extend(f"{path}: missing {k!r}" for k in schema.get("required", []) if k not in value)
        for k, sub in schema.get("properties", {}).items():
            if k in value:
                _validate(value[k], sub, f"{path}.{k}", errors)
    if isinstance(value, list) and "items" in schema:
        for i, v in enumerate(value):
            _validate(v, schema["items"], f"{path}[{i}]", errors)


def validate_ledger(ledger: Mapping) -> None:
    """Raise ValueError listing every way ``ledger`` departs from LEDGER_SCHEMA
    (the subset of JSON Schema it uses: type, const, enum, pattern, required,
    properties, items)."""
    errors: list[str] = []
    _validate(ledger, LEDGER_SCHEMA, "$", errors)
    if errors:
        raise ValueError("invalid forward ledger: " + "; ".join(errors[:20]))


# ── inputs from the database (for the daily bot) ─────────────────────────────

def signals_for_session(session: str, start_date: str | None = None) -> dict[str, dict[str, float]]:
    """The backtest's signal positions and sizes for ``session``, built exactly
    as BacktestEngine.run builds them (rule regimes, walk-forward PDV-21 from
    2010, SignalEngine from the backtest start), on data dated ≤ session."""
    from joint_vol_calibration.backtest.backtest_engine import BT_START, BacktestEngine
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    from joint_vol_calibration.signals.regime_classifier import build_regime_labels, walk_forward_pdv_vol
    from joint_vol_calibration.signals.signal_engine import SignalEngine

    start = start_date or BT_START
    spx = get_spx_ohlcv(as_of_date=session, start_date="2010-01-01")
    vix = get_vix_term_structure_wide(as_of_date=session, start_date="2010-01-01")
    pdv = walk_forward_pdv_vol(spx)
    spx_col = BacktestEngine._ensure_log_return(
        BacktestEngine._to_df_with_date_col(BacktestEngine._to_date_indexed(spx)))
    vix_col = BacktestEngine._to_df_with_date_col(BacktestEngine._to_date_indexed(vix))
    regimes = build_regime_labels(spx_col, vix_col)
    sig = SignalEngine(clf=None).generate(spx_col, vix_col, start_date=start, end_date=session,
                                          precomputed_regimes=regimes, precomputed_pdv_vol=pdv)
    ts = pd.Timestamp(session)
    if ts not in sig.index:
        raise ValueError(f"no signal row for {session}")
    row = sig.loc[ts]
    return {s: {"position": float(row[f"{s}_position"]), "kelly": float(row[f"{s}_kelly"])}
            for s in SLEEVES}


def _session_inputs(session: str, earliest: str | None = None) -> dict[str, Any]:
    """SPX closes up to ``session`` (for settlement) and its VIX row, from the DB.

    The closes start 120 days before ``session`` or at ``earliest`` (the
    oldest expiry still open in the previous ledger) if that is older, so a
    straddle whose expiry fell in a run of sessions without a captured book
    still settles on its own expiry close.
    """
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide

    start = pd.Timestamp(session) - pd.Timedelta(days=120)
    if earliest is not None:
        start = min(start, pd.Timestamp(earliest))
    spx = get_spx_ohlcv(as_of_date=session, start_date=str(start.date()))
    closes = {str(pd.Timestamp(d).date()): float(c) for d, c in zip(spx["date"], spx["close"])}
    vw = get_vix_term_structure_wide(as_of_date=session, start_date=session)
    vix_row = None
    if len(vw):
        vw = vw.set_index(pd.to_datetime(vw["date"]))
        if pd.Timestamp(session) in vw.index:
            vix_row = vw.loc[pd.Timestamp(session)]
    return {"closes": closes, "vix_row": vix_row}


def record_session(session: str, raw_chain: pd.DataFrame, clean=None, spot: float | None = None,
                   r_ref: float | None = None, quote_time: str | None = None,
                   directory: Path = FORWARD_DIR, signals: Mapping | None = None) -> Path:
    """One call for the daily bot, after a session's chain is accepted.

    Builds the session's signals (signals_for_session) unless given, reads the
    previous ledger from ``directory``, the SPX closes (settlement) and the VIX
    row (model mark) from the database, takes the parity carry and T origin
    from ``clean`` (option_chain.CleanChain) when given, and writes
    ledger_<session>.json. Returns its path.
    """
    session = str(pd.Timestamp(session).date())
    prev = latest_ledger(directory, before=session)
    sig = signals if signals is not None else signals_for_session(session)
    open_expiries = [p["contract"]["expiry"] for p in (prev or {}).get("positions", [])]
    inp = _session_inputs(session, min(open_expiries) if open_expiries else None)
    carry, t0 = None, None
    if clean is not None and getattr(clean, "ok", False):
        carry = clean.carry
        diag = getattr(clean, "diagnostics", {}) or {}
        if str(diag.get("t_origin_basis", "")).startswith("quote_time"):
            t0 = diag.get("t_origin")
    led = step(prev, session, raw_chain, sig, spot=spot, r_ref=r_ref, carry=carry, t_origin=t0,
               quote_time=quote_time, settlement_closes=inp["closes"], vix_row=inp["vix_row"])
    return write_ledger(led, directory)
