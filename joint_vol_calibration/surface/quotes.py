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

Inputs are read through the repo's look-ahead-gated getters: the stored
chain from data_store/options/spx_<session>.parquet
(database.options_snapshot_path), the session's SPX close
(get_spx_ohlcv(as_of_date=session)) and the ^IRX bill rate
(get_tbill_rate(session), already continuously compounded).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import ndtr

from joint_vol_calibration.calibration import joint_calibrator as _jc
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
    """
    session: str
    spot: float
    r_ref: float
    quotes: pd.DataFrame
    carry: pd.DataFrame
    raw: pd.DataFrame
    clean_diagnostics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def expiries(self) -> list:
        return list(dict.fromkeys(self.quotes["expiry"]))


def read_stored_chain(session: str, underlying: str = "SPX") -> pd.DataFrame:
    """The committed chain data_store/options/<und>_<session>.parquet."""
    path = db.options_snapshot_path(session, underlying)
    if not path.exists():
        raise FileNotFoundError(f"no stored {underlying} chain for {session}: {path}")
    return pd.read_parquet(path)


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
                        r_ref: float | None = None) -> SessionQuotes:
    """Build the scoring universe of ``session`` from its stored chain.

    chain : a stored/raw chain frame to use instead of the committed parquet
    spot  : SPX close override (default: the stored close of ``session``)
    r_ref : reference rate override (default: ^IRX for ``session``, cc)

    Raises ValueError when the cleaned chain is unusable (the cleaner's
    fail-closed reason is passed through).
    """
    session = str(pd.Timestamp(session).date())
    raw = read_stored_chain(session) if chain is None else chain
    S = session_spot(session) if spot is None else float(spot)
    r = float(db.get_tbill_rate(session)) if r_ref is None else float(r_ref)
    clean = oc.clean_option_chain(raw, session, S, r_ref=r)
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
        "quote_time": clean.diagnostics.get("quote_time"),
        "source": (str(raw["source"].dropna().iloc[0])
                   if "source" in raw.columns and raw["source"].notna().any() else None),
        "universe_filters": {
            "otm_vs_parity_forward": True, "min_T_days": MIN_T_DAYS,
            "max_T_years": MAX_T_YEARS, "K_over_F": [MONEYNESS_LO, MONEYNESS_HI],
            "min_mid_price": MIN_PRICE, "mid_iv_range": [MIN_IV, MAX_IV],
            "note": ("identical to JointCalibrator.spx_surface_full; 0-6 DTE "
                     "expiries are excluded, as in the calibrator"),
        },
    }
    return SessionQuotes(session=session, spot=S, r_ref=r, quotes=quotes,
                         carry=clean.carry.copy(), raw=raw,
                         clean_diagnostics=dict(clean.diagnostics), diagnostics=diag)


def expiry_groups(quotes: pd.DataFrame):
    """Yield (expiry, T, F, D, frame) per expiry in T order (one T per expiry)."""
    for exp, g in quotes.groupby("expiry", sort=False):
        yield (exp, float(g["T"].iloc[0]), float(g["forward"].iloc[0]),
               float(g["discount"].iloc[0]), g)
