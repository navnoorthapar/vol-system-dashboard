"""
report.py — run the surface layer on one stored session and write its
artifacts under data_store/surface/<session>/.

Artifacts (schema_version 1, report_version REPORT_VERSION; floats rounded to 10 significant digits, keys
sorted, no wall-clock fields, so a rerun on the same inputs reproduces the
files byte for byte on the same machine):

residuals.parquet — one row per (model, quote) of the scoring universe:
    session, model, expiry, T_days, strike, right, k (= ln K/F), forward,
    bid, ask (option prices), bid_iv, mid_iv, ask_iv, model_iv (decimal),
    vega (discounted price per 1 vol point at mid IV), half_spread_vp,
    error_vp (model - mid, vol points), error_hs (error_vp / max(half
    spread, 0.01 vp)), inside_spread, outside_spread_vp (distance outside the
    nearer edge), in_sample (fitted to by that model), delta_bucket,
    tenor_bucket, arb_flag (the l1-BA repair moves the quote outside its
    spread), arb_exec (leg of an executable static-arbitrage row), arb_mid
    (leg of a row violated at mids), repair_dprice (LP price change, index
    points). Sorted by model order, then T, strike.

surface.json — inputs (which book and why: the session's newest stored
    Cboe capture when it passes the forward ledger's quote-state gates, else
    its Yahoo chain (quotes.select_session_chain); its path and SHA-256, the
    capture's payload SHA-256, the books refused and why, spot, rate, T
    origin, source, quote time (None for a Yahoo chain: it records none),
    the download instant fetched_at, book_state, the near-ATM two-sided share),
    universe counts, the static-arbitrage screen (row counts and
    violations per CRW row type, mid and executable; the bid / ask box
    certificate; the total-variance calendar diagnostic), the l1-BA repair
    summary with the largest violating rows and the quotes it moves outside
    their spread, and for each fitted surface (essvi, svi, svi_cal) its
    parameters, analytic no-arbitrage margins (and, for SVI surfaces, the
    butterfly and calendar certificates of the published parameters, which
    are written at full precision, and the arbitrage_free_certified gate),
    fit diagnostics, dense-grid checks and quote metrics; plus an ATM term
    structure (eSSVI ATM vol and skew, SVI-cal ATM vol, mid-quote ATM vol
    per quoted expiry, parity carry).

scorecard.json — every model of scorecard.MODEL_REGISTRY on the same quote
    universe: description, parameter count, in-sample size, metrics (all /
    in / out of sample (its own held-out quotes) / common_out_of_sample
    (the quotes no model was fitted to) / excluding arbitrage-flagged
    quotes, by expiry, tenor, delta and tenor x delta), dense-grid checks;
    rankings by the median out-of-sample error in half-spreads, on each
    model's own and on the common held-out quotes. The svi_cal note is
    written from surface.json's certificate fields.

vix_replication.json — official closes, the replicated VIX (Cboe rules;
    parity-forward variant; any-weekday bracket) and VIX9D/3M/6M analogues,
    each term with the rate it used (Cboe's CMT spline per term; rate: the
    Treasury par curve it read), the calculation time and its basis (a Cboe
    book's quote time, a closing book's close, or the chain's T origin), the
    instant the session's Cboe closes print (cboe_close_ny: 16:15 ET, 13:15
    on an NYSE early close), each index with a gap attribution (wing
    truncation measured on the SVI-cal surface, residual = official -
    replicated - truncation); the
    variance-swap strip by expiry (Cboe single-term formula on the stored
    quotes) with the model-free ATM vol and the ATM / VS ratio (the
    empirical beta), model variance swaps from the eSSVI, SVI and SVI-cal
    fits, the 30-day ATM / VIX ratio (model-free), the methodology
    deviations and the gap notes.

index.json (in data_store/surface/) — one summary row per session written.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.surface import arbitrage as arb
from joint_vol_calibration.surface import essvi as es
from joint_vol_calibration.surface import quotes as sqm
from joint_vol_calibration.surface import scorecard as sc
from joint_vol_calibration.surface import svi as sv
from joint_vol_calibration.surface import vix as vx
from joint_vol_calibration.surface._blas import single_thread_blas

SCHEMA_VERSION = 1
# The methodology version of a session's artifacts: report_version in
# surface.json, scorecard.json, vix_replication.json and the index row. Raise
# it with any change to what a committed session's artifacts would say. The
# bot's backfill (scripts/cloud_refresh.artifact_gaps) then rebuilds every
# committed session stored under an older version, MAX_BACKFILL_SESSIONS a
# run, so a method change reaches the whole record, not only new sessions.
#   1  (no field) to C23f: the session's Yahoo chain; a Yahoo download's run
#      clock stored as its quote time
#   2  plan 0.7: the session's Cboe book when it passes the quote-state
#      gates, else the Yahoo chain (no quote time; T from the close), and
#      the choice recorded in surface.json inputs
#   3  a VIX bracket outside Cboe's 23-37 day term window is a deviation
#      (indices.VIX.cboe_term_window); a session without an accepted
#      Heston fit gets a surface, its Heston row unavailable with a reason
#   4  plan 0.5: the VIX replication's rate is Cboe's, per term (the bounded
#      CMT spline on the Treasury par curve before the session, vix.cmt_rates),
#      not one 91-day rate for both terms; the rate departure is listed only
#      when that curve is missing or stale
#   5  plan 0.5: a session built from a Cboe book is calculated at the instant
#      its quotes stand at (the book's quote time; a closing book's close,
#      16:15 ET or 13:15 on an early close), not the chain's T origin,
#      recorded as calculation_time_basis; that is Cboe's own timing rule, so
#      it is no longer a listed departure, and its comparison with the Cboe
#      print (cboe_close_ny) is a gap note
#   6  plan 0.8: a session with a stored closing book (the close slot's, its
#      quotes the ones standing at the close the official VIX-family closes are
#      computed at) is built from it, not from its later overnight book, so its
#      replication gap is a same-instant comparison; a session built from an
#      overnight (Global Trading Hours) book says in its timing note that the
#      gap mixes the replication's error with the overnight move, how many
#      hours after the print the book was quoted, and why no closing book was
#      used (none stored, or it failed the gates: inputs.closing_books counts
#      the session's stored closing books; none stored because the close slot
#      did not store the one it fetched: inputs.closing_book_refusal, the
#      reason its run record gives)
#   7  the overnight note's reason when no closing book is stored states only
#      what the close slot's run records hold (inputs.closing_book_run): the
#      slot captured the book but it was not kept, it stored none because the
#      feed showed another book, or no run of it is recorded for the session
#      (it did not exist yet); version 6 gave one reason for all three, a
#      claim about the feed's hours that is not observed
REPORT_VERSION = 7
SURFACE_DIR = DATA_DIR / "surface"


def stored_report_version(doc: Any) -> int:
    """The report_version a stored artifact (a parsed JSON object) was written
    under; 1 for one that predates the field."""
    v = doc.get("report_version") if isinstance(doc, dict) else None
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else 1
SIG_DIGITS = 10
WIDE_K_GRID = np.linspace(-3.0, 2.0, 2001)      # eSSVI g check (its conditions are analytic)


# ── deterministic serialisation ───────────────────────────────────────────────

def round_sig(a, sig: int = SIG_DIGITS):
    """Round to ``sig`` significant digits (vectorised; NaN/inf/0 unchanged)."""
    x = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
    ok = np.isfinite(x) & (x != 0) & np.isfinite(mag)
    f = np.where(ok, 10.0 ** (sig - 1 - np.where(ok, mag, 0)), 1.0)
    return np.where(ok, np.round(x * f) / f, x)


class _Exact(float):
    """A float written at full precision (17 significant digits, repr) rather
    than SIG_DIGITS: the SVI slice parameters, so the published surface is the
    fitted one bit for bit and its no-arbitrage certificates recompute from
    the JSON exactly (10-digit rounding moves g by up to ~1e-10)."""


def _exact_svi_params(d: dict) -> dict:
    """Copy of an SVISurface.to_dict() with the slice parameters as _Exact."""
    out = dict(d)
    out["slices"] = [{**sl, **{n: _Exact(sl[n]) for n in ("T", "a", "b", "rho", "m", "sigma")}}
                     for sl in d.get("slices", [])]
    return out


def _clean(obj):
    """JSON-safe copy: numpy scalars to Python, floats rounded (_Exact kept
    whole), NaN/inf to None."""
    if isinstance(obj, _Exact):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        return float(round_sig(v))
    if isinstance(obj, (np.str_,)):
        return str(obj)
    return obj


def dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), indent=1, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _frame_rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = round_sig(out[c].to_numpy(float))
    return out


# residuals.parquet is the largest file a book session adds to the repository
# (PROD-6). Its float columns are written BYTE_STREAM_SPLIT (Parquet's lossless
# float encoding: the bytes of each value are regrouped by significance), the
# other columns dictionary-encoded, each column in one data page, zstd level
# 19. One page matters: the quote columns (k, bid_iv, vega, ...) repeat once
# per model, and pyarrow's default 20,000-row pages hide the repeats from zstd.
# Every value is bit for bit what it was; the five committed files are 34-38%
# smaller (2026-06-18: 1.72 MB -> 1.06 MB). Readers need nothing new.
RESIDUALS_ZSTD_LEVEL = 19
RESIDUALS_ROWS_PER_PAGE = 1 << 20


def _parquet_writer_accepts(name: str) -> bool:
    import inspect

    import pyarrow.parquet as pq
    return name in inspect.signature(pq.ParquetWriter.__init__).parameters


def write_residuals(df: pd.DataFrame, path: Path) -> None:
    """Write a (rounded) residuals frame as residuals.parquet, losslessly."""
    floats = [str(c) for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    others = [str(c) for c in df.columns if str(c) not in floats]
    kw: dict[str, Any] = {}
    if _parquet_writer_accepts("max_rows_per_page"):      # pyarrow >= 20; older ones page by 20,000 rows
        kw["max_rows_per_page"] = RESIDUALS_ROWS_PER_PAGE
    df.to_parquet(path, index=False, compression="zstd", compression_level=RESIDUALS_ZSTD_LEVEL,
                  use_byte_stream_split=floats, use_dictionary=others, **kw)


# ── per-session computation ───────────────────────────────────────────────────

# vix_replication.json calculation_time_basis: what the replication's
# calculation time is (_calc_time_ny)
CALC_BOOK_QUOTE_TIME = "book_quote_time"
CALC_CLOSING_BOOK_CLOSE = "closing_book_close"
CALC_T_ORIGIN = "chain_t_origin"


def _calc_time_ny(sq: sqm.SessionQuotes) -> tuple:
    """(calculation time, basis): the VIX replication's calculation time (New
    York, naive) and which of CALC_* it is.

    Cboe measures each term's T from the time of the calculation, on the
    quotes standing at that instant. A session built from a Cboe book is
    calculated at the instant its quotes stand at: the book's quote time (the
    delayed feed's payload timestamp less 15 minutes), when the book is live
    then (a Global Trading Hours book trading for the next session, or one
    quoted before the session's data cutoff, 16:15 ET or 13:15 on an early
    close); for a closing book (quoted after the cutoff and outside the next
    Global Trading Hours session: weekend and holiday runs) the session's
    cutoff (database.close_data_cutoff), when its quotes were left and the
    official close is computed, not the later instant the feed showed them
    at. Not the chain's T origin, which for a closing book is the 16:00 ET
    close the surface measures T from. A chain that records no quote time
    (Yahoo) is calculated at its T origin, the session's 16:00 ET close."""
    from joint_vol_calibration.data import cboe_downloader as cd
    qt = sq.clean_diagnostics.get("quote_time")
    if sq.inputs.get("source") == "cboe" and qt:
        at = datetime.fromisoformat(str(qt))
        if cd.book_state_at(at, sq.session) == "closing":
            return (datetime.combine(date.fromisoformat(sq.session), db.close_data_cutoff(sq.session)),
                    CALC_CLOSING_BOOK_CLOSE)
        return at.astimezone(oc.NY_TZ).replace(tzinfo=None), CALC_BOOK_QUOTE_TIME
    t0 = datetime.fromisoformat(sq.clean_diagnostics["t_origin"])
    return t0.astimezone(oc.NY_TZ).replace(tzinfo=None), CALC_T_ORIGIN


def _surface_section(model_row: dict, surface) -> dict:
    sec = {"description": model_row["description"], "params": model_row["params"],
           "fit": model_row["fit"], "dense_grid": model_row["dense_grid"],
           "metrics": {k: model_row["metrics"][k] for k in
                       ("all", "in_sample", "out_of_sample", "excluding_arbitrage_flagged")
                       if k in model_row["metrics"]}}
    if isinstance(surface, es.ESSVISurface):
        sec["analytic_conditions"] = surface.conditions()
        gmin = min(float(surface.g(WIDE_K_GRID, T).min()) for T in surface.T)
        sec["analytic_g_min_k_-3_to_2"] = gmin
    elif isinstance(surface, sv.SVISurface):
        # the certificates of the PUBLISHED parameters (written whole, _Exact):
        # the same functions the fit accepts a slice with, so the label and
        # the artifact cannot disagree
        pub = sv.SVISurface.from_dict(model_row["params"])
        cert = pub.certificate()
        b = cert["butterfly"]
        lee = [max(s.params[1] * (1 + s.params[2]), s.params[1] * (1 - s.params[2]))
               for s in pub.slices]
        sec["analytic_g_min_on_fit_rows_k_-3_to_2"] = float(min(
            float(sv.svi_g(sv.SVI_GRID, s.params).min()) for s in pub.slices))
        sec["analytic_g_min_k_-3_to_2"] = b["g_min_k_-3_to_2"]
        sec["slices_with_g_negative_on_k_-3_to_2"] = b["slices_with_g_negative_on_k_-3_to_2"]
        sec["butterfly_certificate"] = b
        sec["max_wing_slope_b_1_plus_abs_rho"] = float(max(lee))
        sec["calendar_ordering"] = cert["calendar"]
        sec["butterfly_free_certified"] = bool(b["certified"])
        sec["arbitrage_free_certified"] = bool(cert["arbitrage_free"])
    return sec


def _svi_cal_note(sec: dict | None) -> str:
    """The scorecard's svi_cal label, written from the certificate fields of
    surface.json (the acceptance gate), never asserted independently."""
    tail = ("svi has no calendar constraint and its calendar violations are counted in "
            "dense_grid.")
    if not sec:
        return "svi_cal is not available on this session. " + tail
    b, c = sec["butterfly_certificate"], sec["calendar_ordering"]
    if sec["arbitrage_free_certified"]:
        return ("svi_cal is the arbitrage-free surface on this session: every slice passes "
                "svi.butterfly_certificate (Durrleman g >= 0 on a 1e-4 grid over k in [-3, 2] "
                "and a 1/k grid out to |k| = 2,000-3,000, each local minimum refined, and the "
                f"|k| -> inf limit; minimum g {b['g_min']:.2e}) and every consecutive pair passes "
                "svi.ordering_certificate (total variance ordered to 1e-10 on the same grids, "
                "wing slopes and tied-wing intercepts included; minimum w_(i+1) - w_i "
                f"{c['min_dw']:.1e}). Numerical certificates, not closed-form; fields "
                "surface.json surfaces.svi_cal.butterfly_certificate / calendar_ordering. " + tail)
    return (f"svi_cal is NOT certified arbitrage-free on this session: "
            f"{b['slices_with_g_negative']} slice(s) fail svi.butterfly_certificate and "
            f"{len(c.get('pairs_unordered') or [])} consecutive pair(s) fail "
            "svi.ordering_certificate (surface.json surfaces.svi_cal). " + tail)


def _repaired_quotes(q: pd.DataFrame, flags: pd.DataFrame, n: int = 25) -> dict:
    """The quotes the l1-BA repair moves outside their spread: counts per
    expiry and the ``n`` largest moves (option-price units)."""
    out = flags["repair_outside_spread"].to_numpy(bool) if "repair_outside_spread" in flags else \
        np.zeros(len(q), dtype=bool)
    d = q.loc[out, ["expiry", "T_days", "strike", "right", "bid", "ask", "mid", "mid_iv",
                    "half_spread_vp"]].copy()
    d["repair_dprice"] = flags.loc[out, "repair_dprice"].to_numpy(float)
    half = 0.5 * (d["ask"] - d["bid"])
    d["repair_dprice_half_spreads"] = np.where(half > 0, d["repair_dprice"].abs() / half, np.nan)
    d = d.reindex(d["repair_dprice"].abs().sort_values(ascending=False, kind="mergesort").index)
    per_exp = {str(e): int(c) for e, c in q.loc[out, "expiry"].value_counts(sort=False).items()}
    return {"n": int(out.sum()), "by_expiry": dict(sorted(per_exp.items())),
            "largest": d.head(n).to_dict(orient="records")}


def _quote_source_note(sq: sqm.SessionQuotes) -> str:
    """The scorecard's line on what the quotes are, from the chain's own record."""
    d, inp = sq.diagnostics, sq.inputs
    src = d.get("source")
    if src == "cboe":
        return (f"Quotes are Cboe's delayed SPX book ({inp.get('chain')}) as of {d.get('quote_time')} "
                f"({inp.get('book_state') or 'book state not recorded'}): every model is scored "
                "against the exchange's own bid and ask at one instant.")
    if not oc.records_quote_time(src):
        return ("Quotes are the session's Yahoo chain, which records no quote time"
                + (f" (downloaded {inp['fetched_at']})" if inp.get("fetched_at") else "")
                + ": part of every model's spread misses can be stale quotes rather than model error.")
    return (f"Quotes are from {src or 'an unrecorded source'}"
            + (f", as of {d['quote_time']}." if d.get("quote_time") else ", with no quote time."))


def _vix_term_window(rep: dict, terms: list, calc: datetime) -> dict | None:
    """vx.cboe_term_window of the replication the VIX headline uses (its
    headline_method; cboe_rules when none is headline, so an extrapolated
    pair is still checked)."""
    return vx.cboe_term_window(rep.get(rep.get("headline_method") or "cboe_rules") or {}, terms, calc)


def _heston_note(models: list) -> str:
    """The scorecard's line on the Heston row: its in- and out-of-sample sets,
    or why the session has none (the surface does not wait for a fit)."""
    row = next((m for m in models if str(m.get("model", "")).startswith("heston")), None)
    if row is not None and row.get("available"):
        return ("Heston's in-sample set is its calibration anchors, fitted jointly with the VIX "
                "index leg, and its out-of-sample set is every other quote.")
    if row is not None:
        return f"Heston is not scored on this session: {row.get('reason') or 'no stored fit'}"
    return "Heston is not in this scorecard."


# vix_replication.json rate.method: Cboe's rate per term, or the fallback
CMT_RATE_METHOD = "cboe_cmt_bounded_spline"


def _vix_rate(sq: sqm.SessionQuotes) -> tuple:
    """(rate, rate record, deviation line) of the session's VIX replication.

    Cboe's rate per term (vx.cmt_rates: its bounded spline on the Treasury
    par curve before the session) and no deviation; when that curve is
    missing or stale, the session's reference rate for both terms (the ^IRX
    bill, or the zero curve's 91-day rate) and a deviation saying why."""
    try:
        cmt = vx.cmt_rates(sq.session)
    except (LookupError, db.StaleDataError) as e:
        src = ("the ^IRX bill rate" if sq.r_ref_source == sqm.IRX_SOURCE else
               "the Treasury zero curve's 91-day rate")
        return sq.r_ref, {"method": "one_rate_for_both_terms", "R_cc": sq.r_ref,
                          "source": sq.r_ref_source, "cmt_unavailable": str(e)}, (
            f"The rate is {src} for both terms, not Cboe's spline on the Treasury CMT yields at "
            f"each term's expiration: {e}. A rate 25 bp off moves the 30-day index by about "
            "0.002 points.")
    return cmt, {"method": CMT_RATE_METHOD, **cmt.record(),
                 "source": (f"The session close's curve: the U.S. Treasury par yield curve (CMT) "
                            f"of {cmt.curve_date}, the newest before the session, the one Cboe "
                            "computes the session's close with, at Cboe's tenors. Each term's rate "
                            "is Cboe's bounded natural cubic spline at its expiration, in calendar "
                            "days counted from that curve's date as Cboe's published sample counts "
                            "them, the bond-equivalent yield taken to an annual percentage yield "
                            "and then to continuous compounding (Cboe Volatility Index Mathematics "
                            "Methodology sec. 2.1). A book timed after the close (in Global Trading "
                            "Hours) keeps this curve and this day count, so each term's rate is "
                            "the one the Cboe close it is compared with used; each term records "
                            "the rate it used")}, None


def _calc_time_record(sq: sqm.SessionQuotes, calc: datetime, basis: str) -> str:
    """vix_replication.json calculation_time_source: what the calculation time
    is, in words, from the chain's record (_calc_time_ny)."""
    d, inp = sq.clean_diagnostics, sq.inputs
    if basis == CALC_BOOK_QUOTE_TIME:
        return (f"the Cboe book's quote time {d.get('quote_time')} (the delayed feed's payload "
                f"timestamp less 15 minutes; {inp.get('book_state') or 'book state not recorded'} "
                "book, live at that instant)")
    if basis == CALC_CLOSING_BOOK_CLOSE:
        return (f"the session's {calc:%H:%M} ET close{_early(sq.session)}: the Cboe book is its "
                f"closing book, quoted at {d.get('quote_time')} after the close, its quotes the ones "
                "the close left")
    return f"the chain's T origin ({d.get('t_origin_basis') or 'the session close'})"


def _cboe_close_ny(session: str) -> str:
    """vix_replication.json cboe_close_ny: when the session's Cboe closes print
    (New York, to the minute), its data cutoff: 16:15 ET, 13:15 on an early
    close (database.close_data_cutoff)."""
    return datetime.combine(date.fromisoformat(session),
                            db.close_data_cutoff(session)).isoformat(timespec="minutes")


def _early(session: str) -> str:
    """' (an early close)' for a session that closes early, else ''."""
    return " (an early close)" if db.is_early_close(session) else ""


# why a session built from its overnight book has no closing book, when it holds
# none and no close-slot run refused one: what its close-slot run records hold, as
# quotes.closing_book_run names them
CLOSE_RUN_WHY = {
    sqm.CLOSE_RUN_CAPTURED: (" (the close slot captured the session's closing book, but the store holds "
                             "none: it was not kept)"),
    sqm.CLOSE_RUN_SKIPPED: (" (the close slot's runs stored no closing book: by the clock, or by the quote "
                            "time of the book downloaded, the delayed feed showed another book)"),
    sqm.CLOSE_RUN_NONE: " (no run of the close slot is recorded for the session)",
}


def _timing_note(sq: sqm.SessionQuotes, calc: datetime, basis: str) -> str | None:
    """The gap note on a Cboe book's timing: the replication follows Cboe's
    rule at the book's instant, and the official close it is set against is
    the session's print at its data cutoff, 16:15 ET or 13:15 on an early
    close (None for a chain with no quote time: that is a departure,
    _timing_deviation)."""
    if basis == CALC_BOOK_QUOTE_TIME and sq.inputs.get("book_state") == "gth":
        cutoff = db.close_data_cutoff(sq.session)
        hours = (calc - datetime.combine(date.fromisoformat(sq.session), cutoff)).total_seconds() / 3600.0
        refusal = str(sq.inputs.get("closing_book_refusal") or "").rstrip(".")
        why = (" (the session's closing book did not pass the quote-state gates)"
               if sq.inputs.get("closing_books") else
               # the close slot fetched it and its run record says why it was not stored
               f" (the job did not store the session's closing book: {refusal})" if refusal else
               # what else the close slot's run records hold
               CLOSE_RUN_WHY.get(str(sq.inputs.get("closing_book_run")), CLOSE_RUN_WHY[sqm.CLOSE_RUN_NONE]))
        return (f"The book is the overnight (Global Trading Hours) book{why}, quoted at "
                f"{calc:%Y-%m-%d %H:%M} ET, {hours:.1f} hours after the {cutoff:%H:%M} ET print of "
                f"{sq.session}{_early(sq.session)} it is compared with: the gap mixes the "
                "replication's error with the market's move overnight. The replication is timed "
                "at the book's quote time, as Cboe times a calculation on the quotes standing then.")
    if basis == CALC_BOOK_QUOTE_TIME:
        return (f"The replication is timed at the book's quote time, {calc:%Y-%m-%d %H:%M} ET, as "
                "Cboe times a calculation on the quotes standing then; the Cboe close it is "
                f"compared with is the {db.close_data_cutoff(sq.session):%H:%M} ET print of "
                f"{sq.session}{_early(sq.session)}, so the gap includes the market's move between "
                "the two instants.")
    if basis == CALC_CLOSING_BOOK_CLOSE:
        quoted = datetime.fromisoformat(str(sq.clean_diagnostics.get("quote_time"))).astimezone(oc.NY_TZ)
        return (f"The book is the session's closing book, quoted at {quoted:%Y-%m-%d %H:%M} ET after "
                f"the close: the replication is timed at the {calc:%H:%M} ET close of {sq.session}"
                f"{_early(sq.session)}, when its quotes were left, the instant the Cboe close it is "
                "compared with is computed.")
    return None


def _timing_deviation(sq: sqm.SessionQuotes, calc: datetime) -> str:
    """The calculation-time departure of a chain that is not a Cboe book, from
    its T origin and record (a Cboe book is calculated at its own instant,
    _calc_time_ny, and departs from nothing)."""
    d, inp = sq.clean_diagnostics, sq.inputs
    if str(d.get("t_origin_basis", "")).startswith("quote_time"):
        return (f"Calculation time is the book's quote time, {calc:%Y-%m-%d %H:%M} ET "
                f"({inp.get('book_state') or 'GTH'} book; Cboe's delayed feed, payload timestamp "
                "minus 15 minutes), not the 16:15 ET close the official index is compared with: "
                "the gap mixes the replication with the market's move between the two instants.")
    if d.get("quote_time"):
        return (f"Calculation time is the session's 16:00 ET close; the book was quoted at "
                f"{d['quote_time']} (its closing book) and the official close is the 16:15 ET print.")
    return ("Calculation time is the session's 16:00 ET close: the chain records no quote time "
            "(Yahoo)" + (f"; it was downloaded {inp['fetched_at']}" if inp.get("fetched_at") else "")
            + ". The official close is the 16:15 ET print.")


def _settlement_deviations(terms: list) -> list:
    """Deviation lines for the settlement each stored term was given."""
    out = []
    proxy = [t["expiry"] for t in terms if t.get("am_date_pm_proxy")]
    if proxy:
        out.append(f"SPXW PM lines stand in for the AM SPX monthly on {', '.join(proxy)} (the "
                   "chain stores no SPX-root lines there); their T runs to the 16:00 PM "
                   "settlement, 6.5 h past the 09:30 SOQ the AM series settles on.")
    rootless = [t for t in terms if not t.get("has_root")]
    if rootless:
        am = [t["expiry"] for t in rootless if vx.is_spx_am_settlement_date(t["expiry"])]
        out.append("Root-less (pre-C22) chain: every line is treated as PM-settled"
                   + (f"; on the AM monthly dates {', '.join(am)} T is 6.5 h longer than the AM "
                      "series'." if am else "."))
    return out


def _headline_at_cboe_rate(rep: dict) -> bool:
    """Whether both terms of an index's headline replication were discounted at
    the rate _vix_rate gave: each took Cboe's single-strike forward. A
    parity_forward term discounts at its parity regression's D (R = -ln D / T),
    not at Cboe's rate, so an index replicated on it can owe part of its gap to
    the rate."""
    head = rep.get(rep.get("headline_method") or "") or {}
    return all((head.get(t) or {}).get("forward_method") == "cboe_single_strike"
               for t in ("near", "next"))


def _gap_notes(sq: sqm.SessionQuotes, terms: list, vixrep: dict, official: dict,
               svi_surface, essvi_surface: es.ESSVISurface, calc: datetime,
               svi_cal_surface=None, attribution: dict | None = None,
               indices: dict | None = None, cboe_rate: bool = False,
               calc_basis: str = CALC_T_ORIGIN) -> list:
    """The VIX replication's gap notes. ``cboe_rate``: the VIX's two headline
    terms were discounted at Cboe's rate (_vix_rate's spline and
    _headline_at_cboe_rate), so the rate is not named among the causes of the
    unexplained part of its gap; ``calc_basis``: _calc_time_ny's basis, a
    closing book's leaving out the snapshot's timing."""
    notes = []
    for name, rep in (indices or {}).items():
        if rep.get("headline_method") is None:
            notes.append(f"{name} is not replicated from the stored chain: "
                         f"{rep.get('not_replicated_reason')}.")
    raw = sq.raw
    n_zero = int(((pd.to_numeric(raw["bid"], errors="coerce") <= 0)).sum()) if "bid" in raw else 0
    if n_zero == 0:
        notes.append("The stored chain holds no zero-bid lines (two-sided capture), so Cboe's "
                     "two-consecutive-zero-bid stop cannot trigger: every stored OTM strike with "
                     "a bid enters the strip, which can only add variance relative to Cboe's rule "
                     "on the same book; the stored book's own strike range is the binding limit.")
    else:
        notes.append(f"The stored chain keeps {n_zero} zero-bid lines; the zero-bid stop rule "
                     "is applied as published.")
    timing = _timing_note(sq, calc, calc_basis)
    if timing:
        notes.append(timing)
    if not sq.clean_diagnostics.get("quote_time"):
        o = official.get("VIX", {})
        rng = (f" (the official index traded {o['low']:.2f}-{o['high']:.2f} that session)"
               if "low" in o else "")
        notes.append("No quote time is stored for this snapshot (Yahoo); the calculation uses "
                     f"{calc:%H:%M} ET on the session and the official close is the 16:15 ET "
                     f"print{rng}. Any intraday drift between the snapshot and 16:15 is "
                     "unmeasured.")
    v = vixrep.get("cboe_rules", {})
    if v.get("ok"):
        # a bracket outside Cboe's 23-37 day window is a deviation
        # (vx.term_window_deviation), not a note
        near, nxt = v["near"], v["next"]
        for lab, t in (("near", near), ("next", nxt)):
            notes.append(f"{lab}-term {t['expiry']}: {t['n_puts']} puts / {t['n_calls']} calls, "
                         f"strip K/F {t['lowest_K_over_F']:.3f}-{t['highest_K_over_F']:.3f}"
                         + ("; K0 fell back to the nearest lower strike with both sides quoted"
                            if t["k0_fallback"] else ""))
    T30 = 30.0 / 365.0
    parts = []
    for name, surf in (("SVI-cal", svi_cal_surface), ("SVI", svi_surface), ("eSSVI", essvi_surface)):
        if surf is not None:
            vs = 100.0 * math.sqrt(max(vx.surface_variance_swap(surf, T30), 0.0))
            parts.append(f"{name} {vs:.2f}")
    if parts:
        notes.append("Model variance swaps at 30 days (k in [-3, 2], wings beyond the fitted "
                     "K/F 0.70-1.30 extrapolated, so each inherits its model's wing bias): "
                     + ", ".join(parts) + ".")
    a = attribution or {}
    if a.get("available") and "residual_pts" in a:
        fw = a["far_wing_market_minus_model_pts"]
        if abs(fw) < 0.05:
            wing = "about as rich as that wing"
        elif fw > 0:
            wing = (f"{fw:.2f} index points richer than that wing, so the truncation figure "
                    "likely understates")
        else:
            wing = (f"{-fw:.2f} index points cheaper than that wing, so the truncation figure "
                    "likely overstates")
        notes.append(f"Gap attribution (VIX, SVI-cal surface): replicated - official = "
                     f"{a['gap_pts']:+.2f}. Strikes beyond the stored strip would add "
                     f"{a['truncation_pts']:+.2f} under the surface's wing; the stored quotes "
                     f"beyond the fitted K/F 0.70-1.30 are {wing}. Inside the fitted range "
                     f"market and surface differ by {a['fit_inside_range_pts']:+.2f} (these "
                     f"two terms set the replication against the surface; they are not part "
                     f"of the gap). Left unexplained by the chain: {a['residual_pts']:+.2f} "
                     f"(official - replicated - truncation"
                     + (f"; {100.0 * a['residual_share_of_official_minus_replicated']:.0f}% of "
                        "official - replicated"
                        if a.get("residual_share_of_official_minus_replicated") is not None else "")
                     + "): " + ", ".join(
                         (["snapshot timing vs the 16:15 ET print"]
                          if calc_basis != CALC_CLOSING_BOOK_CLOSE else [])
                         + ["stale quotes"] + ([] if cboe_rate else ["the rate"])) + ".")
    return notes


def drop_unfittable_expiries(sq: sqm.SessionQuotes, max_rounds: int = 3) -> tuple:
    """Screen ``sq.quotes`` for static arbitrage and drop every expiry the
    surfaces cannot be fitted to: the fit set is alternate strikes minus the
    quotes the l1 bid/ask repair moves outside their spread, and an expiry with
    none left has no slice. Its quotes leave the scored universe, the rest are
    screened again (a calendar constraint may have tied them to it), and the
    expiry is recorded in ``sq.diagnostics['dropped_expiries']`` with its reason;
    n_universe and n_expiries count what is scored.

    The scorecard used to raise on such an expiry ('surface fit set misses 1
    quoted expiries'), so one broken expiry cost the session its whole surface
    report and the deploy (found by simulate_next_session.py on a book whose
    expiries a day apart carried independent noise). Returns (screen,
    arb_flag, fit_mask) for the quotes kept."""
    dropped: list[dict] = []
    for _ in range(max_rounds + 1):
        q = sq.quotes
        screen = arb.screen(q)
        arb_flag = screen.quote_flags["repair_outside_spread"].to_numpy(bool)
        fit_mask = sc.surface_fit_mask(q, arb_flag)
        has_fit = pd.Series(fit_mask).groupby(q["expiry"].to_numpy(), sort=False).any()
        empty = [str(e) for e, ok in has_fit.items() if not ok]
        if not empty:
            break
        if len(empty) == len(has_fit):
            raise ValueError(f"{sq.session}: the arbitrage repair leaves no expiry a fit-set quote")
        for e in empty:
            g = q[q["expiry"] == e]
            dropped.append({"expiry": e, "T_days": float(g["T_days"].iloc[0]), "n_quotes": len(g),
                            "reason": "the l1 bid/ask repair moves every fit-set quote (alternate "
                                      "strikes) outside its spread, so no slice can be fitted"})
        sq.quotes = q[~q["expiry"].isin(empty)].reset_index(drop=True)
    else:
        raise ValueError(f"{sq.session}: expiries kept losing their fit set after {max_rounds} rounds")
    if dropped:
        d = sq.diagnostics
        d["dropped_expiries"] = dropped
        d["n_dropped_no_fit_set"] = sum(x["n_quotes"] for x in dropped)
        d["n_universe"] = int(d.get("n_universe", 0)) - d["n_dropped_no_fit_set"]
        d["n_expiries"] = int(sq.quotes["expiry"].nunique())
        d["T_days_min"] = float(sq.quotes["T_days"].min())
        d["T_days_max"] = float(sq.quotes["T_days"].max())
        d["median_half_spread_vp"] = float(np.median(sq.quotes["half_spread_vp"]))
    return screen, arb_flag, fit_mask


def run_session(session: str, *, calibrations_dir: Path | None = None,
                chain: pd.DataFrame | None = None, models: list | None = None,
                source: str = "auto") -> dict:
    """Compute every artifact of ``session``. Returns {'residuals': frame,
    'surface': dict, 'scorecard': dict, 'vix_replication': dict}.

    The quotes are the session's newest stored Cboe book when it passes the
    quote-state gates, else its Yahoo chain (quotes.select_session_chain;
    ``source`` forces one); surface.json inputs record which and why."""
    session = str(pd.Timestamp(session).date())
    with single_thread_blas():
        sq = sqm.load_session_quotes(session, chain=chain, source=source)
        screen, arb_flag, fit_mask = drop_unfittable_expiries(sq)
        q = sq.quotes
        flags = screen.quote_flags
        ctx = sc.ScoreContext(sq, arb_flag, fit_mask,
                              calibrations_dir=Path(calibrations_dir or sc.CALIBRATIONS_DIR))
        card, resid = sc.score_models(ctx, models)
        essvi_surface, _ = sc._essvi(ctx)
        for m in card["models"]:              # SVI slice parameters are published whole
            if m.get("available") and m["model"] in ("svi", "svi_cal"):
                m["params"] = _exact_svi_params(m["params"])

        # residuals: add the arbitrage flags, stable order
        if len(resid):
            nq = len(q)
            reps = len(resid) // nq
            for col in ("arb_exec", "arb_mid"):
                resid[col] = np.tile(flags[col].to_numpy(bool), reps)
            resid["repair_dprice"] = np.tile(flags["repair_dprice"].to_numpy(float), reps)

        # surface.json
        rows = {m["model"]: m for m in card["models"] if m.get("available")}
        svi_surface = ctx.cache.get("svi")
        svi_cal_surface = ctx.cache["svi_cal"][0] if "svi_cal" in ctx.cache else None
        objs = {"essvi": essvi_surface, "svi": svi_surface, "svi_cal": svi_cal_surface}
        surfaces = {}
        for name in ("essvi", "svi", "svi_cal"):
            if name in rows:
                surfaces[name] = _surface_section(rows[name], objs[name])
        term = []
        carry = sq.carry.set_index("expiry")
        for exp, T, F, D, g in sqm.expiry_groups(q):
            c = carry.loc[exp]
            term.append({"expiry": exp, "T_days": T * 365.0, "forward": F, "discount": D,
                         "r": float(c["r"]), "q": float(c["q"]), "carry_method": str(c["method"]),
                         "n_quotes": len(g),
                         "median_half_spread_vp": float(np.median(g["half_spread_vp"])),
                         "atm_vol_essvi": 100.0 * essvi_surface.atm_vol(T),
                         "atm_skew_essvi": essvi_surface.atm_skew(T),
                         "atm_vol_svi_cal": (100.0 * float(svi_cal_surface.iv(np.array([0.0]), T)[0])
                                             if svi_cal_surface is not None else None),
                         "atm_vol_mid_quotes": 100.0 * float(np.interp(
                             0.0, np.sort(g["k"].to_numpy(float)),
                             g["mid_iv"].to_numpy(float)[np.argsort(g["k"].to_numpy(float))]))})
        src = sq.inputs
        surface_doc = {
            "schema_version": SCHEMA_VERSION, "report_version": REPORT_VERSION, "session": session,
            "inputs": {
                # which book (the session's Cboe capture when it passes the
                # quote-state gates, else its Yahoo chain) and why
                "chain": src.get("chain"), "chain_sha256": src.get("chain_sha256"),
                "raw_sha256": src.get("raw_sha256"),
                "chain_rows": len(sq.raw), "source": sq.diagnostics.get("source"),
                "source_reason": src.get("source_reason"),
                "refused_books": list(src.get("refused") or []),
                # None for a Yahoo chain (it records none); fetched_at is when it
                # was downloaded and never a T origin
                "quote_time": sq.diagnostics.get("quote_time"),
                "fetched_at": src.get("fetched_at"),
                "book_state": src.get("book_state"),
                # how many closing books of the session are stored (0 for a session
                # built from its overnight book: none, so the timing note says why)
                "closing_books": src.get("closing_books"),
                # with none stored, why the close slot did not store the one it
                # fetched, as its run record says (None: no record says so)
                "closing_book_refusal": src.get("closing_book_refusal"),
                "near_atm_two_sided_share": sq.diagnostics.get("near_atm_two_sided_share"),
                "t_origin": sq.diagnostics.get("t_origin"),
                "t_origin_basis": sq.diagnostics.get("t_origin_basis"),
                "spot": sq.spot, "r_ref_cc": sq.r_ref, "r_ref_source": sq.r_ref_source,
            },
            "universe": sq.diagnostics,
            "arbitrage": {**screen.summary(), "largest_mid_violations": screen.violations,
                          "quotes_repaired_outside_spread": _repaired_quotes(q, flags),
                          "constraint_set": ("CRW 2020 Table 1: C1-C3 per expiry; C4, C5 and C6 "
                                             "(C6.1 a/b/c and the nearest-neighbour chord) between "
                                             "consecutive expiries only; see arbitrage.py")},
            "fit_set": {"rule": "alternate strikes per expiry (even positions by strike), minus "
                                "quotes the l1-BA repair moves outside their spread",
                        "n": int(ctx.fit_mask.sum()), "n_universe": len(q),
                        "half_spread_floor_vp_in_weights": sc.H_FLOOR_FIT * 100.0},
            "surfaces": surfaces,
            "term_structure": term,
        }

        # scorecard.json
        avail = [m for m in card["models"] if m.get("available")]

        def _rank(split: str) -> list:
            return sorted(({"model": m["model"],
                            "oos_rmse_hs": m["metrics"][split].get("rmse_hs"),
                            "oos_median_abs_hs": m["metrics"][split].get("median_abs_hs"),
                            "oos_inside_spread_pct": m["metrics"][split].get("inside_spread_pct"),
                            "oos_rmse_vp": m["metrics"][split].get("rmse_vp")}
                           for m in avail if m["metrics"][split].get("n")),
                          key=lambda r: (r["oos_median_abs_hs"] is None, r["oos_median_abs_hs"]))
        common = card["common_out_of_sample"]
        own_n = ", ".join(f"{m['model']} {m['metrics']['out_of_sample'].get('n', 0):,}" for m in avail)
        scorecard_doc = {"schema_version": SCHEMA_VERSION, "report_version": REPORT_VERSION,
                         "session": session,
                         "n_quotes": len(q), "models": card["models"],
                         "common_out_of_sample": common,
                         "ranking_by_oos_median_abs_hs": _rank("out_of_sample"),
                         "ranking_by_common_oos_median_abs_hs": _rank("common_out_of_sample"),
                         "notes": [
                             (f"Every model is scored on the same quote universe (the "
                              f"calibrator's cleaned quotes, {len(q):,}) with the same metrics "
                              f"and dense-grid checks. Each model's out_of_sample set is the "
                              f"quotes it was not fitted to, so its size differs by model "
                              f"({own_n}); metrics.common_out_of_sample scores every model on "
                              f"the same {common['n']:,} quotes, those no model was fitted to."),
                             ("Surfaces are fitted to alternate strikes, so their out-of-sample "
                              "quotes are the strikes in between (an interpolation test); "
                              + _heston_note(card["models"])),
                             _svi_cal_note(surfaces.get("svi_cal")),
                             _quote_source_note(sq)]}

        # vix_replication.json
        calc, calc_basis = _calc_time_ny(sq)
        terms = vx.term_tables(sq.raw, session, calc)
        rate, rate_doc, rate_deviation = _vix_rate(sq)
        carry_map = {e: (float(f), float(d)) for e, f, d in
                     zip(sq.carry["expiry"], sq.carry["forward"], sq.carry["discount"])}
        official = vx.official_closes(session)
        indices: dict[str, dict[str, Any]] = {}
        for name, days in vx.CONSTANT_MATURITY_DAYS.items():
            rep: dict[str, Any] = {"cboe_rules": vx.replicate_index(terms, rate, days, spot=sq.spot),
                   "parity_forward": vx.replicate_index(terms, rate, days, carry=carry_map,
                                                        spot=sq.spot),
                   "any_expiry_parity_forward": vx.replicate_index(
                       terms, rate, days, candidates_only=False, carry=carry_map, spot=sq.spot)}
            off = official.get(name)
            rep["official"] = off
            method, why = vx.headline_method(rep)
            rep["headline_method"] = method
            if method is None:
                rep["not_replicated_reason"] = why
            else:
                head = rep[method]
                rep["replicated"] = head["level"]
                if off:
                    rep["gap_vs_official"] = head["level"] - off["close"]
                    if "low" in off:
                        rep["official_inside_day_range"] = bool(off["low"] <= head["level"] <= off["high"])
            indices[name] = rep
        # Cboe's near and next terms have more than 23 and fewer than 37 days
        indices["VIX"]["cboe_term_window"] = _vix_term_window(indices["VIX"], terms, calc)
        model_surfaces = {n: o for n, o in objs.items() if o is not None}
        strip = vx.variance_swap_strip(terms, rate, q, carry=carry_map,
                                       surfaces=model_surfaces)
        fitted_expiries = set(q["expiry"].astype(str))
        for name in vx.CONSTANT_MATURITY_DAYS:
            rep = indices[name]
            off = (rep.get("official") or {}).get("close")
            if rep["headline_method"] is None:
                rep["gap_attribution"] = {"available": False, "reason": "index not replicated"}
                continue
            head = rep[rep["headline_method"]]
            rep["gap_attribution"] = (
                vx.gap_attribution(head, terms, rate, svi_cal_surface, fitted_expiries, off,
                                   carry=carry_map)
                if svi_cal_surface is not None
                else {"available": False, "reason": "no SVI-cal surface"})
        T30 = 30.0 / 365.0
        m30 = indices["VIX"]["headline_method"]
        rep30 = indices["VIX"][m30] if m30 else {"ok": False}
        atm30 = vx.constant_maturity_atm(rep30, strip)
        beta: dict[str, Any] = {"atm_vol_30d_quotes": atm30,
                "atm_vol_30d_method": ("near / next ATM mid IVs of the replicated VIX's own terms, "
                                       "combined with the VIX constant-maturity weights"),
                "atm_vol_30d_essvi": 100.0 * essvi_surface.atm_vol(T30)}
        for name, surf in model_surfaces.items():
            beta[f"var_swap_vol_30d_{name}_k_-3_to_2"] = 100.0 * math.sqrt(
                max(vx.surface_variance_swap(surf, T30), 0.0))
        if svi_cal_surface is not None:
            beta["atm_vol_30d_svi_cal"] = 100.0 * float(svi_cal_surface.iv(np.array([0.0]), T30)[0])
            beta["atm_over_var_swap_30d_svi_cal"] = (
                beta["atm_vol_30d_svi_cal"] / beta["var_swap_vol_30d_svi_cal_k_-3_to_2"])
        if "VIX" in official:
            beta["vix_official"] = official["VIX"]["close"]
            if atm30:
                beta["atm_over_vix_official"] = atm30 / official["VIX"]["close"]
        if indices["VIX"].get("replicated"):
            beta["vix_replicated"] = indices["VIX"]["replicated"]
            if atm30:
                beta["atm_over_vix_replicated"] = atm30 / indices["VIX"]["replicated"]
        vix_doc = {
            "schema_version": SCHEMA_VERSION, "report_version": REPORT_VERSION, "session": session,
            "calculation_time_ny": calc.isoformat(timespec="minutes"),
            "calculation_time_basis": calc_basis,
            # when the session's Cboe closes print: its data cutoff (13:15 ET on an early close)
            "cboe_close_ny": _cboe_close_ny(session),
            "calculation_time_source": _calc_time_record(sq, calc, calc_basis),
            "rate": rate_doc,
            "indices": indices, "strip": strip, "beta": beta,
            "deviations": [
                *filter(None, [rate_deviation]),
                # a Cboe book is calculated at its own instant, as Cboe would: no departure
                # (its comparison with the 16:15 ET print is a gap note, _timing_note)
                *([_timing_deviation(sq, calc)] if calc_basis == CALC_T_ORIGIN else []),
                ("parity_forward replaces the single-strike forward with the cleaned chain's "
                 "parity-regression F and D (robust to one stale strike)."),
                ("K0 falls back to the nearest lower strike with both sides quoted when the listed "
                 "K0 lacks one (two-sided-only storage); flagged per term."),
                ("VIX9D/VIX3M/VIX6M use the VIX candidate set and bracket rule at 9/93/184 days "
                 "(analogues, not Cboe's index-specific series sets)."),
                *_settlement_deviations(terms),
                *filter(None, [vx.term_window_deviation(indices["VIX"]["cboe_term_window"])]),
            ],
            "gap_notes": _gap_notes(sq, terms, indices["VIX"], official,
                                    svi_surface, essvi_surface, calc, svi_cal_surface,
                                    indices["VIX"].get("gap_attribution"), indices,
                                    cboe_rate=(rate_deviation is None
                                               and _headline_at_cboe_rate(indices["VIX"])),
                                    calc_basis=calc_basis),
        }
    resid_cols = list(resid.columns)
    return {"residuals": resid[resid_cols].reset_index(drop=True), "surface": surface_doc,
            "scorecard": scorecard_doc, "vix_replication": vix_doc}


def write_session(session: str, out_root: Path = SURFACE_DIR, **kw) -> dict:
    """run_session + write the four artifacts; returns the index row."""
    out = run_session(session, **kw)
    d = Path(out_root) / session
    d.mkdir(parents=True, exist_ok=True)
    write_residuals(_frame_rounded(out["residuals"]), d / "residuals.parquet")
    dump_json(out["surface"], d / "surface.json")
    dump_json(out["scorecard"], d / "scorecard.json")
    dump_json(out["vix_replication"], d / "vix_replication.json")
    return index_row(out)


def index_row(out: dict) -> dict:
    sc_doc, vx_doc = out["scorecard"], out["vix_replication"]
    models = {}
    for m in sc_doc["models"]:
        if not m.get("available"):
            continue
        a, o = m["metrics"]["all"], m["metrics"]["out_of_sample"]
        models[m["model"]] = {"inside_spread_pct": a["inside_spread_pct"], "rmse_vp": a["rmse_vp"],
                              "median_abs_hs": a["median_abs_hs"],
                              "oos_inside_spread_pct": o.get("inside_spread_pct"),
                              "oos_rmse_vp": o.get("rmse_vp"),
                              "common_oos_median_abs_hs": m["metrics"].get(
                                  "common_out_of_sample", {}).get("median_abs_hs"),
                              "butterfly_violations": m["dense_grid"]["butterfly_violations"],
                              "calendar_violations": m["dense_grid"]["calendar_violations"]}
    v = vx_doc["indices"]["VIX"]
    arbd = out["surface"]["arbitrage"]
    inp = out["surface"].get("inputs") or {}
    return {"session": sc_doc["session"], "n_quotes": sc_doc["n_quotes"],
            "report_version": out["surface"].get("report_version"),
            "source": inp.get("source"), "quote_time": inp.get("quote_time"),
            "arbitrage_quotes_outside_spread": arbd["repair"].get("n_perturbed_outside_spread"),
            "executable_arbitrage": arbd["executable_arbitrage_certificate"].get("executable_arbitrage"),
            "vix_gap_attribution": {k: v.get("gap_attribution", {}).get(k) for k in
                                    ("truncation_pts", "residual_pts")},
            "models": models, "vix_replicated": v.get("replicated"),
            "vix_official": (v.get("official") or {}).get("close"),
            "vix_method": v.get("headline_method"),
            "atm_over_vix_official": vx_doc["beta"].get("atm_over_vix_official")}


def write_index(rows: list, out_root: Path = SURFACE_DIR) -> Path:
    path = Path(out_root) / "index.json"
    dump_json({"schema_version": SCHEMA_VERSION,
               "schema_doc": "joint_vol_calibration/surface/report.py (module docstring)",
               "sessions": sorted(rows, key=lambda r: r["session"])}, path)
    return path
