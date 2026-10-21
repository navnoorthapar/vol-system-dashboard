"""
report.py — run the surface layer on one stored session and write its
artifacts under data_store/surface/<session>/.

Artifacts (schema_version 1; floats rounded to 10 significant digits, keys
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

surface.json — inputs (chain path and SHA-256, spot, rate, T origin,
    source), universe counts, the static-arbitrage screen (row counts and
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
    each with a gap attribution (wing truncation measured on the SVI-cal
    surface, residual = official - replicated - truncation); the
    variance-swap strip by expiry (Cboe single-term formula on the stored
    quotes) with the model-free ATM vol and the ATM / VS ratio (the
    empirical beta), model variance swaps from the eSSVI, SVI and SVI-cal
    fits, the 30-day ATM / VIX ratio (model-free), the methodology
    deviations and the gap notes.

index.json (in data_store/surface/) — one summary row per session written.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
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
SURFACE_DIR = DATA_DIR / "surface"
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


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── per-session computation ───────────────────────────────────────────────────

def _calc_time_ny(sq: sqm.SessionQuotes) -> datetime:
    t0 = datetime.fromisoformat(sq.clean_diagnostics["t_origin"])
    return t0.astimezone(oc.NY_TZ).replace(tzinfo=None)


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


def _gap_notes(sq: sqm.SessionQuotes, terms: list, vixrep: dict, official: dict,
               svi_surface, essvi_surface: es.ESSVISurface, calc: datetime,
               svi_cal_surface=None, attribution: dict | None = None,
               indices: dict | None = None) -> list:
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
        near, nxt = v["near"], v["next"]
        target = date.fromisoformat(sq.session) + timedelta(days=30)
        exp_dates = {t["expiry"] for t in terms}
        d = target + timedelta(days=1)
        while not vx.is_end_of_week_expiry(d.isoformat()):
            d += timedelta(days=1)
        if d.isoformat() not in exp_dates:
            notes.append(f"The first Cboe candidate expiry after 30 days ({d.isoformat()}) is not "
                         f"in the stored chain (the bot stores calibration expiries and "
                         f"neighbours), so the bracket is {near['expiry']} ({near['days']:.1f}d) - "
                         f"{nxt['expiry']} ({nxt['days']:.1f}d) instead of a <= 7-day bracket.")
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
                     + "): snapshot timing vs the 16:15 ET print, stale quotes, rate.")
    return notes


def run_session(session: str, *, calibrations_dir: Path | None = None,
                chain: pd.DataFrame | None = None, models: list | None = None) -> dict:
    """Compute every artifact of ``session``. Returns {'residuals': frame,
    'surface': dict, 'scorecard': dict, 'vix_replication': dict}."""
    session = str(pd.Timestamp(session).date())
    with single_thread_blas():
        sq = sqm.load_session_quotes(session, chain=chain)
        q = sq.quotes
        screen = arb.screen(q)
        flags = screen.quote_flags
        arb_flag = flags["repair_outside_spread"].to_numpy(bool)
        ctx = sc.ScoreContext(sq, arb_flag, sc.surface_fit_mask(q, arb_flag),
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
        chain_path = db.options_snapshot_path(session, "SPX")
        surface_doc = {
            "schema_version": SCHEMA_VERSION, "session": session,
            "inputs": {
                "chain": f"data_store/options/spx_{session}.parquet" if chain is None else "argument",
                "chain_sha256": _sha256(chain_path) if chain is None else None,
                "chain_rows": len(sq.raw), "source": sq.diagnostics.get("source"),
                "quote_time": sq.diagnostics.get("quote_time"),
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
        scorecard_doc = {"schema_version": SCHEMA_VERSION, "session": session,
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
                              "quotes are the strikes in between (an interpolation test); Heston's "
                              "in-sample set is its calibration anchors, fitted jointly with the "
                              "VIX index leg, and its out-of-sample set is every other quote."),
                             _svi_cal_note(surfaces.get("svi_cal")),
                             ("Yahoo snapshots carry no quote time: part of every model's spread "
                              "misses can be stale quotes rather than model error.")]}

        # vix_replication.json
        calc = _calc_time_ny(sq)
        terms = vx.term_tables(sq.raw, session, calc)
        carry_map = {e: (float(f), float(d)) for e, f, d in
                     zip(sq.carry["expiry"], sq.carry["forward"], sq.carry["discount"])}
        official = vx.official_closes(session)
        indices: dict[str, dict[str, Any]] = {}
        for name, days in vx.CONSTANT_MATURITY_DAYS.items():
            rep: dict[str, Any] = {"cboe_rules": vx.replicate_index(terms, sq.r_ref, days, spot=sq.spot),
                   "parity_forward": vx.replicate_index(terms, sq.r_ref, days, carry=carry_map,
                                                        spot=sq.spot),
                   "any_expiry_parity_forward": vx.replicate_index(
                       terms, sq.r_ref, days, candidates_only=False, carry=carry_map, spot=sq.spot)}
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
        model_surfaces = {n: o for n, o in objs.items() if o is not None}
        strip = vx.variance_swap_strip(terms, sq.r_ref, q, carry=carry_map,
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
                vx.gap_attribution(head, terms, sq.r_ref, svi_cal_surface, fitted_expiries, off,
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
            "schema_version": SCHEMA_VERSION, "session": session,
            "calculation_time_ny": calc.isoformat(timespec="minutes"),
            "rate": {"R_cc": sq.r_ref,
                     "source": ("^IRX 13-week bill (Cboe: CMT cubic spline per expiry)"
                                if sq.r_ref_source == sqm.IRX_SOURCE else
                                f"{sq.r_ref_source}; its 91-day rate for both terms "
                                "(Cboe: CMT cubic spline per expiry)")},
            "indices": indices, "strip": strip, "beta": beta,
            "deviations": [
                ("R is the ^IRX bill rate (cc), not Cboe's CMT spline; dVIX/dR ~ VIX T / 2."
                 if sq.r_ref_source == sqm.IRX_SOURCE else
                 "R is the Treasury zero curve's 91-day rate (cc) for both terms, not Cboe's "
                 "CMT spline per expiry; dVIX/dR ~ VIX T / 2."),
                "Calculation time is the chain's T origin; Yahoo snapshots carry no quote time.",
                ("parity_forward replaces the single-strike forward with the cleaned chain's "
                 "parity-regression F and D (robust to one stale strike)."),
                ("K0 falls back to the nearest lower strike with both sides quoted when the listed "
                 "K0 lacks one (two-sided-only storage); flagged per term."),
                ("VIX9D/VIX3M/VIX6M use the VIX candidate set and bracket rule at 9/93/184 days "
                 "(analogues, not Cboe's index-specific series sets)."),
                *_settlement_deviations(terms),
            ],
            "gap_notes": _gap_notes(sq, terms, indices["VIX"], official,
                                    svi_surface, essvi_surface, calc, svi_cal_surface,
                                    indices["VIX"].get("gap_attribution"), indices),
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
    return {"session": sc_doc["session"], "n_quotes": sc_doc["n_quotes"],
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
