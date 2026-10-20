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
    source), universe counts, the static-arbitrage screen and repair
    summary with the largest violations, and for each fitted surface (essvi,
    svi) its parameters, analytic no-arbitrage margins, fit
    diagnostics, dense-grid checks and quote metrics; plus an ATM term
    structure (eSSVI ATM vol and skew per quoted expiry, parity carry).

scorecard.json — every model of scorecard.MODEL_REGISTRY on the same quotes:
    description, parameter count, in-sample size, metrics (all / in /
    out of sample / excluding arbitrage-flagged quotes, by expiry, tenor,
    delta and tenor x delta), dense-grid checks; a ranking by out-of-sample
    RMSE in half-spreads.

vix_replication.json — official closes, the replicated VIX (Cboe rules;
    parity-forward variant; any-weekday bracket) and VIX9D/3M/6M analogues,
    the variance-swap strip by expiry (Cboe single-term formula on the stored
    quotes) with the model-free ATM vol and the ATM / VS ratio, model
    variance swaps from the SVI and eSSVI fits, the 30-day ATM / VIX ratio
    (model-free), the methodology deviations and measured gap diagnostics.

index.json (in data_store/surface/) — one summary row per session written.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

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

SCHEMA_VERSION = 1
SURFACE_DIR = DATA_DIR / "surface"
SIG_DIGITS = 10
WIDE_K_GRID = np.linspace(-3.0, 2.0, 2001)      # analytic g checks of the fitted slices


# ── deterministic serialisation ───────────────────────────────────────────────

def round_sig(a, sig: int = SIG_DIGITS):
    """Round to ``sig`` significant digits (vectorised; NaN/inf/0 unchanged)."""
    x = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(x)))
    ok = np.isfinite(x) & (x != 0) & np.isfinite(mag)
    f = np.where(ok, 10.0 ** (sig - 1 - np.where(ok, mag, 0)), 1.0)
    return np.where(ok, np.round(x * f) / f, x)


def _clean(obj):
    """JSON-safe copy: numpy scalars to Python, floats rounded, NaN/inf to None."""
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


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _threadpool_limit():
    """Single-threaded BLAS for the small dense problems here, if threadpoolctl
    is importable (it ships with scikit-learn). On the shared 4-CPU machine,
    OpenBLAS thread oversubscription slowed the eSSVI fit 70-150x (0.06 s
    single-threaded vs 8.8 s)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:                            # optional speed-up only
        import contextlib
        return contextlib.nullcontext()


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
        gmins = [float(sv.svi_g(WIDE_K_GRID, s.params).min()) for s in surface.slices]
        lee = [max(s.params[1] * (1 + s.params[2]), s.params[1] * (1 - s.params[2]))
               for s in surface.slices]
        sec["analytic_g_min_on_fit_grid_k_-1.5_to_1"] = float(min(
            float(sv.svi_g(sv.SVI_GRID, s.params).min()) for s in surface.slices))
        sec["analytic_g_min_k_-3_to_2"] = float(min(gmins))
        sec["slices_with_g_negative_on_k_-3_to_2"] = int(sum(g < -1e-9 for g in gmins))
        sec["max_wing_slope_b_1_plus_abs_rho"] = float(max(lee))
    return sec


def _gap_notes(sq: sqm.SessionQuotes, terms: list, vixrep: dict, official: dict,
               svi_surface, essvi_surface: es.ESSVISurface, calc: datetime) -> list:
    notes = []
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
    for name, surf in (("SVI", svi_surface), ("eSSVI", essvi_surface)):
        if surf is not None:
            vs = 100.0 * math.sqrt(max(vx.surface_variance_swap(surf, T30), 0.0))
            parts.append(f"{name} {vs:.2f}")
    if parts:
        notes.append("Model variance swaps at 30 days (k in [-3, 2], wings beyond the fitted "
                     "K/F 0.70-1.30 extrapolated, so each inherits its model's wing bias): "
                     + ", ".join(parts) + ".")
    return notes


def run_session(session: str, *, calibrations_dir: Optional[Path] = None,
                chain: Optional[pd.DataFrame] = None, models: Optional[list] = None) -> dict:
    """Compute every artifact of ``session``. Returns {'residuals': frame,
    'surface': dict, 'scorecard': dict, 'vix_replication': dict}."""
    session = str(pd.Timestamp(session).date())
    with _threadpool_limit():
        sq = sqm.load_session_quotes(session, chain=chain)
        q = sq.quotes
        screen = arb.screen(q)
        flags = screen.quote_flags
        arb_flag = flags["repair_outside_spread"].to_numpy(bool)
        ctx = sc.ScoreContext(sq, arb_flag, sc.surface_fit_mask(q, arb_flag),
                              calibrations_dir=Path(calibrations_dir or sc.CALIBRATIONS_DIR))
        card, resid = sc.score_models(ctx, models)
        essvi_surface, _ = sc._essvi(ctx)

        # residuals: add the arbitrage flags, stable order
        if len(resid):
            nq = len(q)
            reps = len(resid) // nq
            for col in ("arb_exec", "arb_mid"):
                resid[col] = np.tile(flags[col].to_numpy(bool), reps)
            resid["repair_dprice"] = np.tile(flags["repair_dprice"].to_numpy(float), reps)

        # surface.json
        rows = {m["model"]: m for m in card["models"] if m.get("available")}
        surfaces = {}
        for name in ("essvi", "svi"):
            if name in rows:
                obj = essvi_surface if name == "essvi" else ctx.cache.get(name)
                surfaces[name] = _surface_section(rows[name], obj)
        term = []
        carry = sq.carry.set_index("expiry")
        for exp, T, F, D, g in sqm.expiry_groups(q):
            c = carry.loc[exp]
            term.append({"expiry": exp, "T_days": T * 365.0, "forward": F, "discount": D,
                         "r": float(c["r"]), "q": float(c["q"]), "carry_method": str(c["method"]),
                         "n_quotes": int(len(g)),
                         "median_half_spread_vp": float(np.median(g["half_spread_vp"])),
                         "atm_vol_essvi": 100.0 * essvi_surface.atm_vol(T),
                         "atm_skew_essvi": essvi_surface.atm_skew(T),
                         "atm_vol_mid_quotes": 100.0 * float(np.interp(
                             0.0, np.sort(g["k"].to_numpy(float)),
                             g["mid_iv"].to_numpy(float)[np.argsort(g["k"].to_numpy(float))]))})
        chain_path = db.options_snapshot_path(session, "SPX")
        surface_doc = {
            "schema_version": SCHEMA_VERSION, "session": session,
            "inputs": {
                "chain": f"data_store/options/spx_{session}.parquet" if chain is None else "argument",
                "chain_sha256": _sha256(chain_path) if chain is None else None,
                "chain_rows": int(len(sq.raw)), "source": sq.diagnostics.get("source"),
                "quote_time": sq.diagnostics.get("quote_time"),
                "t_origin": sq.diagnostics.get("t_origin"),
                "t_origin_basis": sq.diagnostics.get("t_origin_basis"),
                "spot": sq.spot, "r_ref_cc": sq.r_ref, "r_ref_source": "^IRX 13-week bill (tbill_rates, cc)",
            },
            "universe": sq.diagnostics,
            "arbitrage": {**screen.summary(), "largest_mid_violations": screen.violations,
                          "constraint_set": ("CRW 2020 Table 1: C1-C3 per expiry; C4, C5 and C6 "
                                             "(C6.1 a/b/c and the nearest-neighbour chord) between "
                                             "consecutive expiries only; see arbitrage.py")},
            "fit_set": {"rule": "alternate strikes per expiry (even positions by strike), minus "
                                "quotes the l1-BA repair moves outside their spread",
                        "n": int(ctx.fit_mask.sum()), "n_universe": int(len(q)),
                        "half_spread_floor_vp_in_weights": sc.H_FLOOR_FIT * 100.0},
            "surfaces": surfaces,
            "term_structure": term,
        }

        # scorecard.json
        avail = [m for m in card["models"] if m.get("available")]
        ranking = sorted(({"model": m["model"],
                           "oos_rmse_hs": m["metrics"]["out_of_sample"].get("rmse_hs"),
                           "oos_median_abs_hs": m["metrics"]["out_of_sample"].get("median_abs_hs"),
                           "oos_inside_spread_pct": m["metrics"]["out_of_sample"].get("inside_spread_pct"),
                           "oos_rmse_vp": m["metrics"]["out_of_sample"].get("rmse_vp")}
                          for m in avail if m["metrics"]["out_of_sample"].get("n")),
                         key=lambda r: (r["oos_median_abs_hs"] is None, r["oos_median_abs_hs"]))
        scorecard_doc = {"schema_version": SCHEMA_VERSION, "session": session,
                         "n_quotes": int(len(q)), "models": card["models"],
                         "ranking_by_oos_median_abs_hs": ranking,
                         "notes": [
                             "Every model is scored on the same quotes (the calibrator's OOS "
                             "universe) with the same metrics and dense-grid checks.",
                             "Surfaces are fitted to alternate strikes; Heston's in-sample set is "
                             "its 53 calibration anchors, fitted jointly with the VIX index leg.",
                             "Yahoo snapshots carry no quote time: part of every model's spread "
                             "misses can be stale quotes rather than model error."]}

        # vix_replication.json
        calc = _calc_time_ny(sq)
        terms = vx.term_tables(sq.raw, session, calc)
        carry_map = {e: (float(f), float(d)) for e, f, d in
                     zip(sq.carry["expiry"], sq.carry["forward"], sq.carry["discount"])}
        official = vx.official_closes(session)
        indices = {}
        for name, days in vx.CONSTANT_MATURITY_DAYS.items():
            rep = {"cboe_rules": vx.replicate_index(terms, sq.r_ref, days, spot=sq.spot),
                   "parity_forward": vx.replicate_index(terms, sq.r_ref, days, carry=carry_map,
                                                        spot=sq.spot),
                   "any_expiry_parity_forward": vx.replicate_index(
                       terms, sq.r_ref, days, candidates_only=False, carry=carry_map, spot=sq.spot)}
            off = official.get(name)
            rep["official"] = off
            head = rep["cboe_rules"] if rep["cboe_rules"]["ok"] else rep["parity_forward"]
            rep["headline_method"] = "cboe_rules" if rep["cboe_rules"]["ok"] else "parity_forward"
            if head["ok"]:
                rep["replicated"] = head["level"]
                if off:
                    rep["gap_vs_official"] = head["level"] - off["close"]
                    if "low" in off:
                        rep["official_inside_day_range"] = bool(off["low"] <= head["level"] <= off["high"])
            indices[name] = rep
        strip = vx.variance_swap_strip(terms, sq.r_ref, q, carry=carry_map,
                                       surfaces={"svi": ctx.cache.get("svi"),
                                                 "essvi": essvi_surface}
                                       if "svi" in ctx.cache else {"essvi": essvi_surface})
        T30 = 30.0 / 365.0
        rep30 = indices["VIX"]["cboe_rules"] if indices["VIX"]["cboe_rules"]["ok"] \
            else indices["VIX"]["parity_forward"]
        atm30 = vx.constant_maturity_atm(rep30, strip)
        beta = {"atm_vol_30d_quotes": atm30,
                "atm_vol_30d_method": ("near / next ATM mid IVs of the replicated VIX's own terms, "
                                       "combined with the VIX constant-maturity weights"),
                "atm_vol_30d_essvi": 100.0 * essvi_surface.atm_vol(T30)}
        for name in ("svi", "essvi"):
            surf = ctx.cache.get(name) if name == "svi" else essvi_surface
            if surf is not None:
                beta[f"var_swap_vol_30d_{name}_k_-3_to_2"] = 100.0 * math.sqrt(
                    max(vx.surface_variance_swap(surf, T30), 0.0))
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
            "rate": {"R_cc": sq.r_ref, "source": "^IRX 13-week bill (Cboe: CMT cubic spline per expiry)"},
            "indices": indices, "strip": strip, "beta": beta,
            "deviations": [
                "R is the ^IRX bill rate (cc), not Cboe's CMT spline; dVIX/dR ~ VIX T / 2.",
                "Calculation time is the chain's T origin; Yahoo snapshots carry no quote time.",
                "parity_forward replaces the single-strike forward with the cleaned chain's "
                "parity-regression F and D (robust to one stale strike).",
                "K0 falls back to the nearest lower strike with both sides quoted when the listed "
                "K0 lacks one (two-sided-only storage); flagged per term.",
                "VIX9D/VIX3M/VIX6M use the VIX candidate set and bracket rule at 9/93/184 days "
                "(analogues, not Cboe's index-specific series sets).",
                "Root-less (pre-C22) chains are treated as PM-settled.",
            ],
            "gap_notes": _gap_notes(sq, terms, indices["VIX"], official,
                                    ctx.cache.get("svi"), essvi_surface, calc),
        }
    resid_cols = [c for c in resid.columns]
    return {"residuals": resid[resid_cols].reset_index(drop=True), "surface": surface_doc,
            "scorecard": scorecard_doc, "vix_replication": vix_doc}


def write_session(session: str, out_root: Path = SURFACE_DIR, **kw) -> dict:
    """run_session + write the four artifacts; returns the index row."""
    out = run_session(session, **kw)
    d = Path(out_root) / session
    d.mkdir(parents=True, exist_ok=True)
    _frame_rounded(out["residuals"]).to_parquet(d / "residuals.parquet", index=False,
                                                compression="zstd")
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
                              "butterfly_violations": m["dense_grid"]["butterfly_violations"],
                              "calendar_violations": m["dense_grid"]["calendar_violations"]}
    v = vx_doc["indices"]["VIX"]
    return {"session": sc_doc["session"], "n_quotes": sc_doc["n_quotes"],
            "arbitrage_quotes_outside_spread": out["surface"]["arbitrage"]["repair"].get(
                "n_perturbed_outside_spread"),
            "models": models, "vix_replicated": v.get("replicated"),
            "vix_official": (v.get("official") or {}).get("close"),
            "vix_method": v.get("headline_method"),
            "atm_over_vix_official": vx_doc["beta"].get("atm_over_vix_official")}


def write_index(rows: list, out_root: Path = SURFACE_DIR) -> Path:
    path = Path(out_root) / "index.json"
    dump_json({"schema_version": SCHEMA_VERSION, "sessions": sorted(rows, key=lambda r: r["session"])},
              path)
    return path
