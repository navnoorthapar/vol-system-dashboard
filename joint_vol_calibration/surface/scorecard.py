"""
scorecard.py — score any implied-vol model on one session's quotes, in spread
units, with the same metrics and the same dense-grid arbitrage checks.

A model is a ModelSpec: a callable iv(k, T) -> implied vol (decimal) for
log-moneyness k = ln(K/F_T) (array) at maturity T (years), plus the mask of
quotes it was fitted to (in_sample). Prices are forward-normalised, so a
model needs neither F_T nor D_T: under deterministic rates and proportional
dividends C / (D_T F_T) depends on (k, T) only.

Built-in models (MODEL_REGISTRY, in scorecard order):
  essvi      eSSVI global surface (essvi.py), fitted to the surface fit set
  svi        raw SVI per expiry, butterfly certificate per slice (svi.py)
  svi_cal    raw SVI slices with the butterfly certificate AND calendar
             ordering against both neighbour slices (ordering certificate),
             started from the eSSVI fit (svi.fit_svi_calendar_surface): the
             arbitrage-free surface when surface.json's
             arbitrage_free_certified holds
  heston_v2  Heston at the session's stored model_version-2 joint fit
             (data_store/calibrations/joint_cal_<session>.pkl), priced with
             the repo engine (models.heston.heston_call_batch); in_sample =
             the calibrator's anchors for that session

The surface fit set holds alternate strikes of each expiry (positions 0, 2,
4, ... by strike) minus the quotes the arbitrage repair must move outside
their spread; the other quotes are out of sample for the surfaces. Heston's
in-sample set is its calibration anchors (53 quotes on 2026-10-16), so its
out-of-sample RMSE reproduces the pickle's oos.spx_oos_iv_rmse. The two
held-out sets differ, so every model is also scored on their intersection
(metrics.common_out_of_sample): a same-quotes comparison.

Adding a model (rough Heston, Quintic OU, ...): write a factory
``f(ctx) -> ModelSpec | None`` (None = not available for this session) and
``register_model("name", f)``; ctx is a ScoreContext. Nothing else changes.
A model that prices calls needs only a forward-normalised pricer
``call(K_over_F: ndarray, T: float) -> C / (D F)``; iv_from_normalised_calls
turns it into iv(k, T) (OTM side by parity, Black-76 inversion), e.g.

    def rough_heston_factory(ctx):
        cal = load_my_fit(ctx.sq.session)            # None when absent
        if cal is None:
            return None
        iv = iv_from_normalised_calls(lambda K, T: my_pricer(K, T, **cal["params"]))
        return ModelSpec("rough_heston", iv, heston_anchor_mask(ctx.quotes, ctx.sq.session),
                         len(cal["params"]), "rough Heston (lifted, n = 20)")
    register_model("rough_heston", rough_heston_factory)
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from joint_vol_calibration.calibration import joint_calibrator as jc
from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.models.heston import heston_call_batch
from joint_vol_calibration.surface import essvi as es
from joint_vol_calibration.surface import metrics as mt
from joint_vol_calibration.surface import svi as sv
from joint_vol_calibration.surface.quotes import SessionQuotes

CALIBRATIONS_DIR = DATA_DIR / "calibrations"
H_FLOOR_FIT = 5e-4          # 0.05 vol points: floor of the half-spread in fit weights
SVI_CAL_SWEEPS = 8          # block coordinate descent: at most 8 sweeps ...
SVI_CAL_REL_TOL = 1e-3      # ... stopping when a sweep gains < 0.1% of the objective


@dataclass
class ModelSpec:
    name: str
    iv: Callable                       # iv(k: ndarray, T: float) -> ndarray
    in_sample: np.ndarray              # bool mask over ctx.quotes
    n_params: int
    description: str
    params: dict = field(default_factory=dict)
    fit: dict = field(default_factory=dict)


@dataclass
class ScoreContext:
    """What a model factory may use: the session's quotes, the arbitrage flags
    and the surface fit set. Surfaces fitted once are cached in ``cache``."""
    sq: SessionQuotes
    arb_flag: np.ndarray               # quotes the l1-BA repair moves outside their spread
    fit_mask: np.ndarray               # surface fit set (see module docstring)
    calibrations_dir: Path = CALIBRATIONS_DIR
    cache: dict = field(default_factory=dict)

    @property
    def quotes(self) -> pd.DataFrame:
        return self.sq.quotes


def surface_fit_mask(quotes: pd.DataFrame, arb_flag: np.ndarray) -> np.ndarray:
    """Alternate strikes per expiry (even positions by strike), minus arb_flag."""
    pos = quotes.groupby("expiry", sort=False)["strike"].rank(method="first").to_numpy() - 1
    return (pos % 2 == 0) & ~np.asarray(arb_flag, dtype=bool)


def _fit_arrays(ctx: ScoreContext):
    q = ctx.quotes[ctx.fit_mask]
    hs = (q["ask_iv"] - q["bid_iv"]).to_numpy(float) / 2.0
    return q, hs


def _essvi(ctx: ScoreContext) -> tuple:
    if "essvi" not in ctx.cache:
        q, hs = _fit_arrays(ctx)
        ctx.cache["essvi"] = es.fit_essvi(q["k"], q["T"], q["mid_iv"], hs, q["expiry"].to_numpy(),
                                          h_floor=H_FLOOR_FIT)
    return ctx.cache["essvi"]


def _groups(ctx: ScoreContext) -> list:
    q, _ = _fit_arrays(ctx)
    out = []
    for exp, g in q.groupby("expiry", sort=False):
        out.append((exp, float(g["T"].iloc[0]), g["k"].to_numpy(float),
                    g["mid_iv"].to_numpy(float),
                    ((g["ask_iv"] - g["bid_iv"]) / 2.0).to_numpy(float)))
    return out


def _check_fit_covers_expiries(ctx: ScoreContext, T_fit) -> None:
    need = set(np.round(ctx.quotes["T"].to_numpy(float), 12))
    have = set(np.round(np.asarray(T_fit, dtype=float), 12))
    missing = need - have
    if missing:
        raise ValueError(f"surface fit set misses {len(missing)} quoted expiries")


def essvi_factory(ctx: ScoreContext) -> ModelSpec:
    surf, info = _essvi(ctx)
    _check_fit_covers_expiries(ctx, surf.T)
    return ModelSpec(
        "essvi", surf.iv, ctx.fit_mask.copy(), 3 * len(surf.T),
        "eSSVI global surface (theta, psi, rho per expiry); Gatheral-Jacquier butterfly "
        "conditions, monotone theta and wing slopes, and slice ordering certified on a "
        "4,001-point k-grid; inverse half-spread weights",
        params=surf.to_dict(), fit={k: v for k, v in info.items() if k != "x"})


def svi_factory(ctx: ScoreContext) -> ModelSpec:
    surf, _ = _essvi(ctx)
    if "svi" not in ctx.cache:
        ctx.cache["svi"] = sv.fit_svi_surface(_groups(ctx), surf, h_floor=H_FLOOR_FIT)
    s = ctx.cache["svi"]
    _check_fit_covers_expiries(ctx, s.T)
    return ModelSpec(
        "svi", s.iv, ctx.fit_mask.copy(), 5 * len(s.slices),
        "raw SVI per expiry; Durrleman g(k) >= 0 rows on 481 points of k in [-3, 2] plus "
        "cutting planes until svi.butterfly_certificate passes, and Lee wing bounds; no "
        "calendar constraint between slices; inverse half-spread weights",
        params=s.to_dict(),
        fit={"slices_converged": sum(1 for x in s.slices if x.info.get("success")),
             "slices_certified": sum(1 for x in s.slices if x.info.get("certified")),
             "n_slices": len(s.slices),
             "max_constraint_violation": float(max(x.info.get("max_constraint_violation", 0.0)
                                                   for x in s.slices)),
             "slices_returned_start": int(sum(bool(x.info.get("returned_start"))
                                              for x in s.slices)),
             "mean_objective": float(np.mean([x.info["objective"] for x in s.slices]))})


def svi_cal_factory(ctx: ScoreContext) -> ModelSpec:
    surf, _ = _essvi(ctx)
    if "svi_cal" not in ctx.cache:
        if "svi" not in ctx.cache:
            ctx.cache["svi"] = sv.fit_svi_surface(_groups(ctx), surf, h_floor=H_FLOOR_FIT)
        free = [x.params for x in ctx.cache["svi"].slices]     # extra starts per slice update
        ctx.cache["svi_cal"] = sv.fit_svi_calendar_surface(_groups(ctx), surf, h_floor=H_FLOOR_FIT,
                                                           max_sweeps=SVI_CAL_SWEEPS,
                                                           rel_tol=SVI_CAL_REL_TOL, starts=free)
    s, info = ctx.cache["svi_cal"]
    _check_fit_covers_expiries(ctx, s.T)
    return ModelSpec(
        "svi_cal", s.iv, ctx.fit_mask.copy(), 5 * len(s.slices),
        "raw SVI per expiry with Durrleman g(k) >= 0 rows on 481 points of k in [-3, 2] and "
        "Lee wing bounds, and calendar ordering w_(i-1) <= w_i <= w_(i+1); a slice is accepted "
        "only when svi.butterfly_certificate and svi.ordering_certificate pass (cutting planes "
        "until they do); block coordinate descent from the eSSVI fit; inverse half-spread "
        "weights",
        params=s.to_dict(), fit=dict(info.items()))


def iv_from_normalised_calls(call_price: Callable) -> Callable:
    """iv(k, T) from a forward-normalised call pricer call_price(K, T) ->
    C / (D F) at strikes K = K/F (array): the OTM side is priced (puts by
    parity, P = C - (1 - K)) and inverted with the repo's Black-76 solver on
    F = 1, D = 1, as in JointCalibrator._surface_errors. NaN where the price
    has no implied vol."""
    def iv(k, T):
        K = np.exp(np.atleast_1d(np.asarray(k, dtype=float)))
        T = float(T)
        calls = np.asarray(call_price(K, T), dtype=float)
        is_call = K >= 1.0
        prices = np.where(is_call, calls, calls - (1.0 - K))       # OTM put by parity
        return oc.black76_implied_vol(prices, 1.0, K, T, 1.0, is_call)
    return iv


def heston_iv_function(params: dict) -> Callable:
    """iv(k, T) of Heston at ``params`` (kappa, theta, sigma, rho, v0): the
    repo's Lewis engine (models.heston.heston_call_batch) on F = 1, D = 1."""
    kap, th, sg, rh, v0 = (float(params[n]) for n in ("kappa", "theta", "sigma", "rho", "v0"))
    return iv_from_normalised_calls(
        lambda K, T: heston_call_batch(1.0, K, T, 0.0, 0.0, kap, th, sg, rh, v0))


def load_v2_calibration(session: str, calibrations_dir: Path = CALIBRATIONS_DIR) -> dict | None:
    """The stored model_version-2 joint fit for ``session`` (None if absent,
    legacy, or built on another session's quotes)."""
    path = Path(calibrations_dir) / f"joint_cal_{session}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        cal = pickle.load(f)  # noqa: S301 (trusted: the calibrator's own committed result)
    if jc.calibration_is_legacy(cal):
        return None
    if str(cal.get("spx_snapshot_session")) != session:
        return None
    return cal


def heston_anchor_mask(quotes: pd.DataFrame, session: str) -> np.ndarray:
    """The calibrator's anchors among ``quotes``: one representative expiry per
    target tenor (option_chain.select_calibration_expiries) and the z-grid
    anchors of JointCalibrator._anchor_rows, on the same universe."""
    chosen = oc.select_calibration_expiries(quotes["expiry"].unique(), session)
    mask = np.zeros(len(quotes), dtype=bool)
    for exp in chosen:
        g = quotes[quotes["expiry"] == exp]
        frame = pd.DataFrame({"log_moneyness": g["k"].to_numpy(float),
                              "time_to_expiry": g["T"].to_numpy(float),
                              "implied_vol": g["mid_iv"].to_numpy(float),
                              "right": g["right"].to_numpy(),
                              "_row": g.index.to_numpy()})
        picked = jc.JointCalibrator._anchor_rows(frame)
        if len(picked) >= 2:
            mask[picked["_row"].to_numpy(int)] = True
    return mask


def heston_v2_factory(ctx: ScoreContext) -> ModelSpec | None:
    cal = load_v2_calibration(ctx.sq.session, ctx.calibrations_dir)
    if cal is None:
        return None
    p = cal["params"]
    return ModelSpec(
        "heston_v2", heston_iv_function(p), heston_anchor_mask(ctx.quotes, ctx.sq.session), 5,
        "Heston at the stored model_version-2 joint SPX/VIX-index fit (DE + L-BFGS-B on "
        "53 z-grid anchors and 4 VIX indices; repo Lewis engine)",
        params={"model": "Heston", **{k: float(v) for k, v in p.items()},
                "pinned_params": list(cal.get("pinned_params", [])),
                "feller_margin": cal.get("feller_margin"),
                "stored_oos_rmse_vp": cal.get("leg_losses", {}).get("spx_oos_iv_rmse"),
                "stored_in_sample_rmse_vp": cal.get("leg_losses", {}).get("spx_iv_rmse_exact"),
                "run_utc": cal.get("run_utc")},
        fit={"source": f"joint_cal_{ctx.sq.session}.pkl", "n_anchors_stored": cal.get("n_spx_options")})


MODEL_REGISTRY: dict = {
    "essvi": essvi_factory,
    "svi": svi_factory,
    "svi_cal": svi_cal_factory,
    "heston_v2": heston_v2_factory,
}


def register_model(name: str, factory: Callable) -> None:
    """Add (or replace) a model factory; it is scored after the built-ins."""
    MODEL_REGISTRY[name] = factory


def model_ivs(spec: ModelSpec, quotes: pd.DataFrame) -> np.ndarray:
    """spec.iv evaluated on every quote, one call per expiry."""
    out = np.full(len(quotes), np.nan)
    for _, g in quotes.groupby("expiry", sort=False):
        idx = g.index.to_numpy()
        out[idx] = np.asarray(spec.iv(g["k"].to_numpy(float), float(g["T"].iloc[0])), dtype=float)
    return out


def score_models(ctx: ScoreContext, models: list | None = None) -> tuple:
    """Score every available model. Returns (scorecard dict, residuals frame).

    Each model's out_of_sample metrics use its own held-out set (the quotes it
    was not fitted to), so their sizes differ by model; common_out_of_sample
    scores every model on the same quotes: those no available model was
    fitted to (the intersection of the held-out sets)."""
    q = ctx.quotes
    if not (q.index == np.arange(len(q))).all():
        raise AssertionError("quotes must have a RangeIndex")
    ranges = mt.expiry_ranges(q)
    specs = [(name, MODEL_REGISTRY[name](ctx)) for name in (models or list(MODEL_REGISTRY))]
    common = np.ones(len(q), dtype=bool)
    for _, spec in specs:
        if spec is not None:
            common &= ~np.asarray(spec.in_sample, dtype=bool)
    rows, frames = [], []
    for name, spec in specs:
        if spec is None:
            rows.append({"model": name, "available": False})
            continue
        miv = model_ivs(spec, q)
        res = mt.residual_frame(q, miv, model=spec.name, in_sample=spec.in_sample,
                                extra={"arb_flag": ctx.arb_flag})
        frames.append(res)
        metrics = mt.summarise(res)
        metrics["common_out_of_sample"] = mt.stats_on(res, common)
        rows.append({
            "model": spec.name, "available": True, "description": spec.description,
            "n_params": spec.n_params, "n_in_sample": int(np.sum(spec.in_sample)),
            "metrics": metrics,
            "dense_grid": mt.dense_grid_checks(spec.iv, ranges),
            "params": spec.params, "fit": spec.fit,
        })
    resid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    card = {"session": ctx.sq.session, "models": rows,
            "common_out_of_sample": {
                "n": int(common.sum()),
                "models": [r["model"] for r in rows if r.get("available")],
                "rule": ("the quotes no available model was fitted to: the intersection of "
                         "every model's out-of-sample set")}}
    return card, resid
