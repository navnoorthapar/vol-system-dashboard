"""
joint_calibrator.py — Joint SPX/VIX Smile Calibration Engine (C4, revised C22).

Minimises a single weighted loss over the market observables:

  L(kappa, theta, sigma, rho, v0) =
      w1 * MSE_SPX(implied vols)            # SPX smile, 6 tenors x 9 anchors
    + w2 * MSE_VIX_index(levels)            # ^VIX9D/^VIX/^VIX3M/^VIX6M
    + w3 * MSE_VIX_options(implied vols)    # disabled (w3 = 0), see below
   [+ Feller penalty, only when feller="soft"]

with the active weights normalised to sum to one.

What the C22 audit changed (each is a correction, several make the published
fit worse — those are the honest numbers):

  * VIX leg. ^VIX9D/^VIX/^VIX3M/^VIX6M are Cboe SPOT-starting indices: the
    expected average variance from today to 9/30/93/184 days. The leg used to
    compare them with the model's VIX FUTURES curve (30-day vol over the
    forward window [T, T+30d]) at 9/30/91/182 days — the wrong instrument at
    every tenor. It now uses heston_vix_term_index() at
    config.VIX_INDEX_TENOR_DAYS and is reported as ``vix_index_rmse``
    (``vix_futures_rmse`` is kept as a back-compat alias key). True VX futures,
    when the DB has them, are scored out of sample with heston_vix_future_exact.
  * SPX data. Quotes are cleaned by data/option_chain.py: two-sided only,
    never lastPrice, per-expiry put-call-parity forward F_T and discount D_T,
    own Black-76 IVs (Yahoo's IV is a diagnostic only). The SPX leg prices
    each slice off (F_T, D_T); there is no assumed r/q. T runs to 16:00 ET
    (PM) or 09:30 ET (AM) settlement from option_chain.t_origin: the 16:00 ET
    close of the session whose quotes the snapshot holds for a closing book
    (a Sunday snapshot holds Friday's book), the quote time for a Cboe Global
    Trading Hours book (the Mon-Thu book slots, 21:30-00:30 ET). The VIX
    index leg still uses the session's 16:15 ET closes, so a GTH fit carries
    a residual SPX/VIX timing mismatch (methods["vix_leg_timing"]).
  * Anchors. The 9 highest-vega strikes were adjacent ATM-call strikes — the
    fit never saw the put wing. Anchors are now the strikes nearest a
    standardised forward-moneyness grid z = ln(K/F)/(sigma_atm*sqrt(T)) in
    linspace(-2.5, 1.5, 9), OTM against F_T, with >= 3 puts below K/F = 0.97
    wherever such quotes exist. Errors stay vega-normalised. An OUT-OF-SAMPLE
    IV RMSE over the full cleaned OTM surface is reported next to the
    in-sample anchor RMSE.
  * Expiries. One listed expiry per target tenor (14..365d), unique; the
    bucket loop could previously fill several buckets with one expiry.
  * Feller. 2*kappa*theta >= sigma^2 is not a no-arbitrage condition and the
    pricer is valid when it fails (variance touches zero and reflects), so the
    penalty is opt-in (feller="soft"); default "off". The margin and
    PASS/BINDING/VIOLATED state are always reported as a diagnostic, and
    ``active_constraints`` lists the constraints that actually bind.
  * Gate. is_acceptable_calibration() adds SPX data sufficiency (>= 4
    expiries, >= 30 options, put wing present), surface freshness (snapshot
    session == spot session) and a model_version check; calibration_is_legacy()
    identifies pre-C22 results.
  * The SVI/SSVI smoothing helpers were never on the production path (only
    tests called them) and have been removed.

VIX options leg (w3 = 0). The pricer was fixed (puts by parity with the exact
E[VIX_T] from the same CIR law, T = 0 uses the model's spot VIX, the payoff is
integrated against the CIR survival function, which is smooth where the
density is singular) and the chain is now read only when its quote session is
the spot session, with Black-76 vegas/moneyness on the parity-implied VX
forward. It stays disabled: the DB holds same-session VIX chains for only four
sessions (2026-03-24, 03-27, 05-29, 06-18; 0-7 liquid options each after
filtering) and the daily job captures SPX only, so the leg cannot be part of
the routine fit. Its old 37 vp "structural failure" figure was measured with
the broken put pricer, spot-VIX vegas and a stale chain, so it says nothing
about the model.

Optimiser and diagnostics (C23). The loss is a sum of squares and is
minimised as one (HestonJointResiduals + multistart_trf); every result
records its starts, basins and an identifiability report. sigma_profile()
measures the cost of the sigma <= 2 box, calibrate_comparison() fits Heston
and lifted rough Heston like for like, and
scripts/calibration_diagnostics.py writes both for every committed
session (data_store/calibrations/diagnostics/).

Usage
-----
    from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator
    cal = JointCalibrator(as_of_date='2026-03-24')
    result = cal.calibrate()
"""

import functools
import logging
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import qmc

from joint_vol_calibration.config import (
    HESTON_BOUNDS, JOINT_W1, JOINT_W2, JOINT_W3,
    RANDOM_SEED, DATA_DIR, VIX_INDEX_TENOR_DAYS,
    HESTON_START_BOX, SIGMA_PROFILE_CAPS, PROFILE_KAPPA_MAX,
    ROUGH_HESTON_H_BOUNDS, ROUGH_HESTON_N_FACTORS, ROUGH_HESTON_TIME_STEPS,
    ROUGH_HESTON_START_BOX,
)
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.data.option_chain import CAL_EXPIRY_TARGETS_DAYS
from joint_vol_calibration.models.heston import (
    HESTON_PARAM_NAMES,
    heston_call_batch,
    heston_call_batch_grid,
    heston_call_batch_jac,
    heston_vix_term_index,
    heston_vix_term_index_jac,
    heston_vix_index,
    heston_vix_future_exact,
    heston_vix_futures_curve,      # noqa: F401  (re-exported for back-compat)
    heston_vvix,
    bates_call_batch,
    bates_vix_term_index,
    _one_minus_exp_over,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_VERSION = 2              # results written by this code; pre-C22 = 1
_VIX_WINDOW  = 30.0 / 365.0   # VIX 30-day integration window
_FELLER_PENALTY = 50.0         # strength of the opt-in soft Feller penalty
_FELLER_MODES = ("off", "soft")

# Feller reporting tolerance. With feller="soft" the penalty is a soft
# exterior penalty: its gradient vanishes at the boundary, so the optimiser
# converges ~1e-7..1e-6 OUTSIDE the feasible set whenever the constraint
# binds. Anything inside this tolerance is classified BINDING, not VIOLATED.
_FELLER_TOL = 1e-4

# Minimum VIX term-structure tenors for the joint calibration to be identified.
# With a single tenor the VIX leg is one data point, which a 5-parameter Heston
# matches exactly; the joint problem then degenerates into an SPX-only fit
# with a spuriously LOW SPX RMSE (C20).
_MIN_VIX_TENORS = 3

# SPX dividend yield assumed by carry="flat" (and for legacy results that never
# recorded a q). The default carry="parity" takes q per expiry from put-call
# parity instead and records it in the result.
FLAT_CARRY_Q = 0.013

# SPX data sufficiency (C22). The SPX leg has the same failure mode as the VIX
# leg: fewer data, lower RMSE. A one-expiry fit (every production run
# 2026-09-08..09-23) has no term structure, so kappa/theta/v0 are free.
_MIN_SPX_EXPIRIES = 4
_MIN_SPX_OPTIONS = 30
_MIN_PUT_WING_EXPIRIES = 3     # expiries with >= 3 anchors at K/F < 0.97
_PUT_WING_KF = 0.97
_MIN_WING_PUTS_PER_EXPIRY = 3
_MAX_SPX_RMSE = 3.0            # vol points, in-sample anchor RMSE

_MIN_T_DAYS  = oc.CAL_MIN_EXPIRY_DAYS   # minimum expiry to include (days)
_MAX_T_YEARS = 2.0             # maximum expiry to include (years)
_MONEYNESS_LO = 0.70           # K/F lower bound for the filtered surface
_MONEYNESS_HI = 1.30           # K/F upper bound
_MIN_IV = 0.01                 # discard IVs below 1%
_MAX_IV = 2.00                 # discard IVs above 200%
_MIN_PRICE = 0.50              # discard options priced below $0.50
_MIN_VEGA = 0.5                # vega floor (price units per unit vol)

# Calibration anchors: one expiry per target tenor x 9 strikes on a
# standardised forward-moneyness grid (54 options max).
_CAL_EXPIRY_TARGETS_DAYS = list(CAL_EXPIRY_TARGETS_DAYS)  # single source: option_chain
_CAL_STRIKES_PER_EXPIRY  = 9
_ANCHOR_Z_GRID = np.linspace(-2.5, 1.5, _CAL_STRIKES_PER_EXPIRY)
_ANCHOR_Z_TOL = 0.25           # half the grid spacing


# ── VIX term-structure indices (ticker → tenor in years) ──────────────────────
# Spot-starting Cboe windows (config.VIX_INDEX_TENOR_DAYS: 9/30/93/184 days).
_VIX_TS_TENORS = {t: d / 365.0 for t, d in VIX_INDEX_TENOR_DAYS.items()}


# ── Calibration quality gate ──────────────────────────────────────────────────

def classify_feller(kappa: float, theta: float, sigma: float,
                    tol: float = _FELLER_TOL) -> tuple:
    """Classify the Feller condition as PASS / BINDING / VIOLATED.

    Feller (2*kappa*theta >= sigma**2) only says whether the variance process
    can reach zero; Heston prices are well defined either way. It is reported
    as a diagnostic. "BINDING" means |margin| <= tol (a soft penalty, when
    imposed, leaves the optimum ~1e-6 outside the boundary).

    Returns (state, margin) where margin = 2*kappa*theta - sigma**2.
    """
    margin = 2.0 * float(kappa) * float(theta) - float(sigma) ** 2
    if margin > tol:
        return "PASS", margin
    if margin >= -tol:
        return "BINDING", margin
    return "VIOLATED", margin


def _at_bound(val: float, bound: float, span: float, rel_tol: float) -> bool:
    """|val - bound| within rel_tol of the bound's own magnitude (or of the
    box span when the bound is 0). A span-relative tolerance on a [1e-4, 1]
    box called v0 = 1.04e-3 'pinned' at 1e-4."""
    scale = abs(bound) if bound != 0 else span
    return abs(val - bound) <= rel_tol * scale


def count_pinned_params(params: dict, feller_state: Optional[str] = None,
                        rel_tol: float = 1e-3, feller_imposed: bool = True) -> list:
    """Names of the constraints that are active at a fit.

    Box bounds: a parameter within ``rel_tol`` (relative to the bound value)
    of its HESTON_BOUNDS edge. Feller: the pseudo-name "feller" is appended
    when ``feller_imposed`` and the state is BINDING — the Feller surface
    constrains (kappa, theta, sigma) jointly, so it costs one degree of
    freedom but is not attributable to sigma (pre-C22 this appended "sigma").
    Pre-C22 fits all imposed the soft penalty, hence the default True; a fit
    with feller="off" passes feller_imposed=False.

    effective d.o.f. = 5 - len(result).
    """
    pinned = []
    for name in ("kappa", "theta", "sigma", "rho", "v0"):
        if name not in params or name not in HESTON_BOUNDS:
            continue
        val = float(params[name])
        lo, hi = HESTON_BOUNDS[name]
        span = (hi - lo) or 1.0
        if _at_bound(val, lo, span, rel_tol) or _at_bound(val, hi, span, rel_tol):
            pinned.append(name)
    if feller_imposed and feller_state == "BINDING":
        pinned.append("feller")
    return pinned


def active_constraints(params: dict, feller_state: Optional[str] = None,
                       feller_imposed: bool = True, rel_tol: float = 1e-3) -> list:
    """Human-readable list of the active constraints (replaces the old
    'Pinned: rho, sigma at boundary' message)."""
    out = []
    for name in count_pinned_params(params, feller_state, rel_tol, feller_imposed):
        if name == "feller":
            out.append("2*kappa*theta >= sigma^2 (Feller, imposed, binding)")
            continue
        lo, hi = HESTON_BOUNDS[name]
        val = float(params[name])
        side = "floor" if abs(val - lo) <= abs(val - hi) else "ceiling"
        out.append(f"{name} = {lo if side == 'floor' else hi:g} (box {side})")
    return out


def calibration_is_legacy(cal: dict) -> bool:
    """True for results written before C22 (model_version < 2 or missing).

    Legacy fits used the VIX-futures mapping for the spot indices, the
    top-vega ATM-call anchors, Yahoo IVs and a fixed r/q; their RMSEs are not
    comparable with model_version 2 fits. The dashboard should label or
    exclude them.
    """
    try:
        return int(cal.get("model_version", 1)) < MODEL_VERSION
    except (TypeError, ValueError, AttributeError):
        return True


def _session_of(snapshot_date) -> Optional[str]:
    if snapshot_date is None:
        return None
    return oc.session_on_or_before(snapshot_date).isoformat()


def is_acceptable_calibration(params: dict, spx_iv_rmse: Optional[float] = None,
                              n_vix_tenors: Optional[int] = None, *,
                              n_spx_expiries: Optional[int] = None,
                              n_spx_options: Optional[int] = None,
                              n_put_wing_expiries: Optional[int] = None,
                              spx_snapshot_session: Optional[str] = None,
                              spot_date: Optional[str] = None,
                              model_version: Optional[int] = None,
                              allow_legacy: bool = False) -> tuple:
    """Reject degenerate, under-identified or stale Heston fits.

    Two call forms:
      * ``is_acceptable_calibration(result)`` with a full calibration result
        dict (it has a "params" key) — the single source of truth used by
        cloud_refresh and the dashboard. Every check below is applied from the
        result's own fields; a legacy (pre-C22) result is rejected unless
        ``allow_legacy`` (then only the checks its fields support are run,
        with n_spx_options standing in for the missing expiry count).
      * the pre-C22 positional form (params, spx_iv_rmse, n_vix_tenors, ...)
        — checks whose inputs are None are skipped.

    Checks (the first failing one is the reason returned):
      1. surface freshness: the session whose quotes the SPX snapshot holds
         must be the spot session;
      2. model_version >= 2;
      3. degenerate corner (sigma < 0.05, |rho| < 0.10, kappa at a bound,
         v0/theta within 2x of their floor);
      4. in-sample SPX RMSE <= 3 vp;
      5. VIX leg identified (>= 3 tenors) — C20;
      6. SPX leg identified (>= 4 expiries, >= 30 options, put wing present
         in >= 3 expiries) — C22, the same failure mode on the other leg.

    Returns (ok: bool, reason: str).
    """
    if isinstance(params, dict) and isinstance(params.get("params"), dict):
        res = params
        legacy = calibration_is_legacy(res)
        if legacy and not allow_legacy:
            return False, ("legacy calibration (model_version < 2): fitted the "
                           "VIX spot indices as futures with ATM-call anchors — "
                           "not comparable with current fits")
        ll = res.get("leg_losses", res.get("losses", {})) or {}
        snap_session = res.get("spx_snapshot_session") or _session_of(res.get("spx_snapshot_date"))
        if legacy and res.get("spx_snapshot_date") is None:
            return False, "surface provenance unknown (legacy fit without spx_snapshot_date)"
        return is_acceptable_calibration(
            res["params"], ll.get("spx_iv_rmse"), res.get("n_vix_tenors"),
            n_spx_expiries=res.get("n_spx_expiries"),
            n_spx_options=res.get("n_spx_options"),
            n_put_wing_expiries=res.get("n_put_wing_expiries"),
            spx_snapshot_session=snap_session,
            spot_date=res.get("spot_date"),
            model_version=None if legacy else res.get("model_version"),
        )

    sigma = float(params.get("sigma", 0.0))
    rho   = float(params.get("rho", 0.0))
    kappa = float(params.get("kappa", 0.0))
    k_lo, k_hi = HESTON_BOUNDS["kappa"]

    # Provenance first: a stale surface or an old model version invalidates
    # every other number, so it is the reason reported (a months-old chain
    # also has few unexpired slices left, which would otherwise be named).
    if spot_date is not None and spx_snapshot_session is not None \
            and str(spx_snapshot_session)[:10] != str(spot_date)[:10]:
        return False, (f"SPX surface is stale (quotes from session "
                       f"{str(spx_snapshot_session)[:10]}, spot session {str(spot_date)[:10]})")
    if model_version is not None and int(model_version) < MODEL_VERSION:
        return False, f"model_version {model_version} < {MODEL_VERSION}"
    if sigma < 0.05:
        return False, f"vol-of-vol collapsed (sigma={sigma:.4f} < 0.05)"
    if abs(rho) < 0.10:
        return False, f"no skew (|rho|={abs(rho):.4f} < 0.10)"
    if kappa <= 1.1 * k_lo or kappa >= 0.995 * k_hi:
        return False, f"kappa pinned at bound (kappa={kappa:.2f})"
    for name in ("v0", "theta"):
        if name in params and float(params[name]) <= 2.0 * HESTON_BOUNDS[name][0]:
            return False, (f"{name} at its floor ({name}={float(params[name]):.2e} "
                           f"<= 2x {HESTON_BOUNDS[name][0]:g})")
    if spx_iv_rmse is not None and spx_iv_rmse > _MAX_SPX_RMSE:
        return False, f"SPX RMSE too high ({spx_iv_rmse:.2f}vp > {_MAX_SPX_RMSE:g})"
    if n_vix_tenors is not None and n_vix_tenors < _MIN_VIX_TENORS:
        return False, (f"VIX leg under-identified ({n_vix_tenors} tenor(s) "
                       f"< {_MIN_VIX_TENORS}) — joint fit degenerates to SPX-only")
    if n_spx_expiries is not None and n_spx_expiries < _MIN_SPX_EXPIRIES:
        return False, (f"SPX term structure under-identified ({n_spx_expiries} "
                       f"expiry(ies) < {_MIN_SPX_EXPIRIES})")
    if n_spx_options is not None and n_spx_options < _MIN_SPX_OPTIONS:
        return False, (f"SPX smile under-identified ({n_spx_options} options "
                       f"< {_MIN_SPX_OPTIONS})")
    if n_put_wing_expiries is not None and n_put_wing_expiries < _MIN_PUT_WING_EXPIRIES:
        return False, (f"SPX put wing missing ({n_put_wing_expiries} expiries with "
                       f">= {_MIN_WING_PUTS_PER_EXPIRY} puts at K/F < {_PUT_WING_KF} "
                       f"< {_MIN_PUT_WING_EXPIRIES})")
    return True, "ok"


# ── VIX option pricing under Heston (CIR transition law) ──────────────────────

def _vix_affine(kappa: float, theta: float) -> tuple:
    """VIX_T^2 (decimal) = a + b * v_T with b = (1 - e^{-kappa tau})/(kappa tau)."""
    b = float(_one_minus_exp_over(kappa * _VIX_WINDOW))
    return theta * (1.0 - b), b


def _cir_ncx2(T: float, kappa: float, theta: float, sigma: float, v0: float) -> tuple:
    """2c v_T ~ ncx2(df=d, nc=lam) for the CIR transition law."""
    c = 2.0 * kappa / (sigma ** 2 * -np.expm1(-kappa * T))
    d = 4.0 * kappa * theta / sigma ** 2
    lam = 2.0 * c * v0 * np.exp(-kappa * T)
    return c, d, lam


def _vix_payoff_integral(K_vix: float, T: float, kappa: float, theta: float,
                         sigma: float, v0: float, n_grid: int, put: bool) -> float:
    """Undiscounted E[(VIX_T - K)^+] (put=False) or E[(K - VIX_T)^+] (put=True).

    Integration by parts against the CIR law, with g(v) = 100 sqrt(a + b v):
      E[(g(V) - K)^+] = (g(0) - K)^+ + int_{max(v*,0)}^inf g'(v) SF(v) dv
      E[(K - g(V))^+] = int_0^{v*} g'(v) CDF(v) dv          (v* = g^{-1}(K))
    SF/CDF are bounded and continuous even where the density is singular at
    v = 0 (d = 4 kappa theta / sigma^2 < 2), which the old density-times-
    payoff trapezoid on a linear grid under-resolved. The grid is quadratic
    in u (v = v_lo + (v_hi - v_lo) u^2), dense near the lower end.
    """
    a, b = _vix_affine(kappa, theta)
    c, d, lam = _cir_ncx2(T, kappa, theta, sigma, v0)
    g0 = 100.0 * math.sqrt(max(a, 0.0))
    v_star = ((K_vix / 100.0) ** 2 - a) / b
    mean = (d + lam) / (2.0 * c)
    std = math.sqrt(2.0 * (d + 2.0 * lam)) / (2.0 * c)
    v_top = mean + 14.0 * std
    u = np.linspace(0.0, 1.0, n_grid) ** 2
    if put:
        if v_star <= 0:
            return 0.0
        v = v_star * u
        F_v = stats.ncx2.cdf(2.0 * c * v, d, lam)
    else:
        v_lo = max(v_star, 0.0)
        if v_lo >= v_top:
            return max(g0 - K_vix, 0.0)
        v = v_lo + (v_top - v_lo) * u
        F_v = stats.ncx2.sf(2.0 * c * v, d, lam)
    gprime = 50.0 * b / np.sqrt(a + b * v)
    integral = float(np.trapezoid(gprime * F_v, v))
    if put:
        return max(integral, 0.0)
    return max(g0 - K_vix, 0.0) + max(integral, 0.0)


def heston_vix_call_price(
    K_vix: float,
    T: float,
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    r: float = 0.045,
    n_grid: int = 4000,
) -> float:
    """
    VIX call under Heston: e^{-rT} E[max(VIX_T - K, 0)], VIX_T = 100 sqrt(a + b v_T),
    integrated against the exact CIR transition law (see _vix_payoff_integral).

    Note VIX options are written on the VX future of the same expiry; this
    price is the risk-neutral expectation and is consistent with
    heston_vix_future_exact (put-call parity holds with that forward).

    T <= 0 returns the intrinsic value against the model's CURRENT spot VIX
    (heston_vix_index, which depends on v0) — the old code used
    100*sqrt(theta).
    """
    if T <= 0:
        return max(heston_vix_index(kappa, theta, v0) - K_vix, 0.0)
    if sigma < 1e-8:
        a, b = _vix_affine(kappa, theta)
        ev_T = theta + (v0 - theta) * np.exp(-kappa * T)
        vix_det = 100.0 * np.sqrt(max(a + b * ev_T, 0.0))
        return max(vix_det - K_vix, 0.0) * np.exp(-r * T)
    return _vix_payoff_integral(K_vix, T, kappa, theta, sigma, v0, n_grid, put=False) \
        * math.exp(-r * T)


def heston_vix_put_price(
    K_vix: float, T: float, kappa: float, theta: float,
    sigma: float, v0: float, r: float = 0.045, n_grid: int = 4000,
) -> float:
    """VIX put, integrated against the same CIR law as the call.

    Equivalent to parity with the exact forward E[VIX_T]
    (heston_vix_future_exact); the pre-C22 version used sqrt(E[VIX_T^2]),
    which is above E[VIX_T] by the Jensen gap (0.7-1.0 VIX points at 1-2
    months) and produced negative put prices.
    """
    if T <= 0:
        return max(K_vix - heston_vix_index(kappa, theta, v0), 0.0)
    if sigma < 1e-8:
        a, b = _vix_affine(kappa, theta)
        ev_T = theta + (v0 - theta) * np.exp(-kappa * T)
        vix_det = 100.0 * np.sqrt(max(a + b * ev_T, 0.0))
        return max(K_vix - vix_det, 0.0) * np.exp(-r * T)
    return _vix_payoff_integral(K_vix, T, kappa, theta, sigma, v0, n_grid, put=True) \
        * math.exp(-r * T)


# ── Residual model, multi-start TRF and identifiability (C23) ────────────────
#
# The joint loss is a weighted sum of squares, so it is minimised as a
# nonlinear least-squares problem: scipy.optimize.least_squares
# (trust-region reflective, bounds handled by reflection) on the residual
# vector r(p) with sum(r**2) == joint_loss(p), from several Sobol starts plus
# the previous session's accepted fit. Measured on the five committed
# model_version 2 sessions in scripts/calibration_diagnostics.py.

_TRF_N_STARTS = 8
_TRF_MAX_NFEV = 200            # residual evaluations per start (Jacobians not counted)
_TRF_TOL = 1e-12               # ftol = xtol = gtol
_BASIN_LOSS_RTOL = 1e-6        # two starts share a basin if their losses agree to this
_PIN_REL_TOL = 1e-3            # same rule as count_pinned_params


class HestonJointResiduals:
    """Residual vector of the joint loss, with an analytic Jacobian.

    sum(self(p)**2) equals JointCalibrator.joint_loss(p) inside the box:
      SPX   sqrt(w1/n_spx) * (model - market) / vega   per anchor
      VIX   sqrt(w2/n_vix) * (model - market) / 100    per spot index
      VIXo  sqrt(w3/n_vo)  * (model - market) / vega   per VIX option (w3 > 0)
      soft Feller sqrt(50) * max(0, sigma^2 - 2 kappa theta)
    Only arrays and floats are stored, so the object pickles cheaply into
    scipy's worker processes (differential_evolution(workers=n)).

    jac(p): d r / d p from heston_call_batch_jac on the grid the residual
    evaluation at the same p converged on (cached), the closed-form VIX index
    derivative and the Feller term; a VIX-options block (w3 > 0, research
    only) is differenced numerically.
    """

    names = HESTON_PARAM_NAMES

    def __init__(self, groups, vix_tenors, vix_market, w_spx: float, w_vix: float,
                 w_vix_opt: float = 0.0, feller: str = "off", vix_options=None,
                 r_ref: float = 0.0):
        self.groups = [tuple(g) for g in groups]
        self.vix_tenors = np.asarray(vix_tenors, dtype=float)
        self.vix_market = np.asarray(vix_market, dtype=float)
        n_spx = sum(len(g[3]) for g in self.groups)
        self.c_spx = math.sqrt(w_spx / n_spx) if (w_spx > 0 and n_spx) else 0.0
        self.c_vix = (math.sqrt(w_vix / len(self.vix_tenors))
                      if (w_vix > 0 and len(self.vix_tenors)) else 0.0)
        self.feller = feller
        self.r_ref = float(r_ref)
        self.vix_opt = None
        self.c_vo = 0.0
        if w_vix_opt > 0 and vix_options is not None and len(vix_options):
            vo = vix_options
            self.vix_opt = (vo["strike"].to_numpy(float), vo["time_to_expiry"].to_numpy(float),
                            (vo["right"] == "C").to_numpy(), vo["market_price_f"].to_numpy(float),
                            vo["bs_vega"].to_numpy(float))
            self.c_vo = math.sqrt(w_vix_opt / len(vo))
        self.n_calls = 0
        self.n_jac = 0
        self._grid_cache = None

    # -- residuals -----------------------------------------------------------
    def _spx(self, p, keep_grids: bool):
        out, grids = [], []
        for T, F, D, K, is_put, mkt, vega in self.groups:
            r_T = -math.log(D) / T
            calls, grid = heston_call_batch_grid(F, K, T, r_T, r_T, *p)
            model = calls.copy()
            model[is_put] = calls[is_put] - D * (F - K[is_put])
            out.append(self.c_spx * (model - mkt) / vega)
            grids.append(grid)
        if keep_grids:
            self._grid_cache = (np.asarray(p, float).tobytes(), grids)
        return out

    def _vix_opt_resid(self, p):
        kappa, theta, sigma, _, v0 = p
        K, T, is_call, mkt, vega = self.vix_opt
        model = np.array([(heston_vix_call_price if c else heston_vix_put_price)(
            k, t, kappa, theta, sigma, v0, self.r_ref, n_grid=800)
            for k, t, c in zip(K, T, is_call)])
        return self.c_vo * (model - mkt) / vega

    def __call__(self, p) -> np.ndarray:
        self.n_calls += 1
        p = np.asarray(p, dtype=float)
        kappa, theta, sigma, rho, v0 = p
        parts = self._spx(p, keep_grids=True) if self.c_spx else []
        if self.c_vix:
            model = heston_vix_term_index(kappa, theta, sigma, v0, self.vix_tenors)
            parts.append(self.c_vix * (np.atleast_1d(model) - self.vix_market) / 100.0)
        if self.vix_opt is not None:
            parts.append(self._vix_opt_resid(p))
        if self.feller == "soft":
            parts.append(np.array([math.sqrt(_FELLER_PENALTY)
                                   * max(0.0, sigma * sigma - 2.0 * kappa * theta)]))
        return np.concatenate(parts) if parts else np.zeros(0)

    def loss(self, p) -> float:
        """sum(r**2), 1e6 outside HESTON_BOUNDS (the DE objective)."""
        if not JointCalibrator._in_bounds(*np.asarray(p, dtype=float)):
            return 1e6
        r = self(p)
        return float(r @ r)

    # -- Jacobian ------------------------------------------------------------
    def jac(self, p) -> np.ndarray:
        self.n_jac += 1
        p = np.asarray(p, dtype=float)
        kappa, theta, sigma, rho, v0 = p
        key = p.tobytes()
        grids = (self._grid_cache[1] if self._grid_cache is not None
                 and self._grid_cache[0] == key else [None] * len(self.groups))
        rows = []
        if self.c_spx:
            for (T, F, D, K, _, _, vega), grid in zip(self.groups, grids):
                r_T = -math.log(D) / T
                _, J = heston_call_batch_jac(F, K, T, r_T, r_T, *p, grid=grid)
                rows.append(self.c_spx * J / vega[:, None])
        if self.c_vix:
            _, Jv = heston_vix_term_index_jac(kappa, theta, sigma, v0, self.vix_tenors)
            rows.append(self.c_vix * Jv / 100.0)
        if self.vix_opt is not None:
            base = self._vix_opt_resid(p)
            Jo = np.zeros((len(base), 5))
            for j in range(5):
                h = 1e-6 * max(abs(p[j]), 1e-3)
                pp = p.copy()
                pp[j] += h
                Jo[:, j] = (self._vix_opt_resid(pp) - base) / h
            rows.append(Jo)
        if self.feller == "soft":
            row = np.zeros((1, 5))
            if sigma * sigma - 2.0 * kappa * theta > 0:
                row[0] = math.sqrt(_FELLER_PENALTY) * np.array(
                    [-2.0 * theta, -2.0 * kappa, 2.0 * sigma, 0.0, 0.0])
            rows.append(row)
        return np.vstack(rows) if rows else np.zeros((0, 5))


def _box_arrays(bounds: dict, names: Sequence[str]) -> tuple:
    return (np.array([bounds[n][0] for n in names], dtype=float),
            np.array([bounds[n][1] for n in names], dtype=float))


def sobol_starts(n: int, start_box: dict, lo: np.ndarray, hi: np.ndarray,
                 names: Sequence[str], seed: int = RANDOM_SEED) -> np.ndarray:
    """n scrambled-Sobol start points in start_box (log or linear per
    parameter), each coordinate clipped to the interior of [lo, hi]."""
    if n <= 0:
        return np.zeros((0, len(names)))
    m = int(math.ceil(math.log2(max(n, 1))))
    u = qmc.Sobol(len(names), scramble=True, seed=seed).random_base2(m)[:n]
    out = np.empty_like(u)
    for j, name in enumerate(names):
        a, b, scale = start_box[name]
        a, b = max(a, lo[j]), min(b, hi[j])
        if b < a:                                   # start box outside the active box
            a = b = 0.5 * (lo[j] + hi[j])
        if scale == "log" and a > 0:
            out[:, j] = np.exp(np.log(a) + u[:, j] * (np.log(b) - np.log(a)))
        else:
            out[:, j] = a + u[:, j] * (b - a)
    margin = 1e-6 * (hi - lo)
    return np.clip(out, lo + margin, hi - margin)


def _group_basins(losses: Sequence[float], rtol: float = _BASIN_LOSS_RTOL) -> list:
    """Sorted loss levels -> list of index groups whose losses agree to rtol
    (relative to the group's lowest loss). Starts that end on the same flat
    valley at different parameters (kappa is a flat direction) count once."""
    order = np.argsort(np.asarray(losses, dtype=float), kind="stable")
    groups = []
    for i in order:
        if groups and losses[i] <= losses[groups[-1][0]] * (1.0 + rtol) + 1e-300:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def multistart_trf(fun: Callable, jac, starts: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   labels: Optional[Sequence[str]] = None, *, max_nfev: int = _TRF_MAX_NFEV,
                   tol: float = _TRF_TOL, x_scale="jac") -> tuple:
    """least_squares(method='trf') from every start; returns (best, report).

    best: the OptimizeResult with the lowest sum of squares (ties keep the
    earlier start, so a warm start listed first wins a tie). report: one row
    per start (label, x0, x, loss = sum r^2, nfev, njev, status, seconds)
    plus the basin grouping. jac=None differences numerically ('2-point').

    A start whose residuals are not finite at x0 (least_squares raises) or
    whose optimum is not finite is recorded as failed (status "failed",
    loss None) and the others still run, so one bad Sobol corner cannot
    cost the daily fit; RuntimeError only if every start fails.
    """
    labels = list(labels) if labels is not None else [f"start{i}" for i in range(len(starts))]
    runs, best, best_loss = [], None, np.inf
    for start, label in zip(starts, labels):
        x0 = np.clip(np.asarray(start, dtype=float), lo, hi)
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # Lewis cap at extreme trial points
                sol = optimize.least_squares(
                    fun, x0, jac=jac if jac is not None else "2-point", bounds=(lo, hi),
                    method="trf", x_scale=x_scale, ftol=tol, xtol=tol, gtol=tol,
                    max_nfev=max_nfev)
            loss = float(2.0 * sol.cost)
            if not (np.isfinite(loss) and np.all(np.isfinite(sol.x))):
                raise FloatingPointError(f"non-finite optimum (loss {loss})")
        except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError) as exc:
            logger.warning("     TRF start %s failed: %s", label, exc)
            runs.append({"label": label, "x0": [float(v) for v in x0], "x": None,
                         "loss": None, "nfev": None, "njev": None, "status": "failed",
                         "message": f"{type(exc).__name__}: {exc}",
                         "seconds": time.perf_counter() - t0})
            continue
        runs.append({"label": label, "x0": [float(v) for v in x0],
                     "x": [float(v) for v in sol.x], "loss": loss,
                     "nfev": int(sol.nfev), "njev": int(sol.njev or 0),
                     "status": int(sol.status), "message": str(sol.message),
                     "seconds": time.perf_counter() - t0})
        if loss < best_loss:
            best, best_loss = sol, loss
    if best is None:
        raise RuntimeError(f"all {len(runs)} TRF start(s) failed: "
                           + "; ".join(f"{r['label']}: {r['message']}" for r in runs))
    ok = [i for i, r in enumerate(runs) if r["loss"] is not None]
    groups = [[ok[i] for i in g] for g in _group_basins([runs[i]["loss"] for i in ok])]
    best_group = groups[0] if groups else []
    report = {
        "runs": runs,
        "n_starts": len(runs),
        "n_failed_starts": len(runs) - len(ok),
        "n_distinct_basins": len(groups),
        "n_starts_at_best": len(best_group),
        "basin_losses": [runs[g[0]]["loss"] for g in groups],
        "best_label": runs[best_group[0]]["label"] if best_group else None,
    }
    if len(best_group) > 1:
        X = np.array([runs[i]["x"] for i in best_group])
        report["best_basin_param_range"] = [float(v) for v in X.max(0) - X.min(0)]
    return best, report


def identifiability_report(J: np.ndarray, r: np.ndarray, x: np.ndarray,
                           names: Sequence[str], lo: np.ndarray, hi: np.ndarray,
                           rel_tol: float = _PIN_REL_TOL) -> dict:
    """Local identifiability of a least-squares fit from its Jacobian.

    * singular values of the relative-sensitivity Jacobian J diag(|x|)
      (d r / d ln p) and their condition number; cond(J^T J) is its square;
    * the direction of the smallest singular value (which parameter
      combination the data barely see);
    * Gauss-Newton standard errors s^2 (J_f^T J_f)^{-1}, s^2 = sum r^2 / (n - k),
      and the correlation matrix, over the k parameters NOT on a bound (a
      parameter at its box edge has no interior curvature-based error bar).

    The residuals are weighted pseudo-observations whose misfit is mostly
    model error, not independent noise, so the standard errors are a local
    curvature scale (how far a parameter moves before the loss changes by
    the average squared residual), not a sampling distribution.
    """
    J = np.asarray(J, dtype=float)
    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    names = list(names)
    scale = np.where(np.abs(x) > 0, np.abs(x), 1.0)
    Jr = J * scale[None, :]
    sv = np.linalg.svd(Jr, compute_uv=False)
    _, _, Vt = np.linalg.svd(Jr, full_matrices=False)
    weakest = Vt[-1] * np.sign(Vt[-1][np.argmax(np.abs(Vt[-1]))])
    on_bound = []
    for j, n in enumerate(names):
        span = (hi[j] - lo[j]) or 1.0
        if _at_bound(x[j], lo[j], span, rel_tol) or _at_bound(x[j], hi[j], span, rel_tol):
            on_bound.append(n)
    free = [j for j, n in enumerate(names) if n not in on_bound]
    n_obs, k = len(r), len(free)
    out = {
        "param_names": names,
        "n_residuals": int(n_obs),
        "rel_jacobian_singular_values": [float(v) for v in sv],
        "rel_jacobian_condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "rel_jtj_condition_number": float((sv[0] / sv[-1]) ** 2) if sv[-1] > 0 else float("inf"),
        "weakest_direction": {n: float(v) for n, v in zip(names, weakest)},
        "on_bound": on_bound,
        "free_params": [names[j] for j in free],
        "standard_errors": {n: None for n in names},
        "relative_standard_errors": {n: None for n in names},
        "correlation": None,
        "se_note": ("Gauss-Newton s^2 (J^T J)^-1 over the free parameters: a local "
                    "curvature scale, not a sampling distribution (the residuals are "
                    "model misfit, not iid noise)"),
    }
    if k == 0 or n_obs <= k:
        return out
    Jf = J[:, free]
    s2 = float(r @ r) / (n_obs - k)
    try:
        cov = s2 * np.linalg.inv(Jf.T @ Jf)
    except np.linalg.LinAlgError:
        return out
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(se, se)
    for i, j in enumerate(free):
        out["standard_errors"][names[j]] = float(se[i])
        out["relative_standard_errors"][names[j]] = (float(se[i] / abs(x[j]))
                                                     if x[j] != 0 else None)
    out["correlation"] = {"params": [names[j] for j in free],
                          "matrix": [[float(v) for v in row] for row in corr]}
    out["residual_variance"] = s2
    return out


def gauss_newton_profile_check(fun: Callable, x: np.ndarray, report: dict,
                               lo: np.ndarray, hi: np.ndarray,
                               steps: Sequence[float] = (-1.0, 1.0)) -> dict:
    """How well the Gauss-Newton standard errors describe the loss.

    For each free parameter, hold it at x_j + m SE_j and re-fit the other
    free parameters; return dSSR / (m^2 s^2) per step m. A quadratic loss
    with the Gauss-Newton Hessian gives exactly 1. Below 1 the loss rises
    more slowly than J^T J predicts (the dropped sum r_i Hess(r_i) term
    matters because the residuals are model misfit), so that SE is
    optimistic; above 1, conservative. Measured, not assumed.
    """
    names = report["param_names"]
    x = np.asarray(x, dtype=float)
    r0 = fun(x)
    base = float(r0 @ r0)
    s2 = report.get("residual_variance")
    free = [names.index(k) for k in report["free_params"]]
    out = {}
    if not s2:
        return out
    for i in free:
        se = report["standard_errors"][names[i]]
        others = [j for j in free if j != i]
        ratios = {}
        for m in steps:
            xi = x.copy()
            xi[i] = float(np.clip(x[i] + m * se, lo[i], hi[i]))

            def sub(y, xi=xi):
                z = xi.copy()
                z[others] = y
                return fun(z)
            sol = optimize.least_squares(sub, x[others], bounds=(lo[others], hi[others]),
                                         method="trf", x_scale="jac", ftol=1e-12, xtol=1e-12)
            eff = (xi[i] - x[i]) / se
            ratios[f"{m:+g}"] = float((2.0 * sol.cost - base) / (s2 * eff * eff))
        out[names[i]] = ratios
    return out


def previous_accepted_fit(spot_date: str, cal_dir=None) -> tuple:
    """(path, result) of the newest ACCEPTED model_version >= 2 joint Heston
    fit of a session strictly before spot_date, or (None, None).

    Strictly earlier, so a warm start never sees the session being fitted or
    a later one (refitting 2026-03-24 cannot start from 2026-03-27). Accepted
    = is_acceptable_calibration(result) (the gate the dashboard uses).
    """
    cal_dir = Path(cal_dir) if cal_dir is not None else Path(DATA_DIR) / "calibrations"
    if not cal_dir.is_dir():
        return None, None
    target = str(spot_date)[:10]
    for path in sorted(cal_dir.glob("joint_cal_*.pkl"), reverse=True):
        session = path.stem[len("joint_cal_"):]
        if session >= target:
            continue
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError,
                ImportError, ValueError, TypeError):   # unreadable: not a warm start
            continue
        if not isinstance(res, dict) or calibration_is_legacy(res):
            continue
        if str(res.get("spot_date", session))[:10] >= target:
            continue
        if is_acceptable_calibration(res)[0]:
            return path, res
    return None, None


# ── Main Class ────────────────────────────────────────────────────────────────

class JointCalibrator:
    """
    Joint SPX/VIX smile calibration engine for the Heston model.

    Loads market data once at construction, then minimises the weighted loss
    as a least-squares problem: multi-start trust-region reflective
    (scipy least_squares) from the previous session's accepted fit and
    scrambled-Sobol starts, analytic Jacobian (C23). The pre-C23 differential
    evolution + L-BFGS-B path remains as calibrate(optimizer="de") and as an
    optional global check (de_check=True).

    Attributes (set after calibrate())
    ------------------------------------
    params     : dict — calibrated Heston parameters
    result     : dict — the full calibration result (what save() persists)
    fit_time   : float — wall-clock seconds for calibration
    fit_cpu_time : float — CPU seconds of the optimisation (this process)
    losses     : dict — per-leg losses at the optimum
    """

    def __init__(
        self,
        as_of_date: str,
        r: Optional[float] = None,
        q: Optional[float] = None,
        w1: float = JOINT_W1,
        w2: float = JOINT_W2,
        w3: float = JOINT_W3,
        *,
        spx_chain=None,
        carry: str = "parity",
        feller: str = "off",
    ):
        """
        Parameters
        ----------
        as_of_date : 'YYYY-MM-DD' — the latest date whose data may be used;
                     the fit describes the last SPX session on or before it
        r          : reference continuously-compounded rate. None: the ^IRX
                     13-week bill for the spot session (db.get_tbill_rate,
                     stored cc). Used to discount VIX options and as the
                     fallback discount when an expiry's parity slope is
                     implausible; the SPX leg prices off parity D_T, F_T.
        q          : dividend yield, used only with carry="flat"
                     (default 0.013 there)
        w1, w2, w3 : leg weights (SPX IV, VIX index, VIX options); the active
                     ones are normalised to sum to 1
        spx_chain  : optional SPX chain to fit instead of the DB snapshot —
                     an option_chain.CleanChain, or a raw/stored chain frame
                     (cleaned here). The live pipeline passes the chain it
                     just downloaded so a rejected chain never touches the DB.
        carry      : "parity" (default; per-expiry F_T, D_T from put-call
                     parity) or "flat" (S e^{(r-q)T}, e^{-rT}; research only)
        feller     : "off" (default) or "soft" (penalty 50*viol^2)
        """
        if feller not in _FELLER_MODES:
            raise ValueError(f"feller must be one of {_FELLER_MODES}, got {feller!r}")
        if carry not in ("parity", "flat"):
            raise ValueError(f"carry must be 'parity' or 'flat', got {carry!r}")
        self.as_of_date = as_of_date
        self._r_arg, self._q_arg = r, q
        self.carry = carry
        self.feller = feller
        self.w1, self.w2, self.w3 = w1, w2, w3
        self._spx_chain_in = spx_chain

        self.params: Optional[dict] = None
        self.result = None
        self.fit_time: Optional[float] = None
        self.fit_cpu_time: Optional[float] = None
        self.losses: Optional[dict] = None
        self._n_evals = 0

        logger.info("JointCalibrator: loading market data for %s", as_of_date)
        self._load_market_data()

    # ── Market Data Preparation ────────────────────────────────────────────────

    def _load_market_data(self):
        """Load and clean all data legs from the database (or spx_chain)."""
        spx_hist = db.get_spx_ohlcv(as_of_date=self.as_of_date)
        if spx_hist.empty:
            raise ValueError(f"No SPX OHLCV data available for {self.as_of_date}")
        self._spx_hist = spx_hist
        self.S = float(spx_hist.iloc[-1]["close"])
        # The session this fit describes. Every other leg is pinned to it.
        self.spot_date = str(spx_hist.iloc[-1]["date"])[:10]
        logger.info("  SPX spot S = %.2f (session %s)", self.S, self.spot_date)

        if self._r_arg is not None:
            self.r_reference = float(self._r_arg)
            self.r_reference_source = "argument"
        else:
            self.r_reference, self.r_reference_source = self._reference_rate()
        # Back-compat attributes (scalar summaries; set from parity below)
        self.r = self.r_reference
        self.q = FLAT_CARRY_Q if self._q_arg is None else float(self._q_arg)

        self.spx_surface = self._prepare_spx_surface()
        logger.info("  SPX surface: %d anchors across %d expiries (%d puts at K/F<%.2f); "
                    "%d quotes out of sample",
                    len(self.spx_surface), self.spx_surface["expiry"].nunique(),
                    int(((self.spx_surface["right"] == "P")
                         & (self.spx_surface["strike"] / self.spx_surface["forward"]
                            < _PUT_WING_KF)).sum()),
                    _PUT_WING_KF, len(self.spx_surface_full) - len(self.spx_surface))

        self.vix_ts = self._prepare_vix_term_structure()
        logger.info("  VIX term structure: %d spot indices", len(self.vix_ts))
        self.vx_futures = self._prepare_vx_futures()

        self.vix_options = self._prepare_vix_options()
        logger.info("  VIX options: %d liquid contracts on the spot session",
                    len(self.vix_options))

        # Normalise the ACTIVE legs to sum to one, independent of how many
        # rows a disabled leg happens to hold. (Pre-C22 the 0.5/0.6 weights
        # stayed un-normalised whenever >= 5 stale VIX options existed, which
        # changed the effective Feller-penalty strength by ~10%.)
        if self.w3 > 0 and len(self.vix_options) < 5:
            logger.warning("  VIX options too sparse (%d) — setting w3=0",
                           len(self.vix_options))
            self.w3 = 0.0
        total = self.w1 + self.w2 + self.w3
        if total > 0:
            self.w1, self.w2, self.w3 = self.w1 / total, self.w2 / total, self.w3 / total

        if len(self.vix_ts) < _MIN_VIX_TENORS:
            logger.warning(
                "  VIX term structure has only %d tenor(s) (< %d): the VIX leg "
                "cannot identify a curve and the joint fit degenerates toward "
                "an SPX-only calibration", len(self.vix_ts), _MIN_VIX_TENORS)

    def _reference_rate(self) -> tuple:
        """(rate, label) of the ^IRX 13-week bill for the spot session.

        tbill_rates.rate is ALREADY continuously compounded (the data layer
        converts the ^IRX bank-discount yield on download, and migrated the
        old rows), so it is used as is. The first C22 version converted it a
        second time, which put the reference rate ~7 bp too high (2026-06-18:
        3.80% instead of 3.73% cc) and labelled the cc rate a discount yield.
        """
        r = float(db.get_tbill_rate(self.spot_date))
        label = f"^IRX 13-week bill, {r:.4%} cc"
        try:
            start = (pd.Timestamp(self.spot_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            ser = db.get_tbill_rates_series(self.spot_date, start_date=start)
        except Exception:                      # label detail only
            ser = None
        if ser is not None and len(ser) and np.isclose(float(ser["rate"].iloc[-1]), r,
                                                       rtol=0.0, atol=1e-12):
            row = ser.iloc[-1]
            d = row.get("discount_yield")
            label = (f"^IRX 13-week bill {str(row['date'])[:10]}: "
                     + (f"{float(d):.4%} bank discount = " if d is not None and pd.notna(d) else "")
                     + f"{r:.4%} cc")
        return r, label

    def _session_for(self, snapshot_date) -> Optional[str]:
        """Session whose closing book a snapshot taken on ``snapshot_date``
        holds: the last SPX session in the DB on or before that date (a
        Sunday 05-31 snapshot holds Friday 05-29's quotes)."""
        snap = str(snapshot_date)[:10]
        dates = self._spx_hist["date"].astype(str).str[:10]
        prior = dates[dates <= snap]
        if len(prior):
            return str(prior.iloc[-1])
        return oc.session_on_or_before(snap).isoformat()

    def _clean_chain(self) -> "oc.CleanChain":
        src = self._spx_chain_in
        if isinstance(src, oc.CleanChain):
            sess = str(src.diagnostics.get("session", self.spot_date))
            self.spx_snapshot_date = sess
            self.spx_snapshot_session = sess
            return src
        if src is not None:
            raw = src
            snap = (str(raw["snapshot_date"].iloc[0])[:10]
                    if "snapshot_date" in raw.columns and len(raw) else self.spot_date)
        else:
            raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="SPX")
            if raw.empty:
                self.spx_snapshot_date = None
                self.spx_snapshot_session = None
                raise ValueError(f"no usable SPX anchors for {self.as_of_date}: "
                                 "no SPX option snapshot on or before this date")
            snap = str(raw["snapshot_date"].iloc[0])[:10]
        # get_options_surface returns the most recent snapshot <= as_of_date,
        # which can be months stale; record which one this fit is built on
        # and which session's quotes it holds.
        self.spx_snapshot_date = snap
        self.spx_snapshot_session = self._session_for(snap)
        if self.spx_snapshot_session != self.spot_date:
            logger.warning("  SPX snapshot %s holds session %s quotes but spot session is %s "
                           "— the fit will be rejected as stale",
                           snap, self.spx_snapshot_session, self.spot_date)
        sess_close = self._spx_hist.loc[
            self._spx_hist["date"].astype(str).str[:10] == self.spx_snapshot_session, "close"]
        S_sess = float(sess_close.iloc[-1]) if len(sess_close) else self.S
        return oc.clean_option_chain(raw, self.spx_snapshot_session, S_sess,
                                     r_ref=self.r_reference)

    def _prepare_spx_surface(self) -> pd.DataFrame:
        """
        Clean the SPX chain and select the calibration anchors.

        Design (C22):
          - quotes: two-sided, parity F_T / D_T per expiry, own Black-76 IVs
            (option_chain.clean_option_chain); T from option_chain.t_origin
            (session close, or the quote time of a GTH book)
          - surface: OTM against F_T, 7d <= T <= 2y, 0.70 <= K/F <= 1.30,
            mid >= $0.50, 1% <= IV <= 200% (kept in self.spx_surface_full
            for the out-of-sample RMSE and validation)
          - expiries: one per target tenor (14..365d), unique
          - anchors per expiry: the strike nearest each point of
            z = ln(K/F)/(sigma_atm sqrt T) in linspace(-2.5, 1.5, 9), plus
            puts at K/F < 0.97 topped up to 3 where such quotes exist
        """
        cc = self._clean_chain()
        self.spx_chain_diagnostics = dict(cc.diagnostics)
        if not cc.ok:
            raise ValueError(f"no usable SPX anchors for {self.as_of_date}: {cc.reason}")
        q = cc.quotes.copy()

        if self.carry == "flat":
            T = q["time_to_expiry"].to_numpy(float)
            q["forward"] = self.S * np.exp((self.r_reference - self.q) * T)
            q["discount"] = np.exp(-self.r_reference * T)
            q["r_T"], q["q_T"] = self.r_reference, self.q
            q["implied_vol"] = oc.black76_implied_vol(
                q["mid_price"], q["forward"], q["strike"], q["time_to_expiry"],
                q["discount"], (q["right"] == "C").to_numpy())
            q["otm"] = np.where(q["right"] == "C", q["strike"] >= q["forward"],
                                q["strike"] < q["forward"])
            q["log_moneyness"] = np.log(q["strike"] / q["forward"])
        self.spx_carry = cc.carry

        kf = q["strike"] / q["forward"]
        full = q[q["otm"].astype(bool)
                 & (q["time_to_expiry"] >= _MIN_T_DAYS / 365.0)
                 & (q["time_to_expiry"] <= _MAX_T_YEARS)
                 & kf.between(_MONEYNESS_LO, _MONEYNESS_HI)
                 & (q["mid_price"] > _MIN_PRICE)
                 & q["implied_vol"].between(_MIN_IV, _MAX_IV)].copy()
        full["bs_vega"] = np.maximum(oc.black76_vega(
            full["forward"], full["strike"], full["time_to_expiry"],
            full["discount"], full["implied_vol"]), _MIN_VEGA)
        full["market_price_f"] = full["mid_price"].astype(float)
        self.spx_surface_full = full.sort_values(
            ["time_to_expiry", "strike"]).reset_index(drop=True)
        if full.empty:
            raise ValueError(f"no usable SPX anchors for {self.as_of_date}: "
                             f"{cc.diagnostics.get('n_raw', 0)} raw rows, none OTM "
                             "inside the surface filters")

        chosen = oc.select_calibration_expiries(
            full["expiry"].unique(), self.spx_snapshot_session or self.spot_date)
        rows = []
        for exp in chosen:
            g = self.spx_surface_full[self.spx_surface_full["expiry"] == exp]
            picked = self._anchor_rows(g)
            if len(picked) >= 2:
                rows.append(picked)
        if not rows:
            raise ValueError(f"no usable SPX anchors for {self.as_of_date}: no expiry "
                             f"within 50% of any target tenor ({len(full)} OTM quotes)")
        cal_df = pd.concat(rows, ignore_index=True)
        return cal_df.sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)

    @staticmethod
    def _anchor_rows(g: pd.DataFrame) -> pd.DataFrame:
        """Anchors for one expiry on the standardised forward-moneyness grid."""
        g = g.sort_values("log_moneyness").reset_index(drop=True)
        T = float(g["time_to_expiry"].iloc[0])
        k = g["log_moneyness"].to_numpy(float)
        iv = g["implied_vol"].to_numpy(float)
        sigma_atm = float(np.interp(0.0, k, iv))
        z = k / (sigma_atm * math.sqrt(T))
        g = g.assign(z=z)
        used = []
        for zt in _ANCHOR_Z_GRID:
            dist = np.abs(z - zt)
            dist[used] = np.inf
            i = int(np.argmin(dist))
            if dist[i] <= _ANCHOR_Z_TOL:
                used.append(i)
        # Put wing guarantee: >= 3 puts below K/F = 0.97 wherever they exist
        wing = np.where((g["right"] == "P").to_numpy()
                        & (np.exp(k) < _PUT_WING_KF))[0]
        while (sum(1 for i in used if i in set(wing)) < _MIN_WING_PUTS_PER_EXPIRY
               and len(set(wing) - set(used)) > 0):
            cand = [i for i in wing if i not in used]
            # the candidate farthest (in z) from every already-chosen anchor
            best = max(cand, key=lambda i: min(abs(z[i] - z[j]) for j in used) if used else 0)
            used.append(best)
        return g.iloc[sorted(used)].copy()

    def _prepare_vix_term_structure(self) -> pd.DataFrame:
        """
        The Cboe spot VIX indices on the SPX spot session.

        ^VIX9D/^VIX/^VIX3M/^VIX6M are SPOT-starting variance-swap rates over
        9/30/93/184 calendar days (config.VIX_INDEX_TENOR_DAYS) — not futures.
        A missing spot-session row yields an empty leg, which the >= 3-tenor
        gate rejects (C20).

        Returns
        -------
        DataFrame with columns [ticker, tenor_years, market_price]
        """
        empty = pd.DataFrame(columns=["ticker", "tenor_years", "market_price"])
        self.vvix_market = None
        ts_wide = db.get_vix_term_structure_wide(as_of_date=self.as_of_date)
        if ts_wide.empty:
            return empty
        spot = getattr(self, "spot_date", None)
        if spot is not None and "date" in ts_wide.columns:
            on_spot = ts_wide[ts_wide["date"].astype(str).str[:10] == spot]
            if on_spot.empty:
                logger.warning("  VIX term structure has no row for spot session %s "
                               "(latest available %s) -- VIX leg left empty",
                               spot, str(ts_wide["date"].iloc[-1])[:10])
                return empty
            latest = on_spot.iloc[-1]
        else:
            latest = ts_wide.iloc[-1]
        if "^VVIX" in latest and pd.notna(latest["^VVIX"]) and latest["^VVIX"] > 0:
            self.vvix_market = float(latest["^VVIX"])     # out-of-sample check only
        rows = []
        for col, tenor in _VIX_TS_TENORS.items():
            if col in latest and pd.notna(latest[col]) and latest[col] > 0:
                rows.append({"ticker": col, "tenor_years": tenor,
                             "market_price": float(latest[col])})
        if not rows:
            return empty
        return pd.DataFrame(rows).sort_values("tenor_years").reset_index(drop=True)

    def _prepare_vx_futures(self) -> pd.DataFrame:
        """True VX futures on the spot session, if the DB has any
        (vix_futures_daily), at their daily SETTLEMENT (``price`` =
        COALESCE(settle, close)): the last trade can be hours stale on a thin
        back month, and a contract with no trade that day has no close at all.
        Scored out of sample only; empty table -> no-op."""
        cols = ["expiry_month", "price", "days_to_expiry"]
        try:
            fut = db.get_vix_futures_curve(self.spot_date)
        except Exception as exc:          # table absent in a minimal DB
            logger.debug("  VX futures unavailable: %s", exc)
            return pd.DataFrame(columns=cols)
        if fut is None or fut.empty:
            return pd.DataFrame(columns=cols)
        if "price" not in fut.columns:    # a caller-supplied frame
            settle = fut["settle"] if "settle" in fut.columns else pd.Series(np.nan, index=fut.index)
            fut = fut.assign(price=settle.fillna(fut["close"]) if "close" in fut.columns else settle)
        fut = fut[(fut["days_to_expiry"] > 0) & (fut["price"] > 0)]
        return fut.reset_index(drop=True)

    def _prepare_vix_options(self) -> pd.DataFrame:
        """
        VIX options for the (disabled) third leg — read only when the chain's
        quote session is the spot session (pre-C22 one June chain was used
        for every fit with its June T; the DB now holds same-session VIX
        chains for 2026-03-24, 03-27, 05-29 and 06-18).

        Per expiry the VX forward F_T is implied from put-call parity of the
        two-sided quotes (VIX options are written on the future), and the mid
        IV, Black-76 vega and moneyness are measured against F_T — not spot
        VIX. T runs to the 09:30 ET (AM, SOQ) settlement. Subsample: 2
        expiries near 30/60d x 7 strikes evenly spaced in ln(K/F) on
        [ln 0.75, ln 1.40].
        """
        try:
            raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="VIX")
        except Exception:
            return pd.DataFrame()
        if raw.empty:
            return pd.DataFrame()
        snap = str(raw["snapshot_date"].iloc[0])[:10]
        session = self._session_for(snap)
        if session != self.spot_date:
            logger.info("  VIX options snapshot %s (session %s) is not the spot session "
                        "%s — VIX options leg left empty", snap, session, self.spot_date)
            return pd.DataFrame()
        df = raw.copy()
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")
        df["time_to_expiry"] = [oc.year_fraction(session, e, "AM") for e in df["expiry"]]
        df = df[(df["time_to_expiry"] >= 7.0 / 365.0) & (df["time_to_expiry"] <= 91.0 / 365.0)]
        df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])].copy()
        df["mid_price"] = 0.5 * (df["bid"] + df["ask"])
        selected = []
        for target_T in (30.0 / 365.0, 60.0 / 365.0):
            exps = df.groupby("expiry")["time_to_expiry"].first()
            if exps.empty:
                break
            best = (exps - target_T).abs().idxmin()
            T = float(exps[best])
            if abs(T - target_T) > 0.7 * target_T:
                continue
            g = df[df["expiry"] == best]
            c = g[g["right"] == "C"].groupby("strike")["mid_price"].mean()
            p = g[g["right"] == "P"].groupby("strike")["mid_price"].mean()
            common = c.index.intersection(p.index)
            if len(common) < 3:
                continue
            D = math.exp(-self.r_reference * T)
            F = float(np.median(np.asarray(common) + (c[common] - p[common]).to_numpy() / D))
            otm = g[((g["right"] == "C") & (g["strike"] >= F))
                    | ((g["right"] == "P") & (g["strike"] < F))]
            strikes = np.sort(otm["strike"].unique())
            if len(strikes) < 2:
                continue
            for tk in np.exp(np.linspace(math.log(0.75 * F), math.log(1.40 * F), 7)):
                K = float(strikes[np.argmin(np.abs(strikes - tk))])
                row = otm[otm["strike"] == K].iloc[0].copy()
                row["forward"], row["discount"] = F, D
                if not any((r_["expiry"] == best) and (r_["strike"] == K) for r_ in selected):
                    selected.append(row)
        if not selected:
            return pd.DataFrame()
        out = pd.DataFrame(selected).reset_index(drop=True)
        is_call = (out["right"] == "C").to_numpy()
        out["implied_vol"] = oc.black76_implied_vol(
            out["mid_price"], out["forward"], out["strike"], out["time_to_expiry"],
            out["discount"], is_call)
        out = out[(out["mid_price"] > 0.05) & out["implied_vol"].between(_MIN_IV, 5.0)].copy()
        out["bs_vega"] = np.maximum(oc.black76_vega(
            out["forward"], out["strike"], out["time_to_expiry"], out["discount"],
            out["implied_vol"]), 0.05)
        out["market_price_f"] = out["mid_price"].astype(float)
        return out.reset_index(drop=True)

    # ── Loss Function ──────────────────────────────────────────────────────────

    @staticmethod
    def _in_bounds(kappa, theta, sigma, rho, v0) -> bool:
        b = HESTON_BOUNDS
        return (b["kappa"][0] <= kappa <= b["kappa"][1]
                and b["theta"][0] <= theta <= b["theta"][1]
                and b["sigma"][0] <= sigma <= b["sigma"][1]
                and b["rho"][0] <= rho <= b["rho"][1]
                and b["v0"][0] <= v0 <= b["v0"][1])

    def _feller_penalty(self, kappa, theta, sigma) -> float:
        if self.feller != "soft":
            return 0.0
        viol = max(0.0, sigma ** 2 - 2.0 * kappa * theta)
        return _FELLER_PENALTY * viol ** 2

    def joint_loss(self, params_vec: np.ndarray) -> float:
        """
        Weighted joint loss at [kappa, theta, sigma, rho, v0].

        Returns 1e6 outside HESTON_BOUNDS. Adds the Feller penalty only when
        the calibrator was built with feller="soft".
        """
        self._n_evals += 1
        kappa, theta, sigma, rho, v0 = params_vec
        if not self._in_bounds(kappa, theta, sigma, rho, v0):
            return 1e6
        l1 = self._spx_leg(kappa, theta, sigma, rho, v0)
        l2 = self._vix_index_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        return (self.w1 * l1 + self.w2 * l2 + self.w3 * l3
                + self._feller_penalty(kappa, theta, sigma))

    def _spx_groups(self):
        """Per-expiry arrays of the anchor surface (built once)."""
        if getattr(self, "_groups_cache", None) is None or \
                self._groups_cache[0] is not self.spx_surface:
            groups = []
            for _, g in self.spx_surface.groupby("time_to_expiry"):
                groups.append((
                    float(g["time_to_expiry"].iloc[0]),
                    float(g["forward"].iloc[0]),
                    float(g["discount"].iloc[0]),
                    g["strike"].to_numpy(float),
                    (g["right"] == "P").to_numpy(),
                    g["market_price_f"].to_numpy(float),
                    g["bs_vega"].to_numpy(float),
                ))
            self._groups_cache = (self.spx_surface, groups)
        return self._groups_cache[1]

    @staticmethod
    def _model_prices(pricer, T, F, D, strikes, is_put, *model_params):
        """Model prices on (F, D): spot = F with r = q = -ln(D)/T reproduces
        exactly forward F and discount D in the S/r/q pricer interface."""
        r_T = -math.log(D) / T
        calls = pricer(F, strikes, T, r_T, r_T, *model_params)
        out = calls.copy()
        out[is_put] = calls[is_put] - D * (F - strikes[is_put])
        return out

    def _spx_leg(self, kappa: float, theta: float, sigma: float, rho: float,
                 v0: float) -> float:
        """
        SPX leg: vega-normalised MSE between Heston and market prices,
          (model_price - market_price) / vega ~ (model_iv - market_iv),
        in vol^2 units. Each slice is priced off its own parity forward and
        discount factor. Raises if there are no anchors (a weighted data leg
        must never silently return 0).
        """
        groups = self._spx_groups()
        if not groups:
            raise ValueError("SPX leg has no anchors")
        sse, n = 0.0, 0
        for T, F, D, K, is_put, mkt, vega in groups:
            model = self._model_prices(heston_call_batch, T, F, D, K, is_put,
                                       kappa, theta, sigma, rho, v0)
            err = (model - mkt) / vega
            sse += float(np.sum(err ** 2))
            n += len(err)
        return sse / max(n, 1)

    def _vix_index_leg(self, kappa: float, theta: float, sigma: float,
                       v0: float) -> float:
        """
        VIX term-structure (index) leg: MSE between the Heston spot indices
          100 sqrt(theta + (v0 - theta)(1 - e^{-kappa tau})/(kappa tau))
        at the Cboe windows and ^VIX9D/^VIX/^VIX3M/^VIX6M, in vol-decimal
        units (a 1-point miss contributes 1e-4).
        """
        if self.vix_ts.empty:
            return 0.0
        tenors = self.vix_ts["tenor_years"].to_numpy(float)
        market = self.vix_ts["market_price"].to_numpy(float)
        model = heston_vix_term_index(kappa, theta, sigma, v0, tenors)
        return float(np.mean(((model - market) / 100.0) ** 2))

    # Back-compat name: the leg was called the "futures" leg before C22.
    _vix_futures_leg = _vix_index_leg

    def _vix_options_leg(self, kappa: float, theta: float, sigma: float,
                         v0: float) -> float:
        """VIX options leg: vega-normalised MSE (disabled, w3 = 0)."""
        if self.vix_options.empty:
            return 0.0
        sse, n = 0.0, 0
        for _, row in self.vix_options.iterrows():
            K, T = float(row["strike"]), float(row["time_to_expiry"])
            fn = heston_vix_call_price if row["right"] == "C" else heston_vix_put_price
            model_px = fn(K, T, kappa, theta, sigma, v0, self.r_reference, n_grid=800)
            sse += ((model_px - float(row["market_price_f"])) / float(row["bs_vega"])) ** 2
            n += 1
        return sse / max(n, 1)

    # ── Diagnostics at a parameter vector ─────────────────────────────────────

    def _surface_errors(self, surface: pd.DataFrame, pricer, model_params) -> pd.DataFrame:
        """Exact model IVs (Black-76 inversion on F, D) on a quote surface."""
        parts = []
        for _, g in surface.groupby("time_to_expiry"):
            T = float(g["time_to_expiry"].iloc[0])
            F, D = float(g["forward"].iloc[0]), float(g["discount"].iloc[0])
            K = g["strike"].to_numpy(float)
            is_put = (g["right"] == "P").to_numpy()
            model = self._model_prices(pricer, T, F, D, K, is_put, *model_params)
            miv = oc.black76_implied_vol(model, F, K, T, D, ~is_put)
            parts.append(pd.DataFrame({
                "expiry": g["expiry"].to_numpy(), "time_to_expiry": T,
                "strike": K, "right": g["right"].to_numpy(),
                "moneyness": K / F, "market_iv": g["implied_vol"].to_numpy(float),
                "model_iv": miv,
            }))
        if not parts:
            return pd.DataFrame(columns=["expiry", "time_to_expiry", "strike", "right",
                                         "moneyness", "market_iv", "model_iv", "error"])
        out = pd.concat(parts, ignore_index=True)
        out["error"] = out["model_iv"] - out["market_iv"]
        return out

    def _out_of_sample(self, pricer, model_params) -> dict:
        """IV RMSE over the full cleaned OTM surface minus the anchors."""
        full = self.spx_surface_full
        key = ["expiry", "strike", "right"]
        anchors = self.spx_surface[key].assign(_a=1)
        oos = full.merge(anchors, on=key, how="left")
        oos = oos[oos["_a"].isna()].drop(columns="_a")
        err = self._surface_errors(oos, pricer, model_params)
        e = err["error"].dropna()
        wing = err[(err["right"] == "P") & (err["moneyness"] < _PUT_WING_KF)]["error"].dropna()
        by_exp = (err.dropna(subset=["error"]).groupby("expiry")["error"]
                  .apply(lambda s: float(np.sqrt(np.mean(s ** 2)) * 100)).to_dict())
        return {
            "spx_oos_iv_rmse": float(np.sqrt(np.mean(e ** 2)) * 100) if len(e) else float("nan"),
            "spx_oos_put_wing_rmse": (float(np.sqrt(np.mean(wing ** 2)) * 100)
                                      if len(wing) else float("nan")),
            "n_spx_oos": int(len(e)),
            "n_spx_oos_expiries": int(err["expiry"].nunique()),
            "spx_oos_rmse_by_expiry": by_exp,
        }

    def _vix_index_table(self, kappa, theta, sigma, v0, index_fn=None) -> list:
        if self.vix_ts.empty:
            return []
        fn = index_fn or (lambda T: heston_vix_term_index(kappa, theta, sigma, v0, T))
        model = fn(self.vix_ts["tenor_years"].to_numpy(float))
        return [{"ticker": t, "tenor_days": round(T * 365), "market": float(m), "model": float(x)}
                for t, T, m, x in zip(self.vix_ts["ticker"], self.vix_ts["tenor_years"],
                                      self.vix_ts["market_price"], np.atleast_1d(model))]

    def _vx_futures_oos(self, kappa, theta, sigma, v0) -> Optional[float]:
        if self.vx_futures.empty:
            return None
        T = self.vx_futures["days_to_expiry"].to_numpy(float) / 365.0
        model = np.atleast_1d(heston_vix_future_exact(kappa, theta, sigma, v0, T))
        return float(np.sqrt(np.mean((model - self.vx_futures["price"].to_numpy(float)) ** 2)))

    def _provenance(self) -> dict:
        """Data provenance shared by the Heston and Bates results."""
        s = self.spx_surface
        kf = s["strike"] / s["forward"]
        wing_mask = (s["right"] == "P") & (kf < _PUT_WING_KF)
        by_exp = []
        for exp, g in s.groupby("expiry"):
            gk = g["strike"] / g["forward"]
            by_exp.append({
                "expiry": exp, "T_days": round(float(g["time_to_expiry"].iloc[0]) * 365, 2),
                "n": int(len(g)), "n_puts": int((g["right"] == "P").sum()),
                "n_wing_puts": int(((g["right"] == "P") & (gk < _PUT_WING_KF)).sum()),
                "kf_min": float(gk.min()), "kf_max": float(gk.max()),
                "forward": float(g["forward"].iloc[0]),
                "discount": float(g["discount"].iloc[0]),
                "r": float(g["r_T"].iloc[0]), "q": float(g["q_T"].iloc[0]),
            })
        carry = [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                  for k, v in row.items()} for row in self.spx_carry.to_dict("records")]
        anchor_r = s.groupby("expiry")["r_T"].first()
        anchor_q = s.groupby("expiry")["q_T"].first()
        return {
            "n_spx_options": int(len(s)),
            "n_spx_expiries": int(s["expiry"].nunique()),
            "n_spx_puts": int((s["right"] == "P").sum()),
            "n_spx_calls": int((s["right"] == "C").sum()),
            "n_spx_wing_puts": int(wing_mask.sum()),
            "n_put_wing_expiries": int(sum(1 for e in by_exp
                                           if e["n_wing_puts"] >= _MIN_WING_PUTS_PER_EXPIRY)),
            "anchors_by_expiry": by_exp,
            "n_vix_tenors":  int(len(self.vix_ts)),
            "n_vix_options": int(len(self.vix_options)),
            "n_vx_futures":  int(len(self.vx_futures)),
            "spot_date":     self.spot_date,
            "spx_snapshot_date": getattr(self, "spx_snapshot_date", None),
            "spx_snapshot_session": getattr(self, "spx_snapshot_session", None),
            "S": self.S,
            # Rates ACTUALLY used: the SPX leg prices each expiry off its own
            # parity discount/forward; r/q below are their medians over the
            # anchor expiries (r_T = -ln D_T / T, q_T = r_T - ln(F_T/S)/T).
            "r": float(anchor_r.median()),
            "q": float(anchor_q.median()),
            "r_reference": float(self.r_reference),
            "r_reference_source": self.r_reference_source,
            "carry_by_expiry": carry,
            "chain_diagnostics": dict(getattr(self, "spx_chain_diagnostics", {})),
            "weights": {"w1": self.w1, "w2": self.w2, "w3": self.w3},
        }

    def _t_methods(self) -> dict:
        """Where T was measured from (clean_option_chain's diagnostics)."""
        d = getattr(self, "spx_chain_diagnostics", None) or {}
        origin = d.get("t_origin")
        gth = str(d.get("t_origin_basis", "")).startswith("quote_time")
        start = (f"the quote time {origin} (Cboe Global Trading Hours book)" if gth
                 else "the quote session's 16:00 ET close (closing book)")
        return {
            "T": f"from {start} to 16:00 ET (PM) / 09:30 ET (AM) settlement, /365d",
            "T_origin": origin,
            "vix_leg_timing": (
                "VIX index leg: the spot session's Cboe closes (16:15 ET)"
                + ("; the SPX quotes are from the later GTH book, a residual timing "
                   "mismatch the fit does not correct" if gth else "")),
        }

    def _methods(self, de_cfg: dict) -> dict:
        return {
            "vix_leg": "spot_index_term_structure (heston_vix_term_index, Cboe 9/30/93/184d)",
            "spx_quotes": "two-sided mid, no lastPrice",
            "spx_iv": "own Black-76 on parity F_T, D_T (Yahoo IV diagnostic only)",
            "carry": self.carry,
            "anchors": "z-grid linspace(-2.5,1.5,9), OTM vs F_T, >=3 puts at K/F<0.97",
            "expiries": f"nearest listed to {list(_CAL_EXPIRY_TARGETS_DAYS)}d, unique, within 50%",
            **self._t_methods(),
            "feller": self.feller,
            "weights": "active legs normalised to sum to 1",
            "optimizer": de_cfg,
        }

    # ── Calibration ────────────────────────────────────────────────────────────

    def _optimise(self, loss, bounds, de_maxiter, de_popsize, polish, init, label,
                  workers: int = 1):
        """Differential evolution (+ one L-BFGS-B polish).

        workers > 1 evaluates each generation in that many PROCESSES (scipy's
        integer ``workers`` is a multiprocessing.Pool, so the pricer's thread
        safety never mattered; the old ``workers=1`` comment said otherwise)
        and requires updating='deferred', which changes the DE trajectory
        relative to the serial 'immediate' run that produced the committed
        model_version 2 fits. workers = 1 reproduces those runs exactly. The
        objective must pickle (a HestonJointResiduals.loss does; the
        evaluation counter then lives in the workers and is not returned).
        A map-like callable is passed through (``workers=map`` is the serial
        'deferred' run that a process pool must reproduce bit for bit).
        """
        self._n_evals = 0
        t0 = time.time()
        if not callable(workers):
            workers = int(workers)
        de = optimize.differential_evolution(
            loss, bounds=bounds, maxiter=de_maxiter, popsize=de_popsize, tol=1e-6,
            seed=RANDOM_SEED, mutation=(0.5, 1.5), recombination=0.9, init=init,
            workers=workers,
            updating="immediate" if (not callable(workers) and workers == 1) else "deferred",
            polish=False,           # one explicit L-BFGS-B polish below
        )
        x, fun = de.x, de.fun
        logger.info("     %s DE done: loss=%.6g, evals=%d, success=%s",
                    label, fun, self._n_evals, de.success)
        if polish:
            pr = optimize.minimize(loss, x, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
            if pr.fun < fun:
                x, fun = pr.x, pr.fun
            logger.info("     %s polish done: loss=%.6g, evals=%d", label, fun, self._n_evals)
        return de, x, float(fun), time.time() - t0

    # ── Least-squares form of the joint loss (C23) ────────────────────────────

    def residual_model(self, w_spx: Optional[float] = None, w_vix: Optional[float] = None,
                       w_vix_opt: Optional[float] = None) -> HestonJointResiduals:
        """The joint loss as a residual vector (default: this calibrator's
        normalised weights; w_vix=0 gives the SPX-only problem)."""
        return HestonJointResiduals(
            self._spx_groups(),
            self.vix_ts["tenor_years"].to_numpy(float) if not self.vix_ts.empty else [],
            self.vix_ts["market_price"].to_numpy(float) if not self.vix_ts.empty else [],
            self.w1 if w_spx is None else w_spx,
            self.w2 if w_vix is None else w_vix,
            self.w3 if w_vix_opt is None else w_vix_opt,
            feller=self.feller, vix_options=self.vix_options, r_ref=self.r_reference)

    def residuals(self, params_vec) -> np.ndarray:
        """Residual vector r with sum(r**2) == joint_loss(params_vec) in the box."""
        return self.residual_model()(params_vec)

    def warm_start_fit(self) -> tuple:
        """(path, result) of the previous session's accepted fit (strictly
        before this spot session), or (None, None)."""
        return previous_accepted_fit(self.spot_date, Path(DATA_DIR) / "calibrations")

    def _trf_fit(self, model: HestonJointResiduals, lo, hi, n_starts: int, warm,
                 jacobian: str, label: str = "Heston", start_box: Optional[dict] = None) -> tuple:
        """Multi-start TRF on ``model``: warm start(s) first, then Sobol."""
        starts, labels = [], []
        for name, x in warm:
            starts.append(np.clip(np.asarray(x, dtype=float), lo, hi))
            labels.append(name)
        sob = sobol_starts(n_starts, start_box or HESTON_START_BOX, lo, hi, HESTON_PARAM_NAMES)
        starts.extend(sob)
        labels.extend(f"sobol{i}" for i in range(len(sob)))
        if not starts:
            raise ValueError("multi-start TRF needs at least one start")
        jac = model.jac if (jacobian == "analytic") else None
        n0, j0 = model.n_calls, model.n_jac
        t0 = time.perf_counter()
        best, rep = multistart_trf(model, jac, np.array(starts), lo, hi, labels)
        rep["seconds"] = time.perf_counter() - t0
        rep["n_residual_evals"] = model.n_calls - n0      # incl. FD Jacobian calls
        rep["n_jacobian_evals"] = model.n_jac - j0        # analytic Jacobians
        logger.info("     %s TRF: %d starts -> loss %.8g (%d/%d starts in the best basin, "
                    "%d basin(s)) | %d residual + %d Jacobian evals, %.1fs", label,
                    rep["n_starts"], 2 * best.cost, rep["n_starts_at_best"], rep["n_starts"],
                    rep["n_distinct_basins"], rep["n_residual_evals"],
                    rep["n_jacobian_evals"], rep["seconds"])
        return best, rep

    def _identifiability(self, model: HestonJointResiduals, x, lo, hi) -> dict:
        x = np.asarray(x, dtype=float)
        r = model(x)
        J = model.jac(x) if model.vix_opt is None else optimize.approx_fprime(x, model, 1e-7)
        return identifiability_report(J, r, x, HESTON_PARAM_NAMES, lo, hi)

    def calibrate(
        self,
        de_maxiter: int = 200,
        de_popsize: int = 10,
        polish: bool = True,
        verbose: bool = True,
        *,
        optimizer: str = "trf",
        n_starts: int = _TRF_N_STARTS,
        warm_start="auto",
        jacobian: str = "analytic",
        de_check: bool = False,
        de_workers: int = 1,
    ) -> dict:
        """
        Calibrate Heston by minimising joint_loss.

        optimizer="trf" (default since C23): the loss is a sum of squares, so
          it is minimised by scipy least_squares(method="trf") on the residual
          vector (HestonJointResiduals) from the warm start (the previous
          session's accepted fit, strictly earlier; warm_start="auto") and
          ``n_starts`` scrambled-Sobol points in config.HESTON_START_BOX, with
          the analytic Jacobian (jacobian="analytic"; "2-point" differences
          numerically). The lowest loss wins. The result records every start,
          the number of distinct basins and an identifiability report
          (Jacobian singular values, Gauss-Newton standard errors and
          correlations, parameters on bounds).
        optimizer="de": the pre-C23 method, differential evolution (Latin
          hypercube, seed 42, scipy polish off) then one L-BFGS-B polish;
          de_workers=1 reproduces the committed fits exactly.
        de_check=True (with "trf"): also run that DE (de_workers processes)
          as a global check, polish its optimum with TRF, keep whichever loss
          is lower and record both.

        warm_start: "auto", None/False (none), or a params dict / 5-vector.

        Returns the full result dict (also stored in self.result and
        persisted by save()): params, loss, leg_losses (in-sample
        spx_iv_rmse, exact-inversion spx_iv_rmse_exact, out-of-sample
        spx_oos_iv_rmse, vix_index_rmse + alias vix_futures_rmse, ...),
        Feller diagnostics, active constraints, data provenance, rates used,
        method choices and model_version.
        """
        if optimizer not in ("trf", "de"):
            raise ValueError(f"optimizer must be 'trf' or 'de', got {optimizer!r}")
        if jacobian not in ("analytic", "2-point"):
            raise ValueError(f"jacobian must be 'analytic' or '2-point', got {jacobian!r}")
        lvl = logging.INFO if verbose else logging.DEBUG
        logger.log(lvl, "[C4] Joint calibration: as of %s (session %s) | S=%.0f | "
                   "%d SPX anchors / %d expiries | %d VIX indices | feller=%s | %s",
                   self.as_of_date, self.spot_date, self.S, len(self.spx_surface),
                   self.spx_surface["expiry"].nunique(), len(self.vix_ts), self.feller,
                   optimizer)
        logger.log(lvl, "     Weights: w1=%.4f, w2=%.4f, w3=%.4f", self.w1, self.w2, self.w3)

        bounds = [HESTON_BOUNDS[k] for k in HESTON_PARAM_NAMES]
        lo, hi = _box_arrays(HESTON_BOUNDS, HESTON_PARAM_NAMES)
        model = self.residual_model()
        de_cfg = {"maxiter": de_maxiter, "popsize": de_popsize,
                  "members": de_popsize * len(bounds), "seed": RANDOM_SEED, "tol": 1e-6,
                  "scipy_polish": False, "lbfgsb_polish": bool(polish),
                  "workers": int(de_workers),
                  "updating": "immediate" if int(de_workers) == 1 else "deferred",
                  "bounds": {k: list(v) for k, v in HESTON_BOUNDS.items()}}
        multistart = de_report = None
        t_start = time.time()
        c_start = time.process_time()       # this process only (not DE worker processes)
        if optimizer == "de":
            serial = int(de_workers) == 1
            n0 = model.n_calls
            de, best, best_loss, _ = self._optimise(
                self.joint_loss if serial else model.loss, bounds,
                de_maxiter, de_popsize, polish, "latinhypercube", "Heston",
                workers=de_workers)
            # serial: joint_loss counts every call; parallel: the workers'
            # counters are lost, so DE's own nfev plus the in-process polish
            n_evals = self._n_evals if serial else int(de.nfev) + (model.n_calls - n0)
            success, message = bool(de.success), str(de.message)
            opt_cfg = de_cfg
        else:
            warm, warm_info = self._warm_starts(warm_start)
            sol, multistart = self._trf_fit(model, lo, hi, n_starts, warm, jacobian)
            best, best_loss = sol.x, float(2.0 * sol.cost)
            success, message = bool(sol.status > 0), str(sol.message)
            n_evals = multistart["n_residual_evals"] + multistart["n_jacobian_evals"]
            opt_cfg = {"method": "trf_multistart", "n_sobol_starts": int(n_starts),
                       "sobol_seed": RANDOM_SEED, "start_box": dict(HESTON_START_BOX),
                       "warm_start": warm_info, "jacobian": jacobian,
                       "x_scale": "jac", "ftol": _TRF_TOL, "xtol": _TRF_TOL,
                       "gtol": _TRF_TOL, "max_nfev_per_start": _TRF_MAX_NFEV,
                       "basin_loss_rtol": _BASIN_LOSS_RTOL, "de_check": bool(de_check),
                       "bounds": {k: list(v) for k, v in HESTON_BOUNDS.items()}}
            if de_check:
                de_report, best, best_loss = self._de_global_check(
                    model, bounds, lo, hi, de_maxiter, de_popsize, de_workers, jacobian,
                    best, best_loss)
                opt_cfg["de"] = de_cfg
        self.fit_time = time.time() - t_start
        self.fit_cpu_time = time.process_time() - c_start

        kappa, theta, sigma, rho, v0 = (float(x) for x in best)
        self.params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho, "v0": v0}

        l1 = self._spx_leg(kappa, theta, sigma, rho, v0)
        l2 = self._vix_index_leg(kappa, theta, sigma, v0)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        ins = self._surface_errors(self.spx_surface, heston_call_batch,
                                   (kappa, theta, sigma, rho, v0))["error"].dropna()
        oos = self._out_of_sample(heston_call_batch, (kappa, theta, sigma, rho, v0))
        vix_rmse = float(np.sqrt(l2)) * 100
        self.losses = {
            "spx_iv_rmse":       float(np.sqrt(l1)) * 100,   # in-sample, vega-approx (vp)
            "spx_iv_rmse_exact": float(np.sqrt(np.mean(ins ** 2)) * 100) if len(ins) else float("nan"),
            "spx_oos_iv_rmse":   oos["spx_oos_iv_rmse"],
            "spx_oos_put_wing_rmse": oos["spx_oos_put_wing_rmse"],
            "vix_index_rmse":    vix_rmse,                    # VIX points
            "vix_futures_rmse":  vix_rmse,                    # back-compat alias (dashboard)
            "vx_futures_rmse_oos": self._vx_futures_oos(kappa, theta, sigma, v0),
            "vix_options_rmse":  float(np.sqrt(l3)) * 100,
            "total_loss":        best_loss,
        }

        feller_state, feller_margin = classify_feller(kappa, theta, sigma)
        imposed = self.feller != "off"
        result = {
            "model_version": MODEL_VERSION,
            "as_of_date": self.as_of_date,
            "params":     self.params,
            "loss":       best_loss,
            "leg_losses": self.losses,
            "fit_time":   self.fit_time,
            # CPU seconds of the optimisation in this process: stable on a
            # loaded machine where fit_time (wall clock) is not; excludes
            # DE worker processes when de_workers > 1
            "fit_cpu_time": self.fit_cpu_time,
            "n_evals":    n_evals,
            "success":    success,
            "optimizer":  "trf_multistart" if optimizer == "trf" else "de",
            "optimizer_message": message,
            "de_message": message,                 # pre-C23 key (the DE message then)
            "feller":        self.feller,
            "feller_state":  feller_state,
            "feller_margin": float(feller_margin),
            "feller_ok":     feller_state != "VIOLATED",   # derived from the state
            "pinned_params": count_pinned_params(self.params, feller_state,
                                                 feller_imposed=imposed),
            "active_constraints": active_constraints(self.params, feller_state,
                                                     feller_imposed=imposed),
            "vix_index": self._vix_index_table(kappa, theta, sigma, v0),
            "vvix_check": self._vvix_check(kappa, theta, sigma, v0),
            "oos": {k: v for k, v in oos.items() if k not in ("spx_oos_iv_rmse",
                                                              "spx_oos_put_wing_rmse")},
            "methods": self._methods(opt_cfg),
            **self._provenance(),
        }
        if multistart is not None:
            result["multistart"] = multistart
        if de_report is not None:
            result["de_check"] = de_report
        try:
            result["identifiability"] = self._identifiability(model, best, lo, hi)
        except (np.linalg.LinAlgError, ValueError, ArithmeticError) as exc:
            # a diagnostic must never fail the fit
            logger.warning("  identifiability report failed: %s", exc)
            result["identifiability"] = {"error": str(exc)}
        self.result = result
        if verbose:
            self._log_result(result)
        return result

    # ── sigma-profile (C23 diagnostic) ────────────────────────────────────────

    def fit_summary(self, params) -> dict:
        """Leg statistics of the joint Heston loss at a parameter vector:
        loss, in-sample anchor RMSE (vega-approx and exact inversion),
        out-of-sample RMSE (all, put wing, by expiry), VIX index RMSE, VX
        futures OOS, Feller margin."""
        kappa, theta, sigma, rho, v0 = (float(v) for v in params)
        l1 = self._spx_leg(kappa, theta, sigma, rho, v0)
        l2 = self._vix_index_leg(kappa, theta, sigma, v0)
        ins = self._surface_errors(self.spx_surface, heston_call_batch,
                                   (kappa, theta, sigma, rho, v0))["error"].dropna()
        oos = self._out_of_sample(heston_call_batch, (kappa, theta, sigma, rho, v0))
        r = self.residuals([kappa, theta, sigma, rho, v0])   # no box check (profile fits)
        return {
            "params": dict(zip(HESTON_PARAM_NAMES, (kappa, theta, sigma, rho, v0))),
            "loss": float(r @ r),
            "spx_iv_rmse": float(np.sqrt(l1)) * 100,
            "spx_iv_rmse_exact": float(np.sqrt(np.mean(ins ** 2)) * 100) if len(ins) else None,
            "spx_oos_iv_rmse": oos["spx_oos_iv_rmse"],
            "spx_oos_put_wing_rmse": oos["spx_oos_put_wing_rmse"],
            "spx_oos_rmse_by_expiry": oos["spx_oos_rmse_by_expiry"],
            "n_spx_oos": oos["n_spx_oos"],
            "vix_index_rmse": float(np.sqrt(l2)) * 100,
            "vx_futures_rmse_oos": self._vx_futures_oos(kappa, theta, sigma, v0),
            **self._vvix_check(kappa, theta, sigma, v0),
            "feller_margin": float(2.0 * kappa * theta - sigma * sigma),
        }

    def _vvix_check(self, kappa, theta, sigma, v0) -> dict:
        """Model 30-day VIX log-contract vol (heston_vvix) vs the session's
        ^VVIX close: out of sample, since no leg of the loss sees it."""
        model = float(heston_vvix(kappa, theta, sigma, v0))
        mkt = getattr(self, "vvix_market", None)
        return {"vvix_model": model, "vvix_market": mkt,
                "vvix_error": (model - mkt) if mkt is not None else None}

    def sigma_profile(self, caps: Sequence[float] = SIGMA_PROFILE_CAPS,
                      kappa_max: float = PROFILE_KAPPA_MAX, n_starts: int = 4,
                      base=None, verbose: bool = False) -> dict:
        """Refit the joint loss with sigma <= cap for every cap, so the cost
        of the production box (config.HESTON_BOUNDS, sigma <= 2) is measured
        rather than hidden.

        Every refit uses the multi-start TRF with kappa's ceiling widened to
        ``kappa_max`` (config.PROFILE_KAPPA_MAX; the production ceiling would
        bind first, see config). Starts: ``base`` (default: this calibrator's
        fit, else none) clipped into the box, the previous cap's optimum
        (continuation, so the loss can only fall as the cap rises), and
        ``n_starts`` Sobol points. Diagnostic only: it never changes
        self.params or self.result.
        """
        if base is None and self.params is not None:
            base = [self.params[k] for k in HESTON_PARAM_NAMES]
        elif isinstance(base, dict):
            base = [float(base[k]) for k in HESTON_PARAM_NAMES]
        model = self.residual_model()
        rows, prev = [], None
        for cap in caps:
            box = dict(HESTON_BOUNDS)
            box["sigma"] = (HESTON_BOUNDS["sigma"][0], float(cap))
            box["kappa"] = (HESTON_BOUNDS["kappa"][0], float(kappa_max))
            lo, hi = _box_arrays(box, HESTON_PARAM_NAMES)
            warm = []
            if base is not None:
                warm.append(("base", np.clip(base, lo, hi)))
            if prev is not None:
                warm.append(("previous_cap", np.clip(prev, lo, hi)))
            sol, rep = self._trf_fit(model, lo, hi, n_starts, warm, "analytic",
                                     label=f"sigma<={cap:g}")
            x = sol.x
            row = self.fit_summary(x)
            row.update({
                "sigma_cap": float(cap), "kappa_max": float(kappa_max),
                "sigma_at_cap": bool(_at_bound(x[2], cap, cap, _PIN_REL_TOL)),
                "kappa_at_max": bool(_at_bound(x[0], kappa_max, kappa_max, _PIN_REL_TOL)),
                "n_distinct_basins": rep["n_distinct_basins"],
                "n_starts": rep["n_starts"], "seconds": rep["seconds"],
            })
            rows.append(row)
            prev = x
            if verbose:
                logger.info("  sigma <= %-4g loss %.6e | sigma %.3f kappa %.2f | in %.3f vp, "
                            "OOS %.3f vp", cap, row["loss"], x[2], x[0], row["spx_iv_rmse"],
                            row["spx_oos_iv_rmse"])
        return {"caps": [float(c) for c in caps], "kappa_max": float(kappa_max),
                "production_box": {k: list(v) for k, v in HESTON_BOUNDS.items()},
                "rows": rows}

    # ── SPX-only model comparison (C23) ───────────────────────────────────────

    def calibrate_spx_only(self, model: str = "heston", **kwargs) -> dict:
        """calibrate_comparison(model, legs="spx", ...): the VIX leg off."""
        return self.calibrate_comparison(model, legs="spx", **kwargs)

    def calibrate_comparison(self, model: str = "heston", legs: str = "spx",
                             n_starts: int = 4, warm=None,
                             n_factors: int = ROUGH_HESTON_N_FACTORS,
                             n_steps: int = ROUGH_HESTON_TIME_STEPS,
                             numerics_check: bool = True, box: str = "production") -> dict:
        """Heston or lifted rough Heston fitted like for like.

        legs="spx": the SPX anchors only (the VIX index is then out of
        sample); legs="joint": the production joint loss (this calibrator's
        normalised w1, w2 on the same anchors and spot VIX indices; for rough
        Heston the index is rough_heston_vix_term_index).

        Both models see the same anchors and residuals sqrt(1/n) (model -
        market)/vega (the joint loss with the VIX leg off), the production
        box for the five Heston parameters (rough Heston adds H in
        config.ROUGH_HESTON_H_BOUNDS), the same TRF tolerances, and Sobol
        starts plus warm starts. Heston uses the analytic Jacobian; rough
        Heston prices on frozen Lewis grids refreshed at each optimum
        (models/rough_heston.fit_trf_with_grid_refresh) with exact theta / v0
        Jacobian columns and one-sided differences for the rest.

        warm: Heston -> [5-vector or params dict] (default: this calibrator's
        joint fit); rough Heston -> a Heston 5-vector, used at H = 0.30 and
        0.10 (default: the joint fit). Reported: in-sample anchor RMSE (exact
        inversion), out-of-sample RMSE over the full cleaned surface (all,
        put wing, by expiry), the VIX index RMSE (out of sample: the VIX leg
        is off), fit time, evaluations, every start, and an identifiability
        report. numerics_check (rough): anchor IVs re-priced with twice the
        factors and four times the Riccati steps, so the approximation error
        is measured next to the misfit (timed apart from fit_seconds; a
        self-convergence gap; scaled by
        1/(1 - 2^-1.5) for the n^-1.5 lifting rate it estimates the
        20-factor error, cf. the QuantLib Adams comparison in
        tests/test_rough_heston.py). box="wide" (Heston only) fits in the
        sigma-profile box instead (sigma <= max(SIGMA_PROFILE_CAPS), kappa <=
        PROFILE_KAPPA_MAX), so rough Heston can also be compared with a Heston
        that is not held by the production sigma <= 2 prior.
        """
        from joint_vol_calibration.models import rough_heston as rh

        if model not in ("heston", "rough_heston"):
            raise ValueError(f"model must be 'heston' or 'rough_heston', got {model!r}")
        if legs not in ("spx", "joint"):
            raise ValueError(f"legs must be 'spx' or 'joint', got {legs!r}")
        w_spx, w_vix = (1.0, 0.0) if legs == "spx" else (self.w1, self.w2)
        if box not in ("production", "wide") or (box == "wide" and model != "heston"):
            raise ValueError(f"box must be 'production' (or 'wide' for Heston), got {box!r}")
        hbox = dict(HESTON_BOUNDS)
        if box == "wide":
            hbox["sigma"] = (HESTON_BOUNDS["sigma"][0], float(max(SIGMA_PROFILE_CAPS)))
            hbox["kappa"] = (HESTON_BOUNDS["kappa"][0], float(PROFILE_KAPPA_MAX))
        if warm is None and self.params is not None:
            warm = [self.params[k] for k in HESTON_PARAM_NAMES]
        elif isinstance(warm, dict):
            warm = [float(warm[k]) for k in HESTON_PARAM_NAMES]
        groups = self._spx_groups()
        t0 = time.perf_counter()
        if model == "heston":
            names = HESTON_PARAM_NAMES
            lo, hi = _box_arrays(hbox, names)
            res_model = self.residual_model(w_spx=w_spx, w_vix=w_vix, w_vix_opt=0.0)
            warm_list = [("warm", warm)] if warm is not None else []
            sbox = dict(HESTON_START_BOX)
            if box == "wide":       # starts that can reach the large-sigma basin too
                sbox["kappa"] = (0.5, 40.0, "log")
                sbox["sigma"] = (0.2, 8.0, "log")
            sol, rep = self._trf_fit(res_model, lo, hi, n_starts, warm_list, "analytic",
                                     label=f"Heston {legs} ({box} box)", start_box=sbox)
            x, loss = sol.x, float(2.0 * sol.cost)
            J, r = res_model.jac(x), res_model(x)
            pricer = heston_call_batch
            vix_model = heston_vix_term_index(x[0], x[1], x[2], x[4],
                                              self.vix_ts["tenor_years"].to_numpy(float)) \
                if not self.vix_ts.empty else None
            n_evals = rep["n_residual_evals"] + rep["n_jacobian_evals"]
            numerics = None
        else:
            names = rh.ROUGH_HESTON_PARAM_NAMES
            box = dict(HESTON_BOUNDS)
            box["H"] = ROUGH_HESTON_H_BOUNDS
            lo, hi = _box_arrays(box, names)
            resid = rh.RoughHestonSPXResiduals(
                groups, w_spx, n_factors, n_steps,
                vix_tenors=self.vix_ts["tenor_years"].to_numpy(float),
                vix_market=self.vix_ts["market_price"].to_numpy(float), w_vix=w_vix)
            starts, labels = [], []
            if warm is not None:
                for H0 in (0.30, 0.10):
                    starts.append(np.r_[np.asarray(warm, dtype=float), H0])
                    labels.append(f"warm_H{H0:.2f}")
            sob = sobol_starts(n_starts, ROUGH_HESTON_START_BOX, lo, hi, names)
            starts.extend(sob)
            labels.extend(f"sobol{i}" for i in range(len(sob)))
            x, loss, runs = rh.fit_trf_with_grid_refresh(resid, starts, labels, lo, hi)
            groups_b = _group_basins([run["loss"] for run in runs])
            rep = {"runs": runs, "n_starts": len(runs), "n_distinct_basins": len(groups_b),
                   "n_starts_at_best": len(groups_b[0]),
                   "basin_losses": [runs[g[0]]["loss"] for g in groups_b],
                   "best_label": runs[groups_b[0][0]]["label"],
                   "n_residual_evals": resid.n_calls, "n_grid_adaptations": resid.n_adapt}
            r = resid.adapt(x)
            J = resid.jac(x, lo, hi)
            pricer = functools.partial(rh.rough_heston_call_batch, n_factors=n_factors,
                                       n_steps=n_steps)
            vix_model = rh.rough_heston_vix_term_index(
                x[0], x[1], x[2], x[4], x[5], self.vix_ts["tenor_years"].to_numpy(float),
                n_factors) if not self.vix_ts.empty else None
            n_evals = resid.n_calls
            numerics = None
            fit_seconds = time.perf_counter() - t0      # the numerics check is not fit time
            if numerics_check:
                base_iv = self._surface_errors(self.spx_surface, pricer, tuple(x))["model_iv"]
                fine = functools.partial(rh.rough_heston_call_batch, n_factors=2 * n_factors,
                                         n_steps=4 * n_steps)
                fine_iv = self._surface_errors(self.spx_surface, fine, tuple(x))["model_iv"]
                d_iv = (base_iv - fine_iv).to_numpy(float) * 100
                d_iv = d_iv[np.isfinite(d_iv)]
                max_d = float(np.max(np.abs(d_iv))) if len(d_iv) else None
                numerics = {
                    "reference": {"n_factors": 2 * n_factors, "n_steps": 4 * n_steps},
                    "anchor_iv_max_abs_diff_vp": max_d,
                    "anchor_iv_rms_diff_vp": float(np.sqrt(np.mean(d_iv ** 2))) if len(d_iv) else None,
                    # the lifting error falls like n^-1.5 (tests), so the gap
                    # to 2n factors is (1 - 2^-1.5) of the n-factor error
                    "lifting_error_estimate_max_vp": (max_d / (1.0 - 2.0 ** -1.5)
                                                      if max_d is not None else None),
                    "frozen_vs_adaptive_rel_loss_diff": float(max(
                        abs(run["loss"] - run["frozen_loss"]) / max(run["loss"], 1e-300)
                        for run in runs)),
                    "seconds": time.perf_counter() - t0 - fit_seconds,
                }
        seconds = fit_seconds if model == "rough_heston" else time.perf_counter() - t0
        ins = self._surface_errors(self.spx_surface, pricer, tuple(x))["error"].dropna()
        oos = self._out_of_sample(pricer, tuple(x))
        out = {
            "model": model,
            "legs": legs,
            "weights": {"w_spx": float(w_spx), "w_vix": float(w_vix)},
            "box": {n: [float(a), float(b)] for n, a, b in zip(names, lo, hi)},
            "params": {n: float(v) for n, v in zip(names, x)},
            "loss": float(loss),
            "spx_iv_rmse_exact": float(np.sqrt(np.mean(ins ** 2)) * 100) if len(ins) else None,
            "n_spx_anchors": int(len(ins)),
            "spx_oos_iv_rmse": oos["spx_oos_iv_rmse"],
            "spx_oos_put_wing_rmse": oos["spx_oos_put_wing_rmse"],
            "spx_oos_rmse_by_expiry": oos["spx_oos_rmse_by_expiry"],
            "n_spx_oos": oos["n_spx_oos"],
            # out of sample for legs="spx", in sample for legs="joint"
            "vix_index_rmse_oos" if legs == "spx" else "vix_index_rmse": (
                float(np.sqrt(np.mean((np.atleast_1d(vix_model)
                                       - self.vix_ts["market_price"].to_numpy(float)) ** 2)))
                if vix_model is not None else None),
            "fit_seconds": float(seconds),
            "n_evals": int(n_evals),
            "multistart": rep,
            "on_bound": [n for j, n in enumerate(names)
                         if _at_bound(x[j], lo[j], (hi[j] - lo[j]) or 1.0, _PIN_REL_TOL)
                         or _at_bound(x[j], hi[j], (hi[j] - lo[j]) or 1.0, _PIN_REL_TOL)],
        }
        try:
            out["identifiability"] = identifiability_report(J, r, x, names, lo, hi)
        except (np.linalg.LinAlgError, ValueError, ArithmeticError) as exc:
            out["identifiability"] = {"error": str(exc)}
        if model == "rough_heston":
            out["numerics"] = numerics
            out["settings"] = {"n_factors": int(n_factors), "n_steps": int(n_steps),
                               "lewis_rtol": rh.ROUGH_LEWIS_RTOL,
                               "H_bounds": list(ROUGH_HESTON_H_BOUNDS)}
        return out

    def _warm_starts(self, warm_start) -> tuple:
        """[(label, x)] for the TRF and a JSON-able description."""
        if warm_start is None or warm_start is False:
            return [], None
        if isinstance(warm_start, str):
            if warm_start != "auto":
                raise ValueError(f"warm_start must be 'auto', None, a dict or a vector, "
                                 f"got {warm_start!r}")
            path, res = self.warm_start_fit()
            if res is None:
                return [], {"source": None, "reason": "no accepted earlier-session fit"}
            x = [float(res["params"][k]) for k in HESTON_PARAM_NAMES]
            return [("warm:" + path.name, x)], {
                "source": path.name, "session": str(res.get("spot_date"))[:10],
                "params": dict(zip(HESTON_PARAM_NAMES, x))}
        if isinstance(warm_start, dict):
            x = [float(warm_start[k]) for k in HESTON_PARAM_NAMES]
        else:
            x = [float(v) for v in warm_start]
        return [("warm:argument", x)], {"source": "argument",
                                        "params": dict(zip(HESTON_PARAM_NAMES, x))}

    def _de_global_check(self, model, bounds, lo, hi, de_maxiter, de_popsize, de_workers,
                         jacobian, trf_x, trf_loss) -> tuple:
        """DE (in de_workers processes) + a TRF polish of its optimum, as a
        global check on the multi-start TRF. Returns (report, x, loss) with
        the lower of the two."""
        de, de_x, de_loss, de_time = self._optimise(
            model.loss, bounds, de_maxiter, de_popsize, False, "latinhypercube",
            "Heston DE check", workers=de_workers)
        jac = model.jac if jacobian == "analytic" else None
        t0 = time.perf_counter()
        pol, _ = multistart_trf(model, jac, np.array([de_x]), lo, hi, ["de_optimum"])
        pol_time = time.perf_counter() - t0
        pol_loss = float(2.0 * pol.cost)
        rel = (pol_loss - trf_loss) / trf_loss if trf_loss > 0 else 0.0
        report = {
            "de_loss": float(de_loss), "de_polished_loss": pol_loss,
            "de_seconds": float(de_time), "polish_seconds": float(pol_time),
            "de_nfev": int(de.nfev), "de_nit": int(de.nit), "de_success": bool(de.success),
            "de_x": [float(v) for v in de_x], "de_polished_x": [float(v) for v in pol.x],
            "trf_loss": float(trf_loss), "rel_diff_polished_vs_trf": float(rel),
            "agrees": bool(abs(rel) <= _BASIN_LOSS_RTOL),
            "workers": int(de_workers),
        }
        if pol_loss < trf_loss * (1.0 - _BASIN_LOSS_RTOL):
            logger.warning("     DE check found a lower basin (%.8g < %.8g): using it",
                           pol_loss, trf_loss)
            report["used"] = "de"
            return report, pol.x, pol_loss
        report["used"] = "trf"
        return report, trf_x, trf_loss

    # ── Bates (1996) SVJ Calibration ──────────────────────────────────────────

    # Bounds CONTAIN the nested Heston model (lam = 0 or mu_j = sigma_j = 0),
    # so at a true optimum Bates can never fit worse than Heston. The pre-C22
    # box (lam >= 0.5, mu_j <= -0.005, sigma_j >= 0.02) forced a jump
    # component and the published fit sat on three of its edges.
    _BATES_JUMP_BOUNDS = [(0.0, 10.0), (-0.30, 0.10), (0.0, 0.30)]

    def _bates_joint_loss(self, params_vec: np.ndarray) -> float:
        """Joint loss for 8-parameter Bates [kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j]."""
        self._n_evals += 1
        kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = params_vec
        if not self._in_bounds(kappa, theta, sigma, rho, v0):
            return 1e6
        for val, (lo, hi) in zip((lam, mu_j, sigma_j), self._BATES_JUMP_BOUNDS):
            if not lo <= val <= hi:
                return 1e6
        l1 = self._bates_spx_leg(kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
        l2 = self._bates_vix_index_leg(kappa, theta, sigma, v0, lam, mu_j, sigma_j)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        return (self.w1 * l1 + self.w2 * l2 + self.w3 * l3
                + self._feller_penalty(kappa, theta, sigma))

    def _bates_spx_leg(self, kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j) -> float:
        """SPX leg priced with the Bates batch pricer."""
        groups = self._spx_groups()
        if not groups:
            raise ValueError("SPX leg has no anchors")
        sse, n = 0.0, 0
        for T, F, D, K, is_put, mkt, vega in groups:
            model = self._model_prices(bates_call_batch, T, F, D, K, is_put, kappa, theta,
                                       sigma, rho, v0, lam, mu_j, sigma_j)
            err = (model - mkt) / vega
            sse += float(np.sum(err ** 2))
            n += len(err)
        return sse / max(n, 1)

    def _bates_vix_index_leg(self, kappa, theta, sigma, v0, lam, mu_j, sigma_j) -> float:
        """VIX index leg under Bates: the model-free VIX picks up the jump
        term 2 lam (kbar - mu_j) (bates_vix_term_index); lam = 0 is Heston."""
        if self.vix_ts.empty:
            return 0.0
        tenors = self.vix_ts["tenor_years"].to_numpy(float)
        model = bates_vix_term_index(kappa, theta, sigma, v0, lam, mu_j, sigma_j, tenors)
        market = self.vix_ts["market_price"].to_numpy(float)
        return float(np.mean(((model - market) / 100.0) ** 2))

    def calibrate_bates(
        self,
        de_maxiter: int = 200,
        de_popsize: int = 10,
        polish: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Bates SVJ calibration (research only).

        Eight parameters: the Heston five plus
          lam     — jump intensity (jumps/year)   in [0, 10]
          mu_j    — mean log jump size            in [-0.30, 0.10]
          sigma_j — log jump size std             in [0, 0.30]
        The box contains the nested Heston model, and the DE population is
        seeded with the Heston optimum (lam = 0) when calibrate() has run, so
        the Bates loss is <= the Heston loss up to optimiser tolerance. The
        pre-C22 conclusion that the jump component "is not identifiable" was
        drawn from a fit whose bounds excluded lam = 0; it has not been
        re-established and is not claimed here.
        """
        bounds = [HESTON_BOUNDS[k] for k in ("kappa", "theta", "sigma", "rho", "v0")]
        bounds += self._BATES_JUMP_BOUNDS

        def _n(v, lo, hi):
            return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

        base = ([self.params[k] for k in ("kappa", "theta", "sigma", "rho", "v0")]
                if self.params else [3.0, 0.05, 0.6, -0.8, 0.02])
        seeds = [base + [0.0, -0.05, 0.05],
                 base + [0.5, -0.05, 0.05],
                 base + [1.0, -0.10, 0.08],
                 base + [3.0, -0.03, 0.03]]
        seed_pop = np.array([[_n(v, lo, hi) for v, (lo, hi) in zip(row, bounds)]
                             for row in seeds])
        n_total = de_popsize * len(bounds)
        fill = np.random.RandomState(RANDOM_SEED).uniform(
            0.0, 1.0, size=(n_total - len(seeds), len(bounds)))
        init = np.vstack([seed_pop, fill])

        if verbose:
            logger.info("[C4-Bates] Bates SVJ calibration: %s | S=%.0f | %d SPX opts | pop=%d",
                        self.as_of_date, self.S, len(self.spx_surface), n_total)
        de, best, best_loss, fit_time = self._optimise(
            self._bates_joint_loss, bounds, de_maxiter, de_popsize, polish, init, "Bates")
        n_evals = self._n_evals

        kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j = (float(x) for x in best)
        bates_params = {"kappa": kappa, "theta": theta, "sigma": sigma, "rho": rho,
                        "v0": v0, "lam": lam, "mu_j": mu_j, "sigma_j": sigma_j}
        l1 = self._bates_spx_leg(kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j)
        l2 = self._bates_vix_index_leg(kappa, theta, sigma, v0, lam, mu_j, sigma_j)
        l3 = self._vix_options_leg(kappa, theta, sigma, v0) if self.w3 > 0 else 0.0
        oos = self._out_of_sample(bates_call_batch,
                                  (kappa, theta, sigma, rho, v0, lam, mu_j, sigma_j))
        vix_rmse = float(np.sqrt(l2)) * 100
        bates_losses = {
            "spx_iv_rmse":      float(np.sqrt(l1)) * 100,
            "spx_oos_iv_rmse":  oos["spx_oos_iv_rmse"],
            "vix_index_rmse":   vix_rmse,
            "vix_futures_rmse": vix_rmse,      # back-compat alias
            "vix_options_rmse": float(np.sqrt(l3)) * 100,
            "total_loss":       best_loss,
        }
        state, margin = classify_feller(kappa, theta, sigma)
        result = {
            "model_version": MODEL_VERSION,
            "as_of_date": self.as_of_date,
            "params":     bates_params,
            "loss":       best_loss,
            "leg_losses": bates_losses,
            "fit_time":   fit_time,
            "n_evals":    n_evals,
            "success":    bool(de.success),
            "feller":     self.feller,
            "feller_state":  state,
            "feller_margin": float(margin),
            "feller_ok":     state != "VIOLATED",
            "jump_bounds":   {"lam": self._BATES_JUMP_BOUNDS[0],
                              "mu_j": self._BATES_JUMP_BOUNDS[1],
                              "sigma_j": self._BATES_JUMP_BOUNDS[2]},
            **self._provenance(),
        }
        if self.params is not None:
            result["heston_comparison"] = {
                "heston_spx_rmse": self.losses["spx_iv_rmse"],
                "bates_spx_rmse":  bates_losses["spx_iv_rmse"],
                "heston_vix_rmse": self.losses["vix_index_rmse"],
                "bates_vix_rmse":  bates_losses["vix_index_rmse"],
                "heston_loss":     self.result["loss"] if self.result else None,
                "bates_loss":      best_loss,
                "heston_rho":      self.params["rho"],
                "bates_rho":       rho,
            }
        if verbose:
            self._log_bates_result(result)
        return result

    def _log_bates_result(self, result: dict):
        p, ll = result["params"], result["leg_losses"]
        logger.info("── Calibrated Bates SVJ Parameters (%s) ──", self.as_of_date)
        logger.info("  kappa=%.4f theta=%.4f sigma=%.4f rho=%.4f v0=%.4f | "
                    "lam=%.3f/yr mu_j=%.2f%% sigma_j=%.2f%%",
                    p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"],
                    p["lam"], p["mu_j"] * 100, p["sigma_j"] * 100)
        logger.info("  Feller: %s (margin %+.2e, %s)", result["feller_state"],
                    result["feller_margin"], "imposed" if self.feller != "off" else "not imposed")
        logger.info("  SPX RMSE %.3f vp (OOS %.3f) | VIX index RMSE %.3f | fit %.1fs",
                    ll["spx_iv_rmse"], ll["spx_oos_iv_rmse"], ll["vix_index_rmse"],
                    result["fit_time"])
        if "heston_comparison" in result:
            hc = result["heston_comparison"]
            logger.info("  vs Heston: SPX %.3f -> %.3f vp, VIX %.3f -> %.3f, rho %.4f -> %.4f",
                        hc["heston_spx_rmse"], hc["bates_spx_rmse"], hc["heston_vix_rmse"],
                        hc["bates_vix_rmse"], hc["heston_rho"], hc["bates_rho"])

    # ── Validation ────────────────────────────────────────────────────────────

    def smile_reconstruction_error(self) -> pd.DataFrame:
        """
        Model vs market implied vol at every calibration anchor.

        Both sides are Black-76 IVs on the same per-expiry parity forward and
        discount the loss uses (pre-C22 the market side was Yahoo's IV, which
        disagreed with the objective by ~4x and showed a spurious -1.6 vp bias).

        Columns: expiry (days), strike, moneyness (K/F), right, market_iv,
        model_iv, error, abs_error.
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() before validate()")
        p = self.params
        res = self._surface_errors(self.spx_surface, heston_call_batch,
                                   (p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"]))
        res["expiry"] = (res["time_to_expiry"] * 365).round().astype(int)
        res["abs_error"] = res["error"].abs()
        return res[["expiry", "strike", "moneyness", "right", "market_iv", "model_iv",
                    "error", "abs_error"]]

    def validate(self) -> dict:
        """
        Calibration quality report (anchors, exact IV inversion) and the VIX
        index comparison, consistent with the loss.
        """
        if self.params is None:
            raise RuntimeError("Call calibrate() first")
        smile_df = self.smile_reconstruction_error()
        rmse  = float(np.sqrt(np.nanmean(smile_df["error"] ** 2))) * 100
        mae   = float(np.nanmean(smile_df["abs_error"])) * 100
        max_e = float(np.nanmax(smile_df["abs_error"])) * 100
        bias  = float(np.nanmean(smile_df["error"])) * 100

        p = self.params
        if self.vix_ts.empty:
            vix_rmse = float("nan")
        else:
            model = heston_vix_term_index(p["kappa"], p["theta"], p["sigma"], p["v0"],
                                          self.vix_ts["tenor_years"].to_numpy(float))
            vix_rmse = float(np.sqrt(np.mean((model - self.vix_ts["market_price"]
                                              .to_numpy(float)) ** 2)))
        state, margin = classify_feller(p["kappa"], p["theta"], p["sigma"])
        report = {
            "smile_rmse_vol_pts":    rmse,
            "smile_mae_vol_pts":     mae,
            "smile_max_err_vol_pts": max_e,
            "smile_bias_vol_pts":    bias,
            "vix_curve_rmse_pts":    vix_rmse,
            "vix_index_rmse_pts":    vix_rmse,
            "n_spx_options":         len(smile_df),
            "feller_state":          state,
            "feller_margin":         margin,
            "feller_condition_ok":   state != "VIOLATED",
        }
        logger.info("── Calibration Quality Report ── SPX RMSE %.2f / MAE %.2f / max %.2f / "
                    "bias %+.2f vp | VIX index RMSE %.2f pts | Feller %s",
                    rmse, mae, max_e, bias, vix_rmse, state)
        return report

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Persist the full calibration result dict (the same schema the daily
        pipeline writes). Default path: data_store/calibrations/joint_cal_{spot_date}.pkl
        """
        if self.params is None or self.result is None:
            raise RuntimeError("Nothing to save — run calibrate() first")
        if path is None:
            cal_dir = DATA_DIR / "calibrations"
            cal_dir.mkdir(parents=True, exist_ok=True)
            path = str(cal_dir / f"joint_cal_{self.spot_date}.pkl")
        payload = dict(self.result)
        payload.setdefault("losses", payload.get("leg_losses"))   # pre-C22 key
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("Calibration saved to %s", path)
        return path

    @classmethod
    def load_params(cls, path: str) -> dict:
        """Load a saved calibration payload."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Display ───────────────────────────────────────────────────────────────

    def _log_result(self, result: dict):
        p, ll = result["params"], result["leg_losses"]
        logger.info("── Calibrated Heston Parameters (session %s) ──", result["spot_date"])
        logger.info("  kappa=%.4f theta=%.4f (vol %.1f%%) sigma=%.4f rho=%.4f v0=%.4f (vol %.1f%%)",
                    p["kappa"], p["theta"], math.sqrt(p["theta"]) * 100, p["sigma"],
                    p["rho"], p["v0"], math.sqrt(p["v0"]) * 100)
        logger.info("  Feller (diagnostic, %s): 2κθ=%.4f vs σ²=%.4f → %s (margin %+.2e)",
                    "imposed" if self.feller != "off" else "not imposed",
                    2 * p["kappa"] * p["theta"], p["sigma"] ** 2,
                    result["feller_state"], result["feller_margin"])
        ac = result["active_constraints"]
        logger.info("  Active constraints: %s → %d effective d.o.f. of 5",
                    "; ".join(ac) if ac else "none", 5 - len(result["pinned_params"]))
        logger.info("  Data: %d SPX anchors over %d expiries (%d puts, %d at K/F<%.2f), "
                    "%d VIX indices | r=%.4f q=%.4f (parity medians)",
                    result["n_spx_options"], result["n_spx_expiries"], result["n_spx_puts"],
                    result["n_spx_wing_puts"], _PUT_WING_KF, result["n_vix_tenors"],
                    result["r"], result["q"])
        logger.info("  SPX smile RMSE: in-sample %.3f vp (exact %.3f) | out-of-sample %.3f vp "
                    "over %d quotes (put wing %.3f)", ll["spx_iv_rmse"], ll["spx_iv_rmse_exact"],
                    ll["spx_oos_iv_rmse"], result["oos"]["n_spx_oos"], ll["spx_oos_put_wing_rmse"])
        logger.info("  VIX index RMSE: %.3f VIX pts | total loss %.6g | fit %.1fs (%d evals)",
                    ll["vix_index_rmse"], ll["total_loss"], result["fit_time"], result["n_evals"])

    def __repr__(self) -> str:
        if self.params is None:
            return f"JointCalibrator(as_of={self.as_of_date}, not calibrated)"
        p = self.params
        return (f"JointCalibrator(as_of={self.as_of_date}, "
                f"kappa={p['kappa']:.3f}, theta={p['theta']:.4f}, "
                f"sigma={p['sigma']:.3f}, rho={p['rho']:.3f}, v0={p['v0']:.4f})")
