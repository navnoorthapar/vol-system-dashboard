"""
quintic_calibrator.py — Joint calibration of the two-factor Quintic OU model (C13).

Mirrors the DE + L-BFGS-B structure of joint_calibrator.py but targets the
QuinticOU model (Abi Jaber & Li 2025, arXiv:2503.14158, with the odd-plus-
constant polynomial of Abi Jaber, Illand & Li 2022, arXiv:2212.10917).

Inputs, all taken from ONE session (the SPX spot session ≤ as_of_date):
  • ξ_0(t): piecewise-constant forward variance bootstrapped from the Cboe VIX
    indices (^VIX9D/^VIX/^VIX3M/^VIX6M). It is an INPUT, not fitted: the model
    reproduces those index levels by construction, which is reported as an
    input check (xi0_index_reproduction_rmse), not as a fit metric.
  • SPX smile: per expiry, the forward F and discount D come from put-call
    parity on the same chain (D = e^{-rT}, r = ^IRX); OTM mids are inverted
    with Black-76 on F. Strikes are chosen on a log-moneyness grid that
    reaches into the put wing (K/F ≈ 0.75–0.9), not the top-vega strikes.
  • VIX forwards: there are no VX futures in the DB, so the per-expiry VIX
    forward is the put-call-parity forward of the VIX option chain.
  • VIX smile: OTM, liquid VIX option mids inverted with Black-76 on that
    forward (the vendor `implied_vol` column is Black-Scholes on SPOT VIX and
    is not used).

Free parameters (7): λx, λy, θ, a0, a1, a3, ε, where p is normalised so that
α5 = 1 (the model is invariant to rescaling p) and a_k are the remaining
coefficients in units of the stationary std s of Z:
    p(z) ∝ a0 + a1 (z/s) + a3 (z/s)³ + (z/s)⁵   ⇔   α_k = a_k s^{5-k}, α5 = 1.
a_k ≥ 0 and ε ≤ 0 keep the fit inside the set where S is a true martingale
and ε is the sign of the spot-vol correlation (see models/quintic_ou.py).

Loss (all three legs depend on the parameters):
    L = w1 · mean((σ_model − σ_mkt)²)                       SPX Black-76 IVs
      + w2 · mean(((E[VIX_T] − F_mkt) / F_mkt)²)             VIX forwards
      + w3 · mean((σ_model − σ_mkt)²)                       VIX Black-76 IVs
A model IV that cannot be computed counts as a fixed residual of _NAN_RESIDUAL
instead of being dropped (dropping it let the optimiser improve the loss by
pushing hard quotes into pricing failure).
VIX smiles are compared at equal moneyness K/F (model IV at K·F_model/F_mkt on
the model's own forward), as in the authors' reference code; the forward misfit
is the w2 leg.

Zero look-ahead: all data are as of as_of_date, and every leg is pinned to the
SPX spot session (stale snapshots or missing tenors raise QuinticDataError).
"""

import logging
import math
import pickle
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from joint_vol_calibration.config import (
    DATA_DIR, JOINT_W1, JOINT_W2, RANDOM_SEED, VIX_INDEX_TENOR_DAYS,
)
from joint_vol_calibration.data import database as db
from joint_vol_calibration.models.quintic_ou import (
    QuinticOUParams, QUINTIC_BOUNDS, QUINTIC_DEFAULTS, MC_MIN_STEPS, MC_STEPS_PER_YEAR,
    admissibility_violations, black76_implied_vol, implied_vol_surface_forward_mc,
    stationary_var_Z, vix_at_nodes,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_CAL_EXPIRY_TARGETS_DAYS = [14, 30, 60, 91, 180, 365]
_MIN_T_DAYS   = 7
_MAX_T_YEARS  = 2.0
_MIN_IV  = 0.01
_MAX_IV  = 3.00
_MIN_PRICE = 0.50              # SPX: discard OTM mids below $0.50

# SPX strike grid per expiry: _SPX_N_STRIKES targets evenly spaced in k = ln(K/F)
# on [-clip(_SPX_SD_LO·σ√T, _SPX_K_LO_MIN, _SPX_K_LO_MAX), +min(_SPX_K_HI_MAX, _SPX_SD_HI·σ√T)],
# σ = ATM Black-76 vol of that expiry: the put wing reaches K/F ≈ 0.89 or
# deeper at every expiry (0.74 at one year), where the quintic map's skew
# is identified. Targets beyond the quoted range snap to the deepest strike.
_SPX_N_STRIKES = 11
_SPX_SD_LO, _SPX_K_LO_MIN, _SPX_K_LO_MAX = 2.5, 0.12, 0.30
_SPX_SD_HI, _SPX_K_HI_MAX = 1.0, 0.10
_SPX_MIN_STRIKES = 5

# VIX options: expiries in [7d, 1y]; OTM; liquid; strikes on a k = ln(K/F) grid
# on [-0.1, 1.0] (the range of the authors' demo); forward from put-call parity.
_VIX_MAX_T_YEARS = 1.0
_VIX_K_LO, _VIX_K_HI, _VIX_N_STRIKES = -0.10, 1.00, 9
_VIX_MAX_REL_SPREAD = 0.30
_VIX_MIN_STRIKES = 3

# Put-call-parity forward: median of K + (C − P)/D over the strikes nearest the
# money; rejected if those estimates disagree by more than _PCP_MAX_DISPERSION·F.
_PCP_N_STRIKES = {"SPX": 5, "VIX": 3}
_PCP_MAX_DISPERSION = 0.01

# Data-sufficiency gate (C20-style): a joint fit needs a curve on every leg.
_MIN_SPX_EXPIRIES = 3
_MIN_VIX_TENORS   = 3
_MIN_VIX_EXPIRIES = 2

# Loss
_NAN_RESIDUAL = 0.50           # 50 vol points for a model IV that cannot be computed
_QUINTIC_W3 = 0.3              # VIX-options weight (JOINT_W3 = 0 is Heston-specific)

# Monte Carlo / quadrature, used identically in the objective and the report.
_MC_PATHS_CAL   = 10_000       # 5,000 antithetic pairs; CRN (fixed seed) every evaluation
_MC_PATHS_CHECK = 40_000       # fresh-seed re-evaluation of the SPX leg at the optimum
_VIX_QUAD_OUTER = 101          # lattice points per dim of the VIX plane cubature
_VIX_QUAD_INNER = 20           # GL points per smooth piece of the VIX time integral

# DE settings (population seeded with the Section 3.2 fit of Abi Jaber & Li)
_DE_MAXITER  = 60
_DE_POPSIZE  = 8               # 8 × 7 = 56 members
_DE_TOL      = 1e-4
_DE_MUTATION = (0.5, 1.0)
_DE_RECOMB   = 0.7
_LBFGS_MAXITER = 30

# Search space: x = (λx, λy, θ, a0, a1, a3, ε)
SEARCH_NAMES = ["lam_x", "lam_y", "theta", "a0", "a1", "a3", "epsilon"]
SEARCH_BOUNDS = [
    QUINTIC_BOUNDS[0],         # lam_x
    QUINTIC_BOUNDS[1],         # lam_y
    QUINTIC_BOUNDS[2],         # theta ∈ [0, 1]
    (0.0, 60.0),               # a0  } coefficients of p(s·u) / s⁵ in the standardised
    (0.0, 60.0),               # a1  } variable u = Z/s (s = stationary std of Z);
    (0.0, 20.0),               # a3  } the Section 3.2 fit of Abi Jaber & Li ≈ (9.6, 14.7, 0.5)
    QUINTIC_BOUNDS[7],         # epsilon ∈ [-0.99, 0]
]


class QuinticDataError(ValueError):
    """Market data are stale, mixed-date or insufficient for a joint fit."""


def search_to_params(x) -> QuinticOUParams:
    """Map the 7-D search vector to model parameters (α5 = 1)."""
    lam_x, lam_y, theta, a0, a1, a3, eps = (float(v) for v in x)
    s = math.sqrt(stationary_var_Z(lam_x, lam_y, theta))
    return QuinticOUParams(lam_x=lam_x, lam_y=lam_y, theta=theta,
                           alpha0=a0 * s**5, alpha1=a1 * s**4, alpha3=a3 * s**2,
                           alpha5=1.0, epsilon=eps)


def params_to_search(params: QuinticOUParams) -> np.ndarray:
    """Inverse of search_to_params (normalises p to α5 = 1 first)."""
    p = params.normalised()
    s = math.sqrt(stationary_var_Z(p.lam_x, p.lam_y, p.theta))
    return np.array([p.lam_x, p.lam_y, p.theta,
                     p.alpha0 / s**5, p.alpha1 / s**4, p.alpha3 / s**2, p.epsilon])


class _PiecewiseConstVar:
    """
    Picklable piecewise-constant forward-variance curve ξ₀(t).

    Built by bootstrapping a term structure of average variances so that
    100·sqrt((1/T)∫₀ᵀ ξ₀) reproduces the input level at every input tenor.

    A plain closure cannot be pickled, which blocks scipy's `workers=-1`
    multiprocessing; this small class is picklable so the calibrator can run
    the DE population in parallel.

    Step convention: ξ₀ on interval (t_nodes[i-1], t_nodes[i]] equals xi_nodes[i];
    flat extrapolation beyond the last node. `integrated(t)` = ∫₀ᵗ ξ₀ exactly,
    and `breakpoints` lists the jump times (used by the pricers to integrate
    across the jumps exactly).
    """

    def __init__(self, t_nodes: np.ndarray, xi_nodes: np.ndarray):
        self.t_nodes  = np.asarray(t_nodes, dtype=float)
        self.xi_nodes = np.asarray(xi_nodes, dtype=float)

    def __call__(self, t: float) -> float:
        idx = int(np.searchsorted(self.t_nodes, t, side="left"))
        if idx >= len(self.xi_nodes):
            idx = len(self.xi_nodes) - 1
        return float(self.xi_nodes[idx])

    @property
    def breakpoints(self) -> np.ndarray:
        return self.t_nodes[:-1]

    def integrated(self, t: float) -> float:
        """∫₀ᵗ ξ₀(s) ds."""
        if t <= 0.0:
            return 0.0
        edges = np.concatenate([[0.0], self.t_nodes])
        total = 0.0
        for i, xi in enumerate(self.xi_nodes):
            lo, hi = edges[i], edges[i + 1]
            if t <= lo:
                return total
            total += xi * (min(t, hi) - lo)
        if t > edges[-1]:
            total += self.xi_nodes[-1] * (t - edges[-1])
        return total


def _bootstrap_fwd_var(tenors, vols) -> _PiecewiseConstVar:
    """
    Piecewise-constant forward variance from average (term) vols:
    I(T_i) = T_i σ_i², ξ₀ on (T_{i-1}, T_i] = (I(T_i) − I(T_{i-1})) / (T_i − T_{i-1}),
    floored at 1e-6 when the term structure is too steeply inverted (the curve
    then no longer reproduces that input; the index check reports it).
    """
    T = np.asarray(tenors, dtype=float)
    v = np.asarray(vols, dtype=float)
    order = np.argsort(T)
    T, v = T[order], v[order]
    cum = T * v**2
    xi, prev_T, prev_I = [], 0.0, 0.0
    for Ti, Ii in zip(T, cum):
        dT = Ti - prev_T
        fwd = (Ii - prev_I) / dT if dT > 1e-9 else Ii / max(Ti, 1e-9)
        xi.append(max(fwd, 1e-6))
        prev_T, prev_I = Ti, Ii
    return _PiecewiseConstVar(T, np.array(xi))


def _spot_vix(T: float, fwd_var_func) -> float:
    """
    Spot-starting variance-swap vol to horizon T: 100·sqrt((1/T) ∫_0^T ξ₀(s) ds).

    This is what the Cboe ^VIX/^VIX3M-style indices measure (variance from
    today to T), NOT a VIX future. Since E[σ²(s)] = ξ₀(s) by the g₀
    normalisation it is model-free, so it checks the ξ₀ input, not the fit.
    Exact for curves exposing `integrated`; otherwise a 200-point midpoint rule.
    """
    integ = getattr(fwd_var_func, "integrated", None)
    if integ is not None:
        integral = float(integ(T))
    else:
        n = 200
        dt = T / n
        pts = np.linspace(dt * 0.5, T - dt * 0.5, n)
        integral = float(np.sum([fwd_var_func(s) for s in pts]) * dt)
    return 100.0 * math.sqrt(max(integral / T, 0.0))


def _pcp_forward(chain: pd.DataFrame, discount: float, n_strikes: int
                 ) -> Tuple[Optional[float], float, int]:
    """
    Forward from put-call parity C − P = D(F − K) on one expiry's chain.

    F_K = K + (C − P)/D at every strike quoted on both sides. Start from the
    n strikes with the smallest |C − P|, re-centre on the n strikes nearest that
    first estimate, and return the median there, with the median absolute
    deviation as a dispersion measure. Returns (None, nan, 0) if fewer than two
    strikes have both quotes.
    """
    c = chain[chain["right"] == "C"].groupby("strike")["mid_price"].mean()
    p = chain[chain["right"] == "P"].groupby("strike")["mid_price"].mean()
    common = c.index.intersection(p.index)
    if len(common) < 2:
        return None, float("nan"), 0
    K = common.values.astype(float)
    diff = (c.loc[common] - p.loc[common]).values.astype(float)
    Fk = K + diff / discount
    n = min(n_strikes, len(K))
    F0 = float(np.median(Fk[np.argsort(np.abs(diff))[:n]]))
    near = np.argsort(np.abs(K - F0))[:n]
    F = float(np.median(Fk[near]))
    disp = float(np.median(np.abs(Fk[near] - F)))
    return F, disp, n


def _pick_on_grid(k_avail: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Indices of the available points nearest each target (deduplicated, sorted)."""
    if len(k_avail) == 0:
        return np.array([], dtype=int)
    picks = {int(np.argmin(np.abs(k_avail - t))) for t in targets}
    return np.array(sorted(picks), dtype=int)


def _git_revision() -> dict:
    """Best-effort code provenance: HEAD SHA and whether the quintic files are dirty."""
    root = Path(__file__).resolve().parents[2]
    out = {"sha": None, "dirty_files": None}
    try:
        out["sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                    text=True, timeout=10).stdout.strip() or None
        st = subprocess.run(["git", "status", "--porcelain", "--",
                             "joint_vol_calibration/models/quintic_ou.py",
                             "joint_vol_calibration/calibration/quintic_calibrator.py"],
                            cwd=root, capture_output=True, text=True, timeout=10).stdout
        out["dirty_files"] = [ln[3:] for ln in st.splitlines() if ln.strip()]
    except Exception:          # git missing or not a checkout: provenance stays None
        pass
    return out


def recorded_vix_index_levels(result: dict) -> dict:
    """{ticker: level} of the spot VIX indices a stored Quintic fit recorded
    (result["provenance"]["vix_index_levels"]), for
    QuinticCalibrator(..., vix_index_levels=...)."""
    rows = (result.get("provenance") or {}).get("vix_index_levels") or []
    out = {str(r["ticker"]): float(r["vix_level"]) for r in rows}
    if not out:
        raise QuinticDataError("the result records no VIX index levels")
    return out


class QuinticCalibrator:
    """
    Joint SPX/VIX calibrator for the two-factor Quintic OU model.

    Usage
    -----
        cal = QuinticCalibrator(as_of_date='2026-03-24')
        result = cal.calibrate()
        cal.save()
    """

    def __init__(
        self,
        as_of_date:  str,
        r:           Optional[float] = None,
        q:           float = 0.013,
        w1:          float = JOINT_W1,
        w2:          float = JOINT_W2,
        w3:          float = _QUINTIC_W3,
        n_paths:     int = _MC_PATHS_CAL,
        seed:        int = RANDOM_SEED,
        allow_stale: bool = False,
        vix_index_levels: dict | None = None,
    ):
        """
        Parameters
        ----------
        as_of_date : 'YYYY-MM-DD'; the fit describes the last SPX session ≤ this date
        r          : discount rate for the market quotes; None → ^IRX on the spot session
        q          : ignored for pricing (forwards come from put-call parity);
                     kept for call compatibility
        w1, w2, w3 : weights of the SPX-IV, VIX-forward and VIX-IV legs
        n_paths    : SPX Monte Carlo paths per evaluation (antithetic pairs × 2)
        seed       : common-random-numbers seed used in every evaluation
        allow_stale: skip the same-session checks (research use only; recorded)
        vix_index_levels: {ticker: level} replacing the database's spot VIX
                     indices (the same tickers, else QuinticDataError), e.g.
                     recorded_vix_index_levels(result) to rebuild a committed
                     fit's xi0 from the levels it recorded: C23 restated the
                     database's Yahoo float32 prints as Cboe's closes
                     (<= 1e-5 pts apart). None (default): the database
        """
        self.as_of_date = as_of_date
        self.r_override = r
        self.q  = q
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.n_paths = int(n_paths)
        self.seed = int(seed)
        self.allow_stale = bool(allow_stale)
        self._vix_index_levels = (None if vix_index_levels is None
                                  else {str(k): float(v) for k, v in vix_index_levels.items()})
        self.steps_per_year = MC_STEPS_PER_YEAR
        self.min_steps = MC_MIN_STEPS
        self.n_outer = _VIX_QUAD_OUTER
        self.n_inner = _VIX_QUAD_INNER

        self.S: float = float("nan")
        self.r: float = float("nan")
        self.spot_date: Optional[str] = None
        self.spx_snapshot_date: Optional[str] = None
        self.vix_snapshot_date: Optional[str] = None
        self.spx_surface:       pd.DataFrame = pd.DataFrame()
        self.vix_ts:            pd.DataFrame = pd.DataFrame()
        self.vix_opts_surface:  pd.DataFrame = pd.DataFrame()
        self.vix_forwards:      pd.DataFrame = pd.DataFrame()
        self.fwd_var_func = None
        self.result = None
        self._n_evals = 0

        logger.info("[C13] QuinticCalibrator: loading market data for %s", as_of_date)
        self._load_market_data()

    # ── Market data ───────────────────────────────────────────────────────────

    def _load_market_data(self):
        spx_hist = db.get_spx_ohlcv(as_of_date=self.as_of_date)
        if spx_hist.empty:
            raise QuinticDataError(f"No SPX OHLCV data for {self.as_of_date}")
        self.S = float(spx_hist.iloc[-1]["close"])
        self.spot_date = str(spx_hist.iloc[-1]["date"])[:10]
        self.r = (float(self.r_override) if self.r_override is not None
                  else float(db.get_tbill_rate(self.spot_date)))
        logger.info("  SPX spot S = %.2f (session %s), r = %.4f", self.S, self.spot_date, self.r)

        self.spx_surface = self._prepare_spx_surface()
        self.vix_ts = self._prepare_vix_term_structure()
        self.vix_opts_surface = self._prepare_vix_options()
        self._check_data()
        self.fwd_var_func = self._build_fwd_var_func()

        repro = self.xi0_index_reproduction()
        if repro > 0.01:
            logger.warning("  ξ0 does not reproduce the VIX indices (RMSE %.4f pts): the "
                           "bootstrap floored an inverted segment", repro)
        logger.info("  SPX: %d options / %d expiries | VIX indices: %d tenors | "
                    "VIX options: %d / %d expiries",
                    len(self.spx_surface), self._n_expiries(self.spx_surface),
                    len(self.vix_ts), len(self.vix_opts_surface), len(self.vix_forwards))

    @staticmethod
    def _n_expiries(df: pd.DataFrame) -> int:
        return int(df["time_to_expiry"].nunique()) if not df.empty else 0

    def _check_data(self):
        """Same-session and sufficiency gate (the C20/C21 guards of JointCalibrator)."""
        problems = []
        if not self.allow_stale:
            if self.spx_snapshot_date != self.spot_date:
                problems.append(f"SPX option snapshot {self.spx_snapshot_date} is not the "
                                f"spot session {self.spot_date}")
            if self.vix_snapshot_date != self.spot_date:
                problems.append(f"VIX option snapshot {self.vix_snapshot_date} is not the "
                                f"spot session {self.spot_date}")
        if self._n_expiries(self.spx_surface) < _MIN_SPX_EXPIRIES:
            problems.append(f"{self._n_expiries(self.spx_surface)} usable SPX expiries "
                            f"(< {_MIN_SPX_EXPIRIES})")
        if len(self.vix_ts) < _MIN_VIX_TENORS:
            problems.append(f"{len(self.vix_ts)} VIX index tenors on {self.spot_date} "
                            f"(< {_MIN_VIX_TENORS})")
        if len(self.vix_forwards) < _MIN_VIX_EXPIRIES:
            problems.append(f"{len(self.vix_forwards)} usable VIX option expiries "
                            f"(< {_MIN_VIX_EXPIRIES})")
        if problems:
            raise QuinticDataError(f"Quintic calibration refused for {self.as_of_date}: "
                                   + "; ".join(problems))

    def _prepare_spx_surface(self) -> pd.DataFrame:
        """
        SPX calibration set: per target expiry, the put-call-parity forward and
        Black-76 IVs of OTM mids on a log-moneyness grid (see module docstring).
        """
        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="SPX")
        if raw.empty:
            self.spx_snapshot_date = None
            return pd.DataFrame()
        self.spx_snapshot_date = str(raw["snapshot_date"].iloc[0])[:10]

        df = raw[(raw["time_to_expiry"] >= _MIN_T_DAYS / 365.0)
                 & (raw["time_to_expiry"] <= _MAX_T_YEARS)
                 & (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
                 & (raw["mid_price"] > 0)].copy()
        if df.empty:
            return pd.DataFrame()

        available = np.array(sorted(df["time_to_expiry"].unique()))
        chosen = []
        for target_days in _CAL_EXPIRY_TARGETS_DAYS:
            target_T = target_days / 365.0
            best_T = float(available[np.argmin(np.abs(available - target_T))])
            if abs(best_T - target_T) <= 0.5 * target_T and best_T not in chosen:
                chosen.append(best_T)

        out = []
        for T in chosen:
            sl = df[df["time_to_expiry"] == T]
            D = math.exp(-self.r * T)
            F, disp, n_used = _pcp_forward(sl, D, _PCP_N_STRIKES["SPX"])
            if F is None or not disp <= _PCP_MAX_DISPERSION * F:
                logger.warning("  SPX T=%.4f: no clean put-call-parity forward (disp %.2f) — "
                               "expiry skipped", T, disp)
                continue
            otm = sl[((sl["right"] == "P") & (sl["strike"] < F))
                     | ((sl["right"] == "C") & (sl["strike"] >= F))]
            otm = otm[otm["mid_price"] > _MIN_PRICE].copy()
            if otm.empty:
                continue
            otm["market_iv"] = black76_implied_vol(
                otm["mid_price"].values, F, otm["strike"].values, T,
                (otm["right"] == "C").values, D)
            otm = otm[otm["market_iv"].between(_MIN_IV, _MAX_IV)]
            if len(otm) < _SPX_MIN_STRIKES:
                continue
            k = np.log(otm["strike"].values / F)
            atm_iv = float(otm["market_iv"].values[np.argmin(np.abs(k))])
            sd = atm_iv * math.sqrt(T)
            k_lo = min(_SPX_K_LO_MAX, max(_SPX_K_LO_MIN, _SPX_SD_LO * sd))
            targets = np.linspace(-k_lo, min(_SPX_K_HI_MAX, _SPX_SD_HI * sd), _SPX_N_STRIKES)
            sel = otm.iloc[_pick_on_grid(k, targets)].copy()
            if len(sel) < _SPX_MIN_STRIKES:
                continue
            sel["forward"] = F
            sel["discount"] = D
            sel["pcp_dispersion"] = disp
            sel["log_moneyness"] = np.log(sel["strike"].values / F)
            out.append(sel)

        if not out:
            return pd.DataFrame()
        cal = pd.concat(out, ignore_index=True)
        cal = cal.rename(columns={"implied_vol": "vendor_iv"})
        keep = ["expiry", "time_to_expiry", "strike", "right", "bid", "ask", "mid_price",
                "forward", "discount", "pcp_dispersion", "log_moneyness", "market_iv",
                "vendor_iv"]
        return cal[[c for c in keep if c in cal.columns]].sort_values(
            ["time_to_expiry", "strike"]).reset_index(drop=True)

    def _prepare_vix_term_structure(self) -> pd.DataFrame:
        """
        Cboe VIX index levels on the SPX spot session (no borrowing from other
        dates: a missing row or tenor leaves it out, and the ≥3-tenor gate
        decides). Tenors from config.VIX_INDEX_TENOR_DAYS (calendar days).
        """
        vts = db.get_vix_term_structure(as_of_date=self.as_of_date)
        if vts.empty:
            return pd.DataFrame(columns=["ticker", "tenor", "vix_level"])
        on_spot = vts[vts["date"].astype(str).str[:10] == self.spot_date]
        if on_spot.empty:
            logger.warning("  VIX indices have no row on spot session %s (latest %s)",
                           self.spot_date, str(vts["date"].iloc[-1])[:10])
        rows = []
        for ticker, days in VIX_INDEX_TENOR_DAYS.items():
            row = on_spot[on_spot["ticker"] == ticker]
            if not row.empty and pd.notna(row.iloc[-1]["close"]) and row.iloc[-1]["close"] > 0:
                rows.append({"ticker": ticker, "tenor": days / 365.0,
                             "vix_level": float(row.iloc[-1]["close"])})
        levels = getattr(self, "_vix_index_levels", None)
        if levels is not None:
            have = {str(r["ticker"]) for r in rows}
            if have != set(levels):
                raise QuinticDataError(f"vix_index_levels {sorted(levels)} do not "
                                       f"match the database's VIX indices on {self.spot_date}: "
                                       f"{sorted(have)}")
            for r in rows:
                r["vix_level"] = levels[str(r["ticker"])]
        return pd.DataFrame(rows, columns=["ticker", "tenor", "vix_level"])

    def _prepare_vix_options(self) -> pd.DataFrame:
        """
        VIX calibration set: per expiry in [7d, 1y], the put-call-parity forward
        (stored in self.vix_forwards) and Black-76 IVs of liquid OTM mids on a
        log-moneyness grid k ∈ [-0.1, 1.0] relative to that forward.
        """
        self.vix_forwards = pd.DataFrame()
        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="VIX")
        if raw.empty:
            self.vix_snapshot_date = None
            return pd.DataFrame()
        self.vix_snapshot_date = str(raw["snapshot_date"].iloc[0])[:10]
        df = raw[(raw["time_to_expiry"] >= _MIN_T_DAYS / 365.0)
                 & (raw["time_to_expiry"] <= _VIX_MAX_T_YEARS)
                 & (raw["bid"] > 0) & (raw["ask"] >= raw["bid"])
                 & (raw["mid_price"] > 0)].copy()

        out, fwd_rows = [], []
        for T in sorted(df["time_to_expiry"].unique()):
            sl = df[df["time_to_expiry"] == T]
            D = math.exp(-self.r * T)
            F, disp, n_used = _pcp_forward(sl, D, _PCP_N_STRIKES["VIX"])
            if F is None or not disp <= _PCP_MAX_DISPERSION * F:
                continue
            otm = sl[((sl["right"] == "P") & (sl["strike"] < F))
                     | ((sl["right"] == "C") & (sl["strike"] >= F))].copy()
            liquid = ((otm["ask"] - otm["bid"]) / otm["mid_price"] < _VIX_MAX_REL_SPREAD) & (
                (otm["volume"].fillna(0) > 0) | (otm["open_interest"].fillna(0) > 0))
            otm = otm[liquid]
            if otm.empty:
                continue
            otm["market_iv"] = black76_implied_vol(
                otm["mid_price"].values, F, otm["strike"].values, T,
                (otm["right"] == "C").values, D)
            otm = otm[otm["market_iv"].between(_MIN_IV, _MAX_IV)]
            k = np.log(otm["strike"].values / F)
            in_range = (k >= _VIX_K_LO - 0.02) & (k <= _VIX_K_HI + 0.05)
            otm, k = otm[in_range], k[in_range]
            sel = otm.iloc[_pick_on_grid(k, np.linspace(_VIX_K_LO, _VIX_K_HI, _VIX_N_STRIKES))].copy()
            if len(sel) < _VIX_MIN_STRIKES:
                continue
            sel["forward"] = F
            sel["discount"] = D
            sel["log_moneyness"] = np.log(sel["strike"].values / F)
            out.append(sel)
            fwd_rows.append({"expiry": str(sl["expiry"].iloc[0])[:10], "time_to_expiry": float(T),
                             "forward": F, "pcp_dispersion": disp, "pcp_strikes": n_used})
        self.vix_forwards = pd.DataFrame(fwd_rows)
        if not out:
            return pd.DataFrame()
        cal = pd.concat(out, ignore_index=True).rename(columns={"implied_vol": "vendor_iv"})
        keep = ["expiry", "time_to_expiry", "strike", "right", "bid", "ask", "mid_price",
                "volume", "forward", "discount", "log_moneyness", "market_iv", "vendor_iv"]
        return cal[[c for c in keep if c in cal.columns]].sort_values(
            ["time_to_expiry", "strike"]).reset_index(drop=True)

    def _build_fwd_var_func(self) -> "_PiecewiseConstVar":
        """
        Bootstrap the forward-variance curve ξ_0(t) from the VIX indices.

        The Cboe VIX indices are spot-starting variance-swap vols:
        VIX(T)² = (100²/T)·∫₀ᵀ ξ_0(s) ds, so the cumulative variance
        I(T_i) = T_i·(VIX(T_i)/100)² bootstraps a piecewise-constant ξ_0 that
        reproduces every index level (an input, not a fit: the parameters
        cannot change it). The fit then constrains the SHAPE of the smiles and
        the VIX forwards.

        Falls back to the same bootstrap on ATM SPX Black-76 vols if the VIX
        indices are unavailable (only reachable when the data gate is bypassed),
        and to a flat 20% vol if there is no surface at all.
        """
        if not self.vix_ts.empty:
            return _bootstrap_fwd_var(self.vix_ts["tenor"].values,
                                      self.vix_ts["vix_level"].values / 100.0)
        if self.spx_surface.empty:
            return _PiecewiseConstVar(np.array([1.0]), np.array([0.04]))
        surf = self.spx_surface
        k = (surf["log_moneyness"] if "log_moneyness" in surf.columns
             else np.log(surf["strike"] / self.S))
        iv_col = "market_iv" if "market_iv" in surf.columns else "implied_vol"
        atm = (surf.assign(_absk=np.abs(k)).sort_values("_absk")
               .groupby("time_to_expiry", as_index=False).first())
        return _bootstrap_fwd_var(atm["time_to_expiry"].values, atm[iv_col].values)

    def xi0_index_reproduction(self) -> float:
        """RMSE (index points) of _spot_vix(ξ0) against the input VIX index levels."""
        if self.vix_ts.empty or self.fwd_var_func is None:
            return float("nan")
        errs = [(_spot_vix(float(r.tenor), self.fwd_var_func) - float(r.vix_level))**2
                for r in self.vix_ts.itertuples()]
        return float(math.sqrt(np.mean(errs)))

    # ── Model evaluation (shared by the objective and the report) ─────────────

    def evaluate(self, params: QuinticOUParams, n_paths: Optional[int] = None,
                 seed: Optional[int] = None) -> dict:
        """
        Model values and residuals of every leg at `params`, with the
        calibration's MC and quadrature settings (overridable for re-checks).
        """
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        seed = self.seed if seed is None else int(seed)
        fv = self.fwd_var_func
        out = {}

        if not self.spx_surface.empty:
            s = self.spx_surface
            iv, se = implied_vol_surface_forward_mc(
                s["time_to_expiry"].values, s["strike"].values, s["forward"].values,
                params, fv, n_paths=n_paths, seed=seed,
                steps_per_year=self.steps_per_year, min_steps=self.min_steps,
                return_stderr=True)
            resid = iv - s["market_iv"].values
            out["spx_model_iv"] = iv
            out["spx_model_iv_se"] = se
            out["spx_resid"] = np.where(np.isfinite(resid), resid, _NAN_RESIDUAL)
            out["spx_n_nan"] = int(np.sum(~np.isfinite(iv)))

        if not self.vix_forwards.empty:
            F_model = np.empty(len(self.vix_forwards))
            vix_iv = np.full(len(self.vix_opts_surface), np.nan)
            v = self.vix_opts_surface
            for i, row in enumerate(self.vix_forwards.itertuples()):
                T, F_mkt = float(row.time_to_expiry), float(row.forward)
                vix, w = vix_at_nodes(T, params, fv, self.n_outer, self.n_inner)
                Fm = float(np.dot(w, vix))
                F_model[i] = Fm
                mask = (v["time_to_expiry"] == T).values
                if not mask.any():
                    continue
                # Equal moneyness: the market strike K on F_mkt ↔ K·F_model/F_mkt on F_model.
                K_model = v["strike"].values[mask] * (Fm / F_mkt)
                is_call = v["right"].values[mask] == "C"
                payoff = np.where(is_call[None, :], vix[:, None] - K_model[None, :],
                                  K_model[None, :] - vix[:, None])
                prem = np.maximum(payoff, 0.0).T @ w          # undiscounted premia
                vix_iv[mask] = black76_implied_vol(prem, Fm, K_model, T, is_call, 1.0)
            F_mkt_all = self.vix_forwards["forward"].values
            out["vix_forward_model"] = F_model
            out["vix_fwd_rel_resid"] = (F_model - F_mkt_all) / F_mkt_all
            out["vix_fwd_resid_pts"] = F_model - F_mkt_all
            out["vix_model_iv"] = vix_iv
            resid = vix_iv - v["market_iv"].values
            out["vix_resid"] = np.where(np.isfinite(resid), resid, _NAN_RESIDUAL)
            out["vix_n_nan"] = int(np.sum(~np.isfinite(vix_iv)))
        return out

    def _loss_from_eval(self, ev: dict) -> float:
        total = 0.0
        if self.w1 > 0 and "spx_resid" in ev:
            total += self.w1 * float(np.mean(ev["spx_resid"]**2))
        if self.w2 > 0 and "vix_fwd_rel_resid" in ev:
            total += self.w2 * float(np.mean(ev["vix_fwd_rel_resid"]**2))
        if self.w3 > 0 and "vix_resid" in ev:
            total += self.w3 * float(np.mean(ev["vix_resid"]**2))
        return total

    def _loss(self, x: np.ndarray) -> float:
        """Weighted joint loss at the 7-D search vector x (see module docstring)."""
        self._n_evals += 1
        for (lo, hi), v in zip(SEARCH_BOUNDS, x):
            if not (lo <= v <= hi):
                return 1e8
        try:
            return self._loss_from_eval(self.evaluate(search_to_params(x)))
        except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError) as exc:
            logger.debug("loss evaluation failed at %s: %s", x, exc)
            return 1e8

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self, maxiter: int = _DE_MAXITER, popsize: int = _DE_POPSIZE,
                  polish_maxiter: int = _LBFGS_MAXITER, workers: int = -1,
                  disp: bool = False) -> dict:
        """
        Global DE over the 7-D search space (initial population seeded with the
        published Section 3.2 point, QUINTIC_DEFAULTS), then an L-BFGS-B polish.
        The leg metrics are computed with the SAME settings as the objective, so
        the reported loss is the minimised value; the SPX leg is also re-priced
        on fresh paths (_MC_PATHS_CHECK, another seed) to expose any MC overfit.

        Returns the result dict described in `save`.
        """
        logger.info("[C13] Quintic calibration %s | S=%.2f | %d SPX opts | %d VIX opts / %d expiries",
                    self.spot_date, self.S, len(self.spx_surface),
                    len(self.vix_opts_surface), len(self.vix_forwards))
        t0, c0 = time.perf_counter(), time.process_time()
        self._n_evals = 0

        x0 = params_to_search(QUINTIC_DEFAULTS)
        de_result = optimize.differential_evolution(
            self._loss, bounds=SEARCH_BOUNDS, maxiter=maxiter, popsize=popsize,
            tol=_DE_TOL, mutation=_DE_MUTATION, recombination=_DE_RECOMB,
            seed=RANDOM_SEED, polish=False, init="latinhypercube", x0=x0, disp=disp,
            workers=workers, updating="deferred" if workers != 1 else "immediate",
        )
        logger.info("     DE: loss=%.6g nfev=%d nit=%d success=%s (%s)", de_result.fun,
                    de_result.nfev, de_result.nit, de_result.success, de_result.message)
        pol = optimize.minimize(self._loss, de_result.x, method="L-BFGS-B", bounds=SEARCH_BOUNDS,
                                options={"maxiter": polish_maxiter, "ftol": 1e-10, "gtol": 1e-7})
        logger.info("     L-BFGS-B: loss=%.6g nfev=%d success=%s (%s)", pol.fun, pol.nfev,
                    pol.success, pol.message)
        best_x = np.asarray(pol.x if pol.fun <= de_result.fun else de_result.x, dtype=float)
        fit_time, fit_cpu = time.perf_counter() - t0, time.process_time() - c0

        params = search_to_params(best_x)
        ev = self.evaluate(params)
        loss = self._loss_from_eval(ev)
        ev_fresh = self.evaluate(params, n_paths=_MC_PATHS_CHECK, seed=self.seed + 1000)

        def _rmse(a):
            a = np.asarray(a, dtype=float)
            return float(math.sqrt(np.mean(a**2))) if a.size else float("nan")

        spx_fresh_resid = ev_fresh.get("spx_resid", np.array([]))
        leg_losses = {
            "spx_iv_rmse":                 100.0 * _rmse(ev.get("spx_resid", [])),
            "spx_iv_rmse_fresh_paths":     100.0 * _rmse(spx_fresh_resid),
            "vix_futures_rmse":            _rmse(ev.get("vix_fwd_resid_pts", [])),
            "vix_futures_rel_rmse":        _rmse(ev.get("vix_fwd_rel_resid", [])),
            "vix_options_rmse":            100.0 * _rmse(ev.get("vix_resid", [])),
            "xi0_index_reproduction_rmse": self.xi0_index_reproduction(),
        }
        span = {n: (lo, hi) for n, (lo, hi) in zip(SEARCH_NAMES, SEARCH_BOUNDS)}
        at_bound = {n: bool(min(abs(v - lo), abs(hi - v)) <= 0.01 * (hi - lo))
                    for n, v, (lo, hi) in zip(SEARCH_NAMES, best_x, SEARCH_BOUNDS)}

        spx_rows = []
        for i, row in enumerate(self.spx_surface.itertuples()):
            spx_rows.append({"expiry": str(row.expiry)[:10], "T": float(row.time_to_expiry),
                             "K": float(row.strike), "right": row.right,
                             "forward": float(row.forward), "mid": float(row.mid_price),
                             "market_iv": float(row.market_iv),
                             "model_iv": float(ev["spx_model_iv"][i]),
                             "model_iv_se": float(ev["spx_model_iv_se"][i]),
                             "model_iv_fresh": float(ev_fresh["spx_model_iv"][i])})
        vix_rows = []
        for i, row in enumerate(self.vix_opts_surface.itertuples()):
            vix_rows.append({"expiry": str(row.expiry)[:10], "T": float(row.time_to_expiry),
                             "K": float(row.strike), "right": row.right,
                             "forward": float(row.forward), "mid": float(row.mid_price),
                             "market_iv": float(row.market_iv),
                             "model_iv": float(ev["vix_model_iv"][i])})
        fwd_rows = []
        for i, row in enumerate(self.vix_forwards.itertuples()):
            fwd_rows.append({"expiry": row.expiry, "T": float(row.time_to_expiry),
                             "forward_mkt": float(row.forward),
                             "pcp_dispersion": float(row.pcp_dispersion),
                             "forward_model": float(ev["vix_forward_model"][i])})

        result = {
            "schema_version": 2,
            "model": ("two-factor Quintic OU (Abi Jaber & Li 2025, arXiv:2503.14158) with "
                      "p = a0 + a1 z + a3 z^3 + a5 z^5 (Abi Jaber, Illand & Li 2022, "
                      "arXiv:2212.10917), alpha5 = 1"),
            "params":        params,                 # QuinticOUParams (dashboard reads attrs)
            "params_dict":   params.to_dict(),
            "search_vector": dict(zip(SEARCH_NAMES, map(float, best_x))),
            "loss":          float(loss),
            "leg_losses":    leg_losses,
            "leg_definitions": {
                "spx_iv_rmse": "vol points; model vs market Black-76 IV on the put-call-parity "
                               "forward, in-sample CRN paths (the minimised objective)",
                "spx_iv_rmse_fresh_paths": "same, re-priced with %d paths on seed %d"
                                           % (_MC_PATHS_CHECK, self.seed + 1000),
                "vix_futures_rmse": "VIX index points; model E[VIX_T] vs the VIX-option "
                                    "put-call-parity forward (no VX futures in the DB)",
                "vix_futures_rel_rmse": "same as a fraction of the forward (the w2 leg)",
                "vix_options_rmse": "vol points; Black-76 IVs at equal K/F, market on the "
                                    "parity forward, model on its own forward",
                "xi0_index_reproduction_rmse": "VIX index points; input check of the "
                                               "bootstrapped xi0 (parameter-independent)",
            },
            "fit_time":      float(fit_time),
            "fit_cpu_time":  float(fit_cpu),
            "n_evals":       int(de_result.nfev + pol.nfev),
            "success":       bool(de_result.success and pol.success),
            "convergence": {
                "de_success": bool(de_result.success), "de_message": str(de_result.message),
                "de_nit": int(de_result.nit), "de_nfev": int(de_result.nfev),
                "de_loss": float(de_result.fun),
                "polish_success": bool(pol.success), "polish_message": str(pol.message),
                "polish_nfev": int(pol.nfev), "polish_loss": float(pol.fun),
            },
            "at_bound": at_bound,
            "search_bounds": span,
            "admissibility_violations": admissibility_violations(params),
            "nan_model_ivs": {"spx": ev.get("spx_n_nan", 0), "vix": ev.get("vix_n_nan", 0)},
            "spx_rows": spx_rows,
            "vix_rows": vix_rows,
            "vix_forward_rows": fwd_rows,
            "provenance": {
                "as_of_date": self.as_of_date, "spot_date": self.spot_date,
                "spx_snapshot_date": self.spx_snapshot_date,
                "vix_snapshot_date": self.vix_snapshot_date,
                "allow_stale": self.allow_stale,
                "S": self.S, "r": self.r, "r_source": "override" if self.r_override is not None else "^IRX",
                "xi0_t_nodes": self.fwd_var_func.t_nodes.tolist(),
                "xi0_nodes": self.fwd_var_func.xi_nodes.tolist(),
                "vix_index_levels": self.vix_ts.to_dict("records"),
                "n_spx_options": len(self.spx_surface),
                "n_spx_expiries": self._n_expiries(self.spx_surface),
                "n_vix_options": len(self.vix_opts_surface),
                "n_vix_expiries": len(self.vix_forwards),
                "weights": {"w1": self.w1, "w2": self.w2, "w3": self.w3},
                "nan_residual": _NAN_RESIDUAL,
                "mc": {"n_paths": self.n_paths, "seed": self.seed,
                       "steps_per_year": self.steps_per_year, "min_steps": self.min_steps,
                       "estimator": "conditional (mixing) + antithetic pairs + empirical "
                                    "martingale correction of M1 + Richardson (h, h/2)",
                       "check_n_paths": _MC_PATHS_CHECK, "check_seed": self.seed + 1000},
                "vix_quadrature": {"rule": "uniform lattice on the standard-normal plane, |z| <= 8",
                                   "points_per_dim": self.n_outer, "gl_per_piece": self.n_inner},
                "de": {"maxiter": maxiter, "popsize": popsize, "tol": _DE_TOL,
                       "mutation": _DE_MUTATION, "recombination": _DE_RECOMB,
                       "seed": RANDOM_SEED, "init": "latinhypercube",
                       "x0": dict(zip(SEARCH_NAMES, map(float, x0)))},
                "polish_maxiter": polish_maxiter,
                "code": _git_revision(),
                "numpy": np.__version__,
            },
        }
        self.result = result
        self._log_result(result)
        return result

    def _log_result(self, res: dict):
        p, ll = res["params"], res["leg_losses"]
        logger.info(
            "\n── Calibrated Quintic OU Parameters (%s) ──\n  %r\n"
            "── Fit ──\n"
            "  SPX smile RMSE      : %.3f vp (fresh paths %.3f vp)\n"
            "  VIX forward RMSE    : %.3f pts (%.2f%%)\n"
            "  VIX options RMSE    : %.3f vp\n"
            "  ξ0 index reproduction: %.4f pts (input check)\n"
            "  success=%s  evals=%d  fit %.0fs wall / %.0fs CPU (main process)",
            self.spot_date, p, ll["spx_iv_rmse"], ll["spx_iv_rmse_fresh_paths"],
            ll["vix_futures_rmse"], 100 * ll["vix_futures_rel_rmse"], ll["vix_options_rmse"],
            ll["xi0_index_reproduction_rmse"], res["success"], res["n_evals"],
            res["fit_time"], res["fit_cpu_time"],
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Pickle the calibration result to data_store/calibrations/quintic_cal_<date>.pkl.

        Everything except `params` (a QuinticOUParams, kept for the dashboard)
        is plain Python data: params_dict, per-leg RMSEs and their definitions,
        convergence flags, boundary flags, per-option rows (market and model
        IVs), VIX forward rows, and provenance (sessions, S, r, ξ0 nodes, data
        counts, weights, MC/quadrature/DE settings and seeds, git SHA).
        """
        if self.result is None:
            raise RuntimeError("nothing to save: run calibrate() first")
        if path is None:
            cal_dir = DATA_DIR / "calibrations"
            cal_dir.mkdir(parents=True, exist_ok=True)
            path = cal_dir / f"quintic_cal_{self.as_of_date}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.result, f)
        logger.info("Quintic calibration saved: %s", path)
        return Path(path)


if __name__ == "__main__":
    # Spawn-safe entry point so scipy's workers=-1 (macOS uses the 'spawn' start
    # method) can re-import this module cleanly without re-running calibration in
    # every worker. Run with:
    #   python -m joint_vol_calibration.calibration.quintic_calibrator
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cal = QuinticCalibrator(as_of_date="2026-03-24")
    res = cal.calibrate(disp=True)
    cal.save()

    p, ll = res["params"], res["leg_losses"]
    print("=== QUINTIC OU 2026-03-24 ===")
    print(repr(p))
    print("search vector:", {k: round(v, 4) for k, v in res["search_vector"].items()})
    print(f"SPX IV RMSE        = {ll['spx_iv_rmse']:.3f} vp "
          f"(fresh paths {ll['spx_iv_rmse_fresh_paths']:.3f} vp; n={len(res['spx_rows'])})")
    print(f"VIX forward RMSE   = {ll['vix_futures_rmse']:.3f} pts "
          f"({100 * ll['vix_futures_rel_rmse']:.2f}%; n={len(res['vix_forward_rows'])})")
    print(f"VIX options RMSE   = {ll['vix_options_rmse']:.3f} vp (n={len(res['vix_rows'])})")
    print(f"xi0 index check    = {ll['xi0_index_reproduction_rmse']:.4f} pts")
    print(f"success={res['success']}  {res['convergence']}")
    print(f"at_bound={res['at_bound']}")
    print(f"fit_time={res['fit_time']:.0f}s wall, n_evals={res['n_evals']}")
