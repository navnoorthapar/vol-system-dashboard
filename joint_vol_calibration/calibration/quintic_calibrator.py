"""
quintic_calibrator.py — Joint calibration of the two-factor Quintic OU model (C13).

Mirrors the DE + L-BFGS-B structure of joint_calibrator.py but targets the
QuinticOU model instead of Heston. Key differences:

  • 8 parameters instead of 5
  • VIX options leg is analytically tractable (Gaussian integration, not CIR)
  • SPX pricing via MC (batch-per-expiry with antithetic variates)
  • Forward-variance curve ξ_0(t) fixed from ATM SPX term structure before optimisation

Loss function:
    L = w1 · MSE(SPX ivol) + w2 · MSE(VIX futures) + w3 · MSE(VIX options ivol)

VIX options ivol uses the Black-76 formula with F_vix as the forward.

Zero look-ahead: all data loaded as of as_of_date.
"""

import logging
import math
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize

from joint_vol_calibration.config import (
    DATA_DIR, JOINT_W1, JOINT_W2, JOINT_W3, RANDOM_SEED,
)
from joint_vol_calibration.data import database as db
from joint_vol_calibration.models.quintic_ou import (
    QuinticOUParams, QuinticOUModel,
    QUINTIC_BOUNDS, QUINTIC_DEFAULTS,
    eval_poly, ep_squared, var_Z, g0_at,
    price_vix_futures, price_vix_option,
    implied_vol_surface_mc, vix_option_implied_vol,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_CAL_EXPIRY_TARGETS_DAYS = [14, 30, 60, 91, 180, 365]
_CAL_STRIKES_PER_EXPIRY  = 9
_MONEYNESS_LO = 0.70
_MONEYNESS_HI = 1.30
_MIN_IV  = 0.01
_MAX_IV  = 2.00
_MIN_PRICE = 0.50

# Calibration MC settings (speed/accuracy trade-off)
# Common random numbers (fixed MC seed) make the loss surface smooth in the
# parameters, so we can use fewer paths and far fewer DE iterations than with
# per-eval reseeding (which injected noise that DE had to average out).
_MC_PATHS_CAL = 5_000     # paths per expiry during calibration (CRN tolerates fewer)
_VIX_QUAD_OUTER = 10      # GH points per dim (10²=100 pts, was 18²=324 — 3× faster)
_VIX_QUAD_INNER = 16      # GL points for VIX² time integral
_VIX_OPT_STRIKES_CAL = 5  # max VIX option strikes per expiry during calibration

# DE settings — reduced budget thanks to CRN smoothness + multiprocessing
_DE_MAXITER  = 40
_DE_POPSIZE  = 8          # 8 × 8 = 64 members
_DE_TOL      = 1e-5
_DE_MUTATION = (0.5, 1.0)
_DE_RECOMB   = 0.7
_LBFGS_MAXITER = 15       # short polish (finite-diff gradient is costly serially)


class _PiecewiseConstVar:
    """
    Picklable piecewise-constant forward-variance curve ξ₀(t).

    Built by bootstrapping the VIX term structure so that
    100·sqrt((1/T)∫₀ᵀ ξ₀) reproduces the market VIX at every input tenor.

    A plain closure cannot be pickled, which blocks scipy's `workers=-1`
    multiprocessing; this small class is picklable so the calibrator can run
    the DE population in parallel.

    Step convention: ξ₀ on interval (t_nodes[i-1], t_nodes[i]] equals xi_nodes[i];
    flat extrapolation beyond the last node.
    """

    def __init__(self, t_nodes: np.ndarray, xi_nodes: np.ndarray):
        self.t_nodes  = np.asarray(t_nodes, dtype=float)
        self.xi_nodes = np.asarray(xi_nodes, dtype=float)

    def __call__(self, t: float) -> float:
        idx = int(np.searchsorted(self.t_nodes, t, side="left"))
        if idx >= len(self.xi_nodes):
            idx = len(self.xi_nodes) - 1
        return float(self.xi_nodes[idx])

# Quintic OU specific loss weights — independent of global JOINT_W* config
# VIX options are analytically tractable in Quintic OU (unlike Heston CIR density)
# so W3 is enabled here even though JOINT_W3=0
_QUINTIC_W3 = 0.3


def _spot_vix(T: float, fwd_var_func) -> float:
    """
    Spot VIX at horizon T: 100·sqrt((1/T) ∫_0^T ξ₀(s) ds).

    This matches the CBOE ^VIX/^VIX3M convention — both measure integrated
    variance from today to horizon T, not from future time T to T+30d.
    Since E[σ²(s)] = ξ₀(s) by the g₀ normalisation, the formula is model-free.

    Uses a uniform 200-point Riemann sum, which is exact for piecewise-constant
    ξ₀(t) (avoids IntegrationWarning from scipy.quad on jump discontinuities).
    """
    n = 200
    dt = T / n
    pts = np.linspace(dt * 0.5, T - dt * 0.5, n)   # midpoint rule
    xi_vals = np.array([fwd_var_func(s) for s in pts])
    integral = float(np.sum(xi_vals) * dt)
    return 100.0 * math.sqrt(max(integral / T, 0.0))


class QuinticCalibrator:
    """
    Joint SPX/VIX calibrator for the two-factor Quintic OU model.

    Usage
    -----
        cal = QuinticCalibrator(as_of_date='2026-03-24')
        result = cal.calibrate()
    """

    def __init__(
        self,
        as_of_date:  str,
        r:           float = 0.045,
        q:           float = 0.013,
        w1:          float = JOINT_W1,
        w2:          float = JOINT_W2,
        w3:          float = _QUINTIC_W3,
    ):
        self.as_of_date = as_of_date
        self.r  = r
        self.q  = q
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.S: float = 5000.0
        self.spx_surface:       pd.DataFrame = pd.DataFrame()
        self.vix_ts:            pd.DataFrame = pd.DataFrame()
        self.vix_opts_surface:  pd.DataFrame = pd.DataFrame()
        self.fwd_var_func = None
        self.result = None
        self._n_evals = 0

        logger.info("[C13] QuinticCalibrator: loading market data for %s", as_of_date)
        self._load_market_data()

    # ── Market data ─���─────────────────────────────────────────────────────────

    def _load_market_data(self):
        spx_hist = db.get_spx_ohlcv(as_of_date=self.as_of_date)
        if spx_hist.empty:
            raise ValueError(f"No SPX OHLCV data for {self.as_of_date}")
        self.S = float(spx_hist.iloc[-1]["close"])
        logger.info("  SPX spot S = %.2f", self.S)

        self.spx_surface = self._prepare_spx_surface()
        logger.info("  SPX surface: %d options, %d expiries",
                    len(self.spx_surface),
                    self.spx_surface["time_to_expiry"].nunique() if not self.spx_surface.empty else 0)

        self.vix_ts = self._prepare_vix_term_structure()
        logger.info("  VIX TS: %d tenors", len(self.vix_ts))

        self.vix_opts_surface = self._prepare_vix_options()
        logger.info("  VIX options: %d rows", len(self.vix_opts_surface))

        self.fwd_var_func = self._build_fwd_var_func()

    def _prepare_spx_surface(self) -> pd.DataFrame:
        """Filter SPX options and select vega-weighted anchors (same logic as JointCalibrator)."""
        from scipy.stats import norm as _norm

        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="SPX")
        if raw.empty:
            return pd.DataFrame()
        S = self.S
        df = raw.copy()
        df = df[df["time_to_expiry"] >= 7.0 / 365.0]
        df = df[df["time_to_expiry"] <= 2.0]
        df = df[df["strike"].between(S * _MONEYNESS_LO, S * _MONEYNESS_HI)]
        df = df[df["mid_price"] > _MIN_PRICE]
        df = df[df["implied_vol"].between(_MIN_IV, _MAX_IV)].dropna(subset=["implied_vol"])
        call_mask = (df["right"] == "C") & (df["strike"] >= S)
        put_mask  = (df["right"] == "P") & (df["strike"] <  S)
        df = df[call_mask | put_mask].copy().reset_index(drop=True)
        if df.empty:
            return df

        # BS vega
        iv_arr = df["implied_vol"].values.astype(float)
        T_arr  = df["time_to_expiry"].values.astype(float)
        K_arr  = df["strike"].values.astype(float)
        valid  = (iv_arr > 0) & (T_arr > 0)
        d1     = np.where(
            valid,
            (np.log(S / np.where(K_arr > 0, K_arr, 1.0))
             + (self.r - self.q + 0.5 * iv_arr**2) * T_arr)
            / np.where(valid, iv_arr * np.sqrt(T_arr), 1.0),
            0.0,
        )
        df["bs_vega"] = np.where(
            valid,
            np.maximum(S * np.exp(-self.q * T_arr) * _norm.pdf(d1) * np.sqrt(T_arr), 0.5),
            1.0,
        )

        available = sorted(df["time_to_expiry"].unique())
        selected  = []
        for target_days in _CAL_EXPIRY_TARGETS_DAYS:
            target_T = target_days / 365.0
            best_T   = min(available, key=lambda t: abs(t - target_T))
            if abs(best_T - target_T) > target_T * 0.5:
                continue
            sl = df[df["time_to_expiry"] == best_T]
            if len(sl) < 2:
                continue
            selected.append(sl.nlargest(_CAL_STRIKES_PER_EXPIRY, "bs_vega"))

        if not selected:
            return pd.DataFrame()
        cal = pd.concat(selected, ignore_index=True)
        cal["market_price_f"] = cal["mid_price"].astype(float)
        return cal.sort_values(["time_to_expiry", "strike"]).reset_index(drop=True)

    def _prepare_vix_term_structure(self) -> pd.DataFrame:
        vts = db.get_vix_term_structure(as_of_date=self.as_of_date)
        if vts.empty:
            return pd.DataFrame()
        tenors = {
            "^VIX9D": 9.0 / 365.0,
            "^VIX":   30.0 / 365.0,
            "^VIX3M": 91.0 / 365.0,
            "^VIX6M": 182.0 / 365.0,
        }
        rows = []
        for ticker, T in tenors.items():
            row = vts[vts["ticker"] == ticker]
            if not row.empty:
                rows.append({"tenor": T, "vix_level": float(row.iloc[-1]["close"])})
        return pd.DataFrame(rows)

    def _prepare_vix_options(self) -> pd.DataFrame:
        """
        Load VIX options and downsample to _VIX_OPT_STRIKES_CAL near-ATM strikes
        per expiry.  Full 359-row surface is ~3× too expensive per DE evaluation;
        5 strikes captures the ATM vol level and immediate wings adequately.
        """
        raw = db.get_options_surface(as_of_date=self.as_of_date, underlying="VIX")
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df = df[df["time_to_expiry"].between(7.0 / 365.0, 1.0)]
        df = df[df["implied_vol"].between(_MIN_IV, _MAX_IV)].dropna(subset=["implied_vol"])
        if df.empty:
            return df

        # For each expiry keep the _VIX_OPT_STRIKES_CAL strikes closest to ATM
        # (smallest |strike - spot_VIX|, where spot_VIX ≈ 20 is a reasonable proxy)
        spot_vix_proxy = 20.0
        selected = []
        for T_val in df["time_to_expiry"].unique():
            sl = df[df["time_to_expiry"] == T_val].copy()
            sl["atm_dist"] = (sl["strike"] - spot_vix_proxy).abs()
            selected.append(sl.nsmallest(_VIX_OPT_STRIKES_CAL, "atm_dist"))
        return pd.concat(selected, ignore_index=True).drop(columns="atm_dist")

    def _build_fwd_var_func(self) -> "_PiecewiseConstVar":
        """
        Bootstrap the forward-variance curve ξ_0(t) from the VIX term structure.

        The CBOE VIX indices measure annualised integrated variance from today to
        each horizon T:   VIX(T)² = (100²/T) · ∫₀ᵀ ξ_0(s) ds.
        Cumulative integrated variance to each tenor is therefore
            I(T_i) = T_i · (VIX(T_i)/100)²,
        and the piecewise-constant forward variance on (T_{i-1}, T_i] is
            ξ_0 = (I(T_i) − I(T_{i-1})) / (T_i − T_{i-1}).

        With this ξ_0 the model reproduces the VIX term structure *by construction*
        (the spot-VIX leg is no longer something the 8 parameters must fit), so the
        polynomial p(z) and leverage ε are free to fit the SPX and VIX-option smile
        SHAPE — which is exactly the Bourgey–de Marco (2025) calibration design.

        Falls back to the ATM-SPX-variance curve if the VIX term structure is
        unavailable, and to a flat 20% vol if there is no surface at all.
        """
        # ── Primary: bootstrap from VIX term structure ──────────────────────
        if not self.vix_ts.empty:
            ts  = self.vix_ts.sort_values("tenor").reset_index(drop=True)
            T   = ts["tenor"].values.astype(float)
            vix = ts["vix_level"].values.astype(float)
            cum_var = T * (vix / 100.0) ** 2            # I(T_i)

            t_nodes, xi_nodes = [], []
            prev_T, prev_I = 0.0, 0.0
            for Ti, Ii in zip(T, cum_var):
                dT  = Ti - prev_T
                fwd = (Ii - prev_I) / dT if dT > 1e-9 else Ii / max(Ti, 1e-9)
                fwd = max(fwd, 1e-6)                    # clamp steep backwardation
                t_nodes.append(float(Ti))
                xi_nodes.append(float(fwd))
                prev_T, prev_I = Ti, Ii
            return _PiecewiseConstVar(np.array(t_nodes), np.array(xi_nodes))

        # ── Fallback: ATM SPX implied variance ──────────────────────────────
        if self.spx_surface.empty:
            return _PiecewiseConstVar(np.array([1.0]), np.array([0.04]))

        surf = self.spx_surface.copy()
        surf["moneyness"] = surf["strike"] / self.S
        atm_rows = []
        for T_val in surf["time_to_expiry"].unique():
            sl = surf[surf["time_to_expiry"] == T_val]
            closest = sl.loc[(sl["moneyness"] - 1.0).abs().idxmin()]
            atm_rows.append({"T": float(T_val), "atm_iv": float(closest["implied_vol"])})
        atm_df = pd.DataFrame(atm_rows).sort_values("T").reset_index(drop=True)
        return _PiecewiseConstVar(atm_df["T"].values, atm_df["atm_iv"].values ** 2)

    # ── Loss function ─────────────────────────────────────────────────────────

    def _loss(self, x: np.ndarray) -> float:
        """Weighted joint loss for the 8-parameter QuinticOU model."""
        self._n_evals += 1

        # Validate bounds (DE respects bounds; L-BFGS-B may stray slightly)
        for i, (lo, hi) in enumerate(QUINTIC_BOUNDS):
            if not (lo <= x[i] <= hi):
                return 1e8

        params = QuinticOUParams.from_array(x)
        fv     = self.fwd_var_func

        total_loss = 0.0

        # ── SPX leg ──────────────────────────────────────────────────────────
        if self.w1 > 0 and not self.spx_surface.empty:
            expiries = self.spx_surface["time_to_expiry"].values.astype(float)
            strikes  = self.spx_surface["strike"].values.astype(float)
            mkt_ivs  = self.spx_surface["implied_vol"].values.astype(float)

            try:
                # Common random numbers: FIXED seed every evaluation so the loss
                # surface is smooth in the parameters (loss differences reflect
                # parameter changes, not MC noise). This is what lets us cut the
                # DE budget and run a meaningful L-BFGS polish.
                model_ivs = implied_vol_surface_mc(
                    self.S, self.r, self.q, expiries, strikes,
                    params, fv, n_paths=_MC_PATHS_CAL,
                    seed=RANDOM_SEED,
                )
            except Exception:
                return 1e8

            valid = ~np.isnan(model_ivs) & ~np.isnan(mkt_ivs)
            if valid.sum() < 3:
                return 1e8
            spx_mse = float(np.mean((model_ivs[valid] - mkt_ivs[valid])**2))
            total_loss += self.w1 * spx_mse

        # ── VIX futures leg (spot VIX convention) ────────────────────────────
        # Market ^VIX/^VIX3M measure integrated variance from 0 to T (spot VIX),
        # not from T to T+30d (forward VIX). Since E[σ²(s)] = ξ₀(s) by g₀
        # normalisation, spot VIX = 100·sqrt((1/T)·∫_0^T ξ₀(s) ds) — model-free.
        if self.w2 > 0 and not self.vix_ts.empty:
            vix_mse_vals = []
            for _, row in self.vix_ts.iterrows():
                T_vix   = float(row["tenor"])
                mkt_vix = float(row["vix_level"])
                try:
                    model_vix = _spot_vix(T_vix, fv)
                except Exception:
                    continue
                vix_mse_vals.append((model_vix - mkt_vix)**2)
            if vix_mse_vals:
                total_loss += self.w2 * float(np.mean(vix_mse_vals))

        # ── VIX options leg ──────────────────────────────────────────────────
        if self.w3 > 0 and not self.vix_opts_surface.empty:
            vix_opt_mse = []
            # Precompute futures prices for each VIX options expiry
            for T_val in self.vix_opts_surface["time_to_expiry"].unique():
                T_val = float(T_val)
                sl    = self.vix_opts_surface[
                    self.vix_opts_surface["time_to_expiry"] == T_val
                ]
                try:
                    F_vix = price_vix_futures(T_val, params, fv,
                                              n_outer=_VIX_QUAD_OUTER,
                                              n_inner=_VIX_QUAD_INNER)
                except Exception:
                    continue
                for _, row in sl.iterrows():
                    K_vix   = float(row["strike"])
                    mkt_iv  = float(row["implied_vol"])
                    try:
                        call_p  = price_vix_option(T_val, K_vix, params, fv,
                                                   n_outer=_VIX_QUAD_OUTER,
                                                   n_inner=_VIX_QUAD_INNER)
                        model_iv = vix_option_implied_vol(T_val, K_vix, F_vix,
                                                          call_p, self.r)
                    except Exception:
                        continue
                    if model_iv is not None and not np.isnan(model_iv):
                        vix_opt_mse.append((model_iv - mkt_iv)**2)
            if vix_opt_mse:
                total_loss += self.w3 * float(np.mean(vix_opt_mse))

        return total_loss

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(self) -> dict:
        """
        Run joint calibration: global DE search + local L-BFGS-B polish.

        Returns
        -------
        dict with keys:
          params        : QuinticOUParams (calibrated)
          loss          : final total loss
          leg_losses    : {spx_iv_rmse, vix_futures_rmse, vix_options_rmse}
          fit_time      : seconds
          n_evals       : total function evaluations
          success       : bool
        """
        n_spx = len(self.spx_surface)
        n_vts = len(self.vix_ts)
        logger.info(
            "[C13] Quintic calibration: %s  |  S=%.0f  |  %d SPX opts  |  %d VIX tenors",
            self.as_of_date, self.S, n_spx, n_vts,
        )

        t0 = time.perf_counter()

        # ── Global DE ─���──────────────────────────────────────────────────────
        logger.info("     DE: maxiter=%d, popsize=%d × %d = %d members...",
                    _DE_MAXITER, _DE_POPSIZE, len(QUINTIC_BOUNDS),
                    _DE_POPSIZE * len(QUINTIC_BOUNDS))
        np.random.seed(RANDOM_SEED)
        de_result = optimize.differential_evolution(
            self._loss,
            bounds=QUINTIC_BOUNDS,
            maxiter=_DE_MAXITER,
            popsize=_DE_POPSIZE,
            tol=_DE_TOL,
            mutation=_DE_MUTATION,
            recombination=_DE_RECOMB,
            seed=RANDOM_SEED,
            polish=False,
            workers=-1,            # all cores; needs picklable objective (CRN + _PiecewiseConstVar)
            updating="deferred",   # required when workers != 1
        )
        logger.info("     DE done: loss=%.6f, evals=%d", de_result.fun, self._n_evals)

        # ── L-BFGS-B polish ──────────────────────────────────────────────────
        logger.info("     Polishing with L-BFGS-B...")
        pol = optimize.minimize(
            self._loss,
            de_result.x,
            method="L-BFGS-B",
            bounds=QUINTIC_BOUNDS,
            options={"maxiter": _LBFGS_MAXITER, "ftol": 1e-10, "gtol": 1e-6},
        )
        best_x   = pol.x if pol.fun < de_result.fun else de_result.x
        best_val = min(pol.fun, de_result.fun)
        logger.info("     Polish done: loss=%.6f, evals=%d", best_val, self._n_evals)

        fit_time = time.perf_counter() - t0
        params   = QuinticOUParams.from_array(best_x)
        fv       = self.fwd_var_func

        # ── Compute per-leg RMSE ──────────────────────────────────────────────
        spx_rmse = vix_futures_rmse = vix_opts_rmse = 0.0

        if not self.spx_surface.empty:
            expiries = self.spx_surface["time_to_expiry"].values.astype(float)
            strikes  = self.spx_surface["strike"].values.astype(float)
            mkt_ivs  = self.spx_surface["implied_vol"].values.astype(float)
            model_ivs = implied_vol_surface_mc(
                self.S, self.r, self.q, expiries, strikes,
                params, fv, n_paths=_MC_PATHS_CAL * 2,  # 2× for final eval
                seed=RANDOM_SEED,
            )
            valid = ~np.isnan(model_ivs) & ~np.isnan(mkt_ivs)
            if valid.sum() > 0:
                spx_rmse = float(np.sqrt(np.mean((model_ivs[valid] - mkt_ivs[valid])**2))) * 100.0

        if not self.vix_ts.empty:
            sq_errs = []
            for _, row in self.vix_ts.iterrows():
                T_v    = float(row["tenor"])
                mkt_v  = float(row["vix_level"])
                model_v = _spot_vix(T_v, fv)
                sq_errs.append((model_v - mkt_v)**2)
            if sq_errs:
                vix_futures_rmse = float(np.sqrt(np.mean(sq_errs)))

        if self.w3 > 0 and not self.vix_opts_surface.empty:
            sq_errs = []
            for T_val in self.vix_opts_surface["time_to_expiry"].unique():
                T_val = float(T_val)
                sl    = self.vix_opts_surface[
                    self.vix_opts_surface["time_to_expiry"] == T_val
                ]
                F_vix = price_vix_futures(T_val, params, fv)
                for _, row in sl.iterrows():
                    K_vix  = float(row["strike"])
                    mkt_iv = float(row["implied_vol"])
                    call_p = price_vix_option(T_val, K_vix, params, fv)
                    miv    = vix_option_implied_vol(T_val, K_vix, F_vix, call_p, self.r)
                    if miv is not None:
                        sq_errs.append((miv - mkt_iv)**2)
            if sq_errs:
                vix_opts_rmse = float(np.sqrt(np.mean(sq_errs))) * 100.0

        result = {
            "params":     params,
            "loss":       float(best_val),
            "leg_losses": {
                "spx_iv_rmse":       spx_rmse,
                "vix_futures_rmse":  vix_futures_rmse,
                "vix_options_rmse":  vix_opts_rmse,
            },
            "fit_time":   fit_time,
            "n_evals":    int(getattr(de_result, "nfev", 0) + getattr(pol, "nfev", 0)),
            "success":    bool(pol.success or de_result.success),
        }
        self.result = result

        logger.info(
            "\n── Calibrated Quintic OU Parameters (%s) ──\n"
            "  λx=%.4f  λy=%.4f  θ=%.4f  ε=%.4f\n"
            "  α0=%.4f  α1=%.4f  α3=%.4f  α5=%.4f\n"
            "── Loss Breakdown ──\n"
            "  SPX smile RMSE   : %.2f vol pts\n"
            "  VIX futures RMSE : %.4f pts\n"
            "  VIX options RMSE : %.2f vol pts\n"
            "  Fit time         : %.1fs  (%d evals)",
            self.as_of_date,
            params.lam_x, params.lam_y, params.theta, params.epsilon,
            params.alpha0, params.alpha1, params.alpha3, params.alpha5,
            spx_rmse, vix_futures_rmse, vix_opts_rmse,
            fit_time, self._n_evals,
        )

        return result

    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle calibration result to data_store/calibrations/quintic_<date>.pkl."""
        if path is None:
            cal_dir = DATA_DIR / "calibrations"
            cal_dir.mkdir(parents=True, exist_ok=True)
            path = cal_dir / f"quintic_cal_{self.as_of_date}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.result, f)
        logger.info("Quintic calibration saved: %s", path)
        return path


if __name__ == "__main__":
    # Spawn-safe entry point so scipy's workers=-1 (macOS uses the 'spawn' start
    # method) can re-import this module cleanly without re-running calibration in
    # every worker. Run with:
    #   python -m joint_vol_calibration.calibration.quintic_calibrator
    import warnings

    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    cal = QuinticCalibrator(as_of_date="2026-03-24")
    res = cal.calibrate()
    cal.save()

    p  = res["params"]
    ll = res["leg_losses"]
    from joint_vol_calibration.models.quintic_ou import QUINTIC_BOUNDS

    print("=== QUINTIC OU (bootstrapped xi0 + CRN + parallel) 2026-03-24 ===")
    print(f"lam_x={p.lam_x:.4f}  lam_y={p.lam_y:.4f}  theta={p.theta:.4f}  epsilon={p.epsilon:.4f}")
    print(f"alpha0={p.alpha0:.4f}  alpha1={p.alpha1:.4f}  alpha3={p.alpha3:.4f}  alpha5={p.alpha5:.4f}")
    print(f"SPX RMSE         = {ll['spx_iv_rmse']:.4f} vp   (target < 2.0)")
    print(f"VIX Futures RMSE = {ll['vix_futures_rmse']:.4f} pts (target < 1.5)")
    print(f"VIX Options RMSE = {ll['vix_options_rmse']:.4f} vp  (target < 10; Heston=37.14)")
    print(f"fit_time={res['fit_time']:.1f}s  n_evals={res['n_evals']}")
    print(f"lam_x at upper bound ({QUINTIC_BOUNDS[0][1]}): {abs(p.lam_x - QUINTIC_BOUNDS[0][1]) < 20}")
    print(f"epsilon saturated ({QUINTIC_BOUNDS[7][0]}): {abs(p.epsilon - QUINTIC_BOUNDS[7][0]) < 0.02}")
