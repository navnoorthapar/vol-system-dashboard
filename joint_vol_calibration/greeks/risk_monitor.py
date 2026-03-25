"""
risk_monitor.py — Second-Order Greeks Risk Monitor (C6).

Computes the full second-order Greeks surface across strike / maturity from
Heston-calibrated implied vols, plus quadratic variation convexity from a
Monte Carlo simulation of the CIR variance process.

The 'sigma' that is bumped is the Black-Scholes implied volatility at each
(K, T) node (backed out from the Heston price). All Greek formulas are
therefore standard BS market-Greeks evaluated at the model-implied vol.

Greeks computed
---------------
  Vomma  : ∂Vega/∂σ_IV = Vega × d1 × d2 / σ_IV  (analytical)
            Numerical forward-difference cross-check: (Vega(σ+δ) - Vega(σ)) / δ

  Volga  : ∂²V/∂σ_IV²   central-difference on the BS call price
            (V(σ+δ) − 2V(σ) + V(σ−δ)) / δ²
            For BS this equals Vomma analytically; used as independent check.

  Vanna  : ∂²V/∂S∂σ_IV = −exp(−qT)·N′(d1)·d2 / σ_IV  (analytical)
            = ∂Vega/∂S = ∂Delta/∂σ_IV

  QV Convexity : Var[∫₀ᵀ v_t dt]  from MC simulation of the Heston CIR process.
            Measures uncertainty in realised variance; enters variance-swap
            and vol-of-vol pricing.

Sign convention
---------------
  Vomma  > 0  for deep OTM and deep ITM calls  (d1·d2 > 0)
  Vomma  ≈ 0  near ATM  (d1 ≈ 0, d2 ≈ 0)
  Vomma  < 0  for near-ATM calls with small σ√T  (d1 > 0, d2 < 0)

  Vanna  > 0  for OTM calls  (K > F → d2 < 0 → −d2 > 0)
  Vanna  < 0  for deep ITM calls  (d2 > 0)

Unstable hedge points
---------------------
  Flag: |Vomma(K,T) − μ_surface| > 2σ_surface
  At these nodes a 1 vol-pt move in implied vol changes Vega by more than
  2σ relative to the average node. Standard delta-hedging is insufficient —
  positions require vomma hedges or more frequent rebalancing.

Outputs for C7
--------------
  data_store/greeks/greeks_surface.parquet   — full (K, T) DataFrame
  data_store/greeks/vomma_heatmap.png        — heatmap with ×-marks

Usage
-----
  from joint_vol_calibration.greeks.risk_monitor import RiskMonitor

  monitor = RiskMonitor(
      S=6581.0, r=0.045, q=0.013,
      params={"kappa": 4.62, "theta": 0.0764, "sigma": 0.84, "rho": -0.99, "v0": 0.0561},
  )
  surface = monitor.build()
  flagged = monitor.flag()
  fig     = monitor.plot(save_path="vomma.png")
  path    = monitor.save()
  checks  = monitor.validate()
"""

import logging
import time
from datetime import date
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import norm

from joint_vol_calibration.config import DATA_DIR, RANDOM_SEED
from joint_vol_calibration.models.heston import heston_call_batch
from joint_vol_calibration.models.nn_pricer import _bs_iv_vectorized

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

GREEKS_DIR          = DATA_DIR / "greeks"
GREEKS_SURFACE_PATH = GREEKS_DIR / "greeks_surface.parquet"
HEATMAP_PATH        = GREEKS_DIR / "vomma_heatmap.png"

# ── Default grid ───────────────────────────────────────────────────────────────

DEFAULT_MATURITIES_DAYS = np.array([14, 30, 60, 91, 180, 365])
DEFAULT_LOG_MONEYNESS   = np.linspace(-0.30, 0.25, 15)   # 15 strikes per expiry

# Finite-difference bump for Volga central difference (10 bps in vol space)
_VOLGA_BUMP = 0.001


# ── Analytical BS Greeks (vectorised) ─────────────────────────────────────────

def _bs_d1d2(
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
) -> tuple:
    """d1, d2 in the forward-measure Black-Scholes formula."""
    sqrtT = np.sqrt(np.maximum(T, 1e-12))
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def _bs_call_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorised Black-Scholes call price."""
    F = S * np.exp((r - q) * T)
    d1, d2 = _bs_d1d2(F, K, T, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_delta(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """∂C/∂S = exp(−qT)·N(d1)  (always in [0, exp(-qT)])."""
    F = S * np.exp((r - q) * T)
    d1, _ = _bs_d1d2(F, K, T, sigma)
    return np.exp(-q * T) * norm.cdf(d1)


def _bs_vega(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """∂C/∂σ = S·exp(−qT)·N′(d1)·√T  (always positive)."""
    F = S * np.exp((r - q) * T)
    d1, _ = _bs_d1d2(F, K, T, sigma)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(np.maximum(T, 1e-12))


def _bs_vomma(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Analytical Vomma = ∂Vega/∂σ = ∂²C/∂σ² = Vega · d1 · d2 / σ.

    Sign:
      d1·d2 > 0  (same sign) for deep OTM (K >> F) or deep ITM (K << F)
      d1·d2 < 0  (opposite signs) near ATM — option is locally concave in vol
    """
    F = S * np.exp((r - q) * T)
    d1, d2 = _bs_d1d2(F, K, T, sigma)
    vega = _bs_vega(S, K, T, r, q, sigma)
    return vega * d1 * d2 / np.maximum(sigma, 1e-8)


def _bs_vanna(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    ∂²C/∂S∂σ = ∂Vega/∂S = ∂Delta/∂σ = −exp(−qT)·N′(d1)·d2 / σ.

    For OTM calls (K > F): d2 < 0 → Vanna > 0 (delta rises as vol rises).
    For ITM calls (K < F, d2 > 0): Vanna < 0 (delta falls as vol rises).
    """
    F = S * np.exp((r - q) * T)
    d1, d2 = _bs_d1d2(F, K, T, sigma)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / np.maximum(sigma, 1e-8)


def _bs_volga_numerical(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    bump: float = _VOLGA_BUMP,
) -> np.ndarray:
    """
    Numerical Volga via central finite difference on the BS price.

    (V(σ+δ) − 2V(σ) + V(σ−δ)) / δ²

    Cross-check: for BS this should agree with analytical Vomma to O(δ²).
    Bump δ = 0.001 (10 bps) gives errors < 0.1% for typical option parameters.
    """
    sig_up = np.clip(sigma + bump, 1e-4, 5.0)
    sig_dn = np.clip(sigma - bump, 1e-4, 5.0)
    V_up = _bs_call_price(S, K, T, r, q, sig_up)
    V_0  = _bs_call_price(S, K, T, r, q, sigma)
    V_dn = _bs_call_price(S, K, T, r, q, sig_dn)
    return (V_up - 2.0 * V_0 + V_dn) / bump**2


# ── Quadratic Variation Convexity (Monte Carlo CIR) ──────────────────────────

def simulate_qv_convexity(
    kappa: float,
    theta: float,
    sigma: float,
    v0: float,
    T: float,
    n_paths: int = 10_000,
    n_steps: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute quadratic variation convexity Var[∫₀ᵀ v_t dt] via Monte Carlo.

    Simulates the Heston CIR variance process using the full-truncation
    Euler-Maruyama scheme (Lord et al. 2010):

        v_{t+Δt} = max(v_t + κ(θ − v_t)Δt + σ√max(v_t,0)·ΔW_t,  0)

    Quadratic variation per path:
        QV_i = Σₜ v_t · Δt   (Riemann approximation of ∫₀ᵀ v_t dt)

    Returns Var[QV] = E[QV²] − E[QV]²  (sample variance, ddof=0).

    Properties
    ----------
      ≥ 0 always (variance is non-negative).
      = 0 when σ = 0 (deterministic variance path).
      Increases with σ (more vol-of-vol → more spread in realised variance).
      Increases with T (more time → more uncertainty accumulates).

    Parameters
    ----------
    n_paths : MC paths (10K default; use 1K in tests for speed)
    n_steps : time steps per path (100 is sufficient for CIR convergence)
    rng     : numpy Generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)

    v            = np.full(n_paths, v0, dtype=np.float64)
    integrated_v = np.zeros(n_paths, dtype=np.float64)

    for _ in range(n_steps):
        dW = rng.standard_normal(n_paths) * sqrt_dt
        v  = v + kappa * (theta - v) * dt + sigma * np.sqrt(np.maximum(v, 0.0)) * dW
        v  = np.maximum(v, 0.0)          # full truncation
        integrated_v += v * dt

    return float(np.var(integrated_v, ddof=0))   # = E[QV²] − E[QV]²


# ── Core Surface Computation ───────────────────────────────────────────────────

def compute_greeks_surface(
    S: float,
    r: float,
    q: float,
    params: dict,
    strikes: Optional[np.ndarray] = None,
    maturities: Optional[np.ndarray] = None,
    as_of_date: Optional[str] = None,
    mc_n_paths: int = 10_000,
    mc_n_steps: int = 100,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute the full second-order Greeks surface for a Heston calibration.

    For each (K, T) node:
      1. Price the Heston call → invert to BS implied vol σ_IV.
      2. Compute analytical BS Greeks (delta, vega, vomma, vanna).
      3. Compute numerical Volga as independent cross-check.
      4. Attach QV convexity (computed once per maturity via MC).

    Parameters
    ----------
    S, r, q    : spot price, risk-free rate, dividend yield
    params     : dict with keys {kappa, theta, sigma, rho, v0}
    strikes    : absolute strike values; defaults to 15 log-spaced per maturity
    maturities : expiry grid in years; defaults to [14,30,60,91,180,365] days
    as_of_date : recorded in the output DataFrame (ISO 'YYYY-MM-DD')
    mc_n_paths : MC paths for QV convexity (use 1_000 in tests)
    mc_n_steps : MC steps per path

    Returns
    -------
    DataFrame columns:
      [as_of_date,] K, T, T_days, log_moneyness,
      iv, delta, vega, vomma, volga, vanna, qv_convexity
    """
    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    rho   = params["rho"]
    v0    = params["v0"]

    if maturities is None:
        maturities = DEFAULT_MATURITIES_DAYS / 365.0

    rng = np.random.default_rng(RANDOM_SEED)
    t0  = time.time()
    rows: List[dict] = []

    for T in maturities:
        F = S * np.exp((r - q) * T)

        # Build strike grid relative to this maturity's forward
        K_arr = (F * np.exp(DEFAULT_LOG_MONEYNESS)
                 if strikes is None
                 else np.asarray(strikes, dtype=float))
        n_K = len(K_arr)

        # ── Heston call prices → IV (vectorised) ──
        calls = heston_call_batch(S, K_arr, T, r, q, kappa, theta, sigma, rho, v0)
        ivs   = _bs_iv_vectorized(
            calls,
            np.full(n_K, F),
            K_arr,
            np.full(n_K, T),
            np.full(n_K, r),
        )

        valid = ~np.isnan(ivs) & (ivs >= 0.005)
        if not valid.any():
            logger.warning("No valid IVs at T=%.3f — skipping", T)
            continue

        K_v  = K_arr[valid]
        iv_v = ivs[valid]
        n_v  = int(valid.sum())

        # ── Broadcast scalar market variables ──
        Sv = np.full(n_v, S)
        Tv = np.full(n_v, T)
        rv = np.full(n_v, r)
        qv = np.full(n_v, q)

        # ── All Greeks in one vectorised pass ──
        delta_v = _bs_delta(Sv, K_v, Tv, rv, qv, iv_v)
        vega_v  = _bs_vega(Sv, K_v, Tv, rv, qv, iv_v)
        vomma_v = _bs_vomma(Sv, K_v, Tv, rv, qv, iv_v)
        vanna_v = _bs_vanna(Sv, K_v, Tv, rv, qv, iv_v)
        volga_v = _bs_volga_numerical(Sv, K_v, Tv, rv, qv, iv_v)

        # ── QV convexity: one MC run per maturity ──
        qv_conv = simulate_qv_convexity(kappa, theta, sigma, v0, T,
                                        mc_n_paths, mc_n_steps, rng)

        # ── Assemble rows ──
        for j in range(n_v):
            row: dict = {
                "K":             float(K_v[j]),
                "T":             float(T),
                "T_days":        int(round(T * 365)),
                "log_moneyness": float(np.log(K_v[j] / F)),
                "iv":            float(iv_v[j]),
                "delta":         float(delta_v[j]),
                "vega":          float(vega_v[j]),
                "vomma":         float(vomma_v[j]),
                "volga":         float(volga_v[j]),
                "vanna":         float(vanna_v[j]),
                "qv_convexity":  float(qv_conv),
            }
            if as_of_date:
                row["as_of_date"] = as_of_date
            rows.append(row)

        if verbose:
            logger.info(
                "  T=%4dd: %d strikes, IV=[%.1f%%..%.1f%%], QV_conv=%.3e",
                int(round(T * 365)), n_v,
                iv_v.min() * 100, iv_v.max() * 100, qv_conv,
            )

    df = pd.DataFrame(rows)
    if verbose:
        logger.info("Greeks surface: %d cells in %.1fs", len(df), time.time() - t0)
    return df


# ── Unstable Hedge Flagging ────────────────────────────────────────────────────

def flag_unstable_hedges(
    df: pd.DataFrame,
    threshold_sigma: float = 2.0,
    greek: str = "vomma",
) -> pd.DataFrame:
    """
    Mark (K, T) nodes where |greek| is more than threshold_sigma × std
    away from the cross-surface mean.

    Interpretation
    --------------
    Flagged nodes are 'unstable hedge points': a standard 1 vol-pt move in
    implied vol shifts the vega of the position by an unusually large amount.
    Delta-hedging alone does not neutralise this exposure. The desk must either:
      (a) hold offsetting vomma (e.g. via butterfly spreads), or
      (b) re-hedge more frequently near these nodes.

    Columns added
    -------------
    vomma_zscore : z-score of vomma relative to surface mean/std.
    is_unstable  : bool, True where |z| > threshold_sigma.
    """
    out    = df.copy()
    vals   = out[greek].values
    mu     = float(np.mean(vals))
    std    = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    z      = (vals - mu) / max(std, 1e-12)
    out["vomma_zscore"] = z
    out["is_unstable"]  = np.abs(z) > threshold_sigma
    n_flagged = int(out["is_unstable"].sum())
    logger.info(
        "Flagged %d / %d unstable hedge points (|z| > %.1f σ)",
        n_flagged, len(out), threshold_sigma,
    )
    return out


# ── Vomma Heatmap ─────────────────────────────────────────────────────────────

def plot_vomma_heatmap(
    df: pd.DataFrame,
    greek: str = "vomma",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    2-D heatmap: log-moneyness on x-axis, maturity on y-axis, {greek} as colour.

    Uses a diverging RdBu_r colour map centred on zero so positive and
    negative regions are immediately visible. Unstable hedge points
    (where 'is_unstable' == True) are marked with a black × overlay.

    Parameters
    ----------
    df        : output of flag_unstable_hedges or compute_greeks_surface
    greek     : column to plot as colour (default 'vomma')
    save_path : if provided, save the figure to this path as PNG
    """
    df_p = df.copy()
    df_p["log_m_r"] = df_p["log_moneyness"].round(3)

    pivot = df_p.pivot_table(
        index="T_days", columns="log_m_r", values=greek, aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=figsize)

    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = max(np.abs(vals).max(), 1e-6) if len(vals) > 0 else 1.0
    norm_ = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    im = ax.imshow(
        pivot.values, aspect="auto",
        cmap=plt.get_cmap("RdBu_r"), norm=norm_,
        origin="lower",
    )

    x_labels = [f"{v:.2f}" for v in pivot.columns]
    y_labels  = [f"{v}d"   for v in pivot.index]
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Log-Moneyness  ln(K/F)", fontsize=10)
    ax.set_ylabel("Maturity", fontsize=10)
    ax.set_title(
        f"Heston Surface — {greek.capitalize()}\n"
        f"(red = positive  |  blue = negative  |  × = unstable hedge)",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label=greek)

    # Overlay unstable points
    if "is_unstable" in df_p.columns:
        col_list = list(pivot.columns)
        row_list = list(pivot.index)
        for _, row in df_p[df_p["is_unstable"]].iterrows():
            try:
                xi = col_list.index(row["log_m_r"])
                yi = row_list.index(row["T_days"])
                ax.plot(xi, yi, "kx", markersize=10, markeredgewidth=2)
            except (ValueError, KeyError):
                pass

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Heatmap saved to %s", save_path)

    return fig


# ── Persistence ───────────────────────────────────────────────────────────────

def save_greeks_surface(
    df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Persist the Greeks surface to parquet for C7 (backtester) consumption.

    C7 reads via:  pd.read_parquet(GREEKS_SURFACE_PATH)
    The file is overwritten on each call so it always reflects the latest
    calibration.
    """
    out_path = Path(path) if path else GREEKS_SURFACE_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Greeks surface saved: %s  (%d rows)", out_path, len(df))
    return out_path


# ── RiskMonitor Class ─────────────────────────────────────────────────────────

class RiskMonitor:
    """
    Orchestrates the full second-order Greeks risk analysis.

    Typical workflow
    ----------------
      monitor = RiskMonitor(S, r, q, params)
      surface = monitor.build()            # compute Greeks surface
      flagged = monitor.flag()             # identify unstable hedges
      fig     = monitor.plot()             # generate heatmap (saved to disk)
      path    = monitor.save()             # persist to parquet for C7
      checks  = monitor.validate()         # automated sanity checks

    Attributes
    ----------
    surface : DataFrame from compute_greeks_surface (set by build())
    flagged : DataFrame from flag_unstable_hedges (set by flag())
    """

    def __init__(
        self,
        S: float,
        r: float,
        q: float,
        params: dict,
        as_of_date: Optional[str] = None,
    ):
        self.S          = S
        self.r          = r
        self.q          = q
        self.params     = params
        self.as_of_date = as_of_date or date.today().isoformat()
        self.surface: Optional[pd.DataFrame] = None
        self.flagged: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        strikes: Optional[np.ndarray] = None,
        maturities: Optional[np.ndarray] = None,
        mc_n_paths: int = 10_000,
        mc_n_steps: int = 100,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compute full Greeks surface. Must be called before flag/plot/save."""
        logger.info("Building Greeks surface (S=%.1f, date=%s)...",
                    self.S, self.as_of_date)
        self.surface = compute_greeks_surface(
            S=self.S, r=self.r, q=self.q,
            params=self.params,
            strikes=strikes, maturities=maturities,
            as_of_date=self.as_of_date,
            mc_n_paths=mc_n_paths, mc_n_steps=mc_n_steps,
            verbose=verbose,
        )
        return self.surface

    def flag(self, threshold_sigma: float = 2.0) -> pd.DataFrame:
        """Identify unstable hedge nodes (|vomma_z| > threshold_sigma)."""
        if self.surface is None:
            raise RuntimeError("Call build() before flag()")
        self.flagged = flag_unstable_hedges(self.surface, threshold_sigma)
        return self.flagged

    def plot(
        self,
        greek: str = "vomma",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Generate heatmap. Uses flagged surface when available."""
        df = self.flagged if self.flagged is not None else self.surface
        if df is None:
            raise RuntimeError("Call build() before plot()")
        return plot_vomma_heatmap(df, greek=greek,
                                   save_path=save_path or HEATMAP_PATH)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist surface to parquet for C7."""
        df = self.flagged if self.flagged is not None else self.surface
        if df is None:
            raise RuntimeError("Call build() before save()")
        return save_greeks_surface(df, path)

    def validate(self) -> dict:
        """
        Automated sanity checks on the computed surface.

        Checks
        ------
        vega_positive          : BS vega > 0 everywhere (guaranteed by N′(d1) > 0)
        vomma_positive_far_otm : Vomma > 0 for clearly OTM calls (|x| > 0.15)
                                 These have d1, d2 of the same sign → d1·d2 > 0.
        iv_in_range            : All σ_IV ∈ (0.01, 3.0)
        qv_nonnegative         : Var[QV] ≥ 0 (variance always non-negative)
        volga_close_to_vomma   : |Volga_num − Vomma_analytical| < 5%
                                 (sanity check on numerical implementation)
        """
        if self.surface is None:
            raise RuntimeError("Call build() before validate()")
        df = self.surface
        checks: dict = {}

        # 1. Vega always positive
        checks["vega_positive"] = bool((df["vega"] > 0).all())

        # 2. Vomma positive for far OTM / ITM (|log_m| > 0.15)
        far = df[np.abs(df["log_moneyness"]) > 0.15]
        checks["vomma_positive_far_otm"] = bool(
            len(far) == 0 or (far["vomma"] > 0).all()
        )

        # 3. IV in realistic range
        checks["iv_in_range"] = bool(
            (df["iv"] > 0.01).all() and (df["iv"] < 3.0).all()
        )

        # 4. QV convexity non-negative
        checks["qv_nonnegative"] = bool((df["qv_convexity"] >= 0).all())

        # 5. Numerical Volga close to analytical Vomma (within 5% + small abs tol)
        rel_err = np.abs(df["volga"] - df["vomma"]) / (np.abs(df["vomma"]) + 1e-4)
        checks["volga_close_to_vomma"] = bool((rel_err < 0.05).all())

        all_ok = all(checks.values())
        logger.info("Validation: %s — %s",
                    "ALL PASS" if all_ok else "FAILURES DETECTED", checks)
        return checks

    def __repr__(self) -> str:
        rows = len(self.surface) if self.surface is not None else 0
        return (f"RiskMonitor(S={self.S}, date={self.as_of_date}, "
                f"surface_rows={rows})")
