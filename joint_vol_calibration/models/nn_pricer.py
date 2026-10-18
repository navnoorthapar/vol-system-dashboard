"""
nn_pricer.py — Neural-network surrogate for Heston implied vols (C5).

Status: research prototype. No production code path imports this module —
the calibrator prices with heston_call_batch and the Greeks monitor with the
Fourier pricer in models.implied_vol. The surrogate is evaluated here so that
its accuracy and speed claims are reproducible, not because anything relies
on it.

  NN-1 (SPX): (x = ln(K/F), √T, κ, θ, σ, ρ, v0) → Black-Scholes implied vol
  NN-2 (VIX): (κ, θ, σ, v0, tenor) → heston_vix_futures_curve value.
              DEPRECATED: the closed form it approximates is itself cheaper
              than the network (NNPricer.benchmark_vix: 12 µs for
              heston_vix_futures_curve vs 87 µs for the NN-2 forward pass,
              8 tenors), and exact, so NN-2 accelerates nothing; it is kept
              only so old checkpoints and results can be reproduced.

Training domain (stored in every checkpoint; inference refuses inputs outside it)
---------------------------------------------------------------------------------
  The old surrogate was trained on a ±30% box around one day's fit, which
  none of the archived calibrations fall inside (v0 alone ranges 1e-4..0.056
  across them). SPX_TRAINING_DOMAIN is a box that contains every archived
  joint_cal_*.pkl fit (κ 1.7–12.9, θ 0.022–0.076, σ 0.24–0.85, ρ −0.99..−0.73,
  v0 1e-4..0.056) with margin:

      κ ∈ [0.5, 15]  (log-uniform)     θ ∈ [0.01, 0.20] (log-uniform)
      σ ∈ [0.1, 1.2]                   ρ ∈ [−0.99, −0.30]
      v0 ∈ [1e-4, 0.25] (log-uniform)  T ∈ [0.02, 2] (√T uniform)
      x ∈ [−0.30, 0.25] and z = x/(σ_ref√T) ∈ [−6, 4],
      σ_ref² = θ + (v0 − θ)(1 − e^{−κT})/(κT)  (expected average variance)

  The z constraint keeps strikes within a sensible number of standard
  deviations when total variance is small (otherwise most of [−0.30, 0.25]
  is options worth nothing at 7-day, 1%-vol corners).

Labels
------
  Exact Heston prices (Lewis/Fourier, ~1e-14·F; implied_vol.fourier_otm_price)
  inverted on the OTM side by the robust solver. Options whose OTM value is
  below NN_LABEL_MIN_OTM_PRICE_REL·F (1e-6) are not labelled: their implied
  vol is not meaningful. The old generator used heston_call_batch plus a
  Newton loop that failed on 28% of OTM calls and silently dropped them,
  biasing the labels away from the OTM-call wing.

Splits and reporting
--------------------
  Rows are grouped by parameter set (set_id) and split 80/10/10 into
  train/validation/test by group, so no parameter set is on both sides. The
  validation set drives early stopping and the LR schedule; the TEST set is
  what accuracy is reported on. Metrics are reported in each network's own
  units (vol points for NN-1, VIX points for NN-2).

Benchmark
---------
  NNPricer.benchmark() draws fresh parameter sets from the domain (seeded,
  disjoint from training draws), prices a full calibration-like workload per
  set (several maturities, strikes across the whole x/z domain including the
  OTM-call wing), scores the NN against the exact reference on every
  well-posed option (never filtering on the NN's own output), and times the
  NN forward pass against heston_call_batch + IV inversion on the same
  options. Numbers depend on the machine; the result records it.

Usage
-----
  from joint_vol_calibration.models.nn_pricer import NNPricer, train_spx

  result = train_spx()                       # generate data (cached) + train
  pricer = NNPricer.load()                   # load from disk (no auto-train)
  ivs    = pricer.spx_iv_batch(features)     # (N, 7) → (N,), raises out of domain
  report = pricer.benchmark()
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from joint_vol_calibration.config import (
    DATA_DIR,
    HESTON_BOUNDS,
    MC_CACHE_DIR,
    NN_BATCH_SIZE,
    NN_HIDDEN_DIM,
    NN_LEARNING_RATE,
    NN_MAX_EPOCHS,
    NN_N_LAYERS,
    NN_TRAIN_SAMPLES,
    NN_VOL_ERROR_TARGET,
    RANDOM_SEED,
    ROOT_DIR,
)
from joint_vol_calibration.models.heston import (
    characteristic_function,
    heston_call_batch,
    heston_vix_futures_curve,
)
from joint_vol_calibration.models.implied_vol import (  # noqa: F401  (_bs_iv_vectorized re-exported)
    _bs_iv_vectorized,
    fourier_otm_price,
    implied_vol_black,
)

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASELINE_PATH   = DATA_DIR / "calibrations" / "baseline_2026-03-24.json"
_SPX_DATA_PATH   = MC_CACHE_DIR / "nn_spx_train_v2.parquet"
_VIX_DATA_PATH   = MC_CACHE_DIR / "nn_vix_train_v2.parquet"
_SPX_MODEL_PATH  = DATA_DIR / "nn_spx_model.pt"
_VIX_MODEL_PATH  = DATA_DIR / "nn_vix_model.pt"

# ── Constants ──────────────────────────────────────────────────────────────────

_VIX_WINDOW = 30.0 / 365.0
_DEFAULT_CAL_PARAMS = {          # fallback if baseline file not found
    "kappa": 4.62, "theta": 0.0764, "sigma": 0.8407, "rho": -0.95, "v0": 0.0561,
}

SPX_FEATURES = ["x", "sqrt_T", "kappa", "theta", "sigma", "rho", "v0"]
VIX_FEATURES = ["kappa", "theta", "sigma", "v0", "tenor_years"]
_PARAM_NAMES = ("kappa", "theta", "sigma", "rho", "v0")

SPX_TRAINING_DOMAIN = {
    "kappa": [0.5, 15.0],
    "theta": [0.01, 0.20],
    "sigma": [0.10, 1.20],
    "rho":   [-0.99, -0.30],
    "v0":    [1e-4, 0.25],
    "T":     [0.02, 2.0],
    "x":     [-0.30, 0.25],
    "z":     [-6.0, 4.0],
}
VIX_TRAINING_DOMAIN = {
    "kappa": [0.5, 15.0],
    "theta": [0.01, 0.20],
    "sigma": [0.10, 1.20],
    "v0":    [1e-4, 0.25],
    "tenor_years": [0.02, 1.5],
}
_LOG_UNIFORM = ("kappa", "theta", "v0")

NN_LABEL_MIN_OTM_PRICE_REL = 1e-6   # label / benchmark floor on OTM value / F
NN_VIX_TARGET_MAE = 0.05            # VIX points (NN-2's own units)

_SPX_CKPT_FORMAT = "spx_iv_net_v2"
_VIX_CKPT_FORMAT = "vix_futures_net_v2"


# ── Domain helpers ────────────────────────────────────────────────────────────

def _sigma_ref(T, kappa, theta, v0):
    """√ of the expected average variance to T (leading-order ATM implied vol)."""
    T = np.asarray(T, dtype=float)
    kT = np.asarray(kappa, dtype=float) * T
    B = np.where(kT > 1e-10, -np.expm1(-kT) / np.where(kT > 1e-10, kT, 1.0), 1.0)
    return np.sqrt(np.maximum(theta + (v0 - theta) * B, 1e-12))


def _sample_domain_params(n: int, domain: dict, rng: np.random.Generator) -> dict:
    """Draw n Heston parameter sets from a domain box (log-uniform for κ, θ, v0)."""
    out = {}
    for name in _PARAM_NAMES:
        if name not in domain:
            continue
        lo, hi = domain[name]
        if name in _LOG_UNIFORM:
            out[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        else:
            out[name] = rng.uniform(lo, hi, n)
    return out


def _x_interval(T: float, p: dict, domain: dict) -> Tuple[float, float]:
    """Admissible ln(K/F) interval at T: domain x-range ∩ z-range·σ_ref√T."""
    w = float(_sigma_ref(T, p["kappa"], p["theta"], p["v0"])) * np.sqrt(T)
    lo = max(domain["x"][0], domain["z"][0] * w)
    hi = min(domain["x"][1], domain["z"][1] * w)
    return lo, hi


def spx_out_of_domain(features: np.ndarray, domain: dict = SPX_TRAINING_DOMAIN) -> np.ndarray:
    """
    Boolean mask of rows of `features` (N, 7: SPX_FEATURES order) that lie
    outside the training domain (parameter box, T range, x range, z range).
    """
    f = np.asarray(features, dtype=np.float64)
    x, sqT, kappa, theta, sigma, rho, v0 = (f[:, i] for i in range(7))
    T = sqT**2
    eps = 1e-9
    bad = ~np.all(np.isfinite(f), axis=1)
    for name, col in zip(_PARAM_NAMES, (kappa, theta, sigma, rho, v0)):
        lo, hi = domain[name]
        bad |= (col < lo - eps * abs(lo)) | (col > hi + eps * abs(hi))
    bad |= (T < domain["T"][0] * (1 - eps)) | (T > domain["T"][1] * (1 + eps))
    bad |= (x < domain["x"][0] - eps) | (x > domain["x"][1] + eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = x / (_sigma_ref(T, kappa, theta, v0) * np.sqrt(np.maximum(T, 1e-12)))
    bad |= (z < domain["z"][0] - 1e-6) | (z > domain["z"][1] + 1e-6)
    return bad


# ── Exact labels ──────────────────────────────────────────────────────────────

def _heston_otm_rel(x: np.ndarray, T: float, p: dict) -> Tuple[np.ndarray, bool]:
    """Exact Heston OTM forward value / F for one (T, params) slice (Fourier)."""
    w = float(_sigma_ref(T, p["kappa"], p["theta"], p["v0"])) ** 2 * T

    def cf(u):
        return characteristic_function(u, 1.0, T, 0.0, 0.0, **p)

    return fourier_otm_price(cf, x, variance_hint=w)


def _iv_from_otm_rel(otm, x, T, min_otm_price_rel):
    """Implied vols of OTM forward values / F (vectorised over slices)."""
    return np.asarray(implied_vol_black(otm, 1.0, np.exp(x), T, is_call=(x >= 0),
                                        discount=1.0, sigma_min=1e-4, sigma_max=10.0,
                                        min_otm_price=min_otm_price_rel), dtype=float)


def heston_iv_exact(
    x: np.ndarray,
    T: float,
    params: dict,
    min_otm_price_rel: float = NN_LABEL_MIN_OTM_PRICE_REL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Heston implied vols for one (T, params) slice.

    Returns (iv, otm_rel): implied vols (NaN where the OTM value is below
    min_otm_price_rel·F or the Fourier pricer did not converge) and the OTM
    forward value / F. Independent of S, r and q (Heston is homogeneous).
    """
    p = {k: float(params[k]) for k in _PARAM_NAMES}
    x = np.atleast_1d(np.asarray(x, dtype=float))
    otm, ok = _heston_otm_rel(x, T, p)
    if not ok:
        return np.full(x.shape, np.nan), otm
    return _iv_from_otm_rel(otm, x, np.full(x.shape, T), min_otm_price_rel), otm


# ── Training Data Generation ──────────────────────────────────────────────────

def _load_baseline_params() -> dict:
    """Load calibrated params from the permanent baseline file."""
    if _BASELINE_PATH.exists():
        with open(_BASELINE_PATH) as f:
            baseline = json.load(f)
        params = baseline["params"]
        logger.info("Loaded baseline params from %s: %s", _BASELINE_PATH, params)
        return params
    logger.warning("Baseline file not found — using default params")
    return _DEFAULT_CAL_PARAMS


def _sample_heston_params(n: int, center: dict, perturb: float = 0.30, rng: np.random.Generator = None) -> dict:
    """
    Sample n Heston parameter sets uniformly around a centre ±perturb, clipped
    to HESTON_BOUNDS. (The original training design; no longer used by
    generate_*_training_data, which sample SPX_TRAINING_DOMAIN instead.)

    For rho (negative by convention), multiplicative ±30% means:
      rho ~ Uniform[rho_0 * (1+perturb), rho_0 * (1-perturb)]
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    samples = {}
    for name, p0 in center.items():
        lo_bound, hi_bound = HESTON_BOUNDS[name]
        if name == "rho":
            lo = max(p0 * (1 + perturb), lo_bound)
            hi = min(p0 * (1 - perturb), hi_bound)
        else:
            lo = max(p0 * (1 - perturb), lo_bound)
            hi = min(p0 * (1 + perturb), hi_bound)
        if lo > hi:
            lo, hi = hi, lo
        samples[name] = rng.uniform(lo, hi, n)

    return samples


def _spx_rows_for_sets(args) -> pd.DataFrame:
    """
    Worker: exact labels for a block of parameter sets (deterministic per
    set_id). Prices every slice, then inverts the whole block in one solver
    call. With a part_path the block is written there (atomically) and
    reused if it already exists, so an interrupted generation resumes.
    """
    set_ids, params, domain, n_T, n_K, seed, floor, part_path = args
    if part_path is not None and Path(part_path).exists():
        return pd.read_parquet(part_path)
    cols = {k: [] for k in ("set_id", "x", "sqrt_T", *_PARAM_NAMES, "otm_rel")}
    for j, sid in enumerate(set_ids):
        p = {k: float(params[k][j]) for k in _PARAM_NAMES}
        rng = np.random.default_rng([seed, int(sid)])
        sqT = rng.uniform(np.sqrt(domain["T"][0]), np.sqrt(domain["T"][1]), n_T)
        for s in sqT:
            T = float(s * s)
            lo, hi = _x_interval(T, p, domain)
            x = rng.uniform(lo, hi, n_K)
            otm, ok = _heston_otm_rel(x, T, p)
            cols["set_id"].append(np.full(n_K, int(sid)))
            cols["x"].append(x)
            cols["sqrt_T"].append(np.full(n_K, float(s)))
            for k in _PARAM_NAMES:
                cols[k].append(np.full(n_K, p[k]))
            cols["otm_rel"].append(otm if ok else np.full(n_K, np.nan))
    block = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    iv = _iv_from_otm_rel(block["otm_rel"].to_numpy(), block["x"].to_numpy(),
                          block["sqrt_T"].to_numpy() ** 2, floor)
    block.insert(len(block.columns) - 1, "iv_target", iv)
    block = block[np.isfinite(iv)].reset_index(drop=True)
    if part_path is not None:
        tmp = Path(str(part_path) + ".tmp")
        block.to_parquet(tmp, index=False)
        os.replace(tmp, part_path)
    return block


def _read_cached(path: Path, meta_expected: dict) -> Optional[pd.DataFrame]:
    """Load a cached training parquet if its recorded metadata matches."""
    if not path.exists():
        return None
    import pyarrow.parquet as pq
    meta = pq.read_schema(path).metadata or {}
    found = json.loads(meta.get(b"nn_training_meta", b"{}").decode())
    if found != json.loads(json.dumps(meta_expected)):
        logger.info("Cached training data %s was generated with different settings; regenerating", path)
        return None
    logger.info("Loading cached training data from %s", path)
    return pd.read_parquet(path)


def _write_cached(df: pd.DataFrame, path: Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"nn_training_meta"] = json.dumps(meta).encode()
    pq.write_table(table.replace_schema_metadata(md), path)
    logger.info("Saved %d rows to %s", len(df), path)


def generate_spx_training_data(
    n_samples: int = NN_TRAIN_SAMPLES,
    force_regen: bool = False,
    *,
    domain: Optional[dict] = None,
    n_T: int = 6,
    n_K: int = 10,
    seed: int = RANDOM_SEED,
    min_otm_price_rel: float = NN_LABEL_MIN_OTM_PRICE_REL,
    path: Optional[Path] = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Generate (or load from cache) exact training data for NN-1.

    Layout: n_param_sets = n_samples // (n_T·n_K) parameter sets drawn from
    `domain`; per set n_T maturities (√T uniform) and per maturity n_K strikes
    uniform on the admissible x interval. Labels below the price floor are
    not produced, so the row count is somewhat below n_samples.

    The cache is reused only if it was generated with identical settings
    (recorded in the parquet metadata). Results do not depend on n_jobs.

    Returns
    -------
    DataFrame [set_id, x, sqrt_T, kappa, theta, sigma, rho, v0, iv_target, otm_rel]
    """
    domain = domain or SPX_TRAINING_DOMAIN
    path = Path(path) if path else _SPX_DATA_PATH
    n_sets = max(n_samples // (n_T * n_K), 1)
    meta = {"version": 2, "domain": domain, "n_param_sets": n_sets, "n_T": n_T,
            "n_K": n_K, "seed": seed, "min_otm_price_rel": min_otm_price_rel,
            "labels": "Lewis/Fourier Heston price + implied_vol_black (OTM)"}
    if not force_regen:
        cached = _read_cached(path, meta)
        if cached is not None:
            return cached

    logger.info("Generating SPX training data: %d parameter sets × %d T × %d K ...",
                n_sets, n_T, n_K)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    params = _sample_domain_params(n_sets, domain, rng)
    ids = np.arange(n_sets)
    blocks = [b for b in np.array_split(ids, max(1, min(n_sets, n_sets // 25 + 1))) if len(b)]
    # Finished blocks are kept next to the target (keyed by the settings) so an
    # interrupted run resumes instead of starting over; removed at the end.
    key = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    parts_dir = path.parent / f"{path.stem}.parts-{key}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(b, {k: params[k][b] for k in _PARAM_NAMES}, domain, n_T, n_K, seed,
              min_otm_price_rel, parts_dir / f"block_{i:05d}.parquet")
             for i, b in enumerate(blocks)]

    def _progress(done):
        if done % max(len(tasks) // 20, 1) == 0 or done == len(tasks):
            logger.info("  %d/%d blocks (%.0fs)", done, len(tasks), time.time() - t0)

    parts = []
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(n_jobs) as pool:
            for part in pool.imap_unordered(_spx_rows_for_sets, tasks):
                parts.append(part)
                _progress(len(parts))
    else:
        for t in tasks:
            parts.append(_spx_rows_for_sets(t))
            _progress(len(parts))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df.sort_values(["set_id", "sqrt_T", "x"], kind="mergesort").reset_index(drop=True)

    n_attempted = n_sets * n_T * n_K
    logger.info("SPX training data: %d labelled of %d sampled options (%.1f%% below the "
                "%.0e·F floor) in %.0fs", len(df), n_attempted,
                100 * (1 - len(df) / n_attempted), min_otm_price_rel, time.time() - t0)
    _write_cached(df, path, meta)
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)
    return df


def generate_vix_training_data(
    n_samples: int = NN_TRAIN_SAMPLES,
    force_regen: bool = False,
    *,
    domain: Optional[dict] = None,
    n_tenors: int = 10,
    seed: int = RANDOM_SEED + 1,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate (or load) training data for the deprecated NN-2.

    Features: kappa, theta, sigma, v0, tenor_years (rho drops out of the VIX
    formula). Target: heston_vix_futures_curve value in VIX points. Each
    parameter set gets n_tenors tenors; rows carry set_id for group splits.

    Returns
    -------
    DataFrame [set_id, kappa, theta, sigma, v0, tenor_years, vix_target]
    """
    domain = domain or VIX_TRAINING_DOMAIN
    path = Path(path) if path else _VIX_DATA_PATH
    n_sets = max(n_samples // n_tenors, 1)
    meta = {"version": 2, "domain": domain, "n_param_sets": n_sets,
            "n_tenors": n_tenors, "seed": seed}
    if not force_regen:
        cached = _read_cached(path, meta)
        if cached is not None:
            return cached

    t0 = time.time()
    rng = np.random.default_rng(seed)
    params = _sample_domain_params(n_sets, domain, rng)
    lo, hi = domain["tenor_years"]
    tenors = rng.uniform(lo, hi, (n_sets, n_tenors))
    target = np.empty((n_sets, n_tenors))
    for i in range(n_sets):
        target[i] = heston_vix_futures_curve(params["kappa"][i], params["theta"][i],
                                             params["sigma"][i], params["v0"][i], tenors[i])
    df = pd.DataFrame({
        "set_id": np.repeat(np.arange(n_sets), n_tenors),
        **{k: np.repeat(params[k], n_tenors) for k in ("kappa", "theta", "sigma", "v0")},
        "tenor_years": tenors.ravel(),
        "vix_target": target.ravel(),
    })
    df = df[np.isfinite(df["vix_target"])].reset_index(drop=True)
    logger.info("VIX training data: %d rows in %.1fs", len(df), time.time() - t0)
    _write_cached(df, path, meta)
    return df


# ── Network Architecture ──────────────────────────────────────────────────────

class HestonNet(nn.Module):
    """
    MLP used by both surrogates.

    Architecture (per config.py):
      Linear(input_dim, hidden) → SiLU, then (n_layers − 1) × [Linear(hidden, hidden) → SiLU],
      Linear(hidden, 1) → Softplus  (strictly positive output)

    Parameters
    ----------
    input_dim  : int — number of input features
    hidden_dim : int — width of each hidden layer (default from config: 256)
    n_layers   : int — number of hidden layers (default: 4)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = NN_HIDDEN_DIM,
        n_layers: int = NN_N_LAYERS,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Softplus())    # output strictly positive
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SPXImpliedVolNet(nn.Module):
    """
    NN-1: implied vol = σ_ref(T, κ, θ, v0) × HestonNet(standardised features).

    Input: raw SPX_FEATURES (x, √T, κ, θ, σ, ρ, v0). Engineered features:
    x, √T, ln κ, ln θ, σ, ρ, ln v0, ln σ_ref, z = x/(σ_ref√T). The network
    learns the smile relative to the level σ_ref (the expected average vol),
    which spans 1%–50% across the domain; the product is trained with MSE in
    implied-vol units. Standardisation constants are buffers in state_dict.
    """

    N_ENGINEERED = 9

    def __init__(self, hidden_dim: int = NN_HIDDEN_DIM, n_layers: int = NN_N_LAYERS):
        super().__init__()
        self.mlp = HestonNet(self.N_ENGINEERED, hidden_dim, n_layers)
        self.register_buffer("feat_mean", torch.zeros(self.N_ENGINEERED))
        self.register_buffer("feat_scale", torch.ones(self.N_ENGINEERED))

    @staticmethod
    def engineer(raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, sqT, kappa, theta, sigma, rho, v0 = raw.unbind(-1)
        T = sqT * sqT
        kT = kappa * T
        B = -torch.expm1(-kT) / kT
        sig_ref = torch.sqrt(torch.clamp(theta + (v0 - theta) * B, min=1e-12))
        z = x / (sig_ref * sqT)
        eng = torch.stack([x, sqT, torch.log(kappa), torch.log(theta), sigma, rho,
                           torch.log(v0), torch.log(sig_ref), z], dim=-1)
        return eng, sig_ref

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        eng, sig_ref = self.engineer(raw)
        return sig_ref * self.mlp((eng - self.feat_mean) / self.feat_scale)


# ── Training ──────────────────────────────────────────────────────────────────

def _make_tensors(
    X: np.ndarray, y: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.float32, device=device),
    )


def _split_by_group(groups: np.ndarray, val_frac: float, test_frac: float, seed: int):
    """Shuffle the unique groups (seeded) and assign whole groups to train/val/test."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = int(round(len(uniq) * test_frac))
    n_val = max(int(round(len(uniq) * val_frac)), 1)
    test_g, val_g = perm[:n_test], perm[n_test:n_test + n_val]
    is_test = np.isin(groups, test_g)
    is_val = np.isin(groups, val_g)
    is_train = ~(is_test | is_val)
    return np.flatnonzero(is_train), np.flatnonzero(is_val), np.flatnonzero(is_test)


def _error_stats(pred: np.ndarray, y: np.ndarray) -> dict:
    err = np.abs(np.asarray(pred, float) - np.asarray(y, float))
    if err.size == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    return {"n": int(err.size), "mae": float(err.mean()), "rmse": float(np.sqrt((err**2).mean())),
            "p99": float(np.quantile(err, 0.99)), "max": float(err.max())}


def _fit(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    verbose: bool,
    model_name: str,
) -> dict:
    """Adam + ReduceLROnPlateau + early stopping on validation MSE; keeps the best state."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    loss_fn = nn.MSELoss()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_train = X_train.shape[0]
    n_steps = max(n_train // batch_size, 1)

    best_val, best_state, best_epoch, patience_count = float("inf"), None, 0, 0
    train_mse = float("nan")
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n_train, generator=gen).to(X_train.device)
        loss_sum = 0.0
        for step in range(n_steps):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_train[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            loss_sum += loss.item() * len(idx)
        train_mse = loss_sum / (n_steps * batch_size)

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(X_val), y_val).item()
        sched.step(val_mse)

        if val_mse < best_val:
            best_val, best_epoch, patience_count = val_mse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  [{model_name}] epoch {epoch:3d}/{max_epochs}  train_rmse={np.sqrt(train_mse):.3e}  "
                  f"val_rmse={np.sqrt(val_mse):.3e}  lr={opt.param_groups[0]['lr']:.1e}  "
                  f"patience={patience_count}/{patience}  ({time.time() - t0:.0f}s)", flush=True)
        if patience_count >= patience:
            if verbose:
                print(f"  [{model_name}] early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_mse": best_val, "best_epoch": best_epoch, "epochs_run": epoch,
            "train_mse": train_mse, "train_time": time.time() - t0}


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR),
                             capture_output=True, text=True, timeout=5)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:
        return None


def _data_sha256(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def _training_meta(seed: int, X: np.ndarray, y: np.ndarray, n_tr: int, n_val: int, n_te: int) -> dict:
    return {
        "seed": int(seed),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "data_sha256": _data_sha256(X, y),
        "n_train": int(n_tr), "n_val": int(n_val), "n_test": int(n_te),
        "code_revision": _git_revision(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def train_network(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_path: Path,
    model_name: str = "HestonNet",
    hidden_dim: int = NN_HIDDEN_DIM,
    n_layers: int = NN_N_LAYERS,
    lr: float = NN_LEARNING_RATE,
    batch_size: int = NN_BATCH_SIZE,
    max_epochs: int = NN_MAX_EPOCHS,
    val_frac: float = 0.10,
    patience: int = 12,
    verbose: bool = True,
    *,
    test_frac: float = 0.10,
    group_col: Optional[str] = "set_id",
    target_mae: Optional[float] = None,
    target_units: str = "target units",
    seed: int = RANDOM_SEED,
    domain: Optional[dict] = None,
) -> dict:
    """
    Train a generic HestonNet (standardised inputs) on a feature/target frame.

    Protocol
    --------
      - Rows are split by `group_col` (parameter-set id) into train/val/test
        with fractions (1 − val − test, val, test); without that column the
        split falls back to contiguous blocks (last val_frac, then test_frac).
      - Inputs standardised with training-set mean/std only.
      - torch and the shuffling generator are seeded before the model is
        built, so identical inputs give identical weights.
      - Adam, ReduceLROnPlateau (factor 0.5, patience 5), early stopping on
        validation MSE; the best-validation weights are kept and saved.
      - Accuracy is reported on the held-out TEST groups, in target units.

    Returns
    -------
    dict with val_* and test_* metrics in target units (mae, rmse, p99, max),
    passed_target (test MAE < target_mae, or None when no target is given),
    plus the legacy keys train_mse, val_mse, val_rmse_vp, mae_vol_pts
    (= 100 × validation MAE: vol points only for an implied-vol target),
    best_epoch, epochs_run, train_time.
    """
    device = torch.device("cpu")
    X_all = df[feature_cols].to_numpy(dtype=np.float64)
    y_all = df[target_col].to_numpy(dtype=np.float64)

    split = {"group_col": None}
    if group_col and group_col in df.columns:
        groups = df[group_col].to_numpy()
        i_tr, i_val, i_te = _split_by_group(groups, val_frac, test_frac, seed)
        split = {"group_col": group_col,
                 "val_groups": sorted(int(g) for g in np.unique(groups[i_val])),
                 "test_groups": sorted(int(g) for g in np.unique(groups[i_te]))}
    else:
        logger.warning("[%s] no '%s' column: falling back to a contiguous split", model_name, group_col)
        n = len(df)
        n_val = max(int(n * val_frac), 1)
        n_te = int(n * test_frac)
        i_tr = np.arange(0, n - n_val - n_te)
        i_val = np.arange(n - n_val - n_te, n - n_te)
        i_te = np.arange(n - n_te, n)

    mean = X_all[i_tr].mean(axis=0)
    scale = X_all[i_tr].std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Xs = ((X_all - mean) / scale).astype(np.float32)

    X_tr, y_tr = _make_tensors(Xs[i_tr], y_all[i_tr], device)
    X_val, y_val = _make_tensors(Xs[i_val], y_all[i_val], device)
    X_te = torch.tensor(Xs[i_te], dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HestonNet(len(feature_cols), hidden_dim, n_layers).to(device)
    if verbose:
        print(f"\n[{model_name}] {len(i_tr):,} train / {len(i_val):,} val / {len(i_te):,} test rows, "
              f"{len(feature_cols)} features", flush=True)
    fit = _fit(model, X_tr, y_tr, X_val, y_val, lr=lr, batch_size=batch_size,
               max_epochs=max_epochs, patience=patience, seed=seed, verbose=verbose,
               model_name=model_name)

    with torch.no_grad():
        val_pred = model(X_val).cpu().numpy()
        te_pred = model(X_te).cpu().numpy() if len(i_te) else np.empty(0)
    val_stats = _error_stats(val_pred, y_all[i_val])
    test_stats = _error_stats(te_pred, y_all[i_te])
    report = test_stats if test_stats["n"] else val_stats
    passed = None if target_mae is None else bool(report["mae"] < target_mae)

    meta = _training_meta(seed, X_all, y_all, len(i_tr), len(i_val), len(i_te))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format":       "heston_net_v2",
        "state_dict":   model.state_dict(),
        "scaler_mean":  mean.tolist(),
        "scaler_scale": scale.tolist(),
        "feature_cols": list(feature_cols),
        "target_col":   target_col,
        "target_units": target_units,
        "input_dim":    len(feature_cols),
        "hidden_dim":   int(hidden_dim),
        "n_layers":     int(n_layers),
        "best_val_mse": float(fit["best_val_mse"]),
        "best_epoch":   int(fit["best_epoch"]),
        "val_metrics":  val_stats,
        "test_metrics": test_stats,
        "domain":       domain,
        "training":     {**meta, "split": split},
    }, model_path)
    logger.info("[%s] model saved to %s", model_name, model_path)

    if verbose:
        print(f"[{model_name}] best epoch {fit['best_epoch']}, test MAE {report['mae']:.4g} "
              f"{target_units} (p99 {report['p99']:.4g}), val MAE {val_stats['mae']:.4g} "
              f"{target_units}; target {target_mae} → {passed}", flush=True)

    return {
        "train_mse":     fit["train_mse"],
        "val_mse":       fit["best_val_mse"],
        "val_rmse_vp":   float(np.sqrt(fit["best_val_mse"]) * 100),
        "mae_vol_pts":   val_stats["mae"] * 100,
        "val_mae":       val_stats["mae"],
        "test_mae":      test_stats["mae"],
        "test_rmse":     test_stats["rmse"],
        "test_p99":      test_stats["p99"],
        "test_max":      test_stats["max"],
        "target_units":  target_units,
        "n_train":       len(i_tr), "n_val": len(i_val), "n_test": len(i_te),
        "best_epoch":    fit["best_epoch"],
        "epochs_run":    fit["epochs_run"],
        "train_time":    fit["train_time"],
        "passed_target": passed,
    }


def _moneyness_buckets(x: np.ndarray, err: np.ndarray) -> dict:
    edges = [(-0.30, -0.10), (-0.10, 0.0), (0.0, 0.10), (0.10, 0.25)]
    out = {}
    for lo, hi in edges:
        m = (x >= lo) & ((x < hi) if hi < 0.25 else (x <= hi))
        out[f"[{lo:+.2f},{hi:+.2f}]"] = (float(np.mean(err[m])) if m.any() else None, int(m.sum()))
    return out


def train_spx(
    force_regen: bool = False,
    *,
    data: Optional[pd.DataFrame] = None,
    n_samples: int = NN_TRAIN_SAMPLES,
    model_path: Optional[Path] = None,
    hidden_dim: int = NN_HIDDEN_DIM,
    n_layers: int = NN_N_LAYERS,
    lr: float = NN_LEARNING_RATE,
    batch_size: int = NN_BATCH_SIZE,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = 15,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = RANDOM_SEED,
    n_jobs: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Generate (or load) exact SPX training data and train NN-1 (SPXImpliedVolNet).

    `data` lets callers (tests) pass a prepared frame with SPX_FEATURES,
    set_id and iv_target. Accuracy is reported on held-out parameter sets in
    vol points; passed_target compares the test MAE with
    config.NN_VOL_ERROR_TARGET (0.1 vol point).
    """
    df = data if data is not None else generate_spx_training_data(
        n_samples, force_regen, n_jobs=n_jobs)
    model_path = Path(model_path) if model_path else _SPX_MODEL_PATH
    X_all = df[SPX_FEATURES].to_numpy(dtype=np.float64)
    y_all = df["iv_target"].to_numpy(dtype=np.float64)
    groups = df["set_id"].to_numpy()
    i_tr, i_val, i_te = _split_by_group(groups, val_frac, test_frac, seed)
    split = {"group_col": "set_id",
             "val_groups": sorted(int(g) for g in np.unique(groups[i_val])),
             "test_groups": sorted(int(g) for g in np.unique(groups[i_te]))}

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SPXImpliedVolNet(hidden_dim, n_layers)
    with torch.no_grad():
        eng, _ = SPXImpliedVolNet.engineer(torch.tensor(X_all[i_tr], dtype=torch.float64))
        mean = eng.mean(dim=0)
        scale = eng.std(dim=0, unbiased=False)
        model.feat_mean.copy_(mean.float())
        model.feat_scale.copy_(torch.where(scale > 0, scale, torch.ones_like(scale)).float())

    X_tr, y_tr = _make_tensors(X_all[i_tr], y_all[i_tr], torch.device("cpu"))
    X_val, y_val = _make_tensors(X_all[i_val], y_all[i_val], torch.device("cpu"))
    X_te = torch.tensor(X_all[i_te], dtype=torch.float32)
    if verbose:
        print(f"\n[NN-1 (SPX)] {len(i_tr):,} train / {len(i_val):,} val / {len(i_te):,} test rows "
              f"({len(np.unique(df['set_id']))} parameter sets)", flush=True)
    fit = _fit(model, X_tr, y_tr, X_val, y_val, lr=lr, batch_size=batch_size,
               max_epochs=max_epochs, patience=patience, seed=seed, verbose=verbose,
               model_name="NN-1 (SPX)")

    with torch.no_grad():
        val_pred = model(X_val).numpy()
        te_pred = model(X_te).numpy() if len(i_te) else np.empty(0)
    val_vp = {k: (v * 100 if k != "n" else v) for k, v in _error_stats(val_pred, y_all[i_val]).items()}
    test_vp = {k: (v * 100 if k != "n" else v) for k, v in _error_stats(te_pred, y_all[i_te]).items()}
    report = test_vp if test_vp["n"] else val_vp
    target_vp = NN_VOL_ERROR_TARGET * 100
    by_bucket = (_moneyness_buckets(X_all[i_te, 0], np.abs(te_pred - y_all[i_te]) * 100)
                 if len(i_te) else {})

    ckpt = {
        "format":       _SPX_CKPT_FORMAT,
        "state_dict":   model.state_dict(),
        "feature_cols": list(SPX_FEATURES),
        "hidden_dim":   int(hidden_dim),
        "n_layers":     int(n_layers),
        "domain":       SPX_TRAINING_DOMAIN,
        "label_floor_otm_rel": NN_LABEL_MIN_OTM_PRICE_REL,
        "units":        "implied vol, decimal; metrics in vol points",
        "best_val_mse": float(fit["best_val_mse"]),
        "best_epoch":   int(fit["best_epoch"]),
        "val_metrics_vp":  val_vp,
        "test_metrics_vp": test_vp,
        "test_mae_by_moneyness_vp": {k: [v[0], v[1]] for k, v in by_bucket.items()},
        "training":     {**_training_meta(seed, X_all, y_all, len(i_tr), len(i_val), len(i_te)),
                         "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
                         "patience": patience, "epochs_run": int(fit["epochs_run"]),
                         "train_time_s": float(fit["train_time"]),
                         "torch_threads": int(torch.get_num_threads()),
                         "split": split},
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, model_path)
    logger.info("[NN-1 (SPX)] model saved to %s", model_path)
    if verbose:
        print(f"[NN-1 (SPX)] test MAE {report['mae']:.3f} vp, RMSE {report['rmse']:.3f} vp, "
              f"p99 {report['p99']:.3f} vp, max {report['max']:.3f} vp "
              f"(target {target_vp:.2f} vp); by moneyness {by_bucket}", flush=True)

    return {
        "train_mse":     fit["train_mse"],
        "val_mse":       fit["best_val_mse"],
        "val_rmse_vp":   float(np.sqrt(fit["best_val_mse"]) * 100),
        "mae_vol_pts":   report["mae"],
        "val_mae_vp":    val_vp["mae"],
        "test_mae_vp":   test_vp["mae"],
        "test_rmse_vp":  test_vp["rmse"],
        "test_p99_vp":   test_vp["p99"],
        "test_max_vp":   test_vp["max"],
        "test_mae_by_moneyness_vp": by_bucket,
        "n_train":       len(i_tr), "n_val": len(i_val), "n_test": len(i_te),
        "best_epoch":    fit["best_epoch"],
        "epochs_run":    fit["epochs_run"],
        "train_time":    fit["train_time"],
        "passed_target": bool(report["mae"] < target_vp),
    }


# ── Inference Wrapper ─────────────────────────────────────────────────────────

class NNPricer:
    """
    Inference wrapper for NN-1 (SPX implied vols) and the deprecated NN-2.

    Inputs are checked against the training domain stored in the checkpoint;
    out-of-domain rows raise by default (see spx_iv_batch).

    Usage
    -----
        pricer = NNPricer.load()
        iv  = pricer.spx_iv(log_moneyness=-0.05, T=0.25,
                            kappa=4.62, theta=0.076, sigma=0.84, rho=-0.95, v0=0.056)
        ivs = pricer.spx_iv_batch(features)        # (N, 7) in SPX_FEATURES order
    """

    def __init__(self):
        self._spx_model: Optional[nn.Module] = None
        self._vix_model: Optional[HestonNet] = None
        self._spx_scaler_mean:  Optional[np.ndarray] = None   # legacy checkpoints only
        self._spx_scaler_scale: Optional[np.ndarray] = None
        self._vix_scaler_mean:  Optional[np.ndarray] = None
        self._vix_scaler_scale: Optional[np.ndarray] = None
        self._spx_domain: Optional[dict] = None
        self._vix_domain: Optional[dict] = None
        self._spx_meta: dict = {}
        self._vix_meta: dict = {}
        self._device = torch.device("cpu")

    @classmethod
    def load(
        cls,
        spx_path: Optional[str] = None,
        vix_path: Optional[str] = None,
        retrain_if_missing: bool = False,
    ) -> "NNPricer":
        """
        Load NN-1 (required) and NN-2 (optional, deprecated) from disk.

        retrain_if_missing now defaults to False: generating exact labels and
        training takes tens of minutes, which is not something a `load` call
        should do implicitly. A missing NN-1 raises FileNotFoundError; a
        missing NN-2 raises only if vix_path was given explicitly.
        """
        pricer = cls()
        spx_p = Path(spx_path) if spx_path else _SPX_MODEL_PATH
        vix_p = Path(vix_path) if vix_path else _VIX_MODEL_PATH

        if not spx_p.exists():
            if not retrain_if_missing:
                raise FileNotFoundError(f"SPX model not found: {spx_p}")
            logger.info("NN-1 not found — training now...")
            train_spx(model_path=spx_p)
        pricer._load_spx(spx_p)
        logger.info("NN-1 (SPX) loaded from %s", spx_p)

        if vix_p.exists():
            pricer._load_vix(vix_p)
            logger.info("NN-2 (VIX, deprecated) loaded from %s", vix_p)
        elif vix_path is not None:
            raise FileNotFoundError(f"VIX model not found: {vix_p}")
        elif retrain_if_missing:
            train_vix(model_path=vix_p)
            pricer._load_vix(vix_p)
        return pricer

    def _load_spx(self, path: Path):
        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        if ckpt.get("format") == _SPX_CKPT_FORMAT:
            model = SPXImpliedVolNet(ckpt["hidden_dim"], ckpt["n_layers"])
            model.load_state_dict(ckpt["state_dict"])
            self._spx_domain = ckpt["domain"]
        else:
            warnings.warn(
                f"{path} is a legacy NN-1 checkpoint with no recorded training domain; "
                "out-of-domain inputs cannot be detected", RuntimeWarning, stacklevel=3,
            )
            model = HestonNet(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
            model.load_state_dict(ckpt["state_dict"])
            self._spx_scaler_mean  = np.array(ckpt["scaler_mean"],  dtype=np.float32)
            self._spx_scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
            self._spx_domain = None
        model.to(self._device).eval()
        self._spx_model = model
        self._spx_meta = ckpt

    def _load_vix(self, path: Path):
        ckpt = torch.load(path, map_location=self._device, weights_only=True)
        model = HestonNet(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(self._device).eval()
        self._vix_model        = model
        self._vix_scaler_mean  = np.array(ckpt["scaler_mean"],  dtype=np.float32)
        self._vix_scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        self._vix_domain = ckpt.get("domain")
        self._vix_meta = ckpt

    def _normalise(self, X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> torch.Tensor:
        X_sc = (X.astype(np.float32) - mean) / scale
        return torch.tensor(X_sc, dtype=torch.float32, device=self._device)

    def spx_out_of_domain(self, features: np.ndarray) -> np.ndarray:
        """Mask of rows outside NN-1's training domain (all False for legacy checkpoints)."""
        f = np.asarray(features, dtype=np.float64)
        if self._spx_domain is None:
            return np.zeros(len(f), dtype=bool)
        return spx_out_of_domain(f, self._spx_domain)

    def spx_iv(
        self,
        log_moneyness: float,
        T: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
        out_of_domain: str = "raise",
    ) -> float:
        """
        Predict the Black-Scholes implied vol for a single SPX option.

        log_moneyness : ln(K / F) where F = S·exp((r−q)T);  T in years.
        Returns the implied vol in decimal (0.22 = 22%).
        """
        features = np.array([[log_moneyness, np.sqrt(T), kappa, theta, sigma, rho, v0]])
        return float(self.spx_iv_batch(features, out_of_domain=out_of_domain)[0])

    def spx_iv_batch(self, features: np.ndarray, out_of_domain: str = "raise") -> np.ndarray:
        """
        Batch prediction of SPX implied vols.

        Parameters
        ----------
        features      : (N, 7) — columns [x, sqrt_T, kappa, theta, sigma, rho, v0]
        out_of_domain : what to do with rows outside the training domain:
                        "raise" (default) ValueError; "nan" return NaN there;
                        "exact" price them exactly instead (slow, per slice).

        Returns
        -------
        np.ndarray (N,) — implied vols in decimal
        """
        if self._spx_model is None:
            raise RuntimeError("NN-1 not loaded")
        if out_of_domain not in ("raise", "nan", "exact"):
            raise ValueError(f"out_of_domain must be 'raise', 'nan' or 'exact', got {out_of_domain!r}")
        f = np.asarray(features, dtype=np.float64)
        if f.ndim != 2 or f.shape[1] != len(SPX_FEATURES):
            raise ValueError(f"features must have shape (N, {len(SPX_FEATURES)}), got {f.shape}")
        ood = self.spx_out_of_domain(f)
        if ood.any() and out_of_domain == "raise":
            raise ValueError(
                f"{int(ood.sum())} of {len(f)} inputs are outside NN-1's training domain "
                f"{self._spx_domain}; first offending row: {f[np.argmax(ood)].tolist()}. "
                "Pass out_of_domain='nan' or 'exact' to handle them explicitly."
            )

        with torch.no_grad():
            if self._spx_scaler_mean is not None:      # legacy HestonNet checkpoint
                out = self._spx_model(self._normalise(f, self._spx_scaler_mean,
                                                      self._spx_scaler_scale)).cpu().numpy()
            else:
                out = self._spx_model(torch.tensor(f, dtype=torch.float32)).cpu().numpy()
        out = out.astype(np.float64)
        if ood.any():
            if out_of_domain == "nan":
                out[ood] = np.nan
            else:
                out[ood] = _exact_rows(f[ood])
        return out

    def vix_prices(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        v0: float,
        tenors: np.ndarray,
    ) -> np.ndarray:
        """
        DEPRECATED NN-2 prediction of heston_vix_futures_curve (VIX points).

        Call heston_vix_futures_curve directly: it is exact and faster than
        this network. Inputs outside NN-2's training domain raise ValueError.
        """
        warnings.warn("NNPricer.vix_prices (NN-2) is deprecated: heston_vix_futures_curve "
                      "is exact and faster", DeprecationWarning, stacklevel=2)
        if self._vix_model is None:
            raise RuntimeError("NN-2 not loaded")
        tenors = np.atleast_1d(np.asarray(tenors, dtype=float))
        features = np.column_stack([
            np.full(len(tenors), kappa), np.full(len(tenors), theta),
            np.full(len(tenors), sigma), np.full(len(tenors), v0), tenors,
        ])
        if self._vix_domain is not None:
            names = VIX_FEATURES
            for i, n in enumerate(names):
                lo, hi = self._vix_domain[n]
                col = features[:, i]
                if np.any((col < lo - 1e-9 * abs(lo)) | (col > hi + 1e-9 * abs(hi))):
                    raise ValueError(f"NN-2 input {n} outside training domain [{lo}, {hi}]")
        X_t = self._normalise(features, self._vix_scaler_mean, self._vix_scaler_scale)
        with torch.no_grad():
            return self._vix_model(X_t).cpu().numpy()

    def benchmark(
        self,
        S: float = 6581.0,
        r: float = 0.045,
        q: float = 0.013,
        n_options: int = 500,
        *,
        n_param_sets: int = 20,
        maturities_days: tuple = (14, 30, 61, 91, 182, 365),
        param_sets: Optional[list] = None,
        seed: int = RANDOM_SEED + 1000,
        repeats: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Accuracy and speed of NN-1 on a calibration-like workload.

        For each parameter set (fresh draws from the training domain, or
        `param_sets`), n_options options are spread over `maturities_days`
        with strikes uniform on each maturity's admissible x interval (the
        full domain, OTM-call wing included). Accuracy: NN vs the exact
        reference (Fourier price + robust solver) on EVERY option whose exact
        OTM value is at least NN_LABEL_MIN_OTM_PRICE_REL·F — never filtered
        on the NN's output. Speed: per parameter set, one NN forward pass vs
        heston_call_batch + _bs_iv_vectorized for the same options (one call
        per maturity), best of `repeats`; the speedup is reported as the
        median and inter-quartile range over parameter sets.

        Returns a dict with the legacy keys (n_options, heston_ms, nn_ms,
        speedup, mae_vol_pts, rmse_vol_pts, target_passed) plus p99/max error,
        MAE by moneyness and maturity, the production pricer's own error,
        and the machine the timing was taken on.
        """
        if self._spx_model is None:
            raise RuntimeError("NN-1 not loaded")
        domain = self._spx_domain or SPX_TRAINING_DOMAIN
        rng = np.random.default_rng(seed)
        if param_sets is None:
            drawn = _sample_domain_params(n_param_sets, domain, rng)
            param_sets = [{k: float(drawn[k][i]) for k in _PARAM_NAMES} for i in range(n_param_sets)]
        mats = np.asarray(maturities_days, dtype=float) / 365.0
        n_per = max(n_options // len(mats), 1)

        def _best_of(fn):
            best, out = float("inf"), None
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn()
                best = min(best, time.perf_counter() - t0)
            return out, best

        all_x, all_T, all_nn, all_ref, all_prod = [], [], [], [], []
        heston_t, nn_t = [], []
        for p in param_sets:
            xs = []
            for T in mats:
                lo, hi = _x_interval(T, p, domain)
                xs.append(rng.uniform(lo, hi, n_per))
            feats = np.vstack([
                np.column_stack([x, np.full(n_per, np.sqrt(T))]
                                + [np.full(n_per, p[k]) for k in _PARAM_NAMES])
                for x, T in zip(xs, mats)
            ])
            ref = np.concatenate([heston_iv_exact(x, T, p)[0] for x, T in zip(xs, mats)])

            Fs = [S * np.exp((r - q) * T) for T in mats]
            Ks = [F * np.exp(x) for F, x in zip(Fs, xs)]

            def _heston_leg():
                out = []
                for T, F, K in zip(mats, Fs, Ks):
                    calls = heston_call_batch(S, K, T, r, q, **p)
                    n = len(K)
                    out.append(_bs_iv_vectorized(calls, np.full(n, F), K,
                                                 np.full(n, T), np.full(n, r)))
                return np.concatenate(out)

            prod, h_time = _best_of(_heston_leg)
            self.spx_iv_batch(feats, out_of_domain="nan")          # warm-up
            nn_out, n_time = _best_of(lambda: self.spx_iv_batch(feats, out_of_domain="nan"))
            heston_t.append(h_time)
            nn_t.append(n_time)
            all_x.append(feats[:, 0]); all_T.append(feats[:, 1] ** 2)
            all_nn.append(nn_out); all_ref.append(ref); all_prod.append(prod)

        x = np.concatenate(all_x); Tm = np.concatenate(all_T)
        nn_iv = np.concatenate(all_nn); ref = np.concatenate(all_ref); prod = np.concatenate(all_prod)
        ok = np.isfinite(ref)
        err = np.abs(nn_iv - ref) * 100
        e_ok = err[ok]
        prod_ok = ok & np.isfinite(prod)
        speedups = np.array(heston_t) / np.maximum(np.array(nn_t), 1e-12)
        mae = float(np.mean(e_ok))
        result = {
            "n_options":      int(len(ref)),
            "n_well_posed":   int(ok.sum()),
            "n_param_sets":   len(param_sets),
            "heston_ms":      float(np.median(heston_t) * 1000),
            "nn_ms":          float(np.median(nn_t) * 1000),
            "speedup":        float(np.median(speedups)),
            "speedup_iqr":    [float(np.quantile(speedups, 0.25)), float(np.quantile(speedups, 0.75))],
            "mae_vol_pts":    mae,
            "rmse_vol_pts":   float(np.sqrt(np.mean(e_ok**2))),
            "p99_vol_pts":    float(np.quantile(e_ok, 0.99)),
            "max_vol_pts":    float(np.max(e_ok)),
            "mae_by_moneyness_vp": _moneyness_buckets(x[ok], e_ok),
            "mae_by_maturity_vp": {int(d): float(np.mean(err[ok & np.isclose(Tm, d / 365.0)]))
                                   for d in maturities_days
                                   if (ok & np.isclose(Tm, d / 365.0)).any()},
            "pricer_mae_vol_pts": float(np.mean(np.abs(prod - ref)[prod_ok]) * 100),
            "pricer_max_vol_pts": float(np.max(np.abs(prod - ref)[prod_ok]) * 100),
            "pricer_n_nan":   int((ok & ~np.isfinite(prod)).sum()),
            "target_passed":  bool(mae < NN_VOL_ERROR_TARGET * 100),
            "machine": {"platform": platform.platform(), "machine": platform.machine(),
                        "cpu_count": os.cpu_count(), "torch": str(torch.__version__),
                        "torch_threads": torch.get_num_threads(), "numpy": np.__version__},
        }
        if verbose:
            print(f"\n── NN-1 benchmark: {len(param_sets)} parameter sets × {len(ref) // len(param_sets)} "
                  f"options ({ok.sum()} well-posed) ──")
            print(f"  heston_call_batch + IV : {result['heston_ms']:.2f} ms per set (median)")
            print(f"  NN forward pass        : {result['nn_ms']:.2f} ms per set (median)")
            print(f"  Speedup                : {result['speedup']:.1f}× "
                  f"(IQR {result['speedup_iqr'][0]:.1f}–{result['speedup_iqr'][1]:.1f}×)")
            print(f"  NN error vs exact      : MAE {mae:.3f} vp, RMSE {result['rmse_vol_pts']:.3f}, "
                  f"p99 {result['p99_vol_pts']:.3f}, max {result['max_vol_pts']:.3f} vp")
            print(f"  by moneyness           : {result['mae_by_moneyness_vp']}")
            print(f"  heston_call_batch error: MAE {result['pricer_mae_vol_pts']:.4f} vp, "
                  f"max {result['pricer_max_vol_pts']:.3f} vp, {result['pricer_n_nan']} NaN")
            print(f"  Target (<{NN_VOL_ERROR_TARGET * 100:.1f} vp)       : "
                  f"{'PASS' if result['target_passed'] else 'FAIL'}")
        return result

    def benchmark_vix(self, n_tenors: int = 8, repeats: int = 200) -> dict:
        """Time NN-2 against the closed form it approximates (single parameter set)."""
        p = dict(kappa=4.62, theta=0.0764, sigma=0.84, v0=0.0561)
        tenors = np.linspace(0.05, 1.0, n_tenors)
        def _best(fn):
            best = float("inf")
            for _ in range(repeats):
                t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
            return best
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            nn_s = _best(lambda: self.vix_prices(tenors=tenors, **p))
        cf_s = _best(lambda: heston_vix_futures_curve(p["kappa"], p["theta"], p["sigma"], p["v0"], tenors))
        return {"nn_us": nn_s * 1e6, "closed_form_us": cf_s * 1e6, "nn_over_closed_form": nn_s / cf_s}

    def __repr__(self) -> str:
        spx_ok = self._spx_model is not None
        vix_ok = self._vix_model is not None
        return f"NNPricer(NN-1={'loaded' if spx_ok else 'not loaded'}, NN-2={'loaded' if vix_ok else 'not loaded'})"


def _exact_rows(f: np.ndarray) -> np.ndarray:
    """Exact implied vols for arbitrary feature rows (grouped by (T, params) slice)."""
    out = np.full(len(f), np.nan)
    keys = [tuple(row[1:]) for row in f]
    for key in dict.fromkeys(keys):
        idx = [i for i, k in enumerate(keys) if k == key]
        sqT, *pv = key
        p = dict(zip(_PARAM_NAMES, pv))
        out[idx] = heston_iv_exact(f[idx, 0], float(sqT) ** 2, p)[0]
    return out


# ── Convenience Training Entry Points ─────────────────────────────────────────

def train_vix(force_regen: bool = False, *, model_path: Optional[Path] = None, **train_kwargs) -> dict:
    """
    Generate VIX training data and train the deprecated NN-2.

    hidden_dim=64, batch 8192 on a 100K subsample of parameter-set groups
    (the target is a smooth closed form). Split by parameter set, so the
    reported test MAE (VIX points) is out-of-sample.
    """
    df = generate_vix_training_data(force_regen=force_regen)
    rng = np.random.default_rng(RANDOM_SEED)
    sets = np.unique(df["set_id"])
    keep = rng.permutation(sets)[: max(1, min(len(sets), 100_000 // 10))]
    df_train = df[df["set_id"].isin(keep)].reset_index(drop=True)
    kwargs = {
        "hidden_dim":  64,
        "n_layers":    4,
        "batch_size":  8192,
        "max_epochs":  150,
        "patience":    15,
    }
    kwargs.update(train_kwargs)
    return train_network(
        df_train, VIX_FEATURES, "vix_target", Path(model_path) if model_path else _VIX_MODEL_PATH,
        model_name="NN-2 (VIX)", target_mae=NN_VIX_TARGET_MAE, target_units="VIX points",
        domain=VIX_TRAINING_DOMAIN, **kwargs,
    )


# ── Command line ──────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    """
    Reproduce the published NN-1 numbers:

      python -m joint_vol_calibration.models.nn_pricer train [--jobs 4] [--epochs 80] [--vix]
      python -m joint_vol_calibration.models.nn_pricer benchmark [--sets 20]

    `train` generates (or reuses the cached) exact training data and writes
    data_store/nn_spx_model.pt; `benchmark` prints the benchmark dict as JSON.
    Training is pinned to one torch thread so runs are reproducible and do
    not oversubscribe a shared machine.
    """
    import argparse
    parser = argparse.ArgumentParser(description=main.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--jobs", type=int, default=1, help="processes for label generation")
    tr.add_argument("--epochs", type=int, default=NN_MAX_EPOCHS)
    tr.add_argument("--vix", action="store_true", help="also train the deprecated NN-2")
    bm = sub.add_parser("benchmark")
    bm.add_argument("--sets", type=int, default=20)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(1)
    if args.cmd == "train":
        generate_spx_training_data(n_jobs=args.jobs)
        print(json.dumps(train_spx(max_epochs=args.epochs), default=str, indent=1))
        if args.vix:
            print(json.dumps(train_vix(), default=str, indent=1))
    else:
        print(json.dumps(NNPricer.load().benchmark(n_param_sets=args.sets), default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
