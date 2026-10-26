"""
desk_artifacts.py — readers for the C23 artifacts the dashboard renders.

The contract (paths, required fields, units) is written out in
dashboard/ARTIFACTS.md; this module is its executable form. It follows the
schemas the producers write (surface report schema_version 1, calibration
diagnostics schema_version 1, inference_summary/1, forward_ledger/1). Each
reader returns one of three states:

  * ``{"state": "absent", ...}``  the artifact has not been generated yet. The
    page shows an explicit "not yet generated" panel naming the path and its
    producer; this is NOT a loader error, so the existing pages still publish.
  * ``{"state": "ok", ...}``      the artifact exists and passed validation.
  * raises ``ArtifactError``       the artifact exists but breaks the contract
    (missing field, wrong unit, wrong session, non-finite numbers, a broken
    hash chain). The calling loader records it, so the site is not published
    (fail closed).

Readers never write, never reprice and never fill a missing number.
Units: implied vols are decimals in the files (0.183 = 18.3%) except fields
ending in ``_vp`` (vol points) and the VIX-strip vols (vol points, as the
index is quoted); log-moneyness k = ln(K / F_T); T in years, T_days in
calendar days; residuals in half-spreads are ``error_hs``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactError(ValueError):
    """An artifact exists but does not satisfy the contract in ARTIFACTS.md."""


# Producers named on the "not yet generated" panels.
PRODUCERS = {
    "surface": "the surface report (scripts/surface_report.py, joint_vol_calibration/surface)",
    "inference": "the inference report (scripts/inference_report.py, joint_vol_calibration/stats)",
    "scenario": "the risk monitor (joint_vol_calibration/greeks)",
    "identifiability": "the calibration diagnostics (scripts/calibration_diagnostics.py)",
    "forward": ("the forward ledger (joint_vol_calibration/backtest/forward_ledger.py), "
                "one file per captured session"),
}

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def absent(kind: str, *paths: Path, data_dir: Path | None = None) -> dict:
    """The state of an artifact that has not been generated yet."""
    rel = []
    for p in paths:
        try:
            rel.append(str(Path(p).relative_to(Path(data_dir).parent)) if data_dir else str(p))
        except ValueError:
            rel.append(str(p))
    return {"state": "absent", "kind": kind, "paths": rel, "producer": PRODUCERS.get(kind, "")}


def _finite(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _read_json(path: Path, name: str | None = None) -> dict:
    name = name or path.name
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:          # JSONDecodeError, UnicodeDecodeError, OSError
        raise ArtifactError(f"{name}: not readable JSON ({exc})") from exc
    if not isinstance(obj, dict):
        raise ArtifactError(f"{name}: top level must be an object")
    return obj


def _require(df: pd.DataFrame, cols: set, name: str) -> None:
    missing = sorted(c for c in cols if c not in df.columns)
    if missing:
        raise ArtifactError(f"{name}: missing column(s) {', '.join(missing)}")


def _check_session(obj: dict, session: str, name: str) -> None:
    sess = str(obj.get("session", session))[:10]
    if sess != session:
        raise ArtifactError(f"{name}: session {sess} != {session}")


def _pct(x: Any, name: str) -> float | None:
    v = _finite(x)
    if x is not None and v is None:
        raise ArtifactError(f"{name}: not a finite number ({x!r})")
    if v is not None and not -1e-9 <= v <= 100.0 + 1e-9:
        raise ArtifactError(f"{name}: {v} is not a percentage in [0, 100]")
    return v


def _prob(x: Any, name: str) -> float | None:
    v = _finite(x)
    if x is not None and v is None:
        raise ArtifactError(f"{name}: not a finite number ({x!r})")
    if v is not None and not -1e-12 <= v <= 1.0 + 1e-12:
        raise ArtifactError(f"{name}: {v} is not a probability in [0, 1]")
    return v


# ── surface: data_store/surface/<session>/ ───────────────────────────────────
def surface_sessions(data_dir: Path) -> list[str]:
    root = Path(data_dir) / "surface"
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and _DATE.match(p.name)
                  and (p / "residuals.parquet").exists())


def read_residuals(path: Path, session: str) -> pd.DataFrame:
    """Per-(model, quote) residual table, validated, in long format.

    One row per model and quote: the market columns (expiry, k, bid_iv,
    mid_iv, ask_iv, half_spread_vp) repeat for every model. Adds, when the
    producer did not store them: half_spread_vp (from bid/ask IVs), T_days
    (from T), error_vp, error_hs and inside_spread. ``df.attrs["models"]`` is
    the models in the file's order.
    """
    name = f"surface/{session}/{path.name}"
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ArtifactError(f"{name}: unreadable ({exc})") from exc
    _require(df, {"model", "expiry", "k", "mid_iv", "model_iv"}, name)
    if df.empty:
        raise ArtifactError(f"{name}: no quotes")
    df = df.copy()
    if "half_spread_vp" not in df.columns:
        _require(df, {"bid_iv", "ask_iv"}, name)
        df["half_spread_vp"] = (df["ask_iv"] - df["bid_iv"]) * 50.0
    if "T_days" not in df.columns:
        _require(df, {"T"}, name)
        df["T_days"] = df["T"] * 365.0
    if "session" in df.columns and set(df["session"].astype(str).str[:10]) != {session}:
        other = sorted(set(df["session"].astype(str)))[:3]
        raise ArtifactError(f"{name}: rows from another session ({other})")
    mid = pd.to_numeric(df["mid_iv"], errors="coerce")
    if not np.isfinite(mid).all() or (mid <= 0).any():
        raise ArtifactError(f"{name}: mid_iv must be positive and finite")
    if float(mid.median()) > 3.0:
        raise ArtifactError(f"{name}: mid_iv median {mid.median():.2f} looks like vol points; "
                            "the contract is decimals")
    miv = pd.to_numeric(df["model_iv"], errors="coerce")
    if miv.notna().any() and float(miv.dropna().median()) > 3.0:
        raise ArtifactError(f"{name}: model_iv looks like vol points; the contract is decimals")
    hs = pd.to_numeric(df["half_spread_vp"], errors="coerce")
    if (hs < 0).any() or not np.isfinite(hs).all():
        raise ArtifactError(f"{name}: half_spread_vp must be finite and >= 0")
    k = pd.to_numeric(df["k"], errors="coerce")
    if not np.isfinite(k).all():
        raise ArtifactError(f"{name}: k must be finite")
    df["expiry"] = df["expiry"].astype(str).str[:10]
    df["model"] = df["model"].astype(str)
    if "error_vp" not in df.columns:
        df["error_vp"] = (miv - mid) * 100.0
    if "error_hs" not in df.columns:
        # the producer's convention: error over max(half-spread, 0.01 vp)
        df["error_hs"] = df["error_vp"] / np.maximum(hs, 0.01)
    if "inside_spread" not in df.columns:
        if {"bid_iv", "ask_iv"} <= set(df.columns):
            df["inside_spread"] = (miv >= df["bid_iv"]) & (miv <= df["ask_iv"])
        else:
            df["inside_spread"] = df["error_hs"].abs() <= 1.0
    if "in_sample" in df.columns:
        df["in_sample"] = df["in_sample"].fillna(False).astype(bool)
    models = list(dict.fromkeys(df["model"]))
    sizes = df.groupby("model").size()
    if sizes.min() != sizes.max():
        raise ArtifactError(f"{name}: models are scored on different quote sets ({dict(sizes)})")
    df = df.sort_values(["T_days", "k"], kind="mergesort").reset_index(drop=True)
    df.attrs["models"] = models
    return df


def market_quotes(res: pd.DataFrame) -> pd.DataFrame:
    """One row per quote: the first model's rows (the market columns repeat per model)."""
    models = res.attrs.get("models") or list(dict.fromkeys(res["model"]))
    q = res[res["model"] == models[0]]
    q.attrs = dict(res.attrs)
    return q


def residual_stats(res: pd.DataFrame, model: str) -> dict:
    """Spread-unit summary of one model on its out-of-sample quotes.

    With an ``in_sample`` column only the quotes the model was NOT fitted to
    are scored (scope "out-of-sample"); without it every quote is scored and
    the scope says "all quotes", so a page never calls an in-sample number
    out-of-sample. Used only when the session has no scorecard.json.
    """
    q = res[res["model"] == model]
    scope = "all quotes"
    if "in_sample" in q.columns:
        q = q[~q["in_sample"]]
        scope = "out-of-sample"
    q = q[pd.to_numeric(q["model_iv"], errors="coerce").notna()]
    if q.empty:
        return {"model": model, "n": 0, "scope": scope}
    z = pd.to_numeric(q["error_hs"], errors="coerce")
    e = pd.to_numeric(q["error_vp"], errors="coerce")
    return {"model": model, "scope": scope, "n": len(q),
            "median_abs_hs": float(z.abs().median()),
            "inside_spread_pct": float(q["inside_spread"].astype(bool).mean() * 100.0),
            "rmse_vp": float(np.sqrt((e ** 2).mean())), "mean_error_vp": float(e.mean()),
            "within_2hs_pct": float((z.abs() <= 2.0).mean() * 100.0)}


_METRIC_KEYS = ("n", "median_abs_hs", "inside_spread_pct", "rmse_vp")


def _metric_block(m: Any, name: str) -> dict | None:
    if m is None:
        return None
    if not isinstance(m, dict):
        raise ArtifactError(f"{name}: must be an object")
    miss = [k for k in _METRIC_KEYS if k not in m]
    if miss:
        raise ArtifactError(f"{name}: missing {', '.join(miss)}")
    n = _finite(m.get("n"))
    if n is None or n < 0 or n != int(n):
        raise ArtifactError(f"{name}: n must be a non-negative integer")
    out = dict(m)
    out["n"] = int(n)
    for k in ("inside_spread_pct", "within_2hs_pct"):
        if k in m:
            out[k] = _pct(m[k], f"{name}.{k}")
    for k in ("median_abs_hs", "rmse_vp", "rmse_hs", "median_abs_vp"):
        v = _finite(m.get(k))
        if m.get(k) is not None and (v is None or v < 0):
            raise ArtifactError(f"{name}.{k}: must be finite and >= 0")
    return out


def scorecard_rows(obj: dict, session: str) -> list[dict]:
    """The scorecard's model rows, in the producer's out-of-sample ranking order."""
    name = f"surface/{session}/scorecard.json"
    models = obj.get("models")
    if not isinstance(models, list) or not models:
        raise ArtifactError(f"{name}: needs a non-empty 'models' list")
    rows: list[dict[str, Any]] = []
    order: dict[str, int] = {}
    for i, m in enumerate(models):
        if not isinstance(m, dict) or not m.get("model"):
            raise ArtifactError(f"{name}: every model row needs 'model'")
        met = m.get("metrics")
        if not isinstance(met, dict):
            raise ArtifactError(f"{name}: model {m['model']} has no metrics")
        oos = _metric_block(met.get("out_of_sample"), f"{name}: {m['model']}.out_of_sample")
        if oos is None:
            raise ArtifactError(f"{name}: model {m['model']} has no out_of_sample metrics")
        order[str(m["model"])] = i
        rows.append({
            "model": str(m["model"]), "description": m.get("description"),
            "n_params": m.get("n_params"), "n_in_sample": m.get("n_in_sample"),
            "oos": oos, "all": _metric_block(met.get("all"), f"{name}: {m['model']}.all"),
            "in_sample": _metric_block(met.get("in_sample"), f"{name}: {m['model']}.in_sample"),
            "ex_arb": _metric_block(met.get("excluding_arbitrage_flagged"),
                                    f"{name}: {m['model']}.excluding_arbitrage_flagged"),
            "by_expiry": met.get("by_expiry") if isinstance(met.get("by_expiry"), dict) else {},
            "by_tenor": met.get("by_tenor") if isinstance(met.get("by_tenor"), dict) else {},
            "dense": m.get("dense_grid") if isinstance(m.get("dense_grid"), dict) else {},
            "params": m.get("params") if isinstance(m.get("params"), dict) else {},
        })
    rank = [r.get("model") for r in obj.get("ranking_by_oos_median_abs_hs") or [] if isinstance(r, dict)]
    pos = {m: i for i, m in enumerate(rank)}
    rows.sort(key=lambda r: pos.get(r["model"], len(pos) + order[r["model"]]))
    return rows


def validate_surface_json(obj: dict, session: str) -> None:
    name = f"surface/{session}/surface.json"
    arb = obj.get("arbitrage")
    if arb is not None:
        if not isinstance(arb, dict):
            raise ArtifactError(f"{name}: arbitrage must be an object")
        for key in ("n_rows", "mid_violations", "executable_violations"):
            block = arb.get(key)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ArtifactError(f"{name}: arbitrage.{key} must map constraint -> count")
            for c, v in block.items():
                f = _finite(v)
                if f is None or f < 0 or f != int(f):
                    raise ArtifactError(f"{name}: arbitrage.{key}.{c} must be a count")
    for t in obj.get("term_structure") or []:
        if not isinstance(t, dict) or _finite(t.get("T_days")) is None:
            raise ArtifactError(f"{name}: every term_structure row needs a finite T_days")
        for k in ("atm_vol_essvi", "atm_vol_mid_quotes"):
            v = _finite(t.get(k))
            if v is not None and not 0.5 < v < 300:
                raise ArtifactError(f"{name}: term_structure {k} {v} is not in vol points")


def surface_fit_inputs(data_dir: Path, session: str) -> dict:
    """What a session's surface was fitted on (surface/<session>/surface.json):
    ``rule`` (fit_set.rule, which quotes were held out), ``source`` and
    ``quote_time`` (inputs). Empty when the file does not exist."""
    path = Path(data_dir) / "surface" / session / "surface.json"
    if not path.exists():
        return {}
    obj = _read_json(path, f"surface/{session}/surface.json")
    _check_session(obj, session, f"surface/{session}/surface.json")
    validate_surface_json(obj, session)
    fit_set: dict = obj["fit_set"] if isinstance(obj.get("fit_set"), dict) else {}
    inputs: dict = obj["inputs"] if isinstance(obj.get("inputs"), dict) else {}
    return {"rule": fit_set.get("rule"), "source": inputs.get("source"),
            "quote_time": inputs.get("quote_time")}


def validate_vix_replication(obj: dict, session: str) -> None:
    name = f"surface/{session}/vix_replication.json"
    idx = obj.get("indices")
    if not isinstance(idx, dict) or not idx:
        raise ArtifactError(f"{name}: needs an 'indices' object")
    for tick, row in idx.items():
        if not isinstance(row, dict):
            raise ArtifactError(f"{name}: indices.{tick} must be an object")
        rep = row.get("replicated")
        if rep is not None:
            v = _finite(rep)
            if v is None or not 1.0 < v < 200.0:
                raise ArtifactError(f"{name}: indices.{tick}.replicated {rep!r} is not in index points")
    for s in obj.get("strip") or []:
        if not isinstance(s, dict) or not s.get("expiry"):
            raise ArtifactError(f"{name}: every strip row needs an expiry")
        v = _finite(s.get("var_swap_vol"))
        if s.get("var_swap_vol") is not None and (v is None or not 0.5 < v < 300):
            raise ArtifactError(f"{name}: strip var_swap_vol {s.get('var_swap_vol')!r} is not in vol points")


def load_surface_index(data_dir: Path) -> list[dict] | None:
    """data_store/surface/index.json: one summary row per session (optional)."""
    p = Path(data_dir) / "surface" / "index.json"
    if not p.exists():
        return None
    obj = _read_json(p, "surface/index.json")
    rows = obj.get("sessions")
    if not isinstance(rows, list):
        raise ArtifactError("surface/index.json: needs a 'sessions' list")
    for r in rows:
        if not isinstance(r, dict) or not _DATE.match(str(r.get("session", ""))):
            raise ArtifactError("surface/index.json: every row needs a YYYY-MM-DD session")
    return sorted(rows, key=lambda r: r["session"])


def load_surface(data_dir: Path, session: str | None = None) -> dict:
    """The newest (or the given) session's surface artifacts.

    Required for state "ok": residuals.parquet. surface.json, scorecard.json and
    vix_replication.json are optional; each one present must validate and carry
    its directory's session.
    """
    data_dir = Path(data_dir)
    sessions = surface_sessions(data_dir)
    if session is None:
        session = sessions[-1] if sessions else None
    root = data_dir / "surface" / (session or "<session>")
    if session is None or not (root / "residuals.parquet").exists():
        return absent("surface", root / "residuals.parquet", root / "scorecard.json",
                      root / "surface.json", root / "vix_replication.json", data_dir=data_dir)
    out: dict = {"state": "ok", "session": session, "sessions": sessions, "root": str(root)}
    out["residuals"] = read_residuals(root / "residuals.parquet", session)
    out["models"] = list(out["residuals"].attrs["models"])
    for key, fname in (("surface", "surface.json"), ("scorecard", "scorecard.json"),
                       ("vix", "vix_replication.json")):
        p = root / fname
        out[key] = None
        if p.exists():
            obj = _read_json(p, f"surface/{session}/{fname}")
            _check_session(obj, session, f"surface/{session}/{fname}")
            out[key] = obj
    if out["surface"] is not None:
        validate_surface_json(out["surface"], session)
    sc = out["scorecard"]
    out["scorecard_rows"] = scorecard_rows(sc, session) if sc is not None else None
    if out["vix"] is not None:
        validate_vix_replication(out["vix"], session)
    out["index"] = load_surface_index(data_dir)
    return out


def load_surface_desk_inputs(data_dir: Path) -> list[dict]:
    """Every stored surface session's inputs to the desk-coordinate views
    (dashboard/desk_surface.py), oldest first: ``session``, ``surface``
    (surface.json, validated), ``vix`` (vix_replication.json, validated, or
    None) and ``quotes`` (the residual table's model, expiry and k columns, for
    each expiry's quoted ln(K/F) range). A session without surface.json is
    left out: it has no fitted surface to read."""
    data_dir = Path(data_dir)
    out = []
    for session in surface_sessions(data_dir):
        root = data_dir / "surface" / session
        if not (root / "surface.json").exists():
            continue
        obj = _read_json(root / "surface.json", f"surface/{session}/surface.json")
        _check_session(obj, session, f"surface/{session}/surface.json")
        validate_surface_json(obj, session)
        vix = None
        if (root / "vix_replication.json").exists():
            vix = _read_json(root / "vix_replication.json", f"surface/{session}/vix_replication.json")
            _check_session(vix, session, f"surface/{session}/vix_replication.json")
            validate_vix_replication(vix, session)
        name = f"surface/{session}/residuals.parquet"
        try:
            q = pd.read_parquet(root / "residuals.parquet", columns=["model", "expiry", "k"])
        except Exception as exc:
            raise ArtifactError(f"{name}: unreadable ({exc})") from exc
        if not np.isfinite(pd.to_numeric(q["k"], errors="coerce")).all():
            raise ArtifactError(f"{name}: k must be finite")
        q = q.assign(expiry=q["expiry"].astype(str).str[:10], model=q["model"].astype(str))
        out.append({"session": session, "surface": obj, "vix": vix, "quotes": q})
    return out


HESTON_PARAMS = ("kappa", "theta", "sigma", "rho", "v0")


def sigma_profile_refit(data_dir: Path, session: str | None) -> dict | None:
    """The widest-cap refit of a session's σ profile (calibrations/diagnostics/
    <session>.json sigma_profile.rows, largest sigma_cap): ``cap`` and its five
    Heston ``params``. None when the file or a profile is absent; a parameter
    that is not a finite number is an ArtifactError."""
    path = Path(data_dir) / "calibrations" / "diagnostics" / f"{session}.json"
    if not session or not path.exists():
        return None
    name = f"calibrations/diagnostics/{session}.json"
    obj = _read_json(path, name)
    _check_session(obj, session, name)
    rows = [r for r in ((obj.get("sigma_profile") or {}).get("rows") or []) if isinstance(r, dict)
            and _finite(r.get("sigma_cap")) is not None and isinstance(r.get("params"), dict)]
    if not rows:
        return None
    top = max(rows, key=lambda r: float(r["sigma_cap"]))
    params = {}
    for k in HESTON_PARAMS:
        v = _finite(top["params"].get(k))
        if v is None:
            raise ArtifactError(f"{name}: sigma_profile params.{k} is not a finite number")
        params[k] = v
    return {"cap": float(top["sigma_cap"]), "params": params, "file": name}


# ── calibration diagnostics: in the result pickle, or diagnostics/<s>.json ───
def calibration_diagnostics(cal: dict, data_dir: Path, session: str | None) -> dict:
    """Identifiability, multi-start, sigma profile and model comparison of the headline fit.

    The C23 calibrator stores ``identifiability`` and ``multistart`` in its
    result; scripts/calibration_diagnostics.py writes
    data_store/calibrations/diagnostics/<session>.json (identifiability,
    trf.multistart, sigma_profile, spx_only, joint_rough_heston). The pickle's
    identifiability wins when both exist; the file adds what the pickle lacks.
    """
    path = Path(data_dir) / "calibrations" / "diagnostics" / f"{session}.json"
    obj: dict = {}
    if session and path.exists():
        obj = _read_json(path, f"calibrations/diagnostics/{session}.json")
        _check_session(obj, session, f"calibrations/diagnostics/{session}.json")
    ident = cal.get("identifiability") if isinstance(cal.get("identifiability"), dict) else None
    multi = cal.get("multistart") if isinstance(cal.get("multistart"), dict) else None
    src = "result pickle" if ident is not None else None
    if ident is None and obj.get("identifiability") is not None:
        ident, src = obj.get("identifiability"), f"diagnostics/{session}.json"
    if multi is None:
        m = obj.get("multistart") or (obj.get("trf") or {}).get("multistart")
        multi = m if isinstance(m, dict) else None
    if ident is None:
        return absent("identifiability", path, data_dir=data_dir)
    if not isinstance(ident, dict):
        raise ArtifactError("identifiability: must be an object")
    if "error" in ident:
        raise ArtifactError(f"identifiability: the calibrator recorded an error ({ident['error']})")
    names = ident.get("param_names") or []
    sv = ident.get("rel_jacobian_singular_values") or []
    if not names or not sv:
        raise ArtifactError("identifiability: needs param_names and rel_jacobian_singular_values")
    vals = [_finite(v) for v in sv]
    if any(v is None or v < 0 for v in vals):
        raise ArtifactError("identifiability: singular values must be finite and >= 0")
    if list(sv) != sorted(sv, reverse=True):
        raise ArtifactError("identifiability: singular values must be ordered largest first")
    prof = obj.get("sigma_profile")
    rows = []
    if isinstance(prof, dict):
        for r in prof.get("rows") or []:
            ok = isinstance(r, dict) and _finite(r.get("sigma_cap")) is not None
            if not ok or _finite(r.get("loss")) is None:
                raise ArtifactError(f"calibrations/diagnostics/{session}.json: "
                                    "sigma_profile rows need sigma_cap and loss")
            for k in ("vvix_model", "vvix_market"):        # printed, and differenced, by the page
                if r.get(k) is not None and _finite(r.get(k)) is None:
                    raise ArtifactError(f"calibrations/diagnostics/{session}.json: "
                                        f"sigma_profile.{k} is not a finite number ({r.get(k)!r})")
            rows.append(r)
    spx_only = obj.get("spx_only") if isinstance(obj.get("spx_only"), dict) else {}
    return {"state": "ok", "source": src, "ident": ident, "multistart": multi,
            "sigma_profile": rows,
            "kappa_max": prof.get("kappa_max") if isinstance(prof, dict) else None,
            "spx_only": spx_only,
            "joint_rough": (obj.get("joint_rough_heston")
                            if isinstance(obj.get("joint_rough_heston"), dict) else None),
            "trf": obj.get("trf") if isinstance(obj.get("trf"), dict) else None,
            "file": str(path.relative_to(Path(data_dir).parent)) if obj else None}


# The per-session numbers of diagnostics/summary.json the pages read: the
# production (σ ≤ 2) joint Heston fit, the joint and SPX-only rough Heston
# fits, the free-σ refit and the model VVIX. Vol points unless named.
DIAG_SUMMARY_FIELDS = (
    "sigma_cap2_oos", "joint_heston_vix_index", "joint_rough_oos", "joint_rough_in", "joint_rough_H",
    "joint_rough_vix_index", "joint_rough_lifting_error_estimate_max_vp", "heston_spx_oos", "rough_oos",
    "rough_H", "sigma_free", "kappa_free", "sigma_free_oos", "sigma_free_loss_rel", "vvix_market",
    "vvix_model_cap2",
    "rel_jtj_condition_number", "trf_seconds", "trf_evals", "de_serial_seconds", "de_serial_evals",
    "trf_loss_rel_diff")


def load_diagnostics_summary(data_dir: Path) -> dict:
    """data_store/calibrations/diagnostics/summary.json: one row per diagnosed session.

    Written by scripts/calibration_diagnostics.py (the bot adds a row per
    accepted fit). Every field in DIAG_SUMMARY_FIELDS a row carries must be a
    finite number or null; H must lie in [0, 1/2]; sessions are unique.
    """
    path = Path(data_dir) / "calibrations" / "diagnostics" / "summary.json"
    if not path.exists():
        return absent("identifiability", path, data_dir=data_dir)
    name = "calibrations/diagnostics/summary.json"
    obj = _read_json(path, name)
    rows = obj.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ArtifactError(f"{name}: needs a non-empty 'rows' list")
    out = []
    for r in rows:
        if not isinstance(r, dict) or not _DATE.match(str(r.get("session", ""))):
            raise ArtifactError(f"{name}: every row needs a YYYY-MM-DD session")
        row: dict[str, Any] = {"session": r["session"],
                               "on_bound": list(r.get("on_bound") or []),
                               "joint_rough_on_bound": list(r.get("joint_rough_on_bound") or []),
                               "relative_standard_errors": r.get("relative_standard_errors") or {}}
        for k in DIAG_SUMMARY_FIELDS:
            v = _finite(r.get(k))
            if r.get(k) is not None and v is None:
                raise ArtifactError(f"{name}: {r['session']}.{k} is not a finite number ({r.get(k)!r})")
            row[k] = v
        for k in ("joint_rough_H", "rough_H"):
            if row[k] is not None and not 0.0 <= row[k] <= 0.5:
                raise ArtifactError(f"{name}: {r['session']}.{k} = {row[k]} is outside [0, 1/2]")
        out.append(row)
    sessions = [r["session"] for r in out]
    if len(set(sessions)) != len(sessions):
        raise ArtifactError(f"{name}: duplicate sessions")
    return {"state": "ok", "rows": sorted(out, key=lambda r: r["session"]),
            "generated_utc": obj.get("generated_utc"),
            "file": str(path.relative_to(Path(data_dir).parent))}


# The σ-profile fields the SPX/VIX trade-off reads, per refit row: the cap, the
# refit's out-of-sample SPX IV RMSE (vol pts) and the model VVIX (30-day vol of
# VIX at the refit) against the ^VVIX close (index pts).
SIGMA_PROFILE_FIELDS = ("sigma_cap", "spx_oos_iv_rmse", "vvix_model", "vvix_market", "vvix_error",
                        "feller_margin", "loss")


def load_sigma_profiles(data_dir: Path) -> dict:
    """The σ profile of every diagnosed session (calibrations/diagnostics/<session>.json).

    scripts/calibration_diagnostics.py refits the joint Heston fit under each
    cap on σ (config.SIGMA_PROFILE_CAPS, κ ≤ PROFILE_KAPPA_MAX) and records each
    refit's parameters, out-of-sample SPX RMSE and model VVIX; the bot writes
    one file per accepted fit. Returns ``{"state": "ok", "sessions": [...]}``
    with, per session (oldest first): ``session``, ``production_cap`` (the
    production box's σ ceiling the file records), ``stored_fit`` (the published
    fit's file name and its model VVIX against the market) and ``rows`` sorted by
    cap, each with SIGMA_PROFILE_FIELDS (None where the file has no value),
    ``sigma`` / ``kappa`` and ``sigma_at_cap``. A field present but not a finite
    number, or a file of another session, is an ArtifactError.
    """
    root = Path(data_dir) / "calibrations" / "diagnostics"
    files = sorted(f for f in root.glob("*.json") if _DATE.match(f.stem)) if root.is_dir() else []
    if not files:
        return absent("identifiability", root / "<session>.json", data_dir=data_dir)
    out = []
    for f in files:
        name = f"calibrations/diagnostics/{f.name}"
        obj = _read_json(f, name)
        _check_session(obj, f.stem, name)
        prof = obj.get("sigma_profile")
        if not isinstance(prof, dict) or not isinstance(prof.get("rows"), list):
            continue                       # a file without a σ profile has nothing to add here
        rows = []
        for r in prof["rows"]:
            if not isinstance(r, dict):
                raise ArtifactError(f"{name}: sigma_profile rows must be objects")
            row: dict[str, Any] = {}
            for k in SIGMA_PROFILE_FIELDS:
                v = _finite(r.get(k))
                if r.get(k) is not None and v is None:
                    raise ArtifactError(f"{name}: sigma_profile.{k} is not a finite number ({r.get(k)!r})")
                row[k] = v
            params: dict = r["params"] if isinstance(r.get("params"), dict) else {}
            row["sigma"], row["kappa"] = _finite(params.get("sigma")), _finite(params.get("kappa"))
            row["sigma_at_cap"] = bool(r.get("sigma_at_cap"))
            if row["sigma_cap"] is None:
                raise ArtifactError(f"{name}: every sigma_profile row needs sigma_cap")
            rows.append(row)
        box: dict = prof["production_box"] if isinstance(prof.get("production_box"), dict) else {}
        sig_box: list = box["sigma"] if isinstance(box.get("sigma"), list) else []
        sf: dict = obj["stored_fit"] if isinstance(obj.get("stored_fit"), dict) else {}
        stored = {"file": sf.get("file")}
        for k in ("vvix_model", "vvix_market"):
            v = _finite(sf.get(k))
            if sf.get(k) is not None and v is None:
                raise ArtifactError(f"{name}: stored_fit.{k} is not a finite number ({sf.get(k)!r})")
            stored[k] = v
        out.append({"session": f.stem, "production_cap": _finite(sig_box[1]) if len(sig_box) == 2 else None,
                    "kappa_max": _finite(prof.get("kappa_max")), "stored_fit": stored,
                    "rows": sorted(rows, key=lambda r: r["sigma_cap"]), "file": name})
    if not out:
        return absent("identifiability", root / "<session>.json", data_dir=data_dir)
    return {"state": "ok", "sessions": out}


def surface_oos_by_session(data_dir: Path) -> list[dict]:
    """Every stored session's out-of-sample scorecard block per model.

    One row per (session, model) with that session's scorecard.json
    out_of_sample metrics (n, median_abs_hs, inside_spread_pct, rmse_vp) and
    the dense-grid arbitrage counts; sessions without a scorecard are skipped.
    """
    out: list[dict[str, Any]] = []
    for s in surface_sessions(data_dir):
        p = Path(data_dir) / "surface" / s / "scorecard.json"
        if not p.exists():
            continue
        obj = _read_json(p, f"surface/{s}/scorecard.json")
        _check_session(obj, s, f"surface/{s}/scorecard.json")
        out.extend({"session": s, "model": r["model"],
                    **{k: r["oos"].get(k) for k in ("n", "median_abs_hs", "inside_spread_pct", "rmse_vp")},
                    "butterfly": (r["dense"] or {}).get("butterfly_violations"),
                    "calendar": (r["dense"] or {}).get("calendar_violations")}
                   for r in scorecard_rows(obj, s))
    return out


# ── backtest inference: data_store/backtest/inference/ ───────────────────────
INFERENCE_SCHEMA = "inference_summary/"


def load_inference(data_dir: Path) -> dict:
    """Bootstrap CIs, null tests, multiple testing, benchmarks and forecast evaluation.

    summary.json (schema "inference_summary/1") is required for the state "ok";
    benchmarks.json and forecast_eval.json are optional. The replicate-level
    parquet files beside them are the producer's audit trail and are not read.
    """
    root = Path(data_dir) / "backtest" / "inference"
    spath = root / "summary.json"
    if not spath.exists():
        return absent("inference", spath, root / "benchmarks.json", root / "forecast_eval.json",
                      data_dir=data_dir)
    summ = _read_json(spath, "inference/summary.json")
    if not str(summ.get("schema", "")).startswith(INFERENCE_SCHEMA):
        raise ArtifactError(f"inference/summary.json: schema {summ.get('schema')!r} "
                            f"is not {INFERENCE_SCHEMA}*")
    boot = summ.get("bootstrap")
    if not isinstance(boot, dict) or not boot:
        raise ArtifactError("inference/summary.json: needs a 'bootstrap' object")
    for series, stats in boot.items():
        if not isinstance(stats, dict):
            raise ArtifactError(f"inference/summary.json: bootstrap.{series} must be an object")
        for stat, b in stats.items():
            if not isinstance(b, dict) or "ci95" not in b:
                continue
            ci = b["ci95"]
            if (not isinstance(ci, (list, tuple)) or len(ci) != 2 or _finite(ci[0]) is None
                    or _finite(ci[1]) is None or float(ci[0]) > float(ci[1])):
                raise ArtifactError(f"inference/summary.json: bootstrap.{series}.{stat}.ci95 "
                                    "is not an interval")
    for sleeve, tests in ((summ.get("nulls") or {}).get("sleeves") or {}).items():
        for kind, by_stat in (tests or {}).items():
            if not isinstance(by_stat, dict):
                continue
            for stat, t in by_stat.items():
                if isinstance(t, dict):
                    for k in ("p_value", "p_holm_6_sleeves"):
                        if k in t:
                            _prob(t[k], f"inference/summary.json: nulls.{sleeve}.{kind}.{stat}.{k}")
    out: dict = {"state": "ok", "root": str(root), "summary": summ}
    for key, fname in (("benchmarks", "benchmarks.json"), ("forecast", "forecast_eval.json")):
        p = root / fname
        out[key] = _read_json(p, f"inference/{fname}") if p.exists() else None
    return out


def bootstrap_ci(inf: dict, series: str, statistic: str = "sharpe") -> dict | None:
    """{point, lo, hi, level, reps, method, excludes_zero} for one series/statistic, or None."""
    if not inf or inf.get("state") != "ok":
        return None
    summ = inf.get("summary") or {}
    b = ((summ.get("bootstrap") or {}).get(series) or {}).get(statistic)
    if not isinstance(b, dict) or "ci95" not in b:
        return None
    sett = summ.get("bootstrap_settings") or {}
    lo, hi = (_finite(v) for v in b["ci95"])
    return {"point": _finite(b.get("point")), "lo": lo, "hi": hi,
            "level": _finite(sett.get("level")) or 0.95, "reps": sett.get("reps"),
            "method": sett.get("method"), "excludes_zero": bool(b.get("excludes_zero")),
            "se": _finite(b.get("se"))}


# ── forward ledger: data_store/forward/ledger_<session>.json ────────────────
FORWARD_SCHEMA = "forward_ledger/1"


def _canonical_sha256(obj: Any) -> str:
    """SHA-256 of the canonical serialisation the ledger hashes are taken over
    (sorted keys, no whitespace; the files already hold 10-significant-digit floats)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_forward_ledger(data_dir: Path) -> dict:
    """Every ledger file, checked as a hash chain: each file names the previous
    session and the SHA-256 of its canonical JSON. A broken link is an
    ArtifactError (the register is append-only; an edit must not publish)."""
    root = Path(data_dir) / "forward"
    files = sorted(root.glob("ledger_*.json")) if root.is_dir() else []
    if not files:
        return absent("forward", root / "ledger_<session>.json", data_dir=data_dir)
    ledgers = []
    for f in files:
        led = _read_json(f, f"forward/{f.name}")
        sess = f.stem[len("ledger_"):]
        if led.get("schema") != FORWARD_SCHEMA:
            raise ArtifactError(f"forward/{f.name}: schema {led.get('schema')!r} != {FORWARD_SCHEMA}")
        if str(led.get("session")) != sess:
            raise ArtifactError(f"forward/{f.name}: session {led.get('session')} != file name")
        ledgers.append(led)
    for a, b in itertools.pairwise(ledgers):
        prev = b.get("previous") or {}
        if prev.get("session") != a["session"]:
            raise ArtifactError(f"forward/ledger_{b['session']}.json: previous session "
                                f"{prev.get('session')} != {a['session']}")
        if prev.get("sha256") != _canonical_sha256(a):
            raise ArtifactError(f"forward/ledger_{b['session']}.json: the hash of ledger_{a['session']}.json "
                                "does not match (a registered ledger was changed)")
    if ledgers[0].get("previous") is not None:
        raise ArtifactError(f"forward/ledger_{ledgers[0]['session']}.json: the first ledger names a "
                            f"previous one ({ledgers[0]['previous'].get('session')}) that is not stored")
    return {"state": "ok", "n": len(ledgers), "first": ledgers[0]["session"], "last": ledgers[-1]["session"],
            "latest": ledgers[-1], "sessions": [x["session"] for x in ledgers]}


# ── risk scenarios: data_store/greeks/scenario_grid.parquet ──────────────────
def load_scenario_grid(data_dir: Path) -> dict:
    """Spot × vol full-revaluation P&L grid of a reference book.

    Columns: spot_shock_pct, vol_shock_vp, pnl [, dynamics]; parquet metadata
    b"scenario_provenance" (book, as_of_date, units).
    """
    path = Path(data_dir) / "greeks" / "scenario_grid.parquet"
    if not path.exists():
        return absent("scenario", path, data_dir=data_dir)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise ArtifactError(f"scenario_grid.parquet: unreadable ({exc})") from exc
    _require(df, {"spot_shock_pct", "vol_shock_vp", "pnl"}, "greeks/scenario_grid")
    if not np.isfinite(pd.to_numeric(df["pnl"], errors="coerce")).all():
        raise ArtifactError("greeks/scenario_grid: pnl must be finite")
    keys = ["spot_shock_pct", "vol_shock_vp"] + (["dynamics"] if "dynamics" in df.columns else [])
    if df.duplicated(keys).any():
        raise ArtifactError(f"greeks/scenario_grid: duplicate ({', '.join(keys)}) rows")
    prov: dict = {}
    try:
        import pyarrow.parquet as pq
        raw = (pq.read_schema(path).metadata or {}).get(b"scenario_provenance")
        prov = json.loads(raw.decode()) if raw else {}
    except Exception as exc:
        raise ArtifactError(f"greeks/scenario_grid: unreadable provenance ({exc})") from exc
    return {"state": "ok", "grid": df, "provenance": prov}
