#!/usr/bin/env python3
"""
reproduce.py: rebuild the committed artifacts from the committed captures and
diff them against what is committed (`make reproduce`).

    python scripts/ci/reproduce.py                 # fast set, ~3-5 min on 4 cores
    python scripts/ci/reproduce.py --full          # + beta grid, C22 waterfall, S3 null
    python scripts/ci/reproduce.py --only greeks,c7

The pattern is the Open Source Risk Engine's (ORE) Output vs ExpectedOutput:
every step runs the same command the README lists for regenerating an
artifact, inside a throwaway copy of the repository (the tracked files, i.e.
the code, the captures under data_store/ and the committed artifacts), so the
real data_store/ is never written. The regenerated files are then compared
with the committed ones under a per-artifact tolerance, and a report
(<out>/report.md, <out>/report.json) lists the largest deviation found for
every artifact. The exit status is 1 if any artifact is outside tolerance.

Tolerances (TOLERANCES below) are relative to each column's (or value's)
scale. The backtest, Greeks surface and C7 study use 1e-10, the tolerance
the suite already applies to CPU-dependent last bits
(tests/artifact_tolerance.py). Calibrations use 1e-6: differential evolution
re-runs its whole trajectory, and a last-bit difference in one loss
evaluation on another CPU can move the optimum by ~1e-8 relative. The
committed refit is bit for bit on the machine that wrote it (README).

Determinism: the steps run with PYTHONHASHSEED=0 and one BLAS/OpenMP thread.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = "data_store"
# The pickled results hold project objects (TradeRecord, ...): unpickling them
# here needs the package importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keys of a calibration pickle that legitimately differ run to run.
VOLATILE_KEYS = {"fit_time", "run_utc"}
# save() adds the pre-C22 'losses' alias; the daily pipeline's pickles omit it.
OPTIONAL_KEYS = {"losses"}
# Provenance in a calibration pickle: how T, carry and anchors were chosen
# ("methods") and counts describing the input chain ("chain_diagnostics").
# They are compared and every difference is reported, but only the model's
# outputs can fail a check: the committed pickles predate later provenance
# fields, and a fit made from the live download records the raw download's
# line counts where a refit from the stored chain (two-sided lines only)
# records its own.
PROVENANCE_KEYS = {"methods", "chain_diagnostics"}

TOLERANCES = {
    "backtest": 1e-10,
    "greeks": 1e-10,
    "c7": 1e-10,
    "calibration": 1e-6,
    "beta_grid": 1e-9,      # CSV written at 10 significant digits
    "bridge": 1e-9,
    "s3_null": 0.0,         # a 200-path p-value (1 + k)/201: exact
}

ENV = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "BROWSER": "true", "MPLBACKEND": "Agg"}

_BACKTEST = ("from joint_vol_calibration.backtest.run_backtest import _load_data, load_pdv_vol\n"
             "from joint_vol_calibration.backtest.backtest_engine import BacktestEngine\n"
             "spx, vix = _load_data()\n"
             "eng = BacktestEngine(pdv_vol_series=load_pdv_vol())\n"
             "eng.run(spx, vix)\n"
             "eng.compute_metrics()\n"
             "eng.save()\n")
# A committed fit is refitted on the VIX targets it recorded (C23 restated the
# database's VIX family as Cboe's closes; joint_calibrator.recorded_vix_targets).
_REFIT = ("import pickle\n"
          "from joint_vol_calibration.calibration.joint_calibrator import (\n"
          "    JointCalibrator, recorded_vix_targets)\n"
          "from joint_vol_calibration.config import DATA_DIR\n"
          "with open(DATA_DIR / 'calibrations' / 'joint_cal_{session}.pkl', 'rb') as f:\n"
          "    ref = pickle.load(f)\n"
          "c = JointCalibrator(as_of_date={session!r}, vix_targets=recorded_vix_targets(ref))\n"
          "c.calibrate(verbose=False)\n"
          "c.save()\n")


# ── comparison ───────────────────────────────────────────────────────────────

@dataclass
class Check:
    artifact: str
    tolerance: float
    deviation: float = 0.0          # largest relative deviation found
    where: str = ""
    problems: list[str] = field(default_factory=list)   # structural mismatches
    notes: list[str] = field(default_factory=list)      # provenance differences (reported, not failed)

    @property
    def ok(self) -> bool:
        return not self.problems and self.deviation <= self.tolerance

    def note(self, dev: float, where: str) -> None:
        if not (dev <= self.deviation):     # NaN-safe: a NaN deviation is recorded
            self.deviation, self.where = dev, where


def rel_dev(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """max |a - b| / max |b| over finite b (scale of the expected column);
    NaN must meet NaN, an all-zero column must match exactly. Returns
    (deviation, index of the worst element)."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return math.inf, -1
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        return math.inf, int(np.argmax(nan_a != nan_b))
    fin = ~nan_b
    if not fin.any():
        return 0.0, -1
    diff = np.abs(a[fin] - b[fin])
    if not np.all(np.isfinite(diff)):
        bad = ~np.isfinite(diff)
        same = (a[fin] == b[fin])          # +-inf on both sides is a match
        if (bad & ~same).any():
            return math.inf, int(np.flatnonzero(fin)[np.argmax(bad & ~same)])
        diff = np.where(bad, 0.0, diff)
    scale = float(np.max(np.abs(np.where(np.isfinite(b[fin]), b[fin], 0.0))))
    worst = int(np.argmax(diff))
    if scale == 0.0:
        return (0.0 if diff[worst] == 0.0 else math.inf), int(np.flatnonzero(fin)[worst])
    return float(diff[worst] / scale), int(np.flatnonzero(fin)[worst])


def compare_frames(got: pd.DataFrame, exp: pd.DataFrame, check: Check, prefix: str = "") -> None:
    if list(got.columns) != list(exp.columns):
        check.problems.append(f"{prefix}columns differ: {sorted(set(got.columns) ^ set(exp.columns))}")
        return
    if len(got) != len(exp) or not got.index.equals(exp.index):
        check.problems.append(f"{prefix}index differs ({len(got)} vs {len(exp)} rows)")
        return
    for col in exp.columns:
        g, e = got[col], exp[col]
        if pd.api.types.is_numeric_dtype(e) and not pd.api.types.is_bool_dtype(e):
            dev, i = rel_dev(g.to_numpy(dtype=float, na_value=np.nan),
                             e.to_numpy(dtype=float, na_value=np.nan))
            check.note(dev, f"{prefix}{col}[{i}]")
        elif not g.equals(e):
            check.problems.append(f"{prefix}column {col!r} differs")


def compare_objects(got: Any, exp: Any, check: Check, path: str = "",
                    skip: Iterable[str] = (), optional: Iterable[str] = ()) -> None:
    """Recursive comparison of pickled results: numbers by relative deviation,
    frames column by column, everything else exactly. ``skip`` and
    ``optional`` hold dotted key paths ('fit_time', 'metrics.s3_null_p_value'):
    skipped keys are not compared, optional ones may be absent on either side."""
    skip, optional = frozenset(skip), frozenset(optional)

    def sub(g: Any, e: Any, p: str) -> None:
        compare_objects(g, e, check, p, skip, optional)

    if is_dataclass(exp) and not isinstance(exp, type):
        sub(asdict(got) if is_dataclass(got) and not isinstance(got, type) else got, asdict(exp), path)
        return
    if isinstance(exp, pd.DataFrame):
        if not isinstance(got, pd.DataFrame):
            check.problems.append(f"{path}: expected a DataFrame")
        else:
            compare_frames(got, exp, check, prefix=f"{path}.")
        return
    if isinstance(exp, pd.Series):
        sub(got.to_frame() if isinstance(got, pd.Series) else got, exp.to_frame(), path)
        return
    if isinstance(exp, dict):
        if not isinstance(got, dict):
            check.problems.append(f"{path}: expected a dict")
            return

        def kp(k: Any) -> str:
            return f"{path}.{k}" if path else str(k)
        missing = [k for k in exp if k not in got and kp(k) not in skip | optional]
        extra = [k for k in got if k not in exp and kp(k) not in skip | optional]
        if missing or extra:
            check.problems.append(f"{path or 'top level'}: keys missing {sorted(map(str, missing))} "
                                  f"extra {sorted(map(str, extra))}")
        for k in exp:
            if kp(k) in skip or k not in got:
                continue
            sub(got[k], exp[k], kp(k))
        return
    if isinstance(exp, (list, tuple)):
        if not isinstance(got, (list, tuple)) or len(got) != len(exp):
            check.problems.append(f"{path}: length {len(got) if hasattr(got, '__len__') else '?'} "
                                  f"vs {len(exp)}")
            return
        if exp and all(isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in exp):
            dev, i = rel_dev(np.asarray(got, float), np.asarray(exp, float))
            check.note(dev, f"{path}[{i}]")
            return
        for i, (g, e) in enumerate(zip(got, exp)):
            sub(g, e, f"{path}[{i}]")
        return
    if isinstance(exp, np.ndarray) and exp.dtype.kind in "fiuc":
        dev, i = rel_dev(np.asarray(got), exp)
        check.note(dev, f"{path}[{i}]")
        return
    if isinstance(exp, (float, np.floating)) or (isinstance(exp, (int, np.integer))
                                                 and not isinstance(exp, bool)
                                                 and isinstance(got, (float, np.floating))):
        dev, _ = rel_dev(np.array([got], dtype=float), np.array([exp], dtype=float))
        check.note(dev, path)
        return
    if hasattr(exp, "__dict__") and not isinstance(exp, type):
        sub(vars(got) if hasattr(got, "__dict__") else got, vars(exp), path)
        return
    try:
        equal = bool(got == exp)
    except (ValueError, TypeError):
        equal = False
    if not equal:
        check.problems.append(f"{path}: {got!r} != {exp!r}")


def _read(path: Path) -> Any:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _at(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj[part]
    return obj


def compare_file(got_path: Path, exp_path: Path, artifact: str, tolerance: float,
                 skip: Iterable[str] = (), optional: Iterable[str] = (),
                 only: Iterable[str] = (), provenance: Iterable[str] = ()) -> Check:
    """``only``: compare just these dotted key paths of a pickled dict.
    ``provenance``: top-level keys of a pickled dict that describe inputs and
    methods (PROVENANCE_KEYS): compared, but a difference there is reported
    as a note, not a failure."""
    check = Check(artifact, tolerance)
    if not got_path.exists():
        check.problems.append("not regenerated")
        return check
    got, exp = _read(got_path), _read(exp_path)
    meta = [k for k in provenance if isinstance(got, dict) and isinstance(exp, dict)]
    if meta:
        side = Check(artifact, tolerance)
        for k in meta:
            if k in got and k in exp:
                compare_objects(got[k], exp[k], side, k)
            elif (k in got) != (k in exp):
                side.problems.append(f"{k}: present on one side only")
        check.notes.extend(side.problems)
        got = {k: v for k, v in got.items() if k not in meta}
        exp = {k: v for k, v in exp.items() if k not in meta}
    if only:
        for dotted in only:
            try:
                compare_objects(_at(got, dotted), _at(exp, dotted), check, dotted)
            except (KeyError, TypeError):
                check.problems.append(f"{dotted}: missing")
        return check
    if isinstance(exp, pd.DataFrame):
        compare_frames(got, exp, check)
    else:
        compare_objects(got, exp, check, skip=skip, optional=optional)
    return check


# ── steps ────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    name: str
    argv: list[str]
    outputs: list[tuple[str, str, dict[str, Any]]]     # (path, tolerance key, compare kwargs)
    full_only: bool = False


def v2_sessions(root: Path) -> list[str]:
    """Sessions of the committed model_version 2 Heston fits."""
    out = []
    for p in sorted((root / DS / "calibrations").glob("joint_cal_*.pkl")):
        with open(p, "rb") as fh:
            if pickle.load(fh).get("model_version") == 2:
                out.append(p.stem.removeprefix("joint_cal_"))
    return out


def build_steps(root: Path) -> list[Step]:
    py = sys.executable
    bt = f"{DS}/backtest"
    steps = [
        Step("backtest", [py, "-c", _BACKTEST],
             [(f"{bt}/full_results.parquet", "backtest", {}),
              # the 200-path S3 null test is the --full step "s3_null"
              (f"{bt}/full_results.pkl", "backtest", {"skip": {"metrics.s3_null_p_value"}})]),
        # input: the COMMITTED headline fit (the refit step may be rewriting the copy)
        Step("greeks", [py, "-m", "joint_vol_calibration.greeks.risk_monitor", "--calibration",
                        str(root / DS / "calibrations" / "joint_cal_2026-10-16.pkl")],
             [(f"{DS}/greeks/greeks_surface.parquet", "greeks", {})]),
        Step("c7", [py, "-m", "joint_vol_calibration.backtest.run_backtest", "--c7"],
             [(f"{bt}/delta_hedge_2020.parquet", "c7", {})]),
    ]
    steps.extend(Step(f"calibration-{s}", [py, "-c", _REFIT.format(session=s)],
                      [(f"{DS}/calibrations/joint_cal_{s}.pkl", "calibration",
                        {"skip": VOLATILE_KEYS, "optional": OPTIONAL_KEYS, "provenance": PROVENANCE_KEYS})])
                 for s in v2_sessions(root))
    steps += [
        Step("beta_grid", [py, "-m", "joint_vol_calibration.backtest.run_backtest", "--beta-sensitivity",
                           "--csv", f"{bt}/beta_sensitivity.csv"],
             [(f"{bt}/beta_sensitivity.csv", "beta_grid", {})], full_only=True),
        Step("bridge", [py, "-m", "joint_vol_calibration.backtest.c22_bridge",
                        "--csv", f"{bt}/c22_bridge.csv"],
             [(f"{bt}/c22_bridge.csv", "bridge", {})], full_only=True),
        # C23 data-correction waterfall: 4 backtests, about 20 s
        Step("c23_bridge", [py, "-m", "joint_vol_calibration.backtest.c23_bridge",
                            "--csv", f"{bt}/c23_bridge.csv"],
             [(f"{bt}/c23_bridge.csv", "bridge", {})]),
        Step("s3_null", [py, "-m", "joint_vol_calibration.backtest.run_backtest", "--no-wf"],
             [(f"{bt}/full_results.pkl", "s3_null", {"only": ["metrics.s3_null_p_value"]})],
             full_only=True),
    ]
    return steps


def copy_tree(src: Path, dst: Path) -> None:
    """The tracked files of the checkout (code, captures, committed artifacts)."""
    files = subprocess.run(["git", "ls-files", "-z"], cwd=src, capture_output=True, check=True).stdout
    for rel in filter(None, files.decode().split("\0")):
        s = src / rel
        if s.is_file():
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)


def run_step(step: Step, tree: Path, log_dir: Path) -> tuple[int, float]:
    """Run one step in the scratch tree. Its outputs are deleted first, so a
    step that exits 0 without writing them fails as "not regenerated" instead
    of comparing the committed copy with itself."""
    for rel, _, _ in step.outputs:
        (tree / rel).unlink(missing_ok=True)
    t0 = time.time()
    env = {**os.environ, **ENV, "PYTHONPATH": str(tree)}
    with open(log_dir / f"{step.name}.log", "w") as log:
        r = subprocess.run(step.argv, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return r.returncode, time.time() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / ".reproduce")
    ap.add_argument("--full", action="store_true", help="also the beta grid, C22 waterfall and S3 null test")
    ap.add_argument("--only", help="comma-separated step names (prefix match, e.g. calibration)")
    ap.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree")
    args = ap.parse_args(argv)

    steps = [s for s in build_steps(ROOT) if args.full or not s.full_only]
    if args.only:
        want = [w.strip() for w in args.only.split(",") if w.strip()]
        steps = [s for s in steps if any(s.name.startswith(w) for w in want)]
    out = args.out.resolve()
    tree, logs = out / "tree", out / "logs"
    if tree.exists():
        shutil.rmtree(tree)
    logs.mkdir(parents=True, exist_ok=True)
    # The default output (.reproduce/) is inside the checkout: keep it out of git status.
    if out.is_relative_to(ROOT.resolve()) and not (out / ".gitignore").exists():
        (out / ".gitignore").write_text("*\n")
    print(f"copying the tracked files to {tree} ...", flush=True)
    copy_tree(ROOT, tree)

    # the S3 null step rewrites full_results.pkl; run it after the backtest check
    ordered = [s for s in steps if s.name != "s3_null"]
    late = [s for s in steps if s.name == "s3_null"]
    checks: list[Check] = []
    timings: dict[str, float] = {}

    def do(step: Step) -> list[Check]:
        code, secs = run_step(step, tree, logs)
        timings[step.name] = secs
        print(f"  {step.name:<26} exit {code}  {secs:6.1f} s", flush=True)
        if code != 0:
            c = Check(step.name, 0.0)
            c.problems.append(f"command failed (exit {code}); see logs/{step.name}.log")
            return [c]
        out = []
        for rel, tol, kw in step.outputs:
            try:
                out.append(compare_file(tree / rel, ROOT / rel, f"{step.name}: {rel}", TOLERANCES[tol], **kw))
            except Exception as exc:          # a broken artifact is a failed check, not a crash  # noqa: BLE001
                c = Check(f"{step.name}: {rel}", TOLERANCES[tol])
                c.problems.append(f"comparison failed: {exc!r}")
                out.append(c)
        return out

    print(f"running {len(ordered) + len(late)} step(s) with {args.jobs} job(s) ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for res in pool.map(do, ordered):
            checks.extend(res)
    for step in late:
        checks.extend(do(step))

    report(checks, timings, out)
    if not args.keep:
        shutil.rmtree(tree, ignore_errors=True)
    bad = [c for c in checks if not c.ok]
    print(f"\n{len(checks) - len(bad)}/{len(checks)} artifacts reproduce within tolerance"
          f" (report: {out / 'report.md'})")
    return 1 if bad else 0


def report(checks: list[Check], timings: dict[str, float], out: Path) -> None:
    when = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    header = (f"Commit `{_git('rev-parse', '--short', 'HEAD')}`, {when}, Python {sys.version.split()[0]}, "
              f"numpy {np.__version__}, pandas {pd.__version__}.")
    lines = ["# Reproduction report", "", header, "",
             "| Artifact | Largest relative deviation | Where | Tolerance | Result |",
             "|---|---|---|---|---|"]
    for c in checks:
        res = "ok" if c.ok else ("FAIL: " + "; ".join(c.problems) if c.problems else "FAIL")
        if c.notes:
            text = "; ".join(n.replace("|", "/") for n in c.notes)
            res += f" (provenance differs: {text[:240]}{'...' if len(text) > 240 else ''}; see report.json)"
        lines.append(f"| {c.artifact} | {c.deviation:.2e} | {c.where or '-'} | {c.tolerance:.0e} | {res} |")
    lines += ["", "Step times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items())]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(
        [{"artifact": c.artifact, "deviation": c.deviation, "where": c.where, "tolerance": c.tolerance,
          "ok": c.ok, "problems": c.problems, "notes": c.notes} for c in checks], indent=1) + "\n")
    print("\n".join(lines[4:]))


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


if __name__ == "__main__":
    sys.exit(main())
