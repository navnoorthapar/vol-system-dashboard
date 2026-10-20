#!/usr/bin/env python3
"""
mutation_check.py: do the property tests catch plausible bugs in the pricer
and the implied-vol solver? (`make mutation`)

Each mutant below is a one-line change of the kind a refactor could
introduce (a dropped compensator, a sign, a constant, a tolerance, a loop
bound). For each one the script copies the tracked tree to a scratch
directory, applies the mutant, runs the property tests
(tests/test_properties_pricing.py, test_properties_iv.py) with -x and the
derandomized ci profile, and records whether some property failed (the
mutant is "killed") or all passed ("survived"). The working tree is never
modified. A surviving mutant is reported, not fixed: it marks a property
the tests do not pin.

This is a curated list, not a mutation-testing framework (mutmut would
generate hundreds of mutants and take hours here).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HESTON = "joint_vol_calibration/models/heston.py"
IV = "joint_vol_calibration/models/implied_vol.py"
TESTS = ["joint_vol_calibration/tests/test_properties_pricing.py",
         "joint_vol_calibration/tests/test_properties_iv.py"]

# Mutants that cannot change any result on the inputs the properties draw.
# Each is listed with the measurement that shows it, and is reported, not failed.
EQUIVALENT = {
    "IV solver accepts a loose price residual":
        "the residual guard rejects spuriously small Newton steps on inconsistent input; on "
        "400,000 exact Black-76 prices (sigma 0.1%-1000%, T 1d-5y, +-12 sd) the mutant's "
        "output is identical to the original's, so no property over exact prices can kill it",
}

# (name, file, original text, mutated text)
MUTANTS = [
    ("Bates jump compensator dropped", HESTON,
     "return lam * T * (np.exp(1j * z * mu_j - 0.5 * z * z * sigma_j**2) - 1.0 - 1j * z * kbar)",
     "return lam * T * (np.exp(1j * z * mu_j - 0.5 * z * z * sigma_j**2) - 1.0)"),
    ("rho sign flipped in the Heston CF", HESTON,
     "beta = kappa - 1j * rho * sig * z", "beta = kappa + 1j * rho * sig * z"),
    ("v0 term of the CF off by 1%", HESTON, "return C + D * v0", "return C + D * v0 * 1.01"),
    ("kappa*theta term of the CF off by 0.1%", HESTON,
     "C = kappa * theta * (-s * T / bpd - 2.0 * log_ratio / (sig * sig))",
     "C = kappa * theta * (-s * T / bpd - 2.0 * log_ratio / (sig * sig)) * 1.001"),
    ("control-variate integral weight off by 1e-4", HESTON,
     "calls[pos] = bs_cv + pref * integral", "calls[pos] = bs_cv + 1.0001 * pref * integral"),
    ("absolute (not DF*F-relative) quadrature tolerance", HESTON,
     "tol_abs = rtol * df * F", "tol_abs = 1e-3"),
    ("step refinement skipped", HESTON, "for _ in range(12):", "for _ in range(0):"),
    ("integrated variance uses theta for v0", HESTON,
     "return T * (theta + (v0 - theta) * _one_minus_exp_over(kappa * T))",
     "return T * (theta + (theta - theta) * _one_minus_exp_over(kappa * T))"),
    ("IV solver accepts a loose price residual", IV,
     "_LN_RESIDUAL_TOL = 1e-7", "_LN_RESIDUAL_TOL = 1e-1"),
    ("IV solver: parity sign wrong for ITM calls", IV,
     "np.where(is_call, p - (Fs - Ks), p),", "np.where(is_call, p + (Fs - Ks), p),"),
    ("IV solver: Newton iteration capped at 3", IV,
     "    max_iter: int = 100,\n    full_output: bool = False,\n) -> Union",
     "    max_iter: int = 3,\n    full_output: bool = False,\n) -> Union"),
    ("Black-76 price: d1 drops the s/2 term", IV,
     "        d1 = -x / s_safe + 0.5 * s_safe\n        d2 = d1 - s_safe\n        call = x >= 0",
     "        d1 = -x / s_safe\n        d2 = d1 - s_safe\n        call = x >= 0"),
]


def copy_tree(dst: Path) -> None:
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    for rel in filter(None, files.decode().split("\0")):
        s = ROOT / rel
        if s.is_file():
            d = dst / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    for rel in TESTS:                          # the working-tree versions of the tests
        shutil.copy2(ROOT / rel, dst / rel)


def run(tree: Path, timeout: int) -> tuple[str, str]:
    env = {**os.environ, "HYPOTHESIS_PROFILE": "ci", "PYTHONPATH": str(tree)}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "-p", "no:cacheprovider"],
                           cwd=tree, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return "killed", f"timeout after {timeout} s"
    failed = [ln.split(" - ")[0].removeprefix("FAILED ") for ln in r.stdout.splitlines()
              if ln.startswith("FAILED")]
    if r.returncode == 0:
        return "survived", ""
    return "killed", (failed[0].split("::")[-1] if failed else f"exit {r.returncode}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timeout", type=int, default=600, help="seconds per mutant")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="mutation-") as tmp:
        tree = Path(tmp) / "tree"
        copy_tree(tree)
        status, detail = run(tree, args.timeout)
        if status != "survived":
            print(f"the unmutated tree fails the property tests ({detail}); fix that first")
            return 2
        rows = []
        for name, rel, old, new in MUTANTS:
            path = tree / rel
            src = path.read_text()
            if src.count(old) != 1:
                rows.append((name, "stale", "mutation site not found exactly once"))
                continue
            path.write_text(src.replace(old, new))
            t0 = time.time()
            status, detail = run(tree, args.timeout)
            path.write_text(src)
            rows.append((name, status, f"{detail} ({time.time() - t0:.0f} s)"))
            print(f"  {status:<9} {name}: {detail}", flush=True)
    killed = sum(r[1] == "killed" for r in rows)
    missed = [r for r in rows if r[1] != "killed" and r[0] not in EQUIVALENT]
    print("\n| Mutant | Result | First failing property |\n|---|---|---|")
    for name, status, detail in rows:
        why = f" (equivalent: {EQUIVALENT[name]})" if name in EQUIVALENT and status != "killed" else ""
        print(f"| {name} | {status}{why} | {detail} |")
    print(f"\n{killed}/{len(rows)} mutants killed; {len(missed)} unexpected survivor(s)")
    return 0 if not missed else 1


if __name__ == "__main__":
    sys.exit(main())
