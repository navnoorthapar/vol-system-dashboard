"""
test_bench_kernels.py: would compiled code make the Heston slice faster?

The production engine (models/heston.py::_lewis_call_prices) is NumPy. Once its
adaptive step has converged, a slice is one fixed trapezoid grid of n nodes on
the Lewis contour: n characteristic-function evaluations, then the sum
acc_j = sum_k Re(exp(i u_k x_j) gw_k) over m strikes. This file times that
same fixed-grid computation five ways, on the grid the engine itself chose
for the headline fit (data_store/calibrations/joint_cal_2026-10-16.pkl):

  numpy_fixed_grid    the engine's own code path (grid= argument)
  cpp_direct          C++ (g++ -O3, IEEE: no -ffast-math), one sincos per (k, j)
  numba_direct        Numba @njit, same algorithm
  cpp_recurrence      C++, phase advanced by one complex multiply per (k, j)
  numba_recurrence    Numba, same

and, for context, the production call with its adaptive search
(numpy_adaptive). Every implementation must first reproduce the NumPy prices
to 1e-10 DF F; only then is it timed. The C++ cases are skipped without a
C++ compiler, the Numba cases without numba (not a project dependency; to
include them: pip install numba==0.68.0). docs/benchmarks.md has the
measured table and what it does and does not show.
"""
from __future__ import annotations

import ctypes
import math
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.models import heston as H

# the `benchmark` fixture; without the plugin the file skips instead of erroring
pytest.importorskip("pytest_benchmark")

HEADLINE = Path(DATA_DIR) / "calibrations" / "joint_cal_2026-10-16.pkl"
KERNEL_SRC = Path(__file__).resolve().parent / "kernels" / "lewis_kernel.cpp"
PARITY = 1e-10          # units of DF * F


@pytest.fixture(scope="module", params=[7, 30, 365], ids=lambda d: f"{d}d")
def case(request):
    with open(HEADLINE, "rb") as fh:
        p = pickle.load(fh)["params"]
    S, r, q, T = 7659.44, 0.04, 0.013, request.param / 365.0
    k, th, sg, rho, v0 = p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    K = F * np.linspace(0.70, 1.30, 25)
    w = float(H._expected_integrated_variance(T, k, th, v0))

    def lcf(z):
        return H._heston_log_cf(z, T, k, th, sg, rho, v0)

    adaptive, grid = H._lewis_call_prices(S, K, T, r, q, lcf, w)
    x = np.log(F / K)
    sd = math.sqrt(w)
    d1 = x / sd + 0.5 * sd
    bs_cv = DF * (F * ndtr(d1) - K * ndtr(d1 - sd))
    pref = DF * np.sqrt(F * K) / np.pi
    return {"S": S, "r": r, "q": q, "T": T, "F": F, "DF": DF, "K": K, "w": w, "x": x,
            "params": (k, th, sg, rho, v0), "lcf": lcf, "grid": grid, "adaptive": adaptive,
            "bs_cv": bs_cv, "pref": pref, "days": request.param}


def _numpy_fixed(c):
    return H._lewis_call_prices(c["S"], c["K"], c["T"], c["r"], c["q"], c["lcf"], c["w"], grid=c["grid"])[0]


def _check_and_time(benchmark, c, fn, name):
    ref = _numpy_fixed(c)
    got = fn()
    err = float(np.max(np.abs(got - ref)) / (c["DF"] * c["F"]))
    assert err <= PARITY, f"{name}: {err:.2e} DF F from the NumPy engine"
    benchmark.group = f"lewis-slice-{c['days']}d"
    benchmark.extra_info.update(nodes=int(c["grid"].n), strikes=len(c["K"]),
                                max_abs_diff_over_DF_F=err)
    benchmark(fn)


def test_numpy_adaptive(benchmark, case):
    c = case
    benchmark.group = f"lewis-slice-{c['days']}d"
    benchmark.extra_info.update(nodes=int(c["grid"].n), strikes=len(c["K"]))
    benchmark(H._lewis_call_prices, c["S"], c["K"], c["T"], c["r"], c["q"], c["lcf"], c["w"])


def test_numpy_fixed_grid(benchmark, case):
    _check_and_time(benchmark, case, lambda: _numpy_fixed(case), "numpy")


# ── C++ ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cpp_lib(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("lewis") / "liblewis.so"
    subprocess.run([cxx, "-O3", "-std=c++17", "-shared", "-fPIC", str(KERNEL_SRC), "-o", str(out)],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(out))
    dbl, ptr = ctypes.c_double, np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.lewis_sum.argtypes = [ctypes.c_int, dbl, dbl, dbl, dbl, dbl, dbl, dbl, dbl,
                              ctypes.c_int, ptr, ctypes.c_int, ptr]
    lib.lewis_sum.restype = None
    return lib


def _cpp(lib, c, recurrence):
    x = np.ascontiguousarray(c["x"])
    acc = np.empty_like(x)
    g = c["grid"]

    def run():
        lib.lewis_sum(g.n, g.h, c["T"], *c["params"], c["w"], len(x), x, int(recurrence), acc)
        return c["bs_cv"] + c["pref"] * acc
    return run


def test_cpp_direct(benchmark, case, cpp_lib):
    _check_and_time(benchmark, case, _cpp(cpp_lib, case, False), "cpp_direct")


def test_cpp_recurrence(benchmark, case, cpp_lib):
    _check_and_time(benchmark, case, _cpp(cpp_lib, case, True), "cpp_recurrence")


# ── Numba ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def numba_kernel():
    nb = pytest.importorskip("numba")

    @nb.njit(cache=False, fastmath=False)
    def lewis_sum(n, h, T, kappa, theta, sigma, rho, v0, w, x, recurrence):
        sig = max(sigma, 1e-12)
        gw = np.empty(n, dtype=np.complex128)
        for k in range(n):
            u = h * k
            z = complex(u, -0.5)
            s = z * (z + 1j)
            beta = kappa - 1j * rho * sig * z
            d = np.sqrt(beta * beta + sig * sig * s)
            bpd = beta + d
            dz = -d * T
            em1 = dz + 0.5 * dz * dz + dz * dz * dz / 6.0 if abs(dz) < 1e-5 else np.exp(dz) - 1.0
            one_m_e = -em1
            e = 1.0 - one_m_e
            g = -sig * sig * s / (bpd * bpd)
            D = -s * one_m_e / (bpd * (1.0 - g * e))
            y = g * one_m_e / (1.0 - g)
            lr = complex(0.5 * math.log1p(2.0 * y.real + y.real * y.real + y.imag * y.imag),
                         math.atan2(y.imag, 1.0 + y.real))
            C = kappa * theta * (-s * T / bpd - 2.0 * lr / (sig * sig))
            uu = u * u + 0.25
            gw[k] = (math.exp(-0.5 * w * uu) - np.exp(C + D * v0)) / uu * h
        gw[0] *= 0.5
        acc = np.zeros(x.shape[0])
        for j in range(x.shape[0]):
            a = 0.0
            if recurrence:
                rot = complex(math.cos(h * x[j]), math.sin(h * x[j]))
                cur = complex(1.0, 0.0)
                for k in range(n):
                    a += cur.real * gw[k].real - cur.imag * gw[k].imag
                    cur *= rot
            else:
                for k in range(n):
                    ph = h * k * x[j]
                    a += math.cos(ph) * gw[k].real - math.sin(ph) * gw[k].imag
            acc[j] = a
        return acc

    return lewis_sum


def _numba(kernel, c, recurrence):
    g = c["grid"]

    def run():
        return c["bs_cv"] + c["pref"] * kernel(g.n, g.h, c["T"], *c["params"], c["w"], c["x"], recurrence)
    run()                               # compile outside the timed region
    return run


def test_numba_direct(benchmark, case, numba_kernel):
    _check_and_time(benchmark, case, _numba(numba_kernel, case, False), "numba_direct")


def test_numba_recurrence(benchmark, case, numba_kernel):
    _check_and_time(benchmark, case, _numba(numba_kernel, case, True), "numba_recurrence")
