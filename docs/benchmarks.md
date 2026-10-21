# Benchmarks

Measured with pytest-benchmark 5.3.0 (`make bench`; the compact results are in `docs/benchmarks/`, written by
`scripts/bench/summarize.py`). The production paths below are the run of 2026-10-21
([`benchmarks/2026-10-21.json`](benchmarks/2026-10-21.json), code of commit `98e4366`); the kernel comparison further
down is the run of 2026-10-20 ([`benchmarks/2026-10-20.json`](benchmarks/2026-10-20.json), commit `defa301`), the last
one with Numba installed.

**Machine.** Intel Xeon @ 2.10 GHz (AVX-512), 4 vCPU, Linux 6.18 VM; CPython 3.11.15, NumPy 2.4.6, SciPy 1.17.1,
pandas 3.0.6; Numba 0.68.0 and g++ 13.3.0 for the kernel comparison only. One thread (OMP/OpenBLAS/MKL = 1),
`PYTHONHASHSEED=0`.

**Noise.** The VM was shared with other jobs (load average 4.5–8.9 on 4 vCPU during the 2026-10-20 runs, 1.5–3.1
during the 2026-10-21 run, which overlapped a test-suite run), so medians are inflated and unstable. The table reports
the **minimum over rounds**, the usual estimator of the uncontended cost; the two full calibrations are one round each,
so their times carry the load with them. Across the two 2026-10-20 runs the minima of the fixed-grid and compiled
kernels agreed within 0.0–3.5%, the production slice and the loss evaluation within 10%, and the two chain-level IV
benchmarks moved by 22% and 40%. Re-measure on a quiet machine before quoting a median. Every benchmark first checks
that it computes the right thing (the loss recomputes the committed fit's loss to 1e-9; the DE calibration reproduces
the committed parameters to 1e-6 relative; every compiled kernel reproduces the NumPy engine) and none asserts on time.

## Production paths

| What | Input | Min | Median | Per unit |
|---|---|---|---|---|
| `heston_call_batch`, one 25-strike slice, K/F 0.70–1.30 | headline fit, T = 7 d | 0.82 ms | 0.99 ms | |
| same | T = 30 d | 0.67 ms | 0.95 ms | |
| same | T = 365 d | 0.74 ms | 1.24 ms | |
| 7 expiries × 15 strikes | headline fit | 6.4 ms | 9.5 ms | 0.91 ms per expiry |
| `implied_vol_black` (Newton), every clean OTM quote | 3,209 quotes, 2026-10-16 chain | 4.6 ms | 6.3 ms | 1.4 µs per quote |
| `black76_implied_vol` (80-step bisection; the cleaner and the calibrator use it) | same quotes | 12.2 ms | 16.9 ms | 3.8 µs per quote |
| `JointCalibrator.joint_loss`, one evaluation | 53 anchors, 6 expiries, 4 VIX indices | 3.9 ms | 5.8 ms | |
| `JointCalibrator.calibrate()`, the default since C23: multi-start trust-region least squares with the analytic Jacobian | 2026-10-16 | 2.1 s | (1 round) | 276 evaluations with their Jacobian, 7.7 ms each incl. optimiser |
| `JointCalibrator.calibrate(optimizer="de")`: DE 50 × 200 + L-BFGS-B, seed 42, the optimiser of every committed fit | 2026-10-16 | 28.3 s | (1 round) | 5,018 evaluations, 5.6 ms each incl. optimiser |

The DE path reproduces the committed parameters (asserted to 1e-6 relative; `make reproduce` compares every field,
docs/reproduce.md). The default TRF path is a new fit: its loss is asserted at or below the committed one (×(1 + 1e-6))
and its parameters within 2e-3 relative (κ moves 8.5e-4 along a flat direction), not 1e-6. The TRF benchmark runs in
the CI bench smoke test; the DE one is marked slow and runs in `make bench` only. The committed pickle records 50.7 s for its DE fit on the machine that wrote it; the diagnostics measured
1.7–2.1 s for TRF against 19.8–28.3 s for serial DE on the five sessions (quiet machine;
`data_store/calibrations/diagnostics/summary.json`). The bisection inverter cost 2.8×, 1.4× and 2.7× the Newton solver
on the same quotes in the three runs, so the ratio is not established on this machine; on OTM inputs the two agree to
2.1e-12 relative (worst of 1e5 random inputs; `test_properties_iv.py` pins 1e-9).

## Would compiled code make the slice faster?

Once its adaptive step has converged, a slice is a fixed trapezoid grid of n nodes: n characteristic-function
evaluations, then acc_j = Σ_k Re(e^{i u_k x_j} gw_k) over the m strikes. `scripts/bench/test_bench_kernels.py`
times that fixed-grid computation on the grid the engine itself chose for the headline fit, five ways, after
checking each against the NumPy engine (all agree to ≤ 1.3e-16 DF·F):

| T (nodes × strikes) | NumPy, production (adaptive) | NumPy, fixed grid | Numba, direct | C++, direct | Numba, recurrence | C++, recurrence |
|---|---|---|---|---|---|---|
| 7 d (533 × 25) | 881 µs | 473 µs | 454 µs | 297 µs | 136 µs | 143 µs |
| 30 d (581 × 25) | 943 µs | 499 µs | 489 µs | 322 µs | 150 µs | 157 µs |
| 365 d (689 × 25) | 1,055 µs | 567 µs | 564 µs | 381 µs | 181 µs | 190 µs |

"Direct" evaluates one sincos per (node, strike), as the NumPy code does; "recurrence" advances the phase
e^{i u_k x_j} by one complex multiply per (node, strike) (u_k = k h). C++ is `g++ -O3` without `-ffast-math`
(IEEE semantics, as NumPy and Numba use by default); Numba is `@njit(fastmath=False)`.

What this shows:

1. **Half of a production slice is the adaptive search** (truncation scan and step refinement): pricing on the
   converged grid takes 46–47% less time in plain NumPy. Reusing a verified grid across the ~5,000 loss evaluations of a
   calibration is the cheapest available speed-up and needs no new dependency.
2. **For the same algorithm, Numba buys nothing over NumPy** (489 vs 499 µs at 30 d: NumPy's complex exponential is
   already compiled, vectorised C) **and C++ is 1.5× faster than both.** Both compiled builds call libm's `sincos`
   (checked: the C++ library imports `sincos`; Numba's LLVM fuses its `llvm.sin`/`llvm.cos` calls into `sincos`), so
   the gap is not one transcendental call against two. A micro-test of the inner loop alone (581 × 25 phases,
   best of 3,000 calls, same inputs, same glibc `sincos` address) gave g++ 13 180 µs, clang++ 18 227 µs (it calls
   `sin` and `cos` separately) and Numba 260 µs: the difference is in the code each compiler generates around the
   call, not in the transcendental itself. Its exact cause was not established further.
3. **The recurrence is the large win: 3.2–3.3× over the NumPy fixed grid and 6.0–6.3× over the production path at
   30 d, and there C++ and Numba are within about 5%** (157 vs 150 µs). What remains is mostly the characteristic
   function itself, about 0.23 µs per node (complex sqrt, exp, log1p, atan2), which both languages pay alike.

So the answer to "would C++ beat Numba?" is: on the transcendental-bound direct kernel it did (1.5×); once the
algorithm removes the transcendentals from the inner loop, it did not. The speed-ups worth having, grid reuse and the
phase recurrence, are algorithmic, and Numba is enough for the second. Neither is wired into the engine yet (the
calibrator is owned elsewhere); these timings are the evidence for doing so. The analytic Jacobian (3.4–4.0× faster per
call than two-point differences) and differential evolution in four processes (7.9–15.4 s, agreeing with TRF to
2.1e-13) are measured in `data_store/calibrations/diagnostics/summary.json`, not here. Not measured: GPU.

### Where a production slice spends its time

cProfile over 2,000 slices at T = 30 d (25 strikes; profiling inflates the total to 1.62 ms per slice):

| Function | Share |
|---|---|
| `_lewis_sum` (complex exponential of the node × strike outer product, summed) | 35% |
| `_heston_log_cf` + `_clog1p` | 26% |
| `_lewis_call_prices` body (scan, refinement bookkeeping) | 12% |
| `g` (control-variate integrand) | 5% |
| NumPy reductions, `outer`, `geomspace`, other | 22% |

## Reproducing these numbers

```bash
make bench                                     # writes .bench/benchmarks.json (about 1 minute; 53 s on 2026-10-21)
python scripts/bench/summarize.py .bench/benchmarks.json --json docs/benchmarks/$(date +%F).json
pip install numba==0.68.0                      # optional: include the Numba kernels (not a project dependency)
```

CI runs every benchmark body once without timing (`make bench-smoke`); shared CI runners are too noisy for timings.
