# Reproducing the committed artifacts

```bash
make reproduce           # backtest, Greeks surface, C7 study, the five model_version 2 calibrations
make reproduce-full      # + the β grid (6 backtests), the C22 waterfall (15 rows) and the 200-path S3 null test
python scripts/ci/reproduce.py --only greeks,c7      # a subset; --jobs N, --keep, --out DIR
```

`scripts/ci/reproduce.py` follows the Open Source Risk Engine's Output vs ExpectedOutput pattern. It copies the
tracked files (code, the captured market data and option chains under `data_store/`, and the committed artifacts) to a
scratch directory, deletes each artifact there, runs the command the README lists for regenerating it, and compares
the result with the committed file:

| Artifact | Command | Compared | Tolerance (relative to each column's or value's scale) |
|---|---|---|---|
| `backtest/full_results.parquet`, `.pkl` | the canonical `BacktestEngine` run (`run_backtest` without the S3 null test and sub-period windows) | every column; every metric except `s3_null_p_value`, the trade log, the settings | 1e-10 |
| `greeks/greeks_surface.parquet` | `risk_monitor --calibration joint_cal_2026-10-16.pkl` (the committed fit) | every column | 1e-10 |
| `backtest/delta_hedge_2020.parquet` | `run_backtest --c7` | every column | 1e-10 |
| `calibrations/joint_cal_<session>.pkl` × 5 | `JointCalibrator(as_of_date=<session>, **recorded_inputs(<committed fit>)).calibrate(); save()` (the VIX targets and the reference rate the fit recorded: C23 restated the database's VIX family, and new fits discount on the Treasury curve) | every field except `fit_time` and `run_utc`; `methods` and `chain_diagnostics` reported, not failed (below) | 1e-6 |
| `backtest/beta_sensitivity.csv` (`--full`) | `run_backtest --beta-sensitivity` | every cell | 1e-9 (10 significant digits) |
| `backtest/c22_bridge.csv` (`--full`) | `c22_bridge --csv` | every cell | 1e-9 |
| `backtest/c23_bridge.csv` | `c23_bridge --csv` (4 backtests) | every cell | 1e-9 |
| `metrics.s3_null_p_value` (`--full`) | `run_backtest` | the p-value | exact |

1e-10 is the tolerance the suite already uses for last-bit differences between CPUs (AVX-512 vs AVX2,
`tests/artifact_tolerance.py`). The calibrations get 1e-6 because differential evolution re-runs its whole
trajectory: a last-bit difference in one loss evaluation on another CPU can move the optimum by about 1e-8 relative
(README). Steps run with `PYTHONHASHSEED=0` and one BLAS/OpenMP thread. The real `data_store/` is never written, and
the report goes to `<out>/report.md` and `report.json`, with each step's log under `<out>/logs/`.

`methods` (how T, carry and anchors were chosen) and `chain_diagnostics` (line counts of the input chain) describe the
inputs, not the model's outputs. Every difference in them is listed in the report, but none fails a check: the
committed pickles were written before later provenance fields existed, and a fit made by the daily job from the live
download records the download's raw line count, where a refit from the stored chain (which keeps only two-sided
lines, by design) records its own.

## Last result

`make reproduce` at commit `defa301` (the artifacts and the code that writes them are unchanged since), 2026-10-20,
Intel Xeon @ 2.10 GHz with AVX-512, 4 vCPU shared with other jobs, Python 3.11.15, NumPy 2.4.6, pandas 3.0.6,
three steps in parallel: **9 of 9 artifacts reproduce, every number bit for bit** (largest relative deviation
0.0 everywhere).

| Artifact | Largest relative deviation | Step time |
|---|---|---|
| `backtest/full_results.parquet`, `full_results.pkl` | 0.0 | 38 s |
| `greeks/greeks_surface.parquet` | 0.0 | 8 s |
| `backtest/delta_hedge_2020.parquet` (C7) | 0.0 | 23 s |
| `calibrations/joint_cal_2026-03-24.pkl` | 0.0 | 157 s |
| `calibrations/joint_cal_2026-03-27.pkl` | 0.0 | 119 s |
| `calibrations/joint_cal_2026-05-29.pkl` | 0.0 | 151 s |
| `calibrations/joint_cal_2026-06-18.pkl` | 0.0 | 85 s |
| `calibrations/joint_cal_2026-10-16.pkl` | 0.0 | 71 s |

Provenance differences, reported and not failed:

- all five calibrations: the current code writes two more `methods` entries (`T_origin`, `vix_leg_timing`), words
  `methods.T` differently ("from the quote session's 16:00 ET close (closing book) to ..." against "quote session
  16:00 ET -> ..."), and records the T origin in `chain_diagnostics` (`quote_time`, `t_origin`, `t_origin_basis`).
  These fields were added in C22 after the pickles were written (commit `3e8b2cd`, the GTH quote-time origin); the
  numbers they describe did not change;
- 2026-10-16 only: `chain_diagnostics.n_raw` is 6,191 in the refit against 7,583 in the committed pickle, and
  `n_after_expiry_filter` 6,191 against 6,444. The committed fit was made by the daily job from the live download
  (7,583 lines); the stored chain holds the 6,191 two-sided lines of that download, which reproduce the same clean
  quotes and the same fit.

The step times are on a loaded machine (load average 7–13 during the run); a quiet single calibration takes about
42 s (docs/benchmarks.md). The `--full` steps were not run for this record. A re-run of the backtest, Greeks and
C7 steps after the last edit to `reproduce.py` (`--only greeks,c7,backtest`) again gave 4 of 4 at deviation 0.0.
CI runs `make reproduce` weekly and on demand on GitHub's runners (job `reproduce` in tests.yml), which tests the same
claim on a different CPU; there the 1e-10 and 1e-6 tolerances, not bit equality, are the claim.
