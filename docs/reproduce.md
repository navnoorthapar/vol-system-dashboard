# Reproducing the committed artifacts

```bash
make reproduce           # backtest, Greeks surface, scenario grid, C7 study, the model_version 2 calibrations
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
| `greeks/greeks_surface.parquet` | `risk_monitor --calibration <the fit its provenance names>` (the headline fit: 2026-10-16 until the daily bot accepts a newer one and rebuilds it) | every column | 1e-10 |
| `greeks/scenario_grid.parquet` | `python -m joint_vol_calibration.greeks.scenario_grid --calibration <the fit its provenance names>` | every column | 1e-10 |
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

`make reproduce` plus the three `--full` steps at commit `a678e57` with the refit fix of the next commit, 2026-10-21,
Intel Xeon @ 2.10 GHz with AVX-512, 4 vCPU (load average about 1), Python 3.11.15, NumPy 2.4.6, pandas 3.0.6, four
steps in parallel: **14 of 14 artifacts reproduce, every number bit for bit** (largest relative deviation 0.0
everywhere).

| Artifact | Largest relative deviation | Step time |
|---|---|---|
| `backtest/full_results.parquet`, `full_results.pkl` | 0.0 | 8 s |
| `greeks/greeks_surface.parquet` (from `joint_cal_2026-10-16.pkl`, named in its provenance) | 0.0 | 5 s |
| `greeks/scenario_grid.parquet` (same fit) | 0.0 | 4 s |
| `backtest/delta_hedge_2020.parquet` (C7) | 0.0 | 8 s |
| `calibrations/joint_cal_2026-03-24.pkl` | 0.0 | 29 s |
| `calibrations/joint_cal_2026-03-27.pkl` | 0.0 | 22 s |
| `calibrations/joint_cal_2026-05-29.pkl` | 0.0 | 30 s |
| `calibrations/joint_cal_2026-06-18.pkl` | 0.0 | 27 s |
| `calibrations/joint_cal_2026-10-16.pkl` | 0.0 | 29 s |
| `backtest/c23_bridge.csv` | 0.0 | 23 s |
| `backtest/beta_sensitivity.csv` (`--full`) | 0.0 | 33 s |
| `backtest/c22_bridge.csv` (`--full`) | 0.0 | 66 s |
| `metrics.s3_null_p_value` (`--full`) | 0.0 | 179 s |

The five refits run differential evolution with one worker, the optimiser every committed fit was made with (they
record no `optimizer` key); the calibrator's default since C23 is multi-start TRF, which the daily bot's fits record
(`optimizer: trf_multistart`) and are refitted with. Each refit reads the inputs its fit recorded from the committed
pickle in the checkout. From 2566553 (which made the refits read their recorded inputs) until that fix, they read
the scratch copy's pickle, which `run_step` deletes first, so all five calibration steps failed; with the path
right, the default TRF refit would still have missed the committed DE fits by up to 2e-3 relative (per-expiry
out-of-sample RMSE; the loss agrees to about 1e-7).

Provenance differences, reported and not failed:

- all five calibrations: the current code writes three more `methods` entries (`T_origin`, `vix_leg_timing`,
  `vix_targets`), words `methods.T` differently ("from the quote session's 16:00 ET close (closing book) to ..."
  against "quote session 16:00 ET -> ..."), records the T origin in `chain_diagnostics` (`quote_time`,
  `t_origin`, `t_origin_basis`), and adds the C23 records `identifiability` and `rate_reference` (the committed
  fits predate both; their recorded reference is ^IRX). The numbers these describe did not change;
- 2026-10-16 only: `chain_diagnostics.n_raw` is 6,191 in the refit against 7,583 in the committed pickle, and
  `n_after_expiry_filter` 6,191 against 6,444. The committed fit was made by the daily job from the live download
  (7,583 lines); the stored chain holds the 6,191 two-sided lines of that download, which reproduce the same clean
  quotes and the same fit.

CI runs `make reproduce` weekly and on demand on GitHub's runners (job `reproduce` in tests.yml), which tests the same
claim on a different CPU; there the 1e-10 and 1e-6 tolerances, not bit equality, are the claim.
