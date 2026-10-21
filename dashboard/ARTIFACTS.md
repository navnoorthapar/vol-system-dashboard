# Dashboard artifact contract

The dashboard is a read-only view of `data_store/`. This file is the contract
for the artifacts added in C23, written against what their producers write:

| Artifact | Producer | Schema |
|---|---|---|
| `data_store/surface/<session>/` | `scripts/surface_report.py` (`joint_vol_calibration/surface/report.py`) | surface report `schema_version` 1 |
| `data_store/calibrations/diagnostics/<session>.json` | `scripts/calibration_diagnostics.py` | `schema_version` 1 |
| `data_store/backtest/inference/` | `scripts/inference_report.py` (`joint_vol_calibration/stats`) | `inference_summary/1` |
| `data_store/forward/ledger_<session>.json` | `joint_vol_calibration/backtest/forward_ledger.py` | `forward_ledger/1` |
| `data_store/greeks/scenario_grid.parquet` | `joint_vol_calibration/greeks/scenario_grid.py` (the daily bot rebuilds it from each new headline fit) | section 5 |

`dashboard/desk_artifacts.py` is the executable form of this file. If the two
disagree, the code is what runs, and this file should be fixed to match it.
The pre-C23 inputs (`vol_system.db`, the calibration pickles,
`greeks_surface.parquet`, the backtest pickles and parquets) keep the
contracts their loaders in `app.py` already enforce.

## States

Every reader returns one of three states:

| State | When | What the page does | Freeze |
|---|---|---|---|
| `absent` | The file or directory does not exist | Shows an "Artifact not yet generated" panel naming the path and its producer. No number is estimated in its place. | Publishes |
| `ok` | The file exists and validates | Renders it | Publishes |
| `ArtifactError` | The file exists but breaks this contract: a missing column, a wrong unit, another session's rows, a non-finite value, a bad interval or probability, a broken hash chain | The loader records an error, the page shows the red banner, and `X-Dashboard-Loader-Errors` is non-zero | **Blocked** (fail closed) |

## Units used throughout

- **Implied vols** in tables are decimals (`0.183` means 18.3%). Fields ending in `_vp` are vol points (`0.25` means 0.25 vol points), and the VIX-strip and term-structure vols (`var_swap_vol`, `atm_vol_*`) are in vol points, as the index is quoted. A decimal IV column whose median is above 3, or a vol-point field outside (0.5, 300), is rejected as a unit error.
- **Log-moneyness** is `k = ln(K / F_T)`, where `F_T` is the expiry's put–call-parity forward.
- **Maturity** is `T_days` in calendar days (or `T` in years, converted as `T × 365`).
- **Dates** are ISO `YYYY-MM-DD` session dates (the NYSE close the data is as of).
- **Residual in half-spreads** is the producer's `error_hs = error_vp / max(half_spread_vp, 0.01)`, with `error_vp = (model_iv − mid_iv) × 100`. `|error_hs| ≤ 1` means the model is inside the quoted bid–ask; `inside_spread` is the exact test `bid_iv ≤ model_iv ≤ ask_iv`.

## 1. Quote-grade surface: `data_store/surface/<session>/`

**Consumers:** the Surface page, the Overview's "Model vs market" key result, the Calibration page's smile small multiples, and two Model Risk checks.

The newest `<session>` directory that holds a `residuals.parquet` is shown. `residuals.parquet` is required for the `ok` state. The three JSON files are optional, but each one that is present must validate and must carry its directory's `session`.

### `residuals.parquet`: one row per (model, quote)

The market columns repeat for every model; every model must be scored on the same number of quotes.

| Column | Required | Meaning |
|---|---|---|
| `model` | yes | `heston_v2`, `essvi`, `svi`, `svi_cal`, ... The first model whose name starts with `heston` is the production fit. |
| `expiry` | yes | Expiry date (the first 10 characters are used) |
| `k` | yes | `ln(K / F_T)`, finite |
| `mid_iv` | yes | Mid implied vol, decimal, positive and finite |
| `model_iv` | yes | The model's IV at the quote, decimal; NaN where the model price cannot be inverted |
| `half_spread_vp` | or `bid_iv` + `ask_iv` | Half the bid–ask in vol points, finite and ≥ 0 |
| `T_days` | or `T` | Calendar days to expiry |
| `bid_iv`, `ask_iv` | recommended | Decimal; with them the smile charts draw bid–ask bars |
| `in_sample` | recommended | `True` for quotes that model was fitted to. Every score on the pages uses `in_sample == False` only and is labelled "out-of-sample"; without the column the label is "all quotes". |
| `error_vp`, `error_hs`, `inside_spread` | optional | Derived as above when absent |
| `session` | optional | If present, every row must equal the directory's session |

### `scorecard.json`

```json
{"session": "2026-10-16", "n_quotes": 2913, "notes": ["..."],
 "models": [{"model": "heston_v2", "description": "...", "n_params": 5, "n_in_sample": 53,
             "metrics": {"out_of_sample": {"n": 2860, "median_abs_hs": 6.32, "inside_spread_pct": 7.59,
                                           "rmse_vp": 2.96, "mean_error_vp": 0.06, "within_2hs_pct": 14.5},
                         "all": {...}, "in_sample": {...}, "excluding_arbitrage_flagged": {...},
                         "by_expiry": {"2026-10-29": {"T_days": 13.0, "median_abs_hs": 9.76, ...}}},
             "dense_grid": {"butterfly_violations": 0, "calendar_violations": 0}}],
 "ranking_by_oos_median_abs_hs": [{"model": "svi"}, {"model": "heston_v2"}]}
```

- `models` is a non-empty list; each row needs `model` and `metrics.out_of_sample` with `n` (a non-negative integer), `median_abs_hs`, `inside_spread_pct` (in [0, 100]) and `rmse_vp` (≥ 0).
- Rows are shown in the order of `ranking_by_oos_median_abs_hs`.
- `notes` are printed under the scorecard as written (they carry the stale-quote caveat).
- The panel title says "same quotes" only when every row's `out_of_sample.n` is equal; otherwise "each model's out-of-sample quotes".
- `metrics.by_expiry` covers all quotes (in and out of sample), so the pages do not read it: the "Median |z| by expiry" chart is computed from `residuals.parquet` on each model's out-of-sample quotes, the scorecard's own sets.
- The Overview's first key result is the `svi_cal` row's out-of-sample `median_abs_hs` on the newest session, with its range over every stored session beside it; a range across sessions is not a sampling interval, and the card says so.

### `surface.json`

- `arbitrage`: `n_rows`, `mid_violations`, `mid_violations_gt_tick` and `executable_violations` each map a constraint (`C1_outright` ... `C6_calendar_butterfly`) to a count (a non-negative integer); `n_quotes`, `n_quotes_in_executable_violation`, `n_quotes_in_mid_violation`; `repair` (`n_perturbed`, `n_perturbed_outside_spread`, `max_abs_dprice_half_spreads`).
- `surfaces.<model>.dense_grid`: `butterfly_violations`, `calendar_violations`.
- `term_structure`: one row per expiry with `T_days` (required), `atm_vol_mid_quotes`, `atm_vol_essvi` (vol points), `atm_skew_essvi`, `r`, `q`, `n_quotes`.
- `universe`, `fit_set`, `inputs`: descriptive; printed where useful.

### `vix_replication.json`

- `indices.<VIX9D|VIX|VIX3M|VIX6M>`: `replicated` (index points, strictly between 1 and 200), `official` (`{"close", "low", "high", ...}` or a number), `gap_vs_official`, `official_inside_day_range`.
- `strip`: one row per expiry with `expiry` (required), `days`, `var_swap_vol` and `atm_vol_quotes` (vol points), `vs_minus_atm_vp`, `atm_over_vs`, `n_puts`, `n_calls`.
- `beta.atm_over_vix_official`; `deviations` and `gap_notes` (lists of strings, printed in a collapsible section).

### `data_store/surface/index.json` (optional)

`{"sessions": [{"session", "n_quotes", "arbitrage_quotes_outside_spread", "vix_replicated", "vix_official", "atm_over_vix_official", "models": {"<model>": {"median_abs_hs", ...}}}]}`. Every row needs a `YYYY-MM-DD` session. The Surface page lists every session.

## 2. Calibration diagnostics: `data_store/calibrations/diagnostics/<session>.json`

**Consumer:** the Calibration page, "Identifiability" panel (and the parameter tiles' standard errors).

The headline result pickle's own `identifiability` and `multistart` keys win when present; the file adds what the pickle lacks.

- `identifiability` (required for `ok`): `param_names` and `rel_jacobian_singular_values` (finite, ≥ 0, largest first) are required; `rel_jtj_condition_number`, `standard_errors`, `relative_standard_errors`, `weakest_direction`, `on_bound`, `n_residuals` and `se_note` are shown when present. An `error` key is an `ArtifactError`.
- `multistart` or `trf.multistart`: `n_starts`, `n_starts_at_best`, `n_distinct_basins`.
- `sigma_profile.rows`: each row needs `sigma_cap` and `loss`; `params`, `spx_iv_rmse_exact`, `spx_oos_iv_rmse`, `spx_oos_put_wing_rmse`, `vix_index_rmse`, `sigma_at_cap`, `kappa_at_max` are shown.
- `spx_only.<heston|heston_wide_box|rough_heston>`: `params` (with `H` for rough Heston), `spx_iv_rmse_exact`, `spx_oos_iv_rmse`, `spx_oos_put_wing_rmse`, `vix_index_rmse_oos`.
- A `session` that differs from the file name is an `ArtifactError`.
- `summary.json` (optional; the Calibration page's "Heston vs rough Heston" panel and the Overview's joint-fit result):
  `rows[]`, one per diagnosed session (`YYYY-MM-DD`, unique). Read per row, each a finite number or null:
  `sigma_cap2_oos` (the production joint Heston fit's out-of-sample SPX RMSE, vol pts), `joint_heston_vix_index`,
  `joint_rough_oos`, `joint_rough_in`, `joint_rough_H` (in [0, 1/2]), `joint_rough_vix_index`,
  `joint_rough_lifting_error_estimate_max_vp`, `heston_spx_oos`, `rough_oos`, `rough_H`, `sigma_free`, `kappa_free`,
  `sigma_free_oos`, `sigma_free_loss_rel`, `vvix_market`, `vvix_model_cap2`, `rel_jtj_condition_number`, `trf_seconds`, `trf_evals`,
  `de_serial_seconds`, `de_serial_evals`, `trf_loss_rel_diff`; lists `on_bound`, `joint_rough_on_bound`; the object
  `relative_standard_errors`.

## 3. Backtest inference: `data_store/backtest/inference/`

**Consumers:** the Research page (Inference, Benchmarks and Vol forecasts panels) and the Overview's third key result.

`summary.json` is required for `ok`; `benchmarks.json` and `forecast_eval.json` are optional. The replicate-level parquet files beside them are the producer's audit trail and are not read.

- `summary.json`: `schema` starts with `inference_summary/`; `bootstrap.<series>.<statistic>` with `point` and `ci95 = [lo, hi]` (finite, `lo ≤ hi`), `excludes_zero`; `bootstrap_settings` (`level`, `reps`, `method`); `closed_form.<series>` (`psr_gt_0`, `skew`, `kurtosis`); `trials` (`dsr.<series>.n_eff_c16_c22`, `dsr_range`, `n_choices`, `ledger.path`); `nulls.sleeves.<sleeve>.<circular|shuffle>.<total_pnl|sharpe>` with `real`, `null_mean`, `null_q05`, `null_q95`, `timing_component`, `n_paths` and `p_value`, `p_holm_6_sleeves` in [0, 1]; `nulls.settings`; `multiple_testing` (`spa_*`, `stepm_*`, `pbo_*`); `headline` (strings, shown in a collapsible section); `conventions`.
- The Overview's interval is `bootstrap.portfolio.sharpe.ci95`, and its S3 timing p is `nulls.sleeves.s3.circular.total_pnl`.
- `benchmarks.json`: `indices` (list), `co_movement.<series>.<index>.all` with `corr_daily`, `beta`, `sharpe_diff`; `implied_beta.<PUT|WPUT>` with `overall`, `by_year`; `implied_beta.canonical_beta`; `window`.
- `forecast_eval.json`: `backtest_window` with `losses.qlike.<model>`, `rank.qlike`, `r2_vol`, `mcs.qlike."90".included`, `start`, `end`, `n`; `models.<model>` (descriptions).
- Every number here is a statistic of a mark-to-model backtest, and the panels say so.

## 4. Forward ledger: `data_store/forward/ledger_<session>.json`

**Consumer:** the Research page, "Forward test" panel.

- Every file has `schema == "forward_ledger/1"` and a `session` equal to its file name.
- The files form a hash chain: each file's `previous` names the prior file's `session` and the SHA-256 of its canonical JSON (keys sorted, separators `,` and `:`, UTF-8). The first file has `previous: null`. A broken link is an `ArtifactError`: a registered ledger must never change, and an edited one must not publish.
- The panel shows the newest file's `summary.<sleeve>` (`realised_pnl_net`, `open_pnl_exec`, `n_opened`, `n_closed`), `signals.<sleeve>.straddle_direction` and the open `positions` (`contract.expiry`, `contract.strike`, `direction`).

## 5. Scenario grid: `data_store/greeks/scenario_grid.parquet`

**Producer:** `python -m joint_vol_calibration.greeks.scenario_grid --calibration <joint_cal_*.pkl>` (the reference book: a 30-day straddle struck at the forward, delta-hedged at inception, revalued in full on the fit's smile; 9 spot × 6 vol shocks × 2 dynamics). **Consumer:** the Risk page "Scenario grid" panel, which shows "not yet generated" until it exists.

| Column | Meaning |
|---|---|
| `spot_shock_pct` | Spot shock in percent (`-10` means −10%) |
| `vol_shock_vp` | Parallel vol shock in vol points |
| `pnl` | Full-revaluation P&L of the reference book, finite |
| `dynamics` | Optional: `sticky_strike`, `sticky_moneyness` and so on. The page shows `sticky_strike` when present, otherwise the first value in sorted order, and names the others. |

- `(spot_shock_pct, vol_shock_vp[, dynamics])` must be unique.
- Parquet key-value metadata `b"scenario_provenance"` holds a JSON object with `book` (description), `as_of_date` and `units`. These are shown under the grid. The producer also records `source.calibration_file`, the parameters, carry, strike, forward, hedge delta, the shock lists and `n_vol_floored`, and writes the extra columns `spot`, `iv_strike`, `straddle_value`, `hedge_pnl` and `vol_floored`.

## Producing an artifact

1. Write the file atomically (to a temporary file, then rename) into the path above.
2. Run `python dashboard/freeze_site.py <dir>`. It must print `OK` for every page; an `ArtifactError` stops it with the message.
3. `joint_vol_calibration/tests/test_dashboard.py` (`test_artifact_*`) renders small captured samples of each producer's output (`joint_vol_calibration/tests/fixtures/dashboard/`) in a temporary `data_store`, checks all three states, and checks that a malformed file blocks the freeze. When a producer changes its output, update those samples and this file in the same commit.
