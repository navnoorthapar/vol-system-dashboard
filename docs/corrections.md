# Corrections C16–C23

Every round of correction since March 2026, as a blameless postmortem: what was wrong, the root cause, how it was
found, the fix, the regression tests that now pin it, the commits, and what it did to the published result. The
numbers in each round are the ones published when it closed; the backtest has been re-run against the same window
(2018-01-02 → 2025-03-24) after every round, and each round made the result worse or removed a thesis.

The rounds below are rendered from `dashboard/corrections.json` by `scripts/render_corrections.py`, the same registry
the [Corrections page](https://navnoorbawa.me/corrections.html) reads; `test_readme.py` fails if this file falls behind
it, and `test_dashboard.py` checks that every cited test exists and every commit resolves. The narrative notes after
the generated block keep the longer explanations that used to sit in the README.

<!-- BEGIN GENERATED: scripts/render_corrections.py -->

C16–C23: 8 rounds, 28 findings (25 fixed, 3 quantified), 51 regression tests cited. Rendered from `dashboard/corrections.json`, the registry the site's Corrections page reads.

| Round | Date | Severity | Title | Effect on the published result |
|---|---|---|---|---|
| [C16](#c16) | 2026-06-11 | high | Methodology audit: look-ahead, a circular feature, retrospective sizing | The published +19% headline was a label-noise artifact. |
| [C17](#c17) | 2026-06-11 | high | Red-team: the classifier against a persistence baseline; margin; an unbounded tail | The classifier loses to 'predict yesterday's regime' and was demoted from the trading loop; portfolio −9.1% as published after this round. |
| [C18](#c18) | 2026-06-14 | high | Straddles never rolled: one stale-strike position marked past expiry | Inverted the contrarian thesis: S1 −$1.53M → +$463K, S1C −$97K → −$404K, portfolio −9.1% → −19.0%; S3 unchanged. |
| [C19](#c19) | 2026-06-20 | low | Signal-engine logic fixes and data-gap repair | Cleanup; the canonical numbers did not change. |
| [C20](#c20) | 2026-08-04 | high | The calibration selector rewarded broken data | The published 'best calibration' had been the day the VIX feed failed; fits are now gated on at least 3 VIX tenors with provenance recorded. |
| [C21](#c21) | 2026-09-05 | high | The cash hurdle was a test fixture; straddles marked a day too rich | Portfolio −19.0% → −26.1%, Sharpe −1.57 → −1.80, MaxDD −25.9% → −31.5%; S1 +$463K → +$694K, S1C −$404K → −$600K. |
| [C22](#c22) | 2026-10-19 | high | Full-system audit: pricing at the VIX, free hedging, two R2 gates, a z-score P&L | Portfolio P&L −$261,206 (C21 as published) → −$216,092 (NAV $783,908), Sharpe −1.80 → −0.88, max drawdown −31.5% → −21.6%, as published in C22; S1 +$693,938 → −$287,644. Its 15-step waterfall is replayed today on the C23-restated VIX record, so it ends at the C23 canonical run. |
| [C23](#c23) | 2026-10-20 → 2026-10-21 | high | Vendor data, the calibration's identification, spread-unit evidence and statistical inference | Portfolio P&L −$216,092 → −$222,177, Sharpe −0.88 → −0.99 (C23 waterfall, data_store/backtest/c23_bridge.csv: 8 VVIX sessions from Cboe −$6,502; 136 Cboe values +$417); 6 of 4,162 rule labels move; the portfolio Sharpe's 95% bootstrap interval is −1.81 to −0.30. The headline Heston fit is unchanged, now reported with its σ profile and rough Heston beside it. |

### C16

**Methodology audit: look-ahead, a circular feature, retrospective sizing** · 2026-06-11 · severity high · components: signals, classifier, sizing

- **Detection.** Self-audit of the methodology before the result was published.
- **Effect on the published result.** The published +19% headline was a label-noise artifact.
- **Commits.** [`47e1ffe`](https://github.com/navnoorthapar/vol-system-dashboard/commit/47e1ffebc910c5ad31c85c5bac0cb2511172a62b)

**Finding 1 (fixed): PDV look-ahead bias**

- Root cause: The PDV model was fitted on the full sample, so every backtest year traded a forecast that had seen its own future.
- Fix: PDVLinear refitted per backtest year on strictly earlier returns (walk-forward) before it feeds the pdv_iv_spread feature.
- Evidence: S1/S1C now trade a causal PDV forecast
- Pinned by: `joint_vol_calibration/tests/test_pdv.py::TestNoLookAhead::test_walk_forward_no_lookahead_poison`, `joint_vol_calibration/tests/test_pdv.py::TestNoLookAhead::test_poison_actually_bites_later`

**Finding 2 (fixed): Circular classifier feature**

- Root cause: vvix was a training feature while the R2 label is defined as VVIX above a threshold, so 86.2% accuracy partly measured rule recovery.
- Fix: vvix removed from the classifier's features; honest accuracy 63.4%.
- Evidence: Historical C16 figure: honest accuracy 63.4%; C17 then found it lost to the persistence baseline
- Pinned by: `joint_vol_calibration/tests/test_regime_classifier.py::TestVersionRobustPickle::test_shipped_classifier_is_built_on_the_canonical_gate`

**Finding 3 (fixed): Retrospective Kelly netting**

- Root cause: Correlation and Sharpe multipliers used same-day P&L to size the same day's positions.
- Fix: Multipliers shifted one session and applied to position size in a second simulation pass, so contracts and costs reflect the real size.
- Evidence: Ex-ante sizing; no same-day information
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestCorrelationNetting::test_engine_applies_the_overlay_one_session_late`

**Finding 4 (quantified): S1C data snooping**

- Root cause: S1C's direction was chosen after watching S1 fail on 2018–2025; that cannot be undone in code.
- Fix: Tested on 2013–2017, a window not used to design it (C17 re-ran it under corrected labels): S1C −$333K; the apparent edge was an artifact.
- Evidence: Pseudo-out-of-sample under C17: S1C −$333K — the apparent edge was an artifact
- Pinned by: no dedicated regression test (a gap)

**Finding 5 (quantified): S3 arbitrary P&L scale**

- Root cause: S3's P&L was an assumed 2% per z-unit (no single-stock options data).
- Fix: Sensitivity run at 0.5% / 1% / 2% / 4% per z-unit; C22 replaced the proxy with the index leg (a short SPX straddle).
- Evidence: C16 sensitivity run: P&L +$5K…+$52K, win 15–17 of 17 at every scale (before C18, C21 and C22 re-priced every trade and C22 replaced the proxy with the index leg)
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestS3::test_null_test_exposes_the_z_proxy_bias`

### C17

**Red-team: the classifier against a persistence baseline; margin; an unbounded tail** · 2026-06-11 · severity high · components: classifier, risk limits, signals

- **Detection.** Red-team review in the same week as C16.
- **Effect on the published result.** The classifier loses to 'predict yesterday's regime' and was demoted from the trading loop; portfolio −9.1% as published after this round.
- **Commits.** [`8c3ba9b`](https://github.com/navnoorthapar/vol-system-dashboard/commit/8c3ba9b650eeb8554a30bf8c1b5541306fa46abd)

**Finding 1 (fixed): Classifier never benchmarked against persistence**

- Root cause: Regime labels are same-day observables, so yesterday's label is a no-skill baseline; it scored 90.0% on 2020+ against the classifier's 63.4%.
- Fix: ML demoted to research; the backtest trades the lagged deterministic rule labels.
- Evidence: 63.4% vs 90.0% — the ML added negative value
- Pinned by: `joint_vol_calibration/tests/test_regime_classifier.py::TestEvaluateClassifier::test_persistence_baseline_reported`, `joint_vol_calibration/tests/test_regime_classifier.py::TestEvaluateClassifier::test_persistence_uses_previous_session_label`

**Finding 2 (fixed): Margin model documented but not enforced**

- Root cause: MARGIN_RATIO was defined but never applied, so short straddles had unlimited capacity.
- Fix: Short straddles capped at nav / (0.20 * K * 100) contracts at entry.
- Evidence: Cost model now matches its documentation
- Pinned by: `joint_vol_calibration/tests/test_backtest_engine.py::TestC17Fixes::test_margin_cap_on_short_straddles`

**Finding 3 (fixed): S1C had no R2 exit**

- Root cause: An open S1C short straddle could ride a VVIX spike for up to 21 days with no stop.
- Fix: S1C force-closes when the regime moves into R2, as S1X, S2X and S4 do.
- Evidence: Worst-case short-gamma exposure now bounded by the R2 rule
- Pinned by: `joint_vol_calibration/tests/test_signal_engine.py::TestGenerateSignal1Contrarian::test_r2_transition_forces_exit`

**Finding 4 (fixed): Dead monthly recalibration; a flat 5% hurdle**

- Root cause: The monthly Heston loop never produced a calibration (no pre-2026 options data) and nothing consumed its output; Sharpe used a flat 5%.
- Fix: Loop removed; Sharpe and Sortino measured against the per-date ^IRX path (later superseded in C22).
- Evidence: Backtest claims now match what the code does
- Pinned by: `joint_vol_calibration/tests/test_backtest_engine.py::TestC17Fixes::test_compute_metrics_accepts_rf_series`

### C18

**Straddles never rolled: one stale-strike position marked past expiry** · 2026-06-14 · severity high · components: backtest engine

- **Detection.** Word-by-word audit of the engine's position series: positions never returned to flat.
- **Effect on the published result.** Inverted the contrarian thesis: S1 −$1.53M → +$463K, S1C −$97K → −$404K, portfolio −9.1% → −19.0%; S3 unchanged.
- **Commits.** [`3d52f42`](https://github.com/navnoorthapar/vol-system-dashboard/commit/3d52f421dafa9eac13671d54e95ab5d0ec4a5f97)

**Finding 1 (fixed): Same-day re-entry kept one straddle alive**

- Root cause: The state machine re-entered on the day it exited, so the strike was fixed at first entry and the tenor decayed to its 1-day floor: a directional |S - K| bet, not a vol trade. Every P&L figure from C10 to C17 was contaminated.
- Fix: A forced flat day so positions roll into fresh 30-day ATM straddles.
- Evidence: Inverted the thesis: S1 −$1.53M → +$463K, S1C −$97K → −$404K, portfolio −9.1% → −19.0%; S3 unchanged
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestExpiryRoll::test_rolls_into_fresh_atm_straddles`, `joint_vol_calibration/tests/test_backtest_accounting.py::TestExpiryRoll::test_no_trade_outlives_its_expiry`

### C19

**Signal-engine logic fixes and data-gap repair** · 2026-06-20 · severity low · components: signals, data

- **Detection.** Follow-up review of the signal state machines after C18.
- **Effect on the published result.** Cleanup; the canonical numbers did not change.
- **Commits.** [`e1cfbd0`](https://github.com/navnoorthapar/vol-system-dashboard/commit/e1cfbd0eb0f5577d9e7ad79a5cedb9c772d69807)

**Finding 1 (fixed): S2 stop-loss not propagated; dead code; incomplete summaries; data gaps**

- Root cause: Stop-loss exits did not reach the S2 position; signal summaries skipped sleeves; market data had gaps.
- Fix: Stops propagate to the S2 position, dead code removed, summaries cover every sleeve, gaps repaired.
- Evidence: Cleanup — the canonical numbers did not change
- Pinned by: `joint_vol_calibration/tests/test_signal_engine.py::TestGenerateSignal2::test_short_straddle_stopped_when_front_vol_rises`, `joint_vol_calibration/tests/test_signal_engine.py::TestGenerateSignal2::test_long_straddle_stopped_when_front_vol_falls`

### C20

**The calibration selector rewarded broken data** · 2026-08-04 · severity high · components: calibration, dashboard

- **Detection.** The headline fit was traced to a day the VIX feed returned only ^VIX.
- **Effect on the published result.** The published 'best calibration' had been the day the VIX feed failed; fits are now gated on at least 3 VIX tenors with provenance recorded.
- **Commits.** [`35690a2`](https://github.com/navnoorthapar/vol-system-dashboard/commit/35690a261ddf7c944d28be2a9329736794307db6)

**Finding 1 (fixed): Archived fits ranked by SPX RMSE with no data-sufficiency check**

- Root cause: With one VIX tenor the VIX leg is one point Heston matches exactly, so under-identified joint fits scored best and headlined the site.
- Fix: Gate on >= 3 VIX tenors; Feller reported as PASS / BINDING / VIOLATED. C22 went further: the headline is the newest gate-passing fit, never the lowest RMSE.
- Evidence: The published “best calibration” had been the day the VIX feed failed
- Pinned by: `joint_vol_calibration/tests/test_joint_calibrator.py::TestCalibrationQualityGate::test_single_vix_tenor_rejected`, `joint_vol_calibration/tests/test_dashboard.py::test_selector_takes_newest_acceptable_fit_not_lowest_rmse`

### C21

**The cash hurdle was a test fixture; straddles marked a day too rich** · 2026-09-05 · severity high · components: backtest engine, data, daily bot

- **Detection.** Adversarial audit of the backtest inputs and the daily bot.
- **Effect on the published result.** Portfolio −19.0% → −26.1%, Sharpe −1.57 → −1.80, MaxDD −25.9% → −31.5%; S1 +$463K → +$694K, S1C −$404K → −$600K.
- **Commits.** [`15b4db5`](https://github.com/navnoorthapar/vol-system-dashboard/commit/15b4db5f98f05b9ed2bb9f8fa2c2b934a74097cf), [`90dc851`](https://github.com/navnoorthapar/vol-system-dashboard/commit/90dc851581a5197ce2eec126c78673293b2edb90)

**Finding 1 (fixed): ^IRX history inside the window was seven synthetic test rows**

- Root cause: test_rates.py wrote np.linspace(0.015, 0.020) into the tracked database and it was forward-filled for five years.
- Fix: Real ^IRX history loaded (1,809 rows); the suite now works on a private copy and cannot write the tracked database.
- Evidence: Part of the round's combined effect above; the two backtest fixes were not re-run separately
- Pinned by: `joint_vol_calibration/tests/test_c21_audit_fixes.py::TestRiskFreeHurdle::test_metrics_reject_a_stub_rate_series`, `joint_vol_calibration/tests/test_rates.py::TestIsolation::test_database_module_points_at_a_temp_file`

**Finding 2 (fixed): First holding day carried no time decay**

- Root cause: T_rem was computed from the pre-increment day count, so every mark valued the straddle one day too rich, flattering the short side.
- Fix: Time to expiry decays from the first holding day.
- Evidence: Part of the round's combined effect above; the two backtest fixes were not re-run separately
- Pinned by: `joint_vol_calibration/tests/test_c21_audit_fixes.py::TestStraddleTimeDecay::test_first_holding_day_decays`

**Finding 3 (fixed): The bot published classifier output as rule labels**

- Root cause: The daily refresh relabelled regime_labels.parquet with the demoted classifier and refit a frozen option surface.
- Fix: The bot writes deterministic rule labels and calibrates on a live chain.
- Evidence: The published labels are the rule's again
- Pinned by: `joint_vol_calibration/tests/test_c21_audit_fixes.py::TestRegimeLabelProvenance::test_refresh_labels_with_rule_not_classifier`

### C22

**Full-system audit: pricing at the VIX, free hedging, two R2 gates, a z-score P&L** · 2026-10-19 · severity high · components: backtest engine, signals, calibration, daily bot

- **Detection.** A subsystem-by-subsystem audit (about 270 findings, each reproduced before fixing) after the live calibration failed for a month.
- **Effect on the published result.** Portfolio P&L −$261,206 (C21 as published) → −$216,092 (NAV $783,908), Sharpe −1.80 → −0.88, max drawdown −31.5% → −21.6%, as published in C22; S1 +$693,938 → −$287,644. Its 15-step waterfall is replayed today on the C23-restated VIX record, so it ends at the C23 canonical run.
- **Commits.** [`a51eccc`](https://github.com/navnoorthapar/vol-system-dashboard/commit/a51eccc15d25b750c6a1523b3a53331b8596280d)

The C22 waterfall, one correction at a time (`data_store/backtest/c22_bridge.csv`, `python -m joint_vol_calibration.backtest.c22_bridge`). It is replayed on the C23-restated VIX record, so its last row is today's canonical run; the published C22 run ended at NAV $783,908, P&L −$216,092, Sharpe −0.88.

| Row | Correction applied (cumulative) | NAV | P&L | Δ P&L | Sharpe |
|---|---|---|---|---|---|
| 0 | C21 engine as published | $735,173 | −$264,827 |  | −1.84 |
| 1 | VIX tenors 9/30/93/184d, 365-day year | $734,650 | −$265,350 | −$524 | −1.84 |
| 2 | Sharpe on excess P&L, rf=0 | $734,650 | −$265,350 | $0 | −1.17 |
| 3 | ^IRX gaps: previous print | $734,650 | −$265,350 | $0 | −1.17 |
| 4 | S3 proxy exit-day P&L + cost | $733,308 | −$266,692 | −$1,341 | −1.18 |
| 5 | hedge open/unwind costs | $718,056 | −$281,944 | −$15,252 | −1.25 |
| 6 | premium + hedge financing | $738,598 | −$261,402 | +$20,542 | −1.15 |
| 7 | expiry settle / roll | $738,598 | −$261,402 | $0 | −1.15 |
| 8 | open trade in trade log | $738,598 | −$261,402 | $0 | −1.15 |
| 9 | size off sleeve NAV | $790,520 | −$209,480 | +$51,921 | −1.33 |
| 10 | real correlation netting | $806,452 | −$193,548 | +$15,932 | −1.33 |
| 11 | ATM vol = 0.82 × VIX | $914,322 | −$85,678 | +$107,870 | −0.37 |
| 12 | S3 = short SPX straddle (index leg) | $878,695 | −$121,305 | −$35,627 | −0.57 |
| 13 | R2 gate = upstream default (adaptive (rolling 80th pct)) | $739,855 | −$260,145 | −$138,841 | −1.28 |
| 14 | PDV spread: 21-day RMS-vol forecast (daily purged refit) | $777,823 | −$222,177 | +$37,968 | −0.99 |

**Finding 1 (fixed): ATM straddles priced at the VIX**

- Root cause: The VIX is a 30-day variance-swap rate that contains the skew premium; straddles were priced and marked at it.
- Fix: ATM vol = β × VIX, with β estimated from stored SPX chains.
- Evidence: C22 bridge row 11: portfolio P&L +$107,870, Sharpe −1.33 → −0.37 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestAtmVol::test_entry_vol_is_beta_times_vix`, `joint_vol_calibration/tests/test_backtest_reproducibility.py::test_default_beta_is_the_estimate_from_the_real_chains`

**Finding 2 (fixed): Two R2 gates and a hard-coded VVIX override**

- Root cause: Published labels used the adaptive gate while the backtest traded a fixed 100-point VVIX gate, and the signal engine OR-ed a VVIX > 100 override onto any labels.
- Fix: One adaptive gate (rolling 252-day 80th percentile of VVIX) everywhere; no override.
- Evidence: C22 bridge row 13: portfolio P&L −$138,841, Sharpe −0.57 → −1.28 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_regime_classifier.py::TestR2Gate::test_default_gate_is_adaptive`, `joint_vol_calibration/tests/test_signal_engine.py::TestSignalEngineGenerate::test_default_regimes_are_lagged_canonical_rule_labels`

**Finding 3 (fixed): Free hedging, no carry, sizing off $1M, no-op netting**

- Root cause: The opening hedge and unwind were free, the book earned no carry, size used $1M instead of the sleeve's NAV, and the netting factor was clipped into a no-op.
- Fix: Costs charged once each, premium and hedge financed, size off the running sleeve NAV, real netting.
- Evidence: C22 bridge rows 5, 6, 9, 10: portfolio P&L +$73,143, Sharpe −1.18 → −1.33 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestHedgeCosts::test_open_rebalance_and_unwind_charged_once_each`, `joint_vol_calibration/tests/test_backtest_accounting.py::TestFinancing::test_trade_financing_matches_formula`, `joint_vol_calibration/tests/test_backtest_accounting.py::TestBookkeeping::test_sizing_and_margin_use_running_sleeve_nav`, `joint_vol_calibration/tests/test_backtest_accounting.py::TestCorrelationNetting::test_legacy_multiplier_could_never_reduce_size`

**Finding 4 (fixed): S3 P&L was a z-score times an assumed 2%**

- Root cause: The 'survivor' was positive on 99.5% of random-walk null paths: it measured mean reversion of a ratio, not option P&L.
- Fix: S3 is the index leg of the dispersion trade (a short SPX straddle) on the same engine, with a timing null test.
- Evidence: C22 bridge rows 4, 12: portfolio P&L −$36,968, Sharpe −1.17 → −0.57 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestS3::test_index_leg_is_a_short_spx_straddle`, `joint_vol_calibration/tests/test_backtest_accounting.py::TestS3::test_null_real_statistic_is_the_engines_s3`

**Finding 5 (fixed): PDV spread compared the 30-day VIX with a 1-day forecast**

- Root cause: The pre-C22 forecast was next-day E|r|, so 'VIX > forecast + 2 vol pts' held on almost every session.
- Fix: The spread uses a 21-day RMS-vol forecast refitted daily on labels fully observed at the forecast origin.
- Evidence: C22 bridge row 14: portfolio P&L +$37,968, Sharpe −1.28 → −0.99 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_pdv21_wiring.py::TestBacktestEngine::test_canonical_horizon_is_21`, `joint_vol_calibration/tests/test_pdv.py::TestWalkForwardVolForecast::test_no_lookahead_poison`

**Finding 6 (fixed): Sharpe charged cash twice**

- Root cause: Option P&L on unfunded collateral is already an excess return; subtracting ^IRX again double-counted it.
- Fix: Sharpe uses rf = 0 (collateral assumed to earn T-bills), with the T-bill benchmark shown beside it.
- Evidence: C22 bridge row 2: portfolio P&L $0, Sharpe −1.84 → −1.17 (the bridge replayed on the C23-restated record)
- Pinned by: `joint_vol_calibration/tests/test_backtest_accounting.py::TestSharpeConvention::test_engine_does_not_charge_tbills_on_excess_pnl`

**Finding 7 (fixed): VIX indices fitted as futures; ATM-call anchors; one-expiry live fits**

- Root cause: The calibrator fitted spot VIX indices with the futures formula, anchored on adjacent ATM calls only, and the daily bot accepted fits from one SPX expiry (9 options) for a month.
- Fix: Spot-index VIX leg, forward-moneyness z-grid anchors with put-wing quotes, and a gate on expiries, options and the put wing.
- Evidence: Headline Heston fit replaced; one-expiry fits are rejected by the gate
- Pinned by: `joint_vol_calibration/tests/test_joint_calibrator.py::TestSyntheticMarket::test_vix_index_leg_zero_at_generating_params`, `joint_vol_calibration/tests/test_joint_calibrator.py::TestCalibrationQualityGate::test_single_expiry_rejected`, `joint_vol_calibration/tests/test_joint_calibrator.py::TestCalibrationQualityGate::test_missing_put_wing_rejected`

### C23

**Vendor data, the calibration's identification, spread-unit evidence and statistical inference** · 2026-10-20 → 2026-10-21 · severity high · components: data layer, calibration, surface, statistics, rates, daily bot

- **Detection.** A reviewer-standard pass over every subsystem in six parallel chains (surface, calibration, statistics, data, UI, engineering), each result measured on the five committed model_version 2 sessions and merged only with the full suite green.
- **Effect on the published result.** Portfolio P&L −$216,092 → −$222,177, Sharpe −0.88 → −0.99 (C23 waterfall, data_store/backtest/c23_bridge.csv: 8 VVIX sessions from Cboe −$6,502; 136 Cboe values +$417); 6 of 4,162 rule labels move; the portfolio Sharpe's 95% bootstrap interval is −1.81 to −0.30. The headline Heston fit is unchanged, now reported with its σ profile and rough Heston beside it.
- **Commits.** [`c63e515`](https://github.com/navnoorthapar/vol-system-dashboard/commit/c63e515f1c64b311d924d091ac8145bdd3fb6527), [`2566553`](https://github.com/navnoorthapar/vol-system-dashboard/commit/256655374fe2302942cc71e7594e6f2aa91fa7ac), [`958be69`](https://github.com/navnoorthapar/vol-system-dashboard/commit/958be6996b9ea0f837dc256638be5ffb787bccbf), [`49f5fd0`](https://github.com/navnoorthapar/vol-system-dashboard/commit/49f5fd03619dd1559f8e0a5185b8e9eb93ddc310), [`1c886e3`](https://github.com/navnoorthapar/vol-system-dashboard/commit/1c886e33525cfdb0322ad3c88fec25f8f63c2965), [`ca36d24`](https://github.com/navnoorthapar/vol-system-dashboard/commit/ca36d24448b9d3923f4b517499366eb8318366f5), [`9c486c3`](https://github.com/navnoorthapar/vol-system-dashboard/commit/9c486c388f9d274021d1d96473eedf628898a0cb), [`a678e57`](https://github.com/navnoorthapar/vol-system-dashboard/commit/a678e57c10a32d835ee261021a37ea7c971494ad)

The C23 waterfall: the published C22 run restated row by row on the Cboe record (`data_store/backtest/c23_bridge.csv`, `python -m joint_vol_calibration.backtest.c23_bridge`).

| Row | Correction applied (cumulative) | NAV | P&L | Δ P&L | Sharpe |
|---|---|---|---|---|---|
| 0 | C22 canonical as published (pre-C23 record) | $783,908 | −$216,092 |  | −0.88 |
| 1 | same 2-decimal prints restated as Cboe's (|change| <= 0.005) | $783,908 | −$216,092 | −$0 | −0.88 |
| 2 | term-structure rows on NYSE holidays removed | $783,908 | −$216,092 | $0 | −0.88 |
| 3 | VVIX sessions missing from Yahoo added from Cboe | $777,406 | −$222,594 | −$6,502 | −0.99 |
| 4 | term-structure values replaced by Cboe's (|change| > 0.005) | $777,823 | −$222,177 | +$417 | −0.99 |

**Finding 1 (fixed): The labels, backtest and calibrator read Yahoo's copies of the VIX index family**

- Root cause: Production overlaid Cboe's official closes on vix_daily only; every consumer read vix_term_structure, Yahoo's float32 copies with NULL provenance, NYSE-holiday rows and 8 VVIX sessions missing (forward-filled).
- Fix: Cboe's index files are the record (Yahoo only as a provisional same-evening row, replaced when Cboe posts); the database cache re-synced from it; the daily job writes Cboe first and reports cross-source differences above 0.25 pts.
- Evidence: 136 term-structure rows replaced (38 by more than 0.25 pts, 33 of them in the backtest window), 8 VVIX sessions added, 31 holiday rows removed; portfolio −$216,092 → −$222,177 (c23_bridge.csv); labels 2019-08-09, 2020-02-20, 2020-06-11, 2021-05-11 R1 → R2 and 2021-06-09, 2021-07-08 R2 → R1
- Pinned by: `joint_vol_calibration/tests/test_data_layer.py::TestC23RecordCorrection::test_apply_db_removes_holiday_rows_and_installs_the_cboe_values`, `joint_vol_calibration/tests/test_cloud_refresh.py::test_vix_term_structure_is_cboe_primary`, `joint_vol_calibration/tests/test_backtest_reproducibility.py::test_c23_bridge_reproduces_every_row`

**Finding 2 (quantified): The σ ≤ 2 box, not the market, sets the headline fit**

- Root cause: σ sits on its 2.0 ceiling in 2 of the 5 model_version 2 fits (2026-03-24, 2026-10-16); the page said 4 of 5 parameters are set by the market without saying what the box costs.
- Fix: A σ profile (the joint loss refitted under caps 1 to 10, κ ≤ 60) and Gauss–Newton identifiability are stored per session and shown; the production box is kept (the NN domain, the vomma shock sizing and the gate depend on it) and its cost is published.
- Evidence: 2026-10-16: freeing σ gives σ 6.90, κ 27.4, loss −27.7%, out-of-sample RMSE 2.96 → 1.19 vol pts and model VVIX 205 → 113 against the market's 110; on 2026-03-24 the box costs 2.1% of the loss (data_store/calibrations/diagnostics/)
- Pinned by: `joint_vol_calibration/tests/test_identifiability.py::test_sigma_box_rationale_in_config_matches_the_artifacts`, `joint_vol_calibration/tests/test_identifiability.py::test_diagnostics_artifact_is_consistent`

**Finding 3 (fixed): Model error was never measured against the bid–ask, and quotes were not screened for static arbitrage**

- Root cause: No per-quote residuals were stored, the calibration chart drew the model only, and nothing screened the stored quotes for static arbitrage.
- Fix: A surface package: Cohen–Reisinger–Wang screen and l1 repair on HiGHS, an executable-arbitrage certificate, eSSVI and calendar-ordered SVI surfaces, and one scorecard that scores every model on the calibrator's out-of-sample quotes in half-spreads.
- Evidence: Out-of-sample median |model − mid| over the 5 sessions: Heston 4.4–15.6 half-spreads (3.0–12.3% inside the bid–ask); the arbitrage-free SVI surface 0.45–1.98 (32–80% inside); executable static arbitrage on all 5 sessions, 160–629 quotes per session in an executable row (data_store/surface/)
- Pinned by: `joint_vol_calibration/tests/test_surface_arbitrage.py::test_committed_session_screen_counts`, `joint_vol_calibration/tests/test_surface_report.py::test_spread_unit_claims_hold`, `joint_vol_calibration/tests/test_surface_report.py::test_committed_artifacts_match_a_recomputation`

**Finding 4 (fixed): The backtest had no statistical inference**

- Root cause: Sharpe ratios and a 200-path S3 null were published without intervals, deflation for the configurations tried since C10, or exposure-matched nulls for every sleeve.
- Fix: Stationary-bootstrap intervals, PSR/DSR against a committed trials ledger, Hansen SPA and StepM, CSCV PBO, exposure-matched circular-shift and trade-shuffle nulls for every sleeve, and forecast comparisons (QLIKE, Diebold–Mariano, MCS).
- Evidence: Portfolio Sharpe −0.99, 95% CI −1.81 to −0.30; S3 timing p 0.078 against 3,550 circular shifts (Holm over 6 sleeves 0.47); Hansen SPA p 0.65 for the published series against cash; S3's largest DSR over every trial count 0.27 (data_store/backtest/inference/summary.json)
- Pinned by: `joint_vol_calibration/tests/test_stats_inference_report.py::test_headline_is_generated_from_the_stored_numbers`, `joint_vol_calibration/tests/test_stats_inference_report.py::test_bootstrap_rebuilds_exactly_from_the_committed_curve`, `joint_vol_calibration/tests/test_stats_nulls.py::test_circular_null_detects_real_timing_and_not_luck`

**Finding 5 (fixed): Flat rates in the hedger and the Greeks surface**

- Root cause: The C7 hedger hard-coded R = 1.5% ('2020 US Treasury'), and the Greeks surface used one median r and q for every maturity, so its 1-year forward was 0.50% below that expiry's parity forward.
- Fix: A daily Treasury zero curve (bills and par yields, look-ahead gated, refused when more than 7 days old) feeds the calibrator's reference, the chain cleaner, the C7 study and per-maturity Greeks carry.
- Evidence: C7 cumulative P&L 167.95 → 168.34 index points; Greeks surface 1-year forward 7,931.8 → 7,971.7 (the parity forward); the committed fits recorded ^IRX and are unchanged
- Pinned by: `joint_vol_calibration/tests/test_rate_reference.py::test_c7_uses_the_one_year_zero_rate_at_entry`, `joint_vol_calibration/tests/test_rate_reference.py::test_greeks_carry_reprices_the_parity_forwards_on_the_treasury_curve`

**Finding 6 (fixed): A presentation test could lose a captured session**

- Root cause: The daily job committed the chain and the fit only if the dashboard tests passed; a test that pinned a value of the committed data failed on the first new fit, and the 2026-10-19 overnight book (accepted three times) was lost.
- Fix: Separate gates: data tests gate the capture commit, artifact tests the derived artifacts, presentation tests only the deploy; CI simulates the bot's next session before merge.
- Evidence: No published number moved; the simulated next session (scripts/ci/simulate_next_session.py) found 5 more tests that would have blocked the deploy on the first accepted fit
- Pinned by: `joint_vol_calibration/tests/test_infra_config.py::test_a_presentation_test_cannot_lose_the_captures`, `joint_vol_calibration/tests/test_infra_config.py::test_each_commit_adds_its_own_paths`

<!-- END GENERATED -->

## Narrative notes

### The C18 inversion (C16–C22)

The centrepiece of the first five rounds. I had built a contrarian signal (S1C) on the premise that the base signal S1
was a "−$1.53M catastrophe". That catastrophe was the strike-rolling bug: same-day re-entry meant positions never
rolled, so the engine held one stale-strike straddle marked past expiry, and every P&L figure from C10 to C17 was
contaminated. Corrected, S1 looked like a profitable regime-gated short-VRP harvester (+$463K after C18, +$694K after
C21). C22 took that away too: priced at a realistic ATM vol (0.82 × VIX instead of the VIX), with real hedge and carry
costs, one consistent R2 gate and a horizon-matched PDV forecast, S1 lost $288K as published in C22 (−$307K on the C23
record). The "edge" was the skew premium inside the VIX, booked as if it could be sold at the money. A thesis that
holds only under a modelling shortcut is not a signal.

Two C22 bridge rows dominate, in opposite directions. β = 0.82 helps the portfolio, because S1C and S4 are mostly long
straddles, which were bought "at the VIX" too dear. The single R2 gate hurts: the backtest had traded a fixed
VVIX > 100 gate (about 48% of the window in R2) while the published labels used the adaptive gate (about 23%), so
moving to one gate puts a quarter of the window back into R1, where S1C buys straddles.

S1C's pseudo-out-of-sample check (2013–2017, a window not used to design it) failed under the C22 engine: −$261,085,
Sharpe −0.87, 29% win, 58 trades (`run_backtest --pseudo-oos`, as published in C22).

### The S3 null tests (C22, C23)

S3 is priced as the index leg of a dispersion trade (long dispersion ⇒ short SPX straddle) on the same engine, a proxy
because there is no single-stock options data. C22 tested its timing against 200 random-walk null paths of ^VVIX
(p = 0.075 as published; 0.065 on the C23 record, `full_results.pkl`). C23 replaced that with exposure-matched nulls for
every sleeve (3,550 circular shifts and 2,000 trade shuffles that keep each sleeve's own position and size series):
S3's p is 0.078 (Holm over the six series 0.47). The pre-C22 "+$23K, positive at every scale" figure was a z-score times
an assumed 2%, positive on 99.5% of random-walk null paths: it measured mean reversion of a ratio, not option P&L. An
earlier C22 draft reported p = 0.035; that null re-built S3 from the backtest-window features only, a different S3 from
the one the engine trades, and was fixed before publication.

### Why the headline fit is chosen on data sufficiency, not RMSE (C20, C22)

The daily job recalibrates on a live snapshot and the dashboard headlines one archived fit. Ranking archived fits on
SPX RMSE was wrong in production twice:

- **C20.** Yahoo intermittently returned `^VIX` without `^VIX9D/3M/6M`. With one tenor the VIX leg is one data point
  that Heston matches exactly, which frees all five parameters to fit the SPX smile alone, so the fits with broken data
  had the best RMSE (2026-07-23: one tenor, 0.049 vp, headlined the site). Fix: ≥ 3 VIX tenors required.
- **C22.** The same failure on the SPX side. A fit from one SPX expiry (9 options) has a tiny RMSE because there is
  nothing to be jointly consistent with; the selector headlined 2026-09-11 (0.215 vp, 9 options).

The gate (`is_acceptable_calibration`) requires ≥ 4 SPX expiries, ≥ 30 options, a put wing in ≥ 3 expiries, ≥ 3 VIX
tenors, a snapshot dated to the spot session, and `model_version = 2`. The dashboard headlines the newest passing fit
and shows its age in NYSE sessions. The 37 pre-C22 `joint_cal_*.pkl` files are kept as labelled legacy evidence and are
never ranked.

### The live calibration outage, 2026-09-05 → October (C22)

From 2026-06-16 to 2026-09-04 the bot never stored the chain it downloaded, so every "daily" fit refitted the newest
committed SPX snapshot. The C21 fix made it use a live chain and exposed a second set of defects: from 2026-09-07
every Monday–Thursday run was rejected at 176–273 vp, and the six fits it accepted each used 9 options on a single
expiry. All runs were green. Root causes, all fixed in C22:

1. Only the first 8 listed expiries were fetched (about 1.5 weeks of SPXW dailies). Now the nearest listed expiry to
   each target (14/30/60/91/180/365 days) is fetched.
2. The cron fired into the empty after-hours book; the code kept Yahoo's implied vol and fell back to `lastPrice`. Now
   only two-sided quotes are kept, IVs are computed on parity forwards, and a book with < 80% two-sided near-ATM lines
   fails closed with a reason.
3. `cloud_refresh.main()` ignored every return value, so a rejected calibration was a green run. It now returns an exit
   code, fails the job on hard failures or a headline older than 3 sessions, and writes a status file.
4. The chain was stored before the gate ran, and `r` was overwritten after the fit with a rate the fit never used.

C23 closed the last hole in that chain of custody: a presentation test could still fail the commit of an accepted
session (finding 6 of C23 above).

### What C22 changed in the calibration story

The pre-C22 README reported a 2026-05-31 showcase fit (κ 1.77, θ 0.067, σ 0.46, ρ pinned at −0.95, Feller passed) and
concluded that Heston fitted the joint surface with three effective degrees of freedom and that a 37.14 vol-pt VIX-options
RMSE was a structural failure of the CIR variance. None of that survived C22:

- **ρ at −0.95 was an artifact of the anchors**: 45 of 54 anchors were ATM calls, with no put wing beyond the shortest
  slice. The anchors are now the strikes nearest a grid in z = ln(K/F)/(σ_ATM√T) ∈ [−2.5, 1.5], OTM against the parity
  forward, with ≥ 3 wing puts per expiry; ρ lands at −0.66 … −0.82 on every session.
- **The binding Feller constraint was self-imposed** by a default penalty. Feller is not a no-arbitrage condition; it
  is reported as a diagnostic.
- **The VIX leg fitted the wrong instrument**: the spot-starting variance-swap indices were compared with the model's
  futures curve. The leg now uses the index formula; real VX futures are scored out of sample.
- **Carry and time were wrong**: flat r and q, Yahoo's IVs on a parity-violating forward, and a Sunday snapshot that
  made every T two days short.
- **The 37.14 vp came from a broken pricer** (VIX puts by parity with √E[VIX²], vegas on spot VIX, a stale chain). On
  the 2026-03-24 session the Quintic OU's 61 VIX options reprice at 35.2 vp under Heston against the Quintic's 2.77 vp,
  but Heston was not fitted to them, and on the Quintic's 66 SPX targets Heston is better (1.21 vs 1.31 vp). The stored
  Quintic fit did not converge (iteration limit; ε on its −0.99 bound).

C23 added the σ profile, Gauss–Newton identifiability and a rough-Heston comparison on the same quotes (C23 finding 2;
the Calibration page's "Heston vs rough Heston" panel).

### The regime classifier (C16, C17)

The labels are defined by same-day observables (rv₂₀ against VIX for R0/R1; VVIX above its rolling 252-day 80th
percentile for R2), so "predict yesterday's regime" is a legitimate causal baseline, and it beats the XGBoost
classifier out of sample (the shipped model's own evaluation, on the Research page). The classifier is research-only;
the backtest trades the lagged rule labels. Perturbing the inputs within their measurement noise flips 5.9% of labels,
and 10.2% of days sit within one vol point of the R0/R1 cut (`regime_label_noise_audit()`, as published in C22).
