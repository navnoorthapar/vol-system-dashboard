# C22 correction round — handoff

Working notes for continuing the C22 audit/fix round in a fresh session.
**Delete this whole `docs/c22-handoff/` folder before merging to `main`.**

## Where things stand

1. **Audit: done.** 15 specialist auditors produced about 270 findings, and each one carries its file:line locations,
   evidence and proposed fix. They are in `findings/find_<area>.json`. They are **unverified** except
   `verify_tests.json` and `verify_hygiene.json`. Every fixer must reproduce a finding before fixing it.
2. **Stage A (per-subsystem fix chains): partly done.** Each chain worked on its own branch, with exclusive file
   ownership, starting from base commit `3462e2c` (this branch's config commit). The exact prompts, file ownership
   and decisions for each chain are in `fix-chain-prompts.js`. Two chains have completion reports in `chain-reports/`.

| Branch | State | Notes |
|---|---|---|
| `c22/data` | **done** (5 commits, report `chain-reports/data_data-layer.json`) | Session-dated chains (weekend snapshots re-dated 03-28→03-27, 05-31→05-29, 06-20→06-18), NYSE calendar, idempotent snapshots, ^IRX discount→cc rate conversion (negative bills kept), rebuilt Cboe downloader + real VX futures (30,963 rows), parquet persistence helpers, DB migrated (only this branch may change `vol_system.db`). `backfill_calibration_provenance.py --apply` **not yet run**. |
| `c22/signals` | **done** (5 commits, report `chain-reports/signals_signals.json`) | One adaptive R2 gate, no VVIX>100 override, S2 stop direction fixed, S1 exit keyed to entry side, NaN-VVIX sessions kept, causal HMM, OOS calibration, mutation-proof look-ahead tests, `build_features(pdv_vol_series=…)` hook, orphan parquets deleted. |
| `c22/backtest` | commits made, **no final report** (interrupted) | β·VIX ATM proxy, hedge/carry accounting, S3 index-leg pricing + null test, rf=0 Sharpe, C7 hedger fixes, `c22_bridge.py`. Verify the suite passes and review the commits. |
| `c22/pdv` | commits made, **no final report** | Label-window/GARCH/kernel/stress-test fixes, 21-day RMS-vol forecast (`walk_forward_vol_forecast`), rebuilt `pdv_model.pkl`, regime_pdv fixes. Verify. |
| `c22/quintic` | 2 commits, **no final report** | Adapted (Itô) MC, vectorised VIX pricer, parity-forward targets, session guards. The recalibrated pickle may not be done; check. |
| `c22/dashboard` | step 1 (backend) committed (6 commits), **no report**, step 2 (UX/a11y) **not started** | Read-only rendering, visible loader errors + fail-closed freeze, newest gate-passing fit, engine metrics, Greeks grid, PDV percentile. Step 2 still to do: WCAG AA, headings/landmarks, chart text alternatives, mobile, print, regime copy. |
| `c22/nngreeks` | 2 commits + **1 WIP commit (unverified)** | `implied_vol.py` (robust IV solver), risk_monitor fixes done; the WIP is in-progress NN pricer changes. Finish: NN train/benchmark/domain, re-export old helper names. |
| `c22/calib` | **1 WIP commit only (unverified)** | Heston engine work in progress (`heston.py`, tests, `test_heston_reference.py`). Step 1 (engine) needs finishing. **Step 2 (joint calibrator + live pipeline: the production outage) has not started** and is the top priority. |

## Decisions already made (keep them)

- **VIX leg:** ^VIX9D/^VIX/^VIX3M/^VIX6M are spot-starting indices. Fit them with `heston_vix_term_index` at
  `config.VIX_INDEX_TENOR_DAYS` (9/30/93/184). Exact VIX futures via the CIR Laplace transform, used only for real VX futures.
- **Live pipeline:** pick the listed expiry nearest each target (14/30/60/91/180/365d). Keep only two-sided quotes, with
  no `lastPrice` fallback. Derive per-expiry parity forwards and discounts, and compute our own Black-76 IVs. Fail
  closed on an empty after-hours book. Anchors come from a forward-moneyness grid with a real put wing.
- **Feller:** penalty is opt-in and off by default; the margin is reported as a diagnostic. Report active constraints
  honestly.
- **Gate:** SPX sufficiency (at least 4 expiries, 30 options, a put wing), VIX at least 3 tenors, snapshot == session, and
  `model_version=2` with `calibration_is_legacy()`. The dashboard headlines the **newest** passing fit and shows a
  staleness banner.
- **Backtest:** σ_ATM = β·VIX (β estimated from real chains, about 0.85), with a sensitivity table. One adaptive R2 gate.
  Sharpe on excess P&L with rf=0 (collateral assumed to earn ^IRX). S3 priced as the index straddle leg, plus a
  null-test p-value. The PDV spread uses the 21-day RMS-vol forecast.
- **Cron:** split the job. Market data and labels run after the close. The option chain and calibration run when the
  Cboe book is live (about 01:30 UTC, with a retry), because the current run lands in the empty-book window.
- C22 is reported like C16–C21: a before/after table and a waterfall (`c22_bridge.py`) in the README and on the
  dashboard. Numbers will move a lot (S1's profit largely disappears at β≈0.85). Report that honestly.

## Remaining work, in order

1. Finish `c22/calib`: Heston engine (step 1), then calibrator + live pipeline (step 2). This fixes the outage.
2. Verify and finish `c22/backtest`, `c22/pdv`, `c22/quintic`, `c22/nngreeks`. Run dashboard step 2 (UX/a11y).
3. **Integrate:** merge all `c22/*` into `claude/festive-brown-un4n4s`. Suggested order: data, signals, pdv, calib,
   quintic, nngreeks, backtest, dashboard. Resolve conflicts and run the full suite after each merge.
4. **Stage B:** wire the PDV-21 forecast into `build_features`, the backtest and the dashboard. Re-run the canonical backtest,
   the sensitivity grid and `c22_bridge.py`, and commit `data_store/backtest/*`. Run the provenance backfill `--apply`.
   Recalibrate the committed snapshots with the corrected calibrator (new `model_version=2` pickles). Regenerate the
   Greeks surface from the headline fit, plus a like-for-like Heston vs Quintic comparison. Then infra:
   `tests/conftest.py` DB isolation, `pyproject.toml`, a pinned lock/constraints file (pickles from sklearn 1.7.1),
   CI (actions v5/v6, concurrency group, rebase-retry push, split cron, step summary, CPU-only or no torch in the
   daily job, run tests before publishing, persist daily captures with the data-layer parquet helpers), drop the dead
   Procfile, fix `redeploy_site.sh`, print→logging, and harden tests (mutation survivors in `find_tests.json`).
5. **Stage C:** README, `substack_article.md` (add a dated correction note) and the dashboard copy, all matched to the
   final numbers. Also the data-sources table, test counts, component list, citations, and the C22 entry in the
   corrections log.
6. Independent review of the full diff (code-review at high effort), full suite green, push, and open a PR.

## Rules for fixers

Reproduce each finding before you change code. Every fix gets a regression test that fails on the old code. Never
commit unintended `data_store/` mutations made by tests (check `git status`). Don't edit README/substack until
Stage C. End commit messages with the session's attribution lines.
