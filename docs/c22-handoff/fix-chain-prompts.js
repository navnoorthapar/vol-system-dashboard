export const meta = {
  name: 'vol-system-c22-fix',
  description: 'C22 correction round: per-subsystem fix chains, each in its own git worktree with exclusive file ownership',
  phases: [
    { title: 'Fix', detail: 'each chain verifies its findings, fixes them with tests, commits on its own branch' },
  ],
}

const SP = '/tmp/claude-0/-home-user-vol-system-dashboard/017612b3-569a-51ba-baa9-4b3f10b4b51c/scratchpad'
const WT = `${SP}/wt`
const FD = `${SP}/findings`

const COMMON = `
PROJECT: "Joint SPX/VIX Smile Calibration System" - a public quant research repo (Heston/Bates joint SPX/VIX calibration,
Quintic OU, Guyon-Lekeufack-style PDV, NN surrogate pricer, Greeks monitor, regime classifier, signal engine, delta-hedged
straddle backtest 2018-01-01..2025-03-24, Flask dashboard frozen to static HTML and deployed daily by a GitHub Action).
Its stated ethos: "the audit trail of corrections is the deliverable" - every correction is reported honestly even when
it makes results worse. A 15-auditor review just produced ~245 UNVERIFIED findings (JSON files in ${FD}/find_<auditor>.json,
each with id, severity, locations, description, evidence, proposed_fix). We are now executing correction round "C22".

HOW TO WORK
1. You work ONLY inside your git worktree (given below), on its branch. Never modify /home/user/vol-system-dashboard or any
   other worktree under ${WT}. Run all commands (pytest, scripts, calibrations, backtests) inside your worktree.
   Use ${SP}/work-<your-chain> for scratch files.
2. Read your assigned findings from the JSON files. They are UNVERIFIED: for EACH one, first reproduce it (re-run its
   evidence or write a minimal check). If it is wrong, overstated, or already fixed, do NOT change code for it - record
   status "refuted" with the reason. If real, fix it PROPERLY at the root (no band-aids, no silencing), keep public APIs
   backward compatible unless the API itself is the bug (then update all callers you own and report callers you don't own).
3. Every real fix gets a regression test that FAILS on the old code and PASSES on the new code (verify this by temporarily
   reverting or by mutation). No tautological tests, no huge tolerances, seeded randomness, no network in tests (use small
   captured fixtures under joint_vol_calibration/tests/fixtures/ if needed).
4. FILE OWNERSHIP: modify/create only the files listed as yours. If a correct fix needs a change in a file you do not own,
   do not make it - put a precise request (file, function, exact change, why) in cross_chain_requests.
   Do not edit README.md or substack_article.md (a later docs stage owns them) - instead list every published claim your
   fixes invalidate in published_number_changes (what, old value, new value, where published).
   Do not add pip dependencies; if one is genuinely needed, explain in notes_for_integration.
5. DATA ARTIFACTS: tests and scripts can mutate tracked files under data_store/ (SQLite DB, parquet, pickles). Before every
   commit run \`git status --porcelain\` and restore anything you did not intend to change
   (\`git checkout -- <path>\`). Only commit artifact files explicitly allowed for your chain.
6. Before your final commit the FULL suite must pass in your worktree:
   \`python -m pytest joint_vol_calibration/tests -q -p no:cacheprovider\` (baseline: 642 passed, 24 skipped). If a test
   you do not own breaks because of an intended behaviour change, fix the test only if it is in your ownership; otherwise
   explain it in cross_chain_requests and notes_for_integration.
7. Commit in logical, reviewable commits with the repo's style of message: a concise subject, then a body explaining what
   was wrong, the evidence, the fix, and any number that moved (old -> new). End EVERY commit message with exactly:

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KtGbaSszHTT6nRGRYkt347

   Do not push. Do not create branches other than your own. Do not mention any model name elsewhere in commits.
8. Match the surrounding code style (docstring density, naming, comment tone). Prefer clarity and correctness over cleverness.
   Update docstrings/comments that make claims your fix changes. Be honest in code comments: no claims the code does not do.
9. Be exhaustive within your scope. If while working you discover NEW defects in your files, fix them too and list them
   under new_findings.
`

const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          status: { type: 'string', enum: ['fixed', 'partially-fixed', 'refuted', 'deferred'] },
          summary: { type: 'string', description: 'what you verified and what you changed (or why not)' },
          tests: { type: 'string', description: 'tests added/changed that pin it' },
        },
        required: ['id', 'status', 'summary', 'tests'],
      },
    },
    new_findings: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, description: { type: 'string' }, status: { type: 'string' } }, required: ['id', 'description', 'status'] } },
    cross_chain_requests: { type: 'array', items: { type: 'object', properties: { target_file: { type: 'string' }, request: { type: 'string' } }, required: ['target_file', 'request'] } },
    published_number_changes: { type: 'array', items: { type: 'object', properties: { what: { type: 'string' }, old: { type: 'string' }, new: { type: 'string' }, where: { type: 'string' } }, required: ['what', 'old', 'new', 'where'] } },
    api_changes: { type: 'string', description: 'new/changed public functions other chains or integration must know about' },
    commits: { type: 'array', items: { type: 'string' }, description: 'sha + subject, oldest first' },
    test_result: { type: 'string', description: 'final full-suite pytest summary line in your worktree' },
    notes_for_integration: { type: 'string' },
  },
  required: ['verdicts', 'new_findings', 'cross_chain_requests', 'published_number_changes', 'api_changes', 'commits', 'test_result', 'notes_for_integration'],
}

const CHAINS = {
  calib: [
    {
      label: 'heston-engine',
      prompt: `YOUR CHAIN: calib, step 1 of 2 ("Heston/Bates engine"). Worktree: ${WT}/calib (branch c22/calib).
FILES YOU OWN: joint_vol_calibration/models/heston.py, joint_vol_calibration/tests/test_heston.py,
new joint_vol_calibration/tests/test_heston_reference.py.
FINDINGS: every finding in ${FD}/find_heston.json (fix the engine side; calibrator-side wiring of the VIX index leg is
step 2's job). Also check ${FD}/find_jointcal.json "jointcal-bates-bounds-exclude-heston" - fix only if the bounds live in
heston.py.
REQUIRED API (step 2 and other chains depend on these exact names; keep every existing public name working - grep the repo
for imports of heston.py):
 - heston_vix_term_index(kappa, theta, sigma, v0, T) -> VIX-style index level in vol points for spot-starting horizon(s)
   T in years: 100*sqrt(theta + (v0-theta)*(1-exp(-kappa*T))/(kappa*T)), vectorised over T, correct T->0 and kappa->0
   limits. This is what ^VIX9D/^VIX/^VIX3M/^VIX6M measure (see joint_vol_calibration/config.py VIX_INDEX_TENOR_DAYS).
 - heston_vix_future_exact(kappa, theta, sigma, v0, T, tau=30/365) -> exact E[VIX_T] (CIR Laplace-transform integral,
   numerically stable). heston_vix_futures_curve must return exact futures; keep the old sqrt(E[VIX^2]) as an explicitly
   named heston_vix_futures_upper_bound.
 - Bates VIX index/futures formulas must include the jump contribution to model-free variance where the model implies it
   (research the correct expression; document it).
 - heston_call_batch accurate to < 0.01 vol pt vs an independent reference (Lewis 2001 integral with scipy.quad to inf) for
   T in {1,3,7,14,30,91,365} days, K/F in [0.7,1.3], including the archived parameter sets in
   data_store/calibrations/joint_cal_*.pkl (v0 down to 1e-4). Choose truncation adaptively or switch method; handle
   moment explosion for the Carr-Madan damping alpha. Do not call it "FFT" unless an FFT is actually used.
 - Monte Carlo: full-truncation Euler or Andersen QE, documented; the MC<->analytic cross-check test must be meaningful.
 - Fix calibrate_to_spx / HestonModel.calibrate weight indexing, greeks units (document per-day vs per-year, per vol pt),
   implied_vol_from_price consistency (NaN for ill-posed, never a silent absurd IV).
 - Replace tautological tests with independent-reference tests.
Do NOT commit any data_store changes.`,
    },
    {
      label: 'calibrator-live',
      prompt: `YOUR CHAIN: calib, step 2 of 2 ("joint calibrator + live daily pipeline"). Worktree: ${WT}/calib (branch c22/calib).
Step 1 just committed Heston-engine fixes on this branch - read its commits (git log) first; use heston_vix_term_index,
heston_vix_future_exact etc.
FILES YOU OWN: joint_vol_calibration/calibration/joint_calibrator.py, joint_vol_calibration/calibration/__init__.py,
scripts/cloud_refresh.py, new joint_vol_calibration/data/option_chain.py, joint_vol_calibration/tests/test_joint_calibrator.py,
joint_vol_calibration/tests/test_data_pipeline.py, new joint_vol_calibration/tests/test_option_chain.py,
new joint_vol_calibration/tests/test_cloud_refresh.py, and ONLY the cloud_refresh/calibrator-related test functions inside
joint_vol_calibration/tests/test_c21_audit_fixes.py if your behaviour change breaks them.
FINDINGS: every finding in ${FD}/find_livecal.json and ${FD}/find_jointcal.json; the calibrator/pipeline side of
${FD}/find_heston.json "heston-vix-leg-spot-index-vs-forward", ${FD}/find_data.json "data-vix-ts-index-fit-as-futures",
"data-rq-units-and-provenance", "data-get-options-surface-staleness", "data-snapshot-date-wallclock-weekend" (pipeline side);
${FD}/find_ops.json "ops-silent-degradation" (script side), "ops-import-side-effects", "ops-session-not-verified-closed",
"ops-log-ordering", "ops-commit-metadata" (script side).
DECISIONS ALREADY MADE (implement these):
 - Expiry selection: for each target tenor in (14,30,60,91,180,365) days pick the listed expiry (>=7d) nearest the target,
   unique, within 50%; download only those (optionally +1 neighbour each side). Single source of truth for the targets.
 - Chain hygiene in option_chain.py: two-sided quotes only (bid>0, ask>=bid, relative spread cap), never fall back to
   lastPrice, per-expiry parity forward F_T and discount D_T from a regression of (C-P) on K over near-ATM two-sided strikes,
   own Black-76 IVs from (mid, F_T, D_T); Yahoo impliedVolatility kept only as a diagnostic. A quote-state gate: if the near-
   ATM two-sided share is too low (empty after-hours book), fail closed with a clear reason (and a ::warning:: in CI).
 - Anchors: per expiry, strikes on a standardised forward-moneyness grid z = ln(K/F)/(sigma_atm*sqrt(T)) (e.g.
   linspace(-2.5,1.5,9)) or an equivalent delta grid, OTM defined against F_T (not spot), at least 3 puts with K/F<0.97
   wherever quotes exist. Keep vega-normalised errors.
 - Rates: use the per-expiry parity D_T/F_T in the SPX leg; store the r/q actually used (per expiry + summary) in the result;
   delete the post-hoc result["r"] overwrite in cloud_refresh.
 - VIX leg: fit ^VIX9D/^VIX/^VIX3M/^VIX6M with heston_vix_term_index at config.VIX_INDEX_TENOR_DAYS (they are spot-starting
   indices, not futures). Name the leg/RMSE honestly (e.g. "vix_index_rmse") and keep "vix_futures_rmse" as a back-compat
   alias key in the result because the dashboard reads it. If true VX futures exist in the DB (table vix_futures_*), you may
   add them as a separate, optional leg priced with heston_vix_future_exact; it must be a no-op when the table is empty.
 - Feller: make the penalty opt-in (feller in {"off","soft"}, default "off" - Feller is not a no-arbitrage condition and the
   pricer is valid when it binds); always report the margin and PASS/BINDING/VIOLATED as a diagnostic; replace the
   "Pinned: rho, sigma at boundary" message with an honest list of ACTIVE constraints (box bounds hit; Feller only when it is
   imposed and binding); fix count_pinned_params' tolerance.
 - Quality gate is_acceptable_calibration: add SPX data sufficiency (>=4 expiries, >=30 options, put wing present), the
   existing >=3 VIX tenors, surface freshness (snapshot == session), and a model_version check: results written by this new
   code carry model_version=2 (and a dict of method choices); provide calibration_is_legacy(cal) so the dashboard can label
   or exclude pre-C22 pickles. Measure and store an OUT-OF-SAMPLE IV RMSE over the full cleaned OTM surface alongside the
   in-sample anchor RMSE.
 - SVI/SSVI: the smoothing code is dead on the production path - remove the dead code and the claim (list the claim's
   locations in published_number_changes), unless you wire it in properly with tests.
 - VIX options leg: fix what the findings show is wrong, keep it disabled (w3=0) and document why.
 - cloud_refresh.py: no import-time side effects (chdir/logging in main()); CLI to run steps selectively
   (e.g. --steps market,labels,calibrate,freeze) so the workflow can split market data (after close) from the option/
   calibration step (run when the book has live quotes); explicit exit status: non-zero for hard failures (empty SPX
   download, expected NYSE session missing, freeze failure); rejected calibration => ::warning:: + reason, zero exit unless
   the latest accepted calibration is older than a configurable number of sessions (then non-zero); write a Markdown
   summary to $GITHUB_STEP_SUMMARY when set and a machine-readable status JSON (path configurable, e.g. .refresh-status.json
   at repo root, gitignored) containing session date, steps run, accepted/rejected + reason, headline calibration date and
   age, so the workflow can write accurate commit messages; never persist a rejected/garbage chain as an options snapshot
   used by later fits (or mark it unusable); label snapshots by the NY session date; route print() output through logging.
 - PROVE IT: in your worktree, fetch the LIVE ^SPX chain from Yahoo (network works) and run the new pipeline end to end.
   Report n_expiries, n_options, puts/calls, params, in-sample and OOS RMSE, VIX index RMSE, Feller margin, r/q used, and
   whether the gate accepted. Also re-run the corrected calibrator on the committed 2026-05-31 snapshot and report the
   numbers vs the published showcase (kappa 1.77, theta 0.067, sigma 0.46, rho -0.95, v0 0.0127, SPX 0.52 vp, VIX 0.53).
   Do NOT commit the DB or any new/changed pickle - artifact regeneration happens at integration.`,
    },
  ],
  data: [
    {
      label: 'data-layer',
      prompt: `YOUR CHAIN: data ("data layer"). Worktree: ${WT}/data (branch c22/data).
FILES YOU OWN: joint_vol_calibration/data/database.py, joint_vol_calibration/data/yf_downloader.py,
joint_vol_calibration/data/cboe_downloader.py, joint_vol_calibration/data/pipeline.py, joint_vol_calibration/config.py,
scripts/backfill_tbill_history.py, scripts/backfill_calibration_provenance.py, joint_vol_calibration/tests/test_rates.py,
new joint_vol_calibration/tests/test_data_layer.py, new joint_vol_calibration/tests/test_cboe_downloader.py, new fixture files
under joint_vol_calibration/tests/fixtures/, and data_store/vol_system.db (YOU ARE THE ONLY CHAIN ALLOWED TO COMMIT DB
CHANGES - make them deliberate, scripted and documented in the commit message; leave the DB in journal_mode=DELETE, VACUUMed).
Note: another chain (calib) is concurrently creating joint_vol_calibration/data/option_chain.py - do not create a file
with that name.
FINDINGS: every finding in ${FD}/find_data.json (for "data-vix-ts-index-fit-as-futures" fix the data side only: tenor_days
written by the downloader/DB must come from config.VIX_INDEX_TENOR_DAYS, comments/DDL must stop calling the indices futures;
the calibrator side is another chain's). Also ${FD}/find_ops.json "ops-yahoo-history-claim-false" (comments in your files)
and the data-layer side of "ops-db-not-persisted".
DECISIONS:
 - Snapshot dates are the NY market session of the quotes, never wall-clock date. Re-date the committed weekend snapshots in
   the DB to their true session (report the old->new mapping in notes_for_integration; calibration pickles that reference
   them are regenerated at integration).
 - ^IRX is a 13-week discount yield in percent: convert to a continuously compounded decimal rate correctly (bond-equivalent /
   money-market convention documented) and keep negative/zero yields (do not drop them).
 - Rewrite the CBOE downloader against the REAL endpoints (verify live with curl: delayed_quotes/options/_SPX.json and _VIX.json;
   VX per-contract CSVs and the daily settlement CSV) with offline fixtures for tests; ingest true VX futures settlements keyed
   (date, expiry_date); fix the Yahoo fallback reachability.
 - Persistence support for the daily bot (integration will wire it): add database helpers that export/import yearly parquet
   partitions under data_store/market/ (spx_ohlcv, vix_daily, vix_term_structure, tbill_rates) and per-session option
   chains under data_store/options/, with upserts that never delete a row the remote source later dropped. Document the API.
 - Close SQLite connections properly (contextlib.closing), idempotent options snapshot writes (natural key / delete-then-insert
   in one transaction), calendar/cross-source validation hooks (missing NYSE sessions, bad prints) that log warnings.
 - test_rates.py must never write to the tracked production DB (use a temp copy / monkeypatched path).
 - config.py: remove dead/misleading constants or make them true; no import-time side effects beyond what is unavoidable.
Keep every existing public function signature working (grep all callers).`,
    },
  ],
  pdv: [
    {
      label: 'pdv',
      prompt: `YOUR CHAIN: pdv ("path-dependent volatility"). Worktree: ${WT}/pdv (branch c22/pdv).
FILES YOU OWN: joint_vol_calibration/models/pdv.py, joint_vol_calibration/signals/regime_pdv.py,
joint_vol_calibration/tests/test_pdv.py, joint_vol_calibration/tests/test_regime_pdv.py, data_store/pdv_model.pkl (you may
regenerate it; keep it loadable by its current consumers - grep dashboard/app.py and backtest code - or report the change).
FINDINGS: every finding in ${FD}/find_pdv.json. Dashboard-side parts (the NaN percentile in dashboard/app.py, how the
COVID stress card is computed/displayed if it lives in app.py) go to cross_chain_requests with the exact fix.
DECISIONS:
 - The forecast compared with VIX (implied - realised spread, pdv_iv_spread feature, S1/S1C) must be variance-consistent and
   horizon-matched: y21(t) = sqrt(252/21 * sum_{i=1..21} r_{t+i}^2), purged walk-forward with an embargo >= horizon. Fix the
   build_xy horizon>1 window bug. Provide a clean public helper for consumers, e.g.
   walk_forward_vol_forecast(returns, horizon=21, ...) -> pd.Series aligned at t using only information <= t, plus a way to
   get the latest live forecast with its feature date. Keep the 1-day |r| model only where an R^2-on-|r| table is wanted,
   labelled honestly as an E|r| forecast.
 - Recompute and report honest walk-forward R^2 (1-day and 21-day targets; with and without 2020) and what the README claim
   "Guyon-Lekeufack (2023) ... walk-forward R^2 = 0.31" should say. Be precise about how this model differs from G-L.
 - The "BN-S jump filter" on daily data is not a BNS test: implement a statistically valid daily jump flag (e.g. a robust
   local-vol threshold test with a stated significance level) and name it honestly, or implement BNS correctly if intraday
   data exists (it does not here).
 - Fix RegimePDV look-ahead (contemporaneous sigma, full-sample coefficients), Merton MLE (bounds, fallback), GARCH timing,
   kernel extrapolation, EWMA burn-in, and the COVID stress-test alignment (prediction for day t must use info <= t-1 and be
   compared with day t's move, out-of-sample).`,
    },
  ],
  quintic: [
    {
      label: 'quintic',
      prompt: `YOUR CHAIN: quintic ("Quintic OU model"). Worktree: ${WT}/quintic (branch c22/quintic).
FILES YOU OWN: joint_vol_calibration/models/quintic_ou.py, joint_vol_calibration/calibration/quintic_calibrator.py,
joint_vol_calibration/tests/test_quintic_ou.py, new joint_vol_calibration/tests/test_quintic_calibrator.py,
data_store/calibrations/quintic_cal_2026-03-24.pkl (you may regenerate it with full provenance).
FINDINGS: every finding in ${FD}/find_quintic.json (dashboard-panel text -> cross_chain_requests).
DECISIONS:
 - Rewrite the SPX Monte Carlo as an adapted (left-point Ito) scheme with exact joint Gaussian increments for the OU
   integrals and dW; >= 252 steps/yr (min 100); proper antithetics; martingale, step-convergence and no-arbitrage tests.
 - VIX option targets: Black-76 on the VIX forward. There are no VX futures in the DB; derive the per-expiry VIX forward from
   VIX option put-call parity on the same snapshot. Discounting consistent between model and market.
 - Remove the false "VIX futures by construction" claim (or make it true); align parameterisation with Abi Jaber, Illand & Li
   (verify the exact paper title/authors/year by web search and fix both citations); document honestly where the
   implementation departs from the paper (factors, floors, free alpha5).
 - Add the C20-style data guards (same-date VIX tenors, surface freshness, sufficiency) to QuinticCalibrator.
 - Recalibrate on the 2026-03-24 snapshot with the corrected pricer (bounded compute: <= ~45 min CPU; fix the VIX-leg
   performance finding first), save the pickle with provenance (params, data counts, seeds, steps, convergence status), and
   report the new SPX and VIX-options RMSEs measured honestly. A like-for-like Heston comparison on the same date/dataset is
   done later at integration - just report what is needed for it.`,
    },
  ],
  nngreeks: [
    {
      label: 'nngreeks',
      prompt: `YOUR CHAIN: nngreeks ("NN surrogate + Greeks monitor"). Worktree: ${WT}/nngreeks (branch c22/nngreeks).
FILES YOU OWN: joint_vol_calibration/models/nn_pricer.py, joint_vol_calibration/greeks/risk_monitor.py, new
joint_vol_calibration/models/implied_vol.py (torch-free Black-Scholes/Black-76 prices, greeks with dividend yield, and a
robust IV solver; nn_pricer keeps re-exporting the old private helper names for back-compat),
joint_vol_calibration/tests/test_nn_pricer.py, joint_vol_calibration/tests/test_risk_monitor.py, new
joint_vol_calibration/tests/test_implied_vol.py.
FINDINGS: every finding in ${FD}/find_nngreeks.json EXCEPT: heatmap binning/captions rendered by dashboard/app.py or
templates (-> cross_chain_requests with exact fix) and "nngreeks-delta-hedger-vomma-flag-dead" (backtest chain owns
delta_hedger.py).
DECISIONS:
 - IV solver: Manaster-Koehler inflection-point start + bracketed safeguarded Newton on the OTM price (or Jaeckel), NaN for
   non-converged/ill-posed inputs - never return a non-converged iterate. Exact round-trip tests over a dense grid.
 - Greeks surface: no silent node dropping, provenance recorded (calibration file, session, S, r/q, params), units stated;
   vomma and volga are the same Greek - one name, consistently; QV convexity from the exact CIR closed form; an "unstable
   node" metric that is meaningful (not a pooled cross-sectional z-score that always flags ~3 wing nodes) - document it.
   Provide a function/CLI to regenerate data_store/greeks/greeks_surface.parquet from a given calibration pickle; the
   actual regeneration from the corrected headline fit happens at integration (do not commit the parquet).
 - NN: fix train_network on the installed torch, seed everything, sample the training domain to cover the calibration
   bounds actually used (or add a domain guard that refuses out-of-domain inputs), make the benchmark fair (same inputs,
   batch vs batch, full domain including the OTM-call wing, honest MAE). If a reasonably accurate model can be trained in
   <= ~40 min of CPU and the weights are <= ~3 MB, train it, save the weights to the path the tests expect, and report the
   path so integration can decide whether to un-ignore and commit it (you may NOT edit .gitignore). Report honest numbers.`,
    },
  ],
  signals: [
    {
      label: 'signals',
      prompt: `YOUR CHAIN: signals ("signal engine + regime classifier"). Worktree: ${WT}/signals (branch c22/signals).
FILES YOU OWN: joint_vol_calibration/signals/signal_engine.py, joint_vol_calibration/signals/regime_classifier.py,
joint_vol_calibration/tests/test_signal_engine.py, joint_vol_calibration/tests/test_regime_classifier.py,
joint_vol_calibration/tests/test_lookahead.py, data_store/signals/regime_classifier.pkl (may re-save in a version-robust way),
data_store/signals/signals_2015_2025.parquet and data_store/ivr_spread_panel.parquet (delete ONLY if nothing reads them -
grep the whole repo including dashboard/; otherwise regenerate or report).
FINDINGS: every finding in ${FD}/find_signals.json EXCEPT the P&L-side ones owned by the backtest chain
("signals-s3-mechanical-zscore-pnl", "signals-kelly-corr-netting-noop") and dashboard-side ones
("signals-live-pdv-percentile-nan", "signals-app-refresh-writes-classifier-labels", "signals-variants-table-fabricated") -
for those, put any signal-side support you add into api_changes. Also the code side of ${FD}/find_docs.json
"docs-regime-gate-backtest-vs-site" and "docs-classifier-claims".
DECISIONS:
 - ONE R2 gate everywhere: the ADAPTIVE gate (rolling 252-trading-day 80th percentile of VVIX using data <= t) - this is the
   gate the published labels, the dashboard and the documented "improvement step 5" describe. Put it in a single module
   constant/func in regime_classifier.py used by build_regime_labels (default), and remove SignalEngine's hard-coded
   VVIX>100 override; precomputed regimes passed in must never be modified (only lagged). Persistence baseline and all
   regime statistics must be computed on the labels actually used.
 - S2 stop-loss: fix the inverted direction (express it in terms of the straddle actually traded); fix the test to encode
   the correct direction and add the mirror case.
 - HMM: causal forward filtering only (no smoothing/Viterbi on the full sample) and correct state ordering.
 - Isotonic/probability calibration fit out-of-sample (or removed); classifier accuracy claims recomputed walk-forward without
   look-ahead features - report the correct numbers (in-sample, OOS, persistence baseline on the same dates).
 - Make the look-ahead tests catch real look-ahead (mutation-test them: removing the 1-day shift, a full-sample z-score, a
   forward-looking R2 override, a t+1 return must each fail a test).
 - The pdv_iv_spread feature must later switch to a horizon-21 variance-consistent PDV forecast that another chain is building;
   do not implement that model, but make build_features accept an injected forecast series cleanly and report the hook.
 - "IVR" is a misnomer (no implied-vol rank exists): rename in code/docstrings where you own them, report other locations.`,
    },
  ],
  backtest: [
    {
      label: 'backtest',
      prompt: `YOUR CHAIN: backtest ("backtest engine + delta hedger"). Worktree: ${WT}/backtest (branch c22/backtest).
FILES YOU OWN: joint_vol_calibration/backtest/backtest_engine.py, joint_vol_calibration/backtest/delta_hedger.py,
joint_vol_calibration/backtest/report_generator.py, joint_vol_calibration/backtest/run_backtest.py,
joint_vol_calibration/tests/test_backtest_engine.py, joint_vol_calibration/tests/test_delta_hedger.py, new
joint_vol_calibration/tests/test_backtest_accounting.py, new joint_vol_calibration/backtest/c22_bridge.py.
Do NOT commit anything under data_store/backtest/ (the canonical re-run happens at integration after the signals and PDV
chains land).
FINDINGS: every finding in ${FD}/find_backtest.json; plus ${FD}/find_signals.json "signals-s3-mechanical-zscore-pnl" and
"signals-kelly-corr-netting-noop"; ${FD}/find_nngreeks.json "nngreeks-delta-hedger-vomma-flag-dead"; ${FD}/find_docs.json
"docs-sharpe-cash-double-count" and "docs-s3-proxy-and-sensitivity" (engine side).
DECISIONS:
 - ATM vol is not VIX. Price and mark straddles with sigma_ATM = beta * VIX-term-structure interpolation, beta an explicit,
   documented engine parameter (ATM_TO_VIX_RATIO). Estimate beta from ALL real SPX chains you can get: the snapshots in the
   DB plus a fresh LIVE Yahoo ^SPX chain and same-time ^VIX (30d ATM-forward IV interpolated in total variance vs VIX).
   Default = the estimate (2 dp). Provide a sensitivity runner for beta in {1.00, 0.90, estimate}. Report the estimates.
 - Sharpe convention: the strategy P&L on unfunded $1M collateral is already an excess return (collateral assumed to earn
   ^IRX). Sharpe/Sortino use rf=0 on that P&L; apply consistently in compute_metrics, crisis stats, walk-forward stats and
   the report generator; S3's Sharpe is reported, not hidden.
 - S3: its current P&L (notional x delta-z x 0.02) is positive by construction. Re-price S3 as the index leg of the dispersion
   trade with the SAME delta-hedged straddle engine as the other signals (long dispersion => short SPX straddle), documented
   as an index-leg proxy (no single-stock option data). Add a null test for the S3 timing rule (seeded random-walk /
   permutation) and report its p-value; fix the exit-day P&L/cost bug.
 - Delta hedging: include opening and closing hedge costs, drop the phantom exit-day rebalance, and include financing of the
   option premium and the hedge (cash accrues r, stock leg pays q). Correct theta day count and add vanna/volga to the C7
   attribution so the residual is small in a BS world; fix the C7 Run B look-ahead and claim; make the vomma flag functional.
 - Positions expire/roll properly (no T_rem floor that holds a straddle for months); rate fallback = previous available rate
   (keep negative yields); Kelly/correlation netting either correct or removed (no no-op advertised as a feature); sizing on
   current sleeve NAV; trade log includes the open trade at the end; walk-forward either genuinely OOS or renamed.
 - Validate by running the full backtest in your worktree and REPORT the new numbers (portfolio NAV, cumulative P&L, Sharpe,
   max DD; per signal P&L/Sharpe/win%/trades) for the default beta and the sensitivity grid. Build c22_bridge.py: a script
   that re-runs the engine applying your corrections one at a time in a fixed, documented order from the published headline
   (NAV $738,794 / P&L -$261,206 / Sharpe -1.80 / MaxDD -31.5%) and prints a waterfall table; integration will re-run it
   after the signals (adaptive R2 gate, S2 stop fix) and PDV (horizon-21 spread) chains merge, so make those toggles
   pluggable (flags that no-op when the upstream API is absent).`,
    },
  ],
  dashboard: [
    {
      label: 'dashboard-backend',
      prompt: `YOUR CHAIN: dashboard, step 1 of 2 ("dashboard backend correctness"). Worktree: ${WT}/dashboard (branch c22/dashboard).
FILES YOU OWN: dashboard/app.py, dashboard/freeze_site.py, dashboard/templates/*.html (only as needed for backend fixes in
this step), dashboard/make_og_image.py, dashboard/og-image.png, new dashboard/static/*, new
joint_vol_calibration/tests/test_dashboard.py, and ONLY the dashboard-related test functions in
joint_vol_calibration/tests/test_c21_audit_fixes.py.
FINDINGS: every finding in ${FD}/find_dashapp.json, plus these duplicates/related ones from other auditors (fix the app.py
side): find_pdv "pdv-dashboard-spread-percentile-nan", "pdv-stale-model-no-refresh" (display side), "pdv-covid-stress-leak"
(display side); find_docs "docs-pdv-spread-percentile-nan", "docs-local-run-overwrites-labels", "docs-covid-stress-offbyone"
(display side), "docs-backtest-kpi-trade-count", "docs-walkforward-chart-mislabeled", "docs-greeks-smile-from-rejected-fit";
find_signals "signals-live-pdv-percentile-nan", "signals-app-refresh-writes-classifier-labels",
"signals-variants-table-fabricated"; find_ops "ops-app-refresh-overwrites-labels", "ops-canonical-all-root",
"ops-procfile-dead" (app side); find_nngreeks "nngreeks-heatmap-hides-nodes", "nngreeks-greeks-page-false-captions";
find_dashux "dashux-pdv-percentile-nan", "dashux-ntrades-double-count", "dashux-walkforward-flat-rf",
"dashux-heatmap-hidden-nodes", "dashux-smile-chart-wrong-surface", "dashux-silent-zero-render", "dashux-clf-pdv-ordering",
"dashux-calib-r-q-mismatch", "dashux-app-refresh-writes-classifier-labels".
DECISIONS:
 - Rendering must be side-effect free: remove the network/file-writing refresh threads from app.py (data refresh belongs to
   scripts/cloud_refresh.py, which another chain is changing - do not edit it). python dashboard/app.py must never modify
   tracked files.
 - Headline calibration = the LATEST fit that passes is_acceptable_calibration (choosing the min-RMSE fit across history is
   a selection bias). Show its age in NYSE sessions with an escalating staleness banner (warn >= 2, error >= 5). Use
   joint_calibrator.calibration_is_legacy(cal) when it exists (another chain is adding it; guard with getattr/try so your
   branch works without it) to label pre-C22 fits. Display the r/q the fit actually used (from the result; "n/a" if absent).
 - Any loader failure renders a visible error state (never zeros with confident prose); freeze_site.py exits non-zero if
   any page reports a loader error (fail closed so a broken page is never deployed).
 - Metrics (Sharpe, walk-forward, trade counts, win rates) come from one source: the backtest engine's metric functions /
   stored results, under its convention (Sharpe on excess P&L, rf=0, collateral assumed to earn ^IRX); no re-implementations
   with a flat 5% hurdle. Prose numbers are computed from the artifacts, not hard-coded, wherever feasible.
 - Correct the PDV percentile (exclude NaN), render the stale-forecast flag prominently, fix the classifier-feature ordering,
   the heatmap (no silent drops, correct axis definition), per-page canonical/og:url, sitemap lastmod from data dates.`,
    },
    {
      label: 'dashboard-ux',
      prompt: `YOUR CHAIN: dashboard, step 2 of 2 ("dashboard UX, accessibility, web quality"). Worktree: ${WT}/dashboard
(branch c22/dashboard). Step 1 just committed backend fixes on this branch - read its commits first.
FILES YOU OWN: dashboard/templates/*.html, dashboard/app.py (presentation-related changes only), dashboard/freeze_site.py,
dashboard/make_og_image.py, dashboard/og-image.png, new dashboard/static/*, joint_vol_calibration/tests/test_dashboard.py.
FINDINGS: every finding in ${FD}/find_dashux.json not already handled in step 1; ${FD}/find_docs.json dashboard-copy items
("docs-regime-legend-wrong", "docs-calibration-banner-claims", "docs-rmse-chart-and-formatting", "docs-corrections-log-
incomplete" (site part), "docs-test-count-claims" (site/og part)); ${FD}/find_ops.json "ops-cdn-no-sri", "ops-og-image-stale".
GOALS: WCAG 2.2 AA contrast everywhere (keep the dark/neon identity, adjust tokens), a real heading hierarchy, landmarks,
text alternatives for every chart (aria-label + a visually-hidden data summary or table), keyboard-focusable scroll regions,
visible focus styles, aria-current on nav, prefers-reduced-motion, no mobile clipping/overflow at 390px, sticky nav that
works, print CSS, font fallbacks. Vendor Chart.js into dashboard/static (pinned version, with a fallback) or load it with SRI;
no render-blocking failure mode that kills every chart. Regime descriptions must state the actual rule (rv20 vs VIX; adaptive
VVIX 80th-percentile gate). Social card must not hard-code a test count that goes stale (render it from data or drop it).
VERIFY with Playwright/Chromium (PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers): freeze the site in your worktree, screenshot
every page at 1440px and 390px (save under ${SP}/work-dashboard/shots and Read them to inspect), check for horizontal
overflow, console errors and broken links programmatically, and compute contrast ratios of the final tokens. Add tests in
test_dashboard.py for the structural guarantees (e.g. every page has one h1, every canvas has an accessible name, no
canonical pointing to another page, freeze fails on loader error).`,
    },
  ],
}

const SELECTED = Array.isArray(args) && args.length ? args : Object.keys(CHAINS)
log(`chains: ${SELECTED.join(', ')}`)

const results = await parallel(SELECTED.map(key => async () => {
  const steps = CHAINS[key]
  const reports = []
  for (const s of steps) {
    const r = await agent(`${COMMON}\n${s.prompt}`, { label: `${key}:${s.label}`, phase: 'Fix', schema: REPORT_SCHEMA })
    reports.push({ step: s.label, report: r })
    if (!r) { log(`${key}:${s.label} returned nothing - stopping chain`); break }
    const fixed = r.verdicts.filter(v => v.status === 'fixed' || v.status === 'partially-fixed').length
    const refuted = r.verdicts.filter(v => v.status === 'refuted').length
    log(`${key}:${s.label} done - ${fixed} fixed, ${refuted} refuted, ${r.commits.length} commits, tests: ${r.test_result}`)
  }
  return { chain: key, reports }
}))
return results
