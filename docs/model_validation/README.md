# Model validation pack

**Status:** the evidence available at C23 (branch `c23/integration`, 2026-10-21): the C22 record plus the C23 model,
data, statistics and engineering work. Every number here is quoted from a committed artifact (named beside it), a test
that pins it, or a run recorded in `docs/`. The numbers the README also quotes are checked against their artifacts by
`tests/test_readme.py`.

## What this document is, and is not

It organises the project's evidence the way bank model-risk guidance does, so a reader who knows that discipline can
find what they are looking for. The structure and vocabulary (model inventory, materiality, conceptual soundness,
outcomes analysis, ongoing monitoring, effective challenge, vendor and third-party products) follow the Federal
Reserve / OCC / FDIC **Revised Guidance on Model Risk Management**, SR letter 26-2 (17 April 2026,
<https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm>, fetched 2026-10-21), which "supersedes and
replaces SR letter 11-7".

SR 26-2 is supervisory guidance for banking organisations, and this project is not a bank. **Nothing here claims
compliance with it.** The most important departure from the guidance is structural: **the developer and the validator
are the same person**, so there is no independent validation function. Section 5 lists what partly substitutes for it
and what does not.

## 1. Model inventory

"Use" is how the output reaches anything published: **production** (computed by the daily job or shown on the
dashboard), **research** (reported, on no production path), **demoted** (shown only as a negative result).
"Materiality" is how much a published number depends on the model (high: a headline number or its sign; medium: a
secondary published number; low: diagnostics only). The dashboard's Model Risk page carries the same inventory in
short form (`dashboard/model_inventory.json`).

| ID | Model | Code | Use | Materiality | Key limitations | Validation evidence |
|---|---|---|---|---|---|---|
| M1 | Heston / Bates pricing engine (Lewis 2001 contour, BS control variate, adaptive trapezoid) | `models/heston.py` | production: calibration, Greeks, surface scorecard | high | 1e-10 DF·F quadrature target; 5e-9 F at the slowest-decaying corners of the box | `test_heston_reference.py` (independent quadratures), `test_external_oracles.py` (QuantLib), `test_properties_pricing.py`, mutation check |
| M2 | Implied-vol solvers: safeguarded Newton (`implied_vol_black`), bisection (`black76_implied_vol`) | `models/implied_vol.py`, `data/option_chain.py` | production: chain cleaning, calibration targets, Greeks | high | an ITM price whose time value is lost to rounding returns NaN by design | `test_implied_vol.py`, `test_properties_iv.py`, `test_external_oracles.py` (Let's Be Rational) |
| M3 | Option-chain hygiene: two-sided quotes, parity forward and discount per expiry, quote-state gate, session dating | `data/option_chain.py` | production: every fit | high | the calibrator does not drop quotes in static arbitrage (M14 screens and reports them); Yahoo closing books carry no quote time | `test_option_chain.py`, `test_lookahead.py`, `test_rate_reference.py` |
| M4 | Joint Heston calibrator (SPX z-grid anchors + VIX index leg; multi-start TRF with an analytic Jacobian, warm-started; DE as an optional global check) and its quality gate | `calibration/joint_calibrator.py` | production: daily fit, headline | high | σ on its 2.0 box ceiling in 2 of 5 v2 fits; Feller violated in all 5; VIX-options leg off (w3 = 0) | `test_joint_calibrator.py`, `test_identifiability.py`, `make reproduce` |
| M5 | VIX index and VX futures under Heston/Bates (exact CIR Laplace transform) | `models/heston.py` | production: VIX leg; VX futures out of sample | medium | Heston misses real VX futures by 1.85–3.18 pts out of sample | `test_heston.py`, `test_heston_vvix.py`, `test_properties_pricing.py` |
| M6 | Greeks surface (BS Greeks at the model IV, QV convexity, vomma flag) | `greeks/risk_monitor.py` | production: Risk page | medium | the vomma shock scales with the fitted σ, which is on its 2.0 ceiling in the headline fit | `test_risk_monitor.py`, `test_rate_reference.py`, `make reproduce` |
| M7 | β = ATM vol / VIX proxy (0.82, from four 2026 chains plus one live capture) | `backtest/backtest_engine.py` | production: every backtest P&L | high (it decides the sign of S1) | held constant over 2018–2025; the first later chain (2026-10-16) gives 0.70 | `run_backtest --estimate-beta`, β grid, Cboe PUT/WPUT replication (§3) |
| M8 | Backtest engine and signals (S1, S1C, S2, S3, S4; delta-hedged 30-day straddles, mark-to-model) | `backtest/`, `signals/signal_engine.py` | production: Research page (a negative result) | high | all P&L mark-to-model at β·VIX; no option prices inside the window | `test_backtest_accounting.py`, `test_backtest_reproducibility.py` |
| M9 | PDV volatility forecaster (EWMA approximation of Guyon & Lekeufack) | `models/pdv.py` | production: S1/S1C inputs | medium | ranks 6th of 7 on QLIKE among the forecasts compared (§3) | `test_pdv.py`, `test_stats_forecast.py` |
| M10 | Rule regime labels (R0/R1/R2; VVIX vs its rolling 252-day 80th percentile) | `signals/regime_classifier.py` | production: labels, backtest gate | medium | 5.9% of labels flip within measurement noise (C22) | `test_regime_classifier.py`, `test_lookahead.py` |
| M11 | XGBoost regime classifier | `signals/regime_classifier.py` | demoted (Research page only) | low | loses to "predict yesterday" (74.9% against 87.0% on its own evaluation) | `test_regime_classifier.py` |
| M12 | NN surrogate pricer | `models/nn_pricer.py` | research | low | trained domain narrower than the calibrator's box (κ ≤ 15 vs 20, θ ≤ 0.2 vs 1, v₀ ≤ 0.25 vs 1); the speed-up once quoted (14–23×) is not backed by a committed benchmark and is withdrawn | `test_nn_pricer.py` |
| M13 | Two-factor Quintic OU (SPX MC, polynomial VIX²) | `models/quintic_ou.py`, `calibration/quintic_calibrator.py` | research | low | the stored fit did not converge; ε on its −0.99 bound | `test_quintic_ou.py`, `test_quintic_calibrator.py` |
| M14 | Quote-grade surface: static-arbitrage screen and l1 repair (Cohen–Reisinger–Wang on HiGHS), executable-arbitrage certificate, eSSVI, calendar-ordered SVI, one scorecard for every model, VIX replication from the chain | `surface/` | production: written per accepted fit; Surface page; Overview result 1 | medium | held-out strikes lie between fitted ones (an interpolation test); certificates on dense grids, not closed form | `test_surface_*.py` (arbitrage, eSSVI, SVI, metrics, scorecard, VIX, report) |
| M15 | Rough Heston (lifted, 20 factors) | `models/rough_heston.py` | research: per-session comparison | low | H weakly identified (on its 0.02 floor on 2026-10-16; ½ on 2 of 5 sessions) | `test_rough_heston.py` (QuantLib Adams reference) |
| M16 | Spot × vol scenario grid of a reference straddle (sticky-strike and sticky-moneyness) | `greeks/scenario_grid.py` | production: Risk page | low | instantaneous parallel shocks; one reference book | `test_scenario_grid.py` (Heston-revaluation oracle, second-order Taylor check) |
| M17 | Statistical inference of the backtest (stationary bootstrap, PSR/DSR, SPA/StepM, CSCV PBO, exposure-matched nulls, forecast comparison) | `stats/`, `scripts/inference_report.py` | production: Research page; Overview result 3 | medium | inference about the β·VIX proxy, not executable P&L; one window the rules were revised on | `test_stats_*.py`, `test_stats_inference_report.py` |
| M18 | Forward ledger (signals at the captured bid/ask, hash-chained per session) | `backtest/forward_ledger.py` | production: written per accepted fit from C23 | medium (the only executable-price evidence) | no entry committed yet; marks are the captured Yahoo book | `test_forward_ledger.py`, `test_cloud_refresh.py` |
| M19 | Treasury zero curve (bills and par yields, look-ahead gated, refused when older than 7 days) | `data/rates.py` | production: calibrator reference, chain cleaner, C7, Greeks carry | medium | daily; falls back to ^IRX and records it when stale | `test_rates.py`, `test_rate_curve.py`, `test_rate_reference.py` |

**Vendor and third-party inputs (SR 26-2's treatment of vendor products).** Terms and what is committed:
[DATA_SOURCES.md](../../DATA_SOURCES.md).

| ID | Source | What it supplies | Checks in place | Known issues |
|---|---|---|---|---|
| V1 | Yahoo Finance (`yfinance`) | ^GSPC, ^IRX, the SPX chain the bot fits, provisional VIX-family rows | two-sided quotes only, own IVs on parity forwards, quote-state gate, session dating; a Yahoo VIX-family row never replaces a Cboe one and every overlapping session is compared (> 0.25 pts is a warning) | no quote time on closing books; personal-use terms |
| V2 | Cboe (index files, VX futures, delayed option books, strategy indices) | the VIX index family (primary from C23), VX settlements, the daily SPX/VIX book capture, PUT/WPUT/BXM/BFLY/CNDR/VXTH/SET | NYSE-session filter; the capture is refused when the live book is not the session's; each capture records its fetch time, quote time and the SHA-256 of the raw response | 15-minute delay; personal, non-commercial terms |
| V3 | U.S. Treasury, NY Fed | bill rates, par curve, SOFR | look-ahead gate (a rate is used only after it is published); staleness refusal | daily only |
| V4 | Library numerics (NumPy, SciPy, pandas) | every computation | exact lock (`requirements-lock.txt`), CPU-tolerant artifact comparison (`tests/artifact_tolerance.py`) | pip-audit findings fixed by floors (C23-E2) |

## 2. Conceptual soundness

### M1 Pricing engine

- **Design.** Lewis (2001) contour integral on Im u = −1/2 with a Black-Scholes control variate, on the "little trap"
  characteristic function rewritten so σ never divides outside a log1p. Adaptive trapezoid: truncation from |φ| on a
  geometric scan, the step halved until two estimates agree to 1e-10 DF·F. No FFT.
- **Why not Carr-Madan or COS.** The Lewis contour needs only E[(S_T/F)^{1/2}], finite for every parameter and
  maturity; Carr-Madan with α = 1.5 needs E[S_T^{2.5}] and breaks beyond the moment-explosion time. QuantLib's
  `COSHestonEngine` at its defaults misses the adaptive reference by 1.7e-6·S at the σ = 2 headline fit; this engine is
  within 1e-3 of that error (`test_external_oracles.py`; Le Floc'h, arXiv:2005.13248).
- **Evidence.** Independent references written from scratch (Riccati ODE, Gil-Pelaez, Gauss-Legendre Lewis, QAWF,
  ncx2) over 47 parameter sets, 7 maturities, 13 strikes (`test_heston_reference.py`). **External oracle:** QuantLib
  1.44 `AnalyticHestonEngine` at 1e-12 on every committed fit and the stress corners, 1 day to 2 years: worst
  5.6e-12·S, at most 3e-13·S on the fits, asserted below 1e-10·S; Bates against `BatesEngine` 1.6e-12·S. Properties over
  the whole parameter box (no static or calendar arbitrage, parity across engines, the martingale condition, the σ → 0
  limit, Bates ⊃ Heston, time-change and homogeneity invariants); 11 of 12 curated mutants killed.

### M2 Implied-vol solvers

- **Design.** Inversion of the out-of-the-money price only, Newton on transformed objectives with a bisection bracket;
  ill-posed inputs return NaN with a status code, never a boundary value.
- **Evidence.** Against Jäckel's Let's Be Rational (py_lets_be_rational 1.1.2, dev-only) on 20,000 seeded quotes
  (ln K/F in [−1.5, 0.7], 1 day to 3 years): solves every quote LBR solves, max |Δσ|/σ 1.3e-12. Property tests: OTM round
  trip within 1e-9 relative, monotone in price, call/put symmetric.

### M4 Joint calibrator

- **Design.** Vega-normalised SPX price errors on z-grid anchors (OTM against the parity forward, ≥ 3 wing puts per
  expiry) plus the four Cboe spot-starting variance-swap indices; the residual vector is minimised by multi-start
  trust-region least squares (Sobol starts plus the previous accepted fit) with an analytic Jacobian of the Lewis engine.
  Quality gate: ≥ 4 SPX expiries, ≥ 30 options, a put wing in ≥ 3 expiries, ≥ 3 VIX tenors, a snapshot dated to the spot
  session, model_version 2.
- **Optimiser evidence** (`data_store/calibrations/diagnostics/summary.json`, five sessions). TRF reaches the stored
  differential-evolution optimum on every session (relative loss difference −3.9e-8 to −3.6e-7) with one basin over
  8–9 starts, in 1.7–2.1 s and 226–337 evaluations against 19.8–28.3 s and 3,912–5,262 for serial DE (quiet 4 vCPU).
  The analytic Jacobian is 3.4–4.0× faster per call than two-point differences and agrees with central differences to
  2.4e-8. DE in four processes agrees with TRF to 2.1e-13.
- **Identifiability.** cond(JᵀJ) of the relative-sensitivity Jacobian is 150–293. Gauss–Newton relative standard
  errors: κ 12–19%, θ 3–7%, ρ 2–4%, v₀ 2–6%. They are a curvature scale (the residuals are model misfit, not noise); the
  profile check shows up to about 26% optimism.
- **The σ box.** The production box caps σ at 2. On 2026-10-16 the σ profile (caps 1 to 10, κ ≤ 60) shows the cap costs
  27.7% of the loss: freed, σ 6.90 and κ 27.4 give an out-of-sample RMSE of 1.19 instead of 2.96 vol pts and a model
  VVIX of 113 against the market's 110 (205 at the cap). On 2026-03-24 the cap costs 2.1%. The box is kept because the
  NN domain, the vomma shock sizing and the gate constants depend on it; its cost is published instead.
- **Known weaknesses.** Feller is violated in all five fits; at the boxed optima the model's VVIX is 160–211 against a
  market ^VVIX of 86–133. The VIX-options leg is off (w3 = 0) because only four sessions of VIX option quotes are stored.

### M7 β proxy

- **Design.** ATM 30-day implied vol = β × VIX, β = 0.82 from real 2026 chains, constant over the backtest.
- **Why it matters.** S1 moves from +$2,509,627 at β = 1.00 to −$554,935 at β = 0.78; the portfolio is negative at every
  β tested (`data_store/backtest/beta_sensitivity.csv`). The first chain after the estimation set gives 0.70.
- **External check (C23).** Replicating Cboe's PUT index (monthly ATM put writes, real settlements) implies β = 0.80
  over the window, 0.63–0.90 by full year (2018–2024, within-year standard error about 0.05); WPUT 0.80, 0.75–0.89
  (`data_store/backtest/inference/benchmarks.json`). The constant 0.82 sits inside both ranges; the year-to-year spread
  is the model risk the β grid prices.

### M14 Quote-grade surface

- **Design.** Every stored quote is screened for static arbitrage at the bid/ask (Cohen, Reisinger & Wang 2020 rows:
  outright, vertical, butterfly, calendar), repaired with the l1-minimal move inside a linear programme on HiGHS, and an
  executable-arbitrage certificate decides whether any arbitrage-free price vector fits inside the quoted spreads. eSSVI
  (Gatheral & Jacquier 2014) and a calendar-ordered SVI are fitted to alternate strikes; every model, Heston included,
  is scored on the same calibrator out-of-sample universe in units of the half-spread.
- **Evidence.** 0 butterfly and 0 calendar violations on the dense grids of the arbitrage-free surface in all five
  sessions; the VIX replicated from each chain with Cboe's rules lands within 0.09–0.78 pts of the official close, the
  gap attributed to truncation, the far wing, the fit and discretisation (`vix_replication.json`).

### M15 Rough Heston

- **Design.** Lifted multi-factor approximation (Abi Jaber & El Euch 2019) with 20 factors, priced through the same
  Lewis engine, fitted with the same anchors, VIX leg and box plus H ∈ [0.02, ½].
- **Evidence.** Lifting error at H = 0.02 against QuantLib's Adams scheme: at most 7.7e-5·F and 0.085 vol pts (14 days to
  1 year); the numerics check at each fitted point is at most 0.057 vol pts.

### M17 Inference

- **Design.** Stationary-bootstrap intervals (Politis & Romano 1994; 10,000 replicates) as the primary interval;
  PSR/DSR (Bailey & López de Prado) with the trial count from a committed ledger (`docs/trials_ledger.yaml`: 162 raw
  configurations over C16–C22, 4.6 by eigenvalue participation), reported as a range over N and trial-variance choices;
  Hansen SPA and Romano–Wolf StepM; CSCV PBO; timing nulls that keep each sleeve's own exposure (3,550 circular shifts,
  2,000 trade shuffles), Holm-adjusted over six series. Three runs gave byte-identical artifacts.

## 3. Outcomes analysis

| Model | Outcome measured | Result (source) |
|---|---|---|
| M4 | Out-of-sample SPX IV RMSE on the quotes not used as anchors, five v2 sessions | 1.98 / 1.54 / 1.28 / 1.83 / 2.96 vp for 2026-03-24 / 03-27 / 05-29 / 06-18 / 10-16 (`leg_losses.spx_oos_iv_rmse` in the pickles) |
| M4 | Error in units of the quoted half-spread; share of quotes inside the bid/ask | out-of-sample median \|model − mid\| 4.4–15.6 half-spreads; 3.0–12.3% inside (`data_store/surface/<session>/scorecard.json`) |
| M14 | The arbitrage-free SVI surface on held-out strikes, same quotes | median 0.45–1.98 half-spreads; 32–80% inside the bid/ask; 0.55 and 80.0% on 2026-10-16 |
| M14 | Static arbitrage in the stored quotes | executable at the bid/ask on all 5 sessions; 160–629 quotes per session in an executable row (`surface.json`) |
| M14 | VIX replicated from the session's own SPX chain (Cboe rules) against the official close | 26.66 / 30.27 / 15.02 / 16.49 / 19.70 against 26.95 / 31.05 / 15.32 / 16.40 / 19.99 (`vix_replication.json`) |
| M4, M15 | Rough Heston against Heston, joint fit, same out-of-sample quotes | rough lower on 3 of 5 sessions (1.42 vs 1.98, 1.41 vs 1.54, 1.09 vs 2.96 vp); H = ½, identical, on 2 (`diagnostics/summary.json`) |
| M5 | VX futures at their settlement, out of sample | 1.85–3.18 pts |
| M7 | β on the first chain after the estimation set (2026-10-16); β implied by PUT / WPUT | 0.70; 0.80 / 0.80 overall, 0.63–0.90 / 0.75–0.89 by full year |
| M8 | Canonical backtest 2018-01-02 → 2025-03-24 | portfolio −$222,177, Sharpe −0.99 (95% stationary-bootstrap CI −1.81 to −0.30); S1C and S4 intervals also exclude zero, negative (`inference/summary.json`) |
| M8, M17 | Data snooping and timing | Hansen SPA p 0.65 for the six published series against cash (StepM: none); S3 timing p 0.078 against circular shifts (Holm 0.47), 0.057 against trade shuffles (Holm 0.34); S3's largest DSR over every N and trial variance 0.27, and 0.000 at the ledger's N = 162; PBO is low (0.01) because the configurations differ by persistent level effects, which the summary explains is not evidence of skill |
| M8 | Co-movement with real-settlement benchmarks | the always-short straddle (the mark's premium factor) has a daily correlation of 0.55 with WPUT over the window (0.63 / 0.61 / 0.48 in 2018 / 2020 / 2022) |
| M9 | 21-day variance forecasts, 2012-04-30 → 2026-09-18 (3,618 origins) | PDV QLIKE 0.461, 6th of 7 (VIX Mincer–Zarnowitz best at 0.378); no Diebold–Mariano difference against PDV significant at 5%; PDV inside the 90% model confidence set (`forecast_eval.json`) |
| M8 | Reproduction from the captures | `make reproduce` plus `--full`: 14 of 14 artifacts bit for bit (docs/reproduce.md) |
| M11 | Classifier against persistence | 74.9% against 87.0% out of sample (the shipped model's evaluation) |
| M18 | Forward test at executable prices | no ledger entry committed yet: the first is written with the next accepted fit |

## 4. Ongoing monitoring

Controls that run every session in the daily job (`scripts/cloud_refresh.py`, `.github/workflows/daily_refresh.yml`):

| Control | Threshold | Action |
|---|---|---|
| Session guard | the live book must be the session being stored | capture and fit refused |
| Quote-state gate | ≥ 80% of near-ATM lines two-sided | fail closed with a reason; no fit |
| Calibration quality gate (M4) | thresholds in §2 M4 | fit rejected, chain not stored |
| Calibration staleness | newest accepted fit older than 3 NYSE sessions | job fails; site published with a staleness banner |
| VIX-family cross-source check (V1, V2) | Yahoo and Cboe differ by > 0.25 pts on a shared session | warning with the rows; Cboe's value stored |
| Rate curve (M19) | curve older than 7 days or missing | warning; the fit falls back to ^IRX and records it |
| Data gate | the data and capture tests | gates the commit of what was captured (chain, pickle, labels, Cboe books, forward ledger) |
| Artifact gate | surface, diagnostics, Greeks and scenario tests | gates a second commit of the derived artifacts |
| Presentation gate | the dashboard tests | gates only the deploy, never a capture |
| Site freeze | any page reporting a loader error or an SRI mismatch | publish refused |

Model-level checks with stated thresholds are computed at every build on the dashboard's Model Risk page
(`MONITOR_LIMITS` in `dashboard/app.py`): fit age in sessions, parameters on a bound, out-of-sample RMSE (all and worst
expiry within 30 days), VIX-index RMSE, Feller (informational), whether the Greeks surface comes from the headline fit,
β drift of the latest chain from the frozen 0.82, label lag, the classifier against persistence, Heston's median
|model − mid| in half-spreads, the VIX replication gap and the share of quotes in executable static arbitrage. On the
data at this commit, 5 of the 13 checks breach their limits: the worst expiry within 30 days (7.46 vol pts on
2026-10-30), Heston's half-spread error (6.3), β drift (0.702 against 0.82), the classifier (12.1 points below
persistence) and executable arbitrage (6.8% of scored quotes). Day-over-day parameter
z-scores are not computed: with Gauss–Newton errors of 12–19% on κ and five sessions, they would not be informative yet.

Controls in CI (`tests.yml`, `docs/engineering.md`): the full suite, lint and type gates against baselines that may
only shrink, property tests (10× examples weekly), a simulation of the bot's next session through all three gates, an
accessibility gate on the frozen site (axe-core, 0 violations; no horizontal scroll from 320 to 1920 px), a benchmark
smoke test, `make reproduce` weekly, and a dependency audit (report only).

## 5. Effective challenge

What partly substitutes for an independent validator: numerical references written independently of the engine (M1);
external libraries as dev-only oracles (QuantLib for Heston, Bates and rough-Heston lifting error; Let's Be Rational
for implied vols); property tests that state the theory rather than the implementation, with mutation checks that show
they can fail; rebuilding every committed artifact from the captures (`make reproduce`); an external market check of
the backtest's mark (Cboe strategy indices); a committed trials ledger so the deflated Sharpe counts every
configuration tried; and eight correction rounds (findings log below), each re-run against the same window and published
with its effect. What does not: a second person who did not build the models.

## 6. Findings log (C16–C23)

Severity: **high** changed a published headline number or its sign; **medium** changed a secondary published number
or a production control; **low** neither. Each round's postmortem, regression tests and commits are in
[docs/corrections.md](../corrections.md).

| ID | Date | Component | Severity | Finding | Effect on the published result | Status | Commit |
|---|---|---|---|---|---|---|---|
| C16 | 2026-06-11 | M9, M8 | high | Look-ahead in the PDV forecaster, a circular `vvix` feature, retrospective Kelly sizing | headline +19% was a label-noise artifact | fixed | `47e1ffe` |
| C17 | 2026-06-11 | M11 | medium | Classifier never benchmarked against persistence | loses to "predict yesterday"; demoted | fixed | `8c3ba9b` |
| C18 | 2026-06-14 | M8 | high | Strike-rolling bug: same-day re-entry meant positions never rolled | S1 −$1.53M → +$463K, portfolio −9.1% → −19.0% | fixed | `3d52f42` |
| C19 | 2026-06-20 | M8, V1 | low | S2 stop-loss propagation, dead code, data-gap repair | canonical numbers unchanged | fixed | `e1cfbd0` |
| C20 | 2026-08-04, 2026-09-05 | M4 | high | Calibration selector ranked archived fits on SPX RMSE with no data-sufficiency check | gate on ≥ 3 VIX tenors and provenance on every fit | fixed | `35690a2`, `9058371` |
| C21 | 2026-09-05 | M8, V1 | high | The cash hurdle was a unit-test fixture; straddles marked one session too rich; classifier output published as rule labels | portfolio −19.0% → −26.1%, Sharpe −1.57 → −1.80 | fixed | `15b4db5`, `90dc851` |
| C22 | 2026-10-19 | all | high | Full-system audit (~270 findings): ATM vol was the VIX, free hedging, VIX indices fitted as futures, ATM-call anchors, flat carry, a self-imposed Feller penalty, a month-long calibration outage | NAV $738,794 → $783,908, Sharpe −1.80 → −0.88; S1 +$693,938 → −$287,644 | fixed | `a51eccc` |
| C23-D1 | 2026-10-20 | V1, V2, M8, M10 | high | Labels, backtest and calibrator read Yahoo's copies of the VIX family (8 VVIX sessions missing, holiday rows) | portfolio −$216,092 → −$222,177, Sharpe −0.88 → −0.99; 6 labels move | fixed | `c63e515`, `2566553` |
| C23-M1 | 2026-10-20 | M4 | medium | The σ ≤ 2 box sets the headline fit; its cost was not published | none (published with its σ profile) | quantified | `958be69` |
| C23-M2 | 2026-10-20 | M3, M4, M14 | medium | No error measured against the bid–ask; no static-arbitrage screen | Heston 4.4–15.6 half-spreads out of sample; executable arbitrage on all 5 sessions | fixed | `49f5fd0`, `e6a92bc` |
| C23-S1 | 2026-10-20 | M8, M17 | high | No statistical inference on the backtest | portfolio Sharpe interval −1.81 to −0.30; no series beats cash | fixed | `1c886e3` |
| C23-R1 | 2026-10-21 | M6, M19, C7 | medium | Flat R = 1.5% in the hedger; one r, q for every Greeks maturity (1-year forward 0.50% low) | C7 cumulative P&L 167.95 → 168.34 pts | fixed | `ca36d24` |
| C23-O1 | 2026-10-20, 2026-10-21 | operations | medium | A presentation test could fail the commit of an accepted session (the 2026-10-19 book was lost); three tests pinned the committed session | none; the next-session simulation now runs in CI | fixed | `9c486c3`, `a678e57` |
| C23-O2 | 2026-10-21 | operations | low | `make reproduce` refitted from a deleted scratch pickle and with the wrong optimiser | none (the refits failed; now 14 of 14 bit for bit) | fixed | `741dfe3` |
| C23-U1 | 2026-10-21 | dashboard | low | The CI accessibility gate audited two retired pages and could not inject axe under the pages' CSP; every page scrolled horizontally at 320 px when the view was a session behind; the Risk page printed "r = —%" for the per-maturity carry; the ML classifier sat on the Overview | none | fixed | `113efa5` |
| C23-E1 | 2026-10-20 | M3 | low | `year_fraction` raised on an origin with no time in it | none (the path raised; it never returned a wrong T) | fixed | `ba360c8` |
| C23-E2 | 2026-10-20 | V4 | low | pip-audit: known advisories in idna 3.11, urllib3 2.6.3, werkzeug 3.1.8 | none on a number; urllib3 and idna are on the download path | fixed (floors idna ≥ 3.15, urllib3 ≥ 2.8.0, werkzeug ≥ 3.1.9) | `a92308c` |
