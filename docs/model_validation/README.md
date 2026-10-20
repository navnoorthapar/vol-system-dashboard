# Model validation pack

**Status:** skeleton with the evidence available at C22 (commit `a51eccc`, 2026-10-19) plus the C23 engineering
checks. Sections marked **[C23]** are filled when the C23 results are integrated. Every number here is quoted from a
committed artifact, the README, or a run recorded in `docs/`; where a number comes from the README, the README is the
source and this file must change with it.

## What this document is, and is not

It organises the project's existing evidence the way bank model-risk guidance does, so a reader who knows that
discipline can find what they are looking for. The structure and vocabulary (model inventory, materiality,
conceptual soundness, outcomes analysis, ongoing monitoring, effective challenge, vendor and third-party products)
follow the Federal Reserve / FDIC / OCC **Supervisory Guidance on Model Risk Management**, the attachment to SR letter
26-2 (17 April 2026, <https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm>, checked 2026-10-20), which
supersedes and replaces SR 11-7 (2011) and SR 21-8 (2021).

SR 26-2 is supervisory guidance for banking organisations ("This guidance does not set forth enforceable
standards", §I), and this project is not a bank. Nothing here claims compliance with it. The most important
departure from the guidance is structural: **the developer and the validator are the same person**, so there is no independent validation function. Section 5
lists what partly substitutes for it (independent numerical references, property and mutation tests, reproduction
from captures) and what does not.

## 1. Model inventory

"Use" is how the output reaches anything published: **production** (computed by the daily job or shown on the
dashboard), **research** (reported in the README, on no production path), **demoted** (shown only as a negative
result). "Materiality" is how much a published number depends on the model (high: a headline number or its sign;
medium: a secondary published number; low: diagnostics only).

| ID | Model | Code | Use | Materiality | Key limitations (as published) | Validation evidence |
|---|---|---|---|---|---|---|---|
| M1 | Heston / Bates pricing engine (Lewis 2001 contour, BS control variate, adaptive trapezoid) | `models/heston.py` | production: calibration, Greeks | high | 1e-10 DF·F quadrature target; 5e-9 F at the slowest-decaying corners of the box | `test_heston_reference.py` (independent quadratures), `test_properties_pricing.py`, mutation check |
| M2 | Implied-vol solvers: safeguarded Newton (`implied_vol_black`), bisection (`black76_implied_vol`) | `models/implied_vol.py`, `data/option_chain.py` | production: chain cleaning, calibration targets, Greeks | high | an ITM price whose time value is lost to rounding returns NaN by design; the bisection inverter brackets σ in (1e-4, 5) | `test_implied_vol.py`, `test_properties_iv.py` |
| M3 | Option-chain hygiene: two-sided quotes, parity forward and discount per expiry, quote-state gate, session dating | `data/option_chain.py` | production: every fit | high | no static-arbitrage filter (README, state of C22); Yahoo quotes carry no quote time for closing books | `test_option_chain.py`, `test_lookahead.py` |
| M4 | Joint Heston calibrator (SPX z-grid anchors + VIX index leg; DE + L-BFGS-B) and its quality gate | `calibration/joint_calibrator.py` | production: daily fit, headline | high | σ on its 2.0 box ceiling in 2 of 5 v2 fits; Feller violated in all 5; VIX-options leg off (w3 = 0) | `test_joint_calibrator.py`, `make reproduce` |
| M5 | VIX index and VX futures under Heston/Bates (exact CIR Laplace transform) | `models/heston.py` | production: VIX leg; VX futures out of sample | medium | Heston misses real VX futures by 1.9–3.2 pts out of sample (README) | `test_heston.py`, `test_properties_pricing.py` (Jensen bound) |
| M6 | Greeks monitor (BS Greeks at the model IV, QV convexity, vomma flag) | `greeks/risk_monitor.py` | production: Greeks page | medium | flat carry on the grid (one r, q for every node); the vomma shock scales with the fitted σ, which is on its 2.0 ceiling in the headline fit (`model_iv_shock`) | `test_risk_monitor.py`, `make reproduce` |
| M7 | β = ATM vol / VIX proxy (0.82, from four 2026 chains plus one live capture) | `backtest/backtest_engine.py` | production: every backtest P&L | high (it decides the sign of S1) | held constant over 2018–2025; the first later chain (2026-10-16) gives 0.70 | `run_backtest --estimate-beta`, β grid |
| M8 | Backtest engine and signals (S1, S1C, S2, S3, S4; delta-hedged 30-day straddles, mark-to-model) | `backtest/`, `signals/signal_engine.py` | production: backtest page (a negative result) | high | all P&L mark-to-model at β·VIX; no option prices inside the window | `test_backtest_accounting.py`, `test_backtest_reproducibility.py` |
| M9 | PDV volatility forecaster (EWMA approximation of Guyon & Lekeufack) | `models/pdv.py` | production: S1/S1C inputs | medium | pooled R² 0.29 mostly tracks the vol level; walk-forward GARCH(1,1) 0.25 (README) | `test_pdv.py` |
| M10 | Rule regime labels (R0/R1/R2; VVIX vs its rolling 252-day 80th percentile) | `signals/regime_classifier.py` | production: labels, backtest gate | medium | 5.9% of labels flip within measurement noise (README) | `test_regime_classifier.py`, `test_lookahead.py` |
| M11 | XGBoost regime classifier | `signals/regime_classifier.py` | demoted | low | loses to "predict yesterday" (75.0% vs 86.7% OOS) | `test_regime_classifier.py` |
| M12 | NN surrogate pricer | `models/nn_pricer.py` | research | low | trained domain (README: κ 0.5–15, θ 0.01–0.2) narrower than the calibrator's box (`config.HESTON_BOUNDS`: κ 0.1–20, θ 1e-4–1); the README's 14–23× speed-up is not re-measured here **[C23]** | `test_nn_pricer.py` |
| M13 | Two-factor Quintic OU (SPX MC, polynomial VIX²) | `models/quintic_ou.py`, `calibration/quintic_calibrator.py` | research | low | the stored fit did not converge; ε on its −0.99 bound | `test_quintic_ou.py`, `test_quintic_calibrator.py` |

**Vendor and third-party inputs (SR 26-2 §VII).**

| ID | Source | What it supplies | Checks in place | Known issues |
|---|---|---|---|---|
| V1 | Yahoo Finance (`yfinance`) | ^GSPC, VIX family, ^VVIX, ^IRX, SPX and VIX option chains | two-sided quotes only, own IVs on parity forwards, quote-state gate, session dating | the Yahoo series are not filtered to NYSE sessions before storage; VVIX gaps are forward-filled (`regime_classifier.py`); cross-source agreement with Cboe's files **[C23]** |
| V2 | Cboe (index CSVs, VX futures, delayed quotes) | VIX history (overlay on `vix_daily`), VX settlements | NYSE-session filter | the overlay corrects `vix_daily` only; the backtest, delta hedger and labels read the Yahoo `vix_term_structure` table (`backtest_engine.py`, `delta_hedger.py`, `run_backtest.py`, `regime_classifier.py`) |
| V3 | Library numerics (NumPy, SciPy, pandas) | every computation | exact lock (`requirements-lock.txt`), CPU-tolerant artifact comparison (`tests/artifact_tolerance.py`) | pip-audit 2026-10-20: 7 advisories in idna, urllib3, werkzeug (none in the numerical stack); `docs/engineering.md` |

## 2. Conceptual soundness

One subsection per model, with the design, its assumptions, the alternatives considered and the developmental
evidence. M1, M2, M4 and M7 are written; the others follow the same headings **[C23]**.

### M1 Pricing engine

- **Design.** Lewis (2001) contour integral on Im u = −1/2 with a Black-Scholes control variate, on the "little trap"
  characteristic function (Albrecher et al. 2007) rewritten so σ never divides outside a log1p (stable to σ → 0).
  Adaptive trapezoid: truncation from |φ| on a geometric scan, the step halved until two estimates agree to 1e-10
  DF·F. No FFT.
- **Why this and not Carr-Madan.** The Lewis contour needs only E[(S_T/F)^{1/2}], finite for every Heston/Bates
  parameter and maturity; Carr-Madan damping α = 1.5 needs E[S_T^{2.5}] and breaks beyond the moment-explosion time.
  The pre-C22 fixed 512-point Carr-Madan grid was off by 0.8 vp at 7 days and 3.4 vp at 1 day (README).
- **Developmental evidence.** Independent references written from scratch (Riccati ODE, Gil-Pelaez, Gauss-Legendre
  Lewis, QAWF, ncx2) over 47 parameter sets, 7 maturities, 13 strikes: price error < 1e-6·F, IV error < 0.01 vp
  (`test_heston_reference.py`). Agreement with an external library (QuantLib) as a dev-only oracle:
  **[C23]**. Properties over the whole parameter box (`test_properties_pricing.py`, docs/engineering.md):
  no static arbitrage in strike, no calendar arbitrage, put-call parity across the two Lewis engines, the martingale
  condition of both characteristic functions, the σ → 0 Black limit, Bates ⊃ Heston, the time-change and homogeneity
  invariants, and that the public clip moves no price by more than 2e-9 DF·F. 11 of 12 curated mutants of the engine and
  the solver are killed by these properties (the survivor moves 19 of 400,000 exact-price inversions by at most
  2.1e-11 relative, below every property tolerance).

### M2 Implied-vol solvers

- **Design.** Inversion of the out-of-the-money price only (parity first), Newton on transformed objectives below and
  above the Manaster-Koehler inflection point, every step safeguarded by a bisection bracket; ill-posed or unresolved
  inputs return NaN with a status code, never a boundary value.
- **Evidence.** `test_implied_vol.py` (example-based). Property tests: OTM round trip within
  1e-9 relative (measured worst 2.4e-12 over 100,000 random inputs), ITM within the documented 1e-3 resolution bound or
  an explicit status, monotone in price, call/put symmetric; the chain's bisection inverter agrees with the Newton solver
  inside its domain (`test_properties_iv.py`).

### M4 Joint calibrator

- **Design.** Vega-normalised SPX price errors on z-grid anchors (OTM against the parity forward, ≥ 3 wing puts per
  expiry) plus the MSE of the four Cboe spot-starting variance-swap indices; active legs normalised; DE (50 members,
  maxiter 200, seed 42) then one L-BFGS-B polish. Quality gate: ≥ 4 SPX expiries, ≥ 30 options, a put wing in ≥ 3
  expiries, ≥ 3 VIX tenors, a snapshot dated to the spot session, model_version 2.
- **Known weaknesses.** σ is on its 2.0 ceiling in 2 of the 5 v2 fits (2026-03-24 and 2026-10-16, `pinned_params`
  in the pickles), so the box, not the market, sets one of the five parameters there; Feller is violated in all five.
  The VIX-options leg is off (w3 = 0). **[C23]**: the fit without the σ box, parameter standard errors and
  identifiability (multi-start least squares).

### M7 β proxy

- **Design.** ATM 30-day implied vol = β × VIX, β = 0.82 estimated from real 2026 chains, constant over the backtest.
- **Why it matters.** S1 moves from +$2.63M at β = 1.00 to −$545K at β = 0.78; the portfolio is negative at every β
  tested (`data_store/backtest/beta_sensitivity.csv`). The first chain after the estimation set gives β = 0.70.
  **[C23]**: time-varying β, real-price marks.

## 3. Outcomes analysis

| Model | Outcome measured | Result (source) |
|---|---|---|
| M4 | Out-of-sample SPX IV RMSE on the quotes not used as anchors, five v2 sessions | 1.98 / 1.54 / 1.29 / 1.83 / 2.96 vp for 2026-03-24 / 03-27 / 05-29 / 06-18 / 10-16 (`leg_losses.spx_oos_iv_rmse` in the committed pickles; README table) |
| M4 | Model error in units of the quoted half-spread; share of quotes inside the bid/ask | **[C23]** (not stored by the v2 pickles) |
| M5 | VX futures at their settlement, out of sample | 1.85–3.18 pts (README) |
| M7 | β on the first chain after the estimation set (2026-10-16) | 0.70 against 0.80–0.84 in the set (README) |
| M8 | Canonical backtest 2018-01-02 → 2025-03-24 | portfolio −$216,092, Sharpe −0.88; S3 timing null p = 0.075 (README) |
| M8 | Reproduction from the committed captures | `make reproduce`: docs/reproduce.md |
| M11 | Regime classifier vs persistence | 75.0% vs 86.7% out of sample (README) |
| all | **[C23]** statistical inference (bootstrap CIs, deflated Sharpe), forward ledger, real-price marks | |

## 4. Ongoing monitoring

Controls that run today, every session, in the daily job (`scripts/cloud_refresh.py`):

| Control | Threshold | Action |
|---|---|---|
| Quote-state gate | ≥ 80% of near-ATM lines two-sided | fail closed with a reason; no fit |
| Calibration quality gate (M4) | thresholds in §2 M4 | fit rejected, chain not stored |
| Calibration staleness | newest accepted fit older than 3 NYSE sessions | job fails; site published with a staleness banner |
| Publish path | data tests gate the commit; presentation tests gate the deploy | a failing presentation test can no longer lose the captures |
| Site freeze | any page reporting a loader error or an SRI mismatch | publish refused |

Controls added in C23 (CI, `docs/engineering.md`): lint and type gates against shrinking baselines, property tests
(10× examples weekly), a weekly rebuild of the backtest, Greeks, C7 and calibration artifacts (`make reproduce`), an accessibility gate on the
frozen site, and a vulnerability report (report only). **[C23]**: model-level KPIs with thresholds (parameter on a bound,
day-over-day parameter z-scores, β outside its estimation band, next-session out-of-sample error).

## 5. Effective challenge

What partly substitutes for an independent validator: numerical references written independently of the engine
(M1), property tests that state the theory rather than the implementation, mutation checks that show those tests can
fail, rebuilding the backtest, Greeks, C7 and calibration artifacts from the captures (`make reproduce`), and seven correction rounds (findings log
below), each re-run against the same window and published with its effect on the headline. What does not: a second
person who did not build the models. **[C23]**: external oracle tests (QuantLib, Let's Be Rational) as dev-only
dependencies.

## 6. Findings log (C16–C22)

Severity: **high** changed a published headline number or its sign (C20: which fit the site headlined); **medium**
changed a secondary published number or a production control; **low** neither. Effects are quoted from the README
corrections table.

| ID | Date | Component | Severity | Finding | Effect on the published result | Status | Commit |
|---|---|---|---|---|---|---|---|
| C16 | 2026-06-11 | M9, M8 | high | Look-ahead in the PDV forecaster (now refitted walk-forward per year), a circular `vvix` feature, retrospective Kelly sizing | headline +19% was a label-noise artifact | fixed | `47e1ffe` |
| C17 | 2026-06-11 | M11 | medium | Classifier never benchmarked against persistence | loses to "predict yesterday"; demoted from the trading loop | fixed | `8c3ba9b` |
| C18 | 2026-06-14 | M8 | high | Strike-rolling bug: same-day re-entry meant positions never rolled | contaminated every backtest C10→C17; S1 −$1.53M → +$463K, portfolio −9.1% → −19.0% | fixed | `3d52f42` |
| C19 | 2026-06-20 | M8, V1 | low | S2 stop-loss propagation, dead code, signal-summary coverage; data-gap repair | canonical numbers unchanged | fixed | `e1cfbd0` |
| C20 | 2026-08-04, 2026-09-05 | M4 | high | Calibration selector ranked archived fits on SPX RMSE with no data-sufficiency check (one-tenor VIX days scored best and headlined the site) | gate on ≥ 3 VIX tenors and provenance on every fit | fixed | `35690a2`, `9058371` |
| C21 | 2026-09-05 | M8, V1 | high | The cash hurdle was a unit-test fixture written into the tracked DB; straddles marked one session too rich; the bot published classifier output as rule labels and refitted a frozen surface | portfolio −19.0% → −26.1%, Sharpe −1.57 → −1.80 | fixed | `15b4db5`, `90dc851` |
| C22 | 2026-10-19 | all | high | Full-system audit (~270 findings, each reproduced before fixing): ATM vol was the VIX (β), hedge and financing costs free, VIX indices fitted as futures, ATM-call-only anchors, flat carry, a self-imposed Feller penalty, a broken VIX-options pricer, a month-long live calibration outage | NAV $738,794 → $783,908, Sharpe −1.80 → −0.88; S1 +$693,938 → −$287,644; headline Heston fit replaced | fixed | `a51eccc` |
| C23-E1 | 2026-10-20 | M3 | low | `year_fraction` raised on an origin with no time in it ('' or NaN); now the session close | none on a published number (the path raised; it never returned a wrong T) | fixed | `ba360c8` |
| C23-E2 | 2026-10-20 | V3 | low | pip-audit: 7 known advisories in idna 3.11, urllib3 2.6.3, werkzeug 3.1.8 | no numerical package affected; urllib3 and idna are on the daily download path | open: lock bump | `docs/engineering.md` |
| **[C23]** | | | | results of the C23 model, data and statistics work | | | |
