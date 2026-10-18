# Joint SPX/VIX Smile Calibration System

**Joint SPX/VIX volatility calibration, risk, and signal-research system, plus a fully documented negative trading result. Live at https://navnoorbawa.me**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-00ff88?style=flat-square)](https://navnoorbawa.me)
[![CI](https://github.com/navnoorthapar/vol-system-dashboard/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/navnoorthapar/vol-system-dashboard/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff88?style=flat-square)](LICENSE)

---

## What this is, and what it isn't

This is **calibration and risk infrastructure plus an honest research log**, not a profitable strategy.

The headline result is a **negative backtest, presented without adjustment**. Since March 2026 the system has been through
seven rounds of correction (C16–C22): look-ahead bias, a circular classifier feature, label noise, a strike-rolling bug that
had contaminated *every* prior P&L number, a calibration selector that rewarded broken data, a cash hurdle that turned out to
be a unit-test fixture, and finally (C22) a ~270-finding audit of every subsystem that, among other things, found the
backtest had been pricing ATM straddles at the VIX. Each correction made the result worse or inverted a thesis I had
previously believed. **That audit trail is the deliverable.** Curve-fitting a strategy to look profitable in-sample is easy;
understanding precisely why a plausible one *doesn't* work is the harder and more useful exercise.

The short version: the pricing and calibration engines are now validated against independent references, the risk
tooling is honest about what it measures, and **no signal in the system makes money robustly**. The one that is positive in
the canonical run (S3, +$56K over seven years) is a mark-to-model index-leg proxy whose sign depends on an estimated
parameter, and the former "profitable" short-VRP signal (S1) exists only if you price ATM options at the VIX.

---

## The headline numbers (must match the dashboard)

```
Backtest 2018-01-02 → 2025-03-24 (1,816 sessions) · $1M per sleeve · delta-hedged 30-day ATM straddles
  Portfolio = (S1C + S3 + S4) / 3        ATM vol = β · VIX,  β = 0.82 (estimated from real chains)

  NAV:               $783,908
  Cumulative P&L:    −$216,092   (−21.6%)
  Sharpe:            −0.88       (rf = 0 on excess P&L: collateral is assumed to earn ^IRX)
  Max drawdown:      −21.6%      (the portfolio never regained its starting NAV)
  Trades:            208         (S1C 123 + S3 39 + S4 46)
  Cash benchmark:    T-bills (^IRX) +19.1% over the same window

By signal (each on its own $1M sleeve):
  S1C Contrarian PDV (portfolio)   −$431,203   Sharpe −0.57   35% win   123 trades
  S3  Dispersion, index leg        + $56,471   Sharpe  0.18   49% win    39 trades   timing null test p = 0.035
  S4  Volatility risk premium      −$273,544   Sharpe −0.79   37% win    46 trades
  S1  PDV short-VRP (reference)    −$287,644   Sharpe −0.15   59% win    69 trades   (all short)
  S2  VIX term structure (ref.)    −$149,900   Sharpe −0.38   38% win    45 trades
```

Source: `data_store/backtest/full_results.pkl` (metrics) and `full_results.parquet` (daily NAV per sleeve), produced by
`python -m joint_vol_calibration.backtest.run_backtest`.

**The single most important caveat:** every P&L number is **mark-to-model**: Black-Scholes at σ_ATM = β·VIX, delta-hedged
daily. The database holds **zero option prices inside the 2018–2025 backtest window**; every chain it contains is a 2026
snapshot. β itself is estimated from the five real 2026 chains available (four in the DB, one live capture; 0.80–0.84,
mean 0.82) and assumed constant over 2018–2025. The signals are real; the dollars are model-implied, not traded fills.

**β is the number that decides the sign of S1.** The β sensitivity grid (`data_store/backtest/beta_sensitivity.csv`):

| β (ATM vol / VIX) | Portfolio NAV | Portfolio P&L | Sharpe | S1 (reference) | S3 |
|---|---|---|---|---|---|
| 1.00 (pre-C22 assumption) | $816,366 | −$183,634 | −0.87 | **+$2,625,730** | +$395,457 |
| 0.90 | $782,241 | −$217,759 | −0.97 | +$559,802 | +$216,321 |
| 0.85 | $778,938 | −$221,062 | −0.92 | −$49,549 | +$111,897 |
| **0.82 (estimate, canonical)** | **$783,908** | **−$216,092** | **−0.88** | **−$287,644** | **+$56,471** |
| 0.78 | $813,733 | −$186,267 | −0.69 | −$544,623 | −$12,493 |

The portfolio is negative at every β. S1's profit and most of S3's exist only because the VIX, a variance-swap rate that
contains the skew premium, overstates ATM implied vol by ~18%: selling straddles "at the VIX" books that premium as profit.

**Why the backtest stops at 2025-03-24:** that is the window every correction below was re-run against. Market data now runs
past it, so extending it is a re-run, not new research; the numbers here are the ones that have actually been audited. The
April 2025 tariff shock is therefore *not* in the backtest.

---

## The corrections (the actual story)

| # | Correction | Effect on the published result |
|---|-----------|--------|
| C16 | Walk-forward PDV per year; removed circular `vvix` feature; ex-ante Kelly | Headline +19% was a **label-noise artifact** |
| C17 | Benchmarked the regime classifier against a *persistence baseline* | Classifier **loses** to "predict yesterday's regime" → demoted from the trading loop |
| C18 | **Strike-rolling bug**: same-day re-entry meant positions never rolled; the engine held one stale-strike straddle marked past expiry | Contaminated **every** backtest C10→C17; fixing it **inverted my own thesis** (S1 −$1.53M → +$463K; portfolio −9.1% → −19.0%) |
| C19 | S2 stop-loss propagation, dead-code removal, signal-summary coverage; data-gap repair | Cleanup; canonical numbers unchanged |
| C20 | **Calibration selector rewarded broken data**: ranked archived fits on SPX RMSE with no data-sufficiency check, so 1-tenor VIX days scored *best* and headlined the live site | Gate on ≥ 3 VIX tenors + provenance on every fit |
| C21 | **The cash hurdle was a unit-test fixture** (seven synthetic ^IRX rows written into the tracked DB and forward-filled for five years); straddles were **marked one session too rich**; the daily bot published the demoted classifier's output as "rule labels" and refit a frozen option surface | Portfolio −19.0% → −26.1%, Sharpe −1.57 → −1.80, MaxDD −25.9% → −31.5%; S1 +$463K → +$694K, S1C −$404K → −$600K |
| C22 | **Full-system audit (~270 findings, each reproduced before fixing).** Backtest: ATM vol was the VIX; hedge open/unwind and financing were free; sizing off $1M not the sleeve's NAV; correlation netting was a no-op; S3's P&L was a z-score scaled by an assumed 2%; Sharpe charged cash twice. Signals: two different R2 gates; a hard-coded VVIX > 100 override; inverted S2 stop; PDV spread compared the 30-day VIX with a 1-day E\|r\| forecast. Calibration: VIX indices fitted as if they were futures; ATM-call-only anchors; flat r/q; a self-imposed Feller penalty; a broken VIX-options pricer. Live pipeline: a month-long calibration outage (below) | NAV $738,794 → **$783,908**, P&L −26.1% → **−21.6%**, Sharpe −1.80 → **−0.88**, MaxDD −31.5% → **−21.6%**; S1 +$693,938 → **−$287,644**; S3 +$23,212 (z-units) → **+$56,471** (index leg); headline Heston fit replaced (below) |

**The C18 inversion was the centrepiece of the first five rounds.** I had built a contrarian signal (S1C) on the premise
that my base signal S1 was a "−$1.53M catastrophe." That catastrophe was the bug; corrected, S1 looked like a profitable
regime-gated short-VRP harvester (+$463K after C18, +$694K after C21). **C22 took that away too:** priced at a realistic
ATM vol (0.82 × VIX instead of the VIX), with real hedge and carry costs, one consistent R2 gate and a horizon-matched PDV
forecast, S1 loses $288K. The "edge" was the skew premium inside the VIX, booked as if it could be sold at the money. The
lesson from C18 still holds, more strongly: *a thesis that holds only under a modelling shortcut is not a signal.*

### C22 waterfall: from the published C21 headline to today's number

`python -m joint_vol_calibration.backtest.c22_bridge --csv data_store/backtest/c22_bridge.csv` re-runs the engine
applying the C22 corrections one at a time, in a fixed order, starting from the C21 engine (the order matters; each row is
cumulative):

| Row | Correction applied (cumulative) | NAV | P&L | Δ P&L | Sharpe | MaxDD |
|---|---|---|---|---|---|---|
| — | **C21 as published** | $738,794 | −$261,206 | | −1.80 | −31.5% |
| 0 | C21 engine on today's signal code (upstream changes that cannot be switched off) | $734,577 | −$265,423 | −$4,217 | −1.84 | −32.1% |
| 1 | VIX tenors 9/30/93/184 d, 365-day year | $734,053 | −$265,947 | −$524 | −1.85 | −32.2% |
| 2 | Sharpe on excess P&L, rf = 0 | $734,053 | −$265,947 | $0 | −1.17 | −32.2% |
| 3 | ^IRX gaps filled with the previous print | $734,053 | −$265,947 | $0 | −1.17 | −32.2% |
| 4 | S3 proxy exit-day P&L and cost | $732,712 | −$267,288 | −$1,341 | −1.18 | −32.2% |
| 5 | Hedge open/unwind costs | $717,465 | −$282,535 | −$15,247 | −1.26 | −33.5% |
| 6 | Premium and hedge financing | $737,517 | −$262,483 | +$20,052 | −1.15 | −31.7% |
| 7 | Expiry settle / roll | $737,517 | −$262,483 | $0 | −1.15 | −31.7% |
| 8 | Open trade in the trade log | $737,517 | −$262,483 | $0 | −1.15 | −31.7% |
| 9 | Size off the sleeve's NAV | $788,089 | −$211,911 | +$50,572 | −1.36 | −24.1% |
| 10 | Real correlation netting | $805,966 | −$194,034 | +$17,877 | −1.34 | −22.2% |
| 11 | **ATM vol = 0.82 × VIX** | $913,923 | −$86,077 | +$107,957 | −0.37 | −15.3% |
| 12 | S3 = short SPX straddle (index leg) | $878,361 | −$121,639 | −$35,562 | −0.57 | −17.4% |
| 13 | **One adaptive R2 gate** (rolling 252-day 80th percentile of VVIX) | $744,114 | −$255,886 | −$134,247 | −1.15 | −25.7% |
| 14 | PDV spread on the 21-day RMS-vol forecast (= canonical run) | **$783,908** | **−$216,092** | +$39,794 | **−0.88** | **−21.6%** |

Two rows dominate, in opposite directions. β = 0.82 *helps* the portfolio (+$108K) because S1C and S4 are mostly long
straddles, which were bought "at the VIX" too dear. The single R2 gate *hurts* (−$134K): the backtest had traded a fixed
VVIX > 100 gate (≈ 48% of the window in R2) while the published labels used the adaptive gate (≈ 23%), so moving to one
gate puts a quarter of the window back into R1, where S1C buys straddles. S1 moves from +$722K (row 0) to −$508K at row
11: β alone removes it.

**S3 is the survivor, with caveats.** It is now priced as the index leg of a dispersion trade (long dispersion ⇒ short SPX
straddle) on the same engine, a proxy because there is no single-stock options data. Its timing beats 200 random-walk
null paths (p = 0.035), but it is positive only for β ≳ 0.8 (−$12K at β = 0.78), its Sharpe is 0.18, and below the
T-bill return. The pre-C22 "+$23K, positive at every scale" figure was a z-score times an assumed 2%, positive on 98% of
random-walk null paths, i.e. it measured mean reversion of a ratio, not option P&L.

S1C's pseudo-out-of-sample check (2013–2017, a window not used to design it) still fails under the corrected engine:
−$261,085, Sharpe −0.87, 29% win, 58 trades (`run_backtest --pseudo-oos`).

---

## The Calibration Finding

### Calibration provenance

The headline calibration is the **newest model_version 2 fit that passes the quality gate**, not the lowest-RMSE fit. Today
that is the 2026-06-18 session (SPX 7,500.58, VIX 16.40), `data_store/calibrations/joint_cal_2026-06-18.pkl`:

```
Heston parameters (2026-06-18 session):
  κ = 5.016    θ = 0.0613    σ = 1.591    ρ = −0.698    v₀ = 0.0173  (√v₀ = 13.1% spot vol)
  No parameter on a bound (ρ interior); no active constraints

  SPX smile RMSE:    0.85 vol pts in-sample (53 anchors, 6 expiries, 33 puts / 20 calls)
                     1.83 vol pts OUT-OF-SAMPLE (6,976 OTM quotes, 43 expiries; put wing 2.37)
  VIX index RMSE:    0.51 VIX pts (^VIX9D / ^VIX / ^VIX3M / ^VIX6M as spot-starting indices)
  VX futures RMSE:   1.84 VIX pts (8 real Cboe futures, out of sample: not fitted)
  Carry:             parity-implied per expiry; medians r = 4.27%, q = 0.71% (^IRX reference 3.80% cc)
  Feller:            2κθ = 0.615 < σ² = 2.532  VIOLATED, not imposed (a diagnostic, not a no-arbitrage condition)
  Optimiser:         DE (maxiter 200, popsize 10 → 50 members, seed 42) + one L-BFGS-B polish
```

All four committed SPX snapshots were recalibrated with the corrected calibrator; the fits are consistent with one another:

| Session | κ | θ | σ | ρ | v₀ | SPX in-sample / OOS (vp) | VIX index (pts) |
|---|---|---|---|---|---|---|---|
| 2026-03-24 | 6.140 | 0.0653 | 2.000* | −0.823 | 0.0732 | 1.207 / 1.980 | 0.888 |
| 2026-03-27 | 4.292 | 0.0733 | 1.833 | −0.785 | 0.0929 | 1.365 / 1.538 | 0.852 |
| 2026-05-29 | 5.233 | 0.0595 | 1.707 | −0.656 | 0.0130 | 0.911 / 1.285 | 0.523 |
| 2026-06-18 | 5.016 | 0.0613 | 1.591 | −0.698 | 0.0173 | 0.853 / 1.834 | 0.508 |

\* σ on its box ceiling (2.0), reported by the fit as an active constraint. Every fit violates Feller; none imposes it.

### What C22 changed in this story, and why

The previous version of this section reported the 2026-05-31 showcase fit (κ 1.77, θ 0.067, σ 0.46, **ρ pinned at −0.95**,
SPX 0.52 vp, "VIX futures" 0.53, Feller PASS) and drew a conclusion from it: ρ on its floor and σ on the Feller boundary
meant Heston fitted the joint surface with **three effective degrees of freedom**, and a **37.14 vol-pt** VIX-options RMSE
was a *structural* failure of Heston's CIR variance. **None of that survives C22**, and each piece failed for a concrete,
reproducible reason:

- **ρ at −0.95 was an artifact of the anchors.** 45 of the showcase's 54 anchors were ATM *calls* (the 9 top-vega strikes
  per expiry, OTM measured against spot), with no put wing beyond the shortest slice. With no wing to fit, ρ ran to its
  bound. The anchors are now the strikes nearest a grid in z = ln(K/F)/(σ_ATM√T) ∈ [−2.5, 1.5], OTM against the
  parity forward, with ≥ 3 wing puts (K/F < 0.97) per expiry. ρ lands at −0.66 … −0.82, interior, on every snapshot.
- **The "binding Feller constraint" was self-imposed.** The calibrator added a Feller penalty by default; Feller is not a
  no-arbitrage condition and the pricer is validated where it fails. With the penalty off (still available as
  `feller="soft"`), σ goes to 1.6–2.0 and the fit improves. The margin is reported as a diagnostic.
- **The VIX leg was fitting the wrong instrument.** ^VIX9D/^VIX/^VIX3M/^VIX6M are *spot-starting* variance-swap rates
  over 9/30/93/184 days; the leg compared them with the model's VIX *futures* curve at 9/30/91/182 days. At the published
  showcase parameters the correct index model misses the 2026-05-29 market by **2.95 pts**, not 0.53. The leg now uses
  `heston_vix_term_index`; real VX futures (now in the DB) are scored out of sample with the exact CIR Laplace transform.
- **Carry and time were wrong.** Flat r = 4.5% / q = 1.3% and Yahoo's implied vols (computed on a forward that violates
  put-call parity by 1.5–12 vp at one strike) are replaced by two-sided mids, per-expiry parity forwards and discounts, and
  our own Black-76 IVs. T runs from the quote session's close to the contract's settlement; a Sunday snapshot had made
  every T two days short.
- **The 37.14 vp came from a broken pricer.** VIX puts were priced by parity with √E[VIX²] (0.7–1.0 pts too low, sometimes
  negative), vegas were taken on spot VIX rather than the VIX forward, and the chain was stale. On a like-for-like basis
  (same session, same targets), Heston re-priced on the Quintic OU's 61 VIX options gives **35.2 vp** against the Quintic's
  **2.77 vp**, but Heston was not fitted to them (its VIX-options weight is 0) and on the Quintic's 66 SPX targets Heston
  is *better*: **1.21 vp vs 1.31 vp**. The VIX-options leg stays off because the DB holds VIX option quotes for too few
  sessions to fit it routinely, not because of a demonstrated structural failure.

The honest calibration finding is now narrower and better supported. With carry, anchors and instruments right, Heston
fits the SPX surface to ~1 vp in-sample and ~1.3–2.0 vp out-of-sample and the VIX index term structure to ~0.5–0.9 pts,
**but only with a vol-of-vol (σ ≈ 1.6–2.0) that violates Feller by a wide margin** and with κ ≈ 4–6, i.e. variance
half-lives of 1.4–1.9 months. The model reprices the 8 real VX futures 1.8 pts off out of sample, and prices VIX options poorly
when asked (35 vp). Joint SPX/VIX calibration is hard for one-factor diffusive SV models, as the literature has said for
years (Guyon 2020; Abi Jaber, Illand & Li 2022); this repo shows *where* it is hard with numbers you can regenerate.

### The Quintic OU comparison

`data_store/calibrations/quintic_cal_2026-03-24.pkl` (two-factor Quintic OU, Abi Jaber & Li 2025, with the one-factor
polynomial of Abi Jaber, Illand & Li 2022) against the same-session Heston fit
(`data_store/calibrations/baseline_2026-03-24.json`, built by `scripts/heston_quintic_baseline.py`):

| Same session (2026-03-24), same targets | Heston (v2 fit) | Quintic OU |
|---|---|---|
| SPX IV RMSE, 66 Quintic targets | 1.21 vp | 1.31 vp |
| VIX options IV RMSE, 61 options | 35.2 vp (not fitted) | 2.77 vp |
| Model E[VIX_T] vs VIX-option parity forwards (8) | 2.30 pts | 0.38 pts |

The Quintic fit **did not converge** (DE hit its iteration limit; the polish stopped at its limit) and its leverage
parameter ε sits on its −0.99 bound, so these are research numbers, not a solved joint calibration.

### Why the headline fit is chosen on data sufficiency, not RMSE

The daily bot recalibrates on a live snapshot, and the dashboard headlines one archived fit. Ranking archived fits on SPX
RMSE is wrong, and was wrong in production twice:

- **C20:** Yahoo intermittently returned `^VIX` without `^VIX9D/3M/6M`. With one tenor the VIX leg is one data point that
  Heston matches exactly, which frees all five parameters to fit the SPX smile alone, so the fits with *broken* data had
  the *best* RMSE (2026-07-23: one tenor, 0.049 vp headlined the site). Fix: ≥ 3 VIX tenors required.
- **C22:** the same failure on the SPX side. A fit from **one SPX expiry** (9 options) has a tiny RMSE because there is
  nothing to be jointly consistent with; the selector headlined 2026-09-11 (0.215 vp, 9 options) and would have headlined
  2026-09-25 (0.041 vp).

The gate (`is_acceptable_calibration`) now requires ≥ 4 SPX expiries, ≥ 30 options, a put wing in ≥ 3 expiries, ≥ 3 VIX
tenors, a snapshot dated to the spot session, and `model_version = 2`. The dashboard headlines the **newest** passing fit and
shows a staleness banner in NYSE sessions. Of the 40 archived `joint_cal_*.pkl` files, the 4 model_version 2 fits pass;
the 36 pre-C22 fits are kept as labelled legacy evidence and are never ranked.

### The live calibration outage (2026-09-05 → 2026-10), and its root causes

From 2026-06-16 to 2026-09-04 the bot never stored the chain it downloaded, so every "daily" fit refitted the newest
*committed* SPX snapshot, 5–78 days older than its VIX leg (the provenance backfill now records the chain's age in each
pickle). The 2026-09-05 fix (C21) made it use a live chain, and that exposed a second set of defects: from 2026-09-07 every
Monday–Thursday run was rejected at 176–273 vp, and the six fits it accepted (09-11, 09-25, 09-28, 10-02, 10-07, 10-09)
each used 9 options on a single expiry. All runs were green. Root causes, all fixed in C22:

1. Only the first 8 listed expiries were fetched (≈ 1.5 weeks of SPXW dailies), so after the 7-day floor one slice remained.
   Now the nearest listed expiry to each target (14/30/60/91/180/365 d) is fetched (`data/option_chain.py`).
2. The cron fired into Cboe's **empty after-hours book**; the code kept Yahoo's implied vol (a bisection artefact on an empty
   book) and fell back to `lastPrice`. Now only two-sided quotes are kept, never `lastPrice`; IVs are our own, on parity
   forwards; a book with < 80% two-sided near-ATM lines fails closed with a reason.
3. `cloud_refresh.main()` ignored every return value, so a rejected calibration (or an empty SPX download that silently
   re-published the 2026-06-18 close) was a green run. It now returns an exit code, fails the job on hard failures or when
   the newest accepted calibration is older than 3 sessions, and writes a status file and step summary
   (`--steps market,labels,calibrate,freeze` lets the option step run while the book is live).
4. The chain was stored as "the most recent snapshot" *before* the gate ran, and `r` was overwritten after the fit with a
   rate the fit never used. The chain is now stored only after acceptance, and the result records the r/q it used.

The tests that pin this are `tests/test_option_chain.py`, `tests/test_cloud_refresh.py`, `tests/test_joint_calibrator.py` and
`tests/test_data_pipeline.py`.

---

## System Overview

14 components, ~26,700 lines of Python outside the tests (package, dashboard and scripts), causal by construction and by
test (mutation-checked look-ahead tests in `test_lookahead.py`).

| Component | Description |
|-----------|-------------|
| **C1** Data Infrastructure | SQLite (schema v1) + Yahoo and Cboe downloaders; rule-based NYSE calendar; session-dated, idempotent option snapshots; ^IRX converted from bank-discount to a continuously compounded rate; option-chain hygiene (two-sided quotes, parity forwards, own Black-76 IVs) |
| **C2** Heston / Bates Engine | Lewis (2001) contour integral with a Black-Scholes control variate on the "little trap" CF (Albrecher et al. 2007), adaptive trapezoid, validated against independent quadrature; exact VIX index and VIX futures (CIR Laplace transform); Andersen (2008) QE Monte Carlo. No FFT |
| **C3** PDV Model | EWMA approximation of Guyon & Lekeufack (2023), not a replication (fixed half-lives, daily returns). 21-day RMS-vol forecast, walk-forward R² 0.29 (ex-2020 0.27); next-day \|r\| target R² 0.31 (ex-2020 0.20) |
| **C4** Joint Calibration | DE + L-BFGS-B on SPX z-grid anchors (put wing, parity carry) + VIX spot-index leg; headline 0.85 vp in-sample / 1.83 vp out-of-sample; quality gate on data sufficiency |
| **C5** NN Surrogate | Research prototype on no production path. Trained weights are not in the repo, so no speed or accuracy figure is claimed here (`NNPricer.benchmark()` reproduces them); the C22 fits (σ ≈ 1.6–2.0) lie outside its training box (σ ≤ 1.2). The VIX network is deprecated: the closed form it approximates is faster |
| **C6** Greeks Monitor | Vomma/vanna/QV surface from exact Heston prices of the headline fit; 79 of 90 grid nodes (11 below a one-tick price floor, recorded); per-node vomma-P&L instability rule (31 of 79 flagged); exact QV convexity |
| **C7** Delta-Hedge Simulation | 2020 attribution Γ + ν + vanna + volga + calendar Θ + financing; unexplained-variance ratio 0.0001 (run A); walk-forward PDV in run B |
| **C8** Regime Classifier | XGBoost 3-regime, holdout isotonic calibration; 75% out-of-sample vs 87% for "predict yesterday", **demoted, research-only** |
| **C9** Signal Engine | S1/S1C (PDV vs implied), S2 (VIX term structure), S3 (dispersion timing), S4 (VRP); one adaptive R2 gate |
| **C10** Backtest Engine | 2018-01-02 → 2025-03-24, β·VIX ATM proxy, hedge and carry accounting, sleeve-NAV sizing, ex-ante Kelly, S3 null test, sub-period stability (in-sample), HTML report |
| **C11** Regime-Switching PDV | Merton (1976) jumps fitted on post-R2 days; Lee-Mykland-style daily jump test (the old "BNS filter" was not the BNS test and is removed) |
| **C12** Bates SVJ + HMM | Bates (1996) CF and VIX formulas; the stored Bates fit did not converge, so it is not a result. Gaussian HMM with causal (filtered) inference, research only |
| **C13** Improvements | Adaptive VVIX gate (now the only gate), holdout isotonic calibration, portfolio Kelly. Withdrawn in C22: SVI/SSVI smoothing (never wired in; removed) and top-vega anchors (replaced by the z-grid) |
| **C13b** Quintic OU | Two-factor Quintic OU (Abi Jaber & Li 2025): Itô MC with exact Gaussian steps and Richardson extrapolation, polynomial VIX² pricer. 2026-03-24 fit: SPX 1.31 vp, VIX options 2.77 vp; not converged |

### Architecture

```
 Yahoo / Cboe ──► data/ (downloaders, NYSE calendar, option_chain hygiene) ──► data_store/vol_system.db
                                                                                    │
      ┌─────────────────────────────┬──────────────────────────┬───────────────────┤
      ▼                             ▼                          ▼                   ▼
 models/heston.py              models/pdv.py           signals/regime_*      calibration/
 models/quintic_ou.py          (walk-forward σ̂)        (rule labels, R2       joint_calibrator.py
 models/implied_vol.py               │                  gate, research clf)   quintic_calibrator.py
      │                              ▼                          │                   │
      │                     signals/signal_engine.py ◄──────────┘                   │
      │                              ▼                                              ▼
      │                     backtest/ (engine, c22_bridge, delta_hedger)    data_store/calibrations/
      ▼                              ▼                                              │
 greeks/risk_monitor.py ──► data_store/greeks/      data_store/backtest/            │
      └───────────────────────────────┴──────────────────┴───────────► dashboard/app.py ──► freeze_site.py ──► site
```

`scripts/cloud_refresh.py` is the only writer in the daily job (market data, rule labels, calibration); the dashboard is a
read-only renderer (the DB is opened `mode=ro`), and `freeze_site.py` refuses to publish a page that reports a loader error.

---

## Live Dashboard

**[https://navnoorbawa.me](https://navnoorbawa.me)**: four pages, served as frozen HTML from the stored artifacts:

- **Live Market**: SPX/VIX/VVIX, end-of-day rule regime (R0/R1/R2), the 21-day PDV forecast vs implied with a like-for-like
  percentile, the VIX term structure
- **Calibration**: the headline fit with its age, carry, anchors and gate status; SPX smile; RMSE decomposition; same-session
  Heston vs Quintic comparison
- **Greeks Monitor**: vomma heatmap on the surface's own ln(K/F) grid, unstable nodes, vanna/QV by maturity, provenance
- **Backtest**: equity curves per sleeve with the T-bill benchmark, per-signal attribution and the C16→C22 audit table

---

## The Regime Classifier: why it's demoted

The labels are defined by *same-day observables* (`rv_20d` vs VIX for R0/R1; VVIX above its rolling 252-day 80th percentile
for R2), so "predict yesterday's regime" is a legitimate causal predictor. On 2020-01-02 → 2025-12-31 it scores **86.7%**.
The shipped XGBoost classifier (trained 2010–2019, no `vvix` feature) scores **78.7% in-sample and 75.0% out-of-sample**
(75.1% walk-forward); adding lagged VVIX gives 72.4%, and adding yesterday's label as a feature still only reaches 82.9%.
The ML model adds *negative* value, so it is **research-only**; the backtest uses the lagged rule labels directly.
(Before C22 these figures were 86.2% / 63.4% / 90.0%, measured on the fixed VVIX > 100 labels with a full-sample PDV model
that carried look-ahead; reproduce today's with `classifier_accuracy_report()`.)

The label ontology sets its own noise floor: perturbing the inputs within their measurement noise flips **5.9%** of labels,
and **10.2%** of days sit within one vol point of the R0/R1 cut (`regime_label_noise_audit()`).

---

## Data Sources

All free; no Bloomberg or paid feed. Counts are from the committed `data_store/vol_system.db`, whose market history ends
2026-06-18 (the daily job extends its own copy).

| Dataset | Source | Coverage | Rows |
|---------|--------|----------|------|
| SPX OHLCV | Yahoo (`^GSPC`) | 2010-01-04 → 2026-06-18 | 4,140 |
| VIX daily | Yahoo, cross-checked against Cboe's official files | 2010-01-04 → 2026-06-18 | 4,167 |
| VIX term structure | Yahoo (`^VIX9D` from 2011-01-03, `^VIX`, `^VIX3M`, `^VIX6M`, `^VVIX`) | 2010-01-04 → 2026-06-18 | 20,431 |
| SPX options | Yahoo, 4 end-of-day chains (sessions 2026-03-24, 03-27, 05-29, 06-18) | none inside the backtest window | 61,815 |
| VIX options | Yahoo, same 4 sessions | none inside the backtest window | 1,896 |
| VX futures | Cboe per-contract files (174 monthly contracts expiring 2013-01 → 2027-06) | 2013-01-02 → 2026-10-09 | 30,963 |
| VX final settlements | Cboe (expiry-day settles; VRO before mid-2013) | 2010-01-20 → 2026-09-16 | 201 |
| T-bill rate | Yahoo (`^IRX`, stored as cc rate + raw discount yield; negative 2020 bills kept) | 2010-01-04 → 2026-09-04 | 4,192 |

The daily job now stores every SPX chain that passes the calibration gate under its quote session (and the data layer can
persist chains and market partitions as parquet), so the options history grows going forward; it cannot reach back into the
backtest window, which is *why* the backtest is mark-to-model. The Cboe downloader was rebuilt in C22 (the old endpoints
returned 403 and the parser read fields the payload does not have).

---

## Running Locally

```bash
git clone https://github.com/navnoorthapar/vol-system-dashboard
cd vol-system-dashboard
pip install -r requirements.txt pytest
python dashboard/app.py        # read-only preview on 127.0.0.1 (first free port of 5000, 5001, 5050, 8080, 8888)
python dashboard/freeze_site.py .site-build   # the static site exactly as published
```

The repository ships the `data_store/` artifacts (DB, calibration pickles, Greeks surface, rule labels, PDV model, backtest
results), so the dashboard runs with no network access and no recalibration. Rendering never writes to `data_store/`.

### Regenerating every artifact

| Artifact | Command |
|---|---|
| `data_store/backtest/full_results.{pkl,parquet}` + HTML report | `python -m joint_vol_calibration.backtest.run_backtest` |
| `data_store/backtest/beta_sensitivity.csv` | `python -m joint_vol_calibration.backtest.run_backtest --beta-sensitivity --csv data_store/backtest/beta_sensitivity.csv` |
| `data_store/backtest/c22_bridge.csv` | `python -m joint_vol_calibration.backtest.c22_bridge --csv data_store/backtest/c22_bridge.csv` |
| `data_store/backtest/delta_hedge_2020.parquet` (C7) | `python -m joint_vol_calibration.backtest.run_backtest --c7` |
| β estimate from the DB chains (add `--live` for a Yahoo chain) | `python -m joint_vol_calibration.backtest.run_backtest --estimate-beta` |
| S1C pseudo-OOS 2013–2017 | `python -m joint_vol_calibration.backtest.run_backtest --pseudo-oos` |
| `data_store/calibrations/joint_cal_<session>.pkl` | `python -c "from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator as J; c = J(as_of_date='2026-06-18'); c.calibrate(); c.save()"` |
| `data_store/calibrations/quintic_cal_2026-03-24.pkl` | `python -m joint_vol_calibration.calibration.quintic_calibrator` |
| `data_store/calibrations/baseline_2026-03-24.json` | `python scripts/heston_quintic_baseline.py 2026-03-24` |
| `data_store/greeks/greeks_surface.parquet` | `python -m joint_vol_calibration.greeks.risk_monitor --calibration data_store/calibrations/joint_cal_2026-06-18.pkl` |
| `data_store/pdv_model.pkl` | `python -m joint_vol_calibration.models.pdv --as-of <date>` |
| Daily refresh (labels, calibration, site) | `python scripts/cloud_refresh.py --steps market,labels,calibrate,freeze` |
| Calibration provenance backfill | `python scripts/backfill_calibration_provenance.py --apply` |

The backtest and calibration runs are seeded (seed 42) and deterministic on the same library versions.

### Testing

```bash
python -m pytest joint_vol_calibration/tests -q -p no:cacheprovider
# At the C22 commit: 1,342 passed, 8 skipped in ~5 min. The 8 skips are NN-surrogate tests
# that need the gitignored trained weights / training caches.
```

Every push runs the full suite in GitHub Actions (the **CI** badge above), so "passing" is machine-verified rather than a
hand-typed count. The suite includes mutation-checked look-ahead tests (`test_lookahead.py`), Heston prices against an
independent quadrature reference (`test_heston_reference.py`), Monte Carlo vs analytic cross-checks, the β / bridge /
accounting invariants of the backtest (`test_backtest_accounting.py`), and tests that no dashboard route writes to
`data_store/`.

### Requirements

```
Python 3.11+ · flask · pandas · numpy · pyarrow · scipy · scikit-learn · xgboost · hmmlearn · yfinance · matplotlib
torch (only for the research NN surrogate)
```

No QuantLib. No Bloomberg. No paid data.

---

## Known Limitations

- **All backtest P&L is mark-to-model.** Black-Scholes at β·VIX with one β (0.82) estimated from five 2026 chains and held
  constant over 2018–2025. β moves S1 by ~$3M across 0.78–1.00; the portfolio is negative at every β tested.
- **S3 is a proxy.** Only the index leg of the dispersion trade is priced (no single-stock options); its positive P&L needs
  β ≳ 0.8 and is below the T-bill return.
- **No signal is deployable.** The portfolio loses 21.6% while T-bills earn 19.1%; the former short-VRP "edge" is the skew
  premium inside the VIX.
- **Heston needs Feller-violating vol-of-vol** (σ ≈ 1.6–2.0) to fit SPX and the VIX indices together, misses the real VX
  futures by ~1.8 pts out of sample, and is not fitted to VIX options (leg off: too few sessions of VIX option quotes).
  The Quintic OU fit that does price VIX options did not converge.
- **The headline calibration is old.** It is the 2026-06-18 session (78 NYSE sessions before C22 landed) until the corrected
  live pipeline produces a passing fit; the dashboard shows the staleness.
- **Regime classifier loses to a persistence baseline** and is research-only.
- **Options history is thin**: four committed end-of-day chains (Mar–Jun 2026), none inside the backtest window.

---

## References

Implemented or directly followed in the code (see module docstrings for where and how the implementation departs):

- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency
  options. *Review of Financial Studies* 6(2), 327–343. (`models/heston.py`)
- Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options.
  *Review of Financial Studies* 9(1), 69–107. (`models/heston.py`, Bates CF and VIX)
- Lewis, A. L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. Working paper,
  Envision Financial Systems / OptionCity.net. (pricing engine in `models/heston.py`, `implied_vol.fourier_otm_price`)
- Albrecher, H., Mayer, P., Schoutens, W. & Tistaert, J. (2007). The little Heston trap. *Wilmott Magazine*, January, 83–92.
  (characteristic function)
- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational
  Finance* 11(3), 1–42. (QE Monte Carlo)
- Lord, R., Koekkoek, R. & van Dijk, D. (2010). A comparison of biased simulation schemes for stochastic volatility models.
  *Quantitative Finance* 10(2), 177–194. (full-truncation Euler)
- Carr, P. & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies* 22(3), 1311–1341. (jump term in the Bates VIX)
- Carr, P. & Madan, D. (1999). Option valuation using the fast Fourier transform. *Journal of Computational Finance* 2(4),
  61–73. (the damped-call formula behind the **pre-C22** pricer, which evaluated it on a fixed 512-point quadrature grid,
  not an FFT, and was off by 0.8 vp at 7 days and 3.4 vp at 1 day; replaced by the Lewis contour in C22)
- Guyon, J. & Lekeufack, J. (2023). Volatility is (mostly) path-dependent. *Quantitative Finance* 23(9), 1221–1258.
  (`models/pdv.py`, an EWMA approximation)
- Abi Jaber, E., Illand, C. & Li, S. (2022). The quintic Ornstein-Uhlenbeck volatility model that jointly calibrates SPX &
  VIX smiles. arXiv:2212.10917. (`models/quintic_ou.py`, polynomial and admissibility)
- Abi Jaber, E. & Li, S. (2025). Capturing smile dynamics with the quintic volatility model: SPX, skew-stickiness ratio and
  VIX. arXiv:2503.14158. (the two-factor model implemented)
- Guyon, J. (2020). The joint S&P 500/VIX smile calibration puzzle solved. *Risk*, April 2020. (context for the joint problem)
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics* 3,
  125–144. (`signals/regime_pdv.py`)
- Lee, S. S. & Mykland, P. A. (2008). Jumps in financial markets: a new nonparametric test and jump dynamics. *Review of
  Financial Studies* 21(6), 2535–2563. (daily jump test, adapted to daily data)
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160(1),
  246–256. (QLIKE scoring)
- Jäckel, P. (2015). Let's be rational. *Wilmott* 2015(75), 40–53; Manaster, S. & Koehler, G. (1982). The calculation of
  implied variances from the Black-Scholes model. *Journal of Finance* 37(1), 227–230. (implied-vol solver in
  `models/implied_vol.py`)
- Cboe. VIX and VIX term-structure index methodology (9/30/93/184-day constant-maturity indices); VX futures contract
  specifications.

---

## Author

**Navnoor Bawa**, 3rd year CS, Thapar Institute of Engineering & Technology

- Website: [navnoorbawa.me](https://navnoorbawa.me)
- Substack: [navnoorbawa.substack.com](https://navnoorbawa.substack.com)
- LinkedIn: [linkedin.com/in/navnoorbawa](https://www.linkedin.com/in/navnoorbawa/)
- GitHub: [github.com/navnoorthapar](https://github.com/navnoorthapar)

---

## License

Released under the [MIT License](LICENSE), © 2026 Navnoor Bawa. Free to use, copy, modify, and distribute with attribution.
Market data is sourced from free public feeds (Yahoo Finance, Cboe) and is subject to those providers' own terms.

---

*The negative backtest is presented without adjustment. Every correction (look-ahead → circular feature → label noise →
stale-strike bug → broken-data selector → test-fixture hurdle → VIX-as-ATM-vol) made the result worse or flipped a thesis.
The audit trail is the point.*
