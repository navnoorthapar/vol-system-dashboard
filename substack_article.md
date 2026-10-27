# ~~I built the 'holy grail of volatility modelling'~~ I tried joint SPX/VIX calibration — then found the bug that faked my P&L

> **C23 note (21 October 2026).** A second review round (C23) changed the data under the backtest and added what the
> first version lacked: evidence in the units a market maker uses, and statistics. What changed, and why:
>
> - **The VIX data.** The labels, the backtest and the calibrator had read Yahoo's copies of the VIX index family, with
>   8 VVIX sessions missing and rows on market holidays. Cboe's official files are now the record. On the corrected
>   data the portfolio is **−22.2% (NAV $777,823), Sharpe −0.99**, against −21.6% and −0.88 in the C22 note below;
>   S1 is −$307K, S1C −$454K, S3 +$61K. The step-by-step reconciliation is the C23 waterfall in
>   [docs/corrections.md](https://github.com/navnoorthapar/vol-system-dashboard/blob/main/docs/corrections.md#c23).
> - **Intervals and multiple testing.** The portfolio Sharpe's 95% stationary-bootstrap interval is −1.81 to −0.30; no
>   published series beats cash (Hansen SPA p 0.65). Against 3,550 exposure-matched circular shifts, S3's entry timing
>   has p 0.078 (0.47 after a Holm correction over the six series), replacing the 200-path p = 0.075 quoted below.
> - **Model against market.** Scored on quotes it was not fitted to, the joint Heston fit misses by a median 4.4–15.6
>   half-spreads of the bid–ask across the five stored sessions; a certified arbitrage-free SVI surface on the same
>   sessions misses by 0.45–1.98. Freeing Heston's vol-of-vol cap (σ ≤ 2) on 2026-10-16 cuts its out-of-sample error from 2.96
>   to 1.19 vol pts, at σ = 6.9. Rough Heston on the same quotes is lower on 3 of 5 sessions and identical to Heston on
>   the other 2; it does not pin down a roughness exponent.
> - **Withdrawn:** the "14–23× faster" figure for the NN surrogate below, which no committed benchmark backs.
>
> - **The title.** The VIX-options leg is still off, so the project is "joint SPX/VIX calibration" (the SPX smile and
>   the VIX index term structure), not a joint SPX/VIX *smile* fit; the original headline is struck through.
>
> Every number in this note is read from a committed file and checked against it by the repository's tests
> (`test_readme.py`); the [README](https://github.com/navnoorthapar/vol-system-dashboard#readme) names each file.

> **Correction (October 2026, C22).** After this article was published, a full audit of the system (about 270 findings,
> each reproduced before it was fixed) changed several of its central claims. In short:
>
> - **The backtest priced ATM straddles at the VIX.** The VIX is a variance-swap rate that contains the skew premium; real
>   2026 SPX chains put 30-day ATM vol at about 0.82 × VIX. Re-priced at that ratio, with hedge and financing costs, one
>   consistent R2 gate and a horizon-matched PDV forecast, **S1's "+$463K" (later +$694K) becomes −$288K**. Its profit
>   existed only at β ≈ 1. The portfolio (S1C + S3 + S4) is **−21.6%, Sharpe −0.88** (rf = 0 on excess P&L, against
>   T-bills +19.1% over the window), max drawdown −21.6%, NAV $783,908.
> - **S3's "+$23K" was not option P&L**: it was a z-score change times an assumed 2%, positive on 99.5% of random-walk null
>   paths. Priced as the index leg of the dispersion trade it makes +$56K, positive only for β ≳ 0.8, and its entry timing
>   does not beat random dates at the 5% level (null-test p = 0.075). No signal survives.
> - **The calibration finding below was mostly an artifact.** ρ pinned at −0.95 came from ATM-call-only anchors; the
>   "binding Feller constraint" came from a Feller penalty the calibrator imposed on itself; the VIX indices were fitted as
>   if they were futures; and the 37-vol-pt VIX-options RMSE came from a broken pricer. The corrected fits have ρ between
>   −0.66 and −0.82 (interior) and σ between 1.6 and 2.0 (Feller violated, not imposed); the current headline, the
>   2026-10-16 session, fits SPX to 1.34 vp in-sample / 2.96 vp out-of-sample and the VIX indices to 0.82 pts, with σ on
>   its 2.0 ceiling. There is no "three effective degrees of freedom" result.
> - The live daily calibration was **effectively down from 2026-09-05 to October**: it fetched too few expiries, sampled an
>   empty after-hours book, and reported every run as green. That is fixed: the corrected pipeline's first live run accepted
>   the 2026-10-16 session.
> - Smaller fixes: "Carr-Madan FFT" (no FFT was used; the engine is now a Lewis contour integral), "eighteen months" (the
>   project started in March 2026), the classifier figures, the PDV skill comparison and the test count.
>
> Claims that are now false are struck through below, with the corrected statement next to them. The full before/after
> waterfall is in [docs/corrections.md](https://github.com/navnoorthapar/vol-system-dashboard/blob/main/docs/corrections.md#c22).

Every options desk runs two separate calibration routines: one for the SPX smile and one for the VIX term structure. They're run independently, stored in different systems, and never asked to agree. This works until it doesn't — and when it doesn't, you get a book that's simultaneously long skew and short correlation, or a delta hedge that blows up on a VIX spike your SPX model called "small".

The hard problem is joint calibration: find one set of parameters that fits both the SPX smile *and* VIX options *simultaneously*, with the constraint that they're driven by the same instantaneous variance process V(t). That's what this project attempted. *(C22: it is not an open problem in the literature — Guyon (2020) and Abi Jaber, Illand & Li (2022) solve it with richer models; it is hard for classical one-factor diffusions like Heston, which is what this project measures. C23: the fitted VIX leg is the four VIX indices; VIX options are not in the objective.)*

What I want to write about here isn't the calibration. It's the ~~five~~ *(C23: eight rounds, C16–C23, in [docs/corrections.md](https://github.com/navnoorthapar/vol-system-dashboard/blob/main/docs/corrections.md))* times I had to correct my own work — the fourth inverted a thesis I'd spent weeks building on top of, and the fifth I found only after the system had been running unattended for a month.

---

## What we built

~~Over eighteen months~~ Since March 2026, across 14 components, the system covers the full pipeline from raw data to live dashboard:

- **C1–C2**: Data infrastructure + Heston engine (~~Carr-Madan FFT~~ Lewis (2001) contour integral with a Black-Scholes control variate, vectorised batch pricing; no FFT)
- **C3**: PDV model (Guyon-Lekeufack 2023) — path-dependent vol from realised path statistics
- **C4**: Joint calibration engine — one (κ, θ, σ, ρ, v₀) fitting the SPX smile + ~~VIX futures~~ the VIX term-structure indices (spot-starting, not futures)
- **C5**: Neural network acceleration layer ~~(8.1× speedup, MAE = 0.068 vol pts)~~ (research prototype on no production path; C22 retrained it on a domain that covers every fit and published the weights: 0.02 vol pts mean error on held-out parameters, ~~14–23× faster than the Fourier pricer single-threaded~~ (C23: speed-up withdrawn, not backed by a committed benchmark))
- **C6**: Second-order Greeks monitor — vomma, vanna ~~, volga across a 69-node surface~~ and QV convexity on 83 of 90 grid nodes (volga is vomma)
- **C7**: Delta-hedged P&L simulation with full attribution (Γ, ν, Θ, residual)
- **C8**: Volatility regime classifier — XGBoost, 3-way
- **C9**: Regime-gated trading signals (~~IVR~~/PDV, VIX term structure, dispersion, VRP)
- **C10**: Full backtest engine with walk-forward validation and HTML reporting
- **C11–C13b**: Regime-switching jumps (Merton; ~~BNS~~ a Lee-Mykland-style daily jump test), Bates SVJ, ~~SVI/SSVI smoothing~~ (never wired in; removed), two-factor Quintic OU

~~666 unit tests~~ A test suite that CI runs on every push and pull request. Zero look-ahead bias, enforced at the database query level and by mutation-checked look-ahead tests. Live at [navnoorbawa.me](https://navnoorbawa.me).

---

## The calibration finding: ρ at the boundary *(withdrawn in C22 — see the correction note)*

The Heston model calibrated to the 2026-05-31 surface (SPX = 7,580, VIX ≈ 15) returned:

```
κ = 1.77   θ = 0.067   σ = 0.46   ρ = −0.95   v₀ = 0.0127
SPX smile RMSE = 0.52 vol pts   ·   Feller PASS
```

*Correction (C22): that fit used 54 anchors of which 45 were ATM calls, flat r/q, a Sunday-dated snapshot and a VIX leg that
compared spot-starting indices with model futures (at these parameters the correct index model misses by ~2.95 pts, not
0.53). Refitted on the same quotes (session 2026-05-29) with the corrected calibrator: κ = 5.23, θ = 0.0595, σ = 1.71,
ρ = −0.656, v₀ = 0.0130, SPX 0.91 vp in-sample / 1.28 vp out-of-sample, VIX index 0.52 pts, Feller violated (not
imposed), no parameter on a bound.*

~~The SPX leg fits cleanly now — but **ρ is still pinned at its lower boundary** (−0.95, after I'd already tightened it from −0.99). The optimiser still wants more leverage skew than a pure-diffusion model can supply.~~ *C22: ρ was pinned because the anchors had no put wing to fit. With a real put wing ρ is interior (−0.66 to −0.82) on every snapshot.*

In Heston, ρ governs the leverage effect: when the underlying falls, volatility rises, and the correlation between those Brownian motions sets how steep the put skew is. The market's skew geometry wants either discontinuous paths (jumps, à la Bates 1996) or a vol process with long memory (rough vol, Gatheral 2018). ~~ρ at the wall is Heston waving a white flag — not a calibration failure, a diagnostic. And when I turned the **VIX options leg** on, it priced them at a **37-vol-pt RMSE**: a single σ cannot satisfy both the SPX smile curvature and the VVIX-implied vol-of-vol.~~ *C22: the 37 vp came from a broken VIX-options pricer (puts by parity with √E[VIX²], spot-VIX vegas, a stale chain). Like for like on 2026-03-24, Heston re-priced on 61 VIX options gives 35.2 vp against the Quintic OU's 2.77 vp, but Heston was not fitted to them, and on the same SPX targets Heston (1.21 vp) beats the Quintic (1.31 vp), whose fit did not converge.* I disabled the leg (w₃ = 0.0) and documented the reason rather than hide it behind a fudged weight *(it stays off because the database holds VIX option quotes for too few sessions)*.

~~It turns out ρ has company. The Feller condition 2κθ ≥ σ² is enforced by a soft penalty whose gradient dies at the boundary, so when the smile wants more vol-of-vol than Feller allows, the optimiser parks *exactly* on the constraint. Across 19 archived fits, 11 sit at 2κθ − σ² ≈ −1e-6 — σ matching √(2κθ) to six significant figures. My code called that "Feller FAIL." It isn't a failure; it's a binding constraint, and I'd been reporting an active constraint as a broken model. Corrected, the honest statement is sharper than the original one: **two of five parameters are set by boundaries rather than by the market, so Heston fits this joint surface with three effective degrees of freedom.**~~

*C22: the constraint was binding because my calibrator imposed a Feller penalty by default. Feller is not a no-arbitrage condition, and the pricer is validated where it fails. With the penalty off, the fits want σ ≈ 1.6–2.0 and violate Feller by a wide margin (2026-06-18: 2κθ = 0.613 vs σ² = 2.534). The corrected finding is narrower: Heston fits the SPX surface (~1 vp in-sample, 1.3–3.0 vp out-of-sample) and the VIX index term structure (0.5–0.9 pts) together only with a Feller-violating vol-of-vol, misses the real VX futures by 1.9–3.2 pts out of sample, and prices VIX options poorly when asked.*

---

## The fifth correction: my selector was rewarding broken data

I said there were four corrections. There are five *(C23: eight rounds by now)*. This one I found a month after the system went on autopilot, and it is the one I'd least like to have explained in an interview without having caught it first.

The dashboard headlines the best archived calibration, ranked by SPX smile RMSE. Reasonable — except Yahoo intermittently returns `^VIX` without `^VIX9D/3M/6M`, and my term-structure builder took the last row of a pivot without checking it was complete. On those days the VIX leg had **one tenor**.

One tenor is one data point. A five-parameter model hits it exactly — VIX RMSE 0.003 — and all five parameters are then free to chase the SPX smile alone. So the fit gets *better looking* precisely when the data is broken:

| VIX tenors | κ | σ | SPX RMSE |
|---|---|---|---|
| 4 (well-constrained) | ~2.2 | ~0.45 | 0.4 – 2.0 vp |
| 1 (leg unidentified) | 8.9 – 9.9 | ~0.83 | 0.3 – 1.8 vp |

Rank those on RMSE and you don't select the best calibration — **you select the days your data feed failed.** The 2026-07-23 fit (one tenor, SPX RMSE 0.049 vp, roughly ten times "better" than any honest fit) was headlining my live site. The number was advertising that Heston fits the joint surface beautifully, which is the precise opposite of this project's finding. The joint tension *is* the result; I had accidentally published the one day there was no joint tension to have.

Two things made it invisible. The quality gate was working — most 1-tenor days collapsed to a degenerate corner (σ→0.001, ρ→0) and were correctly rejected, which made a two-week data outage look like ordinary noise. And no calibration recorded how much data produced it, so nothing on the artifact could have told me. I reconstructed the tenor counts from CI logs.

The fix is unglamorous: require ≥3 VIX tenors before a fit is written or selected, and persist the input counts with every calibration. *(C22 found the same failure on the SPX side — a fit from a single expiry, 9 options, headlined the site on 2026-09-11 at 0.215 vp — so the gate now also requires ≥ 4 SPX expiries, ≥ 30 options and a put wing, and the dashboard headlines the newest passing fit, never the lowest-RMSE one.)* The lesson is the one this project keeps teaching: **a metric that improves when your data degrades is not a quality metric.** Goodness-of-fit without a degrees-of-freedom check is exactly that metric, and I'd shipped it.

**Postscript, five weeks later: the outage was self-inflicted.** The sub-tenor indices exist on Yahoo only as a current-day row, and yfinance treats its end date as exclusive while my downloader documented it as inclusive. A run on the session date dropped the session that had just closed. Every run that fired before midnight UTC was one session stale and could never see those tenors; the only well-identified August calibrations came from runs GitHub happened to delay past midnight, and six of the fourteen it accepted were one-tenor fits. The failure was legible in my own timestamps the whole time. The fix is one day added to an end date, a term-structure row pinned to the spot session, calibrations named by the session they describe, and a cron moved earlier — plus tests that pin all three. The lesson compounds the last one: when a feed fails "intermittently," check whether the intermittency is your own clock.

*Second postscript (C22): it happened again, louder. Until 2026-09-04 the bot never stored the chain it downloaded, so every "daily" fit had refitted a committed snapshot 5–78 days old. When it switched to a live chain on 2026-09-05 it fetched only the first 8 listed expiries, kept Yahoo's implied vols and fell back to last prices, and fired into Cboe's empty after-hours book; from 2026-09-07 the Monday–Thursday fits were rejected and the few accepted ones were single-expiry fits. Every run was green because the script ignored its own return values. The pipeline now selects expiries by tenor, keeps only two-sided quotes with its own parity-forward IVs, fails closed on an empty book, and fails the job when the newest accepted calibration is stale.*

---

## What broke — the big one: a strike-rolling bug

Here's the correction that matters. My straddle backtest engine **re-entered a position the same day it exited**. Because of that, positions never actually rolled — the engine held *one* stale-strike straddle, marked past its own expiry, for long stretches. Instead of rolling a fresh 30-day ATM straddle each cycle, it was carrying a directional |S − K| bet on an expired contract. **This contaminated every P&L number from C10 onward.**

Fixing it inverted my own thesis. I had built a *contrarian* signal (S1C) on the premise that my base signal S1 was a "−$1.53M catastrophe" — clearly, I reasoned, the right move is to do the opposite. That catastrophe was the bug. Corrected:

- ~~**S1 is a profitable regime-gated short-VRP harvester: +$463K, 76% win, 53 of 54 trades short straddles.** It sells rich implied vol in calm regimes.~~ *C22: S1 −$287,644 (59% win, 69 trades, all short). It looked profitable because straddles were sold at the VIX instead of ATM vol (≈ 0.82 × VIX); at β = 1 it makes +$2.6M, at β = 0.85 −$50K.*
- **S1C — the inversion — is the artifact: ~~−$404K~~ −$431,203 (C22).** ~~Inverting a profitable signal is the wrong trade.~~ *C22: neither S1 nor its inversion makes money; S1C also fails its 2013–2017 pseudo-out-of-sample check (−$261K).*

So S1C is demoted. *A thesis that holds only under a bug is not a signal.* ~~And S1's profit isn't a green light either: it's fat-tailed (one −$333K COVID trade, −$99K in 2022), concentrated (top-5 trades = 73% of the profit), and~~ *(C22: there is no S1 profit to qualify.)* The caveat that dominates everything: **mark-to-model**. The database holds no historical option prices, so all P&L is Black-Scholes ~~on the VIX-ATM vol~~ at 0.82 × VIX, a ratio estimated from five 2026 chains (the first later chain, 2026-10-16, gives 0.70: the ratio is not constant), not traded fills.

The final, honest portfolio (S1C + S3 + S4, equally weighted): ~~**−26.1% cumulative, Sharpe −1.80, max drawdown −31.5%.**~~ **−21.6% cumulative (NAV $783,908), Sharpe −0.88 (rf = 0 on excess P&L; T-bills +19.1% over the window), max drawdown −21.6%, 208 trades** (C22). *C23, on Cboe's VIX data: −22.2% (NAV $777,823), Sharpe −0.99 (95% CI −1.81 to −0.30), max drawdown −22.2%, 206 trades.*

---

## What broke — the classifier loses to "predict yesterday"

I was proud of the regime classifier: 86.2% accuracy, trained 2010–2019, tested 2020–2025. Then I benchmarked it against a no-skill baseline. *(C22: the figures in this section were measured on fixed VVIX > 100 labels with a PDV feature that carried look-ahead. On the single adaptive gate the system now uses, the retrained classifier scores 78.7% in-sample and 75.0% out-of-sample against 86.7% for persistence. The conclusion is unchanged.)*

The regime labels are defined by *same-day observables* — realised vol vs VIX for R0/R1, and VVIX above a threshold for R2. That means "predict yesterday's regime" is a perfectly legitimate causal predictor. It scores **90.0%** on 2020+ *(C22 labels: 86.7%)*. My honest out-of-sample classifier scores **63.4%** *(C22: 75.0%)* — it *loses* to the lag. Part of the original 86.2% was the model recovering its own labelling rule through a circular feature (VVIX, which defines R2). I removed the feature, and the ML model added negative value. *(C22: lagged VVIX is a legitimate predictor, not circular; with it the classifier scores 72.4% out of sample, still below persistence.)*

So the classifier is research-only now. The backtest uses the lagged rule labels directly. This is the kind of result that's embarrassing to find and important to report — and exactly the sort of thing a Sharpe-2 backtest quietly launders.

There's a deeper reason it was never going to work, and it is about the labels, not the model. In any ML project the labels set the noise floor: no model can encode past the information actually in the data, so if the labels are noisy, that noise is the ceiling. So I audited the label *ontology* itself. The R0/R1 boundary is realized vol versus implied vol, and the R2 gate is a VVIX percentile — both hard cutoffs. I re-labelled the entire history under perturbations that sit *inside the data's own measurement noise*: a ±1-day change in the 20-day realized-vol window, and ±0.5 vol points on VIX. **5.9% of the labels flip**, and **10% of days sit within a single vol point of the R0/R1 cut**, where the label is essentially a coin toss. That ~6% flip rate puts the accuracy ceiling near 94%, persistence already reaches ~87–90%, and the classifier lands at 63.4% *(C22: 75.0%)* — nowhere near it, because it is also fighting the fact that these labels are same-day observables that "predict yesterday" already nails. The lesson I keep relearning is that a real ML project is maybe 50% evaluation and 40% data and labels; the training is the easy 2%. Not a day goes by without me thinking about the ontology, and even the old labels have to be reviewed constantly. You cannot train your way under the noise floor; you can only clean the ontology or accept the bound. *(Reproducible: `regime_label_noise_audit()`.)*

---

## What actually survived

**PDV: ~30% of next-day |return| variance, walk-forward — but the "4x over naive" I first claimed was a strawman.** *(C22: much of that pooled 0.31 comes from 2020; ex-2020 it is 0.20. The forecast compared with the VIX is now a 21-day RMS-vol forecast, walk-forward R² 0.29, ex-2020 0.27.)* I originally wrote "R² = 0.31 vs 0.08 naive, a 4x win." That comparison was unfair to the baseline, and a reviewer caught it. The 0.08 is a 20-day realised-vol forecast's *uncalibrated* R²: a 20-day RV runs about 25% hot versus E|r| (RV tracks σ, but E|r| ≈ 0.8σ), so it gets penalised for scale, not for lack of information. PDV is OLS-fit, so it gets that rescaling for free. On a fair, scale-invariant basis (squared correlation, every model scored on the same out-of-sample rows), PDV explains ~~~0.30, a 20-day moving average ~0.20, and a parameter-free RiskMetrics EWMA or a GARCH(1,1) ~0.24~~ 0.31, a 20-day moving average 0.21, a RiskMetrics EWMA 0.21, and a walk-forward GARCH(1,1) 0.25 (also 0.25 if GARCH is fitted in-sample) (C22). So the path-dependent structure (an EWMA approximation of Guyon-Lekeufack, not a replication) does add skill over a trivial vol forecaster — but the honest margin over GARCH is ~0.31 vs ~0.25, a modest edge, not 4x. Volatility is the most forecastable object in markets; most of what any of these models captures is plain vol clustering. (Reproducible: `forecast_skill_comparison()`.)

**S3 dispersion: ~~+$23K, 73% win, 22 trades in seven years~~ +$56K, 49% win, 39 trades in seven years (C22).** The VIX/VVIX ratio as a proxy for implied correlation flagged low-correlation regimes where long dispersion was cheap. ~~No short leg, no complex hedging — a directional bet on a z-score with a 30-day max hold. It is the **only** signal positive at every tested P&L scale and in pseudo-OOS, and it was **unchanged by the strike-rolling fix** because it never re-entered same-day.~~ *C22: the old P&L was a z-score change times an assumed 2% per unit — not option P&L, and positive on 99.5% of random-walk null paths. S3 is now priced as the index leg of the dispersion trade (a short SPX straddle) on the same engine; its timing does not beat random entry dates at the 5% level (p = 0.075 against 200 null paths), it is positive only for β ≳ 0.8, and it earns less than T-bills. C23: +$61K on the corrected VIX data; exposure-matched timing p 0.078, Holm 0.47.* The simplest signal in the system is still the only one with a positive number in the canonical run. It's also too small, too model-dependent and too rare to be a business.

---

## Why the honest negative result is the point

Every hedge fund backtest showing Sharpe > 2 has look-ahead bias somewhere — in the feature construction, the regime labels, the vol surface used for pricing, or the cost model. Usually more than one.

The ~~666~~ tests in this system exist to verify none of those shortcuts were taken: features shift by one day before signal generation, regime labels are computed on the as-of date only, PDV is re-fit walk-forward on strictly pre-year data, and the option engine is held to its own roll schedule. *(C22 showed what tests of the code cannot catch: an assumption — straddles priced at the VIX — that no test questioned.)* The correction history reads: look-ahead → circular feature → label noise → stale-strike bug *→ broken-data selector → test-fixture cash hurdle → VIX-as-ATM-vol*. ~~**Every single fix made the result worse or flipped a thesis.**~~ *(C23: not every one. C19 and C20 left the backtest unchanged, and C22 raised the portfolio from −26.1% to −21.6% while taking S1's profit away; no round restored a positive return.)* That sequence — not any one number — is the deliverable.

The takeaway is not "volatility trading is impossible." It's that the edge is thin, execution-dependent, mark-to-model-sensitive, and dominated by regime risk no model fully captures. ~~ρ at the boundary tells you more about the structural inadequacy of continuous diffusions than any positive backtest would~~ *(C22: ρ at the boundary turned out to be my anchors, not the model)* — and a strike-rolling bug that faked a −$1.53M loss tells you more about backtest hygiene than a clean equity curve ever could.

The code is at [github.com/navnoorthapar/vol-system-dashboard](https://github.com/navnoorthapar/vol-system-dashboard). The live dashboard is at [navnoorbawa.me](https://navnoorbawa.me).

---

*Built in Python. Heston via ~~Carr-Madan FFT~~ the Lewis (2001) contour integral. PDV after Guyon & Lekeufack (2023). Joint calibration via ~~differential evolution + L-BFGS-B polish~~ multi-start trust-region least squares with an analytic Jacobian (C23; every committed fit was made by differential evolution, whose loss it matches to within 3.6e-7 relative). All backtest P&L is mark-to-model (Black-Scholes at 0.82 × VIX). Data: Yahoo Finance, Cboe free historical (including VX futures).*
