# I built the 'holy grail of volatility modelling' — then found the bug that faked my P&L

Every options desk runs two separate calibration routines: one for the SPX smile and one for the VIX term structure. They're run independently, stored in different systems, and never asked to agree. This works until it doesn't — and when it doesn't, you get a book that's simultaneously long skew and short correlation, or a delta hedge that blows up on a VIX spike your SPX model called "small".

The open problem is joint calibration: find one set of parameters that fits both the SPX smile *and* VIX options *simultaneously*, with the constraint that they're driven by the same instantaneous variance process V(t). That's what this project attempted.

What I want to write about here isn't the calibration. It's the five times I had to correct my own work — the fourth inverted a thesis I'd spent weeks building on top of, and the fifth I found only after the system had been running unattended for a month.

---

## What we built

Over eighteen months and 14 components, the system covers the full pipeline from raw data to live dashboard:

- **C1–C2**: Data infrastructure + Heston engine (Carr-Madan FFT, vectorised batch pricing)
- **C3**: PDV model (Guyon-Lekeufack 2023) — path-dependent vol from realised path statistics
- **C4**: Joint calibration engine — one (κ, θ, σ, ρ, v₀) fitting the SPX smile + VIX futures
- **C5**: Neural network acceleration layer (8.1× speedup, MAE = 0.068 vol pts)
- **C6**: Second-order Greeks monitor — vomma, vanna, volga across a 69-node surface
- **C7**: Delta-hedged P&L simulation with full attribution (Γ, ν, Θ, residual)
- **C8**: Volatility regime classifier — XGBoost, 3-way
- **C9**: Regime-gated trading signals (IVR/PDV, VIX term structure, dispersion, VRP)
- **C10**: Full backtest engine with walk-forward validation and HTML reporting
- **C11–C13b**: Regime-switching jumps (Merton/BNS), Bates SVJ, SVI/SSVI smoothing, two-factor Quintic OU

657 unit tests. Zero look-ahead bias enforced at the database query level. Live at [navnoorbawa.me](https://navnoorbawa.me).

---

## The calibration finding: ρ at the boundary

The Heston model calibrated to the 2026-05-31 surface (SPX = 7,580, VIX ≈ 15) returned:

```
κ = 1.77   θ = 0.067   σ = 0.46   ρ = −0.95   v₀ = 0.0127
SPX smile RMSE = 0.52 vol pts   ·   Feller PASS
```

The SPX leg fits cleanly now — but **ρ is still pinned at its lower boundary** (−0.95, after I'd already tightened it from −0.99). The optimiser still wants more leverage skew than a pure-diffusion model can supply.

In Heston, ρ governs the leverage effect: when the underlying falls, volatility rises, and the correlation between those Brownian motions sets how steep the put skew is. The market's skew geometry wants either discontinuous paths (jumps, à la Bates 1996) or a vol process with long memory (rough vol, Gatheral 2018). ρ at the wall is Heston waving a white flag — not a calibration failure, a diagnostic. And when I turned the **VIX options leg** on, it priced them at a **37-vol-pt RMSE**: a single σ cannot satisfy both the SPX smile curvature and the VVIX-implied vol-of-vol. I disabled the leg (w₃ = 0.0) and documented the reason rather than hide it behind a fudged weight.

It turns out ρ has company. The Feller condition 2κθ ≥ σ² is enforced by a soft penalty whose gradient dies at the boundary, so when the smile wants more vol-of-vol than Feller allows, the optimiser parks *exactly* on the constraint. Across 19 archived fits, 11 sit at 2κθ − σ² ≈ −1e-6 — σ matching √(2κθ) to six significant figures. My code called that "Feller FAIL." It isn't a failure; it's a binding constraint, and I'd been reporting an active constraint as a broken model. Corrected, the honest statement is sharper than the original one: **two of five parameters are set by boundaries rather than by the market, so Heston fits this joint surface with three effective degrees of freedom.**

---

## The fifth correction: my selector was rewarding broken data

I said there were four corrections. There are five. This one I found a month after the system went on autopilot, and it is the one I'd least like to have explained in an interview without having caught it first.

The dashboard headlines the best archived calibration, ranked by SPX smile RMSE. Reasonable — except Yahoo intermittently returns `^VIX` without `^VIX9D/3M/6M`, and my term-structure builder took the last row of a pivot without checking it was complete. On those days the VIX leg had **one tenor**.

One tenor is one data point. A five-parameter model hits it exactly — VIX RMSE 0.003 — and all five parameters are then free to chase the SPX smile alone. So the fit gets *better looking* precisely when the data is broken:

| VIX tenors | κ | σ | SPX RMSE |
|---|---|---|---|
| 4 (well-constrained) | ~2.2 | ~0.45 | 0.4 – 2.0 vp |
| 1 (leg unidentified) | 8.9 – 9.9 | ~0.83 | 0.3 – 1.8 vp |

Rank those on RMSE and you don't select the best calibration — **you select the days your data feed failed.** The 2026-07-23 fit (one tenor, SPX RMSE 0.049 vp, roughly ten times "better" than any honest fit) was headlining my live site. The number was advertising that Heston fits the joint surface beautifully, which is the precise opposite of this project's finding. The joint tension *is* the result; I had accidentally published the one day there was no joint tension to have.

Two things made it invisible. The quality gate was working — most 1-tenor days collapsed to a degenerate corner (σ→0.001, ρ→0) and were correctly rejected, which made a two-week data outage look like ordinary noise. And no calibration recorded how much data produced it, so nothing on the artifact could have told me. I reconstructed the tenor counts from CI logs.

The fix is unglamorous: require ≥3 VIX tenors before a fit is written or selected, and persist the input counts with every calibration. The lesson is the one this project keeps teaching: **a metric that improves when your data degrades is not a quality metric.** Goodness-of-fit without a degrees-of-freedom check is exactly that metric, and I'd shipped it.

**Postscript, five weeks later: the outage was self-inflicted.** The sub-tenor indices exist on Yahoo only as a current-day row, and yfinance treats its end date as exclusive while my downloader documented it as inclusive. A run on the session date dropped the session that had just closed. Every run that fired before midnight UTC was one session stale and could never see those tenors; the only well-identified August calibrations came from runs GitHub happened to delay past midnight, and six of the fourteen it accepted were one-tenor fits. The failure was legible in my own timestamps the whole time. The fix is one day added to an end date, a term-structure row pinned to the spot session, calibrations named by the session they describe, and a cron moved earlier — plus tests that pin all three. The lesson compounds the last one: when a feed fails "intermittently," check whether the intermittency is your own clock.

---

## What broke — the big one: a strike-rolling bug

Here's the correction that matters. My straddle backtest engine **re-entered a position the same day it exited**. Because of that, positions never actually rolled — the engine held *one* stale-strike straddle, marked past its own expiry, for long stretches. Instead of rolling a fresh 30-day ATM straddle each cycle, it was carrying a directional |S − K| bet on an expired contract. **This contaminated every P&L number from C10 onward.**

Fixing it inverted my own thesis. I had built a *contrarian* signal (S1C) on the premise that my base signal S1 was a "−$1.53M catastrophe" — clearly, I reasoned, the right move is to do the opposite. That catastrophe was the bug. Corrected:

- **S1 is a profitable regime-gated short-VRP harvester: +$463K, 76% win, 53 of 54 trades short straddles.** It sells rich implied vol in calm regimes.
- **S1C — the inversion — is the artifact: −$404K.** Inverting a profitable signal is the wrong trade.

So S1C is demoted. *A thesis that holds only under a bug is not a signal.* And S1's profit isn't a green light either: it's fat-tailed (one −$333K COVID trade, −$99K in 2022), concentrated (top-5 trades = 73% of the profit), and — the caveat that dominates everything — **mark-to-model**. The database holds no historical option prices, so all P&L is Black-Scholes on the VIX-ATM vol, not traded fills.

The final, honest portfolio (S1C + S3 + S4, equally weighted): **−19.0% cumulative, Sharpe −1.57, max drawdown −25.9%.**

---

## What broke — the classifier loses to "predict yesterday"

I was proud of the regime classifier: 86.2% accuracy, trained 2010–2019, tested 2020–2025. Then I benchmarked it against a no-skill baseline.

The regime labels are defined by *same-day observables* — realised vol vs VIX for R0/R1, and VVIX above a threshold for R2. That means "predict yesterday's regime" is a perfectly legitimate causal predictor. It scores **90.0%** on 2020+. My honest out-of-sample classifier scores **63.4%** — it *loses by 27 points* to the lag. Part of the original 86.2% was the model recovering its own labelling rule through a circular feature (VVIX, which defines R2). I removed the feature, and the ML model added negative value.

So the classifier is research-only now. The backtest uses the lagged rule labels directly. This is the kind of result that's embarrassing to find and important to report — and exactly the sort of thing a Sharpe-2 backtest quietly launders.

There's a deeper reason it was never going to work, and it is about the labels, not the model. In any ML project the labels set the noise floor: no model can encode past the information actually in the data, so if the labels are noisy, that noise is the ceiling. So I audited the label *ontology* itself. The R0/R1 boundary is realized vol versus implied vol, and the R2 gate is a VVIX percentile — both hard cutoffs. I re-labelled the entire history under perturbations that sit *inside the data's own measurement noise*: a ±1-day change in the 20-day realized-vol window, and ±0.5 vol points on VIX. **5.8% of the labels flip**, and **10% of days sit within a single vol point of the R0/R1 cut**, where the label is essentially a coin toss. That ~6% is a hard ceiling set by the ontology, and the classifier lands at 63.4% — nowhere near it, because it is also fighting the fact that these labels are same-day observables that "predict yesterday" already nails. The lesson I keep relearning is that a real ML project is maybe 50% evaluation and 40% data and labels; the training is the easy 2%. Not a day goes by without me thinking about the ontology, and even the old labels have to be reviewed constantly. You cannot train your way under the noise floor; you can only clean the ontology or accept the bound. *(Reproducible: `regime_label_noise_audit()`.)*

---

## What actually survived

**PDV: ~30% of next-day |return| variance, walk-forward — but the "4x over naive" I first claimed was a strawman.** I originally wrote "R² = 0.31 vs 0.08 naive, a 4x win." That comparison was unfair to the baseline, and a reviewer caught it. The 0.08 is a 20-day realised-vol forecast's *uncalibrated* R²: a 20-day RV runs about 25% hot versus E|r| (RV tracks σ, but E|r| ≈ 0.8σ), so it gets penalised for scale, not for lack of information. PDV is OLS-fit, so it gets that rescaling for free. On a fair, scale-invariant basis (squared correlation), PDV explains ~0.30, a 20-day moving average ~0.20, and a parameter-free RiskMetrics EWMA or a GARCH(1,1) ~0.24. So the path-dependent Guyon-Lekeufack structure does add genuine skill over a trivial vol forecaster — but the honest margin is ~0.30 vs ~0.24, a modest edge, not 4x. Volatility is the most forecastable object in markets; most of what any of these models captures is plain vol clustering. (Reproducible: `forecast_skill_comparison()`.)

**S3 dispersion: +$23K, 73% win, 22 trades in seven years.** The VIX/VVIX ratio as a proxy for implied correlation flagged low-correlation regimes where long dispersion was cheap. No short leg, no complex hedging — a directional bet on a z-score with a 30-day max hold. It is the **only** signal positive at every tested P&L scale and in pseudo-OOS, and it was **unchanged by the strike-rolling fix** because it never re-entered same-day. The simplest signal in the system was the only one with a positive expectation that survived every correction. It's also too small and too rare to be a business.

---

## Why the honest negative result is the point

Every hedge fund backtest showing Sharpe > 2 has look-ahead bias somewhere — in the feature construction, the regime labels, the vol surface used for pricing, or the cost model. Usually more than one.

The 657 tests in this system exist to verify none of those shortcuts were taken: features shift by one day before signal generation, regime labels are computed on the as-of date only, PDV is re-fit walk-forward on strictly pre-year data, and the option engine is held to its own roll schedule. The correction history reads: look-ahead → circular feature → label noise → stale-strike bug. **Every single fix made the result worse or flipped a thesis.** That sequence — not any one number — is the deliverable.

The takeaway is not "volatility trading is impossible." It's that the edge is thin, execution-dependent, mark-to-model-sensitive, and dominated by regime risk no model fully captures. ρ at the boundary tells you more about the structural inadequacy of continuous diffusions than any positive backtest would — and a strike-rolling bug that faked a −$1.53M loss tells you more about backtest hygiene than a clean equity curve ever could.

The code is at [github.com/navnoorthapar/vol-system-dashboard](https://github.com/navnoorthapar/vol-system-dashboard). The live dashboard is at [navnoorbawa.me](https://navnoorbawa.me).

---

*Built in Python. Heston via Carr-Madan FFT. PDV after Guyon & Lekeufack (2023). Joint calibration via differential evolution + L-BFGS-B polish. All backtest P&L is mark-to-model (Black-Scholes on VIX-ATM vol). Data: Yahoo Finance, CBOE free historical.*
