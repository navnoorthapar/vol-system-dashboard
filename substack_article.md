# I built the 'holy grail of volatility modelling' — then found the bug that faked my P&L

Every options desk runs two separate calibration routines: one for the SPX smile and one for the VIX term structure. They're run independently, stored in different systems, and never asked to agree. This works until it doesn't — and when it doesn't, you get a book that's simultaneously long skew and short correlation, or a delta hedge that blows up on a VIX spike your SPX model called "small".

The open problem is joint calibration: find one set of parameters that fits both the SPX smile *and* VIX options *simultaneously*, with the constraint that they're driven by the same instantaneous variance process V(t). That's what this project attempted.

What I want to write about here isn't the calibration. It's the four times I had to correct my own backtest — and the last correction, which inverted a thesis I'd spent weeks building on top of.

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

585 unit tests. Zero look-ahead bias enforced at the database query level. Live at [navnoorbawa.me](https://navnoorbawa.me).

---

## The calibration finding: ρ at the boundary

The Heston model calibrated to the 2026-05-31 surface (SPX = 7,580, VIX ≈ 15) returned:

```
κ = 1.77   θ = 0.067   σ = 0.46   ρ = −0.95   v₀ = 0.0127
SPX smile RMSE = 0.52 vol pts   ·   Feller PASS
```

The SPX leg fits cleanly now — but **ρ is still pinned at its lower boundary** (−0.95, after I'd already tightened it from −0.99). The optimiser still wants more leverage skew than a pure-diffusion model can supply.

In Heston, ρ governs the leverage effect: when the underlying falls, volatility rises, and the correlation between those Brownian motions sets how steep the put skew is. The market's skew geometry wants either discontinuous paths (jumps, à la Bates 1996) or a vol process with long memory (rough vol, Gatheral 2018). ρ at the wall is Heston waving a white flag — not a calibration failure, a diagnostic. And when I turned the **VIX options leg** on, it priced them at a **37-vol-pt RMSE**: a single σ cannot satisfy both the SPX smile curvature and the VVIX-implied vol-of-vol. I disabled the leg (w₃ = 0.0) and documented the reason rather than hide it behind a fudged weight.

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

---

## What actually survived

**PDV: R² = 0.31 vs 0.08 naive.** The naive model — predict tomorrow's realised vol from the last 20-day RV — explains 8% of variance. PDV explains 31%, a 4× improvement using three features (two EWMA vol estimates and a leverage term), a linear model, no neural network. The Guyon-Lekeufack formulation is that clean.

**S3 dispersion: +$23K, 73% win, 22 trades in seven years.** The VIX/VVIX ratio as a proxy for implied correlation flagged low-correlation regimes where long dispersion was cheap. No short leg, no complex hedging — a directional bet on a z-score with a 30-day max hold. It is the **only** signal positive at every tested P&L scale and in pseudo-OOS, and it was **unchanged by the strike-rolling fix** because it never re-entered same-day. The simplest signal in the system was the only one with a positive expectation that survived every correction. It's also too small and too rare to be a business.

---

## Why the honest negative result is the point

Every hedge fund backtest showing Sharpe > 2 has look-ahead bias somewhere — in the feature construction, the regime labels, the vol surface used for pricing, or the cost model. Usually more than one.

The 585 tests in this system exist to verify none of those shortcuts were taken: features shift by one day before signal generation, regime labels are computed on the as-of date only, PDV is re-fit walk-forward on strictly pre-year data, and the option engine is held to its own roll schedule. The correction history reads: look-ahead → circular feature → label noise → stale-strike bug. **Every single fix made the result worse or flipped a thesis.** That sequence — not any one number — is the deliverable.

The takeaway is not "volatility trading is impossible." It's that the edge is thin, execution-dependent, mark-to-model-sensitive, and dominated by regime risk no model fully captures. ρ at the boundary tells you more about the structural inadequacy of continuous diffusions than any positive backtest would — and a strike-rolling bug that faked a −$1.53M loss tells you more about backtest hygiene than a clean equity curve ever could.

The code is at [github.com/navnoorthapar/vol-system-dashboard](https://github.com/navnoorthapar/vol-system-dashboard). The live dashboard is at [navnoorbawa.me](https://navnoorbawa.me).

---

*Built in Python. Heston via Carr-Madan FFT. PDV after Guyon & Lekeufack (2023). Joint calibration via differential evolution + L-BFGS-B polish. All backtest P&L is mark-to-model (Black-Scholes on VIX-ATM vol). Data: Yahoo Finance, CBOE free historical.*
