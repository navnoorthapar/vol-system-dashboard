# I built the 'holy grail of volatility modelling' — here's what broke

Every options desk runs two separate calibration routines: one for the SPX smile and one for the VIX term structure. They're run independently, stored in different systems, and never asked to agree. This works until it doesn't — and when it doesn't, you get a book that's simultaneously long skew and short correlation, or a delta hedge that blows up on a VIX spike your SPX model called "small".

The open problem is joint calibration: find one set of parameters that fits both the SPX smile *and* VIX options *simultaneously*, with the constraint that they're driven by the same instantaneous variance process V(t). That's what this project attempted.

---

## What we built

Over three months and 10 components, the system covers the full pipeline from raw data to live dashboard:

- **C1–C2**: Data infrastructure + Heston engine (vectorised, 20M+ paths/sec)
- **C3**: PDV model (Guyon-Lekeufack 2023) — path-dependent vol from realised path statistics
- **C4**: Joint calibration engine — one (κ, θ, σ, ρ, v₀) fitting SPX smile + VIX futures + VIX options simultaneously
- **C5**: Neural network acceleration layer (8.1× speedup, MAE = 0.068 vol pts)
- **C6**: Second-order Greeks monitor — vomma, vanna, volga across a 69-node surface
- **C7**: Delta-hedged P&L simulation with full attribution (Γ, ν, Θ, residual)
- **C8**: Volatility regime classifier — XGBoost, 3-way, 86.2% accuracy
- **C9**: Three regime-gated trading signals with a regime-filtered variant
- **C10**: Full backtest engine with walk-forward validation and HTML reporting

374 unit tests. Zero look-ahead bias enforced at the database query level. Live at [navnoorbawa.me](https://navnoorbawa.me).

---

## The headline finding: ρ = −0.99

The Heston model calibrated to the 2026-03-24 surface returned:

```
κ = 4.6   θ = 0.076   σ = 0.84   ρ = −0.99   v₀ = 0.056
```

ρ = −0.99 is the lower bound. The optimiser ran into the wall.

In Heston, ρ governs the leverage effect: when the underlying falls (dS < 0), volatility rises (dV > 0), and the correlation between these Brownian motions determines how steep the put skew is. A steep 2026 surface — SPX at ~6,576 with VIX at 26.6 after a 4.6% YTD drawdown — demands extreme negative correlation to price correctly under the diffusion-only model.

The mathematical interpretation is clean: hitting the boundary means Heston is *misspecified*. The market's skew geometry requires either discontinuous paths (jumps, à la Bates 1996) or a vol process with long memory (rough vol, à la Gatheral 2018) or both. ρ = −0.99 is Heston waving a white flag.

This is not a calibration failure. It's a diagnostic: the surface is telling you something structural about the data-generating process, and Heston doesn't have the vocabulary to say it. The correct response is not to widen the bounds — it's to reach for a richer model.

---

## What broke

**1. VIX options: RMSE = 37.14 vol pts.**

VIX options price via the CIR transition density for V_T. Their smile implies a vol-of-vol an order of magnitude higher than what SPX options will tolerate. A single σ cannot satisfy both. The joint loss function has to choose — and it chose SPX (RMSE = 5.3 vol pts) at the expense of VIX options.

This is a structural failure of single-factor Heston, not an optimisation failure. The fix is a two-factor model (Bergomi 2005) or adding a jump in V. The takeaway: VIX options carry information about the *distribution* of future vol, not just its level, and Heston's affine structure can't encode both simultaneously.

**2. Backtest: −22.19% cumulative, Sharpe −1.527.**

P&L breakdown: S1 (IVR spread) = −$503K, S2 (VIX term structure) = −$99K, S3 (dispersion proxy) = +$21K. Three sources:

- *Regime mismatch.* S1 shorts vol in R1 (IV > RV) expecting mean reversion within 21 days. In 2022 and 2024 drawdown periods, VIX stayed elevated for months. The regime classifier correctly flagged R2 but the position was already open — entry gates block new trades in R2, but existing ones persist.
- *Instrument mismatch.* The signal is built on PDV vs ATM IV spread; the backtest trades a delta-hedged straddle. The edge visible in feature space doesn't survive the round-trip through actual options pricing and transaction costs.
- *Walk-forward reality.* 10 of 12 out-of-sample windows had Sharpe < 0. Only H1 2023 — the post-SVB vol collapse — showed positive carry. A Sharpe that only works during one specific macro regime is not an edge.

**3. Jump identification fails on daily data.**

The Merton (1976) component added to the PDV model for Regime 2 days immediately collapsed to λ ≈ 50/yr with tiny σ_j. The fix — calibrating on the tail of R2 days where PDV most severely underpredicted — helped, but the λ estimate still hit its upper bound.

The reason is structural: daily returns, even on genuine crash days, look approximately Gaussian once you condition on the 20-day realised vol being elevated. MLE cannot distinguish λ = 2/yr, σ_j = 5% from λ = 10/yr, σ_j = 2% — both produce similar variance contributions at the daily frequency. Identifying jumps requires intraday resolution, specifically the Barndorff-Nielsen & Shephard decomposition of realised quadratic variation into continuous and jump components. This is a data constraint, not a model one.

---

## What worked

**PDV: R² = 0.31 vs 0.08 naive.**

The naive model — predict tomorrow's realised vol using the last 20-day RV — explains 8% of variance. PDV explains 31%. That is a 4× improvement using three features (two EWMA vol estimates and a leverage term), no neural network, and a linear model. The Guyon-Lekeufack formulation is that clean.

**S3 dispersion proxy: +$21K, 92.9% win rate.**

The VIX/VVIX ratio as a proxy for implied correlation correctly identified low-correlation regimes where long dispersion was cheap. No short leg, no complex hedging — a directional bet on a z-score with a 30-day max hold. The simplest signal in the system was the only one with positive expectation over the full backtest period.

**Regime classifier: 86.2% accuracy, R2 recall 95.1%.**

Trained on 2010–2019, tested on 2020–2025. Correctly classified 2020-03-16 (COVID crash) and 2025-04-09 (tariff shock) as R2 (VOMMA_ACTIVE) in real time. The 95.1% recall on R2 — the most dangerous regime — is the number that matters operationally, and the one most likely to be sacrificed by a model that optimises for overall accuracy.

---

## Why the honest negative result is the point

Every hedge fund backtest showing Sharpe > 2 has look-ahead bias. It's either in the feature construction, the regime labels, the vol surface used for pricing, or the transaction cost model. Usually all four.

The 374 tests in this system exist to verify none of those shortcuts were taken: features shift by one day before signal generation, regime labels are computed on the as-of date only, option pricing uses the calibration from the prior close, and walk-forward windows are evaluated sequentially on data the model has never seen. When you enforce these constraints, the realistic result on a carefully designed system is Sharpe = −1.527.

The correct takeaway is not "volatility trading is impossible." It's that volatility edge is thin, execution-dependent, and dominated by regime risk that no model fully captures. ρ = −0.99 tells you more about the structural inadequacy of continuous diffusions than any positive backtest would.

The code is at [github.com/navnoorthapar/vol-system-dashboard](https://github.com/navnoorthapar/vol-system-dashboard). The live dashboard is at [navnoorbawa.me](https://navnoorbawa.me).

---

*Built in Python. Heston via Carr-Madan FFT. PDV after Guyon & Lekeufack (2023). Joint calibration via differential evolution + L-BFGS-B polish. Regime classifier: XGBoost with walk-forward validation. Data: CBOE free historical, Yahoo Finance.*
