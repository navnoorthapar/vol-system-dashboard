# Joint SPX/VIX Smile Calibration System

**Joint SPX/VIX volatility calibration, risk, and signal-research system — plus a fully documented negative trading result. Live at https://navnoorbawa.me**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-00ff88?style=flat-square)](https://navnoorbawa.me)
[![Tests](https://img.shields.io/badge/tests-628%20passing-00ff88?style=flat-square)](#testing)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)](https://python.org)

---

## What this is — and what it isn't

This is **calibration and risk infrastructure plus an honest research log**, not a profitable strategy.

The headline result is a **negative backtest, presented without adjustment**. Over eighteen months of work the trading layer was corrected four times — for look-ahead bias, a circular classifier feature, label noise, and finally a strike-rolling bug that had contaminated *every* prior P&L number. Each correction made the result worse or inverted a thesis I had previously believed. **That audit trail is the deliverable.** Curve-fitting a strategy to look profitable in-sample is easy; understanding precisely why a plausible one *doesn't* work is the harder and more useful exercise.

If you want the short version: the calibration engine is solid, the risk tooling is solid, and the one trading signal that survived every correction (S3, dispersion timing) makes +$23K over seven years — real, but too small and too rare to call a business.

---

## The headline numbers (must match the dashboard)

```
Backtest 2018–2025 · $1M initial capital · delta-hedged straddle portfolio
  Portfolio = (S1C + S3 + S4) / 3

  NAV:               $809,972
  Cumulative P&L:    −$190,028   (−19.0%)
  Sharpe (vs ^IRX):  −1.57
  Max drawdown:      −25.9%

By signal:
  S1  IVR short-VRP (reference)   +$462,976   Sharpe  0.257   76% win   54 trades
  S1C Contrarian PDV (DEMOTED)    −$404,183   Sharpe −0.622   45% win   88 trades
  S3  Dispersion (survivor)       + $23,212   73% win   22 trades   ← positive at every scale + pseudo-OOS
  S4  Volatility risk premium     −$188,519   Sharpe −1.164   32% win   31 trades
```

**The single most important caveat:** every P&L number is **mark-to-model** — Black-Scholes on the VIX-term-structure ATM vol, because the database holds *zero* historical option prices (options are a single 2026-03-24 snapshot). The signals are real; the realised dollars are model-implied, not traded fills.

---

## The four corrections (the actual story)

| # | Correction | Effect |
|---|-----------|--------|
| C16 | Walk-forward PDV per year; removed circular `vvix` feature; ex-ante Kelly | Headline +19% was a **label-noise artifact** |
| C17 | Benchmarked the regime classifier against a *persistence baseline* | Classifier 63.4% **loses** to 90.0% "predict yesterday's regime" → demoted from the trading loop |
| C18 | **Strike-rolling bug**: same-day re-entry meant positions never rolled — the engine held one stale-strike straddle marked past expiry | Contaminated **every** backtest C10→C17; fixing it **inverted my own thesis** |
| C19 | S2 stop-loss propagation, dead-code removal, signal-summary coverage; data-gap repair | Cleanup; canonical numbers unchanged |

**The C18 inversion is the centrepiece.** I had built a contrarian signal (S1C) on the premise that my base signal S1 was a "−$1.53M catastrophe." That catastrophe was the bug. Corrected, **S1 is a profitable regime-gated short-VRP harvester (+$463K, 76% win, 53/54 trades short straddles)** — and **S1C, the inversion, is the artifact (−$404K)**. Inverting a profitable signal is the wrong trade, so S1C is demoted, exactly like the ML classifier in C17: *a thesis that holds only under a bug is not a signal.*

S1's profit is **not** a green light either — it is fat-tailed (a single −$333K COVID trade; −$99K in 2022), concentrated (top-5 trades = 73% of profit), and mark-to-model. Honest read: short-VRP earns premium in calm regimes and pays it back in spikes.

---

## The Calibration Finding

The current showcase calibration (2026-05-31 snapshot, SPX = 7,580, VIX ≈ 15, r = 4.5%):

```
Calibrated Heston parameters:
  κ = 1.77    θ = 0.067    σ = 0.46
  ρ = −0.95   v₀ = 0.0127   (√v₀ = 11.25% spot vol)

  SPX smile RMSE:    0.52 vol pts
  VIX futures RMSE:  0.53 pts
  Feller condition:  2κθ = 0.2372 > σ² = 0.2116   PASS
```

The SPX leg fits well and Feller passes — but **ρ is still pinned at its lower boundary (−0.95)**, the dashboard flags this, and the **VIX options leg is deliberately disabled** (`w₃ = 0.0`). ρ at the wall means even after tightening the bound the optimizer wants more leverage skew than Heston's diffusion can supply — the same misspecification, just no longer catastrophic for the SPX fit. When it was active, Heston's CIR variance density produced a **37.14 vol-pt RMSE on VIX options**: a *structural* failure, not a calibration failure. Heston cannot simultaneously match the SPX smile curvature and the VVIX-implied vol-of-vol surface; calibrating one degrades the other. Rather than hide that with a fudged weight, the loss function makes the tradeoff explicit and the leg is switched off with the reason documented. A two-factor **Quintic OU** model (also in the repo) recovers the VIX options leg to 17.5 vol-pts — better, but still short of the <10 target. Model misspecification, made visible.

---

## System Overview

14 components, ~15,300 lines of Python (ex-tests), **628 passing tests**, zero look-ahead bias.

| Component | Description | 
|-----------|-------------|
| **C1** Data Infrastructure | SQLite + Parquet pipeline, Yahoo Finance + CBOE downloaders, T-bill (`^IRX`) rate curve |
| **C2** Heston Engine | Carr-Madan FFT pricer, vectorised batch pricing |
| **C3** PDV Model | Guyon-Lekeufack (2023) path-dependent vol; walk-forward R² = 0.31 |
| **C4** Joint Calibration | DE + L-BFGS-B, joint SPX/VIX loss; SPX RMSE 0.52 vp |
| **C5** NN Acceleration | HestonNet surrogate pricer, 8.1× speedup, MAE < 0.07 vol pts |
| **C6** Greeks Monitor | Vomma/vanna/volga surface, 69-cell grid, unstable-node detection |
| **C7** Delta-Hedge Simulation | P&L attribution: Γ + ν + Θ + residual, hedge-efficiency metric |
| **C8** Regime Classifier | XGBoost 3-regime; honest OOS 63.4% — **loses to persistence baseline, demoted** |
| **C9** Signal Engine | S1/S1C (IVR/PDV), S2 (VIX term structure), S3 (dispersion), S4 (VRP) |
| **C10** Backtest Engine | Full 2018–2025 backtest, walk-forward validation, ex-ante Kelly, HTML reports |
| **C11** Regime-Switching PDV | Merton (1976) jump component on R2 tail days, BNS jump filter |
| **C12** Bates SVJ + HMM | Bates (1996) jump-diffusion CF; Gaussian-HMM regime alternative |
| **C13** Improvements | SVI/SSVI smoothing, vega anchors, adaptive VVIX, isotonic calibration, portfolio Kelly |
| **C13b** Two-factor Quintic OU | Recovers VIX options leg to 17.5 vp; VIX futures by-construction |

---

## Live Dashboard

**[https://navnoorbawa.me](https://navnoorbawa.me)** — four pages, served as frozen HTML from stored artifacts:

- **Live Market** — SPX/VIX/VVIX, end-of-day regime (R0/R1/R2), PDV forecast vs implied, VIX term structure (data current to 2026-06-18)
- **Calibration** — Heston parameter surface, Feller condition, SPX smile overlay, RMSE decomposition, Quintic OU comparison
- **Greeks Monitor** — Vomma heatmap, unstable nodes, vanna/QV convexity by maturity
- **Backtest** — Equity curve 2018–2025, per-signal attribution, the full C16→C18 correction log, and a "weaknesses found vs fixed" table

---

## The Regime Classifier — why it's demoted

The classifier scores **86.2% in-sample** but only **63.4% out-of-sample** — and that 63.4% **loses by 27 points** to a no-skill baseline. The regime labels are defined by *same-day observables* (`rv_20d` vs VIX for R0/R1; `VVIX > threshold` for R2), so "predict yesterday's regime" is a legitimate causal predictor that scores **90.0%** on 2020+. The ML model adds *negative* value. It is therefore **research-only**; the backtest uses the lagged rule labels directly, never the classifier's forward prediction. This is the kind of result that's embarrassing to find and important to report.

---

## Data Sources

All free, no Bloomberg or paid feed.

| Dataset | Source | Coverage | Rows |
|---------|--------|----------|------|
| SPX OHLCV | Yahoo Finance (`^GSPC`) | 2010 → 2026-06-18 | 4,139 |
| VIX term structure | Yahoo (`^VIX9D`/`^VIX`/`^VIX3M`/`^VIX6M`/`^VVIX`) | 2010 → 2026-06-18 | ~20,135 |
| SPX options | CBOE (single snapshot) | 2026-03-24 | 15,362 |
| VIX options | CBOE (single snapshot) | 2026-03-24 | 591 |
| T-bill rate | Yahoo (`^IRX`) | 2010 → 2026 | daily |
| VIX futures | CBOE CDN — **403 blocked by Cloudflare** | — | 0 (proxied via `^VIX3M`/`^VIX6M`) |

**Single-snapshot options** is a real limitation: a production system needs daily CBOE options data. It is also *why* the backtest is mark-to-model.

---

## Running Locally

```bash
git clone https://github.com/navnoorthapar/vol-system-dashboard
cd vol-system-dashboard
pip install -r requirements.txt
python dashboard/app.py        # auto-selects a free port (5000→5001→5050→8080)
```

The repository ships pre-computed `data_store/` artifacts (calibration pickles, Greeks surface, regime labels, backtest parquet), so the dashboard runs with no re-calibration.

### Testing

```bash
pytest joint_vol_calibration/tests/ -q
# 628 passed
# Includes explicit look-ahead bias checks (test_lookahead.py)
```

### Requirements

```
Python 3.11+ · flask · pandas · numpy · pyarrow
scikit-learn · scipy · xgboost
```

No QuantLib. No Bloomberg. No paid data.

---

## Known Limitations

- **All backtest P&L is mark-to-model.** Black-Scholes on VIX-ATM vol; the DB holds no historical option prices. This is the dominant caveat.
- **Heston is misspecified for the VIX surface.** VIX options leg disabled (`w₃ = 0.0`); the 37.14-vp RMSE is structural. Quintic OU improves but does not solve it.
- **The trading layer is not deployable.** The only robust signal (S3) makes +$23K over 7 years and fires 22 times; the profitable-looking signal (S1) is fat-tailed, concentrated, and mark-to-model.
- **Regime classifier loses to a persistence baseline** and is research-only.
- **Single-snapshot options data** (2026-03-24). Production needs a daily feed.
- **VIX futures proxied** via `^VIX3M`/`^VIX6M` (CBOE CDN Cloudflare-blocked).

---

## Author

**Navnoor Bawa** — 3rd year CS, Thapar Institute of Engineering & Technology

- Website: [navnoorbawa.me](https://navnoorbawa.me)
- Substack: [navnoorbawa.substack.com](https://navnoorbawa.substack.com)
- LinkedIn: [linkedin.com/in/navnoorbawa](https://www.linkedin.com/in/navnoorbawa/)
- GitHub: [github.com/navnoorthapar](https://github.com/navnoorthapar)

---

*The negative backtest is presented without adjustment. Every correction — look-ahead → circular feature → label noise → stale-strike bug — made the result worse or flipped a thesis. The audit trail is the point.*
