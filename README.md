# Joint SPX/VIX Smile Calibration System

**Joint SPX/VIX smile calibration system — the "holy grail of volatility modelling." Live at https://navnoorbawa.me**

[![Live Dashboard](https://img.shields.io/badge/dashboard-live-00ff88?style=flat-square)](https://navnoorbawa.me)
[![Tests](https://img.shields.io/badge/tests-374%20passing-00ff88?style=flat-square)](#testing)
[![Python](https://img.shields.io/badge/python-3.11-blue?style=flat-square)](https://python.org)

---

## The Core Finding

On 2026-03-24 (SPX = 6,581, VIX = 26.62, VVIX = 122.82):

```
Calibrated Heston parameters:
  κ = 4.6213    θ = 0.07605    σ = 0.8400
  ρ = -0.9900   v₀ = 0.05607

  SPX smile RMSE:     3.83 vol pts
  VIX futures RMSE:   1.15 pts
  VIX options RMSE:  37.14 pts   ← structural failure
  Feller condition:  2κθ = 0.70622 < σ² = 0.70677   FAIL (by −0.00055)
```

**ρ hit the lower boundary (−0.99) during calibration.** The optimizer ran into the wall and stopped there.

This is not a numerical accident. The 2026 SPX left skew is too steep for Heston's pure-diffusion framework to fit. Heston's correlation parameter ρ encodes leverage effect via Brownian covariance — but it cannot produce the sharp, asymmetric left wing that post-COVID equity skew requires. The model needs jump risk (Bates, SVJ) or a richer variance kernel to match observed market prices.

**The VIX options RMSE of 37.14 pts is not a bug.** It reflects a genuine structural mismatch: Heston's CIR variance process cannot simultaneously match the SPX smile curvature and the VVIX-implied vol-of-vol surface. Calibrating one degrades the other. The joint calibration loss function makes this tradeoff explicit.

This is model misspecification made visible — which is more useful than a model that hides it.

---

## System Overview

10 components, 15,863 lines of Python, 374 passing tests, zero look-ahead bias.

| Component | Description | Tests |
|-----------|-------------|-------|
| **C1** Data Infrastructure | SQLite + Parquet pipeline, CBOE + Yahoo Finance downloaders | 18 |
| **C2** Heston Engine | Carr-Madan FFT pricer, vectorised batch pricing (160× speedup) | 32 |
| **C3** PDV Model | Guyon-Lekeufack (2023) path-dependent vol, GARCH(1,1) baseline | — |
| **C4** Joint Calibration | DE + L-BFGS-B, joint SPX/VIX loss, 42.6s on live data | 33 |
| **C5** NN Acceleration | HestonNet surrogate pricer, 8.1× speedup, MAE < 0.07 vol pts | 44 |
| **C6** Greeks Monitor | Vomma/vanna/volga surface, 69-cell grid, unstable-node detection | 42 |
| **C7** Delta-Hedge Simulation | P&L attribution: Γ + ν + Θ + residual, hedge efficiency metric | 54 |
| **C8** Regime Classifier | XGBoost 3-regime classifier, 86.2% accuracy, 95.1% R2 recall | 53 |
| **C9** Signal Engine | 3 trading signals (IVR/PDV, VIX term structure, dispersion) | 49 |
| **C10** Backtest Engine | Full 2018–2025 backtest, walk-forward validation, HTML reports | 49 |
| **Total** | | **374** |

---

## Live Dashboard

**[https://navnoorbawa.me](https://navnoorbawa.me)**

Four pages, updated from stored calibration and live market data:

- **Live Market** — SPX/VIX/VVIX, regime classification (R0/R1/R2), PDV forecast vs implied, VIX term structure
- **Calibration** — Heston parameter surface, Feller condition, SPX smile overlay, ρ boundary warning
- **Greeks Monitor** — Vomma heatmap, unstable nodes, vanna/QV convexity by maturity
- **Backtest** — Full equity curve 2018–2025, per-signal attribution, walk-forward Sharpe

> *Screenshot placeholder — visit [navnoorbawa.me](https://navnoorbawa.me) for live view.*

---

## Key Results

### PDV Model (C3) — Guyon-Lekeufack Path-Dependent Volatility

```
Walk-forward R²:   0.31 (linear OLS)
                   0.23 (Nadaraya-Watson kernel)
Naive baseline R²: ~0.08

Fitted equation:   σ̂ = 0.354·σ₁ + 0.241·σ₂ − 1.496·lev + 3.46%
  where σ₁ = 5-day EWMA vol, σ₂ = 60-day EWMA vol, lev = 10-day EMA returns

COVID stress test (2020-03-16):
  PDV forecast:  92.4% annualised vol
  Actual:       202.6% annualised vol
  Error:        −110pp  (PDV cannot price tail-driven vol jumps)
```

R² = 0.31 is 4× the naive baseline, but the COVID error illustrates the hard limit of any path-dependent model without jump terms.

### Joint Calibration (C4)

```
As of:             2026-03-24
SPX spot:          ~6,576
VIX (30d implied): 26.62
VVIX:              122.82

SPX smile RMSE:    3.83 vol pts   (48 options across 6 expiries)
VIX futures RMSE:  1.15 pts       (term structure proxy via VIX9D/3M/6M)
VIX options RMSE: 37.14 pts       (14 options across 2 expiries — structural failure)
Calibration time:  42.6 seconds   (DE: 120 iter × 40 members + L-BFGS-B polish)
```

### Regime Classifier (C8)

```
Train: 2010–2019 (2,493 samples)    Test: 2020–2025 (1,500 samples)

Test accuracy:   86.2%
R0 (LONG_GAMMA)  F1 = 0.696
R1 (SHORT_GAMMA) F1 = 0.839
R2 (VOMMA_ACTIVE) F1 = 0.905   recall = 95.1%

Validation:
  2020-03-16 (COVID crash, VVIX=207):   → R2  ✓
  2025-04-09 (tariff spike, VVIX=142):  → R2  ✓
  Current (2026-03-24, VVIX=122.82):    → R2 at 99.4% confidence
```

### Backtest (C10) — 2018 to 2025, $1M initial capital

```
Cumulative return:  -22.19%
Annualised return:   -3.45%
Sharpe (rf=5%):      -1.527
Max drawdown:        -26.33%
N trades:             129
N days:             1,799

By signal:
  S1 IVR/PDV:        -$502,671   Sharpe -0.652   win rate 48.6%
  S2 VIX term str:    -$98,977   Sharpe -1.143   win rate 36.7%
  S3 Dispersion:      +$21,003   Sharpe -13.59   win rate 92.9%   ← only profitable signal
  Combined:          -$306,953   Sharpe -1.720

Walk-forward: 11/13 six-month windows have Sharpe < 0
Crisis performance: COVID 2020 +0.1% (Sharpe 3.18)
```

**The negative backtest is a feature, not a bug.** The backtest is honest. Three findings:

1. S1 and S2 went short gamma during elevated-vol regimes (R2 days = 52.9% of the sample) where short-gamma strategies systematically lose. The regime classifier was built after the signals — in live use, C8 would filter these entries.

2. S3 (dispersion, long-only, 14 trades in 7 years) is profitable with 92.9% win rate precisely because it fires rarely and only at extreme signal values. High selectivity beats raw frequency.

3. Transaction costs consumed ~$334k over 7 years on options strategies. The signal alpha is marginal relative to implementation costs — a known structural problem in vol trading.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Joint SPX/VIX Vol System                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │         C1: Data Infrastructure      │
          │  SQLite (vol_system.db) + Parquet    │
          │  CBOE options · Yahoo OHLCV · VIX TS │
          └──┬───────────────┬───────────────────┘
             │               │
    ┌─────────▼────┐   ┌─────▼──────────────────────────────┐
    │  C2: Heston  │   │  C3: PDV Model                     │
    │  FFT pricer  │   │  Path-dependent vol (Guyon 2023)   │
    │  batch + NN  │   │  GARCH(1,1) + OLS + kernel         │
    └─────────┬────┘   └─────────────────┬──────────────────┘
              │                          │
    ┌─────────▼──────────────────────────▼──────────────────┐
    │                 C4: Joint Calibration                  │
    │   loss = w₁·MSE(SPX IV) + w₂·MSE(VIX fut)            │
    │          + w₃·MSE(VIX options)                        │
    │   DE (120×40) → L-BFGS-B polish → 42.6s              │
    └─────────┬──────────────────────────────────────────────┘
              │
    ┌─────────▼────┐   ┌──────────────────────────────────┐
    │  C5: NN      │   │  C6: Greeks Monitor              │
    │  HestonNet   │   │  Vomma/vanna/volga surface       │
    │  8.1× speed  │   │  69 cells · 3 unstable nodes     │
    └──────────────┘   └──────────────────────────────────┘
              │
    ┌─────────▼────────────────────────────────────────────┐
    │                   C7: Delta-Hedger                   │
    │   P&L = Γ-PnL + ν-PnL + Θ-PnL + residual           │
    │   Hedge efficiency A=0.0164 · B=0.3034 (PDV)        │
    └─────────┬────────────────────────────────────────────┘
              │
    ┌─────────▼────┐   ┌──────────────────────────────────┐
    │  C8: Regime  │   │  C9: Signal Engine               │
    │  XGBoost     │   │  S1: IVR/PDV spread              │
    │  86.2% acc   │   │  S2: VIX term structure          │
    │  3 regimes   │   │  S3: VIX/VVIX dispersion         │
    └─────────┬────┘   └──────────────┬───────────────────┘
              │                       │
    ┌─────────▼───────────────────────▼───────────────────┐
    │               C10: Backtest Engine                  │
    │   2018–2025 · $1M · 129 trades · HTML reports      │
    └─────────┬───────────────────────────────────────────┘
              │
    ┌─────────▼───────────────────────────────────────────┐
    │             Flask Dashboard (navnoorbawa.me)        │
    │   4 pages · dark theme · Chart.js · read-only      │
    └─────────────────────────────────────────────────────┘
```

---

## Data Sources

All free, no Bloomberg or paid data.

| Dataset | Source | Coverage | Rows |
|---------|--------|----------|------|
| SPX OHLCV | Yahoo Finance (`^GSPC`) | 2010–2026 | 4,078 |
| VIX daily | CBOE CDN (`VIX_History.csv`) | 2010–2026 | 4,051 |
| VIX term structure | Yahoo Finance (`^VIX9D`, `^VIX`, `^VIX3M`, `^VIX6M`, `^VVIX`) | 2010–2026 | 20,130 |
| SPX options | CBOE (snapshot) | 2026-03-24 | 15,362 |
| VIX options | CBOE (snapshot) | 2026-03-24 | 591 |
| VIX futures | CBOE CDN — **403 blocked by Cloudflare** | — | 0 |

VIX futures are proxied via the `^VIX3M`/`^VIX6M` term structure. The CBOE CDN block is noted in the calibration output.

---

## Running Locally

```bash
git clone https://github.com/NavnoorBawa/vol-system-dashboard
cd vol-system-dashboard
pip install -r requirements.txt
python dashboard/app.py
# opens http://localhost:5001 automatically
```

The repository includes pre-computed `data_store/` artifacts (calibration pickle, Greeks surface, regime labels, backtest parquet). No re-calibration needed to run the dashboard.

**To re-run the full pipeline** (requires ~30 min for calibration + NN training):

```bash
# C1: Download data
python -m joint_vol_calibration.data.pipeline

# C3: Fit PDV model
python -m joint_vol_calibration.models.pdv

# C4: Joint calibration (42.6s on 2026-03-24 data)
python -m joint_vol_calibration.calibration.joint_calibrator

# C5: Train NN surrogate (23 min CPU)
python -m joint_vol_calibration.models.nn_pricer

# C6: Greeks surface
python -m joint_vol_calibration.greeks.risk_monitor

# C8: Fit regime classifier
python -m joint_vol_calibration.signals.regime_classifier

# C10: Full backtest
python -m joint_vol_calibration.backtest.run_backtest
```

### Testing

```bash
pytest joint_vol_calibration/tests/ -v
# 374 tests, 0 failures
# Includes explicit look-ahead bias checks (test_lookahead.py)
```

### Requirements

```
Python 3.11+
flask · pandas · numpy · pyarrow
scikit-learn · scipy · xgboost
gunicorn (production)
```

No QuantLib. No Bloomberg. No paid data.

---

## Repository Structure

```
vol-system-dashboard/
├── dashboard/
│   ├── app.py                  # Flask app — 4 routes, read-only from data_store/
│   └── templates/              # base, index, calibration, greeks, backtest
├── joint_vol_calibration/
│   ├── data/                   # C1: pipeline, SQLite, CBOE/Yahoo downloaders
│   ├── models/                 # C2: heston.py · C3: pdv.py · C5: nn_pricer.py
│   ├── calibration/            # C4: joint_calibrator.py
│   ├── greeks/                 # C6: risk_monitor.py
│   ├── backtest/               # C7: delta_hedger.py · C10: backtest_engine.py
│   ├── signals/                # C8: regime_classifier.py · C9: signal_engine.py
│   └── tests/                  # 374 tests across all components
├── data_store/
│   ├── vol_system.db           # SQLite: SPX/VIX/options/regime tables
│   ├── calibrations/           # joint_cal_2026-03-24.pkl
│   ├── greeks/                 # greeks_surface.parquet
│   ├── signals/                # regime_classifier.pkl · regime_labels.parquet
│   └── backtest/               # full_results.parquet
├── requirements.txt
├── Procfile                    # gunicorn for Railway
└── README.md
```

---

## Known Limitations

- **Heston is misspecified for 2026 skew.** ρ = −0.99 boundary hit. Jump-diffusion (Bates/SVJ) would fit better but adds calibration complexity and non-uniqueness.
- **VIX options structural failure.** RMSE = 37.14 pts. Heston's CIR process cannot simultaneously fit SPX curvature and VVIX-implied vol-of-vol.
- **PDV fails in tail events.** COVID March 2020: −110pp forecast error. No path-dependent model without jumps can price tail risk correctly.
- **Backtest costs.** Options transaction costs (~$334k over 7 years) erode marginal signal alpha. Not modelled with realistic slippage curves.
- **VIX futures proxy.** Real VIX futures settlement data blocked by CBOE Cloudflare. `^VIX3M`/`^VIX6M` is an approximation.
- **Single-snapshot options data.** SPX and VIX options are from a single day (2026-03-24). A production system needs daily CBOE data or a paid feed.

---

## Author

**Navnoor Bawa** — 3rd year CS, Thapar Institute of Engineering & Technology

Previous work: Heston engine (30K paths, 20M+ steps/sec), SABR (14.1% smile error), Transformer statistical arbitrage, WTI ML pipeline.

- Website: [navnoorbawa.me](https://navnoorbawa.me)
- Substack: [navnoorbawa.substack.com](https://navnoorbawa.substack.com)
- LinkedIn: [linkedin.com/in/navnoorbawa](https://www.linkedin.com/in/navnoorbawa/)
- GitHub: [github.com/NavnoorBawa](https://github.com/NavnoorBawa)

---

*The negative backtest is presented without adjustment. Curve-fitting a strategy to look profitable in-sample is straightforward. The harder and more useful exercise is understanding precisely why it doesn't work — and this system documents that honestly.*
