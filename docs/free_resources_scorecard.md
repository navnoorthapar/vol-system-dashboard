# How far free resources go

The question this page answers: how much of a volatility desk's research stack can one person build using only free
data, free compute and permissively licensed software, and where does free stop? Every row names the resource, its
terms or licence, and what it achieved here, measured and traceable to a committed artifact or a commit. The terms of
every data source are quoted in [DATA_SOURCES.md](../DATA_SOURCES.md) (checked 2026-10-20); nothing here is legal
advice. Cost to date: $0.

## Data

| Capability | Free resource | Terms / licence | What it achieved here (measured) | Ceiling |
|---|---|---|---|---|
| SPX end-of-day option chains | Yahoo Finance through [yfinance](https://github.com/ranaroussi/yfinance) | yfinance Apache-2.0 (code only); Yahoo's terms bar automated collection and archives ([DATA_SOURCES.md](../DATA_SOURCES.md#yahoo-finance-via-yfinance)) | 5 stored SPX chains (68,006 quotes; 2,913–7,022 scored quotes per session) and 4 VIX chains (1,896) | No quote time on the closing book, no history: a chain not stored on its day is gone. Static arbitrage executable at the quoted bid/ask on all 5 sessions |
| Full, timestamped SPX and VIX books | Cboe delayed quotes, `cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json` and `_VIX.json` | [cboe.com/terms](https://www.cboe.com/terms): personal, non-commercial use | Every line of both books once per session from 2026-10-20 (first capture: SPX 28,980 lines over 55 expiries, VIX 1,520 lines over 13); quote lag measured at 15 min 2 s | 15-minute delay, a snapshot per request and no history; a forward record only |
| VIX index family 2010–2026 | Cboe index files, `cdn.cboe.com/api/global/us_indices/daily_prices/<INDEX>_History.csv` | [cboe.com/terms](https://www.cboe.com/terms) | 20,867 term-structure rows; replaced 136 of Yahoo's copies (38 by more than 0.25 pts), added 8 missing VVIX sessions, removed 31 holiday rows; 6 of 4,162 regime labels moved | Daily closes only |
| VX futures | Cboe per-contract settlement files | [cboe.com/terms](https://www.cboe.com/terms) | 30,963 daily rows, 174 contracts (2013-01 → 2027-06), 201 final settlements; Heston scored against them out of sample (1.85–3.18 pts) | Daily settlements; no intraday |
| Real-settlement short-vol benchmarks | Cboe strategy indices PUT, WPUT, BXM, BFLY, CNDR, VXTH, SET | [cboe.com/terms](https://www.cboe.com/terms) | Replicating PUT and WPUT implies β = 0.80 for the backtest's ATM-vol/VIX mark (0.63–0.90 by full year for PUT); the replication at β = 0.82 tracks PUT with 0.984 daily correlation (`data_store/backtest/inference/benchmarks.json`) | Index levels, not option prices: an external check of the mark, not a real-price backtest |
| Discount curve | U.S. Treasury daily bill rates and par yield curve; NY Fed SOFR API | U.S. Government works (17 U.S.C. § 105); NY Fed terms of use with a required notice | 92,208 quotes (29 series, 2010 → 2026); the 91-day zero rate is within a median 1.23 bp of ^IRX over 2,210 sessions; the Greeks surface's 1-year forward moved from 0.50% below its parity forward onto it | Daily; no intraday curve |
| SPX daily OHLC, 13-week bill | Yahoo (^GSPC, ^IRX) | as above | 4,224 and 4,222 sessions, 2010 → 2026 | — |
| Historical end-of-day SPX chains inside the backtest window (2018–2025) | [OptionsDX](https://www.optionsdx.com/product/spx-option-chain/) SPX end-of-day files 2010–2023, listed at $0.00 | No published licence | **Not used.** Checkout is a manual account flow and the files carry no licence, so they cannot be automated or committed | This is why every backtest P&L here is mark-to-model at β·VIX |
| Historical intraday SPX option quotes (NBBO) | none free | — | — | **Hard ceiling.** Cboe DataShop and OPRA-derived feeds are paid; Databento's $125 sign-up credit buys about one or two event days |
| Single-stock option chains for dispersion | none free at index-constituent scale | — | S3 is priced as the index leg only (a short SPX straddle) | Hard ceiling for a real dispersion book |

## Compute, hosting and operations

| Capability | Free resource | Terms / licence | What it achieved here (measured) | Ceiling |
|---|---|---|---|---|
| Daily production job | GitHub Actions, scheduled workflow ([daily_refresh.yml](../.github/workflows/daily_refresh.yml)) | free for public repositories | Two slots a day: market data and labels after the close; Cboe book capture, the joint fit (multi-start TRF, 1.7–2.1 s per fit on a quiet 4-vCPU machine), the surface report, diagnostics, Greeks, scenario grid and forward ledger after it, each behind its own test gate | Scheduled starts measured 87–319 minutes late, so snapshot timing jitters; no guaranteed runtime |
| CI | GitHub Actions ([tests.yml](../.github/workflows/tests.yml)) | free for public repositories | Full suite, ruff and mypy gates against shrinking baselines, a simulation of the bot's next session, an accessibility gate, a benchmark smoke test | Shared runners: timings are not benchmarks (docs/benchmarks.md quotes minima) |
| Static site | GitHub Pages, [navnoorbawa.me](https://navnoorbawa.me) | free for public repositories | Seven pages frozen from the artifacts; 0 axe-core violations at 1440 and 390 px in both colour schemes, no horizontal scroll from 320 to 1920 px | Static only: no server, no live data between builds |
| Reproducibility | Docker base image pinned by digest, `make reproduce` | — | 14 of 14 committed artifacts rebuilt bit for bit ([docs/reproduce.md](reproduce.md)) | — |

## Software (permissive licences only)

| Capability | Resource | Licence | What it does here |
|---|---|---|---|
| Numerics | NumPy, SciPy (HiGHS linear programs for the arbitrage repair), pandas | BSD-3 (HiGHS: MIT) | the pricing engine, calibrations, the Cohen–Reisinger–Wang repair LP |
| Statistics | arch, statsmodels | NCSA, BSD-3 | stationary bootstrap, SPA/StepM, Diebold–Mariano, GARCH baselines |
| Machine learning (research prototypes) | scikit-learn, XGBoost, PyTorch | BSD-3, Apache-2.0, BSD-3 | the demoted regime classifier and the NN surrogate; neither feeds a published number |
| Independent oracles | [QuantLib](https://pypi.org/project/QuantLib/) 1.44, [py_lets_be_rational](https://github.com/vollib/py_lets_be_rational) 1.1.2 | modified BSD, MIT | dev-only tests: Heston prices and implied vols against an external library |
| Testing and gates | pytest, Hypothesis, ruff, mypy, pytest-benchmark | MIT, MPL-2.0, MIT, MIT, BSD-2 | property tests over the parameter box, lint and type baselines |
| Site | Flask, Jinja2, Apache ECharts 6.1 (vendored), Inter and IBM Plex Mono (subset), Playwright, axe-core | BSD-3, BSD-3, Apache-2.0, OFL-1.1, Apache-2.0, MPL-2.0 | the frozen dashboard and its accessibility gate |
| Figures | matplotlib | Matplotlib licence (PSF-based, permissive) | the README figure, drawn from the committed surface report |
| Literature | arXiv, SSRN, the Cboe methodology papers | free to read | every model here follows a public paper (README references) |

Avoided on licence grounds: GPL (PyFENG, cvxopt), AGPL (pypbo) and unlicensed repositories; their methods were
re-implemented from the papers instead (the arbitrage repair on HiGHS, PBO from Bailey et al.).

## The verdict

At **daily frequency**, free resources reach a desk-standard methodology: full timestamped option books going
forward, every rate curve and Cboe index, real-settlement benchmarks, independent numerical oracles, CI, reproducible
builds and a public site. What free does not reach:

1. **The past.** Free sources give a forward record, not history. A real-price backtest over 2018–2025 needs the
   OptionsDX end-of-day files, downloaded by hand and kept private (no published licence); until then the backtest is a
   mark-to-model proxy and the forward ledger is the only executable-price evidence.
2. **Intraday options microstructure.** Historical intraday SPX NBBO is paid data. Intraday hedging, fill modelling and
   market-making studies would need a paid feed or a different market.
3. **Licensed redistribution.** The raw quotes this repository commits are covered by personal-use terms only; the
   owner accepted that risk (DATA_SOURCES.md). Publishing raw exchange data properly is a licence purchase.
4. **Independent validation.** No free resource replaces a second person who did not build the models.
