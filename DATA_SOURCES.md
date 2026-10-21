# Data sources and licensing register

Checked 2026-10-20. Every terms URL below was fetched that day and every quoted clause was matched against the fetched text. Quoted clauses are verbatim, and each section or heading is named. This register describes what the repository uses and what it publishes. It is not legal advice, and the licensing decision belongs to the repository owner.

## The risk

This repository is public, and it commits market data its vendors' terms do not let it redistribute:

- **Raw option quotes from Yahoo.** These are 5 SPX chains and 4 VIX chains under `data_store/options/`, plus 69,902 quote rows in `data_store/vol_system.db`.
- **Cboe's VX futures history.** 30,963 rows in the database.
- **Cboe index levels.** The VIX family partitions under `data_store/market/`.

Cboe's website terms allow viewing and downloading one copy for personal, non-commercial use. They bar storing the Materials "in an electronic retrieval system", publishing or distributing them, and using them "to verify or correct other data or information" without Cboe's prior written consent. Yahoo's terms bar automated collection without permission, and they bar building a database or archive that substitutes for Yahoo's services.

**Committing raw exchange or vendor quotes to a public repository is a licensing risk.** No commit here resolves that risk.

The owner has four options. They can be combined.

1. **Private data store.** Keep the raw captures (chains, VX futures, index histories) in a private GitHub repository or a private Hugging Face dataset. The Actions bot pulls them with a token secret. The public repository holds code and derived outputs only.
2. **Derived-only publishing.** Publish only what the models compute: calibrated parameters, residual grids, per-expiry carry, summary statistics, and straddle marks. Add a SHA-256 of each private raw snapshot so a result can still be tied to its input (`ChainCapture.sha256` computes one). The Kaggle dataset [kamham/data-derived-from-spx-options-20172025](https://www.kaggle.com/datasets/kamham/data-derived-from-spx-options-20172025) (MIT) uses this model, under what it describes as a Cboe academic licence (see the table below).
3. **A licence.** Cboe DataShop sells historical option data and lists an academic discount. A licence is the only route to publishing raw Cboe data.
4. **Purge history.** After option 1 or 2, run `git filter-repo` to remove raw files from history. This rewrites a public history, and existing clones and forks keep their copies. It needs the owner's decision and coordination.

**Owner decision (2026-10-21):** the raw option quotes stay committed in the public repository, disclosed in this register, and the git history is not rewritten (option 4 is declined). The risk above is accepted by the owner, not resolved. From C23 the daily bot therefore also commits the Cboe delayed-quotes books it captures (below). The rates C23 adds are US Government works and NY Fed data whose terms permit redistribution with a notice (see below). In detail:

- The daily bot's `capture` step (`scripts/cloud_refresh.py capture_cboe_chains`) stores Cboe's full delayed SPX and VIX chains once per book session under `data_store/options/cboe/` (`<und>_<session>_<HHMMSS>Z.parquet` plus a `.json` record with the timestamps and the SHA-256 of the raw response), about 0.9 MB a session (2026-10-20: SPX 28,980 lines, 821 KB; VIX 1,520 lines, 63 KB), so about 220 MB a year. `cboe_downloader.capture_option_chains` itself still writes only where its caller says.
- The new Cboe index histories (VIX1D, VIX1Y, SKEW, COR1M, COR3M, DSPX) and the strategy benchmarks are downloadable into the database. They are not committed: their partitions are opt-in through `database.REFERENCE_TABLES`.
- C23 does make Cboe's official files the stored source of the VIX index family, correcting the Yahoo prints (see the correction log). The risk is the same class as before. Before C23, every session row of `vix_daily` already equalled Cboe's VIX file to 0.005 (275 rows were tagged 'cboe'), and the term structure held Yahoo's republication of the same Cboe indices. Under option 1 or 2 these partitions move private with everything else.
- The cross-source check (`database.compare_sources`, `pipeline.refresh_vix_index_family`) uses Cboe's files to check and correct Yahoo's prints. That is the use Cboe's clause on verifying or correcting other data names.

## What the repository commits

| Path | Content | Source | Status |
|---|---|---|---|
| `data_store/options/spx_*.parquet` | 5 SPX chains (2026-03-24, 03-27, 05-29, 06-18, 10-16): 68,006 quotes | Yahoo (yfinance) | Raw quotes. **High risk** |
| `data_store/options/vix_*.parquet` | 4 VIX chains (2026-03-24 to 06-18): 1,896 quotes | Yahoo (yfinance) | Raw quotes. **High risk** |
| `data_store/options/cboe/` (C23) | Cboe delayed-quotes SPX and VIX chains, every line, one capture per book session (from 2026-10-20), with a JSON record per capture (fetch and quote time, book state, SHA-256 of the raw response) | Cboe (delayed quotes) | Raw quotes. **High risk** |
| `data_store/forward/` (C23) | The forward ledger: one hash-chained JSON per accepted session (signals, listed straddles at the captured bid/ask, marks) | Derived from the Yahoo chains | Derived (holds individual bid/ask prints of the straddles it trades) |
| `data_store/vol_system.db` | The rebuildable cache. Holds the chains above, VX futures (30,963 rows, 2013-01-02 to 2026-10-09), VX final settlements (201), and every market table | Cboe, Yahoo | Raw. **High risk** |
| `data_store/market/vix_term_structure/`, `vix_daily/` | Daily closes of VIX9D/VIX/VIX3M/VIX6M/VVIX and VIX OHLC, 2010 to present | Cboe official files (C23); 2 rows Yahoo-only | Index levels. Medium-high risk |
| `data_store/market/spx_ohlcv/` | S&P 500 daily OHLCV | Yahoo (^GSPC) | Medium risk |
| `data_store/market/tbill_rates/` | 13-week bill discount yield | Yahoo (^IRX) | Medium risk |
| `data_store/market/rates_daily/` (C23) | Treasury bill rates, par yield curve and SOFR, 2010 (SOFR 2018) to present: 92,208 quotes | US Treasury, NY Fed | Low risk (see each source) |
| `data_store/market/corrections/` (C23) | The VIX-family correction log: old and new value, action, backtest-window flag | Derived | Derived |
| `data_store/market/derived/` (C23) | SPX parity-implied rate per expiry against the Treasury curve (`spx_parity_rates_vs_treasury.csv`, 195 expiry rows over the 5 committed chains) and its summary by maturity bucket (`..._summary.csv`) | Derived | Derived |
| `data_store/calibrations/`, `backtest/`, `greeks/`, `signals/`, `surface/` | Model parameters and diagnostics, backtest results, Greeks surface and scenario grid, regime labels, the surface reports (whose residuals hold each quote's bid/ask) | Derived | Derived |

## Sources in use

### Cboe: index history files

- **Used for:**
  - The VIX index family (VIX, VIX9D, VIX3M, VIX6M, VVIX). This is the primary source from C23.
  - VRO (VX final settlements).
  - On request (C23, not committed): VIX1D, VIX1Y, SKEW, COR1M, COR3M, DSPX, VIXEQ, and the PUT/WPUT/BXM/BFLY/CNDR strategy benchmarks.
- **Endpoint:** `https://cdn.cboe.com/api/global/us_indices/daily_prices/<INDEX>_History.csv` (307 to `cdn-api.cboe.com`). No key.
- **Terms:** <https://www.cboe.com/terms>, "Terms and Conditions for Use of Cboe Websites", *Last Updated: November 16, 2022*. Section 2, "Copyrights, Trademarks, Service Marks, And Proprietary Rights":

  > You may view, print and download one copy of the Materials for your personal non-commercial use in connection with products and services offered by Cboe, provided that you maintain all copyright, trademark and other notices contained on the Materials. You may not otherwise copy, reproduce, alter, store either in hard copy or in an electronic retrieval system, license, transmit, display, broadcast, create a derivative work (for example, a financial product, service or index) from, use to verify or correct other data or information, publish, rent, sublicense, distribute, or otherwise use in whole or in part in any other manner the Materials without Cboe's prior written consent except to the extent that such use constitutes "fair use" under the "Copyright Act of 1976", as amended from time to time.

  Section 2 defines "Materials" to include "databases".
- **Committed:** the VIX-family partitions only. The other indices are not committed.
- **Risk:** medium-high.
- **Notes:**
  - Section 2 also requires keeping "all copyright, trademark and other notices contained on the Materials". The index names are Cboe's.
  - Cboe's VIX file prints a close on 33 NYSE holidays from 2022-05-30 to 2026-09-07 (and on 2004-06-11); the VIX9D, VIX3M, VIX6M and VVIX files print none. The file does not say what those prints are; the NYSE was closed, so the database writers drop them.

### Cboe: VX futures

- **Used for:** daily settlements per contract, and final settlements.
- **Endpoints:**
  - `https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_<expiry>.csv`
  - `https://www.cboe.com/us/futures/market_statistics/settlement/csv?dt=<date>`
- **Terms:** as above.
- **Committed:** yes, in `vol_system.db` (30,963 daily rows and 201 settlements).
- **Risk:** medium-high.

### Cboe: delayed option quotes

- **Used for:** the full SPX and VIX chains with greeks, one request each (`_SPX.json`, `_VIX.json`). `cboe_downloader.capture_option_chain` (C23) returns a capture with timestamps and a hash.
- **Endpoint:** `https://cdn.cboe.com/api/global/delayed_quotes/options/_<SYMBOL>.json`.
- **Terms:** as above.
- **Committed:** from C23, one capture of each book per session under `data_store/options/cboe/` (owner decision above). The calibrator and the database still read only the Yahoo chains; the captures are the record for the forward test, a VIX-options leg and a real-price study.
- **Risk:** high (committed).
- **Note:** the payload's `timestamp` is when Cboe generated the file, not when the quotes were live. On 2026-10-20 a fetch at 12:29:03Z returned timestamp 12:28:16Z and Last-Modified 12:28:19Z. The quotes are about 15 minutes older than the timestamp: a payload stamped 16:04:29Z held a latest option trade and an underlying last trade at 11:49:27 ET (15:49:27Z). `parse_cboe_options_payload` therefore stores quote_time = timestamp − 15 minutes.

### Yahoo Finance via yfinance

- **Used for:**
  - SPX OHLCV (^GSPC).
  - The 13-week bill (^IRX).
  - The SPX chain the daily bot fits.
  - Provisional VIX-family prints for a session Cboe has not posted yet.
- **Endpoint:** Yahoo's undocumented endpoints, through `yfinance` 1.7.0 (Apache-2.0; the licence covers the code only).
- **Terms:**
  - <https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html>, Yahoo Terms of Service, *Last updated: 4 August 2026*. Under "Member conduct", users agree not to:

    > access or collect data, or attempt to access or collect data, from our Services using any automated means, devices, programs, algorithms or methodologies, including but not limited to robots, spiders, scrapers, data mining tools, or data gathering or extraction tools, for any purpose without our express, prior permission.

    and not to:

    > use any material or content from, including without limitation any data, (a) to create any database, archive, mobile application, data feed, widget or any other aggregated data source that competes with or constitutes a material substitute for the Services

  - The yfinance README (package metadata of 1.7.0, the opening disclaimer) says it is "intended for research and educational purposes". It also says: "Remember - the Yahoo! finance API is intended for personal use only."
- **Committed:** the option chains, SPX OHLCV and ^IRX (table above).
- **Risk:** high for the chains, medium for the daily series.

### US Treasury: daily bill rates and par yield curve (C23)

- **Used for:** the zero curve in `data/rates.py`.
  - Bills: 4, 6, 8, 13, 17, 26 and 52 weeks, as bank discount and coupon equivalent.
  - Par curve: 1 month to 30 years.
- **Endpoint:** `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/<YYYY>/all?type=daily_treasury_bill_rates|daily_treasury_yield_curve&field_tdr_date_value=<YYYY>&page&_format=csv`. One request per year and file. No key.
- **Terms:** these are works of the US Government (17 U.S.C. § 105). <https://www.usa.gov/government-copyright> says a "Government work is something created by a U.S. government officer or employee as part of their official duties". Treasury's [Site Policies and Notices](https://home.treasury.gov/subfooter/site-policies-and-notices) page states no reuse restriction (checked).
- **Data notes:**
  - The interest-rate pages say the bill quotations "are obtained at approximately 3:30 PM each business day by the Federal Reserve Bank of New York".
  - The par curve is "derived from indicative, bid-side market price quotations (not actual transactions)". It is estimated with "a monotone convex spline method".
  - The nominal CMT series "will be floored at zero". The bills are not: their bank-discount yields were negative in March 2020.
- **Committed:** `data_store/market/rates_daily/` (90,073 quotes from 2010-01-04).
- **Risk:** low.

### NY Fed: SOFR (C23)

- **Used for:** the overnight rate in the box-spread comparison only. It is not a curve input.
- **Endpoint:** `https://markets.newyorkfed.org/api/rates/secured/sofr/search.json?startDate=…&endDate=…`. No key.
- **Terms:** <https://www.newyorkfed.org/privacy/termsofuse>, *Last Updated: 6/9/2023*.
  - "Permissible Use": "The New York Fed grants you a non-exclusive license, subject to the Terms, to use, copy, and distribute Content".
  - Automated access is allowed "provided your access does not have the effect of disabling, damaging, or interfering with the function of the Website".
  - "Use Restrictions", "Reference Rates": "If you use or distribute reference rate data or related information posted to the website, you must include the following notice and disclaimer with your presentation of that data or information". The notice is a template with bracketed fields to complete. The SOFR data "are calculated using data provided under a license granted to the New York Fed by DTCC Solutions LLC".
- **Publication:** SOFR is published "at approximately 8:00 a.m. ET" on the next business day, and `rates.sofr()` gates it accordingly.
- **Committed:** 2,135 SOFR rows in `rates_daily`.
- **Risk:** low, with the notice.
- **Required notice and disclaimer** (the template completed; `rates.SOFR_NOTICE` returns it with every value): *The SOFR Data is subject to the Terms of Use posted at newyorkfed.org. The New York Fed is not responsible for publication of the SOFR by joint-vol-calibration, does not sanction or endorse any particular republication, and has no liability for your use.*

### FRED, St. Louis Fed (implemented, not used by default)

- **Endpoint:** `https://fred.stlouisfed.org/graph/fredgraph.csv?id=…`. No key.
  - FRED drops requests that send a browser User-Agent string (connection closed without a response) and answers an honest one.
  - The multi-series CSV returns each series' full history whatever start date is requested.
  - It serves at most 12 ids per request and silently drops the rest (13 requested, the 13th absent). `rates.download_fred` requests 12 at a time and raises if a response lacks a requested id.
- **Terms:** <https://fred.stlouisfed.org/legal/>.
  - The summary says: "Don't do any data mining, scraping or extraction of FRED data."
  - The full terms prohibit using "the FRED® Services or FRED® Content in connection with the development or training of any software program or system or machine learning".
  - Third-party series carry their own copyright.
- **Why not by default:** FRED's DGS* series equal the Treasury par yields exactly (46,200 of 46,200 values, 2010 to 2026), FRED's SOFR equals the NY Fed's (2,134 of 2,134), and the H.15 3-month bill DTB3 equals Treasury's 13-week bank-discount quote (4,200 of 4,200 dates). The primary sources carry clearer terms. `rates.download_fred` stays as an opt-in fallback (`refresh_rates(fred_ids=…)`) for a Treasury outage.
- **Committed:** none.
- **Risk if automated:** medium.

## Considered and rejected or deferred (checked 2026-10-20)

| Source | What it offers | Decision and reason |
|---|---|---|
| [OptionsDX](https://www.optionsdx.com/product/spx-option-chain/) | SPX option chains 2010–2023 at end-of-day, 30/15/5/1-minute (49 product variants). The WooCommerce Store API (`/wp-json/wc/store/v1/products/<variant>`) prices **all 14 End of Day year variants, 2010 to 2023, at $0.00**. Intraday variants are paid (the product's price range tops out at $50). A free account checkout and ShareFile delivery are needed | **Deferred.** It is the best free source of real 16:00 SPX bid/ask for a real-price backtest. Its [Terms](https://www.optionsdx.com/terms/) page holds no data licence, only "All Sales for Digital Content or Services are Final". Research use only, raw files private, derived results only |
| [Massive (formerly Polygon.io)](https://massive.com/pricing?product=options) Options Basic | "$0/month", "5 API Calls / Minute", "2 Years Historical Data", end-of-day data and minute aggregates. No quotes, trades, greeks or IV | **Deferred.** The [Individuals Terms](https://massive.com/legal/individuals-terms-of-service) (last updated July 18, 2025) grant access "solely for your own personal, non-commercial, and non-business purposes". Market data falls under separate Market Data Terms. Bars are trade-based, not NBBO marks |
| [ThetaData](https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html) free tier | Stock and option end-of-day data from 2023-06-01. "To access any data, you must have the Theta Terminal running". The page gives both 20 and 30 requests/min | **Rejected.** The free tier has no index access, so no SPX |
| [Databento](https://databento.com/pricing) | The pricing page: "Sign up for $125 in free credits". The [2023 announcement](https://roadmap.databento.com/announcements/end-of-early-access-125-in-free-credits-for-all-users): "These credits are shared across your team and expire in 6 months" | **Rejected** for history. The credit might buy one or two event days of options data |
| [Tardis.dev](https://docs.tardis.dev/downloadable-csv-files/overview.md) | "Historical datasets for the first day of each month are available to download without API key". Crypto only (Deribit options) | **Deferred.** Candidate second market. The files were not fetched from this host |
| [DoltHub post-no-preference/options](https://www.dolthub.com/repositories/post-no-preference/options) | SPY and single names (the SQL API's first SPY row is 2019-02-09), a few expiries per date, bid/ask and greeks | **Rejected** for SPX. The C23 research found no SPX, SPXW or VIX rows (a re-check by symbol timed out on the public SQL API). The repository's `LICENSE.md` (read through the SQL API's `dolt_docs` table) is CC BY-SA 4.0, whose share-alike term would bind derived data |
| Kaggle mirrors: [kamham](https://www.kaggle.com/datasets/kamham/data-derived-from-spx-options-20172025) (MIT, derived SPX features 2017–2025), [dudesurfin](https://www.kaggle.com/datasets/dudesurfin/spy-options-eod-volatility-surface-2010-2023) (MIT, SPY 2010–2023), [shankerabhigyan](https://www.kaggle.com/datasets/shankerabhigyan/s-and-p500-options-spy-implied-volatility-2019-24) (CC0, SPY 2014–2025), [kylegraupe](https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022) (CC0) | Licences as read from the Kaggle API (`/api/v1/datasets/view/<owner>/<slug>`) | **Research cross-checks only.** The licences are asserted by the uploaders over data they did not produce: dudesurfin says its raw data "was sourced from OptionsDX"; shankerabhigyan's columns (contractID, mark, bid_size, ...) match Alpha Vantage's HISTORICAL_OPTIONS fields; kylegraupe names no origin. kamham is the model for derived-only publishing: "Only derived variables are included; no raw CBOE data is redistributed. Derived data publication is permitted under the CBOE academic license for non-commercial and non-compete purposes." (its own statement, not Cboe's) |
| [Alpha Vantage](https://www.alphavantage.co/documentation/) | HISTORICAL_OPTIONS is labelled "Premium". The [free tier](https://www.alphavantage.co/support/) gives "25 API requests per day and unlimited API requests for verified open-source or educational projects" | **Rejected.** Options history is premium. Whether an open-source exemption covers it was not pursued |
| Stooq | CSV downloads | **Rejected.** "As March 2026, Stooq requires an API key for data downloads" ([pandas-datareader #1012](https://github.com/pydata/pandas-datareader/issues/1012), opened 2026-04-13). stooq.com reset the connection from this host, so the claim rests on that issue |
| [Cboe DataShop](https://datashop.cboe.com/) | "Option EOD Summary", "Option Quotes" (NBBO intervals), "Option Trades". Paid, with an academic discount page | **Rejected** for cost. It is the licensed route if raw Cboe data is to be published |
| [HF thillsss/SPX-MES-VIX-data](https://huggingface.co/datasets/thillsss/SPX-MES-VIX-data) | About 310 MB of 1-minute bars (SPX, MES, MNQ; daily VIX) | **Rejected.** No licence and no dataset card (the Hub API returns no card data, no licence tag and no README) |
| FRED keyless CSV | The Treasury and NY Fed series again | **Opt-in fallback only** (see above) |

## How the code keeps to this register

- `database.MARKET_TABLES`, which the daily bot restores and commits, is unchanged. The C23 reference tables (`cboe_index_daily`, `rates_daily`) are written to partitions only when a caller passes `tables=database.REFERENCE_TABLES` or names them.
- `cboe_downloader.capture_option_chains(out_dir=None)` writes nothing. Every write goes to a directory the caller passes; the daily bot passes `data_store/options/cboe/` (owner decision above).
- `rates.refresh_rates()` uses Treasury and the NY Fed. FRED is used only when `fred_ids` are passed.
- `rates.sofr()` returns the NY Fed notice with every value.
- Every stored row carries its `source` ('cboe', 'yfinance', 'treasury', 'nyfed', 'fred'). From C23 the writers map aliases to one spelling.
