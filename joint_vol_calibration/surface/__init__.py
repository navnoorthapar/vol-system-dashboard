"""
surface — the quote-grade, arbitrage-aware market surface layer (C23).

Everything here works on one session's stored SPX chain and answers the
questions a market maker asks before trusting any model:

  quotes.py     the scoring universe: the calibrator's cleaned OTM quotes with
                bid / mid / ask implied vols, half-spreads in vol points,
                vega and delta / tenor buckets
  arbitrage.py  static-arbitrage screening of those quotes (vertical spreads,
                butterflies, calendar spreads; mid and executable) and the
                Cohen-Reisinger-Wang (2020) minimal-perturbation repair as a
                linear programme on scipy's HiGHS
  svi.py        raw SVI per expiry, accepted only with a numerical butterfly
                certificate (Durrleman g >= 0, cutting planes), free per
                slice or calendar-ordered against both neighbours with an
                ordering certificate (SVI-cal, started from the eSSVI fit:
                the arbitrage-free surface when both certificates pass)
  essvi.py      the eSSVI global surface (Gatheral-Jacquier butterfly
                conditions, monotone theta / wing slopes plus grid-certified
                slice ordering, Corbetta-Cohort-Laachir-Martini interpolation)
  metrics.py    quote-grade metrics in spread units and dense-grid
                butterfly / calendar violation counts for any iv(k, T)
  scorecard.py  scores any model given as iv(k, T) on the same quote
                universe, on its own held-out quotes and on the quotes no
                model was fitted to (eSSVI, SVI, SVI-cal, Heston at each
                stored model_version-2 fit; a registry and a pricer-to-iv
                hook for further models)
  vix.py        Cboe VIX methodology from the stored chain, the
                variance-swap strip by expiry and the ATM / var-swap ratio
  report.py     per-session artifacts under data_store/surface/<session>/
  _blas.py      single-threaded BLAS around the small dense fits

Conventions: k = ln(K / F_T) with F_T, D_T the per-expiry put-call-parity
forward and discount of option_chain.clean_option_chain; T in years from the
chain's T origin (option_chain.t_origin); implied vols are decimals in code
and vol points (x100) only in columns whose name ends in ``_vp``.
"""
