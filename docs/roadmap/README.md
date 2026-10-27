# UI roadmap after C23 (PR #4)

The owner's brief: make the UI the best in the world at what it does (a volatility research and model-validation desk tool).

## A. Un-defer (worth building; details in docs/roadmap/ui_round2_deferred.json)
- B9  Risk-neutral density + tail probabilities from the certified SVI-cal surface (Breeden–Litzenberger on the fitted
      slices; probabilities only inside the quoted range, wing beyond it labelled as SVI-wing extrapolation).
- B8  Volatility cone (Burghardt–Lane) with the surface's ATM term structure overlaid (realized-vol percentiles from SPX OHLCV).
- B12 Research: round-trip trade panel + worst-drawdown table (resolve the 111 'combined' trade-log records first).
- D12 Sortable, copyable tables (aria-sort, TSV copy) — progressive enhancement, keyboard accessible.
- Q14 Site-wide units/sign conventions block + glossary (<dfn>/<abbr>) linked from first use on every page.
- Q20 Phone first screen: page subject and first number above the fold (compact status strip / ticker / banner).
- B13 Compact vol-context strip only if it fits the chrome budget at 390 px (director rejected for the global ticker; consider per-page).
- Skip: B14 (low value), Q4-rerun, Q8-NAV, Q15, Q19-bands, D10-axes (rejected on merit).

## B. Loop until dry (max 3 rounds)
Each round: freeze HEAD; 3 independent critics in parallel, read-only, each scores every page 1–10 with concrete findings:
  1. senior Citadel vol quant (trust, numbers recomputed from data_store, desk conventions, 10s/60s/5min test),
  2. head of desk-tools design (hierarchy, type, spacing, colour, charts, interaction, phone),
  3. skeptical model validator (SR 26-2 lens: every claim traceable, caveats, overclaims).
Dedup vs findings already seen; one engineer fixes high/medium (tests, gates, commits per item); stop when no new
high/medium and every page median score >= 9, or after 3 rounds.

## C. Performance budget
Per page: transfer size, DOM nodes, time to first chart, long tasks, CLS (measured with Playwright); fix regressions
(surface.html grew past 530 KB).

## D. Final
Full suite, lint/typecheck, a11y gate (incl. the 0.06em font-metric margin), simulate_next_session --full, make reproduce,
merge main data if moved, freeze, screenshots, republish preview (same artifact URL), update PR #4 body.

## Status when this was written (2026-10-26)
- Done and in PR #4 (branch c23/integration, head 7bced23): packages P1–P3 of ui_plan.json and P4.1.
- Not done: P4.2–P4.5 of ui_plan.json (each item has detail and acceptance criteria there), then sections A–D above.
- Every item: a failing-then-passing test, CSP script-src 'self', axe 0 violations, no overflow 320–1920 px (and the 0.06 em
  font-metric margin at 320 px), no test that pins a value the daily bot replaces (check with
  `python scripts/ci/simulate_next_session.py --full`), ruff/mypy baselines may only shrink.
