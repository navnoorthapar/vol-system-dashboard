# Engineering checks

What runs on every change, what it would catch, and what it found when it was introduced (C23, 2026-10-20).
Counts are from that day; the gates keep them from growing.

## CI (`.github/workflows/tests.yml`)

| Job | Runs | Gate |
|---|---|---|
| `test` | the full suite with statement and branch coverage, then a render of the site; Hypothesis profile `ci` (derandomized) on push/PR, `thorough` (10× examples, random) weekly | any failure; a tracked file changed by the suite or the render |
| `lint` | ruff syntax rules (annotated on the PR), then the broad rule set and mypy against committed baselines | a new finding in any (file, rule) pair; any mypy error in a strict module |
| `bench` | every benchmark body once, without timing (`--benchmark-disable`) | a benchmark that no longer computes the right thing (each checks its result first) |
| `a11y` | axe-core 4.13.0 in Playwright 1.63.0 Chromium over the frozen site | any WCAG 2.0/2.1/2.2 A/AA violation at 1440 or 390 px in either colour scheme; horizontal scroll at 320–1920 px; a JS error; a request to another origin |
| `audit` | pip-audit 2.10.1 on `requirements-lock.txt` | none (report only, in the job summary) |
| `reproduce` | `make reproduce` (weekly and on demand) | an artifact outside its tolerance (docs/reproduce.md) |
| `simulate` | `scripts/ci/simulate_next_session.py --full`: the daily bot's book slot, offline, on a copy of the checkout for the session after the database's last one (a fit accepted on a book priced by the headline parameters, the Cboe capture replayed, the forward ledger and every derived artifact), then the bot's three gates (read from `daily_refresh.yml`) and the full suite in that copy | the simulated session not accepted or not recorded; any gate or suite failure, i.e. a test that pins a value the bot replaces (the 2026-10-20 incident class) |

Every job installs `requirements-lock.txt`, which is compiled from `requirements-dev.txt`, so the test tools added in
C23 (Hypothesis, pytest-benchmark, mypy and the pandas/SciPy stubs) reach CI through that compile. The property and
benchmark files skip when their tool is missing, so the `test` and `bench` jobs first check that it is installed and
fail if not: a missing tool cannot turn into silently skipped tests.

Third-party actions are pinned by commit SHA (a test enforces it); `.github/dependabot.yml` proposes grouped weekly
updates for the actions, the pip set and the a11y tools, each release at least 14 days old, and leaves the CPU builds
of torch and xgboost to the lock compile.

## Lint and type gates

`scripts/ci/baseline_gate.py` runs `python -m ruff` / `python -m mypy` (the locked versions), counts findings per
(file, rule) and compares them with `scripts/ci/ruff_baseline.json` / `scripts/ci/mypy_baseline.json`. A count may
fall, never rise; `--update` rewrites a baseline and refuses to raise any count (`make baselines`).

| Tool | Rule set | Findings when introduced | Before |
|---|---|---|---|
| ruff 0.17.1 | pyflakes, pycodestyle, bugbear, blind except, bandit, pylint E/W, pyupgrade, simplify, ruff, numpy, pandas-vet, naive datetimes, import order, pytest style and others (pyproject `[tool.ruff.lint]`) | 1,622 in 56 files | 136 against the old target set; CI gated 4 syntax rules |
| mypy 2.4.0 + pandas-stubs + scipy-stubs | default mode, whole tree | 390 in 30 files | not run |
| mypy, strict flags | `models/implied_vol.py`, `data/option_chain.py` | 0 (from 69) | not run |

Making the two modules strict found one real defect: `option_chain.year_fraction` raised `AttributeError` when given
an origin with no time in it ('' or NaN, as a missing `quote_time` cell reads). It now counts as the session close
(commit `ba360c8`; pinned by `test_properties_iv.py::test_year_fraction_clock`). New code in C23 adds no finding to
either baseline.

## Property tests

`tests/test_properties_pricing.py`, `test_properties_iv.py` and `test_properties_tooling.py` (Hypothesis): 27 tests,
16 s of CPU and 29–38 s wall time under the `ci` profile on the shared 4-vCPU VM, session setup included. The pricing
properties run on the unclipped engine over all of `config.HESTON_BOUNDS`, T from 1 day to 3 years and both signs of
carry; their tolerances derive from the engine's 1e-10 DF·F target (the module docstrings list every property and its
tolerance). They found no defect in the pricing code. The tooling properties found one in the new gate before it was
committed (the mypy-line parser rejected upper-case codes), and the `thorough` profile found one in a test generator
(a subnormal column whose perturbation underflows to no change), fixed in the test.

**Can they fail?** `scripts/ci/mutation_check.py` (`make mutation`) applies 12 one-line mutants to the engine and the
solver in a scratch copy and runs the properties against each:

| Mutant | Killed by |
|---|---|
| Bates jump compensator dropped | static arbitrage in strike |
| ρ sign flipped in the Heston CF | martingale condition of the CF |
| v0 term of the CF off by 1% | σ → 0 Black limit |
| κθ term of the CF off by 0.1% | σ → 0 Black limit |
| control-variate integral weight off by 1e-4 | static arbitrage in strike |
| absolute (not DF·F-relative) quadrature tolerance | static arbitrage in strike |
| step refinement skipped | static arbitrage in strike |
| integrated variance ignores v0 | σ → 0 Black limit |
| IV solver: wrong parity sign for ITM calls | σ → 0 Black limit (through the IV check) |
| IV solver: Newton iterations capped at 3 | σ → 0 Black limit (through the IV check) |
| Black-76: d1 drops the s/2 term | put-call parity between the two Lewis engines |
| IV solver: price-residual guard loosened 1e-7 → 1e-1 | **survives**: on 400,000 exact prices (σ 0.1%–1000%, T 1 d–5 y, ±12 sd) it changes 19 vols by at most 2.1e-11 relative and no status code, far below the solver's own worst error there (8.0e-5) and the round-trip tolerance (1e-9); the guard protects against inconsistent input, which no property over exact prices produces |

## Accessibility

`scripts/ci/a11y/a11y.mjs` on the site frozen from this commit: 0 axe-core violations on all five pages (index,
calibration, greeks, backtest, 404) at 1440 and 390 px in the dark and light schemes, no horizontal scroll at 320, 390,
768, 1024, 1440 or 1920 px, no JS errors, no request to another origin (the vendored Chart.js is used). A copy of the
site with an image without alt text, a 2,400 px element, a JS error and an external stylesheet injected fails with all
four reported.

## Dependencies

pip-audit 2.10.1 on `requirements-lock.txt`, 2026-10-20: 7 distinct advisories (10 rows; OSV lists some twice) in
three packages, none of them in the numerical stack:

| Package | Locked | Advisories | Fixed in |
|---|---|---|---|
| idna | 3.11 | CVE-2026-45409 | 3.15 |
| urllib3 | 2.6.3 | CVE-2026-44431, CVE-2026-44432, CVE-2026-97687, CVE-2026-97688, CVE-2026-97689 | 2.8.0 (2.7.0 for the first two) |
| werkzeug | 3.1.8 | CVE-2026-102598 | 3.1.9 |

torch `2.14.0+cpu` is not on PyPI under that version and was not audited. The fix is a lock recompile with these
three raised. urllib3 and idna sit on the daily job's download path (requests, yfinance); werkzeug serves only the
local preview and the freeze's in-process test client, since the published site is static files.

## Licences of the tools

Runtime dependencies are unchanged. The added development tools are permissive except Hypothesis (MPL-2.0, file-level
copyleft) and axe-core (MPL-2.0); both are used unmodified as test tools and are neither vendored nor shipped.
mypy (MIT), pytest-benchmark (BSD-2), pandas-stubs (BSD-3), scipy-stubs (BSD-3), pip-audit (Apache-2.0, installed in its
CI job only), Playwright (Apache-2.0). Numba (BSD-2) is optional for one benchmark and is not a dependency.
