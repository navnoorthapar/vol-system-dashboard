# Engineering checks

What runs on every change, what it would catch, and what it found when it was introduced (C23, 2026-10-20; the
review round's fixes, 2026-10-21). Counts are from those days; the gates keep them from growing.

## CI (`.github/workflows/tests.yml`)

| Job | Runs | Gate |
|---|---|---|
| `test` | the full suite with statement and branch coverage, then a render of the site; Hypothesis profile `ci` (derandomized) on push/PR, `thorough` (10× examples, random) weekly | any failure; a tracked file changed by the suite or the render |
| `lint` | ruff syntax rules (annotated on the PR), then the broad rule set and mypy against committed baselines, and each committed baseline against the base branch's (`baseline_gate.py --against`) | a new finding in any (file, rule) pair; any mypy error in a strict module; a baseline that grew, unless the PR carries the `re-baseline` label |
| `bench` | every benchmark body once, without timing (`--benchmark-disable`) | a benchmark that no longer computes the right thing (each checks its result first) |
| `a11y` | axe-core 4.13.0 in Playwright 1.63.0 Chromium over the frozen site | any WCAG 2.0/2.1/2.2 A/AA violation at 1440 or 390 px in either colour scheme; horizontal scroll at 320–1920 px; a JS error; a request to another origin |
| `audit` | pip-audit 2.10.1 (installed from the hashed `scripts/ci/pip_audit_lock.txt`) on every pin of `requirements-lock.txt` at its public version, through `scripts/ci/pip_audit_gate.py` | an advisory with no entry in `scripts/ci/pip_audit_allowlist.json` (id or alias, package, a reason, an expiry at most 180 days ahead); an expired or malformed entry; no report after four attempts (an unreachable advisory service fails, never passes) |
| `mutation` | `make mutation` (weekly and on demand); `tests/test_ci_tools.py` checks every mutant's site on each push | a mutant the properties do not kill (other than the documented equivalent); a mutant whose site is no longer unique (stale) |
| `simulate` | `scripts/ci/simulate_next_session.py --full`: the daily bot's book slot, offline, on a copy of the checkout for the session after the database's last one, then the bot's three gates (read from `daily_refresh.yml`) and the full suite in that copy. The book is priced by parameters moved off the headline (κ × 0.6, θ × 1.2, σ × 0.7 below 1.9, ρ + 0.1, v₀ × 1.3) plus about 1 vol point of parity-preserving smile noise, and the job fails unless every fitted parameter moves at least 5%; the Cboe capture is replayed, then the forward ledger and every derived artifact | the simulated session not accepted or not recorded; any gate or suite failure, i.e. a test that pins a value the bot replaces (the 2026-10-20 incident class). What it does not catch: pins on features of real chains that a synthetic constant-carry Heston book does not reproduce |

`make reproduce` (an artifact outside its tolerance fails; docs/reproduce.md) runs weekly and on demand in its own
workflow, `.github/workflows/reproduce.yml`, which commits its result to `data_store/reproduce/status.json` (shown on
Model Risk) and opens or comments on one `reproduce-failure` issue when it fails.

Every job installs `requirements-lock.txt` with `pip install --require-hashes`: the lock records the SHA-256 of every
file each pin may install (from PyPI, and from the PyTorch CPU index for torch), so a file replaced on either index, or
a same-named upload on the other index, fails the install in the jobs that can push to main and gh-pages
(`test_infra_config.py::test_every_lock_pin_is_hashed_and_every_install_requires_hashes`). The lock is compiled from
`requirements-dev.txt`, so the test tools added in
C23 (Hypothesis, pytest-benchmark, mypy and the pandas/SciPy stubs) reach CI through that compile. The property and
benchmark files skip when their tool is missing, so the `test` and `bench` jobs first check that it is installed and
fail if not: a missing tool cannot turn into silently skipped tests.

Third-party actions are pinned by commit SHA (a test enforces it); `.github/dependabot.yml` proposes grouped weekly
updates for the actions, the pip set and the a11y tools, each release at least 14 days old, and leaves the CPU builds
of torch and xgboost to the lock compile.

## The daily bot (`.github/workflows/daily_refresh.yml`)

- **Run records.** Every run commits `data_store/runs/run_<run_utc>_<slot>.json` with its captures (what ran, the
  calibration outcome and the gate's reason, exit code, publish blockers, step timings, the commit and run id;
  `dashboard/ARTIFACTS.md` section 6). The stale-calibration banner names a rejected session from them.
- **Failure alert.** The last step, `if: failure()` and `continue-on-error` (it can never cost a commit or the deploy),
  runs `scripts/ci/bot_alert.py`: when this scheduled run and the scheduled run before it both failed (cancelled,
  queued and manual runs are passed over), it comments on the open issue labelled `bot-failure`, or opens one. One
  open issue at most; closing it re-arms the alert. The workflow has `issues: write` and `actions: read` for it.
- **Forward-ledger chain and attestation.** Each `data_store/forward/ledger_<s>.json` names the SHA-256 of the one
  before it; `scripts/ci/check_captures.py` rejects a new ledger that does not link to the head of the chain (or comes
  before a committed one), and `test_forward_ledger.py::test_committed_ledgers_form_one_valid_chain` checks the whole
  committed chain in the data gate. The job `attest-ledger` then attests each new ledger file of the capture commit
  with `actions/attest-build-provenance` (Sigstore, logged in the public Rekor log: an external timestamp the
  repository owner cannot back-date); it alone holds `id-token: write`, and it never fails the run. Verify a file with
  `gh attestation verify data_store/forward/ledger_<s>.json --repo navnoorthapar/vol-system-dashboard`.
- **Deploy streak.** Before the run's freeze, `scripts/ci/deploy_streak.py` reads the deploy commits on gh-pages
  (`Deploy session <s> from ...`; commit messages only, fetched with `--filter=tree:0` into a scratch repository) and
  writes `.deploy-streak.json` (gitignored): consecutive NYSE sessions with a deploy, across weekends and holidays,
  before this build. The Overview's pipeline panel prints it; a local or CI freeze has no file and no row.

## Lint and type gates

`scripts/ci/baseline_gate.py` runs `python -m ruff` / `python -m mypy` (the locked versions), counts findings per
(file, rule) and compares them with `scripts/ci/ruff_baseline.json` / `scripts/ci/mypy_baseline.json`. A count may
fall, never rise; `--update` rewrites a baseline and refuses to raise any count (`make baselines`). CI enforces the
rule on the baselines themselves: the lint job compares the checkout's baselines with the pull request base's
(`--against`), so a change that adds findings and re-baselines in the same PR fails unless it carries the
`re-baseline` label. The baselines were shrunk to the tree's 1,295 ruff and 370 mypy findings at the C23 integration
tip (from 1,622 and 390 when introduced).

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

**Can they fail?** `scripts/ci/mutation_check.py` (`make mutation`, weekly in CI) applies 13 one-line mutants to the
engine, the analytic Jacobian and the solver in a scratch copy and runs the properties (and the Jacobian tests) against
each. Last run (2026-10-21): 12 of 13 killed, no unexpected survivor:

| Mutant | Killed by |
|---|---|
| Bates jump compensator dropped | static arbitrage in strike |
| ρ sign flipped in the Heston CF | martingale condition of the CF |
| ρ sign flipped in the analytic-Jacobian CF | the Jacobian against fourth-order finite differences (`tests/test_heston_jacobian.py`) |
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

`scripts/ci/a11y/a11y.mjs` audits the site frozen from the commit: the seven pages (Overview, Surface, Calibration,
Risk, Research, Corrections, Model Risk) and the 404, with axe-core 4.13.0 (WCAG 2.0/2.1/2.2 A and AA) at 1440 and
390 px in the dark and light schemes, horizontal scroll at 320, 390, 768, 1024, 1440 and 1920 px, JS errors and
requests to another origin; `greeks.html` and `backtest.html` are redirects since C23 and are not audited. Charts are
the vendored Apache ECharts 6.1.0 (and ECharts-GL 2.1.0 for the one 3-D view). Last result (the review round's
dashboard, commit `70c68f4`, 2026-10-21): 0 failures on all 8 pages at both widths in both schemes, no horizontal
scroll from 320 to 1920 px; axe with best-practice rules added, 0 violations. A copy of the site with an image
without alt text, a 2,400 px element, a JS error and an external stylesheet injected fails with all four reported.

`scripts/ci/a11y/interact.mjs` runs in the same job on the same frozen site, with the published CSP in force, and
checks what a reader sees once `desk.js` has run: every bar of a series marked `signed` is drawn in the loss fill
below zero and the gain fill otherwise (read from the drawn chart's ECharts model against the chart's own spec),
every toned KPI value computes to the `--pos` / `--neg` token, and `?` opens the shortcut dialog centred at 1440
and 390 px. Before the P1 trust pass it reported 15 failures on the published site (the two losing sleeves drawn
in the gain colour, eleven KPI tiles without their tone, the dialog at the top-left corner at both widths).

## Dependencies

pip-audit 2.10.1 on `requirements-lock.txt`, 2026-10-20: 7 distinct advisories (10 rows; OSV lists some twice) in
three packages, none of them in the numerical stack. The lock was recompiled the same day with floors in
`requirements.txt` (commit `a92308c`; `test_infra_config.py::test_lock_keeps_the_security_floors` pins them):

| Package | Locked when audited | Advisories | Fixed in | Locked now |
|---|---|---|---|---|
| idna | 3.11 | CVE-2026-45409 | 3.15 | 3.20 |
| urllib3 | 2.6.3 | CVE-2026-44431, CVE-2026-44432, CVE-2026-97687, CVE-2026-97688, CVE-2026-97689 | 2.8.0 (2.7.0 for the first two) | 2.8.0 |
| werkzeug | 3.1.8 | CVE-2026-102598 | 3.1.9 | 3.1.9 |

torch `2.14.0+cpu` is not on PyPI under that version and was not audited then. urllib3 and idna sit on the daily job's
download path (requests, yfinance); werkzeug serves only the local preview and the freeze's in-process test client,
since the published site is static files.

Since FUND_GRADE_PLAN 0.6 the `audit` job is a gate (`scripts/ci/pip_audit_gate.py`). It audits every pin at its public
version (torch as `2.14.0`, so the CPU build is audited too) and fails on any advisory that
`scripts/ci/pip_audit_allowlist.json` does not cover with a reason and an expiry. Its first run (2026-10-28) found 44
advisories (43 distinct) in four packages; each is an allowlist entry with its reason. The pins were left unchanged
(the change added the hashes only):

| Package | Locked | Advisories | Fixed in | Why allowed, until |
|---|---|---|---|---|
| certifi | 2026.2.25 | PYSEC-2026-4320 (a root whose intermediates were compromised) | 2026.10.18 | applies to the bot's downloads; the fix is inside the 14-day cooldown until 2026-11-01; bump it then (entry expires 2026-11-08) |
| pillow | 12.3.0 | 39 (decoders, TIFF/PDF/EPS writers) | 13.0.0 | nothing here opens or decodes an image; matplotlib only writes the repo's own figures (2026-12-31) |
| filelock | 3.32.3 | 3 (ReadWriteLock, soft locks; two Windows only) | 4.2.1 | torch dependency; nothing here uses those locks (2026-12-31) |
| setuptools | 84.0.0 | GHSA-xmj3-9gf7-h52w (Windows legacy egg install) | 84.1.0 | torch dependency; Linux runners, pip installs (2026-12-31) |

## Licences

**Runtime (Python).** C23 added `arch` (NCSA; it installs `statsmodels`, BSD-3) for the bootstrap, SPA/StepM and the
GARCH baselines, and `PyYAML` (MIT) for the trials ledger, to `requirements.txt`; the rest of the runtime set is
unchanged.

**Shipped with the site.** Apache ECharts 6.1.0 (Apache-2.0, shipped with its NOTICE), ECharts-GL 2.1.0 (BSD-3),
zrender (BSD-3), claygl (BSD-2) and tslib (0BSD), vendored with their licence files under
`dashboard/static/vendor/licenses/`; the Inter and IBM Plex Mono subsets under the SIL Open Font Licence 1.1
(`dashboard/static/fonts/OFL-*.txt`). Chart.js (MIT) is no longer shipped.

**Development tools.** Permissive except Hypothesis (MPL-2.0, file-level copyleft) and axe-core (MPL-2.0); both are
used unmodified as test tools and are neither vendored nor shipped. mypy (MIT), pytest-benchmark (BSD-2),
pandas-stubs (BSD-3), scipy-stubs (BSD-3), pip-audit (Apache-2.0, installed in its CI job only, from
`scripts/ci/pip_audit_lock.txt`), Playwright
(Apache-2.0). Numba (BSD-2) is optional for one benchmark and is not a dependency.
