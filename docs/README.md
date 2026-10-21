# Documentation

| Document | What it covers |
|---|---|
| [install.md](install.md) | the three install routes (exact lock on Linux x86_64, ranges elsewhere, container) and which platforms resolve |
| [reproduce.md](reproduce.md) | `make reproduce`: rebuilding the committed artifacts from the captures and diffing them, with the last result |
| [benchmarks.md](benchmarks.md) | measured cost of pricing, implied vols, one loss evaluation and a full calibration; NumPy vs Numba vs C++ |
| [engineering.md](engineering.md) | CI jobs, lint and type baselines, property tests and their mutation check, accessibility, dependency audit |
| [model_validation/README.md](model_validation/README.md) | model inventory, conceptual soundness, outcomes analysis, ongoing monitoring and the C16–C23 findings log (rendered from `dashboard/corrections.json`), in SR 26-2's structure |
| [corrections.md](corrections.md) | every correction round C16–C23 as a postmortem (root cause, fix, regression tests, commits, effect), rendered from `dashboard/corrections.json` |
| [free_resources_scorecard.md](free_resources_scorecard.md) | every capability, the free resource behind it, its terms, what it achieved here and where free stops |
| [trials_ledger.yaml](trials_ledger.yaml) | every backtest configuration evaluated from C10 to C22 (the trial count behind the deflated Sharpe ratio) |

The README is the summary; these files hold the detail and the measurements behind it. Each number in them names the
artifact, command or run it comes from.
