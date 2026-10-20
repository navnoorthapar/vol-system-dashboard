# Documentation

| Document | What it covers |
|---|---|
| [install.md](install.md) | the three install routes (exact lock on Linux x86_64, ranges elsewhere, container) and which platforms resolve |
| [reproduce.md](reproduce.md) | `make reproduce`: rebuilding the committed artifacts from the captures and diffing them, with the last result |
| [benchmarks.md](benchmarks.md) | measured cost of pricing, implied vols, one loss evaluation and a full calibration; NumPy vs Numba vs C++ |
| [engineering.md](engineering.md) | CI jobs, lint and type baselines, property tests and their mutation check, accessibility, dependency audit |
| [model_validation/README.md](model_validation/README.md) | model inventory, conceptual soundness, outcomes analysis, ongoing monitoring and the C16–C22 findings log, in SR 26-2's structure |

The README is the summary; these files hold the detail and the measurements behind it. Each number in them names the
artifact, command or run it comes from.
