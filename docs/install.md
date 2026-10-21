# Installing

Python 3.11. Three routes, from most to least exact.

## 1. Linux x86_64: the tested set

```bash
python3.11 -m venv .venv && . .venv/bin/activate
make install            # pip install -r requirements-lock.txt
make test-fast          # about 1 minute; `make test` runs the full suite (about 10.5 minutes on 4 vCPU, measured 2026-10-21)
```

`requirements-lock.txt` pins every package, including the CPU builds of torch (`+cpu`, from the PyTorch index) and
`xgboost-cpu`. It is exactly what CI and the daily job install, so it is the environment the published numbers were
produced in. Those two builds exist only for Linux, which is why the lock does not install anywhere else.

## 2. Any other platform: the ranges

```bash
python3.11 -m venv .venv && . .venv/bin/activate
make install-any        # pip install -r requirements-dev.txt (runtime ranges + test/lint tools)
make test-fast
```

`requirements.txt` holds compatible ranges and `requirements-dev.txt` adds the test, lint and benchmark tools
(`pip install -r requirements.txt` alone is enough for `make demo` and the pipeline, not for the suite). On
2026-10-20 the ranges were resolved for each platform with `uv pip compile requirements-dev.txt --python-version 3.11
--python-platform <platform>` (uv 0.8.17; `MACOSX_DEPLOYMENT_TARGET` set where shown):

| Platform | Resolves | torch | xgboost |
|---|---|---|---|
| macOS 14+, Apple Silicon (`aarch64-apple-darwin`, `MACOSX_DEPLOYMENT_TARGET=14.0`) | yes | 2.14.1 | xgboost 3.2.0 |
| macOS 12–13, Apple Silicon (uv's default target) | yes | 2.11.0 (newer torch wheels need macOS 14) | xgboost 3.2.0 |
| Linux aarch64 (`aarch64-manylinux_2_28`) | yes | 2.14.1 | xgboost-cpu 3.2.0 |
| Windows x86_64 (`x86_64-pc-windows-msvc`) | yes | 2.14.1 | xgboost 3.2.0 |
| macOS, Intel (`x86_64-apple-darwin`) | **no**: PyTorch publishes no macOS x86_64 wheels for torch >= 2.4 | | |

numpy 2.4.6, pandas 3.0.6 and scipy 1.17.1 resolve identically on every platform that resolves. torch is required
because the backtest's trade records are unpickled through an import chain that reaches the NN surrogate
(`models/nn_pricer.py` imports torch at module level, `requirements.txt`), although no production computation calls
the surrogate. On an Intel Mac use route 3.

Numbers computed on another platform or library build can differ from the committed artifacts in the last bits;
the test suite compares artifacts at 1e-10 of each column's scale for that reason (`tests/artifact_tolerance.py`).

## 3. Container: Docker, dev container, Codespaces

```bash
docker build -t vol-system-dashboard .
docker run --rm vol-system-dashboard                                  # make test
docker run --rm -v "$PWD":/work vol-system-dashboard make reproduce   # on the mounted checkout (docs/reproduce.md)
```

The image carries the tracked tree without its git history (`.dockerignore`), which is all `make test` needs;
`make reproduce` copies the tracked files with `git ls-files`, so it runs on the mounted checkout.

The image starts from `python:3.11.15-slim-bookworm` pinned by digest and installs the lock on linux/amd64 (on any
other architecture it falls back to route 2 and says so; `docker build --platform linux/amd64 .` gets the lock under
emulation on Apple Silicon). `.devcontainer/devcontainer.json` builds the same image for VS Code dev containers and
GitHub Codespaces; `make help` lists the entry points.

Checked on 2026-10-20 (linux/amd64, Docker 29.3.1): the image builds from the pinned digest (2.9 GB) and
`docker run --rm --network none` runs `make test` to 1,595 passed in 8.6 minutes. The three Hypothesis modules were
skipped because that day's `requirements-lock.txt` did not yet carry the C23 test tools (it is recompiled from
`requirements-dev.txt`). `make reproduce` on the mounted checkout reproduced the Greeks surface at deviation 0.0.

## Entry points

| Command | What it does |
|---|---|
| `make test` / `make test-fast` / `make test-thorough` | full suite / core numerics and property tests / property tests at 10× the examples |
| `make lint` / `make typecheck` | ruff and mypy against their committed baselines (no new findings) |
| `make bench` | pytest-benchmark timings (docs/benchmarks.md) |
| `make reproduce` | rebuild backtest, Greeks, C7 study and calibrations from the captures and diff them (docs/reproduce.md) |
| `make freeze` / `make a11y` | the static site as published / its accessibility gate (Node 20+) |
| `make demo` | the dashboard on 127.0.0.1, read-only, from the committed artifacts |
| `make mutation` | curated mutants of the pricer and IV solver against the property tests |
| `make audit` | pip-audit of the lock (report only) |
