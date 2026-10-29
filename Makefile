# Makefile: one entry point per job a reviewer or CI runs. `make help` lists them.
#
# Every target runs through $(PYTHON), so it uses whatever environment that
# interpreter has: `make install` gives the exact tested set (Linux x86_64),
# `make install-any` resolves the ranges on any other platform (docs/install.md).

PYTHON     ?= python
PYTEST      = $(PYTHON) -m pytest -p no:cacheprovider
SITE       ?= .site-build
OUT        ?= .reproduce
BENCH_JSON ?= .bench/benchmarks.json
# Single-threaded BLAS/OpenMP and a fixed hash seed for runs whose numbers are
# compared or published (reproduce, bench).
DETERMINISM = PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

FAST_TESTS = joint_vol_calibration/tests/test_heston.py \
             joint_vol_calibration/tests/test_heston_reference.py \
             joint_vol_calibration/tests/test_implied_vol.py \
             joint_vol_calibration/tests/test_option_chain.py \
             joint_vol_calibration/tests/test_infra_config.py \
             joint_vol_calibration/tests/test_properties_pricing.py \
             joint_vol_calibration/tests/test_properties_iv.py \
             joint_vol_calibration/tests/test_properties_tooling.py

.DEFAULT_GOAL := help
.PHONY: help install install-any test test-fast test-thorough lint typecheck baselines \
        bench bench-smoke freeze a11y recompute-audit reproduce reproduce-full mutation demo audit \
        docker-build docker-test clean

help:  ## list the targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## exact tested environment (Linux x86_64, CPython 3.11): requirements-lock.txt
	$(PYTHON) -m pip install -r requirements-lock.txt

install-any:  ## any other platform (macOS arm64, Windows, Linux aarch64): resolve the ranges
	$(PYTHON) -m pip install -r requirements-dev.txt

test:  ## the full suite (about 10.5 min on 4 vCPU at load 1-1.5, measured 2026-10-21)
	$(PYTEST) joint_vol_calibration/tests -q

test-fast:  ## pricing, IV, chain hygiene, CI config and the property tests (about 1 min)
	$(PYTEST) $(FAST_TESTS) -q

test-thorough:  ## property tests with 10x the examples, random (HYPOTHESIS_PROFILE=thorough)
	HYPOTHESIS_PROFILE=thorough $(PYTEST) joint_vol_calibration/tests/test_properties_*.py -q

lint:  ## ruff, broad rule set, against scripts/ci/ruff_baseline.json (no new findings)
	$(PYTHON) scripts/ci/baseline_gate.py ruff

typecheck:  ## mypy against scripts/ci/mypy_baseline.json; implied_vol and option_chain strict
	$(PYTHON) scripts/ci/baseline_gate.py mypy

baselines:  ## shrink both baselines after fixing findings (refuses to grow them)
	$(PYTHON) scripts/ci/baseline_gate.py ruff --update
	$(PYTHON) scripts/ci/baseline_gate.py mypy --update

bench:  ## pytest-benchmark timings, JSON to $(BENCH_JSON) (docs/benchmarks.md)
	@mkdir -p .bench $(dir $(BENCH_JSON)) && { [ -f .bench/.gitignore ] || printf '*\n' > .bench/.gitignore; }
	$(DETERMINISM) $(PYTEST) scripts/bench --benchmark-only --benchmark-json=$(BENCH_JSON) \
		--benchmark-sort=fullname -q

bench-smoke:  ## every benchmark body once, no timing (what CI runs)
	$(PYTEST) scripts/bench --benchmark-disable -m "not slow" -q

freeze:  ## the static site exactly as published, into $(SITE)
	$(PYTHON) dashboard/freeze_site.py $(SITE)

a11y: freeze  ## axe-core (WCAG 2.2 AA) + overflow check on the frozen site (needs Node 20+)
	cd scripts/ci/a11y && npm ci --no-audit --no-fund && npx playwright install chromium
	node scripts/ci/a11y/a11y.mjs $(SITE)
	node scripts/ci/a11y/interact.mjs $(SITE)

recompute-audit: freeze  ## every number on the Overview and Surface, recomputed from data_store independently
	$(PYTHON) scripts/ci/recompute_audit.py $(SITE)

reproduce:  ## rebuild backtest, Greeks, C7 and the v2 calibrations in a scratch copy; diff vs committed
	$(DETERMINISM) $(PYTHON) scripts/ci/reproduce.py --out $(OUT)

reproduce-full:  ## reproduce + beta grid, C22 waterfall and the 200-path S3 null test
	$(DETERMINISM) $(PYTHON) scripts/ci/reproduce.py --out $(OUT) --full

mutation:  ## curated pricer / IV-solver mutants against the property tests
	$(PYTHON) scripts/ci/mutation_check.py

demo:  ## the dashboard on 127.0.0.1 from the committed artifacts (read-only)
	$(PYTHON) dashboard/app.py

audit:  ## known vulnerabilities in the locked set (pip-audit; report only)
	@$(PYTHON) -c "import pip_audit" 2>/dev/null || { echo "pip-audit is not installed: pip install pip-audit==2.10.1"; exit 1; }
	$(PYTHON) -m pip_audit -r requirements-lock.txt --no-deps --disable-pip --progress-spinner off || true

docker-build:  ## the pinned image (Dockerfile)
	docker build -t vol-system-dashboard .

docker-test: docker-build  ## make test inside the image
	docker run --rm vol-system-dashboard make test

clean:  ## remove local build, benchmark and reproduction output
	rm -rf $(SITE) .bench $(OUT) .mypy_cache .ruff_cache .hypothesis .benchmarks
