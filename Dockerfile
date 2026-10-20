# Reproducible environment for the test suite, the benchmarks and `make reproduce`.
#
#   docker build -t vol-system-dashboard .
#   docker run --rm vol-system-dashboard                                  # make test
#   docker run --rm -v "$PWD":/work vol-system-dashboard make reproduce   # needs the checkout
#
# The image holds the environment and a copy of the tracked tree without its git
# history (.dockerignore), which is all `make test` needs. `make reproduce` copies
# the tracked files (git ls-files), so it runs on the mounted checkout instead, as
# the dev container does.
#
# The base image is pinned by digest (python:3.11.15-slim-bookworm, multi-arch
# index, pushed 2026-08-05), so a rebuild gets the same OS layer. On linux/amd64
# the Python packages are the exact tested set (requirements-lock.txt, the one
# CI and the daily job install); on any other architecture the lock's
# x86_64-only builds (torch +cpu, xgboost-cpu) do not exist, so the ranges in
# requirements-dev.txt are resolved instead and the build says so. On Apple
# Silicon, `docker build --platform linux/amd64 .` gives the locked set under
# emulation.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# git: reproduce.py copies the tracked files; make: the entry points;
# g++: the optional C++ kernel benchmark (scripts/bench/test_bench_kernels.py).
RUN apt-get update \
 && apt-get install -y --no-install-recommends git make g++ \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt requirements-dev.txt requirements-lock.txt ./
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install -r requirements-lock.txt; \
    else \
        echo "NOTE: $(uname -m): installing the requirements-dev.txt ranges, not the x86_64 lock"; \
        pip install -r requirements-dev.txt; \
    fi

COPY . .
# A checkout mounted at /work belongs to the host user; let git read it as root.
RUN git config --global --add safe.directory /work

CMD ["make", "test"]
