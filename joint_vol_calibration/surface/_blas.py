"""
_blas.py — single-threaded BLAS around the surface fits.

The SLSQP problems here are small and dense (tens to a few hundred variables).
Multi-threaded OpenBLAS only adds synchronisation: on the shared 4-CPU build
machine under load, the eSSVI fit ran 70-150x slower with the default thread
pool (0.06 s single-threaded vs 8.8 s) and a three-slice calendar SVI test
20x slower (1.8 s vs 37 s). threadpoolctl ships with scikit-learn (a runtime
dependency); without it the limit is skipped.
"""

from __future__ import annotations

import contextlib
import functools


def single_thread_blas():
    """Context manager limiting BLAS to one thread (no-op without threadpoolctl)."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:                                  # optional speed-up only
        return contextlib.nullcontext()
    return threadpool_limits(limits=1, user_api="blas")


def blas_single_threaded(fn):
    """Decorator form of single_thread_blas."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with single_thread_blas():
            return fn(*args, **kwargs)
    return wrapper
