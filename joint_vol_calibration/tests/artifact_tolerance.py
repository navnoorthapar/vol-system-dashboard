"""
artifact_tolerance.py — comparing a recomputation with a committed artifact.

The artifacts are written on one machine and recomputed on others (CI runners,
a reviewer's laptop). numpy picks its SIMD kernels for the CPU it runs on
(AVX-512 where the artifacts were written, AVX2 on GitHub's runners), so the
same code on the same inputs can differ in the last bits. CI run 39166521027
failed on exactly that: the C7 study and the Quintic targets, recomputed
without AVX-512, differed from the committed files by 1.1e-15 relative.

Reproduced here with NPY_DISABLE_CPU_FEATURES=X86_V4, the worst case over
every column of the C7 study is 2.8e-13 of the column's largest value
(pnl_residual_a, a difference of nearly equal P&L terms); every other column
is within 1e-14. ARTIFACT_RTOL, taken relative to each column's largest
value, is 360 times the worst case and still fails on any change a code edit
would make (a 1e-8 move in a vol, 1e-6 dollars in a P&L).
"""
import numpy as np
import pandas as pd

ARTIFACT_RTOL = 1e-10


def assert_close_to_artifact(actual, expected, err_msg: str = "") -> None:
    """|actual - expected| <= ARTIFACT_RTOL * max|expected|, element-wise.
    NaN must meet NaN; an all-zero column must match exactly."""
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    finite = np.isfinite(b)
    scale = float(np.max(np.abs(b[finite]))) if finite.any() else 0.0
    np.testing.assert_allclose(a, b, rtol=0, atol=ARTIFACT_RTOL * scale, err_msg=err_msg)


def assert_frame_matches_artifact(df: pd.DataFrame, committed: pd.DataFrame) -> None:
    """Same columns, order, dtypes and index; float columns per
    assert_close_to_artifact, every other column exactly equal."""
    assert list(df.columns) == list(committed.columns)
    pd.testing.assert_index_equal(df.index, committed.index, exact=True)
    pd.testing.assert_series_equal(df.dtypes, committed.dtypes)
    for col in committed.columns:
        if pd.api.types.is_float_dtype(committed[col]):
            assert_close_to_artifact(df[col], committed[col], err_msg=col)
        else:
            np.testing.assert_array_equal(df[col].to_numpy(), committed[col].to_numpy(),
                                          err_msg=col)
