"""
Benchmarks run against the shipped data_store/ exactly as the test suite
does: on a private copy of the database and partitions, with a guard that
fails the session if anything under data_store/ changes. The fixture is the
suite's own (joint_vol_calibration/tests/conftest.py), imported, not copied.
"""
from joint_vol_calibration.tests.conftest import _isolated_data_store  # noqa: F401
