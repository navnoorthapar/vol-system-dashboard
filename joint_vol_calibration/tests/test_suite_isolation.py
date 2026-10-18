"""The suite never touches the tracked data_store/ (see conftest.py)."""
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from joint_vol_calibration import config
from joint_vol_calibration.data import database as dbm
from joint_vol_calibration.tests.conftest import DATA_STORE, REPO_ROOT


def test_repo_root_is_resolved_from_this_file_not_the_cwd():
    assert (REPO_ROOT / "joint_vol_calibration" / "config.py").is_file()
    assert DATA_STORE == REPO_ROOT / "data_store"


def test_data_layer_points_at_a_private_copy():
    shipped = DATA_STORE / "vol_system.db"
    for p in (config.DB_PATH, dbm.DB_PATH):
        assert Path(p).resolve() != shipped.resolve()
        assert Path(p).name == "vol_system.db"
    for p in (dbm.MARKET_DIR, dbm.OPTIONS_DIR, dbm.PARQUET_DIR):
        assert DATA_STORE.resolve() not in Path(p).resolve().parents


def test_modules_that_captured_db_path_at_import_are_redirected():
    from joint_vol_calibration.backtest import run_backtest
    assert Path(run_backtest.DB_PATH).resolve() == Path(dbm.DB_PATH).resolve()


def test_a_write_lands_in_the_copy_not_the_shipped_db():
    shipped = DATA_STORE / "vol_system.db"
    if not shipped.exists():
        return
    sentinel = "1990-01-02"
    dbm.insert_tbill_rates(pd.DataFrame({"date": [sentinel], "rate": [0.0123],
                                         "discount_yield": [0.0122]}), warn_dropped=False)
    with closing(sqlite3.connect(str(dbm.DB_PATH))) as con:
        assert con.execute("SELECT COUNT(*) FROM tbill_rates WHERE date=?",
                           (sentinel,)).fetchone()[0] == 1
        con.execute("DELETE FROM tbill_rates WHERE date=?", (sentinel,))
        con.commit()
    with closing(sqlite3.connect(f"file:{shipped}?mode=ro", uri=True)) as con:
        assert con.execute("SELECT COUNT(*) FROM tbill_rates WHERE date=?",
                           (sentinel,)).fetchone()[0] == 0
