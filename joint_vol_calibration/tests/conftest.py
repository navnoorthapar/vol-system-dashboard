"""
Suite-wide isolation from the tracked data_store/.

1. Every test runs against a private COPY of data_store/vol_system.db, and the
   data layer's persistence directories (market partitions, per-session option
   chains, diagnostics) point into the session's tmp dir. Reads see exactly the
   shipped data; any write lands in the copy.
2. A guard hashes every file under data_store/ before the session and checks
   again after it: a test that modifies, creates or deletes a file there fails
   the run (as a teardown error naming the paths), even if it slipped past
   the redirects (a module that reads ``config.DATA_DIR`` directly).

Individual tests still redirect to their own tmp_path where they need an empty
or synthetic database; this only changes what an un-redirected test touches.
Paths here are resolved from this file, so the suite behaves the same from
any working directory.
"""
import hashlib
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_STORE = REPO_ROOT / "data_store"

# Heavy, untracked caches that tests only read (and may legitimately be absent).
_GUARD_SKIP_DIRS = {"mc_cache"}


def _snapshot(root: Path) -> dict:
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] in _GUARD_SKIP_DIRS:
            continue
        h = hashlib.blake2b(digest_size=16)
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(rel)] = h.hexdigest()
    return out


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_store(tmp_path_factory):
    from joint_vol_calibration import config
    from joint_vol_calibration.data import database as dbm

    before = _snapshot(DATA_STORE)

    work = tmp_path_factory.mktemp("data_store")
    db_copy = work / "vol_system.db"
    if Path(config.DB_PATH).exists():
        shutil.copy2(config.DB_PATH, db_copy)

    originals = {
        "DB_PATH": Path(config.DB_PATH),
        "MARKET_DIR": Path(dbm.MARKET_DIR),
        "OPTIONS_DIR": Path(dbm.OPTIONS_DIR),
        "PARQUET_DIR": Path(dbm.PARQUET_DIR),
    }
    targets = {
        "DB_PATH": db_copy,
        "MARKET_DIR": work / "market",
        "OPTIONS_DIR": work / "options",
        "PARQUET_DIR": work / "parquet",
    }
    # Persisted partitions are inputs as well as outputs: copy what is shipped.
    for name in ("MARKET_DIR", "OPTIONS_DIR"):
        if originals[name].exists():
            shutil.copytree(originals[name], targets[name])

    mp = pytest.MonkeyPatch()
    # Patch every already-imported project module that captured one of these
    # paths with ``from ... import DB_PATH`` (e.g. run_backtest), not just the
    # defining modules; modules imported later read the patched config value.
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("joint_vol_calibration"):
            continue
        for attr, orig in originals.items():
            val = getattr(mod, attr, None)
            if val is not None and Path(val) == orig:
                mp.setattr(mod, attr, targets[attr])
    yield work
    mp.undo()

    after = _snapshot(DATA_STORE)
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    created = sorted(after.keys() - before.keys())
    deleted = sorted(before.keys() - after.keys())
    if changed or created or deleted:
        pytest.fail(
            "the test session wrote to the tracked data_store/: "
            f"changed={changed} created={created} deleted={deleted}",
            pytrace=False,
        )
