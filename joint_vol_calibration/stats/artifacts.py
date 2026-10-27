"""
artifacts.py — deterministic artifact writing for the inference layer.

JSON: keys sorted, floats to 10 significant digits, NaN/inf as null, numpy
and pandas scalars converted, trailing newline. Parquet: the frame as given
(callers fix column order and sort rows), no index. The same inputs give the
same bytes, so a re-run that changes nothing leaves git clean.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SIG_DIGITS = 10


def jsonable(obj: Any, sig: int = SIG_DIGITS) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v, sig) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v, sig) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            return None
        return float(f"{x:.{sig}g}")
    if isinstance(obj, (pd.Timestamp, date)):
        return pd.Timestamp(obj).strftime("%Y-%m-%d")
    return obj


def write_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), sort_keys=True, indent=1, ensure_ascii=False,
                               allow_nan=False) + "\n")
    return path


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
