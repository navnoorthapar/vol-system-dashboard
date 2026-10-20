"""
trials.py — the committed trials ledger (docs/trials_ledger.yaml).

The Deflated Sharpe Ratio needs N, the number of configurations tried before
the reported one was chosen. N is read from the ledger, which enumerates every
(sleeve, configuration, engine version) whose backtest result was looked at
on the 2018-2025 window, round by round, with the commit it came from.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from joint_vol_calibration.config import ROOT_DIR

LEDGER_PATH = ROOT_DIR / "docs" / "trials_ledger.yaml"
_REQUIRED_FAMILY_KEYS = ("name", "varied", "sleeves", "configurations")


def load(path: Path = LEDGER_PATH) -> dict:
    """Parse and validate the ledger."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        led = yaml.safe_load(fh)
    if led.get("schema") != "trials_ledger/1":
        raise ValueError(f"{path}: unknown schema {led.get('schema')!r}")
    seen = set()
    for rnd in led["rounds"]:
        if rnd["id"] in seen:
            raise ValueError(f"duplicate round {rnd['id']}")
        seen.add(rnd["id"])
        for fam in rnd.get("families") or []:
            missing = [k for k in _REQUIRED_FAMILY_KEYS if k not in fam]
            if missing:
                raise ValueError(f"round {rnd['id']} family {fam.get('name')!r} lacks {missing}")
            if int(fam["configurations"]) < 1 or not fam["sleeves"]:
                raise ValueError(f"round {rnd['id']} family {fam['name']!r} counts nothing")
            rep = fam.get("reproducible")
            if rep is not None and rep not in led.get("reproducible", {}):
                raise ValueError(f"family {fam['name']!r} names unknown reproducible set {rep!r}")
    return led


def _round_order(rid: str) -> int:
    digits = "".join(ch for ch in rid.split("-")[0] if ch.isdigit())
    return int(digits) if digits else 10 ** 6


def count_trials(led: dict, rounds: Iterable[str] | None = None,
                 from_round: str | None = None, include_evaluations: bool = False) -> dict:
    """N = Σ over families of len(sleeves) × configurations.

    ``rounds`` restricts to those ids; ``from_round`` keeps rounds whose
    number is ≥ that round's (e.g. "C16" → C16..C22).
    """
    keep = None if rounds is None else set(rounds)
    lo = None if from_round is None else _round_order(from_round)
    by_round, total = {}, 0
    for rnd in led["rounds"]:
        if keep is not None and rnd["id"] not in keep:
            continue
        if lo is not None and _round_order(rnd["id"]) < lo:
            continue
        n = sum(len(f["sleeves"]) * int(f["configurations"]) for f in rnd.get("families") or [])
        by_round[rnd["id"]] = n
        total += n
    out = {"n_trials": int(total), "by_round": by_round}
    if include_evaluations:
        ev = sum(int(e["count"]) for e in led.get("evaluations", []))
        out["n_evaluations"] = int(ev)
        out["n_trials_upper"] = int(total + ev)
    return out
