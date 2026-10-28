"""
desk_inventory.py — the model inventory's prose, computed from the artifacts (pure functions; no I/O).

dashboard/model_inventory.json typed facts that the daily job changes: 'σ sits on
its box ceiling in 2 of 5 fits' (six fits are stored, σ interior in the newest),
'the vomma shock is sized from the fitted σ, on its 2.0 cap in the headline fit'
(the headline σ is 1.833), 'No entry committed yet' (the forward ledger has an
entry). The Model Risk page rendered them beside checks that said otherwise.

Each such fact is now a {placeholder} in the JSON. app.load_model_risk reads the
values from the artifacts and fills each placeholder with one function from this
module, as {vx_futures_oos_rmse} already was. The literal prose around the
placeholders must not state anything that changes with the data. STALE_PROSE
lists what that prose may not state: a count over fits or sessions, a session
date, a parameter on a bound, or an artifact that does not exist yet.
test_dashboard.py holds the JSON to that rule.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from desk_format import _fmt, _num, ordinal, plural, signed

PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")
# the inventory fields the page prints as prose ('outputs_pending' is printed
# only while the output is unwritten, so its 'not yet' is conditional by construction)
PROSE_FIELDS = ("use", "inputs", "limitations", "outputs_note")
# What literal inventory prose may not state, with the reason: each is a fact the
# daily job changes, so it must come from a placeholder.
STALE_PROSE: tuple[tuple[str, str], ...] = (
    (r"\b\d+(?:st|nd|rd|th)? of \d+\b", "a count over stored fits or sessions"),
    (r"\b(?:all|every|none of the|only) \d+\b", "a count over stored fits or sessions"),
    (r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten) of\b", "a count over stored fits or sessions"),
    (r"\b\d{4}-\d{2}-\d{2}\b", "a session date"),
    (r"\b(?:cap|ceiling|floor)\b|\bon (?:its|a|the) [^;,()]{0,24}\bbound\b", "a parameter on a bound"),
    (r"\bno entry\b|\byet\b", "an artifact that does not exist yet"),
    # a share measured on the data is attributed to the correction round that
    # measured it ('5.9% of labels flip … (C22)'), which dates it
    (r"\b\d+(?:\.\d+)?%(?![^;]*\(C\d+\))", "a share measured on the data, not dated by its round"),
)


def stale_prose(text: str) -> list[tuple[str, str]]:
    """(matched text, reason) for each fact the literal prose states that the
    data can change; the {placeholders} are removed first."""
    literal = PLACEHOLDER.sub(" ", text or "")
    return [(m.group(0), why) for pat, why in STALE_PROSE for m in re.finditer(pat, literal)]


def placeholders(text: str) -> list[str]:
    return PLACEHOLDER.findall(text or "")


def fill(text: str, values: Mapping[str, str]) -> str:
    """Replace each {name} with values[name]. An unknown name raises KeyError, so a
    misspelt placeholder fails the build instead of printing braces."""
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def fill_models(models: Iterable[Mapping[str, Any]], values: Mapping[str, str],
                written: Callable[[Any], bool]) -> list[dict]:
    """The inventory's models with every prose field filled; 'outputs_pending'
    is dropped once ``written(outputs)`` says the output exists."""
    out = []
    for m in models:
        row = dict(m)
        for key in (*PROSE_FIELDS, "outputs_pending"):
            if isinstance(row.get(key), str):
                row[key] = fill(row[key], values)
        if row.get("outputs_pending") and written(row.get("outputs")):
            row.pop("outputs_pending")
        out.append(row)
    return out


def _join(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _of(k: int, n: int, what: str) -> str:
    """'2 of 6 gate-passing fits', 'all 6 …', 'none of the 6 …', 'the only …', 'no …'."""
    if k == 0:
        return f"no {what}" if n == 1 else f"none of the {n} {what}s"
    if n == 1:
        return f"the only {what}"
    return f"all {n} {what}s" if k == n else f"{k} of {n} {what}s"


def sigma_box_text(fits: Sequence[tuple[str, str | None]]) -> str:
    """M4: where σ sits in the gate-passing fits. ``fits``: (session, 'ceiling' |
    'floor' | None), the side of the production box σ is pinned to."""
    n = len(fits)
    if not n:
        return "no gate-passing fit is stored to say where σ sits"
    parts = []
    for side in ("ceiling", "floor"):
        on = [s for s, b in fits if b == side]
        if on:
            parts.append(f"σ sits on its box {side} in {_of(len(on), n, 'gate-passing fit')} "
                         f"({', '.join(on)})")
    if not parts:
        return f"σ is inside its box in {_of(n, n, 'gate-passing fit')}"
    return "; ".join(parts)


def sigma_box_cost_text(rows: Sequence[Mapping[str, Any]], boxed: Mapping[str, Mapping[str, Any]],
                        sigma_max: float, kappa_max: float) -> str:
    """M4: what the σ box costs, on the diagnosed session where freeing it lowers
    the joint loss most (diagnostics summary rows: sigma_free, kappa_free,
    sigma_free_oos, sigma_cap2_oos, sigma_free_loss_rel), with the boxed fit's
    σ and κ (``boxed``: session -> params) when it is stored."""
    keys = ("sigma_free", "kappa_free", "sigma_free_oos", "sigma_cap2_oos", "sigma_free_loss_rel")
    cand = [r for r in rows if all(_num(r.get(k)) is not None for k in keys)]
    box = f"refitted with σ ≤ {_fmt(sigma_max, 'g')} and κ ≤ {_fmt(kappa_max, 'g')}"
    if not cand:
        return f"no σ profile is stored ({box})"
    best = min(cand, key=lambda r: float(r["sigma_free_loss_rel"]))
    n = len(cand)
    if float(best["sigma_free_loss_rel"]) > -1e-6:
        return f"{box}, no diagnosed fit's loss falls ({plural(n, 'diagnosed session')})"
    p0 = boxed.get(best["session"]) or {}
    s0, k0 = _num(p0.get("sigma")), _num(p0.get("kappa"))
    k1 = float(best["kappa_free"])
    move = (f"from σ {_fmt(s0, '.2f')}, κ {_fmt(k0, '.1f')} " if s0 is not None and k0 is not None else "")
    ridge = ("; κ follows the σ cap along a κ–σ ridge" if k0 is not None and k1 >= 1.5 * k0 else "")
    return (f"{box}, the {best['session']} fit moves {move}to σ {_fmt(float(best['sigma_free']), '.2f')}, "
            f"κ {_fmt(k1, '.1f')} and its out-of-sample error from "
            f"{_fmt(float(best['sigma_cap2_oos']), '.2f')} to "
            f"{_fmt(float(best['sigma_free_oos']), '.2f')} vol pts "
            f"(loss {signed(float(best['sigma_free_loss_rel']) * 100, '.1f')}%), the box's largest cost over "
            f"{plural(n, 'diagnosed session')}{ridge}")


def vvix_boxed_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """M4: the boxed fits' model VVIX against the ^VVIX close (vvix_model_cap2,
    vvix_market of the diagnostics summary)."""
    pairs = [(float(r["vvix_model_cap2"]), float(r["vvix_market"])) for r in rows
             if _num(r.get("vvix_model_cap2")) is not None and _num(r.get("vvix_market")) is not None]
    if not pairs:
        return "the boxed fits' model VVIX is not stored"
    n = len(pairs)
    above = sum(1 for m, k in pairs if m > k)
    mods, mkts = [m for m, _ in pairs], [k for _, k in pairs]
    return (f"the boxed fits price VVIX above the market on {_of(above, n, 'diagnosed session')} "
            f"(model {_fmt(min(mods), '.0f')}–{_fmt(max(mods), '.0f')} against ^VVIX "
            f"{_fmt(min(mkts), '.0f')}–{_fmt(max(mkts), '.0f')})")


def vix_options_leg_text(weights: Sequence[Any]) -> str:
    """M4: the VIX-options leg's weight (w3) in the gate-passing fits; a fit
    that records no weight is not counted."""
    ws = [float(w) for w in weights if _num(w) is not None]
    n = len(ws)
    if not n:
        return "no gate-passing fit records the VIX-options leg's weight"
    tail = "" if n == len(weights) else " that record it"
    on = sum(1 for w in ws if w > 0)
    if on == 0:
        return f"the VIX-options leg has weight 0 in {_of(n, n, 'gate-passing fit')}{tail}"
    return f"the VIX-options leg is weighted in {_of(on, n, 'gate-passing fit')}{tail}"


def greeks_sigma_text(session: str | None, sigma: Any, box: tuple[float, float], side: str | None) -> str:
    """M6: the σ the vomma shock is sized from (the fit the Greeks surface is
    built from). ``side`` is the bound σ sits on ('upper', 'lower' or None)."""
    if _num(sigma) is None:
        return "the fit the surface is built from records no σ"
    lo, hi = (_fmt(x, "g") for x in box)
    where = {"upper": f"on its σ ≤ {hi} cap, so the box sizes the shock",
             "lower": f"on its σ ≥ {lo} floor, so the box sizes the shock"}.get(
                 side or "", f"inside its box ({lo} ≤ σ ≤ {hi})")
    return f"{_fmt(float(sigma), '.2f')} in the {session or 'undated'} fit the surface is built from, {where}"


def beta_later_text(later: Sequence[tuple[str, Any]]) -> str:
    """M7: ATM vol / VIX on each market-DB chain after β's estimation set
    (app.beta_estimation's 'later' chains, the ones the β-drift check reads)."""
    pts = [(s, float(r)) for s, r in later if _num(r) is not None]
    if not pts:
        return "the market DB holds no chain after the estimation set"
    lead = ("the market DB's one chain after the estimation set gives" if len(pts) == 1
            else f"the market DB's {len(pts)} chains after the estimation set give")
    return f"{lead} " + _join([f"{_fmt(r, '.2f')} ({s})" for s, r in pts])


def pdv_rank_text(rank: Mapping[str, Any] | None) -> str:
    """M9: the PDV forecaster's rank on each loss (app.pdv_forecast_rank)."""
    r = rank or {}
    losses = r.get("losses") or []
    if not losses:
        return "its rank among the compared forecasts is not stored"
    span = f" ({r['start']} → {r['end']})" if r.get("start") and r.get("end") else ""
    return _join([f"{ordinal(x['rank'])} of {x['n']} on {x['label']}" for x in losses]) + \
        f" among the forecasts compared{span}"


def classifier_text(oos_pct: Any, persistence_pct: Any) -> str:
    """M11: the research classifier against the persistence baseline (accuracy, %)."""
    a, b = _num(oos_pct), _num(persistence_pct)
    if a is None or b is None:
        return "Its out-of-sample evaluation is not stored"
    verb = "Loses to" if a < b else "Matches" if a == b else "Beats"
    return (f"{verb} the no-skill persistence baseline ({_fmt(a, '.1f')}% against {_fmt(b, '.1f')}% "
            "accuracy out of sample)")


_QUINTIC_SYMBOLS = {"epsilon": "ε", "lam_x": "λx", "lam_y": "λy", "theta": "θ", "a0": "α0", "a1": "α1",
                    "a3": "α3", "a5": "α5"}


def optimiser_stop_reason(stop_messages: Sequence[Any]) -> str:
    """Why an optimiser that did not converge stopped, from its own messages:
    'iteration limit' when one says so, else its first message."""
    msgs = [str(m) for m in stop_messages if m]
    if any("iteration" in m.lower() for m in msgs):
        return "iteration limit"
    return msgs[0] if msgs else "the optimiser reported failure"


def quintic_text(stored: bool, converged: bool, stop_messages: Sequence[Any],
                 on_bound: Sequence[tuple[str, Any]]) -> str:
    """M13: the stored Quintic OU fit's state (success flag, optimiser stop
    messages, parameters on a bound with their values)."""
    if not stored:
        return "No fit is stored"
    why = None if converged else optimiser_stop_reason(stop_messages)
    bounds = [f"{_QUINTIC_SYMBOLS.get(k, k)} on its {_fmt(_num(v), 'g')} bound" if _num(v) is not None
              else f"{_QUINTIC_SYMBOLS.get(k, k)} on a bound" for k, v in on_bound]
    parts = [p for p in (why, *bounds) if p]
    return ("Converged" if converged else "Not converged") + (f" ({'; '.join(parts)})" if parts else "")


def rough_h_text(rows: Sequence[Mapping[str, Any]], h_lo: float, h_hi: float) -> str:
    """M15: where the joint rough-Heston fit's H sits (joint_rough_H of the
    diagnostics summary): on its floor, and at ½ (which is Heston)."""
    hs = [(str(r["session"]), float(r["joint_rough_H"])) for r in rows
          if _num(r.get("joint_rough_H")) is not None]
    if not hs:
        return "no diagnosed session stores a rough-Heston fit"
    n = len(hs)
    floor = [s for s, h in hs if h <= h_lo + 1e-6]
    half = [s for s, h in hs if h >= h_hi - 1e-9]
    lo = _fmt(h_lo, "g")
    a = f"on its {lo} floor on {_join(floor)}" if floor else f"never on its {lo} floor"
    b = (f"½, which is Heston, on {_of(len(half), n, 'diagnosed session')}: {_join(half)}" if half
         else f"never ½ (Heston) over {plural(n, 'diagnosed session')}")
    return f"{a}; {b}"


def ledger_text(ledger: Mapping[str, Any]) -> str:
    """M18: what the forward ledger holds (desk_artifacts.load_forward_ledger,
    which refuses a broken hash chain)."""
    if ledger.get("state") != "ok" or not ledger.get("n"):
        return "No entry committed yet"
    span = (f"({ledger['first']})" if ledger["n"] == 1 or ledger["first"] == ledger["last"]
            else f"({ledger['first']} → {ledger['last']})")
    return f"{plural(ledger['n'], 'session')} registered {span}, its hash chain intact"
