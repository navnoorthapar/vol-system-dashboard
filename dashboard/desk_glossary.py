"""
desk_glossary.py — the site's units and sign conventions, its glossary, and the
first-use links every page makes to it (Q14; pure functions, no Flask).

Each definition states the formula the code computes (the module named in
``source``), not a textbook variant. The Model Risk page renders TERMS as
#definitions (one id per term, ``def-<id>``); on every page the first use of
each term in the prose of <main> (a paragraph, list item or definition) becomes
a link to it (an <abbr> with the expansion for an acronym). Text inside links,
buttons, headings, tables, tiles, code, scripts, SVG and hidden or
screen-reader-only elements is never touched, and a definition never links
its own term (an element with ``data-term="<id>"`` leaves that term alone).
"""
from __future__ import annotations

import html as _html
import re
from collections.abc import Sequence
from typing import Any

# The house style, stated once (base.html's footer and #definitions link here).
CONVENTIONS: tuple[tuple[str, str], ...] = (
    ("Vol levels", "in %: e.g. an implied or realised vol of 20.00% (σ = 0.2000)."),
    ("Vol differences and errors", "in vol pts, percentage points of vol: 21.00% − 20.00% = 1.00 vol pts."),
    ("VIX-family levels", "VIX, VIX9D, VIX3M, VIX6M and VVIX in index pts, never '%': e.g. VIX 20.00."),
    ("Quote residuals", "in half-spreads (|z|): 1 is the edge of the quoted bid–ask."),
    ("Money", ("$ per SPX contract (×100 multiplier) where a row says so; $ with k / M suffixes; "
               "backtest P&L is marked to model at β·VIX on collateral assumed to earn T-bills "
               "separately, so it is an excess return.")),
    ("Signs", ("model − market (a positive bias is a model above the mid), implied − realised (a "
               "positive premium is implied above realised), replicated − Cboe close; each table's "
               "caption restates its sign.")),
    ("Moneyness and maturity", ("ln(K/F) against the expiry's parity forward F; days are calendar "
                                "days to expiry unless a column says NYSE sessions; 1M = 30 days.")),
    ("Probabilities", ("risk-neutral where the Surface prints them (they price claims; they are not "
                       "forecasts).")),
)

# id, term (as defined), expansion (the <abbr> title; None for a plain phrase),
# the first-use pattern (matched on the page's HTML text, entities as written),
# the definition, its units, where it is used, the code it quotes.
TERMS: tuple[dict[str, Any], ...] = (
    {"id": "half-spread", "term": "half-spread", "abbr": None, "pat": r"\bhalf-spreads?\b",
     "text": "Half a quote's bid–ask in implied-vol terms, (ask IV − bid IV)/2. Residuals are measured in "
             "it; the surface fits divide each residual by it, floored at 0.05 vol pts.",
     "units": "vol pts", "used": "Surface, Calibration, Overview",
     "source": "joint_vol_calibration/surface/essvi.py", "fn": "fit_essvi, h_floor"},
    {"id": "z", "term": "|z|", "abbr": "absolute residual in half-spreads", "pat": r"\|z\|",
     "text": "|model IV − mid IV| / half-spread of one quote; |z| ≤ 1 is inside the quoted bid–ask. "
             "Scorecards report its median over a model's out-of-sample quotes.",
     "units": "half-spreads", "used": "Surface, Calibration, Overview",
     "source": "dashboard/desk_artifacts.py", "fn": "residual_stats"},
    {"id": "vol-pts", "term": "vol pts", "abbr": "volatility points", "pat": r"\bvol pts\b",
     "text": "A difference of two vols in percentage points: 21.00% against 20.00% is 1.00 vol pts. "
             "RMSE, bias, RR, BF and premia are in vol pts; vol levels are in %.",
     "units": "percentage points of vol", "used": "every page", "source": None},
    {"id": "index-pts", "term": "index pts", "abbr": "index points", "pat": r"\bindex pts\b",
     "text": "The unit of a VIX-family level and its errors (VIX 20.00 is 20.00 index pts), and of SPX "
             "option prices (×100 for $ per contract).",
     "units": "index points", "used": "Overview, Calibration, Surface, Model Risk",
     "source": None},
    {"id": "atmf", "term": "ATMF", "abbr": "at-the-money-forward vol", "pat": r"\bATMF\b",
     "text": "The implied vol of the SVI-cal slice at the forward strike, ln(K/F) = 0.",
     "units": "%", "used": "Surface", "source": "dashboard/desk_surface.py", "fn": "delta_grid"},
    {"id": "rr25", "term": "RR25", "abbr": "25-delta risk reversal", "pat": r"\bRR25\b",
     "text": "Vol of the 25Δ call − vol of the 25Δ put, at undiscounted forward delta on the slice's own "
             "vol (not premium-adjusted); negative: puts over calls.",
     "units": "vol pts", "used": "Surface", "source": "dashboard/desk_surface.py", "fn": "delta_grid"},
    {"id": "bf25", "term": "BF25", "abbr": "25-delta butterfly", "pat": r"\bBF25\b",
     "text": "(25Δ call vol + 25Δ put vol)/2 − ATMF, same deltas as RR25.",
     "units": "vol pts", "used": "Surface", "source": "dashboard/desk_surface.py", "fn": "delta_grid"},
    {"id": "svi-cal", "term": "SVI-cal", "abbr": "calendar-ordered SVI surface", "pat": r"\bSVI-cal\b",
     "text": "The calendar-ordered SVI surface: one raw-SVI slice (a, b, ρ, m, σ) per listed expiry, each "
             "with a butterfly certificate (Durrleman's g ≥ 0, so its density is non-negative).",
     "units": "—", "used": "Surface, Calibration, Overview",
     "source": "joint_vol_calibration/surface/svi.py"},
    {"id": "essvi", "term": "eSSVI", "abbr": "extended surface SVI", "pat": r"\beSSVI\b",
     "text": "One extended-SSVI surface fitted jointly to every expiry, its parameters constrained "
             "against static arbitrage.",
     "units": "—", "used": "Surface", "source": "joint_vol_calibration/surface/essvi.py"},
    {"id": "oos", "term": "out of sample", "abbr": None, "pat": r"\bout[- ]of[- ]sample\b",
     "text": "Quotes a model was not fitted to: the surfaces hold out alternate strikes of each expiry; "
             "Heston is fitted to its anchors and scored on every other quote.",
     "units": "—", "used": "Surface, Calibration, Overview", "source": "data_store/surface/",
     "fn": "surface.json fit_set"},
    {"id": "psr", "term": "PSR", "abbr": "probabilistic Sharpe ratio", "pat": r"\bPSR\b",
     "text": "Bailey & López de Prado (2012): P(true Sharpe > 0) given the sample Sharpe, its length, "
             "skew and kurtosis, assuming IID returns (the bootstrap intervals are the primary inference).",
     "units": "probability", "used": "Research", "source": "joint_vol_calibration/stats/sharpe.py"},
    {"id": "dsr", "term": "DSR", "abbr": "deflated Sharpe ratio", "pat": r"\bDSR\b",
     "text": "Bailey & López de Prado (2014): the PSR against the expected maximum Sharpe of N̂ trials, "
             "N̂ = ρ̄ + (1 − ρ̄)·M for M trials with average pairwise correlation ρ̄.",
     "units": "probability", "used": "Research", "source": "joint_vol_calibration/stats/sharpe.py"},
    {"id": "spa", "term": "SPA", "abbr": "superior predictive ability test", "pat": r"\bSPA\b",
     "text": "Hansen (2005), the studentised and recentred Reality Check (arch.bootstrap.SPA). H0: no "
             "strategy has a positive expected excess return; the 'consistent' p-value is the headline.",
     "units": "p-value", "used": "Research", "source": "joint_vol_calibration/stats/multiple_testing.py"},
    {"id": "stepm", "term": "StepM", "abbr": "stepwise multiple testing", "pat": r"\bStepM\b",
     "text": "Romano & Wolf (2005) via arch.bootstrap.StepM: which strategies, if any, beat cash with "
             "family-wise error rate ≤ the test size.",
     "units": "—", "used": "Research", "source": "joint_vol_calibration/stats/multiple_testing.py"},
    {"id": "pbo", "term": "PBO", "abbr": "probability of backtest overfitting", "pat": r"\bPBO\b",
     "text": "Bailey, Borwein, López de Prado & Zhu (2017), by CSCV: the share of splits in which the "
             "strategy with the best in-sample Sharpe ranks at or below the median out of sample "
             "(logit of its relative rank ≤ 0).",
     "units": "probability", "used": "Research", "source": "joint_vol_calibration/stats/multiple_testing.py"},
    {"id": "cscv", "term": "CSCV", "abbr": "combinatorially symmetric cross-validation", "pat": r"\bCSCV\b",
     "text": "The PBO's resampling: the sample is cut into S blocks and every combination of S/2 blocks "
             "serves once as the in-sample set, its complement as the out-of-sample set.",
     "units": "—", "used": "Research", "source": "joint_vol_calibration/stats/multiple_testing.py"},
    {"id": "mcs", "term": "MCS", "abbr": "model confidence set", "pat": r"\bMCS\b",
     "text": "Hansen, Lunde & Nason (2011) via arch: the forecasts not rejected as worse than the best at "
             "the test size, with a p-value per forecast.",
     "units": "p-value", "used": "Research", "source": "joint_vol_calibration/stats/forecast.py"},
    {"id": "qlike", "term": "QLIKE", "abbr": "quasi-likelihood loss", "pat": r"\bQLIKE\b",
     "text": "QLIKE(y², f) = y²/f − log(y²/f) − 1 on annualised variance (Patton 2011: robust to the "
             "noise in a realised-variance proxy); lower is better.",
     "units": "—", "used": "Research, Model Risk", "source": "joint_vol_calibration/stats/forecast.py"},
    {"id": "holm", "term": "Holm", "abbr": "Holm step-down adjustment", "pat": r"\bHolm\b",
     "text": "Holm (1979) step-down adjusted p-values: family-wise error control across the tests "
             "named beside it.",
     "units": "p-value", "used": "Research", "source": "joint_vol_calibration/stats/nulls.py"},
    {"id": "z-vrp", "term": "z_vrp", "abbr": "VRP z-score", "pat": r"\bz_vrp\b",
     "text": "S4's z-score of VRP = VIX − 20-day realised vol (decimal vol, rebuilt as "
             "vix × (1 − 1/fear_premium)): (VRP − its 252-session mean) / its 252-session sd.",
     "units": "sd", "used": "Research", "source": "joint_vol_calibration/signals/signal_engine.py"},
    {"id": "rv21", "term": "RV21", "abbr": "21-session realised vol", "pat": r"\bRV21\b",
     "text": "100·√(252/21 · Σ r²) over 21 daily SPX close-to-close log returns: trailing (known at the "
             "close) or forward (the next 21 sessions).",
     "units": "%", "used": "Overview, Research", "source": "dashboard/desk_market.py"},
    {"id": "regimes", "term": "R0–R2", "abbr": "rule regime labels", "pat": r"\bR[012]\b",
     "text": "The rule's regime labels: R0 long gamma, R1 short gamma, R2 vomma active (VVIX above its "
             "threshold at the close); the backtest reads them lagged one session.",
     "units": "label", "used": "Overview, Research", "source": "joint_vol_calibration/signals/regime_pdv.py"},
    {"id": "vomma-ratio", "term": "vomma P&L ratio", "abbr": None, "pat": r"\bvomma P&amp;L ratio\b",
     "text": "½·vomma·Δσ² / (vega·|Δσ|) = ½·d₁·d₂·|Δσ|/σ for a vol move Δσ: the second-order vol P&L "
             "against the first; a node is flagged convexity-dominated above the monitor's threshold.",
     "units": "dimensionless", "used": "Risk", "source": "joint_vol_calibration/greeks/risk_monitor.py"},
)
TERM_IDS = tuple(t["id"] for t in TERMS)
_PATS = [(t, re.compile(t["pat"])) for t in TERMS]

# elements whose text is never linked; void elements never close
_SKIP = frozenset({"a", "abbr", "dfn", "button", "summary", "th", "h1", "h2", "h3", "h4", "code", "pre",
                   "kbd", "script", "style", "svg", "title", "option", "select", "textarea", "label",
                   "noscript", "dt"})
# a term is linked in running prose only (a paragraph, a list item, a definition),
# never in a tile's value, a label or a table cell
_PROSE = frozenset({"p", "li", "dd"})
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
                   "track", "wbr"})
_TAG = re.compile(r"<!--.*?-->|<(/?)([a-zA-Z][\w-]*)([^>]*)>", re.DOTALL)
_SKIP_ATTR = re.compile(r'\bhidden\b|aria-hidden="true"|class="[^"]*\b(?:sr-only|chip)\b')
_SELF_ATTR = re.compile(r'\bdata-term="([\w-]+)"')


def _link(t: dict, text: str, href: str) -> str:
    inner = f'<abbr title="{_html.escape(t["abbr"])}">{text}</abbr>' if t.get("abbr") else text
    return f'<a class="term" href="{href}#def-{t["id"]}">{inner}</a>'


def link_first_uses(page: str, href: str, skip: Sequence[str] = ()) -> str:
    """``page`` with the first use of each term inside <main> linked to
    ``href``#def-<id> ('' on the page that holds the definitions); terms in
    ``skip`` are left alone."""
    start = page.find("<main")
    end = page.rfind("</main>")
    if start < 0 or end < 0:
        return page
    todo = [(t, p) for t, p in _PATS if t["id"] not in skip]
    out: list[str] = [page[:start]]
    stack: list[tuple[str, bool, str]] = []
    body = page[start:end]
    i = 0
    for m in _TAG.finditer(body):
        text = body[i:m.start()]
        if text and todo and not any(s for _, s, _ in stack) and any(n in _PROSE for n, _, _ in stack):
            own = {o for _, _, o in stack if o}
            if own:                       # inside a term's own definition: not that term
                rest = [(t, p) for t, p in todo if t["id"] in own]
                todo[:] = [(t, p) for t, p in todo if t["id"] not in own]
                text = _link_text(text, todo, href)
                todo.extend(rest)
            else:
                text = _link_text(text, todo, href)
        out.append(text)
        out.append(m.group(0))
        i = m.end()
        if m.group(0).startswith("<!--"):
            continue
        closing, name, attrs = m.group(1) == "/", m.group(2).lower(), m.group(3)
        if closing:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    del stack[k:]
                    break
        elif name not in _VOID and not attrs.rstrip().endswith("/"):
            self_m = _SELF_ATTR.search(attrs)
            stack.append((name, name in _SKIP or bool(_SKIP_ATTR.search(attrs)),
                          self_m.group(1) if self_m else ""))
    out.append(body[i:])
    out.append(page[end:])
    return "".join(out)


def _link_text(text: str, todo: list, href: str) -> str:
    """Link, in one text run, the earliest match of each term still to do
    (non-overlapping, earliest first); the linked terms leave ``todo``."""
    hits = []
    for t, p in todo:
        m = p.search(text)
        if m:
            hits.append((m.start(), m.end(), t))
    if not hits:
        return text
    hits.sort(key=lambda h: (h[0], -h[1]))
    out, last, done = [], 0, set()
    for a, b, t in hits:
        if a < last:
            continue                      # overlaps an earlier term: it waits for its next use
        out += [text[last:a], _link(t, text[a:b], href)]
        last = b
        done.add(t["id"])
    out.append(text[last:])
    todo[:] = [(t, p) for t, p in todo if t["id"] not in done]
    return "".join(out)


def first_uses(page: str) -> dict[str, str]:
    """{term id: its link's href} of the first-use links on a rendered page."""
    return {m.group(2): m.group(1) for m in re.finditer(
        r'<a class="term" href="([^"#]*#def-([\w-]+))">', page)}
