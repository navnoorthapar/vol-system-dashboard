"""page_budget.py: the Surface page's weight against its budget (CI's a11y job).

    python scripts/ci/page_budget.py <site-dir>

surface.html may weigh at most 450,000 bytes at 45 listed expiries, plus 7,000 bytes
for each expiry past 45 (D2): a full Cboe book lists about 45, and each expiry adds a
row to the five heatmaps, the delta grid and the smile's tables. The expiry count is
read from the page itself (its smile's expiry tabs), so a day with more listed
expiries raises the budget with the page instead of failing the job; the daily bot's
deploy is never gated on it (freeze_site.py does not read this). Every other weight
check runs in pytest on synthetic sessions (test_dashboard.py), never on a stored one.
Exits 1 when the page is over its budget (a page without the market surface, so with
no expiry tabs, is held to the base budget).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

BASE_BYTES = 450_000          # at BASE_EXPIRIES listed expiries
BASE_EXPIRIES = 45
PER_EXPIRY = 7_000            # each listed expiry past BASE_EXPIRIES
# the smile's expiry tabs (dashboard/templates/surface.html)
TAB = re.compile(r'<button class="btn[^"]*" type="button" role="tab" aria-selected="(?:true|false)" '
                 r'aria-controls="slices-panel" data-value="\d{4}-\d{2}-\d{2}"')


def budget(n_expiries: int) -> int:
    """The Surface page's byte budget at n listed expiries."""
    return BASE_BYTES + PER_EXPIRY * max(0, n_expiries - BASE_EXPIRIES)


def check(site: Path) -> tuple[bool, str]:
    html = (site / "surface.html").read_bytes()
    n = len(TAB.findall(html.decode("utf-8")))
    cap = budget(n)
    ok = len(html) <= cap
    return ok, (f"surface.html: {len(html):,} bytes at {n} listed expiries, budget {cap:,} "
                f"({BASE_BYTES:,} + {PER_EXPIRY:,} an expiry past {BASE_EXPIRIES}): {'ok' if ok else 'OVER'}")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    ok, msg = check(Path(args[0]))
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
