"""Write desk.css's heatmap colour classes from desk_format.HEAT_CLASSES.

A heatmap cell's colour is one of 101 steps of cividis (q0 … q100) or of the
diverging ramp (d0 … d100), each a class with its background and the text colour
that contrasts with it. desk.css holds the block between the two marker lines;
this script rewrites it, and test_dashboard checks that it is what the ramps give.

    python dashboard/heat_css.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import desk_format  # noqa: E402  (dashboard/, after the path setup)

CSS = HERE / "static" / "desk.css"
BEGIN = "/* heat classes: written by dashboard/heat_css.py from desk_format.HEAT_CLASSES */\n"
END = "/* end of heat classes */\n"


def block() -> str:
    return BEGIN + desk_format.heat_class_css() + END


def main() -> int:
    css = CSS.read_text(encoding="utf-8")
    if BEGIN not in css or END not in css:
        print(f"{CSS}: no heat-class markers", file=sys.stderr)
        return 1
    head, rest = css.split(BEGIN, 1)
    tail = rest.split(END, 1)[1]
    CSS.write_text(head + block() + tail, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
