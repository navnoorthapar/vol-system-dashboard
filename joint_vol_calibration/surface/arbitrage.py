"""
arbitrage.py — static-arbitrage screening of quoted option prices, and the
Cohen-Reisinger-Wang minimal-perturbation repair as a linear programme.

Reference: S. N. Cohen, C. Reisinger, S. Wang, "Detecting and repairing
arbitrage in traded option prices" (2020), arXiv:2008.09454. Reimplemented
here from the paper on scipy.optimize.linprog (HiGHS); the authors' package
(arbitragerepair) is not used because it depends on GPL cvxopt.

Normalisation (CRW section 2.3)
-------------------------------
Every quote is mapped to a call-equivalent price through put-call parity on
its expiry's parity forward F_i and discount D_i (C = P + D_i (F_i - K)), then
normalised: k = K / F_i, c = C / (D_i F_i). Arbitrage-free normalised call
prices satisfy, per expiry, 0 <= c <= 1, c >= (1 - k)+, monotone decreasing
with slope >= -1, convex in k, and are non-decreasing in T at fixed k
(convex order). Each expiry is augmented with the forward itself, the call
struck at 0: (k, c) = (0, 1). For numerical conditioning the LP works in
"forward index points": x = S k, y = S c with S the session spot, so a price
and a tick have their usual size. Bid / ask map through the same parity
shift, so delta_ask = (ask - mid) / (D F) * S and delta_bid likewise.

Constraint rows (A y >= b, each row the price of a test strategy, CRW Defs 1-2)
--------------------------------------------------------------------------------
  C1  outright          y_last >= 0 (last strike of each expiry)
  C2  vertical spread   y_{j-1} - y_j >= 0 (j = 1 uses the forward point),
                        and y_1 >= S - x_1 (VS_{1,0} <= 1)
  C3  vertical butterfly  w_l y_{j-1} - y_j + w_r y_{j+1} >= 0
                        (w_l = h_r/(h_l+h_r), w_r = h_l/(h_l+h_r))
  C4  calendar spread   y_later(x) - y_earlier(x) >= 0 at identical x
  C5  calendar vertical spread: for each earlier quote, the later-expiry
                        quote with the nearest lower x: y_later >= y_earlier
  C6  calendar butterflies (CRW C6.1 a/b/c, absolute location convexity) and
                        the nearest-neighbour chord condition
                        y_earlier(x) <= chord of the later expiry at x
                        (a C6.2-type test butterfly)
Every row is a necessary no-arbitrage condition. Calendar rows (C4-C6) are
built between CONSECUTIVE expiries only; the full CRW set (all expiry pairs,
all C6.2 pairs) is not enumerated, so zero violations certify the absence of
these test-strategy arbitrages, not CRW-complete absence of static arbitrage.
Calendar rows assume proportional (deterministic) dividends, under which c is
non-decreasing in T at fixed k = K/F_T.

Mid vs executable
-----------------
A row violated at mids is an arbitrage in the reference (mid) prices. A row is
EXECUTABLE when buying its positive-weight legs at the ask and selling its
negative-weight legs at the bid still costs less than the row bound (CRW eq.
after (11)); the forward and the bond are taken as tradable without spread.

Repair LP (CRW eq. (10), the l1-BA objective)
---------------------------------------------
    min sum_j t_j   s.t.  t_j >= max(-e_j - db_j + d0, -d0/db_j e_j,
                                     d0/da_j e_j, e_j - da_j + d0)
                          A (y + e) >= b
Moving a price to its bid or ask costs d0, inside the spread the marginal cost
is d0/da_j (or d0/db_j), outside it is 1. CRW take d0 = (1/N) ^ min_j
min(da_j, db_j) in normalised units; this code uses the scale-free
d0 = min_j min(da_j, db_j) / N, which keeps their stated property (moving
every price to its spread edge costs no more than moving one price outside
it by the smallest half-spread). The quotes the l1-BA solution moves outside
[bid, ask] ("effectively perturbed") are the attribution of the arbitrage;
they are not by themselves a proof that one exists, because the l1-BA
objective can trade a small move outside one spread against several moves
inside others.

Executable-arbitrage certificate (bid / ask box LP)
---------------------------------------------------
    min sum_j o_j   s.t.  o_j >= e_j - da_j,  o_j >= -db_j - e_j,  o_j >= 0,
                          A (y + e) >= b
is the least total distance (index points) the prices must move OUTSIDE
their spreads to satisfy every row. It is 0 exactly when some price vector
inside every [bid, ask] satisfies all rows; when it is > 0, by Farkas'
lemma a non-negative combination of rows is a test portfolio that can be
bought at the asks / sold at the bids for less than its bound, i.e. an
executable static arbitrage among these rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROW_TYPES = ("C1_outright", "C2_vertical_spread", "C2_vertical_bound",
             "C3_vertical_butterfly", "C4_calendar_spread",
             "C5_calendar_vertical_spread", "C6_calendar_butterfly")

_EPS_X = 1e-12        # relative tolerance for "identical normalised strike"
TICK = 0.05           # SPX option tick (index points): mid violations below it are rounding
_MIN_DELTA = 1e-4     # floor of a half-spread in forward index points
_CERT_TOL = 1e-7      # box-LP distance (index points) below which the box is feasible


@dataclass
class ArbitrageProblem:
    """The normalised quote set and its constraint rows.

    y, x        : (N,) call-equivalent prices and strikes in forward index points
    delta_ask   : (N,) ask - mid,   delta_bid : (N,) mid - bid  (same units)
    A, b        : sparse (R, N) rows and bounds, A y >= b at no arbitrage
    row_type    : (R,) index into ROW_TYPES
    row_quotes  : list of the quote indices each row involves
    """
    y: np.ndarray
    x: np.ndarray
    delta_ask: np.ndarray
    delta_bid: np.ndarray
    expiry_index: np.ndarray
    A: sparse.csr_matrix
    b: np.ndarray
    row_type: np.ndarray
    scale: float
    price_to_quote: np.ndarray     # multiply a y-perturbation by this for $ (quote units)


@dataclass
class ArbitrageReport:
    """Screening and repair results (see module docstring)."""
    n_quotes: int
    n_rows: dict
    mid_violations: dict
    executable_violations: dict
    max_mid_violation: dict          # largest shortfall per row type (index points)
    quote_flags: pd.DataFrame        # per quote: arb_mid, arb_exec, repair_*, outside_spread
    repair: dict = field(default_factory=dict)
    mid_violations_gt_tick: dict = field(default_factory=dict)   # shortfall > TICK
    violations: list = field(default_factory=list)   # top violating rows (mid), detail
    certificate: dict = field(default_factory=dict)  # bid / ask box LP (outside_spread_lp)
    calendar_tv: dict = field(default_factory=dict)  # calendar_total_variance_mid

    def summary(self) -> dict:
        return {
            "n_quotes": self.n_quotes,
            "n_rows": self.n_rows,
            "mid_violations": self.mid_violations,
            "mid_violations_gt_tick": self.mid_violations_gt_tick,
            "executable_violations": self.executable_violations,
            "max_mid_violation_index_points": self.max_mid_violation,
            "n_quotes_in_mid_violation": int(self.quote_flags["arb_mid"].sum()),
            "n_quotes_in_executable_violation": int(self.quote_flags["arb_exec"].sum()),
            "executable_arbitrage_certificate": self.certificate,
            "calendar_total_variance_mid": self.calendar_tv,
            "repair": self.repair,
        }


def _call_equivalent(quotes: pd.DataFrame, scale: float) -> dict:
    """Normalised call-equivalent mid/bid/ask (forward index points)."""
    F = quotes["forward"].to_numpy(float)
    D = quotes["discount"].to_numpy(float)
    K = quotes["strike"].to_numpy(float)
    is_put = (quotes["right"] == "P").to_numpy()
    shift = np.where(is_put, D * (F - K), 0.0)        # C = P + D (F - K)
    norm = scale / (D * F)
    mid = (quotes["mid"].to_numpy(float) + shift) * norm
    bid = (quotes["bid"].to_numpy(float) + shift) * norm
    ask = (quotes["ask"].to_numpy(float) + shift) * norm
    return {"x": K / F * scale, "y": mid, "y_bid": bid, "y_ask": ask, "norm": norm}


def build_problem(quotes: pd.DataFrame, scale: float | None = None) -> ArbitrageProblem:
    """Constraint rows for ``quotes`` (a quotes.SessionQuotes.quotes frame or any
    frame with expiry, T, strike, right, forward, discount, bid, ask, mid).

    Quotes must hold at most one line per (expiry, strike); the scoring
    universe (OTM only) does.
    """
    q = quotes.reset_index(drop=True)
    if scale is None:
        scale = float(np.median(q["forward"]))
    ce = _call_equivalent(q, scale)
    x, y = ce["x"], ce["y"]
    n = len(q)
    exp_T = q.groupby("expiry", sort=False)["T"].first().sort_values()
    expiries = list(exp_T.index)
    e_idx = q["expiry"].map({e: i for i, e in enumerate(expiries)}).to_numpy()

    rows, cols, vals, bnds, types = [], [], [], [], []
    r = 0

    def add(entries, bound, typ):
        nonlocal r
        for c_, v_ in entries:
            rows.append(r)
            cols.append(c_)
            vals.append(v_)
        bnds.append(bound)
        types.append(typ)
        r += 1

    per_exp = []
    for i in range(len(expiries)):
        idx = np.where(e_idx == i)[0]
        idx = idx[np.argsort(x[idx], kind="mergesort")]
        per_exp.append(idx)
        m = len(idx)
        if m == 0:
            continue
        # C1: outright at the last strike
        add([(idx[-1], 1.0)], 0.0, 0)
        # C2: forward point (x=0, y=scale) -> first strike, and adjacent pairs
        add([(idx[0], -1.0)], -scale, 1)                       # y_1 <= scale
        add([(idx[0], 1.0)], scale - x[idx[0]], 2)             # VS_{1,0} <= 1
        for j in range(1, m):
            add([(idx[j - 1], 1.0), (idx[j], -1.0)], 0.0, 1)
        # C3: butterflies, j = 0 uses the forward point on its left
        for j in range(m - 1):
            xl = 0.0 if j == 0 else x[idx[j - 1]]
            xm, xr = x[idx[j]], x[idx[j + 1]]
            hl, hr = xm - xl, xr - xm
            wl, wr = hr / (hl + hr), hl / (hl + hr)
            ent = [(idx[j], -1.0), (idx[j + 1], wr)]
            bound = 0.0
            if j == 0:
                bound = -wl * scale                            # the forward point's y
            else:
                ent.append((idx[j - 1], wl))
            add(ent, bound, 3)

    # Calendar rows between consecutive expiries
    for i in range(len(expiries) - 1):
        ea, eb = per_exp[i], per_exp[i + 1]                    # earlier, later
        if len(ea) == 0 or len(eb) == 0:
            continue
        xa, xb = x[ea], x[eb]
        # C4 / C5: nearest later quote at or below each earlier strike
        pos = np.searchsorted(xb, xa, side="right") - 1        # last later x <= xa
        for jj, p in enumerate(pos):
            if p < 0:
                # only the forward point lies below: y_later(0) = scale >= y_a holds by C2
                continue
            same = abs(xb[p] - xa[jj]) <= _EPS_X * max(1.0, abs(xa[jj]))
            add([(eb[p], 1.0), (ea[jj], -1.0)], 0.0, 4 if same else 5)
        # C6 chord: earlier strike strictly inside two later neighbours
        for jj in range(len(ea)):
            p = int(np.searchsorted(xb, xa[jj], side="left"))   # first later x >= xa
            if p >= len(xb):
                continue
            if abs(xb[p] - xa[jj]) <= _EPS_X * max(1.0, abs(xa[jj])):
                continue                                         # covered by C4
            xr = xb[p]
            if p == 0:
                xl, il = 0.0, None
            else:
                xl, il = xb[p - 1], eb[p - 1]
            lam = (xr - xa[jj]) / (xr - xl)                      # weight of the left point
            ent = [(ea[jj], -1.0), (eb[p], 1.0 - lam)]
            bound = 0.0
            if il is None:
                bound = -lam * scale
            else:
                ent.append((il, lam))
            add(ent, bound, 6)
        # C6.1 (absolute location convexity) a / b / c
        for jj in range(len(ea)):
            xlo = 0.0 if jj == 0 else xa[jj - 1]
            inside = eb[(xb > xlo) & (xb < xa[jj])]
            for l_ in inside:
                # (a) middle = earlier j, left = later l, right = earlier j+1
                if jj + 1 < len(ea):
                    hl, hr = xa[jj] - x[l_], xa[jj + 1] - xa[jj]
                    add([(l_, hr / (hl + hr)), (ea[jj], -1.0),
                         (ea[jj + 1], hl / (hl + hr))], 0.0, 6)
                # (b) middle = earlier j-1, left = earlier j-2 (or forward), right = later l
                if jj >= 1:
                    xm = xa[jj - 1]
                    xl2 = 0.0 if jj == 1 else xa[jj - 2]
                    hl, hr = xm - xl2, x[l_] - xm
                    ent = [(ea[jj - 1], -1.0), (l_, hl / (hl + hr))]
                    bound = 0.0
                    if jj == 1:
                        bound = -(hr / (hl + hr)) * scale
                    else:
                        ent.append((ea[jj - 2], hr / (hl + hr)))
                    add(ent, bound, 6)
        # (c) later strikes beyond the last earlier strike
        if len(ea) >= 2:
            beyond = eb[xb > xa[-1]]
            xm, xl2 = xa[-1], xa[-2]
            for l_ in beyond:
                hl, hr = xm - xl2, x[l_] - xm
                add([(ea[-2], hr / (hl + hr)), (ea[-1], -1.0), (l_, hl / (hl + hr))], 0.0, 6)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    D = q["discount"].to_numpy(float)
    F = q["forward"].to_numpy(float)
    return ArbitrageProblem(
        y=y, x=x,
        delta_ask=np.maximum(ce["y_ask"] - y, 0.0),
        delta_bid=np.maximum(y - ce["y_bid"], 0.0),
        expiry_index=e_idx,
        A=A, b=np.asarray(bnds, dtype=float), row_type=np.asarray(types, dtype=int),
        scale=scale, price_to_quote=D * F / scale,
    )


def _executable_value(P: ArbitrageProblem) -> np.ndarray:
    """Row value when positive legs are bought at the ask and negative legs
    sold at the bid (the cheapest way to BUY the test portfolio)."""
    A = P.A.tocoo()
    v = A.data
    leg = np.where(v >= 0, P.y[A.col] + P.delta_ask[A.col], P.y[A.col] - P.delta_bid[A.col])
    out = np.zeros(P.A.shape[0])
    np.add.at(out, A.row, v * leg)
    return out


def repair_lp(P: ArbitrageProblem, delta0: float | None = None) -> dict:
    """Solve the l1-BA repair LP (CRW eq. (10)) with HiGHS.

    Returns {'status', 'message', 'eps' (N, index points), 'objective',
    'delta0'}. eps is None when the solver fails.
    """
    n = len(P.y)
    da = np.maximum(P.delta_ask, _MIN_DELTA)
    dbid = np.maximum(P.delta_bid, _MIN_DELTA)
    if delta0 is None:
        delta0 = float(min(da.min(), dbid.min()) / max(n, 1))
    eye = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((P.A.shape[0], n))
    # rows of A_ub [eps, t] <= b_ub
    blocks = [
        (sparse.hstack([-eye, -eye]), dbid - delta0),                 # -e - t <= db - d0
        (sparse.hstack([eye, -eye]), da - delta0),                    #  e - t <= da - d0
        (sparse.hstack([sparse.diags(-delta0 / dbid), -eye]), np.zeros(n)),
        (sparse.hstack([sparse.diags(delta0 / da), -eye]), np.zeros(n)),
        (sparse.hstack([-P.A, Z]), P.A @ P.y - P.b),              # -A e <= A y - b
    ]
    A_ub = sparse.vstack([blk for blk, _ in blocks], format="csr")
    b_ub = np.concatenate([rhs for _, rhs in blocks])
    cost = np.concatenate([np.zeros(n), np.ones(n)])
    bounds = [(None, None)] * n + [(0.0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    ok = res.status == 0                       # fun and x are set when the LP solved
    return {"status": int(res.status), "message": str(res.message), "delta0": delta0,
            "objective": float(cast(float, res.fun)) if ok else None,
            "eps": cast(np.ndarray, res.x)[:n].copy() if ok else None}


def outside_spread_lp(P: ArbitrageProblem) -> dict:
    """The executable-arbitrage certificate (module docstring): the least
    total distance outside the bid / ask spreads that satisfies every row.

    Returns {'status', 'message', 'distance' (index points, None on solver
    failure), 'n_outside' (quotes this solution moves outside their spread)}.
    """
    n = len(P.y)
    da, dbid = P.delta_ask, P.delta_bid
    eye = sparse.identity(n, format="csr")
    Z = sparse.csr_matrix((P.A.shape[0], n))
    A_ub = sparse.vstack([sparse.hstack([eye, -eye]),                # e - o <= da
                          sparse.hstack([-eye, -eye]),               # -e - o <= db
                          sparse.hstack([-P.A, Z])], format="csr")
    b_ub = np.concatenate([da, dbid, P.A @ P.y - P.b])
    cost = np.concatenate([np.zeros(n), np.ones(n)])
    bounds = [(None, None)] * n + [(0.0, None)] * n
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": int(res.status), "message": str(res.message), "distance": None,
                "n_outside": None}
    o = cast(np.ndarray, res.x)[n:]
    return {"status": 0, "message": str(res.message), "distance": float(cast(float, res.fun)),
            "n_outside": int((o > 1e-9).sum())}


def calendar_total_variance_mid(quotes: pd.DataFrame, tol: float = 1e-8) -> dict:
    """Diagnostic calendar check in total implied variance at mids.

    For each pair of consecutive expiries, every earlier-expiry quote whose k
    lies inside the later expiry's quoted k-range is compared with the later
    expiry's mid total variance w = iv^2 T linearly interpolated in k: a
    crossing is w_earlier(k) > w_later(k) + tol. Linear interpolation is not
    a bound on the later smile, so this is a diagnostic of the quoted surface,
    not a certificate; the certified calendar rows are C4-C6.
    """
    q = quotes
    exp_T = q.groupby("expiry", sort=False)["T"].first().sort_values()
    n_pts = n_cross = 0
    pairs: list[dict[str, Any]] = []
    for e1, e2 in zip(exp_T.index[:-1], exp_T.index[1:]):
        a = q[q["expiry"] == e1]
        b = q[q["expiry"] == e2].sort_values("k")
        kb = b["k"].to_numpy(float)
        wb = (b["mid_iv"].to_numpy(float) ** 2) * float(exp_T[e2])
        ka = a["k"].to_numpy(float)
        inside = (ka >= kb.min()) & (ka <= kb.max())
        if not inside.any():
            continue
        wa = (a["mid_iv"].to_numpy(float)[inside] ** 2) * float(exp_T[e1])
        d = np.interp(ka[inside], kb, wb) - wa
        cross = d < -tol
        n_pts += int(inside.sum())
        n_cross += int(cross.sum())
        if cross.any():
            pairs.append({"earlier": str(e1), "later": str(e2), "n_points": int(inside.sum()),
                          "n_crossing": int(cross.sum()), "min_dw": float(d.min())})
    pairs.sort(key=lambda r: r["min_dw"])
    return {"n_points": n_pts, "n_crossing": n_cross,
            "n_pairs_crossing": len(pairs), "worst_pairs": pairs[:10],
            "tolerance_total_variance": tol,
            "method": ("earlier mid total variance vs the later expiry's mid total variance "
                       "interpolated linearly in k (diagnostic, not a certificate)")}


def screen(quotes: pd.DataFrame, *, repair: bool = True, tol: float = 1e-9,
           n_detail: int = 25) -> ArbitrageReport:
    """Screen ``quotes`` for static arbitrage and (optionally) repair them.

    Always computed: row violations at mids and executable per row, the
    bid / ask box certificate (outside_spread_lp) and, when the frame has k
    and mid_iv, the total-variance calendar diagnostic. ``repair``: also the
    CRW l1-BA repair and its per-quote attribution.

    tol : a row counts as violated when its shortfall exceeds tol x scale
          (index points; 1e-9 x 7,000 = 7e-6, far below a 0.05 tick)
    """
    q = quotes.reset_index(drop=True)
    P = build_problem(q)
    slack = P.A @ P.y - P.b
    thr = tol * P.scale
    viol = slack < -thr
    exe = (_executable_value(P) - P.b) < -thr
    names = np.asarray(ROW_TYPES)[P.row_type] if len(P.row_type) else np.array([], dtype=str)
    keys = list(ROW_TYPES)
    n_rows = {k: int((names == k).sum()) for k in keys}
    mid_v = {k: int((viol & (names == k)).sum()) for k in keys}
    exe_v = {k: int((exe & (names == k)).sum()) for k in keys}
    max_v = {k: (float(-slack[viol & (names == k)].min()) if mid_v[k] else 0.0) for k in keys}
    big = slack < -TICK
    mid_big = {k: int((big & (names == k)).sum()) for k in keys}

    A = P.A.tocsr()
    arb_mid = np.zeros(len(q), dtype=bool)
    arb_exec = np.zeros(len(q), dtype=bool)
    for rr in np.where(viol)[0]:
        arb_mid[A.indices[A.indptr[rr]:A.indptr[rr + 1]]] = True
    for rr in np.where(exe)[0]:
        arb_exec[A.indices[A.indptr[rr]:A.indptr[rr + 1]]] = True

    flags = pd.DataFrame({"arb_mid": arb_mid, "arb_exec": arb_exec})
    detail = []
    order = np.argsort(slack)                     # most negative first
    for rr in order[:n_detail]:
        if not viol[rr]:
            break
        legs = A.indices[A.indptr[rr]:A.indptr[rr + 1]]
        w = A.data[A.indptr[rr]:A.indptr[rr + 1]]
        detail.append({
            "type": names[rr],
            "shortfall_index_points": float(-slack[rr]),
            "executable": bool(exe[rr]),
            "legs": [{"expiry": str(q.loc[int(j), "expiry"]),
                      "strike": float(cast(float, q.loc[int(j), "strike"])),
                      "right": str(q.loc[int(j), "right"]), "weight": float(wj)}
                     for j, wj in zip(legs, w)],
        })

    rep = {}
    if repair:
        sol = repair_lp(P)
        rep = {"status": sol["status"], "message": sol["message"],
               "delta0_index_points": sol["delta0"]}
        if sol["eps"] is not None:
            eps = sol["eps"]
            moved = np.abs(eps) > 1e-6
            outside = (eps > P.delta_ask + 1e-6) | (eps < -P.delta_bid - 1e-6)
            post = (P.A @ (P.y + eps) - P.b)
            dq = eps * P.price_to_quote                  # option-price units ($ index pts)
            half = 0.5 * (q["ask"].to_numpy(float) - q["bid"].to_numpy(float))
            with np.errstate(divide="ignore", invalid="ignore"):
                dhs = np.where(half > 0, np.abs(dq) / half, np.nan)
            flags["repair_dprice"] = dq
            flags["repair_outside_spread"] = outside
            rep.update({
                "objective": sol["objective"],
                "n_perturbed": int(moved.sum()),
                "n_perturbed_outside_spread": int(outside.sum()),
                "max_abs_dprice": float(np.abs(dq).max()) if len(dq) else 0.0,
                "max_abs_dprice_half_spreads": (float(np.nanmax(dhs[moved])) if moved.any() else 0.0),
                "sum_abs_dprice": float(np.abs(dq).sum()),
                "max_row_shortfall_after": float(max(0.0, -post.min())) if len(post) else 0.0,
            })
        else:
            flags["repair_dprice"] = np.nan
            flags["repair_outside_spread"] = False
    box = outside_spread_lp(P)
    single_row = bool(exe.any())
    dist = box["distance"]
    certificate = {
        **box,
        "bid_ask_box_feasible": (None if dist is None else bool(dist <= _CERT_TOL)),
        "any_single_row_executable": single_row,
        "executable_arbitrage": bool(single_row or (dist is not None and dist > _CERT_TOL)),
        "tolerance_index_points": _CERT_TOL,
    }
    return ArbitrageReport(n_quotes=len(q), n_rows=n_rows, mid_violations=mid_v,
                           mid_violations_gt_tick=mid_big,
                           executable_violations=exe_v, max_mid_violation=max_v,
                           quote_flags=flags, repair=rep, violations=detail,
                           certificate=certificate,
                           calendar_tv=(calendar_total_variance_mid(q)
                                        if {"k", "mid_iv"} <= set(q.columns) else {}))
