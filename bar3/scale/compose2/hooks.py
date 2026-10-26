"""Round 2's changes to the climb and the oracle arm (COMPOSE2.md §2), each behind its own flag and called from the
asserted replacements in spcore_c2.py and oracle_c2.py (make_copies.py). With every flag off each hook returns at
once and changes nothing.

- SUBGOAL (`subgoal_scores`, `subgoal_deep`, `tie_rank`): bar 3's subgoal score over the candidates posed on the
  target's own root with answers in A (the near misses' values, the projections low and high), on the target's
  probes and the acceptor's rows; the subgoal it singles out gets one deeper probe and wins ties.
- LAWNEAR (`law_near`): the look's held states whose A, X, Y or S passes the depth-1 law view join the near misses.
- ROWCHECK (`row_check`): aliases and duplicates by the candidate's own acceptor (ROWCHECK.md).
- TIEBREAK (`tie_candidate`): the candidate the score singles out against the rest of the way from the practised
  subgoal's value wins ties (PICKS.md's tiescore).
- RESTRICT_ROWS (`share_rows`): the rows a restriction's share and pattern are measured on.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from bar3.scale.compose2 import subgoal_c2 as SG

SUB_DEEP = 10_000_000                           # SUBGOAL.md's deeper probe
LAW_NEAR = 4                                    # LAWNEAR: at most this many states a look


def ints(d):
    """Digits [n, P] (least significant first, a torch tensor) as Python ints."""
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in d.tolist()]


# ---------------------------------------------------------------------------------------------------- SUBGOAL
def _acc_root(target, climb, rows):
    """The acceptor's rows for the target (OracleAcceptor's construction: 60 rows a verify setting, seed 11)."""
    from bar3.scale.compose2.oracle_c2 import OracleRoot, prefix_values  # noqa: F401
    if getattr(target, "_score_root", None) is None:
        target._score_root = OracleRoot("tgt@XN", "target", climb.verify, 60, rows, climb.device, 11)
    return target._score_root


def subgoal_scores(climb, rows, roots):
    """SUBGOAL: score the candidates posed on the target's root; target.subgoal is the one singled out, or None."""
    t = climb.target
    if not getattr(climb, "subgoal_on", False) or t is None:
        return
    from bar3.scale.compose2.oracle_c2 import prefix_values
    t.subgoal = None
    tr = next((r for r in roots if getattr(r, "key", None) == "tgt@XN"), None)
    if tr is None:
        return
    ar = _acc_root(t, climb, rows)
    sts = list(climb.settings) + list(climb.verify)
    draws = list(tr.draws) + list(ar.draws)
    xs = [x for d in draws for x in d.xs]
    Ns = [n for d in draws for n in d.Ns]
    ys = [y for d in draws for y in d.ys]
    caps = [10 ** st.P for st, d in zip(sts, draws) for _ in range(d.n)]
    srows = SG.rows_from(xs, Ns, ys, caps)
    keys = list(zip(xs, Ns))
    cands = {}
    for g in tr.goal_list:
        if not all("A" in tt for tt in g.targets):
            continue
        if g.name in ("low", "high"):
            vals = [y % 10 if g.name == "low" else y // 10 for y in ys]
        elif g.name in t.values:
            prog, ir = t.values[g.name]
            vals = []
            for st, rt in zip(sts, list(tr.roots) + list(ar.roots)):
                vals += ints(prefix_values(prog, st, rt, climb.lib)[ir])
        else:
            continue
        d = SG.first_rows(keys, vals)
        cands[g.name] = [d[k] for k in srows.keys]
    sc = SG.score(cands, srows)
    t.subgoal = SG.singled_out(sc)
    if t.subgoal:
        t.subgoals_seen.append(t.subgoal)
    top = sorted(sc.items(), key=lambda kv: kv[1]["key"], reverse=True)[:6]
    climb.records.append(dict(task="subgoal", level=climb.level, subgoal=t.subgoal, depths=SG.law_depths(srows),
                              rows=len(srows.keys), top=[(n, v["key"], v["law"], v["how"]) for n, v in top]))


def subgoal_deep(climb, rows, deadline):
    """SUBGOAL: one deeper probe of the target's root when the subgoal is still unsolved; with REST_DEEP, when the
    subgoal is solved, of the rest of the way from its value posed with it (PREREG_ADDENDUM2.md); a log line, or
    None."""
    t = climb.target
    if not getattr(climb, "subgoal_on", False) or t is None or getattr(climb, "_sub_deep_level", -1) == climb.level:
        return None
    climb._sub_deep_level = climb.level
    if not getattr(t, "subgoal", None):
        return None
    name = f"{t.subgoal}/tgt@XN"
    r = climb.results.get(name)
    if r is not None and r["found_cost"] is not None:
        return rest_deep(climb, rows, deadline)
    root = next((x for x in climb._level_roots if x.key == "tgt@XN"), None)
    if root is None:
        return None
    old = climb.deep_budget
    climb.deep_budget = SUB_DEEP
    try:
        rec = climb.probe([root], rows, deadline, deep=True)
    finally:
        climb.deep_budget = old
    rec["subgoal_deep"] = name
    climb.records.append(rec)
    return f"L{climb.level} subgoal deep probe {name} at {SUB_DEEP} states: {rec['laws'].get(t.subgoal)}"


def rest_deep(climb, rows, deadline):
    """REST_DEEP: the rest of the way from the subgoal's value, posed with it this level and left unsolved by the
    probes, gets the subgoal's deeper probe over the composition vocabulary (every routine, the focus off); a log
    line, or None."""
    t = climb.target
    if not getattr(climb, "rest_deep", False) or not t.subgoal.startswith("val."):
        return None
    key = f"rest.{t.subgoal[4:]}@XN"
    rr = next((x for x in climb._level_roots if x.key == key), None)
    if rr is None:
        return None
    r = climb.results.get(f"target/{key}")
    if r is not None and r["found_cost"] is not None:
        return None
    old = climb.deep_budget, climb.deep_focus
    climb.deep_budget, climb.deep_focus = SUB_DEEP, False
    try:
        rec = climb.probe([rr], rows, deadline, deep=True)
    finally:
        climb.deep_budget, climb.deep_focus = old
    rec["rest_deep"] = key
    climb.records.append(rec)
    return f"L{climb.level} rest deep probe target/{key} at {SUB_DEEP} states: {rec['laws'].get(f'target/{key}')}"


# ---------------------------------------------------------------------------------------------------- ties
def tie_candidate(climb, status):
    """TIEBREAK: the learnable candidate the score singles out against the rest of the way from the latest practised
    subgoal's value (the rest root posed this level), or None."""
    t = climb.target
    if not getattr(climb, "tiebreak", False) or t is None:
        return None
    from integration.lib.libcore import RI, run_items
    for sub in reversed(getattr(t, "subgoals_seen", [])):
        if f"{sub}/tgt@XN" not in climb.practiced or not sub.startswith("val."):
            continue
        rr = next((x for x in climb._level_roots if x.key == f"rest.{sub[4:]}@XN"), None)
        if rr is None:
            continue
        xs = [v for st, rt in zip(climb.settings, rr.roots) for v in ints(rt.regs[0, RI["X"]].T.long())]
        Ns = [n for d in rr.draws for n in d.Ns]
        ys = [y for d in rr.draws for y in d.ys]
        caps = [10 ** st.P for st, d in zip(climb.settings, rr.draws) for _ in range(d.n)]
        srows = SG.rows_from(xs, Ns, ys, caps)
        keys = list(zip(xs, Ns))
        cands = {}
        for n in climb.learnable(status):
            root = climb.results[n]["root"]
            if not set(getattr(root, "binding", ())) <= {"X", "N"}:
                continue
            lib = {} if climb.results[n]["how"] == "primitives" else climb.lib
            vals = []
            for st, rt in zip(climb.settings, rr.roots):
                b = rt.copy()
                run_items(st, b, status[n]["prog"], lib)
                vals += ints(b.regs[0, RI["A"]].T.long())
            d = SG.first_rows(keys, vals)
            cands[n] = [d[k] for k in srows.keys]
        sc = SG.score(cands, srows)
        tie = SG.singled_out(sc)
        top = sorted(sc.items(), key=lambda kv: kv[1]["key"], reverse=True)[:6]
        climb.records.append(dict(task="tiebreak", level=climb.level, rest=rr.key, tie=tie,
                                  depths=SG.law_depths(srows), top=[(n, v["key"], v["law"], v["how"]) for n, v in top]))
        return tie
    return None


def ties(climb, status):
    """(the subgoal's pool name, the tie-break's), each None when its flag is off or it singles out nothing; computed
    once a level."""
    if getattr(climb, "_ties_level", None) == climb.level:
        return climb._ties
    sub = None
    if getattr(climb, "subgoal_on", False) and climb.target is not None and getattr(climb.target, "subgoal", None):
        sub = f"{climb.target.subgoal}/tgt@XN"
    climb._ties = (sub, tie_candidate(climb, status))
    climb._ties_level = climb.level
    return climb._ties


def tie_rank(climb, status, names):
    """A sort key's middle part: () with both flags off (pool order breaks ties, as before); else whether names hold
    the subgoal, then the tie-break's pick (holding it first)."""
    if not (getattr(climb, "subgoal_on", False) or getattr(climb, "tiebreak", False)):
        return ()
    sub, tie = ties(climb, status)
    return (sub not in names, tie not in names)


# ---------------------------------------------------------------------------------------------------- ROWCHECK
def row_check(climb, name, s, learn_rc, status):
    """ROWCHECK: ("alias", token), ("duplicate", name) or ("learnable", None) for a learnable candidate whose best
    program has signature s, by the candidate's own acceptor (its rows at the verify widths)."""
    acc = climb._accs.get(tuple(name.split("/", 1)))
    for tok in climb.tokens_sig[3].get(s, []):
        if acc is not None and acc.many([(tok,)], climb.lib)[0]:
            return "alias", tok
    for m in learn_rc.get(s, []):
        if acc is not None and acc.many([status[m]["prog"]], climb.lib)[0]:
            return "duplicate", m
    learn_rc.setdefault(s, []).append(name)
    return "learnable", None


# ---------------------------------------------------------------------------------------------------- LAWNEAR
def _law1(G, X, N, T, C):
    """Which rows of G [S, n] (int64) give T on every column by one law of the book over g, X and N, using g, every
    value in [0, C): subgoal_c2.laws at depth 1, on the GPU."""
    S = G.shape[0]
    hit = torch.zeros(S, dtype=torch.bool, device=G.device)

    def test(W):
        nonlocal hit
        hit |= ((W >= 0) & (W < C) & (W == T)).all(1)
    one = torch.ones_like(G)
    for a, b in ((G, X), (X, G), (G, N), (N, G), (G, G)):
        a, b = a * one, b * one
        nz = b != 0
        bb = torch.where(nz, b, one)
        for W in (a + b, (a - b).clamp(min=0), (a - b).abs(), a * b,
                  torch.where(nz, torch.div(a, bb, rounding_mode="floor"), -one),
                  torch.where(nz, torch.remainder(a, bb), -one),
                  torch.minimum(a, b), torch.maximum(a, b), (a < b).long(), (a <= b).long(), (a == b).long(),
                  torch.gcd(a, b)):
            test(W)
    test(G + 1)
    test((G - 1).clamp(min=0))
    return hit & ~(G == T).all(1)


def law_near(target, climb, root, levels, parents, classes, goal):
    """LAWNEAR: [(fit, cost, program)] for up to LAW_NEAR held states of the look (cheapest first, distinct values)
    whose A, X, Y or S passes the depth-1 law view on the target's probes."""
    if not getattr(target, "lawnear", False):
        return []
    from integration.lib.libcore import RI, _OneGoal, programs
    from bar3.scale.compose2.spcore_c2 import _Res
    dev = root.roots[0].regs.device
    X = torch.tensor([x for d in root.draws for x in d.xs], device=dev)
    N = torch.tensor([n for d in root.draws for n in d.Ns], device=dev)
    T = torch.tensor([y for d in root.draws for y in d.ys], device=dev)
    C = torch.tensor([10 ** st.P for st, d in zip(climb.settings, root.draws) for _ in range(d.n)], device=dev)
    out, seen = [], set()
    for d, lv in enumerate(levels):
        if lv is None or d == 0 or lv[0].S == 0:
            continue
        for r in ("A", "X", "Y", "S"):
            G = torch.cat([(b.regs[:, RI[r]].long() * (10 ** torch.arange(b.regs.shape[2], device=dev))[None, :, None]
                            ).sum(1) for b in lv], 1)
            for i in torch.nonzero(_law1(G, X, N, T, C)).flatten().tolist():
                sig = tuple(G[i].tolist())
                if sig in seen:
                    continue
                seen.add(sig)
                pr = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
                if pr:
                    f = int(goal.fits([b.take(torch.tensor([i], device=dev)) for b in lv])[0])
                    out.append((f, d, pr[0], r))
                if len(out) >= LAW_NEAR:
                    return out
    return out


# ---------------------------------------------------------------------------------------------------- RESTRICT_ROWS
def share_rows(target, climb, rows):
    """RESTRICT_ROWS: the climb's rows' top parts at the target's last width, the rows a predicate's share and
    pattern are measured on (instead of the target's last draw)."""
    if getattr(target, "_share", None) is None:
        from bar3.scale.compose2.oracle_c2 import top_parts, value, width_for
        Nd, Xd = rows
        w = width_for("double", climb.settings[-1])
        Nt, xt = top_parts(Nd, Xd, w)
        target._share = SimpleNamespace(xs=value(xt), Ns=value(Nt))
    return target._share
