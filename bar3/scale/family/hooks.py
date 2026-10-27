"""FAMILY.md §2: written by bar3/scale/family/make_copies7.py from bar3/scale/compose2/hooks.py, its
module paths moved here, round 3's 4 asserted replacements, round 4's slicing source, round 5's 3
replacements and xon_term, round 6's 2 and its reading source, then this study's 5 replacements and
its windows (SUBTERMS). The rest of this docstring is the original's.

Round 2's changes to the climb and the oracle arm (COMPOSE2.md §2), each behind its own flag and called from the
asserted replacements in spcore_f.py and oracle_f.py (make_copies.py). With every flag off each hook returns at
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

import os
from types import SimpleNamespace

import torch

from bar3.scale.compose2 import subgoal_c2 as SG

SUB_DEEP = 10_000_000                           # SUBGOAL.md's deeper probe
TARGET_FN = os.environ.get("TARGET_FN", "x2")   # compose3: the target, a function of (x, N)
TARGET_F = {"x2": lambda x, N: x * x % N, "x3": lambda x, N: pow(x, 3, N),
            "10x2p1": lambda x, N: (10 * x * x + 1) % N, "ten": lambda x, N: 10 * x * x % N,  # family:
            "rho": lambda x, N: (x * x + 1) % N, "plus3": lambda x, N: (x * x + 3) % N,    # SP_BREADTH's
            "minus1": lambda x, N: (x * x - 1) % N}[TARGET_FN]                          # formulas
LAW_NEAR = 4                                    # LAWNEAR: at most this many states a look


def ints(d):
    """Digits [n, P] (least significant first, a torch tensor) as Python ints."""
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in d.tolist()]


# ---------------------------------------------------------------------------------------------------- SUBGOAL
def _acc_root(target, climb, rows):
    """The acceptor's rows for the target (OracleAcceptor's construction: 60 rows a verify setting, seed 11)."""
    from bar3.scale.family.oracle_f import OracleRoot, prefix_values  # noqa: F401
    if getattr(target, "_score_root", None) is None:
        target._score_root = OracleRoot("tgt@XN", "target", climb.verify, 60, rows, climb.device, 11)
    return target._score_root


def subgoal_scores(climb, rows, roots):
    """SUBGOAL: score the candidates posed on the target's root; target.subgoal is the one singled out, or None."""
    t = climb.target
    if not getattr(climb, "subgoal_on", False) or t is None:
        return
    from bar3.scale.family.oracle_f import prefix_values
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
        if rr is None and getattr(climb, "tiefollow", False):  # compose3: TIEFOLLOW, the rest found by its value
            rr = follow(climb, sub)
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


def follow(climb, sub):
    """TIEFOLLOW: the rest root posed this level whose starting value, run on the target's probes, is the practised
    subgoal's value there (a value is posed again under a new name each level), or None."""
    from bar3.scale.family.oracle_f import prefix_values
    t = climb.target
    tr = getattr(t, "_root", None)
    if tr is None or sub not in t.values:
        return None
    run = lambda nm: [prefix_values(t.values[nm][0], st, rt, climb.lib)[t.values[nm][1]]   # noqa: E731
                      for st, rt in zip(climb.settings, tr.roots)]
    want = run(sub)
    for x in climb._level_roots:
        if getattr(x, "kind", None) != "rest":
            continue
        nm = "val." + x.key.split("@")[0][len("rest."):]
        if nm != sub and nm in t.values and all(bool((a == b).all()) for a, b in zip(want, run(nm))):
            return x
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
    from bar3.scale.family.spcore_f import _Res
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
        from bar3.scale.family.oracle_f import top_parts, value, width_for
        Nd, Xd = rows
        w = width_for("double", climb.settings[-1])
        Nt, xt = top_parts(Nd, Xd, w)
        target._share = SimpleNamespace(xs=value(xt), Ns=value(Nt))
    return target._share


# ---------------------------------------------------------------------------------------------------- SLICE
SLICE = os.environ.get("SLICE", "")              # slice: the cuts posed ("1": every cut of CUTS); "" off
CUTS = {                                          # an operation of the target's law -> its cuts at one digit place,
    "mul": (("acc.mul", "o,x,d", "row"),          #   as (law, pattern) on this machine: one operand at one digit,
            ("mul", "x,d", "xd")),                #   the rest carried (the row a + x d), or nothing carried (x d);
    "mod": (("red", "Bx+d", "step"),),            #   the dividend at one digit from the top: (B r + d) mod N;
    "add": (("add", "x,d", "add"),),              #   one operand at one digit: x + d
}
SLICE_PAIRS = {}                                  # (law, pattern) -> cut, once the target's law is read
SLICE_DEPTH = 5                                   # the law reader's depth (lawread.read)
LOOPS = os.environ.get("LOOPS", "")              # loops: each posed cut's loop ('1': all; or mul, red, add)
LOOP_OF = {                                       # a cut -> its loop over the digit places, the operation itself
    "row": ("mul", "x,o", "mul"),                 #   the row over the multiplier's digits: the multiply
    "xd": ("mul", "x,o", "mul"),                  #   x d likewise
    "step": ("red", "x+on", "red"),               #   Horner's step over the dividend's digits: the reduction
    "add": ("add", "pair", "add"),                #   x + d over the other operand's digits: the addition
}


def _target_rows(climb, rows):
    """(x, N, t, capacity) per row: the target's probes and acceptance rows, as subgoal_scores reads them."""
    t = climb.target
    tr = t.root(climb, rows)
    ar = _acc_root(t, climb, rows)
    sts = list(climb.settings) + list(climb.verify)
    draws = list(tr.draws) + list(ar.draws)
    xs = [x for d in draws for x in d.xs]
    Ns = [n for d in draws for n in d.Ns]
    ys = [y for d in draws for y in d.ys]
    caps = [st.B ** st.P for st, d in zip(sts, draws) for _ in range(d.n)]
    return xs, Ns, ys, caps


def slice_read(climb, rows):
    """SLICE: the target's law read once on its rows (lawread.read), SLICE_PAIRS set to the cuts of its operations
    that SLICE names and the book does not pose itself; recorded as a "slice" record."""
    if getattr(climb, "_slice_read", False) or climb.target is None:
        return
    climb._slice_read = True
    import time as _time
    from bar3.scale.slice import lawread
    from bar3.scale.family.spcore_f import LAW_BOOK
    t0 = _time.time()
    xs, Ns, ys, caps = _target_rows(climb, rows)
    law, ops, depth = lawread.read(xs, Ns, ys, caps, depth=SLICE_DEPTH)
    names = {c for cs in CUTS.values() for _, _, c in cs} if SLICE == "1" else set(SLICE.split(","))
    for op in ops:
        for lw, pat, cut in CUTS.get(op, ()):
            book = lw in climb.laws and pat in climb.patterns and pat in LAW_BOOK[lw][1]
            if cut in names and not book:
                SLICE_PAIRS[(lw, pat)] = cut
    lnames = {n for _, _, n in LOOP_OF.values()} if LOOPS == "1" else set(LOOPS.split(","))   # loops
    for (lw, pat), cut in list(SLICE_PAIRS.items()):
        if cut in LOOP_OF and LOOP_OF[cut][2] in lnames:
            llw, lpat, lname = LOOP_OF[cut]
            if not (llw in climb.laws and lpat in climb.patterns and lpat in LAW_BOOK[llw][1]):
                SLICE_PAIRS.setdefault((llw, lpat), lname)
    reading_pose(climb, law, LAW_BOOK)              # reading: the step's cuts, the repeated sub-laws, the constants
    climb.records.append(dict(task="slice", level=climb.level, law=law, ops=ops, depth=depth,
                              **({"subterms": dict(SUBTERM_SHAPES)} if SUBTERMS else {}),
                              constants=lawread.constants(xs, Ns, ys, caps), rows=len(set(zip(xs, Ns))),
                              cuts=[f"{lw}/{pat}={c}" for (lw, pat), c in SLICE_PAIRS.items()],
                              seconds=round(_time.time() - t0, 1)))


def pose_laws(climb, rows):
    """[(pattern, laws)] the pose phase poses: the book's laws by pattern, then (SLICE) the target's cuts."""
    from bar3.scale.family.spcore_f import LAW_BOOK
    book = [(pat, [lw for lw in climb.laws if pat in LAW_BOOK[lw][1]]) for pat in climb.patterns]
    if not SLICE:
        return book
    slice_read(climb, rows)
    out = dict(book)
    for (lw, pat) in SLICE_PAIRS:
        ls = out.setdefault(pat, [])
        if lw not in ls:
            ls.append(lw)
    return list(out.items())


def slice_target(law, d):
    """SLICE: a cut's answers on a draw (spcore_f.Draw: x, o, the digits d, N), by Python's integers, in the law
    target's form ({"A": digits [n, P], "Am": mask}); None when they do not vary."""
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    dv = d.dv.tolist()
    key = (law, d.pattern)
    if key in READING_PAIRS:                        # reading: the new laws' answers
        return reading_target(law, d)
    if key == ("acc.mul", "o,x,d"):
        ys = [o + x * e for o, x, e in zip(val(d.o), val(d.x), dv)]
    elif key == ("mul", "x,d"):
        ys = [x * e for x, e in zip(val(d.x), dv)]
    elif key == ("red", "Bx+d"):
        ys = [v % n for v, n in zip(val(d.term("Bx+d", None, None)), val(d.N))]
    elif key == ("mul", "x,o"):                      # loops: the multiply
        ys = [x * o for x, o in zip(val(d.x), val(d.o))]
    elif key == ("red", "x+on"):                     # loops: the reduction
        ys = [v % n for v, n in zip(val(d.term("x+on", None, None)), val(d.N))]
    elif key == ("add", "pair"):                     # loops: the addition, on the pair's kinds
        a, b = d.operands(None, None)
        ys = [p + q for p, q in zip(val(a), val(b))]
    elif key == ("add", "x,d"):
        ys = [x + e for x, e in zip(val(d.x), dv)]
    else:
        raise ValueError(key)
    assert all(0 <= y < B ** P for y in ys), key
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    if bool((A == A[:1]).all()):
        return None
    return {"A": A, "Am": torch.ones(d.n, dtype=torch.bool, device=dev)}


def xon_term(d):
    """LOOPS: the reduction's instance x + o N on a draw (x < N or x = N, o a number of the draw's width), by Python's
    integers, as digits [n, P]: the book's x + o N, which the book builds by the learner's own addition."""
    B, P = d.st.B, d.st.P
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    vs = [x + o * n for x, o, n in zip(val(d.x), val(d.o), val(d.N))]
    assert all(0 <= v < B ** P for v in vs)
    return torch.tensor([[(v // B ** i) % B for i in range(P)] for v in vs], dtype=torch.long, device=d.device)


# ---------------------------------------------------------------------------------------------------- READING (round 6)
DOWN = os.environ.get("DOWN", "")                # down: the step's own cuts ("1": the link and the comparison; or a list)
STEP_CUTS = {                                    # a cut -> its own cuts, down to the comparison
    "step": (("red", "pair", "link"),),          #   Horner's step is B conditional subtractions (B[link]): one is the link
    "link": (("lt", "pair", "cmp"),),            #   the link is a comparison and a select (STA.S lt SEL.S): the comparison
}
SUBLAW = os.environ.get("SUBLAW", "")            # sublaw: the reading's repeated two-operation sub-laws, as laws
SUBLAWS = {("mod", "mul"): ("mod.mul", "x,o,n")}  # outer(inner(a, b), N) -> (law, pattern): (a b) mod N on (x, o, n)
CONST = os.environ.get("CONST", "")              # const: the reading's constants, made from N, and its operations on them
CONST_OPS = ("add", "mul")                       # the reading's operations posed with a constant operand (x + c, x c)
PRIOR = os.environ.get("PRIOR", "")              # prior: under ct2a2, the reading's own concepts first among tied gains
PATTERNS6 = {                                    # the new patterns (spcore_r.PATTERNS, with SUBLAW or CONST on)
    "x,o,n": ((("x", "o", "n"),), "double", ""),  #   the sub-law: x o fits the registers before its reduction
    "n": ((("n",),), "same", ""),                #   a constant made from N (N >= 1 gives the borrow of 0 - N)
    "x,n": ((("x", "n"),), "carry", ""),         #   x + c and x c, N beside them (to make c)
}
READING_PAIRS = set()                            # the (law, pattern)s reading_pose posed, answered by reading_target


def _tree(s):
    """A reading (lawread's string) as nested (op, [args]); leaves as strings."""
    s = s.strip()
    i = s.find("(")
    if i < 0:
        return s
    body, args, depth, start = s[i + 1:-1], [], 0, 0
    for j, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(_tree(body[start:j]))
            start = j + 1
    args.append(_tree(body[start:]))
    return (s[:i], args)


def sublaws(t, out=None):
    """{(outer, inner): count} of the two-operation sub-laws outer(inner(a, b), N) in a reading's tree."""
    out = {} if out is None else out
    if isinstance(t, str):
        return out
    op, args = t
    if len(args) == 2 and args[1] == "N" and not isinstance(args[0], str) and len(args[0][1]) == 2:
        out[(op, args[0][0])] = out.get((op, args[0][0]), 0) + 1
    for a in args:
        sublaws(a, out)
    return out


def constant_ops(t, out=None):
    """{c: [ops]} for each constant leaf c of a reading's tree, with the operations it is an operand of."""
    out = {} if out is None else out
    if isinstance(t, str):
        return out
    op, args = t
    for a in args:
        if isinstance(a, str) and a.isdigit():
            ops = out.setdefault(int(a), [])
            if op not in ops:
                ops.append(op)
        constant_ops(a, out)
    return out


def reading_pose(climb, law, book_laws):
    """READING: what the target's reading poses beyond the cuts and their loops, added to SLICE_PAIRS (and to the law
    book, so that a concept practised in one binding counts as done), each unless the book poses it itself:
    - DOWN: the posed step's own cuts that DOWN names, down to the comparison (STEP_CUTS);
    - SUBLAW: each sub-law of SUBLAWS the reading holds at least twice;
    - CONST: each constant leaf c, made from N (const.c on n), and the reading's operations with c as an operand
      (op.c on x,n, for op in CONST_OPS)."""
    def book(lw, pat):
        return lw in climb.laws and pat in climb.patterns and pat in book_laws.get(lw, ("", ()))[1]

    def pose(lw, pat, name):
        book_laws.setdefault(lw, ("A", (pat,)))
        if not book(lw, pat) and (lw, pat) not in SLICE_PAIRS:
            SLICE_PAIRS[(lw, pat)] = name
            READING_PAIRS.add((lw, pat))

    if DOWN:
        names = {n for cs in STEP_CUTS.values() for _, _, n in cs} if DOWN == "1" else set(DOWN.split(","))
        todo, seen = list(SLICE_PAIRS.values()), set()
        while todo:
            for lw, pat, name in STEP_CUTS.get(todo.pop(0), ()):
                if name not in seen:
                    seen.add(name)
                    todo.append(name)
                    if name in names:
                        pose(lw, pat, name)
    tree = _tree(law) if law else None
    if SUBLAW and tree is not None:
        for k, n in sublaws(tree).items():
            if n >= 2 and k in SUBLAWS:
                pose(*SUBLAWS[k], "sublaw")
    if CONST and tree is not None:
        for c, ops in constant_ops(tree).items():
            pose(f"const.{c}", "n", "const")
            for op in ops:
                if op in CONST_OPS:
                    pose(f"{op}.{c}", "x,n", "const")
    if CONST1 and tree is not None:                 # family: the 1 of the reader's succ and pred
        pose_const1(tree, pose)
    if SUBTERMS and tree is not None:               # family: the reading's windows, smallest first
        pose_subterms(tree, pose)


def reading_target(law, d):
    """READING: the new laws' answers on a draw (spcore_r.Draw), by Python's integers, in the law target's form:
    - the comparison (lt on pair): the carry a < b;
    - the link (red on pair): a - b if a >= b, else a, where that is one conditional subtraction (b > 0, a < 2 b);
    - the sub-law (mod.mul on x,o,n): (x o) mod N;
    - a constant (const.c on n): c on every row, which may not vary;
    - an operation with a constant (op.c on x,n): x + c, x c.
    None when the answers do not vary (but for a constant), or fewer than two rows are in the law's range."""
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: [sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()]   # noqa: E731
    key, n = (law, d.pattern), d.n
    ok = [True] * n
    if key == ("lt", "pair"):
        a, b = (val(t) for t in d.operands(None, None))
        c = torch.tensor([int(p < q) for p, q in zip(a, b)], dtype=torch.long, device=dev)
        if bool((c == c[:1]).all()):
            return None
        return {"c": c, "cm": torch.ones(n, dtype=torch.bool, device=dev)}
    if key == ("red", "pair"):
        a, b = (val(t) for t in d.operands(None, None))
        ok = [q > 0 and p < 2 * q for p, q in zip(a, b)]
        ys = [p - q if p >= q else p for p, q in zip(a, b)]
    elif law in SUBTERM_TREES:                      # family: a window, by the law reader's arithmetic
        return subterm_target(law, d)
    elif key == ("mod.mul", "x,o,n"):
        ys = [x * o % m for x, o, m in zip(val(d.x), val(d.o), val(d.N))]
    elif law.startswith("const.") and d.pattern == "n":
        ys = [int(law.split(".")[1])] * n
    elif law in ("succ", "pred") and d.pattern == "x,n":        # family: CONST1, the reader's own
        ys = [x + 1 if law == "succ" else max(x - 1, 0) for x in val(d.x)]
    elif law.split(".")[0] in CONST_OPS and d.pattern == "x,n":
        op, c = law.split(".")
        ys = [x + int(c) if op == "add" else x * int(c) for x in val(d.x)]
    else:
        raise ValueError(key)
    ok = [k and 0 <= y < B ** P for k, y in zip(ok, ys)]
    ys = [y if k else 0 for k, y in zip(ok, ys)]
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    m = torch.tensor(ok, dtype=torch.bool, device=dev)
    if not law.startswith("const."):
        if int(m.sum()) < 2 or bool((A[m] == A[m][:1]).all()):
            return None
    return {"A": A, "Am": m}


def prior_rank(climb, names):
    """PRIOR: a sort key's tail among tied gains (ct2a2's pairs and singles): how many of names are not the reading's
    own (its cuts and their loops, the step's cuts, its sub-laws and constants); () with PRIOR off, so that pool order
    breaks the ties as before."""
    if not PRIOR:
        return ()
    own = {f"{lw}/{pat}" for lw, pat in SLICE_PAIRS}
    return (sum(nm.split("@")[0] not in own for nm in names),)


# ---------------------------------------------------------------------------------------------------- SUBTERMS (round 7)
import itertools                                 # noqa: E402

SUBTERMS = os.environ.get("SUBTERMS", "")        # subterms: the reading's windows, smallest first, as laws of their own
SUBTERM_OPS = int(os.environ.get("SUBTERM_OPS", 3))   # a window's most operations
SUBTERM_HOLES = 2                                # a window's most holes (in A, X and Y; N holds the modulus)
CONST1 = os.environ.get("CONST1", "")            # const1: the 1 of the reader's succ and pred (needs CONST's patterns)
HOLES = os.environ.get("HOLES", "")              # holes: a window's holes drawn from other rows than its modulus's
ROUNDT = os.environ.get("ROUNDT", "")            # roundt: a round must also give the data's answers at T = 2, 4, 8
ROUNDT_TS = (2, 4, 8)                            # the training data's one-digit T's past 1
PATTERNS7 = {                                    # the windows' patterns (spcore_f.PATTERNS, with SUBTERMS on)
    "w1": ((("x", "n"),), "double", ""),         #   one hole, the modulus beside it (x <= N)
    "w2": ((("x", "o", "n"),), "double", ""),    #   two holes, the modulus beside them
    "w1o": ((("o", "n"),), "double", ""),        #   HOLES: one hole from another row (so above N as often as not)
    "w2o": ((("o", "p", "n"),), "double", ""),   #   HOLES: two holes from two other rows
}
SUBTERM_SHAPES = {}                              # a window's law -> its shape (holes a and b)
SUBTERM_TREES = {}                               # a window's law -> its tree (holes "a" and "b")


def show_tree(t):
    """A tree (_tree's form) as a law string."""
    return t if isinstance(t, str) else f"{t[0]}({','.join(show_tree(a) for a in t[1])})"


def windows(t, most_ops=3, most_holes=2):
    """The reading's windows: every connected run of 2 to most_ops of its operations, each operation's children off
    the run cut off as holes, but N (the modulus) and the constants, which stay. Holes are renamed a, b, ... in the
    order they appear; at most most_holes of them. [(shape string, tree, operations)], each shape once, smallest
    first, then top-down as found."""
    def cut(node, k):                           # [(tree, ops)]: node cut off (a hole), or a run of <= k ops from it
        if isinstance(node, str):
            return [(node if node == "N" or node.isdigit() else "?", 0)]
        res = [("?", 0)]
        if k >= 1:
            for combo in itertools.product(*[cut(a, k - 1) for a in node[1]]):
                ops = 1 + sum(o for _, o in combo)
                if ops <= k:
                    res.append(((node[0], [c for c, _ in combo]), ops))
        return res

    def name_holes(t, names):
        if isinstance(t, str):
            if t != "?":
                return t
            names.append("ab"[len(names)] if len(names) < 2 else f"h{len(names)}")
            return names[-1]
        return (t[0], [name_holes(a, names) for a in t[1]])

    found, seen = [], set()

    def walk(node):
        if isinstance(node, str):
            return
        for tree, ops in cut(node, most_ops):
            if ops >= 2:
                names = []
                named = name_holes(tree, names)
                s = show_tree(named)
                if len(names) <= most_holes and s not in seen:
                    seen.add(s)
                    found.append((s, named, ops))
        for a in node[1]:
            walk(a)

    walk(t)
    return sorted(found, key=lambda w: w[2])      # stable: top-down within a size


def subterm_name(t, taken):
    """A window's law name: its operations top-down, each with its constant operands (w.mod.add1.mul10)."""
    parts = []

    def go(n):
        if isinstance(n, str):
            return
        parts.append(n[0] + "".join(a for a in n[1] if isinstance(a, str) and a.isdigit()))
        for a in n[1]:
            go(a)

    go(t)
    name = "w." + ".".join(parts)
    k = 2
    while name in taken:
        name = f"w.{'.'.join(parts)}.{k}"
        k += 1
    return name


def pose_subterms(tree, pose):
    """SUBTERMS: the reading's windows (at most SUBTERM_OPS operations, SUBTERM_HOLES holes), posed smallest first
    through reading_pose's pose, each on w1 or w2 by its holes."""
    for s, t, _ in windows(tree, SUBTERM_OPS, SUBTERM_HOLES):
        holes = len({h for h in ("a", "b") if _has_hole(t, h)})
        name = subterm_name(t, SUBTERM_TREES)
        SUBTERM_TREES[name], SUBTERM_SHAPES[name] = t, s
        pose(name, f"w{holes}" + ("o" if HOLES else ""), "subterm")


def more_answers(load_rows, digits, device):
    """ROUNDT: every training row with T in ROUNDT_TS, as (T, (N, x, answer) digits). For x2 the answer is the data's
    own; for another target, its function applied T times by Python's integers (as the T = 1 answers are)."""
    out = []
    for T in ROUNDT_TS:
        rows = load_rows(10 ** 6, seed=1, T=T)
        if TARGET_FN != "x2":
            def run(x, N, T=T):
                for _ in range(T):
                    x = TARGET_F(x, N)
                return x
            rows = [dict(r, result=run(int(r["x"]), int(r["modulus"]))) for r in rows]
        out.append((T, tuple(digits(rows, k).to(device) for k in ("modulus", "x", "result"))))
    return out


def round_filter(climb, progs, answers):
    """ROUNDT: of the programs found for the round, the ones right at full size on the T = 1 answers and, run T
    times, on every T of more_answers, in their order; the programs unchanged with ROUNDT off."""
    if not ROUNDT or not progs or answers is None:
        return progs
    from integration.lib.libcore import gen_on, pad, round_ok, round_settings, run_prog
    st = round_settings(climb.cells, climb.P_round)
    keep = []
    for p in progs:
        ok = bool(round_ok(p, climb.lib, climb.cells, answers, climb.P_round, climb.device))
        for T, (Nd, Xd, Yd) in getattr(climb, "more_answers", []) if ok else []:
            X = pad(Xd, climb.P_round)
            for _ in range(T):
                X, _c = run_prog(st, p, climb.lib, {"X": X, "N": pad(Nd, climb.P_round)}, gen_on(climb.device, 1),
                                 climb.device)
            ok = ok and bool((X[:, :Yd.shape[1]] == Yd).all())
        if ok:
            keep.append(p)
    climb.records.append(dict(task="roundt", level=climb.level, found=len(progs), kept=len(keep)))
    return keep


def unary_ops(t, out=None):
    """The one-operand operations of a reading's tree (the law reader's succ and pred)."""
    out = set() if out is None else out
    if not isinstance(t, str):
        if len(t[1]) == 1:
            out.add(t[0])
        for a in t[1]:
            unary_ops(a, out)
    return out


def pose_const1(tree, pose):
    """CONST1: succ and pred are an addition and a subtraction of 1, a constant the reading holds without a leaf.
    When the reading has one, its 1 is made from N (const.1 on n) and each of them is posed (on x,n), as CONST poses a
    constant and the reading's operations on it."""
    ops = sorted(unary_ops(tree) & {"succ", "pred"})
    if ops:
        pose("const.1", "n", "const")
        for op in ops:
            pose(op, "x,n", "const")


def _has_hole(t, h):
    return t == h if isinstance(t, str) else any(_has_hole(a, h) for a in t[1])


def _eval(t, env):
    """A window's value on numpy arrays (the law reader's operations), and a mask of the rows where every value on the
    way is in [0, cap)."""
    import numpy as np
    from bar3.scale.compose2.subgoal_c2 import OPS1, OPS2
    if isinstance(t, str):
        v = env[t] if t in env else np.full_like(env["N"], int(t))
        return v, np.ones(v.shape, bool)
    vals = [_eval(a, env) for a in t[1]]
    f = OPS2.get(t[0]) or OPS1[t[0]]
    v = f(*[a for a, _ in vals])
    ok = np.logical_and.reduce([m for _, m in vals]) & (v >= 0) & (v < env["cap"])
    return v, ok


def subterm_target(law, d):
    """SUBTERMS: a window's answers on a draw (w1: holes x; w2: x and o; N the modulus), by the law reader's own
    arithmetic; rows where a value on the way leaves [0, B^P) are masked. None when they do not vary."""
    import numpy as np
    B, P, dev = d.st.B, d.st.P, d.device
    val = lambda t: np.array([sum(int(v) * B ** i for i, v in enumerate(r)) for r in t.tolist()], np.int64)  # noqa
    kinds = PATTERNS7[d.pattern][0][0]                 # the holes' terms: x and o, or (HOLES) o and p
    env = {"a": val(d.term(kinds[0], None, None)), "N": val(d.N), "cap": np.int64(B ** P)}
    if len(kinds) == 3:
        env["b"] = val(d.term(kinds[1], None, None))
    v, ok = _eval(SUBTERM_TREES[law], env)
    ys = [int(y) if k else 0 for y, k in zip(v.tolist(), ok.tolist())]
    A = torch.tensor([[(y // B ** i) % B for i in range(P)] for y in ys], dtype=torch.long, device=dev)
    m = torch.tensor(ok.tolist(), dtype=torch.bool, device=dev)
    if int(m.sum()) < 2 or bool((A[m] == A[m][:1]).all()):
        return None
    return {"A": A, "Am": m}
