"""FAMILY.md §2: written by bar3/scale/family/make_copies7.py from integration/self_practice/spcore.py, 13 + 2 + 1 + 3 + 4
asserted replacements (rounds 2 and 3's, marked "compose2" and "compose3", round 4's, "slice", round 5's, "loops",
round 6's, "reading", then this study's, "family"). The rest of this docstring is the original's.

SELF-PRACTICE CORE (integration/SELF_PRACTICE.md): stage 3's climb with its designed curriculum taken out. The learner
poses its own practice problems from a law book applied to the rows' own numbers, in every binding of their operands
to the machine's registers, and chooses which to practise by bar 3's rules (bar3/PRACTICE.md, bar3/BREADTH.md), on
libcore's digit machine. Torch only (no numpy), like libcore, so that a stage-3 file can inline it after libcore.

What it holds:
  * the law book: arithmetic operations (lt, eq, min, max, red = mod, div, add, sub, mul, a + b c, a + b + c), each on
    patterns of the data's numbers built by the division law's form a + q b (q one, a digit d, or another row's x; b
    the modulus or the base). Nothing names the target, and most of it is irrelevant to it;
  * every injective binding of a law's operands to the registers A, X, Y, N: no register variant is picked by hand;
  * targets from facts and the machine's own routines only (no hand-written arithmetic): the order facts on the top
    parts (x' <= N', equal exactly when their digit strings are), the laws' own identities (red(a + q N) = a for
    a < N, ...), and routines the learner practised earlier (an addition builds x' + d N'; additions repeated d times
    build x' d). A problem is posed once the routines its construction needs exist: no designed order;
  * the probes: every law posed on the same pattern and binding shares one root, so one breadth-first search (many
    goals at once: bfs_multi) answers all of them;
  * the loop (SelfClimb, one unit of work per call, as libcore.Climb): pose, probe, classify (trivial, learnable,
    alias, duplicate, unsolved), choose (hardest first, or the cheaper pairs signal), practise (BPE over the corpus,
    every pick's routine without side effects on its inputs, Occam), then search the target (the round).
"""

from __future__ import annotations

import itertools
import os
import math
import time
import zlib
from collections import Counter

import torch

from bar3.scale.family import hooks as H                       # compose2
from integration.lib.libcore import (A_, ONE, PRIMS, RI, Batch, Deadline, ItemClass, Setting, _OneGoal, _apply_routine, _fresh,
                                     _late, _signatures, apply_items, bfs, bpe, cost, count_of, enumerate_items,
                                     expand, fresh, gen_on, keys_of, pad, programs, random_states, rewrite, round_ok,
                                     round_problem, run_items, show, steps, top_parts)

# ===================================================================================================== the law book
OPERAND_REGS = ("A", "X", "Y", "N")       # a law's operands go to these registers in every injective way

# A pattern: the instance kinds (one data term per operand; probe i is of kind i mod len), how many digits its
# numbers need beyond the operands' width w ("same": w; "carry": w + 1; "double": 2 w; the generic width rule picks
# the widest w that fits the registers), and a row filter ("ne": only rows with x' != N', where a term would leave the
# law's range).
PATTERNS = {
    "pair": ((("x", "n"), ("n", "x"), ("x", "x"), ("x", "z")), "carry", ""),     # room for add's carry
    "x+n": ((("x+n", "n"),), "carry", ""),
    "x+dn": ((("x+dn", "n"),), "carry", ""),
    "x+on": ((("x+on", "n"),), "double", ""),
    "Bx+d": ((("Bx+d", "n"),), "carry", "ne"),
    "x,d": ((("x", "d"),), "carry", ""),
    "x,o": ((("x", "o"),), "double", ""),
    "o,x,d": ((("o", "x", "d"),), "carry", ""),
}
if H.SUBLAW or H.CONST:                     # reading: the sub-law's and the constants' patterns
    PATTERNS.update(H.PATTERNS6)
if H.SUBTERMS:                              # family: the windows' patterns
    PATTERNS.update(H.PATTERNS7)
# The operations, each with the patterns it is posed on; a law on a pattern abstains on the kinds where neither a fact
# nor a routine gives its value.
LAW_BOOK = {
    "lt": ("c", ("pair", "x+dn")),           # sub.order: a < b exactly when a - b borrows
    "eq": ("c", ("pair",)),                  # a = b
    "min": ("A", ("pair",)),
    "max": ("A", ("pair",)),
    "red": ("A", ("pair", "x+n", "x+dn", "x+on", "Bx+d")),   # red.identity, red.zero, red.invariant, red.step
    "div": ("A", ("x+n", "x+dn", "x+on")),   # the division law's quotient
    "add": ("A", ("pair",)),                 # sub.inverse (checked by red and lt), add.identity
    "sub": ("A", ("pair", "x+n")),           # sub.identity, sub.self, sub.inverse
    "mul": ("A", ("x,d", "x,o")),            # mul.zero, mul.ladder, mul.distribute (as additions)
    "acc.mul": ("A", ("o,x,d",)),            # mul.ladder with its accumulator: a + b c
    "acc.add": ("A", ("o,x,d",)),            # a + b + c
}
GROW = {"same": 0, "carry": 1, "double": 2}


class NotPosable(Exception):
    """A construction needs a routine the learner does not have yet (or its instance leaves the law's range)."""


def width_for(grow, st):
    """The widest operand width whose instance fits the registers (P places)."""
    if grow == "double":
        return st.P // 2
    return st.P - GROW[grow]


def binding_name(b):
    return "".join(b)


def bindings(arity):
    return list(itertools.permutations(OPERAND_REGS, arity))


def digit_string(v, P):
    """A digit (one per row, below the base) as a digit string [n, P]: the digit at place 0, zeros above."""
    out = torch.zeros(v.shape[0], P, dtype=torch.long, device=v.device)
    out[:, 0] = v
    return out


class Ops:
    """The routines the learner can call to build problems: (law, pattern) -> [(token, binding)], every practised
    routine of that law on that pattern in the order practised. run() places operands in a routine's registers and
    runs it on a fresh machine: the first one (data is built with it), or the i-th (a check passes when any of them
    confirms it: a law's own pattern may not tell its routine from an alias, as "x if x < N, else 0" passes red's
    pair pattern; one routine that is right is then enough)."""

    def __init__(self):
        self.by_key, self.version = {}, 0

    def add(self, law, pattern, token, binding):
        self.by_key.setdefault((law, pattern), []).append((token, binding))
        self.version += 1

    def has(self, law, pattern):
        return (law, pattern) in self.by_key

    def count(self, law, pattern):
        return len(self.by_key.get((law, pattern), ()))

    def run(self, law, pattern, st, lib, operands, gen, device, i=0):
        """(A digits [n, P], c [n]) of routine i for law on pattern, operands placed in its bound registers."""
        if (law, pattern) not in self.by_key:
            raise NotPosable(f"{law}@{pattern}")
        tok, bd = self.by_key[(law, pattern)][i]
        ins = {r: v for r, v in zip(bd, operands)}
        n = operands[0].shape[0]
        b = fresh(st, ins, gen, n, device)
        run_items(st, b, (tok,), lib)
        return b.regs[0, A_].T.long(), b.c[0].long()


def repeat_add(ops, st, lib, D, v, times, gen, device):
    """D + times v by the learner's own addition (times a count per row; the additions repeated, mul.ladder)."""
    D = D.clone()
    for k in range(int(times.max()) if times.numel() else 0):
        more = torch.nonzero(times > k).flatten()
        if more.numel():
            D[more] = ops.run("add", "pair", st, lib, [D[more], v[more]], gen, device)[0]
    return D


class Draw:
    """One draw of the data's numbers for a pattern at a setting: n rows' top parts at the pattern's width (x' and N',
    so x' <= N'), another row's x' (o), the digits 0..B-1 over the probes (d), zero; the composite terms are built on
    demand by the learner's routines (NotPosable when one is missing)."""

    def __init__(self, pattern, st, n, rows, gen, device):
        kinds, grow, filt = PATTERNS[pattern]
        P, B = st.P, st.B
        Nd, Xd = rows
        w = width_for(grow, st)
        pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen, device=device)
        Nt, xt = top_parts(Nd[pick], Xd[pick], w)
        eq = (Nt == xt).all(1)
        idx = torch.arange(n, device=device)
        if filt == "ne":
            idx = torch.nonzero(~eq[:3 * n]).flatten()[:n]
        self.n = idx.numel()
        self.x, self.N = pad(xt[idx], P), pad(Nt[idx], P)
        self.o = pad(xt[3 * n:3 * n + self.n], P)
        self.p = pad(xt[2 * n:2 * n + self.n], P)        # family: a third row's x' (HOLES)
        self.eq = eq[idx]
        self.dv = torch.arange(self.n, device=device) % B
        self.d = digit_string(self.dv, P)
        self.z = torch.zeros_like(self.x)
        self.kind = torch.arange(self.n, device=device) % len(kinds)
        self.st, self.gen, self.device, self.pattern = st, gen, device, pattern
        self.cache = {}

    def term(self, name, ops, lib):
        if name in self.cache:
            return self.cache[name]
        st, gen, dev = self.st, self.gen, self.device
        if name in ("x", "z", "o", "d", "p"):                # family: p (HOLES)
            v = getattr(self, name)
        elif name == "n":
            v = self.N
        elif name == "x+n":
            v = ops.run("add", "pair", st, lib, [self.x, self.N], gen, dev)[0]
        elif name == "x+dn":
            v = repeat_add(ops, st, lib, self.x, self.N, self.dv, gen, dev)
        elif name == "x+on" and ("red", "x+on") in H.SLICE_PAIRS:      # loops: x + o N by Python
            v = H.xon_term(self)
        elif name == "x+on":
            v = repeat_add(ops, st, lib, self.x, self.N, count_of(self.o), gen, dev)
        elif name == "Bx+d":
            v = torch.cat([self.dv[:, None], self.x[:, :st.P - 1]], 1)      # x's digits moved up one place, d below
        else:
            raise ValueError(name)
        self.cache[name] = v
        return v

    def operands(self, ops, lib):
        """The operands' digits [n, P] per operand position, each probe of its kind."""
        kinds = PATTERNS[self.pattern][0]
        out = []
        for j in range(len(kinds[0])):
            v = torch.zeros_like(self.x)
            for k, kd in enumerate(kinds):
                m = self.kind == k
                if bool(m.any()):
                    v[m] = self.term(kd[j], ops, lib)[m]
            out.append(v)
        return out


def law_target(law, d, ops, lib):
    """The law's target on a draw: {"A": digits [n, P], "Am": mask [n]} and/or {"c": [n], "cm": mask} and/or a "check"
    (a function of the candidates' A digits [S, n, P] -> bool [S]), from facts and the learner's routines; None when
    the law says nothing on this draw (or its answers do not vary)."""
    if (law, d.pattern) in H.SLICE_PAIRS:               # slice: a cut of the target's law, answered by Python
        return H.slice_target(law, d)
    k, eq, P, dev, st, gen = d.kind, d.eq, d.st.P, d.device, d.st, d.gen
    allm = torch.ones(d.n, dtype=torch.bool, device=dev)
    x, N, z = d.x, d.N, d.z
    pat = d.pattern
    t = None
    if pat == "pair":                          # kinds 0 (x, N), 1 (N, x), 2 (x, x), 3 (x, 0)
        xz = (x == 0).all(1)
        if law == "lt":
            t = {"c": torch.where(k == 0, (~eq).long(), 0), "cm": allm}
        elif law == "eq":
            t = {"c": torch.where(k <= 1, eq.long(), torch.where(k == 2, 1, xz.long())), "cm": allm}
        elif law == "min":
            t = {"A": torch.where((k == 3)[:, None], z, x), "Am": allm}
        elif law == "max":
            t = {"A": torch.where((k <= 1)[:, None], N, x), "Am": allm}
        elif law == "red":
            t = {"A": torch.where(((k == 0) & ~eq)[:, None], x, z), "Am": (k == 0) | (k == 2)}
        elif law == "sub":
            t = {"A": torch.where((k == 3)[:, None], x, z), "Am": k >= 2}
        elif law == "add":
            if not (ops.has("red", "pair") and ops.has("lt", "pair")):
                raise NotPosable("add needs red and lt")
            sel = torch.nonzero((k == 0) & ~eq).flatten()

            def check(A_out, lib):             # sub.inverse: red(a + b, b) = a and a + b >= b, for a < b
                S = A_out.shape[0]
                if sel.numel() == 0:
                    return torch.ones(S, dtype=torch.bool, device=dev)
                flat = A_out[:, sel].reshape(-1, P).long()
                Ns = N[sel].repeat(S, 1)
                ok = torch.zeros(S, dtype=torch.bool, device=dev)
                for i in range(ops.count("red", "pair")):     # any practised red that confirms it
                    r = ops.run("red", "pair", st, lib, [flat, Ns], gen, dev, i)[0].view(S, -1, P)
                    ok |= (r == x[sel][None]).all(-1).all(-1)
                c = ops.run("lt", "pair", st, lib, [flat, Ns], gen, dev)[1].view(S, -1)
                return ok & (c == 0).all(-1)
            t = {"A": x, "Am": k == 3, "check": check}
    elif pat in ("x+n", "x+dn", "x+on"):
        q = {"x+n": digit_string(torch.ones(d.n, dtype=torch.long, device=dev), P), "x+dn": d.d, "x+on": d.o}[pat]
        d.term(pat, ops, lib)                  # the pattern itself needs the addition
        if law == "red":
            t = {"A": torch.where(eq[:, None], z, x), "Am": allm}
        elif law == "div":
            t = {"A": q, "Am": ~eq}
        elif law == "sub" and pat == "x+n":
            t = {"A": x, "Am": allm}
        elif law == "lt" and pat == "x+dn":
            t = {"c": ((d.dv == 0) & ~eq).long(), "cm": allm}
    elif pat == "Bx+d":
        if law == "red":
            v = d.term("Bx+d", ops, lib)
            src = "x+dn" if ops.has("red", "x+dn") else "x+on"
            t = {"A": ops.run("red", src, st, lib, [v, N], gen, dev)[0], "Am": allm}      # red.step, by the chain
    elif pat == "x,d":
        if law == "mul":
            t = {"A": repeat_add(ops, st, lib, z, x, d.dv, gen, dev), "Am": allm}
    elif pat == "x,o":
        if law == "mul":
            t = {"A": repeat_add(ops, st, lib, z, x, count_of(d.o), gen, dev), "Am": allm}
    elif pat == "o,x,d":
        if law == "acc.mul":
            m = repeat_add(ops, st, lib, z, x, d.dv, gen, dev)
            t = {"A": ops.run("add", "pair", st, lib, [m, d.o], gen, dev)[0], "Am": allm}
        elif law == "acc.add":
            s = ops.run("add", "pair", st, lib, [x, d.d], gen, dev)[0]
            t = {"A": ops.run("add", "pair", st, lib, [s, d.o], gen, dev)[0], "Am": allm}
    if t is None:
        return None
    if "A" in t:                               # the answers must vary where the law speaks (else nothing to learn)
        m = t["Am"]
        if int(m.sum()) < 2 and "check" not in t:
            return None
        if "check" not in t and bool((t["A"][m] == t["A"][m][:1]).all()):
            return None
    if "c" in t:
        cm = t["cm"]
        if bool((t["c"][cm] == t["c"][cm][:1]).all()):
            return None
    return t


class Goal:
    """One law's goal on a root's probes (one machine per setting): per setting its targets with masks."""

    def __init__(self, name, targets):
        self.name, self.targets = name, targets           # targets: one dict per setting (law_target's)

    def test(self, batches, lib=None):
        """Which states reach the goal (a check runs the learner's routines: lib, the library of the moment)."""
        dev = batches[0].regs.device
        ok = torch.ones(batches[0].S, dtype=torch.bool, device=dev)
        for b, t in zip(batches, self.targets):
            if "A" in t:
                m = (b.regs[:, A_] == t["A"].T.to(torch.uint8)[None]).all(1)          # [S, n]
                ok &= (m | ~t["Am"][None]).all(-1)
            if "c" in t:
                ok &= ((b.c.long() == t["c"][None]) | ~t["cm"][None]).all(-1)
            if "check" in t and bool(ok.any()):
                idx = torch.nonzero(ok).flatten()
                ok[idx[~t["check"](b.regs[idx, A_].permute(0, 2, 1), lib)]] = False
        return ok

    def fits(self, batches):
        """How many probes each state gets right (A on the masked probes; c too), over every setting."""
        dev = batches[0].regs.device
        f = torch.zeros(batches[0].S, dtype=torch.long, device=dev)
        for b, t in zip(batches, self.targets):
            r = torch.ones(b.S, t["Am"].numel() if "Am" in t else t["cm"].numel(), dtype=torch.bool, device=dev)
            if "A" in t:
                r &= (b.regs[:, A_] == t["A"].T.to(torch.uint8)[None]).all(1) & t["Am"][None]
            if "c" in t:
                r &= (b.c.long() == t["c"][None]) & t["cm"][None]
            f += r.sum(-1)
        return f


class Root:
    """A pattern bound to registers at the search settings: the probes every law on it shares, and the goals."""

    def __init__(self, pattern, binding, settings, NP, rows, ops, lib, device, seed):
        self.pattern, self.binding = pattern, binding
        self.key = f"{pattern}@{binding_name(binding)}"
        gen = gen_on(device, seed + zlib.crc32(self.key.encode()))
        self.draws, self.roots = [], []
        for st in settings:
            d = Draw(pattern, st, NP, rows, gen, device)
            ins = {r: v for r, v in zip(binding, d.operands(ops, lib))}
            self.draws.append(d)
            self.roots.append(fresh(st, ins, gen, d.n, device))

    def goals(self, ops, lib, laws):
        out = []
        for law in laws:
            try:
                ts = [law_target(law, d, ops, lib) for d in self.draws]
            except NotPosable:
                continue
            if all(t is not None for t in ts):
                out.append(Goal(law, ts))
        return out


class Acceptor:
    """A candidate's acceptance: the same law, pattern and binding on fresh rows (60 per setting, seed 11) at the
    verify settings; a program passes if it reaches the goal at every one. Built once per candidate."""

    def __init__(self, law, root, verify, rows, ops, lib, device, n=60, seed=11):
        self.cache = {}
        self.probs = []
        r = Root(root.pattern, root.binding, verify, n, rows, ops, lib, device, seed)
        g = r.goals(ops, lib, [law])
        self.ok = bool(g)
        self.ops, self.version = ops, ops.version          # a check's verdicts change when routines join
        self.checked = bool(g) and any("check" in t for t in g[0].targets)
        if g:
            for st, b, t in zip(verify, r.roots, g[0].targets):
                self.probs.append((st, b, Goal(law, [t])))

    def many(self, progs, lib):
        if self.checked and self.version != self.ops.version:
            self.cache, self.version = {}, self.ops.version
        keys = [show(p) for p in progs]
        live, seen = [], set()
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                seen.add(k)
                live.append(i)
        if not self.ok:
            for i in live:
                self.cache[keys[i]] = False
            live = []
        for st, root, goal in self.probs:
            if not live:
                break
            dev = root.regs.device
            b = root.take(torch.zeros(len(live), dtype=torch.long, device=dev))
            b = apply_items(st, b, torch.arange(len(live), device=dev), [("REP", ONE, tuple(progs[i])) for i in live],
                            lib)
            ok = goal.test([b], lib).tolist()
            for i, o in zip(live, ok):
                if not o:
                    self.cache[keys[i]] = False
            live = [i for i, o in zip(live, ok) if o]
        for i in live:
            self.cache[keys[i]] = True
        return [self.cache.get(k, False) for k in keys]


# ===================================================================================================== many goals
class _Res:
    def __init__(self, parents, found_cost, goals):
        self.parents, self.found_cost, self.goals = parents, found_cost, goals


WHOLE = ("LDA.X", "LDA.Y", "LDA.N", "LDA.S", "STA.X", "STA.Y", "STA.S", "CLA", "CLC", "CLK", "SHL.A", "SHL.X",
         "SHL.Y", "SHR.A", "SHR.X", "SHR.Y", "SEL.S", "BRG.X")    # the instructions that act on whole registers
SOURCES = ("A", "X", "Y", "N", "Z")       # a routed input: a register as it was before the call, or zero


def route(binding, srcs):
    """Straight-line code that fills a routine's input registers (binding, one per operand) from sources (srcs: a
    register as it was before the code, or Z for zero), A as the carrier, the A input last; None when no order of the
    moves does it on this machine (N is read-only; a source overwritten before it is read)."""
    rest = [i for i in range(len(binding)) if binding[i] != "A"]
    for order in itertools.permutations(rest):
        code, held = [], {r: r for r in "AXYNS"}
        ok = True
        for i in list(order) + [i for i in range(len(binding)) if binding[i] == "A"]:
            dst, src = binding[i], srcs[i]
            if held[dst] == src:
                continue
            if dst == "N":
                ok = False
                break
            if held["A"] != src:
                where = [r for r in "AXYNS" if held[r] == src]
                if src == "Z":
                    code.append("CLA")
                elif where:
                    code.append(f"LDA.{where[0]}")
                else:
                    ok = False
                    break
                held["A"] = src
            if dst != "A":
                code.append(f"STA.{dst}")
                held[dst] = src
        if ok and all(held[b] == s_ for b, s_ in zip(binding, srcs)):
            return code
    return None


def routines_in(items, lib, out=None, closure=False):
    """The routines a program calls (through merges and loops); with closure, also the routines those call."""
    out = set() if out is None else out
    for it in items:
        if isinstance(it, tuple):
            if it[1] in lib:
                routines_in(lib[it[1]], lib, out, closure)
            routines_in(it[2], lib, out, closure)
        elif it.startswith("T."):
            if closure and it not in out and it in lib:
                out.add(it)
                routines_in(lib[it], lib, out, closure)
            out.add(it)
        elif it in lib:
            routines_in(lib[it], lib, out, closure)
    return out


def unwrap(prog):
    """A program with its ONE[...] items (the search by calls' routed calls) written out: the same instructions, in a
    form every machine runs (the prediction path's has no ONE)."""
    out = []
    for it in prog:
        if isinstance(it, tuple) and it[1] == ONE:
            out.extend(unwrap(it[2]))
        else:
            out.append(it)
    return tuple(out)


def round_ok_t(prog, lib, cells, rows, P, device, t, seed=1):
    """Does the program, run t times (each round from the last one's answer, a fresh machine), reproduce every answer
    row (Nd, Xd, Yd)? (libcore.round_ok for t = 1.)"""
    Nd, Xd, Yd = rows
    st = Setting(10, P // 2, cells, P)
    gen = gen_on(device, seed)
    x = pad(Xd, P)
    for _ in range(t):
        b = fresh(st, {"X": x, "N": pad(Nd, P)}, gen, Xd.shape[0], device)
        run_items(st, b, prog, lib)
        x = torch.cat([b.regs[0, A_].T.long()[:, :P // 2], torch.zeros_like(x[:, P // 2:])], 1)
    return bool((x[:, :Yd.shape[1]] == Yd).all())


_SALT = torch.randint(-(1 << 62), 1 << 62, (8192,), generator=torch.Generator().manual_seed(4242), dtype=torch.int64)
_SALT_ON = {}
SPELL_K = int(os.environ.get("SPELL", "0") or 0)              # compose2: SPELL, spellings tested a goal state


def bfs_multi(roots, settings, goals, accepts, classes, lib, keep=False, fits=None, **kw):
    """bfs_roots with one root: libcore.bfs with many goals sharing it (each goal's goal states listed one spelling
    each, fewest top-level items first, then fewest primitives, tested by its acceptor at the first cost where it has
    any; the search goes on while any goal is open). Returns ([per goal: dict(found_cost, accepted, aliases)], level
    sizes, stop[, held levels, parents]); fits, a list, gets the best fit of goals[0] at every level."""
    out, sizes, stops, levels, _, parents = bfs_roots([roots], settings, [(0, g) for g in goals], accepts, classes, lib,
                                                      fits=fits, **kw)
    return (out, sizes[0], stops[0], levels, parents) if keep else (out, sizes[0], stops[0])


def bfs_roots(roots, settings, goals, accepts, classes, lib, max_cost=7, max_states=1_000_000, deadline=None,
              chunk=1 << 21, fuse=1, batch=256, spell=8, max_goals=50_000, fits=None):
    """bfs_multi for many roots in one search: roots [R][per setting Batch, S = 1] (the same probe shapes), goals
    [(root index, Goal)], accepts per goal. Every root keeps its own search: its states' keys carry a salt of its own,
    so no two roots' states merge; it has its own budget (a level whose candidates would pass 1.5 max_states streams that
    root's goal states only, and the root stops after it; a level that stores more than max_states also ends it); it
    stops once its goals are closed. Each level's work is shared: one apply_items over every live root's rows.
    Returns ([per goal: dict(found_cost, accepted, aliases)], [per root: level sizes], [per root: stop], held levels,
    their root ids, parents); fits, a list, gets the best fit of goals[0] (on its root) at every level."""
    dev = roots[0][0].regs.device
    R = len(roots)
    if dev not in _SALT_ON:
        _SALT_ON[dev] = _SALT.to(dev)
    salt = _SALT_ON[dev][:R]
    lv0 = [Batch.concat([roots[r][si] for r in range(R)]) for si in range(len(settings))]
    rid0 = torch.arange(R, device=dev)
    levels, rids = [lv0], [rid0]
    seen = torch.sort(keys_of(lv0) ^ salt[rid0]).values
    parents = [None]
    out = [dict(found_cost=None, accepted=None, aliases=[]) for _ in goals]
    sizes, stops = [[1] for _ in range(R)], [""] * R
    open_by = {r: [] for r in range(R)}
    for j, (r, g) in enumerate(goals):
        if bool(g.test([b.take(torch.tensor([r], device=dev)) for b in lv0], lib)[0]):
            out[j].update(found_cost=0, accepted=[()])
        else:
            open_by[r].append(j)
    stopped = torch.tensor([not open_by[r] for r in range(R)], dtype=torch.bool, device=dev)
    if fits is not None:
        fits.append(int(goals[0][1].fits([b.take(torch.tensor([goals[0][0]], device=dev)) for b in lv0]).max()))
    by_cost = {}
    for ci, cl in enumerate(classes):
        by_cost.setdefault(cl.cost, []).append(ci)
    by_cost = {c: torch.tensor(v, dtype=torch.long, device=dev) for c, v in by_cost.items()}
    late = False
    for d in range(1, max_cost + 1):
        if bool(stopped.all()) or late:
            break
        lastr = torch.full((R,), d == max_cost, dtype=torch.bool, device=dev)
        n_cand = torch.zeros(R, dtype=torch.long, device=dev)
        n_fresh = torch.zeros(R, dtype=torch.long, device=dev)
        ck, cc, cs, ci_, crid, chunks = [], [], [], [], [], []
        gk = {j: [] for r in range(R) for j in open_by[r]}
        best = 0
        live_src = {}
        for sl in range(d):                          # each held level's rows of roots still searching
            if levels[sl] is not None:
                live_src[sl] = torch.nonzero(~stopped[rids[sl]]).flatten()

        def blocks(d=d):
            for c, cis in by_cost.items():
                sl = d - c
                if sl < 0 or sl not in live_src or live_src[sl].numel() == 0:
                    continue
                al = live_src[sl]
                for s0 in range(0, al.numel(), chunk):
                    rows_ = al[s0:s0 + chunk]
                    srcs = [b.take(rows_) for b in levels[sl]]
                    i_step = max(1, chunk // rows_.numel())
                    for i0 in range(0, cis.numel(), i_step):
                        yield cis[i0:i0 + i_step], sl, rows_, srcs

        def goal_hits(sub, sub_rid, keys_sub):
            hit = torch.zeros(sub_rid.numel(), dtype=torch.bool, device=dev)
            for r in torch.unique(sub_rid).tolist():
                if not open_by[r]:
                    continue
                sel = torch.nonzero(sub_rid == r).flatten()
                part = [b.take(sel) for b in sub]
                for j in open_by[r]:
                    g = goals[j][1].test(part, lib)
                    if bool(g.any()):
                        if keys_sub is not None:
                            gk[j].append(keys_sub[sel][g])
                        hit[sel[g]] = True
            return hit
        work = blocks()
        nxt = next(work, None)
        while nxt is not None:
            if _late(deadline):
                late = True
                break
            group, n_rows = [], 0
            while nxt is not None and (not group or n_rows + nxt[0].numel() * nxt[2].numel() <= fuse * chunk):
                group.append(nxt)
                n_rows += nxt[0].numel() * nxt[2].numel()
                nxt = next(work, None)
            parts, its, srcrows = [], [], [[] for _ in settings]
            for ii, sl, rows_, srcs in group:
                ns, k = rows_.numel(), ii.numel()
                its.append(torch.arange(k, device=dev).repeat_interleave(ns) + len(parts))
                parts.extend(classes[i].rep for i in ii.tolist())
                rs = torch.arange(ns, device=dev).repeat(k)
                for si, b in enumerate(srcs):
                    srcrows[si].append(b.take(rs))
            ri_all = its[0] if len(its) == 1 else torch.cat(its)
            outs_all = [apply_items(st, r[0] if len(r) == 1 else Batch.concat(r), ri_all, parts, lib)
                        for st, r in zip(settings, srcrows)]
            del srcrows
            off = 0
            for ii, sl, rows_, srcs in group:
                ns, k = rows_.numel(), ii.numel()
                outs = outs_all if len(group) == 1 else [o.take(slice(off, off + k * ns)) for o in outs_all]
                off += k * ns
                ri = torch.arange(k, device=dev).repeat_interleave(ns)
                rs = torch.arange(ns, device=dev).repeat(k)
                rid_rows = rids[sl][rows_][rs]
                keys = keys_of(outs) ^ salt[rid_rows]
                m = _fresh(seen, keys)
                if not bool(m.any()):
                    continue
                idx = torch.nonzero(m).flatten()
                sub = [b.take(idx) for b in outs]
                sub_rid = rid_rows[idx]
                n_fresh += torch.bincount(sub_rid, minlength=R)
                del outs
                if fits is not None:
                    sel = torch.nonzero(sub_rid == goals[0][0]).flatten()
                    if sel.numel():
                        best = max(best, int(goals[0][1].fits([b.take(sel) for b in sub]).max()))
                hit = goal_hits(sub, sub_rid, keys[idx])
                keep = ~lastr[sub_rid] | hit
                if not bool(keep.all()):
                    kk = torch.nonzero(keep).flatten()
                    idx, sub, sub_rid = idx[kk], [b.take(kk) for b in sub], sub_rid[kk]
                if idx.numel() == 0:
                    continue
                ck.append(keys[idx])
                cc.append(ii[ri[idx]])
                cs.append(torch.full((idx.numel(),), sl, dtype=torch.long, device=dev))
                ci_.append(rows_[rs[idx]])
                crid.append(sub_rid)
                chunks.append(sub)
                n_cand += torch.bincount(sub_rid, minlength=R)
                over = torch.nonzero(~lastr & (n_cand > 1.5 * max_states)).flatten()
                if over.numel():                     # these roots stream the rest of this level: goal states only
                    lastr[over] = True
                    for r in over.tolist():
                        stops[r] = f"budget: level {d} streamed (over {max_states} states)"
                    for jj in range(len(chunks)):
                        o_ = torch.isin(crid[jj], over)
                        if not bool(o_.any()):
                            continue
                        h = goal_hits(chunks[jj], crid[jj], None)
                        kk = torch.nonzero(~o_ | h).flatten()
                        ck[jj], cc[jj], cs[jj], ci_[jj], crid[jj] = (t[kk] for t in (ck[jj], cc[jj], cs[jj], ci_[jj],
                                                                                    crid[jj]))
                        chunks[jj] = [b.take(kk) for b in chunks[jj]]
                    n_cand = torch.bincount(torch.cat(crid), minlength=R) if crid else n_cand
            del outs_all
        if late or _late(deadline):
            late = True
            for r in range(R):
                if not bool(stopped[r]):
                    stops[r] = stops[r] or f"deadline at level {d}"
        if ck:
            allk = torch.cat(ck)
            allr = torch.cat(crid)
            uk, inv = torch.unique(allk, sorted=True, return_inverse=True)
            pos = torch.arange(allk.numel(), device=dev)
            first = torch.full((uk.numel(),), allk.numel(), dtype=torch.long, device=dev).scatter_reduce(
                0, inv, pos, reduce="amin")
            sfirst = torch.sort(first).values
            rank = torch.searchsorted(sfirst, first)
            child = rank[inv]
            order = torch.sort(child, stable=True).indices
            parents.append((child[order], torch.cat(cc)[order], torch.cat(cs)[order], torch.cat(ci_)[order]))
            urid = allr[sfirst]                        # each stored state's root, in rank order
            per_root = torch.bincount(urid, minlength=R)
            parts_, offs = [[] for _ in settings], 0
            for jj in range(len(chunks)):
                ch, n = chunks[jj], chunks[jj][0].S
                lo_, hi_ = int(torch.searchsorted(sfirst, offs)), int(torch.searchsorted(sfirst, offs + n))
                rows = sfirst[lo_:hi_] - offs
                for si in range(len(settings)):
                    parts_[si].append(ch[si].take(rows))
                offs += n
            levels.append([Batch.concat(p) for p in parts_])
            rids.append(urid)
            chunks = None
        else:
            uk = torch.zeros(0, dtype=torch.int64, device=dev)
            rank = torch.zeros(0, dtype=torch.long, device=dev)
            per_root = torch.zeros(R, dtype=torch.long, device=dev)
            parents.append(None)
            levels.append(None)
            rids.append(torch.zeros(0, dtype=torch.long, device=dev))
        if fits is not None:
            fits.append(best)
        for r in range(R):
            if bool(stopped[r]):
                continue
            sizes[r].append(int(n_fresh[r]) if bool(lastr[r]) else int(per_root[r]))
            if bool(lastr[r]) and not stops[r]:
                stops[r] = f"exhausted cost {max_cost}"
            if int(per_root[r]) > max_states and not stops[r]:
                stops[r] = f"budget: level {d} has {int(per_root[r])} > {max_states} states"
        for r in range(R):
            if bool(stopped[r]):
                continue
            for j in list(open_by[r]):
                if not gk.get(j):
                    continue
                gs = torch.unique(torch.cat(gk[j]))
                gidx = rank[torch.searchsorted(uk, gs)].tolist()
                res = _Res(parents, d, gidx)
                cand = []
                for gi, g in enumerate(gidx[:max_goals]):
                    if gi % 256 == 0 and _late(deadline):
                        break
                    if SPELL_K and len(cand) >= max_goals:   # compose2: SPELL, at most max_goals programs a level
                        break
                    f = programs(_OneGoal(res, g), classes, limit=SPELL_K or 1)   # compose2: SPELL spellings
                    for q in f[:SPELL_K or 1]:
                        cand.append((len(q), cost(expand(q, lib)), q, g))
                cand.sort(key=lambda c: (c[0], c[1]))
                progs = [c[2] for c in cand]
                good = []
                for i0 in range(0, len(cand), batch):
                    if _late(deadline):
                        break
                    good = [c for c, o in zip(cand[i0:i0 + batch], accepts[j](progs[i0:i0 + batch], lib)) if o]
                    if good:
                        break
                if good:
                    acc = [c[2] for c in good]
                    if spell > 1:
                        more = [q for c in good for q in programs(_OneGoal(res, c[3]), classes, limit=spell)[1:]]
                        acc += [q for q, o in zip(more, accepts[j](more, lib) if more else []) if o]
                    out[j].update(found_cost=d, accepted=acc)
                    open_by[r].remove(j)
                else:
                    out[j]["aliases"].append((d, [show(p) for p in progs[:5]]))
            if stops[r] or not open_by[r] or bool(lastr[r]):
                stopped[r] = True
        if ck:
            seen = torch.sort(torch.cat([seen, uk])).values
    for r in range(R):
        if open_by[r] and not stops[r]:
            stops[r] = f"exhausted cost {max_cost}"
    return out, sizes, stops, levels, rids, parents


# ===================================================================================================== the learner
def side_effects(prog, lib, st, root, binding):
    """Does prog leave its input registers (the binding's, but A, the answer) as they were, and how many registers
    other than A does it change, on these probes? (bar3/learner.py's rule for routines: no side effects on inputs.)"""
    b = root.copy()
    run_items(st, b, prog, lib)
    changed = [(b.regs[0, i] != root.regs[0, i]).any().item() for i in range(5)]
    keeps = all(not changed[RI[r]] for r in binding if r != "A")
    return keeps, sum(changed[1:])


class SelfClimb:
    """The self-practising climb as a state machine a training loop drives (step: one unit of work), libcore.Climb's
    interface. Each level (bar 3's round):
      pose    every law of the book on every pattern and binding whose construction works now (the routines it needs
              exist), minus those already practised: the pool, grouped by root;
      probe   one breadth-first search per root (all its laws at once) over the composition vocabulary (the routines
              and the primitives, loops of bodies up to comp_body), at probe_budget; a root posed for the first time is
              searched once more over the primitives alone at fallback_budget for its open laws;
      choose  trivial (cost <= 1), learnable (cost >= 2, not an alias of a token, not a duplicate of an earlier
              learnable by what it does on random machines), unsolved; then the rule: "lp" (the K costliest learnable,
              ties in pool order), "ct2a2" (the cheaper pairs signal: the target probed with every learnable's routine,
              and with every pair whose member is among the 2 hardest; the best pairs' members), "all";
      sleep   the picks' programs join the corpus, BPE merges, each pick's routine (among its shortest programs, one
              that leaves its inputs as they were and changes the fewest registers, then Occam by expansion and steps);
      target  the round: libcore's search by composition at full size with the answers as its goal.
    """

    rowcheck = tiebreak = subgoal_on = rest_deep = False         # compose2: ROWCHECK, TIEBREAK, SUBGOAL, REST_DEEP
    tiefollow = False                                            # compose3: TIEFOLLOW

    def __init__(self, cells, device, P_round, rule="lp", K=4, NP=16, probe_budget=1_000_000,
                 fallback_budget=1_000_000, deep_budget=5_800_000, deep_k=8, group=16, deep_group=4, max_body=3,
                 comp_body=3, max_cost=7,
                 min_count=2.0, max_merges=12,
                 levels=12, round_budget=4_000_000, round_cost=5, look_budget=300_000, settings=((10, 1), (10, 2)),
                 verify=((10, 1), (10, 2), (10, 3)), chunk=1 << 21, fuse=1, search_s=None, seed=0, log=None,
                 patterns=None, laws=None, round_retry=0, target=None, macro_after=12, macro_cost=2,
                 deep_by="root", deep_focus=True):
        self.cells, self.device, self.P_round = cells, device, P_round
        self.rule, self.K, self.NP = rule, K, NP
        self.probe_budget, self.fallback_budget = probe_budget, fallback_budget
        self.deep_budget, self.deep_k, self._fit = deep_budget, deep_k, {}
        self.group, self.deep_group = group, deep_group
        self._variants = rule not in ("lpc", "lpca")    # pose practised concepts' other bindings (concept rules: no)
        self._want_variants = False
        self.max_body, self.comp_body, self.max_cost = max_body, comp_body, max_cost
        self.min_count, self.max_merges, self.levels = min_count, max_merges, levels
        self.round_budget, self.round_cost, self.look_budget = round_budget, round_cost, look_budget
        self.settings = [Setting(B, W, cells) for B, W in settings]
        self.verify = [Setting(B, W, cells) for B, W in verify]
        self.chunk, self.fuse, self.search_s, self.seed = chunk, fuse, search_s, seed
        self.patterns = list(PATTERNS) if patterns is None else list(patterns)
        self.laws = list(LAW_BOOK) if laws is None else list(laws)
        self.lib, self.corpus, self.ops = {}, {}, Ops()
        self.practiced = {}                     # candidate name -> level
        self.level, self.phase, self.queue = 0, "pose", []
        self.records, self.history, self.events = [], [], []
        self.rounds, self.done = [], False
        self.fallback_seen = set()              # roots whose primitives-alone search ran (it never changes)
        self._roots, self._accs = {}, {}        # key -> Root, (law, key) -> Acceptor (the same probes every level)
        self._sig_cache, self._comp, self._prims = {}, None, None
        self.pool, self.results, self.picks = [], {}, []
        self.tokens_sig = None
        self.looks, self._base_look, self._scores = [], None, {}
        self.round_retry, self._idle, self.t = round_retry, 0, 1
        self.macro_after, self.macro_cost = macro_after, macro_cost     # the round by calls: past this many routines
        self.deep_by, self.deep_focus = deep_by, deep_focus
        self._focus, self._focus_classes = {}, {}        # root -> law -> its closest attempt's routines
        self.target, self._near, self.near_log = target, [], []      # a small target (the tiny core's oracle arm)
        self.log = log

    # ------------------------------------------------------------------ vocabularies
    def routines(self):
        return tuple(sorted(k for k in self.lib if k.startswith("T.")))

    def comp_classes(self, deadline=None):
        r = self.routines()
        if self._comp is None or self._comp[0] != r:
            self._comp = (r, enumerate_items(list(PRIMS) + list(r), self.lib, self.settings, self.device,
                                             max_body=self.comp_body, deadline=deadline, cache=self._sig_cache,
                                             chunk=self.chunk * self.fuse))
        return self._comp[1]

    def prim_classes(self, deadline=None):
        if self._prims is None:
            self._prims = enumerate_items(list(PRIMS), {}, self.settings, self.device, max_body=self.max_body,
                                          deadline=deadline, cache=self._sig_cache, chunk=self.chunk * self.fuse)
        return self._prims

    def signature(self, prog, lib):
        if self.tokens_sig is None or self.tokens_sig[0] != tuple(sorted(lib)):
            gen = gen_on(self.device, 7)
            tests = [random_states(st, 24, 6, gen, self.device) for st in self.settings]
            toks = list(PRIMS) + [k for k in lib if k.startswith("T.")]
            sigs = _signatures(self.settings, tests, [("REP", ONE, (t,)) for t in toks], lib, chunk=self.chunk)
            self.tokens_sig = (tuple(sorted(lib)), tests, set(sigs), {})
            for t_, s_ in zip(toks, sigs):                             # compose2: the tokens by signature
                self.tokens_sig[3].setdefault(s_, []).append(t_)
        tests = self.tokens_sig[1]
        return _signatures(self.settings, tests, [("REP", ONE, tuple(prog))], lib, chunk=self.chunk)[0]

    # ------------------------------------------------------------------ pose
    def pose(self, rows):
        pool, roots = [], []
        for pat, laws in H.pose_laws(self, rows):          # slice: the book's laws by pattern, then the cuts
            if not laws:
                continue
            arity = len(PATTERNS[pat][0][0])
            for bd in bindings(arity):
                if pat in H.PATTERNS7 and bd[-1] != "N":     # family: a window's modulus stays in N
                    continue
                key = f"{pat}@{binding_name(bd)}"
                r = self._roots.get(key)
                if r is None:
                    try:
                        r = Root(pat, bd, self.settings, self.NP, rows, self.ops, self.lib, self.device, self.seed)
                    except NotPosable:
                        continue
                    self._roots[key] = r
                goals = [g for g in r.goals(self.ops, self.lib, laws) if f"{g.name}/{r.key}" not in self.practiced
                         and (self._variants or (g.name, pat) not in self.concepts_done())]
                if not goals:
                    continue
                r.goal_list = goals
                roots.append(r)
                for g in goals:
                    pool.append(f"{g.name}/{r.key}")
        if self.target is not None:                     # the target's restrictions, projections, near misses
            for r, goals in self.target.pose(self, rows, self._near):
                goals = [g for g in goals if f"{g.name}/{r.key}" not in self.practiced]
                if goals:
                    r.goal_list = goals
                    roots.append(r)
                    pool.extend(f"{g.name}/{r.key}" for g in goals)
        return pool, roots

    # ------------------------------------------------------------------ probe
    def acceptors(self, root, rows):
        accs = []
        for g in root.goal_list:
            if (g.name, root.key) not in self._accs:
                self._accs[(g.name, root.key)] = (
                    self.target.acceptor(self, rows, root, g.name) if hasattr(root, "kind") else
                    Acceptor(g.name, root, self.verify, rows, self.ops, self.lib, self.device))
            accs.append(self._accs[(g.name, root.key)])
        return accs

    def probe(self, group, rows, deadline=None, deep=False):
        """A group of roots' probe (bfs_roots: one search, every root its own budget): their laws at once over the
        composition vocabulary at probe_budget, then (a root's first time, once routines exist) the open laws over the
        primitives alone at fallback_budget (that search never changes); with deep, the open laws again at
        deep_budget. Records each open law's closeness (how much of the gap left by the best single instruction the
        closest held state closes): the gate for the deep probes."""
        t0 = time.time()
        dl = deadline
        if self.search_s:
            dl = min(deadline or float("inf"), t0 + self.search_s)
        accs = {r.key: self.acceptors(r, rows) for r in group}
        out = {r.key: {} for r in group}
        if deep:
            for r in group:
                for g in r.goal_list:
                    prev = self.results.get(f"{g.name}/{r.key}")
                    if prev is not None:
                        out[r.key][g.name] = (prev, prev["how"])
        runs = []

        def run(how, classes, lib, cap, members):
            """members: [(root, [goal indices])]."""
            t1 = time.time()
            goals, gacc, owner = [], [], []
            for ri, (r, idx) in enumerate(members):
                for i in idx:
                    goals.append((ri, r.goal_list[i]))
                    gacc.append(accs[r.key][i].many)
                    owner.append(r)
            res, sizes, stops, levels, rids, parents = bfs_roots([r.roots for r, _ in members], self.settings, goals,
                                                              gacc, classes, lib, max_cost=self.max_cost,
                                                              max_states=cap, deadline=dl, chunk=self.chunk,
                                                              fuse=self.fuse)
            fit = {r.key: {} for r, _ in members}
            for (ri, g), x, r in zip(goals, res, owner):
                if x["found_cost"] is None or g.name not in out[r.key]:
                    if x["found_cost"] is not None or g.name not in out[r.key]:
                        out[r.key][g.name] = (x, how)
                if x["found_cost"] is None:             # closeness: the share of the gap single instructions leave
                    tot = sum(int(t["Am"].sum()) if "Am" in t else int(t["cm"].sum()) for t in g.targets)
                    f, best = [], (-1, None, None)
                    for d, (lv, rd) in enumerate(zip(levels, rids)):
                        sel = torch.nonzero(rd == ri).flatten() if lv is not None else None
                        if sel is None or not sel.numel():
                            f.append(0)
                            continue
                        ff = g.fits([b.take(sel) for b in lv])
                        f.append(int(ff.max()))
                        if d and f[-1] > best[0]:
                            best = (f[-1], d, int(sel[int(ff.argmax())]))
                    if best[1] is not None:              # the closest held state's program: its routines are the
                        pr = programs(_OneGoal(_Res(parents, best[1], [best[2]]), best[2]), classes, limit=1)
                        self._focus.setdefault(r.key, {})[g.name] = (
                            routines_in(pr[0], lib, closure=self.deep_focus == "closure") if pr else set())
                    base = max(f[:2])                   # (a law whose goal states acceptance rejected is not out
                    fit[r.key][g.name] = ((max(f) - base) / (tot - base) if tot > base and not x["aliases"]  # of reach)
                                          else 0.0)
                elif x["found_cost"] is not None:
                    out[r.key][g.name] = (x, how)
            del levels
            runs.append(dict(how=how, roots=[r.key for r, _ in members], seconds=round(time.time() - t1, 1),
                             sizes={r.key: sz for (r, _), sz in zip(members, sizes)},
                             stops={r.key: st for (r, _), st in zip(members, stops)},
                             fit={k: {g: round(v, 3) for g, v in fv.items()} for k, fv in fit.items()}))
            return fit
        open_of = lambda r: [i for i, g in enumerate(r.goal_list)
                             if g.name not in out[r.key] or out[r.key][g.name][0]["found_cost"] is None]
        try:                                            # (the vocabularies are shared by every root: built under the
            comp = self.comp_classes(deadline)          #  training's deadline; one search's limit starts after)
            if self.search_s:
                dl = min(deadline or float("inf"), time.time() + self.search_s)
            if deep and self.deep_focus:                # deeper where the closest attempt was, with what it used:
                by = {}                                 # the primitives and its routines (a big library leaves even
                for r in group:                         # a 5.8M search no reach past cost 4)
                    fs = set()
                    for i in open_of(r):
                        name = r.goal_list[i].name
                        if self.deep_by == "bindings":  # what the closest attempts used at any of its bindings
                            for k, fo in self._focus.items():
                                if k.split("@")[0] == r.pattern:
                                    fs |= fo.get(name, set())
                        else:
                            fs |= self._focus.get(r.key, {}).get(name, set())
                    by.setdefault(tuple(sorted(fs)), []).append(r)
                for fkey, rs in by.items():
                    if fkey not in self._focus_classes:
                        if len(self._focus_classes) > 16:
                            self._focus_classes.clear()
                        self._focus_classes[fkey] = enumerate_items(list(PRIMS) + list(fkey), self.lib, self.settings,
                                                                    self.device, max_body=self.comp_body,
                                                                    deadline=deadline, cache=self._sig_cache,
                                                                    chunk=self.chunk * self.fuse)
                    if self.search_s:
                        dl = min(deadline or float("inf"), time.time() + self.search_s)
                    fit = run(f"deep {list(fkey)}", self._focus_classes[fkey], self.lib, self.deep_budget,
                              [(r, open_of(r)) for r in rs])
                    for r in rs:
                        self._fit[r.key] = fit[r.key]
            elif deep:
                fit = run("deep", comp, self.lib, self.deep_budget, [(r, open_of(r)) for r in group])
                for r in group:
                    self._fit[r.key] = fit[r.key]
            else:
                fit = run("routines", comp, self.lib, self.probe_budget,
                          [(r, list(range(len(r.goal_list)))) for r in group])
                fb = [(r, open_of(r)) for r in group if open_of(r) and r.key not in self.fallback_seen]
                if fb and self.fallback_budget and self.routines():
                    prims = self.prim_classes(deadline)
                    if self.search_s:
                        dl = min(deadline or float("inf"), time.time() + self.search_s)
                    fit2 = run("primitives", prims, {}, self.fallback_budget, fb)
                    for r, _ in fb:
                        fit[r.key] = {k: max(v, fit2[r.key].get(k, 0.0)) for k, v in fit[r.key].items()}
                for r in group:
                    self.fallback_seen.add(r.key)
                    self._fit[r.key] = {k: v for k, v in fit[r.key].items() if out[r.key][k][0]["found_cost"] is None}
        except Deadline:
            pass
        rec = dict(roots=[r.key for r in group], level=self.level, seconds=round(time.time() - t0, 1), runs=runs,
                   deep=deep, laws={f"{g}/{k}": dict(found_cost=x["found_cost"], how=how,
                                                    programs=[show(p) for p in (x["accepted"] or [])][:8])
                                    for k, o in out.items() for g, (x, how) in o.items()})
        for r in group:
            for g, (x, how) in out[r.key].items():
                self.results[f"{g}/{r.key}"] = dict(x, how=how, root=r)
        return rec

    def groups(self, roots, size):
        """Roots in groups for one batched search each: the same probe shapes, at most size roots, pool order."""
        by = {}
        for r in roots:
            by.setdefault(tuple(b.regs.shape for b in r.roots), []).append(r)
        out = []
        for rs in by.values():
            out += [rs[i:i + size] for i in range(0, len(rs), size)]
        return out

    def deep_roots(self, roots):
        """The deep probes' roots: the deep_k roots whose closest open law is closest (ties in pool order), among those
        with an open law that the probe brought closer than one instruction does. With deep_by="concept", for each
        open concept (a law on a pattern) its closest binding's root, the deep_k closest such concepts' roots. With
        deep_by="bindings", the closest concepts whole: every binding where the concept is still open, concept by
        concept until deep_k roots are taken (the closest binding may be one no short program fits, as an operand that
        must first be moved out of A)."""
        if self.deep_by == "bindings":
            best = {}
            for i, r in enumerate(roots):
                for law, f in self._fit.get(r.key, {}).items():
                    k = (law, r.pattern)
                    if f > 0 and (k not in best or f > best[k][0]):
                        best[k] = (f, i)
            out = []
            for (law, pat), _ in sorted(best.items(), key=lambda kv: (-kv[1][0], kv[1][1])):
                if len(out) >= self.deep_k:
                    break
                out += [r for r in roots if r.pattern == pat and law in self._fit.get(r.key, {}) and r not in out]
            return out
        if self.deep_by == "concept":
            best = {}
            for i, r in enumerate(roots):
                for law, f in self._fit.get(r.key, {}).items():
                    k = (law, r.pattern)
                    if f > 0 and (k not in best or f > best[k][0]):
                        best[k] = (f, i, r)
            out = []
            for f, i, r in sorted(best.values(), key=lambda c: (-c[0], c[1])):
                if r not in out:
                    out.append(r)
            return out[:self.deep_k]
        cand = [(max(self._fit[r.key].values()), i, r) for i, r in enumerate(roots)
                if self._fit.get(r.key) and max(self._fit[r.key].values()) > 0]
        cand.sort(key=lambda c: (-c[0], c[1]))
        return [r for _, _, r in cand[:self.deep_k]]

    # ------------------------------------------------------------------ choose
    def classify(self):
        """status per pool candidate: trivial / learnable / alias / duplicate / unsolved, with its chosen program."""
        status, learn_sig, learn_rc = {}, {}, {}
        for name in self.pool:
            r = self.results.get(name)
            if r is None or r["found_cost"] is None:
                status[name] = dict(status="unsolved")
                continue
            lib = {} if r["how"] == "primitives" else self.lib
            progs = self.choose_routine(r["accepted"], lib, r["root"])
            st = dict(status="trivial" if r["found_cost"] <= 1 else "learnable", cost=r["found_cost"],
                      prog=progs[0], progs=progs, lib_used=lib is self.lib)
            if st["status"] == "learnable" and self.rowcheck:          # compose2: ROWCHECK (ROWCHECK.md)
                s = self.signature(progs[0], self.lib)
                st["status"], of = H.row_check(self, name, s, learn_rc, status)
                if of is not None:
                    st["of"] = of
            elif st["status"] == "learnable":
                s = self.signature(progs[0], self.lib)
                if s in self.tokens_sig[2]:
                    st["status"] = "alias"
                elif s in learn_sig:
                    st["status"], st["of"] = "duplicate", learn_sig[s]
                else:
                    learn_sig[s] = name
            status[name] = st
        return status

    def choose_routine(self, progs, lib, root):
        """Among the shortest programs: one that leaves its inputs as they were and changes the fewest registers, then
        Occam (fewest primitives written, then fewest executed)."""
        st, r0 = self.settings[-1], root.roots[-1]

        def key(p):
            keeps, touched = side_effects(p, lib, st, r0, root.binding)
            return (not keeps, touched, cost(expand(p, lib)), steps(p, lib))
        best = min(progs, key=key)
        return [best] + [p for p in progs if p != best]

    def learnable(self, status):
        return [n for n in self.pool if status[n]["status"] == "learnable"]

    def hardest(self, status, k):
        """The k costliest learnable candidates (bar 3's lp order: ties in pool order)."""
        return sorted(self.learnable(status),                         # compose2: ties (SUBGOAL, TIEBREAK)
                      key=lambda n: (-status[n]["cost"],) + H.tie_rank(self, status, [n]))[:k]

    def look_jobs(self, status):
        """The cheaper pairs signal's lookaheads (bar3/breadth/cheap.py, anchor_cost 2): every learnable alone, and
        every pair with a member among the 2 hardest."""
        learn = self.learnable(status)
        top = set(self.hardest(status, 2))
        jobs = [(n,) for n in learn]
        jobs += [(a, b) for a, b in itertools.combinations(learn, 2) if a in top or b in top]
        return jobs

    def look(self, rows, answers, deadline, extra):
        """One lookahead: the target searched with extra routines (the small target, or the round), at look_budget."""
        if self.target is not None:
            return self.target.search(self, rows, self.look_budget, deadline, extra=extra)[1]
        return self.search_round(answers, deadline, extra=extra, budget=self.look_budget, t=self.t)[1]

    def mdl(self, rec, n_symbols):
        """Bits to describe the target's answers with the best program found (the small target: bar 3's mdl, from
        each level's best fit; the round: every probe wrong when nothing is found, since its search records only
        whether a level reaches every answer)."""
        if "fits" in rec:
            return self.target.mdl(self, rec, n_symbols)
        exc = math.log2(8) + self.P_round * math.log2(10)
        if rec["found_cost"] is not None:
            return rec["found_cost"] * math.log2(n_symbols)
        return 8 * exc

    def unlockers(self, status, rows):
        """The laws whose practice would let the learner pose a kind of problem (a law on a pattern) it cannot pose
        now: for each law and pattern with a learnable candidate and no routine yet, its routine is tried in the
        constructions of every law and pattern not posed now. Returns {(law, pattern): [what it unlocks]}."""
        posed = {(n.split("/")[0], n.split("/")[1].split("@")[0]) for n in list(self.pool) + list(self.practiced)}
        want = [(law, pat) for pat in self.patterns for law in self.laws
                if pat in LAW_BOOK[law][1] and (law, pat) not in posed]
        out = {}
        for n in self.learnable(status):
            law, rk = n.split("/")
            pat, bd = rk.split("@")
            if self.ops.has(law, pat) or (law, pat) in out or not want:
                continue
            ops = Ops()
            ops.by_key = {k: list(v) for k, v in self.ops.by_key.items()}
            ops.add(law, pat, "T.trial", self.results[n]["root"].binding)
            lib = dict(self.lib, **{"T.trial": status[n]["prog"]})
            got = []
            for l2, p2 in want:
                try:
                    r = Root(p2, bindings(len(PATTERNS[p2][0][0]))[0], self.settings, self.NP, rows, ops, lib,
                             self.device, self.seed)
                    if r.goals(ops, lib, [l2]):
                        got.append(f"{l2}@{p2}")
                except NotPosable:
                    pass
            if got:
                out[(law, pat)] = got
        return out

    def concepts_done(self):
        """The concepts (a law on a pattern) practised in some binding."""
        return {tuple(n.split("@")[0].split("/")) for n in self.practiced if n.split("/")[0] in LAW_BOOK}

    def concepts(self, status, k, ties=False):
        """The k hardest concepts not yet practised in any binding (a concept: a law on a pattern; its hardness: the
        cost of its cheapest learnable binding, so that an awkward binding does not make a concept look hard), each
        in its cleanest learnable binding (the fewest registers touched besides A, then the cheapest, then pool
        order)."""
        done = {tuple(n.split("@")[0].split("/")) for n in self.practiced}
        by = {}
        for i, n in enumerate(self.learnable(status)):
            law, rk = n.split("/")
            pat = rk.split("@")[0]
            if (law, pat) in done:
                continue
            r = self.results[n]["root"]
            lib = {} if self.results[n]["how"] == "primitives" else self.lib
            _, touched = side_effects(status[n]["prog"], lib, self.settings[-1], r.roots[-1], r.binding)
            by.setdefault((law, pat), []).append((touched, status[n]["cost"], i, n))
        ranked = sorted(by.values(), key=lambda v: (-min(c for _, c, _, _ in v),)   # compose2: ties
                        + H.tie_rank(self, status, [n for *_, n in v]) + (min(i for _, _, i, _ in v),))
        if not ties:
            return [min(v)[3] for v in ranked[:k]]
        return [n for v in ranked[:k] for t, c, _, n in sorted(v) if (t, c) == min(v)[:2]]

    def pick(self, status, rows=None):
        if self.rule == "all":
            return self.learnable(status)
        if self.rule == "lp":
            return self.hardest(status, self.K)
        if self.rule in ("lpc", "lpca"):              # concepts: a law on a pattern, hardest by its cheapest binding,
            got = self.concepts(status, self.K, ties=self.rule == "lpca")     # practised once, in its cleanest
            if got or not self._variants:             # binding (lpca: in every binding tied for cleanest). A practised
                self._want_variants = not got         # concept's other bindings are not posed; when no new concept is
                return got                            # learnable, the next level poses them, and picks the hardest
            return self.hardest(status, self.K)
        if self.rule == "lpu":                        # unlockers first (each its hardest candidate), then hardest
            un = self.unlockers(status, rows)
            self._scores = {"unlocks": {f"{k[0]}@{k[1]}": v for k, v in un.items()}}
            first = []
            for n in self.hardest(status, len(self.learnable(status))):
                law, rk = n.split("/")
                key = (law, rk.split("@")[0])
                if key in un and all(not m.startswith(f"{law}/{key[1]}@") for m in first):
                    first.append(n)
            return (first + [n for n in self.hardest(status, self.K + len(first)) if n not in first])[:max(self.K,
                                                                                                         len(first))]
        if self.rule == "ct2a2":
            n_sym = len(PRIMS) + len(self.routines())
            base = self.mdl(self._base_look, n_sym)
            singles = {j[0]: base - self.mdl(r, n_sym + 1) for j, r in self.looks if len(j) == 1}
            pairs = sorted(((base - self.mdl(r, n_sym + 2), j) for j, r in self.looks if len(j) == 2),
                           key=lambda p: (-p[0],) + H.prior_rank(self, p[1]))     # reading: PRIOR
            chosen = []
            for g, j in pairs:
                for n in j:
                    if n not in chosen and len(chosen) < self.K:
                        chosen.append(n)
                if len(chosen) >= self.K:
                    break
            for n in sorted(singles, key=lambda n: (-singles[n],) + H.prior_rank(self, (n,))):   # reading: PRIOR
                if len(chosen) < self.K and n not in chosen:
                    chosen.append(n)
            self._scores = dict(singles=singles, pairs=[(round(g, 1), list(j)) for g, j in pairs[:20]])
            return chosen
        raise ValueError(self.rule)

    # ------------------------------------------------------------------ sleep
    def sleep(self, picks, status):
        mlog = []
        for n in picks:
            self.corpus[n] = list(status[n]["progs"][:8])
            self.practiced[n] = self.level
        for n in self.pool:                          # a pick is not posed again; nor is a trivial problem
            if status[n]["status"] == "trivial":
                self.practiced.setdefault(n, self.level)
        self.corpus, self.lib, new = bpe(self.corpus, self.lib, max_merges=self.max_merges, min_count=self.min_count,
                                         prefix=f"B{self.level}.", log=mlog)
        for n in picks:                              # each pick's routine: its chosen program, rewritten by the merges
            prog = self.corpus[n][0]
            name = "T." + n.replace("/", ".").replace("@", ".")
            if cost(expand(prog, self.lib)) >= 2 and name not in self.lib:
                self.lib[name] = prog
                fn = _apply_routine(prog, name)
                self.corpus = {u: [rewrite(p, fn) for p in ps] for u, ps in self.corpus.items()}
                mlog.append({"name": name, "def": show(prog), "routine": True})
                law, rk = n.split("/")
                pat, _ = rk.split("@")
                self.ops.add(law, pat, name, self.results[n]["root"].binding)
        return mlog

    # ------------------------------------------------------------------ the target (the round)
    def search_round(self, answers, deadline=None, extra=None, budget=None, t=1):
        """The round by composition (libcore.Climb.search_round): the routines and the straight-line primitives at
        the task's full size, the goal the answers. The answers are rows of one T, the smallest the batches showed
        (t): for t = 1 the search's goal is the answer; for t > 1 no state of one round shows it, so every program the
        search lists (one per distinct state, up to T_LIST a level) is run t times and kept if that gives every
        answer."""
        t0 = time.time()
        lib = dict(self.lib, **(extra or {}))
        routines = [k for k in lib if k.startswith("T.")]
        prob = round_problem(self.cells, answers, 8, self.P_round, self.device)
        try:
            classes = enumerate_items(list(PRIMS) + routines, lib, prob.settings, self.device, max_body=0,
                                      deadline=deadline)
            if t == 1:
                res = bfs(prob, classes, lib, max_cost=self.round_cost, max_states=budget or self.round_budget,
                          accept=lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device)
                                             for p in ps], deadline=deadline, chunk=max(1, self.chunk // 4),
                          fuse=self.fuse)
                found, accepted, sizes, stop = res.found_cost, res.accepted, res.level_sizes, res.stop
                if found is None and len(routines) > self.macro_after and not extra:
                    mres = bfs(prob, self.macro_classes(lib), lib, max_cost=self.macro_cost,
                               max_states=budget or self.round_budget,
                               accept=lambda ps: [round_ok(p, lib, self.cells, answers, self.P_round, self.device)
                                                  for p in ps], deadline=deadline, chunk=max(1, self.chunk // 4),
                               fuse=self.fuse)
                    stop = f"{stop}; by calls: levels {mres.level_sizes} {mres.stop}"
                    if mres.found_cost is not None:
                        found, accepted = f"{mres.found_cost} calls", [unwrap(p) for p in mres.accepted]
            else:
                found, accepted, sizes, stop = self._round_many(prob, classes, lib, answers, t, budget, deadline)
        except Deadline:
            found, accepted, sizes, stop = None, None, [], "deadline (building the vocabulary)"
        rec = dict(task="round", level=self.level, found_cost=found, level_sizes=sizes, t=t,
                   seconds=round(time.time() - t0, 1), stop=stop, routines=len(routines),
                   answers=int(answers[0].shape[0]), programs=[show(p) for p in (accepted or [])][:8])
        return accepted or [], rec

    T_LIST = 4096

    def macro_classes(self, lib):
        """The round's search by calls (a big library leaves the instruction-by-instruction search no reach): one item
        per routine and per routing of its inputs (each from A, X, Y, N or zero, by route), and the instructions that
        act on whole registers; every item costs 1."""
        bind = {tok: "".join(bd) for v in self.ops.by_key.values() for tok, bd in v}
        items = [ItemClass(w, 1) for w in WHOLE]
        seen = set()
        for tok in sorted(k for k in lib if k.startswith("T.") and k in bind):
            bd = bind[tok]
            for srcs in itertools.product(SOURCES, repeat=len(bd)):
                code = route(bd, srcs)
                if code is None:
                    continue
                it = ("REP", ONE, tuple(code) + (tok,))
                if it not in seen:
                    seen.add(it)
                    items.append(ItemClass(it, 1))
        return items

    def _round_many(self, prob, classes, lib, answers, t, budget, deadline):
        never = Goal("never", [{"c": torch.full((r.c.shape[1],), 2, dtype=torch.long, device=self.device),
                                "cm": torch.ones(r.c.shape[1], dtype=torch.bool, device=self.device)}
                               for r in prob.roots])
        _, sizes, stop, levels, parents = bfs_multi(prob.roots, prob.settings, [never], [lambda ps, lb: [False] * len(ps)],
                                                    classes, lib, max_cost=self.round_cost,
                                                    max_states=budget or self.round_budget, deadline=deadline,
                                                    chunk=max(1, self.chunk // 4), fuse=self.fuse, keep=True)
        for d in range(1, len(parents)):
            if parents[d] is None:
                continue
            n = int(parents[d][0].max()) + 1 if parents[d][0].numel() else 0
            progs = []
            for i in range(min(n, self.T_LIST)):
                f = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
                if f:
                    progs.append(f[0])
            good = [p for p in progs if round_ok_t(p, lib, self.cells, answers, self.P_round, self.device, t)]
            if good:
                return d, good, sizes, stop
        return None, None, sizes, stop or "no program of one round gives the answers in t rounds"

    # ------------------------------------------------------------------ one unit
    def step(self, rows, answers=None, deadline=None, t=1):
        """One unit of work; answers (Nd, Xd, Yd) are rows of one T, t (the smallest the batches showed)."""
        self.t = t
        if not self.events:
            line = self._step(rows, answers, deadline)
            if line is not None:
                self.events.append(line)
        return self.events.pop(0) if self.events else None

    def _step(self, rows, answers, deadline):
        if self.done:
            if not (self.round_retry and answers is not None and not self.rounds and self.routines()):
                return None
            self._idle += 1
            if self._idle < self.round_retry:
                return None
            self._idle = 0
            progs, rec = self.search_round(answers, deadline, t=self.t)
            rec["retry"] = True
            self.records.append(rec)
            self.rounds = progs
            return (f"L{self.level} round again ({rec['answers']} answer rows, t {self.t}): cost {rec['found_cost']} "
                    f"{rec['seconds']}s {rec['stop']} {rec['programs'][:3]}")
        if self.phase == "pose":
            if self.level >= self.levels:
                self.done = True
                return f"climb done at level {self.level} (the level cap)"
            if self.target is not None:
                _, rec, self._near = self.target.search(self, rows, self.look_budget, deadline, near=12)
                self.near_log.append(dict(level=self.level, fits=rec["fits"], near=rec["near"]))
                self.records.append(rec)
            self.pool, roots = self.pose(rows)
            H.subgoal_scores(self, rows, roots)                         # compose2: SUBGOAL
            self.results, self._fit, self._focus = {}, {}, {}
            self.queue, self._level_roots = self.groups(roots, self.group), list(roots)
            self.phase = "probe"
            return f"L{self.level} pose: {len(self.pool)} candidates on {len(roots)} roots"
        if self.phase == "probe":
            if self.queue:
                group = self.queue.pop(0)
                rec = self.probe(group, rows, deadline)
                self.records.append(rec)
                solved = {g: v["found_cost"] for g, v in rec["laws"].items() if v["found_cost"] is not None}
                return (f"L{self.level} probe {len(group)} roots {rec['seconds']}s solved {solved} "
                        f"{[(r['how'], r['seconds']) for r in rec['runs']]}")
            if self.deep_k and self.deep_budget and self.routines():
                deep = self.deep_roots(self._level_roots)
                self.queue = self.groups(deep, self.deep_group)
                self.phase = "deep"
                return f"L{self.level} deep: {[(r.key, self._fit[r.key]) for r in deep]}"
            self.phase = "classify"
        if self.phase == "deep":
            if self.queue:
                group = self.queue.pop(0)
                rec = self.probe(group, rows, deadline, deep=True)
                self.records.append(rec)
                solved = {g: v["found_cost"] for g, v in rec["laws"].items() if v["found_cost"] is not None}
                return (f"L{self.level} deep {[r.key for r in group]} {rec['seconds']}s solved {solved} "
                        f"{[(r['how'], r['seconds']) for r in rec['runs']]}")
            self.phase = "classify"
        if self.phase == "classify":
            line = H.subgoal_deep(self, rows, deadline)               # compose2: SUBGOAL's deeper probe
            if line is not None:
                return line
            self._status = self.classify()
            self.phase = "choose"
            if self.rule == "ct2a2" and (answers is not None or self.target is not None) and self.learnable(self._status):
                self.queue = self.look_jobs(self._status)
                self.looks = []
                self._base_look = self.look(rows, answers, deadline, {})
                self.phase = "look"
                return (f"L{self.level} look: {len(self.queue)} lookaheads; the target alone: cost "
                        f"{self._base_look['found_cost']}")
        if self.phase == "look":
            if self.queue:
                job = self.queue.pop(0)
                extra = {f"T.look{i}": self._status[n]["prog"] for i, n in enumerate(job)}
                rec = self.look(rows, answers, deadline, extra)
                self.looks.append((job, rec))
                return f"L{self.level} look {job}: cost {rec['found_cost']} {rec['seconds']}s"
            self.phase = "choose"
        if self.phase == "choose":
            status = self._status
            if self.rule == "ct2a2" and not self.looks:
                picks = self.hardest(status, self.K)
            else:
                picks = self.pick(status, rows)
            counts = Counter(v["status"] for v in status.values())
            self.history.append(dict(level=self.level, pool=len(self.pool), counts=dict(counts), picks=list(picks),
                                     learnable={n: (status[n]["cost"], show(status[n]["prog"])) for n in self.pool
                                                if status[n]["status"] == "learnable"},
                                     scores=self._scores if self.rule in ("ct2a2", "lpu") else {}))
            self.picks = picks
            self.phase = "sleep"
            self.looks = []
            return (f"L{self.level} choose ({self.rule}): {dict(counts)}; picks "
                    f"{[(n, status[n]['cost'], show(status[n]['prog'])) for n in picks]}")
        if self.phase == "sleep":
            mlog = self.sleep(self.picks, self._status)
            self.history[-1]["merges"] = mlog
            self.history[-1]["lib"] = {k: show(v) for k, v in self.lib.items()}
            self.phase = "target"
            self._variants = self.rule not in ("lpc", "lpca") or self._want_variants
            self._want_variants = False
            if not self.picks and not self._variants:
                self.done = True
                return f"L{self.level} sleep: nothing learnable to practise; climb done"
            if not self.picks:
                return f"L{self.level} sleep: no new concept learnable; the next level poses practised ones' variants"
            return f"L{self.level} sleep: " + "; ".join(f"{m['name']}={m['def']}" for m in mlog)
        if self.phase == "target":
            self.phase = "pose"
            lvl = self.level
            self.level += 1
            if self.target is not None:
                progs, rec, _ = self.target.search(self, rows, self.round_budget, deadline)
                if progs and answers is not None:           # is the small target's program the round at full size?
                    rec["full_size"] = [bool(round_ok(p, self.lib, self.cells, answers, self.P_round, self.device))
                                        for p in progs[:8]]
                self.records.append(rec)
                if progs:
                    self.rounds = progs
                    self.done = True
                return (f"L{lvl} target: cost {rec['found_cost']} levels {rec['level_sizes']} fits {rec['fits']} "
                        f"{rec['seconds']}s {rec['stop']} {rec['programs'][:3]} full size {rec.get('full_size')}")
            if answers is None or not self.routines():
                return f"L{lvl} target: no answers yet"
            progs, rec = self.search_round(answers, deadline, t=self.t)
            self.records.append(rec)
            if progs:
                self.rounds = progs
                self.done = True
            return (f"L{lvl} round: cost {rec['found_cost']} levels {rec['level_sizes']} {rec['seconds']}s "
                    f"{rec['stop']} {rec['programs'][:3]}")
        raise ValueError(self.phase)
