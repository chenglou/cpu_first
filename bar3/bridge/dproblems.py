"""Candidate practice problems at the digit level: bar3/problems.py's three generic sources, posed on slices
(slices.py) instead of on x^2 mod N's pairs.

A problem is a function from a place's inputs (registers A, R, C, H; N is the base) to its outputs, given as rows per
base. Sources, as in bar 3:
- law: every slice of the digit law book (every law in every binding), except the target;
- restrict: the target's rows where t1 < t2, for terms 0, 1, 2, each input, the base N and the sums of two inputs
  (bar 3's grammar, with 0 added because a flag is 0 or 1); one per distinct subset of rows, proper and not too small;
- project: each output of the target alone;
- value / remainder: the learner's own near misses at the target, run item by item: every value one of their registers
  holds after an item (neither an input nor constant) becomes a problem of its own, and the rest of the way from it
  (the target's outputs with A replaced by that value) becomes one when it is a function, with its restrictions.
A problem is kept when it has at least 4 rows at the search bases and at the check bases, its answers vary, and it is
a function of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from library import dsl
from bar3.bridge import dcm, slices as S

KEY = "ARCH"                     # a row's key: the values of A, R, C, H (N is the base)


def machine_for(bases):
    """bar 3's machine_for(k) with k the largest base: values saturate at 2 k^2, a run may take 50 k^2 + 1000 steps."""
    k = max(bases)
    return dcm.Machine(cap=max(64, 2 * k * k), limit=50 * k * k + 1000)


@dataclass
class Cand:
    name: str
    source: str                  # law | restrict | project | value | remainder | target
    formula: str
    inputs: tuple                # input registers (N always among them)
    outputs: tuple               # output registers
    rows: dict                   # base -> {key (A, R, C, H): output tuple}
    parent: str = ""
    also: list = field(default_factory=list)

    def regs(self):
        """The problem's machine: its inputs and outputs, the base, and one scratch register."""
        return tuple(r for r in dcm.REGS if r in set(self.inputs) | set(self.outputs) | {"N", "T"})

    def key(self):
        return (self.inputs, self.outputs, tuple((B, tuple(sorted(r.items()))) for B, r in sorted(self.rows.items())))

    def n(self, bases):
        return sum(len(self.rows.get(B, {})) for B in bases)

    def to_dict(self):
        return dict(name=self.name, source=self.source, formula=self.formula, inputs=self.inputs, outputs=self.outputs,
                    parent=self.parent, also=self.also, n=self.n(S.SEARCH_BASES), n_check=self.n(S.CHECK_BASES))


def well_posed(c, min_n=4):
    for bases in (S.SEARCH_BASES, S.CHECK_BASES):
        if c.n(bases) < min_n:
            return False
    outs = {o for B in S.SEARCH_BASES for o in c.rows.get(B, {}).values()}
    return len(outs) >= 2


def from_slice(sl, source="law"):
    return Cand(sl.name, source, f"{sl.law} at one place, operands in {','.join(sl.binding)}", sl.inputs, sl.outputs,
                {B: dict(r) for B, r in sl.rows.items()})


def table(c, bases, restrict=None):
    """(inputs {register: [n]}, want [n, outputs], base [n]) over the rows at these bases."""
    cols = {r: [] for r in c.inputs}
    want, bs = [], []
    for B in bases:
        for key, out in sorted(c.rows.get(B, {}).items()):
            ins = dict(zip(KEY, key))
            ins["N"] = B
            for r in c.inputs:
                cols[r].append(ins[r])
            want.append(out)
            bs.append(B)
    return ({r: np.array(v, np.int64) for r, v in cols.items()},
            np.array(want, np.int64).reshape(len(want), len(c.outputs)), np.array(bs, np.int64))


def sample(c, n=32, seed=0):
    """At most n rows at the search bases for the search itself (bar 3 samples 32): two from every base, the rest
    at random."""
    ins, want, bs = table(c, S.SEARCH_BASES)
    k = len(bs)
    if k <= n:
        return ins, want
    rng = np.random.default_rng(seed + k)
    keep = []
    for B in np.unique(bs):
        idx = np.nonzero(bs == B)[0]
        keep += rng.choice(idx, min(2, len(idx)), replace=False).tolist()
    rest = np.setdiff1d(np.arange(k), keep)
    keep += rng.choice(rest, n - len(keep), replace=False).tolist()
    keep = np.sort(np.array(keep))
    return {r: v[keep] for r, v in ins.items()}, want[keep]


def dedupe(cands, known=()):
    by, out = {c.key(): c for c in known}, []
    for c in cands:
        k = c.key()
        if k in by:
            by[k].also.append(c.name)
            continue
        by[k] = c
        out.append(c)
    return out


# ---------------------------------------------------------------------------------------------------------------
# restrictions and projections

def terms(inputs):
    out = {"0": lambda v: 0, "1": lambda v: 1, "2": lambda v: 2}
    ins = [r for r in inputs if r != "N"]
    for r in ins + ["N"]:
        out[r] = lambda v, r=r: v[r]
    for i, a in enumerate(ins):
        for b in (ins + ["N"])[i:]:
            out[f"{a}+{b}"] = lambda v, a=a, b=b: v[a] + v[b]
    return out


def restrictions(c, prefix="restrict", min_n=6, max_share=0.8):
    """c's rows where t1 < t2, for every ordered pair of terms; one per distinct subset of the search rows."""
    out, seen = [], set()
    T = terms(c.inputs)
    srows = [(B, key) for B in S.SEARCH_BASES for key in sorted(c.rows.get(B, {}))]
    for a in T:
        for b in T:
            if a == b or (a in "012" and b in "012"):
                continue
            keep = lambda B, key, a=a, b=b: T[a]({**dict(zip(KEY, key)), "N": B}) < T[b]({**dict(zip(KEY, key)),  # noqa
                                                                                        "N": B})
            sig = tuple(i for i, (B, key) in enumerate(srows) if keep(B, key))
            if len(sig) < min_n or len(sig) > max_share * len(srows) or sig in seen:
                continue
            seen.add(sig)
            rows = {B: {key: o for key, o in r.items() if keep(B, key)} for B, r in c.rows.items()}
            r = Cand(f"{prefix}/{c.name}/{a}<{b}", "restrict", f"{c.name} where {a} < {b}", c.inputs, c.outputs, rows,
                     parent=c.name)
            if well_posed(r):
                out.append(r)
    return out


def projections(c):
    """Each output of c alone (bar 3's projections take parts of one answer; here the answer has parts already)."""
    if len(c.outputs) < 2:
        return []
    out = []
    for j, o in enumerate(c.outputs):
        rows = {B: {key: (v[j],) for key, v in r.items()} for B, r in c.rows.items()}
        p = Cand(f"project/{c.name}/{o}", "project", f"{c.name}: its {o} alone", c.inputs, (o,), rows, parent=c.name)
        if well_posed(p):
            out.append(p)
    return out


# ---------------------------------------------------------------------------------------------------------------
# the learner's own near misses

def _values(prog, lib, c, B):
    """Every register's value after each top-level item of prog, on c's rows at base B: (keys, {(item, reg): v})."""
    keys = sorted(c.rows.get(B, {}))
    if not keys:
        return keys, {}
    ins = {r: np.array([dict(zip(KEY, k))[r] for k in keys], np.int64) for r in c.inputs if r != "N"}
    ins["N"] = np.full(len(keys), B, np.int64)
    b = dcm.fresh(ins)
    m = machine_for(S.GENERAL_BASES)
    out = {}
    for i, it in enumerate(prog):
        dcm.run_items(m, b, (it,), lib)
        if b.dead[0]:
            return keys, None
        for r in dcm.REGS:
            out[(i, r)] = b.regs[0, :, dcm.RI[r]].astype(np.int64).copy()
    return keys, out


def from_near_misses(near, lib, c):
    """near: [(fit, cost, program text)] at the target c. Values and remainders, as bar3/problems.from_near_misses."""
    out = []
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    for fit, cost, ptxt in near:
        prog = dsl.parse(ptxt) if ptxt else ()
        if not prog:
            continue
        per = {B: _values(prog, pl, c, B) for B in c.rows}
        if any(v is None for _, v in per.values()):
            continue
        slots = set()
        for keys, vals in per.values():
            slots |= set(vals)
        for (i, r) in sorted(slots):
            vrows, rrows, const, isin = {}, {}, set(), set()
            for B, (keys, vals) in per.items():
                if (i, r) not in vals:
                    continue
                v = vals[(i, r)]
                vrows[B] = {k: (int(x),) for k, x in zip(keys, v)}
                const |= set(v.tolist())
                for inp in c.inputs:
                    col = np.array([B] * len(keys)) if inp == "N" else np.array([dict(zip(KEY, k))[inp] for k in keys])
                    if not (col == v).all():
                        isin.add(inp)
                rr, bad = {}, False
                for k, x in zip(keys, v):
                    nk = (int(x),) + k[1:]
                    if rr.setdefault(nk, c.rows[B][k]) != c.rows[B][k]:
                        bad = True
                        break
                if not bad:
                    rrows[B] = rr
            if len(const) < 2 or len(isin) < len(c.inputs):        # constant, or just an input
                continue
            pre = dsl.show(prog[: i + 1])
            tag = f"{pre} -> {r}"
            val = Cand(f"value/{tag}", "value", f"the value {r} holds after `{pre}`", c.inputs, ("A",), vrows,
                       parent=c.name)
            if well_posed(val):
                out.append(val)
            if len(rrows) == len(c.rows):
                rem = Cand(f"remainder/{tag}", "remainder", f"{c.name}'s outputs from A = the value {r} holds after "
                           f"`{pre}`", tuple(sorted(set(c.inputs) | {"A"}, key=dcm.REGS.index)), c.outputs, rrows,
                           parent=c.name)
                if well_posed(rem):
                    out.append(rem)
                    out += restrictions(rem, prefix="restrict")
    return out


# ---------------------------------------------------------------------------------------------------------------
# bar 3's law book on the digit-level data (bar3/problems.py's laws): its operations applied to a place's own values

OPS2 = {
    "add": (lambda a, b: a + b, "{a} + {b}"),
    "monus": (lambda a, b: max(a - b, 0), "{a} ∸ {b}"),
    "absdiff": (lambda a, b: abs(a - b), "|{a} − {b}|"),
    "mul": (lambda a, b: a * b, "{a}·{b}"),
    "div": (lambda a, b: a // b if b else None, "{a} div {b}"),
    "mod": (lambda a, b: a % b if b else None, "{a} mod {b}"),
    "min": (lambda a, b: min(a, b), "min({a}, {b})"),
    "max": (lambda a, b: max(a, b), "max({a}, {b})"),
    "lt": (lambda a, b: int(a < b), "[{a} < {b}]"),
    "le": (lambda a, b: int(a <= b), "[{a} ≤ {b}]"),
    "eq": (lambda a, b: int(a == b), "[{a} = {b}]"),
    "gcd": (lambda a, b: __import__("math").gcd(a, b), "gcd({a}, {b})"),
}
OPS1 = {"succ": (lambda a: a + 1, "{a} + 1"), "pred": (lambda a: max(a - 1, 0), "{a} ∸ 1")}
PREDICATES = ("lt", "le", "eq")          # a predicate's answer is a flag: it goes to C (SELF_PRACTICE.md's lt, eq)
PATTERNS = {                             # what A holds, by the division law's form a + q b (bar 3's x, x + N, x + qN)
    "a": lambda a, B, q: a,
    "a+N": lambda a, B, q: a + B,
    "a+qN": lambda a, B, q: a + q * B,
}


def _coin(key, B):
    return (hash((key, B, 7)) >> 3) & 1


def digit_laws(tgt):
    """bar 3's law library applied to the target's own rows: every operation on the place's values (A, R, C, H and
    the base N) in every binding, with A built by the patterns a, a + N and a + qN (q a coin); and select, the flag
    choosing between two values. Values go to A, predicates to C. Nothing here knows the target; most of it is
    irrelevant to it."""
    ops = [r for r in tgt.inputs]
    out = []

    def make(name, formula, regs, fn, outreg, pat):
        rows = {}
        for B, r in tgt.rows.items():
            rr = {}
            for key in r:
                v = dict(zip(KEY, key))
                v["N"] = B
                if "A" in regs:
                    v["A"] = PATTERNS[pat](v["A"], B, _coin(key, B))
                w = fn(v)
                if w is None:
                    continue
                k = tuple(v[x] if x in regs else 0 for x in KEY)
                if rr.setdefault(k, (w,)) != (w,):
                    return None
            rows[B] = rr
        inputs = tuple(x for x in dcm.REGS if x in regs or x == "N")
        c = Cand(name, "law", formula, inputs, (outreg,), rows)
        return c if well_posed(c) else None

    for pat in PATTERNS:
        if pat != "a" and "A" not in ops:
            continue
        for opn, (fn, words) in OPS2.items():
            for b1 in ops:
                for b2 in ops:
                    if pat != "a" and "A" not in (b1, b2):
                        continue
                    outreg = "C" if opn in PREDICATES else "A"
                    c = make(f"law/{opn}({b1},{b2})/{pat}", words.format(a=b1, b=b2) + f", A = {pat}", {b1, b2},
                             lambda v, fn=fn, b1=b1, b2=b2: fn(v[b1], v[b2]), outreg, pat)
                    if c:
                        out.append(c)
        for opn, (fn, words) in OPS1.items():
            for b1 in ops:
                if pat != "a" and b1 != "A":
                    continue
                c = make(f"law/{opn}({b1})/{pat}", words.format(a=b1) + f", A = {pat}", {b1},
                         lambda v, fn=fn, b1=b1: fn(v[b1]), "A", pat)
                if c:
                    out.append(c)
        if "C" in ops:
            for b1 in ops:
                for b2 in ops:
                    if b1 == b2 or "C" in (b1, b2) or (pat != "a" and "A" not in (b1, b2)):
                        continue
                    c = make(f"law/select(C,{b1},{b2})/{pat}", f"{b2} if C else {b1}, A = {pat}", {"C", b1, b2},
                             lambda v, b1=b1, b2=b2: v[b2] if v["C"] else v[b1], "A", pat)
                    if c:
                        out.append(c)
    return out
