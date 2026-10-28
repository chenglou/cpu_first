"""Label-verified speculation over a grown design grammar, bitsliced in torch (64 candidate machines per int64 word).

Every lane of an int64 word is a different candidate machine. A design choice is a per-lane multiplexer: compute each
alternative that some lane needs, keep the one the lane's select mask names (only alternatives present in the lane set
are computed, so a factor that is fixed costs nothing). Per-lane numeric choices (iteration counts, widths) are
predication masks. The gate is either per lane (65,536 truth tables, Shannon-expanded against per-lane table words)
or the full adder (sum = a^b^c, carry = maj).

Grammar (K and W up to 32: 2^76.3 joint machines incl. the gate with --core ports; --kw-max 28 gives the earlier
widths, where --core small is 2^60.7 and --families r2 --ppop ab --law none is the first runs' search, 2^54.4 / 2^63.9
with --core small / ports; the earlier ports grammar, 2^70.0, varied r4q1sh where it now varies r4c3sh, r4c4sh):
  io     gate (65,536) | d2b_mul (Horner v<<a + v<<b, 15) | d2b_order (2) | b2d_thr (>=3..8) | b2d_add (+0..7)
         | b2d_order (2)
  entry/exit  entry_radix (2: s<-2s-[.]N | 4: s<-4s-kN, 3 borrow-select subtractions) | Ke ({22,...,32}) | exq (exit
         quotient bit, 4) | Kx (exit halvings, {22,...,32})
  core   a two-operand Montgomery product MonPro(a, b) = a*b/2^Km mod N (squaring is MonPro(s, s)) built by one of
         four multiplier families (factor `mult`), plus factors the families share:
         shared  ppdir (2) | ppop (whose bits are scanned and whose multiples are added: a.b, b.a, or the
                 squaring-only a.a, b.b) | Km (R = 2^Km, {22,...,32}) | W (register width, {24,...,32}; every wire is
                 WMAX = 32 bits)
         r2   radix 2, two carry-save cells per bit (the reference): l1src (which of Ss, Sc, pp feeds each port of
              the first cell, 27) | l1sh (8) | qsrc (quotient bit source, 7) | l2src (which of u, v, qN feeds each
              port of the second cell, 27) | l2sh (27) | osh (sum-row hand-off >>0..2) | csh (carry-row hand-off)
         r2m  radix 2, one cell per bit on a precomputed b+N: r2mq (quotient formula, 6) | r2mrow (which of 0, b,
              N, b+N each (a_i, q) selects, 24) | osh | csh
         r4   radix 4, Km/2 iterations, four cells per 2-bit digit: r4p1sh (3) | r4hold (tree shape: which of Ss,
              Sc, P0, P1 skips the first cell, 4) | q4 (2-bit quotient from the low bits of the carry-save sum and
              N mod 4, 8) | r4q1sh (3) | r4qtree (where the two q*N rows enter, 3) | r4osh | r4csh | r4cin (the
              carry the shift by 4 leaves in bit 0, 5)
         r4m  radix 4 on precomputed 3b and 3N, two cells per digit: r4mpp (which of 0, b, 2b, 3b each digit
              selects, 24) | q4 | r4mqn (which of 0, N, 2N, 3N each quotient selects, 24) | r4osh | r4csh | r4cin
         ports mode (--core ports) adds per-port shifts (27 each) on the cells of r2m (r2msh), r4 (r4c1sh, r4c2sh,
         r4c3sh, r4c4sh, replacing the row shifts r4p1sh, r4q1sh) and r4m (r4m1sh, r4m2sh)
  control  cw (T-counter width)
  (--core small drops l1src/l2src and the per-port shifts; --families picks the multiplier families; --ppop ab fixes
  ppop)

Stages: A = gate x io against the label-free round trip b2d(d2b(v)) = v on the prompts' own numbers; B = survivors x
entry/exit against the label-free round trip exit(entry(x)) = x (the prompt's own N, x); C0 (with --law) = each
distinct entry/exit of B's survivors x the core against label-free MonPro laws on the prompts' own (N, x):
  id      exit(MonPro(entry(x), entry(1))) = x     (synthesizes the constant 1; pins Km = 2 Ke - Kx; passes every
          left projection f(a, b) = a)
  comm    MonPro(a, 1~) = MonPro(1~, a), a = entry(x), 1~ = entry(1)   (bit-exact; removes the projections)
  comm_sq MonPro(a, a^2) = MonPro(a^2, a), a^2 = MonPro(a, a)   (blind to projections: there a^2 = a)
  assoc   exit((a^2)^2) = exit(a (a a^2))
  mirror  exit(MonPro(a, a)) = exit(MonPro(a', a')), a' = entry(N - x)   (squaring only; synthesizes N - x)
  mirror2 x^4 = (N - x)^4 through two squarings   (squaring only; the second reads an unreduced operand)
  unit    exit(MonPro(1~, 1~)) = 1   (squaring only; synthesizes 1; pins Km = 2 Ke - Kx)
then C = B's survivors joined with the C0 survivors of their entry/exit x cw against T=1 labels (without --law: B's
survivors x the whole core x cw against T=1 labels, as before; with --c-check decimal: no labels, y(x, 1) must print
decimal digits, which removes the base-12/16 io aliases B keeps); C1 (with --count-law t1,t2) = the whole machine against
the semigroup law y(x, t1 + t2) = y(y(x, t1), t2) on the prompts' own (N, x) and synthesized T; D = survivors against
every training row; E = certification profiles (the aliasing audit). --law-synth also screens the MonPro laws on
synthesized Blum moduli N = pq (p = q = 3 mod 4) of a given bit length.

Per-cell laws (CELL_LAWS): --cell-law screens each family's wiring sub-block by sub-block before C0, on synthesized
words: every carry-save cell conserves value with one scale per row class (u + 2v = 2^j x (its state and quotient
rows + 2^e x its digit rows) at natural weight, j per cell, e per iteration), each iteration's quotient zeroes the low
bits and its hand-off divides exactly; C0 then sees only the product of the sub-block survivors. --width-law BITS
drops C0's survivors that can lose a bit anywhere in the core for moduli below 2^BITS (width_law: a bound over the
declared range, not samples; computed once per distinct setting of the factors it reads, width_verdicts). --cell-audit
RUN_JSON runs both on a previous run's stage-E machines; --witness N adds declared-range witness rows (witness_rows).

--collapse groups the machines that reach D into classes the laws make equivalent (design_key: machines that differ
only in their shifts, i.e. the per-cell laws' scale classes, in the order of r2's port sources, and, when every one
passed the width law, in (Ke, Kx, Km, W)); D and E then run on one representative per class, and --collapse-check K
runs E on K random non-representatives per class, each of which must certify exactly as its representative does.
--collapse C1 groups stage C's survivors the same way before C1. design_key keeps cw (the counter, which the semigroup
law tests), and the rest of a class computes the same squarings, so C1 runs on one representative per (class, cw) and
keeps the members of the groups whose representative passes; --collapse-check K then also runs C1 on K random
non-representatives per group.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

WMAX, KMAX, ND, NB, TWMAX = 32, 32, 8, 8, 7  # word width (every wire; W masks registers) and scanned multiplier bits
FA_ID = 0x96 | (0xE8 << 8)
DATA = os.environ.get("H1_DATA", "")
LIMITS: dict[str, list[int]] = {}  # option indices a run allows per factor (--kw-max caps K and W); Space.product reads it

GRAMMAR = {
    "gate": list(range(1 << 16)),
    "d2b_mul": [(a, b) for a in range(5) for b in range(a + 1)],
    "d2b_order": ["msb_first", "lsb_first"],
    "b2d_thr": [3, 4, 5, 6, 7, 8],
    "b2d_add": list(range(8)),
    "b2d_order": ["msb_first", "lsb_first"],
    "entry_radix": [2, 4],
    "Ke": [22, 24, 26, 28, 30, 32],
    "exq": ["s^c", "s", "c", "s&c"],
    "Kx": [22, 24, 26, 28, 30, 32],
    "mult": ["r2", "r2m", "r4", "r4m"],
    "ppdir": ["lsb_first", "msb_first"],
    "ppop": ["a.b", "b.a", "a.a", "b.b"],
    "l1src": list(itertools.product(range(3), repeat=3)),
    "l1sh": list(itertools.product((0, 1), repeat=3)),
    "qsrc": ["u", "v", "u^v", "Ss", "Sc", "Ss^Sc", "~u"],
    "l2src": list(itertools.product(range(3), repeat=3)),
    "l2sh": list(itertools.product((0, 1, 2), repeat=3)),
    "osh": [0, 1, 2],
    "csh": [0, 1, -1],
    "r2mq": ["s^c^db0", "s^c", "s^db0", "c^db0", "db0", "~(s^c^db0)"],
    "r2mrow": list(itertools.permutations(("0", "b", "N", "b+N"))),
    "r4p1sh": [0, 1, 2],
    "r4hold": ["Ss", "Sc", "P0", "P1"],
    "q4": ["gen", "n1", "n3", "x1", "lo", "~S0", "swap", "S0S0"],
    "r4q1sh": [0, 1, 2],
    "r4qtree": ["(u,v,Q0)(Q1)", "(u,Q0,Q1)(v)", "(v,Q0,Q1)(u)"],
    "r4osh": [1, 2, 3],
    "r4csh": [0, 1, 2],
    "r4cin": ["u1", "v0", "u1&v0", "u1|v0", "0"],
    "r4mpp": list(itertools.permutations(("0", "b", "2b", "3b"))),
    "r4mqn": list(itertools.permutations(("0", "N", "2N", "3N"))),
    "r2msh": list(itertools.product((0, 1, 2), repeat=3)),  # per-port shifts of a family's carry-save cells
    "r4c1sh": list(itertools.product((0, 1, 2), repeat=3)),
    "r4c2sh": list(itertools.product((0, 1, 2), repeat=3)),
    "r4c3sh": list(itertools.product((0, 1, 2), repeat=3)),
    "r4c4sh": list(itertools.product((0, 1, 2), repeat=3)),
    "r4m1sh": list(itertools.product((0, 1, 2), repeat=3)),
    "r4m2sh": list(itertools.product((0, 1, 2), repeat=3)),
    "Km": [22, 24, 26, 28, 30, 32],
    "W": [24, 26, 28, 30, 32],
    "cw": [5, 6, 7],
}
REF = {"gate": FA_ID, "d2b_mul": (3, 1), "d2b_order": "msb_first", "b2d_thr": 5, "b2d_add": 3, "b2d_order": "msb_first",
       "entry_radix": 4, "Ke": 26, "exq": "s^c", "Kx": 26, "mult": "r2", "ppdir": "lsb_first", "ppop": "a.b",
       "l1src": (0, 1, 2), "l1sh": (0, 0, 0), "qsrc": "u", "l2src": (0, 1, 2), "l2sh": (0, 1, 0), "osh": 1, "csh": 0,
       "r2mq": "s^c^db0", "r2mrow": ("0", "b", "N", "b+N"), "r4p1sh": 1, "r4hold": "P1", "q4": "gen", "r4q1sh": 1,
       "r4qtree": "(u,v,Q0)(Q1)", "r4osh": 2, "r4csh": 1, "r4cin": "u1", "r4mpp": ("0", "b", "2b", "3b"),
       "r4mqn": ("0", "N", "2N", "3N"), "r2msh": (0, 0, 0), "r4c1sh": (0, 0, 0), "r4c2sh": (0, 0, 0),
       "r4c3sh": (0, 0, 0), "r4c4sh": (0, 0, 0), "r4m1sh": (0, 0, 0), "r4m2sh": (0, 0, 0), "Km": 26, "W": 28, "cw": 7}
REF_IDX = {f: GRAMMAR[f].index(v) for f, v in REF.items()}
IO = ["gate", "d2b_mul", "d2b_order", "b2d_thr", "b2d_add", "b2d_order"]
ENTRY_EXIT = ["entry_radix", "Ke", "exq", "Kx"]
CORE = ["ppdir", "l1src", "l1sh", "qsrc", "l2src", "l2sh", "osh", "csh", "Km", "W", "cw"]  # the radix-2 core of the first runs
CORE_SMALL = [f for f in CORE if f not in ("l1src", "l2src")]  # the 2^54.4 grammar of the first runs
FAMILY = {  # per multiplier family, the factors only it reads (slowest-varying first)
    "r2": ["l1src", "l1sh", "qsrc", "l2src", "l2sh", "osh", "csh"],
    "r2m": ["r2msh", "r2mq", "r2mrow", "osh", "csh"],
    "r4": ["r4hold", "r4qtree", "r4c1sh", "r4c2sh", "r4c3sh", "r4c4sh", "r4p1sh", "q4", "r4q1sh", "r4osh", "r4csh",
           "r4cin"],
    "r4m": ["r4m1sh", "r4m2sh", "r4mpp", "r4mqn", "q4", "r4osh", "r4csh", "r4cin"],
}
PORTS_ONLY = {"l1src", "l2src", "r2msh", "r4c1sh", "r4c2sh", "r4c3sh", "r4c4sh", "r4m1sh", "r4m2sh"}  # port sources and
# per-port shifts
SMALL_ONLY = {"r4p1sh", "r4q1sh"}  # row shifts that r4's per-port shifts subsume
CONTROL = ["cw"]
CORE_KEYS = list(dict.fromkeys(["mult", "ppdir", "ppop"] + [f for fs in FAMILY.values() for f in fs] + ["Km", "W"]))
CONTEXT = ["gate"] + ENTRY_EXIT  # what the MonPro laws read besides the core


def family_core(fam: str, ports: bool = True) -> list[str]:
    """The MonPro block of one family: shared factors + its own (cw is control, not MonPro). `ports` adds the port
    sources (r2) and per-port shifts (r2m, r4, r4m) of every carry-save cell."""
    own = [f for f in FAMILY[fam] if (f not in PORTS_ONLY or ports) and (f not in SMALL_ONLY or not ports)]
    return ["ppdir", "ppop"] + own + ["Km", "W"]


def port_shifts(L: Lanes, f: str, ports: list):
    """Shift each port of a carry-save cell by the lane's option of f."""
    return [L.choose(f, lambda t, j=j: t[j], lambda k, j=j: shl(x, k)) for j, x in enumerate(ports)]


# ---------------------------------------------------------------- hypothesis spaces
class Space:
    """Mixed-radix product of axes. An axis is a list of factor names plus a table of option indices (one row per
    choice), so a product factor and an explicit survivor list are the same thing. Unlisted factors take `base`."""

    def __init__(self, axes: list[tuple[list[str], np.ndarray]], base: dict[str, int] | None = None) -> None:
        self.axes = [(names, np.asarray(table, dtype=np.int64).reshape(len(table), len(names))) for names, table in axes]
        self.base = dict(REF_IDX, **(base or {}))
        self.size = int(np.prod([len(t) for _, t in self.axes], dtype=object))

    @staticmethod
    def product(factors: list[str], base: dict[str, int] | None = None, options: dict[str, list[int]] | None = None):
        options = dict(LIMITS, **(options or {}))
        return Space([([f], np.array(options.get(f, range(len(GRAMMAR[f]))))[:, None]) for f in factors], base)

    @staticmethod
    def of(combos: list[dict[str, int]], base: dict[str, int] | None = None) -> "Space":
        names = sorted({f for c in combos for f in c})
        return Space([(names, np.array([[c[f] for f in names] for c in combos]))], base)

    def times(self, other: "Space") -> "Space":
        return Space(self.axes + other.axes, self.base)

    def decode(self, ids, full: bool = True) -> dict[str, np.ndarray]:
        """Option index per factor for each id (numpy, on the host); without `full`, only the factors the axes vary."""
        ids = ids.cpu().numpy() if torch.is_tensor(ids) else np.asarray(ids, dtype=np.int64)
        out, stride = {}, 1
        for names, table in reversed(self.axes):
            k = (ids // stride) % len(table)
            for j, f in enumerate(names):
                out[f] = table[k, j]
            stride *= len(table)
        if full:
            for f in GRAMMAR:
                if f not in out:
                    out[f] = np.full(ids.shape, self.base[f], dtype=np.int64)
        return out

    def decode_device(self, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """The varied factors' option indices, computed where the ids live (the bulk pass's GPU path)."""
        out, stride = {}, 1
        for names, table in reversed(self.axes):
            k = (ids // stride) % len(table)
            t = torch.as_tensor(table, device=ids.device)
            for j, f in enumerate(names):
                out[f] = t[k, j]
            stride *= len(table)
        return out

    def present_in_range(self, a: int, b: int) -> dict[str, list[int]]:
        """Options present among ids a..b (inclusive), from the mixed-radix structure alone."""
        out, stride = {}, 1
        for names, table in reversed(self.axes):
            n = len(table)
            q0, q1 = a // stride, b // stride
            ks = np.arange(n) if q1 - q0 + 1 >= n else np.arange(q0, q1 + 1) % n
            for j, f in enumerate(names):
                out[f] = np.unique(table[ks, j]).tolist()
            stride *= n
        return out

    def combo(self, i: int) -> dict[str, int]:
        return {f: int(v[0]) for f, v in self.decode([i]).items()}

    def combos(self, ids) -> list[dict[str, int]]:
        if len(ids) == 0:
            return []
        cols = self.decode(ids)
        return [dict(zip(cols, map(int, vals))) for vals in zip(*cols.values())]


def describe(c: dict[str, int], keys=None) -> dict:
    keys = keys or [f for f in GRAMMAR if c[f] != REF_IDX[f]]
    out = {f: GRAMMAR[f][c[f]] for f in keys}
    if "gate" in out:
        out["gate"] = hex(out["gate"])
    return out


# ---------------------------------------------------------------- cells (the one shared 3->2 gate)
class GenericCells:
    """Per-lane truth tables: lane j of word w evaluates table ids[64 w + j] (sum8 | carry8 << 8, index a+2b+4c)."""

    def __init__(self, ids: torch.Tensor) -> None:
        lanes = ids.view(-1, 64)
        shifts = torch.arange(64, device=ids.device)
        m = torch.arange(16, device=ids.device)
        words = (((lanes[None] >> m[:, None, None]) & 1) << shifts).sum(-1)  # [16, words]: one buffer, 16 views
        self.t = [[words[8 * o + j] for j in range(8)] for o in (0, 1)]

    @staticmethod
    def _eval(t, ins):
        for i, x in enumerate(ins):
            if isinstance(x, int):
                return GenericCells._eval([t[m] for m in range(len(t)) if (m >> i) & 1 == x], ins[:i] + ins[i + 1:])
        if not ins:
            return t[0]
        x = ins[0]
        return GenericCells._eval([t[2 * j] ^ (x & (t[2 * j] ^ t[2 * j + 1])) for j in range(len(t) // 2)], ins[1:])

    def cell(self, o, ins):
        rank = max(x.dim() for x in ins if torch.is_tensor(x))
        return self._eval([w.view(-1, *([1] * (rank - 1))) for w in self.t[o]], ins)

    def fa(self, a, b, c): return self.cell(0, [a, b, c]), self.cell(1, [a, b, c])
    def AND(self, a, b): return self.cell(1, [a, b, 0])
    def OR(self, a, b): return self.cell(1, [a, b, 1])
    def XOR(self, a, b): return self.cell(0, [a, b, 0])
    def NOT(self, a): return self.cell(0, [a, 1, 0])
    def MUX(self, c, a, b): return self.OR(self.AND(c, a), self.AND(self.NOT(c), b))


class FACells:
    """The full adder as two-level logic (what a learned table 0xE896 compiles to: LOP3 0x96 and 0xE8)."""

    def fa(self, a, b, c):
        ts = [x for x in (a, b, c) if torch.is_tensor(x)]
        ones = sum(x for x in (a, b, c) if not torch.is_tensor(x))
        if len(ts) == 3:
            x = a ^ b
            return x ^ c, (a & b) | (c & x)
        if len(ts) == 2:
            p, q = ts
            return (p ^ q, p & q) if ones == 0 else (~(p ^ q), p | q)
        p = ts[0]
        return [(p, torch.zeros_like(p)), (~p, p), (p, torch.full_like(p, -1))][ones]

    def AND(self, a, b): return a & b
    def OR(self, a, b): return a | b
    def XOR(self, a, b): return a ^ b
    def NOT(self, a): return ~a
    def MUX(self, c, a, b): return (c & a) | (~c & b)


# ---------------------------------------------------------------- lanes
def lane_view(w: torch.Tensor, rank: int) -> torch.Tensor:
    return w.view(-1, *([1] * (rank - 1)))


class Lanes:
    """A set of candidate machines (lane i = space combo ids[i]) with per-lane select masks and cells."""

    def __init__(self, space: Space, ids: torch.Tensor, pad_to: int = 0, cache: bool = True,
                 id_range: tuple[int, int] | None = None) -> None:
        """`id_range` = (a, b) when ids are exactly a..b: options present then follow from the space's structure and
        the decode stays on the device; otherwise (survivor lists) both happen on the host."""
        self.n = ids.numel()
        self._cache = {} if cache else None  # eager runs memoize group masks; traced (compiled) runs must not mutate
        width = max(pad_to, self.n + (-self.n) % 64)
        self.dev = ids.device
        ids = torch.cat([ids, ids[:1].expand(width - self.n)])  # padding lanes repeat lane 0 and are ignored
        self.NW = ids.numel() // 64
        if id_range is not None:
            self.present = space.present_in_range(*id_range)
            opt = {f: v for f, v in space.decode_device(ids).items() if len(self.present[f]) > 1}
        else:
            host = space.decode(ids, full=False)
            self.present = {f: np.nonzero(np.bincount(v[: self.n], minlength=len(GRAMMAR[f])))[0].tolist()
                            for f, v in host.items()}
            opt = {f: torch.as_tensor(v, device=self.dev) for f, v in host.items() if len(self.present[f]) > 1}
        for f in GRAMMAR:
            self.present.setdefault(f, [space.base[f]])
        self.opt = opt
        self._shifts = torch.arange(64, device=self.dev)
        # one select word per (factor, present option); group masks are ORs of these (no state mutated later)
        self.sel = {}
        for f in GRAMMAR:
            if f != "gate" and len(self.present[f]) > 1:
                opts = torch.tensor(self.present[f], device=self.dev)
                hits = (self.opt[f].view(1, -1, 64) == opts[:, None, None]).long()
                words = (hits << self._shifts).sum(-1)  # [options, words]: one buffer per factor
                self.sel[f] = {o: words[i] for i, o in enumerate(self.present[f])}
        gates = self.present["gate"]
        if gates == [FA_ID]:
            self.cells = FACells()
        elif len(gates) == 1:
            self.cells = GenericCells(torch.full((64,), gates[0], dtype=torch.int64, device=self.dev))
        else:
            self.cells = GenericCells(self.opt["gate"])
        self.wmask = self._widths("W", WMAX)
        self.cmask = self._widths("cw", TWMAX)

    def value(self, f: str, o: int):
        return GRAMMAR[f][o]

    def mask(self, f: str, opts) -> torch.Tensor | None | int:
        """Word mask of lanes whose option for f is in opts: None = every lane, 0 = no lane."""
        opts = sorted(o for o in set(opts) if o in self.present[f])
        if not opts:
            return 0
        if len(opts) == len(self.present[f]):
            return None
        key = (f, tuple(opts))
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        m = self.sel[f][opts[0]]
        for o in opts[1:]:
            m = m | self.sel[f][o]
        if self._cache is not None:
            self._cache[key] = m
        return m

    def choose(self, f: str, key, thunk):
        """Per-lane multiplexer: group the present options of f by key(value), compute thunk(key) once per group."""
        groups: dict = {}
        for o in self.present[f]:
            groups.setdefault(key(self.value(f, o)), []).append(o)
        if len(groups) == 1:
            return thunk(next(iter(groups)))
        out, is_tuple = None, False
        for k, opts in groups.items():
            v = thunk(k)
            is_tuple = isinstance(v, tuple)
            m = self.mask(f, opts)
            part = tuple(x & lane_view(m, x.dim()) for x in (v if is_tuple else (v,)))
            out = part if out is None else tuple(a | b for a, b in zip(out, part))
        return out if is_tuple else out[0]

    def pred(self, f: str, cond, new, old):
        """Lanes whose option value satisfies cond take `new`, the rest keep `old` (designed predication)."""
        m = self.mask(f, [o for o in self.present[f] if cond(self.value(f, o))])
        if m is None:
            return new
        if isinstance(m, int):
            return old
        m = lane_view(m, max(new.dim(), old.dim()))
        return old ^ ((old ^ new) & m)

    def _widths(self, f: str, w: int):
        """Per-lane width as a bit mask [lanes or 1, 1, w]; None when every lane uses the full width."""
        widths = sorted({self.value(f, o) for o in self.present[f]})
        if widths == [w]:
            return None
        if len(widths) == 1:
            return torch.where(torch.arange(w, device=self.dev) < widths[0], -1, 0).view(1, 1, w)
        bits = []
        for b in range(w):
            m = self.mask(f, [o for o in self.present[f] if self.value(f, o) > b])
            bits.append(m if torch.is_tensor(m) else torch.full((self.NW,), -1 if m is None else 0, device=self.dev))
        return torch.stack(bits, -1).view(self.NW, 1, w)


# ---------------------------------------------------------------- wiring helpers
def cat(*vs):
    lead = torch.broadcast_shapes(*[v.shape[:-1] for v in vs])
    return torch.cat([v.expand(*lead, v.shape[-1]) for v in vs], -1)


def shl(v, k):
    if k < 0:
        return shr(v, -k)
    return F.pad(v[..., : v.shape[-1] - k], (k, 0)) if k else v


def shr(v, k):
    if k < 0:
        return shl(v, -k)
    return F.pad(v[..., k:], (0, k)) if k else v


def zeros(dev, w):
    return torch.zeros(1, 1, w, dtype=torch.int64, device=dev)


def word_bits(bits):
    return torch.where(bits.bool(), -1, 0).to(torch.int64)[None]


def masked(v, m):
    return v if m is None else v & m.view(m.shape[0], *([1] * (v.dim() - 2)), m.shape[-1])


def digits_of(vals: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(vals), n), dtype=np.int64)
    v = np.array(vals, dtype=np.int64).copy()
    for i in range(n):
        out[:, i] = v % 10
        v //= 10
    return out


# ---------------------------------------------------------------- blocks
def ks_add(g, a, b, cin):
    gen, prop = g.AND(a, b), g.XOR(a, b)
    G = cat(cin, gen) if torch.is_tensor(cin) else cat(torch.full_like(gen[..., :1], -cin), gen)
    P = cat(torch.zeros_like(prop[..., :1]), prop)
    n, dist = G.shape[-1], 1
    while dist < n:
        Gn, Pn = g.OR(G, g.AND(P, shl(G, dist))), g.AND(P, shl(P, dist))
        G, P = cat(G[..., :dist], Gn[..., dist:]), cat(P[..., :dist], Pn[..., dist:])
        dist *= 2
    return g.XOR(prop, G[..., :-1]), G[..., n - 1:]


def ripple_add(g, a, b):
    outs, c = [], 0
    for j in range(a.shape[-1]):
        s, c = g.fa(a[..., j:j + 1], b[..., j:j + 1], c)
        outs.append(s)
    return cat(*outs)


def d2b(L: Lanes, digs: torch.Tensor, w: int, wm) -> torch.Tensor:
    """LSB-first decimal digits [rows, ND] (or a machine's own BCD output words [lanes, rows, 4 ND]) -> w-bit words:
    carry-save Horner v <- v<<a + v<<b + d, then a CPA."""
    g = L.cells
    bcd = digs.dim() == 3

    def horner(order):
        vs = vc = zeros(L.dev, w)
        nd = digs.shape[-1] // 4 if bcd else digs.shape[1]
        for i in (range(nd - 1, -1, -1) if order == "msb_first" else range(nd)):
            d = F.pad(digs[..., 4 * i:4 * i + 4] if bcd else word_bits((digs[:, i:i + 1] >> torch.arange(4, device=L.dev)) & 1),
                      (0, w - 4))
            pa, pac = L.choose("d2b_mul", lambda m: m[0], lambda a: (shl(vs, a), shl(vc, a)))
            pb, pbc = L.choose("d2b_mul", lambda m: m[1], lambda b: (shl(vs, b), shl(vc, b)))
            s1, c1 = g.fa(pa, pac, pb)
            s2, c2 = g.fa(s1, shl(c1, 1), pbc)
            s3, c3 = g.fa(s2, shl(c2, 1), d)
            vs, vc = masked(s3, wm), masked(shl(c3, 1), wm)
        return masked(ks_add(g, vs, vc, 0)[0], wm)

    return L.choose("d2b_order", lambda o: o, horner)


def b2d(L: Lanes, y: torch.Tensor) -> torch.Tensor:
    """Double dabble with a per-lane rule: before each shift add c to every nibble >= t."""
    g = L.cells

    def run(order):
        reg = zeros(L.dev, 4 * NB)
        for bit in (range(23, -1, -1) if order == "msb_first" else range(24)):
            nib = reg.reshape(*reg.shape[:-1], NB, 4)
            n0, n1, n2, n3 = (nib[..., j:j + 1] for j in range(4))
            ge = {3: lambda: g.OR(n3, g.OR(n2, g.AND(n1, n0))), 4: lambda: g.OR(n3, n2),
                  5: lambda: g.OR(n3, g.AND(n2, g.OR(n1, n0))), 6: lambda: g.OR(n3, g.AND(n2, n1)),
                  7: lambda: g.OR(n3, g.AND(n2, g.AND(n1, n0))), 8: lambda: n3}
            cond = L.choose("b2d_thr", lambda t: t, lambda t: ge[t]())
            add = [L.choose("b2d_add", lambda c, j=j: (c >> j) & 1,
                            lambda bit_on: cond if bit_on else torch.zeros_like(cond)) for j in range(3)]
            a0, c = g.fa(n0, add[0], 0)
            a1, c = g.fa(n1, add[1], c)
            a2, c = g.fa(n2, add[2], c)
            a3, _ = g.fa(n3, 0, c)
            reg = cat(y[..., bit:bit + 1], cat(a0, a1, a2, a3).flatten(-2)[..., :-1])
        return reg

    return L.choose("b2d_order", lambda o: o, run)


def entry(L: Lanes, X, N, wm):
    """x * 2^Ke mod N."""
    g = L.cells
    kmax = max(L.value("Ke", o) for o in L.present["Ke"])

    def radix(r):
        s = X
        if r == 2:
            negN = g.NOT(N)
            for i in range(kmax):
                t = masked(shl(s, 1), wm)
                d, co = ks_add(g, t, negN, 1)
                s = L.pred("Ke", lambda K, i=i: K > i, masked(g.MUX(co, d, t), wm), s)
            return s
        N2 = masked(shl(N, 1), wm)
        N3 = masked(ks_add(g, N, N2, 0)[0], wm)
        negs = [g.NOT(N), g.NOT(N2), g.NOT(N3)]
        for i in range(kmax // 2):
            t = masked(shl(s, 2), wm)
            diffs = [ks_add(g, t, negs[k], 1) for k in range(3)]
            snew = g.MUX(diffs[0][1], diffs[0][0], t)
            snew = g.MUX(diffs[1][1], diffs[1][0], snew)
            snew = g.MUX(diffs[2][1], diffs[2][0], snew)
            s = L.pred("Ke", lambda K, i=i: K // 2 > i, masked(snew, wm), s)
        return s

    return L.choose("entry_radix", lambda r: r, radix)


def operands(L: Lanes, a, b):
    """(multiplier, multiplicand) of MonPro(a, b): whose bits are scanned and whose multiples are added (ppop)."""
    if a is b:
        return a, a
    ops = {"a.b": (a, b), "b.a": (b, a), "a.a": (a, a), "b.b": (b, b)}
    return L.choose("ppop", lambda o: o, lambda o: ops[o])


def perm_mux(L: Lanes, f: str, s0, s1, xs: dict):
    """xs[p[s0 + 2 s1]] per bit, p = the lane's option of f (an ordering of xs's names; a None word is zero).
    The circuit is a one-hot decode of (s0, s1) and an AND-OR plane; f only routes decoded lines to sources, so the
    cost does not grow with the number of orderings present."""
    g = L.cells
    ns0, ns1 = g.NOT(s0), g.NOT(s1)
    lines = [g.AND(ns0, ns1), g.AND(s0, ns1), g.AND(ns0, s1), g.AND(s0, s1)]
    out = None
    for name, x in xs.items():
        if x is None:
            continue
        sel = None
        for k, line in enumerate(lines):
            m = L.mask(f, [o for o in L.present[f] if L.value(f, o)[k] == name])
            if isinstance(m, int):
                continue
            t = line if m is None else line & lane_view(m, line.dim())
            sel = t if sel is None else sel | t
        if sel is not None:
            term = g.AND(sel, x)
            out = term if out is None else g.OR(out, term)
    return out


def shl_in(v, bit):
    """v << 1 with `bit` in the freed bit 0 (the carry a shift by 4 leaves behind)."""
    return cat(bit, v[..., :-1])


def kmax_of(L: Lanes) -> int:
    return max(L.value("Km", o) for o in L.present["Km"])


# Each family's iteration is built from the cell functions below, which the MonPro loops and the per-cell laws
# (cell_law_bad) share: a sub-block law screens exactly the wiring the machine runs.
def wired_cell(L: Lanes, src_f: str, sh_f: str, src: tuple):
    """A carry-save cell whose port j reads src[t[j]] (t = the lane's option of src_f), shifted by the lane's sh_f."""
    ports = [L.choose(src_f, lambda t, j=j: t[j], lambda k: src[k]) for j in range(3)]
    return L.cells.fa(*(L.choose(sh_f, lambda t, j=j: t[j], lambda k, j=j: shl(ports[j], k)) for j in range(3)))


def r2_quotient(L: Lanes, Ss, Sc, u, v):
    g = L.cells
    u0, v0 = u[..., :1], v[..., :1]
    qsrc = {"u": lambda: u0, "v": lambda: v0, "u^v": lambda: g.XOR(u0, v0), "Ss": lambda: Ss[..., :1],
            "Sc": lambda: Sc[..., :1], "Ss^Sc": lambda: g.XOR(Ss[..., :1], Sc[..., :1]), "~u": lambda: g.NOT(u0)}
    return L.choose("qsrc", lambda x: x, lambda x: qsrc[x]())


def handoff2(L: Lanes, s, c, wm):
    """Radix-2 hand-off: the sum row shifted right by osh, the carry row left by csh (-1 = right), both W bits."""
    return (masked(L.choose("osh", lambda k: k, lambda k: shr(s, k)), wm),
            masked(L.choose("csh", lambda k: k, lambda k: shl(c, k)), wm))


def monpro_r2(L: Lanes, mpl, mcd, N, wm):
    """Radix 2, two carry-save cells per multiplier bit (the reference core, with its port-wiring factors)."""
    g = L.cells
    kmax = kmax_of(L)
    pp_all = g.AND(mpl[..., :KMAX, None], mcd[..., None, :])
    Ss = Sc = zeros(L.dev, WMAX)
    for i in range(kmax):
        pp = L.choose("ppdir", lambda d: d, lambda d: pp_all[..., i if d == "lsb_first" else KMAX - 1 - i, :])
        u, v = wired_cell(L, "l1src", "l1sh", (Ss, Sc, pp))
        qN = g.AND(r2_quotient(L, Ss, Sc, u, v), N)
        nSs, nSc = handoff2(L, *wired_cell(L, "l2src", "l2sh", (u, v, qN)), wm)
        Ss = L.pred("Km", lambda K, i=i: K > i, nSs, Ss)
        Sc = L.pred("Km", lambda K, i=i: K > i, nSc, Sc)
    return masked(ks_add(g, Ss, Sc, 0)[0], wm)


def r2m_quotient(L: Lanes, Ss, Sc, d, b0):
    g = L.cells
    s0, c0, db0 = Ss[..., :1], Sc[..., :1], g.AND(d, b0)
    qf = {"s^c^db0": lambda: g.XOR(g.XOR(s0, c0), db0), "s^c": lambda: g.XOR(s0, c0), "s^db0": lambda: g.XOR(s0, db0),
          "c^db0": lambda: g.XOR(c0, db0), "db0": lambda: db0, "~(s^c^db0)": lambda: g.NOT(g.XOR(g.XOR(s0, c0), db0))}
    return L.choose("r2mq", lambda x: x, lambda x: qf[x]())


def r2m_cell(L: Lanes, Ss, Sc, d, q, rows):
    return L.cells.fa(*port_shifts(L, "r2msh", [Ss, Sc, perm_mux(L, "r2mrow", d, q, rows)]))


def monpro_r2m(L: Lanes, mpl, mcd, N, wm):
    """Radix 2, one carry-save cell per bit: the quotient bit first (LSB of Ss + Sc + a_i b), then one row from the
    precomputed {0, b, N, b+N} selected by (a_i, q)."""
    g = L.cells
    kmax = kmax_of(L)
    rows = {"0": None, "b": mcd, "N": N, "b+N": masked(ks_add(g, mcd, N, 0)[0], wm)}
    b0 = mcd[..., :1]
    Ss = Sc = zeros(L.dev, WMAX)
    for i in range(kmax):
        d = L.choose("ppdir", lambda o: o, lambda o: mpl[..., i:i + 1] if o == "lsb_first" else mpl[..., KMAX - 1 - i:KMAX - i])
        u, v = r2m_cell(L, Ss, Sc, d, r2m_quotient(L, Ss, Sc, d, b0), rows)
        nSs, nSc = handoff2(L, u, v, wm)
        Ss = L.pred("Km", lambda K, i=i: K > i, nSs, Ss)
        Sc = L.pred("Km", lambda K, i=i: K > i, nSc, Sc)
    return masked(ks_add(g, Ss, Sc, 0)[0], wm)


Q4_BITS = {  # (q0, q1) from the low bits S0, S1 of the carry-save sum and bit 1 of N; q = -S/N mod 4 makes S + qN = 0 mod 4
    "gen": lambda g, s0, s1, n1: (s0, g.XOR(s1, g.AND(s0, g.NOT(n1)))),  # any odd N
    "n1": lambda g, s0, s1, n1: (s0, g.XOR(s1, s0)),  # N = 1 mod 4 only (every Blum modulus N = pq, p = q = 3 mod 4)
    "n3": lambda g, s0, s1, n1: (s0, s1),  # N = 3 mod 4 only
    "x1": lambda g, s0, s1, n1: (s0, g.XOR(s1, g.AND(s0, n1))),  # the N mod 4 correction with the wrong polarity
    "lo": lambda g, s0, s1, n1: (s0, torch.zeros_like(s0)),  # a radix-2 quotient in a radix-4 loop
    "~S0": lambda g, s0, s1, n1: (g.NOT(s0), s1),
    "swap": lambda g, s0, s1, n1: (s1, s0),
    "S0S0": lambda g, s0, s1, n1: (s0, s0),
}
R4CIN = {"u1": lambda g, u, v: u[..., 1:2], "v0": lambda g, u, v: v[..., :1],
         "u1&v0": lambda g, u, v: g.AND(u[..., 1:2], v[..., :1]), "u1|v0": lambda g, u, v: g.OR(u[..., 1:2], v[..., :1]),
         "0": lambda g, u, v: torch.zeros_like(v[..., :1])}


def handoff4(L: Lanes, u, v, wm):
    """(u + 2v) / 4 = u>>2 + v>>1 + u_1 when u + 2v = 0 mod 4 (then u_0 = 0 and u_1 = v_0): the carry goes to bit 0
    of the next iteration's first carry row."""
    return (masked(L.choose("r4osh", lambda k: k, lambda k: shr(u, k)), wm),
            masked(L.choose("r4csh", lambda k: k, lambda k: shr(v, k)), wm),
            L.choose("r4cin", lambda c: c, lambda c: R4CIN[c](L.cells, u, v)))


def r4_handoff(L: Lanes, u, v, wm, i, Ss, Sc, cin):
    nSs, nSc, ncin = handoff4(L, u, v, wm)
    on = lambda K, i=i: K // 2 > i
    return L.pred("Km", on, nSs, Ss), L.pred("Km", on, nSc, Sc), L.pred("Km", on, ncin, cin)


def digit_bits(L: Lanes, mpl, i):
    """The two multiplier bits of radix-4 digit i, LSB or MSB first."""
    return L.choose("ppdir", lambda d: d, lambda d: (mpl[..., 2 * i:2 * i + 1], mpl[..., 2 * i + 1:2 * i + 2])
                    if d == "lsb_first" else (mpl[..., KMAX - 1 - 2 * i:KMAX - 2 * i], mpl[..., KMAX - 2 - 2 * i:KMAX - 1 - 2 * i]))


def r4_rows(L: Lanes, Ss, Sc, P0, P1):
    return {"Ss": Ss, "Sc": Sc, "P0": P0, "P1": L.choose("r4p1sh", lambda k: k, lambda k: shl(P1, k))}


def r4_cell1(L: Lanes, rows):
    """Tree shape as port routing: the first cell takes the three rows r4hold does not hold back."""
    first = [L.choose("r4hold", lambda h, j=j: [k for k in rows if k != h][j], lambda k: rows[k]) for j in range(3)]
    return L.cells.fa(*port_shifts(L, "r4c1sh", first))


def r4_cell2(L: Lanes, rows, u1, v1, cin):
    return L.cells.fa(*port_shifts(L, "r4c2sh", [u1, shl_in(v1, cin), L.choose("r4hold", lambda h: h, lambda k: rows[k])]))


def r4_quotient(L: Lanes, u2, v2, n1):
    g = L.cells
    s0, s1 = u2[..., :1], g.XOR(u2[..., 1:2], v2[..., :1])
    return L.choose("q4", lambda f: f, lambda f: Q4_BITS[f](g, s0, s1, n1))


def qtree_order(t: str) -> list[str]:
    """"(u,v,Q0)(Q1)": cell 3 = (u, v, Q0), cell 4 adds Q1; the other two shapes as named."""
    return t.replace("(", " ").replace(")", " ").replace(",", " ").split()


def r4_cells34(L: Lanes, u2, v2, q0, q1, N):
    """Cells 3 and 4 add the quotient rows q0 N and q1 N << 1 where r4qtree routes them, each port shifted by the
    lane's r4c3sh / r4c4sh."""
    g = L.cells
    src = {"u": u2, "v": shl(v2, 1), "Q0": g.AND(q0, N),
           "Q1": L.choose("r4q1sh", lambda k: k, lambda k: shl(g.AND(q1, N), k))}
    third = [L.choose("r4qtree", lambda t, j=j: qtree_order(t)[j], lambda k: src[k]) for j in range(4)]
    u3, v3 = g.fa(*port_shifts(L, "r4c3sh", third[:3]))
    return g.fa(*port_shifts(L, "r4c4sh", [u3, shl(v3, 1), third[3]]))


def monpro_r4(L: Lanes, mpl, mcd, N, wm):
    """Radix 4: Km/2 iterations. Per 2-bit digit: rows Ss, Sc, P0 = a_2i b, P1 = a_2i+1 b << 1 meet in two carry-save
    cells; the 2-bit quotient comes from the low two bits of that sum and N mod 4; rows q0 N and q1 N << 1 meet in two
    more cells; then the sum is shifted by 2 bits."""
    g = L.cells
    kmax = kmax_of(L)
    pp_all = g.AND(mpl[..., :KMAX, None], mcd[..., None, :])
    n1 = N[..., 1:2]
    Ss = Sc = zeros(L.dev, WMAX)
    cin = torch.zeros(1, 1, 1, dtype=torch.int64, device=L.dev)
    for i in range(kmax // 2):
        P0, P1 = L.choose("ppdir", lambda d: d, lambda d: (pp_all[..., 2 * i, :], pp_all[..., 2 * i + 1, :])
                          if d == "lsb_first" else (pp_all[..., KMAX - 1 - 2 * i, :], pp_all[..., KMAX - 2 - 2 * i, :]))
        rows = r4_rows(L, Ss, Sc, P0, P1)
        u1, v1 = r4_cell1(L, rows)
        u2, v2 = r4_cell2(L, rows, u1, v1, cin)
        u4, v4 = r4_cells34(L, u2, v2, *r4_quotient(L, u2, v2, n1), N)
        Ss, Sc, cin = r4_handoff(L, u4, v4, wm, i, Ss, Sc, cin)
    return masked(ks_add(g, Ss, Sc, cin)[0], wm)


def r4m_tables(L: Lanes, mcd, N, wm):
    """The precomputed multiples {0, b, 2b, 3b} and {0, N, 2N, 3N}, W bits each."""
    g = L.cells
    b2, N2 = masked(shl(mcd, 1), wm), masked(shl(N, 1), wm)
    return ({"0": None, "b": mcd, "2b": b2, "3b": masked(ks_add(g, mcd, b2, 0)[0], wm)},
            {"0": None, "N": N, "2N": N2, "3N": masked(ks_add(g, N, N2, 0)[0], wm)})


def r4m_cell1(L: Lanes, Ss, Sc, d0, d1, mb):
    return L.cells.fa(*port_shifts(L, "r4m1sh", [Ss, Sc, perm_mux(L, "r4mpp", d0, d1, mb)]))


def r4m_quotient(L: Lanes, u1, v1s, n1):
    g = L.cells
    a0, b0 = u1[..., :1], v1s[..., :1]
    s0, s1 = g.XOR(a0, b0), g.XOR(g.XOR(u1[..., 1:2], v1s[..., 1:2]), g.AND(a0, b0))
    return L.choose("q4", lambda f: f, lambda f: Q4_BITS[f](g, s0, s1, n1))


def r4m_cell2(L: Lanes, u1, v1s, q0, q1, mn):
    return L.cells.fa(*port_shifts(L, "r4m2sh", [u1, v1s, perm_mux(L, "r4mqn", q0, q1, mn)]))


def monpro_r4m(L: Lanes, mpl, mcd, N, wm):
    """Radix 4 on precomputed multiples: per 2-bit digit one row from {0, b, 2b, 3b}, the quotient from the low two
    bits of the carry-save sum and N mod 4, one row from {0, N, 2N, 3N}; two carry-save cells."""
    g = L.cells
    kmax = kmax_of(L)
    mb, mn = r4m_tables(L, mcd, N, wm)
    n1 = N[..., 1:2]
    Ss = Sc = zeros(L.dev, WMAX)
    cin = torch.zeros(1, 1, 1, dtype=torch.int64, device=L.dev)
    for i in range(kmax // 2):
        d0, d1 = digit_bits(L, mpl, i)
        u1, v1 = r4m_cell1(L, Ss, Sc, d0, d1, mb)
        v1s = shl_in(v1, cin)
        u2, v2 = r4m_cell2(L, u1, v1s, *r4m_quotient(L, u1, v1s, n1), mn)
        Ss, Sc, cin = r4_handoff(L, u2, v2, wm, i, Ss, Sc, cin)
    return masked(ks_add(g, Ss, Sc, cin)[0], wm)


MONPRO = {"r2": monpro_r2, "r2m": monpro_r2m, "r4": monpro_r4, "r4m": monpro_r4m}


def monpro(L: Lanes, a, b, N, wm):
    """MonPro(a, b) = (a b + m N) / 2^Km with m = -a b / N mod 2^Km, through the lane's multiplier family. Every correct
    family gives the same integer (m is unique), below 2N for a, b < 2N when 2^Km > 4N."""
    mpl, mcd = operands(L, a, b)
    return L.choose("mult", lambda m: m, lambda m: MONPRO[m](L, mpl, mcd, N, wm))


def mont_square(L: Lanes, s, N, wm):
    return monpro(L, s, s, N, wm)


def exit_block(L: Lanes, s, N, wm):
    """MonPro(s, 1): Kx halvings mod N."""
    g = L.cells
    kmax = max(L.value("Kx", o) for o in L.present["Kx"])
    Ss, Sc = s, zeros(L.dev, WMAX)
    for i in range(kmax):
        a0, b0 = Ss[..., :1], Sc[..., :1]
        rules = {"s^c": lambda: g.XOR(a0, b0), "s": lambda: a0, "c": lambda: b0, "s&c": lambda: g.AND(a0, b0)}
        q = L.choose("exq", lambda x: x, lambda x: rules[x]())
        u, v = g.fa(Ss, Sc, g.AND(q, N))
        Ss = L.pred("Kx", lambda K, i=i: K > i, masked(shr(u, 1), wm), Ss)
        Sc = L.pred("Kx", lambda K, i=i: K > i, masked(v, wm), Sc)
    return masked(ks_add(g, Ss, Sc, 0)[0], wm)


def machine(L: Lanes, Nd, Xd, Td, max_steps: int):
    """The whole candidate machine on prompts (digits LSB first) -> BCD words [lanes, rows, 32]."""
    g = L.cells
    wm = L.wmask
    N, X = d2b(L, Nd, WMAX, wm), d2b(L, Xd, WMAX, wm)
    cnt = d2b(L, Td, TWMAX, L.cmask)
    s = entry(L, X, N, wm)
    for _ in range(max_steps):
        active = g.OR(g.OR(g.OR(cnt[..., 0:1], cnt[..., 1:2]), g.OR(cnt[..., 2:3], cnt[..., 3:4])),
                      g.OR(g.OR(cnt[..., 4:5], cnt[..., 5:6]), cnt[..., 6:7]))
        if not bool(active.any()):
            break
        s = g.MUX(active, mont_square(L, s, N, wm), s)
        cnt = masked(ripple_add(g, cnt, active.expand(*active.shape[:-1], TWMAX)), L.cmask)
    return b2d(L, exit_block(L, s, N, wm))


def semigroup_sides(L: Lanes, r, t1: int, t2: int):
    """The whole machine against the semigroup law of repeated squaring, y(x, t1 + t2) = y(y(x, t1), t2), on the
    prompts' own (N, x) and synthesized T (no label): the counter must count to t1 + t2."""
    T = lambda t: torch.as_tensor(digits_of(np.full(r.n, t), 2), device=L.dev)
    steps = (1 << TWMAX) - 1
    return (machine(L, r.Nd, r.Xd, T(t1 + t2), steps),
            machine(L, r.Nd, machine(L, r.Nd, r.Xd, T(t1), steps), T(t2), steps))


def decimal_sides(L: Lanes, r):
    """The answer's declared format, no label: every output nibble of y(x, 1) on the prompts is a decimal digit (an io
    block that computes in base 12 or 16 prints digits above 9)."""
    reg = machine(L, r.Nd, r.Xd, r.Td, 1)
    nib = reg.view(*reg.shape[:-1], NB, 4)
    bad = nib[..., 3] & (nib[..., 2] | nib[..., 1])  # nibble >= 10
    return bad, torch.zeros_like(bad)


def roundtrip_io(L: Lanes, vd):
    return b2d(L, d2b(L, vd, WMAX, L.wmask))


def roundtrip_entry_exit(L: Lanes, Nd, Xd):
    wm = L.wmask
    N, X = d2b(L, Nd, WMAX, wm), d2b(L, Xd, WMAX, wm)
    return b2d(L, exit_block(L, entry(L, X, N, wm), N, wm))


LAWS = ["id", "comm", "comm_sq", "assoc", "mirror", "mirror2", "unit"]


def law_sides(L: Lanes, r, law: str):
    """Both sides of a label-free MonPro law on (N, x) words; the lane survives iff they agree bit for bit."""
    wm = L.wmask
    N, X = masked(r.Nb, wm), masked(r.Xb, wm)
    if law == "unit":  # squaring only, like the machine: 1^2 = 1 in Montgomery form; pins Km = 2 Ke - Kx
        one = entry(L, masked(r.One, wm), N, wm)
        return exit_block(L, monpro(L, one, one, N, wm), N, wm), masked(r.One, wm)
    a = entry(L, X, N, wm)
    if law in ("id", "comm"):
        one = entry(L, masked(r.One, wm), N, wm)
        if law == "id":
            return exit_block(L, monpro(L, a, one, N, wm), N, wm), X
        return monpro(L, a, one, N, wm), monpro(L, one, a, N, wm)
    sq = monpro(L, a, a, N, wm)
    if law == "comm_sq":  # blind to projections f(a, b) = a: there a^2 = f(a, a) = a, so both sides agree
        return monpro(L, a, sq, N, wm), monpro(L, sq, a, N, wm)
    if law == "assoc":
        return (exit_block(L, monpro(L, sq, sq, N, wm), N, wm),
                exit_block(L, monpro(L, a, monpro(L, a, sq, N, wm), N, wm), N, wm))
    if law in ("mirror", "mirror2"):
        a2 = entry(L, masked(r.Mb, wm), N, wm)
        sq2 = monpro(L, a2, a2, N, wm)
        if law == "mirror2":  # x^4 = (N - x)^4: the second squaring reads an unreduced (< 2N) operand
            sq, sq2 = monpro(L, sq, sq, N, wm), monpro(L, sq2, sq2, N, wm)
        return exit_block(L, sq, N, wm), exit_block(L, sq2, N, wm)
    raise ValueError(law)


# ---------------------------------------------------------------- per-cell (sub-block) laws
# Value laws: every carry-save cell conserves value with one scale per row class. A cell's u + 2v = 2^j (its state and
# quotient rows + 2^e x its digit rows), each row at its natural weight: the state rows are Ss, Sc, the carry into bit 0
# and the running sums of earlier cells; the digit rows are the multiples of the multiplicand b; the quotient rows the
# multiples of N. j is per cell; the digit scale e is one per iteration, shared by its cells (DIGIT_SCALES). e = 2 is
# the x4-scaled datapath (Montgomery with R = 2^(Km - 2)); e = 1 gives R = 2^(Km - 1), which the MonPro laws reject. The
# quotient rows have no scale of their own: a quotient row at another scale than the state rows cannot zero their low
# bits. Per iteration: the quotient makes the iteration's value 0 mod 2^r, and the hand-off divides it by 2^r exactly.
# Each law reads one cell (or one iteration's quotient or hand-off) on synthesized words, so a family's wiring factors
# are screened sub-block by sub-block instead of jointly.
CELL_LAWS = {  # per family: sub-block law -> (the factors it screens, the laws whose survivors it extends)
    "r2": {"r2.cell1": (["l1src", "l1sh"], []), "r2.quot": (["qsrc"], ["r2.cell1"]),
           "r2.cell2": (["l2src", "l2sh"], []), "r2.handoff": (["osh", "csh"], ["r2.cell2"])},
    "r2m": {"r2m.cell": (["r2msh", "r2mrow", "r2mq"], []), "r2m.handoff": (["osh", "csh"], ["r2m.cell"])},
    "r4": {"r4.cell1": (["r4hold", "r4p1sh", "r4c1sh"], []), "r4.cell2": (["r4hold", "r4p1sh", "r4c2sh"], []),
           "r4.quot": (["q4"], ["r4.cell1", "r4.cell2"]), "r4.qtree": (["r4qtree", "r4q1sh", "r4c3sh", "r4c4sh"], []),
           "r4.handoff": (["r4osh", "r4csh", "r4cin"], ["r4.qtree"])},
    "r4m": {"r4m.cell1": (["r4mpp", "r4m1sh"], []), "r4m.cell2": (["r4m2sh"], []),
            "r4m.quot": (["q4", "r4mqn"], ["r4m.cell1"]), "r4m.handoff": (["r4osh", "r4csh", "r4cin"], ["r4m.cell2"])},
}
DIGIT_LAWS = {"r2.cell1", "r2.quot", "r2m.cell", "r2m.handoff", "r4.cell1", "r4.cell2", "r4.quot", "r4m.cell1",
              "r4m.quot"}  # the laws that read digit rows (the others do not depend on e)
CELL_SCALES = (0, 1, 2)
DIGIT_SCALES = (0, 1, 2)
R4_WEIGHT = {"Ss": 1, "Sc": 1, "P0": 1, "P1": 2}  # natural weights of the radix-4 rows (P1 = a_2i+1 b)
R4_DIGIT = {"P0", "P1"}


class CellRows:
    """Synthesized words for the per-cell laws (no prompt, no label): independent random rows below 2^bits, digit,
    quotient and carry bits, and Blum moduli N = pq, p = q = 3 mod 4 (so N = 1 mod 4, the declared family). Values stay
    on the host for the targets; bit words go to the device."""

    def __init__(self, n: int, dev, bits: int = 18, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.n, self.dev = n, dev
        self.v = {k: rng.integers(0, 1 << bits, n) for k in ("Ss", "Sc", "P0", "P1", "b", "u", "v", "x")}
        self.v.update({k: rng.integers(0, 2, n) for k in ("d0", "d1", "q0", "q1", "cin")})
        self.v["N"] = blum_moduli(bits, n, seed)[0]
        v = self.v
        v["u_e"] = v["u"] ^ ((v["u"] + v["x"]) & 1)  # u + 2v + x even: what the radix-2 quotient guarantees
        inv = np.array([pow(int(N), -1, 4) for N in v["N"]])
        q = (-(v["u"] + 2 * v["v"]) * inv) % 4  # radix 4: u + 2v + qN = 0 mod 4
        v["q0r4"], v["q1r4"] = q & 1, q >> 1
        v["kr4m"] = (-(v["u"] + 2 * v["v"] + v["cin"]) * inv) % 4  # r4m: u + 2v + cin + kN = 0 mod 4

    def word(self, vals) -> torch.Tensor:
        return bin_words(vals, self.dev)

    def w(self, k: str) -> torch.Tensor:
        return self.word(self.v[k])

    def bitw(self, vals) -> torch.Tensor:
        return torch.as_tensor(np.where(vals == 1, -1, 0), dtype=torch.int64, device=self.dev)[None, :, None]

    def bit(self, k: str) -> torch.Tensor:
        return self.bitw(self.v[k])

    def sub(self, sl):
        return self  # one row group: a scale must fit every row at once


def cell_value(L: Lanes, u, v, cin=0):
    """u + 2v (+ cin) as one word (the synthesized rows keep it below 2^WMAX)."""
    return ks_add(L.cells, u, shl(v, 1), cin)[0]


def any_scale(lhs, targets) -> torch.Tensor:
    """Mismatch words [lanes or 1, 1]: a lane survives iff lhs equals one of the targets on every row."""
    dead = None
    for t in targets:
        m = or_reduce(or_reduce(lhs ^ t))
        dead = m if dead is None else dead & m
    return dead[..., None]


def low_bits_bad(tot, k: int) -> torch.Tensor:
    """Mismatch words [lanes or 1, 1]: any of the low k bits of tot set on any row."""
    return or_reduce(or_reduce(tot[..., :k]))[..., None]


def cell_law_bad(L: Lanes, r: CellRows, law: str, e: int = 0) -> torch.Tensor:
    """One sub-block law of the core on synthesized words -> mismatch words [lanes or 1, 1]; e = the iteration's
    digit scale (its digit rows count 2^e x their natural weight)."""
    g, v, wm = L.cells, r.v, L.wmask
    scaled = lambda vals: [r.word(vals << j) for j in CELL_SCALES]
    D = lambda x: x << e  # a digit row's value at the iteration's digit scale
    if law == "r2.cell1":  # u + 2v = 2^j (Ss + Sc + 2^e pp)
        u1, v1 = wired_cell(L, "l1src", "l1sh", (r.w("Ss"), r.w("Sc"), r.w("P0")))
        return any_scale(cell_value(L, u1, v1), scaled(v["Ss"] + v["Sc"] + D(v["P0"])))
    if law == "r2.quot":  # the quotient bit zeroes bit 0 of Ss + Sc + 2^e pp + qN
        Ss, Sc, N = r.w("Ss"), r.w("Sc"), r.w("N")
        u1, v1 = wired_cell(L, "l1src", "l1sh", (Ss, Sc, r.w("P0")))
        par = r.word((v["Ss"] + v["Sc"] + D(v["P0"])) & 1)[..., :1]
        return low_bits_bad(g.XOR(par, g.AND(r2_quotient(L, Ss, Sc, u1, v1), N[..., :1])), 1)
    if law == "r2.cell2":  # s + 2c = 2^j (u + 2v + qN)
        s, c = wired_cell(L, "l2src", "l2sh", (r.w("u"), r.w("v"), r.w("x")))
        return any_scale(cell_value(L, s, c), scaled(v["u"] + 2 * v["v"] + v["x"]))
    if law == "r2.handoff":  # the new state rows hold (u + 2v + qN) / 2 exactly
        s, c = wired_cell(L, "l2src", "l2sh", (r.w("u_e"), r.w("v"), r.w("x")))
        nSs, nSc = handoff2(L, s, c, wm)
        return any_scale(ks_add(g, nSs, nSc, 0)[0], [r.word((v["u_e"] + 2 * v["v"] + v["x"]) >> 1)])
    if law in ("r2m.cell", "r2m.handoff"):  # the row encodes d b + q N: the value with the right quotient, then / 2
        Ss, Sc, b, N, d = r.w("Ss"), r.w("Sc"), r.w("b"), r.w("N"), r.bit("d0")
        rows = {"0": None, "b": b, "N": N, "b+N": masked(ks_add(g, b, N, 0)[0], wm)}
        u, c = r2m_cell(L, Ss, Sc, d, r2m_quotient(L, Ss, Sc, d, b[..., :1]), rows)
        s = v["Ss"] + v["Sc"] + D(v["d0"] * v["b"])
        s = s + (s & 1) * v["N"]
        if law == "r2m.cell":
            return any_scale(cell_value(L, u, c), scaled(s))
        return any_scale(ks_add(g, *handoff2(L, u, c, wm), 0)[0], [r.word(s >> 1)])
    if law in ("r4.cell1", "r4.cell2", "r4.quot"):
        rows = r4_rows(L, r.w("Ss"), r.w("Sc"), r.w("P0"), r.w("P1"))
        wt = {k: w << (e if k in R4_DIGIT else 0) for k, w in R4_WEIGHT.items()}  # each row's weight at this scale
        if law == "r4.cell1":  # u1 + 2v1 = 2^j (the three rows r4hold lets through; digit rows at 2^e)
            u1, v1 = r4_cell1(L, rows)
            tot = {h: sum(wt[k] * v[k] for k in wt if k != h) for h in wt}
            return any_scale(cell_value(L, u1, v1),
                             [L.choose("r4hold", lambda h: h, lambda h, j=j: r.word(tot[h] << j)) for j in CELL_SCALES])
        if law == "r4.cell2":  # u2 + 2v2 = 2^j (u1 + 2v1 + cin + the held row at its class's scale)
            u2, v2 = r4_cell2(L, rows, r.w("u"), r.w("v"), r.bit("cin"))
            base = v["u"] + 2 * v["v"] + v["cin"]
            return any_scale(cell_value(L, u2, v2), [L.choose("r4hold", lambda h: h, lambda h, j=j: r.word(
                (base + wt[h] * v[h]) << j)) for j in CELL_SCALES])
        N = r.w("N")  # r4.quot: the 2-bit quotient zeroes the low two bits of Ss + Sc + 2^e (P0 + 2 P1) + cin + qN
        u1, v1 = r4_cell1(L, rows)
        u2, v2 = r4_cell2(L, rows, u1, v1, r.bit("cin"))
        q0, q1 = r4_quotient(L, u2, v2, N[..., 1:2])
        tot = ks_add(g, r.word(v["Ss"] + v["Sc"] + D(v["P0"] + 2 * v["P1"]) + v["cin"]), g.AND(q0, N), 0)[0]
        return low_bits_bad(ks_add(g, tot, shl(g.AND(q1, N), 1), 0)[0], 2)
    if law == "r4.qtree":  # u4 + 2v4 = 2^j (u2 + 2v2 + q0 N + 2 q1 N)
        u4, v4 = r4_cells34(L, r.w("u"), r.w("v"), r.bit("q0"), r.bit("q1"), r.w("N"))
        return any_scale(cell_value(L, u4, v4), scaled(v["u"] + 2 * v["v"] + (v["q0"] + 2 * v["q1"]) * v["N"]))
    if law == "r4.handoff":  # the value is 0 mod 4 (the quotient law's guarantee); the new state holds it / 4
        u4, v4 = r4_cells34(L, r.w("u"), r.w("v"), r.bit("q0r4"), r.bit("q1r4"), r.w("N"))
        nSs, nSc, ncin = handoff4(L, u4, v4, wm)
        tot = v["u"] + 2 * v["v"] + (v["q0r4"] + 2 * v["q1r4"]) * v["N"]
        return any_scale(ks_add(g, nSs, nSc, ncin)[0], [r.word(tot >> 2)])
    if law in ("r4m.cell1", "r4m.quot"):
        Ss, Sc, b, N = r.w("Ss"), r.w("Sc"), r.w("b"), r.w("N")
        mb, mn = r4m_tables(L, b, N, wm)
        u1, v1 = r4m_cell1(L, Ss, Sc, r.bit("d0"), r.bit("d1"), mb)
        s = v["Ss"] + v["Sc"] + D((v["d0"] + 2 * v["d1"]) * v["b"])
        if law == "r4m.cell1":  # u1 + 2v1 = 2^j (Ss + Sc + 2^e d b): the digit picks d b
            return any_scale(cell_value(L, u1, v1), scaled(s))
        v1s = shl_in(v1, r.bit("cin"))  # r4m.quot: the selected multiple of N zeroes the low two bits
        Q = perm_mux(L, "r4mqn", *r4m_quotient(L, u1, v1s, N[..., 1:2]), mn)
        return low_bits_bad(ks_add(g, r.word(s + v["cin"]), Q, 0)[0], 2)
    if law in ("r4m.cell2", "r4m.handoff"):  # the quotient row is the lane's table entry: k N for code q0 + 2 q1
        _, mn = r4m_tables(L, r.w("b"), r.w("N"), wm)
        base = v["u"] + 2 * v["v"] + v["cin"]
        mult = {"0": 0, "N": 1, "2N": 2, "3N": 3}
        if law == "r4m.cell2":  # u2 + 2v2 = 2^j (u1 + 2v1 + cin + k N)
            u2, v2 = r4m_cell2(L, r.w("u"), shl_in(r.w("v"), r.bit("cin")), r.bit("q0"), r.bit("q1"), mn)
            k = lambda p: np.array([mult[p[c]] for c in v["q0"] + 2 * v["q1"]])
            return any_scale(cell_value(L, u2, v2), [L.choose("r4mqn", lambda p: p, lambda p, j=j: r.word(
                (base + k(p) * v["N"]) << j)) for j in CELL_SCALES])
        # r4m.handoff: each lane's code for the multiple that makes the value 0 mod 4; the new state holds it / 4
        code = lambda p: np.array([p.index(["0", "N", "2N", "3N"][k]) for k in v["kr4m"]])
        q0 = L.choose("r4mqn", lambda p: p, lambda p: r.bitw(code(p) & 1))
        q1 = L.choose("r4mqn", lambda p: p, lambda p: r.bitw(code(p) >> 1))
        u2, v2 = r4m_cell2(L, r.w("u"), shl_in(r.w("v"), r.bit("cin")), q0, q1, mn)
        nSs, nSc, ncin = handoff4(L, u2, v2, wm)
        return any_scale(ks_add(g, nSs, nSc, ncin)[0], [r.word((base + v["kr4m"] * v["N"]) >> 2)])
    raise ValueError(law)


def join(parts: list[list[dict[str, int]]]) -> list[dict[str, int]]:
    """Natural join of survivor lists on their shared factors."""
    out = [{}]
    for part in parts:
        out = [dict(a, **b) for a in out for b in part if all(a[f] == b[f] for f in a.keys() & b.keys())]
    return out


def cell_stage(fam: str, free: list[str], rows: CellRows, dev, chunk: int, log: dict,
               base: dict[str, int] | None = None, scales=DIGIT_SCALES) -> list[dict[str, int]]:
    """Screen one family's wiring sub-block by sub-block: each law runs on (the survivors of the laws it extends) x
    (its own free factors), once per digit scale e if it reads digit rows (else once). Every sub-block of a wiring
    must pass at the same e. Returns the joined wiring survivors (option indices of the screened factors), the union
    over e."""
    surv, keys, rec, out, per_e = {}, {}, {}, {}, {}
    terminal = [law for law in CELL_LAWS[fam] if not any(law in deps for _, deps in CELL_LAWS[fam].values())]
    at = lambda law, e: (law, e if law in DIGIT_LAWS else scales[0])
    for e in scales:
        for law, (own, deps) in CELL_LAWS[fam].items():
            if at(law, e) in surv:
                continue
            own = [f for f in own if f in free]
            keys[law] = list(dict.fromkeys([f for d in deps for f in keys[d]] + own))
            space = Space.product(own, base)
            if deps:
                space = Space.of(join([surv[at(d, e)] for d in deps]), base).times(space)
            st, t0 = Stats(), time.time()
            if space.size:
                ids = screen(space, lambda L, r, law=law, e=e: cell_law_bad(L, r, law, e), "bad", [rows], dev, chunk, st)
                surv[at(law, e)] = [{f: c[f] for f in keys[law]} for c in space.combos(ids)]
            else:
                surv[at(law, e)] = []
            name = f"{law} e={e}" if law in DIGIT_LAWS else law
            rec[name] = {"factors": own, "machines": space.size, "survivors": len(surv[at(law, e)]),
                         "seconds": time.time() - t0}
        joined = join([surv[at(t, e)] for t in terminal])
        per_e[e] = len(joined)
        out.update((tuple(sorted(c.items())), c) for c in joined)
    out = list(out.values())
    log[fam] = {"laws": rec, "wiring_survivors": len(out), "wiring_by_digit_scale": per_e}
    print(f"cell laws {fam}: " + ", ".join(f"{k} {v['machines']:,} -> {v['survivors']:,}" for k, v in rec.items())
          + f"; wiring {len(out):,} (by digit scale e: {per_e})", flush=True)
    return out


# ---------------------------------------------------------------- width law
class _Bound:
    """One MonPro iteration, abstractly, for the width law. A signal carries an upper bound on its value and on its
    bit width. A cell keeps the carry-save relation u + 2v = (the sum of its ports), so a value bound is taken by
    pushing an objective back through the cells, splitting each cell's sum between u and v the worst way (the two
    state rows likewise share one bound V on Ss + Sc). Widths follow the bitwise cell: u is as wide as its widest
    port, v as its second widest."""

    def __init__(self, prims: dict[str, tuple[int, int]], V: float, W: int) -> None:
        self.cells: list[list[list[tuple[str, float]]]] = []
        self.val = {k: x for k, (x, _) in prims.items()}
        self.wid = {k: w for k, (_, w) in prims.items()}
        self.V, self.W, self.lost = V, W, []

    def bound(self, obj: dict[str, float]) -> float:
        coef = dict(obj)
        for k in range(len(self.cells) - 1, -1, -1):
            w = max(coef.pop(f"u{k}", 0.0), coef.pop(f"v{k}", 0.0) / 2)
            for port in self.cells[k] if w else []:
                for s, c in port:
                    coef[s] = coef.get(s, 0.0) + w * c
        state = max(coef.pop("Ss", 0.0), coef.pop("Sc", 0.0)) * self.V
        return state + sum(c * self.val[s] for s, c in coef.items())

    def width(self, terms) -> int:  # terms occupy disjoint bits (a shifted row, or 2v + cin)
        return max(max(self.wid[s] + int(math.log2(c)), 0) for s, c in terms)

    def fits(self, label: str, terms: list[tuple[str, float]], bits: int) -> None:
        """No set bit at or above `bits` in sum(c * s): the value bound or the width bound shows it."""
        if self.bound(dict(terms)) >= 2 ** bits and self.width(terms) > bits:
            self.lost.append(label)

    def shifted(self, label: str, terms: list[tuple[str, float]], k: int) -> list[tuple[str, float]]:
        """terms << k on a WMAX-bit word."""
        if k > 0:
            self.fits(f"{label} << {k}", terms, WMAX - k)
        return [(s, c * 2 ** k) for s, c in terms]

    def cell(self, ports: list[list[tuple[str, float]]]) -> tuple[str, str]:
        k = len(self.cells)
        self.cells.append(ports)
        ws = sorted((self.width(p) for p in ports), reverse=True)
        self.wid[f"u{k}"], self.wid[f"v{k}"] = ws[0], ws[1]
        return f"u{k}", f"v{k}"

    def carry_in(self, label: str, v: str, cin: str | None = "cin") -> list[tuple[str, float]]:
        """v << 1 (with cin in bit 0) on a WMAX-bit word: v's top bit must be clear."""
        self.fits(f"{label} carry row << 1", [(v, 1.0)], WMAX - 1)
        return [(v, 2.0)] + ([(cin, 1.0)] if cin else [])

    def register(self, label: str, terms: list[tuple[str, float]]) -> None:
        self.fits(f"{label} into W = {self.W}", terms, self.W)


def _width_iteration(G: dict, B: _Bound):
    """One iteration of the lane's datapath on B (the same routing as the MonPro code); returns nSs + nSc as weights
    on the last cell's outputs, and the new state rows' widths."""
    fam, W = G["mult"], G["W"]
    if fam == "r2":
        src = ("Ss", "Sc", "pp")
        u1, v1 = B.cell([B.shifted(f"cell 1 port {j}", [(src[G["l1src"][j]], 1.0)], G["l1sh"][j])
                                   for j in range(3)])
        src = (u1, v1, "qN")
        last = B.cell([B.shifted(f"cell 2 port {j}", [(src[G["l2src"][j]], 1.0)], G["l2sh"][j])
                                 for j in range(3)])
        osh, csh = G["osh"], G["csh"]
    elif fam == "r2m":
        last = B.cell([B.shifted(f"cell port {j}", [(s, 1.0)], G["r2msh"][j])
                               for j, s in enumerate(("Ss", "Sc", "row"))])
        osh, csh = G["osh"], G["csh"]
    elif fam == "r4":
        rows = {"Ss": [("Ss", 1.0)], "Sc": [("Sc", 1.0)], "P0": [("P0", 1.0)],
                "P1": B.shifted("P1", [("P1", 1.0)], G["r4p1sh"])}
        first = [k for k in rows if k != G["r4hold"]]
        u1, v1 = B.cell([B.shifted(f"cell 1 port {k}", rows[k], G["r4c1sh"][j]) for j, k in enumerate(first)])
        sh = G["r4c2sh"]
        u2, v2 = B.cell([B.shifted("cell 2 port u", [(u1, 1.0)], sh[0]),
                                   B.shifted("cell 2 port v", B.carry_in("cell 1", v1), sh[1]),
                                   B.shifted(f"cell 2 port {G['r4hold']}", rows[G["r4hold"]], sh[2])])
        src = {"u": [(u2, 1.0)], "v": B.carry_in("cell 2", v2, None), "Q0": [("Q0", 1.0)],
               "Q1": B.shifted("Q1", [("Q1", 1.0)], G["r4q1sh"])}
        order, sh3, sh4 = qtree_order(G["r4qtree"]), G["r4c3sh"], G["r4c4sh"]
        u3, v3 = B.cell([B.shifted(f"cell 3 port {k}", src[k], sh3[j]) for j, k in enumerate(order[:3])])
        last = B.cell([B.shifted("cell 4 port u", [(u3, 1.0)], sh4[0]),
                       B.shifted("cell 4 port v", B.carry_in("cell 3", v3, None), sh4[1]),
                       B.shifted(f"cell 4 port {order[3]}", src[order[3]], sh4[2])])
        osh, csh = G["r4osh"], -G["r4csh"]
    else:  # r4m
        u1, v1 = B.cell([B.shifted(f"cell 1 port {j}", [(s, 1.0)], G["r4m1sh"][j])
                                   for j, s in enumerate(("Ss", "Sc", "M"))])
        sh = G["r4m2sh"]
        last = B.cell([B.shifted("cell 2 port u", [(u1, 1.0)], sh[0]),
                                 B.shifted("cell 2 port v", B.carry_in("cell 1", v1), sh[1]),
                                 B.shifted("cell 2 port Q", [("Q", 1.0)], sh[2])])
        osh, csh = G["r4osh"], -G["r4csh"]
    u, v = last
    if csh > 0:  # the radix-2 carry row shifted left on a WMAX-bit word
        B.fits("hand-off carry row << 1", [(v, 1.0)], WMAX - csh)
    B.register("hand-off sum row", [(u, 2.0 ** -osh)])
    B.register("hand-off carry row", [(v, 2.0 ** csh)])
    return {u: 2.0 ** -osh, v: 2.0 ** csh}, {"Ss": min(W, max(B.wid[u] - osh, 0)), "Sc": min(W, max(B.wid[v] + csh, 0))}


def width_law(c: dict[str, int], nbits: int) -> str:
    """Per-cell width law over the declared input range: moduli N < 2^nbits, operands below the Montgomery closure
    bound s* (s' < s^2/R + N keeps every operand of a squaring chain below the fixed point of s^2/R + N, R = 2^(2 Ke -
    Kx) as the MonPro laws pin it), so multiplier bits at or above s*'s bit length are 0 and every other digit is all
    ones, quotient digits at their maximum. Every cell's ports, carry rows, hand-off registers, precomputed multiples and
    the final sum are bounded for all iterations; returns '' when no set bit can fall off a word or a W-bit register,
    else where one can."""
    G = {f: GRAMMAR[f][o] for f, o in c.items()}
    fam, W, Km = G["mult"], G["W"], G["Km"]
    k = 2 * G["Ke"] - G["Kx"]
    R, Nmax = 1 << k, (1 << nbits) - 1
    if R <= 4 * Nmax:
        return f"R = 2^{k} <= 4 N: MonPro outputs can leave [0, 2N)"
    bmax = (R - math.isqrt(R * R - 4 * R * Nmax)) // 2
    iters, per = (Km // 2, 2) if fam in ("r4", "r4m") else (Km, 1)
    if bmax.bit_length() > iters * per:
        return f"multiplier bits beyond the {iters * per} scanned"
    wb, wn = bmax.bit_length(), nbits
    val = {"0": 0, "b": bmax, "2b": 2 * bmax, "3b": 3 * bmax, "N": Nmax, "b+N": bmax + Nmax}
    prim = lambda x: (x, x.bit_length())

    def prims_at(i):
        """Iteration i's rows: a digit bit can be set only below s*'s bit length (ppdir picks which bits i scans)."""
        pos = [per * i + j for j in range(per)]
        on = [(KMAX - 1 - p if G["ppdir"] == "msb_first" else p) < wb for p in pos]
        bits = lambda ok: (0, 1) if ok else (0,)
        codes = [a + 2 * b for a in bits(on[0]) for b in bits(on[-1] if per == 2 else True)]  # r2m: (d, q)
        table = G["r2mrow"] if fam == "r2m" else G["r4mpp"]
        return {"pp": prim(bmax * on[0]), "P0": prim(bmax * on[0]), "P1": prim(bmax * on[-1]), "qN": (Nmax, wn),
                "Q0": (Nmax, wn), "Q1": (Nmax, wn), "cin": (1, 1), "Q": prim(3 * Nmax),
                "row": prim(max(val[table[c]] for c in codes)), "M": prim(max(val[table[c]] for c in codes))}

    pre = {"r2m": {"b + N": bmax + Nmax}, "r4m": {"2b": 2 * bmax, "3b": 3 * bmax, "2N": 2 * Nmax, "3N": 3 * Nmax}}
    for name, x in pre.get(fam, {}).items():
        if x >= 2 ** min(W, WMAX):
            return f"precomputed {name} beyond W = {W}"
    V, ws = 0.0, {"Ss": 0, "Sc": 0}
    for i in range(iters):
        B = _Bound(dict(prims_at(i), Ss=(0, ws["Ss"]), Sc=(0, ws["Sc"])), V, W)
        obj, ws = _width_iteration(G, B)
        if B.lost:
            return f"iteration {i}: {B.lost[0]}"
        V = B.bound(obj)
    cin = 1 if fam in ("r4", "r4m") else 0
    if V + cin >= 2 ** W and max(ws.values()) + 1 > W:
        return f"final sum into W = {W}"
    return ""


# Factors width_law never reads: the gate and the io block, the entry radix and exit quotient, the operand roles (a
# squaring reads one operand), the quotient formulas and tables (each quotient row is bounded by N, or 3N in r4m), the
# carry forms (cin is one bit) and the counter.
WIDTH_IGNORES = set(IO) | {"entry_radix", "exq", "ppop", "qsrc", "r2mq", "q4", "r4mqn", "r4cin", "cw"}


def width_verdicts(combos: list[dict[str, int]], nbits: int) -> tuple[list[str], int]:
    """width_law per machine, computed once per distinct setting of the factors it reads (all but WIDTH_IGNORES);
    also returns how many settings that was."""
    keys = [f for f in GRAMMAR if f not in WIDTH_IGNORES]
    memo: dict[tuple, str] = {}
    out = []
    for c in combos:
        k = tuple(c[f] for f in keys)
        if k not in memo:
            memo[k] = width_law(c, nbits)
        out.append(memo[k])
    return out, len(memo)


def lane_row_bad(L: Lanes, reg: torch.Tensor, want: np.ndarray) -> torch.Tensor:
    """Mismatch words [lanes or 1, rows]: bit j of word w set iff lane 64w+j gets that row's scored digits wrong."""
    yd = torch.as_tensor(digits_of(want, NB), device=L.dev)
    nd = torch.as_tensor(np.array([len(str(int(v))) for v in want]), device=L.dev)
    bits = ((yd[..., None] >> torch.arange(4, device=L.dev)) & 1).flatten(1)
    scored = (torch.arange(NB, device=L.dev)[None] < nd[:, None])[..., None].expand(-1, -1, 4).flatten(1)
    diff = (reg ^ word_bits(bits)) & word_bits(scored.long())
    while diff.shape[-1] > 1:
        diff = diff[..., 0::2] | diff[..., 1::2]
    return diff[..., 0]


def or_reduce(x: torch.Tensor) -> torch.Tensor:
    """Bitwise OR over the last dim."""
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = torch.cat([x, torch.zeros_like(x[..., :1])], -1)
        x = x[..., 0::2] | x[..., 1::2]
    return x[..., 0]


def alive_from_bad(L: Lanes, bad: torch.Tensor) -> torch.Tensor:
    """Mismatch words [lanes or 1, rows] -> bool [n]: lanes with no mismatch on any row."""
    bad = or_reduce(bad).expand(L.NW)
    dead = ((bad[:, None] >> L._shifts[None]) & 1).flatten()
    return dead[: L.n] == 0


def alive_lanes(L: Lanes, reg: torch.Tensor, want: np.ndarray) -> torch.Tensor:
    """Lanes whose BCD output equals `want` on every row's scored (low len(str(v))) digits -> bool [n]."""
    return alive_from_bad(L, lane_row_bad(L, reg, want))


def alive_equal(L: Lanes, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Lanes whose two binary words agree on every row (the two sides of a law) -> bool [n]."""
    return alive_from_bad(L, or_reduce(p ^ q))


def lane_correct_counts(L: Lanes, reg: torch.Tensor, want: np.ndarray) -> torch.Tensor:
    """Number of exactly-right rows per lane -> long [n]."""
    bad = lane_row_bad(L, reg, want).expand(L.NW, -1)
    wrong = ((bad[:, None, :] >> L._shifts[None, :, None]) & 1).flatten(0, 1)[: L.n]
    return (wrong == 0).sum(1)


# ---------------------------------------------------------------- data
def load(split: str) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in open(os.path.join(DATA, split + ".jsonl"))]
    get = lambda k: np.array([r[k] for r in rows], dtype=np.int64)
    return {"N": get("modulus"), "X": get("x"), "T": get("time_steps"), "Y": get("result")}


class Rows:
    def __init__(self, d: dict, idx, dev) -> None:
        self.N, self.X, self.T, self.Y = (d[k][idx] for k in ("N", "X", "T", "Y"))
        t = lambda a, n: torch.as_tensor(digits_of(a, n), device=dev)
        self.Nd, self.Xd, self.Td = t(self.N, ND), t(self.X, ND), t(self.T, 2)
        self.n = len(self.N)

    def sub(self, sl):
        r = Rows.__new__(Rows)
        r.N, r.X, r.T, r.Y = self.N[sl], self.X[sl], self.T[sl], self.Y[sl]
        r.Nd, r.Xd, r.Td = self.Nd[sl], self.Xd[sl], self.Td[sl]
        r.n = len(r.N)
        return r


def bin_words(vals, dev) -> torch.Tensor:
    v = np.asarray(vals, dtype=np.int64)
    return torch.as_tensor(np.where((v[:, None] >> np.arange(WMAX)) & 1, -1, 0)[None], dtype=torch.int64, device=dev)


class LawRows:
    """(N, x) pairs as binary words for the MonPro laws: the prompts' own numbers (the io block is pinned by stage A
    and read nowhere here; no label is read), or synthesized moduli. Also the constant 1 and N - x."""

    def __init__(self, N, X, dev) -> None:
        self.N, self.X = np.asarray(N, dtype=np.int64), np.asarray(X, dtype=np.int64)
        self.Nb, self.Xb, self.Mb = bin_words(self.N, dev), bin_words(self.X, dev), bin_words(self.N - self.X, dev)
        self.One = bin_words([1], dev)
        self.n = len(self.N)

    def sub(self, sl):
        r = LawRows.__new__(LawRows)
        r.N, r.X, r.Nb, r.Xb, r.Mb, r.One = self.N[sl], self.X[sl], self.Nb[:, sl], self.Xb[:, sl], self.Mb[:, sl], self.One
        r.n = len(r.N)
        return r


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def blum_moduli(bits: int, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """n synthesized (N, x): N = pq of exactly `bits` bits with primes p = q = 3 mod 4 (the generator's declared
    family), x uniform in [1, N)."""
    rng = random.Random(seed * 1000 + bits)

    def prime3(b):
        while True:
            p = rng.getrandbits(b) | (1 << (b - 1)) | 3
            if is_prime(p):
                return p

    Ns, Xs = [], []
    while len(Ns) < n:
        p, q = prime3(bits // 2), prime3(bits - bits // 2)
        if p != q and (p * q).bit_length() == bits:
            Ns.append(p * q)
            Xs.append(rng.randrange(1, p * q))
    return np.array(Ns), np.array(Xs)


# ---------------------------------------------------------------- search
class Stats:
    def __init__(self) -> None:
        self.machine_rows = 0
        self.seconds = 0.0
        self.warmup_seconds = 0.0  # first compiled call of a screen (includes compilation), kept out of the rate
        self.alive_after: list[int] = []  # survivors after each row group, summed over chunks


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def screen(space: Space, run, want_fn, row_groups: list, dev, chunk: int, stats: Stats, ids=None,
           compiled: bool = False) -> torch.Tensor:
    """Run every combo of `space` (or `ids`) on row groups in turn, dropping lanes at the first mismatch. `run` returns
    BCD words compared with want_fn(rows), or (with want_fn None) the two sides of a law, compared with each other, or
    (with want_fn "bad") mismatch words directly. With `compiled`, the first (bulk) row group of each chunk runs
    through torch.compile at a fixed padded lane count."""
    total = space.size if ids is None else ids.numel()
    width = min(chunk, total + (-total) % 64)
    crun = torch.compile(run, dynamic=False) if compiled else None
    warm = crun is not None
    out = []
    for start in range(0, total, chunk):
        cur = (torch.arange(start, min(start + chunk, total), device=dev, dtype=torch.int64) if ids is None
               else ids[start:start + chunk])
        for g, rows in enumerate(row_groups):
            if cur.numel() == 0:
                break
            bulk = crun is not None and g == 0
            rng = (start, start + cur.numel() - 1) if ids is None and g == 0 else None
            L = Lanes(space, cur, pad_to=width if bulk else 0, cache=not bulk, id_range=rng)
            t0 = time.perf_counter()
            res = (crun if bulk else run)(L, rows)
            keep = (alive_from_bad(L, res) if want_fn == "bad" else alive_equal(L, *res) if want_fn is None
                    else alive_lanes(L, res, want_fn(rows)))
            sync(dev)
            if bulk and warm:
                stats.warmup_seconds += time.perf_counter() - t0
                warm = False
            else:
                stats.seconds += time.perf_counter() - t0
                stats.machine_rows += cur.numel() * rows.n
            cur = cur[keep]
            if len(stats.alive_after) <= g:
                stats.alive_after.append(0)
            stats.alive_after[g] += cur.numel()
        out.append(cur)
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.int64, device=dev)


def free_switches(combos: list[dict[str, int]]) -> dict[str, list]:
    """Factors a survivor set leaves unpinned, with the surviving values."""
    out = {}
    for f in GRAMMAR:
        vals = sorted({c[f] for c in combos})
        if len(vals) > 1:
            out[f] = [hex(GRAMMAR[f][v]) if f == "gate" else GRAMMAR[f][v] for v in vals]
    return out


def evaluate_lanes(combos: list[dict[str, int]], rows: Rows, dev, batch: int = 2048, lanes: int = 4096) -> np.ndarray:
    """Exactly-right row counts for explicit machines on many rows (counter-halted loop, any T). Machines are run per
    multiplier family, `lanes` at a time, so a family's datapath never runs on another family's lanes. A part with
    fewer lanes takes proportionally more rows per pass (the same lane words x rows), so few machines don't run as
    many small passes."""
    out = np.zeros(len(combos), dtype=np.int64)
    fams = collections.defaultdict(list)
    for i, c in enumerate(combos):
        fams[c["mult"]].append(i)
    for idx in fams.values():
        for s in range(0, len(idx), lanes):
            part = idx[s:s + lanes]
            space = Space.of([combos[i] for i in part])
            ids = torch.arange(len(part), device=dev)
            total = torch.zeros(len(part), dtype=torch.long, device=dev)
            step = batch * max(1, lanes // (len(part) + (-len(part)) % 64))
            for start in range(0, rows.n, step):
                r = rows.sub(slice(start, start + step))
                L = Lanes(space, ids, id_range=(0, len(part) - 1))
                total += lane_correct_counts(L, machine(L, r.Nd, r.Xd, r.Td, (1 << TWMAX) - 1), r.Y)
            out[part] = total.cpu().numpy()
    return out


def bench(dev, chunk: int, rows_list=(1, 4, 16, 64), repeat: int = 2, compiled: bool = False) -> list[dict]:
    """Raw speculation throughput (no early rejection): candidate machines x rows per second, warm."""
    tr = load("train")
    lab = Rows(tr, np.where(tr["T"] == 1)[0][:256], dev)
    cases = [
        ("gate: 65,536 per-lane cells, reference wiring, full T=1 machine", Space.product(["gate"])),
        ("wiring: 979,776 core combos, full adder, full T=1 machine", Space.product(CORE_SMALL)),
        ("joint: per-lane cell x qsrc x l2sh (12.4M), full T=1 machine", Space.product(["gate", "qsrc", "l2sh"])),
        ("round trip: per-lane cell x io (188.7M), d2b + b2d", Space.product(IO)),
    ]
    out = []
    for name, space in cases:
        for R in rows_list:
            n = min(space.size, max(chunk // R, 64))
            ids = torch.arange(n, device=dev)
            r = lab.sub(slice(0, R))
            vd = r.Nd
            fn = (lambda L, vd: roundtrip_io(L, vd)) if "round trip" in name else (
                lambda L, rr: machine(L, rr.Nd, rr.Xd, rr.Td, 1))
            arg, want = (r.Nd, r.N) if "round trip" in name else (r, r.Y)
            if compiled:
                fn = torch.compile(fn, dynamic=False)
            best, first = float("inf"), None
            for _ in range(repeat + (1 if compiled else 0)):
                L = Lanes(space, ids, cache=not compiled, id_range=(0, n - 1))
                sync(dev)
                t0 = time.perf_counter()
                alive = alive_lanes(L, fn(L, arg), want)
                sync(dev)
                dt = time.perf_counter() - t0
                if first is None:
                    first = dt
                    if compiled:
                        continue
                best = min(best, dt)
            rec = {"case": name, "rows": R, "lanes": n, "seconds": best, "machine_rows_per_s": n * R / best,
                   "survivors": int(alive.sum()), "compiled": compiled, "first_call_seconds": first}
            out.append(rec)
            print(f"  {name} | rows={R} lanes={n:,}{' compiled' if compiled else ''}: {best:.3f}s -> "
                  f"{n * R / best:.3g} machine-rows/s (first call {first:.1f}s)", flush=True)
    return out


def alias_demo(dev) -> dict:
    """Ablation: drop the entry/exit round trip. T=1 labels then admit x^2 * 2^(2Ke - Km - Kx) = x^2 whenever
    2Ke = Km + Kx; T=2 rows (exponent 4Ke - 3Km - Kx) are what reject them."""
    tr = load("train")
    space = Space.product(["entry_radix", "Ke", "exq", "Kx", "Km"])
    ids = torch.arange(space.size, device=dev)
    rows1 = Rows(tr, np.where(tr["T"] == 1)[0][:64], dev)
    L = Lanes(space, ids)
    keep = alive_lanes(L, machine(L, rows1.Nd, rows1.Xd, rows1.Td, 1), rows1.Y)
    t1 = [space.combo(int(i)) for i in ids[keep].tolist()]
    cnt2 = evaluate_lanes(t1, Rows(tr, np.where(tr["T"] == 2)[0], dev), dev)
    t2 = [c for c, k in zip(t1, cnt2) if k == 1000]
    keys = ["entry_radix", "Ke", "exq", "Kx", "Km"]
    out = {"space": space.size, "t1_survivors": [describe(c, keys) for c in t1],
           "t2_survivors": [describe(c, keys) for c in t2]}
    print(f"alias demo: {space.size} entry/exit/Km machines (io and core at reference) vs 64 T=1 labels: {len(t1)} "
          f"survivors; after the 1,000 T=2 rows: {len(t2)}", flush=True)
    for c in t1:
        print("   T=1 survivor", describe(c, keys), "T=2 ok" if c in t2 else "T=2 rejected")
    return out


def fuse_probe(dev, lanes: int) -> list[dict]:
    """Eager vs torch.compile on the hot block only (Montgomery square + exit + double dabble): the front end is
    computed once eagerly, then the block is timed on the same inputs. Reports the fusion headroom of the eager
    numbers without compiling the whole machine as one graph."""
    tr = load("train")
    r = Rows(tr, np.where(tr["T"] == 1)[0][:1], dev)
    cases = [("core, full adder", Space.product(CORE_SMALL)),
             ("core + port wiring, full adder", Space.product(CORE)),
             ("core, per-lane cell x qsrc x l2sh", Space.product(["gate", "qsrc", "l2sh"]))]
    out = []
    for name, space in cases:
        n = min(space.size, lanes)
        ids = torch.arange(n, device=dev)
        L = Lanes(space, ids, id_range=(0, n - 1))  # the eager pass fills the mask cache; the compiled closure reads it
        wm = L.wmask
        N = d2b(L, r.Nd, WMAX, wm)
        s0 = entry(L, d2b(L, r.Xd, WMAX, wm), N, wm)

        def block(L, s, N):
            return b2d(L, exit_block(L, mont_square(L, s, N, L.wmask), N, L.wmask))

        def timed(f, k=2):
            best, res = float("inf"), None
            for _ in range(k):
                sync(dev)
                t0 = time.perf_counter()
                res = f(L, s0, N)
                sync(dev)
                best = min(best, time.perf_counter() - t0)
            return best, res

        te, re_ = timed(block)
        cf = torch.compile(lambda L_, s, N_: block(L, s, N_), dynamic=False)
        t0 = time.perf_counter()
        try:
            cf(L, s0, N)
        except Exception as e:  # e.g. the MPS backend's 31-buffer limit on large fused kernels
            print(f"  {name} | {n:,} lanes: eager {n / te:.3g}/s, compile failed: {type(e).__name__}", flush=True)
            out.append({"case": name, "lanes": n, "eager_s": te, "eager_machines_per_s": n / te,
                        "compile_error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        sync(dev)
        tcomp = time.perf_counter() - t0
        tc, rc = timed(cf)
        rec = {"case": name, "lanes": n, "eager_s": te, "compiled_s": tc, "compile_s": tcomp,
               "eager_machines_per_s": n / te, "compiled_machines_per_s": n / tc, "speedup": te / tc,
               "bit_exact": bool(torch.equal(re_.expand_as(rc), rc))}
        out.append(rec)
        print(f"  {name} | {n:,} lanes: eager {n / te:.3g}/s, compiled {n / tc:.3g}/s ({te / tc:.1f}x, "
              f"compile {tcomp:.0f}s, bit-exact {rec['bit_exact']})", flush=True)
    return out


def by_family(combos: list[dict[str, int]]) -> dict[str, int]:
    return dict(collections.Counter(GRAMMAR["mult"][c["mult"]] for c in combos))


def report(log: dict, stage: str, size: int, surv: list[dict[str, int]], st: Stats, t0: float, what: str) -> None:
    log[stage] = {"space": size, "survivors": len(surv), "seconds": time.time() - t0, "warmup_seconds": st.warmup_seconds,
                  "alive_after_row_groups": st.alive_after, "machine_rows": st.machine_rows,
                  "machine_rows_per_s": st.machine_rows / max(st.seconds, 1e-9), "by_family": by_family(surv),
                  "free": free_switches(surv) if surv else {}}
    print(f"{stage}: {size:,} (2^{math.log2(max(size, 1)):.1f}) machines vs {what}: {len(surv)} survivors "
          f"{log[stage]['by_family']}, {time.time() - t0:.1f}s, {log[stage]['machine_rows_per_s']:.3g} machine-rows/s; "
          f"free: {log[stage]['free']}", flush=True)


def stages_ab(args, tr: dict, prompts: Rows, dev, log: dict) -> list[dict[str, int]]:
    """A (gate x io vs the io round trip) and B (A's survivors x entry/exit vs the entry/exit round trip), both
    label-free on the prompts' own numbers. They depend only on --gates and --kw-max, so --ab-cache can store their
    survivors."""
    if args.ab_cache and os.path.exists(args.ab_cache):
        cache = json.load(open(args.ab_cache))
        if cache["gates"] == args.gates and cache.get("kw_max", 28) == args.kw_max:
            log["A"], log["B"] = cache["A"], cache["B"]
            print(f"A, B from {args.ab_cache}: {log['A']['survivors']:,} and {log['B']['survivors']} survivors "
                  f"({log['A']['seconds']:.1f}s + {log['B']['seconds']:.1f}s when computed)", flush=True)
            return cache["combosB"]
    vals = np.concatenate([prompts.N, prompts.X])
    vd = torch.as_tensor(digits_of(vals, ND), device=dev)

    class Nums:
        def __init__(self, sl):
            self.v, self.d, self.n = vals[sl], vd[sl], len(vals[sl])

    gate_opts = sorted(set(range(min(args.gates, 1 << 16))) | {FA_ID})
    spaceA = Space.product(IO, options={"gate": gate_opts})
    st, t0 = Stats(), time.time()
    survA = screen(spaceA, lambda L, r: roundtrip_io(L, r.d), lambda r: r.v,
                   [Nums(slice(0, 1)), Nums(slice(1, 2)), Nums(slice(2, 4)), Nums(slice(4, 16))], dev, args.chunk, st,
                   compiled="A" in args.compile)
    combosA = spaceA.combos(survA)
    report(log, "A", spaceA.size, combosA, st, t0, f"round trip b2d(d2b(v)) = v on {len(vals)} prompt numbers (no labels)")
    log["A"]["distinct_gates"] = len({c["gate"] for c in combosA})

    spaceB = Space.of([{f: c[f] for f in IO} for c in combosA]).times(Space.product(ENTRY_EXIT))
    st, t0 = Stats(), time.time()
    survB = screen(spaceB, lambda L, r: roundtrip_entry_exit(L, r.Nd, r.Xd), lambda r: r.X,
                   [prompts.sub(slice(0, 1)), prompts.sub(slice(1, 2)), prompts.sub(slice(2, 4)), prompts.sub(slice(4, 8))],
                   dev, args.chunk, st, compiled="B" in args.compile)
    combosB = spaceB.combos(survB)
    report(log, "B", spaceB.size, combosB, st, t0, "round trip exit(entry(x)) = x on 8 prompts (no labels)")
    if args.ab_cache:
        json.dump({"gates": args.gates, "kw_max": args.kw_max, "A": log["A"], "B": log["B"], "combosB": combosB},
                  open(args.ab_cache, "w"), default=str)
    return combosB


def stage_c0(args, combosB, prompts: Rows, fam_spaces: dict, laws: list[str], dev, log: dict) -> list[dict[str, int]]:
    """The core alone against label-free MonPro laws, once per distinct (gate, entry/exit) of B's survivors: the io
    block and the T counter are not read, so B's io aliases and the cw factor do not multiply this stage."""
    contexts = [dict(zip(CONTEXT, k)) for k in dict.fromkeys(tuple(c[f] for f in CONTEXT) for c in combosB)]
    law_rows = LawRows(prompts.N, prompts.X, dev)
    checks = [(law, law, [law_rows.sub(slice(0, 1)), law_rows.sub(slice(1, 2)), law_rows.sub(slice(2, 4)),
                          law_rows.sub(slice(4, law_rows.n))]) for law in laws]
    for bits in [int(b) for b in args.law_synth.split(",") if b]:
        syn = LawRows(*blum_moduli(bits, args.law_synth_n), dev)
        checks += [(f"{law}@{bits}bit", law, [syn.sub(slice(0, 1)), syn.sub(slice(1, 4)), syn.sub(slice(4, syn.n))])
                   for law in laws]
    log["C0"] = {"laws": [c[0] for c in checks], "contexts": len(contexts), "families": {}}
    out, t_all, size = [], time.time(), 0
    for fam, fsp in fam_spaces.items():
        space = Space.of(contexts, base={"mult": GRAMMAR["mult"].index(fam)}).times(fsp)
        size += space.size
        rec = {"space": space.size, "checks": []}
        surv = None
        for label, law, groups in checks:
            st, t0 = Stats(), time.time()
            n_in = space.size if surv is None else surv.numel()
            surv = screen(space, lambda L, r, law=law: law_sides(L, r, law), None, groups, dev, args.chunk, st, ids=surv,
                          compiled="0" in args.compile and n_in == space.size)
            rec["checks"].append({"check": label, "machines": n_in, "survivors": surv.numel(), "seconds": time.time() - t0,
                                  "warmup_seconds": st.warmup_seconds, "alive_after_row_groups": st.alive_after,
                                  "machine_rows": st.machine_rows,
                                  "machine_rows_per_s": st.machine_rows / max(st.seconds, 1e-9)})
            print(f"C0 {fam}: {n_in:,} (2^{math.log2(max(n_in, 1)):.1f}) machines vs law {label}: {surv.numel():,} "
                  f"survivors, {time.time() - t0:.1f}s, {rec['checks'][-1]['machine_rows_per_s']:.3g} machine-rows/s",
                  flush=True)
        if surv.numel() > args.join_cap:  # too many to join with labels: summarize on the host, skip stage C
            cols = space.decode(surv)
            rec.update(survivors=surv.numel(), too_many=True,
                       free={f: [GRAMMAR[f][v] for v in np.unique(cols[f])] for f in GRAMMAR if len(np.unique(cols[f])) > 1})
            log["C0"]["families"][fam] = rec
            log["C0"]["too_many"] = True
            continue
        combos = space.combos(surv)
        rec.update(survivors=len(combos), free=free_switches(combos) if combos else {})
        log["C0"]["families"][fam] = rec
        out += combos
    n_out = sum(r["survivors"] for r in log["C0"]["families"].values())
    log["C0"].update(space=size, survivors=n_out, seconds=time.time() - t_all,
                     by_family={f: r["survivors"] for f, r in log["C0"]["families"].items()})
    print(f"C0: {size:,} (2^{math.log2(size):.1f}) core machines x {len(contexts)} entry/exit contexts vs laws "
          f"{log['C0']['laws']} (no labels): {n_out:,} survivors {log['C0']['by_family']}, "
          f"{log['C0']['seconds']:.1f}s", flush=True)
    return None if log["C0"].get("too_many") else out


def c_spaces(combosB, c0, fam_spaces: dict) -> list[Space]:
    """Stage C's machines. With C0: each B survivor joined with the C0 survivors of its own (gate, entry/exit),
    x cw. Without: B's survivors x the whole core x cw (the first runs' stage C)."""
    keep = CONTEXT + IO[1:]
    if c0 is None:
        bs = [{f: b[f] for f in keep} for b in combosB]
        return [Space.of(bs, base={"mult": GRAMMAR["mult"].index(fam)}).times(fsp).times(Space.product(CONTROL))
                for fam, fsp in fam_spaces.items()]
    key = lambda c: tuple(c[f] for f in CONTEXT)
    cores, bs = collections.defaultdict(list), collections.defaultdict(list)
    for c in c0:
        cores[key(c)].append({f: c[f] for f in CORE_KEYS})
    for b in combosB:
        bs[key(b)].append({f: b[f] for f in keep})
    return [Space.of(bs[k]).times(Space.of(cs)).times(Space.product(CONTROL)) for k, cs in cores.items() if bs[k]]


def cell_factored_spaces(args, fam_spaces: dict, opts: dict, dev, log: dict) -> dict:
    """Replace each family's wiring factors by the survivors of its per-cell laws (cost |cell 1| + |cell 2| + ...
    instead of the product); the shared factors (ppdir, ppop, Km, W) stay a product."""
    rows, out, t0 = CellRows(32, dev), {}, time.time()
    scales = tuple(int(e) for e in args.digit_scales.split(","))
    log["cells"] = {"rows": rows.n, "digit_scales": scales, "families": {}}
    for fam, sp in fam_spaces.items():
        free = family_core(fam, args.core == "ports")
        base = {"mult": GRAMMAR["mult"].index(fam)}
        wiring = cell_stage(fam, free, rows, dev, args.chunk, log["cells"]["families"], base, scales)
        wired = sorted({f for c in wiring for f in c})
        rest = [f for f in free if f not in wired]
        out[fam] = Space.of(wiring, base).times(Space.product(rest, base, opts))
        log["cells"]["families"][fam].update(space_before=sp.size, space_after=out[fam].size)
    log["cells"]["seconds"] = time.time() - t0
    log["cells"]["machines"] = sum(r["machines"] for fr in log["cells"]["families"].values() for r in fr["laws"].values())
    print(f"cell laws: core {sum(sp.size for sp in fam_spaces.values()):,} -> "
          f"{sum(sp.size for sp in out.values()):,} machines to screen in C0 ({log['cells']['machines']:,} sub-block "
          f"machines, {log['cells']['seconds']:.1f}s)", flush=True)
    return out


def width_stage(combos: list[dict[str, int]], bits: int, log: dict, stage: str) -> list[dict[str, int]]:
    t0 = time.time()
    why, settings = width_verdicts(combos, bits)
    kept = [c for c, w in zip(combos, why) if not w]
    reasons = collections.Counter(w.split(": ", 1)[-1] for w in why if w)
    log[stage] = {"bits": bits, "machines": len(combos), "survivors": len(kept), "seconds": time.time() - t0,
                  "laws_evaluated": settings, "by_family": by_family(kept), "reasons": dict(reasons.most_common())}
    print(f"{stage}: width law at {bits}-bit moduli: {len(combos):,} -> {len(kept):,} {log[stage]['by_family']} "
          f"({settings:,} distinct settings of the factors it reads, {time.time() - t0:.1f}s); first violations: "
          f"{dict(reasons.most_common(6))}", flush=True)
    return kept


def semigroup_pass(combos: list[dict[str, int]], prompts: Rows, t1: int, t2: int, groups: list[slice], dev, chunk: int,
                   st: Stats) -> np.ndarray:
    """C1's verdict per machine: the semigroup law y(x, t1 + t2) = y(y(x, t1), t2) on the prompts, taken in the row
    groups `groups`, screened per multiplier family."""
    ok = np.zeros(len(combos), bool)
    for m in sorted({c["mult"] for c in combos}):
        idx = np.array([i for i, c in enumerate(combos) if c["mult"] == m])
        sp = Space.of([combos[i] for i in idx])
        surv = screen(sp, lambda L, r: semigroup_sides(L, r, t1, t2), None, [prompts.sub(g) for g in groups], dev,
                      chunk, st)
        ok[idx[surv.cpu().numpy()]] = True
    return ok


# ---------------------------------------------------------------- equivalence classes of the survivors (--collapse)
# What the laws already establish, used to certify one machine per class:
# - within one design (every other factor fixed), the shift settings that pass a family's value and hand-off laws
#   differ only in scale: the cell scale j, the digit scale e, and the hand-offs that undo a x2 cell. A value law fixes
#   each port's shift once its source and the scales are fixed.
# - the cell is the full adder, which is symmetric: permuting r2's port sources together with their shifts gives the
#   same outputs bit for bit.
# - with exact cells and no lost bit (the width law), MonPro is the canonical integer for R = 2^(Km - e) = 2^(2 Ke - Kx)
#   (the id law pins it), so the machine's answer is the same at every (R, W) the width law accepts.
SHIFTS = {"l1sh", "l2sh", "osh", "csh", "r2msh", "r4p1sh", "r4q1sh", "r4c1sh", "r4c2sh", "r4c3sh", "r4c4sh", "r4osh",
          "r4csh", "r4m1sh", "r4m2sh"}  # per-port, row and hand-off shifts
PORT_SOURCES = {"l1src": "l1sh", "l2src": "l2sh"}  # r2's port sources, with the shifts that go with each port
RW_FACTORS = ["Ke", "Kx", "Km", "W"]


def design_key(c: dict[str, int], rw: bool) -> tuple:
    """What a machine shares with the rest of its class: every factor but the shifts, r2's port sources as a multiset,
    and, with rw (every machine passed the width law), not (Ke, Kx, Km, W). The per-cell laws were screened with the
    full adder, so a machine with another cell is a class of its own."""
    if GRAMMAR["gate"][c["gate"]] != FA_ID:
        return tuple(sorted(c.items()))
    key = {f: o for f, o in c.items() if f not in SHIFTS and not (rw and f in RW_FACTORS)}
    for f in PORT_SOURCES:
        key[f] = GRAMMAR[f].index(tuple(sorted(GRAMMAR[f][c[f]])))
    return tuple(sorted(key.items()))


def scaling_of(c: dict[str, int]) -> tuple:
    """A machine's shifts, r2's in sorted port order (to count the scalings in a class)."""
    sh = {f: GRAMMAR[f][c[f]] for f in SHIFTS}
    for src, f in PORT_SOURCES.items():
        sh[f] = tuple(s for _, s in sorted(zip(GRAMMAR[src][c[src]], GRAMMAR[f][c[f]])))
    return tuple(sorted(sh.items()))


def classes_of(combos: list[dict[str, int]], rw: bool) -> tuple[list[int], list[list[int]]]:
    """Group machines by design_key, classes in order of first appearance. Returns each class's representative, the
    member closest to the reference design (fewest factors off REF), and its members, as indices into combos."""
    groups: dict = {}
    for i, c in enumerate(combos):
        groups.setdefault(design_key(c, rw), []).append(i)
    classes = list(groups.values())
    near = lambda i: (sum(combos[i][f] != REF_IDX[f] for f in GRAMMAR), [combos[i][f] for f in GRAMMAR])
    return [min(m, key=near) for m in classes], classes


def collapse(combos: list[dict[str, int]], rw: bool, log: dict) -> tuple[list[int], list[list[int]]]:
    """Group machines by design_key (classes_of) and log the classes."""
    t0 = time.time()
    reps, classes = classes_of(combos, rw)
    free = [f for f in GRAMMAR if len({c[f] for c in combos}) > 1]
    reads = lambda fam: set(["mult", "ppdir", "ppop", "Km", "W"] + CONTEXT + IO + FAMILY[fam] + CONTROL)
    k_of = lambda c: 2 * GRAMMAR["Ke"][c["Ke"]] - GRAMMAR["Kx"][c["Kx"]]
    recs = []
    for k, (m, r) in enumerate(zip(classes, reps)):
        cs, fam = [combos[i] for i in m], GRAMMAR["mult"][combos[r]["mult"]]
        recs.append({"class": k, "size": len(m), "family": fam,
                     "representative": describe(combos[r], [f for f in free if f in reads(fam)]),
                     "port_orders": len({tuple(c[f] for f in PORT_SOURCES) for c in cs}),
                     "scalings": len({scaling_of(c) for c in cs}),
                     "rw_pairs": len({tuple(c[f] for f in RW_FACTORS) for c in cs}),
                     "digit_scales": sorted({GRAMMAR["Km"][c["Km"]] - k_of(c) for c in cs}),
                     "varies": free_switches(cs)})
    shapes = collections.Counter((r["family"], r["size"], r["port_orders"], r["scalings"], r["rw_pairs"],
                                  tuple(r["digit_scales"])) for r in recs)
    log["collapse"] = {"machines": len(combos), "classes": len(classes), "rw_merged": rw, "seconds": time.time() - t0,
                       "by_family": by_family([combos[r] for r in reps]), "machines_by_family": by_family(combos),
                       "class_sizes": {str(s): n for s, n in sorted(collections.Counter(map(len, classes)).items())},
                       "class_shapes": [{"family": f, "size": s, "port_orders": p, "scalings": sc, "rw_pairs": q,
                                         "digit_scales": list(e), "classes": n}
                                        for (f, s, p, sc, q, e), n in sorted(shapes.items())],
                       "class_records": recs,
                       "members": {"factors": free, "rows": [[k] + [combos[i][f] for f in free]
                                                             for k, m in enumerate(classes) for i in m]}}
    print(f"collapse: {len(combos):,} machines -> {len(classes):,} classes {log['collapse']['by_family']} "
          f"({'shifts, port order, (Ke, Kx, Km, W)' if rw else 'shifts, port order; (R, W) kept: no width law'}; "
          f"{time.time() - t0:.1f}s); D and E run on one representative per class", flush=True)
    for (f, s, p, sc, q, e), n in sorted(shapes.items()):
        print(f"   {f}: {n} classes of {s:,} machines ({p} port orders, {sc} scalings, {q} (Ke, Kx, Km, W), digit "
              f"scales e = {list(e)})", flush=True)
    return reps, classes


def combos_from_audit(path: str) -> tuple[list[dict[str, int]], list[dict]]:
    """A previous run's stage-E audit (machines described by their free switches) as option-index combos."""
    d = json.load(open(path))
    audit = (d.get("out") or d)["audit"]

    def idx(f, v):
        return GRAMMAR[f].index(int(v, 16) if f == "gate" else tuple(v) if isinstance(v, list) else v)

    return [dict(REF_IDX, **{f: idx(f, v) for f, v in a["machine"].items()}) for a in audit], audit


def witness_rows(n_random: int, ks: list[int], n_moduli: int = 4, per_modulus: int = 200) -> dict[str, np.ndarray]:
    """Declared-range rows that stress a datapath's width, with exact answers (built as runs/cell_witness_ports.py and
    runs/cell_chain_rows.py build them, for 24-bit moduli): n_random rows with Blum N = pq near 2^24 (primes = 3 mod 4
    in (3000, 4096)), random x, T = 64; and for each R = 2^k, over every x of the n_moduli largest such N, the exact
    Montgomery chains (s0 = x R mod N, s <- (s^2 + mN)/R, T <= 64) of the first per_modulus x whose unreduced operand
    passes halfway from 2^24 to the closure bound s* (as runs/cell_chain_rows.py chose them) and of the per_modulus x
    reaching the largest operands, each at the squaring that reads that operand and at T = 64."""
    rng = random.Random(1)
    ps = [p for p in range(4095, 3000, -1) if p % 4 == 3 and is_prime(p)]
    Ns, Xs, Ts, src = [], [], [], []
    while len(Ns) < n_random:
        p, q = rng.sample(ps[:40], 2)
        if (p * q).bit_length() == 24:
            Ns.append(p * q), Xs.append(rng.randrange(1, p * q)), Ts.append(64), src.append("random")
    big = sorted({p * q for p in ps for q in ps if p < q and (p * q).bit_length() == 24}, reverse=True)[:n_moduli]
    nmax, info = (1 << 24) - 1, {}
    for k in ks:
        R, mask = 1 << k, (1 << k) - 1
        thr = ((1 << 24) + (R - math.isqrt(R * R - 4 * R * nmax)) // 2) // 2
        info[k] = {"threshold": thr}
        for N in big:
            ninv = (-pow(N, -1, R)) & mask
            x = np.arange(1, N, dtype=np.int64)
            s = (x * (R % N)) % N
            best, when, first = np.zeros_like(s), np.zeros_like(s), np.full(s.shape, -1, dtype=np.int64)
            for t in range(64):  # int64 products wrap above 2^63; the masks keep the exact low k bits
                m = (((s * s) & mask) * ninv) & mask
                s = (s * s + m * N) >> k
                when = np.where(s > best, t + 1, when)
                best = np.maximum(best, s)
                first = np.where((first < 0) & (s >= thr), t + 1, first)
            hits = np.nonzero(first > 0)[0]
            top = np.argsort(-best, kind="stable")[:per_modulus]
            info[k][int(N)] = {"past_threshold": len(hits), "max_s_over_N": round(float(best[top[0]] / N), 4),
                               "top_kept_down_to": round(float(best[top[-1]] / N), 4)}
            rows = {(int(x[i]), t) for i in hits[:per_modulus] for t in (min(int(first[i]) + 1, 64), 64)}
            rows |= {(int(x[i]), t) for i in top for t in (min(int(when[i]) + 1, 64), 64)}
            for xi, t in sorted(rows):
                Ns.append(N), Xs.append(xi), Ts.append(t), src.append(f"chain R=2^{k}")
    print(f"witness rows: {n_random} random near 2^24, chains {info}", flush=True)
    return {"N": np.array(Ns), "X": np.array(Xs), "T": np.array(Ts), "src": np.array(src), "chains": info,
            "Y": np.array([pow(int(x), 1 << int(t), int(N)) for x, N, t in zip(Xs, Ns, Ts)])}


def cell_audit(args, dev) -> dict:
    """The per-cell laws on a previous run's training-consistent machines: the first sub-block law each one fails (at
    the first digit scale; a machine passes at the first e where every law holds), the width law's verdict, against
    certification; with --cell-audit-sets, also within another check chain's survivors (a JSON with per-machine 'after'
    lists, the last one used); with --witness, also against declared-range witness rows."""
    combos, audit = combos_from_audit(args.cell_audit)
    cert = np.array([a["seen_maxT"] == 64 and a["ood_maxT"] == 64 for a in audit])
    fams = np.array([GRAMMAR["mult"][c["mult"]] for c in combos])
    rows, t0 = CellRows(32, dev), time.time()
    scales = tuple(int(e) for e in args.digit_scales.split(","))
    value_fail = np.array([""] * len(combos), dtype=object)
    scale = np.full(len(combos), -1)
    for fam in dict.fromkeys(fams):
        idx = np.nonzero(fams == fam)[0]
        sp = Space.of([combos[i] for i in idx])
        for e in scales:
            fail = np.array([""] * len(idx), dtype=object)
            for law in CELL_LAWS[fam]:
                surv = screen(sp, lambda L, r, law=law, e=e: cell_law_bad(L, r, law, e), "bad", [rows], dev,
                              args.chunk, Stats())
                keep = np.zeros(len(idx), bool)
                keep[surv.cpu().numpy()] = True
                fail[~keep & (fail == "")] = law
            if e == scales[0]:
                value_fail[idx] = fail
            scale[idx[(fail == "") & (scale[idx] < 0)]] = e
    value_fail[scale >= 0] = ""
    t_cells, t0 = time.time() - t0, time.time()
    width_fail = np.array([width_law(c, args.width_law or 24) for c in combos], dtype=object)
    t_width = time.time() - t0
    value_ok, width_ok = value_fail == "", width_fail == ""
    subsets = {"all": np.ones(len(combos), bool)}
    if args.cell_audit_sets:
        after = json.load(open(args.cell_audit_sets))["after"]
        subsets[f"after {list(after)[-1]}"] = np.array(after[list(after)[-1]])

    def counts(m):
        return {"machines": int(m.sum()), "certify": int((m & cert).sum()), "not_certify": int((m & ~cert).sum())}

    summary = {"seconds_cell_laws": t_cells, "seconds_width_law": t_width, "subsets": {}}
    for name, s in subsets.items():
        summary["subsets"][name] = {k: counts(s & m) for k, m in [
            ("before", np.ones(len(combos), bool)), ("value laws", value_ok), ("width law", width_ok),
            ("value + width", value_ok & width_ok)]}
        summary["subsets"][name]["value laws by digit scale"] = {
            str(e): counts(s & (scale == e)) for e in scales}
        summary["subsets"][name]["value law failures"] = dict(collections.Counter(value_fail[s & ~value_ok]))
        summary["subsets"][name]["width law failures"] = dict(collections.Counter(
            w.split(": ", 1)[-1] for w in width_fail[s & ~width_ok]))
        print(f"--- {name}", flush=True)
        for k, v in summary["subsets"][name].items():
            print(f"   {k}: {v}", flush=True)
    print(f"cell laws {t_cells:.1f}s, width law {t_width:.1f}s for {len(combos):,} machines", flush=True)
    machines = [{"value_law": str(vf), "digit_scale": int(e), "width_law": str(wf), "certify": bool(ce)}
                for vf, e, wf, ce in zip(value_fail, scale, width_fail, cert)]
    if args.witness:
        t0 = time.time()
        ks = sorted({2 * GRAMMAR["Ke"][c["Ke"]] - GRAMMAR["Kx"][c["Kx"]] for c in combos} & set(range(26, 39)))  # exact int64 chains
        d = witness_rows(args.witness, ks)
        wrong = {}
        for name in dict.fromkeys(d["src"]):
            sel = np.nonzero(d["src"] == name)[0]
            wrong[name] = len(sel) - evaluate_lanes(combos, Rows(d, sel, dev), dev, batch=4096)
            print(f"   witness rows {name} ({len(sel)}): machines with a wrong row {int((wrong[name] > 0).sum()):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        seen = sum((w > 0).astype(int) for w in wrong.values()) > 0
        tab = collections.Counter(
            " | ".join(("certify" if c else "not certify", f"value ok e={e}" if v else "value FAIL",
                        "width ok" if w else "width FAIL", "WITNESS" if x else "no witness"))
            for c, v, e, w, x in zip(cert, value_ok, scale, width_ok, seen))
        summary["witness"] = {"rows": dict(collections.Counter(d["src"].tolist())), "chains": d["chains"],
                              "seconds": time.time() - t0, "table": dict(sorted(tab.items())),
                              "width_law_matches_witness": int((width_ok == ~seen).sum()),
                              "width_ok_with_witness": int((width_ok & seen).sum()),
                              "width_fail_without_witness": int((~width_ok & ~seen).sum())}
        for i, m in enumerate(machines):
            m["wrong_rows"] = {k: int(w[i]) for k, w in wrong.items()}
        print(f"--- witness rows {summary['witness']['rows']} ({summary['witness']['seconds']:.0f}s)", flush=True)
        for k, n in sorted(tab.items()):
            print(f"   {k}: {n}", flush=True)
        print(f"   width law = witness/no witness on {summary['witness']['width_law_matches_witness']:,} of "
              f"{len(combos):,}", flush=True)
    return {"summary": summary, "machines": machines}


def audit_summary(combos: list[dict[str, int]], seen: np.ndarray, ood: np.ndarray) -> dict:
    """Per family, per free switch: value -> [machines, of which certify T=64 on both profiles]."""
    out = {}
    both = (seen == 64) & (ood == 64)
    for fam in dict.fromkeys(GRAMMAR["mult"][c["mult"]] for c in combos):
        idx = [i for i, c in enumerate(combos) if GRAMMAR["mult"][c["mult"]] == fam]
        rec = {"machines": len(idx), "certify_both": int(both[idx].sum())}
        for f in CONTEXT + IO[1:] + family_core(fam) + CONTROL:
            vals = collections.defaultdict(lambda: [0, 0])
            for i in idx:
                v = GRAMMAR[f][combos[i][f]]
                vals[hex(v) if f == "gate" else str(v)][0] += 1
                vals[hex(v) if f == "gate" else str(v)][1] += int(both[i])
            if len(vals) > 1:
                rec[f] = dict(vals)
        out[fam] = rec
    return out


def certify(combos: list[dict[str, int]], dev) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Both certification profiles, T = 1, 2, ..., 64: per machine and profile, the largest T up to which every row is
    exact, and the number of exact rows."""
    certs = {}
    for prof in ("depth_t_", "depth_ood_n_t_"):
        broken = np.zeros(len(combos), bool)
        maxT = np.zeros(len(combos), int)
        totals = np.zeros(len(combos), int)
        for T in (1, 2, 4, 8, 16, 32, 64):
            d = load(f"{prof}{T}")
            cnt = evaluate_lanes(combos, Rows(d, np.arange(len(d["N"])), dev), dev)
            ok = cnt == len(d["N"])
            maxT = np.where(~broken & ok, T, maxT)
            broken |= ~ok
            totals += cnt
        certs[prof] = (maxT, totals)
    return certs


def collapse_check(members: list[dict[str, int]], classes: list[list[int]], reps: list[int], reached: list[int],
                   certs: dict, per_class: int, dev, log: dict) -> None:
    """E on up to per_class random non-representatives of each class whose representative reached E (reached[j] = the
    class of E's machine j). Each must certify exactly as its representative does: the same max certified T and the
    same number of exact rows, on both profiles."""
    t0, rng = time.time(), random.Random(0)
    sample, of = [], []
    for j, k in enumerate(reached):
        others = [i for i in classes[k] if i != reps[k]]
        for i in rng.sample(others, min(per_class, len(others))):
            sample.append(i)
            of.append(j)
    got = certify([members[i] for i in sample], dev)
    outcome = lambda cs, j: [int(cs[p][q][j]) for p in ("depth_t_", "depth_ood_n_t_") for q in (0, 1)]
    free = [f for f in GRAMMAR if len({c[f] for c in members}) > 1]
    rows = []
    for s, (i, j) in enumerate(zip(sample, of)):
        o = outcome(got, s)
        rows.append({"class": reached[j], "machine": describe(members[i], free), "agrees": o == outcome(certs, j),
                     "seen_maxT": o[0], "seen_correct": o[1], "ood_maxT": o[2], "ood_correct": o[3]})
    agree = sum(r["agrees"] for r in rows)
    both = sum(r["seen_maxT"] == 64 and r["ood_maxT"] == 64 for r in rows)
    log["collapse"]["check"] = {"per_class": per_class, "sampled": len(rows), "classes": len(set(of)), "agree": agree,
                                "certify_both": both, "seconds": time.time() - t0, "machines": rows}
    print(f"collapse check: E on {len(rows):,} random non-representatives (up to {per_class} per class, "
          f"{len(set(of))} classes): {agree:,} certify exactly as their representative (max certified T and exact "
          f"rows on both profiles), {both:,} certify T=64 on both ({time.time() - t0:.1f}s)", flush=True)
    for r in [r for r in rows if not r["agrees"]][:20]:
        print("    disagrees:", r, flush=True)


def c1_check(pre: dict, per_group: int, prompts: Rows, t1: int, t2: int, dev, chunk: int, log: dict) -> None:
    """C1 on up to per_group random non-representatives of each (class, cw) group that --collapse C1 formed before C1
    (pre: stage C's survivors, the groups, their representatives and the representatives' verdicts). Each must pass or
    fail the semigroup law as its group's representative did."""
    t0, rng = time.time(), random.Random(0)
    combos, sample, of = pre["combos"], [], []
    for k, (m, r) in enumerate(zip(pre["classes"], pre["reps"])):
        others = [i for i in m if i != r]
        for i in rng.sample(others, min(per_group, len(others))):
            sample.append(i)
            of.append(k)
    ok = semigroup_pass([combos[i] for i in sample], prompts, t1, t2, [slice(0, 8)], dev, chunk, Stats())
    free = [f for f in GRAMMAR if len({c[f] for c in combos}) > 1]
    rows = [{"group": k, "machine": describe(combos[i], free), "passes": bool(o),
             "agrees": bool(o) == bool(pre["passed"][k])} for i, k, o in zip(sample, of, ok)]
    agree = sum(r["agrees"] for r in rows)
    log["C1"]["check"] = {"per_group": per_group, "sampled": len(rows), "groups": len(set(of)), "agree": agree,
                          "pass": int(ok.sum()), "seconds": time.time() - t0, "machines": rows}
    print(f"C1 check: the semigroup law on {len(rows):,} random non-representatives (up to {per_group} per (class, cw) "
          f"group, {len(set(of))} groups): {agree:,} pass or fail as their representative did, {int(ok.sum()):,} pass "
          f"({time.time() - t0:.1f}s)", flush=True)
    for r in [r for r in rows if not r["agrees"]][:20]:
        print("    disagrees:", r, flush=True)


def main() -> None:
    global DATA, WMAX, KMAX
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--device", default="mps")
    p.add_argument("--chunk", type=int, default=1 << 20)
    p.add_argument("--gates", type=int, default=1 << 16, help="gate hypotheses in stage A (first k tables + FA)")
    p.add_argument("--out", default="")
    p.add_argument("--bench", action="store_true", help="only measure raw speculation throughput")
    p.add_argument("--alias-demo", action="store_true", help="T=1 aliases without the entry/exit round trip")
    p.add_argument("--fuse-probe", action="store_true", help="eager vs torch.compile on the Montgomery/exit/b2d block")
    p.add_argument("--compile", default="", help="stages whose bulk pass is fused with torch.compile, e.g. AB0C")
    p.add_argument("--rows", default="1,4,16,64", help="bench: rows per pass")
    p.add_argument("--core", default="small", choices=["small", "ports"],
                   help="small = r2 without port sources (with --families r2 --ppop ab: the 2^54.4 grammar); ports adds "
                        "r2's port sources (2^63.9 that way) and per-port shifts on every cell of r2m, r4 and r4m")
    p.add_argument("--families", default="r2,r2m,r4,r4m", help="multiplier families in the core grammar")
    p.add_argument("--ppop", default="all", choices=["all", "ab"], help="ab fixes MonPro's operand roles (first runs)")
    p.add_argument("--law", default="id,comm",
                   help="label-free MonPro laws screened before the labels, in order (id, comm, assoc, mirror); "
                        "none = labels only (the first runs)")
    p.add_argument("--law-synth", default="", help="also screen the laws on synthesized Blum moduli of these bit lengths")
    p.add_argument("--law-synth-n", type=int, default=16)
    p.add_argument("--count-law", default="", help="T1,T2: after C, check y(x, T1+T2) = y(y(x, T1), T2) (no labels)")
    p.add_argument("--c-check", default="labels", choices=["labels", "decimal"],
                   help="stage C: 64 T=1 labels, or the label-free check that y(x, 1) prints decimal digits")
    p.add_argument("--stop-after", default="E", choices=["B", "C0", "C", "D", "E"])
    p.add_argument("--join-cap", type=int, default=2_000_000, help="max C0 survivors per family joined with labels")
    p.add_argument("--ablate", default="", help="law suites to compare through C0 and C, e.g. 'id;comm;unit,mirror'")
    p.add_argument("--ab-cache", default="", help="JSON file caching the A/B survivors")
    p.add_argument("--cell-law", action="store_true",
                   help="factor the core's wiring with per-cell laws (value per cell, quotient and hand-off per "
                        "iteration, on synthesized words) before C0")
    p.add_argument("--width-law", type=int, default=0,
                   help="BITS: drop core machines that can lose a bit on moduli below 2^BITS (after C0, or after C "
                        "without C0)")
    p.add_argument("--cell-audit", default="", help="run JSON: the per-cell laws on that run's stage-E machines")
    p.add_argument("--cell-audit-sets", default="", help="JSON of per-machine survivors of other checks to cross with")
    p.add_argument("--witness", type=int, default=0,
                   help="with --cell-audit: also run the audited machines on N declared-range witness rows (24-bit Blum "
                        "N near 2^24, T = 64) plus the exhaustive large-operand chain rows of each R (witness_rows)")
    p.add_argument("--digit-scales", default=",".join(map(str, DIGIT_SCALES)),
                   help="the per-cell value laws' digit scales e (0 = one scale per cell, the earlier law)")
    p.add_argument("--kw-max", type=int, default=32,
                   help="largest Ke, Kx, Km and W option, and the word width (28 = the earlier grammars)")
    p.add_argument("--collapse", nargs="?", const="D", default="", choices=["D", "C1"],
                   help="group the machines that reach D into classes the laws make equivalent (design_key) and run D "
                        "and E on one representative per class; --collapse C1 groups stage C's survivors already, and "
                        "C1 runs on one representative per (class, cw)")
    p.add_argument("--collapse-check", type=int, default=2,
                   help="with --collapse: also run E on up to this many random non-representatives per class (and, "
                        "with --collapse C1, C1 on up to this many per (class, cw))")
    args = p.parse_args()
    DATA = args.data
    WMAX = KMAX = args.kw_max
    LIMITS.update({f: [i for i, v in enumerate(GRAMMAR[f]) if v <= args.kw_max] for f in ("Ke", "Kx", "Km", "W")})
    dev = torch.device(args.device)
    if args.cell_audit:
        res = cell_audit(args, dev)
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1)
        return
    if args.fuse_probe:
        torch._dynamo.config.cache_size_limit = 256
        res = fuse_probe(dev, args.chunk)
        if args.out:
            json.dump({"device": str(dev), "torch": torch.__version__, "fuse_probe": res}, open(args.out, "w"), indent=1)
        return
    if args.alias_demo:
        res = alias_demo(dev)
        if args.out:
            json.dump(res, open(args.out, "w"), indent=1, default=str)
        return
    if args.bench:
        torch._dynamo.config.cache_size_limit = 256
        res = bench(dev, args.chunk, tuple(int(x) for x in args.rows.split(",")), compiled=bool(args.compile))
        if args.out:
            json.dump({"device": str(dev), "torch": torch.__version__, "chunk": args.chunk, "compiled": bool(args.compile),
                       "bench": res},
                      open(args.out, "w"), indent=1)
        return
    tr = load("train")
    T1 = np.where(tr["T"] == 1)[0]
    log: dict = {"device": str(dev), "torch": torch.__version__, "compiled": args.compile, "chunk": args.chunk,
                 "args": vars(args)}
    torch._dynamo.config.cache_size_limit = 256
    t_run = time.time()

    def dump():
        log["wall_seconds"] = time.time() - t_run
        if args.out:
            json.dump({"log": log, "audit": audit}, open(args.out, "w"), indent=1, default=str)

    audit: list = []
    prompts = Rows(tr, np.arange(8), dev)
    combosB = stages_ab(args, tr, prompts, dev, log)
    for c in combosB[:8]:
        print("   ", describe(c, IO + ENTRY_EXIT))
    if args.stop_after == "B":
        return dump()

    families = args.families.split(",")
    opts = {"ppop": [0, 1, 2, 3] if args.ppop == "all" else [0]}
    fam_spaces = {fam: Space.product(family_core(fam, args.core == "ports"), base={"mult": GRAMMAR["mult"].index(fam)},
                                     options=opts) for fam in families}
    core = sum(sp.size for sp in fam_spaces.values()) * len(GRAMMAR["cw"])
    gates = len(sorted(set(range(min(args.gates, 1 << 16))) | {FA_ID}))
    joint = gates * math.prod(len(LIMITS.get(f, GRAMMAR[f])) for f in IO[1:] + ENTRY_EXIT) * core
    log["grammar"] = {"families": {fam: {"factors": family_core(fam, args.core == "ports"), "size": sp.size}
                                   for fam, sp in fam_spaces.items()},
                      "core_with_cw": core, "joint": joint}
    print(f"core grammar: {core:,} = 2^{math.log2(core):.1f} machines incl. cw "
          f"({ {f: sp.size for f, sp in fam_spaces.items()} } x 3); joint space {joint:,} = 2^{math.log2(joint):.1f}",
          flush=True)
    if args.cell_law:
        fam_spaces = cell_factored_spaces(args, fam_spaces, opts, dev, log)

    lab = Rows(tr, T1[:64], dev)

    def stage_c(c0, into: dict) -> list[dict[str, int]]:
        """T=1 labels (or, with --c-check decimal, the label-free output-format law on the 8 prompts) on B's
        survivors joined with the core (C0's survivors, or the whole core)."""
        spacesC = c_spaces(combosB, c0, fam_spaces)
        st, t0 = Stats(), time.time()
        out = []
        for sp in spacesC:
            if args.c_check == "decimal":
                surv = screen(sp, decimal_sides, None, [prompts.sub(slice(0, 1)), prompts.sub(slice(1, 2)),
                                                        prompts.sub(slice(2, 8))], dev, args.chunk, st)
            else:
                surv = screen(sp, lambda L, r: machine(L, r.Nd, r.Xd, r.Td, 1), lambda r: r.Y,
                              [lab.sub(slice(0, 1)), lab.sub(slice(1, 2)), lab.sub(slice(2, 4)), lab.sub(slice(4, 64))],
                              dev, args.chunk, st, compiled="C" in args.compile)
            out += sp.combos(surv)
        report(into, "C", sum(sp.size for sp in spacesC), out, st, t0,
               "decimal output digits on 8 prompts (no labels)" if args.c_check == "decimal" else "64 T=1 labels")
        into["C"]["core_designs"] = {fam: len({tuple(c[f] for f in CORE_KEYS) for c in out if GRAMMAR["mult"][c["mult"]] == fam})
                                     for fam in families}
        return out

    if args.ablate:  # several law suites after one A/B pass; C0 and C only
        log["ablation"] = {}
        for suite in args.ablate.split(";"):
            sub: dict = {}
            print(f"--- law suite {suite}", flush=True)
            c0 = stage_c0(args, combosB, prompts, fam_spaces, suite.split(","), dev, sub)
            if c0 is None:
                print(f"    more than {args.join_cap:,} C0 survivors in a family: stage C skipped", flush=True)
            else:
                stage_c(c0, sub)
            log["ablation"][suite] = sub
            dump()
        return dump()

    laws = [] if args.law == "none" else args.law.split(",")
    c0 = stage_c0(args, combosB, prompts, fam_spaces, laws, dev, log) if laws else None
    if args.width_law and c0 is not None:
        c0 = width_stage(c0, args.width_law, log, "CW")
    if args.stop_after == "C0" or (laws and c0 is None):
        return dump()
    combosC = stage_c(c0, log)
    if args.width_law and c0 is None:
        combosC = width_stage(combosC, args.width_law, log, "CW")
    pre1 = None  # with --collapse C1: stage C's survivors in (class, cw) groups, for c1_check
    if args.count_law:  # C1: the T counter against the semigroup law, per family
        t1, t2 = map(int, args.count_law.split(","))
        st, t0 = Stats(), time.time()
        what = f"semigroup law y(x, {t1}+{t2}) = y(y(x, {t1}), {t2}) on 8 prompts (no labels)"
        if args.collapse == "C1":
            # one representative per (class, cw), all 8 prompts in one pass (with this few lanes, row groups for early
            # rejection would add passes without saving work)
            reps1, classes1 = classes_of(combosC, bool(args.width_law))
            t_group = time.time() - t0
            passed = semigroup_pass([combosC[r] for r in reps1], prompts, t1, t2, [slice(0, 8)], dev, args.chunk, st)
            ok = np.zeros(len(combosC), bool)
            for m, k in zip(classes1, passed):
                ok[m] = k
            pre1 = {"combos": combosC, "classes": classes1, "reps": reps1, "passed": passed}
            what += f", run on one representative per (class, cw): {len(reps1):,} groups, {int(passed.sum()):,} pass"
        else:
            ok = semigroup_pass(combosC, prompts, t1, t2, [slice(0, 1), slice(1, 2), slice(2, 8)], dev, args.chunk, st)
        kept = [c for fam in families for c, k in zip(combosC, ok) if k and GRAMMAR["mult"][c["mult"]] == fam]
        report(log, "C1", len(combosC), kept, st, t0, what)
        if pre1 is not None:
            sizes = collections.Counter(map(len, classes1))
            log["C1"]["collapsed"] = {"groups": len(reps1), "representatives_pass": int(passed.sum()),
                                      "grouping_seconds": t_group, "group_sizes": {str(s): sizes[s] for s in sorted(sizes)}}
        combosC = kept
    evaluated = log["A"]["space"] + log["B"]["space"] + log["C"]["space"] + log.get("cells", {}).get("machines", 0) + (
        sum(ch["machines"] for fr in log["C0"]["families"].values() for ch in fr["checks"]) if c0 is not None else 0)
    log["evaluated_machines"] = evaluated
    log["labels_only_C_would_be"] = len(combosB) * core
    print(f"machines evaluated {evaluated:,} = 2^{math.log2(evaluated):.1f} (stage C alone without C0: "
          f"{len(combosB) * core:,} = 2^{math.log2(len(combosB) * core):.1f})", flush=True)
    members, reps, classes = combosC, None, None
    if args.collapse:  # from here combosC[k] is class k's representative
        reps, classes = collapse(combosC, bool(args.width_law), log)
        combosC = [members[r] for r in reps]
    if args.stop_after == "C":
        return dump()
    what = "machines" if classes is None else "representatives"
    in_classes = lambda ks: sum(len(classes[k]) for k in ks)

    # stage D: every training row (T = 1, 2, 4, 8, 16), dropping a machine at its first inexact T
    t0 = time.time()
    alive = list(range(len(combosC)))
    log["D"] = {}
    for T in (1, 2, 4, 8, 16):
        idx = np.where(tr["T"] == T)[0]
        cnt = evaluate_lanes([combosC[i] for i in alive], Rows(tr, idx, dev), dev)
        log["D"][f"T{T}"] = {"rows": len(idx), "machines": len(alive), "exact": int((cnt == len(idx)).sum())}
        print(f"D: T={T} training rows ({len(idx)}): {(cnt == len(idx)).sum()}/{len(alive)} {what} exact "
              f"({time.time() - t0:.1f}s)", flush=True)
        alive = [i for i, k in zip(alive, cnt == len(idx)) if k]
    combosD = [combosC[i] for i in alive]
    log["D"].update({"survivors": len(combosD), "seconds": time.time() - t0, "by_family": by_family(combosD),
                     "free": free_switches(combosD)})
    if classes is not None:
        log["D"]["machines_in_classes"] = in_classes(alive)
    print(f"D: {len(combosD)} {what} fit every training row {log['D']['by_family']}"
          + ("" if classes is None else f" ({in_classes(alive):,} machines in their classes)")
          + f"; free switches: {log['D']['free']}", flush=True)
    if args.stop_after == "D":
        return dump()

    # stage E: the aliasing audit on both certification profiles
    t0 = time.time()
    certs = certify(combosD, dev)
    free = [f for f in GRAMMAR if len({x[f] for x in combosD}) > 1]
    for i, c in enumerate(combosD):
        audit.append({"machine": describe(c, free), "seen_maxT": int(certs["depth_t_"][0][i]),
                      "seen_correct": int(certs["depth_t_"][1][i]), "ood_maxT": int(certs["depth_ood_n_t_"][0][i]),
                      "ood_correct": int(certs["depth_ood_n_t_"][1][i])})
        if classes is not None:
            audit[-1].update({"class": alive[i], "class_size": len(classes[alive[i]])})
    seen, ood = certs["depth_t_"][0], certs["depth_ood_n_t_"][0]
    both = (seen == 64) & (ood == 64)
    certified = [c for c, b in zip(combosD, both) if b]
    log["E"] = {"seconds": time.time() - t0, "certify_both": len(certified), "machines": len(audit),
                "by_family": by_family(certified), "summary": audit_summary(combosD, seen, ood)}
    if classes is not None:
        log["E"]["certify_both_machines_in_classes"] = in_classes([k for k, b in zip(alive, both) if b])
    log["wall_seconds_to_E"] = time.time() - t_run
    print(f"E: {len(certified)}/{len(audit)} training-consistent {what} certify T=64 on both profiles "
          f"{log['E']['by_family']}"
          + ("" if classes is None else f"; their classes hold {log['E']['certify_both_machines_in_classes']:,} machines")
          + f" ({time.time() - t0:.1f}s)", flush=True)
    for fam, rec in log["E"]["summary"].items():
        print(f"   {fam}: {rec}", flush=True)
    if len(audit) <= 200:
        for r in audit:
            print("   ", r)
    if pre1 is not None and args.collapse_check:
        c1_check(pre1, args.collapse_check, prompts, t1, t2, dev, args.chunk, log)
    if classes is not None and args.collapse_check:
        collapse_check(members, classes, reps, alive, certs, args.collapse_check, dev, log)
    dump()


if __name__ == "__main__":
    main()
