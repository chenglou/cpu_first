"""Builds the fast combined file (integration/combined_fast/FAST.md): round 2's file (combined_sp2's
submission_stage3_combined2.py, its committed bytes, md5 asserted) with engineering speedups that change nothing the
climb computes, as named groups of exact edits, each asserted to match exactly once. Writes submission_stage3_fast.py;
NAME=VALUE sets a constant (research builds, written to research/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/combined_fast/make_fast.py [--diff] [--dev] [NAME=VALUE ...]

(--diff: the edits' effect as a unified diff against round 2's file; --dev: a working build that may be over the size
limit, written as *_dev.py; --md5s: each cut's md5 printed instead of asserted, for writing a new cut.)
"""

from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
import re
import subprocess
import symtable
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU = HERE.parents[1]
OUT = HERE / "submission_stage3_fast.py"
LIMIT = 256 * 1024                          # the evaluator's limit on a submission's size
BASE = ("914952f", "integration/combined_sp2/submission_stage3_combined2.py", "4e0e3f09b1ab55d1bfedbbed4fb0afc2")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def base() -> str:
    commit, path, want = BASE
    src = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=CPU, capture_output=True, text=True,
                         check=True).stdout
    assert md5(src) == want, (path, md5(src))
    return src


EDITS = []          # ("edit", group, old, new) or ("cut", group, start, end, md5, new): applied in order, each old
                    # text (or anchor) found exactly once in the text of the moment


def edit(group: str, old: str, new: str) -> None:
    EDITS.append(("edit", group, old, new))


# ================================================================================================ listing
# bfs_roots listed a level's goal states' programs one state at a time (programs(..., limit=1): a recursive walk with
# a device lookup and five host reads per level per state, ~0.4 ms each, 74,332 states in level 0's first probe).
# first_programs walks every goal state's first parent at once, level by level: the same programs, in the same order.
# (The work budget is read once: the listing does no machine work, so _spent gives the same answer at every 256th
# state as at the first.)
edit("listing", '''                res = _Res(parents, d, gidx)
                cand = []
                for gi, g in enumerate(gidx[:max_goals]):
                    if gi % 256 == 0 and _spent(stop_at):
                        break
                    f = programs(_OneGoal(res, g), classes, limit=1)
                    if f:
                        cand.append((len(f[0]), cost(expand(f[0], lib)), f[0], g))
                cand.sort(key=lambda c: (c[0], c[1]))
''', '''                res = _Res(parents, d, gidx)
                cand = [] if _spent(stop_at) else [(len(f), cost(expand(f, lib)), f, g) for f, g in
                                                   zip(first_programs(parents, d, gidx[:max_goals], classes), gidx)]
                cand.sort(key=lambda c: (c[0], c[1]))
''')
edit("listing", '''# ===================================================================================================== the practice problems' helpers
''', '''def first_programs(parents, level, goals, classes):
    """programs(_OneGoal(res, g), classes, limit=1)[0] for every goal state g at level, by one walk on the device
    (each state's first parent, level by level down to the roots): the same programs, in the goals' order."""
    if not goals:
        return []
    dev = parents[level][0].device
    lv = torch.full((len(goals),), level, dtype=torch.long, device=dev)
    ix = torch.tensor(goals, dtype=torch.long, device=dev)
    pick = torch.full((level, len(goals)), -1, dtype=torch.long, device=dev)
    for L in range(level, 0, -1):
        sel = torch.nonzero(lv == L).flatten()
        if sel.numel():
            child, pcl, psl, psi = parents[L]
            pos = torch.searchsorted(child, ix[sel])
            pick[L - 1, sel] = pcl[pos]
            lv[sel], ix[sel] = psl[pos], psi[pos]
    return [tuple(classes[c].rep for c in r if c >= 0) for r in pick.T.tolist()]


# ===================================================================================================== the practice problems' helpers
''')


# ================================================================================================ fused
# The lockstep ran one step at a time, ~60 small kernels a step (_ustep), replayed from a CUDA graph captured anew in
# every call past GRAPH_MIN steps. On a GPU with Triton (a dependency of the evaluator's torch), _lock_k runs every
# row's whole sequence in one kernel: per step, each row gathers its places by its instruction's move, runs the cell at
# its pointer, then the flags, the shift through k, the C blocks and the pointer, exactly as _ustep does (integers only:
# the same state). Elsewhere (a CPU, no Triton, LOCK_FUSED False) _ustep runs as before.
edit("fused", '''from torch import Tensor
''', '''from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:                  # (no Triton: the lockstep runs _ustep's steps)
    triton = None
''')
edit("fused", '''        # the cell table's row (Cells.alu: the subtraction's 2 B^2 rows, the addition's, then the multiply's B^4) as
        # off + a ca + x cx + c cc + h ch + k ck over (a, x, c, h, k)
        self.B = None
''', '''        # the cell table's row (Cells.alu: the subtraction's 2 B^2 rows, the addition's, then the multiply's B^4) as
        # off + a ca + x cx + c cc + h ch + k ck over (a, x, c, h, k)
        self.B, self.tb = None, None

    def table(self, co):
        """The fused lockstep's tables (int32): a row per instruction of flags (14), ri (2), dep, vo (2), co (6); rm."""
        if self.tb is None or self.tb[0] is not co:
            self.tb = (co, torch.cat([self.flags.int(), self.ri.int(), self.dep[:, None].int(), self.vo.int(),
                                      co.int()], 1).contiguous(), self.rm.int().contiguous())
        return self.tb[1:]
''')
edit("fused", '''    S, _, P, NP = b.regs.shape
    dev = b.regs.device
    o = _ops(P, dev)
    co = o.coefs(st.B)
    state = [b.regs.reshape(S, 5 * P, NP).contiguous(), b.c, b.h, b.k, b.p,
''', '''    S, _, P, NP = b.regs.shape
    dev = b.regs.device
    o = _ops(P, dev)
    co = o.coefs(st.B)
    if LOCK_FUSED and triton is not None and dev.type == "cuda" and not (o.sel or o.ldhc):
        return _lock_fused(st, b, o, co, cols, lr, depth, cond)
    state = [b.regs.reshape(S, 5 * P, NP).contiguous(), b.c, b.h, b.k, b.p,
''')
edit("fused", '''def apply_items(st, b, item_of_row, items, lib):
''', '''LOCK_FUSED = True                # on a GPU with Triton, a lockstep is one kernel (_lock_k; False: _ustep's steps)


def _lock_fused(st, b, o, co, cols, lr, depth, cond):
    """_lockstep by one kernel: every row runs its sequence (cols, lr) on copies of b's state, which then replace it."""
    S, _, P, NP = b.regs.shape
    if S == 0:
        return
    Z, f = 5 * P, torch.contiguous_format
    regs = b.regs.reshape(S, Z, NP).clone(memory_format=f)
    c, h, k, p = (x.clone(memory_format=f) for x in (b.c, b.h, b.k, b.p))
    ps = torch.zeros(S, max(depth, 1), dtype=torch.long, device=regs.device)
    ZB, NPB = triton.next_power_of_2(Z), triton.next_power_of_2(NP)
    BR = max(1, 4096 // (ZB * NPB))
    tb, rm = o.table(co)
    _lock_k[(triton.cdiv(S, BR),)](regs, c, h, k, p, ps, torch.zeros_like(c) if cond else c, cols.contiguous(),
                                   lr.contiguous(), rm, tb, st.cells.alu, S, cols.shape[0], cols.shape[1], P, Z,
                                   ps.shape[1], NP=NP, ZB=ZB, NPB=NPB, BR=BR, HAS_OFF=cond, HAS_PS=depth > 0,
                                   num_warps=4 if BR * ZB * NPB <= 4096 else 8)
    b.regs, b.c, b.h, b.k, b.p = regs.view(S, 5, P, NP), c, h, k, p


if triton is not None:
    @triton.jit(do_not_specialize=["S", "n", "items", "P", "Z", "D"])
    def _lock_k(R, C, H, K, PP, PS, OFF, COLS, LR, RM, TB, ALU, S, n, items, P, Z, D, NP: tl.constexpr,
                ZB: tl.constexpr, NPB: tl.constexpr, BR: tl.constexpr, HAS_OFF: tl.constexpr, HAS_PS: tl.constexpr):
        """n of _ustep's steps on BR rows, in place: R [S, Z, NP] the places, C, H, K, OFF [S, NP], PP [S], PS [S, D];
        a row's step t runs COLS[t, LR[row]] with its rows of RM (the move: each place's source, Z for zero) and TB
        (_Ops.table). The cell's inputs are read at their places after the move (rm), as _ustep reads them in g."""
        r = tl.program_id(0) * BR + tl.arange(0, BR)
        ok = r < S
        j, q = tl.arange(0, ZB), tl.arange(0, NPB)
        jj = j[None, :, None]
        m2 = ok[:, None] & (q < NP)[None, :]
        m3 = m2[:, None, :] & (jj < Z)
        r64 = r.to(tl.int64)
        row = R + r64 * Z * NP
        f2 = r64[:, None] * NP + q[None, :]
        at = row[:, None, None] + (jj * NP + q[None, None, :])
        lr = tl.load(LR + r, mask=ok, other=0)
        p = tl.load(PP + r, mask=ok, other=0).to(tl.int32)
        c = tl.load(C + f2, mask=m2, other=0).to(tl.int32)
        h = tl.load(H + f2, mask=m2, other=0).to(tl.int32)
        k = tl.load(K + f2, mask=m2, other=0).to(tl.int32)
        if HAS_OFF:
            off = tl.load(OFF + f2, mask=m2, other=0).to(tl.int32)
        for t in range(n):
            op = tl.load(COLS + t * items + lr, mask=ok, other=0).to(tl.int32)
            tb, rm = TB + op * 25, RM + op * Z
            src = tl.load(rm[:, None] + j[None, :], mask=(j < Z)[None, :], other=0)
            g = tl.load(row[:, None, None] + (src[:, :, None] * NP + q[None, None, :]),
                        mask=m3 & (src < Z)[:, :, None], other=0).to(tl.int32)
            if HAS_OFF:
                old = tl.load(at, mask=m3, other=0).to(tl.int32)
            ri1, vo0 = tl.load(tb + 15), tl.load(tb + 17)
            sa, sx, sh = tl.load(rm + p), tl.load(rm + tl.load(tb + 14) + p), tl.load(rm + ri1 + p)
            a = tl.load(row[:, None] + (sa * NP)[:, None] + q[None, :], mask=m2 & (sa < Z)[:, None], other=0)
            x = tl.load(row[:, None] + (sx * NP)[:, None] + q[None, :], mask=m2 & (sx < Z)[:, None], other=0)
            hs = tl.load(row[:, None] + (sh * NP)[:, None] + q[None, :], mask=m2 & (sh < Z)[:, None], other=0)
            out = tl.load(row[:, None] + (tl.load(tb + 18) * NP)[:, None] + q[None, :], mask=m2, other=0)
            i = (a.to(tl.int32) * tl.load(tb + 19)[:, None] + x.to(tl.int32) * tl.load(tb + 20)[:, None]
                 + c * tl.load(tb + 21)[:, None] + h * tl.load(tb + 22)[:, None] + k * tl.load(tb + 23)[:, None]
                 + tl.load(tb + 24)[:, None])
            od = tl.load(ALU + i * 2, mask=m2, other=0).to(tl.int32)
            of = tl.load(ALU + i * 2 + 1, mask=m2, other=0).to(tl.int32)
            fam = tl.load(tb + 7) != 0
            g = tl.where((jj == p[:, None, None]) & fam[:, None, None], od[:, None, :], g)
            c2 = tl.where(tl.load(tb + 1)[:, None] != 0, 0, tl.where(tl.load(tb + 8)[:, None] != 0, of, c))
            k2 = tl.where(tl.load(tb + 2)[:, None] != 0, 0, tl.where(tl.load(tb + 9)[:, None] != 0, of, k))
            hs = tl.where(((ri1 == 0) & fam)[:, None], od, hs.to(tl.int32))
            h2 = tl.where(tl.load(tb + 3)[:, None] != 0, hs, h)
            sk = tl.load(tb + 11) != 0
            g = tl.where((jj == vo0[:, None, None]) & sk[:, None, None], k[:, None, :], g)
            k2 = tl.where(sk[:, None], out.to(tl.int32), k2)
            if HAS_OFF:
                on = off == 0
                g = tl.where(on[:, None, :], g, old)
                c2, h2, k2 = tl.where(on, c2, c), tl.where(on, h2, h), tl.where(on, k2, k)
                off = tl.where(tl.load(tb + 12)[:, None] != 0, tl.where(off > 0, off + 1, (c == 0).to(tl.int32)),
                               tl.where((tl.load(tb + 13)[:, None] != 0) & (off > 0), off - 1, off))
            if HAS_PS:
                ad = PS + r64 * D + tl.minimum(tl.load(tb + 16), D - 1)
                saved = tl.load(ad, mask=ok, other=0).to(tl.int32)
                tl.store(ad, p.to(tl.int64), mask=ok & (tl.load(tb + 5) != 0))
                p = tl.where(tl.load(tb + 6) != 0, saved, p)
            p = tl.where(tl.load(tb + 4) != 0, (p + 1) % P, p)
            c, h, k = c2, h2, k2
            tl.debug_barrier()
            tl.store(at, g.to(tl.uint8), mask=m3)
            tl.debug_barrier()
        tl.store(C + f2, c.to(tl.uint8), mask=m2)
        tl.store(H + f2, h.to(tl.uint8), mask=m2)
        tl.store(K + f2, k.to(tl.uint8), mask=m2)
        tl.store(PP + r, p.to(tl.int64), mask=ok)


def apply_items(st, b, item_of_row, items, lib):
''')


# ================================================================================================ grouped
# Buckets the cost model gives to _apply_grouped ran item by item, a few small kernels per primitive (step). With the
# fused lockstep those rows run in it too (the same result: apply_items gives run_items' either way), and the work
# _apply_grouped would count is counted as it counts it: its steps' increments, in its order, the same floats
# (_grouped_work). So WORK, and every budget read from it, stays as it was.
edit("grouped", '''    if b.S == 0 or not lock:
        return _apply_grouped(st, b, item_of_row, items, lib)
    WORK[0] += sum(lc[j] for j in lock)              # the locksteps' work (the other buckets' is counted by step)
    out = b.copy()
''', '''    fused = b.S and _fused(b.regs.device)
    if b.S == 0 or not (lock or fused):
        return _apply_grouped(st, b, item_of_row, items, lib)
    cnt = torch.bincount(item_of_row, minlength=len(items)).tolist() if fused else None
    if lock:
        WORK[0] += sum(lc[j] for j in lock)          # the locksteps' work (the other buckets' is counted by step)
    else:
        _grouped_work(st, items, cnt, lib, ent)      # (fused: _apply_grouped's work; its rows run in the lockstep)
    out = b.copy()
''')
edit("grouped", '''        if j in lock:
            cols = _columns([seqs[i][0] for i in per[j]], mx[j], dev)
''', '''        if j in lock or fused:
            if lock and j not in lock:
                _grouped_work(st, [items[i] for i in per[j]], [cnt[i] for i in per[j]], lib, ent)
            cols = _columns([seqs[i][0] for i in per[j]], mx[j], dev)
''')
edit("grouped", '''    if LOCK_FUSED and triton is not None and dev.type == "cuda" and not (o.sel or o.ldhc):
        return _lock_fused(st, b, o, co, cols, lr, depth, cond)
''', '''    if _fused(dev):
        return _lock_fused(st, b, o, co, cols, lr, depth, cond)
''')
edit("grouped", '''LOCK_FUSED = True                # on a GPU with Triton, a lockstep is one kernel (_lock_k; False: _ustep's steps)
''', '''LOCK_FUSED = True                # on a GPU with Triton, a lockstep is one kernel (_lock_k; False: _ustep's steps)


def _fused(dev):
    return LOCK_FUSED and triton is not None and dev.type == "cuda" and COND not in ("sel", "ldhc")


def _steps_of(st, it, lib, ent):
    """How many primitives run_items runs for item it (its trace's instructions but the routines' and C blocks'
    marks), memoized with its trace."""
    got = ent[1].get((it, "n"))
    if got is None:
        got = ent[1][(it, "n")] = sum(1 for o in _trace_of(st, it, lib, 0, ent)[0] if o < NOP)
    return got


def _grouped_work(st, items, cnt, lib, ent):
    """WORK as _apply_grouped counts it on items with cnt[i] rows each, without running it: each group's steps
    (step: PRIM_STEP + its rows PRIM_ROW each), groups in _apply_grouped's order (the items that are not loops one by
    one; then each loop iteration's body elements, by position, rows of one element together)."""
    def add(n, rows):
        v, w = PRIM_STEP + rows * PRIM_ROW, WORK[0]
        for _ in range(n):
            w += v
        WORK[0] = w
    loop = [is_loop(it) and it[1] not in lib and it[1] != "C" for it in items]
    for it, lp, n in zip(items, loop, cnt):
        if n and not lp:
            add(_steps_of(st, it, lib, ent), n)
    elems = {}
    for it, lp in zip(items, loop):
        if lp:
            for e in it[2]:
                elems.setdefault(e, len(elems))
    eid = list(elems)
    L = max((len(it[2]) for it, lp in zip(items, loop) if lp), default=0)
    for itn in range(max((st.count(it[1]) for it, lp, n in zip(items, loop, cnt) if lp and n), default=0)):
        for j in range(L):
            size = {}
            for it, lp, n in zip(items, loop, cnt):
                if lp and n and st.count(it[1]) > itn and j < len(it[2]):
                    e = elems[it[2][j]]
                    size[e] = size.get(e, 0) + n
            for e in sorted(size):
                add(_steps_of(st, eid[e], lib, ent), size[e])
''')


# ================================================================================================ room
# The new code is paid for by code this file never runs (its settings are constants: rule "lpca", deep_by
# "bindings", calls_first True, no small target, bfs's default listing) and by the graph path, which the fused lockstep
# replaces on a GPU (elsewhere _ustep runs step by step, as on a CPU). A span is cut between two anchors that occur once
# (the start kept or not as given), the cut text pinned by its md5.
def cut(group: str, start: str, end: str, digest: str, new: str = "") -> None:
    """The text from start (included) to end (excluded), its md5 digest, replaced by new."""
    EDITS.append(("cut", group, start, end, digest, new))


cut("room", "GRAPH_STEPS = 32 ", "def _lockstep(st, b, cols, lr, depth, cond=False):\n",
    "9a7d9ead7ab763205b26cb2044f124a6")
cut("room", "def _graph_steps(o, alu, co, state, cols, lr, ar, ad):\n", "LOCK_FUSED = True ",
    "880ae1cc023ec06125738ed46bc2bc0e")
edit("room", '''    n = cols.shape[0]
    t0 = 0
    if dev.type == "cuda" and GRAPH_STEPS and n >= GRAPH_MIN:
        state, t0 = _graph_steps(o, st.cells.alu, co, state, cols, lr, ar, ad)
    for t in range(t0, n):
''', '''    for t in range(cols.shape[0]):
''')
cut("room", '''            if listing == "goals":
''', '''            progs = programs(res, classes, limit=limit)
''', "62d8cb9906483bce42da26a9f01fb120")
edit("room", '''def bfs(prob, classes, lib, max_cost=8, max_states=400_000, accept=None, limit=200, stop_at=None, chunk=1 << 21,
        listing="spellings", max_goals=50_000, spell=8, batch=256, fuse=1):''', '''def bfs(prob, classes, lib, max_cost=8, max_states=400_000, accept=None, limit=200, stop_at=None, chunk=1 << 21,
        fuse=1):''')
edit("room", '''streaming goal states only), or past stop_at. The goal programs tested: the first `limit` spellings
    (listing "spellings"), or one spelling per goal state (listing "goals"), tested in batches, fewest top-level items
    first (then fewest primitives), until a batch has one that passes; the accepted goal states then give up to `spell`
    spellings each that pass. A level's work is''', '''streaming goal states only), or past stop_at. The goal programs tested: the first `limit` spellings. A level's
    work is''')
edit("room", '''      choose  trivial (cost <= 1), learnable (cost >= 2, not an alias of a token, not a duplicate of an earlier
              learnable by what it does on random machines), unsolved; then the rule: "lp" (the K costliest learnable,
              ties in pool order), "ct2a2" (the cheaper pairs signal: the target probed with every learnable's routine,
              and with every pair whose member is among the 2 hardest; the best pairs' members), "all";
''', '''      choose  trivial (cost <= 1), learnable (cost >= 2, not an alias of a token, not a duplicate of an earlier
              learnable by what it does on random machines), unsolved; then the rule (pick: lpca, by concepts);
''')
edit("room", '''        self.round_budget, self.round_cost, self.look_budget = round_budget, round_cost, look_budget
''', '''        self.round_budget, self.round_cost = round_budget, round_cost
''')
edit("room", '''        self.looks, self._base_look, self._scores = [], None, {}
''', "")
edit("room", '''        self.target, self._near, self.near_log = target, [], []      # a small target (the tiny core's oracle arm)
''', "")
edit("room", '''                 patterns=None, laws=None, round_retry=0, target=None, macro_after=12, macro_cost=2,
''', '''                 patterns=None, laws=None, round_retry=0, macro_after=12, macro_cost=2,
''')
edit("room", '''        if self.target is not None:                     # the target's restrictions, projections, near misses
            for r, goals in self.target.pose(self, rows, self._near):
                goals = [g for g in goals if f"{g.name}/{r.key}" not in self.practiced]
                if goals:
                    r.goal_list = goals
                    roots.append(r)
                    pool.extend(f"{g.name}/{r.key}" for g in goals)
''', "")
edit("room", '''                self._accs[(g.name, root.key)] = (
                    self.target.acceptor(self, rows, root, g.name) if hasattr(root, "kind") else
                    Acceptor(g.name, root, sets[g.name], rows, self.ops, self.lib, self.device, box=box))
''', '''                self._accs[(g.name, root.key)] = Acceptor(g.name, root, sets[g.name], rows, self.ops, self.lib,
                                                          self.device, box=box)
''')
cut("room", '''        if self.deep_by == "concept":
''', '''    # ------------------------------------------------------------------ choose
''', "71b12cfd28548caa65827ddf4df29e03", '''        raise ValueError(self.deep_by)

''')
edit("room", '''        """The deep probes' roots: the deep_k roots whose closest open law is closest (ties in pool order), among those
        with an open law that the probe brought closer than one instruction does. With deep_by="concept", for each
        open concept (a law on a pattern) its closest binding's root, the deep_k closest such concepts' roots. With
        deep_by="bindings", the closest concepts whole:''', '''        """The deep probes' roots, among those with an open law that the probe brought closer than one instruction
        does (deep_by="bindings"): the closest concepts whole:''')
cut("room", '''    def look_jobs(self, status):
''', '''    def concepts_done(self):
''', "29798b70b3934849606a59e992b81c8c")
cut("room", '''    def pick(self, status, rows=None):
''', '''    # ------------------------------------------------------------------ sleep
''', "d43ffc298e7b2407df8654480ef29200", '''    def pick(self, status, rows=None):
        """lpca (lpc: without the ties): concepts, a law on a pattern, hardest by its cheapest binding, practised once,
        in its cleanest binding (lpca: in every binding tied for cleanest). A practised concept's other bindings are not
        posed; when no new concept is learnable, the next level poses them, and picks the hardest."""
        if self.rule not in ("lpc", "lpca"):
            raise ValueError(self.rule)
        got = self.concepts(status, self.K, ties=self.rule == "lpca")
        if got or not self._variants:
            self._want_variants = not got
            return got
        return self.hardest(status, self.K)

''')
edit("room", '''            if self.target is not None:
                _, rec, self._near = self.target.search(self, rows, self.look_budget, stop_at, near=12)
                self.near_log.append(dict(level=self.level, fits=rec["fits"], near=rec["near"]))
                self.records.append(rec)
''', "")
cut("room", '''            self.phase = "choose"
            if self.rule == "ct2a2" and''', '''        if self.phase == "choose":
''', "476cf7b244b2f550a36746c77d6543e4", '''            self.phase = "choose"
''')
edit("room", '''            status = self._status
            if self.rule == "ct2a2" and not self.looks:
                picks = self.hardest(status, self.K)
            else:
                picks = self.pick(status, rows)
''', '''            status = self._status
            picks = self.pick(status, rows)
''')
edit("room", '''                                     scores=self._scores if self.rule in ("ct2a2", "lpu") else {}))
            self.picks = picks
            self.phase = "sleep"
            self.looks = []
''', '''                                     scores={}))
            self.picks = picks
            self.phase = "sleep"
''')
cut("room", '''            if self.target is not None:
                progs, rec, _ = self.target.search(self, rows, self.round_budget, stop_at)
''', '''            if answers is None or not self.routines():
''', "ba1672c83222fed3c5133a1f0168a9db")
edit("room", '''            if found is None and calls and not self.calls_first:
                mres = bfs(prob, self.macro_classes(lib), lib, max_cost=self.macro_cost,
                           max_states=budget or self.round_budget, accept=ok, stop_at=stop_at,
                           chunk=max(1, self.chunk // 4), fuse=self.fuse)
                stop = f"{stop}; by calls: levels {mres.level_sizes} {mres.stop}"
                if mres.found_cost is not None:
                    found, accepted = f"{mres.found_cost} calls", [unwrap(p) for p in mres.accepted]
''', '''            if found is None and calls and not self.calls_first:
                raise ValueError("calls_first")            # (the search by calls after the flat one: not here)
''')


# ================================================================================================ the build
def undefined_globals(src: str) -> set:
    """Names a function reads as globals that neither the module's top level nor builtins define (symtable)."""
    top = symtable.symtable(src, "fast", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    out = set()

    def walk(t):
        for s in t.get_symbols():
            if s.is_global() and s.get_name() not in defined and not hasattr(builtins, s.get_name()):
                out.add(s.get_name())
        for c in t.get_children():
            walk(c)
    walk(top)
    return out


def build(sets: dict | None = None, dev: bool = False) -> tuple[str, list]:
    """The fast file's text and the log of the edit groups applied; dev: over the size limit is allowed (a working
    build, never the submission)."""
    src = base()
    log = []
    for e in EDITS:
        if e[0] == "edit":
            _, group, old, new = e
            n = src.count(old)
            assert n == 1, (group, n, old[:120])
            src = src.replace(old, new)
        else:
            _, group, start, end, digest, new = e
            assert src.count(start) == 1 and src.count(end) == 1, (group, start[:80], end[:80])
            i = src.index(start)
            j = src.index(end, i)
            got = md5(src[i:j])
            if SHOW_MD5:
                print(f"cut {got} {start[:60]!r}")
            else:
                assert got == digest, (group, start[:80], got)
            src = src[:i] + new + src[j:]
        log.append(group)
    for name, value in (sets or {}).items():                  # research builds only
        src, n = re.subn(rf"^{name} = .*$", f"{name} = {value}", src, count=1, flags=re.M)
        assert n == 1, name
    ast.parse(src)
    compile(src, OUT.name, "exec")
    undef = undefined_globals(src) - {"big", "name", "self"}  # (symtable's false alarm on book_gen's Draw.top, as
    assert not undef, undef                                   #  make_combined.py has it)
    sys.path.insert(0, str(CPU.parent / "one-layer-deeper"))
    from submission_validation import validate_submission_source
    validate_submission_source("submission.py", src, 2 * LIMIT if dev else LIMIT)
    return src, log


SHOW_MD5 = "--md5s" in sys.argv          # (prints each cut's md5 instead of asserting it: writing a new cut)


def main(argv):
    sets = dict(a.split("=", 1) for a in argv if "=" in a and not a.startswith("--"))
    src, log = build(sets, dev="--dev" in argv)
    if "--diff" in argv:
        sys.stdout.writelines(difflib.unified_diff(base().splitlines(True), src.splitlines(True), "combined2", "fast",
                                                   n=1))
        return
    for g in dict.fromkeys(log):
        print(f"{log.count(g):4d}  {g}")
    out = OUT if not sets else HERE / "research" / (OUT.stem + "_" + "_".join(
        f"{k}-{v}" for k, v in sets.items()).replace("/", "") + ".py")
    if "--dev" in argv:
        out = out.with_name(out.stem + "_dev.py")
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    print(f"wrote {out}: {len(src.splitlines())} lines, {len(src.encode()):,} bytes "
          f"({LIMIT - len(src.encode()):,} under the limit), md5 {md5(src)}")


if __name__ == "__main__":
    main(sys.argv[1:])
