"""Library learning inside one training run (INTEGRATION.md, stage 2): the practice problems come from the training
rows' own digits and the law library, the search is tdsl's (torch), the merges are library/merge.py's BPE, solved
tasks become routines, and the cells are whatever tables the run has learned so far.

Where the practice tasks come from. Each is one law of general_lessons' LAWS applied to numbers a row gives (N's top
W digits N' and x's digits at the same places x', so x' <= N' by the mined fact x < N; another row's x' as Q), with
its target either a fact (no arithmetic: a copy of x', a 0, a borrow bit) or the machine's own result of a program the
run found earlier (a "helper"):

    task    law (LAWS)                     what the probe states                                        helpers
    lt      sub.order + mined x < N        c = [x' - N' borrows] = [x', N' differ]; N' - x', x' - x' never  -
    modx    red.identity, red.zero         x' mod N' = x' (0 if equal); N' mod N' = 0                      -
    add     sub.inverse, add.identity      (x' + N') mod N' = x' and x' + N' does not borrow against N'    lt, modx
                                           (inverse checks); x' + 0 = x'
    modB    red.step (below's ladder)      (x' + q N') mod N' = x', q = 0..9 (built by q additions)        add
    mrow    mul.zero, mul.ladder           x' d = d additions of x', d = 0..9                              add
    mac     mul.ladder with an accumulator x' d + N' = add(mrow(x', d), N')                                 mrow, add
    mul     mul.distribute (as additions)  x' Q = Q additions of x'                                        add
    sq      the same with Q = x'           x' x' with the found mul                                        mul
    step    red.step (below's step law)    (r B + d) mod N' with the found modB, r = x' < N', d = 0..9     modB
    reduce  red.invariant                  (x' + Q N') mod N' = x' (Q additions of N')                     add
    sqmod   the round (the target)         (x' x') mod N' with mul and reduce; then the answers of the     mul, reduce
                                           full-size T = 1 rows
The designed parts are said plainly in INTEGRATION.md: which laws become tasks, the helper order (NEEDS), "the
square" and "the round" as tasks, and the register variants below.

Register variants (designed, task-independent): every task is also posed with the machine's two free input registers
X and Y swapped (lt on Y, N; modx of Y; ...), so that each concept is practiced by two tasks and a merge used by both
can pay for itself (LIBRARY.md §5.1: with one task per concept every merge rule stalls). The digits d and q run over
the probes (q = probe index mod B): no number is made up. A task's probes use 8 probes per setting at (10, 1) and
(10, 2) (16 for lt), acceptance 60 fresh rows at widths 1-3, as library/inrun.py.
"""

from __future__ import annotations

import json
import os
import time

import torch

from library import dsl as pdsl
from library import merge
from integration.lib import tdsl, tfast

H1_DIR = os.environ.get("H1_DATA", "/Users/chenglou/github/one-layer/one-layer-deeper/data/generated/"
                                   "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2")
BASE_NEEDS = {"lt": (), "modx": (), "add": ("lt", "modx"), "modB": ("add",), "mrow": ("add",), "mac": ("mrow", "add"),
              "mul": ("add",), "sq": ("mul",), "step": ("modB",), "reduce": ("add",), "sqmod": ("mul", "reduce")}
SWAP = {"X": "Y", "Y": "X", "N": "N"}


def task_names(variants=True):
    out = []
    for t in BASE_NEEDS:
        out.append(t)
        if variants and t != "sqmod":
            out.append(t + "~")          # the same law with X and Y swapped
    return out


def needs(task):
    base = task.rstrip("~")
    return tuple(n + ("~" if task.endswith("~") and n + "~" in task_names() else "") for n in BASE_NEEDS[base])


# ---------------------------------------------------------------------------------------------------------------
# rows as digits (no arithmetic on values)

def load_rows(n=4000, seed=0, split="train", T=None, path=H1_DIR):
    rows = [json.loads(l) for l in open(os.path.join(path, split + ".jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(rows), generator=g)[:n].tolist()
    return [rows[i] for i in idx]


def digits_of_rows(rows, L=8):
    """The rows' N, x and answer y as digit tensors [n, L] (LSB first): read from the decimal strings (the way the
    model's reader reads tokens), never computed."""
    def dig(s):
        s = str(s)
        return [int(ch) for ch in reversed(s)][:L] + [0] * (L - min(L, len(s)))
    return (torch.tensor([dig(r["modulus"]) for r in rows]), torch.tensor([dig(r["x"]) for r in rows]),
            torch.tensor([dig(r["result"]) for r in rows]))


def top_parts(Nd, Xd, w):
    """N's top w digits N' and x's digits at the same places x' [n, w] (digit strings only)."""
    L = Nd.shape[1]
    top = torch.where(Nd != 0, torch.arange(L)[None], -1).max(1).values      # N's top place
    lo = (top - w + 1).clamp(min=0)
    idx = lo[:, None] + torch.arange(w)[None]
    valid = idx <= top[:, None]
    idx = idx.clamp(max=L - 1)
    return Nd.gather(1, idx) * valid, Xd.gather(1, idx) * valid


def pad(d, P):
    return torch.cat([d, torch.zeros(d.shape[0], P - d.shape[1], dtype=d.dtype)], 1) if d.shape[1] < P else d[:, :P]


def width(task, st):
    base = task.rstrip("~")
    if base in ("lt", "modx"):
        return st.P
    if base in ("add", "modB", "mrow", "step"):
        return st.P - 1
    return st.W


# ---------------------------------------------------------------------------------------------------------------
# helpers: found programs run on the machine (the learned cells)

def run_prog(st, prog, lib, ins, gen):
    """Run a found program on inputs {reg: digits [n, P]}; returns (A digits [n, P], c bits [n])."""
    n = next(iter(ins.values())).shape[0]
    b = tdsl.fresh(st, ins, gen, n)
    tdsl.run_items(st, b, prog, lib)
    return b.regs[0, tdsl.A].T.cpu(), b.c[0].cpu()


def digit_value(d):
    """A digit string as a count (for repeated additions: how many times; read off its digits, Horner)."""
    v = torch.zeros(d.shape[0], dtype=torch.long)
    for j in reversed(range(d.shape[1])):
        v = v * 10 + d[:, j]
    return v


def build(task, st, n, gen, rows, helpers, lib):
    """Probes of one task at one setting: (inputs {reg: digits [n, P]}, targets {"A": digits [n, P]} / {"c": [n]},
    inverse check or None), from the rows' own digits (rows: (Nd, Xd) digit tensors)."""
    base, swap = task.rstrip("~"), task.endswith("~")
    reg = (lambda r: SWAP[r]) if swap else (lambda r: r)
    P, B = st.P, st.B
    w = width(task, st)
    Nd, Xd = rows
    pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen)
    Nt, xt = top_parts(Nd[pick], Xd[pick], w)
    Nt, xt, other = pad(Nt[:n], P), pad(xt[:n], P), pad(xt[n:2 * n], P)
    eq = (Nt == xt).all(1)
    zero = torch.zeros_like(Nt)
    run = lambda name, ins: run_prog(st, helpers[name], lib, {reg(k): v for k, v in ins.items()}, gen)
    qd = torch.arange(n) % B                                                     # q, d = 0..B-1 over the probes
    qdig = pad(qd[:, None], P)
    if base == "lt":
        kind = torch.arange(n) % 4                                               # 0, 1: (x', N'); 2: (N', x'); 3: (x', x')
        Xv = torch.where((kind == 2)[:, None], Nt, xt)
        Nv = torch.where((kind < 2)[:, None], Nt, xt)
        c = torch.where(kind < 2, (~eq).long(), 0)
        return {reg("X"): Xv, "N": Nv}, {"c": c}, None
    if base == "modx":
        kind = torch.arange(n) % 3
        Xv = torch.where((kind < 2)[:, None], xt, Nt)
        Av = torch.where(((kind < 2) & ~eq)[:, None], xt, zero)
        return {reg("X"): Xv, "N": Nt}, {"A": Av}, None
    if base == "add":
        keep = torch.nonzero(~(Nt == xt).all(1)).flatten()                      # modx is known below 2N' only
        Nk, xk = Nt[keep], xt[keep]
        m = keep.numel()
        kind = torch.arange(m) % 4                                               # 0..2: x' + N'; 3: x' + 0 = x'
        Nv = torch.where((kind < 3)[:, None], Nk, torch.zeros_like(Nk))

        def inverse(A_out):                                                      # A_out [S, m, P] digits
            S_ = A_out.shape[0]
            ok = torch.ones(S_, dtype=torch.bool)
            A_out = A_out.cpu().long()
            sel = torch.nonzero(kind < 3).flatten()
            flat = A_out[:, sel].reshape(-1, P)
            Ns = Nk[sel].repeat(S_, 1)
            back = run("modx", {"X": flat, "N": Ns})[0].view(S_, -1, P)
            ok &= (back == xk[sel][None]).all(-1).all(-1)
            cb = run("lt", {"X": flat, "N": Ns})[1].view(S_, -1)
            ok &= (cb == 0).all(-1)
            sel0 = torch.nonzero(kind == 3).flatten()
            ok &= (A_out[:, sel0] == xk[sel0][None]).all(-1).all(-1)
            return ok
        return {reg("X"): xk, "N": Nv}, {}, inverse
    if base == "modB":
        D = xt.clone()
        for k in range(B - 1):
            more = qd > k
            if bool(more.any()):
                D[more] = run("add", {"X": D[more], "N": Nt[more]})[0]
        return {reg("X"): D, "N": Nt}, {"A": torch.where(eq[:, None], zero, xt)}, None
    if base == "mrow":
        Tv = torch.zeros_like(xt)
        for k in range(B - 1):
            more = qd > k
            if bool(more.any()):
                Tv[more] = run("add", {"X": Tv[more], "N": xt[more]})[0]
        return {reg("X"): xt, reg("Y"): qdig}, {"A": Tv}, None
    if base == "mac":
        R = run("mrow", {"X": xt, "Y": qdig})[0]
        Tv = run("add", {"X": R, "N": Nt})[0]
        return {reg("X"): xt, reg("Y"): qdig, "N": Nt}, {"A": Tv}, None
    if base == "mul":
        Tv = torch.zeros_like(xt)
        cnt = digit_value(other)
        for k in range(int(cnt.max())):
            more = cnt > k
            if bool(more.any()):
                Tv[more] = run("add", {"X": Tv[more], "N": xt[more]})[0]
        return {reg("X"): xt, reg("Y"): other}, {"A": Tv}, None
    if base == "sq":
        Tv = run("mul", {"X": xt, "Y": xt})[0]
        return {reg("X"): xt}, {"A": Tv}, None
    if base == "step":
        keep = torch.nonzero(~eq).flatten()
        Nk, xk = Nt[keep], xt[keep]
        m = keep.numel()
        d = (torch.arange(m) % B)
        Xv = torch.zeros_like(Nk)
        Xv[:, P - 1] = d                                                         # the digit d at X's top place
        v = torch.cat([d[:, None], xk[:, :P - 1]], 1)                            # r B + d: r's digits moved up, d below
        Tv = run("modB", {"X": v, "N": Nk})[0]
        return {reg("X"): Xv, reg("Y"): xk, "N": Nk}, {"A": Tv}, None
    if base == "reduce":
        cnt = digit_value(other)
        D = xt.clone()
        for k in range(int(cnt.max())):
            more = cnt > k
            if bool(more.any()):
                D[more] = run("add", {"X": D[more], "N": Nt[more]})[0]
        return {reg("X"): D, "N": Nt}, {"A": torch.where(eq[:, None], zero, xt)}, None
    if base == "sqmod":
        S2 = run("mul", {"X": xt, "Y": xt})[0]
        Tv = run("reduce", {"X": S2, "N": Nt})[0]
        return {"X": xt, "N": Nt}, {"A": Tv}, None
    raise ValueError(task)


class LawProblem(tdsl.Problem):
    def __init__(self, task, settings, NP, seed, rows, helpers, lib, device):
        gen = torch.Generator().manual_seed(seed)
        roots, targets, inverse = [], [], []
        for st in settings:
            ins, tgt, inv = build(task, st, 2 * NP if task.rstrip("~") == "lt" else NP, gen, rows, helpers, lib)
            n = next(iter(ins.values())).shape[0]
            roots.append(tdsl.fresh(st, ins, gen, n, device))
            t = {}
            if "A" in tgt:
                t["A"] = tgt["A"].T.to(device=device, dtype=torch.uint8)
            if "c" in tgt:
                t["c"] = tgt["c"].to(device=device, dtype=torch.uint8)
            targets.append(t)
            inverse.append(inv)
        super().__init__(task, settings, roots, targets, inverse)


def accept(task, prog, lib, helpers, rows, settings, device, n=60, seed=11, labels=None):
    """Does prog pass its task's laws on fresh rows at every setting (and, for sqmod, the full-size T = 1 rows'
    answers: labels = (Nd, Xd, Yd) digit tensors)?"""
    out = {}
    for st in settings:
        prob = LawProblem(task, [st], n, seed, rows, helpers, lib, device)
        b = prob.roots[0].copy()
        tdsl.run_items(st, b, prog, lib)
        out[repr(st)] = bool(prob.goal([b])[0])
    if task == "sqmod" and labels is not None:
        st = tdsl.Setting(10, 8, settings[0].cells)
        Nl, Xl, Yl = labels
        A_, _ = run_prog(st, prog, lib, {"X": pad(Xl, 16), "N": pad(Nl, 16)}, torch.Generator().manual_seed(seed))
        out["labels(10,8)"] = float((A_[:, :8] == Yl).all(1).float().mean()) == 1.0
    return out


# ---------------------------------------------------------------------------------------------------------------
# the climb

def simplest(progs, lib):
    return min(progs, key=lambda p: pdsl.cost(pdsl.expand(p, lib)))


def promote(corpus, lib, tasks):
    lib, log = dict(lib), []
    for t in tasks:
        prog = simplest(corpus[t], lib)
        name = f"T.{t}"
        if pdsl.cost(prog) < 2 or name in lib:
            continue
        lib[name] = prog
        fn = merge._apply_pattern(prog, name, False)
        corpus = {u: [merge.rewrite(p, fn) for p in ps] for u, ps in corpus.items()}
        log.append({"name": name, "def": pdsl.show(prog), "routine": True})
    return corpus, lib, log


class Climb:
    """The wake-sleep loop as a state machine a training loop can drive (one search job per `step`): every task
    whose helpers are found is searched over the primitives plus the library; at the end of a level the corpus is
    merged (BPE), each task keeps its simplest program (Occam), newly solved tasks become routines, and the next level
    starts. `cells`: the learned tables (tdsl.Cells) the searches and the probes run with."""

    def __init__(self, cells, rows, labels=None, device="cpu", variants=True, budget=None, max_body=2, max_cost=7,
                 min_count=2.0, max_merges=12, tiebreak="largest", fallback=True, levels=12, log=print,
                 settings=((10, 1), (10, 2)), verify=((10, 1), (10, 2), (10, 3)), NP=8, deadline=None,
                 fallback_budget=None, chunk=1 << 21):
        self.cells, self.rows, self.labels, self.device = cells, rows, labels, device
        self.tasks = task_names(variants)
        self.budget = budget or 3_000_000
        self.max_body, self.max_cost, self.min_count, self.max_merges = max_body, max_cost, min_count, max_merges
        self.tiebreak, self.fallback, self.levels, self.log, self.NP = tiebreak, fallback, levels, log, NP
        self.settings = [tdsl.Setting(B, W, cells) for B, W in settings]
        self.verify = [tdsl.Setting(B, W, cells) for B, W in verify]
        self.corpus, self.lib, self.solved_at, self.level = {}, {}, {}, 0
        self.queue, self.records, self.history = None, [], []
        self.done = False
        self.deadline = deadline
        self.fallback_budget = fallback_budget or self.budget
        self.chunk = chunk
        self._classes = None

    def helpers_for(self, task):
        """The helpers a task's probes run, by their base names (a swapped task's helpers are the swapped ones, which
        read their inputs from the swapped registers, as the task's own probes do)."""
        return {n.rstrip("~"): simplest(self.corpus[n], self.lib) for n in needs(task)}

    def todo(self):
        return [t for t in self.tasks if t not in self.corpus and all(n in self.corpus for n in needs(t))]

    def classes(self, prims_only=False):
        key = ("prims" if prims_only else "lib", tuple(sorted(self.lib)))
        if self._classes is None or self._classes[0] != key:
            tokens = list(pdsl.PRIMS) + ([] if prims_only else [k for k in self.lib])
            self._classes = (key, tfast.enumerate_items(tokens, {} if prims_only else self.lib, self.settings,
                                                        max_body=self.max_body, log=self.log))
        return self._classes[1]

    def search(self, task):
        t0 = time.time()
        helpers = self.helpers_for(task)
        prob = LawProblem(task, self.settings, self.NP, 0, self.rows, helpers, self.lib, self.device)
        cache = {}

        def ok(p):
            k = pdsl.show(p)
            if k not in cache:
                r = accept(task, p, self.lib, helpers, self.rows, self.verify, self.device, labels=self.labels)
                cache[k] = all(r.values())
            return cache[k]
        res = tfast.bfs(prob, self.classes(), self.lib, max_cost=self.max_cost, max_states=self.budget,
                        accept=lambda ps: [ok(p) for p in ps], deadline=self.deadline, chunk=self.chunk)
        how = "library"
        if res.found_cost is None and self.fallback and (self.lib or self.fallback_budget > self.budget):
            cls = self.classes(prims_only=True)
            fb = tfast.bfs(prob, cls, {}, max_cost=self.max_cost, max_states=self.fallback_budget,
                           accept=lambda ps: [ok(p) for p in ps], deadline=self.deadline, chunk=self.chunk)
            if fb.found_cost is not None:
                res, how = fb, "primitives"
        rec = dict(task=task, level=self.level, found_cost=res.found_cost, level_sizes=res.level_sizes,
                   seconds=round(time.time() - t0, 1), stop=res.stop, how=how,
                   programs=[pdsl.show(p) for p in (res.accepted or [])][:20],
                   aliases=res.aliases[:3])
        return res.accepted or [], rec

    def step(self):
        """One unit of work: the next search of this level, or the level's sleep phase. Returns a log line."""
        if self.done:
            return None
        if self.queue is None:
            self.queue = self.todo()
            self.solved_now = []
            if not self.queue or self.level >= self.levels:
                self.done = True
                return f"climb done at level {self.level}: solved {sorted(self.corpus)}"
        if self.queue:
            task = self.queue.pop(0)
            progs, rec = self.search(task)
            self.records.append(rec)
            if progs:
                self.corpus[task] = progs
                self.solved_at[task] = self.level
                self.solved_now.append(task)
            return (f"L{self.level} {task:8s} cost {rec['found_cost']} ({rec['how']}) levels {rec['level_sizes']} "
                    f"{rec['seconds']}s {rec['stop']} {rec['programs'][:2]}")
        # the level's sleep phase: merge, Occam, routines
        mlog = []
        self.corpus, self.lib, new = merge.bpe(self.corpus, self.lib, max_merges=self.max_merges,
                                               min_count=self.min_count, prefix=f"B{self.level}.", log=mlog,
                                               tiebreak=self.tiebreak)
        self.corpus = {t: [simplest(ps, self.lib)] for t, ps in self.corpus.items()}
        self.corpus, self.lib, plog = promote(self.corpus, self.lib, self.solved_now)
        mlog += plog
        for m in mlog:
            m["expands_to"] = pdsl.show(pdsl.expand(self.lib[m["name"]], self.lib)) if m["name"] in self.lib else ""
        self.history.append(dict(level=self.level, solved=list(self.solved_now), merges=mlog,
                                 lib={k: pdsl.show(v) for k, v in self.lib.items()}))
        line = (f"L{self.level} sleep: solved {self.solved_now}; merges "
                + "; ".join(f"{m['name']}={m['def']}" for m in mlog))
        if not mlog and not self.solved_now:
            self.done = True
        self.level += 1
        self.queue = None
        return line

    def program(self, task="sqmod"):
        return simplest(self.corpus[task], self.lib) if task in self.corpus else None
