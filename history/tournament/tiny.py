"""The tournament: bridge/blocksearch.py's block-wise whole-option search, turned into a learned mixture that runs
inside training (TOURNAMENT.md).

    ~/github/sotaku/venv/bin/python -m tournament.tiny [seed] [key=value ...]     # one run, locally (seconds)

The network is the bridge's looped transformer on the tiny long-division grid (bridge/grid.py, bridge/net.py): three
heads copy a source cell's state into each cell's ports, one rule per kind computes the cell's new state, and the
block runs D ticks. Its wiring is chosen at free attention's width: each of the seven ports a kind uses (SHIFT.A, SUB.A,
SUB.B, SUB.C, SEL.A, SEL.B, SEL.C) reads one of `options()` (blocksearch's 34: MENU's 7 rules plus every other
relative offset in rows -4..0, columns -3..3 that the causal mask allows), tied over every cell of its kind: 34^7 =
5.3e10 wirings. The subtraction cell is a ReLU MLP (the H1 form: the table the machine reads is the MLP's outputs on
every one-hot input, bridge notch 1e); the copy and the select are compiled rules.

The wiring's parameters are logits over whole combinations, one vector per block of interacting ports:
    block "sub"  SHIFT.A, SUB.A, SUB.B, SUB.C    34^4 = 1,336,336 combinations
    block "sel"  SEL.A, SEL.B, SEL.C             34^3 =    39,304 combinations
Each training step (one optimizer step on one loss, as the evaluator runs it), with the snapped MLP table:
  * every active combination of block "sub" runs on a grid of one link (the same network on a smaller grid), on the
    batch's own numbers x = R0 (below N) and x = N: its verdict is whether the subtraction borrows exactly when x < N
    (blocksearch's stage 1; label-free: x < N is an order fact of the data);
  * every active pair (a "sub" combination that passes, a "sel" combination) runs on the same grid: x below N must
    come out unchanged and N must become 0 (the one-link law), and, with `answers`, the pair's whole network must get
    the batch's labelled answers right on the full grid (blocksearch's stage 2);
  * the loss is the mixture's: -log sum_c pi(c) verdict(c) for block "sub", and -log sum pi_sub(c1) pi_sel(c2)
    verdict(c1, c2) for the pairs, pi the softmax of the block's logits over every combination. Its gradient raises
    the combinations that pass and lowers the ones that fail (SpecBit's and WIRING.md's whole-option mixtures, here
    with one weight per whole combination). Verdicts are computed under no_grad, inside the loss;
  * a combination is active while its logit is within `prune` nats of its block's best: the others carry less than
    e^-prune of the best one's weight each, and count as failing (the mixture's sum skips them). That is what keeps
    the cost down: the active set shrinks as the tournament goes;
  * the MLP subtraction cell learns from the cell's own lessons (ANCHORS.md in bridge form: x < N on every top part,
    the column and row order laws, bijectivity), as in bridge notch 1e. With `gate`, the wiring's lessons join once
    the snapped table passes all of them for `gate_win` steps (WIRING.md: the switches wait for the tables).
The prediction path (`answer_ste`, rule 8): the argmax combination of each block as a straight-through one-hot over
every combination (its value is exactly the argmax wiring; its gradient reaches every combination's logit through
the ports' marginals), and the MLP's table straight-through; `grad_test` checks that the final answer's gradient
reaches both blocks' logits and the MLP.

A run stops `post` steps after its argmax network first gets every training answer right, or at `steps`. It is
certified, as in the bridge (TINYROUTING.md's test), if that network is exact on every question of the training size
and of (B, w, k + 1) and (B, w + 1, k + 1).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bridge import census as Cs
from bridge import grid as G
from bridge import net as Nn
from bridge import train as Tr
from bridge.grid import NIN, SEL, SHIFT, SUB, VIN, Z

HERE = Path(__file__).resolve().parent
PORTS = Cs.PORTS                                      # (head, kind) of the seven ports, census order
PORT_NAMES = [f"{G.HEADS[h]}.{G.KINDS[kd]}" for h, kd in PORTS]
BLOCKS = (("sub", (0, 1, 2, 3)), ("sel", (4, 5, 6)))


def options(rows: int = 4, cols: int = 3) -> list:
    """MENU's 7 rules plus every other relative offset with row -rows..0 and column -cols..cols that the causal mask
    allows (blocksearch's menu at rows 4, cols 3: 34 options)."""
    extra = [f"rel{dr},{dc}" for dr in range(-rows, 1) for dc in range(-cols, cols + 1)
             if (dr, dc) not in ((-1, 0), (-2, 0), (0, -1), (-1, -1)) and not (dr == 0 and dc >= 0)]
    return list(Nn.MENU) + extra


@dataclass
class Cfg:
    B: int = 3
    w: int = 2
    k: int = 1
    rows: int = 4                     # the menu's window (options())
    cols: int = 3
    seed: int = 0
    steps: int = 600
    post: int = 30                    # stop this many steps after the argmax network first fits every answer
    every: int = 5                    # measure every this many steps
    cell_form: str = "mlp"            # the subtraction cell: "mlp" a ReLU MLP on its ports (bridge notch 1d) |
                                      # "mlptab" the table is a ReLU MLP's outputs on every one-hot input (notch 1e,
                                      # the H1 file's form)
    hidden: int = 0                   # the MLP cell's hidden units (0: the bridge's default, 3 * 2 * B * B)
    lr_cell: float = 0.01             # Adam on the MLP cell
    lr_wire: float = 0.5              # Adam on the combinations' logits
    init: float = 0.1                 # std of the combinations' initial logits
    prune: float = 12.0               # active: within this many nats of the block's best logit
    pair_cap: int = 4096              # at most this many passing "sub" combinations (the best) enter the pairs
    pair_budget: int = 20_000_000     # the pairs' lesson waits while the active pairs exceed this
    answers: bool = True              # the pairs' verdict includes the batch's labelled answers (full grid)
    label_frac: float = 1.0           # the share of training questions that carry an answer
    batch: int = 0                    # questions per step (0: every question of the size)
    verdict: str = "batch"            # "batch": a combination passes if it passes every row of the step; "row": the
                                      # mixture per row (WIRING.md's mixture_nll), averaged
    tables: str = "learned"           # "learned" (the MLP cell) | "given" (the truth tables, wiring only)
    gate: bool = True                 # the wiring's lessons wait until the snapped table passes its own checks
    gate_win: int = 5
    cell: dict = field(default_factory=lambda: {"pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0})
    sub_law: bool = True              # block "sub" has its own lesson (else it learns only through the pairs)
    stage_rows: int = 4               # verdicts are computed on this many rows first, then on all for survivors
    chunk: int = 8192
    cert_sample: int = 20000
    threads: int = 4
    tag: str = ""


# ------------------------------------------------------------------------------------------------ the hard network
def simulate(lay: G.Layout, Nd, vd, src, Td, Tb, ticks=None):
    """The hard network for W wirings at once (bridge census.simulate with the subtraction a table): src [3, W, L]
    source cells per head, Td / Tb [B * B * 2] the snapped subtraction table's digit and borrow, indexed
    (a * B + b) * 2 + c. Returns digits [W, X, L] and borrows [W, X, L] (int64)."""
    B = lay.B
    W, L = src.shape[1:]
    X = Nd.shape[0]
    nin = np.where(lay.kind == NIN)[0]
    vin = np.where(lay.kind == VIN)[0]
    inp = torch.zeros(X, L, dtype=torch.int64)
    inp[:, nin] = Nd[:, lay.src[nin]]
    inp[:, vin] = vd[:, lay.src[vin]]
    is_in = torch.from_numpy(lay.is_input)
    kind = torch.from_numpy(lay.kind)
    dig = inp[None].expand(W, X, L).clone()
    bor = torch.zeros(W, X, L, dtype=torch.int64)
    g = lambda t, s: torch.gather(t, 2, s[:, None, :].expand(W, X, L))
    is_sub, is_shift = kind == SUB, kind == SHIFT
    for _ in range(ticks or lay.depth):
        a, b, c = g(dig, src[0]), g(dig, src[1]), g(bor, src[2])
        idx = (a * B + b) * 2 + c
        nd = torch.where(is_shift, a, torch.where(is_sub, Td[idx], torch.where(c == 1, a, b)))
        nb = torch.where(is_sub, Tb[idx], torch.zeros_like(c))
        dig = torch.where(is_in, inp, nd)
        bor = torch.where(is_in, torch.zeros_like(nb), nb)
    return dig, bor


class Wires:
    """Per-layout source maps: for every port and option, each cell's source (cells of other kinds keep their own
    index; SHIFT's unused ports read cell 0, a constant zero)."""

    def __init__(self, lay: G.Layout, opts: list):
        self.lay = lay
        S = Cs.sources(lay, opts)                                               # [n_opts, L]
        kind = torch.from_numpy(lay.kind)
        L = lay.L
        self.base = torch.arange(L).expand(3, L).clone()
        for h in (1, 2):
            self.base[h, kind == SHIFT] = 0
        self.S = S
        self.cells = [torch.where(kind == kd)[0] for _, kd in PORTS]

    def sources(self, combos: torch.Tensor) -> torch.Tensor:
        """combos [W, 7] option indices -> src [3, W, L]."""
        W = combos.shape[0]
        src = self.base[:, None, :].expand(3, W, self.lay.L).clone()
        for j, (h, kd) in enumerate(PORTS):
            cells = self.cells[j]
            src[h][:, cells] = self.S[combos[:, j]][:, cells]
        return src


def decode(idx: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Flat combination indices [W] -> option indices [W, m] (base n, first port most significant)."""
    place = torch.tensor([n ** (m - 1 - j) for j in range(m)], dtype=torch.int64)
    return (idx[:, None] // place[None]) % n


def encode(opts: torch.Tensor, n: int) -> torch.Tensor:
    m = opts.shape[1]
    place = torch.tensor([n ** (m - 1 - j) for j in range(m)], dtype=torch.int64)
    return (opts * place[None]).sum(1)


# ------------------------------------------------------------------------------------------------ the model
class Tiny:
    def __init__(self, cfg: Cfg):
        self.cfg = c = cfg
        torch.manual_seed(c.seed)
        torch.set_num_threads(c.threads)
        self.opts = options(c.rows, c.cols)
        self.n = n = len(self.opts)
        self.true = [self.opts.index(Nn.TRUE_MENU[p]) for p in PORTS]
        # the MLP subtraction cell and its lessons: the bridge's notch 1e (fixed attention, the table an MLP's
        # outputs); only its rule and its cell-level lessons are used here
        if c.cell_form == "mlptab":
            net = dict(attn="fixed", rule="mlptab", learn=("SUB",), hidden=c.hidden)
        else:
            net = dict(attn="fixed", rule="mlp", mlp_kinds=("SUB",), hidden=c.hidden)
        self.cell = Tr.Run(Tr.RunCfg(B=c.B, w=c.w, k=c.k, net=net, seed=c.seed, dtype="float64", cert=False))
        g = torch.Generator().manual_seed(5000 + c.seed)
        self.theta = {name: torch.nn.Parameter(torch.randn(n ** len(ports), generator=g, dtype=torch.float64) * c.init)
                      for name, ports in BLOCKS}
        groups = [{"params": list(self.theta.values()), "lr": c.lr_wire}]
        if c.tables == "learned":
            groups.append({"params": [p for p in self.cell.net.parameters()], "lr": c.lr_cell})
        self.opt = torch.optim.Adam(groups)
        B, w, k = c.B, c.w, c.k
        self.lay = G.Layout(B, w, k)
        self.lay1 = G.Layout(B, w, 1, links=1)
        self.wires = Wires(self.lay, self.opts)
        self.wires1 = Wires(self.lay1, self.opts)
        self.data = G.Data(B, w, k)
        rng = np.random.default_rng(100 + c.seed)
        self.labelled = np.zeros(self.data.X, dtype=bool)
        self.labelled[rng.choice(self.data.X, max(1, round(c.label_frac * self.data.X)), replace=False)] = True
        self.rng = np.random.default_rng(200 + c.seed)
        self.big = [(G.Layout(B, w, k + 1), G.Data(B, w, k + 1, sample=c.cert_sample, seed=1)),
                    (G.Layout(B, w + 1, k + 1), G.Data(B, w + 1, k + 1, sample=c.cert_sample, seed=1))]
        self.step = 0
        self.gate_run = 0
        self.log = []
        self.events = {}

    # ----- the snapped table -----
    def tables(self):
        """The snapped subtraction table, flat [B * B * 2]: digit Td and borrow Tb, indexed (a * B + m) * 2 + c."""
        B = self.cfg.B
        if self.cfg.tables == "given":
            a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
            x = (a - m - cc).flatten()
            return x % B, (x < 0).long()
        with torch.no_grad():
            Td, Tb = self.cell.rule_table(hard=True)                         # [B, B, 2, B], [B, B, 2, 2]
        return Td.argmax(-1).flatten(), Tb.argmax(-1).flatten()

    def table_right(self, Td, Tb) -> int:
        """Entries (digit and borrow) equal to the truth (a report only; no lesson reads it)."""
        B = self.cfg.B
        a, m, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
        x = (a - m - cc).flatten()
        return int(((Td == x % B) & (Tb == (x < 0).long())).sum())

    def table_checks(self, Td, Tb, data) -> bool:
        """The cell's own lessons, hard, on the snapped table: x < N on every top part (pref), the two order laws
        (colall, rowall: the learned borrows referee), bijectivity of every row and column."""
        B, P = self.cfg.B, self.cfg.w + 1
        T = lambda a, m, c: (Td[(a * B + m) * 2 + c], Tb[(a * B + m) * 2 + c])
        d3 = Td.view(B, B, 2)
        b3 = Tb.view(B, B, 2)
        for m in range(B):                                                 # bijectivity
            for c in range(2):
                if len(set(d3[:, m, c].tolist())) < B or len(set(d3[m, :, c].tolist())) < B:
                    return False
        for m in range(B):                                                 # the column law
            for c in range(2):
                for r in range(B):
                    for r2 in range(r + 1, B):
                        w0, w1, lt = int(b3[r, m, c]), int(b3[r2, m, c]), int(b3[r, r2, 0])
                        first = (w0 < w1) or (w0 == w1 and lt == 1)
                        lo, hi = (d3[r, m, c], d3[r2, m, c]) if first else (d3[r2, m, c], d3[r, m, c])
                        if not lo < hi:
                            return False
        for a in range(B):                                                 # the row law
            for c in range(2):
                for m in range(B):
                    for m2 in range(m + 1, B):
                        w0, w1, lt = int(b3[a, m, c]), int(b3[a, m2, c]), int(b3[m, m2, 0])
                        first_lower = (w0 == 0) if w0 != w1 else (lt == 0)
                        lo, hi = (d3[a, m, c], d3[a, m2, c]) if first_lower else (d3[a, m2, c], d3[a, m, c])
                        if not lo < hi:
                            return False
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)          # x < N on every top part
        Nd = torch.from_numpy(G.digits(pairs[:, 0], B, P))
        Rd = torch.from_numpy(G.digits(pairs[:, 1], B, P))
        for j in range(P):
            c1 = torch.zeros(len(pairs), dtype=torch.int64)
            c2 = torch.zeros(len(pairs), dtype=torch.int64)
            for i in range(j, P):
                _, c1 = T(Rd[:, i], Nd[:, i], c1)
                _, c2 = T(Nd[:, i], Rd[:, i], c2)
            differ = (Rd[:, j:] != Nd[:, j:]).any(1).long()
            if not (bool((c1 == differ).all()) and bool((c2 == 0).all())):
                return False
        return True

    # ----- the rows a step's lessons read -----
    def batch(self):
        c = self.cfg
        d = self.data
        if c.batch and d.X > c.batch:
            return d.subset(np.sort(self.rng.choice(d.X, c.batch, replace=False)))
        return d

    def law_rows(self, data):
        """The one-link grid's rows: x = each (N, R0) pair's R0 (below N), and x = N; their targets x mod N, and
        [x < N] for the subtraction's final borrow."""
        B, w = self.cfg.B, self.cfg.w
        pairs = np.unique(np.stack([data.N, data.R0], 1), axis=0)
        Ns = np.concatenate([pairs[:, 0], np.unique(data.N)])
        xs = np.concatenate([pairs[:, 1], np.unique(data.N)])
        order = self.rng.permutation(len(Ns))                              # the staged first rows vary per step
        Ns, xs = Ns[order], xs[order]
        return dict(N=torch.from_numpy(G.digits(Ns, B, w)), v=torch.from_numpy(G.digits(xs, B, w + 1)),
                    t=torch.from_numpy(G.digits(xs % Ns, B, w + 1)), lt=torch.from_numpy((xs < Ns).astype(np.int64)))

    # ----- verdicts (hard, no_grad) -----
    def staged(self, fn, combos, nrows, L, rowwise=False):
        """fn(combos, rows) -> bool [W, rows]. Batch verdicts: the first stage_rows rows for every combination, then
        every row for those that pass them; returns bool [W]. Row verdicts (rowwise): every row for every
        combination; returns bool [W, nrows]. Chunks hold about `chunk` * 100 cell states."""
        c = self.cfg
        out = []
        if rowwise:
            rows = torch.arange(nrows)
            step = max(1, c.chunk * 100 // max(1, nrows * L))
            for a in range(0, len(combos), step):
                out.append(fn(combos[a:a + step], rows))
            return torch.cat(out, 0) if out else torch.zeros(0, nrows, dtype=torch.bool)
        first = torch.arange(min(c.stage_rows, nrows))
        rest = torch.arange(len(first), nrows)
        step1 = max(1, c.chunk * 100 // max(1, len(first) * L))
        step2 = max(1, c.chunk * 100 // max(1, nrows * L))
        for a in range(0, len(combos), step1):
            cb = combos[a:a + step1]
            alive = fn(cb, first).all(1)
            if len(rest) and bool(alive.any()):
                w = torch.where(alive)[0]
                for b in range(0, len(w), step2):
                    wb = w[b:b + step2]
                    alive[wb] = fn(cb[wb], rest).all(1)
            out.append(alive)
        return torch.cat(out, 0) if out else torch.zeros(0, dtype=torch.bool)

    def v_sub(self, c1: torch.Tensor, law, tabs, rowwise=False):
        """Block "sub" on the one-link grid: the subtraction's final borrow is [x < N] (the select's ports cannot
        reach the subtraction's row, so they are filled with option 0)."""
        lay1, n = self.lay1, self.n
        top = lay1.cell(lay1.Rn - 2, lay1.P - 1)

        def fn(cb, rows):
            combos = torch.cat([decode(cb, n, 4), torch.zeros(len(cb), 3, dtype=torch.int64)], 1)
            _, b = simulate(lay1, law["N"][rows], law["v"][rows], self.wires1.sources(combos), *tabs)
            return b[:, :, top] == law["lt"][rows][None]
        return self.staged(fn, c1, law["N"].shape[0], lay1.L, rowwise)

    def v_pair(self, c1: torch.Tensor, c2: torch.Tensor, law, tabs, data, rowwise=False):
        """Pairs (c1 [W], c2 [W]): the one-link law (x below N unchanged, N to 0, the final borrow [x < N]), then
        (with `answers`) the batch's labelled answers on the full grid for the pairs that pass it."""
        lay1, n = self.lay1, self.n
        last = lay1.Rn - 1
        cells = torch.tensor([lay1.cell(last, j) for j in range(lay1.P)])
        top = lay1.cell(lay1.Rn - 2, lay1.P - 1)
        codes = torch.stack([c1, c2], 1)

        def fn(cb, rows):
            combos = torch.cat([decode(cb[:, 0], n, 4), decode(cb[:, 1], n, 3)], 1)
            d, b = simulate(lay1, law["N"][rows], law["v"][rows], self.wires1.sources(combos), *tabs)
            return (d[:, :, cells] == law["t"][rows][None]).all(-1) & (b[:, :, top] == law["lt"][rows][None])
        ok = self.staged(fn, codes, law["N"].shape[0], lay1.L, rowwise)
        if not self.cfg.answers:
            return ok
        lab = np.where(self.labelled_in(data))[0]
        if len(lab) == 0:
            return ok
        lay = self.lay
        Nd, vd, yd = data.Nd[lab], data.vd[lab], data.yd[lab]
        ans = torch.from_numpy(lay.answer)
        alive = ok.any(1) if rowwise else ok
        w = torch.where(alive)[0]
        if len(w):
            def fa(cb, rows):
                combos = torch.cat([decode(cb[:, 0], n, 4), decode(cb[:, 1], n, 3)], 1)
                d, _ = simulate(lay, Nd[rows], vd[rows], self.wires.sources(combos), *tabs)
                return (d[:, :, ans] == yd[rows][None]).all(-1)
            okA = self.staged(fa, codes[w], len(lab), lay.L, rowwise)
            if rowwise:
                full = torch.zeros(len(codes), ok.shape[1] + len(lab), dtype=torch.bool)
                full[:, :ok.shape[1]] = ok
                full[w, ok.shape[1]:] = okA & ok[w].all(1, keepdim=True)
                return full
            alive[w] = okA
        elif rowwise:
            return torch.cat([ok, torch.zeros(len(codes), len(lab), dtype=torch.bool)], 1)
        return alive

    def labelled_in(self, data):
        if data is self.data:
            return self.labelled
        key = {(int(N), int(v)): i for i, (N, v) in enumerate(zip(self.data.N, self.data.v))}
        return np.array([self.labelled[key[(int(N), int(v))]] for N, v in zip(data.N, data.v)])

    # ----- the lessons -----
    @staticmethod
    def mix(logw: torch.Tensor, verdict: torch.Tensor) -> torch.Tensor | None:
        """-log sum_c w_c verdict_c (verdict [K] or [K, rows]: then per row, averaged); None if nothing passes."""
        if verdict.dim() == 1:
            if not bool(verdict.any()):
                return None
            return -torch.logsumexp(logw[verdict], 0)
        if not bool(verdict.any()):
            return None
        lv = torch.where(verdict, 0.0, -math.inf).to(logw.dtype)
        per = torch.logsumexp(logw[:, None] + lv, 0)
        per = per[torch.isfinite(per)]                                       # rows no combination passes: no signal
        return -per.mean()

    def active(self, name):
        th = self.theta[name].detach()
        return torch.where(th >= th.max() - self.cfg.prune)[0]

    def lessons(self):
        c = self.cfg
        data = self.batch()
        terms, loss, info = {}, 0.0, {}
        if c.tables == "learned":
            for name, wgt in c.cell.items():
                t = self.cell.pref(data) if name == "pref" else getattr(self.cell, name)()
                terms[name] = float(t.detach())
                loss = loss + wgt * t
        tabs = self.tables()
        ok = c.tables == "given" or self.table_checks(*tabs, data)
        self.gate_run = self.gate_run + 1 if ok else 0
        info["checks"] = bool(ok)
        if c.gate and self.gate_run < c.gate_win and c.tables == "learned":
            return loss, terms, info
        if "gate_open" not in self.events:
            self.events["gate_open"] = self.step
        law = self.law_rows(data)
        rowwise = c.verdict == "row"
        lw1 = self.theta["sub"].log_softmax(0)
        lw2 = self.theta["sel"].log_softmax(0)
        with torch.no_grad():
            a1 = self.active("sub")
            v1 = self.v_sub(a1, law, tabs, rowwise)
            pass1 = v1.all(1) if rowwise else v1
        info["active_sub"] = len(a1)
        info["pass_sub"] = int(pass1.sum())
        if c.sub_law:
            t = self.mix(lw1[a1], v1)
            if t is not None:
                terms["sub"] = float(t.detach())
                loss = loss + t
        # the pairs: the best passing "sub" combinations (all of the active ones without a lesson of their own)
        cand = a1[pass1] if c.sub_law else a1
        if len(cand) > c.pair_cap:
            cand = cand[lw1.detach()[cand].topk(c.pair_cap).indices]
        a2 = self.active("sel")
        info["active_sel"] = len(a2)
        info["pairs"] = len(cand) * len(a2)
        if 0 < len(cand) * len(a2) <= c.pair_budget:
            with torch.no_grad():
                c1 = cand.repeat_interleave(len(a2))
                c2 = a2.repeat(len(cand))
                v2 = self.v_pair(c1, c2, law, tabs, data, rowwise)
            info["pass_pairs"] = int((v2.all(1) if rowwise else v2).sum())
            t = self.mix(lw1[c1] + lw2[c2], v2)
            if t is not None:
                terms["pair"] = float(t.detach())
                loss = loss + t
        return loss, terms, info

    # ----- the argmax network and the measures -----
    def argmax_wiring(self) -> list:
        out = []
        for name, ports in BLOCKS:
            out += decode(self.theta[name].detach().argmax()[None], self.n, len(ports))[0].tolist()
        return out

    @torch.no_grad()
    def exact(self, lay, data, wiring=None, tabs=None, wires=None):
        wiring = self.argmax_wiring() if wiring is None else wiring
        tabs = self.tables() if tabs is None else tabs
        wires = Wires(lay, self.opts) if wires is None else wires
        src = wires.sources(torch.tensor([wiring]))
        ans = torch.from_numpy(lay.answer)
        n = 0
        for i in range(0, data.X, 4096):
            sl = slice(i, i + 4096)
            d, _ = simulate(lay, data.Nd[sl], data.vd[sl], src, *tabs)
            n += int((d[0][:, ans] == data.yd[sl]).all(-1).sum())
        return n / data.X

    def trap(self, wiring) -> dict:
        """Each wrong port's sources on the training grid: the share of its cells that read an input cell the
        machine does not (a bypass), a constant zero, or another computed cell (bridge probe.py's classes)."""
        lay = self.lay
        src = self.wires.sources(torch.tensor([wiring]))[:, 0]
        out = {}
        for j, (h, kd) in enumerate(PORTS):
            if wiring[j] == self.true[j]:
                continue
            cells = np.where(lay.kind == kd)[0]
            s = src[h, cells].numpy()
            comp = lay.comp[h, cells]
            same = (s == comp) | ((lay.kind[s] == Z) & (lay.kind[comp] == Z))
            inp = np.isin(lay.kind[s], (NIN, VIN)) & ~same
            zero = (lay.kind[s] == Z) & ~same
            out[PORT_NAMES[j]] = {"option": self.opts[wiring[j]], "input": float(inp.mean()),
                                  "zero": float(zero.mean()), "computed": float((~inp & ~zero & ~same).mean())}
        return out

    # ----- the prediction path (rule 8) -----
    def answer_ste(self, lay, Np, vp):
        """The answer's digit distributions on the prediction path: every port's attention the straight-through
        one-hot of its block's argmax combination (value: exactly that option's pattern; gradient: the ports'
        marginals of the block's softmax), the MLP's table straight-through, one-hot states throughout."""
        wires = Wires(lay, self.opts)
        n = self.n
        L = lay.L
        A = torch.zeros(3, L, L, dtype=torch.float64)
        for name, ports in BLOCKS:
            p = self.theta[name].softmax(0)
            hard = F.one_hot(p.detach().argmax(), p.shape[0]).to(p.dtype)
            st = hard + (p - p.detach())
            allc = decode(torch.arange(p.shape[0]), n, len(ports))
            for i, j in enumerate(ports):
                h, kd = PORTS[j]
                marg = torch.zeros(n, dtype=p.dtype).index_add(0, allc[:, i], st)   # [n] the port's marginal
                cells = wires.cells[j]
                M = torch.zeros(n, L, L, dtype=torch.float64)
                M[torch.arange(n)[:, None], cells[None], wires.S[:, cells]] = 1
                A[h, cells] = A[h, cells] + torch.einsum("o,olk->lk", marg, M)[cells]
        for h in range(3):                                                  # the cells a port does not use
            rest = torch.ones(L, dtype=torch.bool)
            for j, (hh, kd) in enumerate(PORTS):
                if hh == h:
                    rest[wires.cells[j]] = False
            A[h, rest, wires.base[h, rest]] = 1
        B = self.cfg.B
        T = self.cell.net.truth.clone()                                     # copy and select: compiled
        Sd, Sb = self.cell.rule_table()                                     # the MLP on every one-hot (a, m, c)
        soft = torch.cat([Sd, Sb], -1)
        hard = torch.cat([F.one_hot(Sd.argmax(-1), B), F.one_hot(Sb.argmax(-1), 2)], -1).to(soft.dtype)
        T[G.SCRATCH.index(SUB)] = hard + (soft - soft.detach())
        Tcell = torch.zeros(L, 2 * B * B, B + 2, dtype=torch.float64)
        kinds = torch.from_numpy(lay.kind)
        for i, kd in enumerate(G.SCRATCH):
            Tcell[kinds == kd] = T[i].reshape(2 * B * B, B + 2)
        S_in = G.input_states(lay, Np, vp)
        S = S_in
        inp = torch.from_numpy(lay.is_input)
        for _ in range(lay.depth):
            S = G.tick_table(S, A, Tcell, B)
            S = torch.where(inp[None, :, None], S_in, S)
        return S[:, lay.answer, :B]

    def grad_test(self, n: int = 16) -> dict:
        """The final answer's cross-entropy on n training questions, one backward pass: which parameter tensors get a
        nonzero gradient, and whether the prediction path's answers equal the hard network's."""
        d = self.data.subset(np.arange(min(n, self.data.X)))
        for p in self.params():
            p.grad = None
        p = self.answer_ste(self.lay, G.onehot(d.Nd, self.cfg.B), G.onehot(d.vd, self.cfg.B))
        hard = self.exact(self.lay, d)
        same = float((p.argmax(-1) == d.yd).all(-1).double().mean())
        loss = -p.clamp_min(1e-12).log().gather(-1, d.yd[..., None]).mean()
        loss.backward()
        out = {"answers_equal_hard": same == hard, "ste_exact": same, "hard_exact": hard}
        for name, th in self.theta.items():
            out[f"grad_{name}"] = bool(th.grad is not None and th.grad.abs().sum() > 0)
        out["grad_cell"] = all(q.grad is not None and q.grad.abs().sum() > 0 for q in self.cell.net.parameters())
        for q in self.params():
            q.grad = None
        return out

    def params(self):
        return list(self.theta.values()) + list(self.cell.net.parameters())

    # ----- training (one optimizer step per batch: the evaluator's loop) -----
    def train(self) -> dict:
        c = self.cfg
        t0 = time.time()
        hit = None
        prev = None
        last_change = 0
        while self.step < c.steps:
            ts = time.time()
            self.opt.zero_grad()
            loss, terms, info = self.lessons()
            if torch.is_tensor(loss) and loss.requires_grad:
                loss.backward()
                self.opt.step()
            self.step += 1
            wiring = self.argmax_wiring()
            if wiring != prev:
                last_change, prev = self.step, wiring
            if wiring == self.true and "true_wiring" not in self.events:
                self.events["true_wiring"] = self.step
            if self.step % c.every == 0 or self.step == 1:
                tabs = self.tables()
                ex = self.exact(self.lay, self.data, wiring, tabs, self.wires)
                row = {"step": self.step, "exact": ex, "table_right": self.table_right(*tabs),
                       "wiring": [self.opts[o] for o in wiring], "ports_right": sum(int(a == b) for a, b in
                                                                                    zip(wiring, self.true)),
                       "seconds": time.time() - t0, "step_seconds": time.time() - ts, **terms, **info}
                self.log.append(row)
                if ex == 1.0 and hit is None:
                    hit = self.step
                if hit is not None and self.step - hit >= c.post:
                    break
        tabs = self.tables()
        wiring = self.argmax_wiring()
        rec = {"cfg": asdict(c), "options": self.n, "steps": self.step, "hit": hit, "seconds": time.time() - t0,
               "events": self.events, "last_change": last_change, "table_right": self.table_right(*tabs),
               "table_entries": 2 * c.B * c.B, "wiring": {PORT_NAMES[j]: self.opts[o] for j, o in enumerate(wiring)},
               "true_wiring": wiring == self.true, "ports_right": sum(int(a == b) for a, b in zip(wiring, self.true)),
               "train_exact": self.exact(self.lay, self.data, wiring, tabs, self.wires)}
        rec["bigger"] = {f"{L2.B},{L2.w},{L2.k}": self.exact(L2, D2, wiring, tabs) for L2, D2 in self.big}
        rec["certified"] = bool(rec["train_exact"] == 1.0 and all(v == 1.0 for v in rec["bigger"].values()))
        rec["alias"] = rec["certified"] and not rec["true_wiring"]
        rec["trap"] = self.trap(wiring)
        rec["weight_on_argmax"] = {name: float(self.theta[name].detach().softmax(0).max()) for name, _ in BLOCKS}
        rec["log"] = self.log
        return rec


def run(cfg: Cfg, grad: bool = False) -> dict:
    m = Tiny(cfg)
    rec = m.train()
    if grad and cfg.tables == "learned":
        rec["grad_test"] = m.grad_test()
    return rec


def summary(r: dict) -> str:
    c = r["cfg"]
    ev = r["events"]
    return (f"{c['tag'] or ''} seed {c['seed']} | steps {r['steps']} hit {r['hit']} gate {ev.get('gate_open')} "
            f"true@{ev.get('true_wiring')} last change {r['last_change']} | train {r['train_exact']:.3f} bigger "
            + " ".join(f"{v:.3f}" for v in r["bigger"].values())
            + f" | {'CERT' if r['certified'] else 'no'}{' alias' if r['alias'] else ''} | ports {r['ports_right']}/7"
              f" table {r['table_right']}/{r['table_entries']} | {r['seconds']:.0f}s")


def parse(argv) -> Cfg:
    kw = {}
    fields = Cfg.__dataclass_fields__
    for a in argv:
        if "=" not in a:
            kw["seed"] = int(a)
            continue
        k_, v = a.split("=", 1)
        t = fields[k_].type
        if t in ("int", int):
            kw[k_] = int(v)
        elif t in ("float", float):
            kw[k_] = float(v)
        elif t in ("bool", bool):
            kw[k_] = v.lower() in ("1", "true", "yes")
        elif t in ("dict", dict) or k_ == "cell":
            kw[k_] = json.loads(v)
        else:
            kw[k_] = v
    return Cfg(**kw)


if __name__ == "__main__":
    cfg = parse(sys.argv[1:])
    rec = run(cfg, grad=True)
    print(summary(rec))
    for row in rec["log"][-3:]:
        print({k: v for k, v in row.items() if k != "wiring"})
    print(rec["wiring"], rec["trap"], rec.get("grad_test"))
