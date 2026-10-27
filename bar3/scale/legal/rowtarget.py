"""The full-size target mode (LEGAL_TEST.md §2; bar3/LEGAL_SOURCES.md §4–§5): the pieces a composition-harness climb
would call to pose bar 3's target-derived sources on H1's own answer rows at full size, with no Python arithmetic in a
target, an input or a decision. The harness's small target (oracle_s.SmallTarget: x'^2 mod N' at the practice widths,
by Python's integers) is what it replaces.

- **The target's rows:** H1's train rows with T = 1, read as digit strings (`answer_rows`: the composition harness's
  400, core_c3.load_rows(400, seed=1, T=1), or all 1,000).
- **The target:** x in X, N in N, the row's own answer in A, at the round's full-size setting (16 places).
- **The look's held states** (`held`): the round's own search (SelfClimb.search_round's vocabulary: the routines and
  the straight-line primitives) run on every answer row, every state of cost <= max_cost kept.
- **A value:** what a program of the learner leaves in a register, run on its own machine; `clean` when two dirty
  starts give the same digits (CLEAN_VALUES).
- **The law view by routines** (`law_mod`): "the answer is g mod N", tested with the learner's own chain (its
  practised reduction below B N, bound to A and N) and its comparison (lt on the pair, bound to A and N). On the rows
  where the comparison says the chain's result is below N, which are the rows where the chain is exact (g < 11 N),
  the result must equal the answer digit for digit, on at least MIN_ROWS rows; a value equal to the answer on every
  row is the target posed again and does not pass.
- **The rest of the way from a value** (`rest_root`, `rest_acceptor`): X = the value, N = N, the answer the row's own;
  probed at full size over the composition vocabulary (`comp_classes`), accepted on the answer rows.

Nothing here computes a number from a value: rows are read, values come from the learner's programs, and every test is
a routine's result compared digit for digit. (offline.py compares with Python's integers only to report.)
"""

from __future__ import annotations

import json
import os

import torch

from integration.lib.libcore import (A_, ONE, PRIMS, RI, Batch, Cells, apply_items, enumerate_items, fresh, gen_on,
                                     round_settings, run_items)
from bar3.scale.slice.spcore_s import Goal, bfs_multi

P = 16                          # the round's full-size registers (core_c3's P_round)
MIN_ROWS = 4                    # the law view's floor: rows where the chain is exact
REDUCTIONS = ("x+on", "x+dn", "Bx+d", "pair")     # red's patterns, the widest exact range first


def routine_for(ops, law, patterns, binding="AN"):
    """The learner's practised routine for law on the first of patterns that has one bound to binding, or None.
    ops: the climb's record form, {"law@pattern": [[token, binding], ...]}."""
    for pat in patterns:
        for tok, bd in ops.get(f"{law}@{pat}", []):
            if bd == binding:
                return tok
    return None


def h1_rows(h1_dir, T=1, split="train"):
    """H1's rows of one T, in file order (as core_c3.load_rows filters them), read one line at a time."""
    out = []
    with open(os.path.join(h1_dir, split + ".jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["time_steps"] == T:
                out.append(dict(x=r["x"], modulus=r["modulus"], result=r["result"]))
    return out


def answer_rows(h1_dir, n=400, seed=1):
    """The composition harness's answer rows (core_c3.load_rows(n, seed, T=1)): the same torch permutation."""
    rows = h1_rows(h1_dir)
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits(rows, key):
    """A field's decimal string as digits [n, P] (LSB first), read, never computed (core_c3.digits)."""
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:P] + [0] * (P - min(P, len(str(r[key]))))
                         for r in rows])


class Never:
    """No state is the goal (SelfClimb._round_many's 'never')."""

    name = "never"

    def test(self, batches, lib=None):
        return torch.zeros(batches[0].S, dtype=torch.bool, device=batches[0].regs.device)

    def fits(self, batches):
        return torch.zeros(batches[0].S, dtype=torch.long, device=batches[0].regs.device)


class Kept:
    """Every state but the root counts as reached, so that the search keeps its last level too (bfs_roots keeps only
    goal states at max_cost); nothing is accepted, so it never stops early."""

    name = "kept"

    def __init__(self, root):
        self.root = root.features()

    def test(self, batches, lib=None):
        f = batches[0].features()
        return ~(f == self.root).all(1)

    def fits(self, batches):
        return torch.zeros(batches[0].S, dtype=torch.long, device=batches[0].regs.device)


class RowTarget:
    """The target on H1's answer rows at full size."""

    def __init__(self, rows, device, cells=None, seed=0):
        self.device, self.seed, self.n = device, seed, len(rows)
        self.st = round_settings(cells or Cells(10, device), P)
        self.Nd, self.Xd, self.Yd = (digits(rows, k).to(device) for k in ("modulus", "x", "result"))

    # ------------------------------------------------------------------ machines
    def machine(self, inputs, idx=None, seed=None):
        """One fresh full-size machine (S = 1) over the rows idx (every row by default): inputs {reg: digits [n, P]};
        the other registers a dirty start."""
        n = self.n if idx is None else len(idx)
        ins = {r: (v if idx is None else v[idx]) for r, v in inputs.items()}
        return fresh(self.st, ins, gen_on(self.device, self.seed if seed is None else seed), n, self.device)

    def machines(self, S, regs, seed=5):
        """S fresh full-size machines over every row: regs {reg: digits [S, P, n] or [P, n]} (uint8); the rest dirty."""
        gen = gen_on(self.device, seed)
        def rnd(hi, shape):
            return torch.randint(0, hi, shape, generator=gen, device=self.device, dtype=torch.uint8)
        b = Batch(rnd(10, (S, 5, P, self.n)), rnd(2, (S, self.n)), rnd(10, (S, self.n)), rnd(10, (S, self.n)),
                  torch.zeros(S, dtype=torch.long, device=self.device))
        for r, v in regs.items():
            b.regs[:, RI[r]] = v
        return b

    def goal(self, idx=None, name="target"):
        """The answers as a goal (spcore_s.Goal): A = the row's answer on every row of idx."""
        Y = self.Yd if idx is None else self.Yd[idx]
        return Goal(name, [{"A": Y, "Am": torch.ones(Y.shape[0], dtype=torch.bool, device=self.device)}])

    # ------------------------------------------------------------------ values
    def value(self, prog, lib, reg="A", seed=1):
        """The digits [n, P] prog leaves in reg, run from x in X and N in N on a fresh machine (its own program)."""
        b = self.machine({"X": self.Xd, "N": self.Nd}, seed=seed)
        run_items(self.st, b, prog, lib)
        return b.regs[0, RI[reg]].T.long()

    def clean(self, prog, lib, reg="A"):
        """CLEAN_VALUES' test: the same value under two dirty starts."""
        return bool((self.value(prog, lib, reg, seed=1) == self.value(prog, lib, reg, seed=4242)).all())

    # ------------------------------------------------------------------ the look
    def held(self, lib, max_cost=3, max_states=10_000_000, deadline=None, chunk=1 << 15, fuse=1):
        """The round's search on every answer row, no goal, every state of cost <= max_cost kept: (classes, level
        sizes, stop, levels [per cost: [Batch]], parents). The vocabulary is SelfClimb.search_round's (the
        primitives, then the routines in the library's order; straight-line items only)."""
        root = self.machine({"X": self.Xd, "N": self.Nd})
        routines = [k for k in lib if k.startswith("T.")]
        classes = enumerate_items(list(PRIMS) + routines, lib, [self.st], self.device, max_body=0, deadline=deadline)
        refuse = [lambda ps, lb: [False] * len(ps)]
        out, sizes, stop, levels, parents = bfs_multi([root], [self.st], [Kept(root)], refuse, classes, lib, keep=True,
                                                      max_cost=max_cost, max_states=max_states, deadline=deadline,
                                                      chunk=chunk, fuse=fuse, max_goals=1)
        return classes, sizes, stop, levels, parents

    # ------------------------------------------------------------------ the law view, by the learner's routines
    def law_mod(self, G, lib, chain, compare, chunk=2048):
        """Values G [S, P, n] (uint8 digits on every answer row): (rows where the chain is exact [S], passes [S]) for
        "the answer is g mod N". chain: the token of the learner's reduction below B N bound to (A, N); compare: its
        lt on the pair bound to (A, N), whose borrow says A < N."""
        S = G.shape[0]
        dev = self.device
        n_exact = torch.zeros(S, dtype=torch.long, device=dev)
        ok = torch.zeros(S, dtype=torch.bool, device=dev)
        Y = self.Yd.T.to(torch.uint8)[None]
        Nn = self.Nd.T.to(torch.uint8)
        for s0 in range(0, S, chunk):
            g = G[s0:s0 + chunk]
            k = g.shape[0]
            b = self.machines(k, {"A": g, "N": Nn})
            run_items(self.st, b, (chain,), lib)
            r = b.regs[:, A_].clone()
            run_items(self.st, b, (compare,), lib)             # A = the chain's result, N = N: borrow = below N
            below = b.c == 1
            eq = (r == Y).all(1)
            same = (g == Y).all(1).all(1)
            n_exact[s0:s0 + k] = below.sum(1)
            ok[s0:s0 + k] = (below.sum(1) >= MIN_ROWS) & (eq | ~below).all(1) & ~same
        return n_exact, ok

    # ------------------------------------------------------------------ the rest of the way from a value
    def rest_root(self, value, idx):
        """The rest of the way's probes: X = the value, N = N, on the rows idx; its goal, the answers there."""
        return self.machine({"X": value, "N": self.Nd}, idx=idx), self.goal(idx, "rest")

    def rest_acceptor(self, value, idx=None):
        """A program passes if, from X = the value and N = N, it leaves the row's answer in A on every row of idx (all
        by default): spcore_s.Acceptor.many's batched test, on the data's own rows."""
        root = self.machine({"X": value, "N": self.Nd}, idx=idx, seed=11)
        goal = self.goal(idx, "rest")
        cache = {}

        def many(progs, lib):
            keys = [repr(p) for p in progs]
            live = [i for i, k in enumerate(keys) if k not in cache]
            for i0 in range(0, len(live), 256):
                part = live[i0:i0 + 256]
                b = root.take(torch.zeros(len(part), dtype=torch.long, device=self.device))
                b = apply_items(self.st, b, torch.arange(len(part), device=self.device),
                                [("REP", ONE, tuple(progs[i])) for i in part], lib)
                for i, o in zip(part, goal.test([b], lib).tolist()):
                    cache[keys[i]] = bool(o)
            return [cache[k] for k in keys]
        return many

    def comp_classes(self, lib, max_body=3, deadline=None, chunk=1 << 23):
        """The composition vocabulary (SelfClimb.comp_classes: the primitives and every routine, loops of bodies up to
        comp_body), its items told apart on full-size machines."""
        routines = tuple(sorted(k for k in lib if k.startswith("T.")))
        return enumerate_items(list(PRIMS) + list(routines), lib, [self.st], self.device, max_body=max_body,
                               deadline=deadline, chunk=chunk)
