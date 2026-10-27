"""LEGALLOOPS.md §2: written by bar3/scale/legalloops/make_copies6.py from bar3/scale/legalclimb/rowclimb.py, 0 asserted replacements
(this study's, marked "legalloops"). The rest of this docstring is the copy's.

The full-size target mode in the climb (LEGALCLIMB.md §2): bar 3's target-derived sources posed on H1's own answer
rows at full size, with no Python arithmetic in a target, an input or a decision. It is bar3/scale/legal/rowtarget.py
(test 2) with the hooks SelfClimb calls on its target, as oracle_s.SmallTarget has them (search, pose, acceptor, the
subgoal), so that it replaces the small target, x'^2 mod N' by Python's integers, in the composition harness.

Each level:
- **The look** (`search` at the pose phase): the round search's vocabulary (the primitives and the routines,
  straight-line) on every answer row, every state of cost <= look_cost kept (rowtarget.RowTarget.held). Each distinct
  value of A, X, Y and S, in LAWNEAR's order (cost, then register, then index), is tested by the law view by
  routines (RowTarget.law_mod: the learner's chain and comparison, digit equality, at least MIN_ROWS rows where the
  chain is exact). Up to LAW_NEAR that pass are the near misses.
- **The values and their rests** (`pose`): each near miss's value, if clean (two dirty starts agree) and neither an
  input nor constant, named by its digits on the answer rows. It is posed on the target's root (X = x, N = N, the
  value to leave in A), and its rest of the way on a root of its own (X = the value, N = N, the row's answer in A).
  Both at full size, on the round's 8 probe rows.
- **Acceptance** (`acceptor`): every answer row. With IDCHECK, a rest must also pass the identity check at one more
  digit: red(x + o N') = x, with N' = 10 N + d (a digit move), o N' by the learner's multiply and + x by its
  addition (test 2's post-hoc check, built by the learner's own routines).
- **SUBGOAL** (`subgoal_scores`): the value singled out is the one value that passed the law view this level, posed
  this level and not yet practised; none when several passed, or none.
- **The vocabulary** (`classes`): the composition vocabulary on full-size machines (bodies up to full_body items),
  its items' signatures cached across levels, so that a new library only runs its new items.
At the target phase there is no small target: `search` returns nothing, and the round's own search follows
(ComposeClimb with ROUND_TOO).
"""

from __future__ import annotations

import time
import zlib

import torch

from integration.lib.libcore import (A_, ONE, PRIMS, RI, apply_items, enumerate_items, gen_on, hash_rows, pad,
                                     programs, round_settings, show, _OneGoal)
from bar3.scale.legal import rowtarget as RT
from bar3.scale.legal.rowtarget import P, REDUCTIONS, RowTarget
from bar3.scale.legalloops.spcore_k import Goal, _Res, repeat_add

LAW_NEAR = 4                    # LAWNEAR: at most this many values a look (round 2's)
NP = 8                          # the full-size roots' probes: the round's (round_problem's first 8 answer rows)


def ops_routine(ops, law, patterns, binding=("A", "N")):
    """The climb's practised routine for law on the first of patterns that has one bound to binding, or None."""
    for pat in patterns:
        for tok, bd in ops.by_key.get((law, pat), []):
            if tuple(bd) == tuple(binding):
                return tok
    return None


class FullRoot:
    """A target-derived problem's probes at full size (SelfClimb's Root interface): one setting, one batch."""

    full = True

    def __init__(self, key, kind, st, batch, binding=("X", "N")):
        self.key, self.kind, self.binding = key, kind, binding
        self.pattern = key.split("@")[0]
        self.settings, self.roots = [st], [batch]
        self.goal_list = []


class FullAcceptor:
    """Every answer row (spcore_k.Acceptor.many's batched test), then the extra check if any; cached by program."""

    def __init__(self, st, batch, goal, extra=None):
        self.st, self.batch, self.goal, self.extra, self.cache = st, batch, goal, extra, {}

    def many(self, progs, lib):
        keys = [show(p) for p in progs]
        todo = {}
        for i, k in enumerate(keys):
            if k not in self.cache and k not in todo:
                todo[k] = i
        items, dev = list(todo.items()), self.batch.regs.device
        for i0 in range(0, len(items), 256):
            part = items[i0:i0 + 256]
            b = self.batch.take(torch.zeros(len(part), dtype=torch.long, device=dev))
            b = apply_items(self.st, b, torch.arange(len(part), device=dev),
                            [("REP", ONE, tuple(progs[i])) for _, i in part], lib)
            for (k, i), o in zip(part, self.goal.test([b], lib).tolist()):
                self.cache[k] = bool(o) and (self.extra is None or self.extra(progs[i], lib))
        return [self.cache[k] for k in keys]


class RowClimbTarget:
    """The hooks SelfClimb calls on its target, for the target on H1's answer rows at full size."""

    full = True
    PROJECTIONS = ()

    def __init__(self, ans, device, sources=("value", "rest"), lawnear=True, clean=True, idcheck=False, look_cost=3,
                 full_body=None, min_rows=None):
        if min_rows:                                    # ADDENDUM2.md: the law view's floor (rowtarget.MIN_ROWS)
            RT.MIN_ROWS = min_rows
        self.min_rows = RT.MIN_ROWS
        self.ans, self.device, self.sources = ans, device, sources
        self.lawnear, self.clean, self.idcheck, self.look_cost, self.full_body = (lawnear, clean, idcheck, look_cost,
                                                                                  full_body)
        self.rt = None
        self.names, self.values, self._roots = {}, {}, {}      # value digits' hash -> name; name -> (prog, reg, v)
        self.passed, self.posed = [], []                        # this level's near misses and posed value names
        self.subgoal, self.subgoals_seen = None, []
        self._cls, self._sig_cache, self.log = None, {}, []

    # ------------------------------------------------------------------ the rows
    def _rt(self, climb):
        """rowtarget.RowTarget on the answer rows (core_k's digits, read), with the climb's cells."""
        if self.rt is None:
            rt = RowTarget.__new__(RowTarget)
            rt.device, rt.seed, rt.n = self.device, 0, int(self.ans[0].shape[0])
            rt.st = round_settings(climb.cells, P)
            rt.Nd, rt.Xd, rt.Yd = (pad(v.to(self.device), P).long() for v in self.ans)
            self.rt = rt
        return self.rt

    def _all(self, n):
        return torch.ones(n, dtype=torch.bool, device=self.device)

    def root(self, climb, rows):
        return self._roots.get("tgt@XN")

    def target_goal(self, climb, rows):
        rt = self._rt(climb)
        idx = torch.arange(NP, device=self.device)
        return Goal("target", [{"A": rt.Yd[idx], "Am": self._all(NP)}])

    # ------------------------------------------------------------------ the look
    def search(self, climb, rows, budget, deadline=None, extra=None, near=0):
        """At the pose phase (near): the look, and its near misses [(program, register, exact rows)]. At the target
        phase: nothing (no small target)."""
        t0 = time.time()
        rec = dict(task="look" if near else "target", level=climb.level, found_cost=None, level_sizes=[], fits=[],
                   seconds=0.0, stop="", programs=[], near=[])
        self.passed = []
        if not near:
            rec["stop"] = "no small target (TARGET=rows): the round's own search follows"
            return [], rec, []
        chain, compare = ops_routine(climb.ops, "red", REDUCTIONS), ops_routine(climb.ops, "lt", ("pair",))
        rec.update(chain=chain, compare=compare)
        if not (self.lawnear and chain and compare):
            rec["stop"] = "no law view yet: " + ("LAWNEAR off" if not self.lawnear else "no chain or comparison")
            return [], rec, []
        rt = self._rt(climb)
        classes, sizes, stop, levels, parents = rt.held(climb.lib, max_cost=self.look_cost, deadline=deadline,
                                                        chunk=max(1, (1 << 30) // (83 * rt.n)))
        cand, vals, seen = [], [], set()
        for d in range(1, len(levels)):
            if levels[d] is None:
                continue
            lv = levels[d][0]
            for reg in "AXYS":
                G = lv.regs[:, RI[reg]]
                keep = []
                for i, hv in enumerate(hash_rows(G.reshape(G.shape[0], -1)).tolist()):
                    if hv not in seen:
                        seen.add(hv)
                        keep.append(i)
                if keep:
                    vals.append(G[torch.tensor(keep, device=G.device)])
                    cand += [(d, reg, i) for i in keep]
        if not cand:
            rec.update(level_sizes=sizes, stop=stop, values=0, passed=0, seconds=round(time.time() - t0, 1))
            return [], rec, []
        G = torch.cat(vals)
        del vals
        n_exact, ok = rt.law_mod(G, climb.lib, chain, compare)
        passing = torch.nonzero(ok).flatten().tolist()
        for j in passing:
            d, reg, i = cand[j]
            pr = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
            if pr:
                self.passed.append((pr[0], reg, int(n_exact[j])))
            if len(self.passed) >= LAW_NEAR:
                break
        del G, levels, parents
        rec.update(level_sizes=sizes, stop=stop, values=len(cand), passed=len(passing),
                   near=[(show(p), r, e) for p, r, e in self.passed], seconds=round(time.time() - t0, 1))
        return [], rec, list(self.passed)

    # ------------------------------------------------------------------ the values and their rests
    def _root(self, key, kind, inputs, idx):
        if key not in self._roots:
            rt = self.rt
            self._roots[key] = FullRoot(key, kind, rt.st, rt.machine(inputs, idx=idx, seed=zlib.crc32(key.encode())))
        return self._roots[key]

    def pose(self, climb, rows, near):
        """(root, goals) pairs: the near misses' values on the target's root, and each one's rest of the way."""
        self.posed = []
        if not near or not ({"value", "rest"} & set(self.sources)):
            return []
        rt = self._rt(climb)
        for prog, reg, exact in near:
            v = rt.value(prog, climb.lib, reg)
            note = dict(level=climb.level, program=show(prog), reg=reg, exact_rows=exact)
            if bool((v == rt.Xd).all()) or bool((v == rt.Nd).all()) or bool((v == v[:1]).all()):
                self.log.append(dict(note, posed=False, why="an input or constant"))
                continue
            if self.clean and not rt.clean(prog, climb.lib, reg):
                self.log.append(dict(note, posed=False, why="not clean"))
                continue
            h = int(hash_rows(v.to(torch.uint8)).sum())
            name = self.names.setdefault(h, f"val.{len(self.names)}")
            self.values.setdefault(name, (prog, reg, v))
            if name not in self.posed:
                self.posed.append(name)
            self.log.append(dict(note, posed=True, name=name))
        out, idx, ones = [], torch.arange(NP, device=self.device), self._all(NP)
        if self.posed and "value" in self.sources:
            root = self._root("tgt@XN", "target", {"X": rt.Xd, "N": rt.Nd}, idx)
            out.append((root, [Goal(nm, [{"A": self.values[nm][2][idx], "Am": ones}]) for nm in self.posed]))
        if "rest" in self.sources:
            for nm in self.posed:
                root = self._root(f"rest.{nm[4:]}@XN", "rest", {"X": self.values[nm][2], "N": rt.Nd}, idx)
                out.append((root, [Goal("target", [{"A": rt.Yd[idx], "Am": ones}])]))
        return out

    def acceptor(self, climb, rows, root, name):
        rt = self._rt(climb)
        if root.kind == "target":
            goal = Goal(name, [{"A": self.values[name][2], "Am": self._all(rt.n)}])
            return FullAcceptor(rt.st, rt.machine({"X": rt.Xd, "N": rt.Nd}, seed=11), goal)
        v = self.values["val." + root.key.split("@")[0][len("rest."):]][2]
        goal = Goal("target", [{"A": rt.Yd, "Am": self._all(rt.n)}])
        return FullAcceptor(rt.st, rt.machine({"X": v, "N": rt.Nd}, seed=11), goal,
                            self._identity(climb) if self.idcheck else None)

    def _identity(self, climb):
        """IDCHECK: a reduction must also give red(r + o N') = r on every answer row's instance: N' = 10 N + d (N's
        digits moved up one place, d below), r = x + k N (k a digit: k additions of N, so that r spans [0, N')), o
        another row's x moved up one place too (so that r + o N' reaches 15-16 digits, as x^2 does for 24-bit N);
        o N' by the learner's multiply, r and the sum by its addition. None (no check) while either routine is
        missing."""
        rt = self._rt(climb)
        if not (climb.ops.has("mul", "x,o") and climb.ops.has("add", "pair")):
            self.log.append(dict(level=climb.level, idcheck="skipped: no multiply or addition yet"))
            return None
        dev, n = self.device, rt.n
        i = torch.arange(n, device=dev)
        Np = torch.cat([(i // 10 % 10)[:, None], rt.Nd[:, :P - 1]], 1)
        o = torch.roll(rt.Xd, 1, 0)
        o = torch.cat([(i // 100 % 10)[:, None], o[:, :P - 1]], 1)
        gen = gen_on(dev, 7)
        r = repeat_add(climb.ops, rt.st, climb.lib, rt.Xd, rt.Nd, i % 10, gen, dev)
        prod = climb.ops.run("mul", "x,o", rt.st, climb.lib, [o, Np], gen, dev)[0]
        X = climb.ops.run("add", "pair", rt.st, climb.lib, [prod, r], gen, dev)[0]
        batch = rt.machine({"X": X, "N": Np}, seed=13)
        goal = Goal("identity", [{"A": r, "Am": self._all(n)}])

        def check(prog, lib):
            b = batch.take(torch.zeros(1, dtype=torch.long, device=dev))
            b = apply_items(rt.st, b, torch.zeros(1, dtype=torch.long, device=dev), [("REP", ONE, tuple(prog))], lib)
            ok = bool(goal.test([b], lib)[0])
            self.log.append(dict(level=climb.level, idcheck=show(prog), passed=ok))
            return ok
        return check

    # ------------------------------------------------------------------ SUBGOAL, the vocabulary, mdl
    def subgoal_scores(self, climb, rows, roots):
        fresh = [nm for nm in self.posed if f"{nm}/tgt@XN" not in climb.practiced]
        self.subgoal = fresh[0] if len(self.passed) == 1 and fresh else None
        if self.subgoal:
            self.subgoals_seen.append(self.subgoal)
        climb.records.append(dict(task="subgoal", level=climb.level, subgoal=self.subgoal, passed=len(self.passed),
                                  posed=list(self.posed)))

    def classes(self, climb, deadline=None):
        """The composition vocabulary on full-size machines, cached: rebuilt only when the routines change, and then
        only its new items are run (enumerate_items' signature cache)."""
        r = tuple(sorted(k for k in climb.lib if k.startswith("T.")))
        if self._cls is None or self._cls[0] != r:
            t0 = time.time()
            rt = self._rt(climb)
            body = self.full_body if self.full_body is not None else climb.comp_body
            cls = enumerate_items(list(PRIMS) + list(r), climb.lib, [rt.st], self.device, max_body=body,
                                  deadline=deadline, cache=self._sig_cache, chunk=climb.chunk * climb.fuse)
            self._cls = (r, cls)
            climb.records.append(dict(task="full_vocabulary", level=climb.level, routines=len(r), classes=len(cls),
                                      max_body=body, cached=len(self._sig_cache), seconds=round(time.time() - t0, 1)))
        return self._cls[1]

    def mdl(self, climb, rec, n_symbols):
        raise NotImplementedError("TARGET=rows has no small target to look at (rules lp, lpca, all only)")
