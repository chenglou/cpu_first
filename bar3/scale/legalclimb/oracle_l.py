"""LEGALCLIMB.md §2: written by bar3/scale/legalclimb/make_copies5.py from bar3/scale/slice/oracle_s.py, 0 asserted replacements
(this study's, marked "legalclimb"). The rest of this docstring is the copy's.

SLICE.md §2: written by bar3/scale/slice/make_copies4.py from integration/self_practice/oracle.py, 5 + 0 asserted
replacements (rounds 2 and 3's, marked "compose2" and "compose3", then this study's, marked "slice"). The rest of
this docstring is the original's.

The tiny core's second arm (SELF_PRACTICE.md §2): the target itself at the practice widths, x'^2 mod N' on the rows'
top parts, so that bar 3's target-derived sources can be posed (bar3/problems.py, ported to the digit machine):
  * the target's restrictions: its rows where t1 < t2, for terms 1, 2, 3, X, N, X+X, N+N, X+N, X.X, X.N, N.N;
  * its projections: y's lowest digit, the rest (y div 10), [y = 0], [y < x], [y = x];
  * its own best wrong attempts (the target probe's near misses): every value a register holds after one of their
    top-level items (a problem of its own) and "the rest of the way" from it (y from X = that value, N = N').
The answers come from Python's integers: a study's oracle, as bar 3's numpy, never part of a training file (in a run
the target is the round at full size, with the data's answers). Nothing else here computes a number.
"""

from __future__ import annotations

import torch

import math
import time

from integration.lib.libcore import (A_, ONE, PRIMS, RI, _OneGoal, apply_items, enumerate_items, fresh, gen_on,
                                     pad, programs, run_items, show, top_parts)
from bar3.scale.legalclimb.spcore_l import Goal, _Res, bfs_multi, width_for   # compose2: the copy
from bar3.scale.legalclimb import hooks as H

TERMS = {
    "1": lambda X, N: 1, "2": lambda X, N: 2, "3": lambda X, N: 3, "X": lambda X, N: X, "N": lambda X, N: N,
    "X+X": lambda X, N: 2 * X, "N+N": lambda X, N: 2 * N, "X+N": lambda X, N: X + N, "X.X": lambda X, N: X * X,
    "X.N": lambda X, N: X * N, "N.N": lambda X, N: N * N,
}


def value(d):
    return [sum(int(v) * 10 ** i for i, v in enumerate(r)) for r in d.tolist()]


def digits(vals, P, device):
    out = torch.zeros(len(vals), P, dtype=torch.long, device=device)
    for i, v in enumerate(vals):
        for j in range(P):
            out[i, j] = v % 10
            v //= 10
    return out


class OracleDraw:
    """n rows' top parts at the target's width (P / 2: x'^2 fits the registers), filtered by a predicate on (x', N');
    x' < N' (the rows where x' = N' leave x'^2 mod N' = 0: kept, they are the data's)."""

    def __init__(self, st, n, rows, gen, device, pred=None):
        Nd, Xd = rows
        w = width_for("double", st)
        pick = torch.randint(0, Nd.shape[0], (64 * n,), generator=gen, device=device)
        Nt, xt = top_parts(Nd[pick], Xd[pick], w)
        xs, Ns = value(xt), value(Nt)
        keep = [i for i in range(len(xs)) if pred is None or TERMS[pred[0]](xs[i], Ns[i]) < TERMS[pred[1]](xs[i], Ns[i])]
        keep = keep[:n]
        self.n, self.P = len(keep), st.P
        self.xs, self.Ns = [xs[i] for i in keep], [Ns[i] for i in keep]
        self.x, self.N = digits(self.xs, st.P, device), digits(self.Ns, st.P, device)
        self.ys = [H.TARGET_F(x, N) for x, N in zip(self.xs, self.Ns)]      # compose3: TARGET_FN
        self.y = digits(self.ys, st.P, device)


def target_goal(d, kind="target"):
    """The target's answers, or a projection of them, as a goal's targets on a draw."""
    dev = d.x.device
    allm = torch.ones(d.n, dtype=torch.bool, device=dev)
    if kind == "target":
        return {"A": d.y, "Am": allm}
    if kind == "low":
        return {"A": digits([y % 10 for y in d.ys], d.P, dev), "Am": allm}
    if kind == "high":
        return {"A": digits([y // 10 for y in d.ys], d.P, dev), "Am": allm}
    if kind == "zero":
        return {"c": torch.tensor([int(y == 0) for y in d.ys], device=dev), "cm": allm}
    if kind == "ltx":
        return {"c": torch.tensor([int(y < x) for y, x in zip(d.ys, d.xs)], device=dev), "cm": allm}
    if kind == "eqx":
        return {"c": torch.tensor([int(y == x) for y, x in zip(d.ys, d.xs)], device=dev), "cm": allm}
    raise ValueError(kind)


def varies(t):
    if "A" in t:
        return not bool((t["A"] == t["A"][:1]).all())
    return not bool((t["c"] == t["c"][:1]).all())


def prefix_values(prog, st, root, lib):
    """Every register's digits after each top-level item of prog, run from root: {(item, reg): [n, P] long}."""
    b = root.copy()
    out = {}
    for i, it in enumerate(prog):
        run_items(st, b, (it,), lib)
        for r in ("A", "X", "Y", "S"):
            out[(i, r)] = b.regs[0, RI[r]].T.long().clone()
    return out


class OracleRoot:
    """One target-derived problem's probes at some settings: X = x' (or a near miss's value), N = N' (the round's
    placement), the goals by the oracle. kind: "target" (the target root: the projections and the near misses' values
    are goals on it), "restrict" (the target on the rows where pred holds), "rest" (y from X = the value register r
    holds after the near miss's items up to i: "the rest of the way")."""

    def __init__(self, key, kind, settings, NP, rows, device, seed, pred=None, near=None, lib=None):
        self.key, self.kind, self.pred, self.near = key, kind, pred, near
        self.binding, self.pattern = ("X", "N"), key.split("@")[0]
        gen = gen_on(device, seed + sum(map(ord, key)))
        self.draws, self.roots = [], []
        for st in settings:
            d = OracleDraw(st, NP, rows, gen, device, pred if kind == "restrict" else None)
            if d.n < 4:
                raise ValueError("too few rows")
            root = fresh(st, {"X": d.x, "N": d.N}, gen, d.n, device)
            if kind == "rest":                      # X holds the near miss's value
                prog, (i, r) = near
                v = prefix_values(prog, st, root, lib)[(i, r)]
                root = fresh(st, {"X": v, "N": d.N}, gen, d.n, device)
            self.draws.append(d)
            self.roots.append(root)

    def goals_for(self, names, values=None):
        """Goals on this root: the target or its projections (by name), and the near misses' values (as A)."""
        out = []
        for nm in names:
            ts = [target_goal(d, "target" if self.kind != "target" else nm) for d in self.draws]
            if all(varies(t) for t in ts):
                out.append(Goal(nm, ts))
        for nm, vs in (values or {}).items():
            dev = vs[0].device
            ts = [{"A": v, "Am": torch.ones(v.shape[0], dtype=torch.bool, device=dev)} for v in vs]
            if all(varies(t) for t in ts):
                out.append(Goal(nm, ts))
        return out


class OracleAcceptor:
    """A target-derived problem's acceptance: the same construction on fresh rows (60 per setting, seed 11) at the
    verify settings."""

    def __init__(self, root, name, verify, rows, device, lib, values=None):
        self.cache, self.probs = {}, []
        try:
            r = OracleRoot(root.key, root.kind, verify, 60, rows, device, 11, root.pred, root.near, lib)
        except ValueError:
            return
        vals = None
        if values and name in values:              # a value problem: recompute the value on the fresh rows
            prog, (i, rg) = values[name]
            vals = {name: [prefix_values(prog, st, rt, lib)[(i, rg)] for st, rt in zip(verify, r.roots)]}
        g = r.goals_for([] if vals else [name], vals)
        if g:
            for st, b, t in zip(verify, r.roots, g[0].targets):
                self.probs.append((st, b, Goal(name, [t])))

    def many(self, progs, lib):
        keys = [show(p) for p in progs]
        live, seen = [], set()
        for i, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                seen.add(k)
                live.append(i)
        if not self.probs:
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


class SmallTarget:
    """The hooks SelfClimb calls when the target is x'^2 mod N' at the practice widths (spcore.SelfClimb(target=...)):
    the target root, its acceptor, and the sources posed from it each level."""

    PROJECTIONS = ("low", "high", "zero", "ltx", "eqx")

    def __init__(self, sources=("restrict", "project", "value", "rest")):
        self.sources = sources
        self._root = None
        self.values = {}                            # value goal name -> (near-miss program, (item, register))
        self.subgoal, self.subgoals_seen = None, []                 # compose2: SUBGOAL
        self.lawnear = self.restrict_rows = False                   # compose2: LAWNEAR, RESTRICT_ROWS

    def root(self, climb, rows):
        if self._root is None:
            self._root = OracleRoot("tgt@XN", "target", climb.settings, climb.NP, rows, climb.device, climb.seed)
        return self._root

    def target_goal(self, climb, rows):
        return self.root(climb, rows).goals_for(["target"])[0]

    def acceptor(self, climb, rows, root, name):
        return OracleAcceptor(root, name, climb.verify, rows, climb.device, climb.lib, self.values)

    def pose(self, climb, rows, near):
        """The target-derived candidates of this level: (root, goals) pairs."""
        out = []
        tr = self.root(climb, rows)
        names = list(self.PROJECTIONS) if "project" in self.sources else []
        vals = {}
        if "value" in self.sources or "rest" in self.sources:
            for j, prog in enumerate(near):
                pv = [prefix_values(prog, st, rt, climb.lib) for st, rt in zip(climb.settings, tr.roots)]
                for (i, r) in pv[0]:
                    vs = [p[(i, r)] for p in pv]
                    ins = [torch.stack([d.x, d.N]) for d in tr.draws]
                    if any(bool((v == x[0]).all() or (v == x[1]).all()) for v, x in zip(vs, ins)):
                        continue
                    if any(bool((v == v[:1]).all()) for v in vs):
                        continue
                    nm = f"val.{len(self.values)}"
                    sig = tuple(tuple(v.flatten().tolist()) for v in vs)
                    if any(s == sig for s in vals.values()):
                        continue
                    vals[nm] = sig
                    self.values[nm] = (prog[:i + 1], (i, r))
        if names or vals:
            gl = tr.goals_for(names, {nm: [prefix_values(self.values[nm][0], st, rt, climb.lib)[self.values[nm][1]]
                                          for st, rt in zip(climb.settings, tr.roots)] for nm in vals}
                              if "value" in self.sources else None)
            out.append((tr, gl))
        if "rest" in self.sources:
            for nm in vals:
                prog, ir = self.values[nm]
                try:
                    r = OracleRoot(f"rest.{nm[4:]}@XN", "rest", climb.settings, climb.NP, rows, climb.device,
                                   climb.seed, near=(prog, ir), lib=climb.lib)
                except ValueError:
                    continue
                out.append((r, r.goals_for(["target"])))
        if "restrict" in self.sources:
            seen = set()
            for a in TERMS:
                for b in TERMS:
                    if a == b or (a in "123" and b in "123"):
                        continue
                    d0 = H.share_rows(self, climb, rows) if self.restrict_rows else tr.draws[-1]  # RESTRICT_ROWS
                    pat = tuple(int(TERMS[a](x, N) < TERMS[b](x, N)) for x, N in zip(d0.xs, d0.Ns))
                    share = sum(pat) / len(pat)
                    if share < 0.1 or share > 0.8 or pat in seen:
                        continue
                    seen.add(pat)
                    try:
                        r = OracleRoot(f"rst{len(seen)}@XN", "restrict", climb.settings, climb.NP, rows, climb.device,
                                       climb.seed, pred=(a, b))
                    except ValueError:
                        continue
                    out.append((r, r.goals_for(["target"])))
        return out

    def search(self, climb, rows, budget, deadline=None, extra=None, near=0):
        """The small target searched like a practice problem (the composition vocabulary), its acceptance at the
        verify settings; with near, the best near misses of its held levels (most probes right, then cheapest)."""
        t0 = time.time()
        lib = dict(climb.lib, **(extra or {}))
        root = self.root(climb, rows)
        goal = self.target_goal(climb, rows)
        if ("target", "tgt@XN") not in climb._accs:
            climb._accs[("target", "tgt@XN")] = self.acceptor(climb, rows, root, "target")
        acc = climb._accs[("target", "tgt@XN")]
        r0 = climb.routines()
        classes = (climb.comp_classes(deadline) if not extra else
                   enumerate_items(list(PRIMS) + [k for k in lib if k.startswith("T.")], lib, climb.settings,
                                   climb.device, max_body=climb.comp_body, deadline=deadline, cache=climb._sig_cache,
                                   chunk=climb.chunk * climb.fuse))
        fits = []
        res, sizes, stop, levels, parents = bfs_multi(root.roots, climb.settings, [goal], [acc.many], classes, lib,
                                                      max_cost=climb.max_cost, max_states=budget, deadline=deadline,
                                                      chunk=climb.chunk, fuse=climb.fuse, keep=True, fits=fits)
        out = []
        if near:
            cands = []
            for d, lv in enumerate(levels):
                if lv is None or d == 0:
                    continue
                f = goal.fits(lv)
                top = torch.argsort(-f, stable=True)[:50 * near].tolist()
                for i in top:
                    cands.append((int(f[i]), d, i))
            cands.sort(key=lambda c: (-c[0], c[1]))
            seen = set()
            for fit, d, i in cands:
                outs = tuple(levels[d][-1].regs[i, A_].flatten().tolist())
                if outs in seen:
                    continue
                seen.add(outs)
                pr = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
                if pr:
                    out.append((fit, d, pr[0]))
                if len(out) == near:
                    break
        if near and self.lawnear:                                   # compose2: LAWNEAR
            ln = H.law_near(self, climb, root, levels, parents, classes, goal)
            out += [(f, d, p) for f, d, p, _ in ln]
            climb.records.append(dict(task="lawnear", level=climb.level,
                                      near=[(f, d, show(p), r) for f, d, p, r in ln]))
        rec = dict(task="target", level=climb.level, found_cost=res[0]["found_cost"], level_sizes=sizes, fits=fits,
                   NP=sum(d.n for d in root.draws), seconds=round(time.time() - t0, 1), stop=stop,
                   programs=[show(p) for p in (res[0]["accepted"] or [])][:8],
                   near=[(f, d, show(p)) for f, d, p in out])
        return res[0]["accepted"] or [], rec, [p for _, _, p in out]

    def mdl(self, climb, rec, n_symbols):
        """bar 3's mdl: the best level's tokens, plus (which probe, its answer) for every probe it still gets wrong."""
        NP = rec["NP"]
        exc = math.log2(NP) + climb.settings[-1].P * math.log2(10)
        best = math.inf
        for d, f in enumerate(rec["fits"]):
            if f == NP and rec["found_cost"] != d:
                f = NP - 1
            best = min(best, d * math.log2(n_symbols) + (NP - f) * exc)
        return best
