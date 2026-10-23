"""The stage-2 A/B (INTEGRATION.md, "The A/B"): variants of libcore's climb (variant A is libcore.Climb itself, the c5
runs), each one process on one GPU, driven like modal_climb._climb.

  B, the simplified climb: A plus three domain rules and a stall rule.
    1. Routines first: a task is first searched over the solved routines and the straight-line primitives, with loops
       whose bodies are made of those ("compositions"); only if that fails, the library (merges and primitives), then
       the primitives alone (once per task, as A).
    2. A mirrored twin (task~) is not searched again once its original is solved and its own search used up its cap.
       Twins of unsolved concepts stay: BPE needs both spellings.
    3. The state cap is measured (profile_search.py on the H100: the device's memory per stored state and the seconds
       the next level takes per stored state), not a fixed 3M: `cap` in the config.
    4. A task whose searches used up their caps twice is logged as a stall: from then on only rule 1's composition
       search is run for it, and only when new routines appeared since its last one (a new routine can make a stalled
       task a short composition; the merges' and the primitives' searches are not repeated).
  C, B plus the two fixes the stalls point to:
    5. Every goal state is tested: the search lists one spelling per goal state (libcore lists the first 200 spellings,
       which can all belong to a few goal states), with the acceptance probes built once per search (libcore rebuilds
       them, helpers included, for every program); accepted goal states then give up to 8 spellings each for BPE.
    6. mac's addend is the running sum in A (A + x d) instead of N (x d + N): a designed practice task, like A's mac,
       whose program is the accumulating row (LDH.Y CLK P[MUL.X INC]) that mul's loop needs.
Everything else (the other practice tasks, BPE, Occam, routines, the round's search) is libcore's, unchanged.

As run (the records ab_B_exact_h100, ab_C_exact_h100), rule 2 here is symmetric: it would also skip an original whose
twin is solved. libcore's version (stage 3) skips twins only, since the originals are the other originals' helpers: an
in-run stage-3 climb with the symmetric rule skipped add when only add~ was solved and lost the whole original chain.
"""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager

import torch

from integration.lib import libcore as L


# ===================================================================================================== C's search fixes
class Acceptor:
    """A task's acceptance probes, built once per search (libcore.accept builds the same ones, from the same seed, for
    every program): a program passes if it reaches the goal at every verify setting."""

    def __init__(self, task, lib, helpers, rows, settings, device, n=60, seed=11):
        self.lib = lib
        self.probs = [(st, L.LawProblem(task, [st], n, seed, rows, helpers, lib, device)) for st in settings]
        self.cache = {}

    def __call__(self, prog):
        k = L.show(prog)
        if k not in self.cache:
            ok = True
            for st, prob in self.probs:
                b = prob.roots[0].copy()
                L.run_items(st, b, prog, self.lib)
                if not bool(prob.goal([b])[0]):
                    ok = False
                    break
            self.cache[k] = ok
        return self.cache[k]


_PROGRAMS = L.programs                     # libcore's own listing (every_goal_state swaps L.programs while it runs)


class _One:
    def __init__(self, res, g):
        self.parents, self.found_cost, self.goals = res.parents, res.found_cost, [g]


def spellings(res, g, classes, n):
    return _PROGRAMS(_One(res, g), classes, limit=n)


@contextmanager
def every_goal_state(log=None, max_goals=4000):
    """Inside: libcore's search lists one spelling per goal state (up to max_goals goal states) instead of the first
    200 spellings. Yields libcore's own listing and a map from each listed spelling to its goal state; `log` gets each
    listing's size."""
    orig = _PROGRAMS
    goal_of = {}

    def per_goal(res, classes, limit=200):
        out = []
        goal_of.clear()
        for g in res.goals[:max_goals]:
            ps = orig(_One(res, g), classes, limit=1)
            if ps:
                out.append(ps[0])
                goal_of[L.show(ps[0])] = g
        if log is not None:
            log.append({"cost": res.found_cost, "goal_states": len(res.goals), "listed": len(out)})
        return out
    L.programs = per_goal
    try:
        yield orig, goal_of
    finally:
        L.programs = orig


def more_spellings(res, accepted, classes, ok, orig_programs, goal_of, n=8):
    """For every accepted goal state, up to n spellings that pass (BPE counts pairs over a task's spellings)."""
    out, seen = [], set()
    for p in accepted:
        g = goal_of.get(L.show(p))
        for q in (orig_programs(_One(res, g), classes, limit=n) if g is not None else [p]):
            if L.show(q) not in seen and ok(q):
                seen.add(L.show(q))
                out.append(q)
    return out or accepted


# ===================================================================================================== C's mac
_build = L.build


def build_c(task, st, n, gen, rows, helpers, lib, device):
    """Variant C's practice tasks: libcore's, except mac, whose addend is the running sum in A: A + x d."""
    if task.rstrip("~") != "mac":
        return _build(task, st, n, gen, rows, helpers, lib, device)
    reg = (lambda r: L.SWAP[r]) if task.endswith("~") else (lambda r: r)
    P, B = st.P, st.B
    Nd, Xd = rows
    pick = torch.randint(0, Nd.shape[0], (4 * n,), generator=gen, device=device)
    Nt, xt = L.top_parts(Nd[pick], Xd[pick], L.width(task, st))
    xt, other = L.pad(xt[:n], P), L.pad(xt[n:2 * n], P)
    run = lambda name, ins: L.run_prog(st, helpers[name], lib, {reg(k): v for k, v in ins.items()}, gen, device)
    qdig = L.pad((torch.arange(n, device=device) % B)[:, None], P)
    R = run("mrow", {"X": xt, "Y": qdig})[0]
    return {reg("X"): xt, reg("Y"): qdig, "A": other}, {"A": run("add", {"X": R, "N": other})[0]}, None


# ===================================================================================================== B's climb
def mirror(task):
    return task[:-1] if task.endswith("~") else task + "~"


class ClimbB(L.Climb):
    """libcore's climb with B's rules (1-4), and C's search fix (5) when every_goal; see the module docstring."""

    def __init__(self, cells, device, P_round, comp_body=3, comp_cost=7, comp_budget=None, stall_after=2,
                 every_goal=False, search_s=1800.0, **kw):
        super().__init__(cells, device, P_round, **kw)
        self.search_s = search_s                  # one search's time limit (its stop counts as a used-up cap)
        self.comp_body, self.comp_cost = comp_body, comp_cost
        self.comp_budget = comp_budget or self.budget
        self.stall_after, self.every_goal = stall_after, every_goal
        self.used_up = Counter()                  # task -> searches that used up their caps without a program
        self.stalled, self.skipped = {}, {}       # task -> the level from which it is not searched
        self.events, self.t0, self.solved_s = [], time.time(), {}
        self.comp_seen = {}                       # task -> the routines its last composition search had
        self._comp = None

    def routines(self):
        return tuple(sorted(k for k in self.lib if k.startswith("T.")))

    def comp_classes(self):
        """Rule 1's vocabulary: the routines and the primitives, and loops whose bodies (cost <= comp_body) are made of
        those; the merges appear only inside the routines."""
        routines = self.routines()
        if self._comp is None or self._comp[0] != routines:
            self._comp = (routines, L.enumerate_items(list(L.PRIMS) + list(routines), self.lib, self.settings,
                                                      self.device, max_body=self.comp_body))
        return self._comp[1]

    def todo(self):
        out = []
        for t in super().todo():
            if t in self.skipped:
                continue
            if t not in self.stalled and mirror(t) in self.corpus and self.used_up[t] >= 1:                 # rule 2
                self.skipped[t] = self.level
                self.events.append(f"L{self.level} TWIN {t}: not searched again (its mirror {mirror(t)} is solved; "
                                   f"its own search used up its cap)")
                continue
            if t not in self.stalled and self.used_up[t] >= self.stall_after:                               # rule 4
                self.stalled[t] = self.level
                self.events.append(f"L{self.level} STALL {t}: its searches used up their caps {self.used_up[t]} "
                                   f"times (levels {[r['level'] for r in self.records if r.get('task') == t]}); "
                                   f"from now on only compositions, when new routines appear")
            if t in self.stalled and self.comp_seen.get(t) == self.routines():
                continue
            out.append(t)
        return out

    def step(self, rows, answers=None, deadline=None):
        if not self.events:
            line = super().step(rows, answers, deadline)
            if line is not None:
                self.events.append(line)
        return self.events.pop(0) if self.events else None

    def search(self, task, rows, deadline=None):
        t0 = time.time()
        climb_deadline, deadline = deadline, min(deadline or float("inf"), t0 + self.search_s)
        helpers = self.helpers_for(task)
        prob = L.LawProblem(task, self.settings, self.NP, 0, rows, helpers, self.lib, self.device)
        if self.every_goal:
            ok = Acceptor(task, self.lib, helpers, rows, self.verify, self.device)
        else:
            cache = {}

            def ok(p):
                k = L.show(p)
                if k not in cache:
                    cache[k] = L.accept(task, p, self.lib, helpers, rows, self.verify, self.device)
                return cache[k]
        runs = []

        def run(how, classes, lib, cap, max_cost):
            t1, glog = time.time(), []
            if self.every_goal:
                with every_goal_state(glog) as (orig, goal_of):
                    res = L.bfs(prob, classes, lib, max_cost=max_cost, max_states=cap,
                                accept=lambda ps: [ok(p) for p in ps], deadline=deadline, chunk=self.chunk)
                    if res.accepted:
                        res.accepted = more_spellings(res, res.accepted, classes, ok, orig, goal_of)
            else:
                res = L.bfs(prob, classes, lib, max_cost=max_cost, max_states=cap,
                            accept=lambda ps: [ok(p) for p in ps], deadline=deadline, chunk=self.chunk)
            runs.append(dict(how=how, found_cost=res.found_cost, level_sizes=res.level_sizes, stop=res.stop,
                             seconds=round(time.time() - t1, 1), listings=glog[-2:]))
            return res
        res, how = None, ""
        if self.routines():                                                                                 # rule 1
            res, how = run("routines", self.comp_classes(), self.lib, self.comp_budget, self.comp_cost), "routines"
            self.comp_seen[task] = self.routines()
        if (res is None or res.found_cost is None) and (task not in self.stalled or res is None):           # rule 4
            res, how = run("library", self.classes(), self.lib, self.budget, self.max_cost), "library"
        if res.found_cost is None and self.lib and task not in self.fallback_failed and task not in self.stalled:
            fb = run("primitives", self.classes(prims_only=True), {}, self.fallback_budget, self.max_cost)
            if fb.found_cost is not None:
                res, how = fb, "primitives"
            else:
                self.fallback_failed.add(task)
        out_of_time = climb_deadline is not None and time.time() > climb_deadline
        if res.found_cost is None and not out_of_time:
            self.used_up[task] += 1
        if res.found_cost is not None:
            self.solved_s[task] = round(time.time() - self.t0, 1)
        rec = dict(task=task, level=self.level, found_cost=res.found_cost, level_sizes=res.level_sizes,
                   seconds=round(time.time() - t0, 1), stop=res.stop, how=how, runs=runs,
                   programs=[L.show(p) for p in (res.accepted or [])][:8])
        return res.accepted or [], rec


def run(name, config, save):
    """One climb of variant B or C (config["variant"]), the record saved after every unit of work."""
    from integration.lib import inclimb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    variant = config.get("variant", "B")
    if variant == "C":
        L.build = build_c                                       # C's mac, in this process only
    lines = []

    def log(s):
        lines.append(f"{time.time() - t0:7.1f}s {s}")
        print(lines[-1], flush=True)
    rows = inclimb.load_rows(config.get("rows", 4000), seed=config.get("seed", 0))
    Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
    ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(config.get("labels", 400),
                                                                              seed=config.get("seed", 0) + 1, T=1)))
    cells = L.Cells(10, dev)
    c = ClimbB(cells, dev, 16, budget=config["cap"], fallback_budget=config.get("fallback_cap", 16_000_000),
               max_body=config.get("max_body", 3), max_cost=config.get("max_cost", 7), min_count=2.0, max_merges=12,
               levels=config.get("levels", 12), NP=8, round_budget=config.get("round_budget", 4_000_000),
               round_cost=config.get("round_cost", 5), chunk=config.get("chunk", 1 << 21),
               comp_body=config.get("comp_body", 3), stall_after=config.get("stall_after", 2),
               every_goal=variant == "C", search_s=config.get("search_s", 1800))
    deadline = t0 + config["seconds"]

    def record(status):
        return {"name": name, "variant": variant, "config": config, "device": dev, "status": status,
                "seconds": round(time.time() - t0, 1), "log": lines, "records": c.records, "history": c.history,
                "solved_at": c.solved_at, "solved_s": c.solved_s, "stalled": c.stalled, "skipped": c.skipped,
                "used_up": dict(c.used_up), "lib": {k: L.show(v) for k, v in c.lib.items()},
                "corpus": {t: [L.show(p) for p in ps] for t, ps in c.corpus.items()},
                "rounds": [L.show(p) for p in c.rounds],
                "rounds_primitive": [L.show(L.expand(p, c.lib)) for p in c.rounds]}
    log(f"variant {variant}: cap {config['cap']:,} (measured), fallback cap {config.get('fallback_cap', 16_000_000):,}, "
        f"one search at most {config.get('search_s', 1800)} s, the climb at most {config['seconds']} s")
    status = "running"
    while True:
        line = c.step((Nd, Xd), answers=ans, deadline=deadline)
        if line is None:
            status = "done" if not c.stalled else f"stalled: {sorted(c.stalled)}"
            break
        log(line)
        save(record(status))
        if time.time() > deadline:
            log("time up")
            status = "time up"
            break
        if c.rounds and config.get("stop_at_round", True):
            log(f"round found: {[L.show(p) for p in c.rounds]}")
            status = "round found"
            break
    rec = record(status)
    save(rec)
    return rec


# ===================================================================================================== libcore's switches
def run_core(name, config, save):
    """One climb of libcore.Climb itself with switches (config["switches"]: Climb's keyword arguments; config["mac"]:
    libcore.MAC_ADDEND; config["prereq"]: mul after mac, reduce after modB), as stage 3 runs it but alone: the time to
    the round on this device, without the evaluator."""
    from integration.lib import inclimb
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    L.MAC_ADDEND = config.get("mac", "N")
    if config.get("prereq"):
        L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
    lines = []

    def log(s):
        lines.append(f"{time.time() - t0:7.1f}s {s}")
        print(lines[-1], flush=True)
    rows = inclimb.load_rows(config.get("rows", 4000), seed=config.get("seed", 0))
    Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
    ans = tuple(t.to(dev) for t in inclimb.digits_of_rows(inclimb.load_rows(config.get("labels", 400),
                                                                              seed=config.get("seed", 0) + 1, T=1)))
    c = L.Climb(L.Cells(10, dev), dev, 16, **config.get("switches", {}))
    deadline = t0 + config["seconds"]
    status = "running"

    def record():
        return {"name": name, "variant": "core", "config": config, "device": dev, "status": status,
                "seconds": round(time.time() - t0, 1), "log": lines, "records": c.records, "history": c.history,
                "solved_at": c.solved_at, "stalled": c.stalled, "skipped": c.skipped, "used_up": dict(c.used_up),
                "lib": {k: L.show(v) for k, v in c.lib.items()},
                "corpus": {t: [L.show(p) for p in ps] for t, ps in c.corpus.items()},
                "rounds": [L.show(p) for p in c.rounds]}
    log(f"libcore.Climb, switches {config.get('switches')}, mac {L.MAC_ADDEND}, prereq {bool(config.get('prereq'))}")
    while True:
        line = c.step((Nd, Xd), answers=ans, deadline=deadline)
        if line is None:
            status = "done"
            break
        log(line)
        save(record())
        if time.time() > deadline:
            status = "time up"
            log("time up")
            break
        if c.rounds:
            status = "round found"
            log(f"round found: {[L.show(p) for p in c.rounds]}")
            break
    rec = record()
    save(rec)
    return rec
