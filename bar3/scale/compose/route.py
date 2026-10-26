"""The open route, checked where the arms ran (COMPOSE.md §5.5). The oracle arm's draws come from generators on the
device, so only a run on the same GPU sees the arms' probes (a CPU rebuild, restrictions.py's, draws others). On the
target's own root (the climb's rows, settings, seed and key; CUDA):
  1. the target's probes, and those with x'^2 < N';
  2. the restrictions the source poses (restrictions.py's report, on these draws);
  3. with an arm's library (its record, REC), the fit of x'^2 (`LDA.X STA.Y T.mul.x,o.XY`) against the near misses'
     its record logged at its last level;
  4. the rest of the way from x'^2 (X = x'^2, N = N'): the reductions written by hand, right on how many of its probes
     and accepted at width 3 or not (oracle.OracleAcceptor); and a search of it over the composition vocabulary at
     the deep probe's 5.8M states (spcore.bfs_multi, as SmallTarget.search runs one);
  5. with FOCUS ("|"-separated sets of ","-separated routines), the same search over the primitives and each set's
     routines with the ones they call: the deep probe's focused vocabulary (spcore.SelfClimb.probe, deep_focus
     "closure") at its most favourable, the routines the hand-written reduction uses. SKIP_FULL=1 skips step 4's
     search.

    run on Modal: modal_compose.py --arms route  (REC: the Volume's runs/cp_bar3_minus_red_clean.json)
"""

from __future__ import annotations

import json
import os
import time

import torch

from integration.lib import libcore as L
from integration.lib.libcore import A_, run_items, show
from integration.self_practice.oracle import OracleAcceptor, OracleRoot, SmallTarget
from integration.self_practice.spcore import bfs_multi, routines_in
from bar3.scale.compose import climb as CC
from bar3.scale.compose.core_compose import digits, load_rows

SQUARES = ("LDA.X STA.Y T.mul.x,o.XY", "LDA.X STA.Y T.mul.x,o.YX")
REDUCTIONS = ("CLA P[BRG.X B[T.red.pair.AN]]", "LDA.N P[BRG.X B[T.red.pair.AN]]",
              "LDA.N STA.Y P[BRG.X B[T.red.pair.AY]]", "STA.Y P[BRG.X B[T.red.pair.AY]]", "P[BRG.X B[T.red.pair.AN]]")


def fit(prog, root, lib, settings):
    """Probes (all settings) where A after prog is the root's target."""
    n = 0
    for st, rt, d in zip(settings, root.roots, root.draws):
        b = rt.copy()
        run_items(st, b, prog, lib)
        n += int((b.regs[0, A_].T.long() == d.y).all(1).sum())
    return n


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(4096, seed=0)
    R = (digits(rows, "modulus").to(dev), digits(rows, "x").to(dev))
    target = SmallTarget(sources=("restrict",))
    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, target=target,
                        rule="lpca", NP=8, fuse=4 if dev == "cuda" else 1)
    print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", flush=True)
    tr = target.root(c, R)
    for st, d in zip(c.settings, tr.draws):
        print(f"1. the target's root, setting {st}: {d.n} probes, x'^2 < N' on "
              f"{sum(x * x < N for x, N in zip(d.xs, d.Ns))}; x' {d.xs} N' {d.Ns} y {d.ys}", flush=True)
    for r, _ in target.pose(c, R, []):
        if r.kind == "restrict":
            print(f"2. {r.key}: {r.pred[0]} < {r.pred[1]}; " + "; ".join(
                f"setting {st}: {d.n} probes, y = x'^2 on {sum(x * x < N for x, N in zip(d.xs, d.Ns))}, "
                f"y = x' on {sum(y == x for x, y in zip(d.xs, d.ys))}" for st, d in zip(c.settings, r.draws)),
                flush=True)
    rec = json.load(open(os.environ["REC"]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    c.lib = lib
    near = rec["near"][-1]
    print(f"3. {rec['tag']}: its last logged near misses (level {near['level']}): fits {[f for f, _, _ in near['near']]}",
          flush=True)
    for f, d, p in near["near"][:4]:
        print(f"   logged {f}, here {fit(L.parse(p), tr, lib, c.settings)}: {p}", flush=True)
    for s in SQUARES:
        print(f"   x'^2 {s}: fits {fit(L.parse(s), tr, lib, c.settings)} of {sum(d.n for d in tr.draws)}", flush=True)
    sq = L.parse(SQUARES[0])
    rr = OracleRoot("rest.sq@XN", "rest", c.settings, c.NP, R, dev, c.seed, near=(sq, (len(sq) - 1, "A")), lib=lib)
    acc = OracleAcceptor(rr, "target", c.verify, R, dev, lib)
    progs = [L.parse(s) for s in REDUCTIONS]
    ok = acc.many(progs, lib)
    for s, p, a in zip(REDUCTIONS, progs, ok):
        print(f"4. the rest from x'^2, {s} (cost {L.cost(p)}): right on {fit(p, rr, lib, c.settings)} of "
              f"{sum(d.n for d in rr.draws)} probes, accepted at width 3: {a}", flush=True)
    goal = rr.goals_for(["target"])[0]
    budget = int(os.environ.get("MAX_STATES", 5_800_000))
    for fs in [f for f in os.environ.get("FOCUS", "").split("|") if f]:
        focus = sorted(routines_in(tuple(fs.split(",")), lib, closure=True))
        classes = L.enumerate_items(list(L.PRIMS) + focus, lib, c.settings, dev, max_body=c.comp_body,
                                    cache=c._sig_cache, chunk=c.chunk * c.fuse)
        t0, fits = time.time(), []
        res, sizes, stop = bfs_multi(rr.roots, c.settings, [goal], [acc.many], classes, lib, max_cost=c.max_cost,
                                     max_states=budget, chunk=c.chunk, fuse=c.fuse, fits=fits)
        print(f"5. the rest from x'^2, over the primitives and {focus} at {budget:,} states: cost "
              f"{res[0]['found_cost']}, levels {sizes}, fits {fits}, {stop}, {time.time() - t0:.0f} s, programs "
              f"{[show(p) for p in (res[0]['accepted'] or [])][:4]}", flush=True)
    if os.environ.get("SKIP_FULL") == "1":
        return
    t0 = time.time()
    fits = []
    res, sizes, stop = bfs_multi(rr.roots, c.settings, [goal], [acc.many], c.comp_classes(), lib,
                                                  max_cost=c.max_cost, max_states=int(os.environ.get("MAX_STATES", 5_800_000)),
                                                  chunk=c.chunk,
                                                  fuse=c.fuse, fits=fits)
    print(f"4. the rest from x'^2, searched at {int(os.environ.get('MAX_STATES', 5_800_000)):,} states: cost {res[0]['found_cost']}, levels {sizes}, fits {fits}, "
          f"{stop}, {time.time() - t0:.0f} s, programs {[show(p) for p in (res[0]['accepted'] or [])][:4]}", flush=True)


if __name__ == "__main__":
    main()
