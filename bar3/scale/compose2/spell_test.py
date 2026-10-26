"""The test of the guess (PREREG.md): why the climb's search accepts nothing when handed the rest of the way from x²,
though `CLA P[BRG.X B[T.red.pair.AN]]` (cost 5) is right and accepted. The committed search (spcore.bfs_roots), on
round 1's route setup (bar3/scale/compose/route.py): A2c's final library, the rest from x'² on the arms' draws (CUDA),
the primitives with T.red.pair.AN and T.lt.pair.AN, 5.8M states. Measured:
  - the hand-written reductions' final machine states on the probes, keyed as bfs_roots keys them;
  - every held level's states: where (if anywhere) each reduction's state is, and that state's spellings
    (libcore.programs), the first of which is the one bfs_roots tests (limit=1);
  - every program the acceptor was handed, and whether a reduction was among them;
  - each level's goal states, against the 50,000 a level that get a spelling.

    run on Modal: modal_compose2.py --arms spell_test (REC: round 1's cp_bar3_minus_red_clean.json, in the image);
    spell_fix: the same on round 2's copy (COPY=1) with SPELL=16 (PREREG_ADDENDUM.md)
"""

from __future__ import annotations

import json
import os
import time

import torch

from integration.lib import libcore as L
from integration.lib.libcore import _OneGoal, keys_of, programs, run_items, show
if os.environ.get("COPY") == "1":              # the fix's check: round 2's copy, with SPELL from the env
    from bar3.scale.compose2 import spcore_c2 as S
    from bar3.scale.compose2.oracle_c2 import OracleAcceptor, OracleRoot, SmallTarget
else:                                           # the test: the committed search
    from integration.self_practice import spcore as S
    from integration.self_practice.oracle import OracleAcceptor, OracleRoot, SmallTarget
from bar3.scale.compose import climb as CC
from bar3.scale.compose.core_compose import digits, load_rows

REDUCTIONS = ("CLA P[BRG.X B[T.red.pair.AN]]", "LDA.N P[BRG.X B[T.red.pair.AN]]")
SQUARE = "LDA.X STA.Y T.mul.x,o.XY"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    budget = int(os.environ.get("MAX_STATES", 5_800_000))
    rows = load_rows(4096, seed=0)
    R = (digits(rows, "modulus").to(dev), digits(rows, "x").to(dev))
    target = SmallTarget(sources=("restrict",))
    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, target=target,
                        rule="lpca", NP=8, fuse=4 if dev == "cuda" else 1)
    rec = json.load(open(os.environ["REC"]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    c.lib = lib
    print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "budget", budget, flush=True)
    sq = L.parse(SQUARE)
    rr = OracleRoot("rest.sq@XN", "rest", c.settings, c.NP, R, dev, c.seed, near=(sq, (len(sq) - 1, "A")), lib=lib)
    acc = OracleAcceptor(rr, "target", c.verify, R, dev, lib)
    goal = rr.goals_for(["target"])[0]
    if dev not in S._SALT_ON:
        S._SALT_ON[dev] = S._SALT.to(dev)
    salt = S._SALT_ON[dev][:1]
    want = {}
    for s in REDUCTIONS:
        p = L.parse(s)
        outs = []
        for st, rt in zip(c.settings, rr.roots):
            b = rt.copy()
            run_items(st, b, p, lib)
            outs.append(b)
        k = int((keys_of(outs) ^ salt[0])[0])
        ok = acc.many([p], lib)[0]
        print(f"1. {s} (cost {L.cost(p)}): goal on the probes {bool(goal.test(outs, lib)[0])}, accepted {ok}, "
              f"state key {k}", flush=True)
        want[s] = k
    if os.environ.get("FULL") == "1":          # the composition vocabulary, all of the library (the probes')
        focus = "every routine (the composition vocabulary)"
        classes = c.comp_classes()
    else:
        focus = sorted(S.routines_in(("T.red.pair.AN",), lib, closure=True))
        classes = L.enumerate_items(list(L.PRIMS) + focus, lib, c.settings, dev, max_body=c.comp_body,
                                    cache=c._sig_cache, chunk=c.chunk * c.fuse)
    tested = []

    def many(progs, lb):
        tested.append([show(p) for p in progs])
        return acc.many(progs, lb)
    t0 = time.time()
    out, sizes, stops, levels, rids, parents = S.bfs_roots([rr.roots], c.settings, [(0, goal)], [many], classes, lib,
                                                           max_cost=c.max_cost, max_states=budget, chunk=c.chunk,
                                                           fuse=c.fuse)
    print(f"2. the search over the primitives and {focus}: found {out[0]['found_cost']}, sizes {sizes[0]}, "
          f"{stops[0]}, {time.time() - t0:.0f} s; aliases {[(d, a[:3]) for d, a in out[0]['aliases']]}", flush=True)
    flat = [p for batch in tested for p in batch]
    print(f"3. the acceptor was handed {len(flat)} programs in {len(tested)} calls; "
          + "; ".join(f"{s} among them: {s in flat}" for s in REDUCTIONS), flush=True)
    for d, lv in enumerate(levels):
        if lv is None or d == 0 or lv[0].S == 0:
            continue
        keys = keys_of(lv) ^ salt[rids[d]]
        n_goal = int(goal.test(lv, lib).sum())
        print(f"4. level {d}: {lv[0].S} held states, {n_goal} of them goal states (a spelling for the first 50,000)",
              flush=True)
        for s, k in want.items():
            hit = torch.nonzero(keys == k).flatten().tolist()
            if not hit:
                continue
            i = hit[0]
            sp = programs(_OneGoal(S._Res(parents, d, [i]), i), classes, limit=16)
            first = show(sp[0]) if sp else None
            okk = acc.many(sp, lib) if sp else []
            print(f"5. {s}'s state is held at level {d} (index {i}): {len(sp)} spellings, the first (the one tested) "
                  f"{first!r}; the reduction among them: {s in [show(p) for p in sp]}; accepted: "
                  f"{[show(p) for p, o in zip(sp, okk) if o][:4]}; the first tested: {first in flat}", flush=True)
    print("EXIT spell_test", flush=True)


if __name__ == "__main__":
    main()
