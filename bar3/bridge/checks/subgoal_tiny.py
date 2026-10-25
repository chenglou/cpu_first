"""The tiny check of CARRY.md §12's score, local, seconds: each target's round-0 pool rebuilt exactly as dlearner.run
builds it with the conjunctions and the widening on (the near misses from BRIDGE.md's round-0 records), scored by the
loop view (dsubgoal.py): which candidates the target factors through a loop of, and the subgoal.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.subgoal_tiny
"""

from __future__ import annotations

import time

from bar3.bridge import dlearner as L, dproblems as P, dsubgoal as G, dwiden as W, report as R


def pool_of(tsl, posed, near, lib=None):
    tgt = P.from_slice(tsl, "target")
    laws = [P.from_slice(s, "slice") for s in posed if s.name != tsl.name]
    pool = (P.digit_laws(tgt) + laws + P.restrictions(tgt) + P.projections(tgt)
            + P.from_near_misses(near, lib or {}, tgt))
    pool += W.conjunctions(P.restrictions(tgt), pool)
    pool += W.widenings(pool)
    return tgt, [c for c in P.dedupe(pool) if c.key() != tgt.key()]


def main():
    posed, _, _ = L.the_book()
    by = {s.name: s for s in posed}
    for t, sname in L.TARGETS.items():
        rec = R.load(f"all_{t}_r1")[0]
        near = [(f, d, p) for f, d, p in rec["target_probe"].get("near", []) if p]
        tgt, pool = pool_of(by[sname], posed, near)
        t0 = time.time()
        sc, sub = G.subgoal_of(pool, tgt)
        print(f"{t}: pool {len(pool)}, widened {sum(1 for c in pool if getattr(c, 'pins', None))}; "
              f"{len(sc)} with a loop ({time.time() - t0:.1f} s); subgoal: {sub}")
        for name, v in sc.items():
            print(f"    {name}: {v['loop']}; rows {v['rows']}; ignores {v['drop']}; trail {v['sig'] % 10**6}")


if __name__ == "__main__":
    main()
