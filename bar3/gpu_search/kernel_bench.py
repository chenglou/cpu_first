"""Kernel settings, measured (GPU_SEARCH.md §4): real searches under each setting of the interpreter (lanes per
program, warps, machine steps between checks of the loop condition), each record checked equal to the first setting's,
each timed on its second run (the first compiles). Workloads: x² mod N's target searched alone at 1M states over the
35 primitives (6 levels, the last streamed), and x³'s target probe over the 34-routine library carried after 2x + 1
at 1M states.

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.kernel_bench
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from library.inventing import brm
from bar3 import learner, problems as P
from bar3.breadth import targets as TG
from bar3.scale import jobs as J
from bar3.gpu_search import engine as G
from bar3.gpu_search import kernel as K

ROOT = Path(__file__).resolve().parents[2]
TIMING = ("seconds", "item_seconds", "wall")


def workloads():
    big = dict(learner.PROBE_B, max_states=1_000_000, time_limit=3600)
    TG.use("h1")
    out = [("primitives, x² mod N target, 1M", "h1",
            learner.job(P.the_target(7), {}, list(brm.LOOP_KINDS), big, "target_probe", 7))]
    state = json.loads(open(ROOT / "bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json").read())["state"]
    TG.use("cube")
    out.append(("34-routine library, x³ target, 1M", "cube",
                learner.job(P.the_target(7), state["lib"], state["kinds"], big, "target_probe", 7)))
    return out


def main():
    G.use(True)
    settings = list(itertools.product((32, 64, 128), (1, 2, 4), (1, 2, 4, 8)))
    settings = [(b, w, u) for b, w, u in settings if b // w >= 32 and b // w <= 64]
    for label, tname, job in workloads():
        TG.use(tname)
        ref = None
        print(label, flush=True)
        for b, w, u in settings:
            K.BLOCK_INTERP, K.WARPS_INTERP, K.UNROLL = b, w, u
            J.search_job(job)
            G.PROF = G.Prof()
            t = time.time()
            rec = J.search_job(job)
            wall = time.time() - t
            ph = G.PROF.t
            G.PROF = None
            rec = {k: v for k, v in rec.items() if k not in TIMING}
            ref = ref or rec
            print(f"  block {b:3d} warps {w} unroll {u}: search {wall:6.2f}s, kernel {ph.get('kernel', 0):6.2f}s  "
                  f"{'same' if rec == ref else 'DIFF'}  {rec['level_sizes']}", flush=True)


if __name__ == "__main__":
    main()
