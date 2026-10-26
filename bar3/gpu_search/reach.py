"""Reach against budget on the GPU (GPU_SEARCH.md §6): SCALE.md §3's searches again, the target alone (no practice)
with each state budget, posed exactly as bar3/scale/modal_scale.reach posed them (so their level sizes can be checked
against bar3/scale/records/reach_*.json), on the base vocabulary (x² mod N, `h1`) and over the 34-routine library
carried after 2x + 1 (10·x², `ten`).

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.reach OUT.jsonl BUDGET ...
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    out_path, budgets = sys.argv[1], [int(b) for b in sys.argv[2:]]
    from library.inventing import brm
    from bar3 import learner as L, problems as P
    from bar3.breadth import targets as TG
    from bar3.scale import learn
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    G.use(True)
    lib34 = json.loads(open(ROOT / "bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json").read())["state"]
    stored = {}
    for label, f in (("h1_base", "reach_h1_base.json"), ("ten_lib34", "reach_ten_lib34.json")):
        p = ROOT / "bar3/scale/records" / f
        if p.exists():
            stored[label] = {r["budget"]: r for r in json.load(open(p))}
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for label, tname, lib, kinds in (("h1_base", "h1", {}, list(brm.LOOP_KINDS)),
                                     ("ten_lib34", "ten", lib34["lib"], lib34["kinds"])):
        TG.use(tname)
        tgt = P.the_target(7)
        for b in budgets:
            j = learn.job(tgt, lib, kinds, dict(L.FULL, max_states=b, time_limit=5 * 3600, max_cost=10), "target", 7,
                          general=True)
            j["breadth_target"] = tname
            t = time.time()
            r = J.search_job(j)
            wall = time.time() - t
            s = stored.get(label, {}).get(b)
            rec = dict(label=label, budget=b, found=r["found_cost"], levels=r["level_sizes"], stop=r["stop"],
                       seconds=r["seconds"], wall=round(wall, 2), states=sum(r["level_sizes"]),
                       cpu_seconds=s and s["seconds"], same_levels=(s["levels"] == r["level_sizes"]) if s else None,
                       peak_gb=round(G.torch.cuda.max_memory_allocated() / 2 ** 30, 2) if G.DEVICE.type == "cuda" else None)
            print(json.dumps(rec), flush=True)
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
