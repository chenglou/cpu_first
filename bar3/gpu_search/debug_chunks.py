"""Where do the kernels disagree? Runs captured jobs on the GPU engine with every chunk computed by both the Triton
kernel and the CPU kernel (kernel._apply_chunk_cpu), and reports each chunk whose drop flags, kept rows or keys
differ: the item, the source state, both outputs. The search itself goes on with the CPU kernel's result.

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.debug_chunks JOBS.jsonl [KEY ...]
"""

from __future__ import annotations

import json
import sys

import numpy as np

from library import dsl
from bar3.gpu_search import kernel as K

FOUND = []


def checked(gc, codes, c0, S, lo, n, src, NP, cap, limit):
    og, kg, dg = (x.cpu().numpy().copy() for x in K.apply_chunk_triton(gc, codes, c0, S, lo, n, src, NP, cap, limit))
    oc, kc, dc = K._apply_chunk_cpu(codes, c0, S, lo, n, src, NP, cap, limit)
    o, k, d = oc.cpu().numpy(), kc.cpu().numpy(), dc.cpu().numpy()
    keep = ~d
    bad = np.nonzero((dg != d) | (keep & ((og != o).any(1) | (kg != k))))[0]
    if len(bad):
        i = int(bad[0])
        g = lo + i
        ci, s = c0 + g // S, g % S
        srow = src[s, :NP * 5].cpu().numpy().reshape(NP, 5)
        FOUND.append(dict(n_bad=len(bad), item=dsl.show((codes.classes[ci].rep,)), ci=ci, drop_gpu=bool(dg[i]),
                          drop_cpu=bool(d[i]), src=srow.tolist(), gpu=og[i, :NP * 5].reshape(NP, 5).tolist(),
                          cpu=o[i, :NP * 5].reshape(NP, 5).tolist(), cap=cap, limit=limit,
                          code=codes.flat[codes.off[ci]:codes.off[ci + 1]].tolist()))
        print("MISMATCH", json.dumps(FOUND[-1])[:3000], flush=True)
    return oc, kc, dc


def main():
    from bar3.breadth import targets as TG
    from bar3.scale.heldout import targets as _ho  # noqa: F401
    from bar3.scale import jobs as J
    from bar3.scale import engine as E
    from bar3.gpu_search import engine as G
    from bar3.gpu_search.equiv import diff
    K.apply_chunk_triton = K.apply_chunk
    K.apply_chunk = checked
    orig = E.Codes.__init__

    def init(self, classes, lib):           # the chunk check names the item: keep the classes on the codes
        orig(self, classes, lib)
        self.classes = classes

    E.Codes.__init__ = init
    G.use(True)
    keys = set(sys.argv[2:])
    for line in open(sys.argv[1]):
        key, src, job, stored = json.loads(line)
        if keys and key not in keys:
            continue
        if job.get("breadth_target"):
            TG.use(job["breadth_target"])
        FOUND.clear()
        rec = json.loads(json.dumps(J.search_job(job)))
        print(f"{job['kind']} {job['name']}: {len(FOUND)} mismatching chunks; record (CPU kernel) vs stored "
              f"{diff(stored, rec)}", flush=True)


if __name__ == "__main__":
    main()
