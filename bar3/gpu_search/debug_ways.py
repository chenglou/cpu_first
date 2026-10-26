"""One job, three ways on the GPU engine: the Triton kernel's outputs as they come (views of the reused chunk
buffers), the same outputs cloned, and the CPU kernel's; each record's differing fields against the stored one,
with their values.

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.debug_ways JOBS.jsonl.gz [KEY ...]
"""

from __future__ import annotations

import gzip
import json
import sys

from bar3.gpu_search import kernel as K


def main():
    from bar3.breadth import targets as TG
    from bar3.scale.heldout import targets as _ho  # noqa: F401
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    from bar3.gpu_search.equiv import diff
    G.use(True)
    triton = K.apply_chunk

    def cloned(*a):
        o, k, d = triton(*a)
        return o.clone(), k.clone(), d.clone()

    def cpu(gc, codes, c0, S, lo, n, src, NP, cap, limit):
        return K._apply_chunk_cpu(codes, c0, S, lo, n, src, NP, cap, limit)

    keys = set(sys.argv[2:])
    for line in gzip.open(sys.argv[1], "rt"):
        key, src, job, stored = json.loads(line)
        if keys and key not in keys:
            continue
        if job.get("breadth_target"):
            TG.use(job["breadth_target"])
        print(f"{job['kind']} {job['name']} {stored['level_sizes']}", flush=True)
        for name, fn in (("triton", triton), ("cloned", cloned), ("cpu", cpu)):
            K.apply_chunk = fn
            rec = json.loads(json.dumps(J.search_job(job)))
            d = diff(stored, rec)
            print(f"  {name}: {d}", flush=True)
            for f in d:
                print(f"    stored {f}: {json.dumps(stored.get(f))[:1500]}", flush=True)
                print(f"    {name} {f}: {json.dumps(rec.get(f))[:1500]}", flush=True)
        K.apply_chunk = triton


if __name__ == "__main__":
    main()
