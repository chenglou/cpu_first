"""Why did the replay's searches differ? The same stored jobs, four ways on one GPU host: the CPU engine; the GPU engine
with the vocabulary built here (equiv.py's path); the GPU engine with the vocabulary prepared in a worker process
(searcher.py's path); the same with the jobs shared (searcher.py with share). Each record's differing fields against
the stored one, with the level sizes.

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.debug_replay JOBS.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import sys


def main():
    from bar3.breadth import targets as TG
    from bar3.scale.heldout import targets as _ho  # noqa: F401
    from bar3.scale import jobs as J
    from bar3.gpu_search import engine as G
    from bar3.gpu_search.equiv import diff
    from bar3.gpu_search.searcher import Searcher
    rows = [json.loads(line) for line in gzip.open(sys.argv[1], "rt")]
    TG.use("tri")

    def show(way, recs):
        for (key, _, job, stored), r in zip(rows, recs):
            d = diff(stored, r) if isinstance(r, dict) else ["failed"]
            print(f"  {way:12s} {job['kind']:8s} {job['name'][:28]:28s} lib {len(job.get('lib', {})):2d} "
                  f"{'same' if not d else 'DIFF ' + str(d)} {r['level_sizes'][-3:] if isinstance(r, dict) else r!r} "
                  f"stored {stored['level_sizes'][-3:]}", flush=True)

    G.use(False)
    show("cpu", [J.search_job(j) for _, _, j, _ in rows])
    G.use(True)
    show("gpu here", [J.search_job(j) for _, _, j, _ in rows])
    s = Searcher(4, share=False)
    show("gpu worker", s([j for _, _, j, _ in rows]))
    s.close()
    s = Searcher(4, share=True)
    show("gpu shared", s([j for _, _, j, _ in rows]))
    s.close()


if __name__ == "__main__":
    main()
