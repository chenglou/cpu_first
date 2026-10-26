"""The replay test (GPU_SEARCH.md §5): one whole learner run from scratch, every search on the GPU engine (no store
read), its round records against the stored run's, record for record: every field of every round but the times (the
seconds, walls and elapsed times at any depth, the look entries' wall).

    cd cpu_first && <python with numba, torch, a GPU> -m bar3.gpu_search.replay LEARNER CONFIG TARGET TAG \
        STORED_RECORDS.jsonl OUT.jsonl [--engine=gpu|cpu] [--share]

LEARNER as in capture.py. --share runs a batch's jobs that share a search space as one search (jobs.search_jobs), as
the Modal runs did; their records are the same either way.
"""

from __future__ import annotations

import json
import sys
import time

TIMING = {"seconds", "wall", "item_seconds", "elapsed", "launch", "shared", "cached", "engine"}


def strip(x, key=None):
    """x without its times: timing keys at any depth, and each look entry's wall (index 5)."""
    if isinstance(x, dict):
        return {k: strip(v, k) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        if key == "looks":
            return [[strip(v) for i, v in enumerate(e) if i != 5] for e in x]
        return [strip(v) for v in x]
    return x


def compare(a, b):
    """The fields of two round records that differ, times aside."""
    a, b = strip(a), strip(b)
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in sys.argv[1:] if a.startswith("--"))
    spec, config, tname, tag, stored_path, out_path = args
    from bar3.breadth import targets as TG
    from bar3.gpu_search.capture import learner_of
    from bar3.gpu_search import engine as G
    from bar3.scale import jobs as J
    learn, search_job, TapClass, cfg, digits = learner_of(spec, config)
    G.use(opt.get("engine", "gpu") == "gpu")
    share = bool(opt.get("share"))
    name = f"{config}_{tname}_{tag}"
    stats = dict(jobs=0, searches=0, search_s=0.0)
    pool = None
    if int(opt.get("workers", 0)) and opt.get("engine", "gpu") == "gpu":
        from bar3.gpu_search.searcher import Searcher
        pool = Searcher(int(opt["workers"]), share=share)       # vocabularies prepared on CPU processes

    def one(js):
        try:
            return J.search_jobs(js) if len(js) > 1 else [search_job(js[0])]
        except Exception as e:                  # noqa: BLE001  (as pool.Searcher: the learner logs a failed job)
            print(f"  job failed: {js[0]['kind']} {js[0]['name']}: {e!r}", flush=True)
            return [e] * len(js)

    def search_many(jobs):
        t = time.time()
        if pool is not None:
            out = pool(jobs)
            stats["jobs"] += len(jobs)
            stats["search_s"] += time.time() - t
            return out
        out = [None] * len(jobs)
        groups = {}
        for i, j in enumerate(jobs):
            g = J.same_search(j) if share and not j.get("near") else i
            groups.setdefault(g, []).append(i)
        for idx in groups.values():
            for i, r in zip(idx, one([jobs[i] for i in idx])):
                out[i] = json.loads(json.dumps(r)) if isinstance(r, dict) else r
        stats["jobs"] += len(jobs)
        stats["searches"] += len(groups)
        stats["search_s"] += time.time() - t
        return out

    rounds = []
    t0 = time.time()

    def emit(rec):
        line = json.dumps(dict(rec, elapsed=round(time.time() - t0, 1)))
        rounds.append(json.loads(line))         # as stored: JSON has lists where the record has tuples
        with open(out_path, "a") as fh:
            fh.write(line + "\n")

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    open(out_path, "w").close()
    tap = TapClass(search_many)
    TG.use(tname)
    tap.target = tname
    if digits:
        cfg = dict(cfg, target=tname)
    learn.run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=print)
    wall = time.time() - t0
    if pool is not None:
        stats["searches"] = pool.stats["searches"]
        print(f"  searcher: {dict(pool.stats)}", flush=True)
        pool.close()
    stored = [json.loads(line) for line in open(stored_path)]
    print(f"{name} on {G.DEVICE if opt.get('engine', 'gpu') == 'gpu' else 'cpu'}: {len(rounds)} rounds in {wall:.0f}s, "
          f"{stats['jobs']} jobs in {stats['searches']} searches, {stats['search_s']:.0f}s searching", flush=True)
    bad = 0
    for i in range(max(len(rounds), len(stored))):
        if i >= len(rounds) or i >= len(stored):
            print(f"  round {i}: only in {'the stored run' if i >= len(rounds) else 'the replay'}")
            bad += 1
            continue
        d = compare(stored[i], rounds[i])
        bad += bool(d)
        print(f"  round {i}: {'same' if not d else 'DIFF ' + str(d)} (picked {rounds[i]['picked']}, "
              f"general {(rounds[i].get('general') or {}).get('program')})", flush=True)
    print("PASS: record for record" if not bad else f"FAIL: {bad} rounds differ", flush=True)


if __name__ == "__main__":
    main()
