"""Could jobs.classes_for's cache key have changed a record? (GPU_SEARCH.md §9.) The key sorts the library
(json.dumps of sorted(lib.items())), but the item enumeration reads the library in its order (the tokens' order picks
each class's representative and the classes' order, and so the order in which the search applies items, which
programs it reads back first, and which 200 it returns). So two jobs whose libraries hold the same routines in
different orders share one cache entry: in a worker that has run the first, the second gets the first's vocabulary.
jobs.same_search has the same blind spot (it dumps the library with sort_keys), so a call's two such jobs would also
run as one search on the first one's vocabulary.

Step 1 (`orders`, no search): over captured jobs (capture.py: every job of a run, recovered by replaying its learner
from its stores), group the jobs by the cache key and list every key held in more than one library order. A run's
store serves one Modal container (a group of runs as threads, their searches on one worker pool), so a collision
needs two orders within one container; across all the files given at once is the conservative test (any two runs).

Step 2 (`vocab`, item enumeration only): for each key with more than one order, the vocabulary of each order, compared
(classes, representatives, members, costs, bytecode). Orders with the same vocabulary cannot change a record.

Step 3 (`rerun`, searches): each job of a key whose orders' vocabularies differ, searched with its own order's
vocabulary (no cache) and with every other order's, against its stored record.

For runs whose jobs were not captured, two cheaper checks on the round records alone:
- `prefix CAPTURED.jsonl.gz ...` (with each capture's .rounds.jsonl beside it): is every job's library a library some
  round record holds (or none) with routines appended after it? Then any library a job used has the order of a round's
  library, so two of them with the same routines in different orders imply two such round libraries.
- `rounds RECORDS.jsonl[.gz] ...`: over all the round records given (the conservative test: any two runs), every
  library held in more than one order.

    cd cpu_first && <python with numba> -m bar3.gpu_search.order_check orders JOBS.jsonl.gz ... > OUT.json
    cd cpu_first && <python with numba> -m bar3.gpu_search.order_check vocab|rerun OUT.json JOBS.jsonl.gz ...
    cd cpu_first && python3 -m bar3.gpu_search.order_check prefix|rounds FILE ...
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import sys

from library import dsl


def cache_key(job):
    """jobs.classes_for's key for a job (its library as program text, sorted by token; the rest as classes_for gets
    it), hashed."""
    from library.inventing import brm
    lib = job.get("lib", {})
    key = (json.dumps({t: dsl.show(dsl.parse(v)) for t, v in sorted(lib.items())}),
           tuple(job.get("kinds", brm.LOOP_KINDS)), job["k"], job.get("max_body", 2),
           json.dumps(job.get("costs"), sort_keys=True) if job.get("costs") else "")
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()


def read(paths):
    for p in paths:
        with gzip.open(p, "rt") as fh:
            for line in fh:
                key, src, job, rec = json.loads(line)
                yield p, key, job, rec


def orders(paths):
    """{cache key: {library order (tokens, joined): [(file, job key), ...]}} for the keys held in several orders, and
    counts."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    n = 0
    for p, key, job, _ in read(paths):
        n += 1
        by[cache_key(job)][" ".join(job.get("lib", {}))].append((p.rsplit("/", 1)[-1], key))
    multi = {k: dict(v) for k, v in by.items() if len(v) > 1}
    return dict(jobs=n, keys=len(by), multi=multi)


def vocabulary(job, lib_items):
    """The vocabulary jobs.classes_for builds for this job with the library in the given order, uncached."""
    from library.inventing import brm
    from bar3.scale import jobs as J
    J._CLASSES.clear()
    lib = {t: dsl.parse(v) for t, v in lib_items}
    out = J.classes_for(lib, list(job.get("kinds", brm.LOOP_KINDS)), job["k"], job.get("max_body", 2),
                        job.get("costs"))
    J._CLASSES.clear()
    return out


def same_vocab(a, b):
    (ca, na, xa), (cb, nb, xb) = a, b
    return (na == nb and len(ca) == len(cb) and all(p.rep == q.rep and p.cost == q.cost and p.members == q.members
                                                    for p, q in zip(ca, cb))
            and (xa.flat == xb.flat).all() and (xa.off == xb.off).all())


def records(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def prefix(paths):
    """Per capture: the jobs whose library is not a round's library (or none) followed by appended routines."""
    bad_runs = 0
    for p in paths:
        libs = [[]] + [list(r["lib"].items()) for r in records(p.replace(".jsonl.gz", ".rounds.jsonl"))
                       if "round" in r and isinstance(r.get("lib"), dict)]
        n = bad = 0
        for _, _, job, _ in read([p]):
            items = list(job.get("lib", {}).items())
            n += 1
            if not any(items[:len(lb)] == lb for lb in libs):
                bad += 1
                if bad <= 3:
                    print(f"  {p.rsplit('/', 1)[-1]}: {job['kind']} {job['name']}: library "
                          f"{[t for t, _ in items][:12]} is no round library's extension", flush=True)
        bad_runs += bool(bad)
        print(f"{p.rsplit('/', 1)[-1]}: {n} jobs, {bad} not a round library with routines appended", flush=True)
    print(f"{len(paths)} runs, {bad_runs} with such jobs", flush=True)


def rounds(paths):
    """Over every round record given: the libraries (token -> program) held in more than one order."""
    by = collections.defaultdict(lambda: collections.defaultdict(set))
    n = 0
    for p in paths:
        for r in records(p):
            lib = r.get("lib") if "round" in r else None
            if not isinstance(lib, dict) or not lib:
                continue
            n += 1
            key = hashlib.sha1(json.dumps(sorted(lib.items())).encode()).hexdigest()
            by[key][" ".join(lib)].add(p.rsplit("/", 1)[-1])
    multi = {k: {o: sorted(f) for o, f in v.items()} for k, v in by.items() if len(v) > 1}
    for k, v in multi.items():
        print(f"  library {k[:12]} in {len(v)} orders: " + "; ".join(f"{o[:80]!r} in {f[:3]}" for o, f in v.items()))
    print(f"{len(paths)} record files, {n} round libraries, {len(by)} distinct, {len(multi)} held in more than one "
          f"order", flush=True)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "prefix":
        return prefix(args)
    if mode == "rounds":
        return rounds(args)
    if args and args[0] == "--digits":
        from bar3.scale.digits import jobs  # noqa: F401  (the D.B block and the digits jobs)
        args = args[1:]
    from bar3.scale.heldout import targets  # noqa: F401  (the held-out targets)
    if mode == "orders":
        out = orders(args)
        print(json.dumps(out))
        print(f"{out['jobs']} jobs, {out['keys']} cache keys, {len(out['multi'])} held in more than one library order",
              file=sys.stderr)
        return
    found = json.load(open(args[0]))["multi"]
    wanted = {jk for v in found.values() for o in v.values() for _, jk in o}
    jobs = {key: (job, rec) for _, key, job, rec in read(args[1:]) if key in wanted}
    for ck, by_order in found.items():
        reps = {o: jobs[ks[0][1]][0] for o, ks in by_order.items() if ks[0][1] in jobs}
        vocs = {o: vocabulary(j, list(j["lib"].items())) for o, j in reps.items()}
        first = next(iter(vocs.values()))
        same = all(same_vocab(first, v) for v in vocs.values())
        print(f"key {ck[:12]}: {len(by_order)} orders, {sum(len(v) for v in by_order.values())} jobs; vocabularies "
              f"{'the same' if same else 'DIFFER'}", flush=True)
        if mode != "rerun" or same:
            continue
        from bar3.gpu_search.equiv import diff
        from bar3.scale import jobs as J
        for o, ks in by_order.items():
            for _, jk in ks:
                job, stored = jobs[jk]
                for o2, voc in vocs.items():
                    J._CLASSES.clear()
                    J._CLASSES[_classes_key(job)] = voc
                    rec = J.search_job(job)
                    d = diff(stored, rec)
                    print(f"  job {jk[:12]} ({job['kind']} {job['name']}), its order {'(own)' if o2 == o else ''}"
                          f" vocabulary of order {o2[:40]!r}: {'same as stored' if not d else d}", flush=True)
                J._CLASSES.clear()


def _classes_key(job):
    """classes_for's own (unhashed) key, to put a chosen vocabulary in its cache."""
    from library.inventing import brm
    lib = {t: dsl.parse(v) for t, v in job.get("lib", {}).items()}
    return (json.dumps({t: dsl.show(v) for t, v in sorted(lib.items())}), tuple(job.get("kinds", brm.LOOP_KINDS)),
            job["k"], job.get("max_body", 2),
            json.dumps(job.get("costs"), sort_keys=True) if job.get("costs") else "")


if __name__ == "__main__":
    main()
