"""Reruns against the stored runs (GPU_SEARCH.md §8), two ways.

`rounds NEW STORED [NEW STORED ...]`: a rerun's round records against the stored run's, round for round, every field
but the times (replay.compare: the seconds, walls and elapsed times at any depth, the look entries' wall).

`searches NEW_STORE --old OLD_STORE ...`: every search the rerun made (its store: one [key, record] a line) against
the record the earlier stores hold under the same key (pool.py's key, the sha1 of the job's JSON), every field but the
times (equiv.diff). A stored search that stopped on its time limit is counted apart: the CPU's clock cut it, and the
GPU gets further in the same time.

`pins RUN --old OLD_STORE ... --out PINS.jsonl`: the stored records of the searches the CPU cut by its clock, as a
store (one [key, record] a line), for the jobs of RUN (a rerun's store, or the run's captured jobs from capture.py,
which can also serve as OLD). A rerun started from it (modal_learn.py --store) reads those back and runs every other
search, so that its round records can be compared with the stored run's record for record: a search cut by the clock
ends where the CPU's clock put it, which no rerun on any engine can repeat.

    cd cpu_first && python3 -m bar3.gpu_search.compare_runs rounds NEW.jsonl STORED.jsonl
    cd cpu_first && python3 -m bar3.gpu_search.compare_runs searches NEW_STORE.jsonl --old OLD_STORE.jsonl ...
    cd cpu_first && python3 -m bar3.gpu_search.compare_runs pins CAPTURED.jsonl.gz --old CAPTURED.jsonl.gz --out PINS.jsonl
"""

from __future__ import annotations

import collections
import gzip
import json
import sys

from bar3.gpu_search.equiv import diff
from bar3.gpu_search.replay import compare


def lines(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def rounds(pairs):
    bad = 0
    for new, old in pairs:
        a = [r for r in lines(new) if "round" in r]
        b = [r for r in lines(old) if "round" in r]
        out = []
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b):
                out.append(f"round {i}: only in {'the stored run' if i >= len(a) else 'the rerun'}")
                continue
            d = compare(b[i], a[i])
            if d:
                out.append(f"round {i}: DIFF {d}")
        bad += bool(out)
        came = next((r["round"] for r in a if r.get("general")), None)
        print(f"{new.rsplit('/', 1)[-1]}: {len(a)} rounds against {len(b)}: "
              + ("identical, record for record" if not out else "; ".join(out))
              + (f"; came in round {came}: {a[came]['general']['program']}" if came is not None else ""), flush=True)
    print("PASS" if not bad else f"FAIL: {bad} runs differ", flush=True)
    return bad


def searches(new_paths, old_paths):
    new = {}
    for p in new_paths:
        for key, rec in lines(p):
            new.setdefault(key, rec)
    old = {}
    for p in old_paths:
        for key, rec in lines(p):
            if key in new and key not in old:
                old[key] = rec
    tally = collections.Counter()
    shown = 0
    for key, rec in new.items():
        if key not in old:
            tally[(rec.get("kind"), "new")] += 1
            continue
        st = old[key]
        d = diff(st, rec)
        status = "cpu_timed" if st.get("stop", "").startswith("time") else ("same" if not d else "DIFF")
        tally[(rec.get("kind"), status)] += 1
        if status != "same" and shown < 20:
            shown += 1
            print(f"  {status} {rec.get('kind')} {rec.get('name')}: {d} stored {st['level_sizes']} {st['stop']!r} "
                  f"new {rec['level_sizes']} {rec['stop']!r}", flush=True)
    kinds = sorted({k for k, _ in tally})
    for k in kinds:
        print(f"  {k:14s} " + ", ".join(f"{s} {tally[(k, s)]}" for s in ("same", "DIFF", "cpu_timed", "new")
                                        if tally[(k, s)]), flush=True)
    n = {s: sum(v for (_, s_), v in tally.items() if s_ == s) for s in ("same", "DIFF", "cpu_timed", "new")}
    print(f"{len(new)} searches: {n['same']} identical to the stored record, {n['DIFF']} differing, {n['cpu_timed']} "
          f"cut by the CPU's time limit, {n['new']} not in the earlier stores", flush=True)
    print("PASS" if not n["DIFF"] else f"FAIL: {n['DIFF']} differing", flush=True)
    return n["DIFF"]


def pins(new_paths, old_paths, out):
    keys = {line[0] for p in new_paths for line in lines(p)}         # a store's [key, record], a capture's [key, ...]
    n = 0
    with open(out, "w") as fh:
        for p in old_paths:
            for line in lines(p):
                key, rec = line[0], line[-1]
                if key in keys and rec.get("stop", "").startswith("time"):
                    keys.discard(key)
                    fh.write(json.dumps([key, rec]) + "\n")
                    n += 1
                    print(f"  pinned {rec.get('kind')} {rec.get('name')}: {rec['stop']!r}", flush=True)
    print(f"{n} searches cut by the CPU's clock, written to {out}", flush=True)


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "rounds":
        sys.exit(1 if rounds(list(zip(args[::2], args[1::2]))) else 0)
    out = args.pop(args.index("--out") + 1) if "--out" in args else None
    if out:
        args.remove("--out")
    cut = args.index("--old")
    if mode == "pins":
        return pins(args[:cut], args[cut + 1:], out)
    sys.exit(1 if searches(args[:cut], args[cut + 1:]) else 0)


if __name__ == "__main__":
    main()
