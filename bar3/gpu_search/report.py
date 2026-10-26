"""The tables of GPU_SEARCH.md from the records (records/, fetched from the Volume): the equivalence runs, the timing
sample (seconds per level, CPU against each GPU), the reach searches and the costs.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.gpu_search.report [--rates=GPU:$/h,...]

Costs: a CPU search-hour is billed $0.113 (the held-out, digits and scale studies' bills over their search-hours, on
16-core CPU containers running 16 searches at once); a GPU second at the container's rate (GPU, its cores and memory),
from this study's own bills (RATES below, overridable).
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "records"
CPU_RATE = 0.113                                  # $ per CPU search-hour, billed
RATES = {"H100": 4.25, "L40S": 2.30, "A100-80GB": 2.85, "A10G": 1.45, "L4": 1.15}   # $ per container-hour: H100 and
#                       L40S billed; the others their list price plus the $0.35/h of the same cores and memory


def lines(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def equivalence():
    rows = []
    for p in sorted(glob.glob(str(REC / "equiv" / "*.jsonl*"))):
        for r in lines(p):
            rows.append(dict(r, part=os.path.basename(p).split(".")[0]))
    by = collections.defaultdict(collections.Counter)
    cpu = collections.Counter()
    gpu = collections.Counter()
    seen = {}
    for r in rows:
        if r["key"] in seen:
            continue
        seen[r["key"]] = r
        by[r["kind"]][r["status"]] += 1
        cpu[r["kind"]] += r["cpu_s"]
        gpu[r["kind"]] += r["gpu_s"]
    print("| kind | searches | identical | differing | stopped on time (CPU) | stored CPU s | L40S s |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(by):
        c = by[k]
        print(f"| {k} | {sum(c.values()):,} | {c['same']:,} | {c['DIFF']} | {c['cpu_timed']} | {cpu[k]:,.0f} | "
              f"{gpu[k]:,.0f} |")
    tot = collections.Counter()
    for c in by.values():
        tot.update(c)
    print(f"\nall: {sum(tot.values()):,} searches, {tot['same']:,} identical, {tot['DIFF']} differing, "
          f"{tot['cpu_timed']} stopped on time; shared: {sum(1 for r in seen.values() if r['shared'] > 1):,}")
    names = dict(ho1="held-out", ctl1="held-out controls", s1="subgoal", nw1="newton (with picks, rowcheck)",
                 pk1="picks", rc1="rowcheck")
    studies = collections.Counter(names.get(r["file"].split("_")[-1].split(".")[0], "rechecks")
                                  for r in seen.values())
    print("by study:", dict(studies))
    return seen


def bucket(n):
    if n < 1:
        return None
    e = int(math.floor(math.log10(n)))
    return f"1e{e}–1e{e + 1}"


WARM = 4                     # the CPU timing's first jobs, one per worker process: their first level holds numba's compile


def per_level():
    """Seconds per level by level size, each GPU against the CPU, on the timing sample's jobs (same jobs), but the
    first WARM jobs (the compiles: numba's in each CPU worker, Triton's on the GPU)."""
    runs = {}
    for p in sorted(glob.glob(str(REC / "timing" / "t1_*.jsonl"))):
        name = os.path.basename(p)[3:-6]
        runs[name] = {r["key"]: r for r in lines(p)}
    cpu = runs.get("CPU16w4")
    if not cpu:
        print("no CPU timing")
        return
    gpus = [g for g in ("H100", "L40S", "A100-80GB", "A10G", "L4") if g in runs]
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    count = collections.Counter()
    for q, (key, c) in enumerate(cpu.items()):
        if not c["clock"] or q < WARM:
            continue
        cl = [0.0] + c["clock"]
        for g in gpus:
            r = runs[g].get(key)
            if not r or not r["clock"] or len(r["clock"]) != len(c["clock"]):
                continue
            gl = [0.0] + r["clock"]
            for d in range(1, len(cl)):
                b = bucket(c["levels"][d])
                if b is None:
                    continue
                agg[b][g] += gl[d] - gl[d - 1]
                if g == gpus[0]:
                    agg[b]["cpu"] += cl[d] - cl[d - 1]
                    count[b] += 1
    print("| states in the level | levels | CPU s | " + " | ".join(f"{g} s (x)" for g in gpus) + " |")
    print("|---" * (3 + len(gpus)) + "|")
    for b in sorted(agg, key=lambda s: int(s.split("–")[0][2:])):
        a = agg[b]
        cells = " | ".join(f"{a[g]:.3f} ({a['cpu'] / a[g]:.0f}x)" if a[g] else "-" for g in gpus)
        print(f"| {b} | {count[b]} | {a['cpu']:.2f} | {cells} |")
    print()
    print("| engine | jobs | search s (all 56) | of it, the 32 big searches | x the CPU here | x the stored CPU |")
    print("|---|---|---|---|---|---|")
    big = ("probe_b", "target", "probe_deep", "guided")
    for name in ["CPU16w4"] + gpus:
        rs = runs[name]
        s = sum(r["search_s"] for r in rs.values())
        sb = sum(r["search_s"] for r in rs.values() if r["kind"] in big)
        cb = sum(cpu[k]["search_s"] for k, r in rs.items() if r["kind"] in big and k in cpu)
        stb = sum(r["stored_search_s"] for r in rs.values() if r["kind"] in big)
        print(f"| {name} | {len(rs)} | {s:.1f} | {sb:.1f} | {cb / sb:.1f} | {stb / sb:.1f} |")
    print()
    print("| engine | $ for the 32 big searches | x cheaper than the stored CPU's $ |")
    print("|---|---|---|")
    stb = sum(r["stored_search_s"] for r in cpu.values() if r["kind"] in big)
    print(f"| CPU (stored, ${CPU_RATE}/search-hour) | {stb * CPU_RATE / 3600:.4f} | 1 |")
    for g in gpus:
        sb = sum(r["search_s"] for r in runs[g].values() if r["kind"] in big)
        usd = sb * RATES[g] / 3600
        print(f"| {g} (${RATES[g]}/h) | {usd:.4f} | {stb * CPU_RATE / 3600 / usd:.1f} |")


def reach():
    rows = []
    for p in sorted(glob.glob(str(REC / "reach" / "*.jsonl"))):
        g = os.path.basename(p).split("_")[1].split(".")[0]
        rows += [dict(r, gpu=g) for r in lines(p)]
    if not rows:
        return
    print("| search | budget | GPU | level sizes | same levels as the CPU's | CPU s | GPU s | x | GPU $ | CPU $ |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        usd = r["seconds"] * RATES.get(r["gpu"], 0) / 3600
        cpu_usd = (r["cpu_seconds"] or 0) * CPU_RATE / 3600
        x = f"{r['cpu_seconds'] / r['seconds']:.0f}" if r["cpu_seconds"] else "-"
        print(f"| {r['label']} | {r['budget']:,} | {r['gpu']} | {r['levels']} | {r['same_levels']} | "
              f"{r['cpu_seconds']} | {r['seconds']:.1f} | {x} | {usd:.4f} | {cpu_usd:.4f} |")


def main():
    for a in sys.argv[1:]:
        if a.startswith("--rates="):
            for kv in a[8:].split(","):
                g, v = kv.split(":")
                RATES[g] = float(v)
    print("## equivalence\n")
    equivalence()
    print("\n## per level\n")
    per_level()
    print("\n## reach\n")
    reach()


if __name__ == "__main__":
    main()
