"""One round of a run in full: every learnable candidate (cost, program, the curated tasks its routine solves), the
single and pair lookahead scores, and which pairs make the target solvable.

    ~/github/sotaku/venv/bin/python -m bar3.inspect_round bar3/records/ct2_r1.jsonl 0
"""

from __future__ import annotations

import collections
import sys

from bar3.summarize import LAWS, P, covers, load


def main():
    path, rnd = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = load(path)[rnd]
    k = r["k"]
    forms = {c["name"]: c["formula"] for c in r["pool"]}
    forms.update({c.name: c.formula for c in LAWS.setdefault(k, P.laws(k))})
    lib_before = load(path)[rnd - 1]["lib"] if rnd else {}
    by = collections.defaultdict(list)
    for name, s in r["status"].items():
        by[s.get("status", "practice " + s.get("practice", "?"))].append(name)
    print({s: len(v) for s, v in by.items()})
    print("\nlearnable (cost, single-lookahead gain, program, curated tasks it solves):")
    rows = []
    for name in by["learnable"]:
        s = r["status"][name]
        rows.append((s["cost"], r["scores"].get(name), name, s["best"], covers(s["best"], lib_before, k)))
    for cost, g, name, best, cov in sorted(rows, key=lambda t: (-(t[1] if t[1] is not None else -99), t[0])):
        print(f"  {cost} {'' if g is None else f'{g:6.1f}'}  {name:55s} {forms.get(name, '')[:50]:50s} `{best}`"
              + (f"  = {cov}" if cov else ""))
    print("\nunsolved, best fit:")
    for name in sorted(by["unsolved"], key=lambda n: -r["status"][n]["fit"])[:25]:
        s = r["status"][name]
        print(f"  {s['fit']:.2f} {name:55s} {forms.get(name, '')[:60]}  ({s['states'] / 1e6:.1f}M states, {s['stop']})")
    if r["pairs"]:
        solved = [p for p in r["pairs"] if p[3] is not None]
        print(f"\npairs: {len(r['pairs'])} kept in the record, {len(solved)} make the target solvable:")
        for g, a, b, c in solved[:20]:
            print(f"  {g:6.1f} cost {c}: {a}  +  {b}")
        print("top pairs:")
        for g, a, b, c in r["pairs"][:10]:
            print(f"  {g:6.1f} {c}: {a}  +  {b}")
    print("\ntarget probe near misses:", r["target_probe"].get("near", [])[:12])


if __name__ == "__main__":
    main()
