"""The whole runs against their plain twins (GATE.md §6, §9): each gated run (learnrun.py, records `<config>_<target>_<tag>`:
gt1 the fifth launch, gt2 §9's with the anchors kept to the plain search's finds) round by round against the
consolidation arm's record of the same config and target (consolidate/records, tags ab1/ab2): the round the target came
(a general program), the picks, the routines added, and what the gate's second searches found (`<run>.gate.jsonl`: per
round, how many it ran and found, by kind, and whether a gate-found problem was picked).

    cd cpu_first && python3 -m bar3.scale.gate.wcompare bar3/scale/gate/records/learn [gt1|gt2]
"""

from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "consolidate", "records")


def rounds_of(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = [json.loads(line) for line in op(path, "rt") if line.strip()]
    return [r for r in rows if "round" in r]


def came(rounds):
    for r in rounds:
        if r.get("general"):
            g = r["general"]
            return r["round"], (g.get("program") if isinstance(g, dict) else str(g))
    return None, None


def baseline(config, target):
    fs = sorted(glob.glob(os.path.join(BASE, f"{config}_{target}_ab*.jsonl.gz")))
    return rounds_of(fs[0]) if fs else None


def gate_log(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def main(folder, tag="gt1"):
    for path in sorted(glob.glob(os.path.join(folder, f"*_{tag}.jsonl"))):
        name = os.path.basename(path)[:-len(f"_{tag}.jsonl")]
        config, target = name.rsplit("_", 1)
        rs = rounds_of(path)
        base = baseline(config, target)
        gl = gate_log(path[:-len(".jsonl")] + ".gate.jsonl")
        r_g, p_g = came(rs)
        r_b, p_b = came(base) if base else (None, None)
        found = [g for g in gl if g.get("found") is not None]
        by = {}
        for g in gl:
            k = g.get("kind")
            by.setdefault(k, [0, 0])
            by[k][0] += 1
            by[k][1] += g.get("found") is not None
        print(f"{name}: gated {len(rs)} rounds, target {'round ' + str(r_g) if r_g is not None else 'no'}"
              f" | plain twin {('round ' + str(r_b)) if r_b is not None else 'no'} ({len(base) if base else '-'} rounds)"
              f" | gate searches {by} ")
        names = {g["name"] for g in found}
        for r in rs:
            b = next((x for x in (base or []) if x["round"] == r["round"]), None)
            pk = r.get("picked", [])
            same = b is not None and b.get("picked") == pk and sorted(b.get("lib", {})) == sorted(r.get("lib", {}))
            print(f"   round {r['round']}: picked {pk}{' (gate-found: ' + str(sorted(set(pk) & names)) + ')' if set(pk) & names else ''}"
                  f"; {'same picks and library as the twin' if same else 'differs from the twin' if b else 'twin has no such round'}")
        if p_g:
            print(f"   program: {p_g}")
    print()
    table(folder, tag)
    print()
    mod_law(folder, tag)
    print()
    second_searches(folder, tag)
    if tag != "gt1":
        print()
        anchor_check(folder, tag)


NAMES = {"h1": "x² mod N", "affine": "2x + 1", "minus1": "x² − 1", "rho": "x² + 1", "x2x": "x² + x", "cube": "x³",
         "fib": "Fibonacci", "plus3": "x² + 3", "pow2": "2^x", "three": "3·x²", "x1sq": "(x + 1)²",
         "tri": "triangular", "ten": "10·x²", "x4": "x⁴", "tenrho": "10·x² + 1", "x3x": "x³ + x", "lucas": "Lucas",
         "trib": "Tribonacci", "floorsq": "floor(x²/N)"}


def said(rounds):
    r, _ = came(rounds)
    return f"round {r}" if r is not None else f"no ({len(rounds)} rounds)"


def table(folder, tag="gt1"):
    """GATE.md §6's table (§9's with tag gt2, and the fifth launch's outcome beside it): per run, the round the target
    came in the twin (consolidate/records), in the plain rerun (`<run>_pt1.jsonl`, learnrun with no gates, where there is
    one) and with the gate (rounds are numbered from 0); and per round of the gated run, how many of its picks only the
    gate had solved and what its second searches found (found / run)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, f"*_{tag}.jsonl"))):
        name = os.path.basename(path)[:-len(f"_{tag}.jsonl")]
        config, target = name.rsplit("_", 1)
        rs, base = rounds_of(path), baseline(config, target)
        gl = gate_log(path[:-len(".jsonl")] + ".gate.jsonl")
        r_b, _ = came(base) if base else (None, None)
        kind = "held-out" if base is None else "win" if r_b is not None else "miss"
        twin = "–" if base is None else said(base)
        rerun = os.path.join(folder, f"{name}_pt1.jsonl")
        twin += f"; rerun {said(rounds_of(rerun))}" if os.path.exists(rerun) else ""
        gated = said(rs)
        first = os.path.join(folder, f"{name}_gt1.jsonl")
        if tag != "gt1":
            gated = f"{said(rounds_of(first)) if os.path.exists(first) else '–'} | {gated}"
        found = {g["name"] for g in gl if g.get("found") is not None}
        picks = [len(set(r.get("picked", [])) & found) for r in rs]
        tsearch = [g for g in gl if g.get("kind") == "target"]
        searched = f"{sum(g.get('found') is not None for g in gl)} / {len(gl)}"
        if tsearch:
            searched += f"; target {sum(g.get('found') is not None for g in tsearch)} / {len(tsearch)}"
        rows.append((("win", "miss", "held-out").index(kind), r_b if r_b is not None else 9, NAMES.get(target, target),
                     config.replace("_ct2a2_probe", ""), kind, twin, gated, picks, searched))
    print("| formula | config | kind | plain twin | " + ("fifth launch | anchors kept plain" if tag != "gt1" else
          "with the gate") + " | its picks only the gate solved, by round | second searches: found / run |")
    print("|---|---|---|---|---|---|---|" + ("---|" if tag != "gt1" else ""))
    for _, _, f, c, k, t, g, p, s in sorted(rows):
        print(f"| {f} | {c} | {k} | {t} | {g} | {', '.join(map(str, p))} | {s} |")


MOD, MOD_ROUTINE = "law/mod(X,N)/x+qN", "T.mod(X,N)@x+qN"  # the practice problem, and its routine once picked


def calls(program, lib, seen=None):
    """The library routines a program calls, directly or through another routine."""
    seen = set() if seen is None else seen
    for t in re.split(r"[\s\[\]]+", program):
        if t in lib and t not in seen:
            seen.add(t)
            calls(lib[t], lib, seen)
    return seen


def mod_law(folder, tag="gt1"):
    """GATE.md §6's account of the mod law (x mod N, `law/mod(X,N)/x+qN`): per run, the rounds that picked it with the
    gate, in the twin and in the plain rerun, and whether the target's program, where one came, calls it as a routine
    (with tag gt2, the fifth launch's rounds are the last column)."""
    def picked(rounds):
        return "–" if rounds is None else ",".join(str(r["round"]) for r in rounds if MOD in r.get("picked", [])) or "never"

    def uses(rounds):
        r = next((r for r in rounds or [] if r.get("general")), None)
        return "–" if r is None else "yes" if MOD_ROUTINE in calls(r["general"]["program"], r.get("lib", {})) else "no"

    print("the mod law: rounds picked (gate | twin | plain rerun" + (" | fifth launch" if tag != "gt1" else "") +
          "); does the target's program call it (gate | twin)")
    for path in sorted(glob.glob(os.path.join(folder, f"*_{tag}.jsonl"))):
        name = os.path.basename(path)[:-len(f"_{tag}.jsonl")]
        config, target = name.rsplit("_", 1)
        rs, base = rounds_of(path), baseline(config, target)
        rerun = os.path.join(folder, f"{name}_pt1.jsonl")
        rr = rounds_of(rerun) if os.path.exists(rerun) else None
        first = os.path.join(folder, f"{name}_gt1.jsonl")
        fifth = f" | {picked(rounds_of(first))}" if tag != "gt1" and os.path.exists(first) else ""
        print(f"  {NAMES.get(target, target)} ({config.replace('_ct2a2_probe', '')}): picked {picked(rs)} | {picked(base)}"
              f" | {picked(rr)}{fifth}; called {uses(rs)} | {uses(base)}")


def second_searches(folder, tag="gt1"):
    """GATE.md §6's totals of the gate's second searches over every run, by kind (after a 1M probe, after the 10M deep
    probe, after the round's target search): how many ran, how many found a program, and the found programs' prices."""
    by, cost = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(collections.Counter)
    for path in sorted(glob.glob(os.path.join(folder, f"*_{tag}.gate.jsonl"))):
        for g in gate_log(path):
            by[g["kind"]][0] += 1
            if g.get("found") is not None:
                by[g["kind"]][1] += 1
                cost[g["kind"]][g["found"]] += 1
    print("the gate's second searches: kind, found / run, found by price")
    for k, (n, f) in sorted(by.items()):
        print(f"  {k}: {f} / {n} ({100 * f / n:.1f}%) {dict(sorted(cost[k].items()))}")


def anchor_check(folder, tag="gt2"):
    """GATE.md §9's check of the anchor rule: in every round of every run, the two costliest learnable problems the plain
    search solved (not a second search's find of that round's probes, matched by library size), and whether every
    scored pair holds one of them."""
    bad = 0
    for path in sorted(glob.glob(os.path.join(folder, f"*_{tag}.jsonl"))):
        name = os.path.basename(path)[:-len(f"_{tag}.jsonl")]
        gl = gate_log(path[:-len(".jsonl")] + ".gate.jsonl")
        prev, out = 0, []
        for r in rounds_of(path):
            st, sub = r["status"], (r.get("subgoal") or {}).get("name")
            gf = {g["name"] for g in gl if g["lib"] == prev and g.get("found") is not None and g["kind"] != "target"}
            lrn = [c["name"] for c in r["pool"] if st.get(c["name"], {}).get("status") == "learnable"]
            top = sorted([n for n in lrn if n not in gf], key=lambda n: (-st[n]["cost"], n != sub))[:2]
            ok = all({a, b} & set(top) for _, a, b, _ in r.get("pairs", []))
            bad += not ok
            out.append(f"r{r['round']} {'ok' if ok else 'NOT'} ({len(gf & set(lrn))} gate finds kept out)")
            prev = len(r.get("lib", {}))
        print(f"  {name}: " + "; ".join(out))
    print(f"anchor rule: {'every scored pair holds a plain anchor' if not bad else str(bad) + ' rounds break it'}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
