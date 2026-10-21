"""The tables of ANCHORS.md from the run files (runs/*.jsonl, records/*.json).

    python tables.py exact | census | train | records
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def jsonl(name):
    p = HERE / "runs" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def exact():
    """Exact map: tables passing each lesson set on every input of a tiny machine."""
    res = {r["job"]["name"]: r for r in jsonl("x1")}
    plan = json.loads((HERE / "plans" / "x1.json").read_text())
    sets = []
    for j in plan:
        ls = "+".join(j["lessons"])
        if ls not in sets:
            sets.append(ls)

    def cell(B, W, ls, bt):
        r = res.get(f"x_b{B}w{W}_{ls}_{'bt' if bt else 'free'}")
        if r is None:
            return "–" if bt and all(l in ("less", "more", "neg", "pref") for l in ls.split("+")) else "?"
        n = r["tables"]
        capped = r["capped"]
        if r["groups"] == 0 and r.get("timed_out"):
            return "unfinished (600 s)"
        # a group is "the truth" when every entry it sets is true; its other entries are free (never used)
        truthlike = [g for g in r["sols"] if g == "TRUE" or all("->?" in t for t in g.split())]
        wrong = len(r["sols"]) - len(truthlike)
        free = max((0 if g == "TRUE" else sum(t.count("?") for t in g.split()) for g in truthlike), default=0)
        if not capped and len(truthlike) == 1 and wrong == 0:
            return "**1 (the truth)**" if free == 0 else f"**the truth** ({free} output{'s' if free > 1 else ''} left free)"
        s = f"{n:.3g}" if n >= 1e5 else str(n)
        if not capped and wrong and len(truthlike) == 1 and free == 0:
            if n == len(r["sols"]):
                return f"{s}: the truth + {wrong} wrong"
            return f"{s}, in {len(r['sols'])} groups (one the truth)"
        if r.get("timed_out"):
            return f"≥ {s} (600 s)"
        return ("≥ " if capped else "") + s

    print("| lessons | base 3, borrows free | base 3, borrows held true | base 4, borrows free | base 4, borrows held true |")
    print("|---|---|---|---|---|")
    for ls in sets:
        print(f"| {ls.replace('+', ', ')} | {cell(3, 3, ls, False)} | {cell(3, 3, ls, True)} | {cell(4, 2, ls, False)} | "
              f"{cell(4, 2, ls, True)} |")


def census():
    """Tiny-core censuses (t1): seeds reaching the true table, and where the others end."""
    by = collections.defaultdict(list)
    for r in jsonl("t1") + jsonl("p5"):
        if r["job"]["kind"] != "tiny":
            continue
        n = r["job"]["name"].rsplit("_s", 1)[0]
        by[n].append((r["job"]["seed"], r["final"]["rows"], 2 * r["job"]["B"] ** 2, r["final"]["closest"]))
    print("| core | lessons | seeds reaching the true table | entries right at the end (each seed) |")
    print("|---|---|---|---|")
    for n, v in by.items():
        v.sort()
        tot = v[0][2]
        print(f"| {n} | | {sum(x == tot for _, x, _, _ in v)}/{len(v)} | {', '.join(str(x) for _, x, _, _ in v)} of {tot} |")


def train(plans=("p2", "p3", "p5")):
    """H1-machine training runs (core.py): seeds reaching 200/200, the gate step, the first exact step."""
    print("| run | seeds 200/200 | entries right (each seed) | digit lessons start (step) | first 200/200 (step) |")
    print("|---|---|---|---|---|")
    for plan in plans:
        by = collections.defaultdict(list)
        for r in jsonl(plan):
            if r["job"]["kind"] != "core":
                continue
            n = r["job"]["name"].rsplit("_s", 1)[0]
            tot = 2 * r["job"]["B"] ** 2
            first = next((t["step"] for t in r["traj"] if t["rows"] == tot), None)
            by[n].append((r["job"]["seed"], r["final"]["rows"], tot, r["final"].get("gated_on"), first))
        for n, v in by.items():
            v.sort()
            tot = v[0][2]
            gates = [g for *_, g, _ in v if g is not None]
            firsts = [f for *_, f in v if f is not None]
            rng = lambda xs: (f"{min(xs)}–{max(xs)}" if xs and min(xs) != max(xs) else (str(xs[0]) if xs else "–"))
            print(f"| {n} | {sum(x == tot for _, x, _, _, _ in v)}/{len(v)} | {', '.join(str(x) for _, x, _, _, _ in v)} | "
                  f"{rng(gates)}{'' if len(gates) == len(v) else f' ({len(v) - len(gates)} never)'} | {rng(firsts)} |")


if __name__ == "__main__":
    {"exact": exact, "census": census, "train": train}[sys.argv[1]]()
