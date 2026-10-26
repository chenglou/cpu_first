"""NEWTON.md §5: the outcomes, read from the records, each program rechecked. For every run (records/*_nw1.jsonl): the
round its target came in, or that it did not; the program and its size in primitive instructions; its check on every
pair x < N for N <= 120, on machine_for(120) and with no step limit to speak of (10^9 steps, a cap of 10^9), so that
a slow program is told from a wrong one. Then, for each new problem (sources rsum and offset), the rounds it was posed,
learnable (with its cost) and picked, its other statuses, and whether each round's picks and library are the earlier
run's (the run without the sources). No search runs.

    cd cpu_first && <python with numba> -m bar3.scale.newton.verify
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from library import dsl
from library.inventing import brm
from library.inventing.jobs import machine_for
from bar3.breadth import targets as TG
from bar3.scale import engine as E
from bar3.scale.heldout import targets as _ho  # noqa: F401  (registers the held-out targets)

HERE = Path(__file__).resolve().parent
EARLIER = {"tri": HERE.parent / "heldout" / "records" / "{rule}_deep_sub_tri_ho1.jsonl",
           "tenrho": HERE.parent / "subgoal" / "records" / "{rule}_deep_sub_tenrho_s1.jsonl"}
NEW = ("rsum", "offset")


def read(path):
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]


def pairs(top):
    xs = np.array([x for n in range(1, top + 1) for x in range(n)], np.int64)
    Ns = np.array([n for n in range(1, top + 1) for _ in range(n)], np.int64)
    return xs, Ns


def check(prog, lib, tname, m, top=120):
    """Right on how many pairs x < N <= top, each run alone on the compiled engine (bar3/scale/engine._run), and on how
    many it ran out of steps."""
    xs, Ns = pairs(top)
    want = TG.TARGETS[tname].answers(xs, Ns)
    code, depth = E.compile_items(tuple(prog), lib)
    spc, srem = np.empty(depth + 1, np.int64), np.empty(depth + 1, np.int64)
    right = dead = 0
    for x, n, w in zip(xs.tolist(), Ns.tolist(), want.tolist()):
        regs = np.zeros(5, np.int32)
        regs[brm.X], regs[brm.N] = x, n
        if E._run(code, regs, m.cap, m.limit, spc, srem):
            dead += 1
        elif int(regs[brm.A]) == w:
            right += 1
    return f"{right}/{len(xs)}" + (f" ({dead} out of steps)" if dead else "")


def new_problems(recs):
    out = {}
    for r in recs:
        for c in r["pool"]:
            if c["source"] in NEW:
                out.setdefault(c["name"], dict(posed=[], learnable=[], picked=[], other={}))["posed"].append(r["round"])
        for n, v in r["status"].items():
            if n in out:
                if v.get("status") == "learnable":
                    out[n]["learnable"].append((r["round"], v.get("cost")))
                elif v.get("cost") is not None:
                    out[n]["other"].setdefault(v["status"], []).append((r["round"], v.get("cost"), v.get("of")))
        for n in r["picked"]:
            if n in out:
                out[n]["picked"].append(r["round"])
    return out


def main():
    for path in sorted((HERE / "records").glob("*_nw1.jsonl")):
        recs = read(path)
        tname, rule = recs[0]["target_name"], path.name.split("_deep_sub")[0]
        old = read(str(EARLIER[tname]).format(rule=rule))
        same = [r["round"] for r, o in zip(recs, old) if r["picked"] == o["picked"] and r["lib"] == o["lib"]]
        g = next((r for r in recs if r.get("general")), None)
        line = f"{path.name}: {len(recs)} rounds; rounds with the earlier run's picks and library: {same}"
        if g is None:
            line += f"; did not come (the earlier run: {'came' if any(o.get('general') for o in old) else 'did not'})"
        else:
            lib = {t: dsl.parse(v) for t, v in g["lib"].items()}
            prog = dsl.parse(g["general"]["program"])
            prim = dsl.cost(dsl.expand(prog, lib))
            line += (f"; came in round {g['round']}: {g['general']['program']} = {dsl.show(dsl.expand(prog, lib))} "
                     f"({prim} primitive); every pair x < N <= 120: {check(prog, lib, tname, machine_for(120))} on "
                     f"machine_for(120), {check(prog, lib, tname, brm.Machine(cap=10 ** 9, limit=10 ** 9))} with "
                     f"10^9 steps")
        print(line, flush=True)
        for n, v in new_problems(recs).items():
            if v["learnable"] or v["picked"] or v["other"]:
                print(f"  {n}: posed in rounds {v['posed']}; learnable {v['learnable']}; picked {v['picked']}; "
                      f"other {v['other']}", flush=True)


if __name__ == "__main__":
    main()
