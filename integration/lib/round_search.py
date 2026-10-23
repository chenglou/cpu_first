"""The round, found by composition and chosen by the answers (INTEGRATION.md, stage 3): a breadth-first search at the
task's full size over the routines the climb found and the machine's straight-line primitives (no loops: at full size
the loops live inside the routines), whose goal is the answers of the rows with the batch's smallest T: from x in X and
N in N, the accumulator must hold the row's answer y. Nothing in it says what the round computes; the answers decide,
and a round that no short composition of the found routines computes is not found.

    cd cpu_first && PYTHONPATH=. python -m integration.lib.round_search     # offline check with LIBRARY.md's routines
"""

from __future__ import annotations

import json
import os
import time

import torch

from library import dsl as pdsl
from integration.lib import tdsl, tfast, inclimb

STRAIGHT = [p for p in pdsl.PRIMS]          # every primitive (each runs once at the pointer, or moves whole registers)


def round_problem(cells, Nd, Xd, Yd, NP=8, P=16, device="cpu", seed=0):
    """Probes: the first NP answer rows (digit tensors [n, 8], LSB first) at the task's size (10, P // 2)."""
    st = tdsl.Setting(10, P // 2, cells)
    gen = torch.Generator().manual_seed(seed)
    ins = {"X": inclimb.pad(Xd[:NP], P), "N": inclimb.pad(Nd[:NP], P)}
    root = tdsl.fresh(st, ins, gen, NP, device)
    tgt = {"A": inclimb.pad(Yd[:NP], P).T.to(device=device, dtype=torch.uint8)}
    return tdsl.Problem("round", [st], [root], [tgt])


def accept_round(prog, lib, cells, Nd, Xd, Yd, P=16, device="cpu", seed=1):
    """The share of answer rows (all given) the program reproduces at the task's size."""
    st = tdsl.Setting(10, P // 2, cells)
    A, _ = inclimb.run_prog(st, prog, lib, {"X": inclimb.pad(Xd, P), "N": inclimb.pad(Nd, P)},
                            torch.Generator().manual_seed(seed))
    return float((A[:, :Yd.shape[1]] == Yd).all(1).float().mean())


def search_round(lib, routines, cells, Nd, Xd, Yd, NP=8, max_cost=6, budget=4_000_000, device="cpu", log=print,
                 accept_rows=None):
    """The round: BFS over routines + straight-line primitives at full size, goal = the answers on NP rows, accepted
    on accept_rows (Nd, Xd, Yd) when given. Returns (accepted programs, record)."""
    t0 = time.time()
    tokens = list(STRAIGHT) + list(routines)
    prob = round_problem(cells, Nd, Xd, Yd, NP, device=device)
    classes = tfast.enumerate_items(tokens, lib, prob.settings, max_body=0)
    rows = accept_rows or (Nd, Xd, Yd)
    res = tfast.bfs(prob, classes, lib, max_cost=max_cost, max_states=budget,
                    accept=lambda ps: [accept_round(p, lib, cells, *rows, device=device) == 1.0 for p in ps],
                    chunk=1 << 19)
    rec = dict(found_cost=res.found_cost, level_sizes=res.level_sizes, seconds=round(time.time() - t0, 1),
               stop=res.stop, tokens=len(tokens), classes=len(classes),
               programs=[pdsl.show(p) for p in (res.accepted or [])][:20], aliases=res.aliases[:2])
    log(f"round: cost {rec['found_cost']} levels {rec['level_sizes']} {rec['seconds']}s {rec['stop']} "
        f"{rec['programs'][:4]}")
    return res.accepted or [], rec


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    rec = json.load(open(os.path.join(here, "r11_lib.json")))           # LIBRARY.md r11's library (offline climb)
    lib = {k: pdsl.parse(v) for k, v in rec["lib"].items()}
    # the routines a law-derived climb makes (no square, no x^2 mod N: the round is what is searched)
    routines = [k for k in lib if k.startswith("T.") and k not in ("T.sq", "T.sqmod", "T.mulmod")]
    lib = {k: v for k, v in lib.items() if k not in ("T.sq", "T.sqmod", "T.mulmod")}
    task = os.environ.get("TASK", "h1")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = inclimb.H1_DIR if task == "h1" else os.path.join(os.path.dirname(inclimb.H1_DIR), f"litmus_{task}")
    rows = inclimb.load_rows(400, seed=3, T=1, path=path)
    Nd, Xd, Yd = inclimb.digits_of_rows(rows)
    cells = tdsl.Cells(10, dev)
    progs, r = search_round(lib, routines, cells, Nd[:8], Xd[:8], Yd[:8], device=dev,
                            budget=int(os.environ.get("BUDGET", 4_000_000)), accept_rows=(Nd, Xd, Yd))
    print(json.dumps(r, indent=1, default=str))
