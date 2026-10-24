"""The round's search alone, offline (SP_BREADTH.md): a climb's library (parsed from a run's log: its sleep lines, or
the round_found line's library), each formula's answers on H1's rows (computed here with Python's integers: a study
check, never part of a training file), and spbcore's search_round as the training file runs it, with exact cells.

    env: LIB (a record .json or a log: the library at the end of its climb; comma list for several), TASKS (comma list
         of STEP's keys; default all), T (the answers' rounds, default 1), ROWS (answer rows, default 512, ANS_POOL),
         MACRO (the search by calls' depth, default 3), MACRO_AFTER (12), ROUND (states a level, 4e6), CONST (1:
         constants from the data), ENDS (red: a round ends with a reduction mod N)

    run on Modal (a GPU: the full-size search is heavy): modal_spb.py --envs '[{"TAG": "off_g1", "LIB": "..."}]'
"""
import ast
import json
import os
import re
import time

import torch

from integration.sp_breadth import libb as L
from integration.sp_breadth import spbcore as S
from integration.self_practice.core import digits, load_rows

dev = "cuda" if torch.cuda.is_available() else "cpu"
R8 = 10 ** 8
STEP = {"h1": lambda y, n: y * y % n, "rho": lambda y, n: (y * y + 1) % n, "ten": lambda y, n: 10 * y * y % n,
        "cube": lambda y, n: y ** 3 % n, "mont": lambda y, n: y * y * pow(R8, -1, n) % n,
        "tenrho": lambda y, n: (10 * y * y + 1) % n, "plus3": lambda y, n: (y * y + 3) % n,
        "minus1": lambda y, n: (y * y - 1) % n}


def library(path):
    """{name: program text} at the end of the run's climb: its sleep lines in order (every merge and routine)."""
    text = open(path).read()
    if path.endswith(".json"):
        text = json.loads(text)["stdout"]
    m = re.search(r"round_found step=\d+ programs: .*? library: (\{.*\})$", text, flags=re.M)
    if m:
        return ast.literal_eval(m.group(1))
    lib = {}
    for line in text.splitlines():
        m = re.match(r"LR climb step=\d+ L\d+ sleep: (.*) \(\d+\.\ds\)$", line)
        if m and "=" in m.group(1):
            for part in m.group(1).split("; "):
                name, _, prog = part.partition("=")
                lib[name] = prog
    return lib


def climb(lib, **kw):
    c = S.SelfClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, fuse=4, **kw)
    c.lib = {k: L.parse(v) for k, v in lib.items()}
    for tok in lib:
        if tok.startswith("T."):
            *law, pattern, binding = tok[2:].split(".")
            c.ops.add(".".join(law), pattern, tok, tuple(binding))
    return c


def main():
    env = os.environ.get
    t = int(env("T", 1))
    tasks = env("TASKS", ",".join(STEP)).split(",")
    rows = load_rows(int(env("ROWS", 512)), seed=1, T=1)
    kw = dict(macro_cost=int(env("MACRO", 3)), macro_after=int(env("MACRO_AFTER", 12)),
              round_budget=int(float(env("ROUND", 4e6))))
    kw.update(constants=env("CONST") == "1", ends=env("ENDS") or None)
    for path in env("LIB").split(","):
        lib = library(path)
        print(f"{path}: {len(lib)} names, {sum(k.startswith('T.') for k in lib)} routines; t {t}; {kw}", flush=True)
        for task in tasks:
            one(lib, task, rows, t, kw)


def one(lib, task, rows, t, kw):
    """One formula's round search: its answers after t rounds on the rows, the search as the file runs it."""
    t0, step = time.time(), STEP[task]

    def run(y, n):
        for _ in range(t):
            y = step(y, n)
        return y
    rws = [dict(r, result=run(int(r["x"]), int(r["modulus"]))) for r in rows]
    ans = tuple(digits(rws, k).to(dev) for k in ("modulus", "x", "result"))
    progs, rec = climb(lib, **kw).search_round(ans, t=t)
    print(f"{task}: found {rec['found_cost']}{' constant ' + rec['const'] if rec.get('const') else ''} "
          f"({rec['seconds']} s) {rec['stop']} {rec['programs'][:3]} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
