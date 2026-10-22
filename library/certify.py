"""The end-to-end gate (LIBRARY.md §5): a found x^2 mod N program, run at H1's size, on the released ladders.

    ~/github/sotaku/venv/bin/python -m library.certify records/bpe_r3.jsonl [--table seed:step]

It takes the last level of a wake-sleep record, its library and its sqmod program (or, with --task mulmod, its
modular multiplication, fed x in both X and Y), and runs that program
in base 10 with W = 8 (16-place registers) on every prompt of depth_t_{1..64}.jsonl and depth_ood_n_t_{1..64}.jsonl:
x <- program(x, N), T times (the loop over T is Python here), then compares with the label. With --table the
subtraction cell is a learned snapshot from records/learned_tables.json. This checks the algorithm the search found,
not a submission: the prompt's tokens are read here, not by a learned reader.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from library import dsl
from library.inrun import H1_DIR

RUNGS = (1, 2, 4, 8, 16, 32, 64)


def program_from(record_path, task="sqmod"):
    """The task's simplest accepted program (shortest in primitives) in the last level record that has one."""
    from library.wake_sleep import simplest
    recs = [json.loads(l) for l in open(record_path)]
    for rec in reversed(recs):
        if task in rec["corpus"]:
            lib = {k: dsl.parse(v) for k, v in rec["lib"].items()}
            return simplest([dsl.parse(p) for p in rec["corpus"][task]], lib), lib, rec
    return None, None, None


def ladder(prog, lib, st, profile, T, square_via_y=False):
    """square_via_y: the program multiplies X by Y (mulmod), so Y is loaded with x too."""
    rows = [json.loads(l) for l in open(os.path.join(H1_DIR, f"{profile}_{T}.jsonl"))]
    xs = np.array([r["x"] for r in rows], dtype=object)
    Ns = np.array([r["modulus"] for r in rows], dtype=object)
    want = np.array([r["result"] for r in rows], dtype=object)
    rng = np.random.default_rng(0)
    for _ in range(T):
        ins = {"X": dsl.to_digits(xs, st.B, st.P), "N": dsl.to_digits(Ns, st.B, st.P)}
        if square_via_y:
            ins["Y"] = ins["X"]
        b = dsl.fresh(st, ins, rng, len(rows))
        dsl.run_items(st, b, prog, lib)
        xs = np.asarray(dsl.from_digits(b.regs[0, dsl.A].T, st.B), dtype=object)
    return int((xs == want).sum()), len(rows)


def main():
    path = sys.argv[1]
    task = sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else "sqmod"
    if path == "--reference":                   # the hand-written reference program (selftest.py), for timing
        from library.selftest import REFERENCE
        prog, lib, path = dsl.parse(REFERENCE["sqmod"]), {}, Path(__file__).parent / "records" / "reference.jsonl"
    else:
        prog, lib, rec = program_from(path, task)
    if prog is None:
        print(f"no {task} program in", path)
        return
    cells = dsl.Cells(10)
    if "--table" in sys.argv:
        seed, step = map(int, sys.argv[sys.argv.index("--table") + 1].split(":"))
        from library.learned import snapshot
        r = snapshot(seed, step=step)
        cells.sbb_d = np.asarray(r["sbb_d"], np.uint8).reshape(10, 10, 2)
        cells.sbb_c = np.asarray(r["sbb_c"], np.uint8).reshape(10, 10, 2)
        print(f"learned subtraction table: seed {seed} step {step}, {r['right']}/200 right")
    st = dsl.Setting(10, 8, cells=cells)
    print("program:", dsl.show(prog))
    print("in primitives:", dsl.cost(dsl.expand(prog, lib)), "tokens:", dsl.show(dsl.expand(prog, lib)))
    out = {"record": str(path), "program": dsl.show(prog), "primitive": dsl.show(dsl.expand(prog, lib)), "rungs": {}}
    for profile in ("depth_t", "depth_ood_n_t"):
        cert, broken = 0, False
        for T in RUNGS:
            ok, n = ladder(prog, lib, st, profile, T, square_via_y=(task == "mulmod"))
            out["rungs"][f"{profile}_{T}"] = [ok, n]
            broken = broken or ok < n
            cert = cert if broken else T
            print(f"  {profile:14s} T={T:2d}: {ok}/{n}", flush=True)
        out[f"certified_{profile}"] = cert
    print("certified T:", out["certified_depth_t"], "(seen moduli) /", out["certified_depth_ood_n_t"], "(unseen)")
    dest = Path(path).with_name(Path(path).stem + ("" if task == "sqmod" else f"_{task}") + "_certify.json")
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
