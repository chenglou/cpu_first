"""Phase E's six fresh held-out tasks (PREREG.md): drawn at random from the space y <- a y^k + b y + c mod N, T
times, with k in {2, 3}, a in 1..9 and b, c in 0..9, every task used so far left out (phase D's pair included). The
seed is not chosen here: it is the repository's HEAD commit when phase E began, cb623f9, read as a hexadecimal
number (checks/logs/head_at_start.txt). random.Random(seed).sample(space, 6), as phase D drew its two. Writes
heldout.json (the names, poly_A_K_B_C, as legal_wider/tasks_wider.py and core_f.py read them).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_deep/checks/draw_heldout.py
"""
import json
import random
import sys
from pathlib import Path

SEED = int("cb623f9", 16)
USED = {(1, 2, 0, 0): "H1 (and xfirst)", (1, 3, 0, 0): "the cube", (1, 2, 0, 1): "rho", (1, 2, 0, 3): "plus3",
        (1, 2, 1, 0): "plusy (phase C's held-out)", (3, 2, 0, 2): "three2 (phase C's held-out)",
        (8, 3, 8, 3): "poly_8_3_8_3 (phase D's held-out)", (2, 3, 9, 9): "poly_2_3_9_9 (phase D's held-out)"}
# outside the space already: ten and tenrho (a = 10), minus1 (c = -1), mont, xy, M4


def main():
    space = [(a, k, b, c) for k in (2, 3) for a in range(1, 10) for b in range(10) for c in range(10)
             if (a, k, b, c) not in USED]
    draw = random.Random(SEED).sample(space, 6)
    names = [f"poly_{a}_{k}_{b}_{c}" for a, k, b, c in draw]
    print(f"Python {sys.version.split()[0]}; seed {SEED} (0xcb623f9); the space: {len(space)} tasks "
          f"(2 x 9 x 10 x 10 less {len(USED)} used so far: {sorted(USED.values())})")
    for (a, k, b, c), nm in zip(draw, names):
        print(f"  {nm}: y <- {a} y^{k} + {b} y + {c} mod N")
    out = Path(__file__).resolve().parents[1] / "heldout.json"
    out.write_text(json.dumps({"seed": SEED, "space": len(space), "tasks": names}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
