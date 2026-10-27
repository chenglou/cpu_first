"""Phase D's held-out pair (PREREG.md): two tasks drawn at random from the space y <- a y^k + b y + c mod N, T times,
with k in {2, 3}, a in 1..9 and b, c in 0..9, the tasks of earlier rounds left out. The seed is not chosen here: it is
the repository's HEAD commit when phase D began, ab49f60, read as a hexadecimal number. Writes heldout.json (the
names, poly_A_K_B_C, as tasks_wider.py and core_f.py read them).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_wider/checks/draw_heldout.py
"""
import json
import random
import sys
from pathlib import Path

SEED = int("ab49f60", 16)
EARLIER = {(1, 2, 0, 0): "H1 (and xfirst)", (1, 3, 0, 0): "the cube", (1, 2, 0, 1): "rho", (1, 2, 0, 3): "plus3",
           (1, 2, 1, 0): "plusy (phase C's held-out)", (3, 2, 0, 2): "three2 (phase C's held-out)"}
# outside the space already: ten and tenrho (a = 10), minus1 (c = -1), mont, xy, M4


def main():
    space = [(a, k, b, c) for k in (2, 3) for a in range(1, 10) for b in range(10) for c in range(10)
             if (a, k, b, c) not in EARLIER]
    draw = random.Random(SEED).sample(space, 2)
    names = [f"poly_{a}_{k}_{b}_{c}" for a, k, b, c in draw]
    print(f"Python {sys.version.split()[0]}; seed {SEED} (0xab49f60); the space: {len(space)} tasks "
          f"(2 x 9 x 10 x 10 less {len(EARLIER)} of earlier rounds: {sorted(EARLIER.values())})")
    for (a, k, b, c), nm in zip(draw, names):
        print(f"  {nm}: y <- {a} y^{k} + {b} y + {c} mod N")
    out = Path(__file__).resolve().parents[1] / "heldout.json"
    out.write_text(json.dumps({"seed": SEED, "space": len(space), "tasks": names}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
