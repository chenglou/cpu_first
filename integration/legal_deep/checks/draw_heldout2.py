"""Phase E's second six held-out tasks (PREREG_ADDENDUM1.md): drawn at random from the space y <- a y^k + b y + c mod N,
T times, k in {2, 3}, a in 1..9, b and c in 0..9, every task used so far left out (the HEAD-seeded six included). The
seed is the fixed file's own md5, its first 8 hex digits read as a number: any change to the file changes the draw, so
the design cannot be tuned toward it. random.Random(seed).sample(space, 6), as the first six were drawn. Writes
heldout2.json.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_deep/checks/draw_heldout2.py
"""
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FILE_MD5 = "56acca7358c74c65c72b5f33e3b9aafa"  # the file fixed at 19:25:10 PDT (checks/logs/file_fixed.txt)
USED = {(1, 2, 0, 0): "H1 (and xfirst)", (1, 3, 0, 0): "the cube", (1, 2, 0, 1): "rho", (1, 2, 0, 3): "plus3",
        (1, 2, 1, 0): "plusy (phase C's held-out)", (3, 2, 0, 2): "three2 (phase C's held-out)",
        (8, 3, 8, 3): "poly_8_3_8_3 (phase D's held-out)", (2, 3, 9, 9): "poly_2_3_9_9 (phase D's held-out)"}
USED.update({tuple(int(v) for v in t.split("_")[1:]): f"{t} (phase E's HEAD-seeded six)"
             for t in json.loads((HERE / "heldout.json").read_text())["tasks"]})
# outside the space already: ten and tenrho (a = 10), minus1 (c = -1), mont, xy, M4


def main():
    got = hashlib.md5((HERE / "submission_stage3_deep.py").read_bytes()).hexdigest()
    assert got == FILE_MD5, got
    seed = int(got[:8], 16)
    space = [(a, k, b, c) for k in (2, 3) for a in range(1, 10) for b in range(10) for c in range(10)
             if (a, k, b, c) not in USED]
    draw = random.Random(seed).sample(space, 6)
    names = [f"poly_{a}_{k}_{b}_{c}" for a, k, b, c in draw]
    print(f"Python {sys.version.split()[0]}; the file's md5 {got}; seed int('{got[:8]}', 16) = {seed}; the space: "
          f"{len(space)} tasks (2 x 9 x 10 x 10 less {len(USED)} used so far)")
    for (a, k, b, c), nm in zip(draw, names):
        steps = (k == 3 and a > 1 and b > 0) and 4 or None
        print(f"  {nm}: y <- {a} y^{k} + {b} y + {c} mod N{'  (four Horner steps)' if steps else ''}")
    out = HERE / "heldout2.json"
    out.write_text(json.dumps({"seed": seed, "file_md5": got, "space": len(space), "tasks": names}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
