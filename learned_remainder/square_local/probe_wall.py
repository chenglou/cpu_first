"""Domain walls of the symmetric ladder check X * (b + 1) - X * b = X (both rows soft): train unit+zero+peano to its
stall, print each cell's offset f(a, b) - a b (constant within blocks of b, a jump at each wall), then compare the soft
loss after moving one cell at a wall with moving the whole block above it.
  python probe_wall.py SEED STEPS"""
import sys
import torch
import tiny

seed, steps = int(sys.argv[1]), int(sys.argv[2])
names = ["unit", "zero", "peano"]
out = tiny.train(names, seed=seed, steps=steps, quiet=True, log_every=steps)
Llo, Lhi = out["logits"]
f = out["f"]
off = f - tiny.TRUE_F
print("offsets f(a, b) - a*b (rows a, columns b):")
print(off)
N, x = tiny.universe(tiny.ROWS, 1000 + seed)


def loss_of(Llo, Lhi):
    Lo, Hi = Llo.softmax(-1), Lhi.softmax(-1)
    fh = Llo.argmax(-1) + 10 * Lhi.argmax(-1)
    tot, oks = 0.0, []
    for n_ in names:
        l, ok = tiny.lesson(n_, N[:1024], x[:1024], Lo, Hi, fh)
        tot += float(l)
        oks.append(round(ok, 4))
    return round(tot, 4), oks


def put(Llo, Lhi, cells):
    Llo, Lhi = Llo.clone(), Lhi.clone()
    for (a, b), v in cells.items():
        for L, d in ((Llo, v % 10), (Lhi, v // 10)):
            L[a, b] = -4.0
            L[a, b, d] = 4.0
    return Llo, Lhi


print("as trained:", loss_of(Llo, Lhi))
# the first wall in row a (scanning b upward from 1)
for a in range(10):
    walls = [b for b in range(1, 9) if off[a, b + 1] != off[a, b]]
    if not walls:
        continue
    b = walls[0]
    block = [c for c in range(b + 1, 10) if off[a, c] == off[a, b + 1]]
    one = {(a, b + 1): int(f[a, b + 1] - off[a, b + 1] + off[a, b])}
    allb = {(a, c): int(f[a, c] - off[a, c] + off[a, b]) for c in block}
    v1, va = one[(a, b + 1)], [allb[(a, c)] for c in block]
    if not (0 <= v1 <= 99 and all(0 <= v <= 99 for v in va)):
        continue
    keep = {(a, c): int(f[a, c]) for c in block}           # control: the same cells made as sharp, values kept
    print(f"row a={a}: wall between b={b} and b={b + 1} (offsets {int(off[a, b])} | {int(off[a, b + 1])}), block {block}")
    print("   control (sharpened, unmoved):", loss_of(*put(Llo, Lhi, keep)))
    print("   move one cell               :", loss_of(*put(Llo, Lhi, {**keep, **one})))
    print("   move the block              :", loss_of(*put(Llo, Lhi, allb)))
