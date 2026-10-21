"""The ladder's missed-carry stall (tiny core): train zero+peanoboot to the stuck state, then look at the stuck cell's
distributions and at the bootstrap loss as its hi digit's mass slides from the wrong digit to the true one.
  python probe_carry.py SEED STEPS"""
import sys
import torch
import tiny

seed, steps = int(sys.argv[1]), int(sys.argv[2])
out = tiny.train(["zero", "peanoboot"], seed=seed, steps=steps, quiet=True, log_every=steps)
Llo, Lhi = out["logits"]
f = out["f"]
wrong = [(a, b) for a in range(10) for b in range(10) if f[a, b] != a * b]
print("wrong", [(a, b, int(f[a, b])) for a, b in wrong])
for a, b in wrong:
    plo, phi = Llo[a, b].softmax(-1), Lhi[a, b].softmax(-1)
    print(f"cell ({a},{b}) true {a*b}: lo top {plo.topk(2)} hi top {phi.topk(2)}")
N, x = tiny.universe(tiny.ROWS, 1000 + seed)
a, b = wrong[0]
tl, th = (a * b) % 10, (a * b) // 10
wl = int(f[a, b]) // 10
for lam in [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]:
    Lh = Lhi.clone()
    p = Lh[a, b].softmax(-1)
    q = p.clone()
    q[wl] = p[wl] * (1 - lam)
    q[th] = p[th] + p[wl] * lam
    Lh[a, b] = q.clamp_min(1e-9).log()
    Lo_, Hi_ = Llo.softmax(-1), Lh.softmax(-1)
    fh = Llo.argmax(-1) + 10 * Lh.argmax(-1)
    l, ok = tiny.lesson("peanoboot", N[:1024], x[:1024], Lo_, Hi_, fh)
    print(f"move {lam:.2f} of hi({a},{b})'s mass from {wl} to {th}: bootstrap loss {float(l):.5f} hard pass {ok:.4f}")
