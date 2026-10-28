"""Can identities alone teach the shared subtractor, without a trusted adder? Train the base-10 shared subtractor
(r, m, borrow) -> (digit, borrow) on random 4-digit numbers from a - a = 0, a - 0 = a and (a - b) - c = (a - c) - b
only, then score its rows against true subtraction and the 7-bit datapath given the true quotient digits.

Borrow-free subtraction (digit by digit, mod 10) satisfies all three identities too, so they do not pin it; with an
exact adder, (a - b) + b = a does (b10.label_free).

    python -m tinycore.identities            (about 10 seconds)
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .b10 import make, sub_once
from .core import Config, Core, digits


def main(seed=0, steps=1200, B=10, bits=7):
    torch.set_num_threads(2)
    w = 3
    P = w + 1
    m = Core(Config(B=B, w=w, k=w, cell="mult", choice="borrow"), seed=seed)
    o = torch.optim.Adam([m.a, m.b], lr=0.05)
    rng = np.random.default_rng(seed)
    test = make(B, bits, w, 1500, 10_007, "prefix", True)
    oh = lambda d: F.one_hot(d, B).double()

    def ce(p, target):
        return -p.clamp_min(1e-12).log().gather(-1, target[..., None]).mean()

    t = time.time()
    for it in range(steps + 1):
        a, b, c = (torch.from_numpy(digits(rng.integers(0, B ** P, 256), B, P)) for _ in range(3))
        TD, TC = m.tables()
        L = ce(sub_once(m, oh(a), a, TD, TC), torch.zeros_like(a))                 # a - a = 0
        L = L + ce(sub_once(m, oh(a), torch.zeros_like(a), TD, TC), a)             # a - 0 = a
        abc = sub_once(m, sub_once(m, oh(a), b, TD, TC), c, TD, TC)                 # (a - b) - c
        acb = sub_once(m, sub_once(m, oh(a), c, TD, TC), b, TD, TC)                 # (a - c) - b
        L = L + ce(abc, acb.detach().argmax(-1)) + ce(acb, abc.detach().argmax(-1))
        o.zero_grad()
        L.backward()
        o.step()
        if it % 300 == 0:
            print(f"step {it}: loss {L.item():.3f}  exact on the 7-bit datapath given the true digits "
                  f"{m.exact_given(test):.3f}", flush=True)
    TD, TC = m.tables()
    r, mm, cc = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
    right = ((TD.argmax(-1) == (r - mm - cc) % B) & (TC.argmax(-1) == (r - mm - cc < 0).long())).double().mean()
    borrowless = (TD.argmax(-1)[:, :, 0] == (r - mm)[:, :, 0] % B).double().mean()
    print(f"seed {seed}: rows right {right:.3f}; carry-0 rows equal to borrow-free subtraction {borrowless:.3f} "
          f"({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
