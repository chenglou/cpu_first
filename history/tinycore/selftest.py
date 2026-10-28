"""With the true tables every variant is exact: every cell (slot, repeat, mult), choice (given, borrow, est, estc) and
start (prefix, full), at a few sizes; both with the exact cell and with learned logits set to the truth.

    python -m tinycore.selftest            (a few seconds)
"""
import time

import torch

from .core import Config, Core, Data


def est_truth(m):
    B, w = m.cfg.B, m.cfg.w
    with torch.no_grad():
        m.g.zero_()
        for ridx in range(B ** m.kr):
            Rp = sum(((ridx // B ** i) % B) * B ** (i + (w + 1 - m.kr)) for i in range(m.kr))
            for nidx in range(B ** m.kn):
                N = sum(((nidx // B ** i) % B) * B ** (i + (w - m.kn)) for i in range(m.kn))
                q = min(Rp // N, B - 1) if N > 0 else 0
                m.g[ridx * B ** m.kn + nidx, q] = 10.0


def main():
    torch.set_num_threads(2)
    bad = 0
    for B, w, k in [(2, 3, 2), (2, 4, 3), (3, 2, 2), (4, 2, 2), (3, 3, 1)]:
        for cell in ("slot", "repeat", "mult"):
            for choice in ("given", "borrow", "est", "estc"):
                for start in ("prefix", "full"):
                    d = Data(B, w, k, start=start)
                    m = Core(Config(B=B, w=w, k=k, cell=cell, choice=choice, exact_cell=True, start=start))
                    est_truth(m)
                    t = time.time()
                    e, eh = m.exact(d), m.exact(d, hard=True)
                    m2 = Core(Config(B=B, w=w, k=k, cell=cell, choice=choice, start=start))
                    est_truth(m2)
                    m2.set_truth(12.0)
                    e2 = m2.exact(d)
                    ok = min(e, eh, e2) == 1.0
                    bad += not ok
                    print(f"B{B} w{w} k{k} {cell:6s} {choice:6s} {start:6s} X={d.X:5d} exact {e:.3f} hard {eh:.3f} "
                          f"learned-at-truth {e2:.3f} {'' if ok else 'FAIL'} ({time.time() - t:.2f}s)")
    print("all exact" if not bad else f"{bad} FAILED")


if __name__ == "__main__":
    main()
