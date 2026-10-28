"""Exhaustive over x in [1, N): the exact Montgomery squaring chain (R = 2^K, s0 = x 2^K mod N), T <= 64, for the
largest 24-bit Blum moduli. Reports the largest unreduced operand reached and saves rows reaching s >= thr.
argv: K thr n_moduli out_json"""
import json, sys, time
import numpy as np

K, thr, nmod = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
R, mask = 1 << K, (1 << K) - 1


def is_prime(n):
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return n >= 2


ps = [p for p in range(4095, 3000, -1) if p % 4 == 3 and is_prime(p)]
Ns = sorted({p * q for p in ps for q in ps if p < q and (p * q).bit_length() == 24}, reverse=True)[:nmod]
hits = []
for N in Ns:
    t0 = time.time()
    ninv = (-pow(N, -1, R)) & mask
    x = np.arange(1, N, dtype=np.int64)
    s = (x * (R % N)) % N  # x 2^K mod N without overflow: (R mod N) < 2^24, x < 2^24
    best = np.zeros_like(s)
    first = np.full(s.shape, -1, dtype=np.int64)
    for t in range(64):
        lo = (s * s) & mask
        m = (lo * ninv) & mask
        s = (s * s + m * N) >> K
        best = np.maximum(best, s)
        first = np.where((first < 0) & (s >= thr), t + 1, first)
    idx = np.nonzero(first > 0)[0]
    print(f"N={N}: max s over all x, T<=64 = {best.max()} ({best.max() / N:.4f} N); rows reaching s >= {thr}: "
          f"{len(idx)} ({time.time() - t0:.0f}s)", flush=True)
    for i in idx[:200]:
        hits.append({"N": int(N), "x": int(x[i]), "t": int(first[i])})
json.dump(hits, open(sys.argv[4], "w"))
