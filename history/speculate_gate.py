"""Learning the gate by speculative execution over hypothesis space — no axioms, no gradients, labels only.

The binary design (panel 'BitCPU') builds every gate from ONE learned 3->2 truth table (8 sum bits + 8 carry
bits = 65,536 candidate tables): AND=carry(a,b,0), OR=carry(a,b,1), XOR=sum(a,b,0), NOT=sum(a,1,0).
Here all 65,536 candidate machines run at once, bitsliced: bit j of a uint64 word belongs to candidate
64*w + j, so one bitwise op advances 64 hypotheses (the user's 'each bit of the 32 bits as a single bit',
used for learning rather than inference). A candidate survives iff its machine reproduces the H1 answer on
every one of a few T=1 training rows. Netlist ported from panel/critique-binary-cpu/gate_sim.py (T=1 path:
Horner decimal->binary, radix-4 speculative entry x*2^26 mod N, one carry-save Montgomery squaring,
26-halving exit, double-dabble binary->BCD).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import h1

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=4)
p.add_argument("--offset", type=int, default=0)
args = p.parse_args()

W, K, ND, NB = 28, 26, 8, 8
NC = 1 << 16            # candidate tables: c = sum8 | carry8 << 8, table index m = a + 2b + 4c
NW = NC // 64           # uint64 words
ALL = np.uint64(0xFFFFFFFFFFFFFFFF)
cand = np.arange(NC, dtype=np.uint64).reshape(NW, 64)
shifts = np.arange(64, dtype=np.uint64)


def pack(bits):  # bits [NW, 64] of 0/1 -> [NW] uint64
    return (bits.astype(np.uint64) << shifts).sum(1).astype(np.uint64)


TS = [pack((cand >> np.uint64(m)) & np.uint64(1)) for m in range(8)]
TC = [pack((cand >> np.uint64(8 + m)) & np.uint64(1)) for m in range(8)]
TS = [t[:, None, None] for t in TS]
TC = [t[:, None, None] for t in TC]

tr = h1.load("train")
sel = np.where(tr["T"] == 1)[0][args.offset: args.offset + args.rows]
Nv, Xv, Yv = tr["N"][sel], tr["X"][sel], tr["Y"][sel]
B = len(sel)


def const(w, val):
    return np.full((NW, B, w), ALL if val else np.uint64(0), np.uint64)


def from_bits(bits):  # bits [B, w] 0/1, identical for every candidate
    one = np.where(bits[None].astype(bool), ALL, np.uint64(0)).astype(np.uint64)
    return np.broadcast_to(one, (NW,) + one.shape[1:]).copy()


cat = lambda *vs: np.concatenate(vs, axis=2)
shl = lambda v, k: cat(const(k, 0), v[:, :, :-k]) if k else v
shr = lambda v, k: cat(v[:, :, k:], const(k, 0))
bcast = lambda v1, w: np.repeat(v1, w, axis=2)


def FA(a, b, c):
    na, nb, nc = ~a, ~b, ~c
    s = np.zeros_like(a)
    cy = np.zeros_like(a)
    for m in range(8):
        mask = (a if m & 1 else na) & (b if m & 2 else nb) & (c if m & 4 else nc)
        s |= mask & TS[m]
        cy |= mask & TC[m]
    return s, cy


AND = lambda a, b: FA(a, b, np.zeros_like(a))[1]
OR = lambda a, b: FA(a, b, np.full_like(a, ALL))[1]
XOR = lambda a, b: FA(a, b, np.zeros_like(a))[0]
NOT = lambda a: FA(a, np.full_like(a, ALL), np.zeros_like(a))[0]
MUX = lambda c, a, b: OR(AND(c, a), AND(NOT(c), b))


def ks_add(a, b, cin):
    g, pp = AND(a, b), XOR(a, b)
    G, Pp = cat(cin, g), cat(const(1, 0), pp)
    n, dist = G.shape[2], 1
    while dist < n:
        Gs, Ps = cat(const(dist, 0), G[:, :, :-dist]), cat(const(dist, 0), Pp[:, :, :-dist])
        Gn, Pn = OR(G, AND(Pp, Gs)), AND(Pp, Ps)
        keep = np.arange(n) < dist
        G, Pp = np.where(keep, G, Gn), np.where(keep, Pp, Pn)
        dist *= 2
    return XOR(pp, G[:, :, :-1]), G[:, :, n - 1:n]


def d2b(digs, w):
    vs, vc = const(w, 0), const(w, 0)
    for i in range(digs.shape[1] - 1, -1, -1):
        d = cat(from_bits((digs[:, i:i + 1] >> np.arange(4)) & 1), const(w - 4, 0))
        s1, c1 = FA(shl(vs, 3), shl(vc, 3), shl(vs, 1))
        s2, c2 = FA(s1, shl(c1, 1), shl(vc, 1))
        s3, c3 = FA(s2, shl(c2, 1), d)
        vs, vc = s3, shl(c3, 1)
    return ks_add(vs, vc, const(1, 0))[0]


t0 = time.time()
N = d2b(h1.digits(Nv, ND), W)
X = d2b(h1.digits(Xv, ND), W)
N2 = shl(N, 1)
N3, _ = ks_add(N, N2, const(1, 0))
notk = [NOT(N), NOT(N2), NOT(N3)]
s = X
for _ in range(K // 2):  # entry: x * 2^26 mod N, radix-4 speculative steps
    t = shl(s, 2)
    dk, nn = [], []
    for k in range(3):
        dd, co = ks_add(t, notk[k], const(1, 1))
        dk.append(dd)
        nn.append(co)
    s = MUX(bcast(nn[0], W), dk[0], t)
    s = MUX(bcast(nn[1], W), dk[1], s)
    s = MUX(bcast(nn[2], W), dk[2], s)
pp = [AND(bcast(s[:, :, i:i + 1], W), s) for i in range(K)]  # one Montgomery squaring (T = 1)
Ss, Sc = const(W, 0), const(W, 0)
for i in range(K):
    u, v = FA(Ss, Sc, pp[i])
    qn = AND(bcast(u[:, :, 0:1], W), N)
    s2, c2 = FA(u, shl(v, 1), qn)
    Ss, Sc = shr(s2, 1), c2
s, _ = ks_add(Ss, Sc, const(1, 0))
Ss, Sc = s, const(W, 0)
for _ in range(K):  # exit MonPro(s, 1): 26 halvings mod N
    q = XOR(Ss[:, :, 0:1], Sc[:, :, 0:1])
    u, v = FA(Ss, Sc, AND(bcast(q, W), N))
    Ss, Sc = shr(u, 1), v
y, _ = ks_add(Ss, Sc, const(1, 0))
reg = const(4 * NB, 0)
for bit in range(23, -1, -1):  # double-dabble binary -> BCD
    nibs = []
    for k in range(NB):
        n0, n1, n2, n3 = (reg[:, :, 4 * k + j:4 * k + j + 1] for j in range(4))
        cond = OR(n3, AND(n2, OR(n1, n0)))
        z = const(1, 0)
        a0, c = FA(n0, cond, z)
        a1, c = FA(n1, cond, c)
        a2, c = FA(n2, z, c)
        a3, c = FA(n3, z, c)
        nibs += [a0, a1, a2, a3]
    reg = cat(y[:, :, bit:bit + 1], cat(*nibs)[:, :, :-1])

# survivors: candidates whose BCD digits equal the label digits on every scored position of every row
alive = np.full(NW, ALL, np.uint64)
yd, nd = h1.digits(Yv, NB), h1.ndigits(Yv)
for e in range(B):
    for k in range(nd[e]):
        for j in range(4):
            want = ALL if (yd[e, k] >> j) & 1 else np.uint64(0)
            alive &= ~(reg[:, e, 4 * k + j] ^ want)
surv = [int(w * 64 + j) for w in range(NW) for j in range(64) if (int(alive[w]) >> j) & 1]
fmt = lambda c: ("".join(str((c >> m) & 1) for m in range(8)), "".join(str((c >> (8 + m)) & 1) for m in range(8)))
print(f"{NC} candidate machines x {B} T=1 rows, bitsliced: {time.time()-t0:.1f}s  survivors: {len(surv)}")
for c in surv[:8]:
    print(f"  sum={fmt(c)[0]} carry={fmt(c)[1]}  (true full adder: sum=01101001 carry=00010111)")
