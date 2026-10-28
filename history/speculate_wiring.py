"""Learning the *wiring* by speculation: every candidate datapath runs at once, one per bit lane.

Each lane of a uint64 word is a different machine. A wiring choice is a per-lane multiplexer: compute every
alternative signal, keep the one this lane's select mask names (a CPU's "compute all ALU ops, mux the result",
used across hypotheses instead of instructions). The gate is the full adder (identified from labels by
speculate_gate.py). Factors (31,104 joint candidates, ~15 bits):

  d2b    Horner step v <- v<<a + v<<b + digit, (a,b) in {(3,1) =x10, (2,1) =x6, (3,2) =x12, (1,0) =x3}
  qsrc   Montgomery quotient bit source in {u0, v0, Ss0, u0^v0}
  vsh    shift of the carry row into the second compressor in {0,1,2}
  osh    final right shift (divide by 2^k) in {0,1,2}
  csh    carry row hand-off in {as is, <<1, >>1}
  ppdir  multiplier scan order in {LSB first, MSB first}
  exq    exit (MonPro(s,1)) quotient bit in {Ss0^Sc0, Ss0, Sc0, Ss0&Sc0}
  b2d    double-dabble rule: threshold in {>=5, >=4, >=6} x correction in {+3, +2, +6}

Two searches: joint (all factors against T=1 labels), and factored (d2b+b2d first from the label-free
round-trip b2d(d2b(v)) == v on the prompt's own N and x, then the core factors against labels).
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np

import h1

p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=4)
args = p.parse_args()

W, K, ND, NB = 28, 26, 8, 8
ALL, ZERO = np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0)
FACTORS = {"d2b": 4, "qsrc": 4, "vsh": 3, "osh": 3, "csh": 3, "ppdir": 2, "exq": 4, "b2d": 9}
TRUE = {"d2b": 0, "qsrc": 0, "vsh": 1, "osh": 1, "csh": 0, "ppdir": 0, "exq": 0, "b2d": 0}
D2B_SH = [(3, 1), (2, 1), (3, 2), (1, 0)]


class Lanes:
    """A set of candidate machines; `choice[f]` is each lane's option for factor f."""

    def __init__(self, combos: list[dict], rows: int):
        n = len(combos)
        self.n, self.NW, self.B = n, (n + 63) // 64, rows
        self.combos = combos
        self.sel = {}
        for f, k in FACTORS.items():
            vals = np.array([c.get(f, 0) for c in combos] + [-1] * (self.NW * 64 - n))
            self.sel[f] = [self.pack(vals == o) for o in range(k)]

    def pack(self, bits):
        b = bits.reshape(self.NW, 64).astype(np.uint64) << np.arange(64, dtype=np.uint64)
        return b.sum(1).astype(np.uint64)[:, None, None]

    def const(self, w, val):
        return np.full((self.NW, self.B, w), ALL if val else ZERO, np.uint64)

    def from_bits(self, bits):
        one = np.where(bits[None].astype(bool), ALL, ZERO).astype(np.uint64)
        return np.broadcast_to(one, (self.NW,) + one.shape[1:]).copy()

    def mux(self, f, options):  # per-lane select among alternative signals
        out = np.zeros_like(options[0])
        for o, sig in enumerate(options):
            out |= self.sel[f][o] & sig
        return out


def build(L: Lanes, Nv, Xv):
    cat = lambda *vs: np.concatenate(vs, axis=2)
    shl = lambda v, k: cat(L.const(k, 0), v[:, :, :-k]) if k else v
    shr = lambda v, k: cat(v[:, :, k:], L.const(k, 0)) if k else v
    bc = lambda v1, w: np.repeat(v1, w, axis=2)
    FA = lambda a, b, c: (a ^ b ^ c, (a & b) | (c & (a ^ b)))

    def ks_add(a, b, cin):
        g, pp = a & b, a ^ b
        G, P = cat(cin, g), cat(L.const(1, 0), pp)
        n, dist = G.shape[2], 1
        while dist < n:
            Gs, Ps = cat(L.const(dist, 0), G[:, :, :-dist]), cat(L.const(dist, 0), P[:, :, :-dist])
            keep = np.arange(n) < dist
            G, P = np.where(keep, G, G | (P & Gs)), np.where(keep, P, P & Ps)
            dist *= 2
        return pp ^ G[:, :, :-1], G[:, :, n - 1:n]

    def d2b(digs, w):
        vs, vc = L.const(w, 0), L.const(w, 0)
        for i in range(digs.shape[1] - 1, -1, -1):
            d = cat(L.from_bits((digs[:, i:i + 1] >> np.arange(4)) & 1), L.const(w - 4, 0))
            a = L.mux("d2b", [shl(vs, s[0]) for s in D2B_SH])
            ac = L.mux("d2b", [shl(vc, s[0]) for s in D2B_SH])
            b = L.mux("d2b", [shl(vs, s[1]) for s in D2B_SH])
            bcv = L.mux("d2b", [shl(vc, s[1]) for s in D2B_SH])
            s1, c1 = FA(a, ac, b)
            s2, c2 = FA(s1, shl(c1, 1), bcv)
            s3, c3 = FA(s2, shl(c2, 1), d)
            vs, vc = s3, shl(c3, 1)
        return ks_add(vs, vc, L.const(1, 0))[0]

    def b2d(y):
        reg = L.const(4 * NB, 0)
        for bit in range(23, -1, -1):
            nibs = []
            for k in range(NB):
                n0, n1, n2, n3 = (reg[:, :, 4 * k + j:4 * k + j + 1] for j in range(4))
                ge5, ge4, ge6 = n3 | (n2 & (n1 | n0)), n3 | n2, n3 | (n2 & n1)
                conds = [ge5, ge4, ge6]
                cond = L.mux("b2d", [conds[o // 3] for o in range(9)])
                adds = []
                for o in range(9):
                    corr = o % 3  # +3 -> bits (0,1); +2 -> bit 1; +6 -> bits (1,2)
                    adds.append(((1, 1, 0), (0, 1, 0), (0, 1, 1))[corr])
                z = L.const(1, 0)
                addbit = [L.mux("b2d", [cond if adds[o][j] else z for o in range(9)]) for j in range(3)]
                a0, c = FA(n0, addbit[0], z)
                a1, c = FA(n1, addbit[1], c)
                a2, c = FA(n2, addbit[2], c)
                a3, c = FA(n3, z, c)
                nibs += [a0, a1, a2, a3]
            reg = cat(y[:, :, bit:bit + 1], cat(*nibs)[:, :, :-1])
        return reg

    return cat, shl, shr, bc, FA, ks_add, d2b, b2d


def run_machine(L: Lanes, Nv, Xv):
    cat, shl, shr, bc, FA, ks_add, d2b, b2d = build(L, Nv, Xv)
    N, X = d2b(h1.digits(Nv, ND), W), d2b(h1.digits(Xv, ND), W)
    N2 = shl(N, 1)
    N3, _ = ks_add(N, N2, L.const(1, 0))
    notk = [~N, ~N2, ~N3]
    s = X
    for _ in range(K // 2):
        t = shl(s, 2)
        dk, nn = zip(*[ks_add(t, notk[k], L.const(1, 1)) for k in range(3)])
        for k in range(3):
            c = bc(nn[k], W)
            s = (c & dk[k]) | (~c & (t if k == 0 else s))
    order = [list(range(K)), list(range(K - 1, -1, -1))]
    pps = [[bc(s[:, :, i:i + 1], W) & s for i in o] for o in order]
    Ss, Sc = L.const(W, 0), L.const(W, 0)
    for i in range(K):
        u, v = FA(Ss, Sc, L.mux("ppdir", [pps[0][i], pps[1][i]]))
        q = L.mux("qsrc", [u[:, :, 0:1], v[:, :, 0:1], Ss[:, :, 0:1], u[:, :, 0:1] ^ v[:, :, 0:1]])
        s2, c2 = FA(u, L.mux("vsh", [shl(v, k) for k in range(3)]), bc(q, W) & N)
        Ss = L.mux("osh", [shr(s2, k) for k in range(3)])
        Sc = L.mux("csh", [c2, shl(c2, 1), shr(c2, 1)])
    s, _ = ks_add(Ss, Sc, L.const(1, 0))
    Ss, Sc = s, L.const(W, 0)
    for _ in range(K):
        a0, b0 = Ss[:, :, 0:1], Sc[:, :, 0:1]
        q = L.mux("exq", [a0 ^ b0, a0, b0, a0 & b0])
        u, v = FA(Ss, Sc, bc(q, W) & N)
        Ss, Sc = shr(u, 1), v
    y, _ = ks_add(Ss, Sc, L.const(1, 0))
    return b2d(y)


def survivors(L: Lanes, reg, Yv):
    alive = np.full(L.NW, ALL, np.uint64)
    yd, nd = h1.digits(Yv, NB), h1.ndigits(Yv)
    for e in range(L.B):
        for k in range(nd[e]):
            for j in range(4):
                want = ALL if (yd[e, k] >> j) & 1 else ZERO
                alive &= ~(reg[:, e, 4 * k + j] ^ want)
    return [i for i in range(L.n) if (int(alive[i // 64]) >> (i % 64)) & 1]


def roundtrip_survivors(L: Lanes, vals):
    cat, shl, shr, bc, FA, ks_add, d2b, b2d = build(L, vals, vals)
    reg = b2d(d2b(h1.digits(vals, ND), W))
    return survivors(L, reg, vals)


tr = h1.load("train")
idx = np.where(tr["T"] == 1)[0][: args.rows]
Nv, Xv, Yv = tr["N"][idx], tr["X"][idx], tr["Y"][idx]
names = list(FACTORS)
everything = [dict(zip(names, c)) for c in itertools.product(*[range(FACTORS[f]) for f in names])]
show = lambda c: {f: c[f] for f in names if c[f] != TRUE[f]} or "the reference wiring"

t0 = time.time()
L = Lanes(everything, args.rows)
surv = survivors(L, run_machine(L, Nv, Xv), Yv)
print(f"joint: {len(everything)} candidate machines vs {args.rows} T=1 labels, {time.time()-t0:.1f}s -> "
      f"{len(surv)} survivor(s): {[show(everything[i]) for i in surv][:6]}")

t0 = time.time()
io = [c for c in everything if all(c[f] == 0 for f in names if f not in ("d2b", "b2d"))]
io_vals = np.concatenate([tr["N"][: args.rows], tr["X"][: args.rows]])
Lio = Lanes(io, len(io_vals))
io_surv = [io[i] for i in roundtrip_survivors(Lio, io_vals)]
print(f"factored stage A (no labels): {len(io)} converter wirings vs round-trip on {len(io_vals)} prompt numbers -> "
      f"{len(io_surv)} survivor(s): {[(c['d2b'], c['b2d']) for c in io_surv]}")
core = [dict(c, d2b=s["d2b"], b2d=s["b2d"]) for s in io_surv
        for c in everything if c["d2b"] == 0 and c["b2d"] == 0]
Lc = Lanes(core, args.rows)
cs = survivors(Lc, run_machine(Lc, Nv, Xv), Yv)
print(f"factored stage B (labels): {len(core)} core wirings -> {len(cs)} survivor(s): "
      f"{[show(core[i]) for i in cs][:6]}   total {time.time()-t0:.1f}s, "
      f"{len(io) + len(core)} machines evaluated instead of {len(everything)}")
