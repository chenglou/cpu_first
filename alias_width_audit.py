# C5: gate-level re-implementation of the WHOLE design where every gate is the one FA table
# (AND=carry(a,b,0), OR=carry(a,b,1), XOR=sum(a,b,0), NOT=sum(a,1,0), MUX=OR(AND(c,a),AND(NOT c,b))),
# with per-bit critical-path depth tracking. Blocks: carry-save Horner D2B + Kogge-Stone CPA, 3N, radix-4
# speculative entry with borrow-out selection, predicated carry-save Montgomery squarings with a 7-bit FA down
# counter, 26-halving exit, FA-built double-dabble, right-aligned readout. Runs each profile as ONE mixed-T batch.
import json, os, sys, numpy as np
DATA = sys.argv[1]
TABLE = sys.argv[2] if len(sys.argv) > 2 else "01101001,00010111"   # sum8,carry8 over idx=a+2b+4c
TSUM = np.array([int(ch) for ch in TABLE.split(",")[0]], np.uint8)
TCAR = np.array([int(ch) for ch in TABLE.split(",")[1]], np.uint8)
import os
W = int(os.environ.get('WIDTH', 28)); K, ND, NB = W - 2, 8, 8

class V:
    __slots__ = ("b", "d")
    def __init__(self, b, d): self.b = b; self.d = d
    def __getitem__(self, s): return V(self.b[:, s], self.d[s])

def const(Bn, w, val): return V(np.full((Bn, w), val, np.uint8), np.zeros(w, np.int64))
def cat(*vs): return V(np.concatenate([v.b for v in vs], 1), np.concatenate([v.d for v in vs]))
def bcast(v1, w): return V(np.repeat(v1.b, w, 1), np.repeat(v1.d, w))
def shl(v, k): return cat(const(v.b.shape[0], k, 0), v[:-k]) if k else v
def shr(v, k): return cat(v[k:], const(v.b.shape[0], k, 0))

def FA(a, b, c):
    idx = a.b + 2 * b.b + 4 * c.b
    d = np.maximum(np.maximum(a.d, b.d), c.d) + 1
    return V(TSUM[idx], d), V(TCAR[idx], d)
def _z(a): return const(a.b.shape[0], a.b.shape[1], 0)
def _o(a): return const(a.b.shape[0], a.b.shape[1], 1)
def AND(a, b): return FA(a, b, _z(a))[1]
def OR(a, b): return FA(a, b, _o(a))[1]
def XOR(a, b): return FA(a, b, _z(a))[0]
def NOT(a): return FA(a, _o(a), _z(a))[0]
def MUX(c, a, b):   # c ? a : b  (c broadcast to a's width by caller)
    return OR(AND(c, a), AND(NOT(c), b))

def ks_add(a, b, cin):
    """Kogge-Stone adder from FA-gates. a,b: V [B,W]; cin: V [B,1]. returns (sum [B,W], carry_out [B,1])."""
    g = AND(a, b); p = XOR(a, b)
    G = cat(cin, g); P = cat(const(a.b.shape[0], 1, 0), p)       # position -1 holds cin
    n = G.b.shape[1]; dist = 1
    while dist < n:
        Gs = cat(const(a.b.shape[0], dist, 0), G[:-dist]); Ps = cat(const(a.b.shape[0], dist, 0), P[:-dist])
        Gn = OR(G, AND(P, Gs)); Pn = AND(P, Ps)
        keep = np.arange(n) < dist                                   # positions < dist are already final
        G = V(np.where(keep, G.b, Gn.b), np.where(keep, G.d, Gn.d))
        P = V(np.where(keep, P.b, Pn.b), np.where(keep, P.d, Pn.d))
        dist *= 2
    carries = G[:-1]                                                 # c_i = G_{i-1:-1}
    return XOR(p, carries), G[n - 1:n]

def ripple_add(a, b, cin):
    outs = []; c = cin
    for j in range(a.b.shape[1]):
        s, c = FA(a[j:j + 1], b[j:j + 1], c); outs.append(s)
    return cat(*outs), c

def to_int(v): return (v.b.astype(np.int64) << np.arange(v.b.shape[1], dtype=np.int64)).sum(1)

def parse(row):
    ids = row["input_ids"]; iN, iX, iT = ids.index(2), ids.index(3), ids.index(4)
    nd = [t - 7 for t in ids[iN + 1:iX]]; xd = [t - 7 for t in ids[iX + 1:iT]][::-1]; td = [t - 7 for t in ids[iT + 1:]][::-1]
    return nd, xd, td

def d2b(digs, w):
    """carry-save Horner v <- 8v + 2v + d (5->2 rows in 3 FA levels per digit), then KS CPA."""
    Bn = digs.shape[0]
    vs = const(Bn, w, 0); vc = const(Bn, w, 0)
    for i in range(digs.shape[1] - 1, -1, -1):
        d = cat(V(((digs[:, i:i + 1] >> np.arange(4)) & 1).astype(np.uint8), np.zeros(4, np.int64)), const(Bn, w - 4, 0))
        s1, c1 = FA(shl(vs, 3), shl(vc, 3), shl(vs, 1))
        s2, c2 = FA(s1, shl(c1, 1), shl(vc, 1))
        s3, c3 = FA(s2, shl(c2, 1), d)
        vs, vc = s3, shl(c3, 1)
    return ks_add(vs, vc, const(Bn, 1, 0))[0]

def run(rows):
    Bn = len(rows)
    P = [parse(r) for r in rows]
    pad = lambda ds, n: np.array([d + [0] * (n - len(d)) for d in ds])
    Nd, Xd, Td = pad([p[0] for p in P], ND), pad([p[1] for p in P], ND), pad([p[2] for p in P], 2)
    depth = {}
    N = d2b(Nd, W); X = d2b(Xd, W); Tc = d2b(Td, 7)
    pass  # (assert removed for the aliasing audit)
    pass  # (assert removed for the aliasing audit)
    depth["D2B"] = int(max(N.d.max(), X.d.max()))
    N2 = shl(N, 1); N3, _ = ks_add(N, N2, const(Bn, 1, 0))
    depth["3N"] = int(N3.d.max())
    notk = [NOT(N), NOT(N2), NOT(N3)]
    s = X
    for _ in range(K // 2):
        t = shl(s, 2)
        dk, nn = [], []
        for k in range(3):
            dd, co = ks_add(t, notk[k], const(Bn, 1, 1)); dk.append(dd); nn.append(co)   # co=1 <=> t >= (k+1)N
        s = MUX(bcast(nn[0], W), dk[0], t)
        s = MUX(bcast(nn[1], W), dk[1], s)
        s = MUX(bcast(nn[2], W), dk[2], s)
    ref = np.array([(r["x"] << K) % r["modulus"] for r in rows])
    pass  # (assert removed for the aliasing audit)
    depth["entry_end"] = int(s.d.max())
    cnt = Tc; Tmax = int(to_int(Tc).max()); step_depths = []
    for step in range(Tmax):
        o1 = OR(cnt[0:1], cnt[1:2]); o2 = OR(cnt[2:3], cnt[3:4]); o3 = OR(cnt[4:5], cnt[5:6])
        active = OR(OR(o1, o2), OR(o3, cnt[6:7]))
        pp = [AND(bcast(s[i:i + 1], W), s) for i in range(K)]
        Ss = const(Bn, W, 0); Sc = const(Bn, W, 0)
        for i in range(K):
            u, v = FA(Ss, Sc, pp[i])
            pass  # (assert removed for the aliasing audit)
            qn = AND(bcast(u[0:1], W), N)
            s2, c2 = FA(u, shl(v, 1), qn)
            pass  # (assert removed for the aliasing audit)
            Ss = shr(s2, 1); Sc = c2
        snew, co = ks_add(Ss, Sc, const(Bn, 1, 0))
        before = s.d.max()
        s = MUX(bcast(active, W), snew, s)
        step_depths.append(int(s.d.max() - before))
        cnt, _ = ripple_add(cnt, bcast(active, 7), const(Bn, 1, 0))          # cnt += active * 1111111 (== -active)
    depth["per_step"] = sorted(set(step_depths)); depth["loop_end"] = int(s.d.max())
    Ss = s; Sc = const(Bn, W, 0)
    for _ in range(K):
        q = XOR(Ss[0:1], Sc[0:1]); qn = AND(bcast(q, W), N)
        u, v = FA(Ss, Sc, qn)
        Ss = shr(u, 1); Sc = v
    y, co = ks_add(Ss, Sc, const(Bn, 1, 0))
    yv = to_int(y)
    pass  # (assert removed for the aliasing audit)
    depth["exit_end"] = int(y.d.max())
    reg = const(Bn, 4 * NB, 0)
    for bit in range(23, -1, -1):
        nibs = []
        for k in range(NB):
            n0, n1, n2, n3 = (reg[4 * k + j:4 * k + j + 1] for j in range(4))
            cond = OR(n3, AND(n2, OR(n1, n0)))
            z = const(Bn, 1, 0)
            a0, c = FA(n0, cond, z); a1, c = FA(n1, cond, c); a2, c = FA(n2, z, c); a3, c = FA(n3, z, c)
            nibs += [a0, a1, a2, a3]
        reg = cat(y[bit:bit + 1], cat(*nibs)[:-1])                            # shift left, bring in next MSB
    depth["B2D_end"] = int(reg.d.max())
    digs = np.stack([(reg.b[:, 4 * k:4 * k + 4].astype(np.int64) << np.arange(4)).sum(1) for k in range(NB)], 1)
    ok = sum(int(list(digs[e, :len(r["labels"])]) == [t - 7 for t in r["labels"]][::-1]) for e, r in enumerate(rows))
    return ok, depth


def load(name): return [json.loads(l) for l in open(os.path.join(DATA, name + ".jsonl"))]
train1 = [r for r in load("train") if r["time_steps"] == 1]
ok, _ = run(train1)
print(f"W={W}: all {len(train1)} training T=1 rows: {ok}/{len(train1)} exact", flush=True)
for prof in ["depth_t_", "depth_ood_n_t_"]:
    rows = []
    for T in [1, 2, 4, 8, 16, 32, 64]:
        rows += load(f"{prof}{T}")
    ok, _ = run(rows)
    by = {}
    for r in rows:
        pass
    print(f"W={W}: {prof}* (T=1..64, both bit sizes): {ok}/{len(rows)} exact", flush=True)
