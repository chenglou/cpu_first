"""Which complete tables pass which lessons, base 10, on the H1 rows' own numbers (ANCHORS.md §2.3).

ADDER.md §3's table, extended with the lessons of this study. Each candidate is a complete 200-row table; every lesson
runs on the candidate's own numbers as training would: the link inputs of its own hard division of the rows' x^2, the
rows' N and x, the answers through its own machine. The share of instances passed is printed per lesson.

    python landscape.py [ROWS]
"""

from __future__ import annotations

import sys

import torch

import core

LESSONS = ["lc", "lcc", "dc", "less", "more", "xmod", "zero", "self", "neg2", "comm", "wrap", "chain", "dec",
           "pref", "colord", "roword", "bijc", "bijr", "ans1", "ans2"]


def checks(mc, Td, Tb, d):
    """Every lesson's hard pass rate for one complete table on the rows d."""
    B, P, W = mc.B, mc.P, mc.W
    N, x, T, y = d["N"], d["x"], d["T"], d["y"]
    X = len(N)
    M, xd = mc.digits(N), mc.digits(x)
    Z = torch.zeros_like(M)
    out = {}
    _, ins, lb = mc.divide(x * x, M, Td, Tb)
    K = ins.shape[1] * ins.shape[2]
    D = ins.reshape(X * K, P)
    bl = lb.reshape(X * K)
    Mr = M.repeat_interleave(K, 0)
    E, be, rows_e = mc.sub(D, Mr, Td, Tb)
    C, bc, _ = mc.sub(Z, M, Td, Tb)
    Cr, bcr = C.repeat_interleave(K, 0), bc.repeat_interleave(K, 0)
    Fd, bf, _ = mc.sub(E, Cr, Td, Tb)
    out["lc"] = (Fd == D).all(1)
    out["lcc"] = out["lc"] & (be == bcr - bf)
    back, carry = core.add_hard(E, Mr, B)
    out["dc"] = (back == D).all(1) & (carry == be)
    out["less"] = mc.sub(xd, M, Td, Tb)[1] == 1
    out["more"] = mc.sub(M, xd, Td, Tb)[1] == 0
    out["xmod"] = (mc.divide(x, M, Td, Tb, steps=1)[0] == xd).all(1)
    Rg = ins[:, :, 0].reshape(X * W, P)
    Yz, cz, _ = mc.sub(Rg, torch.zeros_like(Rg), Td, Tb)
    out["zero"] = (Yz == Rg).all(1) & (cz == 0)
    Ys, cs, _ = mc.sub(Rg, Rg, Td, Tb)
    out["self"] = (Ys == 0).all(1) & (cs == 0)
    Y2, b2, _ = mc.sub(Z, C, Td, Tb)
    out["neg2"] = (Y2 == M).all(1) & (b2 == bc)
    MW = M.repeat_interleave(W, 0)
    CW, bcw = C.repeat_interleave(W, 0), bc.repeat_interleave(W, 0)
    A1, b1, _ = mc.sub(Rg, CW, Td, Tb)
    CR, bcR, _ = mc.sub(torch.zeros_like(Rg), Rg, Td, Tb)
    A2, b2_, _ = mc.sub(MW, CR, Td, Tb)
    out["comm"] = (A1 == A2).all(1) & (bcw - b1 == bcR - b2_)
    Ex = mc.sub(xd, M, Td, Tb)[0]
    out["wrap"] = mc.sub(Ex, M, Td, Tb)[1] == 0
    sel = bl == 1
    out["chain"] = mc.sub(E[sel], Mr[sel], Td, Tb)[1] == 0
    out["dec"] = (mc.sub(E, D, Td, Tb)[1] == 1) == (bl == 0)
    pref = []
    for j in range(P):
        differ = (xd[:, j:] != M[:, j:]).any(1)
        pref.append((mc.sub(xd[:, j:], M[:, j:], Td, Tb)[1] == differ.long()) & (mc.sub(M[:, j:], xd[:, j:], Td, Tb)[1] == 0))
    out["pref"] = torch.stack(pref, 1).all(1)
    used = torch.bincount(rows_e.flatten(), minlength=2 * B * B) > 0
    for name, (lo, hi) in zip(("colord", "roword"), order_pairs(Tb, used, B)):
        out[name] = Td[lo] < Td[hi]
    tdv = Td.view(B, B, 2)
    out["bijc"] = torch.tensor([len(set(tdv[:, m, c].tolist())) == B for m in range(B) for c in range(2)])
    out["bijr"] = torch.tensor([len(set(tdv[r, :, c].tolist())) == B for r in range(B) for c in range(2)])
    for t in (1, 2):
        s = T == t
        if s.any():
            out[f"ans{t}"] = mc.run(x[s], torch.full_like(x[s], t), M[s], Td, Tb) == (x[s] ** 2 % N[s] if t == 1 else
                                                                                   pow_mod(x[s], 4, N[s]))
    return {k: float(v.float().mean()) if v.numel() else float("nan") for k, v in out.items()}


def pow_mod(x, e, N):
    return torch.tensor([pow(int(a), e, int(n)) for a, n in zip(x.tolist(), N.tolist())])


def order_pairs(Tb, used, B=10):
    """The order laws' pairs (lower, upper), as in submission_anchors.order_pairs, any base."""
    n = 2 * B * B
    ent = torch.arange(n)
    r, m, c = ent // (2 * B), (ent // 2) % B, ent % 2
    e0 = ent.repeat_interleave(B)
    k = torch.arange(B).repeat(n)
    out = []
    e1 = k * 2 * B + (e0 % (2 * B))
    ok = (r[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb, lt = Tb[a], Tb[b], Tb[(r[a] * B + r[b]) * 2]
    first = (wa < wb) | ((wa == wb) & (lt == 1))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    e1 = (r[e0] * B + k) * 2 + c[e0]
    ok = (m[e0] < k) & used[e0] & used[e1]
    a, b = e0[ok], e1[ok]
    wa, wb, lt = Tb[a], Tb[b], Tb[(m[a] * B + m[b]) * 2]
    first = (wa < wb) | ((wa == wb) & (lt == 0))
    out.append((torch.where(first, a, b), torch.where(first, b, a)))
    return out


def main(n):
    mc = core.Machine(10, 8)
    d = core.h1_rows(n, seed=0)
    T1 = core.h1_rows(None)
    sel = (T1["T"] <= 2)
    for k in d:                                                          # every T <= 2 row of the data, for answers
        d[k] = torch.cat([d[k], T1[k][sel][:512]])
    fam = core.families(10)
    print("| table | " + " | ".join(LESSONS) + " |")
    print("|---" * (len(LESSONS) + 1) + "|")
    for name, (Td, Tb) in fam.items():
        r = checks(mc, Td, Tb, d)
        print(f"| {name} | " + " | ".join(f"{r.get(k, float('nan')):.3f}".rstrip("0").rstrip(".") for k in LESSONS) + " |",
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
