"""Which way each lesson pushes the trap rows, at the true table (local, < 1 min, < 1 GB).

The table's logits are SHARP on the true digit and borrow of every row (softmax probability of the truth ~0.98 at
SHARP = 6, as a nearly learned table is). For each lesson, computed as submission_ablate.py computes it on 256
training rows (x, N and the true machine's own link inputs), the gradient of its loss with respect to the digit logits
of each row in a family is read as a push (256 rows): logit(trap digit) - logit(true digit) changes at rate
-(dL/da_trap - dL/da_true) under gradient descent. Positive = toward the trap; per family, summed over its rows,
times 1e4.

    python probe_grad.py [SHARP]
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(HERE)), "learned_remainder", "chain_local", "stub"), HERE]
import torch
import torch.nn.functional as F
import submission_ablate as sa
from probe_trap import rows

FAMILIES = {"(r,0,1) -> r": [(r, 0, 1, r) for r in range(10)], "(r,r,1) -> 0": [(r, r, 1, 0) for r in range(10)],
            "(0,0,1) -> 0": [(0, 0, 1, 0)]}


def lessons(a, b, x, N, M, xd, width, D, Mr, Rg, MW, Z, K):
    TDC = torch.cat([a.softmax(-1), b.softmax(-1)], -1).permute(0, 2, 1, 3).reshape(20, 120)
    hot = lambda v: F.one_hot(v, 10).float()
    log = lambda p: p.clamp_min(1e-12).log()
    dnll = lambda Y, t: -log(Y).gather(-1, t[..., None]).squeeze(-1).sum(-1)
    weight = torch.ones(sa.P)
    weight[0] = 5.0
    out = {}
    Es, be = sa.link_soft(hot(D), Mr, TDC)
    Cs, bc = sa.link_soft(hot(Z), M, TDC)
    back, bf = sa.sub_soft(Es, Cs.repeat_interleave(K, 0), TDC)
    pc, pe, pf = bc[:, 1].repeat_interleave(K, 0), be[:, 1], bf[:, 1]
    agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
    nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
    out["learned check"] = nll.mean() / weight.sum()
    out["learned check, complete"] = (nll - log(agree)).mean() / (weight.sum() + 1)
    backd, carry = sa.add_soft(Es, Mr)
    nlld = (-log(backd).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
    out["designed check, complete"] = (nlld - log(carry * be[:, 1] + (1 - carry) * be[:, 0])).mean() / (weight.sum() + 1)
    Y, bz = sa.link_soft(hot(Rg), torch.zeros_like(Rg), TDC)
    out["R - 0 = R"] = (dnll(Y, Rg) - log(bz[:, 0])).mean() / (sa.P + 1)
    Y, bs = sa.link_soft(hot(Rg), Rg, TDC)
    out["R - R = 0"] = (dnll(Y, torch.zeros_like(Rg)) - log(bs[:, 0])).mean() / (sa.P + 1)
    out["x - N borrows"] = -log(sa.link_soft(hot(xd), M, TDC)[1][:, 1]).mean()
    out["N - x no borrow"] = -log(sa.link_soft(hot(M), xd, TDC)[1][:, 0]).mean()
    out["0 - N borrows"] = -log(bc[:, 1]).mean()
    Ew = sa.link_soft(hot(xd), M, TDC)[0]
    out["(x - N) - N no borrow"] = -log(sa.link_soft(Ew, M, TDC)[1][:, 0]).mean()
    out["x mod N = x"] = sa.ce(sa.divide_soft(x, 1, M, TDC), xd, width)
    Y2, b2 = sa.sub_soft(hot(Z), Cs, TDC)
    out["0 + N = N"] = (dnll(Y2, M) - log(bc[:, 1] * b2[:, 1] + bc[:, 0] * b2[:, 0])).mean() / (sa.P + 1)
    A1, b1 = sa.sub_soft(hot(Rg), Cs.repeat_interleave(sa.W, 0), TDC)
    CR, bcr = sa.link_soft(hot(torch.zeros_like(Rg)), Rg, TDC)
    A2, b2 = sa.sub_soft(hot(MW), CR, TDC)
    c1, c2 = bc[:, 1].repeat_interleave(sa.W, 0) * b1[:, 0], bcr[:, 1] * b2[:, 0]
    out["R + N = N + R"] = -(log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))).mean() / (sa.P + 1)
    with torch.no_grad():
        wrapped = sa.link_hard(D, Mr, *sa.exact_tables("cpu"))[1] == 1
    out["chain: no borrow after a borrow"] = -log(sa.link_soft(Es[wrapped], Mr[wrapped], TDC)[1][:, 0]).mean()
    return out


def main(sharp=6.0):
    data = rows(256)                                                  # autograd keeps every place: 256 rows stay < 1 GB
    N = torch.tensor([r["modulus"] for r in data])
    x = torch.tensor([r["x"] for r in data])
    M, xd = sa.to_digits(N, sa.P), sa.to_digits(x, sa.P)
    width = torch.tensor([len(str(int(v))) for v in N])
    Td, Tb = sa.exact_tables("cpu")
    X, K = x.shape[0], sa.W * (sa.S - 1)
    D, _ = sa.link_inputs(x * x, sa.W, M, Td, Tb)
    D, Mr = D.reshape(X * K, sa.P), M.repeat_interleave(K, 0)
    Rg = D.view(X, sa.W, sa.S - 1, sa.P)[:, :, 0].reshape(X * sa.W, sa.P)
    MW, Z = M.repeat_interleave(sa.W, 0), torch.zeros_like(M)
    a = (sharp * F.one_hot(Td.view(10, 10, 2), 10).float()).requires_grad_()
    b = (sharp * F.one_hot(Tb.view(10, 10, 2), 2).float()).requires_grad_()
    out = lessons(a, b, x, N, M, xd, width, D, Mr, Rg, MW, Z, K)
    print(f"sharp {sharp}: push toward the trap (x 1e4; positive = toward the trap digit), at the true table")
    print("| lesson | loss | " + " | ".join(FAMILIES) + " |")
    print("|---|---|" + "---|" * len(FAMILIES))
    for name, L in out.items():
        g, = torch.autograd.grad(L, a, retain_graph=True)
        pushes = []
        for fam in FAMILIES.values():
            p = sum(float(g[r, m, c, (r - m - c) % 10] - g[r, m, c, d]) for r, m, c, d in fam)
            pushes.append(f"{1e4 * p:+.2f}")
        print(f"| {name} | {float(L):.4f} | " + " | ".join(pushes) + " |", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 6.0)
