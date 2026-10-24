"""Exact loss of every lesson along a straight path from a wrong-row family back to the truth (local, < 1 min, < 1 GB).

The table's logits are sharp (logit SHARP on one digit and one borrow per row). Every row sits at the truth except one
family, whose digit logits slide from the trap's digit (lam = 0) to the true digit (lam = 1). Each lesson is computed
as submission_ablate.py computes it, on fixed numbers: 1,024 training rows' x and N, and the link inputs of the trapped
table's own hard division of x^2 (what training sees while it sits in the trap). A lesson that slopes down toward
lam = 1 pulls the family out; one that is flat or slopes up holds it.

    python probe_trap.py r01 | rr1 | 001 [SHARP]
      r01: (r, 0, borrow 1) writes r instead of r - 1, 10 rows (ADDER.md 5.1, the run of 9s)
      rr1: (r, r, borrow 1) writes 0 instead of 9, 10 rows (the identity's borrow saddle, LEARNED_REMAINDER.md 4.2)
      001: only (0, 0, borrow 1), 0 instead of 9 (CHAIN.md 5.2)
      r01x, rr1x: the same families without (0, 0, borrow 1)
    python probe_trap.py whole:<table> [SHARP]
      the whole table slides from a consistent wrong table of analyze.py (e.g. whole:"swapped borrow code") to the truth
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LR_DIR = os.path.join(os.path.dirname(HERE), "learned_remainder")
sys.path[:0] = [os.path.join(LR_DIR, "chain_local", "stub"), HERE]
import torch
import torch.nn.functional as F
import submission_ablate as sa

DATA = os.environ.get("H1_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "one-layer-deeper", "data", "generated",
    "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def family(name):
    if name.startswith("whole:"):                                     # every row: (r, m, c, digit, borrow) of that table
        from analyze import CAND
        d, b = CAND[name[6:]]
        return [(r, m, c, d[i], b[i]) for i, (r, m, c) in enumerate((r, m, c) for r in range(10) for m in range(10)
                                                                       for c in range(2))]
    if name == "r01":
        return [(r, 0, 1, r) for r in range(10)]
    if name == "rr1":
        return [(r, r, 1, 0) for r in range(10)]
    if name == "001":
        return [(0, 0, 1, 0)]
    if name == "r01x":                                                # the family without (0, 0, 1)
        return [(r, 0, 1, r) for r in range(1, 10)]
    if name == "rr1x":
        return [(r, r, 1, 0) for r in range(1, 10)]
    raise KeyError(name)


def rows(n=1024):
    with open(os.path.join(DATA, "train.jsonl")) as f:
        lines = f.readlines()
    g = torch.Generator().manual_seed(0)
    return [json.loads(lines[i]) for i in torch.randperm(len(lines), generator=g)[:n].tolist()]


def main(name, sharp=6.0):
    data = rows()
    N = torch.tensor([r["modulus"] for r in data])
    x = torch.tensor([r["x"] for r in data])
    M, xd = sa.to_digits(N, sa.P), sa.to_digits(x, sa.P)
    width = torch.tensor([len(str(int(v))) for v in N])
    Td0, Tb0 = sa.exact_tables("cpu")
    fam = family(name)
    fam = [f if len(f) == 5 else (*f, int(f[0] - f[1] - f[2] < 0)) for f in fam]
    trap_d, trap_b = Td0.clone(), Tb0.clone()
    for r, m, c, d, bb in fam:
        trap_d[r * 20 + m * 2 + c], trap_b[r * 20 + m * 2 + c] = d, bb
    X, K = x.shape[0], sa.W * (sa.S - 1)
    D, _ = sa.link_inputs(x * x, sa.W, M, trap_d, trap_b)              # the trapped machine's own link inputs
    D, Mr = D.reshape(X * K, sa.P), M.repeat_interleave(K, 0)
    Rg = D.view(X, sa.W, sa.S - 1, sa.P)[:, :, 0].reshape(X * sa.W, sa.P)
    MW = M.repeat_interleave(sa.W, 0)
    Z = torch.zeros_like(M)
    hot = lambda v: F.one_hot(v, 10).float()
    log = lambda p: p.clamp_min(1e-12).log()
    dnll = lambda Y, t: -log(Y).gather(-1, t[..., None]).squeeze(-1).sum(-1)
    weight = torch.ones(sa.P)
    weight[0] = 5.0
    print(f"family {name} ({len(fam)} rows), sharp {sharp}, {X} rows, {D.shape[0]} link inputs")
    hdr = ["lam", "P(true)", "check learned", "+borrow", "check designed", "+borrow", "wrap", "less", "more", "neg",
           "x mod N", "zero", "self", "0+N=N", "R+N=N+R"]
    print(" | ".join(hdr))
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        a = sharp * F.one_hot(Td0.view(10, 10, 2), 10).float()
        b = sharp * F.one_hot(Tb0.view(10, 10, 2), 2).float()
        for r, m, c, d, bb in fam:
            true, tb = (r - m - c) % 10, int(r - m - c < 0)
            a[r, m, c] = sharp * ((1 - lam) * F.one_hot(torch.tensor(d), 10).float()
                                  + lam * F.one_hot(torch.tensor(true), 10).float())
            b[r, m, c] = sharp * ((1 - lam) * F.one_hot(torch.tensor(bb), 2).float()
                                  + lam * F.one_hot(torch.tensor(tb), 2).float())
        TDC = torch.cat([a.softmax(-1), b.softmax(-1)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        with torch.no_grad():
            Es, be = sa.link_soft(hot(D), Mr, TDC)
            Cs, bc = sa.link_soft(hot(Z), M, TDC)
            back, bf = sa.sub_soft(Es, Cs.repeat_interleave(K, 0), TDC)
            pc = bc[:, 1].repeat_interleave(K, 0)
            pe, pf = be[:, 1], bf[:, 1]
            agree = (1 - pe) * (1 - pf) * (1 - pc) + pe * (1 - pf) * pc + (1 - pe) * pf * pc
            nll = (-log(back).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
            learned = float(nll.mean() / weight.sum())
            learned_c = float((nll - log(agree)).mean() / (weight.sum() + 1))
            backd, carry = sa.add_soft(Es, Mr)
            nlld = (-log(backd).gather(-1, D[..., None]).squeeze(-1) * weight).sum(-1)
            agreed = carry * be[:, 1] + (1 - carry) * be[:, 0]
            designed = float(nlld.mean() / weight.sum())
            designed_c = float((nlld - log(agreed)).mean() / (weight.sum() + 1))
            Ew = sa.link_soft(hot(xd), M, TDC)[0]
            wrap = float(-log(sa.link_soft(Ew, M, TDC)[1][:, 0]).mean())
            less = float(-log(sa.link_soft(hot(xd), M, TDC)[1][:, 1]).mean())
            more = float(-log(sa.link_soft(hot(M), xd, TDC)[1][:, 0]).mean())
            neg = float(-log(bc[:, 1]).mean())
            xmod = float(sa.ce(sa.divide_soft(x, 1, M, TDC), xd, width))
            Y, bz = sa.link_soft(hot(Rg), torch.zeros_like(Rg), TDC)
            zero = float((dnll(Y, Rg) - log(bz[:, 0])).mean() / (sa.P + 1))
            Y, bs = sa.link_soft(hot(Rg), Rg, TDC)
            selfl = float((dnll(Y, torch.zeros_like(Rg)) - log(bs[:, 0])).mean() / (sa.P + 1))
            Y2, b2 = sa.sub_soft(hot(Z), Cs, TDC)
            agree2 = bc[:, 1] * b2[:, 1] + bc[:, 0] * b2[:, 0]
            neg2 = float((dnll(Y2, M) - log(agree2)).mean() / (sa.P + 1))
            A1, b1 = sa.sub_soft(hot(Rg), Cs.repeat_interleave(sa.W, 0), TDC)
            CR, bcr = sa.link_soft(hot(torch.zeros_like(Rg)), Rg, TDC)
            A2, b2 = sa.sub_soft(hot(MW), CR, TDC)
            c1, c2 = bc[:, 1].repeat_interleave(sa.W, 0) * b1[:, 0], bcr[:, 1] * b2[:, 0]
            comm = float(-(log((A1 * A2).sum(-1)).sum(-1) + log(c1 * c2 + (1 - c1) * (1 - c2))).mean() / (sa.P + 1))
        bad = [f for f in fam if f[3] != (f[0] - f[1] - f[2]) % 10] or fam
        ptrue = float(a.softmax(-1)[bad[0][0], bad[0][1], bad[0][2], (bad[0][0] - bad[0][1] - bad[0][2]) % 10])
        vals = [learned, learned_c, designed, designed_c, wrap, less, more, neg, xmod, zero, selfl, neg2, comm]
        print(f"{lam:.1f} | {ptrue:.3f} | " + " | ".join(f"{v:.4f}" for v in vals), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
