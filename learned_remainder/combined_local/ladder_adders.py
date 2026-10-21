"""The product cell's ladder alone (SQUARE.md's lesson, submission_combined.py's ladder_hard targets and verified
frontier), with the true subtraction table, on the training rows' own x and N digits: which soft adder in the soft rows
lets it learn (COMBINED.md section 4.2). Small: under a minute per variant on a laptop CPU.

  python ladder_adders.py [VARIANT ...]     # designed | ripples | split | joint (default: all four)

designed: SQUARE's hand-written soft adder (add_dd); ripples: the learned adder as two soft ripples a - (0 - b)
(add_soft_learned); split: one operand soft at a time (lo soft with hi snapped, and the reverse); joint: the learned
adder with both ripples carried as one chain over their joint borrows (add_soft_joint, the final file's). Prints the
cells right and the share of rung checks passing every 25 steps, and the cells still wrong at the end.
"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(os.path.dirname(HERE), "chain_local", "stub"), os.path.dirname(HERE)]
import torch
import torch.nn.functional as F
import submission_combined as sc
import submission_square as sq

DATA = os.environ.get("H1_DATA", os.path.expanduser(
    "~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def run(kind, XS, NS, steps=200, B=512, seed=0):
    Td, Tb = sc.exact_tables("cpu")
    Plo_t, Phi_t = sc.exact_products("cpu")
    TH = torch.cat([F.one_hot(Td, 10), F.one_hot(Tb, 2)], -1).float().view(10, 10, 2, 12).permute(0, 2, 1, 3)
    TH = TH.reshape(20, 120)
    torch.manual_seed(seed)
    p_lo = torch.nn.Parameter(0.1 * torch.randn(10, 10, 10))
    p_hi = torch.nn.Parameter(0.1 * torch.randn(10, 10, 10))
    opt = torch.optim.Adam([p_lo, p_hi], lr=0.05)
    g = torch.Generator().manual_seed(seed)
    nllf = lambda Y, t: -Y.clamp_min(1e-12).log().gather(-1, t[..., None]).squeeze(-1)
    out = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, XS.shape[0], (B,), generator=g)
        X = torch.cat([XS[idx], NS[idx]])
        n = X.shape[0]
        tabs = (Td, Tb, p_lo.argmax(-1).flatten(), p_hi.argmax(-1).flatten())
        with torch.no_grad():
            _, tgt, ok = sc.ladder_hard(X, tabs)
            w = ok.long().cumprod(1).bool()[:, :9].float()
        PLO, PHI = p_lo.softmax(-1), p_hi.softmax(-1)
        if kind == "split":
            def rows(b):
                i = X * 10 + b
                lo_h, hi_h = F.pad(tabs[2][i], (0, 2)), F.pad(tabs[3][i], (1, 1))
                Y1 = sc.link_soft(sc.zpad(PLO[X, b], 0, 2), sc.sub_hard(torch.zeros_like(hi_h), hi_h, Td, Tb)[0], TH)[0]
                Y2 = sc.link_soft(sc.zpad(PHI[X, b], 1, 1), sc.sub_hard(torch.zeros_like(lo_h), lo_h, Td, Tb)[0], TH)[0]
                return Y1, Y2
            z = torch.zeros(n, sc.Q, dtype=torch.long)
            Y1, Y2 = rows(0)
            loss = (nllf(Y1, z).mean() + nllf(Y2, z).mean()) / 2
            for b in range(9):
                Y1, Y2 = rows(b + 1)
                loss = loss + (((nllf(Y1, tgt[:, b]) + nllf(Y2, tgt[:, b])) / 2).mean(-1) * w[:, b]).sum() / n
        else:
            row = {
                "designed": lambda b: sq.row_soft(X, b, PLO, PHI),
                "ripples": lambda b: sc.add_soft_learned(sc.zpad(PLO[X, b], 0, 2), sc.zpad(PHI[X, b], 1, 1), TH)[0],
                "joint": lambda b: sc.row_soft(X, b, PLO, PHI, TH),
            }[kind]
            loss = -row(0)[..., 0].clamp_min(1e-12).log().mean()
            for b in range(9):
                loss = loss + (nllf(row(b + 1), tgt[:, b]).mean(-1) * w[:, b]).sum() / n
        opt.zero_grad()
        (loss / 10).backward()
        opt.step()
        if step % 25 == 0:
            right = int(((tabs[2] == Plo_t) & (tabs[3] == Phi_t)).sum())
            out.append(f"{step}:{right}/{float(ok.float().mean()):.2f}")
    lo, hi = p_lo.argmax(-1).flatten(), p_hi.argmax(-1).flatten()
    wrong = [(i // 10, i % 10, int(lo[i]), int(hi[i])) for i in range(100) if lo[i] != Plo_t[i] or hi[i] != Phi_t[i]]
    return out, wrong


def main(kinds):
    rows = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"))]
    dig = lambda v: [(v // 10 ** i) % 10 for i in range(8)]
    XS = torch.tensor([dig(r["x"]) for r in rows])
    NS = torch.tensor([dig(r["modulus"]) for r in rows])
    for kind in kinds:
        t0 = time.time()
        out, wrong = run(kind, XS, NS)
        print(f"{kind}: cells_right/rungs_passing {out}  wrong (a, b, lo, hi) {wrong[:8]}  {time.time() - t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["designed", "ripples", "split", "joint"])
