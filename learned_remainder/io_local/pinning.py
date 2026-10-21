"""Which wrong-but-consistent I/O mappings does each lesson of submission_io.py leave, on the training rows?
  (a) the answer round trip vs the writer's program (anchor token, step): which programs reproduce every answer;
  (b) the round trips vs the code: any relabeling of the 10 values applied to reading and writing passes them; how
      "x < N" (true reading) cuts that down: the decisive digit pairs seen;
  (c) the tiling: which assignments of the three digit runs to the fields (N, x: 8 slots, T: 2) fit;
  (d) the 24 complete readings under "x < N", each with its best rank code (Bradley-Terry fit, as in training);
  (e) T's direction: the answers of the T = 16 rows under the reading "61".
Small: CPU, < 1 GB, about a minute.
    python pinning.py
"""
import itertools, math, os, sys, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from local_io import load, batch
from contrast import ext, PAD, END

W = 8


def runs_of(e, end):
    p = torch.arange(e.shape[1])[None]
    isd = (p >= PAD) & (p < PAD + end[:, None]) & (e >= 7) & (e <= 16)
    start = isd & ~F.pad(isd[:, :-1], (1, 0))
    stop = isd & ~F.pad(isd[:, 1:], (0, 1))
    si = torch.where(start, p, e.shape[1] - 2).sort(1).values[:, :3]
    ti = torch.where(stop, p, e.shape[1] - 2).sort(1).values[:, :3]
    return si, ti, ti - si + 1


def main():
    ids_all, lab_all, num_all = load("train")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(ids_all), generator=g)[:20000]
    ids, mask, lab, tp = batch(ids_all[idx], lab_all[idx])
    num = num_all[idx]
    e, end = ext(ids, mask)
    X, Lp = e.shape
    valid = lab != -100
    # (a) the writer's programs vs the answer round trip
    labe = torch.full_like(e, END + 1)
    labe.scatter_(1, torch.where(valid, tp + PAD, 0), torch.where(valid, lab, END + 1)); labe[:, 0] = END + 1
    print("(a) writer programs reproducing every answer token (share of 20,000 training rows):")
    for a in list(range(17)) + [END]:
        for s in (1, -1):
            pmask = (e == a) & (torch.arange(Lp)[None] >= PAD) & (torch.arange(Lp)[None] <= PAD + end[:, None])
            if not pmask.any():
                continue
            q = pmask.float().argmax(1)
            has = pmask.any(1)
            pos = q[:, None] + (torch.arange(W)[None] + 1) * s
            t = labe.gather(1, pos.clamp(0, Lp - 1))
            live = ((t >= 7) & (t <= 16)).long().cumprod(-1).bool()
            out = torch.full_like(e, -1).scatter_(1, pos.clamp(0, Lp - 1), torch.where(live, t, -1))
            got = out.gather(1, torch.where(valid, tp + PAD, 0))
            ok = (torch.where(valid, got, lab) == lab).all(1) & has
            if ok.float().mean() > 0:
                print(f"    anchor={'END' if a == END else a} step={s:+d}: {float(ok.float().mean()):.4f}")
    # (b) the decisive pairs of x < N under the true reading
    P10 = 10 ** torch.arange(9)
    Nd, xd = (num[:, 0][:, None] // P10) % 10, (num[:, 1][:, None] // P10) % 10
    wN, wx = (num[:, 0][:, None] >= P10).sum(1), (num[:, 1][:, None] >= P10).sum(1)
    eq = wN == wx
    diff = (Nd != xd)
    k = torch.where(diff, torch.arange(9)[None], -1).max(1).values
    pa, pb = xd.gather(1, k[:, None])[:, 0], Nd.gather(1, k[:, None])[:, 0]
    pairs = set(zip(pa[eq].tolist(), pb[eq].tolist()))
    print(f"(b) round trips admit {math.factorial(10):,} relabelings of the 10 values; decisive pairs of x < N among "
          f"these rows: {len(pairs)} of 45 (all with a < b: {all(a < b for a, b in pairs)}) -> a code that passes x < N "
          f"on every row is strictly increasing on all of them: only the identity")
    # (c) the tiling
    si, ti, length = runs_of(e, end)
    print("(c) assignments of runs (N's, x's, T's) to fields (N, x, T) that fit (share of rows):")
    for perm in itertools.permutations(range(3)):
        fits = torch.ones(X, dtype=torch.bool)
        for f, r in enumerate(perm):
            fits &= length[:, r] <= (2 if f == 2 else 8)
        print(f"    N<-run{perm[0]} x<-run{perm[1]} T<-run{perm[2]}: {float(fits.float().mean()):.4f} (either direction each)")
    # (d) the 24 readings under x < N, best rank code by Bradley-Terry
    HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
    j = torch.arange(W)
    tokp = e.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1)).view(X, 3, W)
    tokm = e.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1)).view(X, 3, W)
    liv = j[None, None] < length[..., None]
    R = F.pad(torch.stack([torch.where(liv, tokp, -1), torch.where(liv, tokm, -1)], 2), (0, 1), value=-1)
    hyp = torch.tensor(HYP)
    Tn, Tx = R[:, hyp[:, 0], hyp[:, 1]], R[:, hyp[:, 2], hyp[:, 3]]
    place = torch.where(Tx != Tn, torch.arange(9), -1).max(-1).values
    ta = Tx.gather(2, place.clamp(min=0)[..., None])[..., 0]
    tb = Tn.gather(2, place.clamp(min=0)[..., None])[..., 0]
    both = (ta >= 0) & (tb >= 0) & (place >= 0)
    padx = (ta < 0) & (tb >= 0) & (place >= 0)
    u = torch.nn.Parameter(0.1 * torch.randn(len(HYP), 17))
    opt = torch.optim.Adam([u], lr=0.05)
    for _ in range(600):
        ua = u[None].expand(X, -1, -1).gather(2, ta.clamp(min=0)[..., None])[..., 0]
        ub = u[None].expand(X, -1, -1).gather(2, tb.clamp(min=0)[..., None])[..., 0]
        loss = -(F.logsigmoid(ub - ua) * both).sum() / both.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        dig = torch.zeros(17, dtype=torch.bool); dig[7:] = True
        umin = u.masked_fill(~dig[None], 1e9).min(-1).values
        ua = u[None].expand(X, -1, -1).gather(2, ta.clamp(min=0)[..., None])[..., 0]
        ub = u[None].expand(X, -1, -1).gather(2, tb.clamp(min=0)[..., None])[..., 0]
        hp = torch.where(both, ub > ua, torch.where(padx, ub > umin[None], torch.zeros_like(both)))
        rest = 3 - hyp[:, 0] - hyp[:, 2]
        tfits = length.gather(1, rest[None].expand(X, -1)) <= 2
        res = sorted(((float((hp[:, h] & tfits[:, h]).float().mean()), HYP[h],
                       "".join(str(int(v)) for v in (u[h, 7:][:, None] > u[h, 7:][None]).sum(1))) for h in range(len(HYP))),
                     reverse=True)
    print("(d) readings (N's run, dir, x's run, dir; dir 0 = step +1) by the share of rows where x < N holds and T's two "
          "slots can hold the rest, with their best rank code (digits 0..9 -> values):")
    for r, h, c in res[:8]:
        print(f"    {h}: {r:.4f}  code {c}")
    print(f"    ... the other {len(res) - 8} at most {res[8][0]:.4f}")
    # (e) T's direction: the reading "61" on the T = 16 rows
    t16 = num_all[num_all[:, 2] == 16]
    hits = sum(pow(int(x), 2 ** 61, int(n)) == int(y) for n, x, _, y in t16.tolist())
    print(f"(e) T = 16 rows whose answer is also x^(2^61) mod N (the reading '61'): {hits} of {len(t16)}")


if __name__ == "__main__":
    main()
