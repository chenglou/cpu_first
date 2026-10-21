"""Do the pointers generalize to 8 digits? Three pointer mechanisms for the N field, the x field and the writer, each
trained with the strongest possible teacher (supervised on the true slot positions of training prompts: an upper bound
for any lesson), then read the unseen-moduli depth prompts (N up to 8 digits, never in training; x and the answer up
to 8 digits, x has 7 digits in only 2 training rows and the answer never has 8):
  * step: one anchor and one shared step per field (submission_io.py's grammar): slot j at anchor + (j + 1) * step;
  * per_slot: one anchor per field, and every slot its own learned offset from it;
  * absolute: every slot its own learned absolute position.
All three use the same extent rule (a slot reads 0 once a non-digit was met) and the true digit values. A slot that is
live in a training row is taught its position; the first slot past a number's end is taught to land on a non-digit
(the rest are dead anyway). Small: CPU, batch 256, a few seconds per variant.
    python contrast.py [STEPS] [SEEDS]
"""
import os, sys, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE]
from local_io import load, batch

W, PAD = 8, 9                     # 8 slots; prompts padded by 9 on each side (the pointer never leaves the array)
END = 17


def ext(ids, mask):
    end = mask.sum(1)
    p = torch.arange(ids.shape[1] + 1)[None]
    e = torch.where(p < end[:, None], F.pad(ids, (0, 1)), torch.where(p == end[:, None], END, END + 1))
    return F.pad(e, (PAD, PAD), value=END + 1), end


def truth_positions(e, end, field):
    """The true slot positions (padded coordinates) of a field and how many slots are live: N after its marker, x
    before T's marker, the answer before the end, all LSB first."""
    pN = (e == 2).float().argmax(1); pX = (e == 3).float().argmax(1); pT = (e == 4).float().argmax(1)
    j = torch.arange(W)[None]
    if field == "N":
        return pN[:, None] + 1 + j, pX - pN - 1
    if field == "x":
        return pT[:, None] - 1 - j, pT - pX - 1
    return (PAD + end)[:, None] - 1 - j, None                  # the writer: the live count is the answer's length


class Pointer(torch.nn.Module):
    def __init__(self, kind, L):
        super().__init__()
        self.kind = kind
        self.anc = torch.nn.Parameter(0.1 * torch.randn(19))         # anchor token type (17 ids, END, -)
        if kind == "step":
            self.st = torch.nn.Parameter(0.1 * torch.randn(2))
        elif kind == "per_slot":
            self.off = torch.nn.Parameter(0.1 * torch.randn(W, 2 * PAD + 1))
        else:
            self.abs = torch.nn.Parameter(0.1 * torch.randn(W, L))

    def slot_dist(self, e, end):
        """P(slot j reads padded position p) [X, W, Lp]."""
        X, Lp = e.shape
        p = torch.arange(Lp)[None]
        ok = (p >= PAD) & (p <= PAD + end[:, None])
        a = self.anc[e.clamp(max=18)].masked_fill(~ok, -1e9).softmax(-1)            # [X, Lp]
        if self.kind == "abs":
            pa = self.abs.softmax(-1)                                                  # [W, L]
            return F.pad(pa, (PAD, Lp - PAD - pa.shape[1]))[None].expand(X, -1, -1)
        if self.kind == "step":
            s = self.st.softmax(-1)
            out = torch.zeros(X, W, Lp)
            for k, d in enumerate((1, -1)):
                for j in range(W):
                    out[:, j] += s[k] * torch.roll(a, (j + 1) * d, 1)
            return out
        o = self.off.softmax(-1)                                                       # [W, 19]
        out = torch.zeros(X, W, Lp)
        for i, d in enumerate(range(-PAD, PAD + 1)):
            out += o[None, :, i, None] * torch.roll(a, d, 1)[:, None]
        return out

    def hard(self, e, end):
        X, Lp = e.shape
        p = torch.arange(Lp)[None]
        ok = (p >= PAD) & (p <= PAD + end[:, None])
        q = self.anc[e.clamp(max=18)].masked_fill(~ok, -1e9).argmax(-1)
        j = torch.arange(W)[None]
        if self.kind == "abs":
            return (self.abs.argmax(-1) + PAD)[None].expand(X, -1)
        if self.kind == "step":
            return q[:, None] + (j + 1) * (1 if int(self.st.argmax()) == 0 else -1)
        return q[:, None] + (self.off.argmax(-1) - PAD)[None]


def read(e, pos):
    t = e.gather(1, pos.clamp(0, e.shape[1] - 1))
    live = ((t >= 7) & (t <= 16)).long().cumprod(-1).bool()
    return torch.where(live, t - 7, 0), live


def train_eval(kind, field, steps, seed, data, ood):
    torch.manual_seed(seed)
    ids_all, lab_all, num_all = data
    m = Pointer(kind, 22)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(ids_all), (256,), generator=g)
        ids, mask, lab, tp = batch(ids_all[idx], lab_all[idx])
        e, end = ext(ids, mask)
        tpos, width = truth_positions(e, end, field)
        if width is None:
            width = (lab != -100).sum(1)
        P = m.slot_dist(e, end)                                                        # [X, W, Lp]
        j = torch.arange(W)[None]
        live = j < width[:, None]
        p_true = P.gather(2, tpos.clamp(0, e.shape[1] - 1)[..., None])[..., 0]
        nondig = ~((e >= 7) & (e <= 16))
        p_stop = (P * nondig[:, None].float()).sum(-1)                                  # lands on a non-digit
        first_dead = (j == width[:, None]) & (field != "writer")         # the writer writes all its slots
        loss = -(p_true.clamp_min(1e-9).log() * live).sum() / live.sum() \
               - (p_stop.clamp_min(1e-9).log() * first_dead).sum() / first_dead.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
    ids, lab, num = ood
    ids, mask, lab, tp = batch(ids, lab)
    e, end = ext(ids, mask)
    with torch.no_grad():
        digits, _ = read(e, m.hard(e, end))
    val = (digits * 10 ** torch.arange(W)).sum(1)
    if field == "N":
        target, wide = num[:, 0], num[:, 0] >= 10 ** 7
    elif field == "x":
        target, wide = num[:, 1], num[:, 1] >= 10 ** 7
    else:                       # the writer: write the true answer at its positions, compare at the target positions
        pos = m.hard(e, end).clamp(0, e.shape[1] - 1)
        yd = (num[:, 3][:, None] // 10 ** torch.arange(W)) % 10
        out = torch.full_like(e, -1).scatter_(1, pos, yd + 7)
        valid = lab != -100
        got = out.gather(1, torch.where(valid, tp + PAD, 0))
        ok = (torch.where(valid, got, lab) == lab).all(1)
        wide = num[:, 3] >= 10 ** 7
        return float(ok.float().mean()), float(ok[wide].float().mean()), int(wide.sum())
    ok = val == target
    return float(ok.float().mean()), float(ok[wide].float().mean()), int(wide.sum())


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    data, ood = load("train"), load("depth_ood_n_t_1")
    for field in ("N", "x", "writer"):
        for kind in ("step", "per_slot", "abs"):
            res = [train_eval(kind, field, steps, s, data, ood) for s in range(1, seeds + 1)]
            print(f"{field:6s} {kind:8s} all OOD-N prompts: {[round(r[0], 4) for r in res]}  "
                  f"8-digit ones ({res[0][2]}): {[round(r[1], 4) for r in res]}", flush=True)
