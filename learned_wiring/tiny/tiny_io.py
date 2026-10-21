"""Stage 3 in miniature: the reader's and writer's shape as a small mixture of candidate layouts (WIRING.md section 4).

H1's real prompts (`N <N digits LSB first> X <x MSB first> T <T MSB first>`, the answer right-aligned on the prompt's
last positions). The digit tokens and their values are given here (IO.md learns them; this lab isolates the pointer's
shape). Each field (N, x, T and the writer) addresses prompt positions with a mixture over three shapes (IO.md
section 6 contrasted them under supervision):
  step     one anchor (a token type or the prompt's end) and one shared step: slot j at anchor + (j + 1) * step
  offsets  one anchor and every slot its own offset from it, in -OFF..OFF
  absolute every slot its own absolute position (from the prompt's start)
The field's weight on each shape is learned (a softmax over 3), as are each shape's own parameters. A slot reads 0 once
a slot before it met a non-digit (IO.md's extent rule).

Lessons (IO.md's, scored on whole readings):
  * the reading: a normalized per-row mixture over complete readings (N's run and direction, x's run and direction,
    T the remaining run), each scored by the exact verdict x < N (and T's value against the row's T: IO.md settles
    T's direction from the answers), the probability of a field reading run r in direction d being the mixture over
    shapes of that shape's probability of putting slots 0..len-1 on the run's digits in order and slot len on the next
    position past it (where the read stops); plus the tiling term once a reading carries 0.99 (IO.md);
  * the writer: the answer's digits, LSB first, must be at slots 0..len-1 and slot len must land past them (the
    answer round trip of IO.md, here with the answer's own positions), a per-row mixture over shapes.
SHAPES restricts the menu (e.g. "step" alone is IO.md's grammar). Certification: the snapped reader (argmax shape,
argmax parameters) reads every N, x and T of depth_ood_n_t_1 (110 eight-digit moduli) and the writer's slots land on
every answer digit.

    ~/github/sotaku/venv/bin/python -m learned_wiring.tiny.tiny_io seed=0 [KEY=VALUE ...]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

CFG = dict(
    seed=0,
    batch=256,
    rows=20000,        # training prompts sampled across train.jsonl (it is sorted by modulus size)
    init=0.1,
    lr=0.05,
    shapes="step,offsets,absolute",
    tie=0,             # 1: one shape switch shared by the four fields (one switch per kind of connection)
    tile_at=0.99,      # the tiling joins once the reading mixture's mean posterior on its favourite exceeds this
    max_steps=1500,
    done=100,          # stop once the snapped reader is exact on the batch for this many steps
    log=50,
)
DATA = os.environ.get("H1_DATA", os.path.expanduser(
    "~/github/one-layer/one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
W, PAD, OFF = 8, 9, 9
SLOTS = (W, W, 2, W)                           # N, x, T, the writer
V, END, OUTT = 17, 17, 18                      # token ids 0..16; END at the prompt's end; OUT beyond
DIGIT = 7
NAMES = ("N", "x", "T", "writer")
SHAPES = ("step", "offsets", "absolute")


def load(split, n=None, seed=0):
    rows = [json.loads(l) for l in open(os.path.join(DATA, split + ".jsonl"))]
    if n and n < len(rows):
        idx = np.random.default_rng(seed).choice(len(rows), n, replace=False)
        rows = [rows[i] for i in idx]
    return rows


def batch(rows):
    L = max(len(r["input_ids"]) for r in rows)
    T = max(len(r["labels"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), T), -100, dtype=torch.long)
    tpos = torch.full((len(rows), T), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        n, m = len(r["input_ids"]), len(r["labels"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        lab[i, :m] = torch.tensor(r["labels"])
        tpos[i, :m] = torch.arange(n - m, n)
    N = torch.tensor([r["modulus"] for r in rows])
    x = torch.tensor([r["x"] for r in rows])
    Tv = torch.tensor([r["time_steps"] for r in rows])
    return ids, ids != 0, lab, tpos, N, x, Tv


def ext(ids, mask):
    end = mask.long().sum(1)
    p = torch.arange(ids.shape[1] + 1)[None]
    e = torch.where(p < end[:, None], F.pad(ids, (0, 1)), torch.where(p == end[:, None], END, OUTT))
    return F.pad(e, (PAD, PAD), value=OUTT), end


def isdig(t):
    return (t >= DIGIT) & (t < DIGIT + 10)


def runs(e):
    p = torch.arange(e.shape[1])[None]
    isd = isdig(e)
    start = isd & ~F.pad(isd[:, :-1], (1, 0))
    stop = isd & ~F.pad(isd[:, 1:], (0, 1))
    si = torch.where(start, p, e.shape[1] - 2).sort(1).values[:, :3]
    ti = torch.where(stop, p, e.shape[1] - 2).sort(1).values[:, :3]
    return si, ti, ti - si + 1


def value_of(e, pos, live):
    """The number read at positions pos [X, n] (LSB first) with the extent rule."""
    t = e.gather(1, pos.clamp(0, e.shape[1] - 1))
    lv = isdig(t).long().cumprod(-1).bool() & live
    d = torch.where(lv, t - DIGIT, 0)
    return (d * (10 ** torch.arange(pos.shape[1]))).sum(-1)


class Reader(torch.nn.Module):
    def __init__(self, cfg, Lp):
        super().__init__()
        g = torch.Generator().manual_seed(cfg["seed"])
        r = lambda *s: torch.nn.Parameter(cfg["init"] * torch.randn(*s, generator=g))
        self.anc = r(4, V + 2)                                         # anchor token type (incl. END), per field
        self.st = r(4, 2)                                              # shared step +1 | -1
        self.off = r(4, W + 1, 2 * OFF + 1)                            # per-slot offsets (slot W: the stop)
        self.absp = r(4, W + 1, Lp)                                    # per-slot absolute positions
        self.mech = r(4, 3)                                            # the mixture over shapes
        self.allowed = torch.tensor([s in cfg["shapes"].split(",") for s in SHAPES])
        self.tie = bool(cfg["tie"])

    def mech_logp(self):
        m = self.mech[:1].expand(4, -1) if self.tie else self.mech      # tie: one shape switch for every field
        return m.masked_fill(~self.allowed[None], -1e9).log_softmax(-1)

    def anchor(self, f, e, end):
        p = torch.arange(e.shape[1])[None]
        ok = (p >= PAD) & (p <= PAD + end[:, None])
        return self.anc[f][e.clamp(max=V + 1)].masked_fill(~ok, -1e9).softmax(-1)

    def read_prob(self, f, e, end, pos, n):
        """P(field f's slots 0..n-1 land on pos[..., :n] and slot n on pos[..., n]) per shape: pos [X, K, W + 1]
        positions, n [X, K] lengths. Returns [X, K, 3] (step, offsets, absolute)."""
        X, K, _ = pos.shape
        Lp = e.shape[1]
        A = self.anchor(f, e, end)                                     # [X, Lp]
        j = torch.arange(W + 1)
        use = (j[None, None] <= n[..., None]).float()                  # slots 0..n
        # step: the anchor one step before slot 0, and slot n one step past slot n - 1 (pos is in reading order)
        sp = self.st[f].softmax(-1)
        d = pos[..., 1] - pos[..., 0]                                  # +1 or -1 (runs have length >= 1, pos[n] = past)
        q0 = (pos[..., 0] - d).clamp(0, Lp - 1)
        step = A.gather(1, q0) * torch.where(d > 0, sp[0], sp[1])
        # offsets: sum over anchor positions q of A[q] prod_j off_j[pos_j - q]
        lo = self.off[f].log_softmax(-1)                               # [W + 1, 2 OFF + 1]
        q = torch.arange(Lp)
        rel = pos[..., None] - q[None, None, None]                     # [X, K, W + 1, Lp]
        inr = (rel >= -OFF) & (rel <= OFF)
        lp = lo[j[None, None, :, None].expand_as(rel), (rel + OFF).clamp(0, 2 * OFF)]
        lp = torch.where(inr, lp, torch.full_like(lp, -1e4)) * use[..., None]
        offs = (A.clamp_min(1e-30).log()[:, None, :] + lp.sum(2)).logsumexp(-1).exp()
        # absolute
        la = self.absp[f].log_softmax(-1)                              # [W + 1, Lp]
        absl = (la[j[None, None].expand(X, K, -1), pos.clamp(0, Lp - 1)] * use).sum(-1).exp()
        return torch.stack([step, offs, absl], -1)

    def hard_positions(self, f, e, end):
        """The snapped field: its argmax shape's slot positions [X, W] (no grad)."""
        X = e.shape[0]
        m = int(self.mech_logp()[f].argmax())
        n = SLOTS[f]
        if m == 0:
            q = self.anchor(f, e, end).argmax(-1)
            s = 1 if int(self.st[f].argmax()) == 0 else -1
            return q[:, None] + (torch.arange(n)[None] + 1) * s, m
        if m == 1:
            q = self.anchor(f, e, end).argmax(-1)
            return q[:, None] + (self.off[f, :n].argmax(-1) - OFF)[None], m
        return self.absp[f, :n].argmax(-1)[None].expand(X, n), m


def run(cfg, verbose=True):
    torch.manual_seed(cfg["seed"])
    rows = load("train", cfg["rows"], cfg["seed"])
    test = sum((load(f"depth_{p}_{t}") for p in ("t", "ood_n_t") for t in (1, 16, 32, 64)), [])
    Lp = max(len(r["input_ids"]) for r in rows + test) + 1 + 2 * PAD
    model = Reader(cfg, Lp)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    hyp = torch.tensor([(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx
                        for dn in range(2) for dx in range(2)])
    tiling, streak, t0, stop = False, 0, time.time(), None
    for step in range(1, cfg["max_steps"] + 1):
        idx = rng.integers(0, len(rows), cfg["batch"])
        ids, am, lab, tpos, Nv, xv, Tv = batch([rows[i] for i in idx])
        ids = F.pad(ids, (0, Lp - 2 * PAD - 1 - ids.shape[1]))
        am = F.pad(am, (0, Lp - 2 * PAD - 1 - am.shape[1]))
        e, end = ext(ids, am)
        X = e.shape[0]
        si, ti, ln = runs(e)
        jj = torch.arange(W + 1)
        # every run read from each end: positions in reading order, slot len = one past the run [X, 3 runs, 2, W + 1]
        fwd = si[..., None] + jj
        bwd = ti[..., None] - jj
        pos = torch.stack([fwd, bwd], 2)
        nrun = ln[..., None].expand(X, 3, 2)
        # values of every run read each way, LSB first in reading order
        val = torch.stack([value_of(e, pos[:, r, d, :W], (jj[:W][None] < ln[:, r, None])) for r in range(3)
                           for d in range(2)], 1).view(X, 3, 2)
        lmech = model.mech_logp()
        rho = []
        for f in range(3):
            fits = (ln <= SLOTS[f])[..., None]
            pr = model.read_prob(f, e, end, pos.view(X, 6, W + 1), nrun.reshape(X, 6)).view(X, 3, 2, 3)
            rho.append((pr * lmech[f].exp()).sum(-1) * fits)            # [X, run, dir]
        rho = torch.stack(rho, 1)                                      # [X, field, run, dir]
        lik = torch.zeros(X)
        norm = torch.zeros(X)
        post = torch.zeros(X, len(hyp))
        for h, (rn, dn, rx, dx) in enumerate(hyp.tolist()):
            rt = 3 - rn - rx
            good = (val[:, rx, dx] < val[:, rn, dn]).float()            # x < N
            for dt in range(2):
                w = rho[:, 0, rn, dn] * rho[:, 1, rx, dx] * rho[:, 2, rt, dt]
                gt = good * (val[:, rt, dt] == Tv).float()             # T's value (IO.md: from the answers)
                lik = lik + w * (gt + 1e-6)
                norm = norm + w
                post[:, h] = post[:, h] + w.detach()
        loss = -(lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()).mean()
        chosen = float((post / norm.detach()[:, None].clamp_min(1e-30)).max(1).values.mean())
        if not tiling and chosen >= cfg["tile_at"]:
            tiling = True
        if tiling:
            loss = loss - norm.clamp_min(1e-30).log().mean()
        # the writer: the answer's own positions (LSB first: the last target position first), slot len one past them
        m = (lab != -100).sum(1)
        apos = torch.full((X, W + 1), 0, dtype=torch.long)
        last = tpos.gather(1, (m - 1).clamp(min=0)[:, None])[:, 0] + PAD     # the answer's last digit: its LSB
        apos = last[:, None] - jj[None]
        pw = model.read_prob(3, e, end, apos[:, None], m[:, None])[:, 0]     # [X, 3]
        loss = loss - (pw * lmech[3].exp()).sum(-1).clamp_min(1e-30).log().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ok = snapped_ok(model, e, end, Nv, xv, Tv, apos, m)
        streak = streak + 1 if ok == 1.0 else 0
        if verbose and (step % cfg["log"] == 0 or step == 1):
            mech = model.mech_logp().exp()
            print(f"step={step} loss={float(loss):.4f} chosen={chosen:.4f} snapped_ok={ok:.4f} tiling={int(tiling)} "
                  f"shapes=" + " ".join(f"{NAMES[f]}:" + ",".join(f"{float(p):.3f}" for p in mech[f])
                                        for f in range(4)) + f" {time.time() - t0:.1f}s", flush=True)
        if streak >= cfg["done"]:
            stop = step
            break
    res = certify(model, test, Lp)
    res.update(cfg=dict(cfg), stop=stop, secs=round(time.time() - t0, 1),
               shapes=[SHAPES[int(model.mech_logp()[f].argmax())] for f in range(4)],
               mech=[[round(float(p), 4) for p in model.mech_logp()[f].exp()] for f in range(4)])
    return res


@torch.no_grad()
def snapped_ok(model, e, end, Nv, xv, Tv, apos, m):
    """The share of rows whose snapped N, x, T and writer are all right."""
    ok = torch.ones(e.shape[0], dtype=torch.bool)
    for f, want in ((0, Nv), (1, xv), (2, Tv)):
        pos, _ = model.hard_positions(f, e, end)
        ok &= value_of(e, pos, torch.ones_like(pos, dtype=torch.bool)) == want
    pos, _ = model.hard_positions(3, e, end)
    jj = torch.arange(W)
    live = jj[None] < m[:, None]
    ok &= ((pos == apos[:, :W]) | ~live).all(1)
    return float(ok.float().mean())


@torch.no_grad()
def certify(model, test, Lp):
    ids, am, lab, tpos, Nv, xv, Tv = batch(test)
    ids = F.pad(ids, (0, Lp - 2 * PAD - 1 - ids.shape[1]))
    am = F.pad(am, (0, Lp - 2 * PAD - 1 - am.shape[1]))
    e, end = ext(ids, am)
    m = (lab != -100).sum(1)
    last = tpos.gather(1, (m - 1).clamp(min=0)[:, None])[:, 0] + PAD
    apos = last[:, None] - torch.arange(W + 1)[None]
    out = {"ood_n_rows": len(test), "eight_digit_N": int((Nv >= 10 ** 7).sum())}
    allok = torch.ones(len(test), dtype=torch.bool)
    for f, want in ((0, Nv), (1, xv), (2, Tv)):
        pos, _ = model.hard_positions(f, e, end)
        okf = value_of(e, pos, torch.ones_like(pos, dtype=torch.bool)) == want
        out[NAMES[f]] = float(okf.float().mean())
        allok &= okf
    pos, _ = model.hard_positions(3, e, end)
    okw = ((pos == apos[:, :W]) | ~(torch.arange(W)[None] < m[:, None])).all(1)
    out["writer"] = float(okw.float().mean())
    allok &= okw
    out["N_8digit"] = float(allok[Nv >= 10 ** 7].float().mean())
    out["all"] = float(allok.float().mean())
    out["certified"] = bool(allok.all())
    return out


def main(argv):
    cfg = dict(CFG)
    for kv in argv:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if not isinstance(cfg[k], str) else v
    torch.set_num_threads(2)
    res = run(cfg)
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
