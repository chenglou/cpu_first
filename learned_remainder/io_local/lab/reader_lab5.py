"""IO.md §7, item 7 (the trap of reading both backwards). Reader/writer-only lab v5. One teacher per piece:
  * the answer round trip (read the answer at the writer's positions, write it back): the writer's pointer and which
    tokens are digits (the writer can only write digits: P(t | v) = softmax(wc[v] + log digitness));
  * the code round trips (read a token and write it back; write the machine's numbers and read them back), against the
    reader's code held fixed: the writer's code;
  * the tiling (each field reads one whole digit run, fields take different runs) with "x < N" (via the exact
    subtraction table here): the fields' pointers and the reader's code.
Small: batch <= 256, CPU, < 1 GB.

    python reader_lab5.py STEPS B SEED
"""
import itertools, json, os, sys, time
import torch
import torch.nn.functional as F

D = os.environ.get("H1_DATA", "")      # only the unused jsonl loader below needs it; main() uses ../local_io.py
END, OUT = 17, 18
W, P = 8, 9
SLOTS = (8, 8, 2, 8)            # fields: N, x, T, writer
INIT, LR = 0.1, float(os.environ.get("LR", "0.05"))
EPS = 1e-6
PERMS = list(itertools.permutations(range(3)))
P10 = 10 ** torch.arange(W)


def load(split, n=None):
    rows = []
    for line in open(os.path.join(D, split + ".jsonl")):
        rows.append(json.loads(line))
        if n and len(rows) >= n:
            break
    return rows


def collate(rows):
    L = max(len(r["input_ids"]) for r in rows)
    Lt = max(len(r["labels"]) for r in rows)
    ids = torch.zeros(len(rows), L, dtype=torch.long)
    lab = torch.full((len(rows), Lt), -100, dtype=torch.long)
    tp = torch.full((len(rows), Lt), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        n, t = len(r["input_ids"]), len(r["labels"])
        ids[i, :n] = torch.tensor(r["input_ids"])
        lab[i, :t] = torch.tensor(r["labels"])
        tp[i, :t] = torch.arange(n - t, n)
    truth = {k: torch.tensor([r[k] for r in rows]) for k in ("modulus", "x", "time_steps", "result")}
    return ids, ids != 0, lab, tp, truth


def exact_tdc():
    r = torch.arange(10)[:, None, None]; m = torch.arange(10)[None, :, None]; c = torch.arange(2)[None, None, :]
    x = r - m - c
    a = F.one_hot(x % 10, 10).float(); b = F.one_hot((x < 0).long(), 2).float()
    return torch.cat([a, b], -1).permute(0, 2, 1, 3).reshape(20, 120)


def borrow_soft(Dd, Md, TDC):
    """Final borrow of D - N on digit distributions D, N [..., P, 10]: P(borrow) [...]."""
    sh = Dd.shape[:-2]
    Dd, Md = Dd.reshape(-1, *Dd.shape[-2:]), Md.reshape(-1, *Md.shape[-2:])
    X = Dd.shape[0]
    c = torch.zeros(X, 2); c[:, 0] = 1
    for j in range(Dd.shape[1]):
        joint = (Dd[:, j, :, None] * c[:, None, :]).reshape(X, 20)
        c = torch.einsum("xmo,xm->xo", (joint @ TDC).view(X, 10, 12)[..., 10:], Md[:, j])
    return c[:, 1].view(sh)


ZERO = F.one_hot(torch.tensor(0), 10).float()


class IO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dg = torch.nn.Parameter(INIT * torch.randn(17))          # token -> is a digit
        self.cv = torch.nn.Parameter(INIT * torch.randn(17, 10))      # reader: token -> digit value
        self.wc = torch.nn.Parameter(INIT * torch.randn(10, 17))      # writer: value -> token
        self.anc = torch.nn.Parameter(INIT * torch.randn(4, 18))      # field anchor: a token id, or END
        self.strd = torch.nn.Parameter(INIT * torch.randn(4, 2))      # field step: +1 | -1

    def ext(self, ids, mask):
        """Token ids with END at the prompt's end and OUT beyond, padded with OUT by W + 1 on each side."""
        end = mask.long().sum(1)
        p = torch.arange(ids.shape[1] + 1)[None]
        e = torch.where(p < end[:, None], F.pad(ids, (0, 1)), torch.where(p == end[:, None], END, OUT))
        return F.pad(e, (W + 1, W + 1), value=OUT), end

    def digitness(self, e):
        return torch.where(e < 17, torch.sigmoid(self.dg[e.clamp(max=16)]), torch.zeros(()))

    def wsm(self):
        """The writer's code P(t | v) [10, 17]: only tokens believed to be digits can be written."""
        return (self.wc + F.logsigmoid(self.dg)[None]).softmax(-1)

    def anchor(self, f, e, end):
        p = torch.arange(e.shape[1])[None]
        ok = (p >= W + 1) & (p <= W + 1 + end[:, None])
        return self.anc[f][e.clamp(max=END)].masked_fill(~ok, -1e9).softmax(-1)

    def hard_field(self, f, e, end):
        """Snapped reading of field f: digits [X, n] (LSB first; dead slots 0) and positions [X, n] (padded)."""
        p = torch.arange(e.shape[1])[None]
        ok = (p >= W + 1) & (p <= W + 1 + end[:, None])
        q = self.anc[f][e.clamp(max=END)].masked_fill(~ok, -1e9).argmax(-1)
        s = torch.tensor([1, -1])[self.strd[f].argmax()]
        pos = q[:, None] + (torch.arange(SLOTS[f])[None] + 1) * s
        t = e.gather(1, pos)
        live = ((t < 17) & (self.dg[t.clamp(max=16)] > 0)).long().cumprod(-1).bool()
        return torch.where(live, self.cv[t.clamp(max=16)].argmax(-1), 0), pos

    def runs(self, e, end):
        """Hard digit runs: rows with exactly three; each run's first and last position (padded) and length."""
        p = torch.arange(e.shape[1])[None]
        inside = (p >= W + 1) & (p < W + 1 + end[:, None])
        isd = inside & (e < 17) & (self.dg[e.clamp(max=16)] > 0)
        start = isd & ~F.pad(isd[:, :-1], (1, 0))
        stop = isd & ~F.pad(isd[:, 1:], (0, 1))
        ok = start.sum(1) == 3
        Lp = e.shape[1]
        si = torch.where(start, p, Lp - 2).sort(1).values[:, :3]
        ti = torch.where(stop, p, Lp - 2).sort(1).values[:, :3]
        return ok, si, ti, ti - si + 1

    def lessons(self, ids, mask, lab, tp, TDC, stage):
        e, end = self.ext(ids, mask)
        X, Lp = e.shape
        d = self.digitness(e)
        cvd = self.cv.detach().softmax(-1)                                          # the reader's code, held fixed
        wsm = self.wsm()
        parts = {}
        # ---- the answer round trip, per row: a mixture over the writer's programs
        labe = torch.full_like(e, OUT)
        valid = lab != -100
        labe.scatter_(1, torch.where(valid, tp + W + 1, 0), torch.where(valid, lab, OUT))
        labe[:, 0] = OUT
        dl = self.digitness(labe)
        rt_tok = (cvd * wsm.T).sum(-1)                                              # [17] P(write t | read t)
        a = self.anchor(3, e, end)
        q = torch.arange(Lp)[None, :, None]
        lps = []
        for k, s in enumerate((1, -1)):
            pos = (q + (torch.arange(W)[None, None] + 1) * s).clamp(0, Lp - 1).expand(X, Lp, W)
            tok = labe.gather(1, pos.reshape(X, -1)).view(X, Lp, W)
            live = dl.gather(1, pos.reshape(X, -1)).view(X, Lp, W).cumprod(-1)
            tgt = tok < 17
            pw = live * rt_tok[tok.clamp(max=16)]
            lp = (tgt * pw.clamp_min(EPS).log()).sum(-1) + (valid.sum(1)[:, None] - tgt.sum(-1)) * float(torch.log(torch.tensor(EPS)))
            lps.append(lp + a.clamp_min(1e-30).log() + self.strd[3].log_softmax(-1)[k])
        r1 = -torch.logsumexp(torch.stack(lps, -1).reshape(X, -1), -1).mean()
        parts["ans"] = r1
        # ---- the code round trips (train the writer's code against the reader's): tokens of the answers, and the
        # machine's own numbers (here: the digits of the read x squared)
        tl = labe[labe < 17]
        with torch.no_grad():
            xd, _ = self.hard_field(1, e, end)
            xs = (xd * P10).sum(1)
            vals = ((xs * xs) % 10 ** 8)[:, None] // P10 % 10
        rv = (wsm[vals] * cvd.T[vals]).sum(-1)                                       # P(read v | write v)
        rc = -rt_tok[tl].clamp_min(EPS).log().mean() - rv.clamp_min(EPS).log().mean()
        parts["code"] = rc
        loss = r1 + rc
        # ---- the tiling with "x < N": per row, a mixture over complete readings
        ok, si, ti, length = self.runs(e, end)
        parts["rows3"] = ok.float().mean()
        if stage >= 2 and ok.any():
            v = self.cv[e.clamp(max=16)].softmax(-1)
            A = torch.stack([self.anchor(f, e, end) for f in range(3)], 1)          # [X, 3, Lp]
            sl = torch.stack([self.strd[f].softmax(-1) for f in range(3)])          # [3, 2]
            fits = length[:, None, :] <= torch.tensor(SLOTS[:3])[None, :, None]     # [X, field, run]
            left = A.gather(2, (si - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 0]
            right = A.gather(2, (ti + 1).clamp(max=Lp - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 1]
            rho = torch.stack([left, right], -1) * fits[..., None]                  # [X, field, run, dir]
            j = torch.arange(W)
            inrun = (j[None, None] < length[..., None]).float()[..., None]          # [X, run, slot, 1]
            vp = v.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1, 1).expand(-1, -1, 10)).view(X, 3, W, 10)
            vm = v.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1, 1).expand(-1, -1, 10)).view(X, 3, W, 10)
            Vr = torch.stack([inrun * vp + (1 - inrun) * ZERO, inrun * vm + (1 - inrun) * ZERO], 2)  # [X, run, dir, 8, 10]
            Vr = torch.cat([Vr, ZERO.expand(X, 3, 2, P - W, 10)], 3)
            # P(x' < N') for each (x's run, dir, N's run, dir)
            Dx = Vr[:, :, :, None, None].expand(X, 3, 2, 3, 2, P, 10)
            Dn = Vr[:, None, None].expand(X, 3, 2, 3, 2, P, 10)
            pord = borrow_soft(Dx, Dn, TDC)                                          # [X, rx, dx, rn, dn]
            lik = torch.zeros(X)
            for sN, sx, sT in PERMS:
                inner = (rho[:, 1, sx, :, None] * rho[:, 0, sN, None, :] * pord[:, sx, :, sN, :]).sum((1, 2))
                lik = lik + rho[:, 2, sT].sum(-1) * inner
            r3 = -(lik.clamp_min(EPS).log() * ok).sum() / ok.sum()
            parts["tile_order"] = r3
            loss = loss + r3
        return loss, {k: round(float(v), 4) for k, v in parts.items()}

    @torch.no_grad()
    def hard_checks(self, ids, mask, lab, tp, TDC):
        """Label-free / round-trip checks of the snapped I/O (no truth used)."""
        e, end = self.ext(ids, mask)
        ok, si, ti, length = self.runs(e, end)
        labe = torch.full_like(e, OUT)
        valid = lab != -100
        labe.scatter_(1, torch.where(valid, tp + W + 1, 0), torch.where(valid, lab, OUT)); labe[:, 0] = OUT
        _, pos = self.hard_field(3, e, end)
        t = labe.gather(1, pos)
        live = ((t < 17) & (self.dg[t.clamp(max=16)] > 0)).long().cumprod(-1).bool()
        back = torch.where(live, self.wsm().argmax(-1)[self.cv[t.clamp(max=16)].argmax(-1)], -1)
        o = torch.full_like(e, -1).scatter_(1, pos, back)
        ans = (torch.where(valid, o.gather(1, torch.where(valid, tp + W + 1, 0)), lab) == lab).all(1)
        Nd, pN = self.hard_field(0, e, end); xd, px = self.hard_field(1, e, end); Td, pT = self.hard_field(2, e, end)
        xlt = (xd * P10).sum(1) < (Nd * P10).sum(1)                                  # (integer compare in the lab)
        return {"ans": round(float(ans.float().mean()), 4), "rows3": round(float(ok.float().mean()), 4),
                "x<N": round(float(xlt.float().mean()), 4)}

    @torch.no_grad()
    def check(self, ids, mask, lab, tp, truth):
        e, end = self.ext(ids, mask)
        Nd, _ = self.hard_field(0, e, end); xd, _ = self.hard_field(1, e, end); Td, _ = self.hard_field(2, e, end)
        out = {"N": (Nd * P10).sum(1) == truth["modulus"], "x": (xd * P10).sum(1) == truth["x"],
               "T": (Td * P10[:2]).sum(1) == truth["time_steps"]}
        yd = (truth["result"][:, None] // P10) % 10
        _, pos = self.hard_field(3, e, end)
        tok = self.wsm().argmax(-1)[yd]
        o = torch.full_like(e, -1).scatter_(1, pos.clamp(0, e.shape[1] - 1), tok)
        valid = lab != -100
        out["write"] = (torch.where(valid, o.gather(1, torch.where(valid, tp + W + 1, 0)), lab) == lab).all(1)
        return {k: round(float(v.float().mean()), 4) for k, v in out.items()}

    def describe(self):
        s = []
        for f, nm in enumerate("NxTw"):
            a = int(self.anc[f].argmax()); st = ["+1", "-1"][int(self.strd[f].argmax())]
            pr = self.strd[f].softmax(-1).max()
            s.append(f"{nm}:anchor={a if a < 17 else 'END'},step={st}({float(pr):.3f})")
        code = "".join(str(int(c)) for c in self.cv.argmax(-1)[7:])
        dig = "".join("d" if g > 0 else "." for g in self.dg.tolist())
        wr = ",".join(str(int(t)) for t in self.wsm().argmax(-1))
        return " ".join(s) + f" code(7..16)={code} digitness={dig} writer={wr}"


def batch_of(pk, idx):
    """A batch from packed rows, shaped like the evaluator's: ids, mask, labels (left-aligned), target positions."""
    ids, lab, num = pk["ids"][idx], pk["lab"][idx], pk["num"][idx]
    L = int((ids != 0).sum(1).max())
    ids = ids[:, :L]
    mask = ids != 0
    end = mask.sum(1)
    cnt = (lab != -100).sum(1)
    k = torch.arange(lab.shape[1])[None]
    tp = torch.where(k < cnt[:, None], end[:, None] - cnt[:, None] + k, -1)
    truth = {"modulus": num[:, 0], "x": num[:, 1], "time_steps": num[:, 2], "result": num[:, 3]}
    return ids, mask, lab, tp, truth


def main(steps, B, seed):
    torch.manual_seed(seed)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from packdata import pack
    pk = pack("train")
    m = IO()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    TDC = exact_tdc()
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    stage = 1
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pk["ids"]), (B,), generator=g)
        ids, mask, lab, tp, truth = batch_of(pk, idx)
        loss, parts = m.lessons(ids, mask, lab, tp, TDC, stage)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        hk = m.hard_checks(ids, mask, lab, tp, TDC)
        if stage == 1 and hk["rows3"] >= 0.99:
            stage = 2; print(f"stage 2 at {step}", flush=True)
        if step % 25 == 0 or step == 1:
            print(f"step {step} stage {stage} loss {float(loss):.4f} {parts} hard {hk} truth {m.check(ids, mask, lab, tp, truth)} "
                  f"{time.time()-t0:.1f}s", flush=True)
            if step % 100 == 0:
                print("   ", m.describe(), flush=True)
    print(m.describe())
    for split in ["depth_ood_n_t_1", "depth_ood_n_t_32", "depth_t_64"]:
        pks = pack(split)
        ids, mask, lab, tp, truth = batch_of(pks, torch.arange(len(pks["ids"])))
        ck = m.check(ids, mask, lab, tp, truth)
        w8 = truth["modulus"] >= 10 ** 7
        e, end = m.ext(ids, mask)
        Nd, _ = m.hard_field(0, e, end)
        n8 = float(((Nd * P10).sum(1) == truth["modulus"])[w8].float().mean()) if w8.any() else -1
        print(split, ck, "8-digit N read right:", n8, f"({int(w8.sum())} rows)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
