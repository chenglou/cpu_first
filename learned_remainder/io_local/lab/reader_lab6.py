"""IO.md §7, items 6-8 (the final lab: per-hypothesis rank codes, the normalized mixture over readings, then the
tiling). Reader/writer-only lab v6 (all training rows; the I/O alone, no machine). One teacher per piece:
  * the answer round trip (read the answer at the writer's positions, write it back), a per-row mixture over the
    writer's programs: the writer's pointer and which tokens are digits;
  * the tiling (each field reads one whole digit run, fields take different runs) with "x < N", a per-row mixture over
    complete readings (which run each field reads, in which direction): the fields' pointers;
  * per reading hypothesis, a rank code (one score per token; a digit's value is its rank among the digit tokens)
    fit by Bradley-Terry on the decisive digit pair of x < N (the highest place where the two readings differ);
  * the code round trips (read a token and write it back; write the machine's numbers and read them back) against the
    selected reader's code: the writer's code.
    python reader_lab6.py STEPS B SEED
"""
import itertools, os, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packdata import pack

END, OUT = 17, 18
W, P = 8, 9
SLOTS = (8, 8, 2, 8)            # fields: N, x, T, writer
INIT, LR = 0.1, float(os.environ.get("LR", "0.05"))
TAU = float(os.environ.get("TAU", "1"))
EPS = 1e-6
PERMS = list(itertools.permutations(range(3)))
HYP = [(rn, dn, rx, dx) for rn in range(3) for rx in range(3) if rn != rx for dn in range(2) for dx in range(2)]
HIDX = {h: i for i, h in enumerate(HYP)}
P10 = 10 ** torch.arange(W)


def batch_of(pk, idx):
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


class IO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dg = torch.nn.Parameter(INIT * torch.randn(17))            # token -> is a digit
        self.u = torch.nn.Parameter(INIT * torch.randn(len(HYP), 17))   # per reading hypothesis: token scores
        self.wc = torch.nn.Parameter(INIT * torch.randn(10, 17))        # writer: value -> token
        self.anc = torch.nn.Parameter(INIT * torch.randn(4, 18))        # field anchor: a token id, or END
        self.strd = torch.nn.Parameter(INIT * torch.randn(4, 2))        # field step: +1 | -1

    # ---------------- shared pieces
    def ext(self, ids, mask):
        end = mask.long().sum(1)
        p = torch.arange(ids.shape[1] + 1)[None]
        e = torch.where(p < end[:, None], F.pad(ids, (0, 1)), torch.where(p == end[:, None], END, OUT))
        return F.pad(e, (W + 1, W + 1), value=OUT), end

    def isdig(self, e):
        return (e < 17) & (self.dg[e.clamp(max=16)] > 0)

    def ranks(self, k):
        """The code of hypothesis k [.., 17]: each digit token's rank among the digit tokens (others: 0)."""
        isd = self.dg > 0
        u = self.u[k]
        below = ((u[..., None, :] < u[..., :, None]) & isd).sum(-1)       # digit tokens scoring below each token
        return torch.where(isd, below, 0).clamp(max=9)

    def wsm(self):
        return (self.wc + F.logsigmoid(self.dg)[None]).softmax(-1)      # writer: only digit tokens are written

    def anchor(self, f, e, end):
        p = torch.arange(e.shape[1])[None]
        ok = (p >= W + 1) & (p <= W + 1 + end[:, None])
        return self.anc[f][e.clamp(max=END)].masked_fill(~ok, -1e9).softmax(-1)

    def hard_pos(self, f, e, end):
        p = torch.arange(e.shape[1])[None]
        ok = (p >= W + 1) & (p <= W + 1 + end[:, None])
        q = self.anc[f][e.clamp(max=END)].masked_fill(~ok, -1e9).argmax(-1)
        s = torch.tensor([1, -1])[self.strd[f].argmax()]
        return q[:, None] + (torch.arange(SLOTS[f])[None] + 1) * s, q, s

    def runs(self, e, end):
        p = torch.arange(e.shape[1])[None]
        inside = (p >= W + 1) & (p < W + 1 + end[:, None])
        isd = inside & self.isdig(e)
        start = isd & ~F.pad(isd[:, :-1], (1, 0))
        stop = isd & ~F.pad(isd[:, 1:], (0, 1))
        ok = start.sum(1) == 3
        Lp = e.shape[1]
        si = torch.where(start, p, Lp - 2).sort(1).values[:, :3]
        ti = torch.where(stop, p, Lp - 2).sort(1).values[:, :3]
        return ok, si, ti, ti - si + 1

    def selected(self, e, end):
        """Per row, the hypothesis the snapped N and x pointers read (-1: not a whole-run reading)."""
        ok, si, ti, length = self.runs(e, end)
        out = []
        for f in (0, 1):
            pos, q, s = self.hard_pos(f, e, end)
            first = q + s
            r = ((first[:, None] >= si) & (first[:, None] <= ti)).float().argmax(1)
            whole = torch.where(s > 0, q == si.gather(1, r[:, None])[:, 0] - 1, q == ti.gather(1, r[:, None])[:, 0] + 1)
            inrun = ((first[:, None] >= si) & (first[:, None] <= ti)).any(1)
            out.append((r, int(s < 0), whole & inrun & ok & (length.gather(1, r[:, None])[:, 0] <= SLOTS[f])))
        (rn, dn, okn), (rx, dx, okx) = out
        lut = torch.full((3, 2, 3, 2), -1, dtype=torch.long)
        for h, i in HIDX.items():
            lut[h] = i
        k = lut[rn, dn, rx, dx]
        return torch.where(okn & okx & (rn != rx), k, -1)

    def hard_field(self, f, e, end, k):
        """Snapped reading of field f with the code of hypothesis k [X]: digits [X, n] (LSB first, dead slots 0)."""
        pos, _, _ = self.hard_pos(f, e, end)
        t = e.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        code = self.ranks(k.clamp(min=0))                                   # [X, 17]
        return torch.where(live, code.gather(1, t.clamp(max=16)), 0), pos

    def label_seq(self, e, lab, tp):
        labe = torch.full_like(e, OUT)
        valid = lab != -100
        labe.scatter_(1, torch.where(valid, tp + W + 1, 0), torch.where(valid, lab, OUT))
        labe[:, 0] = OUT
        return labe, valid

    # ---------------- lessons
    def lessons(self, ids, mask, lab, tp, stage):
        e, end = self.ext(ids, mask)
        X, Lp = e.shape
        wsm = self.wsm()
        parts = {}
        with torch.no_grad():
            k = self.selected(e, end)
            code = self.ranks(k.clamp(min=0))                               # [X, 17] the selected reader's code
        # ---- answer round trip
        labe, valid = self.label_seq(e, lab, tp)
        dl = torch.where(labe < 17, torch.sigmoid(self.dg[labe.clamp(max=16)]), torch.zeros(()))
        a = self.anchor(3, e, end)
        qq = torch.arange(Lp)[None, :, None]
        lps = []
        for kk, s in enumerate((1, -1)):
            pos = (qq + (torch.arange(W)[None, None] + 1) * s).clamp(0, Lp - 1).expand(X, Lp, W)
            tok = labe.gather(1, pos.reshape(X, -1)).view(X, Lp, W)
            live = dl.gather(1, pos.reshape(X, -1)).view(X, Lp, W).cumprod(-1)
            tgt = tok < 17
            tc = tok.clamp(max=16)
            pw = live * wsm.detach()[code.gather(1, tc.reshape(X, -1)).view(X, Lp, W), tc]  # read, written back
            lp = (tgt * pw.clamp_min(EPS).log()).sum(-1) + (valid.sum(1)[:, None] - tgt.sum(-1)) * float(torch.log(torch.tensor(EPS)))
            lps.append(lp + a.clamp_min(1e-30).log() + self.strd[3].log_softmax(-1)[kk])
        parts["ans"] = -torch.logsumexp(torch.stack(lps, -1).reshape(X, -1), -1).mean()
        lv = labe < 17
        tl = labe[lv]
        # ---- writer's code round trips against the selected reader's code (token side, value side)
        rows_l = lv.nonzero()[:, 0]
        rt = wsm[code[rows_l, tl], tl]                                      # read t, write back t
        with torch.no_grad():
            xd, _ = self.hard_field(1, e, end, k)
            xs = (xd * P10).sum(1)
            vals = ((xs * xs) % 10 ** 8)[:, None] // P10 % 10                # the machine's numbers (stand-in)
            inv = torch.full((X, 10), 0, dtype=torch.long)
            isd = (self.dg > 0)
            for t in range(17):
                if isd[t]:
                    inv.scatter_(1, code[:, t:t + 1], t)                    # the token whose rank is v
        rv = wsm[vals, inv.gather(1, vals)]                                 # write v, read back v
        parts["code"] = -rt.clamp_min(EPS).log().mean() - (rv.clamp_min(EPS).log().mean() if stage >= 3 else 0)
        loss = parts["ans"] + parts["code"]
        # ---- tiling with "x < N", per row a mixture over complete readings; Bradley-Terry per hypothesis
        ok, si, ti, length = self.runs(e, end)
        parts["rows3"] = ok.float().mean()
        if stage >= 2 and ok.any():
            j = torch.arange(W)
            tokp = e.gather(1, (si[..., None] + j).clamp(max=Lp - 1).reshape(X, -1)).view(X, 3, W)
            tokm = e.gather(1, (ti[..., None] - j).clamp(min=0).reshape(X, -1)).view(X, 3, W)
            liv = j[None, None] < length[..., None]
            R = torch.stack([torch.where(liv, tokp, -1), torch.where(liv, tokm, -1)], 2)   # [X, run, dir, 8], -1 dead
            Rp = F.pad(R, (0, P - W), value=-1)
            hyp = torch.tensor(HYP)
            rx_, dx_, rn_, dn_ = hyp[:, 2], hyp[:, 3], hyp[:, 0], hyp[:, 1]
            Tx = Rp[:, rx_, dx_]                                                # [X, H, P] tokens of x' (-1 dead)
            Tn = Rp[:, rn_, dn_]
            diff = Tx != Tn
            place = torch.where(diff, torch.arange(P), -1).max(-1).values       # decisive place [X, H]
            ta = Tx.gather(2, place.clamp(min=0)[..., None])[..., 0]
            tb = Tn.gather(2, place.clamp(min=0)[..., None])[..., 0]
            u = self.u[None].expand(X, -1, -1)                                   # [X, H, 17]
            isd = (self.dg > 0).detach()
            umin = self.u.masked_fill(~isd[None], 1e9).min(-1).values.detach()   # the rank-0 digit's score [H]
            ua = u.gather(2, ta.clamp(min=0)[..., None])[..., 0]
            ub = u.gather(2, tb.clamp(min=0)[..., None])[..., 0]
            both = (ta >= 0) & (tb >= 0) & (place >= 0)
            padx = (ta < 0) & (tb >= 0) & (place >= 0)                          # x' ended first: N's digit > 0
            lp_bt = torch.where(both, F.logsigmoid((ub - ua) / TAU), torch.where(padx, F.logsigmoid((ub - umin[None]) / TAU),
                                                                                    torch.full_like(ua, float(torch.log(torch.tensor(EPS))))))
            okr = ok[:, None].float()
            bt = -((F.logsigmoid((ub - ua) / TAU) * both) * okr).sum() / (both * okr).sum().clamp_min(1)
            parts["bt"] = bt
            hp = torch.where(both, ub > ua, torch.where(padx, ub > umin[None], torch.zeros_like(both)))
            Lk = (hp.float() + EPS).detach()                                     # [X, H] does x' < N' hold (exact)
            rest = 3 - rn_ - rx_                                                   # the run left for T's field
            tfits = length.gather(1, rest[None].expand(X, -1)) <= SLOTS[2]         # [X, H]
            self.hard_pass = ((hp & tfits & ok[:, None]).float().sum(0) / ok.sum()).detach()   # [H]
            loss = loss + bt
        if stage >= 3 and ok.any():
            # pointer mixture
            A = torch.stack([self.anchor(f, e, end) for f in range(3)], 1)
            sl = torch.stack([self.strd[f].softmax(-1) for f in range(3)])
            fits = length[:, None, :] <= torch.tensor(SLOTS[:3])[None, :, None]
            left = A.gather(2, (si - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 0]
            right = A.gather(2, (ti + 1).clamp(max=Lp - 1)[:, None].expand(X, 3, 3)) * sl[None, :, None, 1]
            rho = torch.stack([left, right], -1) * fits[..., None]              # [X, field, run, dir]
            lik, norm = torch.zeros(X), torch.zeros(X)
            post = torch.zeros(X, len(HYP))
            for sN, sx, sT in PERMS:
                for dn in range(2):
                    for dx in range(2):
                        w = rho[:, 2, sT].sum(-1) * rho[:, 0, sN, dn] * rho[:, 1, sx, dx]
                        hk = HIDX[(sN, dn, sx, dx)]
                        lik = lik + w * Lk[:, hk]
                        norm = norm + w
                        post[:, hk] = post[:, hk] + w.detach()
            # speculation: P(x < N | a whole-run reading), the reading chosen by the exact verdict of each hypothesis
            parts["order"] = -((lik.clamp_min(1e-30).log() - norm.clamp_min(1e-30).log()) * ok).sum() / ok.sum()
            loss = loss + parts["order"]
            self.chosen = float((post / norm.detach()[:, None].clamp_min(1e-30)).max(1).values[ok].mean())
            if stage >= 4:                                                   # then: whole runs, distinct fields
                parts["tile"] = -(norm.clamp_min(EPS).log() * ok).sum() / ok.sum()
                loss = loss + parts["tile"]
        return loss, {kk: round(float(v), 4) for kk, v in parts.items()}

    # ---------------- checks
    @torch.no_grad()
    def hard_checks(self, ids, mask, lab, tp):
        """Label-free / round-trip checks of the snapped I/O (no truth)."""
        e, end = self.ext(ids, mask)
        ok, si, ti, length = self.runs(e, end)
        k = self.selected(e, end)
        code = self.ranks(k.clamp(min=0))
        labe, valid = self.label_seq(e, lab, tp)
        pos, _, _ = self.hard_pos(3, e, end)
        t = labe.gather(1, pos)
        live = self.isdig(t).long().cumprod(-1).bool()
        back = torch.where(live, self.wsm().argmax(-1)[code.gather(1, t.clamp(max=16))], -1)
        o = torch.full_like(e, -1).scatter_(1, pos, back)
        ans = (torch.where(valid, o.gather(1, torch.where(valid, tp + W + 1, 0)), lab) == lab).all(1)
        Nd, _ = self.hard_field(0, e, end, k); xd, _ = self.hard_field(1, e, end, k)
        xlt = ((xd * P10).sum(1) < (Nd * P10).sum(1)) & (k >= 0)
        return {"ans": round(float(ans.float().mean()), 4), "rows3": round(float(ok.float().mean()), 4),
                "sel": int(torch.mode(k).values), "x<N": round(float(xlt.float().mean()), 4)}

    @torch.no_grad()
    def check(self, ids, mask, lab, tp, truth):
        e, end = self.ext(ids, mask)
        k = self.selected(e, end)
        Nd, _ = self.hard_field(0, e, end, k); xd, _ = self.hard_field(1, e, end, k); Td, _ = self.hard_field(2, e, end, k)
        out = {"N": (Nd * P10).sum(1) == truth["modulus"], "x": (xd * P10).sum(1) == truth["x"],
               "T": (Td * P10[:2]).sum(1) == truth["time_steps"]}
        yd = (truth["result"][:, None] // P10) % 10
        pos, _, _ = self.hard_pos(3, e, end)
        tok = self.wsm().argmax(-1)[yd]
        o = torch.full_like(e, -1).scatter_(1, pos.clamp(0, e.shape[1] - 1), tok)
        valid = lab != -100
        out["write"] = (torch.where(valid, o.gather(1, torch.where(valid, tp + W + 1, 0)), lab) == lab).all(1)
        return {kk: round(float(v.float().mean()), 4) for kk, v in out.items()}

    def describe(self, e=None, end=None):
        s = []
        for f, nm in enumerate("NxTw"):
            a = int(self.anc[f].argmax()); st = ["+1", "-1"][int(self.strd[f].argmax())]
            pr = self.strd[f].softmax(-1).max()
            s.append(f"{nm}:anchor={a if a < 17 else 'END'},step={st}({float(pr):.3f})")
        dig = "".join("d" if g > 0 else "." for g in self.dg.tolist())
        wr = ",".join(str(int(t)) for t in self.wsm().argmax(-1))
        return " ".join(s) + f" digitness={dig} writer={wr}"


def main(steps, B, seed):
    torch.manual_seed(seed)
    pk = pack("train")
    m = IO()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    stage = 1
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(pk["ids"]), (B,), generator=g)
        ids, mask, lab, tp, truth = batch_of(pk, idx)
        loss, parts = m.lessons(ids, mask, lab, tp, stage)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        hk = m.hard_checks(ids, mask, lab, tp)
        if stage == 1 and hk["rows3"] >= 0.99:
            stage = 2; win = 0; print(f"stage 2 at {step}", flush=True)
        elif stage == 2:
            win = win + 1 if float(m.hard_pass.max()) >= 0.999 else 0
            if win >= 20:
                stage = 3; print(f"stage 3 at {step} best hypothesis {HYP[int(m.hard_pass.argmax())]} "
                                 f"pass {[round(float(v), 3) for v in m.hard_pass.sort(descending=True).values[:4]]}", flush=True)
        elif stage == 3 and getattr(m, "chosen", 0) > 0.99:
            stage = 4; print(f"stage 4 at {step}", flush=True)
        if step % 50 == 0 or step == 1:
            e, end = m.ext(ids, mask)
            k = m.selected(e, end)
            kk = int(torch.mode(k).values)
            code = "".join(str(int(c)) for c in m.ranks(torch.tensor(max(kk, 0)))[7:]) if kk >= 0 else "-"
            print(f"step {step} stage {stage} {parts} hard {hk} code(7..16)={code} truth {m.check(ids, mask, lab, tp, truth)} "
                  f"{time.time()-t0:.1f}s", flush=True)
            if step % 200 == 0:
                print("   ", m.describe(), "hyp", HYP[kk] if kk >= 0 else None, flush=True)
    print(m.describe())
    for split in ["depth_ood_n_t_1", "depth_ood_n_t_32", "depth_t_64"]:
        pks = pack(split)
        ids, mask, lab, tp, truth = batch_of(pks, torch.arange(len(pks["ids"])))
        ck = m.check(ids, mask, lab, tp, truth)
        e, end = m.ext(ids, mask)
        k = m.selected(e, end)
        Nd, _ = m.hard_field(0, e, end, k)
        w8 = truth["modulus"] >= 10 ** 7
        n8 = float(((Nd * P10).sum(1) == truth["modulus"])[w8].float().mean()) if w8.any() else -1
        print(split, ck, "8-digit N read right:", n8, f"({int(w8.sum())} rows)")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
