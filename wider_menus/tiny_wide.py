"""The wide grammar in a tiny H1 step (MENUS.md section 2): square x, reduce mod N, with the multiply's layout and the
division's wiring learned from wide menus (wide.py), every table at the truth (the wiring's search alone; the H1 file
learns the tables too). Base 10, N of `widths` digits in W slots (coprime to 10, as H1's Blum moduli), x < N, 2% of
rows labeled; certified on the whole pool and on rows with N of every width up to W (the wider ones never trained).

Lessons (each switch block is a mixture of whole combinations with crisp verdicts, TINYROUTING.md section 6):
  layout   the routed ladder (WIRING.md section 3): the soft row_j(X, b + 1) (column rules blended on the wire) against
           the hard row_j(X, b) + X 10^j along the verified frontier, for X in the rows' x and N; the number rule by
           distributivity: X * X under each number rule must equal the sum of its single-digit rows
  chain    (once the layout has been exact for `win` steps) chain_law:
             ladder  one step from x + q N (q = 0..9) must give back x (WIRING.md's division ladder: it states long
                     division's step)
             joint   no step law: the chain and the I/O switches enumerated together, taught by the whole-division laws
             whole   no step law: the chain taught by the whole-division laws with the I/O at its argmax, and the I/O
                     by them with the chain at its argmax
  io       (ladder: once the argmax chain has passed the division ladder for `win` steps) the whole-division laws
           `laws`: x (x mod N = x), n (N mod N = 0), q (x + Q N mod N = x, Q random W digits, Q N by the machine's own
           rows), a (the labeled rows' answers through the whole step); io = enum: every combination of the I/O block;
           cs: each switch's options against sampled combinations of the others; local: the step's switches (bring,
           post) by the step law "the register x + q N split into its upper places and its lowest digit: bringing the
           digit back in and reducing gives x", the reading's (vread, start, order, final, read) by the whole-division
           laws.
A run stops `done` steps after the argmax machine first passes every lesson and every batch answer, or at max_steps.

    ~/github/sotaku/venv/bin/python -m wider_menus.tiny_wide seed=0 div=1 mul=1 [KEY=VALUE ...]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import wide as G
except ImportError:                       # run as a script
    import wide as G

CFG = dict(
    W=3, widths=(2,), pool=4000, batch=128, label_frac=0.02, seed=0, init=0.1, lr=0.05,
    div=1, mul=1,                 # menu levels (wide.div_menu, wide.mul_menu)
    io="enum",                    # enum | cs | local
    chain_law="ladder",           # ladder | joint | whole
    laws="xna",                   # the whole-division laws (and answers) of the I/O lessons
    win=10, done=30, max_steps=1500,
    lad_rows=32, io_rows=16, cs_samples=8,
    cert_rows=4000, log=25, threads=0,
)


def coprime_moduli(w: int) -> np.ndarray:
    v = np.arange(max(10 ** (w - 1), 3), 10 ** w)
    return v[(v % 2 != 0) & (v % 5 != 0)]


class Data:
    def __init__(self, W, N, x, lab):
        V = x * x
        self.N, self.x, self.y = N, x, V % N
        self.Nd = G.digits(torch.from_numpy(N), W)
        self.xd = G.digits(torch.from_numpy(x), W)
        self.yd = G.digits(torch.from_numpy(V % N), W)
        self.lab = torch.from_numpy(lab)
        self.n = len(N)

    @classmethod
    def train(cls, W, widths, n, seed, label_frac):
        rng = np.random.default_rng(seed)
        N = rng.choice(np.concatenate([coprime_moduli(w) for w in widths]), n)
        x = (rng.random(n) * N).astype(np.int64)
        return cls(W, N, x, rng.random(n) < label_frac)

    @classmethod
    def cert(cls, W, n, seed):
        rng = np.random.default_rng(seed + 1000)
        N = np.concatenate([rng.choice(coprime_moduli(w), n // W) for w in range(1, W + 1)])
        x = (rng.random(len(N)) * N).astype(np.int64)
        return cls(W, N, x, np.zeros(len(N), dtype=bool))


def mixture_nll(lw: torch.Tensor, verdict: torch.Tensor) -> torch.Tensor:
    """-log sum_k w_k v_k per observation: lw [K] log weights, verdict [K, n]. Zero gradient where nothing passes."""
    return -torch.logsumexp(lw[:, None] + verdict.double().clamp_min(1e-12).log(), 0)


class Tiny(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        W = cfg["W"]
        self.W, self.P, self.C = W, W + 1, 2 * W + 2
        self.dm, self.mm = G.div_menu(cfg["div"], W), G.mul_menu(cfg["mul"], W)
        g = torch.Generator().manual_seed(cfg["seed"])
        r = lambda n: torch.nn.Parameter(cfg["init"] * torch.randn(n, generator=g, dtype=torch.float64))
        self.sw = torch.nn.ParameterDict({k: r(len(v)) for k, v in {**self.dm, **self.mm}.items()})
        self.maps1 = F.one_hot(G.rule_maps(self.mm["lo"], W), W + 1).double()     # [R, W, C, W + 1]
        self.cache = {}

    def am(self, k: str) -> int:
        return int(self.sw[k].argmax())

    def lp(self, k: str) -> torch.Tensor:
        return self.sw[k].log_softmax(-1)

    # ------------------------------------------------------------ the multiply
    def cols(self, num: int | None = None, rows_only: bool = False) -> torch.Tensor:
        key = (self.am("lo"), self.am("hi"), self.am("num") if num is None else num, rows_only)
        if key not in self.cache:
            mm = self.mm
            self.cache[key] = G.claims(mm["lo"][key[0]], mm["hi"][key[1]], mm["num"][key[2]], self.W, rows_only)
        return self.cache[key]

    def square(self, Xd: torch.Tensor) -> torch.Tensor:
        """The machine's square of X (argmax layout): its 2W digits."""
        return G.digits(G.mul_val(Xd, Xd, self.cols()) % 10 ** (2 * self.W), 2 * self.W)

    def row(self, Xd, b, j) -> torch.Tensor:
        return G.row_val(Xd, b, j, self.cols(rows_only=True))

    def mult(self, Nr: torch.Tensor) -> torch.Tensor:
        """The multiples 1..9 of the N register by the machine's own row 0 [X, 9]."""
        return torch.stack([self.row(Nr, torch.full_like(Nr[:, 0], q), 0) for q in range(1, G.S)], 1) % 10 ** self.P

    # ------------------------------------------------------------ the division's wiring
    def idx(self, X: int, over: dict | None = None) -> dict:
        """Per-row option indices: the argmax of every switch, overridden by `over` (name -> [X] indices)."""
        out = {k: torch.full((X,), self.am(k), dtype=torch.long) for k in self.dm}
        out.update(over or {})
        return out

    def divide(self, Vd, Nd, ix: dict) -> torch.Tensor:
        return G.divide_val(Vd, Nd, G.decode(self.dm, ix, self.W), self.mult)

    def step(self, R, d, Nd, ix: dict, bring: bool = True) -> torch.Tensor:
        """One step from the register R: (bring: the move and the incoming digit d), the chain, (bring: the post-move)."""
        w = G.decode(self.dm, ix, self.W)
        MOD = 10 ** self.P
        R0 = R
        if bring:
            R = G.move(R, w["k"], MOD)
            pw = 10 ** w["p"].clamp(min=0)
            rep = R - ((R // pw) % 10) * pw + d * pw
            R = torch.where(w["p"] < 0, R, torch.where(w["mode"] == 0, rep, (R + d * pw) % MOD))
        Nr = torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd)
        if bring:
            R = G.chain_val(R, G.value(Nr), self.mult(Nr), w, MOD, R0, d)
            return G.move(R, w["post"], MOD)
        return G.chain_val(R, G.value(Nr), self.mult(Nr), w, MOD)

    def names(self) -> dict:
        return {k: self.dm[k][self.am(k)] for k in self.dm} | {k: self.mm[k][self.am(k)] for k in self.mm}


def classify(w: dict) -> dict:
    """Which known algorithm the found wiring is, part by part (the report's labels)."""
    key = (w["a"], w["b"], w["op"], w["flag"], w["keep"])
    chain = {("D", "M", "sub", "borrow", "gated"): "repeated subtraction",
             ("D", "M", "sub", "lt", "gated"): "repeated subtraction",          # D < N is D - N's borrow
             ("Rp", "Mq", "sub", "borrow", "gated"): "trial multiples"}.get(key, "other")
    num = {(0, 1): "rows by multiplier digit", (1, 0): "rows by multiplicand digit", (1, -1): "diagonals i - j",
           (-1, 1): "diagonals j - i", (0, -1): "rows by multiplier digit, reversed",
           (-1, 0): "rows by multiplicand digit, reversed", (-1, -1): "diagonals i + j, reversed",
           (1, 1): "anti-diagonals i + j"}.get(tuple(w["num"]), "other")
    return {"chain": chain, "numbers": num}


def run(cfg: dict, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    W = cfg["W"]
    P, C = W + 1, 2 * W + 2
    data = Data.train(W, cfg["widths"], cfg["pool"], cfg["seed"], cfg["label_frac"])
    cert = Data.cert(W, cfg["cert_rows"], cfg["seed"])
    model = Tiny(cfg)
    dm, mm = model.dm, model.mm
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    log = lambda p: p.clamp_min(1e-12).log()
    laws, law = cfg["laws"], cfg["chain_law"]
    if law == "joint":
        blocks = [("joint", G.CHAIN + G.IO)]
    elif cfg["io"] == "local":
        blocks = [("step", ("bring", "post")), ("read", ("vread", "start", "order", "final", "read"))]
    else:
        blocks = [("io", G.IO)]
    if law == "whole":
        blocks = blocks + [("chain", G.CHAIN)]
    BC = {nm: G.combos(dm, names) for nm, names in blocks if not (cfg["io"] == "cs" and nm == "io")}
    CC = G.combos(dm, G.CHAIN)
    sizes = {"div": G.size(dm), "chain": G.size(dm, G.CHAIN), "io": G.size(dm, G.IO), "mul": G.size(mm),
             "total": G.size(dm) * G.size(mm), "enumerated": {nm: int(c.shape[0]) for nm, c in BC.items()}}
    if verbose:
        print("SIZES " + json.dumps(sizes), flush=True)
    st = {"layout": False, "chain": False}
    streak = {"layout": 0, "chain": 0, "all": 0}
    events, t0, stop_at, step, hist = {}, time.time(), None, 0, []

    def fmix(names, cmb, verdict):
        lw = sum(model.lp(n)[cmb[:, i]] for i, n in enumerate(names))
        return mixture_nll(lw, verdict).mean()

    def law_verdicts(ix, xi, Ni, Qr):
        """[rows] bool: every whole-division law in `laws` holds for the per-row wiring ix (inputs pre-repeated)."""
        vs = []
        z = torch.zeros_like(xi)
        if "x" in laws:
            vs.append((model.divide(torch.cat([xi, z], 1), Ni, ix) == xi).all(1))
        if "n" in laws:
            vs.append((model.divide(torch.cat([Ni, z], 1), Ni, ix) == 0).all(1))
        if "q" in laws:
            vs.append((model.divide(Qr, Ni, ix) == xi).all(1))
        if not vs:                                                           # the answers alone teach the I/O
            return torch.ones(xi.shape[0], dtype=torch.bool)
        return torch.stack(vs).all(0)

    for step in range(1, cfg["max_steps"] + 1):
        idx = torch.from_numpy(rng.integers(0, data.n, cfg["batch"]))
        Nd, xd, yd, lab = data.Nd[idx], data.xd[idx], data.yd[idx], data.lab[idx]
        X = Nd.shape[0]
        loss = torch.zeros((), dtype=torch.float64)
        # ------------------------------------------------ the layout: routed ladder (column rules) and number rule
        Xn = torch.cat([xd[: cfg["lad_rows"]], Nd[: cfg["lad_rows"]]])
        n = Xn.shape[0]
        with torch.no_grad():
            hard = torch.stack([torch.stack([model.row(Xn, torch.full((n,), b), j) for b in range(10)], 1)
                                for j in range(W)], 1)                          # [n, W, 10] values
            Xj = torch.stack([G.value(Xn) * 10 ** j for j in range(W)], 1)       # [n, W]
            tgt = (hard[:, :, :9] + Xj[:, :, None]) % 10 ** C                    # rung b + 1's target
            ok = torch.cat([hard[:, :, :1] == 0, hard[:, :, 1:] == tgt], 2)      # [n, W, 10]
            ver = ok.long().cumprod(2).bool()
            lad_ok = bool(ok.all())
        SLO = torch.einsum("r,rjcs->jcs", model.lp("lo").exp(), model.maps1)
        SHI = torch.einsum("r,rjcs->jcs", model.lp("hi").exp(), model.maps1)
        lloss = 0.0
        for b in range(9):
            prod = Xn * (b + 1)
            lo = F.one_hot(F.pad(prod % 10, (0, 1)), 10).double()                # [n, W + 1, 10] (last: nothing = 0)
            hi = F.one_hot(F.pad(prod // 10, (0, 1)), 10).double()
            for j in range(W):
                if not bool(ver[:, j, b].any()):
                    continue
                soft = G.add_soft_exact(torch.einsum("cs,nsk->nck", SLO[j], lo), torch.einsum("cs,nsk->nck", SHI[j], hi))
                nll = -log(soft).gather(-1, G.digits(tgt[:, j, b], C)[..., None]).squeeze(-1).sum(-1)
                lloss = lloss + (nll * ver[:, j, b].double()).mean()
        loss = loss + lloss / (9 * W)
        with torch.no_grad():                                                    # number rules: distributivity
            ref = sum(model.row(Xn, Xn[:, j], j) for j in range(W)) % 10 ** C
            nv = torch.stack([G.mul_val(Xn, Xn, model.cols(num=r)) == ref for r in range(len(mm["num"]))])
            num_ok = bool(nv[model.am("num")].all())
        if len(mm["num"]) > 1:
            loss = loss + mixture_nll(model.lp("num"), nv).mean()
        streak["layout"] = streak["layout"] + 1 if (lad_ok and num_ok) else 0
        if not st["layout"] and streak["layout"] >= cfg["win"]:
            st["layout"] = True
            events["layout"] = step
        # ------------------------------------------------ the division ladder: one step from x + q N gives back x
        nl = min(cfg["lad_rows"], X)
        with torch.no_grad():
            q = torch.arange(10).repeat(nl)
            xs, Ns = xd[:nl].repeat_interleave(10, 0), Nd[:nl].repeat_interleave(10, 0)
            xv = G.value(xs)
            reg = (xv + model.row(Ns, q, 0)) % 10 ** P                          # x + q N by the machine's row 0
            chain_ok = bool((model.step(reg, None, Ns, model.idx(len(q)), bring=False) == xv).all())
        if st["layout"] and law == "ladder":
            K = CC.shape[0]
            with torch.no_grad():
                over = {nm: CC[:, i].repeat_interleave(len(q)) for i, nm in enumerate(G.CHAIN)}
                vc = model.step(reg.repeat(K), None, Ns.repeat(K, 1), model.idx(K * len(q), over), bring=False)
                vc = (vc == xv.repeat(K)).view(K, len(q))
            loss = loss + fmix(G.CHAIN, CC, vc)
        streak["chain"] = streak["chain"] + 1 if chain_ok else 0
        if st["layout"] and not st["chain"] and streak["chain"] >= cfg["win"] and law == "ladder":
            st["chain"] = True
            events["chain"] = step
        # ------------------------------------------------ the whole division's laws (the I/O, and joint / whole)
        ni = min(cfg["io_rows"], X)
        xi, Ni = xd[:ni], Nd[:ni]
        with torch.no_grad():
            Qr = None
            if "q" in laws:
                Q = torch.from_numpy(rng.integers(0, 10, (ni, W)))
                QN = sum(model.row(Ni, Q[:, j], j) for j in range(W))           # Q N by the machine's rows
                Qr = G.digits((QN + G.value(xi)) % 10 ** (2 * W), 2 * W)
            sel_l = torch.nonzero(lab).flatten() if "a" in laws else torch.zeros(0, dtype=torch.long)
            Va = model.square(xd[sel_l])
            io_ok = bool(law_verdicts(model.idx(ni), xi, Ni, Qr).all())
            if sel_l.numel():
                io_ok = io_ok and bool((model.divide(Va, Nd[sel_l], model.idx(sel_l.numel())) == yd[sel_l]).all())
        whole_on = st["chain"] if law == "ladder" else st["layout"]
        if whole_on:
            for bname, names in blocks:
                if bname == "io" and cfg["io"] == "cs":
                    loss = loss + cs_loss(model, names, xi, Ni, Qr, law_verdicts, cfg)
                    continue
                cmb = BC[bname]
                K = cmb.shape[0]
                with torch.no_grad():
                    if bname == "step":                                      # the step law (local)
                        over = {nm: cmb[:, i].repeat_interleave(len(q)) for i, nm in enumerate(names)}
                        v = model.step((reg // 10).repeat(K), (reg % 10).repeat(K), Ns.repeat(K, 1),
                                       model.idx(K * len(q), over)) == xv.repeat(K)
                        v = v.view(K, len(q))
                    else:
                        over = {nm: cmb[:, i].repeat_interleave(ni) for i, nm in enumerate(names)}
                        v = law_verdicts(model.idx(K * ni, over), xi.repeat(K, 1), Ni.repeat(K, 1),
                                         None if Qr is None else Qr.repeat(K, 1)).view(K, ni)
                loss = loss + fmix(names, cmb, v)
                m = sel_l.numel()
                if m and bname in ("io", "read", "joint"):
                    with torch.no_grad():
                        over = {nm: cmb[:, i].repeat_interleave(m) for i, nm in enumerate(names)}
                        va = (model.divide(Va.repeat(K, 1), Nd[sel_l].repeat(K, 1), model.idx(K * m, over))
                              == yd[sel_l].repeat(K, 1)).all(1).view(K, m)
                    loss = loss + fmix(names, cmb, va)
        # ------------------------------------------------ the update and the schedule
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ans_ok = bool((model.divide(model.square(xd), Nd, model.idx(X)) == yd).all())
        all_ok = lad_ok and num_ok and io_ok and ans_ok and (chain_ok or law != "ladder")
        streak["all"] = streak["all"] + 1 if all_ok else 0
        if all_ok and "first_exact" not in events:
            events["first_exact"] = step
        now = json.dumps(model.names(), default=str)
        if not hist or hist[-1][1] != now:
            hist.append((step, now))
        if verbose and (step % cfg["log"] == 0 or step == 1):
            print(f"step={step} loss={float(loss.detach()):.4f} layout={int(lad_ok)}{int(num_ok)} chain={int(chain_ok)} "
                  f"io={int(io_ok)} ans={int(ans_ok)} st={st} t={time.time() - t0:.1f} wiring={now}", flush=True)
        if streak["all"] >= cfg["done"]:
            stop_at = step
            break
    # ------------------------------------------------ certification and the found wiring
    with torch.no_grad():
        res = {}
        for name, d in (("pool", data), ("cert", cert)):
            out = model.divide(model.square(d.xd), d.Nd, model.idx(d.n))
            res[name] = float((out == d.yd).all(1).double().mean())
    wiring = model.names()
    row = {"cfg": dict(cfg), "sizes": sizes, "certified": res["pool"] == 1.0 and res["cert"] == 1.0,
           "true_wiring": all(wiring[k] == G.DIV_TRUE[k] for k in dm) and all(wiring[k] == G.MUL_TRUE[k] for k in mm),
           "pool_exact": res["pool"], "cert_exact": res["cert"], "stopped": stop_at, "steps": step, "events": events,
           "settled_at": hist[-1][0], "changes": len(hist), "wiring": {k: str(v) for k, v in wiring.items()},
           "max_prob": {k: round(float(model.lp(k).exp().max()), 4) for k in model.sw},
           "classes": classify(wiring), "seconds": round(time.time() - t0, 1)}
    if verbose:
        print("RESULT " + json.dumps(row, default=str), flush=True)
    return row


def cs_loss(model, names, xi, Ni, Qr, law_verdicts, cfg):
    """Coordinate-sampled factorized mixture: for every switch of the block, each of its options against M sampled
    combinations of the block's other switches (from their current probabilities; sample 0 their argmax); an option's
    expected verdict is the share of its samples that pass."""
    dm = model.dm
    ni, M = xi.shape[0], cfg["cs_samples"]
    loss = 0.0
    with torch.no_grad():
        samples = {}
        for nm in names:
            s = torch.multinomial(model.lp(nm).exp(), M, replacement=True)
            s[0] = model.am(nm)
            samples[nm] = s
    for nm in names:
        K = len(dm[nm])
        with torch.no_grad():
            over = {nm2: (torch.arange(K).repeat_interleave(M * ni) if nm2 == nm
                          else samples[nm2].repeat_interleave(ni).repeat(K)) for nm2 in names}
            v = law_verdicts(model.idx(K * M * ni, over), xi.repeat(K * M, 1), Ni.repeat(K * M, 1),
                             None if Qr is None else Qr.repeat(K * M, 1))
            v = v.view(K, M, ni).double().mean(1)                            # [K, ni]
        loss = loss - torch.logsumexp(model.lp(nm)[:, None] + v.clamp_min(1e-12).log(), 0).mean()
    return loss


def main(argv):
    cfg = dict(CFG)
    for a in argv:
        k, v = a.split("=", 1)
        cfg[k] = tuple(int(t) for t in v.split(",")) if isinstance(CFG[k], tuple) else type(CFG[k])(v)
    run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
