"""Tiny core for answer-chosen lessons (LESSONS.md section 3): the wide grammar's tiny step (wider_menus/wide.py, every
table at the truth, base 10, W = 3) with

  * reduction lesson sets as candidates, made by one template from a predicate of a small grammar:
      below    the chain moves its register down by multiples of N until it is below N (long division's ladder: from
               x + q N one chain gives x; the step law; the whole-reduction laws with alignment s = 0: red(a) = a for
               every mined a < N, red(N) = 0, red(x + Q N) = x)
      lowzero  the chain moves its register up by multiples of N until its lowest digit is 0 (from 10 x + q N one chain
               gives 10 x, or 10 (x + N); the step law with the digit added in place; alignment s = W:
               red(a 10^W) = a, red(N 10^W) = 0, red(x 10^W + Q N) = x). Admissible only when the miner has found
               N coprime to 10 (otherwise no multiple of N can clear the lowest digit)
      none     no reduction lessons at all (the trap "every lesson switched off": its wiring learns nothing)
    every candidate has its own division switches, taught only by its own lessons; both share the law "the gated
    subtraction of N keeps a < N and takes a + N to a" (csub);
  * a shared round menu, the same for every task: the square times 10 or not, plus a one-digit constant, times the
    round's own input and reduced again or not;
  * a selector over the candidates, trained only by the answers: a factorized mixture over (candidate, round options)
    whose verdict is "this candidate's argmax machine with these round options reproduces the labeled rows". The
    lesson losses never reach the selector, so a candidate cannot win by having no lessons.

The miner (kind 2) keeps relations among N, x, T and the answer y that held on every row of the last `win` batches:
order (a < b), equality (a = b, a constant across rows), range (a < 10^k) and divisibility (a mod d = r, d in 2, 3, 5).
Order facts a < N give the instances of red(a 10^s) = a; "N mod 2 = 1, N mod 5 != 0" makes lowzero admissible; range
facts are only reported (they can hold by accident: N < 100 on 2-digit training moduli, false on the 3-digit ones).

With gates = "loss" each candidate's lesson losses are multiplied by a soft gate trained by those same losses (the
trap "lesson gates scored by the lessons": their optimum is every gate at 0).

    ~/github/sotaku/venv/bin/python general_lessons/tiny_general.py task=mod seed=0 [KEY=VALUE ...]
(one run is about a minute but peaks near 0.7 GB, so the suites ran on Modal through modal_tiny.py)
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wider_menus"))
import wide as G  # noqa: E402

CFG = dict(
    W=3, widths=(2,), pool=4000, batch=128, label_frac=0.02, seed=0, init=0.1, lr=0.05, div=3,
    task="mod", cands="below,lowzero", radd=10, gates="answers",   # gates: answers | loss (the trap: lesson gates
    win=10, done=30, max_steps=1500, chain_rows=16, io_rows=16,    #   trained by the lesson loss itself)
    cert_rows=3000, log=50, threads=0, out="",
)
TASKS = {   # one round: y -> f(y) mod N (R = 10^W, the Montgomery radix)
    "mod": lambda y, N, R: y * y % N,
    "mont": lambda y, N, R: y * y * pow(R, -1, N) % N,
    "rho": lambda y, N, R: (y * y + 1) % N,
    "ten": lambda y, N, R: 10 * y * y % N,
    "cube": lambda y, N, R: y * y * y % N,
    "tenrho": lambda y, N, R: (10 * y * y + 1) % N,              # held out: x10 and +1 together
    "plus2": lambda y, N, R: (y * y + 2) % N,                     # held out: another constant
    "montrho": lambda y, N, R: (y * y + 1) * pow(R, -1, N) % N,   # held out: +1 then Montgomery's reduction
}
NARROW = ("ten", "tenrho")      # x10 needs N < 10^(W - 1): 10 y^2 keeps a top half below N
CHAIN = G.CHAIN                 # a, b, op, flag, keep, nread, g0, g1
STEP = ("bring", "post")
READ = ("vread", "start", "order", "final", "read")
ROUND = ("rmul", "radd", "rcube")


def coprime(w: int) -> np.ndarray:
    v = np.arange(max(10 ** (w - 1), 3), 10 ** w)
    return v[(v % 2 != 0) & (v % 5 != 0)]


def answers(task: str, N: np.ndarray, x: np.ndarray, T: np.ndarray, W: int) -> np.ndarray:
    f, R = TASKS[task], 10 ** W
    out = []
    for n, y, t in zip(N.tolist(), x.tolist(), T.tolist()):
        for _ in range(t):
            y = f(y, n, R)
        out.append(y)
    return np.array(out, dtype=np.int64)


class Data:
    def __init__(self, cfg, N, x, T, lab):
        W = cfg["W"]
        self.N, self.x, self.T = N, x, T
        self.y = answers(cfg["task"], N, x, T, W)
        self.Nd, self.xd, self.yd = (G.digits(torch.from_numpy(v), W) for v in (N, x, self.y))
        self.Tt, self.lab, self.n = torch.from_numpy(T), torch.from_numpy(lab), len(N)

    @classmethod
    def make(cls, cfg, widths, n, seed, label_frac, Ts):
        rng = np.random.default_rng(seed)
        pool = np.concatenate([coprime(w) for w in widths])
        if cfg["task"] in NARROW:
            pool = pool[pool < 10 ** (cfg["W"] - 1)]
        N = rng.choice(pool, n)
        x = (rng.random(n) * N).astype(np.int64)
        return cls(cfg, N, x, rng.choice(Ts, n), rng.random(n) < label_frac)


class Miner:
    """Relations among the rows' numbers (N, x, T, y), kept while they hold on every row of the last `win` batches."""

    def __init__(self, win: int):
        self.win, self.streak, self.fails, self.seen = win, {}, {}, 0

    @staticmethod
    def relations(v: dict) -> dict:
        out = {}
        names = list(v)
        for a, b in itertools.permutations(names, 2):
            out[f"{a} < {b}"] = v[a] < v[b]
        for a, b in itertools.combinations(names, 2):
            out[f"{a} = {b}"] = v[a] == v[b]
        for a in names:
            out[f"{a} constant"] = v[a] == v[a][:1]
            out[f"{a} != 0"] = v[a] != 0
            for k in range(1, 7):
                out[f"{a} < 10^{k}"] = v[a] < 10 ** k
            for d in (2, 3, 5):
                out[f"{a} mod {d} != 0"] = v[a] % d != 0
                for r in range(d):
                    out[f"{a} mod {d} = {r}"] = v[a] % d == r
        return out

    def update(self, v: dict) -> None:
        self.seen += 1
        for k, ok in self.relations(v).items():
            good = bool(ok.all())
            self.streak[k] = self.streak.get(k, 0) + 1 if good else 0
            self.fails[k] = self.fails.get(k, 0) + (not good)

    def facts(self) -> list:
        return [k for k, s in self.streak.items() if s >= self.win]

    def holds(self, k: str) -> bool:
        return self.streak.get(k, 0) >= self.win

    def report(self) -> dict:
        """Mined facts, without those a stronger mined fact implies (a < 10^k when a < 10^(k-1); x mod d = r)."""
        f = set(self.facts())
        keep = []
        for k in sorted(f):
            if " < 10^" in k:
                a, e = k.split(" < 10^")
                if f"{a} < 10^{int(e) - 1}" in f:
                    continue
            keep.append(k)
        return {"facts": keep, "batches": self.seen,
                "never_failed": sorted(k for k in keep if self.fails.get(k, 0) == 0)}


def mixture_nll(lw: torch.Tensor, verdict: torch.Tensor) -> torch.Tensor:
    return -torch.logsumexp(lw[:, None] + verdict.double().clamp_min(1e-12).log(), 0)


class Tiny(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        W = cfg["W"]
        self.W, self.P = W, W + 1
        self.dm = G.div_menu(cfg["div"], W)
        self.rm = {"rmul": [1, 10], "radd": list(range(cfg["radd"])), "rcube": ["none", "times_x"]}
        self.cands = cfg["cands"].split(",")
        g = torch.Generator().manual_seed(cfg["seed"])
        r = lambda n: torch.nn.Parameter(cfg["init"] * torch.randn(n, generator=g, dtype=torch.float64))
        self.sw = torch.nn.ModuleList([torch.nn.ParameterDict({k: r(len(v)) for k, v in self.dm.items()})
                                       for _ in self.cands])
        self.rnd = torch.nn.ParameterDict({k: r(len(v)) for k, v in self.rm.items()})
        self.sel = r(len(self.cands))
        self.gate = r(len(self.cands))      # gates = "loss": per-candidate lesson gates trained by the lesson loss
        self.cols = G.claims((1, 1, 0), (1, 1, 1), (0, 1), W)          # the multiply at the truth (not studied here)
        self.rows_cols = G.claims((1, 1, 0), (1, 1, 1), (0, 1), W, rows_only=True)

    # ------------------------------------------------------------ the machine (values, exact tables)
    def ix(self, c: int, X: int, over: dict | None = None) -> dict:
        out = {k: torch.full((X,), int(self.sw[c][k].argmax()), dtype=torch.long) for k in self.dm}
        out.update(over or {})
        return out

    def mult(self, Nr: torch.Tensor) -> torch.Tensor:
        return torch.stack([G.row_val(Nr, torch.full_like(Nr[:, 0], q), 0, self.rows_cols) for q in range(1, G.S)],
                           1) % 10 ** self.P

    def divide(self, Vd, Nd, ix) -> torch.Tensor:
        return G.divide_val(Vd, Nd, G.decode(self.dm, ix, self.W), self.mult)

    def chain(self, reg, Nd, ix) -> torch.Tensor:
        w = G.decode(self.dm, ix, self.W)
        Nr = torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd)
        return G.chain_val(reg, G.value(Nr), self.mult(Nr), w, 10 ** self.P)

    def step(self, R0, d, Nd, ix) -> torch.Tensor:
        """One whole step from the register R0 with the digit d coming in: the bring-in, the chain, the post move."""
        w = G.decode(self.dm, ix, self.W)
        MOD = 10 ** self.P
        R = G.move(R0, w["k"], MOD)
        pw = 10 ** w["p"].clamp(min=0)
        rep = R - ((R // pw) % 10) * pw + d * pw
        R = torch.where(w["p"] < 0, R, torch.where(w["mode"] == 0, rep, (R + d * pw) % MOD))
        Nr = torch.where(w["nread"][:, None] == 1, Nd.flip(-1), Nd)
        R = G.chain_val(R, G.value(Nr), self.mult(Nr), w, MOD, R0, d)
        return G.move(R, w["post"], MOD)

    def round(self, yd, Nd, ix, rmul, radd, rcube) -> torch.Tensor:
        """One round on the digits yd: the square, times 10 (rmul), plus a constant (radd), reduced; with rcube times
        the round's input and reduced again. rmul, radd, rcube: per-row option indices."""
        W = self.W
        V = G.mul_val(yd, yd, self.cols) % 10 ** (2 * W)
        V = torch.where(rmul == 1, V * 10 % 10 ** (2 * W), V)
        V = (V + torch.tensor(self.rm["radd"])[radd]) % 10 ** (2 * W)
        R = self.divide(G.digits(V, 2 * W), Nd, ix)
        if bool((rcube == 1).any()):
            V3 = G.mul_val(R, yd, self.cols) % 10 ** (2 * W)
            R = torch.where((rcube == 1)[:, None], self.divide(G.digits(V3, 2 * W), Nd, ix), R)
        return R

    def run(self, xd, Nd, T, ix, rmul, radd, rcube) -> torch.Tensor:
        y = xd
        for t in range(int(T.max())):
            y = torch.where((T > t)[:, None], self.round(y, Nd, ix, rmul, radd, rcube), y)
        return y

    def round_idx(self, X: int) -> tuple:
        return tuple(torch.full((X,), int(self.rnd[k].argmax()), dtype=torch.long) for k in ROUND)

    def names(self, c: int) -> dict:
        return {k: self.dm[k][int(self.sw[c][k].argmax())] for k in self.dm}


# ------------------------------------------------------------------------------------------------ the lesson template
def template(pred: str, W: int):
    """A reduction lesson set from its predicate: (the chain ladder's registers and targets, the step law's split and
    targets, the whole-reduction laws' alignment s). Values throughout."""
    if pred == "below":                 # move down by multiples of N until below N
        ladder = lambda x, N, q: (x + q * N, x)
        split = lambda reg: (reg // 10, reg % 10)                        # upper places, the lowest digit
        step_target = lambda x, N, q: x
        return ladder, split, step_target, 0
    if pred == "lowzero":               # move up by multiples of N until the lowest digit is 0
        ladder = lambda x, N, q: (10 * x + q * N, torch.where(q == 0, 10 * x, 10 * (x + N)))
        def split(reg):                 # part e of the lowest digit stays in place, the rest comes in
            d = reg % 10
            e = d // 2
            return reg - d + e, d - e
        step_target = lambda x, N, q: torch.where(q == 0, x, x + N)
        return ladder, split, step_target, W
    raise KeyError(pred)


def run(cfg: dict, verbose: bool = True) -> dict:
    torch.manual_seed(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(cfg["threads"])
    W = cfg["W"]
    MOD = 10 ** (W + 1)
    data = Data.make(cfg, cfg["widths"], cfg["pool"], cfg["seed"], cfg["label_frac"], [1, 2])
    cert_widths = list(range(1, W + 1)) if cfg["task"] not in NARROW else list(range(1, W))
    cert = Data.make(cfg, cert_widths, cfg["cert_rows"], cfg["seed"] + 1000, 0.0, [1, 2, 4, 8])
    model = Tiny(cfg)
    dm, cands = model.dm, model.cands
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["seed"] + 1)
    miner = Miner(cfg["win"])
    CC = G.combos(dm, CHAIN)
    SC = G.combos(dm, STEP)
    RC = G.combos(dm, READ)
    UC = G.combos(model.rm, ROUND)
    if verbose:
        print(f"SIZES chain={len(CC)} step={len(SC)} read={len(RC)} round={len(UC)} cands={cands}", flush=True)
    io_on = [False] * len(cands)
    streak = [0] * len(cands)
    done_streak, events, t0, step = 0, {}, time.time(), 0
    log = lambda p: p.clamp_min(1e-12).log()

    def lw_of(params: torch.nn.ParameterDict, names, combos) -> torch.Tensor:
        return sum(params[n].log_softmax(-1)[combos[:, i]] for i, n in enumerate(names))

    while step < cfg["max_steps"]:
        step += 1
        b = torch.from_numpy(rng.choice(data.n, cfg["batch"], replace=False))
        xd, Nd, T, yd, lab = data.xd[b], data.Nd[b], data.Tt[b], data.yd[b], data.lab[b]
        x, N = G.value(xd), G.value(Nd)
        miner.update({"N": N, "x": x, "T": T, "y": G.value(yd)})
        ident = [a for a in ("x", "T", "y") if miner.holds(f"{a} < N")]       # instances of red(a 10^s) = a
        coprime10 = miner.holds("N mod 2 != 0") and miner.holds("N mod 5 != 0")   # lowzero is admissible
        loss = torch.zeros((), dtype=torch.float64)
        ok_all = []
        for c, pred in enumerate(cands):
            if pred == "none" or (pred == "lowzero" and not coprime10):
                ok_all.append(False)
                continue
            ladder, split, step_target, s = template(pred, W)
            n = min(cfg["chain_rows"], len(b))
            q = torch.arange(10).repeat(n)
            xr, Nr_, Ndr = x[:n].repeat_interleave(10), N[:n].repeat_interleave(10), Nd[:n].repeat_interleave(10, 0)
            reg, tgt = ladder(xr, Nr_, q)
            reg, tgt = reg % MOD, tgt % MOD
            gl = 1.0 if cfg["gates"] == "answers" else torch.sigmoid(model.gate[c])
            # the chain block: every combination, the ladder and the gated subtraction's law (csub)
            with torch.no_grad():
                K = len(CC)
                ixk = {k: CC[:, i].repeat_interleave(len(reg)) for i, k in enumerate(CHAIN)}
                ixk = model.ix(c, K * len(reg), ixk)
                v = (model.chain(reg.repeat(K), Ndr.repeat(K, 1), ixk) == tgt.repeat(K)).view(K, -1)
                g0, g1 = CC[:, CHAIN.index("g0")], CC[:, CHAIN.index("g1")]
                gc0 = torch.tensor([dm["g0"][i] == "commit" for i in range(len(dm["g0"]))])[g0]   # commit on flag 0
                gc1 = torch.tensor([dm["g1"][i] == "commit" for i in range(len(dm["g1"]))])[g1]   # commit on flag 1
                xn = x[:n] + N[:n]
                csub_ok = torch.ones(K, n, dtype=torch.bool)
                for R in (xn, x[:n]):                                     # csub(x + N) = x, csub(x) = x
                    commit = torch.where((R < N[:n])[None], gc1[:, None], gc0[:, None])   # the flag: R - N borrows
                    csub_ok &= torch.where(commit, ((R - N[:n]) % MOD)[None], R[None]) == x[:n][None]
                v = v & csub_ok.repeat_interleave(10, 1)
                arg_ok = bool(v[int(((CC == torch.tensor([int(model.sw[c][k].argmax()) for k in CHAIN])).all(1)
                                     ).nonzero()[0])].all())
            loss = loss + gl * mixture_nll(lw_of(model.sw[c], CHAIN, CC), v).mean()
            streak[c] = streak[c] + 1 if arg_ok else 0
            if not io_on[c] and streak[c] >= cfg["win"]:
                io_on[c] = True
                events[f"io_{pred}"] = step
            if not io_on[c]:
                ok_all.append(False)
                continue
            # the step block: the step law (the chain at its argmax)
            with torch.no_grad():
                K = len(SC)
                R0, d = split(reg)
                st = step_target(xr, Nr_, q) % MOD
                ixk = model.ix(c, K * len(reg), {k: SC[:, i].repeat_interleave(len(reg)) for i, k in enumerate(STEP)})
                v = (model.step(R0.repeat(K), d.repeat(K), Ndr.repeat(K, 1), ixk) == st.repeat(K)).view(K, -1)
                step_ok = bool(((model.step(R0, d, Ndr, model.ix(c, len(reg))) == st)).all())
            loss = loss + gl * mixture_nll(lw_of(model.sw[c], STEP, SC), v).mean()
            # the read block: the whole-reduction laws with alignment s (and the answers, the round at its argmax)
            with torch.no_grad():
                m = min(cfg["io_rows"], len(b))
                laws = [(a, {"x": x, "T": T, "y": G.value(yd)}[a][:m]) for a in ident]
                Q = x.roll(1)[:m]
                cases = [(G.value(G.digits(av, W)) * 10 ** s, av) for _, av in laws]
                cases += [(N[:m] * 10 ** s, torch.zeros_like(N[:m])), (x[:m] * 10 ** s + Q * N[:m], x[:m])]
                K = len(RC)
                v = torch.ones(K, m, dtype=torch.bool)
                law_arg = True
                for Vv, r in cases:
                    Vd = G.digits(Vv, 2 * W)
                    ixk = model.ix(c, K * m, {k: RC[:, i].repeat_interleave(m) for i, k in enumerate(READ)})
                    v &= (G.value(model.divide(Vd.repeat(K, 1), Nd[:m].repeat(K, 1), ixk)) == r.repeat(K)).view(K, m)
                    law_arg &= bool((G.value(model.divide(Vd, Nd[:m], model.ix(c, m))) == r).all())
                rows = torch.nonzero(lab).flatten()
                va = None
                if rows.numel():
                    mr = rows.numel()
                    ru = [u.repeat(K) for u in model.round_idx(mr)]
                    ixk = model.ix(c, K * mr, {k: RC[:, i].repeat_interleave(mr) for i, k in enumerate(READ)})
                    out = model.run(xd[rows].repeat(K, 1), Nd[rows].repeat(K, 1), T[rows].repeat(K), ixk, *ru)
                    va = (out == yd[rows].repeat(K, 1)).all(1).view(K, mr)
            lwr = lw_of(model.sw[c], READ, RC)
            loss = loss + gl * mixture_nll(lwr, v).mean()
            if va is not None:
                loss = loss + gl * mixture_nll(lwr, va).mean()
            ok_all.append(step_ok and law_arg)
        # the round block and the selector: answers only
        rows = torch.nonzero(lab).flatten()
        sel_ok = False
        if any(io_on) and rows.numel():
            with torch.no_grad():
                mr, K = rows.numel(), len(UC)
                ver = []
                for c in range(len(cands)):
                    if not io_on[c]:
                        ver.append(torch.zeros(K, mr, dtype=torch.bool))
                        continue
                    ru = [UC[:, i].repeat_interleave(mr) for i in range(3)]
                    out = model.run(xd[rows].repeat(K, 1), Nd[rows].repeat(K, 1), T[rows].repeat(K),
                                    model.ix(c, K * mr), *ru)
                    ver.append((out == yd[rows].repeat(K, 1)).all(1).view(K, mr))
                ver = torch.cat(ver)                                      # [cands x K, mr]
                cstar = int(model.sel.argmax())
                ustar = torch.tensor([int(model.rnd[k].argmax()) for k in ROUND])
                sel_ok = bool(ver[cstar * K + int(((UC == ustar).all(1)).nonzero()[0])].all())
            lw = (model.sel.log_softmax(-1)[:, None] + lw_of(model.rnd, ROUND, UC)[None]).reshape(-1)
            loss = loss + mixture_nll(lw, ver).mean()
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
        cstar = int(model.sel.argmax())
        good = io_on[cstar] and ok_all[cstar] and sel_ok
        done_streak = done_streak + 1 if good else 0
        if good and "all_ok" not in events:
            events["all_ok"] = step
        if verbose and (step % cfg["log"] == 0 or step == 1):
            sp = model.sel.softmax(-1).tolist()
            print(f"step={step} loss={float(loss):.4f} io_on={io_on} sel={[round(p, 3) for p in sp]} "
                  f"round={[model.rm[k][int(model.rnd[k].argmax())] for k in ROUND]} ok={ok_all} sel_ok={sel_ok} "
                  f"facts={len(miner.facts())}", flush=True)
        if done_streak >= cfg["done"]:
            break
    # certification: the selected candidate's argmax machine with the argmax round options, T up to 8, every width
    cstar = int(model.sel.argmax())
    with torch.no_grad():
        ru = model.round_idx(cert.n)
        out = model.run(cert.xd, cert.Nd, cert.Tt, model.ix(cstar, cert.n), *ru)
        right = (out == cert.yd).all(1)
    per_T = {int(t): float(right[cert.Tt == t].float().mean()) for t in (1, 2, 4, 8)}
    per_w = {}
    for w in cert_widths:
        msk = torch.from_numpy((cert.N >= 10 ** (w - 1)) & (cert.N < 10 ** w))
        per_w[w] = float(right[msk].float().mean()) if bool(msk.any()) else None
    res = {
        "cfg": {k: v for k, v in cfg.items()}, "steps": step, "seconds": round(time.time() - t0, 1),
        "events": events, "selected": cands[cstar], "selector": [round(p, 4) for p in model.sel.softmax(-1).tolist()],
        "round": {k: model.rm[k][int(model.rnd[k].argmax())] for k in ROUND},
        "wirings": {p: {k: str(v) for k, v in model.names(c).items()} for c, p in enumerate(cands)},
        "certified": bool(right.all()), "cert_per_T": per_T, "cert_per_width": per_w, "mined": miner.report(),
        "gates": [round(float(torch.sigmoid(g)), 4) for g in model.gate] if cfg["gates"] == "loss" else None,
    }
    if verbose:
        print("RESULT " + json.dumps(res), flush=True)
    if cfg["out"]:
        Path(cfg["out"]).write_text(json.dumps(res, indent=1))
    return res


def parse(argv) -> dict:
    cfg = dict(CFG)
    for a in argv:
        k, v = a.split("=", 1)
        cfg[k] = type(CFG[k])(v) if not isinstance(CFG[k], tuple) else tuple(int(t) for t in v.split(","))
    return cfg


if __name__ == "__main__":
    run(parse(sys.argv[1:]))
