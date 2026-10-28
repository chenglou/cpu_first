"""The quotient-digit chicken-and-egg in miniature: microlab's long-division datapath with every part a small table,
so every training example can be enumerated and every gradient computed exactly.

A size is (B, w, k): base B, moduli N with w base-B digits (top digit nonzero; odd only when `odd`), and dividends
v < N * B^k, so the quotient has k real digits. The datapath (microlab/ext_matched.LongDiv, made tabular):
- the running remainder R keeps w+1 digit distributions; it starts at v's top w digits (`start="prefix"`, which is
  v // B^k < N) or at 0 (`start="full"`, which then runs w extra steps whose quotient digit is always 0);
- each step forms Rp = B R + v_t (the top digit of R drops out) and tries every slot s = 0..B-1 at once. A trial is a
  ripple of a learned digit cell from the lowest place up, w+1 places, ending in a carry;
- a choice weights the slots, and R becomes the weighted mixture of the trials' digit distributions;
- the loss is the cross-entropy of R's low w digits against the answer v mod N.
Cells (`cell`):
- "slot"    one table (r, n, s, c) -> (digit, carry), carries 0..B-1: what slot s subtracts is learned, as with
            microlab's learned slot embedding. Exact: r - s*n - c.
- "repeat"  one table (r, n, c) -> (digit, carry), carries 0..1, the subtractor of N; slot s runs it s times. What a
            slot means is fixed by construction (symmetry breaking by construction).
- "mult"    the same one subtractor table, shared by every slot: slot s subtracts the exact digits of s*N (given), in
            one pass. Slot meaning is fixed by construction without repeated passes.
The table is applied multilinearly to digit distributions (expected output over independent inputs), as microlab's
`soft_cell` diagnostic applies the truth table.
Choices (`choice`):
- "borrow"  microlab longdiv: the largest slot whose final carry is 0 (P(slot s) = valid_s * prod_{s'>s}(1 - valid_s')).
            With "repeat": keep subtracting while there is no borrow (restoring division).
- "est"     a separate table on the top `kr` digits of Rp and the top `kn` digits of N (microlab longdiv_est); the
            borrow is unused. Applied multilinearly too.
- "estc"    the estimate checked by the borrows next to it (microlab longdiv_estc).
- "given"   the true quotient digits (microlab modq): only the cell learns.
`exact_cell=True` replaces the cell by its truth table (microlab soft_cell): only the choice learns.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def digits(v, B, width):
    v = np.asarray(v, dtype=np.int64).copy()
    out = np.empty((len(v), width), dtype=np.int64)
    for i in range(width):
        out[:, i] = v % B
        v //= B
    return out


class Data:
    """Every (N, v) of a size, or a uniform sample of `sample` of them."""

    def __init__(self, B, w, k, odd=False, start="prefix", moduli=None, sample=0, seed=0, vmax="k", draw=0,
                 qmax=None):
        """draw > 0: draw that many examples at random (N uniform over the moduli, v uniform below N * B^k) instead
        of enumerating; qmax: only quotients whose every digit is <= qmax (a curriculum over the digit range)."""
        self.B, self.w, self.k, self.start = B, w, k, start
        if moduli is None:
            moduli = [N for N in range(max(2, B ** (w - 1)), B ** w) if not odd or N % 2]
        self.moduli = list(moduli)
        if draw:
            rng = np.random.default_rng(seed)
            N = rng.choice(np.array(self.moduli, dtype=np.int64), draw)
            if vmax == "NN":                                                  # microlab: v < N^2
                v = (rng.random(draw) * (N.astype(np.float64) ** 2)).astype(np.int64)
                v = np.minimum(v, N * N - 1)
            elif isinstance(vmax, int):                                       # v < N * B^vmax (a stage)
                v = (rng.random(draw) * (N * float(B) ** vmax)).astype(np.int64)
            elif qmax is None:
                v = (rng.random(draw) * (N * float(B) ** k)).astype(np.int64)
            else:
                qd = rng.integers(0, qmax + 1, (draw, k))
                q = (qd * B ** np.arange(k)).sum(1)
                v = q * N + (rng.random(draw) * N).astype(np.int64)
            v = np.minimum(v, N * B ** k - 1)
        else:
            Ns, vs = [], []
            for N in self.moduli:
                top = N * B ** k if vmax == "k" else N * N
                vs.append(np.arange(top, dtype=np.int64))
                Ns.append(np.full(top, N, dtype=np.int64))
            N, v = np.concatenate(Ns), np.concatenate(vs)
            if qmax is not None:
                qd = digits(v // N, B, k)
                keep = (qd <= qmax).all(1)
                N, v = N[keep], v[keep]
        if sample and sample < len(N):
            idx = np.random.default_rng(seed).choice(len(N), sample, replace=False)
            N, v = N[idx], v[idx]
        self.N, self.v = N, v
        self.y, self.q = v % N, v // N
        self.X = len(N)
        L = w + k
        self.Nd = torch.from_numpy(digits(N, B, w))
        self.Md = torch.stack([torch.from_numpy(digits(s * N, B, w + 1)) for s in range(B)], 1)   # [X, B, w+1]
        self.vd = torch.from_numpy(digits(v, B, L))
        self.yd = torch.from_numpy(digits(self.y, B, w))
        self.qd = torch.from_numpy(digits(self.q, B, L))                      # quotient digits per dividend place
        if start == "prefix":
            self.R0 = torch.from_numpy(digits(v // B ** k, B, w + 1))
            self.steps = list(reversed(range(k)))                             # dividend places, MSB first
        else:
            self.R0 = torch.zeros(self.X, w + 1, dtype=torch.long)
            self.steps = list(reversed(range(L)))

    def subset(self, mask):
        out = object.__new__(Data)
        out.__dict__.update(self.__dict__)
        m = np.asarray(mask)
        for a in ("N", "v", "y", "q"):
            setattr(out, a, getattr(self, a)[m])
        mt = torch.from_numpy(m)
        for a in ("Nd", "vd", "yd", "qd", "R0", "Md"):
            setattr(out, a, getattr(self, a)[mt])
        out.X = int(m.sum()) if m.dtype == bool else len(m)
        return out


def truth_slot(B):
    """One-hot (digit, carry) of r - s*n - c for every (r, n, s, c): [B, B, B, B, B] and [B, B, B, B, B]."""
    r, n, s, c = np.meshgrid(*[np.arange(B)] * 4, indexing="ij")
    x = r - s * n - c
    borrow = (-x + B - 1) // B * (x < 0)
    eye = np.eye(B)
    return torch.from_numpy(eye[x % B]), torch.from_numpy(eye[borrow])


def truth_repeat(B):
    """One-hot (digit, borrow) of r - n - c for every (r, n, c), c in {0, 1}: [B, B, 2, B] and [B, B, 2, 2]."""
    r, n, c = np.meshgrid(np.arange(B), np.arange(B), np.arange(2), indexing="ij")
    x = r - n - c
    return torch.from_numpy(np.eye(B)[x % B]), torch.from_numpy(np.eye(2)[(x < 0).astype(int)])


@dataclass
class Config:
    B: int = 2
    w: int = 3
    k: int = 2
    cell: str = "slot"            # slot | repeat
    choice: str = "est"           # borrow | est | estc | given
    kr: int = 0                   # estimator window: top kr digits of Rp (0 = all w+1)
    kn: int = 0                   # and top kn digits of N (0 = all w)
    exact_cell: bool = False
    init: float = 0.1             # std of the initial logits
    odd: bool = False
    start: str = "prefix"
    loss: str = "ce"              # ce | clock
    score_top: bool = False       # also score the register's top digit, which is 0 for every reduced remainder


class Core:
    """The tabular datapath. Parameters are plain tensors: cell digit/carry logits `a`, `b`; estimator logits `g`."""

    def __init__(self, cfg: Config, seed=0):
        self.cfg = c = cfg
        B, w = c.B, c.w
        self.S = B
        self.kr = c.kr or (w + 1)
        self.kn = c.kn or w
        gen = torch.Generator().manual_seed(seed)
        rnd = lambda *shape: (torch.randn(*shape, generator=gen) * c.init)
        if c.cell == "slot":
            self.C = B
            self.a, self.b = rnd(B, B, B, self.C, B), rnd(B, B, B, self.C, self.C)
            self.TD, self.TC = truth_slot(B)
        else:
            self.C = 2
            self.a, self.b = rnd(B, B, 2, B), rnd(B, B, 2, 2)
            self.TD, self.TC = truth_repeat(B)
        self.g = rnd(B ** self.kr * B ** self.kn, self.S)
        self.params = {"a": self.a, "b": self.b, "g": self.g}
        for p in self.params.values():
            p.requires_grad_(True)

    # ----- tables -----
    def tables(self):
        if self.cfg.exact_cell:
            return self.TD, self.TC
        return self.a.softmax(-1), self.b.softmax(-1)

    def set_truth(self, beta=8.0, which=("a", "b")):
        """Logits beta * (one-hot truth) for the cell (`a`, `b`)."""
        with torch.no_grad():
            if "a" in which:
                self.a.copy_(beta * self.TD)
            if "b" in which:
                self.b.copy_(beta * self.TC)

    # ----- the datapath -----
    def ripple_slot(self, Rp, Nd, TD, TC):
        """Every slot's trial: Rp [X, w+1, B] distributions, Nd [X, w] ints. Returns D [X, S, w+1, B], carry [X, S, C]."""
        X, P, B = Rp.shape
        n = F.pad(Nd, (0, P - Nd.shape[1]))                                     # P places, the top ones 0
        c = torch.zeros(X, self.S, self.C)
        c[..., 0] = 1
        D = []
        for j in range(P):                                                     # one table slice per digit of N, so
            d = cn = 0                                                         # memory stays O(X * S * B)
            for v in range(B):
                mv = (n[:, j] == v)
                if not bool(mv.any()):
                    continue
                mv = mv.double()[:, None, None]
                d = d + mv * torch.einsum("xr,xsc,rsco->xso", Rp[:, j], c, TD[:, v])
                cn = cn + mv * torch.einsum("xr,xsc,rsco->xso", Rp[:, j], c, TC[:, v])
            c = cn
            D.append(d)
        return torch.stack(D, 2), c

    def ripple_mult(self, Rp, Md, TD, TC):
        """Every slot's trial with the shared subtractor: slot s subtracts the digits Md[:, s] of s*N."""
        X, P, B = Rp.shape
        c = torch.zeros(X, self.S, 2)
        c[..., 0] = 1
        D = []
        for j in range(P):
            m = Md[:, :, j]                                                     # [X, S]
            tdn, tcn = TD[:, m], TC[:, m]                                       # [B(r), X, S, 2, out]
            D.append(torch.einsum("xr,rxsco,xsc->xso", Rp[:, j], tdn, c))
            c = torch.einsum("xr,rxsco,xsc->xso", Rp[:, j], tcn, c)
        return torch.stack(D, 2), c

    def ripple_once(self, Rp, Nd, TD, TC):
        """One subtraction of N with the repeat cell: Rp [X, w+1, B] -> D [X, w+1, B], P(final borrow) [X]."""
        X, P, B = Rp.shape
        n = F.pad(Nd, (0, P - Nd.shape[1]))
        c = torch.zeros(X, 2)
        c[:, 0] = 1
        D = []
        for j in range(P):
            tdn, tcn = TD[:, n[:, j]], TC[:, n[:, j]]                         # [B(r), X, 2, out]
            D.append(torch.einsum("xr,rxco,xc->xo", Rp[:, j], tdn, c))
            c = torch.einsum("xr,rxco,xc->xo", Rp[:, j], tcn, c)
        return torch.stack(D, 1), c[:, 1]

    def estimate(self, Rp, Nd):
        B = self.cfg.B
        top = Rp[:, Rp.shape[1] - self.kr:]                                    # [X, kr, B]
        joint = top[:, 0]
        for i in range(1, self.kr):                                            # joint over windows, digit 0 fastest
            joint = (top[:, i, :, None] * joint[:, None, :]).flatten(1)
        nw = Nd[:, Nd.shape[1] - self.kn:]
        nidx = (nw * (B ** torch.arange(self.kn))).sum(1)
        G = self.g.softmax(-1).view(B ** self.kr, B ** self.kn, self.S)[:, nidx]   # [B^kr, X, S]
        return torch.einsum("xr,rxs->xs", joint, G)

    def step(self, R, vt, Nd, qt, TD, TC, choice=None):
        """One division step. Returns the new R, the choice P(slot) [X, S], the trials D and validity; `choice`
        overrides the configured choice (e.g. "given": take qt)."""
        cfg = self.cfg
        B = cfg.B
        choice = choice or cfg.choice
        Rp = torch.cat([F.one_hot(vt, B).double()[:, None], R[:, :cfg.w]], 1)
        if cfg.cell == "slot":
            D, c = self.ripple_slot(Rp, Nd, TD, TC)
            valid = c[..., 0]
        elif cfg.cell == "mult":
            D, c = self.ripple_mult(Rp, self.cur_Md, TD, TC)
            valid = c[..., 0]
        else:                                                                  # slot s = s subtractions
            Ds, nb, cur, alive = [Rp], [], Rp, torch.ones(Rp.shape[0])
            valid = [alive]
            for s in range(1, self.S):
                cur, pb = self.ripple_once(cur, Nd, TD, TC)
                Ds.append(cur)
                alive = alive * (1 - pb)
                valid.append(alive)
            D, valid = torch.stack(Ds, 1), torch.stack(valid, 1)
        self.last_Rp = Rp
        if choice == "given":
            wq = F.one_hot(qt, self.S).double()
        elif choice == "borrow":
            if cfg.cell in ("slot", "mult"):
                higher = torch.cat([torch.flip(torch.cumprod(torch.flip(1 - valid, [1]), 1), [1])[:, 1:],
                                    torch.ones(valid.shape[0], 1)], 1)
                wq = valid * higher
                wq = wq / wq.sum(1, keepdim=True).clamp_min(1e-12)
            else:                                                              # P(exactly s subtractions survive)
                nxt = torch.cat([valid[:, 1:], torch.zeros(valid.shape[0], 1)], 1)
                wq = valid - nxt
        else:
            pe = self.estimate(Rp, Nd)
            if choice == "est":
                wq = pe
            else:
                v1 = F.pad(valid[:, 1:], (0, 1))
                down, stay, up = pe * (1 - valid), pe * valid * (1 - v1), pe * valid * v1
                wq = stay + F.pad(down[:, 1:], (0, 1)) + F.pad(down[:, :1], (0, self.S - 1)) + F.pad(up[:, :-1], (1, 0))
        R = torch.einsum("xs,xspo->xpo", wq, D)
        return R, wq, D, valid

    def forward(self, data: Data, hard=False, path=None, keep_Rp=False):
        """Soft (or, with hard=True, argmax-choice and snapped-digit) run over all examples. Returns the final R
        [X, w+1, B] and the choices [X, steps, S]. `path` [X, steps] gives the slot of every step instead (then
        the choices returned are those slots, one-hot); keep_Rp also stores every step's Rp in self.Rps."""
        cfg = self.cfg
        TD, TC = self.tables()
        R = F.one_hot(data.R0, cfg.B).double()
        self.cur_Md = data.Md
        choices, self.Rps = [], []
        for i, t in enumerate(data.steps):
            if path is None:
                R, wq, D, valid = self.step(R, data.vd[:, t], data.Nd, data.qd[:, t], TD, TC)
            else:
                R, wq, D, valid = self.step(R, data.vd[:, t], data.Nd, path[:, i], TD, TC, choice="given")
            if keep_Rp:
                self.Rps.append(self.last_Rp)
            if hard:
                s = wq.argmax(1)
                R = F.one_hot(D[torch.arange(len(s)), s].argmax(-1), cfg.B).double()
            choices.append(wq)
        return R, torch.stack(choices, 1)

    def loglik(self, data: Data, R):
        """Per-example log-likelihood of the answer digits [X]."""
        P = R[:, :self.cfg.w].clamp_min(1e-12)
        return P.log().gather(-1, data.yd[..., None]).squeeze(-1).sum(1)

    def loss(self, data: Data, R=None):
        """Cross-entropy of the answer's w digits (microlab's loss); with score_top, also of the register's top
        digit, whose target 0 the answer implies (a remainder below N has w digits)."""
        if R is None:
            R, _ = self.forward(data)
        if self.cfg.score_top:
            y = F.pad(data.yd, (0, 1))
            return -R.clamp_min(1e-12).log().gather(-1, y[..., None]).mean()
        P = R[:, :self.cfg.w].clamp_min(1e-12)
        return -P.log().gather(-1, data.yd[..., None]).mean()

    # ----- measures (evaluated in chunks, so memory stays bounded) -----
    CHUNK = 1024

    def chunks(self, data: Data):
        for i in range(0, data.X, self.CHUNK):
            yield data if data.X <= self.CHUNK else data.subset(np.arange(i, min(i + self.CHUNK, data.X)))

    @torch.no_grad()
    def eval_loss(self, data: Data):
        return sum(self.loss(d).item() * d.X for d in self.chunks(data)) / data.X

    @torch.no_grad()
    def exact(self, data: Data, hard=False):
        n = 0
        for d in self.chunks(data):
            R, ch = self.forward(d, hard=hard)
            n += (R[:, :self.cfg.w].argmax(-1) == d.yd).all(1).sum().item()
        return n / data.X

    @torch.no_grad()
    def slot_map(self, data: Data):
        """For every slot, the quotient digit it most often stands for (over the steps where it is the argmax choice),
        and the share of steps whose argmax slot is the true digit, raw and through that map."""
        ss, qs = [], []
        for d in self.chunks(data):
            R, ch = self.forward(d)
            ss.append(ch.argmax(-1))                                           # [X, steps]
            qs.append(torch.stack([d.qd[:, t] for t in d.steps], 1))
        s, q = torch.cat(ss), torch.cat(qs)
        S = self.S
        counts = torch.zeros(S, S)
        counts.index_put_((s.flatten(), q.flatten()), torch.ones(s.numel()), accumulate=True)
        smap = counts.argmax(1)
        used = counts.sum(1) > 0
        raw = (s == q).double().mean().item()
        mapped = (smap[s] == q).double().mean().item()
        return [int(m) if u else -1 for m, u in zip(smap, used)], raw, mapped

    @torch.no_grad()
    def exact_given(self, data: Data):
        """Exact with the true quotient digits given: whether the cell subtracts right on the datapath, whatever
        it does on rows the datapath never visits and however it labels its carries."""
        n = 0
        for d in self.chunks(data):
            q = torch.stack([d.qd[:, t] for t in d.steps], 1)
            R, _ = self.forward(d, path=q)
            n += (R[:, :self.cfg.w].argmax(-1) == d.yd).all(1).sum().item()
        return n / data.X

    @torch.no_grad()
    def cell_right(self):
        """Share of the cell's table rows whose argmax digit and carry are right, slots read as themselves."""
        TD, TC = self.tables()
        d = (TD.argmax(-1) == self.TD.argmax(-1))
        c = (TC.argmax(-1) == self.TC.argmax(-1))
        return (d & c).double().mean().item()


def adam(params, lr):
    return torch.optim.Adam(params, lr=lr)


def train(model: Core, data: Data, steps=2000, lr=0.05, every=50, opt="adam", batch=0, seed=0, test=None,
          stop_at=1.0, trainable=("a", "b", "g"), verbose=False, clip=0.0, callback=None):
    """Full-batch (batch=0) or minibatch training. Returns the curve [(step, loss, exact)] and the first step at
    which exact >= stop_at (or None)."""
    params = [model.params[n] for n in trainable if not (model.cfg.exact_cell and n in "ab")]
    if model.cfg.choice in ("given", "borrow"):
        params = [p for p in params if p is not model.g]
    o = torch.optim.Adam(params, lr=lr) if opt == "adam" else torch.optim.SGD(params, lr=lr)
    rng = np.random.default_rng(seed)
    test = test or data
    curve, hit = [], None
    for it in range(steps + 1):
        if it % every == 0 or it == steps:
            ex = model.exact(test)
            L = model.eval_loss(test)
            curve.append((it, L, ex))
            if verbose:
                print(f"{it:6d} loss {L:.4f} exact {ex:.3f}", flush=True)
            if callback:
                callback(it, model)
            if ex >= stop_at and hit is None:
                hit = it
                break
        if it == steps:
            break
        d = data if not batch or batch >= data.X else data.subset(rng.choice(data.X, batch, replace=False))
        L = model.loss(d)
        o.zero_grad()
        L.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(params, clip)
        o.step()
    return curve, hit
