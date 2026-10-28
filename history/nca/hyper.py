"""Wiring from a generator: every cell connected to every other, each connection's weight produced by one small
learned network g (an indirect encoding, as in HyperNEAT's CPPNs, Stanley, D'Ambrosio and Gauci 2009, and in
hypernetworks, Ha, Dai and Le 2016).

A tick, for every cell i of the grid at once:
    heads    for each of H heads, g scores every source cell j of the grid and of its OUT ring from a description of
             the pair: the row and column offset (one-hot within 3, and as plain numbers), the two cells' kinds, and
             how far each is from the left and right edges. The scores become weights: "softmax" over j (heads
             compete for sources, as attention does) or "sigmoid" (every connection on its own, no competition).
             The head's input to cell i is the weighted sum of the sources' kinds and states
    rule     the shared per-cell rule, an MLP on [cell i's kind, head 1, ..., head H] -> one logit per channel value
    hard     evaluated hard, a softmax head reads its single best source (argmax) and a sigmoid head the sources
             scoring above 0, and every state is snapped
The weights of all L x L connections are a function of a few hundred numbers in g; the symmetry and instancing live in
g, and locality is something g can learn rather than a hard constraint. `mask` can still impose it: "noself" removes
the cell itself from its sources, "causal" also every cell of the rows below.

`compile_ports` writes the bridge's machine (BRIDGE.md §1, the "ports" reach of grid.compiled_clauses, digit and
borrow cells) into g and the rule: per kind, head A, B and C read
    SHIFT  the cell up-left                          (the register moved up a place)
    SUB    the cell above, N's digit in my column, the cell to my left
    SEL    the cell two rows up, the cell above, the top place of the row above
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nca import grid as G
from nca import model as M
from nca.grid import OUT, PROMPT, SEL, SHIFT, SUB

CLIP = 3                                   # offsets one-hot within +-CLIP
EDGE = 2                                   # edge distances one-hot within -1 .. EDGE


def feature_offsets(nk: int):
    """Where each block of phi starts (the order pair_features builds them in)."""
    sizes = [("dr", 2 * CLIP + 1), ("dc", 2 * CLIP + 1), ("raw", 4), ("ki", nk), ("kj", nk), ("right_i", EDGE + 2),
             ("left_i", EDGE + 2), ("right_j", EDGE + 2), ("left_j", EDGE + 2)]
    off, o = {}, 0
    for n, sz in sizes:
        off[n] = o
        o += sz
    return off, o


def pair_features(lay: G.Layout, nk: int, kinds_pad: np.ndarray):
    """phi(i, j) for every inner cell i (R C of them) and every cell j of the padded grid ((R + 2)(C + 2)), as a
    float tensor [R C, (R + 2)(C + 2), F], with a dict of feature offsets (for the compile)."""
    R, C = lay.R, lay.C
    ri, ci = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    ri, ci = ri.ravel() + 1, ci.ravel() + 1                        # padded coordinates of the receivers
    rj, cj = np.meshgrid(np.arange(R + 2), np.arange(C + 2), indexing="ij")
    rj, cj = rj.ravel(), cj.ravel()
    dr = rj[None, :] - ri[:, None]
    dc = cj[None, :] - ci[:, None]
    ki = kinds_pad[ri, ci]
    kj = kinds_pad[rj, cj]
    right_i, left_i = C - ci, ci - 1                                # inner places from the edges (top place: 0)
    right_j, left_j = C - cj, cj - 1
    feats, off = [], {}

    def add(name, block):
        off[name] = sum(b.shape[-1] for b in feats)
        feats.append(block.astype(np.float32))

    oh = lambda x, lo, hi: np.eye(hi - lo + 1)[np.clip(x, lo, hi) - lo]
    add("dr", oh(dr, -CLIP, CLIP))
    add("dc", oh(dc, -CLIP, CLIP))
    add("raw", np.stack([dr / 8, dc / 8, np.abs(dr) / 8, np.abs(dc) / 8], -1))
    add("ki", np.broadcast_to(np.eye(nk)[ki][:, None, :], dr.shape + (nk,)))
    add("kj", np.broadcast_to(np.eye(nk)[kj][None, :, :], dr.shape + (nk,)))
    add("right_i", np.broadcast_to(oh(right_i, -1, EDGE)[:, None, :], dr.shape + (EDGE + 2,)))
    add("left_i", np.broadcast_to(oh(left_i, -1, EDGE)[:, None, :], dr.shape + (EDGE + 2,)))
    add("right_j", np.broadcast_to(oh(right_j, -1, EDGE)[None, :, :], dr.shape + (EDGE + 2,)))
    add("left_j", np.broadcast_to(oh(left_j, -1, EDGE)[None, :, :], dr.shape + (EDGE + 2,)))
    phi = np.concatenate([np.broadcast_to(f, dr.shape + (f.shape[-1],)) for f in feats], -1)
    mask_self = dr == 0
    mask_self &= dc == 0
    below = dr > 0
    return torch.from_numpy(np.ascontiguousarray(phi)), off, torch.from_numpy(mask_self), torch.from_numpy(below)


class HyperNCA(M.NCA):
    """The NCA of model.py with its 3 x 3 rule replaced by generated all-to-all heads and a rule on the heads."""

    def __init__(self, fmt: G.Fmt, cfg: M.Cfg, seed: int = 0, dtype=torch.float32):
        nn.Module.__init__(self)
        self.fmt, self.cfg, self.dtype = fmt, cfg, dtype
        g = torch.Generator().manual_seed(seed)
        self.nk = NK = G.NK_SQ if cfg.kindset == "sq" else G.NK
        self._kinds, self._phi = {}, {}
        self.nextra = 0
        Hh, Hg, Hr = cfg.heads, cfg.ghidden, cfg.hidden
        self.nf = nf = feature_offsets(NK)[1]
        r = lambda *s, fan: nn.Parameter(torch.randn(*s, generator=g, dtype=dtype) * math.sqrt(2 / fan))
        self.G1, self.g1 = r(Hg, nf, fan=nf), nn.Parameter(torch.zeros(Hg, dtype=dtype))
        self.G2, self.g2 = r(Hh, Hg, fan=Hg), nn.Parameter(torch.full((Hh,), cfg.gbias, dtype=dtype))
        nin = NK + Hh * (NK + fmt.D)
        self.W1, self.b1 = r(Hr, nin, fan=nin), nn.Parameter(torch.zeros(Hr, dtype=dtype))
        self.W2 = nn.Parameter(torch.randn(fmt.D, Hr, generator=g, dtype=dtype) * math.sqrt(1 / Hr))
        self.b2 = nn.Parameter(torch.zeros(fmt.D, dtype=dtype))

    # ----- the generated wiring -----
    def phi(self, lay):
        key = lay.key()
        if key not in self._phi:
            kd = self.kinds(lay)[0].argmax(0).numpy()                 # [R + 2, C + 2] kinds with the OUT ring
            self._phi[key] = pair_features(lay, self.nk, kd)
        return self._phi[key]

    def scores(self, lay):
        """g's score of every (receiver, source) pair per head: [H, R C, (R + 2)(C + 2)], masked to -inf."""
        phi, _, mself, below = self.phi(lay)
        h = F.relu(torch.einsum("ijf,gf->ijg", phi.to(self.dtype), self.G1) + self.g1)
        s = torch.einsum("ijg,hg->hij", h, self.G2) + self.g2[:, None, None]
        if self.cfg.mask in ("noself", "causal"):
            s = s.masked_fill(mself[None], -math.inf)
        if self.cfg.mask == "causal":
            s = s.masked_fill(below[None], -math.inf)
        return s

    def weights(self, lay, hard=False):
        s = self.scores(lay)
        if self.cfg.gnorm == "softmax":
            if hard:
                return F.one_hot(s.argmax(-1), s.shape[-1]).to(self.dtype)
            return s.softmax(-1)
        if hard:
            return (s > 0).to(self.dtype)
        return torch.sigmoid(s)                                          # -inf (masked) -> weight 0

    # ----- one tick -----
    def padded(self, S, lay):
        """[X, NK + D, (R + 2)(C + 2)]: every cell's kind and state, the OUT ring blank."""
        X = S.shape[0]
        P = self.blank(X, lay.R + 2, lay.C + 2)
        P[:, :, 1:-1, 1:-1] = S
        inp = torch.cat([self.kinds(lay).expand(X, -1, -1, -1), P], 1)
        return inp.reshape(X, inp.shape[1], -1)

    def logits(self, S, lay, hard=False, W=None):
        X = S.shape[0]
        W = self.weights(lay, hard) if W is None else W                  # [H, Li, Lj]
        heads = torch.einsum("hij,xdj->xihd", W, self.padded(S, lay))     # [X, Li, H, NK + D]
        kind_i = self.kinds(lay)[0, :, 1:-1, 1:-1].reshape(self.nk, -1).t()   # [Li, NK]
        z = self.rule_logits(kind_i.expand(X, -1, -1), heads.reshape(X, heads.shape[1], -1))
        return z.permute(0, 2, 1).reshape(X, self.fmt.D, lay.R, lay.C)

    def query_sub(self, a, m, cin):
        """The rule at a SUB cell whose head A reads a cell holding digit a, head B a prompt cell holding N's digit m,
        head C a SUB cell holding borrow cin (distributions [X, *]): its digit and borrow distributions. The heads'
        roles are the compiled machine's; the lessons that query this hold them to it."""
        f, NK = self.fmt, self.nk
        X = a.shape[0]
        blank = self.blank(X, 1, 1)[:, :, 0, 0]
        kind = lambda k: F.one_hot(torch.full((X,), k), NK).to(self.dtype)
        A, Bh, Ch = blank.clone(), blank.clone(), blank.clone()
        A[:, f.sl("d")] = a
        Bh[:, f.sl("n")] = m
        Ch[:, f.sl("b")] = cin
        heads = torch.cat([kind(SHIFT), A, kind(PROMPT), Bh, kind(SUB), Ch], -1)
        z = self.rule_logits(kind(SUB), heads)
        return z[:, f.sl("d")].softmax(-1), z[:, f.sl("b")].softmax(-1)

    def rule_logits(self, kind_i, heads):
        """The shared rule: kind one-hot [..., NK] and the heads' inputs [..., H (NK + D)] -> logits [..., D]."""
        h = F.relu(torch.cat([kind_i, heads], -1) @ self.W1.t() + self.b1)
        return h @ self.W2.t() + self.b2

    def run(self, lay, S0, ticks, mode="soft", record=False, W=None):
        """As NCA.run, with g's weights computed once (they do not depend on the state), or given."""
        W = self.weights(lay, hard=mode == "hard") if W is None else W
        clamp = self.cfg.clamp
        pm = torch.from_numpy(lay.prompt)[None, None]
        S, P, hist = S0, S0, []
        for _ in range(ticks):
            Z = self.logits(S, lay, W=W)
            P = self.normalize(Z, "soft")
            S = self.normalize(Z, mode) if mode != "soft" else P
            if clamp:
                S = torch.where(pm, S0, S)
                P = torch.where(pm, S0, P)
            if record:
                hist.append(S)
        return (S, P, hist) if record else (S, P)


# ----- the compiled point: the bridge's machine -----

# per (head, kind): the source's description, as conditions on g's one-hot features
PORTS = {
    (0, SHIFT): [("dr", -1), ("dc", -1)],
    (0, SUB): [("dr", -1), ("dc", 0)],
    (1, SUB): [("kj", PROMPT), ("dc", 0)],
    (2, SUB): [("dr", 0), ("dc", -1)],
    (0, SEL): [("dr", -2), ("dc", 0)],
    (1, SEL): [("dr", -1), ("dc", 0)],
    (2, SEL): [("dr", -1), ("right_j", 0)],
}


def port_source(lay, h, kd, r, c):
    """The padded index of the compiled source of head h for an inner cell (r, c) of kind kd (None: unused)."""
    C2 = lay.C + 2
    pr, pc = r + 1, c + 1
    if (h, kd) not in PORTS:
        return None
    conds = dict(PORTS[(h, kd)])
    if "kj" in conds:                                                    # N's digit: the prompt row, same column
        return 1 * C2 + pc
    if "right_j" in conds:                                               # the top place of the row above
        return (pr - 1) * C2 + lay.C
    return (pr + conds["dr"]) * C2 + pc + conds["dc"]


def feature_index(off, name, value):
    if name in ("dr", "dc"):
        return off[name] + value + CLIP
    if name in ("right_i", "left_i", "right_j", "left_j"):
        return off[name] + value + 1
    return off[name] + value                                             # kinds


def compile_ports(net: HyperNCA, B: int, beta: float = 20.0, beta_o: float = 20.0):
    """g and the rule as the bridge's machine (digit-and-borrow cells, the dnb format): g's hidden units are AND
    detectors of (receiver kind, source description), scoring beta on their head; the rule's hidden units are AND
    detectors of (kind, head values), voting beta_o for the values they set. Unused heads score every source 0."""
    f = net.fmt
    NK, D, Hh = net.nk, f.D, net.cfg.heads
    off, _ = feature_offsets(NK)
    G1 = torch.zeros_like(net.G1)
    g1 = torch.full_like(net.g1, -1.0)
    G2 = torch.zeros_like(net.G2)
    for u, ((h, kd), conds) in enumerate(PORTS.items()):
        idx = [feature_index(off, "ki", kd)] + [feature_index(off, n, v) for n, v in conds]
        G1[u, idx] = 1.0
        g1[u] = -(len(idx) - 1)
        G2[h, u] = beta
    # the rule: input = [kind_i (NK), head 0 (NK + D), head 1, head 2]
    hv = lambda h, ch, v: NK + h * (NK + D) + NK + f.off[ch] + v
    clauses = []
    for j in range(B):
        clauses.append((SHIFT, [hv(0, "d", j)], {"d": j}))
    for a in range(B):
        for m in range(B):
            for c in range(2):
                x = a - m - c
                clauses.append((SUB, [hv(0, "d", a), hv(1, "n", m), hv(2, "b", c)], {"d": x % B, "b": int(x < 0)}))
    for j in range(B):
        clauses.append((SEL, [hv(2, "b", 1), hv(0, "d", j)], {"d": j}))
        clauses.append((SEL, [hv(2, "b", 0), hv(1, "d", j)], {"d": j}))
    for kd in (SHIFT, SUB, SEL):                                         # channels a kind does not set are 0
        set_ = {ch for k_, _, o in clauses if k_ == kd for ch in o}
        rest = {ch: 0 for ch, _ in f.chans if ch not in set_}
        if rest:
            clauses.append((kd, [], rest))
    assert net.cfg.hidden >= len(clauses), f"the rule needs {len(clauses)} hidden units"
    W1 = torch.zeros_like(net.W1)
    b1 = torch.full_like(net.b1, -1.0)
    W2 = torch.zeros_like(net.W2)
    for u, (kd, conds, outs) in enumerate(clauses):
        W1[u, kd] = 1.0
        W1[u, conds] = 1.0
        b1[u] = -len(conds)
        for ch, v in outs.items():
            W2[f.off[ch] + v, u] = beta_o
    with torch.no_grad():
        net.G1.zero_(), net.g1.fill_(-1.0), net.G2.zero_()                # units beyond the ports never fire
        net.G1[:len(PORTS)].copy_(G1[:len(PORTS)]), net.g1[:len(PORTS)].copy_(g1[:len(PORTS)])
        net.G2.copy_(G2)
        net.g2.fill_(0.0 if net.cfg.gnorm == "softmax" else -beta / 2)   # sigmoid: on at a match, off elsewhere
        net.W1.copy_(W1), net.b1.copy_(b1), net.W2.copy_(W2), net.b2.zero_()
    return len(PORTS), len(clauses)


def n_rule_clauses(B):
    return B + 2 * B * B + 2 * B + 3


def compiled_hyper(B, dtype=torch.float64, mask="all", gnorm="softmax"):
    fmt = G.compiled_fmt(B, minimal=True)
    cfg = M.Cfg(rule="hyper", heads=3, ghidden=len(PORTS), hidden=n_rule_clauses(B), mask=mask, gnorm=gnorm)
    net = HyperNCA(fmt, cfg, dtype=dtype)
    compile_ports(net, B)
    return net


# ----- probes -----

@torch.no_grad()
def wiring_probe(net: HyperNCA, lay: G.Layout):
    """What the generated wiring settled on, on the hard path (each head's best source per cell): per (head, kind)
    of the compiled machine, the share of cells reading the compiled source (any OUT cell counts for an OUT
    source); the most common source description (row offset, column offset, source kind) and its share; and the
    bypass: the share of cells whose head reads a prompt cell where the compiled source is not one."""
    if net.cfg.gnorm != "softmax":
        W = net.weights(lay, hard=True)
        return {"mean_sources": round(float(W.sum(-1).mean()), 2)}
    src = net.weights(lay, hard=True).argmax(-1).numpy()                   # [H, Li]
    R, C, C2 = lay.R, lay.C, lay.C + 2
    kpad = net.kinds(lay)[0].argmax(0).numpy().ravel()
    out, bypass_n, bypass_d = {}, 0, 0
    for h in range(net.cfg.heads):
        for kd in (SHIFT, SUB, SEL):
            cells = [(r, c) for r in range(R) for c in range(C) if lay.kind[r, c] == kd]
            if not cells:
                continue
            right, pats = 0, {}
            for r, c in cells:
                i = r * C + c
                j = src[h, i]
                jr, jc = divmod(int(j), C2)
                kind_j = G.KINDS[kpad[j]] if kpad[j] in (PROMPT, OUT) else "scratch"
                col = "top" if jc == C and kpad[j] != OUT else f"{jc - (c + 1):+d}"     # the top place of its row
                pat = (jr - (r + 1), col, kind_j)
                pats[pat] = pats.get(pat, 0) + 1
                cj = port_source(lay, h, kd, r, c)
                if cj is not None:
                    right += int(j == cj or (kpad[cj] == OUT and kpad[j] == OUT))
                    if kpad[cj] != PROMPT:
                        bypass_d += 1
                        bypass_n += int(kpad[j] == PROMPT)
            top = max(pats.items(), key=lambda kv: kv[1])
            key = f"{'ABC'[h]}.{G.KINDS[kd]}"
            out[key] = {"top": f"row {top[0][0]:+d}, col {top[0][1]}, {top[0][2]}", "top_share": round(top[1] / len(cells), 3)}
            if (h, kd) in PORTS:
                out[key]["right"] = round(right / len(cells), 3)
    out["bypass"] = round(bypass_n / max(1, bypass_d), 3)
    # up to which head does what: a cell is wired right if its heads, together, read every source the compiled
    # machine's heads read for its kind (any OUT cell standing for an OUT source)
    right_k = {}
    for kd in (SHIFT, SUB, SEL):
        cells = [(r, c) for r in range(R) for c in range(C) if lay.kind[r, c] == kd]
        ok = 0
        for r, c in cells:
            i = r * C + c
            got = [int(src[h, i]) for h in range(net.cfg.heads)]
            need = [port_source(lay, h, kd, r, c) for h in range(net.cfg.heads) if (h, kd) in PORTS]
            pool = list(got)
            good = True
            for cj in need:
                hit = next((x for x in pool if x == cj or (kpad[cj] == OUT and kpad[x] == OUT)), None)
                if hit is None:
                    good = False
                    break
                pool.remove(hit)
            ok += good
        right_k[G.KINDS[kd]] = round(ok / max(1, len(cells)), 3)
    out["cells_wired_right"] = right_k
    return out


def first_pull(net: HyperNCA, lay, Nd, vd, yd, ticks):
    """Where the answers' gradient first pushes the wiring: for every scratch cell and used head, the source whose
    score the answer loss would most like to raise (the most negative gradient), and the share of those that are
    prompt cells where the compiled source is not (BRIDGE.md §6's "first gradient toward an input")."""
    s = net.scores(lay).detach().requires_grad_(True)
    W = s.softmax(-1)
    S0 = net.prompt_state(lay, Nd, vd)
    _, P = net.run(lay, S0, ticks, mode="soft", W=W)
    r = lay.R - 1
    pa = net.channel(P, "d")[:, :, r][:, :, lay.answer[1]].permute(0, 2, 1)
    loss = -pa.clamp_min(1e-12).log().gather(-1, torch.as_tensor(yd)[..., None]).mean()
    loss.backward()
    best = s.grad.argmin(-1).numpy()                                     # [H, Li]
    kpad = net.kinds(lay)[0].argmax(0).numpy().ravel()
    n = d = 0
    for (h, kd) in PORTS:
        for rr in range(lay.R):
            for cc in range(lay.C):
                if lay.kind[rr, cc] != kd:
                    continue
                cj = port_source(lay, h, kd, rr, cc)
                if kpad[cj] == PROMPT:
                    continue
                d += 1
                n += int(kpad[best[h, rr * lay.C + cc]] == PROMPT)
    return round(n / max(1, d), 3)


# ----- how much credit does a nearly right wiring get? -----

# the bridge's menu of relative rules, as source descriptions on g's features
MENU = {"up": [("dr", -1), ("dc", 0)], "up2": [("dr", -2), ("dc", 0)], "left": [("dr", 0), ("dc", -1)],
        "upleft": [("dr", -1), ("dc", -1)], "nrow": [("kj", PROMPT), ("dc", 0)], "uptop": [("dr", -1), ("right_j", 0)],
        "zero": [("kj", OUT), ("dr", -1)]}


@torch.no_grad()
def port_swaps(B=3, w=2, k=1):
    """From the compiled machine, every wiring that differs in one port, and in two: each port's source replaced by
    another of the bridge's 7 relative rules. For each, the share of training answers right on the hard path and the
    soft answer loss; the compiled wiring for reference. Says whether the answers give a nearly right wiring partial
    credit."""
    lay = G.Layout(B, w, k)
    d = G.Data(B, w, k)
    base = compiled_hyper(B)
    off, _ = feature_offsets(base.nk)
    names = {tuple(v): n for n, v in MENU.items()}
    true = {hk: names[tuple(v)] for hk, v in PORTS.items()}

    def score(wiring):
        net = compiled_hyper(B)
        for u, (hk, opt) in enumerate(wiring.items()):
            conds = MENU[opt]
            idx = [feature_index(off, "ki", hk[1])] + [feature_index(off, n, v) for n, v in conds]
            net.G1[u].zero_()
            net.G1[u, idx] = 1.0
            net.g1[u] = -(len(idx) - 1)
        S0 = net.prompt_state(lay, d.Nd, d.vd)
        S, _ = net.run(lay, S0, lay.depth, mode="hard")
        got = net.channel(S, "d")[:, :, lay.R - 1][:, :, lay.answer[1]].argmax(1).numpy()
        hard = float((got == d.yd).all(-1).mean())
        _, P = net.run(lay, S0, lay.depth, mode="soft")
        pa = net.channel(P, "d")[:, :, lay.R - 1][:, :, lay.answer[1]].permute(0, 2, 1)
        nll = float(-pa.clamp_min(1e-12).log().gather(-1, torch.as_tensor(d.yd)[..., None]).mean())
        return hard, nll

    out = {"true": dict(zip(("hard", "nll"), score(true))), "one": [], "two": []}
    ports = list(true)
    for p in ports:
        for opt in MENU:
            if opt != true[p]:
                wv = {**true, p: opt}
                out["one"].append({"port": f"{'ABC'[p[0]]}.{G.KINDS[p[1]]}", "to": opt, **dict(zip(("hard", "nll"),
                                                                                                   score(wv)))})
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.choice(len(ports), 2, replace=False)
        oa = rng.choice([o for o in MENU if o != true[ports[a]]])
        ob = rng.choice([o for o in MENU if o != true[ports[b]]])
        wv = {**true, ports[a]: oa, ports[b]: ob}
        out["two"].append(dict(zip(("hard", "nll"), score(wv))))
    rand = []
    for _ in range(200):
        wv = {p: rng.choice(list(MENU)) for p in ports}
        rand.append(dict(zip(("hard", "nll"), score(wv))))
    out["random"] = rand
    return out


@torch.no_grad()
def check_compiled(sizes=((3, 2, 1), (3, 2, 2), (3, 3, 2), (3, 3, 3), (4, 2, 2), (5, 2, 2), (2, 3, 3), (2, 4, 4))):
    """The compiled generated wiring, checked like compile.py: every question of each size, every cell of every
    channel against grid.machine_reference (the "ports" machine), hard and soft, from blank and from random scratch
    cells, also at twice the ticks; both weightings. Writes nca/records/compile_hyper.json."""
    import json
    from pathlib import Path
    out = {}
    for gnorm in ("softmax", "sigmoid"):
        for B, w, k in sizes:
            net = compiled_hyper(B, gnorm=gnorm)
            lay, d = G.Layout(B, w, k), G.Data(B, w, k)
            ref = G.machine_reference(lay, d, "ports")
            res = {"questions": int(d.X), "g_units": len(PORTS), "g_numbers": int(sum(p.numel() for p in
                                                                                     (net.G1, net.g1, net.G2, net.g2))),
                   "rule_units": n_rule_clauses(B), "connections": int(lay.R * lay.C * (lay.R + 2) * (lay.C + 2) * 3)}
            ok = True
            for mode in ("hard", "soft"):
                for scratch, mult in (("blank", 1), ("random", 1), ("random", 2)):
                    S0 = net.prompt_state(lay, d.Nd, d.vd, scratch, torch.Generator().manual_seed(1))
                    S, _ = net.run(lay, S0, mult * lay.depth, mode=mode)
                    ok &= all(bool((net.channel(S, ch).argmax(1).numpy() == ref[ch]).all()) for ch in ("d", "n", "b"))
            res["every_cell_exact"] = ok
            out[f"{gnorm} {B},{w},{k}"] = res
            print(gnorm, (B, w, k), res, flush=True)
    path = Path(__file__).resolve().parent / "records" / "compile_hyper.json"
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    check_compiled()


@torch.no_grad()
def describe_wiring(net: HyperNCA, lay: G.Layout):
    """Row by row, place by place: which source each head reads on the hard path, as (row offset, column offset or
    "top", source kind class), next to the compiled machine's source. Returns a list of text lines."""
    src = net.weights(lay, hard=True).argmax(-1).numpy()
    R, C, C2 = lay.R, lay.C, lay.C + 2
    kpad = net.kinds(lay)[0].argmax(0).numpy().ravel()
    rows = [n if t < 0 else f"{n}({t},{s})" for n, t, s in lay.rows]

    def desc(r, c, j):
        jr, jc = divmod(int(j), C2)
        kind = "prompt" if kpad[j] == PROMPT else "edge" if kpad[j] == OUT else "cell"
        col = "top" if (jc == C and kpad[j] != OUT) else f"{jc - (c + 1):+d}"
        return f"{jr - (r + 1):+d},{col},{kind}"

    out = []
    for r in range(1, R):
        kd = lay.kind[r, 0]
        for c in range(C):
            i = r * C + c
            got = [desc(r, c, src[h, i]) for h in range(net.cfg.heads)]
            want = [desc(r, c, port_source(lay, h, kd, r, c)) if (h, kd) in PORTS else "-" for h in range(net.cfg.heads)]
            out.append(f"{rows[r]:>10} place {c}: reads {' | '.join(got):45s} machine {' | '.join(want)}")
    return out
