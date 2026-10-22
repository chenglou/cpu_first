"""Up then down: explore while wide, commit while narrow (the parent session's arm, from the user's question "can't we
project up then down?").

A round:
1. **up**: lift the narrow network into a wide one that computes exactly the same function, then add noise of std
   `noise` so the copies can diverge (the first round starts from a random wide network, as the wide-only arm does);
2. train the wide network `K` steps with the notch's lessons (a fresh Adam);
3. **down**: merge the wide network back into a narrow one, and snap it onto an exact program (each menu port to its
   strongest option; each output row of the product circuit to one digit and one borrow, at the row's own scale);
4. check the narrow machine: exact on every training answer on the hard path? If so, stop and certify it;
5. repeat from the narrow point.

Two forms of "wide":
- **blend** (Net2Net): every compiled unit, or every port's gates, in C copies whose outputs are averaged; the merge is
  the inverse of the split (copies' outgoing weights averaged, the weights reading them summed; a menu port's copies
  merged as the log of their mean softmax, which is exact for softmax gates);
- **pop** (a population): C separate narrow networks, each trained on its own and scored on its own (hard training
  share, then the lessons' loss), the best one kept: whole candidates with exact scores, as the wiring tournament does,
  with gradient steps in between.

The notch is any of `notches.NOTCHES` whose freed parts are the product circuit (`gates` "learned") or a menu.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from handwired_bridge import train as T


@dataclass
class UDCfg(T.HRunCfg):
    ud_mode: str = "blend"        # "blend" | "pop"
    ud_copies: int = 4
    ud_k: int = 200               # steps per round
    ud_noise: float = 0.05        # std of the noise added at every lift
    ud_snap: bool = True


# ---------------------------------------------------------------- the product circuit: lift and merge (blend)

def _lift_matrix(W, nz0, C):
    cols = W.repeat_interleave(C, dim=1)
    port, units = cols[:nz0], cols[nz0:]
    return torch.cat([port, units.repeat_interleave(C, dim=0) / C], 0)


def _merge_matrix(W, nz0, C):
    port, units = W[:nz0], W[nz0:]
    units = units.reshape(-1, C, W.shape[1]).sum(1)
    W = torch.cat([port, units], 0)
    return W.reshape(W.shape[0], -1, C).mean(-1)


@torch.no_grad()
def lift_cell(narrow, wide, C, noise, gen):
    nz0 = 3 * narrow.S
    for i in range(narrow.nlayers):
        for name in ("WL", "WG"):
            wide.p[f"{name}{i}"].copy_(_lift_matrix(narrow.p[f"{name}{i}"], nz0, C))
        for name in ("bL", "bG"):
            wide.p[f"{name}{i}"].copy_(narrow.p[f"{name}{i}"].repeat_interleave(C))
        for name in ("WL", "WG", "bL", "bG"):
            p = wide.p[f"{name}{i}"]
            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * noise)


@torch.no_grad()
def merge_cell(wide, narrow, C):
    nz0 = 3 * narrow.S
    for i in range(narrow.nlayers):
        for name in ("WL", "WG"):
            # the masked effective weights: entries outside the mask are not part of the function
            m = getattr(wide, f"m{name[1]}{i}")
            narrow.p[f"{name}{i}"].copy_(_merge_matrix(wide.p[f"{name}{i}"] * m, nz0, C))
        for name in ("bL", "bG"):
            narrow.p[f"{name}{i}"].copy_(wide.p[f"{name}{i}"].reshape(-1, C).mean(-1))


@torch.no_grad()
def snap_cell(cell, B):
    """Each output row of the circuit's last layer to one digit and one borrow, at the row's own largest weight."""
    i = cell.nlayers - 1
    W = cell.p[f"WL{i}"] * getattr(cell, f"mL{i}")
    lo, hi = cell.layer_slices[i - 1] if i > 0 else (0, W.shape[0])
    rows = W[lo:hi]                                                          # [joint units, B + 2]
    for a, b in ((0, B), (B, B + 2)):
        g = rows[:, a:b]
        top = g.abs().argmax(-1)
        keep = F.one_hot(top, b - a).to(g.dtype)
        g.copy_(g * keep)
    W[lo:hi] = rows
    cell.p[f"WL{i}"].copy_(W)


# ---------------------------------------------------------------- menus: lift and merge (blend)

@torch.no_grad()
def lift_menu(narrow_blk, wide_blk, C, noise, gen):
    g = narrow_blk.menu_g.expand(-1, C, -1, -1).clone()
    wide_blk.menu_g.copy_(g + torch.randn(g.shape, generator=gen, dtype=g.dtype) * noise)


@torch.no_grad()
def merge_menu(net_w, wide_blk, narrow_blk):
    w = net_w.mux(wide_blk.menu_g, -1, False).mean(1)                        # [3, NK, M] the ports' blended weights
    if net_w.cfg.mux == "softmax":
        narrow_blk.menu_g.copy_(w.clamp_min(1e-12).log()[:, None])
    else:
        narrow_blk.menu_g.copy_(w[:, None] * wide_blk.menu_g.shape[-1])


@torch.no_grad()
def snap_menu(net, blk, beta=4.0):
    g = blk.menu_g[:, 0]
    hot = F.one_hot(g.argmax(-1), g.shape[-1]).to(g.dtype)
    logit = net.cfg.mux in ("softmax", "sigmoid_free", "softplus")
    blk.menu_g.copy_((beta * (2 * hot - 1) if logit else hot)[:, None])


# ---------------------------------------------------------------- the loop

def _train(run, steps, k):
    for _ in range(steps):
        run.opt.zero_grad()
        loss, terms = run.lessons(k)
        loss.backward()
        run.opt.step()
        run.step += 1
    return float(loss.detach())


def _fresh_opt(run):
    params = [p for p in run.net.parameters() if p.requires_grad]
    run.opt = torch.optim.Adam(params, lr=run.cfg.lr)


def _sub_cell(net):
    return net.blocks[0].cells["SUB"]


def run(cfg: UDCfg) -> dict:
    t0 = time.time()
    c = cfg
    C = c.ud_copies
    gen = torch.Generator().manual_seed(4242 + c.seed)
    narrow_cfg = T.HRunCfg(**{k: v for k, v in asdict(c).items() if k in T.HRunCfg.__dataclass_fields__})
    narrow = T.Run(narrow_cfg)
    is_menu = narrow.ncfg.route == "menu"
    rounds = []
    if c.ud_mode == "blend":
        wide_cfg = copy.deepcopy(narrow_cfg)
        wide_cfg.net = dict(wide_cfg.net, copies=C)
        wide = T.Run(wide_cfg)
    else:
        pop_cfgs = [copy.deepcopy(narrow_cfg) for _ in range(C)]
        for j, pc in enumerate(pop_cfgs):
            pc.seed = c.seed * 100 + j                                       # different random starts
        pop = [T.Run(pc) for pc in pop_cfgs]
    step, hit = 0, None
    while step < c.steps:
        if c.ud_mode == "blend":
            if step:                                  # the first round starts from the wide network's own random start
                if is_menu:
                    lift_menu(narrow.net.blocks[0], wide.net.blocks[0], C, c.ud_noise, gen)
                else:
                    lift_cell(_sub_cell(narrow.net), _sub_cell(wide.net), C, c.ud_noise, gen)
            _fresh_opt(wide)
            loss = _train(wide, c.ud_k, c.k)
            if is_menu:
                merge_menu(wide.net, wide.net.blocks[0], narrow.net.blocks[0])
            else:
                merge_cell(_sub_cell(wide.net), _sub_cell(narrow.net), C)
            wide_exact = wide.exact(c.B, c.w, c.k)
        else:
            if step:                                                         # re-lift: clone the best, add noise
                for r in pop:
                    if r is not best:
                        r.net.load_state_dict(best.net.state_dict())
                    with torch.no_grad():
                        for p in r.net.parameters():
                            p.add_(torch.randn(p.shape, generator=gen, dtype=p.dtype) * c.ud_noise)
            scores = []
            for r in pop:
                _fresh_opt(r)
                l_ = _train(r, c.ud_k, c.k)
                scores.append((r.exact(c.B, c.w, c.k), -l_))
            best = pop[max(range(C), key=lambda j: scores[j])]
            narrow.net.load_state_dict(best.net.state_dict())
            wide_exact = max(s[0] for s in scores)
            loss = -max(scores)[1]
        step += c.ud_k
        merged_exact = narrow.exact(c.B, c.w, c.k)
        if c.ud_snap:
            if is_menu:
                snap_menu(narrow.net, narrow.net.blocks[0])
            elif narrow.ncfg.gates == "learned":
                snap_cell(_sub_cell(narrow.net), c.B)
        narrow_exact = narrow.exact(c.B, c.w, c.k)
        rounds.append({"step": step, "loss": loss, "wide_exact": wide_exact, "merged_exact": merged_exact,
                       "narrow_exact": narrow_exact, **{kx: v for kx, v in narrow.inspect().items()}})
        if c.ud_mode == "pop":
            best.net.load_state_dict(narrow.net.state_dict())               # the snapped point seeds the next round
        if narrow_exact == 1.0:
            hit = step
            break
    rec = {"steps": step, "hit": hit, "rounds": rounds, "seconds": time.time() - t0}
    rec["train_hard"] = narrow.exact(c.B, c.w, c.k)
    rec["train_soft"] = narrow.exact(c.B, c.w, c.k, hard=False)
    rec.update(narrow.inspect())
    bigger = [(c.B, c.w, c.k + 1), (c.B, c.w + 1, c.k + 1)]
    rec["bigger"] = {f"{b},{w_},{k_}": narrow.exact(b, w_, k_, sample=c.cert_sample) for b, w_, k_ in bigger}
    rec["certified"] = bool(rec["train_hard"] == 1.0 and all(v == 1.0 for v in rec["bigger"].values()))
    rec["cfg"] = asdict(cfg)
    return rec
