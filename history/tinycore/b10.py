"""The tiny core at microlab's own sizes: base 10, moduli odd N in [2^(b-1), 2^b), dividends v < N^2, so w = k = the
digits of 2^b - 1 (2 at 6 bits, 3 at 7 bits, 4 at 10 bits). Fresh batches every step, scored on a fixed sample.

Variants (`cell`, `choice`, `fix`):
- slot/borrow joint   microlab longdiv, with a table cell (r, n, slot, carry) -> (digit, carry)
- slot/est joint      microlab longdiv_est (window: top 3 digits of 10R + v_t, top 2 of N)
- mult/...            one shared subtractor (r, m, borrow) -> (digit, borrow); slot s subtracts the exact digits of s*N
- repeat/...          the same subtractor run s times for slot s
- fix "staged"        one quotient digit first (v < 10 N, one division step from v's top digits), then two, ...
- fix "range"         first only quotient digits 0 and 1, then all, then more quotient digits
- fix "checks"        the cell first learns subtraction from label-free identities, then `then` from answers
Options: `switch="borrow"` reads (and trains) the same tables with the cell's own borrow choice after the main phase,
so an estimator can bootstrap the cell; `transfer_bits` runs the trained tables unchanged at other widths;
`score_top` also scores the register's top digit; `mods="wide"` uses every w-digit modulus; `kr`/`kn` set the
estimator's window.

    python -m tinycore.b10 --bits 6 --cell mult --choice borrow --fix staged --steps 3000     (Modal: modal_tinycore.py)
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from .core import Config, Core, Data, digits
from .fixes import exact_adder, add_exact


def moduli(bits):
    return list(range((1 << (bits - 1)) + 1, 1 << bits, 2))


def width(bits, B=10):
    return len(np.base_repr((1 << bits) - 1, B))


def make(B, bits, kk, n, seed, start, final, mods="microlab", qmax=None):
    """A sample of n examples for kk division steps: v < N * B^kk, or v < N^2 on the final stage. mods="wide" uses
    every w-digit modulus instead of microlab's odd N in [2^(b-1), 2^b) (and v < N * B^kk throughout). qmax: every
    quotient digit <= qmax (v = q N + r), a curriculum over the digit's range."""
    w = width(bits, B)
    if mods == "wide":
        return Data(B, w, kk, moduli=list(range(B ** (w - 1), B ** w)), draw=n, seed=seed, start=start, qmax=qmax)
    if qmax is not None:
        return Data(B, w, kk, moduli=moduli(bits), draw=n, seed=seed, start=start, qmax=qmax)
    return Data(B, w, kk, moduli=moduli(bits), draw=n, seed=seed, start=start, vmax="NN" if final else "k")


def sub_once(m, x, bd, TD, TC):
    """x [X, P, B] digit distributions minus the number with digits bd [X, P], with the shared subtractor."""
    X, P, B = x.shape
    c = torch.zeros(X, 2)
    c[:, 0] = 1
    out = []
    for j in range(P):
        tdn, tcn = TD[:, bd[:, j]], TC[:, bd[:, j]]                          # [B(r), X, 2, out]
        out.append(torch.einsum("xr,rxco,xc->xo", x[:, j], tdn, c))
        c = torch.einsum("xr,rxco,xc->xo", x[:, j], tcn, c)
    return torch.stack(out, 1)


def label_free(m, B, w, steps, lr, batch, rng):
    """Subtraction from identities alone: (a - s*b) + s*b = a (slot cell, every slot s), or (a - b) + b = a (shared
    subtractor), with an exact adder, on random a (w+1 digits) and b. No remainders are used."""
    AD, AC = exact_adder(B)
    o = torch.optim.Adam([m.a, m.b], lr=lr)
    for _ in range(steps):
        a = rng.integers(0, B ** (w + 1), batch)
        ad = torch.from_numpy(digits(a, B, w + 1))
        x = F.one_hot(ad, B).double()
        TD, TC = m.tables()
        if m.cfg.cell == "slot":
            b = rng.integers(B ** (w - 1), B ** w, batch)
            bd = torch.from_numpy(digits(b, B, w))
            D, _ = m.ripple_slot(x, bd, TD, TC)
            L = 0
            for s in range(m.S):
                y = D[:, s]
                for _ in range(s):
                    y = add_exact(y, F.pad(bd, (0, 1)), AD, AC)
                L = L - y.clamp_min(1e-12).log().gather(-1, ad[..., None]).mean()
        else:
            b = rng.integers(0, B ** (w + 1), batch)
            bd = torch.from_numpy(digits(b, B, w + 1))
            y = add_exact(sub_once(m, x, bd, TD, TC), bd, AD, AC)
            L = -y.clamp_min(1e-12).log().gather(-1, ad[..., None]).mean()
        o.zero_grad()
        L.backward()
        o.step()


def run(B=10, bits=6, cell="mult", choice="borrow", fix="staged", steps=3000, lr=0.05, batch=256, seed=0, every=100,
        stage=1500, start="prefix", init=0.1, kr=3, kn=2, pre=500, test_n=4000, then="joint", mods="microlab",
        score_top=False, switch=None, switch_steps=0, transfer_bits=()):
    """switch="borrow": after training, read the same tables with the cell's own borrow choice (and train
    switch_steps more steps that way): the estimator bootstraps the cell, the borrow then chooses."""
    torch.manual_seed(seed)
    w = width(bits, B)
    k = w
    cfg = Config(B=B, w=w, k=k, cell=cell, choice=choice, kr=kr, kn=kn, init=init, start=start,
                 score_top=score_top)
    m = Core(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    test = make(B, bits, k, test_n, 10_000 + bits, start, True, mods)
    t0 = time.time()
    if fix == "checks":
        label_free(m, B, w, pre, lr, batch, rng)
    after_pre = m.exact_given(test) if fix == "checks" else None
    params = [m.a, m.b] + ([m.g] if choice in ("est", "estc") else [])
    o = torch.optim.Adam(params, lr=lr)
    # stages (quotient digits, top quotient digit allowed): "staged" adds one quotient digit at a time; "range"
    # first allows only quotient digits 0 and 1 (every lesson is "subtract N once"), then all, then adds digits
    if fix == "staged":
        stages = [(kk, None) for kk in range(1, k + 1)]
    elif fix == "range":
        stages = [(1, 1)] + [(kk, None) for kk in range(1, k + 1)]
    else:
        stages = [(k, None)]
    si = 0
    kk, qm = stages[si]
    stage_test = make(B, bits, kk, 2000, 20_000 + kk, start, kk == k and qm is None, mods, qm)
    since, curve, hit = 0, [], None
    for it in range(steps + 1):
        if it % every == 0:
            ex = m.exact(test)
            exs = m.exact(stage_test) if si < len(stages) - 1 else ex
            curve.append((it, kk, round(ex, 4)))
            if ex >= 0.99 and hit is None:
                hit = it
                break
            if si < len(stages) - 1 and (exs >= 0.99 or since >= stage):
                si, since = si + 1, 0
                kk, qm = stages[si]
                stage_test = make(B, bits, kk, 2000, 20_000 + kk, start, kk == k and qm is None, mods, qm)
        if it == steps:
            break
        d = make(B, bits, kk, batch, seed * 1_000_003 + it, start, kk == k and qm is None, mods, qm)
        L = m.loss(d)
        o.zero_grad()
        L.backward()
        o.step()
        since += 1
    switched = None
    if switch:
        m.cfg.choice = switch
        switched = {"exact_at_switch": m.exact(test)}
        o2 = torch.optim.Adam([m.a, m.b], lr=lr)
        sw_curve = []
        for it in range(switch_steps + 1):
            if it % every == 0:
                ex = m.exact(test)
                sw_curve.append((it, round(ex, 4)))
                if ex >= 0.99:
                    break
            if it == switch_steps:
                break
            d = make(B, bits, k, batch, seed * 1_000_003 + 7_777_777 + it, start, True, mods)
            L = m.loss(d)
            o2.zero_grad()
            L.backward()
            o2.step()
        switched["curve"] = sw_curve
        switched["final"] = sw_curve[-1][1]
    transfer = {}
    for tb in transfer_bits:                                # the same tables, run at another width
        w2 = width(tb, B)
        m2 = Core(Config(B=B, w=w2, k=w2, cell=cell, choice=m.cfg.choice, kr=kr, kn=kn, start=start), seed=seed)
        with torch.no_grad():
            for n in ("a", "b", "g"):
                if m2.params[n].shape == m.params[n].shape:
                    m2.params[n].copy_(m.params[n])
        transfer[tb] = m2.exact(make(B, tb, w2, 2000, 30_000 + tb, start, True, mods))
    smap, raw, mapped = m.slot_map(test)
    return {"B": B, "bits": bits, "w": w, "k": k, "cell": cell, "choice": choice, "fix": fix, "start": start,
            "switch": switch, "switched": switched, "transfer": transfer,
            "mods": mods, "score_top": score_top,
            "seed": seed, "steps": steps, "lr": lr, "hit": hit, "final": curve[-1][2], "curve": curve,
            "exact_hard": m.exact(test, hard=True), "exact_given": m.exact_given(test), "after_pre": after_pre,
            "slots": smap, "raw": raw, "mapped": mapped, "seconds": round(time.time() - t0, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bits", type=int, default=6)
    p.add_argument("--cell", default="mult")
    p.add_argument("--choice", default="borrow")
    p.add_argument("--fix", default="staged")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", default="prefix")
    a = p.parse_args()
    torch.set_num_threads(2)
    r = run(bits=a.bits, cell=a.cell, choice=a.choice, fix=a.fix, steps=a.steps, seed=a.seed, start=a.start)
    print(json.dumps({k: v for k, v in r.items() if k != "curve"}))


if __name__ == "__main__":
    main()
