"""The tiny-core suites of MENUS.md section 2 (each a list of jobs: a name and tiny_wide.CFG overrides)."""

from __future__ import annotations

SEEDS = range(8)


def job(name, **kw):
    return {"name": name, "cfg": kw}


def grow():
    """The menu ladder: every division level against the multiply levels (the io3 recipe: division ladder, whole I/O
    block enumerated, laws x mod N = x, N mod N = 0 and the 2% answers)."""
    out = []
    for div in range(5):
        for mul in (0, 2, 3):
            for s in SEEDS:
                out.append(job(f"d{div}m{mul}_s{s}", div=div, mul=mul, seed=s,
                               max_steps=1500 if div < 4 else 800))
    return out


def laws():
    """Which lessons are needed: no step law (the chain taught by whole-division laws, jointly or block by block),
    and the I/O block's laws."""
    out = []
    for div in (0, 1):
        for law in ("joint", "whole"):
            for lw in ("xna", "xnqa"):
                for s in SEEDS:
                    out.append(job(f"{law}_{lw}_d{div}_s{s}", div=div, mul=2, chain_law=law, laws=lw, seed=s,
                                   max_steps=800))
    for lw in ("xn", "a", "xnqa", "q"):
        for s in SEEDS:
            out.append(job(f"ladder_{lw}_d3_s{s}", div=3, mul=2, laws=lw, seed=s, max_steps=1000))
    return out


def io():
    """How the I/O block is taught when it is large: sampled coordinates, or the local step law."""
    out = []
    for div, mul in ((3, 2), (4, 3)):
        for mode in ("cs", "local"):
            for s in SEEDS:
                out.append(job(f"{mode}_d{div}m{mul}_s{s}", div=div, mul=mul, io=mode, seed=s, max_steps=1000))
    return out


def w4():
    """A deeper tiny core: W = 4 (N of 2 and 3 digits in training, certified up to 4 digits)."""
    out = []
    for div, mul in ((0, 0), (3, 2), (4, 3)):
        for mode in ("enum", "local"):
            if div == 0 and mode == "local":
                continue
            for s in SEEDS:
                out.append(job(f"w4_{mode}_d{div}m{mul}_s{s}", W=4, widths=(2, 3), pool=8000, div=div, mul=mul,
                               io=mode, seed=s, max_steps=1000))
    return out


SUITES = {"grow": grow, "laws": laws, "io": io, "w4": w4}
