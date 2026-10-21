"""Job lists for the tiny cores (modal_tiny.py) and the executor every container runs.

A job is {"name", "kind": "div" | "sq", "cfg": {KEY: VALUE}}; the executor runs tiny_div.run (or tiny_sq.run) with the
defaults updated by cfg and returns its result dict.
"""

from __future__ import annotations

import itertools


def execute(job):
    if job["kind"] == "div":
        from learned_wiring.tiny import tiny_div as m
    elif job["kind"] == "io":
        from learned_wiring.tiny import tiny_io as m
    else:
        from learned_wiring.tiny import tiny_sq as m
    cfg = dict(m.CFG)
    cfg.update(job["cfg"])
    return m.run(cfg)


def s1():
    """Stage 1 in miniature: the recipe (fmix chain + fmix / mix I/O) and the soft chain, 4 seeds, two sizes."""
    jobs = []
    for (W, widths), chain, io, seed in itertools.product(
            [(3, (2,)), (4, (2, 3))], ["fmix", "soft"], ["fmix", "mix"], range(4)):
        jobs.append({"name": f"W{W}_{chain}_{io}_s{seed}", "kind": "div",
                     "cfg": dict(W=W, widths=widths, chain=chain, io=io, seed=seed)})
    return jobs


def s1b():
    """Stage 1 in miniature, W = 4 (N of 2 and 3 digits in training, 4 at certification): the recipe and its variants."""
    base = dict(W=4, widths=(2, 3))
    variants = [
        ("recipe", {}, 6),
        ("chainsoft", dict(chain="soft"), 4),
        ("iomix", dict(io="mix"), 4),
        ("iosoft", dict(io="soft"), 4),
        ("softall", dict(io="soft_all"), 4),
        ("softall_dense", dict(io="soft_all", label_frac=1.0), 4),
        ("nolabels", dict(label_frac=0.0), 4),
        ("answersonly", dict(io_laws=0), 4),
        ("nostecin", dict(ste_cin=0), 2),
        ("union", dict(operands="union"), 2),
        ("designedadder", dict(warm=100000), 2),
        ("notablefirst", dict(chain_from=1), 2),
    ]
    jobs = []
    for name, kw, seeds in variants:
        for seed in range(seeds):
            jobs.append({"name": f"{name}_s{seed}", "kind": "div", "cfg": dict(base, seed=seed, **kw)})
    return jobs


def s2():
    """Stage 2 in miniature: the squaring's layout (per row and column, or tied) and the product cell learned together
    by the routed ladder, with and without the cell's own ladder, 4 seeds, W = 3 and 4."""
    jobs = []
    for W, route, cl, seed in itertools.product([3, 4], ["soft", "tied"], [1.0, 0.0], range(4)):
        jobs.append({"name": f"W{W}_{route}_cl{int(cl)}_s{seed}", "kind": "sq",
                     "cfg": dict(W=W, route=route, cell_ladder=cl, seed=seed)})
    for seed in range(4):                                    # one row per number per step, as at H1 scale
        jobs.append({"name": f"W4_soft_cl1_rows1_s{seed}", "kind": "sq",
                     "cfg": dict(W=4, route="soft", cell_ladder=1.0, rows_per_step=1, seed=seed)})
    return jobs


def s3():
    """Stage 3 in miniature: each field's pointer shape as a mixture over step / offsets / absolute, and each shape
    alone (IO.md section 6's contrast, here learned from the whole-reading lessons)."""
    jobs = []
    for shapes, seeds in (("step,offsets,absolute", 6), ("step", 2), ("offsets", 2), ("absolute", 2),
                          ("step,offsets", 4)):
        for seed in range(seeds):
            jobs.append({"name": f"{shapes.replace(',', '+')}_s{seed}", "kind": "io", "cfg": dict(shapes=shapes, seed=seed)})
    return jobs


def s3b():
    """Stage 3 in miniature, one shape switch shared by the four fields (one switch per kind of connection)."""
    jobs = []
    for shapes, seeds in (("step,offsets,absolute", 8), ("step,offsets", 4)):
        for seed in range(seeds):
            jobs.append({"name": f"tied_{shapes.replace(',', '+')}_s{seed}", "kind": "io",
                         "cfg": dict(shapes=shapes, seed=seed, tie=1)})
    return jobs


def s2b():
    """Stage 2 in miniature with H1's condition: training numbers one digit shorter than at certification (W = 4,
    training numbers below 1000). Per-column sources, the tied offset, the placement rule as three switches, and the
    placement rule with one weight per whole rule; and the whole-rule form with full-width training."""
    jobs = []
    for route, td in (("soft", 3), ("tied", 3), ("rule", 3), ("rule_mix", 3), ("rule_mix", 0)):
        for seed in range(4):
            jobs.append({"name": f"W4_{route}_td{td}_s{seed}", "kind": "sq",
                         "cfg": dict(W=4, route=route, train_digits=td, cell_ladder=1.0, seed=seed)})
    return jobs


SUITES = {"s1": s1, "s1b": s1b, "s2": s2, "s2b": s2b, "s3": s3, "s3b": s3b}
