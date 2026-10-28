"""Job lists for the handwired bridge's sweeps (modal_hb.py) and the executor every container runs.

A job is {"name", "notch", "tier", "size", "cfg": train.HRunCfg fields}. Its record (train.run's output) keeps the curve
every `every` steps. Lesson tiers are the attention bridge's (`bridge.notches.TIERS`); "+f" freezes the routing for the
first 300 steps (tables first), "+l1" adds sparsity pressure on the freed routing or cells.
"""

from __future__ import annotations

import itertools

from bridge.notches import TIERS

from handwired_bridge import notches as Nt

SIZES = [(3, 2, 1), (3, 2, 2)]
SEEDS = range(4)


def job(name, notch, tier, B, w, k, seed, steps=3000, **kw):
    base, _, flags = tier.partition("+")
    net = dict(Nt.NOTCHES[notch])
    cfg = dict(B=B, w=w, k=k, net=net, lessons=dict(TIERS[base]), seed=seed, steps=steps, tag=f"{notch}/{tier}",
               lr=0.01 if Nt.needs_low_lr(net) else 0.05, dtype="float64", patience=1500, readout=Nt.readout(net))
    for flag in flags.split("+") if flags else ():
        if flag == "f":
            cfg["route_freeze"] = 300
        elif flag == "l1":
            cfg["l1"] = 1e-3
        elif flag == "mse":
            cfg["readout"] = "mse"
        else:
            raise ValueError(flag)
    cfg.update(kw)
    return {"name": name, "notch": notch, "tier": tier, "size": f"{B},{w},{k}", "cfg": cfg}


LADDER = [
    ("H1_table", ("A", "G", "AC", "GC", "ACL")),
    ("H1_table_raw", ("AC", "GC")),
    ("H2_gates_sub", ("AC", "GC", "GCL")),
    ("H2_gates_sub_i1", ("AC", "GC")),
    ("H2_gates", ("GC", "GCL")),
    ("H2_dense_sub", ("AC", "GC", "GC+l1", "ACnb")),
    ("H2_dense_sub_full", ("AC", "GC")),
    ("H2_dense", ("GC", "GCL")),
    ("H3_menu_given_softmax", ("A", "AL")),
    ("H3_menu_given_relu", ("A", "AL")),
    ("H3_menu_given_relu_free", ("A", "AL")),
    ("H3_menu_given_raw", ("A", "AL")),
    ("H3_menu_softmax", ("ACL", "ACL+f")),
    ("H3_menu_relu", ("ACL", "ACL+f")),
    ("H3_menu_raw", ("ACL", "ACL+f")),
    ("H3_dense_given_softmax", ("A", "AL")),
    ("H3_dense_given_relu", ("A", "AL", "AL+l1")),
    ("H3_dense_given_raw", ("AL", "AL+l1")),
    ("H3_dense_relu", ("ACL+f",)),
    ("H4_untied_table", ("AC", "GC")),
    ("H4_untied_dense_sub", ("AC", "GC")),
    ("H4_untied_menu_given_softmax", ("AL",)),
    ("H4_untied_menu_given_relu", ("AL",)),
]


def ladder():
    jobs = []
    for (notch, tiers), (B, w, k), seed in itertools.product(LADDER, SIZES, SEEDS):
        for tier in tiers:
            jobs.append(job(f"{notch}.{tier}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed))
    return jobs


def width():
    """The menu's width without softmax competition: the tables given, the answers alone (A) and with the one-link law
    (AL), each port a multiplexer over 7, 10, 13, 16 or 20 relative sources, for every multiplexer (softmax is the
    attention bridge's control). (3, 2, 1), 4 seeds, 2,000 steps (the attention bridge's width table)."""
    jobs = []
    for mux, n, seed in itertools.product(Nt.MUXES, (7, 10, 13, 16, 20), SEEDS):
        notch = f"H3_menu_given_{mux}" if n == 7 else f"H3_rel{n}_given_{mux}"
        for tier in ("A", "AL"):
            jobs.append(job(f"{notch}.{tier}.321.s{seed}", notch, tier, 3, 2, 1, seed, steps=2000))
    return jobs


def width2():
    """The width table for the multiplexer whose gates never die (normalized softplus), and a longer budget (6,000
    steps) for softmax and softplus at 13-20 options: is the edge a budget? (3, 2, 1), 4 seeds."""
    jobs = []
    for n, seed in itertools.product((7, 10, 13, 16, 20), SEEDS):
        notch = "H3_menu_given_softplus" if n == 7 else f"H3_rel{n}_given_softplus"
        for tier in ("A", "AL"):
            jobs.append(job(f"{notch}.{tier}.321.s{seed}", notch, tier, 3, 2, 1, seed, steps=2000))
    for mux, n, seed in itertools.product(("softmax", "softplus"), (13, 16, 20), SEEDS):
        notch = f"H3_rel{n}_given_{mux}"
        j = job(f"{notch}.AL.6k.321.s{seed}", notch, "AL", 3, 2, 1, seed, steps=6000, patience=6000)
        j["tier"] = "AL 6000 steps"
        jobs.append(j)
    return jobs


def diagnose():
    """The compiled machine as a probe (probe.py), on the routing notches and the freed cells: where each port's
    strongest source points at the end (right, a prompt cell, a constant zero, another computed cell), where the
    answers' first gradient pushes it, and the answer loss on the straight line to the compiled weights. (3, 2, 1),
    seeds 0 and 1, 2,000 steps."""
    plan = [("menu_answers", "H3_menu_given_{m}", "A", ("softmax", "relu", "relu_free", "raw")),
            ("menu_link", "H3_menu_given_{m}", "AL", ("relu_free", "sigmoid_free")),
            ("menu10_link", "H3_rel10_given_{m}", "AL", ("softmax", "relu", "raw", "softplus")),
            ("menu13_link", "H3_rel13_given_{m}", "AL", ("softmax", "relu", "raw", "softplus")),
            ("dense_link", "H3_dense_given_{m}", "AL", ("softmax", "relu", "relu_free", "raw")),
            ("dense_answers", "H3_dense_given_{m}", "A", ("softmax", "raw"))]
    jobs = []
    for case, notch, tier, muxes in plan:
        for m, seed in itertools.product(muxes, (0, 1)):
            j = job(f"{case}.{m}.s{seed}", notch.format(m=m), tier, 3, 2, 1, seed, steps=2000)
            j["probe"] = True
            jobs.append(j)
    for notch, tier in (("H2_dense_sub_full", "AC"), ("H2_gates_sub", "AC"), ("H2_dense_sub", "AC"), ("H2_dense", "GC"),
                       ("H2_gates", "GC")):
        for seed in (0, 1):
            j = job(f"{notch}.{tier}.s{seed}", notch, tier, 3, 2, 1, seed, steps=2000)
            j["probe"] = True
            jobs.append(j)
    return jobs


STAB = [("H2_gates_sub", "GC"), ("H3_menu_given_softmax", "AL"), ("H3_menu_given_raw", "AL")]


def wstab():
    """Stability near the compiled point (the Net2Net arm): start at the compiled machine (logits and gates at scale
    4), add Gaussian noise of std sigma to every learned tensor, train 1,000 steps with the notch's lessons or with the
    answers alone, minimal or widened (every compiled unit, or every port's gates, in 4 copies: Net2Net). Is the
    solution kept, recovered or lost? (3, 2, 1), 4 seeds."""
    jobs = []
    for (notch, tier), copies, sigma, seed in itertools.product(STAB, (1, 4), (0.1, 0.3, 1.0, 3.0), SEEDS):
        if notch == "H1_table" and copies > 1:
            continue
        for t in (tier, "A"):
            wide = "wide" if copies > 1 else "min"
            j = job(f"{notch}.{wide}.{t}.s{sigma}.321.s{seed}", notch, t, 3, 2, 1, seed, steps=1000, patience=1000,
                    init="compiled", init_beta=4.0, init_noise=sigma)
            j["cfg"]["net"]["copies"] = copies
            j["tier"] = f"{t} {wide} sigma {sigma}"
            jobs.append(j)
    return jobs


def wlearn():
    """Learnability from random, minimal against widened (4 copies of every unit or every port's gates)."""
    jobs = []
    plan = [("H2_gates_sub", "GC"), ("H3_rel13_given_softmax", "AL"), ("H3_rel13_given_raw", "AL"),
            ("H3_dense_given_raw", "AL"), ("H3_dense_given_softmax", "AL")]
    for (notch, tier), copies, (B, w, k), seed in itertools.product(plan, (1, 4), SIZES, SEEDS):
        wide = "wide" if copies > 1 else "min"
        j = job(f"{notch}.{wide}.{tier}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed)
        j["cfg"]["net"]["copies"] = copies
        j["tier"] = f"{tier} {wide}"
        jobs.append(j)
    return jobs


def proj():
    """Projected training (a second reading of "projection", the snapping one; the user's own reading is `lift`): every
    `project_every` steps the learned parts are snapped onto the nearest exact program (table rows and menu gates to
    their strongest entry at scale 4; product weights to the nearest integer) and training continues from there. From
    random, (3, 2, 1), against the same runs without (the ladder)."""
    jobs = []
    plan = [("H1_table", "A"), ("H1_table", "AC"), ("H3_menu_given_softmax", "A"), ("H3_rel13_given_softmax", "AL"),
            ("H3_rel13_given_raw", "AL"), ("H2_gates_sub", "GC")]
    for (notch, tier), every, (B, w, k), seed in itertools.product(plan, (25, 100), [(3, 2, 1)], SEEDS):
        j = job(f"{notch}.{tier}.p{every}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed, project_every=every,
                project_beta=4.0)
        j["tier"] = f"{tier} every {every}"
        jobs.append(j)
    return jobs


def lift():
    """The user's projection: the subtraction cell's compiled product circuit re-expressed through a fixed random
    projection of each port's state (into 20 or 40 dimensions), exact at the start. Near the compiled point (as wstab:
    noise sigma, 1,000 steps, the cell's lessons or the answers) and from random (as the ladder), against the minimal
    and the Net2Net-widened circuit in wstab / wlearn. 4 seeds."""
    jobs = []
    for notch, sigma, seed in itertools.product(("H2_gates_sub_lift20",), (0.1, 0.3, 1.0, 3.0), SEEDS):
        for t in ("GC", "A"):
            j = job(f"{notch}.proj.{t}.s{sigma}.321.s{seed}", notch, t, 3, 2, 1, seed, steps=1000, patience=1000,
                    init="compiled", init_beta=4.0, init_noise=sigma)
            j["tier"] = f"{t} projected sigma {sigma}"
            jobs.append(j)
    for notch, (B, w, k), seed in itertools.product(("H2_gates_sub_lift20", "H2_gates_sub_lift40"), SIZES, SEEDS):
        jobs.append(job(f"{notch}.GC.{B}{w}{k}.s{seed}", notch, "GC", B, w, k, seed))
    return jobs


def over():
    """Over-parameterization from random, by unit copies (Net2Net-style, 4 per compiled unit): the product circuit with
    the table's pattern given by its output sparsity (x4: AC; GC is in wlearn), and with dense output rows (the table's
    pattern learned too), minimal and x4. 4 seeds, both sizes."""
    jobs = []
    plan = [("H2_gates_sub_x4", ("AC",)), ("H2_gates_sub_do", ("AC", "GC")), ("H2_gates_sub_do_x4", ("AC", "GC")),
            ("H2_dense_sub_b3", ("AC", "GC")), ("H2_dense_sub_x4", ("AC", "GC"))]
    for (notch, tiers), (B, w, k), seed in itertools.product(plan, SIZES, SEEDS):
        for tier in tiers:
            jobs.append(job(f"{notch}.{tier}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed))
    return jobs


def updown():
    """Up then down (updown.py): explore wide, commit narrow, every K = 200 steps; blended (Net2Net copies, merged) or a
    population (4 separate networks, the best kept), the narrow machine snapped onto an exact program and checked after
    every round. The product circuit (table's pattern given, and learned) and the 13-option menu; 2,000 steps, both
    sizes, 4 seeds. The comparisons are in the ladder and width (narrow only), wlearn and over (wide only) and proj
    (snapping only)."""
    jobs = []
    plan = [("H2_gates_sub", "GC"), ("H2_gates_sub_do", "GC"), ("H3_rel13_given_softmax", "AL")]
    for (notch, tier), mode, (B, w, k), seed in itertools.product(plan, ("blend", "pop"), SIZES, SEEDS):
        j = job(f"{notch}.{tier}.{mode}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed, steps=2000, ud_mode=mode)
        j["tier"] = f"{tier} up-down {mode}"
        j["updown"] = True
        jobs.append(j)
    return jobs


def bases():
    """The cell notches at bases 4 and 5 (the attention bridge's mlpcell sizes): the learned table, the freed product
    circuit, the dense ReGLU cell, and the H1 form (the table a dense ReGLU cell's outputs; also at base 3), with the
    cell's lessons (with and without bijectivity). 4 seeds."""
    jobs = []
    plan = [("H1_table", ("AC", "GC")), ("H2_dense_sub", ("AC", "GC", "ACnb")), ("H2_gates_sub", ("GC",)),
            ("H2_densetab_sub", ("AC", "GC"))]
    for (notch, tiers), (B, w, k), seed in itertools.product(plan, ((3, 2, 1), (3, 2, 2), (4, 2, 1), (5, 2, 1)),
                                                             SEEDS):
        if (B, w, k)[0] == 3 and notch != "H2_densetab_sub":
            continue                                                         # base 3: in the ladder
        for tier in tiers:
            jobs.append(job(f"{notch}.{tier}.{B}{w}{k}.s{seed}", notch, tier, B, w, k, seed))
    return jobs


SUITES = {"ladder": ladder, "width": width, "width2": width2, "diagnose": diagnose, "wstab": wstab, "wlearn": wlearn,
          "proj": proj, "bases": bases, "lift": lift, "over": over, "updown": updown}


def execute(j: dict) -> dict:
    import time

    from handwired_bridge import train as T
    t0 = time.time()
    if j.get("updown"):
        from handwired_bridge import updown as U
        rec = U.run(U.UDCfg(**j["cfg"]))
    elif not j.get("probe"):
        rec = T.run(T.HRunCfg(**j["cfg"]))
    else:
        from dataclasses import asdict

        from handwired_bridge import probe as P
        cfg = T.HRunCfg(**j["cfg"])
        run = T.Run(cfg)
        extra = {}
        if run.ncfg.route != "fixed":
            extra["grad0"] = P.grad0(run)
            extra["ports_start"] = P.ports(run)
        rec = run.train()
        rec["cfg"] = asdict(cfg)
        if run.ncfg.route != "fixed":
            extra["ports"] = P.ports(run)
            extra["path"] = P.path(run)
        extra["cells"] = P.cells(run)
        extra["relabel"] = P.relabel(run)
        if run.ncfg.gates == "dense":
            extra["irrelevant"] = P.irrelevant(run)
        rec.update(extra)
    rec.update(name=j["name"], notch=j["notch"], tier=j["tier"], size=j["size"], wall=time.time() - t0)
    return rec
