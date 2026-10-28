"""Job lists for the bridge sweeps (modal_bridge.py) and the executor every container runs.

A job is {"name", "cfg": train.RunCfg fields}. Its record (train.run's output) keeps the curve every `every` steps.
"""

from __future__ import annotations

import itertools

from bridge import notches as Nt

SIZES = [(3, 2, 1), (3, 2, 2)]
SEEDS = range(4)


def heavy(notch):
    n = Nt.NOTCHES[notch]
    return n.get("rule") == "mlp" or n.get("fmt", "digits") != "digits"


def job(name, notch, tier, B, w, k, seed, steps=3000, **kw):
    """Table and menu notches: full batch, float64, lr 0.05. MLP and free-format notches: lr 0.01, float32,
    minibatches of 64 questions, 2,000 steps."""
    cfg = dict(B=B, w=w, k=k, net=dict(Nt.NOTCHES[notch]), lessons=dict(Nt.TIERS[tier]), seed=seed, steps=steps,
               tag=f"{notch}/{tier}")
    if heavy(notch):
        cfg.update(lr=0.01, dtype="float32", batch=64, steps=min(steps, 2000), patience=1000)
    cfg.update(kw)
    return {"name": name, "notch": notch, "tier": tier, "size": f"{B},{w},{k}", "cfg": cfg}


# the ladder: every notch with the lesson tiers it can express (cell-level tiers only where the subtraction cell is
# identifiable; network-level tiers everywhere). "+f": the attention frozen for the first 300 steps (tables first);
# "+s": the grid grown one link at a time
LADDER = [
    ("N1_table", ("A", "G", "AC", "ACL", "GCL")),
    ("N1_tables", ("A", "G", "ACL", "GCL")),
    ("N1_mlp", ("A", "G", "ACL", "GCL")),
    ("N1_mlp_st", ("GCL",)),
    ("N2_narrow_given", ("A",)),
    ("N2_narrow", ("A", "AC", "ACL")),
    ("N2_menu_given", ("A", "AL")),
    ("N2_menu", ("A", "AC", "ACL", "ACL+f")),
    ("N3_free_given", ("A", "AL", "AL+s")),
    ("N3_free", ("A", "AC", "ACL", "ACL+f")),
    ("N3_free_mlp", ("A", "ACL+f", "GCL+f")),
    ("N4_symbols", ("A", "G", "GL")),
    ("N4_symbols_untied", ("A", "G")),
    ("N5_free", ("A", "G", "GL")),
    ("N6_plain", ("A", "G")),
]


def ladder():
    jobs = []
    for (notch, tiers), (B, w, k), seed in itertools.product(LADDER, SIZES, SEEDS):
        for tier in tiers:
            base, _, flag = tier.partition("+")
            kw = {"route_freeze": 300} if flag == "f" else {"link_stages": (1, B - 1)} if flag == "s" else {}
            jobs.append(job(f"{notch}.{tier}.{B}{w}{k}.s{seed}", notch, base, B, w, k, seed, **kw))
    return jobs


def mlpcell():
    """The subtraction cell as a generic ReLU MLP (the copy and the select compiled), applied to the ports directly
    ("mlp", 54 hidden units) or as the table of its outputs on every one-hot input ("mlptab", 256 hidden units: the
    H1 file's form), against the learned table, at bases 3-5; with and without bijectivity; with fixed attention
    and with the wide menu (+ the link law and tables first)."""
    jobs = []
    sizes = [(3, 2, 1), (3, 2, 2), (4, 2, 1), (5, 2, 1)]
    plan = [("N1_table", ("AC", "GC")), ("N1_mlp_sub", ("AC", "GC", "ACnb")),
            ("N1_mlptab_sub_w256", ("AC", "GC", "ACnb")), ("N1_mlp_sub_st", ("GC",)), ("N1_mlp", ("GCL",)),
            ("N2_menu", ("ACL+f",)), ("N2_menu_mlptab_sub", ("ACL+f",))]
    for (notch, tiers), (B, w, k), seed in itertools.product(plan, sizes, SEEDS):
        for tier in tiers:
            base, _, flag = tier.partition("+")
            kw = {"route_freeze": 300} if flag == "f" else {}
            if "mlp" in notch:
                kw.update(lr=0.01, dtype="float64", batch=0, steps=3000, patience=1500)
            jobs.append(job(f"{notch}.{tier}.{B}{w}{k}.s{seed}", notch, base, B, w, k, seed, **kw))
    return jobs


def plain():
    """The plain looped transformer (notch 6) with a larger budget than the ladder's: full batch, 5,000 steps, widths
    32 and 64, learning rates 0.003 and 0.001, answers only, (3, 2, 1). Does it at least fit the training questions?"""
    jobs = []
    for d, lr, seed in itertools.product((32, 64), (0.003, 0.001), (0, 1)):
        j = job(f"N6_plain.d{d}.lr{lr}.321.s{seed}", "N6_plain", "A", 3, 2, 1, seed, lr=lr, batch=0, steps=5000,
                patience=5000, dtype="float32")
        j["cfg"]["net"].update(d=d, hidden=4 * d)
        j["cfg"]["steps"] = 5000                                              # past the heavy notches' 2,000 cap
        j["tier"] = f"A d{d} lr{lr}"
        jobs.append(j)
    return jobs


def softcert():
    """The loose notches judged on their soft path too (the network as trained, softmax attention, unsnapped states):
    does a soft fit of the training questions carry to the bigger sizes? (3, 2, 1), 4 seeds, full batch."""
    jobs = []
    plan = [("N3_free_mlp", ("A",)), ("N4_symbols", ("A", "G")), ("N4_symbols_untied", ("A",)), ("N5_free", ("A", "G"))]
    for (notch, tiers), seed in itertools.product(plan, SEEDS):
        for tier in tiers:
            jobs.append(job(f"{notch}.{tier}.321.s{seed}", notch, tier, 3, 2, 1, seed, batch=0, cert_soft=True,
                            patience=2000))
    for seed in SEEDS:                                                         # notch 6 at the plain suite's best
        j = job(f"N6_plain.d32.lr0.001.321.s{seed}", "N6_plain", "A", 3, 2, 1, seed, lr=0.001, batch=0,
                patience=5000, dtype="float32", cert_soft=True)
        j["cfg"]["steps"] = 5000
        jobs.append(j)
    return jobs


STAB = [("N1_mlp", "GCL"), ("N2_menu_given", "AL"), ("N3_free_given", "AL"), ("N3_free_mlp", "ACL"),
        ("N4_symbols", "GL")]


def widened(notch, wide):
    """The notch's net config and Net2Net factor, minimal or widened (notches.WIDE)."""
    net = dict(Nt.NOTCHES[notch])
    if not wide:
        return net, 1
    extra, widen = Nt.WIDE[notch]
    net.update(extra)
    return net, widen


def wstab():
    """Stability (the Net2Net arm): start at the compiled machine, softened to scale 4 and perturbed by Gaussian
    noise sigma, minimal or widened, and train with the notch's full lessons and with the answers alone. Does
    training keep (or recover) the solution, or drift to an impostor? (3, 2, 1), 4 seeds, 1,000 steps."""
    jobs = []
    for (notch, tier), wide, sigma, seed in itertools.product(STAB, (0, 1), (0.1, 0.3, 1.0, 3.0), SEEDS):
        for t in (tier, "A"):
            net, widen = widened(notch, wide)
            j = job(f"{notch}.{'wide' if wide else 'min'}.{t}.s{sigma}.321.s{seed}", notch, t, 3, 2, 1, seed,
                    init="compiled", init_beta=4.0, init_noise=sigma, widen=widen, steps=1000, patience=1000,
                    batch=0)
            j["cfg"]["net"] = net
            j["cfg"]["steps"] = 1000
            j["tier"] = f"{t} {'wide' if wide else 'min'} sigma {sigma}"
            jobs.append(j)
    return jobs


def wlearn():
    """Learnability (the Net2Net arm): the same notches minimal and widened, trained from random with the same
    lessons. (3, 2, 1) for all, (3, 2, 2) for the table and menu notches; 4 seeds."""
    jobs = []
    plan = [("N1_mlp", "GCL", ((3, 2, 1), (3, 2, 2))), ("N2_rel13_given", "AL", ((3, 2, 1), (3, 2, 2))),
            ("N3_free_given", "AL", ((3, 2, 1), (3, 2, 2))), ("N3_free_mlp", "ACL+f", ((3, 2, 1),)),
            ("N4_symbols", "GL", ((3, 2, 1),))]
    for (notch, tier, sizes), wide, seed in itertools.product(plan, (0, 1), SEEDS):
        for (B, w, k) in sizes:
            base, _, flag = tier.partition("+")
            kw = {"route_freeze": 300} if flag == "f" else {}
            net, _ = widened(notch, wide)
            j = job(f"{notch}.{'wide' if wide else 'min'}.{tier}.{B}{w}{k}.s{seed}", notch, base, B, w, k, seed,
                    batch=0, **kw)
            j["cfg"]["net"] = net
            j["tier"] = f"{tier} {'wide' if wide else 'min'}"
            jobs.append(j)
    return jobs


def bypass():
    """The free-attention bypass: the prompt only at the first tick (TRM-style, `inject` "first"), snapped states
    with straight-through gradients (`st`), both, and the first-tick inputs with four head copies; the tables given
    or the subtraction table learned. (3, 2, 1) and (3, 2, 2), 4 seeds."""
    jobs = []
    plan = [("N3_free_given_first", ("A", "AL"), {}), ("N3_free_given_st", ("AL",), {}),
            ("N3_free_given_first_st", ("AL",), {}), ("N3_free_first", ("ACL+f",), {}),
            ("N3_free_given_first", ("AL",), {"copies": 4})]
    for (notch, tiers, extra), (B, w, k), seed in itertools.product(plan, SIZES, SEEDS):
        for tier in tiers:
            base, _, flag = tier.partition("+")
            kw = {"route_freeze": 300} if flag == "f" else {}
            tag = "c4" if extra else ""
            j = job(f"{notch}{'.' + tag if tag else ''}.{tier}.{B}{w}{k}.s{seed}", notch, base, B, w, k, seed, **kw)
            j["cfg"]["net"].update(extra)
            j["tier"] = tier + (" " + tag if tag else "")
            jobs.append(j)
    return jobs


SUITES = {"ladder": ladder, "mlpcell": mlpcell, "plain": plain, "softcert": softcert, "wstab": wstab,
          "wlearn": wlearn, "bypass": bypass}


def execute(j: dict) -> dict:
    import time

    from bridge import train as T
    t0 = time.time()
    rec = T.run(T.RunCfg(**j["cfg"]))
    rec.update(name=j["name"], notch=j["notch"], tier=j["tier"], size=j["size"], wall=time.time() - t0)
    return rec
