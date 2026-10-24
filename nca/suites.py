"""Job lists for modal_nca.py: each job is a RunCfg (train.py) plus a name.

Suites:
  explore   the first map of the landscape at (3, 2, 1): formats, lessons, states, staging; 2 seeds
"""

from __future__ import annotations

import json

from nca import train as T

GCL = {"ans": 1.0, "xmodn": 1.0, "link": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0}
GL = {"ans": 1.0, "xmodn": 1.0, "link": 1.0}
C = {"pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0}


def run_record(job: dict, path: str, threads: int = 4) -> dict:
    """Run a job and write its record to `path` (through a temporary file, so that the record is whole or absent);
    for nca/modal_nca.py's containers. Returns a summary."""
    import os
    import time
    import torch
    torch.set_num_threads(threads)
    t = time.time()
    r = execute(job)
    r["wall"] = time.time() - t
    r["suite"] = job["suite"]
    if torch.cuda.is_available():
        r["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    return {"name": r["name"], "suite": r["suite"], "hit": r.get("hit"), "certified": r.get("certified")}


def execute(job: dict) -> dict:
    job = dict(job)
    name = job.pop("name")
    job.pop("suite", None)
    kind = job.pop("kind", "train")
    if kind == "train":
        r = T.run(T.RunCfg(**job))
    elif kind == "verify":
        from nca import verify as V
        r = {"checks": V.verify_record(job["record"]), "of": job["record"]["name"]}
    elif kind == "hyper_swaps":
        from nca import hyper as Hy
        r = Hy.port_swaps()
    elif kind == "hyper_pull":
        from nca import grid as G, hyper as Hy, model as M
        import torch
        lay, d = G.Layout(3, 2, 1), G.Data(3, 2, 1)
        out = {}
        for gnorm, mask in (("softmax", "all"), ("softmax", "causal")):
            net = Hy.HyperNCA(G.compiled_fmt(3, minimal=True),
                              M.Cfg(rule="hyper", hidden=128, heads=3, ghidden=64, mask=mask, gnorm=gnorm),
                              seed=job["seed"])
            out[f"{gnorm} {mask}"] = Hy.first_pull(net, lay, d.Nd, d.vd, d.yd, lay.depth)
        r = {"pull": out}
    elif kind == "compile":
        from nca import compile as Cm
        import time
        t = time.time()
        r = {"check": Cm.check_size(job["B"], job["w"], job["k"], job.get("sample", 0), clamp=job.get("clamp", True),
                                    task=job.get("task", "div"), reach=job.get("reach", "3x3")),
             "job": dict(job)}
        r["check"]["seconds"] = round(time.time() - t, 1)
    elif kind == "diag":
        sizes = [tuple(x) for x in job.pop("sizes")]
        r = T.diag(T.RunCfg(**job), sizes)
    else:
        raise ValueError(kind)
    r["name"] = name
    return r


def explore():
    """The first map at (3, 2, 1). Defaults: the compiled format, kinds given, the cell's own state hidden from its
    rule (noself), soft states, Adam lr 0.003, 128 hidden units, cell lessons on a blank context."""
    jobs = []
    base = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank")
    net = dict(hidden=128, hood="noself")
    ste = dict(net, state="ste")
    s3 = dict(net, hood="3x3")
    tf = {"ans": 300, "xmodn": 300, "link": 300}
    logic = dict(rule="logic", hood="noself", layers=(256, 256, 256, 256), group=8)
    arms = {
        "A": dict(lessons={"ans": 1.0}),
        "A_ste": dict(lessons={"ans": 1.0}, net=ste),
        "A_3x3": dict(lessons={"ans": 1.0}, net=s3),
        "GL": dict(lessons=GL),
        "GL_indep": dict(lessons=dict(GL, indep=1.0)),
        "GCL": dict(lessons=GCL),
        "GCL_ste": dict(lessons=GCL, net=ste),
        "GCL_tf": dict(lessons=GCL, lesson_after=tf),
        "GCL_tf_ste": dict(lessons=GCL, lesson_after=tf, net=ste),
        "GCL_link1st": dict(lessons=GCL, lesson_after={"ans": 600, "xmodn": 600}),
        "GCL_lr01": dict(lessons=GCL, lr=0.01),
        "GCL_ctxrand": dict(lessons=GCL, ctx_mode="random"),
        "GCL_3x3": dict(lessons=GCL, net=s3),
        "GCL_scr": dict(lessons=GCL, scratch="random"),
        "trace": dict(lessons={"trace": 1.0}),
        "trace_3x3": dict(lessons={"trace": 1.0}, net=s3),
        "free_A": dict(lessons={"ans": 1.0}, fmt="free:2,3,2"),
        "free_GL": dict(lessons=GL, fmt="free:2,3,2"),
        "free_GL_ste": dict(lessons=GL, fmt="free:2,3,2", net=ste),
        "free_GL_wide": dict(lessons=GL, fmt="free:3,3,3,3,3,3"),
        "free_GL_indep": dict(lessons=dict(GL, indep=1.0), fmt="free:2,3,2"),
        "logic_A": dict(lessons={"ans": 1.0}, net=logic, lr=0.03),
        "logic_GL": dict(lessons=GL, net=logic, lr=0.03),
        "logic_GCL": dict(lessons=GCL, net=logic, lr=0.03),
        "logic_GCL_tf": dict(lessons=GCL, net=logic, lr=0.03, lesson_after=tf),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            j = dict(base, **arm, seed=seed, tag=name)
            j.setdefault("net", net)
            jobs.append(dict(j, name=f"{name}_s{seed}"))
    return jobs


def ladder2():
    """The second map: where every local situation occurs (k = 2), the locality ladder (3 x 3, + N's column, + the
    bridge's ports with digit-and-borrow cells), worksheet labels, a rule split per channel; 2 seeds."""
    jobs = []
    net = dict(hidden=128, hood="noself")
    ste = dict(net, state="ste")
    ncol = dict(net, extra="ncol")
    ports = dict(net, extra="ports")
    ports_ste = dict(ports, state="ste")
    split = dict(net, split=True)
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank")
    k2 = dict(B=3, w=2, k=2, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", cert_sample=5000,
              extra_sizes=((0, 2), (1, 1)))
    arms = {
        "k2_trace": dict(k2, lessons={"trace": 1.0}, net=net),
        "k2_A": dict(k2, lessons={"ans": 1.0}, net=net),
        "k2_GCL": dict(k2, lessons=GCL, net=net),
        "k2_GCL_ste": dict(k2, lessons=GCL, net=ste),
        "k2_sheet": dict(k2, lessons={"sheet": 1.0, "ans": 1.0}, net=net),
        "k1_sheet": dict(k1, lessons={"sheet": 1.0, "ans": 1.0}, net=net),
        "ncol_GCL": dict(k1, lessons=GCL, net=ncol),
        "ports_A": dict(k1, lessons={"ans": 1.0}, net=ports, fmt="dnb"),
        "ports_GCL": dict(k1, lessons=GCL, net=ports, fmt="dnb"),
        "ports_GCL_ste": dict(k1, lessons=GCL, net=ports_ste, fmt="dnb"),
        "ports_GCL_tf": dict(k1, lessons=GCL, net=ports, fmt="dnb", lesson_after={"ans": 300, "xmodn": 300, "link": 300}),
        "ports_trace": dict(k1, lessons={"trace": 1.0}, net=ports, fmt="dnb"),
        "k2_ports_GCL": dict(k2, lessons=GCL, net=ports, fmt="dnb"),
        "split_GCL": dict(k1, lessons=GCL, net=split),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def square1():
    """The square x^2 mod N at (B, w) = (3, 2) (33 questions), certified at (3, 3) and (3, 4): the compiled
    trace (the upper bound), the answers alone, and the answers with the division taught on its own grid by its
    lessons (the same rule); 2 seeds."""
    jobs = []
    net = dict(hidden=256, hood="noself")
    base = dict(task="sq", B=3, w=2, div_k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank",
                cert_sample=3000)
    dGCL = {"dans": 1.0, "xmodn": 1.0, "link": 1.0, "pref": 1.0, "colall": 1.0, "rowall": 1.0, "bij": 1.0}
    arms = {
        "sq_trace": dict(lessons={"trace": 1.0}),
        "sq_A": dict(lessons={"ans": 1.0}),
        "sq_A_ste": dict(lessons={"ans": 1.0}, net=dict(net, state="ste")),
        "sq_dGCL_A": dict(lessons=dict(dGCL, ans=1.0)),
        "sq_dGCL_A_prod": dict(lessons=dict(dGCL, ans=1.0, prod=1.0)),
        "sq_ports_trace": dict(lessons={"trace": 1.0}, net=dict(net, extra="ports"), fmt="dnb"),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            j = dict(base, **arm, seed=seed, tag=name)
            j.setdefault("net", net)
            jobs.append(dict(j, name=f"{name}_s{seed}"))
    return jobs


def diag1():
    """Where the trace-trained rule breaks one size up: trained at (3, 2, 1) on the compiled trace, then one tick
    from the machine's state at (3, 2, 1), (3, 2, 2) and (3, 3, 2), per row and place."""
    net = dict(hidden=128, hood="noself")
    base = dict(kind="diag", B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, lessons={"trace": 1.0},
                sizes=[(3, 2, 1), (3, 2, 2), (3, 3, 2)], cert_sample=3000)
    return [dict(base, net=net, seed=s, tag="diag_trace", name=f"diag_trace_s{s}") for s in (0, 1)]


def logic2():
    """The bit-level arm with per-tick supervision (the compiled rule's trace), where DiffLogic CA learned the Game
    of Life: two widths, at k = 1 and at k = 2 (every local situation present), and with the bridge's ports;
    2 seeds."""
    jobs = []
    narrow = dict(rule="logic", hood="noself", layers=(256, 256, 256, 256), group=8)
    wide = dict(rule="logic", hood="noself", layers=(1024, 1024, 512, 256), group=16)
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.03, ctx_mode="blank", lessons={"trace": 1.0})
    k2 = dict(k1, k=2, cert_sample=5000, extra_sizes=((0, 2), (1, 1)))
    arms = {
        "logic_trace": dict(k1, net=narrow),
        "logic_trace_wide": dict(k1, net=wide),
        "logic_k2_trace_wide": dict(k2, net=wide),
        "logic_ports_trace_wide": dict(k1, net=dict(wide, extra="ports"), fmt="dnb"),
        "logic_GCL_ste_wide": dict(k1, net=dict(wide, state="ste"), lessons=GCL),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def ladder3():
    """Growing the grid: a causal neighbourhood (no reading downward, so no feedback between rows and no second fixed
    point), and training where every local situation occurs ((3, 3, 2), batches of 256); 2 seeds."""
    jobs = []
    causal = dict(hidden=128, hood="causal")
    noself = dict(hidden=128, hood="noself")
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank")
    k2 = dict(k1, k=2, cert_sample=5000, extra_sizes=((0, 2), (1, 1)))
    w3 = dict(k1, w=3, k=2, batch=256, cert_sample=3000, extra_sizes=((0, 2), (1, 1)))
    arms = {
        "causal_trace": dict(k1, lessons={"trace": 1.0}, net=causal),
        "causal_k2_trace": dict(k2, lessons={"trace": 1.0}, net=causal),
        "causal_A": dict(k1, lessons={"ans": 1.0}, net=causal),
        "causal_GCL": dict(k1, lessons=GCL, net=causal),
        "causal_GCL_ste": dict(k1, lessons=GCL, net=dict(causal, state="ste")),
        "causal_k2_GCL": dict(k2, lessons=GCL, net=causal),
        "w3_trace": dict(w3, lessons={"trace": 1.0}, net=noself),
        "w3_causal_trace": dict(w3, lessons={"trace": 1.0}, net=causal),
        "w3_causal_GCL": dict(w3, lessons=GCL, net=causal),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def free2():
    """The free format (d and N's channel written by the prompt, three hidden channels whose meaning is learned),
    with the relabel probe (do hidden channels take the machine's meanings, or copy each other?): the answers, the
    grid laws, snapped states, and the worksheet's visible digits as labels; 2 seeds."""
    jobs = []
    net = dict(hidden=128, hood="noself")
    causal = dict(hidden=128, hood="causal")
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, fmt="free:2,3,2")
    k2 = dict(k1, k=2, cert_sample=5000, extra_sizes=((0, 2), (1, 1)))
    arms = {
        "free_A": dict(k1, lessons={"ans": 1.0}, net=net),
        "free_GL": dict(k1, lessons=GL, net=net),
        "free_GL_ste": dict(k1, lessons=GL, net=dict(net, state="ste")),
        "free_sheet": dict(k1, lessons={"sheet": 1.0, "ans": 1.0}, net=net),
        "free_sheet_ste": dict(k1, lessons={"sheet": 1.0, "ans": 1.0}, net=dict(net, state="ste")),
        "free_k2_sheet_causal": dict(k2, lessons={"sheet": 1.0, "ans": 1.0}, net=causal),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def sparse1():
    """Does a rule that reads few inputs extrapolate? The compiled trace at (3, 2, 1) with an L1 penalty on the
    rule's input weights, at three strengths, with the noself and the causal neighbourhood; and the lessons with
    it; 2 seeds."""
    jobs = []
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", post=300)
    arms = {}
    for hood in ("noself", "causal"):
        for l1 in (0.3, 1.0, 3.0):
            arms[f"{hood}_trace_l1_{l1}"] = dict(k1, lessons={"trace": 1.0}, l1=l1, net=dict(hidden=128, hood=hood))
    arms["causal_GCL_l1_1.0"] = dict(k1, lessons=GCL, l1=1.0, net=dict(hidden=128, hood="causal"))
    arms["causal_GCL_ste_l1_1.0"] = dict(k1, lessons=GCL, l1=1.0, net=dict(hidden=128, hood="causal", state="ste"))
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def causal2():
    """The causal neighbourhood (no downward reads) carried further: more seeds of the trace at (3, 2, 1), the square
    with the compiled trace, and the bit-level arm with the trace."""
    jobs = []
    causal = dict(hidden=128, hood="causal")
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", post=300)
    sq = dict(task="sq", B=3, w=2, div_k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", cert_sample=3000,
              post=300)
    logic = dict(rule="logic", hood="causal", layers=(1024, 1024, 512, 256), group=16)
    arms = {
        "causal_trace_more": (dict(k1, lessons={"trace": 1.0}, net=causal), (2, 3, 4, 5)),
        "sq_causal_trace": (dict(sq, lessons={"trace": 1.0}, net=dict(hidden=256, hood="causal")), (0, 1)),
        "logic_causal_trace": (dict(k1, lessons={"trace": 1.0}, net=logic, lr=0.03), (0, 1)),
        "logic_causal_k2_trace": (dict(k1, k=2, cert_sample=5000, extra_sizes=((0, 2), (1, 1)), lessons={"trace": 1.0},
                                       net=logic, lr=0.03), (0, 1)),
    }
    for name, (arm, seeds) in arms.items():
        for seed in seeds:
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def logic3():
    """The bit-level arm, given the time difflogic usually needs: the causal neighbourhood, the compiled trace with 6
    sampled ticks per step, 10,000 steps, at two widths; at (3, 2, 1), and at (3, 2, 2) in batches of 64."""
    jobs = []
    narrow = dict(rule="logic", hood="causal", layers=(256, 256, 256, 256), group=8)
    mid = dict(rule="logic", hood="causal", layers=(512, 512, 512, 256), group=16)
    k1 = dict(B=3, w=2, k=1, steps=10000, patience=5000, lr=0.05, lessons={"trace": 1.0}, trace_ticks=6, post=300)
    k2 = dict(k1, k=2, batch=64, cert_sample=5000, extra_sizes=((0, 2), (1, 1)))
    arms = {
        "logic3_k1_narrow": dict(k1, net=narrow),
        "logic3_k1_mid": dict(k1, net=mid),
        "logic3_k2_mid": dict(k2, net=mid),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def partial1():
    """What must the rule be told? The compiled trace at (3, 2, 1), causal neighbourhood, supervising only some
    channels: the digit; the digit and the borrow (the arithmetic); the digit, borrow and N's conveyor; everything
    but the gate; 2 seeds. And the square trained at (3, 3), where every situation of its grid occurs."""
    jobs = []
    causal = dict(hidden=128, hood="causal")
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", post=300, net=causal)
    arms = {f"partial_{c.replace(',', '')}": dict(k1, lessons={"trace": 1.0}, trace_chans=c)
            for c in ("d", "d,b", "d,b,n", "d,b,n,m")}
    arms["sq_w3_causal_trace"] = dict(task="sq", B=3, w=3, div_k=1, steps=3000, patience=1500, lr=0.003, post=300,
                                      batch=64, trace_ticks=12, cert_sample=2000, lessons={"trace": 1.0},
                                      net=dict(hidden=256, hood="causal"))
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def cover1():
    """The certified setting, firmed up: the causal neighbourhood and the compiled trace, trained at (3, 3, 2) where
    every local situation occurs, 4 more seeds, measured further out: (3, 3, 4), (3, 4, 3), (3, 4, 4), (3, 5, 4)."""
    jobs = []
    w3 = dict(B=3, w=3, k=2, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", batch=256, cert_sample=2000,
              extra_sizes=((0, 2), (1, 1), (1, 2), (2, 2)), lessons={"trace": 1.0}, net=dict(hidden=128, hood="causal"))
    for seed in (2, 3, 4, 5):
        jobs.append(dict(w3, seed=seed, tag="w3_causal_trace", name=f"w3_causal_trace_s{seed}"))
    return jobs


def cover2():
    """The certified setting, 4 more seeds, every question of (3, 3, 3) and (3, 4, 3) for the certificate (no
    sampling), the weights kept (`verify.py` checks them further out)."""
    jobs = []
    w3 = dict(B=3, w=3, k=2, steps=3000, patience=1500, lr=0.003, ctx_mode="blank", batch=256, cert_sample=0,
              extra_sizes=((0, 2),), tick_mults=(2,), save_weights=True, lessons={"trace": 1.0},
              net=dict(hidden=128, hood="causal"))
    for seed in (6, 7, 8, 9):
        jobs.append(dict(w3, seed=seed, tag="w3_causal_trace", name=f"w3_causal_trace_s{seed}"))
    return jobs


def selfcopy2():
    """The self-copy trap, recorded: the machine's settled grid as the only target (one tick from it must give it
    back), with the cell's own state visible (3x3) or hidden (noself), at (3, 2, 1); 2 seeds."""
    jobs = []
    k1 = dict(B=3, w=2, k=1, steps=600, patience=1500, lr=0.003, lessons={"fixpoint": 1.0})
    for hood in ("3x3", "noself"):
        for seed in (0, 1):
            jobs.append(dict(k1, net=dict(hidden=128, hood=hood), seed=seed, tag=f"fixpoint_{hood}",
                             name=f"fixpoint_{hood}_s{seed}"))
    return jobs


def cover3():
    """cover2 with 4 more seeds (every question of (3, 3, 3) and (3, 4, 3), weights kept)."""
    return [dict(j, seed=j["seed"] + 4, name=f"w3_causal_trace_s{j['seed'] + 4}") for j in cover2()]


def hyper1():
    """The generated all-to-all wiring (hyper.py), digit-and-borrow cells (dnb), 3 heads, at (3, 2, 1) unless noted:
    the lessons ladder (answers; GCL; snapped; the cell's lessons first with g frozen; one link first; sigmoid
    weights; the causal mask) and the trace (all masks; trained at (3, 3, 2), where the fixed-neighbourhood NCA
    certified); 2 seeds."""
    jobs = []
    hy = dict(rule="hyper", hidden=128, heads=3, ghidden=64)
    k1 = dict(B=3, w=2, k=1, steps=3000, patience=1500, lr=0.003, fmt="dnb", post=300)
    w3 = dict(k1, w=3, k=2, batch=256, cert_sample=3000, extra_sizes=((0, 2), (1, 1), (1, 2)))
    tf = {"ans": 300, "xmodn": 300, "link": 300}
    arms = {
        "hyper_A": dict(k1, lessons={"ans": 1.0}, net=hy),
        "hyper_GCL": dict(k1, lessons=GCL, net=hy),
        "hyper_GCL_ste": dict(k1, lessons=GCL, net=dict(hy, state="ste")),
        "hyper_GCL_tf": dict(k1, lessons=GCL, net=hy, lesson_after=tf, freeze_g=300),
        "hyper_GCL_link1st": dict(k1, lessons=GCL, net=hy, lesson_after={"ans": 600, "xmodn": 600}),
        "hyper_sig_GCL": dict(k1, lessons=GCL, net=dict(hy, gnorm="sigmoid", gbias=-4.0)),
        "hyper_causal_GCL": dict(k1, lessons=GCL, net=dict(hy, mask="causal")),
        "hyper_trace": dict(k1, lessons={"trace": 1.0}, net=hy),
        "hyper_causal_trace": dict(k1, lessons={"trace": 1.0}, net=dict(hy, mask="causal")),
        "hyper_sig_trace": dict(k1, lessons={"trace": 1.0}, net=dict(hy, gnorm="sigmoid", gbias=-4.0)),
        "hyper_w3_trace": dict(w3, lessons={"trace": 1.0}, net=hy),
        "hyper_w3_causal_trace": dict(w3, lessons={"trace": 1.0}, net=dict(hy, mask="causal")),
    }
    for name, arm in arms.items():
        for seed in (0, 1):
            jobs.append(dict(arm, seed=seed, tag=name, name=f"{name}_s{seed}"))
    return jobs


def hyper2():
    """The trace arms of hyper1 again with the head-order-free wiring probe and the weights kept (2 more seeds of each),
    and the causal one trained at (3, 3, 2)."""
    jobs = []
    for j in hyper1():
        if j["tag"] in ("hyper_trace", "hyper_causal_trace", "hyper_w3_causal_trace"):
            seed = j["seed"] + 2
            jobs.append(dict(j, seed=seed, save_weights=True, name=f"{j['tag']}_s{seed}"))
    return jobs


def hyper3():
    """More seeds of the generated wiring's trace at (3, 2, 1), weights kept, to look at the wirings that fit."""
    jobs = []
    for j in hyper1():
        if j["tag"] in ("hyper_trace", "hyper_causal_trace") and j["seed"] == 0:
            for seed in range(4, 12):
                jobs.append(dict(j, seed=seed, save_weights=True, name=f"{j['tag']}_s{seed}"))
    return jobs


def hyper_stab():
    """Does a nearly right generated wiring get partial credit? Start at the compiled machine (scales softened to
    4), add Gaussian noise of std sigma to every parameter of g and the rule, train 1,000 steps on the answers alone
    or on GCL, at (3, 2, 1); 2 seeds (BRIDGE.md §7's stability test)."""
    jobs = []
    hy = dict(rule="hyper", hidden=128, heads=3, ghidden=64)
    for sigma in (0.1, 0.3, 1.0):
        for lname, less in (("A", {"ans": 1.0}), ("GCL", GCL)):
            for seed in (0, 1):
                name = f"hstab_{lname}_s{sigma}"
                jobs.append(dict(B=3, w=2, k=1, steps=1000, patience=1000, post=1000, lr=0.003, fmt="dnb", net=hy,
                                 lessons=less, init="compiled", init_beta=4.0, init_noise=sigma, seed=seed, tag=name,
                                 name=f"{name}_s{seed}"))
    return jobs


def hyper_diag():
    """The one-port census of the compiled generated wiring (hyper.port_swaps) and the answers' first pull at
    random initializations (hyper.first_pull), at (3, 2, 1)."""
    return [{"kind": "hyper_swaps", "name": "hyper_swaps"}] + [
        {"kind": "hyper_pull", "name": f"hyper_pull_s{s}", "seed": s} for s in range(4)]


TRIM_SIZES = {"w3": dict(B=3, w=3, k=2, batch=256, cert_sample=3000, extra_sizes=((0, 2), (1, 1)), gpu=True),
              "w2": dict(B=3, w=2, k=1)}
TRIM_ARMS = {"ctl": {}, "wd0.1": dict(optim="adamw", wd=0.1), "wd0.3": dict(optim="adamw", wd=0.3),
             "wd1": dict(optim="adamw", wd=1.0), "and": dict(agree="and"), "dbd": dict(agree="dbd"),
             "and_wd0.3": dict(agree="and", optim="adamw", wd=0.3), "dbd_wd0.3": dict(agree="dbd", optim="adamw", wd=0.3)}
TRIM = dict(lessons=dict(GCL, step=1.0), net=dict(hidden=128, hood="causal"), lr=0.003, ctx_mode="blank", nostop=True,
            every=100, curve_every=500, curve_sample=500, save_weights=True)


def trim_time():
    """How long a step takes, per size and arm, on the hardware trim1 uses: the log's clock at steps 40 and 60 (no
    curve point between them), and a curve point's cost at step 60. (trim_time, the first launch, also tried (3, 3, 2)
    on CPU: 8 GB was not enough.)"""
    return [dict(TRIM, **size, **TRIM_ARMS[aname], steps=60, every=10, curve_every=30, cert=False, save_weights=False,
                 tag=f"{sname}_{aname}", name=f"{sname}_{aname}")
            for sname, size in TRIM_SIZES.items() for aname in ("ctl", "and", "dbd")]


def trim_time3():
    """The one-pass AND-mask's step on GPU, alone; and how much jobs slow each other when 2 or 3 share a GPU, each in
    its own process (the "pack" key); with checkpoints (curve points at 30 and 60)."""
    one = lambda aname, name, pack="": dict(TRIM, **TRIM_SIZES["w3"], **TRIM_ARMS[aname], steps=60, every=10,
                                            curve_every=30, cert=False, save_weights=False, tag=f"w3_{aname}",
                                            name=name, pack=pack)
    return [one("and", "w3_and_alone"), one("ctl", "w3_ctl_p2", "p2"), one("and", "w3_and_p2", "p2"),
            one("ctl", "w3_ctl_p3", "p3"), one("dbd", "w3_dbd_p3", "p3"), one("wd0.3", "w3_wd0.3_p3", "p3")]


def trim_time4():
    """The AND-mask's step on GPU with the groups as one grouped convolution: alone, and three to a GPU."""
    one = lambda aname, name, pack="": dict(TRIM, **TRIM_SIZES["w3"], **TRIM_ARMS[aname], steps=60, every=10,
                                            curve_every=30, cert=False, save_weights=False, tag=f"w3_{aname}",
                                            name=name, pack=pack)
    return [one("and", "w3_and_alone"), one("and", "w3_and_p3", "p3"), one("and_wd0.3", "w3_and_wd0.3_p3", "p3"),
            one("ctl", "w3_ctl_p3", "p3")]


def trim1(seeds=range(4), pack=3):
    """Trimming and agreement, taught by lessons only (GCL and the step law; nothing from the compiled rule), causal,
    a long budget and no early stop, a curve point every 500 steps: plain Adam (the control); AdamW at three weight
    decays; the AND-mask over 8 groups of questions; delta-bar-delta speeds; each of the two with weight decay 0.3.
    (3, 3, 2) in batches of 256 on GPU, up to `pack` runs to a GPU (each in its own process; one AND-mask run, the
    slowest, per pack), and (3, 2, 1) on every question on CPU; 4 seeds."""
    steps = {"w3": 20000, "w2": 50000}
    jobs = [dict(TRIM, **size, **arm, steps=steps[sname], seed=seed, tag=f"{sname}_{aname}",
                 name=f"{sname}_{aname}_s{seed}")
            for sname, size in TRIM_SIZES.items() for aname, arm in TRIM_ARMS.items() for seed in seeds]
    gpu = [j for j in jobs if j.get("gpu")]
    order = [j for j in gpu if j.get("agree") == "and"] + [j for j in gpu if j.get("agree") != "and"]
    npacks = -(-len(gpu) // pack)
    for i, j in enumerate(order):
        j["pack"] = f"p{i % npacks}"
    return jobs


def trim_cols():
    """The optional variant of trim1's agreement: the AND-mask across cell positions (the grid's 4 columns, each its
    own block, must agree on the sign of a component), alone and with weight decay 0.3; at (3, 2, 1) on CPU, as in
    trim1; 4 seeds."""
    arms = {"and_cols": dict(agree="and_cols"), "and_cols_wd0.3": dict(agree="and_cols", optim="adamw", wd=0.3)}
    return [dict(TRIM, **TRIM_SIZES["w2"], **arm, steps=50000, seed=seed, tag=f"w2_{aname}", name=f"w2_{aname}_s{seed}")
            for aname, arm in arms.items() for seed in range(4)]


def trim_stab():
    """Does each trim arm keep the machine's rule once it has it? trim1's and trim_cols' arms at (3, 2, 1), started at
    the compiled rule (output scale 4: exact at (3, 2, 1) on the snapped and the soft path) instead of at random, the
    same lessons, 10,000 steps; 2 seeds."""
    arms = dict(TRIM_ARMS, and_cols=dict(agree="and_cols"),
                **{"and_cols_wd0.3": dict(agree="and_cols", optim="adamw", wd=0.3)})
    return [dict(TRIM, **TRIM_SIZES["w2"], **arm, init="compiled", init_beta=4.0, steps=10000, seed=seed,
                 tag=f"w2_{aname}", name=f"w2_{aname}_s{seed}")
            for aname, arm in arms.items() for seed in range(2)]


def compile_all():
    """The compile check (compile.py), one job per size: the division at every size of compile.SIZES with the
    prompt re-injected and written only at tick 0; the square at every size of compile.SIZES_SQ; and both at a few
    sizes with the longer reaches ("ncol", "ports")."""
    from nca import compile as Cm
    jobs = []
    for B, w, k, sample in Cm.SIZES:
        for clamp in (True, False):
            jobs.append(dict(kind="compile", B=B, w=w, k=k, sample=sample, clamp=clamp,
                             name=f"div_{B}{w}{k}" + ("" if clamp else "_tick0")))
    for B, w, sample in Cm.SIZES_SQ:
        jobs.append(dict(kind="compile", task="sq", B=B, w=w, k=w, sample=sample, name=f"sq_{B}{w}"))
    for reach in ("ncol", "ports"):
        for B, w, k in ((3, 2, 1), (3, 3, 2), (4, 2, 2), (2, 4, 4)):
            jobs.append(dict(kind="compile", B=B, w=w, k=k, reach=reach, name=f"div_{B}{w}{k}_{reach}"))
        for B, w in ((3, 2), (3, 3), (2, 4)):
            jobs.append(dict(kind="compile", task="sq", B=B, w=w, k=w, reach=reach, name=f"sq_{B}{w}_{reach}"))
    return jobs


SUITES = {"explore": explore, "ladder2": ladder2, "square1": square1, "diag1": diag1, "logic2": logic2,
          "compile": compile_all, "ladder3": ladder3, "free2": free2, "sparse1": sparse1,
          "causal2": causal2, "logic3": logic3,
          "partial1": partial1, "cover1": cover1,
          "cover2": cover2, "selfcopy2": selfcopy2, "cover3": cover3,
          "hyper1": hyper1, "hyper2": hyper2, "hyper3": hyper3, "hyper_stab": hyper_stab, "hyper_diag": hyper_diag, "trim_time": trim_time, "trim_time2": trim_time, "trim_time3": trim_time3, "trim_time4": trim_time4, "trim1": trim1, "trim_cols": trim_cols, "trim_stab": trim_stab, "verify": lambda: __import__("nca.verify", fromlist=["jobs"]).jobs()}
