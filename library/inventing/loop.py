"""The inventing learner's loop on the bare machine (INVENTING.md): search, compress, search bigger.

    ~/github/sotaku/venv/bin/python -m library.inventing.loop --config C_tiny      # a seconds-long local check

Arms (CONFIGS):
- A: plain search for the target at one size, no compression.
- C: the target only, at growing sizes k = 2, 3, ... (every pair with N <= k must come out right); after each size
  the solved programs are compressed (MDL at half price, or BPE) and the next size is searched with the new library.
- D: C plus practice the learner makes itself, DreamCoder-style. Before each size's search it samples random
  programs from its current library, runs them on the target's inputs at that size, keeps those that compute
  something new and non-trivial, and solves them as tasks. Their programs join the corpus that is compressed.
A run stops when a target program is right on every pair with N <= 40 (general), or when the sizes run out.

The cross arms, to tell which of our inputs matters (INVENTING.md §3, §6):
- Bbare: our curated practice (target.CURATED) on the bare machine; with routines (rt), every solved task's program
  also becomes one token T.<task>, as in the library study's final run.
- Cdig / Ddig: arms C and D on the library study's digit instructions: the target at growing digit counts
  (DIGIT_SIZES), and in Ddig dreams on the target's inputs at those settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from library import dsl, merge
from library.inventing import brm, target
from library.inventing.digit_jobs import run_on, target_inputs
from library.inventing.jobs import machine_for, search_job
from library.wake_sleep import promote

HERE = Path(__file__).resolve().parent

SMALL = dict(max_cost=10, max_states=3_000_000, time_limit=2400, max_body=2, limit=200)
BIG = dict(SMALL, max_states=16_000_000, time_limit=5400)
S1 = list(range(2, 8))                       # up to N = 7: base 2, 3 digits
S2 = list(range(2, 8)) + [9, 12, 16, 20, 26]  # on to N = 26: base 3, 3 digits
DIGIT = dict(max_cost=7, max_states=3_000_000, time_limit=2400, NP=8, max_body=2, limit=200)   # the library study's
DIGIT_SIZES = [[(2, 1), (3, 1)], [(2, 2), (3, 2)], [(2, 3), (3, 3)]]                             # 1, 2, 3 digits
MI = {"MI": "INC.A DEC.T Z.T[CLR.A MOV.T.N]"}    # the modular increment, given by hand (A_oracle)

CONFIGS = {
    "C_tiny": dict(arm="C", merge="mdl", sizes=[2, 3, 4], budget=dict(SMALL, max_states=60_000, time_limit=20)),
    "A_s1": dict(arm="A", sizes=[7], budget=SMALL),
    "A_s1_big": dict(arm="A", sizes=[7], budget=BIG, big=True),
    "A_s2_big": dict(arm="A", sizes=[26], budget=BIG, big=True),
    "C_mdl": dict(arm="C", merge="mdl", sizes=S1, budget=SMALL),
    "C_bpe": dict(arm="C", merge="bpe", sizes=S1, budget=SMALL),
    "D_mdl": dict(arm="D", merge="mdl", sizes=S1, budget=SMALL),
    "D_bpe": dict(arm="D", merge="bpe", sizes=S1, budget=SMALL),
    "Bbare_mdl": dict(arm="Bbare", merge="mdl", k=7, budget=SMALL),
    "Bbare_bpe": dict(arm="Bbare", merge="bpe", k=7, budget=SMALL),
    "Bbare_mdl_rt": dict(arm="Bbare", merge="mdl", k=7, budget=SMALL, promote=True),
    "Bbare_bpe_rt": dict(arm="Bbare", merge="bpe", k=7, budget=SMALL, promote=True),
    "Cdig_mdl": dict(arm="C", machine="digit", merge="mdl", sizes=DIGIT_SIZES, budget=DIGIT),
    "Cdig_bpe": dict(arm="C", machine="digit", merge="bpe", sizes=DIGIT_SIZES, budget=DIGIT),
    "Ddig_mdl": dict(arm="D", machine="digit", merge="mdl", sizes=DIGIT_SIZES, budget=DIGIT),
    "Ddig_bpe": dict(arm="D", machine="digit", merge="bpe", sizes=DIGIT_SIZES, budget=DIGIT),
    "Cdig_tiny": dict(arm="C", machine="digit", merge="mdl", sizes=DIGIT_SIZES[:2],
                      budget=dict(DIGIT, max_cost=4, max_states=40_000, time_limit=20)),
    "Ddig_tiny": dict(arm="D", machine="digit", merge="mdl", sizes=DIGIT_SIZES[:2], dreams=4,
                      budget=dict(DIGIT, max_cost=4, max_states=40_000, time_limit=20)),
    "A_oracle": dict(arm="A", sizes=[7], lib=MI, budget=dict(SMALL, max_cost=5, max_states=300_000, time_limit=60)),
    "C_oracle": dict(arm="C", merge="mdl", sizes=S1, lib=MI,
                     budget=dict(SMALL, max_cost=6, max_states=300_000, time_limit=60)),
}


def sample_program(tokens, kinds, lib, rng, max_cost=6):
    """A random program from the current library: 1-4 items; an item is a token (60%) or a block over 1-2 tokens."""
    items, cost = [], 0
    for _ in range(int(rng.integers(1, 5))):
        if rng.random() < 0.6:
            it = tokens[int(rng.integers(len(tokens)))]
        else:
            kind = kinds[int(rng.integers(len(kinds)))]
            n = 1 if kind in lib else int(rng.integers(1, 3))
            it = ("REP", kind, tuple(tokens[int(rng.integers(len(tokens)))] for _ in range(n)))
        c = dsl.cost((it,))
        if cost + c > max_cost:
            break
        items.append(it)
        cost += c
    return tuple(items)


def make_dreams(lib, kinds, k, rng, known, n_want=16, tries=6000, max_cost=6):
    """Practice tasks the learner makes itself: random programs' behaviour on the target's inputs at size k."""
    tokens = list(brm.PRIMS) + [t for t in lib if t not in kinds]
    xs, Ns = target.pairs(k)
    m = machine_for(k)
    out = []
    for _ in range(tries):
        prog = sample_program(tokens, kinds, lib, rng, max_cost)
        if not prog:
            continue
        got, dead = brm.evaluate(prog, lib, xs, Ns, m)
        if dead or len(set(got.tolist())) < 2 or (got == xs).all() or (got == Ns).all():
            continue
        key = tuple(got.tolist())
        if key in known:
            continue
        known.add(key)
        out.append((prog, got))
        if len(out) == n_want:
            break
    return out


def make_digit_dreams(lib, kinds, settings, rng, known, n_want=16, tries=6000, max_cost=6, n_pose=8, n_check=64,
                      seed=0):
    """Dreams on the digit machine: random programs' outputs on the target's inputs at these settings, kept when they
    depend on the inputs alone (the same from two random starts), vary, are not X or N, and are new."""
    tokens = list(dsl.PRIMS) + [t for t in lib if t not in kinds]
    pose = [target_inputs(st, n_pose, seed)[0] for st in settings]
    check = [target_inputs(st, n_check, seed + 1)[0] for st in settings]
    out = []
    for _ in range(tries):
        prog = sample_program(tokens, kinds, lib, rng, max_cost)
        if not prog:
            continue
        runs = [[run_on(prog, lib, st, ins, s) for s in (1, 2)] for st, ins in zip(settings * 2, pose + check)]
        if any((a != b).any() for a, b in runs):
            continue
        got = [a for a, _ in runs]
        vals = np.concatenate([dsl.from_digits(g, st.B).astype(np.int64) for g, st in zip(got, settings * 2)])
        xs = np.concatenate([dsl.from_digits(i["X"], st.B).astype(np.int64) for i, st in zip(pose + check, settings * 2)])
        Ns = np.concatenate([dsl.from_digits(i["N"], st.B).astype(np.int64) for i, st in zip(pose + check, settings * 2)])
        key = b"".join(g.tobytes() for g in got)
        if len(set(vals.tolist())) < 2 or (vals == xs).all() or (vals == Ns).all() or key in known:
            continue
        known.add(key)
        out.append(dict(prog=prog, values=vals[:24].tolist(),
                        inputs=[{r: v.tolist() for r, v in i.items()} for i in pose],
                        answers=[g.tolist() for g in got[:len(settings)]],
                        check_inputs=[{r: v.tolist() for r, v in i.items()} for i in check],
                        check_answers=[g.tolist() for g in got[len(settings):]]))
        if len(out) == n_want:
            break
    return out


def digit_jobs(name, cfg, level, settings, lib, kinds, rng, known):
    """The target job at these settings, and in arm D the dream jobs."""
    W = settings[0][1]
    base = dict(machine="digit", k=W, settings=[list(s) for s in settings], lib={t: dsl.show(v) for t, v in lib.items()},
                kinds=list(kinds), **cfg["budget"])
    jobs = [dict(base, name=f"{name}-L{level}-target{W}", task="target", seed=level)]
    dreams = []
    if cfg["arm"] == "D":
        dreams = make_digit_dreams(lib, kinds, [dsl.Setting(B, w) for B, w in settings], rng, known,
                                   n_want=cfg.get("dreams", 16), seed=100 * level)
        for i, d in enumerate(dreams):
            jobs.append(dict(base, name=f"{name}-L{level}-dream{i}", task="dream", max_cost=dsl.cost(d["prog"]),
                             **{f: d[f] for f in ("inputs", "answers", "check_inputs", "check_answers")}))
    return jobs, [dict(program=dsl.show(d["prog"]), outputs=d["values"]) for d in dreams]


def job_for(name, task, k, xs, Ns, want, check, lib, kinds, budget):
    return dict(name=name, task=task, k=k, xs=[int(v) for v in xs], Ns=[int(v) for v in Ns],
                want=[int(v) for v in want], check=[[int(v) for v in c] for c in check],
                lib={t: dsl.show(v) for t, v in lib.items()}, kinds=list(kinds), **budget)


def compress(cfg, corpus, lib, kinds, level, mlog):
    if cfg.get("merge") == "bpe":
        return merge.bpe(corpus, lib, max_merges=cfg.get("max_merges", 12), min_count=cfg.get("min_count", 2.0),
                         prefix=f"B{level}.", log=mlog, tiebreak=cfg.get("tiebreak", "largest"))
    if cfg.get("merge") == "mdl":
        m = dsl if cfg.get("machine") == "digit" else brm
        return merge.mdl(corpus, lib, holes=cfg.get("holes", False), prefix=f"M{level}.", log=mlog, kinds=kinds,
                         lib_weight=cfg.get("lib_weight", 0.5), n_prims=len(m.PRIMS), n_kinds=len(m.LOOP_KINDS))
    return corpus, lib, []


def run(name, cfg, mapper, emit, log=print):
    rng = np.random.default_rng(cfg.get("seed", 0))
    digit = cfg.get("machine") == "digit"
    lib = {t: dsl.parse(v) for t, v in cfg.get("lib", {}).items()}
    kinds, corpus = list(dsl.LOOP_KINDS if digit else brm.LOOP_KINDS), {}
    known = set()                                       # behaviours already posed as tasks
    for level, k in enumerate(cfg["sizes"]):
        if digit:
            jobs, dreams = digit_jobs(name, cfg, level, k, lib, kinds, rng, known)
            k = k[0][1]
        else:
            xs, Ns = target.probes(k, n=cfg.get("np", 32), seed=level)
            cx, cN = target.pairs(k)
            jobs = [job_for(f"{name}-L{level}-target{k}", "target", k, xs, Ns, target.answers(xs, Ns),
                            (cx, cN, target.answers(cx, cN)), lib, kinds, cfg["budget"])]
            known.add(tuple(target.answers(cx, cN).tolist()))
            dreams = []
            if cfg["arm"] == "D":
                for i, (prog, got) in enumerate(make_dreams(lib, kinds, k, rng, known, n_want=cfg.get("dreams", 16))):
                    sub = np.arange(len(cx)) if len(cx) <= cfg.get("np", 32) else np.sort(
                        rng.choice(len(cx), cfg.get("np", 32), replace=False))
                    budget = dict(cfg["budget"], max_cost=dsl.cost(prog))
                    jobs.append(job_for(f"{name}-L{level}-dream{i}", "dream", k, cx[sub], cN[sub], got[sub],
                                        (cx, cN, got), lib, kinds, budget))
                    dreams.append(dict(program=dsl.show(prog), outputs=got.tolist()[:40]))
        recs = mapper(jobs)
        solved, general = [], None
        for job, r in zip(jobs, recs):
            if not isinstance(r, dict):
                log(f"  job failed: {r!r}"[:300])
                continue
            good = [dsl.parse(p["program"]) for p in r.get("programs", []) if p["verified"]]
            if good:
                corpus[job["name"].split("-", 1)[1]] = good        # e.g. "L3-target5", "L3-dream7"
                solved.append(r["task"])
                if r["task"] == "target":
                    best = max((p for p in r["programs"] if p["verified"]), key=lambda p: p.get("general", 0))
                    general = best if best.get("general") == 1.0 else None
            if r["task"] == "target":
                log(f"  L{level} k={k} target cost {r['found_cost']} levels {r['level_sizes']} {r['seconds']}s "
                    f"{r['stop']} | {[p['program'] for p in r.get('programs', []) if p['verified']][:3]}")
        mlog = []
        corpus, lib, new = compress(cfg, corpus, lib, kinds, level, mlog)
        for e in mlog:
            e["expands_to"] = dsl.show(dsl.expand(lib[e["name"]], lib)) if e["name"] in lib else ""
            log(f"    + {e['name']} = {e['def']}    ({e['expands_to']})")
        emit(dict(config=name, level=level, k=k, solved=solved, general=general, records=recs, merges=mlog,
                  dreams=dreams, lib={t: dsl.show(v) for t, v in lib.items()}, kinds=list(kinds),
                  corpus={t: [dsl.show(p) for p in ps[:10]] for t, ps in corpus.items()}))
        if general is not None:
            log(f"  general x^2 mod N at size {k}: {general['program']} = {general['primitive']}")
            break


def run_curated(name, cfg, mapper, emit, log=print):
    """Arm B on the bare machine: the curated tasks (target.CURATED) at one size, level after level, compressing the
    solved programs after each level, until x^2 mod N is solved in general or a level adds nothing."""
    lib, kinds, corpus = {}, list(brm.LOOP_KINDS), {}
    k = cfg["k"]
    for level in range(cfg.get("levels", 8)):
        todo = [t for t in target.CURATED if t not in corpus]
        if not todo:
            break
        jobs = []
        for t in todo:
            xs, Ns, want = target.curated(t, k)
            jobs.append(job_for(f"{name}-L{level}-{t}", "target" if t == "sqmod" else "curated", k, xs, Ns, want,
                                (xs, Ns, want), lib, kinds, cfg["budget"]))
        recs = mapper(jobs)
        solved, general = [], None
        for t, r in zip(todo, recs):
            if not isinstance(r, dict):
                log(f"  job failed: {r!r}"[:300])
                continue
            good = [dsl.parse(p["program"]) for p in r.get("programs", []) if p["verified"]]
            if good:
                corpus[t] = good
                solved.append(t)
                if t == "sqmod":
                    best = max((p for p in r["programs"] if p["verified"]), key=lambda p: p.get("general", 0))
                    general = best if best.get("general") == 1.0 else None
            log(f"  L{level} {t:7s} cost {r['found_cost']} {r['seconds']}s {r['stop']} | "
                f"{[p['program'] for p in r.get('programs', []) if p['verified']][:2]}")
        mlog = []
        corpus, lib, new = compress(cfg, corpus, lib, kinds, level, mlog)
        if cfg.get("promote"):          # every newly solved task's program becomes one routine, T.<task>
            corpus, lib, plog = promote(corpus, lib, solved)
            mlog += plog
            new = list(new) + [e["name"] for e in plog]
        for e in mlog:
            e["expands_to"] = dsl.show(dsl.expand(lib[e["name"]], lib)) if e["name"] in lib else ""
        emit(dict(config=name, level=level, k=k, solved=solved, general=general, records=recs, merges=mlog, dreams=[],
                  lib={t: dsl.show(v) for t, v in lib.items()}, kinds=list(kinds),
                  corpus={t: [dsl.show(p) for p in ps[:10]] for t, ps in corpus.items()}))
        if general is not None or (not solved and not new):
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="C_tiny")
    a = ap.parse_args()
    out = HERE / "records" / f"{a.config}_local.jsonl"
    out.write_text("")

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    cfg = CONFIGS[a.config]
    (run_curated if cfg["arm"] == "Bbare" else run)(a.config, cfg, lambda jobs: [search_job(j) for j in jobs], emit)
    print("wrote", out)


if __name__ == "__main__":
    main()
