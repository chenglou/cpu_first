"""Solve, merge, solve bigger (LIBRARY.md §4): the DreamCoder-style loop with BPE or MDL as the sleep phase.

    ~/github/sotaku/venv/bin/python -m library.wake_sleep --config tiny       # a seconds-long local check

Each level searches every task not yet solved, with the same budget, over the primitives plus every macro so far.
The verified minimal programs of the solved tasks are the corpus. Then the corpus is compressed (BPE pairs, MDL runs,
or MDL with holes), the new macros join the vocabulary, and the next level starts. "none" never merges: it is the
primitive-only search at the same budget. A run stops when a level adds no merge (in-run: and solves nothing).

Options a configuration can set (CONFIGS below):
- budget: the search budget of every job (max_states per level, max_body: the largest loop body built, ...);
- big: run the jobs on the 32 GB containers (and their cache);
- job: extra job fields (settings, prims, learned cells, in-run data, fallback_primitive: search with the
  primitives alone too when the library search fails);
- inrun: tasks wait for the programs their probes need (inrun.NEEDS);
- resume: continue from another run's last level record (Modal driver only);
- promote: every solved task's program also becomes one routine, T.<task> (composition, LIBRARY.md §5.5).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from library import dsl, merge
from library.jobs import search_job
from library.tasks import TASKS, TASKS_R1

HERE = Path(__file__).resolve().parent

BUDGET = dict(max_cost=7, max_states=3_000_000, time_limit=2400, NP=8, max_body=2, max_arity=3, limit=200)

CONFIGS = {
    "tiny": dict(mode="bpe", levels=3, tasks=("copy", "sub", "add", "csub"),
                 budget=dict(BUDGET, max_cost=4, max_states=40_000, time_limit=60)),
    "bpe_r1": dict(mode="bpe", levels=8, tasks=TASKS_R1, budget=BUDGET, weighting="mean", overlapping=False),
    "bpe_r2": dict(mode="bpe", levels=8, tasks=TASKS, budget=BUDGET, overlapping=False),
    "bpe": dict(mode="bpe", levels=8, tasks=TASKS, budget=BUDGET),
    "mdl": dict(mode="mdl", levels=8, tasks=TASKS, budget=BUDGET),
    "mdlh": dict(mode="mdlh", levels=8, tasks=TASKS, budget=BUDGET),
    "none": dict(mode="none", levels=1, tasks=TASKS, budget=BUDGET),
    "mdl_half": dict(mode="mdl", levels=8, tasks=TASKS, budget=BUDGET, lib_weight=0.5),
    "mdlh_half": dict(mode="mdlh", levels=8, tasks=TASKS, budget=BUDGET, lib_weight=0.5),
}
BIG = dict(BUDGET, max_states=16_000_000, time_limit=5400)


def _learned_configs():
    """lc_*: the subtraction half in base 10 with only the subtraction cell, exact or a learned snapshot."""
    from library import learned
    job = dict(prims=learned.SUB_PRIMS, settings=[[10, 1], [10, 2]], verify=[[10, 1], [10, 2], [10, 3], [10, 4]])
    out = {"lc_exact": dict(mode="bpe", levels=8, tasks=learned.SUB_TASKS, budget=BUDGET, job=dict(job))}
    if learned.TABLES.exists():
        for seed, step in ((2, 150), (2, 125), (2, 100), (2, 75), (1, 400)):
            r = learned.snapshot(seed, step=step)
            cells = dict(B=10, sbb_d=r["sbb_d"], sbb_c=r["sbb_c"])
            out[f"lc_s{seed}_{step}"] = dict(mode="bpe", levels=8, tasks=learned.SUB_TASKS, budget=BUDGET,
                                             job=dict(job, cells=cells), table_right=r["right"])
    return out


CONFIGS.update(_learned_configs())


def _inrun_configs():
    """inrun_*: the corpus from the training rows' own numbers and laws (inrun.py), base 10, settings (10, 1) and
    (10, 2), BPE merging any pair a working program uses (min_count 1: the laws give few tasks)."""
    from library import inrun
    if not os.path.isdir(inrun.H1_DIR):
        return {}
    job = dict(inrun=True, settings=[[10, 1], [10, 2]], verify=[[10, 1], [10, 2], [10, 3]], **inrun.job_data())
    return {"inrun_bpe": dict(mode="bpe", levels=10, tasks=inrun.INRUN_TASKS, budget=BIG, big=True, job=job,
                              inrun=True, min_count=1.0, max_merges=6),
            "inrun_bpe_small": dict(mode="bpe", levels=10, tasks=inrun.INRUN_TASKS, budget=BUDGET, job=job,
                                    inrun=True, min_count=1.0, max_merges=6),
            "inrun_mdl_half": dict(mode="mdl", levels=10, tasks=inrun.INRUN_TASKS, budget=BIG, big=True, job=job,
                                   inrun=True, lib_weight=0.5)}


CONFIGS.update(_inrun_configs())

for _m in ("bpe", "mdl", "mdlh", "none", "mdl_half", "mdlh_half"):
    CONFIGS[_m + "_big"] = dict(CONFIGS[_m], budget=BIG, big=True)

# loop bodies up to cost 3 (the multiply loop's body is a row and two shifts): bpe_big continued from its last level,
# and the in-run run from the start
CONFIGS["bpe_big_mb3"] = dict(CONFIGS["bpe_big"], budget=dict(BIG, max_body=3), levels=12, resume="bpe_big_r5")
# solved tasks become routines (composition reuses whole programs, which BPE's two-task threshold never merges)
# (bodies back to cost 2: what is left after the multiply is straight-line composition, and ~80 tokens with cost-3
# bodies would be millions of loop items)
CONFIGS["bpe_big_routines"] = dict(CONFIGS["bpe_big"], resume="bpe_big_mb3_r8", promote=True, levels=14)
if "inrun_bpe" in CONFIGS:
    CONFIGS["inrun_bpe_mb3"] = dict(CONFIGS["inrun_bpe"], budget=dict(BIG, max_body=3), levels=14)
    # ... and searching with the primitives alone whenever the library search fails
    CONFIGS["inrun_bpe_fb"] = dict(CONFIGS["inrun_bpe_mb3"],
                                   job=dict(CONFIGS["inrun_bpe"]["job"], fallback_primitive=True))
    # ... continued with solved tasks as routines (composition, §5.5)
    CONFIGS["inrun_routines"] = dict(CONFIGS["inrun_bpe_fb"], resume="inrun_bpe_fb_r12", promote=True, levels=16)
    # ... and with BPE's ties (every pair weighs 1 in-run) broken toward the pair standing for the most primitives
    CONFIGS["inrun_largest"] = dict(CONFIGS["inrun_routines"], tiebreak="largest")
    # the cleanest in-run settings: mac added (the row then has two uses), one simplest program per task in the corpus,
    # ties toward the largest pair, 12 merges a level, routines, from r12's level 3
    CONFIGS["inrun_v2"] = dict(CONFIGS["inrun_routines"], tiebreak="largest", max_merges=12, one_per_task=True,
                               merge_first=True)


def simplest(progs, lib):
    """Of a task's accepted programs, the one with the shortest expansion into primitives (Occam's choice: on its
    laws' inputs they all agree, and the shortest is the likeliest to be the general function)."""
    return min(progs, key=lambda p: dsl.cost(dsl.expand(p, lib)))


def promote(corpus, lib, tasks):
    """Each task's simplest program (cost >= 2) becomes a routine T.<task>; every program holding it is rewritten."""
    lib, log = dict(lib), []
    for t in tasks:
        prog = simplest(corpus[t], lib)
        name = f"T.{t}"
        if dsl.cost(prog) < 2 or name in lib:
            continue
        lib[name] = prog
        fn = merge._apply_pattern(prog, name, False)
        corpus = {u: [merge.rewrite(p, fn) for p in ps] for u, ps in corpus.items()}
        log.append({"name": name, "def": dsl.show(prog), "routine": True})
    return corpus, lib, log


def resume_state(rec):
    """The state after a level record: corpus, library, loop kinds, where each task was solved."""
    corpus = {t: [dsl.parse(p) for p in ps] for t, ps in rec["corpus"].items()}
    lib = {k: dsl.parse(v) for k, v in rec["lib"].items()}
    return rec["level"] + 1, corpus, lib, list(rec["kinds"]), dict(rec["solved_at"])


def run(name, cfg, mapper, emit, log=print, start=None):
    """start: a level record to continue from (resume_state); its levels are not redone."""
    lib, kinds = {}, list(dsl.LOOP_KINDS)
    corpus, solved_at, first = {}, {}, 0
    if start is not None:
        first, corpus, lib, kinds, solved_at = resume_state(start)
        plog = []
        if cfg.get("one_per_task"):
            corpus = {t: [simplest(ps, lib)] for t, ps in corpus.items()}
        if cfg.get("merge_first") and cfg["mode"] == "bpe":   # this run's sleep phase on the loaded corpus first
            corpus, lib, _ = merge.bpe(corpus, lib, max_merges=cfg.get("max_merges", 12),
                                       min_count=cfg.get("min_count", 2.0), prefix=f"B{first - 1}r.", log=plog,
                                       weighting=cfg.get("weighting", "max"), overlapping=cfg.get("overlapping", True),
                                       tiebreak=cfg.get("tiebreak", "spelling"))
        if cfg.get("promote"):                  # the loaded state's solved tasks become routines before searching
            corpus, lib, plog2 = promote(corpus, lib, list(corpus))
            plog += plog2
        if plog:
            for m in plog:
                m["expands_to"] = dsl.show(dsl.expand(lib[m["name"]], lib))
            emit(dict(config=name, level=first - 1, solved=[], solved_at=dict(solved_at), seconds=0, records=[],
                      merges=plog, lib={k: dsl.show(v) for k, v in lib.items()}, kinds=list(kinds),
                      corpus={t: [dsl.show(p) for p in ps[:20]] for t, ps in corpus.items()},
                      dl_before=0, dl_after=round(merge.description_length(corpus, lib), 1), promoted_on_resume=True))
    for level in range(first, cfg["levels"]):
        todo = [t for t in cfg["tasks"] if t not in corpus]
        if cfg.get("inrun"):    # a task waits until the programs that build its probes are found
            from library.inrun import NEEDS
            todo = [t for t in todo if all(n in corpus for n in NEEDS[t])]
        if not todo:
            break
        lib_txt = {k: dsl.show(v) for k, v in lib.items()}
        jobs = [dict(name=f"{name}-L{level}-{t}", task=t, lib=lib_txt, kinds=list(kinds), **cfg["budget"],
                     **cfg.get("job", {})) for t in todo]
        if cfg.get("inrun"):
            for jb in jobs:
                jb["helpers"] = {n: dsl.show(simplest(corpus[n], lib)) for n in NEEDS[jb["task"]]}
        t0 = time.time()
        recs = mapper(jobs)
        solved = []
        for r in recs:
            if isinstance(r, Exception) or not isinstance(r, dict):
                log(f"  job failed: {r!r}"[:300])
                continue
            good = [dsl.parse(p["program"]) for p in r.get("programs", []) if p["verified"]]
            if good:
                corpus[r["task"]] = good
                solved_at[r["task"]] = level
                solved.append(r["task"])
            log(f"  L{level} {r['task']:7s} cost {r['found_cost']} verified {r.get('n_verified')}/"
                f"{r.get('n_programs')} levels {r['level_sizes']} {r['seconds']}s {r['stop']}")
        before = merge.description_length(corpus, lib)
        mlog, new = [], []
        if cfg["mode"] == "bpe":
            corpus, lib, new = merge.bpe(corpus, lib, max_merges=cfg.get("max_merges", 12),
                                         min_count=cfg.get("min_count", 2.0), prefix=f"B{level}.", log=mlog,
                                         weighting=cfg.get("weighting", "max"),
                                         overlapping=cfg.get("overlapping", True),
                                         tiebreak=cfg.get("tiebreak", "spelling"))
        elif cfg["mode"] in ("mdl", "mdlh"):
            corpus, lib, new = merge.mdl(corpus, lib, holes=cfg["mode"] == "mdlh", prefix=f"M{level}.", log=mlog,
                                         kinds=kinds, lib_weight=cfg.get("lib_weight", 1.0))
        if cfg.get("one_per_task"):             # Occam: a task's corpus is its simplest accepted program only
            corpus = {t: [simplest(ps, lib)] for t, ps in corpus.items()}
        if cfg.get("promote"):                  # every newly solved task's program becomes one routine
            corpus, lib, plog = promote(corpus, lib, solved)
            mlog += plog
            new = new + [m["name"] for m in plog]
        for m in mlog:
            m["expands_to"] = dsl.show(dsl.expand(lib[m["name"]], lib)) if m["name"] in lib else ""
        rec = dict(config=name, level=level, solved=solved, solved_at=dict(solved_at), seconds=round(time.time() - t0, 1),
                   records=[r if isinstance(r, dict) else repr(r) for r in recs], merges=mlog,
                   lib={k: dsl.show(v) for k, v in lib.items()}, kinds=list(kinds),
                   corpus={t: [dsl.show(p) for p in ps[:20]] for t, ps in corpus.items()},
                   dl_before=round(before, 1), dl_after=round(merge.description_length(corpus, lib), 1))
        emit(rec)
        for m in mlog:
            log(f"    + {m['name']} = {m['def']}    ({m['expands_to']})")
        # an unchanged vocabulary would repeat the same searches; in-run, a newly solved task can still unlock
        # the tasks whose probes it builds
        if not new and not (solved and cfg.get("inrun")):
            break
    return corpus, lib, solved_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    a = ap.parse_args()
    cfg = CONFIGS[a.config]
    out = HERE / "records" / f"wake_sleep_{a.config}_local.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("")

    def emit(rec):
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")

    run(a.config, cfg, lambda jobs: [search_job(j) for j in jobs], emit)
    print("wrote", out)


if __name__ == "__main__":
    main()
