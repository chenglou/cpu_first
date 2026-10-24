"""Local plumbing check of every ablation arm (integration/ABLATION_S3.md): loads each arm as the runner would (a
variant file, or s3i with its --set), checks the one thing the arm changes (its task list, its search vocabulary, its
round vocabulary, its answer filter, its constants), and runs a few climb units with toy budgets on made-up rows with
exact cells, so every changed code path runs once. Seconds and well under 1 GB; the real runs are on Modal (plans/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/check_variants.py [ARM ...]
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "wider_menus", "local")]
import torch
import local_wide as LW

PLAN_SETS = json.load(open(os.path.join(HERE, "plans", "arms.json")))      # arm -> {"file": ..., "set": {...}}
TOY = dict(budget=20_000, fallback_budget=20_000, max_body=1, max_cost=4, levels=3, round_budget=10_000, round_cost=3,
           search_s=20)
FULL = ["lt", "modx", "add", "modB", "mrow", "mac", "mul", "step", "reduce"]


def rows(n, gen):
    """n made-up (N, x) rows as digit strings LSB first, 8 places: N of 2-8 digits, x < N."""
    Nv = torch.randint(10, 10 ** 8, (n,), generator=gen)
    xv = (torch.rand(n, generator=gen) * Nv).long()
    dig = lambda v: torch.stack([(v // 10 ** j) % 10 for j in range(8)], 1)
    return dig(Nv), dig(xv)


def check(arm):
    spec = PLAN_SETS[arm]
    sets = [f"{k}={v}" for k, v in spec.get("set", {}).items()]
    sc = LW.load_module(os.path.join(os.path.dirname(INTEG), spec["file"]), sets)
    cells = sc.exact_cells("cpu")
    climb = sc.Climb(cells, "cpu", sc.PR, chunk=1 << 13, **dict(sc.CLIMB, **TOY))
    tasks = climb.tasks
    vocab = [it.rep for it in climb.classes()]
    notes = [f"tasks {len(tasks)}: {' '.join(tasks)}"]
    # the arm's one change, as the climb sees it
    if arm == "no_twins":
        assert tasks == FULL, tasks
    elif arm in ("no_redstep", "no_step", "no_modB", "no_mrow"):
        gone = {"no_redstep": {"modB", "step"}, "no_step": {"step"}, "no_modB": {"modB"}, "no_mrow": {"mrow"}}[arm]
        assert not gone & {t.rstrip("~") for t in tasks}, tasks
        notes.append(f"needs {dict(sc.NEEDS)}")
    elif arm in ("no_brg", "no_sel"):
        gone = {"no_brg": "BRG.X", "no_sel": "SEL.S"}[arm]
        assert gone in sc.PRIMS and gone not in vocab and gone not in sc.VOCAB, gone
        notes.append(f"{gone} in the machine, not in the {len(vocab)}-item vocabulary")
    else:
        assert len(tasks) == 2 * len(FULL), tasks
    for k, v in spec.get("set", {}).items():
        notes.append(f"{k} = {getattr(sc, k)!r}")
    if arm == "prereq_off":
        assert sc.NEEDS["mul"] == ("add",) and sc.NEEDS["reduce"] == ("add",), sc.NEEDS
    if arm == "mac_n":
        assert sc.MAC_ADDEND == "N"
    # a few climb units on toy budgets (every changed path: tasks, vocabularies, sleep, the round)
    gen = torch.Generator().manual_seed(0)
    Nd, Xd = rows(512, gen)
    ans = (Nd[:64], Xd[:64], Xd[:64])                       # a made-up answer pool (no program reproduces it)
    lines = []
    t0 = time.time()
    for _ in range(40):
        line = climb.step((Nd, Xd), answers=ans)
        if line is None:
            break
        lines.append(line)
    if arm == "no_routines":
        climb.lib["B9.1"] = ("SBB.N", "INC")                # a merge: with routines off the round composes it
        climb.round_tried, climb.done, climb.queue = None, False, None
        line = climb.step((Nd, Xd), answers=ans)
        assert line.startswith("L") and " round:" in line and not any(k.startswith("T.") for k in climb.lib), line
        assert climb.records[-1]["routines"] == len(climb.lib), climb.records[-1]
        lines.append(line)
    if arm in ("no_modB", "no_mrow"):                       # the edited target equals s3i's on the same probes
        task, helpers = {"no_modB": ("step", {"modx": "LDA.X STA.S CLC LDA.X P[SBB.N INC] SEL.S"}),
                         "no_mrow": ("mac", {"add": "CLC LDA.N P[ADC.X INC]"})}[arm]
        ref = LW.load_module(os.path.join(INTEG, "submission_stage3_s3i.py"))
        ref_helpers = dict(helpers, modB="B[LDA.X STA.S CLC LDA.X P[SBB.N INC] SEL.S STA.X]",
                           mrow="LDH.Y CLK CLA P[MUL.X INC]")
        for mod, hs in ((sc, helpers), (ref, ref_helpers)):
            st = mod.Setting(10, 2, mod.exact_cells("cpu"))
            ins, tgt, _ = mod.build(task, st, 64, mod.gen_on("cpu", 3), (Nd, Xd),
                                    {k: mod.parse(v) for k, v in hs.items()}, {}, "cpu")
            if mod is sc:
                mine = (ins, tgt)
        assert all(torch.equal(mine[0][k], ins[k]) for k in ins) and torch.equal(mine[1]["A"], tgt["A"]), arm
        notes.append(f"{task}'s 64 probes and targets equal s3i's (helpers {sorted(ref_helpers)} there)")
    if arm == "ans_smallest_t":
        import inspect
        src = inspect.getsource(sc.LibraryMachine.keep_answers)
        assert "!= 1" not in src, src
    notes.append(f"{len(lines)} climb units in {time.time() - t0:.1f}s; last: {lines[-1][:110] if lines else '-'}")
    return notes


def main(argv):
    arms = argv or list(PLAN_SETS)
    bad = 0
    for arm in arms:
        try:
            notes = check(arm)
            print(f"ok   {arm}\n       " + "\n       ".join(notes), flush=True)
        except Exception as exc:
            bad += 1
            print(f"FAIL {arm}: {exc!r}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
