"""The leave-one-out arms of stage 3 that no constant controls (integration/ABLATION_S3.md): each variant is s3i
(integration/submission_stage3_s3i.py, md5 0e813144) with one hand-made part removed by a few text edits, each asserted
to match exactly once. The arms a constant already controls run s3i itself through the runner's --set (plans/).
Writes variants/<arm>.py and variants/MD5SUMS.json; reads s3i and never writes it.

    ~/github/sotaku/venv/bin/python integration/ablation_s3/make_variants.py           # build, or check a build
    ~/github/sotaku/venv/bin/python integration/ablation_s3/make_variants.py --diff ARM  # one arm's edits as a diff
"""

from __future__ import annotations

import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S3I = HERE.parent / "submission_stage3_s3i.py"
S3I_MD5 = "0e8131440e1842546c67e7378caf472d"
OUT = HERE / "variants"

# new constants go just before this line (the A/B rules' block ends with PREREQ)
AT_CONSTANTS = "ROUND_RETRY = 10       # once the climb is done with no round"
AT_NEEDS = '    NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))\n'
DROP_BLOCK = (AT_NEEDS +
              "for _t in DROP_TASKS:                       # ablation: these practice problems are not posed, and\n"
              "    NEEDS.pop(_t)                           # nothing waits for them\n"
              "for _t in NEEDS:\n"
              "    NEEDS[_t] = tuple(n for n in NEEDS[_t] if n not in DROP_TASKS)\n")


def const(line):
    return (AT_CONSTANTS, line + "\n" + AT_CONSTANTS)


def drop_tasks(tasks, why):
    return [const(f"DROP_TASKS = {tasks!r}  # ablation: practice problems not posed ({why})"), (AT_NEEDS, DROP_BLOCK)]


SEARCH_VOCAB = [
    ('         + ["BRG.X"])\n', '         + ["BRG.X"])\nVOCAB = [t for t in PRIMS if t not in SEARCH_DROP]   # ablation: '
     'the searches\' primitives (the machine keeps them all)\n'),
    ("enumerate_items(list(PRIMS) + list(routines), self.lib, self.settings,",
     "enumerate_items(list(VOCAB) + list(routines), self.lib, self.settings,"),
    ("tokens = list(PRIMS) + ([] if prims_only else [k for k in self.lib])",
     "tokens = list(VOCAB) + ([] if prims_only else [k for k in self.lib])"),
    ("classes = enumerate_items(list(PRIMS) + routines, self.lib, prob.settings, self.device, max_body=0,",
     "classes = enumerate_items(list(VOCAB) + routines, self.lib, prob.settings, self.device, max_body=0,"),
]

ARMS = {
    "no_twins": ("no register twins: no task is also posed with X and Y swapped (lt~, modx~, ..., reduce~)", [
        const("TWINS = False          # ablation: no X<->Y twin of any practice problem"),
        ("        self.tasks = task_names()\n", "        self.tasks = task_names(TWINS)\n"),
    ]),
    "no_routines": ("routines off: a solved task's program never becomes a routine T.<task>; the round's search "
                    "composes the library's merges instead of routines", [
        const("ROUTINES = False       # ablation: no solved task becomes a routine; the round composes the merges"),
        ("        for t in self.solved_now:                    # every newly solved task's simplest program becomes a "
         "routine\n",
         "        for t in (self.solved_now if ROUTINES else ()):   # every newly solved task's simplest program becomes "
         "a routine\n"),
        ('        routines = [k for k in self.lib if k.startswith("T.")]\n',
         '        routines = [k for k in self.lib if k.startswith("T.") or not ROUTINES]\n'),
        ('            routines = tuple(sorted(k for k in self.lib if k.startswith("T.")))\n',
         '            routines = tuple(sorted(k for k in self.lib if k.startswith("T.") or not ROUTINES))\n'),
        ("            if not (self.round_retry and answers is not None and not self.rounds and self.routines()):\n",
         "            if not (self.round_retry and answers is not None and not self.rounds\n"
         "                    and (self.routines() or (self.lib and not ROUTINES))):\n"),
    ]),
    "no_redstep": ("the two step-definition laws (red.step) dropped: no modB (the chain) and no step practice "
                   "problems; reduce waits for add only", drop_tasks(("modB", "step"), "red.step")),
    "no_step": ("the step practice problem dropped (red.step's step law); modB kept",
                drop_tasks(("step",), "red.step's step law")),
    "no_modB": ("the chain practice problem dropped (red.step's ladder); step kept, its target computed by B "
                "conditional subtractions (modx) instead of the chain", drop_tasks(("modB",), "red.step's ladder") + [
        ('        return {reg("X"): Xv, reg("Y"): xk, "N": Nk}, {"A": run("modB", {"X": v, "N": Nk})[0]}, None\n',
         '        r = v                                                                # ablation: no modB helper; v < '
         'N B,\n'
         '        for _ in range(B):                                                   # so B conditional '
         'subtractions\n'
         '            r = run("modx", {"X": r, "N": Nk})[0]                            # reduce it\n'
         '        return {reg("X"): Xv, reg("Y"): xk, "N": Nk}, {"A": r}, None\n'),
        (DROP_BLOCK, DROP_BLOCK + 'NEEDS["step"] = ("modx",)                   # ablation: step\'s target needs '
                                  'modx now\n'),
    ]),
    "no_mrow": ("the multiply row practice problem dropped (mul.ladder's row); mac's target computes the row by "
                "repeated addition instead of the mrow helper", drop_tasks(("mrow",), "mul.ladder's row") + [
        ('        R = run("mrow", {"X": xt, "Y": qdig})[0]\n',
         '        R = repeat_add(zero, xt, qd)                                         # ablation: no mrow helper\n'),
    ]),
    "no_brg": ("BRG.X (bring X's top digit into A) removed from every search's primitives",
               [const('SEARCH_DROP = ("BRG.X",)  # ablation: primitives no search uses')] + SEARCH_VOCAB),
    "no_sel": ("SEL.S (keep or discard on the borrow) removed from every search's primitives",
               [const('SEARCH_DROP = ("SEL.S",)  # ablation: primitives no search uses')] + SEARCH_VOCAB),
    "ans_smallest_t": ("the round's answers: the batch's smallest-T rows (answer_rows) whatever their T, no T = 1 "
                       "filter", [
        ('        """The answer rows of T = 1 (as read) join the round\'s pool (phases C and D)."""\n',
         '        """The answer rows of the batch\'s smallest T (as read; ablation: no T = 1 filter) join the round\'s '
         'pool (phases C and D)."""\n'),
        ('        if answer_t(aux["Cd"][sel[:1]]) != 1:\n            return\n', ""),
    ]),
}


# The wider search (ABLATION_S3.md §7): a failed arm again, with the missing piece's library searches (merges, routines
# and primitives) storing up to WIDE_STATES states per level, so level 5 is stored and level 6 streamed (every program
# up to cost 6, against cost 5 at s3i's cap), and with no time limit of their own. Everything else is the arm's.
WIDE_STATES = 60_000_000      # 655 bytes a stored state on the H100 (INTEGRATION.md §2.1): 39 GB of its 80
WIDE_MUL = 110_000_000        # the multiply's level 5 holds 131-152M candidates: 72 GB, on an H200 (141 GB)
SET_ARMS = {"mac_n": 'mac posed as x d + N, by the runner\'s --set MAC="N"'}   # a failed arm made by --set, not edits
WIDE = [
    ("        if self.search_s:\n",
     "        wide = task in WIDE_TASKS                # ablation (wider search): a bigger cap, no time limit of its own\n"
     "        if self.search_s and not wide:\n"),
    ('res, how = run("library", self.classes(deadline=deadline), self.lib, self.budget), "library"',
     'res, how = run("library", self.classes(deadline=deadline), self.lib,\n'
     '                                WIDE_STATES if wide else self.budget), "library"'),
]
WIDE_ARMS = {      # arm: (the failed arm, the missing piece's tasks, the cap)
    "no_sel_wide": ("no_sel", ("modx", "modx~"), WIDE_STATES),
    "no_twins_wide": ("no_twins", ("reduce",), WIDE_STATES),
    "no_redstep_wide": ("no_redstep", ("reduce",), WIDE_STATES),
    "no_brg_wide": ("no_brg", ("step",), WIDE_STATES),
    "mac_n_wide": ("mac_n", ("mul",), WIDE_MUL),
    "no_routines_wide": ("no_routines", ("mul",), WIDE_MUL),
}


def wide_arm(arm: str) -> tuple[str, list]:
    base, tasks, states = WIDE_ARMS[arm]
    what, edits = ARMS[base] if base in ARMS else (SET_ARMS[base], [])
    return (f"{base} ({what}), and a wider search for {', '.join(tasks)}: its library searches store up to "
            f"{states:,} states per level (every program up to cost 6) with no time limit of their own", edits + [
        const(f"WIDE_TASKS = {tasks!r}  # ablation (wider search): these tasks' library searches store up to ..."),
        const(f"WIDE_STATES = {states:_}  # ... this many states per level (every program up to cost 6)"),
    ] + WIDE)


def build(arm: str, src: str) -> str:
    what, edits = wide_arm(arm) if arm in WIDE_ARMS else ARMS[arm]
    kind = "wider-search arm" if arm in WIDE_ARMS else "arm"
    out = f"# Ablation {kind} {arm} (integration/ABLATION_S3.md): s3i (md5 {S3I_MD5[:8]}) with one change: {what}.\n"
    out = (out.replace("with one change: ", "with ") if arm in WIDE_ARMS else out) + src
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{arm}: an edit's old text occurs {n} times, not once: {old[:80]!r}")
        out = out.replace(old, new)
    compile(out, f"{arm}.py", "exec")
    return out


def main(argv):
    src = S3I.read_text()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if md5 != S3I_MD5:
        raise SystemExit(f"{S3I} md5 {md5} != {S3I_MD5}: s3i changed")
    if argv[:1] == ["--diff"]:
        out = build(argv[1], src)
        sys.stdout.writelines(difflib.unified_diff(src.splitlines(True), out.splitlines(True), str(S3I.name),
                                                   f"variants/{argv[1]}.py", n=1))
        return
    OUT.mkdir(exist_ok=True)
    sums = {}
    for arm in [*ARMS, *WIDE_ARMS]:
        what = (wide_arm(arm) if arm in WIDE_ARMS else ARMS[arm])[0]
        out = build(arm, src)
        path = OUT / f"{arm}.py"
        if path.exists() and path.read_text() != out and "--force" not in argv:
            raise SystemExit(f"{path} differs from this build (pass --force to replace it)")
        path.write_text(out)
        sums[arm] = {"md5": hashlib.md5(out.encode()).hexdigest(), "what": what}
        print(f"{arm:15s} {sums[arm]['md5']}  {what}")
    (OUT / "MD5SUMS.json").write_text(json.dumps({"s3i": S3I_MD5, "variants": sums}, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
