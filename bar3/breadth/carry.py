"""Targets in sequence with the library persisting (BREADTH.md §4): bar3.learner.run, line for line, except that it
starts from the state the previous target left (library, loop kinds, corpus, practiced problems, routine names) and
returns its own at the end. The code that runs is learner.run's source with the three changes in PATCHES applied
(each must match exactly once, so a change to learner.py fails here loudly instead of drifting).

Between targets the harness (`sequence`) does two things:
- it renames the routine-name table's keys with the finished target's name, so a later candidate that happens to
  share a name (the same near-miss program, another target's answers) gets a fresh routine name;
- with carry="targets", the solved target joins as a problem of its own: its general programs go into the corpus,
  the one `choose` keeps (no side effects, then the shortest expansion) becomes the routine T.tgt_<name>, exactly as
  a practiced problem's (promote_first). With carry="library" only the practiced problems' routines and merges carry.
"""

from __future__ import annotations

import inspect

from library import dsl
from bar3 import learner
from bar3 import problems as P
from bar3.breadth import targets as TG

PATCHES = (
    ("def run(name, cfg, mapper, emit, log=print):",
     "def run(name, cfg, mapper, emit, log=print, init=None):\n    init = init or {}"),
    ("    lib, kinds = {}, list(brm.LOOP_KINDS)        # lib: token -> program text",
     "    lib, kinds = dict(init.get('lib', {})), list(init.get('kinds', brm.LOOP_KINDS))"),
    ("    corpus, practiced, stems = {}, {}, {}         # corpus: stem -> [program text]; practiced: key -> candidate name",
     "    corpus, practiced, stems = (dict(init.get('corpus', {})), dict(init.get('practiced', {})),\n"
     "                                dict(init.get('stems', {})))"),
)
RETURN = "    return dict(lib=lib, kinds=kinds, corpus=corpus, practiced=practiced, stems=stems, general=gen)\n"


def _build():
    src = inspect.getsource(learner.run)
    for old, new in PATCHES:
        assert src.count(old) == 1, f"learner.run changed: {old!r}"
        src = src.replace(old, new)
    src = src.rstrip("\n") + "\n" + RETURN
    ns = dict(vars(learner))
    exec(compile(src, learner.__file__ + " (carry)", "exec"), ns)
    return ns["run"]


run = _build()


def add_target(state, tname, target_rec, k, k2):
    """The solved target as a practiced problem: its general programs into the corpus, the chosen one a routine."""
    progs = [p["program"] for p in target_rec["programs"] if p.get("general") == 1.0]
    tgt = P.the_target(k, k2)
    progs = learner.choose(progs, state["lib"], tgt, k, "clean")
    stem = f"tgt_{tname}"
    pcorpus = {t: [dsl.parse(p) for p in ps] for t, ps in state["corpus"].items()}
    pcorpus[stem] = [dsl.parse(p) for p in progs[:20]]
    plib = {t: dsl.parse(v) for t, v in state["lib"].items()}
    pcorpus, plib, plog = learner.promote_first(pcorpus, plib, [stem])
    state = dict(state, lib={t: dsl.show(v) for t, v in plib.items()},
                 corpus={t: [dsl.show(p) for p in ps[:20]] for t, ps in pcorpus.items()})
    return state, plog


def to_json(state):
    """The carried state as plain JSON (practiced keys are tuples of tuples of ints)."""
    return dict(state, practiced=[[list(map(list, key)), nm] for key, nm in state["practiced"].items()])


def from_json(d):
    return dict(d, practiced={tuple(map(tuple, key)): nm for key, nm in d["practiced"]})


def sequence(name, cfg, order, mapper, emit, carry="targets", log=print, on_target=None, state=None, save=None):
    """Run cfg on each target of `order` in turn, the state carried from one to the next. emit gets every round
    record with the target's name added; on_target(name) is called before each target (the driver tags jobs);
    save(target, state) after each one (a restarted driver passes the saved state back as `state`)."""
    state = state or {}
    for tname in order:
        TG.use(tname)
        if on_target:
            on_target(tname)
        last = {}

        def emit_t(rec):
            rec = dict(rec, target_name=tname, carry=carry, carried_lib=len(state.get("lib", {})))
            last["rec"] = rec
            emit(rec)

        before = dict(state.get("stems", {}))
        state = run(f"{name}/{tname}", cfg, mapper, emit_t, log=log, init=state)
        new = {n: s for n, s in state["stems"].items() if n not in before}
        state["stems"] = {**before, **{f"{tname}::{n}": s for n, s in new.items()}}
        if carry == "targets" and state["general"] is not None:
            k, k2 = cfg["k"], cfg.get("k2", P.K2)
            state, plog = add_target(state, tname, last["rec"]["target"], k, k2)
            emit(dict(config=f"{name}/{tname}", target_name=tname, carry=carry, kind="carried_target",
                      promoted=plog, lib=state["lib"]))
            log(f"  carried target {tname}: {plog}")
        if save:
            save(tname, state)
    return state
