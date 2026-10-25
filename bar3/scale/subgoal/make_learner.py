"""Writes learn.py (SUBGOAL.md §3): bar3/scale/learn.run copied line for line, with the subgoal hooks applied as exact
replacements, each asserted to match where it should, so the copy cannot drift from the original silently. Rerun it
after any change to bar3/scale/learn.run, then check.py.

    cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/subgoal/make_learner.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "learn.py").read_text().splitlines(keepends=True)
start = next(i for i, line in enumerate(src) if line.startswith("def run("))
end = next(i for i, line in enumerate(src) if line.startswith("def from_json("))
body = "".join(src[start:end]).rstrip() + "\n"

REPL = [
    # the subgoal: scored with the pool, logged
    ('        log(f"[{name}] round {rnd}: target probe mdl {mdl0:.1f} best {tp[\'best_fit\']}; pool {len(pool)}")\n',
     '        log(f"[{name}] round {rnd}: target probe mdl {mdl0:.1f} best {tp[\'best_fit\']}; pool {len(pool)}")\n'
     '        sub = (subgoal_of(pool, tgt, k, k2, chained if cfg["subgoal"].get("chain") else ())\n'
     '               if cfg.get("subgoal") else dict(scores={}, name=None))                                   # subgoal\n'
     '        subs = {sub["name"]} - {None}\n'
     '        tsubs = subs if cfg.get("subgoal", {}).get("ties", True) else set()   # the subgoal\'s ties (off: pool order)\n'
     '        if subs:\n'
     '            log(f"  subgoal: {sub[\'name\']} " + (f"(the rest of the way from {sub[\'after\']})" if sub.get("after")\n'
     '                                                else str(sub["scores"][sub["name"]])))\n'),
    # the routines of the subgoals practised so far
    ('    n_sym = lambda lb: len(brm.PRIMS) + len(lb) + len(kinds)   # noqa: E731\n',
     '    n_sym = lambda lb: len(brm.PRIMS) + len(lb) + len(kinds)   # noqa: E731\n'
     '    chained = []                                            # subgoal: routines of the subgoals practised so far\n'),
    ('        lib = {t: dsl.show(v) for t, v in plib.items()}\n',
     '        lib = {t: dsl.show(v) for t, v in plib.items()}\n'
     '        chained += [f"T.{stems[c.name]}" for c in solved_now if c.name in subs and f"T.{stems[c.name]}" in lib]\n'),
    ('        status = {}\n',
     '        status, deep = {}, []\n'),
    # the deeper probe for the subgoal still unsolved
    ('            for i, r in zip(todo, rb):\n'
     '                res[i] = r\n',
     '            for i, r in zip(todo, rb):\n'
     '                res[i] = r\n'
     '            todo = [i for i, r in enumerate(res) if pool[i].name in subs and r is not None and r["found_cost"] is None]\n'
     '            deep = run_jobs("probe_deep", [job(pool[i], lib, kinds, cfg["subgoal"]["deep"], "probe_deep", k, k2)\n'
     '                                           for i in todo]) if todo else []    # subgoal: a deeper probe\n'
     '            for i, r in zip(todo, deep):\n'
     '                if r is not None:\n'
     '                    res[i] = r\n'),
    # ties: the subgoal first, then pool order (every ordering of the selection step)
    ('key=lambda c: -status[c.name]["cost"])', 'key=lambda c: (-status[c.name]["cost"], c.name not in tsubs))'),
    ('key=lambda c: -scores.get(c.name, -1e9))', 'key=lambda c: (-scores.get(c.name, -1e9), c.name not in tsubs))'),
    ('key=lambda c: (-scores[c.name], status[c.name]["cost"]))',
     'key=lambda c: (-scores[c.name], status[c.name]["cost"], c.name not in tsubs))'),
    ('                pairs.sort(key=lambda p: -p[0])\n',
     '                pairs.sort(key=lambda p: (-p[0], not tsubs & {p[1], p[2]}))\n'),
    # optionally, the subgoal is always practised once it is learnable (in the last pick's place)
    ('        # 5. practice: unsolved picks get the full budget\n',
     '        if cfg.get("subgoal", {}).get("pick") and subs and subs & {c.name for c in learnable} - {c.name for c in pick}:\n'
     '            pick = pick[:K - 1] + [c for c in learnable if c.name in subs]   # subgoal: practised when learnable\n'
     '        # 5. practice: unsolved picks get the full budget\n'),
    # the record
    ('        emit(dict(config=name, round=rnd, k=k, rule=rule, pool=[c.to_dict() for c in pool], status=status,\n',
     '        emit(dict(config=name, round=rnd, k=k, rule=rule, pool=[c.to_dict() for c in pool], status=status,\n'
     '                  subgoal=dict(name=sub["name"], after=sub.get("after"),\n'
     '                               scores=dict(sorted(sub["scores"].items(), key=lambda kv: kv[1]["key"], reverse=True)[:20]),\n'
     '                               deep=[r for r in deep if r is not None]),\n'),
]

HEAD = '''"""The bar-3 learner with a subgoal signal (SUBGOAL.md): bar3/scale/learn.run, line for line (make_learner.py copies
it and applies the hooks below as exact replacements; learn.py itself could not change while runs that import it were
live), plus, with cfg "subgoal":
- each round's pool is scored from the rows alone (score.py), and the subgoal is the candidate whose score is best
  when no other candidate shares that score (a tie means the score singles nothing out; then nothing changes);
- the subgoal, if the two probes leave it unsolved, gets a deeper probe (cfg["subgoal"]["deep"]; configs.py: 10M
  states, so one level more than the 1M-state probe reaches);
- in every ordering of the selection step (hardest first, the anchors, the pairs, the singles), the subgoal wins
  the ties that pool order used to break; everything else keeps pool order (cfg["subgoal"]["ties"] = False: pool
  order for the subgoal too);
- with cfg["subgoal"]["pick"], the subgoal is practised as soon as it is learnable (in the last pick's place);
- with cfg["subgoal"]["chain"], once a subgoal is practised, the next subgoal is the rest of the way from its routine
  to the target, when the pool poses it (sources.routine_remainders' problem): the target is a function of the
  subgoal's answer, and that function is the remainder.
A config without "subgoal" runs exactly as learn.run (check.py compares them).

    ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py --fresh lp_clean_deep_sub --targets ten
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from library import dsl, merge
from library.inventing import brm
from library.inventing.jobs import machine_for
from library.wake_sleep import promote
from bar3 import problems as P
from bar3.breadth import targets as TG
from bar3.breadth.drive import Tap
from bar3.scale import sources as S
from bar3.scale.learn import choose, job, learned_costs, mdl, promote_first, Signatures, stem
from bar3.scale.subgoal import score as SC


def subgoal_of(pool, tgt, k, k2, chained=()):
    """The pool's scores (score.score_pool) and the subgoal. After a subgoal has been practised (its routine in
    `chained`), the rest of the way from that routine, when the pool poses it: the "rrem" problem (the target's answer
    from X = what the routine leaves in A), under that name or another candidate's that posed the same problem; the
    latest routine first. Otherwise the candidate with the best key, when that key is positive and no other candidate
    shares it."""
    sc = SC.score_pool(pool, tgt, k, k2)
    for tok in reversed(list(chained)):
        rest = next((c.name for c in pool if f"rrem/{tok}" in (c.name, *c.also)), None)
        if rest:
            return dict(scores=sc, name=rest, after=tok)
    keys = sorted((v["key"] for v in sc.values()), reverse=True)
    name = None
    if keys and keys[0] > (0, 0) and (len(keys) == 1 or keys[1] < keys[0]):
        name = next(n for n, v in sc.items() if v["key"] == keys[0])
    return dict(scores=sc, name=name)


'''

TAIL = '''

def fresh(name, cfg, tname, search_many, emit, log=print):
    """bar3/scale/drive.fresh with this run."""
    tap = Tap(search_many)
    TG.use(tname)
    tap.target = tname
    run(name, cfg, tap.mapper, tap.emit_with_looks(lambda r: emit(dict(r, target_name=tname))), log=log)
'''


def main():
    out, counts = body, {}
    for old, new in REPL:
        n = out.count(old)
        assert n >= 1, ("no match", old)
        if not old.startswith("key=lambda"):
            assert n == 1, ("more than one match", old)
        counts[old.strip()[:50]] = n
        out = out.replace(old, new)
    (HERE / "learn.py").write_text(HEAD + out + TAIL)
    print("wrote learn.py;", sum(counts.values()), "replacements:", counts)


if __name__ == "__main__":
    main()
