"""The offline test (LEGAL_TEST.md; PREREG.md): the composition route's last two steps, on the full-size target mode
(rowtarget.py) instead of the small target, from s4_x2's library at the start of its level 5
(bar3/scale/slice/records/s4_x2.json, history[4]: after level 4's sleep).

  Q1  the law view by the learner's own routines, over the round search's held states of cost <= 3 on the 400 answer
      rows (registers A, X, Y, S, cheapest first, as LAWNEAR reads them): does it pick x²?
  Q2  the rest of the way from the value it picks, posed at full size (8 probes: the round's; accepted on the 400
      answer rows), probed at 1M states and, if that fails, at REST_DEEP's 10M: does it find the reduction?
  Q3  the round the two make: right on the 400 answer rows, and on all 1,000 T = 1 rows?

Arms, in this order: "full" (the library as recorded), "pieces" (less the ten value routines the small target posed,
and the merge that calls one), "rows780" (Q1 alone on 780 T = 1 rows, the full library: the harness's permutation's
first 780, its 400 among them; 780 is the most rows a state can hold under libcore's state hash, 8,192 words).
Python's integers appear only in the report's diagnostics (`check_x2`, `check_mod`), never in a test.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python -m bar3.scale.legal.offline
    env: ARMS (full,pieces,rows780), LAW_COST (3), BUDGETS (1000000,10000000), SEARCH_S (900), MAX_BODY (3),
         N_ANSWERS (400), SECONDS (2400: Q1's searches' deadline from the start), TAG, OUT, H1_DATA
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SPELL", "16")          # before spcore_s is imported: round 2's SPELL, 16 spellings a goal state

import torch  # noqa: E402

from integration.lib.libcore import RI, _OneGoal, hash_rows, parse, programs, round_ok, show  # noqa: E402
from bar3.scale.slice.spcore_s import SPELL_K, _Res, bfs_roots  # noqa: E402
from bar3.scale.legal.rowtarget import (P, REDUCTIONS, RowTarget, answer_rows, h1_rows,  # noqa: E402
                                        routine_for)

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "slice" / "records" / "s4_x2.json"
H1_DIR = os.environ.get("H1_DATA", str(HERE.parents[3] / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
X2 = "LDA.X STA.Y T.mul.x,o.XY"               # x², as s4_x2's LAWNEAR posed it (val.134): Q2's fallback, handed
env = os.environ.get


def log(*a):
    print(*a, flush=True)


def names(items):
    for it in items:
        if isinstance(it, tuple):
            yield it[1]
            yield from names(it[2])
        else:
            yield it


def library(arm):
    """(library text, ops) at the start of s4_x2's level 5; "pieces": less every T.val routine and what calls one."""
    rec = json.load(open(REC))
    txt = dict(rec["history"][4]["lib"])
    if arm == "pieces":
        drop = {k for k in txt if k.startswith("T.val.")}
        while True:
            more = {k for k, v in txt.items() if k not in drop and any(t in drop for t in names(parse(v)))}
            if not more:
                break
            drop |= more
        txt = {k: v for k, v in txt.items() if k not in drop}
    return txt, rec["ops"]


def ints(D):
    """Digits [.., P] (LSB first) as int64 numbers: the report's diagnostics only."""
    return (D.long() * (10 ** torch.arange(P, device=D.device))).sum(-1)


def check_x2(g, rt):
    """Diagnostic: is the value x² on every row (Python's integers)?"""
    x = ints(rt.Xd)
    return bool((ints(g.T) == x * x).all())


def check_mod(G, rt):
    """Diagnostic: which values g [S, P, n] give the answer as g mod N on every row (numpy's law view, all rows)."""
    v = ints(G.permute(0, 2, 1))
    return (torch.remainder(v, ints(rt.Nd)[None]) == ints(rt.Yd)[None]).all(1)


def q1(rt, lib, ops, max_cost, deadline):
    """The law view by routines over the held states: every distinct value of A, X, Y, S at costs 1..max_cost,
    cheapest first; the ones that pass, with their programs."""
    t0 = time.time()
    chain, compare = routine_for(ops, "red", REDUCTIONS), routine_for(ops, "lt", ("pair",))
    assert chain in lib and compare in lib, (chain, compare)
    chunk = max(1, (1 << 30) // (83 * rt.n))
    classes, sizes, stop, levels, parents = rt.held(lib, max_cost=max_cost, deadline=deadline, chunk=chunk)
    t_held = time.time() - t0
    cand, vals, seen, per_cost = [], [], set(), {}
    for d in range(1, len(levels)):
        if levels[d] is None:
            continue
        lv = levels[d][0]
        for reg in "AXYS":
            G = lv.regs[:, RI[reg]]
            keep = []
            for i, hv in enumerate(hash_rows(G.reshape(G.shape[0], -1)).tolist()):
                if hv not in seen:
                    seen.add(hv)
                    keep.append(i)
            if keep:
                vals.append(G[torch.tensor(keep, device=G.device)])
                cand += [(d, reg, i) for i in keep]
                per_cost[d] = per_cost.get(d, 0) + len(keep)
    G = torch.cat(vals)
    del vals
    t1 = time.time()
    n_exact, ok = rt.law_mod(G, lib, chain, compare)
    t_law = time.time() - t1
    numpy_ok = check_mod(G, rt)
    passed = torch.nonzero(ok).flatten().tolist()
    out = []
    for j in passed[:16]:
        d, reg, i = cand[j]
        pr = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
        out.append(dict(cost=d, reg=reg, program=show(pr[0]) if pr else None, exact_rows=int(n_exact[j]),
                        is_x2=check_x2(G[j], rt), numpy_mod=bool(numpy_ok[j])))
    ne = n_exact.tolist()
    res = dict(rows=rt.n, chain=chain, compare=compare, held_sizes=sizes, held_stop=stop, held_seconds=round(t_held, 1),
               held_states=[0 if lv is None else int(lv[0].S) for lv in levels],
               law_seconds=round(t_law, 1), values=len(cand), values_by_cost=per_cost,
               passed=len(passed), picked=out[0] if out else None, passing=out,
               numpy_passed=int(numpy_ok.sum()), both=int((ok & numpy_ok).sum()),
               routine_only=int((ok & ~numpy_ok).sum()), numpy_only=int((numpy_ok & ~ok).sum()),
               exact_rows_hist={str(k): ne.count(k) for k in sorted(set(ne))[:40]})
    if out:                                    # the numpy-only values: what the routines cannot see (a sample)
        res["numpy_only_sample"] = []
        for j in torch.nonzero(numpy_ok & ~ok).flatten().tolist()[:6]:
            d, reg, i = cand[j]
            pr = programs(_OneGoal(_Res(parents, d, [i]), i), classes, limit=1)
            res["numpy_only_sample"].append(dict(cost=d, reg=reg, program=show(pr[0]) if pr else None,
                                                 exact_rows=int(n_exact[j])))
    del G, levels, parents
    return res


def q2(rt, lib, prog, reg, budgets, search_s, max_body):
    """The rest of the way from prog's value in reg, at full size."""
    t0 = time.time()
    value = rt.value(parse(prog), lib, reg)
    clean = rt.clean(parse(prog), lib, reg)
    idx = torch.arange(8, device=rt.device)
    root, goal = rt.rest_root(value, idx)
    acc = rt.rest_acceptor(value)
    classes = rt.comp_classes(lib, max_body=max_body, chunk=(1 << 23) if rt.device == "cuda" else (1 << 16))
    t_cls = time.time() - t0
    runs = []
    for cap in budgets:
        t1 = time.time()
        res, sizes, stops, levels, rids, parents = bfs_roots(
            [[root]], [rt.st], [(0, goal)], [acc], classes, lib, max_cost=7, max_states=cap, deadline=t1 + search_s,
            chunk=(1 << 21) if rt.device == "cuda" else (1 << 14), fuse=4 if rt.device == "cuda" else 1)
        del levels, parents
        x = res[0]
        runs.append(dict(budget=cap, found_cost=x["found_cost"],
                         programs=[show(p) for p in (x["accepted"] or [])][:8],
                         aliases=[[d, a] for d, a in x["aliases"]][:6], sizes=sizes[0], stop=stops[0],
                         seconds=round(time.time() - t1, 1)))
        log(f"  Q2 at {cap:,} states: {runs[-1]}")
        if x["found_cost"] is not None:
            break
    return dict(value=prog, reg=reg, clean=clean, classes=len(classes), classes_seconds=round(t_cls, 1),
                probes=8, accepted_on=rt.n, runs=runs)


def q3(rt, lib, prog, reg, rest, all_rows):
    """The round: the value, into X, then the rest; right on the answer rows, and on every T = 1 row?"""
    rnd = parse(prog) + ((f"LDA.{reg}",) if reg != "A" else ()) + ("STA.X",) + parse(rest)
    ok = round_ok(rnd, lib, rt.st.cells, (rt.Nd, rt.Xd, rt.Yd), P, rt.device)
    rt2 = RowTarget(all_rows, rt.device)
    ok2 = round_ok(rnd, lib, rt.st.cells, (rt2.Nd, rt2.Xd, rt2.Yd), P, rt.device)
    return dict(round=show(rnd), right_answer_rows=ok, right_all_t1=ok2, all_t1=len(all_rows))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = env("TAG", "legal_local")
    out = Path(env("OUT", str(HERE / "records"))) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(env("SECONDS", 2400))
    arms = env("ARMS", "full,pieces,rows780").split(",")
    law_cost = int(env("LAW_COST", 3))
    budgets = [int(float(b)) for b in env("BUDGETS", "1000000,10000000").split(",")]
    search_s, max_body = float(env("SEARCH_S", 900)), int(env("MAX_BODY", 3))
    answers = answer_rows(H1_DIR, int(env("N_ANSWERS", 400)), 1)
    all_rows = h1_rows(H1_DIR)
    rec = dict(tag=tag, device=torch.cuda.get_device_name() if dev == "cuda" else "cpu", spell=SPELL_K,
               config=dict(arms=arms, law_cost=law_cost, budgets=budgets, search_s=search_s, max_body=max_body,
                           answers=len(answers), all_t1=len(all_rows), record=str(REC.name), level=5),
               arms={})
    log(f"device {rec['device']} SPELL {SPELL_K} config {rec['config']}")

    def save():
        rec["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(rec, indent=1, default=str))

    for arm in arms:
        wide = arm.startswith("rows")
        txt, ops = library("full" if wide else arm)
        lib = {k: parse(v) for k, v in txt.items()}
        a = rec["arms"].setdefault(arm, dict(lib=txt))
        rows = answer_rows(H1_DIR, int(arm[4:]), 1) if wide else answers
        rt = RowTarget(rows, dev)
        t1 = time.time()
        a["q1"] = q1(rt, lib, ops, law_cost, deadline)
        brief = {k: v for k, v in a["q1"].items() if k != "passing"}
        log(f"[{arm}] Q1 ({time.time() - t1:.0f} s): {json.dumps(brief)}")
        for p in a["q1"]["passing"]:
            log(f"    passes: {p}")
        save()
        if wide:
            continue
        pick = a["q1"]["picked"]
        prog, reg, how = (pick["program"], pick["reg"], "picked") if pick else (X2, "A", "handed")
        a["q2"] = dict(q2(rt, lib, prog, reg, budgets, search_s, max_body), how=how)
        save()
        found = next((r for r in a["q2"]["runs"] if r["found_cost"] is not None), None)
        if found:
            a["q3"] = q3(rt, lib, prog, reg, found["programs"][0], all_rows)
            log(f"[{arm}] Q3: {a['q3']}")
        save()
    log(f"done in {time.time() - t0:.0f} s; wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
