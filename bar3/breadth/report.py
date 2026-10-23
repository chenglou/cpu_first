"""Tables for BREADTH.md from the breadth records.

    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --matrix bar3/breadth/records/*_b1.jsonl  # target × arm
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --fresh bar3/breadth/records/*_b1.jsonl   # per target
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --seq bar3/breadth/records/seq_*.jsonl     # carry-over
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --pairs bar3/breadth/records/ct2_*_b1.jsonl  # replay
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --effort bar3/breadth/records/*.jsonl      # how hard
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --same RUN.jsonl PRACTICE_RUN.jsonl ...    # the gate
    ~/github/sotaku/venv/bin/python -m bar3.breadth.report --detail bar3/breadth/records/ct2_clean_rho_b1.jsonl

Concepts: a routine *has* a concept when, run alone from a fresh machine (the argument in X, or in X and A), it
computes that function on every instance of the concept's domain with N <= 10 (concept_of). Pair replay (`--pairs`):
from a ct2 round's full lookahead list, which problems the pair rule would pick when the pairs are taken only among
the k best single problems (by the single lookahead, or by cost), or must have one member among them, whether those
picks hold a pair that makes the target solvable in the lookahead, and what the lookaheads would cost.
"""

from __future__ import annotations

import collections
import itertools
import json
import sys

import numpy as np

from library import dsl
from library.inventing import brm
from bar3 import learner
from bar3 import problems as P
from bar3.breadth import targets as TG

KINDS = ("target_probe", "probe_a", "probe_b", "look", "practice", "target")


def load(path):
    """A run's records. A relaunched run rewrites its file from round 0, and the driver it replaced can still append
    a last record after that (cancelled drivers run on for seconds): keep only the records after the last restart,
    i.e. from the last round-0 record of the file's first target (and only the newest launch's, where marked)."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if recs and "launch" in recs[-1]:
        recs = [r for r in recs if r.get("launch") == recs[-1]["launch"]]
    first = recs[0].get("target_name") if recs else None
    starts = [i for i, r in enumerate(recs) if r.get("round") == 0 and r.get("target_name") == first]
    recs = recs[starts[-1]:] if starts else recs
    while recs and "round" in recs[-1] and recs[-1]["target"] is None:   # a round cut off by stopping its driver
        recs = recs[:-1]
    return recs


def rounds_of(recs):
    return [r for r in recs if "round" in r]


# ---------------------------------------------------------------------------------------------------------------
# concepts a routine may hold

def _domain(kind, upto=10):
    xs, Ns = [], []
    for n in range(1, upto + 1):
        top = {"x<N": n, "x<2N": 2 * n, "x<N2": n * n}[kind]
        for x in range(top):
            xs.append(x)
            Ns.append(n)
    return np.array(xs, np.int64), np.array(Ns, np.int64)


CONCEPTS = {   # name: (domain, function of X and N)
    "link: X mod N for X < 2N": ("x<2N", lambda X, N: X % N),
    "wrap step: (X + 1) mod N": ("x<N", lambda X, N: (X + 1) % N),
    "(X + 2) mod N": ("x<N", lambda X, N: (X + 2) % N),
    "(X + 3) mod N": ("x<N", lambda X, N: (X + 3) % N),
    "unwrap step: (X − 1) mod N": ("x<N", lambda X, N: (X - 1) % N),
    "(X − 2) mod N": ("x<N", lambda X, N: (X - 2) % N),
    "reduction: X mod N for X < N²": ("x<N2", lambda X, N: X % N),
    "square: X²": ("x<N", lambda X, N: X * X),
    "x² mod N": ("x<N", lambda X, N: (X * X) % N),
    "double: 2X": ("x<N", lambda X, N: 2 * X),
    "double mod N: 2X mod N": ("x<N", lambda X, N: (2 * X) % N),
    "product: X·N": ("x<N", lambda X, N: X * N),
    "X + N": ("x<N", lambda X, N: X + N),
    "X + 1": ("x<N", lambda X, N: X + 1),
    "N ∸ X": ("x<N", lambda X, N: N - X),
    "halve: X div 2": ("x<2N", lambda X, N: X // 2),
}


def concept_of(tok, lib):
    """The concepts tok computes into A, from a fresh machine (X = x) or with the argument in A as well (X = A = x:
    a routine that works on A in place, as the modular increment does)."""
    pl = {t: dsl.parse(v) for t, v in lib.items()}
    m = brm.Machine(cap=4000, limit=10 ** 6)
    out = []
    for name, (dom, f) in CONCEPTS.items():
        X, N = _domain(dom)
        for a_too in (False, True):
            b = brm.fresh(X, N)
            if a_too:
                b.regs[0, :, brm.A] = X
            brm.run_items(m, b, (tok,), pl)
            if not b.dead[0] and (b.regs[0, :, brm.A] == f(X, N)).all():
                out.append(name + (" (on A)" if a_too else ""))
                break
    return out


def tokens_in(prog):
    out = []
    for it in dsl.parse(prog):
        stack = [it]
        while stack:
            i = stack.pop()
            if isinstance(i, tuple):
                if i[1] not in brm.LOOP_KINDS:
                    out.append(i[1])
                stack.extend(i[2])
            elif i not in brm.PRIMS:
                out.append(i)
    return out


# ---------------------------------------------------------------------------------------------------------------
# one run

def hours(t, kinds):
    return sum(t.get(k, {}).get("seconds", 0) for k in kinds) / 3600


def summary(recs):
    rs = rounds_of(recs)
    if not rs:
        return None
    assert [r["round"] for r in rs] == list(range(len(rs))), f"rounds out of order in {rs[0]['config']}"
    last = rs[-1]
    t = last["totals"]
    gen = next((r for r in rs if r.get("general")), None)
    forms = {}
    for r in rs:
        forms.update({c["name"]: c["formula"] for c in r["pool"]})
    picks = [[(n, forms.get(n, ""), r["status"].get(n, {}).get("cost")) for n in r["picked"]] for r in rs]
    return dict(config=last["config"], target=last.get("target_name"), rule=last["rule"], rounds=len(rs),
                invented=gen["round"] if gen else None, program=gen["general"]["program"] if gen else "",
                primitive=gen["general"]["primitive"] if gen else "", picks=picks,
                pool=[len(r["pool"]) for r in rs],
                learnable=[sum(1 for v in r["status"].values() if v.get("status") == "learnable") for r in rs],
                states=sum(v["states"] for v in t.values()), hours=hours(t, KINDS),
                probe_h=hours(t, ("target_probe", "probe_a", "probe_b")), look_h=hours(t, ("look",)),
                search_h=hours(t, ("practice", "target")), lib=last["lib"],
                target_stop=[(r["target"] or {}).get("stop") for r in rs],
                target_levels=[(r["target"] or {}).get("level_sizes") for r in rs],
                best_fit=[(r["target"] or {}).get("best_fit") for r in rs],
                probe_best=[r["target_probe"]["best_fit"] for r in rs])


def short(name, formula):
    return formula or name


ARMS = ("lp_clean", "ct2_clean", "ct2a2_clean", "all_once", "curated_clean")
ARM_WORDS = {"lp_clean": "hardest first", "ct2_clean": "pairs signal", "ct2a2_clean": "cheaper pairs (round 0 only)",
             "all_once": "practice everything (one round)", "curated_clean": "our curated list"}


def arm_of(s):
    """The config a run was launched with: its name minus the target and the tag."""
    return s["config"].removesuffix("_" + s["config"].rsplit("_", 1)[1]).removesuffix("_" + s["target"])


def matrix(paths):
    """Target × arm: the round it was invented in, or 'no' with the rounds tried, and the search-hours."""
    rows = [s for s in (summary(rs) for rs in (load(p) for p in paths) if rs and "carry" not in rs[0]) if s]
    cell = {(s["target"], arm_of(s)): s for s in rows}
    print("| target | " + " | ".join(ARM_WORDS[a] for a in ARMS) + " |\n|---|" + "---|" * len(ARMS))
    for t, tg in TG.TARGETS.items():
        out = []
        for ru in ARMS:
            s = cell.get((t, ru))
            if s is None:
                out.append("–")
            elif s["invented"] is not None:
                out.append(f"**round {s['invented']}** ({s['hours']:.1f} h)")
            else:
                out.append(f"no, {s['rounds']} round{'s' if s['rounds'] > 1 else ''} ({s['hours']:.1f} h)")
        print(f"| {t}: {tg.formula.split(' (')[0]} | " + " | ".join(out) + " |")


def fresh_table(paths):
    recs = [load(p) for p in paths]
    rows = [s for s in (summary(rs) for rs in recs if rs and "carry" not in rs[0]) if s]   # sequences: --seq
    rows.sort(key=lambda s: (list(TG.TARGETS).index(s["target"]) if s["target"] in TG.TARGETS else 99, s["rule"]))
    print("| target | rule | invented | program | rounds | pool / learnable (round 0) | states | search-hours "
          "(probes / lookahead / practice + target) |\n|---|---|---|---|---|---|---|---|")
    for s in rows:
        inv = f"round {s['invented']}" if s["invented"] is not None else "no"
        prog = f"`{s['program']}`" if s["program"] else "–"
        print(f"| {s['target']} | {s['rule']} | {inv} | {prog} | {s['rounds']} | {s['pool'][0]} / {s['learnable'][0]} "
              f"| {s['states'] / 1e6:,.1f}M | {s['hours']:.2f} ({s['probe_h']:.2f} / {s['look_h']:.2f} / "
              f"{s['search_h']:.2f}) |")
    print()
    for s in rows:
        print(f"- **{s['target']} / {s['rule']}**: " + "; ".join(
            f"round {i}: " + ", ".join(f"{short(n, f)} ({c})" for n, f, c in pk) for i, pk in enumerate(s["picks"])))
        if s["program"]:
            print(f"  - program `{s['program']}` = `{s['primitive']}`")
        else:
            print(f"  - target search per round: " + "; ".join(
                f"best fit {bf} stop {st}" for bf, st in zip(s["best_fit"], s["target_stop"])))


def detail(path):
    recs = load(path)
    for r in recs:
        if "round" not in r:
            print(f"\n== carried: {r.get('promoted')}")
            continue
        st = collections.Counter(v.get("status", "practice " + v.get("practice", "?")) for v in r["status"].values())
        src = collections.Counter(c["source"] for c in r["pool"])
        print(f"\n== {r['config']} round {r['round']} (lib {len(r['lib'])}): pool {len(r['pool'])} {dict(src)}; "
              f"probes {dict(st)}; target probe best {r['target_probe']['best_fit']} (mdl {r['mdl0']:.1f})")
        print("   near misses:", [(f, d, p) for f, d, p in r["target_probe"].get("near", [])[:8]])
        forms = {c["name"]: c["formula"] for c in r["pool"]}
        for name in r["picked"]:
            s = r["status"].get(name, {})
            sc = r["scores"].get(name)
            print(f"  picked {name}: {forms.get(name, '')}\n     {s.get('status')} cost {s.get('cost')} "
                  f"`{s.get('best', '')}`" + (f" gain {sc:.1f}" if sc is not None else ""))
        if r["pairs"]:
            solved = [p for p in r["pairs"] if p[3] is not None]
            print(f"  pairs: {len(r.get('looks', [])) - len(r['scores'])} looked at, {len(solved)} of the best 200 "
                  f"solve the target: {[(round(g, 1), a, b, c) for g, a, b, c in solved[:4]]}")
            print("  best pairs:", [(round(g, 1), a, b, c) for g, a, b, c in r["pairs"][:4]])
        for e in r["merges"] + r["promoted"]:
            print(f"    + {e['name']} = {e['def']}    ({e.get('expands_to', '')})")
        tr = r["target"]
        print(f"  target: cost {tr and tr['found_cost']} levels {tr and tr['level_sizes']} best fit "
              f"{tr and tr['best_fit']} stop {tr and tr['stop']}")
        for p in (tr or {}).get("programs", [])[:4]:
            print(f"     {p['program']}  (general {p.get('general')})  = {p['primitive']}")
        t = r["totals"]
        print("  totals:", {k: f"{v['jobs']} jobs {v['seconds'] / 3600:.2f} h" for k, v in t.items()})


# ---------------------------------------------------------------------------------------------------------------
# carry-over sequences

def seq_table(paths):
    for path in paths:
        recs = load(path)
        print(f"\n### {path.split('/')[-1]}\n")
        print("| target | rounds | invented | program | uses routines from | library at start / end | states | "
              "search-hours | concepts the library holds at the end |\n|---|---|---|---|---|---|---|---|---|")
        origin = {}
        by_target = collections.OrderedDict()
        for r in recs:
            by_target.setdefault(r["target_name"], []).append(r)
        lib = {}
        for tname, rs in by_target.items():
            start = len(lib)
            for r in rs:
                for e in r.get("merges", []) + r.get("promoted", []):
                    origin.setdefault(e["name"], tname)
                lib = r["lib"]
            s = summary(rs)
            used = sorted({origin.get(t, "?") for t in tokens_in(s["program"])} - {tname}) if s["program"] else []
            concepts = sorted({c for tok in lib for c in concept_of(tok, lib)})
            inv = f"round {s['invented']}" if s["invented"] is not None else "no"
            print(f"| {tname} | {s['rounds']} | {inv} | {'`' + s['program'] + '`' if s['program'] else '–'} | "
                  f"{', '.join(used) or '–'} | {start} / {len(lib)} | {s['states'] / 1e6:,.1f}M | {s['hours']:.2f} | "
                  f"{'; '.join(concepts)} |")


# ---------------------------------------------------------------------------------------------------------------
# replaying the pair choice

def replay(r, prefilter, kk, K=4):
    """The pair rule with pairs only among the kk best learnable problems (prefilter: 'single' = the single
    lookahead's gain, 'cost' = the hardest first), or with every pair that has one member among the kk best singles
    ('anchor': kk·L pairs instead of L²/2). Returns (picks, the pick set holds a solving pair, lookaheads,
    seconds)."""
    by_prog = {v["best"]: n for n, v in r["status"].items() if v.get("status") == "learnable"}
    learnable = [n for n in by_prog.values()]
    order = {n: i for i, n in enumerate(learnable)}
    single, pair = {}, {}
    for p0, p1, fc, bf, NP, sec, nsym in r["looks"]:
        rec = dict(best_fit=bf, NP=NP, found_cost=fc)
        if p1 is None:
            single[by_prog[p0]] = (r["mdl0"] - learner.mdl(rec, nsym, r["k"]), fc, sec)
        else:
            a, b = by_prog[p0], by_prog[p1]
            pair[frozenset((a, b))] = (r["mdl0"] - learner.mdl(rec, nsym, r["k"]), fc, sec)
    if prefilter in ("cost", "anchor_cost"):
        cand = sorted(learnable, key=lambda n: -r["status"][n]["cost"])[:kk]
    else:
        cand = sorted(learnable, key=lambda n: -single[n][0])[:kk]
    pool = learnable if prefilter.startswith("anchor") else cand
    combos = [frozenset(c) for c in itertools.combinations(sorted(pool, key=order.get), 2) if set(c) & set(cand)]
    ranked = sorted(combos, key=lambda c: -pair[c][0])
    chosen = []
    for c in ranked:
        a, b = sorted(c, key=order.get)
        for n in (a, b):
            if n not in chosen and len(chosen) < K:
                chosen.append(n)
        if len(chosen) >= K:
            break
    solving = any(pair[frozenset(c)][1] is not None for c in itertools.combinations(chosen, 2))
    singles_used = prefilter in ("single", "anchor")
    n_look = len(learnable) * singles_used + len(combos)
    secs = sum(v[2] for v in single.values()) * singles_used + sum(pair[c][2] for c in combos)
    return chosen, solving, n_look, secs


def pairs_table(paths):
    print("| run | round | learnable | pairs | solving pairs | top-k singles: smallest k that keeps a solving pair "
          "(lookaheads, search-hours) | top-k hardest | one member in the top-k singles | one member among the k "
          "hardest | all pairs (lookaheads, search-hours) |\n|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        for r in rounds_of(load(path)):
            if r["rule"] != "ct2" or not r.get("looks"):
                continue
            L = sum(1 for v in r["status"].values() if v.get("status") == "learnable")
            n_pairs = sum(1 for lk in r["looks"] if lk[1] is not None)
            n_solve = sum(1 for lk in r["looks"] if lk[1] is not None and lk[2] is not None)
            cells = []
            for pre in ("single", "cost", "anchor", "anchor_cost"):
                hit = None
                for kk in range(2, L + 1):
                    ch, ok, nl, sec = replay(r, pre, kk)
                    if ok:
                        hit = (kk, nl, sec)
                        break
                cells.append(f"k = {hit[0]} ({hit[1]}, {hit[2] / 3600:.2f})" if hit else "none")
            _, ok, nl, sec = replay(r, "single", L)
            print(f"| {r['config']} | {r['round']} | {L} | {n_pairs} | {n_solve} | {cells[0]} | {cells[1]} | "
                  f"{cells[2]} | {cells[3]} | {nl} ({sec / 3600:.2f}){'' if ok else ', no solving pair'} |")


def effort(paths):
    """How hard each run was tried: rounds finished, outcome, every stage's jobs, states and search-hours (a run of a
    sequence counts per target)."""
    print("| run | target | rounds finished | outcome | probe jobs (a / b) | lookaheads | practice searches | "
          "target searches | states | search-hours |\n|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        by = collections.OrderedDict()
        for r in rounds_of(load(path)):
            by.setdefault(r.get("target_name"), []).append(r)
        for tname, rs in by.items():
            t = rs[-1]["totals"]
            j = lambda k: t.get(k, {}).get("jobs", 0)                        # noqa: E731
            gen = next((r for r in rs if r.get("general")), None)
            out = f"general in round {gen['round']}" if gen else "not found"
            run = path.split("/")[-1].removesuffix(".jsonl")
            print(f"| {run} | {tname} | {len(rs)} | {out} | {j('probe_a')} / {j('probe_b')} | {j('look')} | "
                  f"{j('practice')} | {j('target')} | {sum(v['states'] for v in t.values()) / 1e6:,.1f}M | "
                  f"{hours(t, KINDS):.2f} |")


def same(paths):
    """The validation: a breadth run on x² mod N against PRACTICE.md's record of the same config (pairs of paths,
    breadth first), round by round: the pool, every probe's outcome, the picks, the library and the target's
    programs."""
    for a, b in zip(paths[::2], paths[1::2]):
        ra, rb = rounds_of(load(a)), rounds_of(load(b))
        for x, y in zip(ra, rb):
            checks = dict(
                pool=[c["name"] for c in x["pool"]] == [c["name"] for c in y["pool"]],
                status={n: v.get("status") for n, v in x["status"].items()}
                == {n: v.get("status") for n, v in y["status"].items()},
                picked=x["picked"] == y["picked"], lib=x["lib"] == y["lib"],
                target=[p["program"] for p in (x["target"] or {}).get("programs", [])]
                == [p["program"] for p in (y["target"] or {}).get("programs", [])])
            print(f"{a.split('/')[-1]} vs {b.split('/')[-1]} round {x['round']}: {checks}")


def main():
    mode, paths = sys.argv[1], sys.argv[2:]
    {"--fresh": fresh_table, "--seq": seq_table, "--pairs": pairs_table, "--effort": effort, "--same": same,
     "--matrix": matrix}.get(
        mode, lambda ps: [detail(p) for p in ps])(paths if mode.startswith("--") else [mode] + paths)


if __name__ == "__main__":
    main()
