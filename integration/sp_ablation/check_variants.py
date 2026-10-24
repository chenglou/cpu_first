"""Local plumbing check of every arm (integration/sp_ablation/SP_ABLATION.md): loads each arm as the runner would (a
variant file, or sp2 with its --set), checks the one thing it changes as the climb sees it, and runs a few climb units
with toy budgets on made-up rows with exact cells, so every changed code path runs once.
  * the book's arms: with the routines the book's constructions need given (a designed adder, link, comparison and
    chain, sp2's official run's own programs), the pool the climb poses is the control's minus exactly the dropped
    law's or pattern's candidates;
  * the rules: the climb runs its rule, and the rule's picks are what the rule says on the level's statuses.
Seconds and well under 1 GB; the real runs are on Modal (plans/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_ablation/check_variants.py [ARM ...]
"""

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CPU = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local"), HERE]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
import make_variants as MV  # noqa: E402

ARMS = json.load(open(os.path.join(HERE, "plans", "arms.json")))      # arm -> {"file": ..., "set": {...}}
TOY = dict(probe_budget=20_000, fallback_budget=20_000, deep_budget=20_000, deep_k=2, round_budget=10_000,
           round_cost=3, look_budget=5_000, levels=2, search_s=20)
# sp2's official run's own routines for what the book's constructions call (records/runner/..._official.json)
DESIGNED = [("lt", "pair", "T.lt.pair.AY", "CLC P[SBB.Y INC]"),
            ("red", "pair", "T.red.pair.AY", "STA.S T.lt.pair.AY SEL.S"),
            ("add", "pair", "T.add.pair.AY", "CLC P[ADC.Y INC]"),
            ("red", "x+dn", "T.red.x+dn.AY", "B[T.red.pair.AY]")]


def rows(n, gen):
    """n made-up (N, x) rows as digit strings LSB first, 8 places: N of 2-8 digits, x < N."""
    Nv = torch.randint(10, 10 ** 8, (n,), generator=gen)
    xv = (torch.rand(n, generator=gen) * Nv).long()
    dig = lambda v: torch.stack([(v // 10 ** j) % 10 for j in range(8)], 1)
    return dig(Nv), dig(xv)


def load(arm):
    spec = ARMS[arm]
    sets = [f"{k}={v}" for k, v in spec.get("set", {}).items()]
    return LW.load_module(os.path.join(CPU, spec["file"]), sets)


def climb_of(sc, **kw):
    return sc.SelfClimb(sc.exact_cells("cpu"), "cpu", sc.PR, chunk=1 << 13, **dict(sc.SP, **dict(TOY, **kw)))


def designed_pool(sc, data):
    """The pool the climb poses at its first level once the constructions' routines exist (DESIGNED)."""
    c = climb_of(sc)
    for law, pat, name, prog in DESIGNED:
        c.lib[name] = sc.parse(prog)
        c.ops.add(law, pat, name, ("A", "Y"))
    pool, roots = c.pose(data)
    return pool


TOY_PATTERNS = ["pair", "x+n", "x+dn"]   # the rules' shared toy level: its patterns, with DESIGNED given
_SHARED = {}


def shared_level(data):
    """One toy level of the control, probed up to its choice (DESIGNED given, so that every rule has learnable
    candidates of several concepts at toy budgets): the state every rule arm then classifies and chooses from."""
    if "level" not in _SHARED:
        sc = load("control")
        c = climb_of(sc, patterns=TOY_PATTERNS, deep_k=0, comp_body=2, max_body=2, fallback_budget=0)
        for law, pat, name, prog in DESIGNED:
            c.lib[name] = sc.parse(prog)
            c.ops.add(law, pat, name, ("A", "Y"))
        t0 = time.time()
        while not (c.phase == "probe" and c.queue == [] and c.pool):
            c.step(data, answers=None)
        _SHARED["level"] = c
        print(f"     (the shared toy level: {len(c.pool)} candidates probed in {time.time() - t0:.1f}s)", flush=True)
    return _SHARED["level"]


def check(arm, data):
    sc = load(arm)
    c = climb_of(sc)
    notes = [f"rule {c.rule}, {len(c.laws)} laws, {len(c.patterns)} patterns"]
    if "control" not in _SHARED:
        _SHARED["control"] = designed_pool(load("control"), data)
    ctl = _SHARED["control"]
    pool = designed_pool(sc, data)
    law_of = lambda n: n.split("/")[0]
    pat_of = lambda n: n.split("/")[1].split("@")[0]
    if arm.startswith("no_law_"):
        gone = next(l for l in MV.LAWS if f"no_law_{MV.SLUG.get(l, l)}" == arm)
        assert gone not in c.laws and len(c.laws) == len(MV.LAWS) - 1, c.laws
        want = [n for n in ctl if law_of(n) != gone]
        assert pool == want, (len(pool), len(want), sorted(set(pool) ^ set(want))[:6])
        notes.append(f"designed pool {len(pool)} = the control's {len(ctl)} minus {len(ctl) - len(want)} {gone} "
                     f"candidates")
        return notes
    if arm.startswith("no_pat_"):
        gone = next(p for p in MV.PATTERNS if f"no_pat_{MV.SLUG.get(p, p)}" == arm)
        assert gone not in c.patterns and len(c.patterns) == len(MV.PATTERNS) - 1, c.patterns
        want = [n for n in ctl if pat_of(n) != gone]
        assert pool == want, (len(pool), len(want), sorted(set(pool) ^ set(want))[:6])
        notes.append(f"designed pool {len(pool)} = the control's {len(ctl)} minus {len(ctl) - len(want)} on {gone}")
        return notes
    assert pool == ctl, (len(pool), len(ctl))
    assert c.laws == list(MV.LAWS) and c.patterns == list(MV.PATTERNS)
    want_rule = "lpca" if arm == "control" else arm[len("rule_"):]
    assert c.rule == want_rule, (c.rule, want_rule)
    notes.append(f"designed pool {len(pool)}: the control's")
    # the rule on the shared toy level: classify, (lookaheads), choose, with made-up answers
    base = shared_level(data)
    keep = {"rule", "_variants", "_want_variants"}
    c.__dict__.update({k: v for k, v in base.__dict__.items() if k not in keep})
    c.history, c.records, c.events, c.lib, c.corpus = [], [], [], dict(base.lib), dict(base.corpus)
    Nd, Xd = data
    ans = (Nd[:64], Xd[:64], Xd[:64])                       # a made-up answer pool (no program reproduces it)
    t0 = time.time()
    lines = []
    for _ in range(80):
        line = c.step(data, answers=ans)
        lines.append(line)
        if line and " choose (" in line:
            break
    assert f"choose ({c.rule})" in lines[-1], lines[-3:]
    h = c.history[0]
    learn = list(h["learnable"])
    picks = h["picks"]
    assert picks and all(p in learn for p in picks), (picks, learn)
    concepts = {(law_of(n), pat_of(n)) for n in picks}
    if c.rule == "random_ok":
        assert len(picks) == min(c.K, len(learn)) and len(set(picks)) == len(picks), picks
    if c.rule in ("rca", "lpcc", "lpca", "lpc"):
        assert len(concepts) <= c.K, concepts
    if c.rule == "lpcc":                                    # every binding of a picked concept at its cheapest cost
        for law, pat in concepts:
            costs = {n: v[0] for n, v in h["learnable"].items() if (law_of(n), pat_of(n)) == (law, pat)}
            got = {n for n in picks if (law_of(n), pat_of(n)) == (law, pat)}
            assert got == {n for n, v in costs.items() if v == min(costs.values())}, (law, pat, costs, picks)
    if c.rule == "lpc":
        assert len(picks) == len(concepts), picks
    if c.rule in ("ct1", "ct2a2"):
        assert any(l and " look: " in l for l in lines), lines
        if c.rule == "ct1":
            assert set(h["scores"]["singles"]) == set(learn), h["scores"]
    if c.rule == "nov":
        nv = h["scores"]["novelty"]
        assert set(nv) == set(learn) and picks == sorted(learn, key=lambda n: -nv[n])[:c.K], (nv, picks)
        notes.append(f"novelty (top 3) {dict(sorted(nv.items(), key=lambda kv: -kv[1])[:3])}")
    notes.append(f"toy level: {len(learn)} learnable of {len(c.pool)}; picks {picks}")
    notes.append(f"{len(lines)} climb units in {time.time() - t0:.1f}s")
    return notes


def main(argv):
    arms = argv or [a for a in ARMS if not a.startswith("book_")]      # (the books: check_books.py)
    gen = torch.Generator().manual_seed(0)
    data = rows(512, gen)
    bad = 0
    t0 = time.time()
    for arm in arms:
        try:
            notes = check(arm, data)
            print(f"ok   {arm}\n       " + "\n       ".join(notes), flush=True)
        except Exception as exc:
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {arm}: {exc!r}"[:600], flush=True)
    print(f"{len(arms) - bad} of {len(arms)} arms ok in {time.time() - t0:.0f}s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
