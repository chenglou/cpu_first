"""Round 2's first change, a probed root's laws sharing one acceptance Root (SHARED_ACCEPT; COMBINED.md §8), must leave
the climb exactly as it was. On every pattern of the generated book in every binding, with every law it poses there
(as SelfClimb.pose poses them; the first round's tiny core's library as the learner's routines, its 4,096 rows),
each law's acceptance is built twice, by the first round's file and by the second round's with DOUBLE_COUNTS off
(SHARED_ACCEPT alone), each by its own SelfClimb.acceptors on a fresh climb: the same probes (every setting's
machines: inputs, random starts, flags, pointer) and targets (values and masks, and a check where the law has one),
and the same verdicts on the candidates: the first round's tiny core's level-3 programs for that law and root, every
routine of the library, and random programs. Local, CPU.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_sp2/checks/round2_fix1.py [PATTERN ...]
"""
import ast
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COMB = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(os.path.dirname(COMB)), "learned_remainder", "chain_local", "stub"), COMB]
import torch  # noqa: E402
import core_c  # noqa: E402

OLD = os.path.join(COMB, "submission_stage3_combined.py")
NEW = os.path.join(COMB, "research", "submission_stage3_combined2_DOUBLE_COUNTS-False.py")


def setup(path):
    M, md5 = core_c.load(path, [])
    d = json.load(open(os.path.join(COMB, "records", "core_combined.json")))      # the first round's tiny core's
    lib = {k: M.parse(v) for k, v in d["lib"].items()}                            # library, at its end
    ops = M.Ops()
    for k in d["lib"]:
        if k.startswith("T.") and k.count(".") >= 3:
            parts = k[2:].split(".")
            ops.add(".".join(parts[:-2]), parts[-2], k, tuple(parts[-1]))
    return M, md5, lib, ops


def level3_programs():
    """The first round's tiny core's programs at level 3, by law@root."""
    r = json.load(open(os.path.join(COMB, "records", "core_combined.json")))
    out = {}
    for rec in r["records"]:
        if not isinstance(rec, dict) or rec.get("level") != 3 or "laws" not in rec:
            continue
        laws = rec["laws"] if isinstance(rec["laws"], dict) else ast.literal_eval(rec["laws"])
        for k, v in laws.items():
            out.setdefault(k, []).extend(v.get("programs") or [])
    return out


def same_probs(a, b):
    if len(a.probs) != len(b.probs) or a.ok != b.ok or a.checked != b.checked:
        return False
    for (s1, b1, g1), (s2, b2, g2) in zip(a.probs, b.probs):
        if (s1.B, s1.W, s1.P, getattr(s1, "above", False)) != (s2.B, s2.W, s2.P, getattr(s2, "above", False)):
            return False
        if not all(torch.equal(getattr(b1, f), getattr(b2, f)) for f in ("regs", "c", "h", "k", "p")):
            return False
        t1, t2 = g1.targets[0], g2.targets[0]
        if set(t1) != set(t2) or not all(torch.equal(t1[k], t2[k]) for k in t1 if torch.is_tensor(t1[k])):
            return False
    return True


def main():
    t0 = time.time()
    (M1, h1, lib1, ops1), (M2, h2, lib2, ops2) = setup(OLD), setup(NEW)
    assert M1.SP == {k: v for k, v in M2.SP.items()} and not hasattr(M1, "SHARED_ACCEPT") and M2.SHARED_ACCEPT \
        and not M2.DOUBLE_COUNTS
    rows = core_c.load_rows(4096, seed=0)
    data = (core_c.digits(rows, "modulus"), core_c.digits(rows, "x"))
    prog3 = level3_programs()
    rng = random.Random(0)
    routines = sorted(k for k in lib1 if k.startswith("T."))
    toks = list(M1.PRIMS) + routines
    rand = [tuple(rng.choice(toks) for _ in range(rng.randint(1, 4))) for _ in range(20)]
    pats = sys.argv[1:] or list(M1.PATTERNS)
    n_acc = n_same = n_dec = n_diff = n_yes = 0
    secs = [0.0, 0.0]
    for pat in pats:
        laws = [l for l in M1.LAW_BOOK if pat in M1.LAW_BOOK[l][1]]
        arity = len(M1.PATTERNS[pat][0][0])
        for bd in M1.bindings(arity):
            accs = []
            for i, (M, lib, ops) in enumerate(((M1, lib1, ops1), (M2, lib2, ops2))):
                c = M.SelfClimb(M.exact_cells("cpu"), "cpu", M.PR, chunk=1 << 18, **M.SP)
                c.ops, c.lib = ops, lib
                try:
                    r = M.Root(pat, bd, c.settings, c.NP, data, ops, lib, "cpu", c.seed)
                except M.NotPosable:
                    accs.append(None)
                    continue
                r.goal_list = r.goals(ops, lib, laws)
                t1 = time.time()
                accs.append((r, c.acceptors(r, data)))
                secs[i] += time.time() - t1
            if accs[0] is None or accs[1] is None:
                assert accs[0] is None and accs[1] is None, (pat, bd)
                continue
            (r1, a1), (r2, a2) = accs
            assert [g.name for g in r1.goal_list] == [g.name for g in r2.goal_list]
            for g, x, y in zip(r1.goal_list, a1, a2):
                n_acc += 1
                n_same += same_probs(x, y)
                cands = [M1.parse(p) for p in prog3.get(f"{g.name}/{r1.key}", [])] + [(k,) for k in routines] + rand
                v1, v2 = x.many(cands, lib1), y.many(cands, lib2)
                n_dec += len(cands)
                n_diff += sum(p != q for p, q in zip(v1, v2))
                n_yes += sum(v1)
        print(f"{pat}: {n_acc} acceptances so far, {n_same} with the same probes and targets; {n_dec} verdicts, "
              f"{n_diff} different ({n_yes} accepted); setup {secs[0]:.0f} s (round 1) against {secs[1]:.0f} s "
              f"(shared) ({time.time() - t0:.0f} s)", flush=True)
    print(f"files: round 1 {h1}, round 2 with SHARED_ACCEPT alone {h2}")
    print("FIX 1 REPLAYS" if n_same == n_acc and n_diff == 0 else "FIX 1 DIFFERS", flush=True)


if __name__ == "__main__":
    main()
