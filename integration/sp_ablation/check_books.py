"""Local check of the two books (SP_ABLATION.md §5): the generated one (variants/book_gen.py) against sp2's hand-written
one, and the smallest one (variants/book_min.py), with the routines the constructions need given
(check_variants.DESIGNED: sp2's official run's adder, link, comparison and chain):
  1. on every concept sp2 poses (a law on a pattern, sp2's acc.mul and acc.add as the grammar's add.mul and add.add),
     in every binding and at every setting (the probes' and the acceptance's), the generated targets equal sp2's on
     the same probes: the same masks, the same values where they speak (Goal.test and fits read nothing else), the
     same check;
  2. on every concept, sp2's and the new ones, every target the generated book gives is the law's value on the probe's
     numbers, by Python's integers (a check only; the training file never computes a target this way), and the check
     of the addition passes the true sums and fails wrong ones;
  3. the smallest book poses exactly the control's pool restricted to its laws and patterns (make_books.MIN_LAWS on
     MIN_PATTERNS).
Seconds and well under 1 GB.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/sp_ablation/check_books.py
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CPU = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local"), HERE]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
import check_variants as CV  # noqa: E402

RENAME = {"acc.mul": "add.mul", "acc.add": "add.add"}


def setup(sc):
    lib, ops = {}, sc.Ops()
    for law, pat, name, prog in CV.DESIGNED:
        lib[name] = sc.parse(prog)
        ops.add(law, pat, name, ("A", "Y"))
    st = [sc.Setting(B, W, sc.exact_cells("cpu")) for B, W in ((10, 1), (10, 2), (10, 3))]
    return lib, ops, st


def as_int(digits):
    return int("".join(str(int(v)) for v in reversed(digits.tolist())) or "0")


def truth(law, vals):
    a, b = vals[0], vals[1]
    if law == "lt":
        return int(a < b)
    if law == "eq":
        return int(a == b)
    if law == "min":
        return min(a, b)
    if law == "max":
        return max(a, b)
    if law == "red":                      # red.zero, a mod a = 0, as sp2 poses it (also for a = 0)
        return 0 if a == b else (a % b if b else None)
    if law == "div":
        return a // b if b else None
    if law == "sub":
        return a - b
    if law == "mul":
        return a * b
    if law == "add.add":
        return a + (b + vals[2])
    if law == "add.mul":
        return a + b * vals[2]
    raise ValueError(law)


def main():
    t0 = time.time()
    data = CV.rows(512, torch.Generator().manual_seed(0))
    sp2 = CV.load("control")
    gen = LW.load_module(os.path.join(CPU, "integration", "sp_ablation", "variants", "book_gen.py"))
    (lib1, ops1, st1), (lib2, ops2, st2) = setup(sp2), setup(gen)
    same = checked = 0
    # 1. sp2's concepts: the same targets
    for law, (_, pats) in sp2.LAW_BOOK.items():
        for pat in pats:
            ar = len(sp2.PATTERNS[pat][0][0])
            for bd in sp2.bindings(ar):
                r1 = sp2.Root(pat, bd, st1, 8, data, ops1, lib1, "cpu", 0)
                r2 = gen.Root(pat, bd, st2, 8, data, ops2, lib2, "cpu", 0)
                g1, g2 = r1.goals(ops1, lib1, [law]), r2.goals(ops2, lib2, [RENAME.get(law, law)])
                assert bool(g1) == bool(g2), (law, pat, bd, bool(g1), bool(g2))
                for t1, t2 in zip(g1[0].targets if g1 else [], g2[0].targets if g2 else []):
                    assert set(t1) == set(t2), (law, pat, bd, set(t1), set(t2))
                    for v, m in (("A", "Am"), ("c", "cm")):           # the same masks, the same values under them
                        if v in t1:                                    # (no search reads a value outside its mask)
                            assert torch.equal(t1[m], t2[m]), (law, pat, bd, m)
                            assert torch.equal(t1[v][t1[m]], t2[v][t2[m]]), (law, pat, bd, v)
                same += 1
    print(f"1. sp2's {same} (concept, binding) roots: generated targets equal sp2's at all 3 settings "
          f"({time.time() - t0:.0f}s)", flush=True)
    # 2. every concept of the generated book: its targets are the law's values
    concepts = {}
    for law, (_, pats) in gen.LAW_BOOK.items():
        for pat in pats:
            ar = len(gen.PATTERNS[pat][0][0])
            for bd in gen.bindings(ar):
                r = gen.Root(pat, bd, st2, 8, data, ops2, lib2, "cpu", 0)
                g = r.goals(ops2, lib2, [law])
                if not g:
                    continue
                concepts[(law, pat)] = concepts.get((law, pat), 0) + 1
                kinds = gen.PATTERNS[pat][0]
                for d, t in zip(r.draws, g[0].targets):
                    terms = [[d.term(kd[j], ops2, lib2) for kd in kinds] for j in range(ar)]
                    for i in range(d.n):
                        ki = int(d.kind[i])
                        vals = [as_int(terms[j][ki][i]) for j in range(ar)]
                        if law == "add":
                            continue                                   # below: the check
                        want = truth(law, vals)
                        if "c" in t and bool(t["cm"][i]):
                            assert int(t["c"][i]) == want, (law, pat, bd, i, vals, int(t["c"][i]), want)
                            checked += 1
                        if "A" in t and bool(t["Am"][i]):
                            assert want is not None and as_int(t["A"][i]) == want, \
                                (law, pat, bd, i, vals, as_int(t["A"][i]), want)
                            checked += 1
                    if law == "add" and "check" in t:                  # the true sums pass, an off-by-N sum fails
                        x, N = d.x, d.N
                        s = gen.repeat_add(ops2, d.st, lib2, x, N, torch.ones(d.n, dtype=torch.long), d.gen, "cpu")
                        wrong = gen.repeat_add(ops2, d.st, lib2, x, N, torch.full((d.n,), 2), d.gen, "cpu")
                        both = torch.stack([s, wrong]).to(torch.uint8)
                        ok = t["check"](both, lib2)
                        assert ok.tolist() == [True, False], (pat, bd, ok)
                        checked += 1
    new = sorted(k for k in concepts if (k[0], k[1]) not in
                 {(RENAME.get(l, l), p) for l, (_, ps) in sp2.LAW_BOOK.items() for p in ps})
    print(f"2. {len(concepts)} concepts posed (every binding), {checked} probe targets equal the law's value; "
          f"new against sp2's book: {new}", flush=True)
    # 3. the smallest book poses exactly the control's pool restricted to its laws and patterns
    sys.path.insert(0, HERE)
    import make_books as MB
    small = LW.load_module(os.path.join(CPU, "integration", "sp_ablation", "variants", "book_min.py"))
    pool, ctl = CV.designed_pool(small, data), CV.designed_pool(sp2, data)
    want = [n for n in ctl if n.split("/")[0] in MB.MIN_LAWS and n.split("/")[1].split("@")[0] in MB.MIN_PATTERNS]
    assert pool == want, (len(pool), len(want))
    print(f"3. book_min's designed pool: {len(pool)} candidates, exactly the control's restricted to "
          f"{MB.MIN_LAWS} on {MB.MIN_PATTERNS}", flush=True)
    print(f"ok in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
