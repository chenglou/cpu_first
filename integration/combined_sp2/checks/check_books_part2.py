"""sp_ablation/check_books.py's part 2 on the combined file: every concept the generated book poses, in every binding,
at the three settings: every target it gives is the law's value on the probe's numbers by Python's integers, and the
addition's check passes the true sums and fails wrong ones. check_books.py runs it on book_gen.py with sp2's
routines for what the book's constructions call (check_variants.DESIGNED); this file has no SEL.S, so its link is
the if block's (GENERIC_OPS.md §1.1: STA.S T.lt C[LDA.S]); the other three are sp2's. A check only: the training file
never computes a target this way. Called by local_combined.py's `book` part."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CPU = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [os.path.join(CPU, "integration", "sp_ablation"), os.path.join(CPU, "wider_menus", "local")]
import torch  # noqa: E402
import check_books as CB  # noqa: E402  (its truth and as_int)
import check_variants as CV  # noqa: E402  (its rows)

DESIGNED = [("lt", "pair", "T.lt.pair.AY", "CLC P[SBB.Y INC]"),
            ("red", "pair", "T.red.pair.AY", "STA.S T.lt.pair.AY C[LDA.S]"),
            ("add", "pair", "T.add.pair.AY", "CLC P[ADC.Y INC]"),
            ("red", "x+dn", "T.red.x+dn.AY", "B[T.red.pair.AY]")]


def run(M) -> int:
    """(On a file with SEL.S, check_variants' own DESIGNED: sp2's link.)"""
    t0 = time.time()
    data = CV.rows(512, torch.Generator().manual_seed(0))
    lib, ops = {}, M.Ops()
    for law, pat, name, prog in (CV.DESIGNED if "SEL.S" in M.PRIMS else DESIGNED):
        lib[name] = M.parse(prog)
        ops.add(law, pat, name, ("A", "Y"))
    st = [M.Setting(B, W, M.exact_cells("cpu")) for B, W in ((10, 1), (10, 2), (10, 3))]
    concepts, checked = {}, 0
    for law, (_, pats) in M.LAW_BOOK.items():
        for pat in pats:
            ar = len(M.PATTERNS[pat][0][0])
            for bd in M.bindings(ar):
                r = M.Root(pat, bd, st, 8, data, ops, lib, "cpu", 0)
                g = r.goals(ops, lib, [law])
                if not g:
                    continue
                concepts[(law, pat)] = concepts.get((law, pat), 0) + 1
                kinds = M.PATTERNS[pat][0]
                for d, t in zip(r.draws, g[0].targets):
                    terms = [[d.term(kd[j], ops, lib) for kd in kinds] for j in range(ar)]
                    for i in range(d.n):
                        ki = int(d.kind[i])
                        vals = [CB.as_int(terms[j][ki][i]) for j in range(ar)]
                        if law == "add":
                            continue                                   # below: the check
                        want = CB.truth(CB.RENAME.get(law, law), vals)      # (sp2's acc.mul, acc.add)
                        if "c" in t and bool(t["cm"][i]):
                            assert int(t["c"][i]) == want, (law, pat, bd, i, vals, int(t["c"][i]), want)
                            checked += 1
                        if "A" in t and bool(t["Am"][i]):
                            assert want is not None and CB.as_int(t["A"][i]) == want, \
                                (law, pat, bd, i, vals, CB.as_int(t["A"][i]), want)
                            checked += 1
                    if law == "add" and "check" in t:                  # the true sums pass, an off-by-N sum fails
                        x, N = d.x, d.N
                        s = M.repeat_add(ops, d.st, lib, x, N, torch.ones(d.n, dtype=torch.long), d.gen, "cpu")
                        wrong = M.repeat_add(ops, d.st, lib, x, N, torch.full((d.n,), 2), d.gen, "cpu")
                        ok = t["check"](torch.stack([s, wrong]).to(torch.uint8), lib)
                        assert ok.tolist() == [True, False], (pat, bd, ok)
                        checked += 1
    print(f"book: check_books part 2 on this file: {len(concepts)} concepts posed (every binding), {checked} probe "
          f"targets equal the law's value ({time.time() - t0:.0f}s)", flush=True)
    return 0
