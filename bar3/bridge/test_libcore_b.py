"""libcore_b against libcore (integration/lib/libcore.py, what s3i inlines): with the defaults (s3i's three cells and
SEL.S) every primitive, a real vocabulary and the round machine compute exactly what libcore's do, on random machines;
then, with other cells (invented ones, or s3i's under other names), libcore_b's lockstep machine, its item-by-item
machine (step), its round machine and its one-hot machine agree with each other. Local, small (about a minute).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.test_libcore_b
"""
import time

import torch
import torch.nn.functional as F

from integration.lib import libcore as L
from bar3.bridge import libcore_b as LB

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()


def same(a, b):
    return all(bool(torch.equal(getattr(a, f), getattr(b, f))) for f in ("regs", "c", "h", "k", "p"))


def states(M, st, n, seed):
    g = M.gen_on(dev, seed)
    return M.random_states(st, n, 6, g, dev)


def cross(name, items, lib, P=4, n=3000, seed=1):
    """libcore's item-by-item machine against libcore_b's item-by-item and lockstep machines."""
    for W in (1, 2):
        st, stb = L.Setting(10, W, L.Cells(10, dev)), LB.Setting(10, W, LB.Cells(10, dev))
        b, bb = states(L, st, 24, seed), states(LB, stb, 24, seed)
        assert same(b, bb)
        g = torch.Generator().manual_seed(seed)
        ior = torch.randint(0, len(items), (n,), generator=g).to(dev)
        src = torch.randint(0, 24, (n,), generator=g).to(dev)
        ref = L._apply_grouped(st, b.take(src), ior, items, lib)
        got = LB._apply_grouped(stb, bb.take(src), ior, items, lib)
        LB.LOCK_COST, saved = -10 ** 9, LB.LOCK_COST
        try:
            lock = LB.apply_items(stb, bb.take(src), ior, items, lib)
        finally:
            LB.LOCK_COST = saved
        ok = same(ref, got) and same(ref, lock)
        print(f"{name} W={W}: {len(items)} items, {n} rows: {'same' if ok else 'DIFFERENT'} ({time.time() - t0:.1f}s)")
        assert ok, name


def inner(name, M, items, lib, n=3000, seed=2):
    """One module's lockstep machine against its own item-by-item machine."""
    for W in (1, 2, 3):
        st = M.Setting(10, W, M.Cells(10, dev))
        b = states(M, st, 24, seed)
        g = torch.Generator().manual_seed(seed)
        ior = torch.randint(0, len(items), (n,), generator=g).to(dev)
        src = torch.randint(0, 24, (n,), generator=g).to(dev)
        ref = M._apply_grouped(st, b.take(src), ior, items, lib)
        M.LOCK_COST, saved = -10 ** 9, M.LOCK_COST
        try:
            lock = M.apply_items(st, b.take(src), ior, items, lib)
        finally:
            M.LOCK_COST = saved
        ok = same(ref, lock)
        print(f"{name} W={W}: {len(items)} items: {'same' if ok else 'DIFFERENT'} ({time.time() - t0:.1f}s)")
        assert ok, name


def rounds(name, M, progs, lib, rows=256, P=16, seed=4):
    """The round machine against run_prog (the item-by-item machine) from the same fresh draws."""
    cells = M.Cells(10, dev)
    st = M.round_settings(cells, P)
    g = torch.Generator().manual_seed(seed)
    for prog in progs:
        Xd = torch.randint(0, 10, (rows, P // 2), generator=g).to(dev)
        Nd = torch.randint(0, 10, (rows, P // 2), generator=g).to(dev)
        go = torch.ones(rows, dtype=torch.bool, device=dev)
        m = M.round_machine(st, prog, lib, rows, dev)
        got = m.run(Xd, Nd, go, rows, M.gen_on(dev, 9)).clone()
        ref = M.run_prog(st, prog, lib, {"X": M.pad(Xd, P), "N": M.pad(Nd, P)}, M.gen_on(dev, 9), dev)[0]
        ok = bool(torch.equal(got.long(), ref.long()))
        print(f"{name} round machine {M.show(prog)[:60]}: {'same' if ok else 'DIFFERENT'}")
        assert ok, name


def one_hot_vs_hard(name, M, prog, lib, tabs, rows=64, P=8, seed=5):
    """The one-hot machine (exact tables, as the prediction path's straight-through values) against run_prog from a
    clean start (zeros), as round_ste runs it."""
    cells = M.Cells(10, dev)
    st = M.round_settings(cells, P)
    g = torch.Generator().manual_seed(seed)
    Xd = torch.randint(0, 10, (rows, P), generator=g).to(dev)
    Nd = torch.randint(0, 10, (rows, P), generator=g).to(dev)
    z = torch.zeros(rows, P, dtype=torch.long, device=dev)
    hot = lambda d: F.one_hot(d, 10).float()   # noqa: E731
    regs = [hot(z), hot(Xd), hot(z), hot(Nd), hot(z)]
    z2 = F.one_hot(torch.zeros(rows, dtype=torch.long, device=dev), 2).float()
    z10 = F.one_hot(torch.zeros(rows, dtype=torch.long, device=dev), 10).float()
    m = M.SteMachine(tabs, P, lambda x: x) if M is LB else M.SteMachine(*tabs, P, lambda x: x)
    out = m.run(regs, z2, z10, z10, prog, lib)[0][0].argmax(-1)
    b = M.Batch(torch.stack([z, Xd, z, Nd, z]).permute(0, 2, 1)[None].to(torch.uint8),
                torch.zeros(1, rows, dtype=torch.uint8, device=dev), torch.zeros(1, rows, dtype=torch.uint8, device=dev),
                torch.zeros(1, rows, dtype=torch.uint8, device=dev), torch.zeros(1, dtype=torch.long, device=dev))
    M.run_items(st, b, prog, lib)
    ok = bool(torch.equal(out, b.regs[0, 0].T.long()))
    print(f"{name} one-hot machine {M.show(prog)[:60]}: {'same' if ok else 'DIFFERENT'}")
    assert ok, name


def exact_tabs(cells, names, B=10):
    """Straight-through tables of exact cells as one-hot rows: digit B columns, then the flag (2) or carry (B)."""
    out = {}
    for cs in LB.CELLS:
        if cs.name not in names:
            continue
        t = cells.table_of(cs)
        fw = 2 if "c" in cs.writes else B
        out[cs.name] = torch.cat([F.one_hot(t[:, 0], B), F.one_hot(t[:, 1].clamp(max=fw - 1), fw)], 1).float().to(dev)
    return out


# 1. the defaults: libcore exactly
lib = {"T.lt": L.parse("CLC LDA.X P[SBB.N INC]"), "T.add": L.parse("CLC LDA.X P[ADC.N INC]"),
       "T.mac": L.parse("LDH.Y CLK P[MUL.X INC]")}
cross("every primitive (defaults)", list(LB.PRIMS), {})
voc = [c.rep for c in L.enumerate_items(list(L.PRIMS) + list(lib), lib, [L.Setting(10, 1, L.Cells(10, dev))], dev,
                                        max_body=2)]
cross("a vocabulary with loops (defaults)", voc[:3000], lib)
r11 = {"T.mul": L.parse("CLA W[T.mac SHL.X SHR.Y]"), "T.mac": lib["T.mac"],
       "T.modx": L.parse("LDA.X STA.S T.lt SEL.S"), "T.lt": lib["T.lt"], "T.modB": L.parse("B[T.modx STA.X]"),
       "T.reduce": L.parse("CLK CLA P[BRG.X STA.Y T.modB]"), "T.add": lib["T.add"]}
progs = [L.parse("LDA.X STA.Y T.mul STA.X T.reduce"), L.parse("T.add SHL.X T.lt SEL.S"), L.parse("T.mac BRG.X")]
rounds("libcore", L, progs, r11)
rounds("libcore_b (defaults)", LB, progs, r11)
c0 = LB.Cells(10, dev)
one_hot_vs_hard("libcore_b (defaults)", LB, progs[0], r11, exact_tabs(c0, ("SBB", "ADC", "MUL")))

# 2. other cells: s3i's three tables under the invented cells' names, plus a per-place select (no SEL.S)
c = L.Cells(10, dev)
sel_t = torch.zeros(200, 2, dtype=torch.long)    # (CellSpec tables live on the host; Cells moves them)
for a in range(10):
    for x in range(10):
        for f in range(2):
            sel_t[(a * 10 + x) * 2 + f] = torch.tensor([x if f else a, f])
lt_t = torch.stack([c.sbb[:, 0] * 0, c.sbb[:, 1]], 1)
cells = [LB.CellSpec("SUB", "axc", ("A", "c"), c.sbb), LB.CellSpec("ADD", "axc", ("A", "c"), c.adc),
         LB.CellSpec("MAC", "axhk", ("A", "k"), c.mul), LB.CellSpec("SELC", "axc", ("A",), sel_t),
         LB.CellSpec("LT", "axc", ("c",), lt_t)]
prims = LB.set_instructions(cells, sel=False)
print("instruction set:", len(prims), prims)
inner("every primitive (invented names)", LB, list(LB.PRIMS), {})
libn = {"T.lt": LB.parse("CLC LDA.X P[SUB.N INC]"), "T.mac": LB.parse("LDH.Y CLK P[MAC.X INC]"),
        "T.sel": LB.parse("P[SELC.S INC]")}
vocn = [cl.rep for cl in LB.enumerate_items(list(LB.PRIMS) + list(libn), libn, [LB.Setting(10, 1, LB.Cells(10, dev))],
                                            dev, max_body=2)]
inner("a vocabulary with loops (invented names)", LB, vocn[:3000], libn)
r11n = {"T.mul": LB.parse("CLA W[T.mac SHL.X SHR.Y]"), "T.mac": libn["T.mac"],
        "T.modx": LB.parse("LDA.X STA.S T.lt T.sel"), "T.lt": libn["T.lt"], "T.sel": libn["T.sel"],
        "T.modB": LB.parse("B[T.modx STA.X]"), "T.reduce": LB.parse("CLK CLA P[BRG.X STA.Y T.modB]"),
        "T.add": LB.parse("CLC LDA.X P[ADD.N INC]")}
pn = [LB.parse("LDA.X STA.Y T.mul STA.X T.reduce"), LB.parse("CLC P[LT.N INC] P[SELC.X INC]"), LB.parse("T.mac BRG.X")]
rounds("libcore_b (invented names)", LB, pn, r11n)
one_hot_vs_hard("libcore_b (invented names)", LB, pn[0], r11n, exact_tabs(LB.Cells(10, dev),
                                                                       ("SUB", "ADD", "MAC", "SELC", "LT")))
# the same round on both instruction sets: the same answers
st = L.round_settings(L.Cells(10, dev), 16)
stb = LB.round_settings(LB.Cells(10, dev), 16)
g = torch.Generator().manual_seed(8)
Xd, Nd = torch.randint(0, 10, (128, 16), generator=g).to(dev), torch.randint(0, 10, (128, 16), generator=g).to(dev)
a = L.run_prog(st, progs[0], r11, {"X": Xd, "N": Nd}, L.gen_on(dev, 1), dev)[0]
b = LB.run_prog(stb, pn[0], r11n, {"X": Xd, "N": Nd}, LB.gen_on(dev, 1), dev)[0]
print("x^2 mod N's round on s3i's instructions and on the renamed ones:", "same" if torch.equal(a, b) else "DIFFERENT")
assert torch.equal(a, b)
print(f"all checks passed ({time.time() - t0:.1f}s)")
