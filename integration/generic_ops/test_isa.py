"""The generic instruction sets (GENERIC_OPS.md) on every machine that runs programs, against run_items:
  1. libcore_g with "sel,zero,brg" has libcore.py's primitives, in order;
  2. the lockstep machine (apply_items, forced on every bucket) against the item-by-item one (_apply_grouped, run_items):
     every primitive, the vocabulary of loops up to body cost 2 (C blocks among them), and a library of routines that
     use C blocks and shifts through k, nested in loops and in each other, at the climb's settings and at 16 places;
  3. the round machine (round_machine) against run_prog at 16 places, on random programs and on the hand-written round;
  4. the one-hot machine (SteMachine, exact one-hot tables) against run_items from clean starts;
  5. the hand-written pieces each instruction set allows (the link, the chain, the multiply-accumulate row, the multiply,
     the reduction, the round) on H1's T = 1 rows at full size: round_ok, and the one-hot machine's answers.
Local and small on a CPU (about 3 minutes, under 1 GB); on a GPU (modal_g.py, MODULE) the lockstep's and the round
machine's CUDA graphs run too.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.generic_ops.test_isa [ISA ...]
"""
import sys
import time

import torch

from integration.generic_ops import isa as I
from integration.self_practice.core import digits, load_rows

dev = "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time()
ISAS = sys.argv[1:] or ["sel,zero,brg", "if,k", "if1,k", "ldhc,k", "sel,k", "if,zero,brg"]

# the pieces of the round per instruction set: (link in A against Y, the reduction X mod A); the multiply's row and the
# multiply are the same in every set (SELF_PRACTICE.md §3.4's names)
LINK = {"sel": "STA.S T.lt SEL.S", "if": "STA.S T.lt C[LDA.S]", "if1": "STA.S T.lt C[LDA.S]",
        "ldhc": "T.lt LDH.C CLK P[MUL.Y INC]"}
REDUCE = {True: "STA.Y P[BRG.X T.chain]", False: "STA.Y P[SHL.X SHL.A T.chain]"}
ZERO_K = {"k": "CLK ", "zero": ""}        # a shift through k needs k clear first where it must bring in 0


def lib_for(L, isa):
    cond, shift, bring = I.parse_isa(isa)
    src = {"T.lt": "CLC P[SBB.Y INC]", "T.link": LINK[cond], "T.chain": "B[T.link]",
           "T.mac": "LDH.Y CLK P[MUL.X INC]", "T.mul": "CLA W[T.mac INC SHL.X]",
           "T.red": REDUCE[bring] if (bring or shift == "k") else None,
           "T.ten": ZERO_K[shift] + "SHL.A SHL.X SHR.Y"}
    return {k: L.parse(v) for k, v in src.items() if v is not None}


def same(a, b):
    return all(bool(torch.equal(getattr(a, f), getattr(b, f))) for f in ("regs", "c", "h", "k", "p"))


def check_lockstep(L, name, st, items, lib, gen, S=24, NP=6, rows=3000):
    if not items:
        return
    b = L.random_states(st, S, NP, gen, dev)
    n = min(rows, len(items) * S)
    item_of_row = torch.randint(0, len(items), (n,), generator=gen, device=dev)
    src = b.take(torch.randint(0, S, (n,), generator=gen, device=dev))
    keep = src.copy()
    ref = L._apply_grouped(st, src, item_of_row, items, lib)
    one = src.copy()                                     # the reference again, row by row with run_items
    for r in range(0, n, max(1, n // 40)):
        x = one.take(torch.tensor([r], device=dev))
        L.run_items(st, x, (items[int(item_of_row[r])],), lib)
        assert same(x, ref.take(torch.tensor([r], device=dev))), (name, L.show((items[int(item_of_row[r])],)))
    saved = L.LOCK_COST
    L.LOCK_COST = -10 ** 9
    try:
        new = L.apply_items(st, src, item_of_row, items, lib)
    finally:
        L.LOCK_COST = saved
    ok = same(ref, new) and same(src, keep)
    print(f"  lockstep {name}: {len(items)} items, {n} rows: {'same' if ok else 'DIFFERENT'} ({time.time() - t0:.0f}s)",
          flush=True)
    assert ok, name


def run_isa(isa):
    L, S = I.load(isa)
    cond, shift, bring = I.parse_isa(isa)
    print(f"{isa}: {len(L.PRIMS)} primitives {L.PRIMS}; loops {L.LOOP_KINDS}", flush=True)
    if isa == "sel,zero,brg":
        from integration.lib import libcore as L0
        assert L.PRIMS == L0.PRIMS and L.LOOP_KINDS == L0.LOOP_KINDS
        print("  = libcore.py's primitives and loops, in order")
    assert S.WHOLE and all(t in L.PRIMS for t in S.WHOLE)
    gen = L.gen_on(dev, 3)
    cells = L.Cells(10, dev)
    sets = [L.Setting(10, 1, cells), L.Setting(10, 2, cells), L.Setting(10, 8, cells, 16)]
    lib = lib_for(L, isa)
    for st in sets:
        check_lockstep(L, f"primitives {st} P={st.P}", st, list(L.PRIMS), {}, gen)
    voc = [c.rep for c in L.enumerate_items(list(L.PRIMS), {}, sets[:2], dev, max_body=2)]
    print(f"  vocabulary to body cost 2: {len(voc)} items, {sum(1 for v in voc if L.is_loop(v) and v[1] == 'C')} "
          f"C blocks ({time.time() - t0:.0f}s)", flush=True)
    for st in sets[:2]:
        check_lockstep(L, f"vocabulary {st}", st, voc, {}, gen)
    routines = sorted(lib)
    nested = [L.parse(s) for s in ("C[T.link]", "B[C[T.lt]]", "C[C[LDA.S] STA.X]", "P[T.link INC]", "C[T.mac SHL.Y]",
                                   "T.chain SHL.X SHL.A", "C[SHR.X] SHL.Y", "W[C[ADC.X] INC]")]
    nested = [p[0] if len(p) == 1 else ("REP", L.ONE, p) for p in nested if cond in ("if", "if1") or "C[" not in L.show(p)]
    comp = [c.rep for c in L.enumerate_items(list(L.PRIMS) + routines, lib, sets[:2], dev, max_body=1)] + nested
    for st in sets:
        check_lockstep(L, f"routines and nests {st}", st, comp, lib, gen, rows=2000)
    # the round machine against run_prog at 16 places
    st = L.round_settings(cells, 16)
    rows = load_rows(64, seed=1, T=1)
    Nd, Xd, Yd = (digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
    progs = [tuple(L.parse("T.mul STA.X LDA.N")) + tuple(L.parse("T.red")) * ("T.red" in lib)]
    for _ in range(40):
        progs.append(tuple(comp[int(i)] for i in torch.randint(0, len(comp), (5,), generator=gen, device=dev)))
    bad = 0
    for prog in progs:
        ref, _ = L.run_prog(st, prog, lib, {"X": L.pad(Xd, 16), "N": L.pad(Nd, 16)}, L.gen_on(dev, 5), dev)
        m = L.round_machine(st, prog, lib, Xd.shape[0], dev)
        got = m.run(Xd, Nd, torch.ones(Xd.shape[0], dtype=torch.bool, device=dev), Xd.shape[0], L.gen_on(dev, 5))
        bad += not bool(torch.equal(ref, got.long()))
    print(f"  round machine vs run_prog, {len(progs)} programs at 16 places: {bad} differ ({time.time() - t0:.0f}s)",
          flush=True)
    assert bad == 0
    # the one-hot machine against run_items from clean starts (on the CPU)
    cells, st, Xd, Nd = L.Cells(10, "cpu"), L.round_settings(L.Cells(10, "cpu"), 16), Xd.cpu(), Nd.cpu()
    oh = lambda t, n: torch.nn.functional.one_hot(t.long().flatten(), n).float()
    SBB = torch.cat([oh(cells.sbb_d, 10), oh(cells.sbb_c, 2)], 1)
    ADC = torch.cat([oh(cells.adc_d, 10), oh(cells.adc_c, 2)], 1)
    MUL = torch.cat([oh(cells.mul_lo, 10), oh(cells.mul_hi, 10)], 1)
    ste = L.SteMachine(SBB, ADC, MUL, 16, lambda x: x)
    bad = 0
    for prog in progs[:25]:
        prog = S.unwrap(prog)                            # (the prediction path runs no ONE item)
        n = Xd.shape[0]
        b = L.Batch(torch.zeros(1, 5, 16, n, dtype=torch.uint8), torch.zeros(1, n, dtype=torch.uint8),
                    torch.zeros(1, n, dtype=torch.uint8), torch.zeros(1, n, dtype=torch.uint8),
                    torch.zeros(1, dtype=torch.long))
        b.regs[0, L.X_] = L.pad(Xd, 16).T.to(torch.uint8)
        b.regs[0, L.N_] = L.pad(Nd, 16).T.to(torch.uint8)
        L.run_items(st, b, prog, lib)
        z = torch.zeros(n, 16, 10)
        z[..., 0] = 1
        regs = [z, oh(L.pad(Xd, 16), 10).view(n, 16, 10), z, oh(L.pad(Nd, 16), 10).view(n, 16, 10), z]
        z2, z10 = torch.zeros(n, 2), torch.zeros(n, 10)
        z2[:, 0] = z10[:, 0] = 1
        out = ste.run(regs, z2, z10, z10, prog, lib)
        got = torch.stack([r.argmax(-1) for r in out[0]], 0)            # [5, n, 16]
        exact = all(bool(((r == 0) | (r == 1)).all()) for r in out[0])
        same_ = bool(torch.equal(got.permute(0, 2, 1), b.regs[0].long())) and torch.equal(out[1].argmax(-1), b.c[0].long()) \
            and torch.equal(out[3].argmax(-1), b.k[0].long()) and torch.equal(out[2].argmax(-1), b.h[0].long())
        bad += not (same_ and exact)
    print(f"  one-hot machine vs run_items, 25 programs at 16 places: {bad} differ ({time.time() - t0:.0f}s)", flush=True)
    assert bad == 0
    # the round, hand-written from this instruction set's pieces
    if "T.red" in lib:
        rnd = L.parse("LDA.X STA.Y T.mul STA.X LDA.N T.red")
        rows = load_rows(400, seed=1, T=1)
        ans = tuple(digits(rows, k).to(dev) for k in ("modulus", "x", "result"))
        ok = L.round_ok(rnd, lib, L.Cells(10, dev), ans, 16, dev)
        print(f"  the round {L.show(rnd)} with T.link = {L.show(lib['T.link'])}, T.red = {L.show(lib['T.red'])}: "
              f"round_ok on 400 T = 1 rows {ok}; costs: link {L.cost(lib['T.link'])}, reduction {L.cost(lib['T.red'])}",
              flush=True)
        assert ok
    return L


if __name__ == "__main__":
    for isa in ISAS:
        run_isa(isa)
    print(f"all passed ({time.time() - t0:.0f}s)")
