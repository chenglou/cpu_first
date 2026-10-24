"""The count loop on all four machines (LEARNED_LOOP.md §2): run_items (the reference), apply_items (the search's,
lockstep or item by item), OuterMachine (phase D's, RoundMachine stretches) and SteMachine (the prediction path's, on
one-hot digits) must give the same state for programs with count loops at their top level: counts 0-12 in S, a count
above the cap (the error: the probe's machine cleared), loops on other registers, plain items around them. Then the
H1 answer program C.S[T.round1 STA.X] on real rows with sp2's recorded library and round, at full size (16 places):
x^(2^T) mod N for T = 1, 2, 4, 8, 16 on every machine. Exact cells. Local, CPU, seconds.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/test_count.py
"""
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from integration.learned_loop import libcore as L
from integration.learned_loop.rows import digits, load_rows

HERE = os.path.dirname(os.path.abspath(__file__))
dev = "cpu"
cells = L.Cells(10, dev)


def start(st, ins, n, seed):
    return L.fresh(st, {r: L.pad(v, st.P) for r, v in ins.items()}, L.gen_on(dev, seed), n, dev)


def by_ref(st, prog, lib, ins, n, seed):
    b = start(st, ins, n, seed)
    L.run_items(st, b, prog, lib)
    return b


def by_apply(st, prog, lib, ins, n, seed):
    """The program item by item through apply_items, on 3 copies of the state (as the search runs items on many
    states at once)."""
    b = start(st, ins, n, seed).take(torch.zeros(3, dtype=torch.long))
    for it in prog:
        b = L.apply_items(st, b, torch.zeros(3, dtype=torch.long), [it], lib)
    assert all(bool((b.regs[0] == b.regs[i]).all()) for i in range(3))
    return b.take(torch.zeros(1, dtype=torch.long))


def by_outer(st, prog, lib, ins, n, seed):
    m = L.OuterMachine(st, prog, lib, n, dev)
    m.cells(cells)
    return m.run(ins["X"], ins["N"], ins["S"], L.gen_on(dev, seed)).clone()


def by_ste(st, prog, lib, ins, n, seed):
    b = start(st, ins, n, seed)
    hot = lambda t, k: F.one_hot(t.long(), k).float()
    regs = [hot(b.regs[0, r].T, 10) for r in range(5)]
    SBB = torch.cat([hot(cells.sbb_d, 10), hot(cells.sbb_c, 2)], -1).reshape(200, 12)
    ADC = torch.cat([hot(cells.adc_d, 10), hot(cells.adc_c, 2)], -1).reshape(200, 12)
    MUL = torch.cat([hot(cells.mul_lo, 10), hot(cells.mul_hi, 10)], -1).reshape(10 ** 4, 20)
    m = L.SteMachine(SBB, ADC, MUL, st.P, lambda x: x)
    out = m.run(regs, hot(b.c[0], 2), hot(b.h[0], 10), hot(b.k[0], 10), prog, lib, p=0)
    return out[0][0].argmax(-1)


def check(st, prog, lib, ins, n, seed=3, ste=True):
    ref = by_ref(st, prog, lib, ins, n, seed)
    app = by_apply(st, prog, lib, ins, n, seed)
    same = bool((ref.regs == app.regs).all() and (ref.c == app.c).all() and (ref.h == app.h).all()
                and (ref.k == app.k).all() and (ref.p == app.p).all())
    A = ref.regs[0, L.A_].T.long()
    outer = by_outer(st, prog, lib, ins, n, seed)
    ok_outer = bool((outer == A).all())
    ok_ste = bool((by_ste(st, prog, lib, ins, n, seed) == A).all()) if ste else None
    return same, ok_outer, ok_ste, A


def main():
    t0 = time.time()
    rng = random.Random(0)
    st = L.Setting(10, 2, cells, 4)
    n = 16
    g = torch.Generator().manual_seed(0)
    S = torch.randint(0, 13, (n, 4), generator=g) * 0
    S[:, 0] = torch.randint(0, 10, (n,), generator=g)
    S[:4, 1] = 1                                                       # counts 10-12 ...
    S[4, :3] = torch.tensor([0, 5, 1])                                 # ... and 150: above the cap
    ins = {"X": torch.randint(0, 10, (n, 4), generator=g), "N": torch.randint(0, 10, (n, 4), generator=g), "S": S}
    prims = [p for p in L.PRIMS if p != "INC"] + ["INC"] * 3
    fails = 0
    progs = [L.parse("C.S[INC ADC.X]"), L.parse("C.S[ADC.X STA.X]"), L.parse("LDA.X C.S[STA.Y ADC.Y INC] SHL.A"),
             L.parse("C.X[INC]"), L.parse("C.S[]"), L.parse("C.S[SBB.N] C.S[ADC.N STA.Y]")]
    for _ in range(40):
        body = tuple(rng.choice(prims) for _ in range(rng.randint(1, 4)))
        pre = tuple(rng.choice(prims) for _ in range(rng.randint(0, 2)))
        post = tuple(rng.choice(prims) for _ in range(rng.randint(0, 2)))
        kind = rng.choice(L.COUNT_KINDS[1:] if rng.random() < 0.2 else ("C.S",))
        progs.append(pre + (("REP", kind, body),) + post)
    for prog in progs:
        same, ok_o, ok_s, _ = check(st, prog, {}, ins, n)
        if not (same and ok_o and ok_s):
            fails += 1
            print("FAIL", L.show(prog), same, ok_o, ok_s)
    print(f"random programs at P = 4: {len(progs)} checked, {fails} failing ({time.time() - t0:.1f}s)", flush=True)

    # the error path: count 150 clears the probe; counts 0-12 run that many times (C.S[ADC.X CLC] adds X's lowest
    # digit into A's lowest place once a round, the carry dropped: count x digit, mod 10)
    b = by_ref(st, L.parse("CLA CLC C.S[ADC.X CLC]"), {}, ins, n, 3)
    assert int(b.regs[0, :, :, 4].sum()) == 0 and int(b.c[0, 4]) == 0, "the error clears the machine"
    want = (S[:, 0] + 10 * S[:, 1]) * ins["X"][:, 0] % 10
    got = b.regs[0, L.A_, 0].long()
    keep = torch.arange(n) != 4
    assert bool((got[keep] == want[keep]).all()), (got, want)
    print("the count: counts 0-12 run that many times; 150 clears the probe", flush=True)

    # H1's answer program at full size, sp2's official run's library and round
    lib = {k: L.parse(v) for k, v in json.load(open(os.path.join(HERE, "sp2_official_lib.json"))).items()}
    lib["T.round1"] = L.parse("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA")
    st = L.round_settings(cells, 16)
    for prog in (L.parse("C.S[T.round1 STA.X]"),):
        for T in (1, 2, 4, 8, 16):
            rows = load_rows(6, seed=T, T=T)
            Nd, Xd, Yd = (digits(rows, k) for k in ("modulus", "x", "result"))
            Cd = digits(rows, "time_steps", 2)
            ins = {"X": Xd, "N": Nd, "S": Cd}
            t1 = time.time()
            same, ok_o, ok_s, A = check(st, prog, lib, ins, len(rows), ste=T <= 2)
            right = bool((A[:, :8] == Yd).all())
            print(f"{L.show(prog)} T={T}: machines agree {same} outer {ok_o} ste {ok_s}; the answer right {right} "
                  f"({time.time() - t1:.1f}s)", flush=True)
            fails += not (same and ok_o and ok_s is not False and right)
    print("ALL OK" if not fails else f"{fails} FAILURES", f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
