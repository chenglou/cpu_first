"""The digit machine's three cells learned from random, in order, from laws on the rows' own numbers and on the cells
themselves; no answers, no hand-written arithmetic in any lesson (INTEGRATION.md, stage 2):

  sbb (the subtraction table, (a, b, borrow in) -> (digit, borrow out)): the anchors' certified recipe (anchors/core.py
      R2, ANCHORS.md): "x < N on every top part" (sub.order on the mined order fact) teaches every borrow, then the
      order laws and bijectivity (sub.order, sub.bijective) every digit. It joins first;
  adc (addition with carry): sub.inverse on every entry, with the snapped sbb: (a + b + c) - b - c = a, the carry being
      that subtraction's borrow. Its target for (a, b, c) is the digit d whose snapped sbb(d, b, c) gives back a. It
      joins once sbb passes its own checks;
  mul (the multiply-accumulate cell, (a, x, h, k) -> (low, high) of a + x h + k): mul.zero and mul.ladder on every
      entry, with the snapped adc: a + x 0 + k = a + k, and a + x (h + 1) + k = (a + x h + k) + x (the low digit plus x
      with carry, the carry into the high digit). It joins once adc is exact by its own law.
Each cell is a table of logits (Adam), as the submissions' tables. learn() returns the snapped tables (tdsl.Cells.set's
keyword names) and a log of when each joined and became exact by its own checks (and, for the report only, by the
truth).
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
R2 = {"pref": 1, "colall": 1, "rowall": 1, "bij": 1, "bijr": 1}


def _core():
    sys.path.insert(0, os.path.join(ROOT, "anchors"))
    import core
    return core


def adc_targets(Td, Tb, B=10):
    """sub.inverse: for every (a, b, c) the digit d with sbb(d, b, c) = (a, borrow) and that borrow as the carry
    (Td, Tb: the snapped sbb [B * B * 2], flat over (d, b, c)). Where no d gives a back (a table not yet a bijection
    in its first operand), the entry has no target (-1)."""
    Td3, Tb3 = Td.view(B, B, 2), Tb.view(B, B, 2)
    tgt_d = torch.full((B, B, 2), -1, dtype=torch.long)
    tgt_c = torch.full((B, B, 2), -1, dtype=torch.long)
    for d in range(B):                                       # sbb(d, b, c) = Td3[d, b, c]: the d that gives a
        a = Td3[d]                                           # [B (b), 2 (c)]
        bb, cc = torch.meshgrid(torch.arange(B), torch.arange(2), indexing="ij")
        tgt_d[a, bb, cc] = d
        tgt_c[a, bb, cc] = Tb3[d]
    return tgt_d, tgt_c


def mul_targets(Ml, Mh, Ad, Ac, B=10):
    """mul.zero and mul.ladder on every entry, with the snapped adc (Ad, Ac [B, B, 2]) and the snapped mul (Ml, Mh
    [B, B, B, B] over (a, x, h, k)): row h = 0 is a + k (adc(a, k, 0), its carry the high digit); row h + 1 is row h
    plus x (the low digit plus x with carry, that carry added into the high digit)."""
    a, x, k = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(B), indexing="ij")
    lo = torch.empty(B, B, B, B, dtype=torch.long)
    hi = torch.empty(B, B, B, B, dtype=torch.long)
    lo[:, :, 0, :] = Ad[a, k, 0]
    hi[:, :, 0, :] = Ac[a, k, 0]
    for h in range(B - 1):
        l0, h0 = Ml[:, :, h, :], Mh[:, :, h, :]
        l1, c1 = Ad[l0, x, 0], Ac[l0, x, 0]
        lo[:, :, h + 1, :] = l1
        hi[:, :, h + 1, :] = Ad[h0, 0, c1]                    # high digit plus the carry (no carry out: x h + a + k < B^2)
    return lo, hi


def learn(seed=0, sbb_steps=600, rows=4096, batch=64, cell_steps=300, lr=0.05, log=print):
    """The three cells from random, in order. Returns ({sbb_d, sbb_c, adc_d, adc_c, mul_lo, mul_hi} snapped tables,
    a log)."""
    core = _core()
    B = 10
    t0 = time.time()
    out = {"seed": seed}
    lab = core.Lab(B, 8, core.h1_rows(rows, seed=seed), R2, seed=seed, gate=1.0, batch=batch)
    td, tb = lab.mc.true_tables()
    sbb_ok_at, streak = None, 0
    for s in range(1, sbb_steps + 1):
        _, hard = lab.train_step()
        Td, Tb = lab.tables()
        Td, Tb = Td.flatten(), Tb.flatten()
        # sbb's own checks: the order facts and the laws its lessons state on the batch (the anchors' hard checks),
        # the order laws on every entry (colall, rowall) and sub.bijective on every row and column of the table
        own = min(hard.get(k, 0.0) for k in ("less", "more", "zero", "self", "neg", "neg2", "wrap", "colall", "rowall"))
        T3 = Td.view(B, B, 2)
        bij = all(bool((torch.sort(T3[:, b_, c_]).values == torch.arange(B)).all()) and
                  bool((torch.sort(T3[b_, :, c_]).values == torch.arange(B)).all()) for b_ in range(B) for c_ in range(2))
        streak = streak + 1 if (own == 1.0 and bij) else 0
        right = int(((Td == td) & (Tb == tb)).sum())
        if s % 50 == 0:
            log(f"cells sbb step {s}: own checks {own:.3f} bijective {bij} streak {streak}, entries right {right}/200 "
                f"(report only)")
        if streak >= 50:
            sbb_ok_at = s
            break
    out.update(sbb_steps=s, sbb_ok_at=sbb_ok_at, sbb_right=right, sbb_gate=lab.gated_on)
    Td3, Tb3 = Td.view(B, B, 2), Tb.view(B, B, 2)
    # adc by sub.inverse on every entry
    g = torch.Generator().manual_seed(seed + 1)
    adc_l = (0.1 * torch.randn(B, B, 2, B, generator=g)).requires_grad_()
    adc_c = (0.1 * torch.randn(B, B, 2, 2, generator=g)).requires_grad_()
    opt = torch.optim.Adam([adc_l, adc_c], lr=lr)
    tgt_d, tgt_c = adc_targets(Td, Tb, B)
    m = tgt_d >= 0
    for s in range(1, cell_steps + 1):
        loss = (F.cross_entropy(adc_l[m], tgt_d[m]) + F.cross_entropy(adc_c[m], tgt_c[m]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        Ad, Ac = adc_l.argmax(-1), adc_c.argmax(-1)
        if bool(m.all()) and bool((Ad == tgt_d).all() and (Ac == tgt_c).all()):
            break
    out.update(adc_steps=s, adc_own=bool(m.all()) and bool((Ad == tgt_d).all() and (Ac == tgt_c).all()))
    # mul by mul.zero and mul.ladder on every entry, with the snapped adc
    mul_l = (0.1 * torch.randn(B, B, B, B, B, generator=g)).requires_grad_()
    mul_h = (0.1 * torch.randn(B, B, B, B, B, generator=g)).requires_grad_()
    opt = torch.optim.Adam([mul_l, mul_h], lr=lr)
    for s in range(1, 3 * cell_steps + 1):
        with torch.no_grad():
            lo, hi = mul_targets(mul_l.argmax(-1), mul_h.argmax(-1), Ad, Ac, B)
        loss = F.cross_entropy(mul_l.reshape(-1, B), lo.flatten()) + F.cross_entropy(mul_h.reshape(-1, B), hi.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        Ml, Mh = mul_l.argmax(-1), mul_h.argmax(-1)
        lo2, hi2 = mul_targets(Ml, Mh, Ad, Ac, B)
        if bool((Ml == lo2).all() and (Mh == hi2).all()):
            break
    out.update(mul_steps=s, mul_own=bool((Ml == lo2).all() and (Mh == hi2).all()))
    tabs = dict(sbb_d=Td3, sbb_c=Tb3, adc_d=Ad, adc_c=Ac, mul_lo=Ml, mul_hi=Mh)
    # the report only: each table against the truth
    a, b_, c = torch.meshgrid(torch.arange(B), torch.arange(B), torch.arange(2), indexing="ij")
    tr = {"sbb_d": (a - b_ - c) % B, "sbb_c": (a - b_ - c < 0).long(), "adc_d": (a + b_ + c) % B,
          "adc_c": (a + b_ + c >= B).long()}
    a4, x4, h4, k4 = torch.meshgrid(*(torch.arange(B),) * 4, indexing="ij")
    t4 = a4 + x4 * h4 + k4
    tr.update(mul_lo=t4 % B, mul_hi=t4 // B)
    out["right_report"] = {k: f"{int((tabs[k].long() == v).sum())}/{v.numel()}" for k, v in tr.items()}
    out["seconds"] = round(time.time() - t0, 1)
    log(f"cells learned: {out}")
    return tabs, out


if __name__ == "__main__":
    os.environ.setdefault("H1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                                                  "one-layer-deeper", "data", "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
    torch.set_num_threads(2)
    import json
    seeds = [int(v) for v in os.environ.get("SEEDS", sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
    for sd in seeds:
        _, out = learn(seed=sd, log=lambda s: None)
        print("CELLS " + json.dumps(out), flush=True)
