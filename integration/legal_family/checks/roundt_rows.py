"""ROUNDT's rows in a run (FAMILY_FILE.md): H1's training rows of T = 1, 2, 4 and 8 laid out as keep_counts stacks them
(N, x, the count's digits padded to W, y: digits LSB first, the count's as the reader gives them), kept per count as
keep_counts keeps ROUNDT's (the newest ROUNDT of each), then turned into the climb's `more` by climb_unit's own
expression: the T it reads off each count's digits, and ROUNDT on phase B's H1 rounds with those rows.

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 ~/github/sotaku/venv/bin/python integration/legal_family/checks/roundt_rows.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import family_local as FL  # noqa: E402


def main():
    M = FL.load()
    src = open(FL.FILE).read()
    expr = re.search(r"ctl\.climb\.more = (\[.*?\])\n", src, re.S).group(1)       # climb_unit's own expression
    keep = re.search(r"for s in rows:.*?\n(.*?)\n", src, re.S).group(1).strip()     # keep_counts' line for ROUNDT
    print("climb_unit:", " ".join(expr.split()), "\nkeep_counts:", keep, flush=True)
    K, W = M.K, M.W

    class Ctl:
        more = {}
    ctl = Ctl()
    for T in (1, 2, 4, 8):                                  # batches of rows, as keep_counts receives them
        rs = FL.load_rows(600, seed=20 + T, T=T)
        Nd, Xd, yd = (FL.digits_of([int(r[k]) for r in rs]) for k in ("modulus", "x", "result"))
        Cs = FL.digits_of([T for _ in rs], K)               # the count's digits, LSB first (count_of's order)
        for b in range(0, len(rs), 200):
            sl = slice(b, b + 200)
            new = torch.stack([t.long() for t in (Nd[sl], Xd[sl], F.pad(Cs[sl], (0, W - K)), yd[sl])], 1)
            slot = (Cs[sl] * 10 ** torch.arange(K - 1, -1, -1)).sum(1)
            rows = {s: None for s in torch.unique(slot).tolist()}
            for s in rows:
                exec(keep, {"ctl": ctl, "torch": torch, "new": new, "slot": slot, "s": s, "ROUNDT": M.ROUNDT})
    more = eval(expr, {"ctl": type("C", (), {"more": ctl.more})(), "count_of": M.count_of, "K": K})
    print("the counts kept:", {int(M.count_of(r[:1, 2, :K])): r.shape[0] for r in ctl.more.values()},
          "| ROUNDT's T:", [T for T, _ in more], flush=True)
    core = json.load(open(FL.CORE_B))
    c = FL.climb_at_level3(M, "cpu")
    lib = {k: M.parse(v) for k, v in core["lib"].items()}
    c.more = more
    for p in core["rounds"][:2]:
        print(f"  {p}: ROUNDT {c.round_t(M.parse(p), lib)}", flush=True)
    print(f"  x^2 with no reduction: ROUNDT {c.round_t(M.parse('LDA.X STA.Y CLA T.add.mul.o,x,o.AXY'), lib)}",
          flush=True)


if __name__ == "__main__":
    main()
