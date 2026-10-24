"""Which of a run's practised reds can confirm an addition (SELF_PRACTICE.md §3.2): the addition's check is sub.inverse,
red(a + N; N) = a for a < N, by any practised red on the pair pattern. Each red is run, in its own binding, on
a + N (a < N, the sum computed here only to test the routine) and on the pair pattern's own instances (x' <= N').

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/self_practice/check_reds.py core_lp_v12 [core_lpca_v11f]
(CPU, a few seconds; exact cells)
"""
import json
import sys
from pathlib import Path

import torch

from integration.lib import libcore as L
from integration.self_practice import spcore as S

HERE = Path(__file__).resolve().parent


def number(v, P):
    """Digits [n, P] (least significant first) of the integers v."""
    return torch.stack([(v // 10 ** i) % 10 for i in range(P)], 1)


def value(d):
    return sum(d[:, i].long() * 10 ** i for i in range(d.shape[1]))


def main():
    st = L.Setting(10, 2, L.Cells(10, "cpu"))          # widths up to 2, registers of 4 places
    gen = torch.Generator().manual_seed(0)
    N = torch.randint(2, 100, (256,))
    a = torch.stack([torch.randint(0, int(n), ()) for n in N])
    x = torch.stack([torch.randint(0, int(n) + 1, ()) for n in N])     # the pair pattern: x' <= N'
    for tag in [t for t in sys.argv[1:] if not t.startswith("-")]:
        rec = json.loads((HERE / "records" / f"{tag}.json").read_text())
        lib = {k: L.parse(v) for k, v in rec["lib"].items()}
        ops = S.Ops()
        for tok, bd in rec["ops"].get("red@pair", []):
            ops.add("red", "pair", tok, tuple(bd))
        print(f"== {tag}: {ops.count('red', 'pair')} practised reds on the pair pattern")
        for i, (tok, bd) in enumerate(ops.by_key.get(("red", "pair"), [])):
            on_sum = value(ops.run("red", "pair", st, lib, [number(a + N, st.P), number(N, st.P)], gen, "cpu", i)[0])
            on_pair = value(ops.run("red", "pair", st, lib, [number(x, st.P), number(N, st.P)], gen, "cpu", i)[0])
            print(f"  {tok:16} {''.join(bd)}  `{L.show(lib[tok])}`: right on x' <= N' {(on_pair == x % N).float().mean():.2f}; "
                  f"red(a + N; N) = a on {(on_sum == a).float().mean():.2f} (0 where it gives 0: "
                  f"{(on_sum == 0).float().mean():.2f})")


if __name__ == "__main__":
    main()
