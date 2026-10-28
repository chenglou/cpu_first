"""Checks: the true wiring with the true table is exact at every size; so is the operand-swapped wiring with the
transposed table; and what every single wrong option scores with the true table.

    ~/github/sotaku/venv/bin/python -m tinyrouting.selftest
"""

from __future__ import annotations

import torch

from tinyrouting.machine import MUXES, ORDER, Data, hard_exact, transpose_table, true_program, truth_table


def main():
    for B, w, k in [(2, 3, 2), (3, 2, 1), (3, 2, 2), (3, 3, 2), (4, 2, 2), (5, 2, 1)]:
        d = Data(B, w, k)
        TD, TC = truth_table(B)
        ex = hard_exact(d, true_program(), TD, TC)
        sw = dict(true_program(), a=2, b=1)
        exs = hard_exact(d, sw, *transpose_table(TD, TC))
        print(f"B={B} w={w} k={k} X={d.X}: true {ex:.3f}, swapped ports + transposed table {exs:.3f}")
        assert ex == 1.0 and exs == 1.0
        row = []
        for m in ORDER:
            for o in range(1, len(MUXES[m])):
                p = dict(true_program(), **{m: o})
                row.append(f"{m}={MUXES[m][o]}:{hard_exact(d, p, TD, TC):.2f}")
        print("   single wrong option, true table:", " ".join(row))


if __name__ == "__main__":
    torch.set_num_threads(2)
    main()
