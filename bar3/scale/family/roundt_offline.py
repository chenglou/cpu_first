"""FAMILY.md §5.4: ROUNDT's filter applied offline to a record's found rounds (all the programs its round search
accepted at T = 1): which of them also give the training data's answers at T = 2, 4 and 8, run T times at full size.

    cd cpu_first && TARGET_FN=minus1 ~/github/sotaku/venv/bin/python -m bar3.scale.family.roundt_offline RECORD.json
"""

from __future__ import annotations

import json
import sys

from integration.lib import libcore as L
from bar3.scale.family import hooks as H
from bar3.scale.family.core_f import digits, load_rows


def main():
    rec = json.load(open(sys.argv[1]))
    lib = {k: L.parse(v) for k, v in rec["lib"].items()}
    more = H.more_answers(load_rows, digits, "cpu")
    st = L.round_settings(L.Cells(10, "cpu"), 16)
    for i, show in enumerate(rec["rounds"]):
        p = L.parse(show)
        res = []
        for T, (Nd, Xd, Yd) in more:
            X = L.pad(Xd, 16)
            for _ in range(T):
                X, _c = L.run_prog(st, p, lib, {"X": X, "N": L.pad(Nd, 16)}, L.gen_on("cpu", 1), "cpu")
            res.append(f"T={T} {int((X[:, :Yd.shape[1]] == Yd).all(1).sum())}/{Yd.shape[0]}")
        print(f"  round {i}: {show}: {', '.join(res)}", flush=True)


if __name__ == "__main__":
    main()
