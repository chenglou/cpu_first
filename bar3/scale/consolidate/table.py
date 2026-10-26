"""The ablation's table (CONSOLIDATE.md §6; PREREG_ADDENDUM1.md, PREREG_ADDENDUM2.md): a row per earlier win, the round
its target came under each arm (cs1's allocator, then the ablation's arms, tags ab1 and ab2) and under the earlier
records, and how many of the old rule's round-0 picks each rebuilt rule made (`lp_probe` against hardest first's,
`ct2a2_probe` against anchored pairs'). An arm's record is read from ab1, else ab2 (addendum 2 ran `ct2a2_probe` on the
formulas ab1 did not).

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.consolidate.table
"""

from __future__ import annotations

from pathlib import Path

from bar3.scale.consolidate.compare import old_records, outcome, read

HERE = Path(__file__).resolve().parent
ARMS = ("", "_probe", "_lp", "_k4", "_lp_probe", "_ct2a2", "_ct2a2_probe", "_ct2a2_probe_noties", "_kmax_probe")
WINS = [("alloc_deep", t) for t in ("h1", "affine", "rho", "minus1", "plus3", "fib", "cube", "ten", "x2x", "x1sq",
                                     "three", "x4", "pow2")] + [("alloc_ns", "tri"), ("alloc_ns", "tenrho"),
                                                                ("alloc_b_src", "dxmod")]


def main():
    head = ["formula", "alloc"] + [a[1:] for a in ARMS[1:]] + ["old lp", "old ct2a2", "patched", "r0 picks lp_probe/lp",
                                                              "ct2a2_probe/ct2a2"]
    print(" | ".join(head))
    tally = {a: 0 for a in ARMS}
    for base, t in WINS:
        cells, recs = [], {}
        for a in ARMS:
            r = (read(HERE / "records" / f"{base}_{t}_cs1.jsonl") if a == "" else
                 read(HERE / "records" / f"{base}{a}_{t}_ab1.jsonl") or read(HERE / "records" / f"{base}{a}_{t}_ab2.jsonl"))
            recs[a] = r
            o = outcome(r) if r else "-"
            tally[a] += o.startswith("r")
            cells.append(o)
        olds = dict((lab, read(p)) for lab, p in old_records(base, t))
        patched = " ".join(f"{lab} {outcome(o)}" for lab, o in olds.items() if "+" in lab) or "-"

        def same(new, old):
            if not new or not old:
                return "-"
            return f"{len(set(new[0]['picked']) & set(old[0]['picked']))}/{len(old[0]['picked'])}"
        print(" | ".join([t] + cells + [outcome(olds.get("lp")), outcome(olds.get("ct2a2")), patched,
                                        same(recs["_lp_probe"], olds.get("lp")),
                                        same(recs["_ct2a2_probe"], olds.get("ct2a2"))]))
    print("came: " + ", ".join(f"{a[1:] or 'alloc'} {n}" for a, n in tally.items()))


if __name__ == "__main__":
    main()
