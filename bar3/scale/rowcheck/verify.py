"""ROWCHECK.md §5: the runs' outcomes, read from the records. For each run (records/*_rc1.jsonl) against the earlier
run without the check: whether and in which round its target came; the rounds whose picks, library and target search
are the earlier run's; and every status the check changed, by round (the record's status before and after, the cost,
and what it is now an alias or duplicate of, if anything). No search runs.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.rowcheck.verify
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EARLIER = {"lp_clean_deep_sub_ns_rc_tenrho": HERE.parent / "newton" / "records" / "lp_clean_deep_sub_ns_tenrho_nw1.jsonl",
           "lp_clean_b_src_rc_dxsq": HERE.parent / "digits" / "records" / "lp_clean_b_src_dxsq_dg1.jsonl",
           "ct2a2_clean_b_src_rc_dxsq": HERE.parent / "digits" / "records" / "ct2a2_clean_b_src_dxsq_dg1.jsonl"}


def read(path):
    return [r for r in (json.loads(line) for line in open(path) if line.strip()) if "round" in r]


def decided(rec):
    tr = rec["target"] or {}
    return (rec["picked"], rec["lib"], tr.get("found_cost"), [p["program"] for p in tr.get("programs", [])],
            (rec.get("general") or {}).get("program"))


def main():
    for path in sorted((HERE / "records").glob("*_rc1.jsonl")):
        key = path.name.removesuffix("_rc1.jsonl")
        recs, old = read(path), read(EARLIER[key])
        g = next((r for r in recs if r.get("general")), None)
        same = [r["round"] for r, o in zip(recs, old) if decided(r) == decided(o)]
        print(f"{path.name}: {len(recs)} rounds (the earlier run: {len(old)}); "
              + (f"came in round {g['round']}: {g['general']['program']}" if g else "did not come")
              + f"; rounds with the earlier run's picks, library and target search: {same}", flush=True)
        for r, o in zip(recs, old):
            ch = {n: (o["status"][n].get("status"), v.get("status"), v.get("cost"), v.get("of"))
                  for n, v in r["status"].items() if n in o["status"]
                  and o["status"][n].get("status") != v.get("status")}
            if ch:
                print(f"  round {r['round']}: {ch}", flush=True)


if __name__ == "__main__":
    main()
