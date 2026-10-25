"""Can the target search of practise-everything (rule all) reach add-with-carry once the widened piece is a routine
(CARRY.md)? The target searched at the learner's full budget (3M states, 2,400 s) with all_adc's round-0 library, and
with its round-1 library (rebuilt by checks/replay_cache.py from the cache), each plus the composing piece that rule
finds in round 1 (checks/carry_jobs.py). On Modal, through the shared cache.

    cd cpu_first && BRIDGE_APP=bar3-bridge-carry PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_reach <round-1 library json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from bar3.bridge import dlearner as L, dproblems as P, modal_bridge as MB, report as R

HERE = Path(__file__).resolve().parent.parent
PIECE = "INC.A T.lawmod(A,N)aqN T.res86 M0.14"


def main(lib1_path):
    posed, _, gen = L.the_book()
    by = {s.name: s for s in posed}
    tgt = P.from_slice(by["add(A,R)"], "target")
    g = P.from_slice(gen["add(A,R)"], "target")
    lib0 = R.load("all_adc_r1")[0]["lib"]
    lib1 = json.loads(Path(lib1_path).read_text())["lib"]
    todo = [(f"round-0 library ({len(lib0)}) + the piece", dict(lib0, **{"T.wid": PIECE})),
            (f"round-1 library ({len(lib1)}) + the piece", dict(lib1, **{"T.wid": PIECE}))]
    jobs = [L.job(tgt, lib, L.FULL, "target", general=g) for _, lib in todo]
    with MB.app.run():
        recs = list(MB.search_remote.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))
    out = []
    for (label, lib), r in zip(todo, recs):
        if not isinstance(r, dict):
            print(f"{label}: failed {r!r}"[:300])
            continue
        progs = [p["program"] + (" (general)" if p.get("general") else "") for p in r.get("programs", [])]
        print(f"{label}: found {r['found_cost']} levels {r['level_sizes']} tokens {r['n_tokens']} classes "
              f"{r['n_classes']} {r['wall']:.0f} s {r['stop']}; {progs[:4]}")
        out.append(dict(label=label, lib=lib, record=r))
    dest = HERE / "records" / "carry" / "carry_reach.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main(sys.argv[1])
