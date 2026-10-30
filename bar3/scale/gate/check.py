"""Exactness of the gated engine with no gate (GATE.md §2): gjobs.search_job without a gate must give the record
bar3/scale/jobs.search_job gives, field for field but the seconds, on problems of every kind the learner poses: law-book
problems, the target's restrictions, the target probe, and target searches over a carried library. Then the same
jobs with a gate whose delay is always 0 (a gate that delays nothing is the plain search), and with a random gate
(delays change the order: what is found must still be verified, and every record must be well formed).

    cd cpu_first && <python with numba> -m bar3.scale.gate.check            (about a minute locally, under 0.5 GB)
"""

from __future__ import annotations

import json
import time

import numpy as np

from library.inventing import brm
from bar3 import learner as L, problems as P
from bar3.breadth import targets as TG
from bar3.scale import jobs as J
from bar3.scale import learn
from bar3.scale.gate import features as F
from bar3.scale.gate import gjobs
from bar3.scale.gate.model import Gate

SKIP = {"seconds", "wall", "item_seconds", "first_goal", "first_accept", "generated", "states", "delays", "dead",
        "gated_level"}                                      # a gated search's level, its delays included
STATE = "bar3/scale/records/seq_ct2a2_clean_new_targets_L1.state.affine.json"


def jobs_to_check():
    k = 7
    out = []
    TG.use("h1")
    kinds = list(brm.LOOP_KINDS)
    pool = P.laws(k)[:8] + P.restrictions(P.target_gen, k)[:4]
    for c in pool:
        out.append(learn.job(c, {}, kinds, dict(L.PROBE_A), "probe_a", k))
    out.append(learn.job(P.the_target(k), {}, kinds, dict(L.PROBE_A), "target_probe", k, near=12))
    st = json.load(open(STATE))["state"]
    TG.use("cube")
    j = learn.job(P.the_target(k), st["lib"], st["kinds"], dict(L.PROBE_A, max_states=30000), "target", k,
                  general=True)
    j["breadth_target"] = "cube"
    out.append(j)
    return out


def same(a, b):
    a = json.loads(json.dumps({k: v for k, v in a.items() if k not in SKIP}))
    b = json.loads(json.dumps({k: v for k, v in b.items() if k not in SKIP}))
    return a == b, [k for k in set(a) | set(b) if (k in a) != (k in b) or a.get(k) != b.get(k)]


def zero_gate(D=3):
    nf = F.N_STATE + F.N_PROB
    return Gate(np.zeros((nf, 4)), np.zeros(4), np.zeros(4), 0.0, kappa=1.0, ref=0.0, D=D, name="zero")


def random_gate(D=3, seed=1):
    rng = np.random.default_rng(seed)
    nf = F.N_STATE + F.N_PROB
    return Gate(rng.normal(0, 1, (nf, 8)), rng.normal(0, 1, 8), rng.normal(0, 1, 8), 0.0, kappa=1.0, ref=2.0, D=D,
                name="random")


def main():
    t0 = time.time()
    js = jobs_to_check()
    bad = 0
    for j in js:
        if j.get("breadth_target"):
            TG.use(j["breadth_target"])
        a = J.search_job(j)
        b = gjobs.search_job(j)
        ok, diff = same(a, b)
        z = gjobs.search_job(dict(j, gate=zero_gate().to_json()))
        okz, diffz = same(a, z)
        r = gjobs.search_job(dict(j, gate=random_gate().to_json(), budget="hard"))
        bad += (not ok) + (not okz)
        print(f"{j['kind']:>12} {j['name'][:40]:<40} plain {a['found_cost']} {sum(a['level_sizes']):>8} | no gate "
              f"{'same' if ok else diff} | zero gate {'same' if okz else diffz} | random gate found {r['found_cost']} "
              f"generated {r['generated']} first accepted {r['first_accept']} (plain {b['first_accept']}) progs "
              f"{len(r.get('programs', []))}")
        if r["found_cost"] is not None:
            assert r.get("programs"), "a found program must be verified"
    print(f"{len(js)} jobs, {bad} differ, {time.time() - t0:.0f}s")
    assert bad == 0


if __name__ == "__main__":
    main()
