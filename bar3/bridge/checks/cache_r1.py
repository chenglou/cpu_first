"""How much of the lost round 1 (BRIDGE.md §6) the shared search cache already holds: round 1's jobs rebuilt from each
run's round-0 record exactly as dlearner.run builds them, looked up by the key search_remote uses. Reads the cache
only; launches nothing.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.cache_r1 lp_adc ct2_adc all_adc
"""

from __future__ import annotations

import hashlib
import json
import sys

import modal

from bar3.bridge import dlearner as L, dproblems as P, report as R

cache = modal.Dict.from_name("bar3-bridge-cache")


def key(job):
    return "v1" + hashlib.sha1(json.dumps(job, sort_keys=True).encode()).hexdigest()


def pool_of(tgt, near, lib, laws):
    pool = (P.digit_laws(tgt) + laws + P.restrictions(tgt) + P.projections(tgt)
            + [c for c in P.from_near_misses(near, lib, tgt) if c.source in L.SOURCES or c.source == "restrict"])
    return P.dedupe(pool)


def main(names):
    posed, _, _ = L.the_book()
    for name in names:
        cfg = L.CONFIGS[name]
        B = cfg["budgets"]
        rec0 = R.load(f"{name}_r1")[0]
        tsl = next(s for s in posed if s.name == cfg["target"])
        tgt = P.from_slice(tsl, "target")
        laws = [P.from_slice(s, "slice") for s in posed if s.name != tsl.name]
        near0 = [(f, d, p) for f, d, p in rec0["target_probe"].get("near", []) if p]
        pool0 = [c for c in pool_of(tgt, near0, {}, laws) if c.key() != tgt.key()]
        assert [c.name for c in pool0] == [c["name"] for c in rec0["pool"]], name
        practiced = {c.key(): c.name for c in pool0 if c.name in rec0["picked"]}
        lib = rec0["lib"]
        tpj = L.job(tgt, lib, B["target_probe"], "target_probe", near=12)
        tp = cache.get(key(tpj))
        if tp is None:
            print(f"{name}: round 1's target probe is not cached")
            continue
        near = [(f, d, p) for f, d, p in tp.get("near", []) if p]
        pool = [c for c in pool_of(tgt, near, lib, laws) if c.key() not in practiced and c.key() != tgt.key()]
        ja = [L.job(c, lib, B["probe_a"], "probe_a") for c in pool]
        ra = [cache.get(key(j)) for j in ja]
        todo = [i for i, r in enumerate(ra) if r is not None and r["found_cost"] is None]
        jb = [L.job(pool[i], lib, B["probe_b"], "probe_b") for i in todo]
        rb = [cache.get(key(j)) for j in jb]
        sec = lambda rs: sum(r.get("wall", r["seconds"]) for r in rs if r is not None) / 3600   # noqa: E731
        print(f"{name}: pool {len(pool)}; probe_a cached {sum(r is not None for r in ra)}/{len(ra)} "
              f"({sec(ra):.2f} h); probe_b of the cached unsolved: cached {sum(r is not None for r in rb)}/{len(rb)} "
              f"({sec(rb):.2f} h); probe_a not cached {sum(r is None for r in ra)}")


if __name__ == "__main__":
    main(sys.argv[1:])
