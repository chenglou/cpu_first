"""This study's outcomes against the earlier records (CONSOLIDATE.md §5): for each formula, the round its target came
under hardest first and anchored pairs (the earlier wins' records, and the patched rules' where a patch won it) and
under the allocator (this study's records), with the states each run searched and the allocator's spending and picks
round by round.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.consolidate.compare [TAG] [--rounds]
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from bar3.scale.consolidate.configs import CONFIGS

HERE = Path(__file__).resolve().parent
SCALE = HERE.parent

BASES = ("alloc_deep", "alloc_ns", "alloc_b_src")
S1 = ("h1", "affine", "rho", "minus1", "plus3", "fib", "cube", "ten")
HO = ("x2x", "x1sq", "three", "x4", "pow2", "x3x", "lucas", "trib", "floorsq", "tri")


def old_records(config, t):
    """[(label, path)] of the earlier records this run is compared with."""
    if config == "alloc_deep":
        d, tag = ("subgoal", "s1") if t in S1 else ("heldout", "ho1")
        return [(r, SCALE / d / "records" / f"{r}_clean_deep_sub_{t}_{tag}.jsonl") for r in ("lp", "ct2a2")]
    if config == "alloc_ns":
        return ([(r, SCALE / "newton" / "records" / f"{r}_clean_deep_sub_ns_{t}_nw1.jsonl") for r in ("lp", "ct2a2")]
                + [("lp+tie", SCALE / "picks" / "records" / f"lp_clean_deep_sub_ns_rc_tie_{t}_pk1.jsonl")])
    if config == "alloc_b_src":
        return ([(r, SCALE / "digits" / "records" / f"{r}_clean_b_src_{t}_dg1.jsonl") for r in ("lp", "ct2a2")]
                + [("lp+twin", SCALE / "picks" / "records" / f"lp_clean_b_src_twin_{t}_pk1.jsonl"),
                   ("lp+twin5", SCALE / "reach" / "records" / f"lp_clean_b_src_twin5_{t}_rh1.jsonl")])
    raise KeyError(config)


def read(path):
    """A record's rounds, from RECORD.jsonl or, as committed, RECORD.jsonl.gz; None when neither exists."""
    path = Path(path)
    if path.exists():
        return [json.loads(line) for line in open(path)]
    gz = path.with_name(path.name + ".gz")
    return [json.loads(line) for line in gzip.open(gz, "rt")] if gz.exists() else None


def outcome(recs):
    """'r<k>' for the first round whose target came, 'no (<n> rounds)' else, '…' for a run under way."""
    if recs is None:
        return "-"
    for r in recs:
        if r.get("general"):
            return f"r{r['round']}"
    return f"no ({len(recs)})"


def states(recs):
    """The states every search of the run visited, in millions."""
    return sum(v["states"] for r in recs or [] for v in r.get("totals", {}).values()) / 1e6


def main():
    tag = next((a for a in sys.argv[1:] if not a.startswith("--")), "cs1")
    rows = []
    for path in sorted((HERE / "records").glob(f"alloc_*_{tag}.jsonl")):
        name = path.stem[: -len(tag) - 1]
        config = max((c for c in CONFIGS if name.startswith(c + "_")), key=len)
        t = name[len(config) + 1:]
        recs = read(path)
        olds = [(lab, read(p)) for lab, p in old_records(next(b for b in BASES if config.startswith(b)), t)]
        rows.append((config, t, recs, olds))
        print(f"{config:22s} {t:8s} allocator {outcome(recs):9s} ({states(recs):7.0f}M states) | "
              + " | ".join(f"{lab} {outcome(o)} ({states(o):.0f}M)" for lab, o in olds))
        if "--rounds" in sys.argv:
            for r in recs:
                a = r.get("alloc", {})
                kinds = {}
                for s in a.get("steps", []):
                    kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
                print(f"    r{r['round']}: pool {len(r['pool'])}, budget {a.get('budget', 0) / 1e6:.0f}M, spent "
                      f"{a.get('spent', 0) / 1e6:.0f}M, searches {kinds}, picks "
                      f"{[(p, r['status'].get(p, {}).get('gain')) for p in r['picked']]}, came {bool(r['general'])}")
    return rows


if __name__ == "__main__":
    main()
