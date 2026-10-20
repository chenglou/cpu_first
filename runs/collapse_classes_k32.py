"""Do --collapse's classes predict their members? Checked against runs that certified every machine one by one.

1. h100_search_collapse_k32 (one representative per class) against h100_search_cellfree_k32 (the same search, every
   machine through E): are the members the same machines, and does each member's own outcome equal its
   representative's?
2. h100_search_nowidth_k32 (no width law: 53,440 machines, every one through E, 37,232 certify) grouped by design_key,
   with and without merging (Ke, Kx, Km, W): are E outcomes uniform within each class?
3. The same classes against h100_search_witness_k32's declared-range witness rows (wrong rows per row source).

Run from cpu_first/: python runs/collapse_classes_k32.py  (CPU only, about a minute)
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import spec_grammar as sg  # noqa: E402

G = list(sg.GRAMMAR)
outcome = lambda a: (a["seen_maxT"], a["seen_correct"], a["ood_maxT"], a["ood_correct"])

new = json.load(open("runs/h100_search_collapse_k32.json"))["out"]
mem = new["log"]["collapse"]["members"]
members = [(r[0], tuple(dict(sg.REF_IDX, **dict(zip(mem["factors"], r[1:])))[f] for f in G)) for r in mem["rows"]]
old_combos, old_audit = sg.combos_from_audit("runs/h100_search_cellfree_k32.json")
old = {tuple(c[f] for f in G): outcome(a) for c, a in zip(old_combos, old_audit)}
rep = {a["class"]: outcome(a) for a in new["audit"]}
print(f"1. {len(members):,} class members, {len(old):,} machines certified one by one before; same set: "
      f"{ {m for _, m in members} == set(old)}; members whose own outcome differs from their representative's: "
      f"{sum(old[m] != rep[k] for k, m in members)}")

combos, audit = sg.combos_from_audit("runs/h100_search_nowidth_k32.json")
w = json.load(open("runs/h100_search_witness_k32.json"))["out"]["machines"]
assert len(w) == len(combos)
witness = [tuple(sorted(m["wrong_rows"].items())) for m in w]
kept = [m["width_law"] == "" for m in w]
print(f"2-3. {len(combos):,} machines without the width law, "
      f"{sum(o[0] == 64 and o[2] == 64 for o in map(outcome, audit)):,} certify, {sum(kept):,} kept by the width law")
for rw, sub in ((False, "all"), (True, "all"), (True, "kept by the width law")):
    groups = collections.defaultdict(list)
    for i, c in enumerate(combos):
        if sub == "all" or kept[i]:
            groups[sg.design_key(c, rw)].append(i)
    cl = list(groups.values())
    mixed = sum(len({outcome(audit[i]) for i in m}) > 1 for m in cl)
    verdict = sum(len({audit[i]["seen_maxT"] == 64 and audit[i]["ood_maxT"] == 64 for i in m}) > 1 for m in cl)
    wmixed = sum(len({witness[i] for i in m}) > 1 for m in cl)
    wrong = sum(any(v for i in m for _, v in witness[i]) for m in cl)
    print(f"   {'merge' if rw else 'keep '} (Ke, Kx, Km, W), {sub}: {sum(map(len, cl)):,} machines in {len(cl):,} "
          f"classes; E outcomes differ within {mixed:,} classes (certification verdict within {verdict:,}); witness "
          f"outcomes differ within {wmixed:,}; classes with a member that gets witness rows wrong: {wrong:,}")
