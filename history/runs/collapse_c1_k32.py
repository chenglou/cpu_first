"""--collapse C1 against --collapse: the same survivors at every stage, the same classes and members, the same D and E.

Runs (Modal H100, the requested label-free search on the 2^76.3 ports grammar):
- h100_search_collapse_k32: --collapse (classes formed after C1), the earlier file (width law per machine);
- h100_search_collapse_c1_k32: --collapse C1 (C1 on one representative per (class, cw)), this file;
- h100_search_collapse_d_k32: --collapse with this file (the earlier path, width law memoized), as a timing baseline.
Every stage record is compared without its timings (and, for C1, without what --collapse C1 screens differently);
then the classes, every member, D, E and the audit rows; then the times and both spot-checks.

Run from cpu_first/history/: python runs/collapse_c1_k32.py  (reads JSON only, a few seconds)
"""
import json

RUNS = {"before": "h100_search_collapse_k32", "C1": "h100_search_collapse_c1_k32", "same file": "h100_search_collapse_d_k32"}
TIMING = {"seconds", "warmup_seconds", "machine_rows_per_s", "grouping_seconds"}
logs = {k: json.load(open(f"runs/{v}.json"))["out"] for k, v in RUNS.items()}


def strip(rec, stage):
    """A stage record without timings (C1: also without what one representative per group changes: the rows screened
    and the per-group survivors), and C0's per-check records likewise; CW without the memo's settings count."""
    if isinstance(rec, dict):
        drop = TIMING | {"check", "collapsed", "laws_evaluated"} | (
            {"alive_after_row_groups", "machine_rows"} if stage == "C1" else set())
        return {k: strip(v, stage) for k, v in rec.items() if k not in drop}
    if isinstance(rec, list):
        return [strip(v, stage) for v in rec]
    return rec


base = logs["before"]["log"]
for name in ("C1", "same file"):
    log = logs[name]["log"]
    same = {s: strip(log[s], s) == strip(base[s], s) for s in ("A", "B", "cells", "C0", "CW", "C", "C1", "collapse", "D", "E")}
    audit = logs[name]["audit"] == logs["before"]["audit"]
    m, b = log["collapse"]["members"], base["collapse"]["members"]
    print(f"{name} vs before: stage records identical {same}; collapse members identical (class, factors): "
          f"{m == b} ({len(m['rows']):,} rows, {log['collapse']['classes']} classes); audit rows identical: {audit}")

print("\nseconds              " + "".join(f"{k:>12}" for k in RUNS))
rows = [("A", lambda l: l["A"]["seconds"]), ("B", lambda l: l["B"]["seconds"]), ("cell laws", lambda l: l["cells"]["seconds"]),
        ("C0", lambda l: l["C0"]["seconds"]), ("CW (width law)", lambda l: l["CW"]["seconds"]),
        ("C", lambda l: l["C"]["seconds"]), ("C1", lambda l: l["C1"]["seconds"]),
        ("collapse", lambda l: l["collapse"]["seconds"]), ("D", lambda l: l["D"]["seconds"]),
        ("E", lambda l: l["E"]["seconds"]), ("wall to end of E", lambda l: l["wall_seconds_to_E"]),
        ("C1 spot-check", lambda l: l["C1"].get("check", {}).get("seconds", float("nan"))),
        ("E spot-check", lambda l: l["collapse"]["check"]["seconds"]), ("wall", lambda l: l["wall_seconds"])]
for label, get in rows:
    print(f"{label:21}" + "".join(f"{get(logs[k]['log']):12.1f}" for k in RUNS))
c1 = logs["C1"]["log"]["C1"]
print(f"\nC1 with --collapse C1: {c1['space']:,} machines in {c1['collapsed']['groups']:,} (class, cw) groups "
      f"(sizes {c1['collapsed']['group_sizes']}), {c1['collapsed']['representatives_pass']} representatives pass -> "
      f"{c1['survivors']:,} survivors; grouping {c1['collapsed']['grouping_seconds']:.1f}s; rows screened "
      f"{c1['machine_rows']:,} (before: {base['C1']['machine_rows']:,})")
ck = c1["check"]
fails = [r for r in ck["machines"] if not r["agrees"]]
print(f"C1 spot-check: {ck['sampled']:,} non-representatives ({ck['per_group']} per group, {ck['groups']:,} groups): "
      f"{ck['agree']:,} pass or fail as their representative, {ck['pass']:,} pass, {len(fails)} disagree")
for name in RUNS:
    e = logs[name]["log"]["collapse"]["check"]
    print(f"E spot-check ({name}): {e['sampled']:,} non-representatives, {e['agree']:,} certify as their "
          f"representative, {e['certify_both']:,} certify T=64 on both")
print("CW settings evaluated:", {k: logs[k]["log"]["CW"].get("laws_evaluated", logs[k]["log"]["CW"]["machines"]) for k in RUNS})
