"""The tiny core's records (records/<TAG>.json, GENERIC_OPS.md §3): one line per run (instruction set, settings, levels,
where and when the round came), then with --pieces the routines each run practised for the round's pieces (the link,
the chain, the multiply-accumulate row, the multiply, the reduction) and the round's programs.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/generic_ops/summarize_g.py [TAG ...] [--pieces]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIECES = (("link", "red/pair@"), ("chain", "red/x+dn@"), ("mac row", "acc.mul/o,x,d@"), ("multiply", "mul/x,o@"),
          ("reduction", "red/x+on@"))


def runs(tags):
    for p in sorted((HERE / "records").glob("*.json")):
        if p.stem.startswith(("diag", "test")) or (tags and p.stem not in tags):
            continue
        try:
            r = json.load(open(p))
        except ValueError:                           # (a record being fetched)
            continue
        if "config" in r and "history" in r:
            yield p.stem, r


def line(tag, r):
    c = r["config"]
    sets = [f"isa {c.get('isa', 'sel,zero,brg')}"] + [f"{k} {c[k]}" for k in ("alias_fit", "calls_first") if c.get(k)]
    sets += [f"{k} {c[k]:g}" for k, d in (("deep_k", 8), ("deep_budget", 5.8e6), ("round_budget", 4e6))
             if c.get(k, d) != d]
    rnd = next((x for x in r["records"] if x.get("task") == "round" and x.get("found_cost") is not None), None)
    secs = next((u["t"] for u in r["units"] if " round: cost " in u["line"] and "cost None" not in u["line"]), None)
    last = r["units"][-1]["line"][:70] if r["units"] else ""
    where = (f"the round after level {rnd['level'] - 1} ({rnd['found_cost']}), {secs} s" if rnd   # (searched once a level's
             else f"no round; last: {last}")                                                       # sleep has counted it)
    return f"{tag}: {', '.join(sets)}; {len(r['history'])} levels, {r['seconds']} s{' (final)' if r.get('final') else ''}; {where}"


def pieces(r):
    out = []
    for name, pre in PIECES:
        got = [(lv["level"], p, lv["learnable"][p][1]) for lv in r["history"] for p in lv["picks"] if p.startswith(pre)]
        out.append(f"  {name}: " + ("; ".join(f"L{lv} {p.split('@')[1]} `{prog}`" for lv, p, prog in got[:3]) or "-"))
    out.append(f"  round: {r['rounds'][:2]}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for tag, r in runs(set(args)):
        print(line(tag, r))
        if "--pieces" in sys.argv:
            print("\n".join(pieces(r)))
