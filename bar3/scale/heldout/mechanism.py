"""HELDOUT.md §5, which mechanism did the work, read from the records: for each run whose target came, which search
found the program (the plain target search or the guided second one), and every routine it calls, directly or inside
other routines. For each routine: the practised problem it came from, that problem's source (the prefix of its name:
law, law2, restrict, project, value, remainder, dream, recur, const, rrem), the round it was practised in, its cost,
and whether it was that round's subgoal and solved only by the subgoal's deeper probe; or, for a merge (M<round>.<i>),
its definition. No search runs.

    cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.heldout.mechanism [records/*.jsonl ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"(?:T|M\d+)\.[^\s\[\]]+")


def read(path):
    with open(path) as fh:
        return [r for r in (json.loads(line) for line in fh if line.strip()) if "round" in r]


def origins(recs, upto):
    """{routine token: where it came from} for the routines of rounds 0..upto."""
    out = {}
    for r in recs[:upto + 1]:
        inv = {f"T.{s}": n for n, s in r["stems"].items()}
        sub = (r.get("subgoal") or {}).get("name")
        deep = any(d.get("found_cost") is not None for d in (r.get("subgoal") or {}).get("deep", []))
        for n in r["solved"]:
            tok = f"T.{r['stems'][n]}"
            if tok in out:
                continue
            st = r["status"].get(n, {})
            out[tok] = dict(problem=n, source=n.split("/", 1)[0], round=r["round"], cost=st.get("cost"),
                            how=st.get("status"), subgoal=n == sub, by_deeper_probe=n == sub and deep)
        for e in r["merges"]:
            out.setdefault(e["name"], dict(merge=e["def"], round=r["round"]))
        for tok in inv:
            out.setdefault(tok, dict(problem=inv[tok], source=inv[tok].split("/", 1)[0], round=r["round"]))
    return out


def trace(prog, lib, org, seen=None):
    """Every routine prog calls, directly or inside other routines, in the order first called."""
    seen = [] if seen is None else seen
    for tok in TOKEN.findall(prog):
        if tok in lib and tok not in seen:
            seen.append(tok)
            trace(lib[tok], lib, org, seen)
    return seen


def explain(path):
    recs = read(path)
    if not recs:
        return dict(record=Path(path).name, rounds=0)
    out = dict(record=Path(path).name, target=recs[0]["target_name"], rule=recs[0]["rule"], rounds=len(recs))
    r = next((x for x in recs if x.get("general")), None)
    if r is None:
        return dict(out, invented=None)
    g = r["general"]["program"]
    plain = [p["program"] for p in (r.get("target") or {}).get("programs", []) if p.get("general") == 1.0]
    org = origins(recs, r["round"])
    return dict(out, invented=r["round"], program=g, search="target search" if g in plain else "guided search",
                routines=[dict(token=t, definition=r["lib"][t], **org.get(t, {})) for t in trace(g, r["lib"], org)])


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in (HERE / "records").glob("*_ho1.jsonl"))
    for p in paths:
        e = explain(p)
        if not e["rounds"]:
            print(f"{e['record']}: no rounds")
            continue
        if e["invented"] is None:
            print(f"{e['record']}: not invented in {e['rounds']} rounds")
            continue
        print(f"{e['record']}: round {e['invented']}, {e['search']}: {e['program']}")
        for t in e["routines"]:
            if "merge" in t:
                print(f"    {t['token']} = {t['definition']}   (a merge, round {t['round']})")
            else:
                via = (", the subgoal, solved by its deeper probe" if t.get("by_deeper_probe") else
                       ", the subgoal" if t.get("subgoal") else "")
                print(f"    {t['token']} = {t['definition']}   ({t.get('problem')}: source {t.get('source')}, "
                      f"practised in round {t.get('round')}, cost {t.get('cost')}{via})")


if __name__ == "__main__":
    main()
