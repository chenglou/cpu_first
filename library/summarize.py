"""Print a wake-sleep record file level by level.

    ~/github/sotaku/venv/bin/python -m library.summarize library/records/bpe_r1.jsonl [--programs]
"""

from __future__ import annotations

import json
import sys


def main():
    path = sys.argv[1]
    show_progs = "--programs" in sys.argv
    for line in open(path):
        rec = json.loads(line)
        print(f"=== {rec['config']} level {rec['level']}: solved {rec['solved']}  ({rec['seconds']} s)  "
              f"DL {rec['dl_before']} -> {rec['dl_after']} bits")
        for r in rec["records"]:
            if not isinstance(r, dict):
                print("   failed:", r[:200])
                continue
            best = next((p for p in r.get("programs", []) if p["verified"]), None)
            al = r.get("n_programs", 0) - r.get("n_verified", 0) if r.get("programs") else 0
            line = (f"   {r['task']:7s} cost {str(r['found_cost']):4s} states {r['level_sizes']} {r['seconds']:7.1f}s "
                    f"tokens {r['n_tokens']} classes {r['n_classes']} raw 2^{r['raw_log2']}")
            if best:
                line += f"  [{r['n_verified']} verified, {al} aliases] {best['program']}  (= {best['primitive_cost']} prims)"
            elif r.get("programs"):
                line += f"  [only aliases: {r['programs'][0]['program']} worst {r['programs'][0]['worst']}]"
            else:
                line += f"  {r['stop']}"
            print(line)
            if show_progs and r.get("programs"):
                for p in r["programs"][:12]:
                    print(f"        {'ok ' if p['verified'] else 'BAD'} {p['program']}")
        for m in rec["merges"]:
            extra = (f"weight {m['weight']}" if "weight" in m else "routine" if m.get("routine")
                     else f"{m['bits_before']} -> {m['bits_after']} bits")
            print(f"   + {m['name']:7s} = {m['def']:40s} {extra:28s} = {m.get('expands_to', '')}")


if __name__ == "__main__":
    main()
