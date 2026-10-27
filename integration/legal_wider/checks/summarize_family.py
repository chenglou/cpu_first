"""The family's and the held-out pair's H100 records (WIDER.md §6; legal_family/checks/summarize_family.py for this
folder), and H1's: for each, learned or not (T = 64 on both ladders, every
rung 768/768), T on each ladder with every rung's count, the test, unseen-T and unseen-N splits, training and
evaluation seconds, and from the run's log the reading, the round, the answer programs, the freeze and how the climb
ended (its last lines), with its machine work.

    cd cpu_first && python3 integration/legal_wider/checks/summarize_family.py [RECORD.json ...]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "records"
TASKS = ("ten", "rho", "plus3", "minus1", "tenrho", "cube", "plusy", "three2", "poly_8_3_8_3", "poly_2_3_9_9")


def one(path):
    r = json.loads(Path(path).read_text())
    res = r.get("result") or {}
    s = (res.get("seeds") or [{}])[0]
    dp = s.get("depth_profile") or {}
    ev = s.get("evaluation") or {}
    rungs = [x["correct_examples"] for x in dp.get("rungs", [])]
    oodn = [x["correct_examples"] for x in dp.get("ood_n_rungs", [])]
    T, Tn = dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps")
    learned = T == 64 and Tn == 64 and all(c == 768 for c in rungs + oodn)
    out = r.get("stdout", "")
    grab = lambda pat: [m.group(0) for m in re.finditer(pat, out, re.M)]                      # noqa: E731
    read = grab(r"read \S+ on \d+ of \d+ answer rows, \d+ laws tried, [\d.]+s, poses \[[^\]]*\]")
    found = grab(r"^LR round_found step=\d+ programs: \[[^\]]*\]")
    frozen = grab(r"^LR frozen step=\d+")
    rounds = grab(r"^LR climb step=\d+ L\d+ round: cost (?!None)[^\n]{0,240}")
    work = grab(r"\d+(\.\d+)?e\+\d+ in all\]")
    tail = [l for l in out.splitlines() if l.startswith("LR climb")][-2:]
    print(f"== {r['name']} ({r['task']}): {'LEARNED' if learned else 'not learned'}; T = {T} / {Tn}; rungs {rungs} | "
          f"unseen N {oodn}; " + ", ".join(f"{k} {v['correct_examples']}/{v['example_count']}"
                                          for k, v in sorted(ev.items()))
          + f"; training {s.get('training_seconds', 0):.0f} s, evaluation {s.get('evaluation_seconds', 0):.0f} s, "
          f"steps {s.get('completed_training_steps')}; md5 {hashlib.md5(r['submission'].encode()).hexdigest()[:8]}, "
          f"data {((r.get('runner') or {}).get('data_md5') or '')[:8]}")
    print(f"   read: {read[:1] or 'nothing'}{' (and tried ' + str(len(read) - 1) + ' more times)' if len(read) > 1 else ''}")
    print(f"   round: {rounds[:1] or 'none'}")
    print(f"   answer program: {found[:1] or 'none'}; {frozen[:1] or 'not frozen'}; climb work at the end "
          f"{work[-1] if work else '-'}")
    if not learned:
        print("   the climb's last lines: " + " || ".join(t[:300] for t in tail))


def main():
    paths = sys.argv[1:] or [p for p in [REC / "legal_wider_official.json"] + [REC / f"wider_{t}_h100.json" for t in TASKS]
                             if p.exists()]
    for p in paths:
        one(p)


if __name__ == "__main__":
    main()
