"""Writes plans/runs.json (PREREG.md, Runs): the file byte for byte, one H100 each, seed 74 (the runner's), on the
eight family tasks, phase D's pair and the six fresh tasks (heldout.json), named deep_<task>_h100.

    cd cpu_first && python3 integration/legal_deep/make_plan.py [--md5six]
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = ("ten", "rho", "plus3", "minus1", "tenrho", "cube", "plusy", "three2")


def main(argv):
    if "--md5six" in argv:  # PREREG_ADDENDUM1.md: the six drawn by the file's md5 (heldout2.json), plans/runs2.json
        tasks, out = tuple(json.loads((HERE / "heldout2.json").read_text())["tasks"]), HERE / "plans" / "runs2.json"
    else:
        old = json.loads((HERE.parent / "legal_wider" / "heldout.json").read_text())["tasks"]
        fresh = json.loads((HERE / "heldout.json").read_text())["tasks"]
        tasks, out = FAMILY + tuple(old) + tuple(fresh), HERE / "plans" / "runs.json"
    jobs = [dict(name=f"deep_{t}_h100", task=t, file="integration/legal_deep/submission_stage3_deep.py",
                 hardware="h100", memory_gb=48) for t in tasks]
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"wrote {out}: {len(jobs)} jobs: {[j['task'] for j in jobs]}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
