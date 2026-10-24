"""STRICT.md §4's tables: each strict record next to the certified record of the same seed and hardware (the same
climb line for line? the same round and freeze steps?), with its certification and its work.

    python3 integration/strict/checks/results_table.py KIND STRICT_REC.json ...
    (KIND sp2: the certified records are integration/self_practice/records/runner/submission_stage3_sp2_{official,b200_seedN}.json;
     s3i: integration/records/s3i_{official,h100_seedN}.json)
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_runs as C  # noqa: E402
import work_table as WT  # noqa: E402

CPU = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


def certified(kind, rec):
    seed = rec["result"]["seeds"][0]["seed"]
    gpu = rec.get("gpu") or ""
    official = rec.get("manifest_file") is not None and "H100" in gpu and seed == 74
    if kind == "sp2":
        d = os.path.join(CPU, "integration", "self_practice", "records", "runner")
        name = "submission_stage3_sp2_official.json" if official else f"submission_stage3_sp2_b200_seed{seed}.json"
    else:
        d = os.path.join(CPU, "integration", "records")
        name = "s3i_official.json" if official else f"s3i_h100_seed{seed}.json"
    p = os.path.join(d, name)
    return p if os.path.exists(p) else None


def after(out, step):
    prog = [(int(m.group(1)), float(m.group(2))) for m in
            re.finditer(r"^step=(\d+) loss=\S+ accuracy=\S+ elapsed=([0-9.]+)s", out, re.M)]
    return next((e for st, e in prog if st >= step), None)


def main():
    kind, paths = sys.argv[1], sys.argv[2:]
    print("| record | device | seed | max T, seen / unseen | round at step (certified) | frozen at step (certified) | "
          "climb line for line | frozen by s (certified) | training s | climb work | largest search's work | "
          "budget stops |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        rec = json.load(open(p))
        w = WT.scan(p)
        c = certified(kind, rec)
        crec = json.load(open(c)) if c else None
        same = (C.climb_lines(rec["stdout"]) == C.climb_lines(crec["stdout"])) if crec else None
        cw = WT.scan(c) if c else {}
        sr = rec["result"]["seeds"][0]
        fb = after(rec["stdout"], w["frozen_step"]) if w["frozen_step"] else None
        cfb = after(crec["stdout"], cw["frozen_step"]) if crec and cw.get("frozen_step") else None
        sm = w["search_max"]
        print(f"| `{os.path.basename(p)[:-5]}` | {w['gpu'].replace('NVIDIA ', '')} | {sr['seed']} | "
              f"{w['max_t'][0]} / {w['max_t'][1]} | {w['round_step']} ({cw.get('round_step')}) | "
              f"{w['frozen_step']} ({cw.get('frozen_step')}) | {'identical' if same else 'differs' if crec else '-'} | "
              f"{fb and round(fb)} ({cfb and round(cfb)}) | {round(sr['training_seconds'])} | {w['work']:.3g} in "
              f"{w['secs']:.0f} s | {sm and f'{sm[0]:.3g} in {sm[1]} s'} | {rec['stdout'].count('work budget')} |")


if __name__ == "__main__":
    main()
