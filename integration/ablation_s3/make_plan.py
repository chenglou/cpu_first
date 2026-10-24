"""A runner plan (tools/README.md) for some ablation arms (plans/arms.json) and seeds, on one hardware, at the official
budget or a relaxed one. Job names: <arm>_<hardware>[_r<budget>]_seed<seed>.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/make_plan.py NAME --arms a,b --seeds 74 \
        [--hardware h100] [--budget 10800] [--only-missing]
    (--only-missing: leave out jobs whose record is already in records/, to relaunch what a stopped run lost)
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seeds", default="74")
    ap.add_argument("--hardware", default="h100")
    ap.add_argument("--budget", type=int, default=0, help="training seconds (0: the official 3,600, untouched)")
    ap.add_argument("--only-missing", action="store_true")
    a = ap.parse_args()
    arms = json.loads((HERE / "plans" / "arms.json").read_text())
    jobs = []
    for arm in a.arms.split(","):
        spec = arms[arm]
        for seed in [int(s) for s in a.seeds.split(",")]:
            name = f"{arm}_{a.hardware}{f'_r{a.budget}' if a.budget else ''}_seed{seed}"
            if a.only_missing and (HERE / "records" / f"{name}.json").exists():
                continue
            job = {"name": name, "file": spec["file"], "seed": seed, "hardware": a.hardware}
            if spec.get("set"):
                job["set"] = spec["set"]
            if a.budget:
                job["manifest"] = {"runtime": {"total_training_time_seconds": a.budget}}
            jobs.append(job)
    path = HERE / "plans" / f"{a.name}.json"
    path.write_text(json.dumps(jobs, indent=1) + "\n")
    print(f"{path}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
