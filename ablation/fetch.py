"""Wait for run records on the `ablation` Volume, download them to runs/, print one summary line each.

    ~/github/sotaku/venv/bin/python fetch.py [--hours H] PLAN.json|NAME ...   (polls every 60 s; exits when all are in)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def summary(rec: dict) -> str:
    r = rec.get("result") or {}
    dp = r.get("depth_profile", {})
    seed = (r.get("seeds") or [{}])[0]
    out = rec.get("stdout", "")
    frozen = next((l.split("step=")[1].split()[0] for l in out.splitlines() if l.startswith("LR frozen")), "-")
    switch = next((l.split("step=")[1].split()[0] for l in out.splitlines() if l.startswith("LR switch")), "-")
    return (f"{rec['name']} rc={rec['returncode']} maxT={dp.get('max_certified_time_steps')} "
            f"oodN_maxT={dp.get('ood_n_max_certified_time_steps')} switch={switch} frozen={frozen} "
            f"train_s={seed.get('training_seconds', 0):.0f} eval_s={seed.get('evaluation_seconds', 0):.0f}")


def names_of(args) -> list:
    out = []
    for a in args:
        if a.endswith(".json"):
            out += [j["name"] for j in json.loads(Path(a).read_text())]
        else:
            out.append(a)
    return out


def main(argv):
    hours = 6.0
    if argv[:1] == ["--hours"]:
        hours, argv = float(argv[1]), argv[2:]
    pending = {n for n in names_of(argv) if not (RUNS / f"{n}.json").exists()}
    vol = modal.Volume.from_name("ablation")
    RUNS.mkdir(exist_ok=True)
    t0 = time.time()
    while pending and time.time() - t0 < hours * 3600:
        try:
            have = {Path(e.path).stem: e for e in vol.listdir("/")}
        except Exception as exc:  # transient client errors: retry
            print("listdir failed:", repr(exc)[:200], flush=True)
            time.sleep(60)
            continue
        for name in sorted(pending & set(have)):
            dest, part = RUNS / f"{name}.json", RUNS / f"{name}.json.part"
            with open(part, "wb") as f:                      # whole records only: write, then rename
                for chunk in vol.read_file(have[name].path):
                    f.write(chunk)
            part.rename(dest)
            print(time.strftime("%H:%M:%S"), summary(json.loads(dest.read_text())), flush=True)
            pending.discard(name)
        if pending:
            time.sleep(60)
    print("still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
