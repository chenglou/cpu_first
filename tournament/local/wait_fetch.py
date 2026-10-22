"""Wait until the named run records are on the `tournament-h1` Volume (records/), download them to
cpu_first/tournament/h1_records/, print a summary line each (a copy of learned_wiring/wait_fetch.py). Every name that
lands is printed as it lands; with --any the waiter returns as soon as one does.

    python wait_fetch.py [--any] NAME [NAME ...]        (polls every 60 s, up to 6 h)
"""
import json
import sys
import time
from pathlib import Path

import modal

RUNS = Path(__file__).resolve().parent.parent / "h1_records"


def summary(rec: dict) -> str:
    r = rec.get("result") or {}
    dp = r.get("depth_profile", {})
    seed = (r.get("seeds") or [{}])[0]
    ev = {k: round(v["exact_accuracy"], 4) for k, v in seed.get("evaluation", {}).items()}
    sdp = seed.get("depth_profile", {})
    rungs = [(g["time_steps"], g["correct_examples"]) for g in sdp.get("rungs", [])]
    oodr = [(g["time_steps"], g["correct_examples"]) for g in sdp.get("ood_n_rungs", [])]
    return (f"{rec['name']} rc={rec['returncode']} {rec.get('gpu', '')} maxT={dp.get('max_certified_time_steps')} "
            f"oodN_maxT={dp.get('ood_n_max_certified_time_steps')} train_s={seed.get('training_seconds', 0):.0f} "
            f"steps={seed.get('completed_training_steps')} eval_s={seed.get('evaluation_seconds', 0):.0f} eval={ev} "
            f"rungs={rungs} ood_rungs={oodr}")


def main(names, any_=False):
    vol = modal.Volume.from_name("tournament-h1")
    RUNS.mkdir(parents=True, exist_ok=True)
    pending = set(names)
    t0 = time.time()
    while pending and time.time() - t0 < 6 * 3600:
        try:
            have = {Path(e.path).stem: e for e in vol.listdir("records")}
        except Exception as exc:  # transient client errors: retry
            print("listdir failed:", repr(exc)[:200], flush=True)
            time.sleep(60)
            continue
        for name in sorted(pending & set(have)):
            dest = RUNS / f"{name}.json"
            with open(dest, "wb") as f:
                for chunk in vol.read_file(have[name].path):
                    f.write(chunk)
            print(summary(json.loads(dest.read_text())), flush=True)
            pending.discard(name)
            if any_:
                pending = set()
        if pending:
            time.sleep(60)
    print("still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if a != "--any"], "--any" in args)
