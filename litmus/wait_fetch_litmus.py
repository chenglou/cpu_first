"""Wait until the named run records are on the `litmus` Volume (records/), download them to cpu_first/litmus/records/,
print a summary line each (a copy of wider_menus/wait_fetch_wide.py).

    python wait_fetch_litmus.py NAME [NAME ...]        (polls every 60 s, up to 8 h)
"""
import json
import sys
import time
from pathlib import Path

import modal

RECORDS = Path(__file__).resolve().parent / "records"


def summary(rec: dict) -> str:
    r = rec.get("result") or {}
    dp = r.get("depth_profile", {})
    seed = (r.get("seeds") or [{}])[0]
    ev = {k: round(v["exact_accuracy"], 4) for k, v in seed.get("evaluation", {}).items()}
    sdp = seed.get("depth_profile", {})
    rungs = [(g["time_steps"], g["correct_examples"]) for g in sdp.get("rungs", [])]
    oodr = [(g["time_steps"], g["correct_examples"]) for g in sdp.get("ood_n_rungs", [])]
    score = (r.get("score") or {}).get("mean_exact_accuracy")
    return (f"{rec['name']} task={rec.get('task')} rc={rec['returncode']} {rec.get('gpu', '')} score={score} "
            f"maxT={dp.get('max_certified_time_steps')} oodN_maxT={dp.get('ood_n_max_certified_time_steps')} "
            f"train_s={seed.get('training_seconds', 0):.0f} steps={seed.get('completed_training_steps')} "
            f"eval_s={seed.get('evaluation_seconds', 0):.0f} eval={ev} rungs={rungs} ood_rungs={oodr}")


def main(names):
    vol = modal.Volume.from_name("litmus")
    RECORDS.mkdir(parents=True, exist_ok=True)
    pending = set(names)
    t0 = time.time()
    while pending and time.time() - t0 < 8 * 3600:
        try:
            have = {Path(e.path).stem: e for e in vol.listdir("records")}
        except Exception as exc:  # transient client errors (or no records yet): retry
            print("listdir failed:", repr(exc)[:200], flush=True)
            time.sleep(60)
            continue
        for name in sorted(pending & set(have)):
            dest = RECORDS / f"{name}.json"
            with open(dest, "wb") as f:
                for chunk in vol.read_file(have[name].path):
                    f.write(chunk)
            print(summary(json.loads(dest.read_text())), flush=True)
            pending.discard(name)
        if pending:
            time.sleep(60)
    print("still pending:", sorted(pending), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
