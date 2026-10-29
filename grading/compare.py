"""Compares each calibration rerun with the result the hosted evaluator stored for the same upload, field by field:
- the reruns are grading/raw/cal_<id>_*.json, from tools/modal_runner.py;
- the stored results are grading/sources/<id>.result.json, from extract.py.

Writes grading/calibration.json, which holds results only, no participant source.

    python3 grading/compare.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXACT = ("max_t", "ood_max_t", "splits", "rungs", "ood_rungs", "model_state_elements")   # equal means same outcome


def facts(res: dict) -> dict:
    s = (res.get("seeds") or [{}])[0]
    dp, sdp = res.get("depth_profile") or {}, s.get("depth_profile") or {}
    return {"max_t": dp.get("max_certified_time_steps"), "ood_max_t": dp.get("ood_n_max_certified_time_steps"),
            "mean_exact_accuracy": (res.get("score") or {}).get("mean_exact_accuracy"),
            "splits": {k: v.get("correct_examples") for k, v in sorted((s.get("evaluation") or {}).items())},
            "rungs": [r.get("correct_examples") for r in sdp.get("rungs") or []],
            "ood_rungs": [r.get("correct_examples") for r in sdp.get("ood_n_rungs") or []],
            "steps": s.get("completed_training_steps"), "training_seconds": s.get("training_seconds"),
            "model_state_elements": s.get("model_state_elements"), "final_train_loss": s.get("final_train_loss")}


rows = []
for path in sorted((HERE / "raw").glob("cal_*.json")):
    rec = json.loads(path.read_text())
    sid = path.stem.split("_")[1]
    stored = facts(json.loads((HERE / "sources" / f"{sid}.result.json").read_text()))
    rerun = facts(rec.get("result") or {})
    same = {k: stored[k] == rerun[k] for k in EXACT}
    rows.append({"id": sid, "name": rec["name"], "submission_md5": rec["submission_md5"], "gpu": rec["gpu"],
                 "network": rec["runner"].get("network"), "data_md5": rec["runner"]["data_md5"],
                 "returncode": rec["returncode"], "stored": stored, "rerun": rerun, "same": same})
    acc = lambda f: f"{f['mean_exact_accuracy']:.4f}" if f["mean_exact_accuracy"] is not None else "-"
    print(f"{rec['name']}: T {stored['max_t']}/{stored['ood_max_t']} stored, {rerun['max_t']}/{rerun['ood_max_t']} rerun; "
          f"accuracy {acc(stored)} / {acc(rerun)}; steps {stored['steps']} / {rerun['steps']}; "
          f"splits {stored['splits']} / {rerun['splits']}; exact: {[k for k in EXACT if same[k]]}, "
          f"differ: {[k for k in EXACT if not same[k]]}; network {rows[-1]['network']}")
(HERE / "calibration.json").write_text(json.dumps(rows, indent=1) + "\n")
print(f"{len(rows)} reruns compared -> grading/calibration.json")
