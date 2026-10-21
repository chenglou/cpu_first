"""Local check that submission_simple.py trains exactly as submission_ablate.py with the simplified recipe's settings
(plans/w7_simple.json, "LMWnosched"): identical losses and parameters at every step (batch 32, 90 steps, across the
warm start's end at step 70). Small: well under 1 GB and a minute. Needs the H1 jsonl files (H1_DATA).
  python local_equiv_simple.py
"""
import os, time
from local_equiv import module, run, HERE
from local_adder import load
import json

SETS = {k: eval(v) for k, v in next(j for j in json.load(open(os.path.join(HERE, "plans", "w7_simple.json")))
                                    if j["name"] == "simple_LMWnosched_s74")["set"].items() if k != "TABLE_EVERY"}


def main():
    rows = load("train", 4000)
    t0 = time.time()
    a = run(module(os.path.join(HERE, "submission_ablate.py"), "abl"), 90, 32, 1, rows, SETS)
    b = run(module(os.path.join(HERE, "submission_simple.py"), "simple"), 90, 32, 1, rows)
    same = all(x[0] == y[0] and __import__("torch").equal(x[1], y[1]) and __import__("torch").equal(x[2], y[2])
               for x, y in zip(a, b))
    print(f"submission_simple.py vs submission_ablate.py {SETS}: 90 steps, identical losses and parameters at every "
          f"step: {same} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
