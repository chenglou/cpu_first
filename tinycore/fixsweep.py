"""The fix comparison: every fix on the sizes where joint training fails. Rows: runs/fixes.jsonl.

    python -m tinycore.fixsweep
"""
import os
from concurrent.futures import ProcessPoolExecutor
import json

from .fixes import HERE, run_fix

SIZES = [(2, 4, 3), (3, 2, 2), (4, 2, 1), (4, 2, 2), (5, 2, 1)]
FIX_CHOICES = {"joint": ("est", "borrow"), "alternate": ("est", "borrow"), "staged": ("est", "borrow"),
               "mixed": ("est", "borrow"), "repeat": ("est", "borrow"), "hardem": ("est", "borrow"),
               "hardem_local": ("est",), "clock": ("est",), "clock_given": ("est",), "checks": ("est", "borrow")}


def jobs(seeds=6):
    out = []
    for B, w, k in SIZES:
        for fix, choices in FIX_CHOICES.items():
            if fix in ("staged", "mixed") and k == 1:
                continue
            for ch in choices:
                for s in range(seeds):
                    kw = {}
                    f = fix
                    if fix == "hardem_local":
                        f, kw = "hardem", {"estep": "local"}
                    if fix == "clock_given":
                        f, kw = "clock", {"pick": "given", "noise": 0.3}
                    if f == "hardem" and B ** k > 9:
                        kw["steps"] = 1500
                    out.append({"fix": f, "label": fix, "cfg": {"B": B, "w": w, "k": k, "choice": ch, "cell": "slot",
                                                                 "init": 0.1}, "seed": s, "kw": kw})
    return out


def run(job):
    r = run_fix(dict(job, kw=dict(job["kw"])))
    r["label"] = job["label"]
    return r


if __name__ == "__main__":
    js = jobs()
    with ProcessPoolExecutor(6) as ex, open(os.path.join(HERE, "runs", "fixes.jsonl"), "a") as f:
        for r in ex.map(run, js):
            f.write(json.dumps(r) + "\n")
            f.flush()
            extra = f" mult {r['multipliers']} modN {r['exact_mod_N']:.2f}" if "multipliers" in r else ""
            print(f"{r['label']:12s} B{r['B']} w{r['w']} k{r['k']} {r['cell']}/{r['choice']} seed {r['seed']}: hit "
                  f"{r['hit']} final {r['final']:.3f} given {r['exact_given']:.2f} slots {r['slots']}{extra} "
                  f"({r['seconds']}s)", flush=True)
