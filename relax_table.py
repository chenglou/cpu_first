"""Print every relaxed-rules result from runs/ (JSON when the run finished, the last logged eval otherwise)."""

import glob
import json
import os
import re

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def last_eval(log):
    lines = [ln for ln in open(log) if ln.startswith("step ")]
    return lines[-1].strip() if lines else "(no eval yet)"


def show(pattern, fmt, where=RUNS):
    for log in sorted(glob.glob(os.path.join(where, pattern))):
        js = log[:-4] + ".json"
        name = os.path.basename(log)[:-4]
        if os.path.exists(js):
            print(f"{name}: {fmt(json.load(open(js)))}")
        else:
            print(f"{name}: unfinished, last eval: {last_eval(log)[:200]}")


def step_fmt(d):
    parts = [f"train {d['train_time'] / 60:.1f} min"]
    for k in ("single_step_fresh_100k", "single_step_pool_100k"):
        if k in d:
            parts.append(f"{k}={d[k]:.5f}")
    for k in ("ladder_fresh", "ladder_pool", "depth_t", "depth_ood_n_t"):
        if k in d:
            parts.append(f"{k} max_t={d[k]['max_t']}")
    parts.append(f"total {d['total_time'] / 60:.1f} min")
    return ", ".join(parts)


def trace_fmt(d):
    parts = [f"L={d['L']}", f"train {d['train_time'] / 60:.1f} min"]
    for k in ("single_step_pool", "single_step_unseen", "single_step_fresh"):
        if k in d:
            parts.append(f"{k}: ans={d[k][0]:.5f} trace={d[k][1]:.5f} firsterr={d[k][2]}")
    for k in ("ladder_pool", "ladder_unseen", "ladder_fresh", "depth_t", "depth_ood_n_t"):
        if k in d:
            parts.append(f"{k} max_t={d[k]['max_t']} accs={[round(v, 3) for v in d[k]['accs'].values()]}")
    parts.append(f"total {d['total_time'] / 60:.1f} min")
    return ", ".join(parts)


if __name__ == "__main__":
    print("== relaxation 1: per-step labels, H1-shaped finite data (looped transformer)")
    show("relax1_*.log", step_fmt)
    print("== relaxation 2: show your work (trace GPT)")
    show("relax2_*.log", trace_fmt)
    print("== relaxation 3: periodic units tied to N")
    for f in sorted(glob.glob(os.path.join(RUNS, "relax3_train_*_gn.json")) +
                    glob.glob(os.path.join(RUNS, "relax3_train_bidig.json")) +
                    glob.glob(os.path.join(RUNS, "relax3_train_mono_adam.json"))):
        r = json.load(open(f))["results"]
        row = {re.sub(r"_phase_small|_plain", "", k): f"{v['success']}/{v['seeds']}" for k, v in r.items()}
        print(os.path.basename(f), row)
    if os.path.exists(os.path.join(RUNS, "relax3_cert.json")):
        for k, v in json.load(open(os.path.join(RUNS, "relax3_cert.json"))).items():
            print(f"relax3_cert {k}: train {v['train_time']:.2f}s single-step(24b,200k)={v['single_step_24b_200k']} "
                  f"ladder24 max_t={v['ladder_24b']['max_t']} H1 depth_t={v['depth_t']['max_t']} "
                  f"ood={v['depth_ood_n_t']['max_t']}")
    print("== relaxation 4: unlimited fresh data (looped transformer)")
    show("relax4_*.log", step_fmt)
    print("== Modal runs (runs/modal; fetch with: modal volume get relaxed-runs <name>.json runs/modal/)")
    show("t2_*.log", trace_fmt, os.path.join(RUNS, "modal"))
    show("r[14]_*.log", step_fmt, os.path.join(RUNS, "modal"))
