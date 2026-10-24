"""One line per evaluator record (the shared runner's): certified T on both ladders, the worst rung, the splits, the
training and evaluation seconds, the model's numbers, and the run's own events read from its log: the round found, the
answer program found, T resolved, frozen (each by step and by the evaluator's elapsed time at the first progress line
after it), and the answer programs.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/learned_loop/summarize.py REC.json ... [--md]
"""
import json
import re
import sys


def elapsed_after(lines, step):
    """The evaluator's elapsed seconds at its first progress line at or after this step."""
    for l in lines:
        m = re.match(r"step=(\d+) loss=\S+ accuracy=\S+ elapsed=([\d.]+)s", l)
        if m and int(m.group(1)) >= step:
            return float(m.group(2))
    return None


def event(lines, pattern):
    for l in lines:
        m = re.match(pattern, l)
        if m:
            return int(m.group(1))
    return None


def summary(path):
    r = json.load(open(path))
    lines = r.get("stdout", "").splitlines()
    out = dict(record=path.split("/")[-1].removesuffix(".json"), md5=(r.get("submission_md5") or "")[:8],
               gpu=r.get("gpu"), seed=None)
    res = r.get("result")
    if not res:
        err = (r.get("stderr") or "").strip().splitlines()
        out["crashed"] = err[-1][:160] if err else f"returncode {r.get('returncode')}"
        return out
    s = res["seeds"][0]
    dp = s["depth_profile"]
    rungs = dp.get("rungs", []) + dp.get("ood_n_rungs", [])
    out.update(seed=s["seed"], T=dp["max_certified_time_steps"], T_ood=dp["ood_n_max_certified_time_steps"],
               worst_rung=min((g["correct_examples"], g["example_count"]) for g in rungs) if rungs else None,
               splits={k: f"{v['correct_examples']}/{v['example_count']}" for k, v in s["evaluation"].items()},
               train_s=round(s["training_seconds"], 1), eval_s=round(s["evaluation_seconds"], 1),
               numbers=s["model_state_elements"])
    for name, pat in (("round", r"LR climb step=(\d+) L\d+ round(?: again)?.*: cost (?!None)"),
                      ("answer_program", r"LR round_found step=(\d+)"),
                      ("T_resolved", r"LR io_stage=5 T_resolved step=(\d+)"),
                      ("frozen", r"LR frozen step=(\d+)")):
        st = event(lines, pat)
        out[name] = None if st is None else (st, elapsed_after(lines, st))
    out["outer"] = [l.split(" answer program: ", 1)[1][:200] for l in lines if l.startswith("LR outer step=")]
    out["programs"] = next((l[len("ROUND "):] for l in lines if l.startswith("ROUND ")), None)
    return out


def main(argv):
    md = "--md" in argv
    for p in [a for a in argv if not a.startswith("--")]:
        s = summary(p)
        if md:
            if "crashed" in s:
                print(f"| {s['record']} | crashed: {s['crashed']} |")
                continue
            cert = "yes" if s["T"] == 64 and s["T_ood"] == 64 and s["worst_rung"][0] == s["worst_rung"][1] else "no"
            ev = lambda k: f"{s[k][0]} ({s[k][1]:.0f} s)" if s.get(k) and s[k][1] is not None else (
                str(s[k][0]) if s.get(k) else "-")
            print(f"| {s['record']} | {s['gpu']} | {s['seed']} | {s['T']} / {s['T_ood']} | {cert} | "
                  f"{', '.join(f'{k} {v}' for k, v in s['splits'].items())} | {ev('round')} | {ev('answer_program')} | "
                  f"{ev('T_resolved')} | {ev('frozen')} | {s['train_s']} / {s['eval_s']} | {s['programs']} |")
        else:
            print(json.dumps(s))


if __name__ == "__main__":
    main(sys.argv[1:])
