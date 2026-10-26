"""Official-evaluator records of one file, one line each (FAST.md §8): the seed, the certified T on both ladders, the
three splits, and when the climb started, found the round and the answer program, and froze (step and training clock,
the clock read off the evaluator's progress line before the step), the climb's work by the answer program, the
round's programs and the answer programs, training and evaluation seconds, peak host memory; for a record that does
not certify, its last climb lines.

    python3 integration/combined_fast/checks/summarize_seeds.py RECORD.json ...
"""
import json
import re
import sys


def clock(out, step):
    """The training clock at step, about: the evaluator's progress line before it (every 100 steps) plus the climb's own
    unit seconds since (the training steps between them add a little more)."""
    base, at = 0.0, 0
    for l in out:
        m = re.match(r"step=(\d+) .*elapsed=([\d.]+)s", l)
        if m and int(m.group(1)) <= step:
            at, base = int(m.group(1)), float(m.group(2))
    units = 0.0
    for l in out:
        m = re.match(r"LR (climb|outer) step=(\d+) ", l)
        if m and at < int(m.group(2)) <= step:
            t = re.findall(r"\((\d+\.\d)s\)$", l) or re.findall(r" (\d+\.\d)s ", l)[:1]
            units += float(t[0]) if t else 0.0
    return round(base + units)


for path in sys.argv[1:]:
    r = json.load(open(path))
    out = r["stdout"].splitlines()
    res = r.get("result") or {}
    s = (res.get("seeds") or [{}])[0]
    dp = s.get("depth_profile") or {}
    ev = s.get("evaluation") or {}
    marks = {}
    for l in out:
        m = re.match(r"LR (climb_starts|round_found|frozen) step=(\d+)", l)
        if m:
            marks.setdefault(m.group(1), int(m.group(2)))
        m = re.match(r"LR outer step=(\d+) answer program: cost (\S+)", l)
        if m and m.group(2) != "None":
            marks.setdefault("answer", int(m.group(1)))
    work = next((m.group(1) for l in out for m in [re.search(r"\[work \+\S+, (\S+) in all\]", l)]
                 if m and l.startswith("LR outer") and "cost None" not in l), None)
    rf = next((l for l in out if l.startswith("LR round_found")), "")
    progs = re.search(r"programs: (\[.*?\]) rounds:", rf)
    rounds = re.search(r"rounds: (\{.*?\}) library:", rf)
    t = lambda k: f"step {marks[k]} (about {clock(out, marks[k])} s)" if k in marks else "-"
    print(f"== {path.split('/')[-1]}: seed {r['manifest']['runtime']['seeds']}, rc {r['returncode']}, md5 "
          f"{r['submission_md5'][:8]}; T {dp.get('max_certified_time_steps')} / {dp.get('ood_n_max_certified_time_steps')}; "
          + ", ".join(f"{k} {v.get('correct_examples')}/{v.get('example_count')}" for k, v in ev.items()))
    print(f"   climb starts {t('climb_starts')}, answer program {t('answer')}, frozen {t('frozen')}; climb work by then "
          f"{work}; training {round(s.get('training_seconds') or 0, 1)} s ({s.get('completed_training_steps')} steps), "
          f"evaluation {round(s.get('evaluation_seconds') or 0, 1)} s; peak RSS {r['usage'].get('peak_session_rss_mb')} MB")
    print(f"   programs {progs.group(1) if progs else '-'}")
    print(f"   rounds {rounds.group(1)[:400] if rounds else '-'}")
    if dp.get("max_certified_time_steps") != 64 or dp.get("ood_n_max_certified_time_steps") != 64:
        print("   NOT CERTIFIED; the last climb lines:")
        for l in [l for l in out if l.startswith(("LR climb", "LR outer"))][-6:]:
            print("     ", l[:220])
