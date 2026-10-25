"""Each run of the eight fixed copies (make_fix_variants.py, ABLATION_S3.md §8) against the original run it replays.
Two comparisons per pair:
- every study's own script on the pair, unchanged: strict's checks/compare_runs.py and learned_loop's
  diag/compare_climb.py compare two records; sp_breadth's summarize.py (row), sp_ablation's analyze.py (facts) and
  generic_ops' summarize_runner.py summarize one, and their summaries are compared with the seconds left out; the
  bridge has no record comparison of its own, so it gets only the second;
- the same for every pair: the runner's compare (tools/modal_runner.py --mode compare: every field but the md5 and the
  log, the training trace included) and the climb's log lines (LR) with every decimal number and strict's work
  counts masked, since only those may differ.
Writes each pair's output to records/fix/variants/compare/ and prints one line per pair.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/compare_fix_variants.py [STUDY ...]
"""

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

CPU_FIRST = Path(__file__).resolve().parents[3]
OUT = CPU_FIRST / "integration/ablation_s3/records/fix/variants/compare"
SEEDS8 = (74, 1, 2, 3, 4, 5, 6, 7)
TASKS = ("minus1", "ten", "rho", "plus3", "tenrho", "m4", "xfirst")      # sp_breadth's other learned tasks

STRICT, LL, BRIDGE = "integration/strict", "integration/learned_loop", "bar3/bridge"
SPB, SPA, GOPS = "integration/sp_breadth", "integration/sp_ablation", "integration/generic_ops"
PAIRS = {   # study: [(the original record, the fixed copy's record)]
    "s3i_strict": [(f"{STRICT}/records/s3is_official.json", f"{STRICT}/fix/records/s3is_fix_official.json")]
    + [(f"{STRICT}/records/s3is_h100_seed{s}.json", f"{STRICT}/fix/records/s3is_fix_h100_seed{s}.json")
       for s in range(1, 8)],
    "sp2_strict": [(f"{STRICT}/records/sp2s_official.json", f"{STRICT}/fix/records/sp2s_fix_official.json")]
    + [(f"{STRICT}/records/sp2s_b200_seed{s}.json", f"{STRICT}/fix/records/sp2s_fix_b200_seed{s}.json") for s in SEEDS8],
    "ll": [(f"{LL}/records/ll2_official.json", f"{LL}/fix/records/ll2_fix_official.json")]
    + [(f"{LL}/records/b200/ll2_b200_seed{s}.json", f"{LL}/fix/records/ll2_fix_b200_seed{s}.json") for s in SEEDS8],
    "bridge": [(f"{BRIDGE}/records/runner/submission_stage3_bridge_official2.json",
                f"{BRIDGE}/fix/records/bridge_fix_official.json")]
    + [(f"{BRIDGE}/records/runner/submission_stage3_bridge2_b200_seed{s}.json",
        f"{BRIDGE}/fix/records/bridge_fix_b200_seed{s}.json") for s in SEEDS8],
    "spb_g3": [(f"{SPB}/records/g3_h1_h100.json", f"{SPB}/fix/records/spb_g3_fix_official.json")]
    + [(f"{SPB}/records/submission_spb_g3_b200_seed{s}.json", f"{SPB}/fix/records/spb_g3_fix_b200_seed{s}.json")
       for s in SEEDS8]
    + [(f"{SPB}/records/g3_{t}_h100.json", f"{SPB}/fix/records/g3fix_{t}_h100.json") for t in TASKS],
    "book_min": [(f"{SPA}/records/book_min_h100_seed74.json", f"{SPA}/fix/records/book_min_fix_official.json")]
    + [(f"{SPA}/records/book_min_b200_seed{s}.json", f"{SPA}/fix/records/book_min_fix_b200_seed{s}.json")
       for s in SEEDS8],
    "book_gen": [(f"{SPA}/records/book_gen_h100_seed74.json", f"{SPA}/fix/records/book_gen_fix_official.json")]
    + [(f"{SPA}/records/book_gen_b200_seed{s}.json", f"{SPA}/fix/records/book_gen_fix_b200_seed{s}.json")
       for s in SEEDS8],
    "g3": [(f"{GOPS}/records/runner/submission_stage3_g3_official.json", f"{GOPS}/fix/records/g3_fix_official.json")]
    + [(f"{GOPS}/records/runner/submission_stage3_g3_b200_seed{s}.json", f"{GOPS}/fix/records/g3_fix_b200_seed{s}.json")
       for s in SEEDS8],
}


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, CPU_FIRST / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def masked(text):
    """Seconds and strict's work counts out of a summary or log line."""
    text = re.sub(r" \[work [^]]*\]", "", text)
    return re.sub(r"\d+\.\d+(e[+-]?\d+)?", "#", text)


def climb_log(rec):
    return [masked(l) for l in rec["stdout"].splitlines() if l.startswith("LR ")]


def uniform(ref, new):
    out = subprocess.run([sys.executable, "tools/modal_runner.py", "--mode", "compare", ref, new], cwd=CPU_FIRST,
                         capture_output=True, text=True).stdout
    fields = [l.split("DIFFERS")[1].strip().split(":")[0] for l in out.splitlines() if "DIFFERS" in l]
    fields = [f for f in fields if f not in ("submission md5", "log without timings")]
    a, b = (climb_log(json.loads((CPU_FIRST / p).read_text())) for p in (ref, new))
    lines = sum(u != v for u, v in zip(a, b)) + abs(len(a) - len(b))
    first = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), None)
    return out, fields, lines, len(a), (a[first], b[first]) if first is not None else None


def summary_diff(fn, ref, new):
    """A study's one-record summary of both records, compared with the seconds masked."""
    got = []
    for p in (ref, new):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fn(str(CPU_FIRST / p))
        got.append(buf.getvalue() if r is None else r)
    return got


def own(study, ref, new):
    """The study's own script on the pair: (its output, whether it finds them the same)."""
    if study in ("s3i_strict", "sp2_strict"):
        out = subprocess.run([sys.executable, f"{STRICT}/checks/compare_runs.py", ref, new, "--quiet"], cwd=CPU_FIRST,
                             capture_output=True, text=True).stdout
        return out, "IDENTICAL" in out and "SCHEDULE same" in out and "DIFFERS" not in out
    if study == "ll":
        out = subprocess.run([sys.executable, f"{LL}/diag/compare_climb.py", ref, new], cwd=CPU_FIRST,
                             capture_output=True, text=True).stdout
        heads = [l for l in out.splitlines() if l.startswith(("schedule events", "logged losses", "climb lines"))]
        rounds = [l.split(":", 1)[1].strip() for l in out.splitlines() if l.startswith("round,")]
        return out, len(heads) == 3 and all("identical" in l for l in heads) and len(set(rounds)) == 1
    if study == "spb_g3":
        row = load(f"{SPB}/summarize.py").row
        a, b = (row(json.loads((CPU_FIRST / p).read_text())) for p in (ref, new))
        for x in (a, b):
            x["last_round"] = masked(x["last_round"] or "")
        keys = [k for k in a if k not in ("name", "md5", "train", "eval", "round_s", "frozen_s", "done")]
        diff = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
        return json.dumps({"compared": keys, "differ": diff}, default=str, indent=1), not diff
    if study in ("book_min", "book_gen"):
        facts = load(f"{SPA}/analyze.py").facts
        a, b = (facts(str(CPU_FIRST / p)) for p in (ref, new))
        keys = ("certified", "T", "splits", "steps", "climb_step", "round_step", "frozen_step", "T_step", "rounds",
                "round_lib", "error")
        diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
        return json.dumps({"compared": keys, "differ": diff}, default=str, indent=1), not diff
    if study == "g3":
        mod = load(f"{GOPS}/summarize_runner.py")
        a, b = summary_diff(mod.main, ref, new)
        a, b = ([masked(l) for l in s.splitlines()[1:]] for s in (a, b))       # (line 1 names the file and md5)
        return "\n".join(["original:"] + a + ["fixed:"] + b), a == b
    return "(no record comparison of its own)", None


def main(studies):
    OUT.mkdir(parents=True, exist_ok=True)
    for study in studies:
        for ref, new in PAIRS[study]:
            name = Path(new).stem
            if not (CPU_FIRST / new).exists():
                print(f"{study:10s} {name:34s} missing")
                continue
            rec = json.loads((CPU_FIRST / new).read_text())
            dp = (rec.get("result") or {}).get("depth_profile") or {}
            T = (dp.get("max_certified_time_steps"), dp.get("ood_n_max_certified_time_steps"))
            text, fields, lines, n, first = uniform(ref, new)
            own_out, own_same = own(study, ref, new)
            (OUT / f"{name}.txt").write_text(f"{ref}\n{new}\n\n== the study's own script\n{own_out}\n\n"
                                             f"== the runner's compare\n{text}\n"
                                             + (f"\nfirst climb line differing:\n  {first[0]}\n  {first[1]}\n"
                                                if first else ""))
            own_txt = {True: "same", False: "DIFFERENT", None: "-"}[own_same]
            print(f"{study:10s} {name:34s} T = {T[0]} / {T[1]}; own script: {own_txt}; fields differing: "
                  f"{', '.join(fields) or 'none'}; climb lines differing (seconds masked): {lines} of {n}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(PAIRS))
