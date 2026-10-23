"""Tables for INTEGRATION.md from the run records (integration/records/*.json: the evaluator's runs; lib/records/*.json:
the climbs). `python summarize.py` prints them; `python summarize.py fill INTEGRATION.md` replaces the text between
<!-- NAME --> and <!-- /NAME --> markers.

    cd ~/github/one-layer/cpu_first/integration && ~/github/sotaku/venv/bin/python summarize.py fill INTEGRATION.md
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "records"
CLIMB = HERE / "lib" / "records"
GEN = HERE.parent / "general_lessons" / "records"
TOUR = HERE.parent / "tournament" / "h1_records"
TASK_NAMES = {"h1": "H1: x^(2^T) mod N", "rho": "y^2 + 1", "ten": "10 y^2", "cube": "x^(3^T) (y^3)",
              "mont": "Montgomery: y^2 10^-8 mod N", "tenrho": "10 y^2 + 1 (held out)", "plus3": "y^2 + 3 (held out)",
              "minus1": "y^2 - 1 (held out)", "m4": "M4 (practice tier, every T = 8)"}


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def seed_of(rec: dict) -> dict:
    return ((rec.get("result") or {}).get("seeds") or [{}])[0]


def certified(rec: dict) -> bool:
    dp = (rec.get("result") or {}).get("depth_profile", {})
    return dp.get("max_certified_time_steps") == 64 and dp.get("ood_n_max_certified_time_steps") == 64


def maxT(rec: dict) -> str:
    dp = (rec.get("result") or {}).get("depth_profile", {})
    return f"{dp.get('max_certified_time_steps')} / {dp.get('ood_n_max_certified_time_steps')}"


def rungs(rec: dict) -> str:
    sdp = seed_of(rec).get("depth_profile", {})
    allr = [g["correct_examples"] for g in sdp.get("rungs", []) + sdp.get("ood_n_rungs", [])]
    if allr and all(c == 768 for c in allr):
        return "768/768 every rung"
    worst = min(allr) if allr else None
    return f"not all (worst rung {worst}/768)"


def splits(rec: dict) -> str:
    ev = seed_of(rec).get("evaluation", {})
    return ", ".join(f"{k} {v['correct_examples']:,}/{v['example_count']:,}" for k, v in sorted(ev.items())) or "-"


def frozen_step(rec: dict) -> str:
    for line in rec.get("stdout", "").splitlines():
        if line.startswith("LR frozen step="):
            return line.split()[2].split("=")[1]
    return "not frozen"


def chosen(rec: dict) -> str:
    """What the model chose: stage 1, the selected lesson set and round options and the set's chain; stage 3, the
    round program."""
    out = rec.get("stdout", "")
    sel = [l for l in out.splitlines() if l.startswith("LESSONS selected=")]
    if sel:
        m = re.search(r"selected=(\w+) selector=([\w:.,]+) .*round: (rmul=\S+ radd=\S+ rcube=\S+)", sel[-1])
        wiring = {re.search(r"set=(\w+)", l).group(1): re.search(r"chain=(\S+)", l).group(1)
                  for l in out.splitlines() if l.startswith("WIRING set=")}
        if m:
            s = m.group(1)
            return f"{s} ({m.group(2)}); {m.group(3)}; {s}'s chain: {wiring.get(s, '?').replace('_', ' ')}"
    rnd = [l for l in out.splitlines() if l.startswith("ROUND ")]
    if rnd:
        return "round programs (tournament weights): " + rnd[-1][6:].replace(" | ", "; ")
    return "-"


def device_of(rec: dict) -> str:
    gpu = rec.get("gpu", "") or ""
    return next((d for d in ("H100", "H200", "B200") if d in gpu), "cpu")


def seed_in(rec: dict) -> int:
    """The run's seed: the evaluator's result, else the manifest's runtime.seeds (a crashed run has no result)."""
    sd = seed_of(rec)
    if "seed" in sd:
        return sd["seed"]
    m = rec.get("manifest")
    return ((m.get("runtime") or {}).get("seeds") or [74])[0] if isinstance(m, dict) else 74


def md5_of(rec: dict) -> str:
    return hashlib.md5(rec.get("submission", "").encode()).hexdigest()[:8]


def table_runs(names: list) -> str:
    head = ("| record | task | device | seed | max T, seen / unseen moduli | rungs | splits | training s (frozen at step) "
            "| evaluation s | what it chose | md5 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for n in names:
        rec = load(REC / f"{n}.json")
        if rec is None:
            rows.append(f"| {n} | | | | (not run or not fetched) | | | | | | |")
            continue
        if rec.get("returncode"):
            err = (rec.get("stderr") or "").strip().splitlines()[-1:] or ["?"]
            rows.append(f"| {n} | {rec.get('task')} | {device_of(rec)} | {seed_in(rec)} | crashed ({err[0][:90]}) "
                        f"| | | | | | {md5_of(rec)} |")
            continue
        sd = seed_of(rec)
        rows.append(f"| {n} | {rec.get('task')} | {device_of(rec)} | {seed_in(rec)} | {maxT(rec)} | {rungs(rec)} | "
                    f"{splits(rec)} | {sd.get('training_seconds', 0):.0f} ({frozen_step(rec)}) | "
                    f"{sd.get('evaluation_seconds', 0):.0f} | {chosen(rec)} | {md5_of(rec)} |")
    return head + "\n" + "\n".join(rows)


def proxy(files=(("proxy_r11_lib.log", 5), ("proxy_r11_lib_cost6.log", 6))) -> dict:
    """The round proxy (lib/round_proxy.py: the round's search alone, with a library's routines, each task's T = 1
    answers): task -> the outcome at each cost bound."""
    out = {}
    for f, c in files:
        try:
            lines = (REC / f).read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("ROUND "):
                d = json.loads(line[6:])
                out.setdefault(d["task"], {})[c] = (d["programs"] or [None])[0]
    return out


def acceptance() -> str:
    head = ("| task | general lessons (LESSONS.md, the general file unchanged) | stage 1 (this file unchanged): max T "
            "seen / unseen | stage 1: what it chose | stage 3: learned? (H1: the evaluator's run; the others: the "
            "round's search over the routines the H1 run discovered, cost <= 5 as the file / 6) | the round's search "
            "with LIBRARY r11's routines: cost <= 5 / 6 |\n|---|---|---|---|---|---|")
    px = proxy()
    pd = proxy((("proxy_s3d_lib_cost5.log", 5), ("proxy_s3d_lib_cost6_40m.log", 6)))
    rows = []
    for t in ("h1", "rho", "ten", "cube", "mont", "tenrho", "plus3", "minus1", "m4"):
        gname = {"h1": "gen_h1_official", "m4": "gen_m4_relaxed_h100"}.get(t, f"gen_{t}_s74")
        g = load(GEN / f"{gname}.json")
        gtxt = (f"{'certified' if certified(g) else 'not certified'} ({maxT(g)})" if g else "-")
        s1 = load(REC / ("s1_official.json" if t == "h1" else
                         "s1_m4_relaxed_h100.json" if t == "m4" else f"s1_{t}_s74.json"))
        pt, ps = px.get(t, {}), pd.get(t, {})
        ptxt = " / ".join("-" if c not in pt else (f"`{pt[c]}`" if pt[c] else "none") for c in (5, 6))
        if t == "h1":
            s3 = load(REC / "s3f_official.json")
            s3txt = (f"{'certified' if certified(s3) else 'not certified'} ({maxT(s3)}, the official manifest): "
                     f"{chosen(s3)}") if s3 else "(pending)"
        elif t == "m4":
            s3txt = "no: the round's search reads T = 1 answers, and M4 has none"
        else:
            s3txt = " / ".join("-" if c not in ps else (f"`{ps[c]}`" if ps[c] else "no round") for c in (5, 6))
        rows.append(f"| {TASK_NAMES[t]} | {gtxt} | {maxT(s1) if s1 else '(pending)'} | {chosen(s1) if s1 else ''} | "
                    f"{s3txt} | {ptxt} |")
    return head + "\n" + "\n".join(rows)


def climbs() -> str:
    head = ("| climb | cells | device | levels | tasks solved (in order) | the round found | seconds |\n"
            "|---|---|---|---|---|---|---|")
    rows = []
    for f in sorted(CLIMB.glob("c*.json")):
        r = load(f)
        if r is None:
            continue
        cells = "learned (seed {})".format(r["config"].get("seed", 0)) if r["config"].get("learned_cells") else "exact"
        if r.get("climb") == "not started":
            rows.append(f"| {r['name']} | {cells}: {r['cells'].get('right_report')} | {r['device']} | - | the cells never "
                        f"passed their own checks: the climb waits | - | {r['seconds']} |")
            continue
        order = sorted(r.get("solved_at", {}).items(), key=lambda kv: kv[1])
        rows.append(f"| {r['name']} | {cells} | {r['device']} | {len(r.get('history', []))} | "
                    f"{', '.join(f'{t} (L{l})' for t, l in order)} | {'; '.join(r.get('rounds', [])[:2]) or 'no'} | "
                    f"{r['seconds']} |")
    return head + "\n" + "\n".join(rows)


AB_ARMS = [("A", "c5_exact_h100"), ("A", "c5_learned_s0_h100"), ("B", "ab_B_exact_h100"), ("C", "ab_C_exact_h100")]
SOLVED = re.compile(r"^\s*([\d.]+)s L(\d+) (\S+)\s+cost (\d+) \((\w+)\)")


def ab() -> str:
    """The A/B (INTEGRATION.md §2): per arm, what was solved, at which level and when (seconds since the climb's start),
    what stalled or was skipped, the round, and where the climb stopped."""
    head = ("| arm | record | cap (stored states) | solved: task (level, by, seconds) | stalled / twins skipped | the round "
            "| stopped | seconds |\n|---|---|---|---|---|---|---|---|")
    rows = []
    for arm, name in AB_ARMS:
        r = load(CLIMB / f"{name}.json")
        if r is None:
            rows.append(f"| {arm} | {name} | | (running or not fetched) | | | | |")
            continue
        solved = []
        for line in r.get("log", []):
            m = SOLVED.match(line)
            if m:
                solved.append(f"{m.group(3)} (L{m.group(2)}, {m.group(5)}, {float(m.group(1)):,.0f} s)")
        cap = r["config"].get("cap", r["config"].get("budget", 3_000_000))
        stall = ", ".join(f"{t} (L{l})" for t, l in sorted((r.get("stalled") or {}).items(), key=lambda kv: kv[1]))
        skip = ", ".join(f"{t} (L{l})" for t, l in sorted((r.get("skipped") or {}).items(), key=lambda kv: kv[1]))
        last = (r.get("log") or [""])[-1].split("s ", 1)[-1][:80]
        rows.append(f"| {arm} | {name} | {cap:,} | {'; '.join(solved)} | {stall or '-'} / {skip or '-'} | "
                    f"{'; '.join(r.get('rounds', [])[:2]) or 'no'} | {r.get('status', last)} | {r['seconds']:,.0f} |")
    return head + "\n" + "\n".join(rows)


SECTIONS = {
    "S1_RUNS": lambda: table_runs(["s1_official"] + [f"s1_cpu_seed{s}" for s in (74, 1, 2, 3, 4, 5, 6, 7)]),
    "ACCEPT": acceptance,
    "CLIMBS": climbs,
    "S3_RUNS": lambda: table_runs(sorted(p.stem for p in REC.glob("s3*.json"))),
    "S3_B200": lambda: table_runs(sorted(f"b200/{p.stem}" for p in (REC / "b200").glob("s3*.json"))),
    "AB": ab,
}


def fill(path: Path) -> None:
    text = path.read_text()
    for name, fn in SECTIONS.items():
        pat = re.compile(rf"(<!-- {name} -->\n)(.*?)(<!-- /{name} -->)", re.S)
        if pat.search(text):
            text = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(3), text)
    path.write_text(text)
    print("filled", path)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fill"]:
        fill(Path(sys.argv[2]))
    else:
        for name, fn in SECTIONS.items():
            print(f"== {name}\n{fn()}\n")
