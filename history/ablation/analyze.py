"""Tabulate ablation records (runs/<name>.json; also ../runs/learned_remainder/<name>.json): certification, the
schedule, and the snapped table against true subtraction and the consistent wrong tables of
learned_remainder/adder_local/landscape.py (the comparison is done here, outside the submission).

    python analyze.py runs                         # one line per record, every record in runs/
    python analyze.py variants PREFIX [...]        # one markdown row per variant: names PREFIX_<variant>_s<seed>
    python analyze.py traj NAME                    # rows right along the run (the LR table lines)
    python analyze.py cell NAME ...                # one table cell: certified k/n, freeze steps, failures
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRS = [HERE / "runs", HERE.parent / "runs" / "learned_remainder"]


def rows():
    return [(r, m, c) for r in range(10) for m in range(10) for c in range(2)]


def table(fd, fb):
    return [fd(r, m, c) % 10 for r, m, c in rows()], [int(fb(r, m, c)) for r, m, c in rows()]


def candidates() -> dict:
    crt = lambda a2, a5: (5 * a2 + 6 * a5) % 10
    out = {
        "true": table(lambda r, m, c: r - m - c, lambda r, m, c: r - m - c < 0),
        "borrow-free": table(lambda r, m, c: r - m - c, lambda r, m, c: 0),
        "copy the minuend": table(lambda r, m, c: r, lambda r, m, c: 0),
        "swapped borrow code": table(lambda r, m, c: r - m - (1 - c), lambda r, m, c: 1 - (r - m - (1 - c) < 0)),
        "reversed": table(lambda r, m, c: m - r - c, lambda r, m, c: m - r - c < 0),
        "digits relabeled": table(lambda r, m, c: r - m + c, lambda r, m, c: (-r) % 10 - (-m) % 10 - c < 0),
        "mod-5 borrows": table(lambda r, m, c: crt((r - m) % 2, (r % 5 - m % 5 - c) % 5),
                               lambda r, m, c: r % 5 - m % 5 - c < 0),
        "mod-2 borrows": table(lambda r, m, c: crt((r % 2 - m % 2 - c) % 2, (r - m) % 5),
                               lambda r, m, c: r % 2 - m % 2 - c < 0),
        "lazy": table(lambda r, m, c: 0 if (r == m and c == 0) else r,
                      lambda r, m, c: 1 if c == 1 else int(m != 0 and r != m)),
    }
    return out


CAND = candidates()
TD, TB = CAND["true"]


def find(name: str) -> Path | None:
    for d in DIRS:
        p = d / f"{name}.json"
        if p.exists():
            return p
    return None


def family(wrong: list) -> str:
    """A short name for the set of wrong rows."""
    if not wrong:
        return ""
    keys = {(r, m, c) for r, m, c, _, _ in wrong}
    fams = []
    r01 = {k for k in keys if k[1] == 0 and k[2] == 1}
    rr1 = {k for k in keys if k[0] == k[1] and k[2] == 1}
    if r01:
        fams.append(f"(r,0,1)x{len(r01)}")
    if rr1 - r01:
        fams.append(f"(r,r,1)x{len(rr1 - r01)}")
    rest = keys - r01 - rr1
    if rest:
        fams.append(f"other x{len(rest)}")
    return " ".join(fams)


def parse(rec: dict) -> dict:
    out = rec.get("stdout", "")
    lines = out.splitlines()
    r = rec.get("result") or {}
    dp = r.get("depth_profile") or {}
    seed = (r.get("seeds") or [{}])[0]
    info = {"name": rec["name"], "rc": rec["returncode"], "maxT": dp.get("max_certified_time_steps"),
            "oodT": dp.get("ood_n_max_certified_time_steps"), "train_s": seed.get("training_seconds"),
            "eval_s": seed.get("evaluation_seconds"), "steps": seed.get("completed_training_steps"),
            "seed": seed.get("seed")}
    info["certified"] = info["maxT"] == 64 and info["oodT"] == 64
    m = re.search(r"^LR switch_to_own_borrow step=(\d+)", out, re.M)
    info["switch"] = int(m.group(1)) if m else None
    m = re.search(r"^LR frozen step=(\d+)", out, re.M)
    info["frozen"] = int(m.group(1)) if m else None
    dig = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE digit=")), None)
    bor = next((l.split("=", 1)[1] for l in lines if l.startswith("TABLE borrow=")), None)
    if dig and bor:
        d, b = [int(ch) for ch in dig], [int(ch) for ch in bor]
        info.update(stats(d, b))
    hx = [float(x) for x in re.findall(r"^LR step=\d+ phase=\S+ in_phase=\d+ loss=\S+ hard_exact=([\d.]+)", out, re.M)]
    info["last_exact"] = hx[-1] if hx else None
    traj = []
    for l in lines:
        mm = re.match(r"LR table step=(\d+) phase=(\S+) exact=([\d.]+) digit=(\d+) borrow=(\d+)", l)
        if mm:
            st = stats([int(ch) for ch in mm.group(4)], [int(ch) for ch in mm.group(5)])
            traj.append((int(mm.group(1)), mm.group(2), float(mm.group(3)), st["right"], st["closest"],
                         st["closest_n"], st["wrong"]))
    info["traj"] = traj
    ev = seed.get("evaluation", {})
    info["test"] = ev.get("test", {}).get("exact_accuracy")
    return info


def stats(d: list, b: list) -> dict:
    right = sum(d[i] == TD[i] and b[i] == TB[i] for i in range(200))
    digits = sum(d[i] == TD[i] for i in range(200))
    borrows = sum(b[i] == TB[i] for i in range(200))
    match = {k: sum(d[i] == cd[i] and b[i] == cb[i] for i in range(200)) for k, (cd, cb) in CAND.items() if k != "true"}
    closest = max(match, key=match.get)
    wrong = [(r, m, c, d[i], b[i]) for i, (r, m, c) in enumerate(rows()) if d[i] != TD[i] or b[i] != TB[i]]
    return {"right": right, "digits": digits, "borrows": borrows, "closest": closest, "closest_n": match[closest],
            "wrong": wrong}


def line(i: dict) -> str:
    w = i.get("wrong", [])
    return (f"{i['name']}: certified={i['certified']} maxT={i['maxT']}/{i['oodT']} switch={i['switch']} "
            f"frozen={i['frozen']} rows={i.get('right')} digits={i.get('digits')} borrows={i.get('borrows')} "
            f"closest={i.get('closest')} {i.get('closest_n')} exact={i['last_exact']} test={i['test']} "
            f"train_s={i['train_s'] and round(i['train_s'])} wrong={family(w)} {w[:8] if len(w) <= 8 else ''}")


def variants(prefixes: list) -> None:
    groups = defaultdict(list)
    for d in DIRS[:1]:
        for p in sorted(d.glob("*.json")):
            name = p.stem
            for pre in prefixes:
                m = re.match(rf"^{pre}_(.+)_s(\d+)$", name)
                if m:
                    groups[(pre, m.group(1))].append(parse(json.loads(p.read_text())))
    print("| variant | seeds | certified | switch | frozen at | rows right | failure: closest wrong table, wrong rows |")
    print("|---|---|---|---|---|---|---|")
    for (pre, v), infos in sorted(groups.items()):
        infos.sort(key=lambda i: [74, 1, 2, 3, 4, 5, 6, 7].index(i["seed"]) if i["seed"] in (74, 1, 2, 3, 4, 5, 6, 7) else 99)
        cert = sum(i["certified"] for i in infos)
        fails = [f"s{i['seed']}: {i.get('right')}/200, {i.get('closest')} {i.get('closest_n')}, {family(i.get('wrong', []))}"
                 for i in infos if not i["certified"] or i.get("right") != 200]
        print(f"| {v} | {', '.join(str(i['seed']) for i in infos)} | {cert}/{len(infos)} | "
              f"{', '.join(str(i['switch']) for i in infos)} | {', '.join(str(i['frozen'] or '–') for i in infos)} | "
              f"{', '.join(str(i.get('right')) for i in infos)} | {'; '.join(fails)} |")


def summary(names: list) -> str:
    """One cell: certified k/n, the freeze steps (or the end state of the failures)."""
    infos = []
    for n in names:
        try:
            infos.append(parse(json.loads(find(n).read_text())))
        except (AttributeError, json.JSONDecodeError):          # not fetched yet, or being written by fetch.py
            pass
    if not infos:
        return "pending"
    cert = sum(i["certified"] for i in infos)
    fr = [i["frozen"] for i in infos if i["frozen"]]
    out = f"{cert}/{len(infos)}"
    if fr:
        out += f", frozen {min(fr)}–{max(fr)}" if len(fr) > 1 else f", frozen {fr[0]}"
    elif cert:
        out += ", none froze"
    fails = [i for i in infos if not i["certified"]]
    if fails:
        out += "; failed: " + ", ".join(f"{i.get('right')}/200 ({i.get('closest')} {i.get('closest_n')})"
                                        for i in fails)
    return out


def main(argv):
    if argv[0] == "cell":
        print(summary(argv[1:]))
        return
    if argv[0] == "runs":
        for p in sorted((HERE / "runs").glob("*.json")):
            print(line(parse(json.loads(p.read_text()))))
    elif argv[0] == "variants":
        variants(argv[1:])
    elif argv[0] == "traj":
        for name in argv[1:]:
            i = parse(json.loads(find(name).read_text()))
            print(line(i))
            for st, ph, ex, right, cl, cn, wrong in i["traj"]:
                print(f"  step {st} {ph} exact {ex:.3f} rows {right} closest {cl} {cn} {family(wrong)}"
                      + (f" {wrong}" if len(wrong) <= 6 else ""))
    else:
        for name in argv:
            print(line(parse(json.loads(find(name).read_text()))))


if __name__ == "__main__":
    main(sys.argv[1:])
