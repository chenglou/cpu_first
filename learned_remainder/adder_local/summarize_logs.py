"""Per-variant summary of adder_local/logs/*.log: rows right, digit and borrow entries right separately, closest wrong
table, and the hard-check pass rate at the end.  python summarize_logs.py [PREFIX]"""
import collections, glob, re, sys


def final_table(text):
    lines = text.splitlines()
    d = [[[0] * 2 for _ in range(10)] for _ in range(10)]
    b = [[[0] * 2 for _ in range(10)] for _ in range(10)]
    for c in range(2):
        try:
            i = lines.index(f"digit table, borrow in {c} (rows r, columns m) | borrow out")
        except ValueError:
            return None
        for r in range(10):
            left, right = lines[i + 1 + r].split("|")
            for m, (dv, bv) in enumerate(zip(left.split(), right.split())):
                d[r][m][c], b[r][m][c] = int(dv), int(bv)
    return d, b


def main(prefix):
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(f"logs/{prefix}*.log")):
        text = open(f).read()
        var = re.sub(r"_seed\d+$", "", f.split("/")[-1][:-4])
        steps = [l for l in text.splitlines() if l.startswith("step")]
        if not steps:
            continue
        m = re.search(r"win ([\d.]+) true (\d+)/200(?: adder_true (\d+)/200)? closest_wrong (\S+) (\d+)/200", steps[-1])
        t = final_table(text)
        dig = bor = None
        if t:
            d, b = t
            dig = sum(d[r][mm][c] == (r - mm - c) % 10 for r in range(10) for mm in range(10) for c in range(2))
            bor = sum(b[r][mm][c] == int(r - mm - c < 0) for r in range(10) for mm in range(10) for c in range(2))
        rows[var].append(dict(true=m.group(2), win=m.group(1), dig=dig, bor=bor, close=f"{m.group(4)} {m.group(5)}",
                              adder=m.group(3)))
    for var, rs in rows.items():
        j = lambda k: ", ".join(str(r[k]) for r in rs)
        print(f"| {var} | {len(rs)} | {j('true')} | {j('dig')} | {j('bor')} | {j('win')} | {j('close')} |"
              + (f" adder {j('adder')}" if rs[0]["adder"] else ""))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
