"""ADDER.md §1.2: one markdown row per variant from adder_local/logs (python table_rows.py)."""
import summarize_logs as sl
import collections, glob, re

DESC = [
    ("s1_v0_base_b64", "A", "64", "the plan's lessons: check, R − 0 = R, R − R = 0, 0 − N borrows; phase B: x mod N = x, answers"),
    ("s1_v1_neg2_b64", "A", "64", "+ 0 + N = N"),
    ("s1_v2_comm_b64", "A", "64", "+ R + N = N + R"),
    ("s1_v3_neg2comm_b64", "A", "64", "+ both"),
    ("s1_v4_order_b64", "A", "64", "the plan's lessons + x − N borrows, N − x does not"),
    ("s1_v5_all_b64", "A", "64", "everything above"),
    ("s1_v6_all_selfA_b64", "A", "64", "everything, x mod N = x from step 1"),
    ("s1_v7_all_noZS_b64", "A", "64", "everything but R − 0 = R and R − R = 0"),
    ("s1_v8_designed_b64", "designed", "64", "CHAIN.md's recipe (hand-written adder), for comparison"),
    ("s2_a7_all_b512", "A", "512", "everything"),
    ("s2_a1_all_ans1_b512", "A", "512", "everything, answers from step 1"),
    ("s2_a2_all_ans1soft_b512", "A", "512", "everything, answers from step 1 through the soft own-borrow choice (T = 1)"),
    ("s2_a3_all_ans1_fixb_b512", "A", "512", "as a1, borrow table held true (diagnostic)"),
    ("s2_a4_all_ans1soft_fixb_b512", "A", "512", "as a2, borrow table held true (diagnostic)"),
    ("s2_a5_ansonly_b512", "A", "512", "no check: R − 0 = R, R − R = 0, order facts, answers from step 1"),
    ("s2_a6_ansonly_soft_b512", "A", "512", "as a5, answers through the soft choice (T = 1)"),
    ("s2_b1_table_b512", "B", "512", "check through the adder table, (R + N) − N = R, 0 + N = N, R + N = N + R, R − 0 = R, R − R = 0"),
    ("s2_b2_table_order_b512", "B", "512", "+ order facts"),
    ("s2_b3_table_order_ans1_b512", "B", "512", "+ order facts, answers from step 1"),
    ("s2_d1_designed_b512", "designed", "512", "CHAIN.md's recipe, for comparison"),
    ("s3_f_final_b512", "A", "512", "everything + (x − N) − N no borrow, from a random start (1,000 steps)"),
    ("s3_w70_warm_b512", "A", "512", "the same, hand-written adder for the first 70 steps: the final file"),
    ("s3_w100_warm_b512", "A", "512", "the same, hand-written adder for the first 100 steps"),
    ("s4_w10_b512", "A", "512", "hand-written adder for the first 10 steps"),
    ("s4_w30_b512", "A", "512", "hand-written adder for the first 30 steps"),
    ("s4_w50_b512", "A", "512", "hand-written adder for the first 50 steps"),
    ("s4_w70_idonly_b512", "A", "512", "70 warm steps, never sees an answer (REAL_MAX_T = 0)"),
    ("s4_w70_nowrap_b512", "A", "512", "70 warm steps, without (x − N) − N no borrow"),
]


def main():
    data = collections.defaultdict(list)
    for f in sorted(glob.glob("logs/s*.log")):
        text = open(f).read()
        var = re.sub(r"_seed\d+$", "", f.split("/")[-1][:-4])
        seed = re.search(r"_seed(\d+)$", f[:-4]).group(1)
        steps = [l for l in text.splitlines() if l.startswith("step")]
        if not steps:
            continue
        m = re.search(r"^step (\d+) .*win ([\d.]+) true (\d+)/200(?: adder_true (\d+)/200)? closest_wrong (\S+) (\d+)/200",
                      steps[-1])
        t = sl.final_table(text)
        d, b = t
        dig = sum(d[r][mm][c] == (r - mm - c) % 10 for r in range(10) for mm in range(10) for c in range(2))
        bor = sum(b[r][mm][c] == int(r - mm - c < 0) for r in range(10) for mm in range(10) for c in range(2))
        frozen = re.search(r"LR frozen step=(\d+)", text)
        data[var].append(dict(seed=seed, true=m.group(3), dig=dig, bor=bor, win=float(m.group(2)),
                              adder=m.group(4), close=m.group(5).replace("_", " "), cn=m.group(6),
                              frozen=frozen.group(1) if frozen else "–"))
    print("| route | batch | lessons | seeds | rows right | digits right | borrows right | checks passed | frozen at | closest wrong table (entries matching) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, route, batch, desc in DESC:
        rs = data.get(key)
        if not rs:
            print(f"| {route} | {batch} | {desc} | pending | | | | | | |")
            continue
        j = lambda k: ", ".join(str(r[k]) for r in rs)
        rows = j("true") + (f" (adder table: {j('adder')})" if rs[0]["adder"] else "")
        print(f"| {route} | {batch} | {desc} | {j('seed')} | {rows} | {j('dig')} | {j('bor')} | "
              f"{', '.join(f'{r[chr(119)+chr(105)+chr(110)]:.0%}' for r in rs)} | {j('frozen')} | "
              f"{'; '.join('–' if r['frozen'] != '–' else r['close'] + ' ' + r['cn'] for r in rs)} |")


if __name__ == "__main__":
    main()
