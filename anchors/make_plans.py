"""Write the lab plans (plans/<name>.json) for modal_lab.py.

    python make_plans.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLANS = HERE / "plans"
PLANS.mkdir(exist_ok=True)

FACTS = {"less": 1, "more": 1}
ID = {"zero": 1, "self": 1}


def tiny(name, B, W, w, seeds, steps=400, **kw):
    return [{"name": f"{name}_s{s}", "kind": "tiny", "B": B, "W": W, "steps": steps, "seed": s, "w": w, "kw": kw}
            for s in seeds]


def core(name, w, seeds, steps=600, batch=64, **kw):
    return [{"name": f"{name}_s{s}", "kind": "core", "B": 10, "W": 8, "steps": steps, "seed": s, "w": w,
             "kw": dict(batch=batch, **kw)} for s in seeds]


plans = {}

# p1: the order lesson at every width in base 10 (every input of a 2-digit machine, and the real H1 machine)
p = []
p += tiny("t10w2_monk_id_tb", 10, 2, {"monk": 1, **ID}, range(4), tb_fixed=True, mon_k=2)
p += tiny("t10w2_mon_id_tb", 10, 2, {"mon": 1, **ID}, range(2), tb_fixed=True, mon_k=2)
p += tiny("t10w2_gate_monk_id", 10, 2, {"monk": 1, **ID, **FACTS}, range(4), gate=1.0, mon_k=2)
p += tiny("t6w3_monk_id_tb", 6, 3, {"monk": 1, **ID}, range(4), tb_fixed=True, mon_k=2)
p += core("c_gate_monk_id_b64", {"monk": 1, **ID, **FACTS}, range(1, 4), gate=1.0)
p += core("c_gate_monk_id_b512", {"monk": 1, **ID, **FACTS}, range(1, 3), batch=512, gate=1.0)
plans["p1"] = p

# p2: the recipe found locally (borrows from x < N on every top part, then the one-digit order law + column
# bijectivity, then the learned check + wrap 100 steps after the gate), seeds and ablations on the H1 machine
R1 = {"pref": 1, "col": 1, "bij": 1}
LATE = {"late": {"lc": 1, "wrap": 1}, "late_at": 100}
p = []
p += core("r1_b64", R1, range(1, 13), gate=1.0, **LATE)
p += core("r1_b512", R1, range(1, 5), steps=400, batch=512, gate=1.0, **LATE)
abl = {
    "noBij": ({"pref": 1, "col": 1}, LATE),
    "noCol": ({"pref": 1, "bij": 1}, LATE),
    "noLate": (R1, {}),
    "lateLcOnly": (R1, {"late": {"lc": 1}, "late_at": 100}),
    "lateWrapOnly": (R1, {"late": {"wrap": 1}, "late_at": 100}),
    "factsNotPref": ({"less": 1, "more": 1, "col": 1, "bij": 1}, LATE),
    "plusId": ({**R1, "zero": 1, "self": 1}, LATE),
    "lateAtGate": (R1, {"late": {"lc": 1, "wrap": 1}, "late_at": 0}),
}
for name, (w, late) in abl.items():
    p += core(f"abl_{name}", w, range(1, 5), gate=1.0, **late)
plans["p2"] = p

# p3: order + bijectivity on both arguments of the cell (columns: r -> r - m - c; rows: m -> r - m - c)
R2 = {"pref": 1, "colall": 1, "rowall": 1, "bij": 1, "bijr": 1}
p = []
p += core("r2_b64", R2, range(1, 13), steps=400, gate=1.0)
p += core("r2_b512", R2, range(1, 5), steps=400, batch=512, gate=1.0)
p += core("r2late_b64", R2, range(1, 13), steps=400, gate=1.0, **LATE)
abl2 = {
    "noRow": {"pref": 1, "colall": 1, "bij": 1, "bijr": 1},
    "noBijr": {"pref": 1, "colall": 1, "rowall": 1, "bij": 1},
    "noColall": {"pref": 1, "rowall": 1, "bij": 1, "bijr": 1},
    "noBijc": {"pref": 1, "colall": 1, "rowall": 1, "bijr": 1},
    "orderOnly": {"pref": 1, "colall": 1, "rowall": 1},
    "bijOnly": {"pref": 1, "bij": 1, "bijr": 1},
    "facts": {"less": 1, "more": 1, "colall": 1, "rowall": 1, "bij": 1, "bijr": 1},
}
for name, w in abl2.items():
    p += core(f"abl2_{name}", w, range(1, 7), steps=400, gate=1.0)
p += core("abl2_noGate", R2, range(1, 7), steps=400)
for B_, wmin, wmax in ((3, 1, 3), (4, 1, 3), (6, 2, 3)):
    p += [{**j, "B": B_, "W": wmax, "tiny_data": [B_, wmin, wmax, 4000]}
          for j in core(f"r2_tiny_b{B_}", R2, range(1, 5), steps=400, gate=1.0)]
plans["p3"] = p

# x1: the exact map on tiny machines (every input): each signal alone and the combinations, borrows free and held true
SINGLE = ["lc", "lcc", "dc", "less", "more", "less,more", "xmod", "zero", "self", "zero,self", "neg2", "comm", "wrap",
          "chain", "dec", "mon", "pref", "colord", "roword", "bijc", "ans", "ans2", "nn0"]
COMBOS = ["lcc,zero,self,wrap", "lcc,zero,self,wrap,neg2,comm", "less,more,lcc,zero,self,wrap,neg2,comm",
          "colord,roword", "colord,bijc", "pref,colord", "pref,colord,roword", "pref,colord,roword,bijc",
          "less,more,colord,roword"]
p = []
for B_, W_ in ((3, 3), (4, 2)):
    for ls in SINGLE + COMBOS:
        for bt in (False, True):
            if bt and all(l in ("less", "more", "neg", "pref") for l in ls.split(",")):
                continue                                                 # borrow-only lessons: borrows free only
            p.append({"name": f"x_b{B_}w{W_}_{ls.replace(',', '+')}_{'bt' if bt else 'free'}", "kind": "exact",
                      "B": B_, "W": W_, "lessons": ls.split(","), "borrows_true": bt, "limit": 400})
plans["x1"] = p

# p4: the smallest recipe (top-part facts + both order laws, no bijectivity, no gate), and the answers as the only
# digit teacher once the borrows are right (how much answer signal does it take?)
MIN = {"pref": 1, "colall": 1, "rowall": 1}
p = []
p += core("min_b64", MIN, range(1, 13), steps=400)
p += core("min_b512", MIN, range(1, 5), steps=400, batch=512)
for share in (0.02, 0.1, 0.3, 1.0):
    js = core(f"ans{int(share * 100)}_b64", {"pref": 1, "ans": 1}, range(1, 5), steps=600, gate=1.0)
    p += [{**j, "answer_share": share, "rows": 2000 if share == 1.0 else 4096} for j in js]
plans["p4"] = p

# t1: the tiny-core censuses (every input, full batch, exact gradients): the failure and what fixes it
CORE = {"lc": 1, "zero": 1, "self": 1, "less": 1, "more": 1, "wrap": 1, "neg": 1, "neg2": 1, "comm": 1}
WARM = {"dc": 1, "zero": 1, "self": 1, "less": 1, "more": 1, "wrap": 1}
drop = lambda w, *ks: {k: v for k, v in w.items() if k not in ks}
TB = {"tb_fixed": True}
census = [
    ("t3_core", 3, 3, CORE, TB), ("t3_warm", 3, 3, WARM, TB), ("t3_core_noanchors", 3, 3, drop(CORE, "zero", "self"), TB),
    ("t3_core_hardcin", 3, 3, CORE, {**TB, "hard_cin": True}), ("t3_core_dec", 3, 3, {**CORE, "dec": 1}, TB),
    ("t3_core_chain", 3, 3, {**CORE, "chain": 1}, TB), ("t3_core_nn0", 3, 3, {**CORE, "nn0": 1}, TB),
    ("t3_core_ans100", 3, 3, {**CORE, "ans": 1}, TB), ("t3_core_ans10", 3, 3, {**CORE, "ans": 1}, {**TB, "ans_frac": 0.1}),
    ("t3_mon", 3, 3, {"mon": 1}, TB), ("t3_mon_id", 3, 3, {"mon": 1, "zero": 1, "self": 1}, TB),
    ("t3_mon_uncoupled", 3, 3, {"mon": 1}, {**TB, "coupled": False}), ("t3_core_mon", 3, 3, {**CORE, "mon": 1}, TB),
    ("t4_mon", 4, 2, {"mon": 1}, TB), ("t4_mon_zero", 4, 2, {"mon": 1, "zero": 1}, TB),
    ("t4_mon_id", 4, 2, {"mon": 1, "zero": 1, "self": 1}, TB),
    ("t5_mon_id", 5, 2, {"mon": 1, "zero": 1, "self": 1}, TB), ("t5_monk_id", 5, 2, {"monk": 1, "zero": 1, "self": 1}, TB),
    ("t6_mon_id", 6, 2, {"mon": 1, "zero": 1, "self": 1}, TB), ("t6_monk_id", 6, 2, {"monk": 1, "zero": 1, "self": 1}, TB),
]
p = []
for name, B_, W_, w, kw in census:
    p += tiny(name, B_, W_, w, range(8), steps=400, **kw)
plans["t1"] = p

for name, jobs in plans.items():
    (PLANS / f"{name}.json").write_text(json.dumps(jobs, indent=1))
    print(name, len(jobs))
