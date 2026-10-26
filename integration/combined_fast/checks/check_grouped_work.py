"""_grouped_work (FAST.md §2) counts exactly the work _apply_grouped counts: for item lists and rows per item, WORK
after _apply_grouped and after _grouped_work must be the same float, bit for bit. The items: a real vocabulary
(enumerate_items over the primitives and the round-2 tiny core's library, which has loops, C blocks, routines and the
search's ONE bodies), drawn in random lists with random rows per item (some 0), at the climb's settings; plus whole
programs as ONE items (Acceptor.many's). The laptop's CPU, small.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_fast/checks/check_grouped_work.py
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core_c  # noqa: E402
import torch  # noqa: E402

FILE = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "submission_stage3_fast.py")
CASES = int(os.environ.get("CASES", 200))
M, md5 = core_c.load(FILE, [])
rng = random.Random(0)
rec = json.load(open(HERE.parents[1] / "combined_sp2" / "records" / "core_combined2_measure.json"))
lib = {k: M.parse(v) for k, v in rec["lib"].items()}
cells = M.Cells(10, "cpu")
sets = [M.Setting(10, w, cells) for w in (1, 2)]
toks = list(M.PRIMS) + sorted(k for k in lib if k.startswith("T."))[:20]
classes = M.enumerate_items(toks, lib, sets, "cpu", max_body=2)
items = [c.rep for c in classes]
progs = [tuple(rng.choice(items[:300]) for _ in range(rng.randint(1, 4))) for _ in range(200)]
print(f"{len(items)} items ({len(toks)} tokens), {len(progs)} programs", flush=True)
done = 0
for case in range(CASES):
    st = sets[case % 2]
    pick = rng.sample(items, rng.choice([1, 2, 5, 30])) if case % 3 else [("REP", M.ONE, p) for p in
                                                                         rng.sample(progs, rng.choice([1, 4, 16]))]
    cnt = [rng.choice([0, 1, 2, 7]) for _ in pick]
    if not sum(cnt):
        cnt[0] = 1
    ior = torch.tensor([i for i, n in enumerate(cnt) for _ in range(n)], dtype=torch.long)
    b = M.random_states(st, len(ior), 8, torch.Generator().manual_seed(case), "cpu")
    for w0 in (0.0, 1234567.891, 3.1e9):
        M.WORK[0] = w0
        M._apply_grouped(st, b, ior, pick, lib)
        want = M.WORK[0]
        M.WORK[0] = w0
        M._grouped_work(st, pick, cnt, lib, M._traces(st, lib))
        assert M.WORK[0] == want, (case, w0, M.WORK[0], want, pick[:3], cnt)
    done += 1
print(f"_grouped_work: {done} cases x 3 starting values, WORK the same float as _apply_grouped's", flush=True)
