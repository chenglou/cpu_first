"""(COMBINED.md §4) Time each law's Acceptor (the acceptance's probes and targets) on a pattern, at the verify sizes to 3 and to 4 digits,
with the combined file's code, exact cells and g3's official library (its routines for the constructions)."""
import json, os, sys, time, types
sys.path[:0] = ["learned_remainder/chain_local/stub", "integration/combined_sp2", "integration/combined_sp2/checks"]
import torch
import core_c
M, _ = core_c.load("integration/combined_sp2/submission_stage3_combined.py", [])
d = json.load(open("integration/combined_sp2/checks/g3_official_lib.json"))
lib = {k: M.parse(v) for k, v in d["lib"].items()}
ops = M.Ops()
for k in d["lib"]:
    if k.startswith("T.") and k.count(".") >= 3:
        parts = k[2:].split(".")
        ops.add(".".join(parts[:-2]), parts[-2], k, tuple(parts[-1]))
rows = [json.loads(l) for l in open(os.path.join(core_c.H1_DIR, "train.jsonl"))][:4096]
dig = lambda key: core_c.digits(rows, key)
data = (dig("modulus"), dig("x"))
cells = M.exact_cells("cpu")
pat, bd = sys.argv[1], tuple(sys.argv[2])
for W in (3, 4):
    sts = [M.Setting(10, w, cells) for w in range(1, W + 1)]
    tot, out = 0.0, []
    for law, (_, pats) in M.LAW_BOOK.items():
        if pat not in pats:
            continue
        t0, w0 = time.time(), M.WORK[0]
        try:
            acc = M.Acceptor(law, types.SimpleNamespace(pattern=pat, binding=bd), sts, data, ops, lib, "cpu")
            ok = "posed"
        except M.NotPosable as e:
            ok = f"not posable ({e})"
        dt = time.time() - t0
        tot += dt
        out.append(f"{law}: {dt:.1f}s {M.WORK[0] - w0:.3g} units {ok}")
    print(f"{pat}@{''.join(bd)} verify to {W} digits: {tot:.1f}s over {len(out)} laws | " + "; ".join(out), flush=True)
# are the acceptance roots of two laws on the same pattern and binding the same machines?
sts = [M.Setting(10, w, cells) for w in range(1, 5)]
a = M.Acceptor("lt", types.SimpleNamespace(pattern=pat, binding=bd), sts, data, ops, lib, "cpu")
b = M.Acceptor("red", types.SimpleNamespace(pattern=pat, binding=bd), sts, data, ops, lib, "cpu")
same = all(torch.equal(p[1].regs, q[1].regs) and torch.equal(p[1].c, q[1].c) for p, q in zip(a.probs, b.probs))
print(f"lt's and red's acceptance machines on {pat}@{''.join(bd)}: identical inputs and starts {same}", flush=True)
