"""The fused lockstep's speed on shapes the climb runs (FAST.md §2), for builds of the fast file side by side (FILES:
paths, comma-separated), and _ustep's eager steps on the small ones: seconds per call, after a first call (compiles).
Each build's result is checked against the first's, byte for byte.

    env: FILES (the builds), REPS (3)
    cd cpu_first && PYTHONPATH=. python integration/combined_fast/checks/speed_fused.py
"""
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core_c  # noqa: E402
import torch  # noqa: E402

FILES = os.environ.get("FILES", str(HERE.parent / "submission_stage3_fast.py")).split(",")
REPS = int(os.environ.get("REPS", 3))
dev = "cuda"
mods = [core_c.load(f, [])[0] for f in FILES]
print(torch.cuda.get_device_name(), torch.__version__, mods[0].triton.__version__, flush=True)
rng = random.Random(0)
SHAPES = ((4, 8, 1 << 23, 1, 30, 0, False), (4, 8, 1 << 22, 16, 64, 1, True), (4, 8, 1 << 17, 2048, 64, 2, False),
          (16, 8, 1 << 21, 100, 64, 2, True), (4, 6, 1 << 18, 2048, 512, 2, True), (8, 60, 256, 4000, 256, 3, True),
          (16, 60, 64, 16000, 64, 4, False))
for P, NP, S, n, items, depth, cond in SHAPES:
    M0 = mods[0]
    st = M0.Setting(10, P // 2, M0.Cells(10, dev), P)
    b = M0.random_states(st, S, NP, torch.Generator(device=dev).manual_seed(5), dev)
    ops = list(range(M0.NOP)) + [M0.CBEG, M0.CEND] * 2 * cond
    ops += [M0.SAVE + 2 * d + e for d in range(depth) for e in (0, 1)]
    cols = torch.tensor([[rng.choice(ops) for _ in range(n)] for _ in range(items)], device=dev).T.contiguous()
    lr = torch.arange(S, device=dev) * items // S
    outs, line = [], f"P={P} NP={NP} rows={S} steps={n} items={items} depth={depth} cond={cond}:"
    for name, M in zip(FILES, mods):
        M.LOCK_FUSED = True
        tt = []
        for _ in range(REPS + 1):
            x = M.Batch(b.regs.clone(), b.c.clone(), b.h.clone(), b.k.clone(), b.p.clone())
            torch.cuda.synchronize()
            t0 = time.time()
            M._lockstep(st, x, cols, lr, depth, cond)
            torch.cuda.synchronize()
            tt.append(time.time() - t0)
        outs.append(x)
        line += f" {Path(name).stem[-12:]} {min(tt[1:]):.4f}s (first {tt[0]:.2f}s)"
    if S * n <= 1 << 24:
        M0.LOCK_FUSED = False
        x = M0.Batch(b.regs.clone(), b.c.clone(), b.h.clone(), b.k.clone(), b.p.clone())
        torch.cuda.synchronize()
        t0 = time.time()
        M0._lockstep(st, x, cols, lr, depth, cond)
        torch.cuda.synchronize()
        line += f" | _ustep {time.time() - t0:.3f}s"
        outs.append(x)
    same = all(all(torch.equal(getattr(o, f), getattr(outs[0], f)) for f in ("regs", "c", "h", "k", "p")) for o in outs)
    print(line, "| same" if same else "| DIFFER", flush=True)
    assert same
print("SPEED OK", flush=True)
