"""The fused lockstep (FAST.md §2) against _ustep's steps, on a GPU: the state after every lockstep must be the same
byte for byte (registers, flags, pointer), with LOCK_FUSED on and off.
  random  random sequences of every instruction (the primitives, NOP, routine saves and restores at every depth, C
          blocks) on random machines, at every register size and probe count the climb uses, short and long (past
          GRAPH_MIN: the graph path), many rows and few;
  vocab   apply_items over a real vocabulary: enumerate_items' items (primitives, then loops and C blocks of bodies
          up to cost 2) with the round-2 tiny core's final library of routines (records/core_combined2_measure.json),
          on random machines at the climb's settings, its verify sizes and the round's size;
  speed   the two on a few large shapes (seconds only).

    env: FILE (the fast file), CASES (random cases, 300), SEED (0), VOCAB (the sizes: W,NP,machines;...)
    cd cpu_first && PYTHONPATH=. python integration/combined_fast/checks/check_fused.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import core_c  # noqa: E402
import torch  # noqa: E402

FILE = os.environ.get("FILE", str(HERE.parent / "submission_stage3_fast.py"))
M, md5 = core_c.load(FILE, [])
dev = "cuda"
assert M.triton is not None, "no Triton"
rng = random.Random(int(os.environ.get("SEED", 0)))
print("file", FILE, md5, "triton", M.triton.__version__, torch.__version__, torch.cuda.get_device_name(), flush=True)


def state(S, P, NP, gen):
    return M.random_states(M.Setting(10, P // 2, M.Cells(10, dev), P), S, NP, gen, dev)


def run(fused, st, b, cols, lr, depth, cond):
    M.LOCK_FUSED = fused
    x = b.copy()
    M._lockstep(st, x, cols, lr, depth, cond)
    torch.cuda.synchronize()
    return x


def same(x, y):
    return all(torch.equal(getattr(x, f), getattr(y, f)) for f in ("regs", "c", "h", "k", "p"))


def random_case(i):
    P = rng.choice([2, 4, 6, 8, 16])
    NP = rng.choice([1, 2, 6, 8, 60])
    S = rng.choice([1, 3, 17, 300, 4000])
    n = rng.choice([1, 2, 5, 40, 200, 700])
    items = rng.choice([1, 3, 20])
    depth = rng.choice([0, 0, 1, 3])
    cond = rng.random() < 0.5
    ops = list(range(M.NOP + 1)) * 4 + [M.SAVE + 2 * d + e for d in range(depth) for e in (0, 1)] * 2
    ops += [M.CBEG, M.CEND] * (6 if cond else 0)
    seqs = [[rng.choice(ops) for _ in range(n)] for _ in range(items)]
    cols = torch.tensor(seqs, dtype=torch.long, device=dev).T.contiguous()
    lr = torch.tensor([rng.randrange(items) for _ in range(S)], dtype=torch.long, device=dev)
    st = M.Setting(10, P // 2, M.Cells(10, dev), P)
    b = state(S, P, NP, torch.Generator(device=dev).manual_seed(i))
    b.p = torch.randint(0, P, (S,), device=dev)
    a, f = run(False, st, b, cols, lr, depth, cond), run(True, st, b, cols, lr, depth, cond)
    assert same(a, f), dict(case=i, P=P, NP=NP, S=S, n=n, items=items, depth=depth, cond=cond)
    return dict(P=P, NP=NP, S=S, n=n)


t0 = time.time()
cases = int(os.environ.get("CASES", 300))
seen = [random_case(i) for i in range(cases)]
print(f"random: {cases} cases the same (P {sorted({c['P'] for c in seen})}, NP {sorted({c['NP'] for c in seen})}, "
      f"rows up to {max(c['S'] for c in seen)}, steps up to {max(c['n'] for c in seen)}), {time.time() - t0:.0f} s",
      flush=True)

# a real vocabulary, with the tiny core's final library
rec = json.load(open(HERE.parents[1] / "combined_sp2" / "records" / "core_combined2_measure.json"))
lib = {k: M.parse(v) for k, v in rec["lib"].items()}
cells = M.Cells(10, dev)
sets = [M.Setting(10, w, cells) for w in (1, 2)]
toks = list(M.PRIMS) + sorted(k for k in lib if k.startswith("T."))
t0 = time.time()
M.LOCK_FUSED = True
classes = M.enumerate_items(toks, lib, sets, dev, max_body=2)
items = [c.rep for c in classes]
print(f"vocab: {len(items)} items ({len(toks)} tokens), built fused in {time.time() - t0:.0f} s", flush=True)
VOCAB = os.environ.get("VOCAB", "1,8,2;2,8,2;3,60,1;4,60,1;8,60,1,2000;2,6,24")  # W, NP, machines per item[, items]
for w, NP, S, *most in (tuple(int(x) for x in v.split(",")) for v in VOCAB.split(";")):
    st = M.Setting(10, w, cells)
    gen = torch.Generator(device=dev).manual_seed(w * 100 + NP)
    upto = rng.sample(items, most[0]) if most else items
    for i0 in range(0, len(upto), 4000):
        part = upto[i0:i0 + 4000]
        b = M.random_states(st, S * len(part), NP, gen, dev)
        ior = torch.arange(len(part), device=dev).repeat_interleave(S)
        outs = []
        for fused in (False, True):
            M.LOCK_FUSED = fused
            outs.append(M.apply_items(st, b, ior, part, lib))
        torch.cuda.synchronize()
        assert same(*outs), (w, NP, i0)
    print(f"vocab: apply_items the same at W={w}, NP={NP} ({len(upto)} items x {S} machines)", flush=True)

for P, NP, S, n, items_, depth in ((4, 8, 1 << 22, 1, 30, 0), (4, 8, 1 << 17, 2048, 64, 2), (8, 60, 256, 4000, 256, 3),
                                   (16, 60, 64, 16000, 64, 4)):
    st = M.Setting(10, P // 2, cells, P)
    b = state(S, P, NP, torch.Generator(device=dev).manual_seed(5))
    ops = list(range(M.NOP))
    cols = torch.tensor([[rng.choice(ops) for _ in range(n)] for _ in range(items_)], device=dev).T.contiguous()
    lr = torch.arange(S, device=dev) * items_ // S
    tt = []
    for fused in (True, False, True):
        torch.cuda.synchronize()
        t0 = time.time()
        run(fused, st, b, cols, lr, depth, False)
        tt.append(time.time() - t0)
    print(f"speed: P={P} NP={NP} rows={S} steps={n}: _ustep {tt[1]:.3f} s, fused {tt[2]:.4f} s (first {tt[0]:.2f} s)",
          flush=True)
print("FUSED OK", flush=True)
