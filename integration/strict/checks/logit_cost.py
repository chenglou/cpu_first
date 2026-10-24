"""What would real logits in training cost (REVIEW.md §5, rule 8(a); STRICT.md §5)? The strict file's prediction path
(predict_ste: the reader, the countdown running the round programs on one-hot digits with straight-through tables,
the writer) inside autograd on training rows, then a cross-entropy on the final logits and its backward: seconds and
peak GPU memory, for a few row counts and T values. The reader and the cells at the truth (MODE "designed") with the
round programs a certified run found. GPU (Modal: modal_smoke.py --gpu).

    python integration/strict/checks/logit_cost.py [KIND=sp2] [CASES=48x1,48x1a,512x1,64x16a,512x16a] [LIB=...]
    (a case is ROWSxT, with "a" when only the tournament's argmax program runs, as without autograd)
"""
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
STRICT = os.path.dirname(HERE)
INTEG = os.path.dirname(STRICT)
ROOT = os.path.dirname(INTEG)
sys.path[:0] = [os.path.join(ROOT, "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
KIND = args.get("KIND", "sp2")
CASES = args.get("CASES", "48x1,48x1a,512x1a,64x16a,512x16a").split(",")
LIB = os.path.join(ROOT, args.get("LIB", f"integration/strict/libs/{KIND}_official_lib.json"))
DATA = os.environ.get("H1_DATA", os.path.join(os.path.dirname(ROOT), "one-layer-deeper", "data", "generated",
                                              "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = os.path.join(STRICT, f"submission_stage3_{KIND}_strict.py")
src = open(path).read().replace('MODE = "learned"', 'MODE = "designed"', 1)
sc = types.ModuleType("strict")
sys.modules["strict"] = sc
exec(compile(src, path, "exec"), sc.__dict__)
from benchmark import ModelSpec  # noqa: E402
cfg = json.load(open(os.path.join(DATA, "dataset_config.json")))
torch.manual_seed(0)
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).to(dev).eval()
found = json.load(open(LIB))
m.set_round({k: sc.parse(v) for k, v in found["lib"].items()}, [sc.parse(p) for p in found["round"]])
split = {1: "depth_t_1", 2: "depth_t_2", 4: "depth_t_4", 8: "depth_t_8", 16: "depth_t_16"}
cache = {}


def batch(n, T):
    if (n, T) not in cache:
        rows = []
        with open(os.path.join(DATA, split[T] + ".jsonl")) as f:
            for line in f:
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
        while len(rows) < n:
            rows = rows + rows
        rows = rows[:n]
        L, Tl = max(len(r["input_ids"]) for r in rows), max(len(r["labels"]) for r in rows)
        ids = torch.zeros(n, L, dtype=torch.long)
        lab = torch.full((n, Tl), -100, dtype=torch.long)
        tpos = torch.full((n, Tl), -1, dtype=torch.long)
        for i, r in enumerate(rows):
            a, b = len(r["input_ids"]), len(r["labels"])
            ids[i, :a] = torch.tensor(r["input_ids"])
            lab[i, :b] = torch.tensor(r["labels"])
            tpos[i, :b] = torch.arange(a - b, a)
        cache[(n, T)] = (ids.to(dev), (ids != 0).to(dev), lab.to(dev), tpos.to(dev))
    return cache[(n, T)]


argmax_only = [False]
orig_round = sc.LibraryMachine.round_ste


def round_ste(self, s, Nr, SBB, ADC, MUL, ops):
    if not argmax_only[0]:
        return orig_round(self, s, Nr, SBB, ADC, MUL, ops)
    n = len(ops)
    sel = sc.ste(F.one_hot(self.prog[:n].argmax(), n).float(), self.prog[:n].float().softmax(-1))
    i = int(sel.argmax())
    X = s.shape[0]
    zero = torch.zeros(X, sc.PR, 10, device=s.device)
    zero[..., 0] = 1
    regs = [zero, sc.zpad(s, 0, sc.PR - sc.W), zero, sc.zpad(Nr, 0, sc.PR - sc.W), zero]
    z2 = torch.zeros(X, 2, device=s.device)
    z2[:, 0] = 1
    z10 = torch.zeros(X, 10, device=s.device)
    z10[:, 0] = 1
    mach = sc.SteMachine(SBB, ADC, MUL, sc.PR, sc.damp)
    return sel[i] * mach.run_ops(regs, z2, z10, z10, ops[i])[0][0][:, :sc.W]


sc.LibraryMachine.round_ste = round_ste
print(f"device {dev}, {torch.cuda.get_device_name() if dev.type == 'cuda' else ''}; programs {m.n_progs()}, "
      f"instructions {[int(x) for x in m.round_len.tolist()]}", flush=True)
for case in CASES:
    argmax_only[0] = case.endswith("a")
    n, T = (int(x) for x in case.rstrip("a").split("x"))
    ids, am, lab, tpos = batch(n, T)
    for rep in range(2):                         # the second run is the one timed (the first warms up)
        m.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.time()
            e, end = m.ext(ids, am)
            k = m.selected(e, end)
            logits = m.predict_ste(e, end, k, ids.shape[1])
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            tl = logits[torch.arange(n, device=dev)[:, None], tpos.clamp_min(0)]
            valid = lab != -100
            loss = F.cross_entropy(tl[valid].float(), lab[valid])
            loss.backward()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.time()
            exact = float(((tl.argmax(-1) == lab) | ~valid).all(1).float().mean())
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
            got = sum(int(p.grad is not None and bool((p.grad != 0).any())) for p in m.parameters())
            if rep:
                print(f"{case}: rows {n}, T {T}, {'argmax program' if argmax_only[0] else 'every program'}: forward "
                      f"{t1 - t0:.1f}s, backward {t2 - t1:.1f}s, peak {mem:.1f} GiB, exact {exact:.3f}, "
                      f"{got}/16 tensors with a nonzero gradient", flush=True)
        except torch.cuda.OutOfMemoryError as err:
            print(f"{case}: rows {n}, T {T}: out of memory ({str(err)[:120]})", flush=True)
            del err
            torch.cuda.empty_cache()
            break
        del logits
