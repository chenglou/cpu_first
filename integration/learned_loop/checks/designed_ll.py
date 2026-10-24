"""The learned-loop file's prediction path with the reader and every cell at the truth, sp2's official run's library
and round given, and the answer program C.S[T.round1 STA.X] (the count loop over the round): exact on H1's splits,
T = 64 included? The plumbing check of integration/checks/stage3_designed.py for this file. Local, small (N rows per
split), CPU.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/designed_ll.py [N=8] [SPLIT ...]
    (env LL_FILE: the file to check, default integration/learned_loop/submission_stage3_ll.py)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LL = os.path.dirname(HERE)
CPU = os.path.dirname(os.path.dirname(LL))
sys.path[:0] = [os.path.join(CPU, "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402

sys.path.insert(0, os.path.join(CPU, "integration"))
import local_check as lc  # noqa: E402

lc.glc.FILE = os.environ.get("LL_FILE", os.path.join(LL, "submission_stage3_ll.py"))
argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 8)
splits = [a for a in argv if "=" not in a] or ["depth_t_1", "depth_t_8", "depth_t_16", "depth_ood_n_t_2",
                                               "depth_ood_n_t_64"]
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(CPU), "one-layer-deeper", "data", "generated",
                                              "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
sc = lc.glc.load_module([("MODE", '"designed"')])
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
from benchmark import ModelSpec  # noqa: E402
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
lib = {k: sc.parse(v) for k, v in json.load(open(os.path.join(HERE, "sp2_official_lib.json"))).items()}
lib["T.round1"] = sc.parse("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA")
m.ctl.lib = lib
m.ctl.progs = [sc.parse("C.S[T.round1 STA.X]")]
print("parameters", sum(p.numel() for p in m.parameters()), "tensors", len(list(m.parameters())), flush=True)
fails = 0
for split in splits:
    ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, split, n))
    t0 = time.time()
    with torch.no_grad():
        logits = m(ids, am)[0]
    tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1)
    ex = float(((tl == lab) | (lab == -100)).all(1).float().mean())
    fails += ex < 1.0
    print(f"{split} n={len(lab)} exact={ex:.3f} t={time.time() - t0:.1f}s", flush=True)
print("ALL EXACT" if not fails else f"{fails} splits not exact")
