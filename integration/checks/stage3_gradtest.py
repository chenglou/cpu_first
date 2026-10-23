"""The rule-8 gradient test of a stage-3 file: the reader and every cell at the truth (MODE "designed"), the round the
H1 run discovered (lib/s3d_lib.json: its library and its four contestants), the final logits' cross-entropy on a few
H1 rows of one split; which parameter tensors get a gradient, and how large it is at each table. Local, small.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_gradtest.py [N=2] [SPLIT=depth_t_1] [FILE=submission_stage3_s3d.py] [LIB=s3d_lib.json] [OBJ=ce|logit]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, INTEG)
import local_check as lc  # noqa: E402

argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 2)
split = next((a[6:] for a in argv if a.startswith("SPLIT=")), "depth_t_1")
lc.glc.FILE = os.path.join(INTEG, next((a[5:] for a in argv if a.startswith("FILE=")), "submission_stage3_s3d.py"))
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.dirname(INTEG)), "one-layer-deeper", "data",
                                              "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
sc = lc.glc.load_module([("MODE", '"designed"')])
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
from benchmark import ModelSpec  # noqa: E402
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
found = json.load(open(os.path.join(INTEG, "lib", next((a[4:] for a in argv if a.startswith("LIB=")), "s3d_lib.json"))))
m.ctl.lib = {k: sc.parse(v) for k, v in found["lib"].items()}
m.ctl.progs = [sc.parse(p) for p in found["round"]]
ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, split, n))
logits = m(ids, am)[0]
tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)]
valid = lab != -100
obj = next((a[4:] for a in argv if a.startswith("OBJ=")), "ce")
if obj == "logit":           # the right tokens' logits: a gradient signal that does not vanish at a perfect prediction
    loss = -tl[valid].float().gather(1, lab[valid][:, None]).sum()
else:
    loss = F.cross_entropy(tl[valid].float(), lab[valid])
print(f"{split} n={n} exact={float(((tl.argmax(-1) == lab) | ~valid).all(1).float().mean()):.3f} {obj}={float(loss.detach()):.4f}",
      flush=True)
loss.backward()
none, zero, got = [], [], []
for name, p in m.named_parameters():
    if p.grad is None:
        none.append(name)
    elif float(p.grad.abs().max()) == 0.0:
        zero.append(name)
    else:
        got.append((name, float(p.grad.abs().max())))
print("GRAD tensors with a gradient:", len(got), "| none:", none, "| all zero:", zero, flush=True)
for name, g in got:
    print(f"   {name}: max |grad| {g:.3e}", flush=True)
