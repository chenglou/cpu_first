"""The rule-8 gradient test of the learned-loop file (integration/checks/stage3_gradtest.py's, for this file): the
reader and every cell at the truth (MODE "designed"), sp2's official run's library, two answer programs in the
tournament (the count loop over each of sp2's two rounds: with one contestant the tournament's softmax is constant and
its logits get no gradient by construction), the final logits on a few H1 rows of one split; which parameter tensors
get a gradient through the count loop, and how large it is at each. Local, small.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/learned_loop/checks/gradtest_ll.py [N=2] [SPLIT=depth_t_1] [OBJ=logit|ce]
    (sp2 for comparison: LL_FILE=integration/self_practice/submission_stage3_sp2.py PROGS=rounds)
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
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.join(CPU, "integration"))
import local_check as lc  # noqa: E402

argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 2)
split = next((a[6:] for a in argv if a.startswith("SPLIT=")), "depth_t_1")
obj = next((a[4:] for a in argv if a.startswith("OBJ=")), "logit")
lc.glc.FILE = os.environ.get("LL_FILE", os.path.join(LL, "submission_stage3_ll.py"))
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(CPU), "one-layer-deeper", "data", "generated",
                                              "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
sc = lc.glc.load_module([("MODE", '"designed"')] + ([("GRAD_DAMP", os.environ["GRAD_DAMP"])] if os.environ.get("GRAD_DAMP") else []))
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
from benchmark import ModelSpec  # noqa: E402
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
lib = {k: sc.parse(v) for k, v in json.load(open(os.path.join(HERE, "sp2_official_lib.json"))).items()}
lib["T.round1"] = sc.parse("LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA")
lib["T.round2"] = sc.parse("LDA.X STA.Y T.mul.x,o.YX STA.X LDA.N T.red.x+on.XA")
m.ctl.lib = lib
m.ctl.progs = [sc.parse("C.S[T.round1 STA.X]"), sc.parse("C.S[T.round2 STA.X]")]
if os.environ.get("PROGS") == "rounds":    # sp2's file (its hand-written countdown): the two rounds themselves
    m.ctl.progs = [lib["T.round1"], lib["T.round2"]]
ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, split, n))
t0 = time.time()
logits = m(ids, am)[0]
tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)]
valid = lab != -100
if obj == "logit":           # the right tokens' logits: a gradient signal that does not vanish at a perfect prediction
    loss = -tl[valid].float().gather(1, lab[valid][:, None]).sum()
else:
    loss = F.cross_entropy(tl[valid].float(), lab[valid])
print(f"{split} n={n} exact={float(((tl.argmax(-1) == lab) | ~valid).all(1).float().mean()):.3f} {obj}={float(loss.detach()):.4f}"
      f" ({time.time() - t0:.1f}s forward)", flush=True)
loss.backward()
none, zero, got = [], [], []
for name, p in m.named_parameters():
    if p.grad is None:
        none.append(name)
    elif float(p.grad.abs().max()) == 0.0:
        zero.append(name)
    else:
        got.append((name, float(p.grad.abs().max())))
print(f"GRAD {len(list(m.parameters()))} tensors; with a gradient: {len(got)} | none: {none} | all zero: {zero}", flush=True)
for name, g in got:
    print(f"   {name}: max |grad| {g:.3e}", flush=True)
