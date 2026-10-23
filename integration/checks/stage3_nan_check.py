"""Seed 6's phase-D crash (s3g: a non-finite training loss at step 635), reproduced on a small model: the reader at the
truth except T's pointer, which reads one place past the end of the prompt (where seed 6's T reading passed when its
direction turned: the anchor already at the end, the stride not yet), so every row's count reads 0. One training
step of the certified file (s3g), of s3h and of s3i, the lessons of phase D on, on real rows: is the loss finite?

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_nan_check.py [N=32]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "learned_remainder", "chain_local", "stub"), INTEG]
import torch  # noqa: E402
import local_check as lc  # noqa: E402
from benchmark import ModelSpec, TokenLossBatch  # noqa: E402

argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 32)
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.dirname(INTEG)), "one-layer-deeper", "data",
                                              "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, "depth_t_1", n))
for tag, f in (("s3g", "submission_stage3.py"), ("s3h", "submission_stage3_s3h.py"), ("s3i", "submission_stage3_s3i.py")):
    lc.glc.FILE = os.path.join(INTEG, f)
    sc = lc.glc.load_module([("MODE", '"designed"')])
    torch.manual_seed(0)
    m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000))
    lib = {k: sc.parse(v) for k, v in json.load(open(os.path.join(INTEG, "lib", "r11_lib.json")))["lib"].items()}
    ctl = m.ctl
    ctl.lib, ctl.progs = lib, [sc.parse("LDA.X STA.Y T.mul STA.X T.reduce")]
    ctl.phase, ctl.io, ctl.cells_on, ctl.gated_on, ctl.streak, ctl.ans_ok = "D", 4, True, -1000, 0, 1.0
    for broken in (False, True):
        with torch.no_grad():
            m.strd[2] = torch.tensor([5.0, -5.0] if broken else [-5.0, 5.0])   # T read from the end: +1 is past it
        m.train()
        logits, aux = m(ids, am)
        Cd = aux["Cd"]
        loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
        print(f"{tag} T pointer {'one past the end' if broken else 'as designed'}: counts read nonzero on "
              f"{int((Cd != 0).any(1).sum())}/{Cd.shape[0]} rows, gate lesson {float(m.cell_lessons(Cd)):.4f}, "
              f"training loss {float(loss.detach()):.4f} (finite: {bool(torch.isfinite(loss))})", flush=True)
