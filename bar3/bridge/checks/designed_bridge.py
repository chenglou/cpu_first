"""(bar3/bridge: integration/checks/stage3_designed.py for the bridge variant, its library renamed) Stage 3's prediction path with the reader and every cell at the truth and LIBRARY.md's library given (the round
program x^2 mod N found offline, T.sqmod's composition): exact on H1's splits? Local, small (N rows per split).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_designed.py [N=8] [SPLIT ...]
    (env S3_FILE: the stage-3 file to check, default submission_stage3.py)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "integration")
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402

sys.path.insert(0, INTEG)
import local_check as lc  # noqa: E402

lc.glc.FILE = os.path.join(INTEG, os.environ.get("S3_FILE", "../bar3/bridge/submission_stage3_bridge.py"))
argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 8)
splits = [a for a in argv if "=" not in a] or ["depth_t_1", "depth_t_8", "depth_ood_n_t_2", "depth_ood_n_t_64"]
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.dirname(INTEG)), "one-layer-deeper", "data",
                                              "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
sc = lc.glc.load_module([("MODE", '"designed"')])
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
from benchmark import ModelSpec  # noqa: E402
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
lib = {k: sc.parse(v) for k, v in json.load(open(os.path.join(os.path.dirname(os.path.dirname(INTEG)), "cpu_first", "bar3", "bridge", "checks", os.environ.get("LIB", "r11_lib_bridge.json"))))["lib"].items()}
m.ctl.lib = lib
m.ctl.progs = [sc.parse("LDA.X STA.Y T.mul STA.X T.reduce")]
for split in splits:
    ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, split, n))
    t0 = time.time()
    with torch.no_grad():
        logits = m(ids, am)[0]
    tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1)
    ex = float(((tl == lab) | (lab == -100)).all(1).float().mean())
    print(f"{split} n={len(lab)} exact={ex:.3f} t={time.time() - t0:.1f}s", flush=True)
