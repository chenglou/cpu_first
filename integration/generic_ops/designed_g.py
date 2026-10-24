"""The built file's prediction path and round machine with the reader and every cell at the truth (MODE = "designed")
and a hand-written round on its instruction set given (integration/checks/stage3_designed.py for this folder's builds;
the programs parsed by libcore_g, since the build drops the file's own unused parser): exact on H1's splits?
  * the prediction path: the reader, the countdown with the round on one-hot digits (SteMachine), the writer;
  * the round machine (run_rounds, phase D's hard machine) on the same rows' x, N and T.
Local, small (N rows per split; CPU, about a minute).

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/generic_ops/designed_g.py [N=4] [FILE=...] [ISA=if1,k] [SPLIT ...]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "learned_remainder", "chain_local", "stub")]
import torch  # noqa: E402

sys.path.insert(0, INTEG)
import local_check as lc  # noqa: E402
from integration.generic_ops import isa as I  # noqa: E402
from integration.generic_ops.test_isa import lib_for  # noqa: E402

argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 4)
name = next((a[5:] for a in argv if a.startswith("FILE=")), "submission_stage3_g.py")
isa = next((a[4:] for a in argv if a.startswith("ISA=")), "if1,k")
splits = [a for a in argv if "=" not in a] or ["depth_t_1", "depth_t_8", "depth_ood_n_t_2", "depth_ood_n_t_64"]
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.dirname(INTEG)), "one-layer-deeper", "data",
                                              "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
lc.glc.FILE = os.path.join(HERE, name)
sc = lc.glc.load_module([("MODE", '"designed"')])
L, _ = I.load(isa)
assert list(sc.PRIMS) == list(L.PRIMS) and tuple(sc.LOOP_KINDS) == tuple(L.LOOP_KINDS), "the file's set is not ISA's"
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
from benchmark import ModelSpec  # noqa: E402
m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).eval()
m.ctl.lib = lib_for(L, isa)
m.ctl.progs = [L.parse("LDA.X STA.Y T.mul STA.X LDA.N T.red")]
print(f"{name}: {len(sc.PRIMS)} primitives, loops {sc.LOOP_KINDS}; round {L.show(m.ctl.progs[0])}, T.link = "
      f"{L.show(m.ctl.lib['T.link'])}, T.red = {L.show(m.ctl.lib['T.red'])}", flush=True)
bad = 0
for split in splits:
    rows = lc.glc.rows(data, split, n)
    ids, am, lab, tpos = lc.glc.batch(rows)
    t0 = time.time()
    with torch.no_grad():
        logits = m(ids, am)[0]
    tl = logits[torch.arange(len(lab))[:, None], tpos.clamp_min(0)].argmax(-1)
    ex = float(((tl == lab) | (lab == -100)).all(1).float().mean())
    dg = lambda key: torch.tensor([[int(c) for c in reversed(str(r[key]))][:8] + [0] * (8 - min(8, len(str(r[key]))))
                                   for r in rows])
    Td = torch.tensor([[int(c) for c in reversed(str(r["time_steps"]))][:2] + [0] * (2 - len(str(r["time_steps"])))
                       for r in rows])
    with torch.no_grad():
        s = m.run_rounds(dg("x"), Td, dg("modulus"), m.ctl.progs[0])
    hard = float((s == dg("result")).all(1).float().mean())
    bad += ex < 1 or hard < 1
    print(f"{split} n={len(lab)}: prediction path exact {ex:.3f}, round machine exact {hard:.3f} "
          f"({time.time() - t0:.1f}s)", flush=True)
print("all exact" if not bad else f"{bad} splits not exact")
