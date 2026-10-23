"""T's direction lesson in log space (s3h) against the certified file's (s3g), on the same model and rows: the same loss
and gradients where the weights are in float32's range, and a finite gradient where the right reading's weight is far
below it (a confidently wrong T reading, as seed 6's phase D had). Local, small (32 rows of depth_t_16, T = 16, whose
two readings differ: 16 and 61). And s3i's (the countdown on the round machine) against s3h's: the same loss and
gradients bit for bit.

    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_tlesson_check.py [N=32]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTEG = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(os.path.dirname(INTEG), "learned_remainder", "chain_local", "stub"), INTEG]
import torch  # noqa: E402
import local_check as lc  # noqa: E402
from benchmark import ModelSpec  # noqa: E402

argv = sys.argv[1:]
n = next((int(a[2:]) for a in argv if a.startswith("N=")), 32)
data = os.environ.get("H1_DATA", os.path.join(os.path.dirname(os.path.dirname(INTEG)), "one-layer-deeper", "data",
                                              "generated", "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))
cfg = json.load(open(os.path.join(data, "dataset_config.json")))
mods = {}
for tag, f in (("s3g", "submission_stage3.py"), ("s3h", "submission_stage3_s3h.py"), ("s3i", "submission_stage3_s3i.py")):
    lc.glc.FILE = os.path.join(INTEG, f)
    mods[tag] = lc.glc.load_module([("MODE", '"designed"')])
ids, am, lab, tpos = lc.glc.batch(lc.glc.rows(data, "depth_t_16", n))
torch.manual_seed(0)
base = mods["s3g"].build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000)).state_dict()
NAMES = ("anc", "strd", "off", "absp", "shape")


def lesson(tag, edit=None):
    sc = mods[tag]
    m = sc.build_model(ModelSpec(cfg["vocab_size"], cfg["max_seq_len"], 500_000_000))
    m.load_state_dict(base)
    if edit is not None:
        with torch.no_grad():
            edit(m)
    lib = {k: sc.parse(v) for k, v in json.load(open(os.path.join(INTEG, "lib", "r11_lib.json")))["lib"].items()}
    m.ctl.lib, m.ctl.progs = lib, [sc.parse("LDA.X STA.Y T.mul STA.X T.reduce")]
    m.train()
    _, aux = m(ids, am)
    _, yd = m.io_label_lessons(aux, lab, lab != -100, tpos)
    m.zero_grad()
    lt = m.t_lesson(aux, yd)
    lt.backward()
    return float(lt), {k: getattr(m, k).grad.clone() if getattr(m, k).grad is not None else None for k in NAMES}, \
        m.ctl.t_log


def compare(title, edit=None):
    (l0, g0, t0), (l1, g1, t1), (l2, g2, t2) = lesson("s3g", edit), lesson("s3h", edit), lesson("s3i", edit)
    print(f"{title}: loss s3g {l0:.6g} s3h {l1:.6g} | s3g: {t0} | s3h: {t1}", flush=True)
    same = l2 == l1 and t2 == t1 and all((g1[k] is None and g2[k] is None) or torch.equal(g1[k], g2[k]) for k in NAMES)
    print(f"   s3i the same as s3h (loss, log, every gradient bit for bit): {same}", flush=True)
    for k in NAMES:
        a, b = g0[k], g1[k]
        if a is None or b is None:
            print(f"   {k}: grads {a is not None} {b is not None}")
            continue
        fin = lambda t: bool(torch.isfinite(t).all())
        rel = float((a - b).abs().max() / a.abs().max().clamp_min(1e-30)) if fin(a) else float("nan")
        print(f"   {k}: |grad| max s3g {float(a.abs().max()):.4g} s3h {float(b.abs().max()):.4g}, finite "
              f"{fin(a)} {fin(b)}, max rel diff {rel:.2e}", flush=True)


def wrong_t(margin):
    """T's anchor on the T marker and its stride +1: the snapped reading is the wrong direction (16 read as 61); the
    right reading's weight is about e^(-4 margin)."""
    def edit(m):
        V = m.V
        m.anc[2].fill_(-margin)
        m.anc[2][4] = margin                                                # the T marker
        m.anc[2][V] = -margin                                               # END: the right reading's anchor
        m.strd[2] = torch.tensor([margin, -margin])
    return edit


compare("designed reading (the right T direction)")
compare("near the start (every T logit 0)", lambda m: (m.anc[2].zero_(), m.strd[2].zero_()))
for margin in (3.0, 6.0, 12.0, 20.0):
    compare(f"T confidently wrong, margin {margin} (right weight ~e^-{4 * margin:g})", wrong_t(margin))
