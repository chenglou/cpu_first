"""The training path of a strict file against its certified file, locally (STRICT.md §3): the reader and the cells at
the truth (MODE "designed"), the schedule moved to phase C, then evaluator-style training steps (forward,
token_training_loss, backward, clipping, the optimizer, the scheduler) on the same batches, so that the climb runs one
unit per step on the rows each file keeps (the certified file's lists, the strict file's buffers). Tiny budgets.
Prints both runs' climb lines without their timings and work counts, and whether they are the same.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/strict/checks/smoke_compare.py [KIND=sp2|s3i] [STEPS=30]
    [B=64] [K=V ...]   (K=V sets a constant in both files; STRICT_K=V in the strict file only, CERT_K=V in the
    certified file only: CERT_LATE_AFTER=10**9 is ABLATION_S3.md's no_late, the strict file's table lessons)
"""
import contextlib
import io
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STRICT = os.path.dirname(HERE)
INTEG = os.path.dirname(STRICT)
CPU = os.path.dirname(INTEG)
sys.path[:0] = [os.path.join(CPU, "wider_menus", "local")]
import torch  # noqa: E402
import local_wide as LW  # noqa: E402
from benchmark import OptimizerSpec, TokenLossBatch  # noqa: E402

argv = sys.argv[1:]
opt = dict(a.split("=", 1) for a in argv if "=" in a)
KIND, STEPS, B = opt.pop("KIND", "sp2"), int(opt.pop("STEPS", "30")), int(opt.pop("B", "64"))
PHASE, LIB = opt.pop("PHASE", "C"), opt.pop("LIB", None)       # PHASE=D: the round programs of LIB given, io 4
FILES = {"sp2": ("integration/self_practice/submission_stage3_sp2.py", "integration/strict/submission_stage3_sp2_strict.py"),
         "s3i": ("integration/submission_stage3_s3i.py", "integration/strict/submission_stage3_s3i_strict.py")}[KIND]
BASE = {"sp2": ['MODE="designed"', "LOG_EVERY=10", "SP_PROBE=30000", "ROUND_STATES=20000", "SP_LEVELS=2", "ANS_MIN=8"],
        "s3i": ['MODE="designed"', "LOG_EVERY=10", "SEARCH_STATES=40000", "FALLBACK_STATES=40000", "MAX_BODY=1",
                "MAX_COST=4", "LEVELS=3", "ROUND_STATES=20000", "ROUND_COST=3", "ANS_MIN=8"]}[KIND]
both = [f"{k}={v}" for k, v in opt.items() if not k.startswith(("STRICT_", "CERT_"))]
only = [f"{k[7:]}={v}" for k, v in opt.items() if k.startswith("STRICT_")]
cert = [f"{k[5:]}={v}" for k, v in opt.items() if k.startswith("CERT_")]


def run(path, sets):
    sc = LW.sc = LW.load_module(os.path.join("..", path), sets)
    torch.manual_seed(0)
    m = sc.build_model(LW.spec())
    m.ctl.phase, m.ctl.io, m.ctl.gated_on = PHASE, 5 if PHASE == "C" else 4, -1000
    m.ctl.cells_on = True
    if PHASE == "D":
        import json
        found = json.load(open(os.path.join(CPU, LIB)))
        lib = {k: sc.parse(v) for k, v in found["lib"].items()}
        progs = [sc.parse(p) for p in found["round"]]
        if hasattr(m, "set_round"):
            m.set_round(lib, progs)
        else:
            m.ctl.lib, m.ctl.progs = lib, progs
    m.train()
    bundle = sc.build_optimizer(m, OptimizerSpec(3600.0, "cpu"))
    rows = LW.load("train")
    g = torch.Generator().manual_seed(0)
    out = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(out):
        for step in range(1, STEPS + 1):
            idx = torch.randint(0, len(rows), (B,), generator=g)
            ids, am, lab, tpos = LW.batch([rows[i] for i in idx])
            logits, aux = m(ids, am)
            loss = sc.token_training_loss(TokenLossBatch(logits, lab, lab != -100, tpos, aux))
            bundle.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            bundle.optimizer.step()
            bundle.scheduler.step()
            print(f"STEP {step} loss {float(loss.detach()):.6f} phase {m.ctl.phase} io {m.ctl.io} ans_ok {m.ctl.ans_ok} "
                  f"fit {m.ctl.round_fit} streak {m.ctl.streak} t {m.ctl.t_log}")
    keep = []
    for l in out.getvalue().splitlines():
        if l.startswith(("LR climb", "LR round_found", "STEP", "LR step", "LR io_stage", "LR frozen")):
            l = re.sub(r" \[work \+[^]]*\]", "", l)                 # the strict file's work counts
            l = re.sub(r"\(\d+\.\d+s\)$", "(s)", l)                  # the unit's seconds
            l = re.sub(r"\b\d+\.\d+s\b", "_s", l)                     # seconds inside a line
            l = re.sub(r"(['\"]), \d+\.\d+\)", r"\1, _)", l)          # (how, seconds) pairs
            keep.append(l)
    work = sc.WORK[0] if hasattr(sc, "WORK") else None
    return keep, time.time() - t0, work, m


a, ta, _, _ = run(FILES[0], BASE + both + cert)
b, tb, work, m = run(FILES[1], BASE + both + only)
same = a == b
print(f"certified: {len(a)} lines in {ta:.1f}s; strict: {len(b)} lines in {tb:.1f}s, work {work:.4g}; "
      f"{'IDENTICAL' if same else 'DIFFERENT'}")
for x, y in zip(a, b):
    if x != y:
        print("  certified:", x[:300])
        print("  strict:   ", y[:300])
        break
for l in b:
    if l.startswith(("LR climb", "LR round_found")):
        print("  ", l[:220])
for l in b[-3:]:
    print("  ", l[:300])
print("buffers: pool", int(m.pool_sizes.sum()), "answers", int(m.ans_sizes.sum()), "programs", m.n_progs())
