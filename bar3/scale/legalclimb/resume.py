"""The tiny tests (LEGALCLIMB.md §3): s4_x2's climb resumed at the start of its level 5 (its library, routines and
practised problems as recorded, bar3/scale/slice/records/s4_x2.json), inside LegalClimb with TARGET=rows.

- MODE=pieces (local, CPU, no search but a tiny vocabulary): the full-size mode's parts on handed programs. The look's
  near miss is handed (x², `LDA.X STA.Y T.mul.x,o.XY`): the roots and goals pose builds; the value's and the rest's
  acceptors on s4_x2's four spellings, with the identity check built by the learner's own multiply and addition;
  SUBGOAL; the climb's routine choice among the four; the cached full-size vocabulary (bodies up to 1).
- MODE=level (one B200): one whole level of the legal climb from that state (look, pose, probes, choose, sleep, the
  round's search). Does it pose x², find its rest at full size and get the round?

    cd cpu_first && PYTHONDONTWRITEBYTECODE=1 H1_DATA=... MODE=pieces ~/github/sotaku/venv/bin/python -m bar3.scale.legalclimb.resume
    env: MODE, FULL_BODY (the full-size vocabulary's bodies), IDCHECK, PROBE, SEARCH_S, ANSWERS (780), ROWS, OUT, TAG
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("SPELL", "16")

import torch  # noqa: E402

from integration.lib import libcore as L  # noqa: E402
from bar3.scale.legalclimb import core_l as C  # noqa: E402
from bar3.scale.legalclimb.lclimb import LegalClimb  # noqa: E402
from bar3.scale.legalclimb.rowclimb import RowClimbTarget  # noqa: E402

HERE = Path(__file__).resolve().parent
REC = HERE.parent / "slice" / "records" / "s4_x2.json"
X2 = "LDA.X STA.Y T.mul.x,o.XY"
SPELLINGS = ["P[INC BRG.X T.red.Bx+d.AN]", "P[BRG.X T.red.Bx+d.AN INC]", "LDA.N P[BRG.X T.red.Bx+d.AN]",
             "CLA P[BRG.X T.red.Bx+d.AN]"]
env = os.environ.get


def climb_at_level5(dev, ans, idcheck, full_body):
    """LegalClimb with TARGET=rows in s4_x2's state after its level 4: library, ops, practised problems."""
    rec = json.load(open(REC))
    t = RowClimbTarget(ans, dev, sources=("value", "rest"), lawnear=True, clean=True, idcheck=idcheck, look_cost=3,
                       full_body=full_body)
    c = LegalClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, target=t, rule="lpca", K=4,
                   NP=8, probe_budget=int(float(env("PROBE", 1e6))), deep_by="bindings", deep_focus="closure",
                   levels=10, search_s=float(env("SEARCH_S", 900)), fuse=4 if dev == "cuda" else 1,
                   patterns=["pair", "x,o"], laws=["lt", "red", "add", "mul"])
    c.round_too, c.subgoal_on, c.rowcheck, c.rest_deep = True, True, True, True
    c.lib = {k: L.parse(v) for k, v in rec["history"][4]["lib"].items()}
    for key, toks in rec["ops"].items():
        law, pat = key.split("@")
        for tok, bd in toks:
            if tok in c.lib:
                c.ops.add(law, pat, tok, tuple(bd))
    for h in rec["history"][:5]:
        for n in h["picks"]:
            c.practiced[n] = h["level"]
    c.level, c.phase = 5, "pose"
    return c


def pieces(dev, ans, rows):
    t0 = time.time()
    c = climb_at_level5(dev, ans, idcheck=True, full_body=1)
    t = c.target
    print("chain and comparison:", __import__("bar3.scale.legalclimb.rowclimb", fromlist=["x"]).ops_routine(
        c.ops, "red", ("x+on", "x+dn", "Bx+d", "pair")), "|", len(c.lib), "library entries")
    near = [(L.parse(X2), "A", 16)]
    t.passed = list(near)
    out = t.pose(c, rows, near)
    print("pose:", [(r.key, r.kind, [g.name for g in gs]) for r, gs in out], "| log", t.log)
    tr = next(r for r, _ in out if r.key == "tgt@XN")
    rr = next(r for r, _ in out if r.kind == "rest")
    va = t.acceptor(c, rows, tr, t.posed[0])
    print("value's acceptor, x² / x:", va.many([L.parse(X2), L.parse("LDA.X")], c.lib))
    ra = t.acceptor(c, rows, rr, "target")
    print("rest's acceptor (with the identity check), the four spellings:", dict(zip(SPELLINGS,
          ra.many([L.parse(s) for s in SPELLINGS], c.lib))))
    print("  the identity check's verdicts:", [e for e in t.log if "idcheck" in e])
    t.subgoal_scores(c, rows, [tr, rr])
    print("SUBGOAL:", t.subgoal)
    best = c.choose_routine([L.parse(s) for s in SPELLINGS], c.lib, rr)
    print("the climb's routine choice among the four:", L.show(best[0]))
    t1 = time.time()
    cls = t.classes(c)
    t2 = time.time()
    cls2 = t.classes(c)
    c.lib["T.zz.extra.XN"] = L.parse("LDA.X")                 # a new routine: only its new items are run
    t3 = time.time()
    cls3 = t.classes(c)
    print(f"full-size vocabulary (bodies up to 1): {len(cls)} classes in {t2 - t1:.1f} s; again {len(cls2)} "
          f"(cached: {cls2 is cls}); with one routine more {len(cls3)} in {time.time() - t3:.1f} s")
    print(f"done in {time.time() - t0:.0f} s")


def level(dev, ans, rows):
    t0 = time.time()
    c = climb_at_level5(dev, ans, idcheck=env("IDCHECK", "0") == "1", full_body=int(env("FULL_BODY", 0)) or None)
    deadline = t0 + float(env("SECONDS", 2400))
    units = []
    while c.level == 5 and not c.done and time.time() < deadline:
        t1 = time.time()
        line = c.step(rows, answers=ans, deadline=deadline)
        if line is None:
            break
        units.append(dict(t=round(time.time() - t0, 1), s=round(time.time() - t1, 1), line=line[:600]))
        print(f"UNIT {time.time() - t0:8.1f}s +{time.time() - t1:7.1f}s {line[:500]}", flush=True)
    out = Path(env("OUT", str(HERE / "records"))) / f"{env('TAG', 'resume')}.json"
    out.write_text(json.dumps(dict(units=units, rounds=[L.show(p) for p in c.rounds], records=c.records,
                                   lib={k: L.show(v) for k, v in c.lib.items()}, log=c.target.log,
                                   history=c.history, seconds=round(time.time() - t0, 1)), default=str))
    print("TOTAL", round(time.time() - t0, 1), "s; rounds", [L.show(p) for p in c.rounds][:4], flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = C.load_rows(int(env("ROWS", 4096)), seed=0)
    rows = (C.digits(rows, "modulus").to(dev), C.digits(rows, "x").to(dev))
    arows = C.load_rows(int(env("ANSWERS", 780)), seed=1, T=1)
    ans = tuple(C.digits(arows, k).to(dev) for k in ("modulus", "x", "result"))
    (pieces if env("MODE", "pieces") == "pieces" else level)(dev, ans, rows)


if __name__ == "__main__":
    main()
