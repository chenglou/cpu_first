"""READING.md §2: written by bar3/scale/reading/make_copies6.py from integration/self_practice/core.py, 11 + 1 + 1 + 1
asserted replacements (rounds 2 and 3's, marked "compose2" and "compose3", round 4's, marked "slice", round 5's,
marked "loops", then this study's, marked "reading"). The rest of this docstring is the original's.

The tiny core (SELF_PRACTICE.md §2): the self-practising climb (spcore.SelfClimb) alone, offline, on the data's rows
with exact cells, as integration/lib/profile_climb.py runs libcore.Climb: the practice problems at widths 1-2 from the
rows' own top parts, the round at full size against the answers of T = 1 rows. Every unit is logged; the record (the
levels' pools, statuses, picks, merges, routines, every search) is written to records/<TAG>.json as it goes.

    env: RULE (lp | ct2a2 | all), K (4), NP (16), PROBE (1e6), FALLBACK (1e6: primitives alone, once a root), DEEP (5.8e6) and DEEP_K (8: the
         closest roots' second probe), LOOK (3e5), ROUND (4e6), LEVELS (12),
         SECONDS (the climb's limit, 36000), SEARCH_S (one probe's limit, 900), ROWS (4096), ANSWERS (400), TAG,
         PATTERNS / LAWS (";" lists: a subset of the law book), OUT (the records folder),
         TARGET (round: the round at full size against the answers; small: x'^2 mod N' at the practice widths, the
         oracle arm), SOURCES (law,restrict,project,value,rest: which sources pose candidates)
"""
import json
import os
import time
from pathlib import Path

import torch

from bar3.scale.reading import hooks as H                   # compose3
from integration.lib import libcore as L
from bar3.scale.reading import spcore_r as S                    # compose2: the copy
from bar3.scale.reading import climb_r as CC

HERE = Path(__file__).resolve().parent
H1_DIR = os.environ.get("H1_DATA", str(HERE.parents[2] / "one-layer-deeper" / "data" / "generated" /
                                       "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"))


def load_rows(n, seed, T=None, split="train"):
    rows = [json.loads(l) for l in open(os.path.join(H1_DIR, split + ".jsonl"))]
    if T is not None:
        rows = [r for r in rows if r["time_steps"] == T]
    g = torch.Generator().manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=g)[:n].tolist()]


def digits(rows, key, L_=8):
    """A field's decimal strings as digit tensors [n, L] (LSB first), read, never computed."""
    return torch.tensor([[int(ch) for ch in reversed(str(r[key]))][:L_] + [0] * (L_ - min(L_, len(str(r[key]))))
                         for r in rows])


def main():
    env = os.environ.get
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = env("TAG", f"core_{env('RULE', 'lp')}")
    out = Path(env("OUT", str(HERE / "records"))) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(int(env("ROWS", 4096)), seed=0)
    Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
    arows = load_rows(int(env("ANSWERS", 400)), seed=1, T=1)
    if env("TARGET_FN", "x2") != "x2":                          # compose3: TARGET_FN's answers, by Python
        arows = [dict(r, result=H.TARGET_F(int(r["x"]), int(r["modulus"]))) for r in arows]
    ans = tuple(digits(arows, k).to(dev) for k in ("modulus", "x", "result"))
    cfg = dict(rule=env("RULE", "lp"), K=int(env("K", 4)), NP=int(env("NP", 16)),
               probe_budget=int(float(env("PROBE", 1e6))), fallback_budget=int(float(env("FALLBACK", 1e6))),
               deep_budget=int(float(env("DEEP", 5.8e6))), deep_k=int(env("DEEP_K", 8)),
               macro_after=int(env("MACRO_AFTER", 12)), macro_cost=int(env("MACRO_COST", 2)),
               deep_by=env("DEEP_BY", "root"),
               deep_focus={"1": True, "0": False}.get(env("DEEP_FOCUS", "1"), env("DEEP_FOCUS", "1")),
               look_budget=int(float(env("LOOK", 3e5))), round_budget=int(float(env("ROUND", 4e6))),
               levels=int(env("LEVELS", 12)), search_s=float(env("SEARCH_S", 900)) or None,
               fuse=int(env("FUSE", 4 if dev == "cuda" else 1)),
               patterns=env("PATTERNS").split(";") if env("PATTERNS") else None,      # compose: ";"
               laws=env("LAWS").split(";") if env("LAWS") else None)                  # compose: ";"
    sources = env("SOURCES", "law").split(",")
    if "law" not in sources:
        cfg["patterns"] = []
    target = None
    if env("TARGET", "round") == "small":                 # the oracle arm: the target at the practice widths
        from bar3.scale.reading.oracle_r import SmallTarget      # compose2: the copy
        if env("CLEAN_VALUES", "0") == "1":                    # compose: clean near-miss values only
            from bar3.scale.reading.clean_r import CleanTarget as SmallTarget
        target = SmallTarget(sources=tuple(x for x in sources if x != "law"))
        target.lawnear = env("LAWNEAR", "0") == "1"              # compose2: LAWNEAR
        target.restrict_rows = env("RESTRICT_ROWS", "0") == "1"  # compose2: RESTRICT_ROWS
    cfg_rec = dict(cfg, target=env("TARGET", "round"), sources=sources)
    c = CC.ComposeClimb(L.Cells(10, dev), dev, 16, chunk=1 << 21 if dev == "cuda" else 1 << 16, target=target, **cfg)
    c.round_too = env("ROUND_TOO", "0") == "1"                    # compose: the round's search too
    c.subgoal_on, c.tiebreak, c.rowcheck, c.rest_deep = (env(k, "0") == "1" for k in ("SUBGOAL", "TIEBREAK",
                                                                                    "ROWCHECK", "REST_DEEP"))
    flags = {k.lower(): True for k in ("CLEAN_VALUES", "SUBGOAL", "LAWNEAR", "ROWCHECK", "TIEBREAK",
                                       "RESTRICT_ROWS", "REST_DEEP") if env(k, "0") == "1"}   # compose2: on flags only
    if S.SPELL_K:
        flags["spell"] = S.SPELL_K
    c.tiefollow = env("TIEFOLLOW", "0") == "1"                    # compose3: TIEFOLLOW
    if c.tiefollow:
        flags["tiefollow"] = True
    if env("TARGET_FN", "x2") != "x2":
        flags["target_fn"] = env("TARGET_FN")
    if H.SLICE:                                                   # slice: the cuts posed
        flags["slice"] = H.SLICE
    if H.LOOPS:                                                   # loops: the cuts' loops posed
        flags["loops"] = H.LOOPS
    for _k in ("DOWN", "SUBLAW", "CONST", "PRIOR"):                 # reading: its flags
        if getattr(H, _k):
            flags[_k.lower()] = getattr(H, _k)
    cfg = dict(cfg_rec, round_too=c.round_too, **flags)
    t0 = time.time()
    deadline = t0 + float(env("SECONDS", 36000))
    print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "config", cfg, flush=True)
    units = []

    def save(final=False):
        rec = dict(tag=tag, config=cfg, seconds=round(time.time() - t0, 1), final=final, units=units,
                   history=c.history, records=[{k: v for k, v in r.items()} for r in c.records],
                   lib={k: L.show(v) for k, v in c.lib.items()}, rounds=[L.show(p) for p in c.rounds],
                   ops={f"{k[0]}@{k[1]}": [[t, "".join(b)] for t, b in v] for k, v in c.ops.by_key.items()},
                   near=c.near_log)
        tmp = out.with_suffix(".part")
        tmp.write_text(json.dumps(rec, default=str))
        tmp.rename(out)
    last = time.time()
    while True:
        t1 = time.time()
        line = c.step((Nd, Xd), answers=ans, deadline=deadline)
        if line is None:
            break
        dt = time.time() - t1
        units.append(dict(t=round(time.time() - t0, 1), s=round(dt, 1), line=line[:600]))
        mem = (f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB" if dev == "cuda" else "")
        print(f"UNIT {time.time() - t0:8.1f}s +{dt:7.1f}s {line[:400]}{mem}", flush=True)
        if time.time() - last > 60:
            save()
            last = time.time()
        if time.time() > deadline:
            print("DEADLINE", flush=True)
            break
    save(final=True)
    print("TOTAL", round(time.time() - t0, 1), "s; rounds", [L.show(p) for p in c.rounds][:4], flush=True)
    print("LIB", json.dumps({k: L.show(v) for k, v in c.lib.items()}), flush=True)


if __name__ == "__main__":
    main()
