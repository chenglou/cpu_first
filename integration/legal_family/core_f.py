"""Phase C's tiny core (legal_family/FAMILY_FILE.md): combined_fast/core_c.py with a task (TASK) whose answers are made
here by Python's integers on H1's prompts, as the family's data makes them (litmus/tasks.py: H1's prompts row for row,
other answers), and ROUNDT's rows (the climb's `more`: up to the file's ROUNDT rows of each T = 2, 4, 8, the counts a
run keeps past 1). Below, core_c.py's own description.

The tiny core (SELF_PRACTICE.md §2, GENERIC_OPS.md §3) for the combined file: the file's own climb (its SelfClimb with
its own SP settings: g3's search rules and instruction set, the generated book, the fixed checks, the work budgets),
alone, offline, on the data's rows with exact cells; then, once the round is found, the answer program's search
(search_outer) on rows of T = 1, 2, 4 and 8, as the file runs it, and the programs it accepts on rows it never saw.
It trains nothing, and its times do not carry over to the evaluator (GENERIC_OPS.md §2.1, §3): it screens out a
climb that never reaches the round. Every unit is logged with the machine work it did (the file's WORK); the record
goes to OUT/<TAG>.json as it goes.

    env: TASK (h1, cube, ten, rho, plus3, minus1, tenrho, plusy, three2), FILE (the combined file), ROWS (4096 read rows), ANSWERS (400 rows of T = 1), COUNT_ROWS (64 of each T),
         TS (1,2,4,8), SECONDS (a safety limit on the wall clock; the climb itself stops on its work budget,
         CLIMB_WORK), SETS (NAME=VALUE;... constants set in the file's text, research only), TAG, OUT,
         ROUNDS_FROM (a JSON {lib, round}: no climb, its rounds given; a check of the answer program's part)
    cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_sp2/core_c.py
"""
import hashlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "learned_remainder" / "chain_local" / "stub"))   # the evaluator's API
import torch  # noqa: E402

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


STEPS = {                                         # one round of each task (litmus/tasks.py, tasks_general.py, tasks_family.py)
    "h1": lambda y, n: y * y % n, "cube": lambda y, n: y ** 3 % n, "ten": lambda y, n: 10 * y * y % n,
    "rho": lambda y, n: (y * y + 1) % n, "plus3": lambda y, n: (y * y + 3) % n, "minus1": lambda y, n: (y * y - 1) % n,
    "tenrho": lambda y, n: (10 * y * y + 1) % n, "plusy": lambda y, n: (y * y + y) % n,
    "three2": lambda y, n: (3 * y * y + 2) % n,
}
TASK = os.environ.get("TASK", "h1")


def answered(rows):
    """The rows with the task's answers: H1's own; another task's, its round T times by Python's integers."""
    out = []
    for r in rows:
        y, n = int(r["x"]), int(r["modulus"])
        for _ in range(int(r["time_steps"])):
            y = STEPS[TASK](y, n)
        assert TASK != "h1" or y == int(r["result"])
        out.append(dict(r, result=y))
    return out


def load(path, sets):
    src = open(path).read()
    for kv in sets:
        k, v = kv.split("=", 1)
        src, n = re.subn(rf"^{k} = .*$", f"{k} = {v}", src, count=1, flags=re.M)
        assert n == 1, k
    mod = types.ModuleType("combined")
    sys.modules["combined"] = mod
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod, hashlib.md5(src.encode()).hexdigest()


def main():
    env = os.environ.get
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = env("TAG", "core_combined")
    out = Path(env("OUT", str(HERE / "records"))) / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sets = [s for s in env("SETS", "").split(";") if s]
    M, src_md5 = load(env("FILE", str(HERE / "submission_stage3_combined.py")), sets)
    rows = load_rows(int(env("ROWS", 4096)), seed=0)
    Nd, Xd = digits(rows, "modulus").to(dev), digits(rows, "x").to(dev)
    arows = answered(load_rows(int(env("ANSWERS", 400)), seed=1, T=1))
    ans = tuple(digits(arows, k).to(dev) for k in ("modulus", "x", "result"))
    c = M.SelfClimb(M.Cells(10, dev), dev, M.PR, chunk=1 << 21 if dev == "cuda" else 1 << 18, **M.SP)
    for T in (2, 4, 8):                                           # ROUNDT's rows (the file's keep_counts in a run)
        mr = answered(load_rows(M.ROUNDT, seed=10 + T, T=T))
        c.more.append((T, tuple(digits(mr, k).to(dev) for k in ("modulus", "x", "result"))))
    cfg = dict({k: (list(v) if isinstance(v, tuple) else v) for k, v in M.SP.items()}, sets=sets, file_md5=src_md5,
               SEARCH_WORK=M.SEARCH_WORK, CLIMB_WORK=M.CLIMB_WORK, prims=list(M.PRIMS), loops=list(M.LOOP_KINDS),
               isa=f"{M.COND},{M.SHIFT}" + (",brg" if M.BRING else ""), task=TASK,
               roundt=[[T, int(r[0].shape[0])] for T, r in c.more])
    t0, w0 = time.time(), M.WORK[0]
    stop_at = w0 + M.CLIMB_WORK if M.CLIMB_WORK else None        # the file's climb budget, from the climb's start
    limit = t0 + float(env("SECONDS", 36000))
    print("device", torch.cuda.get_device_name() if dev == "cuda" else "cpu", "config", cfg, flush=True)
    units, outer = [], None

    def save(final=False):
        rec = dict(tag=tag, config=cfg, seconds=round(time.time() - t0, 1), final=final, work=M.WORK[0] - w0,
                   units=units, history=c.history, records=c.records, lib={k: M.show(v) for k, v in c.lib.items()},
                   rounds=[M.show(p) for p in c.rounds], outer=outer,
                   ops={f"{k[0]}@{k[1]}": [[t, "".join(b)] for t, b in v] for k, v in c.ops.by_key.items()})
        tmp = out.with_suffix(".part")
        tmp.write_text(json.dumps(rec, default=str))
        tmp.rename(out)
    last = time.time()
    if env("ROUNDS_FROM"):                    # (a check of the answer program's part: a record's library and rounds)
        d = json.load(open(env("ROUNDS_FROM")))
        c.lib, c.rounds = {k: M.parse(v) for k, v in d["lib"].items()}, [M.parse(r) for r in d["round"]]
    while not env("ROUNDS_FROM"):
        t1, w1 = time.time(), M.WORK[0]
        line = c.step((Nd, Xd), answers=ans, stop_at=stop_at)
        if line is None:
            break
        dt = time.time() - t1
        units.append(dict(t=round(time.time() - t0, 1), s=round(dt, 1), work=M.WORK[0] - w1, total=M.WORK[0] - w0,
                          line=line[:600]))
        mem = f" | GPU peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB" if dev == "cuda" else ""
        print(f"UNIT {time.time() - t0:8.1f}s +{dt:7.1f}s work +{M.WORK[0] - w1:.4g} = {M.WORK[0] - w0:.4g} "
              f"{line[:400]}{mem}", flush=True)
        if time.time() - last > 60:
            save()
            last = time.time()
        if time.time() > limit:
            print("SECONDS: the safety limit (not the file's)", flush=True)
            break
    print("CLIMB", round(time.time() - t0, 1), "s, work", f"{M.WORK[0] - w0:.4g}", "of", f"{M.CLIMB_WORK:.4g}",
          "; rounds", [M.show(p) for p in c.rounds][:4], flush=True)
    if c.rounds:                                                  # the answer's program, as the file's outer_unit
        rounds = sorted(c.rounds, key=lambda p: (M.cost(M.expand(p, c.lib)), M.show(p)))[:M.PROG_SLOTS]
        ts = [int(t) for t in env("TS", "1,2,4,8").split(",")]
        crows = answered([r for t in ts for r in load_rows(int(env("COUNT_ROWS", 64)), seed=t, T=t)])
        pool = (digits(crows, "modulus").to(dev), digits(crows, "x").to(dev),
                digits(crows, "time_steps", M.K).to(dev), digits(crows, "result").to(dev))
        t1, w1 = time.time(), M.WORK[0]
        progs, lib2, rec = c.search_outer(pool, rounds, stop_at, M.OUTER_STATES, M.OUTER_COST, M.OUTER_BODY,
                                          M.OUTER_NP)
        rec.update(work=M.WORK[0] - w1, wall=round(time.time() - t1, 1), ts=ts,
                   rounds={f"T.round{i + 1}": M.show(p) for i, p in enumerate(rounds)})
        test = [r for t in (1, 2, 4, 8, 16) for r in load_rows(32, seed=100 + t, T=t)]
        test += [r for t in (32, 64) for r in load_rows(32, seed=100 + t, T=t, split=f"depth_t_{t}")]
        test = answered(test)
        tp = (digits(test, "modulus").to(dev), digits(test, "x").to(dev), digits(test, "time_steps", M.K).to(dev),
              digits(test, "result").to(dev))
        st = M.round_settings(c.cells, M.PR)
        rec["unseen"] = {}
        for p in progs:
            A = M.outer_machine(st, p, lib2, tp[0].shape[0], dev).run(tp[1], tp[0], tp[2], M.gen_on(dev, 5))
            rec["unseen"][M.show(p)] = float((A[:, :8] == tp[3]).all(1).float().mean())
        rec["unseen_rows"] = len(test)
        outer = rec
        print("OUTER", json.dumps({k: v for k, v in rec.items() if k != "level_sizes"}, default=str), flush=True)
    save(final=True)
    print("TOTAL", round(time.time() - t0, 1), "s, work", f"{M.WORK[0] - w0:.4g}", flush=True)
    print("LIB", json.dumps({k: M.show(v) for k, v in c.lib.items()}), flush=True)


if __name__ == "__main__":
    main()
