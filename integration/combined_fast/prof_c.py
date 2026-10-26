"""The tiny core (core_c.py, unchanged) with timers: where the climb's seconds go on a GPU (FAST.md §1). Every function
named in TIMED is wrapped: its calls, its time with and without the timed functions it calls, by call path (the
timed functions above it), and, for the lockstep, by its rows and steps. With SYNC=1 a timed function waits for the
GPU at its entry and exit (the GPU's work then counts where it was launched; functions in NOSYNC never wait, being
too many or inside a graph's capture). A thread samples the GPU's utilization (nvidia-smi) twice a second. The climb
stops after the first unit whose line matches STOP (a regex), or at SECONDS. Each unit prints its own top functions;
the profile goes to OUT/<TAG>_prof.json.

    env: core_c.py's (FILE, ROWS, ANSWERS, SECONDS, TAG, OUT, ...), SYNC (1), STOP (a regex), TOP (15)
    cd cpu_first && PYTHONPATH=. python integration/combined_fast/prof_c.py
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core_c  # noqa: E402
import torch  # noqa: E402

TIMED = ("step run_items _run_count count_down apply_items _apply_grouped _apply_counts _lockstep _graph_steps _columns "
         "keys_of hash_rows _fresh _groups _signatures enumerate_items bfs bfs_roots programs run_prog fresh "
         "random_states scratch side_effects repeat_add times_add above_probes law_target top_parts bpe "
         "Batch.take Batch.concat Batch.copy Ops.run Draw.__init__ Draw.term Draw.operands Root.__init__ Root.goals "
         "Goal.test Goal.fits Acceptor.__init__ Acceptor.many RoundMachine.__init__ RoundMachine.run OuterMachine.run "
         "SelfClimb.probe SelfClimb.acceptors SelfClimb.comp_classes SelfClimb.prim_classes SelfClimb.signature "
         "SelfClimb.pose SelfClimb.classify SelfClimb.choose_routine SelfClimb.sleep SelfClimb.search_round "
         "SelfClimb.search_outer SelfClimb.deep_roots SelfClimb.pick SelfClimb.macro_classes SelfClimb.outer_classes "
         "_lock_fused _grouped_work first_programs"           # (the fast file's)
         ).split()
NOSYNC = set("step Batch.take Batch.concat Batch.copy keys_of hash_rows _fresh _groups fresh random_states count_of "
             "_columns".split())
SYNC = os.environ.get("SYNC", "1") == "1" and torch.cuda.is_available()
STOP = os.environ.get("STOP")
TOP = int(os.environ.get("TOP", 15))
flat = defaultdict(lambda: [0, 0.0, 0.0])          # name -> [calls, inclusive s, exclusive s]
paths = defaultdict(lambda: [0, 0.0, 0.0])         # (timed ancestors..., name) -> the same
locks = defaultdict(lambda: [0, 0.0])              # (log2 rows, log2 steps, NP, graphed) -> [calls, s]
stack, active = [], defaultdict(int)
util = []                                          # (time, GPU utilization %)


def _sync():
    if SYNC:
        torch.cuda.synchronize()


def wrap(name, fn):
    nosync = name in NOSYNC

    def timed(*a, **k):
        if active[name]:                           # a recursive call: timed as part of the outermost
            return fn(*a, **k)
        active[name] += 1
        if not nosync:
            _sync()
        t0 = time.perf_counter()
        stack.append([name, 0.0])
        try:
            return fn(*a, **k)
        finally:
            if not nosync:
                _sync()
            dt = time.perf_counter() - t0
            _, child = stack.pop()
            key = tuple(s[0] for s in stack) + (name,)
            for d in (flat[name], paths[key]):
                d[0] += 1
                d[1] += dt
                d[2] += dt - child
            if stack:
                stack[-1][1] += dt
            active[name] -= 1
            if name == "_lockstep":
                S, n = a[1].regs.shape[0], a[2].shape[0]
                g = int(a[1].regs.device.type == "cuda" and getattr(M, "GRAPH_STEPS", 0) and n >= M.GRAPH_MIN)
                e = locks[(S.bit_length(), n.bit_length(), a[1].regs.shape[3], g)]
                e[0] += 1
                e[1] += dt
    timed.__wrapped__ = fn
    return timed


def instrument(mod):
    for name in TIMED:
        cls, _, meth = name.rpartition(".")
        owner = getattr(mod, cls) if cls else mod
        if cls and not hasattr(owner, meth) or not cls and not hasattr(mod, name):
            print("PROF: no", name, flush=True)
            continue
        if cls:
            setattr(owner, meth, wrap(name, getattr(owner, meth)))
        else:
            setattr(mod, name, wrap(name, getattr(mod, name)))


def sampler():
    try:
        p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
                              "-lms", "500"], stdout=subprocess.PIPE, text=True)
    except OSError:
        return
    for line in p.stdout:
        try:
            util.append((time.perf_counter(), float(line.strip())))
        except ValueError:
            pass


def snapshot():
    return {k: list(v) for k, v in flat.items()}


def top(before, after, k):
    rows = []
    for n, (c, inc, exc) in after.items():
        b = before.get(n, [0, 0.0, 0.0])
        if c - b[0]:
            rows.append((exc - b[2], inc - b[1], c - b[0], n))
    rows.sort(reverse=True)
    return rows[:k]


M = None
_load = core_c.load


def load(path, sets):
    global M
    M, md5 = _load(path, sets)
    instrument(M)
    step = M.SelfClimb.step
    state = dict(stop=False, units=[])

    def unit(self, *a, **k):
        if state["stop"]:
            return None
        before, t0 = snapshot(), time.perf_counter()
        line = step(self, *a, **k)
        t1 = time.perf_counter()
        u = [x for t, x in util if t0 <= t <= t1]
        rows = top(before, snapshot(), TOP)
        state["units"].append(dict(line=(line or "")[:300], s=round(t1 - t0, 2),
                                   gpu=round(sum(u) / len(u), 1) if u else None,
                                   top=[[n, round(e, 3), round(i, 3), c] for e, i, c, n in rows]))
        print(f"PROF unit {t1 - t0:.1f}s gpu {state['units'][-1]['gpu']}% | "
              + "; ".join(f"{n} {e:.1f}/{i:.1f}s x{c}" for e, i, c, n in rows[:10]), flush=True)
        if STOP and line and re.search(STOP, line):
            state["stop"] = True
        save()
        return line

    def save():
        out = Path(os.environ.get("OUT", str(HERE / "records"))) / f"{os.environ.get('TAG', 'prof')}_prof.json"
        rec = dict(sync=SYNC, flat={k: [v[0], round(v[1], 3), round(v[2], 3)] for k, v in flat.items()},
                   paths=[[list(k), v[0], round(v[1], 3), round(v[2], 3)] for k, v in
                          sorted(paths.items(), key=lambda kv: -kv[1][1])],
                   locks=[[list(k), v[0], round(v[1], 3)] for k, v in sorted(locks.items())],
                   units=state["units"], util_mean=round(sum(x for _, x in util) / len(util), 1) if util else None)
        tmp = out.with_suffix(".part")
        tmp.write_text(json.dumps(rec))
        tmp.rename(out)
    M.SelfClimb.step = unit
    return M, md5


core_c.load = load
if __name__ == "__main__":
    threading.Thread(target=sampler, daemon=True).start()
    core_c.main()
    print("PROF flat (exclusive s, inclusive s, calls):", flush=True)
    for n, (c, inc, exc) in sorted(flat.items(), key=lambda kv: -kv[1][2])[:40]:
        print(f"PROF   {n:28s} {exc:9.1f} {inc:9.1f} {c:10d}", flush=True)
    print("PROF lockstep (log2 rows, log2 steps, NP, graphed): calls, s", flush=True)
    for k, (c, s) in sorted(locks.items(), key=lambda kv: -kv[1][1])[:30]:
        print(f"PROF   {k} {c} {s:.1f}", flush=True)
