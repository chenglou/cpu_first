"""One capped search, measured (INTEGRATION.md, the A/B's state cap): the c5 climbs' level-4 vocabulary (rebuilt from
their logged merges) on the mul problem (the search both c5 climbs exhaust), on this device. Prints, per depth, the
states, the seconds, the host process's CPU seconds, the GPU's busy share and memory (nvidia-smi sampled every 0.2 s)
and torch's peak memory; then where one level's time goes (torch.profiler: CUDA kernel time vs. wall time); then the
cap that the device's memory and a time limit per search allow. Not a lesson: a measurement.

    python -m integration.lib.profile_search            (env: TASK=mul, CAP=3000000, SEARCH_S=600, MAX_COST=5)
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import torch

from integration.lib import libcore as L, inclimb

TASK = os.environ.get("TASK", "mul")
CAP = int(os.environ.get("CAP", 3_000_000))
SEARCH_S = float(os.environ.get("SEARCH_S", 600))        # the time one search may take (the climb's own limit)
MAX_COST = int(os.environ.get("MAX_COST", 5))
L4 = {"B0.1": "SBB.N INC", "B0.2": "P[B0.1]", "B0.3": "CLC B0.2", "T.lt": "CLC P[LDA.X B0.1]", "T.lt~": "CLC LDA.Y B0.2",
      "B1.1": "STA.S B0.3", "B1.2": "B1.1 SEL.S", "T.modx": "LDA.X B1.2", "T.modx~": "LDA.Y B1.2",
      "B2.1": "LDA.N CLC", "B2.2": "CLC LDA.N", "B2.3": "ADC.N INC", "B2.4": "P[B2.3]", "B2.5": "CLC B2.4",
      "T.add": "B2.2 P[ADC.X INC]", "T.add~": "B2.2 P[ADC.Y INC]",
      "B3.1": "B[B1.2]", "B3.2": "CLK CLA", "B3.3": "CLA CLK", "T.modB": "LDA.X B3.1", "T.modB~": "LDA.Y B3.1",
      "T.mrow": "LDH.Y B3.2 P[MUL.X INC]", "T.mrow~": "LDH.X B3.2 P[MUL.Y INC]"}


class GpuSampler:
    """nvidia-smi's utilization and memory every 0.2 s, in a thread; mark() starts a window, window() summarizes it."""

    def __init__(self):
        self.samples, self.t_mark = [], time.time()
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                          "--format=csv,noheader,nounits", "-lms", "200"],
                                         stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.proc = None
            return
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                u, m, tot = (float(v) for v in line.strip().split(","))
            except ValueError:
                continue
            self.samples.append((time.time(), u, m, tot))

    def mark(self):
        self.t_mark = time.time()

    def window(self):
        w = [s for s in self.samples if s[0] >= self.t_mark]
        if not w:
            return {}
        return {"gpu_busy_pct": round(sum(s[1] for s in w) / len(w), 1), "gpu_mem_max_gb": round(max(s[2] for s in w) / 1024, 1),
                "gpu_mem_total_gb": round(w[0][3] / 1024, 1), "samples": len(w)}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"task": TASK, "cap": CAP, "device": torch.cuda.get_device_name() if dev == "cuda" else "cpu"}
    gs = GpuSampler()
    rows = inclimb.load_rows(4000, seed=0)
    Nd, Xd, _ = (t.to(dev) for t in inclimb.digits_of_rows(rows))
    cells = L.Cells(10, dev)
    lib = {k: L.parse(v) for k, v in L4.items()}
    settings = [L.Setting(10, 1, cells), L.Setting(10, 2, cells)]
    t0, c0 = time.time(), time.process_time()
    gs.mark()
    classes = L.enumerate_items(list(L.PRIMS) + list(lib), lib, settings, dev, max_body=3)
    out["enumerate"] = {"classes": len(classes), "s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
                        **gs.window()}
    print("PROFILE enumerate", json.dumps(out["enumerate"]), flush=True)
    helpers = {"add": lib["T.add"]}
    prob = L.LawProblem(TASK, settings, 8, 0, (Nd, Xd), helpers, lib, dev)
    out["depths"] = []
    for d in range(2, MAX_COST + 1):                    # each depth from scratch: the last level's cost is the difference
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0, c0 = time.time(), time.process_time()
        gs.mark()
        res = L.bfs(prob, classes, lib, max_cost=d, max_states=CAP, chunk=1 << 21)
        if dev == "cuda":
            torch.cuda.synchronize()
        rec = {"max_cost": d, "levels": res.level_sizes, "stop": res.stop, "found": res.found_cost,
               "s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
               "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev == "cuda" else None,
               **gs.window()}
        out["depths"].append(rec)
        print("PROFILE depth", json.dumps(rec), flush=True)
        del res
        if rec["s"] > 2 * SEARCH_S:
            break
    # where the time of the deepest stored level's expansion goes: one chunk of 2M rows through one item class
    if dev == "cuda":
        res = L.bfs(prob, classes, lib, max_cost=3, max_states=CAP, chunk=1 << 21)
        from torch.profiler import profile, ProfilerActivity
        cost1 = [c.rep for c in classes if c.cost == 1][:8]
        loops = [c.rep for c in classes if c.cost == 3][:256]
        roots = prob.roots
        big = [b.take(torch.zeros(1 << 21, dtype=torch.long, device=dev)) for b in roots]     # 2M copies of the root
        for name, part, ns in (("tokens", cost1, 1 << 21), ("loops", loops, (1 << 21) // len(loops))):
            ri = torch.arange(len(part), device=dev).repeat_interleave(ns)
            rs = torch.arange(ns, device=dev).repeat(len(part))
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                t0 = time.time()
                for _ in range(2):
                    outs = [L.apply_items(st, b.take(rs), ri, part, lib) for st, b in zip(settings, [x.take(slice(0, ns)) for x in big])]
                    keys = L.keys_of(outs)
                    m = L._fresh(torch.sort(keys[: 1 << 16]).values, keys)
                    prob.goal(outs)
                torch.cuda.synchronize()
                wall = time.time() - t0
            ev = prof.key_averages()
            cuda_us = sum(e.self_device_time_total for e in ev if e.device_type is not None and "cuda" in str(e.device_type).lower())
            launches = sum(e.count for e in ev if e.key.startswith("cudaLaunchKernel"))
            rec = {"what": name, "items": len(part), "rows": len(part) * ns * 2, "wall_s": round(wall, 2),
                   "cuda_kernel_s": round(cuda_us / 1e6, 2), "gpu_busy_share": round(cuda_us / 1e6 / max(wall, 1e-9), 3),
                   "kernel_launches": launches, "rows_per_s": int(len(part) * ns * 2 / max(wall, 1e-9))}
            out.setdefault("breakdown", []).append(rec)
            print("PROFILE breakdown", json.dumps(rec), flush=True)
            print(ev.table(sort_by="self_cpu_time_total", row_limit=12), flush=True)
        del res
    # the cap: memory per stored state (torch's peak over the stored states) and the time the next level takes per
    # stored state (its streamed candidates' seconds)
    deep = [r for r in out["depths"] if r["torch_peak_gb"]]
    if len(deep) >= 2:
        a, b = deep[-2], deep[-1]                         # b streams the level after a's last stored level
        stored = sum(a["levels"])
        per_state_gb = a["torch_peak_gb"] / max(stored, 1)
        mem_total = (b.get("gpu_mem_total_gb") or 80)
        next_s_per_state = max(b["s"] - a["s"], 1e-6) / max(a["levels"][-1], 1)
        out["cap"] = {"bytes_per_stored_state": int(per_state_gb * 2 ** 30), "mem_cap_states": int(0.6 * mem_total / per_state_gb),
                      "next_level_s_per_stored_state": round(next_s_per_state * 1e6, 2),
                      "time_cap_states": int(SEARCH_S / next_s_per_state), "search_s": SEARCH_S}
        out["cap"]["cap"] = min(out["cap"]["mem_cap_states"], out["cap"]["time_cap_states"])
        print("PROFILE cap", json.dumps(out["cap"]), flush=True)
    print("PROFILE_JSON=" + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
