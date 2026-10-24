"""Loop-body speculation without the fused modular add (ext_loop_csub.py) on a 32-core Modal CPU box.

The search, the check of every survivor on all 1,000 H1 T=1 rows and on fresh rows, and the certification of the best
correct body all run remotely (the H1 data dir is uploaded, as modal_relaxed.py does). The result lands in
runs_loop_csub/modal_<tag>.json and results_loop_csub.jsonl, and in the `loop-csub-runs` Volume.

    ~/github/tips-for-running-modal/venv/bin/python -m modal run microlab/ext_loop_csub_modal.py --length 5
    ~/github/tips-for-running-modal/venv/bin/python -m modal run microlab/ext_loop_csub_modal.py --length 5 --brute
    ~/github/tips-for-running-modal/venv/bin/python -m modal run --detach microlab/ext_loop_csub_modal.py --length 6 \
        --no-exact --chunk 512 --spawn
--brute merges nothing (every body runs the loop on its own); --scan R also runs every function on the first R labelled
rows, for survivor curves over random label orders. Long runs: --spawn with `modal run --detach`; progress
goes to /runs/<tag>.progress and the result to /runs/<tag>.json on the Volume
(`modal volume get loop-csub-runs <tag>.json microlab/runs_loop_csub/`).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
H1_LOCAL = os.environ.get(
    "H1_DATA",
    str(HERE.parent.parent / "one-layer-deeper" / "data" / "generated" / "squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2"),
)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.0.2")
    .env({"H1_DATA": "/root/h1data", "OMP_NUM_THREADS": "1"})
    .add_local_dir(H1_LOCAL, "/root/h1data")
    .add_local_file(str(HERE.parent / "h1.py"), "/root/cpu_first/h1.py")
    .add_local_dir(str(HERE), "/root/cpu_first/microlab",
                   ignore=["__pycache__", "runs_exact", "runs_loop_csub", "*.jsonl", "*.md"])
)
app = modal.App("loop-csub-spec", image=image)
runs_vol = modal.Volume.from_name("loop-csub-runs", create_if_missing=True)


@app.function(cpu=32, memory=98304, timeout=8 * 3600, volumes={"/runs": runs_vol},
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def spec(L: int, K: int, K_lab: int, workers: int, chunk: int, sub: int, exact: bool, job_lo: int, job_hi: int,
         probe_seed: int, tag: str, brute: bool, serial_depth: int, grow_chunk: int, scan: int) -> dict:
    import sys
    sys.path.insert(0, "/root/cpu_first")
    import numpy as np
    from microlab import ext_loop_csub as C
    t0 = time.time()
    rows = C.h1_label_rows(0)
    labels = (rows["N"][:K_lab], rows["x"][:K_lab], rows["y"][:K_lab])
    sl = (job_lo, job_hi) if job_hi > 0 else None
    scan_rows = (rows["N"][:scan], rows["x"][:scan], rows["y"][:scan]) if scan else None
    print(f"[modal cpu=32] L={L} raw bodies {C.body_space(L):,} (2^{np.log2(C.body_space(L)):.1f}), K={K}, "
          f"workers={workers}, chunk={chunk}, job slice {sl}; first label N={rows['N'][0]} x={rows['x'][0]}", flush=True)
    last = [0.0]
    open(f"/runs/{tag}.progress", "w").close()

    def progress(line):
        with open(f"/runs/{tag}.progress", "a") as fh:
            fh.write(line + "\n")
        if time.time() - last[0] > 120:
            runs_vol.commit()
            last[0] = time.time()

    res = C.speculate(L, labels, K=K, probe_seed=probe_seed, chunk=chunk, sub=sub, workers=workers,
                      exact_count=exact, job_slice=sl, merge=not brute, serial_depth=serial_depth,
                      grow_chunk=grow_chunk, progress=progress, scan=scan_rows)
    res = C.report(res, rows, K_lab, certify=sl is None)
    rec = C._jsonable({"exp": "loop_csub", "where": "modal cpu=32", "tag": tag, "workers": workers, "chunk": chunk,
                       "brute": brute, "serial_depth": serial_depth, "grow_chunk": grow_chunk, "scan": scan,
                       "probe_seed": probe_seed, "label_seed": 0, "K_lab": K_lab, "job_slice": sl,
                       "first_labels": [[int(rows["N"][k]), int(rows["x"][k]), int(rows["y"][k])] for k in range(K_lab)],
                       "wall": time.time() - t0, **res})
    with open(f"/runs/{tag}.json", "w") as fh:
        json.dump(rec, fh, indent=1)
    runs_vol.commit()
    return rec


@app.local_entrypoint()
def main(length: int = 5, k: int = 32, k_lab: int = 8, workers: int = 31, chunk: int = 1024, sub: int = 4096,
         exact: bool = True, job_lo: int = 0, job_hi: int = -1, probe_seed: int = 1, tag: str = "",
         spawn: bool = False, brute: bool = False, serial_depth: int = 3, grow_chunk: int = 1024,
         scan: int = 0) -> None:
    tag = tag or f"L{length}{'_brute' if brute else ''}{f'_scan{scan}' if scan else ''}"
    args = (length, k, k_lab, workers, chunk, sub, exact, job_lo, job_hi, probe_seed, tag, brute, serial_depth,
            grow_chunk, scan)
    if spawn:
        call = spec.spawn(*args)
        print(f"spawned {tag}: {call.object_id} -> Volume loop-csub-runs: {tag}.json")
        return
    r = spec.remote(*args)
    (HERE / "runs_loop_csub").mkdir(exist_ok=True)
    (HERE / "runs_loop_csub" / f"modal_{tag}.json").write_text(json.dumps(r, indent=1))
    with open(HERE / "results_loop_csub.jsonl", "a") as fh:
        fh.write(json.dumps({k_: v for k_, v in r.items() if k_ != "checked"}) + "\n")
    keep = ("L", "merge", "space", "levels", "raw", "bodies", "loop_runs", "functions_exact", "functions_hll",
            "functions_chunk_local", "curve_raw", "curve_fn", "seconds", "wall", "t_enum_cpu", "t_loop_cpu", "jobs",
            "jobs_total", "survivor_table", "raw_table", "certified", "order_stats")
    print(json.dumps({k_: r.get(k_) for k_ in keep}, indent=1))
