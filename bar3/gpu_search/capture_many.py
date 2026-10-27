"""capture.py for many runs at once (GPU_SEARCH.md §9), each in its own worker process: RUNS is a file of lines
"LEARNER CONFIG TARGET TAG STORE[,STORE...]", each run's captured jobs written to OUT_DIR/<LEARNER>_<CONFIG>_<TARGET>_
<TAG>.jsonl.gz (and its round records beside it), as capture.py writes them; a run's replay ends at its first job
the stores lack (capture_run's stored_only), so nothing is searched.

    cd cpu_first && <python with numba> -m bar3.gpu_search.capture_many OUT_DIR RUNS [--workers=N]
"""

from __future__ import annotations

import concurrent.futures as cf
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time


def one(out_dir, spec, config, tname, tag, stores):
    from bar3.gpu_search.capture import capture_run
    out = os.path.join(out_dir, f"{spec}_{config}_{tname}_{tag}.jsonl.gz")
    log = io.StringIO()
    t = time.time()
    try:
        with contextlib.redirect_stdout(log):
            capture_run(spec, config, tname, tag, out, stores.split(","), stored_only=True)
        status = "ok"
    except Exception as e:          # noqa: BLE001  (reported, the other runs go on)
        status = f"failed: {e!r}"
    open(out.replace(".jsonl.gz", ".log"), "w").write(log.getvalue())
    last = log.getvalue().strip().splitlines()[-1:]
    return f"{spec} {config} {tname} {tag}: {status} in {time.time() - t:.0f}s; {last}"


def main():
    out_dir, runs = sys.argv[1], sys.argv[2]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--"))
    os.makedirs(out_dir, exist_ok=True)
    specs = [line.split() for line in open(runs) if line.strip() and not line.startswith("#")]
    from bar3.gpu_search.capture_many import one as run_one     # by module name: run as __main__ (runpy), `one`
    with cf.ProcessPoolExecutor(int(opt.get("workers", os.cpu_count())),     # would not pickle
                                mp_context=mp.get_context("spawn")) as ex:
        for msg in ex.map(run_one, *zip(*[[out_dir] + s for s in specs])):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
