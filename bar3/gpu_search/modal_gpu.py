"""The GPU search on Modal (GPU_SEARCH.md): one function per GPU type, each running a module of this folder in-process
(no child processes) with its arguments, its printout returned and kept on the Volume `bar3-gpu-search` (v1/logs/).
Files the module writes under /out land on the Volume too.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L4 --module bar3.gpu_search.kernel_check
    cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu H100 --module bar3.gpu_search.check --args=--big

Limits: each call has a timeout (--minutes, default 30) and a hard memory limit (1.5 times the 32 GiB requested);
`modal run` without --detach stops the app, and its container, when this client exits.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
image = (modal.Image.debian_slim(python_version="3.13").apt_install("gcc")
         .pip_install("numpy==2.5.4", "numba==0.68.0", "torch==2.14.1"))
for _d in ("library", "library/inventing", "bar3", "bar3/breadth", "bar3/scale", "bar3/scale/subgoal",
           "bar3/scale/heldout", "bar3/scale/newton", "bar3/scale/rowcheck", "bar3/scale/picks", "bar3/gpu_search"):
    for _f in sorted((ROOT / _d).glob("*.py")):
        image = image.add_local_file(_f, f"/root/{_d}/{_f.name}")
for _f in ("bar3/breadth/records/lp_clean_h1_b1.jsonl", "bar3/breadth/records/seq_lp_clean_targets_b1.state.affine.json",
           "bar3/scale/records/reach_h1_base.json", "bar3/scale/records/reach_ten_lib34.json"):
    image = image.add_local_file(ROOT / _f, f"/root/{_f}")
vol = modal.Volume.from_name("bar3-gpu-search", create_if_missing=True)
VERSION = "v1"
app = modal.App("bar3-gpu-search", image=image)
MEM = 32768


def _run(module: str, args: list, tag: str) -> str:
    import contextlib
    import io
    import runpy
    import sys
    import time
    sys.path.insert(0, "/root")
    os.chdir("/root")
    os.makedirs(f"/out/{VERSION}/logs", exist_ok=True)

    class Tee(io.TextIOBase):
        def __init__(self, *fs):
            self.fs = fs

        def write(self, s):
            for f in self.fs:
                f.write(s)
                f.flush()
            return len(s)

    buf = io.StringIO()
    log = open(f"/out/{VERSION}/logs/{tag}.log", "w")
    sys.argv = [module] + list(args)
    t0 = time.time()
    with contextlib.redirect_stdout(Tee(sys.__stdout__, buf, log)):
        try:
            runpy.run_module(module, run_name="__main__")
        except SystemExit as e:
            print(f"[exit {e.code}]")
        except Exception:                      # noqa: BLE001  (the printout says what failed)
            import traceback
            traceback.print_exc(file=sys.stdout)
        print(f"[{module} {' '.join(args)}: {time.time() - t0:.0f}s]")
    log.close()
    vol.commit()
    return buf.getvalue()


KW = dict(cpu=4, memory=(MEM, MEM * 3 // 2), timeout=4 * 3600, volumes={"/out": vol})


@app.function(gpu="T4", **KW)
def run_t4(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="L4", **KW)
def run_l4(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="A10G", **KW)
def run_a10g(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="L40S", **KW)
def run_l40s(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="A100-80GB", **KW)
def run_a100(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="H100", **KW)
def run_h100(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="B200", **KW)
def run_b200(module: str, args: list, tag: str) -> str:
    return _run(module, args, tag)


@app.function(gpu="L40S", cpu=12, memory=(49152, 73728), timeout=4 * 3600, volumes={"/out": vol})
def run_l40s_c12(module: str, args: list, tag: str) -> str:
    """An L40S with 12 cores: a learner run, its vocabularies prepared on a CPU pool (searcher.py)."""
    return _run(module, args, tag)


@app.function(cpu=16, memory=(65536, 98304), timeout=4 * 3600, volumes={"/out": vol})
def run_cpu16(module: str, args: list, tag: str) -> str:
    """The CPU side, on a container like the scale studies' (16 cores, 64 GiB, no GPU)."""
    return _run(module, args, tag)


FNS = {"T4": run_t4, "L4": run_l4, "A10G": run_a10g, "L40S": run_l40s, "A100-80GB": run_a100, "H100": run_h100,
       "B200": run_b200, "CPU16": run_cpu16, "L40S-C12": run_l40s_c12}


@app.local_entrypoint()
def main(gpu: str = "L4", module: str = "bar3.gpu_search.kernel_check", args: str = "", tag: str = "") -> None:
    import time
    tag = tag or f"{module.split('.')[-1]}_{gpu}_{time.strftime('%m%d_%H%M%S')}"
    out = FNS[gpu].remote(module, args.split() if args else [], tag)
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"{tag}.log").write_text(out)
    print(out[-6000:])
