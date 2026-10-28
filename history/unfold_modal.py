"""Run unfold.py / unfold_lattice.py command lists on Modal CPU containers: one container per line, all in parallel.

    modal run unfold_modal.py --commands runs/unfold/modal_rung.txt

Each line is `<name> <script> <args...>`, e.g. `rung_h1 unfold.py rung --data h1 --seeds 5`. In the container,
unfold.py gets `--threads 4 --out <file>` and unfold_lattice.py gets `--log <file>`. Locally this writes
runs/unfold/<name>.log (stdout) and runs/unfold/<name>.json / .jsonl (the results file).
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODULES = ("unfold.py", "unfold_lattice.py", "relax_common.py", "relax_periodic.py", "h1.py")
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install("torch==2.14.1", "numpy==2.5.4")            # same layer as the microlab images (cached)
         .pip_install("sympy", "python-flint")
         .env({"H1_DATA": "/root/h1data", "PYTHONUNBUFFERED": "1"}))
if modal.is_local():
    import h1
    image = image.add_local_dir(h1.DATA, "/root/h1data")
for _f in MODULES:
    image = image.add_local_file(HERE / _f, f"/root/cpu_first/{_f}")
app = modal.App("unfold", image=image)


@app.function(cpu=4, memory=4096, timeout=3 * 3600, retries=modal.Retries(max_retries=1))
def run(name: str, script: str, argv: list[str]) -> dict:
    import os
    import subprocess
    argv = list(argv)
    out = f"/tmp/{name}.out"
    argv += ["--threads", "4", "--out", out] if script == "unfold.py" else ["--log", out]
    p = subprocess.run(["python", script, *argv], cwd="/root/cpu_first", capture_output=True, text=True)
    return {"name": name, "code": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:],
            "result": open(out).read() if os.path.exists(out) else "",
            "ext": "json" if script == "unfold.py" else "jsonl"}


@app.local_entrypoint()
def main(commands: str) -> None:
    jobs = []
    for line in open(commands):
        if line.strip() and not line.startswith("#"):
            name, script, *argv = line.split()
            jobs.append((name, script, argv))
    out_dir = HERE / "runs" / "unfold"
    for res in run.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("failed:", repr(res)[:300], flush=True)
            continue
        (out_dir / f"{res['name']}.log").write_text(res["stdout"] + ("\n" + res["stderr"] if res["code"] else ""))
        if res["result"]:
            (out_dir / f"{res['name']}.{res['ext']}").write_text(res["result"])
        last = [l for l in res["stdout"].splitlines() if l.strip()][-1:] or ["(no output)"]
        print(f"done {res['name']} (exit {res['code']}): {last[0][:200]}", flush=True)
