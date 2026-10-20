"""Run unfold_family.py command lists on Modal CPU containers: one container per line, all in parallel.

    modal run unfold_family_modal.py --commands runs/unfold_family/modal_lattice.txt

Each line is `<name> <mode> <args...>`, e.g. `lattice_all lattice --trials 20`. In the container, unfold_family.py gets
`--out <file>`. Locally this writes runs/unfold_family/<name>.log (stdout) and <name>.json (the results file).
Same image layers as unfold_modal.py (cached); only the copied source files differ.
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODULES = ("unfold_family.py", "unfold_lattice.py", "h1.py")
image = (modal.Image.debian_slim(python_version="3.13")
         .pip_install("torch==2.14.1", "numpy==2.5.4")            # same layers as unfold_modal.py (cached)
         .pip_install("sympy", "python-flint")
         .env({"H1_DATA": "/root/h1data", "PYTHONUNBUFFERED": "1"}))
if modal.is_local():
    import h1
    image = image.add_local_dir(h1.DATA, "/root/h1data")
for _f in MODULES:
    image = image.add_local_file(HERE / _f, f"/root/cpu_first/{_f}")
app = modal.App("unfold-family", image=image)


@app.function(cpu=2, memory=4096, timeout=3 * 3600, retries=modal.Retries(max_retries=1))
def run(name: str, argv: list[str]) -> dict:
    import os
    import subprocess
    out = f"/tmp/{name}.json"
    p = subprocess.run(["python", "unfold_family.py", *argv, "--out", out], cwd="/root/cpu_first",
                       capture_output=True, text=True)
    return {"name": name, "code": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-4000:],
            "result": open(out).read() if os.path.exists(out) else ""}


@app.local_entrypoint()
def main(commands: str) -> None:
    jobs = []
    for line in open(commands):
        if line.strip() and not line.startswith("#"):
            name, *argv = line.split()
            jobs.append((name, argv))
    out_dir = HERE / "runs" / "unfold_family"
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in run.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(res, Exception):
            print("failed:", repr(res)[:300], flush=True)
            continue
        (out_dir / f"{res['name']}.log").write_text(res["stdout"] + ("\n" + res["stderr"] if res["code"] else ""))
        if res["result"]:
            (out_dir / f"{res['name']}.json").write_text(res["result"])
        last = [l for l in res["stdout"].splitlines() if l.strip()][-1:] or ["(no output)"]
        print(f"done {res['name']} (exit {res['code']}): {last[0][:200]}", flush=True)
