"""The tiny cores (tiny_general.py, tiny_table.py) on Modal CPU containers: app `general-lessons-tiny`, at most 20
containers. A plan is a JSON list of {"name", "script", "cfg": {KEY: VALUE}}; each result is written to
general_lessons/tiny/<name>.json.

    cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run \
        ../cpu_first/general_lessons/modal_tiny.py --plan ../cpu_first/general_lessons/plans/tiny1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "tiny"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.1.3", "torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cpu")
    .add_local_file(str(ROOT / "wider_menus" / "wide.py"), "/work/wider_menus/wide.py", copy=True)
    .add_local_file(str(HERE / "tiny_general.py"), "/work/general_lessons/tiny_general.py", copy=True)
    .add_local_file(str(HERE / "tiny_table.py"), "/work/general_lessons/tiny_table.py", copy=True)
    .add_local_file(str(HERE / "local_check.py"), "/work/general_lessons/local_check.py", copy=True)
    .add_local_file(str(HERE / "submission_general.py"), "/work/general_lessons/submission_general.py", copy=True)
    .add_local_dir(str(HERE / "gradtest_data"), "/work/general_lessons/gradtest_data", copy=True)
    .add_local_dir(str(ROOT / "learned_remainder" / "chain_local" / "stub"), "/work/learned_remainder/chain_local/stub",
                   copy=True)
)
app = modal.App("general-lessons-tiny", image=image)


@app.function(cpu=4, memory=4096, timeout=3600, max_containers=2)
def run_one(name: str, script: str, cfg: dict) -> dict:
    import importlib
    import sys
    sys.path.insert(0, "/work/general_lessons")
    mod = importlib.import_module(script)
    full = dict(mod.CFG)
    for k, v in cfg.items():
        full[k] = tuple(v) if isinstance(mod.CFG.get(k), tuple) else v
    full["threads"] = 4
    try:
        res = mod.run(full, verbose=False)
    except Exception as exc:  # keep the others going
        import traceback
        res = {"error": repr(exc), "trace": traceback.format_exc()[-3000:], "cfg": full}
    res["name"] = name
    return res


@app.function(cpu=8, memory=32768, timeout=3600)
def gradtest(sets: list) -> str:
    """local_check.py's gradient test (rule 8) on 8 H1 rows (depth_t_8 and depth_ood_n_t_4), every table and switch at
    the truth, with the constants `sets` (e.g. DESIGNED_SIGMA). Too big for the laptop (11.6 GB once)."""
    import subprocess
    cmd = ["python", "/work/general_lessons/local_check.py", "/work/general_lessons/gradtest_data", "grad", "N=8", *sets]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd="/work/general_lessons")
    return "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "GRADTEST" in l or "Error" in l or "set=" in l)


@app.local_entrypoint()
def main(plan: str = "", grad: bool = False) -> None:
    if grad:
        outs = [gradtest.spawn(s) for s in ([], ['DESIGNED_SIGMA="lowzero"'],
                                           ['DESIGNED_ROUND={"rmul": "ten", "radd": 1, "rcube": "times_x"}'])]
        text = "\n".join(o.get() for o in outs)
        (HERE / "logs" / "gradtest.log").write_text(text + "\n")
        print(text, flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(plan).read_text())
    calls = [(j["name"], run_one.spawn(j["name"], j.get("script", "tiny_general"), j.get("cfg", {}))) for j in jobs]
    for name, c in calls:
        try:
            res = c.get()
        except Exception as exc:
            res = {"name": name, "error": repr(exc)[:2000]}
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
        brief = {k: res.get(k) for k in ("selected", "round", "certified", "cert_per_T", "steps", "error")}
        print(name, json.dumps(brief), flush=True)
