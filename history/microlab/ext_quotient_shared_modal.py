"""Run ext_quotient_shared on Modal: one CPU container per run (slots x size x seed x holdout x variant).

    modal run --detach microlab/ext_quotient_shared_modal.py --sizes 6,7,10 --seeds 4 --holdout 0,4 --tag S1
    modal run microlab/ext_quotient_shared_modal.py --fetch --tag S1        # copy finished runs from the Volume instead

Each run trains with 4 threads on a 4-CPU container (as ext_quotient_modal.py), writes its row to the Volume
`microlab-quotient-shared` under <tag>/, and the entrypoint appends it to microlab/results_quotient_shared.jsonl
(`--fetch` adds the Volume's rows that are not there yet); the weights after each phase stay on the Volume as
<stem>_a.pt and <stem>_b.pt. `--extra` is one JSON object of run options, or a list of them (one variant each);
`name` only names the Volume files; string options may name {bits}, {seed} and {holdout}, e.g. a `from_a` of
"/out/S4/shared_{bits}_s{seed}_ho{holdout}_hold400_hardb_a.pt" (phase B from those phase-A weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.13").pip_install("torch==2.14.1", "numpy==2.5.4")
for _f in ("__init__.py", "models.py", "tasks.py", "run.py", "ext_ladder.py", "ext_matched.py", "ext_quotient.py",
           "ext_quotient_shared.py"):
    image = image.add_local_file(HERE / _f, f"/root/microlab/{_f}")
vol = modal.Volume.from_name("microlab-quotient-shared", create_if_missing=True)
app = modal.App("microlab-quotient-shared", image=image)
OUT = HERE / "results_quotient_shared.jsonl"
OPTIONS = ("slots", "kr", "kn", "lr", "stage", "final", "switch_steps", "warmup", "start", "staged", "first", "hold", "handover", "lr_b", "hard_borrow",
           "hard_b", "freeze_digits", "from_a")


def _key(r):
    return (r["tag"], r["bits"], r["seed"], r["holdout"], json.dumps({k: r.get(k, 0) for k in OPTIONS}, sort_keys=True))


@app.function(cpu=4, memory=4096, timeout=4 * 3600, volumes={"/out": vol}, retries=modal.Retries(max_retries=1))
def run_one(spec: dict) -> dict:
    import os
    import sys
    sys.path.insert(0, "/root")
    from microlab.ext_quotient_shared import run
    spec = dict(spec)
    name = spec.pop("name", "")
    os.makedirs(f"/out/{spec['tag']}", exist_ok=True)
    stem = f"/out/{spec['tag']}/{spec.get('slots', 'shared')}_{spec['bits']}_s{spec['seed']}_ho{spec.get('holdout', 0)}{name}"
    r = run(log=None, threads=4, save=stem, **spec)
    path = f"{stem}.json"
    with open(path, "w") as f:                       # a retry rewrites its own file
        f.write(json.dumps(r) + "\n")
    vol.commit()
    return r


@app.local_entrypoint()
def main(sizes: str = "6,7,10", seeds: int = 4, seed0: int = 0, holdout: str = "0", tag: str = "", extra: str = "{}",
         fetch: bool = False) -> None:
    import sys
    sys.path.insert(0, str(HERE.parent))
    have = {_key(json.loads(l)) for l in open(OUT)} if OUT.exists() else set()
    if fetch:
        n = 0
        for entry in vol.listdir(tag):
            if not entry.path.endswith(".json"):                              # weights (.pt) stay on the Volume
                continue
            for row in b"".join(vol.read_file(entry.path)).decode().splitlines():
                if row.strip() and _key(json.loads(row)) not in have:
                    with open(OUT, "a") as f:
                        f.write(row + "\n")
                    have.add(_key(json.loads(row)))
                    n += 1
        print(f"fetched {n} new runs from {tag}/ -> {OUT}")
        return
    from microlab.ext_quotient_shared import show
    variants = json.loads(extra)
    fill = lambda v, **f: {k: x.format(**f) if isinstance(x, str) and "{" in x else x for k, x in v.items()}
    specs = [dict(bits=int(s), seed=sd, holdout=int(h), tag=tag, **fill(v, bits=s, seed=sd, holdout=h))
             for v in (variants if isinstance(variants, list) else [variants]) for s in sizes.split(",")
             for h in holdout.split(",") for sd in range(seed0, seed0 + seeds)]
    print(f"{len(specs)} runs", flush=True)
    for r in run_one.map(specs, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
        if isinstance(r, Exception):
            print("run failed:", repr(r)[:300], flush=True)
            continue
        if _key(r) not in have:
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            have.add(_key(r))
        print(show(r), flush=True)
