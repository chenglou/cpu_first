"""H1's step in miniature (TINYROUTING.md §5.1), run locally: every job of suites.composite not yet in
runs/composite_local.jsonl, one after another (each run is seconds to a few minutes and well under 1 GB).

    ~/github/sotaku/venv/bin/python -m tinyrouting.composite
"""

from __future__ import annotations

import json

import torch

from tinyrouting.suites import composite, execute
from tinyrouting.summarize import RUNS


def main():
    torch.set_num_threads(2)
    out = RUNS / "composite_local.jsonl"
    done = {json.loads(l)["name"] for l in open(out)} if out.exists() else set()
    for job in composite():
        if job["name"] in done:
            continue
        r = execute(job)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(job["name"], r["hit"], round(r["exact"], 3), r["true_wiring"], r["secs"], flush=True)


if __name__ == "__main__":
    main()
