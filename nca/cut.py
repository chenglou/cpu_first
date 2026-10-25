"""The runs that ended without writing a record: each one's last checkpoint peek (the step it reached and its curve, a
point every 500 steps) from the Volume, gathered into records/<which>_cut.json. Reads the Volume from here: no app,
no container.
- `trim` (trim1, trim_cols; NCA.md §9 cites it): run on 2026-10-24, when trim1's w2_wd0.1_s1 had stopped at step
  47,500; that run was then resumed from its checkpoint and finished, so a rerun now finds every run recorded and keeps
  the file.
- `rest` (rest1, §10): run on 2026-10-24, when Modal stopped every container of the workspace at its spend limit, 13
  of rest1's 24 runs short of their 50,000 steps; those were resumed from their checkpoints once the limit was raised
  and finished, so a rerun now finds every run recorded and keeps the file.

    ~/github/sotaku/venv/bin/python -m nca.cut [trim|rest]      (default trim)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

from nca import suites

HERE = Path(__file__).resolve().parent
VERSION = "v1"
CUTS = {"trim": ("trim1", "trim_cols"), "rest": ("rest1",)}


def main(which="trim"):
    vol = modal.Volume.from_name("nca")
    out = {}
    for suite in CUTS[which]:
        done = {e.path.rsplit("/", 1)[-1].removesuffix(".json") for e in vol.listdir(f"{VERSION}/{suite}")}
        local = HERE / "runs" / f"{suite}.jsonl"
        done |= {json.loads(l)["name"] for l in open(local)} if local.exists() else set()
        for job in suites.SUITES[suite]():
            if job["name"] in done:
                continue
            try:
                pk = json.loads(b"".join(vol.read_file(f"{VERSION}/.ckpt/{suite}/{job['name']}.pt.json")))
            except (FileNotFoundError, modal.exception.NotFoundError):                     # never reached a checkpoint
                pk = {"step": 0, "steps": job["steps"], "t": 0, "curve": []}
            out[f"{suite}/{job['name']}"] = dict(pk, tag=job["tag"], cfg={k: job[k] for k in ("B", "w", "k")})
            print(f"{suite}/{job['name']}: step {pk['step']:,} of {pk['steps']:,}", flush=True)
    if not out:
        print(f"every run has a record: records/{which}_cut.json kept as it is", flush=True)
        return
    (HERE / "records" / f"{which}_cut.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
