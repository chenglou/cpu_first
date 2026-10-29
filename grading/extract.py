"""Writes chosen Hard uploads from the One Layer Deeper submissions archive to grading/sources/, with their stored
results. sources/ is gitignored: participant files keep their own terms, so none are committed.

The archive is huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions; this reads its data/hard.parquet, whose
sha256 is in the archive's checksums.json. Each source is checked against its row's sha256.

    ~/github/sotaku/venv/bin/python -I grading/extract.py PATH/TO/hard.parquet d83efe07 86e9acf2 ...
"""

import hashlib
import sys
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent

parquet, prefixes = sys.argv[1], sys.argv[2:]
rows = pq.read_table(parquet).to_pylist()
(HERE / "sources").mkdir(exist_ok=True)
for p in prefixes:
    [r] = [r for r in rows if r["id"].startswith(p)]
    src = r["source"].encode("utf-8")
    if hashlib.sha256(src).hexdigest() != r["sha256"]:
        raise SystemExit(f"{p}: the source does not match its sha256")
    (HERE / "sources" / f"{p}.py").write_bytes(src)
    (HERE / "sources" / f"{p}.result.json").write_text(r["result_json"])
    print(f"{p} {r['github_login']} {r['created_at'][:19]} rank {r['leaderboard_rank']} {len(src)} bytes "
          f"sha256 {r['sha256'][:12]}")
