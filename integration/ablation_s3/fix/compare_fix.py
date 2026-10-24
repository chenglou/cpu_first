"""Each fixed run (ABLATION_S3.md §8) against the original run it replays: the runner's compare (every field but the
md5 and the log) and the climb's log lines (LR) with every decimal number masked, since only their seconds may differ.
The originals: s3i's official run and H100 seeds 1-7 (integration/records/), sp2's official run and B200 seeds 74 and
1-7 (integration/self_practice/records/runner/).

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/fix/compare_fix.py [RECORDS]
        (RECORDS: the fixed runs' folder, default integration/ablation_s3/records/fix)
"""

import json
import re
import subprocess
import sys
from pathlib import Path

FIX = Path(sys.argv[1] if len(sys.argv) > 1 else "integration/ablation_s3/records/fix")
SP2 = "integration/self_practice/records/runner"
PAIRS = ([("integration/records/s3i_official.json", "submission_stage3_s3i_fix_official.json"),
          (f"{SP2}/submission_stage3_sp2_official.json", "submission_stage3_sp2_fix_official.json")]
         + [(f"integration/records/s3i_h100_seed{s}.json", f"s3i_fix_h100_seed{s}.json") for s in range(1, 8)]
         + [(f"{SP2}/submission_stage3_sp2_b200_seed{s}.json", f"sp2_fix_b200_seed{s}.json")
            for s in (74, 1, 2, 3, 4, 5, 6, 7)])


def climb_log(path):
    return [re.sub(r"\d+\.\d+", "#", l) for l in json.loads(Path(path).read_text())["stdout"].splitlines()
            if l.startswith("LR ")]


for ref, name in PAIRS:
    new = FIX / name
    if not new.exists():
        print(f"{name:42s} missing")
        continue
    out = subprocess.run([sys.executable, "tools/modal_runner.py", "--mode", "compare", ref, str(new)],
                         capture_output=True, text=True).stdout
    fields = [l.split("DIFFERS")[1].strip().split(":")[0] for l in out.splitlines() if "DIFFERS" in l]
    fields = [f for f in fields if f not in ("submission md5", "log without timings")]
    a, b = climb_log(ref), climb_log(new)
    lines = sum(u != v for u, v in zip(a, b)) + abs(len(a) - len(b))
    dp = json.loads(new.read_text())["result"]["depth_profile"]
    print(f"{name:42s} T = {dp['max_certified_time_steps']} / {dp['ood_n_max_certified_time_steps']}; fields "
          f"differing: {', '.join(fields) or 'none'}; climb log lines differing (seconds masked): {lines} of {len(a)}")
