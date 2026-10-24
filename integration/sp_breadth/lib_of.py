"""A record's library for offline.py (libs/<name>.log): its climb's sleep lines, and the round_found line if any.

    ~/github/sotaku/venv/bin/python integration/sp_breadth/lib_of.py records/u0_rho_h100.json libs/u0_rho_final.log
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
rec = json.loads((HERE / sys.argv[1]).read_text())
keep = [l for l in rec["stdout"].splitlines() if l.startswith("LR round_found") or
        (l.startswith("LR climb") and " sleep: " in l)]
(HERE / sys.argv[2]).write_text("\n".join(keep) + "\n")
print(f"{len(keep)} lines -> {sys.argv[2]}")
