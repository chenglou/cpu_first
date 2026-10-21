"""Every learned part of a combined record against the truth: times table, gate, reading, code, writer.

    python combined/check_tables.py [NAME ...]   (default: every cmbanchors_* record)
"""
import json
import sys
from pathlib import Path

REC = Path(__file__).resolve().parent.parent / "records"


def check(rec):
    lines = rec["stdout"].splitlines()
    get = lambda key: next((l.split("=", 1)[1] for l in lines if l.startswith(key)), "")
    plo, phi = get("TABLE prod_lo="), get("TABLE prod_hi=")
    cells = sum(int(plo[i]) == (i // 10) * (i % 10) % 10 and int(phi[i]) == (i // 10) * (i % 10) // 10
                for i in range(100)) if plo and phi else None
    gate = get("TABLE gate=").split()[0] if get("TABLE gate=") else None
    io = [l for l in lines if l.startswith("IO field=")]
    want = ["IO field=N anchor=2 step=+1", "IO field=x anchor=4 step=-1", "IO field=T anchor=END step=-1",
            "IO field=writer anchor=END step=-1"]
    code = get("IO code=")
    writer = get("IO writer=")
    code_ok = code == ",".join(f"{7 + d}:{d}" for d in range(10))
    writer_ok = writer == ",".join(f"{d}:{7 + d}" for d in range(10))
    return (f"{rec['name']:28s} product {cells}/100 gate {gate} (true 10) fields {io == want} code {code_ok} "
            f"writer {writer_ok}")


names = sys.argv[1:] or sorted(p.stem for p in REC.glob("cmbanchors_*.json"))
for n in names:
    print(check(json.loads((REC / f"{n}.json").read_text())))
