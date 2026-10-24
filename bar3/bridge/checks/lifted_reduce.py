"""Why the 61-instruction lifted climb (records/climb/lifted_b200_x4m: 4 times s3i's budgets, modx probes with
N < x < 2N) finds every piece but the reduction: is s3i's reduction program right in that machine, and what does it
cost with the library the climb had built? Checks programs with the climb's own acceptor for `reduce`, on H1's rows.

    cd cpu_first && H1_DATA=... PYTHONPATH=. ~/github/sotaku/venv/bin/python bar3/bridge/checks/lifted_reduce.py
"""
import json
from pathlib import Path

from bar3.bridge import climb_b as CB, libcore_b as L
from integration.lib import inclimb

HERE = Path(__file__).resolve().parent.parent
rec = json.loads((HERE / "records" / "climb" / "lifted_b200_x4m.json").read_text())
cells, _ = CB.build_views(CB.SETS["lifted"]["cells"])
L.set_instructions(cells, sel=False)
L.MAC_ADDEND = "A"
L.NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))
CB.strong_modx()
rows = inclimb.digits_of_rows(inclimb.load_rows(1024, seed=0))[:2]
climb = L.Climb(L.Cells(10, "cpu"), "cpu", 16, chunk=1 << 16, **CB.CLIMB)
for k, v in rec["lib"].items():
    climb.lib[k] = L.parse(v)
for t in ("add", "add~", "modB", "modB~", "modx", "modx~", "lt", "lt~"):
    climb.corpus[t] = [L.parse("T." + t)]
print("merges with CLA or CLK in the climb's library:",
      {k: v for k, v in rec["lib"].items() if not k.startswith("T.") and ("CLA" in v or "CLK" in v)})
acc = L.Acceptor("reduce", climb.lib, climb.helpers_for("reduce"), rows, climb.verify, "cpu")
for p in ("CLK CLA P[BRG.X STA.Y T.modB~]", "CLA P[BRG.X STA.Y T.modB~]", "CLK P[BRG.X STA.Y T.modB~]",
          "CLA CLK P[BRG.X STA.Y T.modB~]"):
    print(f"{p:36s} accepted {acc(L.parse(p))}; cost {L.cost(L.parse(p))} as written with this library")
