"""Builds the bridge variant of stage 3 from s3i (integration/submission_stage3_s3i.py, md5 asserted, never written):
s3i with its cell instructions replaced by the ones bar 3's learner invented (BRIDGE.md), every edit a text replacement
asserted to occur exactly once:

  * the machine reads its cells from a list (make_libcore_b.py's edits, which apply to the libcore body s3i inlines
    verbatim, and set_instructions);
  * the instruction set (BRIDGE_CELLS below) is the invented one, set at import: the cells subtract, add and
    multiply-accumulate at the pointer against any of X, Y, N, S (their tables the ones s3i learns: the subtraction
    from the order of the numbers, the addition by sub.inverse, the multiply by mul.zero and mul.ladder, unchanged),
    and the select at the pointer against any of X, Y, N, S, whose table is the compiled counting program the learner
    invented for it (`L.C[MOV.A.R]`: if the flag is set, take the operand's digit). SEL.S (keep or discard on the
    borrow, the whole register at once) is gone; the frame (moves, the pointer, shifts, BRG.X, loops) is s3i's;
  * the prediction path's one-hot machine takes one straight-through table per cell.

    ~/github/sotaku/venv/bin/python bar3/bridge/make_s3b.py            (writes bar3/bridge/submission_stage3_bridge.py)
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from bar3.bridge.make_libcore_b import EDITS as CORE_EDITS, TAIL  # noqa: E402

S3I = HERE.parent.parent / "integration" / "submission_stage3_s3i.py"
S3I_MD5 = "0e8131440e1842546c67e7378caf472d"
OUT = HERE / "submission_stage3_bridge.py"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


edit('''"""Integration, stage 3 (cpu_first/integration/INTEGRATION.md): an H1 model''', '''"""THE BRIDGE VARIANT (cpu_first/bar3/bridge/BRIDGE.md, built by bar3/bridge/make_s3b.py from s3i): s3i with the cell
instructions bar 3's learner invented from a digit law book (subtract, add and multiply-accumulate at the pointer, and a
select at the pointer in place of SEL.S), each against any register; their tables are learned as s3i learns them, the
select's is its invented counting program compiled. s3i's own description follows.

Integration, stage 3 (cpu_first/integration/INTEGRATION.md): an H1 model''')

edit('''if PREREQ:
    NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))''', '''if PREREQ:
    NEEDS.update(mul=("add", "mac"), reduce=("add", "modB"))


def select_table(B=10):
    """The select cell's table [2 B^2, 2] over (a, x, c): the counting program bar 3's learner invented for the select
    law's slice, `L.C[MOV.A.R]` (if the flag is set, take the operand's digit), run on every input of the place."""
    t = torch.zeros(2 * B * B, 2, dtype=torch.long)
    for a in range(B):
        for x in range(B):
            for c in range(2):
                t[(a * B + x) * 2 + c, 0] = x if c else a
    return t


# the instruction set: the invented cells (BRIDGE.md), then the frame
BRIDGE_CELLS = [CellSpec("SUB", "axc", ("A", "c"), "sbb"), CellSpec("ADD", "axc", ("A", "c"), "adc"),
                CellSpec("MAC", "axhk", ("A", "k"), "mul"), CellSpec("SELC", "axc", ("A",), select_table())]
set_instructions(BRIDGE_CELLS, sel=False)''')

edit('''    def ste_cells(self):
        """The prediction path's straight-through tables: the round machine's SBB, ADC [200, 12] and MUL [10^4, 20]
        (rows flat over the inputs), the countdown's subtraction in stage 1's layout [20, 120], the gate [2, 2]."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        SBB = torch.cat([hot(self.a), hot(self.b)], -1).reshape(200, 12)
        ADC = torch.cat([hot(self.aa), hot(self.ab)], -1).reshape(200, 12)
        MUL = torch.cat([hot(self.ml), hot(self.mh)], -1).reshape(10 ** 4, 20)
        TDC = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return SBB, ADC, MUL, TDC, hot(self.g)''', '''    def ste_cells(self):
        """The prediction path's straight-through tables: one per cell of the instruction set, rows flat over what it
        reads (the learned subtraction, addition [200, 12] and multiply [10^4, 20]; the select's compiled table as fixed
        one-hot rows [200, 12]), the countdown's subtraction in stage 1's layout [20, 120], the gate [2, 2]."""
        hot = lambda p: ste(F.one_hot(p.argmax(-1), p.shape[-1]).float(), p.float().softmax(-1))
        learned = {"sbb": torch.cat([hot(self.a), hot(self.b)], -1).reshape(200, 12),
                   "adc": torch.cat([hot(self.aa), hot(self.ab)], -1).reshape(200, 12),
                   "mul": torch.cat([hot(self.ml), hot(self.mh)], -1).reshape(10 ** 4, 20)}
        TABS = {}
        for cs in CELLS:
            if isinstance(cs.table, str):
                TABS[cs.name] = learned[cs.table]
            else:
                t = cs.table.to(self.a.device)
                fw = 2 if "c" in cs.writes or "k" not in cs.writes else 10
                TABS[cs.name] = torch.cat([F.one_hot(t[:, 0], 10), F.one_hot(t[:, 1].clamp(max=fw - 1), fw)], 1).float()
        TDC = torch.cat([hot(self.a), hot(self.b)], -1).permute(0, 2, 1, 3).reshape(20, 120)
        return TABS, TDC, hot(self.g)''')

edit('''    def round_ste(self, s: Tensor, Nr: Tensor, SBB: Tensor, ADC: Tensor, MUL: Tensor) -> Tensor:''',
     '''    def round_ste(self, s: Tensor, Nr: Tensor, TABS: dict) -> Tensor:''')
edit('''        m = SteMachine(SBB, ADC, MUL, PR, damp)''', '''        m = SteMachine(TABS, PR, damp)''')
edit('''        SBB, ADC, MUL, TDC, G = self.ste_cells()''', '''        TABS, TDC, G = self.ste_cells()''')
edit('''                R = self.round_ste(s, Sn, SBB, ADC, MUL)''', '''                R = self.round_ste(s, Sn, TABS)''')


def build():
    src = S3I.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == S3I_MD5, "s3i changed"
    for old, new in CORE_EDITS:
        n = src.count(old)
        if n == 0 and old.startswith('"""LIBRARY CORE'):       # libcore's docstring is not inlined
            continue
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    tail = TAIL.strip("\n")
    anchor = "\n\n# ===================================================================================================== the one-hot machine"
    assert src.count(anchor) == 1
    src = src.replace(anchor, "\n\n\n" + tail + "\n" + anchor)       # set_instructions, before the template's code
    for old, new in EDITS:
        n = src.count(old)
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    ast.parse(src)
    return src


def main():
    src = build()
    OUT.write_text(src)
    b = src.encode()
    print(f"wrote {OUT} ({len(src.splitlines())} lines, {len(b)} bytes, md5 {hashlib.md5(b).hexdigest()})")
    assert len(b) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    main()
