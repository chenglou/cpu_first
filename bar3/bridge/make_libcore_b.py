"""Builds bar3/bridge/libcore_b.py from integration/lib/libcore.py (the library core s3i inlines verbatim; md5
asserted): the same machine, search and climb, with the cell instructions taken from a list of cells instead of s3i's
three (SBB, ADC, MUL). Every edit is a text replacement asserted to occur exactly once, and nothing else changes:

  * CellSpec and CELLS: a cell is a digit instruction at the pointer given by a table (rows over what it reads, in
    order: a = A[p], x = the operand register's digit at p, c the borrow/carry bit, h the multiplier digit, k the
    multiply carry; columns: the digit written to A[p], the flag or carry written to c or k). By default CELLS is s3i's
    own three (their tables on the Cells object), so that with the defaults this module computes what libcore.py does;
  * set_instructions(cells, sel): the instruction set becomes the frame (s3i's moves, pointer, shifts and BRG.X, and
    SEL.S if sel) plus every cell with every operand register it takes; caches are cleared;
  * the machine reads its cells generically: step(), the lockstep machine's opcode rows (_Ops: the row coefficients
    over (a, x, c, h, k) and what each cell writes), the round machine (RoundMachine) and the one-hot machine
    (SteMachine, which takes a dict of straight-through tables).

    ~/github/sotaku/venv/bin/python bar3/bridge/make_libcore_b.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "integration" / "lib" / "libcore.py"
OUT = HERE / "libcore_b.py"
MD5 = "9ed6154fbf34a2511563719c2c0e09a2"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


edit('''"""LIBRARY CORE (INTEGRATION.md, stages 2-3)''', '''"""LIBRARY CORE WITH CELLS FROM A LIST (bar3/bridge/BRIDGE.md; built by bar3/bridge/make_libcore_b.py from
integration/lib/libcore.py, which s3i inlines: the cell instructions come from CELLS, set by set_instructions, instead of
s3i's three; with the defaults it computes what libcore.py does). libcore.py's own description follows.

LIBRARY CORE (INTEGRATION.md, stages 2-3)''')

edit('''LOOP_KINDS = ("P", "B", "W")
ONE = "ONE"''', '''LOOP_KINDS = ("P", "B", "W")
# the frame: s3i's instructions that are not cells (SEL.S is kept apart: set_instructions(..., sel=True) keeps it)
FRAME = ([f"LDA.{r}" for r in "XYNS"] + [f"STA.{r}" for r in "XYS"] + ["CLA", "CLC", "CLK"]
         + [f"LDH.{r}" for r in "XY"] + ["INC"] + [f"SHL.{r}" for r in "AXY"] + [f"SHR.{r}" for r in "AXY"] + ["BRG.X"])
RADIX = {"a": "B", "x": "B", "c": 2, "h": "B", "k": "B"}


class CellSpec:
    """A digit instruction at the pointer. name: the opcode is name.r for each operand register r in regs (name alone
    when it reads no operand); reads: ordered among a (A[p]), x (r[p]), c, h, k, the table's rows running over them in
    that order (mixed radix: B for a digit, 2 for c); writes: among A (A[p]), c, k; table: [rows, 2] (column 0 the
    digit for A[p], column 1 the flag or carry for c or k), or one of "sbb", "adc", "mul" (s3i's tables on Cells)."""

    def __init__(self, name, reads, writes, table, regs="XYNS"):
        self.name, self.reads, self.writes, self.table = name, tuple(reads), tuple(writes), table
        self.regs = regs if "x" in self.reads else ""

    def mults(self, B):
        rad = [B if RADIX[r] == "B" else RADIX[r] for r in self.reads]
        out, m = [], 1
        for q in reversed(rad):
            out.append(m)
            m *= q
        return list(reversed(out)), m

    def opcodes(self):
        return [f"{self.name}.{r}" for r in self.regs] if self.regs else [self.name]

    def __repr__(self):
        return f"{self.name}({','.join(self.reads)} -> {','.join(self.writes)})"


CELLS = [CellSpec("SBB", "axc", ("A", "c"), "sbb"), CellSpec("ADC", "axc", ("A", "c"), "adc"),
         CellSpec("MUL", "axhk", ("A", "k"), "mul", regs="XYN")]
CELL_BY_NAME = {cs.name: cs for cs in CELLS}
CELL_INDEX = {cs.name: i for i, cs in enumerate(CELLS)}
ONE = "ONE"''')

edit('''        self.alu = torch.cat([self.sbb, self.adc, self.mul]).to(torch.uint8)    # the lockstep machine's one lookup''',
     '''        tabs = [self.table_of(cs) for cs in CELLS]
        self.alu = torch.cat(tabs).to(torch.uint8)                     # the machine's one lookup
        self.offs, off = {}, 0                                         # each cell's first row in alu
        for cs, t in zip(CELLS, tabs):
            self.offs[cs.name] = off
            off += t.shape[0]

    def table_of(self, cs):
        """A cell's table [rows, 2] as long on the device (s3i's three from this object, the others their own)."""
        t = getattr(self, cs.table) if isinstance(cs.table, str) else cs.table
        return t.long().to(self.device).reshape(-1, 2)''')

edit('''    if op in ("SBB", "ADC", "MUL"):
        ar = torch.arange(b.S, device=b.regs.device)
        a, x = b.regs[ar, A_, b.p].long(), b.regs[ar, RI[arg], b.p].long()
        if op == "MUL":
            out = cl.mul[((a * B + x) * B + b.h.long()) * B + b.k.long()]
            b.k = out[..., 1].to(torch.uint8)
        else:
            out = (cl.sbb if op == "SBB" else cl.adc)[(a * B + x) * 2 + b.c.long()]
            b.c = out[..., 1].to(torch.uint8)
        b.regs[ar, A_, b.p] = out[..., 0].to(torch.uint8)''', '''    if op in CELL_BY_NAME:
        cs = CELL_BY_NAME[op]
        ar = torch.arange(b.S, device=b.regs.device)
        v = {"a": b.regs[ar, A_, b.p].long(), "x": b.regs[ar, RI[arg], b.p].long() if arg else None,
             "c": b.c.long(), "h": b.h.long(), "k": b.k.long()}
        mult, _ = cs.mults(B)
        row = cl.offs[op] + sum(v[r] * m for r, m in zip(cs.reads, mult))
        out = cl.alu.long()[row]
        if "c" in cs.writes:
            b.c = out[..., 1].to(torch.uint8)
        if "k" in cs.writes:
            b.k = out[..., 1].to(torch.uint8)
        if "A" in cs.writes:
            b.regs[ar, A_, b.p] = out[..., 0].to(torch.uint8)''')

edit('''        fam, ralu, rldh, dep = (torch.zeros(n, dtype=torch.long) for _ in range(4))''',
     '''        fam, ralu, rldh, dep = (torch.zeros(n, dtype=torch.long) for _ in range(4))
        wA, wc, wk = (torch.zeros(n, dtype=torch.bool) for _ in range(3))''')

edit('''            if op in ("SBB", "ADC", "MUL"):
                fam[o], ralu[o] = {"SBB": 1, "ADC": 2, "MUL": 3}[op], r''', '''            if op in CELL_BY_NAME:
                cs = CELL_BY_NAME[op]
                fam[o], ralu[o] = 1 + CELL_INDEX[op], r
                wA[o], wc[o], wk[o] = "A" in cs.writes, "c" in cs.writes, "k" in cs.writes''')

edit('''                                 + [fam > 0, (fam == 1) | (fam == 2), fam == 3], 1).to(device)''',
     '''                                 + [wA, wc, wk], 1).to(device)''')

edit('''            for tok, o in OPCODE.items():
                op = tok.partition(".")[0]
                if op in ("SBB", "ADC"):
                    co[o] = torch.tensor([2 * B, 2, 1, 0, 0, 0 if op == "SBB" else 2 * B * B], dtype=torch.int32)
                elif op == "MUL":
                    co[o] = torch.tensor([B ** 3, B * B, 0, B, 1, 4 * B * B], dtype=torch.int32)''', '''            off, at = 0, {}
            for cs in CELLS:
                at[cs.name] = off
                off += cs.mults(B)[1]
            for tok, o in OPCODE.items():
                op = tok.partition(".")[0]
                if op in CELL_BY_NAME:
                    cs = CELL_BY_NAME[op]
                    mult, _ = cs.mults(B)
                    row = [0] * 6
                    for r, m in zip(cs.reads, mult):
                        row["axchk".index(r)] = m
                    row[5] = at[op]
                    co[o] = torch.tensor(row, dtype=torch.int32)''')

edit('''        self.sbb = torch.zeros(2, B, B, 2, dtype=torch.long, device=device)
        self.adc = torch.zeros_like(self.sbb)
        self.mul = torch.zeros(2, B, B, B, B, dtype=torch.long, device=device)''', '''        self.alu = torch.zeros(2, sum(cs.mults(B)[1] for cs in CELLS), dtype=torch.long, device=device)
        offs, off = {}, 0
        for cs in CELLS:
            offs[cs.name] = off
            off += cs.mults(B)[1]''')

edit('''            elif op in ("SBB", "ADC"):                                      # A[p], c = the cell at (A[p], R[p], c)
                fns.append(lambda tab=(self.sbb if op == "SBB" else self.adc), a=A[p], x=R[p], to=rows_of(p, 5 * P):
                           St.index_copy_(0, to, tab[:, a, x, c]))
            elif op == "MUL":                                               # A[p], k = the cell at (A[p], R[p], h, k)
                fns.append(lambda a=A[p], x=R[p], to=rows_of(p, 5 * P + 2):
                           St.index_copy_(0, to, self.mul[:, a, x, h, k]))''', '''            elif op in CELL_BY_NAME:                                        # the cell's row from what it reads
                cs = CELL_BY_NAME[op]
                mult, _ = cs.mults(B)
                src = {"a": St[p], "x": R[p], "c": c, "h": h, "k": k}
                parts = [(src[q], m) for q, m in zip(cs.reads, mult)]
                dst = ([p] if "A" in cs.writes else []) + ([5 * P] if "c" in cs.writes else []) + \\
                    ([5 * P + 2] if "k" in cs.writes else [])
                cols = rows_of(*(([0] if "A" in cs.writes else []) + ([1] if ("c" in cs.writes or "k" in cs.writes)
                                                                       else [])))   # on the device: no host copy

                def cell(parts=parts, off=offs[op], to=rows_of(*dst), cols=cols):   # inside a CUDA graph's capture
                    row = off + sum(v * m for v, m in parts)
                    St.index_copy_(0, to, self.alu.index_select(0, cols)[:, row])
                fns.append(cell)''')

edit('''        B = self.st.B
        self.sbb.copy_(cells.sbb.T.reshape(2, B, B, 2))
        self.adc.copy_(cells.adc.T.reshape(2, B, B, 2))
        self.mul.copy_(cells.mul.T.reshape(2, B, B, B, B))''', '''        self.alu.copy_(cells.alu.long().T)''')

edit('''    def __init__(self, SBB, ADC, MUL, P, damp, B=10):
        self.SBB, self.ADC, self.MUL, self.P, self.B, self.damp = SBB, ADC, MUL, P, B, damp''',
     '''    def __init__(self, TABS, P, damp, B=10):
        """TABS: cell name -> straight-through table [rows, B + flag width] (the digit, then the flag or carry)."""
        self.TABS, self.P, self.B, self.damp = TABS, P, B, damp''')

edit('''        if op in ("SBB", "ADC"):
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], c)) @ (self.SBB if op == "SBB" else self.ADC))
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[1] = regs, out[:, B:]
        elif op == "MUL":
            out = self.damp(self._joint(regs[0][:, p], (regs[RI[arg]][:, p], h, k)) @ self.MUL)
            regs[0] = put(regs[0], out[:, :B])
            st[0], st[3] = regs, out[:, B:]''', '''        if op in CELL_BY_NAME:
            cs = CELL_BY_NAME[op]
            src = {"a": regs[0][:, p], "x": regs[RI[arg]][:, p] if arg else None, "c": c, "h": h, "k": k}
            ins = [src[q] for q in cs.reads]
            out = self.damp(self._joint(ins[0], ins[1:]) @ self.TABS[op])
            if "A" in cs.writes:
                regs[0] = put(regs[0], out[:, :B])
                st[0] = regs
            if "c" in cs.writes:
                st[1] = out[:, B:]
            if "k" in cs.writes:
                st[3] = out[:, B:]''')

TAIL = '''


# ===================================================================================================== the instruction set
def set_instructions(cells, sel=True):
    """The machine's instructions become the frame (s3i's moves, pointer, shifts and BRG.X; SEL.S too if sel) and the
    cells (CellSpec), each with every operand register it takes; the opcode tables and every cache built on them are
    rebuilt. Call it before any search."""
    global PRIMS, OPCODE, NOP, SAVE, CELLS, CELL_BY_NAME, CELL_INDEX
    CELLS = list(cells)
    CELL_BY_NAME = {cs.name: cs for cs in CELLS}
    CELL_INDEX = {cs.name: i for i, cs in enumerate(CELLS)}
    PRIMS = [o for cs in CELLS for o in cs.opcodes()] + (["SEL.S"] if sel else []) + FRAME
    OPCODE = {t: i for i, t in enumerate(PRIMS)}
    NOP = len(PRIMS)
    SAVE = NOP + 1
    _OPS.clear()
    _TRACES.clear()
    _ROUND.clear()
    return PRIMS
'''


def main():
    src = SRC.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == MD5, "integration/lib/libcore.py changed"
    for old, new in EDITS:
        n = src.count(old)
        assert n == 1, (n, old[:80])
        src = src.replace(old, new)
    OUT.write_text(src.rstrip() + "\n" + TAIL)
    print("wrote", OUT, hashlib.md5(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
