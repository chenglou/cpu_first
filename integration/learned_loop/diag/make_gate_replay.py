"""A diagnostic copy of the learned-loop file (LEARNED_LOOP.md §5, seed 2): submission_stage3_ll.py with sp2's gate put
back as dead weight: the 2 x 2 parameter created where sp2 creates it, its loop-law lesson in cell_lessons, and the gate
in phase B's switch, exactly as sp2's. Nothing reads the gate (the answer program runs as in the file). With every
parameter drawn as in sp2 and every lesson sp2's until the round, the training should replay sp2's step for step up to
the round, so a seed's climb is sp2's own; after the round the learned loop takes over. Not a submission.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/learned_loop/diag/make_gate_replay.py
"""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "submission_stage3_ll.py"
OUT = HERE / "submission_stage3_ll_gate.py"
SRC_MD5 = "419c15e7"

PATCHES = [
    ("""        self.mh = nn.Parameter(INIT * torch.randn(10, 10, 10, 10, 10))     # ... -> high
        self.prog = nn.Parameter(INIT * torch.randn(PROG_SLOTS))           # the answer programs' tournament""",
     """        self.mh = nn.Parameter(INIT * torch.randn(10, 10, 10, 10, 10))     # ... -> high
        self.g = nn.Parameter(INIT * torch.randn(2, 2))                    # sp2's gate: dead weight (diag)
        self.prog = nn.Parameter(INIT * torch.randn(PROG_SLOTS))           # the answer programs' tournament"""),
    ("""        self.adc_ok = self.mul_ok = False""", """        self.adc_ok = self.mul_ok = self.gate_ok = False"""),
    ("""    def cell_lessons(self) -> Tensor:""", """    def cell_lessons(self, Cd: Tensor) -> Tensor:"""),
    ("""        loss = loss + F.cross_entropy(self.ml.float().reshape(-1, 10), lo.flatten()) + \\
            F.cross_entropy(self.mh.float().reshape(-1, 10), hi.flatten())
        return loss""",
     """        loss = loss + F.cross_entropy(self.ml.float().reshape(-1, 10), lo.flatten()) + \\
            F.cross_entropy(self.mh.float().reshape(-1, 10), hi.flatten())
        Gc = self.g.argmax(-1)                                              # sp2's loop law (diag: nothing reads it)
        with torch.no_grad():
            one = F.one_hot(torch.zeros_like(Cd[:, 0]), Cd.shape[1])
            b_run = sub_hard(Cd, one, Td, Tb)[1]                           # a count of at least 1, minus 1
            b_stop = sub_hard(torch.zeros_like(Cd), one, Td, Tb)[1]        # 0 - 1
            live = (Cd != 0).any(1)
        lg = self.g.float().log_softmax(-1)
        loss = loss - lg[b_stop, 0].mean()
        if bool(live.any()):            # (a batch whose counts all read 0, as when T's reading turns, has no run row)
            loss = loss - lg[b_run[live], 1].mean()
        ctl.gate_ok = bool((Gc[b_run[live]] == 1).all() and (Gc[b_stop] == 0).all())
        return loss"""),
    ("""            loss = loss + self.cell_lessons()
            exact = min(exact, float(ctl.adc_ok and ctl.mul_ok))""",
     """            loss = loss + self.cell_lessons(Cd)
            exact = min(exact, float(ctl.adc_ok and ctl.mul_ok and ctl.gate_ok))"""),
    ("""            ok = ctl.cells_on and table == 1.0 and ctl.adc_ok and ctl.mul_ok""",
     """            ok = ctl.cells_on and table == 1.0 and ctl.adc_ok and ctl.mul_ok and ctl.gate_ok"""),
]


def main() -> None:
    src = SRC.read_text()
    assert hashlib.md5(src.encode()).hexdigest().startswith(SRC_MD5), "not the certified learned-loop file"
    for old, new in PATCHES:
        assert src.count(old) == 1, old[:80]
        src = src.replace(old, new)
    ast.parse(src)
    OUT.write_text(src)
    print(f"wrote {OUT} ({len(src.encode())} bytes, md5 {hashlib.md5(src.encode()).hexdigest()})")


if __name__ == "__main__":
    main()
