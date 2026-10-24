"""libcore_g and spcore_g with a chosen instruction set (GENERIC_OPS.md), for the tests, the tiny core and the builder.

ISA is "COND,SHIFT" or "COND,SHIFT,brg": COND one of sel (SEL.S), if (C blocks), if1 (C blocks of one item in the
searches), ldhc (LDH.C, no conditional); SHIFT one of zero, k; brg puts BRG.X in. "sel,zero,brg" is integration/lib/libcore.py's own; "if,k" is libcore_g.py's
default. The three constant lines of libcore_g.py are rewritten in its text (each asserted to occur once), and both
modules are registered under their own names, so that spcore_g's import finds this libcore_g.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("integration.generic_ops.libcore_g", "integration.generic_ops.spcore_g")


def parse_isa(isa: str) -> tuple[str, str, bool]:
    cond, shift, *rest = isa.split(",")
    assert cond in ("sel", "if", "if1", "ldhc") and shift in ("zero", "k") and rest in ([], ["brg"]), isa
    return cond, shift, rest == ["brg"]


def isa_source(isa: str, src: str | None = None) -> str:
    """libcore_g.py's text (or src) with COND, SHIFT and BRING set to the instruction set's."""
    cond, shift, bring = parse_isa(isa)
    src = (HERE / "libcore_g.py").read_text() if src is None else src
    for name, val in (("COND", repr(cond)), ("SHIFT", repr(shift)), ("BRING", repr(bring))):
        lines = [ln for ln in src.splitlines() if ln.startswith(f"{name} = ")]
        assert len(lines) == 1, name
        cur = lines[0].split("=", 1)[1].split("#", 1)[0].strip()
        src = src.replace(lines[0], lines[0].replace(f"{name} = {cur}", f"{name} = {val}", 1), 1)
    return src


def load(isa: str = "if,k"):
    """(libcore_g, spcore_g) as modules with the instruction set isa."""
    for n in NAMES:
        sys.modules.pop(n, None)
    out = []
    for n, src in zip(NAMES, (isa_source(isa), (HERE / "spcore_g.py").read_text())):
        m = types.ModuleType(n)
        m.__file__ = str(HERE / (n.rsplit(".", 1)[1] + ".py"))
        sys.modules[n] = m
        exec(compile(src, m.__file__, "exec"), m.__dict__)
        out.append(m)
    return tuple(out)
