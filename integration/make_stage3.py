"""Builds integration/submission_stage3.py (INTEGRATION.md, stage 3) from three sources, so that its provenance is
exact:
  * integration/stage3_template.py: the new model (cells by laws, the climb's schedule, the round's tournament, the
    prediction path) with markers;
  * integration/lib/libcore.py: the library core (programs, BPE, the machine, the batched search, the practice tasks,
    the climb, the round search, the one-hot machine), pasted at #@LIBCORE@ without its docstring and imports;
  * integration/submission_stage1.py: the reader and writer and their helpers, extracted by name, line for line, at
    #@STAGE1_TOP: ...@ (top-level functions and classes) and #@STAGE1_METHODS: ...@ (methods), with one patch: the
    reader's label lessons start T's direction in phase D (stage 3's answer phase) instead of C.

    ~/github/sotaku/venv/bin/python integration/make_stage3.py
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "stage3_template.py"
CORE = HERE / "lib" / "libcore.py"
STAGE1 = HERE / "submission_stage1.py"
OUT = HERE / "submission_stage3.py"


def top_level(src: str, name: str) -> str:
    m = re.search(rf"^(def|class) {name}\b", src, flags=re.M)
    if not m:
        raise KeyError(name)
    nxt = re.search(r"^(def |class |[A-Z_]+ = |@|# -{5})", src[m.end():], flags=re.M)
    return src[m.start():m.end() + (nxt.start() if nxt else len(src) - m.end())].rstrip() + "\n"


def method(src: str, name: str) -> str:
    m = re.search(rf"^    def {name}\(", src, flags=re.M)
    if not m:
        raise KeyError(name)
    start = m.start()
    while True:                                                             # decorators directly above
        prev = src.rfind("\n", 0, start - 1)
        line = src[prev + 1:start - 1]
        if line.startswith("    @"):
            start = prev + 1
        else:
            break
    nxt = re.search(r"^    (def |@)|^(def |class )|^    # -{5}", src[m.end():], flags=re.M)
    return src[start:m.end() + (nxt.start() if nxt else len(src) - m.end())].rstrip() + "\n"


def build() -> str:
    tpl, core, s1 = TEMPLATE.read_text(), CORE.read_text(), STAGE1.read_text()
    body = core[core.index("# ===================================================================================================== programs"):]
    out = tpl.replace("#@LIBCORE@", body.rstrip() + "\n")

    def sub_top(m):
        return "\n\n".join(top_level(s1, n) for n in m.group(1).split())

    def sub_methods(m):
        return "\n".join(method(s1, n) for n in m.group(1).split())
    out = re.sub(r"^#@STAGE1_TOP: (.*?)@$", sub_top, out, flags=re.M)
    out = re.sub(r"^#@STAGE1_METHODS: (.*?)@$", sub_methods, out, flags=re.M)
    # the reader's label lessons: T's direction in the answer phase (D), as stage 1 did in its phase C
    old = '        if ctl.io == 4 and ctl.phase == "C" and ctl.streak >= T_AFTER:'
    assert out.count(old) == 1, old
    out = out.replace(old, '        if ctl.io == 4 and ctl.phase == "D" and ctl.ans_ok == 1.0 and ctl.streak >= T_AFTER:')
    old = '            if not ctl.io_exact and ctl.phase == "C":'
    assert out.count(old) == 1, old
    out = out.replace(old, '            if not ctl.io_exact and ctl.phase == "D":')
    assert "#@" not in out, "a marker is left"
    ast.parse(out)
    return out


def main() -> None:
    src = build()
    OUT.write_text(src)
    print(f"wrote {OUT} ({len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
          f"{hashlib.md5(src.encode()).hexdigest()})")
    assert len(src.encode()) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    main()
