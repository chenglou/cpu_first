"""Builds integration/learned_loop/submission_stage3_ll.py (LEARNED_LOOP.md): sp2 (stage 3 with self-posed practice)
with its countdown over T learned instead of written, from
  * learned_loop/stage3_ll_template.py: sp2's template (self_practice/stage3_sp_template.py) with the countdown, its
    gate and the loop law taken out, the answer's program searched and run in their place;
  * learned_loop/libcore.py: integration/lib/libcore.py with the count loop, the answer program's machine, and the
    parts sp2 never runs left out (pasted at #@LIBCORE@ without its docstring and imports, as make_sp.py does);
  * learned_loop/spcore.py: self_practice/spcore.py with the answer program's search (likewise, at #@SPCORE@);
  * integration/submission_stage1.py's reader and writer, extracted by name with make_stage3.py's own helpers and its
    one patch (T's direction in phase D), as make_sp.py does.
No certified file is changed: the builder asserts their md5s first and writes only the new file.

    ~/github/sotaku/venv/bin/python integration/learned_loop/make_ll.py [OUT_NAME]
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
INTEG = HERE.parent
sys.path.insert(0, str(INTEG.parent))
from integration import make_stage3 as M3                                       # noqa: E402

TEMPLATE = HERE / "stage3_ll_template.py"
CORE = HERE / "libcore.py"
SPCORE = HERE / "spcore.py"
SP_FILES = {"self_practice/submission_stage3_sp.py": "f0ef5b06", "self_practice/submission_stage3_sp2.py": "afc70fe9"}


def body_of(path: Path, marker: str) -> tuple[str, list[str]]:
    """A module's body from its first section marker on, and its import lines."""
    src = path.read_text()
    head, body = src.split(marker, 1)
    return marker + body, [l for l in head.splitlines() if l.startswith(("import ", "from ")) and "__future__" not in l]


def build() -> str:
    tpl = TEMPLATE.read_text()
    core, core_imports = body_of(CORE, "# ===================================================================================================== programs")
    sp, sp_imports = body_of(SPCORE, "# ===================================================================================================== the law book")
    for line in core_imports + sp_imports:         # the template must make every imported name available
        if line.startswith("from integration."):
            continue                               # libcore's names: pasted above spcore
        names = line.split(" import ")[-1].split(",") if line.startswith("from ") else [line.split()[1]]
        for name in names:
            assert re.search(rf"^(import|from) .*\b{name.strip()}\b", tpl, flags=re.M), f"template lacks {line}"
    out = tpl.replace("#@LIBCORE@", core.rstrip() + "\n").replace("#@SPCORE@", sp.rstrip() + "\n")
    s1 = M3.STAGE1.read_text()
    out = re.sub(r"^#@STAGE1_TOP: (.*?)@$", lambda m: "\n\n".join(M3.top_level(s1, n) for n in m.group(1).split()), out,
                 flags=re.M)
    out = re.sub(r"^#@STAGE1_METHODS: (.*?)@$", lambda m: "\n".join(M3.method(s1, n) for n in m.group(1).split()), out,
                 flags=re.M)
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
    for name, md5 in list(M3.CERTIFIED.items()) + list(SP_FILES.items()):  # the certified files stay as they are
        assert hashlib.md5((INTEG / name).read_bytes()).hexdigest().startswith(md5), name
    out = HERE / (sys.argv[1] if len(sys.argv) > 1 else "submission_stage3_ll.py")
    src = build()
    out.write_text(src)
    print(f"wrote {out} ({len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
          f"{hashlib.md5(src.encode()).hexdigest()})")
    assert len(src.encode()) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    main()
