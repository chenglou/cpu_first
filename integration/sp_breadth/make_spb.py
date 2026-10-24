"""Builds this study's stage-3 files (SP_BREADTH.md) as integration/self_practice/make_sp.py builds sp2, from this
folder's copies of its sources:
  * template_b.py: self_practice/stage3_sp_template.py with this study's general changes and settings;
  * libb.py: integration/lib/libcore.py with this study's changes (pasted at #@LIBCORE@ without its docstring and
    imports, as make_stage3.py does);
  * spbcore.py: self_practice/spcore.py with this study's changes (likewise, at #@SPCORE@);
  * integration/submission_stage1.py's reader and writer, extracted by name with make_stage3.py's own helpers and its
    one patch (T's direction in phase D), as make_sp.py does.
No certified file is changed: the builder asserts their md5s first and writes only the new file, in this folder.

    ~/github/sotaku/venv/bin/python integration/sp_breadth/make_spb.py OUT_NAME [CONSTANT=VALUE ...]
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

TEMPLATE = HERE / "template_b.py"
SPCORE = HERE / "spbcore.py"
CORE = HERE / "libb.py"


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
    return strip(out)


def strip(src: str) -> str:
    """The file without the top-level functions and classes nothing in it names (libcore and spcore are pasted whole;
    libcore's curriculum climb and its task builders are dead in a self-practice file) and without the earlier of two
    top-level definitions of one name (the later one is the one every call finds). Names are read from the syntax tree
    (f-strings included), each definition's own body not counting. Repeated until nothing changes."""
    while True:
        tree = ast.parse(src)
        defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        used = [(n.id, n.lineno) for n in ast.walk(tree) if isinstance(n, ast.Name)]
        last = {n.name: n for n in defs}
        dead = [n for n in defs if last[n.name] is not n
                or not any(name == n.name and not n.lineno <= line <= n.end_lineno for name, line in used)]
        if not dead:
            return src
        lines = src.splitlines(keepends=True)
        for n in sorted(dead, key=lambda n: -n.lineno):
            lo = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
            hi = n.end_lineno
            while hi < len(lines) and not lines[hi].strip():
                hi += 1
            del lines[lo:hi]
        src = "".join(lines)


def main() -> None:
    for name, md5 in M3.CERTIFIED.items():                                  # the certified files stay as they are
        assert hashlib.md5((INTEG / name).read_bytes()).hexdigest().startswith(md5), name
    out = HERE / sys.argv[1]
    src = build()
    for kv in sys.argv[2:]:                                                 # a build's settings: CONSTANT=VALUE lines
        name, _, value = kv.partition("=")
        pat = rf"^{name} = .*?(?=\s+#|$)"
        assert len(re.findall(pat, src, flags=re.M)) == 1, name
        src = re.sub(pat, f"{name} = {value}", src, count=1, flags=re.M)
    out.write_text(src)
    print(f"wrote {out} ({len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
          f"{hashlib.md5(src.encode()).hexdigest()})")
    assert len(src.encode()) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    main()
