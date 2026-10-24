"""Builds integration/generic_ops/submission_stage3_g.py (GENERIC_OPS.md): self_practice/submission_stage3_sp2.py on a
generic instruction set, the way self_practice/make_sp.py builds sp2, from
  * generic_ops/stage3_g_template.py: sp2's template (self_practice/stage3_sp_template.py) with its docstring's
    instruction set;
  * generic_ops/libcore_g.py with the instruction set's three constants (isa.isa_source), pasted at #@LIBCORE@;
  * generic_ops/spcore_g.py, at #@SPCORE@;
  * integration/submission_stage1.py's reader and writer, with make_sp.py's two patches (T's direction in phase D).
The inlined core's dead code is dropped (strip_dead: s3i's curriculum, which spcore's climb replaces), so that the file
stays under 256 KiB.
Only instruction sets without SEL.S are built; NAME=VALUE sets a template constant (the SP_* search settings). No certified file is changed: the builder asserts their md5s
first and writes only the new file.

    ~/github/sotaku/venv/bin/python integration/generic_ops/make_g.py [ISA (if,k)] [OUT_NAME (submission_stage3_g.py)] [NAME=VALUE ...]
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
from integration.generic_ops import isa as I                                    # noqa: E402
from integration.self_practice import make_sp as SP                             # noqa: E402

TEMPLATE = HERE / "stage3_g_template.py"
SPCORE = HERE / "spcore_g.py"
KEEP = ("e6bd2a1f",)  # this study's builds in official runs (md5 prefixes: submission_stage3_g3.py): never overwritten


def body_of(src: str, marker: str) -> tuple[str, list[str]]:
    """A module's body from its first section marker on, and its import lines (make_sp.body_of, on text)."""
    head, body = src.split(marker, 1)
    return marker + body, [l for l in head.splitlines() if l.startswith(("import ", "from ")) and "__future__" not in l]


def strip_dead(src: str, start: str, end: str) -> tuple[str, list[str]]:
    """The file without the top-level definitions between the markers start and end (the inlined core) that nothing
    else in the file reads (s3i's curriculum: the climb here is spcore's), dropped until none is left; their names."""
    dropped = []
    while True:
        tree = ast.parse(src)
        lines = src.splitlines(keepends=True)
        lo = next(i for i, l in enumerate(lines) if l.startswith(start)) + 1
        hi = next(i for i, l in enumerate(lines) if l.startswith(end)) + 1
        spans = {}
        for node in tree.body:
            if not lo <= node.lineno <= hi:
                continue
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                spans[node.name] = (node.lineno - len(node.decorator_list), node.end_lineno)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                spans[node.targets[0].id] = (node.lineno, node.end_lineno)
        used = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used.setdefault(node.id, []).append(node.lineno)
        later = {n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else n.targets[0].id for n in tree.body
                 if n.lineno > hi and (isinstance(n, (ast.FunctionDef, ast.ClassDef)) or (
                     isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)))}
        dead = [n for n, (a, b) in spans.items()               # (a name defined again after the core: spcore's own)
                if n in later or not any(not a <= ln <= b for ln in used.get(n, []))]
        if not dead:
            return src, dropped
        cut = sorted(spans[n] for n in dead)
        keep = [l for i, l in enumerate(lines, 1) if not any(a <= i <= b for a, b in cut)]
        src = re.sub(r"\n{4,}", "\n\n\n", "".join(keep))
        dropped += sorted(dead)


DROPPED: list[str] = []      # the names strip_dead dropped in the last build


def build(isa: str = "if,k", sets: dict | None = None) -> str:
    """The file on instruction set isa (never with SEL.S), the template's constants NAME set to sets[NAME] (Python
    source, each NAME's line asserted to occur once)."""
    cond, shift, bring = I.parse_isa(isa)
    assert cond != "sel", "only instruction sets without SEL.S"
    tpl = TEMPLATE.read_text()
    for name, val in (sets or {}).items():
        tpl, n = re.subn(rf"^{name} = [^#\n]*?(?= *#|$)", f"{name} = {val}", tpl, count=1, flags=re.M)
        assert n == 1, name
    core, core_imports = body_of(I.isa_source(isa), "# ===================================================================================================== programs")
    sp, sp_imports = body_of(SPCORE.read_text(), "# ===================================================================================================== the law book")
    for line in core_imports + sp_imports:         # the template must make every imported name available
        if line.startswith("from integration."):
            continue                               # libcore_g's names: pasted above spcore_g
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
    out, dropped = strip_dead(out, "# ===================================================================================================== programs",
                              "# ===================================================================================================== the law book")
    DROPPED[:] = dropped
    ast.parse(out)
    return out


def main() -> None:
    for name, md5 in M3.CERTIFIED.items():                                  # the certified files stay as they are
        assert hashlib.md5((INTEG / name).read_bytes()).hexdigest().startswith(md5), name
    for path, md5 in (("submission_stage3_sp.py", SP.KEEP[0]), ("submission_stage3_sp2.py", "afc70fe9")):
        assert hashlib.md5((INTEG / "self_practice" / path).read_bytes()).hexdigest().startswith(md5), path
    args = [a for a in sys.argv[1:] if "=" not in a]
    sets = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    isa = args[0] if args else "if,k"
    out = HERE / (args[1] if len(args) > 1 else "submission_stage3_g.py")
    if out.exists():                                                        # ... and so are this study's own
        assert not hashlib.md5(out.read_bytes()).hexdigest().startswith(KEEP or ("-",)), f"{out.name} is certified"
    src = build(isa, sets)
    assert len(src.encode()) <= 256 * 1024, f"over the 256 KiB limit: {len(src.encode())} bytes"
    out.write_text(src)
    print(f"wrote {out} (ISA {isa}: {len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
          f"{hashlib.md5(src.encode()).hexdigest()})")
    print(f"dropped from the inlined core (nothing reads them): {', '.join(DROPPED)}")


if __name__ == "__main__":
    main()
