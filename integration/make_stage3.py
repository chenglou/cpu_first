"""Builds integration/submission_stage3_s3i.py (INTEGRATION.md, stage 3) from three sources, so that its provenance is
exact (the certified s3g file, submission_stage3.py, and s3h's, submission_stage3_s3h.py, from the same three as they
were, kept in work/s3g/ and work/s3h/):
  * integration/stage3_template.py: the new model (cells by laws, the climb's schedule, the round's tournament, the
    prediction path) with markers;
  * integration/lib/libcore.py: the library core (programs, BPE, the machine, the batched search, the practice tasks,
    the climb, the round search, the one-hot machine), pasted at #@LIBCORE@ without its docstring and imports;
  * integration/submission_stage1.py: the reader and writer and their helpers, extracted by name, line for line, at
    #@STAGE1_TOP: ...@ (top-level functions and classes) and #@STAGE1_METHODS: ...@ (methods), with one patch: the
    reader's label lessons start T's direction in phase D (stage 3's answer phase) instead of C.

    ~/github/sotaku/venv/bin/python integration/make_stage3.py          (s3i: submission_stage3_s3i.py)
    ~/github/sotaku/venv/bin/python integration/make_stage3.py s3g      (checks the certified s3g file, md5 70bac83c,
                                                                         against its sources kept in work/s3g/)
    ~/github/sotaku/venv/bin/python integration/make_stage3.py s3h      (likewise s3h's, md5 3b9d9bcf, work/s3h/)
"""

from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "stage3_template.py"
CORE = HERE / "lib" / "libcore.py"
STAGE1 = HERE / "submission_stage1.py"
OUT = HERE / "submission_stage3_s3i.py"
KEPT = {"s3g": (HERE / "work" / "s3g" / "stage3_template.py", HERE / "work" / "s3g" / "libcore.py",
                HERE / "submission_stage3.py", "70bac83cca41109cde14a1ad2ac4ff4b"),
        "s3h": (HERE / "work" / "s3h" / "stage3_template.py", HERE / "work" / "s3h" / "libcore.py",
                HERE / "submission_stage3_s3h.py", "3b9d9bcfe183fb8b4df4bb3592c2895e")}
CERTIFIED = {"submission_stage1.py": "cfd7b61d", "submission_stage3.py": "70bac83c", "submission_stage3_s3f.py": "31c9cab2",
             "submission_stage3_s3d.py": "373ceb4e", "submission_stage3_s3h.py": "3b9d9bcf",
             "submission_stage3_s3i.py": "0e813144"}


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


def build(template: Path = TEMPLATE, core_file: Path = CORE) -> str:
    tpl, core, s1 = template.read_text(), core_file.read_text(), STAGE1.read_text()
    head, body = core.split("# ===================================================================================================== programs", 1)
    body = "# ===================================================================================================== programs" + body
    for line in head.splitlines():                 # libcore's imports: the template must make every name available
        if line.startswith(("import ", "from ")) and "__future__" not in line:
            names = line.split(" import ")[-1].split(",") if line.startswith("from ") else [line.split()[1]]
            for name in names:
                assert re.search(rf"^(import|from) .*\b{name.strip()}\b", tpl, flags=re.M), f"template lacks {line}"
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
    for name, md5 in CERTIFIED.items():                                     # the certified files stay as they are
        assert hashlib.md5((HERE / name).read_bytes()).hexdigest().startswith(md5), name
    if sys.argv[1:2] and sys.argv[1] in KEPT:
        tpl, core, out, md5 = KEPT[sys.argv[1]]
        src = build(tpl, core)
        assert hashlib.md5(src.encode()).hexdigest() == md5 and out.read_text() == src
        print(f"{out.name} (md5 {md5[:8]}) is built from {tpl.relative_to(HERE)} and {core.relative_to(HERE)}")
        return
    src = build()
    md5 = hashlib.md5(src.encode()).hexdigest()
    if OUT.name in CERTIFIED and not md5.startswith(CERTIFIED[OUT.name]):    # a certified file is only ever rebuilt
        raise SystemExit(f"{OUT.name} is certified (md5 {CERTIFIED[OUT.name]}); the sources now build {md5[:8]}: "
                         f"set OUT to a new name")
    OUT.write_text(src)
    print(f"wrote {OUT} ({len(src.splitlines())} lines, {len(src.encode())} bytes, md5 "
          f"{hashlib.md5(src.encode()).hexdigest()})")
    assert len(src.encode()) <= 256 * 1024, "over the 256 KiB limit"


if __name__ == "__main__":
    main()
