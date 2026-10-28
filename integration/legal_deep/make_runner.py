"""Writes modal_runner_deep.py: the shared runner (tools/modal_runner.py, md5 asserted) with phase C's held-out tasks
(legal_family/tasks_family.py: plusy, three2), phase D's pair (legal_wider/heldout.json) and phase E's fresh pair
(heldout.json here), the pairs by legal_wider/tasks_wider.py (any poly_A_K_B_C), built into its image and accepted as
tasks, and its paths for this folder (a line that also works at /root in the container). Every edit is asserted to
match exactly once.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_deep/make_runner.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "tools" / "modal_runner.py"
OUT = HERE / "modal_runner_deep.py"
WANT = "b1144945206ed6c7d4449c5c949364d4"


def main():
    old_pair = json.loads((HERE.parent / "legal_wider" / "heldout.json").read_text())["tasks"]
    fresh = json.loads((HERE / "heldout.json").read_text())["tasks"]
    fresh2 = json.loads((HERE / "heldout2.json").read_text())["tasks"] if (HERE / "heldout2.json").exists() else []
    polys = old_pair + fresh + fresh2  # (heldout2.json: the six drawn by the file's md5, PREREG_ADDENDUM1.md)
    edits = [
        ('''"""The shared Modal runner (tools/README.md):''',
         '''"""Phase E's copy of the shared runner (integration/legal_deep/make_runner.py writes it from tools/modal_runner.py:
plusy and three2 of legal_family/tasks_family.py, phase D's pair and phase E's fresh pair by legal_wider/tasks_wider.py,
in its image and its task list, and its paths for this folder).

The shared Modal runner (tools/README.md):'''),
        ('''HERE = Path(__file__).resolve().parent  # cpu_first/tools (in a container: /root, and nothing below uses it)
CPU_FIRST = HERE.parent''',
         '''HERE = Path(__file__).resolve().parent  # cpu_first/integration/legal_deep (in a container: /root; unused there)
CPU_FIRST = HERE.parents[1] if HERE.name == "legal_deep" else HERE.parent'''),
        ('''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1")''',
         f'''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1", "plusy", "three2",
          {", ".join(repr(t) for t in polys)})'''),
        ('''    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
)''',
         f'''    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_general data/generated")
    .add_local_file(str(CPU_FIRST / "integration" / "legal_family" / "tasks_family.py"), f"{{REMOTE}}/tasks_family.py",
                    copy=True)
    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_family data/generated plusy three2")
    .add_local_file(str(CPU_FIRST / "integration" / "legal_wider" / "tasks_wider.py"), f"{{REMOTE}}/tasks_wider.py",
                    copy=True)
    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_wider data/generated {" ".join(polys)}")
)'''),
        ('''"runner": {"tool": "cpu_first/tools/modal_runner.py",''',
         '''"runner": {"tool": "cpu_first/integration/legal_deep/modal_runner_deep.py",'''),
    ]
    src = SRC.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == WANT, hashlib.md5(src.encode()).hexdigest()
    for old, new in edits:
        assert src.count(old) == 1, old[:80]
        src = src.replace(old, new)
    compile(src, OUT.name, "exec")
    OUT.write_text(src)
    print(f"wrote {OUT}: md5 {hashlib.md5(src.encode()).hexdigest()}; pairs {polys}")


if __name__ == "__main__":
    main()
