"""Writes modal_runner_wider.py: the shared runner (tools/modal_runner.py, md5 asserted) with phase C's held-out tasks
(legal_family/tasks_family.py: plusy, three2) and phase D's drawn pair (tasks_wider.py, the names in heldout.json)
built into its image and accepted as tasks, and its paths for this folder (a line that also works at /root in the
container: phase C's first launch failed there). Every edit is asserted to match exactly once.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_wider/make_runner.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "tools" / "modal_runner.py"
OUT = HERE / "modal_runner_wider.py"
WANT = "b1144945206ed6c7d4449c5c949364d4"


def main():
    held = json.loads((HERE / "heldout.json").read_text())["tasks"]
    edits = [
        ('''"""The shared Modal runner (tools/README.md):''',
         '''"""Phase D's copy of the shared runner (integration/legal_wider/make_runner.py writes it from tools/modal_runner.py:
the held-out tasks plusy and three2 of legal_family/tasks_family.py and the pair drawn for phase D, tasks_wider.py,
in its image and its task list, and its paths for this folder).

The shared Modal runner (tools/README.md):'''),
        ('''HERE = Path(__file__).resolve().parent  # cpu_first/tools (in a container: /root, and nothing below uses it)
CPU_FIRST = HERE.parent''',
         '''HERE = Path(__file__).resolve().parent  # cpu_first/integration/legal_wider (in a container: /root; unused there)
CPU_FIRST = HERE.parents[1] if HERE.name == "legal_wider" else HERE.parent'''),
        ('''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1")''',
         f'''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1", "plusy", "three2",
          {", ".join(repr(t) for t in held)})'''),
        ('''    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
)''',
         f'''    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_general data/generated")
    .add_local_file(str(CPU_FIRST / "integration" / "legal_family" / "tasks_family.py"), f"{{REMOTE}}/tasks_family.py",
                    copy=True)
    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_family data/generated plusy three2")
    .add_local_file(str(CPU_FIRST / "integration" / "legal_wider" / "tasks_wider.py"), f"{{REMOTE}}/tasks_wider.py",
                    copy=True)
    .run_commands(f"cd {{REMOTE}} && .venv/bin/python -m tasks_wider data/generated {" ".join(held)}")
)'''),
        ('''"runner": {"tool": "cpu_first/tools/modal_runner.py",''',
         '''"runner": {"tool": "cpu_first/integration/legal_wider/modal_runner_wider.py",'''),
    ]
    src = SRC.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == WANT, hashlib.md5(src.encode()).hexdigest()
    for old, new in edits:
        assert src.count(old) == 1, old[:80]
        src = src.replace(old, new)
    compile(src, OUT.name, "exec")
    OUT.write_text(src)
    print(f"wrote {OUT}: md5 {hashlib.md5(src.encode()).hexdigest()}; held-out tasks {held}")


if __name__ == "__main__":
    main()
