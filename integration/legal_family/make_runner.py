"""Writes modal_runner_family.py: the shared runner (tools/modal_runner.py, md5 asserted) with the two held-out tasks of
phase C (tasks_family.py: plusy and three2, PREREG.md) built into its image and accepted as tasks, and its paths
fixed for this folder. Every edit is asserted to match exactly once; nothing else changes.

    cd cpu_first && ~/github/sotaku/venv/bin/python integration/legal_family/make_runner.py
"""

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "tools" / "modal_runner.py"
OUT = HERE / "modal_runner_family.py"
WANT = "b1144945206ed6c7d4449c5c949364d4"

EDITS = [
    ('''"""The shared Modal runner (tools/README.md):''',
     '''"""Phase C's copy of the shared runner (integration/legal_family/make_runner.py writes it from tools/modal_runner.py:
the held-out tasks plusy and three2 of tasks_family.py in its image and its task list, and its paths for this folder).

The shared Modal runner (tools/README.md):'''),
    ('''HERE = Path(__file__).resolve().parent  # cpu_first/tools (in a container: /root, and nothing below uses it)
CPU_FIRST = HERE.parent''',
     '''HERE = Path(__file__).resolve().parent  # cpu_first/integration/legal_family (in a container: /root; unused there)
CPU_FIRST = HERE.parents[1] if HERE.name == "legal_family" else HERE.parent'''),
    ('''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1")''',
     '''LITMUS = ("xfirst", "mont", "ten", "rho", "cube", "xy", "tenrho", "plus3", "minus1", "plusy", "three2")'''),
    ('''    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
)''',
     '''    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_general data/generated")
    .add_local_file(str(CPU_FIRST / "integration" / "legal_family" / "tasks_family.py"), f"{REMOTE}/tasks_family.py",
                    copy=True)
    .run_commands(f"cd {REMOTE} && .venv/bin/python -m tasks_family data/generated plusy three2")
)'''),
    ('''"runner": {"tool": "cpu_first/tools/modal_runner.py",''',
     '''"runner": {"tool": "cpu_first/integration/legal_family/modal_runner_family.py",'''),
]


def main():
    src = SRC.read_text()
    assert hashlib.md5(src.encode()).hexdigest() == WANT, hashlib.md5(src.encode()).hexdigest()
    for old, new in EDITS:
        assert src.count(old) == 1, old[:80]
        src = src.replace(old, new)
    compile(src, OUT.name, "exec")
    OUT.write_text(src)
    print(f"wrote {OUT}: md5 {hashlib.md5(src.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
