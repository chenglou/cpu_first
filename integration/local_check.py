"""Local plumbing checks of the integration files: general_lessons/local_check.py (both prediction paths with every
table and switch at the truth; every lesson set's hard checks at the truth; the gradient test) pointed at an
integration file (LC_FILE, default submission_stage1.py). The designed checks take seconds and well under 1 GB; the
gradient test does not (it took 11.6 GB on the general file) and belongs on Modal.

    LC_FILE=submission_stage1.py python local_check.py DATA [N=8] [SPLIT ...] [K=V ...]
    LC_FILE=submission_stage1.py python local_check.py DATA lessons [N=64] [K=V ...]
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("glc", os.path.join(ROOT, "general_lessons", "local_check.py"))
glc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(glc)
glc.FILE = os.path.join(HERE, os.environ.get("LC_FILE", "submission_stage1.py"))
_load = glc.load_module


def load_module(sets=()):
    """The general harness reads REAL_MAX_T in its `lessons` check; the integration files pick their answer rows from
    the data (answer_rows) instead, so the harness gets v1's value for its own row choice only."""
    mod = _load(sets)
    if not hasattr(mod, "REAL_MAX_T"):
        mod.REAL_MAX_T = 2
    return mod


glc.load_module = load_module

if __name__ == "__main__":
    glc.main(sys.argv[1:])
