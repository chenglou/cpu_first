"""The rule-8 gradient test of an integration file (general_lessons/local_check.py's `grad`, through
integration/local_check.py): every table and switch at the truth, the final logits' cross-entropy on H1 rows
(depth_t_8 and depth_ood_n_t_4 by default), and which parameter tensors get no gradient. About 12 GB: Modal only.

    modal run integration/checks/modal_checks.py --script gradtest --seeds 0 --args "N=16" --tag grad1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
argv = [a for a in sys.argv[1:] if not a.startswith(("SEED=", "FILE="))]
fname = next((a[5:] for a in sys.argv[1:] if a.startswith("FILE=")), "submission_stage1.py")
os.environ["LC_FILE"] = fname
import local_check  # noqa: E402

data = os.environ.get("H1_DATA")
for sets in ([], [("DESIGNED_SIGMA", '"lowzero"')]):
    print(f"== {fname} {sets or 'below selected'}", flush=True)
    local_check.glc.grad(data, int(next((a[2:] for a in argv if a.startswith("N=")), 16)), [], sets)
