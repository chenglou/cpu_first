"""Phase D's tasks (integration/legal_wider/PREREG.md): the held-out pair, named poly_A_K_B_C (y <- A y^K + B y + C mod
N, T times), with H1's generator and configuration (the same prompts as H1 row for row, other answers); the others
are tasks_family.py's, which are tasks_general.py's and litmus/tasks.py's.

    python -m tasks_wider OUT_ROOT TASK ...             # inside the competition repo, like litmus/tasks.py
    python -m tasks_wider --small OUT_ROOT TASK ...
"""

from __future__ import annotations

import argparse

import litmus_tasks as LT
import tasks_family as TF
from data import squaring_mod as sm
from data.reversed_n_squaring_mod import tokenize_reversed_n_squaring_mod_with_result

TASKS = dict(TF.TASKS)


def poly(name: str) -> dict:
    """poly_A_K_B_C's transition: y <- A y^K + B y + C mod N, T times."""
    a, k, b, c = (int(v) for v in name.split("_")[1:])
    return dict(transition=LT.iterate(lambda y, n: (a * y ** k + b * y + c) % n))


def generate(task: str, out_root: str, small: bool = False) -> dict:
    if task.startswith("poly_"):
        TASKS[task] = poly(task)
    LT.TASKS = TASKS                                    # litmus's config() reads TASKS for a task's config overrides
    t = TASKS[task]
    return sm.generate_squaring_mod_dataset(LT.config(task, out_root, small), transition=t["transition"],
                                            tokenizer=t.get("tokenizer", tokenize_reversed_n_squaring_mod_with_result))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    for task in a.tasks:
        cfg = generate(task, a.out_root, a.small)
        print(f"{task}: wrote {cfg['num_examples']} examples, max_seq_len {cfg['max_seq_len']}, "
              f"splits {cfg['split_counts']}", flush=True)


if __name__ == "__main__":
    main()
