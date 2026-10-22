"""The general-lessons tasks: litmus/tasks.py's tasks (H1's generator with another answer or layout), reused as they
are, plus three held out from every design decision of general_lessons/ (LESSONS.md section 4): the same prompts as
H1 row for row, other answers.

    task     answer                        in the shared round menu?
    tenrho   y <- 10 y^2 + 1 mod N, T times   yes, by composition: x 10 and + 1 together (moduli up to 22 bits, as ten)
    plus3    y <- y^2 + 3 mod N, T times      yes: another one-digit constant
    minus1   y <- y^2 - 1 mod N, T times      no: N - 1 is not a one-digit constant (the menu's boundary)

    python -m tasks_general OUT_ROOT [TASK ...]           # inside the competition repo, like litmus/tasks.py
    python -m tasks_general OUT_ROOT --small [TASK ...]
"""

from __future__ import annotations

import argparse

import litmus_tasks as LT
from data import squaring_mod as sm
from data.reversed_n_squaring_mod import tokenize_reversed_n_squaring_mod_with_result

NARROW = dict(ood_n_depth_evaluation_modulus_bits=[14, 18, 22], ood_n_depth_evaluation_examples_per_setting=256)
TASKS = dict(LT.TASKS)
TASKS.update({
    "tenrho": dict(transition=LT.iterate(lambda y, n: (10 * y * y + 1) % n), config=NARROW),
    "plus3": dict(transition=LT.iterate(lambda y, n: (y * y + 3) % n)),
    "minus1": dict(transition=LT.iterate(lambda y, n: (y * y - 1) % n)),
})


def generate(task: str, out_root: str, small: bool = False) -> dict:
    LT.TASKS = TASKS                                    # litmus's config() reads TASKS for a task's config overrides
    t = TASKS[task]
    return sm.generate_squaring_mod_dataset(LT.config(task, out_root, small), transition=t["transition"],
                                            tokenizer=t.get("tokenizer", tokenize_reversed_n_squaring_mod_with_result))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    for task in a.tasks or ["tenrho", "plus3", "minus1"]:
        cfg = generate(task, a.out_root, a.small)
        print(f"{task}: wrote {cfg['num_examples']} examples, max_seq_len {cfg['max_seq_len']}, "
              f"splits {cfg['split_counts']}", flush=True)


if __name__ == "__main__":
    main()
