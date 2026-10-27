"""The family's tasks for phase C (integration/legal_family/PREREG.md): general_lessons/tasks_general.py's tasks (and
litmus/tasks.py's through it), reused as they are, plus two held out, chosen and frozen in the pre-registration before
any run: H1's generator and configuration, the same prompts as H1 row for row, other answers.

    task    answer
    plusy   y <- y^2 + y mod N, T times
    three2  y <- 3 y^2 + 2 mod N, T times

    python -m tasks_family OUT_ROOT [TASK ...]           # inside the competition repo, like litmus/tasks.py
    python -m tasks_family OUT_ROOT --small [TASK ...]
"""

from __future__ import annotations

import argparse

import litmus_tasks as LT
import tasks_general as TG
from data import squaring_mod as sm
from data.reversed_n_squaring_mod import tokenize_reversed_n_squaring_mod_with_result

TASKS = dict(TG.TASKS)
TASKS.update({
    "plusy": dict(transition=LT.iterate(lambda y, n: (y * y + y) % n)),
    "three2": dict(transition=LT.iterate(lambda y, n: (3 * y * y + 2) % n)),
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
    for task in a.tasks or ["plusy", "three2"]:
        cfg = generate(task, a.out_root, a.small)
        print(f"{task}: wrote {cfg['num_examples']} examples, max_seq_len {cfg['max_seq_len']}, "
              f"splits {cfg['split_counts']}", flush=True)


if __name__ == "__main__":
    main()
