"""The litmus tasks: H1's own data generator with only the answers (the transition) or the prompt's layout (the
tokenizer) changed. Research only, not official: there is no official manifest for any of these.

Every task runs the competition's data.squaring_mod.generate_squaring_mod_dataset with H1's configuration (the H1
command in scripts/generate_datasets.sh: 13/16/20-bit Blum moduli grouped by modulus, T = 4, 8, 16 plus 1,000 rows each
of T = 1 and 2, OOD T = 32, depth rungs T = 1..64 on seen and on unseen 14/18/22/24-bit moduli). The generator draws
(N, x, T) from its seeded rng independently of the transition, so a task with H1's tokenizer has exactly H1's prompts,
row for row; only the answers differ.

    task    answer                       prompt                              in the H1 machine's grammar?
    xfirst  x^(2^T) mod N (H1's)         X x N n T t (N's digits reversed)   yes: the reader picks which run is which
    mont    y <- y^2 10^-8 mod N, T times   H1's                             the datapath, yes: the division menu holds
                                                                             a Montgomery reduction (MENUS.md)
    ten     y <- 10 y^2 mod N, T times   H1's (unseen moduli 14/18/22 bits)  the datapath, yes: a column rule with
                                                                             offset +1 places every product one up
    rho     y <- y^2 + 1 mod N, T times  H1's                                no: nothing adds a constant
    cube    x^(3^T) mod N                H1's                                no: a round squares once
    xy      x y^T mod N                  N n X x Y y T t (Y's marker: 5)     no: the reader takes three fields, the
                                                                             multiply one operand

`ten` keeps the moduli below 10^7 (at most 22 bits) so that 10 y^2 still has a top half below N, which the
grammar's long division needs; its unseen-moduli rungs have 256 prompts at each of 14, 18 and 22 bits (768 a rung).
`xy`'s y is a unit drawn from a rng seeded by (N, x), so the transition and the tokenizer agree on it.

    python -m litmus_tasks OUT_ROOT [TASK ...]          # inside the competition repo (imports its data package)
    python -m litmus_tasks OUT_ROOT --small [TASK ...]  # a few rows per setting (local checks)
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

from data import squaring_mod as sm
from data.reversed_n_squaring_mod import tokenize_reversed_n_squaring_mod_with_result

H1 = dict(
    modulus_bits=[13, 16, 20], time_steps=[4, 8, 16], ood_time_steps=[32], ood_examples_per_setting=3334,
    examples_per_setting=12222, extra_train_examples_by_time_step={1: 1000, 2: 1000},
    depth_evaluation_time_steps=[1, 2, 4, 8, 16, 32, 64], depth_evaluation_examples_per_setting=256,
    ood_n_depth_evaluation_modulus_bits=[14, 18, 22, 24], ood_n_depth_evaluation_examples_per_setting=192,
    train_fraction=0.9090909090909091, test_fraction=0.09090909090909091, split_group="modulus",
    factor_modulus=4, factor_remainder=3, separate_ood_splits=True, seed=45, separate_input_output=True,
)
SMALL = dict(examples_per_setting=44, extra_train_examples_by_time_step={1: 8, 2: 8}, ood_examples_per_setting=6,
             depth_evaluation_examples_per_setting=4, ood_n_depth_evaluation_examples_per_setting=4)
R = 10 ** 8                    # the Montgomery radix: 10^W, W = 8 the H1 machine's width
Y_MARK = sm.TOKEN_IDS["ANS"]   # the fourth field's marker (an id the prompts never use)


def iterate(step):
    """A transition (x, T, p, q) -> the state after T applications of step(y, N)."""
    def transition(x: int, time_steps: int, p: int, q: int) -> int:
        n, y = p * q, x
        for _ in range(time_steps):
            y = step(y, n)
        return y
    return transition


def cube(x: int, time_steps: int, p: int, q: int) -> int:
    """x^(3^T) mod N by the trapdoor, as the generator computes x^(2^T) (x is a unit)."""
    return pow(x, pow(3, time_steps, (p - 1) * (q - 1)), p * q)


def y_of(n: int, x: int) -> int:
    rng = random.Random(n * 1_000_003 + x)
    while True:
        y = rng.randrange(2, n)
        if math.gcd(y, n) == 1:
            return y


def xy(x: int, time_steps: int, p: int, q: int) -> int:
    n = p * q
    return x * pow(y_of(n, x), time_steps, n) % n


def tokenize_xfirst(modulus, x, time_steps, result, *, separate_input_output=False):
    """X x N n T t, N's digits reversed as in H1: H1's tokens with the N and X fields swapped."""
    ids, labels = tokenize_reversed_n_squaring_mod_with_result(modulus, x, time_steps, result,
                                                               separate_input_output=separate_input_output)
    n_at, x_at, t_at = ids.index(sm.TOKEN_IDS["N"]), ids.index(sm.TOKEN_IDS["X"]), ids.index(sm.TOKEN_IDS["T"])
    return ids[x_at:t_at] + ids[n_at:x_at] + ids[t_at:], labels


def tokenize_xy(modulus, x, time_steps, result, *, separate_input_output=False):
    """N n X x Y y T t: H1's tokens with the field Y y inserted before T."""
    ids, labels = tokenize_reversed_n_squaring_mod_with_result(modulus, x, time_steps, result,
                                                               separate_input_output=separate_input_output)
    t_at = ids.index(sm.TOKEN_IDS["T"])
    return ids[:t_at] + [Y_MARK] + sm.number_tokens(y_of(modulus, x)) + ids[t_at:], labels


TASKS = {
    "xfirst": dict(transition=sm.trapdoor_squaring_mod, tokenizer=tokenize_xfirst),
    "mont": dict(transition=iterate(lambda y, n: y * y * pow(R, -1, n) % n)),
    "ten": dict(transition=iterate(lambda y, n: 10 * y * y % n),
                config=dict(ood_n_depth_evaluation_modulus_bits=[14, 18, 22],
                            ood_n_depth_evaluation_examples_per_setting=256)),
    "rho": dict(transition=iterate(lambda y, n: (y * y + 1) % n)),
    "cube": dict(transition=cube),
    "xy": dict(transition=xy, tokenizer=tokenize_xy),
}


def config(task: str, out_root: str, small: bool = False) -> sm.SquaringModGenerationConfig:
    kw = dict(H1, **TASKS[task].get("config", {}))
    if small:
        kw.update(SMALL)
    kw.update(output_dir=str(Path(out_root) / f"litmus_{task}"), generator_family=f"litmus_{task}")
    return sm.SquaringModGenerationConfig(**kw)


def generate(task: str, out_root: str, small: bool = False) -> dict:
    t = TASKS[task]
    return sm.generate_squaring_mod_dataset(config(task, out_root, small), transition=t["transition"],
                                            tokenizer=t.get("tokenizer", tokenize_reversed_n_squaring_mod_with_result))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    for task in a.tasks or list(TASKS):
        cfg = generate(task, a.out_root, a.small)
        print(f"{task}: wrote {cfg['num_examples']} examples, max_seq_len {cfg['max_seq_len']}, "
              f"splits {cfg['split_counts']}", flush=True)


if __name__ == "__main__":
    main()
