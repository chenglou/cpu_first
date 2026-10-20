# microlab — one small hypothesis, about a minute

A tiny harness for testing ideas about learning arithmetic on small versions of the One Layer Deeper problem,
before (or instead of) building a whole system. Run from `cpu_first/`:

```bash
python -m microlab.run sqmod --size 8 --model xf --steps 1000
```

```bash
python -m microlab.run sqmod --size 8 --model xf --aux pre --steps 1000
```

One line per seed:

```
sqmod@8 xf+pre seed0: exact 0.412  ood@9 0.230  digits(lo->hi) [0.61 0.55 0.93]  70.1s
```

- **exact**: share of fresh examples with every target digit right (leading zeros included).
- **digits(lo->hi)**: per-digit accuracy from the lowest digit up — shows *which* digits a model learned.
- **ood@s**: exact match at a bigger size with the same digit width (e.g. 9-bit moduli after training on 8-bit).
- Every run is appended to `microlab/results.jsonl` (task, size, model, supervision, seed, numbers, `--tag`).

## Tasks (`tasks.py`)
| task | size means | target | optional `--aux` targets |
|---|---|---|---|
| `add` | digits k | a + b | — |
| `mul` | digits k | a × b | — |
| `sq` | digits k | x² | — |
| `mod` | modulus bits b | v mod N for v < N² | `q` = v // N |
| `sqmod` | modulus bits b | x² mod N | `pre` = x² (the value before the remainder), `q` = x² // N |
| `powmod2`, `powmod4` | modulus bits b | x^(2^T) mod N | `s1..s(T−1)` = every intermediate residue |

A b-bit modulus is any odd N in [2^(b−1), 2^b), the same stand-in the relaxed-rules experiments use (balanced Blum
semiprimes barely exist below 13 bits). Add a task by subclassing `Task` and registering it in `TASKS`.

## Models (`models.py`)
- `mlp`: one-hot digits, two hidden layers.
- `xf`: small transformer encoder over digit tokens plus one query token per output digit (`--depth`, `--d`).
- `loop`: one shared transformer block applied `--depth` times.

Add a model with `@model("name")`. It receives the input widths, output width and the requested aux widths, and
returns a dict of logits (`"target"` plus one entry per aux name). Positions are learned per index, so these
baselines are not expected to extrapolate in width.

## Options worth knowing
- `--pool 1000` trains on a fixed set of 1,000 examples (the competition's single-step budget) instead of fresh data.
- `--seeds 3` repeats with different seeds; `--steps`, `--lr`, `--batch` as usual.
- `--device mps` or `--threads N`: this laptop is shared by several experiment sessions, so cap threads (2–4) when
  running in parallel. Use Modal (`~/github/tips-for-running-modal`) for anything big.
- Python: `from microlab.run import run; run("sqmod", 8, model="xf", aux=("pre",), steps=1000)` returns the dicts.

This is exploration tooling, so it uses freely generated data, which the competition's rules would not allow.
