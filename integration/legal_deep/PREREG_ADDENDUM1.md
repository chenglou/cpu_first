# Phase E, addendum 1: six more held-out tasks, seeded by the file's md5

Written 2026-10-27 20:41 PDT, after every run of PREREG.md and before any run of these six. PREREG.md is unchanged.

## Why

The coordinator's approval, queued while phase E went ahead, asked for the held-out seed to come from the fixed
file's md5, not from HEAD. HEAD (cb623f9) was knowable while the design was being made, so a design could in
principle be tuned toward the draw it seeds. A seed taken from the file's own md5 rules that out: any change to the
file changes the draw.

The first six were drawn from HEAD's seed at 19:25:16 PDT (`checks/logs/draw_heldout.log`), six seconds after the
file was fixed at 19:25:10 (`checks/logs/file_fixed.txt`), but their seed was knowable from 17:48:55
(`checks/logs/head_at_start.txt`). They stay reported, separately. **These six are the clean test of generality.** The
write-up reports them on their own, the HEAD six on their own, and the total out of 12.

## The rule and the draw

- **The space:** y ← a·y^k + b·y + c mod N, T times, k in {2, 3}, a in 1..9, b and c in 0..9, less the 14 tasks of it
  used so far: H1, the cube, rho, plus3, plusy, three2, phase D's pair, and the HEAD six. That leaves 1,786 tasks.
- **The seed:** the fixed file's md5, `56acca7358c74c65c72b5f33e3b9aafa`, its first 8 hex digits read as a number:
  int('56acca73', 16) = 1,454,164,595. `checks/draw_heldout2.py` asserts the file's md5 and draws
  `random.Random(seed).sample(space, 6)` (Python 3.9.6), at 20:40:02 PDT (`checks/logs/draw_heldout2.log`).
- **The draw:**

  | task | answer | Horner's form, as the file reads it | four Horner steps (depth 4) |
  |---|---|---|---|
  | `poly_7_2_9_5` | y ← 7y² + 9y + 5 mod N | (7x + 9)·x + 5 | no |
  | `poly_2_3_6_8` | y ← 2y³ + 6y + 8 mod N | ((x·x)·2 + 6)·x + 8 | **yes** |
  | `poly_4_3_4_5` | y ← 4y³ + 4y + 5 mod N | ((x·x)·4 + 4)·x + 5 | **yes** |
  | `poly_2_2_9_3` | y ← 2y² + 9y + 3 mod N | (2x + 9)·x + 3 | no |
  | `poly_7_3_6_4` | y ← 7y³ + 6y + 4 mod N | ((x·x)·7 + 6)·x + 4 | **yes** |
  | `poly_7_3_1_0` | y ← 7y³ + y mod N | ((x·x)·7 + 1)·x | **yes** |

  Data: `legal_wider/tasks_wider.py`, H1's generator and configuration, the same prompts as H1 row for row.

**Depth 4, in each set:** in these six, four (2y³ + 6y + 8, 4y³ + 4y + 5, 7y³ + 6y + 4, 7y³ + y); in the HEAD six, one
(9y³ + 7y; y³ + 5y + 1 takes three steps, (x·x + 5)·x + 1).

## Runs

The file byte for byte (md5 `56acca73`): a tiny core of each (B200, the file's own budget), its answer programs on both
ladders; and one H100 each, seed 74 (`plans/runs2.json`). The tiny cores and the H100 runs are launched together: the
H100 runs are pre-registered whatever the cores show.

## Predictions (written knowing the HEAD six's and the family's results: all learned)

| what | prediction | confidence |
|---|---|---|
| 7y² + 9y + 5, 2y² + 9y + 3 | learned: (a·x + b), then (t·x + c) mod N, as 4y² + 3y + 9 and 5y² + 2y + 6 were | high |
| 2y³ + 6y + 8, 7y³ + y | learned with x kept, as 2y³ + 9y + 9 and 9y³ + 7y were | medium-high |
| 4y³ + 4y + 5 | learned, as 8y³ + 8y + 3 (a = b), whose run needed 1.21e10 of the 1.3e10 budget | medium |
| 7y³ + 6y + 4 | learned | medium |
| the six | 5 or 6 learned | medium |
| the learned runs' training | under 2,000 s each | medium |
| the tiny cores | some depth-4 cores run out of budget where their H100 runs do not (as 8y³ + 8y + 3's did) | medium |

## Changes after PREREG.md's freeze, all outside the file

- `checks/draw_heldout2.py`: new.
- `make_runner.py`: also reads `heldout2.json`, so the runner builds these six's data. `modal_runner_deep.py` was
  written again: md5 `085a2be9`, was `385199bc`.
- `make_plan.py`: `--md5six` writes `plans/runs2.json`. `plans/runs.json` is unchanged.

## Money

Unchanged: soft $400, cap $550. `spend.py 4750` runs before every launch, and the spend watcher covers the new apps.
Phase E so far: about $103.
