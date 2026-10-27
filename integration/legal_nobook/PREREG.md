# The no-book file (phase B): pre-registration

Written 2026-10-27 03:07 PDT, before any run of this file on a GPU, its tiny cores included, and after phase A's
official run had reported. NOBOOK.md §1 repeats this text unchanged and records the time and hash it was frozen with.

## The question

The wide file (phase A, integration/legal_wide/WIDE.md) still keeps three laws in its book: the comparison, the link and
the addition, on the pair. Its reader needs all three as routines before it can read anything. Round 6
(bar3/scale/reading/READING.md) emptied the book offline by cutting Horner's step down to the link and the
comparison, but its step came from a numpy reading at level 0, which the contest file cannot make. The brief: empty
the book in the phase-A file, every practice answer still from facts, identities or the learner's routines.

## Design (`make_nobook.py`, from the wide file, md5 `3bb5ceeb` asserted)

- **The book is empty** (`BOOK = ((), ())`).
- **The cheapest reading** (`read_fact`), before any routine exists:
  - the rule, which names no task: if every answer row is below one prompt field F, the law ends in a reduction by F,
    so the reduction's step x + d·F is posed, cut to its parts;
  - the check compares the answer and the field as digit strings, from their top place down (`compare`, the file's
    reader's comparison): no number is made;
  - the catalog's reductions are by N, so for F = N the step's parts are posed on the pair: the comparison (lt), the
    link (red), and, since x + d·N is a sum, the addition (add) (`STEP_PARTS`, a designed cut table, as round 6's
    `STEP_CUTS`);
  - the decision and every row it was read from (N:y) go to the climb's record and the log.
- **Their answers** are the legal file's: the comparison's by the order facts of r + q·N forms, the link's by the
  division law's identities (and the fix's probes from the learned addition table), the addition's by its identity and
  the sub.inverse check run by the learner's link and comparison. None is a sum or difference made by Python.
- **The wide reading** follows once the three are routines, unchanged from phase A.
- **Training:** the climb is handed answer rows once 64 are kept (`ANS_MIN`); until the cheapest reading is made, an
  empty pose is posed again at the next unit instead of ending the climb.
- **Room:** branches the file never takes with its settings (`deep_union`'s other mode, the deep probe without a
  focus, `pure_first`'s other mode, `alias_fit` off, the guard for other choosing rules), and shorter docstrings.
- **The arm `STEP_ADD=False`** (a research build, tiny core only): the addition is posed only when the reader asks for
  it, that is once the comparison and the link are routines and it is not.

## Runs

1. Two tiny cores on B200s: `t_nobook` (the file) and `t_nobook_addreader` (`STEP_ADD=False`). Each then has its answer
   programs checked on both ladders, T = 1 to 64, seen and unseen moduli (legal_wide's `checks/ladder_core.py`).
2. One official H100 run of the file, checked by `tools/verify_records.py`.
3. If it certifies, seeds 1–7 on H100s.

## Seen before this was written (local, CPU)

- **The file:** `submission_stage3_nobook.py`, md5 `b42ec304`, 261,893 bytes (251 under the limit); the arm's research
  build `research/submission_stage3_nobook_STEP_ADD-False.py`, md5 `49ff1146`. The contest's source check passes.
- **The first readings** (`checks/fact_local.py`; a fresh climb, an empty book, 400 answer rows of T = 1):
  - level 0, no routine: the cheapest reading reads every answer below N, on all 400 rows (the rows logged), and poses
    the comparison and the link (23 candidates); the addition is not posable yet (its check needs the link and the
    comparison). All 46 draws' answers equal Python's integers;
  - with the comparison and the link as routines (the legal tiny core's): the addition is posed (12 candidates), 24 of
    24 draws right;
  - with the addition too: the wide reading, `mod(mul(x,x),N)` on all 400 rows, 200 laws, 0.3 s, and its six laws,
    192 of 192 draws right.
- **combined_sp2's local checks** (`checks/logs/local_nobook.log`: 305 s, peak 3.1 GiB): the wide file's results part
  for part (`checks/compare_local.py`), but the climb's two parts. The toy climb waits for answer rows until step 27,
  then reads every answer below N on the 9 rows kept (the log shows them) and practises the comparison at level 0. The
  tiny-budget climb never has 8 answer rows in its 12 steps, so it never poses (and spends no work).

## Predictions

| what | prediction | confidence |
|---|---|---|
| t_nobook: level 0 | the cheapest reading: every answer below N, on all 400 rows; the comparison practised | high |
| t_nobook: levels 1–2 | the link at level 1, the addition at level 2 | high |
| t_nobook: level 3 | the wide reading, `mod(mul(x,x),N)` on all 400 rows; the row and the step | high |
| t_nobook: the round | level 4, then the answer program `C.S[T.round1 STA.X]` | high |
| t_nobook: both ladders | every answer program 768/768 on all 14 rungs | high |
| t_nobook_addreader | the same levels and round as t_nobook (the addition is posable only once the link and the comparison are routines, either way) | high |
| the official run: the first pose | waits for 64 answer rows, then reads every answer below N on all of them | high |
| the official run | the same levels; certifies: T = 64 on both ladders, every split exact | medium-high |
| the official run: training | 400–700 s | medium |
| seeds 1–7 | 7 of 7 certify | medium-high |
| money, phase B | about $10–15 | medium |

## Money

Phase A and B together: soft $150, cap $220. `integration/ablation_s3/spend.py 4750` runs before each launch. One
spend watcher for every app of phases A and B, its total the cap less one hour of eight GPUs (about $34): it stops
everything at $185. Every app is stopped when its runs end.
