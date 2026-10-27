# The wide file (phase A): pre-registration

Written 2026-10-27 02:00 PDT, before any run on a GPU, the tiny core included. WIDE.md §1 repeats this text
unchanged and records the time and hash it was frozen with.

## The question

The legal file (integration/combined_legal/LEGAL_FILE.md, certified 8 of 8) reads the target's law with a grammar made
for H1: chains of add, mul and mod over x and N, depth 2. The brief: widen it, with subtract, power, constants read
from the data and depth 3, beside add, mul and mod, every value still computed by the learner's own routines; show it
still reads x² mod N's law, how many candidate laws it now weighs, and what the extra time costs; then certify the
file.

## Design (`make_wide.py`, from the legal file's committed bytes at 8e3c7fe, md5 asserted)

- **The grammar** (`LawReader`): left-deep chains over the leaves x and N, depth 1 to 3, of five operations:
  - add: the learner's addition, where the sum fits the places;
  - sub: the difference the learner's comparison leaves, where it says "not below" and the learner's addition gives
    the minuend back;
  - mul: repeated addition by doubling (`repeat_add`, the book's own), where both factors fit 8 places;
  - pow: repeated mul, for a count of 1 to 8;
  - mod: the learner's link by doubling: b, 2b, 4b, ... made by the learner's addition while the comparison says below
    the dividend, then the link with each, largest first (the legal file repeated the link one at a time, at most
    1,000 times).
- **The data's constants**: when no law of x and N is read, a number c by which the reduction mod N of a leaf or a
  one-operation chain misses every answer, either way round (mod N, by the same operations), joins the leaves, and
  the chains are made again.
- **The order**: depth by depth; each depth's candidates computed together, operation by operation, their rows
  stacked; only the first chain of each distinct value (its digits on its rows, and its rows) kept; the first exact on
  every row where it is computed, at least 4 rows, is read.
- **What it poses**: as before, each operation's laws (f and add.f; sub has none in the book, pow is mul's) on the
  patterns whose count is a digit or another number.
- **Room** (the file stays under 256 KiB): code the file never runs with its settings (counts always by doubling; one
  acceptance root shared, kept for a check's own use), the miner's report-only parts (its facts about the answers, and
  divisibility), and the settings' history notes.
- Everything else is the legal file's.

## Runs

1. The tiny core on one B200 (`core_c.py`, combined_fast's), then its answer programs on both ladders, T = 1 to 64,
   seen and unseen moduli (`checks/ladder_core.py`).
2. One official H100 run (`tools/modal_runner.py --mode official`, the manifest untouched), checked by
   `tools/verify_records.py`.
3. If it certifies, seeds 1–7 on H100s.

## Seen before this was written (local, CPU)

- **The file**: `submission_stage3_wide.py`, md5 `3bb5ceeb`, 261,465 bytes (679 under the limit); the contest's
  source check passes.
- **The reading on H1** (`checks/read_wide.py` MODE=read; the legal tiny core at the start of its level 3, 400 answer
  rows of T = 1): `mod(mul(x,x),N)` on all 400 rows (the legal file's reader: 83), 200 candidates weighed (its: 35),
  0.3 s (its: 2.2 s). The link by doubling reduces every row; the repeated link stopped at 1,000 links.
- **Its reach** (MODE=breadth; the same rows, the answers of other formulas made by Python's integers, a test outside
  any run):

  | formula | read | rows | candidates | seconds |
  |---|---|---|---|---|
  | x³ mod N | `mod(mul(mul(x,x),x),N)` | 158 of 400 | 1,140 | 1.0 |
  | x² + 1 mod N | `mod(add(mul(x,x),1),N)`, the constant 1 read | 400 | 3,825 | 3.1 |
  | x² + 3 mod N | `mod(add(mul(x,x),3),N)`, the constant 3 read | 400 | 8,025 | 5.6 |
  | x² − 1 mod N | `mod(sub(mul(x,x),1),N)`, the constant 1 read | 400 | 3,825 | 3.2 |
  | x² + x mod N | `mod(add(mul(x,x),x),N)` | 400 | 1,140 | 1.0 |
  | 10 x² mod N | none (10 is no gap) | | 1,140 | 1.0 |
  | 10 x² + 1 mod N | none (depth 4) | | 1,140 | 1.0 |

- **The pose** (MODE=pose): the six laws in every binding, 96 candidates; 192 of 192 targets equal Python's integers.
- **combined_sp2's local checks** (`checks/logs/local_wide.log`: 358 s, peak 3.8 GiB): the legal file's results part
  for part (`checks/compare_local.py`), but the book's code (92 lines from `book_gen_fix.py`'s: the room edits);
  check_books part 2 the same, 6,204 targets equal the laws' values. A first build cut the acceptance's unshared path,
  which `check_fix_variants` uses; it is kept, made compact.
- **The ladder check** (`checks/ladder_core.py`) on the legal file's own tiny core: its two answer programs 768/768 on
  all 14 rungs, T = 1 to 64, seen and unseen moduli.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the tiny core: the reading | level 3, `mod(mul(x,x),N)`, on all 400 answer rows, about 200 candidates, under 30 s | high |
| the tiny core: the climb | the legal tiny core's levels; the round at level 4; the answer program `C.S[T.round1 STA.X]` | high |
| the tiny core: both ladders | every answer program 768/768 on all 14 rungs | high |
| the official run | certifies: T = 64 on both ladders, every split exact | medium-high |
| the official run: the reading | level 3, on every answer row kept by then, under 30 s | medium-high |
| the official run: training | 400–650 s (the legal file's: 465 s) | medium |
| seeds 1–7 | 7 of 7 certify | medium-high |
| money | about $10–15 | medium |

## Money

Phase A and B together: soft $150, cap $220. `integration/ablation_s3/spend.py 4750` runs before each launch. One
spend watcher for this study's apps, its total the cap less one hour of eight GPUs (about $34): it stops everything at
$185. Every app is stopped when its runs end.
