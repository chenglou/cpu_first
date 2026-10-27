# The family file (phase C): pre-registration

Written 2026-10-27 11:20 PDT, before any run of this file on a GPU, its tiny cores included. FAMILY_FILE.md §1 repeats
this text unchanged and records the time and hash it was frozen with.

## The question

The no-book file (phase B, integration/legal_nobook/NOBOOK.md) reads H1's law from the answers and certifies on H1.
Round 7 (bar3/scale/family/FAMILY.md) found, offline and with round 6's numpy reading, four general pieces that bring
SP_BREADTH's family: the reading's windows as laws of their own (SUBTERMS), its constants (round 6's CONST), the 1 of
the reader's pred and succ (CONST1), and a round kept only if it also gives the data's answers at T = 2, 4 and 8
(ROUNDT). The brief: port the four into the no-book file legally, free the bytes without changing behaviour, recertify
H1, and run the family at full size with the same file byte for byte, plus two held-out tasks chosen here.

## Design (`make_family.py`, from the no-book file's committed bytes at f7dc5f8, md5 `b42ec304` asserted)

- **Windows** (`windows`, `read_more`): every connected run of 2 or 3 of the reading's operations, each child off the
  run cut off as a hole (at most 2, named a and b in order), N and the constants kept, each shape once, smallest
  first. Each is posed as a law of its own on `w1` (x, N) or `w2` (x, o, N) by its holes, at double width, its last
  operand bound to N (3 or 6 bindings), named by its operations and constants (`w.mod.mul`, `w.mod.add1.mul`).
- **Constants** (`read_more`): each constant c of the reading, made from N (`const.c` on `n`), and the reading's
  operations with c as an operand among add, sub and mul (`add.c`, `sub.c`, `mul.c` on `x,n`): round 6's CONST, with
  sub added.
- **CONST1**: round 7's reader wrote x − 1 as pred(x), a 1 with no leaf, so CONST1 made that 1 and posed pred. The
  legal reader writes it as the leaf 1 (a gap constant, phase A), so the constants above already pose `const.1` and
  `sub.1`: the reader's own 1 and its own subtraction. There is no separate CONST1 code.
- **Their answers** (`read_target`): the window's tree computed by the law reader's own operations, which are the
  learner's routines (the addition, the comparison and the link) at the draw's size: add by the addition, sub by the
  comparison's difference confirmed by the addition, mul by repeated addition by doubling, mod by the link by
  doubling. A row where a value on the way is not computed is masked. A constant's digits are the reading's text. No
  answer is made by Python's arithmetic.
- **ROUNDT** (`round_t`, `keep_counts`): a round must also give the answers of every count T > 1 the batches showed,
  run T times at full size, on the newest 4,096 rows of each count. H1's training data has T = 1 (1,000 rows), 2
  (1,000), 4, 8 and 16 (33,333 each); `keep_counts` keeps the counts that read the same both ways (one digit, or a
  palindrome), so 1, 2, 4 and 8, and ROUNDT uses 2, 4 and 8, round 7's. The family's data is made with H1's
  configuration (litmus/tasks.py), so the same. ROUNDT sits in the round search's acceptance next to the full-size
  check of the T = 1 answers: a cost level whose programs all fail it counts as aliases and the search goes on to the
  next cost (round 7 filtered the found rounds after the search).
  - Why 4,096 rows, not the answer program's 64: round 7's wrong minus1 rounds fail only where a round's input is 0.
    Of H1's prompts with minus1's answers, 24 of 33,333 rows of T = 4 and 61 of T = 8 pass through 0
    (`checks/roundt_power.py`). 4,096 random rows hold one with probability 0.96 (T = 4) and 0.9997 (T = 8); 64 rows,
    0.05 and 0.11.
- **Room**, with no change in behaviour: phase B's `STEP_ADD` switch (always True here); 32 docstrings and one comment
  that a design doc already explains, cut to what they say plus a pointer to the doc; the rules between a class's
  methods shortened (combined_sp2's book check reads the section rules, which stay).
- **Size:** 261,651 bytes, 493 under the limit; the contest's source check passes.

## The held-out pair (frozen here)

| task | answer | how it is made |
|---|---|---|
| `plusy` | y ← y² + y mod N, T times | `tasks_family.py`: H1's generator and configuration, the same prompts as H1 row for row |
| `three2` | y ← 3y² + 2 mod N, T times | the same |

The coordinator's suggestion, taken unchanged. Locally, `plusy` reads `mod(add(mul(x,x),x),N)` and `three2` reads
nothing (depth 4); both were seen before this was written (below).

## Runs

1. Tiny cores on B200s (`modal_family.py`, `core_f.py`: the file's own climb offline, each task's answers made by
   Python's integers on H1's prompts as the family's data makes them, ROUNDT's rows of T = 2, 4 and 8): `t_family_h1`
   and `t_family_<task>` for cube, ten, rho, plus3, minus1, tenrho, plusy and three2. Each core that finds an answer
   program has it checked on both ladders, T = 1 to 64, seen and unseen moduli (`checks/ladder_family.py`).
2. H1: one official H100 run (seed 74) through `modal_runner_family.py`, checked by `tools/verify_records.py`; if it
   certifies, seeds 1–7 on H100s.
3. The family: ten, rho, plus3, minus1, tenrho, the cube, plusy and three2, one H100 each, seed 74, the file byte for
   byte, through `modal_runner_family.py` (H1's manifest with the task's data root, as SP_BREADTH ran them; plusy and
   three2's data by `tasks_family.py` in the runner's image).
4. "Learned" means certified: T = 64 on both ladders, every rung 768/768. Reported for each task: learned or not, the
   largest T with every rung right on each ladder, the test score, training time, and why for each failure.

## Seen before this was written (local, CPU)

- **The file:** `submission_stage3_family.py`, md5 `457216d7`, 261,651 bytes (493 under the limit); the contest's
  source check passes. Its code differs from the no-book file's only where the port says: 13 functions changed and 12
  new (an AST comparison with the docstrings left out); the room changes no code.
- **combined_sp2's local checks** (`checks/logs/local_family.log`: 278 s, peak 4.6 GiB): the no-book file's results
  part for part (`checks/compare_local.py`), but the size; and the book region's diff against book_gen_fix, 105 lines
  against 92 (the port's patterns and dispatch, the Ops docstring), with check_books part 2 the same (17 concepts,
  6,204 targets). A full run of an earlier build gave the grad part a gradient on `prog` (8.3e-6) where phase B's log
  has none; eight reruns of that part on each file gave none (the tournament's starting weights are random), and this
  build's run gives none.
- **The readings and what they pose** (`checks/family_local.py`: the legal tiny core's state at its level 3, each
  task's answers on H1's 400 answer rows of T = 1 made by Python's integers, outside any run):

  | task | read (rows of 400; laws weighed) | its own laws | candidates |
  |---|---|---|---|
  | H1 | `mod(mul(x,x),N)` (400; 200) | `w.mod.mul` on w2 | 102 |
  | the cube | `mod(mul(mul(x,x),x),N)` (158; 1,140) | `w.mod.mul` | 102 |
  | rho | `mod(add(mul(x,x),1),N)` (400; 3,825) | `const.1`, `add.1`, `w.mod.add1` (w1), `w.add1.mul`, `w.mod.add1.mul` | 163 |
  | plus3 | `mod(add(mul(x,x),3),N)` (400; 8,025) | the same with 3 | 163 |
  | minus1 | `mod(sub(mul(x,x),1),N)` (400; 3,825) | `const.1`, `sub.1`, `w.mod.sub1` (w1), `w.sub1.mul`, `w.mod.sub1.mul` | 115 |
  | plusy | `mod(add(mul(x,x),x),N)` (400; 1,140) | `w.mod.add` on w2 | 150 |
  | ten, tenrho, three2 | none (1,140) | none | 0 |

  Every posed law's answers agree with Python's integers on every draw: 60 law-pattern pairs, 0 disagreements. The
  windows answer 58 to 94 of the 78 to 96 rows where Python's values stay in range (the reader computes fewer: a sum keeps its
  top place free, a product's factors half the places); minus1's windows are masked where the product is 0.
- **ROUNDT** (`checks/family_local.py` MODE=roundt, `checks/roundt_rows.py`, `checks/roundt_power.py`): phase B's four
  H1 rounds pass on 1,000, 4,096 and 4,096 rows of T = 2, 4 and 8; with one answer of T = 4 changed each fails, and so
  does x² without its reduction. Rows stacked as keep_counts stacks them and read by climb_unit's own expression give
  the counts 1, 2, 4 and 8 and ROUNDT's T = 2, 4 and 8. minus1's rows through 0: 24 of 33,333 at T = 4 and 61 at
  T = 8; the tiny core's draw holds 1 and 5.
- **The tiny core's driver** (`core_f.py`, a CPU smoke on minus1, 282 s): the fact read on minus1's own answers and
  level 0 probed; ROUNDT's rows 1,000, 4,096 and 4,096.
- **The held-out data** (`tasks_family.py --small`, with the competition's generator): 644 rows of each, every answer
  its formula's.
- **The runner copy** (`make_runner.py`; `--dry`): the eight family jobs on H100s, H1's manifest with the data root
  `litmus_<task>`; H1's official job with the untouched manifest.
- **The watcher copy** (a synthetic billing report): a stopped app's estimate ends at its stop ($17.83 where the old
  rule gave $27.83); an app the list no longer shows counts its bill only.

## Predictions

| what | prediction | confidence |
|---|---|---|
| t_family_h1: levels 0–2 | t_nobook's, unit for unit: the same picks, library and work | high |
| t_family_h1: level 3 | the same reading, `mod(mul(x,x),N)` on 400 rows; its pose adds the window `w.mod.mul` on w2 (6 bindings): 102 candidates against 96 | high |
| t_family_h1: levels 3–4 and the round | t_nobook's picks and its two-call round; or, if the window is learnable at level 4, the window among the picks and a one-call round | medium |
| t_family_h1: both ladders | every answer program 768/768 on all 14 rungs | high |
| H1, official run | certifies: T = 64 on both ladders, every split exact | medium-high |
| H1, official run: training | 450–900 s (phase B: 574 s) | medium |
| H1, seeds 1–7 | 7 of 7 certify | medium-high |
| plusy | learned: its round is H1's with A = x in place of 0 (x + x·x, then the reduction), two calls | medium-high |
| the cube | learned: two calls of the window `w.mod.mul` | medium-low |
| rho | learned: a window with its + 1 (`w.mod.add1.mul`, or `w.add1.mul` and the reduction) | medium-low |
| plus3 | learned, as rho with 3 | low |
| minus1 | not learned: its windows' answers are masked where the square is 0 (the reader's subtraction stops at 0), so a round built on them is wrong at 0, and ROUNDT, with 1 and 5 rows through 0 at T = 4 and 8 in the tiny core's draw, rejects it; if one is kept, it is right on both ladders | medium |
| ten, tenrho, three2 | not learned: nothing is read (10 is a factor, not a gap; 10y² + 1 and 3y² + 2 need depth 4), so only the fact's three laws are posed and no round is found | high |
| ROUNDT | keeps every first round that is right (as in round 7) | high |
| the family's runs: training | 450–1,000 s for H1-like tasks (plusy); up to the hour for the rest | medium |
| money, phase C | about $60–100 | medium |

## Money

Soft $150, cap $220. `integration/ablation_s3/spend.py 4750` runs before each launch. One spend watcher
(`watch_spend.py`, round 7's with a stopped app's hour under way ending at its stop) for every app of phase C, its
total the cap less one hour of nine B200s (about $56): it stops everything at $164, and any app at $90. Every app is
stopped when its runs end.
