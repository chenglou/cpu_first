# The wider file (phase D): pre-registration

Written 2026-10-27 15:01 PDT, before any GPU run of this file. The file was fixed (md5 below, 14:57:23 PDT) before the
held-out pair was drawn, and has not changed since. WIDER.md §1 repeats this text unchanged and records the time and
hash it was frozen with.

## The question

The family file (phase C, legal_family/FAMILY_FILE.md) learns H1 and 4 of 8 family tasks at full size, and 1 of its
2 held-out tasks. REVIEW flagged ROUNDT's 4,096 rows per count, kept in a Python dict outside the state dict (rules 4
and 5). The brief, phase D:
1. ROUNDT's rows into registered buffers, counted in the state dict; a fresh model loaded from it gives the same
   logits.
2. A cheaper ROUNDT that keeps the same rounds as today's on H1 and the four learned family tasks.
3. A wider reader: constant factors (10y², 3y²), depth 4 (10y² + 1, 3y² + 2), and the smaller constants a constant
   is built from posed with it, so that y² + 3 can make its 3. Every new law answered by the learner's routines.
4. The stale first line of the docstring fixed; bytes freed without changing behaviour.

Then: recertify H1, rerun the family with the file byte for byte, and two new held-out tasks drawn at random from a
stated space. Only the held-out tasks test generality.

## Design (`make_wider.py`, from the family file's committed bytes at ce9f118, md5 `457216d7` asserted)

- **ROUNDT's rows in buffers:** `more_rows` [4 slots, 4,096 rows, 4 fields, 8 digits] (as `count_rows` lays them
  out), `more_n` and `more_s` [4]: the newest 4,096 rows of each count T > 1 the batches show, at most 4 counts
  (ROUNDT_SLOTS), in the order first seen. On H1's data the counts kept are 1, 2, 4 and 8, so T = 2, 4 and 8 fill 3
  slots. The model's state at the start grows from 494,712 to 1,019,008 numbers (524,296 of them ROUNDT's), far
  below rule 5's 500,000,000. `Control.more` is gone.
- **ROUNDT on the round machine:** the round's trace (`answer_trace`) on a `RoundMachine`, run T times per count from
  `run_prog`'s fresh draws (the same generator, the same rows), its work counted as `run_prog` counts it (25.004 per
  primitive), the first count it misses ending it. Same verdicts and same work by construction, a CUDA graph's replays
  instead of `run_items`' interpretation. The suggested 64-row pre-check is not used: a machine's run costs its
  capture and launches, nearly the same for 64 rows as for 4,096, so it would add runs and save none.
- **The reader's third stage** (`LawReader.poly`, `tail`): when the chains over x and N and the gaps read nothing,
  x's polynomials with small coefficients by Horner's rule. From x, up to READ_POLY = 3 steps, each multiplying by x
  or a constant 2..10 or adding x or a constant 1..10 (the machine's digits and its base; 21 steps); each value
  reduced mod N, and exact or off by one constant either way round (the gap). Values computed by the learner's
  routines as the reader computes everything (add, the comparison's difference, repeated addition, the link by
  doubling); the first exact law wins; READ_CHUNK = 512 values a check.
  - Its reach: every a y² + b y + c with a, b ≤ 10 and c any constant, and a y³ + b y + c only when a = 1 or b = 0
    (3 steps). a y³ + b y + c with a ≠ 1 and b ≠ 0 needs 4 steps.
- **A reading's own law at a size too small for it** (10 is no 1-digit number) says nothing there (an empty target)
  instead of dropping the law at every size; a root keeps a law only if it says something at some size
  (`read_target`, `Root.goals`). Book laws never give an empty target, so nothing changes for them.
- **Constants and what they are built from** (`builds`, `read_more`): each constant c of a reading is posed with 1 and
  its doublings below c (1, 2, 4, ...), each made by the learner's addition (v + v) while the learner's comparison
  says it is below c, smallest first. For 3: 1, 2. For 10: 1, 2, 4, 8. For 1: nothing.
- **The docstring's first line** names this file's doc and builder.
- **Room**, no change in behaviour: in the code after the settings, the padding before each trailing comment cut to
  two spaces (comments continued below move with it; the builder asserts the AST unchanged); nine docstrings and one
  settings comment that a design doc explains cut to a pointer.
- **Size:** 261,801 bytes, 343 under the limit; the contest's source check passes. An AST comparison with the family
  file, docstrings left out: 13 functions changed and 3 new (`poly`, `tail`, `builds`), all the above.

## The held-out pair (drawn and frozen here)

- **The space:** y ← a·y^k + b·y + c mod N, T times, with k in {2, 3}, a in 1..9 and b, c in 0..9 (2 × 9 × 10 × 10 =
  1,800), less the 6 tasks of earlier rounds in it (H1, the cube, rho, plus3, plusy, three2): 1,794 tasks.
- **The seed:** not chosen. It is the repository's HEAD commit when phase D began, `ab49f60`, read as a hexadecimal
  number (179,609,440). `checks/draw_heldout.py`: `random.Random(seed).sample(space, 2)` (Python 3.9.6), run at
  14:57:23 PDT, after the file's md5 was recorded (`checks/logs/draw_heldout.log`).
- **The draw:**

  | task | answer |
  |---|---|
  | `poly_8_3_8_3` | y ← 8y³ + 8y + 3 mod N |
  | `poly_2_3_9_9` | y ← 2y³ + 9y + 9 mod N |

  Both are cubics with a ≠ 1 and b ≠ 0: outside the frozen reader's reach (above). Data: `tasks_wider.py`, H1's
  generator and configuration, the same prompts as H1 row for row.

## Arms

- **The file** (`submission_stage3_wider.py`, md5 `59060b5c`): every run below.
- **READ_POLY = 4** (`research/submission_stage3_wider_READ_POLY-4.py`, md5 `9fd20cca`, `make_wider.py READ_POLY=4`;
  the same file with one more Horner step). Added after the draw, to see whether one more step reads the pair and
  whether the climb then learns them. Tiny cores on the two held-out tasks only. It is not a test of the file's
  generality, and its held-out results are not reported as such.

## Runs

1. Tiny cores on B200s (`modal_wider.py`, `core_f.py`): the file on H1, the eight family tasks and the two held-out
   tasks (11); the arm on the two held-out tasks (2). Each core that finds answer programs has them checked on both
   ladders (`checks/ladder_wider.py`).
2. ROUNDT keeps the same rounds: (a) `checks/roundt_same_local.py`, old and new on the same rounds and rows (below);
   (b) the file's tiny cores on H1, plusy, the cube, rho and minus1 against phase C's (`checks/compare_cores.py`).
3. H1: one official H100 run (seed 74) through `modal_runner_wider.py`, `tools/verify_records.py`; then seeds 1–7.
4. The family and the pair: ten, rho, plus3, minus1, tenrho, the cube, plusy, three2, poly_8_3_8_3 and poly_2_3_9_9,
   one H100 each, seed 74, the file byte for byte (`plans/family.json`).
5. "Learned" means certified: T = 64 on both ladders, every rung 768/768. Reported per task: learned or not, T on
   each ladder, the test score, training time, why for each failure.

## Seen before this was written (local, CPU)

- **combined_sp2's local checks** (`checks/logs/local_wider.log`: 313 s, peak 3.8 GiB): the family file's results
  part for part (`checks/compare_local.py`), but the size, the buffer count and the book region's diff. Buffers:
  860,783 numbers against 336,487 (ROUNDT's 524,296 more), with a fresh model from the state dict giving the same
  logits. The book region: 165 lines differ against book_gen_fix, 105 before (`Root.goals` and the comment squeeze),
  and check_books part 2 is the same (17 concepts, 6,204 targets).
- **ROUNDT's buffers** (`checks/roundt_buffers.py`): after each of 300 random batches (counts 1, 2, 4, 8, 11, 3, 9, 0
  and 22), the rows `climb_unit`'s own expression reads from the buffers equal the family file's dict, its counts
  T > 1, the first 4: (2, 4,096 rows), (4, 4,096), (8, 4,096), (11, 2,193). A fresh model loaded with the state dict
  has the same ROUNDT buffers and gives the same logits; the state at the start is 1,019,008 numbers.
- **ROUNDT, old against new** (`checks/roundt_same_local.py`; the rows of core_f.py's draw): 31 rounds, H1's 12
  (phase B's), the cube's 1, rho's 3 and minus1's 13 (phase C's tiny cores), and minus1's 2 plain rounds with its
  level-5 library. The same verdicts on all 31 (false for the 2 plain rounds and for one of minus1's, which reads a
  register before setting it: its verdict depends on the device's draws, and the CPU's reject it), every count's
  outputs equal bit for bit, the same work. 80 s in all against 232 s on the CPU.
- **The readings** (`checks/wider_local.py`, phase C's check; the legal tiny core's state at level 3, each task's
  answers by Python's integers on H1's 400 rows of T = 1):
  - H1, the cube, rho, minus1 and plusy read and pose exactly as in phase C, line for line;
  - plus3 reads as before and also poses `const.1` and `const.2`;
  - ten reads `mod(mul(mul(x,x),10),N)`, tenrho `mod(add(mul(mul(x,x),10),1),N)` and three2
    `mod(add(mul(mul(x,x),3),2),N)`, each on 158 of 400 rows, weighing 1,581 laws in 1.6–1.8 s, and they pose their
    constants with the doublings below them, `mul.10` or `mul.3`, `add.1` or `add.2`, and their windows;
  - all 115 posed law-pattern pairs agree with Python's integers on every draw.
- **The held-out pair:** the file reads neither, weighing 6,726 laws (1,140 chains, the gaps, 5,586 polynomials) in
  9.1–9.2 s each (`logs/pose_heldout.log`). The arm reads both (`logs/pose_heldout_poly4.log`): poly_8_3_8_3 as
  `mod(add(mul(add(mul(mul(x,x),x),x),8),3),N)` (8(x³ + x) + 3) on 16 of 400 rows, and poly_2_3_9_9 as
  `mod(add(mul(add(mul(mul(x,x),2),9),x),9),N)` ((2x² + 9)x + 9) on 147, weighing 60,087 laws in 27–30 s (peak
  7.8 GiB), every posed law right.
- **The data and the tools:** `tasks_wider.py --small` writes 644 rows of each held-out task, every answer its
  formula's; the runner copy's dry runs give the ten jobs with H1's manifest and `litmus_<task>`, and H1's official job
  untouched; the watcher, on a synthetic report with 11 containers running at $4, stops everything at $136 of the
  $180 cap.

## Predictions

| what | prediction | confidence |
|---|---|---|
| tiny cores of H1, plusy, the cube, rho and minus1 | phase C's, unit for unit: the same lines (seconds masked) and work, picks, library, rounds and answer programs (their readings and posing are phase C's; ROUNDT's verdicts and work are the same by construction) | high |
| their round searches that keep rounds | at most half phase C's seconds (H1's after level 4: 70 s on the B200) | medium |
| their answer programs on both ladders | 768/768 on all 14 rungs (minus1: as phase C's, two of three) | high |
| plus3's tiny core | makes 1, 2 and 3, and finds its round within the work budget | medium-low |
| ten's, tenrho's and three2's tiny cores | read and pose as above; a round for three2 | low |
| ten's and tenrho's | no round (the ×10 laws need the constants' doublings, or a shift found by search) | medium-low |
| the pair, the file | nothing read (4 Horner steps), nothing learned | high |
| the pair, the arm's tiny cores | read both; learned: neither (their rounds need three calls or more) | medium-low |
| H1, official run | certifies | high |
| H1, official run: training | 450–700 s (ROUNDT no longer slows the round search) | medium |
| H1, seeds 1–7 | 7 of 7 certify | high |
| the family at full size | plusy, rho, the cube and minus1 learned again, minus1 in less training than phase C's 1,534 s | medium-high |
| plus3, ten, tenrho, three2 at full size | as their tiny cores | medium |
| the pair at full size | not learned: nothing read | high |
| money, phase D | about $50–80 | medium |

## Money

Soft $120, cap $180. `integration/ablation_s3/spend.py 4750` runs before each launch (the month: $4,096 at 14:58
PDT). One spend watcher (`watch_spend.py`: phase C's, with the margin counted from the containers still running) for
every app of phase D: it stops everything once the estimate plus one more hour of the running containers (their
number times $6.25 for the B200 cores, $4 for the H100 runs) reaches $180, and any app at $90. Every app is stopped
when its runs end.
