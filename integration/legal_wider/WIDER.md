# The wider file (phase D): ROUNDT in the buffers and cheaper, a wider reader, H1 and the family

<!-- PLAIN -->
## In plain words

**What changed** (`make_wider.py`, from the family file's committed bytes at ce9f118):
- **ROUNDT's rows are in registered buffers** (`more_rows`, `more_n`, `more_s`): the newest 4,096 rows of each count
  T > 1, at most 4 counts. They are counted in the state dict, and a fresh model loaded from it gives the same
  logits. The model's state at the start grows from 494,712 to 1,019,008 numbers; rule 5's limit is 500,000,000.
- **ROUNDT is cheaper, with the same verdicts.** It runs each round on the round machine, a CUDA graph, from
  `run_prog`'s own random draws, and counts its work as `run_prog` counts it. The suggested 64-row pre-check is not
  used: on the machine a run costs about the same for 64 rows as for 4,096.
- **The reader has a third stage:** when nothing else reads, x's polynomials with small coefficients by Horner's
  rule. From x, up to 3 steps, each multiplying by x or by a constant 2..10, or adding x or a constant 1..10, then the
  reduction mod N, the last constant by the gap. Every value is the learner's routines', as before.
- **What the reader weighs:** when its first two stages read, what phase C's reader weighed (200 to 8,025 laws on the
  family's tasks); when they read nothing, up to 5,586 polynomials more, 6,726 laws in all on the held-out pair, at
  3–4 s a try on a B200 and at most 3 tries, against a climb of 5–10 minutes. One more Horner step would weigh 60,087.
- **Each constant is posed with what it is built from:** 1 and its doublings below it (for 3: 1, 2; for 10: 1, 2, 4,
  8), each made by the learner's addition and checked by its comparison.
- **A law at a size too small for it says nothing there** (10 is no 1-digit number), instead of being dropped.
- **The first line of the docstring** names this file's doc and builder. The bytes came from squeezing the padding
  before trailing comments to two spaces, and nine docstrings cut to a pointer; no code changed.

**ROUNDT keeps the same rounds.**
- Old and new ROUNDT on 31 rounds of the five learned tasks (and on minus1's plain rounds, which it rejects): the
  same verdicts, every output bit for bit, the same work.
- The tiny cores of H1, plusy and the cube are phase C's in every unit, line and work; rho's differs only by one
  extra comparison in its reading unit.
- minus1's climb changed at its level-3 pose (one more law is posed, by the empty-target rule, not ROUNDT). An arm
  with phase C's ROUNDT swapped back gives the same 67 units, the same rounds and answer programs (addendum 1).
- Its round searches are 2.6–7 times faster: H1's after level 4 took 18 s against 70 s on a B200, and minus1's
  after level 6 71 s against the arm's 512 s on the same climb.

**H1 recertifies.**
- The official H100 run certifies: T = 64 on both ladders, every rung 768/768, test 9,999/9,999, unseen T and unseen
  moduli 10,002/10,002; training 522 s (phase C's 783 s).
- So do seeds 1–7 ("8 of 8 records check out"). Every seed finds its answer program and freezes at phase C's step.
  Training takes 509–777 s against phase C's 562–986 s; the round search with ROUNDT's checks 6–51 s against 36–344 s.

**The family and the held-out pair**, one H100 each, seed 74, the file byte for byte, each task's data as SP_BREADTH
or phase C made it, the pair's by H1's generator:

| task | learned? | T, both ladders | test | training | phase C |
|---|---|---|---|---|---|
| plusy, y² + y | **yes** | 64 / 64 | 9,999/9,999 | 520 s | yes, 631 s |
| rho, y² + 1 | **yes** | 64 / 64 | 9,999/9,999 | 720 s | yes, 695 s |
| the cube, y³ | **yes** | 64 / 64 | 9,999/9,999 | 664 s | yes, 825 s |
| minus1, y² − 1 | **yes** | 64 / 64 | 9,999/9,999 | 816 s | yes, 1,534 s |
| plus3, y² + 3 | **yes** | 64 / 64 | 9,999/9,999 | 681 s | no: it never made 3 |
| ten, 10y² | **yes** | 64 / 64 | 9,999/9,999 | 749 s | no: nothing read |
| tenrho, 10y² + 1 | **yes** | 64 / 64 | 9,999/9,999 | 677 s | no: nothing read |
| three2, 3y² + 2 (phase C's held-out) | no | – | 4/9,999 | (2,951 s, all 20,000 steps) | no: nothing read |
| poly_8_3_8_3, 8y³ + 8y + 3 (held out, drawn) | no | – | 3/9,999 | (the hour, 4,215 steps) | not run |
| poly_2_3_9_9, 2y³ + 9y + 9 (held out, drawn) | no | – | 19/9,999 | (the hour, 4,155 steps) | not run |

Seven of the eight family tasks are learned (phase C: four). Each certifies as H1 does: every rung 768/768 on both
ladders, every split exact. The two held-out tasks, drawn at random for this phase, are not read, so neither is
learned.

**What it means.**
- **Only the held-out pair tests generality, and the file learns neither (0 of 2).** The two tasks were drawn at
  random after the file was fixed: 8y³ + 8y + 3 and 2y³ + 9y + 9. Both take four Horner steps, and the reader stops
  at three. So it reads nothing and poses nothing new, and the climb only practises the step's parts. One more step
  (the READ_POLY = 4 arm, not the file) reads both, at nine times the laws. Even so, its tiny cores ran out of work
  budget at level 8 with no round.
- **The family's gains are not tests of generality.** The eight family tasks were known when the reader was widened:
  10y², 3y², 10y² + 1 and 3y² + 2 were the brief's own targets, and the doublings were made for plus3's 3. On them the
  file does what it was built to do:
  - ten and tenrho are read by the third stage and learned. ×10 is found as a one-digit shift once the empty target
    poses it.
  - plus3 makes its 1 (posed with its 3), then 3 as 1 + 1 + 1, and is learned.
  - three2 is read and posed, but the climb's work runs out at the level that learns its window.
- **ROUNDT is now legal under rules 4 and 5, and cheaper.** Its rows are buffers in the state dict, and its verdicts
  and work are the same. Its round searches are 2.6–7 times faster: H1 trains in 509–777 s against 562–986 s, and
  minus1 in 816 s against 1,534 s.
- **What would move the held-out pair** (not tried): a fourth Horner step, and a climb that reaches a round for such a
  law within its work budget. three2 shows the budget is already tight at three steps.
- **Newly designed** (§9): the third stage's steps and depth, the doublings, the empty target and ROUNDT's four slots.
- **Not covered:** SP_BREADTH's mont, M4, xfirst and xy, as in phase C.

**Cost:** about $43 of the soft $120:
- the fourteen B200 tiny cores, $12.15;
- H1's eight H100 runs, $9.92;
- the family and the pair, $18.13 billed, plus about $3 for their last minutes, not yet in the billing report.

Every app has stopped. The month stood at $4,136 billed afterwards.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, frozen 2026-10-27 15:01:27 PDT (read-only, sha256
`baaf0dbee38e6f7b943cec2f0aec3472bdce7a92c67b0761feae4887e5df0b2d`), before any GPU run of this phase. The file was
fixed (md5 `59060b5c`) at 14:57:23 PDT, before the held-out pair was drawn, and has not changed since.
`PREREG_ADDENDUM1.md`, frozen 15:14:25 PDT (sha256
`da45bd3197da9219a44d100cc13df3dd3cf3089bad787927da6ba0aad72c594c`), before its arm's run: ROUNDT's equivalence on
minus1's own climb. Hashes of both and of the code: `checks/logs/prereg_sha256.txt`. §7 sets each prediction against
the outcome.

**Two changes after the freeze, both outside the file.** `make_wider.py` gained `--skip=GROUP` to build addendum 1's
arm (md5 `93274593`, frozen as `6eaf0aee`); it still writes the file byte for byte (md5 `59060b5c`, checked in
memory). `modal_wider.py` stopped leaving `research/` out of what it sends to the tiny cores, so the arms' files reach
them (md5 `06a85c64`, frozen as `f3fae0e7`).

### 2. The changes (`make_wider.py`)

From the family file's committed bytes at ce9f118 (md5 `457216d7` asserted), groups of exact edits, each asserted to
match once:

| group | edits | what |
|---|---|---|
| doc | 1 | the module's docstring: this file's doc and builder in its first line, the third stage, the buffers |
| buffers | 5 | ROUNDT_SLOTS; `more_rows` [4, 4,096, 4, 8], `more_n`, `more_s` registered; `keep_counts` keeps the newest 4,096 rows of each count T > 1 in its slot (the first free one when new), at most 4 counts; `climb_unit` hands them to the climb; `Control.more` removed |
| roundt | 1 | `round_t` on a `RoundMachine` built from the round's trace (`answer_trace`): T runs per count from `run_prog`'s fresh draws (the same generator and rows), work counted as `run_prog` counts it (25.004 per primitive), the first count it misses ending it |
| reader | 10 | READ_POLY = 3, READ_CHUNK = 512; `layer` takes a list of steps (the chains' order unchanged); `poly` (the third stage) and `tail` (its reduction and gap); `builds` (1 and its doublings below a constant, by the learner's addition and comparison) and `read_more` posing them first; `read_target`'s empty target at a size too small for a law, and `Root.goals` keeping a law only if it says something at some size |
| room | 12 | the comment squeeze (below); nine docstrings that a design doc explains cut to a pointer (`windows`, `read_more`, `read_target`, `above_probes`, `deep_roots`, `_ustep`, `link_soft`, `set_round`, `_run_count`); the FULL_CHECK/ABOVE_CHECK comment to a pointer; a long line split |

**The comment squeeze.** In the code after the settings (from the first `def` or `class`), the padding before each
trailing comment cut to two spaces; comment-only lines just below it at its old column move with it. No text changes;
the builder asserts the AST unchanged. It freed 3.5 KB of the 4.4 KB the changes needed.

The file: `submission_stage3_wider.py`, md5 `59060b5cd7a7b536e25302e1d5be7ef7`, 261,801 bytes (343 under the limit);
the contest's source check passes. An AST comparison with the family file, docstrings left out: 13 functions changed
and 3 new (`LawReader.poly`, `LawReader.tail`, `builds`).

**Choices, and why:**
- **ROUNDT's 64-row pre-check, not used.** On the GPU the machine's run is its capture and its kernel launches, about
  the same for 64 rows as for 4,096, so a pre-check adds runs and saves none; the machine is where the time went.
- **ROUNDT's work counted as before,** instruction by instruction at `run_prog`'s rate, so the climbs' budgets behave
  as in phase C, and a climb is the same unit for unit (§4).
- **Four slots** for ROUNDT's counts: H1's data and the family's keep 1, 2, 4 and 8, so 3 counts past 1.
- **The third stage reads x's polynomials by Horner's rule:** chains from x by multiplications and additions with x or
  the machine's own constants (its digits and its base), the value reduced mod N at the end, the last constant by the
  gap. Three steps read every a y² + b y + c with a, b ≤ 10, and a y³ + b y + c only when a = 1 or b = 0. It runs only
  when the first two stages read nothing, so the readings of phase C's readable tasks are unchanged.
- **The doublings below a constant** are made by the learner's addition (v + v from 1) and stopped by its comparison:
  every constant posed is a routine's output, never Python's arithmetic.
- **The empty target** keeps a law a size cannot hold (10 at 1-digit size) instead of dropping it from every size;
  without it ten's and tenrho's ×10 laws are never posed (the first local check showed it).

### 3. Checks (local, CPU)

| check | result |
|---|---|
| combined_sp2's `checks/local_combined.py`, every part (`checks/logs/local_wider.log`: 313 s, peak 3.8 GiB) | the family file's results part for part (`checks/compare_local.py`), but the size; the buffer count, 860,783 against 336,487 (ROUNDT's 524,296), a fresh model from the state dict giving the same logits; and the book region's diff, 165 lines against 105 (`Root.goals`, the comment squeeze), check_books part 2 the same (17 concepts, 6,204 targets) |
| `checks/roundt_buffers.py` (`logs/roundt_buffers.log`) | after each of 300 random batches (counts 1, 2, 4, 8, 11, 3, 9, 0, 22), the rows `climb_unit`'s own expression reads from the buffers equal the family file's dict, counts T > 1, the first 4: (2, 4,096), (4, 4,096), (8, 4,096), (11, 2,193); a fresh model from the state dict: the same buffers, the same logits; the state at the start 1,019,008 numbers |
| `checks/roundt_same_local.py` (`logs/roundt_same.log`) | old and new ROUNDT on 31 rounds (H1's 12, the cube's 1, rho's 3, minus1's 13 from phase C's tiny cores, minus1's 2 plain rounds at level 5): the same verdicts (3 false: the 2 plain rounds, and one of minus1's rounds that reads a register before setting it, so its verdict depends on the device's draws), every count's outputs equal bit for bit, the same work; 80 s in all against 232 s on the CPU |
| `checks/wider_local.py` (`logs/pose.log`) | H1, the cube, rho, minus1 and plusy read and pose as in phase C, line for line; plus3 also poses `const.1` and `const.2`; ten, tenrho and three2 read (below); all 115 posed law-pattern pairs agree with Python's integers |
| the held-out pair (`logs/pose_heldout.log`, `logs/pose_heldout_poly4.log`) | the file reads neither (6,726 laws, 9 s each); the arm (READ_POLY = 4) reads both (60,087 laws, 27–30 s, peak 7.8 GiB), every posed law right |
| `tasks_wider.py --small`; the runner copy's dry runs; the watcher on a synthetic report | 644 rows of each held-out task, every answer its formula's; the ten jobs with H1's manifest and `litmus_<task>`, H1's official job untouched; with 11 containers running at $4 the watcher stops everything at $136 of $180 |

What the reader reads and weighs (H1's 400 rows of T = 1 with each task's answers; the tiny cores' times are a
B200's):

| task | read (rows of 400) | laws weighed | seconds, CPU / B200 |
|---|---|---|---|
| H1 | `mod(mul(x,x),N)` (400) | 200 | 0.3 / 0.3 |
| plusy | `mod(add(mul(x,x),x),N)` (400) | 1,140 | 1.0 / 0.6 |
| the cube | `mod(mul(mul(x,x),x),N)` (158) | 1,140 | 1.0 / 1.2 |
| rho, minus1 | `mod(add(mul(x,x),1),N)`, `mod(sub(mul(x,x),1),N)` (400) | 3,825 | 3.2 / 1.5 |
| plus3 | `mod(add(mul(x,x),3),N)` (400) | 8,025 | 5.4 / 1.8 |
| ten | `mod(mul(mul(x,x),10),N)` (158), new | 1,581 | 1.7 / 1.0 |
| tenrho | `mod(add(mul(mul(x,x),10),1),N)` (158), new | 1,581 | 1.6 / 0.9 |
| three2 | `mod(add(mul(mul(x,x),3),2),N)` (158), new | 1,581 | 1.7 / 1.9 |
| poly_8_3_8_3, poly_2_3_9_9 | nothing (they need 4 Horner steps) | 6,726 | 9.2 / 3.2–3.7, at each of 3 tries |
| the arm on them | `mod(add(mul(add(mul(mul(x,x),x),x),8),3),N)` (16), `mod(add(mul(add(mul(mul(x,x),2),9),x),9),N)` (147) | 60,087 | 27–30 / 6.6–6.9 |

The first two stages weigh what phase C's reader weighed (200 to 8,025 laws). The third adds up to 5,586 polynomials
at 3 steps (55,947 more at 4), all through the learner's routines; in a run it costs seconds, against a climb of 5–10
minutes.

### 4. The tiny cores

`core_f.py` (phase C's, with the held-out tasks' names) on one B200 each through `modal_wider.py` (app and Volume
`legal-wider`), as phase C ran them: the file's own climb, offline, 4,096 read rows and 400 answer rows of T = 1,
ROUNDT's rows of T = 2, 4, 8; then the answer program's search. Every answer program found was run on both ladders,
T = 1 to 64, seen and unseen moduli (`checks/ladder_wider.py`, `checks/logs/ladder_t_wider_<task>.log`; H1's, plusy's,
the cube's and rho's are phase C's, their tiny cores being the same).

| task | the round | answer programs | both ladders | climb + answer search |
|---|---|---|---|---|
| H1 | after level 4, phase C's 12 rounds | phase C's 2 | 768/768, all 14 rungs | 229 s; 1.21e9 + 0.11e9 |
| plusy | after level 4, phase C's | phase C's 4 | 768/768 | 217 s; 1.23e9 + 0.11e9 |
| the cube | after level 5, two calls of the window (x·o) mod N | 1 | 768/768 | 366 s; 2.22e9 + 1.47e9 |
| rho | after level 5, phase C's | 2 | 768/768 | 324 s; 2.78e9 + 0.11e9 |
| minus1 | after level 6, x + N or N − x then the window (x·x − 1) mod N | 3 | two 768/768 on all 14 rungs, the third wrong on 151 rows (as phase C's) | 452 s; 3.48e9 + 0.11e9 |
| plus3, **new** | after level 5: its window x·o + 3 (`T.w.add3.mul`), then the reduction | 2 | both 768/768 | 324 s; 2.76e9 + 0.11e9 |
| ten, **new** | after level 5: the window 10·x·o (`T.w.mul10.mul`), then the reduction | 2 | both 768/768 | 306 s; 2.25e9 + 0.11e9 |
| tenrho, **new** | after level 6: the window 10·x·o + 1 (`T.w.add1.mul10.mul`), then the reduction | 1 | 768/768 | 417 s; 4.21e9 + 0.12e9 (past the 4.3e9 budget by 0.02e9 in the answer search) |
| three2 | none: it learned 1 and ×3 at level 3, + 2 at level 4, its windows 3·x·o + 2 and 3·x + 2 and its 3 at level 5, its 2 at level 7; no round of two calls at levels 4–6 (why not traced), and the work budget ran out at level 7 (4.43e9) | – | – | 616 s |
| poly_8_3_8_3, poly_2_3_9_9 | none: nothing read | – | – | 119–121 s |
| the arm on them | none: both read, the budget out at level 8 (4.30e9, 4.53e9) | – | – | 548–554 s |

**ROUNDT against phase C's** (`checks/compare_cores.py`, `logs/compare_cores_*.log`):
- H1 (52 units), plusy (53) and the cube (61): every unit the same, line (seconds masked) and work; the same picks,
  library, rounds and answer programs.
- rho (62 units): the same, but its reading unit has 825 units more work (`builds`' one comparison for its 1), and a
  few later units' work differs at 1e-7 (float rounding of the offset totals); the same picks, library, rounds and
  answer programs.
- minus1: different from its level-3 pose, 115 candidates against phase C's 114 on the GPU. One of its laws had fewer
  than two computed rows at the smallest size on the GPU's draws, so phase C dropped it, and the empty target keeps
  it. From there its climb differs: its round after level 6 as before, 13 rounds, not the same ones, the same three
  answer-program names. **Addendum 1's arm** (the file with phase C's `round_t`, `make_wider.py --skip=roundt`): the
  same 67 units as the file's minus1 core, line and work, the same 13 rounds and answer programs, in 975 s against
  452 s.
- The round searches that check rounds: H1's after level 4, 18.4 s against 70.2; plusy's 9.2 against 34.9; the
  cube's after level 5, 5.2 against 13.7; rho's 5.8 against 18.7; minus1's after level 6, 70.7 against phase C's
  461.7 (another climb) and the arm's 511.5 (the same climb).

### 5. H1: the official run and seeds 1–7

`modal_runner_wider.py --mode official` (H1's manifest untouched, seed 74, one H100) and `--mode seeds` (seeds 1–7,
one H100 each); records `records/legal_wider_official.json`, `records/legal_wider_h100_seed<N>.json`.
`tools/verify_records.py` on the eight: "8 of 8 records check out" (`checks/logs/verify_h1.log`). Every one: T = 64 on
both ladders, every rung 768/768, test 9,999/9,999, unseen T and unseen moduli 10,002/10,002; the model state at the
start 1,019,008 numbers in every record.

| seed | the answer program, step (phase C) | frozen (phase C) | the round search after level 4, s (phase C) | training s (phase C) | evaluation s |
|---|---|---|---|---|---|
| 74 (official) | 632 (632) | 831 (831) | 24.1 (186.8) | 522 (783) | 271 |
| 1 | 653 (653) | 852 (852) | 6.4 (36.4) | 509 (568) | 324 |
| 2 | 649 (649) | 848 (848) | 30.7 (161.5) | 777 (783) | 423 |
| 3 | 676 (676) | 875 (875) | 25.9 (140.2) | 720 (804) | 623 |
| 4 | 716 (716) | 915 (915) | 25.9 (165.9) | 581 (724) | 337 |
| 5 | 673 (673) | 872 (872) | 51.0 (344.4) | 690 (986) | 349 |
| 6 | 687 (687) | 886 (886) | 21.1 (99.0) | 581 (562) | 379 |
| 7 | 632 (632) | 831 (831) | 21.3 (103.8) | 579 (570) | 396 |

Every seed's answer program and freeze come at phase C's step; the round search with ROUNDT's checks takes 6–51 s
against 36–344 s. Training is 509–777 s against phase C's 562–986 s (phase B's, with no ROUNDT: 472–634 s).

### 6. The family and the held-out pair at full size

`modal_runner_wider.py --mode plan` (`plans/family.json`): each task on one H100, seed 74, the file byte for byte
(md5 `59060b5c` in every record), H1's manifest with only the name and `data.data_root` changed (`litmus_<task>`). The
data is SP_BREADTH's where it ran the task: each record's data md5 equals its SP_BREADTH record's (ten `f02bdb5a`, rho
`b19d39e7`, plus3 `a9d9e672`, minus1 `3089dc1e`, tenrho `0b18439b`, the cube `75a99480`), and phase C's for plusy
(`feb26ee2`) and three2 (`343745f1`); the pair's by `tasks_wider.py` in the runner's image (poly_8_3_8_3 `89379526`,
poly_2_3_9_9 `96cf55ca`). Summary: `checks/summarize_family.py` (`checks/logs/summarize_family.log`).

| task | learned? | T, seen / unseen moduli | test | training s | the round (from the run's log) | phase C |
|---|---|---|---|---|---|---|
| H1 (official, §5) | **yes** | 64 / 64 | 9,999/9,999 | 522 | after level 4: the row's loop x·o, then the reduction | yes, 783 s |
| plusy | **yes** | 64 / 64 | 9,999/9,999 | 520 | after level 4: x + x·x, then the reduction | yes, 631 s |
| rho | **yes** | 64 / 64 | 9,999/9,999 | 720 | after level 5: the window x·o + 1, then the reduction | yes, 695 s |
| the cube | **yes** | 64 / 64 | 9,999/9,999 | 664 | after level 5: two calls of the window (x·o) mod N | yes, 825 s |
| minus1 | **yes** | 64 / 64 | 9,999/9,999 | 816 | after level 6: x moved by N first, then the window (x·o − 1) mod N | yes, 1,534 s |
| plus3 | **yes** | 64 / 64 | 9,999/9,999 | 681 | after level 5: the window x·o + 3, then the reduction | no: 3 never made |
| ten | **yes** | 64 / 64 | 9,999/9,999 | 749 | after level 5: the window (10·x·o) mod N, then the reduction | no: nothing read |
| tenrho | **yes** | 64 / 64 | 9,999/9,999 | 677 | after level 5: the window 10·x·o + 1, then the reduction | no: nothing read |
| three2 (phase C's held-out) | no | – / – | 4/9,999 | 2,951 (all 20,000 steps) | none: its window 3·x·o + 2 came at level 6, and the work budget ran out there | no: nothing read |
| poly_8_3_8_3 (held out, drawn) | no | – / – | 3/9,999 | 3,608 (the hour, 4,215 steps) | none: nothing read | not run |
| poly_2_3_9_9 (held out, drawn) | no | – / – | 19/9,999 | 3,605 (the hour, 4,155 steps) | none: nothing read | not run |

Every learned task: every rung 768/768 on both ladders, unseen T and unseen moduli 10,002/10,002. Each found its
answer programs one step after its round (the simplest `C.S[T.round1 STA.X]`) and froze 199 steps later. The model
state at the start is 1,019,008 numbers in every record but ten's and tenrho's, 1,018,944: their sequences are two
tokens shorter (19 against 21), as in phase C's records (494,648 against 494,712).

**How the three new ones were learned** (the runs' logs; their tiny cores the same way):
- **plus3** makes its 1, then its 3. Its reading poses 1 and 2 with 3 (the doublings below 3). The 1 is learned at
  level 3 (rho's four-instruction program), 3 at level 4 as 1 + 1 + 1 (`T.const.1.n.N STA.Y ADC.Y ADC.Y`) with
  x + 3, and the window x·o + 3 at level 5. Its 2 came last, as 3 − 1, and is not used. Phase C posed only the 3,
  and nothing in reach made it.
- **ten** is read by the third stage, `mod(mul(mul(x,x),10),N)` on 71 of the 141 answer rows. The empty target poses
  its ×10 laws, and ×10 is learned at level 3 as a one-digit shift, `CLK SHL.A` (the machine's base is 10). The
  window (10·x·o) mod N at level 5 is x·o, the shift and the reduction. Its doublings 1, 2, 4, 8 were learned too and
  are not used.
- **tenrho** is read the same way (72 of 144 rows). 10·x + 1 is learned at level 3 (`CLK SHL.X T.red.pair.XY
  ADC.X`), the window 10·x·o + 1 at level 5, and the round right after; its tiny core found it a level later.

**Why the others were not learned:**
- **three2: the work budget ran out just after the window it needs.** Its reading poses 1, 2 and 3, ×3 and + 2, and
  five windows. The climb learns 1 and ×3 at level 3, and + 2 and 2 at level 4. At level 5 it learns the windows
  3x + 2, (3x + 2) mod N and 3·x·o, and the round search finds no round of two calls. 3·x·o + 2 comes at level 6.
  That level's probes take the climb's work from 3.17e9 to 4.36e9, past its 4.3e9 budget, so no round search runs
  after it, and levels 7 and 8 pick nothing. Its tiny core learned that window a level earlier and still found no
  two-call round at levels 5 and 6. Why was not traced.
- **The held-out pair: nothing read.**
  - The fact read (every answer below N) poses the step's parts, which are practised at levels 0–2.
  - The law reader then tries at levels 3, 4 and 5, on 141, 159 and 201 answer rows, weighing 6,726 laws each time
    (2.5–3.0 s on the H100). It reads nothing: both laws need four Horner steps.
  - With nothing read, nothing more is posed. The climb runs to its level cap (level 10) with 9.8e7 work.
  - The round search then retries every 10 steps, 8.5 s each, until the hour ends. So each run makes only about 4,200
    of its 20,000 steps.

### 7. Predictions against outcomes

| what | prediction (confidence) | outcome |
|---|---|---|
| tiny cores of H1, plusy, the cube, rho and minus1 | phase C's, unit for unit (high) | **right for H1, plusy and the cube.** **rho: right but for one comparison**, `builds` on its 1 (+825 work in its reading unit); the same picks, library, rounds and answer programs. **minus1: wrong**: its level-3 pose has one more law (the empty target), so its climb is not phase C's; addendum 1's arm shows ROUNDT is not the cause |
| their round searches that keep rounds | at most half phase C's seconds (medium) | **right**: 26–38% for H1, plusy, the cube and rho; minus1's 15% of phase C's, 14% of the arm's |
| their answer programs on both ladders | 768/768 on all 14 rungs; minus1 two of three (high) | **right** |
| addendum 1: the arm against the file on minus1 | the same unit for unit (high); slower (high) | **right**: the same 67 units, rounds and answer programs; 975 s against 452 s |
| plus3's tiny core | makes 1, 2 and 3; a round within the budget (medium-low) | **right**: 1 at level 3, 3 at level 4 (1 + 1 + 1), 2 at level 5 (3 − 1), its round after level 5 |
| ten's, tenrho's and three2's tiny cores | read and pose as in the local check; a round for three2 (low) | **read and posed: right. three2's round: wrong**: none, the work budget out at level 7 |
| ten's and tenrho's | no round (the ×10 laws need the doublings, or a shift found by search) (medium-low) | **wrong**: the search found the shift at level 3 (`CLK SHL.A`); rounds after level 5 (ten) and 6 (tenrho) |
| the pair, the file | nothing read (4 Horner steps), nothing learned (high) | **right** in the tiny cores; at full size too (nothing read at levels 3–5; test 3 and 19 of 9,999) |
| the pair, the arm's tiny cores | read both; learn neither (their rounds need three calls or more) (medium-low) | **right**: both read, neither learned; the budget out at level 8 (why not traced: the three calls are not shown) |
| H1, official run | certifies (high) | **right** |
| H1, official run: training | 450–700 s (medium) | **right**: 522 s |
| H1, seeds 1–7 | 7 of 7 certify (high) | **right** |
| the family at full size | plusy, rho, the cube and minus1 learned again, minus1 in less than phase C's 1,534 s (medium-high) | **right**: all four; minus1 816 s |
| plus3, ten, tenrho, three2 at full size | as their tiny cores (medium) | **right** for plus3, ten and tenrho (learned); three2 **right** (no round: the work budget out at level 6) |
| the pair at full size | not learned: nothing read (high) | **right**: nothing read, nothing learned. Each run spent the hour on round searches retried every 10 steps |
| money, phase D | about $50–80 (medium) | **lower**: about $43 ($40.19 billed, and about $3 for the last minutes) |

### 8. What REVIEW needs to switch to this file

- **The file to judge:** `integration/legal_wider/submission_stage3_wider.py`, md5
  `59060b5cd7a7b536e25302e1d5be7ef7`, 261,801 bytes (343 under 262,144), written by `make_wider.py` from the family
  file's committed bytes at ce9f118; the contest's source check passes.
- **Its records:** `records/legal_wider_official.json` (seed 74) and `records/legal_wider_h100_seed<1..7>.json`.
  `python3 tools/verify_records.py <each>=integration/legal_wider/submission_stage3_wider.py`: "8 of 8 records check
  out" (`checks/logs/verify_h1.log`). `tools/verify_records.py`'s own default list (its JUDGE and records) still names
  the family file; it needs the same switch ab49f60 made for that file.
- **Rule 5:** 207,370 parameters, as before. Every record counts 1,019,008 numbers of model state at the start
  (ten's and tenrho's 1,018,944, §6):
  - the parameters;
  - phase C's buffers, 287,342: the kept rows (`pool_rows` 65,536, `ans_rows` 12,288, `count_rows` 204,800), their
    sizes and counts (4,709), and the answer program's 9, empty at the start;
  - ROUNDT's 524,296: `more_rows` [4, 4,096, 4, 8] = 524,288, `more_n` [4], `more_s` [4].

  With the generic instructions' official rounds in the answer program's buffers, 860,783 buffer numbers (the local
  check). A fresh model loaded with the state dict gives the same logits (`checks/logs/local_wider.log`,
  `checks/logs/roundt_buffers.log`).
- **Rule 4, the rows kept across batches,** now all buffers (lines 4101–4113): `pool_rows` (4,096 read rows),
  `ans_rows` (512 answer rows), `count_rows` (the newest of each count, at most 64 each and COUNT_POOL in all) and
  ROUNDT's `more_rows` (the newest 4,096 of each count T > 1, at most 4 counts: on H1, T = 2, 4 and 8). They come in
  through the loss, which reads the labels: `keep_answers` (line 4827) and `keep_counts` (line 4842; ROUNDT's lines
  just below `rows = ...`). `climb_unit` (line 4714) hands ROUNDT's rows from the buffers to the climb (line 4729).
  `Control.more` is gone; REVIEW's flag on it (rules 4 and 5) can go.
- **What the file still keeps outside the state dict,** as before, none of it rows of the data:
  - the climb's working state on `Control` (`ctl.climb`, a `SelfClimb`: its library and routines, the readings and
    what they pose, probes and records), as an optimizer's state would (STRICT.md §1), and Control's schedule
    (phase, step, streaks, history);
  - module-level state that a run changes: `READ_LAWS` (line 2972, a reading's own laws and their trees), the law
    book's added entries (`LAW_BOOK.setdefault` in `read_more`), the work counter `WORK` (line 1424), and the caches
    `_TRACES` (line 703) and `_OUTER` (line 1393);
  - `ctl.climb.more`, the climb's handle on ROUNDT's rows: views of the buffers, made again at each climb unit
    (line 4729), with no rows of its own;
  - ROUNDT's round machines, built per check and dropped.
- **Rule 7, what is newly designed:** the polynomial stage (its 21 steps: ×x, ×2..10, +x, +1..10, the machine's digits
  and base; READ_POLY = 3; the gap for the last constant), the doublings below a constant, the empty-target rule, and
  ROUNDT_SLOTS = 4; phase B's and C's lists stand (NOBOOK.md §7, FAMILY_FILE.md §9).
- **Generality:** only the held-out pair tests it. The file reads neither of the two drawn tasks and learns neither: 0
  of 2. The family's eight were known when the reader was widened (10y², 3y², 10y² + 1, 3y² + 2 were the brief's
  targets), so they do not.
- **REVIEW's line numbers.** REVIEW's §2, §5 and §6 cite 100 lines of the family file, 152 times.
  `checks/review_lines.py` gives this file's line for each (`checks/logs/review_lines.log`). Most moved by −3 to +60.
  Six lines changed:
  - the docstring's first line (1);
  - `READ_TRIES` (104 → 102), with `READ_POLY` and `READ_CHUNK` after it and `ROUNDT_SLOTS` at 98;
  - `read_more` (3005 → 3051);
  - `Control.more` (4022), gone: its rows are now the buffers at 4111–4113;
  - `climb_unit`'s hand-over (4674 → 4729);
  - `keep_counts`' ROUNDT lines (4805 → 4859–4865).

  New code REVIEW may cite: `Root.goals`' empty-target clause (line 2385), `LawReader.poly` and `tail` (2927, 2944),
  `builds` (3040), `read_target`'s empty target (3081).
- **The family's records:** `records/wider_<task>_h100.json` (ten), §6.

### 9. What is still designed

Phase B's and C's lists stand (NOBOOK.md §7, FAMILY_FILE.md §9). This phase adds:
- **The polynomial stage:** chains from x with ×x, ×c, +x, +c for c among the machine's digits and its base, up to
  three steps, reduced mod N at the end, the last constant by the gap; it runs only when the first two stages read
  nothing.
- **The constants' doublings:** 1 and its doublings below a constant, each a routine's output.
- **The empty target:** a reading's own law says nothing at a size too small for it.
- **ROUNDT's slots:** at most four counts, the first seen.

### 10. Where it still touches Python

As the family file (FAMILY_FILE.md §10), plus:
- **The third stage's constants** are the digit strings "1" to "10", made from Python's small integers as the
  machine's own constants (its digits and its base), as `READ_POW` is a number; every value made from them is the
  learner's routines'.
- **A constant's doublings** are named from the routine's output digits, read back as text.
- **ROUNDT's slot** for a count is found by comparing count digits; its rows are copied within the GPU's buffers.

### 11. Cost

`cost.py` (`checks/logs/cost.log`; the billing report's complete hours, read at 16:20 PDT):

| app | what | billed |
|---|---|---|
| ap-0ZshSUOtWIDx9OIvxJgRnN | the 13 tiny cores (B200s): the file on 11 tasks, the READ_POLY = 4 arm on 2 | $10.25 |
| ap-j7cQbKXviC365iqEDT16mS | addendum 1's arm: the file with phase C's ROUNDT, minus1's tiny core (B200) | $1.90 |
| ap-HXhUisJ4o3TnnTnNRV13uI | H1's official run (H100) | $1.00 |
| ap-fFB4KczSeFGdMQPiPm0Vlb | H1's seeds 1–7 (H100s) | $8.92 |
| ap-GXIkI4b2vi2WJEZxwQwSRT | the family and the held-out pair, ten tasks (H100s) | $18.13 |
| all | | **$40.19**, with about $3 more for the last three runs' minutes in the 23:00 UTC hour (about 42 H100-minutes) |

**Limits.** Soft $120, cap $180. The month stood at $4,096 when the first app started (14:58 PDT) and at $4,136 billed
afterwards (16:21 PDT). `spend.py 4750` passed before every launch.

**The spend watchers** (`logs/watch_*.log`) watched every app of the phase:
- They stop everything once the estimate plus one more hour of the containers still running reaches $180: $98.75
  with the thirteen B200 tiny cores, and $105 to $126 with the twelve containers of the H100 runs.
- They stop any app at $90.
- The estimate never came near either. Its last reading, at 16:17 PDT, was about $46 with the hour under way.

Every app has stopped, and so has the last watcher.

### 12. Files and how to rerun

Everything is in `cpu_first/integration/legal_wider/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_wider.py` | the file, md5 `59060b5cd7a7b536e25302e1d5be7ef7` |
| `research/submission_stage3_wider_READ_POLY-4.py` | the arm with one more Horner step (`make_wider.py READ_POLY=4`), md5 `9fd20cca` |
| `research/submission_stage3_wider_no-roundt.py` | addendum 1's arm, the file with phase C's ROUNDT (`make_wider.py --skip=roundt`), md5 `c764e681` |
| `make_wider.py` | the builder (§2), with `--skip=GROUP` for research builds |
| `PREREG.md`, `PREREG_ADDENDUM1.md` | frozen; `checks/logs/prereg_sha256.txt` holds their hashes and the code's |
| `heldout.json`, `checks/draw_heldout.py` | the drawn pair and how it was drawn |
| `tasks_wider.py` | the pair's data (poly_A_K_B_C, H1's generator) |
| `core_f.py`, `modal_wider.py` | the tiny core and its launcher (app and Volume `legal-wider`) |
| `make_runner.py`, `modal_runner_wider.py` | the shared runner with plusy, three2 and the pair in its image and task list |
| `watch_spend.py` | the spend watcher: the cap less one hour of the containers still running |
| `cost.py` | §11 |
| `plans/family.json` | the ten family and held-out jobs |
| `checks/wider_local.py` | the readings and their posing, and ROUNDT on phase B's rounds (MODE=roundt) |
| `checks/roundt_same_local.py`, `checks/roundt_buffers.py` | ROUNDT old against new; ROUNDT's buffers and the state dict |
| `checks/compare_local.py`, `checks/compare_cores.py` | combined_sp2's local checks against the family file's; a tiny core against phase C's |
| `checks/review_lines.py` | REVIEW's line numbers in the family file, and this file's (§8) |
| `checks/ladder_wider.py`, `checks/summarize_family.py` | a tiny core's answer programs on both ladders; the H100 records |
| `checks/logs/`, `logs/` | the checks' logs; the spend watchers |
| `records/` | the tiny cores (`t_wider_<task>`, `t_wider4_<task>`, `t_wider_oldroundt_minus1`), H1's eight records, the family's ten |

From `cpu_first/`, with `PY=~/github/sotaku/venv/bin/python` and `PYTHONDONTWRITEBYTECODE=1` exported:

```bash
$PY integration/legal_wider/make_wider.py
$PY integration/legal_wider/make_wider.py READ_POLY=4
$PY integration/legal_wider/make_wider.py --skip=roundt
$PY integration/legal_wider/checks/draw_heldout.py
$PY integration/legal_wider/make_runner.py
$PY integration/legal_wider/checks/wider_local.py
TASKS=poly_8_3_8_3,poly_2_3_9_9 $PY integration/legal_wider/checks/wider_local.py
$PY integration/legal_wider/checks/roundt_same_local.py
COMBINED_FILE=integration/legal_wider/submission_stage3_wider.py PYTHONPATH=. $PY integration/legal_wider/checks/roundt_buffers.py
COMBINED_FILE=integration/legal_wider/submission_stage3_wider.py PYTHONPATH=. $PY integration/combined_sp2/checks/local_combined.py policy grad machines isa designed buffers replay climb climb_tiny fix book
python3 integration/legal_wider/checks/compare_local.py
$PY integration/ablation_s3/spend.py 4750
$PY -m modal run --detach integration/legal_wider/modal_wider.py --envs '[{"TAG": "t_wider_h1", "TASK": "h1"}, ...]'
python3 integration/legal_wider/checks/compare_cores.py h1
TASK=plus3 $PY integration/legal_wider/checks/ladder_wider.py integration/legal_wider/records/t_wider_plus3.json
$PY -m modal run --detach integration/legal_wider/modal_runner_wider.py --mode official --file integration/legal_wider/submission_stage3_wider.py --name legal_wider_official --out integration/legal_wider/records
$PY -m modal run --detach integration/legal_wider/modal_runner_wider.py --mode seeds --file integration/legal_wider/submission_stage3_wider.py --seeds 1,2,3,4,5,6,7 --hardware h100 --name legal_wider --out integration/legal_wider/records
$PY -m modal run --detach integration/legal_wider/modal_runner_wider.py --mode plan --plan integration/legal_wider/plans/family.json --out integration/legal_wider/records
python3 tools/verify_records.py integration/legal_wider/records/legal_wider_official.json=integration/legal_wider/submission_stage3_wider.py integration/legal_wider/records/legal_wider_h100_seed{1,2,3,4,5,6,7}.json=integration/legal_wider/submission_stage3_wider.py
python3 integration/legal_wider/checks/summarize_family.py
python3 integration/legal_wider/checks/review_lines.py
$PY integration/legal_wider/cost.py
```

(The tiny cores' full list of tasks is in `PREREG.md`, Runs; the arms add `"FILE": "research/..."` to an entry.)
