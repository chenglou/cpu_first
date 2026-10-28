# The deep file (phase E): pre-registration

Written 2026-10-27 19:26 PDT, before any GPU run of this file. The file was fixed (md5 below, 19:25:10 PDT) before the
six fresh tasks were drawn, and has not changed since. DEEP.md §1 records the time and hash this was frozen with.

## The question

The wider file (phase D, legal_wider/WIDER.md) learns H1 and 7 of the 8 family tasks, but not three2 (3y² + 2), and
neither of its 2 held-out tasks (8y³ + 8y + 3, 2y³ + 9y + 9: they need a fourth Horner step). The brief, phase E,
with the user's update:
1. The reader's fourth step. Does four steps cover every task in the space y ← a·y^k + b·y + c mod N (k 2, 3; a
   1..9; b, c 0..9)? The reader's laws per try, and its time.
2. A bigger work budget, still inside the hour: set so that the depth-4 tiny cores find their rounds; the worst case
   bounded from measured rates, with a margin; H1 the same line for line.

Then: recertify H1 (the official H100 run, seeds 1–7); the 8 family tasks, phase D's pair (no longer held out) and
six fresh tasks drawn at random from the space, one H100 each, the file byte for byte. Only the six test generality.
Each is reported on its own, with the count out of six; "it learns tasks drawn at random from this space" only if all
six are learned.

## Seen before the design was fixed (B200 tiny cores, CPU checks)

1. **The reader covers the space** (`read_space.py`, `checks/logs/read_space.log`; the reader with only READ_POLY =
   4): four Horner steps reach every a·y^k + b·y + c of the space, and that reader read all 1,800 tasks from the
   first 141 of H1's answer rows of T = 1 (a full-size run's first try): 200 to 60,087 laws a try (median 6,726),
   0.40–5.21 s on a B200 (median 1.54); the fewest rows a reading rested on, 6 (READ_MIN is 4).
2. **A bigger budget alone brings no depth-4 round** (`records/x_deep_*`, that reader, no climb budget): phase D's
   pair and three2 climbed to levels 9–10 (7.4e9 to 9.2e9 units of work) with no round, the last levels picking
   nothing.
3. **three2's failure in phase D was a check gap** (`checks/try_round.py`, `checks/accept_gap.py`): its learned
   window 3·x·o + 2 multiplies by o's last digit (wrong on 400 of 400 full-size rows). A reading's own law is computed
   by the reader, whose multiplication needed both factors to fit half the places: at a verify size of k digits,
   3·(a·b) was computable only where a·b < 10^k; on the Acceptor's 60 rows a size, 38 rows at one digit, 5 at two,
   none at three or four, so a routine multiplying by b's last digit passes when those 5 draws are kind. Counting
   every row where the product fits the places: 38, 16, 25 and 18.
4. **Reduced windows are not enough for depth 4** (`records/xc_*`): with each window also posed reduced mod N,
   8y³ + 8y + 3's climb learned (8a·b + 8) mod N and (a·b + 3) mod N, each right on 400 of 400 full-size rows, but no
   2-call round passed: the first routine leaves x² in X and 0 in Y (learned routines use X and Y as scratch), and
   the round search routes a call's inputs only from the registers the call before left. x is gone.
5. **The design's arms** (B200 tiny cores, no climb budget; `checks/summarize_cores.py`): the check fix alone
   (`xb_three2`) learns three2 after level 7 at 4.88e9 units; with every piece below (`xd_*`), H1 is phase D's unit
   for unit (all 52 units, line and work); plusy, rho and three2 find their rounds after levels 4, 5 and 5 (1.29e9,
   2.85e9, 3.94e9); 2y³ + 9y + 9 finds a round with x kept in S after level 7 at 1.07e10 and its answer programs at
   1.25e10 in all; 8y³ + 8y + 3 had no round by level 7 at 1.31e10, past the 1.3e10 budget (its round's first window
   not yet learned). Every answer program these arms found is right on every rung of both ladders
   (`checks/logs/ladder_x*.log`: three2 twice, rho, plusy, 2y³ + 9y + 9).
6. **The machines read a count loop's count once** (`_run_count`, `SteMachine._count`, `OuterMachine`): a round that
   writes S inside `C.S[...]` does not change the number of rounds.
7. **Local checks of the file** (`checks/logs/local_deep.log`, `checks/compare_local.py`, `checks/logs/pose.log`):
   combined_sp2's local checks are the wider file's part for part, but the size, the fix part's file path and the
   law book's diff region (170 lines against 165: a docstring cut); every posed law agrees with Python's integers (186
   checks); H1 and the cube pose as in phase D, line for line; plusy, rho, minus1 and plus3 pose one more law,
   (a·b) mod N; the pair is read in Horner's form, ((x·x)·8 + 8)·x + 3 and ((x·x)·2 + 9)·x + 9.

## Design (`make_deep.py`, from the wider file's committed bytes at 80082c0, md5 `59060b5c` asserted)

| group | what |
|---|---|
| doc | the docstring's first line names DEEP.md and make_deep.py |
| reader | READ_POLY = 4: x's polynomials up to four Horner steps |
| check | a reading's own law (`read_target`) and x's polynomials (`LawReader.poly`) multiply wherever the product fits the places, not only where both factors fit half of them |
| horner | x's polynomials by Horner's steps only, x·x first as before, no "+ x": a·x³ + b·x + c is read ((x·x)·a + b)·x + c, never (x³ + x)·a |
| reduced1 | each of a reading's windows, and each single operation of it, also posed reduced mod N (`w.mod.…`), after the windows; one the same as a window is posed once |
| keepx | when the round's search by calls finds none and the law read uses x again above a deeper operation, the same search once more with x kept in S: the round starts `LDA.X STA.S`, and S is a routed input |
| budget13 | CLIMB_WORK = 13,000,000,000 (was 4,300,000,000); SEARCH_WORK unchanged |
| room | seven docstrings that a design doc explains cut to their first words and the pointer |

The file: `submission_stage3_deep.py`, md5 `56acca7358c74c65c72b5f33e3b9aafa`, 261,679 bytes (465 under the limit);
the contest's source check passes.

**Past the brief.** The brief asked for the fourth step and a bigger budget; check, horner, reduced1 and keepx go
further, because a budget alone brought no depth-4 round (above). The coordinator was told the evidence and this
default at 19:05 PDT, with a deadline of 19:25 for an objection; none came, and the file was fixed at 19:25:10.

## The bound on training time

From phase D's H100 records (`checks/rates.py`, `checks/logs/rates_phaseD.log`), every fixed cost at its measured
maximum: the climb starts by 136 s; from the answer program to the freeze (199 steps) at most 283 s; from the freeze
to step 20,000 at most 190 s. That is 609 s. A run that learns spends at most CLIMB_WORK on its climb and its answer
program's search together (both stop at the budget); the worst case spends all of it before its round.

Rates, seconds per million units of work:
- phase D's long H100 climbs (3e9 units or more, with their answer programs): 0.101–0.149 (minus1 the slowest);
- this phase's B200 tiny cores with x kept: 2y³ + 9y + 9, 0.092 over 1.25e10 (its climb and answer search);
  8y³ + 8y + 3, 0.104 over its first 1.23e10; B200 to H100 × 1.18 (phase D's minus1: 0.126 on its B200 core, 0.149
  on the H100).

**The bound, CLIMB_WORK = 13e9:** 609 s + 13,000 × 0.149 × 1.25 = 609 + 2,421 = **3,030 s**, 570 s inside the hour
(phase D's slowest long climb, plus 25%). With this phase's own climbs (0.104 × 1.18 = 0.123, plus 25%): 609 + 2,000
= 2,610 s. Not covered: a whole climb at the round searches' rate (0.78 s per million on the H100, the slowest kind
of work); no measured long climb comes near it, and the levels' round searches are inside the climbs measured. A run
that does not learn trains to the hour, as phase D's pair did, whatever the budget.

## The six fresh tasks (drawn and frozen here)

- **The space:** y ← a·y^k + b·y + c mod N, T times, k in {2, 3}, a in 1..9, b and c in 0..9 (1,800 tasks), less the
  8 tasks used so far that are in it (H1, the cube, rho, plus3, plusy, three2, and phase D's pair): 1,792 tasks.
- **The seed:** not chosen. It is the repository's HEAD commit when phase E began, `cb623f9`, read as a hexadecimal
  number (213,263,353; `checks/logs/head_at_start.txt`, written 17:48:55 PDT). `checks/draw_heldout.py`:
  `random.Random(seed).sample(space, 6)` (Python 3.9.6), run at 19:25 PDT, after the file's md5 was recorded
  (`checks/logs/file_fixed.txt`, 19:25:10; `checks/logs/draw_heldout.log`).
- **The draw:**

  | task | answer | Horner's form, as this file reads it |
  |---|---|---|
  | `poly_4_2_3_9` | y ← 4y² + 3y + 9 mod N | (4x + 3)·x + 9 |
  | `poly_9_3_7_0` | y ← 9y³ + 7y mod N | ((x·x)·9 + 7)·x |
  | `poly_1_2_0_6` | y ← y² + 6 mod N | x·x + 6 (the first two stages: a gap constant) |
  | `poly_5_2_2_6` | y ← 5y² + 2y + 6 mod N | (5x + 2)·x + 6 |
  | `poly_1_3_5_1` | y ← y³ + 5y + 1 mod N | (x·x + 5)·x + 1 |
  | `poly_1_2_0_7` | y ← y² + 7 mod N | x·x + 7 (the first two stages: a gap constant) |

  Data: `legal_wider/tasks_wider.py`, H1's generator and configuration, the same prompts as H1 row for row.

## Runs

1. Tiny cores (B200, `modal_deep.py`, `core_f.py`): the file on H1, the eight family tasks, phase D's pair and the
   six (17), the file's own budget; every answer program found on both ladders (`legal_wider/checks/ladder_wider.py`).
   H1 against phase D's tiny core (`legal_wider/checks/compare_cores.py`). The reader on the space again, with this
   file (`read_space.py`).
2. H1: the official H100 run (seed 74) through `modal_runner_deep.py`; `tools/verify_records.py`; then seeds 1–7.
3. The 16 tasks, one H100 each, seed 74, the file byte for byte (`plans/runs.json`).
4. Learned means certified: T = 64 on both ladders, every rung 768/768. Per task: learned or not, T on each ladder,
   the test, training time, why for each failure.

## Predictions

| what | prediction | confidence |
|---|---|---|
| H1's tiny core | phase D's, unit for unit (line and work) | high |
| the cube's | phase D's picks, library, rounds and answer programs; more work (its law uses x again, so the search with x kept runs where the plain one finds none) | medium |
| plusy's, rho's, minus1's, plus3's | each finds its round and answer programs (one more law posed, (a·b) mod N; rounds may differ from phase D's) | medium-high |
| ten's, tenrho's, three2's | each finds its round and answer programs | medium |
| 2y³ + 9y + 9's | finds its round with x kept and its answer programs within the budget (its arm core: 1.25e10 of 1.3e10) | medium |
| 8y³ + 8y + 3's | no round within the budget | medium |
| every answer program found | 768/768 on every rung of both ladders | high |
| H1, official run | certifies; its climb phase D's line for line (seconds masked); training 450–800 s | high |
| H1, seeds 1–7 | 7 of 7 certify | high |
| the eight family tasks and phase D's pair at full size | as their tiny cores | medium |
| y² + 6, y² + 7 (fresh) | learned, as plus3 (their constants from 1, 2, 4) | medium |
| 4y² + 3y + 9, 5y² + 2y + 6 (fresh) | learned: (a·x + b), then (t·x + c) mod N with x kept; many constants to make | medium-low |
| y³ + 5y + 1 (fresh) | learned: (x·x + 5) mod N, then (t·x + 1) mod N with x kept | low |
| 9y³ + 7y (fresh) | not learned within the budget (four steps and the constants 9 and 7) | medium |
| the six | 2 to 4 learned; not all six | medium |
| the reader on the space, this file | every task read at the first try's 141 rows | medium-high |
| learned runs' training | under 2,000 s each (the bound: 3,030 s) | medium |
| money, phase E | $150–250 | medium |

## Money

Soft $400, cap $550 (the user's update). `integration/ablation_s3/spend.py 4750` before every launch (the month:
$4,140 when phase E began). One spend watcher (`watch_spend.py`) over every app of phase E: it stops everything
once the estimate plus one more hour of the containers still running ($6.25 a container-hour) reaches $550, and any
app at $275. Every app is stopped when its runs end.
