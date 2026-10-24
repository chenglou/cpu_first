# Integration: the tournament, the general lessons and the library in one H1 model

*Written 2026-10-22. Everything is in `cpu_first/integration/`; run records in `integration/records/` (the evaluator's
runs) and `integration/lib/records/` (the climbs). Nothing committed, nothing submitted, no certified file touched (the
builders assert the md5 of every file they read).*

## In plain words

<!-- PLAIN -->
**What the one model learns.** A prompt gives a modulus N, a start x and a count T; the answer is x squared T times,
keeping the remainder after dividing by N each time. The stage-3 model starts knowing none of this. In one training run
it learns, in order:
1. to read: which tokens are digits, which digit each one is, where N, x and T start and which way they run;
2. three small digit tables: subtracting two digits with a borrow, adding two with a carry, and "a + x times h + k" for
   four digits. Nobody gives it the tables: it finds them from facts about its own training rows (x is always smaller
   than N, so "x minus N" must borrow) and from laws (subtracting and adding back gives the start again; multiplying by
   h + 1 is multiplying by h and adding once more);
3. a library of small programs over a general digit machine (five registers of digits, a pointer, 32 single-digit
   instructions, and loops). It practises on small problems made of its own rows' digits, each problem one law (compare
   x with N; take N away from x if that does not go below zero; add N; and so on). It searches for the shortest program
   that solves each one, turns the pieces that keep coming back into new single instructions, and keeps every solved
   problem as a reusable routine. Out of that come, one at a time, the pieces of long division (subtract digit by
   digit; subtract only if it fits; repeat; bring the next digit down) and of school multiplication (a row of digit
   products; the rows added up);
4. the round itself: it tries short combinations of its routines on the training answers, and keeps the ones that
   reproduce them. For H1 that is "copy x, multiply, divide": x squared, then the remainder.
Then it runs that program T times, digit by digit, and writes the answer.

**What changed from the earlier models.** The certified models all had a hand-written machine: one register of nine
digits, eight steps of nine subtractions, a fixed way of multiplying. Training chose among options plugged into it.
Stage 3 has no such machine. The machine it runs was put together in training out of programs it found.

**What is still hand-made** (§4): the digit machine's instruction set and its sizes; the reader's grammar (stage 1's);
the law library and which laws become practice problems, in which order, with each problem also posed with two input
registers swapped, and one of them (mac) posed so that the multiply's row is a problem of its own; the search and
merging rules (their state caps are measured); where the round is looked for (short combinations of routines, on the
answers of rows with T = 1); and the countdown over T. Removing each part in turn, on two seeds
([ABLATION_S3.md](ABLATION_S3.md)), shows what the result needs: two of the instructions (keep-or-discard on the
borrow, bring the next digit in), the problems that define long division's step, the swapped registers, mac's posing,
routines, and the tie-break toward faster programs. The order, the merges, skipping solved twins and the T = 1 rows
only save time. The hand-picked list itself is not needed: a learner that poses its own problems from a law book of 11
laws, picking by one general rule, finds the same nine pieces and certifies ([SELF_PRACTICE.md](SELF_PRACTICE.md)).
The book and the rule are designed instead.

**How each stage went.**
- **Stage 1** (the tournament and the general lessons in one file): **certified** on the untouched official manifest on
  an H100 (1,861 s of training, 56 s of evaluation) and on 8 CPU seeds (1,441-3,127 s of training each). With no change
  to the file it also learns y^2 + 1, 10 y^2, cubing, Montgomery's squaring, 10 y^2 + 1 and y^2 + 3; y^2 - 1 fails (no
  menu option fits). It also learns M4, the practice tier where every prompt has T = 8, which the general file did not
  (0.8%): its answers now come from the rows with the batch's smallest T instead of a constant "T <= 2".
- **Stage 2** (the library inside one run, with learned tables, tables first): the tables are learned by laws for 6 of
  8 seeds (2 never pass their own checks, so their climb never starts). The climb as first built (A) learns the
  comparison, the conditional subtraction, the adder, the chain, the multiply's row and the long-division step, but
  never the multiply. Two reasons, both found and fixed (§2.1):
  - its search tested only the first 200 candidate programs, which can all belong to a few wrong answers;
  - no practice problem ever produced the "add this row into the running total" piece the multiply's loop needs.

  With those fixes and the review's rules (routines first, twins skipped, measured caps, stalls logged: variant C), the
  climb finds the multiply.
- **Stage 3** (no skeleton, no menus): **certified on the untouched official manifest** on an H100: T = 64 on both
  ladders, every rung 768/768, every split exact. The final file, s3i, **certifies on every seed** on the official
  manifest on H100s (74 and 1-7: the round by 948-1,099 s, the freeze by 982-1,149 s, under a third of the budget,
  1,133-1,350 s of training). The file before it, s3h, certified every seed too, its freeze by 1,341-2,086 s; s3i
  trains step for step as s3h does, with phase D's countdown and two parts of the search made faster (§3). s3g
  certified 5 of 8; s3h fixed its three remaining failures (§3): an acceptance test that rejected the right round for
  some pool sizes, a NaN in phase D, and a climb too slow for the slowest hosts (s3h's runs reach the freeze 2.0-2.4
  times sooner than s3g's). In one training run the model reads, learns its digit tables from laws, climbs its
  library, and discovers the round:
  - the multiply, as a loop of its own accumulating row;
  - the reduction, as a loop of its own long-division step;
  - x^2 mod N as "copy x, multiply, reduce", found at step 671 (seed 74), by 1,095 s (s3h: 1,232 s; s3g: 2,989 s).

  Getting there took the A/B's variant C (test every candidate's result; pose the multiply's row as a practice problem
  of its own), two fixes found in-run (Occam's ties to the faster program; skip only the twins), and practising the
  pieces first (the multiply after its row, the reduction after its chain) with an exact cache across levels. The
  first in-run discovery, without the last two, needed 8,552 s of climbing and certified only on a relaxed budget. (On
  the final file's faster engine, the order alone saves only about 280-320 s: [ABLATION_S3.md](ABLATION_S3.md).)
  Certifying every seed then took an exact acceptance test for the round, a guard in one lesson, and a search engine
  that runs every candidate program's instructions in lockstep on the GPU (the same search, more than twice as fast).
  The final file's margin (every seed frozen by 1,149 s) came mostly from the countdown that checks the answers: its
  round program now runs on all rows at once, replayed from a CUDA graph, with the same results (seed 6's steps
  600-800: 95 s instead of 1,028 s); two refinements of the search reach the round sooner too (the climb alone, on
  two H100s the same day: 964 s against 1,133 s).

  The final file has two weak checks, found by the ablation: in about 3% of runs its reduction check accepts a
  repeated subtraction that is right only for small quotients, and such a run cannot certify; and its conditional
  subtraction's check never tests x between N and 2N. None of its eight seeds drew either. A fixed copy replays all
  eight runs field for field (§3).

**Is Bar 2 cleared?** Partly (§6). No hand-written algorithm runs any more, and the model that answers was discovered
in training and certifies on the official budget. But the algorithm's decomposition still lives in the training
recipe: in two of the instructions, and in how the practice problems are posed (the step's definitions, the swapped
registers, mac). Which problems, and in which order, matter less than it seemed: the order only saves time, and a
designed law book with a general choosing rule can stand in for the hand-picked list.
<!-- /PLAIN -->

## 1. Stage 1: the general lessons and the tournament

### 1.1 What the file is

[`submission_stage1.py`](submission_stage1.py) is built by [`make_stage1.py`](make_stage1.py) from three certified files:
the tournament file ([TOURNAMENT.md](../tournament/TOURNAMENT.md): MLP digit cells, the division's chain at free
attention's width, the division's wiring as tournaments, the link and loop laws, the moved-up rows), the general file
([LESSONS.md](../general_lessons/LESSONS.md): the law library, the mined facts, the answer-chosen lesson sets, the shared
round menu) and their common ancestor ([`submission_wide_anchored.py`](../wider_menus/submission_wide_anchored.py)). The
builder runs `git merge-file` of the two over the ancestor (14 conflicts), resolves each in the script, replaces three
methods whole and patches a few lines, so the file's difference from either parent is the script.

What the merge decided:
- **Every lesson set owns a whole tournament.** The general file's two lesson sets ("below", long division; "lowzero",
  Montgomery's reduction) each hold a copy of the tournament's four blocks ("link", "gate", "step", "read"), taught
  only by the set's own lessons. The selector over the sets and the round menu are trained by the answers alone, as in
  the general file.
- **The link law comes from the set's predicate** (`red.link`, new in the law library): below, one link from x signals
  "below N" and takes N to 0 (the tournament's link law); lowzero, one link from 10 x and from N adds N with no carry
  out. The ladder (the pairs) is the general file's per-predicate ladder.
- **The gate table's two laws** filter the pairs' gate settings: the tournament's loop law (`loop.count`, now in the
  library) and the general file's gated-subtraction law (`csub`). Each alone pins the gate to "commit on no borrow".
- **The moved-up rows (CHAIN_NORM) are the below set's only.** A first build moved the lowzero set's rows too and
  broke its own ladder at the truth (0/1 on the ladder, 0.55 on the step law): a register moved up has a lowest digit 0,
  and N·10^l is not coprime to 10, so it is no instance of the lowzero predicate. The lowzero set therefore keeps the
  top-place gap TOURNAMENT.md describes, which only matters on data where the answers choose it.
- **Every block is pruned to its active combinations**, the step and reading blocks included (the tournament file
  pruned only the chain's). Combinations more than 12 nats behind already counted as failing, so the lessons are the
  same; it keeps the second lesson set's I/O lessons affordable.
- **The answer rows come from the data.** The general file read the answers of rows with one-digit T ≤ 2
  (`REAL_MAX_T`), an H1-shaped constant that left M4 (every T = 8) with no answer. Here they are the rows with the
  batch's smallest nonzero T, compared the way the reader compares numbers, capped at 96 row-rounds of work
  (`answer_rows`). On H1 that is the T = 1 rows. general_lessons v2 made a similar fix that keeps `REAL_MAX_T = 2` as a
  floor; this one has no such constant.

### 1.2 Results

<!-- S1_RUNS -->
| record | task | device | seed | max T, seen / unseen moduli | rungs | splits | training s (frozen at step) | evaluation s | what it chose | md5 |
|---|---|---|---|---|---|---|---|---|---|---|
| s1_official | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1861 (716) | 56 | below (lowzero:0.0014,below:0.9986); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed74 | h1 | cpu | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1490 (724) | 439 | below (lowzero:0.0018,below:0.9982); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed1 | h1 | cpu | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3127 (813) | 579 | below (lowzero:0.0008,below:0.9992); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed2 | h1 | cpu | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1871 (835) | 596 | below (lowzero:0.0008,below:0.9992); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed3 | h1 | cpu | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1957 (857) | 437 | below (lowzero:0.0012,below:0.9988); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed4 | h1 | cpu | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 2622 (849) | 678 | below (lowzero:0.0010,below:0.9990); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed5 | h1 | cpu | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1441 (725) | 419 | below (lowzero:0.0010,below:0.9990); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed6 | h1 | cpu | 6 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1482 (705) | 541 | below (lowzero:0.0015,below:0.9985); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
| s1_cpu_seed7 | h1 | cpu | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1893 (789) | 434 | below (lowzero:0.0007,below:0.9993); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | cfd7b61d |
<!-- /S1_RUNS -->

### 1.3 Acceptance: other formulas with zero per-task changes

The litmus and general-lessons formulas, each run on H1's prompts with the task's answers (litmus/tasks.py's
generators; general_lessons/tasks_general.py's held-out ones), the file byte for byte, at most 1,500 steps on an H100,
seed 74; M4 on its own manifest with a relaxed budget (3,600 s, 1,500 steps).

<!-- ACCEPT -->
| task | general lessons (LESSONS.md, the general file unchanged) | stage 1 (this file unchanged): max T seen / unseen | stage 1: what it chose | stage 3: learned? (H1: the evaluator's run; the others: the round's search over the routines the H1 run discovered, cost <= 5 as the file / 6) | the round's search with LIBRARY r11's routines: cost <= 5 / 6 |
|---|---|---|---|---|---|
| H1: x^(2^T) mod N | certified (64 / 64) | 64 / 64 | below (lowzero:0.0014,below:0.9986); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | certified (64 / 64, the official manifest): round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | `T.add STA.Y T.mul STA.X T.reduce` / - |
| y^2 + 1 | certified (64 / 64) | 64 / 64 | below (lowzero:0.0014,below:0.9986); rmul=one radd=1 rcube=none; below's chain: repeated subtraction | no round / - | none / none |
| 10 y^2 | certified (64 / 64) | 64 / 64 | below (lowzero:0.0013,below:0.9987); rmul=ten radd=0 rcube=none; below's chain: repeated subtraction | no round / - | none / `T.shl STA.Y T.mul STA.X SHR.X T.reduce` |
| x^(3^T) (y^3) | certified (64 / 64) | 64 / 64 | below (lowzero:0.0014,below:0.9986); rmul=one radd=0 rcube=times_x; below's chain: repeated subtraction | no round / - | none / none |
| Montgomery: y^2 10^-8 mod N | certified (64 / 64) | 64 / 64 | lowzero (lowzero:0.9988,below:0.0012); rmul=one radd=0 rcube=none; lowzero's chain: montgomery, trial multiples (the register's digit tested) | no round / - | none / - |
| 10 y^2 + 1 (held out) | certified (64 / 64) | 64 / 64 | below (lowzero:0.0015,below:0.9985); rmul=ten radd=1 rcube=none; below's chain: repeated subtraction | no round / - | none / - |
| y^2 + 3 (held out) | certified (64 / 64) | 64 / 64 | below (lowzero:0.0015,below:0.9985); rmul=one radd=3 rcube=none; below's chain: repeated subtraction | no round / - | none / - |
| y^2 - 1 (held out) | not certified (None / None) | None / None | below (lowzero:0.4188,below:0.5812); rmul=one radd=7 rcube=times_x; below's chain: repeated subtraction | no round / - | none / none |
| M4 (practice tier, every T = 8) | not certified (None / None) | 64 / 64 | below (lowzero:0.0011,below:0.9989); rmul=one radd=0 rcube=none; below's chain: repeated subtraction | no: the round's search reads T = 1 answers, and M4 has none | - / - |
<!-- /ACCEPT -->

Stage 3's acceptance is read off the round's search alone. Everything before the round (reading, cells, climb) uses only
the rows' N and x, so it is the same climb on every task, and the round's search with the routines the H1 run
discovered (`lib/s3d_lib.json`) decides each task. At stage 3's cost bound of 5, only H1 has a round (exhaustive,
`records/proxy_s3d_lib_cost5.log`). 10 y^2 is one step further: `LDA.X SHL.A STA.Y T.mul~ STA.X T.reduce` (cost 6)
reproduces 10 x^2 mod N on 200 of 200 H1 rows (checked directly; the cost-6 search over every task was stopped to free
an H100). y^2 + 1, y^2 + 3, y^2 - 1 and 10 y^2 + 1 need a constant, which no routine or register holds. M4 gives the
round's search no answers at all: it reads rows with T = 1, an H1-shaped constant this file keeps.

## 2. Stage 2: the library inside one training run, with learned cells

<!-- STAGE2 -->
**The question.** LIBRARY.md rediscovered the skeleton's parts by searching small digit programs and merging what
recurred, but offline, with exact cells and a curated task list; its in-run prototype stalled before the multiply and the
reduction. Stage 2 asks for the same inside one training run: practice problems made from the rows' own numbers and the
law library, the search over the primitive digit instructions, the merges, and learned cells, with the tables staged
first (laws), then the routines.

**What was built** (`integration/lib/`):
- `tdsl.py`, `tfast.py`: the library's machine and breadth-first search on torch tensors. `tfast` applies many items at
  once: rows are (item, state) pairs; a token runs once on every row that applies it, and a loop runs its iterations on
  all of its rows together, the body's elements grouped position by position. On the library study's own problems the
  two give the same 1,704 item classes in the same order and the same programs as `library/search.py`
  (`test_tdsl.py`, `test_tfast.py`); the batched version enumerates items 4x faster even on a CPU and is what makes loop
  bodies of cost 3 (the multiply's) affordable.
- `cells.py`: the three cells from random, in order, by laws only: the subtraction by the anchors' certified recipe
  (ANCHORS.md R2), stopped only when its own checks hold on the batch and on every entry (the order laws on all
  entries, sub.bijective) for 50 steps; the addition by sub.inverse on every entry ((a + b + c) - b - c = a, the target
  read off the snapped subtraction); the multiply-accumulate (a + x h + k) by mul.zero and mul.ladder on every entry.
  A first stop rule (the batch's checks only) accepted a table with 1 wrong entry of 200; the addition's law exposed it
  (that column was no bijection), which is why the table-wide checks are the gate.
- `libcore.py`: one torch-only module (no numpy) with the programs, BPE, the machine, the batched search, the practice
  tasks, the climb and the round search; stage 3 inlines it, and the stage-2 climbs run it as is.

**The practice problems, and which are designed.** Each is one law applied to digits the rows give: N's top w digits N'
and x's digits at the same places x' (x' <= N' by the mined fact x < N), another row's x' as Q, the digits 0-9 run over
the probes (nothing made up), the targets either a fact (a copy of x', a 0, a borrow bit) or the machine's own result of
a program the climb found earlier:

| task | law (LAWS) | target | built with | derived or designed |
|---|---|---|---|---|
| lt | sub.order + the mined x < N | the borrow of x' - N' (1 iff the strings differ), of N' - x' and x' - x' (0) | - | derived |
| modx | red.identity, red.zero | x' mod N' = x' (0 if equal); N' mod N' = 0 | - | derived |
| add | sub.inverse, add.identity | (x' + N') mod N' = x' and x' + N' does not borrow against N' (inverse checks); x' + 0 = x' | lt, modx | derived |
| modB | red.step (below's ladder) | (x' + q N') mod N' = x', q = 0..9 | add | derived (the general file's template) |
| mrow | mul.ladder | x' d = d additions of x', d = 0..9 | add | derived |
| mac | mul.ladder with an accumulator | x' d + N' (A, B); in C and the certified stage 3, A + x' d, the running sum in A | mrow, add | **designed**: added so the row has a second use (the library study's r14 fix); C's addend makes its program the multiply's accumulating row (§2.1) |
| mul | mul.distribute, as additions | x' Q = Q additions of x' | add (the certified stage 3: add, mac) | derived |
| step | red.step's step law | (r B + d) mod N' with the found chain, r = x' | modB | derived (the template's step law) |
| reduce | red.invariant | (x' + Q N') mod N' = x' | add (the certified stage 3: add, modB) | derived |

Designed content around them: which laws become tasks, the helpers' order (`NEEDS`), each task's operand width, the
top-part construction, **the register variants** (every task also posed with X and Y swapped: one task per concept made
every merge rule stall in LIBRARY.md §5.1; the swap gives each concept two tasks without picking them), the probe counts
and the settings (10, 1) and (10, 2), acceptance on 60 fresh rows at widths 1-3 (for the reduction, too weak in about
3% of runs: §3). Not designed: the square, x^2 mod N or "the round" are no practice task; the round is searched by
composition against the answers (§3).

**The climb** (`libcore.Climb`, one unit of work per call): every task whose helpers are found is searched (cost <= 7,
3M stored states a level, loop bodies up to cost 3); a failed search tries the primitives alone at 16M (a program that
uses no merge, such as the cost-6 multiply row; a task whose fallback failed once is not tried again, since that search
never changes); at the end of a level every accepted spelling goes to BPE (pairs used by two tasks, ties toward the pair
standing for the most primitives), each task then keeps its simplest program (Occam), every newly solved task becomes a
routine `T.<task>` that restores the pointer it was called with, and the round is searched whenever new routines
appeared. That is climb A; the certified stage 3 runs variant C's climb with the in-run fixes (§2.1, §3).

**Results.**
<!-- /STAGE2 -->

<!-- CLIMBS -->
| climb | cells | device | levels | tasks solved (in order) | the round found | seconds |
|---|---|---|---|---|---|---|
| c4_learned_s1 | learned (seed 1): {'sbb_d': '184/200', 'sbb_c': '200/200', 'adc_d': '184/200', 'adc_c': '200/200', 'mul_lo': '7624/10000', 'mul_hi': '9636/10000'} | cpu | - | the cells never passed their own checks: the climb waits | - | 71.6 |
| c5_exact_h100 | exact | cuda | 8 | lt (L0), lt~ (L0), modx (L1), modx~ (L1), add (L2), add~ (L2), modB (L3), modB~ (L3), mrow (L3), mrow~ (L3), mac (L4), mac~ (L4), step (L4), reduce (L5), reduce~ (L6) | no | 15661.1 |
| c5_learned_s0_h100 | learned (seed 0) | cuda | 8 | lt (L0), lt~ (L0), modx (L1), modx~ (L1), add (L2), add~ (L2), modB (L3), modB~ (L3), mrow (L3), mrow~ (L3), mac (L4), mac~ (L4), step (L4), reduce (L5), reduce~ (L6) | no | 18267.3 |
<!-- /CLIMBS -->

### 2.1 The A/B: a simplified climb

<!-- ABTEXT -->
**Why.** A review of the climb asked for a simpler one, run next to the c5 climbs (left untouched), with three domain
rules from the engineering guidelines: routines before primitives, no re-searching a mirrored twin that cannot pay off,
and a state cap set from a measurement instead of a fixed 3M; and for long runs to be checked early, a task that uses
up its cap twice logged as a stall instead of waiting out the 25,000 s, with every run's limits written in its script.

**The arms** ([`lib/climb_ab.py`](lib/climb_ab.py); runner [`lib/modal_ab.py`](lib/modal_ab.py), which holds every
limit: an H100, the container's 6 h timeout and 64 GB memory, the climb's 5 h, one search's 30 min). The same rows
(4,000, seed 0), answers (400 rows with T = 1), exact cells and settings as c5_exact_h100.

| arm | the climb |
|---|---|
| **A** | libcore's climb as stage 3 inlines it (c5_exact_h100; c5_learned_s0_h100 with cells learned first): every task and twin searched at every level over the merges, the routines and the primitives (3M stored states), then the primitives alone (16M, once per task) |
| **B** | A plus four rules. (1) *Routines first*: a task is first searched over the solved routines and the 32 primitives, with loops whose bodies (up to 3 items) are made of those; only then over the merges, then the primitives alone. (2) *Twins*: a twin (X and Y swapped) whose mirror is solved and whose own search used up its cap is not searched again; twins of unsolved concepts stay, since BPE needs two tasks per pair. (3) *The cap* from the measurement below. (4) *Stalls*: a task whose searches used up their caps twice is logged as a stall; from then on only rule 1's search runs for it, and only when new routines appeared since its last one |
| **C** | B plus the two fixes the stalls below point to: every goal state is tested (one spelling per goal state, the acceptance probes built once per search; the accepted goal states then give up to 8 spellings each for BPE), and mac's addend is the running sum (A + x d) instead of N (x d + N) |

**The measurement** ([`lib/profile_search.py`](lib/profile_search.py) on an H100, record
`lib/records/profile_mul_3m.log`): one capped search as A runs it, the multiply's problem with the c5 climbs'
level-4 vocabulary (175,299 item classes):
- enumerating the classes takes 417 s, 412 s of it on the host CPU, with the GPU busy 24.5% of the time;
- the search to cost 4 takes 206 s; to cost 5, 434 s (level 4 stores 2.20M states; level 5 streams 100.9M new states
  in 228 s); the GPU is busy 22-26% throughout and the host thread all the time (CPU seconds = wall seconds);
- where a level's time goes (torch.profiler on 2M-row chunks): one instruction applied to 2M rows keeps the GPU 99% busy
  (51M rows/s, 129 kernel launches per item); 256 loop items over 8,192 rows each keep it 40% busy (2M rows/s, 64,458
  kernel launches), because a loop's body runs element by element, grouped per position.

So the search is bound by Python and kernel launches, not by the GPU. Memory: 655 bytes per stored state (working
buffers included), so 78M states fit in 60% of the H100. Time: the next level costs 1.03e-4 s per stored state (228 s
for 2.2M), so a 10-minute search can store about 5.8M. **The cap is time-bound: 5.8M** (A's fixed 3M is within a factor
of two; for this vocabulary any cap from 2M to 67M stores the same levels, since level 4 fits and level 5 does not).
The primitives-alone fallback, from the c5 and stage-3 runs' own H100 logs (level 6 streamed from 12.3M stored states
in 150-280 s): 1.2-2.3e-5 s per stored state, so 27M for 10 minutes (A: 16M). The script's own line (7.8M) divided by
level 4's new-state count in the cost-4 run (2.95M, duplicates included) instead of the stored count; B and C use 5.8M.

**A's two stalls, diagnosed** (both also stop the in-run climbs of §3):
1. *The reduction (the c5 climbs).* The c5 climbs never accepted reduce, at level 3 or 4, though the stage-3 runs found
   it at level 3 (`LDA.N P[BRG.X B[B1.2]]`). [`lib/debug_reduce.py`](lib/debug_reduce.py) rebuilt c5's level-4
   vocabulary and reran the search on c5's rows (record `lib/records/profile_reduce_c5rows.log`): at cost 4 there are
   58 goal states, and libcore tests the first 200 spellings, which reach only 31 of them (a few goal states' alias
   spellings come first). The reduction, `CLK CLA P[BRG.X B3.1]`, is in one of the 27 never tested. Listing one
   spelling per goal state tests all 58 and accepts it (130 s). Which goal states the 200 cover depends on the probes,
   which is why the in-run rows found it and c5's did not.
2. *The multiply (every climb).* mul = `CLK CLA W[LDH.Y P[MUL.X INC] INC SHL.X]` is a loop over y's digits whose body
   costs 6 primitives: the accumulating row (multiply X by one digit of Y and add into A without clearing A) and two
   moves. Loop bodies cost at most 3, so the row must be one item, a merge or a routine, and it never becomes one:
   - A's mac (x d + N) is found as `T.mrow B2.5` (the cleared row, then add N: cost 2), so no program holds the
     accumulating row and BPE never merges it;
   - T.mrow clears A first (CLA), so a loop over it cannot accumulate;
   - every adding routine adds N (T.add = N + X, T.add~ = N + Y). The multiply's problems leave N dirty, and no
     instruction writes it (there is no STA.N).

   So "mul as a loop over T.mrow plus T.add" cannot be written on this machine: the accumulator would have to be N.
   A, X, Y and S are writable, and every addition in the curriculum adds N. C's mac poses A + x d (the running sum in
   A). Its shortest program is the accumulating row itself, `LDH.Y CLK P[MUL.X INC]`. Once that is the routine
   T.mac, mul = `CLA W[T.mac INC SHL.X]` is a composition of cost 5 with a 3-item body. It passes mul's laws at widths
   1-3; with T.mrow in its place it does not ([`lib/test_climb_ab.py`](lib/test_climb_ab.py)).

**What each arm solved, and when** (the table below; seconds from the climb's start, one H100 each):
- **A** (c5_exact_h100) solves its last level-4 task (step) at 4,277 s and ends level 4 at 6,311 s. It finds reduce
  only at level 5 (8,968 s: `B3.2 P[STA.Y T.step]`, a composition with the step routine) and reduce~ at level 6
  (12,209 s). It never finds mul: it searches mul and mul~ again at every level, 600-800 s each time, and stops at
  15,661 s when level 7 solves nothing. No round.
- **B** reaches the same frontier faster: step at 2,631 s and level 4's end at 4,203 s. Routines first finds modx, add, modB,
  mac and step as compositions in seconds to minutes (`B[T.modx STA.X]` for modB; `T.mrow STA.X T.add` for mac;
  `LDA.Y BRG.X STA.Y T.modB~` for step). At level 5 it logs mul, mul~, reduce and reduce~ as stalls (two used-up
  searches each, levels 3 and 4), skips step~ as a twin, retries the stalls by composition once with the new
  routines, and ends at 6,098 s with the stalls named, instead of running to its limit. It never finds reduce, even
  with T.step: its search still tests only the first 200 spellings.
- **C** finds the round. It climbs like B through level 3 (slower per search: it tests every goal state, one by one).
  At level 4 its mac search returns exactly the accumulating row, `LDH.Y CLK P[MUL.X INC]` (3,548 s), and step comes
  as `LDA.Y BRG.X STA.Y T.modB~` (4,979 s). reduce is missed at level 4, because its search met 10,893 goal states at
  cost 5 and C tests the first 4,000 in the order they were generated, and two-item compositions like
  `CLA P[BRG.X STA.Y T.modB~]` are generated last (libcore now tests every goal state, fewest pieces first).

  At level 5 C logs mul, mul~, reduce and reduce~ as stalls and skips step~. Its composition retries then find:
  - mul at 8,756 s: `T.mrow~ P[INC SHL.Y T.mac~]`, a row then a loop of accumulating rows, school multiplication;
  - reduce at 10,095 s: `P[T.mac~ SHL.Y] P[T.step]`, clear the remainder by shifting it out, then repeat the
    long-division step.

  The round search at level 6 finds x^2 mod N at 11,079 s: `LDA.X STA.Y T.mul STA.X T.reduce` and nine equivalent
  spellings, each reproducing all 400 answer rows.

**The A/B's reading.** The review's rules pay as asked. Routines first finds most tasks as compositions in seconds,
twins of solved tasks stop costing a search per level, and stalls end a hopeless climb at 6,098 s (B) instead of
15,661 s (A), naming the stalled tasks. They do not change what the climb can reach: B stops where A stops. Reaching
the round took C's two fixes, which act on the two diagnosed stalls: test every goal state, and pose mac with the
running sum as its addend. Routines first also made a new failure possible, seen only in-run (§3): a cheaper
composition can be an alias that its own practice problem accepts. Occam's step tie-break is the fix that let the
in-run climb through.
<!-- /ABTEXT -->

<!-- AB -->
| arm | record | cap (stored states) | solved: task (level, by, seconds) | stalled / twins skipped | the round | stopped | seconds |
|---|---|---|---|---|---|---|---|
| A | c5_exact_h100 | 3,000,000 | lt (L0, library, 51 s); lt~ (L0, library, 70 s); modx (L1, library, 193 s); modx~ (L1, library, 220 s); add (L2, library, 406 s); add~ (L2, library, 501 s); modB (L3, library, 628 s); modB~ (L3, library, 630 s); mrow (L3, primitives, 850 s); mrow~ (L3, primitives, 1,194 s); mac (L4, library, 3,040 s); mac~ (L4, library, 3,040 s); step (L4, library, 4,277 s); reduce (L5, library, 8,968 s); reduce~ (L6, library, 12,209 s) | - / - | no | L7 sleep: solved [];  | 15,661 |
| A | c5_learned_s0_h100 | 3,000,000 | lt (L0, library, 87 s); lt~ (L0, library, 109 s); modx (L1, library, 246 s); modx~ (L1, library, 276 s); add (L2, library, 476 s); add~ (L2, library, 580 s); modB (L3, library, 719 s); modB~ (L3, library, 721 s); mrow (L3, primitives, 968 s); mrow~ (L3, primitives, 1,350 s); mac (L4, library, 3,436 s); mac~ (L4, library, 3,436 s); step (L4, library, 4,882 s); reduce (L5, library, 10,325 s); reduce~ (L6, library, 14,180 s) | - / - | no | L7 sleep: solved [];  | 18,267 |
| B | ab_B_exact_h100 | 5,800,000 | lt (L0, library, 42 s); lt~ (L0, library, 60 s); modx (L1, routines, 133 s); modx~ (L1, routines, 142 s); add (L2, routines, 195 s); add~ (L2, routines, 215 s); modB (L3, routines, 254 s); modB~ (L3, routines, 256 s); mrow (L3, primitives, 502 s); mrow~ (L3, primitives, 749 s); mac (L4, routines, 1,938 s); mac~ (L4, routines, 1,941 s); step (L4, routines, 2,631 s) | mul (L5), mul~ (L5), reduce (L5), reduce~ (L5) / step~ (L5) | no | stalled: ['mul', 'mul~', 'reduce', 'reduce~'] | 6,098 |
| C | ab_C_exact_h100 | 5,800,000 | lt (L0, library, 74 s); lt~ (L0, library, 103 s); modx (L1, routines, 226 s); modx~ (L1, routines, 242 s); add (L2, routines, 335 s); add~ (L2, routines, 370 s); modB (L3, routines, 436 s); modB~ (L3, routines, 438 s); mrow (L3, primitives, 892 s); mrow~ (L3, primitives, 1,346 s); mac (L4, routines, 3,550 s); mac~ (L4, routines, 3,696 s); step (L4, routines, 4,981 s); mul (L5, routines, 8,763 s); mul~ (L5, routines, 9,159 s); reduce (L5, routines, 10,097 s) | mul (L5), mul~ (L5), reduce (L5), reduce~ (L5) / step~ (L5) | T.reduce STA.X T.mul STA.X T.reduce; T.modx STA.Y T.mul STA.X T.reduce | round found | 11,080 |
<!-- /AB -->

## 3. Stage 3: no skeleton, no menus

<!-- STAGE3 -->
**The file.** [`submission_stage3.py`](submission_stage3.py) is built by [`make_stage3.py`](make_stage3.py) from
[`stage3_template.py`](stage3_template.py) (the new model), [`lib/libcore.py`](lib/libcore.py) (pasted whole) and stage
1's reader and writer, extracted by name, line for line. It holds no division or multiplication datapath and no menu.

**How one training run goes** (one evaluator step: one forward, one loss, one backward; the climb is one unit of work
per step, inside the forward pass, like the tournament's scoring):
- **A.** The reader learns to read (stage 1's I/O lessons). The subtraction table learns its borrows from the mined
  order facts and its digits from the laws (stage 1's table lessons, on registers made of the rows' x and other rows'
  x, since there is no square yet).
- **B.** Once the subtraction passes its checks and is a bijection for 20 steps: the addition table by sub.inverse, the
  multiply-accumulate table by mul.zero and mul.ladder (both on every entry), the countdown's gate by the loop law.
- **C.** Once every cell passes its own checks for 20 steps, the climb (§2) starts on the read rows (the last 4,096 rows
  the reader read, as digits). After every level that made new routines, the round is searched at full size (16
  places): the routines and the straight-line primitives, cost <= 5, the goal the answers of 8 rows with T = 1 (the
  smallest T; the last 512 such rows kept), accepted only if the program reproduces every kept answer row.
- **D.** The round programs that pass (the 4 simplest by expansion into primitives) are the contestants of a tournament
  the answers keep scoring each step; T's direction by speculation; the freeze after 200 exact steps.

**The prediction (rule 8).** The reader, then the countdown (the subtraction table and the gate, stage 1's), each round
the selected program run by `SteMachine` on one-hot digits: every cell lookup is the joint one-hot of its inputs times a
straight-through table (the value: the snapped table's row exactly; the gradient: the table's softmax), every cell's
output damped (x0.1) as the certified files damp every link, the pointer a Python int (control never depends on data).
With the reader and every cell at the truth and the library study's routines as the round, the path is exact on
depth_t_1, depth_t_8, depth_ood_n_t_2 and depth_ood_n_t_64 (`checks/stage3_designed.py`), and a gradient from the
output reaches the subtraction and multiply tables (the multiply's, through the 2,560 lookups of the reduction, is
about 1e-12: deep and damped, the certified files' known edge).

**Results.** Stage 3 as built (variant A's climb) does not certify. On the untouched official manifest (s3_official,
H100) the reader and the three cells are done within about 80 s (phases A and B, 640 steps), and the climb then takes
the rest of the budget: lt (level 0), modx (1), add (2), modB, mrow and reduce (3), mac and step (4), and the training
ends inside level 4 with no multiply and so no round: 0.1% exact. The last climb unit ran 81 s past the 3,600 s (the
search checks its deadline between groups of item classes, which can take that long). The relaxed runs (18,000 s,
H100: s3b_relaxed_h100 and s3_final_h100) show the multiply is not a matter of time: their climbs end at level 5, a
level that solves nothing and merges nothing, with mul and mul~ still unsolved (§2.1 says why), so the round is never
searched with a multiply and the rest of the budget would only idle. I stopped them at steps 8,550 and 3,500 and the
seven stage-3 acceptance runs (CPU, at level 3 of the same climb) with them; their logs are in `records/partial/`.

**Stage 3 with the A/B's variant C** (§2.1; the switches in `libcore.Climb`, turned on in the template):
- *s3c* (C's rules as in the A/B): stopped by me at level 3. In-run, an Occam tie at level 0 took lt's slower spelling
  (`CLC P[LDA.X SBB.N INC]`, which reloads X inside the loop), so routines first made modx
  `T.lt LDA.X STA.S CLA SEL.S` = "x if x < N else 0", which passes modx's laws (LIBRARY.md's r10 alias). Every add then
  failed its inverse check through that helper, and the twin rule, written symmetric, skipped add because add~ was
  solved, which killed the whole original chain. Two fixes: Occam's ties go to the program that executes fewer steps
  (`occam_steps`; it picks the full subtraction and csub), and only twins (task~) are skipped.
- *s3d* (those fixes): **the climb discovers the whole round inside the training run** (s3d_relaxed_h100, 18,000 s):
  - level 4, step 662: mac = the accumulating row `LDH.Y CLK P[MUL.X INC]` (T.mac);
  - level 5: mul = `CLA W[T.mac~ SHL.Y SHR.X]` and `T.mrow P[INC SHL.X T.mac]`, then reduce = `P[MUL.N SHL.Y] P[T.step]`
    (clear the remainder, then repeat the long-division step `T.step = LDA.Y BRG.X STA.Y T.modB~`);
  - level 7, step 685: the round, `LDA.X STA.Y T.mul~ STA.X T.reduce` (x copied, multiplied, reduced) and three
    equivalent spellings, the tournament's contestants. Phase D followed: the answers check 1.0, the exact streak 165
    at step 850, the freeze at step 884 after 200 exact steps.

  **It certifies** (relaxed budget, 18,000 s): T = 64 on both ladders, every rung 768/768, ood_n_t 10,002/10,002,
  ood_t 10,002/10,002, test 9,999/9,999; 9,142 s of training (the round at step 685 and the freeze at 884; the
  steps after the freeze, at learning rate 0, run to the file's 20,000-step limit), 180 s of evaluation. The file
  is [`submission_stage3_s3d.py`](submission_stage3_s3d.py) (md5 373ceb4e, saved from the record).

  **The budget gap.** The phases before the climb take about 80 s. The climb's units take 8,552 s until the round:
  level 0 84 s, level 1 43 s, level 2 67 s, level 3 1,622 s, level 4 3,433 s, level 5 2,894 s, level 6 370 s, level
  7 39 s. Most of levels 4 and 5 is searches that cannot succeed yet, each running to its cap or to the 900 s limit:
  - mul before T.mac exists;
  - reduce and reduce~ at level 4, testing 4,000 goal states one by one;
  - reduce~ at level 5.

  So the official 3,600 s is missed by about 2.4x. s3e and s3f below close the gap: most of it was searches that
  could not succeed yet, and the rest was overhead (§2.1's profile: the GPU is busy a quarter of the time).

  The round search at level 6 (the same library, without T.reduce~) found nothing. Rerun offline with that library,
  on the data's T = 1 answers ([`lib/debug_round.py`](lib/debug_round.py)) and on the model's own learned answer pool
  ([`checks/stage3_pool_check.py`](checks/stage3_pool_check.py), seed 74, 510 rows, all consistent), it finds the
  round at cost 5 in 22 s; the in-run miss at level 6 is not explained.
  s3d_official (the untouched 3,600 s): not certified, the climb inside level 4 at the deadline, and its last unit ran
  346 s past it. The search now checks the deadline inside each item batch, each acceptance batch and the vocabulary's
  construction.
- *s3e* (+ goal states tested fewest pieces first, no cap, the acceptance batched, the deadline checked inside the
  search): reduce comes at level 4, one level earlier than s3d, as `B3.1 P[BRG.X STA.Y T.modB~]` (clear, then bring a
  digit in and reduce, digit by digit; 591 s). s3e_official (the untouched 3,600 s): 3,600 s of training exactly, the
  deadline now holding inside the search. The climb gets through level 4 and into level 5 (level 3 1,753 s, level 4
  1,499 s), so not certified. About 1,880 s of it are mul's and reduce's searches at levels 3 and 4, before the routines
  they are made of exist.
- *s3f* (+ the pieces practised first: mul after mac, reduce after modB; and an exact cache of the item signatures
  across levels, since the test machines and an item's meaning never change): **certified on the untouched official
  manifest** (s3f_official, H100): T = 64 on both ladders, every rung 768/768, ood_n_t 10,002/10,002, ood_t
  10,002/10,002, test 9,999/9,999; 3,134 s of training (the 20,000-step limit, reached after the freeze), 190 s of
  evaluation. The file is kept as [`submission_stage3_s3f.py`](submission_stage3_s3f.py) (md5 31c9cab2). Its climb:
  - levels 0-2 (lt, modx, add and twins), about 160 s;
  - level 3 (modB, mrow), 375 s, against s3e's 1,753 s, since mul and reduce now wait for their pieces;
  - level 4 (mac = the accumulating row, step, reduce = `CLK CLA P[BRG.X STA.Y T.modB~]`), about 1,330 s;
  - level 5 (mul = `CLA W[T.mac SHL.X SHR.Y]`, a loop of its own accumulating rows), about 640 s;
  - level 6, step 671, about 2,620 s into training: the round, `LDA.X STA.Y T.mul STA.X T.reduce`, with
    `LDA.X STA.Y T.mul~ STA.X T.reduce` and two `T.add ...` variants as the tournament's other contestants.

  T's direction resolved at step 691, and the freeze came at step 870, about 2,950 s in. The time that closed the gap
  came first from practising the pieces in order (about 1,900 s of searches that could not succeed yet), then from the
  signature cache and the batched acceptance.

  **Its seeds** (the same file, the official manifest, seeds 1-7 on H100s): seeds 1, 4 and 7 certify the same way
  (T = 64 on both ladders, every rung 768/768; 3,378, 3,085 and 2,973 s of training), so 4 of 8 with seed 74. Seeds 3,
  5 and 6 find the multiply but no round, for one reason: at level 5's sleep BPE merged mul's whole program into a
  single item (`B5.4 = CLA W[T.mac~ SHL.Y SHR.X]`, likewise mul~'s). A solved task became a routine only if its program
  had at least two items, so neither got one, and the round's search composes routines, so it never saw the multiply.
  The climb then ended at level 6 with nothing left to do. (s3d hit the same thing for mul and was saved by mul~.)
  Seed 2 is slower and is still in level 5 at the deadline; its last climb unit ran 36 s past the 3,600 s.
- *s3g* (+ a solved task always becomes a routine, by its program's expanded size, `merged_routines`; and the climb
  stops 120 s before the budget instead of 30 s). Seeds 1-7 on the official manifest (H100):
  - **seeds 1, 2, 3 and 4 certify** (3,192, 3,554, 3,106 and 2,950 s of training). Seed 3 is one that the merged
    multiply stopped under s3f, so the fix holds;
  - seeds 5 and 7 run out of time at level 5. Every level of their climbs takes about 1.5 times as long (level 4: 2,303
    and 2,167 s against 1,221-1,373 s), which points to slower hosts: the search spends three quarters of its time on
    the host CPU (§2.1), so the host's speed is the climb's speed;
  - seed 6 finds the round, then stops at step 635 on a non-finite training loss (the evaluator's check). In phase D
    its T reading was confidently wrong (share 1.000, right 0.000) and the loss was 24.4 at step 600. T's direction
    lesson takes the log of a weight clamped at 1e-30, whose gradient (up to 1e30) can overflow float32. The lesson
    should work in log space.

  s3g_official (the untouched manifest, seed 74): **certified**, T = 64 on both ladders, every rung 768/768, every
  split exact, 3,494 s of training, 220 s of evaluation. This was the final file before s3h:
  [`submission_stage3.py`](submission_stage3.py) (md5 70bac83c); s3f's is kept as `submission_stage3_s3f.py` (md5
  31c9cab2). Its seed-74 run follows s3f's step for step: the round at step 671 and the freeze at 870, but its climb
  takes 2,817 s against s3f's 2,543 s, so the freeze
  comes at about 3,300 s.

  **Where stage 3 stands on the official budget.** The certifying runs are s3f_official and s3f's seeds 1, 4 and 7
  (4 of 8), and s3g_official and s3g's seeds 1, 2, 3 and 4 (5 of 8). All of them have T = 64 on both ladders, every
  rung 768/768, every split exact, and 2,950 to 3,554 s of training. The rest fail for three named reasons:
  - the merged multiply (fixed in s3g);
  - a slow host (the climb takes 2,400-2,600 s on a fast one and about 3,450 s on a slow one, against about 3,480 s
    available);
  - one numerical edge in phase D.

  Certifying every seed needs the search to stop being host-bound (twice the speed would leave room on any host) and
  T's direction lesson in log space.

  **s3g's seeds on B200s** (a research check, via the shared runner `tools/modal_runner.py --mode seeds`,
  run `s3g-b200-seeds`, records in `records/b200/`; the official claim stays the H100 runs above):
  - **6 of 8 certify**: seeds 74, 1, 2, 3, 4 and 7, with 1,707-3,286 s of training. Seed 74 trains in 1,707 s against
    3,494 s on the H100, and seed 7, out of time on the H100, certifies. B200 hosts vary too, though: seeds 2, 3, 4
    and 7 took 2,985-3,286 s.
  - **Seed 6** crashes in phase D exactly as on the H100: the same non-finite loss from T's direction lesson, so that
    failure belongs to the seed, not the host.
  - **Seed 5** finds the multiply and its routine (`T.mul = CLA W[T.mac~ INC SHL.Y]`, by the s3g fix), but the round's
    search at level 6 returns nothing. With no new routine after that, the climb ends and the round is never searched
    again. This is the miss s3d had at level 6, where the next level's new routine (T.reduce~) set off a search that
    found the round; offline reproductions of that search find it.

  So a third fix belongs with the two above: while no round is found, search for it again on the growing answer pool
  instead of letting a finished climb end.

  **s3g's seeds on CPUs** (a research check, relaxed to 18,000 s, one CPU core each; plan `s3g_all`):
  - **6 of 8 certify**: seeds 74, 1, 2, 3, 5 and 7, with 4,951-9,579 s of training (1.4-3.0 times the H100's, on the
    seeds that certify on both).
  - **Seed 6** crashes at step 635 with the same non-finite loss as on the H100 and the B200.
  - **Seed 4** runs out of the 18,000 s inside the climb. At level 4 it solves the mirrored reduction, which the H100
    run of the same seed does not, as a loop of loops: `T.reduce~ = SBB.N B[MUL.Y B[T.modB~]]`, where `T.modB~` is
    itself a loop over a routine with a loop. Every later search that runs this routine pays for it. The round's
    search at level 5 takes 858 s (161 s on seed 3, whose library does not have it). The multiply's search, whose
    items put it inside further loops, uses 4,048 s without finishing its first level. That is four and a half times
    its 900 s limit: the limit is checked between batches of work, and the batch under way at 900 s ran for over
    3,100 s more. s3h was not run on CPUs; its lockstep machine (below) targets exactly this cost, the loop bodies.
- *s3h* ([`submission_stage3_s3h.py`](submission_stage3_s3h.py), built by `make_stage3.py` from the template and
  libcore; `make_stage3.py s3g` checks the certified s3g file against its sources, kept in `work/s3g/`). Three causes,
  each found by reproducing it, and their fixes. Each fix is exact (the same search results) or changes only what was
  a defect. The reproductions are replays: a diagnostic copy of the file (`lib/debug_patch.py`) skips the climb and
  sets the recorded run's library and round programs at the recorded step. The climb uses its own random generators
  and changes no parameter, so the training before and after is the recorded run's (seed 5's climb joins at step 588
  and seed 6's loss is 24.448063 at step 600, as recorded), and phase D comes after about 2 minutes instead of 25.
  1. **The round-search miss was an acceptance test, not the search.** A replay of B200 seed 5 dumped the
     round's answer pool at each in-run round search: 159 rows at level 6, every one consistent with x^2 mod N. Rerun
     offline with that pool and that library, the search reaches `LDA.X STA.Y T.mul STA.X T.reduce` at cost 5 and
     then rejects it: acceptance was `round_fit(...) == 1.0`, and round_fit is a float32 mean, which a GPU computes as
     the sum times 1/n. For 137 of the first 1,024 pool sizes (41, 47, 55, 61, 82, ..., 121, 159, ...) that is
     0.99999994 when every row is right. The offline reproductions used 512 rows, a power of two, so they never missed;
     s3d's miss at level 6 and find at level 7, with the same first rows as probes (pools of roughly 200 and 215 rows,
     not logged exactly), fit the same cause. On a CPU, torch's mean of an all-ones row is exactly 1.0 at every size up
     to 1,024 (checked), consistent with s3g's relaxed CPU seed 5 certifying. Fix:
     `libcore.round_ok`, an exact test (every row reproduced). Every certifying run found its round at a pool size that
     rounds to exactly 1.0, so none of them changes. Also, as asked: a finished climb with no round searches again
     every 10 steps on the answers kept by then (`ROUND_RETRY`).

     A side finding while ruling out the search: `hash_rows`, the search's 64-bit state key, keeps only 8 bits of each
     word's top byte (the last of 8 probes): 200,000 distinct rows that differ only there get 256 keys. Real round
     searches merge 1-61 distinct states at their last stored level and 40-1,979 among the last level's streamed ones
     through it (`lib/round_hash_diag.py`: the same level-6 searches with a key that mixes each word first, on 12
     random answer sets for each of three recorded libraries). All 72 find the round at cost 5 and list the same first
     programs under both keys, and the in-run miss happens with both keys. It is left as it is: a new key would reorder
     the goal states' listing, so which spellings are accepted could change.
  2. **Phase D's NaN was the gate lesson, not T's direction lesson.** A replay of seed 6 reproduces its training
     exactly (loss 24.448063 at step 600 and a non-finite loss at step 635, as on the H100 and the B200). Every
     gradient stayed finite, and the right T reading's weight was 2e-11 at step 600 and doubling each step. At step 635,
     while T's reading turned (its anchor already on the prompt's end, its stride not yet), T's pointer read one place
     past the prompt: every row's count read 0, and the gate's "a count of at least 1 runs its round" term took the
     mean of an empty selection. Fix: that term only when some count is nonzero (`checks/stage3_nan_check.py`: the
     designed model with T's pointer one past the end gives NaN under s3g and a finite loss under s3h). T's direction
     lesson also works in log space now, as asked (`shape_logprobs`, `shape_logrho`): the same loss and gradients as
     s3g's to within 7e-5 relative (`checks/stage3_tlesson_check.py`: the designed reading, every T logit at 0, and T
     confidently wrong at four margins, where the right reading's weight bottoms out near e^-15.6), and with no clamp,
     so a weight below 1e-30 keeps its gradient instead of losing it to s3g's clamp. The s3h replay of seed 6 passes
     step 635, resolves T at step 652, freezes at 779 and certifies (T = 64 on both ladders, every split exact).
  3. **The climb's speed.** `lib/profile_climb.py` (the climb alone with the template's settings, every unit's time
     split) found loop items in `apply_items` taking 95-98% of every unit, on the H100 and the B200 alike: a loop's
     body ran element by element, one gather, run and scatter per distinct element. s3h replaces that with a lockstep
     machine:
     - each item is traced into the primitive instructions `run_items` would execute (memoized; a routine between a
       save and a restore of its pointer), and all rows of a call step together, one instruction each per step, in
       about 40 kernels for every row at once (`_ustep`: register moves as one gather of 8-byte words of the probes'
       digits, the three cells as one table lookup);
     - rows are grouped by their sequence's length (powers of 2), so short sequences do not wait for long ones, and a
       time model picks the old item-by-item path where it is cheaper;
     - a long lockstep (192 steps or more) replays its steps from a captured CUDA graph of 32 steps (the same kernels,
       one launch per graph);
     - a search level's blocks are fused, up to 4 chunks of rows per call (`FUSE`), and the vocabulary's signatures
       likewise.

     The search's results are the same: `lib/test_lockstep.py` compares every state (registers, flags, pointer) with
     the old path bit for bit on real vocabularies at every setting, and bfs's levels, goals, parents and accepted
     programs for fuse 1, 2 and 4 (on a CPU and on an H100, graphs included); `test_libcore`, `test_climb_ab`,
     `checks/stage3_designed.py` and `checks/stage3_train_smoke.py` print what they printed for s3g. And the whole
     climb, run alone on the same rows with s3g's libcore and with s3h's (`lib/profile_climb.py`, records
     `lib/records/s3h/pc_s3g_h100_nosync.log`, `pc_g_f4_h100_nosync.log`), gives the same 26 search records (task,
     level, cost, level sizes, programs, stop) and the same 40-entry library, in 1,325 s to the round against 3,057 s
     on an H100 (lt 49.7 to 4.6 s, mrow 251 to 59 s, mac 208 to 74 s, step~ 375 to 106 s, reduce 399 to 212 s, reduce~
     465 to 259 s); peak GPU memory 29 GiB (fuse 8: 34 GiB, 2% faster). The reductions gain least: about 40% of
     reduce's time is accepting candidates, a few hundred at a time with instruction sequences up to 16,000 long, which
     the lockstep runs top-level item by item (a follow-up, not in the certified file: run each candidate whole, which
     cuts that from 81 to 59 s, `work/s3h_followup_acceptor.diff`; and a time model that charges the item-by-item
     path's per-row cost once per group, which would send single-item calls such as the round's search back to it).

     Tried on the way, each measured once on an H100 (`lib/bench_apply.py`: ten 2M-row calls with 1 to 2,600
     distinct loop items; `lib/profile_task.py`: level 4's mac and step~ from a recorded library; `lib/profile_climb.py`:
     the whole climb; logs in `lib/records/s3h/`, times given new against old):
     - dropped: the first lockstep step, with the three cells as three 64-bit table lookups: slower than the old path
       on large calls (673 against 416 ms for 2,600 items), 79% of its GPU time in those lookups. One 8-bit lookup and
       register moves by 8-byte words made it 1.5-3.4x faster than the old path on nine of the ten calls (on the
       tenth, one item of 4 instructions, 24 against 18 ms; the time model keeps the old path there);
     - kept without a measured gain: a leaner step (about 65 to 40 kernels; the step in the file). Its one benchmark
       run, on another host, was 1.16-1.29x slower than the 65-kernel step on every call; that host's old path was as
       fast on the GPU-bound calls (0.99-1.02x) and slower only on the host-bound ones (1.4-1.5x), so the leaner step
       may be the slower one. Not rerun on one host (a follow-up: the 65-kernel step is in the intermediate build
       8103f399, kept as the `submission` of `records/b200/s3h_b200_seed74.json`);
     - not enough alone: the lockstep without fused blocks and CUDA graphs (the 65-kernel step, whole climb): levels
       0-3 4.2x faster in all (153 against 648 s), level 4 only 1.08x (1,480 against 1,592 s; reduce slower, 430
       against 399 s): its loops over routines run sequences of up to 36,400 instructions, so its steps are
       launch-bound. Fused blocks and graphs brought mac from 177 to 72 s;
     - dropped: fuse 8 instead of 4, 2% faster on the whole climb at 34 instead of 29 GiB of GPU memory.

  **Results** (md5 3b9d9bcf, the final file before s3i; the S3_RUNS table below, `s3h_official` and
  `s3h_h100_seed1`-`7`):
  - **On the official manifest on H100s, every seed certifies** (74 and 1-7): T = 64 on both ladders, every rung
    768/768, ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999. Each run follows its s3g run step for step
    as far as that run went (the same losses, the climb from the same step, the round at the same step: steps
    580-679; the freeze at steps 779-878, the same as s3g's where it froze), about twice as fast: the
    round by 1,058-1,297 s and the freeze by 1,341-2,086 s (the evaluator's elapsed time at the first logged
    step after each), against 2,483-3,111 s and 2,749-3,379 s for s3g's five certifying H100 seeds. Training takes
    1,499-2,229 s (most of what follows the freeze is the learning-rate-0 steps up to the file's 20,000),
    evaluation 156-216 s.
  - The three failures are gone. Seed 5 accepts its round at level 6 (step 619), the search s3g rejected on the B200;
    seed 6 passes step 635 (T resolved at step 652, frozen at 779, by 2,086 s); seed 7, out of time under s3g,
    freezes by 1,396 s.
  - Seed 6 freezes last. While its T reading is still wrong (53 steps), its T lesson runs the countdown from both
    readings, up to 61 rounds of the round program per step, on phase D's hard machine (`run_rounds`, plain
    `run_items`, about 17 s a step on the H100: 900 s from step 600 to T's resolution at 652). That is phase D's cost,
    not the climb's, and the one place left near the 2x margin: a follow-up could run that countdown on all rows at
    once with a fixed shape and replay it from a CUDA graph.
  - On B200s (research, `s3hf_b200_seed*`, the table after S3_RUNS): 8 of 8 certify, the freeze by 996-1,554 s
    (s3g's: 6 of 8, by 1,588-3,037 s). Seeds 6 and 7 were each preempted once and rerun from the start.
- *s3i* ([`submission_stage3_s3i.py`](submission_stage3_s3i.py), md5 0e813144, built by `make_stage3.py`; s3h's
  sources are kept in `work/s3h/`, and `make_stage3.py s3h` checks that they still build the certified s3h file). It
  takes s3h's three follow-ups, each exact (the same states, the same search results, the same training):
  1. **Phase D's hard machine.** `run_rounds`, the countdown on digits behind the tournament's lesson and T's direction
     lesson, ran `run_prog` on the committing rows round by round. That is about 7,100 primitive instructions, each a few
     small kernels: about 0.28 s a round on an H100, however few the rows. The tournament's lesson ran it for all four
     programs on every phase-D step, and T's lesson for up to 61 rounds a step while seed 6's T reading was wrong.
     Now every row runs every round with fixed shapes (a row that does not commit keeps its state), on a round machine
     (`libcore.RoundMachine`):
     - control never depends on data, so the pointer is resolved once, on the host: INC and the routines' saves and
       restores (3,380 of seed 6's 7,098 instructions) cost nothing;
     - every other instruction is one or two fixed slice operations on one state tensor;
     - on a GPU the whole program is one CUDA graph, captured once per program and row count (rows rounded up to 128)
       and replayed; on a CPU it runs eagerly;
     - the machines' dirty starts are `run_prog`'s own: the same draws on the same generator for the committing rows,
       placed on those rows in order.

     It is checked three ways. `lib/test_lockstep.py` compares every committing row's whole state (registers and
     flags) with `run_prog`'s machine after each of five rounds on one generator: every primitive, random 300-instruction
     programs at six settings (bases 2, 3 and 10), and the recorded rounds of s3h's seed 6, s3f, s3d and the library
     study, on a CPU, a B200 and an H100, graph replays included. T's direction lesson gives s3h's loss and gradients
     bit for bit (`checks/stage3_tlesson_check.py`, six cases, up to 61 rounds). And seed 6's replay, with every
     `run_rounds` call also run by s3h's code (`lib/debug_patch.py` RR_CHECK), gives the same state on all 845 calls of
     phase D (4,029 rounds; T's lesson ran 61 rounds on 53 of them), and phase D line for line as s3h's replay: every
     loss, T resolved at step 652, the freeze at step 779.

     Phase D on the replays (B200, `records/s3h_debug/`): steps 600-700 (T's direction lesson on 53 of them) took
     58-66 s, against 1,100 s for s3h's replay the same day (640 s on an earlier host); steps 700-800 took 13-19 s
     against 110 s. On a CPU, a round of seed 6's program on 128 rows takes 9 ms against 46 ms.
  2. **The two search refinements.** `Acceptor.many` runs each candidate program whole, as one item (s3h's follow-up
     diff, `work/s3h_followup_acceptor.diff`). And `apply_items`' time model charges the item-by-item path's per-row
     cost once per group of rows it takes and puts back (an item, or one element of a loop body at one iteration), not
     once per instruction. So calls with a few straight items on many rows, such as the round's search, go back to it.

     The whole climb was run on two H100s the same day (`lib/profile_climb.py`, the same rows, s3h's libcore kept as
     `lib/libcore_s3h.py` against s3i's; logs `lib/records/s3h/pc_s3h_ab_h100_nosync.log`, `pc_s3i_h100_nosync.log`). It
     gives the same 26 search records, library and rounds (also the same as s3h's and s3g's earlier profiles), and
     reaches the round in 964 s against 1,133 s:
     - the round's searches at levels 5 and 6: 17.9 and 25.7 s against 40.2 and 53.9 s (s3g's: 18.1 and 26.3 s);
     - reduce: 136 s against 179 s, and reduce~ 175 s against 214 s; their acceptance takes 58 s against 93 s and
       79 s against 113 s.

     `test_libcore`, `test_climb_ab` and `checks/stage3_designed.py` print what they print with s3h's libcore or
     file. `checks/stage3_train_smoke.py` prints s3h's log, but for one printed loss's last digit (step 31), which also
     flips between hosts for s3h itself.

  **Results** (the final file, md5 0e813144; the S3_RUNS table, `s3i_official` and `s3i_h100_seed1`-`7`):
  - **On the official manifest on H100s, every seed certifies again** (74 and 1-7): T = 64 on both ladders, every rung
    768/768, ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999. Each run follows its s3h run exactly: every
    search of the climb, the round and library, T's resolution, the freeze step, and the loss at every logged step.
  - The round comes by 948-1,099 s and the freeze by 982-1,149 s (the evaluator's elapsed time at the first logged step
    after each), against 1,058-1,297 s and 1,341-2,086 s for s3h. Training takes 1,133-1,350 s (s3h: 1,499-2,229 s),
    evaluation 155-219 s.
  - Seed 6, s3h's last, freezes by 1,149 s against 2,086 s: its steps 600-800 took 95 s against 1,028 s. Every seed
    now freezes within a third of the 3,600 s budget.
  - On B200s (research, `s3i_b200_seed*`, the table after S3_RUNS): 8 of 8 certify, each following its s3h B200 run
    exactly, the freeze by 809-1,076 s (s3h's: 996-1,554 s; seed 6 1,076 s against 1,554 s).

**What the result rests on.** Two later studies of s3i have their own write-ups; §4 marks every hand-made part with
their verdicts, and §6 draws the Bar 2 conclusion.
- [ABLATION_S3.md](ABLATION_S3.md) removes each hand-made part in turn and retrains on the official manifest, one H100
  per run, on seed 74 (the manifest untouched) and seed 1; the unchanged file replays s3i's own runs on both exactly.
  Seven parts are needed on both seeds (the round never comes without them), one on one seed of two (the chain problem
  alone), four only save time, and five are not needed.
- [SELF_PRACTICE.md](SELF_PRACTICE.md) swaps the nine designed practice problems and their order for a law book from
  which the learner poses its own problems, and a general rule that picks what to practise.
  [`self_practice/submission_stage3_sp.py`](self_practice/submission_stage3_sp.py) (md5 f0ef5b06) certifies on the
  official manifest, the round by 746 s, and on 7 of 8 B200 seeds;
  [`self_practice/submission_stage3_sp2.py`](self_practice/submission_stage3_sp2.py) (md5 afc70fe9) on the official
  manifest, the round by 1,029 s, and on all 8.

**A weak check, found by the ablation.** s3i's reduction check tests a program on 60 probes at each of 1, 2 and 3
digits. It can accept `B[P[T.modB~ T.modB]]` (cost 4, met before the real reduction at cost 5), a repeated subtraction
that is right only for small quotients: it passes when none of the 60 three-digit probes has a quotient above 600,
about 3% of runs on H1's rows ([`reduce_check.py`](ablation_s3/reduce_check.py)). Once accepted, the reduction is
never searched again and no round can use it, so such a run cannot certify. None of s3i's eight seeds drew it, on
H100s or B200s; one ablation run did (no_step, seed 74). The conditional subtraction's check is weak too: it never
tests x between N and 2N, and Occam's tie-break is what keeps the search off "x if x < N, else 0". The fix
([ABLATION_S3.md §8](ABLATION_S3.md#8-fixing-the-two-weak-checks)) adds full-size probes to the first check and probes
between N and 2N to the second, in new copies of s3i and sp2. Both copies replay their originals' 17 certified runs
field for field; with the fix, s3i needs neither Occam's tie-break nor the step problem.
<!-- /STAGE3 -->

<!-- S3_RUNS -->
| record | task | device | seed | max T, seen / unseen moduli | rungs | splits | training s (frozen at step) | evaluation s | what it chose | md5 |
|---|---|---|---|---|---|---|---|---|---|---|
| s3_official | h1 | H100 | 74 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3681 (not frozen) | 5 | round programs (tournament weights): none | b0ef5238 |
| s3_relaxed_h100 | h1 | H100 | 74 | crashed (AttributeError: 'NoneType' object has no attribute '__dict__'. Did you mean: '__dir__'?) | | | | | | 1fc2cf7c |
| s3_smoke_cpu | h1 | cpu | 74 | crashed (AttributeError: 'NoneType' object has no attribute '__dict__'. Did you mean: '__dir__'?) | | | | | | 1fc2cf7c |
| s3b_smoke_cpu | h1 | cpu | 74 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3001 (not frozen) | 8 | round programs (tournament weights): none | 3bb65b6c |
| s3d_official | h1 | H100 | 74 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3946 (not frozen) | 4 | round programs (tournament weights): none | 373ceb4e |
| s3d_relaxed_h100 | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 9142 (884) | 180 | round programs (tournament weights): LDA.X STA.Y T.mul~ STA.X T.reduce (0.173); LDA.X STA.Y T.mul~ STA.Y T.reduce~ (0.380); T.add STA.Y T.mul~ STA.X T.reduce (0.195); T.add STA.Y T.mul~ STA.Y T.reduce~ (0.252) | 373ceb4e |
| s3e_official | h1 | H100 | 74 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 5 | round programs (tournament weights): none | b7435fa8 |
| s3f_official | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3134 (870) | 190 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 31c9cab2 |
| s3f_seed1_h100 | h1 | H100 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3378 (812) | 201 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 31c9cab2 |
| s3f_seed2_h100 | h1 | H100 | 2 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3636 (not frozen) | 5 | round programs (tournament weights): none | 31c9cab2 |
| s3f_seed3_h100 | h1 | H100 | 3 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 5 | round programs (tournament weights): none | 31c9cab2 |
| s3f_seed4_h100 | h1 | H100 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3085 (878) | 184 | round programs (tournament weights): LDA.X STA.Y T.mul~ STA.X T.reduce (0.383); T.add STA.Y T.mul~ STA.X T.reduce (0.227); T.modx STA.Y T.mul~ STA.X T.reduce (0.207); T.modB STA.Y T.mul~ STA.X T.reduce (0.184) | 31c9cab2 |
| s3f_seed5_h100 | h1 | H100 | 5 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 4 | round programs (tournament weights): none | 31c9cab2 |
| s3f_seed6_h100 | h1 | H100 | 6 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 5 | round programs (tournament weights): none | 31c9cab2 |
| s3f_seed7_h100 | h1 | H100 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 2973 (864) | 189 | round programs (tournament weights): LDA.X STA.Y T.mul~ STA.X T.reduce (0.231); T.add STA.Y T.mul~ STA.X T.reduce (0.248); T.modx STA.Y T.mul~ STA.X T.reduce (0.239); T.modB STA.Y T.mul~ STA.X T.reduce (0.282) | 31c9cab2 |
| s3g_cpu_seed1 | h1 | cpu | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 9579 (814) | 6213 | round programs (tournament weights): LDA.X STA.Y T.mrow STA.X T.reduce (0.158); LDA.X STA.Y T.mul~ STA.X T.reduce (0.294); T.add STA.Y T.mrow STA.X T.reduce (0.294); T.add STA.Y T.mul~ STA.X T.reduce (0.254) | 70bac83c |
| s3g_cpu_seed2 | h1 | cpu | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 7656 (862) | 3394 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); LDA.X STA.Y T.mul~ STA.X T.reduce (0.245); T.add STA.Y T.mul STA.X T.reduce (0.234); T.add STA.Y T.mul~ STA.X T.reduce (0.280) | 70bac83c |
| s3g_cpu_seed3 | h1 | cpu | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 5822 (783) | 998 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); LDA.X STA.Y T.mul~ STA.X T.reduce (0.220); T.add STA.Y T.mul STA.X T.reduce (0.263); T.add STA.Y T.mul~ STA.X T.reduce (0.266) | 70bac83c |
| s3g_cpu_seed4 | h1 | cpu | 4 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 18000 (not frozen) | 9 | round programs (tournament weights): none | 70bac83c |
| s3g_cpu_seed5 | h1 | cpu | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 5399 (818) | 2557 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.273); LDA.X STA.Y T.mul~ STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.242); T.add STA.Y T.mul~ STA.X T.reduce (0.235) | 70bac83c |
| s3g_cpu_seed6 | h1 | cpu | 6 | crashed (FloatingPointError: non-finite training loss at step 635, pass 1) | | | | | | 70bac83c |
| s3g_cpu_seed7 | h1 | cpu | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 5360 (863) | 2835 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.240); LDA.X STA.Y T.mul~ STA.X T.reduce (0.274); T.add STA.Y T.mul STA.X T.reduce (0.055); T.add STA.Y T.mul~ STA.X T.reduce (0.431) | 70bac83c |
| s3g_cpu_seed74 | h1 | cpu | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 4951 (870) | 2678 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.430); LDA.X STA.Y T.mul~ STA.X T.reduce (0.197); T.add STA.Y T.mul STA.X T.reduce (0.257); T.add STA.Y T.mul~ STA.X T.reduce (0.116) | 70bac83c |
| s3g_official | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3494 (870) | 220 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 70bac83c |
| s3g_seed1_h100 | h1 | H100 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3192 (812) | 179 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 70bac83c |
| s3g_seed2_h100 | h1 | H100 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3554 (862) | 199 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 70bac83c |
| s3g_seed3_h100 | h1 | H100 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3106 (783) | 203 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 70bac83c |
| s3g_seed4_h100 | h1 | H100 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 2950 (878) | 184 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 70bac83c |
| s3g_seed5_h100 | h1 | H100 | 5 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 5 | round programs (tournament weights): none | 70bac83c |
| s3g_seed6_h100 | h1 | H100 | 6 | crashed (FloatingPointError: non-finite training loss at step 635, pass 1) | | | | | | 70bac83c |
| s3g_seed7_h100 | h1 | H100 | 7 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 5 | round programs (tournament weights): none | 70bac83c |
| s3h_h100_seed1 | h1 | H100 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1499 (812) | 166 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 3b9d9bcf |
| s3h_h100_seed2 | h1 | H100 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1578 (862) | 176 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 3b9d9bcf |
| s3h_h100_seed3 | h1 | H100 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1646 (783) | 192 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 3b9d9bcf |
| s3h_h100_seed4 | h1 | H100 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1516 (878) | 164 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 3b9d9bcf |
| s3h_h100_seed5 | h1 | H100 | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1640 (818) | 182 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.280); T.add STA.Y T.mul STA.X T.reduce (0.223); T.modx STA.Y T.mul STA.X T.reduce (0.256); T.modB STA.Y T.mul STA.X T.reduce (0.241) | 3b9d9bcf |
| s3h_h100_seed6 | h1 | H100 | 6 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 2229 (779) | 156 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.267); T.add STA.Y T.mul STA.X T.reduce (0.264); T.modx STA.Y T.mul STA.X T.reduce (0.229); T.modB STA.Y T.mul STA.X T.reduce (0.240) | 3b9d9bcf |
| s3h_h100_seed7 | h1 | H100 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1592 (864) | 166 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.231); LDA.X STA.Y T.mul~ STA.X T.reduce (0.248); T.add STA.Y T.mul STA.X T.reduce (0.239); T.add STA.Y T.mul~ STA.X T.reduce (0.282) | 3b9d9bcf |
| s3h_official | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1691 (870) | 216 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 3b9d9bcf |
| s3i_h100_seed1 | h1 | H100 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1250 (812) | 190 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 0e813144 |
| s3i_h100_seed2 | h1 | H100 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1133 (862) | 155 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 0e813144 |
| s3i_h100_seed3 | h1 | H100 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1170 (783) | 176 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 0e813144 |
| s3i_h100_seed4 | h1 | H100 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1180 (878) | 196 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 0e813144 |
| s3i_h100_seed5 | h1 | H100 | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1277 (818) | 184 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.280); T.add STA.Y T.mul STA.X T.reduce (0.223); T.modx STA.Y T.mul STA.X T.reduce (0.256); T.modB STA.Y T.mul STA.X T.reduce (0.241) | 0e813144 |
| s3i_h100_seed6 | h1 | H100 | 6 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1350 (779) | 206 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.267); T.add STA.Y T.mul STA.X T.reduce (0.264); T.modx STA.Y T.mul STA.X T.reduce (0.229); T.modB STA.Y T.mul STA.X T.reduce (0.240) | 0e813144 |
| s3i_h100_seed7 | h1 | H100 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1347 (864) | 219 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.231); LDA.X STA.Y T.mul~ STA.X T.reduce (0.248); T.add STA.Y T.mul STA.X T.reduce (0.239); T.add STA.Y T.mul~ STA.X T.reduce (0.282) | 0e813144 |
| s3i_official | h1 | H100 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1340 (870) | 210 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 0e813144 |
<!-- /S3_RUNS -->

The research seeds on B200s (`records/b200/`; the official claims are the H100 runs above): s3g's
(`s3g_b200_b200_seed*`), an intermediate s3h build's (`s3h_b200_seed*`, md5 8103f399: the three fixes with the first
lockstep machine, stopped after two seeds to free GPUs for the H100 runs), s3h's certified file's (`s3hf_b200_seed*`) and
s3i's (`s3i_b200_seed*`).

<!-- S3_B200 -->
| record | task | device | seed | max T, seen / unseen moduli | rungs | splits | training s (frozen at step) | evaluation s | what it chose | md5 |
|---|---|---|---|---|---|---|---|---|---|---|
| b200/s3g_b200_b200_seed1 | h1 | B200 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1748 (812) | 98 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 70bac83c |
| b200/s3g_b200_b200_seed2 | h1 | B200 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 2985 (862) | 166 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 70bac83c |
| b200/s3g_b200_b200_seed3 | h1 | B200 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3286 (783) | 190 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 70bac83c |
| b200/s3g_b200_b200_seed4 | h1 | B200 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3181 (878) | 171 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 70bac83c |
| b200/s3g_b200_b200_seed5 | h1 | B200 | 5 | None / None | not all (worst rung 0/768) | ood_n_t 5/10,002, ood_t 20/10,002, test 7/9,999 | 3600 (not frozen) | 4 | round programs (tournament weights): none | 70bac83c |
| b200/s3g_b200_b200_seed6 | h1 | B200 | 6 | crashed (FloatingPointError: non-finite training loss at step 635, pass 1) | | | | | | 70bac83c |
| b200/s3g_b200_b200_seed7 | h1 | B200 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 3100 (863) | 171 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.231); LDA.X STA.Y T.mul~ STA.X T.reduce (0.248); T.add STA.Y T.mul STA.X T.reduce (0.239); T.add STA.Y T.mul~ STA.X T.reduce (0.282) | 70bac83c |
| b200/s3g_b200_b200_seed74 | h1 | B200 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1707 (870) | 102 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 70bac83c |
| b200/s3h_b200_seed4 | h1 | B200 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1695 (878) | 97 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 8103f399 |
| b200/s3h_b200_seed74 | h1 | B200 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1665 (870) | 99 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 8103f399 |
| b200/s3hf_b200_seed1 | h1 | B200 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1143 (812) | 93 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 3b9d9bcf |
| b200/s3hf_b200_seed2 | h1 | B200 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1149 (862) | 105 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 3b9d9bcf |
| b200/s3hf_b200_seed3 | h1 | B200 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1655 (783) | 178 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 3b9d9bcf |
| b200/s3hf_b200_seed4 | h1 | B200 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1482 (878) | 153 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 3b9d9bcf |
| b200/s3hf_b200_seed5 | h1 | B200 | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1494 (818) | 154 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.280); T.add STA.Y T.mul STA.X T.reduce (0.223); T.modx STA.Y T.mul STA.X T.reduce (0.256); T.modB STA.Y T.mul STA.X T.reduce (0.241) | 3b9d9bcf |
| b200/s3hf_b200_seed6 | h1 | B200 | 6 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1683 (779) | 98 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.267); T.add STA.Y T.mul STA.X T.reduce (0.264); T.modx STA.Y T.mul STA.X T.reduce (0.229); T.modB STA.Y T.mul STA.X T.reduce (0.240) | 3b9d9bcf |
| b200/s3hf_b200_seed7 | h1 | B200 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1108 (863) | 92 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.231); LDA.X STA.Y T.mul~ STA.X T.reduce (0.248); T.add STA.Y T.mul STA.X T.reduce (0.239); T.add STA.Y T.mul~ STA.X T.reduce (0.282) | 3b9d9bcf |
| b200/s3hf_b200_seed74 | h1 | B200 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1506 (870) | 163 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 3b9d9bcf |
| b200/s3i_b200_seed1 | h1 | B200 | 1 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 956 (812) | 96 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.210); LDA.X STA.Y T.mul~ STA.X T.reduce (0.254); T.add STA.Y T.mul STA.X T.reduce (0.313); T.add STA.Y T.mul~ STA.X T.reduce (0.223) | 0e813144 |
| b200/s3i_b200_seed2 | h1 | B200 | 2 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 940 (862) | 99 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.241); T.add STA.Y T.mul STA.X T.reduce (0.245); T.modx STA.Y T.mul STA.X T.reduce (0.234); T.modB STA.Y T.mul STA.X T.reduce (0.280) | 0e813144 |
| b200/s3i_b200_seed3 | h1 | B200 | 3 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1271 (783) | 214 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.251); T.add STA.Y T.mul STA.X T.reduce (0.220); T.modx STA.Y T.mul STA.X T.reduce (0.263); T.modB STA.Y T.mul STA.X T.reduce (0.266) | 0e813144 |
| b200/s3i_b200_seed4 | h1 | B200 | 4 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 940 (878) | 96 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.383); LDA.X STA.Y T.mul~ STA.X T.reduce (0.227); T.add STA.Y T.mul STA.X T.reduce (0.207); T.add STA.Y T.mul~ STA.X T.reduce (0.184) | 0e813144 |
| b200/s3i_b200_seed5 | h1 | B200 | 5 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1233 (818) | 167 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.280); T.add STA.Y T.mul STA.X T.reduce (0.223); T.modx STA.Y T.mul STA.X T.reduce (0.256); T.modB STA.Y T.mul STA.X T.reduce (0.241) | 0e813144 |
| b200/s3i_b200_seed6 | h1 | B200 | 6 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 1284 (779) | 183 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.267); T.add STA.Y T.mul STA.X T.reduce (0.264); T.modx STA.Y T.mul STA.X T.reduce (0.229); T.modB STA.Y T.mul STA.X T.reduce (0.240) | 0e813144 |
| b200/s3i_b200_seed7 | h1 | B200 | 7 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 938 (863) | 100 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.231); LDA.X STA.Y T.mul~ STA.X T.reduce (0.248); T.add STA.Y T.mul STA.X T.reduce (0.239); T.add STA.Y T.mul~ STA.X T.reduce (0.282) | 0e813144 |
| b200/s3i_b200_seed74 | h1 | B200 | 74 | 64 / 64 | 768/768 every rung | ood_n_t 10,002/10,002, ood_t 10,002/10,002, test 9,999/9,999 | 939 (870) | 101 | round programs (tournament weights): LDA.X STA.Y T.mul STA.X T.reduce (0.219); LDA.X STA.Y T.mul~ STA.X T.reduce (0.470); T.add STA.Y T.mul STA.X T.reduce (0.163); T.add STA.Y T.mul~ STA.X T.reduce (0.149) | 0e813144 |
<!-- /S3_B200 -->

## 4. What is still hand-made

<!-- HANDMADE -->
Everything below was written by hand; nothing in it was chosen for H1 by looking at H1's answers, but all of it was
written knowing the task family. "Gone" marks what stage 3 no longer has and stage 1 still does.

Two later studies of the final file test the stage-3 column; the two right-hand columns give their verdicts, and the
write-ups have the details:
- [ABLATION_S3.md](ABLATION_S3.md) removes each part in turn and retrains on the official manifest, one H100 per run:
  the main arms on seed 74 (the manifest untouched) and seed 1, five later ones on seed 74 (two of them on seed 1
  too). The unchanged file, run beside the arms, replays s3i's own runs on both seeds exactly. **Needed**: the round is
  never found without it, on either seed. **Needed on one seed of two**: one seed certifies, the other never finds the
  round. **Speed only**: the run still certifies on both seeds, frozen by 1,584 s at the latest. **Not needed**: it
  certifies about as fast. **Not tested**: no arm removed it. Every failure of the seven needed parts used up its state
  cap (every program up to cost 5) and none reached the 900 s search limit, so more training time would not rescue it;
  a wider search (every program up to cost 6) found nothing for six of them (ABLATION_S3.md §7).
- [SELF_PRACTICE.md](SELF_PRACTICE.md) replaces the curriculum with a law book from which the learner poses its own
  practice problems, and one general rule that picks what to practise. Its files certify (§6); its column says which
  rows it replaces.

| what | stage 1 (certified) | stage 3 | stage 3 without it ([ABLATION_S3.md](ABLATION_S3.md)) | self-practice ([SELF_PRACTICE.md](SELF_PRACTICE.md)) |
|---|---|---|---|---|
| **the instruction set** | the H1 datapath: one register of 9 places, 8 steps of 9 links (bring a digit in, run the chain, move), a final correction and a readout; the multiply as one product per digit pair, placed and summed; the countdown | **gone**: a general digit machine instead (library/dsl.py's): 5 registers (A, X, Y, N, S), a shared pointer, a borrow/carry, a multiplier digit and a multiply carry; 32 primitive instructions (subtract-with-borrow, add-with-carry and multiply-accumulate at the pointer against any register; keep-or-discard on the borrow; load, store, clear; move the pointer; shift a register; bring X's top digit into A) and loops that repeat their body P, B or W times; a routine restores the pointer it was called with. The countdown stays (stage 1's) | SEL.S (keep or discard on the borrow) and BRG.X (bring X's top digit in), each taken out of the searches' vocabulary: **needed** (without SEL.S no conditional subtraction, so nothing after it; without BRG.X no step and no reduction). The other instructions, the loops and the countdown: not tested | kept: the same machine |
| **the menus** | the division's menu at free attention's width (operands at offsets -3..3, borrow routes, flags, written values: 7.9e9 chains), the step's and the reading's menus, the multiply's 33M layouts, the round menu (x10, + one digit, x the input) | **gone**: what runs is a program found by search and composition; nothing lists the options | - | - |
| **the alphabet and sizes** | base 10 digits; 8, 8 and 2 digits read and 8 written; P = 9 places | the same reading; the round machine's registers 16 places; the practice problems at widths 1 and 2 | not tested | the practice problems' widths by one rule (the widest whose instance fits the registers); the rest kept |
| **the cells** | a subtraction MLP and a product MLP; a 2 x 2 gate | three tables of logits (subtraction, addition, multiply-accumulate (a, x, h, k)) and the gate: the cells' input shapes are designed | not tested | kept |
| **the reader and writer** | IO.md / WIRING.md stage 3: 24 reading hypotheses, 3 pointer shapes, the run rules | the same, line for line | not tested | kept |
| **the law library** | general_lessons' LAWS (+ red.link, loop.count) | a subset (sub.order, sub.bijective, sub.inverse, sub.identity, sub.zero_minus, add.commute, mul.zero, mul.ladder, mul.distribute, red.identity, red.step, red.invariant, loop.count) | the late table lessons (sub.inverse, sub.identity, add.commute, the mined facts' borrows, the register-width wrap-around fact, sub.zero_minus; they join 100 steps after the digit lessons, `LATE_AFTER`): **not needed** (no_late: the tables pass their checks at the same steps and the climb is identical line for line on both seeds; s3i's ANCHOR_* constants, the wrap fact's included, are never read). red.step: the next row. The rest: not tested | the table lessons kept; the practice problems come from its own law book instead (the curriculum row) |
| **lessons that define a step** | the ladder and the step law of each lesson set (red.step), the link law | the same laws become practice problems (modB, step) instead of checks on a menu; the template's predicates: below only (no Montgomery practice problems) | red.step's two problems, the chain (modB) and the long-division step (step): **needed** (no_redstep: no reduction is found). One at a time: the chain alone is **needed on one seed of two** (no_modB: the step is then solved with the chain spelled out, on X or on Y at the same cost, and the search's listing order breaks the tie; on seed 1 it was Y and the run certified, on seed 74 X, and no reduction was found); the step alone is **not needed** (no_step: s3i's reduction never calls it; seed 1 certified 264 s sooner, and seed 74 failed by the weak reduction check, §3, not for want of the step) | **replaced**: the law book's remainder law on x' + d N' and on 10 x' + d (patterns of the division law's form a + q b) poses the chain and the step among its candidates, unmarked |
| **the curriculum** | - | which laws become practice problems, their helper order (in the final file the pieces first: mul after mac, reduce after modB), the register variants (X <-> Y), mac posed with the running sum as its addend (A + x d: designed, so that the multiply's row is a problem of its own), widths, settings, probe counts (§2); the round's answers are the rows with T = 1 (an H1-shaped constant: M4, whose every T is 8, gives the round's search no answers) | the order (`PREREQ`): **speed only**, +277 s and +321 s to the round (seeds 74 and 1). The X/Y twins: **needed** (no reduction: s3i's runs the chain on Y, and only the chain's twin works on Y). mac with the running sum as its addend: **needed** (posed as x d + N, no multiply). The round's answers from rows with T = 1: **speed only** (the batch's smallest T also works: 82 steps later on seed 74, where one batch had no T = 1 row, and at the same step on seed 1). Which laws: red.step's two are needed together, the chain alone on one seed of two, the step alone not (the row above); the multiply's row problem (mrow): **not needed** (one seed: the multiply comes a level early, the reduction a level late in another form, the round a step early); lt, modx, add, mul and reduce cannot be dropped without redesigning the others' checks. Widths, settings, probe counts: not tested (the reduction's acceptance probes turned out weak: §3) | **replaced** by a law book (11 laws on 8 patterns of the data's N and x, each posed in every binding of its operands to the registers, the X/Y twins among them) and one general choosing rule (`lpca`): a law is posed once the routines its construction needs exist (no order), a + b c is one law in its 24 bindings (nothing marks the accumulator), the answers come from the smallest T. All nine of s3i's problems appear among its candidates, unmarked. Settings and probe counts stay as search sizes. Designed in their place: the law book and the choosing rule |
| **the search and the merges** | the tournament (one logit per whole combination, pass/fail per batch, pruning at 12 nats) | breadth-first search by cost (caps measured on the H100: 5.8M / 27M stored states for a 10-minute search; cost <= 7, loop bodies <= 3); the A/B's rules: compositions of the routines first, then the merges, then the primitives; twins skipped once their original is solved; a stall after two used-up searches (compositions only after that); every goal state tested, fewest pieces first; one search at most 900 s. BPE (two tasks, ties to the largest), Occam's simplest program (ties to the fewest executed steps), routines for solved tasks; the round's search space (the routines and the straight-line primitives, cost <= 5); the round programs' tournament (4 contestants, Occam's order) | compositions first (`COMP_FIRST`): **not needed**. Occam's ties to fewer executed steps (`OCCAM_STEPS`): **needed**, but only because modx's check is weak (without it modx becomes "x if x < N, else 0", s3c's failure, and no adder follows; with the check fixed it certifies on both seeds). Routines: **needed** (no multiply: its loop holds the accumulating row only as one instruction). Twins skipped (`SKIP_TWINS`): **speed only**, +364 s and +416 s. BPE: **speed only**, +282 s and +291 s. The round's cost limit above 5 (`ROUND_COST`): **not needed** (one seed: at 6 nothing changes, since the round's search stops at its state cap after cost 5 either way; at 4 no round is found, since every round program found costs 5). The caps, the stall rule and the 900 s limit: not tested (a wider search, up to cost 6, found nothing for the needed parts) | kept, with its own sizes (probes, deep probes, the round's search by routine calls) |
| **the mined facts** | the miner's grammar (order, equality, range, nonzero, divisibility) | the same | not tested (their late borrow lesson: not needed, the law library row) | kept |
| **the schedule** | phases A-C, the reader's stages, freeze after 200 exact steps | phases A-D (cells, then the climb, then the round), the same freeze | not tested | kept |
| **the answers** | choose the lesson set and the round options; the reading; T's direction | the round's goal and its tournament; the reading; T's direction | the round's goal from rows with T = 1: speed only (the curriculum row); the rest not tested | the round's answers from the smallest T (the curriculum row); the rest kept |<!-- /HANDMADE -->

## 5. Rules 7, 8 and 14

<!-- RULES -->
The rules are [one-layer-deeper/README.md §Rules](../../one-layer-deeper/README.md#rules). "Gray" means a reasonable
reader could object; the point is flagged, not argued away. The other rules stand as for the parents (OVERVIEW.md §6):
one file, self-contained (torch and the benchmark API only; stage 3 also no numpy), Adam, no backward or grad calls,
every parameter from a random start, the evaluator owning the loop.

| rule | stage 1 | stage 3 |
|---|---|---|
| **7. no hard-coded algorithm in the forward pass** | **gray, as its parents**: the H1 datapath (one register, 8 steps of 9 links, bring-in, chain, move; the multiply as placed digit products; the countdown) is written by hand, and the menus that plug into it hold the right wirings (the division's at free attention's width, the multiply's 33M layouts, the reading's, the round's x10 / + one digit / x the input). What is chosen in training is which option, never the datapath | **much less gray**: the forward pass runs a program the training run found (search, merges, routines, composition against the answers) on learned tables; no datapath and no menu is in the file. Still gray: the instruction set leans toward digit arithmetic (cells at a pointer against a register, keep-on-borrow, bring the top digit in, loops by the number's length and the base); the practice problems walk the search up the same pieces the skeleton had (the ripple, the link, the chain, the step, the reduction; the row, the multiply), and two of them are the template's definitions of long division's step (red.step's ladder and step law), as in stage 1; the countdown is stage 1's. A strict reader can say the algorithm moved into the instruction set and the curriculum (LIBRARY.md §6 says the same of its in-run sketch) |
| **8. final logits inside autograd, unbroken gradient path** | yes, with the parents' gray edges. Gradient test (Modal, `checks/gradtest.py`, 16 rows of depth_t_8 and depth_ood_n_t_4, every table and switch at the truth): with the below set selected every parameter tensor gets a gradient except the unselected lowzero set's four blocks; with lowzero selected, also the round's + constant switch (the damped gradient underflows at the first Montgomery step), as LESSONS.md §5 found for the general file. Training returns zero logits: the lessons teach, not the final logits | the prediction path is one-hot digits through straight-through tables (`SteMachine`) and stage 1's countdown and reader; the round program is a straight-through choice over the tournament's contestants. Gradient test on the certified file (`checks/stage3_gradtest.py`: submission_stage3.py with the library and the four contestants its official run discovered, `lib/s3f_lib.json`, reader and cells at the truth, depth_t_1): every one of the 16 parameter tensors is in the graph of the final logits. With the right tokens' logits as the objective, all 16 get a nonzero gradient. With cross-entropy at a perfect prediction (loss 0), two underflow to exactly zero in float32: the tournament's logits and the addition table's carry, which sits behind thousands of damped lookups, stage 1's lowzero edge again. s3h's prediction path is s3g's, unchanged (its changes are in the search, the round's acceptance, the gate lesson and T's lesson). Gray: the program itself is a discrete result of search, not a differentiable parameter; training returns zero logits |
| **14. no data inspection, augmentation, task-specific solvers, custom loops, participant backward** | gray as its parents: the lessons carry designed knowledge (the law library, the template's ladders and step laws, the link law, the loop law, csub); the miner computes statistics over the training batches inside the model; the moved-up rows (CHAIN_NORM) are numbers made from each row's own x and N; the answer rows are chosen by the batch's smallest T (a data-derived rule, no H1 constant) | gray in a new place: the climb is a program search inside the training forward pass, the same kind of thing as the tournament's scoring but over a much bigger space and over many steps; its practice problems are made from the batch rows' own digits (N's top digits and x's at the same places, another row's x, the digits 0-9 run over the probes: no data file is read and no number is made up) and the law library. The curriculum (which laws become practice problems, the register variants, mac) and the round's search space are designed: a strict reader could call the curriculum a task-specific solver written as lessons. No backward call, one optimizer step per evaluator step |
| **9. everything on the GPU** | yes | the search, the probes and the machines are GPU tensors (generators on the device, no `.cpu()`); Python holds the bookkeeping (programs as tuples, BPE over program text, the schedule), as the schedule always did. s3h: each program's primitive instructions are traced on the host from its text (memoized) and copied to the GPU as opcode columns; the machines run them there, long runs replayed from CUDA graphs |
| **11. 3,600 s of H100 training** | yes: 1,861 s | **yes, on every seed** (s3i, the final file): the freeze by 982-1,149 s on the eight H100 runs, under a third of the budget (s3h: 1,341-2,086 s; s3g: by 3,298 s on seed 74, and out of time on two seeds). The first certified build (s3d) needed a relaxed budget (9,142 s), and an earlier build overran the budget by 346 s inside one climb unit (the deadline was checked only between groups of item classes); the search now checks it inside every batch, acceptance batch and vocabulary build |
<!-- /RULES -->

## 6. Is Bar 2 cleared?

<!-- BAR2 -->
**The bar.** "Nothing a strict judge would call the hand-written algorithm." Stage 1 does not clear it: its datapath is
long division's and the schoolbook multiply's, written by hand, and the menus that plug into it hold the right wirings.
Stage 3 removes both: the file holds no datapath and no menu, and the program that answers is one the training run found
by searching a general digit instruction set, merging what recurred, reusing solved problems as routines, and composing
them against the answers.

**What a strict judge could still say.**
1. *The instruction set is digit arithmetic.* Subtract-with-borrow, add-with-carry and multiply-accumulate at a pointer,
   keep-or-discard on the borrow, a bring-in shift and loops by the number's length and the base are what long division
   and the schoolbook multiply are made of. The instructions are general (they compute many things the task never
   needs), but they were chosen knowing the family. Two of them are needed: without keep-or-discard on the borrow
   (SEL.S) or the bring-in (BRG.X), the round is never found ([ABLATION_S3.md](ABLATION_S3.md)).
2. *The curriculum walks the search up the skeleton.* The practice problems are laws, applied to the rows' own
   numbers, but which laws, in which order, is a designer's choice, and two of them (the chain's ladder and the step
   law: red.step) are the definitions of long division's step that stage 1 also uses. LIBRARY.md said it of its own
   in-run sketch: "the curriculum is where 'how' moved." The two studies below narrow this: the step's definitions
   and how the problems are posed are needed, but the hand-picked list and its order are not.
3. *The round's search space* (routines plus straight-line primitives, cost at most 5) and the countdown are designed.
   So is the learning machinery, and two parts of it are needed: routines, and Occam's ties to the program that runs
   fewer steps.

**The call.** Bar 2 is **partly cleared**, and further than before. For the forward pass it is cleared: no
hand-written skeleton or menu runs. The program that answers was discovered inside one training run (the multiply as
a loop of its own accumulating row, the reduction as a loop of its own long-division step, the round as "copy x,
multiply, reduce"), and that run certifies at T = 64 on both ladders, on every seed within the official budget. What
keeps it partial is *the training recipe*, where the algorithm's decomposition still lives. Two studies of the final
file say where (§4 marks every hand-made part with their verdicts):
- **What the result needs** ([ABLATION_S3.md](ABLATION_S3.md): each part removed in turn, one H100 run per arm and
  seed, on seed 74 (the official manifest untouched) and seed 1). Two task-shaped instructions (SEL.S, BRG.X); the
  practice problems that define long division's step (red.step's chain and step); how the multiply's row and the
  registers are posed (mac with the running sum as its addend, designed so that the row becomes a problem of its own;
  every problem also posed with X and Y swapped); and routines and Occam's tie-break. Without any one of them the
  round never comes, on either seed. Each of those failures searched every program up to cost 5 without the missing
  piece, and none reached the 900 s search limit, so more training time would not rescue it; a wider search, up to
  cost 6, found nothing either (ABLATION_S3.md §7). Of the practice problems one at a time, the chain is needed on
  one seed of two (without it the reduction depends on a tie between two equal-cost spellings of the step, which the
  search's listing order breaks), and the step and the multiply's row (one seed) are not; nor is a round cost limit
  above 5 (one seed; at 4 there is no round).
- **What it does not need: the hand-picked list of problems and their order** ([SELF_PRACTICE.md](SELF_PRACTICE.md)).
  The learner poses its own problems from a law book (11 laws on 8 patterns of the data's N and x, each posed in every
  register binding), and one general rule picks what to practise. The round still comes:
  [`self_practice/submission_stage3_sp.py`](self_practice/submission_stage3_sp.py) (md5 f0ef5b06) certifies on the
  official manifest (the round by 746 s) and on 7 of 8 B200 seeds, and
  [`self_practice/submission_stage3_sp2.py`](self_practice/submission_stage3_sp2.py) (md5 afc70fe9) on the official
  manifest (the round by 1,029 s) and on all 8. All nine of s3i's problems appear among its candidates, unmarked.
  What it needs in their place is designed too: the law book, and the choosing rule, picked after bar 3's rules failed
  by watching whether the round came. So are the instruction set and the search sizes.

The official budget no longer keeps it partial. s3i meets it on every seed: the freeze by 1,149 s at the latest
(s3h's by 2,086 s). Practising the pieces in a designed order (the multiply after its row, the reduction after its
chain) was needed on s3e's engine, where the climb without that order reached level 5 at the deadline, about 500 s
short of the round. On s3i's engine, the same search and training run more than twice as fast, and dropping the order
costs 277 s and 321 s (seeds 74 and 1): the run still certifies, frozen by 1,378 s at the latest (ABLATION_S3.md,
prereq_off).

The acceptance tests say the same from the other side: with the routines the H1 run discovered, no other formula's
round is a composition of cost <= 5 (§1.3's table). The curriculum, not the data, bounds what stage 3 can learn.
Self-practice moves that bound into the law book and the round's search: it learns none of the other formulas either
(SELF_PRACTICE.md §4.5). Bar 3 (a general algorithm learner) is still open. INVENTING.md's loop never invents
x^2 mod N without curated practice tasks. Self-practice gets there only with a designed law book, and a choosing rule
picked by watching whether H1's round came.
<!-- /BAR2 -->

## 7. Reproduce

<!-- REPRO -->
From `cpu_first/` with the sotaku venv's Python. Local commands take seconds and stay well under 1 GB; every search with
a real budget, every training run and the gradient tests run on Modal.

Build the two files (each builder asserts the md5 of the certified files it reads; `make_stage3.py` writes
`submission_stage3_s3i.py` and refuses to replace a certified file with a different build, so a new build needs a
new `OUT`):

```bash
~/github/sotaku/venv/bin/python integration/make_stage1.py
```

```bash
~/github/sotaku/venv/bin/python integration/make_stage3.py
```

Local plumbing checks (stage 1: both prediction paths with every table and switch at the truth, and each lesson set's
hard checks at the truth; stage 3: the prediction path with the reader and cells at the truth and the library study's
routines as the round; a few training steps from phase B with toy budgets):

```bash
cd integration && ~/github/sotaku/venv/bin/python local_check.py ../../one-layer-deeper/data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2 N=8
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_designed.py N=4
```

```bash
FROM_B=1 ~/github/sotaku/venv/bin/python integration/checks/stage3_train_smoke.py STEPS=60 B=64 'MODE="designed"' SEARCH_STATES=40000 FALLBACK_STATES=40000 MAX_BODY=1 MAX_COST=4 LEVELS=3 ROUND_STATES=20000 ROUND_COST=3 ANS_MIN=8 CELL_WIN=3
```

The torch search against the library study's numpy search (small budgets):

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_libcore
```

Stage 1's phase-C check and its gradient test (Modal CPU containers):

```bash
~/github/sotaku/venv/bin/python -m modal run integration/checks/modal_checks.py --file submission_stage1.py --seeds 0,1,2,3 --args "STEPS=140" --tag pc1
```

```bash
~/github/sotaku/venv/bin/python -m modal run integration/checks/modal_checks.py --script gradtest --seeds 0 --args "N=16 FILE=submission_stage1.py" --tag grad_s1
```

The evaluator's runs (app and Volume `integration`, records in `integration/records/`), from `../one-layer-deeper`:
`s1_wave1.json` (stage 1: the official run, 8 CPU seeds, the acceptance tasks), `s3_wave2.json` (stage 3: a CPU run
and a relaxed H100 run), `s3_official.json` (stage 3, the untouched manifest), `s3_accept.json` (stage 3's acceptance
tasks, CPU, relaxed):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode plan --plan ../cpu_first/integration/plans/s1_wave1.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/integration/modal_integration.py --mode fetch
```

The round-search proxy (the acceptance tasks' deciding step with a given library, on an H100):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode proxy --plan ../cpu_first/integration/lib/r11_lib.json
```

The climbs alone (app `integration-climb`, records in `integration/lib/records/`), from `cpu_first/`:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/lib/modal_climb.py --name c5_exact_h100 --config '{"seconds": 25000}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/lib/modal_climb.py --name c5_learned_s0_h100 --config '{"learned_cells": true, "seed": 0, "seconds": 25000}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run integration/lib/modal_climb.py --fetch
```

The A/B (§2.1; app `integration-ab`, every limit in `lib/modal_ab.py`), from `../one-layer-deeper`: the profile of one
capped search, the reduction's diagnostic, then variants B and C (A is the c5 runs above):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_ab.py --mode profile --env '{"TASK": "mul", "TAG": "mul_3m"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_ab.py --mode profile --env '{"MODULE": "integration.lib.debug_reduce", "TAG": "reduce_c5rows", "ROWS": "4000", "MAX_STATES": "3000000"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_ab.py --mode climb --name ab_B_exact_h100 --config '{"variant": "B", "cap": 5800000, "fallback_cap": 27000000, "seconds": 18000, "search_s": 1800}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_ab.py --mode climb --name ab_C_exact_h100 --config '{"variant": "C", "cap": 5800000, "fallback_cap": 27000000, "seconds": 18000, "search_s": 1800}'
```

```bash
~/github/sotaku/venv/bin/python ../cpu_first/integration/lib/peek_ab.py ab_B_exact_h100 ab_C_exact_h100
```

Stage 3's s3i (`make_stage3.py` builds `submission_stage3_s3i.py`; `make_stage3.py s3h` checks that the certified s3h
file is still built by its sources, kept in `work/s3h/`, and `make_stage3.py s3g` likewise s3g's in `work/s3g/`), from
`cpu_first/`: the official run on an H100, its seeds 1-7 on the official manifest on H100s, and the eight seeds on
B200s (research), through the shared runner:

```bash
~/github/sotaku/venv/bin/python integration/make_stage3.py
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/submission_stage3_s3i.py --name s3i_official --out integration/records --run s3i-official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/submission_stage3_s3i.py --seeds 1,2,3,4,5,6,7 --hardware h100 --name s3i --out integration/records --run s3i-h100-seeds
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/submission_stage3_s3i.py --seeds 8 --hardware b200 --name s3i --out integration/records/b200 --run s3i-b200-seeds
```

s3i's checks are s3h's below with `submission_stage3_s3i.py` for the file (`test_lockstep` now also checks the round
machine against run_prog's; the T lesson check and the NaN check include s3i). The climb A/B, s3h's libcore (kept as
`lib/libcore_s3h.py`) against s3i's on two H100s, from `../one-layer-deeper`:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_s3h.py --mode run --gpu H100 --envs '[{"MODULE": "integration.lib.profile_climb", "SYNC": "0", "FUSE": "4", "TAG": "pc_s3i_h100_nosync"}, {"MODULE": "integration.lib.profile_climb", "SYNC": "0", "FUSE": "4", "MODULE_FILE": "integration/lib/libcore_s3h.py", "TAG": "pc_s3h_ab_h100_nosync"}]'
```

Seed 6's replays (`debug_patch.py`'s RR_CHECK: every `run_rounds` call also run by s3h's code, compared; plans
`plans/s3i_debug1.json` (with s3h's replay the same day) and `plans/s3i_debug2.json` (the final file's), records
`records/s3h_debug/`):

```bash
~/github/sotaku/venv/bin/python integration/lib/debug_patch.py integration/submission_stage3_s3i.py integration/work/s3i_debug.py
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode plan --plan integration/plans/s3i_debug2.json --out integration/records/s3h_debug --run s3i-debug2
```

s3h (`submission_stage3_s3h.py`, md5 3b9d9bcf, built from its sources now kept in `work/s3h/`): the official run, its
seeds 1-7 on H100s and the eight seeds on B200s, through the shared runner:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/submission_stage3_s3h.py --name s3h_official --out integration/records --run s3h-official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/submission_stage3_s3h.py --seeds 1,2,3,4,5,6,7 --hardware h100 --name s3h --out integration/records --run s3h-h100-seeds
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/submission_stage3_s3h.py --seeds 8 --hardware b200 --name s3hf --out integration/records/b200 --run s3hf-b200-seeds
```

s3h's checks, from `cpu_first/` (local, seconds; the training smoke check on Modal, since it takes about 2 GB): the
lockstep machine against the item-by-item one, bfs for every fuse and the batched acceptance; T's direction lesson
in log space against s3g's; the gate's NaN; the designed check and the smoke check on s3h:

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_lockstep
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_tlesson_check.py
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_nan_check.py
```

```bash
S3_FILE=submission_stage3_s3h.py PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/checks/stage3_designed.py N=4
```

```bash
~/github/sotaku/venv/bin/python -m modal run integration/checks/modal_checks.py --script stage3_train_smoke --file submission_stage3_s3h.py --seeds 0 --args "FROM_B=1 STEPS=60 B=64 MODE=\"designed\" SEARCH_STATES=40000 FALLBACK_STATES=40000 MAX_BODY=1 MAX_COST=4 LEVELS=3 ROUND_STATES=20000 ROUND_COST=3 ANS_MIN=8 CELL_WIN=3" --tag smoke_s3h
```

The diagnostics behind s3h's three fixes (app `integration-s3h`, `lib/modal_s3h.py`: a lib/ module on a GPU; logs in
`lib/records/s3h/`), from `../one-layer-deeper`: the climb alone, every unit timed (`profile_climb`; `MODULE_FILE`
loads another libcore, e.g. `lib/libcore_s3g.py`); one search with every apply_items call timed (`profile_task`); the
round's search on seed 5's dumped pool (`round_pool_diag`); the state hash's merges (`round_hash_diag`):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_s3h.py --mode run --gpu H100 --env '{"MODULE": "integration.lib.profile_climb", "SYNC": "0", "FUSE": "4", "TAG": "pc_g_f4_h100_nosync"}'
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/lib/modal_s3h.py --mode run --gpu B200 --env '{"MODULE": "integration.lib.round_pool_diag", "STEP": "619", "TAG": "rp_s3h_s3g5_619"}'
```

The replays (`lib/debug_patch.py` writes a diagnostic copy of a stage-3 file: REPLAY sets a recorded climb's library
and round programs at its step instead of climbing, DUMP_STEPS prints the answer pool, GRAD_CHECK checks every
gradient in phase D), through the shared runner (plans `plans/s3h_debug1.json`, `plans/s3h_debug2.json`, records
`records/s3h_debug/`):

```bash
~/github/sotaku/venv/bin/python integration/lib/debug_patch.py integration/submission_stage3_s3h.py integration/work/s3h_debug.py
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode plan --plan integration/plans/s3h_debug2.json --out integration/records/s3h_debug --run s3h-debug2
```

s3f (`submission_stage3_s3f.py`, md5 31c9cab2; built when it was made by `make_stage3.py` as `submission_stage3.py`):
the official run, then its seeds (1-7 on the official manifest on H100s; 74 and 1-7 on CPUs, relaxed to 18,000 s):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode plan --plan ../cpu_first/integration/plans/s3f_official.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode plan --plan ../cpu_first/integration/plans/s3f_seeds.json
```

The first certified build (relaxed budget; `submission_stage3_s3d.py`, saved from s3d_relaxed_h100's record):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode plan --plan ../cpu_first/integration/plans/s3d_wave1.json
```

The round's search alone with the library the H1 run discovered, on every acceptance task (cost 5, then 6 with a 40M
state cap):

```bash
ROUND_COST=6 ROUND_STATES=40000000 PROXY_TAG=_cost6_40m ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/integration/modal_integration.py --mode proxy --plan ../cpu_first/integration/lib/s3d_lib.json
```

The answer pool and the round's search inside the submission's own code, from cpu_first/ (H100):

```bash
~/github/sotaku/venv/bin/python -m modal run integration/checks/modal_checks.py --script stage3_pool_check --file submission_stage3.py --seeds 74 --args "STEPS=830" --tag pool_s3_74 --gpu
```

The variants' local test (tiny budgets: the cached acceptance against libcore's, C's mac and mul, the rules, libcore's
switches against C line by line, Occam's step tie-break):

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python -m integration.lib.test_climb_ab
```

Then the tables: `cd integration && ~/github/sotaku/venv/bin/python summarize.py fill INTEGRATION.md`.

**Files** (`cpu_first/integration/`): `make_stage1.py` and `submission_stage1.py`; `stage3_template.py`,
`make_stage3.py` and `submission_stage3_s3i.py` (s3i, the final file: every seed certified on the official manifest,
the freeze by 1,149 s at the latest), `submission_stage3_s3h.py` (s3h, the file before it: every seed certified; its
sources in `work/s3h/`),
`submission_stage3.py` (s3g, certified, official; its sources in `work/s3g/`), `submission_stage3_s3f.py` (certified,
official), `submission_stage3_s3d.py` (certified, relaxed);
`lib/` (`libcore.py`, the core stage 3 inlines (s3i's), `libcore_s3h.py` and `libcore_s3g.py` (s3h's and s3g's, for A/B
timing); `s3h6_lib.json` (s3h seed 6's recorded library and round programs); s3h's tools:
`test_lockstep.py`, `profile_climb.py`, `profile_task.py`, `bench_apply.py`, `modal_s3h.py`, `debug_patch.py`,
`round_pool_diag.py`, `round_hash_diag.py`, `s3g5_pools.json` (seed 5's dumped answer pools), `s3g5_sleeps.txt`,
`s3g6_sleeps.txt` (recorded libraries); `tdsl.py`, `tfast.py`,
`inclimb.py`, `cells.py`, `round_search.py`, `round_proxy.py`, `dslste.py`, their tests, `modal_climb.py`,
`r11_lib.json`: the library study's offline library, for the checks and the proxy; the A/B: `climb_ab.py`,
`modal_ab.py`, `profile_search.py`, `debug_reduce.py`, `debug_round.py`, `peek_ab.py`, `test_climb_ab.py`;
`s3d_lib.json`, `s3f_lib.json`: the libraries the certified runs discovered); `checks/` (the phase-C check, the
gradient tests, stage 3's designed, smoke and answer-pool checks, s3h's `stage3_tlesson_check.py` and
`stage3_nan_check.py`, `modal_checks.py`); `modal_integration.py`,
`wait_fetch.py`, `wait_climbs.py`, `wait_vol.py`, `wait_log.py` (waiters on the Volume and a live log), `peek.py` (live
progress of the runs), `summarize.py`; `plans/`, `records/` (`records/partial/`: the logs of the stopped stage-3 runs;
`records/b200/`: the research seeds on B200s; `records/s3h_debug/`: s3h's and s3i's replays); `work/merged_raw.py` (the
raw 3-way merge, for reading the conflicts), `work/s3g/` and `work/s3h/` (s3g's and s3h's template and libcore),
`work/s3g_debug.py`, `work/s3h_debug.py`, `work/s3h_final_debug.py`, `work/s3i_debug.py` (the replays' diagnostic
copies).
<!-- /REPRO -->
