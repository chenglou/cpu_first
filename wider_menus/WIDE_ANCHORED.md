# The wide-menu machine with no hand-written adder

## In plain words

**What this is.** Two parallel steps, merged:
- [MENUS.md](MENUS.md)'s `submission_wide.py` chooses its own wiring from very wide menus: about 4·10^9 ways to wire
  the long division, 3·10^7 ways to lay out the squaring and 3 reader shapes. But its subtraction table still got its
  start from a hand-written adder, as in WIRING.md's stage 3: from the first step until 70 steps after the reader had
  worked out the prompt, the table was checked against it.
- [ANCHORED.md](../learned_wiring/ANCHORED.md) removed that warm start from stage 3. The table learns what the digits
  mean from the order of the numbers instead: x < N gives the borrows, and two laws of subtraction then fix every digit.

[`submission_wide_anchored.py`](submission_wide_anchored.py) is the wide-menu file with the same change: the
hand-written adder is deleted, and the anchors' lessons take its place. Nothing else changed: the menus, the datapath,
the reader, the switches' lessons and the schedule after the table are `submission_wide.py`'s. No hand-written
arithmetic runs anywhere, in training or in the answer.

**Did it certify?** Yes. The official run used the untouched H1 manifest on an H100 (`wide_anchored_official`, the file
byte for byte):
- every one of 768 questions was right at every T up to 64, on seen and unseen moduli, and every split was exact;
- both tables came out exact and the coverage audit was clean;
- it chose exactly the wiring and layout that `submission_wide.py`'s own official run chose: long division by
  repeated subtraction, with two of MENUS.md's harmless aliases, and the schoolbook rows summed in reverse order.

All 8 CPU seeds certified through the same evaluator, every table exact on each. Seven chose repeated subtraction and
one (seed 1) trial multiples, the other correct long division in the menu; `submission_wide.py`'s seed 1 chose trial
multiples too.

**Times.** On the H100, training took 558 s of its 3,600 s budget (frozen at step 588) and evaluation 41 s of
1,800 s; `submission_wide.py` took 592 s and 43 s. The anchored file is cheaper before its table is exact (step
300 at 123 s against 153 s) and froze 22 steps later. On the CPU seeds training took 1,913–2,361 s and evaluation
448–674 s (`submission_wide.py`: 1,620–3,117 s and 403–879 s); those are CPU times, not the budget's H100.

**What broke, and why.** Nothing in certification, and no new interaction from the wider menus. Every lesson
MENUS.md added (the number rule, the 9,408-combination chain lesson, the step law, x + Q·N mod N = x) already waits
for exact tables, so the anchors' table, exact a few dozen steps later or earlier, only shifts them. Two things carried
over from earlier work:
- **as in stage 3,** the untrained reader read nonsense before the reading (on every diagnostic seed it read N as 0
  on most rows for a step or more); the table had no lesson then, and within two steps of its first lesson the reader
  read every row right (on three of the four seeds, 7–11 steps before it);
- **the rule-8 edge of `submission_wide.py`** is still there: on the deeper gradient test the final-correction switch
  gets exactly 0 gradient on 3 of 4 trained diagnostic models. It comes from the menu's never-firing final
  subtractions (MENUS.md §4.5), not from this change.

The slower turn of T's reading that ANCHORED.md §3.4 found in stage 3 does not show here (§3).

**Still written by hand**: everything ANCHORED.md lists for stage 3, with MENUS.md's wider menus in place of stage
3's:
- the menus themselves, the sizes and the loop over T;
- a + b = a − (0 − b);
- the lessons and their schedule. Among them are the anchors' three statements and MENUS.md's two step laws ("one
  step from x + q·N gives back x", and how the step takes its digit in), which are definitions of long division's
  step.

## Details

### 1. The change

The same edit as ANCHORED.md §1, applied hunk by hunk: `submission_wide.py`'s table lessons are stage 3's line for
line, apart from one extra times-table lesson (the number rule), which is untouched. The diff against
`submission_wide.py` is the diff of `submission_anchored.py` against `submission_io3.py`, plus the docstring:
- removed: the designed adder (`add_hard`, `add_soft`) and `ADDER_WARM`;
- added: `GATE_WIN = 10`, `LATE_AFTER = 100`, `order_pairs`, the top-part facts, the order laws and bijectivity, the
  gate between them;
- the table's hard checks (what phase B, the ladder and the switches wait for) run from step 1, as in ANCHORED.md.

Local checks (WIRING.md's harness as `wider_menus/local/local_wide.py`, each under a minute and 1 GB): at the truth both
prediction paths are exact with identical logits (4 prompts each of the T = 1, 8 and OOD-N 64 rungs); a smoke test runs
every new lesson path on a random table with the true reader.

### 2. Results

The official run is the untouched manifest `benchmark/manifests/h100_hard_h1.json` on a Modal H100, the file byte for
byte (md5 `c82b11fa8a93b09c0724c3931cf8de98`); the CPU seeds are the same evaluator with the device set to cpu and the
seed changed. `submission_wide.py`'s nine runs (MENUS.md) follow for comparison. "Never trained" is the largest count
of table rows, cells or gate rows that evaluation used but training never did.

<!-- RUNS -->
| run | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtraction, product, gate | division chain | multiply grouping | layout | shape | reading chosen | digit lessons | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wide_anchored_official (H100) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | 134 | 179 | 199 | 234 | 352 | 385 | 409 | 588 | 0 | 558.0 | 41.4 |
| wide_anchored_cpu_seed74 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | 134 | 179 | 199 | 234 | 352 | 385 | 409 | 588 | 0 | 1954.6 | 553.0 |
| wide_anchored_cpu_seed1 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit | 288/288 | step | 145 | 180 | 223 | 243 | 280 | 393 | 423 | 454 | 633 | 0 | 2131.5 | 479.6 |
| wide_anchored_cpu_seed2 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 99 | 141 | 184 | 204 | 241 | 354 | 376 | 468 | 587 | 0 | 2249.9 | 485.6 |
| wide_anchored_cpu_seed3 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 115 | 156 | 199 | 219 | 256 | 373 | 407 | 440 | 619 | 0 | 2006.2 | 572.0 |
| wide_anchored_cpu_seed4 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 135 | 243 | 283 | 303 | 343 | 465 | 495 | 525 | 704 | 0 | 2360.9 | 673.8 |
| wide_anchored_cpu_seed5 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 121 | 162 | 204 | 224 | 262 | 389 | 419 | 451 | 630 | 0 | 2264.4 | 480.2 |
| wide_anchored_cpu_seed6 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 136 | 172 | 215 | 235 | 272 | 376 | 411 | 445 | 624 | 0 | 1912.6 | 448.1 |
| wide_anchored_cpu_seed7 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 105 | 170 | 212 | 232 | 270 | 391 | 422 | 522 | 634 | 0 | 2292.1 | 517.7 |
| wide_official (H100) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | - | 97 | 149 | - | 314 | 361 | 387 | 566 | 0 | 591.8 | 42.9 |
| wide_cpu_seed74 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | - | 97 | 149 | - | 314 | 361 | 388 | 567 | 0 | 1619.6 | 477.2 |
| wide_cpu_seed1 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit | 288/288 | step | 140 | - | 140 | 193 | - | 361 | 400 | 441 | 620 | 0 | 1790.4 | 432.2 |
| wide_cpu_seed2 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 99 | - | 138 | 167 | - | 327 | 350 | 440 | 569 | 0 | 1784.5 | 879.1 |
| wide_cpu_seed3 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 115 | - | 157 | 210 | - | 374 | 424 | 468 | 647 | 0 | 1716.9 | 403.2 |
| wide_cpu_seed4 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 135 | - | 135 | 246 | - | 417 | 457 | 499 | 678 | 0 | 3117.0 | 551.8 |
| wide_cpu_seed5 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 121 | - | 121 | 239 | - | 414 | 454 | 489 | 668 | 0 | 2270.5 | 666.1 |
| wide_cpu_seed6 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 134 | - | 134 | 192 | - | 340 | 388 | 478 | 613 | 0 | 2157.5 | 477.8 |
| wide_cpu_seed7 (CPU) | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 105 | - | 105 | 200 | - | 368 | 390 | 486 | 623 | 0 | 2598.6 | 655.8 |
<!-- /RUNS -->

The coverage audit (CHAIN.md: how often training's hard checks used each table row and cell, and what evaluation used
that training never did). As in ANCHORED.md §2.3, the training counts include the hard checks from step 1, before the
table's first lesson:

<!-- AUDIT -->
| run | subtraction rows used at evaluation | fewest training visits of a row | product cells used at evaluation | fewest training visits of a cell | count-down rows used | gate rows used | used at evaluation but never trained (rows, cells, count-down, gate) |
|---|---|---|---|---|---|---|---|
| wide_anchored_official | 200/200 | 1032239 | 100/100 | 193135 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed74 | 200/200 | 1032240 | 100/100 | 193135 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed1 | 200/200 | 1001076 | 100/100 | 200598 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed2 | 200/200 | 1046124 | 100/100 | 191111 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed3 | 200/200 | 1102810 | 100/100 | 201005 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed4 | 200/200 | 1214045 | 100/100 | 196230 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed5 | 200/200 | 906966 | 100/100 | 197608 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed6 | 200/200 | 848259 | 100/100 | 206426 | 24 | 2/2 | 0, 0, 0, 0 |
| wide_anchored_cpu_seed7 | 200/200 | 842882 | 100/100 | 198774 | 24 | 2/2 | 0, 0, 0, 0 |
<!-- /AUDIT -->

Seconds to a training step in the two official runs, from the evaluator's progress lines:

<!-- TIMES -->
| run | frozen at step | T settled at step | s to step 100 | s to step 200 | s to step 300 | s to step 400 | s to step 500 | s to step 600 | s to step 1000 | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wide_anchored_official | 588 | 409 | 33.0 | 66.9 | 123.4 | 195.3 | 310.7 | 410.3 | 413.4 | 558.0 | 41.4 |
| wide_official | 566 | 387 | 37.7 | 88.3 | 153.2 | 251.8 | 362.8 | 434.4 | 437.4 | 591.8 | 42.9 |
<!-- /TIMES -->

### 3. The interactions

The same diagnostics as ANCHORED.md §3 (`anchored/local_wide_anchored.py`: the reader checked against each row's own
numbers before the step's lessons, the table's borrows and digits against true subtraction, a line every 5 steps, and
at the end every depth rung on both prediction paths and `local_wide.py`'s two gradient tests), on seeds 74, 1, 2
and 4:

<!-- DIAG -->
| run | most rows read as N = 0 before the reading (logged steps with 90% or more) | N and x read right on every row from | reading chosen: the table's first lesson | most rows misread after it | borrows 200/200 | digit lessons | digits 200/200 | the table's checks all pass from (logged steps failing after) | phase B | ladder | pieces' lessons | chain lesson | I/O lesson | T settled | frozen | rungs exact on both paths, identical logits | gradient test, T = 1 and 2 | T = 8 and OOD-N T = 4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1_s74 | 1.00 (2) | 95 | 105 | 0.0000 | 135 | 141 | 175 | 175 (0) | 185 | 205 | 241 | 357 | 390 | 415 | 594 | 14/14 | True | False (0 for sw.final) |
| w1_s1 | 1.00 (2) | 145 | 152 | 0.0000 | 180 | 186 | 215 | 220 (0) | 228 | 248 | 286 | 398 | 428 | 458 | 637 | 14/14 | True | False (0 for sw.final) |
| w1_s2 | 1.00 (3) | 100 | 98 | 0.0000 | 140 | 145 | 175 | 175 (0) | 188 | 208 | 245 | 358 | 380 | 472 | 591 | 14/14 | True | True |
| w1_s4 | 1.00 (1) | 125 | 136 | 0.0000 | 225 | 230 | 260 | 260 (0) | 270 | 290 | 330 | 452 | 483 | 513 | 692 | 14/14 | True | False (0 for sw.final) |
<!-- /DIAG -->

**The reader before the reading.** As in stage 3 the untrained reader read nonsense: on every seed it read N as 0 on
90% of the rows or more for one to three logged steps before the reading was chosen. The table had no lesson then.
On seeds 74, 1 and 4 the snapped reader read N and x right on every row from 7–11 steps before the reading was chosen.
On seed 2 the last logged misreads were at step 95, three steps before it (98). At those steps every row was read
whole and in the right order (x < N on every row) but with wrong digits, and from step 100 on every row was right.
No row was misread after that on any seed.

**The table.** The borrows were all right 28–89 steps after the reading (seed 4 the slowest again: its table starts
from the same random numbers as in stage 3, since the tables are the model's first parameters). The digits were all
right 29–34 steps after the digit lessons opened. The table's hard checks all passed from 68–124 steps after the
reading and never failed on a logged step after that.

**The wider menus' own lessons.** Every lesson MENUS.md added waits for exact tables, as stage 3's switches do:
- the multiply's number rule joins with the times table's ladder;
- the chain lesson (9,408 combinations of the chain's switches and the gate's rows) joins once the table and the
  ladder have been exact for 20 steps;
- the I/O lesson (its step law and the laws x mod N = x, N mod N = 0, x + Q·N mod N = x) joins after the chain.

So a table that is exact a few dozen steps later or earlier only shifts them. The ladder-to-chain wait, which is the
times table, the layout and the number rule settling, is long in the wide model with or without the anchors: 150–162
steps in these diagnostics and 153 in the official run here, against 165 in `wide_official`.

Both correct long divisions still occur. Seeds 74, 2 and 4 chose repeated subtraction and seed 1 trial multiples (in
`submission_wide.py`'s CPU runs, seed 1 also chose trial multiples). Each came with some of MENUS.md's harmless aliases:
the digit added into the empty place, or a final subtraction that never fires.

**Rule 8: the same edge as before.** At the end of each diagnostic run, `local_wide.py`'s first gradient test
(T = 1 and 2) passes on all four. The deeper one (T = 8 and OOD-N T = 4) passes on seed 2. On seeds 74, 1 and 4 it
gives exactly 0 gradient to one switch, the final correction. That is the edge MENUS.md §4.5 found in
`submission_wide.py` itself: the menu's never-firing final subtractions are aliases whose gradients cancel, and the
one option that differs carries almost no weight. It comes from the wide menu, not from this change, and it would go
with a menu without those aliases.

**T's direction.** ANCHORED.md §3.4 found that in stage 3, when T's reading started backwards, T's lesson took longer to
turn it round with the anchors than without (108–239 steps against 69–93). Phase B's fresh optimizer, which clears
Adam's state, moves here too: it comes once the table is exact (steps 179–283) instead of at or soon after the reading
(97–157). But T's turn is not slower here:

<!-- TCPU -->
| seed | this file: reading, phase B, I/O lesson, T settled | steps from the I/O lesson to T settled | submission_wide.py: reading, phase B, I/O lesson, T settled | steps |
|---|---|---|---|---|
| 74 | 97, 179, 385, 409 | 24 | 97, 97, 361, 388 | 27 |
| 1 | 145, 223, 423, 454 | 31 | 140, 140, 400, 441 | 41 |
| 2 | 99, 184, 376, 468 | 92 | 99, 138, 350, 440 | 90 |
| 3 | 115, 199, 407, 440 | 33 | 115, 157, 424, 468 | 44 |
| 4 | 135, 283, 495, 525 | 30 | 135, 135, 457, 499 | 42 |
| 5 | 121, 204, 419, 451 | 32 | 121, 121, 454, 489 | 35 |
| 6 | 136, 215, 411, 445 | 34 | 134, 134, 388, 478 | 90 |
| 7 | 105, 212, 422, 522 | 100 | 105, 105, 390, 486 | 96 |
<!-- /TCPU -->

Where T's turn was long (about 90–100 steps from the I/O lesson), it took 92 and 100 steps here (seeds 2 and 7) and
90, 90 and 96 in `submission_wide.py` (seeds 2, 6 and 7). In the diagnostics T settled 25–92 steps after the I/O
lesson. So the moved reset does not by itself slow T's turn, which agrees with ANCHORED.md's finding that it is not the
whole story there.

### 4. Reproduce

The official run and the CPU seeds go through MENUS.md's unchanged runner `modal_wide.py` (app and Volume
`wider-menus`), with the file passed by name; `wait_fetch_wide.py` downloads the records to
`cpu_first/runs/wider_menus/<name>.json`. The CPU seeds run as two plans, because the runner caps a plan at 7 CPU
containers.

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/wider_menus/modal_wide.py --mode official --name wide_anchored_official --file submission_wide_anchored.py
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/wider_menus/modal_wide.py --mode plan --plan ../cpu_first/wider_menus/anchored/plans/wide_anchored_cpu_a.json --file submission_wide_anchored.py
```

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/wider_menus/modal_wide.py --mode plan --plan ../cpu_first/wider_menus/anchored/plans/wide_anchored_cpu_b.json --file submission_wide_anchored.py
```

```bash
cd ~/github/one-layer/cpu_first/wider_menus && ~/github/sotaku/venv/bin/python wait_fetch_wide.py wide_anchored_official wide_anchored_cpu_seed74 wide_anchored_cpu_seed1 wide_anchored_cpu_seed2 wide_anchored_cpu_seed3 wide_anchored_cpu_seed4 wide_anchored_cpu_seed5 wide_anchored_cpu_seed6 wide_anchored_cpu_seed7
```

The diagnostics (`anchored/local_wide_anchored.py`, ANCHORED.md's harness on `local_wide.py`) run through
`anchored/modal_diag_wide_anchored.py` (app and Volume `wider-menus-anchored`, at most 4 containers; logs to
`anchored/logs/`); `anchored/summarize_wide_anchored.py fill` regenerates this write-up's tables:

```bash
cd ~/github/one-layer/one-layer-deeper && ~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/wider_menus/anchored/modal_diag_wide_anchored.py --plan ../cpu_first/wider_menus/anchored/plans/w1.json
```

```bash
cd ~/github/one-layer/cpu_first/wider_menus/anchored && ~/github/sotaku/venv/bin/python summarize_wide_anchored.py fill ../WIDE_ANCHORED.md
```

### 5. Files

In `cpu_first/wider_menus/`:
- **`submission_wide_anchored.py`**: the file, certified byte for byte (md5 `c82b11fa8a93b09c0724c3931cf8de98`);
- `WIDE_ANCHORED.md`: this write-up;
- `anchored/`: `local_wide_anchored.py`, `modal_diag_wide_anchored.py`, `summarize_wide_anchored.py`, `plans/`
  (`wide_anchored_cpu_a.json` and `_b.json`: the CPU seeds; `w1.json`: the diagnostics), `logs/`;
- records: `cpu_first/runs/wider_menus/wide_anchored_*.json`.

The certified files are untouched: `submission_wide.py` (`43db5606…`), `learned_wiring/submission_anchored.py`
(`6328a7d2…`) and the earlier ones.
