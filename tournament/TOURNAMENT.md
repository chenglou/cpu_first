# The tournament: the machine picks its own wiring inside training

## In plain words

**The question.** The bridge ([BRIDGE.md](../bridge/BRIDGE.md)) showed that this project's digit machine is one weight
setting of a generic looped transformer, and that the transformer's attention can be loosened only so far before
training stops finding the wiring. With a short menu per connection (up to 10 choices) it learns; with free
attention, where every connection can read any of about 34 nearby positions, it never did. Yet a search that tried
whole combinations of connections, block by block, found the one right wiring among 5·10^10 in 45 seconds. That
search ran outside the model, so it did not count under the competition's rules. Isn't search learning, though?
Learning is the data choosing among hypotheses. The question here: can that search run inside ordinary training, as
something the model learns, and does it then find the wiring by itself?

**How the model now picks its wiring: a tournament.** Group the connections into blocks of ones that work together
(in the tiny machine: the shift and the three inputs of the subtraction; then the three inputs of the keep-or-discard
step). Every complete combination of a block is a contestant, and the model holds one number per contestant, its
weight. Then, at every training step:
1. every contestant still in the running is plugged into the machine and run on that step's own numbers;
2. it is scored pass or fail against simple facts that need no answer key: a number below N comes out unchanged, N
   comes out as 0, the subtraction "borrows" exactly when the number is below N. Pairs of contestants from two
   blocks are also scored on the answers of the questions that come with one;
3. the training loss is "minus the log of the total weight of the contestants that passed". One ordinary gradient
   step on it raises every passing contestant's weight and lowers every failing one's;
4. a contestant whose weight falls far behind the leader (by e^12) stops being run. That is what keeps it cheap.

There is no separate search loop: the competition's own training loop runs, one step at a time, and the elimination
happens through the gradient. The digit arithmetic (a small neural network for "subtract with a borrow") learns at
the same time from its own lessons, and the wiring's lessons wait until it passes them.

**How wide the choice is.**
- **The tiny machine** (base 3, the bridge's grid): each of the 7 connections reads one of 34 relative positions,
  free attention's width. That is 5.3·10^10 wirings, against 823,543 for the bridge's 7-option menu, which soft
  training could still learn, and 62.7 million for the 13-option menu, where it already failed.
- **The H1 machine** (the competition's task): its division's chain, the part that subtracts N again and again, got
  the same treatment. Each operand can be any of 7 registers moved up or down by up to 3 places; the subtraction's
  borrow can come from 1, 2 or 3 places below or not at all; the keep-or-discard decision can read the borrow out of
  any place; and what a step writes back can be any of 7 values at any offset. The chain's choices grow from 9,408
  to 7.9·10^9, and the whole machine's from about 4·10^17 to about 3·10^23.

**Did it work?**
- **Tiny: yes, every time the facts and the answers pin the wiring down.** The tournament found the one right wiring
  among 5.3·10^10 in every run of the recipe (8 of 8), 11 to 12 steps after the digit cell passed its checks, and the
  network then certified on bigger sizes. The same held with the digit tables given, with two quotient digits, at
  base 4, with 16 questions per step, with 10% of the questions labelled, and with wider windows of 52 and 61
  positions per connection (10^12 and 3·10^12 wirings), each in every run. Soft attention at 34 positions found it in
  none of its runs (BRIDGE.md).
- **H1: yes, after two fixes.** <!-- PLAIN_H1 -->On the untouched official manifest on an H100, the file run byte for
  byte reached T = 64 on seen and unseen moduli: every rung 768/768, every split exact, 2,114 s of training (the
  budget is 3,600) and 90 s of evaluation. It chose long division by repeated subtraction among the chain's 7.9·10^9
  wirings, with both digit networks exact. All 8 CPU seeds certified too, through the same evaluator. A first
  version without the loop law (below) certified on 3 of its 8.<!-- /PLAIN_H1 -->

**What broke, and why.**
- **When the checks do not single out the wiring, the tournament crowns an impostor.** In the tiny machine, the facts
  that need no answers pass 4,391 wirings. 4,327 of them are the bridge's old "bypass" trap: they read the prompt's
  own clean numbers where the machine reads its computed ones, which on a single subtraction looks the same. 36 more
  read a constant zero, the "do nothing" trap. Only the answers rule them out. With no answers, all 8 runs ended on a
  bypass; with only 2 answers, 5 of 8 did. The tournament finds what the checks identify, and nothing more.
- **At H1 the wider menu broke identification itself.** No training modulus has 8 digits, so the register's top
  place is never used in training. Wirings that differ from the right one only there pass every training check and
  fail on the unseen 8-digit moduli: an operand moved up one place and back, or the keep-or-discard step reading the
  borrow out of the second-highest place instead of the highest. Counted exactly: 88 chain wirings pass every
  training lesson, and 78 of them are wrong on 8-digit moduli. Without a fix, 4 of 8 CPU seeds chose such a wiring
  and failed the unseen moduli. The fix is a lesson that reaches the top place: the chain's checks also run on each
  training row's numbers moved up until N fills all 8 places.
- **A second gap: the gate's meaning could flip.** Letting a link choose what it writes back made two forms of the
  same chain equally good: "on no borrow, commit and write the result" and "on no borrow, hold, and write the
  result on hold". The chain cannot tell them apart. But the same keep-or-discard table also decides whether the
  countdown runs another round, and in the flipped form the rounds never run. A first version of the H1 file had no
  lesson for this; 3 of its 8 CPU seeds certified and the other 5 chose the flipped form, never got anything right,
  and ran out of time. The fix is the loop's own law: every question's count is at least 1, so its first round must
  run, and a count of 0 must stop the loop. With both fixes, exactly 5 chain wirings pass every training lesson,
  all of them correct.
- **Scoring each row separately fails.** If every row of a batch is its own little vote, a team of impostors that
  each pass some of the rows can share the weight between them, and the one wiring that passes every row never pulls
  ahead: 0 of 8 such runs found it. Pass or fail on the whole batch is what makes losers lose.
- **Some digit cells never learned their table**, so the wiring's lessons never started: 2 of 8 tiny runs with the
  H1 file's cell form (a table read off an MLP's outputs) and 3 of 8 at base 5. That is the cell, not the tournament:
  the bridge saw the same stalls (BRIDGE.md §3).

**What is still hand-written.**
- The skeleton the connections plug into: in the tiny machine the grid of cells, which cell subtracts, copies or
  keeps-or-discards, and the number of ticks; at H1 the whole datapath of the certified file.
- The menus' windows (34 relative positions, offsets up to 3 places), which connections form a block, and the order
  of the blocks.
- The lessons: the facts about x and N, the one-link checks, the definition of one division step (H1's ladder), the
  digit cell's own lessons, and at H1 the two new ones, the rows moved up until N fills 8 places and the loop's law.
  A strict reader can call some of these definitions of the algorithm (§6).

## Contents

1. The tiny tournament
2. Tiny results
3. What the checks let through: the traps, counted
4. Cost
5. H1: the chain at free attention's width
6. Rules 7, 8 and 14
7. Caveats
8. Reproduce

## 1. The tiny tournament

`tiny.py`. The machine is the bridge's tiny long division (BRIDGE.md §1): base B, two-digit moduli, dividends below
N·B^k, laid out as a grid of cells, one cell per table application; three attention heads copy each cell's inputs
from other cells, one rule per cell kind computes it, and the block runs D ticks. The subtraction's rule is a ReLU MLP
(the bridge's notch 1d, its best cell: 31 of 32); the copy and the select are compiled.

**The wiring at free attention's width.** Each of the seven ports a kind uses (SHIFT.A, SUB.A, SUB.B, SUB.C, SEL.A,
SEL.B, SEL.C) reads one of blocksearch.py's 34 options: the bridge's 7 semantic rules plus every other relative offset
in rows −4..0 and columns −3..3 that the causal mask allows. A port's option is tied over every cell of its kind, as a
relative-position attention would be. 34^7 = 5.3·10^10 wirings.

**The blocks.** The ports are split into the two blocks blocksearch.py found to interact:
- "sub": SHIFT.A and the three ports of the subtraction, 34^4 = 1,336,336 combinations;
- "sel": the three ports of the select, 34^3 = 39,304.

Each block holds one logit per whole combination, initialized at 0.1 × normal.

**One training step** (one optimizer step on one loss, as the evaluator runs it):
- **The cell's lessons**, from the bridge (ANCHORS.md's lessons on a cell): x < N on every top part, the column and
  row order laws, bijectivity. They query the subtraction's MLP directly on made-up ports, so they do not depend on
  the wiring.
- **The gate.** The wiring's lessons join once the snapped table passes all of those lessons for 5 steps in a row
  (WIRING.md: the switches wait for the tables).
- **Block "sub"'s law.** Every active combination runs on a grid of one link (the same network on a smaller grid) on
  the batch's own numbers, x = each question's register start R0 (below N) and x = N. It passes if the subtraction's
  final borrow is [x < N] on every row. That is blocksearch.py's stage 1. Only x < N is used, an order fact of the
  data.
- **The pairs.** Every pair of a "sub" combination that passed and an active "sel" combination runs on the same grid:
  x below N must come out unchanged, N must come out 0, and the borrow must still be [x < N] (the one-link law). The
  pairs that pass then run on the full training grid, on the batch's labelled questions, and must get every answer
  right.
- **The loss** of each lesson is the mixture's, −log Σ_c π(c)·[c passes], with π the softmax of the block's logits over
  every combination (for pairs, the product of the two blocks' π). Verdicts are computed under no_grad, inside the
  loss. The gradient raises the passing combinations' logits and lowers the failing ones'. Under Adam each logit moves
  by about the learning rate per step whatever its weight, so the tournament is close to "+0.5 for a pass, −0.5 for a
  fail" (lr 0.5 on the combinations' logits; 0.01 on the MLP, the H1 file's `LR_CELL`).
- **The active set.** A combination is run only while its logit is within 12 nats of its block's best. The others
  count as failing; each carries less than e^−12 of the leader's weight.
- **Batch verdicts.** A combination passes a lesson if it passes on every row of the step's batch (the `row` variant
  uses WIRING.md's per-row mixture instead).

**Measured.** Every 5 steps the argmax network (each block's best combination, the snapped MLP) runs on every training
question. A run stops 30 steps after it first gets every answer right, or at 600 steps. It is certified, as in the
bridge, if it is also exact on every question of (B, w, k + 1) and (B, w + 1, k + 1).

**The prediction path** (`answer_ste`, rule 8): each block's argmax combination as a straight-through one-hot over all
of its combinations. Its value is exactly the argmax wiring. Its gradient reaches every combination's logit through
the marginal of each port's option. The MLP's table is straight-through too. `grad_test` runs it on 16 training
questions and checks that the answers equal the hard network's and that the cross-entropy's gradient reaches both
blocks' logits and every MLP weight.

The verdict code reproduces blocksearch.json exactly: 114 "sub" combinations pass, 4,391 pairs pass the one-link law,
and exactly one, the true wiring, fits every training answer.

## 2. Tiny results

Modal sweeps (`modal_tiny.py`: suites `main`, `slow`, `wide`; one 4-CPU container per run; records in
`runs/<suite>.jsonl`). 8 seeds per variant. "Steps from the gate to the true wiring" counts from the step the wiring's
lessons joined to the step the argmax wiring became the true one; "steps to the first exact network" is measured
every 5 steps.

<!-- TINY -->
| variant | size | certified | true wiring | steps from the gate to the true wiring | steps to the first exact network | wall s, median | grad test | failures |
|---|---|---|---|---|---|---|---|---|
| the recipe: MLP subtraction cell learned, the one-link laws and every answer, gate on, lr 0.5 | (3, 2, 1) | 8/8 | 8/8 | 12 (11-12) | 137.5 (115-165) | 327 | 8/8 | - |
| the truth tables given: the wiring alone | (3, 2, 1) | 8/8 | 8/8 | 12 (11-12) | 15 (15-15) | 328 | · | - |
| two quotient digits | (3, 2, 2) | 8/8 | 8/8 | 12 (11-12) | 137.5 (115-165) | 341 | 8/8 | - |
| base 4 (three links per step) | (4, 2, 1) | 8/8 | 8/8 | 12 (11-12) | 125 (100-145) | 403 | 8/8 | - |
| base 5 (four links per step) | (5, 2, 1) | 5/8 | 5/8 | 12 (11-12) | 120 (100-135) | 697 | 7/8 | 3 table never learned |
| 10% of the questions labelled (10 of 99) | (3, 2, 1) | 8/8 | 8/8 | 12 (11-12) | 137.5 (115-165) | 318 | 8/8 | - |
| 2% labelled (2 of 99) | (3, 2, 1) | 3/8 | 3/8 | 11 (11-12) | 145 (140-165) | 363 | 8/8 | 5 bypass |
| no answers: label-free laws only | (3, 2, 1) | 0/8 | 0/8 | · | · | 375 | 8/8 | 8 bypass |
| minibatches of 32, 5% labelled | (3, 2, 1) | 7/8 | 7/8 | 23 (20-35) | 145 (125-185) | 420 | 8/8 | 1 bypass |
| minibatches of 16 questions | (3, 2, 1) | 8/8 | 8/8 | 12 (11-13) | 140 (115-155) | 322 | 8/8 | - |
| the wiring's lessons from step 1 (no waiting for the table) | (3, 2, 1) | 8/8 | 8/8 | 120 (96-151) | 122.5 (100-155) | 1830 | 8/8 | - |
| verdicts per row (WIRING.md's mixture) instead of per batch | (3, 2, 1) | 0/8 | 0/8 | · | · | 10422 | 8/8 | 8 bypass |
| lr 0.2 on the combinations' logits | (3, 2, 1) | 8/8 | 8/8 | 26 (25-28) | 152.5 (130-180) | 707 | 8/8 | - |
| lr 0.05 (the H1 file's switch rate) | (3, 2, 1) | 8/8 | 8/8 | 90.5 (87-98) | 220 (195-245) | 2983 | 8/8 | - |
| the H1 file's cell form: the table an MLP's outputs (bridge notch 1e) | (3, 2, 1) | 6/8 | 6/8 | 12 (11-12) | 77.5 (70-90) | 293 | 7/8 | 2 table never learned |
| a wider window: rows -5..0, columns -4..4 (52 options per port, 1.0e12 wirings) | (3, 2, 1) | 8/8 | 8/8 | 14 (13-15) | 140 (120-170) | 1516 | 8/8 | - |
| rows -6..0, columns -4..4 (61 options, 3.1e12 wirings) | (3, 2, 1) | 4/4 | 4/4 | 14 (13-15) | 130 (120-145) | 2413 | 4/4 | - |
<!-- /TINY -->

<!-- TINY_READING -->
- **The recipe finds the true wiring every time, at every size tried.** 8 of 8 at (3, 2, 1), (3, 2, 2) and (4, 2, 1),
  always the true wiring itself (no alias), 11 to 12 steps after the wiring's lessons joined. Without the cell to
  learn (tables given) it takes the same 12 steps; with the cell, most of a run is the cell learning its table (the
  gate opens at steps 103-154).
- **12 steps is the learning rate's arithmetic.** Under Adam every passing logit rises and every failing one falls by
  about lr a step, whatever its weight. Block "sub"'s 114 passers therefore take over its weight after about
  ln(1,336,336 / 114) / (2 · lr) steps, and only then does the pairs' verdict, which only the true "sub" combination
  passes, separate them. That predicts 9 steps at lr 0.5 (observed 11-12), 24 at lr 0.2 (25-28) and 94 at lr 0.05
  (87-98). Every run at every rate certified; the slower rates cost more (§4).
- **A wider window changes the cost, not the outcome.** At 52 options per port (rows −5..0, columns −4..4: 1.0·10^12
  wirings) 8 of 8 runs found the true wiring, 13-15 steps after the gate; at 61 options (3.1·10^12) 4 of 4 did, also
  13-15 steps after it. Block "sub"'s law passes 114, 168 and 195 combinations at 34, 52 and 61 options, and the
  step count follows ln(block / passers) / (2 · lr): 9, 11 and 11 predicted, 11-15 observed.
- **Batches of 16 questions and 10 labelled questions change nothing measurable.** Every step's batch still contains
  answers that rule the bypasses out, and the true wiring passes every batch.
- **2 labelled questions are too few** (3 of 8), and no answers at all never work (0 of 8): §3 shows why.
- **The wiring's lessons need not wait for the table.** With them on from step 1, every run certified (8 of 8), the
  true wiring arriving within one measurement (5 steps) of the table becoming exact. While the table is wrong, block
  "sub"'s law already passes the 114 combinations it will pass at the end (the cell learns its borrows before its
  digits), but no pair passes: the true wiring fails together with every impostor, so nothing is rewarded wrongly.
  Waiting only saves compute: these runs evaluate every combination at every step until the table is right, a median
  of 1,830 s a run against 330 s.
- **Noisy batches slow the tournament without fooling it, as long as the answers identify.** With minibatches of 32
  and 5% of the questions labelled (the H1 regime: few answers, different rows every step), 7 of 8 runs found the true
  wiring, 20-35 steps after the gate instead of 12: a step whose batch carries no deciding answer moves the passing
  impostors along with the true wiring. The eighth ended one port short (§3).
- **Verdicts must be per batch, not per row.** With WIRING.md's per-row mixture (each row's term is −log of the
  weight of the combinations that pass that row), <!-- ROW -->0 of 8 runs<!-- /ROW --> found the wiring. The
  pairs' loss stalls near 1.06 with the weight spread over a team of impostors (the bypass's select among them), each
  passing some of the rows, while the one pair that passes every row is there all along (`pass_pairs` 1). Under Adam
  every combination whose net push is upwards rises by about the same amount a step, so the true pair never pulls
  ahead; with a verdict per batch the impostors simply fail. These runs also cost 2.5-5 hours each, since a verdict per
  row cannot stop at the first row that fails. (The H1 file keeps the certified file's per-row mixture for its two
  I/O blocks, whose few combinations fail in different rows; there it worked in every run.)
- **At base 5** (four links per step), 5 of 8 runs certified, each 11-12 steps after the gate. The other 3 are the
  cell's failures: the MLP never learned its table (3-5 of its 50 entries right), so the gate never opened.
- **The H1 file's cell form** (the table read off an MLP's outputs, bridge notch 1e) stalled in 2 of 8 seeds at 5 of
  18 entries, the stall the bridge saw at (3, 2, 1) and (5, 2, 1) (BRIDGE.md §3). Its gate then never opened. When it
  learned, it learned faster than the direct MLP cell (gate at steps 56-78), and the wiring followed 11-12 steps later.
  The one run whose gradient test failed is one of the two stalled ones: some of its MLP's units never activate.
<!-- /TINY_READING -->

## 3. What the checks let through: the traps, counted

`traps.py` (record `records/traps.json`): with the truth tables, every combination that passes block "sub"'s law and
every pair that passes the one-link law, each classified by what its wrong ports read on the training grid. "Bypass":
some wrong port reads a prompt cell the machine does not read. "Constant source": a wrong port reads only zeros.
"Impostor": wrong ports that read other computed cells.

<!-- TRAPS -->
| what passes | combinations | true | bypass | constant source | other impostor | fit every training answer |
|---|---|---|---|---|---|---|
| block "sub"'s law (the subtraction borrows exactly when x < N, one link) | 114 of 1,336,336 | 1 | 112 | 0 | 1 | · |
| + the one-link law (x below N unchanged, N to 0), over every select | 4,391 of 4,480,656 | 1 | 4327 | 36 | 27 | 1 (the true one) |
<!-- /TRAPS -->

<!-- TRAPS_READING -->
- **The label-free laws let almost only bypasses through.** On a single link, the register the subtraction should
  read (the shift row) holds the same digits as the prompt's own register moved one column. So a subtraction that
  reads the prompt directly passes block "sub"'s law, and 112 of its 114 survivors do exactly that. Of the 4,391 pairs
  that also pass the one-link law, 4,327 are such bypasses, 36 read their new value from a constant zero (on one link
  the only number that is actually reduced is N, whose result is all zeros, BRIDGE.md §4), 27 are other impostors,
  and 1 is the true wiring.
- **The answers leave only the true one**, at free attention's width as at every width the bridge counted.
- **When the answers are too few, the bypass wins.** Every tiny run that failed with a learned table (8 of 8 with no
  answers, 5 of 8 with 2, 1 of 8 with 5% labelled in minibatches) ended on the same core or half of it: the
  subtraction's first input reads the prompt's register moved one column (`rel-2,-1`) instead of the shift row, and
  the select's kept value reads it too (`rel-3,-1`; the minibatch run got everything else right). That is right on a
  question's first link and wrong on its second, so those networks answer the questions whose quotient digit is at
  most 1: 13-45% of the training questions. With no answers, 4,029 pairs pass every step and the tournament stays
  spread over them (the leader holds about 2% of block "sub"'s weight): it picks one at random.
- **The constant-source trap and the bypass never win when the answers are in the verdicts.** They are BRIDGE.md's
  two failures of free attention (§6 and §8). There, every port blended many sources and the gradient pulled towards
  the clean prompt or a zero. Here each whole combination keeps a verdict of its own, so a bypass that fails one
  answer simply loses.
<!-- /TRAPS_READING -->

## 4. Cost

<!-- COST -->
**Tiny.** On a 4-CPU Modal container, a step of the recipe costs about 24 s while every combination is still
active (1,336,336 for block "sub", then 4.5 million pairs; each verdict first on 4 rows, then on all rows for the
combinations that pass those). That lasts about prune / (2 · lr) = 12 steps at lr 0.5, after which the active sets
are the 114 "sub" combinations and the 1 select, and a step costs 0.04 s. The cell's own lessons cost about 0.01 s a
step. A whole run takes about 5 minutes, nine tenths of them in those first 12 steps. In other words the tournament
repeats blocksearch.py's search (45 s on a laptop, once) about 12 times while the weights separate, and then almost
nothing. A lower learning rate on the logits makes it slower and dearer, since the full evaluation lasts until the
weights separate: a median of 12 minutes a run at lr 0.2, and 50 minutes at lr 0.05 (2,983 s). A wider window makes
each full step dearer and leaves their number about the same: 84 s a step at 52 options (7.3 million "sub"
combinations, then 23.6 million pairs) and about 170 s at 61 (13.8 million and 44 million), so 22-50 minutes a run at
52 options. The pairs' lesson waits while the active pairs exceed a cap (`pair_budget`); a first set of 52-option
runs with the cap at 20 million never ran it and never learned the select (`runs/wide_w52_budget20M.jsonl`).

**H1.** <!-- COST_H1 -->The official run trained for 2,114 s on the H100, against 1,048 s for the certified MLP file
(`bridge/h1/records/wide_mlp_official.json`), and evaluated in 90 s against 98 s. The schedule is about the same
(frozen at step 705 against 743); the extra time sits in steps 500-700, 11 s a step against 1.8 s. There the I/O
lessons and T's lesson run whole divisions, 72 links each, for every combination of their blocks, and each link now
selects among many more options (operands at offsets, 12 flags, written values at offsets); on the GPU those steps
are bound by the number of small kernels a link launches. The chain's tournament itself is cheap: its first steps
evaluate 5-9 million pairs, and steps 400-500, which include its start at step 471, took 145 s in all. On the CPU seeds
(8 cores) training took <!-- CPU_TIMES -->1,476-2,977 s and evaluation 457-658 s (the certified MLP file's CPU seeds:
1,574-3,015 s and 192-593 s)<!-- /CPU_TIMES -->. Every run of the file trained all 20,000 steps within the 3,600 s
budget and evaluated within the 1,800 s one, so no relaxed-budget run was needed. (The first version's 5 failed seeds
used the whole hour: their schedule never froze.)<!-- /COST_H1 -->
<!-- /COST -->

## 5. H1: the chain at free attention's width

<!-- H1TEXT -->
[`submission_tournament.py`](submission_tournament.py) is the certified
[`bridge/h1/submission_wide_mlp.py`](../bridge/h1/submission_wide_mlp.py) (md5 `22bd34d0`, untouched): the wide-menu
machine with both digit tables as ReLU MLPs. Three things change: the chain's menu, its teaching as tournaments, and
two new lessons. Everything else (the reader and writer, the MLP cells and their lessons, the multiply and its layout
menu, the countdown, the schedule) is that file's, line for line.

**1. The chain at free attention's width** (`DIV_LEVEL = 5`). The H1 machine is a register machine, not a grid, so
"free attention's width" is taken as its analogue: every connection of a division link may read any of the link's
sources at any relative place, as a relative-position attention head would.

| connection (the tiny port it plays) | level 4 (certified) | level 5 |
|---|---|---|
| `a`, `b`: the operands (SUB.A, SUB.B) | one of 7 registers (D, Rp, M, Mq, 0, R, d) | any of the 7, moved up or down by 0..3 places: 43 options each |
| `cin`: the borrow route (SUB.C) | from the place below (hand-written in the subtractor) | from 1, 2 or 3 places below, or none: 4 options |
| `flag`: what the keep-or-discard gate reads (SEL.C) | the final borrow, 2 zero tests, A < N | + the borrow out of any of places 0..7: 12 options |
| `cv`, `hv`: what the link writes on commit and on hold (SEL.B, SEL.A) | the result E / the chain register D | any of E, D, A, B, Rp, M, 0, moved by 0..3 places: 43 options each |
| `op`, `keep`, `nread`, the gate table | as level 4 | as level 4 |

The chain's wirings, with the gate table's 4 settings, grow from 9,408 to 7.9·10^9; the division's from 4.1·10^9 to
3.4·10^15; the whole machine's from about 4·10^17 to about 3·10^23 (with the multiply's 3.3·10^7 layouts and the
reader's 3 shapes). Every level-4 option is among them, at offset 0.

**2. The division's wiring as tournaments.** The switches are grouped into four blocks, each with one logit per whole
combination (`self.blk`, Adam at `LR_WIRE` = 0.5):
- **"link"** (a, b, op, cin, nread: 29,584 combinations). Its own lesson is the **link law**, blocksearch.py's stage
  1 in H1 form: one link from the register r = x and r = N (the batch's own numbers, x < N) must borrow, or carry,
  exactly when r < N, and must take N to 0.
- **"gate"** (flag, keep, cv, hv: 66,564 combinations). It has no lesson of its own. The **pairs** (the link
  combinations that pass the link law, at most the 64 best, × the active gate combinations × the gate table's
  settings that pass the loop law, item 3) are scored by the certified file's **division ladder**: from the register
  x + q·N, q = 0..9, one step's chain of 9 links must give back x.
- **"step"** (bring, post: 1,700) and **"read"** (vread, start, order, final, read: 256) keep the certified file's
  lessons (the step law; the whole-division laws and the answers) and its per-row mixture over every combination,
  now with one logit per whole combination.

As in the tiny tournament, a combination passes the link law or the ladder when it passes on every register of the
step's batch, and only combinations within `PRUNE` = 12 nats of their block's best are run. The verdicts are computed
in `chain_lesson` under no_grad; the loss is `tourney_nll`, −log Σ weight·[passes]. On the prediction path each switch
is its block's argmax combination as a straight-through one-hot over every combination, seen through the switch's
marginal (`ste_switches`).

**3. Two gaps in identification, and the lessons that close them.**

*The top place.* The first local trial of the chain's tournament (`local/chain_check.py`: every table at the truth,
the chain lesson alone, 25 steps on real batches) converged in 13 steps on a chain that is right on every training
register and wrong on the unseen moduli: repeated subtraction, but with the gate reading **the borrow out of place 7**
instead of the final borrow (place 8). The reason is in the data. Every training modulus is below 2^20, so every
register of the division stays below 10N < 1.05·10^7, and place 8 (often place 7 too) is always 0 in training. The
two borrows then always agree. On the unseen ladder the moduli reach 8 digits, the registers reach place 8, and they
differ. The same holds for operands moved up one place and moved back down when written. The level-4 menu had no such
options, so it never met the problem.

The fix, **`CHAIN_NORM`**: the chain's lessons (the link law, the ladder, and through its registers the step law) also
run on each training row's x and N moved up until N's top digit is at place 7, so that the ladder's registers fill all
9 places. It is a new lesson made of the data's own numbers, moved by the machine's own move, and a new gray item
under rule 14 (§6).

*The gate's meaning.* The gate table (flag → hold or commit) is learned, and the same table decides whether the
countdown runs a round: a round runs when the gate commits on the borrow of count − 1. In the certified file a link
wrote E on commit and kept D on hold, so the chain's ladder fixed what "commit" means and the countdown inherited it.
With the commit and hold values free, "commit on no borrow, write E on commit, keep D on hold" and "hold on no borrow,
keep D on commit, write E on hold" are the same chain, and the ladder passes both. The countdown only works with the
first. A first version of the file without a lesson for this (md5 `312d9422`) was run on 8 CPU seeds (below): 3
certified, and the other 5 chose the flipped form. Their rounds never ran, nothing came out right, the schedule never
froze, and they used the whole hour. Both H100 runs of that version chose it too and were stopped (their logs are in
`h1_records/partial/`).

The fix, **`LOOP_LAW`**: the chain's pairs only take the gate table settings that pass the loop's law, on the batch's
own counts with the snapped table: every row's count is at least 1, so its first round must run (the gate commits on
the borrow of C − 1), and a count of 0 must stop the loop (the gate holds on the borrow of 0 − 1). One of the 4 gate
settings passes. A third new gray item: it states what the loop's stop rule is (§6).

*Counted.* `h1_census.py` (record `records/h1_census.json`) runs the chain's lessons on many rows at once with every
table at the truth: the link law over all 29,584 link combinations on 256 training rows, then every pair with every
gate combination and gate table on those rows' ladder registers. The survivors' steps are then run on the ladder
registers of all 440 8-digit moduli of the depth_ood_n splits. That check covers the chain step alone, so the flipped
forms count as right there; only the loop law tells them apart.
<!-- /H1TEXT -->

<!-- CENSUS -->
| the chain's lessons on 256 training rows (the link law passes 16 of 29,584 link combinations) | chain wirings passing every lesson (of 7.9e9) | their step right on all 440 8-digit moduli | wrong there | the gate as the loop needs it (commit on no borrow) |
|---|---|---|---|---|
| the plain lessons | 88 | 10 | 78 | half |
| + the loop law | 44 | 5 | 39 | all |
| + the moved-up rows | 10 | 10 | 0 | half |
| + both (the file) | 5 | 5 | 0 | all |
<!-- /CENSUS -->

**The runs.** The official one is the untouched manifest `benchmark/manifests/h100_hard_h1.json` on a Modal H100,
the file sent byte for byte (`modal_h1.py`, a copy of the bridge's launcher); the CPU seeds are the same evaluator with
the manifest's device set to cpu and the seed changed (`h1_plans/`). Each record (`h1_records/<name>.json`) holds the
evaluator's RESULT_JSON, the full log and the exact source.

<!-- H1 -->
**The file** (md5 `7ae547d9`; a `nonorm` record is the same source with `CHAIN_NORM = False`):

| record | device | seed | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen at step) | evaluation s | both MLP cells exact | the chain it chose | md5 |
|---|---|---|---|---|---|---|---|---|---|---|
| tourney2_official | H100 | 74 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2114 (705) | 90 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed1 | cpu | 1 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1921 (721) | 508 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed2 | cpu | 2 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1476 (663) | 525 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed3 | cpu | 3 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1847 (681) | 457 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed4 | cpu | 4 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2263 (736) | 646 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed5 | cpu | 5 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2024 (740) | 535 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed6 | cpu | 6 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2128 (750) | 565 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed7 | cpu | 7 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2312 (694) | 554 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_cpu_seed74 | cpu | 74 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2977 (704) | 658 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 7ae547d9 |
| tourney2_nonorm_cpu_seed1 | cpu | 1 | 64 / None | not all (worst unseen rung 658/768) | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1622 (728) | 559 | yes | trial multiples, the gate reading the borrow out of place 7; flag b7, a (Rp,0), b (Mq,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed2 | cpu | 2 | 4 / None | not all (worst unseen rung 384/768) | 9,999/9,999, 9,995/10,002, 10,002/10,002 | 1619 (662) | 726 | yes | repeated subtraction, the gate reading the borrow out of place 6; flag b6, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed3 | cpu | 3 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2196 (681) | 589 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed4 | cpu | 4 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2069 (770) | 607 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed5 | cpu | 5 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1883 (743) | 558 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed6 | cpu | 6 | 4 / None | not all (worst unseen rung 384/768) | 9,999/9,999, 9,995/10,002, 10,002/10,002 | 1915 (788) | 597 | yes | repeated subtraction, the gate reading the borrow out of place 6; flag b6, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed7 | cpu | 7 | 64 / None | not all (worst unseen rung 658/768) | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2185 (699) | 592 | yes | repeated subtraction, the gate reading the borrow out of place 7; flag b7, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |
| tourney2_nonorm_cpu_seed74 | cpu | 74 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 1497 (703) | 499 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | bbb2f140 |

**The first version, without the loop law** (md5 `312d9422`):

| record | device | seed | max T, seen / unseen moduli | rungs | test, ood_t, ood_n_t | training s (frozen at step) | evaluation s | both MLP cells exact | the chain it chose | md5 |
|---|---|---|---|---|---|---|---|---|---|---|
| tourney_cpu_seed1 | cpu | 1 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2165 (724) | 495 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 312d9422 |
| tourney_cpu_seed2 | cpu | 2 | None / None | not all (worst unseen rung 0/768) | 7/9,999, 20/10,002, 5/10,002 | 3603 (·) | 9 | yes | repeated subtraction, the gate flipped; flag fin, a (D,0), b (M,0), commit (D,0), hold (E,0), gate HC | 312d9422 |
| tourney_cpu_seed3 | cpu | 3 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2007 (681) | 442 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 312d9422 |
| tourney_cpu_seed4 | cpu | 4 | None / None | not all (worst unseen rung 0/768) | 7/9,999, 20/10,002, 5/10,002 | 3601 (·) | 7 | yes | repeated subtraction, the gate flipped; flag fin, a (D,0), b (M,0), commit (D,0), hold (E,0), gate HC | 312d9422 |
| tourney_cpu_seed5 | cpu | 5 | 64 / 64 | 768/768 every rung | 9,999/9,999, 10,002/10,002, 10,002/10,002 | 2265 (745) | 559 | yes | repeated subtraction; flag fin, a (D,0), b (M,0), commit (E,0), hold (D,0), gate CH | 312d9422 |
| tourney_cpu_seed6 | cpu | 6 | None / None | not all (worst unseen rung 0/768) | 7/9,999, 20/10,002, 5/10,002 | 3600 (·) | 7 | yes | repeated subtraction, the gate flipped; flag fin, a (D,0), b (M,0), commit (D,0), hold (E,0), gate HC | 312d9422 |
| tourney_cpu_seed7 | cpu | 7 | None / None | not all (worst unseen rung 0/768) | 7/9,999, 20/10,002, 5/10,002 | 3603 (·) | 8 | yes | repeated subtraction, the gate flipped; flag fin, a (D,0), b (M,0), commit (D,0), hold (E,0), gate HC | 312d9422 |
| tourney_cpu_seed74 | cpu | 74 | None / None | not all (worst unseen rung 0/768) | 7/9,999, 20/10,002, 5/10,002 | 3602 (·) | 7 | yes | repeated subtraction, the gate flipped; flag fin, a (D,0), b (M,0), commit (D,0), hold (E,0), gate HC | 312d9422 |
<!-- /H1 -->

<!-- H1_READING -->
- **The file certified on the official run and on all 8 CPU seeds.** In every run the chain's tournament joined in
  phase C (steps 431-518), settled within a few steps, and the I/O lessons joined 24-27 steps later (20 of them the
  certified file's waiting window). The freezes came at steps 663-750; the certified MLP file's at 681-876.
- **Every run chose long division by repeated subtraction** in its plain form: a = D, b = M, the final borrow, commit
  E, hold D, the gate committing on no borrow. The census's other right forms (the hold value read as A, which is D
  there; the flag "A < N"; trial multiples) did not win in these 9 runs. The I/O took harmless aliases as in the
  certified file: the incoming digit added rather than written into the empty place (4 runs) and a final correction
  that never fires (5 runs). The multiply's products were grouped four ways: rows by multiplier digit (4 runs), the
  same rows summed in reverse (1), rows by multiplicand digit (3), diagonals (1).
- **Both MLP cells came out exact in every run**, and the audits found no subtraction row, product cell or gate row
  used at evaluation that training never used.
- **Without the moved-up rows** (the `nonorm` records: the file with `CHAIN_NORM = False`, 8 CPU seeds), 4 runs
  certified and 4 chose a top-place alias. Seeds 1 and 7 chose the borrow out of place 7 (seed 1 within trial
  multiples): every rung on the seen moduli exact, and on the unseen ones 658-708 of 768 right, the misses being
  8-digit moduli. No register that training can produce tells that gate from the right one. Seeds 2 and 6 chose the
  borrow out of place 6, which differs from the final borrow only on registers of 10^7 or more, too rare among the
  chain's 16 rows a step to be caught: one seen prompt wrong from T = 8, and 384-387 of 768 right on every unseen
  rung. The census counted 39 such wirings among the 44 that pass every lesson with the loop law alone.
- **The first version** (no loop law) certified on 3 of 8 CPU seeds. The 5 others, and both of its H100 runs, chose
  the flipped gate; the chain was right and the countdown never ran a round.
<!-- /H1_READING -->

## 6. Rules 7, 8 and 14

<!-- RULES -->
The rules are [one-layer-deeper/README.md §Rules](../../one-layer-deeper/README.md#rules). This covers
[`submission_tournament.py`](submission_tournament.py) against the certified file it changes
([`bridge/h1/submission_wide_mlp.py`](../bridge/h1/submission_wide_mlp.py), whose rules 1–16 follow
[OVERVIEW.md](../OVERVIEW.md) §6). "Gray" means a reasonable reader could object; the point is flagged, not argued
away. The tiny tournament is an experiment, not a submission, but it is built the same way (one loss per step, one
optimizer step, verdicts inside the loss).

| rule | before | now | still gray |
|---|---|---|---|
| **7. no hard-coded algorithm in the forward pass** | gray: the skeleton is designed and the menus are written by hand and contain the true wiring, though they hold two long divisions, a Montgomery reduction and six groupings | **less gray in the chain**: each link's operands, its borrow route, what the gate reads and what the link writes are chosen from relative positions (any register at any offset), not from a menu of meaningful parts; the forward pass at evaluation is the machine the tournament chose | the skeleton (one register, 8 steps of 9 links, bring-in → chain → move; the multiply; the T loop), the I/O menus and the multiply's layout menu are the certified file's; the offsets' window (−3..3) and the block structure are designed |
| **8. final logits inside autograd, unbroken gradient path** | yes, with gray edges (COMBINED.md, MENUS.md §4.5) | **the same, extended to the blocks**: on the prediction path each switch is its block's argmax combination as a straight-through one-hot over every combination; with every table and switch at the truth, the final logits' gradient reaches every parameter tensor, every one of the 98,104 combinations' logits included (`local_wide.py grad`) | the certified file's edges (training returns zero logits; the damped gradient; aliases whose gradients can cancel). On trained models (`local_wide.py train` on Modal, `local/modal_diag.py`; logs in `local/logs/`): the file, seeds 74 and 2 (frozen at steps 697 and 660), and the first version, seed 1 (728). On each, both prediction paths give identical, exact logits on 128 prompts of every one of the 14 rungs, and every parameter tensor gets a gradient on both test batches (T = 1, 2 and T = 8, unseen-moduli T = 4). A few combinations get exactly 0: up to 12 of the gate block's 66,564, 16-32 of the read block's 256, up to 2 of the step block's 1,700; every link combination gets one |
| **14. no data inspection, augmentation, task-specific solvers, custom loops, participant backward** | gray: the lessons carry designed knowledge (the division ladder, the step law, x mod N = x, ...); the switches' verdicts come from running the hard machine on candidate wirings inside the loss | **gray in the same way, with four new items**: (1) the link law, one link from x and from N must borrow exactly when the register is below N and take N to 0, is a statement of what one link does (blocksearch.py's stage 1); (2) the moved-up rows (`CHAIN_NORM`) are new numbers made from each row's own x and N, moved up by the count of N's leading zero digits so that N fills 8 places: a hand-written transformation of the data inside a lesson, close to augmentation; (3) the loop law (`LOOP_LAW`), every count is at least 1 so its first round runs and a count of 0 stops the loop, states the loop's stop rule (the loop itself was already hand-written); (4) the tournament's active set, which leaves out of the loss the combinations more than 12 nats behind (they count as failing) | no custom loop: one forward, one loss, one backward by the evaluator per step; no backward or grad call in the file; the pruning is a function of the current logits, recomputed each step |

The tiny tournament's prediction path passes its gradient test in every run with a learned cell (§2): the answers
equal the hard network's, and the cross-entropy's gradient reaches both blocks' logits and every MLP weight.
<!-- /RULES -->

## 7. Caveats

<!-- CAVEATS -->
- **The tiny menu is blocksearch.py's.** Its 34 options are relative offsets plus the bridge's 7 rules, tied per
  kind; free attention can also prefer key kinds, which no menu here enumerates (BRIDGE.md §10).
- **The blocks are designed.** Which ports are scored together, and that block "sub" has a lesson of its own, comes
  from blocksearch.py's analysis of which ports interact. Without block "sub"'s own law, the pairs would be
  1.3·10^6 × 39,304 = 5·10^10 per step, and the pairs' lesson never runs (`sub_law`, `pair_budget`).
- **The tournament is the search, repeated.** Each step re-runs every active combination; until the weights separate
  (about prune / (2 · lr) = 12 steps at lr 0.5) that is a full block-wise search per step. Pruning is what makes it
  cheap afterwards (§4).
- **A combination that is eliminated cannot come back.** Its weight is below e^-12 of the leader's and it is no longer
  run. That is safe here because the wiring's lessons wait for the table (the gate), and because with a wrong table the
  true wiring fails together with everything else (the no-gate runs: nothing passes, nothing moves). A lesson that
  could pass impostors while rejecting the true wiring would make elimination permanent.
- **The answers do the identification in the tiny machine.** The label-free laws leave 4,391 wirings, almost all of
  them bypasses (§3); the tournament then picks one at random. So "no answers" is not an option at this size.
- **The H1 fixes are lessons, and they are gray.** `CHAIN_NORM` exists because the data never fills the register's
  top place, `LOOP_LAW` because the free commit and hold values let the gate's meaning flip. The level-4 menu needed
  neither: it had no options that read the top place differently, and it fixed what a link writes on commit. Both
  gaps were found by running the file, not foreseen; the census then counted them. A wider menu may hide others that
  these runs did not meet.
- **H1 runs are few.** One official H100 run and 8 CPU seeds of the file, 8 CPU seeds of the variant without the
  moved-up rows, the first version's 8 CPU seeds (its two H100 runs were stopped once they had chosen the flipped
  gate), and three trained-model diagnostics; the tiny sweeps have 8 seeds per variant (4 for the widest menu,
  <!-- ROWN -->8<!-- /ROWN --> for per-row verdicts).
- **Not run:** the H1 file at a wider offset window; the multiply's layout as a tournament block (it keeps the
  certified file's soft routed ladder).
<!-- /CAVEATS -->

## 8. Reproduce

<!-- REPRO -->
From `cpu_first/`, with the sotaku venv's Python. Local runs take seconds to minutes; the sweeps and every H1 run go
to Modal.

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python -m tournament.tiny 0
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tournament/modal_tiny.py --suite main
```

```bash
TOURNAMENT_CONTAINERS=10 ~/github/sotaku/venv/bin/python -m modal run --detach tournament/modal_tiny.py --suite slow,wide
```

```bash
~/github/sotaku/venv/bin/python -m modal run tournament/modal_tiny.py --suite main --fetch
```

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python -m tournament.traps
```

```bash
~/github/sotaku/venv/bin/python -m modal run tournament/h1_census.py --rows 256
```

The H1 file's local checks (the designed plumbing on both prediction paths, the gradient test, the chain's tournament
alone, and the training step from phase C on):

```bash
~/github/sotaku/venv/bin/python wider_menus/local/local_wide.py $PWD/tournament/submission_tournament.py designed N=8
```

```bash
~/github/sotaku/venv/bin/python wider_menus/local/local_wide.py $PWD/tournament/submission_tournament.py grad N=16
```

```bash
~/github/sotaku/venv/bin/python tournament/local/chain_check.py STEPS=25
```

```bash
~/github/sotaku/venv/bin/python tournament/local/phase_c_check.py STEPS=60
```

The same checks on several seeds at once, on Modal (8 CPU containers):

```bash
~/github/sotaku/venv/bin/python -m modal run tournament/local/modal_checks.py --script phase_c_check --args "STEPS=80"
```

The official run, the CPU seeds, and the variant without the moved-up rows, from `../one-layer-deeper` (app and
Volume `tournament-h1`; records in `tournament/h1_records/`). `h1_plans/tourney2_cpu.json` holds the 8 seeds and the
variant's seed 74, `h1_plans/tourney2_nonorm_cpu.json` the variant's other 7 seeds; `h1_plans/tourney_cpu.json` is the
first version's plan, run from that version's source (md5 `312d9422`, embedded in its records):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/tournament/modal_h1.py --file submission_tournament.py --mode official --name tourney2_official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/tournament/modal_h1.py --file submission_tournament.py --mode plan --plan ../cpu_first/tournament/h1_plans/tourney2_cpu.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/tournament/modal_h1.py --file submission_tournament.py --mode plan --plan ../cpu_first/tournament/h1_plans/tourney2_nonorm_cpu.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/tournament/modal_h1.py --mode fetch
```

The trained-model checks (training through the local harness on a Modal CPU, then both prediction paths on every rung
and the gradient test; logs in `local/logs/`):

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/tournament/local/modal_diag.py --plan ../cpu_first/tournament/h1_plans/diag2_train.json
```

Then `PYTHONPATH=. ~/github/sotaku/venv/bin/python -c "from tournament import fill; fill.main()"` regenerates the
tables.

Files:
- `tiny.py`: the tiny tournament (the model, its lessons, the prediction path, the measures);
- `modal_tiny.py`: the tiny sweeps; `runs/`: their records;
- `traps.py`: what the tiny checks let through (`records/traps.json`);
- `submission_tournament.py`: the H1 file; `modal_h1.py`, `h1_plans/`, `h1_records/`: its runs;
- `h1_census.py`: which chain wirings pass every H1 training lesson, and which are right on 8-digit moduli
  (`records/h1_census.json`);
- `local/chain_check.py`, `local/phase_c_check.py`: the H1 file's tournament in isolation; `local/modal_checks.py`
  runs them on several seeds; `local/modal_diag.py` and `local/logs/`: the trained-model checks;
  `local/wait_fetch.py`: waits for H1 records on the Volume;
- `h1_records/partial/`: the live logs of the first version's two stopped H100 runs, and the trained-model logs;
- `runs/wide_w52_budget20M.jsonl`: the 52-option runs whose pairs exceeded the cap;
- `fill.py`: this file's generated tables.
<!-- /REPRO -->
