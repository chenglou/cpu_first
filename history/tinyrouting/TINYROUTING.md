# Can a digit machine learn its own wiring? A tiny machine, computed exactly

## The answer in plain words

**The question.** Every arithmetic table of the H1 machine is learned now
([LEARNED_REMAINDER.md](../../learned_remainder/LEARNED_REMAINDER.md), [CHAIN.md](../../learned_remainder/CHAIN.md),
[SQUARE.md](../../learned_remainder/SQUARE.md), [CONTROL.md](../../learned_remainder/CONTROL.md)). Its wiring is still
written by hand: which cell feeds which, and in what order. Under a strict reading of the rules, that wiring is the
algorithm. Can a machine learn its wiring too, together with its tables, from the answers plus checks that need no
answers? And how should the wiring be described so that it can be learned?

**Short answer: yes, in tiny machines, under three conditions.**
1. Every table is taught by a check that ignores the wiring.
2. The wiring is described as blends that keep every option in play: soft switches, or a mixture of whole wirings,
   with switches that depend on each other weighed together.
3. Every switch gets an unscrambled lesson: one shared switch per kind of connection, a law that reads it directly,
   or one division step at a time.

Under these conditions a tiny version of H1's whole step (multiply, then reduce; about 2^42 wirings plus two tables)
learned its wiring and tables together, graded only on the remainder. From the answers alone it does not work: the
untrained table gets bypassed. Committing early to one wiring (straight-through, sampling, one-change search) does not
work either. For H1 this predicts tied soft switches plus the tables' own laws, a routed ladder for the squaring, the
division added one step at a time, and a small mixture for the reading and readout. That costs about 2 machines per
training step.

**The tiny machine.** A long-division machine that is H1's remainder (CHAIN.md) in miniature:
- base 3, two-digit moduli N from 3 to 8, dividends below 3N: 99 questions in all;
- one learned table, digit subtraction with a borrow (18 rows).

Seven of its connections are left open. Each is a switch with 2 to 4 positions:
1. read the dividend's digits as written, or reversed;
2. the same for N;
3. when the next dividend digit comes in, the register moves up (right), moves down, stays, or ignores the digit;
4. the subtractor's first input is the previous trial (right), the step's starting number, or N;
5. its second input is N (right), the previous trial, or the starting number;
6. the result kept is the last trial before a subtraction first goes negative (right), one trial later, always the
   last trial, or the unchanged number;
7. the answer is read from the low places (right) or from the high ones.

That makes 1,152 wirings. One computes the remainder. One more, its "twin", would too if the table were mirrored:
swap the subtractor's two inputs and teach it b − a instead of a − b.

**Can the wiring be learned with the table? Yes, if three things hold.**
1. **The table has a teacher that does not depend on the wiring.** The check (a − b) + b = a, with a designed adder,
   teaches the table to subtract whatever the wiring does. H1's chain uses the same check.
   - With it, all seven switches come out right at every size tested (bases 3–5, up to 3-digit moduli and 2 quotient
     digits), in 37–175 steps, and the machine is then exact on bigger machines it never saw.
   - From the answers alone, almost nothing certifies (§2).
2. **The wiring is described so that every option keeps being tried, and options that depend on each other are tried
   together.**
   - *A mixture of whole machines*: run all 1,152 wirings side by side and blend their answers (SpecBit's trick). It
     found the wiring in every run at every size (§2).
   - *Soft switches*: each switch passes on a blend of its options' digits. They are far cheaper and nearly as reliable
     (27 of 28 runs), but weaker as the chain gets deeper (§2.2, §3.7).
   - Switches that depend on each other must be weighed together: how the dividend is read, how the register shifts,
     where the answer is read. Give each its own weight (a factorized mixture, or a hybrid that splits them) and they
     can lock into a wrong combination that hangs together (§2.1).

   What fails is committing early: running only the current best wiring and correcting it one switch at a time
   (straight-through gradients, Gumbel sampling, REINFORCE, "try one change and keep it if better"). With the right
   table, 180 of the 1,152 wirings are dead ends. No single switch change improves them (§3.4).
3. **Every switch gets a lesson that is not scrambled on the way.** Three ways work:
   - *one shared switch* for a connection that repeats (every step, every link), as one table serves every place;
   - *a law that reads the switch directly* and needs no answer;
   - *adding one division step at a time*, so that the newest step is always the one next to the answer.

   If instead every step and link has its own switch (2^20–2^34 wirings), and none of those three applies, soft
   switches fail even with the right table: the early copies learn only through later copies that are still
   undecided (§4).

**The whole H1 step, in miniature** (§5.1). Multiply x by y digit by digit, then reduce mod N, with all of these
learned together, graded only on the remainder:
- where each product digit goes, one switch per digit;
- the division's seven switches;
- the times table and the subtraction table.

That is about 2^42 wirings. It is learned when every table has its law and the placement has the routed ladder
(below):
- with soft switches and per-digit placement, 10 of 10 runs, in 100–125 steps;
- with a tied placement, 8 of 8 once the division's reading, shift and readout are enumerated together (soft alone:
  5 of 10).

Without the routed ladder 0 of 20 runs succeed. From the answers alone, none ends with the reference wiring; one is
exact as a twin.

**A small example**, the soft switches with the cell check (`phase` row `321_soft_cell_s0`):
- at step 0 every switch is undecided and the table is noise; 31% of answers come out right by luck;
- by step 25 the table is 89% right, and most switches lean the right way. The subtractor's first input does not: it
  leans to "the starting number", at 13% on the right option. Subtracting from a clean number beats chaining a noisy
  table;
- by step 50 the table is right. That switch has come back to 63% on the right option, all seven argmax positions are
  right, and every answer is exact;
- the same wiring and table are exact on every question with two quotient digits and with three-digit moduli.

**Why answers alone fail: the untrained table gets bypassed.**
- At first the table outputs noise. The wirings that score best go around it: keep the number unchanged, or subtract
  from the starting number instead of chaining.
- The switches move there within 25 steps. The part of the table that those wirings skip then gets no lesson, and the
  machine settles as a shallow one that subtracts at most once (66% right).
- With the cell check the same dip happens, but the table is right by step 50 and the switches come back.
- The mixtures of whole machines can recover, because they judge each wiring as a whole:
  - the plain mixture, which keeps each wiring's own weight alive, fits every training answer in 2 of 4 runs;
  - the factorized one does so in 4 of 4 runs, all with the true wiring, once the table's two inputs get fixed roles
    (x − 0 = x) and the remainder laws are added (x mod N = x, N mod N = 0).

  None of them is exact beyond training, but the wiring is not to blame. Table rows that the training questions never
  use stay wrong, exactly as when the wiring is given.

**The same in schoolbook multiplication** (§5). A tiny squaring circuit whose wiring is which column each product
digit goes to:
- graded on the product's own digits, every description finds the wiring in 25–50 steps, even with a separate switch
  for every product digit;
- graded only on the remainder after dividing by N (H1's situation), one shared placement rule is always found. A
  separate switch per digit works at some sizes and stalls at others (97% right at the smallest): digits that are
  rarely nonzero end up two columns off, and the remainder hides it;
- a check that needs no answers removes that dependence: row(x, b+1) = row(x, b) + x, computed through the learned
  placement (SQUARE.md's ladder). With it, every placement at every size comes out exact, even while the times table
  itself is learned, with no labels at all.

**The tricks** (§6).
- **What changes the picture** is structural:
  - a table teacher that ignores the wiring;
  - one switch per kind of connection;
  - one division step at a time: for switches that are not shared, and for shared ones once the chain is deep (binary
    from 2 quotient digits, base 3 from 4). Then 72 of 72 runs succeed at every depth tried, in 50–100 steps;
  - a local law that reads the wiring (the routed ladder);
  - enumerating switches that depend on each other together.
- **What only changes the waiting:** one step at a time for shared switches in shallow chains (it shortens the flat
  stretch), try-and-keep on top of soft switches (slightly shorter), and freezing the switches while the table learns
  first (longer).
- **What hurts:** anything that commits early: the machine's own hard path, large random starting preferences,
  straight-through. Checks that a do-nothing machine passes ("x mod N = x", "the register stays below N") also hurt:
  as teachers they pull toward the bypass, though they are good filters once the table is right.
- **From the answers alone, almost nothing rescues soft switches or straight-through**: not a fixed meaning for the
  table's inputs, not symmetry breaking at the start, not training the table first, not try-and-keep. The trap there
  is the bypass, not a symmetry. Where the bypass is weak, one step at a time helps. In binary, where chaining and
  "subtract from the start" coincide, soft switches certify from the answers alone. The factorized mixture of whole
  machines is also different: for it a fixed meaning plus the remainder laws is enough to find the wiring (above).

**Prediction for H1** (§7).
- **The parametrization that should scale** combines five things:
  - tied soft switches, one per kind of connection: about 15 for H1, 2^24 wirings;
  - every table taught by a check that ignores the wiring;
  - the squaring's placement taught by the routed ladder;
  - the few switches that only the answers can decide and that depend on each other (reading, readout, start and step
    order) enumerated together as a small mixture;
  - division steps added one at a time.
- **Its cost** is about 1.5–3 single machines per step, whatever the size of the wiring space. Its limit is steps:
  a few hundred, minutes on the H100.
- **A mixture over whole wirings** fits about 2^10–2^12 wirings in the hour as H1's code runs today, perhaps 2^15
  with every wiring in one fused batch. It is the most robust description in the tiny cores (every seed, every size,
  table given or taught). Keep it for the few switches only the answers can decide that depend on each other:
  reading order, start register, step order, output. IO.md already does this with 24 readings.
- **Straight-through, sampling and one-change search** should not be expected to find H1's wiring.

## 1. The tiny cores

### 1.1 The division core
`machine.py`. A size is (B, w, k): base B, moduli N with w base-B digits (top digit nonzero), dividends v < N·B^k
(k quotient digits); every (N, v) of a size is used, so every gradient is exact (float64). The true machine is
CHAIN.md's reduction:
- the register R (w+1 places) starts at v's top w digits, which are below N;
- each step shifts R up one place and brings the next dividend digit in at place 0: Rp = B·R + v_t;
- a chain of B−1 links: trial 0 = Rp, trial s = trial (s−1) − N, each link a ripple of the learned table
  (a, b, borrow) → (digit, borrow) from place 0 up;
- the write-back keeps the last trial before the first borrowing link;
- the answer is R's low w places.

The switches (`MUXES`; option 0 is the true one):

| switch | options | what a wrong option does with the true table, at (3, 2, 1) |
|---|---|---|
| vread | given, reversed | reversed: 0.51 of answers right |
| nread | given, reversed | 0.49 |
| shift | up, down, stay, skip | down 0.28, stay 0.33, skip 0.21 |
| a (minuend) | prev, start, N | start 0.67 (subtracts at most once), N 0.18 |
| b (subtrahend) | N, prev, start | prev 0.18, start 0.18 |
| keep | stop, over, all, none | 0.33 each |
| read | low, high | 0.18 |

One switch serves every step and link (one switch replicated, like the table); §4 unties them. The table is applied
to digit distributions multilinearly, as in [TINYCORE.md](../tinycore/TINYCORE.md).

**Parametrizations of the wiring** (`train.py`):

| name | what is learned | forward pass | cost per step at (3, 2, 2), batch 64 |
|---|---|---|---|
| mix | one weight per whole wiring (1,152) | every wiring run as a hard machine; the answers' likelihoods blended | 112 ms (linear in wirings: 2.4 ms at 2, 16 ms at 128, 37 ms at 384) |
| fmix | one weight per switch option (20); a wiring's weight is the product | as mix | 115 ms |
| hybrid | a mixture over the 64 wirings of vread, nread, shift, keep; soft a, b, read | as mix over 64 | 15 ms (9 ms with vread, nread, shift, read enumerated) |
| soft | one weight per switch option | each switch blends its options' digit distributions on the wire | 3.1 ms |
| st | as soft | the argmax wiring, hard; gradients of the soft blend at that point (straight-through) | 3.1 ms |
| gumbel | as soft | a Gumbel-perturbed argmax; tempered softmax gradients | as st |
| reinforce | as soft | 16 sampled wirings; score-function gradients with a mean baseline | 5.4 ms |

Costs: laptop, 2 threads.

**Table recipes**:
- *answers*: cross-entropy of the answer's digits;
- *cell*: (a − b) + b = a with a designed adder on every pair of (w+1)-digit numbers, plus the final borrow =
  [a < b]. It never looks at the wiring;
- *cellown*: the same identity on the machine's own link inputs along its argmax wiring, as H1's chain does;
- *laws*: x mod N = x for x < N and N mod N = 0, run through the chain and write-back (reads a, b, keep);
- *range*: after every step the register is below N (a designed comparator);
- *anchor*: x − 0 = x on the table (a fixed meaning for its two inputs; breaks the twin).

**Measures.** Adam, lr 0.05, logits N(0, 0.1²). A run stops 100 steps after the hard machine (argmax wiring,
snapped table) first gets every training answer right ("hit"). It is *certified* when that machine is also exact on
every question of two bigger sizes, (B, w, k+1) and (B, w+1, k+1). Sizes above 300 questions train on batches of 128.

### 1.2 The schoolbook core
`school.py`. x and y have w base-B digits; every cell (i, j) looks up a times table (a, b) → (lo, hi) on (x_i, y_j)
and sends lo and hi each to a column; a designed adder sums the columns into 2w digits. The truth sends lo to column
i + j and hi to i + j + 1 (SQUARE.md's rows). Routing:
- *tied*: one offset per output kind, shared by every cell: column = i + j + d, d ∈ {−1, 0, 1, 2} (16 wirings);
- *rel*: every cell's lo and hi pick their own offset (4^(2w²) wirings: 65,536 at w = 2);
- *abs*: every cell's lo and hi pick any column (4^8 at w = 2).

Labels are the product's digits, or only x·y mod N after the division core with its true wiring and table (N has w
digits, x, y < N). The times table is given, or learned from the label-free ladder a·0 = 0, a·(b+1) = a·b + a.

## 2. Phase map

### 2.1 Wiring and table together
`phase` suite: 7 switches (1,152 wirings), 4 seeds, at most 2,000 steps. Each cell: certified / seeds (median step at
which every training answer is right, over the certified seeds); "f" counts seeds that fit every training answer but
failed a bigger size. Hybrid: a mixture over the 64 wirings of vread, nread, shift and keep, with soft a, b and read.

**With the cell check** (answers + (a − b) + b = a on all pairs):

<!-- PHASE_CELL -->
| | mix | fmix | hybrid | soft | st |
|---|---|---|---|---|---|
| (3, 2, 1) | 4/4 (75) | 4/4 (50) | 4/4 (75) | 4/4 (50) | 0/4 |
| (3, 2, 2) | 4/4 (75) | 0/4 | 0/4 | 4/4 (175) | 0/4 |
| (3, 3, 1) | 4/4 (75) | 4/4 (50) | 4/4 (75) | 4/4 (50) | 0/4 |
| (3, 3, 2) | 4/4 (75) | 4/4 (50) | 4/4 (62) | 4/4 (62) | 0/4 |
| (4, 2, 1) | 4/4 (37) | 4/4 (50) | 4/4 (75) | 4/4 (50) | 0/4 |
| (4, 2, 2) | 4/4 (62) | 3/4 (50) | 4/4 (62) | 3/4 (75) | 0/4 |
| (5, 2, 1) | 4/4 (75) | 4/4 (50) | 4/4 (50) | 4/4 (62) | 1/4 (50) |
<!-- /PHASE_CELL -->

**From the answers alone:**

<!-- PHASE_ANS -->
| | mix | fmix | hybrid | soft | st |
|---|---|---|---|---|---|
| (3, 2, 1) | 0/4 f2 | 0/4 | 0/4 | 0/4 | 0/4 |
| (3, 2, 2) | 0/1 | 0/2 | 0/4 | 0/4 | 0/4 |
| (3, 3, 1) | · | · | 1/4 (225) | 0/4 | 0/4 |
| (3, 3, 2) | · | · | 1/4 (900) | 0/4 | 0/4 |
| (4, 2, 1) | 0/2 | 0/2 | 0/4 | 0/4 | 0/4 |
| (4, 2, 2) | · | · | 0/4 | 0/4 | 0/4 |
| (5, 2, 1) | · | · | 0/4 | 0/4 | 0/4 |
<!-- /PHASE_ANS -->

The whole-wiring mixtures from the answers alone ran only at the three smallest sizes (2 seeds beyond (3, 2, 1),
batch 64, 1,500 steps): they cost a full machine per wiring per step.

- **With the cell check, the whole-wiring mixture finds the wiring at every size** (28 of 28 runs, 37–75 steps).
  Soft switches find it in 27 of 28 (50–175 steps). The one failure, (4, 2, 2) seed 2, ended with a right table and
  a wrong wiring (shift "stay", inputs N − previous trial), 13% right.
- **The factorized mixture and the hybrid fail at (3, 2, 2) in every seed, by the same co-adaptation.**
  - The table is right in every one of those runs.
  - The reading, the shift and the readout lock into a combination that is wrong but hangs together: v read backwards,
    the shift down (or skip, or stay), the answer read from the high places.
  - A factorized weight cannot hold that the right reading goes with the right readout while both are still
    undecided. Each switch learns from the average over the others' current options and follows them. The
    whole-wiring mixture gives every combination its own weight and escapes.
  - With the table given (§2.2) both are 4/4 at (3, 2, 2): the trap forms while the table is still wrong.
- **Straight-through commits early and fails**: 1 of 28 runs certified.
- **Grouping the interacting switches fixes the hybrid** (local runs, 4 seeds, with the cell check):
  - enumerating vread, nread, shift and *read* together, with a, b, keep soft: 4/4 at (3, 2, 2) in 75 steps, faster
    than all-soft (175);
  - enumerating a, b, keep, with the rest soft: 4/4 (150–200);
  - enumerating vread, shift, read: 4/4 (75–125);
  - the suite's grouping (vread, nread, shift, keep enumerated; read soft) splits the readout from the switches it
    depends on, and fails.
  - Switches whose right option depends on each other belong in the same enumerated group. Shifting down partly undoes
    reading the high places.
- **From the answers alone almost nothing certifies.**
  - Soft switches and straight-through: 0 of 56 at every size. The whole-wiring mixture at (3, 2, 1) fits every training
    answer in 2 of 4 seeds, but leaves unused table rows wrong.
  - The exceptions are the hybrid at w = 3 (1/4 at (3, 3, 1), 1/4 at (3, 3, 2)), where 18 moduli instead of 6 give
    the table more coverage. The (3, 3, 2) success is the twin: inputs swapped (N − previous trial) with a mirrored
    table, exact everywhere.
- **The failed runs' wrong options** (all uncertified runs, both recipes) are mostly the bypasses and the I/O:
  minuend = start (54) or N (46), v read reversed (52), keep the unchanged number (49), N read reversed (44), read the
  high places (37).

### 2.2 The wiring alone (table given)
`alone` suite: the true table given, only the 7 switches learned from the answers, 4 seeds. Gumbel and REINFORCE only
at the three smallest sizes.

<!-- ALONE -->
| | mix | fmix | hybrid | soft | st | gumbel | reinforce |
|---|---|---|---|---|---|---|---|
| (2, 3, 3) | 4/4 (37) | 4/4 (50) | 0/4 | 0/4 | 0/4 | · | · |
| (2, 4, 3) | 4/4 (50) | 4/4 (25) | 0/4 | 0/4 | 0/4 | · | · |
| (3, 2, 1) | 4/4 (75) | 4/4 (50) | 4/4 (37) | 4/4 (50) | 0/4 | 0/4 | 1/4 (75) |
| (3, 2, 2) | 4/4 (62) | 4/4 (25) | 4/4 (50) | 4/4 (100) | 0/4 | 0/4 | 0/4 |
| (3, 3, 1) | 4/4 (62) | 4/4 (25) | 4/4 (25) | 4/4 (50) | 0/4 | · | · |
| (3, 3, 2) | 4/4 (50) | 4/4 (25) | 4/4 (50) | 4/4 (62) | 0/4 | · | · |
| (3, 3, 3) | 4/4 (50) | 4/4 (25) | 4/4 (75) | 4/4 (75) | 0/4 | · | · |
| (4, 2, 1) | 4/4 (37) | 4/4 (25) | 4/4 (50) | 4/4 (50) | 0/4 | 0/4 | 2/4 (62) |
| (4, 2, 2) | 4/4 (37) | 4/4 (25) | 4/4 (50) | 4/4 (75) | 0/4 | · | · |
| (5, 2, 1) | 4/4 (50) | 4/4 (25) | 4/4 (25) | 4/4 (50) | 0/4 | · | · |
| (5, 2, 2) | 4/4 (37) | 4/4 (25) | 4/4 (75) | 2/4 (150) | 0/4 | · | · |
<!-- /ALONE -->

- **With the right table, both mixtures choose the wiring at every size**: 4/4 everywhere, in 25–75 steps; the
  factorized one is fastest (25 steps at most sizes).
- **Soft switches stop being reliable as the chain gets deeper**: 4/4 up to (3, 3, 3) and (4, 2, 2), 2/4 at
  (5, 2, 2) (four links per step), 0/4 in binary with three quotient digits. The binary runs all settle on "N − N":
  a machine whose register is always 0, right on the 18% (11% at w = 4) of questions whose remainder is 0. That is
  TINYCORE.md's copying collapse in wiring form. Along the straight path from undecided to right (§3.2), the soft loss
  recovers 7–9% in the first half where soft switches always succeed, but only 4% in those binary sizes and 1.4% at
  (5, 2, 2).
- **Committing early fails even here**: straight-through and Gumbel 0/4 at every size; REINFORCE (16 sampled wirings
  per step) 0–2/4. This is the landscape of §3.4: hundreds of wirings that no single change improves.
- **The hybrid's (3, 2, 2) failure needs a learned table**: with the table given it is 4/4 in 50 steps. The co-
  adaptation of §2.1 forms while the table is still wrong. In binary the hybrid fails like the soft switches, on the
  same "N − N" wiring: its input switches a and b are soft.

### 2.3 The wiring given: which table recipes work
`baseline.py`, 6 seeds, 1,500 steps:

| recipe | (3, 2, 1): fit / certified | (3, 2, 2): fit / certified |
|---|---|---|
| answers | 4/6 (100–125 steps) / 0 | 1/6 / 0 |
| answers, along the own hard path | 0/6 / 0 | 0/6 / 0 |
| answers + cell | 6/6 (50) / 6 | 6/6 (50) / 6 |
| answers + cell on own link inputs | 4/6 (50–75) / 4 | 4/6 / 4 |
| answers + laws | 4/6 / 0 | 3/6 / 0 |
| answers + range | 3/6 / 0 | 0/6 / 0 |
| cell alone, no answers | 6/6 (25) / 6 | 6/6 (25) / 6 |

- The answers alone leave 1–2 table rows wrong that no training question uses (table 0.89–0.94 right when the answers
  fit), so nothing certifies, and half the seeds do not even fit.
- The identity on the machine's own link inputs fails 2 of 6 seeds in a known way: the table ignores its incoming
  borrow (its borrow-in-1 rows copy the borrow-in-0 rows), its intermediate borrows then never fire, and nothing
  teaches those rows. That is ADDER.md's and CHAIN.md's "borrow ignored" trap; LEARNED_REMAINDER.md fixed it with a
  heavier lowest place. On all pairs of numbers the trap does not appear.

## 3. Exact computations

### 3.1 The first lesson the switches get
`exact.py grad0`, `gradtrue`. At the exact uniform start (every logit 0):
- **The subtractor's two inputs and N's reading get exactly zero lesson** (≤ 10⁻¹⁶), for every objective, soft or
  mixture. A table that outputs uniform digits ignores its inputs, so which register feeds it does not matter.
- **The write-back is pushed toward "keep the unchanged number"** by every objective (answers, laws, range). Trial 0
  is the only trial that is not noise.
- **The answers push the other switches** toward the right reading of v and the right answer places, and the shift
  toward "stay" (soft) or "up" (mixture).
- From random starts (logits N(0, 0.1²)) the ports get lessons of 10⁻⁴, in no consistent direction.

With the **true table** and the uniform wiring the inside of the machine is still pushed toward bypasses:

| objective | vread | nread | shift | a | b | keep | read |
|---|---|---|---|---|---|---|---|
| answers, soft, (3, 2, 1) | given ✓ | given ✓ | up ✓ | start ✗ | start ✗ | none ✗ | low ✓ |
| answers, soft, (3, 2, 2) | given ✓ | given ✓ | stay ✗ | start ✗ | start ✗ | none ✗ | low ✓ |
| answers, fmix, (3, 2, 1) | given ✓ | given ✓ | up ✓ | start ✗ | N ✓ | none ✗ | low ✓ |
| laws, soft | – | – | – | N ✗ | prev ✗ | none ✗ | – |
| range, soft | given ✓ | reversed ✗ | stay ✗ | N ✗ | start ✗ | none ✗ | – |

- The laws push "keep the unchanged number" because it satisfies x mod N = x on every row but one per modulus.
- The range check pushes "stay" because a register that never grows stays below N.
- Both checks are satisfied by doing nothing. CONTROL.md hit the same thing: its "x mod N = x" fact taught the commit
  gate to always hold until the borrows were right.
- Untied (§4), every position of every internal switch, early or late, gets the same kind of push. So the outcome is
  decided by what happens after the input and output switches commit, not by the first lesson.

### 3.2 The loss on the straight path from undecided to right
`exact.py eps`. True table; every switch at (1 − ε)·right + ε·uniform; answer loss in nats per digit (chance: ln 2 =
0.69, ln 3 = 1.10, ln 4 = 1.39, ln 5 = 1.61):

| size | ε = 1 | 0.5 | 0.3 | 0.1 | 0.01 |
|---|---|---|---|---|---|
| (3, 2, 1) soft | 1.02 | 0.94 | 0.83 | 0.46 | 0.06 |
| (3, 2, 1) mixture | 1.05 | 0.27 | 0.14 | 0.04 | 0.004 |
| (3, 2, 2) soft | 1.05 | 1.03 | 0.98 | 0.76 | 0.15 |
| (3, 2, 2) mixture | 1.09 | 0.27 | 0.14 | 0.04 | 0.004 |
| (4, 2, 1) soft | 1.31 | 1.21 | 1.09 | 0.67 | 0.09 |
| (3, 3, 1) soft | 1.06 | 0.97 | 0.85 | 0.47 | 0.06 |
| (2, 3, 1) soft | 0.63 | 0.56 | 0.46 | 0.21 | 0.02 |
| (2, 3, 3) soft | 0.67 | 0.64 | 0.60 | 0.46 | 0.09 |
| (2, 3, 3) mixture | 0.73 | 0.18 | 0.10 | 0.03 | 0.003 |
| (5, 2, 2) soft | 1.59 | 1.56 | 1.53 | 1.36 | 0.43 |
| (5, 2, 2) mixture | 1.79 | 0.32 | 0.17 | 0.05 | 0.005 |

- **Unlike exact whole-number blocks, soft switches get partial credit.**
  [FINDINGS_exact_blocks.md](../microlab/FINDINGS_exact_blocks.md) measured a loss flat until ε ≈ 1/N² once a
  remainder follows the choice. Here the blend is of digit probabilities, each bounded by 1, so the loss falls
  monotonically.
- **But the slope near "undecided" is small, and shrinks with depth.** The first half of the path recovers 7–11% of
  the loss with one quotient digit, 2% at (3, 2, 2), 4% at (2, 3, 3) and 1.4% at (5, 2, 2). The last two are where
  soft switches start to fail with the table given (§2.2).
- The mixture recovers 75% at once: the right wiring's share of the blend alone makes the answer likely.

### 3.3 The twin: an exact symmetric saddle
`exact.py saddle`. Only the two input switches are learned, from the answers, with the table. Exchanging the two
switches and mirroring the table changes nothing, an exact symmetry. Plain gradient descent (lr 1) from a twin-symmetric
start (equal switches, mirror-symmetric table):

| size | stays symmetric (twin-breaking part after 1,500 steps) | most negative curvature, soft / mixture (always in the twin-breaking directions) | steps per tenfold growth of a twin-breaking push, soft | where a 10⁻² push exits (3,000 steps) |
|---|---|---|---|---|
| (3, 2, 1) | 2·10⁻¹⁵ | −2.2·10⁻⁴ / −7.6·10⁻⁵ | 10,700 | does not exit; 43% right |
| (3, 2, 2) | 6·10⁻³ (rounding, grown at the fast rate) | −2.4·10⁻² / −1.8·10⁻² | 95 | N − start (31%) |
| (4, 2, 1) | 4·10⁻⁹ | −2.2·10⁻² / −1.4·10⁻² | 103 | N − start (51%) |
| (3, 3, 1) | 5·10⁻¹⁴ | −2.6·10⁻³ / −1.9·10⁻³ | 881 | N − start (55%) |

- The symmetric plateau is a saddle whose only ways down break the twin, the "all slots alike" saddle of TINYCORE.md
  §3.2.
- Its exits do not lead to the right wiring or its twin. They lead to "N − start", the twin of the shallow
  subtract-at-most-once machine: the bypass again.
- With equal options that are *truly* equivalent the exit would not matter. Here the saddle sits next to the bypass,
  and the bypass wins.

### 3.4 Whole wirings under one change: dead ends everywhere
`exact.py flips`. With the true table, score every one of the 1,152 wirings (share of answers right), and ask which
are local optima: no single switch change improves them.

| size | local optima (all but one wrong) | greedy one-change climb from a random wiring reaches the truth | best wrong local optimum |
|---|---|---|---|
| (3, 2, 1) | 180 | 14% (9% by log-likelihood) | v reversed + shift down: 52% |
| (3, 2, 2) | 279 | 14% (6%) | shift stay + ports wrong + keep none: 33% |
| (4, 2, 1) | 139 | 17% (10%) | v reversed + shift down: 37% |

- This is the landscape seen by anything that runs one hard wiring and moves one switch at a time: straight-through
  gradients, Gumbel sampling, try-one-change-and-keep.
- The best wrong local optima pair two wrong switches that partly cover for each other: reading v backwards and shifting
  down.
- A soft blend or a mixture sees all combinations at once, so it does not stop there.

### 3.5 Impostors
**Wirings that pass the label-free checks** (`exact.py checks`, the true table, 1,152 wirings):

| size | every answer right | pass the laws (x mod N = x, N mod N = 0) | pass the range check | pass both |
|---|---|---|---|---|
| (3, 2, 1) | 1 | 64 (the right one among them) | 398 | 20 |
| (3, 2, 2) | 1 | 64 | 396 | 18 |

- The laws pin b = N and keep = stop. They leave a ∈ {prev, start} open (with x ≤ N at most one subtraction is ever
  needed), and they never see the reading, the shift or the readout, which they bypass by design.
- The range check rules out most shifts and write-backs that let the register grow.
- Together they leave 18–20 wirings, one right. The rest have a wrong shift ("stay", "skip"), the high-places
  readout, or the harmless one-subtraction alias.
- So as *filters* on whole wirings, with the table right, the checks cut the space 60-fold. As *teachers* of soft
  switches from scratch they pull toward the bypass (§3.1, §6).

**Wirings that fit every answer with a table of their own.** Every wiring gets its own table, fitted from the answers
(best of 2 seeds, 600 steps):
- **(3, 2, 1), 4 switches** (`exact.py impostors`): with the true table, only the right wiring fits every answer. With
  a table fitted per wiring from the answers, 2 wirings fit every training answer: the right one and its twin with a
  mirrored table. Neither certifies (unused table rows). A typical wrong wiring fits 29% (median; 90th percentile
  40%).
- With the answers plus the cell check, only the right wiring fits, and it certifies. The cell check fixes the table's
  meaning ("first input minus second"), which removes the twin.
- **(3, 2, 1), 7 switches** (`census` suite, all 1,152 wirings):
  - from the answers alone, exactly 2 wirings fit every training answer: the right one and the twin (a = N,
    b = prev, with a mirrored table). Neither certifies: one or two table rows no training question uses stay wrong
    (0.91–0.95 at (3, 3, 2)). A typical wrong wiring fits 26% of the training answers (90th percentile 36%, 99th 44%);
  - with the answers plus the cell check, exactly 1 wiring fits: the right one, and it certifies. No wrong wiring of
    the 1,152 can fit the answers with any table the cell check allows.

So at this size the answers do pick out the wiring (up to the twin). Failing from the answers alone is a search
problem, not an identifiability one.

### 3.6 Lock-in versus a passing dip
Probabilities of the right option over training (`phase` rows, (3, 2, 1)):

| run | step 25 | step 50 | step 200 | at the end |
|---|---|---|---|---|
| soft, cell: minuend "prev" | 0.13 | 0.63 (all switches right; exact) | – | 1.00 (step 150) |
| soft, answers only: minuend "prev" | 0.08 | 0.01 | 0.000 | 0.000 at step 2,000; every other switch right; 66% of answers |
| factorized mixture, answers only: "prev" / "stop" | 0.11 / 0.70 | 0.01 / 0.92 | 0.001 / 0.58 | 0.000 / 0.60 (step 1,275, no change for 600 steps) |
| whole-wiring mixture, answers only: "prev" (seed 1) | 0.33 | 0.33 | 0.06 | 0.22 at step 2,000, still rising |
| whole-wiring mixture, answers only: "prev" (seeds 2, 3) | 0.33 | 0.32–0.33 | 0.54 | 1.00: every training answer right at steps 425 and 1,050 |

- The soft switches and the factorized mixture drive a whole option to zero at once: every wiring with "prev" loses
  weight together.
- The whole-wiring mixture keeps a separate weight per wiring, so the right one stays reachable while the table
  improves. Its two successes still fail beyond training: table rows those questions never use stay wrong.

### 3.7 How the flat stretch grows with size
`depth` suite: the 7 tied switches, 4 seeds (2 for the slowest cells), (3, 2, k) for k = 1–5 and (2, 3, k) for
k = 1–6. "Joint" trains on all k quotient digits from the start (soft: up to 4,000 steps); "staged" adds one quotient
digit at a time (600 steps at most per stage). The factorized mixture ran only where it is affordable.

<!-- DEPTH -->
Table given:

| | soft joint | soft staged | fmix joint | fmix staged |
|---|---|---|---|---|
| (2, 3, 1) | 4/4 (75) | · | 4/4 (50) | · |
| (2, 3, 2) | 0/4 | 4/4 (75) | 4/4 (50) | 4/4 (50) |
| (2, 3, 3) | 0/4 | 4/4 (75) | 4/4 (50) | 4/4 (50) |
| (2, 3, 4) | 0/4 | 4/4 (75) | 4/4 (25) | 4/4 (50) |
| (2, 3, 5) | 0/4 | 4/4 (75) | · | · |
| (2, 3, 6) | 1/4 (75) | 4/4 (75) | · | · |
| (3, 2, 1) | 4/4 (50) | · | 4/4 (50) | · |
| (3, 2, 2) | 4/4 (100) | 4/4 (50) | 4/4 (25) | 4/4 (50) |
| (3, 2, 3) | 4/4 (175) | 4/4 (50) | 4/4 (25) | 4/4 (50) |
| (3, 2, 4) | 4/4 (162) | 4/4 (50) | · | · |
| (3, 2, 5) | 4/4 (75) | 4/4 (50) | · | · |

Table taught by the cell check:

| | soft joint | soft staged | fmix joint | fmix staged |
|---|---|---|---|---|
| (2, 3, 1) | 4/4 (50) | · | 4/4 (50) | · |
| (2, 3, 2) | 1/4 (150) | 4/4 (50) | 4/4 (50) | 4/4 (50) |
| (2, 3, 3) | 0/4 | 4/4 (50) | 4/4 (62) | 4/4 (50) |
| (2, 3, 4) | 0/4 | 4/4 (50) | 4/4 (62) | 4/4 (50) |
| (2, 3, 5) | 0/4 | 4/4 (50) | · | · |
| (2, 3, 6) | 0/4 | 4/4 (50) | · | · |
| (3, 2, 1) | 4/4 (50) | · | 4/4 (50) | · |
| (3, 2, 2) | 4/4 (175) | 4/4 (50) | 2/4 (75) | 4/4 (50) |
| (3, 2, 3) | 4/4 (262) | 4/4 (50) | 1/4 (75) | 4/4 (50) |
| (3, 2, 4) | 3/4 (600) | 4/4 (50) | · | · |
| (3, 2, 5) | 3/4 (275) | 4/4 (50) | · | · |
<!-- /DEPTH -->

- **Trained on all quotient digits at once, the soft switches' flat stretch grows with depth until they fail.**
  - Base 3, with the table taught at the same time: 50, 175 and 262 steps for 1–3 quotient digits; at 4–5 digits some
    runs fail (3/4 each) and the rest take 275–600 steps.
  - Binary fails much sooner: from 2 quotient digits on, 0–1 of 4, with or without the table given. The binary
    failures are the "N − N" machine of §2.2.
- **The size of the numbers matters much less than the depth**: across the phase map (§2.1), one quotient digit takes
  50 steps at bases 3–5 and widths 2–3.
- **Added one digit at a time, the flat stretch does not grow at all**: 50–100 steps (median 50–75) at every depth,
  both bases, 72 of 72 runs. A tied switch learned with one quotient digit is already right for all of them, so the
  later stages cost nothing.
- **The factorized mixture barely notices depth, except for its own trap** (91 of 96 runs, 25–75 steps). Each whole
  wiring is judged on its own answers. Its 5 failures are the co-adaptation of §2.1 (v read backwards, the high places
  read out), at (3, 2, 2) and (3, 2, 3) with the table taught at the same time. Added one digit at a time it is 4/4
  there too.
- **The twin saddle's escape rate** (§3.3) is −2·10⁻⁴ to −2·10⁻² and does not order by size. Under Adam it is not
  what sets the flat stretch here; the bypass dips and the deep blends are.

## 4. Tied or untied switches, and one step at a time
`untie` suite, 4 seeds. "Untied" gives a switch one copy per position: shift and keep one per division step, a and b
one per step and link. At (3, 2, 3) untying all four makes 21 switches and about 2^34 wirings. "Staged" trains on one
quotient digit, then two, and so on (600 steps at most per stage); a position never trained reuses the last trained
one. Rows: size, and the table given ("exact") or taught by the cell check ("learn").

<!-- UNTIE -->
| | soft a+b joint | soft a+b staged | soft shift+a+b+keep joint | soft shift+a+b+keep staged | soft shift+keep joint | soft shift+keep staged | soft tied joint | st shift+a+b+keep joint | st shift+a+b+keep staged |
|---|---|---|---|---|---|---|---|---|---|
| (2, 3, 3) exact | 0/4 | 3/4 (125) | 0/4 | 4/4 (125) | 0/4 | 4/4 (125) | 0/4 | 0/4 | 0/4 |
| (2, 3, 3) learn | 0/4 | 4/4 (87) | 0/4 | 4/4 (125) | 0/4 | 4/4 (100) | 0/4 | 0/4 | 0/4 |
| (3, 2, 2) exact | 0/4 | 4/4 (75) | 0/4 | 4/4 (75) | 4/4 (350) | 4/4 (75) | 4/4 (100) | 0/4 | 0/4 |
| (3, 2, 2) learn | 0/4 | 4/4 (87) | 0/4 | 4/4 (100) | 4/4 (537) | 4/4 (87) | 4/4 (175) | 0/4 | 0/4 |
| (3, 2, 3) exact | 0/4 | 4/4 (100) | 0/4 | 4/4 (100) | 0/4 | 4/4 (112) | 4/4 (175) | 0/4 | 0/4 |
| (3, 2, 3) learn | 0/4 | 4/4 (100) | 0/4 | 4/4 (112) | 0/4 | 4/4 (100) | 4/4 (262) | 0/4 | 0/4 |
| (3, 3, 2) exact | 3/4 (100) | 4/4 (75) | 4/4 (137) | 4/4 (87) | 4/4 (75) | 4/4 (75) | 4/4 (62) | 0/4 | 0/4 |
| (3, 3, 2) learn | 4/4 (100) | 4/4 (87) | 4/4 (112) | 4/4 (100) | 4/4 (62) | 4/4 (75) | 4/4 (62) | 0/4 | 0/4 |
| (4, 2, 2) exact | 3/4 (500) | 4/4 (100) | 0/4 | 4/4 (125) | 4/4 (100) | 4/4 (75) | 4/4 (75) | 0/4 | 0/4 |
| (4, 2, 2) learn | 1/4 (375) | 4/4 (87) | 0/4 | 4/4 (100) | 3/4 (100) | 4/4 (75) | 3/4 (75) | 0/4 | 0/4 |
<!-- /UNTIE -->

- **Untied soft switches mostly fail when trained on all quotient digits at once.** Untied ports fail at 6 of the 10
  rows, all four untied fail at 8 of 10. Untied shift and write-back work at two quotient digits (at (3, 2, 2) slowly:
  350–537 steps against 100–175 tied) and fail at three. The exception is (3, 3, 2), where 18 moduli instead of 6
  make every untied variant work.
- **In every failed run the last step's ports come out right and the first step's come out wrong** (local runs). The
  first step's lessons pass through a later step that is still undecided, as in TINYCORE.md §3.4.
- **Tied**, a switch is taught by every position at once, including the one next to the answer. That works at every
  size but binary (2, 3, 3), where even the tied soft switches fail (0/4; §2.2), and in one run at (4, 2, 2).
- **One step at a time is the most robust recipe**: each new step is the one next to the answer while it is learned.
  Every untied soft variant then certifies at every size, binary included (119 of 120 runs), in 75–125 steps, at one
  step more than trained.
- **Straight-through fails everywhere**, untied or not, staged or not (0 of 80): committing early is not fixed by a
  curriculum.
- After staging, a few positions differ from the reference without being wrong: for a chain's first link, "previous
  trial" and "the step's start" are the same register.

## 5. The schoolbook core
`school` suite, 4 seeds. Each cell: runs whose argmax wiring is exactly the true one and whose answers are all right /
runs (median steps); in brackets, the share of answers right in the others. Columns: times table given ("exact") or
learned from its per-cell ladder ("ladder"); labels on the product or only after mod N; "rladder" uses no labels at
all, only the routed ladder and the per-cell ladder (the table learned).

<!-- SCHOOL -->
| | exact mod | exact product | ladder mod | ladder product | rladder no labels |
|---|---|---|---|---|---|
| (3, 2) abs soft | 0/4 [0.97-0.97] | 4/4 (25) | · | · | 4/4 (100) |
| (3, 2) rel soft | 0/4 [0.97-0.97] | 4/4 (25) | 0/4 [0.97-0.97] | 4/4 (87) | 4/4 (100) |
| (3, 2) rel st | 0/4 [0.48-0.95] | 0/4 [0.21-0.37] | · | · | · |
| (3, 2) tied mix | 4/4 (25) | 4/4 (25) | · | · | · |
| (3, 2) tied soft | 4/4 (75) | 4/4 (25) | 4/4 (262) | 4/4 (75) | 4/4 (75) |
| (3, 2) tied st | 2/4 (37) [0.40-0.84] | 3/4 (50) [0.21-0.21] | · | · | · |
| (3, 3) abs soft | 0/4 [0.98-0.98] | 4/4 (50) | · | · | 4/4 (125) |
| (3, 3) rel soft | 4/4 (25) | 4/4 (25) | 4/4 (200) | 4/4 (100) | 4/4 (112) |
| (3, 3) rel st | 0/4 [0.20-0.27] | 0/4 [0.07-0.09] | · | · | · |
| (3, 3) tied mix | 4/4 (25) | 4/4 (25) | · | · | · |
| (3, 3) tied soft | 4/4 (25) | 4/4 (25) | 4/4 (50) | 4/4 (62) | 4/4 (62) |
| (3, 3) tied st | 2/4 (25) [0.15-0.66] | 2/4 (25) [0.07-0.07] | · | · | · |
| (4, 2) abs soft | 4/4 (25) | 4/4 (25) | · | · | 4/4 (75) |
| (4, 2) rel soft | 4/4 (25) | 4/4 (25) | 0/4 [0.94-0.94] | 4/4 (75) | 4/4 (100) |
| (4, 2) rel st | 0/4 [0.29-0.49] | 0/4 [0.13-0.38] | · | · | · |
| (4, 2) tied mix | 4/4 (25) | 4/4 (25) | · | · | · |
| (4, 2) tied soft | 4/4 (25) | 4/4 (25) | 4/4 (75) | 4/4 (75) | 4/4 (75) |
| (4, 2) tied st | 0/4 [0.24-0.65] | 3/4 (25) [0.12-0.12] | · | · | · |
<!-- /SCHOOL -->

- **Graded on the product, every soft description works**, even with a switch per product digit: each product digit
  lands in an answer digit, so every connection gets its own direct lesson.
- **Graded only on the remainder, the tied rule always works** (whole-wiring mixture and soft, at every size). **Per-
  digit switches depend on the size:**
  - they stall at (3, 2), at 97% of remainders right (91% of products). The stalled wiring sends the high digits of
    cells (0, 1), (1, 0) and (1, 1) to column i + j − 1 instead of i + j + 1. In base 3 with two digits a high digit
    is nonzero only when both input digits are 2, so those connections are exercised on few rows, and through the
    remainder their lesson is scrambled. The remainder also hides part of the error (97% of remainders right with 91%
    of products);
  - at (3, 3) per-digit offsets work and any-column switches stall at 98%; at (4, 2), where high digits are nonzero
    far more often, both work;
  - with the times table learned at the same time, per-digit switches through the remainder fail at (3, 2) and
    (4, 2) (97%, 94%).
- **A local law that reads the wiring removes the dependence on size.** The routed ladder checks row(x, b + 1) =
  row(x, b) + x·B^j for every x and every row j, through the learned placement and a designed adder, only where the
  rungs below check out (SQUARE.md's verified frontier). It never sees an answer, and the remainder never gets in its
  way: with the times table learned from its own ladder at the same time, every routing at every size is exact,
  4/4, in 62–125 steps.
- **Straight-through fails per digit everywhere** (0 of 24) and succeeds in only 12 of 24 runs even tied.

### 5.1 H1's step in miniature: multiply, then reduce, everything learned
`school.py` with `div_learn` (rows in `runs/composite_local.jsonl`, run locally). x·y in the schoolbook core, then the
division core, graded only on x·y mod N. Learned together:
- the placement: tied (2 switches) or per digit (16 switches at w = 2);
- all 7 division switches;
- the times table and the subtraction table.

That is 18,432 tied wirings, or 4^16 · 1,152 ≈ 2^42 per-digit ones. (3, 2): 199 questions, 6 seeds; (4, 2): 1,226
questions, 4 seeds. Each cell: runs exact with the reference wiring / runs (steps to exact); in brackets, the share
of answers right in the others.

| recipe | size | tied, soft | per digit, soft | tied, straight-through | tied, soft, the division's reading, shift and readout enumerated (32 sub-wirings) |
|---|---|---|---|---|---|
| answers + cell check + times-table ladder + routed ladder | (3, 2) | 4/6 (125–200) [47%] | 6/6 (100–125) | 2/6 (50) [36–47%] | 4/4 (150–175) |
| | (4, 2) | 1/4 (150) [29%] | 4/4 (100–125) | 1/4 (125) [23%] | 4/4 (125) |
| the same without the routed ladder | (3, 2) | 0/6 [49–60%] | 0/6 [65%] | · | · |
| | (4, 2) | 0/4 [40%] | 0/4 [40–45%] | · | · |
| answers alone | (3, 2) | 0/6 [64%] | 0/6 [61–73%] | · | · |
| | (4, 2) | 0/4, one of them exact as a twin [34–100%] | 0/4 [37–67%] | · | · |

- **With a law for every table and for the placement, the whole step is learned**, wiring and tables together, from
  answers that only come after the remainder: 10 of 10 runs with per-digit placement, 8 of 8 with tied placement once
  the division's interacting switches are enumerated together.
- **Without the routed ladder it fails in all 20 runs**, even though both tables are taught by their checks: the
  placement is then seen only through the remainder.
- **The tied soft failures** (5 of 10) got the placement and both tables exactly right: every product exact. They
  went wrong in the division's switches: the dividend read backwards, the answer read from the high places, and with
  them the write-back. No law reads those; they learn from the answers, and they depend on each other. Enumerating
  them together (the last column) fixes it, as the hybrid grouping did in §2.1.
- **The one exact answers-only run is a twin**: its placement swaps the low and high digit columns, and its times
  table swaps the low and high digits (only the 7 rows whose product is 0 agree with the reference). That machine
  computes every product exactly, so it is right everywhere, just not the reference wiring.

## 6. The tricks
The known tricks, applied to the switches (`tricks` suite, 7 switches, 4 seeds; `tricks2` for the whole-wiring
mixture; `stages` and `untie` for one step at a time). Each cell: certified / seeds (median steps); "f" counts seeds
that fit every training answer and failed a bigger size.

| trick | what it does here |
|---|---|
| fixed meaning (anchor) | the law x − 0 = x on the table: its two inputs get fixed roles, which removes the twin |
| local self-checks (laws) | x mod N = x, N mod N = 0 through the chain and write-back, and "the register stays below N" after every step |
| table first | the switches frozen for the first 200 steps (answers only) or 50 steps (with the cell check) |
| symmetry breaking at init (init1) | routing logits drawn with σ = 1 instead of 0.1 |
| no symmetry breaking (sym) | every logit exactly 0 |
| try-and-keep (search) | every 50 steps, the best single-switch change of the argmax wiring on the current table is taken |
| own hard path (ownpath) | hard switches and hard write-back forward, the soft table on the chosen path (H1's phase B) |

**From the answers alone:**

<!-- TRICKS_ANS -->
| | (3, 2, 1) fmix | (3, 2, 1) soft | (3, 2, 1) st | (3, 2, 2) soft | (3, 2, 2) st |
|---|---|---|---|---|---|
| anchor | 0/4 f2 | 0/4 | 0/4 | 0/4 | 0/4 |
| anchor+laws | 0/4 f4 | 0/4 | 0/4 | 0/4 | 0/4 |
| init1 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| laws | 0/4 f3 | 0/4 | 0/4 | 0/4 | 0/4 |
| none | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| search | · | 0/4 | 0/4 | 0/4 | 0/4 |
| sym | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| tablefirst | 0/4 f2 | 0/4 | 0/4 | 0/4 | 0/4 |
<!-- /TRICKS_ANS -->

**With the cell check:**

<!-- TRICKS_CELL -->
| | (3, 2, 1) fmix | (3, 2, 1) soft | (3, 2, 1) st | (3, 2, 2) fmix | (3, 2, 2) soft | (3, 2, 2) st |
|---|---|---|---|---|---|---|
| init1 | 4/4 (50) | 2/4 (62) | 1/4 (50) | 2/4 (62) | 2/4 (150) | 1/4 (75) |
| laws | 4/4 (87) | 0/4 | 1/4 (100) | 4/4 (150) | 0/4 | 1/4 (425) |
| none | 4/4 (50) | 4/4 (50) | 0/4 | 0/4 | 4/4 (175) | 0/4 |
| ownpath | · | 1/4 (75) | 1/4 (200) | · | 1/4 (75) | 0/4 |
| search | · | 4/4 (50) | 0/4 | · | 4/4 (137) | 0/4 |
| sym | 4/4 (50) | 4/4 (75) | 0/4 | 0/4 | 4/4 (175) | 0/4 |
| tablefirst | 4/4 (75) | 4/4 (100) | 0/4 | 4/4 (100) | 4/4 (175) | 0/4 |
<!-- /TRICKS_CELL -->

<!-- TRICKS2 -->
**The whole-wiring mixture from the answers alone** (`tricks2`, (3, 2, 1)):

| | (3, 2, 1) mix |
|---|---|
| anchor | 0/4 |
| anchor+laws | 0/4 |
| init1 | 0/4 |
| laws | 0/4 |
| none | 0/4 f2 |
| sym | 0/4 |
| tablefirst | 0/4 f1 |
<!-- /TRICKS2 -->

**One quotient digit at a time** (`stages` suite; 4 seeds; the joint factorized mixture at (3, 2, 3) did not finish
and was stopped):

<!-- STAGES -->
From the answers alone:

| | joint | staged |
|---|---|---|
| (2, 4, 3) soft | 3/4 (225) | 4/4 (112) |
| (2, 4, 3) st | 0/4 | 0/4 |
| (3, 2, 3) soft | 0/4 | 0/4 |
| (3, 2, 3) st | 0/4 | 0/4 |
| (4, 2, 2) soft | 0/4 | 0/4 f4 |
| (4, 2, 2) st | 0/4 | 0/4 |

With the cell check:

| | joint | staged |
|---|---|---|
| (2, 4, 3) fmix | 4/4 (50) | 4/4 (50) |
| (2, 4, 3) soft | 0/4 | 4/4 (50) |
| (2, 4, 3) st | 0/4 | 0/4 |
| (3, 2, 3) fmix | · | 4/4 (50) |
| (3, 2, 3) soft | 4/4 (262) | 4/4 (50) |
| (3, 2, 3) st | 0/4 | 0/4 |
| (4, 2, 2) fmix | 3/3 (50) | 4/4 (50) |
| (4, 2, 2) soft | 3/4 (75) | 4/4 (50) |
| (4, 2, 2) st | 0/4 | 0/4 |
<!-- /STAGES -->

- With the cell check, staging makes soft switches 4/4 at every size here (50 steps). Joint training fails at binary
  (2, 4, 3) (0/4) and once at (4, 2, 2).
- From the answers alone, staging helps where the bypass is weak:
  - binary (2, 4, 3) certifies 4/4 (and 3 of 4 even joint). In binary the chain has a single link, so "subtract from
    the start" and "chain" are the same wiring and the main bypass does not exist; 8 moduli also cover the 8-row
    table;
  - at (4, 2, 2) staging finds the wiring in 4 of 4 runs, which then fail beyond training on unused table rows;
  - at (3, 2, 3) it does not help.
- Straight-through fails staged or not.

**Which tricks change the picture, which only shorten the flat stretch, which hurt:**
- **Change the picture:**
  - a table teacher that ignores the wiring (the cell check): from almost nothing to every run of the whole-wiring
    mixture and 27 of 28 for soft switches (§2.1);
  - one switch per kind of connection (tying): untied 0/4 → 4/4 (§4);
  - one step at a time, for untied switches (0/4 → 4/4, §4) and for tied ones in deep chains: binary with 2–6
    quotient digits and base 3 with 4–5 go from 0–3 of 4 to 4/4 (§3.7);
  - a local law that reads the wiring (the routed ladder): per-digit placement through the remainder with the times
    table learned goes from 0 of 8 to every run at every size (§5), and H1's step in miniature from 0 of 20 to 10 of
    10 (§5.1);
  - enumerating interacting switches together: the hybrid at (3, 2, 2) 0/4 → 4/4 (§2.1); H1's step in miniature,
    tied, 5 of 10 → 8 of 8 (§5.1);
  - for the factorized mixture's own trap at (3, 2, 2) with the cell check, table first (0/4 → 4/4) and the laws
    (0/4 → 4/4) both work. Table first lets the table become right before the reading and readout can co-adapt (with
    the table given the factorized mixture is 4/4 there, §2.2). The laws test each whole wiring on "N mod N = 0";
  - **for the factorized mixture only**, a fixed meaning plus the laws, from the answers alone: it goes from 0 of 4
    seeds fitting to 4 of 4 fitting with the true wiring (anchor alone 2, laws alone 3, the twin in two of them). They
    still fail beyond training, but only because one table row that no training question uses stays wrong, exactly
    as with the wiring given (§2.3). Each whole wiring either passes "N mod N = 0" or not, so in a mixture the laws
    are a crisp test; in a soft blend they reward the half-and-half do-nothing wiring. The whole-wiring mixture does
    not benefit within this budget: with the anchor or the laws 0 of 4 fit, against 2 of 4 without (`tricks2`). Those
    runs ended with the weight still spread over many wirings (per-switch marginals 0.5–0.8). They were stopped after
    600 steps without a change of the leading wiring, and its successes had taken 425–1,050 steps.
- **Only shorten or lengthen the flat stretch:**
  - one step at a time for tied switches in shallow chains: (3, 2, 3) from 175–262 steps to 50 (§3.7);
  - try-and-keep on top of soft switches: (3, 2, 2) 175 → 137 steps;
  - table first: 50 → 75–100 steps;
  - an exactly uniform start versus small random logits: 50 → 75 at (3, 2, 1), the same at (3, 2, 2). The cell check
    already breaks the twin, so no symmetry needs breaking; without it (answers only) breaking symmetry does not help
    either, because the trap is the bypass.
- **Hurt:**
  - committing early: own hard path (4/4 → 1/4 at both sizes), large random routing logits (4/4 → 2/4);
  - the laws as teachers of soft switches (4/4 → 0/4 at both sizes). CONTROL.md's commit gate showed the same pull
    and needed the laws phased in after the borrows were right;
  - straight-through itself: 0–1/4 with every trick, including try-and-keep.

## 7. Prediction for H1

### 7.1 H1's hand-made wiring as switches
From CONTROL.md §2, SQUARE.md and CHAIN.md, the wiring still written by hand, as tied switches (one per kind of
connection, shared by every digit, step, link and round where it repeats):

| part | connection | plausible options | has a local law that reads it? |
|---|---|---|---|
| squaring | column of a product's low digit (offset from i + j) | 4 | yes: the ladder, routed (§5) |
| squaring | column of a product's high digit | 4 | yes: the same |
| squaring | which register is squared (the state, N, T, …) | 3 | no |
| squaring | the adder's operands, a − (0 − b) | 3 | yes: (a − b) + b = a on all pairs |
| division | which register is divided | 3 | no |
| division | starting register (top digits, bottom digits) | 2 | partly: the start is below N |
| division | where the next digit comes in (shift) | 4 | partly: the range check |
| division | each link's minuend and subtrahend | 3 × 3 | yes: x mod N = x, N mod N = 0 (after the table is right) |
| division | which trial is kept (the commit gate's target) | 4 | yes: the same laws |
| division | step order (from the top digit) | 2 | no |
| loop | the countdown's two operands (C − 1) | 3 × 3 | partly: the count's own borrow |
| loop | what a commit updates (state, count) | 3 | no |
| output | which register and places are the answer | 2 | no |
| input | reading N, x and T (IO.md, already learned) | 24 ways | yes: x < N |

About 15 switches and 2^24 tied wirings (2^28 with the 24 readings), if each kind of connection is one switch.
Untied, every digit place, step, link and cell has its own switch: thousands of switches and astronomically many
wirings.

### 7.2 Which parametrization should scale
The tiny cores say:
1. **Teach every table from a check that does not depend on the wiring, before or alongside the wiring.**
   - The subtractor: (a − b) + b = a on spec-made pairs (all digit pairs, or random numbers), not only on the machine's
     own link inputs. The own-input version has the borrow-ignored trap (§2.3) and lets a wrong wiring starve rows.
   - The times table: its per-cell ladder (SQUARE.md).
   - Without such a teacher the wiring's first lessons bypass the untrained table, and the bypass locks in (§3.1, §3.6).
2. **Describe the wiring as tied soft switches.** One switch per kind of connection (about 15 for H1), soft-routed.
   Seven soft switches cost about 1.5 single machines per step (§1.1), and the cost grows with the options, not with
   the 2^24 wirings they span. With a table teacher they find the 7 tied switches in 27 of the 28 runs of the phase
   map (§2.1), in 50–175 steps. Their landscape flattens as the chain deepens, though (§3.2). With the table given
   they fail in binary with three quotient digits and in half the runs at (5, 2, 2) (§2.2). H1's chain is deep
   (8 steps of 9 links), so soft switches there need point 4.
3. **Give the squaring's placement its own local law.** The ladder computed through the learned placement
   (row(x, b+1) = row(x, b) + x·B^j) teaches even per-cell placement in 62–125 steps without any answer, while the
   times table is learned too (§5). Without it the placement sees only remainders. There per-cell switches work only
   at some sizes and a tied rule always does (§5); in the full tiny step nothing works without it (0 of 20, §5.1).
4. **Add one division step at a time.** This is the most dependable recipe for soft switches in the tiny cores: 72
   of 72 runs across 2–6 quotient digits and 119 of 120 untied runs, in 50–125 steps (§3.7, §4).
   - A tied switch learned at one quotient digit is already right at all of them, so the later stages are free.
   - Trained on all digits at once, tied soft switches fail from 2 quotient digits in binary and from 4 in base 3;
     H1 divides with 8 steps.
   - Untied switches need it in any case: each new step is learned while it is the one next to the answer.
5. **Mix whole wirings only for the few switches that only the answers see and that interact**, enumerated together:
   the readings (as IO.md already does with its 24 ways), the start register, the step order, the output. That is
   2^5–2^10 wirings.
   - In the tiny H1 step (§5.1) this was the missing piece. With every table and the placement taught by their laws,
     soft switches still failed the division's reading and readout in a third of the runs at (3, 2) and three quarters
     at (4, 2). Enumerating those four switches (32 sub-wirings) and keeping the rest soft made 8 of 8 runs exact.
   - Splitting interacting switches between the mixture and the soft part locks them into a consistent wrong pair
     (hybrid, §2.1).
6. **Use the remainder laws and the range check as filters, not as first teachers.** With the right table they cut
   1,152 wirings to 20 (§3.5). From scratch they reward doing nothing: keep the number unchanged, or let the register
   stay small (§3.1). CONTROL.md found the same with its commit gate.
7. **Do not commit early.** Straight-through, Gumbel, REINFORCE and one-change search stop in dead ends (§3.4).
   Straight-through certified 1 of 28 runs in the phase map, 0 of 80 untied, 0 of 24 with per-digit placement. Its
   few successes come where a local law gives every switch a clean lesson (§5, §5.1).

### 7.3 How big a wiring space fits in an hour on one H100
- **Whole-wiring mixture: about 2^10–2^12 wirings as H1's machine is written today, up to about 2^13–2^15 fused.**
  Its cost per step grows linearly with the wirings (§1.1).
  - *Launch-bound* (today's code): H1's machine trains at about 0.23 s per step for one wiring (SQUARE.md: frozen at
    step 499, 113 s in). Those steps are bound by Python and kernel launches, so extra wirings riding along the batch
    are nearly free until the GPU saturates. Assume a 30–100× margin before it does, and the 7 s per step that a
    500-step hour allows: 30–100 × 30 ≈ 10^3–3·10^3 wirings.
  - *Compute-bound* (every wiring in one fused batch): one soft table application at base 10 is about 2,400
    multiply-adds (CHAIN.md's link_soft). A T = 1 row needs about 650 of them in the division (8 steps × 9 links × 9
    places) and about 450 in the squaring's learned adds: about 2.6 million forward, 8 million with the backward pass,
    so 16 MFLOP per row and wiring. At a realistic 10 TFLOP/s for such small matrices that is 6·10^5 row-wirings per
    second, 2·10^9 per hour. A mixture needs about 100 steps of 512 rows once the tables are right (§2.2: 25–75 steps
    in the tiny core), so 5·10^4 rows per wiring: about 4·10^4 ≈ 2^15 wirings, or 2^13 if it must run 500 steps.
- **Tied soft switches: the wiring space is not the limit.** 2^24 tied wirings cost about 1.5–3 single machines per
  step. The limit is steps, and depth: in the tiny core the flat stretch grows by about 50 steps per extra quotient
  digit when trained all at once and stays at about 50 when added one digit at a time (§3.7). At 0.23 s per step,
  even 2,000 steps is 8 minutes.
- **Untied soft switches with staging, or with a local law:** a few hundred steps per stage (§4). H1's 8 division steps
  would be 8 stages, about 1,000–2,000 steps. Per-cell placement in the squaring needs the routed ladder rather than
  stages.

## 8. Caveats
- **The options are designed.** Every switch offers a handful of options I chose, and the space contains the right
  wiring. "Learning the wiring" here means picking among designed alternatives. The algorithm still lives in the
  option sets ([REPORT.md](../REPORT.md) §6), now spread over many small choices instead of one fixed design. What is
  measured is how much of the choosing can move into training, and with which description.
- **Tables, not MLPs, and tiny sizes.** Base 2–5, moduli of 2–3 digits, 1–6 quotient digits. H1 is base 10 with
  8-digit moduli and a 16-digit square. The symmetry and bypass arguments hold for any cell whose inputs are routed;
  the step counts do not carry over one to one.
- **The checks bring their own design.** The cell check needs a designed adder (as TINYCORE.md's and H1's do). The
  routed ladder also states what row j means (the multiplier digit at place j, so its row adds x·B^j). The laws and
  the range check state facts about remainders.
- **Untied switches at a longer run.** A position never trained (one more division step than in training) reuses the
  last trained position's switch. Certification at k+1 relies on that.
- **The census** fits one table per wiring from 2 seeds of 600 steps; a wiring whose table is hard to fit could be
  missed. The 7-switch census ran at (3, 2, 1) only.
- **Certification** is two bigger sizes of the same tiny task (one more quotient digit; one more digit of N and one
  more quotient digit), not H1's T-ladder.
- **Small budgets.** 4 seeds per cell (6 for the baselines and the (3, 2) composite, 3 for the first local schoolbook
  runs), at most 2,000 steps (4,000 for soft routing in the depth series), one learning rate (0.05) and init scale
  (0.1). Modal CPUs run about 2–5× slower per thread than the laptop, so the seconds in the rows are not comparable
  across machines.
- **Early stopping.** A run stops 600 steps after its argmax wiring and exact share last changed. That saves the
  long tails of runs locked into a wrong wiring, but it can cut short the whole-wiring mixture, which recovers slowly
  from the answers alone (its successes took 425–1,050 steps).
- **Unfinished jobs.** Six jobs were stopped unfinished: one answers-only whole-wiring mixture at (3, 2, 2) and five
  joint factorized-mixture runs of the `stages` suite. The `depth` suite covers the same cells.
- **The hybrid in the phase map used a bad grouping** (§2.1); the good groupings were tested locally (4 seeds at
  (3, 2, 1) and (3, 2, 2)), and in the composite (§5.1).

## 9. Reproduce
Run from `cpu_first/` with the sotaku venv's Python. Local runs are seconds to a few minutes and stay under 1 GB; the
suites run on Modal (app and Volume `tinyrouting`, folder `v1/`; 4-CPU containers, at most 20).

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.selftest
```

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.baseline
```

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.exact grad0
```

The other exact analyses: `exact gradtrue` (first lessons with the true table), `exact eps` (straight paths),
`exact saddle` (the twin), `exact flips` (one-change local optima), `exact checks` (wirings that pass the label-free
checks), `exact impostors` (4-switch census; 1.4 GB, so better on Modal).

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tinyrouting/modal_tinyrouting.py --suite phase,alone,untie,school,depth,tricks,tricks2,census,stages
```

```bash
~/github/sotaku/venv/bin/python -m modal run tinyrouting/modal_tinyrouting.py --suite phase,alone,untie,school,depth,tricks,tricks2,census,stages --fetch
```

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.composite
```

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.summarize all
```

```bash
~/github/sotaku/venv/bin/python -m tinyrouting.fill
```

`composite` runs §5.1 locally (rows in `runs/composite_local.jsonl`). `fill` rewrites the tables of this file from the
rows (between the `<!-- NAME -->` markers).

Suites (`suites.py`):
- `phase`: parametrization × recipe × size, 7 switches;
- `alone`: the wiring alone, the table given;
- `untie`: tied vs per-step / per-link switches, joint vs one step at a time;
- `depth`: the flat stretch against the number of quotient digits;
- `school`: the schoolbook core;
- `composite`: H1's step in miniature (run locally by `composite.py`);
- `tricks`, `tricks2`, `stages`: §6;
- `census`: the 7-switch impostor census.

Code:
- `machine.py`: the division core, its switches, the multilinear datapath (any number of wirings at once);
- `train.py`: parametrizations, checks, tricks, training and certification;
- `exact.py`: the exact analyses;
- `school.py`: the schoolbook core;
- `baseline.py`, `selftest.py`, `summarize.py`, `render.py`, `fill.py`, `composite.py`, `suites.py`,
  `modal_tinyrouting.py`.
- `runs/<suite>.jsonl`: one row per run; `runs/old/`: an earlier version of the phase rows that stopped at the first
  exact step (superseded).
