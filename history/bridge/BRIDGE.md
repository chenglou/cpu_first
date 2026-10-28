# The bridge: from the digit machine to a generic looped transformer

## In plain words

**The question.** The best H1 models in this project are hand-built calculators whose parts are learned: digit tables,
a keep-or-discard gate, a reader and a writer, and the wiring, chosen from menus ([OVERVIEW.md](../../OVERVIEW.md),
[MENUS.md](../../wider_menus/MENUS.md), [WIDE_ANCHORED.md](../../wider_menus/WIDE_ANCHORED.md)). Generic networks, the
kind nearly every leaderboard entry used, get about 0%. Is the calculator a different kind of thing, or one particular
setting of a generic network? And if it is one setting, how far can the network around it be loosened before training
from random stops finding it?

**A tiny example.** Base 3, two-digit moduli N from 3 to 8, and dividends below 3N (one quotient digit) or 9N (two).
The answer is v mod N. Take N = 5 and v = 14, written lowest digit first as `2 1` and `2 1 1`. Long division the way
the machine does it:
- the register starts at v's top digits and brings in the last one: 14;
- 14 − 5 = 9 does not go below zero, so the register keeps 9;
- 9 − 5 = 4 does not either, so it keeps 4;
- the answer is 4.

**Step 1. The machine is one weight setting of a looped transformer.** Write the division out on a grid, the way it
looks on paper. Every digit of every intermediate number is one token, about 30 to 100 of them. Then:
- three attention heads copy digits between tokens: that is the wiring;
- one small MLP does the digit arithmetic: subtract with a borrow, copy, or keep-or-discard. Those are the tables;
- the same block runs again and again until every token has settled: that is the loop.

With the right weights this network gets every answer exactly right, on every question, at 14 sizes: bases 2 to 5,
moduli of up to 4 digits, up to 4 quotient digits. Every intermediate token matches the machine, even when the scratch
tokens start from random values. The most generic of the three settings checked uses free relative-position
attention and a plain ReLU MLP: 5,081 numbers at base 3, most of them the attention's relative-position tables. So the
generic architecture really contains the machine.

**Step 2. Loosen it one notch at a time, and train from random at every notch.** Each notch used the most generic
lessons it can still express:
- the answers;
- x mod N = x;
- the order facts (x < N);
- a remainder's consistency with itself;
- and, while the network still has a recognizable subtraction cell, the anchors' lessons on that cell.

What happened (§3):
1. **The digit tables can be generic MLPs.** A plain ReLU MLP as the subtraction cell certified in 31 of 32 runs at
   bases 3 to 5. The learned table, the machine's own form, did so in 23 of 32, failing mostly at base 5 on one
   self-consistent wrong entry (§6). Both need the cell's own lessons:
   - x < N on every top part (this teaches the borrows);
   - two order laws of subtraction;
   - every row and column of the cell uses each digit once. Without this one, the MLP cell certified in 11 of 16.

   From the answers alone, nothing certifies.
2. **The plumbing is harder.** Copying a digit and keeping-or-discarding are rules too. When they are learned as
   well, as tables, 4 of 8 runs certify with every lesson. When one MLP has to learn all three rules, 4 of 16 runs
   certify with every lesson, and none of 32 without bijectivity. Its failures get far more training answers right
   on the soft path than once snapped (up to 99% against 37%). It passes blurred digits between cells, a code that
   exact digits cannot carry.
3. **The wiring can be learned only from a short menu.** Each attention head may pick its source from a menu of
   relative positions ("the cell above", "the cell to the left", "N's digit in this column", ...). Training finds
   the true wiring in every run in two cases:
   - with the tables given, up to 10 options per connection (4 of 4 at 10, 8 of 8 at 7);
   - with the tables learned at the same time, at 7 options (8 of 8), provided the tables are learned first while
     the wiring waits.

   Both need a local lesson: the same network, run on a grid of just one subtraction, must leave a number below N
   unchanged and turn N itself into 0. With 13 or more options, no run finds the wiring, even with the tables given.
   Free attention, which can read any cell, never finds it.
4. **Looser still never learns:** cells whose digit meanings are learned, continuous states, untied blocks, a plain
   looped transformer. Several of them do fit the training questions with their soft, blended computation, but one
   size up they score about what "always answer 0" scores. They memorize, like the leaderboard's generic networks.

**Step 3. Why it stops.** The compiled machine serves as a probe: compare the trained network with it, cell by cell
and head by head.
- **Free attention falls for the bypass.** At the start, the answers' gradient pushes most connections toward the
  prompt's own digits, which are clean, instead of the computed cells the machine reads, which are still noise. That
  is the "untrained parts get bypassed" trap of TINYROUTING.md, in attention form. Training follows that push into a
  separate valley: on the straight line from the trained weights to the compiled ones, the loss first rises, then
  falls to 0.
- **The 7-option menu falls for a near miss.** Every seed lands on the same wrong wiring:
  - the subtraction takes away the register's top digit instead of N's digit;
  - the gate reads the borrow of the place below, not the final borrow;
  - the carry route stays undecided.

  Yet the answers alone single out the true wiring. Running every wiring as a hard network shows that exactly one
  wiring gets every training answer right: one of 823,543 at 7 options, of 10 million at 10, of 62.7 million at 13,
  and of 1.28 billion at 20 (§4). So the wiring is identified; the soft network just cannot find it. The local
  one-link lesson pins most of the connections by itself, and with it the search succeeds up to 10 options.
- **The MLP cell falls for ties.** It settles on tables where two entries of a column give the same digit. The order
  laws alone do not break the tie; bijectivity does.

Known fixes tried, in their most generic form:
- on free attention:
  - the local lesson on a sub-grid, one link at a time, a locality window, a lower learning rate;
  - two from the user's looped Sudoku solver (§8): the prompt fed only at the first tick, and snapped digits fed
    back with straight-through gradients. Both remove the bypass's pull, but the connections then settle on
    constant zeros instead;
  - 4 parallel copies of every head (§7);
- on the 7-option menu, taught by the answers: whole options scored one connection at a time, with the others
  blended.

None of them works. The fix that does work in this project is to score whole combinations of connections pass or
fail ([WIRING.md](../learned_wiring/WIRING.md), [MENUS.md](../../wider_menus/MENUS.md)). Done by brute force, it finds
the one right wiring among 62.7 million at 13 options, where every soft run failed. Done block by block, each block
scored by a lesson that reads only it, it also works at free attention's width. With 34 relative sources per
connection (5·10^10 wirings), it found the one wiring that fits every answer, the true one, in 45 seconds on a
laptop (§4). Brute force at that width would have taken about 100 CPU-hours. Splitting the search into blocks with
local lessons closes that budget gap.

**Step 4. H1.** The loosest notch that learns reliably at the tiny size and is new at H1 is the MLP digit cell, and
the H1 machine already learns its wiring from wide menus. So the certified wide-menu machine
([`submission_wide_anchored.py`](../../wider_menus/submission_wide_anchored.py)) got its two tables replaced by generic
ReLU MLPs, with every lesson and schedule unchanged ([`h1/submission_wide_mlp.py`](h1/submission_wide_mlp.py), §9).
**It certified.** On the untouched official manifest on an H100, the file run byte for byte reached T = 64 on both
ladders, seen and unseen moduli, every rung 768/768, and every split was exact. Training took 1,048 s and evaluation
98 s. Both MLPs came out computing exactly the true tables. The same file also certified on all 8 CPU seeds
through the same evaluator, every one of them with both MLPs exact (§9).

**An over-parameterized compile (§7).** Exact transcribed weights might be too compressed to move, so each compile
also got a redundant twin that computes the same thing, in the style of Net2Net: duplicated hidden units, parallel
copies of every attention head, and a digit alphabet with 4 symbols per meaning.
- **Near the solution, redundancy helps** where the structure is constrained or a local lesson gives a signal. The
  7-option menu with 4 head copies recovers from heavy noise on the answers alone in 4 of 4 runs, against 1 of 4 with
  single heads. The widened all-MLP rule recovers from moderate noise in 4 of 4, against 2 of 4.
- **From random it rescues almost nothing:** 1 of 32 widened runs certified, against 0 of 32 minimal ones.
- **In the free formats exact starts drift away** into impostors or blurred analog codes, and the redundant alphabet
  drifts worst.

Redundancy helps the optimizer move; it does not supply a signal that is not there.

**So how much built-in structure does learning need?** In this machine:
- **Not needed:** hand-written digit tables. A generic ReLU MLP does as the digit cell, at the tiny size and at H1,
  if the cell gets the anchors' lessons.
- **Not needed as information:** the menu. The answers alone single out the true wiring among more than a billion. The
  menu is needed only for the soft search, which finds the wiring among about 10 options per connection and not among
  13. A whole-option search finds it at free attention's width.
- **Needed:**
  - the skeleton: the grid of tokens; which token is a subtraction, a copy or a keep-or-discard; what a cell's state
    means; the number of ticks;
  - a short menu per connection for soft training, or else a search over whole combinations of connections, scored
    block by block by local lessons (45 s at free attention's width here);
  - lessons stated on the cells (order facts, bijectivity) and on small sub-grids (one link's x mod N = x).

The lessons on cells and sub-grids stop being expressible exactly where that skeleton is removed (§5). That is where
every generic network here stops learning.

## Contents

1. The grid and the compile
2. The notches
3. The ladder: every notch trained from random
4. The menu's width
5. Which lessons each notch can express
6. Diagnosis: the compiled machine as a probe
7. Over-parameterized compiles: the Net2Net arm
8. Two more fixes for the bypass
9. H1: the MLP digit cells
10. Caveats
11. Reproduce

## 1. The grid and the compile

`grid.py`, `net.py`, `compile.py`. The tiny machine is TINYROUTING.md's division core (base B, w-digit moduli,
dividends v < N·B^k), with H1's commit-gate chain: each link subtracts N from the chain register, and the result
replaces the register only if the link did not borrow ([CHAIN.md](../../learned_remainder/CHAIN.md),
[CONTROL.md](../../learned_remainder/CONTROL.md)). P = w + 1 places.

**The grid** for (B, w, k) = (3, 2, 1). Rows run top to bottom and columns run −1 .. P − 1. N0.. are N's digits and
v0.. are v's; `.` is a constant zero:

```
         N    .  N0  N1   .
        R0   v0  v1  v2   .
  SHIFT(0)    .  sh  sh  sh
  SUB(0,1)    .  su  su  su
  SEL(0,1)    .  se  se  se
  SUB(0,2)    .  su  su  su
  SEL(0,2)    .  se  se  se
```

Each step is a SHIFT row (the register moved up one place, the next digit brought in from column −1), then B − 1
links, each a SUB row (one place of the subtraction's ripple per cell) and a SEL row (keep the old register if the
link borrowed). The last SEL row of a step holds, at column −1, the digit the next step brings in. The answer is the
last row's places 0 .. w − 1. In the example N = 5, v = 14, the rows hold:
- SHIFT 14;
- SUB 9, no borrow, so SEL 9;
- SUB 4, no final borrow, so SEL 4.

**The looped transformer** (`net.py`). Every cell is a token whose state is a digit distribution plus a borrow
distribution. One block runs D times on all cells at once; inputs are re-injected after every tick. A tick is:
1. Three attention heads, A, B and C, each copy another cell's state into the cell's port of that name. The compiled
   patterns, relative to a cell at (r, c):

   | kind | A | B | C | rule |
   |---|---|---|---|---|
   | SHIFT | (r − 1, c − 1) | a zero | a zero | copy A's digit |
   | SUB | (r − 1, c) | N's row, column c | (r, c − 1) | A − B − C's borrow: digit and borrow |
   | SEL | (r − 2, c) | (r − 1, c) | row r − 1's top place | C's borrow ? A : B |

2. One MLP, the rule: (kind, A's digit, B's digit, C's borrow) → (digit, borrow).

After D = k(1 + (B − 1)(P + 1)) ticks, the circuit's depth, every cell holds the machine's value, whatever the scratch
cells held before.

**The compile** (`compile.py`, record `records/compile.json`). Three weight settings of the same code:
- "fixed": the compiled patterns and the truth tables;
- "menu": the menu heads (§2) with their logits set on the true options;
- "free": the most generic point. Its free relative-position attention scores β · (how many of the compiled pattern's
  features a key matches), and its rule is a 2-layer ReLU MLP whose hidden units each detect one (kind, a, b, c) and
  whose output weights are β_o times the truth tables. That is 5,081 numbers at B = 3: 3,996 in the relative-bias tables
  and 1,085 in the MLP, which has 54 hidden units.

Every question of each size was checked (a random 20,000 for the two largest):
- on the hard path, with argmax attention and snapped states;
- on the soft path, plain softmax attention and unsnapped states, which is the network as trained;
- with the scratch cells started from random states.

The widened twins of §7 are checked the same way (`compile_wide.py`, record `records/compile_wide.json`): all 8 are
exact at (3, 2, 1), (3, 2, 2) and (4, 2, 2). They cover head copies averaged or side by side, Net2Net-duplicated
hidden units, a prompt that only the first tick sees (its cells read themselves), and the free-symbol format with 1
or 4 symbols per meaning.

<!-- COMPILE -->
| (B, w, k) | questions | cells | ticks | every setting exact, hard and soft, every cell, from random scratch | smallest soft probability of the right answer |
|---|---|---|---|---|---|
| (3,2,1) | 99 | 28 | 9 | yes | 1.000000000 |
| (3,2,2) | 297 | 48 | 18 | yes | 1.000000000 |
| (3,2,3) | 891 | 68 | 27 | yes | 0.999999999 |
| (3,2,4) | 2,673 | 88 | 36 | yes | 0.999999995 |
| (3,3,2) | 2,835 | 60 | 22 | yes | 1.000000000 |
| (3,3,3) | 8,505 | 85 | 33 | yes | 0.999999998 |
| (2,3,3) | 176 | 55 | 18 | yes | 1.000000000 |
| (2,4,4) | 1,472 | 84 | 28 | yes | 1.000000000 |
| (4,2,2) | 1,824 | 64 | 26 | yes | 0.999999999 |
| (4,2,3) | 7,296 | 92 | 39 | yes | 0.999999974 |
| (5,2,2) | 7,250 | 80 | 34 | yes | 0.999999988 |
| (4,3,2) | 30,336 | 80 | 32 | yes | 0.999999998 |
| (5,3,2) | 20,000 (sample) | 100 | 42 | yes | 0.999999969 |
| (3,4,3) | 20,000 (sample) | 102 | 39 | yes | 0.999999998 |
<!-- /COMPILE -->

## 2. The notches

`notches.py`. Each loosens one thing relative to the one above it.

| notch | attention | rule | cell state | what is still compiled |
|---|---|---|---|---|
| 1a `N1_table` | fixed patterns | the SUB table learned (200 entries at base 10, 18 at base 3); copy and select compiled | digits + borrow | everything else |
| 1b `N1_tables` | fixed | all three rules learned tables | digits + borrow | wiring, kinds, format |
| 1c `N1_mlp` | fixed | one ReLU MLP for all three kinds (54 hidden units at B = 3) | digits + borrow | wiring, kinds, format |
| 1d `N1_mlp_sub` | fixed | the SUB rule a ReLU MLP on the ports; copy and select compiled | digits + borrow | wiring, kinds, format, plumbing rules |
| 1e `N1_mlptab_sub` | fixed | the SUB table the outputs of a ReLU MLP on every one-hot input (the H1 form, §9) | digits + borrow | as 1d |
| 2a `N2_narrow` | per head and kind, a soft mixture over 1-3 semantic options (TINYROUTING-like switches; the carry route fixed) | SUB table | digits + borrow | the menus, kinds, format |
| 2b `N2_menu` | per head and kind, a soft mixture over 7 relative rules: up, up 2, left, up-left, N's row, the top place of the row above, a zero | SUB table | digits + borrow | the menu, kinds, format |
| 2c `N2_rel` | the 7 rules plus up to 13 more relative offsets (rows −3..0, columns −2..2) | tables | digits + borrow | the menu, kinds, format |
| 3 `N3_free` | free: softmax over every cell of a learned 2-D relative bias per (query kind, key kind, row offset, column offset) plus a top-place term; a 2-D causal (raster) mask | SUB table, or one ReLU MLP | digits + borrow | kinds, format |
| 4 `N4_symbols` | free | one ReLU MLP over whole port states | K free symbols; learned input embedding and readout; value projections | kinds |
| 4' `N4_symbols_untied` | free, one block per tick | as 4 | as 4 | kinds |
| 5 `N5_free` | free | a residual LayerNorm block | continuous vectors (d = 16) | kinds |
| 6 `N6_plain` | content QK attention with learned absolute row and column embeddings (causal) | residual block | continuous (d = 32) | nothing but the token count, the tick count and the answer's place |

"Given" variants start the rules at the truth and learn only the wiring.

## 3. The ladder: every notch trained from random

`suites.py` (`ladder`, `mlpcell`, `softcert`, `plain`), records in `runs/<suite>.jsonl`, 4 seeds each (2 in
`plain`). Sizes are (3, 2, 1), 99 questions and 28 cells, and (3, 2, 2), 297 questions and 48 cells, plus (4, 2, 1)
and (5, 2, 1) in `mlpcell`.

Training uses every question of the size, full batch, Adam:
- table and menu notches: lr 0.05, float64, at most 3,000 steps;
- MLP and free-format notches: lr 0.01, float32, batches of 64 and at most 2,000 steps in `ladder`; full batch,
  float64 and 3,000 steps in `mlpcell`.

A run stops 100 steps after its hard network first gets every training answer right. It is **certified** if that
network is also exact on every question of two bigger sizes, (B, w, k + 1) and (B, w + 1, k + 1) (up to 20,000
sampled). That is TINYROUTING.md's test: one more quotient digit, and one more digit of N.

Cells read "certified/runs (median step at which the training answers first all came out right)". "fit" counts runs
exact on training but wrong on a bigger size, and the bracket is the median training share of the runs that did not
certify.

Lesson tiers (`notches.py` TIERS):

| tier | lessons |
|---|---|
| A | the answers |
| G | + x mod N = x (each question's own register start R0 < N, and N itself) + the step law |
| AC | the answers + the cell's lessons: x < N on top parts (pref), the column and row order laws, bijectivity |
| GC | G + the cell's lessons |
| AL / GL | + the one-link law (a sub-grid) |
| ACL / GCL | + the cell's lessons + the one-link law |
| +f | the attention frozen for the first 300 steps (tables first) |
| +s | the grid grown one link at a time |

The `ladder` suite was launched before bijectivity joined the cell's lessons. Its AC/GC/ACL/GCL rows are without
bijectivity, and the `mlpcell` rows are with it (each record keeps its lesson weights).

<!-- LADDER -->
| notch | lessons | (3,2,1) | (3,2,2) |
|---|---|---|---|
| 1a. the subtraction table learned | A | 0/4 fit2 [0.77] | 0/4 fit2 [0.67] |
| 1a. the subtraction table learned | AC | 4/4 (37) | 2/4 (37) [0.19] |
| 1a. the subtraction table learned | ACL | 4/4 (37) | 4/4 (37) |
| 1a. the subtraction table learned | G | 0/4 fit2 [0.76] | 0/4 fit2 [0.68] |
| 1a. the subtraction table learned | GCL | 4/4 (50) | 4/4 (75) |
| 1b. all three rules learned as tables | A | 0/4 [0.50] | 0/4 [0.30] |
| 1b. all three rules learned as tables | ACL | 0/4 [0.30] | 0/4 [0.26] |
| 1b. all three rules learned as tables | G | 0/4 [0.42] | 0/4 [0.30] |
| 1b. all three rules learned as tables | GCL | 2/4 (837) [0.27] | 2/4 (350) [0.53] |
| 1c. all three rules one ReLU MLP | A | 0/4 [0.31] | 0/4 [0.21] |
| 1c. all three rules one ReLU MLP | ACL | 0/4 [0.28] | 0/4 [0.22] |
| 1c. all three rules one ReLU MLP | G | 0/4 [0.27] | 0/4 [0.24] |
| 1c. all three rules one ReLU MLP | GCL | 0/4 [0.37] | 0/4 [0.28] |
| 1c'. the same, straight-through states | GCL | 0/4 [0.18] | 0/4 [0.15] |
| 2a. narrow menu (2-3 options), tables given | A | 4/4 (37) | 4/4 (75) |
| 2a. narrow menu, table learned | A | 0/4 [0.44] | 0/4 [0.28] |
| 2a. narrow menu, table learned | AC | 3/4 (100) [0.42] | 1/4 (75) [0.17] |
| 2a. narrow menu, table learned | ACL | 4/4 (87) | 4/4 (87) |
| 2b. wide menu (7 options), tables given | A | 0/4 [0.39] | 0/4 [0.30] |
| 2b. wide menu (7 options), tables given | AL | 4/4 (100) | 4/4 (112) |
| 2b. wide menu, table learned | A | 0/4 [0.21] | 0/4 [0.24] |
| 2b. wide menu, table learned | AC | 0/4 [0.42] | 0/4 [0.31] |
| 2b. wide menu, table learned | ACL | 0/4 [0.33] | 1/4 (775) [0.21] |
| 2b. wide menu, table learned | ACL+f | 4/4 (425) | 4/4 (425) |
| 3. free attention, tables given | A | 0/4 [0.47] | 0/4 [0.30] |
| 3. free attention, tables given | AL | 0/4 [0.44] | 0/4 [0.18] |
| 3. free attention, tables given | AL+s | 0/4 [0.31] | 0/4 [0.18] |
| 3. free attention, table learned | A | 0/4 [0.49] | 0/4 [0.18] |
| 3. free attention, table learned | AC | 0/4 [0.39] | 0/4 [0.22] |
| 3. free attention, table learned | ACL | 0/4 [0.42] | 0/4 [0.19] |
| 3. free attention, table learned | ACL+f | 0/4 [0.48] | 0/4 [0.18] |
| 3. free attention, one ReLU MLP | A | 0/4 [0.26] | 0/4 [0.18] |
| 3. free attention, one ReLU MLP | ACL+f | 0/4 [0.18] | 0/4 [0.15] |
| 3. free attention, one ReLU MLP | GCL+f | 0/4 [0.17] | 0/4 [0.13] |
| 4. free symbols (format learned) | A | 0/4 [0.19] | 0/4 [0.18] |
| 4. free symbols (format learned) | G | 0/4 [0.32] | 0/4 [0.17] |
| 4. free symbols (format learned) | GL | 0/4 [0.22] | 0/4 [0.17] |
| 4. free symbols, untied blocks | A | 0/4 [0.28] | 0/4 [0.17] |
| 4. free symbols, untied blocks | G | 0/4 [0.28] | 0/4 [0.18] |
| 5. continuous states (residual block) | A | 0/4 [0.30] | 0/4 [0.17] |
| 5. continuous states (residual block) | G | 0/4 [0.31] | 0/4 [0.16] |
| 5. continuous states (residual block) | GL | 0/4 [0.28] | 0/4 [0.16] |
| 6. plain looped transformer | A | 0/4 [0.20] | 0/4 [0.20] |
| 6. plain looped transformer | G | 0/4 [0.22] | 0/4 [0.18] |
<!-- /LADDER -->

**The loose notches on their soft path** (`softcert`: (3, 2, 1), 4 seeds, full batch, the network as trained, with
softmax attention and unsnapped states, certified on the same two bigger sizes). The ladder judges every notch on
its snapped, hard path. For notches 3 to 6 the soft path is the network itself, so this is the fairer test of them:

<!-- SOFTCERT -->
| notch | lessons | runs | fit training, snapped | fit training, soft | median soft share at (3, 2, 2) | at (3, 3, 2) |
|---|---|---|---|---|---|---|
| 3. free attention, one ReLU MLP | A | 4 | 0 | 4 | 0.20 | 0.06 |
| 4. free symbols, untied blocks | A | 4 | 0 | 0 | 0.22 | 0.07 |
| 4. free symbols (format learned) | A | 4 | 0 | 0 | 0.19 | 0.06 |
| 4. free symbols (format learned) | G | 4 | 0 | 0 | 0.21 | 0.06 |
| 5. continuous states (residual block) | A | 4 | 0 | 4 | 0.25 | 0.07 |
| 5. continuous states (residual block) | G | 4 | 0 | 4 | 0.23 | 0.07 |
| 6. plain looped transformer | A | 4 | 0 | 4 | 0.21 | 0.05 |
| always answer 0 | | | | | 0.18 | 0.06 |
<!-- /SOFTCERT -->

Several generic notches do fit the 99 training questions on their soft path: free attention with an MLP rule, and
continuous states in every run. One size up they are right about as often as a network that always answers 0. They
memorized the training questions, which is what the leaderboard's generic networks do at full size.

**The MLP digit cell, at bases 3-5** (`mlpcell`). AC and GC include bijectivity; ACnb is AC without it:

<!-- MLPCELL -->
| notch | lessons | (3,2,1) | (3,2,2) | (4,2,1) | (5,2,1) |
|---|---|---|---|---|---|
| 1a. the subtraction table learned | AC | 4/4 (50) | 2/4 (37) [0.17] | 4/4 (100) | 0/4 [0.24] |
| 1a. the subtraction table learned | GC | 4/4 (50) | 4/4 (87) | 4/4 (137) | 1/4 (100) [0.34] |
| 1c. all three rules one ReLU MLP | GCL | 1/4 (450) [0.41] | 1/4 (1750) [0.24] | 1/4 (325) [0.23] | 1/4 (1500) [0.08] |
| 1d. only the subtraction cell a ReLU MLP | AC | 4/4 (112) | 4/4 (125) | 4/4 (112) | 4/4 (112) |
| 1d. only the subtraction cell a ReLU MLP | ACnb | 2/4 (162) [0.86] | 2/4 (125) [0.85] | 3/4 (150) [0.97] | 4/4 (137) |
| 1d. only the subtraction cell a ReLU MLP | GC | 4/4 (100) | 4/4 (150) | 4/4 (112) | 3/4 (100) [0.07] |
| 1d'. the same, straight-through states | GC | 1/4 (1000) [0.52] | 0/4 [0.38] | 0/4 [0.44] | 0/4 [0.25] |
| 1e, 256 hidden units | AC | 3/4 (75) [0.22] | 2/4 (75) [0.19] | 3/4 (75) [0.14] | 0/4 [0.34] |
| 1e, 256 hidden units | ACnb | 1/4 (75) [0.92] | 1/4 (75) [0.22] | 0/4 [0.46] | 1/4 (125) [0.34] |
| 1e, 256 hidden units | GC | 4/4 (75) | 3/4 (75) [0.75] | 4/4 (87) | 0/4 [0.34] |
| 2b. wide menu, table learned | ACL+f | 4/4 (412) | 4/4 (412) | 4/4 (587) | 0/4 [0.22] |
| 2b. wide menu, the table an MLP's outputs | ACL+f | 0/4 [0.47] | 0/4 [0.35] | 0/4 [0.27] | 0/4 [0.20] |
<!-- /MLPCELL -->

What the table says:
- **The direct MLP cell (1d)** certifies in 31 of 32 runs with the cell's lessons, bases 3 to 5. The one failure, at
  (5, 2, 1), never started: 12 of its 50 digit entries right when it stopped at step 1,525. Without bijectivity (ACnb) it is 11 of
  16; the stopped seeds have tied entries.
- **The H1 form (1e, the table an MLP's outputs, 256 hidden units)** certifies in 19 of 24 runs at bases 3 and 4. At
  (5, 2, 1) it stalls in every seed, like the learned table, on one self-consistent wrong entry (§6). It certified
  at H1 (§9).
- **The learned table (1a)** certifies in 22 of 24 runs at bases 3 and 4, and 1 of 8 at (5, 2, 1).
- **Straight-through snapping** of the MLP cell's states (1d') certifies in 1 of 16. **One MLP for all three rules**
  (1c, with every lesson and bijectivity) certifies in 4 of 16.
- **The wide menu with the MLP-generated table** fails, although each half learns alone. The wiring comes out right,
  but the table stalls at 10-15 of its 18 digits. While the attention is frozen it is still a uniform blend, and the
  one-link law's and the answers' gradients reach the shared MLP through that blur. A separate table's entries shrug
  this off (2b, learned table: 12 of 12 at bases 3 and 4); one MLP's shared weights do not. At base 5 even the
  learned table fails here, on the stuck entry of §6.
  - Holding the one-link law back until the attention joins at step 300 makes 3 of 4 runs certify
    (`runs/gate_local.jsonl`, (3, 2, 1)). That is "tables first" applied to the lessons, not only to the weights.
  - Holding the answers back too changes nothing. The fourth seed's table stalls under the cell's own lessons, as
    one seed of 1e does with fixed attention.

## 4. The menu's width

The wide menu with the tables given, taught by the answers and the one-link law (AL), with extra relative offsets
added to every port's menu (`runs/menu_width_local.jsonl`, (3, 2, 1), 4 seeds, 2,000 steps):

<!-- WIDTH -->
| options per port | certified | median steps | training share of the others |
|---|---|---|---|
| 7 | 4/4 | 100 | · |
| 10 | 4/4 | 175 | · |
| 13 | 0/4 | · | 0.25 |
| 16 | 0/4 | · | 0.19 |
| 20 | 0/4 | · | 0.24 |
<!-- /WIDTH -->

The edge is sharp: 10 options per port is learned by every seed, 13 by none. Free attention offers a cell about 15
to 25 sources, past the edge.

**Search, not identification.** `census.py` runs every wiring of a menu as the hard network with the truth tables.
Each of the seven ports a kind uses takes any option, at (3, 2, 1) (record `records/census_<options>.json`):

<!-- CENSUS -->
| options per port | wirings | fit every training answer | pass the one-link law | both | both and certify | the true wiring among them |
|---|---|---|---|---|---|---|
| 7 | 823,543 | 1 | 4 | 1 | 1 | yes |
| 10 | 10,000,000 | 1 | 11 | 1 | 1 | yes |
| 13 | 62,748,517 | 1 | 158 | 1 | 1 | yes |
| 20 | 1,280,000,000 | 1 | 1,573 | 1 | 1 | yes |
<!-- /CENSUS -->

- **The answers alone single out the wiring, at every width tested.** Of the 823,543 wirings of the 7-option menu,
  exactly one gets every training answer right: the true one, and it certifies. The same holds for the 10 million
  of the 10-option menu, the 62.7 million of the 13-option menu and the 1.28 billion of the 20-option menu, where
  every soft run failed even with the one-link law. So the soft network's failures (`menu_answers`, §6, and the width curve above) are failures to find
  the wiring, not a lack of information. That matches MENUS.md §3 for the H1 menus.
- **The one-link law pins six of the seven ports by itself.** At 7 options, four wirings pass it. They are the true
  one and three that differ only in where the select reads its new value: the subtraction's own cell, the cell to
  its upper left, the top place, or a zero. On a single link the only number that is actually reduced is N itself,
  whose result is all zeros, so any source of zeros passes. At 10 options 11 wirings pass it, at 13 options 158 and
  at 20 options 1,573. With the answers added, only the true wiring is left at every width.
- So the local law turns the search into a nearly direct lesson. Each port it pins gets its own unscrambled check,
  which is why "the answers + the one-link law" learns the 7- and 10-option menus while the answers alone do not.
- **Whole-option search is the known fix, and it does work.** The census is that search done by brute force: it
  scores every whole wiring pass or fail and keeps the one that passes, WIRING.md's way. Its cost grows as
  (options)^7: 50 s on a laptop at 7 options and 9.5 minutes at 10. The 13-option census (62.7 million wirings)
  took 8,920 container-seconds on 12 Modal CPU containers, 16.5 minutes of wall time. The 20-option one (1.28
  billion), with every wiring checked on 16 questions first and on all 99 only if those pass, took 45 minutes on 16
  containers.

**Whole-option search at free attention's width, block by block** (`blocksearch.py`, record
`records/blocksearch.json`). The menu is MENU's 7 rules plus every other relative offset with row −4..0 and column
−3..3 that the causal mask allows: 34 options per connection, 34^7 = 5.3·10^10 wirings, which is about the number of
sources free attention offers a cell. Brute force would take about 100 CPU-hours at this size. Instead the search
enumerates one block of interacting connections at a time, each scored by a lesson that reads only that block:
1. **The shift and the subtraction** (34^4 = 1,336,336 blocks): on a grid of one link, the subtraction must borrow
   exactly when x < N. 114 blocks pass. The select's connections do not reach the subtraction row, so they do not
   matter here.
2. **The select**, for each of those 114 blocks (34^3 = 39,304 each, 4.5 million in all): 4,391 pass the one-link
   law. Exactly 1 of them also gets every training answer right: the true wiring, and it certifies.

The whole search took 45 s on a laptop. So the known fix works at free attention's width, and blocks plus local
lessons make it affordable. What fails there is only the soft gradient (§3, §6).

## 5. Which lessons each notch can express

| lesson | what it needs | 1a-1e | 2 | 3 | 4-5 | 6 |
|---|---|---|---|---|---|---|
| the answers | the prompt and the answer | yes | yes | yes | yes | yes |
| x mod N = x, N mod N = 0 (network level) | a prompt built from the data's own numbers | yes | yes | yes | yes | yes |
| the step law, f(v) = f(B · f(v // B) + v₀) | the network's own answer fed back as a prompt | yes | yes | yes | yes | yes |
| the one-link law (a sub-grid) | the grid's kinds, to build a grid of one link; reading that link's row | yes | yes | yes | yes, through the learned readout | no: there are no kinds to build a link from |
| the cell's lessons (top-part order facts, order laws, bijectivity) | a recognizable subtraction cell with a known port format: query it on made-up ports | yes | yes | yes, bypassing the attention | no: the cell's format is learned | no |
| subtract-then-add-back, the times-table ladder | as above, plus an adder or a product cell to compose with | not used: the tiny division machine has neither (at H1 they are cell lessons like the order facts) | as 1 | as 1 | no | no |

The cell's lessons make the tables learnable, and they stop being expressible at notch 4, where the meanings of
the digits inside the network are learned. The one-link law makes the wiring learnable, and it survives to notch 5
through the learned readout, but not to notch 6, which has no kinds to build a link from. Notch 6 therefore has only
the network-level lessons. From those alone, not even the fixed-wiring notch 1a certifies (tier G: 0 of 8; it fits
the training answers in half the runs and misses the bigger sizes).

## 6. Diagnosis: the compiled machine as a probe

`probe.py`, `diagnose.py`, record `records/diagnose.json`; (3, 2, 1), seeds 0 and 1. The probes:
- **heads**: every head's argmax source against the compiled one. "input" means the cell reads a prompt cell the
  machine does not (a bypass), "computed" another computed cell, "zero" a constant;
- **grad0**: which key each head's answer gradient pushes up most, at the start;
- **cells**: every row's digits against the machine's own;
- **path**: the answer loss on the straight line from the trained weights to the compiled ones;
- **relabel**: for the free-symbol format, whether each cell holds the machine's digit under a consistent renaming.

<!-- DIAG -->
| case | seed | certified | training, hard / soft | ports right | cells reading an input | first gradient toward an input | loss on the line to the compiled weights | rules right (SUB digits, copy, select) |
|---|---|---|---|---|---|---|---|---|
| menu_answers | 0 | no | 0.42 / 0.42 | 4/7 | 0.00 | 0.60 | 0.80 -> 2.63 -> 0.00 | 18, 3/3, 18/18 |
| menu_answers | 1 | no | 0.42 / 0.42 | 4/7 | 0.10 | 0.60 | 0.80 -> 2.53 -> 0.00 | 18, 3/3, 18/18 |
| menu13_link | 0 | no | 0.51 / 0.40 | 3/7 | 0.24 | 0.57 | 0.88 -> 1.40 -> 0.00 | 18, 3/3, 18/18 |
| menu13_link | 1 | no | 0.33 / 0.20 | 3/7 | 0.05 | 0.57 | 0.98 -> 1.08 -> 0.00 | 18, 3/3, 18/18 |
| free_link | 0 | no | 0.44 / 0.40 | 0/7 | 0.36 | 0.57 | 0.71 -> 1.06 -> 0.00 | 18, 3/3, 18/18 |
| free_link | 1 | no | 0.44 / 0.45 | 0/7 | 0.38 | 0.57 | 0.74 -> 1.04 -> 0.00 | 18, 3/3, 18/18 |
| free_link_staged | 0 | no | 0.25 / 0.38 | 0/7 | 0.40 | 0.57 | 1.00 -> 1.31 -> 0.00 | 18, 3/3, 18/18 |
| free_link_staged | 1 | no | 0.40 / 0.48 | 0/7 | 0.48 | 0.57 | 0.85 -> 1.22 -> 0.00 | 18, 3/3, 18/18 |
| mlp_all | 0 | no | 0.12 / 0.18 | · | · | · | · | 7, 1/3, 3/18 |
| mlp_all | 1 | no | 0.84 / 1.00 | · | · | · | · | 17, 3/3, 18/18 |
| mlp_sub_nobij | 0 | yes | 1.00 / 1.00 | · | · | · | · | 18, 3/3, 18/18 |
| mlp_sub_nobij | 1 | yes | 1.00 / 1.00 | · | · | · | · | 18, 3/3, 18/18 |
| tables_all | 0 | no | 0.22 / 0.29 | · | · | · | · | 5, 1/3, 10/18 |
| tables_all | 1 | no | 0.21 / 0.34 | · | · | · | · | 16, 0/3, 7/18 |
| free_rows | 0 | no | 0.38 / 0.42 | 2/7 | 0.10 | 0.26 | 0.76 -> 0.98 -> 0.00 | 18, 3/3, 18/18 |
| free_rows | 1 | no | 0.43 / 0.36 | 1/7 | 0.05 | 0.26 | 0.72 -> 1.31 -> 0.00 | 18, 3/3, 18/18 |
| free_lr01 | 0 | no | 0.54 / 0.42 | 0/7 | 0.38 | 0.57 | 0.66 -> 1.25 -> 0.00 | 18, 3/3, 18/18 |
| free_lr01 | 1 | no | 0.44 / 0.45 | 0/7 | 0.43 | 0.57 | 0.68 -> 1.07 -> 0.00 | 18, 3/3, 18/18 |
| free_lr003 | 0 | no | 0.37 / 0.40 | 0/7 | 0.38 | 0.57 | 0.74 -> 0.90 -> 0.00 | 18, 3/3, 18/18 |
| free_lr003 | 1 | no | 0.43 / 0.41 | 0/7 | 0.29 | 0.57 | 0.78 -> 0.91 -> 0.00 | 18, 3/3, 18/18 |
| menu_pmix | 0 | no | 0.36 / 0.35 | 3/7 | 0.05 | 0.60 | 0.85 -> 1.00 -> 0.00 | 18, 3/3, 18/18 |
| menu_pmix | 1 | no | 0.36 / 0.35 | 3/7 | 0.05 | 0.60 | 0.85 -> 0.99 -> 0.00 | 18, 3/3, 18/18 |
| symbols | 0 | no | 0.26 / 0.45 | 0/7 | 0.55 | 0.24 | · |  |
| symbols | 1 | no | 0.21 / 0.44 | 0/7 | 0.60 | 0.67 | · |  |
| free_first | 0 | no | 0.09 / 0.35 | 1/7 | 0.00 | 0.00 | 0.92 -> 2.34 -> 0.00 | 18, 3/3, 18/18 |
| free_first | 1 | no | 0.22 / 0.31 | 1/7 | 0.02 | 0.00 | 0.88 -> 1.19 -> 0.00 | 18, 3/3, 18/18 |
| free_st | 0 | no | 0.18 / 0.18 | 0/7 | 0.05 | 0.00 | 14.65 -> 14.65 -> 0.00 | 18, 3/3, 18/18 |
| free_st | 1 | no | 0.18 / 0.18 | 0/7 | 0.26 | 0.00 | 14.65 -> 14.65 -> 0.00 | 18, 3/3, 18/18 |
<!-- /DIAG -->

What the probes say, trap by trap (the project's named traps, [OVERVIEW.md](../../OVERVIEW.md) §3):

- **Scrambled credit, leading to a bypass** (free attention).
  - At the start, the answers' gradient points most ports at input cells: the minuend's port on every cell, and the
    gate's old and new ports on 5 of 6.
  - Training ends there. No port comes out right everywhere, and about 37% of the cells' sources are input cells
    (0.36 and 0.38 on the two seeds). On seed 1, for example, the shift reads the wrong input digits, and the
    subtractor's minuend reads the input register itself instead of the shifted register.
  - The straight line to the compiled weights rises before it falls (0.71 → 1.06 → 0 nats per digit on seed 0), so
    this is a separate valley, not a slow road.
  - The known fixes:
    - the local check (the one-link law) does not help, because a bypass satisfies "x mod N = x";
    - one link at a time does not help: even the one-link stage fails;
    - a 2-row locality window gets the shift, the gate's old port and the carries right, but the minuend still
      reads the clean input;
    - lr 0.01 and 0.003 do not help;
    - whole options per port (each option tried hard, the other ports soft; `menu_pmix`, tried on the 7-option menu)
      land on the same family of near misses as the soft blend: 0.36 of answers on both seeds.

  The fix that works is joint enumeration of interacting ports, scored pass or fail. By brute force (the census of
  §4) it finds the one wiring that passes at 13 options. Block by block with local lessons (`blocksearch.py`, §4) it
  does the same at free attention's width, 34 options per connection, in 45 seconds.
- **Impostor wirings** (the 7-option menu).
  - From the answers, both seeds land on one near miss (42% of answers): the subtrahend reads the top place of the
    row above instead of N's row, the gate reads the borrow of the cell below-left instead of the row's final borrow,
    and the carry route stays near uniform.
  - The census (§4) says the answers alone allow only the true wiring, so the soft blend misses a wiring the answers
    do identify.
  - The one-link law, run through the network's own wiring on a grid of one link, rules it out: 8/8 with the tables
    given (`ladder`, AL).
  - With the tables learned at the same time, the wiring locks in while the table is still wrong (TINYROUTING.md
    §2.1's co-adaptation). Freezing the attention for 300 steps, tables first, fixes it: 4/4 at both sizes.
- **Impostor tables** (the MLP cell). The MLP settles with two entries of a column giving the same digit. The order
  laws are then unsatisfiable and their gradient stalls. Bijectivity, the anchors' third lesson, breaks the tie
  (`mlpcell`, 1d with and without bijectivity, §3).
- **A self-consistent wrong entry** (both table forms at base 5). At (5, 2, 1), most runs of the learned table and of
  the MLP-generated table stop with exactly one entry wrong: (1, 0, 1), computed as 4 with a borrow (as if
  1 − 0 − 1 went below zero) instead of 0 without one.
  - The order laws take the learned borrows as their referee, and under a wrong borrow the wrong digit is exactly
    what they ask for, so the pair holds itself up.
  - The top-part fact "N − R0 never borrows" contradicts it, but it is outvoted: the stuck runs end with every
    lesson's loss the same to three digits.
  - The direct MLP cell (1d), which reads blended digits, never stops there.
  - At H1 the MLP-generated table certified on every run, the H100 one and 8 of 8 CPU seeds (§9). So this is a
    small-data trap of the tiny core: few moduli, one quotient digit.
- **Analog codes** (one MLP for all rules). The soft network is right where the snapped one is not: seed 1 has 1.00
  of training answers on the soft path and 0.84 on the hard one. Intermediate digits are passed on blurred, and the
  next MLP reads the blur. Straight-through snapping, the generic form of "digits snapped to exact values", makes it
  worse (0 of 16).
- **Identical copies.** These did not appear as the binding trap. The kinds give every cell a fixed meaning, which is
  the project's fix for interchangeable parts. Without kinds (notch 6), the network has learned absolute positions
  instead, and it fails earlier, from the search.

## 7. Over-parameterized compiles: the Net2Net arm

**The idea** (from the user's handwired project). Exact transcribed weights may be too compressed to move, and a
redundant version of the same program may train where the compressed one will not. So next to every minimal compile
there is a function-preserving widened one, in the style of Net2Net (Chen, Goodfellow and Shlens, 2016). Each
widened compile is checked exact on every question at (3, 2, 1), (3, 2, 2) and (4, 2, 2), on both paths and from
random scratch states:
- **hidden units duplicated** (`widen`): every AND-detector unit of the MLP rule copied 4 times, its outgoing weights
  divided by 4 (`compile.set_mlp`);
- **routing heads in parallel copies** (`copies`): 4 copies of every head, a port reading their mean (`copy_mode`
  "mean"), or, for MLP rules, all of them side by side with the input weights split ("concat");
- **digits embedded in a larger alphabet** (`compile.set_symbols`): the free-symbol format with K = 24 symbols instead
  of 6, each (digit, borrow) meaning spread evenly over 4 symbols.

Gaussian noise added after compiling lets the copies diverge. Two questions at each notch with a compiled point,
minimal against widened, at (3, 2, 1), 4 seeds:

**1. Stability** (`wstab`). Start at the compiled machine with every scale softened to 4 (attention scores, output
logits; compiled, they are 20-30, where gradients vanish). Add noise of std σ to every learned tensor, train 1,000
steps with the notch's full lessons, or with the answers alone, and see whether the solution is kept, recovered,
or lost. A cell reads "certified at the end / runs (runs exact before training)":

<!-- WSTAB -->
| notch | compiled as | lessons | σ = 0.1 | σ = 0.3 | σ = 1 | σ = 3 |
|---|---|---|---|---|---|---|
| 1c. all three rules one ReLU MLP | minimal | GCL | 4/4 (2) | 2/4 (0) | 0/4 (0) | 0/4 (0) |
| 1c. all three rules one ReLU MLP | widened | GCL | 4/4 (4) | 4/4 (0) | 0/4 (0) | 0/4 (0) |
| 1c. all three rules one ReLU MLP | minimal | the answers | 4/4 (2) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 1c. all three rules one ReLU MLP | widened | the answers | 4/4 (4) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 2b. wide menu (7 options), tables given | minimal | AL | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (0) |
| 2b. wide menu (7 options), tables given | widened | AL | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (0) |
| 2b. wide menu (7 options), tables given | minimal | the answers | 4/4 (4) | 4/4 (4) | 4/4 (4) | 1/4 (0) |
| 2b. wide menu (7 options), tables given | widened | the answers | 4/4 (4) | 4/4 (4) | 4/4 (4) | 4/4 (0) |
| 3. free attention, tables given | minimal | AL | 4/4 (4) | 4/4 (4) | 3/4 (2) | 0/4 (0) |
| 3. free attention, tables given | widened | AL | 4/4 (4) | 4/4 (4) | 3/4 (4) | 0/4 (0) |
| 3. free attention, tables given | minimal | the answers | 4/4 (4) | 4/4 (4) | 3/4 (2) | 0/4 (0) |
| 3. free attention, tables given | widened | the answers | 4/4 (4) | 4/4 (4) | 3/4 (4) | 0/4 (0) |
| 3. free attention, one ReLU MLP | minimal | ACL | 4/4 (3) | 3/4 (0) | 0/4 (0) | 0/4 (0) |
| 3. free attention, one ReLU MLP | widened | ACL | 4/4 (1) | 1/4 (0) | 0/4 (0) | 0/4 (0) |
| 3. free attention, one ReLU MLP | minimal | the answers | 0/4 (3) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 3. free attention, one ReLU MLP | widened | the answers | 1/4 (1) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 4. free symbols (format learned) | minimal | GL | 2/4 (3) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 4. free symbols (format learned) | widened | GL | 0/4 (3) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 4. free symbols (format learned) | minimal | the answers | 3/4 (3) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
| 4. free symbols (format learned) | widened | the answers | 0/4 (3) | 0/4 (0) | 0/4 (0) | 0/4 (0) |
<!-- /WSTAB -->

**2. Learnability from random** (`wlearn`): the same widened architectures, trained from a random start with the
same lessons as the minimal ones:

<!-- WLEARN -->
| notch | lessons | (3,2,1) | (3,2,2) |
|---|---|---|---|
| 1c. all three rules one ReLU MLP | GCL min | 0/4 [0.58] | 0/4 [0.34] |
| 1c. all three rules one ReLU MLP | GCL wide | 1/4 (1075) [0.21] | 0/4 [0.24] |
| 3. free attention, tables given | AL min | 0/4 [0.44] | 0/4 [0.18] |
| 3. free attention, tables given | AL wide | 0/4 [0.39] | 0/4 [0.19] |
| 3. free attention, one ReLU MLP | ACL+f min | 0/4 [0.20] | · |
| 3. free attention, one ReLU MLP | ACL+f wide | 0/4 [0.29] | · |
| 4. free symbols (format learned) | GL min | 0/4 [0.30] | · |
| 4. free symbols (format learned) | GL wide | 0/4 [0.23] | · |
| 2c. 13-option menu, tables given | AL min | 0/4 [0.25] | 0/4 [0.18] |
| 2c. 13-option menu, tables given | AL wide | 0/4 [0.18] | 0/4 [0.19] |
<!-- /WLEARN -->

<!-- NET2NET_READING -->
What the arm says:
- **Near a solution, redundancy widens the basin where the structure is constrained or a local lesson gives a
  signal.**
  - The all-MLP rule with fixed attention recovers from noise σ = 0.3 in 4 of 4 runs when Net2Net-widened, against 2
    of 4 minimal. Both need the cell's lessons: on the answers alone, neither recovers.
  - The 7-option menu with 4 head copies recovers from σ = 3 on the answers alone in 4 of 4 runs, against 1 of 4 for
    a single head.
  - With free attention, 4 copies average the noise: at σ = 1 all 4 copied runs start exact on the hard path,
    against 2 of 4 single-head runs.
- **In the free formats the compiled point is not stable, and redundancy does not save it.** Trained on the lessons
  they can express, starts that are exact drift away:
  - Impostors that fit every training answer but fail a bigger size. Free attention with 4 copies at σ = 1 settles
    on a hard wiring exact on all 99 training questions and 54% one quotient digit up: a wiring outside the
    kind-agnostic menus the census could enumerate. The free-attention MLP and the free symbols drift to near misses
    that fail 1-5% of (3, 3, 2).
  - Analog codes: the free-attention MLP stays at 100% on its soft path while the snapped network falls to 82-84%.
    Its lessons decide which way it goes. At σ = 0.1, with the cell's lessons and the one-link law, all 3 exact
    minimal starts stay exact. On the answers alone, all 3 drift away.
  - The lifted alphabet (4 symbols per meaning) makes this worse. Softened, its soft path spreads over 24 symbols and
    is no better than "always answer 0". Its gradients therefore cannot see the hard solution, and none of its 3
    exact starts survives training (they end at 24-83% on training). The minimal alphabet keeps 1 or 2 of its 3.
- **From random, widening rescues almost nothing.** With 4 head copies and 4× the hidden units, the 13-option menu,
  free attention, the free-attention MLP and the free symbols certify in none of 24 runs, as the minimal ones. The
  all-MLP rule certifies in 1 of 8 widened runs, against 0 of 8 minimal (`wlearn`). That is the caveat
  the idea came with: redundancy helps the optimizer move, but it does not create a signal where the wrap-around
  gives none.

How this squares with the literature the idea came with:
- **Mode connectivity** (Garipov et al., 2018; Draxler et al., 2018). Over-parameterized networks have connected
  valleys of solutions. Here widening widens the valley around the compiled solution in the constrained notches,
  but the trained free-attention networks still sit in separate valleys (the probe's straight line rises first, §6).
- **Lottery tickets** (Frankle and Carbin, 2019). Head copies from random do not let one lucky copy find the right
  source and win. A port reads the mean of its copies, so at the start every copy gets the same push: on two seeds,
  all 4 copies of each port were pushed toward a prompt cell on exactly the same share of cells (for the minuend,
  1.00 on all four).
- **Over-parameterization as acceleration** (Arora et al., 2018) shows up only as faster or more frequent
  recovery near the solution, never as learning from scratch.
- **Lazy (NTK) training** predicts that wide networks stay near their start. The constrained notches do that; the
  free ones drift, because soft gradients reward analog or impostor solutions as much as the compiled one.
<!-- /NET2NET_READING -->

## 8. Two more fixes for the bypass

Two fixes from the user's looped-transformer Sudoku project, sotaku (`STALE_EXPERIMENTS_DOC.md`, "No X After Init";
the loop feeds its own predictions back), tried against free attention's bypass (§6), with the same lessons as
before (`bypass`, (3, 2, 1) and (3, 2, 2), 4 seeds):
- **The prompt only at the first tick** (`inject` "first", TRM-style; on Sudoku this cost 0.7 points). Here the
  network otherwise re-injects the prompt after every tick, so the prompt's cells stay clean while every computed
  cell is still noise, and that is what the first gradient points the connections at. With the prompt only in the
  initial state, the prompt's cells are updated like any cell. They hold their digits only if their own connection
  learns to read themselves, and otherwise they blur after the first tick like everything else. The compile with
  self-reading prompt cells is exact (§1).
- **Snapped digits fed back** (`st`): every cell is snapped to its argmax digit on the forward path, with
  straight-through gradients. A computed cell that is right then looks as clean as the prompt.
- **Both together**, and the first-tick prompt with 4 head copies.

<!-- BYPASS -->
| notch | lessons | (3,2,1) | (3,2,2) |
|---|---|---|---|
| 3. free attention, tables given, snapped states | AL | 0/4 [0.18] | 0/4 [0.18] |
| 3. free attention, tables given, prompt at tick 0 only | A | 0/4 [0.36] | 0/4 [0.16] |
| 3. free attention, tables given, prompt at tick 0 only | AL | 0/4 [0.16] | 0/4 [0.17] |
| 3. free attention, tables given, prompt at tick 0 only | AL c4 | 0/4 [0.24] | 0/4 [0.18] |
| 3. the same, snapped states | AL | 0/4 [0.18] | 0/4 [0.18] |
| 3. free attention, table learned, prompt at tick 0 only | ACL+f | 0/4 [0.38] | 0/4 [0.18] |
<!-- /BYPASS -->

<!-- BYPASS_READING -->
What the probe says (`diagnose.py` cases `free_first` and `free_st`, (3, 2, 1), 2 seeds):
- **Both fixes remove the bypass's pull.** At the start, the share of cells whose answer gradient points at a prompt
  cell drops from 0.57 (plain free attention) to 0.00. By the end, almost no cell reads the prompt: 0.00-0.02 of
  cells with the first-tick prompt, against 0.36-0.38 before.
- **Neither learns, because the search fails in a new way.**
  - The connections settle on constant zeros instead: 19-29% of cells' sources with the first-tick prompt, 36-40%
    with snapped states. That is the do-nothing attractor.
  - With the first-tick prompt, the prompt cells do not learn to keep themselves either: only 23-31% read themselves,
    and 61-73% of the prompt's digits survive to the last tick. Keeping the input alive becomes a second unlearned
    job.
  - Snapped states from a random start snap almost every cell to the same digit, and training stays at the "always
    answer 0" level.
- So the bypass was the first trap but not the only one. With the clean prompt gone, the next-cleanest source wins:
  a constant. The soft search over free attention still finds no right wiring; the whole-option search of §4 does.
<!-- /BYPASS_READING -->

## 9. H1: the MLP digit cells

[`h1/submission_wide_mlp.py`](h1/submission_wide_mlp.py) is the certified
[`submission_wide_anchored.py`](../../wider_menus/submission_wide_anchored.py) (md5 `c82b11fa`, untouched), with
everything else line for line the certified file's:
- **the subtraction table** is now a ReLU MLP: (r, m, borrow in) one-hot (22 inputs) → 256 hidden → 12 logits;
- **the product cell** likewise: (a, b) one-hot (20 inputs) → 256 → 20 logits;
- **what the machine reads** is each MLP's outputs on every one-hot input, so at evaluation, with one-hot digits, the
  machine applies the MLP to its digits;
- **the lessons and their schedule** are the certified file's; the lessons that taught the tables' entries now teach
  the MLPs' weights;
- **the cells' Adam rate** is 0.01, and the rest stays at 0.05;
- **26,275 learned numbers**, 19,488 of them in the cells (the certified file has 11,187).

Against the certified file, the diff replaces 11 lines: the four table tensors, their designed values, the product
cell's Adam reset and the optimizer, which gets a second parameter group. It adds the `CellMLP` class, four
properties that give the machine the tables it reads (`a`, `b`, `p_lo`, `p_hi`), and a paragraph of docstring.

Local plumbing check (`wider_menus/local/local_wide.py`, designed mode, the MLPs compiled to AND detectors): all 14
rungs exact on both prediction paths with identical logits, n = 8 each. The gradient test gives every parameter
tensor a nonzero gradient, the cells' MLPs included.

**The official run** (`h1/records/wide_mlp_official.json`; Modal H100, the official manifest untouched, the file
byte for byte, md5 `22bd34d0`, seed 74):

| what | result |
|---|---|
| max certified T, seen / unseen moduli | **64 / 64**, every rung 768/768 on both ladders |
| test, ood_t, ood_n_t | 9,999/9,999, 10,002/10,002, 10,002/10,002 |
| training s (frozen at step) / evaluation s | 1,047.7 (743) / 98.0 (the certified wide_anchored_official: 558.0 (588) / 41.4) |
| learned numbers | 26,275 (19,488 in the two MLP cells) |
| the subtraction MLP on its 200 inputs | every digit and every borrow the true table's |
| the product MLP on its 100 inputs | every low and high digit the true table's |
| audit | 200/200 subtraction rows and 100/100 product cells used at evaluation, none of them never trained |
| the wiring and layout it chose | long division by trial multiples (a = the step's start, b = the link's multiple of N), the digit added at place 0, the final correction enabled; schoolbook rows by multiplier digit, summed in reverse order (288/288 placements right) |
| schedule | reading chosen at step 158, digit lessons from 229, phase B 297, ladder 331, chain lesson 483, I/O wiring 533, T resolved 564, frozen 743 |

The schedule ran later than the certified file's: there the reading came at step 97, the digits at 134, phase B at
179, T at 409 and the freeze at 588. Over steps 100-300 a step cost 0.55 s here against 0.45 s there, since each MLP
is recomputed wherever a table is read. Over steps 500-700 it cost 1.8 s, in the phase where the I/O wiring is
enumerated. Training took 1,048 s against 558 s, well inside the 3,600 s budget.

Eight of eight CPU seeds certified, each with both MLP cells exact, and every record's source md5 equals the file's.
Seven chose long division by repeated subtraction, and seed 74 chose trial multiples, as it did on the H100. CPU
training took 1,574-3,015 s and evaluation 192-593 s; the certified wide-anchored file's CPU seeds took 1,913-2,361
s and 448-674 s.

<!-- H1CPU -->
**CPU seeds** (the same evaluator with `device = cpu`, the file byte for byte; `h1/plans/wide_mlp_cpu.json`):

| seed | max T, seen / unseen | every rung 768/768 | splits exact | training s (frozen at step) | evaluation s | both MLP cells exact | chain | md5 |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 / 64 | yes | yes | 2470 (700) | 565 | yes | repeated subtraction | 22bd34d0 |
| 2 | 64 / 64 | yes | yes | 2292 (699) | 593 | yes | repeated subtraction | 22bd34d0 |
| 3 | 64 / 64 | yes | yes | 2058 (706) | 192 | yes | repeated subtraction | 22bd34d0 |
| 4 | 64 / 64 | yes | yes | 3015 (876) | 568 | yes | repeated subtraction | 22bd34d0 |
| 5 | 64 / 64 | yes | yes | 1656 (685) | 427 | yes | repeated subtraction | 22bd34d0 |
| 6 | 64 / 64 | yes | yes | 1636 (681) | 518 | yes | repeated subtraction | 22bd34d0 |
| 7 | 64 / 64 | yes | yes | 1574 (743) | 405 | yes | repeated subtraction | 22bd34d0 |
| 74 | 64 / 64 | yes | yes | 2208 (769) | 531 | yes | trial multiples | 22bd34d0 |
<!-- /H1CPU -->

## 10. Caveats

- **Tiny sizes and few seeds:** bases 2-5, 2-digit moduli for training, 4 seeds per cell, one learning rate per
  family. A cell like "0/4" means no seed out of 4, not never.
- **The grid is the compile's.** Even the plain notch 6 gets the compiled number of scratch tokens, the compiled
  number of ticks and the compiled answer place. The free notches keep the 2-D causal mask, which guarantees that a
  feed-forward circuit settles in D ticks. That mask is a generic built-in, the decoder mask in 2-D, but it is a
  built-in.
- **The menus are designed.** The 7 relative rules contain the true pattern of every port. The width experiment adds
  plain offsets, not adversarial ones.
- **Lessons carry design.** The one-link law needs the grid's kinds, to build a grid of one link, and it reads that
  link's register. The cell's lessons query a recognizable subtraction cell. The order laws use the digit symbols'
  order. These are the same lessons the H1 machine uses, phrased on cells and sub-grids.
- **Certification at the tiny size** is TINYROUTING.md's: exact on every question of the training size and of two
  bigger sizes. The plain notch's learned absolute positions cannot run a bigger grid (its bigger sizes read
  untrained position embeddings), so it cannot certify by construction. Its rows read as a failure to fit or to
  transfer, never as identifiability.
- **The loose notches' budgets are small.** The ladder gave notches 3-6 2,000 steps of 64 questions. The `plain`
  suite gave notch 6 5,000 full-batch steps at widths 32 and 64 and two learning rates. At d = 32 and lr 0.001 it
  fits all 99 training questions on its soft path in 2 of 2 seeds, though not once snapped (22% and 27%); the other
  settings do not fit. `softcert` judged notches 3-6 on their soft path too. A larger network or a longer run might
  memorize more; nothing here suggests it would transfer.
- **Whole-option search was tried only per port.** Joint enumeration of a block of interacting ports was not run on
  free attention. At about 10^3-10^4 combinations per block it is affordable at this size, and it is the known fix.
  It is what wider_menus does at H1.
- **The ladder suite predates bijectivity** (§3).
- **The Net2Net arm's stability test is one protocol among many.**
  - The compiled scales are softened to 4 everywhere, and the noise is absolute (the same σ for attention scores
    of scale 4 and for AND-detector weights of scale 1).
  - It trains 1,000 steps at (3, 2, 1). The MLP notches use float32.
  - Only the notches with a compiled point in their format are tested. Continuous states and the plain transformer
    have no compile here.
  - The widened twins use 4 copies throughout.
- **The block-wise search's menu is kind-agnostic.** Its 34 relative offsets cover the positions free attention can
  reach in a 5 × 7 window, plus N's row and the top place. Free attention can also prefer a key kind, which the
  census and the block search do not enumerate. The impostor that free attention found from a noisy compiled start
  (§7) lies outside them.

## 11. Reproduce

Run from `cpu_first/` with the sotaku venv's Python. Local runs take seconds to minutes and stay under 1 GB; the
sweeps run on Modal (app and Volume `bridge`, 20 CPU containers at most).

```bash
~/github/sotaku/venv/bin/python -m bridge.compile
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach bridge/modal_bridge.py --suite ladder
```

```bash
~/github/sotaku/venv/bin/python -m modal run bridge/modal_bridge.py --suite ladder --fetch
```

The other sweeps go to `--suite` the same way:
- `mlpcell`: the MLP digit cell at bases 3-5;
- `plain`: notch 6 with a larger budget;
- `softcert`: the loose notches certified on their soft path;
- `wstab` and `wlearn`: the Net2Net arm, stability and learnability;
- `bypass`: the two bypass fixes.

```bash
~/github/sotaku/venv/bin/python -m bridge.compile_wide
```

```bash
~/github/sotaku/venv/bin/python -m bridge.blocksearch
```

`compile_wide.py` checks the widened compiles (record `records/compile_wide.json`). `blocksearch.py` runs the
block-wise whole-option search at free attention's width in about a minute (record `records/blocksearch.json`).

```bash
~/github/sotaku/venv/bin/python -m bridge.diagnose
```

```bash
~/github/sotaku/venv/bin/python -m bridge.census 3
```

```bash
~/github/sotaku/venv/bin/python -m modal run bridge/modal_census.py --extra 6
```

`census.py N` runs every wiring of the menu with N extra offsets (0: 7 options, 3: 10) locally, in about 1 and 10
minutes. `modal_census.py` splits the 13-option one (62.7 million wirings) over 12 CPU containers.

```bash
~/github/sotaku/venv/bin/python -m bridge.fill
```

H1, from `../one-layer-deeper` (app and Volume `bridge-h1`; records in `history/bridge/h1/records/`):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/history/bridge/h1/modal_h1.py --file submission_wide_mlp.py --mode official --name wide_mlp_official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach ../cpu_first/history/bridge/h1/modal_h1.py --file submission_wide_mlp.py --mode plan --plan ../cpu_first/history/bridge/h1/plans/wide_mlp_cpu.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run ../cpu_first/history/bridge/h1/modal_h1.py --mode fetch
```

Files:
- `grid.py`: the grid layout, the machine's reference values, the compiled attention, the truth tables;
- `net.py`: the looped transformer at every notch;
- `compile.py`: step 1;
- `train.py`: the lessons, training and certification;
- `notches.py`: the notches and lesson tiers;
- `suites.py` and `modal_bridge.py`: the sweeps;
- `summarize.py` and `fill.py`: the tables;
- `probe.py` and `diagnose.py`: step 3;
- `census.py` and `modal_census.py`: every wiring of a menu, run as the hard network;
- `blocksearch.py`: the block-wise whole-option search at free attention's width;
- `compile_wide.py`: the Net2Net arm's widened compiles;
- `h1/`: the H1 file, its launcher, plans and records.
