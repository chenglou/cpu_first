# How wide can the wiring's menu get?

## In plain words

<!-- PLAIN -->
**The question.** The H1 machine already learns its own wiring ([WIRING.md](../learned_wiring/WIRING.md)): each
connection is a switch that picks one option from a short menu, and those menus held the connections of long
division and schoolbook squaring plus a few hundred to a few thousand wrong ones. How general can the menus get before
the machine stops finding a right wiring? And do they still single out long division and schoolbook squaring?

**How big the menus became.** Each connection now has many options (§1):
- **The division.** A link can take any of several registers on either input (including zero and the multiples of N).
  It can add or subtract, and stop on several kinds of signal. The register can move any number of places, take the
  incoming digit at any place, start empty or full, and give its answer from several windows. The menu went from
  3,456 wirings (io3) to 530 million (level 3) and 4.1 billion (level 4).
- **The multiply.** Each digit product can go to any column given by a simple formula of its digit positions, and the
  products can be grouped into the numbers to be added in nine ways. The menu went from 576 layouts to 5.8 million
  (level 2) and 33 million (level 3).
- **Together**, with the reader's 3 pointer shapes, from about 6 million whole machines to about 10^16 (levels 3 + 2)
  and 4·10^17 (levels 4 + 3, the certified H1 file).

**Does the menu still single out long division and schoolbook squaring?** Partly no, partly yes.
- **It now holds genuinely different algorithms, and the machine picks among the ones that work.** A division step
  can be done two ways in this menu: subtract N again and again until it would go negative (io3's way), or try the
  step's number minus s·N for every s and keep the last that stays positive (the trial multiples of long division on
  paper, using the machine's own multiplier). Both pass every check. In the tiny machines the seeds split evenly
  between them (48 and 48 of the 96 runs whose menu holds both). On H1, the certified file's official run and 7 of
  its 8 CPU seeds chose repeated subtraction (one seed exactly io3's wiring), and one seed chose trial multiples; at
  levels 1–3 all three H100 runs chose trial multiples. For the multiply, six groupings of the digit products work:
  rows by either digit, or diagonals, each in either order. All six occurred in the tiny machines; H1's runs chose the
  schoolbook rows, or the same rows summed in reverse order.
- **It holds a Montgomery reduction**, the right-to-left method that cryptographic hardware uses. It runs exactly, but
  it computes a different number (the dividend times 10^−W, mod N), so the checks reject it, and the machine never
  chooses it. To use it, a machine would need a conversion in and out, and the conversion in is itself a long division.
- **What still singles out long division** is the skeleton that the switches plug into: one register, W steps of 9
  links, a digit brought in, a reduction, a move. Two of the checks single it out too, both about a single step (§5).
  "One step from x + q·N must give back x" teaches the step's inner wiring, and a second law teaches how the step
  takes its digit in. Both are definitions of long division's step. Without them, the general laws alone ("x mod N =
  x", "N mod N = 0", "x + Q·N mod N = x", and the answers) still single out exactly the right wirings, but they can
  teach a menu only by trying every whole combination of its parts at once. The cost then grows with the product of
  the parts:
  - on H1 this certified with a 57,600-wiring division menu (level 1), on an H100;
  - with only the second check dropped, it certified with the whole 147,000-combination I/O block (level 3), on an H100;
  - beyond about 10^5 combinations at once it is not affordable (§2.3, §4).

**Where learning breaks, and why.**
- **Impostors that pass the general laws.** "x mod N = x" and "N mod N = 0" put the numbers in the dividend's bottom
  half. A wiring that never reads the top half passes both, for example one that starts with the bottom digits and
  never brings anything in. The two laws even pull the search toward such wirings, and only the 2% of rows with known
  answers push it back. In the tiny machines:
  - with those two laws alone, every run ended on an impostor (8 of 8);
  - with the answers added, 2 of 120 runs still failed and 2 took about 450 extra steps;
  - with the answers alone, every run found the right wiring.

  The law "x + Q·N mod N = x", with a many-digit Q, fills the whole dividend and removes the impostors; the H1 file
  uses it.
- **Search, not identification.** In every menu I counted exactly (§3), the wirings that pass all the checks are all
  correct. At H1's width that is 8 of 5·10^8 division wirings at level 3 and 18 of 4·10^9 at level 4 (the same counts in
  the tiny core); for the multiply, 1 column-rule pair of 470,000. Learning fails only where the search cannot see the
  right combination. If each switch is shown a few sampled combinations of the others
  instead of every one, the impostors catch it (14 of 16 runs at levels 3 and 4).
- **Cost.** Trying every combination of a block costs in proportion to the block. That is cheap for the step's
  chain (3,600–9,400 combinations) and for the step and the reading when they are taught apart (under 2,000 at H1). It
  is not cheap for the whole I/O block at once (147,000–435,000 at H1): an H100 manages it, a CPU does not.

**What certified.**
- **The H1 file at the widest levels** (`submission_wide.py`: 4·10^17 machines) certified on the untouched official
  manifest on an H100. It reached T = 64 on seen and unseen moduli, every rung 768/768, in 592 s of training and 43 s
  of evaluation (io3: 729 s and 52 s). It chose long division by repeated subtraction and the schoolbook rows summed
  in reverse order. **All 8 CPU seeds certified** through the same evaluator (training 1,620–3,117 s, evaluation
  403–879 s): 7 chose repeated subtraction (one of them io3's wiring exactly), 1 trial multiples.
- **Four more H100 runs certified**, each a variant of the file:
  - level 3 + 2, which chose trial multiples (600 s);
  - an earlier build at 4 + 3 (578 s);
  - the whole I/O block taught without the step law (704 s);
  - the general laws alone, with neither of long division's step laws, at level 1 + 2 (1,003 s, trial multiples).
- None needed a relaxed budget.
<!-- /PLAIN -->

## Rule by rule: 7, 8 and 14

The rules are [one-layer-deeper/README.md](../../one-layer-deeper/README.md) §Rules; WIRING.md's table covers these
three for io3. "Gray" means a reasonable reader could object; the point is flagged, not argued away.

| rule | io3 (WIRING.md) | now | still gray |
|---|---|---|---|
| **7. no hard-coded algorithm in the forward pass** | less gray: the wiring is learned switches, but each menu held long division's and schoolbook's connections plus a few alternatives | **less gray again**: the menus no longer single out one algorithm. They hold two long divisions (repeated subtraction, trial multiples), a Montgomery reduction, six groupings of the multiply's products, and up to 10^17 other machines. The machine picks; different seeds pick different correct algorithms. The forward pass at evaluation is the machine the switches select | the skeleton is designed (one register, W steps of 9 links, bring-in, chain, move; one product per digit pair, placed and summed), the menus are mine, and they contain the true wiring; the sizes, the T loop and the adder's composition are fixed |
| **8. logits inside autograd, unbroken gradient path** | every switch straight-through on the prediction path | **the same for every new switch**; each switch's options are made lazily (all of them inside autograd, only the chosen one without, the same value). The gradient test passes at the truth and on the trained models, except one switch on one batch for two of the four models checked (§4.5) | as io3's: a switch's gradient arrives only through the places where its options differ, so on some batches some switches get none. At the truth the final correction gets 0 on T = 1, 2 rows with small moduli (their dividend's top half is empty) and a gradient on T = 8 and OOD-N T = 4 rows; on two trained models it gets 0 on that second batch, because its leading options are aliases whose contributions cancel. The gradient is the discounted straight-through surrogate |
| **14. no data inspection, task-specific solvers, custom loops, participant backward** | gray: the lessons carry designed knowledge (the division ladder, x mod N = x, N mod N = 0, the routed ladder) | **gray in the same way; the new lessons are mostly general laws**: distributivity for the number rule, x + Q·N mod N = x with Q another row's x (the batch's own numbers; Q·N by the machine's own rows) | the step law states long division's bring-in, as the division ladder states its step (§5); the verdicts come from running the hard machine on candidate wirings inside the loss, under no_grad |

## Details

### 1. The grammar

WIRING.md's machine learns its wiring as a set of switches, one per kind of connection, each choosing from a short
hand-written menu. Here every menu is replaced by a family of options that grows level by level. The machine is the
same: a W-digit modulus, a 2W-digit dividend reduced in W steps on a register of P = W + 1 places, each step a chain of
9 links of the learned subtraction table and the learned keep-or-discard gate, and a multiply made of the learned
product cell's digits. What changes is how many ways the parts can be connected.

**The division** (`wide.py` `div_menu`, `submission_wide.py` `DIV_MENU`). A step brings one digit of the dividend into
the register, runs its chain, and moves the register. The switches, in two blocks:

| block | switch | what it chooses | level 0 (io3) | level 1 | level 2 | level 3 | level 4 |
|---|---|---|---|---|---|---|---|
| chain | `a`, `b` | each link's two operands | 3 sources each | any of 5 registers: the chain register, the step's start, N's register, the link's multiple of N (link s takes s·N, made by the machine's own multiplier), zero | = | = | + the register before the bring-in, the incoming digit (7) |
| chain | `op` | subtract A − B, or add A + B (the learned adder) | subtract | both | = | = | = |
| chain | `flag` | what the gate reads | the final borrow | + "the chain register's lowest digit is 0", "the result's lowest digit is 0" (the table's own zero test: the borrow of d − 1) | = | = | + "A < N" (the borrow of A − N) |
| chain | `keep` | what a link writes | always / when the gate commits / never | = | = | = | = |
| chain | `nread` | N's register | as read / reversed | = | = | = | = |
| chain | gate | the learned 2×2 table: commit or hold on flag 0, on flag 1 | 4 tables | = | = | = | = |
| I/O | `vread` | the dividend's digit order | given / reversed | = | = | = | = |
| I/O | `start` | the register's first content | the top W digits | = | top / zero / bottom W digits | = | + N |
| I/O | `order` | the digit brought in at step t | downwards | = | down / up | = | = |
| I/O | `bring` | move the register by k places, then write the digit at place p, replacing or adding | io3's 4 moves (down, stay, up, skip) | = | the 4 moves × replace or add (8) | any k in −(P−1)..P−1, any p or none, replace or add (70 at W = 3, 340 at H1) | = |
| I/O | `post` | move the register after the chain | none | = | −1, 0, +1 | = | −2..2 |
| I/O | `final` | after the last step | nothing | = | nothing / subtract N if no borrow / add the dividend's top W digits, then subtract | = | + subtract twice |
| I/O | `read` | the answer's places | the low or the high W | = | × in order or reversed (4) | = | = |

| level | division wirings at W = 3 (tiny core) | at W = 8 (H1) | chain block | I/O block (W = 3 / H1) |
|---|---|---|---|---|
| 0 (io3's menu) | 3,456 | 3,456 | 216 | 16 |
| 1 | 57,600 | 57,600 | 3,600 | 16 |
| 2 | 1.2·10^7 | 1.2·10^7 | 3,600 | 3,456 |
| 3 | 1.1·10^8 | 5.3·10^8 | 3,600 | 30,240 / 146,880 |
| 4 | 8.4·10^8 | 4.1·10^9 | 9,408 | 89,600 / 435,200 |

(Counts include the gate's 4 tables, as WIRING.md's "864 wirings times the gate's 4 tables" = 3,456.)

**What the division menu contains.** Checked by running every wiring on the true tables (`exact_wide.py`, §3):
- **Long division by repeated subtraction** (CHAIN.md, io3's truth): each link subtracts N from the chain register
  and keeps the result while it does not borrow.
- **Long division by trial multiples**: each link s computes (the step's start) − s·N, with the multiples made by the
  machine's own multiplier, and keeps the last that does not borrow. The same quotient digit, found by multiplying
  instead of by subtracting again and again: the way long division is taught on paper.
- **Aliases**: the digit added rather than written into the empty lowest place, a final subtraction that never fires,
  "A < N" read instead of the borrow (level 4). They compute the same function.
- **A Montgomery reduction** (REDC with R = 10^W): start empty, take the dividend's digits from the bottom, add N while
  the lowest digit is not 0, shift down, add the dividend's top half, subtract N once if needed. It is exact: on every
  test it computes V·10^−W mod N (`check_grammar.py`). That is the Montgomery form of V mod N, a
  different number, so "x mod N = x" rules it out. A Montgomery machine would need a conversion in and out, and
  the conversion in is itself a long division.
- **Repeated subtraction with other stop rules**, and very many machines that compute nothing useful.

Not in the menu: non-restoring division, Barrett reduction, a division on a 2W-digit register (the step works on a
P-place window), different sizes, untied switches (one per step or link).

**The multiply** (`mul_menu`, `MUL_MENU`). Every pair of digits (s_i, s_j) goes through the learned product cell; its
low digit and its high digit are placed in a column and in a number, and the numbers are summed by the learned adder.

| level | column rule (one per output kind: low digit, high digit) | number rule | layouts at W = 3 | at H1 |
|---|---|---|---|---|
| 0 (io3) | f(i) + g(j) + d: 24 rules | by multiplier digit j (fixed) | 576 | 576 |
| 1 | any a·i + b·j + c, a, b in −2..2, c in −2(W−1)..C−1 (300 / 800 rules) | by j (fixed) | 9.0·10^4 | 6.4·10^5 |
| 2 | = | α·i + β·j + γ, α, β in −1..1 (9 rules) | 8.1·10^5 | 5.8·10^6 |
| 3 | a, b in −3..3, c in −3(W−1)..C−1 (686 / 1,911 rules) | = | 4.2·10^6 | 3.3·10^7 |

**What the multiply menu contains.** One column rule pair puts every digit at its true weight (i + j for the low
digit, i + j + 1 for the high digit); no other pair of the 90,000–470,000 does (§3). Six number rules are correct, any
grouping of the partial products in which no two digits of one number share a column: rows by multiplier digit
(schoolbook, io3's), rows by multiplicand digit, the diagonals i − j (each also in reverse order). The other three (the
anti-diagonals i + j, everything in one number, and the reversed anti-diagonals) drop digits.

**Combined** (with io3's reader, 3 pointer shapes): io3 2.0·10^6 wirings times 3; level 3 + 2: 3.0·10^15 times 3;
level 4 + 3 (the certified H1 file): 1.3·10^17 times 3.

**What is still hand-written.** The skeleton the switches plug into: one register of P places, W steps, 9 links per
step, the order bring-in → chain → move, one final correction, one readout; the multiply as one product per digit
pair, placed and summed; the T count-down and its loop; the adder as A − (0 − B). And every menu: the options are mine,
and they include the true wiring.

### 2. The tiny core

<!-- TINY -->
`tiny_wide.py`: H1's round in miniature (square x, reduce mod N) with every table at the truth, so that what is
measured is the wiring's search alone. The H1 file learns the tables as well (§4). Settings:
- base 10, W = 3: N has 2 digits in training (coprime to 10, as H1's Blum moduli are), x < N, dividend x² in 2W
  digits, 2% of the rows labeled with x² mod N (H1: the T = 1 and 2 rows);
- certified when the argmax machine is exact on the whole training pool and on 4,000 rows with N of 1 to 3 digits.
  The 3-digit moduli are never trained, as H1's 8-digit ones;
- 8 seeds per cell, Adam (lr 0.05) on the switches' logits (σ 0.1 at the start), batches of 128 rows. A run stops 30
  steps after its argmax machine first passes every lesson and every batch answer, or at 1,500 steps (800 at level 4).

The lessons are H1's (`submission_wide.py`), each a mixture of whole combinations with crisp verdicts (TINYROUTING.md
§6), in H1's order:
- the routed ladder for the column rules (blended on the wire) and distributivity for the number rule, from step 1;
- the division ladder for the chain block, once the layout has passed for 10 steps;
- the whole-division laws x mod N = x and N mod N = 0 plus the answers for the I/O block, once the argmax chain has
  passed the division ladder for 10 steps. (The H1 file adds x + Q·N mod N = x, because of what §2.1 and §2.3 found.)

"Steps to the final wiring" is the step at which the argmax wiring last changed.

#### 2.1 The menu ladder

<!-- TINY_GROW -->
| levels | menu: division / multiply / total | certified | steps to the final wiring, median (range) | layout found at | chain lesson passes at | division chain chosen | multiply grouping chosen | failures |
|---|---|---|---|---|---|---|---|---|
| division 0, multiply 0 | 3.5e3 / 5.8e2 / 2.0e6 | 8/8 | 58 (38-63) | 44 (25-49) | 56 (36-62) | 8 repeated subtraction | 1 kind (8 rows by multiplier digit) | - |
| division 0, multiply 2 | 3.5e3 / 8.1e5 / 2.8e9 | 8/8 | 96 (84-123) | 83 (71-110) | 94 (82-122) | 8 repeated subtraction | 4 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 2 rows by multiplier digit, reversed) | - |
| division 0, multiply 3 | 3.5e3 / 4.2e6 / 1.5e10 | 8/8 | 96 (70-103) | 84 (57-91) | 96 (70-101) | 8 repeated subtraction | 4 kinds (4 rows by multiplier digit, reversed, 2 rows by multiplier digit, 1 rows by multiplicand digit) | - |
| division 1, multiply 0 | 5.8e4 / 5.8e2 / 3.3e7 | 8/8 | 80 (38-146) | 46 (38-60) | 84 (48-146) | 5 trial multiples, 3 repeated subtraction | 1 kind (8 rows by multiplier digit) | - |
| division 1, multiply 2 | 5.8e4 / 8.1e5 / 4.7e10 | 8/8 | 114 (74-160) | 78 (70-100) | 120 (81-160) | 4 trial multiples, 4 repeated subtraction | 3 kinds (4 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals j - i) | - |
| division 1, multiply 3 | 5.8e4 / 4.2e6 / 2.4e11 | 8/8 | 116 (84-165) | 81 (71-106) | 116 (94-165) | 5 trial multiples, 3 repeated subtraction | 3 kinds (4 rows by multiplier digit, 2 rows by multiplier digit, reversed, 2 diagonals i - j) | - |
| division 2, multiply 0 | 1.2e7 / 5.8e2 / 7.2e9 | 8/8 | 86 (55-127) | 46 (27-56) | 78 (50-123) | 5 trial multiples, 3 repeated subtraction | 1 kind (8 rows by multiplier digit) | - |
| division 2, multiply 2 | 1.2e7 / 8.1e5 / 1.0e13 | 7/8 | 129 (113-168) | 79 (61-94) | 102 (78-150) | 4 trial multiples, 4 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 3 rows by multiplier digit, 1 diagonals i - j) | d2m2_s2: 0.65 of training rows right at step 1500 (chain repeated subtraction; start bottom, bring (0, -1, 'add'), final none) |
| division 2, multiply 3 | 1.2e7 / 4.2e6 / 5.3e13 | 7/8 | 159 (100-165) | 90 (75-110) | 128 (86-163) | 4 trial multiples, 4 repeated subtraction | 3 kinds (6 rows by multiplier digit, reversed, 1 rows by multiplicand digit, 1 rows by multiplier digit) | d2m3_s2: 0.65 of training rows right at step 1500 (chain repeated subtraction; start bottom, bring (0, -1, 'add'), final none) |
| division 3, multiply 0 | 1.1e8 / 5.8e2 / 6.3e10 | 8/8 | 100 (54-145) | 44 (33-62) | 89 (47-123) | 5 trial multiples, 3 repeated subtraction | 1 kind (8 rows by multiplier digit) | - |
| division 3, multiply 2 | 1.1e8 / 8.1e5 / 8.8e13 | 8/8 | 134 (106-622) | 76 (66-111) | 115 (81-162) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 2 rows by multiplier digit, 2 rows by multiplicand digit) | - |
| division 3, multiply 3 | 1.1e8 / 4.2e6 / 4.6e14 | 8/8 | 144 (95-619) | 88 (68-99) | 109 (80-173) | 5 trial multiples, 3 repeated subtraction | 4 kinds (4 rows by multiplier digit, 2 rows by multiplier digit, reversed, 1 diagonals j - i) | - |
| division 4, multiply 0 | 8.4e8 / 5.8e2 / 4.9e11 | 8/8 | 104 (53-148) | 48 (18-54) | 62 (34-99) | 6 repeated subtraction, 2 trial multiples | 1 kind (8 rows by multiplier digit) | - |
| division 4, multiply 2 | 8.4e8 / 8.1e5 / 6.8e14 | 8/8 | 135 (101-179) | 76 (57-87) | 95 (74-129) | 6 repeated subtraction, 2 trial multiples | 3 kinds (6 rows by multiplier digit, 1 rows by multiplier digit, reversed, 1 rows by multiplicand digit) | - |
| division 4, multiply 3 | 8.4e8 / 4.2e6 / 3.6e15 | 8/8 | 143 (103-199) | 85 (59-95) | 101 (76-142) | 6 repeated subtraction, 2 trial multiples | 3 kinds (5 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 1 rows by multiplier digit, reversed) | - |
<!-- /TINY_GROW -->

- **The menus can grow a billion-fold and the wiring is still found.** From io3's menu (2·10^6 wirings in all) to
  level 4 + 3 (3.6·10^15), 118 of 120 runs certified. The steps to the final wiring grow from a median of 58 to 96–159.
  Most of the growth is the multiply's column rules, which take longer to settle among 300–686 rules than among 24:
  with io3's multiply, the division's levels alone take the median from 58 to 80–104.
- **The machine chooses between genuinely different correct algorithms**, seed by seed: wherever the menu holds both
  chains (levels 1–4), 48 runs ended on trial multiples and 48 on repeated subtraction. All six correct groupings of
  the products occur (rows by multiplier digit 34 times, the same reversed 27, rows by multiplicand digit 6 and 6
  reversed, the diagonals 4 and 3, of the 80 runs whose menu has the number rule).
- **Aliases** are chosen too: the digit added rather than written into the empty place (49 of 120 runs), a final
  subtraction that never fires (18).
- **The two failures, and the slow runs, are the same impostor.** At level 2, seed 2 ended on "start with the
  dividend's bottom digits and never bring anything in" with both wide multiplies (65% of the training rows right; the
  division's switches start the same in both). At level 3, seed 1 sat in it for about 450 steps with both wide
  multiplies before the 2% answers moved it out. It passes x mod N = x and N mod N = 0 (§3: their dividends have an
  empty top half).

#### 2.2 Teaching a large I/O block

At level 3 the I/O block has 30,240 combinations (147,000 at H1 width), at level 4 89,600 (435,000).

<!-- TINY_IO -->
| variant | menu: division / multiply / total | certified | steps to the final wiring, median (range) | layout found at | chain lesson passes at | division chain chosen | multiply grouping chosen | failures |
|---|---|---|---|---|---|---|---|---|
| cs_d3m2 | 1.1e8 / 8.1e5 / 8.8e13 | 0/8 | - | 76 (66-111) | 115 (81-162) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 2 rows by multiplier digit, 2 rows by multiplicand digit) | cs_d3m2_s1: 0.65 of training rows right at step 1000 (chain trial multiples; start zero, bring (0, 0, 'add'), final none); cs_d3m2_s7: 0.65 of training rows right at step 1000 (chain repeated subtraction; start zero, bring (0, 0, 'rep'), final none); 6 more alike |
| cs_d4m3 | 8.4e8 / 4.2e6 / 3.6e15 | 2/8 | 346 (164-527) | 85 (59-95) | 101 (76-142) | 4 repeated subtraction, 2 other, 2 trial multiples | 3 kinds (5 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 1 rows by multiplier digit, reversed) | cs_d4m3_s4: 0.65 of training rows right at step 1000 (chain repeated subtraction; start bottom, bring (0, -1, 'add'), final csub); cs_d4m3_s3: 0.66 of training rows right at step 1000 (chain other; start bottom, bring (0, -1, 'rep'), final csub2); 4 more alike |
| local_d3m2 | 1.1e8 / 8.1e5 / 8.8e13 | 8/8 | 128 (89-168) | 76 (66-111) | 115 (81-162) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 2 rows by multiplier digit, 2 rows by multiplicand digit) | - |
| local_d4m3 | 8.4e8 / 4.2e6 / 3.6e15 | 8/8 | 116 (83-147) | 85 (59-95) | 101 (76-142) | 6 repeated subtraction, 2 trial multiples | 3 kinds (5 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 1 rows by multiplier digit, reversed) | - |
<!-- /TINY_IO -->

- **"enum"** (the grow suite's runs): every combination of the block, every step. It works, at a cost that grows with
  the block: a step takes 0.24 s at io3's menu, 0.59 s at level 3 and 1.4–1.8 s at level 4 (medians on 4 CPUs,
  everything included; at level 4 the chain's 9,408 combinations are much of it).
- **"local"**: the step's switches (bring 70 × post 3–5) are taught by the step law on the division ladder's
  registers, the reading's (vread, start, order, final, read: 144–256) by the whole-division laws and the answers. It
  tries about 1% of the combinations (210 + 144 at level 3, 350 + 256 at level 4), takes 0.32 s and 0.58 s a step, and
  certifies 8 of 8 at both levels. The step law excludes the top-half impostors by construction (a step that brings
  nothing in fails it). It is long division's bring-in, though (§5).
- **"cs"**, coordinate-sampled: each switch's options are tried against 8 sampled combinations of the other switches
  (one of them their argmax), so the cost is the sum of the option counts, not their product. It certifies 2 of 16:
  the other runs lock into the top-half impostors (start empty or with the bottom digits, or write the digit and shift
  after the chain and read the high places) and stay there. Exhaustive enumeration escapes because every combination keeps a
  verdict of its own, so the answers can tell the right one from the impostors; sampled verdicts mostly miss the right
  combination.

#### 2.3 Without the steps' laws

<!-- TINY_LAWS -->
| variant | menu: division / multiply / total | certified | steps to the final wiring, median (range) | layout found at | chain lesson passes at | division chain chosen | multiply grouping chosen | failures |
|---|---|---|---|---|---|---|---|---|
| joint_xna_d0 | 3.5e3 / 8.1e5 / 2.8e9 | 8/8 | 87 (73-112) | 83 (71-110) | - | 8 repeated subtraction | 4 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 2 rows by multiplier digit, reversed) | - |
| joint_xna_d1 | 5.8e4 / 8.1e5 / 4.7e10 | 8/8 | 102 (71-143) | 78 (70-100) | - | 5 trial multiples, 3 repeated subtraction | 3 kinds (4 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals j - i) | - |
| joint_xnqa_d0 | 3.5e3 / 8.1e5 / 2.8e9 | 8/8 | 82 (73-97) | 80 (70-90) | - | 8 repeated subtraction | 4 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 2 rows by multiplier digit, reversed) | - |
| joint_xnqa_d1 | 5.8e4 / 8.1e5 / 4.7e10 | 8/8 | 102 (72-130) | 79 (71-95) | - | 5 trial multiples, 3 repeated subtraction | 3 kinds (4 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals j - i) | - |
| ladder_a_d3 | 1.1e8 / 8.1e5 / 8.8e13 | 8/8 | 128 (96-166) | 76 (66-111) | 115 (81-162) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 2 rows by multiplier digit, 2 rows by multiplicand digit) | - |
| ladder_q_d3 | 1.1e8 / 8.1e5 / 8.8e13 | 8/8 | 116 (82-174) | 74 (61-90) | 104 (79-169) | 4 trial multiples, 4 repeated subtraction | 5 kinds (2 rows by multiplier digit, 2 rows by multiplier digit, reversed, 2 rows by multiplicand digit) | - |
| ladder_xn_d3 | 1.1e8 / 8.1e5 / 8.8e13 | 0/8 | - | 76 (66-111) | 115 (81-162) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, reversed, 2 rows by multiplier digit, 2 rows by multiplicand digit) | ladder_xn_d3_s3: 0.66 of training rows right at step 1000 (chain repeated subtraction; start bottom, bring (0, -1, 'rep'), final none); ladder_xn_d3_s6: 0.64 of training rows right at step 1000 (chain trial multiples; start bottom, bring (0, -1, 'rep'), final none); 6 more alike |
| ladder_xnqa_d3 | 1.1e8 / 8.1e5 / 8.8e13 | 8/8 | 120 (82-170) | 74 (61-90) | 104 (79-169) | 4 trial multiples, 4 repeated subtraction | 5 kinds (2 rows by multiplier digit, 2 rows by multiplier digit, reversed, 2 rows by multiplicand digit) | - |
| whole_xna_d0 | 3.5e3 / 8.1e5 / 2.8e9 | 5/8 | 178 (110-583) | 83 (71-110) | - | 5 repeated subtraction, 3 other | 4 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 2 rows by multiplier digit, reversed) | whole_xna_d0_s3: 0.28 of training rows right at step 800 (chain other; start top, bring (1, 0, 'rep'), final none); whole_xna_d0_s0: 0.28 of training rows right at step 800 (chain other; start top, bring (1, 0, 'rep'), final none); 1 more alike |
| whole_xna_d1 | 5.8e4 / 8.1e5 / 4.7e10 | 5/8 | 119 (74-374) | 78 (70-100) | - | 3 repeated subtraction, 3 other, 2 trial multiples | 3 kinds (4 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals j - i) | whole_xna_d1_s5: 0.47 of training rows right at step 800 (chain other; start top, bring (1, 0, 'rep'), final none); whole_xna_d1_s4: 0.47 of training rows right at step 800 (chain other; start top, bring (1, 0, 'rep'), final none); 1 more alike |
| whole_xnqa_d0 | 3.5e3 / 8.1e5 / 2.8e9 | 3/8 | 84 (76-89) | 80 (70-90) | - | 5 other, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, 2 rows by multiplier digit, reversed, 2 rows by multiplicand digit, reversed) | whole_xnqa_d0_s1: 0.03 of training rows right at step 800 (chain other; start top, bring (0, 0, 'rep'), final none); whole_xnqa_d0_s2: 0.03 of training rows right at step 800 (chain other; start top, bring (0, 0, 'rep'), final none); 3 more alike |
| whole_xnqa_d1 | 5.8e4 / 8.1e5 / 4.7e10 | 4/8 | 105 (73-147) | 79 (71-95) | - | 4 other, 2 trial multiples, 2 repeated subtraction | 3 kinds (4 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals j - i) | whole_xnqa_d1_s1: 0.03 of training rows right at step 800 (chain other; start top, bring (1, 0, 'rep'), final none); whole_xnqa_d1_s7: 0.04 of training rows right at step 800 (chain other; start top, bring (0, -1, 'rep'), final none); 2 more alike |
<!-- /TINY_LAWS -->

- **joint**: no division ladder; the chain and the I/O switches enumerated together (3,456 combinations at level 0,
  57,600 at level 1) under the whole-division laws and the answers.
- **whole**: no division ladder either, but the chain and the I/O blocks enumerated separately, each with the other
  at its argmax.
- **ladder_…**: the division ladder kept, the I/O block (level 3) under other laws: x and n only, the answers only,
  all of them with x + Q·N mod N = x, and that law alone.

<!-- TINY_LAWS_NOTES -->
- **The general laws alone are enough, if every combination is tried together** ("joint"): 32 of 32 runs at levels 0
  and 1, with and without the many-digit law, in about the same number of steps as with the division ladder. What
  limits it is cost: the joint block is the product of the chain and the I/O blocks (3,456 and 57,600 here; 4.3·10^8
  at level 3), and every combination is a whole division. On H1 at level 1 it certified on the H100 too (§4).
- **Taught block by block, the general laws fail half the time** ("whole": 17 of 32). Two different ways:
  - with x mod N = x and N mod N = 0 only, any chain that takes N away correctly once passes both laws, because neither
    ever needs two subtractions in a step. Examples: subtract N once from the step's start, or subtract the link's
    multiple from the running register. The answers would rule these out, but here they reach only the I/O block. The
    chain's switches split between such a chain and long division, and 6 of 16 runs ended on the wrong side, with
    3.5–47% of answers right;
  - with x + Q·N mod N = x added, both blocks deadlock in 9 of 16 runs: no chain passes while the I/O is wrong, no
    I/O passes while the chain is wrong, so neither mixture gets a signal. The I/O switches stay undecided (the
    bring-in's largest probability 0.26–0.56 after 800 steps). The stricter the law, the flatter the landscape away
    from the answer.
- **With the division ladder for the chain**, the I/O block's laws decide what it can be fooled by (level 3):
  - x mod N = x and N mod N = 0 alone: 0 of 8. Every run ends on "start with the bottom digits, bring nothing in"
    (64–66% of answers right), the impostor §3 predicts;
  - x + Q·N mod N = x alone: 8 of 8, as fast as with every law (median 116 steps);
  - all of them with the answers: 8 of 8;
  - the 2% answers alone: 8 of 8 (median 128 steps).

  So x mod N = x and N mod N = 0 are what pull the search toward the top-half impostors: those pass them perfectly,
  while they answer only 64–66% of the questions. Without the two laws nothing pulls there, and the answers alone
  find the wiring. This is TINYROUTING.md's "laws as teachers reward the bypass" in a new form. Here the remedy is
  a law with a fuller dividend, x + Q·N mod N = x, not leaving the laws out: the H1 file keeps all three and the
  answers.

So a law that reads one block directly, as the division ladder reads the chain, is what lets the blocks be learned
apart. Learning them apart is what keeps the cost a sum of the blocks' sizes instead of their product (TINYROUTING.md's
third condition, "an unscrambled lesson per switch"). That law is long division's.
<!-- /TINY_LAWS_NOTES -->

#### 2.4 A deeper tiny core

W = 4: N of 2 and 3 digits in training, certified up to 4 digits (4 division steps).

<!-- TINY_W4 -->
| variant | menu: division / multiply / total | certified | steps to the final wiring, median (range) | layout found at | chain lesson passes at | division chain chosen | multiply grouping chosen | failures |
|---|---|---|---|---|---|---|---|---|
| w4_enum_d0m0 | 3.5e3 / 5.8e2 / 2.0e6 | 8/8 | 58 (40-67) | 46 (27-55) | 57 (40-65) | 8 repeated subtraction | 1 kind (8 rows by multiplier digit) | - |
| w4_enum_d3m2 | 1.7e8 / 1.4e6 / 2.4e14 | 8/8 | 135 (55-177) | 78 (27-90) | 119 (39-142) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals i - j) | - |
| w4_enum_d4m3 | 1.3e9 / 7.8e6 / 1.0e16 | 8/8 | 122 (102-179) | 84 (65-102) | 100 (80-151) | 6 repeated subtraction, 2 trial multiples | 5 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 1 diagonals j - i) | - |
| w4_local_d3m2 | 1.7e8 / 1.4e6 / 2.4e14 | 8/8 | 132 (62-147) | 78 (27-90) | 119 (39-142) | 5 trial multiples, 3 repeated subtraction | 4 kinds (3 rows by multiplier digit, 3 rows by multiplier digit, reversed, 1 diagonals i - j) | - |
| w4_local_d4m3 | 1.3e9 / 7.8e6 / 1.0e16 | 8/8 | 112 (90-155) | 84 (65-102) | 100 (80-151) | 6 repeated subtraction, 2 trial multiples | 5 kinds (3 rows by multiplier digit, 2 rows by multiplicand digit, reversed, 1 rows by multiplicand digit) | - |
<!-- /TINY_W4 -->

- **Every run certified** (40 of 40): io3's menu, levels 3 + 2 and 4 + 3 (1.0·10^16 wirings at W = 4), with the whole
  I/O block or the step law. The steps to the final wiring are about those of W = 3 (medians 112–135 at the wide
  levels, 58 at io3's). One more division step changed nothing measurable.

#### 2.5 The failure modes, by kind

- **Impostors that pass the lessons in use.** The only kind that cost a run in the main recipe. x mod N = x and N mod
  N = 0 have an empty top half, so machines that mishandle the top half pass them (§3: 24 of the 28 wirings passing
  both at level 3). The 2% answers rule them out, slowly: 2 failures and 2 slow runs in 120 (§2.1), 14 of 16 failures
  when the search is sampled (§2.2), and 8 of 8 failures when the two laws teach the I/O block without the answers
  (§2.3). x + Q·N mod N = x and the step law each remove them.
- **Exactly symmetric options.** The aliases (replace or add into an empty place; a final subtraction that never
  fires; the six correct number rules; the two correct chains) compute the same function, so no lesson prefers one.
  Their weights stay split: at the end of the certified tiny runs the bring-in's largest probability has a median of
  0.47, the number rule's 0.47 (every run below 0.75), the final correction's 0.78. This is harmless, because the
  argmax machine is correct whichever alias leads; it is also why seeds end on different algorithms. The two chains
  are the exception: they differ in two switches at once (a and b), and the factorized mixture pushes the pair to one
  corner, since a mixed pair (start minus N, or chain register minus s·N) is wrong.
- **Ties between a right and a wrong option** when the lessons cannot tell them apart. With the general laws taught
  block by block, "subtract N once" ties with long division (§2.3): 6 of 16 runs ended on the wrong side.
- **Deadlocks** when a block has no lesson that reads it alone. Taught block by block with the strict laws, 9 of 16
  runs never left the start (§2.3): every verdict is 0 while the other block is wrong.
- **Searches that sample.** A crisp law gives a verdict only to the combinations it is shown. Sampled combinations
  mostly miss the right one, while the impostors are many and easy to hit (§2.2).

Not seen: TINYROUTING's twin (swapped ports with a mirrored table): the tables are given in the tiny core, and at H1
their checks fix which input is which. Nor any failure of the multiply's search: every run found its column rules and
a correct number rule.
<!-- /TINY -->

### 3. Exact counts

<!-- EXACT -->
`exact_wide.py`, on exact arithmetic (every table at the truth), W = 3 and, for the division, W = 8 (H1's width).
Every combination of a block runs on 256 training rows (each of the 10 rungs q for the division ladder) while the
other block stays at the truth. A combination "passes" a law when it holds on every row; the passing ones then run
the whole machine on 3,000 certification rows (N of 1 to 3 digits; at H1 width 2,000 rows, N of 1 to 8 digits). The
multiply's column rules run the routed ladder on every rung of every row j, the number rules distributivity.

| block (level) | combinations | pass the division ladder | pass x mod N = x | + N mod N = 0 | answers alone | x + n + answers | x + n + (x + Q·N mod N = x) | all | of these, correct |
|---|---|---|---|---|---|---|---|---|---|
| chain (1, 2, 3) | 3,600 | 2 | | | | | | | 2: repeated subtraction, trial multiples |
| chain (4) | 9,408 | 3 | | | | | | | 3: + repeated subtraction stopping on "D < N" |
| I/O (1) | 16 | | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| I/O (3) | 30,240 | | 28 | 28 | 4 | 4 | 4 | 4 | 4: × replace or add, × final subtraction or none |
| I/O (4) | 89,600 | | 65 | 39 | 6 | 6 | 6 | 6 | 6: × final none, once or twice |
| chain + I/O together, no division ladder (1) | 57,600 | | 1,942 | 4 | | 2 | 2 | 2 | 2 |
| **at H1 width** (W = 8, N of 4 to 7 digits; 64 rows, 48 at level 4) | | | | | | | | | |
| chain (3) | 3,600 | 2 | | | | | | | 2 |
| chain (4) | 9,408 | 3 | | | | | | | 3 |
| I/O (3) | 146,880 | | 31 | 28 | 4 | 4 | 4 | 4 | 4 |
| I/O (4) | 435,200 | | 79 | 39 | 6 | 6 | 6 | 6 | 6 |

| multiply (level) | column-rule pairs | pass the routed ladder | number rules | distributive |
|---|---|---|---|---|
| 1, 2 | 90,000 | 1 (i + j, i + j + 1) | 9 | 6 |
| 3 | 470,596 | 1 | 9 | 6 |

- **Every wiring that passes all its lessons is correct**, at every level counted, in the tiny core and at H1's width.
  Their number is small: 2 × 4 = 8 of 1.1·10^8 (W = 3) and of 5.3·10^8 (H1) division wirings at level 3; 3 × 6 = 18 of
  8.4·10^8 and of 4.1·10^9 at level 4. The checks identify the right wirings exactly; what remains is the search.
- **The general laws' blind spot.** x mod N = x and N mod N = 0 run on dividends whose top half is empty. At level 3,
  28 wirings pass both; 24 are machines that mishandle the top half: they start empty and never bring the top digits
  in, start with the bottom digits and bring nothing in, write the digit before the shift, or add the top half at the
  end. In the tiny core each answers the 69–70% of questions whose square fits in the bottom half. Either the answers
  or the law x + Q·N mod N = x (whose dividend fills both halves) removes all 24, in the tiny core and at H1's width.
- **The general laws alone** (no division ladder, no step law, both blocks together at level 1): 1,942 of 57,600
  pass x mod N = x, 4 also pass N mod N = 0, and 2, the two long divisions, pass all.
- **The step law** (it teaches the bring-in and the post-move: 210 combinations at level 3, 1,020 at H1 width) was not
  counted here. In the H1 file at the truth, the reading block it leaves (144 combinations at level 3, 256 at level 4)
  has 6 and 8 combinations passing all three laws on 32 rows: the truth and its aliases.
- **The multiply**: of 90,000 (470,596) column-rule pairs, one puts every digit at its true weight; 6 of the 9 number
  rules group the products without collisions (rows by either digit, diagonals, each in either order).
<!-- /EXACT -->

### 4. H1

<!-- H1 -->
#### 4.1 The file

`submission_wide.py` is `learned_wiring/submission_io3.py` (WIRING.md stage 3) with its two menus replaced by the
grammar of §1, at levels 4 (division) and 3 (multiply), and these changes:
- **the machine on digits**: the division with per-row switch codes (any source on either port, add or subtract, the
  zero tests and "A < N" through the table's own borrows, the bring-in's move and insert, the post-move, the final
  correction, the readout); N's multiples from the machine's own row 0 (the learned product cell, the argmax column
  rules, the learned adder); the multiply's numbers summed one by one, empty ones skipped;
- **the prediction path**: the same machine on one-hot digits with every switch straight-through; each switch's
  options are made lazily: all of them inside autograd, only the chosen one without (as at evaluation);
- **the lessons**: the number rule's distributivity (new), the division ladder over the wider chain block (9,408
  combinations with the gate's rows), the step law for the bring-in and the post-move (new), and the reading's
  whole-division laws with x + Q·N mod N = x added (Q another row's x, Q·N from the machine's own rows; new). IO_MODE
  "enum" and "joint" swap in the other ways of teaching the I/O block (§2.2, §2.3);
- unchanged: the tables' lessons and schedule, the hand-written adder's 70-step warm start (the anchors session
  removes it separately), the reader and the writer, T's lesson, the freeze.

It has 11,187 numbers (io3: 7,037) and is 139 KB (the limit is 256 KiB). The plumbing check (every table and
switch at the truth) is exact on both prediction paths with identical logits at levels 0, 3 and 4, on the true
wiring and on the correct alternatives (trial multiples, every correct number rule, the aliases). The gradient test
(rule 8) passes at the truth and on the trained models checked, with one exception (§4.5).

#### 4.2 The runs

Every H100 run is the untouched official manifest (seed 74, 3,600 s training, 1,800 s evaluation) with the file sent
byte for byte. The CPU seeds are the same evaluator with the manifest's device set to cpu and the seed changed. The
other H100 runs ran a variant of the file through `--set` (the record keeps the exact source):
- `wide_try2_h100` and `wide_d4m3_try1_h100` are earlier builds at levels 3 + 2 and 4 + 3 (md5 `5b669453…` and
  `1512926b…`; the file is `43db5606…`);
- `wide_joint_d1m2_h100` drops both steps' laws ("joint", level 1 + 2);
- `wide_enum_d3m2_h100` drops the step law ("enum", level 3 + 2);
- `wide_try1_h100` (not in the table) was the first try at level 3 + 2. It crashed when the chain lesson joined: a bug
  in my lesson code, which passed N's multiples as a function to a chain that indexed them as a tensor. The fix is in
  every later build; its record is kept.

<!-- H1_RUNS -->
| run | seed | max T seen / unseen | rungs (fewest right of 768, seen / unseen) | test, ood_t, ood_n_t | subtractor, product cell, gate | division chain | multiply grouping | layout columns right | shape | reading chosen | ladder joins | chain lesson joins | I/O lesson joins | T settled | frozen | never trained | training s | evaluation s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wide_official (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | 149 | 314 | 361 | 387 | 566 | 0 | 591.8 | 42.9 |
| wide_try2_h100 (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit | 288/288 | step | 134 | 185 | 331 | 354 | 396 | 575 | 0 | 599.8 | 43.9 |
| wide_d4m3_try1_h100 (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | 149 | 314 | 361 | 387 | 566 | 0 | 578.2 | 44.6 |
| wide_joint_d1m2_h100 (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit, reversed | 288/288 | step | 117 | 225 | 369 | 370 | 465 | 573 | 0 | 1002.6 | 47.4 |
| wide_enum_d3m2_h100 (H100) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit | 288/288 | step | 134 | 185 | 331 | 354 | 378 | 557 | 0 | 703.6 | 42.7 |
| wide_cpu_seed74 (CPU) | 74 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit, reversed | 288/288 | step | 97 | 149 | 314 | 361 | 388 | 567 | 0 | 1619.6 | 477.2 |
| wide_cpu_seed1 (CPU) | 1 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | trial multiples | rows by multiplier digit | 288/288 | step | 140 | 193 | 361 | 400 | 441 | 620 | 0 | 1790.4 | 432.2 |
| wide_cpu_seed2 (CPU) | 2 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 99 | 167 | 327 | 350 | 440 | 569 | 0 | 1784.5 | 879.1 |
| wide_cpu_seed3 (CPU) | 3 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 115 | 210 | 374 | 424 | 468 | 647 | 0 | 1716.9 | 403.2 |
| wide_cpu_seed4 (CPU) | 4 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 135 | 246 | 417 | 457 | 499 | 678 | 0 | 3117.0 | 551.8 |
| wide_cpu_seed5 (CPU) | 5 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 121 | 239 | 414 | 454 | 489 | 668 | 0 | 2270.5 | 666.1 |
| wide_cpu_seed6 (CPU) | 6 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 134 | 192 | 340 | 388 | 478 | 613 | 0 | 2157.5 | 477.8 |
| wide_cpu_seed7 (CPU) | 7 | 64 / 64 | 768 / 768 | 1.0, 1.0, 1.0 | 200/200, 100/100, 10 | repeated subtraction | rows by multiplier digit | 288/288 | step | 105 | 200 | 368 | 390 | 486 | 623 | 0 | 2598.6 | 655.8 |
<!-- /H1_RUNS -->

<!-- H1_NOTES -->
#### 4.3 What each run chose

| run | menu (division / multiply, I/O teaching) | division chain | aliases | multiply grouping | chain lesson joins → I/O lesson joins → T settled → frozen |
|---|---|---|---|---|---|
| `wide_official` (the file) | 4 / 3, step law | repeated subtraction | digit added into the empty place; a final subtraction that never fires | rows by multiplier digit, reversed | 314 → 361 → 387 → 566 |
| `wide_d4m3_try1_h100` | 4 / 3, step law | repeated subtraction | the same | rows by multiplier digit, reversed | 314 → 361 → 387 → 566 |
| `wide_try2_h100` | 3 / 2, step law | trial multiples | a final subtraction that never fires | rows by multiplier digit (schoolbook) | 331 → 354 → 396 → 575 |
| `wide_enum_d3m2_h100` | 3 / 2, the whole I/O block, no step law | trial multiples | none | rows by multiplier digit | 331 → 354 → 378 → 557 |
| `wide_joint_d1m2_h100` | 1 / 2, no step laws at all | trial multiples | none | rows by multiplier digit, reversed | 369 → (joint, 370) → 465 → 573 |

<!-- H1_CPU_CHOICES -->
| CPU seed | certified | division chain | aliases | multiply grouping | frozen at |
|---|---|---|---|---|---|
| 74 | yes | repeated subtraction | digit added, final subtraction | rows by multiplier digit, reversed | 567 |
| 1 | yes | trial multiples | final subtraction | rows by multiplier digit | 620 |
| 2 | yes | repeated subtraction | final subtraction, stops on "D < N" | rows by multiplier digit | 569 |
| 3 | yes | repeated subtraction | final subtraction | rows by multiplier digit | 647 |
| 4 | yes | repeated subtraction | digit added | rows by multiplier digit | 678 |
| 5 | yes | repeated subtraction | final subtraction twice | rows by multiplier digit | 668 |
| 6 | yes | repeated subtraction | none (io3's own wiring) | rows by multiplier digit | 613 |
| 7 | yes | repeated subtraction | digit added, final subtraction | rows by multiplier digit | 623 |
<!-- /H1_CPU_CHOICES -->

- **Which long division wins depends on the menu, not only on the seed.** At level 4 repeated subtraction won in the
  official run and 7 of the 8 CPU seeds, and in 18 of 24 tiny runs at level 4; at levels 1–3 trial multiples
  led (all three H100 runs there; 42 of 72 tiny runs). Level 4 adds a third correct chain, repeated subtraction
  stopping on "D < N" (the same borrow), so two of the three correct corners of the chain's switches are repeated
  subtraction, and the mixture's weight flows there more often.
- **Every H100 run certified**, at every menu level and with every way of teaching the I/O. In each, the snapped
  tables are exact (200/200 subtraction rows, 100/100 product cells, the gate committing on "no borrow"), all 288
  columns of the layout's rows are right, the reader has the step shape, and `AUDIT ... used_but_never_trained=0`.
- **The final file and its earlier build** ran the same training (the only differences are the reports' labels and
  code the "local" mode does not use) and follow the same schedule step for step.
- **The wiring is found within a few steps once its lesson joins.** The chain passed the division ladder 3–27 steps
  after its lesson joined (the I/O lesson then waits 20 steps). The whole machine was exact 4–22 steps after the I/O
  lesson joined: T's lesson starts after 20 exact steps, and here T settled at its first step. Without any step law,
  the joint enumeration of 57,600 combinations took about 75 steps.

#### 4.4 Time

| run | training s | seconds at step 300 / 400 / 500 / 600 | evaluation s |
|---|---|---|---|
| io3 (WIRING.md, `io3v2_official`) | 728.6 | 77 / 335 / 485 / (700: 509) | 51.8 |
| `wide_official` | 591.8 | 153 / 252 / 363 / 434 | 42.9 |
| `wide_try2_h100` (3 / 2) | 599.8 | 151 / 245 / 345 / 423 | 43.9 |
| `wide_d4m3_try1_h100` | 578.2 | 146 / 242 / 357 / 434 | 44.6 |
| `wide_enum_d3m2_h100` (whole I/O block) | 703.6 | 196 / 302 / 436 / 513 | 42.7 |
| `wide_joint_d1m2_h100` (no step laws) | 1002.6 | 171 / 295 / 727 / 811 | 47.4 |

- The wide file trains in the same time as io3 on the H100 (io3's extra time was T's lesson: there T started the wrong
  way). Phase C costs about 1.1 s a step with the step law; with the whole I/O block (146,880 combinations of 4 rows
  each) 1.3 s; with the joint block (57,600 whole divisions of 4 rows, by 3 laws) about 4 s.
- The evaluation is the frozen machine and takes the same time at every level: the prediction path makes only the
  chosen option of each switch.
- No run needed a relaxed budget; the official budgets are 3,600 s and 1,800 s.

<!-- H1_CPU_TIME -->
- **CPU seeds** (8 of 8 certified): training 1620–3117 s, frozen at steps 567–678, evaluation 403–879 s (io3's CPU seeds: 580–1,060 s and 380–559 s). On a CPU the wide file's phase C costs about 5–7 s a step, mostly the chain lesson's 9,408 combinations, which run every step until the freeze.
<!-- /H1_CPU_TIME -->

#### 4.5 Rule 8

- **Plumbing** (`local_wide.py designed`): every table and switch at the truth, both prediction paths exact with
  identical logits on 16 prompts of the T = 1, 8, 64 and OOD-N 2, 8, 64 rungs, at levels 0, 3 and 4. The same at
  levels 3 and 4 with the correct alternatives set by hand: trial multiples, the diagonals and the reversed rows, the
  add-bring, the final subtraction once or twice, the flag "A < N".
- **The gradient test at the truth** (32 rows of T = 8 and OOD-N T = 4): every parameter tensor gets a finite,
  nonzero gradient at levels 3 and 4. On T = 1 and 2 rows of small moduli the final correction's switch gets none: the
  dividend's top half is empty there, so its three options are equal.
- **Trained models** (the harness diagnostics, CPU, `local/logs/e2_*`): the runs at level 0 (io3's menu), level 3
  (seeds 74 and 1) and level 4 (seed 74, the file's levels) froze at steps 530, 588, 667 and 573. Every one of 128
  prompts of all 14 rungs is exact on both paths, with identical logits. The gradient test passes on both batches at
  level 0 and on level 3 seed 74. On level 3 seed 1 and on level 4 it passes on the T = 1, 2 batch, but on the deeper
  batch one switch, the final correction, gets exactly 0. Its leading options are aliases with equal values on every
  row (no subtraction, or subtractions that never fire: two at level 3, three at level 4; the largest holds 0.51 and
  0.38 of the weight).
  Their contributions to the gradient cancel, and the one option that differs (add the top half) carries almost no
  weight, so what is left underflows in fp32 under the 10× per-link discount. This is the gray edge of WIRING.md §5. The
  switch is on the path and gets a gradient on the other batch, but a reader can object that on some batches it gets
  none. The menu's never-firing subtractions are what cause it; a menu without them would not have this edge.
<!-- /H1_NOTES -->
<!-- /H1 -->

### 5. Which checks still encode the algorithm

<!-- CHECKS -->
Every lesson of the H1 file, and what it states. "General" means any correct machine of the menu passes it, whatever
algorithm it uses; "long division" means it describes long division's own steps.

| lesson | what it states | what it teaches | general or algorithm-specific |
|---|---|---|---|
| the table's check (D − N) + N = D on the row's own numbers; R − 0 = R, R − R = 0, 0 + N = N, R + N = N + R; x − N borrows, N − x does not, (x − N) − N does not, 0 − N borrows (COMBINED.md) | properties of subtraction, and x < N | the subtraction table | general (arithmetic) |
| the designed adder for the first 70 steps after the reading is chosen (the warm start) | addition, written by hand | the subtraction table | hand-written; unchanged from io3 (the anchors session replaces it) |
| the cell's own ladder a·(b + 1) = a·b + a | distributivity on single digits | the product cell | general |
| the routed ladder row_j(X, b + 1) = row_j(X, b) + X·10^j | distributivity for a multiplier with one digit, at place j | the column rules | general: with the number rule free it says nothing about how the products are grouped |
| distributivity X·X = Σ_j X·X_j·10^j | distributivity over the multiplier's digits | the number rule | general |
| **the division ladder**: one step from x + q·N (q = 0..9) gives back x | a step turns a register below 10·N into its remainder | the chain's switches and the gate | **long division**: it presumes the reduction is made of such steps. A Montgomery step fails it, and so would any reduction whose steps do not each leave a remainder |
| **the step law** (IO_MODE "local"): x + q·N split into its upper places and its lowest digit; bringing that digit back in and running the step gives x | a step moves the register up one place and takes the next digit at the bottom | the bring-in and the post-move | **long division**: its bring-in |
| x mod N = x, N mod N = 0, x + Q·N mod N = x (Q another row's x, Q·N by the machine's own rows) | the definition of the remainder | the reading's switches (all I/O switches in "enum", everything in "joint") | general |
| the answers of the T ≤ 2 rows | the task | the I/O switches | the task's own data |
| the reader's and writer's lessons, T's direction (IO.md, WIRING.md stage 3) | x < N, whole readings, round trips | the reader, the writer | unchanged from io3 |

So two checks still encode the algorithm, both about the division's step: the division ladder and the step law. The
H1 file needs the division ladder in every mode except "joint", and the step law in "local", its default.
- **Without the step law** ("enum"), the I/O block is taught by the general laws alone. That works in the tiny core
  at every level (§2.1) and on an H100 at full size (§4). A CPU cannot afford it at level 3 and beyond (§2.2).
- **Without the division ladder too** ("joint"), the general laws alone single out the right wirings (§3). But every
  combination of the chain and the I/O must be tried together. The tiny core learns that way at io3's menu (§2.3); the
  cost multiplies with every switch added.
<!-- /CHECKS -->

### 6. Reproduce

<!-- REPRO -->
Everything heavy runs on Modal; the local commands below each take well under a minute and 1 GB.

**The grammar and the exact counts.**

```bash
cd ~/github/one-layer/cpu_first
~/github/sotaku/venv/bin/python wider_menus/check_grammar.py      # sizes; long division x 2, aliases, Montgomery, groupings
~/github/sotaku/venv/bin/python -m modal run wider_menus/modal_exact.py --jobs "W=3 div=3 mul=3 part=chain,io,mul"
```

`modal_exact.py` (app `wider-menus-exact`, at most 3 containers) runs `exact_wide.py` and writes
`runs/exact/<job>.json`. The jobs of §3 are listed there.

**The tiny core.** `tiny_wide.py` runs one job (`python -m wider_menus.tiny_wide seed=0 div=3 mul=2 io=local`);
`suites.py` holds the suites; `modal_tiny_wide.py` runs them (app `wider-menus-tiny`, at most 12 containers):

```bash
cd ~/github/one-layer/cpu_first
~/github/sotaku/venv/bin/python -m modal run --detach wider_menus/modal_tiny_wide.py --suite grow,io,laws,w4
~/github/sotaku/venv/bin/python -m modal run wider_menus/modal_tiny_wide.py --suite grow,io,laws,w4 --fetch
~/github/sotaku/venv/bin/python -m wider_menus.summarize
```

**H1.** `modal_wide.py` is a copy of `learned_wiring/modal_wiring.py` with its own app and Volume (`wider-menus`); it
sends the file byte for byte to the unmodified evaluator, with the untouched official manifest (`--mode official`)
or with the manifest's device and seed changed (`--mode plan`). `--set` rewrites constants of the source (the
record keeps the exact source). `wait_fetch_wide.py` fetches the records to `cpu_first/runs/wider_menus/`, and
`summarize_wide.py` reads them.

```bash
cd ~/github/one-layer/one-layer-deeper
M=~/github/one-layer/cpu_first/wider_menus/modal_wide.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode official --name wide_official --file submission_wide.py
~/github/sotaku/venv/bin/python -m modal run --detach $M --mode plan --plan ~/github/one-layer/cpu_first/wider_menus/plans/wide_cpu.json --file submission_wide.py
cd ../cpu_first/wider_menus
~/github/sotaku/venv/bin/python wait_fetch_wide.py wide_official wide_cpu_seed74
~/github/sotaku/venv/bin/python summarize_wide.py --stage wide
```

Local checks of the file (`local/local_wide.py`, a copy of `learned_wiring/local/local_wiring.py` that sets
constants in the source before it loads, since the menus are made at import):

```bash
cd ~/github/one-layer/cpu_first/wider_menus/local
~/github/sotaku/venv/bin/python local_wide.py submission_wide.py designed N=16 depth_t_1 depth_t_64 depth_ood_n_t_64
~/github/sotaku/venv/bin/python local_wide.py submission_wide.py grad N=32 depth_t_8 depth_ood_n_t_4
~/github/sotaku/venv/bin/python local_wide.py submission_wide.py designed N=16 DIV_LEVEL=3 MUL_LEVEL=2
```

The training diagnostics (`local/modal_diag_wide.py`, app `wider-menus-diag`, logs to `local/logs/`, plans in
`local/plans/`):

```bash
cd ~/github/one-layer/one-layer-deeper
~/github/sotaku/venv/bin/python -m modal run ~/github/one-layer/cpu_first/wider_menus/local/modal_diag_wide.py --plan ~/github/one-layer/cpu_first/wider_menus/local/plans/e2.json
```

Files in `cpu_first/wider_menus/`:
- `submission_wide.py`: the H1 file (DIV_LEVEL 4, MUL_LEVEL 3, IO_MODE "local", IO_LAWS "xnq"), certified byte for
  byte (§4);
- `wide.py` (the grammar on exact values), `tiny_wide.py`, `suites.py`, `modal_tiny_wide.py`, `summarize.py`,
  `runs/<suite>.jsonl`: the tiny core;
- `exact_wide.py`, `modal_exact.py`, `check_grammar.py`, `runs/exact/`: the exact counts;
- `modal_wide.py`, `wait_fetch_wide.py`, `summarize_wide.py`, `plans/`: the H1 runs; `local/`: the harness, the
  diagnostics and their logs; `logs/`: the launchers' output;
- records: `cpu_first/runs/wider_menus/<name>.json` (RESULT_JSON, the full log, the exact source, the manifest).

No file outside `wider_menus/` (and `runs/wider_menus/`) was changed. The certified files of the other pieces keep
their md5s (checked at the end): `learned_wiring/submission_io3.py` `c2331362…`, `submission_div.py` `11cccf13…`,
`submission_sq.py` `781928ea…`, `learned_wiring/modal_wiring.py` `aad49070…`,
`learned_remainder/submission_combined.py` `4f5cecc7…`, `learned_remainder/modal_chain.py` `ad5fbd9c…`,
`ablation/submission_simple.py` `f374ec70…`, `anchors/submission_anchors.py` `a9e20f66…`,
`anchors/submission_combined_anchors.py` `8887fdae…`.
<!-- /REPRO -->
